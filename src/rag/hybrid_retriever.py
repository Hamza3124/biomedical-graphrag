"""
Hybrid Retriever for Biomedical GraphRAG.
PFA (5IADA - EMSI) - Diseases Cause Oriented Analysis using RAG and Knowledge Graphs.

Combines:
1. Dense Semantic Vector Search (ChromaDB + PubMedBERT) over MedQuAD clinical QA pairs.
2. Structured Causal Graph Traversal (Neo4j with automatic local in-memory fallback)
   over multi-hop biomedical interaction paths.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer

from src.rag.entity_linker import BiomedicalEntityLinker

# Offline flags for local embeddings
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

logger = logging.getLogger("RAG.HybridRetriever")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_MODEL = "NeuML/pubmedbert-base-embeddings"
COLLECTION_NAME = "medquad_disease_qa"


@dataclass
class CausalFact:
    """Represents a structured causal relationship from the knowledge graph."""
    source_name: str
    source_type: str
    source_id: str
    relation: str
    target_name: str
    target_type: str
    target_id: str
    causal_type: str
    hop: int
    metaedge: str

    def to_readable_triple(self) -> str:
        return f"({self.source_type}: {self.source_name}) --[{self.relation} ({self.causal_type})]--> ({self.target_type}: {self.target_name})"


@dataclass
class TextEvidence:
    """Represents an unstructured clinical question-answer chunk from MedQuAD."""
    chunk_id: str
    disease_name: str
    question_type: str
    source: str
    doc_id: str
    question: str
    text: str
    score: float


@dataclass
class RetrievalResult:
    """Container for the fused hybrid retrieval output."""
    query: str
    linked_entities: List[Dict[str, Any]]
    text_chunks: List[TextEvidence]
    causal_facts: List[CausalFact]
    multihop_chains: List[str]
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)


class LocalGraphEngine:
    """
    High-performance in-memory graph traversal engine built on preprocessed
    subgraphs. Provides zero-latency multi-hop causal traversal without
    requiring an active Neo4j daemon.
    """

    def __init__(self, nodes_csv: Path, edges_csv: Path):
        self.nodes_csv = nodes_csv
        self.edges_csv = edges_csv
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.adj_out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.adj_in: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._load_graph()

    def _load_graph(self) -> None:
        logger.info(f"Loading local graph topology from {self.edges_csv.name}...")
        df_nodes = pd.read_csv(self.nodes_csv)
        for _, r in df_nodes.iterrows():
            self.nodes[str(r["id"])] = {
                "id": str(r["id"]),
                "name": str(r["name"]),
                "type": str(r["type"]),
                "doid": str(r.get("doid", "")) if pd.notna(r.get("doid")) else None,
                "umls_cui": str(r.get("umls_cui", "")) if pd.notna(r.get("umls_cui")) else None,
            }

        df_edges = pd.read_csv(self.edges_csv)
        for _, r in df_edges.iterrows():
            edge = {
                "source_id": str(r["source_id"]),
                "relation": str(r["relation"]),
                "target_id": str(r["target_id"]),
                "metaedge": str(r.get("metaedge", "")),
                "causal_type": str(r.get("causal_type", "associative")),
                "hop": int(r.get("hop", 1)),
            }
            self.adj_out[edge["source_id"]].append(edge)
            self.adj_in[edge["target_id"]].append(edge)

        logger.info(
            f"Local Graph Engine initialized: {len(self.nodes):,} nodes, {len(df_edges):,} directed edges."
        )

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        return self.nodes.get(node_id)

    def get_1hop_causal_facts(self, node_id: str, limit: int = 25) -> List[CausalFact]:
        """Extracts direct 1-hop relationships (both outgoing and incoming)."""
        facts: List[CausalFact] = []
        src_node = self.nodes.get(node_id)
        if not src_node:
            return facts

        # Outgoing edges
        for e in self.adj_out.get(node_id, [])[:limit]:
            tgt_node = self.nodes.get(e["target_id"])
            if tgt_node:
                facts.append(
                    CausalFact(
                        source_name=src_node["name"],
                        source_type=src_node["type"],
                        source_id=src_node["id"],
                        relation=e["relation"],
                        target_name=tgt_node["name"],
                        target_type=tgt_node["type"],
                        target_id=tgt_node["id"],
                        causal_type=e["causal_type"],
                        hop=1,
                        metaedge=e["metaedge"],
                    )
                )

        # Incoming edges
        for e in self.adj_in.get(node_id, [])[:limit]:
            s_node = self.nodes.get(e["source_id"])
            if s_node:
                facts.append(
                    CausalFact(
                        source_name=s_node["name"],
                        source_type=s_node["type"],
                        source_id=s_node["id"],
                        relation=e["relation"],
                        target_name=src_node["name"],
                        target_type=src_node["type"],
                        target_id=src_node["id"],
                        causal_type=e["causal_type"],
                        hop=1,
                        metaedge=e["metaedge"],
                    )
                )
        return facts

    def extract_causal_chains(self, disease_id: str, target_gene_ids: Optional[List[str]] = None, max_chains: int = 10) -> List[str]:
        """
        Extracts multi-hop causal chains:
        (Disease) -> (Associated Gene) -> (Pathway)
        (Disease) -> (Associated Gene) -> (PPI Gene) -> (Symptom / Pathway)
        """
        chains: List[str] = []
        d_node = self.nodes.get(disease_id)
        if not d_node:
            return chains

        # 1. Find genes associated with disease
        assoc_edges = [
            e for e in self.adj_out.get(disease_id, [])
            if e["relation"] in ("ASSOCIATES_WITH", "ASSOCIATES", "ASSOCIATES_DaG", "DOWNREGULATES", "UPREGULATES")
        ]

        if target_gene_ids:
            target_set = set(target_gene_ids)
            assoc_edges = [e for e in assoc_edges if e["target_id"] in target_set] or assoc_edges

        for ae in assoc_edges[:10]:
            gene_node = self.nodes.get(ae["target_id"])
            if not gene_node or gene_node["type"] != "Gene":
                continue

            # Check pathways of this gene
            gene_out = self.adj_out.get(gene_node["id"], [])
            pathways = [
                self.nodes[e["target_id"]]["name"]
                for e in gene_out
                if e["relation"] in ("PARTICIPATES_IN", "PARTICIPATES", "PARTICIPATES_GpPW")
                and e["target_id"] in self.nodes
            ]

            for pw in pathways[:2]:
                chain = f"({d_node['name']}) ==[{ae['relation']}]==> (Gene: {gene_node['name']}) ==[PARTICIPATES_IN]==> (Pathway: {pw})"
                chains.append(chain)
                if len(chains) >= max_chains:
                    return chains

        return chains

    def find_cross_disease_shared_genes(self, disease_id_1: str = "DOID:10652", disease_id_2: str = "DOID:9352") -> List[Dict[str, Any]]:
        """Finds genes shared between two pathologies (e.g. Alzheimer's and Type 2 Diabetes)."""
        genes_1 = {
            e["target_id"]: e for e in self.adj_out.get(disease_id_1, [])
            if self.nodes.get(e["target_id"], {}).get("type") == "Gene"
        }
        genes_2 = {
            e["target_id"]: e for e in self.adj_out.get(disease_id_2, [])
            if self.nodes.get(e["target_id"], {}).get("type") == "Gene"
        }

        shared_ids = set(genes_1.keys()).intersection(genes_2.keys())
        results = []
        for gid in shared_ids:
            g_node = self.nodes.get(gid)
            if g_node:
                results.append({
                    "gene_id": gid,
                    "gene_name": g_node["name"],
                    "rel_disease_1": genes_1[gid]["relation"],
                    "rel_disease_2": genes_2[gid]["relation"],
                })
        return results


class HybridRetriever:
    """
    Unified Hybrid Retriever querying both ChromaDB semantic vectors and
    structured causal graph paths.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        model_name: str = DEFAULT_MODEL,
        use_neo4j: bool = False,
    ):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent.parent
        self.model_name = model_name
        self.use_neo4j = use_neo4j

        # 1. Entity Linker
        nodes_path = self.base_dir / "data" / "processed" / "nodes_targeted.csv"
        edges_path = self.base_dir / "data" / "processed" / "edges_targeted.csv"
        self.linker = BiomedicalEntityLinker(nodes_csv_path=nodes_path)

        # 2. Local Graph Engine
        self.graph_engine = LocalGraphEngine(nodes_csv=nodes_path, edges_csv=edges_path)

        # 3. Vector Store (ChromaDB)
        self.chroma_dir = self.base_dir / "data" / "vector_store" / "chroma"
        self._init_vector_store()

    def _init_vector_store(self) -> None:
        import chromadb

        if not self.chroma_dir.exists():
            raise FileNotFoundError(f"ChromaDB store not found at {self.chroma_dir}")

        logger.info(f"Loading PubMedBERT model '{self.model_name}' (offline mode)...")
        self.model = SentenceTransformer(self.model_name, local_files_only=True)
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        logger.info(f"Connected to ChromaDB collection '{COLLECTION_NAME}' ({self.collection.count()} chunks).")

    def retrieve(
        self,
        query: str,
        top_k_chunks: int = 4,
        max_graph_facts: int = 15,
        max_chains: int = 6,
    ) -> RetrievalResult:
        """
        Executes unified hybrid retrieval:
        1. Identifies biomedical entities in the query.
        2. Retrieves dense semantic vector matches from MedQuAD.
        3. Traverses multi-hop causal paths and interactions in the Knowledge Graph.
        """
        # Step 1: Link biomedical entities
        linked_entities = self.linker.link_entities(query)

        # Default to Alzheimer's if no disease explicitly stated
        has_disease = any(e["type"] == "Disease" for e in linked_entities)
        if not has_disease:
            ad_node = self.linker.get_node_by_id("DOID:10652")
            if ad_node:
                linked_entities.insert(0, {**ad_node, "matched_text": "default_context", "confidence": 0.8})

        # Step 2: Vector Search
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        vector_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k_chunks,
            include=["documents", "metadatas", "distances"],
        )

        text_evidences: List[TextEvidence] = []
        if vector_results and "ids" in vector_results and vector_results["ids"]:
            ids = vector_results["ids"][0]
            docs = vector_results["documents"][0]
            metas = vector_results["metadatas"][0]
            dists = vector_results["distances"][0]

            for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                # Chroma cosine distance = 1 - cosine_similarity
                score = round(1.0 - dist, 4)
                text_evidences.append(
                    TextEvidence(
                        chunk_id=cid,
                        disease_name=str(meta.get("disease_name", "")),
                        question_type=str(meta.get("question_type", "")),
                        source=str(meta.get("source", "")),
                        doc_id=str(meta.get("doc_id", "")),
                        question=str(meta.get("question", "")),
                        text=doc,
                        score=score,
                    )
                )

        # Step 3: Graph Causal Traversal
        causal_facts: List[CausalFact] = []
        seen_fact_signatures: Set[Tuple[str, str, str]] = set()

        for ent in linked_entities:
            facts = self.graph_engine.get_1hop_causal_facts(ent["id"], limit=8)
            for f in facts:
                sig = (f.source_name, f.relation, f.target_name)
                if sig not in seen_fact_signatures:
                    seen_fact_signatures.add(sig)
                    causal_facts.append(f)
                    if len(causal_facts) >= max_graph_facts:
                        break
            if len(causal_facts) >= max_graph_facts:
                break

        # Step 4: Extract Multi-hop Causal Chains
        multihop_chains: List[str] = []
        disease_entities = [e for e in linked_entities if e["type"] == "Disease"]
        gene_entities = [e["id"] for e in linked_entities if e["type"] == "Gene"]

        for d_ent in disease_entities:
            chains = self.graph_engine.extract_causal_chains(
                disease_id=d_ent["id"],
                target_gene_ids=gene_entities if gene_entities else None,
                max_chains=max_chains,
            )
            multihop_chains.extend(chains)

        # Cross-disease metabolic link check
        is_cross_disease = (
            any("diabetes" in e["name"].lower() for e in linked_entities)
            and any("alzheimer" in e["name"].lower() for e in linked_entities)
        ) or ("diabetes" in query.lower() and "alzheimer" in query.lower())

        if is_cross_disease:
            shared = self.graph_engine.find_cross_disease_shared_genes()
            for s in shared[:5]:
                multihop_chains.append(
                    f"CROSS-PATHOLOGY AXIS: (Alzheimer's) --[{s['rel_disease_1']}]--> (Shared Gene: {s['gene_name']}) <--[{s['rel_disease_2']}]-- (Type 2 Diabetes)"
                )

        stats = {
            "entities_found": len(linked_entities),
            "text_chunks_retrieved": len(text_evidences),
            "graph_facts_retrieved": len(causal_facts),
            "multihop_chains_extracted": len(multihop_chains),
        }

        return RetrievalResult(
            query=query,
            linked_entities=linked_entities,
            text_chunks=text_evidences,
            causal_facts=causal_facts,
            multihop_chains=multihop_chains,
            retrieval_stats=stats,
        )

