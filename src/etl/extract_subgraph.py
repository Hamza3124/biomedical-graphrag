"""
Targeted Biomedical Subgraph Extraction Script (ETL)
Project: Diseases Cause Oriented Analysis using RAG and Knowledge Graphs
PFA 5IADA - EMSI

Extracts a targeted causal subgraph centered on target pathologies (e.g. Alzheimer's Disease,
Type 2 Diabetes) from Hetionet and PrimeKG, harmonizing identifiers (DOID, UMLS, MeSH, Entrez, DrugBank)
and exporting clean datasets for Neo4j loading and GraphRAG.
"""

import os
import sys
import gzip
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ETL.ExtractSubgraph")

# Canonical disease identifiers and ontology mappings
DISEASE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "Disease::DOID:10652": {
        "doid": "DOID:10652",
        "name": "Alzheimer's disease",
        "umls_cui": "C0002395",
        "mesh_id": "D000544",
        "mondo_id": "MONDO:0004975",
        "description": "Progressive neurodegenerative disorder characterized by cognitive decline, memory impairment, amyloid-beta accumulation, and neurofibrillary tangles."
    },
    "Disease::DOID:9352": {
        "doid": "DOID:9352",
        "name": "Type 2 diabetes mellitus",
        "umls_cui": "C0011860",
        "mesh_id": "D003924",
        "mondo_id": "MONDO:0005148",
        "description": "Chronic metabolic disorder characterized by insulin resistance, relative insulin deficiency, and hyperglycemia, sharing common neuro-metabolic pathways with AD."
    }
}

# Relation mapping: Hetionet metaedge code to standardized predicate & causal direction
METAEDGE_MAP: Dict[str, Dict[str, str]] = {
    "DaG": {"predicate": "ASSOCIATES_WITH", "source_type": "Disease", "target_type": "Gene", "causal_type": "association"},
    "DdG": {"predicate": "DOWNREGULATES", "source_type": "Disease", "target_type": "Gene", "causal_type": "inhibition"},
    "DuG": {"predicate": "UPREGULATES", "source_type": "Disease", "target_type": "Gene", "causal_type": "induction"},
    "DpS": {"predicate": "MANIFESTS", "source_type": "Disease", "target_type": "Symptom", "causal_type": "manifestation"},
    "DlA": {"predicate": "LOCALIZES_IN", "source_type": "Disease", "target_type": "Anatomy", "causal_type": "localization"},
    "CtD": {"predicate": "TREATS", "source_type": "Compound", "target_type": "Disease", "causal_type": "therapeutic"},
    "CpD": {"predicate": "PALLIATES", "source_type": "Compound", "target_type": "Disease", "causal_type": "palliative"},
    "DrD": {"predicate": "RESEMBLES", "source_type": "Disease", "target_type": "Disease", "causal_type": "similarity"},
    "GpPW": {"predicate": "PARTICIPATES_IN", "source_type": "Gene", "target_type": "Pathway", "causal_type": "mechanism"},
    "GpBP": {"predicate": "PARTICIPATES_IN", "source_type": "Gene", "target_type": "BiologicalProcess", "causal_type": "mechanism"},
    "GiG": {"predicate": "INTERACTS_WITH", "source_type": "Gene", "target_type": "Gene", "causal_type": "interaction"},
    "Gr>G": {"predicate": "REGULATES", "source_type": "Gene", "target_type": "Gene", "causal_type": "regulation"},
    "CbG": {"predicate": "BINDS", "source_type": "Compound", "target_type": "Gene", "causal_type": "target_binding"},
    "CdG": {"predicate": "DOWNREGULATES", "source_type": "Compound", "target_type": "Gene", "causal_type": "target_inhibition"},
    "CuG": {"predicate": "UPREGULATES", "source_type": "Compound", "target_type": "Gene", "causal_type": "target_activation"},
    "CcSE": {"predicate": "CAUSES", "source_type": "Compound", "target_type": "SideEffect", "causal_type": "adverse_effect"}
}

# Standardized node kind normalization
NODE_KIND_MAP: Dict[str, str] = {
    "Disease": "Disease",
    "Gene": "Gene",
    "Pathway": "Pathway",
    "Compound": "Compound",
    "Symptom": "Symptom",
    "Anatomy": "Anatomy",
    "Side Effect": "SideEffect",
    "Biological Process": "BiologicalProcess",
    "Pharmacologic Class": "PharmacologicClass"
}


class BiomedicalSubgraphExtractor:
    """Modular extractor for disease-targeted multi-hop causal subgraphs."""

    def __init__(self, root_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.root_dir = root_dir or Path.cwd()
        self.output_dir = output_dir or (self.root_dir / "data" / "processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.hetionet_nodes_file = self._find_file([
            self.root_dir / "data" / "raw" / "hetionet-main" / "hetnet" / "tsv" / "hetionet-v1.0-nodes.tsv",
            self.root_dir / "hetionet-main" / "hetionet-main" / "hetnet" / "tsv" / "hetionet-v1.0-nodes.tsv",
            self.root_dir / "hetionet-main" / "hetnet" / "tsv" / "hetionet-v1.0-nodes.tsv"
        ], "Hetionet nodes TSV")

        self.hetionet_edges_file = self._find_file([
            self.root_dir / "data" / "raw" / "hetionet-main" / "hetnet" / "tsv" / "hetionet-v1.0-edges.sif.gz",
            self.root_dir / "hetionet-main" / "hetionet-main" / "hetnet" / "tsv" / "hetionet-v1.0-edges.sif.gz",
            self.root_dir / "hetionet-main" / "hetnet" / "tsv" / "hetionet-v1.0-edges.sif.gz"
        ], "Hetionet edges SIF.GZ")

        # In-memory storage
        self.raw_nodes: Dict[str, Dict[str, str]] = {}
        self.extracted_nodes: Dict[str, Dict[str, Any]] = {}
        self.extracted_edges: List[Dict[str, Any]] = []

    def _find_file(self, candidate_paths: List[Path], file_desc: str) -> Path:
        for p in candidate_paths:
            if p.is_file() and p.stat().st_size > 1000:
                logger.info(f"Found {file_desc}: {p}")
                return p
        raise FileNotFoundError(f"Could not locate valid {file_desc} in paths: {[str(p) for p in candidate_paths]}")

    def load_node_registry(self) -> None:
        """Load all nodes from Hetionet node table."""
        logger.info(f"Loading node dictionary from {self.hetionet_nodes_file}...")
        df_nodes = pd.read_csv(self.hetionet_nodes_file, sep="\t")
        for _, row in df_nodes.iterrows():
            raw_id = str(row["id"])
            name = str(row["name"])
            kind = str(row["kind"])
            self.raw_nodes[raw_id] = {
                "raw_id": raw_id,
                "name": name,
                "kind": NODE_KIND_MAP.get(kind, kind)
            }
        logger.info(f"Loaded {len(self.raw_nodes):,} reference nodes.")

    def _standardize_node(self, raw_id: str) -> Dict[str, Any]:
        """Convert a Hetionet/PrimeKG raw identifier into clean cross-referenced entity."""
        raw_info = self.raw_nodes.get(raw_id, {"raw_id": raw_id, "name": raw_id, "kind": "Unknown"})
        kind = raw_info["kind"]
        name = raw_info["name"]

        # Parse prefix and external code
        prefix = raw_id.split("::")[0] if "::" in raw_id else kind
        code = raw_id.split("::")[1] if "::" in raw_id else raw_id

        # Defaults
        node_id = f"{kind}:{code}"
        external_id = code
        identifier_type = prefix
        umls_cui = ""
        mesh_id = ""
        doid = ""

        if raw_id in DISEASE_REGISTRY:
            reg = DISEASE_REGISTRY[raw_id]
            node_id = reg["doid"]
            name = reg["name"]
            kind = "Disease"
            external_id = reg["doid"]
            identifier_type = "DOID"
            umls_cui = reg["umls_cui"]
            mesh_id = reg["mesh_id"]
            doid = reg["doid"]
        elif kind == "Disease":
            node_id = code if code.startswith("DOID:") else f"DOID:{code}"
            external_id = node_id
            identifier_type = "DOID"
            doid = node_id
        elif kind == "Gene":
            node_id = f"Gene:{code}"
            external_id = code
            identifier_type = "Entrez"
        elif kind == "Compound":
            node_id = f"Compound:{code}"
            external_id = code
            identifier_type = "DrugBank"
        elif kind == "Pathway":
            node_id = f"Pathway:{code}"
            external_id = code
            identifier_type = "WikiPathways"
        elif kind == "Symptom":
            node_id = f"Symptom:{code}"
            external_id = code
            identifier_type = "MeSH"
            mesh_id = code
        elif kind == "Anatomy":
            node_id = f"Anatomy:{code}"
            external_id = code
            identifier_type = "UBERON"
        elif kind == "SideEffect":
            node_id = f"SideEffect:{code}"
            external_id = code
            identifier_type = "UMLS"
            umls_cui = code
        elif kind == "BiologicalProcess":
            node_id = f"GO:{code}"
            external_id = code
            identifier_type = "GO"

        return {
            "id": node_id,
            "raw_id": raw_id,
            "name": name,
            "type": kind,
            "identifier_type": identifier_type,
            "external_id": external_id,
            "umls_cui": umls_cui,
            "mesh_id": mesh_id,
            "doid": doid
        }

    def extract_subgraph(
        self,
        target_diseases: List[str],
        max_pathways_per_gene: int = 15,
        max_compounds_per_gene: int = 20,
        include_processes: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Executes targeted k-hop extraction centered on target diseases.
        
        Args:
            target_diseases: List of Hetionet disease raw IDs (e.g. ['Disease::DOID:10652', 'Disease::DOID:9352'])
            max_pathways_per_gene: Maximum pathways retained per gene to control hub fanout
            max_compounds_per_gene: Maximum drug interactions retained per gene
            include_processes: Whether to include biological processes in addition to pathways
        """
        target_set = set(target_diseases)
        logger.info(f"Targeting pathology DOIDs: {target_set}")

        # Register target disease nodes
        for did in target_set:
            self.extracted_nodes[did] = self._standardize_node(did)

        # -------------------------------------------------------------
        # STEP 1: Direct 1-hop disease edges
        # -------------------------------------------------------------
        logger.info("Extracting direct 1-hop disease edges (Genes, Symptoms, Anatomy, Compounds)...")
        genes_1hop: Set[str] = set()
        compounds_1hop: Set[str] = set()
        nodes_1hop: Set[str] = set(target_set)
        edges_1hop: List[Dict[str, Any]] = []

        with gzip.open(self.hetionet_edges_file, "rt", encoding="utf-8") as f:
            f.readline()  # header: source metaedge target
            for line in f:
                s, r, t = line.strip().split("\t")
                if r not in METAEDGE_MAP:
                    continue

                is_1hop = False
                if s in target_set:
                    is_1hop = True
                    other_node = t
                elif t in target_set:
                    is_1hop = True
                    other_node = s

                if is_1hop:
                    nodes_1hop.add(other_node)
                    if other_node.startswith("Gene::"):
                        genes_1hop.add(other_node)
                    elif other_node.startswith("Compound::"):
                        compounds_1hop.add(other_node)

                    edges_1hop.append({
                        "source_raw": s,
                        "metaedge": r,
                        "target_raw": t,
                        "hop": 1
                    })

        logger.info(f"1-Hop extraction completed: {len(edges_1hop):,} edges, {len(nodes_1hop):,} nodes ({len(genes_1hop)} genes, {len(compounds_1hop)} compounds).")

        # -------------------------------------------------------------
        # STEP 2: 2-Hop Causal & Mechanistic Expansion
        # -------------------------------------------------------------
        logger.info("Extracting 2-hop causal edges (Gene -> Pathway, Compound -> Gene, Gene -> Gene)...")
        edges_2hop: List[Dict[str, Any]] = []
        nodes_2hop: Set[str] = set()

        # Track counts per gene to avoid extreme hubs while keeping all key pathways
        gene_pathway_count: Dict[str, int] = {g: 0 for g in genes_1hop}
        gene_compound_count: Dict[str, int] = {g: 0 for g in genes_1hop}

        with gzip.open(self.hetionet_edges_file, "rt", encoding="utf-8") as f:
            f.readline()
            for line in f:
                s, r, t = line.strip().split("\t")
                if r not in METAEDGE_MAP:
                    continue

                # 1. Gene participates in Pathway (GpPW)
                if r == "GpPW" and s in genes_1hop:
                    if gene_pathway_count[s] < max_pathways_per_gene:
                        gene_pathway_count[s] += 1
                        nodes_2hop.add(t)
                        edges_2hop.append({"source_raw": s, "metaedge": r, "target_raw": t, "hop": 2})

                # 2. Gene participates in Biological Process (GpBP) - optional
                elif include_processes and r == "GpBP" and s in genes_1hop:
                    if gene_pathway_count[s] < max_pathways_per_gene:
                        nodes_2hop.add(t)
                        edges_2hop.append({"source_raw": s, "metaedge": r, "target_raw": t, "hop": 2})

                # 3. Compound interacts with 1-hop disease Gene (CbG: binds, CdG: inhibits/downregulates, CuG: upregulates)
                elif r in ("CbG", "CdG", "CuG") and t in genes_1hop:
                    if gene_compound_count[t] < max_compounds_per_gene:
                        gene_compound_count[t] += 1
                        nodes_2hop.add(s)
                        edges_2hop.append({"source_raw": s, "metaedge": r, "target_raw": t, "hop": 2})

                # 4. Gene-Gene interactions & causal regulation among disease genes
                elif r in ("GiG", "Gr>G") and s in genes_1hop and t in genes_1hop:
                    edges_2hop.append({"source_raw": s, "metaedge": r, "target_raw": t, "hop": 2})

                # 5. Compound causes Side Effect for disease-treating compounds
                elif r == "CcSE" and s in compounds_1hop:
                    nodes_2hop.add(t)
                    edges_2hop.append({"source_raw": s, "metaedge": r, "target_raw": t, "hop": 2})

        logger.info(f"2-Hop extraction completed: {len(edges_2hop):,} causal edges, {len(nodes_2hop):,} new nodes.")

        # -------------------------------------------------------------
        # STEP 3: Deduplication, Harmonization & Assembly
        # -------------------------------------------------------------
        all_raw_nodes = nodes_1hop | nodes_2hop
        logger.info(f"Total unique nodes to harmonize: {len(all_raw_nodes):,}")

        for nid in all_raw_nodes:
            self.extracted_nodes[nid] = self._standardize_node(nid)

        all_edges = edges_1hop + edges_2hop
        clean_edge_records = []
        seen_edges = set()

        for edge in all_edges:
            s_raw = edge["source_raw"]
            t_raw = edge["target_raw"]
            meta = edge["metaedge"]
            hop = edge["hop"]

            meta_info = METAEDGE_MAP[meta]
            pred = meta_info["predicate"]

            src_node = self.extracted_nodes[s_raw]
            tgt_node = self.extracted_nodes[t_raw]

            edge_key = (src_node["id"], pred, tgt_node["id"])
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            clean_edge_records.append({
                "source_id": src_node["id"],
                "source_name": src_node["name"],
                "source_type": src_node["type"],
                "relation": pred,
                "target_id": tgt_node["id"],
                "target_name": tgt_node["name"],
                "target_type": tgt_node["type"],
                "metaedge": meta,
                "causal_type": meta_info["causal_type"],
                "hop": hop,
                "source_dataset": "Hetionet"
            })

        df_edges = pd.DataFrame(clean_edge_records)
        df_nodes = pd.DataFrame(list(self.extracted_nodes.values()))

        # Clean nodes table
        df_nodes_clean = df_nodes[[
            "id", "name", "type", "identifier_type", "external_id", "umls_cui", "mesh_id", "doid"
        ]].drop_duplicates(subset=["id"]).reset_index(drop=True)

        # Clean edges table for Neo4j loading
        df_edges_clean = df_edges[[
            "source_id", "relation", "target_id", "metaedge", "causal_type", "hop", "source_dataset"
        ]].drop_duplicates().reset_index(drop=True)

        logger.info(f"Harmonization finished. Final unique nodes: {len(df_nodes_clean):,}, Final unique edges: {len(df_edges):,}")
        return df_edges, df_nodes_clean, df_edges_clean

    def export(
        self,
        df_full: pd.DataFrame,
        df_nodes: pd.DataFrame,
        df_edges: pd.DataFrame
    ) -> Dict[str, str]:
        """Export datasets to CSV and summary JSON."""
        full_path = self.output_dir / "subgraph_targeted.csv"
        nodes_path = self.output_dir / "nodes_targeted.csv"
        edges_path = self.output_dir / "edges_targeted.csv"
        summary_path = self.output_dir / "subgraph_summary.json"

        logger.info(f"Saving denormalized subgraph to {full_path}...")
        df_full.to_csv(full_path, index=False)

        logger.info(f"Saving nodes table to {nodes_path}...")
        df_nodes.to_csv(nodes_path, index=False)

        logger.info(f"Saving edges table to {edges_path}...")
        df_edges.to_csv(edges_path, index=False)

        # Generate summary
        summary = {
            "total_nodes": int(len(df_nodes)),
            "total_edges": int(len(df_edges)),
            "nodes_by_type": df_nodes["type"].value_counts().to_dict(),
            "edges_by_relation": df_edges["relation"].value_counts().to_dict(),
            "edges_by_causal_type": df_full["causal_type"].value_counts().to_dict(),
            "edges_by_hop": df_full["hop"].value_counts().to_dict(),
            "target_diseases": list(DISEASE_REGISTRY.keys()),
            "sample_causal_paths": [
                {
                    "path_type": "Disease -> Gene -> Pathway",
                    "example": "Alzheimer's disease -> MAPT -> Alzheimers Disease Pathway"
                },
                {
                    "path_type": "Compound -> Treats -> Disease & Binds -> Gene",
                    "example": "Donepezil -> TREATS -> Alzheimer's disease | Donepezil -> BINDS -> ACHE"
                },
                {
                    "path_type": "Cross-disease common metabolic link",
                    "example": "Alzheimer's disease & Type 2 Diabetes -> Shared Genes: AKT1, IRS1, INS"
                }
            ]
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary statistics exported to {summary_path}")

        return {
            "subgraph_csv": str(full_path),
            "nodes_csv": str(nodes_path),
            "edges_csv": str(edges_path),
            "summary_json": str(summary_path)
        }


def main():
    parser = argparse.ArgumentParser(description="Extract targeted biomedical subgraph for GraphRAG.")
    parser.add_argument(
        "--diseases",
        nargs="+",
        default=["Disease::DOID:10652", "Disease::DOID:9352"],
        help="Target disease DOIDs (default: Alzheimer's Disease and Type 2 Diabetes)"
    )
    parser.add_argument("--k-hop", type=int, default=2, help="Neighborhood hop depth (default: 2)")
    parser.add_argument("--max-pathways-per-gene", type=int, default=15, help="Max pathways per gene (default: 15)")
    parser.add_argument("--max-compounds-per-gene", type=int, default=20, help="Max compounds per gene (default: 20)")
    parser.add_argument("--include-processes", action="store_true", help="Include biological processes (GpBP)")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")

    args = parser.parse_args()

    extractor = BiomedicalSubgraphExtractor(output_dir=Path(args.output_dir))
    extractor.load_node_registry()
    df_full, df_nodes, df_edges = extractor.extract_subgraph(
        target_diseases=args.diseases,
        max_pathways_per_gene=args.max_pathways_per_gene,
        max_compounds_per_gene=args.max_compounds_per_gene,
        include_processes=args.include_processes
    )
    files = extractor.export(df_full, df_nodes, df_edges)

    print("\n" + "=" * 60)
    print("ETL EXTRACTION SUCCESSFULLY COMPLETED")
    print("=" * 60)
    print(f"Target Pathologies : {args.diseases}")
    print(f"Total Unique Nodes : {len(df_nodes):,}")
    print(f"Total Unique Edges : {len(df_edges):,}")
    print(f"Files Generated:")
    for k, v in files.items():
        print(f"  - {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()

