"""
Vector Search and Cosine Similarity Test Script
Project: Diseases Cause Oriented Analysis using RAG and Knowledge Graphs
PFA 5IADA - EMSI

Performs dense vector similarity search against the indexed MedQuAD knowledge base
using PubMedBERT embeddings and returns top-k matching medical contexts with cosine scores.
"""

import os
import sys

# Configure offline mode for instant cache loading
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Vector.QueryTest")

DEFAULT_MODEL = "NeuML/pubmedbert-base-embeddings"
COLLECTION_NAME = "medquad_disease_qa"

TEST_QUERIES = [
    "What genetic mutations and genes like APP, PSEN1, and APOE cause Alzheimer's disease?",
    "Which pharmaceutical drugs and acetylcholinesterase inhibitors treat cognitive decline in dementia?",
    "How does amyloid beta peptide aggregation form plaques and damage nerve cells in the brain?",
    "What are the earliest clinical symptoms, memory impairment signs, and behavioral changes in dementia?",
    "How does insulin resistance in diabetes connect to metabolic pathways and neurodegeneration?"
]


class VectorQueryEngine:
    """Performs semantic similarity search over indexed biomedical Q&A vectors."""

    def __init__(
        self,
        backend: str = "chroma",
        model_name: str = DEFAULT_MODEL,
        chroma_dir: Optional[Path] = None,
        qdrant_url: str = "http://localhost:6333"
    ):
        self.backend = backend
        self.model_name = model_name
        self.chroma_dir = chroma_dir or (Path.cwd() / "data" / "vector_store" / "chroma")
        self.qdrant_url = qdrant_url

        logger.info(f"Loading embedding model '{self.model_name}' (offline mode)...")
        self.model = SentenceTransformer(self.model_name, local_files_only=True)
        self.client = None
        self.collection = None

        self._init_backend()

    def _init_backend(self) -> None:
        if self.backend == "chroma":
            import chromadb
            if not self.chroma_dir.exists():
                raise FileNotFoundError(f"ChromaDB store not found at {self.chroma_dir}. Run index_medquad.py first.")
            self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self.collection = self.client.get_collection(name=COLLECTION_NAME)
            logger.info(f"Connected to ChromaDB collection '{COLLECTION_NAME}' ({self.collection.count()} vectors).")

        elif self.backend == "qdrant":
            from qdrant_client import QdrantClient
            self.client = QdrantClient(url=self.qdrant_url, timeout=5.0)
            logger.info(f"Connected to Qdrant at {self.qdrant_url}.")

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_qtype: Optional[str] = None,
        filter_disease: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Encodes query and retrieves top-k most similar chunks with cosine scores."""
        query_vector = self.model.encode(query, normalize_embeddings=True).tolist()
        results = []

        if self.backend == "chroma":
            where_filter = {}
            if filter_qtype:
                where_filter["question_type"] = filter_qtype
            if filter_disease:
                where_filter["disease_name"] = filter_disease

            res = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )

            docs = res["documents"][0]
            metas = res["metadatas"][0]
            distances = res["distances"][0]

            for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
                cosine_sim = 1.0 - (dist / 2.0) if dist > 1.0 else 1.0 - dist
                results.append({
                    "rank": rank,
                    "score": round(float(cosine_sim), 4),
                    "distance": round(float(dist), 4),
                    "disease_name": meta.get("disease_name", ""),
                    "question_type": meta.get("question_type", ""),
                    "source": meta.get("source", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "cui": meta.get("cui", ""),
                    "question": meta.get("question", ""),
                    "text": doc
                })

        elif self.backend == "qdrant":
            from qdrant_client.http import models as qmodels
            conditions = []
            if filter_qtype:
                conditions.append(qmodels.FieldCondition(key="question_type", match=qmodels.MatchValue(value=filter_qtype)))
            if filter_disease:
                conditions.append(qmodels.FieldCondition(key="disease_name", match=qmodels.MatchValue(value=filter_disease)))
            query_filter = qmodels.Filter(must=conditions) if conditions else None

            q_res = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter
            )

            for rank, hit in enumerate(q_res, start=1):
                payload = hit.payload or {}
                results.append({
                    "rank": rank,
                    "score": round(float(hit.score), 4),
                    "disease_name": payload.get("disease_name", ""),
                    "question_type": payload.get("question_type", ""),
                    "source": payload.get("source", ""),
                    "doc_id": payload.get("doc_id", ""),
                    "cui": payload.get("cui", ""),
                    "question": payload.get("question", ""),
                    "text": payload.get("text", "")
                })

        return results


def print_search_results(query: str, results: List[Dict[str, Any]]) -> None:
    """Pretty prints query results with scores and highlighted causal metadata."""
    print("\n" + "=" * 80)
    print(f"QUERY: \"{query}\"")
    print("=" * 80)

    if not results:
        print("No matching records found.")
        return

    for r in results:
        print(f"\n[Rank #{r['rank']}] Cosine Similarity: {r['score']:.4f} | Disease: {r['disease_name']}")
        print(f"Topic: {r['question_type']} | Source: {r['source']} | Doc ID: {r['doc_id']} | CUI: {r['cui']}")
        print(f"Question: {r['question']}")
        print("-" * 80)
        snippet = r["text"].replace("\n", " ")
        if len(snippet) > 350:
            snippet = snippet[:350] + "..."
        print(f"Snippet: {snippet}")
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Query MedQuAD vector store with cosine similarity.")
    parser.add_argument("--query", type=str, default=None, help="Natural language query string")
    parser.add_argument("--top-k", type=int, default=3, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--backend", choices=["chroma", "qdrant"], default="chroma", help="Vector store backend")
    parser.add_argument("--qtype", type=str, default=None, help="Filter by question_type metadata")
    parser.add_argument("--disease", type=str, default=None, help="Filter by disease_name metadata")
    parser.add_argument("--run-all-tests", action="store_true", help="Execute the complete suite of 5 test queries")

    args = parser.parse_args()

    engine = VectorQueryEngine(backend=args.backend)

    if args.query:
        queries = [args.query]
    elif args.run_all_tests or args.query is None:
        queries = TEST_QUERIES
    else:
        queries = [TEST_QUERIES[0]]

    for q in queries:
        results = engine.search(
            query=q,
            top_k=args.top_k,
            filter_qtype=args.qtype,
            filter_disease=args.disease
        )
        print_search_results(q, results)


if __name__ == "__main__":
    main()
