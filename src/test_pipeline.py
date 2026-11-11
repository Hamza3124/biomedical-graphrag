"""
End-to-End Test and Validation Script for GraphRAG Phase 2
Project: Diseases Cause Oriented Analysis using RAG and Knowledge Graphs
PFA 5IADA - EMSI

Validates all ETL outputs, graph structural integrity, and semantic vector retrieval.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Offline HF mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Test.Pipeline")


def test_etl_artifacts() -> bool:
    logger.info("=== STEP 1: Testing ETL Extraction Artifacts ===")
    subgraph_csv = Path("data/processed/subgraph_targeted.csv")
    nodes_csv = Path("data/processed/nodes_targeted.csv")
    edges_csv = Path("data/processed/edges_targeted.csv")
    summary_json = Path("data/processed/subgraph_summary.json")

    assert subgraph_csv.exists(), f"Missing {subgraph_csv}"
    assert nodes_csv.exists(), f"Missing {nodes_csv}"
    assert edges_csv.exists(), f"Missing {edges_csv}"
    assert summary_json.exists(), f"Missing {summary_json}"

    df_subgraph = pd.read_csv(subgraph_csv)
    df_nodes = pd.read_csv(nodes_csv)
    df_edges = pd.read_csv(edges_csv)

    with open(summary_json, "r") as f:
        summary = json.load(f)

    logger.info(f"Loaded Subgraph : {len(df_subgraph):,} edges")
    logger.info(f"Loaded Nodes    : {len(df_nodes):,} nodes across {df_nodes['type'].nunique()} labels")
    logger.info(f"Loaded Edges    : {len(df_edges):,} relationships across {df_edges['relation'].nunique()} types")

    # Assertions
    assert len(df_nodes) >= 1000, "Node count below expected threshold"
    assert len(df_edges) >= 5000, "Edge count below expected threshold"
    assert "Disease" in df_nodes["type"].values, "Missing Disease nodes"
    assert "Gene" in df_nodes["type"].values, "Missing Gene nodes"
    assert "Pathway" in df_nodes["type"].values, "Missing Pathway nodes"
    assert "Compound" in df_nodes["type"].values, "Missing Compound nodes"
    assert "TREATS" in df_edges["relation"].values, "Missing TREATS relation"
    assert "ASSOCIATES_WITH" in df_edges["relation"].values, "Missing ASSOCIATES_WITH relation"
    assert "PARTICIPATES_IN" in df_edges["relation"].values, "Missing PARTICIPATES_IN relation"

    logger.info("ETL Artifacts Test: PASSED [OK]\n")
    return True


def test_graph_causal_chains() -> bool:
    logger.info("=== STEP 2: Testing Graph Causal Chains ===")
    df = pd.read_csv("data/processed/subgraph_targeted.csv")

    # 1. Check Alzheimer 1-hop associations
    ad_edges = df[(df["source_name"] == "Alzheimer's disease") | (df["target_name"] == "Alzheimer's disease")]
    logger.info(f"Alzheimer's disease direct edges: {len(ad_edges):,}")
    assert len(ad_edges) > 50, "Alzheimer direct edge count too low"

    # 2. Check Causal Chain: Disease -> Gene -> Pathway
    ad_genes = set(df[(df["source_name"] == "Alzheimer's disease") & (df["relation"] == "ASSOCIATES_WITH")]["target_id"])
    pw_edges = df[(df["source_id"].isin(ad_genes)) & (df["relation"] == "PARTICIPATES_IN")]
    distinct_pw = pw_edges["target_name"].nunique()
    logger.info(f"Alzheimer associated genes ({len(ad_genes)}) participate in {distinct_pw} distinct biological pathways.")
    assert distinct_pw > 10, "Pathway fanout too low"

    # 3. Check Therapeutic Repurposing: Compound -> Treats -> Disease
    treats = df[(df["relation"] == "TREATS") & (df["target_name"] == "Alzheimer's disease")]
    drugs = treats["source_name"].unique().tolist()
    logger.info(f"Alzheimer's approved treatments in graph: {drugs}")
    assert len(drugs) >= 3, "Expected at least 3 approved Alzheimer drugs"

    logger.info("Graph Causal Chains Test: PASSED [OK]\n")
    return True


def test_vector_search() -> bool:
    logger.info("=== STEP 3: Testing Vector Search Engine ===")
    chroma_dir = Path("data/vector_store/chroma")
    if not chroma_dir.exists():
        logger.warning(f"ChromaDB not found at {chroma_dir}. Skipping vector search test.")
        return False

    from src.vector.query_vector_test import VectorQueryEngine

    engine = VectorQueryEngine(backend="chroma", chroma_dir=chroma_dir)

    query = "What genetic changes in APP, PSEN1, and APOE cause Alzheimer's disease?"
    results = engine.search(query=query, top_k=3)

    logger.info(f"Query: \"{query}\"")
    logger.info(f"Top 1 match: [Score: {results[0]['score']}] {results[0]['disease_name']} ({results[0]['question_type']})")
    logger.info(f"Question: {results[0]['question']}")

    assert len(results) > 0, "No results returned"
    assert results[0]["score"] > 0.5, f"Top match similarity score too low: {results[0]['score']}"
    assert "alzheimer" in results[0]["disease_name"].lower(), "Top match disease mismatch"

    logger.info("Vector Search Test: PASSED [OK]\n")
    return True


def main():
    print("=" * 70)
    print("GRAPHRAG BIOMEDICAL PIPELINE - PHASE 2 TEST SUITE")
    print("=" * 70)

    etl_ok = test_etl_artifacts()
    graph_ok = test_graph_causal_chains()
    vector_ok = test_vector_search()

    print("=" * 70)
    print("TEST SUITE SUMMARY")
    print(f"  1. ETL Extraction & Harmonization : {'PASSED' if etl_ok else 'FAILED'}")
    print(f"  2. Graph Causal Modeling Integrity: {'PASSED' if graph_ok else 'FAILED'}")
    print(f"  3. Semantic Vector Search Engine   : {'PASSED' if vector_ok else 'PENDING'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
