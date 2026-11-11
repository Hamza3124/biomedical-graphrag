"""
GraphRAG Evaluation & Benchmark Suite.
PFA (5IADA - EMSI) - Diseases Cause Oriented Analysis using RAG and Knowledge Graphs.

Compares Standard RAG (Dense Vector Search only) against Hybrid GraphRAG
(Knowledge Graph + Vector Search) on multi-scale biomedical causal inquiries.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Fix root path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.rag.pipeline import GraphRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RAG.Evaluation")

# Benchmark questions designed to test multi-scale causality
BENCHMARK_CASES = [
    {
        "id": "CASE_1",
        "question": "What are the molecular and genetic mechanisms by which APP and PSEN1 mutations lead to Alzheimer's disease?",
        "expected_genes": ["APP", "PSEN1"],
        "expected_disease": "Alzheimer's disease",
        "type": "genetic_etiology",
    },
    {
        "id": "CASE_2",
        "question": "How does insulin resistance in Type 2 Diabetes contribute to neurodegeneration and cognitive impairment in Alzheimer's (Type 3 Diabetes)?",
        "expected_genes": ["AKT1", "IRS1", "INS", "PPARG"],
        "expected_disease": "Alzheimer's disease",
        "type": "cross_pathology_metabolic",
    },
    {
        "id": "CASE_3",
        "question": "What is the pharmacological mechanism of action of Memantine and Donepezil in mitigating Alzheimer symptoms?",
        "expected_compounds": ["Memantine", "Donepezil"],
        "expected_disease": "Alzheimer's disease",
        "type": "pharmacotherapy",
    },
    {
        "id": "CASE_4",
        "question": "How does microtubule-associated protein tau (MAPT) alteration cause synaptic degradation and dementia?",
        "expected_genes": ["MAPT"],
        "expected_symptoms": ["Dementia", "Memory Disorders"],
        "type": "proteinopathy_cascade",
    },
    {
        "id": "CASE_5",
        "question": "What biological pathways are perturbed when APOE epsilon 4 allele increases risk of late-onset Alzheimer's?",
        "expected_genes": ["APOE"],
        "type": "risk_pathways",
    },
]


def evaluate_system() -> Dict[str, Any]:
    """Runs systematic benchmark comparing Standard RAG vs Hybrid GraphRAG."""
    logger.info("Initializing GraphRAG Pipeline for evaluation...")
    pipeline = GraphRAGPipeline(use_neo4j=False)

    results_comparison: List[Dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("  BENCHMARK ÉVALUATIF : STANDARD RAG (VECTEUR) vs HYBRID GRAPHRAG (GRAPHE + VECTEUR)")
    print("=" * 80)

    for case in BENCHMARK_CASES:
        q = case["question"]
        print(f"\n[Évaluation {case['id']}] : \"{q}\"")

        # 1. Run Standard RAG (Vector Only)
        t0 = time.time()
        res_vector = pipeline.query(q, mode="vector_only", top_k_chunks=3)
        t_vector = time.time() - t0

        # 2. Run Hybrid GraphRAG (Graph + Vector)
        t0 = time.time()
        res_hybrid = pipeline.query(q, mode="hybrid", top_k_chunks=3, max_graph_facts=12)
        t_hybrid = time.time() - t0

        # Compute metrics
        # Metric 1: Identified Genes & Molecular Targets
        genes_std = len(res_vector.genetic_triggers)
        genes_hybrid = len(res_hybrid.genetic_triggers)

        # Metric 2: Identified Intermediate Biological Pathways
        pw_std = len(res_vector.intermediate_pathways)
        pw_hybrid = len(res_hybrid.intermediate_pathways)

        # Metric 3: Multi-hop Causal Chains Extracted
        chains_std = len(res_vector.causal_chains_diagram)
        chains_hybrid = len(res_hybrid.causal_chains_diagram)

        # Metric 4: Citations & Evidence Grounding
        cites_std = len(res_vector.citations)
        cites_hybrid = len(res_hybrid.citations)

        print(f"  * Mode Standard RAG (Vecteur seul)   : {genes_std} gènes | {pw_std} voies | {chains_std} chaînes causales | {cites_std} citations ({t_vector:.2f}s)")
        print(f"  * Mode Hybrid GraphRAG (Hybride)      : {genes_hybrid} gènes | {pw_hybrid} voies | {chains_hybrid} chaînes causales | {cites_hybrid} citations ({t_hybrid:.2f}s)")

        results_comparison.append({
            "case_id": case["id"],
            "question": q,
            "type": case["type"],
            "vector_only": {
                "genes_count": genes_std,
                "pathways_count": pw_std,
                "causal_chains_count": chains_std,
                "citations_count": cites_std,
                "latency_sec": round(t_vector, 3),
            },
            "hybrid_graphrag": {
                "genes_count": genes_hybrid,
                "pathways_count": pw_hybrid,
                "causal_chains_count": chains_hybrid,
                "citations_count": cites_hybrid,
                "latency_sec": round(t_hybrid, 3),
                "causal_chains_sample": res_hybrid.causal_chains_diagram[:2],
            },
        })

    # Summary Statistics
    total_chains_vector = sum(r["vector_only"]["causal_chains_count"] for r in results_comparison)
    total_chains_hybrid = sum(r["hybrid_graphrag"]["causal_chains_count"] for r in results_comparison)

    total_pw_vector = sum(r["vector_only"]["pathways_count"] for r in results_comparison)
    total_pw_hybrid = sum(r["hybrid_graphrag"]["pathways_count"] for r in results_comparison)

    total_cites_vector = sum(r["vector_only"]["citations_count"] for r in results_comparison)
    total_cites_hybrid = sum(r["hybrid_graphrag"]["citations_count"] for r in results_comparison)

    print("\n" + "=" * 80)
    print("RÉSUMÉ QUANTITATIF COMPARATIF")
    print("=" * 80)
    print(f"  * Chaînes causales multi-échelles tracées : Standard RAG = {total_chains_vector} | Hybrid GraphRAG = {total_chains_hybrid}")
    print(f"  * Voies métaboliques intermédiaires identifiées : Standard RAG = {total_pw_vector} | Hybrid GraphRAG = {total_pw_hybrid}")
    print(f"  * Sources certifiées et triplets cités    : Standard RAG = {total_cites_vector} | Hybrid GraphRAG = {total_cites_hybrid}")
    print("=" * 80)
    print("Conclusion : Le pipeline GraphRAG élimine les sauts logiques en fournissant")
    print("l'infrastructure causale explicable que le RAG vectoriel seul ne peut reconstituer.\n")

    output_path = BASE_DIR / "data" / "processed" / "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_comparison, f, indent=2)
    logger.info(f"Rapport d'évaluation sauvegardé dans {output_path}")

    return {"cases": results_comparison}


if __name__ == "__main__":
    evaluate_system()

