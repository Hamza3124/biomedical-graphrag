"""
Interactive CLI for Biomedical GraphRAG.
PFA (5IADA - EMSI) - Diseases Cause Oriented Analysis using RAG and Knowledge Graphs.

Allows interactive querying of the hybrid causal reasoning engine.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from src.rag.pipeline import GraphRAGPipeline

SAMPLE_QUESTIONS = [
    "How do mutations in APP, PSEN1, and APOE lead to Alzheimer's disease and memory disorders?",
    "How does insulin resistance in Type 2 Diabetes contribute to cognitive decline in Alzheimer's (Type 3 Diabetes)?",
    "What is the pharmacological mechanism of action of Donepezil and Memantine in treating dementia?",
    "How does tau protein (MAPT) hyperphosphorylation cause neurofibrillary tangles and synaptic degradation?",
    "What are the microvascular complications and neuropathies caused by chronic hyperglycemia?",
]


def print_banner():
    print("=" * 75)
    print("  BIOMEDICAL GRAPHRAG INTERACTIVE ENGINE (PFA 5IADA - EMSI)")
    print("  Causal Reasoning & Multi-Scale Pathophysiological Analysis")
    print("=" * 75)


def run_interactive_cli():
    print_banner()
    print("\nInitialisation du pipeline GraphRAG (ChromaDB + Knowledge Graph)...")
    pipeline = GraphRAGPipeline(use_neo4j=False)
    print("Pipeline prêt !\n")

    while True:
        print("\n" + "-" * 75)
        print("Menu Interactif :")
        print("  1-5 : Sélectionner une question clinique type")
        print("  C   : Saisir une question clinique personnalisée")
        print("  M   : Changer de mode (actuel: HYBRIDE [Graphe + Vecteur])")
        print("  Q   : Quitter")
        print("-" * 75)

        choice = input("Votre sélection : ").strip()

        if choice.upper() == "Q":
            print("\nFermeture du moteur GraphRAG. Au revoir !")
            break

        question = ""
        if choice in ("1", "2", "3", "4", "5"):
            idx = int(choice) - 1
            question = SAMPLE_QUESTIONS[idx]
            print(f"\n[Question sélectionnée] : \"{question}\"")
        elif choice.upper() == "C":
            question = input("\nEntrez votre question clinique : ").strip()
            if not question:
                print("Question vide, retour au menu.")
                continue
        elif choice.upper() == "M":
            print("\nModes disponibles :")
            print("  1. hybrid       : GraphRAG Hybride (Graphe relationnel + Base vectorielle)")
            print("  2. vector_only : RAG Standard (Base vectorielle MedQuAD seule)")
            print("  3. graph_only   : Graphe de Connaissances seul")
            m_choice = input("Choisissez le mode [1-3] : ").strip()
            m_map = {"1": "hybrid", "2": "vector_only", "3": "graph_only"}
            selected_mode = m_map.get(m_choice, "hybrid")
            print(f"-> Mode configuré : {selected_mode.upper()}")
            continue
        else:
            print("Option invalide.")
            continue

        # Execute query
        print("\n[Traitement GraphRAG en cours...]")
        report = pipeline.query(question, mode="hybrid", top_k_chunks=3, max_graph_facts=12)

        print("\n" + "=" * 75)
        print(f"RAPPORT CAUSAL STRUCTURÉ ({report.retrieval_mode.upper()})")
        print("=" * 75)
        print(f"• Pathologie : {report.disease_scope}")
        print(f"• Gènes identifiés : {', '.join(report.genetic_triggers) if report.genetic_triggers else 'N/A'}")
        print(f"• Voies altérées : {', '.join(report.intermediate_pathways) if report.intermediate_pathways else 'N/A'}")
        print(f"• Manifestations : {', '.join(report.clinical_manifestations)}")
        print(f"• Thérapeutiques : {', '.join(report.therapeutic_targets)}")

        if report.causal_chains_diagram:
            print("\n[Chaînes Causales Multi-Échelles du Graphe] :")
            for chain in report.causal_chains_diagram[:4]:
                print(f"  -> {chain}")

        print("\n[Citations & Traçabilité] :")
        for cite in report.citations[:4]:
            print(f"  * {cite}")

        print("\n[Synthèse Textuelle Complète] :")
        print(report.full_narrative)


def main():
    parser = argparse.ArgumentParser(description="GraphRAG Interactive Query CLI")
    parser.add_argument("--query", type=str, default=None, help="Direct query to execute non-interactively")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "vector_only", "graph_only"])
    args = parser.parse_args()

    if args.query:
        pipeline = GraphRAGPipeline(use_neo4j=False)
        report = pipeline.query(args.query, mode=args.mode)
        print(report.full_narrative)
    else:
        run_interactive_cli()


if __name__ == "__main__":
    main()

