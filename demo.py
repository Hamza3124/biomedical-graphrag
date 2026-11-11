"""
Script de Démonstration Interactive - Projet PFA (5IADA - EMSI)
"Diseases Cause Oriented Analysis using RAG and Knowledge Graphs"

A exécuter lors du point d'avancement avec l'encadrant.
"""

import sys
import os
from pathlib import Path

# Fix paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def print_header(title: str):
    print("\n" + "=" * 75)
    print(f"  {title.upper()}")
    print("=" * 75)

def demo_summary():
    print_header("1. Présentation des Métriques Clés (Phase 2 & 3)")
    print("""
  Projet : Analyse causale de pathologies via GraphRAG biomédical
  Équipe : Hamza Essamoudi | Salma Ouahman | Ghita El Hajjouji (5IADA - EMSI)
  Pathologies cibles : 
    - Maladie d'Alzheimer (DOID:10652, UMLS: C0002395, MeSH: D000544)
    - Diabète de Type 2   (DOID:9352,  UMLS: C0011860, MeSH: D003924)
    
  [Knowledge Graph - data/processed/nodes_targeted.csv & edges_targeted.csv]
    * Nœuds uniques     : 4 518 (3 045 Gènes, 773 Voies, 347 Molécules, 44 Symptômes, 15 Lifestyle...)
    * Arêtes typées     : 19 951 relations biomédicales (PPI, régulations, cibles, lifestyle...)
    * Causalité ciblée  : 781 arêtes directes Alzheimer, 196 gènes associés (APP, PSEN1, APOE...)
    * Traitements FDA   : Donepezil, Memantine, Galantamine, Rivastigmine
    * Arborescence Mode : Human Lifestyle (Nutrition, Exercice aérobie, Sommeil, Stress...)

  [Base Vectorielle Sémantique - data/vector_store/chroma/]
    * Source de données : NIH MedQuAD (Questions-Réponses cliniques certifiées)
    * Chunks indexés    : 719 chunks enrichis avec métadonnées (qtype, disease, source)
    * Modèle Embedding  : PubMedBERT (NeuML/pubmedbert-base-embeddings, 768 dimensions)
    * Métrique distance : Similarité cosinus normalisée
""")

def demo_pipeline_test():
    print_header("2. Exécution de la Suite de Validation Intégrée")
    print("-> Lancement du script de test global : src/test_pipeline.py\n")
    from src.test_pipeline import main as run_pipeline_tests
    run_pipeline_tests()

def demo_vector_search():
    print_header("3. Démonstration de la Recherche Vectorielle Sémantique (MedQuAD)")
    from src.vector.query_vector_test import VectorQueryEngine
    
    engine = VectorQueryEngine(backend="chroma")
    queries = [
        "What genetic changes in APP, PSEN1, and APOE cause Alzheimer's disease?",
        "What are the approved treatments, cholinesterase inhibitors, and memantine for Alzheimer's?",
        "How does insulin resistance link type 2 diabetes to neurodegenerative risk?",
    ]
    
    for i, q in enumerate(queries, 1):
        print(f"\n[Question Clinique {i}] : \"{q}\"")
        results = engine.search(q, top_k=2)
        for r in results:
            print(f"  * [Score: {r['score']:.4f}] Maladie: {r['disease_name']} | Type: {r['question_type']} | Source: {r['source']}")
            print(f"    Q: {r['question']}")
            print(f"    R: {r['text'][:140]}...")

def demo_graph_causal():
    print_header("4. Démonstration des Chaînes Causales du Knowledge Graph")
    import pandas as pd
    
    subgraph_df = pd.read_csv(BASE_DIR / "data" / "processed" / "subgraph_targeted.csv")
    
    print("\n[A] Traitements approuvés de la maladie d'Alzheimer dans le graphe :")
    treats = subgraph_df[(subgraph_df["relation"] == "TREATS") & (subgraph_df["target_name"] == "Alzheimer's disease")]
    drugs = sorted(treats["source_name"].unique().tolist())
    print(f"  -> Molécules approuvées identifiées ({len(drugs)}) : {drugs}")

    print("\n[B] Chaîne causale multi-échelles (Gène -> Voie Métabolique) :")
    key_genes = ["APP", "PSEN1", "APOE", "MAPT"]
    for g_name in key_genes:
        pw_edges = subgraph_df[(subgraph_df["source_name"] == g_name) & (subgraph_df["relation"] == "PARTICIPATES_IN")].head(3)
        pw_names = pw_edges["target_name"].tolist()
        print(f"  * Gène {g_name} participe aux voies : {pw_names}")

    print("\n[C] Axe Métabolique - Neurodégénératif (Gènes partagés Alzheimer & Diabète T2) :")
    ad_genes = set(subgraph_df[(subgraph_df["source_name"] == "Alzheimer's disease") & (subgraph_df["relation"] == "ASSOCIATES_WITH")]["target_name"])
    t2d_genes = set(subgraph_df[(subgraph_df["source_name"] == "Type 2 diabetes mellitus") & (subgraph_df["relation"] == "ASSOCIATES_WITH")]["target_name"])
    shared = ad_genes.intersection(t2d_genes)
    print(f"  -> {len(shared)} gènes clés partagés découverts : {sorted(list(shared))}")
    print("     (Valide l'hypothèse de l'insulino-résistance cérébrale / 'Diabète de Type 3')")

def demo_graphrag_pipeline():
    print_header("5. Interrogation en Direct du Pipeline Hybride GraphRAG (Phase 3)")
    from src.rag.pipeline import GraphRAGPipeline
    pipeline = GraphRAGPipeline(use_neo4j=False)
    q = "How do mutations in APP, PSEN1, and APOE lead to Alzheimer's disease and memory disorders?"
    print(f"\n[Question Clinique Traitée] : \"{q}\"")
    report = pipeline.query(q, mode="hybrid")
    print(f"  • Pathologie : {report.disease_scope}")
    print(f"  • Gènes identifiés : {report.genetic_triggers}")
    print(f"  • Voies altérées : {report.intermediate_pathways}")
    print(f"  • Manifestations : {report.clinical_manifestations}")
    print(f"  • Thérapeutiques : {report.therapeutic_targets}")
    print("\n  [Chaînes Causales Multi-Échelles du Graphe] :")
    for ch in report.causal_chains_diagram[:3]:
        print(f"    -> {ch}")
    print("\n  [Citations Certifiées] :")
    for c in report.citations[:4]:
        print(f"    * {c}")

def demo_evaluation_benchmark():
    print_header("6. Exécution du Benchmark Évaluatif (Standard RAG vs GraphRAG)")
    from src.rag.evaluate_rag import evaluate_system
    evaluate_system()

def main():
    while True:
        print_header("Démonstrateur PFA - Pipeline GraphRAG Biomédical (Phases 2 & 3)")
        print("  1. Vue d'ensemble et métriques du projet")
        print("  2. Lancer la suite de tests automatisée (Passage à 100%)")
        print("  3. Tester la recherche vectorielle sémantique (MedQuAD + PubMedBERT)")
        print("  4. Explorer les chaînes causales du Knowledge Graph")
        print("  5. Poser une question clinique en direct au Pipeline GraphRAG (Phase 3)")
        print("  6. Lancer le Benchmark Évaluatif (Standard RAG vs GraphRAG)")
        print("  7. Lancer toute la démonstration en continu")
        print("  Q. Quitter")
        
        choice = input("\nVotre choix [1-7 / Q] : ").strip().upper()
        if choice == "1":
            demo_summary()
        elif choice == "2":
            demo_pipeline_test()
        elif choice == "3":
            demo_vector_search()
        elif choice == "4":
            demo_graph_causal()
        elif choice == "5":
            demo_graphrag_pipeline()
        elif choice == "6":
            demo_evaluation_benchmark()
        elif choice == "7":
            demo_summary()
            demo_pipeline_test()
            demo_vector_search()
            demo_graph_causal()
            demo_graphrag_pipeline()
            demo_evaluation_benchmark()
        elif choice == "Q":
            print("\nFin de la démonstration.")
            break
        else:
            print("Choix invalide.")
        input("\nAppuyez sur Entrée pour continuer...")

if __name__ == "__main__":
    # If run with argument --all, run everything non-interactively
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        demo_summary()
        demo_pipeline_test()
        demo_vector_search()
        demo_graph_causal()
        demo_graphrag_pipeline()
    else:
        main()
