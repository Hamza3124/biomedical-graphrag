"""
Script de génération de l'architecture du projet GraphRAG Biomédical en image haute résolution (300 DPI).
PFA (5IADA - EMSI) - Diseases Cause Oriented Analysis using RAG and Knowledge Graphs.
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

# Configure output directory
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG = DOCS_DIR / "architecture_graphrag.png"

def create_architecture_diagram():
    # Figure setup (16:9 aspect ratio, 300 DPI)
    fig, ax = plt.subplots(figsize=(18, 10.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Background color (subtle off-white/light-slate)
    fig.patch.set_facecolor("#0F172A") # Deep Slate Navy
    ax.set_facecolor("#0F172A")

    # Fonts & Colors palette
    COLOR_TITLE = "#F8FAFC"
    COLOR_SUBTITLE = "#94A3B8"
    COLOR_BORDER = "#334155"
    
    # 5 Major Layer Colors
    COLOR_LAYER1 = "#1E293B" # ETL & Data
    COLOR_LAYER2_G = "#0F766E" # Knowledge Graph (Teal)
    COLOR_LAYER2_V = "#4338CA" # Vector Store (Indigo)
    COLOR_LAYER3 = "#0284C7" # Hybrid Retriever (Sky Blue)
    COLOR_LAYER4 = "#D97706" # Context Fusion (Amber)
    COLOR_LAYER5 = "#059669" # LLM Causal Generator (Emerald)

    # 1. Main Header
    ax.text(50, 96.5, "ARCHITECTURE DU SYSTÈME HYBRIDE GRAPHRAG BIOMÉDICAL", 
            ha="center", va="center", fontsize=20, fontweight="bold", color=COLOR_TITLE, fontfamily="sans-serif")
    ax.text(50, 93.5, "Analyse Causale Multi-Échelles de Pathologies (Alzheimer & Diabète de Type 2) | Projet PFA - 5IADA (EMSI)", 
            ha="center", va="center", fontsize=11, color=COLOR_SUBTITLE, fontfamily="sans-serif")

    # Helper function to draw rounded boxes
    def draw_box(x, y, w, h, bg_color, border_color="#475569", radius=1.5, alpha=0.95):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle=f"round,pad=0.3,rounding_size={radius}",
                              facecolor=bg_color, edgecolor=border_color, linewidth=1.5, alpha=alpha, zorder=2)
        ax.add_patch(rect)
        return rect

    # Helper function for tech badges
    def draw_badge(x, y, text, color="#38BDF8", bg="#0F172A"):
        ax.text(x, y, f" {text} ", ha="center", va="center", fontsize=8.5, fontweight="bold",
                color=color, bbox=dict(boxstyle="round,pad=0.2", facecolor=bg, edgecolor=color, linewidth=1.0), zorder=5)

    # =========================================================================
    # ÉTAPE 1 : SOURCES DE DONNÉES & PIPELINE ETL
    # =========================================================================
    draw_box(4, 53, 25, 36, COLOR_LAYER1, border_color="#38BDF8")
    ax.text(16.5, 86, "ÉTAPE 1 : ETL & HARMONISATION", ha="center", va="center", fontsize=12, fontweight="bold", color="#38BDF8")
    ax.text(16.5, 83.5, "Acquisition & Normalisation Multi-Ontologies", ha="center", va="center", fontsize=9, color="#94A3B8")

    # Sub-cards
    # PrimeKG & Hetionet
    draw_box(6, 68, 21, 13, "#334155", "#64748B")
    ax.text(16.5, 78, "Réseaux Biomédicaux Relations", ha="center", va="center", fontsize=10, fontweight="bold", color="#F1F5F9")
    ax.text(16.5, 75, "• PrimeKG-Plus (Ontologies & Mappings)\n• Hetionet v1.0 (2,25M arêtes globales)\n• DOID, UMLS, MeSH, Entrez, Reactome", 
            ha="center", va="center", fontsize=8.2, color="#CBD5E1")
    draw_badge(16.5, 69.5, "Python 3.12 | Pandas | Git LFS", color="#38BDF8")

    # MedQuAD
    draw_box(6, 54.5, 21, 11.5, "#334155", "#64748B")
    ax.text(16.5, 63, "Littérature Médicale Certifiée", ha="center", va="center", fontsize=10, fontweight="bold", color="#F1F5F9")
    ax.text(16.5, 60, "• NIH MedQuAD (47 457 QA Pairs)\n• Dossiers NINDS, GHR, NIDDK, MPlus\n• Découpage contextuel [Disease] [QType]", 
            ha="center", va="center", fontsize=8.2, color="#CBD5E1")
    draw_badge(16.5, 55.8, "XML Parsing | Chunking Sémantique", color="#A855F7")

    # =========================================================================
    # ÉTAPE 2 : KNOWLEDGE GRAPH (NEO4J) & BASE VECTORIELLE (CHROMADB)
    # =========================================================================
    # Branch A: Knowledge Graph
    draw_box(33, 68, 30, 21, "#115E59", border_color="#2DD4BF")
    ax.text(48, 86, "ÉTAPE 2A : KNOWLEDGE GRAPH CAUSAL", ha="center", va="center", fontsize=12, fontweight="bold", color="#2DD4BF")
    ax.text(48, 83.5, "Topologie Relationnelle & Causalité Multi-Échelles", ha="center", va="center", fontsize=9, color="#CCFBF1")
    ax.text(48, 77.5, "• 4 503 Nœuds uniques (Disease, Gene, Pathway, Compound...)\n• 19 911 Relations typées (PPI, régulations, cibles, phénotypes)\n• 772 Arêtes directes Alzheimer (APP, PSEN1, APOE, MAPT)\n• Ingestion Batch UNWIND & Contraintes d'intégrité", 
            ha="center", va="center", fontsize=8.5, color="#F0FDFA")
    draw_badge(48, 70, "Neo4j 5.26 | Cypher | APOC | Docker Compose", color="#2DD4BF")

    # Branch B: Vector Store
    draw_box(33, 44, 30, 21, "#312E81", border_color="#818CF8")
    ax.text(48, 62, "ÉTAPE 2B : BASE VECTORIELLE DENSE", ha="center", va="center", fontsize=12, fontweight="bold", color="#818CF8")
    ax.text(48, 59.5, "Indexation Sémantique des Connaissances NIH", ha="center", va="center", fontsize=9, color="#E0E7FF")
    ax.text(48, 53.5, "• 719 Chunks contextuels MedQuAD enrichis\n• Embeddings biomédicaux PubMedBERT (768 dimensions)\n• Métadonnées cliniques : disease_name, qtype, doc_id\n• Recherche de similarité cosinus normalisée", 
            ha="center", va="center", fontsize=8.5, color="#EEF2FF")
    draw_badge(48, 46, "PubMedBERT | Sentence-Transformers | ChromaDB / Qdrant", color="#818CF8")

    # =========================================================================
    # ÉTAPE 3 : RETRIEVER HYBRIDE & FUSION DE CONTEXTE
    # =========================================================================
    draw_box(67, 53, 29, 36, "#0369A1", border_color="#38BDF8")
    ax.text(81.5, 86, "ÉTAPE 3 : MOTEUR HYBRIDE GRAPHRAG", ha="center", va="center", fontsize=12, fontweight="bold", color="#38BDF8")
    ax.text(81.5, 83.5, "Traversée de Graphe k-Hop + Recherche Vectorielle", ha="center", va="center", fontsize=9, color="#BAE6FD")

    # Sub-card: Entity Linker & Hybrid Retriever
    draw_box(68.5, 68, 26, 13, "#0C4A6E", "#0284C7")
    ax.text(81.5, 78, "Retrieval Conjoint Structuré / Dense", ha="center", va="center", fontsize=10, fontweight="bold", color="#F0F9FF")
    ax.text(81.5, 74.5, "• BiomedicalEntityLinker (Regex, alias, symboles)\n• Traçage k-hop : (Maladie)->(Gènes)->(Voies)->(Phénotype)\n• Axe Diabète-Alzheimer : 36 gènes partagés\n• Top-k Chunks MedQuAD avec score de pertinence", 
            ha="center", va="center", fontsize=8.2, color="#E0F2FE")
    draw_badge(81.5, 69.5, "LocalGraphEngine | HybridRetriever", color="#38BDF8")

    # Sub-card: Context Fusion Engine
    draw_box(68.5, 54.5, 26, 11.5, "#0C4A6E", "#0284C7")
    ax.text(81.5, 63, "Fusion & Formateur Causal", ha="center", va="center", fontsize=10, fontweight="bold", color="#F0F9FF")
    ax.text(81.5, 59.8, "• Linéarisation des triplets causaux & chemins\n• Structuration conjointe (Faits structurés + Texte NIH)\n• Élimination des hallucinations et sauts logiques", 
            ha="center", va="center", fontsize=8.2, color="#E0F2FE")
    draw_badge(81.5, 55.8, "ContextFusionEngine | Pydantic", color="#F59E0B")

    # =========================================================================
    # ÉTAPE 4 & 5 : RAISONNEMENT CAUSAL, SYNTHÈSE LLM & ÉVALUATION
    # =========================================================================
    draw_box(4, 5, 92, 34, "#064E3B", border_color="#34D399")
    ax.text(50, 36, "ÉTAPE 4 & 5 : RAISONNEMENT CAUSAL MULTI-ÉCHELLES & ÉVALUATION COMPARATIVE", 
            ha="center", va="center", fontsize=13, fontweight="bold", color="#34D399")
    ax.text(50, 33.5, "Génération d'Explications Causales Sourcées & Benchmarking Rigoureux", 
            ha="center", va="center", fontsize=9.5, color="#A7F3D0")

    # Column 1: Causal Generator
    draw_box(6, 9, 28, 22, "#022C22", "#059669")
    ax.text(20, 28, "Moteur de Synthèse Causale", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#6EE7B7")
    ax.text(20, 20.5, "1. Déclencheurs étiologiques (Mutations APP, PSEN1, APOE)\n2. Cascades intermédiaires (Apoptose, métabolisme)\n3. Tableau clinique (Pertes de mémoire, démence)\n4. Repositionnement (Donepezil, Memantine)\n5. Traçabilité (Triplets graphe + DocIDs NIH)", 
            ha="center", va="center", fontsize=8.0, color="#ECFDF5")
    draw_badge(20, 11, "CausalGenerator | LLM Routing (Gemini/OpenAI)", color="#34D399")

    # Column 2: Benchmark Evaluation
    draw_box(36, 9, 30, 22, "#022C22", "#059669")
    ax.text(51, 28, "Évaluation : Standard RAG vs GraphRAG", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#6EE7B7")
    ax.text(51, 20.5, "• Chaînes causales tracées : 0 (Standard) vs 23 (GraphRAG)\n• Couverture des voies métaboliques : +100% via le graphe\n• Triplets et citations d'évidences : 15 vs 35\n• Traçabilité complète et reproductibilité certifiée\n• Rapport automatisé : evaluation_results.json", 
            ha="center", va="center", fontsize=8.0, color="#ECFDF5")
    draw_badge(51, 11, "Benchmark Suite | evaluate_rag.py", color="#F59E0B")

    # Column 3: Interfaces & Démonstration
    draw_box(68, 9, 26, 22, "#022C22", "#059669")
    ax.text(81, 28, "Interfaces & Déploiement", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#6EE7B7")
    ax.text(81, 20.5, "• Démonstrateur interactif clé en main (demo.py)\n• Lanceur Windows autonome (run_demo.bat)\n• Suite de tests globale (test_pipeline.py)\n• Dépôt Git : Hamza3124/biomedical-graphrag\n• Documentation académique complète (README)", 
            ha="center", va="center", fontsize=8.0, color="#ECFDF5")
    draw_badge(81, 11, "Docker | CLI Interactif | Git / GitHub", color="#38BDF8")

    # =========================================================================
    # CONNECTING ARROWS & DATA FLOWS
    # =========================================================================
    def draw_arrow(x1, y1, x2, y2, color="#38BDF8", style="->", rad=0.0):
        connection = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=18,
            color=color,
            linewidth=2.0,
            connectionstyle=connection,
            zorder=10
        )
        ax.add_patch(arrow)

    # ETL -> Knowledge Graph
    draw_arrow(29, 78.5, 33, 78.5, color="#2DD4BF")
    ax.text(31, 80, "Triplets\nCSVs", ha="center", va="center", fontsize=7.5, color="#2DD4BF", fontweight="bold")

    # ETL -> Vector Store
    draw_arrow(29, 60, 33, 54.5, color="#818CF8", rad=-0.1)
    ax.text(31, 57, "Chunks\nXML", ha="center", va="center", fontsize=7.5, color="#818CF8", fontweight="bold")

    # Knowledge Graph -> Hybrid Retriever
    draw_arrow(63, 78.5, 67, 78.5, color="#2DD4BF")
    ax.text(65, 80.5, "k-Hop\nPaths", ha="center", va="center", fontsize=7.5, color="#2DD4BF", fontweight="bold")

    # Vector Store -> Hybrid Retriever
    draw_arrow(63, 54.5, 67, 60, color="#818CF8", rad=0.1)
    ax.text(65, 57, "Top-k\nChunks", ha="center", va="center", fontsize=7.5, color="#818CF8", fontweight="bold")

    # Hybrid Retriever -> Causal Generation / Reasoning
    draw_arrow(81.5, 53, 81.5, 39, color="#F59E0B")
    ax.text(84, 46, "Fused\nContext", ha="left", va="center", fontsize=8.0, color="#F59E0B", fontweight="bold")

    # Output file
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, facecolor=fig.get_facecolor(), edgecolor="none", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Diagramme d'architecture généré avec succès dans : {OUTPUT_IMG}")

if __name__ == "__main__":
    create_architecture_diagram()
