"""
Script de génération de l'architecture du projet GraphRAG Biomédical en image haute résolution (300 DPI).
Projet PFA (5IADA - EMSI) : "Diseases Cause Oriented Analysis using RAG and Knowledge Graphs"
Auteurs : Hamza Essamoudi • Salma Ouahman • Ghita El Hajjouji
"""

import os
import tempfile
from pathlib import Path

# Prevent Matplotlib cache warnings
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Configuration des chemins
BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
DOCS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_IMG = DOCS_DIR / "architecture_graphrag.png"

def create_architecture_diagram():
    # Figure setup (16:9 widescreen, 300 DPI pour rendu ultra net)
    fig, ax = plt.subplots(figsize=(19, 11), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Fond d'écran sombre élégant (Deep Slate Navy)
    BG_COLOR = "#0B1120"
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Palette de couleurs professionnelle
    COLOR_TITLE = "#F8FAFC"
    COLOR_AUTHOR = "#38BDF8"
    COLOR_SUBTITLE = "#94A3B8"
    
    # Couleurs par étape
    COLOR_STEP1 = "#0284C7"     # Étape 1 : ETL (Sky Blue)
    COLOR_STEP2_KG = "#0D9488"  # Étape 2A : Neo4j (Teal)
    COLOR_STEP2_V = "#6366F1"   # Étape 2B : Base Vectorielle (Indigo)
    COLOR_STEP3 = "#D97706"     # Étape 3 : GraphRAG Hybrid (Amber)
    COLOR_STEP4_5 = "#059669"   # Étape 4 & 5 : Raisonnement & UI (Emerald)

    # -------------------------------------------------------------------------
    # EN-TÊTE PRINCIPAL
    # -------------------------------------------------------------------------
    ax.text(50, 97.2, "ARCHITECTURE DU SYSTÈME HYBRIDE GRAPHRAG BIOMÉDICAL", 
            ha="center", va="center", fontsize=20, fontweight="bold", color=COLOR_TITLE, fontfamily="sans-serif")
    
    ax.text(50, 94.4, "Hamza Essamoudi   •   Salma Ouahman   •   Ghita El Hajjouji    |    Projet PFA — 5IADA (EMSI)", 
            ha="center", va="center", fontsize=11.5, fontweight="bold", color=COLOR_AUTHOR, fontfamily="sans-serif")
    
    ax.text(50, 91.8, "Analyse Causale Multi-Échelles : Maladie d'Alzheimer (DOID:10652) • Diabète Type 2 (DOID:9352) • Ontologie Human Lifestyle", 
            ha="center", va="center", fontsize=9.2, fontstyle="italic", color=COLOR_SUBTITLE, fontfamily="sans-serif")

    # Fonction utilitaire pour boîtes arrondies
    def draw_box(x, y, w, h, bg_color, border_color="#475569", radius=1.2, alpha=0.95, lw=1.5):
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle=f"round,pad=0.25,rounding_size={radius}",
                              facecolor=bg_color, edgecolor=border_color, linewidth=lw, alpha=alpha, zorder=2)
        ax.add_patch(rect)
        return rect

    # Fonction utilitaire pour badges technologiques
    def draw_badge(x, y, text, color="#38BDF8", bg="#030712", fs=8.0):
        ax.text(x, y, f"  {text}  ", ha="center", va="center", fontsize=fs, fontweight="bold",
                color=color, bbox=dict(boxstyle="round,pad=0.22", facecolor=bg, edgecolor=color, linewidth=1.0), zorder=5)

    # =========================================================================
    # ÉTAPE 1 : SOURCES MULTI-ONTOLOGIES & PIPELINE ETL
    # =========================================================================
    draw_box(3, 44, 28, 45.5, "#0F172A", border_color=COLOR_STEP1, lw=2.0)
    ax.text(17, 87.5, "ÉTAPE 1 : ACQUISITION & ETL MULTI-ONTOLOGIES", ha="center", va="center", fontsize=11, fontweight="bold", color="#38BDF8")
    ax.text(17, 85.5, "Harmonisation Multi-Sources & Normalisation CUI/MeSH", ha="center", va="center", fontsize=8, color="#94A3B8")

    # 1.1 Réseaux Biomédicaux
    draw_box(4.5, 73.5, 25, 10.5, "#1E293B", "#475569")
    ax.text(17, 81.5, "1.1 Réseaux Biomédicaux Relationnels", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#F8FAFC")
    ax.text(17, 77.5, "• PrimeKG-Plus (Ontologies & mappings)\n• Hetionet v1.0 (2,25M arêtes globales)\n• DOID, UMLS, MeSH, Entrez Gene, Reactome", 
            ha="center", va="center", fontsize=7.6, color="#CBD5E1")
    draw_badge(17, 74.0, "Python 3.12 | Pandas | Git LFS", color="#38BDF8")

    # 1.2 Human Lifestyle Ontology (NOUVEAU)
    draw_box(4.5, 59.0, 25, 12.8, "#1E293B", "#0D9488")
    ax.text(17, 69.2, "1.2 Human Lifestyle Ontology (Nouveau)", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#2DD4BF")
    ax.text(17, 64.5, "• Arborescence hiérarchique (MeSH D008019)\n• 15 Nœuds (Nutrition, Activité, Sommeil, Stress)\n• 40 Arêtes causales (PROTECTS, AGGRAVATES)\n• Facteurs modifiables & exposition comportementale", 
            ha="center", va="center", fontsize=7.5, color="#CCFBF1")
    draw_badge(17, 59.8, "MeSH D008019 | UMLS C0023674 | Causal Rules", color="#2DD4BF")

    # 1.3 Littérature Médicale Certifiée
    draw_box(4.5, 45.2, 25, 12.2, "#1E293B", "#475569")
    ax.text(17, 55.2, "1.3 Littérature Médicale Certifiée", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#F8FAFC")
    ax.text(17, 50.8, "• NIH MedQuAD (47 457 Q/R cliniques)\n• Dossiers ciblés : NINDS, GHR, NIDDK, SeniorHealth\n• Chunking sémantique contextualisé [Disease] [QType]\n• 719 Chunks (190 focalisés sur prévention & mode de vie)", 
            ha="center", va="center", fontsize=7.4, color="#CBD5E1")
    draw_badge(17, 46.0, "XML Parsing | Chunking Contextualisé", color="#A855F7")

    # =========================================================================
    # ÉTAPE 2 : STOCKAGE RELATIONNEL ET BASE VECTORIELLE DENSE
    # =========================================================================
    draw_box(34.5, 44, 30.5, 45.5, "#0F172A", border_color="#14B8A6", lw=2.0)
    ax.text(49.7, 87.5, "ÉTAPE 2 : STOCKAGE & INDEXATION DUAL-CORE", ha="center", va="center", fontsize=11, fontweight="bold", color="#2DD4BF")
    ax.text(49.7, 85.5, "Graphe de Causalité Relationnelle & Espace Dense Sémantique", ha="center", va="center", fontsize=8, color="#94A3B8")

    # 2A : Knowledge Graph Causal (Neo4j)
    draw_box(36, 67.5, 27.5, 16.5, "#042F2E", "#14B8A6")
    ax.text(49.7, 81.5, "2A. Knowledge Graph Causal (Neo4j)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#5EEAD4")
    ax.text(49.7, 75.0, "• 4 518 Nœuds uniques (9 labels hiérarchisés)\n• 19 951 Relations causales & fonctionnelles (17 types)\n• 15 Nœuds Lifestyle reliés directement aux maladies\n• 36 Gènes partagés Diabète-Alzheimer (Hypothèse T3D)\n• Ingestion batch UNWIND & contraintes d'intégrité", 
            ha="center", va="center", fontsize=7.6, color="#E6FFFA")
    draw_badge(49.7, 68.3, "Neo4j 5.26 | Cypher | APOC | Docker Compose", color="#2DD4BF")

    # 2B : Base Vectorielle Clinique Dense
    draw_box(36, 45.2, 27.5, 20.5, "#1E1B4B", "#818CF8")
    ax.text(49.7, 63.2, "2B. Base Vectorielle Clinique Dense", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#A5B4FC")
    ax.text(49.7, 55.2, "• 719 Chunks contextuels MedQuAD indexés\n• Modèle biomédical NeuML/pubmedbert-base-embeddings\n• Espace vectoriel dense certifié : 768 dimensions\n• Métadonnées cliniques (qtype, source, disease, focus)\n• Architecture hybride de persistance :\n  - ChromaDB : moteur embarqué local sans dépendance\n  - Qdrant : microservice Docker pour production", 
            ha="center", va="center", fontsize=7.5, color="#EEF2FF")
    draw_badge(49.7, 46.2, "PubMedBERT (768d) | ChromaDB | Qdrant Docker", color="#818CF8")

    # =========================================================================
    # ÉTAPE 3 : RETRIEVER HYBRIDE & FUSION DE CONTEXTE
    # =========================================================================
    draw_box(68.5, 44, 28.5, 45.5, "#0F172A", border_color=COLOR_STEP3, lw=2.0)
    ax.text(82.7, 87.5, "ÉTAPE 3 : MOTEUR HYBRIDE GRAPHRAG", ha="center", va="center", fontsize=11, fontweight="bold", color="#FBBF24")
    ax.text(82.7, 85.5, "Traversée k-Hop Ciblée + Recherche Vectorielle Dense", ha="center", va="center", fontsize=8, color="#94A3B8")

    # 3.1 Retrieval Conjoint
    draw_box(70, 67.5, 25.5, 16.5, "#451A03", "#F59E0B")
    ax.text(82.7, 81.5, "3.1 Retrieval Conjoint Structuré / Dense", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#FDE68A")
    ax.text(82.7, 75.0, "• BiomedicalEntityLinker (30+ alias & concepts)\n• Traçage k-hop : (Lifestyle) → (Gènes) → (Voies) → (Maladie)\n• Passerelle métabolique : 36 gènes partagés (IRS1, AKT1, TNF)\n• Recherche vectorielle par similarité cosinus (Top-k)\n• Scores combinés de pertinence contextuelle", 
            ha="center", va="center", fontsize=7.6, color="#FEF3C7")
    draw_badge(82.7, 68.3, "HybridRetriever | LocalGraphEngine", color="#FBBF24")

    # 3.2 Fusion de Contexte & Linéarisation Causal
    draw_box(70, 45.2, 25.5, 20.5, "#451A03", "#F59E0B")
    ax.text(82.7, 63.2, "3.2 Fusion & Structuration Causal", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#FDE68A")
    ax.text(82.7, 55.2, "• Linéarisation ordonnée des triplets relationnels\n• Structuration conjointe (Graphe formel + Extraits NIH)\n• Élimination documentée des hallucinations médicales\n• Éradication des ruptures déductives et faux liens\n• Préparation du prompt enrichi avec citations traçables", 
            ha="center", va="center", fontsize=7.6, color="#FEF3C7")
    draw_badge(82.7, 46.2, "ContextFusionEngine | Pydantic Schema", color="#F59E0B")

    # =========================================================================
    # ÉTAPES 4 & 5 : RAISONNEMENT CAUSAL, BENCHMARK & APPLICATION STREAMLIT
    # =========================================================================
    draw_box(3, 4, 94, 36.5, "#064E3B", border_color="#10B981", lw=2.0)
    ax.text(50, 37.8, "ÉTAPE 4 & 5 : RAISONNEMENT CAUSAL MULTI-ÉCHELLES, BENCHMARK & INTERFACES UTILISATEUR", 
            ha="center", va="center", fontsize=11.5, fontweight="bold", color="#6EE7B7")
    ax.text(50, 36.0, "Synthèse Clinique Explicable, Validation Expérimentale & Déploiement Multi-Canaux", 
            ha="center", va="center", fontsize=8.2, color="#A7F3D0")

    # Colonne 1 : Moteur de Raisonnement Causal
    draw_box(4.5, 5.8, 28.5, 28.5, "#022C22", "#059669")
    ax.text(18.7, 31.8, "Moteur de Synthèse Causale", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#6EE7B7")
    ax.text(18.7, 20.5, "Raisonnement explicable en 5 étapes articulées :\n\n"
                        "1. Déclencheurs étiologiques (APP, PSEN1, APOE, facteurs de risque)\n"
                        "2. Cascades intermédiaires (Résistance insuline, neuro-inflammation)\n"
                        "3. Impact Lifestyle : Neuroprotection (BDNF) vs Toxicité (TNF)\n"
                        "4. Tableau clinique (Plaques amyloïdes, perte synaptique, mémoire)\n"
                        "5. Repositionnement (Donepezil, Memantine, Metformine, Diète)\n"
                        "6. Traçabilité absolue : Triplets graphe & identifiants NIH MedQuAD", 
            ha="center", va="center", fontsize=7.3, color="#ECFDF5")
    draw_badge(18.7, 7.5, "CausalGenerator | LLM Routing (Gemini / Déterministe)", color="#34D399")

    # Colonne 2 : Benchmark d'Évaluation Rigoureux
    draw_box(35.5, 5.8, 28.5, 28.5, "#022C22", "#059669")
    ax.text(49.7, 31.8, "Benchmark : Standard RAG vs GraphRAG", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#6EE7B7")
    ax.text(49.7, 20.5, "Comparatif quantitatif sur 5 cas cliniques complexes :\n\n"
                        "• Chaînes causales tracées : 0 (Standard RAG) vs 23 (GraphRAG)\n"
                        "• Couverture des voies métaboliques : +100% via le graphe\n"
                        "• Triplets et citations d'évidences : 15 (Standard) vs 35 (GraphRAG)\n"
                        "• Connexions croisées Diabète-Alzheimer : 0 vs 100% identifiées\n"
                        "• Traçabilité complète et reproductibilité certifiée\n"
                        "• Fichier de résultats généré : data/processed/evaluation_results.json", 
            ha="center", va="center", fontsize=7.3, color="#ECFDF5")
    draw_badge(49.7, 7.5, "evaluate_rag.py | Benchmark Suite | 5 Cas Cliniques", color="#F59E0B")

    # Colonne 3 : Interfaces Utilisateur & Déploiement
    draw_box(66.5, 5.8, 28.5, 28.5, "#022C22", "#059669")
    ax.text(80.7, 31.8, "Interfaces Utilisateur & Déploiement", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#6EE7B7")
    ax.text(80.7, 20.5, "Écosystème interactif clé en main pour encadrant / jury :\n\n"
                        "• Application Web Interactive Streamlit (app.py & run_app.bat) :\n"
                        "  - Onglet 1 : Chatbot GraphRAG avec streaming & citations\n"
                        "  - Onglet 2 : Visualiseur 2D/3D (36 gènes communs & lifestyle)\n"
                        "  - Onglet 3 : Dashboard d'évaluation Standard RAG vs GraphRAG\n"
                        "  - Onglet 4 : Documentation & architecture détaillée\n"
                        "• CLI Interactif (src/rag/cli.py) & Démo console (demo.py)\n"
                        "• Dépôt GitHub : Hamza3124/biomedical-graphrag (Branch main)", 
            ha="center", va="center", fontsize=7.2, color="#ECFDF5")
    draw_badge(80.7, 7.5, "Streamlit (app.py) | CLI | Docker | GitHub CI", color="#38BDF8")

    # =========================================================================
    # FLÈCHES DE FLUX DIRECTIONNELLES (DATAFLOW ARROWS)
    # =========================================================================
    def draw_arrow(x1, y1, x2, y2, color="#38BDF8", rad=0.0):
        connection = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            color=color,
            linewidth=2.0,
            connectionstyle=connection,
            zorder=10
        )
        ax.add_patch(arrow)

    # ETL -> Knowledge Graph
    draw_arrow(31, 75.5, 34.5, 75.5, color="#2DD4BF")
    ax.text(32.7, 77.2, "Triplets\nCSVs", ha="center", va="center", fontsize=7.2, color="#2DD4BF", fontweight="bold")

    # ETL -> Vector Store
    draw_arrow(31, 55.5, 34.5, 55.5, color="#818CF8")
    ax.text(32.7, 57.2, "Chunks\nXML", ha="center", va="center", fontsize=7.2, color="#818CF8", fontweight="bold")

    # Knowledge Graph -> Hybrid Retriever
    draw_arrow(65, 75.5, 68.5, 75.5, color="#2DD4BF")
    ax.text(66.7, 77.5, "Chemins k-Hop\n& 36 Gènes", ha="center", va="center", fontsize=7.0, color="#2DD4BF", fontweight="bold")

    # Vector Store -> Hybrid Retriever
    draw_arrow(65, 55.5, 68.5, 55.5, color="#818CF8")
    ax.text(66.7, 57.5, "Top-k Chunks\nPubMedBERT", ha="center", va="center", fontsize=7.0, color="#818CF8", fontweight="bold")

    # Hybrid Retriever -> Causal Generation / Reasoning
    draw_arrow(82.7, 44.0, 82.7, 40.5, color="#F59E0B")
    ax.text(85.5, 42.2, "Contexte Causal Fusionné", ha="left", va="center", fontsize=7.5, color="#FBBF24", fontweight="bold")

    # Sauvegarde de l'image
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, facecolor=fig.get_facecolor(), edgecolor="none", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Diagramme d'architecture généré avec succès dans : {OUTPUT_IMG}")

if __name__ == "__main__":
    create_architecture_diagram()
