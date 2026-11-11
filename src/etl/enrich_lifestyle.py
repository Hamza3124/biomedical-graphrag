"""
Human Lifestyle Ontology & Causal Enrichment Module.
PFA (5IADA - EMSI) - Diseases Cause Oriented Analysis using RAG and Knowledge Graphs.

Builds a hierarchical Human Lifestyle ontology and integrates causal links
connecting behavioral factors (diet, exercise, sleep, habits, stress) to
genes, biological pathways, symptoms, and target pathologies (Alzheimer & Diabetes).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger("ETL.LifestyleEnrichment")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


# Definition of Hierarchical Lifestyle Nodes
LIFESTYLE_NODES = [
    # Root Category
    {
        "id": "Lifestyle:Root",
        "name": "Human Lifestyle",
        "type": "LifestyleCategory",
        "identifier_type": "CustomOntology",
        "external_id": "LS_ROOT",
        "umls_cui": "C0023674", # Life Style
        "mesh_id": "D008019",
        "doid": None,
    },
    # Sub-Categories (Level 1)
    {
        "id": "Lifestyle:Nutrition",
        "name": "Nutrition and Dietary Habits",
        "type": "LifestyleCategory",
        "identifier_type": "CustomOntology",
        "external_id": "LS_NUT",
        "umls_cui": "C0012150", # Diet
        "mesh_id": "D004032",
        "doid": None,
    },
    {
        "id": "Lifestyle:Physical_Activity",
        "name": "Physical Activity and Exercise",
        "type": "LifestyleCategory",
        "identifier_type": "CustomOntology",
        "external_id": "LS_PA",
        "umls_cui": "C0015259", # Exercise
        "mesh_id": "D015444",
        "doid": None,
    },
    {
        "id": "Lifestyle:Sleep",
        "name": "Sleep and Circadian Rhythm",
        "type": "LifestyleCategory",
        "identifier_type": "CustomOntology",
        "external_id": "LS_SLEEP",
        "umls_cui": "C0037315", # Sleep
        "mesh_id": "D012890",
        "doid": None,
    },
    {
        "id": "Lifestyle:Toxic_Habits",
        "name": "Toxic Habits and Substances",
        "type": "LifestyleCategory",
        "identifier_type": "CustomOntology",
        "external_id": "LS_TOX",
        "umls_cui": "C0038332", # Substance-Related Disorders
        "mesh_id": "D019966",
        "doid": None,
    },
    {
        "id": "Lifestyle:Stress_Mental",
        "name": "Psychosocial Stress and Cognitive Engagement",
        "type": "LifestyleCategory",
        "identifier_type": "CustomOntology",
        "external_id": "LS_STRESS",
        "umls_cui": "C0038435", # Psychological Stress
        "mesh_id": "D013315",
        "doid": None,
    },
    # Specific Behavioral Factors (Level 2 - Modifiable Exposures)
    {
        "id": "Lifestyle:Mediterranean_Diet",
        "name": "Mediterranean and DASH Diet",
        "type": "LifestyleFactor",
        "identifier_type": "CustomOntology",
        "external_id": "LS_MED_DIET",
        "umls_cui": "C0454378", # Mediterranean Diet
        "mesh_id": "D038441",
        "doid": None,
    },
    {
        "id": "Lifestyle:High_Sugar_Ultraprocessed_Diet",
        "name": "High Glycemic and Ultra-Processed Diet",
        "type": "LifestyleFactor",
        "identifier_type": "CustomOntology",
        "external_id": "LS_SUGAR_DIET",
        "umls_cui": "C0012170", # Western diet
        "mesh_id": "D000078022",
        "doid": None,
    },
    {
        "id": "Lifestyle:Aerobic_Exercise",
        "name": "Regular Aerobic Physical Exercise",
        "type": "LifestyleFactor",
        "identifier_type": "CustomOntology",
        "external_id": "LS_AEROBIC",
        "umls_cui": "C0015259", # Aerobic exercise
        "mesh_id": "D000072336",
        "doid": None,
    },
    {
        "id": "Lifestyle:Sedentary_Behavior",
        "name": "Sedentary Behavior and Inactivity",
        "type": "LifestyleFactor",
        "identifier_type": "CustomOntology",
        "external_id": "LS_SEDENTARY",
        "umls_cui": "C4086663", # Sedentary lifestyle
        "mesh_id": "D000074122",
        "doid": None,
    },
    {
        "id": "Lifestyle:Chronic_Sleep_Deprivation",
        "name": "Chronic Sleep Deprivation and Sleep Apnea",
        "type": "LifestyleFactor",
        "identifier_type": "CustomOntology",
        "external_id": "LS_SLEEP_DEP",
        "umls_cui": "C0520679", # Sleep deprivation
        "mesh_id": "D012892",
        "doid": None,
    },
    {
        "id": "Lifestyle:Tobacco_Smoking",
        "name": "Tobacco Smoking Habit",
        "type": "LifestyleFactor",
        "identifier_type": "CustomOntology",
        "external_id": "LS_SMOKING",
        "umls_cui": "C0037369", # Smoking
        "mesh_id": "D012907",
        "doid": None,
    },
    {
        "id": "Lifestyle:Chronic_Excessive_Alcohol",
        "name": "Chronic Excessive Alcohol Intake",
        "type": "LifestyleFactor",
        "identifier_type": "CustomOntology",
        "external_id": "LS_ALCOHOL",
        "umls_cui": "C0001948", # Alcohol drinking
        "mesh_id": "D000428",
        "doid": None,
    },
    {
        "id": "Lifestyle:Chronic_Psychological_Stress",
        "name": "Chronic Psychological Stress and Hypercortisolemia",
        "type": "LifestyleFactor",
        "identifier_type": "CustomOntology",
        "external_id": "LS_CHRONIC_STRESS",
        "umls_cui": "C0038435",
        "mesh_id": "D013315",
        "doid": None,
    },
    {
        "id": "Lifestyle:Cognitive_Stimulation",
        "name": "Cognitive Reserve and Intellectual Engagement",
        "type": "LifestyleFactor",
        "identifier_type": "CustomOntology",
        "external_id": "LS_COG_RESERVE",
        "umls_cui": "C1516597", # Cognitive reserve
        "mesh_id": "D000075222",
        "doid": None,
    },
    # Auxiliary Biological Node (BDNF - Brain-Derived Neurotrophic Factor if not present)
    {
        "id": "Gene:627",
        "name": "BDNF",
        "type": "Gene",
        "identifier_type": "Entrez",
        "external_id": "627",
        "umls_cui": "C0006097",
        "mesh_id": "D019208",
        "doid": None,
    }
]


# Causal Relationships connecting Lifestyle to the Biological Graph
LIFESTYLE_EDGES = [
    # --- 1. Ontological Hierarchy (SUBCLASS_OF) ---
    {"source_id": "Lifestyle:Nutrition", "relation": "SUBCLASS_OF", "target_id": "Lifestyle:Root", "metaedge": "LScLS", "causal_type": "hierarchy", "hop": 1},
    {"source_id": "Lifestyle:Physical_Activity", "relation": "SUBCLASS_OF", "target_id": "Lifestyle:Root", "metaedge": "LScLS", "causal_type": "hierarchy", "hop": 1},
    {"source_id": "Lifestyle:Sleep", "relation": "SUBCLASS_OF", "target_id": "Lifestyle:Root", "metaedge": "LScLS", "causal_type": "hierarchy", "hop": 1},
    {"source_id": "Lifestyle:Toxic_Habits", "relation": "SUBCLASS_OF", "target_id": "Lifestyle:Root", "metaedge": "LScLS", "causal_type": "hierarchy", "hop": 1},
    {"source_id": "Lifestyle:Stress_Mental", "relation": "SUBCLASS_OF", "target_id": "Lifestyle:Root", "metaedge": "LScLS", "causal_type": "hierarchy", "hop": 1},
    
    {"source_id": "Lifestyle:Mediterranean_Diet", "relation": "SUBCLASS_OF", "target_id": "Lifestyle:Nutrition", "metaedge": "LFcLS", "causal_type": "hierarchy", "hop": 1},
    {"source_id": "Lifestyle:High_Sugar_Ultraprocessed_Diet", "relation": "SUBCLASS_OF", "target_id": "Lifestyle:Nutrition", "metaedge": "LFcLS", "causal_type": "hierarchy", "hop": 1},
    {"source_id": "Lifestyle:Aerobic_Exercise", "relation": "SUBCLASS_OF", "target_id": "Lifestyle:Physical_Activity", "metaedge": "LFcLS", "causal_type": "hierarchy", "hop": 1},
    {"source_id": "Lifestyle:Sedentary_Behavior", "relation": "SUBCLASS_OF", "target_id": "Lifestyle:Physical_Activity", "metaedge": "LFcLS", "causal_type": "hierarchy", "hop": 1},
    {"source_id": "Lifestyle:Chronic_Sleep_Deprivation", "relation": "SUBCLASS_OF", "target_id": "Lifestyle:Sleep", "metaedge": "LFcLS", "causal_type": "hierarchy", "hop": 1},
    {"source_id": "Lifestyle:Tobacco_Smoking", "relation": "SUBCLASS_OF", "target_id": "Lifestyle:Toxic_Habits", "metaedge": "LFcLS", "causal_type": "hierarchy", "hop": 1},
    {"source_id": "Lifestyle:Chronic_Excessive_Alcohol", "relation": "SUBCLASS_OF", "target_id": "Lifestyle:Toxic_Habits", "metaedge": "LFcLS", "causal_type": "hierarchy", "hop": 1},
    {"source_id": "Lifestyle:Chronic_Psychological_Stress", "relation": "SUBCLASS_OF", "target_id": "Lifestyle:Stress_Mental", "metaedge": "LFcLS", "causal_type": "hierarchy", "hop": 1},
    {"source_id": "Lifestyle:Cognitive_Stimulation", "relation": "SUBCLASS_OF", "target_id": "Lifestyle:Stress_Mental", "metaedge": "LFcLS", "causal_type": "hierarchy", "hop": 1},

    # --- 2. Causal Direct Protection / Mitigation (PROTECTS) ---
    {"source_id": "Lifestyle:Mediterranean_Diet", "relation": "PROTECTS", "target_id": "DOID:10652", "metaedge": "LFpD", "causal_type": "protective", "hop": 1},
    {"source_id": "Lifestyle:Mediterranean_Diet", "relation": "PROTECTS", "target_id": "DOID:9352", "metaedge": "LFpD", "causal_type": "protective", "hop": 1},
    {"source_id": "Lifestyle:Aerobic_Exercise", "relation": "PROTECTS", "target_id": "DOID:10652", "metaedge": "LFpD", "causal_type": "protective", "hop": 1},
    {"source_id": "Lifestyle:Aerobic_Exercise", "relation": "PROTECTS", "target_id": "DOID:9352", "metaedge": "LFpD", "causal_type": "protective", "hop": 1},
    {"source_id": "Lifestyle:Cognitive_Stimulation", "relation": "PROTECTS", "target_id": "DOID:10652", "metaedge": "LFpD", "causal_type": "protective", "hop": 1},

    # --- 3. Causal Aggravation / Risk Factor (AGGRAVATES / PREDISPOSES) ---
    {"source_id": "Lifestyle:High_Sugar_Ultraprocessed_Diet", "relation": "AGGRAVATES", "target_id": "DOID:9352", "metaedge": "LFaD", "causal_type": "aggravating", "hop": 1},
    {"source_id": "Lifestyle:High_Sugar_Ultraprocessed_Diet", "relation": "AGGRAVATES", "target_id": "DOID:10652", "metaedge": "LFaD", "causal_type": "aggravating", "hop": 1},
    {"source_id": "Lifestyle:Sedentary_Behavior", "relation": "AGGRAVATES", "target_id": "DOID:9352", "metaedge": "LFaD", "causal_type": "aggravating", "hop": 1},
    {"source_id": "Lifestyle:Sedentary_Behavior", "relation": "AGGRAVATES", "target_id": "DOID:10652", "metaedge": "LFaD", "causal_type": "aggravating", "hop": 1},
    {"source_id": "Lifestyle:Chronic_Sleep_Deprivation", "relation": "AGGRAVATES", "target_id": "DOID:10652", "metaedge": "LFaD", "causal_type": "aggravating", "hop": 1},
    {"source_id": "Lifestyle:Tobacco_Smoking", "relation": "AGGRAVATES", "target_id": "DOID:10652", "metaedge": "LFaD", "causal_type": "aggravating", "hop": 1},
    {"source_id": "Lifestyle:Chronic_Excessive_Alcohol", "relation": "AGGRAVATES", "target_id": "DOID:10652", "metaedge": "LFaD", "causal_type": "aggravating", "hop": 1},
    {"source_id": "Lifestyle:Chronic_Psychological_Stress", "relation": "AGGRAVATES", "target_id": "DOID:10652", "metaedge": "LFaD", "causal_type": "aggravating", "hop": 1},
    {"source_id": "Lifestyle:Chronic_Psychological_Stress", "relation": "AGGRAVATES", "target_id": "DOID:9352", "metaedge": "LFaD", "causal_type": "aggravating", "hop": 1},

    # --- 4. Molecular Mechanisms: Regulation of Target Genes (UPREGULATES / DOWNREGULATES) ---
    # Aerobic Exercise upregulates neuroprotective BDNF, insulin-sensitizing AKT1 and PPARG
    {"source_id": "Lifestyle:Aerobic_Exercise", "relation": "UPREGULATES", "target_id": "Gene:627", "metaedge": "LFuG", "causal_type": "upregulation", "hop": 1}, # BDNF
    {"source_id": "Lifestyle:Aerobic_Exercise", "relation": "UPREGULATES", "target_id": "Gene:207", "metaedge": "LFuG", "causal_type": "upregulation", "hop": 1}, # AKT1
    {"source_id": "Lifestyle:Aerobic_Exercise", "relation": "UPREGULATES", "target_id": "Gene:5468", "metaedge": "LFuG", "causal_type": "upregulation", "hop": 1}, # PPARG
    
    # Mediterranean diet modulates PPARG and anti-inflammatory pathways
    {"source_id": "Lifestyle:Mediterranean_Diet", "relation": "UPREGULATES", "target_id": "Gene:5468", "metaedge": "LFuG", "causal_type": "upregulation", "hop": 1}, # PPARG
    {"source_id": "Lifestyle:Mediterranean_Diet", "relation": "UPREGULATES", "target_id": "Gene:3667", "metaedge": "LFuG", "causal_type": "upregulation", "hop": 1}, # IRS1

    # High Sugar Diet downregulates insulin signaling and upregulates inflammatory TNF/IL6
    {"source_id": "Lifestyle:High_Sugar_Ultraprocessed_Diet", "relation": "DOWNREGULATES", "target_id": "Gene:3667", "metaedge": "LFdG", "causal_type": "downregulation", "hop": 1}, # IRS1
    {"source_id": "Lifestyle:High_Sugar_Ultraprocessed_Diet", "relation": "UPREGULATES", "target_id": "Gene:7124", "metaedge": "LFuG", "causal_type": "upregulation", "hop": 1}, # TNF
    {"source_id": "Lifestyle:High_Sugar_Ultraprocessed_Diet", "relation": "UPREGULATES", "target_id": "Gene:3569", "metaedge": "LFuG", "causal_type": "upregulation", "hop": 1}, # IL6

    # Chronic stress increases inflammatory cytokines (TNF, IL6)
    {"source_id": "Lifestyle:Chronic_Psychological_Stress", "relation": "UPREGULATES", "target_id": "Gene:7124", "metaedge": "LFuG", "causal_type": "upregulation", "hop": 1}, # TNF
    {"source_id": "Lifestyle:Chronic_Psychological_Stress", "relation": "UPREGULATES", "target_id": "Gene:3569", "metaedge": "LFuG", "causal_type": "upregulation", "hop": 1}, # IL6

    # --- 5. Phenotypic Symptoms Induced (INDUCES) ---
    {"source_id": "Lifestyle:High_Sugar_Ultraprocessed_Diet", "relation": "INDUCES", "target_id": "Symptom:D011596", "metaedge": "LFiS", "causal_type": "induction", "hop": 1},
    {"source_id": "Lifestyle:Sedentary_Behavior", "relation": "INDUCES", "target_id": "Symptom:D011596", "metaedge": "LFiS", "causal_type": "induction", "hop": 1},
    {"source_id": "Lifestyle:Chronic_Sleep_Deprivation", "relation": "INDUCES", "target_id": "Symptom:D003072", "metaedge": "LFiS", "causal_type": "induction", "hop": 1}, # Cognitive Disorders
]


def enrich_subgraph_with_lifestyle():
    """Merges lifestyle ontology and causal edges into targeted dataset files."""
    logger.info("Enriching Biomedical Knowledge Graph with Human Lifestyle Hierarchy...")

    nodes_file = DATA_DIR / "nodes_targeted.csv"
    edges_file = DATA_DIR / "edges_targeted.csv"
    subgraph_file = DATA_DIR / "subgraph_targeted.csv"
    summary_file = DATA_DIR / "subgraph_summary.json"

    # 1. Enrich Nodes
    df_nodes = pd.read_csv(nodes_file)
    existing_node_ids = set(df_nodes["id"])
    
    new_nodes = [n for n in LIFESTYLE_NODES if n["id"] not in existing_node_ids]
    if new_nodes:
        df_new_nodes = pd.DataFrame(new_nodes)
        df_nodes = pd.concat([df_nodes, df_new_nodes], ignore_index=True)
        df_nodes.to_csv(nodes_file, index=False)
        logger.info(f"Added {len(new_nodes)} new Lifestyle & Neurotrophic nodes to {nodes_file.name} (Total: {len(df_nodes):,}).")

    node_dict = {str(r["id"]): str(r["name"]) for _, r in df_nodes.iterrows()}
    node_type_dict = {str(r["id"]): str(r["type"]) for _, r in df_nodes.iterrows()}

    # 2. Enrich Edges
    df_edges = pd.read_csv(edges_file)
    
    # Filter valid edges where both source and target exist
    valid_lifestyle_edges = []
    for e in LIFESTYLE_EDGES:
        if e["source_id"] in node_dict and e["target_id"] in node_dict:
            valid_lifestyle_edges.append({
                "source_id": e["source_id"],
                "relation": e["relation"],
                "target_id": e["target_id"],
                "metaedge": e["metaedge"],
                "causal_type": e["causal_type"],
                "hop": e["hop"],
                "source_dataset": "Human_Lifestyle_Ontology"
            })

    # Avoid duplicate edges
    existing_edges = set(zip(df_edges["source_id"], df_edges["relation"], df_edges["target_id"]))
    unique_new_edges = [
        e for e in valid_lifestyle_edges 
        if (e["source_id"], e["relation"], e["target_id"]) not in existing_edges
    ]

    if unique_new_edges:
        df_new_edges = pd.DataFrame(unique_new_edges)
        df_edges = pd.concat([df_edges, df_new_edges], ignore_index=True)
        df_edges.to_csv(edges_file, index=False)
        logger.info(f"Added {len(unique_new_edges)} Lifestyle causal edges to {edges_file.name} (Total: {len(df_edges):,}).")

    # 3. Enrich Subgraph Targeted (denormalized)
    df_subgraph = pd.read_csv(subgraph_file)
    new_subgraph_rows = []
    for e in unique_new_edges:
        src_id = e["source_id"]
        tgt_id = e["target_id"]
        new_subgraph_rows.append({
            "source_id": src_id,
            "source_name": node_dict.get(src_id, src_id),
            "source_type": node_type_dict.get(src_id, "LifestyleFactor"),
            "relation": e["relation"],
            "target_id": tgt_id,
            "target_name": node_dict.get(tgt_id, tgt_id),
            "target_type": node_type_dict.get(tgt_id, "Unknown"),
            "metaedge": e["metaedge"],
            "causal_type": e["causal_type"],
            "hop": e["hop"],
            "source_dataset": "Human_Lifestyle_Ontology"
        })

    if new_subgraph_rows:
        df_new_subgraph = pd.DataFrame(new_subgraph_rows)
        df_subgraph = pd.concat([df_subgraph, df_new_subgraph], ignore_index=True)
        df_subgraph.to_csv(subgraph_file, index=False)
        logger.info(f"Updated {subgraph_file.name} with {len(new_subgraph_rows)} rows (Total: {len(df_subgraph):,}).")

    # 4. Update Summary JSON
    if summary_file.exists():
        with open(summary_file, "r") as f:
            summary = json.load(f)
        summary["total_nodes"] = len(df_nodes)
        summary["total_edges"] = len(df_edges)
        summary["lifestyle_nodes_count"] = len([n for n in df_nodes["type"] if "Lifestyle" in n])
        summary["lifestyle_edges_count"] = len(df_edges[df_edges["source_dataset"] == "Human_Lifestyle_Ontology"])
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Updated subgraph_summary.json with lifestyle metrics.")

    print("\n" + "=" * 70)
    print("ENRICHISSEMENT 'HUMAN LIFESTYLE' RÉUSSI")
    print(f"  • Nouveaux nœuds ajoutés : {len(new_nodes)}")
    print(f"  • Nouvelles arêtes causales ajoutées : {len(unique_new_edges)}")
    print(f"  • Total Nœuds dans le Graphe : {len(df_nodes):,}")
    print(f"  • Total Relations dans le Graphe : {len(df_edges):,}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    enrich_subgraph_with_lifestyle()
