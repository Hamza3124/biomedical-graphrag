"""
Biomedical Entity Linker for GraphRAG.
PFA (5IADA - EMSI) - Diseases Cause Oriented Analysis using RAG and Knowledge Graphs.

Identifies biomedical entities in user clinical queries (Diseases, Genes, Pathways,
Compounds, Symptoms) and links them to canonical Knowledge Graph nodes.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger("RAG.EntityLinker")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# Common biomedical synonyms and acronyms mapping to canonical names
BIOMEDICAL_ALIASES: Dict[str, str] = {
    # Alzheimer's Disease & Neurodegeneration
    "ad": "Alzheimer's disease",
    "alzheimer": "Alzheimer's disease",
    "alzheimers": "Alzheimer's disease",
    "alzheimer's": "Alzheimer's disease",
    "alzheimer disease": "Alzheimer's disease",
    "alzheimer's disease": "Alzheimer's disease",
    "dementia": "Dementia",
    "memory loss": "Memory Disorders",
    "memory disorder": "Memory Disorders",
    "memory disorders": "Memory Disorders",
    "cognitive decline": "Cognitive Disorders",
    "cognitive impairment": "Cognitive Disorders",
    "agnosia": "Agnosia",
    "aphasia": "Aphasia",
    "apraxia": "Apraxia",
    "confusion": "Confusion",
    # Key Alzheimer Genes
    "amyloid precursor protein": "APP",
    "amyloid-beta": "APP",
    "amyloid beta": "APP",
    "beta amyloid": "APP",
    "presenilin 1": "PSEN1",
    "presenilin-1": "PSEN1",
    "presenilin 2": "PSEN2",
    "presenilin-2": "PSEN2",
    "apolipoprotein e": "APOE",
    "apoe4": "APOE",
    "apoe-e4": "APOE",
    "tau": "MAPT",
    "microtubule-associated protein tau": "MAPT",
    "hyperphosphorylated tau": "MAPT",
    # Type 2 Diabetes & Metabolism
    "t2d": "Type 2 diabetes mellitus",
    "t2dm": "Type 2 diabetes mellitus",
    "type 2 diabetes": "Type 2 diabetes mellitus",
    "type ii diabetes": "Type 2 diabetes mellitus",
    "diabetes": "Type 2 diabetes mellitus",
    "diabetes mellitus": "Type 2 diabetes mellitus",
    "hyperglycemia": "Hyperglycemia",
    "insulin resistance": "Type 2 diabetes mellitus",
    # Cross-Disease Metabolic Genes
    "insulin": "INS",
    "insulin receptor": "INSR",
    "insulin receptor substrate 1": "IRS1",
    "akt": "AKT1",
    "protein kinase b": "AKT1",
    "ppar gamma": "PPARG",
    "pparg": "PPARG",
    "gsk3b": "GSK3B",
    "glycogen synthase kinase 3 beta": "GSK3B",
    # Therapeutics & Compounds
    "donepezil": "Donepezil",
    "aricept": "Donepezil",
    "memantine": "Memantine",
    "namenda": "Memantine",
    "galantamine": "Galantamine",
    "reminyl": "Galantamine",
    "rivastigmine": "Rivastigmine",
    "exelon": "Rivastigmine",
    "metformin": "Metformin",
    # Human Lifestyle & Environmental Factors
    "lifestyle": "Human Lifestyle",
    "human lifestyle": "Human Lifestyle",
    "diet": "Nutrition and Dietary Habits",
    "nutrition": "Nutrition and Dietary Habits",
    "mediterranean diet": "Mediterranean and DASH Diet",
    "dash diet": "Mediterranean and DASH Diet",
    "high sugar diet": "High Glycemic and Ultra-Processed Diet",
    "sugar": "High Glycemic and Ultra-Processed Diet",
    "western diet": "High Glycemic and Ultra-Processed Diet",
    "exercise": "Regular Aerobic Physical Exercise",
    "aerobic exercise": "Regular Aerobic Physical Exercise",
    "physical activity": "Regular Aerobic Physical Exercise",
    "cardio": "Regular Aerobic Physical Exercise",
    "sedentary": "Sedentary Behavior and Inactivity",
    "sedentary lifestyle": "Sedentary Behavior and Inactivity",
    "inactivity": "Sedentary Behavior and Inactivity",
    "sleep": "Sleep and Circadian Rhythm",
    "sleep deprivation": "Chronic Sleep Deprivation and Sleep Apnea",
    "lack of sleep": "Chronic Sleep Deprivation and Sleep Apnea",
    "sleep apnea": "Chronic Sleep Deprivation and Sleep Apnea",
    "smoking": "Tobacco Smoking Habit",
    "tobacco": "Tobacco Smoking Habit",
    "cigarette": "Tobacco Smoking Habit",
    "alcohol": "Chronic Excessive Alcohol Intake",
    "drinking": "Chronic Excessive Alcohol Intake",
    "stress": "Chronic Psychological Stress and Hypercortisolemia",
    "psychological stress": "Chronic Psychological Stress and Hypercortisolemia",
    "cortisol": "Chronic Psychological Stress and Hypercortisolemia",
    "cognitive reserve": "Cognitive Reserve and Intellectual Engagement",
    "cognitive stimulation": "Cognitive Reserve and Intellectual Engagement",
    "bdnf": "BDNF",
}


class BiomedicalEntityLinker:
    """
    Identifies biomedical entities in clinical questions and links them to
    canonical nodes in the Knowledge Graph.
    """

    def __init__(self, nodes_csv_path: Optional[Path] = None):
        self.nodes_csv_path = nodes_csv_path or (
            Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "nodes_targeted.csv"
        )
        self.exact_name_index: Dict[str, Dict[str, Any]] = {}
        self.gene_symbol_index: Dict[str, Dict[str, Any]] = {}
        self.id_index: Dict[str, Dict[str, Any]] = {}
        self.disease_nodes: List[Dict[str, Any]] = []

        self._load_and_index()

    def _load_and_index(self) -> None:
        """Loads and indexes targeted nodes for sub-millisecond retrieval."""
        if not self.nodes_csv_path.exists():
            raise FileNotFoundError(f"Nodes file not found at {self.nodes_csv_path}")

        df = pd.read_csv(self.nodes_csv_path)
        logger.info(f"Indexing {len(df):,} biomedical nodes from {self.nodes_csv_path.name}...")

        for _, row in df.iterrows():
            node = {
                "id": str(row["id"]),
                "name": str(row["name"]),
                "type": str(row["type"]),
                "identifier_type": str(row.get("identifier_type", "")),
                "external_id": str(row.get("external_id", "")),
                "umls_cui": str(row.get("umls_cui", "")) if pd.notna(row.get("umls_cui")) else None,
                "mesh_id": str(row.get("mesh_id", "")) if pd.notna(row.get("mesh_id")) else None,
                "doid": str(row.get("doid", "")) if pd.notna(row.get("doid")) else None,
            }

            self.id_index[node["id"]] = node

            # Index by normalized name
            norm_name = node["name"].strip().lower()
            self.exact_name_index[norm_name] = node

            # Dedicated index for Gene symbols
            if node["type"] == "Gene":
                self.gene_symbol_index[node["name"].strip().upper()] = node
            elif node["type"] == "Disease":
                self.disease_nodes.append(node)

        logger.info(
            f"Entity Linker ready: {len(self.exact_name_index):,} indexed names, "
            f"{len(self.gene_symbol_index):,} gene symbols."
        )

    def link_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extracts and links all biomedical entities found in the query text.
        Handles multi-word entities, acronyms, and gene symbols.
        """
        text_lower = text.lower()
        linked_entities: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        # Step 1: Check curated biomedical aliases (highest precision)
        # Sort by length descending to match longest spans first
        for alias, canonical_name in sorted(BIOMEDICAL_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
            # Regex with word boundaries
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, text_lower):
                target_norm = canonical_name.strip().lower()
                node = self.exact_name_index.get(target_norm)
                if node and node["id"] not in seen_ids:
                    seen_ids.add(node["id"])
                    linked_entities.append({
                        **node,
                        "matched_text": alias,
                        "confidence": 1.0,
                        "source": "curated_alias",
                    })

        # Step 2: Search for Gene symbols (uppercase word tokens like APP, APOE, AKT1, PSEN1)
        tokens = re.findall(r"\b[A-Za-z0-9_-]{2,10}\b", text)
        for token in tokens:
            upper_token = token.upper()
            if upper_token in self.gene_symbol_index:
                node = self.gene_symbol_index[upper_token]
                if node["id"] not in seen_ids:
                    seen_ids.add(node["id"])
                    linked_entities.append({
                        **node,
                        "matched_text": token,
                        "confidence": 0.95,
                        "source": "gene_symbol",
                    })

        # Step 3: Exact & Substring match against full node names
        # (Compounds, Symptoms, Anatomy, Pathways)
        for norm_name, node in self.exact_name_index.items():
            if len(norm_name) < 4:
                continue
            if node["id"] in seen_ids:
                continue

            pattern = r"\b" + re.escape(norm_name) + r"\b"
            if re.search(pattern, text_lower):
                seen_ids.add(node["id"])
                linked_entities.append({
                    **node,
                    "matched_text": norm_name,
                    "confidence": 0.90,
                    "source": "exact_name",
                })

        logger.debug(f"Linked {len(linked_entities)} entities for query: '{text[:50]}...'")
        return linked_entities

    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a node by its unique identifier."""
        return self.id_index.get(node_id)

