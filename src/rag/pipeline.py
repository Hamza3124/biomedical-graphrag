"""
GraphRAG Pipeline Orchestrator.
PFA (5IADA - EMSI) - Diseases Cause Oriented Analysis using RAG and Knowledge Graphs.

Orchestrates entity linking, hybrid retrieval (ChromaDB + Graph),
context fusion, and causal report generation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.rag.hybrid_retriever import HybridRetriever, RetrievalResult
from src.rag.context_fusion import ContextFusionEngine
from src.rag.causal_generator import CausalGenerator, CausalAnalysisReport

logger = logging.getLogger("RAG.Pipeline")


class GraphRAGPipeline:
    """
    Main entry point for biomedical causal analysis via GraphRAG.
    """

    def __init__(self, use_neo4j: bool = False):
        logger.info("Initializing GraphRAG Pipeline...")
        self.retriever = HybridRetriever(use_neo4j=use_neo4j)
        self.fusion = ContextFusionEngine()
        self.generator = CausalGenerator()
        logger.info("GraphRAG Pipeline ready.")

    def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k_chunks: int = 4,
        max_graph_facts: int = 15,
    ) -> CausalAnalysisReport:
        """
        Executes causal inquiry under the requested mode:
        - 'hybrid' : Full GraphRAG (Knowledge Graph + Semantic Vector Search)
        - 'vector_only' : Standard RAG baseline (MedQuAD vectors only)
        - 'graph_only' : Structured Knowledge Graph traversal only
        """
        if mode not in ("hybrid", "vector_only", "graph_only"):
            raise ValueError(f"Invalid mode '{mode}'. Choose from: 'hybrid', 'vector_only', 'graph_only'.")

        logger.info(f"Processing query in [{mode.upper()}] mode: '{question[:60]}...'")

        # 1. Hybrid Retrieval
        retrieval_result = self.retriever.retrieve(
            query=question,
            top_k_chunks=top_k_chunks if mode != "graph_only" else 0,
            max_graph_facts=max_graph_facts if mode != "vector_only" else 0,
        )

        # 2. Causal Generation
        report = self.generator.generate(retrieval_result, mode=mode)
        return report

