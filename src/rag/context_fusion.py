"""
Context Fusion Engine for GraphRAG.
PFA (5IADA - EMSI) - Diseases Cause Oriented Analysis using RAG and Knowledge Graphs.

Merges structured causal paths from the Knowledge Graph with unstructured
clinical QA evidence from MedQuAD into a coherent multi-scale context.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.rag.hybrid_retriever import RetrievalResult

logger = logging.getLogger("RAG.ContextFusion")


class ContextFusionEngine:
    """
    Combines structured graph causal paths and unstructured text passages
    into a standardized clinical prompt for causal reasoning.
    """

    @staticmethod
    def build_graph_context(result: RetrievalResult) -> str:
        """Formats graph entities, 1-hop facts, and multi-hop paths."""
        lines: List[str] = []

        # 1. Linked Entities Summary
        lines.append("### A. Identified Biomedical Entities")
        for ent in result.linked_entities:
            lines.append(f"- **{ent['type']}**: {ent['name']} (ID: `{ent['id']}`)")

        # 2. Multi-hop Causal Cascades
        if result.multihop_chains:
            lines.append("\n### B. Multi-Scale Causal Chains (Graph Traversal)")
            for chain in result.multihop_chains:
                lines.append(f"- {chain}")

        # 3. Direct 1-hop Causal Facts
        if result.causal_facts:
            lines.append("\n### C. Direct Biomedical Relationships")
            for f in result.causal_facts:
                lines.append(f"- {f.to_readable_triple()}")

        return "\n".join(lines)

    @staticmethod
    def build_text_context(result: RetrievalResult) -> str:
        """Formats unstructured MedQuAD question-answer passages."""
        lines: List[str] = []
        lines.append("### Certified Clinical & Biomedical Literature (NIH MedQuAD)")

        for i, chunk in enumerate(result.text_chunks, start=1):
            lines.append(
                f"\n**[Evidence {i}]** (Source: `{chunk.source}` | DocID: `{chunk.doc_id}` | QType: `{chunk.question_type}` | Cosine Score: `{chunk.score:.4f}`)"
            )
            lines.append(f"**Medical Query**: {chunk.question}")
            lines.append(f"**Clinical Excerpt**: {chunk.text.strip()}\n")

        return "\n".join(lines)

    def fuse_contexts(self, result: RetrievalResult) -> Dict[str, str]:
        """Produces full structured context blocks for the reasoning generator."""
        graph_ctx = self.build_graph_context(result)
        text_ctx = self.build_text_context(result)

        full_prompt_context = f"""
================================================================================
KNOWLEDGE GRAPH EVIDENCE (STRUCTURED CAUSAL PATHS & GENE-PATHWAY INTERACTIONS)
================================================================================
{graph_ctx}

================================================================================
LITERATURE EVIDENCE (UNSTRUCTURED NIH MEDQUAD CLINICAL PASSAGES)
================================================================================
{text_ctx}
""".strip()

        return {
            "graph_context": graph_ctx,
            "text_context": text_ctx,
            "fused_context": full_prompt_context,
        }
