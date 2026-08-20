"""Re-ranker for search results using cross-encoder models.

Uses FlagEmbedding's BGE-Reranker-v2-m3 to re-score retrieved documents.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder re-ranker for improving retrieval precision.

    Wraps FlagEmbedding's BGE-Reranker to re-score retrieved passages
    against the query, producing a more accurate ranking than embedding
    similarity alone.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        self.model_name = model_name
        self._model = None
        self._initialized = False

    def _ensure_model(self) -> None:
        """Lazy-load the reranker model."""
        if self._initialized:
            return
        try:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(self.model_name, use_fp16=True)
            self._initialized = True
            logger.info(f"Reranker loaded: {self.model_name}")
        except ImportError:
            logger.warning(
                "FlagEmbedding not installed. Reranking disabled. "
                "Install with: pip install FlagEmbedding"
            )
            self._initialized = True  # Mark as initialized to avoid repeated warnings

    def rerank(
        self,
        query: str,
        results: List[dict],
        top_k: Optional[int] = None,
    ) -> List[dict]:
        """Re-rank search results using cross-encoder scores.

        Args:
            query: Original search query.
            results: List of result dicts with 'text' key.
            top_k: Number of top results to return (default: all).

        Returns:
            Re-ranked list with updated 'score' and added 'rerank_score'.
        """
        if not results or self._model is None:
            return results

        self._ensure_model()

        if self._model is None:
            return results

        pairs = [[query, r["text"]] for r in results]
        scores = self._model.compute_score(pairs, normalize=True)

        # Handle single result (returns float, not list)
        if isinstance(scores, float):
            scores = [scores]

        for i, score in enumerate(scores):
            results[i]["rerank_score"] = round(score, 4)

        results.sort(key=lambda r: r.get("rerank_score", 0.0), reverse=True)

        if top_k:
            results = results[:top_k]

        return results
