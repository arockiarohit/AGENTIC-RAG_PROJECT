"""
Reranker Service.
Re-scores retrieved documents using a cross-encoder model for improved ranking.
"""

import logging
import time
from typing import Optional

from app.config import RERANKER_ENABLED, RERANKER_MODEL, RERANKER_TOP_K

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Re-ranks retrieved documents using a cross-encoder model.

    Uses sentence-transformers CrossEncoder for scoring query-document pairs.
    Gracefully falls back to returning results unchanged if the model
    is unavailable or reranking is disabled.
    """

    def __init__(self) -> None:
        """Initialize the reranker (model is lazily loaded)."""
        self._model = None
        self._model_loaded: bool = False
        self._load_failed: bool = False

    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Re-rank retrieved results using the cross-encoder model.

        Args:
            query: The search query.
            results: List of result dicts with 'content' key.
            top_k: Number of top results to return after reranking.

        Returns:
            Re-ranked list of result dicts. Returns original results
            if reranking is disabled or fails.
        """
        if not RERANKER_ENABLED:
            logger.debug("[RERANKER] Reranking disabled, passing through.")
            return results

        if not results:
            return results

        if self._load_failed:
            logger.debug("[RERANKER] Model load previously failed, skipping.")
            return results

        try:
            start_time: float = time.time()

            model = self._get_model()
            if model is None:
                return results

            # Prepare query-document pairs
            pairs: list[list[str]] = [
                [query, result.get("content", "")]
                for result in results
            ]

            # Score all pairs
            scores = model.predict(pairs)

            # Attach reranker scores and sort
            for i, result in enumerate(results):
                result["reranker_score"] = float(scores[i])

            reranked: list[dict] = sorted(
                results,
                key=lambda x: x.get("reranker_score", 0),
                reverse=True,
            )

            # Apply top-k limit
            effective_top_k: int = top_k or RERANKER_TOP_K
            reranked = reranked[:effective_top_k]

            elapsed: float = time.time() - start_time
            logger.info(
                "[RERANKER] Reranked %d results to top-%d in %.2f seconds",
                len(results), effective_top_k, elapsed
            )

            return reranked

        except Exception as e:
            logger.error("[RERANKER] Reranking failed: %s", str(e))
            return results

    def _get_model(self):
        """
        Lazily load the cross-encoder model.

        Returns:
            The CrossEncoder model, or None if loading fails.
        """
        if self._model_loaded:
            return self._model

        try:
            from sentence_transformers import CrossEncoder

            logger.info(
                "[RERANKER] Loading cross-encoder model: %s", RERANKER_MODEL
            )

            self._model = CrossEncoder(RERANKER_MODEL)
            self._model_loaded = True

            logger.info("[RERANKER] Cross-encoder model loaded successfully.")
            return self._model

        except ImportError:
            logger.warning(
                "[RERANKER] sentence-transformers CrossEncoder not available. "
                "Reranking disabled."
            )
            self._load_failed = True
            return None
        except Exception as e:
            logger.error(
                "[RERANKER] Failed to load cross-encoder model: %s", str(e)
            )
            self._load_failed = True
            return None


# --------------------------------------------------------------------------
# Global singleton instance
# --------------------------------------------------------------------------
reranker: RerankerService = RerankerService()
