"""
Hybrid Retriever Service.
Combines FAISS semantic search with BM25 lexical search for improved retrieval.
"""

import logging
import time
import hashlib
from typing import Optional

from app.config import TOP_K, BM25_TOP_K, HYBRID_SEARCH_ENABLED

logger = logging.getLogger(__name__)


class HybridRetrieverService:
    """
    Hybrid retrieval combining FAISS (semantic) and BM25 (lexical) search.

    FAISS provides dense vector similarity search.
    BM25 provides sparse keyword-based search.
    Results are merged and deduplicated.
    """

    def __init__(self) -> None:
        """Initialize the retriever."""
        self._bm25_index = None
        self._bm25_corpus: list[list[str]] = []
        self._bm25_documents: list[dict] = []

    def retrieve(self, query: str) -> list[dict]:
        """
        Retrieve documents using hybrid search (FAISS + BM25).

        Falls back to FAISS-only if hybrid search is disabled or BM25 fails.

        Args:
            query: The search query.

        Returns:
            List of result dicts with 'content', 'metadata', and 'score' keys.
        """
        start_time: float = time.time()

        # FAISS retrieval (always active)
        faiss_results: list[dict] = self._faiss_search(query)

        if not HYBRID_SEARCH_ENABLED:
            elapsed: float = time.time() - start_time
            logger.info(
                "[RETRIEVER] FAISS-only retrieval: %d results in %.2f seconds",
                len(faiss_results), elapsed
            )
            return faiss_results

        # BM25 retrieval
        bm25_results: list[dict] = self._bm25_search(query)

        # Merge and deduplicate
        merged: list[dict] = self._merge_results(faiss_results, bm25_results)

        elapsed = time.time() - start_time
        logger.info(
            "[RETRIEVER] Hybrid retrieval: %d FAISS + %d BM25 = %d merged "
            "in %.2f seconds",
            len(faiss_results), len(bm25_results), len(merged), elapsed
        )

        return merged

    def _faiss_search(self, query: str) -> list[dict]:
        """
        Perform FAISS vector similarity search.

        Args:
            query: The search query.

        Returns:
            List of result dicts from FAISS.
        """
        try:
            from app.vector_store import vector_store
            results: list[dict] = vector_store.search(query, top_k=TOP_K)
            return results
        except Exception as e:
            logger.error("[RETRIEVER] FAISS search failed: %s", str(e))
            return []

    def _bm25_search(self, query: str) -> list[dict]:
        """
        Perform BM25 lexical search.

        Lazily builds the BM25 index from the vector store's documents.

        Args:
            query: The search query.

        Returns:
            List of result dicts from BM25.
        """
        try:
            self._ensure_bm25_index()

            if self._bm25_index is None:
                return []

            # Tokenize query
            query_tokens: list[str] = query.lower().split()

            # Get BM25 scores
            scores = self._bm25_index.get_scores(query_tokens)

            # Get top-k indices
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:BM25_TOP_K]

            results: list[dict] = []
            for idx in top_indices:
                if scores[idx] > 0 and idx < len(self._bm25_documents):
                    doc = self._bm25_documents[idx].copy()
                    doc["score"] = float(scores[idx])
                    doc["retrieval_method"] = "bm25"
                    results.append(doc)

            return results

        except Exception as e:
            logger.error("[RETRIEVER] BM25 search failed: %s", str(e))
            return []

    def _ensure_bm25_index(self) -> None:
        """
        Build or rebuild the BM25 index from the vector store's documents.
        Only rebuilds if the document count has changed.
        """
        try:
            from app.vector_store import vector_store
            from rank_bm25 import BM25Okapi

            current_docs: list[dict] = vector_store.documents

            # Skip if already built and document count matches
            if (self._bm25_index is not None
                    and len(self._bm25_documents) == len(current_docs)):
                return

            if not current_docs:
                logger.warning("[RETRIEVER] No documents for BM25 index.")
                return

            # Build corpus (tokenized)
            self._bm25_corpus = [
                doc.get("content", "").lower().split()
                for doc in current_docs
            ]

            self._bm25_documents = list(current_docs)
            self._bm25_index = BM25Okapi(self._bm25_corpus)

            logger.info(
                "[RETRIEVER] BM25 index built with %d documents.",
                len(current_docs)
            )

        except ImportError:
            logger.warning(
                "[RETRIEVER] rank-bm25 not installed. "
                "BM25 search disabled. Install with: pip install rank-bm25"
            )
            self._bm25_index = None
        except Exception as e:
            logger.error(
                "[RETRIEVER] Failed to build BM25 index: %s", str(e)
            )
            self._bm25_index = None

    @staticmethod
    def _merge_results(
        faiss_results: list[dict],
        bm25_results: list[dict],
    ) -> list[dict]:
        """
        Merge and deduplicate results from FAISS and BM25.

        Uses content hash for deduplication. FAISS results take priority
        for duplicates since they have calibrated similarity scores.

        Args:
            faiss_results: Results from FAISS search.
            bm25_results: Results from BM25 search.

        Returns:
            Merged and deduplicated list of result dicts.
        """
        seen_hashes: set[str] = set()
        merged: list[dict] = []

        # Add FAISS results first (higher quality scores)
        for result in faiss_results:
            content: str = result.get("content", "")
            content_hash: str = hashlib.md5(
                content.encode()
            ).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                result["retrieval_method"] = "faiss"
                merged.append(result)

        # Add BM25 results (skip duplicates)
        for result in bm25_results:
            content = result.get("content", "")
            content_hash = hashlib.md5(
                content.encode()
            ).hexdigest()

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                merged.append(result)

        return merged

    def invalidate_bm25_cache(self) -> None:
        """
        Force rebuild of the BM25 index on next search.
        Call this after documents are added or removed.
        """
        self._bm25_index = None
        self._bm25_corpus = []
        self._bm25_documents = []
        logger.info("[RETRIEVER] BM25 cache invalidated.")


# --------------------------------------------------------------------------
# Global singleton instance
# --------------------------------------------------------------------------
hybrid_retriever: HybridRetrieverService = HybridRetrieverService()
