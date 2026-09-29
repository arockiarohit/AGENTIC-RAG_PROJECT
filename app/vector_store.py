"""
FAISS Vector Store module.
Manages the FAISS index for document storage and similarity search.
"""

import os
import json
import faiss
import numpy as np
from app.config import FAISS_INDEX_DIR, TOP_K
from app.embeddings import EmbeddingService


class VectorStore:
    """Manages FAISS index and document metadata."""

    def __init__(self):
        self.index = None
        self.documents = []  # Stores {content, metadata} for each vector
        self.index_path = os.path.join(FAISS_INDEX_DIR, "index.faiss")
        self.docs_path = os.path.join(FAISS_INDEX_DIR, "documents.json")
        self._load_existing()

    def _load_existing(self):
        """Load existing FAISS index and documents from disk if available."""
        if os.path.exists(self.index_path) and os.path.exists(self.docs_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.docs_path, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
                print(f"Loaded existing index with {self.index.ntotal} vectors.")
            except Exception as e:
                print(f"Error loading existing index: {e}. Starting fresh.")
                self.index = None
                self.documents = []

    def add_documents(self, chunks: list[dict]):
        """
        Add document chunks to the FAISS index.
        
        Args:
            chunks: List of dicts with 'content' and 'metadata' keys.
        """
        texts = [chunk["content"] for chunk in chunks]
        embeddings = EmbeddingService.embed_texts(texts)

        dimension = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vectors)

        self.index.add(embeddings)
        self.documents.extend(chunks)

        # Persist to disk
        self._save()
        print(f"Added {len(chunks)} chunks. Total vectors: {self.index.ntotal}")

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Search for the most similar documents to the query.
        
        Args:
            query: The search query.
            top_k: Number of results to return.
        
        Returns:
            List of dicts with 'content', 'metadata', and 'score' keys.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = EmbeddingService.embed_query(query)
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents) and idx >= 0:
                result = self.documents[idx].copy()
                result["score"] = float(score)
                results.append(result)

        return results

    def _save(self):
        """Persist FAISS index and documents to disk."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.docs_path, "w", encoding="utf-8") as f:
                json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def get_stats(self) -> dict:
        """Return current index statistics."""
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "total_documents": len(self.documents),
            "dimension": self.index.d if self.index else 0
        }

    def clear(self):
        """Clear all data from the index."""
        self.index = None
        self.documents = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.docs_path):
            os.remove(self.docs_path)
        print("Vector store cleared.")


# Global instance
vector_store = VectorStore()
