# src/medcompliance_agent/hybrid_retriever.py

from typing import List, Dict, Tuple, Any
from rank_bm25 import BM25Okapi
import numpy as np

from .vector_store import QdrantStore
from .config import (
    ALPHA_DENSE,
    BETA_SPARSE,
    TOP_K_BM25,
    TOP_K_DENSE,
    TOP_K_HYBRID,
)


class HybridRetriever:
    """
    Hybrid retriever combining:
    - Qdrant dense vector search
    - BM25 sparse search
    """

    def __init__(self, qdrant: QdrantStore, chunks: List[Dict], embed_fn):
        """
        qdrant   : QdrantStore instance
        chunks   : raw text chunks (used only for sparse BM25)
        embed_fn : function that produces embeddings for texts
        """
        self.qdrant = qdrant
        self.embed_fn = embed_fn

        # BM25 setup
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]
        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def _sparse_search(self, query: str, top_k: int = TOP_K_BM25):
        scores = self.bm25.get_scores(query.split())

        # Normalize BM25 → [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()

        # Pick top-k
        idxs = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in idxs:
            results.append({
                "text": self.texts[idx],
                "metadata": self.chunks[idx],
                "score": float(scores[idx]),
            })

        return results

    def _dense_search(self, query: str, top_k: int = TOP_K_DENSE):
        """
        Calls Qdrant vector search.
        """
        dense_results = self.qdrant.search(query, self.embed_fn, top_k=top_k)

        # Already returns list of:
        # {"text": ..., "metadata": ..., "score": ...}
        return dense_results

    def retrieve(self, query: str, top_k: int = TOP_K_HYBRID):
        """
        Hybrid scoring:
            hybrid_score = ALPHA * dense + BETA * sparse
        """
        dense_hits = self._dense_search(query)
        sparse_hits = self._sparse_search(query)

        combined = {}

        # Add dense hits
        for d in dense_hits:
            key = d["text"]
            combined[key] = {
                "text": d["text"],
                "metadata": d["metadata"],
                "dense": d["score"],
                "sparse": 0.0
            }

        # Add sparse hits
        for s in sparse_hits:
            key = s["text"]
            if key not in combined:
                combined[key] = {
                    "text": s["text"],
                    "metadata": s["metadata"],
                    "dense": 0.0,
                    "sparse": s["score"]
                }
            else:
                combined[key]["sparse"] = s["score"]

        # Compute hybrid score
        final = []
        for item in combined.values():
            hybrid_score = ALPHA_DENSE * item["dense"] + BETA_SPARSE * item["sparse"]
            final.append((item, hybrid_score))

        # Sort by hybrid score
        final = sorted(final, key=lambda x: x[1], reverse=True)

        # Return top-k chunks
        results = []
        for item, score in final[:top_k]:
            results.append(({"text": item["text"], "metadata": item["metadata"]}, score))

        return results
