# src/hybrid_retriever.py
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from .config import (
    EMBEDDING_MODEL_NAME,
    ALPHA_DENSE,
    BETA_SPARSE,
    TOP_K_BM25,
    TOP_K_DENSE,
    TOP_K_HYBRID,
)


class HybridRetriever:
    """
    Simple hybrid retriever over regulation chunks.

    - BM25 (sparse)
    - Dense embeddings (SentenceTransformers)
    """

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.texts = [c["text"] for c in chunks]

        # Prepare BM25
        tokenized_corpus = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Prepare dense model + embeddings
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embeddings = self.model.encode(self.texts, convert_to_numpy=True)

    def _bm25_scores(self, query: str) -> np.ndarray:
        scores = np.array(self.bm25.get_scores(query.split()), dtype=float)
        # normalize to [0, 1]
        if scores.max() > 0:
            scores = scores / scores.max()
        return scores

    def _dense_scores(self, query: str) -> np.ndarray:
        query_emb = self.model.encode(query, convert_to_numpy=True)
        # cosine similarity
        dot = np.dot(self.embeddings, query_emb)
        norm_docs = np.linalg.norm(self.embeddings, axis=1)
        norm_query = np.linalg.norm(query_emb) + 1e-8
        sims = dot / (norm_docs * norm_query)
        # map from [-1,1] to [0,1]
        sims = (sims + 1.0) / 2.0
        return sims

    def retrieve(self, query: str, top_k: int = TOP_K_HYBRID) -> List[Tuple[Dict, float]]:
        """
        Return top_k chunks with combined hybrid scores.
        """
        sparse = self._bm25_scores(query)
        dense = self._dense_scores(query)

        combined = ALPHA_DENSE * dense + BETA_SPARSE * sparse

        # get top indices
        top_indices = np.argsort(combined)[::-1][:top_k]
        results: List[Tuple[Dict, float]] = []

        for idx in top_indices:
            results.append((self.chunks[idx], float(combined[idx])))

        return results
