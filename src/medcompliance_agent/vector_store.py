from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)
import uuid
from typing import List, Dict, Any

class QdrantStore:
    def __init__(self, collection_name="medregulations", embedding_dim=384):
        self.client = QdrantClient("http://localhost:6333")
        self.collection_name = collection_name

        # Create collection if not exists
        if collection_name not in [c.name for c in self.client.get_collections().collections]:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )

    def add_chunks(self, chunks: List[Dict[str, Any]], embed_fn):
        """
        chunks: [
          { "id", "text", "section_type", "section_title", ... }
        ]
        embed_fn: function returning vector for given text
        """

        points = []
        for c in chunks:
            vector = embed_fn(c["text"])
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=c  # store metadata
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def _extract_score(self, r):
        """
        Handles all possible Qdrant return types:
        - ScoredPoint
        - (id, score, payload)
        - {"score": ...}
        """
        # Case 1 — ScoredPoint
        if hasattr(r, "score"):
            return float(r.score)

        # Case 2 — tuple format: (id, score, payload)
        if isinstance(r, tuple) and len(r) >= 2:
            if isinstance(r[1], (int, float)):
                return float(r[1])
            try:
                return float(r[1])
            except:
                pass

        # Case 3 — dict-like: {"score": ...}
        if isinstance(r, dict) and "score" in r:
            return float(r["score"])

        raise ValueError(f"Qdrant returned unknown score format: {r}")

    def search(self, query: str, embed_fn, top_k=5, filters=None):
        qvec = embed_fn(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=qvec,
            query_filter=filters,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )

        docs = []

        # Some Qdrant versions return objects, some return tuples → handle both.
        for r in results:

            # --- Case 1: r is tuple ---
            if isinstance(r, tuple):
                # (id, version, score, payload, vector)
                # Real structure varies; safest is detecting dict
                payload = None
                score = None

                for item in r:
                    if isinstance(item, dict):
                        payload = item
                    if isinstance(item, float):
                        score = item

                if payload is None:
                    payload = {}

                docs.append({
                    "text": payload.get("text", ""),
                    "metadata": payload,
                    "score": float(score) if score is not None else 0.0
                })

            # --- Case 2: r is ScoredPoint object ---
            else:
                payload = getattr(r, "payload", {}) or {}
                docs.append({
                    "text": payload.get("text", ""),
                    "metadata": payload,
                    "score": float(getattr(r, "score", 0.0))
                })

        return docs
