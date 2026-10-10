from __future__ import annotations

from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.retrieval.noise import NOISE_LABELS

# Required for BGE retrieval. Passages are embedded without this prefix.
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    def __init__(
        self,
        qdrant_path: str | None = None,
        collection_name: str | None = None,
        model_name: str | None = None,
    ):
        self.client = QdrantClient(
            path=qdrant_path or settings.qdrant_path
        )
        self.collection_name = (
            collection_name or settings.qdrant_collection
        )
        self.model = SentenceTransformer(
            model_name or settings.embedding_model
        )

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a question with the model's query prompt.

        Documents were indexed with encode_document(). Queries use
        encode_query() so BGE applies its retrieval instruction.
        """
        query_vector = self.model.encode_query(
            query,
            prompt=QUERY_PROMPT,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        vector = np.asarray(query_vector, dtype=float)
        if vector.ndim == 2:
            vector = vector[0]

        return vector.tolist()

    def _noise_filter(self) -> models.Filter:
        return models.Filter(
            must_not=[
                models.FieldCondition(
                    key="noise",
                    match=models.MatchAny(any=list(NOISE_LABELS)),
                )
            ]
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        exclude_noise: bool = False,
    ) -> list[dict[str, Any]]:
        query_vector = self.embed_query(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._noise_filter() if exclude_noise else None,
            limit=top_k,
            with_payload=True,
        ).points

        retrieved = []

        for rank, point in enumerate(results, start=1):
            payload = point.payload or {}

            retrieved.append(
                {
                    "rank": rank,
                    "score": float(point.score),
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "page_start": payload.get("page_start"),
                    "page_end": payload.get("page_end"),
                    "section": payload.get("section"),
                    "text": payload.get("text", ""),
                }
            )

        return retrieved
