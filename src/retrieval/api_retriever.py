from __future__ import annotations

from typing import Any, Protocol

from qdrant_client import QdrantClient

from src.config import settings


class QueryEmbedder(Protocol):
    def encode_query(self, query: str) -> list[float]:
        """Embed one question in the same space as the indexed chunks."""


class ApiRetriever:
    """Dense search over a collection built by an API embedding model."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        collection_name: str,
        qdrant_path: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.collection_name = collection_name
        self.client = QdrantClient(path=qdrant_path or settings.qdrant_path)

    def close(self) -> None:
        self.client.close()

    def search(
        self,
        query: str,
        top_k: int = 5,
        exclude_noise: bool = False,
    ) -> list[dict[str, Any]]:
        del exclude_noise
        query_vector = self.embedder.encode_query(query)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
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
