"""Search helpers for sentence-window and parent-child retrieval.

Both methods embed a small unit and return a larger one. The returned list
is deduped so Hit@10 counts 10 passages, not 10 overlapping hits.
"""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from src.config import settings
from src.retrieval.api_retriever import QueryEmbedder

CANDIDATE_POOL = 80


class StructuredRetriever:
    """Dense search that can expand a hit to its window or its parent."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        collection_name: str,
        mode: str,
        qdrant_path: str | None = None,
        query_cache: dict[str, list[float]] | None = None,
    ) -> None:
        if mode not in {"plain", "window", "parent"}:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        self.embedder = embedder
        self.collection_name = collection_name
        self.mode = mode
        self.query_cache = query_cache if query_cache is not None else {}
        self.client = QdrantClient(path=qdrant_path or settings.qdrant_path)

    def close(self) -> None:
        self.client.close()

    def search(
        self,
        query: str,
        top_k: int = 5,
        exclude_noise: bool = False,
        candidate_pool: int | None = None,
    ) -> list[dict[str, Any]]:
        del exclude_noise
        if query not in self.query_cache:
            self.query_cache[query] = self.embedder.encode_query(query)

        pool = top_k if self.mode == "plain" else max(
            candidate_pool or CANDIDATE_POOL,
            top_k,
        )
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=self.query_cache[query],
            limit=pool,
            with_payload=True,
        ).points

        selected = self._select(points, top_k)
        retrieved = []
        for rank, point in enumerate(selected, start=1):
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            if self.mode == "parent":
                chunk_id = payload.get("parent_id") or chunk_id
            retrieved.append(
                {
                    "rank": rank,
                    "score": float(point.score),
                    "chunk_id": chunk_id,
                    "document_id": payload.get("document_id"),
                    "page_start": payload.get("page_start"),
                    "page_end": payload.get("page_end"),
                    "section": payload.get("section"),
                    "text": payload.get("text", ""),
                }
            )
        return retrieved

    def _select(self, points, top_k: int):
        if self.mode == "plain":
            return list(points[:top_k])
        if self.mode == "parent":
            return _unique_parents(points, top_k)
        return _unique_windows(points, top_k)


def _unique_parents(points, top_k: int):
    selected = []
    seen: set[str] = set()
    for point in points:
        payload = point.payload or {}
        parent_id = str(payload.get("parent_id") or payload.get("chunk_id"))
        if parent_id in seen:
            continue
        seen.add(parent_id)
        selected.append(point)
        if len(selected) == top_k:
            break
    return selected


def _unique_windows(points, top_k: int):
    selected = []
    covered: dict[str, set[int]] = {}
    for point in points:
        payload = point.payload or {}
        document_id = str(payload.get("document_id"))
        center = int(payload.get("sentence_index"))
        if center in covered.get(document_id, set()):
            continue
        start = int(payload.get("window_start", center))
        end = int(payload.get("window_end", center))
        covered.setdefault(document_id, set()).update(range(start, end + 1))
        selected.append(point)
        if len(selected) == top_k:
            break
    return selected
