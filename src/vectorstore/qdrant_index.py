from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models


class QdrantIndex:
    """
    Stores chunk embeddings and their payloads in Qdrant.

    Important:
    We configure COSINE here, but no similarity search is performed
    during indexing. This only tells Qdrant how vectors should be
    compared later when a query is introduced.
    """

    def __init__(
        self,
        path: str,
        collection_name: str,
        vector_size: int,
    ) -> None:
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(path=path)

    def recreate_collection(self) -> None:
        if self.client.collection_exists(
            collection_name=self.collection_name
        ):
            self.client.delete_collection(
                collection_name=self.collection_name
            )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """
        Produce a deterministic UUID for each chunk ID.

        Re-running with the same chunk_id gives the same Qdrant point ID.
        """
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"nordic-rag:{chunk_id}",
            )
        )

    def upload(
        self,
        chunks: Sequence[dict[str, Any]],
        embeddings: np.ndarray,
        batch_size: int = 128,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        points: list[models.PointStruct] = []

        for chunk, embedding in zip(chunks, embeddings):
            payload = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                **chunk["metadata"],
            }

            points.append(
                models.PointStruct(
                    id=self._point_id(chunk["chunk_id"]),
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )

        self.client.upload_points(
            collection_name=self.collection_name,
            points=points,
            batch_size=batch_size,
            parallel=1,
            max_retries=3,
            wait=True,
        )

    def count(self) -> int:
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
        )
        return int(result.count)

    def set_page_ranges(
        self,
        chunks: Sequence[dict[str, Any]],
    ) -> int:
        """
        Add page_start and page_end to existing points.

        Point IDs stay the same, so the stored vectors are left unchanged.
        """
        updated = 0

        for chunk in chunks:
            metadata = chunk["metadata"]
            page_start = metadata.get("page_start")
            page_end = metadata.get("page_end")

            if page_start is None or page_end is None:
                raise ValueError(
                    f"{chunk['chunk_id']} is missing page_start or page_end."
                )

            self.client.set_payload(
                collection_name=self.collection_name,
                payload={
                    "page_start": page_start,
                    "page_end": page_end,
                },
                points=[self._point_id(chunk["chunk_id"])],
            )
            updated += 1

        return updated
