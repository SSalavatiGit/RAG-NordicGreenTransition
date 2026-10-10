from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5",
    )

    reranker_model: str = os.getenv(
        "RERANKER_MODEL",
        "BAAI/bge-reranker-base",
    )

    embedding_batch_size: int = int(
        os.getenv("EMBEDDING_BATCH_SIZE", "32")
    )

    chunks_path: str = os.getenv(
        "CHUNKS_PATH",
        "data/processed/chunks.jsonl",
    )

    qdrant_path: str = os.getenv(
        "QDRANT_PATH",
        "data/qdrant",
    )

    qdrant_collection: str = os.getenv(
        "QDRANT_COLLECTION",
        "nordic_green_transition_bge_small_v1",
    )

    qdrant_upload_batch_size: int = int(
        os.getenv("QDRANT_UPLOAD_BATCH_SIZE", "128")
    )


settings = Settings()