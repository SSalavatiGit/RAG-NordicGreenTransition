from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "recursive_page_aware"
    chunk_size: int = 500
    chunk_overlap: int = 75
    minimum_chunk_size: int = 100
    tokenizer: str = "cl100k_base"


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    title: str | None
    year: int | None
    organization: str | None
    country: str | None
    topic: str | None
    secondary_topics: list[str]
    section: str | None
    section_path: list[str]
    page_start: int
    page_end: int
    content_type: str
    text: str
    token_count: int
    chunk_index: int
    source_url: str | None
    chunking: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
