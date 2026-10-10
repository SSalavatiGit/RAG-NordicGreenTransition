from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TEXT_KEYS = ("text", "content", "chunk_text", "page_content")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)

                if not isinstance(row, dict):
                    raise ValueError(
                        f"Line {line_number} is not a JSON object."
                    )

                rows.append(row)

        return rows

    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(obj, list):
            return obj

        if isinstance(obj, dict) and isinstance(obj.get("chunks"), list):
            return obj["chunks"]

        raise ValueError(
            "JSON input must either be a list of chunks or "
            "an object containing a top-level 'chunks' list."
        )

    raise ValueError(
        f"Unsupported file type: {suffix}. "
        "Use .jsonl or .json."
    )


def _get_text(row: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = row.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(
        "Chunk contains no text. Expected one of these fields: "
        + ", ".join(TEXT_KEYS)
    )


def _get_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata")

    if isinstance(metadata, dict):
        result = dict(metadata)
    else:
        result = {}

    # Preserve common RAG metadata even when stored at top level.
    for key in (
        "document_id",
        "title",
        "year",
        "organization",
        "country",
        "topic",
        "secondary_topics",
        "section",
        "page",
        "page_start",
        "page_end",
        "source_url",
        "license",
        "publication_number",
    ):
        if key in row and key not in result:
            result[key] = row[key]

    return {
        key: value
        for key, value in result.items()
        if value is not None
    }


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    """
    Load existing parsed/chunked data.

    Normalized output:
    {
        "chunk_id": "...",
        "text": "...",
        "metadata": {...}
    }
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {path.resolve()}"
        )

    raw_rows = _read_rows(path)
    chunks: list[dict[str, Any]] = []

    for index, row in enumerate(raw_rows):
        text = _get_text(row)
        metadata = _get_metadata(row)

        chunk_id = (
            row.get("chunk_id")
            or metadata.get("chunk_id")
            or f"chunk_{index:06d}"
        )

        chunks.append(
            {
                "chunk_id": str(chunk_id),
                "text": text,
                "metadata": metadata,
            }
        )

    if not chunks:
        raise ValueError("No chunks were loaded.")

    return chunks
