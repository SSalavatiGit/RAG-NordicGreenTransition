"""Retrieval units for the chunking comparison.

The fixed-size baseline stays 500 tokens with 75 overlap. These builders
match that scale:

- sentence window: embed one sentence, return neighbors up to 500 tokens
- semantic chunks: break where adjacent sentences diverge, then cap at 500
- parent-child: embed 125-token children, return the existing 500-token parent
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.chunking.recursive_chunker import (
    RecursivePageAwareChunker,
    _SENTENCE_SPLIT,
)

MAX_WINDOW_TOKENS = 500
MAX_SENTENCE_TOKENS = 80
SEMANTIC_TARGET_MEDIAN = 450
SEMANTIC_MAX_TOKENS = 500
SEMANTIC_MIN_TOKENS = 100
SEMANTIC_OVERFLOW_OVERLAP = 75
CHILD_TOKENS = 125
CHILD_OVERLAP = 20

STARTER_DOCUMENT_IDS = (
    "doc_002",
    "doc_004",
    "doc_010",
    "doc_013",
    "doc_014",
    "doc_015",
    "doc_016",
    "doc_018",
    "doc_022",
)


def _token_count(counter, text: str) -> int:
    return int(counter.count(text))


def _join(parts: list[dict[str, Any]]) -> str:
    return " ".join(part["text"] for part in parts).strip()


def _split_words(text: str, page: int, max_tokens: int, counter) -> list[dict[str, Any]]:
    words = text.split()
    pieces: list[dict[str, Any]] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            pieces.append({"text": " ".join(buffer), "page": page})
            buffer.clear()

    for word in words:
        candidate = " ".join([*buffer, word])
        if buffer and _token_count(counter, candidate) > max_tokens:
            flush()
        buffer.append(word)
    flush()
    return pieces


def _sentence_pieces(text: str, page: int, counter) -> list[dict[str, Any]]:
    pieces: list[dict[str, Any]] = []
    for raw in _SENTENCE_SPLIT.split(text):
        sentence = raw.strip()
        if not sentence:
            continue
        if _token_count(counter, sentence) <= MAX_SENTENCE_TOKENS:
            pieces.append({"text": sentence, "page": page})
            continue
        pieces.extend(
            _split_words(sentence, page, MAX_SENTENCE_TOKENS, counter)
        )
    return pieces


def _metadata(document: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    source = document.get("metadata") or {}
    metadata = {
        "document_id": document["document_id"],
        "title": source.get("title"),
        "year": source.get("year"),
        "organization": source.get("organization"),
        "country": source.get("country"),
        "topic": source.get("topic"),
        "secondary_topics": source.get("secondary_topics"),
        "source_url": source.get("source_url"),
        "section": None,
    }
    metadata.update(extra)
    return {
        key: value
        for key, value in metadata.items()
        if value is not None
    }


def load_starter_documents(processed_dir: Path) -> list[dict[str, Any]]:
    documents = []
    for document_id in STARTER_DOCUMENT_IDS:
        path = processed_dir / f"{document_id}.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("document_id") != document_id:
            raise ValueError(f"{path} has document_id {document.get('document_id')}.")
        documents.append(document)
    return documents


def build_sentences(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One record per sentence, with the surrounding window as the returned text."""
    counter = RecursivePageAwareChunker().tokens
    cleaner = RecursivePageAwareChunker()
    records: list[dict[str, Any]] = []

    for document in documents:
        pieces: list[dict[str, Any]] = []
        for page in document.get("pages") or []:
            page_number = int(page["page_number"])
            raw = page.get("text") or ""
            if not raw.strip():
                continue
            for block in cleaner._clean_page_to_blocks(raw):
                pieces.extend(_sentence_pieces(block, page_number, counter))

        piece_tokens = [_token_count(counter, piece["text"]) for piece in pieces]
        for index, piece in enumerate(pieces):
            start, end = _fit_window(piece_tokens, index)
            window = pieces[start:end]
            window_text = _join(window)
            records.append(
                {
                    "chunk_id": f"{document['document_id']}_sent_{index:05d}",
                    "text": window_text,
                    "embed_text": piece["text"],
                    "metadata": _metadata(
                        document,
                        {
                            "page_start": min(item["page"] for item in window),
                            "page_end": max(item["page"] for item in window),
                            "sentence_index": index,
                            "sentence_page": piece["page"],
                            "window_start": start,
                            "window_end": end - 1,
                            "unit": "sentence_window",
                        },
                    ),
                }
            )
    return records


def _fit_window(piece_tokens: list[int], index: int) -> tuple[int, int]:
    """Grow equally around a sentence until the next sentence would pass 500 tokens.

    Piece counts plus one token per joining space stay under the cap. The
    joined window can be a few tokens shorter than the sum.
    """
    start = index
    end = index + 1
    total = piece_tokens[index]
    while True:
        grew = False
        if start > 0:
            added = piece_tokens[start - 1] + 1
            if total + added <= MAX_WINDOW_TOKENS:
                start -= 1
                total += added
                grew = True
        if end < len(piece_tokens):
            added = piece_tokens[end] + 1
            if total + added <= MAX_WINDOW_TOKENS:
                end += 1
                total += added
                grew = True
        if not grew:
            return start, end


def _pack(
    pieces: list[dict[str, Any]],
    max_tokens: int,
    overlap: int,
    counter,
) -> list[list[dict[str, Any]]]:
    expanded: list[dict[str, Any]] = []
    for piece in pieces:
        if _token_count(counter, piece["text"]) <= max_tokens:
            expanded.append(piece)
        else:
            expanded.extend(
                _split_words(piece["text"], piece["page"], max_tokens, counter)
            )

    groups: list[list[dict[str, Any]]] = []
    buffer: list[dict[str, Any]] = []

    def packed_tokens(items: list[dict[str, Any]]) -> int:
        if not items:
            return 0
        return _token_count(counter, _join(items))

    for piece in expanded:
        if buffer and packed_tokens(buffer + [piece]) > max_tokens:
            groups.append(buffer)
            tail: list[dict[str, Any]] = []
            for item in reversed(buffer):
                candidate = [item, *tail]
                if packed_tokens(candidate) > overlap:
                    break
                tail = candidate
            buffer = [*tail, piece]
            if packed_tokens(buffer) > max_tokens:
                buffer = [piece]
        else:
            buffer.append(piece)
    if buffer:
        groups.append(buffer)
    return groups


def _group_tokens(counter, group: list[dict[str, Any]]) -> int:
    return _token_count(counter, _join(group))


def _merge_small(
    groups: list[list[dict[str, Any]]],
    flags: list[bool],
    counter,
) -> tuple[list[list[dict[str, Any]]], list[bool]]:
    if not groups:
        return [], []
    merged = [groups[0]]
    merged_flags = [flags[0]]
    for group, flag in zip(groups[1:], flags[1:]):
        current = _group_tokens(counter, group)
        previous = _group_tokens(counter, merged[-1])
        combined = _group_tokens(counter, [*merged[-1], *group])
        if (
            current < SEMANTIC_MIN_TOKENS or previous < SEMANTIC_MIN_TOKENS
        ) and combined <= SEMANTIC_MAX_TOKENS:
            merged[-1] = [*merged[-1], *group]
            merged_flags[-1] = merged_flags[-1] or flag
        else:
            merged.append(group)
            merged_flags.append(flag)
    return merged, merged_flags


def _chunks_from_threshold(
    by_document: list[tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]],
    threshold: float,
    counter,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for document, pieces, distances in by_document:
        groups: list[list[dict[str, Any]]] = []
        current = [pieces[0]]
        for index, distance in enumerate(distances):
            if float(distance) > threshold:
                groups.append(current)
                current = [pieces[index + 1]]
            else:
                current.append(pieces[index + 1])
        groups.append(current)

        capped_groups: list[list[dict[str, Any]]] = []
        capped_flags: list[bool] = []
        for group in groups:
            if _group_tokens(counter, group) <= SEMANTIC_MAX_TOKENS:
                capped_groups.append(group)
                capped_flags.append(False)
                continue
            for part in _pack(
                group,
                SEMANTIC_MAX_TOKENS,
                SEMANTIC_OVERFLOW_OVERLAP,
                counter,
            ):
                capped_groups.append(part)
                capped_flags.append(True)

        merged, final_flags = _merge_small(capped_groups, capped_flags, counter)

        for index, group in enumerate(merged):
            text = _join(group)
            records.append(
                {
                    "chunk_id": f"{document['document_id']}_sem_{index:04d}",
                    "text": text,
                    "embed_text": text,
                    "metadata": _metadata(
                        document,
                        {
                            "page_start": min(item["page"] for item in group),
                            "page_end": max(item["page"] for item in group),
                            "unit": "semantic",
                            "capped": final_flags[index],
                        },
                    ),
                }
            )
    return records


def build_semantic_chunks(
    documents: list[dict[str, Any]],
    sentence_records: list[dict[str, Any]],
    sentence_embeddings: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose the distance cutoff whose median chunk is nearest 450 tokens."""
    if len(sentence_records) != len(sentence_embeddings):
        raise ValueError("Sentence embeddings do not match the sentence records.")

    counter = RecursivePageAwareChunker().tokens
    vectors_by_doc: dict[str, list[np.ndarray]] = {}
    records_by_doc: dict[str, list[dict[str, Any]]] = {}
    for record, vector in zip(sentence_records, sentence_embeddings):
        document_id = record["metadata"]["document_id"]
        records_by_doc.setdefault(document_id, []).append(record)
        vectors_by_doc.setdefault(document_id, []).append(vector)

    prepared: list[tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]] = []
    distances: list[float] = []

    for document in documents:
        own_records = records_by_doc.get(document["document_id"], [])
        own_vectors = vectors_by_doc.get(document["document_id"], [])
        if not own_records:
            continue
        pieces = [
            {
                "text": record["embed_text"],
                "page": int(record["metadata"]["sentence_page"]),
            }
            for record in own_records
        ]
        matrix = np.vstack(own_vectors)
        if len(pieces) == 1:
            pair_distances = np.asarray([], dtype=np.float32)
        else:
            pair_distances = 1.0 - np.sum(matrix[:-1] * matrix[1:], axis=1)
            distances.extend(float(value) for value in pair_distances)
        prepared.append((document, pieces, pair_distances))

    if not distances:
        raise ValueError("Need at least two sentences to place semantic breaks.")

    distance_array = np.asarray(distances, dtype=np.float32)
    best_records: list[dict[str, Any]] | None = None
    best_stats: dict[str, Any] | None = None
    best_gap: float | None = None

    for percentile in range(40, 100):
        threshold = float(np.percentile(distance_array, percentile))
        records = _chunks_from_threshold(prepared, threshold, counter)
        tokens = [_token_count(counter, record["text"]) for record in records]
        median = float(np.median(tokens))
        gap = abs(median - SEMANTIC_TARGET_MEDIAN)
        if best_gap is not None and gap >= best_gap:
            continue
        best_gap = gap
        best_records = records
        best_stats = {
            "percentile": percentile,
            "distance_threshold": threshold,
            "chunks": len(records),
            "median_tokens": median,
            "mean_tokens": float(np.mean(tokens)),
            "min_tokens": int(min(tokens)),
            "max_tokens": int(max(tokens)),
            "capped_chunks": sum(
                1 for record in records if record["metadata"].get("capped")
            ),
            "target_median_tokens": SEMANTIC_TARGET_MEDIAN,
            "max_tokens_cap": SEMANTIC_MAX_TOKENS,
            "min_tokens_floor": SEMANTIC_MIN_TOKENS,
        }

    if best_records is None or best_stats is None:
        raise RuntimeError("Semantic threshold search produced no chunks.")
    return best_records, best_stats


def build_children(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split each fixed-size parent into 125-token children with 20 overlap."""
    counter = RecursivePageAwareChunker().tokens
    children: list[dict[str, Any]] = []

    for parent in parents:
        metadata = parent["metadata"]
        page_start = int(metadata["page_start"])
        page_end = int(metadata["page_end"])
        pieces = _sentence_pieces(parent["text"], page_start, counter)
        groups = _pack(pieces, CHILD_TOKENS, CHILD_OVERLAP, counter)
        for index, group in enumerate(groups):
            child_text = _join(group)
            child_metadata = dict(metadata)
            child_metadata.update(
                {
                    "page_start": page_start,
                    "page_end": page_end,
                    "parent_id": parent["chunk_id"],
                    "unit": "child",
                }
            )
            children.append(
                {
                    "chunk_id": f"{parent['chunk_id']}_child_{index:03d}",
                    "text": parent["text"],
                    "embed_text": child_text,
                    "metadata": child_metadata,
                }
            )
    return children
