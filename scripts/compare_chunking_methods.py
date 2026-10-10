"""Compare sentence-window, semantic, and parent-child retrieval.

The fixed-size OpenAI collection is left in place and reused as the control.
New units are stored in their own collections. Scoring is the same Hit@k
rule as the embedding comparison: correct document, labeled page inside
page_start–page_end, 10 returned passages.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.chunking.comparison_units import (
    CHILD_OVERLAP,
    CHILD_TOKENS,
    MAX_WINDOW_TOKENS,
    build_children,
    build_semantic_chunks,
    build_sentences,
    load_starter_documents,
)
from src.chunking.recursive_chunker import RecursivePageAwareChunker
from src.config import settings
from src.embeddings.api_embedder import OpenAIEmbedder
from src.evaluation.evaluate_retrieval import (
    QA_PATH,
    evaluate_dataset,
    load_qa,
    print_metrics,
)
from src.retrieval.structured_retriever import StructuredRetriever
from src.utils.chunk_loader import load_chunks
from src.vectorstore.qdrant_index import QdrantIndex

OUTPUT_DIR = Path("data/evaluation")
UNIT_DIR = Path("data/processed/chunks")
EMBEDDING_DIR = Path("data/embeddings")
PROCESSED_DIR = Path("data/processed")
COMPARISON_REPORT = OUTPUT_DIR / "chunking_method_comparison.json"

PROTECTED_COLLECTIONS = {
    "nordic_green_transition_bge_small_v1",
    "nordic_green_transition_openai_3_large",
    "nordic_green_transition_gemini_002",
    "nordic_green_transition_qwen3_0_6b",
}

PRIOR_REPORTS = (
    ("BGE-small fixed", "fixed-size embedding", OUTPUT_DIR / "retrieval_baseline.json"),
    ("Qwen3-0.6B fixed", "fixed-size embedding", OUTPUT_DIR / "retrieval_qwen3_0_6b.json"),
    ("Gemini embedding-2 fixed", "fixed-size embedding", OUTPUT_DIR / "retrieval_gemini_002.json"),
    ("OpenAI fixed 500/75", "fixed-size embedding", OUTPUT_DIR / "retrieval_openai_3_large.json"),
    ("BGE E1 filter", "BGE retrieval experiment", OUTPUT_DIR / "e1_structural_filter.json"),
    ("BGE E2 top-20", "BGE retrieval experiment", OUTPUT_DIR / "e2_dense_top20.json"),
    ("BGE E3 reranker", "BGE retrieval experiment", OUTPUT_DIR / "e3_reranker_top10.json"),
)


def _save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {path}", flush=True)


def _collection_count(collection_name: str) -> int | None:
    index = QdrantIndex(
        path=settings.qdrant_path,
        collection_name=collection_name,
        vector_size=1,
    )
    try:
        if not index.client.collection_exists(collection_name):
            return None
        return index.count()
    finally:
        index.client.close()


def _index_embeddings(
    collection_name: str,
    records: list[dict],
    embeddings: np.ndarray,
) -> None:
    if collection_name in PROTECTED_COLLECTIONS or collection_name == settings.qdrant_collection:
        raise RuntimeError(f"Refusing to rebuild {collection_name}.")

    index = QdrantIndex(
        path=settings.qdrant_path,
        collection_name=collection_name,
        vector_size=int(embeddings.shape[1]),
    )
    try:
        index.recreate_collection()
        index.upload(
            chunks=records,
            embeddings=embeddings,
            batch_size=settings.qdrant_upload_batch_size,
        )
        stored = index.count()
    finally:
        index.client.close()

    if stored != len(records):
        raise RuntimeError(
            f"{collection_name} stored {stored} points for {len(records)} records."
        )


def _token_stats(texts: list[str]) -> dict[str, float]:
    counter = RecursivePageAwareChunker().tokens
    counts = [counter.count(text) for text in texts]
    array = np.asarray(counts)
    return {
        "units": int(array.size),
        "min_tokens": int(array.min()),
        "median_tokens": float(np.median(array)),
        "mean_tokens": float(array.mean()),
        "max_tokens": int(array.max()),
    }


def _load_prior() -> list[dict]:
    rows = []
    for name, group, path in PRIOR_REPORTS:
        if not path.exists():
            print(f"Missing prior report: {path}", flush=True)
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "name": name,
                "group": group,
                "metrics": report["metrics"],
            }
        )
    return rows


def _units() -> tuple[list[dict], list[dict], list[dict], dict]:
    sentence_path = UNIT_DIR / "sentence_windows.jsonl"
    if sentence_path.exists():
        sentences = _load_jsonl(sentence_path)
        print(f"Loaded {len(sentences):,} sentence windows", flush=True)
    else:
        documents = load_starter_documents(PROCESSED_DIR)
        sentences = build_sentences(documents)
        _save_jsonl(sentence_path, sentences)
        print(f"Built {len(sentences):,} sentence windows", flush=True)

    child_path = UNIT_DIR / "parent_children.jsonl"
    if child_path.exists():
        children = _load_jsonl(child_path)
        print(f"Loaded {len(children):,} child chunks", flush=True)
    else:
        parents = load_chunks(settings.chunks_path)
        children = build_children(parents)
        _save_jsonl(child_path, children)
        print(f"Built {len(children):,} child chunks from {len(parents):,} parents", flush=True)

    semantic_path = UNIT_DIR / "semantic_chunks.jsonl"
    semantic_stats: dict
    if semantic_path.exists():
        semantic = _load_jsonl(semantic_path)
        stats_path = UNIT_DIR / "semantic_chunk_stats.json"
        semantic_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print(f"Loaded {len(semantic):,} semantic chunks", flush=True)
    else:
        documents = load_starter_documents(PROCESSED_DIR)
        embedder = OpenAIEmbedder()
        print(f"Embedding {len(sentences):,} sentences", flush=True)
        sentence_vectors = embedder.encode_documents(
            [record["embed_text"] for record in sentences],
            cache_path=str(EMBEDDING_DIR / "openai_sentences.npy"),
        )
        semantic, semantic_stats = build_semantic_chunks(
            documents,
            sentences,
            sentence_vectors,
        )
        _save_jsonl(semantic_path, semantic)
        stats_path = UNIT_DIR / "semantic_chunk_stats.json"
        stats_path.write_text(
            json.dumps(semantic_stats, indent=2),
            encoding="utf-8",
        )
        print("Semantic chunk stats:", json.dumps(semantic_stats), flush=True)

    return sentences, semantic, children, semantic_stats


def _run_method(
    name: str,
    records: list[dict],
    collection: str,
    mode: str,
    cache_path: Path,
    report_path: Path,
    embedder: OpenAIEmbedder,
    qa_examples: list[dict],
    query_cache: dict[str, list[float]],
    extra: dict,
) -> dict:
    print("=" * 60, flush=True)
    print(name, flush=True)
    print("=" * 60, flush=True)
    stored = _collection_count(collection)
    if stored == len(records) and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"Reusing {report_path}", flush=True)
        print_metrics(report["metrics"])
        return report["metrics"]

    if stored != len(records):
        print(f"Embedding {len(records):,} units", flush=True)
        embeddings = embedder.encode_documents(
            [record["embed_text"] for record in records],
            cache_path=str(cache_path),
        )
        _index_embeddings(collection, records, embeddings)
        print(f"Indexed {len(records):,} points in {collection}", flush=True)

    retriever = StructuredRetriever(
        embedder=embedder,
        collection_name=collection,
        mode=mode,
        query_cache=query_cache,
    )
    try:
        report = evaluate_dataset(retriever, qa_examples)
    finally:
        retriever.close()

    report["model"] = embedder.model_name
    report["collection"] = collection
    report["retrieval_mode"] = mode
    report.update(extra)
    print_metrics(report["metrics"])
    _save_report(report_path, report)
    return report["metrics"]


def _print_table(rows: list[dict]) -> None:
    print(flush=True)
    print(
        f"{'Method':<32} {'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8} {'Hit@10':>8} {'MRR':>8}",
        flush=True,
    )
    for row in rows:
        metrics = row["metrics"]
        print(
            f"{row['name']:<32} "
            f"{metrics['hit@1']:>8.2%} "
            f"{metrics['hit@3']:>8.2%} "
            f"{metrics['hit@5']:>8.2%} "
            f"{metrics['hit@10']:>8.2%} "
            f"{metrics['mrr']:>8.4f}",
            flush=True,
        )


def main() -> None:
    if settings.qdrant_collection in {
        "nordic_green_transition_openai_sentence_window",
        "nordic_green_transition_openai_semantic",
        "nordic_green_transition_openai_parent_child",
    }:
        raise RuntimeError("QDRANT_COLLECTION points at a chunking-experiment collection.")

    qa_examples = load_qa(QA_PATH)
    sentences, semantic, children, semantic_stats = _units()
    window_stats = _token_stats([record["text"] for record in sentences])
    sentence_stats = _token_stats([record["embed_text"] for record in sentences])
    child_stats = _token_stats([record["embed_text"] for record in children])
    print("Sentence tokens:", json.dumps(sentence_stats), flush=True)
    print("Window tokens:", json.dumps(window_stats), flush=True)
    print("Child tokens:", json.dumps(child_stats), flush=True)
    print("Semantic tokens:", json.dumps(semantic_stats), flush=True)

    embedder = OpenAIEmbedder()
    query_cache: dict[str, list[float]] = {}
    methods = (
        {
            "name": "OpenAI sentence-window",
            "records": sentences,
            "collection": "nordic_green_transition_openai_sentence_window",
            "mode": "window",
            "cache": EMBEDDING_DIR / "openai_sentences.npy",
            "report": OUTPUT_DIR / "retrieval_openai_sentence_window.json",
            "extra": {
                "max_window_tokens": MAX_WINDOW_TOKENS,
                "sentence_stats": sentence_stats,
                "window_stats": window_stats,
            },
        },
        {
            "name": "OpenAI semantic",
            "records": semantic,
            "collection": "nordic_green_transition_openai_semantic",
            "mode": "plain",
            "cache": EMBEDDING_DIR / "openai_semantic_chunks.npy",
            "report": OUTPUT_DIR / "retrieval_openai_semantic.json",
            "extra": {"semantic_stats": semantic_stats},
        },
        {
            "name": "OpenAI parent-child",
            "records": children,
            "collection": "nordic_green_transition_openai_parent_child",
            "mode": "parent",
            "cache": EMBEDDING_DIR / "openai_children.npy",
            "report": OUTPUT_DIR / "retrieval_openai_parent_child.json",
            "extra": {
                "child_tokens": CHILD_TOKENS,
                "child_overlap": CHILD_OVERLAP,
                "child_stats": child_stats,
            },
        },
    )

    rows = _load_prior()
    for method in methods:
        metrics = _run_method(
            name=method["name"],
            records=method["records"],
            collection=method["collection"],
            mode=method["mode"],
            cache_path=method["cache"],
            report_path=method["report"],
            embedder=embedder,
            qa_examples=qa_examples,
            query_cache=query_cache,
            extra=method["extra"],
        )
        rows.append(
            {
                "name": method["name"],
                "group": "OpenAI chunking",
                "collection": method["collection"],
                "metrics": metrics,
            }
        )

    comparison = {
        "questions": len(qa_examples),
        "relevance": "same document_id and labeled page inside page_start–page_end",
        "returned_passages": 10,
        "new_methods_embedding": "text-embedding-3-large",
        "fixed_size_control": {
            "chunk_size": 500,
            "overlap": 75,
            "collection": "nordic_green_transition_openai_3_large",
        },
        "sentence_window": {
            "max_window_tokens": MAX_WINDOW_TOKENS,
            "embedded_unit": sentence_stats,
            "returned_window": window_stats,
        },
        "semantic": semantic_stats,
        "parent_child": {
            "child_tokens": CHILD_TOKENS,
            "child_overlap": CHILD_OVERLAP,
            "children": child_stats,
        },
        "methods": rows,
    }
    _save_report(COMPARISON_REPORT, comparison)
    _print_table(rows)


if __name__ == "__main__":
    main()
