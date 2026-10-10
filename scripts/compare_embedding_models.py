"""
Embed the existing chunks with OpenAI and Gemini, then score retrieval
against the frozen BGE-small baseline.

The baseline collection is never deleted or rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import settings
from src.embeddings.api_embedder import (
    DailyQuotaExceeded,
    GeminiEmbedder,
    OpenAIEmbedder,
)
from src.embeddings.qwen_embedder import QwenEmbedder
from src.evaluation.evaluate_retrieval import (
    QA_PATH,
    evaluate_dataset,
    load_qa,
    print_metrics,
)
from src.retrieval.api_retriever import ApiRetriever
from src.utils.chunk_loader import load_chunks
from src.vectorstore.qdrant_index import QdrantIndex

OUTPUT_DIR = Path("data/evaluation")
BASELINE_REPORT = OUTPUT_DIR / "retrieval_baseline.json"
COMPARISON_REPORT = OUTPUT_DIR / "embedding_model_comparison.json"

EXPERIMENTS = (
    {
        "name": "OpenAI text-embedding-3-large",
        "embedder": OpenAIEmbedder,
        "collection": "nordic_green_transition_openai_3_large",
        "report": OUTPUT_DIR / "retrieval_openai_3_large.json",
    },
    {
        "name": "Gemini gemini-embedding-2",
        "embedder": lambda: GeminiEmbedder(model_name="gemini-embedding-2"),
        "collection": "nordic_green_transition_gemini_002",
        "report": OUTPUT_DIR / "retrieval_gemini_002.json",
        "cache": Path("data/embeddings/gemini_embedding_2.npy"),
    },
    {
        "name": "Qwen3-Embedding-0.6B",
        "embedder": QwenEmbedder,
        "collection": "nordic_green_transition_qwen3_0_6b",
        "report": OUTPUT_DIR / "retrieval_qwen3_0_6b.json",
    },
)


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
    chunks: list[dict],
    embeddings: np.ndarray,
) -> None:
    if collection_name == settings.qdrant_collection:
        raise RuntimeError(
            "Refusing to rebuild the BGE-small baseline collection."
        )

    index = QdrantIndex(
        path=settings.qdrant_path,
        collection_name=collection_name,
        vector_size=int(embeddings.shape[1]),
    )
    try:
        index.recreate_collection()
        index.upload(
            chunks=chunks,
            embeddings=embeddings,
            batch_size=settings.qdrant_upload_batch_size,
        )
        stored = index.count()
    finally:
        index.client.close()

    if stored != len(chunks):
        raise RuntimeError(
            f"{collection_name} stored {stored} points for {len(chunks)} chunks."
        )


def _save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {path}")


def main() -> None:
    chunks = load_chunks(settings.chunks_path)
    texts = [chunk["text"] for chunk in chunks]
    qa_examples = load_qa(QA_PATH)
    baseline = json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))

    print(f"Chunks: {len(chunks):,}")
    print(f"Questions: {len(qa_examples)}")
    print(f"Baseline collection left untouched: {settings.qdrant_collection}")
    print()

    rows = [
        {
            "name": "BGE-small-en-v1.5",
            "collection": settings.qdrant_collection,
            "metrics": baseline["metrics"],
        }
    ]

    for experiment in EXPERIMENTS:
        name = experiment["name"]
        collection = experiment["collection"]
        print("=" * 60)
        print(name)
        print("=" * 60)

        stored = _collection_count(collection)
        embedder = experiment["embedder"]()
        report_path = experiment["report"]
        if stored == len(chunks) and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            print(f"Reusing {report_path}")
            print_metrics(report["metrics"])
            rows.append(
                {
                    "name": name,
                    "collection": collection,
                    "metrics": report["metrics"],
                }
            )
            print()
            continue

        if stored == len(chunks):
            print(f"Collection {collection} already has {stored:,} points.")
        else:
            print(f"Embedding {len(texts):,} chunks with {embedder.model_name}")
            try:
                embeddings = embedder.encode_documents(
                    texts,
                    **(
                        {"cache_path": str(experiment["cache"])}
                        if "cache" in experiment
                        else {}
                    ),
                )
            except DailyQuotaExceeded:
                print(
                    "Daily Gemini embedding quota is exhausted. "
                    "Saved progress will be reused on the next run."
                )
                rows.append(
                    {
                        "name": name,
                        "collection": collection,
                        "metrics": None,
                        "status": "daily quota exhausted before indexing finished",
                    }
                )
                continue
            print(f"Embedding matrix shape: {embeddings.shape}")
            _index_embeddings(collection, chunks, embeddings)
            print(f"Indexed {len(chunks):,} points in {collection}")

        retriever = ApiRetriever(
            embedder=embedder,
            collection_name=collection,
        )
        try:
            report = evaluate_dataset(retriever, qa_examples)
        finally:
            retriever.close()

        report["model"] = embedder.model_name
        report["collection"] = collection
        print_metrics(report["metrics"])
        _save_report(experiment["report"], report)
        rows.append(
            {
                "name": name,
                "collection": collection,
                "metrics": report["metrics"],
            }
        )
        print()

    comparison = {
        "questions": len(qa_examples),
        "chunks": len(chunks),
        "relevance": "same document_id and labeled page inside page_start–page_end",
        "models": rows,
    }
    _save_report(COMPARISON_REPORT, comparison)

    print()
    print(f"{'Model':<34} {'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8} {'Hit@10':>8} {'MRR':>8}")
    for row in rows:
        metrics = row["metrics"]
        if not metrics:
            print(f"{row['name']:<34} {row.get('status', 'unavailable')}")
            continue
        print(
            f"{row['name']:<34} "
            f"{metrics['hit@1']:>8.2%} "
            f"{metrics['hit@3']:>8.2%} "
            f"{metrics['hit@5']:>8.2%} "
            f"{metrics['hit@10']:>8.2%} "
            f"{metrics['mrr']:>8.4f}"
        )


if __name__ == "__main__":
    main()
