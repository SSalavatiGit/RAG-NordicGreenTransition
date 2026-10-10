"""
Retrieval experiments after the frozen dense baseline.

E0 is data/evaluation/retrieval_baseline.json and is not rewritten.

E1 excludes TOC, imprint, bibliography, and blank chunks, then evaluates top 10.
E2 uses the same filtered retriever with a candidate pool of 20.
E3 reranks that pool and evaluates the reranked top 10.

Chunks that cover a labeled QA page are never marked as noise.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean

from qdrant_client import QdrantClient

from src.config import settings
from src.evaluation.evaluate_retrieval import load_qa, score_results
from src.retrieval.noise import NOISE_LABELS, classify_noise
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import DenseRetriever
from src.utils.chunk_loader import load_chunks
from src.vectorstore.qdrant_index import QdrantIndex

QA_PATH = Path("QA.txt")
OUTPUT_DIR = Path("data/evaluation")
E0_PATH = OUTPUT_DIR / "retrieval_baseline.json"

HEADLINE_KS = [1, 3, 5, 10]


def covers_labeled_page(chunk: dict, examples: list[dict]) -> bool:
    metadata = chunk["metadata"]
    document_id = metadata.get("document_id")
    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")
    if document_id is None or page_start is None or page_end is None:
        return False

    for example in examples:
        expected = example["metadata"]
        if expected["document_id"] != document_id:
            continue
        if int(page_start) <= int(expected["page"]) <= int(page_end):
            return True
    return False


def plan_noise_labels(
    chunks: list[dict],
    examples: list[dict],
) -> list[tuple[dict, str | None]]:
    planned = []
    for chunk in chunks:
        label = classify_noise(chunk["text"])
        if label is not None and covers_labeled_page(chunk, examples):
            label = None
        planned.append((chunk, label))
    return planned


def print_noise_plan(
    planned: list[tuple[dict, str | None]],
    protected: int,
) -> None:
    counts = Counter(label or "kept" for _, label in planned)
    print("Noise labels:")
    for label in ("kept", *NOISE_LABELS):
        print(f"  {label:<12} {counts[label]}")
    print(f"  excluded     {sum(counts[label] for label in NOISE_LABELS)}")
    print(f"  protected    {protected} labeled-page chunks kept despite a noise match")


def tag_noise(planned: list[tuple[dict, str | None]]) -> None:
    client = QdrantClient(path=settings.qdrant_path)
    ids_by_label: dict[str, list[str]] = {label: [] for label in NOISE_LABELS}
    clear_ids: list[str] = []

    for chunk, label in planned:
        point_id = QdrantIndex._point_id(chunk["chunk_id"])
        if label is None:
            clear_ids.append(point_id)
        else:
            ids_by_label[label].append(point_id)

    for label, point_ids in ids_by_label.items():
        if not point_ids:
            continue
        client.set_payload(
            collection_name=settings.qdrant_collection,
            payload={"noise": label},
            points=point_ids,
        )

    if clear_ids:
        client.delete_payload(
            collection_name=settings.qdrant_collection,
            keys=["noise"],
            points=clear_ids,
        )

    client.close()


def aggregate(evaluations: list[dict], ks: list[int]) -> dict:
    metrics = {
        f"hit@{k}": mean(int(item[f"hit@{k}"]) for item in evaluations)
        for k in ks
    }
    metrics["mrr"] = mean(item["reciprocal_rank"] for item in evaluations)
    metrics["num_questions"] = len(evaluations)
    return metrics


def save_report(name: str, report: dict) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {path}")
    return path


def print_comparison(rows: list[tuple[str, dict]]) -> None:
    print()
    print(
        f"{'Experiment':<28} {'Hit@1':>8} {'Hit@3':>8} "
        f"{'Hit@5':>8} {'Hit@10':>8} {'Hit@20':>8} {'MRR':>8}"
    )
    for name, metrics in rows:
        hit20 = metrics.get("hit@20")
        hit20_text = f"{hit20:.2%}" if hit20 is not None else "-"
        print(
            f"{name:<28} "
            f"{metrics['hit@1']:>8.2%} "
            f"{metrics['hit@3']:>8.2%} "
            f"{metrics['hit@5']:>8.2%} "
            f"{metrics['hit@10']:>8.2%} "
            f"{hit20_text:>8} "
            f"{metrics['mrr']:>8.4f}"
        )


def main() -> None:
    examples = load_qa(QA_PATH)
    chunks = load_chunks(settings.chunks_path)
    planned = plan_noise_labels(chunks, examples)
    protected = sum(
        classify_noise(chunk["text"]) is not None and label is None
        for chunk, label in planned
    )
    print_noise_plan(planned, protected)
    print("Writing noise labels into the existing Qdrant payloads.")
    tag_noise(planned)

    print("Loading embedding model.")
    retriever = DenseRetriever()
    print(f"Loading reranker: {settings.reranker_model}")
    reranker = CrossEncoderReranker()

    e1_results = []
    e2_results = []
    e3_results = []

    for index, example in enumerate(examples, start=1):
        print(f"[{index}/{len(examples)}] {example['question'][:80]}")
        pool = retriever.search(
            query=example["question"],
            top_k=20,
            exclude_noise=True,
        )
        e1_results.append(score_results(example, pool[:10], HEADLINE_KS))
        e2_results.append(score_results(example, pool, [*HEADLINE_KS, 20]))

        reranked = reranker.rerank(example["question"], pool)
        e3 = score_results(example, reranked[:10], HEADLINE_KS)
        e3["dense_pool_rank"] = e2_results[-1]["first_relevant_rank"]
        e3_results.append(e3)

    retriever.client.close()

    excluded = sum(label is not None for _, label in planned)
    e1_metrics = aggregate(e1_results, HEADLINE_KS)
    e2_metrics = aggregate(e2_results, [*HEADLINE_KS, 20])
    e3_metrics = aggregate(e3_results, HEADLINE_KS)

    e0_report = json.loads(E0_PATH.read_text(encoding="utf-8"))
    e0_metrics = e0_report["metrics"]

    save_report(
        "e1_structural_filter.json",
        {
            "experiment": "E1",
            "description": "E0 retriever with TOC, imprint, bibliography, and blank chunks removed. Top 10.",
            "excluded_chunks": excluded,
            "metrics": e1_metrics,
            "results": e1_results,
        },
    )
    save_report(
        "e2_dense_top20.json",
        {
            "experiment": "E2",
            "description": "Filtered dense retrieval. Top 20 is the candidate pool, not the answer set.",
            "excluded_chunks": excluded,
            "metrics": e2_metrics,
            "results": e2_results,
        },
    )
    save_report(
        "e3_reranker_top10.json",
        {
            "experiment": "E3",
            "description": "Filtered dense top 20, reordered by the reranker. Metrics use the reranked top 10.",
            "reranker_model": settings.reranker_model,
            "excluded_chunks": excluded,
            "metrics": e3_metrics,
            "results": e3_results,
        },
    )

    comparison = {
        "E0": e0_metrics,
        "E1": e1_metrics,
        "E2": e2_metrics,
        "E3": e3_metrics,
    }
    save_report("experiment_comparison.json", comparison)
    print_comparison(
        [
            ("E0 dense top-10", e0_metrics),
            ("E1 structural filter", e1_metrics),
            ("E2 dense top-20", e2_metrics),
            ("E3 top-20 + reranker", e3_metrics),
        ]
    )


if __name__ == "__main__":
    main()
