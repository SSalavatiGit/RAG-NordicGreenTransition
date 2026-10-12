"""Rerank the OpenAI parent-child shortlist with Jev Score.

Dense search still builds the shortlist: 20 unique parents from a pool of
200 child hits. Jev scores each parent against the question. Metrics use
the reranked order and the same page-span rule as the other evaluations.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

from src.config import settings
from src.embeddings.api_embedder import OpenAIEmbedder
from src.evaluation.evaluate_retrieval import (
    QA_PATH,
    HIT_K_VALUES,
    is_relevant,
    load_qa,
    mean_precision_recall_f1,
    print_metrics,
    score_results,
)
from src.utils.chunk_loader import load_chunks
from src.retrieval.jev_reranker import jev_client, rerank, score_passage
from src.retrieval.structured_retriever import StructuredRetriever

load_dotenv()

COLLECTION = "nordic_green_transition_openai_parent_child"
SHORTLIST = 20
CHILD_POOL = 200
WORKERS = 8
CACHE_PATH = Path("data/evaluation/jev_parent_child_scores.jsonl")
REPORT_PATH = Path("data/evaluation/retrieval_openai_parent_child_jev_rerank.json")
BASELINE_PATH = Path("data/evaluation/retrieval_openai_parent_child.json")


def _cache_key(question_index: int, chunk_id: str) -> str:
    return f"{question_index}:{chunk_id}"


def _load_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    scores = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scores[_cache_key(row["question_index"], row["chunk_id"])] = float(row["jev_score"])
    return scores


def _append_cache(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _relevant_totals(examples: list[dict]) -> list[int]:
    units = load_chunks(settings.chunks_path)
    totals = []
    for example in examples:
        expected = example["metadata"]
        totals.append(
            sum(
                1
                for unit in units
                if is_relevant(
                    {
                        "document_id": unit["metadata"]["document_id"],
                        "page_start": unit["metadata"]["page_start"],
                        "page_end": unit["metadata"]["page_end"],
                    },
                    expected["document_id"],
                    expected["page"],
                )
            )
        )
    return totals


def _mean_metrics(evaluations: list[dict]) -> dict:
    from statistics import mean

    metrics = {
        f"hit@{k}": mean(int(item[f"hit@{k}"]) for item in evaluations)
        for k in HIT_K_VALUES
    }
    metrics["mrr"] = mean(item["reciprocal_rank"] for item in evaluations)
    metrics["num_questions"] = len(evaluations)
    return metrics


def main() -> None:
    examples = load_qa(QA_PATH)
    cache = _load_cache(CACHE_PATH)
    embedder = OpenAIEmbedder()
    retriever = StructuredRetriever(
        embedder=embedder,
        collection_name=COLLECTION,
        mode="parent",
    )
    client = jev_client()
    cache_lock = Lock()
    evaluations = []

    def _score_one(question_index: int, question: str, passage: dict) -> float:
        key = _cache_key(question_index, str(passage["chunk_id"]))
        with cache_lock:
            cached = cache.get(key)
        if cached is not None:
            return cached
        score = score_passage(client, question, passage["text"])
        row = {
            "question_index": question_index,
            "chunk_id": passage["chunk_id"],
            "jev_score": score,
        }
        with cache_lock:
            cache[key] = score
            _append_cache(CACHE_PATH, row)
        return score

    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for question_index, example in enumerate(examples, start=1):
                question = example["question"]
                shortlist = retriever.search(
                    question,
                    top_k=SHORTLIST,
                    candidate_pool=CHILD_POOL,
                )
                scores = list(
                    pool.map(
                        lambda passage, question_index=question_index, question=question: (
                            _score_one(question_index, question, passage)
                        ),
                        shortlist,
                    )
                )
                ordered = rerank(shortlist, scores)
                evaluations.append(score_results(example, ordered))
                print(
                    f"[{question_index}/{len(examples)}] "
                    f"shortlist {len(shortlist)} "
                    f"top jev {max(scores):.2f}",
                    flush=True,
                )
    finally:
        retriever.close()
        client.close()

    metrics = _mean_metrics(evaluations)
    metrics.update(mean_precision_recall_f1(evaluations, _relevant_totals(examples)))
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["metrics"]
    report = {
        "retrieval": "OpenAI parent-child",
        "reranker": "Jev Score",
        "model": "jev-1.13.0",
        "shortlist": SHORTLIST,
        "child_pool": CHILD_POOL,
        "levels": [
            "Does not answer the question.",
            "Same topic, but does not answer the question.",
            "Answers part of the question.",
            "Answers the question.",
        ],
        "metrics": metrics,
        "parent_child_without_rerank": {
            key: baseline[key] for key in ("hit@1", "hit@3", "hit@5", "hit@10", "mrr")
        },
        "results": evaluations,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print_metrics(metrics)
    print(f"Saved {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
