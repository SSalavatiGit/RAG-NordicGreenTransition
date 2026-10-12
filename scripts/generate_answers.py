"""Generate answers from the saved Jev top 5.

Retrieval is not run again. Each question uses the first five parents in
data/evaluation/retrieval_openai_parent_child_jev_rerank.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.generation.answer import GENERATION_MODEL, generate_answer

load_dotenv()

TOP_K = 5
RETRIEVAL_PATH = Path("data/evaluation/retrieval_openai_parent_child_jev_rerank.json")
CACHE_PATH = Path("data/evaluation/generation_parent_child_jev_k5.jsonl")
REPORT_PATH = Path("data/evaluation/generation_parent_child_jev_k5.json")


def _load_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["question_index"])] = row
    return rows


def _append_cache(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _top_passages(result: dict) -> list[dict]:
    top = [item for item in result["retrieved"] if int(item["rank"]) <= TOP_K]
    top.sort(key=lambda item: int(item["rank"]))
    return top


def main() -> None:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env.")

    retrieval = json.loads(RETRIEVAL_PATH.read_text(encoding="utf-8"))
    results = retrieval["results"]
    cache = _load_cache(CACHE_PATH)
    client = OpenAI(api_key=api_key)
    answers = []

    for question_index, result in enumerate(results, start=1):
        passages = _top_passages(result)
        if len(passages) != TOP_K:
            raise RuntimeError(
                f"Question {question_index} has {len(passages)} passages "
                f"in the top {TOP_K}."
            )

        cached = cache.get(question_index)
        if cached is None:
            generated = generate_answer(client, result["question"], passages)
            cached = {
                "question_index": question_index,
                "question": result["question"],
                "answer": generated["answer"],
                "citations": generated["citations"],
            }
            _append_cache(CACHE_PATH, cached)
            cache[question_index] = cached
            print(
                f"[{question_index}/{len(results)}] "
                f"{len(cached['citations'])} citations",
                flush=True,
            )

        answers.append(
            {
                "question": result["question"],
                "answer": cached["answer"],
                "citations": cached["citations"],
                "passages": [
                    {
                        "rank": int(passage["rank"]),
                        "chunk_id": passage.get("chunk_id"),
                        "document_id": passage["document_id"],
                        "page_start": passage["page_start"],
                        "page_end": passage["page_end"],
                    }
                    for passage in passages
                ],
            }
        )

    report = {
        "model": GENERATION_MODEL,
        "temperature": 0,
        "retrieval": "OpenAI parent-child + Jev Score",
        "k": TOP_K,
        "source": str(RETRIEVAL_PATH).replace("\\", "/"),
        "num_questions": len(answers),
        "results": answers,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    sample = answers[0]
    print(f"Saved {REPORT_PATH}", flush=True)
    print(flush=True)
    print(sample["question"], flush=True)
    print(sample["answer"], flush=True)
    print(json.dumps(sample["citations"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
