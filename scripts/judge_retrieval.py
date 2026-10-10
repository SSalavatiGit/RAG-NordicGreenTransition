"""Evidence judge for the selected retriever.

Setup: OpenAI parent-child, top 5 parents.
Judge: gpt-4.1-mini-2025-04-14, temperature 0.
The prompt contains the question, the labeled evidence_text, and one passage.
It does not contain page numbers, document ids, ranks, or similarity scores.

supports means the passage states the reference evidence.
partial means it states only part of that evidence.
miss means it does not state it.

Evidence Hit@5 is true when any of the five labels is supports.
Partial matches are counted separately and are not treated as hits.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.embeddings.api_embedder import _retry
from src.evaluation.evaluate_retrieval import QA_PATH, load_qa

load_dotenv()

JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"
TOP_K = 5
RETRIEVAL_PATH = Path("data/evaluation/retrieval_openai_parent_child.json")
CACHE_PATH = Path("data/evaluation/judge_parent_child_k5.jsonl")
REPORT_PATH = Path("data/evaluation/retrieval_judge_parent_child_k5.json")

SYSTEM_PROMPT = """You judge whether a retrieved passage contains a reference evidence statement.

Use only the passage. A related topic is not enough.
supports: the passage states the reference evidence, including its specific claims.
partial: the passage states some of the reference evidence and leaves out a material part.
miss: the passage does not state the reference evidence.

A summary or a neighboring discussion that only shares the topic is partial or miss, not supports.
Return JSON with keys label and missing. missing is what the passage leaves out, or an empty string."""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "evidence_judgment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["supports", "partial", "miss"],
                },
                "missing": {"type": "string"},
            },
            "required": ["label", "missing"],
            "additionalProperties": False,
        },
    },
}


def _cache_key(question_index: int, rank: int) -> str:
    return f"{question_index}:{rank}"


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[_cache_key(row["question_index"], row["rank"])] = row
    return rows


def _append_cache(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _judge_passage(client, question: str, evidence: str, passage: str) -> dict:
    response = _retry(
        lambda: client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=0,
            response_format=SCHEMA,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Reference evidence:\n{evidence}\n\n"
                        f"Passage:\n{passage}"
                    ),
                },
            ],
        )
    )
    payload = json.loads(response.choices[0].message.content)
    label = payload["label"]
    if label not in {"supports", "partial", "miss"}:
        raise RuntimeError(f"Unexpected judge label: {label}")
    return {"label": label, "missing": payload.get("missing", "")}


def _summarize(judged: list[dict]) -> dict:
    supports = 0
    partial = 0
    miss = 0
    evidence_hits = 0
    partial_only = 0
    strict_hits = 0
    strict_miss_but_supports = 0
    strict_hit_but_miss = 0
    for question in judged:
        labels = [item["label"] for item in question["judgments"]]
        supports += labels.count("supports")
        partial += labels.count("partial")
        miss += labels.count("miss")
        has_support = "supports" in labels
        has_partial = "partial" in labels
        if has_support:
            evidence_hits += 1
        elif has_partial:
            partial_only += 1
        if question["strict_hit@5"]:
            strict_hits += 1
            if not has_support and not has_partial:
                strict_hit_but_miss += 1
        elif has_support:
            strict_miss_but_supports += 1

    questions = len(judged)
    passages = questions * TOP_K
    return {
        "questions": questions,
        "passages": passages,
        "supports": supports,
        "partial": partial,
        "miss": miss,
        "evidence_hit@5": evidence_hits / questions,
        "partial_only@5": partial_only / questions,
        "evidence_precision@5": supports / passages,
        "strict_hit@5": strict_hits / questions,
        "strict_miss_but_evidence_supports": strict_miss_but_supports,
        "strict_hit_but_judge_miss": strict_hit_but_miss,
    }


def main() -> None:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env.")

    examples = load_qa(QA_PATH)
    evidence_by_question = {
        example["question"]: example["metadata"]["evidence_text"]
        for example in examples
    }
    retrieval = json.loads(RETRIEVAL_PATH.read_text(encoding="utf-8"))
    results = retrieval["results"]
    if len(results) != len(examples):
        raise RuntimeError(
            f"Retrieval file has {len(results)} questions, QA has {len(examples)}."
        )

    client = OpenAI(api_key=api_key)
    cache = _load_cache(CACHE_PATH)
    judged = []

    for question_index, result in enumerate(results, start=1):
        question = result["question"]
        evidence = evidence_by_question.get(question)
        if not evidence:
            raise RuntimeError(f"No evidence_text for question {question_index}.")
        top = [item for item in result["retrieved"] if int(item["rank"]) <= TOP_K]
        if len(top) != TOP_K:
            raise RuntimeError(
                f"Question {question_index} has {len(top)} passages in the top {TOP_K}."
            )

        judgments = []
        for passage in top:
            key = _cache_key(question_index, int(passage["rank"]))
            cached = cache.get(key)
            if cached is None:
                verdict = _judge_passage(
                    client,
                    question,
                    evidence,
                    passage["text"],
                )
                cached = {
                    "question_index": question_index,
                    "rank": int(passage["rank"]),
                    "label": verdict["label"],
                    "missing": verdict["missing"],
                }
                _append_cache(CACHE_PATH, cached)
                cache[key] = cached
                print(
                    f"[{question_index}/90] rank {passage['rank']}: {cached['label']}",
                    flush=True,
                )
            judgments.append(
                {
                    "rank": int(passage["rank"]),
                    "label": cached["label"],
                    "missing": cached["missing"],
                    "chunk_id": passage.get("chunk_id"),
                }
            )

        judged.append(
            {
                "question": question,
                "strict_hit@5": bool(result["hit@5"]),
                "judgments": judgments,
            }
        )

    summary = _summarize(judged)
    report = {
        "judge_model": JUDGE_MODEL,
        "retrieval": "OpenAI parent-child",
        "k": TOP_K,
        "reference": "metadata.evidence_text",
        "labels": {
            "supports": "the passage states the reference evidence",
            "partial": "the passage states only part of the reference evidence",
            "miss": "the passage does not state the reference evidence",
        },
        "evidence_hit_rule": "any of the k passages is supports; partial is not a hit",
        "metrics": summary,
        "results": judged,
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
