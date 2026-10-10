"""Add macro precision, recall, and F1 to the saved retrieval comparison.

Recall uses every indexed unit whose page span contains the labeled page.
Parent-child is scored on the returned parents, so its relevant set is the
fixed-size chunks. The other OpenAI chunking runs use their own units.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import settings
from src.evaluation.evaluate_retrieval import (
    QA_PATH,
    is_relevant,
    load_qa,
    mean_precision_recall_f1,
)
from src.utils.chunk_loader import load_chunks

OUTPUT_DIR = Path("data/evaluation")
COMPARISON_PATH = OUTPUT_DIR / "chunking_method_comparison.json"

FIXED_METHODS = {
    "BGE-small fixed",
    "Qwen3-0.6B fixed",
    "Gemini embedding-2 fixed",
    "OpenAI fixed 500/75",
    "BGE E1 filter",
    "BGE E2 top-20",
    "BGE E3 reranker",
    "OpenAI parent-child",
}

REPORTS = {
    "BGE-small fixed": OUTPUT_DIR / "retrieval_baseline.json",
    "Qwen3-0.6B fixed": OUTPUT_DIR / "retrieval_qwen3_0_6b.json",
    "Gemini embedding-2 fixed": OUTPUT_DIR / "retrieval_gemini_002.json",
    "OpenAI fixed 500/75": OUTPUT_DIR / "retrieval_openai_3_large.json",
    "BGE E1 filter": OUTPUT_DIR / "e1_structural_filter.json",
    "BGE E2 top-20": OUTPUT_DIR / "e2_dense_top20.json",
    "BGE E3 reranker": OUTPUT_DIR / "e3_reranker_top10.json",
    "OpenAI sentence-window": OUTPUT_DIR / "retrieval_openai_sentence_window.json",
    "OpenAI semantic": OUTPUT_DIR / "retrieval_openai_semantic.json",
    "OpenAI parent-child": OUTPUT_DIR / "retrieval_openai_parent_child.json",
}


def _as_result(unit: dict) -> dict:
    metadata = unit.get("metadata") or {}
    return {
        "document_id": metadata.get("document_id", unit.get("document_id")),
        "page_start": metadata.get("page_start", unit.get("page_start")),
        "page_end": metadata.get("page_end", unit.get("page_end")),
    }


def _relevant_totals(units: list[dict], examples: list[dict]) -> list[int]:
    totals = []
    for example in examples:
        expected = example["metadata"]
        totals.append(
            sum(
                1
                for unit in units
                if is_relevant(
                    _as_result(unit),
                    expected["document_id"],
                    expected["page"],
                )
            )
        )
    return totals


def main() -> None:
    examples = load_qa(QA_PATH)
    fixed_units = load_chunks(settings.chunks_path)
    sentence_units = load_chunks("data/processed/chunks/sentence_windows.jsonl")
    semantic_units = load_chunks("data/processed/chunks/semantic_chunks.jsonl")

    catalogs = {
        "fixed": _relevant_totals(fixed_units, examples),
        "sentence": _relevant_totals(sentence_units, examples),
        "semantic": _relevant_totals(semantic_units, examples),
    }
    for name, totals in catalogs.items():
        ordered = sorted(totals)
        middle = ordered[len(ordered) // 2]
        print(
            f"{name}: relevant units min {min(totals)}, "
            f"median {middle}, max {max(totals)}"
        )

    comparison = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    for row in comparison["methods"]:
        name = row["name"]
        report = json.loads(REPORTS[name].read_text(encoding="utf-8"))
        if name == "OpenAI sentence-window":
            totals = catalogs["sentence"]
            relevant_set = "sentence windows whose span contains the labeled page"
        elif name == "OpenAI semantic":
            totals = catalogs["semantic"]
            relevant_set = "semantic chunks whose span contains the labeled page"
        elif name in FIXED_METHODS:
            totals = catalogs["fixed"]
            relevant_set = "fixed-size chunks whose span contains the labeled page"
        else:
            raise KeyError(name)

        scores = mean_precision_recall_f1(report["results"], totals)
        row["metrics"].update(scores)
        row["relevant_set"] = relevant_set
        print(
            f"{name:<28} "
            f"P@10 {scores['precision@10']:.2%}  "
            f"R@10 {scores['recall@10']:.2%}  "
            f"F1@10 {scores['f1@10']:.2%}"
        )

    comparison["precision_recall"] = (
        "Macro-average over 90 questions. A unit is relevant when document_id "
        "matches and the labeled page lies inside page_start–page_end. "
        "Precision@k divides by k. Recall@k divides by the number of relevant "
        "units in that method's index. F1 is the harmonic mean of those two, "
        "averaged per question."
    )
    COMPARISON_PATH.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Updated {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
