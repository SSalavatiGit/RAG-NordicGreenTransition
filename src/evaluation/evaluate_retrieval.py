import json
from pathlib import Path
from statistics import mean

from src.retrieval.retriever import DenseRetriever


QA_PATH = Path("QA.txt")

TOP_K = 10

HIT_K_VALUES = [1, 3, 5, 10]


def load_qa(path: Path) -> list[dict]:
    """
    QA.txt is JSONL:
    one JSON object per line.
    """
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

    return examples


def page_in_range(result: dict, expected_page: int) -> bool:
    """
    A chunk is on the labeled page when that page falls inside
    its stored page_start–page_end span.
    """
    page_start = result.get("page_start")
    page_end = result.get("page_end")

    if page_start is None or page_end is None:
        return False

    try:
        return int(page_start) <= int(expected_page) <= int(page_end)
    except (TypeError, ValueError):
        return False


def is_relevant(
    result: dict,
    expected_document_id: str,
    expected_page: int,
) -> bool:
    """
    Baseline relevance rule:

    correct document
    AND
    labeled page inside the chunk page range.
    """
    if result.get("document_id") != expected_document_id:
        return False

    return page_in_range(result, expected_page)


def score_results(
    example: dict,
    results: list[dict],
    hit_k_values: list[int] | None = None,
) -> dict:
    question = example["question"]
    metadata = example["metadata"]

    expected_document_id = metadata["document_id"]
    expected_page = metadata["page"]
    ks = hit_k_values or HIT_K_VALUES

    relevant_ranks = []

    for result in results:
        if is_relevant(
            result=result,
            expected_document_id=expected_document_id,
            expected_page=expected_page,
        ):
            relevant_ranks.append(result["rank"])

    first_relevant_rank = (
        min(relevant_ranks)
        if relevant_ranks
        else None
    )

    reciprocal_rank = (
        1.0 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )

    hits = {
        f"hit@{k}": (
            first_relevant_rank is not None
            and first_relevant_rank <= k
        )
        for k in ks
    }

    return {
        "question": question,
        "expected_document_id": expected_document_id,
        "expected_page": expected_page,
        "first_relevant_rank": first_relevant_rank,
        "reciprocal_rank": reciprocal_rank,
        **hits,
        "retrieved": results,
    }


def precision_recall_f1(
    relevant_in_list: int,
    cutoff: int,
    total_relevant: int,
) -> dict[str, float]:
    """
    Set metrics at a cutoff.

    A retrieved unit is relevant under the same page-span rule as Hit@k.
    Precision divides by the cutoff. Recall divides by the number of
    relevant units in that method's index.
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive.")

    precision = relevant_in_list / cutoff
    if total_relevant <= 0:
        recall = 0.0
    else:
        recall = min(relevant_in_list, total_relevant) / total_relevant

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def mean_precision_recall_f1(
    evaluations: list[dict],
    relevant_totals: list[int],
    cutoffs: list[int] | None = None,
) -> dict[str, float]:
    """Macro-average precision, recall, and F1 at each cutoff."""
    if len(evaluations) != len(relevant_totals):
        raise ValueError("Each question needs one relevant-set size.")

    ks = cutoffs or HIT_K_VALUES
    totals = {
        f"precision@{k}": [] for k in ks
    }
    totals.update({f"recall@{k}": [] for k in ks})
    totals.update({f"f1@{k}": [] for k in ks})

    for evaluation, total_relevant in zip(evaluations, relevant_totals):
        retrieved = evaluation["retrieved"]
        for k in ks:
            relevant_in_list = 0
            for result in retrieved:
                rank = result.get("rank")
                if rank is not None and int(rank) > k:
                    continue
                if is_relevant(
                    result,
                    evaluation["expected_document_id"],
                    evaluation["expected_page"],
                ):
                    relevant_in_list += 1
            scores = precision_recall_f1(relevant_in_list, k, total_relevant)
            totals[f"precision@{k}"].append(scores["precision"])
            totals[f"recall@{k}"].append(scores["recall"])
            totals[f"f1@{k}"].append(scores["f1"])

    return {
        name: mean(values)
        for name, values in totals.items()
    }


def evaluate_example(
    retriever: DenseRetriever,
    example: dict,
    top_k: int = 10,
    exclude_noise: bool = False,
) -> dict:
    results = retriever.search(
        query=example["question"],
        top_k=top_k,
        exclude_noise=exclude_noise,
    )
    return score_results(example, results)


def evaluate_dataset(
    retriever: DenseRetriever,
    qa_examples: list[dict],
) -> dict:
    evaluations = []

    for i, example in enumerate(qa_examples, start=1):
        print(
            f"[{i}/{len(qa_examples)}] "
            f"{example['question'][:80]}"
        )

        result = evaluate_example(
            retriever=retriever,
            example=example,
            top_k=TOP_K,
            exclude_noise=False,
        )

        evaluations.append(result)

    metrics = {}

    for k in HIT_K_VALUES:
        metric_name = f"hit@{k}"

        metrics[metric_name] = mean(
            int(item[metric_name])
            for item in evaluations
        )

    metrics["mrr"] = mean(
        item["reciprocal_rank"]
        for item in evaluations
    )

    metrics["num_questions"] = len(evaluations)

    return {
        "metrics": metrics,
        "results": evaluations,
    }


def save_results(report: dict) -> None:
    output_dir = Path("data/evaluation")
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_path = output_dir / "retrieval_baseline.json"

    with full_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    failures_path = (
        output_dir
        / "retrieval_failures.jsonl"
    )

    with failures_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for item in report["results"]:
            if not item["hit@5"]:
                f.write(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print()
    print(f"Saved full report to: {full_path}")
    print(f"Saved Hit@5 failures to: {failures_path}")


def print_metrics(metrics: dict) -> None:
    print()
    print("=" * 40)
    print("Retrieval Evaluation")
    print("=" * 40)

    print(
        f"Questions: {metrics['num_questions']}"
    )

    for k in HIT_K_VALUES:
        score = metrics[f"hit@{k}"]

        print(
            f"Hit@{k:<2}: {score:.2%}"
        )

    print(
        f"MRR   : {metrics['mrr']:.4f}"
    )


def main():
    qa_examples = load_qa(QA_PATH)

    print(
        f"Loaded {len(qa_examples)} QA examples."
    )

    retriever = DenseRetriever()

    report = evaluate_dataset(
        retriever=retriever,
        qa_examples=qa_examples,
    )

    print_metrics(
        report["metrics"]
    )

    save_results(report)


if __name__ == "__main__":
    main()