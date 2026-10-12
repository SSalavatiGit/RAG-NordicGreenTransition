"""Reorder a retrieval shortlist with Jev Score.

Each passage is scored alone against the question. The levels run from
"does not answer" to "answers," and the returned score is the position
on that scale. Higher scores are ranked first.
"""

from __future__ import annotations

import os
from typing import Any

from typesafe_sdk import Score, TypeSafeClient

JEV_MODEL = "jev-1.13.0"
RELEVANCE_LEVELS = [
    "Does not answer the question.",
    "Same topic, but does not answer the question.",
    "Answers part of the question.",
    "Answers the question.",
]


def jev_client() -> TypeSafeClient:
    api_key = os.getenv("JEV_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("JEV_API_KEY is missing from .env.")
    return TypeSafeClient(api_key=api_key, model=JEV_MODEL, timeout=30.0)


def score_passage(client: TypeSafeClient, question: str, passage: str) -> float:
    response = client.system_one(
        state={"question": question, "passage": passage},
        questions={
            "relevance": Score(
                instructions="How well does this passage answer the question?",
                criteria=RELEVANCE_LEVELS,
            )
        },
        model=JEV_MODEL,
    )
    return float(response.scores["relevance"].score)


def rerank(passages: list[dict[str, Any]], scores: list[float]) -> list[dict[str, Any]]:
    if len(passages) != len(scores):
        raise ValueError("Each passage needs one Jev score.")
    ordered = sorted(
        zip(scores, passages),
        key=lambda item: item[0],
        reverse=True,
    )
    reranked = []
    for rank, (score, passage) in enumerate(ordered, start=1):
        item = dict(passage)
        item["dense_rank"] = passage.get("rank")
        item["dense_score"] = passage.get("score")
        item["rank"] = rank
        item["score"] = score
        reranked.append(item)
    return reranked
