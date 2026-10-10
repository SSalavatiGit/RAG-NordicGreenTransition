from __future__ import annotations

from typing import Any

from sentence_transformers import CrossEncoder

from src.config import settings


class CrossEncoderReranker:
    """
    Reorders a dense-retrieval candidate pool.

    The dense score finds the neighborhood. This model scores how directly
    each passage answers the question, then the evaluation uses the new order.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        passages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not passages:
            return []

        pairs = [(query, passage.get("text", "")) for passage in passages]
        scores = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        ordered = sorted(
            zip(scores, passages, strict=True),
            key=lambda item: float(item[0]),
            reverse=True,
        )

        reranked = []
        for rank, (score, passage) in enumerate(ordered, start=1):
            item = dict(passage)
            item["dense_score"] = item.get("score")
            item["dense_rank"] = item.get("rank")
            item["score"] = float(score)
            item["rank"] = rank
            reranked.append(item)

        return reranked
