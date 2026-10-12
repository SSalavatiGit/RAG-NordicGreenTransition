"""Answer a question from the five retrieved parents.

The model sees the question and the passages only. Labeled answers and
evidence text stay out of the prompt. Citations are passage ranks; document
ids and page spans are copied from the retrieved parents.
"""

from __future__ import annotations

import json
from typing import Any

from src.embeddings.api_embedder import _retry

GENERATION_MODEL = "gpt-4.1-mini-2025-04-14"

SYSTEM_PROMPT = """You answer questions about Nordic green-transition reports.

Use only the passages. Do not add facts from outside them.
If the passages do not state the answer, say so in one sentence and return no citations.
When a passage supports a claim, cite that passage by its rank, document id, and page span.
"""

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "grounded_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "rank": {"type": "integer"},
                            "document_id": {"type": "string"},
                            "page_start": {"type": "integer"},
                            "page_end": {"type": "integer"},
                        },
                        "required": [
                            "rank",
                            "document_id",
                            "page_start",
                            "page_end",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["answer", "citations"],
            "additionalProperties": False,
        },
    },
}


def format_passages(passages: list[dict[str, Any]]) -> str:
    blocks = []
    for passage in passages:
        blocks.append(
            f"[{passage['rank']}] {passage['document_id']}, "
            f"pages {passage['page_start']}-{passage['page_end']}\n"
            f"{passage['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(
    client,
    question: str,
    passages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return an answer and the passages it cites."""
    if not passages:
        raise ValueError("Generation needs at least one passage.")

    response = _retry(
        lambda: client.chat.completions.create(
            model=GENERATION_MODEL,
            temperature=0,
            response_format=SCHEMA,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Passages:\n{format_passages(passages)}"
                    ),
                },
            ],
        )
    )
    payload = json.loads(response.choices[0].message.content)
    answer = str(payload["answer"]).strip()
    if not answer:
        raise RuntimeError("The model returned an empty answer.")

    by_rank = {int(passage["rank"]): passage for passage in passages}
    citations = []
    seen: set[int] = set()
    for item in payload["citations"]:
        rank = int(item["rank"])
        if rank not in by_rank or rank in seen:
            continue
        source = by_rank[rank]
        citations.append(
            {
                "rank": rank,
                "document_id": source["document_id"],
                "page_start": int(source["page_start"]),
                "page_end": int(source["page_end"]),
            }
        )
        seen.add(rank)

    return {"answer": answer, "citations": citations}
