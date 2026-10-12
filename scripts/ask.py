"""Ask one new question through the selected RAG setup.

Search the OpenAI parent-child index, rerank 20 parents with Jev Score,
and answer from the top 5. This does not read the saved 90-question file.

Example:
uv run python -u -m scripts.ask "Why can banning non-essential uses of critical raw materials improve supply security?"
"""

from __future__ import annotations

import argparse
import json
import os

from dotenv import load_dotenv

from src.embeddings.api_embedder import OpenAIEmbedder
from src.generation.answer import generate_answer
from src.retrieval.jev_reranker import jev_client, rerank, score_passage
from src.retrieval.structured_retriever import StructuredRetriever

load_dotenv()

COLLECTION = "nordic_green_transition_openai_parent_child"
SHORTLIST = 20
CHILD_POOL = 200
TOP_K = 5


def ask(question: str) -> dict:
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing from .env.")

    retriever = StructuredRetriever(
        embedder=OpenAIEmbedder(),
        collection_name=COLLECTION,
        mode="parent",
    )
    jev = jev_client()
    try:
        shortlist = retriever.search(
            question,
            top_k=SHORTLIST,
            candidate_pool=CHILD_POOL,
        )
        scores = [
            score_passage(jev, question, passage["text"])
            for passage in shortlist
        ]
        top = rerank(shortlist, scores)[:TOP_K]
        generated = generate_answer(OpenAI(api_key=api_key), question, top)
    finally:
        retriever.close()
        jev.close()

    return {
        "question": question,
        "answer": generated["answer"],
        "citations": generated["citations"],
        "passages": [
            {
                "rank": passage["rank"],
                "document_id": passage["document_id"],
                "page_start": passage["page_start"],
                "page_end": passage["page_end"],
                "jev_score": passage["score"],
                "chunk_id": passage.get("chunk_id"),
            }
            for passage in top
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask one question with parent-child retrieval, Jev Score, and generation."
    )
    parser.add_argument("question", help="The question to ask.")
    args = parser.parse_args()

    result = ask(args.question)
    print(result["answer"], flush=True)
    print(flush=True)
    for passage in result["passages"]:
        print(
            f"[{passage['rank']}] {passage['document_id']} "
            f"pages {passage['page_start']}-{passage['page_end']} "
            f"jev {passage['jev_score']:.2f}",
            flush=True,
        )
    print(flush=True)
    print(json.dumps(result["citations"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
