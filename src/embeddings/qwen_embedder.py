from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

# Official Qwen3 retrieval instruction. Passages are embedded without it.
QUERY_PROMPT = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\nQuery:"
)


class QwenEmbedder:
    """Local Qwen3-Embedding-0.6B. 1,024 dimensions."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

        prompts = dict(getattr(self.model, "prompts", {}) or {})
        if not prompts.get("query"):
            prompts["query"] = QUERY_PROMPT
        prompts.setdefault("document", "")
        self.model.prompts = prompts

        dimension = self.model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Could not determine the Qwen embedding dimension.")
        self.dimension = int(dimension)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            prompt_name="document",
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape != (len(texts), self.dimension):
            raise RuntimeError(
                f"Expected embeddings {(len(texts), self.dimension)}, "
                f"got {matrix.shape}."
            )
        return matrix

    def encode_query(self, query: str) -> list[float]:
        vector = self.model.encode(
            query,
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim == 2:
            array = array[0]
        return array.tolist()
