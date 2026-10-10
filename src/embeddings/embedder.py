from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseEmbedder:
    """
    Dense document embedder for the first RAG baseline.

    Baseline model:
        BAAI/bge-small-en-v1.5

    The model name is configurable so later experiments can swap the
    embedding model without changing the rest of the pipeline.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        self.model = SentenceTransformer(model_name)

        dimension = self.model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError(
                "Could not determine embedding dimension."
            )

        self.dimension = int(dimension)

    def encode_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Encode corpus chunks.

        normalize_embeddings=True makes every vector unit length.
        This is convenient for later cosine-similarity retrieval.
        """
        embeddings = self.model.encode_document(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embeddings
