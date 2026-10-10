from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("An embedding vector has length zero.")
    return matrix / norms


class DailyQuotaExceeded(RuntimeError):
    """The provider rejected the request until a daily quota resets."""


def _retry(call, attempts: int = 6):
    delay = 2.0
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            text = str(exc).lower()
            if "free_tier" in text or (
                "429" in text and "retry in" in text and "h" in text
            ):
                raise DailyQuotaExceeded(
                    "Daily embedding quota is exhausted."
                ) from exc
            retryable = status in {429, 500, 503} or any(
                token in text
                for token in ("429", "resource_exhausted", "unavailable", "rate limit")
            )
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2


class OpenAIEmbedder:
    """text-embedding-3-large. Queries and passages use the same endpoint."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        batch_size: int = 64,
        dimension: int = 3072,
    ) -> None:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing from .env.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = dimension
        self.client = OpenAI(api_key=api_key)

    def encode_documents(
        self,
        texts: Sequence[str],
        cache_path: str | None = None,
    ) -> np.ndarray:
        return self._encode(list(texts), cache_path=cache_path)

    def encode_query(self, query: str) -> list[float]:
        return self._encode([query])[0].tolist()

    def _encode(
        self,
        texts: list[str],
        cache_path: str | None = None,
    ) -> np.ndarray:
        cache = Path(cache_path) if cache_path else None
        cached = None
        if cache is not None and cache.exists():
            cached = np.load(cache)
            if cached.ndim != 2 or cached.shape[1] != self.dimension:
                raise RuntimeError(
                    f"Cache {cache} has shape {cached.shape}."
                )
            if cached.shape[0] > len(texts):
                raise RuntimeError(
                    f"Cache {cache} has {cached.shape[0]} rows "
                    f"for {len(texts)} texts."
                )

        done = 0 if cached is None else int(cached.shape[0])
        parts: list[np.ndarray] = [] if cached is None else [cached]
        pending = texts[done:]

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            response = _retry(
                lambda batch=batch: self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    dimensions=self.dimension,
                )
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            batch_matrix = _normalize(
                np.asarray(
                    [item.embedding for item in ordered],
                    dtype=np.float32,
                )
            )
            parts.append(batch_matrix)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, np.vstack(parts))
            if len(texts) > 1:
                completed = done + start + len(batch)
                print(
                    f"  embedded {completed:,}/{len(texts):,}",
                    flush=True,
                )

        matrix = np.vstack(parts)
        if matrix.shape != (len(texts), self.dimension):
            raise RuntimeError(
                f"Expected embeddings {(len(texts), self.dimension)}, "
                f"got {matrix.shape}."
            )
        return matrix


class GeminiEmbedder:
    """Gemini embeddings.

    gemini-embedding-001 uses task_type.
    gemini-embedding-2 uses the documented retrieval prompt instead.
    """

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        batch_size: int = 32,
        dimension: int = 3072,
    ) -> None:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing from .env.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = dimension
        self._uses_task_type = model_name == "gemini-embedding-001"
        self._types = types
        self.client = genai.Client(api_key=api_key)

    def encode_documents(
        self,
        texts: Sequence[str],
        cache_path: str | None = None,
    ) -> np.ndarray:
        prepared = [
            text if self._uses_task_type else f"title: none | text: {text}"
            for text in texts
        ]
        task_type = "RETRIEVAL_DOCUMENT" if self._uses_task_type else None
        return self._encode(prepared, task_type=task_type, cache_path=cache_path)

    def encode_query(self, query: str) -> list[float]:
        if self._uses_task_type:
            prepared = query
            task_type = "RETRIEVAL_QUERY"
        else:
            prepared = f"task: search result | query: {query}"
            task_type = None
        return self._encode([prepared], task_type=task_type)[0].tolist()

    def _encode(
        self,
        texts: list[str],
        task_type: str | None,
        cache_path: str | None = None,
    ) -> np.ndarray:
        cache = Path(cache_path) if cache_path else None
        cached = None
        if cache is not None and cache.exists():
            cached = np.load(cache)
            if cached.shape[1] != self.dimension or cached.shape[0] > len(texts):
                raise RuntimeError(
                    f"Cache {cache} has shape {cached.shape}."
                )

        done = 0 if cached is None else int(cached.shape[0])
        parts: list[np.ndarray] = [] if cached is None else [cached]
        pending = texts[done:]

        config_args: dict = {"output_dimensionality": self.dimension}
        if task_type is not None:
            config_args["task_type"] = task_type

        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            # A list of strings is one combined embedding on gemini-embedding-2.
            # Each text has to be its own Content to get one vector per chunk.
            if self._uses_task_type:
                contents = batch
            else:
                contents = [
                    self._types.Content(
                        parts=[self._types.Part(text=text)],
                    )
                    for text in batch
                ]
            response = _retry(
                lambda contents=contents: self.client.models.embed_content(
                    model=self.model_name,
                    contents=contents,
                    config=self._types.EmbedContentConfig(**config_args),
                )
            )
            embeddings = response.embeddings or []
            if len(embeddings) != len(batch):
                raise RuntimeError(
                    f"Gemini returned {len(embeddings)} embeddings "
                    f"for {len(batch)} texts."
                )
            batch_matrix = _normalize(
                np.asarray(
                    [item.values for item in embeddings],
                    dtype=np.float32,
                )
            )
            parts.append(batch_matrix)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, np.vstack(parts))
            if len(texts) > 1:
                completed = done + start + len(batch)
                print(f"  embedded {completed:,}/{len(texts):,}", flush=True)

        matrix = np.vstack(parts)
        if matrix.shape != (len(texts), self.dimension):
            raise RuntimeError(
                f"Expected embeddings {(len(texts), self.dimension)}, "
                f"got {matrix.shape}."
            )
        return matrix
