from __future__ import annotations

from src.config import settings
from src.embeddings.embedder import DenseEmbedder
from src.utils.chunk_loader import load_chunks
from src.vectorstore.qdrant_index import QdrantIndex


def main() -> None:
    print("STEP 1 — Load existing chunks")
    print(f"Input: {settings.chunks_path}")

    chunks = load_chunks(settings.chunks_path)

    print(f"Loaded: {len(chunks):,} chunks")
    print()

    print("STEP 2 — Load embedding model")
    embedder = DenseEmbedder(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )

    print(f"Model:     {embedder.model_name}")
    print(f"Dimension: {embedder.dimension}")
    print()

    print("STEP 3 — Embed chunk text")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.encode_documents(texts)

    print(f"Embedding matrix shape: {embeddings.shape}")
    print()

    print("STEP 4 — Create Qdrant collection")
    index = QdrantIndex(
        path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        vector_size=embedder.dimension,
    )

    index.recreate_collection()

    print(f"Collection: {settings.qdrant_collection}")
    print("Distance metric configured for future retrieval: cosine")
    print()

    print("STEP 5 — Upload embeddings + text + metadata")
    index.upload(
        chunks=chunks,
        embeddings=embeddings,
        batch_size=settings.qdrant_upload_batch_size,
    )

    print()
    print("INDEXING COMPLETE")
    print(f"Qdrant points: {index.count():,}")
    print()
    print(
        "No query search has been performed. "
        "The corpus is now ready for the retrieval stage."
    )


if __name__ == "__main__":
    main()
