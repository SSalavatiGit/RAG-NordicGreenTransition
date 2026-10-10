from __future__ import annotations

from src.config import settings
from src.utils.chunk_loader import load_chunks
from src.vectorstore.qdrant_index import QdrantIndex


def main() -> None:
    chunks = load_chunks(settings.chunks_path)
    print(f"Loaded {len(chunks):,} chunks from {settings.chunks_path}")

    index = QdrantIndex(
        path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        vector_size=384,
    )

    updated = index.set_page_ranges(chunks)
    print(f"Updated page ranges on {updated:,} points")
    print(f"Collection still contains {index.count():,} points")

    index.client.close()


if __name__ == "__main__":
    main()
