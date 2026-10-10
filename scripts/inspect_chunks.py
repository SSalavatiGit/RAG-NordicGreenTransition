from __future__ import annotations

import random

from src.config import settings
from src.utils.chunk_loader import load_chunks


def main() -> None:
    chunks = load_chunks(settings.chunks_path)

    print(f"Loaded {len(chunks):,} chunks")
    print()

    sample_size = min(5, len(chunks))

    for chunk in random.sample(chunks, sample_size):
        print("=" * 90)
        print("chunk_id:", chunk["chunk_id"])
        print("metadata:", chunk["metadata"])
        print()
        print(chunk["text"][:1200])
        print()


if __name__ == "__main__":
    main()
