from collections import Counter

from langdetect import detect, DetectorFactory

from src.config import settings
from src.utils.chunk_loader import load_chunks

DetectorFactory.seed = 0


def main():
    chunks = load_chunks(settings.chunks_path)

    counts = Counter()
    non_english = []

    for chunk in chunks:
        text = chunk["text"]

        try:
            lang = detect(text)
        except Exception:
            lang = "unknown"

        counts[lang] += 1

        if lang != "en":
            non_english.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "lang": lang,
                    "text": text[:250].replace("\n", " "),
                }
            )

    print(f"Total chunks: {len(chunks)}")
    print("\nLanguage counts:")

    for lang, count in counts.most_common():
        percentage = 100 * count / len(chunks)
        print(f"{lang:10s} {count:5d}  {percentage:6.2f}%")

    print(f"\nNon-English chunks: {len(non_english)}")

    print("\nFirst 20 non-English chunks:")
    for item in non_english[:20]:
        print("=" * 80)
        print(f"{item['chunk_id']} | language={item['lang']}")
        print(item["text"])


if __name__ == "__main__":
    main()