from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pymupdf
import requests


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCES_PATH = PROJECT_ROOT / "Sources.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------
# Starter corpus
# ---------------------------------------------------------

STARTER_DOCUMENT_IDS = [
    "doc_002",
    "doc_004",
    "doc_010",
    "doc_013",
    "doc_014",
    "doc_015",
    "doc_016",
    "doc_018",
    "doc_022",
]


# ---------------------------------------------------------
# Load manifest
# ---------------------------------------------------------

def load_sources(
    path: Path = SOURCES_PATH,
) -> list[dict[str, Any]]:
    """
    Load and validate Sources.json.
    """

    with path.open("r", encoding="utf-8") as f:
        sources = json.load(f)

    if not isinstance(sources, list):
        raise ValueError(
            "Sources.json must contain a JSON list."
        )

    required_fields = {
        "document_id",
        "title",
        "year",
        "organization",
        "country",
        "topic",
        "pdf_url",
        "file_path",
    }

    for i, source in enumerate(sources):
        missing = required_fields - source.keys()

        if missing:
            raise ValueError(
                f"Source #{i} "
                f"({source.get('document_id', 'unknown')}) "
                f"is missing fields: {sorted(missing)}"
            )

    return sources


# ---------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Conservative cleanup suitable for RAG preprocessing.
    """

    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")

    # Normalize spaces/tabs but preserve line breaks
    text = re.sub(r"[ \t]+", " ", text)

    # Strip individual lines
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    # Avoid many consecutive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------
# Download PDF
# ---------------------------------------------------------

def download_pdf(
    url: str,
    destination: Path,
    timeout: int = 60,
) -> Path:
    """
    Download PDF only if it does not already exist.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):
        print("  ↳ PDF already exists, skipping download.")
        return destination

    response = requests.get(
        url,
        timeout=timeout,
    )

    response.raise_for_status()

    content_type = (
        response.headers
        .get("content-type", "")
        .lower()
    )

    if (
        "pdf" not in content_type
        and not response.content.startswith(b"%PDF")
    ):
        raise ValueError(
            f"URL does not appear to return a PDF: {url}"
        )

    destination.write_bytes(
        response.content
    )

    print("  ↳ PDF downloaded.")

    return destination


# ---------------------------------------------------------
# Parse PDF
# ---------------------------------------------------------

def parse_pdf(
    pdf_path: Path,
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract page-level text and metadata from a PDF.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_index, page in enumerate(document):
        raw_text = page.get_text("text")

        text = clean_text(raw_text)

        pages.append(
            {
                "page_number": page_index + 1,
                "text": text,
                "char_count": len(text),
            }
        )

    result = {
        "document_id":
            source_metadata["document_id"],

        "metadata": {
            "publication_number":
                source_metadata.get(
                    "publication_number"
                ),

            "title":
                source_metadata["title"],

            "year":
                source_metadata["year"],

            "organization":
                source_metadata["organization"],

            "country":
                source_metadata["country"],

            "topic":
                source_metadata["topic"],

            "secondary_topics":
                source_metadata.get(
                    "secondary_topics"
                ),

            "source_url":
                source_metadata.get(
                    "source_url"
                ),

            "pdf_url":
                source_metadata["pdf_url"],

            "license":
                source_metadata.get(
                    "license"
                ),

            "language":
                source_metadata.get(
                    "language"
                ),
        },

        "parser": {
            "name": "pymupdf",
            "page_count": len(pages),
        },

        "pages": pages,
    }

    document.close()

    return result


# ---------------------------------------------------------
# Save parsed JSON
# ---------------------------------------------------------

def save_parsed_document(
    parsed_document: dict[str, Any],
    output_dir: Path = PROCESSED_DIR,
) -> Path:
    """
    Save parsed output as JSON.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_id = parsed_document[
        "document_id"
    ]

    output_path = (
        output_dir
        / f"{document_id}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            parsed_document,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


# ---------------------------------------------------------
# Process one source
# ---------------------------------------------------------

def process_source(
    source: dict[str, Any],
    download: bool = True,
) -> Path:
    """
    Download, parse, and save one document.
    """

    pdf_path = (
        PROJECT_ROOT
        / source["file_path"]
    )

    if download:
        download_pdf(
            url=source["pdf_url"],
            destination=pdf_path,
        )

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    parsed = parse_pdf(
        pdf_path=pdf_path,
        source_metadata=source,
    )

    return save_parsed_document(
        parsed
    )


# ---------------------------------------------------------
# Process selected corpus
# ---------------------------------------------------------

def process_all_sources(
    document_ids: list[str] | None = None,
    limit: int | None = None,
) -> None:
    """
    Process selected approved documents.
    """

    sources = load_sources()

    approved_sources = [
        source
        for source in sources
        if source.get("status") == "approved"
    ]

    selected_sources = approved_sources

    # Select specific document IDs
    if document_ids:
        wanted_ids = set(document_ids)

        selected_sources = [
            source
            for source in selected_sources
            if source["document_id"] in wanted_ids
        ]

        # Preserve requested order
        source_lookup = {
            source["document_id"]: source
            for source in selected_sources
        }

        selected_sources = [
            source_lookup[doc_id]
            for doc_id in document_ids
            if doc_id in source_lookup
        ]

    # Optional numeric limit
    if limit is not None:
        selected_sources = (
            selected_sources[:limit]
        )

    print(
        f"Selected {len(selected_sources)} "
        f"documents from "
        f"{len(approved_sources)} approved sources."
    )

    print("\nDocuments:")

    for source in selected_sources:
        print(
            f"  - {source['document_id']}: "
            f"{source['title']}"
        )

    successes = []
    failures = []

    for source in selected_sources:
        document_id = source[
            "document_id"
        ]

        try:
            print(
                f"\nParsing {document_id}: "
                f"{source['title']}"
            )

            output_path = (
                process_source(source)
            )

            successes.append(
                document_id
            )

            print(
                f"  ✓ Saved to {output_path}"
            )

        except Exception as exc:
            failures.append(
                {
                    "document_id":
                        document_id,

                    "error":
                        str(exc),
                }
            )

            print(
                f"  ✗ Failed: {exc}"
            )

    print(
        "\n--- Parsing summary ---"
    )

    print(
        f"Successful: {len(successes)}"
    )

    print(
        f"Failed:     {len(failures)}"
    )

    if failures:
        print("\nFailures:")

        for failure in failures:
            print(
                f"- "
                f"{failure['document_id']}: "
                f"{failure['error']}"
            )


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Download and parse "
            "Nordic RAG documents."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of "
            "documents to process."
        ),
    )

    parser.add_argument(
        "--document-ids",
        nargs="+",
        default=None,
        help=(
            "Specific document IDs, "
            "e.g. doc_002 doc_013"
        ),
    )

    parser.add_argument(
        "--starter-corpus",
        action="store_true",
        help=(
            "Use the predefined "
            "9-document starter corpus."
        ),
    )

    args = parser.parse_args()

    document_ids = args.document_ids

    if args.starter_corpus:
        document_ids = (
            STARTER_DOCUMENT_IDS
        )

    process_all_sources(
        document_ids=document_ids,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()