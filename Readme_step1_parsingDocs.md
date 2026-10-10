# Step 1 — Document Ingestion, Parsing, and Validation

This step builds the first stage of the Nordic Green Transition RAG pipeline:

```text
Sources.json
    ↓
Select starter corpus
    ↓
Download PDFs
    ↓
Parse PDF text page-by-page
    ↓
Save structured JSON
    ↓
Run pytest validation
```

The goal of this step is to create a small, reproducible, and testable document corpus before moving to cleaning, chunking, embeddings, or retrieval.

---

## 1. Starter corpus

Instead of using all documents in `Sources.json`, the first RAG version uses a smaller, thematically related corpus of 9 circular-economy/materials reports:

```text
doc_002
doc_004
doc_010
doc_013
doc_014
doc_015
doc_016
doc_018
doc_022
```

This keeps the first RAG experiment small enough to inspect manually while still supporting multi-document retrieval.

The downloaded PDFs are stored in:

```text
data/raw/
```

The parsed outputs are stored in:

```text
data/processed/
```

After running the parser, the project contained:

```text
data/
├── raw/
│   ├── doc_002_....pdf
│   ├── doc_004_....pdf
│   ├── doc_010_....pdf
│   ├── doc_013_....pdf
│   ├── doc_014_....pdf
│   ├── doc_015_....pdf
│   ├── doc_016_....pdf
│   ├── doc_018_....pdf
│   └── doc_022_....pdf
│
└── processed/
    ├── doc_002.json
    ├── doc_004.json
    ├── doc_010.json
    ├── doc_013.json
    ├── doc_014.json
    ├── doc_015.json
    ├── doc_016.json
    ├── doc_018.json
    └── doc_022.json
```

---

# 2. Terminal commands used

## Run the test suite

```powershell
uv run pytest -v
```

### Goal

Run all tests in the `tests/` directory using the Python environment managed by `uv`.

The `-v` flag means **verbose**, so pytest prints the name and result of every individual test.

Example:

```text
tests/test_parsing.py::test_sources_json_loads PASSED
tests/test_parsing.py::test_document_ids_are_unique PASSED
```

---

## Run the document parser

```powershell
uv run python -m src.parsing.parse_documents --starter-corpus
```

### Goal

Run the parser as a Python module and process only the predefined starter corpus.

The command performs:

```text
Load Sources.json
        ↓
Keep approved documents
        ↓
Select STARTER_DOCUMENT_IDS
        ↓
Download missing PDFs
        ↓
Parse each PDF
        ↓
Save parsed JSON
```

The final run produced:

```text
Selected 9 documents from 22 approved sources.

--- Parsing summary ---
Successful: 9
Failed:     0
```

---

## Optional command: process a limited number

```powershell
uv run python -m src.parsing.parse_documents --starter-corpus --limit 5
```

### Goal

Process only the first 5 documents from the starter corpus.

This is useful for fast experiments or debugging.

---

## Optional command: select documents manually

```powershell
uv run python -m src.parsing.parse_documents --document-ids doc_002 doc_013 doc_014
```

### Goal

Process only explicitly selected document IDs.

---

# 3. Python package setup

To make imports such as:

```python
from src.parsing.parse_documents import load_sources
```

work correctly, these files were added:

```text
src/
├── __init__.py
└── parsing/
    ├── __init__.py
    └── parse_documents.py
```

The `__init__.py` files mark these directories as Python packages.

---

# 4. pytest configuration

The following configuration was added to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

## Purpose

### `pythonpath = ["."]`

Adds the project root to Python's import path so pytest can import:

```python
src.parsing.parse_documents
```

### `testpaths = ["tests"]`

Tells pytest to discover tests inside:

```text
tests/
```

---

# 5. PDF parser

File:

```text
src/parsing/parse_documents.py
```

The parser uses:

```python
import pymupdf
```

PyMuPDF is used for extracting text from PDFs page-by-page.

## Complete parser code

```python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pymupdf
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCES_PATH = PROJECT_ROOT / "Sources.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


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


def load_sources(
    path: Path = SOURCES_PATH,
) -> list[dict[str, Any]]:
    """Load and validate Sources.json."""

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


def clean_text(text: str) -> str:
    """Conservative cleanup suitable for RAG preprocessing."""

    text = text.replace("\u00a0", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)

    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def download_pdf(
    url: str,
    destination: Path,
    timeout: int = 60,
) -> Path:
    """Download PDF only if it does not already exist."""

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

    destination.write_bytes(response.content)

    print("  ↳ PDF downloaded.")

    return destination


def parse_pdf(
    pdf_path: Path,
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Extract page-level text and metadata from a PDF."""

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
        "document_id": source_metadata["document_id"],

        "metadata": {
            "publication_number":
                source_metadata.get("publication_number"),
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
                source_metadata.get("secondary_topics"),
            "source_url":
                source_metadata.get("source_url"),
            "pdf_url":
                source_metadata["pdf_url"],
            "license":
                source_metadata.get("license"),
            "language":
                source_metadata.get("language"),
        },

        "parser": {
            "name": "pymupdf",
            "page_count": len(pages),
        },

        "pages": pages,
    }

    document.close()

    return result


def save_parsed_document(
    parsed_document: dict[str, Any],
    output_dir: Path = PROCESSED_DIR,
) -> Path:
    """Save parsed output as JSON."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    document_id = parsed_document["document_id"]

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


def process_source(
    source: dict[str, Any],
    download: bool = True,
) -> Path:
    """Download, parse, and save one document."""

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

    return save_parsed_document(parsed)


def process_all_sources(
    document_ids: list[str] | None = None,
    limit: int | None = None,
) -> None:
    """Process selected approved documents."""

    sources = load_sources()

    approved_sources = [
        source
        for source in sources
        if source.get("status") == "approved"
    ]

    selected_sources = approved_sources

    if document_ids:
        wanted_ids = set(document_ids)

        selected_sources = [
            source
            for source in selected_sources
            if source["document_id"] in wanted_ids
        ]

        source_lookup = {
            source["document_id"]: source
            for source in selected_sources
        }

        selected_sources = [
            source_lookup[doc_id]
            for doc_id in document_ids
            if doc_id in source_lookup
        ]

    if limit is not None:
        selected_sources = selected_sources[:limit]

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
        document_id = source["document_id"]

        try:
            print(
                f"\nParsing {document_id}: "
                f"{source['title']}"
            )

            output_path = process_source(source)
            successes.append(document_id)

            print(
                f"  ✓ Saved to {output_path}"
            )

        except Exception as exc:
            failures.append(
                {
                    "document_id": document_id,
                    "error": str(exc),
                }
            )

            print(
                f"  ✗ Failed: {exc}"
            )

    print("\n--- Parsing summary ---")
    print(f"Successful: {len(successes)}")
    print(f"Failed:     {len(failures)}")

    if failures:
        print("\nFailures:")

        for failure in failures:
            print(
                f"- "
                f"{failure['document_id']}: "
                f"{failure['error']}"
            )


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
        document_ids = STARTER_DOCUMENT_IDS

    process_all_sources(
        document_ids=document_ids,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
```

---

# 6. Purpose of the parser functions

## `load_sources()`

Reads `Sources.json` and validates required metadata.

Required fields include:

```text
document_id
title
year
organization
country
topic
pdf_url
file_path
```

This catches malformed manifest entries before attempting to download or parse documents.

## `clean_text()`

Performs conservative cleanup on extracted text.

It currently:

- converts non-breaking spaces to normal spaces
- removes soft hyphens
- normalizes repeated spaces and tabs
- strips unnecessary whitespace around lines
- collapses excessive blank lines

The cleaning is intentionally conservative because more advanced preprocessing will be handled later.

## `download_pdf()`

Downloads a document from its `pdf_url`.

Before downloading, it checks whether the file already exists locally.

It also performs a basic validation that the HTTP response appears to contain a PDF.

## `parse_pdf()`

Uses PyMuPDF to read every page.

For each page it stores:

```json
{
  "page_number": 1,
  "text": "...",
  "char_count": 1234
}
```

The page number is preserved so later RAG answers can cite the original source page.

## `save_parsed_document()`

Writes structured parsed output to:

```text
data/processed/<document_id>.json
```

Example:

```text
data/processed/doc_002.json
```

## `process_source()`

Handles the complete pipeline for one document:

```text
source metadata
    ↓
download PDF
    ↓
parse PDF
    ↓
save JSON
```

## `process_all_sources()`

Processes multiple selected documents and records successes and failures.

It supports:

- approved-document filtering
- document-ID filtering
- numeric limits
- starter-corpus selection

## `main()`

Defines the command-line interface.

Supported arguments:

```text
--starter-corpus
--limit
--document-ids
```

This allows the corpus to be changed from the terminal without editing the processing logic.

---

# 7. Parsed JSON structure

Each processed document has approximately this structure:

```json
{
  "document_id": "doc_002",

  "metadata": {
    "publication_number": "...",
    "title": "...",
    "year": 2025,
    "organization": "Nordic Council of Ministers",
    "country": "Nordic region",
    "topic": "critical-raw-materials",
    "secondary_topics": "...",
    "source_url": "...",
    "pdf_url": "...",
    "license": "CC BY 4.0",
    "language": "English"
  },

  "parser": {
    "name": "pymupdf",
    "page_count": 100
  },

  "pages": [
    {
      "page_number": 1,
      "text": "...",
      "char_count": 1400
    }
  ]
}
```

The PDF is deliberately **not converted into one giant text string**. Page boundaries are preserved for later citation and retrieval diagnostics.

---

# 8. Test file

File:

```text
tests/test_parsing.py
```

The tests cover four categories:

```text
Manifest validation
       +
Text-cleaning tests
       +
PDF parsing tests
       +
Parsing-quality sanity tests
```

---

# 9. What pytest looks for

## A. `Sources.json` validity

### `test_sources_json_loads`

Checks that:

- `Sources.json` can be read
- its root object is a list
- the list is not empty

### `test_document_ids_are_unique`

Checks that no two documents share the same `document_id`.

Duplicate IDs would cause problems later when storing chunks, embeddings, or metadata.

### `test_required_metadata_present`

Checks that every source contains required metadata such as:

```text
document_id
title
year
organization
country
topic
pdf_url
file_path
```

### `test_pdf_urls_look_valid`

Checks that PDF URLs:

- begin with `http://` or `https://`
- appear to point to a `.pdf`

This is a basic manifest sanity check.

### `test_file_paths_are_unique`

Checks that two documents do not accidentally write to the same local PDF path.

---

# 10. Text-cleaning tests

### `test_clean_text_collapses_spaces`

Input:

```text
This    has       too many spaces.
```

Expected:

```text
This has too many spaces.
```

### `test_clean_text_removes_extra_blank_lines`

Checks that multiple consecutive blank lines are reduced to a normal paragraph break.

### `test_clean_text_removes_non_breaking_space`

Checks that Unicode non-breaking spaces are converted to normal spaces.

---

# 11. PDF tests

The PDF tests work on documents that exist locally in:

```text
data/raw/
```

For the starter RAG, the local-source selection should be restricted to the 9 starter document IDs.

### `test_at_least_one_pdf_available`

Checks that at least one source PDF exists locally.

If no PDFs are present, it fails with:

```text
No PDFs were found in data/raw/.
Run the parser/downloader first.
```

### `test_pdf_can_be_opened`

Uses PyMuPDF to confirm that each PDF:

- can be opened
- has at least one page

This catches corrupted or invalid PDF files.

### `test_pdf_extracts_text`

Combines all extracted page text and checks that a meaningful amount of text was extracted.

Example threshold:

```python
assert len(total_text) > 1000
```

A very low value may indicate:

- a scanned PDF
- extraction failure
- an unusual document structure
- a corrupted file

### `test_page_numbers_are_sequential`

Checks that parsed pages are stored as:

```text
1, 2, 3, 4, 5, ...
```

without gaps or duplicated page numbers.

### `test_parser_preserves_document_id`

Checks that the parsed output retains the correct `document_id` from `Sources.json`.

### `test_parser_preserves_metadata`

Checks that metadata such as:

```text
title
year
topic
```

is copied correctly from the manifest into the parsed JSON.

This is important because this metadata will later accompany chunks in the retrieval system.

---

# 12. Parsing-quality sanity checks

These tests do not prove that extraction is perfect. They detect obvious extraction problems.

### `test_not_too_many_empty_pages`

Counts pages containing almost no extracted text.

The test checks that fewer than 30% of pages are nearly empty.

A high empty-page ratio could indicate:

- image-based/scanned pages
- failed text extraction
- unusual layouts
- PDFs requiring OCR

### `test_average_page_has_reasonable_text`

Calculates the average number of extracted characters per non-empty page.

Example threshold:

```python
average_chars > 200
```

A very small value suggests the parser may be extracting only fragments of the document.

---

# 13. Why automated tests are not enough

A PDF parser can technically pass all automated tests while still producing poor text.

For example, a multi-column page might be extracted in the wrong reading order:

```text
column 2
column 1
figure caption
column 1 continuation
```

Words may also be broken incorrectly:

```text
circular econ-
omy
```

Therefore parsing quality should also be checked manually.

For a few representative reports, compare random parsed pages against the original PDF and inspect:

- reading order
- missing paragraphs
- repeated headers
- repeated footers
- broken words
- tables
- page numbering
- strange symbols
- unexpectedly empty pages

---

# 14. Result of Step 1

The starter corpus was successfully downloaded and parsed:

```text
Selected documents: 9
Successful parsing: 9
Failed parsing:     0
```

The pipeline now produces:

```text
Sources.json
    ↓
9 selected reports
    ↓
data/raw/*.pdf
    ↓
PyMuPDF parser
    ↓
minimal text cleaning
    ↓
page-level structured JSON
    ↓
data/processed/*.json
    ↓
pytest validation
```

---

# 15. What is intentionally NOT done yet

This step does **not** yet perform:

```text
advanced header/footer removal
section detection
table reconstruction
chunking
embeddings
vector database storage
retrieval
reranking
LLM generation
```

Those steps should be added only after the parsed corpus has been validated.

---

# 16. Next step

The next stage is:

```text
Parsed JSON
    ↓
Parsing-quality analysis
    ↓
Preprocessing / cleaning
    ↓
Section detection
    ↓
Chunking
```

The goal of the next stage is to prepare reliable text units for retrieval while retaining enough metadata for traceable RAG citations.
