import json
from pathlib import Path

import pymupdf
import pytest

from src.parsing.parse_documents import (
    PROJECT_ROOT,
    clean_text,
    load_sources,
    parse_pdf,
)


# ---------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------


def test_sources_json_loads():
    sources = load_sources()

    assert isinstance(sources, list)
    assert len(sources) > 0


def test_document_ids_are_unique():
    sources = load_sources()

    document_ids = [
        source["document_id"]
        for source in sources
    ]

    assert len(document_ids) == len(set(document_ids))


def test_required_metadata_present():
    sources = load_sources()

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

    for source in sources:
        missing = required_fields - source.keys()

        assert not missing, (
            f"{source.get('document_id')} missing fields: {missing}"
        )


def test_pdf_urls_look_valid():
    sources = load_sources()

    for source in sources:
        assert source["pdf_url"].startswith(
            ("http://", "https://")
        )

        assert ".pdf" in source["pdf_url"].lower()


def test_file_paths_are_unique():
    sources = load_sources()

    paths = [
        source["file_path"]
        for source in sources
    ]

    assert len(paths) == len(set(paths))


# ---------------------------------------------------------
# Cleaning tests
# ---------------------------------------------------------


def test_clean_text_collapses_spaces():
    text = "This    has       too many spaces."

    cleaned = clean_text(text)

    assert cleaned == "This has too many spaces."


def test_clean_text_removes_extra_blank_lines():
    text = "Paragraph one.\n\n\n\nParagraph two."

    cleaned = clean_text(text)

    assert cleaned == "Paragraph one.\n\nParagraph two."


def test_clean_text_removes_non_breaking_space():
    text = "Nordic\u00a0countries"

    cleaned = clean_text(text)

    assert cleaned == "Nordic countries"


# ---------------------------------------------------------
# Local PDF tests
# ---------------------------------------------------------


def local_sources():
    """
    Return only sources whose PDFs have already been downloaded.

    This allows pytest to work without depending on the internet.
    """

    sources = load_sources()

    return [
        source
        for source in sources
        if (PROJECT_ROOT / source["file_path"]).exists()
    ]


def test_at_least_one_pdf_available():
    """
    Run this after downloading the corpus.
    """

    sources = local_sources()

    assert len(sources) > 0, (
        "No PDFs were found in data/raw/. "
        "Run the parser/downloader first."
    )


@pytest.mark.parametrize(
    "source",
    local_sources(),
    ids=lambda x: x["document_id"],
)
def test_pdf_can_be_opened(source):
    pdf_path = PROJECT_ROOT / source["file_path"]

    document = pymupdf.open(pdf_path)

    assert document.page_count > 0

    document.close()


@pytest.mark.parametrize(
    "source",
    local_sources(),
    ids=lambda x: x["document_id"],
)
def test_pdf_extracts_text(source):
    pdf_path = PROJECT_ROOT / source["file_path"]

    parsed = parse_pdf(
        pdf_path,
        source,
    )

    total_text = "".join(
        page["text"]
        for page in parsed["pages"]
    )

    assert len(total_text) > 1000, (
        f"{source['document_id']} produced suspiciously little text."
    )


@pytest.mark.parametrize(
    "source",
    local_sources(),
    ids=lambda x: x["document_id"],
)
def test_page_numbers_are_sequential(source):
    pdf_path = PROJECT_ROOT / source["file_path"]

    parsed = parse_pdf(
        pdf_path,
        source,
    )

    page_numbers = [
        page["page_number"]
        for page in parsed["pages"]
    ]

    expected = list(
        range(1, len(parsed["pages"]) + 1)
    )

    assert page_numbers == expected


@pytest.mark.parametrize(
    "source",
    local_sources(),
    ids=lambda x: x["document_id"],
)
def test_parser_preserves_document_id(source):
    pdf_path = PROJECT_ROOT / source["file_path"]

    parsed = parse_pdf(
        pdf_path,
        source,
    )

    assert parsed["document_id"] == source["document_id"]


@pytest.mark.parametrize(
    "source",
    local_sources(),
    ids=lambda x: x["document_id"],
)
def test_parser_preserves_metadata(source):
    pdf_path = PROJECT_ROOT / source["file_path"]

    parsed = parse_pdf(
        pdf_path,
        source,
    )

    assert parsed["metadata"]["title"] == source["title"]
    assert parsed["metadata"]["year"] == source["year"]
    assert parsed["metadata"]["topic"] == source["topic"]


# ---------------------------------------------------------
# Extraction quality sanity tests
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    local_sources(),
    ids=lambda x: x["document_id"],
)
def test_not_too_many_empty_pages(source):
    pdf_path = PROJECT_ROOT / source["file_path"]

    parsed = parse_pdf(
        pdf_path,
        source,
    )

    page_count = len(parsed["pages"])

    empty_pages = sum(
        1
        for page in parsed["pages"]
        if len(page["text"].strip()) < 20
    )

    empty_ratio = empty_pages / page_count

    assert empty_ratio < 0.30, (
        f"{source['document_id']} has "
        f"{empty_ratio:.1%} nearly-empty pages."
    )


@pytest.mark.parametrize(
    "source",
    local_sources(),
    ids=lambda x: x["document_id"],
)
def test_average_page_has_reasonable_text(source):
    pdf_path = PROJECT_ROOT / source["file_path"]

    parsed = parse_pdf(
        pdf_path,
        source,
    )

    non_empty_pages = [
        page
        for page in parsed["pages"]
        if page["text"].strip()
    ]

    average_chars = sum(
        page["char_count"]
        for page in non_empty_pages
    ) / max(len(non_empty_pages), 1)

    assert average_chars > 200, (
        f"{source['document_id']} average extracted "
        f"text is only {average_chars:.0f} characters/page."
    )