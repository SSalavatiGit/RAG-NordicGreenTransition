from __future__ import annotations

import re

NOISE_LABELS = ("toc", "imprint", "references", "blank")

_BOILERPLATE = re.compile(
    r"isbn|disclaimer|creative commons|about this publication|"
    r"nordic council of ministers|cover photo|cover illustration|"
    r"rights and permissions|this translation was not produced|"
    r"this publication is also available|this publication was funded|"
    r"content does not necessarily reflect|all rights reserved",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_NUMBERED_LINE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")
_HEADING = re.compile(r"\b\d+(?:\.\d+)*\.?\s+[A-Za-z]")
_YEAR = re.compile(r"\(\d{4}[a-z]?\)")
_URL = re.compile(r"https?://")


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _long_sentences(text: str, min_words: int) -> list[str]:
    flat = " ".join(text.split())
    sentences = []
    for part in _SENTENCE_SPLIT.split(flat):
        words = part.split()
        if len(words) >= min_words:
            sentences.append(part.strip())
    return sentences


def _substantive_sentences(text: str) -> list[str]:
    """Prose that is not publication boilerplate."""
    return [
        sentence
        for sentence in _long_sentences(text, min_words=15)
        if not _BOILERPLATE.search(sentence)
    ]


def classify_noise(text: str) -> str | None:
    """
    Label a chunk that should not compete in retrieval.

    Mixed chunks that still contain topical prose are kept.
    """
    flat = " ".join(text.split())
    if len(flat) < 80:
        return "blank"

    lines = _lines(text)
    if not lines:
        return "blank"

    substantive = _substantive_sentences(text)
    numbered = sum(bool(_NUMBERED_LINE.match(line)) for line in lines)
    head = flat[:100].lower()
    # Numbered section titles can be long. Prose is an unnumbered line.
    prose_lines = [
        line
        for line in lines
        if len(line.split()) >= 15 and not _NUMBERED_LINE.match(line)
    ]

    starts_as_contents = head.startswith("contents") or head.startswith(
        "table of contents"
    )
    # Extracted contents pages often join several headings onto one line.
    # A real summary after the contents list has many more words per heading.
    heading_count = len(_HEADING.findall(flat))
    word_count = len(flat.split())
    if (
        starts_as_contents
        and heading_count >= 6
        and word_count / heading_count <= 18
    ):
        return "toc"

    if numbered >= 8 and not prose_lines and not substantive:
        return "toc"

    has_imprint_heading = bool(
        re.search(r"about this publication", flat, re.IGNORECASE)
    )
    isbn_count = len(re.findall(r"\bISBN\b", flat))
    if (
        not substantive
        and (isbn_count >= 1 or has_imprint_heading)
        and (
            has_imprint_heading
            or "disclaimer" in flat.lower()
            or "creative commons" in flat.lower()
        )
    ):
        return "imprint"

    citation_lines = sum(
        1
        for line in lines
        if _URL.search(line) or _YEAR.search(line) or "doi.org" in line.lower()
    )
    if (
        not substantive
        and len(lines) >= 4
        and citation_lines >= 5
        and citation_lines / len(lines) >= 0.6
    ):
        return "references"

    return None
