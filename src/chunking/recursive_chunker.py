from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .models import Chunk, ChunkingConfig

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# A line containing only a page number.
_PAGE_NUMBER_ONLY = re.compile(r"^\s*\d{1,4}\s*$")

# Common bullet/list formats.
_BULLET_START = re.compile(
    r"^\s*(?:[-•▪◦–—]|\d+[.)]|[A-Za-z][.)])\s+"
)


class TokenCounter:
    """Token counting with tiktoken when available."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self._encoding = (
            tiktoken.get_encoding(encoding_name)
            if tiktoken
            else None
        )

    def count(self, text: str) -> int:
        if self._encoding:
            return len(self._encoding.encode(text))

        return len(text.split())

    def tail(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""

        if self._encoding:
            tokens = self._encoding.encode(text)
            return self._encoding.decode(tokens[-max_tokens:])

        words = text.split()
        return " ".join(words[-max_tokens:])


class RecursivePageAwareChunker:
    """
    Create traceable chunks from parsed page-level documents.

    Expected input:

    {
        "document_id": "doc_001",
        "metadata": {...},
        "pages": [
            {
                "page_number": 1,
                "text": "...",
                "char_count": 1234
            }
        ]
    }

    Cleaning performed before chunking:

    1. Normalise whitespace.
    2. Remove standalone page numbers.
    3. Preserve true paragraph breaks.
    4. Join PDF visual line-wraps.
    5. Preserve likely headings.
    6. Preserve list items.
    7. Split oversized blocks by sentence, then words.

    Note:
    Section metadata is still left empty because the parser itself
    does not currently provide trustworthy document structure.
    """

    def __init__(
        self,
        config: ChunkingConfig | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.tokens = TokenCounter(
            self.config.tokenizer
        )

    def chunk_document(
        self,
        document: dict,
    ) -> list[Chunk]:

        document_id = document["document_id"]
        metadata = document.get("metadata", {})
        pages = document.get("pages", [])

        units: list[dict] = []

        for page in pages:
            page_number = int(page["page_number"])

            raw_text = page.get("text") or ""

            if not raw_text.strip():
                continue

            # --------------------------------------------------
            # NEW:
            # clean PDF text before chunking
            # --------------------------------------------------
            blocks = self._clean_page_to_blocks(
                raw_text
            )

            for block in blocks:

                # If a block itself is > chunk_size,
                # recursively split it.
                parts = self._split_block(block)

                for part in parts:
                    clean = part.strip()

                    if clean:
                        units.append(
                            {
                                "text": clean,
                                "page": page_number,
                            }
                        )

        raw_chunks = self._pack_units(units)

        raw_chunks = (
            self._merge_tiny_final_chunk(
                raw_chunks
            )
        )

        chunks: list[Chunk] = []

        for index, item in enumerate(
            raw_chunks,
            start=1,
        ):
            text = item["text"].strip()

            chunk_id = (
                f"{document_id}_chunk_"
                f"{index:04d}"
            )

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    title=metadata.get("title"),
                    year=metadata.get("year"),
                    organization=metadata.get(
                        "organization"
                    ),
                    country=metadata.get(
                        "country"
                    ),
                    topic=metadata.get(
                        "topic"
                    ),
                    secondary_topics=(
                        _normalise_topics(
                            metadata.get(
                                "secondary_topics"
                            )
                        )
                    ),

                    # Parser does not yet provide
                    # reliable section structure.
                    section=None,
                    section_path=[],

                    page_start=item[
                        "page_start"
                    ],
                    page_end=item[
                        "page_end"
                    ],

                    content_type="paragraph",

                    text=text,

                    token_count=(
                        self.tokens.count(text)
                    ),

                    chunk_index=index,

                    source_url=metadata.get(
                        "source_url"
                    ),

                    chunking=asdict(
                        self.config
                    ),
                )
            )

        return chunks

    # ==========================================================
    # PDF CLEANING
    # ==========================================================

    def _clean_page_to_blocks(
        self,
        text: str,
    ) -> list[str]:
        """
        Convert noisy PDF page text into cleaner
        semantic blocks.

        Example input:

        Enhanced collaboration between PSS providers – whether
        within the same product category or across different
        sectors facing similar challenges – can unlock significant
        synergies.

        becomes:

        Enhanced collaboration between PSS providers – whether
        within the same product category or across different
        sectors facing similar challenges – can unlock significant
        synergies.

        but internally it is stored as one continuous paragraph.
        """

        # ------------------------------------------
        # Normalise newline formats
        # ------------------------------------------

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        # Remove Unicode soft hyphen.
        text = text.replace(
            "\u00ad",
            "",
        )

        # Replace non-breaking spaces.
        text = text.replace(
            "\xa0",
            " ",
        )

        raw_lines = text.split("\n")

        lines: list[str] = []

        # ------------------------------------------
        # First pass:
        # clean each extracted PDF line
        # ------------------------------------------

        for raw_line in raw_lines:

            line = re.sub(
                r"[ \t]+",
                " ",
                raw_line,
            ).strip()

            # Preserve blank line as paragraph boundary.
            if not line:

                if (
                    lines
                    and lines[-1] != ""
                ):
                    lines.append("")

                continue

            # Remove page numbers such as:
            #
            # 9
            #
            if _PAGE_NUMBER_ONLY.fullmatch(
                line
            ):
                continue

            lines.append(line)

        # Remove trailing empty boundaries.
        while (
            lines
            and lines[-1] == ""
        ):
            lines.pop()

        blocks: list[str] = []

        paragraph_lines: list[str] = []

        def flush_paragraph() -> None:
            if not paragraph_lines:
                return

            joined = (
                self._join_wrapped_lines(
                    paragraph_lines
                )
            )

            if joined:
                blocks.append(joined)

            paragraph_lines.clear()

        # ------------------------------------------
        # Second pass:
        # construct paragraphs/headings/lists
        # ------------------------------------------

        for i, line in enumerate(lines):

            if line == "":
                flush_paragraph()
                continue

            prev_line = (
                lines[i - 1]
                if i > 0
                else ""
            )

            next_line = (
                lines[i + 1]
                if i + 1 < len(lines)
                else ""
            )

            # --------------------------------------
            # List items
            # --------------------------------------

            if _BULLET_START.match(line):

                flush_paragraph()

                blocks.append(line)

                continue

            # --------------------------------------
            # Likely heading
            # --------------------------------------

            if self._looks_like_heading(
                line,
                prev_line,
                next_line,
            ):
                flush_paragraph()

                blocks.append(line)

                continue

            # --------------------------------------
            # Ordinary paragraph line
            # --------------------------------------

            paragraph_lines.append(line)

        flush_paragraph()

        return [
            block
            for block in blocks
            if block.strip()
        ]

    def _looks_like_heading(
        self,
        line: str,
        prev_line: str,
        next_line: str,
    ) -> bool:
        """
        Conservative heuristic for headings.

        Intended to identify things such as:

        Collaboration and knowledge-sharing

        Environmental potential of PSS

        without marking every short PDF line
        as a heading.
        """

        if not line:
            return False

        # Headings are usually reasonably short.
        if len(line) > 90:
            return False

        # Too many words = probably body text.
        if len(line.split()) > 12:
            return False

        # Headings usually do not end like sentences.
        if line.endswith(
            (
                ".",
                ",",
                ";",
                ":",
                "?",
                "!",
            )
        ):
            return False

        # Usually begins with uppercase text.
        starts_plausibly = (
            line[0].isupper()
            or line[0].isdigit()
        )

        if not starts_plausibly:
            return False

        # A heading often starts a new block.
        boundary_before = (
            prev_line == ""
            or len(prev_line) <= 45
        )

        # Body line after heading is usually
        # substantially longer.
        next_is_body_like = (
            bool(next_line)
            and len(next_line) >= 40
        )

        return (
            boundary_before
            and next_is_body_like
        )

    def _join_wrapped_lines(
        self,
        lines: list[str],
    ) -> str:
        """
        Join line wraps created only by PDF layout.

        Example:

        "Enhanced collaboration between PSS providers – whether"
        "within the same product category"

        becomes:

        "Enhanced collaboration between PSS providers – whether
        within the same product category"
        """

        if not lines:
            return ""

        result = lines[0]

        for line in lines[1:]:

            if not line:
                continue

            # --------------------------------------
            # Handle words broken across lines
            # --------------------------------------
            if self._is_split_word_hyphen(
                result,
                line,
            ):
                result = (
                    result[:-1]
                    + line
                )

            else:
                # Ordinary visual PDF line-wrap.
                result += " " + line

        # Collapse accidental repeated whitespace.
        result = re.sub(
            r"\s+",
            " ",
            result,
        )

        return result.strip()

    @staticmethod
    def _is_split_word_hyphen(
        left_text: str,
        right_line: str,
    ) -> bool:
        """
        Detect obvious words broken by PDF line-wrap.

        Conservative because some words really are
        hyphenated, e.g.:

            product-oriented
            use-oriented
            result-oriented

        We should NOT turn those into:
            productoriented
        """

        if not left_text.endswith("-"):
            return False

        left_match = re.search(
            r"([A-Za-z]{2,})-$",
            left_text,
        )

        right_match = re.match(
            r"([a-z]{2,})\b",
            right_line,
        )

        if (
            not left_match
            or not right_match
        ):
            return False

        right_word = (
            right_match
            .group(1)
            .lower()
        )

        # Frequent legitimate compound endings.
        # Extend if needed after inspecting corpus.
        compound_components = {
            "based",
            "oriented",
            "related",
            "specific",
            "term",
            "level",
            "scale",
            "wide",
            "driven",
            "efficient",
            "intensive",
        }

        return (
            right_word
            not in compound_components
        )

    # ==========================================================
    # RECURSIVE SPLITTING
    # ==========================================================

    def _split_block(
        self,
        text: str,
    ) -> list[str]:
        """
        Keep the whole paragraph/block when possible.

        If too large:
            block
              ↓
            sentences
              ↓
            words
        """

        text = text.strip()

        if not text:
            return []

        # Whole paragraph fits.
        if (
            self.tokens.count(text)
            <= self.config.chunk_size
        ):
            return [text]

        sentences = [
            s.strip()
            for s in _SENTENCE_SPLIT.split(
                text
            )
            if s.strip()
        ]

        # No useful sentence boundaries.
        if len(sentences) == 1:
            return self._split_by_words(
                text
            )

        parts: list[str] = []
        buffer: list[str] = []

        for sentence in sentences:

            candidate = " ".join(
                buffer + [sentence]
            ).strip()

            if (
                buffer
                and self.tokens.count(
                    candidate
                )
                > self.config.chunk_size
            ):

                parts.append(
                    " ".join(buffer)
                )

                buffer = [sentence]

            else:
                buffer.append(sentence)

        if buffer:

            tail = " ".join(buffer)

            if (
                self.tokens.count(tail)
                > self.config.chunk_size
            ):

                parts.extend(
                    self._split_by_words(
                        tail
                    )
                )

            else:
                parts.append(tail)

        return parts

    def _split_by_words(
        self,
        text: str,
    ) -> list[str]:

        words = text.split()

        if not words:
            return []

        parts: list[str] = []
        buf: list[str] = []

        for word in words:

            candidate = " ".join(
                buf + [word]
            )

            if (
                buf
                and self.tokens.count(
                    candidate
                )
                > self.config.chunk_size
            ):

                parts.append(
                    " ".join(buf)
                )

                buf = [word]

            else:
                buf.append(word)

        if buf:
            parts.append(
                " ".join(buf)
            )

        return parts

    # ==========================================================
    # CHUNK PACKING
    # ==========================================================

    def _pack_units(
        self,
        units: list[dict],
    ) -> list[dict]:

        chunks: list[dict] = []
        buffer: list[dict] = []

        def render(
            items: list[dict],
        ) -> str:

            return "\n\n".join(
                x["text"]
                for x in items
            ).strip()

        for unit in units:

            candidate = (
                buffer + [unit]
            )

            if (
                buffer
                and self.tokens.count(
                    render(candidate)
                )
                > self.config.chunk_size
            ):

                chunks.append(
                    self._make_raw_chunk(
                        buffer
                    )
                )

                buffer = (
                    self._overlap_units(
                        buffer
                    )
                )

                # Avoid pathological case where
                # overlap itself leaves no room.
                while (
                    buffer
                    and self.tokens.count(
                        render(
                            buffer + [unit]
                        )
                    )
                    > self.config.chunk_size
                ):
                    buffer.pop(0)

            buffer.append(unit)

        if buffer:
            chunks.append(
                self._make_raw_chunk(
                    buffer
                )
            )

        return chunks

    # ==========================================================
    # OVERLAP
    # ==========================================================

    def _overlap_units(
        self,
        previous: list[dict],
    ) -> list[dict]:

        if (
            self.config.chunk_overlap
            <= 0
        ):
            return []

        kept: list[dict] = []
        used = 0

        for unit in reversed(previous):

            n = self.tokens.count(
                unit["text"]
            )

            if (
                kept
                and used + n
                > self.config.chunk_overlap
            ):
                break

            # If a single unit is already larger
            # than overlap, take only its tail.
            if (
                not kept
                and n
                > self.config.chunk_overlap
            ):

                tail = self.tokens.tail(
                    unit["text"],
                    self.config.chunk_overlap,
                )

                if not tail:
                    return []

                return [
                    {
                        "text": tail,
                        "page": unit["page"],
                    }
                ]

            kept.append(unit)

            used += n

            if (
                used
                >= self.config.chunk_overlap
            ):
                break

        return list(
            reversed(kept)
        )

    # ==========================================================
    # RAW CHUNK CREATION
    # ==========================================================

    def _make_raw_chunk(
        self,
        items: list[dict],
    ) -> dict:

        return {
            "text": "\n\n".join(
                x["text"]
                for x in items
            ).strip(),

            "page_start": min(
                x["page"]
                for x in items
            ),

            "page_end": max(
                x["page"]
                for x in items
            ),
        }

    # ==========================================================
    # SMALL FINAL CHUNK
    # ==========================================================

    def _merge_tiny_final_chunk(
        self,
        chunks: list[dict],
    ) -> list[dict]:

        if len(chunks) < 2:
            return chunks

        last = chunks[-1]

        if (
            self.tokens.count(
                last["text"]
            )
            >= self.config.minimum_chunk_size
        ):
            return chunks

        prev = chunks[-2]

        combined = (
            prev["text"]
            + "\n\n"
            + last["text"]
        ).strip()

        # Permit a small excess only when this
        # prevents an unusably tiny final chunk.
        if (
            self.tokens.count(combined)
            <= int(
                self.config.chunk_size
                * 1.2
            )
        ):

            chunks[-2] = {
                "text": combined,

                "page_start": min(
                    prev["page_start"],
                    last["page_start"],
                ),

                "page_end": max(
                    prev["page_end"],
                    last["page_end"],
                ),
            }

            chunks.pop()

        return chunks


def _normalise_topics(
    value,
) -> list[str]:

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    if isinstance(value, str):
        return [
            x.strip()
            for x in value.split(";")
            if x.strip()
        ]

    return [str(value)]


def load_document(
    path: Path,
) -> dict:

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_jsonl(
    chunks: Iterable[Chunk],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for chunk in chunks:

            f.write(
                json.dumps(
                    chunk.to_dict(),
                    ensure_ascii=False,
                )
                + "\n"
            )


def chunk_directory(
    input_dir: Path,
    output_file: Path,
    config: ChunkingConfig,
    pattern: str = "doc_*.json",
) -> list[Chunk]:

    chunker = (
        RecursivePageAwareChunker(
            config
        )
    )

    all_chunks: list[Chunk] = []

    for path in sorted(
        input_dir.glob(pattern)
    ):

        document = load_document(
            path
        )

        chunks = (
            chunker.chunk_document(
                document
            )
        )

        all_chunks.extend(chunks)

    write_jsonl(
        all_chunks,
        output_file,
    )

    return all_chunks


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Chunk parsed Nordic Green "
            "Transition RAG documents."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "data/processed"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/chunks/"
            "baseline_recursive_500_75.jsonl"
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=75,
    )

    parser.add_argument(
        "--minimum-chunk-size",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    config = ChunkingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        minimum_chunk_size=(
            args.minimum_chunk_size
        ),
    )

    chunks = chunk_directory(
        args.input_dir,
        args.output,
        config,
    )

    print(
        f"Wrote {len(chunks)} "
        f"chunks to {args.output}"
    )


if __name__ == "__main__":
    main()