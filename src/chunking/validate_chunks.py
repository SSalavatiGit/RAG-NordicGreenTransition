from __future__ import annotations

import argparse
import json
import re

from collections import Counter, defaultdict
from pathlib import Path


# Detect page-number-only lines.
_PAGE_NUMBER_LINE = re.compile(
    r"(?m)^\s*\d{1,4}\s*$"
)

# Detect a single newline rather than the intentional
# \n\n boundaries between semantic units.
_SUSPICIOUS_SOFT_WRAP = re.compile(
    r"(?<!\n)\n(?!\n)"
)


def load_jsonl(
    path: Path,
) -> list[dict]:

    rows: list[dict] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_number, line in enumerate(
            f,
            start=1,
        ):

            if not line.strip():
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {exc}"
                ) from exc

    return rows


def validate(
    rows: list[dict],
) -> dict:

    errors: list[str] = []
    warnings: list[str] = []

    ids = [
        row.get("chunk_id")
        for row in rows
    ]

    duplicates = [
        k
        for k, v
        in Counter(ids).items()
        if k is not None
        and v > 1
    ]

    if duplicates:
        errors.append(
            "Duplicate chunk IDs: "
            f"{duplicates[:10]}"
        )

    per_doc_indices: dict[
        str,
        list[int],
    ] = defaultdict(list)

    token_counts: list[int] = []

    required = {
        "chunk_id",
        "document_id",
        "text",
        "page_start",
        "page_end",
        "token_count",
        "chunk_index",
        "chunking",
    }

    for i, row in enumerate(
        rows,
        start=1,
    ):

        missing = (
            required - set(row)
        )

        if missing:

            errors.append(
                f"Row {i} missing fields: "
                f"{sorted(missing)}"
            )

            continue

        chunk_id = row["chunk_id"]

        text = str(
            row["text"]
        )

        # ------------------------------------------
        # Required text
        # ------------------------------------------

        if not text.strip():

            errors.append(
                f"{chunk_id}: empty text"
            )

        # ------------------------------------------
        # Page range
        # ------------------------------------------

        if (
            row["page_start"]
            > row["page_end"]
        ):

            errors.append(
                f"{chunk_id}: "
                "page_start > page_end"
            )

        # ------------------------------------------
        # Token count
        # ------------------------------------------

        if row["token_count"] <= 0:

            errors.append(
                f"{chunk_id}: "
                "non-positive token_count"
            )

        else:

            token_counts.append(
                row["token_count"]
            )

        # ------------------------------------------
        # Chunk size warning
        # ------------------------------------------

        config = (
            row.get("chunking")
            or {}
        )

        target = int(
            config.get(
                "chunk_size",
                500,
            )
        )

        if (
            row["token_count"]
            > int(target * 1.2)
        ):

            warnings.append(
                f"{chunk_id}: "
                f"{row['token_count']} tokens "
                f"exceeds 120% of target "
                f"{target}"
            )

        # ==========================================
        # NEW PDF CLEANING VALIDATION
        # ==========================================

        # Standalone page number such as:
        #
        # \n9\n
        #
        if _PAGE_NUMBER_LINE.search(
            text
        ):

            warnings.append(
                f"{chunk_id}: contains "
                "a standalone page-number line"
            )

        # Normal chunks should now contain:
        #
        # paragraph text
        #
        # heading
        #
        # another paragraph
        #
        # rather than one newline for every PDF
        # visual line.
        if _SUSPICIOUS_SOFT_WRAP.search(
            text
        ):

            warnings.append(
                f"{chunk_id}: contains "
                "a single newline; possible "
                "uncleaned PDF soft line-wrap"
            )

        # Unicode soft hyphen.
        if "\u00ad" in text:

            warnings.append(
                f"{chunk_id}: contains "
                "a soft-hyphen character"
            )

        # Possible extraction hyphenation.
        if re.search(
            r"\b[A-Za-z]{2,}-\s+\n?"
            r"[a-z]{2,}\b",
            text,
        ):

            warnings.append(
                f"{chunk_id}: possible "
                "line-break hyphenation artifact"
            )

        per_doc_indices[
            row["document_id"]
        ].append(
            int(
                row["chunk_index"]
            )
        )

    # ----------------------------------------------
    # Validate contiguous chunk indices
    # ----------------------------------------------

    for doc_id, indices in (
        per_doc_indices.items()
    ):

        expected = list(
            range(
                1,
                len(indices) + 1,
            )
        )

        if sorted(indices) != expected:

            errors.append(
                f"{doc_id}: chunk_index "
                "is not contiguous from 1"
            )

    summary = {

        "chunk_count": len(rows),

        "document_count": len(
            per_doc_indices
        ),

        "min_tokens": (
            min(token_counts)
            if token_counts
            else 0
        ),

        "max_tokens": (
            max(token_counts)
            if token_counts
            else 0
        ),

        "avg_tokens": (
            round(
                sum(token_counts)
                / len(token_counts),
                1,
            )
            if token_counts
            else 0
        ),

        "errors": errors,
        "warnings": warnings,

        "ok": not errors,
    }

    return summary


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Validate generated "
            "chunk JSONL."
        )
    )

    parser.add_argument(
        "path",
        type=Path,
    )

    args = parser.parse_args()

    summary = validate(
        load_jsonl(args.path)
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    raise SystemExit(
        0
        if summary["ok"]
        else 1
    )


if __name__ == "__main__":
    main()