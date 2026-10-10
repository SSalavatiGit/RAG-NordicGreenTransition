# Chunking Pipeline — Baseline v1

This document describes the first chunking stage of the **Nordic Green Transition RAG** project.

The chunker is designed to consume the already parsed and manually checked report files and convert them into smaller, metadata-rich text units that can later be embedded, indexed in a vector database, retrieved, evaluated, and cited.

---

## 1. Goal

The goal of this stage is to transform parsed reports into retrieval-ready chunks while preserving enough metadata to trace every chunk back to its source document and page range.

The baseline pipeline is intentionally simple and reproducible:

```text
Parsed JSON documents
        ↓
Page-aware recursive chunking
        ↓
~500-token chunks
        ↓
75-token overlap where applicable
        ↓
Attach document + page metadata
        ↓
Write JSONL
        ↓
Validate generated chunks
```

The baseline is used as the reference configuration for later experiments with different chunk sizes, overlaps, section-aware chunking, semantic chunking, and other retrieval strategies.

### Baseline configuration

```text
Strategy:           recursive_page_aware
Target chunk size:  500 tokens
Chunk overlap:      75 tokens
Minimum chunk size: 100 tokens
Tokenizer:          cl100k_base
Split preference:   paragraph → sentence → word
Output format:      JSONL
```

### Why the baseline is page-aware

The current parser preserves page structure but does not yet extract section headings as structured metadata.

Therefore, the chunker does **not** invent section information. In the baseline output:

```json
"section": null,
"section_path": []
```

Page information is preserved through:

```json
"page_start": 5,
"page_end": 6
```

A later experiment can add true section-aware chunking and compare it objectively against this baseline.

---

## 2. Input Data

The chunker expects parsed report files in:

```text
data/processed/
```

For example:

```text
data/processed/doc_001.json
data/processed/doc_002.json
data/processed/doc_003.json
...
```

The expected parsed-document structure is approximately:

```json
{
  "document_id": "doc_002",
  "metadata": {
    "title": "...",
    "year": 2025,
    "organization": "Nordic Council of Ministers",
    "country": "Nordic region",
    "topic": "critical-raw-materials"
  },
  "parser": {
    "name": "pymupdf",
    "page_count": 42
  },
  "pages": [
    {
      "page_number": 1,
      "text": "...",
      "char_count": 1234
    }
  ]
}
```

The first nine parsed documents were manually inspected before chunking and used to test the pipeline.

---

## 3. Generated Code

The chunking implementation is located under:

```text
src/chunking/
```

### `src/chunking/models.py`

Defines the data structures used by the chunking pipeline.

Its purpose is to keep the chunk representation consistent across chunk generation, validation, retrieval, and later evaluation steps.

Important chunk fields include:

```text
chunk_id
document_id
title
year
organization
country
topic
secondary_topics
section
section_path
page_start
page_end
content_type
text
token_count
chunk_index
source_url
chunking
```

---

### `src/chunking/recursive_chunker.py`

Implements the baseline chunking algorithm.

Main responsibilities:

- load parsed JSON documents;
- iterate through page text;
- split text recursively using paragraph, sentence, and word boundaries;
- target approximately 500 tokens per chunk;
- preserve overlap where appropriate;
- keep page-range information;
- propagate document metadata to each chunk;
- generate deterministic chunk IDs;
- write all chunks to a JSONL output file.

Chunk IDs follow this format:

```text
doc_001_chunk_0001
doc_001_chunk_0002
doc_002_chunk_0001
...
```

This makes chunks easy to inspect and debug later.

---

### `src/chunking/validate_chunks.py`

Validates the generated JSONL file before embeddings are created.

The validator checks for problems such as:

- duplicate chunk IDs;
- empty chunk text;
- invalid page ranges;
- inconsistent chunk indexes;
- invalid token counts;
- unexpectedly oversized chunks;
- malformed records.

The purpose of this step is to detect chunking problems **before** they propagate into embeddings and retrieval evaluation.

---

### `src/chunking/__init__.py`

Marks the directory as a Python package and exposes the chunking components for module-based execution/imports.

---

### `tests/test_chunking.py`

Contains automated tests for the baseline chunker and validator.

The initial implementation passed the included synthetic tests before being run on the real parsed reports.

---

### `requirements-chunking.txt`

Contains the additional dependencies required by this stage, including the tokenizer and test framework.

The baseline uses:

```text
tiktoken
pytest
```

---

## 4. Terminal Commands

Run the following commands from the **project root**.

### Install dependencies

```bash
pip install tiktoken pytest
```

Alternatively, if the chunking requirements file is used:

```bash
pip install -r requirements-chunking.txt
```

---

### Generate baseline chunks

```bash
python -m src.chunking.recursive_chunker \
  --input-dir data/processed \
  --output data/processed/chunks/baseline_recursive_500_75.jsonl
```

This command reads the parsed JSON files and generates the baseline chunk dataset.

---

### Validate the generated chunks

```bash
python -m src.chunking.validate_chunks \
  data/processed/chunks/baseline_recursive_500_75.jsonl
```

The validation step should be run before moving to embeddings.

---

### Run automated tests

```bash
pytest -q
```

---

## 5. Resulting Files

The main generated file is:

```text
data/processed/chunks/baseline_recursive_500_75.jsonl
```

The relevant project structure becomes:

```text
data/
├── processed/
│   ├── doc_001.json
│   ├── doc_002.json
│   ├── doc_003.json
│   ├── ...
│   └── chunks/
│       └── baseline_recursive_500_75.jsonl
│
└── raw/
    └── ...

src/
└── chunking/
    ├── __init__.py
    ├── models.py
    ├── recursive_chunker.py
    └── validate_chunks.py

tests/
└── test_chunking.py
```

### Why JSONL?

JSONL stores one JSON object per line:

```json
{"chunk_id":"doc_001_chunk_0001","text":"..."}
{"chunk_id":"doc_001_chunk_0002","text":"..."}
{"chunk_id":"doc_002_chunk_0001","text":"..."}
```

This format is convenient for:

- streaming large collections;
- loading chunks into Python;
- importing records into a vector database;
- comparing experiment outputs;
- inspecting individual chunks;
- processing one record at a time.

---

## 6. Example of a Resulting Chunk

One real record from the generated baseline file is:

```json
{
  "chunk_id": "doc_002_chunk_0006",
  "document_id": "doc_002",
  "title": "Reducing and re-using critical raw materials in the Nordics: Evaluation of potential measures for increased resource efficiency and waste prevention",
  "year": 2025,
  "organization": "Nordic Council of Ministers",
  "country": "Nordic region",
  "topic": "critical-raw-materials",
  "secondary_topics": [
    "circular-economy",
    "resource-efficiency",
    "waste-prevention"
  ],
  "section": null,
  "section_path": [],
  "page_start": 5,
  "page_end": 6,
  "content_type": "paragraph",
  "text": "functionality it must be replaced by an alternative substance that provides similar properties while being less hazardous or scarce...",
  "token_count": 284,
  "chunk_index": 6,
  "source_url": "https://pub.norden.org/temanord2025-534/about-this-publication.html",
  "chunking": {
    "strategy": "recursive_page_aware",
    "chunk_size": 500,
    "chunk_overlap": 75,
    "minimum_chunk_size": 100,
    "tokenizer": "cl100k_base"
  }
}
```

### Explanation of the fields

#### `chunk_id`

```json
"chunk_id": "doc_002_chunk_0006"
```

A deterministic unique identifier.

It means:

```text
document 002
+
chunk 0006
```

This makes retrieval failures easy to trace and debug.

---

#### `document_id`

```json
"document_id": "doc_002"
```

Links the chunk back to its original report.

---

#### Document metadata

```json
"title": "...",
"year": 2025,
"organization": "Nordic Council of Ministers",
"country": "Nordic region",
"topic": "critical-raw-materials"
```

These fields are copied from the document metadata.

They will later support operations such as:

```text
retrieve relevant chunks
AND year >= 2024
AND topic = critical-raw-materials
```

They are also useful for citations and debugging.

---

#### `secondary_topics`

```json
"secondary_topics": [
  "circular-economy",
  "resource-efficiency",
  "waste-prevention"
]
```

Stores additional document-level topic labels that can later be used for filtering or analysis.

---

#### `section` and `section_path`

```json
"section": null,
"section_path": []
```

These are intentionally empty in the baseline.

The current parser preserves page structure but does not yet identify headings as structured sections. The chunker therefore does not fabricate section metadata.

These fields are kept in the schema so that a later **section-aware chunking experiment** can populate them without redesigning the rest of the RAG pipeline.

---

#### `page_start` and `page_end`

```json
"page_start": 5,
"page_end": 6
```

This chunk contains text originating from pages 5 and 6 of the parsed document.

This information is important for:

- source traceability;
- citations;
- retrieval diagnostics;
- inspecting the original report when a chunk looks incorrect.

---

#### `content_type`

```json
"content_type": "paragraph"
```

The baseline currently treats ordinary textual content broadly as paragraph content.

Later this field may be expanded to distinguish content such as:

```text
paragraph
table
list
executive_summary
appendix
```

For example, this particular chunk contains a list of measures, so a more advanced parser could eventually classify it as `list` rather than `paragraph`.

---

#### `text`

```json
"text": "functionality it must be replaced by an alternative substance..."
```

This is the actual retrieved content that will later be converted into an embedding.

The metadata is stored separately so the system can preserve traceability without unnecessarily mixing every metadata field into the semantic text.

---

#### `token_count`

```json
"token_count": 284
```

The baseline targets approximately 500 tokens, but a chunk does **not** have to contain exactly 500 tokens.

The chunker prefers meaningful structural boundaries instead of forcing every chunk to exactly the same size.

Other real chunks in the same document were close to the target, for example:

```text
chunk 0007 → 489 tokens
chunk 0008 → 488 tokens
```

---

#### `chunk_index`

```json
"chunk_index": 6
```

Stores the chunk's order inside the document.

This can later support neighbor retrieval such as:

```text
retrieved chunk 27
→ optionally inspect chunk 26
→ optionally inspect chunk 28
```

---

#### `source_url`

```json
"source_url": "https://pub.norden.org/temanord2025-534/about-this-publication.html"
```

Links the chunk back to the public source of the report and can later be surfaced in citations or the user interface.

---

#### `chunking`

```json
"chunking": {
  "strategy": "recursive_page_aware",
  "chunk_size": 500,
  "chunk_overlap": 75,
  "minimum_chunk_size": 100,
  "tokenizer": "cl100k_base"
}
```

Records exactly how the chunk was generated.

This is important for reproducible experiments. Later, several outputs can be compared, for example:

```text
recursive_300_50
recursive_500_75
recursive_800_100
section_aware
```

and retrieval metrics can determine which configuration works best.

---

## 7. Observations from the First Real Output

Manual inspection of the generated chunks revealed several useful points.

### 7.1 Page numbers remain in some extracted text

For example, one chunk contains text similar to:

```text
... increased health hazards. 5

6
The following measures are discussed in this report:
```

The standalone `5` and `6` are page-number artifacts that survived preprocessing.

This does not mean the chunker is failing, but it indicates that the preprocessing stage can later be improved to remove repeated or isolated page numbers from the text.

---

### 7.2 Some reports contain more than one language

For example, subsequent chunks from `doc_002` contain the Norwegian heading:

```text
Sammendrag
```

and Norwegian summary text even though the report metadata identifies the primary language as English.

This should be handled as a deliberate corpus decision before final retrieval experiments. Possible strategies include:

```text
keep multilingual content
```

or

```text
retain only the English report sections
```

The baseline keeps the parsed content unchanged so that this decision can be evaluated explicitly rather than silently modifying the source.

---

### 7.3 Chunk boundaries should be manually inspected

One chunk begins with:

```text
functionality it must be replaced by an alternative substance...
```

This begins in the middle of a thought from the preceding content.

This is exactly why chunk overlap and later chunking experiments matter. Before embeddings are finalized, a sample of generated chunks should be manually inspected for coherence and boundary quality.

---

## 8. Validation Before Embeddings

Before moving to the embedding stage, inspect approximately 30–50 chunks across the nine documents.

Check that:

1. chunk text is understandable in isolation;
2. no chunks are empty;
3. chunk IDs are unique;
4. page ranges are correct;
5. chunks are not dominated by headers, footers, or page numbers;
6. important paragraphs are not badly fragmented;
7. lists and tables remain usable;
8. overlap provides useful context without excessive duplication;
9. token-size distribution is reasonable;
10. multilingual or duplicated sections are understood and handled deliberately.

Only after this baseline passes validation should the project proceed to:

```text
Chunks
  ↓
Embeddings
  ↓
Vector database
  ↓
Dense retrieval baseline
  ↓
Retrieval evaluation
```

---

## 9. Role of This Baseline in Later Experiments

The purpose of `baseline_recursive_500_75.jsonl` is not to claim that 500-token recursive chunks are optimal.

It provides a controlled starting point.

Later experiments can compare configurations such as:

| Experiment | Strategy | Chunk Size | Overlap |
|---|---|---:|---:|
| A | Recursive | 300 | 50 |
| B | Recursive | 500 | 75 |
| C | Recursive | 800 | 100 |
| D | Section-aware | Variable | — |

The same evaluation queries and retrieval metrics can then be used to determine whether changing the chunking strategy actually improves retrieval.

The final choice should be based on measured RAG performance rather than an arbitrary chunk-size preference.
