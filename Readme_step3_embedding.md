# Step 3 — Embeddings and Qdrant Indexing

This step starts **after document parsing and chunking are already complete**.

The goal of this stage was to convert the existing text chunks into dense vector embeddings and store them, together with their text and metadata, in Qdrant so that the corpus is ready for semantic retrieval in the next stage.

---

## 1. Starting Point

Before this step, the project already had:

```text
Documents
   ↓
Parsing
   ↓
Chunking
```

The baseline chunk file used in this stage was:

```text
data/processed/chunks/baseline_recursive_500_75.jsonl
```

The chunking baseline was:

```text
Recursive chunking
chunk_size = 500
chunk_overlap = 75
```

The chunk inspection showed:

```text
1,425 chunks
```

Each chunk included text plus useful metadata such as:

```text
chunk_id
document_id
title
year
organization
country
topic
secondary_topics
source_url
```

---

## 2. Goal of This Step

The indexing pipeline implemented in this step is:

```text
Existing chunks
      ↓
Embedding model
      ↓
Dense vector embeddings
      ↓
Qdrant
      ↓
Vectors + chunk text + metadata
```

At this stage, there is still:

```text
No user query
No similarity search
No Top-K retrieval
No reranking
No LLM generation
```

Those belong to the next retrieval stage.

---

## 3. Embedding Model

### Baseline model

```text
BAAI/bge-small-en-v1.5
```

This model was selected as the **first baseline embedding model**, not necessarily the final model.

### Why this model?

- Lightweight and practical for local experimentation.
- Runs locally without an embedding API.
- Produces 384-dimensional dense embeddings.
- Integrates directly with `sentence-transformers`.
- Gives a reproducible baseline before testing stronger or hosted models.
- Fits the project's evaluation-first philosophy: the final model should be chosen using retrieval metrics, not assumption.

Possible later comparisons:

```text
A stronger open-source embedding model
OpenAI embeddings
Voyage embeddings
A multilingual embedding model
```

---

## 4. Corpus Language Check

Because `BAAI/bge-small-en-v1.5` is primarily an English model, the corpus language distribution was checked before indexing.

Result:

```text
Total chunks: 1,425

English      1,352   94.88%
Swedish         43    3.02%
Norwegian       26    1.82%
Danish           2    0.14%
Estonian         1    0.07%
Unknown          1    0.07%
```

Since almost 95% of the corpus was English, keeping BGE-small as the first baseline was considered reasonable.

The non-English portion remains a useful motivation for a later multilingual-embedding experiment.

---

## 5. Libraries Used

Installed with:

```powershell
uv add sentence-transformers qdrant-client python-dotenv tqdm
```

and for the language check:

```powershell
uv add langdetect
```

### `sentence-transformers`

Purpose:

```text
Chunk text
   ↓
Embedding model
   ↓
Dense vector
```

It loads `BAAI/bge-small-en-v1.5` and converts chunk text into embeddings.

The indexing pipeline uses:

```text
encode_document()
```

for corpus chunks.

### `qdrant-client`

Purpose:

```text
Store vectors
Store text
Store metadata
Create collections
Search collections later
```

Important separation:

```text
sentence-transformers → creates embeddings
Qdrant               → stores and later searches embeddings
```

### `python-dotenv`

Purpose:

Loads configuration from `.env`, including:

```text
Embedding model
Chunk path
Qdrant path
Collection name
Batch sizes
```

### `tqdm`

Purpose:

Shows progress bars during longer operations such as embedding many chunks.

### `langdetect`

Purpose:

Estimates chunk languages so the suitability of the English baseline model could be checked.

---

## 6. Project Configuration

The `.env` configuration used was:

```env
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

EMBEDDING_BATCH_SIZE=32

CHUNKS_PATH=data/processed/chunks/baseline_recursive_500_75.jsonl

QDRANT_PATH=data/qdrant

QDRANT_COLLECTION=nordic_green_transition_bge_small_v1

QDRANT_UPLOAD_BATCH_SIZE=128
```

### Main settings

`EMBEDDING_MODEL` — embedding model to load.

`EMBEDDING_BATCH_SIZE` — number of chunks embedded together.

`CHUNKS_PATH` — path to the already-created chunk file.

`QDRANT_PATH` — local Qdrant storage directory.

`QDRANT_COLLECTION` — collection name for this baseline experiment.

`QDRANT_UPLOAD_BATCH_SIZE` — number of points uploaded to Qdrant at once.

---

## 7. Local Qdrant Instead of HTTP Qdrant

Initially, Qdrant was configured as if a separate server were running:

```python
QdrantClient(url="http://localhost:6333")
```

That mode requires a separate Qdrant server process.

Because Docker was not available, local Qdrant mode was used instead:

```python
QdrantClient(path="data/qdrant")
```

Architecture:

```text
Python process
   │
   ├── sentence-transformers
   │       ↓
   │   embeddings
   │
   └── local Qdrant
           ↓
       data/qdrant/
```

No separate Qdrant server is required in this mode.

Later, for deployment, the project can switch to HTTP/server mode without changing the overall retrieval design.

---

## 8. Code Added Under `src/`

The `src/` directory contains reusable application logic.

```text
src/
├── config.py
├── embeddings/
│   └── embedder.py
├── utils/
│   └── chunk_loader.py
└── vectorstore/
    └── qdrant_index.py
```

### `src/config.py`

Purpose:

- loads `.env`
- exposes project settings
- centralizes model names, paths, collection names, and batch sizes

Conceptually:

```text
.env
 ↓
config.py
 ↓
project settings
```

### `src/utils/chunk_loader.py`

Purpose:

- opens the existing JSONL/JSON chunk file
- reads text and metadata
- normalizes chunk structure
- returns clean Python chunk objects

### `src/embeddings/embedder.py`

Purpose:

- loads the embedding model
- determines embedding dimension
- converts chunk texts into dense vectors
- normalizes embeddings

For this corpus:

```text
Embedding matrix shape: (1425, 384)
```

meaning:

```text
1,425 chunks
×
384 values per embedding
```

### `src/vectorstore/qdrant_index.py`

Purpose:

- initializes local Qdrant
- creates the collection
- configures vector dimension
- configures cosine distance for future retrieval
- uploads embeddings
- stores text and metadata as payload
- verifies stored point count

Configuring cosine here does **not** perform similarity search yet. It only tells Qdrant how vectors should be compared later.

---

## 9. Code Added Under `scripts/`

The `scripts/` directory contains runnable workflows.

```text
scripts/
├── inspect_chunks.py
├── check_languages.py
└── build_index.py
```

The distinction is:

```text
src/     = reusable logic
scripts/ = run a concrete task
```

### `scripts/inspect_chunks.py`

Purpose:

- load the chunk file
- verify it can be read
- print random chunks
- inspect text quality
- inspect metadata preservation

Run with:

```powershell
uv run python -m scripts.inspect_chunks
```

This was a quality-control step only. It did not create embeddings or write to Qdrant.

### `scripts/check_languages.py`

Purpose:

- detect chunk languages
- count English and non-English chunks
- validate the choice of baseline embedding model

Run with:

```powershell
uv run python -m scripts.check_languages
```

### `scripts/build_index.py`

Purpose:

Runs the full embedding/indexing pipeline:

```text
Load chunks
   ↓
Load embedding model
   ↓
Generate embeddings
   ↓
Create Qdrant collection
   ↓
Upload vectors + text + metadata
   ↓
Verify stored count
```

Run with:

```powershell
uv run python -m scripts.build_index
```

---

## 10. Why `python -m` Was Used

The first attempt used:

```powershell
uv run python scripts/inspect_chunks.py
```

This produced:

```text
ModuleNotFoundError: No module named 'src'
```

The solution was to run scripts as modules from the repository root:

```powershell
uv run python -m scripts.inspect_chunks
```

and:

```powershell
uv run python -m scripts.build_index
```

An empty file was also added:

```text
scripts/__init__.py
```

---

## 11. Errors and Fixes

### Python code entered directly into PowerShell

Python code such as:

```python
from qdrant_client import QdrantClient
```

was initially interpreted by PowerShell.

Fix:

```powershell
uv run python
```

then run the Python statements inside the Python prompt.

---

### Python syntax error in interactive mode

Two statements were accidentally pasted onto one line.

Fix: run them separately:

```python
from qdrant_client import QdrantClient
client = QdrantClient(path="data/qdrant")
```

---

### `ModuleNotFoundError: No module named 'src'`

Cause:

```powershell
uv run python scripts/inspect_chunks.py
```

Fix:

```powershell
uv run python -m scripts.inspect_chunks
```

---

### Chunk file not found

The configured path did not initially match the actual chunk location.

Fix:

```env
CHUNKS_PATH=data/processed/chunks/baseline_recursive_500_75.jsonl
```

After correction, all 1,425 chunks loaded successfully.

---

### Qdrant HTTP connection refused

The first indexing run created all embeddings successfully but failed while creating the collection.

Cause:

```python
QdrantClient(url="http://localhost:6333")
```

was still being used even though no Qdrant server was running.

Error:

```text
WinError 10061
No connection could be made because the target machine actively refused it
```

Fix:

```python
QdrantClient(path="data/qdrant")
```

and:

```env
QDRANT_PATH=data/qdrant
```

After this change, indexing completed successfully.

---

### Hugging Face unauthenticated warning

Hugging Face warned that no `HF_TOKEN` was configured.

This was not an error; the model downloaded and loaded successfully.

---

### Embedding-dimension method rename warning

A warning reported that:

```text
get_sentence_embedding_dimension
```

was renamed to:

```text
get_embedding_dimension
```

This did not affect indexing.

The code can be updated accordingly.

---

## 12. Final Successful Run

The final run completed successfully:

```text
STEP 1 — Load existing chunks
Loaded: 1,425 chunks

STEP 2 — Load embedding model
Model:     BAAI/bge-small-en-v1.5
Dimension: 384

STEP 3 — Embed chunk text
Embedding matrix shape: (1425, 384)

STEP 4 — Create Qdrant collection
Collection: nordic_green_transition_bge_small_v1
Distance metric configured for future retrieval: cosine

STEP 5 — Upload embeddings + text + metadata

INDEXING COMPLETE
Qdrant points: 1,425
```

---

## 13. What Is Stored in Qdrant?

Each point contains:

```text
384-dimensional vector
+
chunk text
+
metadata
```

Typical metadata includes:

```text
chunk_id
document_id
title
year
organization
country
topic
secondary_topics
source_url
```

This allows future retrieval results to remain traceable to the original evidence.

---

## 14. Current Project Status

Completed:

```text
Documents
   ↓
Parsing
   ↓
Chunking
   ↓
Chunk inspection
   ↓
Language analysis
   ↓
Embedding model selection
   ↓
Embedding generation
   ↓
Qdrant indexing
```

Current indexed corpus:

```text
1,425 chunks
1,425 vectors
384 dimensions per vector
```

No retrieval has been performed yet.

---

## 15. Next Step

The next stage is **dense retrieval**:

```text
User question
      ↓
Query embedding
      ↓
Qdrant vector search
      ↓
Cosine similarity
      ↓
Top-K chunks
```

The first objective will be to check whether the correct evidence appears in the retrieved Top-K chunks.

After manual sanity checks, retrieval can be evaluated using metrics such as:

```text
Recall@K
Hit Rate
MRR
NDCG
```

This establishes a measurable dense-retrieval baseline before moving to hybrid search, reranking, or LLM generation.
