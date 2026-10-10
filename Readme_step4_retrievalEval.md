# Step 4 — Retrieval and Retrieval Evaluation

## Overview

This step introduced the first actual retrieval stage of the RAG pipeline.

Up to this point, the project already had:

- PDF parsing
- page-aware chunking
- dense embeddings
- a local Qdrant vector index

However, there was still no way to take a user question and search the indexed knowledge base.

The goal of Step 4 was therefore to:

1. embed a user question using the same embedding model used for the documents,
2. search Qdrant for the most similar chunks,
3. evaluate whether the retriever finds the known relevant evidence,
4. establish a quantitative retrieval baseline before introducing an LLM.

This was intentionally done **before generation** so retrieval quality could be measured independently from LLM answer quality.

A later check uses an LLM only as a reader. It asks whether the five retrieved parents state the labeled evidence. That judge is described in section 31. It does not generate answers.

---

# 1. Existing Retrieval Setup

The current baseline uses:

- **Embedding model:** `BAAI/bge-small-en-v1.5`
- **Embedding dimension:** 384
- **Chunking:** recursive, page-aware
- **Approximate chunk size:** 500 tokens
- **Overlap:** 75 tokens
- **Number of indexed chunks:** 1,425
- **Vector database:** Qdrant
- **Collection:** `nordic_green_transition_bge_small_v1`
- **Storage:** local Qdrant database under `data/qdrant`

The basic retrieval pipeline is:

```text
Question
   ↓
BGE-small query embedding
   ↓
384-dimensional vector
   ↓
Qdrant similarity search
   ↓
Top-k most similar chunks
```

No generative LLM is used at this stage.

The embedding model is responsible for representing the meaning of both questions and document passages in the same vector space.

---

# 2. Why Retrieval Is Evaluated Before Adding an LLM

A complete RAG system normally looks like:

```text
Question
   ↓
Retriever
   ↓
Relevant context
   ↓
LLM
   ↓
Generated answer
```

If the final answer is wrong, there are at least two possible causes:

1. the retriever failed to provide the correct evidence,
2. the retriever succeeded, but the LLM generated a poor answer.

By evaluating retrieval first, these two problems are separated.

This lets us answer a much simpler question:

> Can the system retrieve the relevant evidence before an LLM is involved?

Only after retrieval quality is understood should the generation stage be added.

---

# 3. QA Evaluation Dataset

The project already contained `QA.txt`.

Each line is one JSON object.

Example:

```json
{
  "question": "What measures does the report propose to reduce helium consumption without jeopardizing important medical and technical uses?",
  "ground_truth_answer": "The report proposes banning helium sales to private citizens or its use for entertainment, improving helium recovery from MRI machines, replacing older MRI equipment with low-helium or helium-free alternatives, imposing a refundable import tax or deposit to finance recycling, stockpiling helium for critical uses, and increasing production of substitute gases such as argon and neon.",
  "metadata": {
    "document_id": "doc_002",
    "page": 22,
    "section": "2.1.1 Reduced use of helium for entertainment purposes",
    "evidence_text": "The report proposes bans on private or entertainment helium use, improved recycling and replacement of older MRI equipment, a refundable import tax, national stockpiling, and production of argon and neon as substitutes."
  }
}
```

This dataset acts as the evaluation benchmark.

The fields are used as follows:

| Field | Current use |
|---|---|
| `question` | Query sent to the retriever |
| `document_id` | Ground-truth source document |
| `page` | Ground-truth evidence page |
| `section` | Additional structural metadata |
| `evidence_text` | Human-readable reference evidence |
| `ground_truth_answer` | Reserved mainly for later generation evaluation |

For the current baseline, retrieval is considered correct when the returned chunk belongs to the labeled document and its page span contains the labeled page.

---

# 4. Dense Retriever

A simple dense retriever was added.

Example implementation:

```python
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(
        self,
        qdrant_path: str = "data/qdrant",
        collection_name: str = "nordic_green_transition_bge_small_v1",
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str) -> list[float]:
        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
        )

        return query_vector.tolist()

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        query_vector = self.embed_query(query)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        retrieved = []

        for rank, point in enumerate(results, start=1):
            payload = point.payload or {}

            retrieved.append(
                {
                    "rank": rank,
                    "score": float(point.score),
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "page": payload.get("page"),
                    "section": payload.get("section"),
                    "text": payload.get("text", ""),
                }
            )

        return retrieved
```

The retriever performs three operations:

1. encode the query,
2. send that vector to Qdrant,
3. return the highest-scoring chunks together with metadata.

---

# 5. Retrieval Evaluation Code

The evaluation script loads every question in `QA.txt`, retrieves the top 10 chunks, and checks where the first ground-truth passage appears.

The baseline evaluates:

- Hit@1
- Hit@3
- Hit@5
- Hit@10
- Mean Reciprocal Rank (MRR)

A simplified version of the evaluation logic is:

```python
def is_relevant(
    result: dict,
    expected_document_id: str,
    expected_page: int,
) -> bool:
    if result.get("document_id") != expected_document_id:
        return False

    return page_matches(
        result.get("page"),
        expected_page,
    )
```

For every question, the evaluator finds the rank of the first relevant result.

Example:

```text
Expected:
doc_002 / page 22

Retrieved:
1. doc_005 / page 18
2. doc_002 / page 21
3. doc_002 / page 22
```

In this case:

```text
Hit@1 = 0
Hit@3 = 1
Hit@5 = 1
Hit@10 = 1

Reciprocal Rank = 1 / 3
```

The evaluator repeats this process across all questions.

---

# 6. Evaluation Metrics

## Hit@k

Hit@k measures whether at least one relevant chunk appears among the first `k` retrieval results.

For example:

```text
Hit@1
```

means:

> Was the labeled evidence the very first retrieved result?

And:

```text
Hit@10
```

means:

> Was the labeled evidence found anywhere in the first ten results?

---

## Mean Reciprocal Rank

MRR measures how high the first correct result appears.

If the first correct result is:

```text
rank 1 → reciprocal rank = 1.0
rank 2 → reciprocal rank = 0.5
rank 3 → reciprocal rank = 0.333
rank 10 → reciprocal rank = 0.1
```

MRR is the average of these reciprocal ranks across all questions.

Unlike Hit@10, MRR therefore rewards good ranking.

---

## Precision, recall, and F1

Hit@k only asks whether one relevant unit appears. Precision, recall, and F1 count how many of the retrieved units cover the labeled page.

A unit is relevant under the same rule as Hit@k: the `document_id` matches, and the labeled page lies inside `page_start`–`page_end`.

At cutoff `k`:

```text
Precision@k = relevant units in the first k / k
Recall@k    = relevant units in the first k / relevant units in the index
F1@k        = harmonic mean of that precision and that recall
```

The reported value is the average of the 90 per-question scores. Precision@1 is the same number as Hit@1, because the first position is either relevant or not.

Recall depends on how many indexed units contain the labeled page. For the fixed-size chunks, the semantic chunks, and the parent-child parents, that count is usually 2 and never more than 4. For sentence-window it is usually 42, because overlapping windows all contain the same page. Sentence-window recall is therefore not on the same scale as the other methods.

The scores are in section 27.

---

## Evidence judge

Hit@k, precision, recall, and F1 all use the labeled page. They count a passage as a miss when the evidence is stated on another page.

For the selected OpenAI parent-child list of 5, `gpt-4.1-mini-2025-04-14` reads each parent and compares it with `evidence_text`. The labels are `supports`, `partial`, and `miss`. Evidence Hit@5 counts only `supports`. The prompt, the scores, and the comparison with the strict page metric are in section 31.

---

# 7. Baseline Results

The evaluation dataset contains **90 questions**.

The dense baseline produced:

| Metric | Score |
|---|---:|
| Questions | 90 |
| Hit@1 | 37.78% |
| Hit@3 | 61.11% |
| Hit@5 | 73.33% |
| Hit@10 | 88.89% |
| MRR | 0.5331 |

This means approximately:

- 34 of 90 questions retrieve the labeled evidence at rank 1,
- 55 of 90 retrieve it in the first 3,
- 66 of 90 retrieve it in the first 5,
- 80 of 90 retrieve it in the first 10.

The first interpretation was that retrieval was relatively strong at larger `k`, but ranking needed improvement.

However, detailed error analysis showed an even more important result.

---

# 8. Error Analysis

The retrieval baseline was analyzed before making any further changes.

Among the 90 questions:

- **10** are Hit@10 misses,
- **14** have their first labeled hit at ranks 6–10.

These 24 cases were inspected manually.

The most important discovery was:

- **10/10 Hit@10 misses still contain the correct report somewhere in the top 10**
- **9/10 Hit@10 misses rank the correct report at rank 1**

Therefore the main problem is not:

> finding the wrong report

but rather:

> finding the correct report and ranking the wrong passage from that report above the labeled passage.

This is a much narrower problem.

---

# 9. Failure Categories

The 24 difficult cases were grouped as follows:

| What happened | Hit@10 misses | Ranks 6–10 |
|---|---:|---:|
| Neighboring page holds the passage | 3 | 3 |
| Summary or later restatement ranks higher | 2 | 6 |
| Same report, different section | 3 | 3 |
| Different report ranks first | 1 | 2 |
| Table of contents ranks first | 1 | 0 |

Several important conclusions follow from this.

---

# 10. Neighboring Pages Are Not Always Real Failures

Some cases are labeled as failures only because the evaluation currently requires the exact annotated page.

For example:

```text
Ground truth:
doc_022 / page 107
```

but the retrieved chunk may contain the same evidence on page 106, 108, or a chunk spanning neighboring pages.

For a practical RAG system, such retrieval can still be perfectly useful.

The current page-based metric is therefore intentionally strict.

Later, evaluation should distinguish between:

```text
exact annotated-page retrieval
```

and:

```text
answer-supporting evidence retrieval
```

---

# 11. Summaries and Restatements Can Be Valid Retrievals

A report often repeats an idea.

For example:

```text
Page 27:
detailed recommendation

Page 110:
summary of the same recommendation
```

The QA dataset may label page 27 as the evidence page.

However, the embedding model may rank page 110 above it because the summary is semantically closer to the wording of the question.

Under the current evaluator:

```text
page 110 = incorrect
```

but for RAG:

```text
page 110 may still contain fully sufficient evidence
```

This means the current Hit@k metrics measure exact annotation retrieval, not necessarily whether the system retrieved usable evidence.

---

# 12. Main Ranking Problem: Wrong Section in the Correct Report

The clearest real retrieval weakness is when BGE correctly identifies the report but selects another semantically related section.

Example:

Question:

```text
Why does the report recommend expanding and upgrading
Extended Producer Responsibility schemes for CRM-rich waste?
```

Ground truth:

```text
doc_013 / page 86
```

Top retrieval:

```text
doc_013 / page 63
```

The rank-1 passage discusses:

- CRM-enriched waste streams,
- collection and sorting,
- EPR schemes,

so it is semantically highly related.

However, it does not directly contain the specific recommendation asked by the question.

This explains why dense embeddings can retrieve the right topic while still ranking the wrong passage first.

---

# 13. Example Difficult Case

For the EPR question above, the top results included:

| Rank | Document | Pages | Score |
|---:|---|---|---:|
| 1 | doc_013 | 63 | 0.745 |
| 2 | doc_013 | 19 | 0.745 |
| 3 | doc_010 | 25–26 | 0.741 |
| 4 | doc_013 | 78–79 | 0.730 |
| 5 | doc_013 | 9–11 | 0.730 |
| 6 | doc_013 | 11–12 | 0.728 |
| 7 | doc_002 | 3–4 | 0.724 |
| 8 | doc_004 | 13 | 0.723 |
| 9 | doc_013 | 90–91 | 0.721 |
| 10 | doc_013 | 172–173 | 0.720 |

The labeled page 86 is outside the top 10.

But seven of the top ten results are from the correct report.

This demonstrates that document-level semantic retrieval is working well, while passage-level discrimination is weaker.

---

# 14. Table of Contents Problem

One failure was caused by the table of contents.

A table of contents contains many important keywords and section titles, making it highly similar to many queries.

However, it usually does not provide useful evidence.

This is therefore not primarily an embedding-model problem.

A later improvement should consider filtering structural pages such as:

- table of contents,
- copyright pages,
- blank pages,
- possibly references or other non-answer-bearing sections.

---

# 15. What We Learned

The initial metrics alone suggested:

```text
Hit@10 = 88.89%
```

and therefore ten retrieval failures.

The error analysis showed that this interpretation was too simple.

The retriever is actually very strong at identifying the relevant report.

The more precise conclusion is:

> The dense retriever usually finds the correct document and the correct semantic neighborhood, but it does not always distinguish the exact best passage from repeated, neighboring, summarized, or closely related passages within the same report.

This is an important result because it tells us what kind of improvement to try next.

---

# 16. Why We Should Not Immediately Change the Embedding Model

The analysis does not currently suggest that the embedding model completely fails to understand the queries.

If the model were fundamentally poor, we would expect many cases where:

```text
correct report is absent from the top 10
```

Instead:

```text
10 / 10 Hit@10 misses still contain the correct report.
```

And:

```text
9 / 10 have the correct report at rank 1.
```

Therefore replacing the embedding model or changing chunk size was not the first response to the error analysis.

A controlled embedding comparison was still run afterward, with the chunks and the questions held fixed. That comparison is documented in section 20. It measures whether a stronger model moves the strict page metric. It does not replace the finding that most misses are the right report and the wrong passage.

---

# 17. Reranking

The current situation is well suited to reranking.

Dense retrieval can first retrieve a larger candidate set:

```text
Question
   ↓
BGE dense retrieval
   ↓
Top 10 or Top 20 candidate chunks
```

Then a reranker can compare the question and each candidate passage more carefully:

```text
Question + candidate 1
Question + candidate 2
...
Question + candidate N
        ↓
Reranker
        ↓
Reordered candidates
```

Dense embeddings are efficient because the question and chunks are encoded independently.

A reranker can be more precise because it evaluates the question and passage together.

This may help distinguish:

```text
passage that is generally about EPR
```

from:

```text
passage that directly answers why EPR should be expanded
```

That reranking pass was later run on the BGE index as experiment E3. The scores are in section 26. It was not repeated on the OpenAI index.

---

# 18. Evaluation Should Also Be Improved

The current evaluation should be preserved because it provides a useful strict baseline.

However, future evaluation should report at least two concepts separately.

## Strict annotated-page retrieval

Did the retrieved result contain the exact page labeled in the QA dataset?

For example:

```text
Hit@5_exact
```

This is what the current metrics measure.

---

## Evidence-level retrieval

Did the retrieved chunk contain enough evidence to answer the question correctly, even if it came from another page or summary?

For example:

```text
Hit@5_evidence
```

This would better represent real RAG performance.

The existing `evidence_text` field can help support this evaluation.

---

# 19. Baseline to Preserve

This step establishes the project's first retrieval benchmark:

```text
Experiment E0 — Dense Retrieval Baseline

Embedding model:
BAAI/bge-small-en-v1.5

Chunking:
500-token recursive chunks
75-token overlap

Vector store:
Qdrant

Questions:
90

Hit@1:
37.78%

Hit@3:
61.11%

Hit@5:
73.33%

Hit@10:
88.89%

MRR:
0.5331
```

These results should not be overwritten when future experiments are added.

Future experiments should be compared against this baseline.

---

# 20. Embedding Model Comparison

After the BGE-small baseline and the failure analysis, the same retrieval test was repeated with three other embedding models.

The comparison holds everything else fixed:

- the same 1,425 chunks,
- the same 90 questions in `QA.txt`,
- the same relevance rule: correct `document_id`, and the labeled page inside `page_start`–`page_end`,
- the same metrics: Hit@1, Hit@3, Hit@5, Hit@10, and MRR,
- top 10 results per question.

The BGE collection `nordic_green_transition_bge_small_v1` was not rebuilt. Each new model was stored in its own Qdrant collection under `data/qdrant`.

Run the comparison with:

```powershell
uv run python -u -m scripts.compare_embedding_models
```

If a collection already has 1,425 points and its report already exists, that model is reused. The script refuses to recreate the BGE baseline collection.

---

# 21. Models in the Comparison

| Model | Where it runs | Dimensions | Qdrant collection |
|---|---|---:|---|
| `BAAI/bge-small-en-v1.5` | Local | 384 | `nordic_green_transition_bge_small_v1` |
| `Qwen/Qwen3-Embedding-0.6B` | Local | 1,024 | `nordic_green_transition_qwen3_0_6b` |
| `gemini-embedding-2` | Gemini API | 3,072 | `nordic_green_transition_gemini_002` |
| `text-embedding-3-large` | OpenAI API | 3,072 | `nordic_green_transition_openai_3_large` |

API keys stay in `.env`, which is listed in `.gitignore`:

```env
OPENAI_API_KEY=
GEMINI_API_KEY=
```

The BGE query prefix is not reused for the other models. Each model uses its own query format.

**BGE-small.** Questions are prefixed with `Represent this sentence for searching relevant passages: `. Passages are embedded without that prefix.

**Qwen3-Embedding-0.6B.** Questions use the model’s retrieval instruction:

```text
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:
```

Passages are embedded with no instruction. Qwen3 also has 4B and 8B embedding models. Those were not run. The 0.6B model is the size that fits this machine. The larger models are the same family with more capacity, and they need much more memory and time.

**OpenAI `text-embedding-3-large`.** Questions and passages use the same embeddings endpoint. There is no extra instruction prefix. Vectors are 3,072-dimensional.

**Gemini `gemini-embedding-2`.** This model does not take a `task_type` argument. The task is written into the text:

```text
Documents: title: none | text: {chunk}
Questions: task: search result | query: {question}
```

The first Gemini attempt used `gemini-embedding-001`. On the free plan that model allows 1,000 embedding requests per day. This corpus needs 1,425 chunk vectors plus 90 question vectors, so the run stopped before any `gemini-embedding-001` vectors were saved. `gemini-embedding-2` then hit the same daily cap after 992 chunk vectors. Those vectors were written to `data/embeddings/gemini_embedding_2.npy`. After the Gemini key was moved to a paid plan, the comparison continued from that file and finished the index.

Code for this comparison:

```text
src/embeddings/api_embedder.py
src/embeddings/qwen_embedder.py
src/retrieval/api_retriever.py
scripts/compare_embedding_models.py
```

Saved reports:

```text
data/evaluation/retrieval_baseline.json
data/evaluation/retrieval_openai_3_large.json
data/evaluation/retrieval_gemini_002.json
data/evaluation/retrieval_qwen3_0_6b.json
data/evaluation/embedding_model_comparison.json
```

---

# 22. Comparison Results

| Model | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| BGE-small-en-v1.5 | 37.78% | 61.11% | 73.33% | 88.89% | 0.5331 |
| Qwen3-Embedding-0.6B | 45.56% | 65.56% | 76.67% | 84.44% | 0.5812 |
| Gemini gemini-embedding-2 | 42.22% | 71.11% | 81.11% | 88.89% | 0.5747 |
| OpenAI text-embedding-3-large | 43.33% | 70.00% | 80.00% | 91.11% | 0.5885 |

In question counts, out of 90:

| Model | Hit@1 | Hit@3 | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|
| BGE-small | 34 | 55 | 66 | 80 |
| Qwen3-0.6B | 41 | 59 | 69 | 76 |
| Gemini embedding-2 | 38 | 64 | 73 | 80 |
| OpenAI text-embedding-3-large | 39 | 63 | 72 | 82 |

OpenAI has the highest MRR and the highest Hit@10. Qwen3-0.6B places the labeled page first most often, then drops it out of the top 10 more often than the others. Gemini has the highest Hit@3 and Hit@5. Its Hit@10 matches the BGE baseline.

All three new models beat BGE-small on Hit@1 and on MRR. The Hit@10 gain is small, except for Qwen, where Hit@10 is lower than the baseline. That fits the earlier failure analysis: the baseline already finds the right report, so a stronger embedding model mainly changes which passage inside that report ranks first. It does not remove cases where a summary or a neighboring page says the same thing as the labeled page. Under this strict metric those cases remain misses for every model.

---

# 23. Chunking Comparison

After the embedding comparison, OpenAI `text-embedding-3-large` was kept as the embedding model. The 500-token recursive chunks stayed as the control. Three other ways of cutting the same reports were then indexed and scored.

Everything else stays fixed:

- the same 9 parsed reports,
- the same 90 questions in `QA.txt`,
- the same relevance rule: correct `document_id`, and the labeled page inside `page_start`–`page_end`,
- the same metrics: Hit@1, Hit@3, Hit@5, Hit@10, and MRR,
- 10 passages returned per question.

The fixed-size OpenAI collection `nordic_green_transition_openai_3_large` was not rebuilt. Each new method has its own collection.

Run the comparison with:

```powershell
uv run python -u -m scripts.compare_chunking_methods
```

A finished collection is reused when its point count matches the saved units and its report already exists. The script refuses to recreate the BGE collection or the three fixed-size collections from the embedding comparison.

---

# 24. The Three Chunking Methods

The returned passage is kept near the fixed-size chunk. On this corpus those chunks have a median of 450 tokens and a mean of 414. A sentence here has a median of 18 tokens.

| Method | What is embedded | What is returned and scored | Size |
|---|---|---|---|
| Fixed-size control | The 500-token chunk | That same chunk | 500 target, 75 overlap, minimum 100. Already indexed. |
| Sentence-window | One sentence | Neighbors around that sentence | Grow equally left and right until the next sentence would pass 500 tokens. 24,821 sentences. Returned windows have a median of 468 tokens and a maximum of 497. |
| Semantic | The semantic chunk | That same chunk | Break where the cosine distance between adjacent sentence embeddings is high. The cutoff is the 93rd percentile of those distances, chosen because it puts the median chunk at 453 tokens. Cap 500. Floor 100. 1,517 chunks. |
| Parent-child | A child piece | The existing 500-token parent | Child target 125 tokens, overlap 20. 6,287 children from the 1,425 parents. The child median is 105 tokens because the last piece of a parent is shorter. Search fetches 80 children and keeps 10 unique parents. |

Sentence-window and semantic chunking start from the parsed pages, using the same page cleaning as the fixed-size chunker. Parent-child starts from the existing chunks, so the parent text and the parent page span are unchanged.

Semantic chunking did not produce many chunks that were naturally about 450 tokens. 1,013 of the 1,517 semantic chunks were a topic group that ran past 500 tokens and was then split with the 500/75 rule. The median of 453 is mostly that cap. A few leftovers are shorter than 100 tokens because merging them would have passed 500. The smallest is 9 tokens.

Code and saved units:

```text
src/chunking/comparison_units.py
src/retrieval/structured_retriever.py
scripts/compare_chunking_methods.py

data/processed/chunks/sentence_windows.jsonl
data/processed/chunks/semantic_chunks.jsonl
data/processed/chunks/parent_children.jsonl
```

Collections:

```text
nordic_green_transition_openai_sentence_window
nordic_green_transition_openai_semantic
nordic_green_transition_openai_parent_child
```

Reports:

```text
data/evaluation/retrieval_openai_sentence_window.json
data/evaluation/retrieval_openai_semantic.json
data/evaluation/retrieval_openai_parent_child.json
data/evaluation/chunking_method_comparison.json
```

---

# 25. BGE Filter and Reranker Experiments

Before the embedding swap, three retrieval changes were scored on the frozen BGE index. They are included here so the later table can be read against every saved evaluation. They were not rerun with OpenAI.

| Experiment | What changed | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---:|---:|---:|---:|---:|
| E0, BGE fixed | Dense top 10 | 37.78% | 61.11% | 73.33% | 88.89% | 0.5331 |
| E1, structural filter | Drop TOC, imprint, bibliography, and blank chunks, then take top 10. 24 chunks removed. A chunk that covers a labeled page is kept. | 38.89% | 63.33% | 74.44% | 90.00% | 0.5445 |
| E2, wider pool | Same filter, candidate pool of 20 | 38.89% | 63.33% | 74.44% | 90.00% | 0.5500 |
| E3, reranker | Reorder that pool with `BAAI/bge-reranker-base`. Metrics use the reranked top 10. | 42.22% | 74.44% | 77.78% | 92.22% | 0.5888 |

E2 also records Hit@20 of 97.78%. Hit@1 through Hit@10 match E1, because the first 10 dense hits are the same list. MRR is slightly higher because a relevant page that appears at ranks 11–20 now counts.

Reports:

```text
data/evaluation/e1_structural_filter.json
data/evaluation/e2_dense_top20.json
data/evaluation/e3_reranker_top10.json
data/evaluation/experiment_comparison.json
```

---

# 26. All Saved Retrieval Results

| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| BGE-small fixed | 37.78% | 61.11% | 73.33% | 88.89% | 0.5331 |
| Qwen3-0.6B fixed | 45.56% | 65.56% | 76.67% | 84.44% | 0.5812 |
| Gemini embedding-2 fixed | 42.22% | 71.11% | 81.11% | 88.89% | 0.5747 |
| OpenAI fixed 500/75 | 43.33% | 70.00% | 80.00% | 91.11% | 0.5885 |
| BGE E1 filter | 38.89% | 63.33% | 74.44% | 90.00% | 0.5445 |
| BGE E2 top-20 | 38.89% | 63.33% | 74.44% | 90.00% | 0.5500 |
| BGE E3 reranker | 42.22% | 74.44% | 77.78% | 92.22% | 0.5888 |
| OpenAI sentence-window | 33.33% | 63.33% | 80.00% | 88.89% | 0.5237 |
| OpenAI semantic | 48.89% | 66.67% | 76.67% | 87.78% | 0.6082 |
| OpenAI parent-child | 52.22% | 74.44% | 83.33% | 96.67% | 0.6591 |

In question counts, out of 90, for the three OpenAI chunking runs:

| Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 |
|---|---:|---:|---:|---:|
| OpenAI fixed 500/75 | 39 | 63 | 72 | 82 |
| OpenAI sentence-window | 30 | 57 | 72 | 80 |
| OpenAI semantic | 44 | 60 | 69 | 79 |
| OpenAI parent-child | 47 | 67 | 75 | 87 |

Parent-child is the strongest row. It is the only method above 50% Hit@1, and it places the labeled page in the top 10 for 87 of 90 questions. The search unit is small enough to find the passage. The returned unit is still the original parent, so the labeled page stays inside the page span.

Semantic chunking raises Hit@1 and MRR above the fixed OpenAI chunks. Hit@3, Hit@5, and Hit@10 are lower. Most of its chunks were cut by the 500-token cap, so this is a small change in where the boundary falls, not a different chunk length.

Sentence-window is the weakest OpenAI run. Embedding one sentence often matches the right topic and the wrong page. The 500-token window around that sentence does not recover the labeled page often enough to beat the fixed chunk.

The BGE reranker reaches the same MRR as fixed-size OpenAI, 0.5888 against 0.5885, and a higher Hit@3. It does not reach the OpenAI parent-child scores. That reranker has not been applied to the OpenAI parent-child pool.

---

# 27. Precision, Recall, and F1

These scores were computed from the saved top-k lists. No collection was searched again.

```powershell
uv run python -u -m scripts.score_precision_recall
```

The relevant-set sizes and the formulas are in section 6. The numbers below are also stored on each method in `data/evaluation/chunking_method_comparison.json`.

### @1

Precision@1 is the same number as Hit@1.

| Method | P@1 | R@1 | F1@1 |
|---|---:|---:|---:|
| BGE-small fixed | 37.78% | 17.78% | 23.70% |
| Qwen3-0.6B fixed | 45.56% | 20.65% | 27.85% |
| Gemini embedding-2 fixed | 42.22% | 18.24% | 25.26% |
| OpenAI fixed 500/75 | 43.33% | 18.24% | 25.44% |
| BGE E1 filter | 38.89% | 18.15% | 24.26% |
| BGE E2 top-20 | 38.89% | 18.15% | 24.26% |
| BGE E3 reranker | 42.22% | 19.17% | 25.81% |
| OpenAI sentence-window | 33.33% | 0.83% | 1.61% |
| OpenAI semantic | 48.89% | 23.52% | 31.11% |
| OpenAI parent-child | 52.22% | 24.35% | 32.67% |

### @3

| Method | P@3 | R@3 | F1@3 |
|---|---:|---:|---:|
| BGE-small fixed | 26.30% | 34.07% | 29.19% |
| Qwen3-0.6B fixed | 28.89% | 38.43% | 32.35% |
| Gemini embedding-2 fixed | 35.93% | 45.74% | 39.71% |
| OpenAI fixed 500/75 | 34.81% | 44.07% | 38.38% |
| BGE E1 filter | 27.04% | 35.74% | 30.19% |
| BGE E2 top-20 | 27.04% | 35.74% | 30.19% |
| BGE E3 reranker | 29.63% | 38.52% | 32.93% |
| OpenAI sentence-window | 26.30% | 1.95% | 3.62% |
| OpenAI semantic | 30.00% | 41.39% | 34.06% |
| OpenAI parent-child | 31.11% | 42.50% | 35.32% |

### @5

| Method | P@5 | R@5 | F1@5 |
|---|---:|---:|---:|
| BGE-small fixed | 20.89% | 45.19% | 28.13% |
| Qwen3-0.6B fixed | 22.44% | 48.89% | 30.30% |
| Gemini embedding-2 fixed | 28.00% | 59.63% | 37.69% |
| OpenAI fixed 500/75 | 28.00% | 58.89% | 37.53% |
| BGE E1 filter | 21.33% | 46.30% | 28.77% |
| BGE E2 top-20 | 21.33% | 46.30% | 28.77% |
| BGE E3 reranker | 20.89% | 44.26% | 27.98% |
| OpenAI sentence-window | 21.11% | 2.55% | 4.53% |
| OpenAI semantic | 22.44% | 52.13% | 30.85% |
| OpenAI parent-child | 23.33% | 51.94% | 31.69% |

### @10

| Method | P@10 | R@10 | F1@10 |
|---|---:|---:|---:|
| BGE-small fixed | 14.00% | 60.00% | 22.48% |
| Qwen3-0.6B fixed | 14.11% | 60.65% | 22.66% |
| Gemini embedding-2 fixed | 17.33% | 73.15% | 27.78% |
| OpenAI fixed 500/75 | 17.67% | 75.09% | 28.33% |
| BGE E1 filter | 14.22% | 60.93% | 22.83% |
| BGE E2 top-20 | 14.22% | 60.93% | 22.83% |
| BGE E3 reranker | 14.33% | 61.39% | 23.01% |
| OpenAI sentence-window | 13.33% | 3.19% | 5.11% |
| OpenAI semantic | 15.22% | 69.07% | 24.65% |
| OpenAI parent-child | 15.67% | 68.15% | 25.21% |

Precision falls as the list gets longer. A question has only one to four chunks that contain the labeled page, so most of a ten-slot list is irrelevant even when the page is found.

Parent-child has the highest F1@1, 32.67%. Gemini has the highest F1@3, 39.71%, and the highest F1@5, 37.69%. Fixed-size OpenAI has the highest F1@10, 28.33%, and the highest recall@10, 75.09%. Parent-child still leads Hit@k and MRR. On recall and F1 past the first rank it is behind Gemini and fixed-size OpenAI, because it more often places one covering parent in the list and less often places every covering chunk there.

Sentence-window precision stays near the other methods: 33.33% at 1 and 13.33% at 10. Its recall stays near 1–3% because the denominator is every overlapping window that contains the page, a median of 42 windows, while the run returns at most 10. That F1 is not on the same scale as the fixed-chunk F1.

F1@3 and F1@5 are higher than F1@10 for every method except sentence-window. The extra list positions add fewer relevant chunks than they add slots, so precision falls faster than recall rises.

---

# 28. Selection

The saved tables support a two-step choice.

**1. Choose the retrieval setup from MRR and Hit@k.**

MRR shows whether a relevant chunk is ranked high. Hit@k shows whether the labeled page is found at all. F1 is not used in this step.

On those metrics, OpenAI parent-child is the strongest setup in the comparison:

- best Hit@1, 52.22%
- best Hit@5, 83.33%
- best Hit@10, 96.67%
- best MRR, 0.6591

Parent-child is a chunking and retrieval setup. It is not an embedding model. The chunks are embedded with OpenAI `text-embedding-3-large`. The child is the 125-token piece that is embedded. The parent is the existing 500-token chunk that is returned.

If the question is only which embedding model to keep, compare the fixed-chunk rows: BGE-small, Qwen3-0.6B, Gemini embedding-2, and OpenAI. Among those, OpenAI fixed 500/75 is the strongest overall. Gemini is close. Qwen3-0.6B has the best Hit@1 of the four fixed-chunk models and the weakest Hit@10.

**2. Choose k for that setup from F1@k and Hit@k.**

F1 is the precision/recall balance of the returned list. Hit@k is the extra coverage gained by a longer list.

For OpenAI parent-child:

| k | Hit@k | F1@k |
|---:|---:|---:|
| 1 | 52.22% | 32.67% |
| 3 | 74.44% | 35.32% |
| 5 | 83.33% | 31.69% |
| 10 | 96.67% | 25.21% |

Within this setup, k=3 has the best F1 and already finds the labeled page for about three questions in four. k=5 gives up some F1 and raises Hit from 74.44% to 83.33%. k=10 reaches 96.67% Hit and drops F1 to 25.21%, so most of the ten passages do not cover the labeled page.

The selected operating point is OpenAI parent-child with k=5. Hit@5 is 83.33%, F1 remains in the same range as k=3, and the list is half as long as k=10. k=3 is the cleaner list. k=5 is the one carried forward so the generator is more likely to receive the labeled page.

---

# 29. What Was Accomplished in Step 4

At the end of this step, the project now has:

- query embedding,
- Qdrant similarity search,
- configurable top-k retrieval,
- retrieval metadata including rank and similarity score,
- automated evaluation over all 90 QA questions,
- Hit@1, Hit@3, Hit@5, Hit@10, and MRR,
- macro precision, recall, and F1 at those cutoffs,
- a frozen BGE-small baseline,
- the same evaluation on OpenAI, Gemini, and Qwen3-0.6B,
- separate Qdrant collections for each embedding model,
- BGE experiments for filtering, a pool of 20, and reranking,
- sentence-window, semantic, and parent-child retrieval on OpenAI,
- an evidence judge on the selected OpenAI parent-child top 5,
- saved retrieval reports,
- failure analysis of the BGE baseline.

The selected retrieval setup is OpenAI `text-embedding-3-large` with parent-child units, returning 5 parents. The 125-token child is what gets embedded. The existing 500-token chunk is what gets returned.

---

# 30. Current Pipeline Status

```text
1. Parsing                         Done

2. Fixed-size chunking             Done
   500 tokens, 75 overlap

3. BGE embeddings + indexing       Done

4. Dense retrieval                 Done

5. Retrieval evaluation            Baseline completed

6. Embedding model comparison      Done
   OpenAI selected

7. BGE filter and reranker         Done
   E1, E2, E3

8. Chunking comparison             Done
   Selected: OpenAI parent-child, k=5

9. Evidence judge                  Done
   gpt-4.1-mini on the top 5 parents
   Evidence Hit@5: 78.89%
   Partial only: 21.11%

10. LLM generation                 Not started

11. Final RAG evaluation           Not started
```

The setup carried forward is OpenAI embeddings, 125-token children, the existing 500-token parents, and k=5. The evidence judge finds a full statement of the labeled evidence in 71 of 90 questions, and a partial statement in the other 19. Generation has not been started.

---

# 31. Evidence Judge

The strict page score cannot see a passage that states the evidence on a different page. For the selected setup, OpenAI parent-child at k=5, a judge reads each returned parent and compares it with the labeled `evidence_text`.

The judge is `gpt-4.1-mini-2025-04-14`, with temperature 0. Each of the five parents is judged alone. The prompt contains the question, the reference evidence, and the passage text. It does not contain the page number, the document id, the rank, or the similarity score.

The labels are:

- `supports`: the passage states the reference evidence, including its specific claims
- `partial`: the passage states some of that evidence and leaves out a material part
- `miss`: the passage does not state the reference evidence

A related paragraph is not `supports`. `partial` is reported on its own and is not counted as a hit. Evidence Hit@5 is true when any of the five labels is `supports`.

```powershell
uv run python -u -m scripts.judge_retrieval
```

The run is saved in `data/evaluation/retrieval_judge_parent_child_k5.json`. Raw labels are in `data/evaluation/judge_parent_child_k5.jsonl`.

| | Questions | Share |
|---|---:|---:|
| Strict Hit@5, labeled page in the five parents | 75 | 83.33% |
| Evidence Hit@5, at least one `supports` | 71 | 78.89% |
| Partial only, no `supports` | 19 | 21.11% |
| Neither `supports` nor `partial` | 0 | 0% |

Across the 450 passages, 99 are `supports` (22.00%), 174 are `partial` (38.67%), and 177 are `miss` (39.33%). Evidence precision@5 is 22.00%.

Of the 90 questions:

- 67 are both a strict hit and a `supports`
- 8 contain the labeled page, and the judge calls the five passages only `partial`
- 4 miss the labeled page, and a retrieved passage still `supports` the evidence
- 11 miss the labeled page, and the closest passage is `partial`

Every question has at least a partial statement of the evidence in the five parents. The strict metric is slightly higher than full evidence support, because covering the labeled page does not always mean the parent states every claim in `evidence_text`. The 4 strict misses that still `support` the evidence are the cases the page rule treats as failures and the judge treats as usable.

---

# 32. Conclusion

This project will use **OpenAI parent-child retrieval with k=5**.

The embedding model is OpenAI `text-embedding-3-large`. Parent-child is the retrieval setup on top of that model, not a separate embedder. Each existing 500-token chunk is the parent that gets returned. That parent is split into children of about 125 tokens, and those children are what get embedded and searched. At query time the search returns the 5 unique parents of the best children.

That choice comes from the two comparisons in this step. Among fixed-size chunks, OpenAI was the strongest embedding model, with Gemini close behind. Among retrieval setups on that OpenAI index, parent-child had the best Hit@1 (52.22%), the best Hit@5 (83.33%), the best Hit@10 (96.67%), and the best MRR (0.6591). k=5 is the list length carried forward: Hit@5 is 83.33% and F1@5 is 31.69%. k=3 has a slightly higher F1 and misses the labeled page more often. k=10 finds the page more often and fills the list with passages that do not cover it.

An evidence judge on those five parents, `gpt-4.1-mini-2025-04-14`, finds a full statement of the labeled evidence for 71 of the 90 questions (78.89%). The other 19 have only a partial statement. None of the 90 is empty of the evidence. Generation is the next stage. It will receive those 5 parent passages. The BGE baseline, the other embedding indexes, and the sentence-window and semantic indexes stay in place as comparison results. They are not the setup used from here.
