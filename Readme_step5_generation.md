# Step 5 — Basic generation

This step writes an answer for each of the 90 questions. Retrieval is not run again.

The context is the top 5 parents already saved for OpenAI parent-child retrieval after the Jev Score reorder. That list is the one whose Hit@5 is 91.11%.

```text
Question
   ↓
saved Jev top 5 parents
   ↓
gpt-4.1-mini
   ↓
answer + citations
```

## Library and model

The call uses the `openai` Python SDK, which is already a project dependency. The same SDK embeds the OpenAI index and runs the evidence judge. LangChain and LlamaIndex are not used.

The model is `gpt-4.1-mini-2025-04-14`, with temperature 0. Jev Score stays the reranker. It does not write the answer.

## What the model sees

Each passage is sent with its passage number, `document_id`, and `page_start`–`page_end`.

The prompt does not include `ground_truth_answer` or `evidence_text`. Those fields are the answer key for a later check.

The instruction has three rules:

- Answer only from the five passages.
- If they do not state the answer, say so in one sentence.
- Cite the passage number, document id, and page span for each claim.

Saved citations copy the document id and page span from the retrieved parent for that rank. A rank outside the five passages is dropped.

## How to run it

```powershell
uv run python -u -m scripts.generate_answers
```

Finished questions are appended to `data/evaluation/generation_parent_child_jev_k5.jsonl`, so a rerun skips them. The full report is `data/evaluation/generation_parent_child_jev_k5.json`.

The code is `src/generation/answer.py` and `scripts/generate_answers.py`.

## What this step does not do

Answers are not scored against `ground_truth_answer`. That comparison is the next step.
