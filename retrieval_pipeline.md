# Retrieval Pipeline

This note describes the current retrieval path and tracks the recent
adaptability cleanup. It is separate from `DESIGN.md` so we can keep working
notes without making the submission document noisy.

## Current Flow

```text
Question
  |
  v
Embed query
  - all-MiniLM-L6-v2
  - same embedding model used for transcript chunks
  |
  v
Hybrid candidate retrieval
  |
  +-- Dense search
  |     - ChromaDB persistent collection
  |     - cosine distance over chunk embeddings
  |
  +-- Lexical search
        - token coverage
        - numbers
        - proper names
        - configured concept aliases
        - literal query-term injection for reranking pool
  |
  v
Merge and deduplicate candidates
  |
  v
Cross-encoder reranking
  - cross-encoder/ms-marco-MiniLM-L-6-v2
  - scores each (question, chunk) pair
  - filters weak candidates with CE threshold
  |
  v
Source selection
  |
  +-- Local mode
  |     - strongest anchor chunk
  |     - nearby transcript neighbors for context
  |
  +-- Diverse mode
        - strongest anchor chunk
        - additional chunks covering different query terms/topics
  |
  v
Prompt builder
  - selected excerpts only
  - timestamps included in context labels
  - full transcript is never sent to the LLM
  |
  v
LLM answer
  |
  v
API response
  - natural-language answer
  - ranked source references
```

## Previous State

The retrieval behavior was working, but several corpus-specific hints were
hardcoded directly inside Python modules:

- `app/chunker.py` contained fixed regexes for historical era detection:
  `victorian`, `edwardian`, `tudor`, and `postwar`.
- `app/retriever.py` contained fixed concept aliases, such as
  `legislation -> law/regulation`.
- `app/retriever.py` also contained hardcoded broad-query and comparison
  markers, such as `both`, `compare`, `versus`, and `different from`.
- Era query filters were duplicated separately from chunk-era detection.
- Changing corpus metadata rules required code edits and could accidentally
  reuse a stale Chroma index.

That made the system feel more tied to one documentary than necessary. The
core RAG architecture was reusable, but the domain hints were buried in code.

## Adaptability Refactor

We introduced `config/retrieval.yaml` as the home for corpus-specific retrieval
rules:

- `metadata.eras`: chunk-time and query-time era patterns.
- `lexical.concept_aliases`: query terms and their transcript wording aliases.
- `query_intent.diverse_evidence_markers`: phrases that indicate broad or
  multi-topic evidence should be selected.
- `query_intent.explicit_comparison_markers`: phrases that indicate comparison
  behavior.

Then we changed the Python code to consume those rules:

- `app/config.py`
  - added `load_retrieval_config()`
  - centralized config paths under `CONFIG_DIR`

- `app/chunker.py`
  - removed hardcoded era regex constants
  - compiles chunk-era patterns from `retrieval.yaml`
  - still stores `era` metadata on each chunk

- `app/retriever.py`
  - removed hardcoded concept aliases
  - removed hardcoded era query regexes
  - removed hardcoded diverse/comparison marker lists
  - compiles these values from `retrieval.yaml`
  - includes `retrieval_config_hash` in index metadata
  - bumps `index_version` so stale Chroma indexes rebuild once

- `app/embedder.py` and `app/retriever.py`
  - set `USE_TF=0` before importing `sentence_transformers`
  - keeps the project on the intended PyTorch path
  - avoids accidental TensorFlow/Keras import failures in mixed environments

## Result

The retrieval behavior remains the same in spirit: dense retrieval, lexical
fallback, cross-encoder reranking, and source selection still work as before.

The improvement is that documentary-specific tuning is now visible and
editable in YAML. For another transcript, we can adjust eras, aliases, and
query-intent hints without rewriting retrieval code.

Changing `config/retrieval.yaml` also changes the stored
`retrieval_config_hash`, so the persisted Chroma index knows when it should be
rebuilt.

## Verification

Checks run with the project virtual environment:

```powershell
.\archivox\Scripts\python.exe -m py_compile app\config.py app\chunker.py app\embedder.py app\retriever.py
```

Additional smoke checks confirmed:

- `config/retrieval.yaml` loads correctly.
- `app.retriever` imports correctly.
- transcript chunking still works.
- chunk metadata still includes the expected eras:
  `edwardian`, `general`, `postwar`, `tudor`, `victorian`.
