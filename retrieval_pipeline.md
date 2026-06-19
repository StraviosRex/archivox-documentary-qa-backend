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
  +-- Metadata pre-filtering
  |     - query patterns detect configured metadata labels
  |     - Chroma where clause prunes irrelevant chunks
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

### Phase 1: Config-Driven Retrieval Hints

We introduced `config/retrieval.yaml` as the home for corpus-specific retrieval
rules:

- `metadata.fields.era`: chunk-time and query-time era patterns.
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
  - stores configured metadata on each chunk

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

### Phase 2A: Generic Metadata Tagging

The first config extraction still left the Python code aware of one special
metadata field: `era`. Phase 2A generalizes that into metadata fields:

```yaml
metadata:
  fields:
    era:
      default: general
      labels:
        victorian:
          chunk_patterns:
            - '\bvictorian\b'
          query_patterns:
            - '\bvictorian\b'
```

The chunker now applies all configured metadata fields and stores them in a
generic `metadata` dictionary on each chunk. The retriever flattens those
simple key/value pairs into Chroma metadata, so Chroma can apply native
pre-retrieval filters.

The retriever also detects query filters generically:

- `metadata.fields.<field>.labels.<label>.query_patterns` identify labels
  mentioned by the user.
- matching labels become Chroma `where` filters.
- configured defaults, such as `era: general`, are included when a field is
  filtered so general context is not accidentally excluded.

At the moment only `era` is defined, preserving current behavior. New fields
such as `speaker`, `topic`, `hazard_type`, or `chapter` can now be added in
YAML without changing the chunker or retriever.

## Result

The retrieval behavior remains the same in spirit: dense retrieval, lexical
fallback, cross-encoder reranking, and source selection still work as before.

The improvement is that documentary-specific tuning is now visible and
editable in YAML. For another transcript, we can adjust metadata fields,
aliases, and query-intent hints without rewriting retrieval code.

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

Recent validation after Phase 2A:

- `tests/test_ask.py`: 9/9 passed.
- `tests/audit_sources.py`: 9/9 passed after the excerpt matching was loosened
  enough to handle transcript timestamp boundaries.

## Source Audit Utility

We added `tests/audit_sources.py` as a lightweight grounding audit for API
responses. It reuses the evaluation questions from `tests/test_ask.py`, calls
`/ask`, and checks returned source citations against `data/transcript.txt`.

The audit verifies:

- each source timestamp has the expected `HH:MM:SS-HH:MM:SS` shape.
- each source start and end timestamp exists in the transcript.
- each displayed excerpt is traceable to the original transcript text.
- shortened excerpts with leading or trailing `...` are handled as fragments.
- punctuation and spacing differences are tolerated with a loose fallback.

The audit intentionally does not decide whether a source is semantically the
best source for the answer. That remains a retrieval/evaluation judgement. The
script only answers the narrower question: "Did this citation come from the
transcript it claims to cite?"

Run it after starting the server:

```powershell
.\archivox\Scripts\python.exe -m tests.audit_sources
```

For local models or cases where rate limits are not a concern:

```powershell
.\archivox\Scripts\python.exe -m tests.audit_sources --delay 0
```

This gives a second confidence check alongside `tests/test_ask.py`: the
answers still need to pass the behavioral assertions, and the displayed
citations must be traceable to the source transcript.
