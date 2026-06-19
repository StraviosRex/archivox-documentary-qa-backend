# Retrieval Pipeline

```
Question
    │
    ▼
┌─────────────┐
│   Embedder  │  (all-MiniLM-L6-v2)
└──────┬──────┘
       │ query vector
       ▼
┌──────────────────────────────────────┐
│           Hybrid Retrieval           │
│                                      │
│  ┌─────────────┐  ┌───────────────┐  │
│  │ Dense search│  │ Lexical search│  │
│  │  (ChromaDB/ │  │ (token match, │  │
│  │    HNSW)    │  │  name boost)  │  │
│  └──────┬──────┘  └──────┬────────┘  │
│         └────────┬────────┘           │
│               merge                  │
└───────────────┬──────────────────────┘
                │ ~50 candidates
                ▼
       ┌─────────────────┐
       │ Confidence gate │  cosine dist < 0.48
       └────────┬────────┘
                │ filtered candidates
                ▼
       ┌─────────────────┐
       │ Cross-encoder   │  (ms-marco-MiniLM-L-6-v2)
       │   re-ranking    │  scores each (query, chunk) pair
       └────────┬────────┘
                │ sorted by CE score
                ▼
       ┌──────────────────────────────────┐
       │        Final source selection    │
       │                                  │
       │  anchor = highest CE score       │
       │                                  │
       │  local mode:                     │
       │    anchor + transcript neighbors │
       │                                  │
       │  diverse mode (multi-topic):     │
       │    anchor + candidates covering  │
       │    uncovered query topics        │
       └────────┬─────────────────────────┘
                │ up to 3 sources
                │ (1 = primary anchor,
                │  2-3 = supporting evidence)
                ▼
       ┌─────────────────┐
       │ Prompt builder  │  excerpts + timestamps
       └────────┬────────┘
                │
                ▼
              LLM
                │
                ▼
         answer + sources
         (ranked by relevance)
```
