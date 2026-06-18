# Design Document: Documentary Q&A Backend

## Overview

Archivox answers natural-language questions about a documentary transcript, grounded in the source material. At startup the transcript is chunked, embedded, and indexed. Each question is matched against that index using hybrid retrieval; a cross-encoder re-ranker then scores and filters candidates before the LLM is called. The backend constructs source citations directly from the retrieved chunks, making timestamps deterministic rather than LLM-generated.

## 1. Chunking Strategy

The transcript is structured as alternating timestamp lines and spoken-text blocks in `HH:MM:SS` format. The application parses it into `(timestamp, text)` pairs and groups every 2 consecutive segments into an overlapping chunk (1-segment overlap), producing approximately 259 chunks of 200–350 words each.

Timestamp-based grouping was chosen over character or sentence splitting because the transcript already provides meaningful speech boundaries and traceable timecodes. Character splitting risks cutting mid-sentence or separating a claim from its supporting context, degrading both retrieval precision and source citation accuracy. The 1-segment overlap ensures topics that span two chunks are fully capturable by either.

Each chunk stores `start_timestamp`, `end_timestamp`, `segment_start_index`, `segment_end_index`, and the full text. API source references are built directly from this metadata, so returned timestamps always correspond to actual transcript locations.

## 2. Retrieval

**Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers, local, 384 dimensions). Similarity is measured as cosine distance, which is invariant to vector magnitude and reliable for comparing unit-normalized text embeddings.

**Vector store:** ChromaDB with local persistence, using an HNSW index for sub-linear approximate nearest-neighbor lookup. The index is rebuilt automatically when the transcript, embedding model, or chunking configuration changes.

**Hybrid retrieval.** Dense vector search has a structural weakness with proper names: a name like "Thomas Crapper" carries less discriminative signal in embedding space than descriptive vocabulary. A lexical path runs in parallel, scoring chunks using query-term coverage, exact proper-name phrases, capitalized entities, numerical matches, concept aliases, and literal query terms. Dense and lexical candidates are merged and deduplicated before re-ranking.

**Cross-encoder re-ranking.** After candidate generation, each `(query, chunk)` pair is scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`. Unlike a bi-encoder, a cross-encoder processes the query and chunk together as a single sequence, allowing full attention across both and producing a finer-grained relevance signal than cosine distance. When re-ranking is active, the CE score is the sole eligibility gate: a chunk passes if its score exceeds `−4.5`. This threshold was calibrated empirically from logged score distributions across the evaluation questions; genuinely relevant chunks consistently scored above `−4.5` while the noise floor for common-word literal matches fell below `−8.0`. When re-ranking is disabled, a cosine distance confidence filter applies instead: in-scope questions consistently produced top candidates below `0.48`, while out-of-scope questions with no shared vocabulary produced distances above `0.65`. Re-ranking can be disabled via `RERANKING_ENABLED=false` for a faster fallback path.

**Context-aware source selection.** The highest CE-scoring chunk serves as the primary evidence anchor. Supporting sources are selected in one of two modes: local mode adds immediate transcript neighbours of the anchor for context continuity; diverse mode prioritises candidates that introduce new query-term coverage or come from distinct transcript regions. Neighbour chunks may be included without independently passing the CE threshold.

The service returns up to three sources. The first is the primary relevance anchor, while later sources provide relevant context or additional topic coverage. Sources 2 and 3 therefore reflect relevance-aware supporting evidence rather than the second- and third-highest raw CE scores.

## 3. Prompt Construction

The LLM receives only the retrieved chunks as labelled excerpts with timestamp ranges. The system prompt instructs it to answer in 2–4 sentences from the provided material only, use a fixed refusal phrase when excerpts are insufficient, and return plain prose without timestamps or Markdown.

```text
System: answer only from provided excerpts; refuse with a fixed phrase if
excerpts are insufficient; 2-4 sentences; plain prose, no timestamps or Markdown.

User:
[Excerpt 1 | 00:10:33 - 00:13:21]
Boracic acid was a component of a product called borax, used during the
Victorian period to neutralise the acid in sour milk...

[Excerpt 2 | 00:13:21 - 00:16:14]
The real problem is it doesn't get rid of the bacteria, the underlying
cause of the acid...

Question: What was borax used for in Victorian milk?
```

The `sources` array is constructed by the backend directly from retrieved chunks. The LLM is never asked to produce or reference chunk identifiers. An LLM instructed to cite sources can confabulate timestamps that were never retrieved; separating answer generation from citation generation removes that failure mode entirely.

## 4. LLM Provider Configuration

LLM backends are configured as named profiles in `config/models.yaml`, each specifying provider, model, base URL, and API key environment variable. All supported providers (Groq, Gemini, OpenRouter, Ollama) expose OpenAI-compatible endpoints, so adding a new provider requires only a YAML entry and no code changes. The active profile is set via `LLM_PROFILE` in `.env` or overridden per request via the `profile` field in `/ask`.

Ollama was tested across all five evaluation categories and produced correct answers, including accurate out-of-scope refusal. It is therefore a viable fully offline option rather than only an architectural placeholder. Response times were measured across all evaluation question types; the default Groq profile averaged 3.47s end-to-end with re-ranking enabled, and the local Ollama fallback completed in approximately 9s. Both are well within the 30-second requirement.

## 5. What I Would Improve Given More Time

- **BM25 lexical retrieval:** Replace the hand-crafted token-weighted scoring with BM25, a probabilistic retrieval model that applies saturating term frequency and document-length normalization, making lexical scores more principled and reducing manual calibration.
- **Stratified pre-CE pool with RRF:** Use Reciprocal Rank Fusion to preserve the strongest dense, lexical, and per-topic candidates while bounding the number of chunks sent to the cross-encoder. This could reduce re-ranking latency without sacrificing multi-topic coverage.
- **Dynamic chunking:** Use topic-shift detection to create variable-length chunks based on content boundaries rather than fixed segment windows.
- **Streaming:** Add optional token streaming for the answer while keeping the same source-return structure.
