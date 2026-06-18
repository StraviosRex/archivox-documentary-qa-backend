# Design Document: Documentary Q&A Backend

## Overview

Archivox answers natural-language questions about a documentary transcript, grounded in the source material. At startup the transcript is chunked, embedded, and indexed. Each question is embedded and matched against that index using hybrid retrieval; low-confidence chunks are dropped before the LLM is called. The LLM generates the natural-language answer; the backend constructs citations directly from the retrieved chunks, making timestamps deterministic rather than LLM-generated.

## 1. Chunking Strategy

The transcript is structured as alternating timestamp lines and spoken-text blocks in `HH:MM:SS` format. The application parses it into `(timestamp, text)` pairs and groups every 2 consecutive segments into an overlapping chunk (1-segment overlap), producing approximately 259 chunks of 200–350 words each.

Timestamp-based grouping was chosen over character or sentence splitting because the documentary's natural segment boundaries already correspond to topic changes. Character splitting risks cutting mid-sentence or separating a claim from its supporting context, degrading both retrieval precision and source citation accuracy. The 1-segment overlap ensures that a topic spanning two chunks is fully capturable by either.

Each chunk stores `start_timestamp`, `end_timestamp`, `segment_start_index`, `segment_end_index`, and the full text. API source references are built directly from this metadata, so returned timestamps always correspond to actual transcript locations.

## 2. Retrieval

**Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers, local, 384 dimensions). Sentence transformers fine-tune a transformer encoder so that semantically similar sentences are mapped to nearby points in the embedding space. Similarity is measured as cosine distance - the angular distance between two vectors - which is invariant to vector magnitude and reliable for comparing unit-normalized text embeddings. The Dockerfile installs CPU-only PyTorch explicitly before the rest of the dependencies; without this, sentence-transformers pulls the default CUDA-enabled build which adds ~900MB of unused GPU libraries on a CPU-only deployment. The embedding model is also pre-downloaded at image build time so it is baked into the layer cache and the first request has no cold-start delay.

**Vector store:** ChromaDB with local persistence. Under the hood it uses an HNSW (Hierarchical Navigable Small World) index, which builds a graph of approximate nearest neighbors that allows sub-linear lookup without scanning every stored vector. For a corpus of ~259 chunks this is effectively instant, but the design holds at larger scales. The index is rebuilt when the transcript file, embedding model, or chunking configuration changes.

**Why hybrid retrieval.** Dense vector search alone has a structural weakness with proper names: a name like "Thomas Crapper" carries less discriminative signal in embedding space than descriptive vocabulary, because the model has been exposed to that name across many unrelated contexts. A lexical fallback runs in parallel, scanning chunks for exact proper-name phrases or full query-substring matches. A chunk found by both dense and lexical search is a stronger candidate than one found by either alone, so the confidence filter admits a chunk if either signal independently clears its bar, rather than letting a borderline dense distance override an already-strong lexical match.

**Confidence filtering.** Chunks exceeding a cosine distance of 0.48 and lacking a qualifying lexical match are dropped before generation. This threshold was calibrated empirically: in-scope questions consistently produced top candidates below 0.48; out-of-scope questions with no shared vocabulary produced distances above 0.65. Questions that share general documentary vocabulary (e.g. asking about the internet in the context of "Victorian households") can produce candidates in the 0.44–0.48 range, because dense embeddings measure topical proximity, not answerability. The LLM acts as a second filter in those cases, correctly refusing when the retrieved excerpts don't actually address the question.

**Cross-encoder re-ranking.** After candidate generation, each `(query, chunk)` pair is scored by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Unlike a bi-encoder, a cross-encoder processes the query and chunk together as a single sequence, allowing full attention across both - this produces a finer-grained relevance signal than cosine distance. When re-ranking is active, the CE score is the sole eligibility gate: a chunk passes if its CE score exceeds `-4.5`, regardless of its dense distance or lexical score. The threshold was calibrated from logged score distributions across all evaluation question types; the noise floor (common-word literal matches) consistently falls below `-8.0`, while genuinely relevant topic-specific chunks score above `-4.5` even when dense distance alone would not surface them. The model is pre-downloaded at image build time, so there is no cold-start delay on the first request. Re-ranking can be disabled via `RERANKING_ENABLED=false` for a fast fallback path (0.80s avg) at the cost of multi-topic retrieval coverage.

**Named-entity prioritization.** After filtering, chunks containing an exact proper-name phrase match are promoted to the front of the context window. Only their immediate neighbors (±1 index position) are kept as supporting context; remaining candidates are discarded. This prevents topically adjacent but less specific chunks from burying the directly relevant one.

**Source ranking.** Returned sources are ordered by relevance to the question, with cross-encoder score as the primary signal. For multi-topic questions the ranking reflects best evidence coverage rather than raw scores; a lower-scoring chunk covering a missing topic is preferred over a higher-scoring one that duplicates already-covered ground. Numerical scores are kept internal; the UI presents sources as an ordered list under the heading "ranked by relevance".

## 3. Prompt Construction

The LLM receives only the retrieved chunks, formatted as labeled excerpts with timestamp ranges, not the full transcript. The system prompt instructs it to answer in 2–4 sentences from the provided material only, to use a fixed refusal phrase when the excerpts are insufficient, and to return plain prose without timestamps or Markdown.

```text
System:
You are a Q&A assistant for a documentary transcript. Answer using ONLY
the provided transcript excerpts. If the excerpts do not contain enough
information, respond only with: "I don't have enough information in the
transcript to answer that question."

Rules: do not invent facts; keep answers to 2–4 sentences; no Markdown;
never reproduce the [Excerpt N | ...] labels; do not include timestamps
or time codes in the answer text; begin directly with the answer in plain
prose.

User:
Based on the following transcript excerpts, answer the question.

Transcript excerpts:
[Excerpt 1 | 00:10:33 - 00:13:21]
Boracic acid was a component of a product called borax, used during the
Victorian period to neutralise the acid in sour milk...

[Excerpt 2 | 00:13:21 - 00:16:14]
The real problem is it doesn't get rid of the bacteria, the underlying
cause of the acid...

Question:
What was borax used for in Victorian milk?
```

Timestamps are kept out of the answer text deliberately. The `sources` array already carries precise timestamp ranges for every cited chunk, so including them in the prose would be redundant and make the answer read less naturally. The `sources` array is constructed by the backend directly from the retrieved and ranked chunks - the LLM is never asked to produce or reference chunk identifiers. This is deliberate: an LLM instructed to cite sources can confabulate timestamps that were never retrieved. Generating the answer text and generating traceable citations are fundamentally different reliability problems, and separating them removes the second problem entirely.

## 4. LLM Provider Configuration

LLM backends are configured as named profiles in `config/models.yaml`, each specifying provider, model, base URL, and which environment variable holds the API key. All supported providers (Groq, Gemini, OpenRouter, Ollama) expose OpenAI-compatible chat completion endpoints, so adding a new provider requires only a YAML entry - no code changes. The active profile is set via `LLM_PROFILE` in `.env` or overridden per-request via the `profile` field in `/ask`.

`groq_llama8b` is the default; it was the most extensively exercised profile during retrieval and threshold tuning. Ollama was tested with llama3.2:3b across all five evaluation categories and produced correct, well-grounded answers in every case, including accurate refusal on the out-of-scope question. 

It is not the default since it requires a local Ollama installation the reviewer may not have, but it is a genuinely viable fully-offline option, not just an architectural placeholder. When running inside Docker, `OLLAMA_BASE_URL` must be set to `http://host.docker.internal:11434/v1` (the `.env.example` has this as the default); `localhost` from inside the container resolves to the container itself, not the host where Ollama is running. If the repo is cloned to the machine directly, OLLAMA_BASE_URL should be set to "http://localhost:11434/v1".

## 5. API Response Construction

```json
{
  "answer": "According to some people, five grams of borax is sufficient to potentially kill a small child.",
  "sources": [
    { "timestamp": "00:12:25-00:14:18", "excerpt": "..." },
    { "timestamp": "00:13:21-00:15:16", "excerpt": "..." }
  ],
  "profile": "groq_llama8b",
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "model_used": "llama-3.1-8b-instant"
}
```

The number of sources varies with retrieval confidence rather than being capped at a fixed count. Padding the response with weak sources to hit a number contradicts the goal of grounded citations.

Before returning, one deterministic post-processing pass runs on the LLM output: it checks for the instructed refusal phrase and returns empty sources if detected. This keeps refusal behaviour consistent without relying on the LLM to suppress sources itself.

Response times were measured directly against the 30-second requirement across all six evaluation question types (including the hard multi-topic case). With cross-encoder re-ranking enabled, the default Groq profile (`llama-3.1-8b-instant`) averaged 3.51s end-to-end across six questions, with a slowest single response of 4.23s. The fully local Ollama fallback (`llama3.2:3b`) completed in approximately 9 seconds; retrieval accounts for ~3s, with local CPU inference adding ~6s for generation. All configurations are comfortably within the 30-second limit.

## 6. What I Would Improve Given More Time

- **BM25 lexical retrieval:** The current lexical path uses a hand-crafted token-weighted scoring function with manual boosts for proper names and numerics. Replacing it with BM25 - a probabilistic retrieval model that applies saturating term frequency and document-length normalization - would make lexical scores more principled and comparable across chunks of different lengths, reducing the need for manual calibration and threshold tuning.
- **Stratified pre-CE pool with Reciprocal Rank Fusion:** The current pipeline sends all merged candidates (up to ~46–60 chunks) to CE inference, which accounts for most of the per-request latency (~2.5–3.3s). A stratified selector would take the top-N dense candidates, top-N lexical candidates, and the best candidate per explicit query term, then fill to a cap of 25–30 using Reciprocal Rank Fusion (RRF). RRF combines ranked lists by summing reciprocal rank scores rather than raw signals, so a chunk present on only one axis still receives credit. This would reduce CE inference time without sacrificing multi-topic coverage.
- **Dynamic chunking:** Use topic-shift detection to create variable-length chunks based on content boundaries rather than fixed segment windows. Documentary transcripts often have natural topic transitions that don't align with fixed-size windows, causing some topics to be split across chunk boundaries.
- **Streaming:** Add optional token streaming for the answer while keeping the same source-return structure.
