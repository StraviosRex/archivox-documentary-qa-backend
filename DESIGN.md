# Design Document: Documentary Q&A Backend

## Overview

Archivox answers natural-language questions about a documentary transcript, grounded in the source material. At startup the transcript is chunked, embedded, and indexed. Each question is embedded and matched against that index using hybrid retrieval; low-confidence chunks are dropped before the LLM is called. The LLM generates the natural-language answer; the backend constructs citations directly from the retrieved chunks, making timestamps deterministic rather than LLM-generated.

## 1. Chunking Strategy

The transcript is structured as alternating timestamp lines and spoken-text blocks in `HH:MM:SS` format. The application parses it into `(timestamp, text)` pairs and groups every 3 consecutive segments into an overlapping chunk (1-segment overlap), producing approximately 130 chunks of 350–450 words each.

Timestamp-based grouping was chosen over character or sentence splitting because the documentary's natural segment boundaries already correspond to topic changes. Character splitting risks cutting mid-sentence or separating a claim from its supporting context, degrading both retrieval precision and source citation accuracy. The 1-segment overlap ensures that a topic spanning two chunks is fully capturable by either.

Each chunk stores `start_timestamp`, `end_timestamp`, `segment_start_index`, `segment_end_index`, and the full text. API source references are built directly from this metadata, so returned timestamps always correspond to actual transcript locations.

## 2. Retrieval

**Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers, local, 384 dimensions). Sentence transformers fine-tune a transformer encoder so that semantically similar sentences are mapped to nearby points in the embedding space. Similarity is measured as cosine distance — the angular distance between two vectors — which is invariant to vector magnitude and reliable for comparing unit-normalized text embeddings.

**Vector store:** ChromaDB with local persistence. Under the hood it uses an HNSW (Hierarchical Navigable Small World) index, which builds a graph of approximate nearest neighbors that allows sub-linear lookup without scanning every stored vector. For a corpus of ~130 chunks this is effectively instant, but the design holds at larger scales. The index is rebuilt when the transcript file, embedding model, or chunking configuration changes.

**Why hybrid retrieval.** Dense vector search alone has a structural weakness with proper names: a name like "Thomas Crapper" carries less discriminative signal in embedding space than descriptive vocabulary, because the model has been exposed to that name across many unrelated contexts. A lexical fallback runs in parallel, scanning chunks for exact proper-name phrases or full query-substring matches. A chunk found by both dense and lexical search is a stronger candidate than one found by either alone, so the confidence filter admits a chunk if either signal independently clears its bar, rather than letting a borderline dense distance override an already-strong lexical match.

**Confidence filtering.** Chunks exceeding a cosine distance of 0.48 and lacking a qualifying lexical match are dropped before generation. This threshold was calibrated empirically: in-scope questions consistently produced top candidates below 0.48; out-of-scope questions with no shared vocabulary produced distances above 0.65. Questions that share general documentary vocabulary (e.g. asking about the internet in the context of "Victorian households") can produce candidates in the 0.44–0.48 range, because dense embeddings measure topical proximity, not answerability. The LLM acts as a second filter in those cases, correctly refusing when the retrieved excerpts don't actually address the question.

**Named-entity prioritization.** After filtering, chunks containing an exact proper-name phrase match are promoted to the front of the context window. Only their immediate neighbors (±1 index position) are kept as supporting context; remaining candidates are discarded. This prevents topically adjacent but less specific chunks from burying the directly relevant one.

## 3. Prompt Construction

The LLM receives only the retrieved chunks, formatted as labeled excerpts with timestamp ranges, not the full transcript. The system prompt instructs it to answer in 2–4 sentences from the provided material only, to use a fixed refusal phrase when the excerpts are insufficient, and to open non-refusal answers with a natural timestamp phrase.

```text
System:
You are a Q&A assistant for a documentary transcript. Answer using ONLY
the provided transcript excerpts. If the excerpts do not contain enough
information, respond only with: "I don't have enough information in the
transcript to answer that question."

Rules: do not invent facts; keep answers to 2–4 sentences; no Markdown;
never reproduce the [Excerpt N | ...] labels; begin answerable responses
with a natural timestamp phrase (e.g. "Between 00:10:33 and 00:13:21,").

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

The `sources` array is constructed by the backend directly from the retrieved and ranked chunks — the LLM is never asked to produce or reference chunk identifiers. This is deliberate: an LLM instructed to cite sources can confabulate timestamps that were never retrieved. Generating the answer text and generating traceable citations are fundamentally different reliability problems, and separating them removes the second problem entirely.

## 4. LLM Provider Configuration

LLM backends are configured as named profiles in `config/models.yaml`, each specifying provider, model, base URL, and which environment variable holds the API key. All supported providers (Groq, Gemini, OpenRouter, Ollama) expose OpenAI-compatible chat completion endpoints, so adding a new provider requires only a YAML entry — no code changes. The active profile is set via `LLM_PROFILE` in `.env` or overridden per-request via the `profile` field in `/ask`.

`groq_llama8b` is the default; it was the most extensively exercised profile during retrieval and threshold tuning. Ollama was tested with llama3.2:3b across all five evaluation categories and produced correct, well-grounded answers in every case, including accurate refusal on the out-of-scope question. 

It is not the default since it requires a local Ollama installation the reviewer may not have, but it is a genuinely viable fully-offline option, not just an architectural placeholder.

## 5. API Response Construction

```json
{
  "answer": "Between 00:10:33 and 00:13:21, borax was used...",
  "sources": [
    { "timestamp": "00:10:33-00:13:21", "excerpt": "..." }
  ],
  "profile": "groq_llama8b",
  "provider": "groq",
  "model": "llama-3.1-8b-instant",
  "model_used": "llama-3.1-8b-instant"
}
```

The number of sources varies with retrieval confidence rather than being capped at a fixed count. Padding the response with weak sources to hit a number contradicts the goal of grounded citations.

Before returning, two deterministic post-processing passes run on the LLM output: the first checks for the instructed refusal phrase and returns empty sources if detected; the second checks whether the answer opens with a timestamp phrase and prepends one from the top-ranked chunk if not. This keeps the timestamp convention consistent without relying on the LLM to produce a specific output format in every case.

Response times were measured directly against the 30-second requirement. The default Groq profile typically responds in under 5 seconds. The fully local Ollama fallback, the slowest configuration tested, completed in roughly 16 seconds. Both are comfortably within the limit.

## 6. What I Would Improve Given More Time

- **Re-ranking:** Add a cross-encoder re-ranker, such as `cross-encoder/ms-marco-MiniLM-L-6-v2`, after initial retrieval to improve source ordering and provide a relevance signal less sensitive to dominant transcript vocabulary than raw cosine distance.
- **Dynamic chunking:** Use topic-shift detection to create variable-length chunks based on content rather than fixed segment windows.
- **Formal evaluation harness:** The manual smoke-test script (`tests/test_ask.py`) exercises the five evaluation categories but does not assert pass/fail. A proper test suite with explicit assertions on refusal behavior, source accuracy, and answer grounding would catch regressions automatically rather than requiring manual inspection.
- **Streaming:** Add optional token streaming for the answer while keeping the same source-return structure.
