# Design Document: Documentary Q&A Backend

## 1. Chunking Strategy

The transcript is provided as plain text with alternating timestamp lines and spoken text blocks. Each timestamp uses the `HH:MM:SS` format. The supplied transcript contains approximately 260 timestamped segments and spans about 3 hours and 53 minutes.

The application parses the transcript into `(timestamp, text)` pairs, then groups every 3 consecutive transcript segments into one chunk with a 1-segment overlap. With this configuration, the transcript produces approximately 130 overlapping chunks. Each chunk is usually around 350 to 450 words, which is large enough to preserve local context but still small enough for precise retrieval.

This strategy was chosen because the transcript timestamps already provide natural boundaries. Fixed character splitting could cut through a sentence or separate a claim from its explanation, which would make both retrieval and source citation weaker. Grouping 3 segments gives the retriever enough context to capture a complete point, while the 1-segment overlap protects topics that continue across chunk boundaries.

Each stored chunk contains:

- `chunk_id`
- `start_timestamp`
- `end_timestamp`
- `segment_start_index`
- `segment_end_index`
- `text`

The API source references are generated from this metadata, so the returned timestamps always correspond to actual transcript locations.

## 2. Retrieval

The system uses a retrieval-based approach rather than sending the full transcript to the LLM. At startup, the service parses the transcript, chunks it, embeds the chunks, and stores them in a local vector index.

**Embedding model:** `all-MiniLM-L6-v2` from `sentence-transformers`. It runs locally, requires no API key, produces 384-dimensional embeddings, and is fast enough for a corpus of this size. The model may be downloaded during setup or first startup, then cached locally.

**Vector store:** ChromaDB with local persistence. The persisted index is reused on later starts, but it is rebuilt when the transcript hash, embedding model, or chunking configuration changes. This avoids accidentally serving stale embeddings after the transcript or retrieval settings are updated.

For each question, the service embeds the user query with the same embedding model and retrieves the top 5 most relevant chunks using vector similarity search. The top 5 chunks are passed to the LLM as context. The top 2 to 3 ranked chunks are returned as structured source references in the API response.

A lightweight lexical fallback can also be used for exact names, places, or unusual phrases. Dense embeddings are good for semantic questions, but exact keyword matching can improve recall for named-person or named-location questions. Retrieved dense and lexical candidates are deduplicated and ranked before prompt construction.

If retrieval confidence is too low, the service returns a conservative response saying that the transcript does not contain enough information to answer the question. This prevents the LLM from guessing when the available evidence is weak.


Retrieval confidence is determined empirically rather than by an arbitrary cutoff. Cosine distances were measured across a set of known in-scope and out-of-scope test questions. For questions well covered by the transcript, retrieved chunk distances consistently stayed below approximately 0.48 across the top 10 dense candidates. For a clearly out-of-scope question, distances started above 0.65, with no overlap between the two ranges. A similarity threshold of 0.55 was chosen to sit within this gap, rejecting chunks that fall outside the demonstrated range of genuinely relevant content while leaving margin on both sides for natural variation across different question phrasings.


## 3. Prompt Construction

The LLM receives only the retrieved transcript excerpts, not the full transcript. The prompt is built from a system instruction, the ranked context chunks, and the user question.

```text
System: You are a Q&A assistant. Answer the user's question using only the
provided transcript excerpts. If the excerpts do not contain enough information,
say: "I don't have enough information in the transcript to answer that question."

Do not invent facts, names, dates, causes, or conclusions. Keep the answer
focused on what the transcript says.

Context:
[Source 1 | 00:10:33-00:13:21]
<chunk text>

[Source 2 | 00:13:21-00:16:14]
<chunk text>

...

User question:
<question>
```

The LLM is responsible only for generating the natural-language answer. It is not trusted to generate source metadata. The backend constructs the `sources` array directly from the ranked retrieved chunks. This makes timestamps and excerpts deterministic, easier to test, and less vulnerable to hallucinated citations.

Each source returned by the API contains a timestamp range and a short excerpt from the retrieved chunk. Excerpts are trimmed to the most relevant portion where possible, rather than returning an entire long chunk.

## 4. LLM Provider Configuration

The LLM backend is configurable through environment variables, so the provider can be changed without code changes.

| Variable | Example | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `groq`, `ollama`, `gemini` | Selects the provider |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model name |
| `LLM_API_KEY` | `gsk_...` | API key, not needed for Ollama |
| `LLM_BASE_URL` | `http://localhost:11434` | Optional base URL override |

Groq and Ollama are accessed through OpenAI-compatible chat completion interfaces. Gemini can be configured through its OpenAI-compatible endpoint or handled through a provider-specific adapter if needed.

The default provider is Groq free tier using `llama-3.3-70b-versatile`, because it offers fast inference and strong instruction following without requiring a paid subscription. A fully local Ollama setup is supported as an optional configuration for offline use.

## 5. API Response Construction

The API exposes a single endpoint:

```http
POST /ask
```

Request body:

```json
{
  "question": "What was borax used for in Victorian milk?"
}
```

Response body:

```json
{
  "answer": "Borax was used to neutralize the acid in sour milk, making spoiled milk taste fresh again. However, it did not remove the bacteria that caused the milk to spoil, so it could hide dangerous contamination.",
  "sources": [
    {
      "timestamp": "00:10:33-00:13:21",
      "excerpt": "Boracic acid was a component of a product called borax... used during the Victorian period to prolong the life of milk."
    },
    {
      "timestamp": "00:13:21-00:16:14",
      "excerpt": "The real problem is it doesn't get rid of the bacteria, the underlying cause of the acid."
    }
  ]
}
```

The source list is ranked according to retrieval relevance. For out-of-scope questions, the response uses a clear refusal rather than attempting to infer an answer from unrelated transcript sections.

## 6. What I Would Improve Given More Time

- **Hybrid retrieval:** Add a proper BM25 index and combine it with vector search for better handling of exact names, locations, and rare terms.
- **Re-ranking:** Add a cross-encoder re-ranker, such as `cross-encoder/ms-marco-MiniLM-L-6-v2`, after initial retrieval to improve source ordering.
- **Dynamic chunking:** Use topic-shift detection to create variable-length chunks based on content rather than fixed segment windows.
- **Evaluation harness:** Build a small test set covering factual, synthesis, named-entity, vague, and out-of-scope questions to measure retrieval and answer quality.
- **Streaming:** Add optional token streaming for the answer while keeping the same source-return structure.
- **Minimal UI:** Add a simple web page with a question field and answer/source display for easier manual testing.
