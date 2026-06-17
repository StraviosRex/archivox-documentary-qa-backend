# Archivox

A documentary Q&A backend. Ask natural-language questions and receive answers grounded in timestamped transcript excerpts, with source citations pointing to the exact moments in the recording.

## How it works

At startup, Archivox parses a plain-text transcript, splits it into overlapping chunks, embeds them with a local sentence-transformer model, and stores the index in ChromaDB. When a question arrives, the backend runs hybrid retrieval (dense vector search + lexical fallback), passes the top chunks to a configurable LLM, and returns a structured answer with timestamp-linked source references.

The index is automatically rebuilt if the transcript file, embedding model, or chunking settings change. Otherwise the persisted index is reused, so subsequent starts are fast.

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, runs locally) |
| Vector store | ChromaDB (local persistence) |
| LLM | Configurable — OpenRouter, Groq, Gemini, Ollama |
| Config | Pydantic Settings + `.env` + `config/models.yaml` |

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

The embedding model (`all-MiniLM-L6-v2`) is downloaded automatically on first run and then cached locally. The Dockerfile pre-downloads it at build time.

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set the API key for the provider you want to use:

```env
LLM_PROFILE=openrouter_llama_free
OPENROUTER_API_KEY=your_key_here
```

See [LLM profiles](#llm-profiles) below for all available options.

### 3. Add your transcript

Place your transcript at `data/transcript.txt`. The expected format is alternating timestamp and text lines:

```
00:00:00
Spoken text for the first segment goes here.
00:01:23
The next segment of spoken text continues here.
```

### 4. Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The transcript is indexed on startup. Open `http://localhost:8000` for the web UI, or `http://localhost:8000/docs` for the interactive API docs.

## Docker

```bash
docker compose up --build
```

The Chroma index is stored in a named volume (`chroma_data`) so it persists across container restarts.

## API

### `POST /ask`

Ask a question about the transcript.

**Request**
```json
{
  "question": "What was borax used for in Victorian milk?",
  "profile": "groq_llama70b"
}
```

`profile` is optional. When omitted, the `LLM_PROFILE` from `.env` is used.

**Response**
```json
{
  "answer": "Between 00:10:33 and 00:13:21, borax was used to neutralize acid in sour milk, making spoiled milk taste fresh again. It did not remove the bacteria, so it could mask dangerous contamination.",
  "sources": [
    {
      "timestamp": "00:10:33-00:13:21",
      "excerpt": "Boracic acid was a component of a product called borax... used during the Victorian period to prolong the life of milk."
    },
    {
      "timestamp": "00:13:21-00:16:14",
      "excerpt": "The real problem is it doesn't get rid of the bacteria, the underlying cause of the acid."
    }
  ],
  "profile": "openrouter_llama_free",
  "provider": "openrouter",
  "model": "meta-llama/llama-3.3-70b-instruct:free",
  "model_used": "meta-llama/llama-3.3-70b-instruct"
}
```

For out-of-scope questions, `sources` is empty and the answer says the transcript does not contain enough information.

### `GET /profiles`

Returns the list of available LLM profiles and the current default.

### `GET /health`

Returns `{"status": "ok"}`.

## LLM profiles

Profiles are defined in [config/models.yaml](config/models.yaml). The active profile is selected via `LLM_PROFILE` in `.env`, or per-request via the `profile` field in `/ask`.

| Profile ID | Provider | Model |
|---|---|---|
| `openrouter_free_router` | OpenRouter | Auto-routed free model |
| `openrouter_llama_free` | OpenRouter | Llama 3.3 70B (free) |
| `openrouter_deepseek_free` | OpenRouter | DeepSeek R1 (free) |
| `openrouter_qwen_free` | OpenRouter | Qwen Coder (free) |
| `groq_llama8b` | Groq | Llama 3.1 8B |
| `groq_llama70b` | Groq | Llama 3.3 70B |
| `gemini_flash` | Google | Gemini 2.5 Flash |
| `ollama_llama32_3b` | Ollama (local) | Llama 3.2 3B |
| `ollama_qwen25_3b` | Ollama (local) | Qwen 2.5 3B |

OpenRouter profiles with fallback models automatically retry on other free models if the primary is unavailable. Ollama profiles require no API key.

To add a new profile, add an entry to `config/models.yaml` — no code changes needed.

## Configuration reference

All settings can be set in `.env`. See [.env.example](.env.example) for the full list.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROFILE` | `openrouter_llama_free` | Active LLM profile from `models.yaml` |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `GROQ_API_KEY` | — | Groq API key |
| `GEMINI_API_KEY` | — | Gemini API key |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama endpoint |
| `TOP_K` | `5` | Chunks retrieved per query |
| `SOURCES_IN_RESPONSE` | `3` | Sources returned in the response |
| `SIMILARITY_THRESHOLD` | `0.55` | Max cosine distance for a chunk to be included |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model name |
| `CHUNK_WINDOW_SIZE` | `3` | Transcript segments per chunk |
| `CHUNK_OVERLAP` | `1` | Overlapping segments between consecutive chunks |
| `TRANSCRIPT_PATH` | `data/transcript.txt` | Path to transcript file |
| `CHROMA_PERSIST_DIR` | `chroma_db` | Directory for ChromaDB persistence |

## Running the demo script

`tests/test_ask.py` is a manual smoke-test script, not an automated assertion-based suite. It sends the 5 question types from the evaluation criteria (factual, synthesis, named-entity, vague, out-of-scope) to a running server and prints each answer and its sources so the output can be inspected by eye. It does not pass or fail on its own; it is meant as a quick way to exercise all five categories at once during development or review.

```bash
# Terminal 1 — start the server
uvicorn app.main:app --port 8000

# Terminal 2 — run the demo script
python -m tests.test_ask
```
