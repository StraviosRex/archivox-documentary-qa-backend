# Response Time Benchmarks

Measured directly against the assignment's "response time under 30 seconds for a typical question on standard hardware" requirement.

## Method

Each request was timed end-to-end (full HTTP round trip, including retrieval and LLM generation) using `time curl`, against an already-running server with the transcript already indexed. Same question, same retrieval result, different LLM profile, isolating LLM generation time as the variable.

## Question used

```
What were the dangers in Victorian homes?
```

## Results

| Profile | Provider | Model | Time | Notes |
|---|---|---|---|---|
| `groq_llama8b` (default) | Groq | `llama-3.1-8b-instant` | 3.93s | Hosted, default profile |
| `ollama_llama32_3b` | Ollama (local) | `llama3.2:3b` | 16.10s | Fully offline, slowest tested configuration |

Both retrieval runs returned identical source chunks (`00:00:05-00:02:02`, `00:17:15-00:19:05`, `02:29:06-02:31:03`), confirming retrieval time and result are independent of which LLM profile is used; the time difference reflects generation speed only.

## Second question (synthesis-style)

```
How did food adulteration and the lack of food safety legislation together contribute to public health problems in Victorian England?
```

| Profile | Provider | Model | Time |
|---|---|---|---|
| `groq_llama8b` (default) | Groq | `llama-3.1-8b-instant` | 0.89s |
| `ollama_llama32_3b` | Ollama (local) | `llama3.2:3b` | 8.99s |

Same two source chunks returned for both profiles, confirming retrieval consistency holds for a synthesis-style question as well as a vague one.

## Conclusion

Both the default profile and the slowest tested fallback complete well within the 30-second requirement, with the hosted default responding roughly 4x faster than the fully local option.

---

## After cross-encoder re-ranker (phase 2)

A `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranker was added to fix multi-topic retrieval failure. It scores each `(query, chunk)` pair independently, allowing queries that span multiple transcript sections (e.g. "Borax, Celluloid, and Asbestos") to surface all relevant sections instead of only the dominant one.

The re-ranker can be toggled via `RERANKING_ENABLED=false` in the environment, allowing a clean A/B comparison using the same test runner, server, index, and provider.

### Method

End-to-end response times measured via `python -m tests.test_ask` (5 question types, `groq_llama8b` profile, server already running with index built). Both runs used identical conditions — same server process, same ChromaDB index, same provider.

### Controlled latency comparison

| # | Question type | Baseline (CE off) | CE enabled | Delta |
|---|---|---|---|---|
| 1 | Factual | 3.30s | 10.76s | +7.5s |
| 2 | Synthesis | 0.96s | 7.32s | +6.4s |
| 3 | Named person/location | 0.66s | 3.69s | +3.0s |
| 4 | Vague | 0.74s | 5.99s | +5.3s |
| 5 | Out-of-scope | 0.69s | 5.30s | +4.6s |
| | **Average** | **1.27s** | **6.61s** | **+5.3s** |

All 5 assertions passed in both configurations.

### Retrieval accuracy

| Query | Expected topics | Baseline coverage | CE coverage |
|---|---|---|---|
| "What were the dangers of Borax, Celluloid, and Asbestos?" | 3 | 1/3 (asbestos only) | 3/3 |

### Notes

- The CE adds ~5s overhead on average. Queries with short or uncommon proper names (Thomas Crapper — "thomas" and "crapper" are under 5 chars) incur less overhead because fewer literal-term chunks are injected before scoring.
- Both configurations pass all 5 test assertions and remain well within the 30-second requirement.
- Because retrieval returned identical chunks for standard queries, the observed latency difference is primarily attributable to the CE inference stage rather than the LLM or provider.

## Raw output

```
$ time curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What were the dangers in Victorian homes?"}'

# groq_llama8b
real    0m3.927s
user    0m0.015s
sys     0m0.031s

# ollama_llama32_3b
real    0m16.095s
user    0m0.000s
sys     0m0.016s
```

```
$ time curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "How did food adulteration and the lack of food safety legislation together contribute to public health problems in Victorian England?"}'

# groq_llama8b
real    0m0.892s
user    0m0.000s
sys     0m0.030s

# ollama_llama32_3b
real    0m8.987s
user    0m0.000s
sys     0m0.030s
```