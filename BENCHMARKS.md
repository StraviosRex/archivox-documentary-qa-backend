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