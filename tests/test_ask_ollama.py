"""
Sanity check tests for the /ask endpoint using the local Ollama profile.

Covers the same 5 question types as test_ask.py but targets the
ollama profile. No rate-limit delay
is needed since Ollama runs locally.

Usage:
    Start Ollama and the server first:
        ollama serve
        HF_HUB_OFFLINE=1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

    Then run:
        python -m tests.test_ask_ollama
"""

import httpx
import asyncio
import sys
import time

BASE_URL = "http://localhost:8000"
PROFILE = "ollama::llama3.2:3b"

# Ollama runs locally so there is no token-rate budget to worry about.
# A small pause is kept to avoid hammering the embedding model between requests.
DELAY_BETWEEN_REQUESTS_SECONDS = 2

# Ollama on CPU is slower than a cloud API. 16s was observed in testing;
# 120s gives comfortable headroom on slower machines.
REQUEST_TIMEOUT_SECONDS = 120.0

QUESTIONS = [
    {
        "type": "1. Factual",
        "question": "How much borax could potentially kill a small child?",
        "checks": {
            "sources_non_empty": True,
            "answer_includes": ["five", "gram"],
        },
    },
    {
        "type": "2. Synthesis",
        "question": "How did food adulteration and the lack of food safety legislation together contribute to public health problems in Victorian England?",
        "checks": {
            "sources_non_empty": True,
        },
    },
    {
        "type": "3. Named person/location",
        "question": "Who was Thomas Crapper and what did he invent?",
        "checks": {
            "sources_non_empty": True,
            "answer_includes": ["crapper", "siphon"],
        },
    },
    {
        "type": "4. Vague",
        "question": "What were the dangers in Victorian homes?",
        "checks": {
            "sources_non_empty": True,
        },
    },
    {
        "type": "5. Out-of-scope",
        "question": "What role did the internet play in Victorian household safety?",
        "checks": {
            "sources_empty": True,
            "answer_includes": ["don't have enough information"],
        },
    },
]


def _run_checks(
    data: dict,
    elapsed: float,
    checks: dict,
) -> list[str]:
    """Return a list of failure messages, empty if all checks pass."""
    failures: list[str] = []

    normalized_answer = (
        data.get("answer", "")
        .lower()
        .replace("'", "'")
        .replace("`", "'")
    )
    sources = data.get("sources", [])

    if checks.get("sources_non_empty") and not sources:
        failures.append("expected sources but got none")

    if checks.get("sources_empty") and sources:
        failures.append(f"expected no sources but got {len(sources)}")

    for term in checks.get("answer_includes", []):
        if term.lower() not in normalized_answer:
            failures.append(f"answer missing expected term: '{term}'")

    if elapsed > 30:
        failures.append(f"response time {elapsed:.2f}s exceeds 30s limit")

    return failures


async def run_tests():
    timings = []
    results: list[tuple[str, bool]] = []

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        # Health check
        try:
            health = await client.get(f"{BASE_URL}/health")
            health.raise_for_status()
            print(f"Server is up. Using profile: {PROFILE}\n")
        except Exception as e:
            print(f"Server not reachable at {BASE_URL}: {e}")
            print("Start the server first: uvicorn app.main:app --port 8000")
            return

        # Run each question
        for i, q in enumerate(QUESTIONS):
            if i > 0:
                print(f"(waiting {DELAY_BETWEEN_REQUESTS_SECONDS}s...)\n")
                await asyncio.sleep(DELAY_BETWEEN_REQUESTS_SECONDS)

            print(f"{'='*60}")
            print(f"[{q['type']}]")
            print(f"Q: {q['question']}")
            print(f"{'-'*60}")

            elapsed = 0.0
            start = time.perf_counter()
            try:
                response = await client.post(
                    f"{BASE_URL}/ask",
                    json={"question": q["question"], "profile": PROFILE},
                )

                response.raise_for_status()
                elapsed = time.perf_counter() - start
                data = response.json()

                print(f"A: {data['answer'][:300]}")
                if len(data["answer"]) > 300:
                    print("   ...")
                print(f"\nSources ({len(data['sources'])}):")
                for j, src in enumerate(data["sources"], 1):
                    print(f"  {j}. [{src['timestamp']}] {src['excerpt'][:100]}...")

                print(f"\nResponse time: {elapsed:.2f}s")

                failures = _run_checks(
                    data=data,
                    elapsed=elapsed,
                    checks=q.get("checks", {}),
                )

                if failures:
                    for msg in failures:
                        print(f"  FAIL: {msg}")
                    results.append((q["type"], False))
                else:
                    print("  PASS")
                    results.append((q["type"], True))

            except Exception as e:
                elapsed = time.perf_counter() - start
                print(f"ERROR: {e}")
                print(f"Time before failure: {elapsed:.2f}s")
                results.append((q["type"], False))

            finally:
                timings.append((q["type"], elapsed))

            print()

    if timings:
        print(f"{'='*60}")
        print("Response time summary")
        print(f"{'-'*60}")
        for label, seconds in timings:
            flag = "  (over 30s)" if seconds > 30 else ""
            print(f"  {label:<28} {seconds:.2f}s{flag}")
        avg = sum(t for _, t in timings) / len(timings)
        print(f"\n  Average: {avg:.2f}s")
        print()

    if results:
        print(f"{'='*60}")
        print("Assertion summary")
        print(f"{'-'*60}")
        for label, passed in results:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {label}")
        total = len(results)
        passed_count = sum(1 for _, p in results if p)
        print(f"\n  {passed_count}/{total} passed")
        print()

        if passed_count < total:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_tests())
