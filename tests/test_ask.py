"""
Sanity check tests for the /ask endpoint.

Covers the 5 question types from the evaluation criteria plus regression cases
for multi-topic retrieval and cross-era filtering:
1. Factual question about a specific event
2. Synthesis question spanning two parts of the transcript
3. Question about a named person or location
4. Vague / broadly phrased question
5. Out-of-scope question (answer not in transcript)
6. Hard multi-topic query asserting 3/3 topic coverage from source excerpts
7. Tudor era question (era filter: include tudor only)
8. Post-war era question (era filter: include postwar only)
9. Cross-era question (no era filter; diverse mode)

Usage:
    Start the server first:
        uvicorn app.main:app --port 8000

    Then run:
        python -m tests.test_ask
"""

import httpx
import asyncio
import sys
import time

BASE_URL = "http://localhost:8000"

# Delay between requests, in seconds. Groq's free tier enforces a tokens-
# per-minute budget; five questions back to back can exhaust it by the final
# request, since each sends ~1700 tokens of context plus generates a response.
# 15 seconds gives enough headroom across all five questions. This delay is a
# test-script convenience only; the production /ask endpoint does not pace requests.
DELAY_BETWEEN_REQUESTS_SECONDS = 15

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
    {
        "type": "6. Multi-topic",
        "question": "What were the dangers of Borax, Celluloid, and Asbestos?",
        "checks": {
            "sources_non_empty": True,
            "sources_cover_topics": {
                "borax": ["borax", "brucella", "bacteria"],
                "celluloid": ["celluloid", "parkesine", "flammable", "combust"],
                "asbestos": ["asbestos", "mesothelioma", "fibres", "fibers"],
            },
        },
    },
    {
        "type": "7. Tudor era",
        "question": "Why were Tudor chimneys prone to catching fire?",
        "checks": {
            "sources_non_empty": True,
            "sources_cover_topics": {
                "chimneys": ["chimney", "timber", "wattle", "soot", "fire", "flue"],
            },
        },
    },
    {
        "type": "8. Post-war era",
        "question": "How did carbon monoxide become a hazard in post-war bathrooms?",
        "checks": {
            "sources_non_empty": True,
            "sources_cover_topics": {
                "carbon monoxide": ["carbon monoxide", "co", "toxic", "gas", "oxygen"],
            },
        },
    },
    {
        "type": "9. Cross-era",
        "question": "Compare Victorian and Edwardian approaches to electrical safety.",
        "checks": {
            "sources_non_empty": True,
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
        .replace("’", "’")
        .replace("`", "’")
    )
    sources = data.get("sources", [])
    combined_sources = " ".join(
        src.get("excerpt", "") for src in sources
    ).lower()

    if checks.get("sources_non_empty") and not sources:
        failures.append("expected sources but got none")

    if checks.get("sources_empty") and sources:
        failures.append(f"expected no sources but got {len(sources)}")

    for term in checks.get("answer_includes", []):
        if term.lower() not in normalized_answer:
            failures.append(f"answer missing expected term: ‘{term}’")

    for topic, markers in checks.get("sources_cover_topics", {}).items():
        if not any(marker.lower() in combined_sources for marker in markers):
            failures.append(f"sources do not cover topic ‘{topic}’")

    if elapsed > 30:
        failures.append(f"response time {elapsed:.2f}s exceeds 30s limit")

    return failures


async def run_tests():
    timings = []
    results: list[tuple[str, bool]] = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Health check
        try:
            health = await client.get(f"{BASE_URL}/health")
            health.raise_for_status()
            print("Server is up.\n")
        except Exception as e:
            print(f"Server not reachable at {BASE_URL}: {e}")
            print("Start the server first: uvicorn app.main:app --port 8000")
            return

        # Run each question
        for i, q in enumerate(QUESTIONS):
            if i > 0:
                print(f"(waiting {DELAY_BETWEEN_REQUESTS_SECONDS}s to avoid provider rate limits...)\n")
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
                    json={"question": q["question"]},
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
