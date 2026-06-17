"""
Sanity check tests for the /ask endpoint.

Covers the 5 question types from the evaluation criteria:
1. Factual question about a specific event
2. Synthesis question spanning two parts of the transcript
3. Question about a named person or location
4. Vague / broadly phrased question
5. Out-of-scope question (answer not in transcript)

Usage:
    Start the server first:
        uvicorn app.main:app --port 8000

    Then run:
        python -m tests.test_ask
"""

import httpx
import asyncio
import json

BASE_URL = "http://localhost:8000"

# Delay between requests, in seconds. Groq's free tier enforces a tokens-
# per-minute budget; firing all five questions back to back can exhaust it
# partway through this script, since each question's retrieved context plus
# generated answer consumes a meaningful share of that budget. This delay
# is a test-script convenience only, the production /ask endpoint itself
# does not rate-limit or pace requests.
DELAY_BETWEEN_REQUESTS_SECONDS = 8

QUESTIONS = [
    {
        "type": "1. Factual",
        "question": "How much borax could potentially kill a small child?",
    },
    {
        "type": "2. Synthesis",
        "question": "How did food adulteration and the lack of food safety legislation together contribute to public health problems in Victorian England?",
    },
    {
        "type": "3. Named person/location",
        "question": "Who was Thomas Crapper and what did he invent?",
    },
    {
        "type": "4. Vague",
        "question": "What were the dangers in Victorian homes?",
    },
    {
        "type": "5. Out-of-scope",
        "question": "What role did the internet play in Victorian household safety?",
    },
]


async def run_tests():
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

            try:
                response = await client.post(
                    f"{BASE_URL}/ask",
                    json={"question": q["question"]},
                )
                response.raise_for_status()
                data = response.json()

                print(f"A: {data['answer'][:300]}")
                if len(data["answer"]) > 300:
                    print("   ...")
                print(f"\nSources ({len(data['sources'])}):")
                for i, src in enumerate(data["sources"], 1):
                    print(f"  {i}. [{src['timestamp']}] {src['excerpt'][:100]}...")
            except Exception as e:
                print(f"ERROR: {e}")

            print()


if __name__ == "__main__":
    asyncio.run(run_tests())