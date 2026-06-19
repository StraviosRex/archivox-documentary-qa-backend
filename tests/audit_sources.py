"""
Audit /ask source citations against the original transcript.

This script complements tests/test_ask.py. It does not judge whether the LLM
answer is perfect, and it does not decide whether a source is semantically
"relevant"; it checks that returned source excerpts and timestamp anchors are
traceable to data/transcript.txt.

Usage:
    Start the server first:
        uvicorn app.main:app --port 8000

    Then run:
        python -m tests.audit_sources
"""

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from tests.test_ask import (
    DELAY_BETWEEN_REQUESTS_SECONDS,
    QUESTIONS,
)


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TRANSCRIPT_PATH = "data/transcript.txt"
TIMESTAMP_RANGE_PATTERN = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2})-(?P<end>\d{2}:\d{2}:\d{2})$"
)
TIMESTAMP_LINE_PATTERN = re.compile(
    r"^\d{2}:\d{2}:\d{2}$",
    re.MULTILINE,
)


def _transcript_body_text(text: str) -> str:
    """Remove standalone timestamp lines from transcript text."""
    return TIMESTAMP_LINE_PATTERN.sub(
        " ",
        text,
    )


def _normalize_text(text: str) -> str:
    """Normalize text for robust transcript/excerpt comparison."""
    normalized = (
        text.lower()
        .replace("’", "'")
        .replace("`", "'")
        .replace("—", " ")
        .replace("–", " ")
    )

    return " ".join(normalized.split())


def _loose_normalize_text(text: str) -> str:
    """Normalize text while ignoring punctuation and spacing differences."""
    normalized = _normalize_text(text)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def _excerpt_fragments(excerpt: str) -> list[str]:
    """Return meaningful excerpt fragments, ignoring ellipsis boundaries."""
    normalized_excerpt = _normalize_text(excerpt)

    fragments = [
        fragment.strip()
        for fragment in normalized_excerpt.split("...")
        if len(fragment.strip()) >= 24
    ]

    if fragments:
        return fragments

    if len(normalized_excerpt) >= 24:
        return [normalized_excerpt]

    return []


def _fragments_appear_in_order(
    fragments: list[str],
    normalized_text: str,
) -> bool:
    """Return True if all fragments occur in text order."""
    search_start = 0

    for fragment in fragments:
        index = normalized_text.find(
            fragment,
            search_start,
        )

        if index < 0:
            return False

        search_start = index + len(fragment)

    return True


def _timestamp_window_text(
    timestamp: str,
    transcript_text: str,
) -> str:
    """Return transcript text covered by a source timestamp range."""
    match = TIMESTAMP_RANGE_PATTERN.match(timestamp)

    if match is None:
        return transcript_text

    start = match.group("start")
    end = match.group("end")
    start_index = transcript_text.find(start)

    if start_index < 0:
        return transcript_text

    end_index = transcript_text.find(
        end,
        start_index + len(start),
    )

    if end_index < 0:
        return transcript_text[start_index:]

    return transcript_text[start_index:end_index]


def _audit_timestamp(
    timestamp: str,
    transcript_text: str,
) -> list[str]:
    """Return timestamp audit failures."""
    match = TIMESTAMP_RANGE_PATTERN.match(timestamp)

    if match is None:
        return [f"timestamp has unexpected format: {timestamp}"]

    failures: list[str] = []
    start = match.group("start")
    end = match.group("end")

    if start not in transcript_text:
        failures.append(f"start timestamp not found: {start}")

    if end not in transcript_text:
        failures.append(f"end timestamp not found: {end}")

    return failures


def _audit_source(
    source: dict[str, Any],
    transcript_text: str,
    normalized_transcript: str,
) -> list[str]:
    """Return audit failures for one source object."""
    failures: list[str] = []

    timestamp = str(source.get("timestamp", ""))
    excerpt = str(source.get("excerpt", ""))

    failures.extend(
        _audit_timestamp(
            timestamp=timestamp,
            transcript_text=transcript_text,
        )
    )

    fragments = _excerpt_fragments(excerpt)

    if not fragments:
        failures.append("excerpt is too short or empty to audit")
        return failures

    timestamp_window = _timestamp_window_text(
        timestamp=timestamp,
        transcript_text=transcript_text,
    )
    normalized_window = _normalize_text(
        _transcript_body_text(timestamp_window)
    )

    if _fragments_appear_in_order(
        fragments=fragments,
        normalized_text=normalized_window,
    ):
        return failures

    if _fragments_appear_in_order(
        fragments=fragments,
        normalized_text=normalized_transcript,
    ):
        return failures

    loose_fragments = [
        _loose_normalize_text(fragment)
        for fragment in fragments
    ]
    loose_fragments = [
        fragment
        for fragment in loose_fragments
        if len(fragment) >= 24
    ]

    if loose_fragments and (
        _fragments_appear_in_order(
            fragments=loose_fragments,
            normalized_text=_loose_normalize_text(
                _transcript_body_text(timestamp_window)
            ),
        )
        or _fragments_appear_in_order(
            fragments=loose_fragments,
            normalized_text=_loose_normalize_text(
                _transcript_body_text(transcript_text)
            ),
        )
    ):
        return failures

    failures.append("excerpt text was not traceable to transcript")

    return failures


def _audit_response_sources(
    data: dict[str, Any],
    transcript_text: str,
    normalized_transcript: str,
) -> list[str]:
    """Return source-grounding failures for an /ask response."""
    failures: list[str] = []
    sources = data.get("sources", [])

    if not isinstance(sources, list):
        return ["sources is not a list"]

    for index, source in enumerate(
        sources,
        start=1,
    ):
        if not isinstance(source, dict):
            failures.append(f"source {index} is not an object")
            continue

        source_failures = _audit_source(
            source=source,
            transcript_text=transcript_text,
            normalized_transcript=normalized_transcript,
        )

        for failure in source_failures:
            failures.append(f"source {index}: {failure}")

    return failures


async def _run_audit(
    base_url: str,
    transcript_path: Path,
    delay_seconds: int,
) -> bool:
    """Run source audits for the shared evaluation questions."""
    transcript_text = transcript_path.read_text(encoding="utf-8")
    normalized_transcript = _normalize_text(
        _transcript_body_text(transcript_text)
    )
    passed_count = 0

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            health = await client.get(f"{base_url}/health")
            health.raise_for_status()
        except Exception as exc:
            print(f"Server not reachable at {base_url}: {exc}")
            return False

        for index, question_case in enumerate(QUESTIONS):
            if index > 0 and delay_seconds > 0:
                print(
                    f"(waiting {delay_seconds}s to avoid provider rate limits...)\n"
                )
                await asyncio.sleep(delay_seconds)

            label = question_case["type"]
            question = question_case["question"]

            print("=" * 60)
            print(f"[{label}]")
            print(f"Q: {question}")
            print("-" * 60)

            start = time.perf_counter()

            try:
                response = await client.post(
                    f"{base_url}/ask",
                    json={"question": question},
                )
                response.raise_for_status()
                elapsed = time.perf_counter() - start
                data = response.json()
            except Exception as exc:
                print(f"ERROR: {exc}")
                continue

            answer = data.get("answer", "")
            sources = data.get("sources", [])
            failures = _audit_response_sources(
                data=data,
                transcript_text=transcript_text,
                normalized_transcript=normalized_transcript,
            )

            print(f"A: {str(answer)[:240]}")
            if len(str(answer)) > 240:
                print("   ...")

            print(f"\nSources ({len(sources)}):")
            for source_index, source in enumerate(
                sources,
                start=1,
            ):
                timestamp = source.get("timestamp", "?")
                excerpt = source.get("excerpt", "")
                print(
                    f"  {source_index}. [{timestamp}] {str(excerpt)[:100]}..."
                )

            print(f"\nResponse time: {elapsed:.2f}s")

            if failures:
                for failure in failures:
                    print(f"  FAIL: {failure}")
            else:
                print("  PASS: all returned sources trace to transcript")
                passed_count += 1

            print()

    total = len(QUESTIONS)
    print("=" * 60)
    print("Source audit summary")
    print("-" * 60)
    print(f"{passed_count}/{total} responses passed source audit")

    return passed_count == total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit /ask source citations against transcript text."
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"API base URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--transcript",
        default=DEFAULT_TRANSCRIPT_PATH,
        help=f"Transcript path. Default: {DEFAULT_TRANSCRIPT_PATH}",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=DELAY_BETWEEN_REQUESTS_SECONDS,
        help="Delay between requests in seconds.",
    )

    args = parser.parse_args()
    passed = asyncio.run(
        _run_audit(
            base_url=args.base_url.rstrip("/"),
            transcript_path=Path(args.transcript),
            delay_seconds=args.delay,
        )
    )

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
