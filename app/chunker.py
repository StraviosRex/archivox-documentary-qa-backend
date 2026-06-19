import re
from dataclasses import dataclass

from app.config import load_retrieval_config


@dataclass
class Chunk:
    text: str
    start_timestamp: str
    end_timestamp: str
    index: int
    segment_start_index: int
    segment_end_index: int
    era: str = "general"


TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def _compile_era_patterns() -> dict[str, re.Pattern]:
    """Build optional corpus-era classifiers from retrieval config."""
    retrieval_config = load_retrieval_config()
    eras = (
        retrieval_config
        .get("metadata", {})
        .get("eras", {})
    )

    if not isinstance(eras, dict):
        return {}

    compiled: dict[str, re.Pattern] = {}

    for era, rule in eras.items():
        patterns = (
            rule.get("chunk_patterns", [])
            if isinstance(rule, dict)
            else []
        )

        if not patterns:
            continue

        compiled[str(era)] = re.compile(
            "|".join(str(pattern) for pattern in patterns),
            re.IGNORECASE,
        )

    return compiled


_ERA_PATTERNS = _compile_era_patterns()


def _detect_chunk_era(text: str) -> str:
    if not _ERA_PATTERNS:
        return "general"

    counts = {era: len(pattern.findall(text)) for era, pattern in _ERA_PATTERNS.items()}
    max_count = max(counts.values())
    if max_count == 0:
        return "general"
    dominant = [era for era, count in counts.items() if count == max_count]
    if len(dominant) == 1:
        return dominant[0]
    return "general"


def parse_transcript(filepath: str) -> list[tuple[str, str]]:
    """
    Parse the transcript file into timestamped text segments.

    Expected format:
        HH:MM:SS
        <spoken text block>
        HH:MM:SS
        <spoken text block>
        ...
    """
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.read().strip().splitlines()

    segments: list[tuple[str, str]] = []
    current_timestamp: str | None = None
    current_text_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if TIMESTAMP_PATTERN.match(stripped):
            if current_timestamp and current_text_lines:
                text = " ".join(current_text_lines).strip()

                if text:
                    segments.append((current_timestamp, text))

            current_timestamp = stripped
            current_text_lines = []
            continue

        if stripped:
            current_text_lines.append(stripped)

    if current_timestamp and current_text_lines:
        text = " ".join(current_text_lines).strip()

        if text:
            segments.append((current_timestamp, text))

    return segments


def create_chunks(
    segments: list[tuple[str, str]],
    window_size: int = 2,
    overlap: int = 1,
) -> list[Chunk]:
    """
    Group consecutive transcript segments into overlapping chunks.

    Args:
        segments:
            Parsed transcript segments represented as timestamp-text pairs.
        window_size:
            Number of consecutive transcript segments included in each chunk.
        overlap:
            Number of transcript segments shared between consecutive chunks.

    Returns:
        Ordered Chunk objects containing transcript text and timestamp metadata.

    The start timestamp is taken from the first segment in the chunk. When a
    following transcript segment exists, its timestamp is used as the
    approximate end of the current chunk.
    """
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0.")

    if overlap >= window_size:
        raise ValueError("overlap must be smaller than window_size.")

    if not segments:
        return []

    chunks: list[Chunk] = []
    step = window_size - overlap
    previous_end_index = -1

    for start_index in range(0, len(segments), step):
        window = segments[start_index : start_index + window_size]

        if not window:
            break

        end_index = start_index + len(window) - 1

        # Prevent a trailing partial chunk that contributes no new segment.
        if end_index <= previous_end_index:
            break

        combined_text = " ".join(
            text.strip()
            for _, text in window
            if text.strip()
        )

        if not combined_text:
            continue

        next_segment_index = end_index + 1

        if next_segment_index < len(segments):
            end_timestamp = segments[next_segment_index][0]
        else:
            end_timestamp = window[-1][0]

        era = _detect_chunk_era(combined_text)

        chunks.append(
            Chunk(
                text=combined_text,
                start_timestamp=window[0][0],
                end_timestamp=end_timestamp,
                index=len(chunks),
                segment_start_index=start_index,
                segment_end_index=end_index,
                era=era,
            )
        )

        previous_end_index = end_index

    return chunks


def load_and_chunk(
    filepath: str,
    window_size: int = 2,
    overlap: int = 1,
) -> list[Chunk]:
    """Parse a transcript file and return overlapping transcript chunks."""
    segments = parse_transcript(filepath)

    return create_chunks(
        segments=segments,
        window_size=window_size,
        overlap=overlap,
    )
