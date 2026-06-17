import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    start_timestamp: str
    end_timestamp: str
    index: int
    segment_start_index: int
    segment_end_index: int


TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def parse_transcript(filepath: str) -> list[tuple[str, str]]:
    """
    Parse the transcript file into (timestamp, text) pairs.

    Expected format:
        HH:MM:SS
        <spoken text block>
        HH:MM:SS
        <spoken text block>
        ...
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()

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
    window_size: int = 3,
    overlap: int = 1,
) -> list[Chunk]:
    """
    Group consecutive transcript segments into overlapping chunks.

    Args:
        segments: List of (timestamp, text) pairs.
        window_size: Number of transcript segments per chunk.
        overlap: Number of segments shared between consecutive chunks.

    Returns:
        List of Chunk objects with text and timestamp metadata.
    """
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0.")

    if overlap >= window_size:
        raise ValueError("overlap must be smaller than window_size.")

    chunks: list[Chunk] = []
    step = window_size - overlap

    for start_index in range(0, len(segments), step):
        window = segments[start_index : start_index + window_size]
        if not window:
            break

        end_index = start_index + len(window) - 1
        combined_text = " ".join(text for _, text in window)

        chunks.append(
            Chunk(
                text=combined_text,
                start_timestamp=window[0][0],
                end_timestamp=window[-1][0],
                index=len(chunks),
                segment_start_index=start_index,
                segment_end_index=end_index,
            )
        )

    return chunks


def load_and_chunk(
    filepath: str,
    window_size: int = 3,
    overlap: int = 1,
) -> list[Chunk]:
    """Parse transcript and return overlapping chunks."""
    segments = parse_transcript(filepath)
    return create_chunks(segments, window_size=window_size, overlap=overlap)