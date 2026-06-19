import re
from dataclasses import dataclass, field

from app.config import load_retrieval_config


@dataclass
class Chunk:
    text: str
    start_timestamp: str
    end_timestamp: str
    index: int
    segment_start_index: int
    segment_end_index: int
    metadata: dict[str, str] = field(default_factory=dict)


TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")


def _metadata_field_configs() -> dict:
    """Return metadata field rules from retrieval config."""
    retrieval_config = load_retrieval_config()
    fields = (
        retrieval_config
        .get("metadata", {})
        .get("fields", {})
    )

    return fields if isinstance(fields, dict) else {}


def _compile_metadata_patterns() -> dict[str, dict]:
    """Build optional corpus metadata classifiers from retrieval config."""
    fields = _metadata_field_configs()

    if not fields:
        return {}

    compiled: dict[str, dict] = {}

    for field_name, field_rule in fields.items():
        if not isinstance(field_rule, dict):
            continue

        labels = field_rule.get("labels", {})

        if not isinstance(labels, dict):
            continue

        compiled_labels: dict[str, re.Pattern] = {}

        for label, label_rule in labels.items():
            patterns = (
                label_rule.get("chunk_patterns", [])
                if isinstance(label_rule, dict)
                else []
            )

            if not patterns:
                continue

            compiled_labels[str(label)] = re.compile(
                "|".join(str(pattern) for pattern in patterns),
                re.IGNORECASE,
            )

        if not compiled_labels:
            continue

        default_label = (
            str(field_rule["default"])
            if field_rule.get("default") is not None
            else None
        )

        compiled[str(field_name)] = {
            "default": default_label,
            "labels": compiled_labels,
        }

    return compiled


_METADATA_PATTERNS = _compile_metadata_patterns()


def _detect_chunk_metadata(text: str) -> dict[str, str]:
    """Apply configured metadata field rules to transcript chunk text."""
    metadata: dict[str, str] = {}

    for field_name, field_rule in _METADATA_PATTERNS.items():
        label_patterns = (
            field_rule.get("labels", {})
            if isinstance(field_rule, dict)
            else {}
        )

        if not label_patterns:
            continue

        counts = {
            label: len(pattern.findall(text))
            for label, pattern in label_patterns.items()
        }

        max_count = max(counts.values())
        default_label = field_rule.get("default")

        if max_count == 0:
            if default_label:
                metadata[field_name] = str(default_label)

            continue

        dominant = [
            label
            for label, count in counts.items()
            if count == max_count
        ]

        if len(dominant) == 1:
            metadata[field_name] = dominant[0]
        elif default_label:
            metadata[field_name] = str(default_label)

    return metadata


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

        metadata = _detect_chunk_metadata(combined_text)

        chunks.append(
            Chunk(
                text=combined_text,
                start_timestamp=window[0][0],
                end_timestamp=end_timestamp,
                index=len(chunks),
                segment_start_index=start_index,
                segment_end_index=end_index,
                metadata=metadata,
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
