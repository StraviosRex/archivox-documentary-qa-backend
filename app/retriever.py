"""
Index and retrieve transcript evidence for Archivox.

This module owns the retrieval pipeline end to end for the assignment-sized
app: Chroma index lifecycle, query metadata filters, lexical candidate scoring,
cross-encoder reranking, and final source selection. Keeping those pieces in
one module avoids premature file splitting while the functions below keep each
stage explicit and testable.
"""

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import chromadb

os.environ.setdefault("USE_TF", "0")

from sentence_transformers.cross_encoder import CrossEncoder

from app.chunker import load_and_chunk
from app.config import load_retrieval_config, settings
from app.embedder import embed_query, embed_texts


logger = logging.getLogger(__name__)
RETRIEVAL_CONFIG = load_retrieval_config()

COLLECTION_NAME = "transcript"
INDEX_METADATA_FILE = "index_metadata.json"

CANDIDATE_MULTIPLIER = 4
MIN_DENSE_CANDIDATES = 12
MAX_RETURNED_SOURCES = 3
TOP_DENSE_SOFT_MARGIN = 0.05

_CE_LOCAL = Path(__file__).parent.parent / "models" / "cross-encoder" / "ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_MODEL = str(_CE_LOCAL) if _CE_LOCAL.exists() else "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_THRESHOLD = -4.5
CE_LITERAL_INJECTION_CAP = 15
RERANKING_ENABLED = settings.reranking_enabled

_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder

TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d[\d,]*"
)

PROPER_NAME_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z.'’-]*"
    r"(?:\s+[A-Z][A-Za-z.'’-]*)+\b"
)

CAPITALIZED_TERM_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z.'’-]*\b"
)

STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "been", "being",
    "by", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "hers", "him", "his", "how", "in", "into", "is", "it", "its",
    "many", "much", "of", "on", "or", "our", "she", "that", "the", "their",
    "theirs", "them", "these", "they", "this", "those", "to", "was", "were",
    "what", "when", "where", "which", "who", "whom", "whose", "why", "with",
    "would", "documentary", "transcript",
}

LITERAL_TOPIC_STOPWORDS = set(
    str(term)
    for term in (
        RETRIEVAL_CONFIG
        .get("query_intent", {})
        .get("literal_topic_stopwords", [])
    )
)

SHORT_TOPIC_TERMS = set(
    str(term)
    for term in (
        RETRIEVAL_CONFIG
        .get("query_intent", {})
        .get("short_topic_terms", [])
    )
)

CONCEPT_ALIASES = {
    str(term): tuple(str(alias) for alias in aliases)
    for term, aliases in (
        RETRIEVAL_CONFIG
        .get("lexical", {})
        .get("concept_aliases", {})
        .items()
    )
    if isinstance(aliases, list)
}

QUESTION_WORDS = {
    "what", "who", "why", "how", "when",
    "where", "which", "whose", "whom",
}

DIVERSE_EVIDENCE_MARKERS = tuple(
    str(marker)
    for marker in (
        RETRIEVAL_CONFIG
        .get("query_intent", {})
        .get("diverse_evidence_markers", [])
    )
)

MULTI_ITEM_LIST_PATTERN = re.compile(
    r"\b\w+,\s+\w+.*\band\b",
    re.IGNORECASE,
)

BOTH_SUBJECT_PATTERN = re.compile(
    r"\bboth\s+(.+?)\s+and\s+(.+?)(?:[?!.]|$)",
    re.IGNORECASE,
)

EXPLICIT_COMPARISON_MARKERS = tuple(
    str(marker)
    for marker in (
        RETRIEVAL_CONFIG
        .get("query_intent", {})
        .get("explicit_comparison_markers", [])
    )
)

_collection: chromadb.Collection | None = None


# ---------------------------------------------------------------------------
# Index lifecycle
# ---------------------------------------------------------------------------


def _transcript_hash(path: str) -> str:
    """Return a SHA256 hash of the transcript file."""
    transcript_path = Path(path)

    if not transcript_path.exists():
        raise FileNotFoundError(
            f"Transcript file not found at {transcript_path}. "
            "Check TRANSCRIPT_PATH in your .env file."
        )

    digest = hashlib.sha256()

    with open(transcript_path, "rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _metadata_path() -> Path:
    """Return the path used for persisted index metadata."""
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    return persist_dir / INDEX_METADATA_FILE


def _current_index_metadata() -> dict[str, Any]:
    """Describe the current inputs used to build the vector index."""
    return {
        "transcript_path": str(Path(settings.transcript_path)),
        "transcript_hash": _transcript_hash(settings.transcript_path),
        "retrieval_config_hash": hashlib.sha256(
            json.dumps(
                RETRIEVAL_CONFIG,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "embedding_model": settings.embedding_model,
        "chunk_window_size": settings.chunk_window_size,
        "chunk_overlap": settings.chunk_overlap,
        "index_version": "6",
    }


def _load_saved_index_metadata() -> dict[str, Any] | None:
    """Load previously saved index metadata."""
    path = _metadata_path()

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_index_metadata(metadata: dict[str, Any]) -> None:
    """Persist index metadata to disk."""
    path = _metadata_path()

    with open(path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def _index_needs_rebuild() -> bool:
    """Return True when the stored index no longer matches its inputs."""
    return _load_saved_index_metadata() != _current_index_metadata()


def get_collection() -> chromadb.Collection:
    """Get or create the persistent ChromaDB collection."""
    global _collection

    if _collection is not None:
        return _collection

    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir
    )

    if _index_needs_rebuild():
        logger.info(
            "Transcript index is missing or stale. "
            "Rebuilding ChromaDB index."
        )

        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            logger.debug(
                "No existing ChromaDB collection needed deletion.",
                exc_info=True,
            )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        _index_transcript(collection)
        _save_index_metadata(_current_index_metadata())

    _collection = collection

    return _collection


def _chunk_metadata_for_chroma(chunk: Any) -> dict[str, Any]:
    """Return base plus configured scalar metadata for Chroma storage."""
    metadata = {
        "start_timestamp": chunk.start_timestamp,
        "end_timestamp": chunk.end_timestamp,
        "index": chunk.index,
        "segment_start_index": chunk.segment_start_index,
        "segment_end_index": chunk.segment_end_index,
    }

    metadata.update(
        {
            str(key): value
            for key, value in chunk.metadata.items()
            if isinstance(value, str | int | float | bool)
        }
    )

    return metadata


def _index_transcript(
    collection: chromadb.Collection,
) -> None:
    """Parse, chunk, embed, and store the transcript."""
    chunks = load_and_chunk(
        settings.transcript_path,
        window_size=settings.chunk_window_size,
        overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise RuntimeError(
            "No transcript chunks were created from "
            f"{settings.transcript_path}."
        )

    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)

    collection.add(
        ids=[
            f"chunk_{chunk.index}"
            for chunk in chunks
        ],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            _chunk_metadata_for_chroma(chunk)
            for chunk in chunks
        ],
    )

    logger.info(
        "Indexed %s chunks into ChromaDB.",
        len(chunks),
    )


# ---------------------------------------------------------------------------
# Text normalization and lexical features
# ---------------------------------------------------------------------------


def _normalize_token(token: str) -> str:
    """Normalize a word or numerical token."""
    normalized = token.strip().lower()

    if not normalized:
        return ""

    if normalized[0].isdigit():
        return re.sub(r"\D", "", normalized)

    return normalized.strip("'")


def _tokens(text: str) -> list[str]:
    """Extract normalized word and numerical tokens."""
    normalized_tokens: list[str] = []

    for raw_token in TOKEN_PATTERN.findall(text):
        token = _normalize_token(raw_token)

        if token:
            normalized_tokens.append(token)

    return normalized_tokens


def _query_terms(query: str) -> list[str]:
    """Extract meaningful query terms, preserving dates and numbers."""
    terms: list[str] = []

    for token in _tokens(query):
        if token.isdigit():
            terms.append(token)
            continue

        if len(token) <= 2:
            continue

        if token in STOPWORDS:
            continue

        terms.append(token)

    return list(dict.fromkeys(terms))


def _normalize_phrase(text: str) -> str:
    """Normalize text for phrase comparison."""
    return " ".join(_tokens(text))


def _proper_names(query: str) -> list[str]:
    """Extract useful multiword proper-name phrases."""
    names: list[str] = []

    for phrase in PROPER_NAME_PATTERN.findall(query):
        phrase_tokens = _tokens(phrase)

        if not phrase_tokens:
            continue

        if phrase_tokens[0] in QUESTION_WORDS:
            continue

        names.append(phrase)

    return list(dict.fromkeys(names))


def _capitalized_entities(query: str) -> list[str]:
    """Extract useful single-word capitalized names and locations."""
    entities: list[str] = []

    for raw_term in CAPITALIZED_TERM_PATTERN.findall(query):
        term = _normalize_token(raw_term)

        if not term:
            continue

        if term in QUESTION_WORDS:
            continue

        if term in STOPWORDS:
            continue

        if len(term) <= 2:
            continue

        entities.append(term)

    return list(dict.fromkeys(entities))


def _extract_literal_query_terms(query: str) -> set[str]:
    """Return lowercase content words (len >= 5, non-stopword) for literal chunk matching."""
    tokens = TOKEN_PATTERN.findall(query.lower())
    return {
        t for t in tokens
        if len(t) >= 5 and t not in STOPWORDS and t not in QUESTION_WORDS
    }


def _literal_topic_terms(query: str) -> set[str]:
    """Return literal query terms that are likely to name distinct topics."""
    terms = _extract_literal_query_terms(query) - LITERAL_TOPIC_STOPWORDS
    terms.update(
        SHORT_TOPIC_TERMS.intersection(set(_tokens(query)))
    )
    return terms


def _lexical_features(
    query: str,
    text: str,
) -> dict[str, Any]:
    """Calculate simple lexical relevance features."""
    query_terms = set(_query_terms(query))
    text_terms = set(_tokens(text))
    normalized_text = _normalize_phrase(text)

    matched_terms = query_terms.intersection(text_terms)

    # Match concepts when the question and transcript use different wording.
    for canonical_term, aliases in CONCEPT_ALIASES.items():
        if canonical_term not in query_terms:
            continue

        alias_found = any(
            (
                alias in normalized_text
                if " " in alias
                else alias in text_terms
            )
            for alias in aliases
        )

        if alias_found:
            matched_terms.add(canonical_term)

    query_term_count = len(query_terms)
    matched_term_count = len(matched_terms)

    coverage = (
        matched_term_count / query_term_count
        if query_term_count
        else 0.0
    )

    numeric_terms = {
        term
        for term in query_terms
        if term.isdigit()
    }

    numeric_matches = len(
        numeric_terms.intersection(text_terms)
    )

    proper_name_matches = sum(
        1
        for name in _proper_names(query)
        if _normalize_phrase(name) in normalized_text
    )

    entity_terms = set(_capitalized_entities(query))

    entity_matches = len(
        entity_terms.intersection(text_terms)
    )

    lexical_score = (
        matched_term_count
        + coverage * 5.0
        + numeric_matches * 4.0
        + proper_name_matches * 8.0
        + entity_matches * 2.0
    )

    return {
        "matched_terms": sorted(matched_terms),
        "matched_term_count": matched_term_count,
        "query_term_count": query_term_count,
        "lexical_coverage": coverage,
        "numeric_matches": numeric_matches,
        "proper_name_matches": proper_name_matches,
        "entity_matches": entity_matches,
        "lexical_score": lexical_score,
    }


def _has_strong_lexical_match(
    features: dict[str, Any],
) -> bool:
    """Return True when lexical evidence is independently meaningful."""
    if "matched_term_count" not in features:
        return False
    matched_count = features["matched_term_count"]
    coverage = features["lexical_coverage"]

    if features["proper_name_matches"] > 0:
        return True

    if (
        features["numeric_matches"] > 0
        and matched_count >= 2
    ):
        return True

    if (
        features["entity_matches"] > 0
        and matched_count >= 2
    ):
        return True

    if matched_count >= 4:
        return True

    if matched_count >= 3 and coverage >= 0.45:
        return True

    return False


# ---------------------------------------------------------------------------
# Chunk loading and metadata filtering
# ---------------------------------------------------------------------------


def _base_chunk(
    chunk_id: str,
    text: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Create the internal representation of a stored transcript chunk."""
    base_keys = {
        "start_timestamp",
        "end_timestamp",
        "index",
        "segment_start_index",
        "segment_end_index",
    }

    chunk_metadata = {
        str(key): value
        for key, value in metadata.items()
        if key not in base_keys and value is not None
    }

    chunk = {
        "id": chunk_id,
        "text": text,
        "start_timestamp": metadata.get(
            "start_timestamp",
            "",
        ),
        "end_timestamp": metadata.get(
            "end_timestamp",
            "",
        ),
        "index": metadata.get("index"),
        "segment_start_index": metadata.get(
            "segment_start_index"
        ),
        "segment_end_index": metadata.get(
            "segment_end_index"
        ),
        "metadata": chunk_metadata,
    }

    chunk.update(chunk_metadata)

    return chunk


def _load_all_chunks(
    collection: chromadb.Collection,
) -> list[dict[str, Any]]:
    """Load all stored transcript chunks and metadata."""
    results = collection.get(
        include=["documents", "metadatas"]
    )

    ids = results.get("ids", []) or []
    documents = results.get("documents", []) or []
    metadatas = results.get("metadatas", []) or []

    chunks: list[dict[str, Any]] = []

    for position, chunk_id in enumerate(ids):
        chunks.append(
            _base_chunk(
                chunk_id=chunk_id,
                text=documents[position],
                metadata=metadatas[position] or {},
            )
        )

    return chunks


def _metadata_field_configs() -> dict[str, Any]:
    """Return configured metadata field rules."""
    fields = (
        RETRIEVAL_CONFIG
        .get("metadata", {})
        .get("fields", {})
    )

    return fields if isinstance(fields, dict) else {}


def _compile_metadata_query_patterns() -> dict[str, dict[str, Any]]:
    """Build optional query-time metadata filters from retrieval config."""
    fields = _metadata_field_configs()

    if not fields:
        return {}

    compiled: dict[str, dict[str, Any]] = {}

    for field_name, field_rule in fields.items():
        if not isinstance(field_rule, dict):
            continue

        labels = field_rule.get("labels", {})

        if not isinstance(labels, dict):
            continue

        compiled_labels: dict[str, re.Pattern] = {}

        for label, label_rule in labels.items():
            patterns = (
                label_rule.get("query_patterns", [])
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


METADATA_QUERY_PATTERNS = _compile_metadata_query_patterns()


def _detect_query_filters(query: str) -> dict[str, frozenset[str]]:
    """Return metadata filters explicitly mentioned in the query."""
    filters: dict[str, frozenset[str]] = {}

    for field_name, field_rule in METADATA_QUERY_PATTERNS.items():
        label_patterns = field_rule.get("labels", {})

        matched_labels = {
            label
            for label, pattern in label_patterns.items()
            if pattern.search(query)
        }

        if not matched_labels:
            continue

        default_label = field_rule.get("default")

        if default_label:
            matched_labels.add(str(default_label))

        filters[field_name] = frozenset(matched_labels)

    return filters


def _chunk_matches_query_filters(
    chunk: dict[str, Any],
    query_filters: dict[str, frozenset[str]],
) -> bool:
    """Return True when a chunk satisfies all detected metadata filters."""
    metadata = chunk.get("metadata", {})

    for field_name, allowed_values in query_filters.items():
        value = metadata.get(field_name, chunk.get(field_name))

        if value not in allowed_values:
            return False

    return True


def _where_from_query_filters(
    query_filters: dict[str, frozenset[str]],
) -> dict[str, Any] | None:
    """Build a Chroma where clause from query metadata filters."""
    if not query_filters:
        return None

    clauses = [
        {
            field_name: {
                "$in": sorted(allowed_values),
            }
        }
        for field_name, allowed_values in sorted(query_filters.items())
    ]

    if len(clauses) == 1:
        return clauses[0]

    return {"$and": clauses}


def _explicit_filter_label_count(
    query_filters: dict[str, frozenset[str]],
) -> int:
    """Count non-default metadata labels detected in the query."""
    count = 0

    for field_name, allowed_values in query_filters.items():
        field_rule = METADATA_QUERY_PATTERNS.get(
            field_name,
            {},
        )
        default_label = field_rule.get("default")
        explicit_values = set(allowed_values)

        if default_label:
            explicit_values.discard(str(default_label))

        count += len(explicit_values)

    return count


def _should_relax_metadata_filters(
    query: str,
    query_filters: dict[str, frozenset[str]],
) -> bool:
    """Return True when a multi-part query should search beyond one label."""
    if not query_filters:
        return False

    if not _needs_diverse_evidence(query):
        return False

    return _explicit_filter_label_count(query_filters) == 1


# ---------------------------------------------------------------------------
# Dense retrieval and candidate fusion
# ---------------------------------------------------------------------------


def _dense_results(
    collection: chromadb.Collection,
    query: str,
    candidate_limit: int,
    where: dict | None = None,
) -> dict[str, dict[str, Any]]:
    """Retrieve dense semantic candidates using the complete question."""
    query_embedding = embed_query(query)

    query_kwargs: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(candidate_limit, collection.count()),
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    dense: dict[str, dict[str, Any]] = {}

    for rank, chunk_id in enumerate(ids, start=1):
        chunk = _base_chunk(
            chunk_id=chunk_id,
            text=documents[rank - 1],
            metadata=metadatas[rank - 1] or {},
        )

        chunk["distance"] = distances[rank - 1]
        chunk["dense_rank"] = rank

        dense[chunk_id] = chunk

    return dense


def _combined_score(
    distance: float | None,
    lexical_score: float,
    proper_name_matches: int,
    numeric_matches: int,
) -> float:
    """Combine dense relevance and lexical relevance."""
    dense_relevance = (
        max(0.0, 1.0 - distance)
        if distance is not None
        else 0.0
    )

    normalized_lexical = min(
        1.0,
        lexical_score / 12.0,
    )

    return (
        dense_relevance * 0.55
        + normalized_lexical * 0.45
        + proper_name_matches * 0.10
        + numeric_matches * 0.05
    )


def _decorate_chunk(
    query: str,
    chunk: dict[str, Any],
    distance: float | None = None,
    dense_rank: int | None = None,
    retrieval_method: str = "lexical",
) -> dict[str, Any]:
    """Add retrieval and lexical scoring metadata to a chunk."""
    decorated = dict(chunk)

    features = _lexical_features(
        query=query,
        text=chunk["text"],
    )

    decorated.update(features)
    decorated["distance"] = distance
    decorated["dense_rank"] = dense_rank
    decorated["retrieval_method"] = retrieval_method

    decorated["relevance_score"] = _combined_score(
        distance=distance,
        lexical_score=features["lexical_score"],
        proper_name_matches=features["proper_name_matches"],
        numeric_matches=features["numeric_matches"],
    )

    return decorated


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------


def _apply_cross_encoder(
    query: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score each candidate against the query using a cross-encoder and re-rank."""
    if not candidates:
        return candidates

    cross_encoder = _get_cross_encoder()

    pairs = [[query, chunk["text"]] for chunk in candidates]
    scores = cross_encoder.predict(pairs, show_progress_bar=False).tolist()

    for chunk, score in zip(candidates, scores):
        chunk["cross_encoder_score"] = score

    candidates.sort(key=lambda c: c["cross_encoder_score"], reverse=True)

    for chunk in candidates:
        logger.debug(
            "CE score=%.3f sources=%s ts=%s terms=%s",
            chunk["cross_encoder_score"],
            chunk.get("candidate_sources", []),
            chunk.get("start_timestamp", "?"),
            chunk.get("matched_terms", []),
        )

    return candidates


def _build_candidates(
    query: str,
    all_chunks: list[dict[str, Any]],
    dense_chunks: dict[str, dict[str, Any]],
    candidate_limit: int,
) -> list[dict[str, Any]]:
    """Merge dense candidates with the strongest lexical candidates."""
    all_by_id = {
        chunk["id"]: chunk
        for chunk in all_chunks
    }

    lexical_candidates: list[dict[str, Any]] = []

    for chunk in all_chunks:
        decorated = _decorate_chunk(
            query=query,
            chunk=chunk,
        )

        if _has_strong_lexical_match(decorated):
            lexical_candidates.append(decorated)

    lexical_candidates.sort(
        key=lambda chunk: (
            -chunk["lexical_score"],
            -chunk["lexical_coverage"],
            -chunk["matched_term_count"],
        )
    )

    lexical_candidates = lexical_candidates[
        :candidate_limit
    ]

    logger.debug(
        "Candidates: dense=%s lexical=%s",
        len(dense_chunks),
        len(lexical_candidates),
    )

    candidate_ids = set(dense_chunks)
    candidate_ids.update(
        chunk["id"]
        for chunk in lexical_candidates
    )

    candidates: list[dict[str, Any]] = []

    for chunk_id in candidate_ids:
        base = all_by_id[chunk_id]
        dense = dense_chunks.get(chunk_id)

        if dense is not None:
            lexical_method = _lexical_features(
                query=query,
                text=base["text"],
            )

            method = (
                "hybrid"
                if _has_strong_lexical_match(
                    lexical_method
                )
                else "dense"
            )

            candidate = _decorate_chunk(
                query=query,
                chunk=base,
                distance=dense["distance"],
                dense_rank=dense["dense_rank"],
                retrieval_method=method,
            )
        else:
            candidate = _decorate_chunk(
                query=query,
                chunk=base,
                retrieval_method="lexical",
            )

        candidates.append(candidate)

    candidates.sort(
        key=lambda chunk: (
            -chunk["relevance_score"],
            chunk["distance"]
            if chunk["distance"] is not None
            else 999.0,
            chunk["index"]
            if isinstance(chunk.get("index"), int)
            else 999999,
        )
    )

    return candidates


# ---------------------------------------------------------------------------
# Relevance filtering and query intent
# ---------------------------------------------------------------------------


def _passes_relevance_filter(
    chunk: dict[str, Any],
) -> bool:
    """Return True when a candidate is sufficiently relevant."""
    ce_score = chunk.get("cross_encoder_score")
    if ce_score is not None:
        # CE is authoritative — no other signal overrides it.
        return ce_score >= CROSS_ENCODER_THRESHOLD

    # CE unavailable (reranking disabled): fall back to lexical and dense signals.
    if _has_strong_lexical_match(chunk):
        return True

    distance = chunk.get("distance")

    if distance is None:
        return False

    if distance <= settings.similarity_threshold:
        return True

    dense_rank = chunk.get("dense_rank")

    if (
        dense_rank == 1
        and distance
        <= settings.similarity_threshold
        + TOP_DENSE_SOFT_MARGIN
    ):
        return True

    return False


def _has_literal_topic_match(
    query: str,
    chunk: dict[str, Any],
) -> bool:
    """Return True when a chunk exactly matches a specific query topic."""
    literal_terms = _literal_topic_terms(query)
    matched_terms = set(chunk.get("matched_terms", []))
    matched_literals = literal_terms.intersection(matched_terms)

    return any(len(term) >= 6 for term in matched_literals)


def _passes_diverse_relevance_filter(
    query: str,
    chunk: dict[str, Any],
) -> bool:
    """Keep exact-topic matches for broad or multi-part questions."""
    if _passes_relevance_filter(chunk):
        return True

    if not _needs_diverse_evidence(query):
        return False

    return _has_literal_topic_match(
        query=query,
        chunk=chunk,
    )


def _needs_diverse_evidence(query: str) -> bool:
    """Detect questions that clearly request broad or multi-part evidence."""
    lowered = query.lower()

    if any(marker in lowered for marker in DIVERSE_EVIDENCE_MARKERS):
        return True

    if MULTI_ITEM_LIST_PATTERN.search(query):
        return True

    return False


def _has_explicit_comparison(query: str) -> bool:
    """Return True for questions that explicitly compare separate topics."""
    lowered = query.lower()

    return any(
        marker in lowered
        for marker in EXPLICIT_COMPARISON_MARKERS
    )


def _strongest_chunk_covers_both(
    query: str,
    relevant: list[dict[str, Any]],
) -> bool:
    """Check whether the strongest chunk contains both named subjects."""
    if not relevant:
        return False

    match = BOTH_SUBJECT_PATTERN.search(query)

    if match is None:
        return False

    left_terms = _query_terms(match.group(1))
    right_terms = _query_terms(match.group(2))

    if not left_terms or not right_terms:
        return False

    left_subject = left_terms[0]
    right_subject = right_terms[0]

    anchor_terms = set(
        _tokens(relevant[0]["text"])
    )

    return (
        left_subject in anchor_terms
        and right_subject in anchor_terms
    )


# ---------------------------------------------------------------------------
# Final source selection
# ---------------------------------------------------------------------------


def _chunk_index(chunk: dict[str, Any]) -> int | None:
    """Return a chunk index when it is valid."""
    index = chunk.get("index")

    return index if isinstance(index, int) else None


def _neighbor_chunks(
    query: str,
    anchor: dict[str, Any],
    all_by_index: dict[int, dict[str, Any]],
    candidate_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the immediate transcript neighbors of an anchor chunk."""
    anchor_index = _chunk_index(anchor)

    if anchor_index is None:
        return []

    neighbors: list[dict[str, Any]] = []

    for neighbor_index in (
        anchor_index - 1,
        anchor_index + 1,
    ):
        base = all_by_index.get(neighbor_index)

        if base is None:
            continue

        existing = candidate_by_id.get(base["id"])

        if existing is not None:
            neighbor = dict(existing)
        else:
            neighbor = _decorate_chunk(
                query=query,
                chunk=base,
                retrieval_method="neighbor",
            )

        neighbor["context_neighbor"] = True
        neighbors.append(neighbor)

    neighbors.sort(
        key=lambda chunk: (
            -chunk["relevance_score"],
            chunk["index"],
        )
    )

    return neighbors


def _select_local_sources(
    query: str,
    relevant: list[dict[str, Any]],
    all_chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Select the strongest local evidence and neighboring context."""
    if not relevant:
        return []

    result_limit = min(
        top_k,
        MAX_RETURNED_SOURCES,
    )

    anchor = relevant[0]
    selected = [anchor]
    selected_ids = {anchor["id"]}

    all_by_index = {
        chunk["index"]: chunk
        for chunk in all_chunks
        if isinstance(chunk.get("index"), int)
    }

    candidate_by_id = {
        chunk["id"]: chunk
        for chunk in relevant
    }

    for neighbor in _neighbor_chunks(
        query=query,
        anchor=anchor,
        all_by_index=all_by_index,
        candidate_by_id=candidate_by_id,
    ):
        if len(selected) >= result_limit:
            break

        if neighbor["id"] in selected_ids:
            continue

        selected.append(neighbor)
        selected_ids.add(neighbor["id"])

    # When CE is active, all chunks in `relevant` already passed the CE
    # threshold — trust CE ranking and skip the old fused-score gate.
    anchor_ce = anchor.get("cross_encoder_score")
    minimum_score = (
        None if anchor_ce is not None
        else anchor["relevance_score"] * 0.65
    )

    for candidate in relevant[1:]:
        if len(selected) >= result_limit:
            break

        if candidate["id"] in selected_ids:
            continue

        if minimum_score is not None and candidate["relevance_score"] < minimum_score:
            continue

        selected.append(candidate)
        selected_ids.add(candidate["id"])

    return selected


def _select_diverse_sources(
    query: str,
    relevant: list[dict[str, Any]],
    all_chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Select relevant chunks that cover distinct parts of a broad question."""
    if not relevant:
        return []

    result_limit = min(
        top_k,
        MAX_RETURNED_SOURCES,
    )

    anchor = relevant[0]
    selected = [anchor]
    selected_ids = {anchor["id"]}
    covered_terms = set(anchor["matched_terms"])

    # When CE is active, all chunks in `relevant` already passed the CE
    # threshold — trust CE ranking and skip the old fused-score gate.
    anchor_ce = anchor.get("cross_encoder_score")
    minimum_score = (
        None if anchor_ce is not None
        else anchor["relevance_score"] * 0.45
    )

    literal_query_terms = _literal_topic_terms(query)

    # Prioritise candidates that introduce new query term coverage before
    # same-topic duplicates crowd out other requested topics.
    anchor_terms = set(anchor["matched_terms"])

    def _literal_terms(chunk: dict[str, Any]) -> set[str]:
        return (
            set(chunk.get("matched_terms", []))
            .intersection(literal_query_terms)
        )

    def _is_distinct_region(chunk: dict[str, Any]) -> bool:
        candidate_index = _chunk_index(chunk)
        return all(
            candidate_index is None
            or _chunk_index(existing) is None
            or abs(
                candidate_index
                - _chunk_index(existing)
            ) > 1
            for existing in selected
        )

    while len(selected) < result_limit:
        uncovered_literal = literal_query_terms - covered_terms
        eligible: list[dict[str, Any]] = []

        for candidate in relevant[1:]:
            if candidate["id"] in selected_ids:
                continue

            if minimum_score is not None and candidate["relevance_score"] < minimum_score:
                continue

            candidate_terms = set(
                candidate["matched_terms"]
            )

            uncovered_candidate_literal = _literal_terms(candidate) - covered_terms

            if uncovered_literal and not uncovered_candidate_literal:
                continue

            adds_new_terms = bool(
                candidate_terms - covered_terms
            )

            if not adds_new_terms and not _is_distinct_region(candidate):
                continue

            eligible.append(candidate)

        if not eligible:
            break

        eligible.sort(
            key=lambda c: (
                -len(_literal_terms(c) - covered_terms),
                -max(
                    (
                        len(term)
                        for term in _literal_terms(c) - covered_terms
                    ),
                    default=0,
                ),
                -len(_literal_terms(c)),
                not _is_distinct_region(c),
                -c.get("cross_encoder_score", -999.0),
            ),
        )

        candidate = eligible[0]
        candidate_terms = set(
            candidate["matched_terms"]
        )

        selected.append(candidate)
        selected_ids.add(candidate["id"])
        covered_terms.update(candidate_terms)

    if len(selected) < 2:
        all_by_index = {
            chunk["index"]: chunk
            for chunk in all_chunks
            if isinstance(chunk.get("index"), int)
        }

        candidate_by_id = {
            chunk["id"]: chunk
            for chunk in relevant
        }

        for neighbor in _neighbor_chunks(
            query=query,
            anchor=anchor,
            all_by_index=all_by_index,
            candidate_by_id=candidate_by_id,
        ):
            if len(selected) >= result_limit:
                break

            if neighbor["id"] in selected_ids:
                continue

            selected.append(neighbor)
            selected_ids.add(neighbor["id"])

    for candidate in relevant[1:]:
        if len(selected) >= result_limit:
            break

        if candidate["id"] in selected_ids:
            continue

        if minimum_score is not None and candidate["relevance_score"] < minimum_score:
            continue

        selected.append(candidate)
        selected_ids.add(candidate["id"])

    return selected


# ---------------------------------------------------------------------------
# Public retrieval entry point
# ---------------------------------------------------------------------------


def retrieve(
    query: str,
    top_k: int | None = None,
) -> list[dict]:
    """
    Retrieve ranked transcript evidence for a natural-language question.

    Factual questions use the strongest result and its immediate transcript
    neighbors. Broad or explicit synthesis questions may select evidence from
    distinct transcript regions, each passing relevance filtering.

    Neighbor chunks are context expansion — they are attached to an anchor
    that passed relevance filtering, but do not independently pass it
    themselves. They are marked with context_neighbor=True.
    """
    query = query.strip()

    if not query:
        return []

    if top_k is None:
        top_k = settings.top_k

    if top_k <= 0:
        return []

    collection = get_collection()

    collection_count = collection.count()

    if collection_count == 0:
        return []

    candidate_limit = min(
        max(
            top_k * CANDIDATE_MULTIPLIER,
            MIN_DENSE_CANDIDATES,
        ),
        collection_count,
    )

    all_chunks = _load_all_chunks(collection)

    query_filters = _detect_query_filters(query)

    if _should_relax_metadata_filters(
        query=query,
        query_filters=query_filters,
    ):
        logger.debug(
            "Relaxing metadata filters for multi-part query: %s",
            query_filters,
        )
        query_filters = {}

    logger.debug(
        "Metadata filters: %s",
        query_filters,
    )

    if query_filters:
        all_chunks = [
            chunk
            for chunk in all_chunks
            if _chunk_matches_query_filters(
                chunk=chunk,
                query_filters=query_filters,
            )
        ]

    logger.debug(
        "Metadata sample (first 3 chunks): %s",
        [
            (
                chunk.get("id"),
                chunk.get("metadata"),
            )
            for chunk in all_chunks[:3]
        ],
    )

    t0 = time.perf_counter()

    metadata_where = _where_from_query_filters(query_filters)

    t_dense_start = time.perf_counter()
    dense_chunks = _dense_results(
        collection=collection,
        query=query,
        candidate_limit=candidate_limit,
        where=metadata_where,
    )
    t_dense_ms = round((time.perf_counter() - t_dense_start) * 1000)

    t_build_start = time.perf_counter()
    candidates = _build_candidates(
        query=query,
        all_chunks=all_chunks,
        dense_chunks=dense_chunks,
        candidate_limit=candidate_limit,
    )
    t_build_ms = round((time.perf_counter() - t_build_start) * 1000)

    t_ce_start = time.perf_counter()
    if RERANKING_ENABLED:
        # Inject literal query-term matches using token-based matching so only
        # exact word forms qualify (not substrings of longer words).
        candidate_ids = {c["id"] for c in candidates}
        literal_terms = _literal_topic_terms(query)
        literal_injections: list[dict[str, Any]] = []
        for chunk in all_chunks:
            if chunk["id"] in candidate_ids:
                continue
            chunk_tokens = set(_tokens(chunk["text"]))
            if any(term in chunk_tokens for term in literal_terms):
                decorated = _decorate_chunk(query=query, chunk=chunk)
                decorated["candidate_sources"] = ["literal"]
                literal_injections.append(decorated)

        # Cap injections to avoid flooding the CE pool with noise (e.g. a
        # common word like "dangers" matching dozens of unrelated chunks).
        literal_injections.sort(key=lambda c: -c["relevance_score"])
        for inj in literal_injections[:CE_LITERAL_INJECTION_CAP]:
            candidates.append(inj)
            candidate_ids.add(inj["id"])

        ce_candidate_count = len(candidates)
        _apply_cross_encoder(query=query, candidates=candidates)
        # _apply_cross_encoder sorts candidates in-place; no second sort needed.
    else:
        ce_candidate_count = 0
    t_ce_ms = round((time.perf_counter() - t_ce_start) * 1000)

    t_filter_start = time.perf_counter()
    use_diverse = _needs_diverse_evidence(query)

    relevant = [
        candidate
        for candidate in candidates
        if (
            _passes_diverse_relevance_filter(
                query=query,
                chunk=candidate,
            )
            if use_diverse
            else _passes_relevance_filter(candidate)
        )
    ]
    t_filter_ms = round((time.perf_counter() - t_filter_start) * 1000)

    if (
        "both " in query.lower()
        and not _has_explicit_comparison(query)
        and _strongest_chunk_covers_both(
            query=query,
            relevant=relevant,
        )
    ):
        use_diverse = False

    t_select_start = time.perf_counter()
    if use_diverse:
        selected = _select_diverse_sources(
            query=query,
            relevant=relevant,
            all_chunks=all_chunks,
            top_k=top_k,
        )
        mode = "diverse"
    else:
        selected = _select_local_sources(
            query=query,
            relevant=relevant,
            all_chunks=all_chunks,
            top_k=top_k,
        )
        mode = "local"
    t_select_ms = round((time.perf_counter() - t_select_start) * 1000)

    t_total_ms = round((time.perf_counter() - t0) * 1000)
    context_chars = sum(len(c["text"]) for c in selected)

    logger.info(
        "Retrieval completed. mode=%s selected=%s total_ms=%s",
        mode,
        len(selected),
        t_total_ms,
    )
    logger.debug(
        (
            "Retrieval diagnostics: dense=%s merged=%s ce_pool=%s relevant=%s "
            "context_chars=%s "
            "dense_ms=%s build_ms=%s ce_ms=%s filter_ms=%s select_ms=%s "
            "ce_top_score=%s"
        ),
        len(dense_chunks),
        len(candidates),
        ce_candidate_count,
        len(relevant),
        context_chars,
        t_dense_ms,
        t_build_ms,
        t_ce_ms,
        t_filter_ms,
        t_select_ms,
        round(candidates[0]["cross_encoder_score"], 3) if candidates and "cross_encoder_score" in candidates[0] else None,
    )

    return selected
