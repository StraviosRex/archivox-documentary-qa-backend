import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import chromadb

from app.chunker import load_and_chunk
from app.config import settings
from app.embedder import embed_query, embed_texts

logger = logging.getLogger(__name__)

COLLECTION_NAME = "transcript"
INDEX_METADATA_FILE = "index_metadata.json"

# Minimum lexical score required for a chunk to survive on lexical grounds
# alone. Set to match the proper-name match weight, so a chunk only passes
# via a full query-substring match (10.0) or at least one proper-name phrase
# match (8.0). Accumulated generic single-term matches (1.0 each) can no
# longer qualify a chunk by themselves, since common domain vocabulary
# (e.g. "Victorian", "household") appears throughout the transcript
# regardless of what a specific question is actually asking about.
MIN_LEXICAL_SCORE = 8.0

_collection: chromadb.Collection | None = None


def _transcript_hash(path: str) -> str:
    """Return a SHA256 hash of the transcript file."""
    transcript_path = Path(path)

    if not transcript_path.exists():
        raise FileNotFoundError(
            f"Transcript file not found at {transcript_path}. "
            "Check TRANSCRIPT_PATH in your .env file."
        )

    digest = hashlib.sha256()

    with open(transcript_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _metadata_path() -> Path:
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return persist_dir / INDEX_METADATA_FILE


def _current_index_metadata() -> dict[str, Any]:
    """Build metadata describing the current index inputs/settings."""
    return {
        "transcript_path": str(Path(settings.transcript_path)),
        "transcript_hash": _transcript_hash(settings.transcript_path),
        "embedding_model": settings.embedding_model,
        "chunk_window_size": settings.chunk_window_size,
        "chunk_overlap": settings.chunk_overlap,
    }


def _load_saved_index_metadata() -> dict[str, Any] | None:
    path = _metadata_path()

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_index_metadata(metadata: dict[str, Any]) -> None:
    path = _metadata_path()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _index_needs_rebuild() -> bool:
    saved = _load_saved_index_metadata()
    current = _current_index_metadata()

    return saved != current


def get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection, rebuilding stale indexes."""
    global _collection

    if _collection is not None:
        return _collection

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    if _index_needs_rebuild():
        logger.info("Transcript index is missing or stale. Rebuilding ChromaDB index.")

        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        _index_transcript(collection)
        _save_index_metadata(_current_index_metadata())

    _collection = collection
    return _collection


def _index_transcript(collection: chromadb.Collection) -> None:
    """Parse, chunk, embed, and store the transcript."""
    chunks = load_and_chunk(
        settings.transcript_path,
        window_size=settings.chunk_window_size,
        overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise RuntimeError(
            f"No transcript chunks were created from {settings.transcript_path}."
        )

    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)

    collection.add(
        ids=[f"chunk_{chunk.index}" for chunk in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "start_timestamp": chunk.start_timestamp,
                "end_timestamp": chunk.end_timestamp,
                "index": chunk.index,
                "segment_start_index": chunk.segment_start_index,
                "segment_end_index": chunk.segment_end_index,
            }
            for chunk in chunks
        ],
    )

    logger.info("Indexed %s chunks into ChromaDB.", len(chunks))


def _tokenize_query(query: str) -> list[str]:
    """Extract useful lowercase query terms for lexical fallback."""
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "was", "were", "is", "are", "did", "do", "does", "what", "who", "why",
        "how", "about", "he", "she", "it", "they", "his", "her", "their",
        "this", "that", "these", "those", "documentary", "transcript",
        "contribute", "contributed", "contribution", "safety", "safe",

        # Generic documentary/domain words that are too broad for lexical ranking.
        "victorian", "victorians",
        "home", "homes", "house", "houses",
        "danger", "dangers", "dangerous",
        "new", "ideas", "idea", "led", "lead",
    }

    terms = re.findall(r"[A-Za-z][A-Za-z0-9']+", query.lower())
    return [term for term in terms if len(term) > 2 and term not in stopwords]


def _proper_name_phrases(query: str) -> list[str]:
    """Extract proper-name phrases such as 'Thomas Crapper'."""
    return re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query)


def _lexical_score(query: str, text: str) -> float:
    """
    Simple lexical score for exact-name and rare-term fallback.

    Higher score means stronger lexical match.
    """
    query_lower = query.lower()
    text_lower = text.lower()

    score = 0.0

    if query_lower in text_lower:
        score += 10.0

    for phrase in _proper_name_phrases(query):
        if phrase.lower() in text_lower:
            score += 8.0

    for term in _tokenize_query(query):
        if term in text_lower:
            score += 1.0

    return score


def _lexical_search(
    collection: chromadb.Collection,
    query: str,
    limit: int,
) -> list[dict]:
    """
    Search all stored chunks using lexical matching.

    Only chunks that clear MIN_LEXICAL_SCORE are admitted. This keeps the
    fallback effective for its intended purpose, exact names, places, and
    full-phrase matches, while preventing chunks from qualifying purely on
    accumulated common-word overlap, which provides no real evidence of
    relevance for a transcript where that vocabulary appears throughout.
    """
    results = collection.get(include=["documents", "metadatas"])

    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    scored: list[tuple[float, dict]] = []

    for i, text in enumerate(documents):
        score = _lexical_score(query, text)

        if score < MIN_LEXICAL_SCORE:
            continue

        metadata = metadatas[i] or {}

        scored.append(
            (
                score,
                {
                    "id": ids[i],
                    "text": text,
                    "start_timestamp": metadata.get("start_timestamp", ""),
                    "end_timestamp": metadata.get("end_timestamp", ""),
                    "index": metadata.get("index"),
                    "segment_start_index": metadata.get("segment_start_index"),
                    "segment_end_index": metadata.get("segment_end_index"),
                    "distance": None,
                    "lexical_score": score,
                    "retrieval_method": "lexical",
                },
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:limit]]


def _prioritize_exact_name_matches(query: str, chunks: list[dict]) -> list[dict]:
    """
    Prioritize exact proper-name matches for named-entity questions.

    For proper-name queries such as "Thomas Crapper", dense retrieval can return
    broad but irrelevant chunks. In that case, keep exact-name chunks first and
    only keep immediate neighboring chunks as supporting context.
    """
    proper_names = _proper_name_phrases(query)

    if not proper_names:
        return chunks

    exact_name_chunks = [
        chunk
        for chunk in chunks
        if any(name.lower() in chunk["text"].lower() for name in proper_names)
    ]

    if not exact_name_chunks:
        return chunks

    exact_ids = {chunk["id"] for chunk in exact_name_chunks}
    exact_indexes = {
        chunk.get("index")
        for chunk in exact_name_chunks
        if isinstance(chunk.get("index"), int)
    }

    neighboring_chunks: list[dict] = []

    for chunk in chunks:
        if chunk["id"] in exact_ids:
            continue

        chunk_index = chunk.get("index")

        if isinstance(chunk_index, int) and any(
            abs(chunk_index - exact_index) <= 1 for exact_index in exact_indexes
        ):
            neighboring_chunks.append(chunk)

    return exact_name_chunks + neighboring_chunks


def _passes_similarity_threshold(chunk: dict) -> bool:
    """
    Decide whether a chunk clears the similarity bar.

    A chunk is kept if either signal independently validates it:
    - it has no distance at all (pure lexical match), or
    - its lexical score already clears MIN_LEXICAL_SCORE, even if it also
      has a dense distance from being found by both methods (hybrid), or
    - its dense distance falls within the configured threshold.

    Without the lexical_score check here, a chunk found by both dense and
    lexical search (hybrid) could be wrongly rejected on dense distance
    alone, even though the lexical match alone would have been strong
    enough to admit it on its own. Being found by both methods is a
    stronger relevance signal, not a weaker one, and should not be
    penalized relative to a pure lexical match.
    """
    lexical_score = chunk.get("lexical_score") or 0

    if lexical_score >= MIN_LEXICAL_SCORE:
        return True

    distance = chunk.get("distance")

    if distance is None:
        return True

    return distance <= settings.similarity_threshold


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """
    Retrieve the most relevant chunks for a query.

    Uses hybrid retrieval:
    - dense vector search for semantic similarity
    - lexical fallback for exact names, places, and rare terms

    Chunks whose dense similarity distance falls outside the configured
    threshold are dropped before the final cut, so the number of returned
    chunks naturally varies with how well a question is covered by the
    transcript instead of always returning a fixed count.

    Returns chunks ranked by combined relevance.
    """
    query = query.strip()

    if not query:
        return []

    if top_k is None:
        top_k = settings.top_k

    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)

    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k * 2, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    dense_chunks: list[dict] = []

    ids = dense_results.get("ids", [[]])[0]
    documents = dense_results.get("documents", [[]])[0]
    metadatas = dense_results.get("metadatas", [[]])[0]
    distances = dense_results.get("distances", [[]])[0]

    for i in range(len(ids)):
        metadata = metadatas[i] or {}

        dense_chunks.append(
            {
                "id": ids[i],
                "text": documents[i],
                "start_timestamp": metadata.get("start_timestamp", ""),
                "end_timestamp": metadata.get("end_timestamp", ""),
                "index": metadata.get("index"),
                "segment_start_index": metadata.get("segment_start_index"),
                "segment_end_index": metadata.get("segment_end_index"),
                "distance": distances[i],
                "lexical_score": _lexical_score(query, documents[i]),
                "retrieval_method": "dense",
            }
        )

    lexical_chunks = _lexical_search(
        collection=collection,
        query=query,
        limit=top_k,
    )

    merged: dict[str, dict] = {}

    for chunk in dense_chunks + lexical_chunks:
        chunk_id = chunk["id"]

        if chunk_id not in merged:
            merged[chunk_id] = chunk
            continue

        merged[chunk_id]["lexical_score"] = max(
            merged[chunk_id].get("lexical_score") or 0,
            chunk.get("lexical_score") or 0,
        )

        if merged[chunk_id].get("retrieval_method") == "dense":
            merged[chunk_id]["retrieval_method"] = "hybrid"

    chunks = list(merged.values())

    chunks.sort(
        key=lambda chunk: (
            -(chunk.get("lexical_score") or 0),
            chunk["distance"] if chunk.get("distance") is not None else 999.0,
        )
    )

    chunks = _prioritize_exact_name_matches(query, chunks)

    chunks = [chunk for chunk in chunks if _passes_similarity_threshold(chunk)]

    return chunks[:top_k]