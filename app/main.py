import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import list_public_profiles, settings
from app.llm import call_llm
from app.prompt import build_messages
from app.retriever import get_collection, retrieve


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
TIMESTAMP_PATTERN = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    profile: str | None = None


class Source(BaseModel):
    timestamp: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    profile: str | None = None
    provider: str | None = None
    model: str | None = None
    model_used: str | None = None


def _query_terms(text: str) -> list[str]:
    """Extract useful words, numbers, and proper-name phrases."""
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to",
        "in", "on", "for", "with", "was", "were",
        "is", "are", "did", "do", "does",
        "what", "who", "why", "how",
        "about", "he", "she", "it", "they",
        "his", "her", "their",
        "this", "that", "these", "those",
        "documentary", "transcript",
        "between", "from", "at",
    }

    proper_phrases = re.findall(
        r"\b[A-Z][A-Za-z.'’-]*"
        r"(?:\s+[A-Z][A-Za-z.'’-]*)+\b",
        text,
    )

    raw_terms = re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?|\d[\d,]*",
        text.lower(),
    )

    useful_terms: list[str] = []

    for raw_term in raw_terms:
        term = raw_term.replace(",", "")

        if term.isdigit():
            useful_terms.append(term)
            continue

        if len(term) <= 2:
            continue

        if term in stopwords:
            continue

        useful_terms.append(term)

    combined = proper_phrases + useful_terms

    return list(dict.fromkeys(combined))


def _sentence_matches(
    sentence: str,
    terms: list[str],
) -> set[str]:
    """Return supplied terms found in a sentence."""
    lowered_sentence = sentence.lower()

    sentence_tokens = {
        token.replace(",", "")
        for token in re.findall(
            r"[A-Za-z]+(?:'[A-Za-z]+)?|\d[\d,]*",
            lowered_sentence,
        )
    }

    matches: set[str] = set()

    for term in terms:
        lowered_term = term.lower()

        if " " in lowered_term:
            if lowered_term in lowered_sentence:
                matches.add(lowered_term)

            continue

        if lowered_term in sentence_tokens:
            matches.add(lowered_term)

    return matches


def _question_term_weight(term: str) -> float:
    """Return the score contribution of a question-term match."""
    if " " in term:
        return 4.0

    if term.isdigit():
        return 3.0

    if len(term) >= 8:
        return 1.5

    return 1.0


def _answer_term_weight(term: str) -> float:
    """Return the score contribution of an answer-term match."""
    if " " in term:
        return 6.0

    if term.isdigit():
        return 4.0

    if len(term) >= 8:
        return 3.0

    return 2.0


def _sentence_score(
    sentence: str,
    question_terms: list[str],
    answer_terms: list[str],
) -> tuple[float, set[str], set[str]]:
    """Score a sentence against the question and generated answer."""
    question_matches = _sentence_matches(
        sentence=sentence,
        terms=question_terms,
    )

    answer_matches = _sentence_matches(
        sentence=sentence,
        terms=answer_terms,
    )

    answer_only_matches = (
        answer_matches - question_matches
    )

    score = sum(
        _question_term_weight(term)
        for term in question_matches
    )

    score += sum(
        _answer_term_weight(term)
        for term in answer_only_matches
    )

    return (
        score,
        question_matches,
        answer_matches,
    )


def _shorten_around_term(
    text: str,
    terms: list[str],
    max_chars: int,
) -> str:
    """Shorten text around its most specific matching term."""
    lowered_text = text.lower()

    matching_terms = [
        term
        for term in terms
        if term.lower() in lowered_text
    ]

    matching_terms.sort(
        key=len,
        reverse=True,
    )

    if matching_terms:
        anchor_index = lowered_text.find(
            matching_terms[0].lower()
        )
    else:
        anchor_index = 0

    half_window = max_chars // 2
    start = max(
        anchor_index - half_window,
        0,
    )
    end = min(
        start + max_chars,
        len(text),
    )

    start = max(
        end - max_chars,
        0,
    )

    excerpt = text[start:end].strip()

    if start > 0:
        excerpt_parts = excerpt.split(" ", 1)

        if len(excerpt_parts) == 2:
            excerpt = excerpt_parts[1]

        excerpt = "..." + excerpt

    if end < len(text):
        excerpt = (
            excerpt.rsplit(" ", 1)[0]
            + "..."
        )

    return excerpt


def make_excerpt(
    text: str,
    question: str,
    answer: str,
    max_chars: int = 600,
) -> str:
    """
    Return the most relevant sentence group from a retrieved chunk.

    The complete chunk is sent to the LLM. This function only chooses the
    shorter source preview displayed in the API response and frontend.
    """
    cleaned = " ".join(text.split())

    if len(cleaned) <= max_chars:
        return cleaned

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(cleaned)
        if sentence.strip()
    ]

    answer_without_timestamps = TIMESTAMP_PATTERN.sub(
        " ",
        answer,
    )

    question_terms = _query_terms(question)
    answer_terms = _query_terms(
        answer_without_timestamps
    )

    fallback_terms = list(
        dict.fromkeys(
            answer_terms + question_terms
        )
    )

    if not sentences:
        return _shorten_around_term(
            text=cleaned,
            terms=fallback_terms,
            max_chars=max_chars,
        )

    best_start = 0
    best_end = 0
    best_score = -1.0
    best_coverage = -1
    best_length = max_chars + 1

    for start_index in range(len(sentences)):
        window_sentences: list[str] = []
        window_question_matches: set[str] = set()
        window_answer_matches: set[str] = set()
        window_score = 0.0

        for end_index in range(
            start_index,
            min(
                start_index + 3,
                len(sentences),
            ),
        ):
            sentence = sentences[end_index]

            candidate_sentences = (
                window_sentences + [sentence]
            )

            candidate_text = " ".join(
                candidate_sentences
            )

            if (
                len(candidate_text) > max_chars
                and window_sentences
            ):
                break

            (
                sentence_score,
                question_matches,
                answer_matches,
            ) = _sentence_score(
                sentence=sentence,
                question_terms=question_terms,
                answer_terms=answer_terms,
            )

            window_sentences.append(sentence)
            window_score += sentence_score

            window_question_matches.update(
                question_matches
            )
            window_answer_matches.update(
                answer_matches
            )

            answer_only_matches = (
                window_answer_matches
                - window_question_matches
            )

            coverage = (
                len(window_question_matches)
                + len(answer_only_matches)
            )

            combined_score = (
                window_score
                + len(window_question_matches) * 0.5
                + len(answer_only_matches) * 1.0
            )

            candidate_length = len(candidate_text)

            should_replace = (
                combined_score > best_score
                or (
                    combined_score == best_score
                    and coverage > best_coverage
                )
                or (
                    combined_score == best_score
                    and coverage == best_coverage
                    and candidate_length < best_length
                )
            )

            if should_replace:
                best_start = start_index
                best_end = end_index
                best_score = combined_score
                best_coverage = coverage
                best_length = candidate_length

    if best_score <= 0:
        return _shorten_around_term(
            text=cleaned,
            terms=fallback_terms,
            max_chars=max_chars,
        )

    selected_text = " ".join(
        sentences[best_start : best_end + 1]
    )

    if len(selected_text) > max_chars:
        selected_text = _shorten_around_term(
            text=selected_text,
            terms=fallback_terms,
            max_chars=max_chars,
        )
    else:
        if best_start > 0:
            selected_text = (
                "..." + selected_text
            )

        if best_end < len(sentences) - 1:
            selected_text += "..."

    return selected_text


def build_sources(
    chunks: list[dict],
    question: str,
    answer: str,
) -> list[Source]:
    """Build ranked source references from retrieved evidence chunks."""
    sources: list[Source] = []

    for chunk in chunks[
        : settings.sources_in_response
    ]:
        timestamp = (
            f'{chunk["start_timestamp"]}-'
            f'{chunk["end_timestamp"]}'
        )

        sources.append(
            Source(
                timestamp=timestamp,
                excerpt=make_excerpt(
                    text=chunk["text"],
                    question=question,
                    answer=answer,
                ),
            )
        )

    return sources


def is_not_enough_info_answer(
    answer: str | None,
) -> bool:
    """Detect transcript-abstention answers robustly."""
    if not answer:
        return False

    normalized = (
        answer.strip()
        .lower()
        .replace("’", "'")
        .replace("`", "'")
    )

    return (
        normalized.startswith(
            "i don't have enough information"
        )
        or (
            "don't have enough information "
            "in the transcript"
            in normalized
        )
        or (
            "do not have enough information "
            "in the transcript"
            in normalized
        )
        or "there is no mention of" in normalized
        or "no mention of" in normalized
        or "not mentioned in the transcript" in normalized
        or "the transcript does not" in normalized
        or "the transcript doesn't" in normalized
        or "not covered in the transcript" in normalized
        or "not discussed in the transcript" in normalized
    )


def log_retrieved_chunks(
    chunks: list[dict],
) -> None:
    """Log ranked retrieval metadata for local evaluation."""
    for rank, chunk in enumerate(
        chunks,
        start=1,
    ):
        logger.info(
            (
                "Retrieved rank=%s "
                "timestamp=%s-%s "
                "distance=%s "
                "lexical_score=%s "
                "method=%s "
                "segment_range=%s-%s"
            ),
            rank,
            chunk.get("start_timestamp"),
            chunk.get("end_timestamp"),
            chunk.get("distance"),
            chunk.get("lexical_score"),
            chunk.get("retrieval_method"),
            chunk.get("segment_start_index"),
            chunk.get("segment_end_index"),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Index the transcript on startup."""
    logger.info("Indexing transcript...")
    get_collection()
    logger.info("Ready.")

    yield


app = FastAPI(
    title="Archivox",
    description=(
        "Documentary Q&A Backend. Ask questions and receive answers "
        "grounded in transcript sources."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/profiles")
async def profiles():
    """Return frontend-safe LLM profile metadata."""
    return {
        "default_profile": settings.llm_profile,
        "profiles": list_public_profiles(),
    }


@app.post(
    "/ask",
    response_model=AskResponse,
)
async def ask(request: AskRequest):
    """Answer a question using retrieved transcript chunks."""
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    retrieved_chunks = retrieve(question)

    if not retrieved_chunks:
        return AskResponse(
            answer=(
                "I don't have enough information "
                "in the transcript to answer "
                "that question."
            ),
            sources=[],
        )

    evidence_chunks = retrieved_chunks[
        : settings.sources_in_response
    ]

    if not evidence_chunks:
        return AskResponse(
            answer=(
                "I don't have enough information "
                "in the transcript to answer "
                "that question."
            ),
            sources=[],
        )

    log_retrieved_chunks(evidence_chunks)

    messages = build_messages(
        question=question,
        retrieved_chunks=evidence_chunks,
    )

    try:
        llm_result = await call_llm(
            messages,
            profile_name=request.profile,
        )

        answer = llm_result.content
    except Exception:
        logger.exception("LLM request failed.")

        raise HTTPException(
            status_code=502,
            detail="LLM request failed.",
        )

    if not answer or not answer.strip():
        raise HTTPException(
            status_code=502,
            detail="LLM returned an empty answer.",
        )

    answer = answer.strip()

    if is_not_enough_info_answer(answer):
        return AskResponse(
            answer=answer,
            sources=[],
            profile=llm_result.profile,
            provider=llm_result.provider,
            model=llm_result.model_requested,
            model_used=llm_result.model_used,
        )

    return AskResponse(
        answer=answer,
        sources=build_sources(
            chunks=evidence_chunks,
            question=question,
            answer=answer,
        ),
        profile=llm_result.profile,
        provider=llm_result.provider,
        model=llm_result.model_requested,
        model_used=llm_result.model_used,
    )


@app.get("/health")
async def health():
    """Return the service health status."""
    return {"status": "ok"}


@app.get("/")
async def root():
    """Serve the frontend or return basic API information."""
    index_path = STATIC_DIR / "index.html"

    if index_path.exists():
        return FileResponse(index_path)

    return {
        "name": "Archivox",
        "message": (
            "Use POST /ask to ask questions "
            "about the transcript."
        ),
        "docs": "/docs",
        "health": "/health",
    }


if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )