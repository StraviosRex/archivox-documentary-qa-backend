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


def _query_terms(question: str) -> list[str]:
    """Extract useful terms and proper-name phrases from the question."""
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
        "was", "were", "is", "are", "did", "do", "does", "what", "who", "why",
        "how", "about", "he", "she", "it", "they", "his", "her", "their",
        "this", "that", "these", "those", "documentary", "transcript",
    }

    proper_phrases = re.findall(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        question,
    )

    terms = re.findall(r"[A-Za-z][A-Za-z0-9']+", question.lower())
    useful_terms = [
        term for term in terms
        if len(term) > 2 and term not in stopwords
    ]

    return proper_phrases + useful_terms


def make_excerpt(text: str, question: str, max_chars: int = 300) -> str:
    """
    Return a compact source excerpt centered around the most relevant query term.
    """
    cleaned = " ".join(text.split())

    if len(cleaned) <= max_chars:
        return cleaned

    lowered = cleaned.lower()
    anchor_index = -1

    for term in _query_terms(question):
        index = lowered.find(term.lower())

        if index != -1:
            anchor_index = index
            break

    if anchor_index == -1:
        return cleaned[:max_chars].rsplit(" ", 1)[0] + "..."

    half_window = max_chars // 2
    start = max(anchor_index - half_window, 0)
    end = min(start + max_chars, len(cleaned))

    start = max(end - max_chars, 0)

    excerpt = cleaned[start:end].strip()

    if start > 0:
        excerpt = excerpt.split(" ", 1)[-1]
        excerpt = "..." + excerpt

    if end < len(cleaned):
        excerpt = excerpt.rsplit(" ", 1)[0] + "..."

    return excerpt


def build_sources(chunks: list[dict], question: str) -> list[Source]:
    """Build deterministic source references from ranked retrieved chunks."""
    sources: list[Source] = []

    for chunk in chunks[: settings.sources_in_response]:
        timestamp = f'{chunk["start_timestamp"]}-{chunk["end_timestamp"]}'

        sources.append(
            Source(
                timestamp=timestamp,
                excerpt=make_excerpt(chunk["text"], question),
            )
        )

    return sources


def is_not_enough_info_answer(answer: str | None) -> bool:
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
        normalized.startswith("i don't have enough information")
        or "don't have enough information in the transcript" in normalized
        or "do not have enough information in the transcript" in normalized
    )


def answer_starts_with_timestamp(answer: str) -> bool:
    """Return True if the answer starts with a transcript-style timestamp phrase."""
    return bool(
        re.match(
            r"^\s*(timestamp:\s*)?(between|from|at)?\s*\[?\d{2}:\d{2}:\d{2}",
            answer.lower(),
        )
    )


def ensure_answer_has_timestamp(answer: str, chunks: list[dict]) -> str:
    """
    Ensure grounded answers start with at least one timestamp.

    Some LLMs place timestamps at the end or omit them. This keeps API output
    consistent without changing retrieval behavior.
    """
    if answer_starts_with_timestamp(answer):
        return answer

    if not chunks:
        return answer

    first_chunk = chunks[0]
    start = first_chunk.get("start_timestamp")
    end = first_chunk.get("end_timestamp")

    if not start or not end:
        return answer

    return f"Between {start} and {end}, {answer[0].lower() + answer[1:]}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Index the transcript on startup so the first query is fast."""
    logger.info("Indexing transcript...")
    get_collection()
    logger.info("Ready.")
    yield


app = FastAPI(
    title="Archivox",
    description=(
        "Documentary Q&A Backend. Ask questions and get answers grounded "
        "in transcript sources."
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


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Answer a natural-language question using retrieved transcript chunks."""
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retrieve(question)

    if not chunks:
        return AskResponse(
            answer="I don't have enough information in the transcript to answer that question.",
            sources=[],
        )

    messages = build_messages(question, chunks)

    try:
        llm_result = await call_llm(messages, profile_name=request.profile)
        answer = llm_result.content
    except Exception:
        logger.exception("LLM request failed.")
        raise HTTPException(status_code=502, detail="LLM request failed.")

    if not answer or not answer.strip():
        raise HTTPException(status_code=502, detail="LLM returned an empty answer.")

    if is_not_enough_info_answer(answer):
        return AskResponse(
            answer=answer,
            sources=[],
            profile=llm_result.profile,
            provider=llm_result.provider,
            model=llm_result.model_requested,
            model_used=llm_result.model_used,
        )

    answer = ensure_answer_has_timestamp(answer.strip(), chunks)

    return AskResponse(
        answer=answer,
        sources=build_sources(chunks, question),
        profile=llm_result.profile,
        provider=llm_result.provider,
        model=llm_result.model_requested,
        model_used=llm_result.model_used,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"

    if index_path.exists():
        return FileResponse(index_path)

    return {
        "name": "Archivox",
        "message": "Use POST /ask to ask questions about the transcript.",
        "docs": "/docs",
        "health": "/health",
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)