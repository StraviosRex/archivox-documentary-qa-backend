from sentence_transformers import SentenceTransformer

from app.config import settings


_model: SentenceTransformer | None = None
_model_name: str | None = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it across requests."""
    global _model, _model_name

    model_name = settings.embedding_model

    if _model is None or _model_name != model_name:
        _model = SentenceTransformer(model_name)
        _model_name = model_name

    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts and return vector representations."""
    if not texts:
        return []

    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    return embed_texts([query])[0]