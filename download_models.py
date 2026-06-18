"""
Run once after installing requirements to cache model weights locally.

    python download_models.py
"""

from pathlib import Path

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

MODELS_DIR = Path(__file__).parent / "models"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def download_embedding_model() -> None:
    print(f"Downloading embedding model: {EMBEDDING_MODEL}")
    SentenceTransformer(EMBEDDING_MODEL)
    print("Done.\n")


def download_cross_encoder() -> None:
    local_dir = MODELS_DIR / "cross-encoder" / "ms-marco-MiniLM-L-6-v2"
    print(f"Downloading cross-encoder model to {local_dir}")
    snapshot_download(
        repo_id=CROSS_ENCODER_MODEL,
        local_dir=str(local_dir),
    )
    print("Done.\n")


if __name__ == "__main__":
    download_embedding_model()
    download_cross_encoder()
    print("All models ready.")
