import asyncio
from pathlib import Path
from typing import Any

import httpx
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
MODELS_YAML = CONFIG_DIR / "models.yaml"
RETRIEVAL_YAML = CONFIG_DIR / "retrieval.yaml"


class Settings(BaseSettings):
    # Default profile
    llm_profile: str = Field(default="groq_llama8b")

    # Optional manual overrides
    llm_provider: str = Field(default="")
    llm_model: str = Field(default="")
    llm_base_url: str = Field(default="")

    # API keys
    openrouter_api_key: str = Field(default="")
    groq_api_key: str = Field(default="")
    gemini_api_key: str = Field(default="")

    # Ollama — URL differs by deployment; set in .env
    ollama_base_url: str = Field(default="http://localhost:11434/v1")

    # Retrieval
    top_k: int = Field(default=5)
    sources_in_response: int = Field(default=3)
    similarity_threshold: float = Field(default=0.48)
    reranking_enabled: bool = Field(default=True)

    # Embedding/chunking
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    chunk_window_size: int = Field(default=3)
    chunk_overlap: int = Field(default=1)

    # Paths
    transcript_path: str = Field(default="data/transcript.txt")
    chroma_persist_dir: str = Field(default="chroma_db")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()


def load_model_profiles() -> dict[str, dict[str, Any]]:
    """Load all approved LLM profiles from config/models.yaml."""
    if not MODELS_YAML.exists():
        raise FileNotFoundError(f"Model config not found: {MODELS_YAML}")

    with open(MODELS_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    profiles = data.get("profiles", {})

    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("config/models.yaml must contain a non-empty 'profiles' mapping.")

    return profiles


def load_retrieval_config() -> dict[str, Any]:
    """Load corpus-specific retrieval hints from config/retrieval.yaml."""
    if not RETRIEVAL_YAML.exists():
        return {}

    with open(RETRIEVAL_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("config/retrieval.yaml must contain a mapping.")

    return data


def _resolve_base_url(profile: dict[str, Any]) -> str:
    """Resolve a profile's base URL, expanding base_url_env references via settings."""
    env_key = profile.get("base_url_env", "")
    if env_key:
        return getattr(settings, env_key.lower(), "") or settings.ollama_base_url
    return profile.get("base_url", "")


async def _fetch_ollama_model_ids(base_url: str) -> list[str]:
    """Return model IDs from an Ollama instance, or [] if unreachable.

    Filters out :latest aliases when a versioned tag for the same base model exists.
    """
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{base_url}/models")
            resp.raise_for_status()
            data = resp.json()
            all_ids = list(dict.fromkeys(m["id"] for m in data.get("data", [])))
            base_names = {mid.split(":")[0] for mid in all_ids if ":" in mid and not mid.endswith(":latest")}
            return [mid for mid in all_ids if not (mid.endswith(":latest") and mid.split(":")[0] in base_names)]
    except Exception:
        return []


async def list_public_profiles() -> list[dict[str, str]]:
    """Return frontend-safe profile metadata. Ollama entries are discovered live."""
    profiles = load_model_profiles()

    ollama_entries = [
        (pid, p) for pid, p in profiles.items() if p.get("provider") == "ollama"
    ]

    discovered: dict[str, list[str]] = {}
    if ollama_entries:
        model_lists = await asyncio.gather(
            *[_fetch_ollama_model_ids(_resolve_base_url(p)) for _, p in ollama_entries]
        )
        for (pid, _), model_ids in zip(ollama_entries, model_lists):
            discovered[pid] = model_ids

    result = []
    for profile_id, profile in profiles.items():
        provider = profile.get("provider", "")

        if provider == "ollama":
            label_prefix = profile.get("label", profile_id)
            for model_id in discovered.get(profile_id, []):
                result.append({
                    "id": f"{profile_id}::{model_id}",
                    "label": f"{label_prefix} - {model_id}",
                    "provider": "ollama",
                    "model": model_id,
                })
            continue

        result.append({
            "id": profile_id,
            "label": profile.get("label", profile_id),
            "provider": provider,
            "model": profile.get("model", ""),
        })

    return result


def resolve_llm_profile(profile_name: str | None = None) -> dict[str, Any]:
    """
    Resolve selected LLM profile.

    Handles two forms:
    - Named yaml profiles (e.g. "groq_llama8b")
    - Dynamic Ollama profiles (e.g. "ollama::llama3.2:3b") discovered at runtime

    Manual overrides (LLM_PROVIDER, LLM_MODEL, LLM_BASE_URL) apply only to the
    default .env-selected profile, not per-request frontend selections.
    """
    selected_profile = profile_name or settings.llm_profile

    # Dynamic Ollama profiles: "ollama::llama3.2:3b", "ollama::gemma4:27b", etc.
    if "::" in selected_profile:
        profile_key, model = selected_profile.split("::", 1)
        profiles = load_model_profiles()
        ollama_profile = profiles.get(profile_key, {})
        base_url = _resolve_base_url(ollama_profile)
        label_prefix = ollama_profile.get("label", profile_key)
        return {
            "profile": selected_profile,
            "label": f"{label_prefix} - {model}",
            "provider": "ollama",
            "model": model,
            "base_url": base_url,
            "api_key": "",
            "fallback_models": [],
        }

    profiles = load_model_profiles()

    if selected_profile not in profiles:
        available = ", ".join(profiles.keys())
        raise ValueError(
            f"Unknown LLM profile '{selected_profile}'. Available profiles: {available}"
        )

    profile = profiles[selected_profile]

    provider = profile["provider"]
    model = profile["model"]
    base_url = _resolve_base_url(profile)

    # Manual overrides only apply when no explicit frontend profile is selected.
    if profile_name is None:
        provider = settings.llm_provider or provider
        model = settings.llm_model or model
        base_url = settings.llm_base_url or base_url

    api_key_env = profile.get("api_key_env")
    api_key = ""

    if api_key_env:
        api_key = getattr(settings, api_key_env.lower(), "")

    return {
        "profile": selected_profile,
        "label": profile.get("label", selected_profile),
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "fallback_models": profile.get("fallback_models", []),
    }
