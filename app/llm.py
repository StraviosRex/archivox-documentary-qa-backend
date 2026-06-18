import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import resolve_llm_profile

logger = logging.getLogger(__name__)

#This here is to prevent submitting the text as an API, in case of a misconfiguration
PLACEHOLDER_KEYS = {
    "",
    "your_api_key_here",
    "your_openrouter_api_key_here",
    "your_groq_api_key_here",
    "your_gemini_api_key_here",
}


@dataclass
class LLMResult:
    content: str
    profile: str
    provider: str
    model_requested: str
    model_used: str


def _valid_api_key(api_key: str | None) -> bool:
    """Return True only when the API key is non-empty and not a template value."""
    return bool(api_key and api_key.strip() not in PLACEHOLDER_KEYS)


def _dedupe_models(models: list[str]) -> list[str]:
    """Remove duplicate model IDs while preserving order."""
    seen: set[str] = set()
    deduped: list[str] = []

    for model in models:
        if not model:
            continue

        if model in seen:
            continue

        seen.add(model)
        deduped.append(model)

    return deduped


async def call_llm(messages: list[dict], profile_name: str | None = None) -> LLMResult:
    """
    Send messages to the selected LLM profile.

    profile_name comes from the frontend dropdown.
    If missing, the default LLM_PROFILE from .env is used.
    """
    profile = resolve_llm_profile(profile_name)

    selected_profile = profile["profile"]
    provider = profile["provider"]
    model = profile["model"]
    base_url = profile["base_url"]
    api_key = profile["api_key"]
    fallback_models = profile.get("fallback_models") or []

    if not base_url:
        raise RuntimeError("LLM base URL is not configured.")

    if not model:
        raise RuntimeError("LLM model is not configured.")

    if provider != "ollama" and not _valid_api_key(api_key):
        raise RuntimeError(
            f"API key is missing for provider '{provider}'. "
            "Set the correct provider API key in your .env file."
        )

    url = f"{base_url.rstrip('/')}/chat/completions"

    headers = {
        "Content-Type": "application/json",
    }

    if _valid_api_key(api_key):
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "messages": messages,
        "temperature": 0.0,
    }

    if provider == "openrouter":
        payload["max_completion_tokens"] = 1024

        route_models = _dedupe_models([model] + fallback_models)

        if len(route_models) > 1:
            # OpenRouter controlled fallback routing.
            # The primary model is first, then fallback models.
            payload["models"] = route_models
        else:
            payload["model"] = model

        logger.info(
            "Calling LLM profile=%s provider=%s route_models=%s",
            selected_profile,
            provider,
            route_models,
        )
    else:
        payload["model"] = model
        payload["max_tokens"] = 1024

        logger.info(
            "Calling LLM profile=%s provider=%s model=%s",
            selected_profile,
            provider,
            model,
        )

    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(url, json=payload, headers=headers)

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "LLM request failed. profile=%s provider=%s model=%s status=%s body=%s",
            selected_profile,
            provider,
            model,
            exc.response.status_code,
            exc.response.text[:1500],
        )
        raise

    data = response.json()

    try:
        message = data["choices"][0]["message"]
        content = message.get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise RuntimeError(f"Unexpected LLM response format: {data}") from exc

    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"LLM returned empty content: {data}")

    model_used = data.get("model") or model

    logger.info(
        "LLM response received. profile=%s provider=%s requested_model=%s used_model=%s",
        selected_profile,
        provider,
        model,
        model_used,
    )

    return LLMResult(
        content=content.strip(),
        profile=selected_profile,
        provider=provider,
        model_requested=model,
        model_used=model_used,
    )