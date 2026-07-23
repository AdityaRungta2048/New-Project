"""Backend factory + auto/strict fallback logic."""

from __future__ import annotations

from ..config import Settings, get_settings
from .base import Backend
from .llm import AnthropicBackend, OllamaBackend, OpenAIBackend
from .mock import MockBackend


def _construct(name: str, settings: Settings) -> Backend:
    name = (name or "mock").lower()
    if name == "openai":
        return OpenAIBackend(settings.openai_model, settings.openai_api_key)
    if name == "anthropic":
        return AnthropicBackend(settings.anthropic_model, settings.anthropic_api_key)
    if name == "ollama":
        return OllamaBackend(settings.ollama_model, settings.ollama_host)
    if name == "mock":
        return MockBackend()
    raise ValueError(f"Unknown backend '{name}'. Valid: openai|anthropic|ollama|mock.")


def get_backend(name: str, settings: Settings | None = None) -> Backend:
    """Return the requested backend, honouring ARBITER_BACKEND_MODE.

    - ``auto`` (default): if the requested real backend isn't usable (missing
      key/lib/host), transparently fall back to the deterministic mock backend
      so the pipeline always runs.
    - ``strict``: raise if the requested real backend is unavailable.
    """
    settings = settings or get_settings()
    backend = _construct(name, settings)
    if backend.name == "mock" or backend.available:
        return backend

    if settings.backend_mode == "strict":
        raise RuntimeError(
            f"Backend '{name}' is configured but unavailable (missing API key, "
            f"library, or host). Set ARBITER_BACKEND_MODE=auto to fall back to mock."
        )
    return MockBackend()


def describe_backend(name: str, settings: Settings | None = None) -> dict:
    """Report the effective backend + model for a configured name (for /health)."""
    settings = settings or get_settings()
    requested = _construct(name, settings)
    effective = get_backend(name, settings)
    return {
        "requested": requested.name,
        "requested_model": requested.model,
        "effective": effective.name,
        "effective_model": effective.model,
        "available": requested.name == "mock" or requested.available,
    }
