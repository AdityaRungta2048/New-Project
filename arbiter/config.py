"""Runtime configuration, driven entirely by environment variables.

The defaults are chosen so that a fresh checkout with **no** API keys runs the
whole pipeline against the deterministic `mock` backend. Set the relevant
`*_BACKEND` vars (and credentials) to route critics through real models — the
spec's recommended routing is accuracy->GPT-4o, logic->Claude,
completeness->local Llama.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .models import Dimension


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class BackendMode:
    AUTO = "auto"      # use real backend if available, else silently fall back to mock
    STRICT = "strict"  # raise if a configured real backend is unavailable


@dataclass
class Settings:
    """Immutable snapshot of the arbiter's runtime configuration."""

    # Which backend serves each role.
    accuracy_backend: str = field(default_factory=lambda: _env("ARBITER_ACCURACY_BACKEND", "mock"))
    logic_backend: str = field(default_factory=lambda: _env("ARBITER_LOGIC_BACKEND", "mock"))
    completeness_backend: str = field(
        default_factory=lambda: _env("ARBITER_COMPLETENESS_BACKEND", "mock")
    )
    adjudicator_backend: str = field(
        default_factory=lambda: _env("ARBITER_ADJUDICATOR_BACKEND", "mock")
    )

    backend_mode: str = field(default_factory=lambda: _env("ARBITER_BACKEND_MODE", "auto"))

    # Model names.
    openai_model: str = field(default_factory=lambda: _env("ARBITER_OPENAI_MODEL", "gpt-4o"))
    anthropic_model: str = field(
        default_factory=lambda: _env("ARBITER_ANTHROPIC_MODEL", "claude-sonnet-5")
    )
    ollama_model: str = field(default_factory=lambda: _env("ARBITER_OLLAMA_MODEL", "llama3.1"))

    # Credentials / hosts.
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))

    # Reliability.
    max_retries: int = field(default_factory=lambda: int(_env("ARBITER_MAX_RETRIES", "2")))
    retry_backoff: float = field(default_factory=lambda: float(_env("ARBITER_RETRY_BACKOFF", "1.5")))

    # Storage.
    db_path: str = field(default_factory=lambda: _env("ARBITER_DB_PATH", "data/arbitrations.sqlite"))
    json_dir: str = field(default_factory=lambda: _env("ARBITER_JSON_DIR", "data/arbitrations"))

    def backend_for(self, dimension: Dimension) -> str:
        return {
            Dimension.ACCURACY: self.accuracy_backend,
            Dimension.LOGIC: self.logic_backend,
            Dimension.COMPLETENESS: self.completeness_backend,
        }[dimension]

    def routing_summary(self) -> dict:
        """Human-readable record of the active routing, stored with each result."""
        return {
            "accuracy": self.accuracy_backend,
            "logic": self.logic_backend,
            "completeness": self.completeness_backend,
            "adjudicator": self.adjudicator_backend,
            "backend_mode": self.backend_mode,
        }


_settings: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    """Return a process-wide `Settings` singleton (re-read env with refresh=True)."""
    global _settings
    if _settings is None or refresh:
        _settings = Settings()
    return _settings
