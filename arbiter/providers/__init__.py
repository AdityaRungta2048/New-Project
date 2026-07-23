"""LLM backend abstraction.

A *backend* is a thin adapter over one LLM provider that can produce a
structured `Critique` or `Verdict`. Real backends (OpenAI, Anthropic, Ollama)
use the `instructor` library to enforce structured output; the `mock` backend
uses a deterministic heuristic engine so the entire pipeline runs offline with
no API keys.
"""

from .base import Backend, BackendError
from .registry import get_backend, describe_backend

__all__ = ["Backend", "BackendError", "get_backend", "describe_backend"]
