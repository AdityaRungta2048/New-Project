"""Real LLM backends: OpenAI, Anthropic, and Ollama.

All three use the `instructor` library to coerce the model into returning one of
our Pydantic models directly (Phase 1.2 — "Use the instructor library to enforce
structured outputs from every model"). Imports are lazy so the package works
even when these optional dependencies aren't installed; in that case the backend
reports itself unavailable and the registry falls back to the mock backend under
`ARBITER_BACKEND_MODE=auto`.
"""

from __future__ import annotations

import socket
from typing import Optional
from urllib.parse import urlparse

from ..models import Critique, Dimension, Verdict
from .base import Backend, BackendError


class _InstructorBackend(Backend):
    """Common structured-output plumbing shared by the real backends."""

    def _client(self):  # pragma: no cover - exercised only with real creds
        raise NotImplementedError

    def _create(self, client, *, system: str, user: str, response_model, max_retries: int):
        """Provider-specific instructor call. Returns a validated response_model."""
        raise NotImplementedError  # pragma: no cover

    def critique(
        self,
        *,
        dimension: Dimension,
        system_prompt: str,
        user_prompt: str,
        output_text: str,
        original_prompt: Optional[str],
    ) -> Critique:
        try:
            client = self._client()
            result = self._create(
                client,
                system=system_prompt,
                user=user_prompt,
                response_model=Critique,
                max_retries=1,
            )
        except Exception as exc:  # normalise every provider error
            raise BackendError(f"{self.name} critique failed: {exc}") from exc
        # Force the dimension to match the critic's role regardless of model drift.
        result.dimension = dimension
        return result

    def adjudicate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        output_text: str,
        original_prompt: Optional[str],
        reports: list,
        disagreements: list,
    ) -> Verdict:
        try:
            client = self._client()
            return self._create(
                client,
                system=system_prompt,
                user=user_prompt,
                response_model=Verdict,
                max_retries=1,
            )
        except Exception as exc:
            raise BackendError(f"{self.name} adjudication failed: {exc}") from exc


class OpenAIBackend(_InstructorBackend):
    name = "openai"

    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        self.api_key = api_key

    @property
    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import instructor  # noqa: F401
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def _client(self):
        import instructor
        from openai import OpenAI

        return instructor.from_openai(OpenAI(api_key=self.api_key))

    def _create(self, client, *, system, user, response_model, max_retries):
        return client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_model=response_model,
            max_retries=max_retries,
            temperature=0,
        )


class AnthropicBackend(_InstructorBackend):
    name = "anthropic"

    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        self.api_key = api_key

    @property
    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import instructor  # noqa: F401
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def _client(self):
        import instructor
        from anthropic import Anthropic

        return instructor.from_anthropic(Anthropic(api_key=self.api_key))

    def _create(self, client, *, system, user, response_model, max_retries):
        return client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user}],
            response_model=response_model,
            max_retries=max_retries,
        )


class OllamaBackend(_InstructorBackend):
    """Local Llama (or any Ollama model) via its OpenAI-compatible endpoint."""

    name = "ollama"

    def __init__(self, model: str, host: str):
        super().__init__(model)
        self.host = host.rstrip("/")

    @property
    def available(self) -> bool:
        try:
            import instructor  # noqa: F401
            import openai  # noqa: F401
        except ImportError:
            return False
        return self._reachable()

    def _reachable(self, timeout: float = 0.4) -> bool:
        parsed = urlparse(self.host)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 11434)
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def _client(self):
        import instructor
        from openai import OpenAI

        return instructor.from_openai(
            OpenAI(base_url=f"{self.host}/v1", api_key="ollama"),
            mode=instructor.Mode.JSON,
        )

    def _create(self, client, *, system, user, response_model, max_retries):
        return client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_model=response_model,
            max_retries=max_retries,
            temperature=0,
        )
