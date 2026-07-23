"""Backend interface shared by every provider."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..models import Critique, Dimension, Verdict


class BackendError(RuntimeError):
    """Raised when a backend cannot produce a structured result."""


class Backend(ABC):
    """Produces structured critiques / verdicts from a single LLM provider.

    Both `critique()` and `adjudicate()` receive the natural-language prompt
    (used by real LLM backends) *and* the raw structured inputs (used by the
    mock backend). This lets any backend be swapped in without touching the
    critic or adjudicator code.
    """

    #: short identifier, e.g. "openai"
    name: str = "base"

    def __init__(self, model: str):
        self.model = model

    @property
    def available(self) -> bool:
        """Whether this backend can actually serve requests right now."""
        return True

    @abstractmethod
    def critique(
        self,
        *,
        dimension: Dimension,
        system_prompt: str,
        user_prompt: str,
        output_text: str,
        original_prompt: Optional[str],
    ) -> Critique:
        ...

    @abstractmethod
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
        ...
