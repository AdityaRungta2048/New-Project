"""The three specialised critic agents (Phase 1.1 / 1.3).

Each critic is bound to a dimension and routed to its own backend — the spec's
recommended routing is accuracy->GPT-4o, logic->Claude, completeness->local
Llama. The point of using different models is that their disagreements are the
most valuable signal; shared models share blind spots.

Each critic owns its own reliability: it retries with exponential backoff and,
if it still fails, returns a `CriticReport` with ``ok=False`` so the graph can
degrade gracefully instead of crashing (Phase 2.4).
"""

from __future__ import annotations

import time
from typing import Optional

from .config import Settings, get_settings
from .models import CriticReport, Dimension
from .prompts import CRITIC_SYSTEM_PROMPTS, render_critic_user_prompt
from .providers import get_backend


class Critic:
    """A single-dimension critic bound to a backend."""

    def __init__(self, dimension: Dimension, settings: Settings | None = None):
        self.dimension = dimension
        self.settings = settings or get_settings()
        self.backend = get_backend(self.settings.backend_for(dimension), self.settings)

    def run(self, output_text: str, original_prompt: Optional[str]) -> CriticReport:
        system_prompt = CRITIC_SYSTEM_PROMPTS[self.dimension]
        user_prompt = render_critic_user_prompt(self.dimension, output_text, original_prompt)

        attempts = 0
        last_error: Optional[str] = None
        start = time.perf_counter()
        max_attempts = max(1, self.settings.max_retries + 1)

        while attempts < max_attempts:
            attempts += 1
            try:
                critique = self.backend.critique(
                    dimension=self.dimension,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_text=output_text,
                    original_prompt=original_prompt,
                )
                return CriticReport(
                    dimension=self.dimension,
                    backend=self.backend.name,
                    model=self.backend.model,
                    ok=True,
                    critique=critique,
                    attempts=attempts,
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
            except Exception as exc:  # noqa: BLE001 - deliberately broad; report, don't crash
                last_error = str(exc)
                if attempts < max_attempts:
                    time.sleep(self.settings.retry_backoff ** attempts * 0.01)

        # Graceful degradation: report the failure without a critique.
        return CriticReport(
            dimension=self.dimension,
            backend=self.backend.name,
            model=self.backend.model,
            ok=False,
            critique=None,
            error=last_error or "unknown error",
            attempts=attempts,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def build_critics(settings: Settings | None = None) -> dict[Dimension, Critic]:
    settings = settings or get_settings()
    return {dim: Critic(dim, settings) for dim in Dimension}
