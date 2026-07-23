"""The adjudicator agent (Phase 3).

The adjudicator receives the original output, all three critic reports, and the
list of detected disagreements. Its job is to weigh the evidence, reason through
each disagreement, resolve conflicts, and emit a single structured `Verdict`
(quality 1-10, confidence, confirmed issues, dismissed flags, one-paragraph
summary).

Two shortcuts live here too:
- ``high_confidence_pass`` builds the verdict for the unanimous-pass
  short-circuit (Phase 2.4) without calling a model at all.
- If the adjudicator backend fails, we fall back to a deterministic mock
  synthesis so the pipeline still returns a verdict.
"""

from __future__ import annotations

import time
from typing import Optional

from .config import Settings, get_settings
from .models import (
    ConfirmedIssue,
    CriticReport,
    Dimension,
    Disagreement,
    Verdict,
)
from .prompts import ADJUDICATOR_SYSTEM_PROMPT, render_adjudicator_user_prompt
from .providers import get_backend
from .providers.mock import MockBackend


class Adjudicator:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.backend = get_backend(self.settings.adjudicator_backend, self.settings)

    def run(
        self,
        output_text: str,
        original_prompt: Optional[str],
        reports: list[CriticReport],
        disagreements: list[Disagreement],
    ) -> Verdict:
        system_prompt = ADJUDICATOR_SYSTEM_PROMPT
        user_prompt = render_adjudicator_user_prompt(
            output_text, original_prompt, reports, disagreements
        )
        max_attempts = max(1, self.settings.max_retries + 1)
        last_error: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            try:
                return self.backend.adjudicate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    output_text=output_text,
                    original_prompt=original_prompt,
                    reports=reports,
                    disagreements=disagreements,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < max_attempts:
                    time.sleep(self.settings.retry_backoff ** attempt * 0.01)

        # Last-resort fallback: deterministic synthesis so we always return a
        # verdict rather than crashing the whole arbitration.
        fallback = MockBackend().adjudicate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_text=output_text,
            original_prompt=original_prompt,
            reports=reports,
            disagreements=disagreements,
        )
        fallback.summary = (
            f"[adjudicator backend failed: {last_error}; verdict synthesised from critic "
            f"reports] " + fallback.summary
        )
        fallback.confidence = round(max(0.3, fallback.confidence - 0.1), 2)
        return fallback


def high_confidence_pass(reports: list[CriticReport]) -> Verdict:
    """Verdict for the unanimous-pass short-circuit — no model call needed."""
    ok = [r for r in reports if r.ok and r.critique is not None]
    confidence = round(
        min(0.98, sum(r.critique.confidence for r in ok) / max(1, len(ok)) + 0.05), 2
    )
    dims = ", ".join(sorted(r.dimension.value for r in ok))
    return Verdict(
        quality_score=10,
        confidence=confidence,
        confirmed_issues=[],
        dismissed_flags=[],
        summary=(
            f"All critics ({dims}) independently returned a flawless, issue-free "
            "assessment, so the adjudicator was short-circuited and the output receives "
            "a high-confidence clean pass."
        ),
    )


def degraded_note(reports: list[CriticReport]) -> tuple[bool, list[Dimension]]:
    """Return (degraded, failed_dimensions) for the audit record."""
    failed = [r.dimension for r in reports if not r.ok]
    return (bool(failed), failed)


# Re-exported for callers that want the confirmed-issue type without importing models.
__all__ = ["Adjudicator", "high_confidence_pass", "degraded_note", "ConfirmedIssue"]
