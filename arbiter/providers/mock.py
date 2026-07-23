"""Deterministic offline backend.

Serves critiques from the heuristic engine and synthesises verdicts with a
transparent, rule-based adjudication policy. Because it is deterministic, it is
also what the test-suite runs against.
"""

from __future__ import annotations

from typing import Optional

from ..models import (
    ConfirmedIssue,
    Critique,
    Dimension,
    DismissedFlag,
    ValidatedClaim,
    Verdict,
)
from .base import Backend
from . import heuristics


def _norm(quote: str) -> str:
    return " ".join(quote.lower().split())


class MockBackend(Backend):
    """Rule-based backend used for offline runs and deterministic tests."""

    name = "mock"

    def __init__(self, model: str = "heuristic-v1"):
        super().__init__(model)

    def critique(
        self,
        *,
        dimension: Dimension,
        system_prompt: str,
        user_prompt: str,
        output_text: str,
        original_prompt: Optional[str],
    ) -> Critique:
        return heuristics.build_critique(dimension, output_text, original_prompt)

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
        ok_reports = [r for r in reports if r.ok and r.critique is not None]
        degraded = any(not r.ok for r in reports)

        # Collect and merge issues across critics (dedupe on overlapping quotes).
        merged: dict[str, dict] = {}
        for report in ok_reports:
            soft = heuristics.soft_quotes(report.dimension, output_text, original_prompt)
            for issue in report.critique.issues:
                key = _norm(issue.quote)
                # Merge with any existing overlapping quote.
                match_key = None
                for existing in merged:
                    if existing in key or key in existing:
                        match_key = existing
                        break
                entry = merged.get(match_key) if match_key else None
                is_soft = issue.quote in soft
                if entry:
                    entry["severity"] = max(entry["severity"], issue.severity)
                    entry["raised_by"].add(report.dimension)
                    entry["confidence"] = max(entry["confidence"], report.critique.confidence)
                    entry["soft"] = entry["soft"] and is_soft
                    entry["problem"] = entry["problem"] or issue.problem
                else:
                    merged[key] = {
                        "dimension": report.dimension,
                        "quote": issue.quote,
                        "problem": issue.problem,
                        "severity": issue.severity,
                        "raised_by": {report.dimension},
                        "confidence": report.critique.confidence,
                        "soft": is_soft,
                    }

        confirmed: list[ConfirmedIssue] = []
        dismissed: list[DismissedFlag] = []
        for entry in merged.values():
            uphold = entry["severity"] >= 3 or (
                entry["severity"] == 2 and entry["confidence"] >= 0.7 and not entry["soft"]
            )
            if uphold:
                corroborated = len(entry["raised_by"]) > 1
                confirmed.append(
                    ConfirmedIssue(
                        dimension=entry["dimension"],
                        quote=entry["quote"],
                        problem=entry["problem"],
                        severity=entry["severity"],
                        evidence=(
                            "Upheld: "
                            + ("corroborated by multiple critics. " if corroborated else "")
                            + f"severity {entry['severity']} with critic confidence "
                            f"{entry['confidence']:.2f}."
                        ),
                        raised_by=sorted(entry["raised_by"], key=lambda d: d.value),
                    )
                )
            else:
                dismissed.append(
                    DismissedFlag(
                        dimension=entry["dimension"],
                        quote=entry["quote"],
                        problem=entry["problem"],
                        raised_by=entry["dimension"],
                        reasoning=(
                            "Overruled: low-severity"
                            + (", heuristic/soft" if entry["soft"] else "")
                            + f" flag (severity {entry['severity']}, confidence "
                            f"{entry['confidence']:.2f}); insufficient evidence to uphold."
                        ),
                    )
                )

        # Quality score: blend the mean critic score with the *minimum* so a
        # single badly-failing dimension drags the verdict down (a shallow but
        # factually correct answer shouldn't score like a great one). Map the
        # 1-5 blend onto 1-10, then penalise confirmed severe issues.
        scores = [r.critique.score for r in ok_reports] or [1]
        blended = 0.6 * (sum(scores) / len(scores)) + 0.4 * min(scores)
        quality = (blended - 1) / 4 * 9 + 1
        quality -= sum(1.0 for ci in confirmed if ci.severity >= 4)
        quality -= sum(0.5 for ci in confirmed if ci.severity == 3)
        quality_score = int(max(1, min(10, round(quality))))

        # Confidence: mean critic confidence, adjusted for degradation/conflict.
        if ok_reports:
            mean_conf = sum(r.critique.confidence for r in ok_reports) / len(ok_reports)
        else:
            mean_conf = 0.4
        conf = mean_conf
        if degraded:
            conf -= 0.15
        if len(disagreements) > 2:
            conf -= 0.1
        confidence = round(max(0.3, min(0.97, conf)), 2)

        # Positively validate checkable claims the accuracy critic didn't flag.
        confirmed_quotes = {_norm(ci.quote) for ci in confirmed}
        validated = [
            ValidatedClaim(quote=quote, note=note)
            for quote, note in heuristics.detect_validated(output_text)
            if _norm(quote) not in confirmed_quotes
        ]

        summary = self._summary(
            quality_score, confirmed, dismissed, disagreements, degraded, reports
        )
        return Verdict(
            quality_score=quality_score,
            confidence=confidence,
            confirmed_issues=confirmed,
            dismissed_flags=dismissed,
            validated_claims=validated,
            summary=summary,
        )

    @staticmethod
    def _summary(quality, confirmed, dismissed, disagreements, degraded, reports) -> str:
        verdict_word = (
            "excellent" if quality >= 9 else
            "solid" if quality >= 7 else
            "mixed" if quality >= 5 else
            "weak" if quality >= 3 else
            "poor"
        )
        bits = [
            f"Overall this output is {verdict_word} (quality {quality}/10)."
        ]
        if confirmed:
            worst = max(confirmed, key=lambda c: c.severity)
            dims = sorted({c.dimension.value for c in confirmed})
            bits.append(
                f"The adjudicator upheld {len(confirmed)} issue(s) across "
                f"{', '.join(dims)}, the most serious being a "
                f"severity-{worst.severity} {worst.dimension.value} problem."
            )
        else:
            bits.append("No issues survived adjudication.")
        if dismissed:
            bits.append(
                f"{len(dismissed)} critic flag(s) were overruled as low-confidence or soft."
            )
        if disagreements:
            bits.append(
                f"The critics disagreed in {len(disagreements)} place(s), which the "
                "adjudicator resolved on the evidence."
            )
        if degraded:
            failed = [r.dimension.value for r in reports if not r.ok]
            bits.append(
                f"Note: the {', '.join(failed)} critic(s) failed, so confidence is reduced "
                "and that dimension is not fully covered."
            )
        return " ".join(bits)
