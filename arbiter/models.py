"""Type-safe data models for the arbitration pipeline.

Every LLM in the system is constrained to emit one of these Pydantic models
via the `instructor` library, so the whole pipeline is type-safe end to end.
These same models double as the audit trail that gets persisted to SQLite/JSON
and the response schema that FastAPI publishes as OpenAPI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class Dimension(str, Enum):
    """The three evaluation dimensions, one per specialised critic."""

    ACCURACY = "accuracy"
    LOGIC = "logic"
    COMPLETENESS = "completeness"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def confidence_to_level(confidence: float) -> ConfidenceLevel:
    if confidence >= 0.75:
        return ConfidenceLevel.HIGH
    if confidence >= 0.5:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


SEVERITY_LABELS = {
    1: "trivial",
    2: "minor",
    3: "moderate",
    4: "serious",
    5: "critical",
}


def severity_label(severity: int) -> str:
    return SEVERITY_LABELS.get(int(severity), "moderate")


# ---------------------------------------------------------------------------
# Critic-level models (Phase 1)
# ---------------------------------------------------------------------------
class Issue(BaseModel):
    """A single problem a critic found in the output being evaluated."""

    quote: str = Field(
        ...,
        description="A verbatim quote from the original output that the issue refers to.",
    )
    problem: str = Field(
        ...,
        description="Clear description of what is wrong with the quoted text.",
    )
    severity: int = Field(
        ...,
        ge=1,
        le=5,
        description="How damaging the issue is: 1=trivial, 3=moderate, 5=critical.",
    )

    @property
    def severity_label(self) -> str:
        return severity_label(self.severity)


class Critique(BaseModel):
    """Structured critique returned by a single critic for a single dimension."""

    dimension: Dimension = Field(..., description="Which dimension this critique covers.")
    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="Overall quality on this dimension: 1=terrible, 5=flawless.",
    )
    issues: list[Issue] = Field(
        default_factory=list,
        description="Specific issues found, each anchored to a quote from the original.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The critic's confidence in its own assessment (0-1).",
    )
    summary: str = Field(
        default="",
        description="One or two sentences summarising the critic's overall take.",
    )

    @property
    def confidence_level(self) -> ConfidenceLevel:
        return confidence_to_level(self.confidence)


class CriticReport(BaseModel):
    """Wraps a critique with the operational metadata the orchestrator needs.

    When a critic's backend fails after all retries, `ok` is False, `critique`
    is None, and `error` explains the failure — this is how graceful
    degradation is represented in the audit trail.
    """

    dimension: Dimension
    backend: str = Field(..., description="Backend that served this critic, e.g. 'openai'.")
    model: str = Field(..., description="Concrete model name, e.g. 'gpt-4o'.")
    ok: bool = True
    critique: Optional[Critique] = None
    error: Optional[str] = None
    attempts: int = 1
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Disagreement models (Phase 2)
# ---------------------------------------------------------------------------
class DisagreementKind(str, Enum):
    SCORE_DIVERGENCE = "score_divergence"
    SEVERITY_CONFLICT = "severity_conflict"
    UNIQUE_FINDING = "unique_finding"
    EXISTENCE = "existence"


class Disagreement(BaseModel):
    """A detected conflict between critics that the adjudicator must resolve."""

    kind: DisagreementKind
    description: str
    dimensions: list[Dimension] = Field(default_factory=list)
    quote: Optional[str] = None
    severity_gap: Optional[int] = None


# ---------------------------------------------------------------------------
# Verdict models (Phase 3)
# ---------------------------------------------------------------------------
class ConfirmedIssue(BaseModel):
    """An issue the adjudicator upheld after weighing the critic evidence."""

    dimension: Dimension
    quote: str
    problem: str
    severity: int = Field(..., ge=1, le=5)
    evidence: str = Field(
        ...,
        description="The adjudicator's justification for upholding this issue.",
    )
    raised_by: list[Dimension] = Field(default_factory=list)

    @property
    def severity_label(self) -> str:
        return severity_label(self.severity)


class DismissedFlag(BaseModel):
    """An issue a critic raised but the adjudicator overruled, with reasoning."""

    dimension: Dimension
    quote: str
    problem: str
    raised_by: Dimension
    reasoning: str = Field(
        ...,
        description="Why the adjudicator overruled this flag.",
    )


class ValidatedClaim(BaseModel):
    """A claim in the output the adjudicator explicitly checked and endorsed."""

    quote: str
    note: str = Field(..., description="Why the claim is considered validated.")


class Verdict(BaseModel):
    """The adjudicator's final, synthesised judgement (Phase 3.3)."""

    quality_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Overall quality of the evaluated output, 1=unusable, 10=excellent.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Adjudicator confidence in the verdict (0-1).",
    )
    confirmed_issues: list[ConfirmedIssue] = Field(default_factory=list)
    dismissed_flags: list[DismissedFlag] = Field(default_factory=list)
    validated_claims: list[ValidatedClaim] = Field(
        default_factory=list,
        description="Claims the adjudicator explicitly checked and endorsed.",
    )
    summary: str = Field(..., description="One-paragraph assessment of the output.")

    @property
    def confidence_level(self) -> ConfidenceLevel:
        return confidence_to_level(self.confidence)

    @property
    def passed(self) -> bool:
        return self.quality_score >= 7 and not any(
            ci.severity >= 4 for ci in self.confirmed_issues
        )


# ---------------------------------------------------------------------------
# Top-level result / audit record
# ---------------------------------------------------------------------------
class ArbitrationResult(BaseModel):
    """The complete, persisted record of one arbitration run.

    This is what POST /v1/arbitrate returns and what GET
    /v1/arbitrations/{id} retrieves — a full audit trail of every critic
    report, every detected disagreement, and the final verdict.
    """

    id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    original_prompt: Optional[str] = None
    output_text: str
    reports: list[CriticReport] = Field(default_factory=list)
    disagreements: list[Disagreement] = Field(default_factory=list)
    verdict: Verdict
    short_circuited: bool = Field(
        default=False,
        description="True when all critics unanimously passed and the adjudicator was skipped.",
    )
    degraded: bool = Field(
        default=False,
        description="True when one or more critics failed and the verdict is based on the rest.",
    )
    degraded_dimensions: list[Dimension] = Field(default_factory=list)
    config: dict = Field(
        default_factory=dict,
        description="Which backend/model served each role, for reproducibility.",
    )

    # -- convenience accessors used by analytics / UI ----------------------
    def critique_for(self, dimension: Dimension) -> Optional[Critique]:
        for report in self.reports:
            if report.dimension == dimension and report.ok:
                return report.critique
        return None

    @property
    def num_issues_found(self) -> int:
        return sum(len(r.critique.issues) for r in self.reports if r.ok and r.critique)


# ---------------------------------------------------------------------------
# API request/response models (Phase 5)
# ---------------------------------------------------------------------------
class ArbitrateRequest(BaseModel):
    output: str = Field(..., description="The LLM-generated output to evaluate.")
    prompt: Optional[str] = Field(
        default=None,
        description="Optional original prompt/question the output was responding to.",
    )


class BatchArbitrateRequest(BaseModel):
    items: list[ArbitrateRequest] = Field(..., min_length=1)


class BatchArbitrateResponse(BaseModel):
    results: list[ArbitrationResult]
