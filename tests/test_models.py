"""Model-level invariants and serialisation."""

import pytest
from pydantic import ValidationError

from arbiter.models import (
    ArbitrationResult,
    ConfidenceLevel,
    Critique,
    CriticReport,
    Dimension,
    Issue,
    Verdict,
    confidence_to_level,
    severity_label,
)


def test_issue_severity_bounds():
    Issue(quote="q", problem="p", severity=1)
    Issue(quote="q", problem="p", severity=5)
    with pytest.raises(ValidationError):
        Issue(quote="q", problem="p", severity=6)
    with pytest.raises(ValidationError):
        Issue(quote="q", problem="p", severity=0)


def test_critique_score_and_confidence_bounds():
    with pytest.raises(ValidationError):
        Critique(dimension=Dimension.ACCURACY, score=7, confidence=0.5)
    with pytest.raises(ValidationError):
        Critique(dimension=Dimension.ACCURACY, score=3, confidence=1.5)


def test_confidence_level_mapping():
    assert confidence_to_level(0.9) is ConfidenceLevel.HIGH
    assert confidence_to_level(0.6) is ConfidenceLevel.MEDIUM
    assert confidence_to_level(0.2) is ConfidenceLevel.LOW


def test_severity_labels():
    assert severity_label(1) == "trivial"
    assert severity_label(5) == "critical"


def test_verdict_passed_property():
    good = Verdict(quality_score=9, confidence=0.9, summary="s")
    assert good.passed is True
    bad = Verdict(quality_score=4, confidence=0.9, summary="s")
    assert bad.passed is False


def test_arbitration_result_roundtrip():
    result = ArbitrationResult(
        id="abc",
        output_text="hello",
        reports=[
            CriticReport(
                dimension=Dimension.ACCURACY,
                backend="mock",
                model="m",
                ok=True,
                critique=Critique(
                    dimension=Dimension.ACCURACY, score=5, confidence=0.9, issues=[]
                ),
            )
        ],
        verdict=Verdict(quality_score=8, confidence=0.8, summary="s"),
    )
    dumped = result.model_dump_json()
    loaded = ArbitrationResult.model_validate_json(dumped)
    assert loaded.id == "abc"
    assert loaded.critique_for(Dimension.ACCURACY).score == 5
    assert loaded.num_issues_found == 0
