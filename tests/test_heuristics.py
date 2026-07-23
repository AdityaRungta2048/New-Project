"""The heuristic engine must find the right kind of issue per dimension."""

from arbiter.models import Dimension
from arbiter.providers.heuristics import (
    build_critique,
    detect_accuracy,
    detect_completeness,
    detect_logic,
    detect_validated,
)


def test_accuracy_detects_geocentric_error():
    findings = detect_accuracy("The sun revolves around the earth every day.")
    assert findings
    assert any("orbit" in f.problem.lower() or "geocentric" in f.problem.lower() for f in findings)
    assert max(f.severity for f in findings) >= 4


def test_accuracy_detects_wrong_numeric_fact():
    findings = detect_accuracy("Water boils at 50 degrees celsius.")
    assert any("100" in f.problem for f in findings)


def test_accuracy_clean_on_true_statement():
    assert detect_accuracy("The Earth orbits the Sun once per year.") == []


def test_logic_detects_fallacies():
    findings = detect_logic("Everyone knows this is right. Anyone who disagrees is a fool.")
    problems = " ".join(f.problem.lower() for f in findings)
    assert "popularity" in problems
    assert "ad hominem" in problems


def test_logic_slippery_slope():
    findings = detect_logic("If we allow this, then everything will inevitably collapse.")
    assert any("slippery" in f.problem.lower() for f in findings)


def test_completeness_detects_coverage_gap():
    findings = detect_completeness(
        "Both were big wars. etc.",
        original_prompt="Compare the causes, course, and consequences of the two world wars "
        "and explain their key economic and political differences.",
    )
    assert findings
    assert any("coverage" in f.problem.lower() or "brief" in f.problem.lower() for f in findings)


def test_validated_claims_detected():
    validated = detect_validated("Paris is the capital of France.")
    assert validated
    assert "paris" in validated[0][0].lower()


def test_build_critique_clean_scores_five():
    crit = build_critique(Dimension.LOGIC, "The sky is blue because of Rayleigh scattering.", None)
    assert crit.score == 5
    assert crit.issues == []
    assert crit.dimension is Dimension.LOGIC


def test_build_critique_dirty_scores_low():
    crit = build_critique(
        Dimension.ACCURACY, "The sun revolves around the earth and has 106 bones.", None
    )
    assert crit.score <= 2
    assert crit.issues
