"""Disagreement detector + short-circuit condition."""

from arbiter.disagreement import detect_disagreements, is_unanimous_pass
from arbiter.models import Critique, CriticReport, Dimension, DisagreementKind, Issue


def _report(dim, score, confidence, issues=()):
    return CriticReport(
        dimension=dim,
        backend="mock",
        model="m",
        ok=True,
        critique=Critique(
            dimension=dim, score=score, confidence=confidence, issues=list(issues)
        ),
    )


def test_score_divergence_flagged():
    reports = [
        _report(Dimension.ACCURACY, 1, 0.9, [Issue(quote="x", problem="p", severity=5)]),
        _report(Dimension.LOGIC, 5, 0.9),
        _report(Dimension.COMPLETENESS, 5, 0.9),
    ]
    kinds = {d.kind for d in detect_disagreements(reports)}
    assert DisagreementKind.SCORE_DIVERGENCE in kinds
    assert DisagreementKind.EXISTENCE in kinds  # one clean, one serious


def test_unique_finding_flagged():
    reports = [
        _report(Dimension.ACCURACY, 3, 0.8, [Issue(quote="unique bad claim here", problem="wrong", severity=4)]),
        _report(Dimension.LOGIC, 4, 0.8),
        _report(Dimension.COMPLETENESS, 4, 0.8),
    ]
    dis = detect_disagreements(reports)
    assert any(d.kind is DisagreementKind.UNIQUE_FINDING for d in dis)


def test_severity_conflict_on_overlapping_quote():
    shared = "the model claims the moon is made of cheese"
    reports = [
        _report(Dimension.ACCURACY, 2, 0.8, [Issue(quote=shared, problem="false", severity=5)]),
        _report(Dimension.LOGIC, 4, 0.8, [Issue(quote=shared, problem="minor", severity=1)]),
        _report(Dimension.COMPLETENESS, 4, 0.8),
    ]
    dis = detect_disagreements(reports)
    assert any(d.kind is DisagreementKind.SEVERITY_CONFLICT for d in dis)


def test_no_disagreement_when_aligned():
    reports = [
        _report(Dimension.ACCURACY, 4, 0.8),
        _report(Dimension.LOGIC, 4, 0.8),
        _report(Dimension.COMPLETENESS, 5, 0.8),
    ]
    assert detect_disagreements(reports) == []


def test_unanimous_pass_detection():
    reports = [_report(d, 5, 0.9) for d in Dimension]
    assert is_unanimous_pass(reports) is True

    reports[0].critique.score = 4
    assert is_unanimous_pass(reports) is False


def test_unanimous_pass_false_when_degraded():
    reports = [_report(d, 5, 0.9) for d in Dimension]
    reports[0].ok = False
    reports[0].critique = None
    assert is_unanimous_pass(reports) is False
