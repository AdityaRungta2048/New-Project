"""End-to-end orchestration: graph + plain paths, degradation, persistence."""

import pytest

from arbiter import run_arbitration, run_batch
from arbiter.critics import Critic
from arbiter.models import Dimension
from arbiter.storage import Storage
from arbiter import graph as graph_module


@pytest.mark.parametrize("use_graph", [True, False])
def test_pipeline_runs_both_orchestrators(use_graph):
    if use_graph and not graph_module.langgraph_available():
        pytest.skip("langgraph not installed")
    result = run_arbitration(
        "The sun revolves around the earth. Everyone knows that.",
        prompt="State an astronomy fact.",
        persist=False,
        use_graph=use_graph,
    )
    # Exactly three critic reports, deterministic order.
    assert [r.dimension for r in result.reports] == list(Dimension)
    assert result.verdict.quality_score <= 6
    assert result.verdict.confirmed_issues


def test_short_circuit_on_clean_output():
    result = run_arbitration(
        "Photosynthesis lets plants convert sunlight, water and carbon dioxide into "
        "glucose and oxygen inside their chloroplasts.",
        prompt="What is photosynthesis?",
        persist=False,
    )
    assert result.short_circuited is True
    assert result.verdict.quality_score == 10
    assert result.verdict.confidence >= 0.75


def test_graceful_degradation(monkeypatch):
    """If a critic's backend keeps failing, the pipeline still returns a verdict."""
    from arbiter.providers.mock import MockBackend

    original = MockBackend.critique

    def flaky(self, *, dimension, **kwargs):
        if dimension is Dimension.ACCURACY:
            raise RuntimeError("simulated API failure")
        return original(self, dimension=dimension, **kwargs)

    monkeypatch.setattr(MockBackend, "critique", flaky)

    result = run_arbitration("Some ordinary sentence.", persist=False, use_graph=False)
    assert result.degraded is True
    assert Dimension.ACCURACY in result.degraded_dimensions
    # The failed critic is reported but not fatal.
    acc = next(r for r in result.reports if r.dimension is Dimension.ACCURACY)
    assert acc.ok is False and acc.error
    assert acc.attempts >= 1
    # A verdict is still produced from the surviving critics.
    assert result.verdict is not None


def test_critic_retries_then_reports_failure(monkeypatch):
    from arbiter.providers.mock import MockBackend

    calls = {"n": 0}

    def always_fail(self, **kwargs):
        calls["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(MockBackend, "critique", always_fail)
    critic = Critic(Dimension.LOGIC)
    report = critic.run("text", None)
    assert report.ok is False
    assert calls["n"] == report.attempts >= 1


def test_persistence_and_retrieval():
    result = run_arbitration("Water boils at 50 degrees celsius.", persist=True)
    storage = Storage()
    fetched = storage.get(result.id)
    assert fetched is not None
    assert fetched.id == result.id
    assert storage.count() >= 1
    listing = storage.list()
    assert any(row["id"] == result.id for row in listing)


def test_run_batch():
    results = run_batch(
        [("The sun revolves around the earth.", None), ("Paris is the capital of France.", None)],
        persist=False,
    )
    assert len(results) == 2
    assert results[0].verdict.quality_score <= results[1].verdict.quality_score
