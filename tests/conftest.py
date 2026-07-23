"""Shared fixtures. Force the deterministic mock backend so tests never touch
the network, and isolate storage to a temp dir per test session."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _mock_backends(monkeypatch, tmp_path):
    for var in (
        "ARBITER_ACCURACY_BACKEND",
        "ARBITER_LOGIC_BACKEND",
        "ARBITER_COMPLETENESS_BACKEND",
        "ARBITER_ADJUDICATOR_BACKEND",
    ):
        monkeypatch.setenv(var, "mock")
    monkeypatch.setenv("ARBITER_MAX_RETRIES", "1")
    monkeypatch.setenv("ARBITER_DB_PATH", str(tmp_path / "arb.sqlite"))
    monkeypatch.setenv("ARBITER_JSON_DIR", str(tmp_path / "json"))
    # Refresh the settings singleton so the env vars above take effect.
    from arbiter import config

    config.get_settings(refresh=True)
    yield
    config.get_settings(refresh=True)
