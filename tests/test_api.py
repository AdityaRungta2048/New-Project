"""FastAPI surface: endpoints, validation, OpenAPI."""

import pytest
from fastapi.testclient import TestClient

from arbiter.api import app


@pytest.fixture()
def client():
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["backends"]) == {"accuracy", "logic", "completeness", "adjudicator"}


def test_arbitrate_single(client):
    resp = client.post(
        "/v1/arbitrate",
        json={"output": "The sun revolves around the earth.", "prompt": "astronomy fact"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"]["quality_score"] <= 6
    assert data["id"]


def test_arbitrate_empty_rejected(client):
    resp = client.post("/v1/arbitrate", json={"output": "   "})
    assert resp.status_code == 422


def test_arbitrate_batch(client):
    resp = client.post(
        "/v1/arbitrate/batch",
        json={"items": [{"output": "2+2=5 is proven."}, {"output": "Paris is in France."}]},
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 2


def test_retrieve_and_list(client):
    created = client.post("/v1/arbitrate", json={"output": "Water boils at 50 celsius."}).json()
    got = client.get(f"/v1/arbitrations/{created['id']}")
    assert got.status_code == 200 and got.json()["id"] == created["id"]

    listing = client.get("/v1/arbitrations").json()
    assert listing["total"] >= 1


def test_retrieve_missing_returns_404(client):
    assert client.get("/v1/arbitrations/does-not-exist").status_code == 404


def test_analytics_endpoint(client):
    client.post("/v1/arbitrate", json={"output": "The sun revolves around the earth."})
    data = client.get("/v1/analytics").json()
    assert data["total_arbitrations"] >= 1
    assert "per_critic" in data


def test_openapi_documented(client):
    spec = client.get("/openapi.json").json()
    assert "/v1/arbitrate" in spec["paths"]
    assert "/v1/arbitrate/batch" in spec["paths"]
    assert "/v1/arbitrations/{arbitration_id}" in spec["paths"]
