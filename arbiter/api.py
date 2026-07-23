"""FastAPI service (Phase 5.1).

Endpoints:
    POST /v1/arbitrate         -> arbitrate one output, return the full verdict
    POST /v1/arbitrate/batch   -> arbitrate many outputs
    GET  /v1/arbitrations      -> list past arbitrations (index rows)
    GET  /v1/arbitrations/{id} -> retrieve a past verdict (full audit record)
    GET  /v1/analytics         -> critic-behaviour meta-analysis
    GET  /health               -> service + backend routing status

Interactive OpenAPI docs are served at /docs and the raw spec at
/openapi.json.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from . import __version__
from .analytics import compute_analytics
from .config import get_settings
from .models import (
    ArbitrateRequest,
    ArbitrationResult,
    BatchArbitrateRequest,
    BatchArbitrateResponse,
    Dimension,
)
from .pipeline import run_arbitration, run_batch
from .providers import describe_backend
from .storage import Storage

app = FastAPI(
    title="LLM Output Arbitration System",
    version=__version__,
    description=(
        "A multi-agent pipeline that routes any LLM-generated output to three "
        "specialised critic models (factual accuracy, logical consistency, "
        "completeness), detects where they disagree, and has an adjudicator "
        "synthesise a single confidence-scored verdict with actionable callouts."
    ),
    contact={"name": "LLM Output Arbitration System"},
    license_info={"name": "MIT"},
)


def _storage() -> Storage:
    return Storage(get_settings())


@app.get("/health", tags=["meta"], summary="Service + backend routing status")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "backends": {
            "accuracy": describe_backend(settings.accuracy_backend, settings),
            "logic": describe_backend(settings.logic_backend, settings),
            "completeness": describe_backend(settings.completeness_backend, settings),
            "adjudicator": describe_backend(settings.adjudicator_backend, settings),
        },
    }


@app.post(
    "/v1/arbitrate",
    response_model=ArbitrationResult,
    tags=["arbitration"],
    summary="Arbitrate a single LLM output",
)
def arbitrate(req: ArbitrateRequest) -> ArbitrationResult:
    if not req.output.strip():
        raise HTTPException(status_code=422, detail="`output` must not be empty.")
    return run_arbitration(req.output, req.prompt)


@app.post(
    "/v1/arbitrate/batch",
    response_model=BatchArbitrateResponse,
    tags=["arbitration"],
    summary="Arbitrate multiple outputs in one call",
)
def arbitrate_batch(req: BatchArbitrateRequest) -> BatchArbitrateResponse:
    items = [(item.output, item.prompt) for item in req.items if item.output.strip()]
    if not items:
        raise HTTPException(status_code=422, detail="No non-empty outputs supplied.")
    results = run_batch(items)
    return BatchArbitrateResponse(results=results)


@app.get(
    "/v1/arbitrations",
    tags=["arbitration"],
    summary="List past arbitrations (index rows)",
)
def list_arbitrations(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict:
    storage = _storage()
    return {"total": storage.count(), "items": storage.list(limit=limit, offset=offset)}


@app.get(
    "/v1/arbitrations/{arbitration_id}",
    response_model=ArbitrationResult,
    tags=["arbitration"],
    summary="Retrieve a past verdict (full audit record)",
)
def get_arbitration(arbitration_id: str) -> ArbitrationResult:
    result = _storage().get(arbitration_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No arbitration with id '{arbitration_id}'.")
    return result


@app.get(
    "/v1/analytics",
    tags=["analytics"],
    summary="Critic-behaviour meta-analysis across all arbitrations",
)
def analytics() -> dict:
    return compute_analytics(get_settings())


@app.get("/", tags=["meta"], summary="Service metadata")
def root() -> dict:
    return {
        "service": "LLM Output Arbitration System",
        "version": __version__,
        "dimensions": [d.value for d in Dimension],
        "docs": "/docs",
        "openapi": "/openapi.json",
    }
