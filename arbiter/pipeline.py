"""High-level arbitration entry point.

`run_arbitration` is the single function the API, UI, demo, and tests all call.
It drives the LangGraph pipeline when LangGraph is installed and otherwise falls
back to an equivalent plain-Python orchestrator (same nodes, same parallel
fan-out via a thread pool), then assembles and (optionally) persists the full
`ArbitrationResult` audit record.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from .adjudicator import Adjudicator, degraded_note, high_confidence_pass
from .config import Settings, get_settings
from .critics import build_critics
from .disagreement import detect_disagreements, is_unanimous_pass
from .models import ArbitrationResult, Dimension
from . import graph as graph_module


def _plain_orchestrator(
    output_text: str, original_prompt: Optional[str], settings: Settings
) -> dict:
    """LangGraph-free orchestrator with identical semantics (fallback path)."""
    critics = build_critics(settings)

    # Parallel critic dispatch.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            dim: pool.submit(critics[dim].run, output_text, original_prompt)
            for dim in Dimension
        }
        reports = [futures[dim].result() for dim in Dimension]

    disagreements = detect_disagreements(reports)
    short_circuited = is_unanimous_pass(reports)

    if short_circuited:
        verdict = high_confidence_pass(reports)
    else:
        verdict = Adjudicator(settings).run(
            output_text, original_prompt, reports, disagreements
        )

    return {
        "output_text": output_text,
        "original_prompt": original_prompt,
        "reports": reports,
        "disagreements": disagreements,
        "verdict": verdict,
        "short_circuited": short_circuited,
    }


def run_arbitration(
    output: str,
    prompt: Optional[str] = None,
    *,
    settings: Settings | None = None,
    persist: bool = True,
    use_graph: bool = True,
    arbitration_id: Optional[str] = None,
) -> ArbitrationResult:
    """Run the full arbitration pipeline and return the audit record."""
    settings = settings or get_settings()
    output = (output or "").strip()
    prompt = prompt.strip() if isinstance(prompt, str) and prompt.strip() else None

    if use_graph and graph_module.langgraph_available():
        state = graph_module.run_graph(output, prompt, settings)
    else:
        state = _plain_orchestrator(output, prompt, settings)

    # Parallel fan-in completes in nondeterministic order; sort for a stable
    # audit record (accuracy, logic, completeness).
    order = {d: i for i, d in enumerate(Dimension)}
    reports = sorted(state["reports"], key=lambda r: order[r.dimension])
    degraded, degraded_dims = degraded_note(reports)

    result = ArbitrationResult(
        id=arbitration_id or uuid.uuid4().hex[:12],
        created_at=datetime.now(timezone.utc),
        original_prompt=prompt,
        output_text=output,
        reports=reports,
        disagreements=state.get("disagreements", []),
        verdict=state["verdict"],
        short_circuited=bool(state.get("short_circuited")),
        degraded=degraded,
        degraded_dimensions=degraded_dims,
        config={
            "routing": settings.routing_summary(),
            "models": {
                r.dimension.value: f"{r.backend}/{r.model}" for r in reports
            },
            "orchestrator": (
                "langgraph" if (use_graph and graph_module.langgraph_available()) else "plain"
            ),
        },
    )

    if persist:
        # Imported lazily so the core pipeline has no hard storage dependency.
        from .storage import Storage

        Storage(settings).save(result)

    return result


def run_batch(
    items: list[tuple[str, Optional[str]]],
    *,
    settings: Settings | None = None,
    persist: bool = True,
) -> list[ArbitrationResult]:
    """Arbitrate many outputs (Phase 4.3 / 5.1 batch mode)."""
    settings = settings or get_settings()
    return [
        run_arbitration(output, prompt, settings=settings, persist=persist)
        for output, prompt in items
    ]
