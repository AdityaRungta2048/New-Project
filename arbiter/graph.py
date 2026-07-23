"""LangGraph orchestration layer (Phase 2).

Wires the critics, disagreement detector, and adjudicator into a single state
graph with the lifecycle the spec calls for:

    parse -> [accuracy | logic | completeness]  (parallel fan-out)
          -> collect  (fan-in)
          -> detect_disagreements
          -> {short_circuit | adjudicate}   (conditional)
          -> synthesize -> END

The three critic nodes run in the same superstep (true parallel dispatch) and
fan back in at `collect` thanks to the additive `reports` reducer. If all
critics unanimously pass, the graph short-circuits the adjudicator (Phase 2.4).

LangGraph is optional: if it isn't installed, `build_graph` raises and the
pipeline falls back to an equivalent plain-Python orchestrator.
"""

from __future__ import annotations

from typing import Optional

from .adjudicator import Adjudicator, high_confidence_pass
from .config import Settings, get_settings
from .critics import build_critics
from .disagreement import detect_disagreements, is_unanimous_pass
from .models import Dimension
from .state import ArbitrationState

try:  # LangGraph is optional at runtime.
    from langgraph.graph import END, START, StateGraph

    _HAS_LANGGRAPH = True
except Exception:  # pragma: no cover - only when langgraph missing
    _HAS_LANGGRAPH = False


def langgraph_available() -> bool:
    return _HAS_LANGGRAPH


def build_graph(settings: Settings | None = None):
    """Compile and return the LangGraph arbitration graph."""
    if not _HAS_LANGGRAPH:  # pragma: no cover
        raise RuntimeError("langgraph is not installed; use the plain-Python orchestrator.")

    settings = settings or get_settings()
    critics = build_critics(settings)
    adjudicator = Adjudicator(settings)

    # -- nodes -------------------------------------------------------------
    def parse(state: ArbitrationState) -> dict:
        # Input parsing / normalisation node.
        text = (state.get("output_text") or "").strip()
        prompt = state.get("original_prompt")
        prompt = prompt.strip() if isinstance(prompt, str) else prompt
        return {"output_text": text, "original_prompt": prompt}

    def make_critic_node(dimension: Dimension):
        def node(state: ArbitrationState) -> dict:
            report = critics[dimension].run(
                state["output_text"], state.get("original_prompt")
            )
            return {"reports": [report]}

        return node

    def collect(state: ArbitrationState) -> dict:
        # Fan-in / critique-collection node. NOTE: `reports` uses an additive
        # reducer, so this node must NOT re-emit it (that would double the list).
        # It only derives the non-reduced `degraded` flag from what fanned in.
        reports = state.get("reports", [])
        return {"degraded": any(not r.ok for r in reports)}

    def detect(state: ArbitrationState) -> dict:
        reports = state.get("reports", [])
        disagreements = detect_disagreements(reports)
        short_circuit = is_unanimous_pass(reports)
        return {"disagreements": disagreements, "short_circuited": short_circuit}

    def short_circuit(state: ArbitrationState) -> dict:
        return {"verdict": high_confidence_pass(state["reports"])}

    def adjudicate(state: ArbitrationState) -> dict:
        verdict = adjudicator.run(
            state["output_text"],
            state.get("original_prompt"),
            state["reports"],
            state.get("disagreements", []),
        )
        return {"verdict": verdict}

    def synthesize(state: ArbitrationState) -> dict:
        # Verdict-synthesis node: final pass-through where post-processing/
        # validation would live. The verdict is already a validated model.
        return {}

    def route_after_detect(state: ArbitrationState) -> str:
        return "short_circuit" if state.get("short_circuited") else "adjudicate"

    # -- wiring ------------------------------------------------------------
    graph = StateGraph(ArbitrationState)
    graph.add_node("parse", parse)
    graph.add_node("critic_accuracy", make_critic_node(Dimension.ACCURACY))
    graph.add_node("critic_logic", make_critic_node(Dimension.LOGIC))
    graph.add_node("critic_completeness", make_critic_node(Dimension.COMPLETENESS))
    graph.add_node("collect", collect)
    graph.add_node("detect_disagreements", detect)
    graph.add_node("short_circuit", short_circuit)
    graph.add_node("adjudicate", adjudicate)
    graph.add_node("synthesize", synthesize)

    graph.add_edge(START, "parse")
    # Parallel fan-out to all three critics.
    graph.add_edge("parse", "critic_accuracy")
    graph.add_edge("parse", "critic_logic")
    graph.add_edge("parse", "critic_completeness")
    # Fan-in: collect waits for all three critic nodes.
    graph.add_edge("critic_accuracy", "collect")
    graph.add_edge("critic_logic", "collect")
    graph.add_edge("critic_completeness", "collect")
    graph.add_edge("collect", "detect_disagreements")
    graph.add_conditional_edges(
        "detect_disagreements",
        route_after_detect,
        {"short_circuit": "short_circuit", "adjudicate": "adjudicate"},
    )
    graph.add_edge("short_circuit", "synthesize")
    graph.add_edge("adjudicate", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def run_graph(
    output_text: str,
    original_prompt: Optional[str],
    settings: Settings | None = None,
) -> ArbitrationState:
    """Execute the compiled graph and return the final state."""
    compiled = build_graph(settings)
    result = compiled.invoke(
        {"output_text": output_text, "original_prompt": original_prompt}
    )
    return result
