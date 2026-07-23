"""Verdict Explorer — Streamlit UI (Phase 4).

Three views:
  1. Single arbitration: original output with inline colour-coded annotations
     (red=confirmed, amber=low-confidence/dismissed, green=validated) whose
     evidence chain is exposed on hover and in an expandable list, plus a
     side-by-side critic comparison panel that highlights agreements (green)
     and disagreements (orange).
  2. Batch mode: submit many outputs and browse a sortable results table.
  3. Analytics: the critic-behaviour meta-analysis.

Run with:  streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make the `arbiter` package importable when run via `streamlit run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arbiter import run_arbitration  # noqa: E402
from arbiter.analytics import compute_analytics  # noqa: E402
from arbiter.config import get_settings  # noqa: E402
from arbiter.models import ArbitrationResult, Dimension  # noqa: E402
from ui.annotate import Annotation, annotate_html, legend_html  # noqa: E402

st.set_page_config(page_title="LLM Verdict Explorer", page_icon="⚖️", layout="wide")

DIM_ORDER = [Dimension.ACCURACY, Dimension.LOGIC, Dimension.COMPLETENESS]
DIM_MODEL_HINT = {
    Dimension.ACCURACY: "Factual Accuracy",
    Dimension.LOGIC: "Logical Consistency",
    Dimension.COMPLETENESS: "Completeness",
}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _annotations_for(result: ArbitrationResult) -> list[Annotation]:
    anns: list[Annotation] = []
    for ci in result.verdict.confirmed_issues:
        anns.append(
            Annotation(
                quote=ci.quote,
                kind="confirmed",
                evidence=f"{ci.problem} — {ci.evidence}",
                label=f"{ci.dimension.value} · severity {ci.severity}",
            )
        )
    for df in result.verdict.dismissed_flags:
        anns.append(
            Annotation(
                quote=df.quote,
                kind="flag",
                evidence=f"{df.problem} — overruled: {df.reasoning}",
                label=f"{df.raised_by.value} · dismissed",
            )
        )
    for vc in result.verdict.validated_claims:
        anns.append(
            Annotation(quote=vc.quote, kind="validated", evidence=vc.note, label="validated")
        )
    return anns


def _quality_color(score: int) -> str:
    if score >= 8:
        return "#1e874b"
    if score >= 5:
        return "#e0a800"
    return "#e74c3c"


def render_verdict_header(result: ArbitrationResult) -> None:
    v = result.verdict
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f"<h1 style='color:{_quality_color(v.quality_score)};margin-bottom:0'>"
        f"{v.quality_score}<span style='font-size:1rem;color:gray'>/10</span></h1>"
        "<div style='color:gray'>Quality score</div>",
        unsafe_allow_html=True,
    )
    c2.metric("Confidence", f"{v.confidence:.0%}", v.confidence_level.value)
    c3.metric("Confirmed issues", len(v.confirmed_issues))
    c4.metric("Dismissed flags", len(v.dismissed_flags))

    badges = []
    if result.short_circuited:
        badges.append("✅ Unanimous clean pass (adjudicator short-circuited)")
    if result.degraded:
        dims = ", ".join(d.value for d in result.degraded_dimensions)
        badges.append(f"⚠️ Degraded: {dims} critic(s) failed — reduced confidence")
    if v.passed and not result.short_circuited:
        badges.append("✅ Passed")
    if not v.passed and not result.short_circuited:
        badges.append("❌ Issues found")
    st.write(" &nbsp; ".join(badges), unsafe_allow_html=True)
    st.info(v.summary)


def render_annotated_output(result: ArbitrationResult) -> None:
    st.subheader("Annotated output")
    st.markdown(legend_html(), unsafe_allow_html=True)
    anns = _annotations_for(result)
    st.markdown(annotate_html(result.output_text, anns), unsafe_allow_html=True)
    st.caption("Hover any marker to see its evidence chain, or expand the list below.")

    v = result.verdict
    if v.confirmed_issues:
        st.markdown("##### 🔴 Confirmed issues — evidence chain")
        for i, ci in enumerate(v.confirmed_issues, 1):
            with st.expander(
                f"{i}. [{ci.dimension.value} · severity {ci.severity} · {ci.severity_label}] "
                f"{ci.problem}"
            ):
                st.markdown(f"> {ci.quote}")
                st.write(f"**Adjudicator evidence:** {ci.evidence}")
                st.write(f"**Raised by:** {', '.join(d.value for d in ci.raised_by)}")
    if v.dismissed_flags:
        st.markdown("##### 🟡 Dismissed flags — overruled by the adjudicator")
        for i, df in enumerate(v.dismissed_flags, 1):
            with st.expander(f"{i}. [{df.raised_by.value}] {df.problem}"):
                st.markdown(f"> {df.quote}")
                st.write(f"**Why overruled:** {df.reasoning}")
    if v.validated_claims:
        st.markdown("##### 🟢 Validated claims")
        for vc in v.validated_claims:
            st.markdown(f"- > {vc.quote}  \n  _{vc.note}_")


def render_critic_panel(result: ArbitrationResult) -> None:
    st.subheader("Critic comparison panel")
    # Which dimensions are in a disagreement -> highlight orange.
    conflicting: set[str] = set()
    for d in result.disagreements:
        for dim in d.dimensions:
            conflicting.add(dim.value)

    cols = st.columns(3)
    by_dim = {r.dimension: r for r in result.reports}
    for col, dim in zip(cols, DIM_ORDER):
        report = by_dim.get(dim)
        with col:
            in_conflict = dim.value in conflicting
            border = "#e67e22" if in_conflict else "#1e874b"
            tag = "⚠️ disagreement" if in_conflict else "✅ aligned"
            st.markdown(
                f"<div style='border-top:4px solid {border};padding-top:6px'>"
                f"<b>{DIM_MODEL_HINT[dim]} Critic</b><br>"
                f"<span style='color:gray;font-size:.85rem'>{tag}</span></div>",
                unsafe_allow_html=True,
            )
            if not report or not report.ok:
                st.error(f"Failed: {report.error if report else 'no report'}")
                continue
            crit = report.critique
            st.caption(f"model: `{report.backend}/{report.model}`")
            st.metric("Score", f"{crit.score}/5")
            st.progress(crit.confidence, text=f"self-confidence {crit.confidence:.0%}")
            if crit.issues:
                for iss in crit.issues:
                    st.markdown(
                        f"<div style='background:rgba(230,126,34,.10);border-left:3px solid "
                        f"#e67e22;padding:6px 8px;margin:6px 0;border-radius:4px;font-size:.85rem'>"
                        f"<b>sev {iss.severity}</b> · {iss.problem}<br>"
                        f"<i>“{iss.quote[:120]}”</i></div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.success("No issues on this dimension")

    if result.disagreements:
        st.markdown("##### Detected disagreements")
        for d in result.disagreements:
            st.markdown(
                f"<div style='background:rgba(230,126,34,.12);padding:6px 10px;margin:4px 0;"
                f"border-radius:4px'>🟠 <b>{d.kind.value}</b> — {d.description}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<div style='background:rgba(30,135,75,.12);padding:6px 10px;border-radius:4px'>"
            "🟢 The critics are broadly aligned — no disagreements detected.</div>",
            unsafe_allow_html=True,
        )


def render_result(result: ArbitrationResult) -> None:
    render_verdict_header(result)
    st.divider()
    render_annotated_output(result)
    st.divider()
    render_critic_panel(result)
    with st.expander("Raw audit record (JSON)"):
        st.json(result.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------
EXAMPLES = {
    "— pick an example —": ("", ""),
    "Factually incorrect": (
        "Explain some basic science facts.",
        "The sun revolves around the earth. Water boils at 50 degrees celsius at sea "
        "level. The human body has 106 bones, and Einstein invented the light bulb.",
    ),
    "Logically flawed": (
        "Should the city build the new highway?",
        "Everyone knows the highway is a good idea. If we don't build it, then traffic "
        "will worsen, then businesses will leave, then the city will collapse entirely. "
        "Anyone who disagrees is simply a fool.",
    ),
    "Technically answers, misses the point": (
        "Compare the causes, course, and consequences of World War I and World War II, "
        "and explain their differences.",
        "World War I and World War II were both large wars in the 20th century. Many "
        "countries were involved. They were very significant. etc.",
    ),
    "Genuinely good response": (
        "What is photosynthesis?",
        "Photosynthesis is the process by which green plants use sunlight to synthesize "
        "food from carbon dioxide and water, producing glucose and releasing oxygen. It "
        "takes place in the chloroplasts, primarily using the pigment chlorophyll.",
    ),
}


def view_single() -> None:
    st.title("⚖️ LLM Output Arbitration — Verdict Explorer")
    st.caption(
        "Three specialised critics independently evaluate an LLM output; an adjudicator "
        "resolves their disagreements into one confidence-scored verdict."
    )
    example = st.selectbox("Load an example", list(EXAMPLES.keys()))
    ex_prompt, ex_output = EXAMPLES[example]

    prompt = st.text_area("Original prompt (optional)", value=ex_prompt, height=80)
    output = st.text_area("LLM output to evaluate", value=ex_output, height=180)

    if st.button("Arbitrate", type="primary", use_container_width=True):
        if not output.strip():
            st.warning("Enter an output to evaluate.")
            return
        with st.spinner("Running critics in parallel and adjudicating…"):
            result = run_arbitration(output, prompt or None)
        st.session_state["last_result"] = result

    if "last_result" in st.session_state:
        st.divider()
        render_result(st.session_state["last_result"])


def view_batch() -> None:
    st.title("📦 Batch arbitration")
    st.caption("Submit multiple outputs (one per line, or separate blocks with a line of ---).")
    shared_prompt = st.text_input("Shared original prompt (optional)")
    raw = st.text_area(
        "Outputs",
        height=220,
        placeholder="First output…\n---\nSecond output…\n---\nThird output…",
    )
    if st.button("Arbitrate batch", type="primary"):
        blocks = [b.strip() for b in raw.split("---")] if "---" in raw else raw.splitlines()
        items = [b.strip() for b in blocks if b.strip()]
        if not items:
            st.warning("Enter at least one output.")
            return
        results = []
        prog = st.progress(0.0)
        for i, text in enumerate(items, 1):
            results.append(run_arbitration(text, shared_prompt or None))
            prog.progress(i / len(items))
        st.session_state["batch_results"] = results

    results = st.session_state.get("batch_results")
    if results:
        st.divider()
        rows = [
            {
                "id": r.id,
                "excerpt": r.output_text[:70] + ("…" if len(r.output_text) > 70 else ""),
                "quality": r.verdict.quality_score,
                "confidence": round(r.verdict.confidence, 2),
                "issues_found": r.num_issues_found,
                "confirmed": len(r.verdict.confirmed_issues),
                "disagreements": len(r.disagreements),
            }
            for r in results
        ]
        st.markdown("#### Results (click column headers to sort)")
        st.dataframe(rows, use_container_width=True, hide_index=True)
        ids = [r.id for r in results]
        chosen = st.selectbox("Inspect one arbitration", ids)
        chosen_result = next(r for r in results if r.id == chosen)
        st.divider()
        render_result(chosen_result)


def view_analytics() -> None:
    st.title("📊 Critic-behaviour analytics")
    st.caption("Meta-analysis over every arbitration recorded in the local store.")
    data = compute_analytics(get_settings())
    if data.get("total_arbitrations", 0) == 0:
        st.info("No arbitrations recorded yet. Run some in the Single or Batch views.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total arbitrations", data["total_arbitrations"])
    c2.metric("Avg quality", data.get("avg_quality_score"))
    c3.metric("Disagreement rate", f"{data['agreement']['disagreement_rate']:.0%}")
    c4.metric("Most issues found by", data.get("most_issues_found_by") or "—")

    st.markdown("#### Per-critic behaviour")
    st.dataframe(
        [{"dimension": k, **v} for k, v in data["per_critic"].items()],
        use_container_width=True,
        hide_index=True,
    )

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Common failure types")
        if data["common_failure_types"]:
            st.bar_chart({k: v for k, v in data["common_failure_types"]})
        else:
            st.write("None yet.")
    with colB:
        st.markdown("#### Adjudication")
        st.write(f"Confirmed: **{data['adjudication']['total_confirmed']}**")
        st.write(f"Dismissed: **{data['adjudication']['total_dismissed']}**")
        st.write(f"Confirm rate: **{data['adjudication']['confirm_rate']:.0%}**")
        st.write(f"Most-overruled critic: **{data.get('most_overruled_critic') or '—'}**")
        st.write(f"Short-circuited passes: **{data['agreement']['short_circuited_passes']}**")


def main() -> None:
    with st.sidebar:
        st.header("Verdict Explorer")
        view = st.radio("View", ["Single", "Batch", "Analytics"])
        st.divider()
        st.caption("Active backend routing")
        s = get_settings()
        st.code(
            f"accuracy      → {s.accuracy_backend}\n"
            f"logic         → {s.logic_backend}\n"
            f"completeness  → {s.completeness_backend}\n"
            f"adjudicator   → {s.adjudicator_backend}\n"
            f"mode          → {s.backend_mode}",
            language="text",
        )
        st.caption(
            "Set ARBITER_*_BACKEND env vars to route critics through real models "
            "(GPT-4o / Claude / Llama). Defaults to the offline mock backend."
        )

    if view == "Single":
        view_single()
    elif view == "Batch":
        view_batch()
    else:
        view_analytics()


main()
