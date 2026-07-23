"""Run the four compelling portfolio test cases and render their verdicts.

    python -m examples.demo            # print to console
    python -m examples.demo --write    # also write docs/sample_verdicts.md

Cases (Phase 6.1):
  1. A factually incorrect response (planted errors)
  2. A logically flawed argument
  3. A response that technically answers but misses the point
  4. A genuinely good response (clean bill of health)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arbiter import run_arbitration  # noqa: E402
from arbiter.analytics import compute_analytics  # noqa: E402
from arbiter.config import get_settings  # noqa: E402
from arbiter.models import ArbitrationResult  # noqa: E402

CASES = [
    (
        "1 · Factually incorrect (planted errors)",
        "State a few basic science facts.",
        "The sun revolves around the earth. Water boils at 50 degrees celsius at sea level. "
        "The human body has 106 bones, and Einstein invented the light bulb.",
    ),
    (
        "2 · Logically flawed argument",
        "Should the town approve the new stadium?",
        "Everyone knows the stadium is a great idea. If we don't build it, then tourism will "
        "collapse, then the shops will close, then the town will die. Anyone who disagrees is "
        "simply a fool.",
    ),
    (
        "3 · Technically answers but misses the point",
        "Compare the causes, course, and consequences of World War I and World War II, and "
        "explain their key political and economic differences.",
        "World War I and World War II were both major wars in the 20th century. Many nations "
        "fought in them. They were very significant events. etc.",
    ),
    (
        "4 · Genuinely good response",
        "What is photosynthesis?",
        "Photosynthesis is the process by which green plants use sunlight to make food from "
        "carbon dioxide and water, producing glucose and releasing oxygen. It happens in the "
        "chloroplasts using the pigment chlorophyll.",
    ),
]

BAR = "=" * 78


def _print_console(title: str, result: ArbitrationResult) -> None:
    v = result.verdict
    print(f"\n{BAR}\n{title}\n{BAR}")
    print(f"OUTPUT: {result.output_text}\n")
    for r in result.reports:
        if r.ok:
            print(
                f"  [{r.dimension.value:12}] {r.backend}/{r.model}: score {r.critique.score}/5, "
                f"{len(r.critique.issues)} issue(s), self-conf {r.critique.confidence:.2f}"
            )
        else:
            print(f"  [{r.dimension.value:12}] FAILED: {r.error}")
    if result.disagreements:
        print(f"\n  Disagreements ({len(result.disagreements)}):")
        for d in result.disagreements:
            print(f"    - {d.kind.value}: {d.description}")
    print(
        f"\n  >>> VERDICT: quality {v.quality_score}/10 | confidence {v.confidence:.0%} "
        f"({v.confidence_level.value}) | "
        f"{'SHORT-CIRCUIT PASS' if result.short_circuited else 'adjudicated'}"
    )
    print(f"      confirmed: {len(v.confirmed_issues)} | dismissed: {len(v.dismissed_flags)} "
          f"| validated: {len(v.validated_claims)}")
    for ci in v.confirmed_issues:
        print(f"        - [{ci.dimension.value} sev {ci.severity}] {ci.problem}")
    print(f"      summary: {v.summary}")


def _md(result: ArbitrationResult, title: str) -> str:
    v = result.verdict
    lines = [f"## {title}", ""]
    if result.original_prompt:
        lines += [f"**Prompt:** {result.original_prompt}", ""]
    lines += [f"**Output under evaluation:**", "", f"> {result.output_text}", ""]
    lines += [
        "| Critic | Model | Score | Issues | Self-confidence |",
        "|---|---|---|---|---|",
    ]
    for r in result.reports:
        if r.ok:
            lines.append(
                f"| {r.dimension.value} | `{r.backend}/{r.model}` | {r.critique.score}/5 | "
                f"{len(r.critique.issues)} | {r.critique.confidence:.2f} |"
            )
        else:
            lines.append(f"| {r.dimension.value} | `{r.backend}/{r.model}` | FAILED | — | — |")
    lines.append("")
    if result.disagreements:
        lines.append("**Disagreements detected:**")
        for d in result.disagreements:
            lines.append(f"- _{d.kind.value}_ — {d.description}")
        lines.append("")
    badge = "✅ short-circuit clean pass" if result.short_circuited else "⚖️ adjudicated"
    lines += [
        f"**Verdict — quality {v.quality_score}/10, confidence "
        f"{v.confidence:.0%} ({v.confidence_level.value}), {badge}**",
        "",
    ]
    if v.confirmed_issues:
        lines.append("Confirmed issues:")
        for ci in v.confirmed_issues:
            lines.append(
                f"- 🔴 **[{ci.dimension.value} · severity {ci.severity}]** {ci.problem}  \n"
                f"  quote: _“{ci.quote}”_ — {ci.evidence}"
            )
    if v.dismissed_flags:
        lines.append("\nDismissed flags:")
        for df in v.dismissed_flags:
            lines.append(f"- 🟡 **[{df.raised_by.value}]** {df.problem} — _{df.reasoning}_")
    if v.validated_claims:
        lines.append("\nValidated claims:")
        for vc in v.validated_claims:
            lines.append(f"- 🟢 _“{vc.quote}”_ — {vc.note}")
    lines += ["", f"> {v.summary}", "", "---", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write docs/sample_verdicts.md")
    args = parser.parse_args()

    settings = get_settings()
    print("Backend routing:", settings.routing_summary())

    results = []
    for title, prompt, output in CASES:
        result = run_arbitration(output, prompt, persist=True)
        results.append((title, result))
        _print_console(title, result)

    print(f"\n{BAR}\nCROSS-ARBITRATION ANALYTICS\n{BAR}")
    analytics = compute_analytics(settings)
    print(f"  total arbitrations : {analytics['total_arbitrations']}")
    print(f"  most issues found  : {analytics.get('most_issues_found_by')}")
    print(f"  most overruled     : {analytics.get('most_overruled_critic')}")
    print(f"  disagreement rate  : {analytics['agreement']['disagreement_rate']:.0%}")
    print(f"  common failures    : {analytics['common_failure_types']}")

    if args.write:
        out_path = Path(__file__).resolve().parent.parent / "docs" / "sample_verdicts.md"
        out_path.parent.mkdir(exist_ok=True)
        body = [
            "# Sample Verdicts",
            "",
            "Generated by `python -m examples.demo --write`. These are the four "
            "canonical portfolio cases, evaluated end-to-end by the arbitration "
            "pipeline. (Backend routing for this run: "
            f"`{settings.routing_summary()}`.)",
            "",
        ]
        for title, result in results:
            body.append(_md(result, title))
        out_path.write_text("\n".join(body), encoding="utf-8")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
