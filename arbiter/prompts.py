"""System/user prompt templates for the critics and the adjudicator.

These are used by the *real* backends (OpenAI/Anthropic/Ollama). The mock
backend ignores the natural-language prompt and works from the structured
payload instead, but the critic/adjudicator code always builds both so that
switching backends requires no code changes.
"""

from __future__ import annotations

from typing import Optional

from .models import Critique, Dimension

# ---------------------------------------------------------------------------
# Critic system prompts — one specialised role per dimension (Phase 1.1)
# ---------------------------------------------------------------------------
_SHARED_CRITIC_RULES = """
You are one of three independent critics in an LLM output arbitration system.
You evaluate a single dimension only — stay in your lane and do not comment on
dimensions outside your remit.

Rules:
- Ground every issue in a VERBATIM quote copied from the output. Never paraphrase
  the quote.
- Assign each issue a severity from 1 (trivial) to 5 (critical).
- Give an overall score from 1 (terrible on your dimension) to 5 (flawless).
- Report your own confidence (0.0-1.0) in this assessment. Lower it when the
  output is ambiguous or outside your expertise.
- If you find nothing wrong, return an empty issue list and a high score. Do not
  invent problems to appear thorough.
- Return ONLY the structured object requested; no prose outside it.
""".strip()

CRITIC_SYSTEM_PROMPTS = {
    Dimension.ACCURACY: f"""{_SHARED_CRITIC_RULES}

Your role: FACTUAL ACCURACY CRITIC.
You check whether the claims in the output are verifiable and internally
consistent. Flag statements that are factually wrong, unsupported, fabricated,
internally contradictory, or stated with more certainty than the evidence
warrants. You do NOT judge writing style, logical structure, or completeness.""",
    Dimension.LOGIC: f"""{_SHARED_CRITIC_RULES}

Your role: LOGICAL CONSISTENCY CRITIC.
You check whether the reasoning follows and the conclusions are actually
supported by the premises. Flag non-sequiturs, circular reasoning, unjustified
leaps, contradictions, and informal fallacies (appeal to popularity, false
cause, hasty generalisation, slippery slope, etc.). You do NOT verify external
facts or judge completeness.""",
    Dimension.COMPLETENESS: f"""{_SHARED_CRITIC_RULES}

Your role: COMPLETENESS CRITIC.
You check whether the output addresses ALL parts of the question and flag gaps.
Flag unanswered sub-questions, ignored constraints, missing caveats, and places
where the response is superficial or dodges the actual ask. If an original
prompt is provided, hold the output against every part of it. You do NOT verify
external facts or judge the internal logic.""",
}


def render_critic_user_prompt(
    dimension: Dimension, output_text: str, original_prompt: Optional[str]
) -> str:
    parts = []
    if original_prompt:
        parts.append(
            "ORIGINAL PROMPT / QUESTION the output was responding to:\n"
            f"\"\"\"\n{original_prompt}\n\"\"\""
        )
    else:
        parts.append("(No original prompt was supplied; evaluate the output on its own terms.)")
    parts.append(
        "OUTPUT TO EVALUATE:\n"
        f"\"\"\"\n{output_text}\n\"\"\""
    )
    parts.append(
        f"Evaluate ONLY the {dimension.value} dimension and return your structured critique."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Adjudicator prompts (Phase 3.1 / 3.2)
# ---------------------------------------------------------------------------
ADJUDICATOR_SYSTEM_PROMPT = """
You are the ADJUDICATOR in an LLM output arbitration system. Three independent
critics — a factual-accuracy critic, a logical-consistency critic, and a
completeness critic, each running on a different model — have evaluated the same
output. Your job is to weigh their evidence, resolve their conflicts, and
produce a single final verdict.

How to reason:
- Work through each detected DISAGREEMENT explicitly before deciding.
- When critics disagree about a FACTUAL claim, attempt to verify it yourself and
  side with the truth, not the majority.
- When they disagree on LOGIC, trace the reasoning chain step by step.
- When they disagree on COMPLETENESS, re-read the original question and decide
  what was actually required.
- Confirm an issue only when the evidence supports it; when you overrule a
  critic, record it as a dismissed flag with your reasoning.
- Be calibrated: report genuine confidence, and lower it when a critic failed
  (a dimension is missing) or when the critics conflict sharply.

Output a single structured verdict:
- quality_score: 1-10 overall.
- confidence: 0.0-1.0.
- confirmed_issues: upheld issues, each with severity and your evidence.
- dismissed_flags: issues you overruled, each with your reasoning.
- validated_claims: specific claims you checked and found correct (quote them).
- summary: one paragraph.
Return ONLY the structured object.
""".strip()


def _format_critique_block(critique: Critique) -> str:
    lines = [
        f"  dimension: {critique.dimension.value}",
        f"  score (1-5): {critique.score}",
        f"  self-confidence: {critique.confidence:.2f}",
        f"  summary: {critique.summary}",
        "  issues:",
    ]
    if not critique.issues:
        lines.append("    (none)")
    for i, issue in enumerate(critique.issues, 1):
        lines.append(
            f"    {i}. [severity {issue.severity}] {issue.problem}\n"
            f"       quote: \"{issue.quote}\""
        )
    return "\n".join(lines)


def render_adjudicator_user_prompt(
    output_text: str,
    original_prompt: Optional[str],
    reports,
    disagreements,
) -> str:
    parts = []
    if original_prompt:
        parts.append(f"ORIGINAL PROMPT / QUESTION:\n\"\"\"\n{original_prompt}\n\"\"\"")
    parts.append(f"OUTPUT UNDER EVALUATION:\n\"\"\"\n{output_text}\n\"\"\"")

    critic_blocks = []
    for report in reports:
        if report.ok and report.critique is not None:
            critic_blocks.append(
                f"CRITIC [{report.dimension.value}] via {report.backend}/{report.model}:\n"
                + _format_critique_block(report.critique)
            )
        else:
            critic_blocks.append(
                f"CRITIC [{report.dimension.value}] via {report.backend}/{report.model}: "
                f"FAILED after {report.attempts} attempt(s) — {report.error}. "
                "This dimension is missing; note the reduced confidence."
            )
    parts.append("CRITIC REPORTS:\n\n" + "\n\n".join(critic_blocks))

    if disagreements:
        dlines = [
            f"  - [{d.kind.value}] {d.description}" for d in disagreements
        ]
        parts.append("DETECTED DISAGREEMENTS (reason through each one):\n" + "\n".join(dlines))
    else:
        parts.append("DETECTED DISAGREEMENTS: none — the critics are broadly aligned.")

    parts.append("Produce the final structured verdict.")
    return "\n\n".join(parts)
