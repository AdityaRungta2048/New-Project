"""Disagreement detector (Phase 2.3).

After all critiques are collected, compare them and surface the places where the
critics genuinely diverge. These disagreements are the whole point of a
multi-model panel — they are exactly the cases a single model's self-evaluation
would miss — and they are what the adjudicator is asked to resolve.

Detected kinds:
- score_divergence : overall dimension scores span more than 2 points.
- severity_conflict: two critics quote overlapping text but rate its severity
                     more than 2 points apart.
- unique_finding   : a substantive issue (severity >= 3) that exactly one critic
                     raised and the others missed entirely.
- existence        : at least one critic passed the output clean while another
                     flagged a serious problem.
"""

from __future__ import annotations

from itertools import combinations

from .models import (
    CriticReport,
    Dimension,
    Disagreement,
    DisagreementKind,
)

_MAX_UNIQUE = 6


def _norm(quote: str) -> str:
    return " ".join(quote.lower().split())


def _overlap(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) >= 0.6


def detect_disagreements(reports: list[CriticReport]) -> list[Disagreement]:
    ok = [r for r in reports if r.ok and r.critique is not None]
    disagreements: list[Disagreement] = []
    if len(ok) < 2:
        return disagreements

    # -- score divergence --------------------------------------------------
    scored = [(r.dimension, r.critique.score) for r in ok]
    hi = max(scored, key=lambda x: x[1])
    lo = min(scored, key=lambda x: x[1])
    if hi[1] - lo[1] > 2:
        disagreements.append(
            Disagreement(
                kind=DisagreementKind.SCORE_DIVERGENCE,
                description=(
                    f"The {hi[0].value} critic scored the output {hi[1]}/5 while the "
                    f"{lo[0].value} critic scored it {lo[1]}/5 — a {hi[1] - lo[1]}-point gap."
                ),
                dimensions=[hi[0], lo[0]],
                severity_gap=hi[1] - lo[1],
            )
        )

    # Flatten issues, remembering which critic raised each.
    flat = [
        (r.dimension, issue)
        for r in ok
        for issue in r.critique.issues
    ]

    # -- severity conflicts on overlapping quotes --------------------------
    conflicted_ids: set[int] = set()
    for (dim_a, iss_a), (dim_b, iss_b) in combinations(flat, 2):
        if dim_a == dim_b:
            continue
        if _overlap(iss_a.quote, iss_b.quote) and abs(iss_a.severity - iss_b.severity) > 2:
            conflicted_ids.add(id(iss_a))
            conflicted_ids.add(id(iss_b))
            disagreements.append(
                Disagreement(
                    kind=DisagreementKind.SEVERITY_CONFLICT,
                    description=(
                        f"On overlapping text, the {dim_a.value} critic rated severity "
                        f"{iss_a.severity} but the {dim_b.value} critic rated it "
                        f"{iss_b.severity}."
                    ),
                    dimensions=[dim_a, dim_b],
                    quote=iss_a.quote,
                    severity_gap=abs(iss_a.severity - iss_b.severity),
                )
            )

    # -- unique findings (found by one, missed by the rest) ----------------
    uniques: list[Disagreement] = []
    for dim, issue in flat:
        if issue.severity < 3 or id(issue) in conflicted_ids:
            continue
        overlaps_other = any(
            other_dim != dim and _overlap(issue.quote, other_issue.quote)
            for other_dim, other_issue in flat
            if id(other_issue) != id(issue)
        )
        if not overlaps_other:
            uniques.append(
                Disagreement(
                    kind=DisagreementKind.UNIQUE_FINDING,
                    description=(
                        f"Only the {dim.value} critic caught this "
                        f"(severity {issue.severity}); the other critics missed it: "
                        f"{issue.problem}"
                    ),
                    dimensions=[dim],
                    quote=issue.quote,
                    severity_gap=None,
                )
            )
    uniques.sort(key=lambda d: -(d.severity_gap or 0))
    disagreements.extend(uniques[:_MAX_UNIQUE])

    # -- existence disagreement -------------------------------------------
    clean = [r.dimension for r in ok if r.critique.score >= 5 and not r.critique.issues]
    flaggers = [
        r.dimension
        for r in ok
        if any(i.severity >= 3 for i in r.critique.issues)
    ]
    if clean and flaggers:
        disagreements.append(
            Disagreement(
                kind=DisagreementKind.EXISTENCE,
                description=(
                    f"The {', '.join(d.value for d in clean)} critic(s) passed the output "
                    f"clean, but the {', '.join(d.value for d in flaggers)} critic(s) flagged "
                    "a serious problem — the panel disagrees on whether the output is sound."
                ),
                dimensions=list(dict.fromkeys(clean + flaggers)),
            )
        )

    return disagreements


def is_unanimous_pass(reports: list[CriticReport]) -> bool:
    """True when every critic ran successfully and gave a flawless, issue-free pass.

    This is the short-circuit condition (Phase 2.4): if all critics agree the
    output is perfect, the adjudicator can be skipped.
    """
    ok = [r for r in reports if r.ok and r.critique is not None]
    if len(ok) != len(reports) or not ok:
        return False
    return all(r.critique.score == 5 and not r.critique.issues for r in ok)
