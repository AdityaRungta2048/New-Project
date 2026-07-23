"""Meta-analysis of critic behaviour across many arbitrations (Phase 5.2).

This is the "portfolio gold" layer: over a corpus of arbitrations it answers
which critic finds the most issues, which critic the adjudicator overrules most
often, which failure types recur, and how often the critics agree vs. disagree —
i.e. it quantifies the payoff of a multi-model panel over single-model
self-evaluation.
"""

from __future__ import annotations

from collections import Counter

from .config import Settings, get_settings
from .storage import Storage


def compute_analytics(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    storage = Storage(settings)

    with storage._connect() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM arbitrations").fetchone()["c"]
        if total == 0:
            return {"total_arbitrations": 0, "note": "No arbitrations recorded yet."}

        # Per-critic aggregates.
        rows = conn.execute(
            """SELECT dimension,
                      SUM(num_issues)    AS issues,
                      SUM(num_confirmed) AS confirmed,
                      SUM(num_dismissed) AS dismissed,
                      SUM(ok)            AS runs_ok,
                      COUNT(*)           AS runs_total,
                      AVG(score)         AS avg_score,
                      AVG(confidence)    AS avg_confidence
               FROM critic_reports GROUP BY dimension"""
        ).fetchall()

        agg = conn.execute(
            """SELECT
                 SUM(CASE WHEN num_disagreements > 0 THEN 1 ELSE 0 END) AS with_disagreement,
                 SUM(short_circuited) AS short_circuits,
                 SUM(degraded)        AS degraded,
                 AVG(quality_score)   AS avg_quality,
                 AVG(confidence)      AS avg_confidence,
                 SUM(num_confirmed)   AS total_confirmed,
                 SUM(num_dismissed)   AS total_dismissed
               FROM arbitrations"""
        ).fetchone()

        # Failure-type distribution, mined from the stored verdicts.
        verdict_rows = conn.execute("SELECT result_json FROM arbitrations").fetchall()

    per_critic = {}
    for r in rows:
        per_critic[r["dimension"]] = {
            "issues_found": r["issues"] or 0,
            "issues_confirmed": r["confirmed"] or 0,
            "flags_overruled": r["dismissed"] or 0,
            "runs_ok": r["runs_ok"] or 0,
            "runs_total": r["runs_total"] or 0,
            "avg_score": round(r["avg_score"], 2) if r["avg_score"] is not None else None,
            "avg_confidence": (
                round(r["avg_confidence"], 2) if r["avg_confidence"] is not None else None
            ),
            "overrule_rate": (
                round((r["dismissed"] or 0) / r["issues"], 3) if (r["issues"] or 0) else 0.0
            ),
        }

    def _top(metric: str):
        ranked = [(d, v[metric]) for d, v in per_critic.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[0][0] if ranked and ranked[0][1] else None

    # Failure taxonomy: bucket confirmed-issue problems by keyword.
    failure_types: Counter = Counter()
    import json as _json

    for vr in verdict_rows:
        data = _json.loads(vr["result_json"])
        for ci in data.get("verdict", {}).get("confirmed_issues", []):
            failure_types[_bucket(ci.get("problem", ""), ci.get("dimension", ""))] += 1

    with_dis = agg["with_disagreement"] or 0
    return {
        "total_arbitrations": total,
        "per_critic": per_critic,
        "most_issues_found_by": _top("issues_found"),
        "most_overruled_critic": _top("flags_overruled"),
        "agreement": {
            "arbitrations_with_disagreement": with_dis,
            "unanimous_or_aligned": total - with_dis,
            "disagreement_rate": round(with_dis / total, 3),
            "short_circuited_passes": agg["short_circuits"] or 0,
        },
        "adjudication": {
            "total_confirmed": agg["total_confirmed"] or 0,
            "total_dismissed": agg["total_dismissed"] or 0,
            "confirm_rate": _ratio(agg["total_confirmed"], agg["total_dismissed"]),
        },
        "common_failure_types": failure_types.most_common(8),
        "degraded_runs": agg["degraded"] or 0,
        "avg_quality_score": round(agg["avg_quality"], 2) if agg["avg_quality"] else None,
        "avg_confidence": round(agg["avg_confidence"], 2) if agg["avg_confidence"] else None,
    }


def _ratio(confirmed, dismissed) -> float:
    confirmed = confirmed or 0
    dismissed = dismissed or 0
    denom = confirmed + dismissed
    return round(confirmed / denom, 3) if denom else 0.0


_FAILURE_KEYWORDS = [
    ("factual error", ("factual", "wrong", "incorrect", "misattribut", "myth")),
    ("overstated certainty", ("certainty", "unsupported", "proof", "proven")),
    ("logical fallacy", ("fallacy", "popularity", "slippery", "circular", "ad hominem")),
    ("correlation/causation", ("correlation", "causation", "cause")),
    ("contradiction", ("contradict",)),
    ("coverage gap", ("coverage", "gap", "does not address", "missing", "parts")),
    ("superficial", ("superficial", "hand-wave", "hand-waves", "brief", "evasive")),
]


def _bucket(problem: str, dimension: str) -> str:
    low = problem.lower()
    for label, keywords in _FAILURE_KEYWORDS:
        if any(k in low for k in keywords):
            return label
    return f"{dimension} issue"
