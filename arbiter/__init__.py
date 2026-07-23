"""LLM Output Arbitration System.

A multi-agent pipeline that routes any LLM-generated output to three specialised
critic models (factual accuracy, logical consistency, completeness), detects
where they disagree, and has an adjudicator synthesise a single confidence-scored
verdict with actionable callouts.
"""

from .config import Settings, get_settings
from .models import (
    ArbitrationResult,
    ConfidenceLevel,
    ConfirmedIssue,
    Critique,
    CriticReport,
    Dimension,
    Disagreement,
    DismissedFlag,
    Issue,
    Verdict,
)
from .pipeline import run_arbitration, run_batch

__version__ = "1.0.0"

__all__ = [
    "run_arbitration",
    "run_batch",
    "get_settings",
    "Settings",
    "ArbitrationResult",
    "Verdict",
    "Critique",
    "CriticReport",
    "Issue",
    "Dimension",
    "Disagreement",
    "ConfirmedIssue",
    "DismissedFlag",
    "ConfidenceLevel",
    "__version__",
]
