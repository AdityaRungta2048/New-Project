"""Shared graph state for the LangGraph orchestration.

`reports` uses an additive reducer so the three critic nodes can write to it
concurrently (parallel fan-out) and have their results merged on fan-in rather
than overwriting one another.
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from .models import CriticReport, Disagreement, Verdict


class ArbitrationState(TypedDict, total=False):
    output_text: str
    original_prompt: Optional[str]
    # Additive reducer -> parallel critic writes merge instead of clobbering.
    reports: Annotated[list[CriticReport], operator.add]
    disagreements: list[Disagreement]
    verdict: Optional[Verdict]
    short_circuited: bool
    degraded: bool
