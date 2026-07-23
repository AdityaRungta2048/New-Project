"""Deterministic heuristic critique engine that powers the `mock` backend.

This is NOT a real LLM. It is a rule-based stand-in so the arbitration pipeline
runs end-to-end, deterministically, with no API keys — which makes the whole
system reviewable and testable offline. Each dimension has its own detectors,
so the mock produces genuinely *different* critiques per critic (accuracy finds
factual errors, logic finds fallacies, completeness finds gaps). Real backends
replace this with GPT-4o / Claude / Llama.

Each detector returns (quote, problem, severity, soft) where `soft=True` marks a
weak/heuristic signal that the adjudicator may reasonably overrule.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..models import Critique, Dimension, Issue

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these",
    "those", "it", "its", "as", "at", "by", "from", "how", "what", "why", "when",
    "which", "who", "whom", "you", "your", "please", "can", "could", "would",
    "should", "do", "does", "explain", "describe", "list", "give", "provide",
    "tell", "about", "into", "over", "than", "then", "also", "some", "any", "all",
    "each", "both", "between", "will", "shall", "may", "might", "must", "not",
}


@dataclass
class _Finding:
    quote: str
    problem: str
    severity: int
    soft: bool = False


def _find_quote(text: str, match: re.Match) -> str:
    """Return the sentence containing a regex match, for a natural quote."""
    start, end = match.span()
    left = text.rfind(".", 0, start)
    left = 0 if left == -1 else left + 1
    right = text.find(".", end)
    right = len(text) if right == -1 else right + 1
    return text[left:right].strip() or match.group(0)


# ---------------------------------------------------------------------------
# ACCURACY detectors
# ---------------------------------------------------------------------------
# Curated common-knowledge falsehoods — useful for planted-error demos.
_FALSE_CLAIMS = [
    (r"sun\s+(revolves|orbits|goes|rotates)\s+around\s+the\s+earth",
     "Geocentric error: the Earth orbits the Sun, not the other way around.", 5),
    (r"earth\s+is\s+the\s+(cent(er|re))\s+of\s+the\s+(solar\s+system|universe)",
     "The Earth is not the centre of the solar system or universe.", 5),
    (r"great\s+wall\s+of\s+china\s+is\s+visible\s+from\s+(space|the\s+moon)",
     "Common myth: the Great Wall is not visible from space with the naked eye.", 3),
    (r"einstein\s+(invented|discovered)\s+(the\s+)?(light\s*bulb|telephone|gravity)",
     "Misattribution: Einstein did not invent this.", 4),
    (r"humans?\s+(only\s+)?use\s+10\s*%\s+of\s+(their|the)\s+brain",
     "Debunked myth: humans use virtually all of their brain.", 3),
    (r"goldfish\s+have\s+a\s+(three|3)[\s-]*second\s+memory",
     "Myth: goldfish memory spans months, not seconds.", 2),
    (r"lightning\s+never\s+strikes\s+the\s+same\s+place\s+twice",
     "False: lightning frequently strikes the same place repeatedly.", 2),
]

# Numeric fact checks: (regex with a number group, expected value, tolerance, label)
_NUMERIC_FACTS = [
    (r"water\s+boils\s+at\s+(\d+(?:\.\d+)?)\s*°?\s*(?:degrees\s*)?(?:c\b|celsius|centigrade)",
     100.0, 1.0, "Water boils at 100 °C at sea level"),
    (r"(?:human|adult|the)\s+(?:body|skeleton)\s+has\s+(\d+)\s+bones",
     206.0, 3.0, "An adult human body has 206 bones"),
    (r"there\s+are\s+(\d+)\s+planets\s+in\s+(?:our|the)\s+solar\s+system",
     8.0, 0.0, "There are 8 planets in the solar system"),
    (r"there\s+are\s+(\d+)\s+continents",
     7.0, 0.0, "There are 7 continents"),
    (r"speed\s+of\s+light\s+is\s+(?:about\s+)?(\d+(?:,\d{3})*)\s*km",
     299792.0, 5000.0, "The speed of light is ~299,792 km/s"),
]


def detect_accuracy(text: str) -> list[_Finding]:
    findings: list[_Finding] = []
    low = text.lower()
    for pattern, problem, severity in _FALSE_CLAIMS:
        m = re.search(pattern, low)
        if m:
            findings.append(_Finding(_find_quote(text, m), problem, severity))
    for pattern, expected, tol, label in _NUMERIC_FACTS:
        m = re.search(pattern, low)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                value = float(raw)
            except ValueError:
                continue
            if abs(value - expected) > tol:
                findings.append(
                    _Finding(
                        _find_quote(text, m),
                        f"Factual error: the text states {value:g} but {label}.",
                        4,
                    )
                )
    # Over-certain unsupported claims (soft signal).
    for m in re.finditer(r"(studies\s+prove|it\s+is\s+a\s+proven\s+fact|100%\s+guaranteed|"
                         r"scientists\s+have\s+proven)", low):
        findings.append(
            _Finding(
                _find_quote(text, m),
                "Overstated certainty: sweeping claim of proof without a cited source.",
                2,
                soft=True,
            )
        )
    return findings


# Curated common-knowledge TRUTHS — lets the accuracy critic positively
# validate checkable claims (green markers in the UI), not just find errors.
_TRUE_CLAIMS = [
    (r"paris\s+is\s+the\s+capital\s+of\s+france", "Paris is indeed the capital of France."),
    (r"water\s+(is\s+made\s+of|consists\s+of|is)\s+h2?o|chemical\s+formula\s+(for\s+)?water\s+is\s+h2?o",
     "Water's chemical formula is H2O — correct."),
    (r"earth\s+(orbits|revolves\s+around|goes\s+around)\s+the\s+sun",
     "The Earth orbits the Sun — correct heliocentric model."),
    (r"speed\s+of\s+light\s+is\s+(?:about\s+|approximately\s+)?299[,.]?792",
     "The speed of light is ~299,792 km/s — correct."),
    (r"(human|adult)\s+(body|skeleton)\s+has\s+206\s+bones",
     "An adult human body has 206 bones — correct."),
    (r"photosynthesis\s+.*(sunlight|light).*(glucose|sugar|oxygen)",
     "Accurate description of photosynthesis."),
]


def detect_validated(text: str) -> list[tuple[str, str]]:
    """Return (quote, note) for checkable claims that are correct."""
    low = text.lower()
    out = []
    for pattern, note in _TRUE_CLAIMS:
        m = re.search(pattern, low)
        if m:
            out.append((_find_quote(text, m), note))
    return out


# ---------------------------------------------------------------------------
# LOGIC detectors
# ---------------------------------------------------------------------------
_FALLACIES = [
    (r"everyone\s+(knows|agrees)|everybody\s+(knows|agrees)|most\s+people\s+(agree|believe|know)"
     r"|it'?s\s+common\s+knowledge",
     "Appeal to popularity: something is not true merely because many believe it.", 3, False),
    (r"if\s+we\s+(allow|permit|let|start)\b.*\b(then|will\s+lead|inevitably|end\s+up)",
     "Slippery slope: chains an extreme outcome to a first step without justification.", 3, False),
    (r"only\s+an?\s+(idiot|fool|moron)|anyone\s+who\s+disagrees\s+is",
     "Ad hominem: attacks the person or dismisses dissent instead of the argument.", 3, False),
    (r"correlat\w*\b.*\bcaus|caus\w*\b.*\bcorrelat",
     "Correlation/causation: treats a correlation as proof of causation.", 3, False),
    (r"\b(this|it|the\s+statement)\s+is\s+true\s+because\b.*\b(true|correct|right)\b",
     "Circular reasoning: the conclusion is used as its own premise.", 3, False),
    (r"\b(always|never|all|none|every|no\s+one|everyone)\b",
     "Sweeping generalisation: an absolute claim that a single exception would break.", 2, True),
]


def detect_logic(text: str) -> list[_Finding]:
    findings: list[_Finding] = []
    low = text.lower()
    seen_spans: list[tuple[int, int]] = []
    for pattern, problem, severity, soft in _FALLACIES:
        m = re.search(pattern, low)
        if not m:
            continue
        # Avoid double-flagging the same sentence for the soft catch-all.
        span = m.span()
        if soft and any(abs(span[0] - s[0]) < 40 for s in seen_spans):
            continue
        seen_spans.append(span)
        findings.append(_Finding(_find_quote(text, m), problem, severity, soft))
    # Explicit contradiction: same subject asserted and negated.
    if re.search(r"\bis\s+safe\b", low) and re.search(r"\bis\s+(not\s+safe|dangerous|unsafe)\b", low):
        m = re.search(r"\bis\s+(not\s+safe|dangerous|unsafe)\b", low)
        findings.append(
            _Finding(_find_quote(text, m),
                     "Internal contradiction: the text both affirms and denies the same claim.", 4)
        )
    return findings


# ---------------------------------------------------------------------------
# COMPLETENESS detectors
# ---------------------------------------------------------------------------
_HANDWAVE = [
    r"\betc\.?\b", r"and\s+so\s+on", r"among\s+others", r"\.\.\.",
    r"left\s+as\s+an\s+exercise", r"beyond\s+the\s+scope",
    r"i\s+won'?t\s+go\s+into", r"too\s+(complex|complicated)\s+to\s+explain",
]


def _keywords(prompt: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-']{3,}", prompt.lower())
    return [w for w in words if w not in _STOPWORDS]


def _split_asks(prompt: str) -> list[str]:
    """Very rough split of a prompt into its distinct sub-asks."""
    # Numbered / bulleted parts first.
    parts = re.split(r"(?:\n\s*(?:\d+[.)]|[-*])\s*)|(?:\?\s+)|(?:;\s+)|(?:\band\s+also\b)", prompt)
    parts = [p.strip(" .\n") for p in parts if p and len(p.strip()) > 8]
    return parts or [prompt.strip()]


def detect_completeness(text: str, original_prompt: Optional[str]) -> list[_Finding]:
    findings: list[_Finding] = []
    low = text.lower()

    # Hand-waving / superficial markers.
    for pattern in _HANDWAVE:
        m = re.search(pattern, low)
        if m:
            findings.append(
                _Finding(_find_quote(text, m),
                         "Superficial coverage: hand-waves over detail instead of addressing it.",
                         2, soft=True)
            )
            break

    if original_prompt:
        asks = _split_asks(original_prompt)
        # Coverage by keyword overlap.
        kws = set(_keywords(original_prompt))
        missing = sorted(k for k in kws if k not in low)
        # Only treat as a gap when a meaningful share of prompt keywords is absent.
        if kws and len(missing) / max(1, len(kws)) > 0.5 and len(missing) >= 2:
            sample = ", ".join(missing[:6])
            findings.append(
                _Finding(
                    original_prompt.strip()[:160],
                    f"Coverage gap: the response does not address key parts of the prompt "
                    f"(e.g. {sample}).",
                    3,
                )
            )
        # Multi-part question but very short answer.
        if len(asks) >= 2 and len(text.split()) < 25 * len(asks) // 2:
            findings.append(
                _Finding(
                    text.strip()[:160] or original_prompt.strip()[:160],
                    f"The prompt has ~{len(asks)} parts but the response is too brief to "
                    "cover them all.",
                    3,
                )
            )
    else:
        # No prompt: only flag obvious dodging.
        if re.search(r"\bit\s+depends\b", low) and len(text.split()) < 40:
            m = re.search(r"\bit\s+depends\b", low)
            findings.append(
                _Finding(_find_quote(text, m),
                         "Evasive: says 'it depends' without explaining the factors involved.",
                         2, soft=True)
            )
    return findings


# ---------------------------------------------------------------------------
# Assembly: findings -> Critique
# ---------------------------------------------------------------------------
_DETECTORS = {
    Dimension.ACCURACY: lambda text, prompt: detect_accuracy(text),
    Dimension.LOGIC: lambda text, prompt: detect_logic(text),
    Dimension.COMPLETENESS: lambda text, prompt: detect_completeness(text, prompt),
}


def _score_from_findings(findings: list[_Finding]) -> int:
    if not findings:
        return 5
    penalty = sum(0.55 + 0.35 * (f.severity - 1) for f in findings)
    if any(f.severity >= 5 for f in findings):
        return max(1, min(2, round(5 - penalty)))
    return max(1, round(5 - penalty))


def _confidence_from_findings(findings: list[_Finding]) -> float:
    base = 0.86 if not findings else 0.8
    if any(f.soft for f in findings):
        base -= 0.12
    # Strong, high-severity findings raise confidence slightly.
    if any(f.severity >= 4 and not f.soft for f in findings):
        base += 0.08
    return round(max(0.4, min(0.95, base)), 2)


def build_critique(
    dimension: Dimension, output_text: str, original_prompt: Optional[str]
) -> Critique:
    findings = _DETECTORS[dimension](output_text, original_prompt)
    issues = [
        Issue(quote=f.quote, problem=f.problem, severity=f.severity) for f in findings
    ]
    score = _score_from_findings(findings)
    confidence = _confidence_from_findings(findings)
    if issues:
        summary = (
            f"Found {len(issues)} {dimension.value} issue(s); most severe is "
            f"severity {max(f.severity for f in findings)}."
        )
    else:
        summary = f"No {dimension.value} problems detected; the output looks sound on this dimension."
    return Critique(
        dimension=dimension,
        score=score,
        issues=issues,
        confidence=confidence,
        summary=summary,
    )


# Expose which findings were "soft" so the mock adjudicator can mirror an LLM's
# tendency to overrule weak flags. Keyed by (dimension, quote).
def soft_quotes(dimension: Dimension, output_text: str, original_prompt: Optional[str]) -> set[str]:
    findings = _DETECTORS[dimension](output_text, original_prompt)
    return {f.quote for f in findings if f.soft}
