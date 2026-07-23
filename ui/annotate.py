"""Pure helper for rendering an output with inline, colour-coded annotations.

Kept dependency-free (no Streamlit import) so it can be unit-tested. Produces the
HTML for the main verdict view (Phase 4.1): each flagged span is wrapped in a
coloured marker — red for confirmed issues, amber for low-confidence/dismissed
flags, green for explicitly validated text — with the evidence chain exposed on
hover via the title attribute and a superscript index that ties back to the
evidence list rendered beneath it.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

# priority: higher wins when spans overlap
COLORS = {
    "confirmed": ("#e74c3c", "#fdecea", 3),   # red
    "flag": ("#e0a800", "#fef7e0", 2),        # amber / yellow
    "validated": ("#1e874b", "#e9f9ef", 1),   # green
}


@dataclass
class Annotation:
    quote: str
    kind: str          # confirmed | flag | validated
    evidence: str
    label: str = ""    # short label shown in the tooltip


def _find(text_lower: str, quote: str) -> tuple[int, int] | None:
    q = " ".join(quote.lower().split())
    if not q:
        return None
    # Try exact (whitespace-normalised) match first.
    idx = text_lower.find(q)
    if idx != -1:
        return idx, idx + len(q)
    # Fall back to the first distinctive chunk of the quote.
    chunk = q[:60]
    idx = text_lower.find(chunk)
    if idx != -1:
        return idx, idx + len(chunk)
    return None


def annotate_html(text: str, annotations: list[Annotation]) -> str:
    """Return HTML for `text` with non-overlapping coloured markers."""
    if not text:
        return "<em>(empty output)</em>"

    text_lower = " ".join(text.split()).lower()
    # We match against a whitespace-collapsed copy but render the original, so
    # rebuild an index map from collapsed positions to original positions.
    collapsed, index_map = _collapse_with_map(text)
    collapsed_lower = collapsed.lower()

    spans: list[tuple[int, int, Annotation]] = []
    for ann in annotations:
        found = _find(collapsed_lower, ann.quote)
        if found:
            spans.append((found[0], found[1], ann))

    # Resolve overlaps: sort by priority desc, then length desc; greedily keep.
    spans.sort(key=lambda s: (COLORS[s[2].kind][2], s[1] - s[0]), reverse=True)
    chosen: list[tuple[int, int, Annotation]] = []
    for start, end, ann in spans:
        if all(end <= cs or start >= ce for cs, ce, _ in chosen):
            chosen.append((start, end, ann))
    chosen.sort(key=lambda s: s[0])

    out: list[str] = []
    cursor = 0
    marker_no = 0
    for start, end, ann in chosen:
        orig_start = index_map[start]
        orig_end = index_map[min(end, len(index_map) - 1)]
        out.append(html.escape(text[cursor:orig_start]))
        marker_no += 1
        fg, bg, _ = COLORS[ann.kind]
        tooltip = html.escape(f"[{ann.label}] {ann.evidence}" if ann.label else ann.evidence)
        out.append(
            f'<mark title="{tooltip}" '
            f'style="background:{bg};border-bottom:2px solid {fg};'
            f'padding:0 2px;border-radius:3px;">'
            f"{html.escape(text[orig_start:orig_end])}"
            f'<sup style="color:{fg};font-weight:700;">{marker_no}</sup></mark>'
        )
        cursor = orig_end
    out.append(html.escape(text[cursor:]))
    rendered = "".join(out).replace("\n", "<br>")
    return (
        '<div style="line-height:1.9;font-size:1rem;padding:14px 16px;'
        'border:1px solid rgba(128,128,128,.25);border-radius:8px;">' + rendered + "</div>"
    )


def _collapse_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse runs of whitespace to single spaces, mapping back to originals."""
    collapsed_chars: list[str] = []
    index_map: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            collapsed_chars.append(" ")
            index_map.append(i)
            prev_space = True
        else:
            collapsed_chars.append(ch)
            index_map.append(i)
            prev_space = False
    index_map.append(len(text))
    return "".join(collapsed_chars), index_map


def legend_html() -> str:
    items = [
        ("#e74c3c", "#fdecea", "Confirmed issue"),
        ("#e0a800", "#fef7e0", "Low-confidence / dismissed flag"),
        ("#1e874b", "#e9f9ef", "Validated / clean"),
    ]
    chips = "".join(
        f'<span style="background:{bg};border-bottom:2px solid {fg};'
        f'padding:2px 8px;border-radius:4px;margin-right:10px;font-size:.85rem;">{label}</span>'
        for fg, bg, label in items
    )
    return f'<div style="margin:8px 0;">{chips}</div>'
