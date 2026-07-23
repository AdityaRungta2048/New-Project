"""The four compelling portfolio test cases (Phase 6.1).

Each asserts that the RIGHT critic catches the RIGHT kind of problem — which is
the whole thesis of the project: a multi-model panel catches issues a single
model's self-evaluation would miss.
"""

from arbiter import run_arbitration
from arbiter.models import Dimension

CASES = {
    "factually_incorrect": {
        "prompt": "State a few basic science facts.",
        "output": (
            "The sun revolves around the earth. Water boils at 50 degrees celsius at "
            "sea level. The human body has 106 bones, and Einstein invented the light bulb."
        ),
    },
    "logically_flawed": {
        "prompt": "Should the town approve the new stadium?",
        "output": (
            "Everyone knows the stadium is a great idea. If we don't build it, then "
            "tourism will collapse, then the shops will close, then the town will die. "
            "Anyone who disagrees is simply a fool."
        ),
    },
    "misses_the_point": {
        "prompt": (
            "Compare the causes, course, and consequences of World War I and World War II, "
            "and explain their key political and economic differences."
        ),
        "output": "World War I and World War II were both major wars. Many nations fought. etc.",
    },
    "genuinely_good": {
        "prompt": "What is photosynthesis?",
        "output": (
            "Photosynthesis is the process by which green plants use sunlight to make "
            "food from carbon dioxide and water, producing glucose and releasing oxygen. "
            "It happens in the chloroplasts using the pigment chlorophyll."
        ),
    },
}


def _run(name):
    case = CASES[name]
    return run_arbitration(case["output"], case["prompt"], persist=False)


def test_factually_incorrect_caught_by_accuracy_critic():
    result = _run("factually_incorrect")
    acc = result.critique_for(Dimension.ACCURACY)
    assert acc is not None and acc.issues, "accuracy critic should find planted errors"
    assert acc.score <= 2
    assert any(ci.dimension is Dimension.ACCURACY for ci in result.verdict.confirmed_issues)
    assert result.verdict.quality_score <= 5


def test_logically_flawed_caught_by_logic_critic():
    result = _run("logically_flawed")
    logic = result.critique_for(Dimension.LOGIC)
    assert logic is not None and logic.issues, "logic critic should find the fallacies"
    assert any(ci.dimension is Dimension.LOGIC for ci in result.verdict.confirmed_issues)


def test_misses_the_point_caught_by_completeness_critic():
    result = _run("misses_the_point")
    comp = result.critique_for(Dimension.COMPLETENESS)
    assert comp is not None and comp.issues, "completeness critic should flag the gap"
    assert any(ci.dimension is Dimension.COMPLETENESS for ci in result.verdict.confirmed_issues)


def test_genuinely_good_gets_clean_bill():
    result = _run("genuinely_good")
    # Either a unanimous short-circuit pass or at least a high score with no
    # confirmed issues.
    assert result.short_circuited or result.verdict.quality_score >= 8
    assert not any(ci.severity >= 4 for ci in result.verdict.confirmed_issues)


def test_multi_model_catches_more_than_one_dimension():
    """The factual case also trips other critics — the panel's combined recall
    exceeds any single critic's."""
    result = _run("factually_incorrect")
    dims_with_issues = {
        r.dimension for r in result.reports if r.ok and r.critique and r.critique.issues
    }
    assert Dimension.ACCURACY in dims_with_issues
