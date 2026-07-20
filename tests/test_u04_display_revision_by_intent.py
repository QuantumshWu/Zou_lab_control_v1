"""A panel's display revision follows its own intent, not its position.

`_build_live_configuration` stamped each panel's presentation identity with a
display revision chosen by panel INDEX - 0 got the image revision, 1 the curve,
2 the histogram, anything else 0.  That is only correct while the board is
frozen to one topology in one order, which is exactly the constraint S4-1c(b)
exists to remove.  The moment panels may be reordered or mixed, a CURVE sitting
at index 0 would be stamped with the image's revision: editing the curve's
display state would not bump its identity, and the front would go stale without
anything noticing.

Keying on the panel's own view intent is position-independent, so opening up the
topology cannot reintroduce that class of stale front.  This test pins the
mapping directly, including the orders the current validators still forbid.
"""

from __future__ import annotations

from pathlib import Path

from zlc_frontend.figure.model import ViewIntent
from zlc_workbench.live import _display_revisions_by_intent


ROOT = Path(__file__).resolve().parents[1]


class _State:
    """The one thing the mapping reads off a display state."""

    def __init__(self, revision: int) -> None:
        self.revision = revision


def test_each_intent_maps_to_its_own_revision():
    revisions = _display_revisions_by_intent(_State(7), _State(11), _State(13))

    assert revisions[ViewIntent.IMAGE] == 7
    assert revisions[ViewIntent.CURVE] == 11
    assert revisions[ViewIntent.HISTOGRAM] == 13


def test_the_answer_does_not_depend_on_any_panel_order():
    """The same lookup serves a board whose panels are in any arrangement."""

    revisions = _display_revisions_by_intent(_State(1), _State(2), _State(3))

    # A CURVE first and an IMAGE third must still get 2 and 1 respectively -
    # under the old index ladder they would have got 1 and 3.
    reordered = (ViewIntent.CURVE, ViewIntent.HISTOGRAM, ViewIntent.IMAGE)
    assert [revisions[intent] for intent in reordered] == [2, 3, 1]


def test_an_absent_display_state_contributes_no_entry():
    """A raw-only board has no curve or histogram state and must not invent one."""

    revisions = _display_revisions_by_intent(_State(4), None, None)

    assert revisions == {ViewIntent.IMAGE: 4}
    # Callers fall back to 0 for an intent with no state, which is what a panel
    # that carries no editable display parameter should report.
    assert revisions.get(ViewIntent.METER, 0) == 0
    assert revisions.get(ViewIntent.CURVE, 0) == 0


def test_no_positional_display_revision_ladder_survives():
    """Mechanical: the index-keyed ladder must not come back."""

    source = (ROOT / "zlc_workbench" / "live.py").read_text(encoding="utf-8")
    assert "if index == 0 and image_display" not in source
    assert "if index == 1 and curve_display" not in source
    assert "if index == 2 and histogram_display" not in source
    assert source.count("def _display_revisions_by_intent(") == 1
    assert "display_revisions.get(" in source
