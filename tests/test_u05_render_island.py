"""The console reaches the legacy render worker only through the isolation island.

The design document allows an un-migrated shared-Figure RenderLoop to survive in
exactly one place - line 596: it "only" gets to live inside the
``SerializedLegacyAggBridge`` island, and its presence there may never be read as
having satisfied the current lane contract.  The island was built and then never
wired: until this slice the console constructed a RenderLoop directly and called
barrier/busy/submit/stop on it raw, so the mandate was prose rather than fact.

Three of those barrier calls ignored the result.  A barrier that times out means
the render worker may still own the Figure, so ignoring it let the GUI thread walk
into a Figure it did not own - the exact hazard a fail-closed island exists to
prevent.  Those paths now abort, which is a deviation from ``main``'s behaviour
under timeout and is registered in the section 2.2 ledger as
``LEDGERED_PENDING_APPROVAL``; the design document (line 129) reserves that route
for safety-driven deviations precisely so the work is not blocked on approval.

There is no other active coverage of this: ``test_render_loop_contract.py`` and
``test_figure_viewer.py`` both drive ``_render_loop`` directly and both sit
outside ``tests/migration_active_tests.txt``, so they are frozen - not run, not
modified, and not evidence for this branch.  This file is the oracle.
"""

from __future__ import annotations

#: C41 -- this oracle guards a legacy artifact and is deleted in the same commit that
#: deletes it (swept by test_design_charter).
DIES_WITH = 'Zou_lab_control/frontend/task_console.py'

import pytest

from Zou_lab_control.neutral_atom.core.signals import SignalHub
from zlc_workbench.legacy import LegacyHandoffTimeout, SerializedLegacyAggBridge


def _console():
    from zlc_frontend.qt_widgets import ensure_qt_app

    ensure_qt_app()
    from Zou_lab_control.frontend.task_console import TaskConsole

    return TaskConsole(hub=SignalHub())


def test_the_console_owns_a_bridge_and_never_the_raw_loop():
    """Containment as a fact about the built object, not about the source text."""

    console = _console()
    assert isinstance(console._render, SerializedLegacyAggBridge)
    # The old attribute must be gone rather than merely unused: a lingering raw
    # reference is how a future edit would quietly bypass the island again.
    assert not hasattr(console, "_render_loop")


def test_a_refused_handoff_becomes_False_rather_than_an_exception():
    """The panels were handed a bool-returning barrier; that contract still holds.

    The bridge raises, so exactly one place may translate.  If this leaked, a
    timeout would surface as an unhandled exception inside a Qt slot.
    """

    console = _console()

    def refuse(_timeout=5.0):
        raise LegacyHandoffTimeout("refused")

    console._render.settle = refuse
    assert console._render_barrier() is False
    assert console._render_barrier(0.0) is False       # zero is a legal budget


def test_a_refused_handoff_aborts_the_operation_and_says_so():
    """The behaviour change this slice is for.

    Before, a refused barrier fell through and the GUI touched the figures anyway.
    Now the action is abandoned and the operator is told which one, on the
    permanent status strip rather than in a popup nobody kept.
    """

    console = _console()
    seen = []
    console.status_strip.show_message = lambda text, severity="info": seen.append((severity, text))

    def refuse(_timeout=5.0):
        raise LegacyHandoffTimeout("refused")

    console._render.settle = refuse
    console.refresh_once()

    assert seen, "a refused handoff must be reported, not swallowed"
    severity, text = seen[-1]
    assert severity == "error"
    assert "Refresh" in text


def test_zero_is_a_legal_handoff_budget():
    """A spent teardown budget means "do not wait", not "invalid argument".

    ``positive_real`` rejected 0.0, which would have forced the panel-teardown path
    to keep a raw loop reference and bypass the bridge - defeating the containment
    this file exists to prove.
    """

    class _Loop:
        busy = False

        def submit(self, job):
            return True

        def barrier(self, timeout=5.0):
            return True

        def stop(self, timeout):
            return True

    bridge = SerializedLegacyAggBridge(_Loop())
    bridge.settle(0.0)
    with pytest.raises(ValueError):
        bridge.settle(-1.0)
