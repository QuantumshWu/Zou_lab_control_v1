"""Two mechanically-pinned invariants the scan/pulse flows depend on (api slot AND scan slot must
work, and a manually-configured logic node must behave like the one-click task):

1. The console signal picker ROUND-TRIPS a configured input even when that signal is not live yet
   (a node's own future output, or a not-yet-started producer).  A dropped input silently builds a
   node with an empty y-expression -> every scan point NaN -> an empty grid on Start.  Pinned so the
   tree/flat picker branches can never diverge again (only the flat branch used to keep a waiting name).

2. Every bundled FIREABLE pulse template authors on the HARDWARE clock grid: its ``time_step_ns``
   equals the one hardware tick (``1e9 / DEFAULT_CLOCK_HZ`` = 20 ns).  A finer authoring grid (the old
   ``time_step_ns = 1``) lets an api/scan DURATION sweep produce sub-tick durations that fail the
   50 MHz grid validation -> the sweep cannot fire ("api slot does not work").  Snap == fire == author.
"""

from __future__ import annotations

import glob
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_signal_picker_round_trips_a_not_yet_live_input():
    import pytest
    pytest.importorskip("PyQt5")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app, FluentTreeComboBox
    from Zou_lab_control.frontend.task_console import fill_grouped_signal_combo, read_editable_combo

    ensure_qt_app()
    combo = FluentTreeComboBox()
    # "frame_0" is NOT among the currently-live signals -> it must still be selectable + read back.
    fill_grouped_signal_combo(combo, names=["survival"], sources={"survival": ["temperature"]},
                              formats={"survival": "(7,)"}, current="frame_0")
    assert read_editable_combo(combo) == "frame_0", (
        "the picker dropped a configured input that is not a currently-live signal; a pulse-scan y "
        "referencing its own frame_0 (or a not-yet-running producer) would lose its binding on Start.")


def test_bundled_fireable_templates_author_on_the_hardware_clock_grid():
    from Zou_lab_control.neutral_atom.operations.measurements.pulse_scan import _resolve_probe_template
    from Zou_lab_control.neutral_atom.timing.pulse_table import DEFAULT_CLOCK_HZ

    tick_ns = 1_000_000_000.0 / DEFAULT_CLOCK_HZ            # the ONE hardware tick (single source)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates = sorted(glob.glob(os.path.join(here, "pulses", "*.json")))
    assert templates, "no bundled pulse templates found"
    for path in templates:
        try:
            state = _resolve_probe_template(path)
        except Exception:
            continue                                       # a non-pulse-table json (e.g. a saved program) -- skip
        assert state.time_step_ns == tick_ns, (
            f"{os.path.basename(path)} authors at time_step_ns={state.time_step_ns} ns, not the hardware "
            f"tick {tick_ns} ns; a DURATION api/scan sweep of it produces off-grid values that cannot fire.")
