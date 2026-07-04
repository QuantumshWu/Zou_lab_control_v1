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


def test_template_tick_comes_from_the_connected_device_not_a_constant():
    """The fireable tick is a DEVICE property: ``_resolve_probe_template(template, sequencer=dev)`` snaps
    the template to ``1e9 / dev.clock_hz`` -- read off the connected sequencer, exactly as confocal reads
    a device's resolution off the device.  A board reporting a DIFFERENT clock snaps to a DIFFERENT tick;
    only when NO device is passed (a GUI template preview) does it fall back to the config default.  Pins
    that the snap is not hardcoded to a module constant."""
    from Zou_lab_control.neutral_atom.operations.measurements.pulse_scan import (
        _resolve_probe_template, _hardware_tick_ns)
    from Zou_lab_control.neutral_atom.timing.pulse_table import DEFAULT_CLOCK_HZ

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template = os.path.join(here, "pulses", "mot_field_template.json")

    class _Board:
        def __init__(self, clock_hz): self.clock_hz = clock_hz

    for clock_hz in (50e6, 100e6, 25e6):
        assert _hardware_tick_ns(_Board(clock_hz)) == 1_000_000_000.0 / clock_hz
        state = _resolve_probe_template(template, sequencer=_Board(clock_hz))
        assert state.time_step_ns == 1_000_000_000.0 / clock_hz, "template tick must follow the board clock"
    # no device in scope -> the streamer-config default clock (single source), never a bare literal
    assert _hardware_tick_ns(None) == 1_000_000_000.0 / DEFAULT_CLOCK_HZ
