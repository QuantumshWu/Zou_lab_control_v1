"""A configured signal binding survives while its producer is not live yet.

The console signal picker ROUND-TRIPS a configured input even when that signal is not live yet
   (for example a PulseScan y from a not-yet-started producer).  A dropped input silently builds a
   node with an empty y-expression -> every scan point NaN -> an empty grid on Start.  Pinned so the
   tree/flat picker branches can never diverge again (only the flat branch used to keep a waiting name).

Current pulse assets, semantic API/scan parameters, and hardware-grid rejection are owned by the
``zlc_pulse`` document/compiler tests.  This legacy frontend test does not parse current assets with
the historical timing loader.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_signal_picker_round_trips_a_not_yet_live_input():
    import pytest
    pytest.importorskip("PyQt5")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app, FluentTreeComboBox
    from Zou_lab_control.frontend.param_widgets import fill_grouped_signal_combo, read_editable_combo

    ensure_qt_app()
    combo = FluentTreeComboBox()
    # "frame_0" is NOT among the currently-live signals -> it must still be selectable + read back.
    fill_grouped_signal_combo(combo, names=["survival"], sources={"survival": ["temperature"]},
                              formats={"survival": "(7,)"}, current="frame_0")
    assert read_editable_combo(combo) == "frame_0", (
        "the picker dropped a configured input that is not a currently-live signal; a PulseScan y "
        "referencing a not-yet-running external producer would lose its binding on Start.")
