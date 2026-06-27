"""#H3s-F7 backend contract: ``PulseScanMeasurement.build`` dispatches on ``scan_mode``.

ONE ``scan_code`` (the active mode's program) + a ``scan_mode`` of "scan" | "api" | "none":
  * "scan" -> the hardware scan slots, SNAPPED to the FPGA grid;
  * "api"  -> the software api sweep, NO snap (native units);
  * "none" -> a single fixed point (no sweep).
A legacy spec without ``scan_mode`` is inferred once on load (no dual runtime path).

These exercise ``build`` directly (no node run) so they stay fast; the full node-driven sweeps live
in test_pulse_param_scan.py.  Single source for the mode names: pulse_scan.SCAN_MODES.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.operations.measurements.pulse_scan import (
    SCAN_MODES,
    SCAN_MODE_API,
    SCAN_MODE_NONE,
    SCAN_MODE_SCAN,
    _resolve_scan_mode,
)


def _safe_unlink(path) -> None:
    import gc
    p = Path(path)
    for _ in range(3):
        try:
            p.unlink(missing_ok=True)
            return
        except PermissionError:
            gc.collect()


def _calibrated():
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (48, 60)})
    exp.readout.sitemap(method="box", frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    return exp


def _probe_with_scan_slot() -> str:
    """A probe with ONE hardware scan slot bound to its image duration."""
    from Zou_lab_control.neutral_atom.timing import PulseTableState
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod
    state = PulseTableState(channels=["trap", "cooling", "probe", "emCCD"],
                            visible_channels=["trap", "cooling", "probe", "emCCD"])
    state.periods.append(PulsePeriod(0.002, (1, 1, 0, 0), unit="s", name="load"))
    state.periods.append(PulsePeriod(0.005, (1, 0, 1, 1), unit="s", name="image"))
    state.bind_field("duration", "1", label="probe", unit="us")
    state.validate()
    out = Path("pulses") / "_test_build_modes_scan.json"
    state.save(str(out))
    return str(out)


def _probe_with_api_slot() -> str:
    """A probe with ONE API slot (a duration handle) and NO scan slot."""
    from Zou_lab_control.neutral_atom.timing import PulseTableState
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod
    state = PulseTableState(channels=["trap", "cooling", "probe", "emCCD"],
                            visible_channels=["trap", "cooling", "probe", "emCCD"])
    state.periods.append(PulsePeriod(0.002, (1, 1, 0, 0), unit="s", name="load"))
    state.periods.append(PulsePeriod(0.005, (1, 0, 1, 1), unit="s", name="image"))
    state.bind_api_field("duration", "1", unit="us")
    state.validate()
    out = Path("pulses") / "_test_build_modes_api.json"
    state.save(str(out))
    return str(out)


def _spec(exp):
    return {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]


def test_scan_mode_snaps_hardware_scan_table():
    """scan_mode='scan' takes the hardware path: the scan-slot column is SNAPPED to integer clock
    ticks (the same snap the pulse GUI + server use)."""
    exp = _calibrated()
    probe = _probe_with_scan_slot()
    try:
        plan = _spec(exp).build(
            template=probe,
            pulse_slots={"api": {}, "scan_mode": SCAN_MODE_SCAN,
                         "scan_code": "import numpy as np\n"
                         "scan_table = np.linspace(10000, 40000, 5).reshape(-1, 1)"})
        assert plan.scan_names == ["s0"] and plan.api_names == []
        arr = plan.scan_arrays[0]
        assert arr.size == 5
        assert np.allclose(arr, np.round(arr))          # snapped to whole ticks
    finally:
        _safe_unlink(probe)
        exp.close()


def test_api_mode_sweeps_api_slots_without_snap():
    """scan_mode='api' takes the software path: the api column is the native float sweep (NO snap),
    using the SAME single scan_code program."""
    exp = _calibrated()
    probe = _probe_with_api_slot()
    try:
        plan = _spec(exp).build(
            template=probe,
            pulse_slots={"api": {}, "scan_mode": SCAN_MODE_API,
                         "scan_code": "import numpy as np\n"
                         "scan_table = np.linspace(2.5, 8.5, 4).reshape(-1, 1)"})
        assert plan.api_names == ["a1"] and plan.scan_names == []
        assert np.allclose(plan.api_arrays[0], [2.5, 4.5, 6.5, 8.5])   # native, un-snapped
    finally:
        _safe_unlink(probe)
        exp.close()


def test_none_mode_builds_single_fixed_point():
    """scan_mode='none' fires a single point at the fixed api values -- no sweep at all (empty scan
    + api arrays; the plan's axis is a lone point)."""
    exp = _calibrated()
    probe = _probe_with_scan_slot()
    try:
        plan = _spec(exp).build(
            template=probe, pulse_slots={"api": {}, "scan_mode": SCAN_MODE_NONE})
        assert plan.scan_arrays == [] and plan.api_arrays == []
        assert plan.axis.values.size == 1               # one fixed point
    finally:
        _safe_unlink(probe)
        exp.close()


def test_legacy_spec_without_scan_mode_infers_scan():
    """A legacy saved task (no scan_mode) with a scan program + bound scan slots infers 'scan' and
    builds the hardware (snapped) path -- old tasks stay runnable."""
    exp = _calibrated()
    probe = _probe_with_scan_slot()
    try:
        plan = _spec(exp).build(
            template=probe,
            pulse_slots={"api": {},   # NO scan_mode key
                         "scan_code": "import numpy as np\n"
                         "scan_table = np.linspace(10000, 30000, 3).reshape(-1, 1)"})
        assert plan.scan_names == ["s0"]
        assert plan.scan_arrays[0].size == 3
    finally:
        _safe_unlink(probe)
        exp.close()


def test_legacy_api_scan_key_infers_api():
    """A legacy task that carried the OLD separate ``api_scan`` program (and an api-only template)
    infers 'api' mode."""
    exp = _calibrated()
    probe = _probe_with_api_slot()
    try:
        from Zou_lab_control.neutral_atom.timing import PulseTableState
        state = PulseTableState.load(probe)
        # the inference helper reads scan_mode/scan_code/api_scan + the template's slots
        assert _resolve_scan_mode({"api_scan": "scan_table = [[2.0]]"}, state) == SCAN_MODE_API
        assert _resolve_scan_mode({}, state) == SCAN_MODE_NONE       # api slots, no program -> none
    finally:
        _safe_unlink(probe)
        exp.close()


def test_scan_modes_single_source():
    """The mode names live in ONE tuple shared by widget + build + tests."""
    assert SCAN_MODES == (SCAN_MODE_NONE, SCAN_MODE_API, SCAN_MODE_SCAN)
    assert SCAN_MODES == ("none", "api", "scan")
