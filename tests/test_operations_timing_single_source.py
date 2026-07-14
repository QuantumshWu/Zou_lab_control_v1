"""MECHANICAL guards for single-source contracts across operations / timing / subsystems
(AGENTS.md §2: the same fact is typed exactly once; a second spelling drifts into a bug).

Each test pins one rule:

* a consumer's default camera-frame input derives from the ONE camera signal-naming
  source (``camera_frame_keys`` -> ``FRAME_0``), never a producer-prefixed guess;
* the shipped probe-template path is typed once, in the timing layer beside its factory;
* tick tie-rounding has ONE rule (ties away from zero, ``sequence.round_ticks``) shared
  by the seconds-domain and ns-domain snap paths -- the same user duration lands on the
  same hardware tick whichever way it enters;
* ``PulseSequence``'s JSON schema identity lives on the class (like its sister
  ``PulseTableState``) and payload dispatch reads it from there;
* the "pulse must be a PulseController" gate and the detection-times rule each have
  exactly one spelling.
"""

from __future__ import annotations

from conftest import raw_device_set

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control

_NA_PKG = Path(Zou_lab_control.__file__).resolve().parent / "neutral_atom"


# ------------------------------------------------------------------ default frame input
def test_default_frame_input_derives_from_camera_frame_keys():
    """The bare default frame name is DERIVED (``FRAME_0 = camera_frame_keys(1)[0]``) and
    every consumer default references it: a single camera instance publishes the bare
    ``frame_0`` (the signal namespace is the instance; a device identity is never baked
    into a name), so a consumer default assuming a producer prefix would wait forever."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import FRAME_0, camera_frame_keys
    from Zou_lab_control.neutral_atom.operations.processors.analysis import analysis
    from Zou_lab_control.neutral_atom.operations.processors.roi import RoiProcessor

    assert FRAME_0 == camera_frame_keys(1)[0]          # derived, not a retyped literal
    # empty pick -> the node falls back to the ONE bare name (a publishable signal)
    node = RoiProcessor(SignalHub())
    assert tuple(node.consumes) == (FRAME_0,)
    spec = analysis(object())                        # params need no live readout
    decl = next(p for p in spec.params if p.key == "source")
    assert decl.default["inputs"] == [FRAME_0]


# ------------------------------------------------------------------ probe-template path
def test_probe_template_path_is_typed_once_in_timing():
    """The shipped single-image probe program's path lives ONCE, in the timing layer
    beside its in-memory factory; both defaulting measurements alias that object."""
    from Zou_lab_control.neutral_atom.operations.measurements import pulse_scan, readout_duration
    from Zou_lab_control.neutral_atom.timing import PROBE_TEMPLATE_PATH

    assert pulse_scan.DEFAULT_PROBE_TEMPLATE is PROBE_TEMPLATE_PATH
    assert readout_duration.DEFAULT_IMAGING_TEMPLATE is PROBE_TEMPLATE_PATH
    # the quoted literal itself appears only beside the factory (docstrings use ``..``)
    needle = '"' + PROBE_TEMPLATE_PATH + '"'
    offenders = sorted(
        str(f.relative_to(_NA_PKG))
        for f in _NA_PKG.rglob("*.py")
        if "__pycache__" not in f.parts and needle in f.read_text(encoding="utf-8"))
    assert offenders == [str(Path("timing") / "pulse_table.py")], offenders


# ------------------------------------------------------------------ tick tie rounding
def test_tick_tie_rounding_is_one_rule_on_both_snap_paths():
    """Exact .5-tick ties land on the SAME hardware tick whether the duration enters via
    the seconds-domain snap (a notebook scan axis, ``snap_seconds_to_clock``) or the
    ns-domain quantizer (GUI / scan table, ``quantized_time_steps``): ties away from zero
    (50 ns on the 20 ns grid -> 60 ns, never 40 ns) -- the MAINTAINER_NOTES §4 rule,
    owned by the shared ``round_ticks`` primitive."""
    from Zou_lab_control.neutral_atom.timing.pulse_table import quantized_time_steps
    from Zou_lab_control.neutral_atom.timing.sequence import round_ticks, snap_seconds_to_clock

    clock = 50e6                       # 20 ns tick
    step_ns = 1e9 / clock
    for ns, want_ticks in ((50.0, 3), (130.0, 7), (10.0, 1), (40.0, 2)):
        assert ns * 1e-9 * clock == ns / step_ns          # same raw tick count both paths
        assert quantized_time_steps(ns, time_step_ns=step_ns, allow_zero=True) == want_ticks
        assert snap_seconds_to_clock(ns * 1e-9, clock) * clock == pytest.approx(want_ticks)
    # the ONE tie rule: away from zero, both signs
    assert round_ticks(2.5) == 3 and round_ticks(-2.5) == -3 and round_ticks(0.5) == 1
    assert quantized_time_steps(-50.0, time_step_ns=step_ns,
                                allow_zero=True, allow_negative=True) == -3
    # clamp policies stay CALLER-side (only the tie rule is shared): a sub-tick duration
    # snaps UP to one tick (seconds path, min_ticks=1); an allow-zero quantize keeps 0
    assert snap_seconds_to_clock(1e-12, clock) == pytest.approx(1.0 / clock)
    assert quantized_time_steps(0.4 * step_ns, time_step_ns=step_ns, allow_zero=True) == 0


# ------------------------------------------------------------------ PulseSequence schema
def test_pulse_sequence_schema_identity_lives_on_the_class(tmp_path):
    """The plain schema name is the one persisted format identity.

    Writer, reader and the timing-layer payload loader all read the class-owned
    name; there is no numeric edit counter or upgrade path.
    """
    from Zou_lab_control.neutral_atom.timing import (
        PulseSequence, PulseTableState, load_pulse_payload, single_imaging_template)

    assert isinstance(PulseSequence.schema, str) and PulseSequence.schema
    seq = PulseSequence([], name="probe")
    payload = seq.to_dict()
    assert payload["schema"] == PulseSequence.schema
    assert PulseSequence.from_dict(payload).name == "probe"
    with pytest.raises(ValueError):
        PulseSequence.from_dict({**payload, "schema": "nope"})

    # the loader (owned by the DATA layer, beside the two classes) dispatches BOTH sister
    # payloads by their class-owned schema
    seq_path = tmp_path / "seq.json"
    seq_path.write_text(json.dumps(payload), encoding="utf-8")
    assert isinstance(load_pulse_payload(seq_path), PulseSequence)
    table_path = tmp_path / "table.json"
    table_path.write_text(json.dumps(single_imaging_template().to_dict()), encoding="utf-8")
    assert isinstance(load_pulse_payload(table_path), PulseTableState)


# ------------------------------------------------------------------ pulse gate + times rule
def test_pulse_controller_gate_and_times_rule_have_one_spelling():
    """The 'pulse must be a PulseController' predicate + guidance text lives ONCE
    (``operations.measurement.require_pulse_controller``), and the detection-times rule
    lives once in the readout subsystem -- the source scan fails on any re-spelling."""
    gate_hits = sorted(
        str(f.relative_to(_NA_PKG))
        for f in _NA_PKG.rglob("*.py")
        if "__pycache__" not in f.parts
        and "PulseController returned by" in f.read_text(encoding="utf-8"))
    assert gate_hits == [str(Path("operations") / "measurement.py")], gate_hits
    times_hits = sorted(
        str(f.relative_to(_NA_PKG))
        for f in _NA_PKG.rglob("*.py")
        if "__pycache__" not in f.parts
        and "positive finite detection times" in f.read_text(encoding="utf-8"))
    assert times_hits == [str(Path("subsystems") / "readout.py")], times_hits


def test_pulse_gate_fires_identically_at_every_entry_point():
    """The engine (``ScannedMeasurement``) and the readout scan builders reject a
    non-controller with the SAME TypeError text -- one helper, one message."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.operations.measurement import (
        ScannedMeasurement, require_pulse_controller)

    with pytest.raises(TypeError) as helper_err:
        require_pulse_controller(object())
    msg = str(helper_err.value)

    with pytest.raises(TypeError) as engine_err:        # gate fires before axis/calibration use
        ScannedMeasurement(object(), None, None, None, None, None, None)
    assert str(engine_err.value) == msg

    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3)})
    with pytest.raises(ValueError, match="positive finite detection times"):
        exp.readout.build_detection_scan([])            # the ONE times rule, fail-fast
    exp.readout.sitemap(display=False)                  # same contract path real hardware runs
    with pytest.raises(TypeError) as builder_err:
        exp.readout.build_detection_scan([0.005], pulse=object())
    assert str(builder_err.value) == msg


# ------------------------------------------------------------ timing subsystem ownership
def test_timing_orchestration_lives_on_the_subsystem_not_the_session():
    """``exp.timing`` OWNS its orchestration bodies (the subsystems/base contract); the
    session facade hosts NO shadow copies -- a forwarding shell whose logic lives on the
    session is two homes for one capability and drifts (AGENTS §2 single source)."""
    from Zou_lab_control.neutral_atom.session import NeutralAtomSession
    from Zou_lab_control.neutral_atom.subsystems.timing import TimingSubsystem

    for name in ("_configure_imaging", "_preflight", "_write_verilog", "_load_pulse_payload"):
        assert not hasattr(NeutralAtomSession, name), (
            f"session must not own {name}: the logic body belongs to TimingSubsystem")
    for name in ("configure_imaging", "preflight", "write_verilog", "bind_pulse"):
        assert callable(getattr(TimingSubsystem, name))


def test_capture_routes_exposure_through_the_one_configure_imaging_path(monkeypatch):
    """The conventional readout capture shares the ONE configure-imaging path: its camera
    gets the exposure and the matching session sequence is rebuilt before arm-before-fire.
    ``configure_imaging`` has no arbitrary-camera escape hatch because it owns this pair."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.subsystems.timing import TimingSubsystem

    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3)})
    calls: list[dict] = []
    orig = TimingSubsystem.configure_imaging

    def spy(self, **kwargs):
        calls.append(kwargs)
        return orig(self, **kwargs)

    monkeypatch.setattr(TimingSubsystem, "configure_imaging", spy)
    seq_before = exp.sequence
    exp.capture(exposure=2e-3, display=False)
    assert calls and calls[0]["exposure"] == pytest.approx(2e-3)
    assert "camera" not in calls[0]                     # timing owns only the readout role
    assert raw_device_set(exp).camera_exposure() == pytest.approx(2e-3)   # the camera really got it
    assert exp.sequence is not seq_before               # and the imaging sequence was rebuilt


def test_capture_free_running_camera_never_rebuilds_or_fires(monkeypatch):
    """A no-trigger camera is an explicit direct-acquire path.  Its exposure changes only
    that camera; the unrelated readout sequence and sequencer remain untouched, and the
    result records no fake pulse provenance."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.subsystems.timing import TimingSubsystem

    exp = na.connect("virtual")
    try:
        monitor = raw_device_set(exp)["monitor_camera"]
        assert monitor.effective_trigger_channels == ()
        sequence_before = exp.sequence
        raw_device_set(exp).sequencer.history.clear()

        def forbidden_rebuild(*args, **kwargs):
            raise AssertionError("free-running capture must not rebuild a readout pulse")

        monkeypatch.setattr(TimingSubsystem, "configure_imaging", forbidden_rebuild)
        result = exp.capture(camera="monitor_camera", exposure=3e-3, display=False)

        assert monitor.exposure == pytest.approx(3e-3)
        assert exp.sequence is sequence_before
        assert raw_device_set(exp).sequencer.history == []
        assert result.sequence is None
        assert result.summary()["sequence"] is None
    finally:
        exp.close()


def test_capture_rejects_non_readout_external_camera_before_hardware(monkeypatch):
    """A non-readout trigger wire has no session-owned pulse.  Reject before configure,
    acquire, prepare, or fire, with the owner task named in the actionable error."""
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual")
    try:
        monitor = raw_device_set(exp)["monitor_camera"]
        monitor.trigger_source = "mot_trigger"
        monitor.capture_trigger_channels = ("mot_trigger",)
        assert monitor.effective_trigger_channels == ("mot_trigger",)
        sequencer = raw_device_set(exp).sequencer
        sequencer.history.clear()

        def forbidden(*args, **kwargs):
            raise AssertionError("rejected capture touched hardware")

        monkeypatch.setattr(monitor, "configure", forbidden)
        monkeypatch.setattr(monitor, "acquire", forbidden)
        monkeypatch.setattr(sequencer, "prepare", forbidden)
        monkeypatch.setattr(sequencer, "fire", forbidden)

        with pytest.raises(RuntimeError, match="Optimize-MOT-field") as exc:
            exp.capture(camera="monitor_camera", exposure=3e-3, display=False)
        assert "mot_trigger" in str(exc.value)
        assert sequencer.history == []
    finally:
        exp.close()
