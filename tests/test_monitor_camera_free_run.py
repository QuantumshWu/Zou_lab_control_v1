"""Free-running (``Software``) monitor camera: virtual == real Basler behaviour.

The real MOT monitor is a Basler in ``Software`` mode (``TriggerMode Off`` -- the discovery
config's default): it free-runs, so ``acquire`` yields frames with NO pulse wiring at all, and a
dark bench simply gives dark frames.  The virtual :class:`VirtualMotCamera` keeps the SAME two
modes under the SAME ``trigger_source`` vocabulary (:data:`devices.base.SOFTWARE_TRIGGER`, the one
constant both backends compare against):

* ``Software`` (the default, matching the real monitor's discovery config): every ``acquire``
  delivers a frame imaging what the streamer's outputs DRIVE -- the continuously firing program,
  else the steady state the last finite fire left LATCHED on the DAC pins
  (``VirtualSequencer.last_fired``), else the safe state (all-zero levels -> the dim background);
* an explicit line name (``"mot_trigger"``): the pure externally-triggered grabber -- an idle
  wire yields NO frame (the old hard-trigger contract, still selectable).

One rendering core (``_render_at_levels``) serves both modes -- only the LEVELS SOURCE differs,
so the free-run physics can never fork from the edge-gated physics.  The end-to-end test drives
the REAL console path (Add Panel -> camera row -> Start -> step): the exact user flow that used to
freeze forever when the monitor camera was selected without a mot_trigger pulse running.  It does
NOT pin the frame signal's hub name (the non-default-camera prefix is owned elsewhere) -- it reads
``node.published_signals()`` and asserts the block shape.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from Zou_lab_control.neutral_atom.devices.base import SOFTWARE_TRIGGER
from Zou_lab_control.neutral_atom.devices.camera_trigger import DEFAULT_CAMERA_TRIGGER_CHANNELS
from Zou_lab_control.neutral_atom.devices.registry import load_devices
from Zou_lab_control.neutral_atom.devices.virtual import VirtualMotCamera, VirtualSequencer
from Zou_lab_control.neutral_atom.operations.measurement import triggered_frames
from Zou_lab_control.neutral_atom.operations.tasks.mot_field import DEFAULT_MOT_TEMPLATE
from Zou_lab_control.neutral_atom.timing import resolve_fireable_template, single_imaging_template

MOT_TEMPLATE = str(REPO_ROOT / DEFAULT_MOT_TEMPLATE)


def _mot_sequence_at(cam, values_fn):
    """The MOT coil template compiled at per-bus levels chosen by ``values_fn(bus)`` -- levels
    always derive from the device's own public model (``cam.b0``), never re-typed constants."""
    state = resolve_fireable_template(MOT_TEMPLATE, default_name=DEFAULT_MOT_TEMPLATE,
                                      default_factory=single_imaging_template)
    slots = [s.name for s in state.api_slots if s.kind == "dac"]
    values = {name: int(values_fn(bus)) for name, bus in zip(slots, cam.coil_buses)}
    return state.with_api_resolved(values).to_sequence()


# --------------------------------------------------------------------------- (a) free-run default
def test_free_run_monitor_delivers_a_dark_frame_with_nothing_fired():
    """The virtual config's monitor camera IS free-running by default (== the real Basler
    discovery config), so with NOTHING fired ``acquire(1)`` still delivers one full-sensor frame
    -- imaging the SAFE state (all coil DACs parked at signed level 0), not a fabricated MOT."""
    ds = load_devices("virtual", open_devices=False)
    cam = ds.devices["monitor_camera"]
    assert cam.trigger_source == SOFTWARE_TRIGGER          # default aligns with the real discovery config
    frames = cam.acquire(1)
    assert len(frames) == 1
    assert frames[0].shape == tuple(cam.sensor_shape)
    assert cam.last_levels == {bus: 0.0 for bus in cam.coil_buses}   # the safe-state levels were sensed


# --------------------------------------------------------------------------- (b) driven then safe
def test_free_run_brightness_follows_the_held_output_state():
    """Free-run images the STEADY state the streamer's outputs hold: after a finite fire at the
    coil optimum the frame brightens (the DAC word stays latched on the pins -- ``last_fired``);
    ``set_safe_state`` parks the outputs and the image falls back to the safe-state background.
    Brightness ordering follows the camera's own public model (``mot_efficiency``), never a
    re-typed constant."""
    ds = load_devices("virtual", open_devices=False)
    cam, seqr = ds.devices["monitor_camera"], ds.devices["sequencer"]

    dark = cam.acquire(1)[0]                               # safe state: eff(0) -- the dim background
    seq = _mot_sequence_at(cam, lambda bus: cam.b0[bus])   # drive every coil to its optimum
    seqr.prepare(seq)
    seqr.fire(seq)                                         # finite fire: the word stays latched
    bright = cam.acquire(1)[0]
    assert cam.last_levels == {bus: int(cam.b0[bus]) for bus in cam.coil_buses}
    # the model itself orders the two scenes; the frames must follow it.  The faithful sensor is
    # full-frame (the spot is a tiny fraction of 2.3 MP), so brightness is read AT the spot centre
    # and the threshold derives from the camera's own model -- never a re-typed constant.
    safe_eff = cam.mot_efficiency({bus: 0.0 for bus in cam.coil_buses})
    gain = cam.peak_counts * (cam.mot_efficiency(cam.last_levels) - safe_eff)
    assert gain > 0
    cy, cx = cam.height // 2, cam.width // 2

    def spot_mean(frame):
        return float(frame[cy - 3:cy + 4, cx - 3:cx + 4].mean())

    assert spot_mean(bright) > spot_mean(dark) + 0.5 * gain

    seqr.set_safe_state()                                  # park the outputs -> back to background
    back = cam.acquire(1)[0]
    assert cam.last_levels == {bus: 0.0 for bus in cam.coil_buses}
    assert spot_mean(back) < spot_mean(dark) + 0.5 * gain  # the bright state is gone


# --------------------------------------------------------------------------- (c) explicit hard trigger
def test_explicit_line_name_keeps_the_pure_grabber_contract():
    """``trigger_source="mot_trigger"`` selects the old externally-triggered contract explicitly:
    an idle wire yields NO frame (the live view freezes), a fired template delivers exactly its
    edge-gated frame, and the counting line derives from the one trigger_source knob."""
    seqr = VirtualSequencer()
    cam = VirtualMotCamera(trigger_source="mot_trigger", sequencer=seqr, seed=0)
    assert cam.capture_trigger_channels == ("mot_trigger",)   # the wiring fact derives from the knob
    assert cam.acquire(1) == []                               # idle wire -> no edge -> no frame

    seq = _mot_sequence_at(cam, lambda bus: cam.b0[bus])
    frames = triggered_frames(cam, seqr, seq, 1)
    assert len(frames) == 1 and frames[0].shape == tuple(cam.sensor_shape)
    assert cam.last_levels == {bus: int(cam.b0[bus]) for bus in cam.coil_buses}

    seqr.set_safe_state()
    assert cam.acquire(1) == []                               # still a pure grabber after the fire


# --------------------------------------------------------------------------- (d) real-camera parity
def test_pylon_capture_trigger_channels_is_configurable_closed():
    """The real adapter's trigger-line declaration is a constructor fact, assertable CLOSED (no
    Basler runtime): the conservative default is the shared camera-layer default, a hardware-trigger
    config overrides it with the real cable's sequencer channel, and both backends speak the ONE
    ``SOFTWARE_TRIGGER`` vocabulary."""
    from Zou_lab_control.neutral_atom.devices.pylon import PylonCamera

    free = PylonCamera(trigger_source=SOFTWARE_TRIGGER)
    assert free.trigger_source == SOFTWARE_TRIGGER
    assert free.capture_trigger_channels == tuple(DEFAULT_CAMERA_TRIGGER_CHANNELS)

    wired = PylonCamera(trigger_source="Line1", capture_trigger_channels=("ch05",))
    assert wired.capture_trigger_channels == ("ch05",)
    assert wired.snapshot()["capture_trigger_channels"] == ["ch05"]


def test_free_run_counting_line_declaration_is_identical_on_both_backends():
    """The machine-readable "which line does a FREE-RUN camera count on" fact has ONE answer on
    both backends (virtual == real): the WIRING declaration (``capture_trigger_channels``) is the
    same inert conservative default the real discovery config carries, and the ACTIVE counting
    line (the base-class ``effective_trigger_channels``) is ``()`` -- a free-running sensor never
    consults its trigger input, so its edge counts are honestly zero.  Hardware-trigger mode keeps
    the wiring as the counting line on both."""
    from Zou_lab_control.neutral_atom.devices.pylon import PylonCamera

    v_free = VirtualMotCamera(trigger_source=SOFTWARE_TRIGGER)
    r_free = PylonCamera(trigger_source=SOFTWARE_TRIGGER)
    assert v_free.capture_trigger_channels == r_free.capture_trigger_channels \
        == tuple(DEFAULT_CAMERA_TRIGGER_CHANNELS)
    assert v_free.effective_trigger_channels == r_free.effective_trigger_channels == ()
    assert v_free.primary_trigger_channel is None and r_free.primary_trigger_channel is None

    v_wired = VirtualMotCamera(trigger_source="mot_trigger")
    r_wired = PylonCamera(trigger_source="Line1", capture_trigger_channels=("ch05",))
    assert v_wired.effective_trigger_channels == ("mot_trigger",)
    assert r_wired.effective_trigger_channels == ("ch05",)
    assert v_wired.primary_trigger_channel == "mot_trigger"
    assert r_wired.primary_trigger_channel == "ch05"


# --------------------------------------------------------------------------- end-to-end (console)
def test_console_monitor_camera_streams_with_no_pulse_running():
    """USER ACCEPTANCE PATH: the real console flow -- Add Panel -> camera row -> pick
    ``monitor_camera`` -> Start -> step -- publishes a live frame block WITHOUT any pulse firing
    anywhere (the exact scenario that used to freeze: the hard-wired virtual monitor never
    produced a frame unless a mot_trigger pulse ran).  The frame signal's NAME is deliberately not
    pinned (the non-default-camera signal prefix is owned by DeviceSet); the node's own
    ``published_signals()`` is the source and the hub block's shape is the assertion."""
    import Zou_lab_control.neutral_atom as na
    from conftest import add_logic_row, make_console

    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        row = add_logic_row(con, ("camera", "live"))
        # The user's gesture: pick "monitor_camera" in the node's REAL Edit form (Start collects
        # the form's values, so the choice must land on the form widget, not a bypassed dict).
        con._logic_editors[id(row)].form.seed_values({"camera": "monitor_camera"})
        con._start_logic_node(row)
        node = con._logic_nodes[id(row)]
        assert node.camera is exp.devices["monitor_camera"]
        node.step()                                        # one synchronous cycle -- NO pulse fired
        sigs = sorted(node.published_signals())
        assert len(sigs) == 1                              # one frame signal for frames_per_cycle=1
        block = np.asarray(con.hub.latest(sigs[0]))
        assert block.shape == (1, 1) + tuple(exp.devices["monitor_camera"].sensor_shape)
    finally:
        con.shutdown()
        exp.close()
