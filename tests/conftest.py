from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
root_text = str(REPO_ROOT)
if sys.path[0] != root_text:
    sys.path.insert(0, root_text)


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    yield
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.close("all")


def fire_imaging_pulse(sequencer, *, exposure=20e-3, cooling=2e-3):
    """Fire a CONTINUOUS imaging pulse (``repeat_forever``) on a raw sequencer -- the
    software model of the pulse GUI's "On Pulse".  An externally-triggered camera produces
    frames ONLY while the streamer is firing such a pulse; ``sequencer.set_safe_state()``
    ("Stop Pulse") clears it and the live image freezes.  Returns the fired sequence.

    Camera tests use this so they take the SAME firing path as real hardware (the camera is
    purely trigger-driven -- there is no fabricated live frame)."""
    from Zou_lab_control.neutral_atom.timing import imaging_sequence
    seq = imaging_sequence(exposure=exposure, load=True, name="live", cooling=cooling).forever()
    sequencer.prepare(seq)
    sequencer.fire()
    return seq


def fire_live_imaging(exp, *, exposure=None):
    """Fire the live imaging pulse on a session's sequencer (see :func:`fire_imaging_pulse`).
    Mirrors the user clicking "On Pulse" in the pulse GUI so the session's live camera streams."""
    devs = exp.devices
    cooling = getattr(getattr(devs, "trap_array", None), "mot_load_s", None)
    kw = {} if cooling is None else {"cooling": float(cooling)}
    if exposure is None:
        exposure = devs.camera.exposure
    return fire_imaging_pulse(devs.sequencer, exposure=exposure, **kw)
