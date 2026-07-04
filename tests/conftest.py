import os
from pathlib import Path
import sys

import pytest

# Pin the non-interactive Agg backend for the WHOLE test session, in ONE place (this conftest
# loads before any test module).  Otherwise matplotlib lazily resolves to the GUI 'qtagg' backend
# the first time a test creates a figure under PyQt5, and every plt.figure() then spawns a real Qt
# FigureManager.  The suite USED to set Agg only in scattered per-file `matplotlib.use("Agg")`
# calls, so any test that ran before the first such call (alphabetically earlier) created live Qt
# managers; those are destroyed at interpreter shutdown racing with the QApplication teardown -> an
# INTERMITTENT process-exit segfault (exit 139) even though every test passed.  Pinning Agg here is
# the single source that makes the backend deterministic and kills that flaky teardown crash.  The
# GUI panel canvas (EmbeddedFigureCanvas, an explicit FigureCanvasQTAgg) is unaffected: it imports
# the Qt backend directly, independent of this default.
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")


REPO_ROOT = Path(__file__).resolve().parents[1]
root_text = str(REPO_ROOT)
if sys.path[0] != root_text:
    sys.path.insert(0, root_text)


# The virtual backend is a REAL-TIME hardware simulator (sleep_scale=1.0 by default), so a
# fired pulse program takes its real wall-clock duration and the live camera paces with the
# pulse.  The pytest suite must NOT pay that wall-clock -- fast-forward it: flip the virtual
# default to 0 so the SAME data/physics path runs without the timing sleeps.  Real-time itself
# is exercised explicitly (sleep_scale=1.0) in tests/test_virtual_realtime_pacing.py.
import Zou_lab_control.neutral_atom.devices.virtual as _virtual_backend  # noqa: E402
_virtual_backend.DEFAULT_SLEEP_SCALE = 0.0


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
