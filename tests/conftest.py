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


@pytest.fixture
def fit_thread_guard(monkeypatch):
    """Arm the opt-in fit-thread guard for a test: while active, a REAL unbounded curve-fit solve
    (``core.fitting._solve_candidates`` via ``fit_selected``) that runs ON the Qt application thread
    RAISES (#6).  Off by default so a notebook main-thread solve stays legal; a worker-node solve and a
    ``solve=False`` reconstruction both pass.  A per-kind test uses this to mechanically prove a console
    fit never solves on the GUI event loop."""
    from Zou_lab_control.neutral_atom.core.fitting import _FIT_THREAD_GUARD_ENV
    monkeypatch.setenv(_FIT_THREAD_GUARD_ENV, "1")
    yield


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


def make_console(exp, *, running_nodes=None, window_px=(900, 600)):
    """A real offscreen TaskConsole wired exactly like ``exp.task_console()`` -- the ONE
    console factory for lifecycle tests (build / start / stop / task flows), so every test
    drives the same specs + hub wiring the GUI entry point does.  The background tick timer
    is stopped for determinism: tests drive frames themselves -- use :func:`tick` when the
    test asserts RENDERED state (``con._tick()`` alone only submits to the render worker)."""
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    ensure_qt_app()
    con = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                      measurements=exp.readout.measurement_specs(),
                      processors=exp.readout.processor_specs(),
                      tasks=exp.readout.task_specs(), window_px=window_px,
                      running_nodes=list(running_nodes or ()))
    con._timer.stop()
    return con


def tick(con):
    """ONE deterministic console frame.  ``_tick()`` SUBMITS the dirty panels to the
    background render worker; the finished batch normally comes back through the queued
    ``job_done`` signal -- which never runs for a test that does not spin the Qt event
    loop.  ``barrier()`` both waits out the compose and delivers the batch to the GUI
    pass synchronously, so after this call every submitted panel is built + presented."""
    con._tick()
    con._render_loop.barrier()


def write_saved_npz(path, *, data_x, data_y, **info):
    """Write a hand-crafted saved-figure ``.npz`` through the ONE production writer
    (``data_figure._write_saved_npz``), filling the required-but-unspecified ``info`` records with
    inert defaults.  A reload test that needs a bespoke payload (a site map with a stored frame, a
    pulse recipe, a promote-fallback with no value role) builds it HERE instead of a raw
    ``np.savez(info=...)`` -- so a test fixture can never drift from the current envelope again (the
    old raw-``savez`` fixtures silently rotted the moment the envelope grew its typed records)."""
    from Zou_lab_control.frontend.data_figure import _write_saved_npz
    from Zou_lab_control.neutral_atom.operations.figure_capture import raw_data_flow_graph
    record = {
        "name": "fig", "kind": None, "labels": ["X", "Y", "Z"], "unit": "",
        "points_done": 0, "repeat_cur": 1, "view": {}, "fit": None, "size": "2x2",
        "signals": {}, "figure_recipe": None, "provenance": {},
    }
    record.update(info)
    # Every current envelope carries a flow_graph inside provenance and, for a pulse recipe, a
    # panel_size -- fill the ones a partial fixture omitted so the payload validates like a real save.
    provenance = dict(record["provenance"] or {})
    provenance.setdefault("flow_graph", raw_data_flow_graph())
    record["provenance"] = provenance
    recipe = record["figure_recipe"]
    if isinstance(recipe, dict) and recipe.get("kind") == "pulse" and "panel_size" not in recipe:
        # panel_size is a REQUIRED non-empty recipe key: a real pulse save stamps the OPTIMAL default
        # size (``default_pulse_size``) when the operator has not chosen one, so a fixture that omits it
        # gets exactly that -- the same single source the save and the viewer's fallback both use.
        from Zou_lab_control.frontend.live import default_pulse_size
        from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState
        state = PulseTableState.from_dict(recipe["pulse_state"])
        recipe = {**recipe, "panel_size": default_pulse_size(
            state, include_always_off=bool(recipe.get("include_always_off", True)))}
        record["figure_recipe"] = recipe
    _write_saved_npz(path, data_x=data_x, data_y=data_y, info=record)


def add_logic_row(con, data):
    """Add a logic row through the REAL Add-Panel path (kind combo -> ``_add_panel``) and
    return it -- e.g. ``add_logic_row(con, ("camera", "live"))``."""
    kc = con.kind_combo
    i = next(j for j in range(kc.count()) if kc.itemData(j) == data)
    kc.setCurrentIndex(i)
    con._add_panel()
    return con.logic_nodes[-1]
