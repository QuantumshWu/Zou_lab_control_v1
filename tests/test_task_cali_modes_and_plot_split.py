"""MECHANICAL guard for the deep #4 task / measurement requirements.

These are exactly the behaviours that silently regress, so they are pinned here
(the repo rule: a mechanically-enforceable requirement is a test, not a doc line):

  * the calibrate TASK really honours its declared params -- it is NOT cosmetic:
      - source = "live"             -> acquire now (camera + imaging pulse at the given exposure),
                                       write the canonical `folder/calibration.json` + a report;
      - source = "saved frames"     -> calibrate from raw frames already in `folder`;
        (there is NO "saved calibration" source: reusing a finished calibration is not a
        calibration run -- the Judge-occupancy processor loads its calibration.json directly.)
      - the cali computes EVERY readout model (box / per-site PSF / uniform PSF) into ONE
        calibration; the OccupancyProcessor picks one at read time -- there is no `mode` param;
      - threshold = otsu / bimodal;
      - one `folder` (input + output, never blank) replaces the old save_path/load_path/data_dir.
  * the measurement PLOT split: a measurement called from the NOTEBOOK API defaults
    display=True (it auto-opens its default plot); the SAME measurement driven as a
    GUI/task logic node is plot=False -- it only publishes to the hub (the user wires
    a Plot panel to that signal), and starting it opens NO plot.

Offscreen Qt + virtual backend (the same contract path real hardware takes; only the
camera frames are simulated).
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys
import time

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from conftest import fire_live_imaging   # the live "On Pulse" the trigger-driven camera needs


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def _calibrated(grid=(3, 4)):
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": grid, "image_shape": (48, 60)})
    exp.readout.sitemap(method="box", frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    return exp


# --------------------------------------------- calibrate ONCE, processor picks the method
@pytest.mark.parametrize("threshold_method", ["otsu", "bimodal"])
def test_calibrate_task_computes_every_method_processor_picks(threshold_method):
    """The cali runs ONCE and produces a MULTI-METHOD calibration (box / per-site PSF /
    uniform PSF), each with finite per-site thresholds; the OccupancyProcessor then reads
    with whichever method it chooses (the method is a readout choice, not a cali choice)."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.calibration import ALL_READOUT_METHODS
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement, OccupancyProcessor

    exp = _calibrated()
    try:
        task = exp.readout.calibrate_task(
            SignalHub(), threshold_method=threshold_method,
            threshold_frames=16,
            readout_exposure=0.02)
        task.run_to_completion()
        cal = task.calibration
        assert np.asarray(cal.centers).shape == (12, 2)
        assert set(cal.methods()) == set(ALL_READOUT_METHODS)          # cali once -> every method
        for m in ALL_READOUT_METHODS:
            thr = np.asarray(cal.thresholds_for(m), dtype=float).reshape(-1)
            assert thr.shape == (12,) and np.isfinite(thr).all()
        # the processor PICKS the method -> calibration.detect(method=...) routes correctly
        fire_live_imaging(exp)                 # On Pulse: the trigger-driven camera streams
        for m in ALL_READOUT_METHODS:
            hub = SignalHub()
            cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
            occ = OccupancyProcessor(hub, calibration=cal, source_expr={"inputs": ["frame_0"], "source": "value = signal"}, method=m)
            cam.step(); occ.step()
            occupied = np.asarray(hub.latest("occupied"))
            assert occupied.ndim == 3 and occupied.shape[1:] == (1, 12)   # (repeat, 1, n_sites) -- data_points kept
            # the judged frame block is published atomically -> rings + underlay are the same shots; the
            # frame_judged (repeat, 1, H, W) is the camera's frame UNCHANGED (a frame keeps one shape raw or judged)
            assert np.array_equal(hub.latest("frame_judged"), np.asarray(hub.latest("frame_0")))
    finally:
        exp.close()


def test_calibrate_task_pins_live_camera_exposure_to_the_readout_gate():
    """#issue-2 (live sitemap path): after a live calibration the session camera exposure is PINNED to
    the gate the thresholds were learnt at (``threshold_exposure``), so the live OccupancyProcessor --
    which judges the RUNNING pulse's frames at the camera exposure -- self-matches the calibration
    instead of imaging at a stale default (which would classify every atom dark while the report says
    99 %).  Calibrate at a SHORT 4 ms readout while the camera default is 20 ms, then verify the live
    judge is accurate (not chance)."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement, OccupancyProcessor

    exp = _calibrated()
    try:
        exp.devices.camera.exposure = 0.020                  # stale default -- DIFFERENT from the readout gate
        task = exp.readout.calibrate_task(
            SignalHub(), threshold_method="otsu", threshold_frames=24, readout_exposure=0.004)
        task.run_to_completion()
        # the live readout exposure is now the calibrated gate, not the stale 20 ms default
        assert exp.devices.camera.exposure == pytest.approx(0.004), exp.devices.camera.exposure
        assert (task.calibration.metadata or {}).get("threshold_exposure") == pytest.approx(0.004)
        # and the live judge (camera frame at the pinned exposure) is accurate, not ~0.5 chance
        fire_live_imaging(exp)
        accs = []
        for _ in range(8):
            hub = SignalHub()
            cam = CameraMeasurement(hub, exp.devices.camera, sequencer=exp.devices.sequencer)
            occ = OccupancyProcessor(hub, calibration=task.calibration,
                                     source_expr={"inputs": ["frame_0"], "source": "value = signal"}, method="box")
            cam.step(); occ.step()
            pred = np.asarray(hub.latest("occupied"), dtype=bool).reshape(-1)
            truth = np.asarray(exp.devices.trap_array.occupancy, dtype=bool).reshape(-1)
            accs.append(float((pred == truth).mean()))
        assert np.mean(accs) >= 0.75, f"live occupancy must match at the pinned exposure, got {np.mean(accs):.3f}"
    finally:
        exp.close()


def test_calibrate_task_live_writes_canonical_calibration_json_for_the_detector(tmp_path):
    """A live calibration writes the CANONICAL ``folder/calibration.json`` -- the stable,
    named file the Judge-occupancy detector loads -- plus the report artifacts, ALL directly
    in ``folder`` (no hidden timestamped sub-folder).  Reusing it is NOT a calibration run:
    the file just loads (here via TrapCalibration.load, in the app via the processor's
    calibration field), with NO second acquisition."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.core.calibration import TrapCalibration

    exp = _calibrated()
    try:
        folder = tmp_path / "cal"
        made = exp.readout.calibrate_task(
            SignalHub(), source="live", folder=str(folder),
            threshold_frames=12)
        made.run_to_completion()
        # the canonical latest calibration the detector defaults to
        canonical = folder / "calibration.json"
        assert canonical.exists()
        # the report artifacts land in the SAME explicit folder (report_dir == folder)
        report_dir = Path(made.result["report_dir"])
        assert report_dir == folder
        assert report_dir.exists() and (report_dir / "calibration.json").exists()
        # reloading is a plain file load (no calibration re-run): same centers
        loaded = TrapCalibration.load(canonical)
        assert np.allclose(np.asarray(loaded.centers), np.asarray(made.calibration.centers))
    finally:
        exp.close()


def test_calibrate_task_builds_long_short_long_bracket_from_two_exposures(tmp_path):
    """The cali takes TWO operator-set exposures -- ``reference_exposure`` (LONG) and
    ``readout_exposure`` (SHORT) -- and fires the long-short-long bracket built from BOTH
    (rb87 readout flow): the loaded template supplies the channels + the ONE cooling cycle, and
    the two exposures set the three image windows.  The bracket must be CONTINUOUS -- one cooling
    load, the trap held the whole time, NO re-cooling between the three consecutive emCCD frames
    (re-cooling would scramble the atoms and void the labels)."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.timing import default_imaging_template

    exp = _calibrated()
    try:
        prog = tmp_path / "my_imaging.json"             # the user's own imaging program
        st = default_imaging_template()
        st.save(prog)

        # BOTH exposures are settable cali params -- the long is NOT buried in the template.
        task = exp.readout.calibrate_task(
            SignalHub(), source="live", pulse_template=str(prog),
            reference_exposure=0.030, readout_exposure=0.005, threshold_frames=12)
        assert task.pulse_template == str(prog)
        params = task.acquisition_parameters()
        assert params["reference_exposure"] == pytest.approx(0.030)   # LONG is a real param
        assert params["readout_exposure"] == pytest.approx(0.005)     # SHORT is a real param

        # file == fired: the TEMPLATE FILE itself is the long-short-long bracket (3 emCCD
        # frames already in it, not derived at fire time) -- the user's complaint that the
        # template "still has only one emCCD" must never recur.
        emccd_bit = st.channels.index("emCCD")
        assert sum(1 for p in st.periods if p.states[emccd_bit]) == 3
        # The cali sets ONLY the three exposures BY NAME (a1 = first long, a2 = short readout,
        # a3 = second long) and fires THAT template -- it does not invent or unroll a bracket.
        fired = task._resolve_template(task.pulse_template)
        fired.set_api("a1", task.reference_exposure)
        fired.set_api("a2", task.readout_exposure)
        fired.set_api("a3", task.reference_exposure)
        bracket = fired.to_sequence(name="ref")
        cooling = [(p.start, p.duration) for p in bracket.pulses if p.channel == "cooling" and p.value]
        trap = [p for p in bracket.pulses if p.channel == "trap" and p.value]
        emccd = sorted(p.start for p in bracket.pulses if p.channel == "emCCD" and p.value)
        probe = [round(p.duration, 6) for p in sorted(
            (p for p in bracket.pulses if p.channel == "probe" and p.value), key=lambda p: p.start)]
        assert len(cooling) == 1 and cooling[0][1] == pytest.approx(0.002)   # ONE cooling load only
        assert len(trap) == 1                                                # trap held continuously
        assert len(emccd) == 3                                               # three consecutive emCCD
        assert probe == [pytest.approx(0.030), pytest.approx(0.005), pytest.approx(0.030)]  # 30-5-30
        task.run_to_completion()
        assert np.asarray(task.calibration.centers).shape == (12, 2)
    finally:
        exp.close()


def test_calibrate_task_saved_frames_source(tmp_path):
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = _calibrated()
    try:
        folder = tmp_path / "run"
        na.write_virtual_run(str(folder), groups=8, grid_shape=(3, 4), seed=0)
        task = exp.readout.calibrate_task(
            SignalHub(), source="saved frames", folder=str(folder))
        assert task.source == "saved frames"
        task.run_to_completion()
        assert np.asarray(task.calibration.centers).shape == (12, 2)
    finally:
        exp.close()


def test_calibrate_task_saved_frames_uses_reference_brackets_for_held_out_fidelity(tmp_path):
    """A SAVED run is grouped exactly like a live bracket (long ``ref_shots`` vote ground
    truth around the ``short_shot`` readout).  The saved-frames flow must regroup those
    reference frames and take the SAME held-out training path as the live flow: distinct
    box / per-site PSF / uniform PSF fidelity + reference-trained per-site thresholds
    (``threshold_method='per_site_reference'``) -- NOT the affine-invariant self-consistent
    estimate that reports a bitwise-identical fidelity for every method.  Regression for the
    folder flow silently degrading to that estimate (audit finding [1])."""
    import Zou_lab_control.neutral_atom as na
    import Zou_lab_control.frontend  # noqa: F401  (registers the viewer so the report renders)
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.calibration import ALL_READOUT_METHODS

    exp = _calibrated((4, 5))
    try:
        folder = tmp_path / "run"
        # a fidelity-LIMITED short readout (small exposure) so the methods genuinely separate;
        # many groups so every site gets both bright + dark labelled shots to train on.
        na.write_virtual_run(str(folder), groups=160, grid_shape=(4, 5),
                             short_exposure=5e-4, seed=5)
        task = exp.readout.calibrate_task(
            SignalHub(), source="saved frames", folder=str(folder))
        task.run_to_completion()

        # the run's reference frames were regrouped into per-group brackets (one short readout
        # scored against its long reference frames), exactly like the live flow.
        n_ref = 3                                          # write_virtual_run default ref_shots
        assert len(task._reference_groups) == len(task._readout_by_group) > 0
        assert len(task._reference_groups[0]) == n_ref
        # the held-out training wrote reference-trained per-site thresholds back into the cal
        assert task.calibration.metadata.get("threshold_method") == "per_site_reference"

        # every method got a real held-out classification fidelity and the three are NOT the
        # bitwise-identical number the self-consistent estimate would force.
        assert set(task._method_fidelity) == set(ALL_READOUT_METHODS)
        means = {m: float(np.nanmean(d["fidelity"])) for m, d in task._method_fidelity.items()}
        assert all(0.5 <= v <= 1.0 for v in means.values())
        assert max(means.values()) - min(means.values()) > 1e-3, (
            f"per-method fidelity must differ from saved frames, got {means}")
    finally:
        exp.close()


def test_calibrate_task_live_save_frames_round_trip(tmp_path):
    """source=live with save_frames=True puts the acquired raw frames in a clean
    ``<folder>/frames/`` sub-folder (img<n>.npy) -- NOT at the root, so the cali folder root
    stays uncluttered (canonical artifacts + paired figure saves only).  A run_schema.json at
    the root records the layout (incl. ``frames_subdir``) so a later source="saved frames" run
    re-calibrates from those frames WITHOUT re-acquiring."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = _calibrated()
    try:
        folder = tmp_path / "live_saved"
        n_groups = 12
        made = exp.readout.calibrate_task(
            SignalHub(), source="live", folder=str(folder), save_frames=True,
            threshold_frames=n_groups)
        made.run_to_completion()
        # raw frames in a CLEAN sub-folder, NOT at the root (cali folder root stays tidy)
        assert sorted(folder.glob("img*.npy")) == [], "raw frames must NOT be at the cali root"
        frames_dir = folder / "frames"
        assert frames_dir.is_dir()
        imgs = sorted(frames_dir.glob("img*.npy"))
        assert len(imgs) == n_groups * 3                # 3 frames/group (ref-short-ref)
        assert (folder / "run_schema.json").exists()
        # re-calibrate from those saved frames -- no second acquisition, same site count
        reused = exp.readout.calibrate_task(
            SignalHub(), source="saved frames", folder=str(folder))
        reused.run_to_completion()
        assert np.asarray(reused.calibration.centers).shape == (12, 2)
    finally:
        exp.close()


def test_calibrate_task_live_save_frames_writes_a_png_companion_per_npy(tmp_path):
    """#5: every saved emCCD bracket frame (``img<n>.npy``, the round-trip DATA) gets a paired
    ``img<n>.png`` 2D-IMAGE companion so the operator can eyeball the frames the cali actually
    used.  The png is rendered through the viewer SEAM (``active_plotter().save_frame_image``) --
    na never imports the frontend -- so this pins the wiring: ONE png call per saved npy, same
    stem, in the frames sub-folder.  (Headless with NO viewer -> npy only; covered by the
    round-trip test above, which registers no plotter and asserts only the npy.)"""
    import Zou_lab_control.frontend  # noqa: F401 -- ensure the real viewer is registered
    from Zou_lab_control._viewer_registry import active_plotter, register_plotter
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    orig = active_plotter()
    assert orig is not None, "the frontend viewer must be registered for this end-to-end seam test"

    class _RecordingPlotter:                                  # wraps the REAL viewer, records the frame-png call
        def __init__(self, inner):
            self._inner = inner
            self.calls = []
        def save_frame_image(self, path, frame, *, title=None):
            self.calls.append((Path(path).name, tuple(np.asarray(frame).shape)))
            return self._inner.save_frame_image(path, frame, title=title)   # render the REAL png
        def __getattr__(self, name):                         # everything else (plot/display/...) -> the real viewer
            return getattr(self._inner, name)

    exp = _calibrated()
    rec = _RecordingPlotter(orig)
    register_plotter(rec)
    try:
        folder = tmp_path / "live_saved_png"
        n_groups = 2
        made = exp.readout.calibrate_task(
            SignalHub(), source="live", folder=str(folder), save_frames=True,
            threshold_frames=n_groups)
        made.run_to_completion()
        frames_dir = folder / "frames"
        npys = sorted(frames_dir.glob("img*.npy"))
        pngs = sorted(frames_dir.glob("img*.png"))
        assert len(npys) == n_groups * 3                     # 3 frames/group (ref-short-ref)
        # one png companion per npy, SAME stem (img1.npy <-> img1.png)
        assert [p.stem for p in pngs] == [p.stem for p in npys]
        assert len(rec.calls) == len(npys)                   # the seam was called once per saved frame
        assert all(name.endswith(".png") and len(shape) == 2 for name, shape in rec.calls)
    finally:
        register_plotter(orig)                               # never leak the recording plotter into other tests
        exp.close()


def test_frontend_save_frame_image_reuses_the_2d_image_plot_type(tmp_path, monkeypatch):
    """#5 render side: ``save_frame_image`` must render the frame through the SAME "2D image" plot TYPE
    the live console uses (:class:`Live2DDis`, kind="2d") -- REUSE, not a hand-rolled imshow -- so the
    saved picture looks identical to the console's 2D camera image (same cmap + side distribution +
    colorbar).  It writes a real, non-empty PNG and ONLY the png (NO ``.npz``: the ``.npy`` beside it
    in the frames folder IS the data)."""
    import Zou_lab_control.frontend.live as live
    from Zou_lab_control.frontend.calibration_report import save_frame_image

    used = {}
    real = live.Live2DDis
    monkeypatch.setattr(live, "Live2DDis", lambda *a, **k: used.setdefault("via", real(*a, **k)))

    frame = np.random.default_rng(0).normal(600.0, 30.0, size=(40, 50))
    out = Path(save_frame_image(tmp_path / "img1.png", frame))
    assert "via" in used, "the frame png must be rendered via the Live2DDis 2D-image plot type (reuse)"
    assert out.exists() and out.suffix == ".png" and out.stat().st_size > 0
    assert not (tmp_path / "img1.npz").exists()              # png only -- the data lives in the .npy companion


def test_calibrate_task_live_save_frames_off_writes_no_frames(tmp_path):
    """save_frames=False writes ONLY the calibration + report -- no raw frames, no schema,
    no frames sub-folder."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = _calibrated()
    try:
        folder = tmp_path / "no_save"
        made = exp.readout.calibrate_task(
            SignalHub(), source="live", folder=str(folder), save_frames=False,
            threshold_frames=8)
        made.run_to_completion()
        assert (folder / "calibration.json").exists()       # the calibration is still written
        assert sorted(folder.glob("img*.npy")) == []        # no frames at root
        assert not (folder / "frames").exists()             # no frames sub-folder either
        assert not (folder / "run_schema.json").exists()
    finally:
        exp.close()


def test_calibrate_task_bool_param_renders_as_toggle_switch():
    """A bool task param (save frames) renders in the Edit form as a sliding on/off toggle
    SWITCH (FluentSwitch), not a checkbox -- the user's "bool -> toggle widget" UI rule."""
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.frontend.qt_fluent import FluentSwitch
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = _calibrated()
    try:
        console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                              tasks=exp.readout.task_specs(), window_px=(900, 600))
        console._timer.stop()
        kc = console.kind_combo
        spec = exp.readout.task_specs()[0]
        i = next(j for j in range(kc.count()) if kc.itemData(j) == ("task", spec.name))
        kc.setCurrentIndex(i)
        console._add_panel()
        row = console.logic_nodes[-1]
        console._edit_logic_node(row)
        editor = console._logic_editors[id(row)]
        widget = editor.form._widgets["save_frames"]
        assert editor.form._decls["save_frames"].kind == "bool"
        assert isinstance(widget, FluentSwitch)
    finally:
        console.shutdown()
        exp.close()


def test_sitemap_is_single_slot_while_other_kinds_allow_multi_slot():
    """E1 guard: the signal-picker + ``value = ...`` expression MECHANISM is universal (every
    kind has the expression box), but whether a kind can GROW slots (+signal / −signal) is
    data-driven from PANEL_SINGLE_SLOT_KINDS -- the site map takes EXACTLY ONE occupancy signal
    (its centres + frame underlay resolve from signal[0]), so NO +/-; other kinds are multi-slot.
    Locks the design so the slot UI can never regress to a hardcoded-per-panel +/- on the sitemap."""
    from Zou_lab_control.frontend.task_console import (
        PanelCard, PanelConfig, panel_allows_multi_slot)
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app

    ensure_qt_app()
    assert panel_allows_multi_slot("sites") is False
    assert panel_allows_multi_slot("2d") and panel_allows_multi_slot("1d") and panel_allows_multi_slot("monitor")

    sites = PanelCard(PanelConfig(kind="sites"))
    assert sites._multi_slot is False
    assert not hasattr(sites, "add_slot_button")     # NO +/- on the site map
    assert hasattr(sites, "source_edit")             # but it KEEPS the universal expression box

    two_d = PanelCard(PanelConfig(kind="2d"))
    assert two_d._multi_slot is True
    assert hasattr(two_d, "add_slot_button")         # +/- present on a multi-slot kind


def test_pulse_scan_slots_form_is_template_driven():
    """The Pulse-scan measurement carries ONE auto-form (kind ``pulse_slots``) bound to the
    sibling ``template`` field: one numeric row per API slot + a Scan-mode toggle over ONE shared
    scan-table program, rebuilt whenever the template path changes; ``collect_values`` returns the
    {'api', 'scan_mode', 'scan_code', 'extra_delay'} dict the build() consumes (#H3s-F7)."""
    from Zou_lab_control.frontend.task_console import MeasurementPanel

    exp = _calibrated()
    try:
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
        panel = MeasurementPanel([spec], single=True)
        widget = panel._widgets["pulse_slots"]
        assert panel._decls["pulse_slots"].kind == "pulse_slots"
        out = panel.collect_values()["pulse_slots"]
        # the form returns the api numeric rows + the single Scan-mode + the active scan_code program
        # + the inter-point extra_delay -- ONE scan_code (no separate api_scan key anymore).
        assert isinstance(out, dict) and set(out) == {"api", "scan_mode", "scan_code", "extra_delay"}
        assert out["scan_mode"] in {"none", "api", "scan"}
        # the imaging template carries 3 unique API handles -> 3 numeric rows after a switch
        panel._widgets["template"].setText("imaging_template.json")
        panel._repopulate_pulse_slots("pulse_slots")
        out2 = panel.collect_values()["pulse_slots"]
        assert set(out2["api"]) >= {"a1", "a2", "a3"}
    finally:
        if hasattr(panel, "shutdown"):
            panel.shutdown()
        exp.close()


# ------------------------------------------------- camera measurement: editable region
def test_camera_measurement_region_is_editable_and_applies_to_virtual():
    """The camera measurement exposes its ROI as an editable ``region`` (the SAME
    field for virtual / real -- only the camera differs), and setting it actually
    windows the virtual camera (the frame is cropped).  This pins that virtual==real:
    both honour camera.configure(roi=)."""
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = _calibrated((3, 4))
    try:
        # the camera Edit auto-form carries region (next to exposure / frames_per_cycle)
        console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                              tasks=exp.readout.task_specs(), window_px=(900, 600))
        console._timer.stop()
        kc = console.kind_combo
        i = next(j for j in range(kc.count()) if kc.itemData(j) == ("camera", "live"))
        kc.setCurrentIndex(i)
        console._add_panel()
        row = console.logic_nodes[-1]
        console._edit_logic_node(row)
        editor = console._logic_editors[id(row)]
        assert {"frames_per_cycle", "exposure", "region"} <= set(editor.form._widgets)
        console.shutdown()

        # building with a region windows the VIRTUAL camera -> the frame is cropped
        node = exp.readout.camera_spec().build(SignalHub(), region="10, 50, 8, 40")
        assert node.camera.roi is not None
        fire_live_imaging(exp)                            # On Pulse: the trigger-driven camera streams
        frame = node.camera.acquire(1, sequence=getattr(node.sequencer, "firing", None))[0]
        assert np.asarray(frame).shape != (40, 50)        # ROI actually crops the virtual frame
        assert "region" in node.acquisition_parameters()  # round-trips (endpoints)
    finally:
        exp.close()


# ------------------------------------------------------- measurement plot split
def test_notebook_measurement_defaults_display_true():
    """A measurement called from the notebook API auto-plots (display=True default)."""
    from Zou_lab_control.neutral_atom.operations.measurement import ScannedMeasurement

    exp = _calibrated()
    try:
        # the scanned-measurement run + a readout measurement method both default on
        assert inspect.signature(ScannedMeasurement.run).parameters["display"].default is True
        assert inspect.signature(exp.readout.temperature).parameters["display"].default is True
    finally:
        exp.close()


def test_gui_measurement_node_is_plot_false_publishes_to_hub_only():
    """The SAME measurement driven as a GUI logic node is plot=False: it publishes
    its result signal to the hub and opens NO plot panel (the user wires a Plot)."""
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = _calibrated()
    specs = exp.readout.measurement_specs()
    console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                          measurements=specs, tasks=exp.readout.task_specs(), window_px=(900, 600))
    console._timer.stop()
    try:
        spec = specs[0]
        kc = console.kind_combo
        i = next(j for j in range(kc.count()) if kc.itemData(j) == ("measurement", spec.name))
        kc.setCurrentIndex(i)
        console._add_panel()
        row = console.logic_nodes[-1]
        console._start_logic_node(row)
        published_y = f"{spec.key}_{spec.y_key}"        # node publishes under its slug
        deadline = time.monotonic() + 8.0
        while published_y not in console.hub.names() and time.monotonic() < deadline:
            time.sleep(0.03)
        assert published_y in console.hub.names()       # data on the hub
        assert console.cards == []                     # plot=False: no auto plot panel
    finally:
        console.shutdown()
        exp.close()


def test_shipped_imaging_template_is_a_continuous_three_trigger_bracket():
    """MECHANICAL guard that the imaging template IS the long-short-long bracket -- it is NOT a
    single window the cali secretly unrolls (the design the user demanded).  The shipped
    ``default_imaging_template`` (== ``pulses/imaging_template.json``) must, on its own:
      * trigger the camera THREE DISTINCT times (the trap-held gaps keep the frames separate);
      * cool the cloud ONCE and hold the trap continuously (no re-cool between frames);
      * expose the two exposures as API slots a1 (the long reference frames) + a2 (the short
        readout), so the cali sets ONLY those durations BY NAME -- file == fired."""
    from Zou_lab_control.neutral_atom.timing import default_imaging_template
    from Zou_lab_control.neutral_atom.devices.camera_trigger import count_trigger_pulses

    st = default_imaging_template()
    # the FILE itself triggers the camera 3x (not 1) -- a continuous bracket, no unroll needed
    assert count_trigger_pulses(st.to_sequence(name="b")) == 3
    cooling = [p for p in st.to_sequence(name="b").pulses if p.channel == "cooling" and p.value]
    trap = [p for p in st.to_sequence(name="b").pulses if p.channel == "trap" and p.value]
    assert len(cooling) == 1 and len(trap) == 1                 # one cooling load, trap held whole time

    # three exposures, three UNIQUE API handles: a1=first long, a2=short readout, a3=second long
    assert st.api_names() == ["a1", "a2", "a3"]
    # setting the slots by name changes ONLY those durations -> the long-short-long the cali fires
    st.set_api("a1", 0.03)
    st.set_api("a2", 0.008)
    st.set_api("a3", 0.03)
    probe = [round(p.duration, 6) for p in sorted(
        (p for p in st.to_sequence(name="b").pulses if p.channel == "probe" and p.value), key=lambda p: p.start)]
    assert probe == [pytest.approx(0.03), pytest.approx(0.008), pytest.approx(0.03)]
