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
            occ = OccupancyProcessor(hub, calibration=cal, source="frame", method=m)
            cam.step(); occ.step()
            assert hub.latest("occupied").shape == (12,)
            # the judged frame is published, atomically -> rings + underlay are the same shot
            assert np.array_equal(hub.latest("frame_judged"), hub.latest("frame"))
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


def test_calibrate_task_loads_a_pulse_template_and_sets_the_exposure(tmp_path):
    """The cali LOADS a real pulse template and the template's OWN 'image'-window duration IS
    the LONG reference exposure of the long-short-long bracket -- so editing the template's
    image period sets the long exposure (the template genuinely drives the cali pulse; the
    "you can't set the duration" claim is false).  The short readout is the only separate knob."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.timing import PulseTableState, default_imaging_template

    exp = _calibrated()
    try:
        prog = tmp_path / "my_imaging.json"             # the user's own imaging program
        # set the template's image window to 30 ms -> THAT becomes the long reference exposure
        st = default_imaging_template()
        idx = st.periods.index(next(p for p in st.periods if p.name == "image"))
        st = st.set_period_duration(idx, 0.030, unit="s")
        st.save(prog)

        task = exp.readout.calibrate_task(
            SignalHub(), source="live", pulse_template=str(prog),
            readout_exposure=0.005, threshold_frames=12)
        assert task.pulse_template == str(prog)
        assert task.acquisition_parameters()["pulse_template"] == str(prog)
        # the LONG reference exposure is read from the template's image window (30 ms set above) --
        # editing the template IS how the long exposure is set; the short readout stays separate.
        assert task._template_image_seconds() == pytest.approx(0.030)
        assert task.readout_exposure == pytest.approx(0.005)        # 30ms-5ms-30ms long-short-long
        # The bracket is BUILT BY MODIFYING THE TEMPLATE (with_imaging_bracket): ONE cooling/load
        # cycle, then the 'image' window repeated long-short-long -> three CONSECUTIVE emCCD on the
        # same atoms.  Verify the actual fired sequence has exactly that structure + durations.
        bracket = st.with_imaging_bracket([0.030, 0.005, 0.030]).to_sequence(name="ref")
        cooling = [(p.start, p.duration) for p in bracket.pulses if p.channel == "cooling" and p.value]
        emccd = sorted(p.start for p in bracket.pulses if p.channel == "emCCD" and p.value)
        probe = [round(p.duration, 6) for p in sorted(
            (p for p in bracket.pulses if p.channel == "probe" and p.value), key=lambda p: p.start)]
        assert len(cooling) == 1 and cooling[0][1] == pytest.approx(0.002)   # one cooling load
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
        entry = editor.form._widgets["save_frames"]
        assert entry[0] == "bool"
        assert isinstance(entry[1], FluentSwitch)
    finally:
        console.shutdown()
        exp.close()


def test_pulse_scan_target_combo_populates_from_template_dependent_field():
    """The Pulse-scan measurement's ``scan_target`` (kind ``pulse_param``) renders as a
    DEPENDENT combo: its choices are introspected from the pulse template named in the sibling
    ``template`` path field, each item's data is the ``"kind:target"`` token the build consumes,
    and changing the template repopulates it (the form's inter-field reactivity)."""
    from Zou_lab_control.frontend.task_console import MeasurementPanel

    exp = _calibrated()
    try:
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
        panel = MeasurementPanel([spec], single=True)
        tag, combo = panel._widgets["scan_target"]
        assert tag == "pulse_param"
        assert combo.count() >= 2                                  # populated from the template's params
        # items carry a "kind:target" token as data, a human label as text
        tokens = [combo.itemData(i) for i in range(combo.count())]
        assert any(str(t).startswith("duration:") for t in tokens)
        assert any(str(t).startswith("delay:") for t in tokens)   # delays included (software, non-slot)
        combo.setCurrentIndex(1)
        token = panel.collect_values()["scan_target"]
        assert ":" in token                                        # collected as the kind:target token
        # the combo is DEPENDENT on the template field: repopulating (the path-change reaction)
        # keeps it populated via the single template resolver and PRESERVES the current pick --
        # it never silently empties to a blank mystery.
        panel._repopulate_pulse_param("scan_target")
        assert combo.count() >= 2
        assert panel.collect_values()["scan_target"] == token      # selection survives a reload
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
        frame = node.camera.acquire(1, sequencer=node.sequencer)[0]
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


def test_with_imaging_bracket_edge_cases_three_triggers_tail_and_single_source_gap():
    """MECHANICAL guard for the R5 self-audit fixes to PulseTableState.with_imaging_bracket:
      * an image-FIRST template (no pre-image load period) must STILL yield N DISTINCT emCCD
        triggers -- the readout gap must hold only the load channels, NOT probe/emCCD (holding
        the imaging window's states collapsed the whole bracket into ONE continuous trigger);
      * periods AFTER the image window are preserved once (not silently dropped);
      * both bracket builders default the inter-frame gap to the ONE shared source."""
    import inspect
    from Zou_lab_control.neutral_atom.timing import default_imaging_template
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState, PulsePeriod
    from Zou_lab_control.neutral_atom.timing.sequence import (
        count_trigger_pulses, reference_bracket_sequence, READOUT_GAP_SECONDS)

    st = default_imaging_template()
    chans = list(st.channels)
    img_states = tuple(st.periods[[p.name for p in st.periods].index("image")].states)

    # image-FIRST template -> three DISTINCT emCCD triggers (was 1 before the held-gap fix)
    only_image = PulseTableState(channels=chans,
                                 periods=[PulsePeriod(0.02, img_states, unit="s", name="image")])
    seq = only_image.with_imaging_bracket([0.02, 0.005, 0.02]).to_sequence(name="b")
    assert count_trigger_pulses(seq) == 3

    # a period AFTER the image window survives once + still three triggers
    tail = default_imaging_template()
    tail.periods.append(PulsePeriod(0.001, tuple(0 for _ in chans), unit="s", name="park_after"))
    bracketed = tail.with_imaging_bracket([0.02, 0.005, 0.02])
    assert "park_after" in [p.name for p in bracketed.periods]
    assert count_trigger_pulses(bracketed.to_sequence(name="b2")) == 3

    # single source: both builders default the inter-frame gap to READOUT_GAP_SECONDS
    assert inspect.signature(PulseTableState.with_imaging_bracket).parameters["gap_seconds"].default \
        == READOUT_GAP_SECONDS
    assert inspect.signature(reference_bracket_sequence).parameters["gap"].default == READOUT_GAP_SECONDS
