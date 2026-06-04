import matplotlib

matplotlib.use("Agg")

import importlib.util
import json
import queue
from pathlib import Path
import re
import socket
import subprocess
import time
import sys
import threading
import types
from urllib.parse import unquote

import numpy as np
from matplotlib.patches import Circle

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.frontend.content.tutorials import neutral_atom_fpga_server_cells, neutral_atom_hardware_tutorial_cells


def wait_until_done(task, *, timeout=10.0):
    measurement = task.measurement
    assert measurement is not None
    deadline = time.perf_counter() + timeout
    while not measurement.done and time.perf_counter() < deadline:
        time.sleep(0.01)
    measurement.refresh(draw=False)
    if measurement.error is not None:
        raise measurement.error
    assert measurement.done
    task.plot = measurement.plot
    return task


def test_virtual_jupyter_session_runs_end_to_end(tmp_path):
    exp = na.connect("virtual")

    for attr in ("camera", "readout", "timing"):
        assert hasattr(exp, attr)
    assert not hasattr(exp, "sites")
    assert not hasattr(exp, "threshold")
    assert not hasattr(exp, "detector")
    assert isinstance(exp.camera, na.CameraDevice)
    assert exp.calibration_data is None

    preflight = exp.timing.preflight()
    capture = exp.camera.capture(display=False)
    assert [patch for patch in capture.plot.ax.patches if isinstance(patch, Circle)] == []
    assert "truth_available" not in capture.summary()
    assert not hasattr(capture, "truth")
    assert not hasattr(exp.camera, "last_truth")
    assert not hasattr(exp.devices.trap_array, "centers")
    sitemap = exp.readout.sitemap(frames=6, display=False)
    capture_after_sitemap = exp.camera.capture(display=False)
    assert [patch for patch in capture_after_sitemap.plot.ax.patches if isinstance(patch, Circle)] == []
    threshold = exp.readout.thresholds(frames=12, display=False)
    capture_after_threshold = exp.camera.capture(display=False)
    assert [patch for patch in capture_after_threshold.plot.ax.patches if isinstance(patch, Circle)] == []
    detection = exp.readout.detect(display=False)
    scan = exp.readout.detection_time([5e-6, 2e-5, 0.0002], shots=8, live=False, display=False)
    verilog_path = exp.timing.write_verilog(tmp_path)
    cal_path = exp.readout.save(tmp_path / "calibration.json")

    assert preflight.ok
    assert capture.image.shape == exp.devices.trap_array.image_shape
    assert capture.data_figure is capture.plot.data_figure
    assert sitemap.calibration.n_sites == exp.devices.trap_array.n_sites
    assert sitemap.data_figure is sitemap.plot.data_figure
    assert threshold.counts.shape == (12, exp.devices.trap_array.n_sites)
    assert threshold.plot is not None
    assert threshold.data_figure is threshold.plot.data_figure
    assert detection.occupied.shape == (exp.devices.trap_array.n_sites,)
    assert detection.plot is not None
    assert detection.data_figure is detection.plot.data_figure
    assert not hasattr(detection, "truth")
    sitemap_radius = next(patch.radius for patch in sitemap.plot.ax.patches if isinstance(patch, Circle))
    detect_radii = [patch.radius for patch in detection.plot.ax.patches if isinstance(patch, Circle)]
    assert sitemap_radius in detect_radii
    assert sitemap_radius >= 4.5
    assert scan.summary()["finished"] is True
    assert np.all(np.isfinite(scan.fidelities))
    assert scan.reference_threshold is not None
    assert verilog_path.exists()
    assert cal_path.exists()
    status = exp.status()
    assert "occupied" not in status["devices"]["trap_array"]
    assert "last_truth_count" not in status["devices"]["camera"]
    assert "NeutralAtomSession" in exp._repr_html_()


def test_device_contracts_are_explicit_and_validated():
    assert issubclass(na.VirtualCamera, na.CameraDevice)
    assert issubclass(na.QCMOSCamera, na.CameraDevice)
    assert issubclass(na.ManualSequencer, na.SequencerDevice)
    assert issubclass(na.RemoteSequencer, na.SequencerDevice)
    assert issubclass(na.RuntimeSequencer, na.SequencerDevice)
    assert issubclass(na.VirtualSequencer, na.SequencerDevice)
    assert issubclass(na.VerilogSequencer, na.SequencerDevice)
    assert issubclass(na.VirtualTrapArray, na.TrapArrayDevice)

    try:
        na.load_devices({"camera": {"type": "builtins.object"}})
    except TypeError as exc:
        assert "must inherit CameraDevice" in str(exc)
    else:
        raise AssertionError("invalid camera device should fail at load time")

    try:
        class IncompleteCamera(na.CameraDevice):
            pass

        IncompleteCamera()
    except TypeError as exc:
        assert "abstract" in str(exc)
    else:
        raise AssertionError("incomplete CameraDevice subclass should not instantiate")


def test_device_registry_can_register_external_classes():
    class RegisteredSequencer(na.RuntimeSequencer):
        pass

    na.register_device_class("RegisteredSequencerForTest", RegisteredSequencer)
    devices = na.load_devices(
        {
            "sequencer": {
                "type": "RegisteredSequencerForTest",
                "params": {"channels": ["trap", "cooling", "probe", "qcm_trigger"]},
            }
        }
    )

    registry = na.device_class_registry()
    assert isinstance(devices.sequencer, RegisteredSequencer)
    assert registry["RegisteredSequencerForTest"].endswith("RegisteredSequencer")
    assert registry["QCMOSCamera"].endswith(".QCMOSCamera")


def test_public_standalone_operations_live_outside_session_subsystem():
    assert na.calibrate_sitemap_from_images.__module__.endswith("operations.calibration")
    assert na.calibrate_threshold_from_images.__module__.endswith("operations.calibration")
    assert na.detect_image.__module__.endswith("operations.detection")
    assert issubclass(na.ReadoutSubsystem, na.ExperimentSubsystem)
    assert na.ReadoutSubsystem.__module__.endswith("subsystems.readout")


def test_live_detection_scan_uses_frontend_session():
    exp = na.connect("virtual")
    exp.readout.sitemap(frames=4, display=False)

    scan = exp.readout.detection_time([5e-6, 2e-5, 8e-5], shots=3, display=False, update_time=0.01)
    assert scan.measurement is not None
    assert scan.plot is scan.measurement.plot
    assert scan.points_done >= 0
    wait_until_done(scan)

    assert scan.measurement is not None
    assert scan.plot is not None
    assert scan.data_figure is scan.plot.data_figure
    assert np.all(np.isfinite(scan.fidelities))
    assert scan.summary()["finished"] is True
    fit_result, popt = scan.data_figure.decay(is_display=False)
    assert fit_result.function == "decay"
    assert popt is not None


def test_live_detection_scan_can_be_interrupted():
    exp = na.connect("virtual")
    exp.readout.sitemap(frames=4, display=False)

    clock_hz = exp.devices.sequencer.clock_hz
    time_ticks = np.linspace(int(round(5e-6 * clock_hz)), int(round(2e-3 * clock_hz)), 25, dtype=int)
    scan = exp.readout.detection_time(time_ticks / clock_hz, shots=5, display=False, update_time=0.01)
    assert scan.measurement is not None
    scan.stop()
    assert scan.plot._stopped

    assert not scan.running
    assert scan.points_done <= len(scan.times)


def test_threshold_requires_sitemap_first():
    cfg = {
        "trap_array": {"type": "VirtualTrapArray", "params": {"grid_shape": [1, 2], "image_shape": [32, 40]}},
        "camera": {"type": "VirtualCamera", "params": {"trap_array": "$device:trap_array"}},
        "sequencer": {"type": "VirtualSequencer"},
    }
    exp = na.connect(cfg)
    assert exp.readout.current is None

    try:
        exp.readout.thresholds(frames=2, display=False)
    except RuntimeError as exc:
        assert "exp.readout.sitemap" in str(exc)
    else:
        raise AssertionError("threshold calibration should require a site map")

    try:
        exp.readout.detect(display=False)
    except RuntimeError as exc:
        assert "exp.readout.sitemap" in str(exc)
    else:
        raise AssertionError("detection should require a site map and thresholds")


def test_virtual_config_accepts_experiment_parameters():
    exp = na.connect(
        "virtual",
        bright_count_rate=1234,
        background_count_rate=9,
        loss_rate=0.2,
        exposure=0.001,
        sitemap={"grid_shape": (2, 3), "image_shape": (48, 64), "roi_radius": 2, "sitemap_exposure": 0.004},
    )

    assert exp.devices.trap_array.atom_rate == 1234
    assert exp.devices.trap_array.background_rate == 9
    assert exp.devices.trap_array.detection_lifetime == 5
    assert exp.devices.trap_array.grid_shape == (2, 3)
    assert exp.devices.trap_array.image_shape == (48, 64)
    assert exp.camera.exposure == 0.001

    sitemap = exp.readout.sitemap(frames=3, display=False)
    assert sitemap.calibration.roi_radius == 2


def test_timing_and_verilog_boundaries(tmp_path):
    seq = na.imaging_sequence(exposure=1e-3, load=True).delay("qcm_trigger", 4e-9)
    report = seq.validate(clock_hz=250e6, channels=["trap", "cooling", "probe", "qcm_trigger"])
    build = na.generate_verilog(seq, channels=["trap", "cooling", "probe", "qcm_trigger"], clock_hz=250e6)
    files = na.write_verilog_bundle(build, tmp_path)

    assert report.ok
    assert seq.delays == {"qcm_trigger": 4e-09}
    assert build.ticks
    assert files.verilog_path.exists()
    assert files.manifest_path.exists()


def test_multiframe_acquisition_repeats_camera_trigger_sequence():
    exp = na.connect("virtual")
    seq = na.imaging_sequence(exposure=1e-3, load=True, name="multi_frame")
    expanded = na.sequence_for_frame_count(seq, 3)
    images = exp.camera.acquire(3, sequence=seq, sequencer=exp.devices.sequencer)

    assert len(images) == 3
    assert na.count_trigger_pulses(seq) == 1
    assert na.count_trigger_pulses(expanded) == 3
    prepares = [row for row in exp.devices.sequencer.history if row["action"] == "prepare"]
    fires = [row for row in exp.devices.sequencer.history if row["action"] == "fire"]
    assert prepares[-1]["duration"] == expanded.duration
    assert fires[-1]["duration"] == expanded.duration


def test_pulse_sequence_repeat_metadata_does_not_expand_for_runtime_counts():
    seq = na.PulseSequence(name="huge_repeat").pulse("qcm_trigger", 0.0, 1e-6).repeated(100_000, period=2e-6)
    report = seq.validate(clock_hz=100_000_000, channels=["trap", "cooling", "probe", "qcm_trigger"])
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "qcm_trigger"], clock_hz=100_000_000)

    assert report.ok
    assert na.count_trigger_pulses(seq) == 100_000
    assert program.repeat_forever is False
    assert program.loop_count == 100_000
    assert len(program.ticks) < 16


def test_detection_time_scan_sends_probe_duration_per_point():
    exp = na.connect("virtual", sitemap={"grid_shape": (1, 2), "image_shape": (32, 48)})
    exp.readout.sitemap(frames=3, display=False)
    exp.devices.sequencer.history.clear()

    times = np.array([8e-6, 20e-6, 60e-6])
    exp.readout.detection_time(times, shots=4, reference_shots=2, live=False, display=False)

    prepares = [row for row in exp.devices.sequencer.history if row["action"] == "prepare"]
    scan_prepares = [row for row in prepares if row["sequence"] == "detect_time_scan"]
    expected = [
        na.sequence_for_frame_count(na.imaging_sequence(exposure=float(t), load=True, name="detect_time_scan"), 4).duration
        for t in times
    ]
    assert [row["duration"] for row in scan_prepares] == expected


def test_runtime_sequencer_service_contract():
    seq = na.sequence_for_frame_count(na.imaging_sequence(exposure=1e-4, load=True), 2)
    sequencer = na.RuntimeSequencer(channels=["trap", "cooling", "probe", "qcm_trigger"], sleep_scale=0.0)
    program = sequencer.prepare(seq)
    sequencer.fire(seq)

    assert program.trigger_count == 2
    assert sequencer.wait_done(timeout=1.0)
    snapshot = sequencer.snapshot()
    assert snapshot["state"] == "done"
    assert snapshot["prepared_program"]["trigger_count"] == 2


def test_fpga_pulse_streamer_writes_hdl_and_upload_tcl(tmp_path):
    seq = na.sequence_for_frame_count(na.imaging_sequence(exposure=4e-6, load=True), 2)
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "qcm_trigger"], clock_hz=100e6)

    files = na.write_pulse_streamer_hdl_bundle(tmp_path / "hdl", channels=["trap", "cooling", "probe", "qcm_trigger"], max_edges=16)
    tcl_path = na.write_vivado_pulse_streamer_tcl(
        tmp_path / "prepare.tcl",
        "prepare",
        program=program,
        project="D:/fake/project.xpr",
        bitstream="D:/fake/main.bit",
        probes="D:/fake/main.ltx",
        max_edges=16,
        channel_count=4,
    )

    na.validate_pulse_streamer_program(program, max_edges=16, channel_count=4)
    core = files.core_path.read_text(encoding="utf-8")
    top = files.top_example_path.read_text(encoding="utf-8")
    tcl = tcl_path.read_text(encoding="utf-8")
    assert "module zlc_pulse_streamer" in core
    assert "Runtime-programmable edge-table pulse streamer" in core
    assert '(* ram_style = "distributed" *)' in core
    assert "first_tick_shadow" in core
    assert "loop_start_mask_shadow" in core
    assert "tick_mem[0]" not in core
    assert "mask_mem[0]" not in core
    assert "mask_mem[loop_start_active]" not in core
    assert "reg reset_meta = 1'b0;" in core
    assert "reg reset_sync = 1'b0;" in core
    assert "reg start_meta = 1'b0;" in core
    assert "reg prog_we_meta = 1'b0;" in core
    assert "wire start_event = start_sync && !start_prev" in core
    assert "wire start_event = start_sync != start_prev" not in core
    assert "wire prog_we_event = prog_we_sync != prog_we_prev" in core
    assert "if (reset_sync && prog_we_event)" in core
    assert "probe_out4 zlc_prog_tick" in top
    assert "probe_out10 zlc_loop_count" in top
    assert "proc zlc_stage_probe" in tcl
    assert "proc zlc_commit_probes" in tcl
    assert "global zlc_probe_cache" in tcl
    assert "set cache_key" in tcl
    assert "zlc_probe_cache($cache_key)" in tcl
    assert "ZLC_PS_VERBOSE_VIO" in tcl
    assert "global zlc_verbose_vio" in tcl
    assert "zlc_commit_probes $zlc_batch" in tcl
    assert "zlc_stage_probe $vio $zlc_prog_count_probe" in tcl
    assert "zlc_stage_probe $vio $zlc_repeat_forever_probe" in tcl
    assert "zlc_stage_probe $vio $zlc_loop_start_addr_probe" in tcl
    assert "zlc_stage_probe $vio $zlc_loop_end_tick_probe" in tcl
    assert "zlc_stage_probe $vio $zlc_loop_count_probe" in tcl
    assert "zlc_stage_probe $vio $zlc_prog_tick_probe" in tcl
    assert "zlc_stage_probe $vio $zlc_prog_mask_probe" in tcl
    assert "zlc_start_toggle_value" not in tcl
    assert "Available probes on matched VIO:" in tcl
    assert "Vivado project not found" in tcl
    assert "Vivado probe file not found" in tcl
    assert "Vivado bitstream not found for programming" in tcl
    assert "set zlc_reset_probe {zlc_reset probe_out0}" in tcl
    assert "set zlc_prog_tick_probe {zlc_prog_tick probe_out4}" in tcl
    assert "set zlc_repeat_forever_probe {zlc_repeat_forever probe_out7}" in tcl
    assert "set zlc_loop_count_probe {zlc_loop_count probe_out10}" in tcl
    assert "set zlc_done_probe {zlc_done probe_in1}" in tcl
    assert "string match \"*/$name\"" in tcl
    assert "probe aliases" in tcl
    assert f"wrote {len(program.ticks)}/{len(program.ticks)} edge rows" in tcl


def test_fpga_pulse_streamer_rejects_program_that_does_not_fit():
    seq = na.sequence_for_frame_count(na.imaging_sequence(exposure=4e-6, load=True), 2)
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "qcm_trigger"], clock_hz=100e6)

    try:
        na.validate_pulse_streamer_program(program, max_edges=1, channel_count=4)
    except ValueError as exc:
        assert "edges" in str(exc)
    else:
        raise AssertionError("pulse-streamer validation should reject too many edges")


def test_fpga_pulse_streamer_rejects_runtime_edge_table_hazards():
    bad_duplicate_tick = na.RuntimeSequenceProgram(
        sequence_id="bad",
        sequence_name="bad",
        clock_hz=100e6,
        channels=["trap", "qcm_trigger"],
        ticks=[0, 10, 10],
        masks=[1, 2, 0],
        duration=1e-7,
        trigger_count=1,
    )
    bad_final_mask = na.RuntimeSequenceProgram(
        sequence_id="bad",
        sequence_name="bad",
        clock_hz=100e6,
        channels=["trap", "qcm_trigger"],
        ticks=[0, 10],
        masks=[1, 1],
        duration=1e-7,
        trigger_count=1,
    )
    bad_duplicate_channel = na.RuntimeSequenceProgram(
        sequence_id="bad",
        sequence_name="bad",
        clock_hz=100e6,
        channels=["trap", "trap"],
        ticks=[0, 10],
        masks=[1, 0],
        duration=1e-7,
        trigger_count=0,
    )

    for program, expected in (
        (bad_duplicate_tick, "strictly increasing"),
        (bad_final_mask, "final mask"),
        (bad_duplicate_channel, "unique"),
    ):
        try:
            na.validate_pulse_streamer_program(program, channel_count=2)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"pulse-streamer validation should reject {expected}")


def test_fpga_pulse_streamer_rejects_bad_top_level_channel_names():
    for channels, expected in (
        (["a-b", "a_b"], "collide"),
        (["clk", "probe"], "top-level"),
    ):
        try:
            na.generate_pulse_streamer_top_example(channels=channels)
        except ValueError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"top example should reject {channels!r}")


def test_fpga_pulse_streamer_fire_dry_run_does_not_require_program_file(tmp_path):
    from Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer import run_action

    tcl_path = run_action("fire", state_dir=tmp_path, dry_run=True)

    tcl = tcl_path.read_text(encoding="utf-8")
    assert "if {[zlc_output_probe_bool $vio $zlc_start_probe]} {" in tcl
    assert "zlc_stage_probe $vio $zlc_start_probe 1" in tcl
    assert "zlc_set_probe $vio $zlc_start_probe 0" in tcl
    assert "ZLC pulse-streamer start pulse sent" in tcl


def test_fpga_pulse_streamer_dry_run_uses_project_local_artifacts(tmp_path, monkeypatch):
    from Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer import run_action

    root = Path(__file__).resolve().parents[1]
    unsafe_project_dir = root / "fpga" / "pulse_streamer" / "build" / "zlc_pulse_streamer_40ch"
    unsafe_bitstream = unsafe_project_dir / "zlc_pulse_streamer_40ch.runs" / "impl_1" / "zlc_pulse_streamer_top_40ch.bit"
    unsafe_probes = unsafe_project_dir / "zlc_pulse_streamer_40ch.runs" / "impl_1" / "zlc_pulse_streamer_top_40ch.ltx"
    default_project_dir = root / "fpga" / "build" / "p40"
    monkeypatch.setenv("ZLC_PS_PROJECT_DIR", str(unsafe_project_dir))
    monkeypatch.setenv("ZLC_PS_VIVADO_PROJECT", str(unsafe_project_dir / "zlc_pulse_streamer_40ch.xpr"))
    monkeypatch.setenv("ZLC_PS_VIVADO_BIT", str(unsafe_bitstream))
    monkeypatch.setenv("ZLC_PS_VIVADO_LTX", str(unsafe_probes))
    for name in (
        "ZLC_PS_PROJECT_ROOT",
        "ZLC_VIVADO_PROJECT",
        "ZLC_VIVADO_BIT",
        "ZLC_VIVADO_LTX",
    ):
        monkeypatch.delenv(name, raising=False)

    tcl_path = run_action("fire", state_dir=tmp_path, dry_run=True)

    tcl = tcl_path.read_text(encoding="utf-8")
    assert str(unsafe_project_dir) not in tcl
    assert str(unsafe_bitstream) not in tcl
    assert str(unsafe_probes) not in tcl
    assert str(default_project_dir) in tcl
    assert "p40.xpr" in tcl
    assert "p40.runs" in tcl
    assert "zlc_pulse_streamer_top_40ch.bit" in tcl
    assert "zlc_pulse_streamer_top_40ch.ltx" in tcl


def test_fpga_pulse_streamer_module_cli_generates_hdl(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer",
            "generate_hdl",
            "--output-dir",
            str(tmp_path),
            "--channels",
            "trap",
            "cooling",
            "probe",
            "qcm_trigger",
            "--max-edges",
            "16",
            "--tick-width",
            "32",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout
    assert "RuntimeWarning" not in result.stdout
    assert (tmp_path / "zlc_pulse_streamer.v").exists()
    assert (tmp_path / "zlc_pulse_streamer_top_example.v").exists()
    assert (tmp_path / "zlc_pulse_streamer.manifest.json").exists()


def test_pulse_gui_launcher_aligns_subset_state_to_full_hardware_channels(monkeypatch):
    monkeypatch.delenv("ZLC_PS_REMOTE_HOST", raising=False)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("zlc_root_pulse_gui_for_test", root / "pulse_gui.py")
    assert spec is not None and spec.loader is not None
    pulse_gui_launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pulse_gui_launcher)

    args = types.SimpleNamespace(
        channels=None,
        channel_count=None,
        xdc=root / "fpga" / "pulse_streamer" / "zlc_pulse_streamer_40ch.xdc",
        max_channel_count=40,
        trigger_channels=None,
    )
    subset_state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod(100, (1, 0), unit="ns"),
            na.PulsePeriod(20, (1, 1), unit="ns"),
            na.PulsePeriod(100, (0, 0), unit="ns"),
        ],
        visible_channels=["ch00", "ch03"],
        channel_labels={"ch00": "trap", "ch03": "qcm_trigger"},
        time_step_ns=10,
        repeat_forever=True,
    )

    channels = pulse_gui_launcher._resolve_channels(args, subset_state)
    aligned = subset_state.aligned_to_channels(channels)
    program = aligned.compile(clock_hz=100_000_000, trigger_channels=["ch03"])

    assert channels == [f"ch{i:02d}" for i in range(40)]
    assert aligned.channels == channels
    assert aligned.visible_channels == ["ch00", "ch03"]
    assert all(period.states[1:3] == (0, 0) for period in aligned.periods)
    assert max(program.masks) < (1 << 40)
    assert pulse_gui_launcher._resolve_trigger_channels(args, channels) == ["ch03"]
    labels = pulse_gui_launcher._resolve_channel_labels(args, channels, subset_state)
    assert labels["ch00"] == "trap"
    assert labels["ch03"] == "qcm_trigger"
    assert labels["ch04"] == "cooling_pgc"
    assert labels["ch39"] == "da_bias_x[7]"

    explicit_args = types.SimpleNamespace(**{**args.__dict__, "channel_count": 4})
    assert pulse_gui_launcher._resolve_channels(explicit_args, subset_state) == ["ch00", "ch01", "ch02", "ch03"]

    remote_args = types.SimpleNamespace(
        **{
            **args.__dict__,
            "remote_host": "127.0.0.1",
            "remote_port": 18861,
            "clock_hz": 100_000_000,
        }
    )

    class FailingRemoteNa:
        class RemoteSequencer:
            def __init__(self, **_kwargs):
                raise ConnectionRefusedError("server is not running")

    sequencer, fallback_channels, fallback_triggers, notice = pulse_gui_launcher._connect_remote_or_offline(
        remote_args,
        subset_state,
        FailingRemoteNa,
        explicit_remote=False,
    )
    assert sequencer is None
    assert fallback_channels == channels
    assert fallback_triggers == ["ch03"]
    assert "opened offline editor" in notice
    assert pulse_gui_launcher._remote_host_was_requested([]) is False
    assert pulse_gui_launcher._remote_host_was_requested(["--remote-host", "192.168.0.20"]) is True

    try:
        pulse_gui_launcher._connect_remote_or_offline(remote_args, subset_state, FailingRemoteNa, explicit_remote=True)
    except ConnectionRefusedError:
        pass
    else:
        raise AssertionError("explicit --remote-host should not silently fall back to offline mode")


def test_fpga_pulse_streamer_repo_vivado_entrypoint_contract():
    root = Path(__file__).resolve().parents[1]
    fpga = root / "fpga" / "pulse_streamer"
    root_readme = (root / "README.md").read_text(encoding="utf-8")
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")
    install_bat = (root / "install_requirements.bat").read_text(encoding="utf-8")
    install_kernel_py = (root / "install_current_kernel.py").read_text(encoding="utf-8")
    pulse_gui_bat = (root / "pulse_gui.bat").read_text(encoding="utf-8")
    pulse_gui_launcher = (root / "pulse_gui.py").read_text(encoding="utf-8")
    tutorials_bat = (root / "start_tutorials_jupyter_lab.bat").read_text(encoding="utf-8")
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    runbook = (root / "docs" / "neutral_atom_hardware_manual" / "REAL_HARDWARE_RUNBOOK.md").read_text(encoding="utf-8")
    project_overview = (root / "docs" / "PROJECT_OVERVIEW.md").read_text(encoding="utf-8")
    pulse_report = (
        root / "docs" / "pulse_streamer_test_report" / "pulse_streamer_test_report_zh.texbody"
    ).read_text(encoding="utf-8")
    fpga_readme = (root / "fpga" / "README.md").read_text(encoding="utf-8")
    fpga_streamer_readme = (root / "fpga" / "pulse_streamer" / "README.md").read_text(encoding="utf-8")
    frontend_readme = (root / "Zou_lab_control" / "frontend" / "README.md").read_text(encoding="utf-8")
    frontend_manual_template = (
        root / "Zou_lab_control" / "frontend" / "content" / "manual_templates" / "frontend_manual_zh.texbody"
    ).read_text(encoding="utf-8")
    frontend_manual_pdf_tex = (root / "docs" / "frontend_manual" / "frontend_manual_zh.tex").read_text(encoding="utf-8")
    pulses_readme = (root / "pulses" / "README.md").read_text(encoding="utf-8")
    tests_readme = (root / "tests" / "README.md").read_text(encoding="utf-8")
    hardware_quickstart_template = (
        root / "Zou_lab_control" / "neutral_atom" / "content" / "manual_templates" / "hardware_quickstart_zh.texbody"
    ).read_text(encoding="utf-8")
    hardware_quickstart_pdf_tex = (
        root / "docs" / "neutral_atom_hardware_manual" / "neutral_atom_hardware_quickstart_zh.tex"
    ).read_text(encoding="utf-8")
    for name in ("install_requirements.bat", "pulse_gui.bat", "start_tutorials_jupyter_lab.bat"):
        assert (root / name).exists(), name
    assert not (root / "build_and_program.bat").exists()
    assert not (root / "run_server.bat").exists()
    assert "standalone frontend layer" in root_readme
    assert "standalone FPGA pulse-streamer hardware side" in root_readme
    assert "fpga\\build_and_program.bat --check" in root_readme
    assert "fpga\\build\\p40" in root_readme
    assert "%run ../install_current_kernel.py" in root_readme
    assert "Pulse presets" in root_readme
    assert "Test strategy" in root_readme
    root_bats = {path.name for path in root.glob("*.bat")}
    assert root_bats == {"install_requirements.bat", "pulse_gui.bat", "start_tutorials_jupyter_lab.bat"}
    legacy_root_bats = {"build_4ch_bitstream.bat", "start_server_4ch.bat", "simulate_4ch_core.bat", "smoke_test_4ch_upload.bat"}
    assert root_bats.isdisjoint(legacy_root_bats)
    assert not list((root / "docs" / "pulse_streamer_test_report" / "assets").glob("*4ch*"))

    fpga_build_bat = root / "fpga" / "build_and_program.bat"
    fpga_server_bat = root / "fpga" / "run_server.bat"
    assert fpga_build_bat.exists()
    assert fpga_server_bat.exists()
    assert (root / "fpga" / "README.md").exists()
    assert (root / "Zou_lab_control" / "frontend" / "README.md").exists()
    assert not list(fpga.glob("*.bat"))

    required = [
        "zlc_pulse_streamer.v",
        "zlc_pulse_streamer_top_40ch.v",
        "zlc_pulse_streamer_40ch.xdc",
        "zlc_pulse_streamer_40ch.xdc.template",
        "create_project_40ch.tcl",
        "check_40ch_synth.tcl",
        "diagnose_hw_target.tcl",
        "program_fpga_40ch.tcl",
        "README.md",
    ]
    for name in required:
        assert (fpga / name).exists(), name
    for dependency in ("PyQt5", "PyQt-Frameless-Window", "rpyc", "ipykernel", "jupyterlab", "nbconvert", "notebook"):
        assert dependency in pyproject
        assert dependency in requirements
    assert 'pip install -e "%~dp0."' in install_bat
    assert ".zlc_python_path" in install_bat
    assert "sys.executable" in install_bat
    assert "--help" in install_bat
    assert "install_requirements.bat failed with code" in install_bat
    assert ":install_failed" in install_bat
    assert "pip\", \"install\", \"-e\", str(ROOT)" in install_kernel_py
    assert "PYTHON_PATH_RECORD.write_text(sys.executable" in install_kernel_py
    assert ".zlc_python_path" in install_kernel_py
    assert ".zlc_python_path" in gitignore
    assert "ZLC_PULSE_GUI_PYTHON" in pulse_gui_bat
    assert ".zlc_python_path" in pulse_gui_bat
    assert "ZLC_PULSE_GUI_INNER" in pulse_gui_bat
    assert "ZLC pulse GUI failed with code" in pulse_gui_bat
    assert "Keep this window open and read the messages above." in pulse_gui_bat
    assert 'if "%ZLC_NO_PAUSE%"=="" pause' in pulse_gui_bat
    assert "DEFAULT_PULSE_GUI_MAX_CHANNELS = 40" in pulse_gui_launcher
    assert "infer_xdc_channel_count" in pulse_gui_launcher
    assert "connect_on_init=True" in pulse_gui_launcher
    assert "opened offline editor" in pulse_gui_launcher
    assert "_remote_host_was_requested" in pulse_gui_launcher
    assert "ZLC_PULSE_GUI_AUTO_CLOSE_MS" in pulse_gui_launcher
    assert "ZLC_TUTORIALS_PYTHON" in tutorials_bat
    assert ".zlc_python_path" in tutorials_bat
    assert "--check" in tutorials_bat
    assert "ZLC_TUTORIALS_MODE=check" in tutorials_bat
    assert "%ZLC_PYTHON_CMD% -m jupyter lab ." in tutorials_bat
    assert ".zlc_python_path" in runbook
    assert "ZLC_PULSE_GUI_PYTHON" in runbook
    assert "ZLC_TUTORIALS_PYTHON" in runbook
    assert "ZLC_FPGA_SERVER_PYTHON" in runbook
    assert "start_tutorials_jupyter_lab.bat --check" in runbook
    assert "Standalone Entry Points" in project_overview
    assert "The repo root keeps only user-facing launchers" in project_overview
    assert "`fpga/build_and_program.bat`" in project_overview
    assert "`fpga/run_server.bat`" in project_overview
    assert "`ZLC project dir` is the source of truth" in project_overview
    assert "tests/README.md" in project_overview
    assert "prefer the scoped matrix" in project_overview

    top40 = (fpga / "zlc_pulse_streamer_top_40ch.v").read_text(encoding="utf-8")
    tcl40 = (fpga / "create_project_40ch.tcl").read_text(encoding="utf-8")
    check40 = (fpga / "check_40ch_synth.tcl").read_text(encoding="utf-8")
    program_tcl40 = (fpga / "program_fpga_40ch.tcl").read_text(encoding="utf-8")
    diagnose_hw_tcl = (fpga / "diagnose_hw_target.tcl").read_text(encoding="utf-8")
    build_bat = fpga_build_bat.read_text(encoding="utf-8")
    server_bat = fpga_server_bat.read_text(encoding="utf-8")
    core = (fpga / "zlc_pulse_streamer.v").read_text(encoding="utf-8")
    xdc40 = (fpga / "zlc_pulse_streamer_40ch.xdc").read_text(encoding="utf-8")
    fpga_notebook_text = "\n".join(cell["source"] for cell in neutral_atom_fpga_server_cells())
    hardware_notebook_text = "\n".join(cell["source"] for cell in neutral_atom_hardware_tutorial_cells())

    assert '(* ram_style = "distributed" *)' in core
    assert "first_tick_shadow" in core
    assert "loop_start_mask_shadow" in core
    assert "tick_mem[0]" not in core
    assert "mask_mem[loop_start_active]" not in core
    assert "wire [39:0] zlc_prog_mask" in top40
    assert "wire [9:0] zlc_prog_addr" in top40
    assert "wire [10:0] zlc_prog_count" in top40
    assert ".EDGE_ADDR_WIDTH(10)" in top40
    assert "CONFIG.C_PROBE_OUT3_WIDTH {10}" in tcl40
    assert "CONFIG.C_PROBE_OUT5_WIDTH {40}" in tcl40
    assert "CONFIG.C_PROBE_OUT6_WIDTH {11}" in tcl40
    assert "CONFIG.C_PROBE_OUT8_WIDTH {10}" in tcl40
    assert "ZLC_PS_40CH_XDC" in tcl40
    assert "ZLC_PS_XDC" in tcl40
    assert "ZLC create_project_40ch contract: CHANNEL_COUNT=40 MAX_EDGES=1024 EDGE_ADDR_WIDTH=10" in tcl40
    assert "zlc_default_project_root" in tcl40
    assert "[file join $script_dir .. build]" in tcl40
    assert "set project_name p40" in tcl40
    assert "set project_dir [zlc_safe_project_dir" in tcl40
    assert "still contains <PIN_CHxx> placeholders" in tcl40
    assert "repo normally includes zlc_pulse_streamer_40ch.xdc derived from address_switch" in tcl40
    assert "zlc_path_under" in tcl40
    assert "Ignoring old fpga/pulse_streamer/build ZLC_PS_PROJECT_DIR" in tcl40
    assert "Move the repo to a shorter project folder such as D:/ZLC" in tcl40
    assert "Create zlc_pulse_streamer_40ch.xdc from zlc_pulse_streamer_40ch.xdc.template" not in tcl40
    assert "CONFIG.C_PROBE_OUT3_WIDTH {10}" in check40
    assert "CONFIG.C_PROBE_OUT5_WIDTH {40}" in check40
    assert "CONFIG.C_PROBE_OUT6_WIDTH {11}" in check40
    assert "ZLC check_40ch_synth contract: CHANNEL_COUNT=40 MAX_EDGES=1024 EDGE_ADDR_WIDTH=10" in check40
    assert "ZLC_PS_CHECK_PROJECT_DIR" in check40
    assert "Ignoring old fpga/pulse_streamer/build ZLC_PS_CHECK_PROJECT_DIR" in check40
    assert "Move the repo to a shorter project folder such as D:/ZLC" in check40
    assert "c40" in check40
    assert "[file join $script_dir build zlc_pulse_streamer_40ch]" not in tcl40
    assert "[file join $script_dir build zlc_pulse_streamer_40ch_check]" not in check40
    assert "EDGE_ADDR_WIDTH bound to: 7" not in pulse_report
    assert "pytest -q tests\\test_neutral_atom_lightweight.py tests\\test_frontend_smoke.py" not in pulse_report
    assert 'tests\\test_neutral_atom_lightweight.py -k "repo_vivado_entrypoint_contract' in pulse_report
    assert 'tests\\test_frontend_smoke.py -k "render_tex_pdf or pulse_gui"' in pulse_report
    assert "zlc_check_utilization" in check40
    assert "40ch synth LUT utilization is too high" in check40
    assert "ZLC 40ch synth check complete" in check40
    assert "ZLC_PS_VIVADO_BIT" in program_tcl40
    assert "ZLC_PS_VIVADO_LTX" in program_tcl40
    assert "ZLC program_fpga_40ch contract: CHANNEL_COUNT=40 MAX_EDGES=1024 EDGE_ADDR_WIDTH=10" in program_tcl40
    assert "zlc_safe_artifact_path" in program_tcl40
    assert "Ignoring old fpga/pulse_streamer/build $label" in program_tcl40
    assert "VIO probe file not found" in program_tcl40
    assert "load_features labtools" in program_tcl40
    assert "open_hw" in program_tcl40
    assert "get_hw_targets" in program_tcl40
    assert "No Vivado hardware target found" in program_tcl40
    assert "Retrying open_hw_target with -jtag_mode on" in program_tcl40
    assert "Target opened but no FPGA devices were detected" in diagnose_hw_tcl
    assert "Vivado hardware Tcl commands are unavailable" in program_tcl40
    assert "allow_non_jtag" not in program_tcl40
    assert "--help" in build_bat
    assert "--check" in build_bat
    assert "--diagnose" in build_bat
    assert "create_project_40ch.tcl" in build_bat
    assert "program_fpga_40ch.tcl" in build_bat
    assert "check_40ch_synth.tcl" in build_bat
    assert "diagnose_hw_target.tcl" in build_bat
    assert "ZLC_PS_40CH_XDC" in build_bat
    assert "ZLC_PS_VIVADO_BIN" in build_bat
    assert "ZLC_PS_CHECK_PROJECT_DIR" in build_bat
    assert "ZLC_PS_BUILD_ROOT" in build_bat
    assert "ZLC_REPO_ROOT=%REPO_ROOT%" in build_bat
    assert "zlc_verify_40ch_sources" in build_bat
    assert "ZLC 40ch source contract: channels=40 max_edges=1024 edge_addr_width=10 prog_count_width=11" in build_bat
    assert "selected XDC does not define ch[39]" in build_bat
    assert "Expected VIO probe_out3 width 10" in build_bat
    assert "Expected VIO probe_out6 width 11" in build_bat
    assert "zlc_clear_unsafe_artifact ZLC_PS_VIVADO_BIT" in build_bat
    assert "zlc_clear_unsafe_artifact ZLC_PS_VIVADO_LTX" in build_bat
    assert "Ignoring old pulse_streamer build-local %~1" in build_bat
    assert "ZLC build root:" in build_bat
    assert "fpga\\build\\p40" in build_bat
    assert "\\p40" in build_bat
    assert "\\c40" in build_bat
    assert "\\logs" in build_bat
    assert 'if not exist "!ZLC_PS_BUILD_ROOT!\\"' in build_bat
    assert "subst " not in build_bat.lower()
    assert "ZLC_PS_SHORT_DRIVE" not in build_bat
    assert "ZLC_PS_DISABLE_SUBST" not in build_bat
    assert "infer_channel_count" in server_bat
    assert "infer_channels" in server_bat
    assert "ZLC_PS_MAX_CHANNEL_COUNT=40" in server_bat
    assert "ZLC_PS_MAX_EDGES=1024" in server_bat
    assert "ZLC_PS_BUILD_ROOT" in server_bat
    assert "ZLC_REPO_ROOT=%REPO_ROOT%" in server_bat
    assert "zlc_verify_40ch_sources" in server_bat
    assert "ZLC 40ch source contract: channels=40 max_edges=1024 edge_addr_width=10 prog_count_width=11" in server_bat
    assert "selected XDC does not define ch[39]" in server_bat
    assert "zlc_clear_unsafe_artifact ZLC_PS_VIVADO_BIT" in server_bat
    assert "zlc_clear_unsafe_artifact ZLC_PS_VIVADO_LTX" in server_bat
    assert "Ignoring old pulse_streamer build-local %~1" in server_bat
    assert "ZLC build root:" in server_bat
    assert "fpga\\build\\p40" in server_bat
    assert "\\p40" in server_bat
    assert "\\state40" in server_bat
    assert 'if not exist "!ZLC_PS_BUILD_ROOT!\\"' in server_bat
    assert "for /r \"%CD%\\fpga\\pulse_streamer\\build\"" not in server_bat
    assert "ZLC_PS_PROFILE" not in server_bat
    assert "ZLC_PS_SERVER_BACKEND=vivado-session" in server_bat
    assert "ZLC_FPGA_SERVER_PYTHON" in server_bat
    assert ".zlc_python_path" in server_bat
    assert "zlc_normalize_python_cmd" in server_bat
    assert 'call "%ZLC_FPGA_SERVER_PYTHON%"' in server_bat
    assert 'call "!ZLC_STORED_PY!"' in server_bat
    assert "ZLC_PY_ARG" in server_bat
    assert 'if /I "%ZLC_PS_SERVER_BACKEND%"=="command" goto zlc_run_command_backend' in server_bat
    assert "--prepare-command" in server_bat
    assert "ch00 ch01 ch02 ch03" in server_bat
    assert "ch36 ch37 ch38 ch39" in server_bat
    assert "--trigger-channels ch03" in server_bat
    assert "--no-warm-start" not in server_bat
    assert "--check-config" in server_bat
    assert "ZLC server config check complete" in server_bat
    assert "ZLC !ZLC_ACTION! completed successfully" in server_bat
    assert "ZLC !ZLC_ACTION! failed with code" in server_bat
    assert "ZLC_ACTION=server config check" in server_bat
    assert "Server stopped normally" in server_bat
    assert "trap cooling probe qcm_trigger" not in server_bat
    assert "ZLC !ZLC_ACTION! completed successfully" in build_bat
    assert "ZLC !ZLC_ACTION! failed with code" in build_bat
    assert "ZLC_ACTION=hardware diagnose" in build_bat
    assert "ZLC_ACTION=40ch build" in build_bat
    assert "derived from the old address_switch pin map" in build_bat
    assert 'if "%ZLC_NO_PAUSE%"=="" pause' in server_bat
    for pattern in ("build/", ".Xil/", "*.jou", "*.str", "*.ltx", "*.runs/", "*.cache/", "*.hw/", "*.sim/"):
        assert pattern in gitignore
    assert "address_switch.srcs/constrs_1/new/addre.xdc" in xdc40
    assert "[get_ports {ch[0]}]" in xdc40
    assert "[get_ports {ch[39]}]" in xdc40
    assert "<PIN_CH" not in xdc40
    assert "ZLC project dir" in runbook
    assert "<ZLC project dir>\\p40.xpr" in runbook
    assert "C:\\ZLCPS\\p40\\zlc_pulse_streamer_40ch.xpr" not in runbook
    assert "docs/**/*_zh.txt" in gitignore
    assert "真实 build/program 默认使用 `fpga\\pulse_streamer\\zlc_pulse_streamer_40ch.xdc`" in fpga_notebook_text
    assert "需要先把 `fpga\\pulse_streamer\\zlc_pulse_streamer_40ch.xdc.template`" not in fpga_notebook_text
    assert "standalone hardware side" in fpga_readme
    assert "fpga\\build\\p40" in fpga_readme
    assert "debug path guard" in fpga_readme
    assert "ZLC_PS_VIVADO_LTX" in fpga_readme
    assert "must not load a stale repo-local" in fpga_readme
    assert "ZLC project dir" in fpga_readme
    assert "run_server.bat --check-config" in fpga_readme
    assert "differential upload" in fpga_streamer_readme
    assert "differential upload" in runbook
    assert "wrote 3/6 edge rows" in runbook
    assert "ZLC_PS_VIVADO_LTX" in runbook
    assert 'duration="x"' in runbook
    assert "pulse.x = ..." in runbook
    assert "差分 prepare" in hardware_quickstart_template
    assert "wrote 3/6 edge rows" in hardware_quickstart_template
    assert "shadow-critical rows" in hardware_quickstart_template
    assert 'duration="x"' in hardware_quickstart_template
    assert "x_ns=19980000" in hardware_quickstart_template
    assert 'exp.readout.detection_time(times, shots=20, live=False, pulse=pulse)' in hardware_quickstart_template
    assert "RUN_SINGLE_PULSE_TEST = False" in hardware_quickstart_template
    assert "pulse.on_pulse(wait=True, timeout=10.0, repeat_forever=False)" in hardware_quickstart_template
    assert "live_scan = exp.readout.detection_time(" in hardware_quickstart_template
    assert "pulse=pulse" in hardware_quickstart_template
    assert "live_scan.stop()" in hardware_quickstart_template
    assert "差分 prepare" in hardware_quickstart_pdf_tex
    assert "wrote 3/6 edge rows" in hardware_quickstart_pdf_tex
    assert "shadow-critical rows" in hardware_quickstart_pdf_tex
    assert 'duration="x"' in hardware_quickstart_pdf_tex
    assert "x_ns=19980000" in hardware_quickstart_pdf_tex
    assert 'exp.readout.detection_time(times, shots=20, live=False, pulse=pulse)' in hardware_quickstart_pdf_tex
    assert "RUN_SINGLE_PULSE_TEST = False" in hardware_quickstart_pdf_tex
    assert "pulse.on_pulse(wait=True, timeout=10.0, repeat_forever=False)" in hardware_quickstart_pdf_tex
    assert "live_scan = exp.readout.detection_time(" in hardware_quickstart_pdf_tex
    assert "pulse=pulse" in hardware_quickstart_pdf_tex
    assert "live_scan.stop()" in hardware_quickstart_pdf_tex
    assert 'duration="x", unit="str (ns)"' in hardware_notebook_text
    assert "test_widths_ns" in hardware_notebook_text
    assert "RUN_SINGLE_PULSE_TEST = False" in hardware_notebook_text
    assert "pulse.on_pulse(wait=True, timeout=10.0, repeat_forever=False)" in hardware_notebook_text
    assert "RUN_LIVE_READOUT_SCAN = False" in hardware_notebook_text
    assert "Optional live readout-time scan" in hardware_notebook_text
    assert "live_scan = exp.readout.detection_time(" in hardware_notebook_text
    assert "pulse=pulse" in hardware_notebook_text
    assert "live_scan.stop()" in hardware_notebook_text
    assert "shadow-critical rows" in fpga_readme
    assert "shadow-critical rows" in fpga_streamer_readme
    assert "full hardware channel list" in fpga_readme
    assert "standalone user-interface layer" in frontend_readme
    assert "Pulse GUI Contract" in frontend_readme
    assert "DEFAULT_PULSE_GUI_MAX_CHANNELS = 40" in frontend_readme
    assert "127.0.0.1:18861" in frontend_readme
    assert "--remote-host" in frontend_readme
    assert "--no-sequencer" in frontend_readme
    assert "默认创建 40 路本地" not in frontend_manual_template
    assert "默认创建 40 路本地" not in frontend_manual_pdf_tex
    assert "127.0.0.1:18861" in frontend_manual_template
    assert "127.0.0.1:18861" in frontend_manual_pdf_tex
    assert "默认会尝试连接本机" in frontend_manual_template
    assert "离线打开 editor" in frontend_manual_template
    assert "显式传入 \\pyapi{--remote-host}" in frontend_manual_template
    assert ".\\pulse_gui.bat --no-sequencer" in frontend_manual_template
    assert ".\\pulse_gui.bat --no-sequencer" in frontend_manual_pdf_tex
    assert "默认创建 40 路本地" not in hardware_quickstart_template
    assert "默认创建 40 路本地" not in hardware_quickstart_pdf_tex
    assert "127.0.0.1:18861" in hardware_quickstart_template
    assert "127.0.0.1:18861" in hardware_quickstart_pdf_tex
    assert "默认会尝试连接本机" in hardware_quickstart_template
    assert "GUI 会离线打开" in hardware_quickstart_template
    assert "保留窗口并显示错误" in hardware_quickstart_template
    assert ".\\pulse_gui.bat --no-sequencer" in hardware_quickstart_template
    assert ".\\pulse_gui.bat --no-sequencer" in hardware_quickstart_pdf_tex
    assert "render_tex_pdf(tex, output_pdf)" in frontend_readme
    assert "does not own hardware actions" in frontend_readme
    assert "camera_imaging_40ch.json" in pulses_readme
    assert "duration=\"x\"" in pulses_readme
    assert "pulse.x" in pulses_readme
    assert "19980000" in pulses_readme
    assert "keeps `ch03` low" in pulses_readme
    assert "slow table-boundary qCMOS spike" in pulses_readme
    assert "finite trigger sequence" in pulses_readme
    assert "Run the smallest check" in tests_readme
    assert "Full `pytest -q` is reserved for broad handoff" in tests_readme
    assert "fpga\\build_and_program.bat --check" in tests_readme
    assert "render_tex_pdf" in tests_readme
    for text in (hardware_quickstart_template, hardware_quickstart_pdf_tex):
        assert "prog_we} 翻转一次" in text
        assert "level write-enable" not in text
        assert "prog_we=1" not in text
        assert "从按下 On Pulse 到 FPGA pin 变化" in text
        assert "GUI 不直接写硬件" in text
        assert "RPyC 只传一次描述" in text
        assert "第一条需要写的 edge row 放在同一批 VIO commit" in text
        assert "外部 pin 只连到 \\pyapi{state_mask}" in text
        assert "为什么第一次慢、后面快，以及 100 ms 的边界" in text
        assert "正确升级方向不是改 GUI" in text
    assert "Pulse GUI On Pulse:" in fpga_streamer_readme
    assert "read widgets -> PulseTableState" in fpga_streamer_readme
    assert "RemoteSequencer.prepare" in fpga_streamer_readme
    assert "Runtime `prepare` writes VIO rows" in fpga_streamer_readme
    assert "Runtime `fire` is one small VIO commit" in fpga_streamer_readme
    assert "replace only the upload transport" in fpga_streamer_readme


def test_fpga_pulse_streamer_40ch_vio_widths_match_python_generator():
    root = Path(__file__).resolve().parents[1]
    fpga = root / "fpga" / "pulse_streamer"
    top40 = (fpga / "zlc_pulse_streamer_top_40ch.v").read_text(encoding="utf-8")
    tcl40 = (fpga / "create_project_40ch.tcl").read_text(encoding="utf-8")

    generated_top = na.generate_pulse_streamer_top_example(
        channels=[f"ch{i:02d}" for i in range(40)],
        top_module_name="zlc_pulse_streamer_top_40ch",
        max_edges=1024,
    )

    width_contract = {
        "prog_addr": "9:0",
        "prog_tick": "31:0",
        "prog_mask": "39:0",
        "prog_count": "10:0",
        "loop_start_addr": "9:0",
        "loop_end_tick": "31:0",
        "loop_count": "31:0",
    }
    for signal, width in width_contract.items():
        assert f"wire [{width}] zlc_{signal};" in top40
        assert f"wire [{width}] zlc_{signal};" in generated_top

    for expected in {
        "CONFIG.C_PROBE_OUT3_WIDTH {10}",
        "CONFIG.C_PROBE_OUT4_WIDTH {32}",
        "CONFIG.C_PROBE_OUT5_WIDTH {40}",
        "CONFIG.C_PROBE_OUT6_WIDTH {11}",
        "CONFIG.C_PROBE_OUT8_WIDTH {10}",
        "CONFIG.C_PROBE_OUT9_WIDTH {32}",
        "CONFIG.C_PROBE_OUT10_WIDTH {32}",
    }:
        assert expected in tcl40

    assert ".EDGE_ADDR_WIDTH(10)" in top40
    assert ".EDGE_ADDR_WIDTH(10)" in generated_top
    assert "CONFIG.C_PROBE_OUT3_WIDTH {7}" not in tcl40
    assert "CONFIG.C_PROBE_OUT6_WIDTH {8}" not in tcl40
    assert ".EDGE_ADDR_WIDTH(7)" not in top40
    assert ".EDGE_ADDR_WIDTH(7)" not in generated_top


def test_fpga_pulse_streamer_capacity_doc_matches_checked_in_ram_strategy():
    root = Path(__file__).resolve().parents[1]
    fpga = root / "fpga" / "pulse_streamer"
    core = (fpga / "zlc_pulse_streamer.v").read_text(encoding="utf-8")
    top40 = (fpga / "zlc_pulse_streamer_top_40ch.v").read_text(encoding="utf-8")
    capacity = (root / "docs" / "FPGA_PULSE_STREAMER_CAPACITY.md").read_text(encoding="utf-8")
    streamer_readme = (fpga / "README.md").read_text(encoding="utf-8")

    assert '(* ram_style = "distributed" *)' in core
    assert ".CHANNEL_COUNT(40)" in top40
    assert ".EDGE_ADDR_WIDTH(10)" in top40
    assert "The HDL currently marks `tick_mem` and `mask_mem` as distributed RAM" in capacity
    assert "1024-row 40-channel operational profile" in capacity
    assert "BRAM-friendly synchronous-read" in capacity
    assert "pipeline and a faster upload transport" in capacity
    assert "AXI, JTAG-to-AXI, UART/SPI" in capacity
    assert "current `edge_index` row" in streamer_readme
    assert "1024 rows" in streamer_readme
    assert "probe_out3  zlc_prog_addr       width 10" in streamer_readme
    assert "probe_out6  zlc_prog_count      width 11" in streamer_readme


def test_user_facing_markdown_local_links_exist():
    root = Path(__file__).resolve().parents[1]
    markdown_files = [
        root / "README.md",
        root / "AGENTS.md",
        root / "docs" / "PROJECT_OVERVIEW.md",
        root / "docs" / "DOCUMENTATION_GUIDE.md",
        root / "docs" / "FPGA_PULSE_STREAMER_CAPACITY.md",
        root / "docs" / "FRONTEND_FLUENT_STYLE_GUIDE.md",
        root / "docs" / "neutral_atom_hardware_manual" / "REAL_HARDWARE_RUNBOOK.md",
        root / "fpga" / "README.md",
        root / "fpga" / "pulse_streamer" / "README.md",
        root / "Zou_lab_control" / "frontend" / "README.md",
        root / "pulses" / "README.md",
        root / "tests" / "README.md",
    ]
    missing: list[str] = []
    for markdown_file in markdown_files:
        assert markdown_file.exists(), markdown_file
        text = markdown_file.read_text(encoding="utf-8", errors="replace")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            target = target.strip()
            if not target or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
                continue
            path_part = target.split("#", 1)[0].strip()
            if not path_part:
                continue
            resolved = (markdown_file.parent / unquote(path_part)).resolve()
            if not resolved.exists():
                missing.append(f"{markdown_file.relative_to(root)} -> {target}")
    assert missing == []


def test_repo_tree_has_no_generated_fpga_or_latex_work_products():
    root = Path(__file__).resolve().parents[1]
    ignored_parts = {"references", ".git"}
    bad_directories = {"build", ".Xil", ".runs", ".cache", ".hw", ".sim"}
    bad_suffixes = {
        ".aux",
        ".toc",
        ".out",
        ".fls",
        ".fdb_latexmk",
        ".synctex.gz",
        ".jou",
        ".str",
        ".log",
        ".build.log",
        ".ltx",
        ".rpt",
        ".dcp",
    }
    found: list[str] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in ignored_parts:
            continue
        if path.is_dir() and path.name in bad_directories:
            found.append(str(relative))
        if path.is_file() and any(path.name.endswith(suffix) for suffix in bad_suffixes):
            found.append(str(relative))
    assert found == []


def test_repo_bat_entrypoints_are_minimal_and_grouped_by_submodule():
    root = Path(__file__).resolve().parents[1]
    ignored_roots = {".git", "references", "reference"}
    bat_files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.bat") if not (set(path.relative_to(root).parts) & ignored_roots))

    assert bat_files == [
        "fpga/build_and_program.bat",
        "fpga/run_server.bat",
        "install_requirements.bat",
        "pulse_gui.bat",
        "start_tutorials_jupyter_lab.bat",
    ]
    assert not any("4ch" in path.lower() or "smoke" in path.lower() or "simulate" in path.lower() for path in bat_files)


def test_fpga_pulse_streamer_xdc_infers_full_channel_count(tmp_path):
    xdc = tmp_path / "board.xdc"
    xdc.write_text(
        "\n".join(
            f"set_property -dict {{PACKAGE_PIN P{index} IOSTANDARD LVCMOS33}} [get_ports {{ch[{index}]}}]"
            for index in range(6)
        ),
        encoding="utf-8",
    )

    assert na.infer_xdc_channel_count(xdc, default=40, max_count=40) == 6
    assert na.infer_xdc_channels(xdc, default=40, max_count=40) == ["ch00", "ch01", "ch02", "ch03", "ch04", "ch05"]
    assert na.hardware_channel_names(4) == ["ch00", "ch01", "ch02", "ch03"]
    labeled = tmp_path / "labeled.xdc"
    labeled.write_text(
        "\n".join(
            [
                "set_property PACKAGE_PIN A1 [get_ports {ch[0]}] ;# ch00 <- trap",
                "set_property PACKAGE_PIN A2 [get_ports {ch[1]}] ;# ch01 <- cooling_pgc",
                "set_property PACKAGE_PIN A3 [get_ports {ch[2]}] ;# ch02 <- trig / qcm_trigger",
            ]
        ),
        encoding="utf-8",
    )
    assert na.infer_xdc_channel_labels(labeled, default=40, max_count=40) == {
        "ch00": "trap",
        "ch01": "cooling_pgc",
        "ch02": "qcm_trigger",
    }

    sparse = tmp_path / "sparse.xdc"
    sparse.write_text(
        "set_property PACKAGE_PIN A1 [get_ports {ch[0]}]\nset_property PACKAGE_PIN A3 [get_ports {ch[2]}]\n",
        encoding="utf-8",
    )
    try:
        na.infer_xdc_channel_count(sparse)
    except ValueError as exc:
        assert "not contiguous" in str(exc)
        assert "1" in str(exc)
    else:
        raise AssertionError("sparse XDC channel map should be rejected")


def test_fpga_pulse_streamer_prepare_tcl_covers_full_edge_table_boundary(tmp_path):
    program = na.RuntimeSequenceProgram(
        sequence_id="full",
        sequence_name="full_table",
        clock_hz=100e6,
        channels=["trap", "cooling", "probe", "qcm_trigger"],
        ticks=list(range(1024)),
        masks=[1] * 1023 + [0],
        duration=1023 / 100e6,
        trigger_count=0,
    )

    na.validate_pulse_streamer_program(program, max_edges=1024, channel_count=4)
    tcl_path = na.write_vivado_pulse_streamer_tcl(
        tmp_path / "prepare_full.tcl",
        "prepare",
        program=program,
        project="",
        bitstream="",
        probes="",
        max_edges=1024,
        channel_count=4,
    )
    tcl = tcl_path.read_text(encoding="utf-8")

    assert "zlc_stage_probe $vio $zlc_prog_count_probe 1024" in tcl
    assert "load_features labtools" in tcl
    assert "open_hw" in tcl
    assert "get_hw_targets" in tcl
    assert "No Vivado hardware target found" in tcl
    assert "Vivado hardware Tcl commands are unavailable" in tcl
    assert "allow_non_jtag" not in tcl
    assert "VIO filter '$vio_filter' failed" in tcl
    assert "using the only available VIO core" in tcl
    assert "zlc_stage_probe $vio $zlc_prog_addr_probe 1023" in tcl
    assert "zlc_stage_probe $vio $zlc_prog_tick_probe 1023" in tcl
    assert "zlc_stage_probe $vio $zlc_prog_mask_probe 0" in tcl
    assert tcl.count("set zlc_prog_we_toggle_value [expr {$zlc_prog_we_toggle_value ? 0 : 1}]") == 1024
    assert tcl.count("zlc_stage_probe $vio $zlc_prog_we_probe $zlc_prog_we_toggle_value") == 1024
    assert "set zlc_prog_we_toggle_value [zlc_output_probe_bool $vio $zlc_prog_we_probe]" in tcl
    assert "zlc_stage_probe $vio $zlc_prog_we_probe 0" not in tcl
    assert "zlc_set_probe $vio $zlc_prog_we_probe 0" not in tcl


def test_fpga_pulse_streamer_edge_table_python_model_matches_contract():
    seq = na.PulseSequence(name="contract").pulse("trap", 0.0, 5e-8).pulse("probe", 2e-8, 8e-8).pulse("qcm_trigger", 2e-8, 4e-8)
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "qcm_trigger"], clock_hz=100e6)
    na.validate_pulse_streamer_program(program, max_edges=16, channel_count=4)

    history = _simulate_pulse_streamer(program.ticks, program.masks)
    by_tick = {tick: state for tick, state, running, done in history}

    assert program.ticks == [0, 2, 5, 6, 10]
    assert program.masks[-1] == 0
    assert by_tick[0] & 0b0001
    assert by_tick[2] & 0b1101
    assert by_tick[5] == 0b1100
    assert by_tick[6] == 0b0100
    assert by_tick[10] == 0
    assert history[-1][2] is False
    assert history[-1][3] is True


def test_fpga_pulse_streamer_compiles_channel_delay_and_repeated_frames():
    seq = na.PulseSequence(name="delay_repeat").pulse("trap", 0.0, 2e-8).pulse("qcm_trigger", 0.0, 1e-8).delay("qcm_trigger", 1e-8)
    repeated = seq.repeated(2, period=5e-8)
    program = na.compile_runtime_program(repeated, channels=["trap", "cooling", "probe", "qcm_trigger"], clock_hz=100e6)

    assert program.ticks == [0, 1, 2, 5]
    assert program.masks == [0b0001, 0b1001, 0b0000, 0b0000]
    assert program.masks[-1] == 0
    assert program.trigger_count == 2
    assert program.repeat_forever is False
    assert program.loop_start_index == 0
    assert program.loop_end_tick == 5
    assert program.loop_count == 2


def test_pulse_table_state_compiles_repeat_visibility_and_delays(tmp_path):
    unnamed = na.PulseTableState(channels=["ch00"])
    assert re.fullmatch(r"pulse_\d{8}_\d{6}", unnamed.name)

    state = na.PulseTableState(
        channels=["trap", "cooling", "probe", "qcm_trigger", "aod0", "aod1"],
        periods=[
            na.PulsePeriod(100, (1, 0, 0, 0, 0, 0), unit="ns", name="load"),
            na.PulsePeriod("2*x", (1, 0, 1, 1, 0, 0), unit="str (ns)", name="image"),
            na.PulsePeriod(100, (0, 0, 0, 0, 0, 0), unit="ns", name="idle"),
        ],
        delays={"qcm_trigger": "x/2"},
        delay_units={"qcm_trigger": "str (ns)"},
        x_ns=100,
        time_step_ns=1,
        repeat_start=1,
        repeat_end=2,
        repeat_count=3,
    )

    sequence = state.to_sequence()
    program = state.compile(clock_hz=100e6, trigger_channels=["qcm_trigger"])
    saved = state.save(tmp_path / "pulse.json")
    loaded = na.PulseTableState.load(saved)

    assert state.time_step_ns == 1
    assert state.total_duration_steps() == 1000
    assert state.total_duration_steps(x_ns=200, time_step_ns=10) == 160
    assert state.total_duration_ns() == 100 + 3 * (200 + 100)
    assert state.total_duration_ns(x_ns=200) == 100 + 3 * (400 + 100)
    assert state.periods[1].duration_steps(x_ns=200, time_step_ns=state.time_step_ns) == 400
    assert state.delay_ns("qcm_trigger", x_ns=200) == 100
    assert state.delay_steps("qcm_trigger", x_ns=200, time_step_ns=10) == 10
    assert state.with_x(200).x_ns == 200
    assert state.x_ns == 100
    assert na.count_trigger_pulses(sequence, trigger_channels=["qcm_trigger"]) == 3
    assert program.trigger_count == 3
    assert program.repeat_forever is True
    assert program.ticks == [0, 10, 15, 30, 35, 40]
    assert program.masks == [0b000001, 0b000101, 0b001101, 0b001000, 0, 0]
    assert program.loop_start_index == 1
    assert program.loop_end_tick == 40
    assert program.loop_count == 3
    program_x = state.compile(clock_hz=100e6, trigger_channels=["qcm_trigger"], x_ns=200)
    assert program_x.trigger_count == 3
    assert program_x.repeat_forever is True
    assert program_x.ticks[-1] == 60
    assert program_x.duration == 1.6e-6
    assert loaded.to_dict() == state.to_dict()

    state.hide_channel("aod0")
    assert "aod0" not in state.visible_channels
    assert "aod1" not in state.active_channels()
    try:
        state.hide_channel("trap")
    except ValueError as exc:
        assert "active" in str(exc)
    else:
        raise AssertionError("active channel should require explicit clearing before hiding")


def test_pulse_table_state_compiles_hardware_repeat_without_expanding_edges():
    state = na.PulseTableState(
        channels=["ch00", "ch01", "ch02", "ch03"],
        periods=[
            na.PulsePeriod(10, (1, 0, 0, 0), unit="ns", name="load"),
            na.PulsePeriod(20, (0, 1, 0, 1), unit="ns", name="trigger"),
            na.PulsePeriod(10, (0, 0, 0, 0), unit="ns", name="idle"),
        ],
        time_step_ns=10,
        repeat_start=1,
        repeat_end=1,
        repeat_count=500,
        visible_channels=["ch00", "ch03"],
        channel_labels={"ch00": "trap", "ch03": "qcm_trigger"},
    )

    hardware_channels = [f"ch{i:02d}" for i in range(40)]
    program = na.compile_pulse_table_runtime_program(
        state,
        channels=hardware_channels,
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
    )
    aligned = state.aligned_to_channels(hardware_channels)

    assert len(program.channels) == 40
    assert program.channels == hardware_channels
    assert program.ticks == [0, 1, 3, 4]
    assert program.masks == [0b0001, 0b1010, 0, 0]
    assert program.duration == 1002 / 100_000_000
    assert program.repeat_forever is True
    assert program.loop_start_index == 1
    assert program.loop_end_tick == 3
    assert program.loop_count == 500
    assert program.trigger_count == 1
    assert len(program.ticks) == len(state.periods) + 1
    assert len(program.ticks) < state.repeat_count
    assert aligned.channels == hardware_channels
    assert aligned.periods[0].states[:4] == (1, 0, 0, 0)
    assert all(value == 0 for value in aligned.periods[0].states[4:])
    na.validate_pulse_streamer_program(program, max_edges=1024, tick_width=32, channel_count=40)


def test_sequencer_service_pads_gui_subset_state_to_full_40ch_hardware():
    hardware_channels = [f"ch{i:02d}" for i in range(40)]
    prepared_programs: list[na.RuntimeSequenceProgram] = []
    service = na.SequencerService(
        channels=hardware_channels,
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
        prepare_callback=prepared_programs.append,
    )
    state = na.PulseTableState(
        channels=["ch00", "ch01", "ch02", "ch03"],
        periods=[
            na.PulsePeriod(20, (1, 0, 0, 0), unit="ns", name="load"),
            na.PulsePeriod(20, (1, 0, 0, 1), unit="ns", name="camera"),
            na.PulsePeriod(20, (0, 0, 0, 0), unit="ns", name="idle"),
        ],
        time_step_ns=10,
        repeat_forever=True,
        visible_channels=["ch00", "ch03"],
        channel_labels={"ch00": "trap", "ch03": "qcm_trigger"},
    )

    payload = service.prepare(state.to_dict())
    program = na.RuntimeSequenceProgram.from_dict(payload)

    assert len(program.channels) == 40
    assert program.channels == hardware_channels
    assert program.masks == [1 << 0, (1 << 0) | (1 << 3), 0, 0]
    assert all(mask >> 4 == 0 for mask in program.masks)
    assert program.trigger_count == 1
    assert prepared_programs == [program]
    na.validate_pulse_streamer_program(program, max_edges=1024, tick_width=32, channel_count=40)


def test_fpga_loop_repeat_keeps_post_loop_idle_before_repeat_forever():
    state = na.PulseTableState(
        channels=["ch00", "ch01", "ch02", "ch03"],
        periods=[
            na.PulsePeriod(10, (1, 0, 0, 0), unit="ns", name="load"),
            na.PulsePeriod(20, (0, 0, 0, 1), unit="ns", name="trigger"),
            na.PulsePeriod(10, (0, 0, 0, 0), unit="ns", name="post_idle"),
        ],
        time_step_ns=10,
        repeat_start=1,
        repeat_end=1,
        repeat_count=2,
        repeat_forever=True,
    )

    program = na.compile_pulse_table_runtime_program(
        state,
        channels=[f"ch{i:02d}" for i in range(40)],
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
    )
    history = _simulate_pulse_streamer_program_steps(program, steps=8)
    core = (Path(__file__).resolve().parents[1] / "fpga" / "pulse_streamer" / "zlc_pulse_streamer.v").read_text(
        encoding="utf-8"
    )

    assert program.ticks == [0, 1, 3, 4]
    assert program.masks == [0b0001, 0b1000, 0, 0]
    assert program.loop_start_index == 1
    assert program.loop_end_tick == 3
    assert program.loop_count == 2
    assert program.repeat_forever is True
    assert history[:7] == [0b0001, 0b1000, 0b1000, 0b1000, 0b1000, 0, 0b0001]
    assert "final_tick <= (prog_count == 0) ? {TICK_WIDTH{1'b0}} : final_tick_shadow;" in core
    assert "loop_end_tick : final_tick_shadow" not in core


def test_pulse_table_reports_repeat_forever_table_boundary_high_channels():
    state = na.PulseTableState(
        channels=["ch00", "ch01", "ch02", "ch03"],
        periods=[
            na.PulsePeriod(10, (1, 1, 0, 0), unit="ns", name="load"),
            na.PulsePeriod(10, (0, 0, 0, 1), unit="ns", name="trigger"),
            na.PulsePeriod(10, (0, 0, 0, 0), unit="ns", name="post_idle"),
        ],
        time_step_ns=10,
        repeat_start=1,
        repeat_end=1,
        repeat_count=3,
        repeat_forever=True,
        channel_labels={"ch00": "trap", "ch01": "cooling", "ch03": "qcm_trigger"},
    )

    program = na.compile_pulse_table_runtime_program(
        state,
        channels=[f"ch{i:02d}" for i in range(40)],
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
    )
    history = _simulate_pulse_streamer_program_steps(program, steps=7)
    boundary_tick = round(program.duration * program.clock_hz)

    assert state.repeat_forever_boundary_active_channels() == ["ch00", "ch01"]
    assert [state.label_for(channel) for channel in state.repeat_forever_boundary_active_channels()] == ["trap", "cooling"]
    assert program.ticks == [0, 1, 2, 3]
    assert program.loop_start_index == 1
    assert program.loop_end_tick == 2
    assert program.loop_count == 3
    assert boundary_tick == 5
    assert all((mask & 0b0011) == 0 for mask in history[1:boundary_tick])
    assert history[boundary_tick] & 0b0011 == 0b0011


def test_pulse_table_no_boundary_warning_when_repeat_bracket_covers_whole_table():
    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod(10, (1, 0), unit="ns"),
            na.PulsePeriod(10, (0, 1), unit="ns"),
        ],
        time_step_ns=10,
        repeat_start=0,
        repeat_end=1,
        repeat_count=3,
        repeat_forever=True,
    )

    assert state.repeat_forever_boundary_active_channels() == []


def test_partial_hardware_channels_default_missing_outputs_off():
    hardware_channels = [f"ch{i:02d}" for i in range(40)]
    seq = na.PulseSequence(name="partial_hardware").pulse("ch00", 0.0, 10e-9).pulse("ch03", 10e-9, 10e-9).forever(period=30e-9)
    sequence_program = na.compile_runtime_program(seq, channels=hardware_channels, clock_hz=100_000_000, trigger_channels=["ch03"])

    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod(10, (1, 0), unit="ns"),
            na.PulsePeriod(10, (0, 1), unit="ns"),
            na.PulsePeriod(10, (0, 0), unit="ns"),
        ],
        time_step_ns=10,
        visible_channels=["ch00", "ch03"],
    )
    table_program = na.compile_pulse_table_runtime_program(
        state,
        channels=hardware_channels,
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
    )

    assert sequence_program.channels == hardware_channels
    assert sequence_program.masks == [1 << 0, 1 << 3, 0, 0]
    assert sequence_program.repeat_forever is True
    assert len(table_program.channels) == 40
    assert table_program.channels == hardware_channels
    assert table_program.masks == [1 << 0, 1 << 3, 0, 0]
    assert table_program.repeat_forever is True
    assert table_program.trigger_count == 1
    assert all(mask & ~((1 << 0) | (1 << 3)) == 0 for mask in table_program.masks)
    assert all(mask >> 4 == 0 for mask in table_program.masks)


def test_40ch_gui_visible_subset_compiles_as_full_width_fpga_program():
    hardware_channels = [f"ch{i:02d}" for i in range(40)]
    state = na.PulseTableState(
        channels=hardware_channels,
        periods=[
            na.PulsePeriod(100, (1, 1, 0, 0, *([0] * 36)), unit="ns", name="load"),
            na.PulsePeriod(20, (1, 0, 1, 1, *([0] * 36)), unit="ns", name="camera"),
            na.PulsePeriod(100, (0,) * 40, unit="ns", name="off"),
        ],
        time_step_ns=10,
        visible_channels=["ch00", "ch01", "ch02", "ch03"],
        channel_labels={"ch00": "trap", "ch01": "cooling", "ch02": "probe", "ch03": "qcm_trigger"},
    )

    program = na.compile_pulse_table_runtime_program(
        state,
        channels=hardware_channels,
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
    )

    assert state.visible_channels == hardware_channels[:4]
    assert program.channels == hardware_channels
    assert len(program.channels) == 40
    assert program.masks == [0b0011, 0b1101, 0, 0]
    assert all(mask >> 4 == 0 for mask in program.masks)
    assert program.trigger_count == 1
    assert program.repeat_forever is True
    na.validate_pulse_streamer_program(program, max_edges=1024, tick_width=32, channel_count=40)


def test_pulse_table_unknown_channel_is_not_silently_ignored():
    state = na.PulseTableState(
        channels=["ch00", "not_on_fpga"],
        periods=[na.PulsePeriod(10, (1, 1), unit="ns")],
        time_step_ns=10,
    )
    try:
        na.compile_pulse_table_runtime_program(state, channels=[f"ch{i:02d}" for i in range(40)], clock_hz=100_000_000)
    except ValueError as exc:
        assert "not in hardware channels" in str(exc)
        assert "not_on_fpga" in str(exc)
    else:
        raise AssertionError("unknown pulse-table channels should be rejected")


def test_checked_in_camera_imaging_pulse_compiles_for_40ch_fpga():
    path = Path(__file__).resolve().parents[1] / "pulses" / "camera_imaging_40ch.json"
    state = na.PulseTableState.load(path)
    program = state.compile(clock_hz=100_000_000, trigger_channels=["ch03"])

    assert state.channels == [f"ch{i:02d}" for i in range(40)]
    assert state.visible_channels == ["ch00", "ch01", "ch02", "ch03"]
    assert len(state.channel_labels) == 40
    assert state.channel_labels["ch00"] == "trap"
    assert state.channel_labels["ch03"] == "qcm_trigger"
    assert state.channel_labels["ch04"] == "cooling_pgc"
    assert state.channel_labels["ch18"] == "da_clk0"
    assert state.channel_labels["ch39"] == "da_bias_x[7]"
    assert state.delay_steps("ch00", time_step_ns=10) == 0
    assert state.delay_steps("ch03", time_step_ns=10) == 0
    assert state.repeat_start is None
    assert state.repeat_end is None
    assert state.repeat_count == 1
    assert state.repeat_forever is True
    assert state.repeat_forever_boundary_active_channels() == []
    assert state.x_ns == 19_980_000
    exposure_period = next(period for period in state.periods if period.name == "camera_exposure")
    assert exposure_period.duration == "x"
    assert exposure_period.unit == "str (ns)"
    assert state.periods[0].states[state.channel_index("ch03")] == 0
    assert program.channels == state.channels
    assert program.ticks == [0, 200_000, 210_000, 212_000, 2_210_000, 2_212_000]
    assert program.masks == [0b0011, 0b0001, 0b1101, 0b0101, 0b0001, 0]
    assert program.trigger_count == 1
    assert program.repeat_forever is True
    assert program.loop_start_index == 0
    assert program.loop_end_tick == 2_212_000
    assert program.loop_count == 1
    na.validate_pulse_streamer_program(program, max_edges=1024, tick_width=32, channel_count=40)

    shorter = state.compile(clock_hz=100_000_000, trigger_channels=["ch03"], x_ns=2_000_000)
    assert shorter.ticks == [0, 200_000, 210_000, 212_000, 412_000, 414_000]
    assert shorter.masks == program.masks
    assert shorter.trigger_count == 1
    finite = na.finite_frame_sequence(state.with_x(2_000_000), 3, trigger_channels=["ch03"])
    finite_program = na.compile_runtime_program(
        finite,
        channels=state.channels,
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
    )
    assert finite_program.repeat_forever is False
    assert finite_program.trigger_count == 3


def test_pulse_table_repeat_forever_can_be_disabled_for_single_shot():
    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod(100, (1, 0), unit="ns"),
            na.PulsePeriod(20, (0, 1), unit="ns"),
            na.PulsePeriod(100, (0, 0), unit="ns"),
        ],
        time_step_ns=10,
        repeat_forever=False,
    )

    program = na.compile_pulse_table_runtime_program(
        state,
        channels=["ch00", "ch01", "ch02", "ch03"],
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
        repeat_forever=state.repeat_forever,
    )
    service_program = na.compile_runtime_program_for_payload(
        state,
        channels=["ch00", "ch01", "ch02", "ch03"],
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
    )
    restored = na.PulseTableState.from_dict(state.to_dict())

    assert restored.repeat_forever is False
    assert program.repeat_forever is False
    assert service_program.repeat_forever is False
    assert program.masks == [1 << 0, 1 << 3, 0, 0]


def test_bind_pulse_controller_updates_x_and_fires_runtime_sequencer():
    sequencer = na.RuntimeSequencer(
        channels=["ch00", "ch03"],
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
        sleep_scale=0.0,
    )
    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod("x", (1, 0), unit="str (ns)"),
            na.PulsePeriod(20, (0, 1), unit="ns"),
            na.PulsePeriod(20, (0, 0), unit="ns"),
        ],
        time_step_ns=10,
        x_ns=100,
        repeat_forever=False,
    )

    pulse = na.bind_pulse(sequencer, state)
    assert pulse.snapshot()["last_program"] is None
    assert pulse.snapshot()["sequencer_channels"] == ["ch00", "ch03"]
    pulse.x = 200
    program = pulse.on_pulse(wait=True, timeout=1.0)

    assert program.ticks == [0, 20, 22, 24]
    assert program.masks == [1 << 0, 1 << 1, 0, 0]
    assert program.repeat_forever is False
    assert sequencer.snapshot()["state"] == "done"
    snapshot = pulse.snapshot()
    assert snapshot["x_ns"] == 200
    assert snapshot["last_program"]["edge_count"] == len(program.ticks)
    assert snapshot["last_program"]["trigger_count"] == program.trigger_count


def test_bind_pulse_controller_can_override_repeat_forever_for_scope_debug():
    sequencer = na.RuntimeSequencer(
        channels=["ch00", "ch03"],
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
        sleep_scale=0.0,
    )
    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod(100, (1, 0), unit="ns"),
            na.PulsePeriod(20, (1, 1), unit="ns"),
            na.PulsePeriod(20, (0, 0), unit="ns"),
        ],
        time_step_ns=10,
        repeat_forever=True,
    )

    pulse = na.bind_pulse(sequencer, state)
    program = pulse.on_pulse(wait=True, timeout=1.0, repeat_forever=False)

    assert state.repeat_forever is True
    assert program.repeat_forever is False
    assert sequencer.snapshot()["state"] == "done"


def test_bind_pulse_controller_rejects_waiting_indefinitely_for_repeat_forever():
    sequencer = na.RuntimeSequencer(
        channels=["ch00", "ch03"],
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
        sleep_scale=0.0,
    )
    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod(100, (1, 0), unit="ns"),
            na.PulsePeriod(20, (1, 1), unit="ns"),
            na.PulsePeriod(20, (0, 0), unit="ns"),
        ],
        time_step_ns=10,
        repeat_forever=True,
    )

    pulse = na.bind_pulse(sequencer, state)
    try:
        pulse.on_pulse(wait=True)
    except RuntimeError as exc:
        assert "repeat_forever" in str(exc)
        assert "repeat_forever=False" in str(exc)
    else:
        raise AssertionError("waiting indefinitely for repeat_forever pulse should be rejected")

    assert pulse.last_program is None
    assert sequencer.snapshot()["state"] == "idle"


def test_runtime_sequencer_repeat_forever_wait_done_times_out():
    sequencer = na.RuntimeSequencer(
        channels=["ch00", "ch03"],
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
        sleep_scale=0.0,
    )
    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod(100, (1, 0), unit="ns"),
            na.PulsePeriod(20, (1, 1), unit="ns"),
            na.PulsePeriod(20, (0, 0), unit="ns"),
        ],
        time_step_ns=10,
        repeat_forever=True,
    )

    sequencer.prepare(state)
    sequencer.fire()

    try:
        sequencer.wait_done()
    except RuntimeError as exc:
        assert "repeat_forever" in str(exc)
    else:
        raise AssertionError("wait_done without timeout should reject repeat_forever program")

    assert sequencer.wait_done(timeout=0.01) is False
    assert sequencer.snapshot()["state"] == "timeout"


def test_bind_pulse_controller_repeat_forever_wait_timeout_raises():
    sequencer = na.RuntimeSequencer(
        channels=["ch00", "ch03"],
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
        sleep_scale=0.0,
    )
    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod(100, (1, 0), unit="ns"),
            na.PulsePeriod(20, (1, 1), unit="ns"),
            na.PulsePeriod(20, (0, 0), unit="ns"),
        ],
        time_step_ns=10,
        repeat_forever=True,
    )

    pulse = na.bind_pulse(sequencer, state)
    try:
        pulse.on_pulse(wait=True, timeout=0.01)
    except TimeoutError as exc:
        assert "did not report done" in str(exc)
    else:
        raise AssertionError("repeat_forever wait with timeout should raise TimeoutError")

    assert pulse.last_program is not None
    assert pulse.last_program.repeat_forever is True
    assert sequencer.snapshot()["state"] == "timeout"


def test_bind_pulse_controller_can_override_sequence_repeat_forever():
    sequencer = na.RuntimeSequencer(
        channels=["ch00", "ch03"],
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
        sleep_scale=0.0,
    )
    sequence = (
        na.PulseSequence(name="sequence_scope_debug")
        .pulse("ch00", 0.0, 100e-9)
        .pulse("ch03", 100e-9, 20e-9)
        .forever(period=200e-9)
    )

    pulse = na.bind_pulse(sequencer, sequence)
    program = pulse.on_pulse(wait=True, timeout=1.0, repeat_forever=False)

    assert sequence.repeat_forever is True
    assert program.repeat_forever is False
    assert program.ticks == [0, 10, 12, 20]
    assert program.masks == [1 << 0, 1 << 1, 0, 0]
    assert sequencer.snapshot()["state"] == "done"


def test_detection_time_scan_uses_bound_40ch_pulse_controller():
    exp = na.connect("virtual")
    exp.readout.sitemap(frames=3, display=False)
    hardware_channels = [f"ch{i:02d}" for i in range(40)]

    class RecordingRuntimeSequencer(na.RuntimeSequencer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepared_programs = []

        def prepare(self, sequence):
            program = super().prepare(sequence)
            self.prepared_programs.append(program)
            return program

    sequencer = RecordingRuntimeSequencer(channels=hardware_channels, clock_hz=100_000_000, trigger_channels=["ch03"], sleep_scale=0.0)
    state = na.PulseTableState(
        channels=["ch00", "ch02", "ch03"],
        periods=[
            na.PulsePeriod(100, (1, 0, 0), unit="ns"),
            na.PulsePeriod("x", (1, 1, 1), unit="str (ns)"),
            na.PulsePeriod(100, (0, 0, 0), unit="ns"),
        ],
        time_step_ns=10,
        x_ns=1_000,
        repeat_forever=True,
    )
    pulse = na.bind_pulse(sequencer, state)

    scan = exp.readout.detection_time(
        [2e-6, 4e-6],
        shots=2,
        reference_shots=2,
        reference_exposure=8e-6,
        live=False,
        display=False,
        pulse=pulse,
    )

    assert scan.summary()["finished"] is True
    assert pulse.x_ns == 4_000
    assert [program.loop_end_tick for program in sequencer.prepared_programs] == [820, 220, 420]
    assert [program.trigger_count for program in sequencer.prepared_programs] == [2, 2, 2]
    assert [program.source_sequence["repeat_count"] for program in sequencer.prepared_programs] == [2, 2, 2]
    assert sequencer.last_program is not None
    assert sequencer.last_program.channels == hardware_channels
    assert sequencer.last_program.trigger_count == 2
    assert sequencer.last_program.repeat_forever is False
    assert all(mask < (1 << 4) for mask in sequencer.last_program.masks)


def test_timing_subsystem_bind_pulse_loads_json_for_40ch_remote_style_scan(tmp_path):
    exp = na.connect("virtual")
    exp.readout.sitemap(frames=3, display=False)
    hardware_channels = [f"ch{i:02d}" for i in range(40)]

    class RecordingRuntimeSequencer(na.RuntimeSequencer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.prepared_programs = []

        def prepare(self, sequence):
            program = super().prepare(sequence)
            self.prepared_programs.append(program)
            return program

    sequencer = RecordingRuntimeSequencer(channels=hardware_channels, clock_hz=100_000_000, trigger_channels=["ch03"], sleep_scale=0.0)
    exp.devices.devices["sequencer"] = sequencer
    state = na.PulseTableState(
        channels=["ch00", "ch02", "ch03"],
        periods=[
            na.PulsePeriod(100, (1, 0, 0), unit="ns"),
            na.PulsePeriod("x", (1, 1, 1), unit="str (ns)"),
            na.PulsePeriod(100, (0, 0, 0), unit="ns"),
        ],
        time_step_ns=10,
        x_ns=1_000,
        repeat_forever=True,
    )
    path = state.save(tmp_path / "camera_imaging.json")

    pulse = exp.timing.bind_pulse(path)
    assert pulse.sequencer is exp.sequencer
    assert pulse.snapshot()["sequencer_channels"] == hardware_channels
    pulse.x = 2_000
    single = pulse.on_pulse(wait=True, timeout=1.0, repeat_forever=False)

    assert single.channels == hardware_channels
    assert single.repeat_forever is False
    assert single.loop_end_tick == 220
    assert all(mask < (1 << 4) for mask in single.masks)

    scan = exp.readout.detection_time(
        [2e-6, 4e-6],
        shots=2,
        reference_shots=2,
        reference_exposure=8e-6,
        live=False,
        display=False,
        pulse=pulse,
    )

    assert scan.summary()["finished"] is True
    assert pulse.x_ns == 4_000
    assert sequencer.prepared_programs[-1].channels == hardware_channels
    assert sequencer.prepared_programs[-1].trigger_count == 2
    assert sequencer.prepared_programs[-1].repeat_forever is False


def test_bound_pulse_frame_sequence_uses_requested_frames_not_gui_repeat_count():
    sequencer = na.RuntimeSequencer(
        channels=["ch00", "ch03"],
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
        sleep_scale=0.0,
    )
    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod(100, (1, 0), unit="ns"),
            na.PulsePeriod(20, (1, 1), unit="ns"),
            na.PulsePeriod(20, (0, 0), unit="ns"),
        ],
        time_step_ns=10,
        repeat_start=1,
        repeat_end=1,
        repeat_count=500,
        repeat_forever=True,
    )
    pulse = na.bind_pulse(sequencer, state)

    sequence = pulse.frame_sequence(2)
    program = sequencer.prepare(sequence)

    assert na.count_trigger_pulses(sequence, trigger_channels=["ch03"]) == 2
    assert program.trigger_count == 2
    assert program.repeat_forever is False
    assert program.loop_count == 2
    assert program.loop_end_tick == 14
    assert program.ticks == [0, 10, 12, 14]
    assert program.duration == 28 / 100_000_000


def test_pulse_table_rejects_times_off_minimal_grid():
    state = na.PulseTableState(
        channels=["trap", "qcm_trigger"],
        periods=[
            na.PulsePeriod(100, (1, 0), unit="ns"),
            na.PulsePeriod("x", (0, 1), unit="str (ns)"),
        ],
        x_ns=20,
        time_step_ns=1,
    )

    assert state.to_sequence(x_ns=20, time_step_ns=10).validate(clock_hz=100e6, channels=state.channels).ok
    try:
        state.to_sequence(x_ns=25, time_step_ns=10)
    except ValueError as exc:
        assert "integer multiple" in str(exc)
    else:
        raise AssertionError("pulse table should reject x values off the minimal time grid")

    try:
        na.PulseTableState(
            channels=["trap"],
            periods=[na.PulsePeriod(2.5, (1,), unit="ns")],
            time_step_ns=1,
        )
    except ValueError as exc:
        assert "integer multiple" in str(exc)
    else:
        raise AssertionError("pulse table should reject non-integer-ns duration at 1 ns step")


def test_pulse_sequence_clock_validation_rejects_off_tick_edges():
    seq = na.PulseSequence(name="off_grid").pulse("trap", 0.0, 25e-9)
    report = seq.validate(clock_hz=100e6, channels=["trap"])

    assert not report.ok
    assert any("clock grid" in error for error in report.errors)
    try:
        na.compile_runtime_program(seq, channels=["trap"], clock_hz=100e6)
    except ValueError as exc:
        assert "clock grid" in str(exc)
    else:
        raise AssertionError("runtime compile should reject pulses that are off the FPGA clock grid")


def test_pulse_table_from_sequence_materializes_delays_without_double_applying():
    seq = na.PulseSequence(name="delayed").pulse("qcm_trigger", 0.0, 20e-9).delay("qcm_trigger", 10e-9)
    state = na.PulseTableState.from_sequence(seq, channels=["trap", "qcm_trigger"], clock_hz=100e6)
    round_trip = state.to_sequence()

    assert state.delays == {}
    assert [(p.channel, round(p.start, 10), round(p.duration, 10)) for p in round_trip.effective_pulses()] == [
        ("qcm_trigger", 10e-9, 20e-9)
    ]


def _simulate_pulse_streamer(ticks, masks):
    """Small behavioral model of the HDL run loop after the start transition is accepted."""

    active_count = len(ticks)
    final_tick = 0 if active_count == 0 else int(ticks[-1])
    edge_index = 0
    time_count = 0
    state_mask = 0
    running = active_count != 0
    done = active_count == 0
    history = []
    while running:
        if edge_index < active_count and time_count == int(ticks[edge_index]):
            state_mask = int(masks[edge_index])
            edge_index += 1
        if time_count >= final_tick:
            running = False
            done = True
            state_mask = 0
        history.append((time_count, state_mask, running, done))
        if running:
            time_count += 1
    return history


def _simulate_pulse_streamer_program_steps(program, *, steps: int) -> list[int]:
    """Physical-cycle model for FPGA loop/repeat metadata."""

    ticks = [int(tick) for tick in program.ticks]
    masks = [int(mask) for mask in program.masks]
    active_count = len(ticks)
    if active_count == 0:
        return [0] * steps

    final_tick = ticks[-1]
    loop_start_index = int(program.loop_start_index)
    loop_start_tick = ticks[loop_start_index]
    loop_start_mask = masks[loop_start_index]
    loop_end_tick = int(program.loop_end_tick)
    loop_count = max(1, int(program.loop_count))
    loops_remaining = loop_count
    repeat_forever = bool(program.repeat_forever)

    if ticks[0] == 0:
        state_mask = masks[0]
        time_count = 1
        edge_index = 1
    else:
        state_mask = 0
        time_count = 0
        edge_index = 0
    running = True
    history = [state_mask]

    while len(history) < steps:
        if running:
            if loop_count > 1 and loops_remaining > 1 and time_count >= loop_end_tick:
                state_mask = loop_start_mask
                time_count = loop_start_tick + 1
                edge_index = loop_start_index + 1
                loops_remaining -= 1
            elif time_count >= final_tick:
                if repeat_forever:
                    if ticks[0] == 0:
                        state_mask = masks[0]
                        time_count = 1
                        edge_index = 1
                    else:
                        state_mask = 0
                        time_count = 0
                        edge_index = 0
                    loops_remaining = loop_count
                else:
                    running = False
                    state_mask = 0
            else:
                if edge_index < active_count and time_count == ticks[edge_index]:
                    state_mask = masks[edge_index]
                    edge_index += 1
                time_count += 1
        history.append(state_mask)
    return history


def test_command_sequencer_backend_writes_program_and_runs_fire_command(tmp_path):
    marker = tmp_path / "fire_marker.txt"
    command = (
        f'"{sys.executable}" -c '
        f'"import os, pathlib; pathlib.Path(r\'{marker}\').write_text(os.environ[\'ZLC_SEQUENCE_ID\'])"'
    )
    seq = na.imaging_sequence(exposure=1e-4, load=True)
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "qcm_trigger"])
    backend = na.CommandSequencerBackend(tmp_path, fire_command=command)

    backend.fire(program)

    assert marker.read_text(encoding="utf-8") == program.sequence_id
    payload = json.loads((tmp_path / "prepared_program.json").read_text(encoding="utf-8"))
    assert payload["trigger_count"] == 1
    assert payload["source_sequence"]["name"] == seq.name


def test_command_sequencer_backend_error_includes_log_tail(tmp_path):
    command = f'"{sys.executable}" -c "print(\'prepare failed detail\'); raise SystemExit(7)"'
    seq = na.imaging_sequence(exposure=1e-4, load=True)
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "qcm_trigger"])
    backend = na.CommandSequencerBackend(tmp_path, prepare_command=command)

    try:
        backend.prepare(program)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("prepare command should have failed")

    assert "failed with code 7" in message
    assert "prepare.log tail" in message
    assert "prepare failed detail" in message
    assert "prepare failed detail" in (tmp_path / "prepare.log").read_text(encoding="utf-8")


def test_sequencer_service_skips_duplicate_prepare_uploads():
    prepared: list[str] = []

    def prepare_callback(program):
        prepared.append(program.sequence_id)

    service = na.SequencerService(
        channels=["ch00", "ch03"],
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
        prepare_callback=prepare_callback,
    )
    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod("x", (1, 0), unit="str (ns)"),
            na.PulsePeriod(20, (1, 1), unit="ns"),
            na.PulsePeriod(20, (0, 0), unit="ns"),
        ],
        time_step_ns=10,
        x_ns=100,
        repeat_forever=False,
    )

    first = service.prepare(state)
    second = service.prepare(state)
    third = service.prepare(state.with_x(200))

    assert first["sequence_id"] == second["sequence_id"]
    assert third["sequence_id"] != first["sequence_id"]
    assert prepared == [first["sequence_id"], third["sequence_id"]]
    assert [row["cached"] for row in service.history if row["action"] == "prepare"] == [False, True, False]


def test_sequencer_service_safe_state_invalidates_prepare_cache():
    prepared: list[str] = []
    safe_calls: list[bool] = []

    def prepare_callback(program):
        prepared.append(program.sequence_id)

    service = na.SequencerService(
        channels=["ch00", "ch03"],
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
        prepare_callback=prepare_callback,
        safe_state_callback=lambda: safe_calls.append(True),
    )
    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod(20, (1, 1), unit="ns"),
            na.PulsePeriod(20, (0, 0), unit="ns"),
        ],
        time_step_ns=10,
        repeat_forever=True,
    )

    first = service.prepare(state)
    second = service.prepare(state)
    service.set_safe_state()
    assert service.snapshot()["prepared_program"] is None
    third = service.prepare(state)
    service.abort()
    assert service.snapshot()["prepared_program"] is None

    assert first["sequence_id"] == second["sequence_id"] == third["sequence_id"]
    assert prepared == [first["sequence_id"], third["sequence_id"]]
    assert safe_calls == [True, True]
    assert [row["cached"] for row in service.history if row["action"] == "prepare"] == [False, True, False]
    assert [row["invalidated"] for row in service.history if row["action"] in {"safe", "abort"}] == [True, True]


class _FakeVivadoStdout:
    def __init__(self):
        self.items: queue.Queue[str | None] = queue.Queue()

    def push(self, line: str) -> None:
        self.items.put(line)

    def close(self) -> None:
        self.items.put(None)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        item = self.items.get(timeout=5)
        if item is None:
            raise StopIteration
        return item


class _FakeVivadoStdin:
    def __init__(self, stdout: _FakeVivadoStdout):
        self.stdout = stdout
        self.writes: list[str] = []

    def write(self, text: str) -> int:
        self.writes.append(text)
        match = re.search(r"ZLC_SESSION_(\d{6})_END", text)
        if match:
            marker = f"ZLC_SESSION_{match.group(1)}"
            self.stdout.push(f"{marker}_OK\n")
            self.stdout.push(f"{marker}_END\n")
        return len(text)

    def flush(self) -> None:
        return None


class _FakeVivadoProcess:
    def __init__(self, args, **_kwargs):
        self.args = args
        self.stdout = _FakeVivadoStdout()
        self.stdin = _FakeVivadoStdin(self.stdout)
        self.returncode = None

    def wait(self, timeout=None):
        self.returncode = 0
        self.stdout.close()
        return 0

    def terminate(self):
        self.returncode = -15
        self.stdout.close()


def test_vivado_pulse_streamer_session_reuses_one_vivado_process(tmp_path, monkeypatch):
    from Zou_lab_control.neutral_atom.devices import fpga_pulse_streamer as fps

    created: list[_FakeVivadoProcess] = []

    def fake_popen(args, **kwargs):
        process = _FakeVivadoProcess(args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(fps.subprocess, "Popen", fake_popen)
    hardware_channels = [f"ch{i:02d}" for i in range(40)]
    seq = na.PulseSequence(name="session").pulse("ch03", 0.0, 1e-6).forever(period=2e-6)
    program = na.compile_runtime_program(seq, channels=hardware_channels, clock_hz=100e6, trigger_channels=["ch03"])
    session = na.VivadoPulseStreamerSession(state_dir=tmp_path, vivado="fake_vivado", max_edges=1024, channel_count=40)

    session.prepare(program)
    session.fire(program)
    assert session.wait_done(program, timeout=1.0)
    session.safe_state()
    session.close()

    assert len(created) == 1
    assert created[0].args[:3] == ["fake_vivado", "-mode", "tcl"]
    script = "".join(created[0].stdin.writes)
    assert "connect_hw_server" in script
    assert "zlc_stage_probe $vio $zlc_prog_mask_probe" in script
    assert "zlc_stage_probe $vio $zlc_repeat_forever_probe 1" in script
    assert "ZLC pulse-streamer start pulse sent" in script
    assert "zlc_start_toggle_value" not in script
    assert "ZLC pulse-streamer safe state requested" in script
    assert json.loads((tmp_path / "prepared_program.json").read_text(encoding="utf-8"))["repeat_forever"] is True


def test_vivado_pulse_streamer_session_prepare_uses_differential_edge_upload(tmp_path, monkeypatch):
    from Zou_lab_control.neutral_atom.devices import fpga_pulse_streamer as fps

    created: list[_FakeVivadoProcess] = []

    def fake_popen(args, **kwargs):
        process = _FakeVivadoProcess(args, **kwargs)
        created.append(process)
        return process

    monkeypatch.setattr(fps.subprocess, "Popen", fake_popen)
    channels = ["ch00", "ch01", "ch02", "ch03"]
    first = na.RuntimeSequenceProgram(
        sequence_id="first",
        sequence_name="diff",
        clock_hz=100_000_000,
        channels=channels,
        ticks=[0, 10, 20, 30],
        masks=[1, 0, 8, 0],
        duration=30 / 100_000_000,
        trigger_count=1,
        repeat_forever=True,
        loop_start_index=0,
        loop_end_tick=30,
        loop_count=1,
    )
    second = na.RuntimeSequenceProgram(
        sequence_id="second",
        sequence_name="diff",
        clock_hz=100_000_000,
        channels=channels,
        ticks=[0, 10, 22, 30],
        masks=[1, 0, 8, 0],
        duration=30 / 100_000_000,
        trigger_count=1,
        repeat_forever=True,
        loop_start_index=0,
        loop_end_tick=30,
        loop_count=1,
    )
    moved_loop = na.RuntimeSequenceProgram(
        sequence_id="moved-loop",
        sequence_name="diff",
        clock_hz=100_000_000,
        channels=channels,
        ticks=[0, 10, 22, 30],
        masks=[1, 0, 8, 0],
        duration=30 / 100_000_000,
        trigger_count=1,
        repeat_forever=True,
        loop_start_index=2,
        loop_end_tick=30,
        loop_count=2,
    )
    session = na.VivadoPulseStreamerSession(state_dir=tmp_path, vivado="fake_vivado", max_edges=1024, channel_count=4)

    session.prepare(first)
    session.prepare(second)
    session.prepare(moved_loop)
    session.close()

    assert len(created) == 1
    full_prepare = created[0].stdin.writes[1]
    diff_prepare = created[0].stdin.writes[2]
    loop_metadata_prepare = created[0].stdin.writes[3]
    assert full_prepare.count("zlc_stage_probe $vio $zlc_prog_addr_probe") == 4
    assert full_prepare.count("zlc_commit_probes $zlc_batch") == 6
    assert full_prepare.index("zlc_stage_probe $vio $zlc_reset_probe 1") < full_prepare.index(
        "zlc_stage_probe $vio $zlc_prog_addr_probe 0"
    )
    assert "after $zlc_prepare_reset_settle_ms" in full_prepare
    assert "wrote 4/4 edge rows" in full_prepare
    assert "reset_settle_ms=$zlc_prepare_reset_settle_ms" in full_prepare
    assert diff_prepare.count("zlc_stage_probe $vio $zlc_prog_addr_probe") == 3
    assert diff_prepare.count("zlc_commit_probes $zlc_batch") == 5
    assert "zlc_stage_probe $vio $zlc_prog_addr_probe 0" in diff_prepare
    assert "zlc_stage_probe $vio $zlc_prog_addr_probe 1" not in diff_prepare
    assert "zlc_stage_probe $vio $zlc_prog_addr_probe 2" in diff_prepare
    assert "zlc_stage_probe $vio $zlc_prog_addr_probe 3" in diff_prepare
    assert "wrote 3/4 edge rows" in diff_prepare
    assert "reset_settle_ms=$zlc_prepare_reset_settle_ms" in diff_prepare
    assert loop_metadata_prepare.count("zlc_stage_probe $vio $zlc_prog_addr_probe") == 3
    assert loop_metadata_prepare.count("zlc_commit_probes $zlc_batch") == 5
    assert "zlc_stage_probe $vio $zlc_loop_start_addr_probe 2" in loop_metadata_prepare
    assert "zlc_stage_probe $vio $zlc_loop_count_probe 2" in loop_metadata_prepare
    assert loop_metadata_prepare.index("zlc_stage_probe $vio $zlc_loop_start_addr_probe 2") < loop_metadata_prepare.index(
        "zlc_stage_probe $vio $zlc_prog_addr_probe 0"
    )
    assert "zlc_stage_probe $vio $zlc_prog_addr_probe 0" in loop_metadata_prepare
    assert "zlc_stage_probe $vio $zlc_prog_addr_probe 1" not in loop_metadata_prepare
    assert "zlc_stage_probe $vio $zlc_prog_addr_probe 2" in loop_metadata_prepare
    assert "zlc_stage_probe $vio $zlc_prog_addr_probe 3" in loop_metadata_prepare
    assert "wrote 3/4 edge rows reset_settle_ms=$zlc_prepare_reset_settle_ms repeat_forever=1 loop_start=2 loop_end=30 loop_count=2" in loop_metadata_prepare


def test_sequencer_server_warm_starts_vivado_session_before_accepting_clients(tmp_path, monkeypatch):
    from Zou_lab_control.neutral_atom.devices import fpga_pulse_streamer as fps
    from Zou_lab_control.neutral_atom.devices import sequencer_server

    events: list[str] = []

    class FakeVivadoSession:
        def __init__(self, *, state_dir):
            events.append(f"init:{Path(state_dir).name}")

        def start(self):
            events.append("start")
            return self

        def prepare(self, program):
            events.append(f"prepare:{program.sequence_id}")

        def fire(self, program):
            events.append(f"fire:{program.sequence_id}")

        def wait_done(self, program, timeout=None):
            events.append(f"wait:{program.sequence_id}")
            return True

        def safe_state(self):
            events.append("safe")

    def fake_serve(service, *, host, port, start):
        events.append(f"serve:{host}:{port}:{start}:{service.snapshot()['state']}")
        return object()

    monkeypatch.setattr(fps, "VivadoPulseStreamerSession", FakeVivadoSession)
    monkeypatch.setattr(sequencer_server, "serve_runtime_sequencer", fake_serve)

    sequencer_server.run_server(
        channels=[f"ch{i:02d}" for i in range(40)],
        trigger_channels=["ch03"],
        host="127.0.0.1",
        port=18861,
        clock_hz=100_000_000,
        state_dir=tmp_path / "state40",
        backend="vivado-session",
        warm_start=True,
    )

    assert events[:3] == ["init:state40", "start", "serve:127.0.0.1:18861:True:idle"]


def test_sequencer_server_can_disable_vivado_warm_start_for_diagnostics(tmp_path, monkeypatch):
    from Zou_lab_control.neutral_atom.devices import fpga_pulse_streamer as fps
    from Zou_lab_control.neutral_atom.devices import sequencer_server

    events: list[str] = []

    class FakeVivadoSession:
        def __init__(self, *, state_dir):
            events.append(f"init:{Path(state_dir).name}")

        def start(self):
            events.append("start")
            return self

        def prepare(self, program):
            events.append(f"prepare:{program.sequence_id}")

        def fire(self, program):
            events.append(f"fire:{program.sequence_id}")

        def wait_done(self, program, timeout=None):
            return True

        def safe_state(self):
            return None

    monkeypatch.setattr(fps, "VivadoPulseStreamerSession", FakeVivadoSession)
    monkeypatch.setattr(sequencer_server, "serve_runtime_sequencer", lambda service, **kwargs: events.append("serve"))

    sequencer_server.run_server(
        channels=["ch00", "ch03"],
        trigger_channels=["ch03"],
        state_dir=tmp_path / "state40",
        backend="vivado-session",
        warm_start=False,
    )

    assert events == ["init:state40", "serve"]


def test_remote_sequencer_round_trip_uses_json_protocol(tmp_path):
    try:
        import rpyc  # noqa: F401
    except ImportError:
        return

    backend = na.CommandSequencerBackend(tmp_path)
    service = na.SequencerService(
        channels=["trap", "cooling", "probe", "qcm_trigger"],
        clock_hz=100e6,
        trigger_channels=["qcm_trigger"],
        prepare_callback=backend.prepare,
        fire_callback=backend.fire,
        wait_done_callback=backend.wait_done,
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = na.serve_runtime_sequencer(service, host="127.0.0.1", port=port, start=False)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(0.2)
    remote = na.RemoteSequencer(host="127.0.0.1", port=port, channels=["trap", "cooling", "probe", "qcm_trigger"], clock_hz=100e6)
    seq = na.sequence_for_frame_count(na.imaging_sequence(exposure=12e-6, load=True), 4)
    try:
        program = remote.prepare(seq)
        remote.fire(seq)
        assert remote.wait_done(timeout=1.0)
    finally:
        remote.close()
        server.close()

    payload = json.loads((tmp_path / "prepared_program.json").read_text(encoding="utf-8"))
    assert program.trigger_count == 4
    assert payload["trigger_count"] == 4
    assert payload["source_sequence"]["name"] == seq.name


def test_remote_pulse_controller_sends_pulse_table_x_over_json_protocol(tmp_path):
    try:
        import rpyc  # noqa: F401
    except ImportError:
        return

    prepared_programs: list[na.RuntimeSequenceProgram] = []
    fired_programs: list[na.RuntimeSequenceProgram] = []

    def prepare_callback(program):
        prepared_programs.append(program)

    def fire_callback(program):
        fired_programs.append(program)

    hardware_channels = [f"ch{i:02d}" for i in range(40)]
    service = na.SequencerService(
        channels=hardware_channels,
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
        prepare_callback=prepare_callback,
        fire_callback=fire_callback,
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = na.serve_runtime_sequencer(service, host="127.0.0.1", port=port, start=False)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(0.2)

    remote = na.RemoteSequencer(
        host="127.0.0.1",
        port=port,
        channels=["ch00", "ch03"],
        clock_hz=1.0,
        trigger_channels=["ch03"],
    )
    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod(100, (1, 0), unit="ns"),
            na.PulsePeriod("x", (1, 1), unit="str (ns)"),
            na.PulsePeriod(100, (0, 0), unit="ns"),
        ],
        time_step_ns=10,
        x_ns=1_000,
        repeat_forever=True,
        visible_channels=["ch00", "ch03"],
    )
    pulse = na.bind_pulse(remote, state)
    try:
        pulse.x = 2_000
        program = pulse.on_pulse(wait=True, timeout=1.0, repeat_forever=False)
        snapshot = pulse.snapshot()
    finally:
        remote.close()
        server.close()

    assert remote.channels == hardware_channels
    assert remote.clock_hz == 100_000_000
    assert pulse.x_ns == 2_000
    assert program.channels == hardware_channels
    assert program.ticks == [0, 10, 210, 220]
    assert program.masks == [1 << 0, (1 << 0) | (1 << 3), 0, 0]
    assert program.repeat_forever is False
    assert program.trigger_count == 1
    assert all(mask >> 4 == 0 for mask in program.masks)
    assert [p.sequence_id for p in prepared_programs] == [program.sequence_id]
    assert [p.sequence_id for p in fired_programs] == [program.sequence_id]
    assert snapshot["sequencer_channels"] == hardware_channels
    assert snapshot["last_program"]["repeat_forever"] is False


def test_remote_detection_time_scan_uses_bound_pulse_controller_over_json_protocol(tmp_path):
    try:
        import rpyc  # noqa: F401
    except ImportError:
        return

    prepared_programs: list[na.RuntimeSequenceProgram] = []
    fired_programs: list[na.RuntimeSequenceProgram] = []
    hardware_channels = [f"ch{i:02d}" for i in range(40)]
    service = na.SequencerService(
        channels=hardware_channels,
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
        prepare_callback=lambda program: prepared_programs.append(program),
        fire_callback=lambda program: fired_programs.append(program),
    )
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = na.serve_runtime_sequencer(service, host="127.0.0.1", port=port, start=False)
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(0.2)

    exp = na.connect("virtual")
    exp.readout.sitemap(frames=3, display=False)
    remote = na.RemoteSequencer(
        host="127.0.0.1",
        port=port,
        channels=["ch00", "ch03"],
        clock_hz=1.0,
        trigger_channels=["ch03"],
    )
    state = na.PulseTableState(
        channels=["ch00", "ch02", "ch03"],
        periods=[
            na.PulsePeriod(100, (1, 0, 0), unit="ns"),
            na.PulsePeriod("x", (1, 1, 1), unit="str (ns)"),
            na.PulsePeriod(100, (0, 0, 0), unit="ns"),
        ],
        time_step_ns=10,
        x_ns=1_000,
        repeat_forever=True,
    )
    pulse = na.bind_pulse(remote, state)
    try:
        scan = exp.readout.detection_time(
            [2e-6, 4e-6],
            shots=2,
            reference_shots=2,
            reference_exposure=8e-6,
            live=False,
            display=False,
            pulse=pulse,
        )
    finally:
        remote.close()
        server.close()
        exp.close()

    assert scan.summary()["finished"] is True
    assert remote.channels == hardware_channels
    assert remote.clock_hz == 100_000_000
    assert pulse.x_ns == 4_000
    assert [program.channels for program in prepared_programs] == [hardware_channels] * 3
    assert [program.trigger_count for program in prepared_programs] == [2, 2, 2]
    assert [program.loop_end_tick for program in prepared_programs] == [820, 220, 420]
    assert all(not program.repeat_forever for program in prepared_programs)
    assert [program.sequence_id for program in fired_programs] == [program.sequence_id for program in prepared_programs]
    assert all(mask < (1 << 4) for program in prepared_programs for mask in program.masks)


def test_hardware_tutorial_is_real_hardware_not_virtual_demo():
    hardware_text = "\n".join(cell["source"] for cell in neutral_atom_hardware_tutorial_cells())
    fpga_text = "\n".join(cell["source"] for cell in neutral_atom_fpga_server_cells())

    assert 'na.connect("virtual")' not in hardware_text
    assert "VirtualCamera" not in hardware_text
    assert "VirtualSequencer" not in hardware_text
    assert '"remote_template"' in hardware_text
    assert "open_devices=True" in hardware_text
    assert "zf.require_attrs" not in hardware_text
    assert "isinstance(" not in hardware_text
    assert "exp.camera.open()" not in hardware_text
    assert "exp.devices.sequencer.open()" not in hardware_text
    assert "results_real_hardware" not in hardware_text
    assert "neutral_atom.devices.sequencer_server" in fpga_text
    assert "fpga_pulse_streamer" in fpga_text
    assert "legacy_address_switch" not in fpga_text
    assert "na.run_sequencer_server" in fpga_text
    assert "address_switch.xpr" not in fpga_text
    assert "qCMOS.py" not in hardware_text + fpga_text
    assert "pxie_control" not in hardware_text + fpga_text


def test_real_device_templates_load_without_hardware_connection():
    manual = na.load_devices("manual_template")
    remote = na.load_devices("remote_template", overrides={"sequencer": {"host": "192.168.0.21", "port": 18862}})
    hardware_channels = [f"ch{i:02d}" for i in range(40)]

    assert isinstance(manual.camera, na.QCMOSCamera)
    assert manual.camera.dcam_module_name == na.DEFAULT_DCAM_MODULE
    assert isinstance(manual.sequencer, na.ManualSequencer)
    assert manual.sequencer.channels == hardware_channels
    assert manual.sequencer.trigger_channels == ("ch03",)
    assert isinstance(remote.camera, na.QCMOSCamera)
    assert remote.camera.dcam_module_name == na.DEFAULT_DCAM_MODULE
    assert isinstance(remote.sequencer, na.RemoteSequencer)
    assert remote.sequencer.host == "192.168.0.21"
    assert remote.sequencer.port == 18862
    assert remote.sequencer.channels == hardware_channels
    assert remote.sequencer.trigger_channels == ("ch03",)
    assert remote.sequencer.snapshot()["connected"] is False

    exp = na.connect("remote_template", sequencer={"host": "192.168.0.22"})
    assert exp.devices.sequencer.host == "192.168.0.22"
    assert exp.camera.dcam_module_name == na.DEFAULT_DCAM_MODULE
    assert exp.sequence.channels == ["ch00", "ch01", "ch02", "ch03"]

    try:
        na.load_devices("remote_template", overrides={"sequencer": {"host": "0.0.0.0"}})
    except ValueError as exc:
        assert "RemoteSequencer host" in str(exc)
    else:
        raise AssertionError("RemoteSequencer accepted a non-client host.")


def test_load_devices_can_open_device_graph(monkeypatch):
    from Zou_lab_control.neutral_atom.devices import registry

    class TrackingCamera(na.CameraDevice):
        events = []

        @property
        def exposure(self):
            return 1e-3

        def configure(self, *, exposure=None, **kwargs):
            pass

        def acquire(self, frames=1, *, sequence=None, sequencer=None, **kwargs):
            return [np.zeros((2, 2))]

        def open(self):
            self.events.append("camera.open")
            return self

        def close(self):
            self.events.append("camera.close")

    class TrackingSequencer(na.SequencerDevice):
        events = TrackingCamera.events
        channels = ["qcm_trigger"]
        clock_hz = 1e6

        def prepare(self, sequence):
            pass

        def fire(self, sequence=None):
            pass

        def open(self):
            self.events.append("sequencer.open")
            return self

        def close(self):
            self.events.append("sequencer.close")

    monkeypatch.setitem(registry.DEVICE_CLASSES, "TrackingCamera", TrackingCamera)
    monkeypatch.setitem(registry.DEVICE_CLASSES, "TrackingSequencer", TrackingSequencer)

    devices = na.load_devices(
        {
            "camera": {"type": "TrackingCamera"},
            "sequencer": {"type": "TrackingSequencer"},
        },
        open_devices=True,
    )

    assert TrackingCamera.events == ["sequencer.open", "camera.open"]
    devices.close()
    assert TrackingCamera.events == ["sequencer.open", "camera.open", "camera.close", "sequencer.close"]


def test_sequencer_server_reports_client_endpoints(monkeypatch, capsys):
    from Zou_lab_control.neutral_atom.devices import sequencer_server

    assert sequencer_server._client_addresses("192.168.0.20") == ["192.168.0.20"]

    monkeypatch.setattr(sequencer_server, "_client_addresses", lambda host: ["192.168.0.20", "10.0.0.5"])
    sequencer_server._print_client_endpoints("0.0.0.0", 18861)
    output = capsys.readouterr().out

    assert "Client endpoints:" in output
    assert "192.168.0.20:18861" in output
    assert "10.0.0.5:18861" in output
    assert 'sequencer={"host": "192.168.0.20", "port": 18861}' in output


def test_qcmos_camera_acquire_uses_dcam_and_expanded_sequencer(monkeypatch):
    class FakeApi:
        initialized = False

        @classmethod
        def init(cls):
            cls.initialized = True

        @classmethod
        def uninit(cls):
            cls.initialized = False

    class FakeDcam:
        instance = None

        def __init__(self, index):
            self.index = index
            self.props = []
            self.frames = 0
            self.started = False
            self.released = False
            FakeDcam.instance = self

        def dev_open(self):
            return True

        def dev_close(self):
            self.closed = True

        def lasterr(self):
            return "fake error"

        def prop_setvalue(self, prop, value):
            self.props.append((prop, value))
            return True

        def buf_alloc(self, frames):
            self.frames = int(frames)
            return True

        def cap_start(self, bSequence=True):
            self.started = bool(bSequence)
            return True

        def wait_capevent_frameready(self, timeout):
            return True

        def cap_transferinfo(self):
            return types.SimpleNamespace(nFrameCount=self.frames)

        def buf_getframedata(self, index):
            return index, np.full((3, 4), index, dtype=np.uint16)

        def cap_stop(self):
            self.started = False
            return True

        def buf_release(self):
            self.released = True
            return True

    fake_module = types.SimpleNamespace(
        Dcamapi=FakeApi,
        Dcam=FakeDcam,
        DCAM_IDPROP=types.SimpleNamespace(
            EXPOSURETIME="EXPOSURETIME",
            TRIGGERSOURCE="TRIGGERSOURCE",
            TRIGGERACTIVE="TRIGGERACTIVE",
            TRIGGERPOLARITY="TRIGGERPOLARITY",
            READOUTSPEED="READOUTSPEED",
            SUBARRAYMODE="SUBARRAYMODE",
            SUBARRAYHSIZE="SUBARRAYHSIZE",
            SUBARRAYHPOS="SUBARRAYHPOS",
            SUBARRAYVSIZE="SUBARRAYVSIZE",
            SUBARRAYVPOS="SUBARRAYVPOS",
        ),
        DCAMPROP=types.SimpleNamespace(
            TRIGGERSOURCE=types.SimpleNamespace(EXTERNAL="EXTERNAL"),
            TRIGGERACTIVE=types.SimpleNamespace(EDGE="EDGE"),
            TRIGGERPOLARITY=types.SimpleNamespace(POSITIVE="POSITIVE"),
            MODE=types.SimpleNamespace(ON="ON"),
        ),
    )
    monkeypatch.setitem(sys.modules, "fake_dcam_for_qcmos_test", fake_module)

    class FakeSequencer:
        channels = ["trap", "cooling", "probe", "qcm_trigger"]
        clock_hz = 250e6
        trigger_channels = ("qcm_trigger",)

        def __init__(self):
            self.prepared = None
            self.fired = None
            self.done_wait = None

        def prepare(self, sequence):
            self.prepared = sequence

        def fire(self, sequence):
            self.fired = sequence

        def wait_done(self, timeout):
            self.done_wait = timeout
            return True

    camera = na.QCMOSCamera({"exposure": 2e-3, "roi": [1, 4, 2, 3], "timeout_ms": 100}, dcam_module="fake_dcam_for_qcmos_test")
    sequencer = FakeSequencer()
    sequence = na.imaging_sequence(exposure=1e-3, load=True)
    images = camera.acquire(2, sequence=sequence, sequencer=sequencer)

    assert len(images) == 2
    assert FakeApi.initialized is True
    assert FakeDcam.instance.released is True
    exposure_writes = [value for prop, value in FakeDcam.instance.props if prop == "EXPOSURETIME"]
    assert exposure_writes[-1] == 1e-3
    assert ("TRIGGERSOURCE", "EXTERNAL") in FakeDcam.instance.props
    assert ("TRIGGERACTIVE", "EDGE") in FakeDcam.instance.props
    assert na.count_trigger_pulses(sequencer.prepared) == 2
    assert sequencer.fired is sequencer.prepared
    assert sequencer.done_wait is not None
    camera.close()
    assert FakeApi.initialized is False

    camera = na.QCMOSCamera({"exposure": 2e-3, "timeout_ms": 100}, dcam_module="fake_dcam_for_qcmos_test")
    sequencer = FakeSequencer()
    sequencer.channels = ["ch00", "ch03"]
    sequencer.trigger_channels = ("ch03",)
    pulse_state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod(100, (1, 0), unit="ns"),
            na.PulsePeriod(20, (1, 1), unit="ns"),
            na.PulsePeriod(100, (0, 0), unit="ns"),
        ],
        time_step_ns=10,
        repeat_forever=True,
    )
    images = camera.acquire(3, sequence=pulse_state, sequencer=sequencer)

    assert len(images) == 3
    assert isinstance(sequencer.prepared, na.PulseSequence)
    assert sequencer.prepared.repeat_forever is False
    assert na.count_trigger_pulses(sequencer.prepared, trigger_channels=["ch03"]) == 3
    assert sequencer.fired is sequencer.prepared
    assert sequencer.done_wait is not None
    camera.close()
    assert FakeApi.initialized is False

    camera = na.QCMOSCamera({"exposure": 2e-3, "timeout_ms": 100}, dcam_module="fake_dcam_for_qcmos_test")
    sequencer = FakeSequencer()
    sequencer.channels = ["ch00", "ch03"]
    sequencer.trigger_channels = ("ch03",)
    pulse = na.bind_pulse(sequencer, pulse_state)
    images = camera.acquire(2, sequence=pulse)

    assert len(images) == 2
    assert isinstance(sequencer.prepared, na.PulseSequence)
    assert na.count_trigger_pulses(sequencer.prepared, trigger_channels=["ch03"]) == 2
    assert sequencer.fired is sequencer.prepared
    camera.close()
    assert FakeApi.initialized is False


def test_standalone_calibration_and_detection_from_arrays():
    exp = na.connect("virtual")
    sequence = na.imaging_sequence(exposure=exp.camera.exposure, load=True, name="sitemap")
    images = exp.camera.acquire(4, sequence=sequence)
    sitemap = na.calibrate_sitemap_from_images(images, grid_shape=exp.devices.trap_array.grid_shape, display=False)
    threshold_images = exp.camera.capture(frames=8, display=False).images
    threshold = na.calibrate_threshold_from_images(threshold_images, sitemap.calibration, display=False)
    shot = na.detect_image(threshold_images[-1], threshold.calibration, display=False)

    assert sitemap.centers.shape[0] == exp.devices.trap_array.n_sites
    assert threshold.thresholds.shape == (exp.devices.trap_array.n_sites,)
    assert shot.occupied.dtype == bool
