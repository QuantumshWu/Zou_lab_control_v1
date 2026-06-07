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
import pytest

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
                "params": {"channels": ["trap", "cooling", "probe", "trig"]},
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
    seq = na.imaging_sequence(exposure=1e-3, load=True).delay("emCCD", 20e-9)
    report = seq.validate(clock_hz=50_000_000, channels=["trap", "cooling", "probe", "emCCD"])
    build = na.generate_verilog(seq, channels=["trap", "cooling", "probe", "emCCD"], clock_hz=50_000_000)
    files = na.write_verilog_bundle(build, tmp_path)

    assert report.ok
    assert seq.delays == {"emCCD": 20e-09}
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
    seq = na.PulseSequence(name="huge_repeat").pulse("emCCD", 0.0, 1e-6).repeated(100_000, period=2e-6)
    report = seq.validate(clock_hz=50_000_000, channels=["trap", "cooling", "probe", "emCCD"])
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "emCCD"], clock_hz=50_000_000)

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
    sequencer = na.RuntimeSequencer(channels=["trap", "cooling", "probe", "emCCD"], sleep_scale=0.0)
    program = sequencer.prepare(seq)
    sequencer.fire(seq)

    assert program.trigger_count == 2
    assert sequencer.wait_done(timeout=1.0)
    snapshot = sequencer.snapshot()
    assert snapshot["state"] == "done"
    assert snapshot["prepared_program"]["trigger_count"] == 2


def test_fpga_pulse_streamer_writes_hdl_and_upload_tcl(tmp_path):
    seq = na.sequence_for_frame_count(na.imaging_sequence(exposure=4e-6, load=True), 2)
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "emCCD"], clock_hz=50_000_000)

    files = na.write_pulse_streamer_hdl_bundle(tmp_path / "hdl", channels=["trap", "cooling", "probe", "emCCD"], max_edges=16)
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
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "emCCD"], clock_hz=50_000_000)

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
        channels=["trap", "trig"],
        ticks=[0, 10, 10],
        masks=[1, 2, 0],
        duration=1e-7,
        trigger_count=1,
    )
    bad_final_mask = na.RuntimeSequenceProgram(
        sequence_id="bad",
        sequence_name="bad",
        clock_hz=100e6,
        channels=["trap", "trig"],
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
    unsafe_project_dir = root / "fpga" / "pulse_streamer" / "build" / "zlc_pulse_streamer_legacy"
    unsafe_bitstream = unsafe_project_dir / "legacy.runs" / "impl_1" / "legacy_top.bit"
    unsafe_probes = unsafe_project_dir / "legacy.runs" / "impl_1" / "legacy_top.ltx"
    default_project_dir = root / "fpga" / "build" / "address_switch"
    monkeypatch.setenv("ZLC_PS_PROJECT_DIR", str(unsafe_project_dir))
    monkeypatch.setenv("ZLC_PS_VIVADO_PROJECT", str(unsafe_project_dir / "legacy.xpr"))
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
    assert "address_switch.xpr" in tcl
    assert "address_switch.runs" in tcl
    assert "zlc_pulse_streamer_top_address_switch.bit" in tcl
    assert "zlc_pulse_streamer_top_address_switch.ltx" in tcl


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
            "trig",
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
        xdc=root / "references" / "source_archives" / "address_switch" / "address_switch.srcs" / "constrs_1" / "new" / "addre.xdc",
        max_channel_count=62,
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
        channel_labels={"ch00": "trap"},
        time_step_ns=20,
        repeat_forever=True,
    )

    channels = pulse_gui_launcher._resolve_channels(args, subset_state)
    aligned = subset_state.aligned_to_channels(channels)
    program = aligned.compile(clock_hz=50_000_000, trigger_channels=["ch11"])

    assert channels == [f"ch{i:02d}" for i in range(62)]
    assert aligned.channels == channels
    assert aligned.visible_channels == ["ch00", "ch03"]
    assert all(period.states[1:3] == (0, 0) for period in aligned.periods)
    assert max(program.masks) < (1 << 62)
    labels = pulse_gui_launcher._resolve_channel_labels(args, channels, subset_state)
    pins = pulse_gui_launcher._resolve_channel_pins(args, channels)
    assert pulse_gui_launcher._resolve_trigger_channels(args, channels, labels) == ["ch11"]
    assert labels["ch00"] == "trap"
    assert labels["ch03"] == "probe"
    assert labels["ch06"] == "trig"
    assert labels["ch11"] == "emCCD"
    assert labels["ch04"] == "pushout"
    assert labels["ch39"] == "da_clk1"
    assert pins["ch00"] == "F15"
    assert pins["ch11"] == "M13"

    explicit_args = types.SimpleNamespace(**{**args.__dict__, "channel_count": 4})
    assert pulse_gui_launcher._resolve_channels(explicit_args, subset_state) == ["ch00", "ch01", "ch02", "ch03"]

    remote_args = types.SimpleNamespace(
        **{
            **args.__dict__,
            "remote_host": "127.0.0.1",
            "remote_port": 18861,
            "clock_hz": 50_000_000,
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
    assert fallback_triggers == ["ch11"]
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
    # The docs reorg replaced PROJECT_OVERVIEW.md and the implementation half of
    # the old hardware runbook with the consolidated maintainer note, and turned
    # the hardware manual into the three-manual layout. Read the new homes.
    maintainer_notes = (root / "docs" / "MAINTAINER_NOTES.md").read_text(encoding="utf-8")
    main_manual_template = (root / "Zou_lab_control" / "neutral_atom" / "content" / "manual_templates" / "main_manual_zh.texbody").read_text(encoding="utf-8")
    fpga_manual_template = (root / "Zou_lab_control" / "neutral_atom" / "content" / "manual_templates" / "fpga_manual_zh.texbody").read_text(encoding="utf-8")
    fpga_manual_tex = (root / "docs" / "fpga_manual" / "fpga_manual_zh.tex").read_text(encoding="utf-8")
    fpga_readme = (root / "fpga" / "README.md").read_text(encoding="utf-8")
    streamer_readme = (fpga / "README.md").read_text(encoding="utf-8")
    frontend_readme = (root / "Zou_lab_control" / "frontend" / "README.md").read_text(encoding="utf-8")
    pulses_readme = (root / "pulses" / "README.md").read_text(encoding="utf-8")
    tests_readme = (root / "tests" / "README.md").read_text(encoding="utf-8")
    frontend_manual_template = (root / "Zou_lab_control" / "frontend" / "content" / "manual_templates" / "frontend_manual_zh.texbody").read_text(encoding="utf-8")
    fpga_notebook_text = "\n".join(cell["source"] for cell in neutral_atom_fpga_server_cells())
    hardware_notebook_text = "\n".join(cell["source"] for cell in neutral_atom_hardware_tutorial_cells())

    for name in ("install_requirements.bat", "pulse_gui.bat", "start_tutorials_jupyter_lab.bat"):
        assert (root / name).exists(), name
    assert not (root / "build_and_program.bat").exists()
    assert not (root / "run_server.bat").exists()
    assert {path.name for path in root.glob("*.bat")} == {
        "install_requirements.bat",
        "pulse_gui.bat",
        "start_tutorials_jupyter_lab.bat",
    }

    required = {
        # Validated edge-table engine + the JTAG-to-AXI on-chip loader = the active
        # build target.  The loader copies the program image from BRAM into the
        # (unchanged, seamless) engine, so repeat + scan stay gapless and tick-exact.
        "zlc_pulse_streamer.v",
        "zlc_axi_program_loader.v",
        "zlc_pulse_streamer_loader_top.v",
        "create_project_loader.tcl",
        "program_fpga_loader.tcl",
        # VIO edge-table path kept as an alternate backend (same engine, VIO control).
        "zlc_pulse_streamer_top_address_switch.v",
        "create_project_address_switch.tcl",
        "check_address_switch_synth.tcl",
        "diagnose_hw_target.tcl",
        "program_fpga_address_switch.tcl",
        "README.md",
    }
    assert required.issubset({path.name for path in fpga.iterdir()})
    # The discarded per-channel run-length residue must be gone (no legacy residue).
    assert {path.name for path in fpga.iterdir()}.isdisjoint({
        "zlc_pulse_streamer_runlength.v",
        "zlc_runlength_engine.v",
        "zlc_pulse_streamer_runlength_top.v",
        "create_project_runlength.tcl",
        "program_fpga_runlength.tcl",
    })
    legacy_width = "40" + "ch"
    removed = {
        f"zlc_pulse_streamer_top_{legacy_width}.v",
        f"zlc_pulse_streamer_{legacy_width}.xdc",
        f"zlc_pulse_streamer_{legacy_width}.xdc.template",
        f"create_project_{legacy_width}.tcl",
        f"check_{legacy_width}_synth.tcl",
        f"program_fpga_{legacy_width}.tcl",
    }
    assert removed.isdisjoint({path.name for path in fpga.iterdir()})
    # The legacy-width artifacts must not survive anywhere under docs/ either
    # (the old pulse_streamer_test_report assets dir was deleted in the reorg).
    assert not list((root / "docs").rglob(f"*{legacy_width}*"))

    top = (fpga / "zlc_pulse_streamer_top_address_switch.v").read_text(encoding="utf-8")
    create_tcl = (fpga / "create_project_address_switch.tcl").read_text(encoding="utf-8")
    check_tcl = (fpga / "check_address_switch_synth.tcl").read_text(encoding="utf-8")
    program_tcl = (fpga / "program_fpga_address_switch.tcl").read_text(encoding="utf-8")
    build_bat = (root / "fpga" / "build_and_program.bat").read_text(encoding="utf-8")
    server_bat = (root / "fpga" / "run_server.bat").read_text(encoding="utf-8")
    launcher = (root / "pulse_gui.py").read_text(encoding="utf-8")
    preset = root / "pulses" / "camera_imaging_address_switch.json"

    assert "module zlc_pulse_streamer_top_address_switch" in top
    assert "localparam integer CHANNEL_COUNT = 62" in top
    assert "assign trig = out[6];" in top
    assert "assign trap = out[9];" in top
    assert "assign probe = out[3];" in top
    assert "assign cooling = out[0];" in top
    assert "wire [CHANNEL_COUNT-1:0] zlc_prog_mask" in top
    assert "CONFIG.C_PROBE_OUT5_WIDTH {62}" in create_tcl
    assert "set project_name address_switch" in create_tcl
    assert "zlc_pulse_streamer_top_address_switch" in create_tcl
    assert "ZLC_PS_XDC" in create_tcl
    legacy_xdc_env = "ZLC_PS_" + legacy_width.upper() + "_XDC"
    assert legacy_xdc_env not in create_tcl
    assert "CONFIG.C_PROBE_OUT5_WIDTH {62}" in check_tcl
    assert "ZLC check_address_switch_synth contract" in check_tcl
    assert "ZLC program_fpga_address_switch contract" in program_tcl
    assert "zlc_pulse_streamer_top_address_switch.bit" in program_tcl
    assert "zlc_pulse_streamer_top_address_switch.ltx" in program_tcl

    # build_and_program.bat builds the FINAL single design (JTAG-to-AXI, 1-tick
    # FIFO prefetch + streaming scan): one create_project.tcl, no variants, no VIO
    # address-switch and no discarded run-length engine.
    assert "create_project.tcl" in build_bat
    assert "program_fpga.tcl" in build_bat
    assert "zlc_verify_sources" in build_bat
    assert "zlc_force_latency2" in build_bat              # forces edge BRAM latency 2
    assert "blk_mem_gen_edge_tick" in build_bat           # 3 parallel edge BRAMs
    assert "create_project_runlength.tcl" not in build_bat
    assert "create_project_address_switch.tcl" not in build_bat
    assert "create_project_loader.tcl" not in build_bat
    assert "ZLC_PS_VARIANT" not in build_bat              # one path, no variants
    assert 'set "ZLC_PROJ_SUB=pulse_streamer"' in build_bat
    assert r'set "ZLC_PS_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\!ZLC_PROJ_SUB!"' in build_bat
    assert legacy_xdc_env not in build_bat
    # run_server.bat starts the edge-table loader JTAG-to-AXI server.
    assert "ZLC_PS_CLOCK_HZ=50000000" in server_bat
    assert "zlc_verify_loader_sources" in server_bat
    assert "ZLC_PS_SERVER_BACKEND=jtag-axi" in server_bat
    assert "zlc_pulse_streamer_loader_top.ltx" in server_bat
    assert "fpga\\build\\address_switch" not in server_bat
    assert "echo Trigger:" not in server_bat
    assert "--trigger-channels ch03" not in server_bat

    assert "DEFAULT_PULSE_GUI_FALLBACK_CHANNELS = 62" in launcher
    assert "infer_xdc_channel_count" in launcher
    assert "infer_trigger_channels" in server_bat
    assert preset.exists()

    for text in (
        root_readme,
        maintainer_notes,
        main_manual_template,
        fpga_manual_template,
        fpga_manual_tex,
        fpga_readme,
        streamer_readme,
        frontend_readme,
        pulses_readme,
        frontend_manual_template,
        fpga_notebook_text,
        hardware_notebook_text,
    ):
        assert f"camera_imaging_{legacy_width}" not in text
        assert f"zlc_pulse_streamer_top_{legacy_width}" not in text
        assert f"zlc_pulse_streamer_{legacy_width}" not in text
        assert legacy_xdc_env not in text
        assert "fpga\\build\\p40" not in text

    assert "camera_imaging_address_switch.json" in pulses_readme
    assert "ch11" in pulses_readme

    # The 50 MHz / 20 ns clock fact moved from the deleted runbook into the
    # consolidated maintainer note and is also taught in the main manual.
    assert "50 MHz" in maintainer_notes
    assert "20 ns" in maintainer_notes
    assert "50 MHz" in main_manual_template
    assert "20 ns" in main_manual_template
    # The Vivado prepare/fire/wait_done/safe_state lifecycle and the persistent
    # VIO-session contract now live in the maintainer note + main manual.
    assert "prepare" in maintainer_notes and "fire" in maintainer_notes
    assert "wait_done" in maintainer_notes and "safe_state" in maintainer_notes
    assert "VIO" in maintainer_notes
    assert "prepare / fire / wait\\_done / safe\\_state" in main_manual_template
    # The build/resource knob moved into the maintainer note + fpga README.
    assert "ZLC_PS_RESOURCE_TARGET_PCT" in maintainer_notes
    assert "ZLC_PS_RESOURCE_TARGET_PCT" in fpga_readme
    # "address-switch" was the PROJECT_OVERVIEW keyword; it now anchors the new
    # root README and the maintainer note.
    assert "address-switch" in root_readme
    assert "address-switch" in maintainer_notes
    assert "Run the smallest check" in tests_readme
    assert "Full `pytest -q` is reserved for broad handoff" in tests_readme

def _vio_probe_map(text):
    """Return ``{probe_out_index: zlc_signal_name}`` from a top-module VIO instance."""

    return {int(index): name for index, name in re.findall(r"\.probe_out(\d+)\((zlc_[A-Za-z0-9_]+)\)", text)}


def _wire_width_expr(text, signal):
    """Return the bracketed width expression of a ``wire [...] zlc_<signal>;`` declaration."""

    match = re.search(rf"wire(?: signed)? \[([^\]]+)\] zlc_{re.escape(signal)};", text)
    return None if match is None else match.group(1).strip()


def test_fpga_pulse_streamer_address_switch_vio_widths_match_python_generator():
    root = Path(__file__).resolve().parents[1]
    fpga = root / "fpga" / "pulse_streamer"
    top = (fpga / "zlc_pulse_streamer_top_address_switch.v").read_text(encoding="utf-8")

    generated_top = na.generate_pulse_streamer_top_example(
        channels=[f"ch{i:02d}" for i in range(62)],
        top_module_name="zlc_pulse_streamer_top_address_switch",
        max_edges=1024,
    )

    # The N-slot packed VIO layout has exactly 28 probe_out (0..27) and 2 probe_in,
    # and the same signal must sit on the same probe_out index in both files.
    top_map = _vio_probe_map(top)
    gen_map = _vio_probe_map(generated_top)
    expected_map = {
        0: "zlc_reset",
        1: "zlc_start",
        2: "zlc_prog_we",
        3: "zlc_prog_addr",
        4: "zlc_prog_tick",
        5: "zlc_prog_mask",
        6: "zlc_prog_count",
        7: "zlc_repeat_forever",
        8: "zlc_loop_start_addr",
        9: "zlc_loop_end_tick",
        10: "zlc_loop_count",
        11: "zlc_prog_tick_coeffs",
        12: "zlc_scan_enable",
        13: "zlc_scan_prog_we",
        14: "zlc_scan_prog_addr",
        15: "zlc_scan_prog_values",
        16: "zlc_scan_count",
        17: "zlc_loop_end_coeffs",
        18: "zlc_bus_prog_we",
        19: "zlc_bus_prog_bus",
        20: "zlc_bus_prog_addr",
        21: "zlc_bus_prog_start_tick",
        22: "zlc_bus_prog_stop_tick",
        23: "zlc_bus_prog_start_value",
        24: "zlc_bus_prog_stop_value",
        25: "zlc_bus_prog_mode",
        26: "zlc_bus_counts",
        27: "zlc_bus_prog_value_select",
        28: "zlc_bus_prog_start_tick_coeffs",
        29: "zlc_bus_prog_stop_tick_coeffs",
    }
    assert top_map == expected_map
    assert gen_map == expected_map
    assert sorted(top_map) == list(range(30))
    assert "probe_in1" in top and "probe_in1" in generated_top
    assert "probe_out30" not in top and "probe_out30" not in generated_top

    # Scalar / addr / value wires carry identical numeric widths in both files.
    width_contract = {
        "prog_addr": "9:0",
        "prog_tick": "31:0",
        "prog_count": "10:0",
        "loop_start_addr": "9:0",
        "loop_end_tick": "31:0",
        "loop_count": "31:0",
        "scan_prog_addr": "9:0",
        "scan_count": "10:0",
        "bus_prog_bus": "1:0",
        "bus_prog_addr": "5:0",
        "bus_prog_start_tick": "31:0",
        "bus_prog_stop_tick": "31:0",
        "bus_prog_start_value": "9:0",
        "bus_prog_stop_value": "9:0",
        "bus_prog_mode": "1:0",
        "bus_counts": "27:0",
        "bus_prog_value_select": "2:0",
    }
    for signal, width in width_contract.items():
        assert f"wire [{width}] zlc_{signal};" in top, signal
        assert f"wire [{width}] zlc_{signal};" in generated_top, signal

    # prog_mask is parameterized to CHANNEL_COUNT in the checked-in top.
    assert "wire [CHANNEL_COUNT-1:0] zlc_prog_mask;" in top
    assert "wire [61:0] zlc_prog_mask;" in generated_top

    # The N-slot packed buses: prog_tick_coeffs = NUM_SLOTS*COEFF_WIDTH = 4*16 = 64,
    # scan_prog_values = NUM_SLOTS*TICK_WIDTH = 4*32 = 128, loop_end_coeffs = 64.
    # The checked-in top keeps them parameterized; the generator emits the numbers.
    assert "localparam integer NUM_SLOTS = 4;" in top
    assert "wire [NUM_SLOTS*COEFF_WIDTH-1:0] zlc_prog_tick_coeffs;" in top
    assert "wire [NUM_SLOTS*COEFF_WIDTH-1:0] zlc_loop_end_coeffs;" in top
    assert "wire [NUM_SLOTS*TICK_WIDTH-1:0] zlc_scan_prog_values;" in top
    # Affine bus-segment tick coefficients (DAC+duration+delay simultaneous scan).
    assert "wire [NUM_SLOTS*COEFF_WIDTH-1:0] zlc_bus_prog_start_tick_coeffs;" in top
    assert "wire [NUM_SLOTS*COEFF_WIDTH-1:0] zlc_bus_prog_stop_tick_coeffs;" in top
    assert _wire_width_expr(generated_top, "prog_tick_coeffs") == "63:0"
    assert _wire_width_expr(generated_top, "loop_end_coeffs") == "63:0"
    assert _wire_width_expr(generated_top, "scan_prog_values") == "127:0"
    assert _wire_width_expr(generated_top, "bus_prog_start_tick_coeffs") == "63:0"
    assert _wire_width_expr(generated_top, "bus_prog_stop_tick_coeffs") == "63:0"
    assert ".NUM_SLOTS(NUM_SLOTS)" in top
    assert ".NUM_SLOTS(4)" in generated_top

    # The old per-x/y coefficient probes are gone from both files.
    for removed in (
        "prog_tick_x_coeff",
        "prog_tick_y_coeff",
        "scan_prog_x",
        "scan_prog_y",
        "loop_end_x_coeff",
        "loop_end_y_coeff",
    ):
        assert f"zlc_{removed}" not in top, removed
        assert f"zlc_{removed}" not in generated_top, removed

    assert ".EDGE_ADDR_WIDTH(10)" in top
    assert ".EDGE_ADDR_WIDTH(10)" in generated_top
    assert ".SCAN_ADDR_WIDTH(10)" in top
    assert ".SCAN_ADDR_WIDTH(10)" in generated_top
    assert ".EDGE_ADDR_WIDTH(7)" not in top
    assert ".EDGE_ADDR_WIDTH(7)" not in generated_top


def test_repo_batch_files_use_crlf_line_endings():
    """Windows .bat files MUST be CRLF in the working tree.  With bare-LF endings
    cmd.exe intermittently fails to seek :labels ("The system cannot find the batch
    label specified ...") -- the failure depends on where labels fall relative to
    cmd's internal read buffer, so it can pass once and break after an unrelated
    edit shifts byte offsets.  .gitattributes forces *.bat eol=crlf; guard the
    actual working-tree bytes so a stray LF rewrite can't silently reintroduce it."""

    root = Path(__file__).resolve().parents[1]
    # Police only our own entrypoints, not archived/third-party material under
    # references/ or generated build artifacts.
    skip_parts = {
        ".git", "build", "__pycache__", "node_modules", ".venv", "site-packages",
        "references",
    }
    bats = [
        p for p in root.rglob("*.bat")
        if not (skip_parts & set(p.relative_to(root).parts))
    ]
    assert bats, "expected .bat entrypoints in the repo"
    offenders = []
    for bat in bats:
        data = bat.read_bytes()
        if any(b == 0x0A and (i == 0 or data[i - 1] != 0x0D) for i, b in enumerate(data)):
            offenders.append(bat.relative_to(root).as_posix())
    assert not offenders, (
        "these .bat files have bare-LF line endings (cmd.exe label seek will fail): "
        f"{offenders}"
    )


def test_repo_gitattributes_forces_crlf_for_batch_files():
    root = Path(__file__).resolve().parents[1]
    attrs = (root / ".gitattributes").read_text(encoding="utf-8")
    assert "*.bat text eol=crlf" in attrs


def test_fpga_pulse_streamer_capacity_doc_matches_checked_in_ram_strategy():
    root = Path(__file__).resolve().parents[1]
    fpga = root / "fpga" / "pulse_streamer"
    core = (fpga / "zlc_pulse_streamer.v").read_text(encoding="utf-8")
    top = (fpga / "zlc_pulse_streamer_top_address_switch.v").read_text(encoding="utf-8")
    # The standalone capacity doc was folded into the consolidated maintainer
    # note and the FPGA manual template during the docs reorg.
    notes = (root / "docs" / "MAINTAINER_NOTES.md").read_text(encoding="utf-8")
    fpga_manual = (
        root
        / "Zou_lab_control"
        / "neutral_atom"
        / "content"
        / "manual_templates"
        / "fpga_manual_zh.texbody"
    ).read_text(encoding="utf-8")
    streamer_readme = (fpga / "README.md").read_text(encoding="utf-8")

    # The checked-in RTL marks every big runtime table as distributed RAM
    # (async single-row reads). The new N-slot packed memories are
    # coeff_mem (NUM_SLOTS*COEFF_WIDTH wide) and scan_value_mem
    # (NUM_SLOTS*TICK_WIDTH wide); tick_mem and mask_mem carry the per-edge
    # absolute tick and full output mask.
    assert '(* ram_style = "distributed" *)' in core
    for mem in ("tick_mem", "coeff_mem", "mask_mem", "scan_value_mem"):
        assert re.search(
            rf'\(\* ram_style = "distributed" \*\) reg \[[^\]]+\] {mem} ',
            core,
        ), mem
    # Packed widths must be the N-slot ones, not the old per-x/y coefficients.
    assert "reg [COEFF_BITS-1:0] coeff_mem" in core
    assert "reg [SLOT_BITS-1:0] scan_value_mem" in core
    assert "localparam integer COEFF_BITS = NUM_SLOTS * COEFF_WIDTH;" in core
    assert "localparam integer SLOT_BITS = NUM_SLOTS * TICK_WIDTH;" in core

    assert "localparam integer CHANNEL_COUNT = 62" in top
    assert ".CHANNEL_COUNT(CHANNEL_COUNT)" in top
    assert ".EDGE_ADDR_WIDTH(10)" in top

    # The documented RAM strategy (now in the maintainer note + FPGA manual)
    # must match that RTL: distributed RAM today, with a BRAM-friendly
    # synchronous-read pipeline + faster transport as the documented scale path.
    assert 'ram_style="distributed"' in notes
    assert "tick_mem" in notes and "mask_mem" in notes
    assert "coeff_mem" in notes and "scan_value_mem" in notes
    assert "CHANNEL_COUNT=62" in notes
    # The documented scale path is a BRAM-friendly synchronous-read pipeline plus
    # a faster transport.  The run-length engine locks that transport to
    # JTAG-to-AXI (was an open list while the design was still deferred).
    assert "BRAM-friendly synchronous-read" in notes
    assert "pipeline" in notes
    assert "JTAG-to-AXI" in notes

    assert 'ram_style = "distributed"' in fpga_manual
    assert "BRAM" in fpga_manual
    assert "AXI" in fpga_manual and "JTAG-to-AXI" in fpga_manual

    # The trimmed subsystem README still documents the VIO probe map; the new
    # N-slot layout puts scan_count on probe_out16 (probe_out18 is bus_prog_we).
    assert "One edge row means" in streamer_readme
    assert "MAX_EDGES=1024" in streamer_readme
    assert "probe_out3  zlc_prog_addr" in streamer_readme
    assert "probe_out6  zlc_prog_count" in streamer_readme
    assert "probe_out16 zlc_scan_count" in streamer_readme


def _user_facing_markdown_files(root):
    """Discover the current user-facing markdown set.

    The docs were reorganized into a single root README, one consolidated
    maintainer note, and trimmed subsystem pointers. Rather than hard-code a
    stale list, glob for ``*.md`` and drop the non-user-facing ones: historical
    source archives, tool caches, and the notebook-cell source templates
    (``*.cells.md``) that are not standalone documents.
    """

    skip_parts = {"references", ".git", ".pytest_cache", "__pycache__", "build"}
    markdown_files = []
    for path in sorted(root.rglob("*.md")):
        if any(part in skip_parts for part in path.relative_to(root).parts):
            continue
        if path.name.endswith(".cells.md"):
            continue
        markdown_files.append(path)
    return markdown_files


def test_user_facing_markdown_local_links_exist():
    root = Path(__file__).resolve().parents[1]
    markdown_files = _user_facing_markdown_files(root)

    # The reorganized layout: a single root entry point, the consolidated
    # maintainer note, and the trimmed subsystem pointers. Guard that the new
    # canonical docs are in the discovered set so the test cannot silently pass
    # on an empty list, and that the deleted docs stay deleted.
    discovered = {path.relative_to(root).as_posix() for path in markdown_files}
    expected_present = {
        "README.md",
        "docs/MAINTAINER_NOTES.md",
        "fpga/README.md",
        "fpga/pulse_streamer/README.md",
        "pulses/README.md",
        "Zou_lab_control/frontend/README.md",
        "tests/README.md",
    }
    assert expected_present.issubset(discovered), expected_present - discovered
    for deleted in (
        "AGENTS.md",
        "docs/PROJECT_OVERVIEW.md",
        "docs/DOCUMENTATION_GUIDE.md",
        "docs/FPGA_PULSE_STREAMER_CAPACITY.md",
        "docs/FRONTEND_FLUENT_STYLE_GUIDE.md",
        "docs/neutral_atom_hardware_manual/REAL_HARDWARE_RUNBOOK.md",
    ):
        assert deleted not in discovered, deleted

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
    import os
    import subprocess

    root = Path(__file__).resolve().parents[1]
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

    def git_ignored(rel_posix: str) -> bool:
        try:
            return subprocess.run(
                ["git", "-C", str(root), "check-ignore", "-q", "--", rel_posix]
            ).returncode == 0
        except Exception:  # pragma: no cover - git not available
            return False

    # Walk with pruning: a git-ignored subtree (e.g. fpga/build from a local Vivado
    # run, or references/) is never committed, so it cannot pollute the repo -- skip
    # it entirely.  Only NON-ignored generated artifacts are real problems.
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(str(root)):
        relative = Path(dirpath).relative_to(root)
        dirnames[:] = [
            d for d in dirnames if d != ".git" and not git_ignored((relative / d).as_posix())
        ]
        for d in dirnames:
            if d in bad_directories:
                found.append((relative / d).as_posix())
        for name in filenames:
            child = (relative / name).as_posix()
            if any(name.endswith(suffix) for suffix in bad_suffixes) and not git_ignored(child):
                found.append(child)
    assert found == [], found


def test_repo_bat_entrypoints_are_minimal_and_grouped_by_submodule():
    root = Path(__file__).resolve().parents[1]
    # "build" excludes the git-ignored Vivado project output (fpga/build/r/...),
    # which contains IP-generated runme.bat files that are not entrypoints.
    ignored_roots = {".git", "references", "reference", "build"}
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
                "set_property PACKAGE_PIN A3 [get_ports {ch[2]}] ;# ch02 <- trig / trig",
            ]
        ),
        encoding="utf-8",
    )
    assert na.infer_xdc_channel_labels(labeled, default=40, max_count=40) == {
        "ch00": "trap",
        "ch01": "cooling_pgc",
        "ch02": "trig",
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


def test_address_switch_xdc_infers_62_outputs_trigger_and_bus_channels():
    root = Path(__file__).resolve().parents[1]
    xdc = root / "references" / "source_archives" / "address_switch" / "address_switch.srcs" / "constrs_1" / "new" / "addre.xdc"
    count = na.infer_xdc_channel_count(xdc, default=1, max_count=None)
    labels = na.infer_xdc_channel_labels(xdc, default=count, max_count=None)
    channels = [f"ch{index:02d}" for index in range(count)]
    buses = na.infer_bus_channels(channels, labels)

    assert count == 62
    assert labels["ch03"] == "probe"
    assert labels["ch06"] == "trig"
    assert labels["ch11"] == "emCCD"
    assert labels["ch09"] == "trap"
    assert na.infer_xdc_trigger_channels(xdc, default=count, max_count=None) == ["ch11"]
    assert buses["da_dipole"] == [f"ch{index:02d}" for index in range(18, 28)]
    assert buses["da_bias_x"] == [f"ch{index:02d}" for index in range(40, 50)]
    assert buses["da_bias_y"] == [f"ch{index:02d}" for index in range(38, 28, -1)]
    assert buses["da_bias_z"] == [f"ch{index:02d}" for index in range(60, 50, -1)]


def test_fpga_pulse_streamer_prepare_tcl_covers_full_edge_table_boundary(tmp_path):
    program = na.RuntimeSequenceProgram(
        sequence_id="full",
        sequence_name="full_table",
        clock_hz=100e6,
        channels=["trap", "cooling", "probe", "trig"],
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
    seq = na.PulseSequence(name="contract").pulse("trap", 0.0, 5e-8).pulse("probe", 2e-8, 8e-8).pulse("trig", 2e-8, 4e-8)
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "trig"], clock_hz=100e6)
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
    seq = na.PulseSequence(name="delay_repeat").pulse("trap", 0.0, 2e-8).pulse("emCCD", 0.0, 1e-8).delay("emCCD", 1e-8)
    repeated = seq.repeated(2, period=5e-8)
    program = na.compile_runtime_program(repeated, channels=["trap", "cooling", "probe", "emCCD"], clock_hz=100e6)

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
        channels=["trap", "cooling", "probe", "trig", "aod0", "aod1"],
        periods=[
            na.PulsePeriod(100, (1, 0, 0, 0, 0, 0), unit="ns", name="load"),
            na.PulsePeriod("2*s0", (1, 0, 1, 1, 0, 0), unit="str (ns)", name="image"),
            na.PulsePeriod(100, (0, 0, 0, 0, 0, 0), unit="ns", name="idle"),
        ],
        delays={"trig": "s0/2"},
        delay_units={"trig": "str (ns)"},
        scan_slots=[{"kind": "duration", "target": "1", "unit": "ns", "nominal": 100.0}],
        scan_table=[[100.0]],
        time_step_ns=1,
        repeat_start=1,
        repeat_end=2,
        repeat_count=3,
    )

    sequence = state.to_sequence()
    program = state.compile(clock_hz=100e6, trigger_channels=["trig"])
    saved = state.save(tmp_path / "pulse.json")
    loaded = na.PulseTableState.load(saved)

    s0_100 = {"s0": 100.0}
    s0_200 = {"s0": 200.0}

    assert state.time_step_ns == 1
    assert state.scan_var_names == ["s0"]
    assert state.reference_slots() == s0_100
    assert state.total_duration_steps() == 1000
    assert state.total_duration_steps(slots=s0_200, time_step_ns=10) == 160
    assert state.total_duration_ns() == 100 + 3 * (200 + 100)
    assert state.total_duration_ns(slots=s0_200) == 100 + 3 * (400 + 100)
    assert state.periods[1].duration_steps(slots=s0_200, time_step_ns=state.time_step_ns) == 400
    assert state.delay_ns("trig", slots=s0_200) == 100
    assert state.delay_steps("trig", slots=s0_200, time_step_ns=10) == 10
    resolved = state.with_slots_resolved(s0_200)
    assert resolved.scan_slots == []
    assert resolved.scan_table == []
    assert na.count_trigger_pulses(sequence, trigger_channels=["trig"]) == 3
    assert program.trigger_count == 3
    assert program.repeat_forever is True
    assert program.ticks == [0, 10, 15, 30, 35, 40]
    assert program.masks == [0b000001, 0b000101, 0b001101, 0b001000, 0, 0]
    assert program.loop_start_index == 1
    assert program.loop_end_tick == 40
    assert program.loop_count == 3
    program_x = state.compile(clock_hz=100e6, trigger_channels=["trig"], slots=s0_200)
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
        channel_labels={"ch00": "trap", "ch03": "trig"},
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


def test_pulse_table_analog_bus_modes_compile_to_runtime_bus_segments(tmp_path):
    channels = ["ch00", "ch01", "ch02", "ch03"]
    labels = {"ch00": "da_test[0]", "ch01": "da_test[1]", "ch02": "da_test[2]", "ch03": "trig"}
    state = na.PulseTableState(
        channels=channels,
        channel_labels=labels,
        visible_channels=channels,
        time_step_ns=20,
        periods=[
            na.PulsePeriod(100, (0, 0, 0, 0), unit="ns"),
            na.PulsePeriod(100, (0, 0, 0, 0), unit="ns"),
            na.PulsePeriod(100, (0, 0, 0, 0), unit="ns"),
        ],
    )
    state.set_analog_bus_mode(0, "da_test", "edge", value=0)
    state.set_analog_bus_mode(2, "da_test", "ramp", value=7)
    state.apply_analog_bus_modes_to_period_states()
    saved = state.save(tmp_path / "analog_bus.json")
    loaded = na.PulseTableState.load(saved)
    program = loaded.compile(clock_hz=50_000_000, trigger_channels=["ch03"], repeat_forever=False)

    assert loaded.bus_channels()["da_test"] == ["ch00", "ch01", "ch02"]
    assert loaded.analog_bus_modes["da_test"] == [
        {"mode": "edge", "value": 0},
        {"mode": "hold", "value": None},
        {"mode": "ramp", "value": 7},
    ]
    assert loaded.periods[1].states[:3] == (0, 0, 1)
    assert program.ticks == [0, 5, 10, 15]
    assert program.masks == [0, 0, 0, 0]
    assert program.bus_names == ["da_test"]
    assert [segment.to_dict() for segment in (program.bus_segments or [])] == [
        {
            "bus_index": 0,
            "bus_name": "da_test",
            "start_tick": 0,
            "stop_tick": 10,
            "start_value": 0,
            "stop_value": 7,
            "mode": "ramp",
            "value_select": 0,
            "start_tick_coeffs": [],
            "stop_tick_coeffs": [],
        }
    ]
    roundtrip = na.RuntimeSequenceProgram.from_dict(program.to_dict())
    assert [segment.to_dict() for segment in (roundtrip.bus_segments or [])] == [
        segment.to_dict() for segment in (program.bus_segments or [])
    ]
    na.validate_pulse_streamer_program(program, max_edges=16, max_bus_segments=4, tick_width=32, channel_count=4)


def test_pulse_table_scan_rejects_analog_bus_ramp_mode():
    state = na.PulseTableState(
        channels=["ch00", "ch01"],
        channel_labels={"ch00": "da_test[0]", "ch01": "da_test[1]"},
        visible_channels=["ch00", "ch01"],
        time_step_ns=20,
        periods=[
            na.PulsePeriod(100, (0, 0), unit="ns"),
            na.PulsePeriod(100, (0, 0), unit="ns"),
        ],
    )
    # Scan the duration of the first period (a time slot) ...
    state.bind_field("duration", "0")
    state.set_scan_table([[20.0], [40.0]])
    # ... while the analog bus ramps: the seamless hardware scan path cannot do both.
    state.set_analog_bus_mode(0, "da_test", "edge", value=0)
    state.set_analog_bus_mode(1, "da_test", "ramp", value=3)

    with pytest.raises(ValueError, match="scan array cannot currently combine"):
        na.compile_pulse_table_scan_runtime_program(state, channels=["ch00", "ch01"], clock_hz=50_000_000)


def _v2_loop_steps_resolved(eff_ticks, masks, loop_start_index, eff_loop_end, loop_count, repeat_forever, steps):
    """v2 loop FSM run on already-resolved (effective) ticks -- the ground truth
    for a loop at one scan point."""

    ticks = list(eff_ticks)
    masks = list(masks)
    final = ticks[-1]
    loop_start_tick = ticks[loop_start_index]
    loop_start_mask = masks[loop_start_index]
    loop_count = max(1, int(loop_count))
    loops_remaining = loop_count
    if ticks[0] == 0:
        state_mask, time_count, edge_index = masks[0], 1, 1
    else:
        state_mask, time_count, edge_index = 0, 0, 0
    history = [state_mask]
    while len(history) < steps:
        if loop_count > 1 and loops_remaining > 1 and time_count >= eff_loop_end:
            state_mask, time_count, edge_index = loop_start_mask, loop_start_tick + 1, loop_start_index + 1
            loops_remaining -= 1
        elif time_count >= final:
            if repeat_forever:
                state_mask, time_count, edge_index = (masks[0], 1, 1) if ticks[0] == 0 else (0, 0, 0)
                loops_remaining = loop_count
            else:
                state_mask = 0
        else:
            if edge_index < len(ticks) and time_count == ticks[edge_index]:
                state_mask = masks[edge_index]
                edge_index += 1
            time_count += 1
        history.append(state_mask)
    return history


def test_program_bram_depth_fits_device_and_matches_build():
    """The program BRAM must (a) hold the whole edge-table image, (b) be a power of
    two that fits the 35T's 50 RAMB36, and (c) match the depth + address widths wired
    in the loader build Tcl and loader top.  Guards the BRAM sizing against drift."""

    import math
    from Zou_lab_control.neutral_atom.devices.edgetable_image import EdgeTableImageParams

    p = EdgeTableImageParams()
    depth = 32768
    assert depth & (depth - 1) == 0, "BRAM depth must be a power of two for axi_bram_ctrl"
    assert depth >= p.total_words, "BRAM too small to hold the edge-table image"
    assert math.ceil(depth / 1024) <= 50, "BRAM exceeds the 35T's 50 RAMB36"

    root = Path(__file__).resolve().parents[1]
    tcl = (root / "fpga" / "pulse_streamer" / "create_project_loader.tcl").read_text(encoding="utf-8")
    # Depth is set strictly (not via failure-tolerant zlc_try) with a read-back guard.
    assert f"set zlc_bram_depth {depth}" in tcl
    assert "CONFIG.MEM_DEPTH $zlc_bram_depth" in tcl
    assert "CONFIG.Write_Depth_A $zlc_bram_depth" in tcl
    assert "MEM_DEPTH did not take" in tcl
    assert "Write_Depth_A reverted" in tcl

    top = (root / "fpga" / "pulse_streamer" / "zlc_pulse_streamer_loader_top.v").read_text(encoding="utf-8")
    # axi_bram_ctrl bram_addr_a is byte-addressed (17 bits); the loader drives BRAM
    # port B by WORD address (15 bits for 32768 words).
    assert "wire [16:0] bram_addra" in top
    assert "addra(bram_addra[16:2])" in top
    assert "wire [14:0] ldr_addr" in top
    assert ".addrb(ldr_addr)" in top
def test_edgetable_image_packs_and_loader_walk_reconstructs_program():
    """The edge-table BRAM image (loader path) must round-trip a full program:
    pack(program) -> loader-walk unpack == program.  This proves the on-chip
    loader delivers the SAME edge/scan/bus/loop data the validated VIO upload
    delivered, so the unchanged seamless engine stays tick-exact + gapless."""

    from Zou_lab_control.neutral_atom.devices.sequencer import (
        RuntimeSequenceProgram,
        RuntimeBusSegment,
    )
    from Zou_lab_control.neutral_atom.devices.edgetable_image import (
        pack_program,
        unpack_program,
        EdgeTableImageParams,
    )

    prog = RuntimeSequenceProgram(
        sequence_id="abc", sequence_name="t", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 50, 120, 400],
        masks=[0, (1 << 0) | (1 << 5), (1 << 61), 0],  # exercise the high mask bit
        duration=8e-6, trigger_count=0,
        repeat_forever=True, loop_start_index=1, loop_end_tick=400, loop_count=3,
        slot_count=2, slot_kinds=["delay", "dac"],
        loop_end_slot_coeffs=[256, 0],
        tick_slot_coeffs=[[0, 0], [256, 0], [256, 0], [256, 0]],
        scan_points=[[0, 0], [256, 256], [512, -768]],  # negative slot value too
        scan_coeff_frac_bits=8,
        bus_names=["da0"],
        bus_segments=[
            RuntimeBusSegment(bus_index=0, start_tick=50, stop_tick=120, start_value=0,
                              stop_value=0, mode="edge", value_select=2,
                              start_tick_coeffs=[256, 0], stop_tick_coeffs=[256, 0]),
            RuntimeBusSegment(bus_index=2, start_tick=10, stop_tick=400, start_value=512,
                              stop_value=1023, mode="ramp", value_select=0),
        ],
    )
    p = EdgeTableImageParams()
    image = pack_program(prog, p)
    out = unpack_program(image, p)

    def pad(row, n):
        return list(row) + [0] * (n - len(row))

    assert out["ticks"] == prog.ticks
    assert out["masks"] == prog.masks
    assert out["tick_slot_coeffs"] == [pad(r, p.num_slots) for r in prog.tick_slot_coeffs]
    assert out["scan_points"] == prog.scan_points
    assert out["slot_count"] == 2
    assert out["repeat_forever"] is True
    assert out["loop_start_index"] == 1
    assert out["loop_count"] == 3
    assert out["loop_end_tick"] == 400
    assert out["loop_end_slot_coeffs"] == pad([256, 0], p.num_slots)

    bus = {s["bus_index"]: s for s in out["bus_segments"]}
    assert len(out["bus_segments"]) == 2
    assert bus[0]["start_tick"] == 50 and bus[0]["stop_tick"] == 120
    assert bus[0]["value_select"] == 2 and bus[0]["mode"] == "edge"
    assert bus[0]["start_tick_coeffs"] == pad([256, 0], p.num_slots)
    assert bus[2]["start_value"] == 512 and bus[2]["stop_value"] == 1023
    assert bus[2]["mode"] == "ramp" and bus[2]["value_select"] == 0

    # The whole image must fit a 32-RAMB36 (32768-word) program BRAM.
    assert p.total_words <= 32768
    # Only used rows are emitted -> the host uploads a small image.
    assert len(image) < 200


def test_edgetable_loader_top_and_build_wire_engine_and_ips():
    """The loader top must instantiate the validated edge-table engine + the loader
    + the three JTAG-to-AXI IPs, keep the exact 62-pin board map, and route the
    engine's DAC bus_out to the pins (no longer tied to 0).  The build Tcl must read
    all three RTL files, set the loader top, and size the BRAM strictly."""

    root = Path(__file__).resolve().parents[1]
    fpga = root / "fpga" / "pulse_streamer"
    top = (fpga / "zlc_pulse_streamer_loader_top.v").read_text(encoding="utf-8")

    assert "zlc_axi_program_loader #(" in top
    assert "zlc_pulse_streamer #(" in top
    assert "jtag_axi_0 zlc_jtag_axi_i" in top
    assert "axi_bram_ctrl_0 zlc_bram_ctrl_i" in top
    assert "blk_mem_gen_0 zlc_prog_bram_i" in top
    # port B is driven by the loader (word address), port A by AXI
    assert ".addrb(ldr_addr)" in top and ".web({4{ldr_we}})" in top
    assert "addra(bram_addra[16:2])" in top
    # exact board pin map preserved + DAC buses now driven by the engine
    assert "assign da_clk3 = out[61];" in top
    assert "assign da_dipole[0] = zlc_bus_out[0];" in top
    assert ".bus_out(zlc_bus_out)" in top
    assert "zlc_bus_out = 40'b0" not in top  # DACs are real now, not tied off

    tcl = (fpga / "create_project_loader.tcl").read_text(encoding="utf-8")
    assert "zlc_pulse_streamer.v" in tcl
    assert "zlc_axi_program_loader.v" in tcl
    assert "zlc_pulse_streamer_loader_top.v" in tcl
    assert "set top zlc_pulse_streamer_loader_top" in tcl
    assert "set zlc_bram_depth 32768" in tcl
    assert "MEM_DEPTH did not take" in tcl  # strict depth guard preserved

    prog = (fpga / "program_fpga_loader.tcl").read_text(encoding="utf-8")
    assert "zlc_pulse_streamer_loader_top" in prog
    assert "get_hw_axis" in prog


def test_final_image_solver_90pct_and_packs_round_trip():
    """The FINAL BRAM image (fpga/pulse_streamer/host/image.py): solve_capacity at
    <=90% maximises edges (35T -> 4096 edges + a 2-bank resident scan window) with
    scan points UNBOUNDED via streaming; pack/unpack round-trips a full program
    incl. a 0..1023 DAC ramp (one segment) + loop + scan resident."""

    from fpga.pulse_streamer.host.image import solve_capacity, pack_program, unpack_program, scan_bank_words
    from Zou_lab_control.neutral_atom.devices.sequencer import RuntimeSequenceProgram, RuntimeBusSegment

    s = solve_capacity("xc7a35t", channel_count=62, target_pct=90.0)
    assert s.params.max_edges >= 4096
    assert s.params.bank_size >= 512
    assert s.all_within_budget() and s.resource_report["ramb36"]["pct"] <= 90.0
    big = solve_capacity("xc7a200t", channel_count=62)
    assert big.params.max_edges >= s.params.max_edges

    p = s.params
    prog = RuntimeSequenceProgram(
        sequence_id="a", sequence_name="t", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 5, 25, 400], masks=[0, (1 << 0) | (1 << 5), (1 << 61), 0],
        duration=8e-6, trigger_count=0, repeat_forever=True, loop_start_index=1,
        loop_end_tick=400, loop_count=3, slot_count=2, slot_kinds=["delay", "dac"],
        loop_end_slot_coeffs=[256, 0], tick_slot_coeffs=[[0, 0], [256, 0], [256, 0], [256, 0]],
        scan_points=[[k, k * 2] for k in range(5)], scan_coeff_frac_bits=8, bus_names=["da0"],
        bus_segments=[RuntimeBusSegment(bus_index=0, start_tick=5, stop_tick=25, start_value=0,
                                        stop_value=1023, mode="ramp", value_select=2,
                                        start_tick_coeffs=[256, 0], stop_tick_coeffs=[256, 0])],
    )
    out = unpack_program(pack_program(prog, p), p)
    pad = lambda r, n: list(r) + [0] * (n - len(r))
    assert out["ticks"] == prog.ticks and out["masks"] == prog.masks
    assert out["tick_slot_coeffs"] == [pad(r, p.num_slots) for r in prog.tick_slot_coeffs]
    assert out["scan_points_resident"] == [list(pt) for pt in prog.scan_points]
    assert out["scan_count"] == 5 and out["loop_count"] == 3 and out["repeat_forever"]
    b = out["bus_segments"][0]
    assert b["mode"] == "ramp" and b["stop_value"] == 1023 and b["value_select"] == 2
    # a streamed chunk (beyond the resident window) packs into the right bank.
    assert scan_bank_words(prog, p, 0)  # chunk 0 non-empty


def test_final_engine_model_fifo_1tick_and_streaming_scan():
    """The FINAL engine model (fpga/pulse_streamer/host/engine_model.py): the edge
    FIFO prefetch is tick-exact at 1-tick spacing (latency 1 AND 2), and the scan
    ping-pong streaming reproduces the full N-point sweep gaplessly for any bank
    size while the host keeps up, and STALLS (never a wrong point) when starved."""

    from fpga.pulse_streamer.host.engine_model import (
        EngineProgram, reference_play, prefetch_play, streaming_scan_play, ScanUnderflow,
    )

    def prog(**kw):
        b = dict(ticks=[], masks=[], tick_slot_coeffs=[], scan_points=[], slot_count=0,
                 frac_bits=8, loop_start_index=0, loop_end_tick=0, loop_end_slot_coeffs=[],
                 loop_count=1, repeat_forever=False)
        b.update(kw)
        b["tick_slot_coeffs"] = b["tick_slot_coeffs"] or [[0] * b["slot_count"] for _ in b["ticks"]]
        b["loop_end_slot_coeffs"] = b["loop_end_slot_coeffs"] or [0] * b["slot_count"]
        return EngineProgram(**b)

    # --- FIFO prefetch (1-tick) == combinatorial reference, latency 1 and 2 ---
    fifo_cases = {
        "b2b_1tick": prog(ticks=[0, 1, 2, 3, 4, 20], masks=[0, 1, 2, 3, 4, 0], loop_end_tick=20, repeat_forever=True),
        "scan": prog(ticks=[0, 10, 20, 100], masks=[0, 1, 2, 0], tick_slot_coeffs=[[0], [256], [256], [256]],
                     scan_points=[[0], [256], [512]], slot_count=1, loop_end_tick=100, repeat_forever=True),
        "loop3": prog(ticks=[0, 10, 30, 60], masks=[0, 1, 2, 0], loop_start_index=1, loop_end_tick=30, loop_count=3),
    }
    for name, pr in fifo_cases.items():
        ref = reference_play(pr, 400)
        for lat in (1, 2):
            assert prefetch_play(pr, 400, read_latency=lat, fifo_depth=lat + 1) == ref, (name, lat)

    # --- streaming: 20 points, constant duration, waveform differs per point ---
    N = 20
    sp = prog(ticks=[0, 5, 25, 40], masks=[0, 1, 2, 0], tick_slot_coeffs=[[0], [256], [0], [0]],
              scan_points=[[k] for k in range(N)], slot_count=1, loop_end_tick=40, repeat_forever=False)
    NT = N * 41 + 60
    ref = reference_play(sp, NT)
    for bank_size in (1, 4, 5, 8):
        out, stalled, played = streaming_scan_play(sp, NT, bank_size=bank_size, refill_delay=0)
        assert out == ref and not stalled and played == N, (bank_size, played)

    # --- starved refill: stalls (no wrong point); the un-starved prefix matches ref ---
    out2, stalled2, _ = streaming_scan_play(sp, NT, bank_size=4, refill_delay=10 ** 9)
    assert stalled2
    first_diff = next((i for i in range(NT) if out2[i] != ref[i]), NT)
    assert first_diff >= 4 * 41  # gapless through the first two resident banks (8 points)
    with pytest.raises(ScanUnderflow):
        streaming_scan_play(sp, NT, bank_size=4, refill_delay=10 ** 9, raise_on_underflow=True)


def test_edge_streamer_rtl_mirror_matches_reference():
    """`rtl_mirror_play` re-implements the EXACT register transfers of
    fpga/pulse_streamer/zlc_edge_streamer.v (arm shift-down FIFO + nv + pend
    in-flight shift + the issue-occupancy condition + FIFO_DEPTH-shadow boundary
    reseeds).  It must equal the combinatorial reference for every program shape
    -- 1-tick spacing included -- at read latency 1, 2 AND 3.  This is the no-sim
    proof that the SPECIFIC RTL realisation (not just the abstract algorithm) is
    tick-exact and 1-tick gapless across start/loop/scan/repeat boundaries."""

    import random
    from fpga.pulse_streamer.host.engine_model import (
        EngineProgram, reference_play, rtl_mirror_play,
    )

    def prog(**kw):
        b = dict(ticks=[], masks=[], tick_slot_coeffs=[], scan_points=[], slot_count=0,
                 frac_bits=8, loop_start_index=0, loop_end_tick=0, loop_end_slot_coeffs=[],
                 loop_count=1, repeat_forever=False)
        b.update(kw)
        b["tick_slot_coeffs"] = b["tick_slot_coeffs"] or [[0] * b["slot_count"] for _ in b["ticks"]]
        b["loop_end_slot_coeffs"] = b["loop_end_slot_coeffs"] or [0] * b["slot_count"]
        return EngineProgram(**b)

    cases = {
        "b2b_1tick": prog(ticks=[0, 1, 2, 3, 4, 20], masks=[0, 1, 2, 3, 4, 0], loop_end_tick=20, repeat_forever=True),
        "b2b_nonzero": prog(ticks=[5, 6, 7, 8, 9, 30], masks=[1, 2, 3, 4, 5, 0], loop_end_tick=30, repeat_forever=True),
        "sparse": prog(ticks=[0, 5, 12, 40], masks=[0, 3, 4, 0], loop_end_tick=40, repeat_forever=True),
        "scan_1tick": prog(ticks=[0, 1, 2, 3, 40], masks=[0, 1, 2, 3, 0],
                           tick_slot_coeffs=[[0], [256], [256], [256], [256]],
                           scan_points=[[0], [256], [512], [768]], slot_count=1, loop_end_tick=40, repeat_forever=True),
        "loop_1tick": prog(ticks=[0, 1, 2, 3, 4, 5, 6, 40], masks=[0, 1, 2, 3, 4, 5, 6, 0],
                           loop_start_index=2, loop_end_tick=4, loop_count=4),
        "single": prog(ticks=[0], masks=[5], loop_end_tick=10, repeat_forever=True),
        "finite": prog(ticks=[0, 5, 20], masks=[0, 1, 0], loop_end_tick=20, loop_count=1),
    }
    N = 600
    for name, pr in cases.items():
        ref = reference_play(pr, N)
        for lat in (1, 2, 3):
            assert rtl_mirror_play(pr, N, rd_lat=lat, fifo_depth=lat + 1) == ref, (name, lat)

    # fuzz 1-tick-heavy random programs (the spacing stress the prefetch must survive)
    rnd = random.Random(7)
    for _ in range(200):
        ticks = sorted({rnd.randint(0, 12) for _ in range(rnd.randint(1, 8))})
        ticks = [0] + [t for t in ticks if t > 0]
        masks = [rnd.randint(0, 7) for _ in ticks]
        fin = ticks[-1] + rnd.randint(1, 6)
        pr = prog(ticks=ticks + [fin], masks=masks + [0], loop_end_tick=fin,
                  repeat_forever=(rnd.random() < 0.5),
                  loop_start_index=rnd.randint(0, max(0, len(ticks) - 1)),
                  loop_count=rnd.choice([1, 1, 3]))
        ref = reference_play(pr, N)
        for lat in (1, 2):
            assert rtl_mirror_play(pr, N, rd_lat=lat, fifo_depth=lat + 1) == ref, (ticks, masks, lat)


def test_edge_streamer_rtl_has_proven_structure():
    """Lock the final RTL engine to the proven design so it cannot silently drift
    from rtl_mirror_play: FIFO_DEPTH(=RD_LAT+1) shadow seed, parallel tick/coeff/
    mask edge read, 2-bank streaming with bank_ready stall + cursor, bus LUTRAM,
    and no leftover WIP/do-not-build marker."""

    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "fpga" / "pulse_streamer" / "zlc_edge_streamer.v"
    text = src.read_text(encoding="utf-8")
    # one clean module, no abandoned draft marker
    assert "module zlc_edge_streamer" in text and text.count("endmodule") == 1
    assert "WIP" not in text and "do-not-build" not in text.lower()
    # depth-(latency+1) prefetch FIFO + the issue-occupancy guard
    assert "RD_LAT" in text and "FIFO_DEPTH" in text
    assert "pend <= {pend[RD_LAT-2:0], issue}" in text
    assert "nv_after_fire" in text and "clamp3" in text
    # 8 boundary shadows (e0..e3 + ls0..ls3) -> FIFO_DEPTH-shadow seed
    for sh in ("sh_e0_t", "sh_e1_t", "sh_e2_t", "sh_e3_t", "sh_ls0_t", "sh_ls1_t", "sh_ls2_t", "sh_ls3_t"):
        assert sh in text, sh
    assert "seed_from_edge0" in text
    # 3 PARALLEL edge BRAMs read in lockstep (whole edge per access)
    for sig in ("edge_tick_rdata", "edge_coeff_rdata", "edge_mask_rdata"):
        assert sig in text, sig
    # 2-bank ping-pong streaming + handshake (unbounded scan points)
    assert "bank_ready" in text and "scan_cursor" in text and "underflow" in text
    assert "scan_addr_of" in text and "bank_of" in text
    # bus tables stay LUTRAM (combinational per-tick read)
    assert text.count('ram_style = "distributed"') >= 7


def test_final_top_regions_match_image_and_has_structure():
    """The final top zlc_pulse_streamer_top.v decodes the SAME word-address
    regions the host packs (host.image.region_bases), instantiates the FINAL
    engine with 3 parallel edge BRAMs + the streaming handshake, and exposes the
    cursor/bank_ready ports.  Locks top <-> host so they cannot drift."""

    import pathlib, re, dataclasses
    from fpga.pulse_streamer.host import image as im

    src = pathlib.Path(__file__).resolve().parents[1] / "fpga" / "pulse_streamer" / "zlc_pulse_streamer_top.v"
    text = src.read_text(encoding="utf-8")

    # solved build geometry (the create-project tcl uses the same source)
    p = dataclasses.replace(im.StreamerParams(), bank_size=2048, max_edges=4096)
    rb = im.region_bases(p)
    me = 4096
    top = {
        "tick": 64,
        "coeff": 64 + me,
        "mask": 64 + me + me * 2,
        "scan": 64 + me + me * 2 + me * 2,
        "bus": 64 + me + me * 2 + me * 2 + (2 * 2048) * 4,
    }
    for k in ("tick", "coeff", "mask", "scan", "bus"):
        assert rb[k] == top[k], (k, rb[k], top[k])

    # one clean module, instantiates the final engine (no variant), 3 edge BRAMs
    assert "module zlc_pulse_streamer_top" in text and text.count("endmodule") == 1
    assert "zlc_edge_streamer" in text
    for ip in ("blk_mem_gen_edge_tick", "blk_mem_gen_edge_coeff", "blk_mem_gen_edge_mask",
               "blk_mem_gen_scan", "blk_mem_gen_busimg"):
        assert ip in text, ip
    # streaming handshake + cursor read-back
    assert "bank_ready" in text and "scan_cursor" in text and "C_CURSOR" in text and "C_BANK_READY" in text
    # CTRL word map matches host.image.CtrlWords
    cw = im.CtrlWords
    for name, off in (("C_COMMAND", cw.COMMAND), ("C_STATUS", cw.STATUS), ("C_PROG_COUNT", cw.PROG_COUNT),
                      ("C_BANK_SIZE", cw.BANK_SIZE), ("C_CURSOR", cw.CURSOR), ("C_BANK_READY", cw.BANK_READY)):
        m = re.search(r"localparam integer %s = (\d+);" % name, text)
        assert m and int(m.group(1)) == off, (name, off, m and m.group(1))
    assert "jtag_axi_0" in text and "axi_bram_ctrl_0" in text


def test_edgetable_prefetch_engine_is_tick_exact_and_gapless():
    """The Architecture-D BRAM+prefetch engine must produce a per-tick output
    byte-identical to the validated combinatorial engine, for every program shape
    and at BRAM read latency 1 AND 2 -- proving the prefetch retiming preserves
    tick-exactness AND 1-tick gaplessness (incl. across scan/loop/repeat
    boundaries).  No Verilog sim in repo, so this Python co-sim is the proof."""

    from Zou_lab_control.neutral_atom.devices.edgetable_engine_model import (
        EngineProgram, reference_play, prefetch_play, PrefetchStall,
    )
    from Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer import _apply_scan_tick

    def prog(**kw):
        base = dict(ticks=[], masks=[], tick_slot_coeffs=[], scan_points=[], slot_count=0,
                    frac_bits=8, loop_start_index=0, loop_end_tick=0, loop_end_slot_coeffs=[],
                    loop_count=1, repeat_forever=False)
        base.update(kw)
        base["tick_slot_coeffs"] = base["tick_slot_coeffs"] or [[0] * base["slot_count"] for _ in base["ticks"]]
        base["loop_end_slot_coeffs"] = base["loop_end_slot_coeffs"] or [0] * base["slot_count"]
        return EngineProgram(**base)

    cases = {
        "simple_rf": prog(ticks=[0, 5, 12, 40], masks=[0, 0b11, 0b100, 0], loop_end_tick=40, repeat_forever=True),
        # back-to-back 1-tick edges: the spacing stress case prefetch must survive.
        "b2b_1tick": prog(ticks=[0, 1, 2, 3, 4, 20], masks=[0, 1, 2, 3, 4, 0], loop_end_tick=20, repeat_forever=True),
        "loop3": prog(ticks=[0, 10, 30, 60], masks=[0, 1, 2, 0], loop_start_index=1, loop_end_tick=30, loop_count=3),
        "scan": prog(ticks=[0, 10, 20, 100], masks=[0, 1, 2, 0],
                     tick_slot_coeffs=[[0, 0], [256, 0], [256, 0], [256, 0]],
                     scan_points=[[0, 0], [256, 0], [512, 0]], slot_count=2, loop_end_tick=100, repeat_forever=True),
        # 1-tick edges immediately after a scan-point boundary (edge0@0, edge1@1).
        "scan_b2b": prog(ticks=[0, 1, 8, 50], masks=[0b1, 0b11, 0b100, 0],
                         tick_slot_coeffs=[[0], [0], [256], [256]],
                         scan_points=[[0], [256], [512]], slot_count=1, loop_end_tick=50, repeat_forever=True),
        "single": prog(ticks=[0], masks=[0b101], loop_end_tick=10, repeat_forever=True),
    }

    N = 400
    for name, pr in cases.items():
        ref = reference_play(pr, N)
        for lat in (1, 2):
            for depth in (lat + 1, 4):
                try:
                    pf = prefetch_play(pr, N, read_latency=lat, fifo_depth=depth)
                except PrefetchStall as exc:  # a stall would be a hardware gap
                    raise AssertionError(f"{name} stalled at latency {lat}/depth {depth}: {exc}")
                assert pf == ref, f"{name}: prefetch != reference at latency {lat}/depth {depth}"

    # Independent brute-force ground truth for a single pass (no loop/repeat/scan):
    # level at tick t = mask of the last edge whose effective tick <= t.
    sp = prog(ticks=[0, 5, 12, 40], masks=[0, 0b11, 0b100, 0], loop_end_tick=40, loop_count=1)
    effs = [_apply_scan_tick(sp.ticks[i], sp.tick_slot_coeffs[i], [], sp.frac_bits) for i in range(len(sp.ticks))]
    brute = []
    for t in range(45):
        m = 0
        for i, e in enumerate(effs):
            if e <= t:
                m = sp.masks[i]
        brute.append(m)
    assert reference_play(sp, 45) == brute
    assert prefetch_play(sp, 45, read_latency=2, fifo_depth=3) == brute


def test_edgetable_d_engine_rtl_structure_matches_design():
    """zlc_pulse_streamer_d.v must implement the verified Architecture-D design:
    edge/scan tables are EXTERNAL BRAM (read ports, not internal LUTRAM), the bus
    tables stay LUTRAM, and the depth-1 prefetch + first/loop_start/final shadows
    are present.  Guards the RTL against drifting from the proven model."""

    root = Path(__file__).resolve().parents[1]
    src = (root / "fpga" / "pulse_streamer" / "zlc_pulse_streamer_d.v").read_text(encoding="utf-8")

    # edge/scan are external BRAM read ports (NOT internal LUTRAM tables)
    assert "edge_raddr" in src and "edge_rdata" in src
    assert "scan_raddr" in src and "scan_rdata" in src
    assert "reg [TICK_WIDTH-1:0] tick_mem" not in src   # no internal edge LUTRAM
    assert "scan_value_mem" not in src                  # no internal scan LUTRAM
    # bus tables DO stay in LUTRAM (the per-tick combinatorial bus engine)
    assert 'ram_style = "distributed"' in src and "bus_start_tick_mem" in src
    # depth-1 prefetch + shadows
    assert "pre_tick" in src and "pre_valid" in src and "rd_wait" in src
    assert "first_tick_shadow" in src and "loop_start_tick_shadow" in src and "final_tick_shadow" in src
    assert "cur_eff" in src and "is_edge0" in src
    # the seamless reload sites are present (loop rewind / scan advance / repeat)
    assert "loops_remaining" in src and "repeat_forever_active" in src and "scan_point_index" in src
    # the validated effective-tick MAC is reused unchanged
    assert "function [TICK_WIDTH-1:0] zlc_effective_tick" in src

    # The engine is built ONLY by the D build tcl (with its top), never by the
    # loader/address-switch builds (those use the validated combinatorial engine).
    fpga = root / "fpga" / "pulse_streamer"
    for tcl in fpga.glob("create_project_*.tcl"):
        text = tcl.read_text(encoding="utf-8")
        if "zlc_pulse_streamer_d.v" in text:
            assert tcl.name == "create_project_d.tcl", tcl.name
            assert "zlc_pulse_streamer_d_top.v" in text


def test_edgetable_d_image_packs_and_round_trips_at_solved_geometry():
    """The Architecture-D AXI write image round-trips a full program at the solved
    35T geometry, and its region bases match the D top localparams (64 / 64 /
    64+2048*8 / +4096*4) so the host->BRAM write addresses are correct."""

    from Zou_lab_control.neutral_atom.devices.sequencer import RuntimeSequenceProgram, RuntimeBusSegment
    from Zou_lab_control.neutral_atom.devices.edgetable_image import (
        pack_program_d, unpack_program_d, d_region_bases, solve_capacity,
    )

    s = solve_capacity("xc7a35t", channel_count=62)
    p = s.params
    bases = d_region_bases(p)
    assert bases["ctrl"] == 0 and bases["edge"] == 64
    assert bases["scan"] == 64 + p.max_edges * 8
    assert bases["bus"] == bases["scan"] + p.max_scan_points * p.num_slots

    prog = RuntimeSequenceProgram(
        sequence_id="a", sequence_name="t", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 50, 120, 400], masks=[0, (1 << 0) | (1 << 5), (1 << 61), 0],
        duration=8e-6, trigger_count=0, repeat_forever=True, loop_start_index=1,
        loop_end_tick=400, loop_count=3, slot_count=2, slot_kinds=["delay", "dac"],
        loop_end_slot_coeffs=[256, 0], tick_slot_coeffs=[[0, 0], [256, 0], [256, 0], [256, 0]],
        scan_points=[[0, 0], [256, 256], [512, -768]], scan_coeff_frac_bits=8, bus_names=["da0"],
        bus_segments=[
            RuntimeBusSegment(bus_index=0, start_tick=50, stop_tick=120, start_value=0, stop_value=0,
                              mode="edge", value_select=2, start_tick_coeffs=[256, 0], stop_tick_coeffs=[256, 0]),
            RuntimeBusSegment(bus_index=2, start_tick=10, stop_tick=400, start_value=512, stop_value=1023, mode="ramp"),
        ],
    )
    out = unpack_program_d(pack_program_d(prog, p), p)
    pad = lambda r, n: list(r) + [0] * (n - len(r))
    assert out["ticks"] == prog.ticks and out["masks"] == prog.masks
    assert out["tick_slot_coeffs"] == [pad(r, p.num_slots) for r in prog.tick_slot_coeffs]
    assert out["scan_points"] == [pad(pt, p.num_slots) for pt in prog.scan_points]  # incl. negative slot
    assert out["loop_start_index"] == 1 and out["loop_count"] == 3 and out["repeat_forever"]
    bus = {b["bus_index"]: b for b in out["bus_segments"]}
    assert bus[0]["value_select"] == 2 and bus[0]["mode"] == "edge"
    assert bus[2]["start_value"] == 512 and bus[2]["stop_value"] == 1023 and bus[2]["mode"] == "ramp"


def test_edgetable_d_top_and_build_structure():
    """The D top instantiates the D engine + the edge/scan/bus BRAMs + the JTAG-to-
    AXI IPs, keeps the exact 62-pin board map with DAC buses driven by the engine,
    and the build tcl creates those IPs.  (Structural contract; the multi-BRAM AXI
    integration itself is bring-up-validated, no Verilog sim in repo.)"""

    root = Path(__file__).resolve().parents[1]
    fpga = root / "fpga" / "pulse_streamer"
    top = (fpga / "zlc_pulse_streamer_d_top.v").read_text(encoding="utf-8")

    assert "zlc_pulse_streamer_d #(" in top                  # the D engine
    assert "jtag_axi_0 zlc_jtag_axi_i" in top
    assert "axi_bram_ctrl_0 zlc_bram_ctrl_i" in top
    assert "blk_mem_gen_edge zlc_edge_bram_i" in top         # asymmetric edge BRAM
    assert "blk_mem_gen_scan zlc_scan_bram_i" in top         # asymmetric scan BRAM
    assert "blk_mem_gen_busimg zlc_bus_img_i" in top         # bus image BRAM
    # the engine reads edge/scan directly (no LUTRAM tables, no staging copy)
    assert ".edge_rdata(edge_doutb" in top and ".scan_rdata(scan_doutb)" in top
    # bus mini-loader drives the engine bus_prog_* (bus stays LUTRAM in the engine)
    assert ".bus_prog_we(bus_prog_we)" in top
    # exact board map + DACs driven by the engine bus_out
    assert "assign da_clk3 = out[61];" in top
    assert ".bus_out(zlc_bus_out)" in top and "zlc_bus_out = 40'b0" not in top
    assert "assign da_dipole[0] = zlc_bus_out[0];" in top

    tcl = (fpga / "create_project_d.tcl").read_text(encoding="utf-8")
    assert "zlc_pulse_streamer_d.v" in tcl and "zlc_pulse_streamer_d_top.v" in tcl
    assert "set top zlc_pulse_streamer_d_top" in tcl
    for ip in ("jtag_axi", "axi_bram_ctrl", "blk_mem_gen_edge", "blk_mem_gen_scan", "blk_mem_gen_busimg"):
        assert ip in tcl, ip
    # edge port-B 256b, scan port-B 128b (asymmetric wide read = 1 edge/point per read)
    assert "Write_Width_B $zlc_edge_portb_bits" in tcl
    assert "set zlc_edge_portb_bits 256" in tcl and "set zlc_scan_portb_bits 128" in tcl


def test_edgetable_d1_prefetch_matches_reference_with_min_spacing():
    """The depth-1 prefetch engine (zlc_pulse_streamer_d.v's design): for programs
    whose minimum edge spacing >= settle+1 it is byte-identical to the reference;
    for closer edges it STALLS -- proving the host-enforced min edge spacing is a
    real requirement, not an unguarded assumption."""

    from Zou_lab_control.neutral_atom.devices.edgetable_engine_model import (
        EngineProgram, reference_play, prefetch_d1_play, min_edge_spacing, PrefetchStall,
    )

    def prog(**kw):
        base = dict(ticks=[], masks=[], tick_slot_coeffs=[], scan_points=[], slot_count=0,
                    frac_bits=8, loop_start_index=0, loop_end_tick=0, loop_end_slot_coeffs=[],
                    loop_count=1, repeat_forever=False)
        base.update(kw)
        base["tick_slot_coeffs"] = base["tick_slot_coeffs"] or [[0] * base["slot_count"] for _ in base["ticks"]]
        base["loop_end_slot_coeffs"] = base["loop_end_slot_coeffs"] or [0] * base["slot_count"]
        return EngineProgram(**base)

    settle = 2
    spaced = {
        "simple": prog(ticks=[0, 5, 12, 40], masks=[0, 3, 4, 0], loop_end_tick=40, repeat_forever=True),
        "loop3": prog(ticks=[0, 10, 30, 60], masks=[0, 1, 2, 0], loop_start_index=1, loop_end_tick=30, loop_count=3),
        "scan": prog(ticks=[0, 10, 20, 100], masks=[0, 1, 2, 0],
                     tick_slot_coeffs=[[0, 0], [256, 0], [256, 0], [256, 0]],
                     scan_points=[[0, 0], [256, 0], [512, 0]], slot_count=2, loop_end_tick=100, repeat_forever=True),
        "edge0_nonzero": prog(ticks=[4, 9, 20], masks=[1, 2, 0], loop_end_tick=20, repeat_forever=True),
    }
    for name, pr in spaced.items():
        assert min_edge_spacing(pr) >= settle + 1, name
        assert prefetch_d1_play(pr, 400, settle=settle) == reference_play(pr, 400), name

    close = prog(ticks=[0, 1, 2, 10], masks=[0, 1, 2, 0], loop_end_tick=10, repeat_forever=True)
    assert min_edge_spacing(close) < settle + 1
    with pytest.raises(PrefetchStall):
        prefetch_d1_play(close, 50, settle=settle)


def test_edgetable_capacity_solver_is_parameterised_and_within_budget():
    """solve_capacity derives (edges, points, addr widths, BRAM depth) from the
    FPGA part + XDC channel count with EVERY resource <= 75% -- so swapping the
    XDC or the FPGA needs no hand-edit.  Architecture D: edge+scan in BRAM, bus in
    LUTRAM."""

    from Zou_lab_control.neutral_atom.devices.edgetable_image import solve_capacity, part_profile

    # 35T target: must reach at least the requested 2048 edges + 2048 points, with
    # every resource within 75%.
    s = solve_capacity("xc7a35tfgg484-2", channel_count=62, target_pct=75.0)
    assert s.part == "xc7a35t"
    assert s.params.max_edges >= 2048
    assert s.params.max_scan_points >= 2048
    assert s.pong_depth >= 256  # streaming window for unbounded points
    assert s.all_within_budget(), s.resource_report
    assert s.resource_report["ramb36"]["pct"] <= 75.0
    assert s.resource_report["lut"]["pct"] <= 75.0
    assert s.resource_report["ff"]["pct"] <= 75.0
    assert s.resource_report["dsp"]["pct"] <= 75.0
    # address widths track the solved depths (power-of-two)
    assert (1 << s.edge_addr_width) == s.params.max_edges
    assert (1 << s.scan_addr_width) == s.params.max_scan_points

    # Bigger part: same or larger capacity, far under budget (no rewrite needed).
    big = solve_capacity("xc7a200t", channel_count=62)
    assert big.params.max_edges >= s.params.max_edges
    assert big.params.max_scan_points >= s.params.max_scan_points
    assert big.resource_report["ramb36"]["pct"] < s.resource_report["ramb36"]["pct"]

    # Fewer channels (different XDC) still solves within budget.
    narrow = solve_capacity("xc7a35t", channel_count=16)
    assert narrow.all_within_budget()
    assert narrow.params.channel_count == 16

    # Unknown part errors loudly rather than silently mis-sizing.
    import pytest as _pytest
    with _pytest.raises(KeyError):
        part_profile("xc7zynqfoo")

    # tighter budget shrinks capacity but stays within budget.
    tight = solve_capacity("xc7a35t", channel_count=62, target_pct=40.0)
    assert tight.all_within_budget()
    assert tight.resource_report["ramb36"]["used"] <= int(0.40 * 50)


def test_edgetable_loader_rtl_constants_match_python_image():
    """The RTL loader hardcodes the image's CTRL offsets, magic and CTRL_WORDS for
    speed; they MUST equal the edgetable_image source of truth (a drift here would
    make the on-chip loader read the wrong BRAM words)."""

    from Zou_lab_control.neutral_atom.devices import edgetable_image as ei

    root = Path(__file__).resolve().parents[1]
    src = (root / "fpga" / "pulse_streamer" / "zlc_axi_program_loader.v").read_text(encoding="utf-8")

    def localparam(name):
        m = re.search(rf"localparam\s+integer\s+{name}\s*=\s*(\d+)", src)
        assert m, f"loader missing localparam {name}"
        return int(m.group(1))

    assert localparam("CTRL_WORDS") == ei.CTRL_WORDS == 32
    assert localparam("C_MAGIC") == ei.CtrlWords.MAGIC
    assert localparam("C_COMMAND") == ei.CtrlWords.COMMAND
    assert localparam("C_STATUS") == ei.CtrlWords.STATUS
    assert localparam("C_PROG_COUNT") == ei.CtrlWords.PROG_COUNT
    assert localparam("C_SCAN_COUNT") == ei.CtrlWords.SCAN_COUNT
    assert localparam("C_SCAN_ENABLE") == ei.CtrlWords.SCAN_ENABLE
    assert localparam("C_REPEAT_FOREVER") == ei.CtrlWords.REPEAT_FOREVER
    assert localparam("C_LOOP_START") == ei.CtrlWords.LOOP_START_ADDR
    assert localparam("C_LOOP_COUNT") == ei.CtrlWords.LOOP_COUNT
    assert localparam("C_LOOP_END_TICK") == ei.CtrlWords.LOOP_END_TICK
    assert localparam("C_LOOP_END_LO") == ei.CtrlWords.LOOP_END_COEFF_LO
    assert localparam("C_LOOP_END_HI") == ei.CtrlWords.LOOP_END_COEFF_HI
    assert localparam("C_BUS_COUNTS") == ei.CtrlWords.BUS_COUNTS
    assert localparam("C_SLOT_COUNT") == ei.CtrlWords.SLOT_COUNT

    # magic constant matches "ZLE1"
    m = re.search(r"IMAGE_MAGIC\s*=\s*32'h([0-9A-Fa-f]+)", src)
    assert m and int(m.group(1), 16) == ei.IMAGE_MAGIC

    # The loader drives the engine's toggle-triggered write port (must hold reset
    # during the load and toggle prog_we / scan_prog_we / bus_prog_we).
    assert "prog_we <= ~prog_we" in src
    assert "scan_prog_we <= ~scan_prog_we" in src
    assert "bus_prog_we <= ~bus_prog_we" in src
    assert "eng_reset" in src and "eng_start" in src


def test_edgetable_loader_fsm_cosim_reconstructs_program():
    """Cycle-accurate co-sim of the on-chip loader FSM (zlc_axi_program_loader.v):
    stepping the FSM over a packed image and capturing the engine's toggle-writes
    must reconstruct exactly what edgetable_image decodes -- proving the loader
    walks edges/scan/bus in the right order with correct addresses + field slices
    and never writes while the engine reset is low.  (No Verilog sim in repo, so
    this Python mirror is the pre-hardware verification of the new sequencer.)"""

    from Zou_lab_control.neutral_atom.devices.sequencer import (
        RuntimeSequenceProgram,
        RuntimeBusSegment,
    )
    from Zou_lab_control.neutral_atom.devices.edgetable_image import (
        pack_program,
        unpack_program,
        EdgeTableImageParams,
    )
    from Zou_lab_control.neutral_atom.devices.edgetable_loader_model import run_loader_model

    p = EdgeTableImageParams()

    def check(prog):
        image = pack_program(prog, p)
        decoded = unpack_program(image, p)
        loaded = run_loader_model(image, p)
        loaded.pop("_cycles")
        for key, value in decoded.items():
            assert loaded[key] == value, f"loader mismatch on {key}"
        return loaded

    # 1) full program: scan + affine coeffs + loop + two DAC bus segments
    full = RuntimeSequenceProgram(
        sequence_id="abc", sequence_name="t", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 50, 120, 400], masks=[0, (1 << 0) | (1 << 5), (1 << 61), 0],
        duration=8e-6, trigger_count=0,
        repeat_forever=True, loop_start_index=1, loop_end_tick=400, loop_count=3,
        slot_count=2, slot_kinds=["delay", "dac"], loop_end_slot_coeffs=[256, 0],
        tick_slot_coeffs=[[0, 0], [256, 0], [256, 0], [256, 0]],
        scan_points=[[0, 0], [256, 256], [512, -768]], scan_coeff_frac_bits=8,
        bus_names=["da0"],
        bus_segments=[
            RuntimeBusSegment(bus_index=0, start_tick=50, stop_tick=120, start_value=0,
                              stop_value=0, mode="edge", value_select=2,
                              start_tick_coeffs=[256, 0], stop_tick_coeffs=[256, 0]),
            RuntimeBusSegment(bus_index=2, start_tick=10, stop_tick=400, start_value=512,
                              stop_value=1023, mode="ramp", value_select=0),
        ],
    )
    check(full)

    # 2) larger edge table (stress the row walk + addressing, no off-by-one)
    n = 200
    big = RuntimeSequenceProgram(
        sequence_id="big", sequence_name="big", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[10 * i for i in range(n)],
        masks=[(i * 2654435761) & ((1 << 62) - 1) for i in range(n)],
        duration=1e-3, trigger_count=0,
        repeat_forever=True, loop_start_index=0, loop_end_tick=10 * n, loop_count=1,
        slot_count=1, slot_kinds=["delay"], loop_end_slot_coeffs=[0],
        tick_slot_coeffs=[[(i % 7) - 3] for i in range(n)],  # signed coeffs incl. negatives
        scan_points=[[k * 64] for k in range(16)],
        scan_coeff_frac_bits=8,
    )
    out = check(big)
    assert out["ticks"][199] == 1990
    assert len(out["scan_points"]) == 16

    # 3) no scan, no bus (plain repeat) still loads
    plain = RuntimeSequenceProgram(
        sequence_id="p", sequence_name="p", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 100, 200], masks=[1, 0, 1],
        duration=4e-6, trigger_count=0,
        repeat_forever=True, loop_start_index=0, loop_end_tick=200, loop_count=1,
        slot_count=0,
    )
    plain_out = check(plain)
    assert plain_out["scan_points"] == []
    assert plain_out["bus_segments"] == []


def test_pulse_table_state_compiles_pair_array_scan_to_full_40ch_template():
    hardware_channels = [f"ch{i:02d}" for i in range(40)]
    # Two bound time slots: s0 scans the camera duration, s1 scans the trailing
    # idle (kept as ``s1+20`` so the affine path carries a non-zero base too).
    state = na.PulseTableState(
        channels=["ch00", "ch01", "ch02", "ch03"],
        periods=[
            na.PulsePeriod(20, (1, 0, 0, 0), unit="ns", name="load"),
            na.PulsePeriod("s0", (0, 0, 0, 1), unit="str (ns)", name="camera"),
            na.PulsePeriod("s1+20", (0, 0, 0, 0), unit="str (ns)", name="idle"),
        ],
        scan_slots=[
            {"kind": "duration", "target": "1", "unit": "ns", "nominal": 20.0},
            {"kind": "duration", "target": "2", "unit": "ns", "nominal": 20.0},
        ],
        scan_table=[[20.0, 20.0], [40.0, 40.0]],
        time_step_ns=20,
        visible_channels=["ch00", "ch03"],
    )

    program = na.compile_pulse_table_scan_runtime_program(
        state,
        channels=hardware_channels,
        clock_hz=50_000_000,
        trigger_channels=["ch03"],
    )

    assert program.channels == hardware_channels
    assert program.scan_enabled is True
    assert program.slot_count == 2
    assert program.slot_kinds == ["duration", "duration"]
    assert program.scan_points == [[1, 1], [2, 2]]
    assert program.ticks == [0, 1, 1, 2]
    assert program.masks == [1, 1 << 3, 0, 0]
    assert program.tick_slot_coeffs == [[0, 0], [0, 0], [256, 0], [256, 256]]
    assert program.trigger_count == 2
    assert len(program.ticks) == 4
    assert len(program.ticks) < len(program.scan_points) * len(state.periods)
    assert len(program.channels) == 40
    # round-trips through the JSON schema with the N-slot coefficient rows intact.
    roundtrip = na.RuntimeSequenceProgram.from_dict(program.to_dict())
    assert roundtrip.scan_points == [[1, 1], [2, 2]]
    assert roundtrip.tick_slot_coeffs == [[0, 0], [0, 0], [256, 0], [256, 256]]
    na.validate_pulse_streamer_program(program, max_edges=1024, max_scan_points=1024, tick_width=32, channel_count=40)


def test_pulse_table_seamless_dac_value_scan_compiles_and_uploads(tmp_path):
    # A 2-bit DAC bus (da_test[0..1]) plus a TTL channel.  The DAC level in the
    # second period is scanned across three points; timing is fixed, so this is
    # the seamless hardware DAC-value scan path.
    state = na.PulseTableState(
        channels=["ch00", "ch01", "ch02"],
        channel_labels={"ch00": "da_test[0]", "ch01": "da_test[1]", "ch02": "trig"},
        visible_channels=["ch00", "ch01", "ch02"],
        time_step_ns=20,
        periods=[
            na.PulsePeriod(100, (0, 0, 1), unit="ns", name="load"),
            na.PulsePeriod(200, (0, 0, 0), unit="ns", name="hold"),
        ],
    )
    # Bind the DAC value of period 1 to a scan slot, then sweep raw 2-bit codes.
    state.bind_field("dac", "da_test@1", unit="value", label="da_test")
    state.set_scan_table([[0.0], [2.0], [3.0]])

    program = na.compile_pulse_table_scan_runtime_program(
        state, channels=["ch00", "ch01", "ch02"], clock_hz=50_000_000, trigger_channels=["ch02"]
    )

    assert program.scan_enabled is True
    assert program.slot_count == 1
    assert program.slot_kinds == ["dac"]
    # DAC codes are stored verbatim (no ns->tick scaling) and never move an edge.
    assert program.scan_points == [[0], [2], [3]]
    assert all(all(c == 0 for c in row) for row in (program.tick_slot_coeffs or []))
    # The scanned period start (100 ns / 20 ns = tick 5) carries value_select=1.
    scanned = [s for s in (program.bus_segments or []) if s.value_select]
    assert len(scanned) == 1
    assert scanned[0].value_select == 1
    assert scanned[0].start_tick == 5
    # The TTL bus member bits must NOT also appear in the digital edge masks.
    assert all((mask & 0b011) == 0 for mask in program.masks)

    na.validate_pulse_streamer_program(
        program, max_edges=1024, max_scan_points=1024, tick_width=32, channel_count=3
    )

    roundtrip = na.RuntimeSequenceProgram.from_dict(program.to_dict())
    assert [s.value_select for s in (roundtrip.bus_segments or [])] == [
        s.value_select for s in (program.bus_segments or [])
    ]

    tcl_path = na.write_vivado_pulse_streamer_tcl(
        tmp_path / "prepare_dac_scan.tcl",
        "prepare",
        program=program,
        project="",
        bitstream="",
        probes="",
        max_edges=1024,
        channel_count=3,
    )
    tcl = tcl_path.read_text(encoding="utf-8")
    # The upload stages the value_select probe (probe_out27) for each segment.
    assert "zlc_bus_prog_value_select_probe" in tcl
    assert "probe_out27" in tcl


def _rtl_bus_held_value(program, bus_index, tick, scan_point, *, bus_width=10):
    """Python re-implementation of the RTL bus engine's held DAC value.

    Faithfully mirrors ``zlc_bus_apply_segment`` / ``zlc_bus_seg_start`` in
    ``fpga/pulse_streamer/zlc_pulse_streamer.v``: at a scan point the bus walks
    its segments in *effective*-tick order and holds the most recent one whose
    effective start tick <= ``tick``.  The effective tick applies the segment's
    affine coefficients to the current scan point (so a scanned duration moves
    the segment), and an edge segment with ``value_select = j+1`` reads the low
    ``bus_width`` bits of scan slot ``j`` (so the DAC value tracks the scan).
    Together this models the simultaneous DA-value + duration + delay scan.
    """

    from Zou_lab_control.neutral_atom.devices.sequencer import _apply_affine_ticks

    mask = (1 << bus_width) - 1
    frac = int(getattr(program, "scan_coeff_frac_bits", 8))
    point = list(program.scan_points[scan_point]) if program.scan_points else []

    def eff_start(seg):
        coeffs = getattr(seg, "start_tick_coeffs", None)
        if coeffs and point:
            return _apply_affine_ticks(int(seg.start_tick), coeffs, point, frac)
        return int(seg.start_tick)

    segments = sorted(
        (s for s in (program.bus_segments or []) if int(s.bus_index) == int(bus_index)),
        key=eff_start,
    )
    value = 0
    for seg in segments:
        if eff_start(seg) > tick:
            break
        sel = int(getattr(seg, "value_select", 0))
        if sel:
            value = int(point[sel - 1]) & mask
        else:
            value = int(seg.stop_value) & mask
    return value


def test_dac_value_scan_behavioral_model_tracks_scanned_code():
    """End-to-end (logic) proof that a seamless DAC scan reaches bus_out.

    Compiles a real DAC-value scan, then runs a faithful Python model of the RTL
    bus engine over the uploaded program and checks that, for *every* scan point,
    the DAC output during the scanned period equals that point's code and is the
    prior (unscanned) level just before it.  Without a Verilog simulator this is
    the strongest available evidence that ``value_select`` carries the scanned
    value all the way through to the DAC output.
    """

    hw = [f"ch{i:02d}" for i in range(12)]
    labels = {f"ch{i:02d}": f"da[{i}]" for i in range(10)}
    labels["ch10"] = "trig"
    state = na.PulseTableState(
        channels=hw,
        channel_labels=labels,
        visible_channels=hw,
        time_step_ns=20,
        periods=[
            na.PulsePeriod(100, tuple([0] * 10 + [1, 0]), unit="ns"),  # da=0, trig high
            na.PulsePeriod(200, tuple([0] * 12), unit="ns"),           # scanned da level here
            na.PulsePeriod(100, tuple([0] * 12), unit="ns"),
        ],
    )
    state.bind_field("dac", "da@1")                 # scan the 10-bit "da" bus in period 1
    codes = [0, 256, 768, 1023]
    state.set_scan_table([[c] for c in codes])

    program = na.compile_pulse_table_scan_runtime_program(state, channels=hw, clock_hz=50_000_000)
    na.validate_pulse_streamer_program(program, max_edges=1024, max_scan_points=1024, tick_width=32, channel_count=12)

    scanned = [s for s in (program.bus_segments or []) if int(getattr(s, "value_select", 0))]
    assert len(scanned) == 1, "exactly one scanned DAC segment expected"
    seg = scanned[0]
    bus = int(seg.bus_index)
    assert int(seg.start_tick) == 5  # period-1 start = 100 ns / 20 ns

    for point_index, code in enumerate(codes):
        # During the scanned period the DAC equals THIS point's code...
        assert _rtl_bus_held_value(program, bus, int(seg.start_tick), point_index) == code
        assert _rtl_bus_held_value(program, bus, int(seg.start_tick) + 3, point_index) == code
        # ...and the prior (period-0) level is still the unscanned 0 just before it.
        assert _rtl_bus_held_value(program, bus, int(seg.start_tick) - 1, point_index) == 0

    # Consecutive scan points really produce different DAC outputs (seamless sweep).
    sweep = [_rtl_bus_held_value(program, bus, int(seg.start_tick), p) for p in range(len(codes))]
    assert sweep == codes


def test_pulse_streamer_rtl_has_dac_value_select_path():
    """The checked-in RTL must carry the DAC value_select (PSEL) scan path.

    Guards the four pieces that make a bus segment read its DAC code from a scan
    slot at runtime, so a refactor cannot silently drop seamless DAC scanning.
    """

    root = Path(__file__).resolve().parents[1]
    core = (root / "fpga" / "pulse_streamer" / "zlc_pulse_streamer.v").read_text(encoding="utf-8")
    top = (root / "fpga" / "pulse_streamer" / "zlc_pulse_streamer_top_address_switch.v").read_text(encoding="utf-8")
    tcl = (root / "fpga" / "pulse_streamer" / "create_project_address_switch.tcl").read_text(encoding="utf-8")

    # core: the input port, the per-segment memory, the write, and the slot read.
    assert "bus_prog_value_select" in core
    assert "bus_value_select_mem" in core
    assert "slot_vec[(sel - 1'b1)*TICK_WIDTH +: BUS_WIDTH]" in core
    # the slot vector is threaded explicitly into the apply/start tasks.
    assert "zlc_bus_start_table(scan_value_mem[next_scan_addr])" in core
    # affine bus-segment ticks: per-segment coeff memories + effective-tick eval.
    assert "bus_start_tick_coeff_mem" in core
    assert "bus_stop_tick_coeff_mem" in core
    assert "zlc_bus_seg_start" in core
    assert "eff_tk_start = zlc_effective_tick(bus_start_tick_mem[addr]" in core
    # top + IP: probe_out27 value_select, probe_out28/29 tick coeffs, 30 probe_outs.
    assert "wire [2:0] zlc_bus_prog_value_select;" in top
    assert ".probe_out27(zlc_bus_prog_value_select)" in top
    assert ".probe_out28(zlc_bus_prog_start_tick_coeffs)" in top
    assert ".probe_out29(zlc_bus_prog_stop_tick_coeffs)" in top
    assert "CONFIG.C_NUM_PROBE_OUT {30}" in tcl
    assert "CONFIG.C_PROBE_OUT27_WIDTH {3}" in tcl
    assert "CONFIG.C_PROBE_OUT28_WIDTH {64}" in tcl
    assert "CONFIG.C_PROBE_OUT29_WIDTH {64}" in tcl


def test_dac_plus_duration_scan_behavioral_model_value_and_timing():
    """Behavioral proof of the simultaneous DA-value + duration scan.

    A duration is scanned BEFORE the DAC period, so the DAC segment's effective
    tick shifts with each point.  The RTL bus-engine model must show the scanned
    DAC code appearing at the SHIFTED tick (and the prior 0 level just before it)
    for every scan point -- i.e. value and timing scan together.
    """

    hw = [f"ch{i:02d}" for i in range(12)]
    labels = {f"ch{i:02d}": f"da[{i}]" for i in range(10)}
    labels["ch10"] = "trig"
    state = na.PulseTableState(
        channels=hw,
        channel_labels=labels,
        visible_channels=hw,
        time_step_ns=20,
        periods=[
            na.PulsePeriod(100, tuple([0] * 10 + [1, 0]), unit="ns"),  # period 0 duration scanned
            na.PulsePeriod(200, tuple([0] * 12), unit="ns"),           # DAC level scanned here
        ],
    )
    state.bind_field("duration", "0")     # s0: period-0 duration (moves period-1 start)
    state.bind_field("dac", "da@1")       # s1: scanned DAC level in period 1
    # rows: [period-0 duration ns, DAC code]
    durations_ns = [100, 200, 400]
    codes = [0, 512, 1023]
    state.set_scan_table([[d, c] for d, c in zip(durations_ns, codes)])

    program = na.compile_pulse_table_scan_runtime_program(state, channels=hw, clock_hz=50_000_000)
    na.validate_pulse_streamer_program(program, max_edges=1024, max_scan_points=1024, tick_width=32, channel_count=12)

    seg = next(s for s in program.bus_segments if int(getattr(s, "value_select", 0)))
    bus = int(seg.bus_index)
    from Zou_lab_control.neutral_atom.devices.sequencer import _apply_affine_ticks

    frac = program.scan_coeff_frac_bits
    for point_index, code in enumerate(codes):
        point = program.scan_points[point_index]
        eff = _apply_affine_ticks(int(seg.start_tick), seg.start_tick_coeffs, point, frac)
        dur_ticks = durations_ns[point_index] // 20
        assert eff == dur_ticks  # period-1 start = scanned period-0 duration
        # DAC code present at/after the SHIFTED tick, and 0 just before it.
        assert _rtl_bus_held_value(program, bus, eff, point_index) == code
        assert _rtl_bus_held_value(program, bus, eff + 5, point_index) == code
        if eff > 0:
            assert _rtl_bus_held_value(program, bus, eff - 1, point_index) == 0


def test_pulse_table_dac_duration_delay_scan_simultaneously():
    """DAC value + a duration BEFORE it + a delay all scan together.

    The DAC bus segment must carry affine tick coefficients so its effective
    tick moves in lockstep with the scanned duration, while its value still
    tracks the scanned DAC code -- the simultaneous DA+duration+delay scan.
    """

    state = na.PulseTableState(
        channels=["ch00", "ch01", "ch02"],
        channel_labels={"ch00": "da_test[0]", "ch01": "da_test[1]", "ch02": "trig"},
        visible_channels=["ch00", "ch01", "ch02"],
        time_step_ns=20,
        periods=[
            na.PulsePeriod(100, (0, 0, 1), unit="ns"),  # period 0 duration scanned
            na.PulsePeriod(200, (0, 0, 0), unit="ns"),  # DAC level scanned here (period 1)
        ],
    )
    state.bind_field("duration", "0", unit="ns", label="load dur")   # s0
    state.bind_field("delay", "ch02", unit="ns", label="trig delay")  # s1
    state.bind_field("dac", "da_test@1", unit="value", label="da_test")  # s2
    # rows: [period-0 duration ns, trig delay ns, DAC code]
    state.set_scan_table([[40.0, 0.0, 0.0], [80.0, 20.0, 3.0], [120.0, 40.0, 2.0]])

    program = na.compile_pulse_table_scan_runtime_program(
        state, channels=["ch00", "ch01", "ch02"], clock_hz=50_000_000
    )
    assert program.slot_kinds == ["duration", "delay", "dac"]
    scanned = [s for s in (program.bus_segments or []) if int(getattr(s, "value_select", 0))]
    assert len(scanned) == 1
    seg = scanned[0]
    # The DAC segment sits at period-1 start = scanned period-0 duration, so its
    # start-tick coefficient for slot s0 (duration) must be non-zero.
    assert seg.start_tick_coeffs is not None and seg.start_tick_coeffs[0] != 0
    # ...and zero for the delay/dac slots (they don't move period-1's start).
    assert seg.start_tick_coeffs[2] == 0

    # The DAC value tracks the scanned code, and the effective tick moves with the
    # scanned period-0 duration -- verified by the affine evaluation per point.
    from Zou_lab_control.neutral_atom.devices.sequencer import _apply_affine_ticks

    frac = program.scan_coeff_frac_bits
    for point in program.scan_points:
        dur_ticks = point[0]  # period-0 duration in ticks for this point
        eff = _apply_affine_ticks(seg.start_tick, seg.start_tick_coeffs, point, frac)
        assert eff == seg.start_tick + dur_ticks  # base (period-1 start) shifts by the scanned duration

    na.validate_pulse_streamer_program(
        program, max_edges=1024, max_scan_points=1024, tick_width=32, channel_count=3
    )


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
        channel_labels={"ch00": "trap", "ch03": "trig"},
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
    assert "final_tick <= (prog_count == 0) ? {TICK_WIDTH{1'b0}} : zlc_effective_tick(" in core
    assert "final_coeffs_shadow" in core
    assert "final_x_coeff_shadow" not in core
    assert "final_y_coeff_shadow" not in core
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
        channel_labels={"ch00": "trap", "ch01": "cooling", "ch03": "trig"},
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
        channel_labels={"ch00": "trap", "ch01": "cooling", "ch02": "probe", "ch03": "trig"},
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
        na.compile_pulse_table_runtime_program(state, channels=[f"ch{i:02d}" for i in range(62)], clock_hz=50_000_000)
    except ValueError as exc:
        assert "not in hardware channels" in str(exc)
        assert "not_on_fpga" in str(exc)
    else:
        raise AssertionError("unknown pulse-table channels should be rejected")


def test_checked_in_camera_imaging_pulse_compiles_for_address_switch_fpga():
    path = Path(__file__).resolve().parents[1] / "pulses" / "camera_imaging_address_switch.json"
    state = na.PulseTableState.load(path)
    program = state.compile(clock_hz=50_000_000, trigger_channels=["ch11"])

    assert state.channels == [f"ch{i:02d}" for i in range(62)]
    assert state.visible_channels == ["ch09", "ch00", "ch03", "ch11"]
    assert state.time_step_ns == 20
    assert len(state.channel_labels) == 62
    assert state.channel_labels["ch00"] == "cooling"
    assert state.channel_labels["ch03"] == "probe"
    assert state.channel_labels["ch06"] == "trig"
    assert state.channel_labels["ch11"] == "emCCD"
    assert state.channel_labels["ch09"] == "trap"
    assert state.channel_labels["ch18"] == "da_dipole[0]"
    assert state.channel_labels["ch39"] == "da_clk1"
    assert state.delay_steps("ch00", time_step_ns=20) == 0
    assert state.delay_steps("ch11", time_step_ns=20) == 0
    assert state.repeat_start is None
    assert state.repeat_end is None
    assert state.repeat_count == 1
    assert state.repeat_forever is True
    assert state.repeat_forever_boundary_active_channels() == []
    # The camera exposure is bound to a single time slot whose nominal is the
    # default 19.98 ms exposure; the reference render uses that nominal.
    assert [slot.kind for slot in state.scan_slots] == ["duration"]
    assert state.primary_time_slot() == "s0"
    assert state.reference_slots() == {"s0": 19_980_000}
    exposure_period = next(period for period in state.periods if period.name == "camera_exposure")
    assert exposure_period.duration == "s0"
    assert exposure_period.unit == "str (ns)"
    assert state.slot_index_for("duration", str(state.periods.index(exposure_period))) == 0
    assert state.periods[0].states[state.channel_index("ch11")] == 0
    assert program.channels == state.channels
    assert program.ticks == [0, 100_000, 105_000, 106_000, 1_105_000, 1_106_000]
    assert program.masks == [513, 512, 2568, 520, 512, 0]
    assert program.trigger_count == 1
    assert program.repeat_forever is True
    assert program.loop_start_index == 0
    assert program.loop_end_tick == 1_106_000
    assert program.loop_count == 1
    na.validate_pulse_streamer_program(program, max_edges=1024, tick_width=32, channel_count=62)

    shorter = state.compile(clock_hz=50_000_000, trigger_channels=["ch11"], slots={"s0": 2_000_000})
    assert shorter.ticks == [0, 100_000, 105_000, 106_000, 206_000, 207_000]
    assert shorter.masks == program.masks
    assert shorter.trigger_count == 1
    finite = na.finite_frame_sequence(state.with_slots_resolved({"s0": 2_000_000}), 3, trigger_channels=["ch11"])
    finite_program = na.compile_runtime_program(
        finite,
        channels=state.channels,
        clock_hz=50_000_000,
        trigger_channels=["ch11"],
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


def test_bind_pulse_controller_updates_slot_and_fires_runtime_sequencer():
    sequencer = na.RuntimeSequencer(
        channels=["ch00", "ch03"],
        clock_hz=100_000_000,
        trigger_channels=["ch03"],
        sleep_scale=0.0,
    )
    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod("s0", (1, 0), unit="str (ns)"),
            na.PulsePeriod(20, (0, 1), unit="ns"),
            na.PulsePeriod(20, (0, 0), unit="ns"),
        ],
        scan_slots=[{"kind": "duration", "target": "0", "unit": "ns", "nominal": 100.0}],
        time_step_ns=10,
        repeat_forever=False,
    )

    pulse = na.bind_pulse(sequencer, state)
    assert pulse.snapshot()["last_program"] is None
    assert pulse.snapshot()["sequencer_channels"] == ["ch00", "ch03"]
    pulse.set_time(200)
    program = pulse.on_pulse(wait=True, timeout=1.0)

    assert program.ticks == [0, 20, 22, 24]
    assert program.masks == [1 << 0, 1 << 1, 0, 0]
    assert program.repeat_forever is False
    assert sequencer.snapshot()["state"] == "done"
    snapshot = pulse.snapshot()
    assert snapshot["slots"] == {"s0": 200.0}
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
            na.PulsePeriod("s0", (1, 1, 1), unit="str (ns)"),
            na.PulsePeriod(100, (0, 0, 0), unit="ns"),
        ],
        scan_slots=[{"kind": "duration", "target": "1", "unit": "ns", "nominal": 1_000.0}],
        time_step_ns=10,
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
    # The bound exposure slot is driven per scan point via frame_sequence(time_ns=...):
    # reference 8 us -> 820, then 2 us -> 220 and 4 us -> 420 loop end ticks.
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
            na.PulsePeriod("s0", (1, 1, 1), unit="str (ns)"),
            na.PulsePeriod(100, (0, 0, 0), unit="ns"),
        ],
        scan_slots=[{"kind": "duration", "target": "1", "unit": "ns", "nominal": 1_000.0}],
        time_step_ns=10,
        repeat_forever=True,
    )
    path = state.save(tmp_path / "camera_imaging.json")

    pulse = exp.timing.bind_pulse(path)
    assert pulse.sequencer is exp.sequencer
    assert pulse.snapshot()["sequencer_channels"] == hardware_channels
    pulse.set_time(2_000)
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
    assert [program.loop_end_tick for program in sequencer.prepared_programs[-3:]] == [820, 220, 420]
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
        channels=["trap", "trig"],
        periods=[
            na.PulsePeriod(100, (1, 0), unit="ns"),
            na.PulsePeriod("s0", (0, 1), unit="str (ns)"),
        ],
        scan_slots=[{"kind": "duration", "target": "1", "unit": "ns", "nominal": 20.0}],
        time_step_ns=1,
    )

    assert state.to_sequence(slots={"s0": 20}, time_step_ns=10).validate(clock_hz=100e6, channels=state.channels).ok
    try:
        state.to_sequence(slots={"s0": 25}, time_step_ns=10)
    except ValueError as exc:
        assert "integer multiple" in str(exc)
    else:
        raise AssertionError("pulse table should reject slot values off the minimal time grid")

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
    seq = na.PulseSequence(name="delayed").pulse("trig", 0.0, 20e-9).delay("trig", 10e-9)
    state = na.PulseTableState.from_sequence(seq, channels=["trap", "trig"], clock_hz=100e6)
    round_trip = state.to_sequence()

    assert state.delays == {}
    assert [(p.channel, round(p.start, 10), round(p.duration, 10)) for p in round_trip.effective_pulses()] == [
        ("trig", 10e-9, 20e-9)
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
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "emCCD"])
    backend = na.CommandSequencerBackend(tmp_path, fire_command=command)

    backend.fire(program)

    assert marker.read_text(encoding="utf-8") == program.sequence_id
    payload = json.loads((tmp_path / "prepared_program.json").read_text(encoding="utf-8"))
    assert payload["trigger_count"] == 1
    assert payload["source_sequence"]["name"] == seq.name


def test_command_sequencer_backend_error_includes_log_tail(tmp_path):
    command = f'"{sys.executable}" -c "print(\'prepare failed detail\'); raise SystemExit(7)"'
    seq = na.imaging_sequence(exposure=1e-4, load=True)
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "emCCD"])
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
            na.PulsePeriod("s0", (1, 0), unit="str (ns)"),
            na.PulsePeriod(20, (1, 1), unit="ns"),
            na.PulsePeriod(20, (0, 0), unit="ns"),
        ],
        scan_slots=[{"kind": "duration", "target": "0", "unit": "ns", "nominal": 100.0}],
        time_step_ns=10,
        repeat_forever=False,
    )

    first = service.prepare(state)
    second = service.prepare(state)
    third = service.prepare(state.with_slots_resolved({"s0": 200}))

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
    assert (
        "wrote 3/4 edge rows, 0/0 scan points, and 0/0 bus segments "
        "reset_settle_ms=$zlc_prepare_reset_settle_ms repeat_forever=1 scan=0 "
        "loop_start=2 loop_end=30 loop_count=2 bus_counts=0"
    ) in loop_metadata_prepare


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


def test_sequencer_server_jtag_axi_backend_warm_starts_axi_session(tmp_path, monkeypatch):
    """The default 'jtag-axi' backend brings up the run-length VivadoAxiStreamerSession
    and wires its prepare/fire/wait_done/safe_state into the RPyC service."""

    from Zou_lab_control.neutral_atom.devices import axi_session
    from Zou_lab_control.neutral_atom.devices import sequencer_server

    events: list[str] = []

    class FakeAxiSession:
        def __init__(self, **kwargs):
            events.append(f"init:{Path(kwargs['state_dir']).name}:clk={int(kwargs['clock_hz'])}")

        def start(self):
            events.append("start")
            return self

        def prepare(self, program):
            events.append("prepare")

        def fire(self, program=None):
            events.append("fire")

        def wait_done(self, program=None, timeout=None):
            events.append("wait")
            return True

        def safe_state(self):
            events.append("safe")

    def fake_serve(service, *, host, port, start):
        events.append(f"serve:{host}:{port}")
        return object()

    monkeypatch.setattr(axi_session, "VivadoAxiStreamerSession", FakeAxiSession)
    monkeypatch.setattr(sequencer_server, "serve_runtime_sequencer", fake_serve)

    service = sequencer_server.run_server(
        channels=[f"ch{i:02d}" for i in range(62)],
        trigger_channels=["ch03"],
        host="127.0.0.1",
        port=18861,
        clock_hz=50_000_000,
        state_dir=tmp_path / "state_loader",
        backend="jtag-axi",
        warm_start=True,
    )

    assert events[:3] == ["init:state_loader:clk=50000000", "start", "serve:127.0.0.1:18861"]
    assert service is not None


class _FakeLoaderHardware:
    """In-memory stand-in for the programmed FPGA: a BRAM dict + the loader's
    COMMAND/STATUS mailbox.  On LOAD it runs the cycle-accurate loader model over the
    uploaded image and only asserts STATUS_LOADED if the reconstructed engine tables
    match the decoded image -- so the test exercises the FULL host->upload->loader
    path, not just the Tcl shape."""

    def __init__(self, params):
        from Zou_lab_control.neutral_atom.devices.edgetable_image import CtrlWords
        self.params = params
        self.CtrlWords = CtrlWords
        self.bram: dict[int, int] = {}
        self.status = 0
        self.load_ok = False
        self.fired = False

    def __call__(self, lines, action, timeout):
        from Zou_lab_control.neutral_atom.devices.edgetable_image import (
            unpack_program, CtrlWords, CMD_LOAD, CMD_FIRE, CMD_SAFE,
            STATUS_LOADED, STATUS_RUNNING, STATUS_DONE,
        )
        from Zou_lab_control.neutral_atom.devices.edgetable_loader_model import run_loader_model

        text = "\n".join(lines)
        # writes: -address AAAA -data DDDD ... -type write
        for addr_hex, data_hex in re.findall(
            r"-address ([0-9A-Fa-f]+) -data ([0-9A-Fa-f]+) -len 1 -type write", text
        ):
            word = int(addr_hex, 16) // 4
            value = int(data_hex, 16)
            self.bram[word] = value
            if word == CtrlWords.COMMAND and value != 0:
                if value & CMD_SAFE:
                    self.status = 0
                    self.load_ok = False
                if value & CMD_LOAD:
                    # Verify the uploaded image with the loader co-sim model.
                    decoded = unpack_program(self.bram, self.params)
                    loaded = run_loader_model(self.bram, self.params)
                    loaded.pop("_cycles")
                    self.load_ok = all(loaded[k] == decoded[k] for k in decoded)
                    self.status = STATUS_LOADED if self.load_ok else 0x8
                if value & CMD_FIRE:
                    self.fired = True
                    self.status = (self.status | STATUS_RUNNING | STATUS_DONE) & ~0x8
        # reads: return the requested word as "ZLCDATA <hex>"
        m = re.search(r"-address ([0-9A-Fa-f]+) -len 1 -type read", text)
        if m:
            word = int(m.group(1), 16) // 4
            if word == CtrlWords.STATUS:
                return f"ZLCDATA {self.status:08X}\n"
            return f"ZLCDATA {self.bram.get(word, 0):08X}\n"
        return "ok\n"


def test_vivado_axi_session_loads_and_fires_edge_table_program(tmp_path):
    """prepare/fire/wait_done/safe_state drive the loader COMMAND/STATUS mailbox over
    create_hw_axi_txn writes/reads, and the uploaded image round-trips through the
    loader model (no Vivado, no hardware)."""

    from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession
    from Zou_lab_control.neutral_atom.devices.edgetable_image import (
        EdgeTableImageParams, CtrlWords, CMD_LOAD, CMD_FIRE, CMD_SAFE,
    )
    from Zou_lab_control.neutral_atom.devices.sequencer import (
        RuntimeSequenceProgram, RuntimeBusSegment,
    )

    program = RuntimeSequenceProgram(
        sequence_id="abc", sequence_name="t", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 50, 120, 400], masks=[0, (1 << 0) | (1 << 5), (1 << 61), 0],
        duration=8e-6, trigger_count=0,
        repeat_forever=False, loop_start_index=0, loop_end_tick=400, loop_count=1,
        slot_count=2, slot_kinds=["delay", "dac"], loop_end_slot_coeffs=[0, 0],
        tick_slot_coeffs=[[0, 0], [256, 0], [256, 0], [256, 0]],
        scan_points=[[0, 0], [256, 256], [512, 768]], scan_coeff_frac_bits=8,
        bus_names=["da0"],
        bus_segments=[
            RuntimeBusSegment(bus_index=0, start_tick=50, stop_tick=120, start_value=0,
                              stop_value=0, mode="edge", value_select=2,
                              start_tick_coeffs=[256, 0], stop_tick_coeffs=[256, 0]),
        ],
    )

    params = EdgeTableImageParams()
    hw = _FakeLoaderHardware(params)
    session = VivadoAxiStreamerSession(state_dir=tmp_path, params=params, tcl_executor=hw)

    session.prepare(program)
    assert hw.load_ok, "loader model must accept the uploaded image"
    session.fire()
    assert hw.fired
    assert session.wait_done(timeout=1.0) is True
    session.safe_state()

    # The COMMAND mailbox always writes 0 before a command (clean rising edge) and the
    # commands themselves were issued at the COMMAND word (byte addr = word*4).
    cmd_addr = f"{CtrlWords.COMMAND * 4:08X}"
    assert hw.bram[CtrlWords.PROG_COUNT] == 4  # edges uploaded
    assert hw.bram[CtrlWords.SCAN_COUNT] == 3  # scan points uploaded


def test_vivado_axi_session_d_variant_uploads_and_enforces_min_spacing(tmp_path):
    """The 'd' variant packs the Architecture-D image, the uploaded BRAM image
    round-trips through unpack_program_d, and the host REJECTS programs whose edges
    are closer than the depth-1 prefetch can serve (the min-spacing contract)."""

    import re as _re
    from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession, D_MIN_EDGE_SPACING
    from Zou_lab_control.neutral_atom.devices.edgetable_image import (
        solve_capacity, unpack_program_d, DCtrl, D_STATUS_LOADED, D_STATUS_RUNNING, D_STATUS_DONE, D_CMD_LOAD, D_CMD_FIRE,
    )
    from Zou_lab_control.neutral_atom.devices.sequencer import RuntimeSequenceProgram

    params = solve_capacity("xc7a35t", channel_count=62).params

    class DHw:
        def __init__(self):
            self.bram = {}
            self.status = 0
            self.load_ok = False

        def __call__(self, lines, action, timeout):
            text = "\n".join(lines)
            for a, d in _re.findall(r"-address ([0-9A-Fa-f]+) -data ([0-9A-Fa-f]+) -len 1 -type write", text):
                word = int(a, 16) // 4
                val = int(d, 16)
                self.bram[word] = val
                if word == DCtrl.COMMAND and val != 0:
                    if val & D_CMD_LOAD:
                        dec = unpack_program_d(self.bram, params)
                        self.load_ok = (dec["ticks"] == prog.ticks and dec["masks"] == prog.masks
                                        and dec["scan_points"] == [list(p) + [0] * (params.num_slots - len(p)) for p in prog.scan_points])
                        self.status = D_STATUS_LOADED if self.load_ok else 0x8
                    if val & D_CMD_FIRE:
                        self.status = (self.status | D_STATUS_RUNNING | D_STATUS_DONE) & ~0x8
            m = _re.search(r"-address ([0-9A-Fa-f]+) -len 1 -type read", text)
            if m:
                word = int(m.group(1), 16) // 4
                return f"ZLCDATA {(self.status if word == DCtrl.STATUS else self.bram.get(word, 0)):08X}\n"
            return "ok\n"

    # edges spaced >= D_MIN_EDGE_SPACING; a 2-slot scan
    prog = RuntimeSequenceProgram(
        sequence_id="d", sequence_name="d", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 10, 40, 200], masks=[0, 1, 2, 0], duration=4e-6, trigger_count=0,
        repeat_forever=True, loop_start_index=0, loop_end_tick=200, loop_count=1,
        slot_count=2, slot_kinds=["delay", "dac"], loop_end_slot_coeffs=[0, 0],
        tick_slot_coeffs=[[0, 0], [256, 0], [256, 0], [256, 0]],
        scan_points=[[0, 0], [256, 100], [512, 200]], scan_coeff_frac_bits=8,
    )
    hw = DHw()
    session = VivadoAxiStreamerSession(state_dir=tmp_path, params=params, variant="d", tcl_executor=hw)
    session.prepare(prog)
    assert hw.load_ok, "D image must round-trip through unpack_program_d"
    session.fire()
    assert session.wait_done(timeout=1.0) is True

    # too-close edges (gap 1 < D_MIN_EDGE_SPACING) are rejected at prepare.
    close = RuntimeSequenceProgram(
        sequence_id="c", sequence_name="c", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 1, 2, 50], masks=[0, 1, 2, 0], duration=1e-6, trigger_count=0,
        repeat_forever=True, loop_start_index=0, loop_end_tick=50, loop_count=1, slot_count=0,
    )
    import pytest as _pytest
    with _pytest.raises(ValueError, match="ticks apart"):
        VivadoAxiStreamerSession(state_dir=tmp_path, params=params, variant="d", tcl_executor=DHw()).prepare(close)


def test_vivado_axi_session_repeat_forever_treats_running_as_done(tmp_path):
    """A repeat_forever program never asserts DONE; wait_done must return once RUNNING
    is seen instead of blocking for the whole timeout."""

    from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession
    from Zou_lab_control.neutral_atom.devices.edgetable_image import EdgeTableImageParams, STATUS_RUNNING
    from Zou_lab_control.neutral_atom.devices.sequencer import RuntimeSequenceProgram

    program = RuntimeSequenceProgram(
        sequence_id="p", sequence_name="p", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 100, 200], masks=[1, 0, 1],
        duration=4e-6, trigger_count=0,
        repeat_forever=True, loop_start_index=0, loop_end_tick=200, loop_count=1,
        slot_count=0,
    )

    class _ForeverHw(_FakeLoaderHardware):
        def __call__(self, lines, action, timeout):
            from Zou_lab_control.neutral_atom.devices.edgetable_image import CMD_FIRE, CtrlWords, STATUS_LOADED
            out = super().__call__(lines, action, timeout)
            # override: a forever program is RUNNING but NEVER DONE
            if self.fired:
                self.status = (STATUS_LOADED | STATUS_RUNNING)
            return out if "-type read" not in "\n".join(lines) else f"ZLCDATA {self.status:08X}\n"

    params = EdgeTableImageParams()
    hw = _ForeverHw(params)
    session = VivadoAxiStreamerSession(state_dir=tmp_path, params=params, tcl_executor=hw)
    session.prepare(program)
    session.fire()
    # DONE never sets, but wait_done returns True because RUNNING is observed.
    assert session.wait_done(timeout=1.0) is True


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
        channels=["trap", "cooling", "probe", "emCCD"],
        clock_hz=50_000_000,
        trigger_channels=["emCCD"],
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
    remote = na.RemoteSequencer(host="127.0.0.1", port=port, channels=["trap", "cooling", "probe", "emCCD"], clock_hz=50_000_000)
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
            na.PulsePeriod("s0", (1, 1), unit="str (ns)"),
            na.PulsePeriod(100, (0, 0), unit="ns"),
        ],
        scan_slots=[{"kind": "duration", "target": "1", "unit": "ns", "nominal": 1_000.0}],
        time_step_ns=10,
        repeat_forever=True,
        visible_channels=["ch00", "ch03"],
    )
    pulse = na.bind_pulse(remote, state)
    try:
        pulse.set_time(2_000)
        program = pulse.on_pulse(wait=True, timeout=1.0, repeat_forever=False)
        snapshot = pulse.snapshot()
    finally:
        remote.close()
        server.close()

    assert remote.channels == hardware_channels
    assert remote.clock_hz == 100_000_000
    assert snapshot["slots"] == {"s0": 2_000.0}
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
            na.PulsePeriod("s0", (1, 1, 1), unit="str (ns)"),
            na.PulsePeriod(100, (0, 0, 0), unit="ns"),
        ],
        scan_slots=[{"kind": "duration", "target": "1", "unit": "ns", "nominal": 1_000.0}],
        time_step_ns=10,
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
    assert "fpga\\build\\address_switch\\address_switch.xpr" in fpga_text
    assert "qCMOS.py" not in hardware_text + fpga_text
    assert "pxie_control" not in hardware_text + fpga_text


def test_real_device_templates_load_without_hardware_connection():
    manual = na.load_devices("manual_template")
    remote = na.load_devices("remote_template", overrides={"sequencer": {"host": "192.168.0.21", "port": 18862}})
    hardware_channels = [f"ch{i:02d}" for i in range(62)]

    assert isinstance(manual.camera, na.QCMOSCamera)
    assert manual.camera.dcam_module_name == na.DEFAULT_DCAM_MODULE
    assert isinstance(manual.sequencer, na.ManualSequencer)
    assert manual.sequencer.channels == hardware_channels
    assert manual.sequencer.clock_hz == 50_000_000
    assert manual.sequencer.trigger_channels == ("ch11",)
    assert isinstance(remote.camera, na.QCMOSCamera)
    assert remote.camera.dcam_module_name == na.DEFAULT_DCAM_MODULE
    assert isinstance(remote.sequencer, na.RemoteSequencer)
    assert remote.sequencer.host == "192.168.0.21"
    assert remote.sequencer.port == 18862
    assert remote.sequencer.channels == hardware_channels
    assert remote.sequencer.clock_hz == 50_000_000
    assert remote.sequencer.trigger_channels == ("ch11",)
    assert remote.sequencer.snapshot()["connected"] is False

    exp = na.connect("remote_template", sequencer={"host": "192.168.0.22"})
    assert exp.devices.sequencer.host == "192.168.0.22"
    assert exp.camera.dcam_module_name == na.DEFAULT_DCAM_MODULE
    assert exp.sequence.channels == ["ch00", "ch03", "ch09", "ch11"]

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
        channels = ["aux"]
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
        channels = ["trap", "cooling", "probe", "emCCD"]
        clock_hz = 50_000_000
        trigger_channels = ("emCCD",)

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
