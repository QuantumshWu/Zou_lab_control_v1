import matplotlib

matplotlib.use("Agg")

import json
from pathlib import Path
import re
import socket
import subprocess
import time
import sys
import threading
import types

import numpy as np
from matplotlib.patches import Circle

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.devices.legacy_address_switch import legacy_imaging_parameters, write_vivado_vio_tcl
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
    assert "probe_out4 zlc_prog_tick" in top
    assert "zlc_set_probe $vio $zlc_prog_count_probe" in tcl
    assert "zlc_set_probe $vio $zlc_prog_tick_probe" in tcl
    assert "zlc_set_probe $vio $zlc_prog_mask_probe" in tcl
    assert "Available probes on matched VIO:" in tcl
    assert "Vivado project not found" in tcl
    assert "Vivado probe file not found" in tcl
    assert "Vivado bitstream not found for programming" in tcl
    assert "set zlc_reset_probe {zlc_reset probe_out0}" in tcl
    assert "set zlc_prog_tick_probe {zlc_prog_tick probe_out4}" in tcl
    assert "set zlc_done_probe {zlc_done probe_in1}" in tcl
    assert "string match \"*/$name\"" in tcl
    assert "probe aliases" in tcl
    assert f"with {len(program.ticks)} edges" in tcl


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
    assert "zlc_set_probe $vio $zlc_start_probe 1" in tcl
    assert "ZLC pulse-streamer start edge sent" in tcl


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


def test_fpga_pulse_streamer_repo_vivado_entrypoint_contract():
    root = Path(__file__).resolve().parents[1]
    fpga = root / "fpga" / "pulse_streamer"
    required = [
        "zlc_pulse_streamer.v",
        "zlc_pulse_streamer_top_4ch.v",
        "zlc_pulse_streamer_top_40ch.v",
        "zlc_pulse_streamer_4ch.xdc",
        "zlc_pulse_streamer_40ch.xdc.template",
        "create_project_4ch.tcl",
        "create_project_40ch.tcl",
        "check_40ch_synth.tcl",
        "program_fpga_4ch.tcl",
        "program_fpga_40ch.tcl",
        "build_4ch_bitstream.bat",
        "program_4ch_fpga.bat",
        "build_40ch_bitstream.bat",
        "program_40ch_fpga.bat",
        "check_40ch_synth.bat",
        "start_server_4ch.bat",
        "start_server_40ch.bat",
        "vivado_env.bat",
        "vivado_run_tcl.bat",
        "tb_zlc_pulse_streamer_4ch.v",
        "simulate_4ch_core.bat",
        "smoke_test_4ch.py",
        "smoke_test_4ch_upload.bat",
        "README.md",
    ]
    for name in required:
        assert (fpga / name).exists(), name

    top4 = (fpga / "zlc_pulse_streamer_top_4ch.v").read_text(encoding="utf-8")
    top40 = (fpga / "zlc_pulse_streamer_top_40ch.v").read_text(encoding="utf-8")
    tcl4 = (fpga / "create_project_4ch.tcl").read_text(encoding="utf-8")
    tcl40 = (fpga / "create_project_40ch.tcl").read_text(encoding="utf-8")
    check40 = (fpga / "check_40ch_synth.tcl").read_text(encoding="utf-8")
    program_tcl4 = (fpga / "program_fpga_4ch.tcl").read_text(encoding="utf-8")
    build_bat = (fpga / "build_4ch_bitstream.bat").read_text(encoding="utf-8")
    program_bat = (fpga / "program_4ch_fpga.bat").read_text(encoding="utf-8")
    vivado_env_bat = (fpga / "vivado_env.bat").read_text(encoding="utf-8")
    vivado_run_bat = (fpga / "vivado_run_tcl.bat").read_text(encoding="utf-8")
    server40_bat = (fpga / "start_server_40ch.bat").read_text(encoding="utf-8")
    sim4_bat = (fpga / "simulate_4ch_core.bat").read_text(encoding="utf-8")
    sim4_tb = (fpga / "tb_zlc_pulse_streamer_4ch.v").read_text(encoding="utf-8")
    smoke_bat = (fpga / "smoke_test_4ch_upload.bat").read_text(encoding="utf-8")
    xdc4 = (fpga / "zlc_pulse_streamer_4ch.xdc").read_text(encoding="utf-8")
    core = (fpga / "zlc_pulse_streamer.v").read_text(encoding="utf-8")

    assert '(* ram_style = "distributed" *)' in core
    assert ".probe_out3(zlc_prog_addr)" in top4
    assert ".probe_out4(zlc_prog_tick)" in top4
    assert ".probe_out5(zlc_prog_mask)" in top4
    assert ".probe_out6(zlc_prog_count)" in top4
    assert "wire [3:0] zlc_prog_mask" in top4
    assert "wire [39:0] zlc_prog_mask" in top40
    assert "CONFIG.C_NUM_PROBE_IN {2}" in tcl4
    assert "CONFIG.C_NUM_PROBE_OUT {7}" in tcl4
    assert "CONFIG.C_PROBE_OUT3_WIDTH {10}" in tcl4
    assert "CONFIG.C_PROBE_OUT4_WIDTH {32}" in tcl4
    assert "CONFIG.C_PROBE_OUT5_WIDTH {4}" in tcl4
    assert "CONFIG.C_PROBE_OUT6_WIDTH {11}" in tcl4
    assert "zlc_require_run_complete synth_1" in tcl4
    assert "zlc_require_run_complete impl_1" in tcl4
    assert "VIO probe file was not generated" in tcl4
    assert "CONFIG.C_PROBE_OUT5_WIDTH {40}" in tcl40
    assert "still contains <PIN_CHxx> placeholders" in tcl40
    assert "CONFIG.C_PROBE_OUT5_WIDTH {40}" in check40
    assert "ZLC 40ch synth check complete" in check40
    assert "ZLC_PS_VIVADO_BIT" in program_tcl4
    assert "ZLC_PS_VIVADO_LTX" in program_tcl4
    assert "VIO probe file not found" in program_tcl4
    assert "2019.1 2019.2" in vivado_env_bat
    assert "C:\\Xilinx\\Vivado\\%%V\\bin\\vivado.bat" in vivado_env_bat
    assert "subst !ZLC_SHORT_DRIVE!" in vivado_run_bat
    assert "--help" in build_bat
    assert "create_project_4ch.tcl" in build_bat
    assert "exit /b %ZLC_STATUS%" in build_bat
    assert "exit /b %ZLC_STATUS%" in program_bat
    assert "ZLC_PS_CHANNEL_COUNT=40" in server40_bat
    assert "ch00 ch01 ch02 ch03" in server40_bat
    assert "xsim" in sim4_bat
    assert "ZLC_SIM_PASS" in sim4_tb
    assert "zlc_pulse_streamer_top_4ch.bit" in smoke_bat
    assert "zlc_pulse_streamer_top_4ch.ltx" in smoke_bat
    assert "ZLC_PS_CHANNEL_COUNT=4" in smoke_bat
    assert "exit /b %ZLC_STATUS%" in smoke_bat

    for port, pin in {
        "clk": "R4",
        "trap": "M17",
        "cooling": "F15",
        "probe": "N15",
        "qcm_trigger": "R17",
    }.items():
        assert f"PACKAGE_PIN {pin}" in xdc4
        assert f"[get_ports {port}]" in xdc4


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

    assert "zlc_set_probe $vio $zlc_prog_count_probe 1024" in tcl
    assert "zlc_set_probe $vio $zlc_prog_addr_probe 1023" in tcl
    assert "zlc_set_probe $vio $zlc_prog_tick_probe 1023" in tcl
    assert "zlc_set_probe $vio $zlc_prog_mask_probe 0" in tcl
    assert tcl.count("zlc_set_probe $vio $zlc_prog_we_probe 1") == 1024
    assert tcl.count("zlc_set_probe $vio $zlc_prog_we_probe 0") == 1025


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

    assert program.ticks == [0, 1, 2, 5, 6, 7]
    assert program.masks == [0b0001, 0b1001, 0b0000, 0b0001, 0b1001, 0b0000]
    assert program.masks[-1] == 0
    assert program.trigger_count == 2


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
    program_x = state.compile(clock_hz=100e6, trigger_channels=["qcm_trigger"], x_ns=200)
    assert program_x.trigger_count == 3
    assert program_x.ticks[-1] == 160
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


def test_checked_in_camera_imaging_pulse_compiles_for_40ch_fpga():
    path = Path(__file__).resolve().parents[1] / "pulses" / "camera_imaging_40ch.json"
    state = na.PulseTableState.load(path)
    program = state.compile(clock_hz=100_000_000, trigger_channels=["ch03"])

    assert state.channels == [f"ch{i:02d}" for i in range(40)]
    assert state.visible_channels == ["ch00", "ch01", "ch02", "ch03"]
    assert state.channel_labels == {"ch00": "trap", "ch01": "cooling", "ch02": "probe", "ch03": "qcm_trigger"}
    assert state.delay_steps("ch00", time_step_ns=10) == 0
    assert state.delay_steps("ch03", time_step_ns=10) == 0
    assert state.repeat_start is None
    assert state.repeat_end is None
    assert state.repeat_count == 1
    assert program.channels == state.channels
    assert program.ticks == [0, 200_000, 210_000, 212_000, 2_210_000, 2_212_000]
    assert program.masks == [0b0011, 0b0001, 0b1101, 0b0101, 0b0001, 0]
    assert program.trigger_count == 1
    na.validate_pulse_streamer_program(program, max_edges=1024, tick_width=32, channel_count=40)


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


def test_fpga_pulse_streamer_smoke_test_script_writes_valid_program(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "fpga" / "pulse_streamer" / "smoke_test_4ch.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--state-dir",
            str(tmp_path),
            "--write-only",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout
    assert "expected outputs at 100 MHz" in result.stdout
    payload = json.loads((tmp_path / "fpga_4ch_smoke_program.json").read_text(encoding="utf-8"))
    program = na.RuntimeSequenceProgram.from_dict(payload)
    na.validate_pulse_streamer_program(program, max_edges=1024, tick_width=32, channel_count=4)
    assert program.channels == ["trap", "cooling", "probe", "qcm_trigger"]
    assert program.trigger_count == 1


def _simulate_pulse_streamer(ticks, masks):
    """Small behavioral model of the HDL run loop after the start edge is accepted."""

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


def test_legacy_address_switch_extracts_probe_lasting_and_writes_tcl(tmp_path, monkeypatch):
    monkeypatch.setenv("ZLC_LEGACY_SINGLE_CAMERA_TRIGGER_CONFIRMED", "1")
    seq = na.sequence_for_frame_count(na.imaging_sequence(exposure=4e-6, load=True), 5)
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "qcm_trigger"], clock_hz=100e6)

    params = legacy_imaging_parameters(program, clock_hz=100e6)
    tcl_path = write_vivado_vio_tcl(tmp_path / "prepare.tcl", params, project="", bitstream="", probes="")

    assert params == {"pulse_lasting": 400, "cycle_counts": 5}
    tcl = tcl_path.read_text(encoding="utf-8")
    assert 'set vio_filter {CELL_NAME=~"*vio*"}' in tcl
    assert "Available VIO cores:" in tcl
    assert "set_vio_probe $vio {pulse_lasting} {400}" in tcl
    assert "set_vio_probe $vio {cycle_counts} {5}" in tcl


def test_legacy_address_switch_prepare_requires_trigger_confirmation(monkeypatch):
    from Zou_lab_control.neutral_atom.devices.legacy_address_switch import build_assignments

    monkeypatch.delenv("ZLC_LEGACY_SINGLE_CAMERA_TRIGGER_CONFIRMED", raising=False)
    seq = na.sequence_for_frame_count(na.imaging_sequence(exposure=4e-6, load=True), 2)
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "qcm_trigger"], clock_hz=100e6)

    try:
        build_assignments("prepare", program)
    except RuntimeError as exc:
        assert "exactly one positive edge per cycle" in str(exc)
    else:
        raise AssertionError("legacy backend should require qCMOS trigger confirmation")


def test_legacy_address_switch_missing_vivado_writes_clear_log(tmp_path, monkeypatch):
    from Zou_lab_control.neutral_atom.devices.legacy_address_switch import run_action

    monkeypatch.setenv("ZLC_LEGACY_SINGLE_CAMERA_TRIGGER_CONFIRMED", "1")
    seq = na.imaging_sequence(exposure=4e-6, load=True)
    program = na.compile_runtime_program(seq, channels=["trap", "cooling", "probe", "qcm_trigger"], clock_hz=100e6)
    program_path = tmp_path / "program.json"
    program_path.write_text(json.dumps(program.to_dict()), encoding="utf-8")

    try:
        run_action("prepare", program_path=program_path, state_dir=tmp_path, vivado="zlc_missing_vivado_executable")
    except RuntimeError as exc:
        assert "could not start Vivado" in str(exc)
    else:
        raise AssertionError("missing Vivado executable should fail clearly")

    log_text = (tmp_path / "legacy_address_switch_prepare.log").read_text(encoding="utf-8")
    assert "Vivado executable was not found" in log_text
    assert "ZLC_VIVADO_BIN" in log_text


def test_legacy_address_switch_wait_done_resets_config_ready():
    from Zou_lab_control.neutral_atom.devices.legacy_address_switch import build_assignments

    assert build_assignments("fire", None) == {"config_ready": 1}
    assert build_assignments("wait_done", None) == {"config_ready": 0}
    assert build_assignments("safe_state", None) == {"config_ready": 0, "debug": 0}


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

    assert isinstance(manual.camera, na.QCMOSCamera)
    assert manual.camera.dcam_module_name == na.DEFAULT_DCAM_MODULE
    assert isinstance(manual.sequencer, na.ManualSequencer)
    assert isinstance(remote.camera, na.QCMOSCamera)
    assert remote.camera.dcam_module_name == na.DEFAULT_DCAM_MODULE
    assert isinstance(remote.sequencer, na.RemoteSequencer)
    assert remote.sequencer.host == "192.168.0.21"
    assert remote.sequencer.port == 18862
    assert remote.sequencer.snapshot()["connected"] is False

    exp = na.connect("remote_template", sequencer={"host": "192.168.0.22"})
    assert exp.devices.sequencer.host == "192.168.0.22"
    assert exp.camera.dcam_module_name == na.DEFAULT_DCAM_MODULE

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
