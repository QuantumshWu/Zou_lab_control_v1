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


def _decode_axi_writes(text: str) -> list[tuple[int, int]]:
    """Decode ``create_hw_axi_txn ... -type write`` transactions (single-beat OR INCR
    burst) into ``(word_addr, value)`` pairs -- the mock-hardware counterpart of
    ``axi_session._write_burst_tcl`` (burst ``-data`` is one concatenated hex value,
    high-address word first, so we reverse to put the base-address word first)."""

    out: list[tuple[int, int]] = []
    for addr_hex, data_hex, n_str in re.findall(
        r"-address ([0-9A-Fa-f]+) -data ([0-9A-Fa-f]+) -len (\d+) -type write", text
    ):
        base = int(addr_hex, 16) // 4
        n = int(n_str)
        words = [int(data_hex[i * 8:(i + 1) * 8], 16) for i in range(n)]
        words.reverse()
        for k, value in enumerate(words):
            out.append((base + k, value))
    return out


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
        # the FINAL single design (1-tick FIFO prefetch + 2-bank streaming scan),
        # driven over JTAG-to-AXI -- the ONLY build target (no variants).
        "zlc_edge_streamer.v",
        "zlc_pulse_streamer_top.v",
        "create_project.tcl",
        "program_fpga.tcl",
        "diagnose_hw_target.tcl",
        "README.md",
    }
    present = {path.name for path in fpga.iterdir()}
    assert required.issubset(present), required - present
    # NO legacy residue: the old LUTRAM/VIO engine, the depth-1 'D' engine, the
    # on-chip AXI loader, and the VIO address-switch top + all their tcl are GONE.
    assert present.isdisjoint({
        "zlc_pulse_streamer.v", "zlc_pulse_streamer_d.v", "zlc_pulse_streamer_d_top.v",
        "zlc_axi_program_loader.v", "zlc_pulse_streamer_loader_top.v",
        "zlc_pulse_streamer_top_address_switch.v",
        "create_project_d.tcl", "create_project_loader.tcl", "create_project_address_switch.tcl",
        "program_fpga_d.tcl", "program_fpga_loader.tcl", "program_fpga_address_switch.tcl",
        "check_address_switch_synth.tcl",
        "zlc_pulse_streamer_runlength.v", "zlc_runlength_engine.v",
    }), "legacy residue HDL/tcl still present in fpga/pulse_streamer"
    legacy_width = "40" + "ch"
    legacy_xdc_env = "ZLC_PS_" + legacy_width.upper() + "_XDC"
    assert not list((root / "docs").rglob(f"*{legacy_width}*"))

    top = (fpga / "zlc_pulse_streamer_top.v").read_text(encoding="utf-8")
    engine = (fpga / "zlc_edge_streamer.v").read_text(encoding="utf-8")
    create_tcl = (fpga / "create_project.tcl").read_text(encoding="utf-8")
    program_tcl = (fpga / "program_fpga.tcl").read_text(encoding="utf-8")
    build_bat = (root / "fpga" / "build_and_program.bat").read_text(encoding="utf-8")
    server_bat = (root / "fpga" / "run_server.bat").read_text(encoding="utf-8")
    launcher = (root / "pulse_gui.py").read_text(encoding="utf-8")
    preset = root / "pulses" / "camera_imaging_address_switch.json"

    # final top: 62-pin board map + the FINAL engine instance + forced-latency build.
    assert "module zlc_pulse_streamer_top" in top
    assert "parameter integer CHANNEL_COUNT = 62" in top
    assert "assign trig = out[6];" in top
    assert "assign trap = out[9];" in top
    assert "assign probe = out[3];" in top
    assert "assign cooling = out[0];" in top
    assert "zlc_edge_streamer" in top                      # instantiates the final engine
    assert "module zlc_edge_streamer" in engine
    # SHORT project name "ps" (-> ps.runs) keeps Vivado's deep run/.Xil temp path
    # under Windows MAX_PATH while staying in fpga/build (no out-of-repo relocation,
    # no extra drive/junction).
    assert "set project_name ps" in create_tcl
    assert "set project_name pulse_streamer" not in create_tcl
    assert "zlc_edge_streamer.v" in create_tcl and "zlc_pulse_streamer_top.v" in create_tcl
    assert "zlc_force_latency2" in create_tcl              # forced edge-BRAM read latency 2
    assert "ZLC_PS_XDC" in create_tcl
    assert legacy_xdc_env not in create_tcl
    # BRAM IP enable-pin contract (the build is blind -- no Verilog sim in CI). The
    # top drives BOTH .ena and .enb on every BRAM, so each IP must expose BOTH
    # enable pins. A drift to Enable_B {Always_Enabled} drops the enb port and synth
    # dies with "named port connection 'enb' does not exist". Keep ena/enb symmetric.
    assert ".ena(" in top and ".enb(" in top
    assert "Always_Enabled" not in create_tcl
    assert create_tcl.count("Enable_A {Use_ENA_Pin}") == 5   # all 5 BRAMs expose ENA
    assert create_tcl.count("Enable_B {Use_ENB_Pin}") == 5   # ...and ENB (top drives both)
    # AXI4 BURST upload path: jtag_axi + axi_bram_ctrl are FULL AXI4 (not Lite), and the
    # top wires the INCR-burst sidebands -- so one create_hw_axi_txn -len N moves up to
    # 256 words and a few-thousand-word BRAM upload drops from seconds to ~100 ms.  A
    # drift back to AXI4-Lite (single beat) would silently make uploads slow again.
    assert "CONFIG.PROTOCOL {AXI4}" in create_tcl
    assert "CONFIG.PROTOCOL {AXI4LITE}" not in create_tcl
    assert "m_axi_awlen" in top and "m_axi_awburst" in top and "m_axi_wlast" in top
    assert ".s_axi_awlen(" in top and ".s_axi_awburst(" in top and ".s_axi_wlast(" in top
    # additive-delay repeat: the engine rewinds repeat_forever to the steady frame
    # (loop_start), not edge 0, so the real-startup preamble plays exactly once.
    assert "repeat_from_loop_start" in engine and "repeat_from_loop_start" in top
    assert "set top zlc_pulse_streamer_top" in program_tcl
    assert "ps.runs" in program_tcl
    assert "pulse_streamer.runs" not in program_tcl

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
    assert 'set "ZLC_PROJ_SUB=ps"' in build_bat
    assert r'set "ZLC_PS_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\!ZLC_PROJ_SUB!"' in build_bat
    assert legacy_xdc_env not in build_bat
    # Long-path safety: the build stays IN fpga/build (no out-of-repo relocation,
    # no extra drive); the short "ps" project name keeps Vivado's deep run/.Xil
    # temp path under the limit. Both bats default the project dir under fpga/build,
    # and the tcl guard fails clearly (advising a shorter CHECKOUT) if even that is
    # too long.
    assert "zlc_safe_project_dir" in create_tcl
    assert "LOCALAPPDATA" not in build_bat and "LOCALAPPDATA" not in server_bat
    assert r"%FPGA_DIR%build" in build_bat                # build root stays in fpga\build
    assert r"%FPGA_DIR%build" in server_bat
    # run_server.bat starts the FINAL JTAG-to-AXI server (no loader/variant residue).
    assert "ZLC_PS_CLOCK_HZ=50000000" in server_bat
    assert "zlc_verify_sources" in server_bat
    assert "zlc_verify_loader_sources" not in server_bat
    assert "ZLC_PS_SERVER_BACKEND=jtag-axi" in server_bat
    assert "zlc_pulse_streamer_top.ltx" in server_bat
    assert "zlc_pulse_streamer_loader_top" not in server_bat
    assert "ZLC_PS_VARIANT" not in server_bat
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
    # The prepare/fire/wait_done/safe_state lifecycle lives in the maintainer note
    # + main manual (design-agnostic host contract).
    assert "prepare" in maintainer_notes and "fire" in maintainer_notes
    assert "wait_done" in maintainer_notes and "safe_state" in maintainer_notes
    assert "prepare / fire / wait\\_done / safe\\_state" in main_manual_template
    # Capacity is now fixed by the host image solver (no per-build env knob); the
    # source of truth is named in the maintainer note + fpga README.
    assert "solve_capacity" in maintainer_notes
    assert "solve_capacity" in fpga_readme
    # The FINAL transport (JTAG-to-AXI) anchors the new root README and the
    # maintainer note (replacing the old VIO/address-switch control keyword).
    assert "JTAG-to-AXI" in root_readme
    assert "JTAG-to-AXI" in maintainer_notes
    assert "Run the smallest check" in tests_readme
    assert "Full `pytest -q` is reserved for broad handoff" in tests_readme

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
            "stop_value_select": 0,
            "start_tick_coeffs": [],
            "stop_tick_coeffs": [],
        }
    ]
    roundtrip = na.RuntimeSequenceProgram.from_dict(program.to_dict())
    assert [segment.to_dict() for segment in (roundtrip.bus_segments or [])] == [
        segment.to_dict() for segment in (program.bus_segments or [])
    ]
    na.validate_pulse_streamer_program(program, max_edges=16, max_bus_segments=4, tick_width=32, channel_count=4)


def test_analog_ramp_can_scan_both_value_endpoints_round_trip():
    """R16: a ramp may scan BOTH value endpoints independently -- the start reads one
    scan slot, the stop another -- via the dual value_select.  The host image
    round-trips both selects and the validator accepts them (no longer rejected)."""

    from fpga.pulse_streamer.host.image import StreamerParams, pack_program, unpack_program
    from Zou_lab_control.neutral_atom.devices.sequencer import RuntimeSequenceProgram, RuntimeBusSegment

    prog = RuntimeSequenceProgram(
        sequence_id="r", sequence_name="r", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 50, 200], masks=[0, 1, 0], duration=4e-6, trigger_count=0,
        repeat_forever=False, loop_start_index=0, loop_end_tick=200, loop_count=1,
        slot_count=2, slot_kinds=["dac", "dac"], loop_end_slot_coeffs=[0, 0],
        tick_slot_coeffs=[[0, 0], [0, 0], [0, 0]],
        scan_points=[[100, 900], [200, 800]], scan_coeff_frac_bits=8,
        bus_names=["da0"],
        bus_segments=[
            RuntimeBusSegment(bus_index=0, start_tick=50, stop_tick=120, start_value=0, stop_value=0,
                              mode="ramp", value_select=1, stop_value_select=2,
                              start_tick_coeffs=[0, 0], stop_tick_coeffs=[0, 0]),
        ],
    )
    params = StreamerParams(max_edges=16, bank_size=4)
    seg = unpack_program(pack_program(prog, params), params)["bus_segments"][0]
    assert seg["mode"] == "ramp"
    assert seg["value_select"] == 1 and seg["stop_value_select"] == 2   # both endpoints scanned
    na.validate_pulse_streamer_program(prog, max_edges=16, max_bus_segments=4, tick_width=32,
                                       channel_count=62, num_slots=2)
    # round-trip through the program dict preserves both selects too
    rseg = na.RuntimeSequenceProgram.from_dict(prog.to_dict()).bus_segments[0]
    assert rseg.value_select == 1 and rseg.stop_value_select == 2


def test_bus_play_models_dual_endpoint_ramp_and_held_value():
    """Cycle-accurate bus-engine model (engine_model.bus_play) proves the dual
    value_select path end-to-end: a ramp scans BOTH endpoints (start slot A -> stop
    slot B) and an edge/hold segment tracks its scanned slot -- the bus-path
    counterpart of the edge rtl_mirror proof (closes the no-bus-model gap)."""

    from fpga.pulse_streamer.host.engine_model import bus_play
    from Zou_lab_control.neutral_atom.devices.sequencer import RuntimeSequenceProgram, RuntimeBusSegment

    def prog(segs, points):
        return RuntimeSequenceProgram(
            sequence_id="b", sequence_name="b", clock_hz=50e6,
            channels=[f"ch{i:02d}" for i in range(62)],
            ticks=[0, 5, 200], masks=[0, 1, 0], duration=4e-6, trigger_count=0,
            repeat_forever=False, loop_start_index=0, loop_end_tick=200, loop_count=1,
            slot_count=2, slot_kinds=["dac", "dac"], loop_end_slot_coeffs=[0, 0],
            tick_slot_coeffs=[[0, 0], [0, 0], [0, 0]], scan_points=points, scan_coeff_frac_bits=8,
            bus_names=["da0"], bus_segments=segs)

    ramp = RuntimeBusSegment(bus_index=0, start_tick=10, stop_tick=70, start_value=0, stop_value=0,
                             mode="ramp", value_select=1, stop_value_select=2,
                             start_tick_coeffs=[0, 0], stop_tick_coeffs=[0, 0])
    p = prog([ramp], [[100, 900], [900, 100]])
    up = bus_play(p, 0, 100, scan_point=0)        # ramp scanned-A(100) -> scanned-B(900)
    assert up == sorted(up)                        # non-decreasing 0 -> 100 -> ... -> 900
    assert 100 in up and max(up) == 900 and up[-1] == 900
    down = bus_play(p, 0, 100, scan_point=1)       # ramp scanned-A(900) -> scanned-B(100)
    assert max(down) == 900 and down[-1] == 100
    s = down.index(900)
    assert down[s:] == sorted(down[s:], reverse=True)   # non-increasing 900 -> 100

    hold = RuntimeBusSegment(bus_index=0, start_tick=10, stop_tick=10, start_value=0, stop_value=0,
                             mode="edge", value_select=1, stop_value_select=1,
                             start_tick_coeffs=[0, 0], stop_tick_coeffs=[0, 0])
    ph = prog([hold], [[300, 0], [700, 0]])
    assert bus_play(ph, 0, 40, scan_point=0)[-1] == 300   # held DAC value tracks the scan slot
    assert bus_play(ph, 0, 40, scan_point=1)[-1] == 700


def test_edge_streamer_has_dual_value_select():
    """The RTL bus engine has the dual start/stop value_select path (a ramp can read
    a different scan slot for each endpoint)."""

    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "fpga" / "pulse_streamer"
    eng = (root / "zlc_edge_streamer.v").read_text(encoding="utf-8")
    top = (root / "zlc_pulse_streamer_top.v").read_text(encoding="utf-8")
    assert "bus_stop_value_select_mem" in eng
    assert "bus_prog_stop_value_select" in eng and "bus_prog_stop_value_select" in top
    assert "start_sel" in eng and "stop_sel" in eng        # independent endpoint reads


def test_pulse_table_scan_allows_analog_bus_ramp_with_timing_scan():
    """A fixed-endpoint analog ramp combined with a scanned DURATION compiles: the
    ramp's start/stop ticks are emitted as affine expressions so the ramp stretches
    in lockstep with the scanned timing (no longer rejected)."""

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
    # ... while the analog bus ramps between FIXED value endpoints: now supported.
    state.set_analog_bus_mode(0, "da_test", "edge", value=0)
    state.set_analog_bus_mode(1, "da_test", "ramp", value=3)

    program = na.compile_pulse_table_scan_runtime_program(
        state, channels=["ch00", "ch01"], clock_hz=50_000_000
    )
    ramps = [s for s in (program.bus_segments or []) if s.mode == "ramp"]
    assert ramps, "expected a ramp segment"
    r = ramps[0]
    # fixed value endpoints (no scanned-endpoint value_select) ...
    assert r.start_value == 0 and r.stop_value == 3 and r.value_select == 0
    # ... but affine ticks: the scanned-duration slot moves a ramp endpoint tick so
    # the ramp stretches in lockstep with the scan.
    assert any(c != 0 for c in (list(r.start_tick_coeffs or []) + list(r.stop_tick_coeffs or []))), \
        "a ramp endpoint tick must be affine under the scan"


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


def test_final_status_bits_match_host():
    """The top STATUS bit map MUST equal host.image STATUS_*; in particular the
    streaming UNDERFLOW bit must be a DISTINCT bit from the host's fatal ERROR bit,
    else a recoverable streaming stall would crash the host (regression guard for
    the ST_UNDERFLOW=8 vs STATUS_ERROR=8 collision)."""

    import pathlib, re
    from fpga.pulse_streamer.host import image as im
    top = (pathlib.Path(__file__).resolve().parents[1] / "fpga" / "pulse_streamer" / "zlc_pulse_streamer_top.v").read_text(encoding="utf-8")
    m = re.search(r"ST_LOADED = 5'd(\d+), ST_RUNNING = 5'd(\d+), ST_DONE = 5'd(\d+), ST_UNDERFLOW = 5'd(\d+);", top)
    assert m, "could not find ST_* localparams in the top"
    loaded, running, done, under = (int(g) for g in m.groups())
    assert (loaded, running, done, under) == (im.STATUS_LOADED, im.STATUS_RUNNING, im.STATUS_DONE, im.STATUS_UNDERFLOW)
    assert (loaded, running, done, under) == (1, 2, 4, 16)
    # the recoverable stall bit must NOT collide with the host's fatal ERROR bit.
    for v in (loaded, running, done, under):
        assert v != im.STATUS_ERROR, "a STATUS bit collides with host STATUS_ERROR (bit 3)"


def test_vivado_axi_session_tolerates_transient_underflow(tmp_path):
    """STATUS_UNDERFLOW (bit 4) is a transient streaming stall, a DISTINCT bit from
    the fatal STATUS_ERROR (bit 3).  wait_done must keep polling and complete on the
    later DONE -- it must NEVER raise on an underflow."""

    import re as _re
    from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession
    from fpga.pulse_streamer.host.image import (
        StreamerParams, CtrlWords, STATUS_RUNNING, STATUS_DONE, STATUS_UNDERFLOW, STATUS_LOADED, CMD_LOAD, CMD_FIRE,
    )
    from Zou_lab_control.neutral_atom.devices.sequencer import RuntimeSequenceProgram

    params = StreamerParams(max_edges=16, bank_size=4)
    program = RuntimeSequenceProgram(
        sequence_id="u", sequence_name="u", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 100, 200], masks=[1, 0, 0], duration=4e-6, trigger_count=0,
        repeat_forever=False, loop_start_index=0, loop_end_tick=200, loop_count=1, slot_count=0,
    )

    class Hw:
        def __init__(self):
            self.bram = {}; self.status = 0; self.fired = False; self.polls = 0
        def __call__(self, lines, action, timeout):
            text = "\n".join(lines)
            for w, v in _decode_axi_writes(text):
                self.bram[w] = v
                if w == CtrlWords.COMMAND and v & CMD_LOAD: self.status = STATUS_LOADED
                if w == CtrlWords.COMMAND and v & CMD_FIRE: self.fired = True; self.status = STATUS_RUNNING
            m = _re.search(r"-address ([0-9A-Fa-f]+) -len 1 -type read", text)
            if m:
                w = int(m.group(1), 16) // 4
                if w == CtrlWords.STATUS:
                    if self.fired:
                        self.polls += 1
                        if self.polls <= 3:   # transient stall: RUNNING + UNDERFLOW
                            return f"ZLCDATA {STATUS_RUNNING | STATUS_UNDERFLOW:08X}\n"
                        return f"ZLCDATA {STATUS_RUNNING | STATUS_DONE:08X}\n"
                    return f"ZLCDATA {self.status:08X}\n"
                return f"ZLCDATA {self.bram.get(w, 0):08X}\n"
            return "ok\n"

    hw = Hw()
    session = VivadoAxiStreamerSession(state_dir=tmp_path, params=params, tcl_executor=hw)
    session.prepare(program); session.fire()
    assert session.wait_done(timeout=2.0) is True   # underflow tolerated, completes on DONE


def test_edge_streamer_repeat_streaming_structure():
    """Engine + top carry the bank_chunk handshake: the scan advance is gated on the
    bank holding the RIGHT chunk (never a stale point), and the repeat_forever wrap
    waits for chunk 0 to be reloaded -- so a finite streamed scan re-sweeps."""

    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "fpga" / "pulse_streamer"
    eng = (root / "zlc_edge_streamer.v").read_text(encoding="utf-8")
    top = (root / "zlc_pulse_streamer_top.v").read_text(encoding="utf-8")
    assert "bank_chunk0" in eng and "bank_chunk1" in eng
    assert "scan_point_resident" in eng                       # advance gated on ready AND right chunk
    assert "bank_chunk0 == {SCAN_COUNT_WIDTH{1'b0}}" in eng   # repeat wrap waits for chunk 0
    assert "C_BANK0_CHUNK" in top and "C_BANK1_CHUNK" in top
    assert "bank_chunk0(ctrl_reg[C_BANK0_CHUNK]" in top and "bank_chunk1(ctrl_reg[C_BANK1_CHUNK]" in top


def test_vivado_axi_session_rejects_dac_value_over_bus_width(tmp_path):
    """A scanned DAC code wider than bus_width (would silently truncate on hardware)
    is rejected at prepare."""

    from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession
    from fpga.pulse_streamer.host.image import StreamerParams
    from Zou_lab_control.neutral_atom.devices.sequencer import RuntimeSequenceProgram

    params = StreamerParams(max_edges=16, bank_size=4, bus_width=10)
    program = RuntimeSequenceProgram(
        sequence_id="dh", sequence_name="dh", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 50, 200], masks=[0, 1, 0], duration=4e-6, trigger_count=0,
        repeat_forever=False, loop_start_index=0, loop_end_tick=200, loop_count=1,
        slot_count=1, slot_kinds=["dac"], loop_end_slot_coeffs=[0],
        tick_slot_coeffs=[[0], [0], [0]], scan_points=[[100], [2000]], scan_coeff_frac_bits=8,  # 2000 > 1023
    )
    session = VivadoAxiStreamerSession(state_dir=tmp_path, params=params, tcl_executor=lambda *a: "ok\n")
    with pytest.raises(ValueError, match="does not fit 10 bits"):
        session.prepare(program)


def test_vivado_axi_session_wait_done_is_reentrant(tmp_path):
    """A finite streamed scan whose wait_done returns early (timeout) must RESUME on
    the next call -- it must NOT reload from chunk 2 over the bank that now holds a
    later chunk.  Each chunk is loaded exactly once, in order, across the two calls."""

    import re as _re
    from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession
    from fpga.pulse_streamer.host.image import (
        StreamerParams, CtrlWords, STATUS_RUNNING, STATUS_DONE, STATUS_LOADED, CMD_LOAD, CMD_FIRE,
    )
    from Zou_lab_control.neutral_atom.devices.sequencer import RuntimeSequenceProgram

    N = 10
    params = StreamerParams(max_edges=16, bank_size=2)   # 5 chunks; chunks 2,3,4 streamed
    program = RuntimeSequenceProgram(
        sequence_id="re", sequence_name="re", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 5, 40], masks=[0, 1, 0], duration=4e-6, trigger_count=0,
        repeat_forever=False, loop_start_index=0, loop_end_tick=40, loop_count=1,
        slot_count=1, slot_kinds=["delay"], loop_end_slot_coeffs=[0],
        tick_slot_coeffs=[[0], [256], [0]], scan_points=[[k] for k in range(N)], scan_coeff_frac_bits=8,
    )

    class Hw:
        def __init__(self):
            self.bram = {}; self.status = 0; self.fired = False; self.cursor = 0; self.cap = 0
            self.chunk_writes = []     # order of (BANK*_CHUNK) values written after fire
        def __call__(self, lines, action, timeout):
            text = "\n".join(lines)
            for w, v in _decode_axi_writes(text):
                self.bram[w] = v
                if w == CtrlWords.COMMAND and v & CMD_LOAD: self.status = STATUS_LOADED
                if w == CtrlWords.COMMAND and v & CMD_FIRE: self.fired = True; self.status = STATUS_RUNNING
                if self.fired and w in (CtrlWords.BANK0_CHUNK, CtrlWords.BANK1_CHUNK):
                    self.chunk_writes.append(v)
            m = _re.search(r"-address ([0-9A-Fa-f]+) -len 1 -type read", text)
            if m:
                w = int(m.group(1), 16) // 4
                if w == CtrlWords.CURSOR:
                    return f"ZLCDATA {self.cursor:08X}\n"
                if w == CtrlWords.STATUS:
                    if self.fired:               # advance toward the allowed cap, then DONE at N
                        self.cursor = min(self.cap, self.cursor + params.bank_size)
                        if self.cursor >= N:
                            self.status |= STATUS_DONE
                    return f"ZLCDATA {self.status:08X}\n"
                return f"ZLCDATA {self.bram.get(w, 0):08X}\n"
            return "ok\n"

    hw = Hw()
    session = VivadoAxiStreamerSession(state_dir=tmp_path, params=params, tcl_executor=hw)
    session.prepare(program)
    session.fire()
    hw.cap = 6                                   # only let the engine reach cursor 6 this call
    assert session.wait_done(timeout=0.3) is False   # not done yet; some chunks streamed
    partial = list(hw.chunk_writes)
    assert partial == sorted(partial) and partial[0] == 2   # loaded 2,3,... in order, none repeated
    hw.cap = N                                   # allow it to finish
    assert session.wait_done(timeout=1.0) is True
    # every streamed chunk loaded exactly once, strictly increasing across BOTH calls
    assert hw.chunk_writes == [2, 3, 4], hw.chunk_writes


def test_vivado_axi_session_rejects_nonmonotonic_program(tmp_path):
    """The host validates the program before upload (defence in depth): an affine
    scan that makes the effective edge ticks non-monotonic (an edge would overtake a
    later one and be silently dropped on hardware) is rejected at prepare, not
    uploaded."""

    from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession
    from fpga.pulse_streamer.host.image import StreamerParams
    from Zou_lab_control.neutral_atom.devices.sequencer import RuntimeSequenceProgram

    params = StreamerParams(max_edges=16, bank_size=4)
    # edge 1 has a large positive slot coeff, so at slot=1000 it lands at tick
    # 100+1000=1100, OVERTAKING the fixed edge 2 at 200 -> non-monotonic.
    program = RuntimeSequenceProgram(
        sequence_id="bad", sequence_name="bad", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 100, 200], masks=[0, 1, 0], duration=4e-6, trigger_count=0,
        repeat_forever=False, loop_start_index=0, loop_end_tick=200, loop_count=1,
        slot_count=1, slot_kinds=["delay"], loop_end_slot_coeffs=[0],
        tick_slot_coeffs=[[0], [256], [0]], scan_points=[[0], [1000]], scan_coeff_frac_bits=8,
    )
    session = VivadoAxiStreamerSession(state_dir=tmp_path, params=params, tcl_executor=lambda *a: "ok\n")
    with pytest.raises(ValueError, match="non-increasing"):
        session.prepare(program)


def test_vivado_axi_session_repeat_streaming_refills_cyclically(tmp_path):
    """repeat_forever over a FINITE STREAMED scan (N > 2*bank_size) re-sweeps: the
    background refill thread reloads chunk 0 (and 1) at each sweep seam, cyclically,
    keeping the engine fed across re-sweeps.  Never raises; safe_state stops it."""

    import re as _re, time as _time
    from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession
    from fpga.pulse_streamer.host.image import (
        StreamerParams, CtrlWords, STATUS_RUNNING, STATUS_LOADED, CMD_LOAD, CMD_FIRE, CMD_SAFE,
    )
    from Zou_lab_control.neutral_atom.devices.sequencer import RuntimeSequenceProgram

    N = 12
    params = StreamerParams(max_edges=16, bank_size=2)   # total_chunks = 6 (streamed)
    program = RuntimeSequenceProgram(
        sequence_id="rs", sequence_name="rs", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 5, 40], masks=[0, 1, 0], duration=4e-6, trigger_count=0,
        repeat_forever=True, loop_start_index=0, loop_end_tick=40, loop_count=1,
        slot_count=1, slot_kinds=["delay"], loop_end_slot_coeffs=[0],
        tick_slot_coeffs=[[0], [256], [0]], scan_points=[[k] for k in range(N)], scan_coeff_frac_bits=8,
    )

    class Hw:
        def __init__(self):
            self.bram = {}; self.status = 0; self.fired = False; self.cursor = 0; self.reloads0 = 0
        def __call__(self, lines, action, timeout):
            text = "\n".join(lines)
            for w, v in _decode_axi_writes(text):
                self.bram[w] = v
                if w == CtrlWords.COMMAND and v & CMD_LOAD: self.status = STATUS_LOADED
                if w == CtrlWords.COMMAND and v & CMD_FIRE: self.fired = True; self.status = STATUS_RUNNING; self.cursor = 0
                if w == CtrlWords.COMMAND and v & CMD_SAFE: self.status = 0; self.fired = False
                # the engine wraps once the host reloads chunk 0 at the sweep seam
                if w == CtrlWords.BANK0_CHUNK and v == 0 and self.cursor >= N:
                    self.reloads0 += 1; self.cursor = 0
            m = _re.search(r"-address ([0-9A-Fa-f]+) -len 1 -type read", text)
            if m:
                w = int(m.group(1), 16) // 4
                if w == CtrlWords.STATUS:
                    return f"ZLCDATA {self.status:08X}\n"
                if w == CtrlWords.CURSOR:
                    if self.fired and self.cursor < N:
                        self.cursor = min(N, self.cursor + params.bank_size)
                    return f"ZLCDATA {self.cursor:08X}\n"
                return f"ZLCDATA {self.bram.get(w, 0):08X}\n"
            return "ok\n"

    hw = Hw()
    session = VivadoAxiStreamerSession(state_dir=tmp_path, params=params, tcl_executor=hw)
    try:
        session.prepare(program)
        session.fire()
        assert session.wait_done(timeout=1.0) is True       # RUNNING -> returns; thread feeds
        deadline = _time.monotonic() + 3.0
        while hw.reloads0 < 2 and _time.monotonic() < deadline:
            _time.sleep(0.02)
    finally:
        session.safe_state()
    assert hw.reloads0 >= 2, f"expected the streamed scan to re-sweep cyclically, got {hw.reloads0}"


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


def _rtl_bus_held_value(program, bus_index, tick, scan_point, *, bus_width=10):
    """Python re-implementation of the RTL bus engine's held DAC value.

    Faithfully mirrors ``zlc_bus_apply_segment`` / ``zlc_bus_seg_start`` in
    ``fpga/pulse_streamer/zlc_edge_streamer.v``: at a scan point the bus walks
    its segments in *effective*-tick order and holds the most recent one whose
    effective start tick <= ``tick``.  The effective tick applies the segment's
    affine coefficients to the current scan point (so a scanned duration moves
    the segment).  For an edge/hold segment the RTL holds ``vstop`` (the STOP
    endpoint), so a held value with ``stop_value_select = j+1`` reads the low
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
        # edge/hold holds vstop in the RTL -> use the STOP endpoint select (which for
        # an edge/hold segment equals value_select, since start==stop).
        sel = int(getattr(seg, "stop_value_select", getattr(seg, "value_select", 0)))
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

    assert program.ticks == [0, 1, 3, 4]
    assert program.masks == [0b0001, 0b1000, 0, 0]
    assert program.loop_start_index == 1
    assert program.loop_end_tick == 3
    assert program.loop_count == 2
    assert program.repeat_forever is True
    assert history[:7] == [0b0001, 0b1000, 0b1000, 0b1000, 0b1000, 0, 0b0001]


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


def test_pulse_table_snaps_times_to_minimal_grid():
    # The pulse-table (GUI) path must AUTO-SNAP off-grid durations to the nearest
    # tick instead of rejecting them -- the hardware clock can only land on ticks.
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
    # 25 ns at a 10 ns step snaps to 30 ns (ties away from zero) and stays valid.
    snapped = state.to_sequence(slots={"s0": 25}, time_step_ns=10)
    assert snapped.validate(clock_hz=100e6, channels=state.channels).ok
    assert state.periods[1].duration_ns(slots={"s0": 25}, time_step_ns=10) == 30.0

    # A duration that rounds toward zero must still snap UP to one tick (never 0).
    tiny = na.PulseTableState(
        channels=["trap"],
        periods=[na.PulsePeriod(2.5, (1,), unit="ns")],
        time_step_ns=1,
    )
    assert tiny.periods[0].duration_steps(time_step_ns=10) == 1  # 2.5 ns -> one 10 ns tick
    assert na.PulsePeriod(3.0, (1,), unit="ns").duration_ns(time_step_ns=1) == 3.0


def test_pulse_table_snapped_snaps_literals_and_keeps_expressions():
    from Zou_lab_control.neutral_atom.timing.pulse_table import ScanSlot, snap_scan_table

    state = na.PulseTableState(
        channels=["trap", "trig"],
        periods=[
            na.PulsePeriod(50, (1, 0), unit="ns"),    # off-grid -> 60 ns at 20 ns step
            na.PulsePeriod("s0", (0, 1), unit="str (ns)"),  # expression: must be kept
        ],
        delays={"trig": 30.0, "trap": "s0"},          # 30 -> 40; "s0" kept
        delay_units={"trig": "ns", "trap": "ns"},
        scan_slots=[{"kind": "duration", "target": "1", "unit": "ns", "nominal": 20.0}],
        scan_table=[[51.0], [9.0]],
        time_step_ns=20,
    )
    snapped = state.snapped()
    # literal duration snaps to a whole tick (50 -> 60), expression preserved
    assert snapped.periods[0].duration == 60
    assert snapped.periods[1].duration == "s0"
    # literal delay snaps (30 -> 40), expression delay preserved
    assert snapped.delays["trig"] == 40
    assert snapped.delays["trap"] == "s0"
    # scan table: time slot snaps to the tick grid
    assert snapped.scan_table[0][0] == 60.0   # 51 ns -> 60 ns
    assert snapped.scan_table[1][0] == 0.0    # 9 ns rounds to 0 (a delay slot may be 0)
    # the original state is untouched (snapped returns a copy)
    assert state.periods[0].duration == 50

    # snap_scan_table is the shared helper used by the GUI; DAC slots round to an
    # integer code, time slots snap to the tick grid.
    dac_slot = ScanSlot(kind="dac", target="da_dipole@0", unit="value")
    time_slot = ScanSlot(kind="delay", target="trig", unit="ns")
    snapped_rows = snap_scan_table([[51.0, 512.4], [9.0, 800.6]], [time_slot, dac_slot], time_step_ns=20)
    assert snapped_rows == [[60.0, 512.0], [0.0, 801.0]]


def test_timing_payload_to_dict_snaps_pulse_table():
    from Zou_lab_control.neutral_atom.devices.sequencer import timing_payload_to_dict

    state = na.PulseTableState(
        channels=["trap", "trig"],
        periods=[na.PulsePeriod(50, (1, 0), unit="ns"), na.PulsePeriod(120, (0, 1), unit="ns")],
        time_step_ns=20,
    )
    payload = timing_payload_to_dict(state)
    # the transferred pulse-API payload carries snapped whole-tick durations (50 -> 60)
    assert payload["periods"][0]["duration"] == 60
    assert payload["periods"][1]["duration"] == 120


def test_axi_session_burst_coalesces_contiguous_and_preserves_order(tmp_path):
    from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession

    s = VivadoAxiStreamerSession(state_dir=tmp_path, tcl_executor=lambda *a: "ok\n", burst_max=4)
    # an address-contiguous run is coalesced and split at burst_max (4): 6 words -> 4+2
    pending = [(0x40 + 4 * i, i) for i in range(6)]
    assert s._burst_runs(pending) == [(0x40, [0, 1, 2, 3]), (0x50, [4, 5])]
    # an order-dependent same-address sequence (COMMAND 0 then cmd) must NOT be merged
    # or reordered -- it stays two len-1 writes in order.
    assert s._burst_runs([(0x4, 0), (0x4, 2)]) == [(0x4, [0]), (0x4, [2])]
    # non-contiguous addresses stay separate len-1 writes
    assert s._burst_runs([(0x0, 9), (0x10, 8)]) == [(0x0, [9]), (0x10, [8])]
    # the burst Tcl encodes one INCR transaction whose data round-trips to the SAME
    # words at consecutive addresses (writer/decoder agree on the high-addr-first order)
    lines = s._write_burst_tcl(0x40, [0xAA, 0xBB, 0xCC])
    text = "\n".join(lines)
    assert "-len 3 -type write" in text and text.count("run_hw_axi") == 1
    assert _decode_axi_writes(text) == [(16, 0xAA), (17, 0xBB), (18, 0xCC)]


def test_axi_session_self_test_catches_scrambled_burst(tmp_path):
    from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession

    class Hw:
        def __init__(self, scramble=False):
            self.bram = {}; self.scramble = scramble
        def __call__(self, lines, action, timeout):
            text = "\n".join(lines)
            writes = _decode_axi_writes(text)
            if self.scramble and len(writes) > 1:   # simulate wrong burst data ordering
                addrs = [w for w, _ in writes]; vals = [v for _, v in writes]
                writes = list(zip(addrs, list(reversed(vals))))
            for w, v in writes:
                self.bram[w] = v
            m = re.search(r"-address ([0-9A-Fa-f]+) -len 1 -type read", text)
            if m:
                return f"ZLCDATA {self.bram.get(int(m.group(1), 16) // 4, 0):08X}\n"
            return "ok\n"

    good = VivadoAxiStreamerSession(state_dir=tmp_path, tcl_executor=Hw(scramble=False))
    assert good.axi_self_test(count=8) is True
    bad = VivadoAxiStreamerSession(state_dir=tmp_path, tcl_executor=Hw(scramble=True))
    with pytest.raises(RuntimeError):
        bad.axi_self_test(count=8)


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

        def axi_self_test(self, **kwargs):
            events.append("self_test")
            return True

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

    # warm start: construct -> start the Vivado session -> AXI burst self-test (fail-fast
    # bring-up check) -> serve.
    assert events[:4] == ["init:state_loader:clk=50000000", "start", "self_test", "serve:127.0.0.1:18861"]
    assert service is not None


class _FakeStreamerHardware:
    """In-memory stand-in for the programmed FPGA running the FINAL design: a BRAM
    dict + the CTRL COMMAND/STATUS/CURSOR/BANK_READY mailbox.  On LOAD it verifies
    the uploaded image round-trips through host.image.unpack_program (so the test
    exercises the full host->upload->decode path).  On FIRE it advances a CURSOR so
    the host's streaming refill loop runs; it records each BANK_READY write so the
    streaming handshake can be asserted.  ``forever`` => RUNNING but never DONE."""

    def __init__(self, params, *, forever=False, total_points=0):
        from fpga.pulse_streamer.host.image import CtrlWords
        self.params = params
        self.CtrlWords = CtrlWords
        self.forever = bool(forever)
        self.total_points = int(total_points)
        self.bram: dict[int, int] = {}
        self.status = 0
        self.load_ok = False
        self.fired = False
        self.cursor = 0
        self.bank_ready_writes: list[int] = []   # post-fire BANK_READY values

    def __call__(self, lines, action, timeout):
        from fpga.pulse_streamer.host.image import (
            unpack_program, CtrlWords, CMD_LOAD, CMD_FIRE, CMD_SAFE,
            STATUS_LOADED, STATUS_RUNNING, STATUS_DONE,
        )
        text = "\n".join(lines)
        for word, value in _decode_axi_writes(text):
            self.bram[word] = value
            if word == CtrlWords.BANK_READY and self.fired:
                self.bank_ready_writes.append(value)
            if word == CtrlWords.COMMAND and value != 0:
                if value & CMD_SAFE:
                    self.status = 0; self.load_ok = False
                if value & CMD_LOAD:
                    decoded = unpack_program(self.bram, self.params)
                    self.load_ok = (decoded["ticks"] and decoded["masks"]
                                    and len(decoded["ticks"]) == self.bram.get(CtrlWords.PROG_COUNT, 0))
                    self.status = STATUS_LOADED if self.load_ok else 0x8
                if value & CMD_FIRE:
                    self.fired = True; self.cursor = 0
                    self.status = (self.status | STATUS_RUNNING) & ~0x8
        m = re.search(r"-address ([0-9A-Fa-f]+) -len 1 -type read", text)
        if m:
            word = int(m.group(1), 16) // 4
            if word == CtrlWords.CURSOR:
                # report progress only; the engine (STATUS poll) drives the cursor.
                return f"ZLCDATA {self.cursor:08X}\n"
            if word == CtrlWords.STATUS:
                if self.fired and not self.forever:
                    if self.total_points:
                        # advance one bank per status poll (the engine's pace), giving
                        # the host a full bank to refill ahead -> gapless streaming.
                        self.cursor = min(self.total_points, self.cursor + self.params.bank_size)
                        if self.cursor >= self.total_points:
                            self.status |= STATUS_DONE
                    else:
                        self.status |= STATUS_DONE
                return f"ZLCDATA {self.status:08X}\n"
            return f"ZLCDATA {self.bram.get(word, 0):08X}\n"
        return "ok\n"


def test_vivado_axi_session_loads_and_fires_edge_table_program(tmp_path):
    """prepare/fire/wait_done/safe_state drive the CTRL COMMAND/STATUS mailbox over
    create_hw_axi_txn writes/reads, and the uploaded image round-trips through the
    final host packer/unpacker (no Vivado, no hardware)."""

    from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession
    from fpga.pulse_streamer.host.image import StreamerParams, CtrlWords
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

    params = StreamerParams(max_edges=16, bank_size=4)
    hw = _FakeStreamerHardware(params)
    session = VivadoAxiStreamerSession(state_dir=tmp_path, params=params, tcl_executor=hw)

    session.prepare(program)
    assert hw.load_ok, "final unpacker must accept the uploaded image"
    # banks 0,1 armed at prepare (3 points fit in 2 banks of 4).
    assert hw.bram[CtrlWords.BANK_READY] == 0b11
    session.fire()
    assert hw.fired
    assert session.wait_done(timeout=1.0) is True
    session.safe_state()

    assert hw.bram[CtrlWords.PROG_COUNT] == 4  # edges uploaded
    assert hw.bram[CtrlWords.SCAN_COUNT] == 3  # scan points uploaded


def test_vivado_axi_session_streams_unbounded_scan(tmp_path):
    """A scan with more points than the 2-bank window (N > 2*bank_size) STREAMS:
    wait_done polls CURSOR and refills each freed ping-pong bank with the next
    chunk, re-arming its BANK_READY bit, until the whole N-point sweep is played."""

    from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession
    from fpga.pulse_streamer.host.image import StreamerParams, CtrlWords, region_bases
    from Zou_lab_control.neutral_atom.devices.sequencer import RuntimeSequenceProgram

    N = 9                      # 9 points, bank_size 2 -> 5 chunks (chunks 2,3,4 streamed)
    params = StreamerParams(max_edges=16, bank_size=2)
    program = RuntimeSequenceProgram(
        sequence_id="s", sequence_name="s", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 5, 25, 40], masks=[0, 1, 2, 0], duration=4e-6, trigger_count=0,
        repeat_forever=False, loop_start_index=0, loop_end_tick=40, loop_count=1,
        slot_count=1, slot_kinds=["delay"], loop_end_slot_coeffs=[0],
        tick_slot_coeffs=[[0], [256], [0], [0]],
        scan_points=[[k] for k in range(N)], scan_coeff_frac_bits=8,
    )
    hw = _FakeStreamerHardware(params, total_points=N)
    session = VivadoAxiStreamerSession(state_dir=tmp_path, params=params, tcl_executor=hw)
    session.prepare(program)
    assert hw.bram[CtrlWords.BANK_READY] == 0b11      # banks 0,1 resident
    session.fire()
    assert session.wait_done(timeout=2.0) is True

    # chunks 2,3,4 were streamed in: each refill de-arms then re-arms its bank, so
    # exactly 2*(total_chunks-2) post-fire BANK_READY writes occurred.
    total_chunks = -(-N // params.bank_size)          # ceil
    assert total_chunks == 5
    assert len(hw.bank_ready_writes) == 2 * (total_chunks - 2)
    # the last write re-armed both banks ready.
    assert hw.bank_ready_writes[-1] == 0b11
    # the streamed chunks actually landed in their (alternating) banks.
    bases = region_bases(params)
    scan_base = bases["scan"]
    bank0_word = scan_base + 0 * params.bank_size * params.scan_words
    bank1_word = scan_base + 1 * params.bank_size * params.scan_words
    # chunk 4 (even) -> bank 0 first slot value == point 8
    assert hw.bram[bank0_word] == 8
    # chunk 3 (odd) -> bank 1 first slot value == point 6
    assert hw.bram[bank1_word] == 6


def test_vivado_axi_session_repeat_forever_treats_running_as_done(tmp_path):
    """A repeat_forever program never asserts DONE; wait_done must return once RUNNING
    is seen instead of blocking for the whole timeout."""

    from Zou_lab_control.neutral_atom.devices.axi_session import VivadoAxiStreamerSession
    from fpga.pulse_streamer.host.image import StreamerParams
    from Zou_lab_control.neutral_atom.devices.sequencer import RuntimeSequenceProgram

    program = RuntimeSequenceProgram(
        sequence_id="p", sequence_name="p", clock_hz=50e6,
        channels=[f"ch{i:02d}" for i in range(62)],
        ticks=[0, 100, 200], masks=[1, 0, 0],
        duration=4e-6, trigger_count=0,
        repeat_forever=True, loop_start_index=0, loop_end_tick=200, loop_count=1,
        slot_count=0,
    )
    params = StreamerParams(max_edges=16, bank_size=4)
    hw = _FakeStreamerHardware(params, forever=True)
    session = VivadoAxiStreamerSession(state_dir=tmp_path, params=params, tcl_executor=hw)
    session.prepare(program)
    session.fire()
    # DONE never sets, but wait_done returns True because RUNNING is observed.
    assert session.wait_done(timeout=1.0) is True


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
    assert "axi_session" in fpga_text                    # final JTAG-to-AXI backend
    assert "legacy_address_switch" not in fpga_text
    assert "na.run_sequencer_server" in fpga_text
    # final design: jtag-axi backend + the short in-repo build dir (no VIO project)
    assert "jtag-axi" in fpga_text
    assert "fpga\\build\\ps" in fpga_text
    assert "fpga\\build\\pulse_streamer" not in fpga_text
    assert "address_switch.xpr" not in fpga_text
    assert "vivado-session" not in fpga_text
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


def test_top_status_fsm_clears_running_on_safe_then_reloads():
    """Cycle-accurate model of the top's STATUS/command FSM, modeling the TWO always
    blocks (the FSM sets ldr_status_val; a SEPARATE writeback block applies it to
    ctrl_reg[C_STATUS] ONE cycle later).  That delay is what made the off->on
    "STATUS=0x00000002" bug: a CMD_SAFE that should clear RUNNING was bounced back by
    the DONE/UNDERFLOW refresh re-reading the stale ctrl_reg[C_STATUS] the next cycle,
    so the next CMD_LOAD's LOADED never stuck.  The fix gates the refresh on an
    FSM-owned ``status_running`` flag (cleared atomically by the command).  This test
    reproduces the bounce with the OLD gate and proves the NEW gate clears + reloads,
    and asserts the RTL actually uses the fixed gate.  (The top FSM has no Verilog sim,
    so this models it directly.)"""
    from pathlib import Path

    LOADED, RUNNING, DONE, UNDER = 1, 2, 4, 16
    LOAD, FIRE, RESET, SAFE = 1, 2, 4, 8

    class Fsm:
        def __init__(self, mode):
            self.mode = mode
            self.status = 0; self.command = 0; self.ldr_we = 0; self.ldr_val = 0
            self.status_running = 0; self.eng_reset = 1; self.cmd_seen = 0
            self.lstate = "IDLE"; self.ctr = 0; self.done = 0; self.under = 0

        def tick(self, cmd_write=None):
            c = dict(self.__dict__)
            n_status = c["ldr_val"] if c["ldr_we"] else c["status"]   # Block A (delayed writeback)
            n_command = cmd_write if cmd_write is not None else c["command"]
            n_we = 0; n_val = c["ldr_val"]; n_run = c["status_running"]
            n_res = c["eng_reset"]; n_seen = c["cmd_seen"]; n_lstate = c["lstate"]; n_ctr = c["ctr"]
            cmd_now = c["command"] & 0xF; edge = cmd_now & (~c["cmd_seen"]) & 0xF
            if c["lstate"] == "IDLE":
                n_seen = cmd_now
                if edge & RESET or edge & SAFE:
                    n_res = 1; n_run = 0; n_we = 1; n_val = 0
                elif edge & LOAD:
                    n_res = 1; n_run = 0; n_lstate = "LOAD"; n_ctr = 3
                elif (edge & FIRE) and (c["status"] & LOADED):
                    n_lstate = "FIRE"
            elif c["lstate"] == "LOAD":
                if c["ctr"] > 0: n_ctr = c["ctr"] - 1
                else: n_we = 1; n_val = LOADED; n_lstate = "IDLE"
            elif c["lstate"] == "FIRE":
                n_res = 0; n_run = 1; n_we = 1; n_val = RUNNING; n_seen = cmd_now; n_lstate = "IDLE"
            cond = (c["status"] & RUNNING) if self.mode == "old" else c["status_running"]
            if c["lstate"] == "IDLE" and edge == 0 and cond:
                n_we = 1
                n_val = ((0 if c["done"] else RUNNING) | (DONE if c["done"] else 0) | (UNDER if c["under"] else 0))
                if self.mode == "new" and c["done"]: n_run = 0
            self.status, self.command = n_status, n_command
            self.ldr_we, self.ldr_val = n_we, n_val
            self.status_running, self.eng_reset = n_run, n_res
            self.cmd_seen, self.lstate, self.ctr = n_seen, n_lstate, n_ctr

        def cmd(self, c):
            self.tick(0); self.tick(c)
            for _ in range(10): self.tick()

    def scenario(mode):
        f = Fsm(mode)
        f.cmd(SAFE); f.cmd(LOAD); on1_loaded = bool(f.status & LOADED)
        f.cmd(FIRE); on1_running = bool(f.status & RUNNING)
        for _ in range(10):
            f.tick()                       # repeat-forever: engine never asserts done
        f.cmd(SAFE); off_status = f.status
        f.cmd(SAFE); f.cmd(LOAD); on2_loaded = bool(f.status & LOADED)
        return on1_loaded, on1_running, off_status, on2_loaded

    # OLD gate reproduces the reported bug: STATUS stuck RUNNING -> 2nd LOAD never loads.
    old = scenario("old")
    assert old[2] == RUNNING and old[3] is False
    # NEW gate: SAFE clears STATUS, the next LOAD asserts LOADED, off->on works.
    new = scenario("new")
    assert new[0] is True and new[1] is True
    assert new[2] == 0, "CMD_SAFE must clear STATUS even from RUNNING"
    assert new[3] is True, "off->on must reload (LOADED) after a prior run"

    # The RTL must actually use the fixed (status_running) gate, not the buggy one.
    top = (Path(__file__).resolve().parents[1] / "fpga" / "pulse_streamer" / "zlc_pulse_streamer_top.v").read_text(encoding="utf-8")
    assert "status_running" in top
    assert "ctrl_reg[C_STATUS][1]) begin" not in top


def test_pulse_table_delay_is_cyclic_in_preview():
    """A channel delay in the preview (to_sequence) is a CYCLIC rotation within the
    frame (delay %% total_duration): a pulse pushed past the frame end wraps to the
    front.  This is the periodic ("inf") view, correct for ANY delay (>= total, or
    negative) -- not the old additive shift that only worked for delay < total."""
    import Zou_lab_control.neutral_atom as na

    st = na.PulseTableState(
        channels=["ch0"],
        periods=[na.PulsePeriod(1000, (1,), unit="ns"), na.PulsePeriod(1000, (0,), unit="ns")],
        time_step_ns=20,
    )  # frame = 2000 ns, ch0 ON [0,1000)

    def on_intervals(delay_ns):
        st.delays = {"ch0": delay_ns}; st.delay_units = {"ch0": "ns"}
        seq = st.to_sequence()
        return sorted((round(p.start * 1e9), round((p.start + p.duration) * 1e9)) for p in seq.effective_pulses())

    assert on_intervals(0) == [(0, 1000)]
    assert on_intervals(500) == [(500, 1500)]
    assert on_intervals(1500) == [(0, 500), (1500, 2000)]      # wraps past the frame end
    assert on_intervals(2500) == [(500, 1500)]                 # 2500 %% 2000 == 500
    assert on_intervals(-500) == [(0, 500), (1500, 2000)]      # -500 %% 2000 == 1500
    # frame total is unchanged by delay (cyclic, never extends the period)
    st.delays = {"ch0": 1500}
    assert round(st.to_sequence().duration * 1e9) == 2000


def _additive_truth(state, *, slots, time_step_ns, channels, n_ticks):
    """Independent ADDITIVE delay oracle (the trusted root for hardware delay): build
    each digital channel's UN-delayed ON intervals, shift them by a PURE additive delay
    (no modulo/wrap), re-translate the whole frame by G = max(0, -min delay) so the
    earliest event is >= 0, extend the frame to fit the latest shifted edge, and emit
    the per-tick mask by interval membership over the repeating extended frame.  This is
    correct by inspection and is structurally different from to_sequence (cyclic) and
    from reference_play (plays a given table) -- so it independently proves the compiler
    builds the right additive edge table."""

    starts = [0]
    for period in state.periods:
        starts.append(starts[-1] + period.duration_steps(slots=slots, time_step_ns=time_step_ns))
    table_end = starts[-1]
    bus_members = {c for members in state.bus_channels().values() for c in members}
    delays, intervals = {}, {}
    for ci, ch in enumerate(state.channels):
        if ch in bus_members:
            continue
        delays[ch] = state.delay_steps(ch, slots=slots, time_step_ns=time_step_ns)
        ivals, active = [], None
        for pi, period in enumerate(state.periods):
            v = int(period.states[ci])
            if v and active is None:
                active = starts[pi]
            elif not v and active is not None:
                ivals.append((active, starts[pi])); active = None
        if active is not None:
            ivals.append((active, table_end))
        intervals[ch] = ivals
    g = max(0, -min(delays.values())) if delays else 0
    eff = {ch: delays[ch] + g for ch in delays}
    T = table_end
    bits = {ch: channels.index(ch) for ch in delays if ch in channels}
    out = []
    for t in range(n_ticks):
        mask = 0
        for ch, ivals in intervals.items():
            d = eff[ch]
            if t < d or T <= 0:                 # channel hasn't started (real delay)
                continue
            phase = (t - d) % T                 # period PRESERVED at T (physical delay)
            for a, b in ivals:
                if a <= phase < b:
                    mask |= 1 << bits[ch]; break
        out.append(mask)
    return out


def test_pulse_table_repeat_forever_delay_is_additive_in_hardware():
    """HARDWARE delay is a PURE ADDITIVE shift (NOT the cyclic %total preview): the
    channel comes out `delay` later with NO wrap, the frame EXTENDS to fit it, and the
    loop repeats that extended frame.  The FIRST pulses after fire are exact (the cyclic
    view would fake a wrapped-in tail at t=0, corrupting the experiment startup).
    Proven against the independent additive oracle via the cycle-accurate engine model
    (no Verilog/hardware needed)."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.devices.sequencer import compile_pulse_table_runtime_program
    from fpga.pulse_streamer.host import engine_model as em

    st = na.PulseTableState(
        channels=["ch0"],
        periods=[na.PulsePeriod(1000, (1,), unit="ns"), na.PulsePeriod(1000, (0,), unit="ns")],
        time_step_ns=20,
    )  # frame 2000 ns = 100 ticks; ch0 ON [0,1000) = ticks [0,50)
    st.delays = {"ch0": 1500}; st.delay_units = {"ch0": "ns"}   # +75 ticks (additive: extends)

    prog = compile_pulse_table_runtime_program(st, clock_hz=50e6, repeat_forever=True)
    assert prog.masks[-1] == 0                  # final mask is safe-idle
    # period is PRESERVED at T=100: a preamble frame [0,100) plays once, then the steady
    # frame [100,200) loops -- so loop_end=200 and the engine rewinds to tick 100, NOT 0.
    assert prog.loop_end_tick == 200
    assert prog.repeat_from_index > 0 and prog.ticks[prog.repeat_from_index] == 100

    truth = _additive_truth(st, slots={}, time_step_ns=20, channels=["ch0"], n_ticks=400)
    ep = em.EngineProgram.from_program(prog)
    assert em.reference_play(ep, 400) == truth
    # the BRAM FIFO engine + its exact RTL register mirror both reproduce the additive
    # truth -- the no-Verilog-sim proof that the repeat_from_loop_start rewind is correct.
    assert em.prefetch_play(ep, 400) == truth
    assert em.rtl_mirror_play(ep, 400) == truth
    # the additive hardware is OFF at fire and turns ON only at tick 75 -- it does NOT
    # show the cyclic preview's wrapped tail at t=0; steady state then has period 100.
    assert truth[0] == 0 and truth[74] == 0 and truth[75] == 1 and truth[124] == 1 and truth[125] == 0
    assert truth[175] == 1 and truth[224] == 1 and truth[225] == 0   # repeats every 100 ticks


def test_pulse_table_negative_delay_global_retranslate_in_hardware():
    """A NEGATIVE delay re-translates the WHOLE frame (G = -min delay added to every
    channel) so the delayed channel precedes the rest and the earliest event is >= 0 --
    never a runtime negative tick.  Proven against the additive oracle."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.devices.sequencer import compile_pulse_table_runtime_program
    from fpga.pulse_streamer.host import engine_model as em

    st = na.PulseTableState(
        channels=["a", "b"],
        periods=[na.PulsePeriod(1000, (1, 1), unit="ns"), na.PulsePeriod(1000, (0, 0), unit="ns")],
        time_step_ns=20,
    )  # frame 100 ticks; a,b ON [0,50)
    st.delays = {"a": -500}; st.delay_units = {"a": "ns"}   # a -25 ticks -> G=25: a stays, b shifts +25
    prog = compile_pulse_table_runtime_program(st, clock_hz=50e6, repeat_forever=True)
    assert min(prog.ticks) >= 0                  # never a negative tick
    truth = _additive_truth(st, slots={}, time_step_ns=20, channels=list(prog.channels), n_ticks=300)
    assert em.reference_play(em.EngineProgram.from_program(prog), 300) == truth
