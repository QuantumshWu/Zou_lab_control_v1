from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np

from zlc_data import AxisId, CoordinateFrameId
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.devices.camera.contract import (
    CameraPhysicalFacts,
    ReadoutBindingKey,
)
from zlc_neutral_atom.logic_nodes.readout.calibration import outputs as output_owner
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import GridOrder
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.sitemap import (
    ReadoutGridGeometry,
    SitemapAcquisitionProfile,
    load_sitemap_pulse,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.task import (
    PreparedCalibrationTask,
    write_calibration_post_final_exports,
)
from zlc_pulse import PulseTarget


def _post_final_facts(tmp_path):
    source = CaptureArtifactRef("source-run/capture.json")
    calibration = CalibrationArtifactRef("calibration-run/calibration.json")
    captures_root = (tmp_path / "captures").resolve()
    calibrations_root = (tmp_path / "calibrations").resolve()
    (calibrations_root / "calibration-run").mkdir(parents=True)
    capture = SimpleNamespace(
        camera_provenance=SimpleNamespace(
            binding=SimpleNamespace(value="camera")
        )
    )
    computation = SimpleNamespace(
        artifact=SimpleNamespace(
            source_binding=SimpleNamespace(source_capture_ref=source)
        )
    )
    return (
        source,
        calibration,
        captures_root,
        calibrations_root,
        capture,
        computation,
    )


def test_post_final_export_uses_declared_outputs_without_second_projection(
    tmp_path,
    monkeypatch,
) -> None:
    import zlc_neutral_atom.capture.artifact as capture_artifact
    import zlc_neutral_atom.logic_nodes.readout.calibration.repository as repository

    (
        source,
        calibration,
        captures_root,
        calibrations_root,
        capture,
        computation,
    ) = _post_final_facts(tmp_path)
    expected_outputs = {"site_map": object()}
    observed = []
    monkeypatch.setattr(
        capture_artifact,
        "load_capture_artifact",
        lambda root, reference, *, materialize: (
            capture
            if root == captures_root and reference == source and not materialize
            else None
        ),
        raising=False,
    )
    monkeypatch.setattr(
        repository,
        "load_calibration_computation",
        lambda calibration_root, capture_root, reference: (
            computation
            if (
                calibration_root == calibrations_root
                and capture_root == captures_root
                and reference == calibration
            )
            else None
        ),
    )
    monkeypatch.setattr(
        output_owner,
        "calibration_final_outputs",
        lambda loaded, reference: (
            expected_outputs
            if loaded is computation and reference == calibration
            else None
        ),
    )

    def export(destination, outputs):
        root = Path(destination)
        root.mkdir(parents=False, exist_ok=False)
        (root / "marker.txt").write_text("report", encoding="utf-8")
        observed.append(outputs)

    write_calibration_post_final_exports(
        source,
        calibration,
        captures_root=captures_root,
        calibrations_root=calibrations_root,
        export_plots=export,
        save_frames=False,
        expected_camera_role="camera",
    )

    run_root = calibrations_root / "calibration-run"
    assert (run_root / "report" / "marker.txt").is_file()
    assert observed == [expected_outputs]
    assert not (run_root / "source_frames.npy").exists()
    assert not (run_root / "source_frame_validity.npy").exists()
    assert not (run_root / "calibration_ref.json").exists()


def test_post_final_save_frames_preserves_source_dtype(tmp_path, monkeypatch) -> None:
    import zlc_neutral_atom.capture.artifact as capture_artifact
    import zlc_neutral_atom.logic_nodes.readout.calibration.repository as repository

    (
        source,
        calibration,
        captures_root,
        calibrations_root,
        capture,
        computation,
    ) = _post_final_facts(tmp_path)
    values = np.arange(24, dtype=np.uint16).reshape(2, 3, 2, 2)
    validity = np.asarray([[True, True, False], [True, False, True]])
    snapshot = SimpleNamespace(
        block=SimpleNamespace(
            values=values,
            validity=SimpleNamespace(mask=validity),
        )
    )
    capture.materialize_snapshot = lambda: snapshot
    observed_materialize = []

    def load_capture(root, reference, *, materialize):
        assert root == captures_root
        assert reference == source
        observed_materialize.append(materialize)
        return capture

    monkeypatch.setattr(
        capture_artifact,
        "load_capture_artifact",
        load_capture,
        raising=False,
    )
    monkeypatch.setattr(
        repository,
        "load_calibration_computation",
        lambda *_args: computation,
    )
    monkeypatch.setattr(
        output_owner,
        "calibration_final_outputs",
        lambda *_args: {"site_map": object()},
    )

    write_calibration_post_final_exports(
        source,
        calibration,
        captures_root=captures_root,
        calibrations_root=calibrations_root,
        export_plots=lambda destination, *_args: Path(destination).mkdir(),
        save_frames=True,
        expected_camera_role="camera",
    )

    run_root = calibrations_root / "calibration-run"
    assert observed_materialize == [True]
    saved = np.load(run_root / "source_frames.npy", allow_pickle=False)
    saved_validity = np.load(
        run_root / "source_frame_validity.npy",
        allow_pickle=False,
    )
    assert saved.dtype == values.dtype
    np.testing.assert_array_equal(saved, values)
    np.testing.assert_array_equal(saved_validity, validity)


def test_post_final_export_failure_is_warning_and_preserves_final_outputs(
    monkeypatch,
) -> None:
    source = CaptureArtifactRef("source-run/capture.json")
    calibration = CalibrationArtifactRef("calibration-run/calibration.json")
    computation = object()
    expected_outputs = {"site_map": object()}
    observed = []

    def fail_export(
        source_ref,
        calibration_ref,
        *,
        save_frames,
        expected_camera_role,
    ) -> None:
        observed.append(
            (
                source_ref,
                calibration_ref,
                save_frames,
                expected_camera_role,
            )
        )
        raise RuntimeError("deterministic post-FINAL export failure")

    monkeypatch.setattr(
        output_owner,
        "calibration_final_outputs",
        lambda loaded, reference: (
            expected_outputs
            if loaded is computation and reference == calibration
            else None
        ),
    )
    command = object.__new__(PreparedCalibrationTask)
    command._lock = threading.Lock()
    command._analysis_handle = SimpleNamespace(
        result=lambda timeout: calibration,
    )
    command._dependencies = SimpleNamespace(
        load_calibration_computation=lambda reference: (
            computation if reference == calibration else None
        ),
        write_calibration_post_final_exports=fail_export,
    )
    command._plan = SimpleNamespace(
        intent=SimpleNamespace(save_frames=True, camera_role="camera")
    )
    command._source_capture_ref = source
    command._post_final_exports_attempted = False
    command._post_final_warning = None

    outputs = command.final_dataset_outputs(calibration)

    assert outputs is expected_outputs
    assert observed == [(source, calibration, True, "camera")]
    assert command.post_final_warning() == (
        "RuntimeError: deterministic post-FINAL export failure"
    )


def test_custom_sitemap_pulse_is_rebound_to_the_live_profile_target() -> None:
    base = load_sitemap_pulse(Path(__file__).resolve().parents[1] / "pulses")
    y_axis = AxisId("camera-y")
    x_axis = AxisId("camera-x")
    frame = CoordinateFrameId("camera-output")
    camera_facts = CameraPhysicalFacts(
        camera_identity="camera-id",
        sensor_identity="sensor-id",
        optical_path="imaging",
        capture_trigger_channels=("ch11",),
        sensor_shape_yx=(8, 8),
        roi_origin_yx=(0, 0),
        roi_shape_yx=(8, 8),
        binning_yx=(1, 1),
        spatial_y_axis_id=y_axis,
        spatial_x_axis_id=x_axis,
        coordinate_frame=frame,
        dtype=np.dtype("<u2"),
        count_unit="count",
        exposure_seconds=0.005,
        required_external_trigger_interval_seconds=0.0,
        external_trigger_integration_start_offset_seconds=0.0,
        gain=1.0,
        readout_mode="virtual",
        opaque_frame_settings_fingerprint="f" * 64,
    )
    geometry = ReadoutGridGeometry(
        frame_shape_yx=(8, 8),
        spatial_y_axis_id=y_axis,
        spatial_x_axis_id=x_axis,
        coordinate_frame=frame,
        grid_shape_yx=(1, 1),
        ordering=GridOrder.ROW_MAJOR,
        expected_centers_xy=np.asarray([[4.0, 4.0]]),
    )
    profile = SitemapAcquisitionProfile(
        readout_binding=ReadoutBindingKey("camera"),
        sequencer_role="sequencer",
        camera_facts=camera_facts,
        geometry=geometry,
        maximum_site_residual_px=1.0,
        pulse_target=base.target,
        trigger_channel="ch11",
    )
    authored_target = PulseTarget(
        base.target.raw_lanes,
        tuple(
            replace(port, key="camera_trigger") if port.key == "ch11" else port
            for port in base.target.ports
        ),
    )
    authored = replace(
        base,
        target=authored_target,
        visible_ports=tuple(
            "camera_trigger" if key == "ch11" else key
            for key in base.visible_ports
        ),
    )

    configured = profile.configured_document_for_repeats(
        2,
        reference_exposure_s=0.020,
        readout_exposure_s=0.005,
        pulse_document=authored,
    )

    assert configured.target is base.target
    assert "ch11" in configured.target.by_key
    assert configured.visible_ports == base.visible_ports
