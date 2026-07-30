from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np

from zlc_frontend import (
    HistogramDisplayState,
    PlotPanelSession,
)
from zlc_frontend.plot_panel import PlotPanelComposeRequest
from zlc_frontend.plot_report import render_plot_report
from zlc_frontend.encoded_raster import encode_raster_buffer_png
from zlc_frontend.plot_layout import (
    PANEL_EXPORT_PIXEL_RATIO,
)
from zlc_data import (
    CellValidity,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    CoordinateFrameId,
    ValidityContract,
    ValueSchema,
)
from zlc_neutral_atom.devices.camera.contract import (
    CameraPhysicalFacts,
    ReadoutBindingKey,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    CalibrationAnalysisRequest,
    GridOrder,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.ui.workbench_jobs import (
    project_calibration_plot_report,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.projection import (
    CalibrationModelReportProjection,
    CalibrationReportProjection,
)
from zlc_neutral_atom.logic_nodes.readout.calibration import (
    projection as calibration_projection,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.result_bundle import (
    CALIBRATION_RESULT_BUNDLE_FORMAT,
    write_calibration_result_bundle,
)
from zlc_neutral_atom.logic_nodes.readout.calibration import (
    result_bundle as calibration_result_bundle,
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
from zlc_neutral_atom.capture.reference import (
    CaptureArtifactRef,
)
from zlc_pulse import PulseTarget


def _render_current_report(view):
    return render_plot_report(project_calibration_plot_report(view))


def test_calibration_result_bundle_is_discoverable_non_authoritative_export(
    tmp_path,
) -> None:
    camera_frame = CoordinateFrameId("calibration-result-test-camera-frame")
    frame_schema = ValueSchema(
        (
            AxisSpec(
                AxisId("calibration-result-test-y"),
                "camera y",
                SPATIAL_Y,
                4,
                tuple(range(4)),
                "pixel",
                camera_frame,
            ),
            AxisSpec(
                AxisId("calibration-result-test-x"),
                "camera x",
                SPATIAL_X,
                4,
                tuple(range(4)),
                "pixel",
                camera_frame,
            ),
        ),
        ValidityContract.components(
            AxisId("calibration-result-test-y"),
            AxisId("calibration-result-test-x"),
        ),
        np.dtype("<u2"),
        "count",
    )
    site_axis = AxisSpec(
        AxisId("calibration-result-test-site"),
        "readout site",
        SITE,
        1,
        ("site-0",),
    )
    model = CalibrationModelReportProjection(
        label="box",
        is_default=True,
        signals=np.asarray([[1.0], [8.0], [2.0], [9.0]]),
        signal_validity=np.asarray([[True], [True], [False], [True]]),
        quick_thresholds=np.asarray([5.0]),
        formal_thresholds=np.asarray([5.5]),
        runtime_thresholds=np.asarray([5.5]),
        runtime_threshold_sources=("formal",),
        feature_validity=np.asarray([True]),
        runtime_usable=np.asarray([True]),
        bright_above=np.asarray([True]),
        model_fidelity=np.asarray([0.98]),
        heldout_fidelity=np.asarray([1.0]),
        dark_fidelity=np.asarray([1.0]),
        bright_fidelity=np.asarray([1.0]),
        dark_mean=np.asarray([1.5]),
        dark_sigma=np.asarray([0.5]),
        bright_mean=np.asarray([8.5]),
        bright_sigma=np.asarray([0.5]),
        n_test=np.asarray([2], dtype=np.int64),
        n_train_dark=np.asarray([1], dtype=np.int64),
        n_train_bright=np.asarray([1], dtype=np.int64),
        runtime_model_fidelity_mean=0.98,
        aggregate_fidelity=1.0,
        global_threshold=5.0,
        global_bright_above=True,
        global_fidelity=0.95,
        ablation_drop_worst_k=np.asarray([0], dtype=np.int64),
        ablation_excluded_sites=np.asarray([[False]], dtype=np.bool_),
        ablation_fidelity=np.asarray([1.0]),
        ablation_errors=np.asarray([0], dtype=np.int64),
        ablation_n_valid=np.asarray([2], dtype=np.int64),
    )
    calibration_ref = CalibrationArtifactRef("run-calibration/calibration.json")
    capture_ref = CaptureArtifactRef("run-capture/capture.json")
    view = CalibrationReportProjection(
        frame_schema=frame_schema,
        site_axis=site_axis,
        coordinate_frame=camera_frame,
        reference_average=np.arange(16, dtype=np.float64).reshape(4, 4),
        reference_average_validity=np.ones((4, 4), dtype=np.bool_),
        actual_centers_xy=np.asarray([[1.5, 2.0]]),
        expected_centers_xy=np.asarray([[1.4, 2.1]]),
        site_validity=np.asarray([True]),
        default_boxes_xywh=np.asarray([[1.0, 1.0, 2.0, 2.0]]),
        grid_shape_yx=(1, 1),
        site_grid_positions_yx=((0, 0),),
        site_labels=("site-0",),
        occupied_labels=np.asarray([[False], [True], [False], [True]]),
        dark_labels=np.asarray([[True], [False], [True], [False]]),
        label_validity=np.asarray([[True], [False], [True], [True]]),
        models=(model,),
        psf_kernels=None,
        psf_mode=None,
        psf_fit_ok=None,
        psf_sigma_xy=None,
        calibration_ref=calibration_ref,
        source_capture_identity=capture_ref.target_ref,
        binding="camera",
        camera_identity="virtual-qcmos",
        roi_shape_yx=(4, 4),
        exposure_seconds=0.005,
        group_count=4,
        software_lineage=(("numpy", np.__version__),),
    )
    destination = tmp_path / "report"

    write_calibration_result_bundle(
        destination,
        view,
        calibration_ref,
        capture_ref,
        render_report=_render_current_report,
    )

    # Calibration supplies only the typed SiteMap page.  The report writer
    # must persist exactly the raster produced by the frontend report contract;
    # the real TaskConsole FINAL-card route is covered by the GUI product-flow
    # test rather than reconstructed here.
    document = project_calibration_plot_report(view)
    overview = next(page for page in document.pages if page.key == "overview")
    per_site = next(page for page in document.pages if page.key == "hist-box")
    assert all(not page.key.startswith("pooled-") for page in document.pages)
    histogram = per_site.source.snapshot.block
    assert histogram.schema.physical_shape == (4, 1, 1)
    assert histogram.schema.cell_schema.is_scalar
    np.testing.assert_array_equal(
        histogram.values,
        np.asarray([1.0, 8.0, 2.0, 9.0]).reshape(4, 1, 1),
    )
    assert isinstance(histogram.validity, CellValidity)
    np.testing.assert_array_equal(
        histogram.validity.mask,
        np.asarray([[True], [True], [False], [True]]),
    )
    assert isinstance(per_site.display, HistogramDisplayState)
    assert per_site.display.thresholds == (5.5,)
    assert per_site.contract.figure.value_label == "count"
    assert overview.contract.size_name == "2x2"
    report_runtime_contract = replace(
        overview.contract,
        pixel_ratio=PANEL_EXPORT_PIXEL_RATIO,
    )
    session = PlotPanelSession(report_runtime_contract)
    try:
        direct = session.compose(
                PlotPanelComposeRequest(
                    overview.source,
                    overview.display,
                )
        )
        assert direct.frame is not None
        assert len(direct.frame.panels) == 1
        stamp = direct.frame.panels[0].coherence_stamp
        assert stamp is not None
        assert overview.source.site_map is not None
        assert stamp.inputs == tuple(
            sorted(
                (
                    overview.source.site_map.background_input,
                    overview.source.site_map.site_state_input,
                ),
                key=lambda value: value.dataset_id.value,
            )
        )
        direct_png = encode_raster_buffer_png(direct.frame.panels[0].raster)
    finally:
        session.close()
    assert (destination / "overview.png").read_bytes() == direct_png

    assert {path.name for path in destination.iterdir()} == {
        "README.txt",
        "summary.json",
        "diagnostics.npz",
        "sites.csv",
        "overview.png",
        "fidelity.png",
        "hist-box.png",
    }
    summary = json.loads((destination / "summary.json").read_text("utf-8"))
    assert summary["schema"] == CALIBRATION_RESULT_BUNDLE_FORMAT
    assert summary["authority"]["calibration_ref"]["record_path"] == (
        "run-calibration/calibration.json"
    )
    assert summary["authority"]["source_capture_ref"]["record_path"] == (
        "run-capture/capture.json"
    )
    assert set(summary["authority"]) == {
        "calibration_ref",
        "source_capture_ref",
        "record",
        "rule",
    }
    assert summary["authority"]["record"] == "../calibration.json"
    assert all(
        set(metadata) == {"description", "size_bytes"}
        for metadata in summary["files"].values()
    )
    assert summary["site_map"]["centers_xy"] == [[1.5, 2.0]]
    model_summary = summary["models"][0]
    assert model_summary["runtime_thresholds"] == [5.5]
    assert model_summary["heldout_fidelity"] == [1.0]
    assert model_summary["dark_fidelity"] == [1.0]
    assert model_summary["bright_fidelity"] == [1.0]
    assert model_summary["dark_mean"] == [1.5]
    assert model_summary["dark_sigma"] == [0.5]
    assert model_summary["bright_mean"] == [8.5]
    assert model_summary["bright_sigma"] == [0.5]
    assert model_summary["n_test"] == [2]
    assert model_summary["n_train_dark"] == [1]
    assert model_summary["n_train_bright"] == [1]
    assert model_summary["ablation"] == [
        {
            "drop_worst_k": 0,
            "excluded_sites": [False],
            "fidelity": 1.0,
            "errors": 0,
            "n_valid": 2,
        }
    ]
    assert "only machine authority" in summary["authority"]["rule"]
    with np.load(destination / "diagnostics.npz", allow_pickle=False) as arrays:
        assert arrays["reference_average"].shape == (4, 4)
        assert arrays["model_0_signals"].shape == (4, 1)
        assert arrays["model_0_dark_fidelity"].tolist() == [1.0]
        assert arrays["model_0_bright_fidelity"].tolist() == [1.0]
        assert arrays["model_0_n_train_dark"].tolist() == [1]
        assert arrays["model_0_n_train_bright"].tolist() == [1]
        assert arrays["model_0_ablation_excluded_sites"].tolist() == [[False]]
        assert arrays["model_0_ablation_n_valid"].tolist() == [2]
    with (destination / "sites.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["model_0_dark_fidelity"] == "1.0"
    assert rows[0]["model_0_bright_fidelity"] == "1.0"
    assert rows[0]["model_0_n_test"] == "2"
    assert rows[0]["model_0_n_train_dark"] == "1"
    assert rows[0]["model_0_n_train_bright"] == "1"


def test_post_final_export_writes_report_without_a_second_pointer(
    tmp_path,
    monkeypatch,
) -> None:
    import zlc_neutral_atom.capture.artifact as capture_artifact
    import zlc_neutral_atom.logic_nodes.readout.calibration.repository as repository

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
        calibration_projection,
        "project_calibration_report",
        lambda loaded, reference: (loaded, reference),
    )

    def write_report(destination, *_args, **_kwargs) -> None:
        root = Path(destination)
        root.mkdir(parents=False, exist_ok=False)
        (root / "marker.txt").write_text("report", encoding="utf-8")

    monkeypatch.setattr(
        calibration_result_bundle,
        "write_calibration_result_bundle",
        write_report,
    )

    write_calibration_post_final_exports(
        source,
        calibration,
        captures_root=captures_root,
        calibrations_root=calibrations_root,
        save_frames=False,
        expected_camera_role="camera",
        render_report=lambda _view: None,
    )

    run_root = calibrations_root / "calibration-run"
    assert (run_root / "report" / "marker.txt").is_file()
    assert not (run_root / "source_frames.npy").exists()
    assert not (run_root / "source_frame_validity.npy").exists()
    assert not (run_root / "calibration_ref.json").exists()


def test_post_final_save_frames_preserves_source_dtype(tmp_path, monkeypatch) -> None:
    import zlc_neutral_atom.capture.artifact as capture_artifact
    import zlc_neutral_atom.logic_nodes.readout.calibration.repository as repository

    source = CaptureArtifactRef("source-run/capture.json")
    calibration = CalibrationArtifactRef("calibration-run/calibration.json")
    captures_root = (tmp_path / "captures").resolve()
    calibrations_root = (tmp_path / "calibrations").resolve()
    run_root = calibrations_root / "calibration-run"
    run_root.mkdir(parents=True)
    values = np.arange(24, dtype=np.uint16).reshape(2, 3, 2, 2)
    validity = np.asarray([[True, True, False], [True, False, True]])
    snapshot = SimpleNamespace(
        block=SimpleNamespace(
            values=values,
            validity=SimpleNamespace(mask=validity),
        )
    )
    capture = SimpleNamespace(
        camera_provenance=SimpleNamespace(
            binding=SimpleNamespace(value="camera")
        ),
        materialize_snapshot=lambda: snapshot,
    )
    computation = SimpleNamespace(
        artifact=SimpleNamespace(
            source_binding=SimpleNamespace(source_capture_ref=source)
        )
    )
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
        calibration_projection,
        "project_calibration_report",
        lambda loaded, reference: (loaded, reference),
    )
    monkeypatch.setattr(
        calibration_result_bundle,
        "write_calibration_result_bundle",
        lambda destination, *_args, **_kwargs: Path(destination).mkdir(),
    )

    write_calibration_post_final_exports(
        source,
        calibration,
        captures_root=captures_root,
        calibrations_root=calibrations_root,
        save_frames=True,
        expected_camera_role="camera",
        render_report=lambda _view: None,
    )

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
        calibration_projection,
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
