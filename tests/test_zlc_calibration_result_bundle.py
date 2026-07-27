from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np

from zlc_frontend import (
    PlotPanelComposeRequest,
    PlotPanelSession,
    render_plot_report,
)
from zlc_frontend.encoded_raster import encode_raster_buffer_png
from zlc_frontend.plot_layout import (
    PANEL_EXPORT_PIXEL_RATIO,
)
from zlc_data import (
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
from zlc_neutral_atom.logic_nodes.readout.calibration.ui.report_projection import (
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
from zlc_neutral_atom.logic_nodes.readout.calibration.repository import (
    CalibrationRepository,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.sitemap import (
    ReadoutGridGeometry,
    SitemapAcquisitionProfile,
    load_sitemap_pulse,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.task import (
    CalibrationTaskIntent,
    prepare_calibration_task,
    write_calibration_task_outputs,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.task_output import (
    read_calibration_task_output,
)
from zlc_neutral_atom.capture.application import (
    CAPTURE_READOUT_EVENT_AXIS_ID,
)
from zlc_neutral_atom.capture.artifact import CaptureRepository
from zlc_neutral_atom.capture.reference import (
    CaptureArtifactRef,
)
from zlc_neutral_atom.logic_nodes.readout.contracts import (
    CalibrationCaptureLayout,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.commit import (
    CommitTarget,
    PersistentCommitJournal,
    PublishedManifest,
    RepositoryCommitCoordinator,
)
from zlc_neutral_atom.runtime.resources import ResourceArbiter
from zlc_neutral_atom.runtime.run import CancelOutcome, RunController, RunPlan
from zlc_pulse import PulseTarget
from zlc_storage import RepositoryRootLease
from zlc_storage.paths import PROJECT_ROOT


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
        threshold_centered_signals=np.asarray(
            [[-4.5], [2.5], [-3.5], [3.5]]
        ),
        signal_validity=np.ones((4, 1), dtype=np.bool_),
        bin_edges=np.linspace(0.0, 10.0, 9),
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
        label_validity=np.ones((4, 1), dtype=np.bool_),
        models=(model,),
        psf_kernels=None,
        psf_mode=None,
        psf_fit_ok=None,
        psf_sigma_xy=None,
        calibration_identity="calibration/" + "a" * 64,
        source_capture_identity="capture/" + "b" * 64,
        binding="camera",
        camera_identity="virtual-qcmos",
        roi_shape_yx=(4, 4),
        exposure_seconds=0.005,
        group_count=4,
        software_lineage=(("numpy", np.__version__),),
    )
    calibration_ref = CalibrationArtifactRef("calibration-repository", "a" * 64)
    capture_ref = CaptureArtifactRef("capture-repository", "b" * 64)
    destination = tmp_path / "report"

    write_calibration_result_bundle(
        destination,
        view,
        calibration_ref,
        capture_ref,
        calibration_repository_root=tmp_path / "workspace" / "calibrations",
        capture_repository_root=tmp_path / "workspace" / "captures",
        render_report=_render_current_report,
    )

    # Calibration supplies only the typed SiteMap page.  The report writer
    # must persist exactly the raster produced by the frontend report contract;
    # the real TaskConsole FINAL-card route is covered by the GUI product-flow
    # test rather than reconstructed here.
    document = project_calibration_plot_report(view)
    overview = next(page for page in document.pages if page.key == "overview")
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
                overview.provenance,
            )
        )
        assert direct.frame is not None
        assert len(direct.frame.panels) == 1
        presentation = direct.frame.panels[0].coherence_stamp.presentations
        assert len(presentation) == 1
        assert presentation[0].panel_id == report_runtime_contract.panel_id
        assert presentation[0].panel_revision == overview.display.revision
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
        "pooled-box.png",
    }
    summary = json.loads((destination / "summary.json").read_text("utf-8"))
    assert summary["schema"] == CALIBRATION_RESULT_BUNDLE_FORMAT
    assert summary["authority"]["calibration_ref"]["manifest_digest"] == "a" * 64
    assert summary["authority"]["source_capture_ref"]["manifest_digest"] == "b" * 64
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


def _analysis_request() -> CalibrationAnalysisRequest:
    return CalibrationAnalysisRequest(
        CalibrationCaptureLayout(
            CAPTURE_READOUT_EVENT_AXIS_ID,
            (0, 2),
            1,
        ),
        (1, 1),
        box_radius=1,
        expected_centers_xy=np.asarray([[4.0, 4.0]]),
        maximum_site_residual_px=1.0,
    )


class _CommittedCalibrationRun:
    def __init__(
        self,
        root: Path,
        result: CalibrationArtifactRef,
    ) -> None:
        self.result = result
        self.root_lease = RepositoryRootLease(root)
        self.journal = PersistentCommitJournal(
            root / "calibration-commit.journal",
            result.repository_id,
        )
        self.coordinator = RepositoryCommitCoordinator(
            self.journal,
            lambda _intent: None,
            root_lease=self.root_lease,
        )
        self.resources = ResourceArbiter()
        self.controller = RunController(self.resources)

    def start(self):
        result = self.result
        target = CommitTarget(
            result.repository_id,
            "calibration",
            "zlc.test.calibration",
            result.target_ref,
            result.manifest_digest,
        )

        def finalize(context, _executed):
            run_id = context.authorize_commit_preparation()
            operation = self.coordinator.prepare(
                f"calibration-final-{run_id}-{result.manifest_digest}",
                run_id,
                target,
                lambda: PublishedManifest(
                    target.target_ref,
                    result.manifest_digest,
                    result,
                ),
            )
            return context.commit_final(operation)

        return self.controller.start(
            RunPlan(
                name="focused calibration commit",
                resource_claims=(),
                bound_devices=(),
                preflight=lambda _context: None,
                execute=lambda _context, prepared: prepared,
                cleanup=lambda _context, _prepared, _primary: CleanupReport.complete(),
                finalize=finalize,
                requires_final_commit=True,
            )
        )

    def close(self) -> None:
        assert self.controller.shutdown(2.0)
        self.resources.shutdown()
        self.coordinator.close()
        self.root_lease.close()


class _SavedCalibrationDependencies:
    def __init__(
        self,
        source: CaptureArtifactRef,
        child: _CommittedCalibrationRun,
        writing: threading.Event,
        release: threading.Event,
    ) -> None:
        self.source = source
        self.child = child
        self.writing = writing
        self.release = release
        self.admitted_path: Path | None = None
        self.written: tuple[str, str] | None = None

    def admit_saved_calibration_capture(
        self,
        source_path: Path,
        *,
        expected_camera_role: str,
    ) -> CaptureArtifactRef:
        self.admitted_path = source_path
        assert expected_camera_role == "camera"
        return self.source

    def sitemap_analysis_request(self, **_kwargs) -> CalibrationAnalysisRequest:
        return _analysis_request()

    def start_calibration_analysis(self, source, analysis):
        assert source == self.source
        assert analysis == _analysis_request()
        return self.child.start()

    def write_calibration_task_outputs(
        self,
        source,
        calibration,
        *,
        folder: str,
        frame_export_policy: str,
        expected_camera_role: str,
    ) -> None:
        assert source == self.source
        assert calibration == self.child.result
        assert expected_camera_role == "camera"
        self.written = (folder, frame_export_policy)
        self.writing.set()
        if not self.release.wait(2.0):
            raise TimeoutError("focused output writer was not released")


def test_saved_task_uses_project_path_preserves_frames_and_rejects_late_cancel(
    tmp_path,
    monkeypatch,
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    relative_folder = f"_output/focused-calibration-{tmp_path.name}"
    source = CaptureArtifactRef("capture-repository", "b" * 64)
    calibration = CalibrationArtifactRef("calibration-repository", "a" * 64)
    child = _CommittedCalibrationRun(tmp_path / "commit-repository", calibration)
    writing = threading.Event()
    release = threading.Event()
    dependencies = _SavedCalibrationDependencies(source, child, writing, release)
    intent = CalibrationTaskIntent(
        source_mode="saved frames",
        folder=relative_folder,
        save_frames=False,
        pulse="pulses/imaging_template.json",
        threshold_method="otsu",
        reference_exposure_s=0.020,
        readout_exposure_s=0.005,
        threshold_frames=2,
        roi_radius=1,
        camera_role="camera",
    )
    expected_root = (PROJECT_ROOT / relative_folder).resolve()

    try:
        assert Path(intent.folder) == expected_root
        prepared = prepare_calibration_task(intent, dependencies)
        assert dependencies.admitted_path == expected_root / "frames"
        handle = prepared.start()
        assert writing.wait(2.0)
        assert handle.snapshot().final_committed
        assert (
            handle.cancel("after calibration commit")
            is CancelOutcome.TOO_LATE_ALREADY_COMMITTED
        )
        release.set()
        assert handle.result(2.0) == calibration
        assert dependencies.written == (str(expected_root), "preserve")
        assert prepared.completion_summary(calibration) == (
            f"done; results: {expected_root}; report: {expected_root / 'report'}"
        )
    finally:
        release.set()
        child.close()


def test_preserve_policy_keeps_admitted_saved_frame_export(
    tmp_path,
    monkeypatch,
) -> None:
    source = CaptureArtifactRef("capture-repository", "b" * 64)
    calibration = CalibrationArtifactRef("calibration-repository", "a" * 64)
    capture_repository = CaptureRepository(tmp_path / "captures")
    calibration_repository = CalibrationRepository(tmp_path / "calibrations")
    task_root = tmp_path / "task-output"
    frames = task_root / "frames"
    frames.mkdir(parents=True)
    sentinel = frames / "saved-frame-owner.txt"
    sentinel.write_text("keep", encoding="utf-8")
    admitted = SimpleNamespace(
        artifact=SimpleNamespace(
            camera_provenance=SimpleNamespace(
                binding=SimpleNamespace(value="camera")
            )
        )
    )
    computation = SimpleNamespace(
        artifact=SimpleNamespace(
            source_binding=SimpleNamespace(source_capture_ref=source)
        )
    )

    monkeypatch.setattr(
        CaptureRepository,
        "admit",
        lambda _repository, reference: admitted if reference == source else None,
    )
    monkeypatch.setattr(
        CalibrationRepository,
        "load_computation",
        lambda _repository, reference: (
            computation if reference == calibration else None
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

    try:
        write_calibration_task_outputs(
            source,
            calibration,
            folder=task_root,
            frame_export_policy="preserve",
            capture_repository=capture_repository,
            calibration_repository=calibration_repository,
            expected_camera_role="camera",
            render_report=lambda _view: None,
        )
    finally:
        calibration_repository.close()
        capture_repository.close()

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert (task_root / "report" / "marker.txt").is_file()
    pointer = read_calibration_task_output(task_root / "calibration_ref.json")
    assert pointer.calibration_ref == calibration
    assert pointer.source_capture_ref == source
    assert not tuple(task_root.glob(".*.tmp"))
    assert not tuple(task_root.glob(".*.old"))


def test_custom_sitemap_pulse_is_rebound_to_the_live_profile_target() -> None:
    base = load_sitemap_pulse()
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
        pulse_document=base,
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
