"""Current FINAL calibration durability and authority contracts."""

from __future__ import annotations

from dataclasses import fields
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import zlc_neutral_atom.logic_nodes.readout.calibration.analysis as calibration_analysis_module
from zlc_neutral_atom.logic_nodes.readout.calibration.analysis import (
    CalibrationAnalysisResult,
    CalibrationComputation,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    CalibrationAnalysisRequest,
    ResolvedCalibration,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.application import (
    CalibrationArtifactRequest,
    build_calibration_artifact_request,
    calibration_request_from_computation,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.repository import (
    compile_calibration_artifact_plan,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
    calibration_artifact_ref_from_tree,
    calibration_artifact_ref_to_tree,
)
from zlc_neutral_atom.logic_nodes.readout.bimodal import BimodalFit, fit_bimodal
from zlc_neutral_atom.logic_nodes.readout.calibration.sitemap import (
    SitemapCalibrationRequest,
    build_sitemap_calibration_request,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.application import (
    DetectionRequest,
    build_detection_request,
)


ROOT = Path(__file__).parents[1]


def _run_isolated(script: str, workspace: Path) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), str(workspace)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if completed.returncode:
        pytest.fail(
            "isolated calibration durability probe failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    marker = "RESULT_JSON="
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    pytest.fail(f"isolated probe returned no result marker: {completed.stdout}")


def test_current_calibration_surface_rejects_access_after_experiment_close(
    tmp_path,
) -> None:
    from Zou_lab_control.api import WorkspacePaths, connect

    experiment = connect(
        "virtual",
        workspace=WorkspacePaths.for_workspace((tmp_path / "workspace").resolve()),
    )
    calibration = experiment.nodes.calibration
    assert calibration.current_calibration_ref is None
    experiment.close()

    with pytest.raises(RuntimeError, match="closing or closed"):
        _ = calibration.current_calibration_ref
    with pytest.raises(RuntimeError, match="closing or closed"):
        calibration.current_calibration_ref = None


def test_calibration_analysis_and_resolved_values_have_no_authority_tokens():
    assert CalibrationAnalysisRequest.__module__ == (
        "zlc_neutral_atom.logic_nodes.readout.calibration.calibration"
    )
    assert CalibrationComputation.__module__ == (
        "zlc_neutral_atom.logic_nodes.readout.calibration.analysis"
    )
    assert CalibrationAnalysisResult.__module__ == CalibrationComputation.__module__
    assert BimodalFit.__module__ == (
        "zlc_neutral_atom.logic_nodes.readout.bimodal"
    )
    assert fit_bimodal.__module__ == BimodalFit.__module__
    assert not hasattr(calibration_analysis_module, "BimodalFit")
    assert not hasattr(calibration_analysis_module, "fit_bimodal")
    assert inspect.isclass(CalibrationComputation)
    assert tuple(field.name for field in fields(CalibrationAnalysisResult)) == (
        "artifact",
        "report",
        "source",
        "_source_resolution",
    )
    assert tuple(field.name for field in fields(ResolvedCalibration)) == (
        "reference",
        "artifact",
        "run_id",
    )


def test_calibration_reference_is_one_strict_relative_record_path():
    reference = CalibrationArtifactRef("run-001/calibration.json")
    tree = calibration_artifact_ref_to_tree(reference)

    assert tuple(field.name for field in fields(CalibrationArtifactRef)) == (
        "record_path",
    )
    assert reference.record_path == "run-001/calibration.json"
    assert calibration_artifact_ref_from_tree(tree) == reference

    for invalid in (
        "calibration.json",
        "run-001/other.json",
        "nested/run-001/calibration.json",
        "../run-001/calibration.json",
        "/run-001/calibration.json",
        "run-001\\calibration.json",
        "run-001//calibration.json",
    ):
        with pytest.raises((TypeError, ValueError)):
            CalibrationArtifactRef(invalid)


def test_compile_plan_requires_explicit_source_binding_and_deadline():
    signature = inspect.signature(compile_calibration_artifact_plan)
    assert tuple(signature.parameters) == (
        "source_capture_ref",
        "captures_root",
        "calibrations_root",
        "request",
        "expected_readout_binding",
        "timeout_seconds",
        "on_committed",
    )
    for name in (
        "expected_readout_binding",
        "timeout_seconds",
        "on_committed",
    ):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_public_calibration_and_detection_requests_are_deadline_free():
    assert tuple(field.name for field in fields(CalibrationArtifactRequest)) == (
        "source_capture_ref",
        "readout_binding",
        "analysis",
    )
    assert tuple(field.name for field in fields(SitemapCalibrationRequest)) == (
        "capture_request",
        "analysis",
    )
    assert tuple(field.name for field in fields(DetectionRequest)) == (
        "source_capture_ref",
        "calibration_ref",
        "readout_binding",
        "readout_event_axis_id",
        "model_kind",
    )
    for function in (
        build_calibration_artifact_request,
        calibration_request_from_computation,
        build_sitemap_calibration_request,
        build_detection_request,
    ):
        assert "timeout_seconds" not in inspect.signature(function).parameters
        assert "calibration_timeout_seconds" not in inspect.signature(
            function
        ).parameters

    from zlc_neutral_atom.logic_nodes.readout.calibration.api import CalibrationApi
    from zlc_neutral_atom.logic_nodes.readout.occupancy.api import OccupancyApi

    for function in (
        CalibrationApi.sitemap_request,
        CalibrationApi.sitemap,
        CalibrationApi.calibration_request,
        CalibrationApi.start_calibration_analysis,
        CalibrationApi.calibration_gui,
        CalibrationApi.calibration_edit_gui,
        OccupancyApi.detection_request,
    ):
        parameters = inspect.signature(function).parameters
        assert "timeout_seconds" not in parameters
        assert "calibration_timeout_seconds" not in parameters


def test_final_calibration_reopens_from_disk_with_exact_capture_authority(tmp_path):
    result = _run_isolated(
        """
        import json
        from pathlib import Path
        import sys

        import numpy as np

        from Zou_lab_control.api import WorkspacePaths, connect
        import zlc_neutral_atom.logic_nodes.readout.calibration.repository as repository
        from zlc_storage import decode

        workspace = Path(sys.argv[1])
        project = Path.cwd().resolve()
        write_order = []
        written_array_dtypes = {}
        write_npy = repository._write_npy
        write_record = repository.atomic_write_bytes

        def observe_array(path, value):
            write_order.append(("array", path))
            written_array_dtypes[path] = value.dtype.str
            write_npy(path, value)

        def observe_record(path, payload):
            write_order.append(("record", path))
            write_record(path, payload)

        repository._write_npy = observe_array
        repository.atomic_write_bytes = observe_record
        experiment = connect(
            "virtual",
            workspace=WorkspacePaths(
                project,
                project / "pulses",
                project / "tasks",
                workspace / "_output",
            ),
            seed=7,
        )
        reference = experiment.nodes.calibration.sitemap(frames=4)
        resolved = experiment.nodes.calibration.load_calibration(reference)
        computation = experiment.nodes.calibration.load_calibration_computation(reference)
        source_reference = resolved.artifact.source_binding.source_capture_ref
        live_result = {
            "reference_path": reference.record_path,
            "resolved_matches": resolved.reference == reference,
            "source_path": source_reference.record_path,
            "model_kinds": [model.kind.value for model in resolved.artifact.models],
            "report_kinds": [model.kind.value for model in computation.report.models],
            "grid_shape": list(resolved.artifact.site_map.grid_shape_yx),
            "run_id": resolved.run_id,
        }
        experiment.close()

        output = workspace / "_output"
        captures = output / "captures"
        calibrations = output / "calibrations"
        record_path = calibrations / reference.record_path
        record = decode(record_path.read_bytes())
        reopened = repository.load_calibration_artifact(
            calibrations,
            captures,
            reference,
        )
        reopened_computation = repository.load_calibration_computation(
            calibrations,
            captures,
            reference,
        )
        run_directory = record_path.parent
        array_files = tuple(sorted((run_directory / "arrays").glob("*.npy")))
        arrays_preserve_dtype = all(
            np.load(path, allow_pickle=False).dtype.str
            == written_array_dtypes[path]
            for path in array_files
        )
        live_result.update(
            {
                "reopened_matches": reopened.reference == reference,
                "reopened_source_matches": (
                    reopened.artifact.source_binding.source_capture_ref
                    == source_reference
                ),
                "reopened_grid_shape": list(
                    reopened.artifact.site_map.grid_shape_yx
                ),
                "record_run_id": record["run_id"],
                "record_fields": sorted(record),
                "record_schema": record["schema"],
                "source_ref_fields": sorted(record["source_capture_ref"]),
                "array_count": len(array_files),
                "array_dtypes_preserved": arrays_preserve_dtype,
                "record_written_last": (
                    bool(write_order)
                    and write_order[-1] == ("record", record_path)
                    and all(kind == "array" for kind, _path in write_order[:-1])
                ),
                "reopened_report_kinds": [
                    model.kind.value
                    for model in reopened_computation.report.models
                ],
            }
        )
        print("RESULT_JSON=" + json.dumps(live_result, sort_keys=True))
        """,
        tmp_path / "calibration-durability",
    )
    assert result["reference_path"].endswith("/calibration.json")
    assert result["source_path"].endswith("/capture.json")
    assert result["resolved_matches"] is True
    assert result["reopened_matches"] is True
    assert result["reopened_source_matches"] is True
    assert result["record_run_id"] == result["run_id"]
    assert result["record_fields"] == [
        "artifact",
        "report",
        "run_id",
        "schema",
        "source_capture_ref",
    ]
    assert result["record_schema"] == (
        "zlc_neutral_atom.logic_nodes.readout.calibration.record"
    )
    assert result["source_ref_fields"] == ["record_path", "schema"]
    assert result["array_count"] > 0
    assert result["array_dtypes_preserved"] is True
    assert result["record_written_last"] is True
    assert result["grid_shape"] == [5, 7]
    assert result["reopened_grid_shape"] == [5, 7]
    assert result["model_kinds"] == ["box", "psf", "uniform_psf"]
    assert result["report_kinds"] == ["box", "psf", "uniform_psf"]
    assert result["reopened_report_kinds"] == ["box", "psf", "uniform_psf"]


def test_removed_calibration_commit_wrappers_are_not_reintroduced():
    import zlc_neutral_atom.logic_nodes.readout.calibration.repository as repository_module

    for removed in (
        "CalibrationRepository",
        "CALIBRATION_MANIFEST_FORMAT",
        "CalibrationCommit",
        "CalibrationCheckpoint",
        "analyze_calibration",
    ):
        assert not hasattr(repository_module, removed)
