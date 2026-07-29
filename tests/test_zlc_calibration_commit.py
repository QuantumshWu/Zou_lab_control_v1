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


def test_calibration_analysis_owner_is_split_from_commit_authority():
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
    with pytest.raises(TypeError, match="returned by a committed calibration Run"):
        CalibrationAnalysisResult()
    with pytest.raises(TypeError, match="returned by CalibrationRepository.admit"):
        ResolvedCalibration()


def test_compile_plan_requires_explicit_source_binding_and_deadline():
    signature = inspect.signature(compile_calibration_artifact_plan)
    assert tuple(signature.parameters) == (
        "source_capture_ref",
        "capture_repository",
        "calibration_repository",
        "request",
        "expected_readout_binding",
        "timeout_seconds",
    )
    for name in (
        "expected_readout_binding",
        "timeout_seconds",
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

        from Zou_lab_control.api import WorkspacePaths, connect
        from zlc_neutral_atom.capture.artifact import CaptureRepository
        from zlc_neutral_atom.logic_nodes.readout.calibration.repository import (
            CalibrationRepository,
        )
        from zlc_storage import decode

        workspace = Path(sys.argv[1])
        experiment = connect(
            "virtual",
            workspace=WorkspacePaths.for_workspace(
                Path.cwd(),
                repository_root=workspace,
            ),
            seed=7,
        )
        reference = experiment.nodes.calibration.sitemap(frames=4)
        resolved = experiment.nodes.calibration.load_calibration(reference)
        computation = experiment.nodes.calibration.load_calibration_computation(reference)
        source_reference = resolved.artifact.source_binding.source_capture_ref
        live_result = {
            "reference_repository": reference.repository_id,
            "reference_digest": reference.manifest_digest,
            "resolved_matches": resolved.reference == reference,
            "source_repository": source_reference.repository_id,
            "model_kinds": [model.kind.value for model in resolved.artifact.models],
            "report_kinds": [model.kind.value for model in computation.report.models],
            "grid_shape": list(resolved.artifact.site_map.grid_shape_yx),
        }
        experiment.close()

        captures = CaptureRepository(workspace / "captures")
        calibrations = CalibrationRepository(workspace / "calibrations")
        try:
            manifest = decode(
                calibrations._store_authority.read_manifest(
                    "calibration",
                    reference.manifest_digest,
                )
            )
            reopened = calibrations.admit(
                reference,
                captures,
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
                    "manifest_run_id": manifest["run_id"],
                }
            )
        finally:
            calibrations.close()
            captures.close()
        print("RESULT_JSON=" + json.dumps(live_result, sort_keys=True))
        """,
        tmp_path / "calibration-durability",
    )
    assert result["reference_repository"] == "zlc-neutral-calibration"
    assert len(result["reference_digest"]) == 64
    assert result["source_repository"] == "zlc-neutral-capture"
    assert result["resolved_matches"] is True
    assert result["reopened_matches"] is True
    assert result["reopened_source_matches"] is True
    assert isinstance(result["manifest_run_id"], str)
    assert result["manifest_run_id"]
    assert result["grid_shape"] == [5, 7]
    assert result["reopened_grid_shape"] == [5, 7]
    assert result["model_kinds"] == ["box", "psf", "uniform_psf"]
    assert result["report_kinds"] == ["box", "psf", "uniform_psf"]


def test_removed_calibration_commit_wrappers_are_not_reintroduced():
    import zlc_neutral_atom.logic_nodes.readout.calibration.repository as repository_module

    for removed in (
        "CalibrationCommit",
        "CalibrationCheckpoint",
        "analyze_calibration",
    ):
        assert not hasattr(repository_module, removed)
