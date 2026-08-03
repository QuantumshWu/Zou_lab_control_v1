"""Current FINAL calibration durability and authority contracts."""

from __future__ import annotations

from dataclasses import fields
import inspect
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
from zlc_neutral_atom.logic_nodes.readout.calibration.artifact import (
    CommittedCalibration,
    compile_calibration_analysis_plan,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
    calibration_artifact_ref_from_input,
    calibration_artifact_ref_from_tree,
    calibration_artifact_ref_to_tree,
)
from zlc_neutral_atom.logic_nodes.readout.bimodal import BimodalFit, fit_bimodal


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
    reference = CalibrationArtifactRef(
        "tasks/calibration/run-001/calibration.json"
    )
    tree = calibration_artifact_ref_to_tree(reference)

    assert tuple(field.name for field in fields(CalibrationArtifactRef)) == (
        "record_path",
    )
    assert reference.record_path == "tasks/calibration/run-001/calibration.json"
    assert reference.target_ref == reference.record_path
    assert calibration_artifact_ref_from_tree(tree) == reference

    for invalid in (
        "run-001/calibration.json",
        "tasks/calibration/run-001/other.json",
        "tasks/other/run-001/calibration.json",
        "../tasks/calibration/run-001/calibration.json",
        "/tasks/calibration/run-001/calibration.json",
        "tasks\\calibration\\run-001\\calibration.json",
        "tasks/calibration//run-001/calibration.json",
    ):
        with pytest.raises((TypeError, ValueError)):
            CalibrationArtifactRef(invalid)


def test_calibration_input_freezes_current_task_or_saved_record_to_one_ref(tmp_path):
    reference = CalibrationArtifactRef(
        "tasks/calibration/run-001/calibration.json"
    )
    absolute = tmp_path / reference.record_path

    assert calibration_artifact_ref_from_input(tmp_path, reference) is reference
    assert calibration_artifact_ref_from_input(tmp_path, absolute) == reference
    assert calibration_artifact_ref_from_input(
        tmp_path,
        reference.record_path,
    ) == reference
    with pytest.raises(ValueError):
        calibration_artifact_ref_from_input(tmp_path, tmp_path.parent / "outside")


def test_compile_plan_requires_explicit_source_binding_and_deadline():
    signature = inspect.signature(compile_calibration_analysis_plan)
    assert tuple(signature.parameters) == (
        "source_capture_ref",
        "project_root",
        "request",
        "expected_readout_binding",
        "timeout_seconds",
    )
    for name in (
        "expected_readout_binding",
        "timeout_seconds",
    ):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def test_calibration_leaf_has_one_descriptor_and_no_prepared_lifecycle():
    from zlc_neutral_atom.logic_nodes.readout.calibration.logic_node import LOGIC_NODE
    from zlc_neutral_atom.logic_nodes.readout.calibration.task import (
        CalibrationTaskRequest,
    )

    assert LOGIC_NODE.definition.kind == "task"
    assert len(LOGIC_NODE.outputs) == 7
    assert LOGIC_NODE.device_requirements == (
        ("camera_instance_id", ("camera.capture",)),
        ("sequencer_instance_id", ("pulse.execute",)),
    )
    assert tuple(field.name for field in fields(CalibrationTaskRequest))[-2:] == (
        "camera_instance_id",
        "sequencer_instance_id",
    )
    assert tuple(field.name for field in fields(CommittedCalibration)) == (
        "reference",
        "result",
    )


def test_removed_calibration_commit_wrappers_are_not_reintroduced():
    import zlc_neutral_atom.logic_nodes.readout.calibration.artifact as repository_module

    for removed in (
        "CalibrationRepository",
        "CALIBRATION_MANIFEST_FORMAT",
        "CalibrationCommit",
        "CalibrationCheckpoint",
        "analyze_calibration",
    ):
        assert not hasattr(repository_module, removed)
