"""Calibration axis, validity, and pure-computation contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from zlc_data.axis import AxisId
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    CalibrationAnalysisRequest,
)
from zlc_neutral_atom.logic_nodes.readout.contracts import CalibrationCaptureLayout
from zlc_neutral_atom.logic_nodes.readout.model_contract import (
    ReadoutModelKind,
    readout_model_authoring_schema,
    readout_model_kind_from_authoring,
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
            "isolated calibration stream probe failed\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    marker = "RESULT_JSON="
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    pytest.fail(f"isolated probe returned no result marker: {completed.stdout}")


def test_request_owns_independent_expected_center_evidence():
    caller_centers = np.asarray(
        ((5.0, 5.0), (15.0, 5.0), (5.0, 15.0), (15.0, 15.0)),
        dtype="<f8",
    )
    request = CalibrationAnalysisRequest(
        layout=CalibrationCaptureLayout(AxisId("event"), (0, 2), 1),
        grid_shape_yx=(2, 2),
        expected_centers_xy=caller_centers,
        maximum_site_residual_px=2.0,
    )
    caller_centers[:] = -1.0
    assert np.array_equal(
        request.expected_centers_xy,
        np.asarray(((5, 5), (15, 5), (5, 15), (15, 15)), dtype="<f8"),
    )
    assert not request.expected_centers_xy.flags.writeable
    with pytest.raises(ValueError, match="less than half"):
        CalibrationAnalysisRequest(
            layout=request.layout,
            grid_shape_yx=(2, 2),
            expected_centers_xy=np.asarray(
                ((0, 0), (2, 0), (0, 2), (2, 2)), dtype="<f8"
            ),
            maximum_site_residual_px=1.0,
        )


def test_current_sitemap_preserves_repeat_event_image_and_site_axes(tmp_path):
    result = _run_isolated(
        """
        import json
        from pathlib import Path
        import sys

        from Zou_lab_control.api import WorkspacePaths, connect

        workspace = Path(sys.argv[1])
        experiment = connect(
            "virtual",
            workspace=WorkspacePaths.for_workspace(
                Path.cwd(),
                repository_root=workspace,
            ),
            seed=7,
        )
        try:
            reference = experiment.nodes.calibration.sitemap(frames=4)
            computation = experiment.nodes.calibration.load_calibration_computation(reference)
            artifact = computation.artifact
            source = experiment.readout.load_capture(
                artifact.source_binding.source_capture_ref
            )
            schema = source.frame_source.schema
            result = {
                "computation_type": type(computation).__name__,
                "physical_shape": list(schema.physical_shape),
                "repeat_axis": [
                    schema.repeat_axis.axis_id.value,
                    schema.repeat_axis.role.value,
                    schema.repeat_axis.size,
                ],
                "point_columns": [
                    [
                        column.coordinate_id.value,
                        column.role.value,
                        len(column.values),
                    ]
                    for column in schema.point_table.columns
                ],
                "has_grid_topology": schema.grid_topology is not None,
                "data_shape": list(schema.cell_schema.data_shape),
                "group_contexts": [
                    [[axis.value, index] for axis, index in context]
                    for context in computation.report.group_contexts
                ],
                "model_kinds": [model.kind.value for model in artifact.models],
                "threshold_shapes": [
                    list(model.thresholds.shape) for model in artifact.models
                ],
                "validity_shapes": [
                    list(model.usable_sites.mask.shape) for model in artifact.models
                ],
                "validity_axes": [
                    [axis.value for axis in model.usable_sites.axis_ids]
                    for model in artifact.models
                ],
                "site_shape": list(artifact.site_map.grid_shape_yx),
                "request_models": [
                    kind.value for kind in computation.report.request.model_kinds
                ],
            }
            print("RESULT_JSON=" + json.dumps(result, sort_keys=True))
        finally:
            experiment.close()
        """,
        tmp_path / "calibration-stream",
    )
    assert result["computation_type"] == "CalibrationComputation"
    assert result["physical_shape"] == [4, 3, 96, 128]
    assert result["repeat_axis"] == ["capture.repeat", "repeat", 4]
    assert result["point_columns"] == [
        ["capture.readout_event", "readout-event", 3]
    ]
    assert result["has_grid_topology"] is False
    assert result["data_shape"] == [96, 128]
    assert result["group_contexts"] == [
        [["capture.repeat", repeat]] for repeat in range(4)
    ]
    assert result["model_kinds"] == ["box", "psf", "uniform_psf"]
    assert result["request_models"] == ["box", "psf", "uniform_psf"]
    assert result["threshold_shapes"] == [[35], [35], [35]]
    assert result["validity_shapes"] == [[35], [35], [35]]
    assert result["validity_axes"] == [
        ["readout-site"],
        ["readout-site"],
        ["readout-site"],
    ]
    assert result["site_shape"] == [5, 7]


def test_model_batch_is_real_current_domain_not_a_scalar_only_placeholder():
    assert tuple(kind.value for kind in ReadoutModelKind) == (
        "box",
        "psf",
        "uniform_psf",
    )


def test_readout_model_selection_contract_is_owned_by_the_readout_family():
    authored_default = readout_model_authoring_schema().freeze({})["model_kind"]
    assert readout_model_kind_from_authoring(authored_default) is None
    assert readout_model_kind_from_authoring("psf") is ReadoutModelKind.PER_SITE_PSF
    with pytest.raises(ValueError, match="unknown readout model choice"):
        readout_model_kind_from_authoring("missing")
