"""The operator flows that the tutorial spine does not reach, run for real.

`test_tutorial_notebook_spine` already walks capture -> sitemap -> detect, which
is the arc a reader follows.  This test keeps the independent FINAL
capture -> Fit -> Figure notebook composition under one real virtual session.

Area/Cross/Fit signals published by a live Figure are exercised at the
TaskConsole/Figure boundary.  They are deliberately not Camera Measurement
request fields: display selection must never reconfigure acquisition.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import tempfile

import numpy as np

import Zou_lab_control.api as zlc
from zlc_data.axis import AxisSourceRef, SPATIAL_X, SPATIAL_Y
from zlc_data.fit_model import fit_model_catalog
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.logic_nodes.mot_field.mot_field_task import MotFieldTaskIntent

ROOT = Path(__file__).resolve().parents[1]
IMAGING_PULSE = ROOT / "pulses" / "imaging_template.json"
MOT_FIELD_PULSE = ROOT / "pulses" / "mot_field_template.json"
PROBE_PULSE = ROOT / "pulses" / "probe_template.json"


def _workspace(project_root: Path) -> zlc.WorkspacePaths:
    project = project_root.resolve()
    pulses = project / "pulses"
    pulses.mkdir(parents=True, exist_ok=True)
    for source in (IMAGING_PULSE, MOT_FIELD_PULSE, PROBE_PULSE):
        shutil.copy2(source, pulses / source.name)
    return zlc.WorkspacePaths.for_workspace(project)


def test_fit_and_figure_run_on_the_virtual_installation() -> None:
    with tempfile.TemporaryDirectory() as workspace:
        with zlc.connect(
            "virtual",
            workspace=_workspace(Path(workspace) / "ws"),
        ) as exp:
            capture = exp.run(exp.readout.capture_request(IMAGING_PULSE.name))

            # The axes come from the FINAL capture rather than being guessed
            # from array rank or copied into a Camera Measurement request.
            source_schema = exp.readout.load_capture(capture).frame_source.schema
            frame_axes = source_schema.cell_schema.data_axes
            assert len(frame_axes) == 2, "a camera frame carries two spatial axes"
            axes_by_role = {axis.role: axis for axis in frame_axes}
            x_axis = axes_by_role[SPATIAL_X]
            y_axis = axes_by_role[SPATIAL_Y]
            common_batch_sources = (
                *(
                    (AxisSourceRef.tensor(source_schema.repeat_axis.axis_id),)
                    if source_schema.repeat_axis.size > 1
                    else ()
                ),
                *(
                    (AxisSourceRef.point_rows(),)
                    if source_schema.point_table.row_count > 1
                    else ()
                ),
            )

            # --- fitting.  A 2-D model resolves its own axes on a frame; a 1-D model
            # is AMBIGUOUS there and must be told which axis, which is the domain
            # refusing to silently pick one rather than a gap.
            models = {model.model_id for model in fit_model_catalog()}
            assert {"radial_gaussian_center", "gaussian_offset"} <= models
            assert exp.fit(
                capture,
                model="radial_gaussian_center",
                independent_sources=(
                    AxisSourceRef.tensor(x_axis.axis_id),
                    AxisSourceRef.tensor(y_axis.axis_id),
                ),
                batch_sources=common_batch_sources,
            ) is not None
            assert exp.fit(
                capture,
                model="gaussian_offset",
                independent_sources=(AxisSourceRef.tensor(x_axis.axis_id),),
                batch_sources=(
                    *common_batch_sources,
                    AxisSourceRef.tensor(y_axis.axis_id),
                ),
            ) is not None

            # --- and the same capture projected into a figure: what the viewer shows
            # is a view OF this artifact, not a document reopened from disk.
            figure = exp.figure(capture, point_ordinals=(0,))
            assert figure.document.datasets, "a figure names the data it draws"


def test_mot_task_projects_stateless_outputs_from_generic_capture_final(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "mot-workspace"
    workspace_paths = _workspace(workspace)
    with zlc.connect("virtual", workspace=workspace_paths) as exp:
        command = exp.nodes.mot_field.prepare_mot_field_task(
            MotFieldTaskIntent(
                pulse=MOT_FIELD_PULSE.name,
                center_x=0.0,
                center_y=0.0,
                center_z=0.0,
                span=2.0,
                points=2,
                roi_cx=None,
                roi_cy=None,
                roi_radius=8.0,
                camera_role="mot_camera",
            )
        )
        reference = command.start().result()
        assert isinstance(reference, CaptureArtifactRef)

        result = command.mot_field_result(reference)
        first = command.final_dataset_outputs(reference)
        second = command.final_dataset_outputs(reference)

    assert len(result.best_field) == 3
    assert np.isfinite(result.best_field).all()
    assert np.isfinite(result.best_intensity)
    assert result.best_intensity == float(np.max(result.intensity))
    assert first.keys() == second.keys() == {"mot_field", "scan"}
    assert first["scan"].snapshot is not second["scan"].snapshot
    assert first["scan"].snapshot.ref == second["scan"].snapshot.ref
    assert np.array_equal(
        first["scan"].snapshot.block.values,
        second["scan"].snapshot.block.values,
    )
    assert first["mot_field"].snapshot.block.schema is command.output_schema
    assert second["mot_field"].snapshot.block.schema is command.output_schema
    assert np.array_equal(
        first["mot_field"].snapshot.block.values,
        second["mot_field"].snapshot.block.values,
    )
    assert (
        workspace_paths.output_root
        / "captures"
        / reference.record_path
    ).is_file()


def test_readout_duration_uses_the_public_experiment_path(tmp_path: Path) -> None:
    """One current API flow proves the coupled duration Measurement end to end."""

    with zlc.connect("virtual", workspace=_workspace(tmp_path / "duration")) as exp:
        calibration = exp.nodes.calibration.sitemap(frames=4)
        request = (
            exp.nodes.readout_duration_fidelity.readout_duration_fidelity_request(
                "probe_template.json",
                duration_seconds=(2e-6, 4e-6),
                shots=2,
                calibration_ref=calibration,
            )
        )
        prepared = (
            exp.nodes.readout_duration_fidelity
            .prepare_readout_duration_fidelity(request)
        )
        result = prepared.start().result()

    block = result.snapshot.block
    assert block.schema.physical_shape == (1, 2, 1)
    assert block.schema.point_table.columns[0].values == (2e-6, 4e-6)
    assert len(result.capture_terminals) == len(result.pulse_terminals) == 2
    assert not hasattr(result, "program_fingerprint")
    capture_reordered = replace(
        result,
        capture_terminals=tuple(reversed(result.capture_terminals)),
    )
    pulse_reordered = replace(
        result,
        pulse_terminals=tuple(reversed(result.pulse_terminals)),
    )
    assert capture_reordered.capture_terminals == tuple(
        reversed(result.capture_terminals)
    )
    assert pulse_reordered.pulse_terminals == tuple(
        reversed(result.pulse_terminals)
    )
    final = prepared.final_dataset_outputs(result)["fidelity"]
    assert final.snapshot is result.snapshot
