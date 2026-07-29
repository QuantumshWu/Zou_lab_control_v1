"""The operator flows that the tutorial spine does not reach, run for real.

`test_tutorial_notebook_spine` already walks capture -> sitemap -> detect, which
is the arc a reader follows.  This test keeps the independent committed
capture -> Fit -> Figure notebook composition under one real virtual session.

Area/Cross/Fit signals published by a live Figure are exercised at the
TaskConsole/Figure boundary.  They are deliberately not Camera Measurement
request fields: display selection must never reconfigure acquisition.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import Zou_lab_control.api as zlc
from zlc_data.axis import AxisSourceRef, SPATIAL_X, SPATIAL_Y
from zlc_data.fit_model import fit_model_catalog
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.logic_nodes.mot_field import MotFieldTaskIntent
import zlc_neutral_atom.logic_nodes.mot_field.mot_field_task as mot_task_impl

ROOT = Path(__file__).resolve().parents[1]
IMAGING_PULSE = ROOT / "pulses" / "imaging_template.json"
MOT_FIELD_PULSE = ROOT / "pulses" / "mot_field_template.json"


def test_fit_and_figure_run_on_the_virtual_installation() -> None:
    with tempfile.TemporaryDirectory() as workspace:
        with zlc.connect("virtual", repository=Path(workspace) / "ws") as exp:
            capture = exp.run(exp.readout.capture_request(IMAGING_PULSE))

            # The axes come from the committed capture rather than being guessed
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


def test_mot_task_analyzes_once_and_reuses_that_result_for_every_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    analyzed = []
    reported = []
    projected = []
    original_analyze = mot_task_impl.analyze_mot_scan
    original_report = mot_task_impl.write_mot_field_report
    original_outputs = mot_task_impl.mot_field_final_outputs

    def count_analysis(request, source):
        result = original_analyze(request, source)
        analyzed.append(result)
        return result

    def observe_report(result, folder):
        reported.append(result)
        return original_report(result, folder)

    def observe_outputs(result, source):
        projected.append((result, source))
        return original_outputs(result, source)

    monkeypatch.setattr(mot_task_impl, "analyze_mot_scan", count_analysis)
    monkeypatch.setattr(mot_task_impl, "write_mot_field_report", observe_report)
    monkeypatch.setattr(mot_task_impl, "mot_field_final_outputs", observe_outputs)

    workspace = tmp_path / "mot-workspace"
    report_folder = tmp_path / "mot-report"
    with zlc.connect("virtual", repository=workspace) as exp:
        command = exp.nodes.mot_field.prepare_mot_field_task(
            MotFieldTaskIntent(
                pulse=str(MOT_FIELD_PULSE),
                center_x=0.0,
                center_y=0.0,
                center_z=0.0,
                span=2.0,
                points=2,
                roi_cx=None,
                roi_cy=None,
                roi_radius=8.0,
                folder=str(report_folder),
                camera_role="mot_camera",
            )
        )
        reference = command.start().result()
        assert isinstance(reference, CaptureArtifactRef)

        first = command.final_dataset_outputs(reference)
        second = command.final_dataset_outputs(reference)

    assert len(analyzed) == 1
    assert reported == [analyzed[0]]
    assert len(projected) == 2
    assert all(result is analyzed[0] for result, _source in projected)
    assert projected[0][1] is projected[1][1]
    assert first.keys() == second.keys() == {"mot_field", "scan"}
    assert first["mot_field"].join_digest == second["mot_field"].join_digest
    assert first["scan"].snapshot is projected[0][1].snapshot
    assert second["scan"].snapshot is projected[0][1].snapshot
    assert (report_folder / "mot_field_scan.npz").is_file()
