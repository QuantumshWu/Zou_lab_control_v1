"""Operator flows that the tutorial spine does not reach, run for real."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np

import Zou_lab_control.api as zlc

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


def test_mot_task_runs_through_the_generic_node_host(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "mot-workspace"
    workspace_paths = _workspace(workspace)
    with zlc.connect("virtual", workspace=workspace_paths) as exp:
        request = exp.nodes.mot_field.build(
            pulse=MOT_FIELD_PULSE.name,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
            span=2.0,
            points=2,
            roi_cx=None,
            roi_cy=None,
            roi_radius=8.0,
            camera_instance_id="mot-camera",
            sequencer_instance_id="sequencer",
        )
        assert request.camera_instance_id == "mot-camera"
        assert request.sequencer_instance_id == "sequencer"
        result = exp.nodes.mot_field.run(
            pulse=MOT_FIELD_PULSE.name,
            center_x=0.0,
            center_y=0.0,
            center_z=0.0,
            span=2.0,
            points=2,
            roi_cx=None,
            roi_cy=None,
            roi_radius=8.0,
            camera_instance_id="mot-camera",
            sequencer_instance_id="sequencer",
        )

    assert len(result.best_field) == 3
    assert np.isfinite(result.best_field).all()
    assert np.isfinite(result.best_intensity)
    assert result.best_intensity == float(np.max(result.intensity))
    records = tuple((workspace / "runs" / "camera").glob("*/capture.json"))
    assert len(records) == 1


def test_readout_duration_request_uses_the_current_node_contract(
    tmp_path: Path,
) -> None:
    """The source-neutral Measurement is authored through ``exp.nodes``.

    Execution additionally requires an explicitly connected y signal.  That
    connection belongs to the workbench/SignalPlane, so this operator test
    only exercises the public request contract; the bound execution path is
    covered by the owning duration-fidelity contract tests.
    """

    with zlc.connect("virtual", workspace=_workspace(tmp_path / "duration")) as exp:
        request = exp.nodes.readout_duration_fidelity.build(
            sequencer_instance_id="sequencer",
            pulse="probe_template.json",
            duration=(2.0, 4.0, 2),
            shots=2,
        )

        assert request.sequencer_instance_id == "sequencer"
        assert request.pulse == "probe_template.json"
        assert request.duration_seconds == (2e-6, 4e-6)
        assert request.shots == 2
        assert exp.nodes.readout_duration_fidelity.descriptor.input_specs
