import numpy as np

from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.logic import (
    CalibrateReadoutTask,
    SignalSpec,
    TaskOutput,
)


def test_task_output_canonicalizes_numeric_data_and_keeps_stage_control_plane():
    def specs():
        return (
            SignalSpec("frame", "frame", points_shape=(1,), data_shape=(2, 3),
                       dtype=np.uint16, repeat_capacity=1),
            SignalSpec("progress", "progress", dtype=np.float64, repeat_capacity=1),
        )

    output = TaskOutput(spec_provider=specs)
    output.publish(
        frame=np.arange(6, dtype=np.uint16).reshape(2, 3),
        progress=0.25,
        stage="acquiring",
    )
    assert output.latest("frame").shape == (1, 1, 2, 3)
    assert output.latest("progress").shape == (1, 1, 1)
    assert output.progress == 0.25
    assert output.latest("stage") == "acquiring"
    assert "stage" not in output._schemas


def test_calibration_task_declares_frame_and_site_geometry_without_site_as_points():
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    try:
        task = CalibrateReadoutTask(
            SignalHub(), exp.devices.camera, sequencer=exp.devices.sequencer,
            grid_shape=(3, 4), threshold_frames=2)
        specs = {spec.name: spec for spec in task.output_specs()}
        assert specs["frame"].points_shape == (1,)
        assert specs["frame"].data_shape == exp.devices.camera.frame_shape
        assert specs["centers"].points_shape == (1,)
        assert specs["centers"].data_shape == (12, 2)
        assert specs["thresholds"].data_shape == (12,)

        task.output.publish(
            frame=np.zeros(exp.devices.camera.frame_shape, dtype=np.uint16),
            progress=0.5,
            stage="halfway",
        )
        assert task.output.latest("frame").shape == (
            1, 1, *exp.devices.camera.frame_shape)
    finally:
        exp.close()
