import numpy as np

from Zou_lab_control.neutral_atom.operations.logic import (
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
