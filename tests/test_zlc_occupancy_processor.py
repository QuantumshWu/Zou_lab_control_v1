"""Current pure Occupancy Processor declaration."""

from __future__ import annotations

def test_occupancy_is_a_pure_processor_with_dataset_and_calibration_inputs():
    from zlc_neutral_atom.logic_nodes.readout.occupancy.logic_node import LOGIC_NODE
    from zlc_neutral_atom.input_spec import ArtifactInputSpec, DatasetInputSpec

    assert LOGIC_NODE.definition.kind == "processor"
    assert tuple(type(value) for value in LOGIC_NODE.input_specs) == (
        DatasetInputSpec,
        ArtifactInputSpec,
    )
    assert LOGIC_NODE.input_specs[1].allow_saved_reference
    assert tuple(value.name for value in LOGIC_NODE.outputs) == (
        "counts",
        "occupied",
        "rate",
    )
    assert LOGIC_NODE.task_previews == ()
    assert LOGIC_NODE.device_requirements == ()
    assert LOGIC_NODE.artifact_outputs == ()
    assert LOGIC_NODE.ui_contributions == ()
    assert dict(LOGIC_NODE.operations) == {}
