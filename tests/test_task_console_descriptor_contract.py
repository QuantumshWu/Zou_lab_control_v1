"""Current TaskConsole boundary: descriptors in, one host factory out."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from Zou_lab_control.api import WorkspacePaths, connect
from Zou_lab_control.workbench._composition import task_console_dependencies
from zlc_neutral_atom.installation import DeviceCatalogView
from zlc_neutral_atom.logic_node import LogicNodeDescriptor
from zlc_neutral_atom.logic_node import SelectionParameterPatch
from zlc_neutral_atom.logic_nodes.camera_measurement.definition import (
    CameraMeasurementRequest,
)
from zlc_neutral_atom.logic_nodes.camera_measurement.logic_node import (
    camera_selection_parameter_patch,
)
from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointTable,
    REPEAT,
    SPATIAL_X,
    SPATIAL_Y,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
    VALID,
)
from zlc_plot import NumericRange, RectangleRange, SelectionData, SelectorKind, SelectorState
from zlc_neutral_atom.processing.signal_plane import SignalDataPlane
from zlc_workbench.task_console.logic_node_parameter_panel import (
    _device_choice_projection,
)


def _workspace(root: Path) -> WorkspacePaths:
    return WorkspacePaths.for_workspace(root.resolve())


def test_composition_exposes_only_current_task_console_facts(tmp_path: Path) -> None:
    with connect("virtual", workspace=_workspace(tmp_path / "project")) as exp:
        dependencies = task_console_dependencies(exp)
        assert set(dependencies) == {
            "descriptors",
            "device_catalog",
            "host_factory",
            "data_plane",
            "project_root",
            "pulses_root",
            "tasks_root",
            "figures_root",
            "selection_patch_sink",
        }
        assert dependencies["descriptors"] == exp.nodes.descriptors
        assert all(
            isinstance(value, LogicNodeDescriptor)
            for value in dependencies["descriptors"]
        )
        assert dependencies["device_catalog"] is exp.device_catalog
        assert isinstance(dependencies["device_catalog"], DeviceCatalogView)
        assert isinstance(dependencies["data_plane"], SignalDataPlane)
        assert callable(dependencies["host_factory"])
        assert callable(dependencies["selection_patch_sink"])

        catalog = dependencies["device_catalog"]
        for descriptor in dependencies["descriptors"]:
            for projection in _device_choice_projection(
                descriptor,
                catalog,
            ).values():
                values = tuple(choice.value for choice in projection.choices)
                assert values
                assert projection.default in values
                assert all(value in catalog for value in values)


def test_removed_task_console_owners_do_not_exist() -> None:
    root = Path(__file__).resolve().parents[1] / "zlc_workbench" / "task_console"
    for name in (
        "application_ports.py",
        "capability.py",
        "attachment_builders.py",
        "declaration_projection.py",
        "catalog_bridge.py",
        "artifact_resolution.py",
        "input_binding.py",
    ):
        assert not (root / name).exists()


def test_camera_area_patch_is_device_draft_not_measurement_or_hardware() -> None:
    repeat = AxisSpec(AxisId("repeat"), "repeat", REPEAT, 1, (0,))
    y = AxisSpec(AxisId("image.y"), "image.y", SPATIAL_Y, 4, (0, 1, 2, 3))
    x = AxisSpec(AxisId("image.x"), "image.x", SPATIAL_X, 5, (0, 1, 2, 3, 4))
    schema = DatasetSchema(
        repeat,
        PointTable(1),
        None,
        ValueSchema((y, x), ValidityContract.value(), np.dtype("uint8")),
    )
    block = DataBlock(
        BlockId("camera-area-patch"),
        DatasetRevision(0),
        np.zeros((1, 1, 4, 5), dtype=np.uint8),
        VALID,
        schema,
    )
    source = OwnedSnapshot(block.ref(StreamGenerationId("camera-area-patch")), block)
    selection = SelectionData(
        SelectorState(
            SelectorKind.AREA,
            RectangleRange(NumericRange(1, 3), NumericRange(0, 2)),
        ),
        None,
        None,
        (0,),
        0,
        _source=source,
    )
    patch = camera_selection_parameter_patch(
        CameraMeasurementRequest("camera"),
        selection,
        source,
    )
    assert isinstance(patch, SelectionParameterPatch)
    assert patch.target_instance_id == "camera"
    assert dict(patch.values) == {
        "roi_x": 1,
        "roi_y": 0,
        "roi_width": 3,
        "roi_height": 3,
    }
