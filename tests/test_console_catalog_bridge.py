"""TaskConsole consumes one explicit, closed attachment tuple."""

from __future__ import annotations

import ast
import pathlib

import pytest

from Zou_lab_control.api import WorkspacePaths, connect
from Zou_lab_control.workbench._composition import task_console_ports
from zlc_neutral_atom.logic_nodes.camera_measurement.definition import (
    CAMERA_MEASUREMENT_LOGIC_NODE,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.declaration import (
    OCCUPANCY_LOGIC_NODE,
)
from zlc_neutral_atom.catalog import definition_key_to_tree
from zlc_neutral_atom.logic_node_declaration import OutputPresentation
from zlc_workbench.task_console.application_ports import TaskConsoleApplicationPorts
from zlc_workbench.task_console.catalog_bridge import ConsoleCatalogView
from zlc_workbench.task_console.input_binding import (
    freeze_input_selections,
    project_input_fields,
)


REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def composed_ports_and_view(tmp_path):
    experiment = connect(
        "virtual",
        workspace=WorkspacePaths.for_workspace((tmp_path / "workspace").resolve()),
    )
    try:
        ports = task_console_ports(experiment)
        yield experiment, ports, ConsoleCatalogView(
            tuple(attachment.spec for attachment in ports.attachments)
        )
    finally:
        experiment.close()


def test_every_supplied_attachment_projects_once_with_definition_owned_kind(
    composed_ports_and_view,
) -> None:
    _experiment, ports, view = composed_ports_and_view
    specs = view.specs()

    assert specs == tuple(attachment.spec for attachment in ports.attachments)
    assert len({spec.key for spec in specs}) == len(ports.attachments)
    assert {spec.kind for spec in specs} == {"measurement", "processor", "task"}
    assert all(ports.attachment_for(spec.key).spec is spec for spec in specs)
    for spec in specs:
        assert view.spec_for_key(spec.key) is spec
        assert view.spec_for_definition(definition_key_to_tree(spec.key)) is spec
        assert spec.form.fields
        assert spec.title and spec.description
        assert spec.definition is spec.declaration.definition


def test_application_ports_reject_duplicate_definition_keys(
    composed_ports_and_view,
) -> None:
    _experiment, ports, _view = composed_ports_and_view
    attachment = ports.attachment_for(CAMERA_MEASUREMENT_LOGIC_NODE.definition.key)
    assert attachment is not None
    with pytest.raises(ValueError, match="duplicate TaskConsole attachment"):
        TaskConsoleApplicationPorts(
            attachments=(attachment, attachment),
            data_plane=ports.data_plane,
            tasks_root=ports.tasks_root,
            output_root=ports.output_root,
        )


def test_camera_is_a_measurement_and_request_owns_frame_vocabulary(
    composed_ports_and_view,
) -> None:
    experiment, _ports, view = composed_ports_and_view
    camera = view.spec_for_definition(
        definition_key_to_tree(CAMERA_MEASUREMENT_LOGIC_NODE.definition.key)
    )
    assert camera is not None

    assert camera.kind == "measurement"
    role_field = next(field for field in camera.form.fields if field.key == "camera_role")
    assert tuple(choice.value for choice in role_field.choices) == tuple(
        experiment.device_catalog.roles("camera")
    )
    assert role_field.default == "camera"
    request = camera.build_request(
        {"camera_role": "camera", "frames_per_cycle": 3, "repeat": 0}
    )
    assert tuple(output.name for output in request.output_declarations) == (
        "frame_0",
        "frame_1",
        "frame_2",
    )
    presentations = camera.outputs_for(request)
    assert all(isinstance(output, OutputPresentation) for output in presentations)
    assert tuple(output.name for output in presentations) == (
        "frame_0",
        "frame_1",
        "frame_2",
    )
    assert tuple(output.declaration for output in presentations) == (
        request.output_declarations
    )
    assert {output.axis_label for output in presentations} == {"Counts"}
    configured = camera.build_request(
        {
            "camera_role": "camera",
            "frames_per_cycle": 1,
            "exposure": 0.013,
            "repeat": 0,
        }
    )
    assert configured.exposure_seconds == 0.013


def test_saved_calibration_requires_one_explicit_record_path() -> None:
    fields = project_input_fields(
        OCCUPANCY_LOGIC_NODE.input_specs,
        path_presentations={
            hint.field_key: hint
            for hint in OCCUPANCY_LOGIC_NODE.input_path_presentations
        },
    )
    path = next(
        field for field in fields if field.key == "calibration_path"
    )
    assert path.default is None
    assert path.file_filter.startswith("Calibration record (calibration.json)")

    with pytest.raises(ValueError, match="select an explicit saved calibration"):
        freeze_input_selections(
            OCCUPANCY_LOGIC_NODE.input_specs,
            {
                "camera_frame": "@logic/camera/frame_0",
                "calibration_source": "saved",
            },
        )


def test_the_generic_bridge_is_qt_free_and_has_no_concrete_node_dispatch() -> None:
    tree = ast.parse(
        (REPO / "zlc_workbench" / "task_console" / "catalog_bridge.py").read_text(
            encoding="utf-8"
        )
    )
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)

    roots = {module.split(".")[0] for module in modules}
    assert "PyQt5" not in roots and "matplotlib" not in roots, roots
    assert "Zou_lab_control" not in roots, modules
    assert not any(
        module == "zlc_neutral_atom.logic_nodes"
        or module.startswith("zlc_neutral_atom.logic_nodes.")
        for module in modules
    ), modules
