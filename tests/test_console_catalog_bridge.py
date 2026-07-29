"""TaskConsole consumes one explicit, closed attachment tuple."""

from __future__ import annotations

import ast
import pathlib

import pytest

from zlc_neutral_atom.logic_nodes.camera_measurement.definition import (
    CAMERA_MEASUREMENT_LOGIC_NODE,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.declaration import (
    OCCUPANCY_LOGIC_NODE,
)
from zlc_neutral_atom.catalog import definition_key_to_tree
from zlc_neutral_atom.logic_node_declaration import OutputPresentation
from zlc_neutral_atom.processing.signal_plane import SignalDataPlane
from zlc_workbench.task_console.application_ports import TaskConsoleApplicationPorts
from zlc_workbench.task_console.catalog_bridge import ConsoleCatalogView
from zlc_workbench.task_console.declaration_projection import (
    project_processor_declaration,
    project_run_declaration,
)


REPO = pathlib.Path(__file__).resolve().parents[1]


def _ports_and_view(
    root: pathlib.Path,
) -> tuple[TaskConsoleApplicationPorts, ConsoleCatalogView]:
    path_roots = {
        "output": root / "output",
        "tasks": root / "tasks",
    }
    attachments = (
        project_run_declaration(
            CAMERA_MEASUREMENT_LOGIC_NODE,
            prepare=lambda intent, event_source: intent,
            dynamic_choices=CAMERA_MEASUREMENT_LOGIC_NODE.resolve_dynamic_choices(
                ("camera", "mot_camera", "science_camera")
            ),
            path_roots=path_roots,
        ),
        project_processor_declaration(
            OCCUPANCY_LOGIC_NODE,
            prepare=lambda request: request,
            resolve_artifact_reference=lambda binding: binding,
            path_roots=path_roots,
        ),
    )
    ports = TaskConsoleApplicationPorts(
        attachments=attachments,
        data_plane=SignalDataPlane(),
        tasks_root=root / "tasks",
        output_root=root / "output",
    )
    return ports, ConsoleCatalogView(
        tuple(attachment.spec for attachment in ports.attachments)
    )


def test_every_supplied_attachment_projects_once_with_definition_owned_kind(
    tmp_path,
) -> None:
    ports, view = _ports_and_view(tmp_path)
    specs = view.specs()

    assert specs == tuple(attachment.spec for attachment in ports.attachments)
    assert len({spec.key for spec in specs}) == len(ports.attachments)
    assert {spec.kind for spec in specs} == {"measurement", "processor"}
    assert all(ports.attachment_for(spec.key).spec is spec for spec in specs)
    for spec in specs:
        assert view.spec_for_key(spec.key) is spec
        assert view.spec_for_definition(definition_key_to_tree(spec.key)) is spec
        assert spec.form.fields
        assert spec.title and spec.description
        assert spec.definition is spec.declaration.definition


def test_application_ports_reject_duplicate_definition_keys(tmp_path) -> None:
    attachment = project_run_declaration(
        CAMERA_MEASUREMENT_LOGIC_NODE,
        prepare=lambda intent, event_source: intent,
        dynamic_choices=CAMERA_MEASUREMENT_LOGIC_NODE.resolve_dynamic_choices(
            ("camera",)
        ),
        path_roots={
            "output": tmp_path / "output",
            "tasks": tmp_path / "tasks",
        },
    )
    with pytest.raises(ValueError, match="duplicate TaskConsole attachment"):
        TaskConsoleApplicationPorts(
            attachments=(attachment, attachment),
            data_plane=SignalDataPlane(),
            tasks_root=tmp_path / "tasks",
            output_root=tmp_path / "output",
        )


def test_camera_is_a_measurement_and_request_owns_frame_vocabulary(tmp_path) -> None:
    _ports, view = _ports_and_view(tmp_path)
    (camera,) = view.specs("measurement")

    assert camera.kind == "measurement"
    role_field = next(field for field in camera.form.fields if field.key == "camera_role")
    assert tuple(choice.value for choice in role_field.choices) == (
        "camera",
        "mot_camera",
        "science_camera",
    )
    assert role_field.default == "camera"
    request = camera.build_request(
        {"camera_role": "science_camera", "frames_per_cycle": 3, "repeat": 0}
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
