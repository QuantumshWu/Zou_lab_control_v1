"""TaskConsole consumes one explicit, closed attachment tuple."""

from __future__ import annotations

import ast
import pathlib

import pytest

from Zou_lab_control.workbench.task_console_attachments.camera_measurement import (
    camera_measurement_attachment,
)
from Zou_lab_control.workbench.task_console_attachments.occupancy import (
    occupancy_attachment,
)
from zlc_neutral_atom.catalog import definition_key_to_tree
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.logic_nodes.camera_measurement import CameraMeasurementRequest
from zlc_workbench.task_console.application_ports import TaskConsoleApplicationPorts
from zlc_workbench.task_console.catalog_bridge import ConsoleCatalogView


REPO = pathlib.Path(__file__).resolve().parents[1]


def _camera_request_builder(*, camera_role="camera", repeat=0, frames_per_cycle=1):
    return CameraMeasurementRequest(
        DeviceRef("test-installation", "test-runtime", str(camera_role)),
        repeat=int(repeat),
        frames_per_cycle=int(frames_per_cycle),
    )


def _ports_and_view() -> tuple[TaskConsoleApplicationPorts, ConsoleCatalogView]:
    attachments = (
        camera_measurement_attachment(
            installed_camera_roles=("camera", "mot_camera"),
            request_builder=_camera_request_builder,
            prepare=lambda request: request,
        ),
        occupancy_attachment(prepare=lambda request: request),
    )
    ports = TaskConsoleApplicationPorts(
        attachments=attachments,
        resolve_artifact_reference=lambda reference: reference,
    )
    return ports, ConsoleCatalogView(
        tuple(attachment.spec for attachment in ports.attachments)
    )


def test_every_supplied_attachment_projects_once_with_definition_owned_kind() -> None:
    ports, view = _ports_and_view()
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


def test_application_ports_reject_duplicate_definition_keys() -> None:
    attachment = camera_measurement_attachment(
        installed_camera_roles=("camera",),
        request_builder=_camera_request_builder,
        prepare=lambda request: request,
    )
    with pytest.raises(ValueError, match="duplicate TaskConsole attachment"):
        TaskConsoleApplicationPorts(
            attachments=(attachment, attachment),
            resolve_artifact_reference=lambda reference: reference,
        )


def test_camera_is_a_measurement_and_request_owns_frame_vocabulary() -> None:
    _ports, view = _ports_and_view()
    (camera,) = view.specs("measurement")

    assert camera.kind == "measurement"
    request = camera.build_request(
        {"camera_role": "mot_camera", "frames_per_cycle": 3, "repeat": 0}
    )
    assert request.output_names == ("frame_0", "frame_1", "frame_2")
    assert tuple(output.name for output in camera.outputs_for(request)) == (
        "frame_0",
        "frame_1",
        "frame_2",
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
