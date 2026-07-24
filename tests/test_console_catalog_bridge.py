"""TaskConsole catalog is a headless, lossless projection of domain definitions."""

from __future__ import annotations

import ast
import pathlib

from zlc_neutral_atom.camera_measurement import CameraMeasurementRequest
from zlc_neutral_atom.catalog import definition_key_to_tree
from zlc_neutral_atom.installation import DeviceRef
from zlc_workbench.task_console.catalog_bridge import ConsoleCatalogView


REPO = pathlib.Path(__file__).resolve().parents[1]


def _camera_request_builder(*, camera_role="camera", repeat=0, frames_per_cycle=1):
    return CameraMeasurementRequest(
        DeviceRef("test-installation", "test-runtime", str(camera_role)),
        repeat=int(repeat),
        frames_per_cycle=int(frames_per_cycle),
    )


def _view() -> ConsoleCatalogView:
    return ConsoleCatalogView(
        installed_camera_roles=("camera", "mot_camera"),
        sitemap_camera_roles=("camera",),
        installed_rf_roles=(),
        camera_request_builder=_camera_request_builder,
    )


def test_every_definition_projects_to_one_stable_key_and_current_form() -> None:
    view = _view()
    specs = view.specs()
    assert specs
    assert len({spec.key for spec in specs}) == len(specs)
    assert {spec.kind for spec in specs} <= {
        "camera", "measurement", "processor", "task"
    }
    for spec in specs:
        assert view.spec_for_key(spec.key) is spec
        assert view.spec_for_definition(definition_key_to_tree(spec.key)) is spec
        assert spec.form.fields
        assert spec.title and spec.description

    camera = next(spec for spec in specs if spec.kind == "camera")
    request = camera.build_request(
        {"camera_role": "mot_camera", "frames_per_cycle": 3, "repeat": 0}
    )
    assert request.output_names == ("frame_0", "frame_1", "frame_2")
    assert tuple(output.name for output in camera.outputs_for(request)) == request.output_names


def test_the_bridge_module_is_qt_free_by_construction() -> None:
    tree = ast.parse(
        (REPO / "zlc_workbench" / "task_console" / "catalog_bridge.py").read_text(
            encoding="utf-8"
        )
    )
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert "PyQt5" not in roots and "matplotlib" not in roots, roots
