"""CATALOG seam contract: the console vocabulary is a faithful catalog projection.

Every composed domain Definition maps to exactly ONE ConsoleNodeSpec (an unknown
definition type refuses rather than dropping -- the omission the projection
exists to catch), the projection layer stays headless (no Qt import), and the
camera entry freezes a real typed request against a virtual installation.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_every_definition_projects_to_exactly_one_spec_headless():
    code = (
        "import tempfile, pathlib\n"
        "from Zou_lab_control.notebook import connect\n"
        "from zlc_workbench.task_console.catalog_bridge import ConsoleCatalogView\n"
        "from zlc_workbench.task_console.intent import compose_task_console_catalog\n"
        "import sys\n"
        "exp = connect('virtual', repository=pathlib.Path(tempfile.mkdtemp())/'ws')\n"
        "try:\n"
        "    view = ConsoleCatalogView(exp)\n"
        "    catalog = compose_task_console_catalog()\n"
        "    assert len(view.specs()) == len(catalog.definitions)\n"
        "    keys = [spec.key for spec in view.specs()]\n"
        "    assert len(set(keys)) == len(keys)\n"
        "    assert {spec.kind for spec in view.specs()} <= {'camera', 'measurement', 'processor', 'task'}\n"
        "    for spec in view.specs():\n"
        "        assert view.spec_named(spec.title) is spec\n"
        "        assert spec.form_spec.fields\n"
        "        assert spec.declared_outputs\n"
        "    camera = next(s for s in view.specs() if s.kind == 'camera')\n"
        "    request = camera.build_request({'camera_role': view.camera_roles()[0]})\n"
        "    assert type(request).__name__ == 'CameraMonitorRequest'\n"
        "    assert not any(name == 'PyQt5' or name.startswith('PyQt5.') for name in sys.modules)\n"
        "finally:\n"
        "    exp.close()\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)


def test_the_bridge_module_is_qt_free_by_construction():
    tree = ast.parse((REPO / "zlc_workbench" / "task_console" / "catalog_bridge.py")
                     .read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert "PyQt5" not in roots and "matplotlib" not in roots, roots
