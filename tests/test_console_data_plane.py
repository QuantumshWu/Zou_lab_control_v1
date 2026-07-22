"""MONITOR seam contract: changed sources freeze once, unchanged fronts are reused.

Worth a test where it touches reality: a live virtual monitor is frozen through
the data plane and must yield the camera's actual block (shape, dtype, unit off
the producer's own cell schema), a version that ADVANCES with the stream, and
coverage instead of a global shot counter -- the fiction the purge removed.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
from types import SimpleNamespace

from zlc_workbench.task_console.data_plane import ConsoleDataPlane

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_unchanged_sources_reuse_their_immutable_front() -> None:
    plane = ConsoleDataPlane()
    node = SimpleNamespace(name="camera")
    slot = object()
    calls = []
    plane._freeze_one = lambda *_args: calls.append(object()) or {}
    plane.attach(node, slot)

    plane.freeze()
    plane.freeze()
    assert len(calls) == 1

    plane.mark_changed(node)
    plane.freeze()
    assert len(calls) == 2


def test_a_live_monitor_freezes_into_real_values_with_advancing_versions():
    code = (
        "import tempfile, pathlib, time\n"
        "from Zou_lab_control.notebook import connect\n"
        "from Zou_lab_control.notebook.facade import "
        "_prepare_camera_monitor_for_workbench\n"
        "from zlc_frontend.figure import DatasetId\n"
        "from zlc_workbench.live import LiveDatasetSlot\n"
        "from zlc_workbench.task_console.catalog_bridge import ConsoleCatalogView\n"
        "from zlc_workbench.task_console.run_bridge import ConsoleRunNode\n"
        "from zlc_workbench.task_console.data_plane import ConsoleDataPlane\n"
        "exp = connect('virtual', repository=pathlib.Path(tempfile.mkdtemp())/'ws')\n"
        "try:\n"
        "    view = ConsoleCatalogView(exp)\n"
        "    spec = next(s for s in view.specs() if s.kind == 'camera')\n"
        "    plane = ConsoleDataPlane()\n"
        "    node = ConsoleRunNode(spec, {},\n"
        "        prepare=lambda r: _prepare_camera_monitor_for_workbench(exp, r),\n"
        "        request_owner_wake=lambda: None)\n"
        "    def start(command):\n"
        "        def factory(vs):\n"
        "            slot = LiveDatasetSlot(vs, dataset_id=DatasetId('console-1'),\n"
        "                                   retain_on_terminal=False)\n"
        "            plane.attach(node, slot)\n"
        "            slot.set_change_listener(lambda: plane.mark_changed(node))\n"
        "            return slot\n"
        "        return command.start_with_view(factory=factory)\n"
        "    node.start(start)\n"
        "    t = time.monotonic() + 60\n"
        "    while node.handle is None and node.last_error is None and time.monotonic() < t:\n"
        "        node.poll(); time.sleep(0.05)\n"
        "    assert node.last_error is None, node.last_error\n"
        "    time.sleep(1.5)\n"
        "    snapshot = plane.freeze()\n"
        "    assert not snapshot.failures, snapshot.failures\n"
        "    value = snapshot.value('frame')\n"
        # the projection is positional against the spec's declared outputs
        "    assert value is not None and value.name == spec.declared_outputs[0].name\n"
        "    assert value.source == spec.name\n"
        # real block: a camera frame history, typed and united by its own producer
        "    assert len(value.shape) == 4 and value.shape[-1] > 1\n"
        "    assert value.dtype is not None and value.unit and value.axes\n"
        # coverage replaces the shot clock
        "    assert value.coverage.total_cells > 0\n"
        "    assert value.behind == value.coverage.missed_events\n"
        "    first = snapshot.versions()\n"
        "    time.sleep(1.0)\n"
        "    later = plane.freeze().versions()\n"
        "    assert any(later[k] > first[k] for k in first), (first, later)\n"
        "    node.cancel()\n"
        "    t = time.monotonic() + 60\n"
        "    while node.running and time.monotonic() < t:\n"
        "        node.poll(); time.sleep(0.05)\n"
        "    node.shutdown()\n"
        "finally:\n"
        "    exp.close()\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)


def test_the_data_plane_holds_no_toolkit_and_no_domain_authority():
    """It receives slots; it never reaches for Qt, matplotlib or the facade."""

    tree = ast.parse((REPO / "zlc_workbench" / "task_console" / "data_plane.py")
                     .read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert not roots & {"PyQt5", "matplotlib", "Zou_lab_control"}, roots
