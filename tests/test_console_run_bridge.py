"""RUN seam contract: a console node owns a real Run, and never on the GUI thread.

The seam is worth a test only where it touches the domain for real: a frozen
request from the CATALOG seam starts an actual monitor Run against a virtual
installation, reaches RUNNING, and cancels to a terminal state -- with every
prepare/start round trip on the worker, which is what keeps the board alive
while a camera is opening.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_a_console_node_starts_and_cancels_a_real_monitor_run():
    code = (
        "import tempfile, pathlib, time\n"
        "from Zou_lab_control.notebook import connect\n"
        "from Zou_lab_control.notebook.facade import "
        "_prepare_camera_monitor_for_workbench\n"
        "from zlc_frontend.figure import DatasetId, FigureEvaluationPolicy\n"
        "from zlc_workbench.live import LiveDatasetSlot\n"
        "from zlc_workbench.task_console.catalog_bridge import ConsoleCatalogView\n"
        "from zlc_workbench.task_console.run_bridge import ConsoleRunNode\n"
        "exp = connect('virtual', repository=pathlib.Path(tempfile.mkdtemp())/'ws')\n"
        "try:\n"
        "    view = ConsoleCatalogView(exp)\n"
        "    spec = next(s for s in view.specs() if s.kind == 'camera')\n"
        "    woke = []\n"
        "    node = ConsoleRunNode(spec, {},\n"
        "        prepare=lambda r: _prepare_camera_monitor_for_workbench(exp, r),\n"
        "        request_owner_wake=lambda: woke.append(1))\n"
        # a blank role field must resolve to the FREE-RUNNING camera, not the capture one
        "    assert node.request.camera_ref.role == 'monitor_camera'\n"
        "    slots = []\n"
        "    def start(command):\n"
        "        def factory(view_spec):\n"
        "            slot = LiveDatasetSlot(view_spec, dataset_id=DatasetId('console-monitor-1'),\n"
        "                                   evaluation_policy=FigureEvaluationPolicy(),\n"
        "                                   retain_on_terminal=False)\n"
        "            slots.append(slot)\n"
        "            return slot\n"
        "        return command.start_with_view(downstream_peak_bytes=1 << 20, factory=factory)\n"
        "    node.start(start)\n"
        "    t = time.monotonic() + 60\n"
        "    while node.handle is None and node.last_error is None and time.monotonic() < t:\n"
        "        node.poll(); time.sleep(0.05)\n"
        "    assert node.last_error is None, node.last_error\n"
        "    assert node.running and node.poll().state.name == 'RUNNING'\n"
        "    assert woke, 'the worker must wake the owner, never block it'\n"
        "    assert slots, 'the view factory must have been called on the worker'\n"
        "    node.cancel()\n"
        "    t = time.monotonic() + 60\n"
        "    while node.running and time.monotonic() < t:\n"
        "        node.poll(); time.sleep(0.05)\n"
        "    assert not node.running\n"
        "    assert node.poll().state.terminal\n"
        "    node.shutdown()\n"
        "finally:\n"
        "    exp.close()\n"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)


def test_the_run_bridge_does_not_import_the_notebook_facade():
    """Layering: the console package takes prepare/start closures, not authority."""

    import ast

    tree = ast.parse((REPO / "zlc_workbench" / "task_console" / "run_bridge.py")
                     .read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert "Zou_lab_control" not in roots and "PyQt5" not in roots, roots
