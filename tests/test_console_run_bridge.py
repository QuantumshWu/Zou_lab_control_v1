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
from types import SimpleNamespace

from zlc_neutral_atom.runtime.run import RunId, RunSnapshot, RunState
from zlc_neutral_atom.scan import ScanArtifactRef
from zlc_workbench.task_console.run_bridge import ConsoleRunNode

REPO = pathlib.Path(__file__).resolve().parents[1]


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


def test_successful_run_result_is_the_only_final_artifact_authority() -> None:
    reference = ScanArtifactRef("test-scan-repository", "b" * 64)
    terminal = RunSnapshot(
        RunId("console-final-result"),
        RunState.SUCCEEDED,
        "complete",
        True,
        None,
        None,
        (),
        None,
    )

    class Handle:
        def snapshot(self):
            return terminal

        def result(self, *, timeout=None):
            assert timeout == 0.0
            return reference

        def cancel(self, _reason=""):
            raise AssertionError("a successful Run must not be cancelled")

    spec = SimpleNamespace(
        key=SimpleNamespace(stable_definition_id="final-result"),
        name="Pulse scan",
        build_request=lambda values: values,
    )
    node = ConsoleRunNode(
        spec,
        {},
        prepare=lambda request: request,
        request_owner_wake=lambda: None,
    )
    node._handle = Handle()
    try:
        assert node.poll() is terminal
        assert node.final_result_resolved
        assert node.final_result == reference
    finally:
        node.shutdown()
