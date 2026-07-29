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

from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.runtime.run import RunId, RunSnapshot, RunState
from zlc_neutral_atom.logic_nodes.pulse_scan import ScanArtifactRef
from zlc_neutral_atom.runtime.hosted_run import HostedRun

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_hosted_run_does_not_import_the_application_or_gui_layer():
    """Layering: the hosted Run takes prepare/start closures, not authority."""

    import ast

    tree = ast.parse((REPO / "zlc_neutral_atom" / "runtime" / "hosted_run.py")
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

    node = HostedRun(
        definition_key=DefinitionKey("test", "final-result"),
        request={},
        instance_id="final-result-instance",
        dataset_output_declarations=(),
        prepare=lambda request: request,
        qualify_output=lambda name: f"@logic/final-result-instance/{name}",
        request_owner_wake=lambda: None,
    )
    node._handle = Handle()
    try:
        assert node.poll() is terminal
        assert node.final_result_resolved
        assert node.final_result == reference
    finally:
        node.shutdown()


def test_run_attachment_calls_the_capability_live_output_starter() -> None:
    """The generic host must not shadow the injected domain start adapter."""

    import time

    from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema
    from zlc_neutral_atom.catalog import MeasurementDefinition
    from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
    from zlc_neutral_atom.logic_node_declaration import (
        LogicNodeDeclaration,
        OutputPresentation,
    )
    from zlc_neutral_atom.processing.signal_plane import SignalDataPlane
    from zlc_neutral_atom.runtime.live_output_host import LiveDatasetHost
    from zlc_workbench.task_console.attachment_builders import run_attachment
    from zlc_workbench.task_console.capability import ConsoleNodeHost
    from zlc_workbench.task_console.declaration_projection import (
        project_declaration_spec,
    )

    key = DefinitionKey("tests", "live-output-start")
    declaration = LogicNodeDeclaration(
        definition=MeasurementDefinition(
            key,
            "Live output",
            "tests.LiveOutputRequest",
            "tests.LiveOutputBinding",
        ),
        description="test live start seam",
        authoring_schema=AuthoringSchema(
            (AuthoringField("enabled", "bool", "Enabled", default=True),)
        ),
        input_specs=(),
        outputs=(
            OutputPresentation(
                DatasetOutputDeclaration("value", "tests.live-output"),
                "value",
                "Value",
            ),
        ),
        build_request=lambda values: bool(values["enabled"]),
        bind_request=lambda request, _inputs: request,
    )
    spec = project_declaration_spec(declaration)
    plane = SignalDataPlane()
    host = ConsoleNodeHost(
        data_plane=plane,
        resolve_inputs=lambda _spec, _values: {},
        request_owner_wake=lambda: None,
    )
    observed: list[tuple[object, object]] = []
    terminal = RunSnapshot(
        RunId("live-output-start-run"),
        RunState.SUCCEEDED,
        "complete",
        True,
        None,
        None,
        (),
        None,
    )

    class Handle:
        run_id = terminal.run_id

        @staticmethod
        def snapshot():
            return terminal

        @staticmethod
        def result(*, timeout=None):
            assert timeout == 0.0
            return None

        @staticmethod
        def cancel(_reason=""):
            raise AssertionError("completed handle must not be cancelled")

    def capability_start(command, live_host):
        observed.append((command, live_host))
        return Handle()

    attachment = run_attachment(
        spec,
        bind_request=lambda request, _inputs: request,
        prepare=lambda request: ("prepared", request),
        start_with_live_output=capability_start,
    )
    node = attachment.create_node(
        host,
        spec,
        {"enabled": True},
        "live-output-instance",
    )
    try:
        node.start()
        deadline = time.monotonic() + 2.0
        while (
            (not observed or not node.final_result_resolved)
            and time.monotonic() < deadline
        ):
            node.poll()
            time.sleep(0.005)
        assert observed and observed[0][0] == ("prepared", True)
        assert isinstance(observed[0][1], LiveDatasetHost)
        assert node.final_result_resolved
    finally:
        node.poll()
        node.shutdown()
        plane.close()
