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
import time

import pytest

from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.logic_nodes.pulse_scan.reference import ScanArtifactRef
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.hosted_run import HostedRun
from zlc_neutral_atom.runtime.resources import ResourceArbiter
from zlc_neutral_atom.runtime.run import RunController, RunPlan

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "alias",
    (
        "test-scan//scan.json",
        "test-scan/./scan.json",
        "test\\alias/scan.json",
    ),
)
def test_scan_artifact_ref_rejects_noncanonical_path_aliases(alias: str) -> None:
    with pytest.raises(ValueError):
        ScanArtifactRef(alias)


def _successful_handle(result, *, name: str):
    return RunController(ResourceArbiter()).start(
        RunPlan(
            name=name,
            resource_claims=(),
            bound_devices=(),
            preflight=lambda _context: None,
            execute=lambda _context, _prepared: result,
            cleanup=lambda _context, _prepared, _primary: CleanupReport.complete(),
            finalize=lambda _context, executed: executed,
        )
    )


def _failed_handle(message: str, *, name: str):
    def fail(_context, _prepared):
        raise RuntimeError(message)

    return RunController(ResourceArbiter()).start(
        RunPlan(
            name=name,
            resource_claims=(),
            bound_devices=(),
            preflight=lambda _context: None,
            execute=fail,
            cleanup=lambda _context, _prepared, _primary: CleanupReport.complete(),
            finalize=lambda _context, executed: executed,
        )
    )


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
    reference = ScanArtifactRef("test-scan/scan.json")
    handle = _successful_handle(reference, name="console final result")
    node = HostedRun(
        definition_key=DefinitionKey("test", "final-result"),
        request={},
        instance_id="final-result-instance",
        dataset_output_declarations=(),
        prepare=lambda request: request,
        qualify_output=lambda name: f"@logic/final-result-instance/{name}",
        request_owner_wake=lambda: None,
    )
    try:
        node.start(lambda _prepared: handle)
        deadline = time.monotonic() + 2.0
        observed = None
        while observed is None and time.monotonic() < deadline:
            observed = node.poll()
            time.sleep(0.002)
        deadline = time.monotonic() + 2.0
        while not node.final_result_resolved and time.monotonic() < deadline:
            observed = node.poll()
            time.sleep(0.002)
        assert observed is not None and observed.run_id == handle.run_id
        assert node.final_result_resolved
        assert node.final_result == reference
    finally:
        node.shutdown()


def test_command_context_sequences_two_flat_runs_without_a_composite_handle() -> None:
    """A two-stage command exposes the real child Runs and owns no Run identity."""

    command = object()
    capture = _successful_handle("capture-ref", name="calibration capture")
    analysis = _successful_handle("calibration-ref", name="calibration analysis")
    observed_children = []
    node = HostedRun(
        definition_key=DefinitionKey("test", "two-flat-runs"),
        request=command,
        instance_id="two-flat-runs-instance",
        dataset_output_declarations=(),
        prepare=lambda request: request,
        qualify_output=lambda name: f"@logic/two-flat-runs-instance/{name}",
        request_owner_wake=lambda: None,
    )

    def start_prepared(prepared):
        assert prepared is command
        source = node.command_context.start_and_wait(lambda: capture)
        observed_children.append((capture.run_id, source))
        return analysis

    try:
        node.start(start_prepared)
        deadline = time.monotonic() + 2.0
        while not node.final_result_resolved and time.monotonic() < deadline:
            node.poll()
            time.sleep(0.002)
        assert observed_children == [(capture.run_id, "capture-ref")]
        assert node.handle is analysis
        assert node.final_result == "calibration-ref"
        assert capture.run_id != analysis.run_id
        assert node.prepared_command is command
        assert not hasattr(command, "run_id")
        assert not hasattr(command, "snapshot")
    finally:
        node.shutdown()


def test_failed_second_run_keeps_capture_provenance_and_exact_run_identity() -> None:
    class CalibrationCommand:
        source_capture_ref = None

    command = CalibrationCommand()
    capture = _successful_handle("capture-ref", name="calibration capture")
    analysis = _failed_handle("analysis failed", name="calibration analysis")
    node = HostedRun(
        definition_key=DefinitionKey("test", "failed-second-run"),
        request=command,
        instance_id="failed-second-run-instance",
        dataset_output_declarations=(),
        prepare=lambda request: request,
        qualify_output=lambda name: f"@logic/failed-second-run-instance/{name}",
        request_owner_wake=lambda: None,
    )

    def start_prepared(prepared):
        prepared.source_capture_ref = node.command_context.start_and_wait(
            lambda: capture
        )
        return analysis

    try:
        node.start(start_prepared)
        observed = None
        deadline = time.monotonic() + 2.0
        while (
            (observed is None or not observed.state.terminal or not node.worker_idle)
            and time.monotonic() < deadline
        ):
            observed = node.poll()
            time.sleep(0.002)
        assert observed is not None
        assert observed.run_id == analysis.run_id
        assert observed.state.name == "FAILED"
        assert observed.primary_error == "RuntimeError: analysis failed"
        assert command.source_capture_ref == "capture-ref"
        assert node.prepared_command is command
        assert node.handle is analysis
        assert node.start_exception is None
        assert not node.final_result_resolved
    finally:
        node.shutdown()


def test_run_attachment_calls_the_leaf_owned_prepared_starter() -> None:
    """The generic host must not shadow the injected leaf start capability."""

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
    def start_prepared(command, live_host, command_context):
        assert not command_context.cancel_requested()
        observed.append((command, live_host))
        return _successful_handle(None, name="live output start")

    attachment = run_attachment(
        spec,
        bind_request=lambda request, _inputs: request,
        prepare=lambda request, _event_source: ("prepared", request),
        start_prepared=start_prepared,
    )
    node = attachment.create_node(
        host,
        spec,
        {"enabled": True},
        "live-output-instance",
    )
    plane.reserve(node)
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
