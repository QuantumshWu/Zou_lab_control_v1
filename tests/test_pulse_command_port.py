"""Current declarative pulse application and Workbench composition boundary."""

from __future__ import annotations

from dataclasses import fields
import inspect
from pathlib import Path

from Zou_lab_control.workbench import open_pulse_editor
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.devices.sequencer.application import (
    PulseRunRequest,
    PulseTargetDescriptor,
)
from zlc_pulse import (
    PulseExecutionForm,
    load_pulse_document,
    pulse_target_manifest_from_lanes,
)


ROOT = Path(__file__).parents[1]
IMAGING_TEMPLATE = ROOT / "pulses" / "imaging_template.json"


def _descriptor() -> PulseTargetDescriptor:
    document = load_pulse_document(IMAGING_TEMPLATE)
    return PulseTargetDescriptor(
        DeviceRef("installation-a", "runtime-a", "sequencer"),
        pulse_target_manifest_from_lanes(document.target),
        50e6,
        0x1234ABCD,
    )


def test_pulse_target_descriptor_is_capability_free() -> None:
    descriptor = _descriptor()

    assert {field.name for field in fields(PulseTargetDescriptor)} == {
        "sequencer_ref",
        "manifest",
        "clock_hz",
        "geometry_fingerprint",
    }
    assert descriptor.time_step_ns == 20.0
    for forbidden in (
        "sequencer",
        "prepare",
        "fire",
        "set_safe_state",
        "execute_command",
    ):
        assert not hasattr(descriptor, forbidden)


def test_pulse_run_request_freezes_intent_without_a_hardware_callback() -> None:
    descriptor = _descriptor()
    document = load_pulse_document(IMAGING_TEMPLATE)
    api_values = tuple(
        (
            parameter.parameter_id,
            document.field_value(parameter.field)[0],
        )
        for parameter in document.api_parameters
    )
    request = PulseRunRequest(
        document,
        PulseExecutionForm.STATIC_ONCE,
        descriptor.sequencer_ref,
        3.0,
        api_values,
    )

    assert request.document is document
    assert request.api_values == api_values
    assert request.scan_sweep_count == 1
    assert request.sequencer_ref == descriptor.sequencer_ref
    assert request.execution_form is PulseExecutionForm.STATIC_ONCE
    assert {field.name for field in fields(PulseRunRequest)} == {
        "document",
        "execution_form",
        "sequencer_ref",
        "timeout_seconds",
        "api_values",
        "scan_sweep_count",
    }
    assert not hasattr(request, "port")
    assert not hasattr(request, "device")


def test_current_workbench_entry_points_never_accept_a_raw_sequencer() -> None:
    parameters = inspect.signature(open_pulse_editor).parameters

    assert {"experiment", "document", "path", "remote_endpoint", "workspace"} <= set(
        parameters
    )
    assert not {
        "sequencer",
        "command_port",
        "repository",
    }.intersection(parameters)


def test_standalone_launcher_composes_the_current_product_surface() -> None:
    # The launcher opens the current PulseDocument surface through one composition
    # root.  Remote selection is declarative; no raw device/client enters the script.
    source = (ROOT / "pulse_gui.py").read_text(encoding="utf-8")

    assert "from Zou_lab_control.workbench import open_pulse_editor" in source
    assert "remote_endpoint=remote_endpoint" in source
    assert '"--remote-host"' in source
    assert '"--document"' in source
    assert "connect(" not in source
    assert "managed_pulse_command_port" not in source
    assert "RemoteSequencer" not in source
    assert "VirtualSequencer" not in source
