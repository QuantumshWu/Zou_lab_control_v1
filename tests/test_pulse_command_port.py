"""Current declarative pulse application and Workbench composition boundary."""

from __future__ import annotations

from dataclasses import fields
import inspect
from pathlib import Path

from Zou_lab_control.workbench import (
    open_offline_pulse_workbench,
    open_pulse_workbench,
)
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.pulse_application import (
    PulseRunRequest,
    PulseTargetDescriptor,
)
from zlc_pulse import (
    PulseExecutionForm,
    load_pulse_document,
    resolve_api_parameters,
)


ROOT = Path(__file__).parents[1]
IMAGING_TEMPLATE = ROOT / "zlc_neutral_atom" / "assets" / "imaging_template.json"


def _descriptor() -> PulseTargetDescriptor:
    document = load_pulse_document(IMAGING_TEMPLATE)
    return PulseTargetDescriptor(
        DeviceRef("installation-a", "runtime-a", "sequencer"),
        document.target,
        50e6,
        0x1234ABCD,
        4_096,
    )


def test_pulse_target_descriptor_is_capability_free() -> None:
    descriptor = _descriptor()

    assert {field.name for field in fields(PulseTargetDescriptor)} == {
        "sequencer_ref",
        "target",
        "clock_hz",
        "geometry_fingerprint",
        "resident_scan_point_capacity",
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
    document = resolve_api_parameters(
        document,
        {
            parameter.parameter_id: document.field_value(parameter.field)[0]
            for parameter in document.api_parameters
        },
    )
    request = PulseRunRequest(
        document,
        PulseExecutionForm.STATIC_ONCE,
        descriptor.sequencer_ref,
        3.0,
    )

    assert request.document is document
    assert request.sequencer_ref == descriptor.sequencer_ref
    assert request.execution_form is PulseExecutionForm.STATIC_ONCE
    assert {field.name for field in fields(PulseRunRequest)} == {
        "document",
        "execution_form",
        "sequencer_ref",
        "timeout_seconds",
    }
    assert not hasattr(request, "port")
    assert not hasattr(request, "device")


def test_current_workbench_entry_points_never_accept_a_raw_sequencer() -> None:
    online = inspect.signature(open_pulse_workbench).parameters
    offline = inspect.signature(open_offline_pulse_workbench).parameters

    assert tuple(online) == ("experiment", "document", "path")
    assert "sequencer" not in online and "command_port" not in online
    assert "sequencer" not in offline and "command_port" not in offline
    assert "target" in offline and "time_step_ns" in offline


def test_standalone_launcher_composes_the_current_product_surface() -> None:
    source = (ROOT / "pulse_gui.py").read_text(encoding="utf-8")

    assert 'connect(\n            "virtual"' in source
    assert "open_pulse_workbench" in source
    assert "open_offline_pulse_workbench" in source
    assert "managed_pulse_command_port" not in source
    assert "RemoteSequencer" not in source
    assert "VirtualSequencer" not in source
