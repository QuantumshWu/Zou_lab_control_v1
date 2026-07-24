"""Focused current contracts for continuous pulse timing requests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from zlc_neutral_atom.devices.sequencer.port import (
    ContinuousPulseExecutionRequest,
    FinitePulseExecutionRequest,
    PulseTerminalAck,
    SimulatedPulseReceipt,
    validate_pulse_terminal_for_artifact,
)
from zlc_pulse.artifact import PulseExecutionForm
from zlc_pulse.compiler import compile_pulse_artifact
from zlc_pulse.document import FrozenScanTable, load_pulse_document
from zlc_pulse.scan_execution import resolve_scan_point


ROOT = Path(__file__).parents[1]
MOT_SCAN = ROOT / "pulses" / "mot_field_template.json"


def _scan_document():
    source = load_pulse_document(MOT_SCAN)
    columns = tuple(parameter.parameter_id for parameter in source.scan_parameters)
    row = tuple(source.field_value(parameter.field)[0] for parameter in source.scan_parameters)
    return replace(
        source,
        scan_table=FrozenScanTable(columns, (row,)),
        scan_recipe=None,
    )


def _continuous_case(execution_form: PulseExecutionForm):
    scan_document = _scan_document()
    document = (
        resolve_scan_point(scan_document, 0)
        if execution_form is PulseExecutionForm.CONTINUOUS_MONITOR
        else scan_document
    )
    return document, compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=execution_form,
    )


@pytest.mark.parametrize(
    "execution_form",
    [
        PulseExecutionForm.CONTINUOUS_MONITOR,
        PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
    ],
)
def test_both_cyclic_forms_use_the_continuous_request_contract(execution_form):
    document, artifact = _continuous_case(execution_form)

    request = ContinuousPulseExecutionRequest(document, artifact)

    assert request.artifact is artifact
    assert request.artifact_digest == artifact.fingerprint
    with pytest.raises(ValueError, match="finite pulse execution cannot use"):
        FinitePulseExecutionRequest(document, artifact)


@pytest.mark.parametrize(
    "execution_form",
    [
        PulseExecutionForm.CONTINUOUS_MONITOR,
        PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
    ],
)
def test_both_cyclic_forms_reject_finite_terminal_evidence(execution_form):
    _document, artifact = _continuous_case(execution_form)
    acknowledgement = PulseTerminalAck(
        "focused-session",
        "focused-binding",
        SimulatedPulseReceipt(
            artifact.fingerprint,
            "focused-simulator",
            (),
            0.0,
            0.0,
        ),
    )

    with pytest.raises(ValueError, match="cannot have a finite terminal receipt"):
        validate_pulse_terminal_for_artifact(acknowledgement, artifact)


@pytest.mark.parametrize(
    "execution_form",
    [
        PulseExecutionForm.STATIC_ONCE,
        PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
    ],
)
def test_finite_forms_cannot_enter_the_continuous_request_contract(execution_form):
    scan_document = _scan_document()
    document = (
        resolve_scan_point(scan_document, 0)
        if execution_form is PulseExecutionForm.STATIC_ONCE
        else scan_document
    )
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=execution_form,
    )

    with pytest.raises(ValueError, match="requires a cyclic continuous artifact"):
        ContinuousPulseExecutionRequest(document, artifact)
