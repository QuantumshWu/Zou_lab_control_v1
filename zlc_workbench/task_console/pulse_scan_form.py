"""Headless TaskConsole authoring declaration for PulseScan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from zlc_frontend.form import FormFieldProps
from zlc_pulse.scan_template import SWEEP_API_SLOT, SWEEP_SCAN_SLOT
from zlc_neutral_atom.pulse_programs import DEFAULT_PROBE_PULSE_PATH
from zlc_neutral_atom.scan import (
    ApiSegmentTable,
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
)
from zlc_pulse import (
    commit_scan_table,
    evaluate_numeric_scan_program,
    freeze_scan_program,
    load_pulse_document,
)

from .pulse_scan_binding import PulseScanBindingIntent

__all__ = [
    "PulseScanFormSpec",
    "build_pulse_scan_binding",
    "pulse_scan_form",
]


@dataclass(frozen=True, slots=True)
class PulseScanFormSpec:
    """Explicit PulseScan presenter declaration.

    The path and signal are ordinary reusable form leaves.  The slot table and
    program editor are a Workbench-owned structured presenter, not a synthetic
    ``FormFieldKind`` in the generic frontend form engine.
    """

    pulse_field: FormFieldProps
    y_signal_field: FormFieldProps

    def __post_init__(self) -> None:
        if self.pulse_field.key != "pulse" or self.pulse_field.kind != "path":
            raise ValueError("PulseScan pulse_field must be the 'pulse' path field")
        if (
            self.y_signal_field.key != "y_signal"
            or self.y_signal_field.kind != "signal"
        ):
            raise ValueError(
                "PulseScan y_signal_field must be the 'y_signal' signal field"
            )

    @property
    def fields(self) -> tuple[FormFieldProps, FormFieldProps]:
        """The ordinary leaves used for signal-provider projection."""

        return (self.pulse_field, self.y_signal_field)

    @property
    def keys(self) -> tuple[str, str, str]:
        return ("pulse", "pulse_slots", "y_signal")

    def default_values(self) -> dict[str, object]:
        return {
            "pulse": self.pulse_field.default,
            "pulse_slots": {},
            "y_signal": self.y_signal_field.default,
        }


def pulse_scan_form() -> PulseScanFormSpec:
    return PulseScanFormSpec(
        pulse_field=FormFieldProps(
            "pulse",
            "path",
            "Pulse template",
            default=DEFAULT_PROBE_PULSE_PATH,
            required=True,
            path_mode="file",
            base_dir="pulses",
            file_filter="Pulse program (*.json);;All files (*)",
            description=(
                "Current PulseDocument whose declared scan/API slots are edited below"
            ),
        ),
        y_signal_field=FormFieldProps(
            "y_signal",
            "signal",
            "Exact source (y)",
            required=True,
            description=(
                "Current scan-clocked sources are Camera frame, Occupancy "
                "counts/occupied, or Figure Area data derived from one of "
                "them. Static/display-only signals are rejected at Start. "
                "The scan binds a dedicated exact source pipeline and never "
                "samples a displayed/latest raster."
            ),
        ),
    )


def build_pulse_scan_binding(
    values: Mapping[str, object],
) -> PulseScanBindingIntent:
    """Commit one authored program with the Pulse editor's sole evaluator."""

    pulse = values.get("pulse")
    if not pulse:
        raise ValueError("pulse scan needs a PulseDocument path")
    document = load_pulse_document(pulse)
    slots = dict(values.get("pulse_slots") or {})
    sweep_kind = str(slots.get("sweep_kind") or "")
    source = str(slots.get("program") or "")
    y_signal = values.get("y_signal")
    if not isinstance(y_signal, str) or not y_signal.strip():
        raise ValueError(
            "Pulse scan requires an exact Camera frame, Occupancy "
            "counts/occupied, or Figure Area signal"
        )
    if sweep_kind == SWEEP_SCAN_SLOT:
        table, _normalization = freeze_scan_program(document, source)
        committed = commit_scan_table(
            document,
            table,
            recipe_source=source,
        )
        api_values = dict(slots.get("api") or {})
        api_order = tuple(
            parameter.parameter_id for parameter in committed.api_parameters
        )
        expected = set(api_order)
        missing = tuple(
            parameter_id
            for parameter_id in api_order
            if parameter_id not in api_values
        )
        extra = tuple(
            parameter_id for parameter_id in api_values if parameter_id not in expected
        )
        if missing or extra:
            raise ValueError(
                "SCAN_SLOT requires one fixed value for every API parameter; "
                f"missing={missing}, extra={extra}"
            )
        program = AutonomousScanSlotProgram(
            committed,
            tuple(
                (parameter_id, api_values[parameter_id])
                for parameter_id in api_order
            ),
        )
        return PulseScanBindingIntent(program, y_signal.strip())
    if sweep_kind == SWEEP_API_SLOT:
        columns = tuple(
            parameter.parameter_id for parameter in document.api_parameters
        )
        rows = evaluate_numeric_scan_program(source, width=len(columns))
        program = ApiSlotSegmentedProgram(
            document,
            ApiSegmentTable(columns, rows),
            "Explicit API-slot sweep authored in TaskConsole",
        )
        return PulseScanBindingIntent(program, y_signal.strip())
    raise ValueError("choose a SCAN_SLOT or API_SLOT sweep")
