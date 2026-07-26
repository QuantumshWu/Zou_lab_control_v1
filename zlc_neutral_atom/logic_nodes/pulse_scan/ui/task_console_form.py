"""PulseScan-owned authoring projection consumed by Workbench surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_frontend.form import FormFieldProps

__all__ = [
    "PulseScanFormSpec",
    "pulse_scan_form",
]


@dataclass(frozen=True, slots=True)
class PulseScanFormSpec:
    """Explicit PulseScan presenter declaration.

    The owner-projected path is one ordinary reusable form leaf.  The slot
    table/program editor is the sole Workbench-owned structured presenter.
    """

    pulse_field: FormFieldProps

    def __post_init__(self) -> None:
        if self.pulse_field.key != "pulse" or self.pulse_field.kind != "path":
            raise ValueError("PulseScan pulse_field must be the 'pulse' path field")

    @property
    def fields(self) -> tuple[FormFieldProps]:
        """The ordinary owner-projected leaf beside the composite program."""

        return (self.pulse_field,)

    @property
    def keys(self) -> tuple[str, str]:
        return ("pulse", "pulse_slots")

    def default_values(self) -> dict[str, object]:
        return {
            "pulse": self.pulse_field.default,
            "pulse_slots": {},
        }


def pulse_scan_form(pulse_field: FormFieldProps) -> PulseScanFormSpec:
    return PulseScanFormSpec(
        pulse_field=pulse_field,
    )
