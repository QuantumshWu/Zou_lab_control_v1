"""Physical scan-column descriptions derived by the pulse authoring owner.

Ranges, native units, clock quanta and ordered slot identity describe values
that the pulse compiler will consume.  They therefore live beside
``PulseDocument`` rather than in a GUI which happens to edit one.  Frontends
receive these immutable descriptions and may choose how to draw them; they do
not rediscover pulse physics from form rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from zlc_storage import canonical_text, finite_real, positive_real

from .document import (
    FIELD_DAC,
    FIELD_DELAY,
    FIELD_DURATION,
    TIME_UNIT_TO_NS,
    ApiParameter,
    PulseDocument,
    PulseFieldRef,
    ScanParameter,
)

__all__ = [
    "ScanColumnSpec",
    "ScanSlotSchema",
    "api_column_specs",
    "parameter_value_in_unit",
    "scan_column_specs",
    "scan_slot_schema",
]


@dataclass(frozen=True)
class ScanColumnSpec:
    """One pulse parameter column and its physically valid starter range.

    ``name`` is the human-readable Python/table column identifier supplied by
    the pulse authoring owner.  DAC columns carry signed device codes.  Timing
    columns carry values in ``unit`` and require their positive native-unit
    ``quantum`` so a frontend template cannot seed off-clock values.  Duration
    values are positive; delay values are signed and carry ``is_delay=True``.
    """

    name: str
    lo: float
    hi: float
    is_dac: bool = False
    unit: str = "ns"
    label: str = ""
    quantum: float | None = None
    is_delay: bool = False

    def __post_init__(self) -> None:
        name = canonical_text(self.name, "scan column name")
        unit = canonical_text(self.unit, "scan column unit")
        label = canonical_text(self.label, "scan column label", empty=True)
        lo = finite_real(self.lo, "scan column lo")
        hi = finite_real(self.hi, "scan column hi")
        if hi < lo:
            raise ValueError("ScanColumnSpec.hi must not be below lo")
        if type(self.is_dac) is not bool:
            raise TypeError("ScanColumnSpec.is_dac must be bool")
        if type(self.is_delay) is not bool:
            raise TypeError("ScanColumnSpec.is_delay must be bool")
        if self.is_dac and self.is_delay:
            raise ValueError("a DAC scan column cannot also be a delay column")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        if self.is_dac:
            if self.quantum is not None:
                raise ValueError("a DAC scan column cannot carry a time quantum")
            return
        object.__setattr__(
            self,
            "quantum",
            positive_real(
                self.quantum,
                "time scan column native-unit quantum",
            ),
        )


@dataclass(frozen=True)
class ScanSlotSchema:
    """Compiler-relevant identity and order of a PulseDocument scan table."""

    columns: tuple[str, ...]
    fields: tuple[tuple[str, str | None, str | None, str], ...]
    time_step_ns: float
    target_abi_fingerprint: str

    def __post_init__(self) -> None:
        columns = tuple(
            canonical_text(column, "scan schema column")
            for column in tuple(self.columns)
        )
        if len(set(columns)) != len(columns):
            raise ValueError("scan schema columns must be unique")
        fields = tuple(tuple(field) for field in tuple(self.fields))
        if len(fields) != len(columns):
            raise ValueError("scan schema fields must align one-to-one with columns")
        normalized_fields = []
        for field in fields:
            if len(field) != 4:
                raise ValueError("scan schema field must contain kind/period/port/unit")
            kind, period_id, port, unit = field
            normalized_fields.append(
                (
                    canonical_text(kind, "scan schema field kind"),
                    (
                        None
                        if period_id is None
                        else canonical_text(period_id, "scan schema period_id")
                    ),
                    (
                        None
                        if port is None
                        else canonical_text(port, "scan schema port")
                    ),
                    canonical_text(unit, "scan schema field unit"),
                )
            )
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "fields", tuple(normalized_fields))
        object.__setattr__(
            self,
            "time_step_ns",
            positive_real(self.time_step_ns, "scan schema time_step_ns"),
        )
        object.__setattr__(
            self,
            "target_abi_fingerprint",
            canonical_text(
                self.target_abi_fingerprint,
                "scan schema target ABI fingerprint",
            ),
        )


def scan_slot_schema(document: PulseDocument) -> ScanSlotSchema:
    """Describe scan-column identity without mutable nominal field values."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    parameters = tuple(document.scan_parameters)
    return ScanSlotSchema(
        tuple(parameter.parameter_id for parameter in parameters),
        tuple(
            (
                parameter.field.kind,
                parameter.field.period_id,
                parameter.field.port,
                parameter.unit,
            )
            for parameter in parameters
        ),
        float(document.time_step_ns),
        document.target.abi_fingerprint,
    )


def parameter_value_in_unit(
    document: PulseDocument,
    parameter: ApiParameter | ScanParameter,
) -> int | float:
    """Read a bound pulse field in the parameter's declared native unit."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(parameter, (ApiParameter, ScanParameter)):
        raise TypeError("parameter must be ApiParameter or ScanParameter")
    value, authored_unit = document.field_value(parameter.field)
    if parameter.field.kind not in (FIELD_DURATION, FIELD_DELAY):
        return value
    return (
        float(value)
        * TIME_UNIT_TO_NS[authored_unit]
        / TIME_UNIT_TO_NS[parameter.unit]
    )


def _unique_template_name(raw_name: str, used: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw_name)).strip("_").lower()
    stem = stem or "scan_value"
    if stem[0].isdigit():
        stem = f"scan_{stem}"
    candidate = stem
    suffix = 2
    while candidate in used:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _period_context(
    document: PulseDocument,
    field: PulseFieldRef,
) -> tuple[int, object, str]:
    period_positions = {
        period.period_id: index for index, period in enumerate(document.periods)
    }
    if field.period_id is None or field.period_id not in period_positions:
        raise ValueError("duration/DAC parameter must identify a current period")
    period_index = period_positions[field.period_id]
    period = document.periods[period_index]
    title = f"Period {period_index + 1}"
    if period.name:
        title += f" {period.name!r}"
    return period_index, period, title


def scan_column_specs(document: PulseDocument) -> tuple[ScanColumnSpec, ...]:
    """Derive ordered starter sweeps from the exact PulseDocument semantics."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    specs: list[ScanColumnSpec] = []
    used_names: set[str] = set()
    for parameter in document.scan_parameters:
        field = parameter.field
        period_index, _period, period_title = _period_context(document, field)
        if field.kind == FIELD_DAC:
            if field.port is None:
                raise ValueError("DAC scan parameter must identify a port")
            port = document.target.by_key[field.port]
            port_title = port.label
            display_label = f"{period_title} · {port_title}"
            semantic_name = f"{port_title}_p{period_index + 1}"
            authored_label = "" if parameter.label == port_title else parameter.label
            template_name = _unique_template_name(
                authored_label or semantic_name,
                used_names,
            )
            if port.signed_range is None:
                raise ValueError("DAC scan port has no signed value range")
            lo, hi = port.signed_range
            specs.append(
                ScanColumnSpec(
                    template_name,
                    float(lo),
                    float(hi),
                    is_dac=True,
                    unit="value",
                    label=display_label,
                )
            )
            continue
        if field.kind != FIELD_DURATION:
            raise ValueError("only duration and DAC fields may be scan columns")
        display_label = f"{period_title} · duration"
        semantic_name = f"duration_p{period_index + 1}"
        template_name = _unique_template_name(
            parameter.label or semantic_name,
            used_names,
        )
        nominal = float(parameter_value_in_unit(document, parameter))
        tick = float(document.time_step_ns) / TIME_UNIT_TO_NS[parameter.unit]
        specs.append(
            ScanColumnSpec(
                template_name,
                tick,
                max(nominal * 2.0, 100.0 * tick),
                unit=parameter.unit,
                label=display_label,
                quantum=tick,
            )
        )
    return tuple(specs)


def api_column_specs(document: PulseDocument) -> tuple[ScanColumnSpec, ...]:
    """Derive API-slot ranges/clock quanta from the same pulse authority."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    result: list[ScanColumnSpec] = []
    for parameter in document.api_parameters:
        value = parameter_value_in_unit(document, parameter)
        if parameter.field.kind == FIELD_DAC:
            if parameter.field.port is None:
                raise ValueError("DAC API parameter must identify a port")
            port = document.target.by_key[parameter.field.port]
            if port.signed_range is None:
                raise ValueError("DAC API port has no signed value range")
            lo, hi = port.signed_range
            result.append(
                ScanColumnSpec(
                    parameter.parameter_id,
                    float(lo),
                    float(hi),
                    is_dac=True,
                    unit="value",
                    label=parameter.parameter_id,
                )
            )
            continue
        if parameter.field.kind not in (FIELD_DURATION, FIELD_DELAY):
            raise ValueError("API parameter has an unsupported pulse field kind")
        tick = float(document.time_step_ns) / TIME_UNIT_TO_NS[parameter.unit]
        is_delay = parameter.field.kind == FIELD_DELAY
        if is_delay:
            extent = max(abs(float(value)) * 2.0, 100.0 * tick)
            lo, hi = -extent, extent
        else:
            lo, hi = tick, max(float(value) * 2.0, 100.0 * tick)
        result.append(
            ScanColumnSpec(
                parameter.parameter_id,
                lo,
                hi,
                unit=parameter.unit,
                label=parameter.parameter_id,
                quantum=tick,
                is_delay=is_delay,
            )
        )
    return tuple(result)
