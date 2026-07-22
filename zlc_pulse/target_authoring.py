"""Pure authoring transforms for an offline pulse target manifest."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .document import FIELD_DURATION, FrozenScanTable, PulseDocument
from .manifest import PulseTargetManifest, pulse_target_manifest
from .target import (
    DAC_OFFSET_BINARY,
    PORT_CLOCK,
    PORT_DAC,
    PORT_DIGITAL,
    PulsePortSpec,
    PulseTarget,
)


@dataclass(frozen=True, slots=True)
class PulseTargetPortDraft:
    """One editable programmable port; a DAC owns its latch clock."""

    key: str
    kind: str
    signal: str
    endpoints: tuple[str, ...]
    clock_key: str | None = None
    clock_endpoint: str | None = None
    lane_order: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not isinstance(self.signal, str):
            raise TypeError("target port key and signal must be text")
        key = self.key
        signal = self.signal
        endpoints = tuple(self.endpoints)
        if (
            not key
            or not signal
            or key != key.strip()
            or signal != signal.strip()
        ):
            raise ValueError("target port key and signal must be canonical text")
        if any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in endpoints
        ):
            raise ValueError("target endpoints must be non-empty canonical text")
        if self.kind == PORT_DIGITAL:
            if len(endpoints) != 1:
                raise ValueError("a digital port requires exactly one endpoint")
            if self.clock_key is not None or self.clock_endpoint is not None:
                raise ValueError("a digital port cannot own a latch clock")
        elif self.kind == PORT_DAC:
            if len(endpoints) < 2:
                raise ValueError("a DAC port requires at least two data endpoints")
            clock_key = "" if self.clock_key is None else self.clock_key
            clock_endpoint = "" if self.clock_endpoint is None else self.clock_endpoint
            if not isinstance(clock_key, str) or not isinstance(clock_endpoint, str):
                raise TypeError("DAC latch-clock key and endpoint must be text")
            if not clock_key or not clock_endpoint:
                raise ValueError("a DAC port requires one latch-clock endpoint")
            if clock_key != clock_key.strip() or clock_endpoint != clock_endpoint.strip():
                raise ValueError("DAC latch-clock identity must be canonical text")
            object.__setattr__(self, "clock_key", clock_key)
            object.__setattr__(self, "clock_endpoint", clock_endpoint)
        else:
            raise ValueError("editable target ports must be digital or DAC")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "signal", signal)
        object.__setattr__(self, "endpoints", endpoints)
        lane_order = tuple(self.lane_order) or tuple(range(len(endpoints)))
        if sorted(lane_order) != list(range(len(endpoints))):
            raise ValueError("target lane_order must be a permutation of its width")
        object.__setattr__(self, "lane_order", lane_order)

    @property
    def width(self) -> int:
        return len(self.endpoints)


@dataclass(frozen=True, slots=True)
class PulseTargetEditImpact:
    cleared_references: tuple[str, ...] = ()

    @property
    def destructive(self) -> bool:
        return bool(self.cleared_references)


@dataclass(frozen=True, slots=True)
class PulseTargetEditResult:
    document: PulseDocument
    manifest: PulseTargetManifest
    impact: PulseTargetEditImpact


class DestructivePulseTargetEditError(ValueError):
    def __init__(self, impact: PulseTargetEditImpact) -> None:
        self.impact = impact
        super().__init__(
            "target edit removes or changes ports used by the pulse document"
        )


def pulse_target_port_drafts(
    manifest: PulseTargetManifest,
) -> tuple[PulseTargetPortDraft, ...]:
    """Project one manifest into editable logical rows without exposing clocks."""

    if not isinstance(manifest, PulseTargetManifest):
        raise TypeError("manifest must be PulseTargetManifest")
    result: list[PulseTargetPortDraft] = []
    by_key = manifest.by_key
    raw_position = {
        lane: index for index, lane in enumerate(manifest.target.raw_lanes)
    }
    for exposed in manifest.ports:
        port = manifest.target.by_key[exposed.port_key]
        if port.kind == PORT_CLOCK:
            continue
        if port.kind == PORT_DIGITAL:
            result.append(
                PulseTargetPortDraft(
                    port.key,
                    port.kind,
                    port.label,
                    exposed.endpoints,
                    lane_order=(0,),
                )
            )
            continue
        assert port.latch_clock is not None
        clock = by_key.get(port.latch_clock)
        if clock is None:
            raise ValueError(
                f"manifest DAC {port.key!r} does not expose its latch clock"
            )
        result.append(
            PulseTargetPortDraft(
                port.key,
                port.kind,
                port.label,
                exposed.endpoints,
                port.latch_clock,
                clock.endpoints[0],
                tuple(
                    sorted(port.lanes, key=raw_position.__getitem__).index(lane)
                    for lane in port.lanes
                ),
            )
        )
    return tuple(result)


def build_pulse_target_manifest(
    drafts: tuple[PulseTargetPortDraft, ...],
) -> PulseTargetManifest:
    """Build one exact target and manifest in the visible row order."""

    rows = tuple(drafts)
    if not rows or any(not isinstance(row, PulseTargetPortDraft) for row in rows):
        raise ValueError("an offline target requires at least one port row")
    keys = tuple(row.key for row in rows)
    clock_keys = tuple(row.clock_key for row in rows if row.clock_key is not None)
    if len(set((*keys, *clock_keys))) != len(keys) + len(clock_keys):
        raise ValueError("target port and latch-clock keys must be unique")
    signals = tuple(row.signal for row in rows)
    if len(set(signals)) != len(signals):
        raise ValueError("target signal names must be unique")
    all_endpoints = tuple(
        endpoint
        for row in rows
        for endpoint in (
            *row.endpoints,
            *((row.clock_endpoint,) if row.clock_endpoint is not None else ()),
        )
    )
    if len(set(all_endpoints)) != len(all_endpoints):
        raise ValueError("one target endpoint cannot belong to two output lanes")

    raw_lanes: list[str] = []
    ports: list[PulsePortSpec] = []
    endpoints_by_port: dict[str, tuple[str, ...]] = {}
    bus_index = 0

    def allocate_lanes(count: int) -> tuple[str, ...]:
        allocated = tuple(
            f"ch{index:02d}"
            for index in range(len(raw_lanes), len(raw_lanes) + count)
        )
        raw_lanes.extend(allocated)
        return allocated

    for row in rows:
        lane_block = allocate_lanes(row.width)
        lanes = tuple(lane_block[index] for index in row.lane_order)
        if row.kind == PORT_DIGITAL:
            ports.append(
                PulsePortSpec(
                    row.key,
                    PORT_DIGITAL,
                    lanes,
                    row.signal,
                    None,
                    1,
                    "binary",
                    0,
                    None,
                )
            )
            endpoints_by_port[row.key] = row.endpoints
            continue

        assert row.clock_key is not None and row.clock_endpoint is not None
        ports.append(
            PulsePortSpec(
                row.key,
                PORT_DAC,
                lanes,
                row.signal,
                bus_index,
                row.width,
                DAC_OFFSET_BINARY,
                1 << (row.width - 1),
                row.clock_key,
            )
        )
        endpoints_by_port[row.key] = row.endpoints
        clock_lanes = allocate_lanes(1)
        ports.append(
            PulsePortSpec(
                row.clock_key,
                PORT_CLOCK,
                clock_lanes,
                row.clock_key,
                None,
                1,
                "binary",
                0,
                None,
            )
        )
        endpoints_by_port[row.clock_key] = (row.clock_endpoint,)
        bus_index += 1

    target = PulseTarget(tuple(raw_lanes), tuple(ports))
    return pulse_target_manifest(target, endpoints_by_port)


def _compatible_port(old, new) -> bool:
    if old is None or new is None or old.kind != new.kind:
        return False
    return old.kind == PORT_DIGITAL or (
        old.kind == PORT_DAC and old.width == new.width
    )


def pulse_document_port_references(
    document: PulseDocument,
    incompatible: frozenset[str],
) -> tuple[str, ...]:
    references: list[str] = []
    lane_index = {lane: index for index, lane in enumerate(document.target.raw_lanes)}
    for port_key in incompatible:
        port = document.target.by_key.get(port_key)
        if port is None or port.kind == PORT_CLOCK:
            continue
        if port.kind == PORT_DIGITAL:
            index = lane_index[port.lanes[0]]
            references.extend(
                f"period {period.period_id}: {port.label} is high"
                for period in document.periods
                if period.states[index]
            )
        else:
            references.extend(
                f"period {period.period_id}: DAC {port.label} has an action"
                for period in document.periods
                if any(step.port == port_key for step in period.analog_steps)
            )
        if any(delay.port == port_key for delay in document.delays):
            references.append(f"delay: {port.label}")
        references.extend(
            f"scan parameter: {parameter.parameter_id}"
            for parameter in document.scan_parameters
            if parameter.field.port == port_key
        )
        references.extend(
            f"API parameter: {parameter.parameter_id}"
            for parameter in document.api_parameters
            if parameter.field.port == port_key
        )
    return tuple(dict.fromkeys(references))


def restrict_pulse_document_to_manifest(
    document: PulseDocument,
    manifest: PulseTargetManifest,
) -> PulseDocument:
    """Reject authoritative use of ports a backend does not expose.

    ``PulseDocument.visible_ports`` is the saved Offline authoring preference,
    not a backend capability.  Online views intersect/project it separately and
    must never rewrite that document field merely because a backend exposes a
    smaller operator manifest.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(manifest, PulseTargetManifest):
        raise TypeError("manifest must be PulseTargetManifest")
    if document.target.abi_fingerprint != manifest.target.abi_fingerprint:
        raise ValueError("PulseDocument target differs from backend manifest")
    unavailable = frozenset(
        port.key
        for port in document.target.ports
        if port.kind in (PORT_DIGITAL, PORT_DAC)
        and port.key not in manifest.available_port_keys
    )
    references = pulse_document_port_references(document, unavailable)
    if references:
        raise ValueError(
            "backend does not expose ports used by this pulse: "
            + "; ".join(references)
        )
    return document


def replace_pulse_document_target(
    document: PulseDocument,
    manifest: PulseTargetManifest,
    *,
    cascade: bool = False,
) -> PulseTargetEditResult:
    """Atomically remap one document by stable port key to an offline target."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(manifest, PulseTargetManifest):
        raise TypeError("manifest must be PulseTargetManifest")
    if set(manifest.available_port_keys) != {
        port.key
        for port in manifest.target.ports
        if port.kind in (PORT_DIGITAL, PORT_DAC)
    }:
        raise ValueError("an offline authoring manifest must expose every programmable port")

    new_target = manifest.target
    old_by_key = document.target.by_key
    new_by_key = new_target.by_key
    incompatible = frozenset(
        key
        for key, old in old_by_key.items()
        if old.kind != PORT_CLOCK and not _compatible_port(old, new_by_key.get(key))
    )
    impact = PulseTargetEditImpact(
        pulse_document_port_references(document, incompatible)
    )
    if impact.destructive and not cascade:
        raise DestructivePulseTargetEditError(impact)

    old_lane_index = {
        lane: index for index, lane in enumerate(document.target.raw_lanes)
    }
    new_lane_index = {lane: index for index, lane in enumerate(new_target.raw_lanes)}
    periods = []
    for period in document.periods:
        states = [0] * len(new_target.raw_lanes)
        for new_port in new_target.ports:
            old_port = old_by_key.get(new_port.key)
            if not _compatible_port(old_port, new_port) or new_port.kind != PORT_DIGITAL:
                continue
            states[new_lane_index[new_port.lanes[0]]] = period.states[
                old_lane_index[old_port.lanes[0]]
            ]
        periods.append(
            replace(
                period,
                states=tuple(states),
                analog_steps=tuple(
                    step
                    for step in period.analog_steps
                    if step.port not in incompatible
                    and _compatible_port(old_by_key.get(step.port), new_by_key.get(step.port))
                ),
            )
        )

    scan_parameters = tuple(
        parameter
        for parameter in document.scan_parameters
        if parameter.field.kind == FIELD_DURATION
        or parameter.field.port not in incompatible
    )
    api_parameters = tuple(
        parameter
        for parameter in document.api_parameters
        if parameter.field.kind == FIELD_DURATION
        or parameter.field.port not in incompatible
    )
    removed_scan = {
        parameter.parameter_id
        for parameter in document.scan_parameters
        if parameter not in scan_parameters
    }
    scan_table = document.scan_table
    scan_recipe = document.scan_recipe
    if scan_table is not None and removed_scan:
        if scan_parameters:
            kept = tuple(
                index
                for index, column in enumerate(scan_table.columns)
                if column not in removed_scan
            )
            scan_table = FrozenScanTable(
                tuple(scan_table.columns[index] for index in kept),
                tuple(tuple(row[index] for index in kept) for row in scan_table.rows),
            )
        else:
            scan_table = None
        scan_recipe = None

    previous_visible = set(document.visible_ports)
    old_programmable = {
        port.key
        for port in document.target.ports
        if port.kind in (PORT_DIGITAL, PORT_DAC)
    }
    visible = tuple(
        key
        for key in manifest.available_port_keys
        if key in previous_visible or key not in old_programmable
    )
    rebuilt = replace(
        document,
        target=new_target,
        periods=tuple(periods),
        scan_parameters=scan_parameters,
        scan_table=scan_table,
        scan_recipe=scan_recipe,
        api_parameters=api_parameters,
        visible_ports=visible,
        delays=tuple(
            delay for delay in document.delays if delay.port not in incompatible
        ),
    )
    return PulseTargetEditResult(rebuilt, manifest, impact)


__all__ = [
    "DestructivePulseTargetEditError",
    "PulseTargetEditImpact",
    "PulseTargetEditResult",
    "PulseTargetPortDraft",
    "build_pulse_target_manifest",
    "pulse_document_port_references",
    "pulse_target_port_drafts",
    "restrict_pulse_document_to_manifest",
    "replace_pulse_document_target",
]
