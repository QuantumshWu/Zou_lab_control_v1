"""Bind an immutable authoring document to one verified live PulseTarget."""

from __future__ import annotations

from dataclasses import replace

from .authoring import _map_field_ref_ports
from .document import PulseDocument
from .target import PORT_CLOCK, PulsePortSpec, PulseTarget


def bind_pulse_document_target(
    document: PulseDocument,
    target: PulseTarget,
) -> PulseDocument:
    """Re-key logical ports by exact physical ownership, never by name or shape."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(target, PulseTarget):
        raise TypeError("target must be PulseTarget")
    if document.target.abi_fingerprint == target.abi_fingerprint:
        return document if document.target is target else replace(document, target=target)
    if document.target.raw_lanes != target.raw_lanes:
        raise ValueError("PulseDocument raw lane order differs from live target")

    source_clock_lanes = {
        port.lanes for port in document.target.ports if port.kind == PORT_CLOCK
    }
    live_clock_lanes = {
        port.lanes for port in target.ports if port.kind == PORT_CLOCK
    }
    if source_clock_lanes != live_clock_lanes:
        raise ValueError(
            "live target clock-mux outputs differ from the authored physical topology"
        )

    by_physical: dict[tuple[object, ...], list[PulsePortSpec]] = {}
    for port in target.ports:
        by_physical.setdefault(_physical_signature(target, port), []).append(port)

    referenced = set(document.visible_ports)
    referenced.update(
        parameter.field.port
        for parameter in (*document.scan_parameters, *document.api_parameters)
        if parameter.field.port is not None
    )
    referenced.update(
        step.port for period in document.periods for step in period.analog_steps
    )
    referenced.update(delay.port for delay in document.delays)
    source_owner = {
        lane: port for port in document.target.ports for lane in port.lanes
    }
    referenced.update(
        source_owner[lane].key
        for lane_index, lane in enumerate(document.target.raw_lanes)
        if any(period.states[lane_index] for period in document.periods)
    )
    referenced.update(
        port.key for port in document.target.ports if port.kind == PORT_CLOCK
    )
    mapping: dict[str, str] = {}
    for key in referenced:
        source = document.target.by_key.get(key)
        if source is None:
            raise ValueError(f"PulseDocument references unknown source port {key!r}")
        candidates = by_physical.get(
            _physical_signature(document.target, source),
            (),
        )
        if len(candidates) != 1:
            raise ValueError(
                f"referenced port {key!r} has no unique physically equivalent live owner"
            )
        mapping[key] = candidates[0].key
    rebound_visible = tuple(mapping[key] for key in document.visible_ports)
    if len(set(rebound_visible)) != len(rebound_visible):
        raise ValueError("two visible authoring ports collapse onto one live port")
    return replace(
        document,
        target=target,
        visible_ports=rebound_visible,
        periods=tuple(
            replace(
                period,
                analog_steps=tuple(
                    replace(step, port=mapping[step.port])
                    for step in period.analog_steps
                ),
            )
            for period in document.periods
        ),
        scan_parameters=tuple(
            replace(
                parameter,
                field=_map_field_ref_ports(parameter.field, mapping),
            )
            for parameter in document.scan_parameters
        ),
        api_parameters=tuple(
            replace(
                parameter,
                field=_map_field_ref_ports(parameter.field, mapping),
            )
            for parameter in document.api_parameters
        ),
        delays=tuple(
            replace(delay, port=mapping[delay.port]) for delay in document.delays
        ),
    )


def _physical_signature(
    target: PulseTarget,
    port: PulsePortSpec,
) -> tuple[object, ...]:
    latch_lanes = (
        None
        if port.latch_clock is None
        else target.by_key[port.latch_clock].lanes
    )
    return (
        port.kind,
        port.lanes,
        port.bus_index,
        port.width,
        port.encoding,
        port.safe_value,
        latch_lanes,
    )


__all__ = ["bind_pulse_document_target"]
