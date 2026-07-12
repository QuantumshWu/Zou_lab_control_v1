"""Bind an immutable authoring document to one verified live PulseTarget."""

from __future__ import annotations

from dataclasses import replace

from .document import PulseDocument
from .target import PulsePortSpec, PulseTarget


def bind_pulse_document_target(
    document: PulseDocument,
    target: PulseTarget,
) -> PulseDocument:
    """Re-key declared logical ports by exact physical ownership.

    Raw state vectors and delay lanes are already physical and remain unchanged.
    Only referenced logical keys are rebound.  A referenced port must have one
    live port with the same kind and exact ordered lane tuple; names, labels,
    widths, and singleton shapes are never used to guess ownership.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(target, PulseTarget):
        raise TypeError("target must be PulseTarget")
    if document.target.abi_fingerprint == target.abi_fingerprint:
        return document
    if document.target.raw_lanes != target.raw_lanes:
        raise ValueError("PulseDocument raw lane order differs from live target")

    by_physical: dict[tuple[str, tuple[str, ...]], list[PulsePortSpec]] = {}
    for port in target.ports:
        by_physical.setdefault((port.kind, port.lanes), []).append(port)

    referenced = set(document.visible_ports)
    referenced.update(
        _dac_target_key(slot.target) for slot in document.scan_slots if slot.kind == "dac"
    )
    referenced.update(
        _dac_target_key(slot.target) for slot in document.api_slots if slot.kind == "dac"
    )
    referenced.update(key for key, _steps in document.analog_bus_programs)
    mapping: dict[str, str] = {}
    for key in referenced:
        source = document.target.by_key.get(key)
        if source is None:
            raise ValueError(f"PulseDocument references unknown source port {key!r}")
        candidates = by_physical.get((source.kind, source.lanes), ())
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
        scan_slots=tuple(
            replace(
                slot,
                target=_rebind_slot_target(slot.kind, slot.target, mapping),
            )
            for slot in document.scan_slots
        ),
        api_slots=tuple(
            replace(
                slot,
                target=_rebind_slot_target(slot.kind, slot.target, mapping),
            )
            for slot in document.api_slots
        ),
        analog_bus_programs=tuple(
            (mapping[key], steps)
            for key, steps in document.analog_bus_programs
        ),
    )


def _dac_target_key(target: str) -> str:
    key, separator, period = target.partition("@")
    if not separator or not key or not period:
        raise ValueError("DAC slot target must be port@period")
    return key


def _rebind_slot_target(
    kind: str,
    target: str,
    mapping: dict[str, str],
) -> str:
    if kind != "dac":
        return target
    key = _dac_target_key(target)
    return f"{mapping[key]}@{target.split('@', 1)[1]}"


__all__ = ["bind_pulse_document_target"]
