"""Operator-visible pulse target manifests.

``PulseTarget`` remains the executable ABI.  A manifest adds the physical or
simulated endpoint names an operator needs and, by the ports it contains,
declares the subset that one backend actually exposes.  Remote manifests are
built from the server-side XDC; virtual manifests are built from simulator
wiring.  The GUI never guesses either source from lane rank or a client XDC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Iterable, Mapping

from zlc_storage import canonical_digest, exact_mapping

from .target import (
    PORT_CLOCK,
    PORT_DAC,
    PulseTarget,
    pulse_target_from_tree,
    pulse_target_to_tree,
)


PULSE_TARGET_MANIFEST_SCHEMA = "zlc_pulse.PulseTargetManifest"


@dataclass(frozen=True, slots=True)
class PulsePortManifest:
    """One exposed logical port and its lane-aligned endpoint names."""

    port_key: str
    endpoints: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.port_key, str) or not self.port_key:
            raise ValueError("manifest port_key must be non-empty text")
        if self.port_key != self.port_key.strip():
            raise ValueError("manifest port_key must be canonical text")
        endpoints = tuple(self.endpoints)
        if not endpoints or any(
            not isinstance(value, str)
            or not value
            or value != value.strip()
            for value in endpoints
        ):
            raise ValueError("manifest endpoints must be non-empty canonical text")
        object.__setattr__(self, "endpoints", endpoints)


@dataclass(frozen=True, slots=True)
class PulseTargetManifest:
    """One target plus the exact logical ports a backend exposes to operators."""

    target: PulseTarget
    ports: tuple[PulsePortManifest, ...]
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.target, PulseTarget):
            raise TypeError("manifest target must be PulseTarget")
        ports = tuple(self.ports)
        if not ports or any(not isinstance(value, PulsePortManifest) for value in ports):
            raise ValueError("manifest ports must contain PulsePortManifest values")
        keys = tuple(value.port_key for value in ports)
        if len(keys) != len(set(keys)):
            raise ValueError("manifest port keys must be unique")
        order = {port.key: index for index, port in enumerate(self.target.ports)}
        if any(key not in order for key in keys):
            unknown = tuple(key for key in keys if key not in order)
            raise ValueError(f"manifest exposes unknown target ports: {unknown}")
        if tuple(sorted(keys, key=order.__getitem__)) != keys:
            raise ValueError("manifest ports must follow PulseTarget physical order")
        for value in ports:
            target_port = self.target.by_key[value.port_key]
            if len(value.endpoints) != target_port.width:
                raise ValueError(
                    f"manifest endpoints for {value.port_key!r} do not match port width"
                )
            if (
                target_port.kind == PORT_DAC
                and target_port.latch_clock not in keys
            ):
                raise ValueError(
                    f"manifest DAC {value.port_key!r} must expose its latch clock"
                )
        object.__setattr__(self, "ports", ports)
        object.__setattr__(
            self,
            "_fingerprint",
            canonical_digest(pulse_target_manifest_to_tree(self)),
        )

    @property
    def by_key(self) -> dict[str, PulsePortManifest]:
        return {value.port_key: value for value in self.ports}

    @property
    def available_port_keys(self) -> tuple[str, ...]:
        return tuple(
            value.port_key
            for value in self.ports
            if self.target.by_key[value.port_key].kind != PORT_CLOCK
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def with_target(self, target: PulseTarget) -> "PulseTargetManifest":
        """Carry endpoint facts across a label-only target edit."""

        if not isinstance(target, PulseTarget):
            raise TypeError("target must be PulseTarget")
        if target.abi_fingerprint != self.target.abi_fingerprint:
            raise ValueError("manifest endpoint facts cannot cross a target ABI change")
        return self if target == self.target else PulseTargetManifest(target, self.ports)


def pulse_port_manifest_to_tree(value: PulsePortManifest) -> dict[str, object]:
    if not isinstance(value, PulsePortManifest):
        raise TypeError("value must be PulsePortManifest")
    return {"port_key": value.port_key, "endpoints": list(value.endpoints)}


def pulse_port_manifest_from_tree(tree: object) -> PulsePortManifest:
    value = exact_mapping(
        tree,
        {"port_key", "endpoints"},
        "pulse port manifest",
        discriminator=None,
    )
    endpoints = value["endpoints"]
    if not isinstance(endpoints, list):
        raise TypeError("pulse port manifest endpoints must be a list")
    return PulsePortManifest(value["port_key"], tuple(endpoints))


def pulse_target_manifest_to_tree(value: PulseTargetManifest) -> dict[str, object]:
    if not isinstance(value, PulseTargetManifest):
        raise TypeError("value must be PulseTargetManifest")
    return {
        "schema": PULSE_TARGET_MANIFEST_SCHEMA,
        "target": pulse_target_to_tree(value.target),
        "ports": [pulse_port_manifest_to_tree(port) for port in value.ports],
    }


def pulse_target_manifest_from_tree(tree: object) -> PulseTargetManifest:
    value = exact_mapping(
        tree,
        {"schema", "target", "ports"},
        PULSE_TARGET_MANIFEST_SCHEMA,
    )
    ports = value["ports"]
    if not isinstance(ports, list):
        raise TypeError("PulseTargetManifest ports must be a list")
    return PulseTargetManifest(
        pulse_target_from_tree(value["target"]),
        tuple(pulse_port_manifest_from_tree(port) for port in ports),
    )


def pulse_target_manifest(
    target: PulseTarget,
    endpoints_by_port: Mapping[str, Iterable[str]],
) -> PulseTargetManifest:
    """Build a manifest in target order from one explicit backend mapping."""

    if not isinstance(target, PulseTarget):
        raise TypeError("target must be PulseTarget")
    requested = {}
    for key, values in endpoints_by_port.items():
        if not isinstance(key, str):
            raise TypeError("manifest mapping keys must be text")
        requested[key] = tuple(values)
    unknown = tuple(key for key in requested if key not in target.by_key)
    if unknown:
        raise ValueError(f"manifest mapping names unknown target ports: {unknown}")
    return PulseTargetManifest(
        target,
        tuple(
            PulsePortManifest(port.key, requested[port.key])
            for port in target.ports
            if port.key in requested
        ),
    )


def pulse_target_manifest_from_lanes(target: PulseTarget) -> PulseTargetManifest:
    """Describe an unbound/offline target by its own stable lane identities."""

    if not isinstance(target, PulseTarget):
        raise TypeError("target must be PulseTarget")
    return pulse_target_manifest(
        target,
        {port.key: port.lanes for port in target.ports},
    )


_NOT_A_PULSE_LANE = frozenset(
    {
        "clk",
        "reset",
        "start",
        "running",
        "done",
        "uart_rx",
        "uart_tx",
        "led",
        "zlc_running_led",
        "zlc_done_led",
    }
)
_GROUND_PIN = re.compile(r"GND\d*", re.IGNORECASE)
_PIN_LINE = re.compile(
    r"PACKAGE_PIN\s+(?P<pin>\w+).*?\[\s*get_ports\s+"
    r"\{?\s*(?P<signal>[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?)\s*\}?\s*\]",
    re.IGNORECASE,
)


def _is_pulse_signal(signal: str) -> bool:
    base = signal.split("[", 1)[0]
    return base not in _NOT_A_PULSE_LANE and _GROUND_PIN.fullmatch(signal) is None


def read_xdc_pulse_lanes(path: str | Path) -> tuple[tuple[str, str, str], ...]:
    """Return ``(lane, signal, package_pin)`` once, in XDC output order."""

    source = Path(path)
    seen: set[str] = set()
    result: list[tuple[str, str, str]] = []
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _PIN_LINE.search(stripped)
        if match is None:
            continue
        signal = match.group("signal")
        if not _is_pulse_signal(signal) or signal in seen:
            continue
        seen.add(signal)
        result.append((f"ch{len(result):02d}", signal, match.group("pin")))
    if not result:
        raise ValueError(f"{source} declares no pulse output lanes")
    return tuple(result)


def pulse_target_manifest_from_xdc(
    target: PulseTarget,
    path: str | Path,
) -> PulseTargetManifest:
    """Validate one deployed target against its server-side XDC and add pins."""

    if not isinstance(target, PulseTarget):
        raise TypeError("target must be PulseTarget")
    lanes = read_xdc_pulse_lanes(path)
    if tuple(lane for lane, _signal, _pin in lanes) != target.raw_lanes:
        raise ValueError("XDC pulse lane order differs from deployed PulseTarget")
    signals = {lane: signal for lane, signal, _pin in lanes}
    pins = {lane: pin for lane, _signal, pin in lanes}
    endpoints: dict[str, tuple[str, ...]] = {}
    for port in target.ports:
        if port.kind == PORT_DAC:
            expected = tuple(f"{port.key}[{bit}]" for bit in range(port.width))
        else:
            expected = (port.label,)
        actual = tuple(signals[lane] for lane in port.lanes)
        if actual != expected:
            raise ValueError(
                f"XDC signals for {port.key!r} are {actual}, expected {expected}"
            )
        endpoints[port.key] = tuple(pins[lane] for lane in port.lanes)
    return pulse_target_manifest(target, endpoints)


__all__ = [
    "PULSE_TARGET_MANIFEST_SCHEMA",
    "PulsePortManifest",
    "PulseTargetManifest",
    "pulse_port_manifest_from_tree",
    "pulse_port_manifest_to_tree",
    "pulse_target_manifest",
    "pulse_target_manifest_from_tree",
    "pulse_target_manifest_from_lanes",
    "pulse_target_manifest_from_xdc",
    "pulse_target_manifest_to_tree",
    "read_xdc_pulse_lanes",
]
