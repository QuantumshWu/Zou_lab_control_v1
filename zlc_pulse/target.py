"""Immutable pulse-output topology owned by the pulse bounded context."""

from __future__ import annotations

from dataclasses import dataclass
import json
from importlib.resources import files
from pathlib import Path
import re

from zlc_storage import canonical_digest


PULSE_TARGET_SCHEMA = "zlc_pulse.PulseTarget/v1"
PORT_DIGITAL = "digital"
PORT_DAC = "dac"
PORT_CLOCK = "clock"
PORT_KINDS = (PORT_DIGITAL, PORT_DAC, PORT_CLOCK)
DAC_OFFSET_BINARY = "offset_binary"
_BUS_LABEL = re.compile(r"(?P<base>.+)\[(?P<bit>\d+)\]")
_DAC_CLOCK = re.compile(r"da_clk(?P<index>\d+)", re.IGNORECASE)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _integer(value: object, field: str, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        suffix = " or None" if optional else ""
        raise TypeError(f"{field} must be an integer{suffix}")
    return value


@dataclass(frozen=True)
class PulsePortSpec:
    key: str
    kind: str
    lanes: tuple[str, ...]
    label: str
    bus_index: int | None
    width: int
    encoding: str
    safe_value: int
    latch_clock: str | None

    def __post_init__(self) -> None:
        key = _text(self.key, "pulse port key")
        kind = _text(self.kind, "pulse port kind")
        if kind not in PORT_KINDS:
            raise ValueError(f"pulse port kind must be one of {PORT_KINDS}")
        lanes = tuple(_text(lane, "raw lane") for lane in self.lanes)
        if not lanes or len(lanes) != len(set(lanes)):
            raise ValueError("pulse port lanes must be unique and non-empty")
        width = _integer(self.width, "pulse port width")
        assert width is not None
        if width != len(lanes):
            raise ValueError("pulse port width differs from lane cardinality")
        bus_index = _integer(self.bus_index, "bus_index", optional=True)
        safe = _integer(self.safe_value, "safe_value")
        assert safe is not None
        if kind in (PORT_DIGITAL, PORT_CLOCK):
            if width != 1 or bus_index is not None:
                raise ValueError("digital/clock ports require width one and no bus_index")
            if self.encoding != "binary" or safe != 0:
                raise ValueError(
                    "digital/clock ports require the frozen engine's low safe state"
                )
            if self.latch_clock is not None:
                raise ValueError("digital/clock ports cannot reference a latch clock")
        else:
            if width < 2 or bus_index is None or bus_index < 0:
                raise ValueError("DAC ports require width >= 2 and non-negative bus_index")
            if self.encoding != DAC_OFFSET_BINARY:
                raise ValueError("only offset-binary DAC encoding is supported")
            if safe != 1 << (width - 1):
                raise ValueError(
                    "DAC safe value must be the frozen engine's offset-binary midpoint"
                )
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "lanes", lanes)
        object.__setattr__(self, "label", _text(self.label, "pulse port label"))
        object.__setattr__(self, "bus_index", bus_index)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "safe_value", safe)
        if self.latch_clock is not None:
            object.__setattr__(
                self,
                "latch_clock",
                _text(self.latch_clock, "latch_clock"),
            )

    @property
    def signed_range(self) -> tuple[int, int] | None:
        if self.kind != PORT_DAC:
            return None
        half = 1 << (self.width - 1)
        return (-half, half - 1)


@dataclass(frozen=True)
class PulseTarget:
    raw_lanes: tuple[str, ...]
    ports: tuple[PulsePortSpec, ...]

    def __post_init__(self) -> None:
        raw = tuple(_text(lane, "target raw lane") for lane in self.raw_lanes)
        ports = tuple(self.ports)
        if not raw or len(raw) != len(set(raw)):
            raise ValueError("PulseTarget raw lanes must be unique and non-empty")
        if any(not isinstance(port, PulsePortSpec) for port in ports):
            raise TypeError("PulseTarget ports must contain PulsePortSpec values")
        keys = tuple(port.key for port in ports)
        if len(keys) != len(set(keys)):
            raise ValueError("PulseTarget port keys must be unique")
        owner: dict[str, str] = {}
        for port in ports:
            for lane in port.lanes:
                if lane not in raw:
                    raise ValueError(f"port {port.key!r} owns unknown lane {lane!r}")
                if lane in owner:
                    raise ValueError(
                        f"lane {lane!r} belongs to both {owner[lane]!r} and {port.key!r}"
                    )
                owner[lane] = port.key
        missing = tuple(lane for lane in raw if lane not in owner)
        if missing:
            raise ValueError(f"raw lanes have no logical owner: {missing}")
        by_key = {port.key: port for port in ports}
        clocks = {port.key for port in ports if port.kind == PORT_CLOCK}
        for port in ports:
            if port.latch_clock is not None and port.latch_clock not in clocks:
                raise ValueError(
                    f"DAC port {port.key!r} references missing clock {port.latch_clock!r}"
                )
            if port.latch_clock is not None and by_key[port.latch_clock].kind != PORT_CLOCK:
                raise ValueError("DAC latch_clock must name a clock port")
        bus_indices = sorted(
            port.bus_index for port in ports if port.kind == PORT_DAC
        )
        if bus_indices != list(range(len(bus_indices))):
            raise ValueError("DAC bus indices must be contiguous from zero")
        object.__setattr__(self, "raw_lanes", raw)
        object.__setattr__(self, "ports", ports)

    @property
    def by_key(self) -> dict[str, PulsePortSpec]:
        return {port.key: port for port in self.ports}

    @property
    def abi_fingerprint(self) -> str:
        # Labels are presentation-only and intentionally excluded from the ABI.
        return canonical_digest(
            {
                "schema": "zlc_pulse.PulseTargetABI/v1",
                "raw_lanes": list(self.raw_lanes),
                "ports": [
                    {
                        key: value
                        for key, value in pulse_port_to_tree(port).items()
                        if key != "label"
                    }
                    for port in self.ports
                ],
            }
        )


def pulse_port_to_tree(port: PulsePortSpec) -> dict[str, object]:
    if not isinstance(port, PulsePortSpec):
        raise TypeError("port must be PulsePortSpec")
    return {
        "key": port.key,
        "kind": port.kind,
        "lanes": list(port.lanes),
        "label": port.label,
        "bus_index": port.bus_index,
        "width": port.width,
        "encoding": port.encoding,
        "safe_value": port.safe_value,
        "latch_clock": port.latch_clock,
    }


def pulse_port_from_tree(tree: object) -> PulsePortSpec:
    fields = {
        "key",
        "kind",
        "lanes",
        "label",
        "bus_index",
        "width",
        "encoding",
        "safe_value",
        "latch_clock",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("pulse port has an unknown field set")
    lanes = tree["lanes"]
    if not isinstance(lanes, list):
        raise TypeError("pulse port lanes must be a list")
    return PulsePortSpec(
        tree["key"],
        tree["kind"],
        tuple(lanes),
        tree["label"],
        tree["bus_index"],
        tree["width"],
        tree["encoding"],
        tree["safe_value"],
        tree["latch_clock"],
    )


def pulse_target_to_tree(target: PulseTarget) -> dict[str, object]:
    if not isinstance(target, PulseTarget):
        raise TypeError("target must be PulseTarget")
    return {
        "schema": PULSE_TARGET_SCHEMA,
        "raw_lanes": list(target.raw_lanes),
        "ports": [pulse_port_to_tree(port) for port in target.ports],
        "abi_fingerprint": target.abi_fingerprint,
    }


def pulse_target_from_tree(tree: object) -> PulseTarget:
    if not isinstance(tree, dict) or set(tree) != {
        "schema",
        "raw_lanes",
        "ports",
        "abi_fingerprint",
    }:
        raise ValueError("PulseTarget has an unknown field set")
    if tree["schema"] != PULSE_TARGET_SCHEMA:
        raise ValueError("PulseTarget schema differs")
    raw = tree["raw_lanes"]
    ports = tree["ports"]
    if not isinstance(raw, list) or not isinstance(ports, list):
        raise TypeError("PulseTarget raw_lanes and ports must be lists")
    target = PulseTarget(
        tuple(raw),
        tuple(pulse_port_from_tree(port) for port in ports),
    )
    if tree["abi_fingerprint"] != target.abi_fingerprint:
        raise ValueError("PulseTarget ABI fingerprint differs")
    return target


def load_pulse_target(path: str | Path) -> PulseTarget:
    return pulse_target_from_tree(json.loads(Path(path).read_text(encoding="utf-8")))


def load_deployed_pulse_target() -> PulseTarget:
    """Load the packaged installation target shared by real and virtual composition."""

    resource = files("zlc_pulse").joinpath("assets", "deployed_target.json")
    return pulse_target_from_tree(json.loads(resource.read_text(encoding="utf-8")))


def save_pulse_target(target: PulseTarget, path: str | Path) -> Path:
    if not isinstance(target, PulseTarget):
        raise TypeError("target must be PulseTarget")
    destination = Path(path)
    if destination.suffix.lower() != ".json":
        destination = destination.with_suffix(".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            pulse_target_to_tree(target),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def pulse_target_from_legacy_tree(tree: object) -> PulseTarget:
    """Translate the one historical PortCatalog schema at a file-load boundary."""

    if not isinstance(tree, dict) or set(tree) != {
        "schema",
        "version",
        "raw_lanes",
        "ports",
        "fingerprint",
    }:
        raise ValueError("legacy PortCatalog has an unknown field set")
    if tree["schema"] != "Zou_lab_control.neutral_atom.PortCatalog" or tree["version"] != 1:
        raise ValueError("unsupported legacy PortCatalog schema/version")
    raw = tree["raw_lanes"]
    ports = tree["ports"]
    if not isinstance(raw, list) or not isinstance(ports, list):
        raise TypeError("legacy PortCatalog arrays are invalid")
    # The old 16-character fingerprint was not a file-load gate; structural
    # reconstruction remains the authority and the live target is checked later.
    return PulseTarget(
        tuple(raw),
        tuple(pulse_port_from_tree(port) for port in ports),
    )


def pulse_target_from_legacy_channels(
    channels: object,
    *,
    channel_labels: object = None,
    analog_buses: object = None,
    clk_channels: object = None,
) -> PulseTarget:
    """Reconstruct v1-v3 topology fields without importing the old owner."""

    if not isinstance(channels, (list, tuple)):
        raise TypeError("legacy channels must be an ordered array")
    raw = tuple(_text(item, "legacy raw lane") for item in channels)
    if not isinstance(channel_labels or {}, dict):
        raise TypeError("legacy channel_labels must be an object")
    if not isinstance(analog_buses or {}, dict):
        raise TypeError("legacy analog_buses must be an object")
    if not isinstance(clk_channels or (), (list, tuple)):
        raise TypeError("legacy clk_channels must be an ordered array")
    labels = {str(key): str(value) for key, value in (channel_labels or {}).items()}
    explicit: dict[str, tuple[str, ...]] = {}
    for key, lanes in (analog_buses or {}).items():
        if not isinstance(lanes, (list, tuple)):
            raise TypeError("legacy analog bus lanes must be an ordered array")
        explicit[str(key)] = tuple(str(lane) for lane in lanes)
    inferred: dict[str, dict[int, str]] = {}
    for lane in raw:
        match = _BUS_LABEL.fullmatch(labels.get(lane, lane).strip())
        if match is not None:
            inferred.setdefault(match.group("base"), {})[int(match.group("bit"))] = lane
    buses: dict[str, tuple[str, ...]] = {}
    for key, bits in inferred.items():
        order = sorted(bits)
        if len(order) >= 2 and order == list(range(order[0], order[-1] + 1)):
            buses[key] = tuple(bits[index] for index in order)
    buses.update(explicit)
    position = {lane: index for index, lane in enumerate(raw)}
    ordered_buses = sorted(
        buses.items(), key=lambda item: min(position[lane] for lane in item[1])
    )
    members = {lane for _key, lanes in ordered_buses for lane in lanes}
    clocks = {str(item) for item in (clk_channels or ())}
    for lane in raw:
        if _DAC_CLOCK.fullmatch(labels.get(lane, lane).strip()):
            clocks.add(lane)
    clock_key_by_index: dict[int, str] = {}
    ports: list[PulsePortSpec] = []
    for lane in raw:
        if lane not in clocks:
            continue
        label = labels.get(lane, lane)
        match = _DAC_CLOCK.fullmatch(label)
        if match is not None:
            clock_key_by_index[int(match.group("index"))] = label
        ports.append(
            PulsePortSpec(label, PORT_CLOCK, (lane,), label, None, 1, "binary", 0, None)
        )
    for bus_index, (key, lanes) in enumerate(ordered_buses):
        width = len(lanes)
        ports.append(
            PulsePortSpec(
                key,
                PORT_DAC,
                lanes,
                key,
                bus_index,
                width,
                DAC_OFFSET_BINARY,
                1 << (width - 1),
                clock_key_by_index.get(bus_index),
            )
        )
    occupied = members | clocks
    for lane in raw:
        if lane not in occupied:
            ports.append(
                PulsePortSpec(
                    lane,
                    PORT_DIGITAL,
                    (lane,),
                    labels.get(lane, lane),
                    None,
                    1,
                    "binary",
                    0,
                    None,
                )
            )
    ports.sort(key=lambda port: min(position[lane] for lane in port.lanes))
    return PulseTarget(raw, tuple(ports))


__all__ = [
    "DAC_OFFSET_BINARY",
    "PORT_CLOCK",
    "PORT_DAC",
    "PORT_DIGITAL",
    "PORT_KINDS",
    "PULSE_TARGET_SCHEMA",
    "PulsePortSpec",
    "PulseTarget",
    "load_pulse_target",
    "load_deployed_pulse_target",
    "pulse_port_from_tree",
    "pulse_port_to_tree",
    "pulse_target_from_legacy_tree",
    "pulse_target_from_legacy_channels",
    "pulse_target_from_tree",
    "pulse_target_to_tree",
    "save_pulse_target",
]
