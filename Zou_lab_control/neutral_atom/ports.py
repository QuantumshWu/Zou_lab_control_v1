"""Immutable sequencer output topology.

The FPGA exposes *ports* to an experimenter, not an undifferentiated list of
one-bit lanes.  A digital output owns one raw lane, a clock output owns one raw
lane, and a DAC owns an ordered group of raw data lanes plus (optionally) its
latch-clock port.  :class:`PortCatalog` is the one description shared by the
device, compiler, persisted pulse document and remote protocol.

Raw lanes remain an implementation detail needed by the edge-table encoder.
They are deliberately carried separately from the public port identifiers so
``da_bias_x`` can never accidentally reappear as ten editable TTL channels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Mapping, Sequence

from ._serialization import require_array, require_exact_fields


PORT_DIGITAL = "digital"
PORT_DAC = "dac"
PORT_CLOCK = "clock"
PORT_KINDS = (PORT_DIGITAL, PORT_DAC, PORT_CLOCK)

DAC_OFFSET_BINARY = "offset_binary"

_BUS_LABEL_RE = re.compile(r"(?P<base>.+)\[(?P<bit>\d+)\]")
_DAC_CLOCK_RE = re.compile(r"da_clk(?P<index>\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class PortSpec:
    """One logical sequencer output.

    ``lanes`` are raw compiler lane names in least-significant-bit order for a
    DAC.  ``bus_index`` is the physical runtime DAC engine index, never inferred
    from a pulse/template dictionary order.  ``latch_clock`` names a
    :class:`PORT_CLOCK` spec in the same catalog.
    """

    key: str
    kind: str
    lanes: tuple[str, ...]
    label: str = ""
    bus_index: int | None = None
    width: int = 1
    encoding: str = "binary"
    safe_value: int = 0
    latch_clock: str | None = None

    def __post_init__(self) -> None:
        key = str(self.key).strip()
        kind = str(self.kind).strip().lower()
        lanes = tuple(str(item).strip() for item in self.lanes)
        if not key:
            raise ValueError("sequencer port key must not be empty")
        if kind not in PORT_KINDS:
            raise ValueError(f"sequencer port kind must be one of {PORT_KINDS}, got {kind!r}")
        if not lanes or any(not lane for lane in lanes) or len(set(lanes)) != len(lanes):
            raise ValueError(f"sequencer port {key!r} must own unique non-empty raw lanes")
        width = int(self.width)
        if width != len(lanes):
            raise ValueError(
                f"sequencer port {key!r} width={width} does not match its {len(lanes)} raw lanes")
        if kind in (PORT_DIGITAL, PORT_CLOCK) and width != 1:
            raise ValueError(f"{kind} port {key!r} must have width 1")
        if kind == PORT_DAC:
            if width < 2:
                raise ValueError(f"DAC port {key!r} must own at least two raw lanes")
            if self.bus_index is None or int(self.bus_index) < 0:
                raise ValueError(f"DAC port {key!r} needs a non-negative physical bus_index")
            if str(self.encoding) != DAC_OFFSET_BINARY:
                raise ValueError(
                    f"DAC port {key!r} uses unsupported encoding {self.encoding!r}; "
                    f"expected {DAC_OFFSET_BINARY!r}")
            maximum = (1 << width) - 1
            if not 0 <= int(self.safe_value) <= maximum:
                raise ValueError(
                    f"DAC port {key!r} safe_value must fit its {width}-bit wire code")
        elif self.bus_index is not None:
            raise ValueError(f"non-DAC port {key!r} must not carry bus_index")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "lanes", lanes)
        object.__setattr__(self, "label", str(self.label or key))
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "encoding", str(self.encoding))
        object.__setattr__(self, "safe_value", int(self.safe_value))
        object.__setattr__(self, "bus_index", None if self.bus_index is None else int(self.bus_index))
        object.__setattr__(self, "latch_clock", None if self.latch_clock is None else str(self.latch_clock))

    @property
    def signed_range(self) -> tuple[int, int] | None:
        if self.kind != PORT_DAC:
            return None
        half = 1 << (self.width - 1)
        return (-half, half - 1)

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "kind": self.kind,
            "lanes": list(self.lanes),
            "label": self.label,
            "bus_index": self.bus_index,
            "width": self.width,
            "encoding": self.encoding,
            "safe_value": self.safe_value,
            "latch_clock": self.latch_clock,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PortSpec":
        fields = {
            "key", "kind", "lanes", "label", "bus_index", "width",
            "encoding", "safe_value", "latch_clock",
        }
        require_exact_fields(payload, fields, what="sequencer port spec")
        return cls(
            key=str(payload["key"]),
            kind=str(payload["kind"]),
            lanes=tuple(str(v) for v in require_array(
                payload["lanes"], what="sequencer port spec lanes")),
            label=str(payload["label"]),
            bus_index=None if payload["bus_index"] is None else int(payload["bus_index"]),
            width=int(payload["width"]),
            encoding=str(payload["encoding"]),
            safe_value=int(payload["safe_value"]),
            latch_clock=None if payload["latch_clock"] is None else str(payload["latch_clock"]),
        )


@dataclass(frozen=True)
class PortCatalog:
    """The complete immutable logical/physical output contract of one sequencer."""

    raw_lanes: tuple[str, ...]
    ports: tuple[PortSpec, ...]

    schema = "Zou_lab_control.neutral_atom.PortCatalog"

    def __post_init__(self) -> None:
        raw = tuple(str(item).strip() for item in self.raw_lanes)
        ports = tuple(self.ports)
        if not raw or any(not item for item in raw) or len(set(raw)) != len(raw):
            raise ValueError("PortCatalog.raw_lanes must be unique and non-empty")
        keys = [port.key for port in ports]
        if len(set(keys)) != len(keys):
            raise ValueError("PortCatalog port keys must be unique")
        owner: dict[str, str] = {}
        for port in ports:
            for lane in port.lanes:
                if lane not in raw:
                    raise ValueError(f"port {port.key!r} owns unknown raw lane {lane!r}")
                if lane in owner:
                    raise ValueError(
                        f"raw lane {lane!r} belongs to both {owner[lane]!r} and {port.key!r}")
                owner[lane] = port.key
        missing = [lane for lane in raw if lane not in owner]
        if missing:
            raise ValueError(f"PortCatalog has raw lanes with no logical owner: {missing}")
        clocks = {port.key for port in ports if port.kind == PORT_CLOCK}
        for port in ports:
            if port.latch_clock is not None and port.latch_clock not in clocks:
                raise ValueError(
                    f"DAC port {port.key!r} references missing clock port {port.latch_clock!r}")
        bus_indices = sorted(port.bus_index for port in ports if port.kind == PORT_DAC)
        if bus_indices != list(range(len(bus_indices))):
            raise ValueError(
                f"DAC bus indices must be unique and contiguous from zero, got {bus_indices}")
        object.__setattr__(self, "raw_lanes", raw)
        object.__setattr__(self, "ports", ports)

    @property
    def fingerprint(self) -> str:
        """The host↔bitstream ABI hash: the PHYSICAL output contract only (raw lanes + each port's
        wire identity), NEVER the human ``label``.  A channel rename is a display edit, so it must
        not change this fingerprint (or renaming a channel would falsely read as a hardware mismatch
        at connect / block loading a pulse) -- labels live outside the ABI on purpose."""
        abi = {
            "raw_lanes": list(self.raw_lanes),
            "ports": [
                {k: v for k, v in port.to_dict().items() if k != "label"}
                for port in self.ports
            ],
        }
        canonical = json.dumps(abi, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def with_label(self, key: str, label: str) -> "PortCatalog":
        """A COPY of this catalog with port ``key``'s display ``label`` changed -- the ONE way a
        channel is renamed.  The ABI (lanes / kind / bus_index / fingerprint) is untouched, so a
        rename never invalidates a saved pulse or a hardware match; only the human name moves."""
        key = str(key)
        if key not in self.by_key:
            raise KeyError(f"no sequencer port {key!r} to relabel")
        label = str(label).strip() or key
        ports = tuple(
            PortSpec(port.key, port.kind, port.lanes, label=(label if port.key == key else port.label),
                     bus_index=port.bus_index, width=port.width, encoding=port.encoding,
                     safe_value=port.safe_value, latch_clock=port.latch_clock)
            for port in self.ports
        )
        return PortCatalog(raw_lanes=self.raw_lanes, ports=ports)

    @property
    def by_key(self) -> dict[str, PortSpec]:
        return {port.key: port for port in self.ports}

    @property
    def dac_ports(self) -> tuple[PortSpec, ...]:
        return tuple(sorted(
            (port for port in self.ports if port.kind == PORT_DAC),
            key=lambda port: int(port.bus_index),
        ))

    @property
    def clock_ports(self) -> tuple[PortSpec, ...]:
        return tuple(port for port in self.ports if port.kind == PORT_CLOCK)

    @property
    def digital_ports(self) -> tuple[PortSpec, ...]:
        return tuple(port for port in self.ports if port.kind == PORT_DIGITAL)

    @property
    def analog_buses(self) -> dict[str, list[str]]:
        return {port.key: list(port.lanes) for port in self.dac_ports}

    @property
    def channel_labels(self) -> dict[str, str]:
        labels: dict[str, str] = {}
        for port in self.ports:
            if port.kind == PORT_DAC:
                labels.update({lane: f"{port.key}[{bit}]" for bit, lane in enumerate(port.lanes)})
            else:
                labels[port.lanes[0]] = port.label
        return labels

    @property
    def latch_clock_lanes(self) -> tuple[str, ...]:
        clocks = self.by_key
        return tuple(
            clocks[port.latch_clock].lanes[0]
            for port in self.dac_ports
            if port.latch_clock is not None
        )

    def port_for_lane(self, lane: str) -> PortSpec:
        lane = str(lane)
        for port in self.ports:
            if lane in port.lanes:
                return port
        raise KeyError(lane)

    def matching_port(self, source: PortSpec) -> PortSpec:
        """Resolve one template port against this hardware catalog.

        Hardware identity wins by exact logical key.  A reusable template may
        instead carry the same unique semantic label on a different raw lane;
        that is the only fallback.  Kind changes and ambiguous labels are
        rejected rather than guessed.
        """

        exact = self.by_key.get(source.key)
        if exact is not None:
            if exact.kind != source.kind:
                raise ValueError(
                    f"port {source.key!r} is {source.kind} in the template but "
                    f"{exact.kind} in hardware.")
            return exact
        candidates = [
            port for port in self.ports
            if port.kind == source.kind and port.label == source.label
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(
                f"template port {source.key!r} ({source.kind}, label={source.label!r}) "
                "has no matching hardware port.")
        raise ValueError(
            f"template port label {source.label!r} is ambiguous in hardware: "
            f"{[port.key for port in candidates]}.")

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": self.schema,
            "raw_lanes": list(self.raw_lanes),
            "ports": [port.to_dict() for port in self.ports],
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PortCatalog":
        fields = {"schema", "raw_lanes", "ports", "fingerprint"}
        require_exact_fields(payload, fields, what="sequencer port catalog")
        if payload["schema"] != cls.schema:
            raise ValueError(
                f"unsupported sequencer port-catalog schema {payload['schema']!r}; "
                f"expected {cls.schema!r}")
        raw_lanes = require_array(payload["raw_lanes"], what="PortCatalog.raw_lanes")
        ports = require_array(payload["ports"], what="PortCatalog.ports")
        # Reconstruct and structurally validate in ``__post_init__``.  The stored
        # ``fingerprint`` is NOT a load-time gate: it is the host↔bitstream ABI hash, checked at
        # connect / prepare against the LIVE device catalog (sequencer.py).  Gating deserialization on
        # it made merely OPENING a pulse for editing fail whenever the file predated a topology/label
        # change -- an editor must load any well-formed document; hardware compatibility is a connect
        # concern, not a file-open one.
        return cls(
            raw_lanes=tuple(str(v) for v in raw_lanes),
            ports=tuple(PortSpec.from_dict(v) for v in ports),
        )

    @classmethod
    def from_channels(
        cls,
        channels: Sequence[str],
        *,
        channel_labels: Mapping[str, str] | None = None,
        analog_buses: Mapping[str, Sequence[str]] | None = None,
        clk_channels: Sequence[str] | None = None,
    ) -> "PortCatalog":
        """Build the catalog once at a hardware/XDC boundary.

        Label groups such as ``da_bias_x[0..9]`` become a DAC, and explicit
        ``analog_buses`` take precedence.  Pulse documents, compilers and GUIs
        consume the resulting catalog and never repeat this inference.
        """

        raw = tuple(str(item) for item in channels)
        labels = {str(k): str(v) for k, v in dict(channel_labels or {}).items() if str(k) in raw}
        explicit = {
            str(key): tuple(str(lane) for lane in lanes)
            for key, lanes in dict(analog_buses or {}).items()
        }
        short = [key for key, lanes in explicit.items() if len(lanes) < 2]
        if short:
            raise ValueError(f"DAC ports must own at least two raw lanes: {short}")
        inferred: dict[str, dict[int, str]] = {}
        for lane in raw:
            match = _BUS_LABEL_RE.fullmatch(labels.get(lane, lane).strip())
            if match:
                inferred.setdefault(match.group("base"), {})[int(match.group("bit"))] = lane
        buses: dict[str, tuple[str, ...]] = {}
        for key, bits in inferred.items():
            order = sorted(bits)
            if len(order) >= 2 and order == list(range(order[0], order[-1] + 1)):
                buses[key] = tuple(bits[bit] for bit in order)
        buses.update(explicit)
        lane_pos = {lane: i for i, lane in enumerate(raw)}
        ordered_buses = sorted(buses.items(), key=lambda item: min(lane_pos[lane] for lane in item[1]))
        bus_members = {lane for _key, lanes in ordered_buses for lane in lanes}

        clock_lanes = {str(item) for item in (clk_channels or ())}
        unknown_clocks = sorted(clock_lanes - set(raw))
        if unknown_clocks:
            raise ValueError(f"clock lanes are not in raw_lanes: {unknown_clocks}")
        # XDC-labelled da_clkN ports are physical DAC latch clocks and therefore
        # topology, not a pulse-document toggle.
        for lane in raw:
            if _DAC_CLOCK_RE.fullmatch(labels.get(lane, lane).strip()):
                clock_lanes.add(lane)

        clock_bus_overlap = sorted(clock_lanes & bus_members)
        if clock_bus_overlap:
            raise ValueError(
                f"clock lanes cannot belong to DAC ports: {clock_bus_overlap}")

        clock_key_by_index: dict[int, str] = {}
        specs: list[PortSpec] = []
        for lane in raw:
            if lane not in clock_lanes:
                continue
            label = labels.get(lane, lane)
            key = label
            match = _DAC_CLOCK_RE.fullmatch(label)
            if match:
                clock_key_by_index[int(match.group("index"))] = key
            specs.append(PortSpec(key, PORT_CLOCK, (lane,), label=label))

        for bus_index, (key, lanes) in enumerate(ordered_buses):
            width = len(lanes)
            specs.append(PortSpec(
                key=key,
                kind=PORT_DAC,
                lanes=lanes,
                label=key,
                bus_index=bus_index,
                width=width,
                encoding=DAC_OFFSET_BINARY,
                safe_value=1 << (width - 1),
                latch_clock=clock_key_by_index.get(bus_index),
            ))

        occupied = bus_members | clock_lanes
        for lane in raw:
            if lane in occupied:
                continue
            specs.append(PortSpec(lane, PORT_DIGITAL, (lane,), label=labels.get(lane, lane)))

        # Keep the serialized port list stable in physical order while DAC
        # ``bus_index`` remains explicit.
        specs.sort(key=lambda port: min(lane_pos[lane] for lane in port.lanes))
        return cls(raw_lanes=raw, ports=tuple(specs))


def coerce_port_catalog(
    value: PortCatalog | Mapping[str, object] | None,
    *,
    channels: Sequence[str] | None = None,
    channel_labels: Mapping[str, str] | None = None,
    analog_buses: Mapping[str, Sequence[str]] | None = None,
    clk_channels: Sequence[str] | None = None,
) -> PortCatalog:
    if isinstance(value, PortCatalog):
        return value
    if isinstance(value, Mapping):
        return PortCatalog.from_dict(value)
    if channels is None:
        raise ValueError("a sequencer needs port_catalog or raw channels")
    return PortCatalog.from_channels(
        channels,
        channel_labels=channel_labels,
        analog_buses=analog_buses,
        clk_channels=clk_channels,
    )


__all__ = [
    "DAC_OFFSET_BINARY",
    "PORT_CLOCK",
    "PORT_DAC",
    "PORT_DIGITAL",
    "PORT_KINDS",
    "PortCatalog",
    "PortSpec",
    "coerce_port_catalog",
]
