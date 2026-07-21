"""The board pin map is where a lane stops being ``ch07`` and becomes ``trig``.

A pulse lane has three facts an operator cares about: its position in the FPGA
bit order, the name the lab calls it, and the physical pin it leaves the board
on.  All three live in one file -- the platform's XDC constraints -- so they are
read here ONCE, together, into one value.  Reading that file three times for
three answers is how the three drift apart.

Without the names this layer supplies, a catalog built from bare lanes cannot
group anything: :meth:`PortCatalog.from_channels` infers a DAC bus from LABELS
like ``da_bias_x[0]``, so an unlabelled catalog shows every raw lane separately
and the operator sees ``ch00`` where the board says ``cooling``.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Mapping, Sequence

from .ports import PORT_CLOCK, PORT_DAC, PortCatalog

__all__ = ["BoardConfig", "DEFAULT_BOARD_CONFIG", "load_board_config"]

#: A DAC bus latches on its own clock pin; bus N is latched by ``da_clkN``.
_DAC_LATCH_CLOCK_PREFIX = "da_clk"
#: The DAC's wire encoding: unsigned codes whose MID point is 0 V, so the user-facing
#: level is signed and the safe idle level is the mid code.
_DAC_ENCODING = "offset_binary"
#: A single-lane port carries one bit.
_BINARY_ENCODING = "binary"

#: The in-repo platform copy; ``fpga/board_config/README.md`` documents keeping it in step.
#: Anchored to the repository, not to the working directory -- a notebook opened from the
#: user's data folder must find the same board as the launcher does.
DEFAULT_BOARD_CONFIG = Path(__file__).resolve().parents[2] / "fpga" / "board_config" / "board.xdc"

#: Fixed-function pins that are NOT pulse lanes: the clock input, the reset/status
#: handshake, the UART side channel and the board LEDs.  Everything else a constraints
#: file pins IS a lane, so adding a channel to the board needs no code change here.
_NOT_A_LANE = frozenset({
    "clk", "reset", "start", "running", "done",     # the top module's control handshake
    "uart_rx", "uart_tx",                           # the fast-control side channel
    "led", "zlc_running_led", "zlc_done_led",       # status indicators
})

#: Ground straps carry no signal.  They are pinned like any other port, so only their
#: name distinguishes them, and there are enough of them to shift every lane index.
_GROUND_PIN = re.compile(r"GND\d*", re.IGNORECASE)

#: ``set_property -dict {PACKAGE_PIN F15 IOSTANDARD LVCMOS33} [get_ports cooling]`` and the
#: braced form ``[get_ports {da_bias_x[0]}]`` that a vector element needs.  The name is
#: matched as identifier-then-optional-index rather than "anything but a brace", so that a
#: vector element keeps the ``[0]`` a DAC bus is recognised by.
_PIN_LINE = re.compile(
    r"PACKAGE_PIN\s+(?P<pin>\w+).*?\[\s*get_ports\s+"
    r"\{?\s*(?P<port>[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?)\s*\}?\s*\]",
    re.IGNORECASE,
)


def _lane_base(port_name: str) -> str:
    """``da_bias_x[3]`` -> ``da_bias_x``; a scalar port is its own base."""

    return port_name.split("[", 1)[0]


def _is_pulse_lane(port_name: str) -> bool:
    """A pinned port is a lane unless it is board infrastructure or a ground strap."""

    return (_lane_base(port_name) not in _NOT_A_LANE
            and not _GROUND_PIN.fullmatch(port_name))


@dataclass(frozen=True)
class BoardConfig:
    """One reading of the board pin map: order, names, pins.

    ``lanes`` are the synthetic bit-order keys (``ch00`` ...) the pulse stack addresses
    lanes by; ``labels`` and ``pins`` map each of those to what the board calls it.
    """

    lanes: tuple[str, ...]
    labels: Mapping[str, str]
    pins: Mapping[str, str]

    def pulse_target(self) -> "PulseTarget":
        """The board as the pulse pipeline's target: topology PLUS the hardware facts.

        ``PulsePortSpec`` is a strict superset of the authoring ``PortSpec`` -- it also
        carries ``width`` / ``encoding`` / ``safe_value`` / ``bus_index`` / ``latch_clock``.
        Those are properties of the BOARD, so they are derived here from the same reading
        that names the lanes, rather than being re-stated by whoever happens to build a
        target.  A DAC's lanes are ordered by BIT INDEX, not by the order the constraints
        file happens to pin them: ``da_bias_y`` is pinned MSB-first, and taking file order
        would silently reverse that bus.

        The board is the topology's origin; a saved document carries a copy so it can be
        checked against the board it was authored for (``test_board_config_is_the_one_topology``).
        """

        from zlc_pulse.target import PulsePortSpec, PulseTarget

        catalog = self.port_catalog()
        clock_keys = {port.key for port in catalog.ports if port.kind == PORT_CLOCK}
        bus_index = 0
        specs: list[PulsePortSpec] = []
        for port in catalog.ports:
            if port.kind == PORT_DAC:
                latch_clock = f"{_DAC_LATCH_CLOCK_PREFIX}{bus_index}"
                if latch_clock not in clock_keys:
                    raise ValueError(
                        f"DAC bus {port.key!r} has no latch clock {latch_clock!r} on this board.")
                width = len(port.lanes)
                specs.append(PulsePortSpec(
                    port.key, port.kind, port.lanes, port.label, bus_index, width,
                    _DAC_ENCODING,
                    1 << (width - 1),      # mid code == 0 V, the safe idle level
                    latch_clock))
                bus_index += 1
            else:
                specs.append(PulsePortSpec(
                    port.key, port.kind, port.lanes, port.label, None, 1,
                    _BINARY_ENCODING, 0, None))
        return PulseTarget(catalog.raw_lanes, tuple(specs))

    def port_catalog(self, *, clk_channels: Sequence[str] | None = None) -> PortCatalog:
        """The catalog every consumer downstream reads -- names included.

        The labels are what let ``from_channels`` recognise ``da_bias_x[0..n]`` as one DAC
        bus, so this is also the step that turns N raw lanes into the smaller set of ports
        the operator actually sees.
        """

        return PortCatalog.from_channels(
            self.lanes, channel_labels=dict(self.labels), clk_channels=clk_channels)


def load_board_config(path: str | Path | None = None) -> BoardConfig:
    """Read the pin map.  Lane order is the order the constraints file pins them."""

    source = Path(DEFAULT_BOARD_CONFIG if path is None else path)
    text = source.read_text(encoding="utf-8", errors="replace")
    labels: dict[str, str] = {}
    pins: dict[str, str] = {}
    lanes: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = _PIN_LINE.search(stripped)
        if match is None:
            continue
        port_name = match.group("port")
        if not _is_pulse_lane(port_name) or port_name in seen:
            continue     # a port constrained twice (pin + IO standard) is still one lane
        seen.add(port_name)
        lane = f"ch{len(lanes):02d}"
        lanes.append(lane)
        labels[lane] = port_name
        pins[lane] = match.group("pin")
    if not lanes:
        raise ValueError(f"{source} pins no pulse lanes; is it the right constraints file?")
    return BoardConfig(lanes=tuple(lanes), labels=labels, pins=pins)
