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

from .ports import PortCatalog

__all__ = ["BoardConfig", "DEFAULT_BOARD_CONFIG", "load_board_config"]

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
