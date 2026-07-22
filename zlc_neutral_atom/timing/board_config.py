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
from pathlib import Path
from typing import Mapping, Sequence

from zlc_pulse.manifest import read_xdc_pulse_lanes

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
    """Project the pulse owner's one XDC reading into the neutral-atom catalog."""

    source = Path(DEFAULT_BOARD_CONFIG if path is None else path)
    bindings = read_xdc_pulse_lanes(source)
    return BoardConfig(
        lanes=tuple(lane for lane, _signal, _pin in bindings),
        labels={lane: signal for lane, signal, _pin in bindings},
        pins={lane: pin for lane, _signal, pin in bindings},
    )
