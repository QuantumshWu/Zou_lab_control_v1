"""Real crossed-AOD rearrangement actuator (FPGA DAC-ramp backend).

This is the hardware counterpart of :class:`devices.virtual.VirtualAOD`.  A 2D crossed AOD positions a
tweezer by its X / Y RF tones; each tone's frequency is set by a DAC voltage -> VCO -> deflection angle.
To DRAG an atom from one site to another we ramp the X and Y DAC buses from the source-site codes to the
destination-site codes over a move window -- the tweezer (and the atom in it) follows adiabatically.  This
reuses the streamer's existing analog-bus RAMP primitive (``PulseTableState.set_analog_bus_mode(...,
"ramp")``) -- no new RTL.

The site -> DAC-code calibration (which voltage code parks the tweezer over each tweezer site) and the
move / settle timing are HARDWARE-SPECIFIC and must be measured on the real machine; they are injected via
config (``x_codes`` / ``y_codes`` per grid column / row, or a linear ``x_code_base`` + ``x_code_step``),
never guessed here.  ``apply_moves`` drags atoms one moving tweezer at a time (the reference
PythonCamDemo ``moveoff_atom`` scheme), firing one ramp program per move.

The orchestration (``operations.rearrangement`` + the rearrange task) calls ``aod.apply_moves(plan.moves)``
identically on this and the virtual AOD -- only the execution differs (real ramps vs simulated occupancy),
which is the virtual==real seam.
"""

from __future__ import annotations

from typing import Sequence

from .base import AODDevice, SequencerDevice
from ..core.analysis import grid_shape_tuple
from ..timing import PulseTableState
from ..timing.pulse_table import PulsePeriod


def _positive(value: float, name: str) -> float:
    out = float(value)
    if not (out > 0.0):
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return out


def _linear_codes(n: int, base: float, step: float) -> list[int]:
    """Per-index DAC codes for a linearly-calibrated axis: ``code(i) = round(base + step * i)``."""
    return [int(round(float(base) + float(step) * i)) for i in range(int(n))]


class RampAOD(AODDevice):
    """Drag atoms between tweezer sites by ramping the X / Y DAC buses (see module docstring).

    ``sequencer``     the pulse streamer that fires the ramp programs.
    ``grid_shape``    ``(rows, cols)`` of the tweezer array (matches the readout grid).
    ``x_bus``/``y_bus`` analog-bus LABELS for the X / Y deflection DACs.
    ``x_channels``/``y_channels`` the BOARD channels (LSB-first) wired to the X / Y DAC -- the real
                      hardware wiring (e.g. ``["ch20".."ch29"]``); default = ``bus_width`` synthetic
                      ``aod_x0..`` names (for the virtual / compile path).
    ``x_codes``/``y_codes`` explicit per-column / per-row signed DAC codes (length cols / rows); if None,
                      built linearly from ``{x,y}_code_base`` + ``{x,y}_code_step``.
    ``move_duration_s``/``settle_s`` ramp window and post-ramp settle per move.
    ``move_survival`` reported per-move transit survival (book-keeping; the real loss is physical).
    """

    def __init__(
        self,
        sequencer: SequencerDevice,
        *,
        grid_shape: Sequence[int] = (5, 7),
        x_bus: str = "aod_x",
        y_bus: str = "aod_y",
        x_channels: Sequence[str] | None = None,
        y_channels: Sequence[str] | None = None,
        x_codes: Sequence[int] | None = None,
        y_codes: Sequence[int] | None = None,
        x_code_base: float = 0.0,
        x_code_step: float = 40.0,
        y_code_base: float = 0.0,
        y_code_step: float = 40.0,
        bus_width: int = 10,
        move_duration_s: float = 100e-6,
        settle_s: float = 20e-6,
        move_survival: float = 0.98,
    ):
        self.sequencer = sequencer
        self.grid_shape = grid_shape_tuple(grid_shape)
        self.x_bus = str(x_bus)
        self.y_bus = str(y_bus)
        self.bus_width = int(bus_width)
        self.x_channels = ([str(c) for c in x_channels] if x_channels is not None
                           else [f"{self.x_bus}{bit}" for bit in range(self.bus_width)])
        self.y_channels = ([str(c) for c in y_channels] if y_channels is not None
                           else [f"{self.y_bus}{bit}" for bit in range(self.bus_width)])
        if len(self.x_channels) != self.bus_width or len(self.y_channels) != self.bus_width:
            raise ValueError(f"x_channels/y_channels must each list {self.bus_width} board channels "
                             f"(the DAC bus width), got {len(self.x_channels)}/{len(self.y_channels)}.")
        ny, nx = self.grid_shape
        self.x_codes = list(x_codes) if x_codes is not None else _linear_codes(nx, x_code_base, x_code_step)
        self.y_codes = list(y_codes) if y_codes is not None else _linear_codes(ny, y_code_base, y_code_step)
        if len(self.x_codes) != nx or len(self.y_codes) != ny:
            raise ValueError(f"x_codes/y_codes must have length {nx}/{ny} for grid {self.grid_shape}.")
        # Validate every site code against the bus's SIGNED range NOW (at config/connect time), not
        # lazily on the first move that touches an out-of-range edge site deep into a run: a too-wide
        # array / too-large step would otherwise pass connect and raise mid-experiment.
        lo, hi = -(1 << (self.bus_width - 1)), (1 << (self.bus_width - 1)) - 1
        for axis, codes in (("x", self.x_codes), ("y", self.y_codes)):
            for code in codes:
                if not (lo <= int(code) <= hi):
                    raise ValueError(f"AOD {axis} site code {code} is outside the signed {self.bus_width}-bit "
                                     f"DAC range [{lo}, {hi}] -- recalibrate the site->code map or bus_width.")
        self.move_duration_s = _positive(move_duration_s, "move_duration_s")
        self.settle_s = _positive(settle_s, "settle_s")
        self.move_survival = float(move_survival)

    # -- bus / site geometry -------------------------------------------------
    def _codes_for(self, site: int) -> tuple[int, int]:
        ny, nx = self.grid_shape
        row, col = int(site) // nx, int(site) % nx
        if not (0 <= row < ny and 0 <= col < nx):
            raise ValueError(f"site {site} out of range for grid {self.grid_shape}.")
        return self.x_codes[col], self.y_codes[row]

    def move_program(self, src: int, dst: int) -> PulseTableState:
        """Build the ramp program that drags an atom from site ``src`` to site ``dst`` -- THREE periods:
        (0) PARK the X / Y buses at the SOURCE codes (an edge, so the ramp's carry-in is the source),
        (1) RAMP from the source codes to the destination codes over ``move_duration_s`` (the chirp that
        drags the atom), (2) SETTLE holding the destination codes.  The park period is essential: a ramp
        carries in the PREVIOUS period's end value, so without a source-park first the ramp would start
        from 0 V (array centre) and the tweezer would not be over the atom -- it would be dropped on every
        move.  Returned (not fired) so it can be inspected / unit-tested."""
        sx, sy = self._codes_for(src)
        dx, dy = self._codes_for(dst)
        members_x, members_y = self.x_channels, self.y_channels
        channels = members_x + members_y
        zero = tuple(0 for _ in channels)
        table = PulseTableState(
            channels=channels,
            analog_buses={self.x_bus: members_x, self.y_bus: members_y},
            periods=[
                PulsePeriod(self.settle_s * 1e9, zero, unit="ns", name="park"),
                PulsePeriod(self.move_duration_s * 1e9, zero, unit="ns", name="ramp"),
                PulsePeriod(self.settle_s * 1e9, zero, unit="ns", name="settle"),
            ],
            repeat_forever=False,
            name=f"aod_move_{src}_{dst}",
        )
        table.set_bus_value(0, self.x_bus, int(sx))           # PARK at the source codes (the ramp carry-in)
        table.set_bus_value(0, self.y_bus, int(sy))
        table.set_bus_value(2, self.x_bus, int(dx))           # SETTLE holds at the destination codes
        table.set_bus_value(2, self.y_bus, int(dy))
        table.set_analog_bus_mode(1, self.x_bus, "ramp", value=int(dx))   # period 1 ramps src -> dst
        table.set_analog_bus_mode(1, self.y_bus, "ramp", value=int(dy))
        return table

    # -- AODDevice contract --------------------------------------------------
    def apply_moves(self, moves, *, survival: float | None = None) -> None:
        for m in moves:
            program = self.move_program(int(m.src), int(m.dst))
            self.sequencer.prepare(program)
            self.sequencer.fire()
        self.sequencer.set_safe_state()

    def snapshot(self) -> dict[str, object]:
        return {"type": type(self).__name__, "grid_shape": self.grid_shape,
                "x_bus": self.x_bus, "y_bus": self.y_bus,
                "move_duration_s": self.move_duration_s, "settle_s": self.settle_s}

    def open(self) -> "RampAOD":
        return self

    def close(self) -> None:
        pass
