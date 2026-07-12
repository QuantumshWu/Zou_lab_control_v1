"""Workbench-owned pulse target and command boundary.

The frontend edits pulse values and renders previews.  It never receives a sequencer,
adapter, or SDK handle; all executable operations cross this generation-bound port.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

from Zou_lab_control.neutral_atom.ports import PortCatalog
from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState


@dataclass(frozen=True, slots=True)
class PulseTargetDescriptor:
    installation_id: str
    installation_generation: int
    port_catalog: PortCatalog
    clock_hz: float
    connection_label: str

    def __post_init__(self) -> None:
        installation_id = str(self.installation_id)
        if not installation_id or installation_id.strip() != installation_id:
            raise ValueError("installation id must be canonical non-empty text")
        object.__setattr__(self, "installation_id", installation_id)
        if (
            isinstance(self.installation_generation, bool)
            or not isinstance(self.installation_generation, int)
        ):
            raise TypeError("installation generation must be int")
        if self.installation_generation < 1:
            raise ValueError("installation generation must be positive")
        if not isinstance(self.port_catalog, PortCatalog):
            raise TypeError("pulse target needs PortCatalog")
        clock_hz = float(self.clock_hz)
        if not math.isfinite(clock_hz) or clock_hz <= 0:
            raise ValueError("pulse target clock must be finite and positive")
        object.__setattr__(self, "clock_hz", clock_hz)
        label = str(self.connection_label)
        if not label:
            raise ValueError("pulse target connection label cannot be empty")
        object.__setattr__(self, "connection_label", label)

    @property
    def time_step_ns(self) -> float:
        return 1e9 / self.clock_hz


class PulseCommandPort:
    """Generation-bound semantic commands backed by the installation authority."""

    __slots__ = ("_facade", "_target", "_generation_reader")

    def __init__(
        self,
        facade: object,
        target: PulseTargetDescriptor,
        generation_reader: Callable[[], int],
    ) -> None:
        if facade is None:
            raise TypeError("pulse command port needs an authority facade")
        if not isinstance(target, PulseTargetDescriptor):
            raise TypeError("pulse command port target must be PulseTargetDescriptor")
        if not callable(generation_reader):
            raise TypeError("generation reader must be callable")
        self._facade = facade
        self._target = target
        self._generation_reader = generation_reader

    @property
    def target(self) -> PulseTargetDescriptor:
        return self._target

    def prepare(self, state: PulseTableState):
        from Zou_lab_control.neutral_atom.devices.sequencer import bind_pulse

        self._check_generation()
        return bind_pulse(self._facade, state).prepare(repeat_forever=True)

    def run(self, state: PulseTableState):
        from Zou_lab_control.neutral_atom.devices.sequencer import bind_pulse

        self._check_generation()
        return bind_pulse(self._facade, state).on_pulse(
            repeat_forever=True, wait=False
        )

    def stop(self) -> None:
        self._check_generation()
        self._facade.set_safe_state()

    def scan_progress(self) -> dict[str, object]:
        self._check_generation()
        return dict(self._facade.scan_progress())

    def snapshot(self) -> dict[str, object]:
        self._check_generation()
        return dict(self._facade.snapshot())

    def _check_generation(self) -> None:
        current = self._generation_reader()
        if isinstance(current, bool) or not isinstance(current, int):
            raise TypeError("installation generation reader returned non-int")
        if current != self._target.installation_generation:
            raise RuntimeError(
                "pulse command belongs to a stale installation generation"
            )


def managed_pulse_command_port(session, runtime, sequencer) -> PulseCommandPort:
    """Composition factory; raw sequencer remains private to the runtime facade."""

    runtime.ensure_connections((sequencer,))
    facade = runtime.pulse_facade(sequencer)
    catalog = session.device_catalog
    port_catalog = getattr(facade, "port_catalog", None)
    if not isinstance(port_catalog, PortCatalog):
        raise RuntimeError("sequencer connection did not publish a PortCatalog")
    clock_hz = getattr(facade, "clock_hz", None)
    if clock_hz is None:
        raise RuntimeError("sequencer connection did not publish its clock")
    target = PulseTargetDescriptor(
        catalog.installation_id,
        catalog.installation_generation,
        port_catalog,
        float(clock_hz),
        "Session installation authority",
    )
    return PulseCommandPort(
        facade,
        target,
        lambda: session.device_catalog.installation_generation,
    )


__all__ = ["PulseCommandPort", "PulseTargetDescriptor"]
