"""Typed, console-local bindings for the finite occupancy processor."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_neutral_atom.catalog import DefinitionKey
from zlc_storage import canonical_text

__all__ = ["ConsoleProducerBinding", "OccupancyBindingIntent"]


@dataclass(frozen=True, slots=True)
class OccupancyBindingIntent:
    """Exact producer/output keys selected in the processor form."""

    camera_frame_signal: str
    calibration_signal: str

    def __post_init__(self) -> None:
        canonical_text(self.camera_frame_signal, "camera_frame_signal")
        canonical_text(self.calibration_signal, "calibration_signal")
        if self.camera_frame_signal == self.calibration_signal:
            raise ValueError(
                "occupancy source and calibration must be distinct console outputs"
            )


@dataclass(frozen=True, slots=True)
class ConsoleProducerBinding:
    """One exact output resolved against one row in this TaskConsole."""

    signal_key: str
    producer_label: str
    definition_key: DefinitionKey
    output_name: str
    request: object
    run_node: object | None
    final_result_resolved: bool
    final_result: object | None

    def __post_init__(self) -> None:
        canonical_text(self.signal_key, "signal_key")
        canonical_text(self.producer_label, "producer_label")
        if not isinstance(self.definition_key, DefinitionKey):
            raise TypeError("definition_key must be DefinitionKey")
        canonical_text(self.output_name, "output_name")
        if not isinstance(self.final_result_resolved, bool):
            raise TypeError("final_result_resolved must be bool")
        if not self.final_result_resolved and self.final_result is not None:
            raise ValueError(
                "an unresolved producer cannot expose a FINAL result"
            )

    @property
    def running(self) -> bool:
        return bool(
            self.run_node is not None
            and getattr(self.run_node, "running", False)
        )
