"""Virtual endpoint for the single pulse-clocked RF table Port."""

from __future__ import annotations

from zlc_neutral_atom.devices.rf import (
    CompleteRfTable,
    PrepareRfTable,
    RF_DETUNING_CONTROL_KEY,
    RfTableCapabilitySnapshot,
    RfTableTerminal,
)
from zlc_neutral_atom.runtime.ports import (
    BoundDevice,
    SessionCloseCommand,
    SessionClosedAck,
)
from zlc_neutral_atom.runtime.resources import device_binding_stamp_to_tree
from zlc_storage import canonical_digest

from .apparatus import VirtualRfSource


class VirtualRfTableEndpoint:
    def __init__(self, source: VirtualRfSource, maximum_points: int) -> None:
        if not isinstance(source, VirtualRfSource):
            raise TypeError("source must be VirtualRfSource")
        self._source = source
        self._maximum_points = int(maximum_points)
        if self._maximum_points < 1:
            raise ValueError("maximum_points must be positive")

    def capability_probe(self, binding: BoundDevice) -> RfTableCapabilitySnapshot:
        fingerprint = canonical_digest(
            {
                "owner": "zlc_neutral_atom.virtual-rf-table",
                "binding_stamp": device_binding_stamp_to_tree(binding.binding_stamp),
                "control_key": RF_DETUNING_CONTROL_KEY,
                "unit": "Gamma",
                "clock_source": "sequencer-scan-point",
                "maximum_points": self._maximum_points,
            }
        )
        return RfTableCapabilitySnapshot(
            binding.binding_stamp,
            self._maximum_points,
            1.0,
            fingerprint,
        )

    def execute_command(self, binding: BoundDevice, command: object) -> object:
        if isinstance(command, PrepareRfTable):
            capability = self.capability_probe(binding)
            if command.capability_fingerprint != capability.capability_fingerprint:
                raise RuntimeError("RF prepare uses another capability generation")
            if len(command.table.detuning_gamma) > capability.maximum_points:
                raise ValueError("RF detuning table exceeds endpoint capacity")
            self._source.prepare_table(
                command.session_id,
                command.table.pulse_artifact_digest,
                command.table.digest,
                command.table.detuning_gamma,
            )
            return command.table.digest
        if isinstance(command, CompleteRfTable):
            count, digest = self._source.complete_table(
                command.session_id,
                command.table_digest,
            )
            return RfTableTerminal(
                command.session_id,
                command.table_digest,
                count,
                digest,
            )
        raise TypeError(f"unsupported RF command {type(command).__name__}")

    def close_session(
        self,
        binding: BoundDevice,
        command: SessionCloseCommand,
    ) -> SessionClosedAck:
        self._source.close_session(command.session_id)
        return SessionClosedAck(
            command.session_id,
            binding.binding_instance_id,
            True,
            True,
            True,
            canonical_digest(
                {
                    "owner": "zlc_neutral_atom.virtual-rf-table.close",
                    "session_id": command.session_id,
                    "binding_instance_id": binding.binding_instance_id,
                }
            ),
        )

    def interrupt(self) -> None:
        self._source.set_safe_state()


__all__ = ["VirtualRfTableEndpoint"]
