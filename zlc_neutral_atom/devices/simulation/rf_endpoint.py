"""Virtual endpoint for the single pulse-clocked RF table Port."""

from __future__ import annotations

from zlc_neutral_atom.devices.rf import (
    CompleteRfTable,
    PrepareRfTable,
    RfTableCapabilitySnapshot,
    RfTableTerminal,
)
from zlc_neutral_atom.runtime.ports import (
    BoundDevice,
    SessionCloseCommand,
    SessionClosedAck,
)

from .apparatus import VirtualRfSource


class VirtualRfTableEndpoint:
    def __init__(self, source: VirtualRfSource) -> None:
        if not isinstance(source, VirtualRfSource):
            raise TypeError("source must be VirtualRfSource")
        self._source = source

    def capability_probe(self, binding: BoundDevice) -> RfTableCapabilitySnapshot:
        return RfTableCapabilitySnapshot(
            binding.binding_stamp,
            1.0,
        )

    def execute_command(self, binding: BoundDevice, command: object) -> object:
        if isinstance(command, PrepareRfTable):
            return self._source.prepare_table(
                command.session_id,
                command.table,
            )
        if isinstance(command, CompleteRfTable):
            point_indices = self._source.complete_table(
                command.session_id,
                command.table,
            )
            return RfTableTerminal(
                command.session_id,
                point_indices,
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
        )

    def interrupt(self) -> None:
        self._source.set_safe_state()


__all__ = ["VirtualRfTableEndpoint"]
