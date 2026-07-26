"""The one current Port for a pulse-scan-clocked RF detuning table."""

from __future__ import annotations

import math
from dataclasses import dataclass

from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.ports import (
    BoundDevice,
    SafetyInterrupt,
    SafetyOperation,
    VerifiedDeviceCapability,
    admit_bound_capability,
    cleanup_device_session,
)
from zlc_neutral_atom.runtime.resources import DeviceBindingStamp, ResourceClaim
from zlc_neutral_atom.runtime.run import RunContext
from zlc_storage import canonical_digest, canonical_text, positive_integer, positive_real


RF_DETUNING_CONTROL_KEY = "two_photon_detuning_gamma"


def _table(values: object) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("RF detuning table must be a numeric sequence")
    try:
        raw = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("RF detuning table must be a numeric sequence") from exc
    if not raw:
        raise ValueError("RF detuning table must not be empty")
    result: list[float] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("RF detuning values must be real numbers")
        item = float(value)
        if not math.isfinite(item):
            raise ValueError("RF detuning values must be finite")
        result.append(item)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class RfTableCapabilitySnapshot:
    binding_stamp: DeviceBindingStamp
    max_blocking_call_seconds: float
    capability_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.binding_stamp, DeviceBindingStamp):
            raise TypeError("binding_stamp must be DeviceBindingStamp")
        object.__setattr__(
            self,
            "max_blocking_call_seconds",
            positive_real(
                self.max_blocking_call_seconds,
                "RF max_blocking_call_seconds",
            ),
        )
        canonical_text(self.capability_fingerprint, "RF capability_fingerprint")


@dataclass(frozen=True, slots=True)
class RfDetuningTable:
    pulse_artifact_digest: str
    detuning_gamma: tuple[float, ...]

    def __post_init__(self) -> None:
        canonical_text(self.pulse_artifact_digest, "pulse_artifact_digest")
        object.__setattr__(self, "detuning_gamma", _table(self.detuning_gamma))

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "control_key": RF_DETUNING_CONTROL_KEY,
                "unit": "Gamma",
                "clock_source": "sequencer-scan-point",
                "pulse_artifact_digest": self.pulse_artifact_digest,
                "detuning_gamma": self.detuning_gamma,
            }
        )

    @property
    def advancement_digest(self) -> str:
        """Evidence for the only valid table walk: every index, once, in order."""

        return canonical_digest(
            {
                "pulse_artifact_digest": self.pulse_artifact_digest,
                "table_digest": self.digest,
                "advanced_points": tuple(range(len(self.detuning_gamma))),
            }
        )


@dataclass(frozen=True, slots=True)
class PrepareRfTable:
    session_id: str
    table: RfDetuningTable
    capability_fingerprint: str

    def __post_init__(self) -> None:
        canonical_text(self.session_id, "RF session_id")
        if not isinstance(self.table, RfDetuningTable):
            raise TypeError("table must be RfDetuningTable")
        canonical_text(self.capability_fingerprint, "RF capability_fingerprint")


@dataclass(frozen=True, slots=True)
class CompleteRfTable:
    session_id: str
    table_digest: str

    def __post_init__(self) -> None:
        canonical_text(self.session_id, "RF session_id")
        canonical_text(self.table_digest, "RF table_digest")


@dataclass(frozen=True, slots=True)
class RfTableTerminal:
    session_id: str
    table_digest: str
    advanced_points: int
    advancement_digest: str

    def __post_init__(self) -> None:
        for field in ("session_id", "table_digest", "advancement_digest"):
            canonical_text(getattr(self, field), f"RF {field}")
        object.__setattr__(
            self,
            "advanced_points",
            positive_integer(self.advanced_points, "RF advanced_points"),
        )


@dataclass(frozen=True, slots=True)
class BoundRfTablePort:
    capability_attestation: VerifiedDeviceCapability

    def __post_init__(self) -> None:
        admit_bound_capability(
            self.capability_attestation,
            RfTableCapabilitySnapshot,
        )
        if not self.device.session_cleanup_capable:
            raise ValueError("RF table Port requires session cleanup")
        if SafetyOperation.SAFE_STATE not in self.device.interrupt_capabilities:
            raise ValueError("RF table Port requires SAFE_STATE")

    @property
    def device(self) -> BoundDevice:
        return self.capability_attestation.device

    @property
    def capability(self) -> RfTableCapabilitySnapshot:
        snapshot = self.capability_attestation.snapshot
        assert isinstance(snapshot, RfTableCapabilitySnapshot)
        return snapshot

    @property
    def resource_claim(self) -> ResourceClaim:
        return ResourceClaim(self.device.key)

    @property
    def interrupt_operations(self) -> tuple[SafetyInterrupt, ...]:
        return (SafetyInterrupt(self.device.key, SafetyOperation.SAFE_STATE),)

    def prepare(
        self,
        context: RunContext,
        session_id: str,
        table: RfDetuningTable,
    ) -> None:
        if not isinstance(table, RfDetuningTable):
            raise TypeError("table must be RfDetuningTable")
        if (
            self.device.validate_capability(self.capability_attestation)
            is not self.capability
        ):
            raise RuntimeError("RF capability attestation changed")
        observed_digest = context.device(self.device.key).execute(
            PrepareRfTable(
                session_id,
                table,
                self.capability.capability_fingerprint,
            )
        )
        if observed_digest != table.digest:
            raise RuntimeError("RF endpoint prepared another table")

    def complete(
        self,
        context: RunContext,
        session_id: str,
        table: RfDetuningTable,
    ) -> RfTableTerminal:
        terminal = context.device(self.device.key).execute(
            CompleteRfTable(session_id, table.digest)
        )
        if not isinstance(terminal, RfTableTerminal):
            raise TypeError("RF endpoint returned another terminal type")
        if (
            terminal.session_id != session_id
            or terminal.table_digest != table.digest
            or terminal.advanced_points != len(table.detuning_gamma)
            or terminal.advancement_digest != table.advancement_digest
        ):
            raise RuntimeError("RF terminal does not cover the frozen table")
        return terminal

    def cleanup(self, context: RunContext, session_id: str) -> CleanupReport:
        return cleanup_device_session(
            context.cleanup_device(self.device.key),
            session_id,
            self.capability.max_blocking_call_seconds,
        )

    def verify_idle(self, _context: RunContext) -> CleanupReport:
        return CleanupReport.complete()


__all__ = [
    "BoundRfTablePort",
    "CompleteRfTable",
    "PrepareRfTable",
    "RF_DETUNING_CONTROL_KEY",
    "RfDetuningTable",
    "RfTableCapabilitySnapshot",
    "RfTableTerminal",
]
