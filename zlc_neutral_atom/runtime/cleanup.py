"""Run-scoped safety proof and cleanup result facts."""

from __future__ import annotations

from dataclasses import dataclass

from .resources import (
    ResourceKey,
    SafeReceipt,
    SafetyDecision,
    SafetyOutcome,
)


_SAFETY_PROOF_TOKEN = object()


class SafetyProof:
    """Opaque, run-scoped proof minted after an adapter safety acknowledgement."""

    __slots__ = ("_run_id", "_receipt", "_nonce")

    def __init__(
        self,
        token: object,
        *,
        run_id: str,
        receipt: SafeReceipt,
        nonce: object,
    ) -> None:
        if token is not _SAFETY_PROOF_TOKEN:
            raise PermissionError("SafetyProof can only be minted by RunContext")
        object.__setattr__(self, "_run_id", run_id)
        object.__setattr__(self, "_receipt", receipt)
        object.__setattr__(self, "_nonce", nonce)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("SafetyProof is immutable")

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def receipt(self) -> SafeReceipt:
        return self._receipt


def _mint_safety_proof(
    *,
    run_id: str,
    receipt: SafeReceipt,
    nonce: object,
) -> SafetyProof:
    return SafetyProof(
        _SAFETY_PROOF_TOKEN,
        run_id=run_id,
        receipt=receipt,
        nonce=nonce,
    )


@dataclass(frozen=True)
class CleanupReport:
    """Plan-owned cleanup facts; the controller checks total hazard coverage."""

    safety_proofs: tuple[SafetyProof, ...] = ()
    decisions: tuple[SafetyDecision, ...] = ()
    errors: tuple[BaseException, ...] = ()

    def __post_init__(self) -> None:
        proofs = tuple(self.safety_proofs)
        decisions = tuple(self.decisions)
        errors = tuple(self.errors)
        if any(not isinstance(value, SafetyProof) for value in proofs):
            raise TypeError("cleanup safety_proofs must contain SafetyProof values")
        if len({id(value) for value in proofs}) != len(proofs):
            raise ValueError("cleanup safety proofs must be unique")
        if any(not isinstance(value, SafetyDecision) for value in decisions):
            raise TypeError("cleanup decisions must contain SafetyDecision values")
        if len({value.key for value in decisions}) != len(decisions):
            raise ValueError("cleanup decisions must have unique ResourceKeys")
        if any(value.outcome is SafetyOutcome.SAFE for value in decisions):
            raise ValueError("SAFE cleanup decisions require a RunContext SafetyProof")
        if any(not isinstance(error, BaseException) for error in errors):
            raise TypeError("cleanup errors must contain exceptions")
        object.__setattr__(self, "safety_proofs", proofs)
        object.__setattr__(
            self,
            "decisions",
            tuple(sorted(decisions, key=lambda value: value.key)),
        )
        object.__setattr__(self, "errors", errors)

    @classmethod
    def safe(
        cls,
        proofs: tuple[SafetyProof, ...],
        *,
        errors: tuple[BaseException, ...] = (),
    ) -> "CleanupReport":
        return cls(safety_proofs=tuple(proofs), errors=tuple(errors))

    @classmethod
    def unsafe(
        cls,
        keys: tuple[ResourceKey, ...],
        *,
        reason: str,
        recovery_action: str,
        errors: tuple[BaseException, ...] = (),
    ) -> "CleanupReport":
        return cls(
            decisions=tuple(
                SafetyDecision.unsafe(
                    key,
                    reason=reason,
                    recovery_action=recovery_action,
                )
                for key in keys
            ),
            errors=tuple(errors),
        )
