"""Persistent installation-level safety journal for hardware ownership facts."""

from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any, Mapping

from zlc_storage import (
    FramedJournal,
    canonical_digest,
    durable_mkdir,
)
from zlc_storage.file_lock import (
    FileLockBusy,
    acquire_file_lock,
    open_durable_lock_file,
    release_file_lock,
)

from .resources import (
    HazardRecord,
    HazardAppendStatus,
    RecoveryBundle,
    RecoveryClaim,
    RecoveryEvidence,
    ResourceKey,
    SafeReceipt,
    SafetyDispositionBundle,
    SafetyDispositionRecord,
    SafetyJournalSnapshot,
    SafetyOutcome,
    _replay_entries,
)


SafetyEntry = HazardRecord | SafetyDispositionBundle | RecoveryBundle


class SafetyAuthorityBusy(RuntimeError):
    pass


class _InstallationOwnerLock:
    def __init__(self, path: Path) -> None:
        durable_mkdir(path.parent)
        self._stream = open_durable_lock_file(path)
        self._creator_pid = os.getpid()
        self._closed = False
        try:
            acquire_file_lock(self._stream, blocking=False)
        except FileLockBusy as exc:
            self._stream.close()
            raise SafetyAuthorityBusy(
                "another process owns the installation safety authority"
            ) from exc
        except BaseException:
            self._stream.close()
            raise

    def close(self) -> None:
        if self._closed:
            return
        if os.getpid() != self._creator_pid:
            # Never explicitly unlock a flock inherited from the parent.
            self._stream.close()
            self._closed = True
            return
        error: BaseException | None = None
        try:
            release_file_lock(self._stream)
        except BaseException as exc:
            error = exc
        try:
            self._stream.close()
        except BaseException as exc:
            if error is None:
                error = exc
        self._closed = True
        if error is not None:
            raise error

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _object(value: Any, fields: set[str], kind: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"invalid {kind} fields")
    return value


def _receipt_value(receipt: SafeReceipt | None) -> dict[str, Any] | None:
    if receipt is None:
        return None
    return {
        "key": str(receipt.key),
        "stable_device_identity": receipt.stable_device_identity,
        "connection_generation": receipt.connection_generation,
        "operation_id": receipt.operation_id,
        "acknowledgement_digest": receipt.acknowledgement_digest,
    }


def _receipt_from(value: Any) -> SafeReceipt | None:
    if value is None:
        return None
    item = _object(
        value,
        {
            "key",
            "stable_device_identity",
            "connection_generation",
            "operation_id",
            "acknowledgement_digest",
        },
        "safe receipt",
    )
    return SafeReceipt(
        key=ResourceKey.parse(item["key"]),
        stable_device_identity=item["stable_device_identity"],
        connection_generation=item["connection_generation"],
        operation_id=item["operation_id"],
        acknowledgement_digest=item["acknowledgement_digest"],
    )


def _hazard_value(record: HazardRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "key": str(record.key),
        "stable_device_identity": record.stable_device_identity,
        "connection_generation": record.connection_generation,
        "run_id": record.run_id,
        "activated_at": record.activated_at,
    }


def _hazard_from(value: Any) -> HazardRecord:
    item = _object(
        value,
        {
            "record_id",
            "key",
            "stable_device_identity",
            "connection_generation",
            "run_id",
            "activated_at",
        },
        "hazard",
    )
    return HazardRecord(
        record_id=item["record_id"],
        key=ResourceKey.parse(item["key"]),
        stable_device_identity=item["stable_device_identity"],
        connection_generation=item["connection_generation"],
        run_id=item["run_id"],
        activated_at=item["activated_at"],
    )


def _disposition_value(record: SafetyDispositionRecord) -> dict[str, Any]:
    return {
        "disposition_id": record.disposition_id,
        "key": str(record.key),
        "outcome": record.outcome.value,
        "hazard_record_id": record.hazard_record_id,
        "stable_device_identity": record.stable_device_identity,
        "connection_generation": record.connection_generation,
        "safe_receipt": _receipt_value(record.safe_receipt),
        "reason": record.reason,
        "recovery_action": record.recovery_action,
    }


def _disposition_from(value: Any) -> SafetyDispositionRecord:
    item = _object(
        value,
        {
            "disposition_id",
            "key",
            "outcome",
            "hazard_record_id",
            "stable_device_identity",
            "connection_generation",
            "safe_receipt",
            "reason",
            "recovery_action",
        },
        "safety disposition",
    )
    return SafetyDispositionRecord(
        disposition_id=item["disposition_id"],
        key=ResourceKey.parse(item["key"]),
        outcome=SafetyOutcome(item["outcome"]),
        hazard_record_id=item["hazard_record_id"],
        stable_device_identity=item["stable_device_identity"],
        connection_generation=item["connection_generation"],
        safe_receipt=_receipt_from(item["safe_receipt"]),
        reason=item["reason"],
        recovery_action=item["recovery_action"],
    )


def _safety_value(bundle: SafetyDispositionBundle) -> dict[str, Any]:
    return {
        "kind": "SAFETY_BUNDLE",
        "bundle_id": bundle.bundle_id,
        "run_id": bundle.run_id,
        "records": [_disposition_value(record) for record in bundle.records],
        "recorded_at": bundle.recorded_at,
    }


def _safety_from(value: Any) -> SafetyDispositionBundle:
    item = _object(
        value,
        {"kind", "bundle_id", "run_id", "records", "recorded_at"},
        "safety bundle",
    )
    if item["kind"] != "SAFETY_BUNDLE" or not isinstance(item["records"], list):
        raise ValueError("invalid safety bundle payload")
    return SafetyDispositionBundle(
        bundle_id=item["bundle_id"],
        run_id=item["run_id"],
        records=tuple(_disposition_from(record) for record in item["records"]),
        recorded_at=item["recorded_at"],
    )


def _claim_value(claim: RecoveryClaim) -> dict[str, Any]:
    return {
        "key": str(claim.key),
        "stable_device_identity": claim.stable_device_identity,
        "quarantine_record_ids": list(claim.quarantine_record_ids),
        "hazard_record_ids": list(claim.hazard_record_ids),
    }


def _claim_from(value: Any) -> RecoveryClaim:
    item = _object(
        value,
        {"key", "stable_device_identity", "quarantine_record_ids", "hazard_record_ids"},
        "recovery claim",
    )
    if not isinstance(item["quarantine_record_ids"], list) or not isinstance(
        item["hazard_record_ids"], list
    ):
        raise ValueError("recovery claim ids must be lists")
    return RecoveryClaim(
        key=ResourceKey.parse(item["key"]),
        stable_device_identity=item["stable_device_identity"],
        quarantine_record_ids=tuple(item["quarantine_record_ids"]),
        hazard_record_ids=tuple(item["hazard_record_ids"]),
    )


def _evidence_value(evidence: RecoveryEvidence) -> dict[str, Any]:
    return {
        "stable_device_identity": evidence.stable_device_identity,
        "connection_generation": evidence.connection_generation,
        "health_digest": evidence.health_digest,
        "safe_state_digest": evidence.safe_state_digest,
        "verified_at": evidence.verified_at,
    }


def _evidence_from(value: Any) -> RecoveryEvidence:
    item = _object(
        value,
        {
            "stable_device_identity",
            "connection_generation",
            "health_digest",
            "safe_state_digest",
            "verified_at",
        },
        "recovery evidence",
    )
    return RecoveryEvidence(**item)


def _recovery_value(bundle: RecoveryBundle) -> dict[str, Any]:
    return {
        "kind": "RECOVERY_BUNDLE",
        "bundle_id": bundle.bundle_id,
        "claim": _claim_value(bundle.claim),
        "evidence": _evidence_value(bundle.evidence),
        "recorded_at": bundle.recorded_at,
    }


def _recovery_from(value: Any) -> RecoveryBundle:
    item = _object(
        value,
        {"kind", "bundle_id", "claim", "evidence", "recorded_at"},
        "recovery bundle",
    )
    if item["kind"] != "RECOVERY_BUNDLE":
        raise ValueError("invalid recovery bundle payload")
    return RecoveryBundle(
        bundle_id=item["bundle_id"],
        claim=_claim_from(item["claim"]),
        evidence=_evidence_from(item["evidence"]),
        recorded_at=item["recorded_at"],
    )


def _entries(records: tuple[tuple[str, Any], ...]) -> tuple[SafetyEntry, ...]:
    result: list[SafetyEntry] = []
    for _record_id, value in records:
        if not isinstance(value, dict):
            raise ValueError("safety journal record must be an object")
        kind = value.get("kind")
        if kind == "HAZARD_BATCH":
            item = _object(value, {"kind", "records"}, "hazard batch")
            if not isinstance(item["records"], list):
                raise ValueError("hazard batch records must be a list")
            result.extend(_hazard_from(record) for record in item["records"])
        elif kind == "SAFETY_BUNDLE":
            result.append(_safety_from(value))
        elif kind == "RECOVERY_BUNDLE":
            result.append(_recovery_from(value))
        else:
            raise ValueError("unknown safety journal record kind")
    return tuple(result)


class PersistentSafetyJournal:
    """Crash-replayable journal stored outside switchable artifact roots."""

    def __init__(self, path: str | Path) -> None:
        journal_path = Path(path).resolve()
        self._creator_pid = os.getpid()
        self._lifecycle_lock = threading.RLock()
        self._owner = _InstallationOwnerLock(
            journal_path.with_name(journal_path.name + ".owner.lock")
        )
        self._closed = False
        self._authority_token: object | None = None
        try:
            self._journal = FramedJournal(journal_path)
            _replay_entries(_entries(self._journal.records()))
        except BaseException:
            self._owner.close()
            raise

    def close(self) -> None:
        if not self._in_creator_process():
            self._owner.close()
            self._closed = True
            raise RuntimeError("installation safety journal belongs to another process")
        with self._lifecycle_lock:
            if self._closed:
                return
            if self._authority_token is not None:
                raise RuntimeError(
                    "installation safety journal is owned by ResourceArbiter; use authority shutdown"
                )
            self._close_owner()

    def _bind_authority(self, token: object) -> None:
        self._require_creator_process()
        with self._lifecycle_lock:
            self._ensure_open()
            if self._authority_token is not None:
                raise RuntimeError("installation safety journal already has an authority")
            self._authority_token = token

    def _close_from_authority(self, token: object) -> None:
        self._require_creator_process()
        with self._lifecycle_lock:
            if self._authority_token is not token:
                raise PermissionError("invalid installation safety authority")
            self._authority_token = None
            self._close_owner()

    def _close_owner(self) -> None:
        try:
            self._owner.close()
        finally:
            self._closed = True

    def __enter__(self) -> "PersistentSafetyJournal":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("installation safety authority is closed")

    def _in_creator_process(self) -> bool:
        return os.getpid() == self._creator_pid

    def _require_creator_process(self) -> None:
        if not self._in_creator_process():
            raise RuntimeError("installation safety journal belongs to another process")

    def snapshot(self) -> SafetyJournalSnapshot:
        self._require_creator_process()
        with self._lifecycle_lock:
            self._ensure_open()
            return _replay_entries(_entries(self._journal.records()))

    def append_hazards(self, records: tuple[HazardRecord, ...]) -> HazardAppendStatus:
        self._require_creator_process()
        with self._lifecycle_lock:
            self._ensure_open()
            records = tuple(records)
            if any(not isinstance(record, HazardRecord) for record in records):
                raise TypeError("hazard append requires HazardRecord values")
            if not records:
                return HazardAppendStatus.APPENDED
            if len({record.record_id for record in records}) != len(records):
                raise ValueError("hazard append record ids must be unique")
            if len({record.key for record in records}) != len(records):
                raise ValueError("hazard append keys must be unique")
            if len({record.run_id for record in records}) != 1:
                raise ValueError("one hazard append must belong to one run")
            value = {
                "kind": "HAZARD_BATCH",
                "records": [_hazard_value(record) for record in records],
            }
            record_id = f"hazards:{canonical_digest(value)}"
            appended = self._append_checked(record_id, value)
            if appended:
                return HazardAppendStatus.APPENDED
            unresolved = {
                record.record_id for record in self.snapshot().unresolved_hazards
            }
            requested = {record.record_id for record in records}
            if requested <= unresolved:
                return HazardAppendStatus.ALREADY_UNRESOLVED_SAME
            if requested.isdisjoint(unresolved):
                return HazardAppendStatus.ALREADY_RESOLVED
            raise ValueError("hazard retry has partially resolved durable state")

    def append_safety_bundle(self, bundle: SafetyDispositionBundle) -> None:
        self._require_creator_process()
        with self._lifecycle_lock:
            self._ensure_open()
            if not isinstance(bundle, SafetyDispositionBundle):
                raise TypeError("bundle must be SafetyDispositionBundle")
            self._append_checked(f"safety:{bundle.bundle_id}", _safety_value(bundle))

    def append_recovery_bundle(self, bundle: RecoveryBundle) -> None:
        self._require_creator_process()
        with self._lifecycle_lock:
            self._ensure_open()
            if not isinstance(bundle, RecoveryBundle):
                raise TypeError("bundle must be RecoveryBundle")
            self._append_checked(
                f"recovery:{bundle.bundle_id}",
                _recovery_value(bundle),
            )

    def _append_checked(self, record_id: str, value: dict[str, Any]) -> bool:
        return self._journal.append_checked(
            record_id,
            value,
            lambda records: _replay_entries(_entries(records)),
        )
