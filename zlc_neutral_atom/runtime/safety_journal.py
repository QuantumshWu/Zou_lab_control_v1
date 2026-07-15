"""Persistent installation-level safety journal for hardware ownership facts."""

from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any, Mapping

from zlc_storage import (
    FramedJournal,
    canonical_digest,
)
from zlc_storage.file_lock import FileLockBusy

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
    _SafetyProjection,
    device_binding_stamp_from_tree,
    device_binding_stamp_to_tree,
    physical_device_identity_from_tree,
    physical_device_identity_to_tree,
)


SafetyEntry = HazardRecord | SafetyDispositionBundle | RecoveryBundle


class SafetyAuthorityBusy(RuntimeError):
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
        "binding_stamp": device_binding_stamp_to_tree(receipt.binding_stamp),
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
            "binding_stamp",
            "operation_id",
            "acknowledgement_digest",
        },
        "safe receipt",
    )
    return SafeReceipt(
        key=ResourceKey.parse(item["key"]),
        binding_stamp=device_binding_stamp_from_tree(item["binding_stamp"]),
        operation_id=item["operation_id"],
        acknowledgement_digest=item["acknowledgement_digest"],
    )


def _hazard_value(record: HazardRecord) -> dict[str, Any]:
    return {
        "record_id": record.record_id,
        "key": str(record.key),
        "binding_stamp": device_binding_stamp_to_tree(record.binding_stamp),
        "run_id": record.run_id,
        "activated_at": record.activated_at,
    }


def _hazard_from(value: Any) -> HazardRecord:
    item = _object(
        value,
        {
            "record_id",
            "key",
            "binding_stamp",
            "run_id",
            "activated_at",
        },
        "hazard",
    )
    return HazardRecord(
        record_id=item["record_id"],
        key=ResourceKey.parse(item["key"]),
        binding_stamp=device_binding_stamp_from_tree(item["binding_stamp"]),
        run_id=item["run_id"],
        activated_at=item["activated_at"],
    )


def _disposition_value(record: SafetyDispositionRecord) -> dict[str, Any]:
    return {
        "disposition_id": record.disposition_id,
        "key": str(record.key),
        "outcome": record.outcome.value,
        "hazard_record_id": record.hazard_record_id,
        "binding_stamp": device_binding_stamp_to_tree(record.binding_stamp),
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
            "binding_stamp",
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
        binding_stamp=device_binding_stamp_from_tree(item["binding_stamp"]),
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
        "physical_identity": physical_device_identity_to_tree(claim.physical_identity),
        "blocking_record_id": claim.blocking_record_id,
    }


def _claim_from(value: Any) -> RecoveryClaim:
    item = _object(
        value,
        {"key", "physical_identity", "blocking_record_id"},
        "recovery claim",
    )
    return RecoveryClaim(
        key=ResourceKey.parse(item["key"]),
        physical_identity=physical_device_identity_from_tree(item["physical_identity"]),
        blocking_record_id=item["blocking_record_id"],
    )


def _evidence_value(evidence: RecoveryEvidence) -> dict[str, Any]:
    return {
        "binding_stamp": device_binding_stamp_to_tree(evidence.binding_stamp),
        "safe_state_digest": evidence.safe_state_digest,
    }


def _evidence_from(value: Any) -> RecoveryEvidence:
    item = _object(
        value,
        {
            "binding_stamp",
            "safe_state_digest",
        },
        "recovery evidence",
    )
    return RecoveryEvidence(
        binding_stamp=device_binding_stamp_from_tree(item["binding_stamp"]),
        safe_state_digest=item["safe_state_digest"],
    )


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
        self._closed = False
        self._authority_token: object | None = None
        self._session = None
        try:
            self._session = FramedJournal.open_exclusive(journal_path)
            self._projection = _SafetyProjection()
            for entry in _entries(self._session.records()):
                self._projection.apply(entry)
        except FileLockBusy as exc:
            raise SafetyAuthorityBusy(
                "another process owns the installation safety authority"
            ) from exc
        except BaseException:
            if self._session is not None:
                self._session.close()
            raise

    def close(self) -> None:
        if not self._in_creator_process():
            if self._closed:
                return
            assert self._session is not None
            self._session.close()
            self._session = None
            self._closed = True
            raise RuntimeError("installation safety journal belongs to another process")
        with self._lifecycle_lock:
            if self._closed:
                # Logical authority may already be closed even if closing the
                # now-unlocked OS handle previously failed.  Retrying here is
                # cleanup only and can never revive journal authority.
                if self._session is not None:
                    self._session.close()
                    self._session = None
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
            try:
                self._close_owner()
            finally:
                # Once the file lock is gone, the logical authority is gone
                # even if closing its inert handle raised.  Never leave a token
                # that can still serve reads or appends without the hard lock.
                if self._closed:
                    self._authority_token = None

    def _close_owner(self) -> None:
        session = self._session
        assert session is not None
        try:
            session.close()
        except BaseException:
            if not session.owns_file_lock:
                self._closed = True
            raise
        self._session = None
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
            return self._projection.snapshot()

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
            appended = self._append_checked(record_id, value, records)
            if appended:
                return HazardAppendStatus.APPENDED
            unresolved = set(self._projection.hazards)
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
            self._append_checked(
                f"safety:{bundle.bundle_id}",
                _safety_value(bundle),
                (bundle,),
            )

    def append_recovery_bundle(self, bundle: RecoveryBundle) -> None:
        self._require_creator_process()
        with self._lifecycle_lock:
            self._ensure_open()
            if not isinstance(bundle, RecoveryBundle):
                raise TypeError("bundle must be RecoveryBundle")
            self._append_checked(
                f"recovery:{bundle.bundle_id}",
                _recovery_value(bundle),
                (bundle,),
            )

    def _append_checked(
        self,
        record_id: str,
        value: dict[str, Any],
        entries: tuple[SafetyEntry, ...],
    ) -> bool:
        assert self._session is not None
        try:
            for entry in entries:
                self._projection.apply(entry)
            appended = self._session.append(record_id, value)
        except BaseException:
            # append() repairs an ambiguous torn tail while retaining the lock;
            # rebuild the semantic projection from that recovered byte stream.
            projection = _SafetyProjection()
            for entry in _entries(self._session.records()):
                projection.apply(entry)
            self._projection = projection
            raise
        return appended
