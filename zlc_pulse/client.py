"""Bounded current-only client for the remote pulse execution service."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from zlc_storage import decode

from .artifact import CompiledPulseArtifact
from .server import (
    PreparedPulseRef,
    PulseCompletion,
    decode_continuous_failure_message,
    decode_completion_message,
    decode_prepared_ref_message,
    encode_artifact_message,
    encode_prepared_ref_message,
    prepared_pulse_ref_from_tree,
)
from .manifest import PulseTargetManifest, pulse_target_manifest_from_tree
from .target import PulseTarget


@dataclass(frozen=True)
class PulseServerSnapshot:
    connection_generation: str
    manifest: PulseTargetManifest
    clock_hz: float
    geometry_fingerprint: int
    resident_scan_point_capacity: int
    state: str
    prepared_ref: PreparedPulseRef | None
    backend: dict[str, object]

    @property
    def target(self) -> PulseTarget:
        return self.manifest.target


def pulse_server_snapshot_from_tree(tree: object) -> PulseServerSnapshot:
    fields = {
        "schema",
        "connection_generation",
        "manifest",
        "clock_hz",
        "geometry_fingerprint",
        "resident_scan_point_capacity",
        "state",
        "prepared_ref",
        "backend",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("PulseExecutionSnapshot has an unknown field set")
    if tree["schema"] != "zlc_pulse.PulseExecutionSnapshot":
        raise ValueError("PulseExecutionSnapshot schema differs")
    generation = tree["connection_generation"]
    state = tree["state"]
    if not isinstance(generation, str) or not generation:
        raise ValueError("PulseExecutionSnapshot connection generation is invalid")
    if not isinstance(state, str) or not state:
        raise ValueError("PulseExecutionSnapshot state is invalid")
    clock = tree["clock_hz"]
    geometry = tree["geometry_fingerprint"]
    resident_capacity = tree["resident_scan_point_capacity"]
    if (
        isinstance(clock, bool)
        or not isinstance(clock, (int, float))
        or not math.isfinite(float(clock))
        or clock <= 0
    ):
        raise ValueError("PulseExecutionSnapshot clock is invalid")
    if isinstance(geometry, bool) or not isinstance(geometry, int) or not 0 <= geometry <= 0xFFFFFFFF:
        raise ValueError("PulseExecutionSnapshot geometry fingerprint is invalid")
    if (
        isinstance(resident_capacity, bool)
        or not isinstance(resident_capacity, int)
        or resident_capacity < 1
    ):
        raise ValueError("PulseExecutionSnapshot resident scan capacity is invalid")
    backend = tree["backend"]
    if not isinstance(backend, dict):
        raise TypeError("PulseExecutionSnapshot backend must be a map")
    raw_ref = tree["prepared_ref"]
    return PulseServerSnapshot(
        generation,
        pulse_target_manifest_from_tree(tree["manifest"]),
        float(clock),
        geometry,
        resident_capacity,
        state,
        None if raw_ref is None else prepared_pulse_ref_from_tree(raw_ref),
        dict(backend),
    )


class RemotePulseExecutionClient:
    """One non-reconnecting control owner for one server connection generation."""

    def __init__(
        self,
        connection: object,
        interrupt_connection: object,
        *,
        transport_timeout_seconds: float = 120.0,
    ) -> None:
        timeout = float(transport_timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("transport_timeout_seconds must be finite and positive")
        root = getattr(connection, "root", None)
        if root is None:
            raise TypeError("pulse client connection must expose root")
        for method in (
            "current_snapshot",
            "current_prepare",
            "current_fire",
            "current_complete",
            "current_wait_continuous_failure",
        ):
            if not callable(getattr(root, method, None)):
                raise TypeError(f"pulse server connection is missing {method}()")
        if interrupt_connection is connection:
            raise ValueError(
                "pulse control and interrupt paths require distinct connections"
            )
        interrupt_root = getattr(interrupt_connection, "root", None)
        if interrupt_root is None or not callable(
            getattr(interrupt_root, "current_interrupt_safe_state", None)
        ):
            raise TypeError(
                "pulse interrupt connection is missing current_interrupt_safe_state()"
            )
        self._connection = connection
        self._interrupt_connection = interrupt_connection
        self._root = root
        self._interrupt_root = interrupt_root
        self._transport_timeout = timeout
        self._safe_state_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._close_failure_summary: str | None = None
        snapshot = self.snapshot()
        self._generation = snapshot.connection_generation

    @classmethod
    def connect(
        cls,
        host: str,
        port: int = 18861,
        *,
        transport_timeout_seconds: float = 120.0,
    ) -> "RemotePulseExecutionClient":
        try:
            import rpyc
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError("remote pulse execution requires rpyc") from exc
        timeout = float(transport_timeout_seconds)
        use_ipv6 = ":" in str(host)
        connection = rpyc.connect(
            host,
            int(port),
            config={"allow_public_attrs": True, "sync_request_timeout": timeout},
            ipv6=use_ipv6,
        )
        interrupt_connection = None
        try:
            interrupt_connection = rpyc.connect(
                host,
                int(port),
                config={"allow_public_attrs": True, "sync_request_timeout": timeout},
                ipv6=use_ipv6,
            )
            return cls(
                connection,
                interrupt_connection,
                transport_timeout_seconds=timeout,
            )
        except BaseException:
            connection.close()
            if interrupt_connection is not None:
                interrupt_connection.close()
            raise

    @property
    def connection_generation(self) -> str:
        return self._generation

    @property
    def transport_timeout_seconds(self) -> float:
        return self._transport_timeout

    def snapshot(self) -> PulseServerSnapshot:
        self._require_open()
        snapshot = pulse_server_snapshot_from_tree(
            decode(bytes(self._root.current_snapshot()))
        )
        known = getattr(self, "_generation", None)
        if known is not None and snapshot.connection_generation != known:
            raise RuntimeError("pulse server connection generation changed without reconnect")
        return snapshot

    def prepare(self, artifact: CompiledPulseArtifact) -> PreparedPulseRef:
        self._require_open()
        if not isinstance(artifact, CompiledPulseArtifact):
            raise TypeError("artifact must be CompiledPulseArtifact")
        reference = decode_prepared_ref_message(
            bytes(self._root.current_prepare(encode_artifact_message(artifact)))
        )
        self._validate_reference(reference)
        if reference.artifact_digest != artifact.fingerprint:
            raise RuntimeError("pulse server prepared a different artifact digest")
        return reference

    def fire(self, reference: PreparedPulseRef) -> None:
        self._validate_reference(reference)
        if self._root.current_fire(encode_prepared_ref_message(reference)) is not True:
            raise RuntimeError("pulse server did not acknowledge FIRE")

    def complete(
        self,
        reference: PreparedPulseRef,
        *,
        timeout: float,
    ) -> PulseCompletion:
        self._validate_reference(reference)
        logical_timeout = float(timeout)
        if not math.isfinite(logical_timeout) or logical_timeout <= 0:
            raise ValueError("completion timeout must be finite and positive")
        if logical_timeout >= self._transport_timeout:
            raise ValueError("completion timeout must be shorter than the transport backstop")
        completion = decode_completion_message(
            bytes(
                self._root.current_complete(
                    encode_prepared_ref_message(reference),
                    logical_timeout,
                )
            )
        )
        if completion.prepared_ref != reference:
            raise RuntimeError("pulse completion belongs to another prepared reference")
        return completion

    def wait_continuous_failure(
        self,
        reference: PreparedPulseRef,
        *,
        timeout: float,
    ) -> str | None:
        """Long-poll one backend-owned failure notification on the control lane."""

        self._validate_reference(reference)
        logical_timeout = float(timeout)
        if not math.isfinite(logical_timeout) or logical_timeout <= 0.0:
            raise ValueError("continuous failure wait timeout must be positive")
        if logical_timeout >= self._transport_timeout:
            raise ValueError(
                "continuous failure wait must be shorter than the transport backstop"
            )
        return decode_continuous_failure_message(
            bytes(
                self._root.current_wait_continuous_failure(
                    encode_prepared_ref_message(reference),
                    logical_timeout,
                )
            )
        )

    def safe_state(self, *, timeout: float | None = None) -> PulseServerSnapshot:
        """Enter SAFE through the interrupt connection within a logical deadline.

        The transport timeout is only a final socket backstop.  Callers own the
        shorter operation deadline; if it expires, client authority is revoked
        and both local streams are severed so a late reply cannot be accepted.
        The server coalesces the resulting disconnect SAFE with the in-flight
        generation-bound SAFE request.
        """

        logical_timeout = (
            self._transport_timeout * 0.9
            if timeout is None
            else float(timeout)
        )
        if not math.isfinite(logical_timeout) or logical_timeout <= 0:
            raise ValueError("safe_state timeout must be finite and positive")
        if logical_timeout >= self._transport_timeout:
            raise ValueError(
                "safe_state timeout must be shorter than the transport backstop"
            )
        with self._safe_state_lock:
            return self._safe_state_owned(logical_timeout)

    def _safe_state_owned(self, logical_timeout: float) -> PulseServerSnapshot:
        self._require_open()
        try:
            import rpyc
            from rpyc.core.async_ import AsyncResultTimeout
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError("remote pulse execution requires rpyc") from exc
        try:
            pending = rpyc.timed(
                self._interrupt_root.current_interrupt_safe_state,
                logical_timeout,
            )(self._generation)
            payload = pending.value
        except AsyncResultTimeout as exc:
            self._abort_connections()
            raise TimeoutError(
                "remote pulse safe_state exceeded its logical deadline"
            ) from exc
        snapshot = pulse_server_snapshot_from_tree(decode(bytes(payload)))
        if snapshot.connection_generation != self._generation:
            raise RuntimeError("interrupt safe_state returned another connection generation")
        if snapshot.state != "SAFE" or snapshot.prepared_ref is not None:
            raise RuntimeError("pulse server acknowledged safe_state without publishing SAFE")
        return snapshot

    def close(self) -> None:
        with self._safe_state_lock:
            with self._close_lock:
                if self._closed:
                    if self._close_failure_summary is not None:
                        raise RuntimeError(
                            "remote pulse client remains fail-closed after an "
                            "unconfirmed shutdown: " + self._close_failure_summary
                        )
                    return
            failure: BaseException | None = None
            try:
                self._safe_state_owned(self._transport_timeout * 0.9)
            except BaseException as error:
                failure = error
            try:
                self._close_connections()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    try:
                        failure.add_note(
                            "closing the remote pulse transports also failed: "
                            f"{type(error).__name__}: {error}"
                        )
                    except BaseException:
                        pass
            if failure is not None:
                summary = f"{type(failure).__name__}: {failure}"
                with self._close_lock:
                    self._close_failure_summary = summary
                raise failure

    def _abort_connections(self) -> None:
        """Revoke both local transports without waiting for remote replies."""

        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        for connection in (self._connection, self._interrupt_connection):
            channel = getattr(connection, "_channel", None)
            abort = getattr(channel, "close", None)
            if callable(abort):
                try:
                    abort()
                except Exception:
                    pass

    def _close_connections(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        close = getattr(self._connection, "close", None)
        try:
            if callable(close):
                close()
        finally:
            close_interrupt = getattr(
                self._interrupt_connection,
                "close",
                None,
            )
            if callable(close_interrupt):
                close_interrupt()

    def _validate_reference(self, reference: PreparedPulseRef) -> None:
        self._require_open()
        if not isinstance(reference, PreparedPulseRef):
            raise TypeError("reference must be PreparedPulseRef")
        if reference.connection_generation != self._generation:
            raise RuntimeError("prepared pulse reference belongs to another connection generation")

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("remote pulse execution client is closed")


__all__ = [
    "PulseServerSnapshot",
    "RemotePulseExecutionClient",
    "pulse_server_snapshot_from_tree",
]
