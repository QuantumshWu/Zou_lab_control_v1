"""Compiled-only owner of the frozen pulse-streamer register protocol."""

from __future__ import annotations

import math
import threading
import time
from typing import Protocol, Sequence

from fpga.pulse_streamer.host.image import (
    CMD_FIRE,
    CMD_LOAD,
    CMD_SAFE,
    CTRL_WORDS,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_LOADED,
    STATUS_UNDERFLOW,
    CtrlWords,
    StreamerParams,
    build_fingerprint,
    region_bases,
)

from ..artifact import CompiledPulseArtifact, PulseExecutionForm
from ..deployment import (
    _validate_artifact_against_bound_deployment,
    validate_deployed_target,
)
from ..evidence import (
    AUTONOMOUS_TABLE_READ_RECIPE,
    POST_TERMINAL_TAIL_WAIT_RECIPE,
    STATIC_STATUS_READ_RECIPE,
    AutonomousTableTerminalEvidence,
    PostTerminalTailEvidence,
    PulseBackendCompletion,
    PulseHardwareTerminalEvidence,
    StaticOnceTerminalEvidence,
    hardware_terminal_evidence_to_tree,
    validate_terminal_for_artifact,
)
from ..fpga import scan_chunk_wire_words
from ..target import PulseTarget
from .lease import DeviceLease


_CONTINUOUS_OBSERVER_INTERVAL_SECONDS = 0.2


class TransportAborted(RuntimeError):
    """An in-flight register transaction was cancelled during safe-state."""


class RegisterTransport(Protocol):
    """Ordered word-addressed access to the already-programmed streamer."""

    transport_id: str
    def start(self) -> None: ...

    def close(self) -> None: ...

    def write_words(
        self,
        rows: Sequence[tuple[int, int]],
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> None: ...

    def read_word(
        self,
        word_offset: int,
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> int: ...

    def record_diagnostic(self, name: str, text: str) -> None: ...


class DeployedStreamerSession:
    """One exact compiled artifact, one FIRE, one terminal proof, one safe owner."""

    def __init__(
        self,
        transport: RegisterTransport,
        *,
        device_lease: DeviceLease,
        deployed_target: PulseTarget,
        params: StreamerParams,
        clock_hz: float,
        load_timeout: float = 5.0,
        action_timeout: float = 120.0,
        terminal_poll_interval: float = 0.001,
    ) -> None:
        for method in (
            "start",
            "close",
            "write_words",
            "read_word",
            "record_diagnostic",
        ):
            if not callable(getattr(transport, method, None)):
                raise TypeError(f"register transport is missing {method}()")
        for method in ("acquire", "release"):
            if not callable(getattr(device_lease, method, None)):
                raise TypeError(f"device lease is missing {method}()")
        if not isinstance(params, StreamerParams):
            raise TypeError("params must be StreamerParams")
        validate_deployed_target(deployed_target, params)
        if clock_hz <= 0:
            raise ValueError("clock_hz must be positive")
        bounded = (load_timeout, action_timeout, terminal_poll_interval)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in bounded
        ):
            raise ValueError("session timeouts and poll interval must be finite and positive")

        self.transport = transport
        self.device_lease = device_lease
        self.params = params
        self.clock_hz = float(clock_hz)
        self.deployed_target = deployed_target
        self.load_timeout = float(load_timeout)
        self.action_timeout = float(action_timeout)
        self.terminal_poll_interval = float(terminal_poll_interval)

        self._lock = threading.RLock()
        self._safe_lock = threading.Lock()
        self._state = "IDLE"
        self._closed = False
        self._closing = False
        self._started = False
        self._close_failure: BaseException | None = None
        self._operation_epoch = 0
        self._layout_verified = False
        self._artifact: CompiledPulseArtifact | None = None
        self._artifact_digest: str | None = None
        self._total_points = 0
        self._total_chunks = 0
        self._scan_chunk_rows: tuple[
            tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]], ...
        ] = ()
        self._next_monotonic_chunk = 2
        self._bank_ready = 0b11
        self._scan_sweep = 0
        self._last_cursor = 0
        self._tail_seconds = 0.0
        self._drain_until = 0.0

        self._worker: threading.Thread | None = None
        self._worker_stop: threading.Event | None = None
        self._fire_gate: threading.Event | None = None
        self._terminal_event = threading.Event()
        self._terminal_error: BaseException | None = None
        self._terminal: PulseHardwareTerminalEvidence | None = None
        self._terminal_observed_monotonic_ns: int | None = None
        self._status_sample_count = 0
        self._cursor_sample_count = 0
        self._underflow_observed = False
        self._operation_stop = threading.Event()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def start(self) -> "DeployedStreamerSession":
        with self._lock:
            if self._closed:
                raise RuntimeError("pulse session is closed")
            if self._closing or self._state == "CLOSE_FAILED":
                raise RuntimeError("pulse session close failed and cannot be restarted")
            if self._started:
                return self
            self.device_lease.acquire()
            try:
                self.transport.start()
            except BaseException:
                self.device_lease.release()
                raise
            self._started = True
        return self

    def close(self) -> None:
        with self._lock:
            if self._closed:
                if self._close_failure is not None:
                    raise RuntimeError(
                        "pulse session was revoked without verified safe closure"
                    ) from self._close_failure
                return
            if self._closing:
                raise RuntimeError("pulse session close is already in progress")
            self._closing = True
            self._operation_epoch += 1
            self._operation_stop.set()
            started = self._started
            layout_verified = self._layout_verified
        failure: BaseException | None = None
        try:
            if started and layout_verified:
                try:
                    self.safe_state()
                except BaseException as error:
                    failure = error
        finally:
            try:
                self._stop_worker()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(
                        "pulse worker teardown also failed while closing: "
                        f"{type(error).__name__}: {error}"
                    )
            transport_closed = False
            try:
                self.transport.close()
                transport_closed = True
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(
                        "register transport close also failed: "
                        f"{type(error).__name__}: {error}"
                    )
            with self._lock:
                self._closed = transport_closed
                self._close_failure = failure if transport_closed else None
                self._state = (
                    "CLOSED"
                    if transport_closed and failure is None
                    else "CLOSE_FAILED"
                )
                if transport_closed and failure is None:
                    self._started = False
                if not transport_closed:
                    self._closing = False
            if transport_closed and failure is None:
                try:
                    self.device_lease.release()
                except BaseException as error:
                    with self._lock:
                        self._close_failure = error
                        self._state = "CLOSE_FAILED"
                    failure = error
        if failure is not None:
            raise failure

    def check_register_layout(self) -> None:
        self._require_started()
        self._check_register_layout(stop=None)

    def _check_register_layout(self, *, stop: threading.Event | None) -> None:
        expected = build_fingerprint(self.params) & 0xFFFFFFFF
        actual = self.transport.read_word(CtrlWords.LAYOUT_ID, stop=stop)
        if actual != expected:
            raise RuntimeError(
                "geometry/layout mismatch: running bitstream reports "
                f"0x{actual:08X}, approved host geometry requires 0x{expected:08X}"
            )
        with self._lock:
            self._layout_verified = True

    def clear_host_config(self) -> None:
        """Verify layout, then clear only host-owned configuration."""

        self.check_register_layout()
        deadline = time.monotonic() + self.action_timeout
        self._enter_safe(deadline=deadline)
        bases = region_bases(self.params)
        rows = [
            (bases["ctrl"] + word, 0)
            for word in range(20, CTRL_WORDS)
        ]
        rows.extend(
            (bases["delay"] + word, 0)
            for word in range(self.params.delay_region_words)
        )
        self.transport.write_words(
            tuple(rows),
            deadline=deadline,
        )
        self._verify_clock_muxes_disabled(deadline=deadline)
        self._enter_safe(deadline=deadline)
        with self._lock:
            self._state = "SAFE"

    def transport_self_test(self, *, count: int = 16) -> None:
        """Exercise ordered multi-write and readback in side-effect-free CTRL scratch."""

        self.check_register_layout()
        base = region_bases(self.params)["ctrl"] + self.params.ctrl_scratch_base
        length = max(2, min(int(count), CTRL_WORDS - self.params.ctrl_scratch_base))
        pattern = tuple((base + i, (0xC0DE0000 + i) & 0xFFFFFFFF) for i in range(length))
        try:
            self.transport.write_words(pattern)
            actual = tuple(self.transport.read_word(base + i) for i in range(length))
        finally:
            self.transport.write_words(tuple((base + i, 0) for i in range(length)))
        expected = tuple(value for _address, value in pattern)
        if actual != expected:
            raise RuntimeError(
                f"{self.transport.transport_id} register self-test readback mismatch"
            )

    def prepare(self, artifact: CompiledPulseArtifact) -> None:
        """Validate, upload, and LOAD one immutable current artifact before FIRE."""

        self._require_started()
        with self._lock:
            if self._closed:
                raise RuntimeError("pulse session is closed")
            if self._state not in {"IDLE", "SAFE", "DONE"}:
                raise RuntimeError(
                    f"pulse session state {self._state} requires verified safe_state "
                    "before prepare"
                )
            prior_state = self._state
            self._operation_epoch += 1
            operation_epoch = self._operation_epoch
            self._operation_stop = threading.Event()
            operation_stop = self._operation_stop
        _validate_artifact_against_bound_deployment(
            artifact,
            self.deployed_target,
            self.params,
            self.clock_hz,
        )
        artifact_digest = artifact.fingerprint
        ir = artifact.target_ir
        total_points = len(ir.scan_points)
        total_chunks = (
            (total_points + self.params.bank_size - 1) // self.params.bank_size
            if total_points
            else 0
        )
        scan_chunk_rows = tuple(
            (
                scan_chunk_wire_words(ir, self.params, chunk, target_bank=0),
                scan_chunk_wire_words(ir, self.params, chunk, target_bank=1),
            )
            for chunk in range(total_chunks)
        )
        with self._lock:
            if (
                operation_epoch != self._operation_epoch
                or operation_stop.is_set()
                or self._closed
                or self._closing
            ):
                raise TransportAborted(
                    "pulse prepare was superseded during deployment validation"
                )
        self._stop_worker()
        # This server owns a frozen bitstream for the transport lifetime.  Bring-up
        # verifies its layout before RPC admission, so the same JTAG read need not be
        # repeated for every On Pulse.
        if not self._layout_verified:
            self._check_register_layout(stop=operation_stop)
        remaining_tail = self._drain_until - time.monotonic()
        if remaining_tail > 0 and operation_stop.wait(remaining_tail):
            raise TransportAborted("prepare interrupted while draining the prior output tail")

        with self._lock:
            if (
                operation_epoch != self._operation_epoch
                or operation_stop.is_set()
                or self._closed
                or self._closing
            ):
                raise TransportAborted(
                    "pulse prepare was superseded before hardware upload"
                )
            self._artifact = None
            self._artifact_digest = None
            self._total_points = total_points
            self._total_chunks = total_chunks
            self._scan_chunk_rows = scan_chunk_rows
            self._next_monotonic_chunk = 2
            self._bank_ready = 0b11
            self._scan_sweep = 0
            self._last_cursor = 0
            self._tail_seconds = artifact.max_configured_output_delay_ticks / ir.clock_hz
            self._drain_until = 0.0
            self._terminal_event.clear()
            self._terminal_error = None
            self._terminal = None
            self._terminal_observed_monotonic_ns = None
            self._status_sample_count = 0
            self._cursor_sample_count = 0
            self._underflow_observed = False
            self._state = "PREPARING"

        # Replacement cancellation and session close have already proved SAFE.
        # Repeating the full physical transition here added no new safety fact.
        if prior_state != "SAFE":
            self._drive_physical_safe(
                deadline=time.monotonic() + self.action_timeout
            )
        rows = list(artifact.wire_image.words)
        rows.append((CtrlWords.BANK_READY, 0b11))
        # Ordered BRAM writes and LOAD can share one transport batch.  Vivado still
        # executes each AXI transaction in order; only the host round trip is merged.
        rows.extend(
            (
                (CtrlWords.COMMAND, 0),
                (CtrlWords.COMMAND, CMD_LOAD),
            )
        )
        self.transport.write_words(tuple(rows), stop=operation_stop)
        if not self._wait_status(
            wait_mask=STATUS_LOADED,
            timeout=self.load_timeout,
            stop=operation_stop,
        ):
            status = self.transport.read_word(CtrlWords.STATUS, stop=operation_stop)
            raise RuntimeError(
                "pulse streamer did not report LOADED after artifact upload "
                f"(STATUS=0x{status:08X})"
            )
        with self._lock:
            if (
                operation_epoch != self._operation_epoch
                or operation_stop.is_set()
            ):
                raise TransportAborted(
                    "pulse prepare was superseded by interrupt-to-safe"
                )
            self._artifact = artifact
            self._artifact_digest = artifact_digest
            self._state = "PREPARED"

    def fire(self, artifact: CompiledPulseArtifact) -> None:
        """FIRE the exact prepared object after starting its terminal observer."""

        self._require_started()
        with self._lock:
            if self._state != "PREPARED" or artifact is not self._artifact:
                raise RuntimeError(
                    "FIRE artifact is not the exact immutable prepared artifact"
                )
            operation_epoch = self._operation_epoch
            self._state = "FIRING"
            self._terminal_event.clear()
            self._terminal_error = None
            self._terminal = None
            self._terminal_observed_monotonic_ns = None
            self._status_sample_count = 0
            self._cursor_sample_count = 0
            self._underflow_observed = False
        operation_stop = self._operation_stop
        self._start_worker(operation_epoch)
        try:
            # BANK_READY precedes FIRE in the same ordered batch.  This preserves
            # the hardware protocol and removes a redundant Vivado round trip.
            self.transport.write_words(
                (
                    (CtrlWords.BANK_READY, 0b11),
                    (CtrlWords.COMMAND, 0),
                    (CtrlWords.COMMAND, CMD_FIRE),
                ),
                stop=operation_stop,
            )
        except BaseException:
            self._cancel_worker_before_fire()
            with self._lock:
                if operation_epoch == self._operation_epoch:
                    self._state = "FAILED"
            raise
        superseded = False
        with self._lock:
            if (
                operation_epoch != self._operation_epoch
                or operation_stop.is_set()
            ):
                superseded = True
            else:
                self._state = "RUNNING"
        if superseded:
            self._cancel_worker_before_fire()
            raise TransportAborted(
                "pulse FIRE was superseded by interrupt-to-safe"
            )
        if self._fire_gate is not None:
            self._fire_gate.set()
        self.transport.record_diagnostic("fire_time", str(time.monotonic()))

    def await_completion(
        self,
        artifact: CompiledPulseArtifact,
        timeout: float | None = None,
    ) -> PulseBackendCompletion | None:
        """Wait only; the FIRE-owned worker is the sole terminal-status I/O owner."""

        self._require_started()
        with self._lock:
            if artifact is not self._artifact:
                raise RuntimeError("completion artifact differs from prepared artifact")
            if artifact.target_ir.repeat_forever:
                raise RuntimeError("continuous monitor has no logical terminal")
            operation_epoch = self._operation_epoch
            operation_stop = self._operation_stop
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        wait_seconds = None if deadline is None else max(0.0, deadline - time.monotonic())
        if not self._terminal_event.wait(wait_seconds):
            return None
        error = self._terminal_error
        if error is not None:
            raise error
        terminal = self._terminal
        observed_ns = self._terminal_observed_monotonic_ns
        if terminal is None or observed_ns is None:
            raise RuntimeError("streamer terminal evidence is absent")
        validate_terminal_for_artifact(terminal, artifact)
        drain_until = observed_ns / 1_000_000_000 + self._tail_seconds
        remaining = drain_until - time.monotonic()
        if remaining > 0:
            if deadline is not None and time.monotonic() + remaining > deadline:
                with self._lock:
                    if (
                        operation_epoch != self._operation_epoch
                        or operation_stop.is_set()
                    ):
                        raise TransportAborted(
                            "output-tail wait was superseded by interrupt-to-safe"
                        )
                    self._drain_until = drain_until
                return None
            if operation_stop.wait(remaining):
                raise TransportAborted(
                    "output-tail wait was interrupted by safe_state"
                )
        with self._lock:
            if (
                operation_epoch != self._operation_epoch
                or operation_stop.is_set()
            ):
                raise TransportAborted(
                    "output-tail completion was superseded by interrupt-to-safe"
                )
            self._drain_until = 0.0
        elapsed_ns = max(0, time.monotonic_ns() - observed_ns)
        tail = PostTerminalTailEvidence(
            terminal.fingerprint,
            POST_TERMINAL_TAIL_WAIT_RECIPE,
            artifact.max_configured_output_delay_ticks,
            artifact.target_ir.clock_hz,
            elapsed_ns,
        )
        return PulseBackendCompletion(terminal, tail)

    def wait_continuous_failure(
        self,
        artifact: CompiledPulseArtifact,
        timeout: float,
    ) -> str | None:
        """Wait for the existing observer to report a continuous-run failure."""

        wait_seconds = float(timeout)
        if not math.isfinite(wait_seconds) or wait_seconds <= 0.0:
            raise ValueError("continuous failure wait timeout must be positive")
        self._require_started()
        with self._lock:
            if artifact is not self._artifact:
                raise RuntimeError("continuous failure wait artifact differs")
            if not artifact.target_ir.repeat_forever:
                raise RuntimeError("continuous failure wait requires cyclic execution")
            if self._state not in {"FIRING", "RUNNING", "FAILED"}:
                raise RuntimeError(
                    f"continuous failure wait is invalid while session is {self._state}"
                )
            operation_epoch = self._operation_epoch
            terminal_event = self._terminal_event
        if not terminal_event.wait(wait_seconds):
            return None
        with self._lock:
            if operation_epoch != self._operation_epoch:
                return None
            error = self._terminal_error
            if error is None:
                return "continuous pulse observer stopped without a failure record"
            return f"{type(error).__name__}: {error}"

    def safe_state(self) -> None:
        """Abort immediately, invalidate the capability, and join the only I/O worker."""

        deadline = time.monotonic() + self.action_timeout
        if not self._safe_lock.acquire(timeout=self._remaining_seconds(deadline)):
            raise TimeoutError("pulse SAFE timed out waiting for the safety owner")
        try:
            self._require_started()
            with self._lock:
                if self._closed:
                    raise RuntimeError("pulse session is closed")
                if not self._layout_verified:
                    raise RuntimeError(
                        "pulse SAFE is forbidden before register layout verification"
                    )
                interrupted = self._state in {"FIRING", "RUNNING"}
                self._operation_epoch += 1
                safe_epoch = self._operation_epoch
                self._operation_stop.set()
                self._state = "INTERRUPTING"
                stop = self._worker_stop
                if stop is not None:
                    stop.set()
            failure: BaseException | None = None
            try:
                self._drive_physical_safe(deadline=deadline)
            except BaseException as error:
                failure = error
            try:
                self._stop_worker(timeout=self._remaining_seconds(deadline))
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(
                        "pulse worker did not terminate after the SAFE attempt: "
                        f"{type(error).__name__}: {error}"
                    )
            with self._lock:
                current = safe_epoch == self._operation_epoch
                if current:
                    self._artifact = None
                    self._artifact_digest = None
                    self._drain_until = 0.0
                    self._state = "SAFE" if failure is None else "SAFE_FAILED"
                    if interrupted and not self._terminal_event.is_set():
                        self._terminal_error = TransportAborted(
                            "pulse execution was interrupted by safe_state"
                        )
                        self._terminal_event.set()
            if failure is not None:
                raise failure
            if not current:
                raise TransportAborted("safe_state was superseded by a newer interrupt")
        finally:
            self._safe_lock.release()

    def request_interrupt(self) -> None:
        """Non-blocking cancellation edge used by the independent abort path."""

        with self._lock:
            self._operation_epoch += 1
            self._operation_stop.set()
            stop = self._worker_stop
            if stop is not None:
                stop.set()
            if self._state in {"PREPARING", "FIRING", "RUNNING"}:
                self._terminal_error = TransportAborted(
                    "pulse execution interrupt was requested"
                )
                self._terminal_event.set()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            terminal = self._terminal
            return {
                "schema": "zlc_pulse.DeployedStreamerSessionSnapshot",
                "transport": self.transport.transport_id,
                "geometry_fingerprint": build_fingerprint(self.params) & 0xFFFFFFFF,
                "clock_hz": self.clock_hz,
                "state": self._state,
                "prepared_artifact_digest": self._artifact_digest,
                "scan_points": self._total_points,
                "scan_chunks": self._total_chunks,
                "next_monotonic_scan_chunk": self._next_monotonic_chunk,
                "scan_sweep": self._scan_sweep,
                "last_confirmed_cursor": self._last_cursor,
                "cursor_sample_count": self._cursor_sample_count,
                "underflow_observed": self._underflow_observed,
                "terminal": (
                    None
                    if terminal is None
                    else hardware_terminal_evidence_to_tree(terminal)
                ),
            }

    def _command(
        self,
        command: int,
        *,
        wait_mask: int | None = None,
        timeout: float | None = None,
        stop: threading.Event | None = None,
    ) -> bool:
        effective = float(timeout) if timeout is not None else self.action_timeout
        deadline = time.monotonic() + max(0.0, effective)
        self.transport.write_words(
            (
                (CtrlWords.COMMAND, 0),
                (CtrlWords.COMMAND, int(command) & 0xF),
            ),
            stop=stop,
            deadline=deadline,
        )
        if wait_mask is None:
            return True
        return self._wait_status(
            wait_mask=wait_mask,
            timeout=self._remaining_seconds(deadline),
            stop=stop,
        )

    def _wait_status(
        self,
        *,
        wait_mask: int,
        timeout: float,
        stop: threading.Event | None,
    ) -> bool:
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            status = self.transport.read_word(
                CtrlWords.STATUS,
                stop=stop,
                deadline=deadline,
            )
            if status & STATUS_ERROR:
                raise RuntimeError("pulse streamer reported STATUS_ERROR")
            if (status & wait_mask) == wait_mask:
                return True
            if time.monotonic() >= deadline:
                return False
            if stop is not None:
                if stop.wait(0.01):
                    raise TransportAborted("pulse command wait was interrupted")
            else:
                time.sleep(0.01)

    def _enter_safe(self, *, deadline: float | None = None) -> None:
        """Issue SAFE and prove the frozen loader consumed it using existing readback."""

        # STATUS_ERROR is host-only in the frozen RTL.  Writing it before the
        # command creates an acknowledgement sentinel: STATUS can become zero
        # only after the loader consumes SAFE and asserts engine reset.
        absolute_deadline = (
            time.monotonic() + self.action_timeout
            if deadline is None
            else deadline
        )
        self.transport.write_words(
            (
                (CtrlWords.STATUS, STATUS_ERROR),
                (CtrlWords.COMMAND, 0),
                (CtrlWords.COMMAND, CMD_SAFE),
            ),
            deadline=absolute_deadline,
        )
        retry_at = time.monotonic() + min(
            0.05,
            max(0.001, self._remaining_seconds(absolute_deadline) / 2),
        )
        retried = False
        stable_zero = 0
        while time.monotonic() < absolute_deadline:
            status = self.transport.read_word(
                CtrlWords.STATUS,
                deadline=absolute_deadline,
            )
            stable_zero = stable_zero + 1 if status == 0 else 0
            if stable_zero >= 2:
                return
            if not retried and time.monotonic() >= retry_at:
                self.transport.write_words(
                    (
                        (CtrlWords.COMMAND, 0),
                        (CtrlWords.COMMAND, CMD_SAFE),
                    ),
                    deadline=absolute_deadline,
                )
                retried = True
            time.sleep(min(0.001, self.terminal_poll_interval))
        raise TimeoutError("pulse streamer did not acknowledge SAFE with stable STATUS=0")

    def _disable_clock_muxes(self, *, deadline: float) -> None:
        base = region_bases(self.params)["ctrl"] + CtrlWords.CLK_ENABLE
        self.transport.write_words(
            tuple((base + index, 0) for index in range(self.params.clk_enable_words)),
            deadline=deadline,
        )
        self._verify_clock_muxes_disabled(deadline=deadline)

    def _drive_physical_safe(self, *, deadline: float) -> None:
        """Execute the frozen streamer's proven three-phase SAFE protocol.

        These are hardware state transitions, not independent host writes that may
        be coalesced.  The first SAFE resets the engine, clock muxes are then
        disabled and read back, and the final SAFE proves the post-disable state.
        """

        self._enter_safe(deadline=deadline)
        self._disable_clock_muxes(deadline=deadline)
        self._enter_safe(deadline=deadline)

    def _read_words(
        self,
        word_offsets: Sequence[int],
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> tuple[int, ...]:
        reader = getattr(self.transport, "read_words", None)
        if callable(reader):
            return tuple(reader(tuple(word_offsets), stop=stop, deadline=deadline))
        return tuple(
            self.transport.read_word(address, stop=stop, deadline=deadline)
            for address in word_offsets
        )

    def _verify_clock_muxes_disabled(self, *, deadline: float) -> None:
        base = region_bases(self.params)["ctrl"] + CtrlWords.CLK_ENABLE
        values = tuple(
            self.transport.read_word(
                base + index,
                deadline=deadline,
            )
            for index in range(self.params.clk_enable_words)
        )
        if any(values):
            raise RuntimeError(
                "pulse SAFE could not verify that every live clock mux is disabled"
            )

    def _start_worker(self, operation_epoch: int) -> None:
        self._worker_stop = threading.Event()
        self._fire_gate = threading.Event()
        self._worker = threading.Thread(
            target=self._observer_worker,
            args=(operation_epoch,),
            name="zlc-pulse-state-observer",
            daemon=True,
        )
        self._worker.start()

    def _cancel_worker_before_fire(self) -> None:
        if self._worker_stop is not None:
            self._worker_stop.set()
        if self._fire_gate is not None:
            self._fire_gate.set()
        self._stop_worker()

    def _observer_worker(self, operation_epoch: int) -> None:
        stop = self._worker_stop
        gate = self._fire_gate
        if stop is None or gate is None:
            return
        gate.wait()
        if stop.is_set():
            return
        artifact = self._artifact
        continuous = bool(artifact is not None and artifact.target_ir.repeat_forever)
        streamed = self._total_chunks > 2
        poll_interval = (
            max(
                self.terminal_poll_interval,
                _CONTINUOUS_OBSERVER_INTERVAL_SECONDS,
            )
            if continuous and not streamed
            else self.terminal_poll_interval
        )
        previous_cursor = 0
        try:
            while not stop.is_set():
                status = self.transport.read_word(CtrlWords.STATUS, stop=stop)
                self._status_sample_count += 1
                if status & STATUS_ERROR:
                    raise RuntimeError("pulse streamer reported STATUS_ERROR")
                if status & STATUS_UNDERFLOW:
                    self._underflow_observed = True
                    raise RuntimeError(
                        "pulse streamer underflowed; seamless scan timing is invalid"
                    )
                if status & STATUS_DONE:
                    terminal = self._read_terminal_evidence(status, stop)
                    artifact = self._artifact
                    if artifact is None:
                        raise RuntimeError("terminal observer lost its compiled artifact")
                    validate_terminal_for_artifact(terminal, artifact)
                    observed_ns = time.monotonic_ns()
                    with self._lock:
                        if (
                            operation_epoch != self._operation_epoch
                            or stop.is_set()
                        ):
                            return
                        self._terminal = terminal
                        self._terminal_observed_monotonic_ns = observed_ns
                        self._drain_until = (
                            observed_ns / 1_000_000_000 + self._tail_seconds
                        )
                        self._state = "DONE"
                    self._terminal_event.set()
                    return

                if self._total_points:
                    cursor = self.transport.read_word(CtrlWords.CURSOR, stop=stop)
                    if not 0 <= cursor < self._total_points:
                        raise RuntimeError(
                            "pulse streamer cursor is outside the active scan table"
                        )
                    if continuous and cursor < previous_cursor:
                        self._scan_sweep += 1
                    previous_cursor = cursor
                    with self._lock:
                        if operation_epoch != self._operation_epoch or stop.is_set():
                            return
                        self._last_cursor = cursor
                        self._cursor_sample_count += 1
                    if streamed:
                        self._refill_freed_scan_banks(
                            artifact,
                            cursor,
                            continuous=continuous,
                            stop=stop,
                        )
                # A resident continuous pulse needs only low-rate progress sampling.
                # A streamed scan instead polls at the configured transport cadence:
                # this one worker is the sole post-FIRE status/cursor/refill owner.
                if stop.wait(poll_interval):
                    return
        except TransportAborted:
            return
        except BaseException as error:
            with self._lock:
                if operation_epoch != self._operation_epoch:
                    return
                self._terminal_error = error
                self._state = "FAILED"
            try:
                self._drive_physical_safe(
                    deadline=time.monotonic() + self.action_timeout
                )
            except BaseException as safety_error:
                error.add_note(
                    f"CMD_SAFE after worker failure also failed: {safety_error!r}"
                )
            self.transport.record_diagnostic("stream_failure", repr(error))
            self._terminal_event.set()

    def _refill_freed_scan_banks(
        self,
        artifact: CompiledPulseArtifact | None,
        cursor: int,
        *,
        continuous: bool,
        stop: threading.Event,
    ) -> None:
        """Keep the frozen RTL's ping-pong banks one chunk ahead of playback."""

        if artifact is None:
            raise RuntimeError("scan refill lost its compiled artifact")
        total_chunks = self._total_chunks
        if total_chunks <= 2:
            return
        engine_chunk = self._scan_sweep * total_chunks + (
            min(cursor, self._total_points - 1) // self.params.bank_size
        )
        while (
            (continuous or self._next_monotonic_chunk < total_chunks)
            and engine_chunk >= self._next_monotonic_chunk - 1
        ):
            monotonic_chunk = self._next_monotonic_chunk
            data_chunk = (
                monotonic_chunk % total_chunks
                if continuous
                else monotonic_chunk
            )
            bank = monotonic_chunk % 2
            unarmed = self._bank_ready & ~(1 << bank)
            words = self._scan_chunk_rows[data_chunk][bank]
            marker = (
                CtrlWords.BANK0_CHUNK
                if bank == 0
                else CtrlWords.BANK1_CHUNK
            )
            rearmed = unarmed | (1 << bank)
            deadline = time.monotonic() + self.action_timeout
            rewrite = getattr(self.transport, "rewrite_scan_bank", None)
            if callable(rewrite):
                rewrite(
                    unarmed_bank_ready=unarmed,
                    bank_words=words,
                    chunk_word=marker,
                    chunk_index=data_chunk,
                    rearmed_bank_ready=rearmed,
                    stop=stop,
                    deadline=deadline,
                )
            else:
                # Injectable model transports use the same ordered row contract.
                # Production transports provide rewrite_scan_bank() so their
                # acknowledgement granularity cannot re-arm a partial bank.
                self.transport.write_words(
                    (
                        (CtrlWords.BANK_READY, unarmed),
                        *words,
                        (marker, data_chunk),
                        (CtrlWords.BANK_READY, rearmed),
                    ),
                    stop=stop,
                    deadline=deadline,
                )
            committed_status, committed_cursor = self._read_words(
                (CtrlWords.STATUS, CtrlWords.CURSOR),
                stop=stop,
                deadline=deadline,
            )
            if committed_status & STATUS_ERROR:
                raise RuntimeError(
                    "pulse streamer reported STATUS_ERROR during scan-bank refill"
                )
            if committed_status & STATUS_UNDERFLOW:
                self._underflow_observed = True
                raise RuntimeError(
                    "pulse streamer underflowed during scan-bank refill; "
                    "seamless scan timing is invalid"
                )
            if not 0 <= committed_cursor < self._total_points:
                raise RuntimeError(
                    "pulse streamer cursor is outside the active scan table "
                    "after scan-bank refill"
                )
            if (
                committed_cursor // self.params.bank_size
                != cursor // self.params.bank_size
            ):
                raise RuntimeError(
                    "pulse streamer crossed a scan-bank boundary while its next "
                    "bank refill was in flight; seamless timing cannot be proven"
                )
            self._bank_ready = rearmed
            self._next_monotonic_chunk += 1

    def _read_terminal_evidence(
        self,
        first_status: int,
        stop: threading.Event,
    ) -> PulseHardwareTerminalEvidence:
        artifact = self._artifact
        if artifact is None:
            raise RuntimeError("terminal observer has no compiled artifact")
        if artifact.execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
            cursor_first = self.transport.read_word(CtrlWords.CURSOR, stop=stop)
            second_status = self.transport.read_word(CtrlWords.STATUS, stop=stop)
            self._status_sample_count += 1
            cursor_second = self.transport.read_word(CtrlWords.CURSOR, stop=stop)
            with self._lock:
                self._last_cursor = cursor_second
                self._cursor_sample_count += 2
            return AutonomousTableTerminalEvidence(
                AUTONOMOUS_TABLE_READ_RECIPE,
                self.transport.transport_id,
                first_status,
                cursor_first,
                second_status,
                cursor_second,
                self._underflow_observed,
                self._status_sample_count,
            )
        second_status = self.transport.read_word(CtrlWords.STATUS, stop=stop)
        self._status_sample_count += 1
        return StaticOnceTerminalEvidence(
            STATIC_STATUS_READ_RECIPE,
            self.transport.transport_id,
            first_status,
            second_status,
            self._underflow_observed,
            self._status_sample_count,
        )

    def _stop_worker(self, *, timeout: float | None = None) -> None:
        stop = self._worker_stop
        gate = self._fire_gate
        worker = self._worker
        if stop is not None:
            stop.set()
        if gate is not None:
            gate.set()
        if worker is not None and worker is not threading.current_thread():
            join_timeout = self.action_timeout + 1.0 if timeout is None else timeout
            worker.join(timeout=max(0.0, join_timeout))
            if worker.is_alive():
                raise RuntimeError("pulse terminal observer did not terminate")
        self._worker = None
        self._worker_stop = None
        self._fire_gate = None

    def _require_started(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("pulse session is closed")
            if not self._started:
                raise RuntimeError("pulse session has not acquired its device lease")

    @staticmethod
    def _remaining_seconds(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("pulse safe-state deadline expired")
        return remaining


__all__ = [
    "DeployedStreamerSession",
    "RegisterTransport",
    "TransportAborted",
]
