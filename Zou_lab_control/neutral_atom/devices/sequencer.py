"""Sequencer devices and the runtime pulse-table service boundary."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from Zou_lab_control._clock import default_clock_hz as _default_clock_hz
from Zou_lab_control._streamer_geometry import DEFAULT_COEFF_FRAC_BITS
from .._serialization import (
    require_array,
    require_bool,
    require_exact_fields,
    require_int,
    require_number,
    require_object,
    require_string,
)
from ..core.analysis import nonnegative_float, positive_int
from .base import SequencerDevice
from ..ports import PortCatalog, coerce_port_catalog
from ..timing import (
    PulseSequence,
    PulseTableState,
    affine_coeffs,
    channel_names,
    positive_float,
    slot_var,
)
from ..timing.pulse_table import (
    UNITS_TO_NS,
    analog_bus_ticks as _pulse_table_analog_bus_ticks,
    _analog_bus_value_at_tick as _pulse_table_analog_bus_value_at_tick,
    bus_zero_code,
    snap_scan_table as _snap_scan_table,
    slot_ref_index as _parse_slot_ref_index,
)


DEFAULT_RUNTIME_CLOCK_HZ = _default_clock_hz()












# Serialized RuntimeSequenceProgram schema version -- ONE source, written by to_dict AND checked by
# from_dict (#G4) so a future schema bump fails fast with a rebuild message instead of mis-decoding.
_RUNTIME_PROGRAM_VERSION = 4




#: The idle scan-progress reading -- no scan running.  SINGLE SOURCE for the dict shape every
#: SequencerDevice.scan_progress() returns, so the GUI poll + the virtual/real backends agree.
SCAN_PROGRESS_IDLE: dict[str, object] = {
    "scanning": False, "point": 0, "n_points": 0, "sweep": 0, "n_repeats": 0,
}

#: The ONE deadlock-guard message for "wait forever on a repeat_forever program" -- shared by the
#: service's wait_done and the virtual backend's real-time override, so the two can never drift.
WAIT_FOREVER_MESSAGE = ("sequencer wait_done cannot wait forever for a repeat_forever program; "
                        "pass a timeout or stop the pulse.")















# The compile math moved to zlc_neutral_atom.timing.runtime_compiler (the names
# below ARE the moved objects); this module keeps only the legacy transport.
from zlc_neutral_atom.timing.runtime_compiler import (  # noqa: F401,E402
    RuntimeBusDelay,
    RuntimeBusSegment,
    RuntimeSequenceProgram,
    _affine_add,
    _affine_expr,
    _affine_row_index,
    _apply_affine_ticks,
    _channel_delays_list,
    _check_unrolled_edge_budget,
    _clk_enable_mask_for_channels,
    _dedupe_same_tick_edges,
    _ensure_final_off_edge,
    _fold_global_delay_shift,
    _insert_mask_edge_at_tick,
    _is_plain_number,
    _plain_rpc_payload,
    _pulse_table_affine_loop_metadata,
    _pulse_table_affine_period_starts,
    _pulse_table_affine_rows,
    _pulse_table_bus_delay_steps,
    _pulse_table_bus_segments,
    _pulse_table_edge_table,
    _pulse_table_effective_duration_ticks,
    _pulse_table_has_analog_activity,
    _pulse_table_has_any_delay,
    _pulse_table_has_delays,
    _resolve_hardware_catalog,
    _slot_ref_index,
    _stable_affine_groups,
    _time_ns_to_ticks,
    _time_to_ticks,
    compile_pulse_table_runtime_program,
    compile_pulse_table_scan_runtime_program,
    compile_runtime_program,
    compile_runtime_program_for_payload,
    decode_wire_payload,
    encode_wire_payload,
    scan_progress_fields,
    sequence_from_payload,
    timing_from_payload,
    timing_payload_to_dict,
)

class SequencerService:
    """Stateful service that mirrors the final FPGA runtime protocol.

    The same object can run in-process for tests, or be exposed over RPyC on
    the FPGA/Vivado computer.  Hardware-specific callbacks can be attached
    later without changing the client-side ``SequencerDevice`` contract.
    """

    def __init__(
        self,
        *,
        channels: Sequence[str] | None = None,
        port_catalog: PortCatalog | Mapping[str, object] | None = None,
        clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
        prepare_callback: Callable[[RuntimeSequenceProgram], None] | None = None,
        fire_callback: Callable[[RuntimeSequenceProgram], None] | None = None,
        wait_done_callback: Callable[[RuntimeSequenceProgram, float | None], bool] | None = None,
        safe_state_callback: Callable[[], None] | None = None,
        scan_progress_callback: Callable[[], dict] | None = None,
        sleep_scale: float = 0.0,
        cache_prepared: bool = True,
    ):
        self.port_catalog = coerce_port_catalog(port_catalog, channels=channels)
        self.channels = list(self.port_catalog.raw_lanes)
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        self.prepare_callback = prepare_callback
        self.fire_callback = fire_callback
        self.wait_done_callback = wait_done_callback
        self.safe_state_callback = safe_state_callback
        self.scan_progress_callback = scan_progress_callback
        # SCAN PROGRESS IS A DEVICE-TRUTH READING, never a GUI-local timer.  A backend with a real
        # scan-point source wires ``scan_progress_callback`` -- the FPGA cursor (AXI) or the virtual
        # backend's real-time-paced reading (``VirtualSequencer._scan_progress``) -- and the reading
        # always comes from there.  WITHOUT such a source (the command backend on real hardware,
        # whose fire is a fire-and-forget lab command with no cursor to read) the service does NOT
        # know where the engine's scan is, so it reports the HONEST reading: point 0 while a scan is
        # loaded and running, NEVER an advancing count -- that is what stops a real command-backed
        # device from showing "progress advancing" while nothing is physically streaming (the
        # user-reported "progress climbs but the device is idle").
        self.sleep_scale = nonnegative_float(sleep_scale, "sleep_scale")
        self.cache_prepared = bool(cache_prepared)
        self._lock = threading.RLock()
        self.prepared_program: RuntimeSequenceProgram | None = None
        # The SOURCE payload (PulseTableState/PulseSequence dict) of the last successful
        # prepare, as a JSON string.  This is what "sync to device" pulls: the GUI (or any
        # raw-API caller) can reconstruct the editable state that is actually applied on
        # the device -- the compiled program alone cannot be edited back.
        self.last_payload_json: str | None = None
        self.state = "idle"
        self.history: list[dict[str, object]] = []
        # Pure-software scan-progress derivation (no FPGA cursor): the wall-clock fire instant of the
        # running STREAMED scan + a done latch for a finite K-sweep scan (see scan_progress).
        self._scan_fire_time: float = 0.0
        self._scan_done: bool = False

    def prepare(self, sequence_payload) -> dict[str, object]:
        timing_payload = timing_from_payload(sequence_payload)
        try:
            program = compile_runtime_program_for_payload(
                timing_payload,
                channels=self.channels,
                clock_hz=self.clock_hz,
                port_catalog=self.port_catalog,
            )
            # Backstop: validate the compiled program against the FPGA geometry / delay-line
            # depth REGARDLESS of which backend's prepare_callback runs.  An AXI backend
            # validates with its own params, but a mock/no-op backend would otherwise cache an
            # invalid program (e.g. channel_delays beyond the event-FIFO capacity).  Local
            # import breaks the fpga_pulse_streamer <-> sequencer import cycle.
            from .fpga_pulse_streamer import validate_pulse_streamer_program
            # Scan points STREAM through the 2-bank window, so their count is UNBOUNDED --
            # pass the program's own count (like the AXI backend does), never the resident
            # window size, and sample the per-point monotonicity sweep so a huge scan does
            # not stall prepare().
            validate_pulse_streamer_program(
                program,
                channel_count=len(self.channels),
                max_scan_points=max(1, len(program.scan_points or [])),
                max_validated_scan_points=4096,
            )
        except Exception:
            # A REJECTED program (e.g. a channel delay over the event-FIFO capacity) raises
            # HERE, before prepare_callback -- so the backend's own stop/safe never runs.  If
            # a prior repeat_forever STREAMED scan is still being fed by the backend's
            # background refill thread, leaving it alive lets it keep using the single Vivado
            # Tcl session and wedge the NEXT (good) On Pulse.  Best-effort safe the backend so
            # the orphaned stream is stopped; never mask the original validation error.
            prev = self.prepared_program
            streamed = bool(
                prev is not None
                and getattr(prev, "repeat_forever", False)
                and getattr(prev, "scan_points", None)
            )
            if streamed and self.safe_state_callback is not None:
                try:
                    self.safe_state_callback()
                except Exception:
                    pass
            raise
        with self._lock:
            cached = (
                self.cache_prepared
                and self.prepared_program is not None
                and self.prepared_program.sequence_id == program.sequence_id
            )
            if self.prepare_callback is not None and not cached:
                self.prepare_callback(program)
            self.prepared_program = program
            self._record_source_payload(sequence_payload)
            self.state = "prepared"
            self.history.append(
                {
                    "action": "prepare",
                    "sequence_id": program.sequence_id,
                    "sequence": program.sequence_name,
                    "duration": program.duration,
                    "cached": cached,
                }
            )
        return program.to_dict()

    def _record_source_payload(self, sequence_payload) -> None:
        """Record the exact editable source applied to this sequencer.

        ``PulseTableState.to_sequence`` carries its source document explicitly;
        consuming it here avoids the old lossy edge decompile that erased DAC
        ports, slots and scan code.  A genuinely low-level PulseSequence has no
        such document, so its fallback table is built against this service's
        PortCatalog: even that honest best-effort view preserves logical DAC
        rows rather than exposing their raw bit lanes.
        """
        step = 1e9 / float(self.clock_hz)
        timing = timing_from_payload(sequence_payload)   # parses str / dict / object
        if isinstance(timing, PulseTableState):
            table = timing.snapped(time_step_ns=step)
        elif timing.source_table is not None:
            table = PulseTableState.from_dict(timing.source_table).snapped(time_step_ns=step)
        else:
            table = PulseTableState.from_sequence(
                timing, port_catalog=self.port_catalog, clock_hz=self.clock_hz)
        self.last_payload_json = json.dumps(table.to_dict())

    def fire(self, sequence_payload=None) -> dict[str, object]:
        with self._lock:
            program = self._require_prepared()
            if sequence_payload is not None:
                requested = compile_runtime_program_for_payload(
                    timing_from_payload(sequence_payload),
                    channels=self.channels,
                    clock_hz=self.clock_hz,
                    port_catalog=self.port_catalog,
                )
                if requested.sequence_id != program.sequence_id:
                    raise RuntimeError("fire(sequence) does not match the prepared runtime program.")
                # fire() with an explicit sequence (a Task / API caller) records it too, so a
                # sync AFTER the task started reflects what is RUNNING (prepare alone used to be
                # the only recorder -> a fired-but-not-prepared-here sequence left a stale table).
                self._record_source_payload(sequence_payload)
            if self.fire_callback is not None:
                self.fire_callback(program)
            self.state = "running"
            # Stamp the fire instant for a STREAMED scan so scan_progress() can derive the played-point
            # count from the wall clock when no hardware cursor callback is wired (pure software).  The
            # gate is "this is a streamed scan" = scan_points present, NOT repeat_forever -- a finite
            # single-pass scan (repeat_forever=False) streams its points too and must show progress.
            if getattr(program, "scan_points", None):
                self._scan_fire_time = time.monotonic()
                self._scan_done = False
            self.history.append({"action": "fire", "sequence_id": program.sequence_id, "sequence": program.sequence_name})
            return program.to_dict()

    def wait_done(self, timeout: float | None = None, *, stop=None) -> bool:
        with self._lock:
            program = self._require_prepared()
        if program.repeat_forever and timeout is None:
            raise RuntimeError(WAIT_FOREVER_MESSAGE)
        if self.wait_done_callback is not None:
            ok = bool(self.wait_done_callback(program, timeout))
        elif program.repeat_forever:
            ok = False
        else:
            delay = program.duration * self.sleep_scale
            if timeout is not None and delay > float(timeout):
                ok = False
            else:
                if delay > 0:
                    SequencerDevice._sleep_interruptible(delay, stop)   # cooperatively cancellable, like settle
                ok = not (stop is not None and stop.is_set())           # cancelled mid-wait -> not done
        with self._lock:
            self.state = "done" if ok else "timeout"
            self.history.append({"action": "wait_done", "sequence_id": program.sequence_id, "ok": ok})
        return ok

    def settle(self, seconds: float, *, stop=None) -> None:
        """Idle ``seconds`` between software-stepped fires, scaled by THIS service's
        ``sleep_scale`` -- the ONE home of the "a virtual settle fast-forwards with
        sleep_scale" rule (``sleep_scale=0`` under pytest -> instant), which both composing
        backends (e.g. ``VirtualSequencer``) delegate to.  ``stop`` keeps
        the wait cooperatively cancellable via the shared interruptible sleep
        (``SequencerDevice._sleep_interruptible`` -- not a second sleep loop)."""
        SequencerDevice._sleep_interruptible(float(seconds) * float(self.sleep_scale), stop)

    def abort(self) -> None:
        with self._lock:
            self.prepared_program = None
            self.state = "aborted"
            self.history.append({"action": "abort", "invalidated": True})
        if self.safe_state_callback is not None:
            self.safe_state_callback()

    def set_safe_state(self) -> None:
        with self._lock:
            self.prepared_program = None
            self.state = "safe"
            self.history.append({"action": "safe", "invalidated": True})
        if self.safe_state_callback is not None:
            self.safe_state_callback()

    def scan_progress(self) -> dict:
        """Where the running scan is now -- a DEVICE-TRUTH reading, never a GUI-local timer.

        A backend that can report its real scan position wires ``scan_progress_callback`` (the FPGA
        cursor on the AXI backend, the real-time simulator on the virtual backend); the reading then
        comes ENTIRELY from there, so it advances if and only if the device actually plays points.

        WITHOUT such a source (the command backend on real hardware -- a fire-and-forget lab command
        with no cursor) the service CANNOT know the engine's scan position, so it reports the honest
        reading: a loaded, running scan sits at ``point 0`` and does NOT advance.  It NEVER derives a
        climbing count from the wall clock -- that would claim the device is sweeping when nothing may
        be streaming (the reported "progress climbs but the device is idle")."""
        if self.scan_progress_callback is not None:
            return dict(self.scan_progress_callback())
        program = self.prepared_program
        # A streamed scan reports progress whether it is CYCLIC (repeat_forever) or a FINITE single-pass
        # one (repeat_forever=False) -- both stream their scan_points -- so the gate is scan_points, NOT
        # the cyclic flag (which is what wrongly idled a single-pass scan's readout before).
        if (self.state != "running" or program is None
                or not getattr(program, "scan_points", None)):
            return dict(SCAN_PROGRESS_IDLE)              # nothing streaming -> no scan position
        n = len(program.scan_points)
        k = max(0, int(getattr(program, "scan_repeats", 0)))
        if self._scan_done:                              # finite scan already played its K sweeps -> latched done
            return scan_progress_fields(max(1, k) * n, n, k)
        # No real cursor: report the honest static reading (running at the start of the scan), so a
        # device with an un-readable position never fabricates a climbing count while nothing is
        # confirmed streaming.
        return scan_progress_fields(0, n, k)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "type": type(self).__name__,
                "raw_channels": list(self.port_catalog.raw_lanes),
                "port_catalog": self.port_catalog.to_dict(),
                "port_catalog_fingerprint": self.port_catalog.fingerprint,
                "clock_hz": self.clock_hz,
                "state": self.state,
                "cache_prepared": self.cache_prepared,
                "prepared_program": None if self.prepared_program is None else self.prepared_program.to_dict(),
                # source payload of the last prepare (JSON string) -- the sync-to-device
                # handle: PulseTableState.from_dict(json.loads(...)) reconstructs the
                # editable state that is actually applied on the device.
                "last_payload_json": self.last_payload_json,
                "history_length": len(self.history),
            }

    def _require_prepared(self) -> RuntimeSequenceProgram:
        if self.prepared_program is None:
            raise RuntimeError("sequencer service has no prepared sequence.")
        return self.prepared_program


class ManualSequencer(SequencerDevice):
    """Sequencer adapter for first-light tests with a manually started FPGA.

    ``fire`` intentionally does not drive hardware.  It records that the camera
    is armed and that the operator or an external free-running FPGA must provide
    the trigger pulses before the qCMOS timeout expires.
    """

    def __init__(
        self,
        *,
        channels: Sequence[str] | None = None,
        port_catalog: PortCatalog | Mapping[str, object] | None = None,
        clock_hz: float = DEFAULT_RUNTIME_CLOCK_HZ,
        message: str | None = None,
    ):
        self.port_catalog = coerce_port_catalog(port_catalog, channels=channels)
        self.channels = list(self.port_catalog.raw_lanes)
        self.clock_hz = positive_float(clock_hz, "clock_hz")
        self.message = message or "Camera is armed. Start the FPGA/manual trigger sequence now."
        self.prepared_sequence: PulseSequence | PulseTableState | None = None
        self.state = "idle"
        self.history: list[dict[str, object]] = []

    def prepare(self, sequence: PulseSequence | PulseTableState) -> RuntimeSequenceProgram:
        program = compile_runtime_program_for_payload(
            sequence,
            channels=self.channels,
            clock_hz=self.clock_hz,
            port_catalog=self.port_catalog,
        )
        self.prepared_sequence = sequence
        self.state = "prepared"
        self.history.append({"action": "prepare", "sequence_id": program.sequence_id})
        return program

    def fire(self, sequence: PulseSequence | PulseTableState | None = None) -> None:
        if self.prepared_sequence is None:
            raise RuntimeError("ManualSequencer.fire() called before prepare().")
        if sequence is not None and sequence is not self.prepared_sequence:
            raise RuntimeError("ManualSequencer.fire() received a sequence that was not prepared.")
        self.state = "manual_trigger_wait"
        self.history.append({"action": "fire_manual", "message": self.message})
        print(self.message)

    def wait_done(self, timeout: float | None = None, *, stop=None) -> bool:
        self.state = "unknown_done"
        self.history.append({"action": "wait_done_manual", "timeout": timeout})
        return True

    def abort(self) -> None:
        self.state = "aborted"
        self.history.append({"action": "abort"})

    def set_safe_state(self) -> None:
        self.state = "safe_requested"
        self.history.append({"action": "safe"})

    def snapshot(self) -> dict[str, object]:
        out = super().snapshot()          # the ``type`` key has ONE producer: BaseDevice.snapshot
        out.update({
            "raw_channels": list(self.port_catalog.raw_lanes),
            "port_catalog": self.port_catalog.to_dict(),
            "port_catalog_fingerprint": self.port_catalog.fingerprint,
            "clock_hz": self.clock_hz,
            "state": self.state,
            "prepared": self.prepared_sequence is not None,
            "history_length": len(self.history),
        })
        return out


class RemoteSequencer(SequencerDevice):
    """RPyC client-side sequencer proxy for the FPGA/Vivado computer."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        ssl: bool = False,
        ca_certs: str | None = None,
        connect_on_init: bool = False,
    ):
        self.host = str(host).strip()
        if self.host in {"", "0.0.0.0", "::"}:
            raise ValueError("RemoteSequencer host must be the server address reachable from the control computer.")
        self.port = int(port)
        # A disconnected remote client has no hardware facts.  Topology and clock
        # are bound atomically from the server snapshot in ``open()``; accepting
        # either as a constructor input would create a second, drift-prone truth.
        self.port_catalog: PortCatalog | None = None
        self.channels: list[str] = []
        self.clock_hz: float | None = None
        self.ssl = bool(ssl)
        self.ca_certs = ca_certs
        self._conn = None
        self._last_program: RuntimeSequenceProgram | None = None
        if connect_on_init:
            self.open()

    # ------------------------------------------------------------------ config schema (self-describing)
    @classmethod
    def config_params(cls):
        """Remote connection form; hardware facts come from the server snapshot.

        ``port`` defaults to the server's own serve default (one source).  The form
        contains connection parameters only: no client-side topology or clock copy.
        """
        import inspect as _inspect
        from dataclasses import replace

        from .base import config_params_from_signature

        serve_port = int(_inspect.signature(serve_runtime_sequencer).parameters["port"].default)
        rows = []
        for decl in config_params_from_signature(cls):
            if decl.key == "host":
                decl = replace(decl, tooltip="FPGA/Vivado computer address reachable from this "
                                             "control computer (never 0.0.0.0).")
            elif decl.key == "port":
                decl = replace(decl, default=serve_port, required=False)
            rows.append(decl)
        return tuple(rows)

    def open(self) -> "RemoteSequencer":
        if self._conn is not None:
            if not getattr(self._conn, "closed", False):
                return self
            # The link died (server restart / network drop): a dead connection object
            # would otherwise be returned FOREVER and every call would keep raising
            # EOFError until the user restarted the GUI/notebook too.  Drop it and
            # reconnect transparently on this call.
            self.close()
        try:
            import rpyc
        except ImportError as exc:  # pragma: no cover - depends on lab install
            raise RuntimeError("RemoteSequencer requires `rpyc`. Install it on the control computer.") from exc
        # ONE client connection policy, shared by both transports: pickle allowed (the payloads
        # are trusted lab JSON/arrays) and a FINITE backstop timeout that is GENEROUS -- it must
        # exceed the longest server-side action (e.g. wait_done on a big finite scan) so a wedged
        # server cannot block the caller (the GUI worker thread) forever, while prepare/fire that
        # return in seconds are unaffected.  The SSL transport carries the SAME config, so a
        # secured lab's long scans do not silently hit rpyc's 30 s default and AsyncResultTimeout.
        rpyc_config = {"allow_pickle": True, "sync_request_timeout": 3600.0}
        if self.ssl:
            self._conn = rpyc.utils.classic.ssl_connect(
                host=self.host, port=self.port, ca_certs=self.ca_certs, config=rpyc_config)
        else:
            self._conn = rpyc.connect(self.host, self.port, config=rpyc_config)
        try:
            # Parse both facts before publishing either one on the client.  A partial
            # or stale server cannot leave a half-bound RemoteSequencer behind.
            snap = self._conn.root.snapshot()
            remote_catalog = PortCatalog.from_dict(snap["port_catalog"])
            remote_clock_hz = positive_float(snap["clock_hz"], "server snapshot clock_hz")
        except Exception as exc:
            self.close()
            raise RuntimeError(
                "the sequencer server snapshot must contain a complete PortCatalog and a "
                "positive clock_hz; update/restart the server before connecting this client"
            ) from exc
        self.port_catalog = remote_catalog
        self.channels = list(remote_catalog.raw_lanes)
        self.clock_hz = remote_clock_hz
        return self

    @property
    def is_open(self) -> bool:
        connection = self._conn
        return bool(
            connection is not None
            and not getattr(connection, "closed", False)
            and self.port_catalog is not None
            and self.clock_hz is not None
        )

    def _require_live_connection(self):
        if not self.is_open:
            raise RuntimeError(
                "RemoteSequencer connection is not established; reconnect only through "
                "the installation ConnectionEstablishmentClaim/RecoveryClaim"
            )
        return self._conn

    def prepare(self, sequence: PulseSequence | PulseTableState) -> RuntimeSequenceProgram:
        connection = self._require_live_connection()
        step = 1e9 / float(self.clock_hz)
        head, blobs = encode_wire_payload(
            timing_payload_to_dict(sequence, time_step_ns=step), _WIRE_ARRAY_FIELDS_PAYLOAD)
        prog_head, prog_blobs = connection.root.prepare(head, blobs)
        self._last_program = RuntimeSequenceProgram.from_dict(decode_wire_payload(prog_head, prog_blobs))
        if self._last_program.port_catalog_fingerprint != self.port_catalog.fingerprint:
            raise RuntimeError(
                "remote compiler returned a program for a different PortCatalog: "
                f"program={self._last_program.port_catalog_fingerprint}, "
                f"connected={self.port_catalog.fingerprint}")
        return self._last_program

    def fire(self, sequence: PulseSequence | PulseTableState | None = None) -> None:
        connection = self._require_live_connection()
        step = 1e9 / float(self.clock_hz)
        if sequence is None:
            connection.root.fire(None, None)
        else:
            head, blobs = encode_wire_payload(
                timing_payload_to_dict(sequence, time_step_ns=step), _WIRE_ARRAY_FIELDS_PAYLOAD)
            connection.root.fire(head, blobs)

    def wait_done(self, timeout: float | None = None, *, stop=None) -> bool:
        connection = self._require_live_connection()
        # ``stop`` is accepted for the uniform sequencer contract; the actual wait is server-side (the AXI
        # backend's wait_done_callback), bounded by ``timeout`` + the RPyC transport, so cancellation across
        # the RPyC boundary is not threaded here -- a Stop lands when the bounded remote call returns.
        return bool(connection.root.wait_done(timeout))

    def scan_progress(self) -> dict:
        # Lightweight poll for the GUI's live scan progress.  This is a required
        # server capability: pretending an old/incompatible server is idle hides
        # both version skew and a running experiment.
        connection = self._require_live_connection()
        return dict(connection.root.scan_progress())

    def abort(self) -> None:
        self._require_live_connection().root.abort()

    def set_safe_state(self) -> None:
        self._require_live_connection().root.set_safe_state()

    def snapshot(self) -> dict[str, object]:
        out = super().snapshot()          # the ``type`` key has ONE producer: BaseDevice.snapshot
        catalog = self.port_catalog
        out.update({
            "host": self.host,
            "port": self.port,
            "raw_channels": [] if catalog is None else list(catalog.raw_lanes),
            "port_catalog": None if catalog is None else catalog.to_dict(),
            "port_catalog_fingerprint": None if catalog is None else catalog.fingerprint,
            "clock_hz": self.clock_hz,
            "connected": self._conn is not None,
            "last_program": None if self._last_program is None else self._last_program.to_dict(),
        })
        if self._conn is not None:
            try:
                remote = dict(self._conn.root.snapshot())
                out["remote"] = remote
                # flatten the sync-to-device handles (str() materialises rpyc netrefs)
                lpj = remote.get("last_payload_json")
                out["last_payload_json"] = None if lpj is None else str(lpj)
                out["state"] = str(remote.get("state", ""))
            except Exception as exc:
                out["remote_error"] = str(exc)
        return out

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
        self.port_catalog = None
        self.channels = []
        self.clock_hz = None


class PulseController:
    """Notebook helper that binds a pulse payload to a sequencer.

    It keeps readout scans terse.  A single-point scan sets one slot per shot::

        pulse.set_slot("exposure", 200)  # ns into a semantic duration slot
        pulse.on_pulse()

    A multi-point hardware scan uploads a whole scan table once::

        pulse.set_scan_table([[10], [20], [30]]).on_pulse()

    The controller owns no hardware; it delegates to the supplied local or
    remote ``SequencerDevice``.
    """

    def __init__(self, sequencer: SequencerDevice, pulse: PulseSequence | PulseTableState):
        self._sequencer = sequencer
        self.pulse = pulse
        self.scan_table = [list(row) for row in (getattr(pulse, "scan_table", []) or [])]
        self.slots: dict[str, float] = {}
        self.last_program: RuntimeSequenceProgram | None = None

    def set_slot(self, key: str, value: float) -> "PulseController":
        """Set one semantic scan-slot value for a single-shot resolve.

        Time slots take ns.  DAC slots take the SIGNED user value (0 = true 0 V,
        -2^(B-1)..+2^(B-1)-1); the offset-binary wire code is produced by the
        compiler -- never pass a raw 0..1023 code here.  Positional ``sN``
        tokens and integer columns are compiler internals and are rejected."""

        if not isinstance(key, str):
            raise TypeError("scan slot must be addressed by its semantic ScanSlot.name")
        if not isinstance(self.pulse, PulseTableState):
            raise TypeError("only PulseTableState exposes semantic scan slots")
        name = str(key)
        if name not in self.pulse.scan_names:
            raise ValueError(
                f"unknown semantic scan slot {name!r}; available: {self.pulse.scan_names}")
        self.slots[name] = float(value)
        return self

    def set_time(self, value_ns: float) -> "PulseController":
        """Set the first duration scan slot (in ns) for the next shot."""

        name = self.pulse.primary_time_slot() if isinstance(self.pulse, PulseTableState) else None
        if name is None:
            raise TypeError("pulse has no duration scan slot; bind one via the GUI scan dot or state.bind_field(...).")
        return self.set_slot(name, value_ns)

    def set_scan_table(self, rows: Sequence[Sequence[float]] | None) -> "PulseController":
        # Accept a NumPy array as well as a list-of-lists: `rows or []` / `if rows:` raise
        # "truth value of an array is ambiguous" on an ndarray.  A 1-D array is read as a
        # COLUMN of points (N x 1), matching load_scan_table's single-slot convention.
        if rows is None:
            self.scan_table = []
            return self
        import numpy as np

        array = np.asarray(rows, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("scan_table must be a 1-D or 2-D array.")
        self.scan_table = [[float(v) for v in row] for row in array]
        return self

    def frame_sequence(
        self,
        frames: int,
        *,
        time_ns: float | None = None,
        slots: Mapping[str, float] | None = None,
        trigger_channels: Sequence[str] | None = None,
    ) -> PulseSequence:
        """Resolve the bound pulse at the requested ``time_ns``/``slots`` and return it as a finite
        ``PulseSequence`` for one scan point.

        The pulse carries its OWN camera triggers (a single-image readout pulse has one; a
        release-recapture bracket has two with the trap-off period between them), so this NEVER
        expands a one-trigger pulse into N copies -- the measurement reads ``frames`` capture windows
        from the sequence the pulse already defines (a multi-trigger bracket is ONE loading read
        twice; a one-trigger pulse read N times is N independent reloads, decided per frame by the
        atom device).  ``frames``/``trigger_channels`` are accepted for the scan-engine + session
        adapter contract but no longer drive any 1->N rewrite -- which channel gates a frame is the
        CAMERA's property, parsed downstream from its ``capture_trigger_channels``."""

        positive_int(frames, "frames")
        merged = dict(slots or {})
        if time_ns is not None:
            name = self.pulse.primary_time_slot() if isinstance(self.pulse, PulseTableState) else None
            if name is None:
                raise TypeError("pulse has no duration/delay scan slot to set from time_ns.")
            merged[name] = float(time_ns)
        payload = self.payload(slots=merged, scan_table=[], repeat_forever=False)
        if isinstance(payload, PulseTableState):
            ref_slots = payload._reference_slots()
            sequence = payload.to_sequence(
                slots=payload._semantic_values_from_compiler(ref_slots),
                time_step_ns=payload.time_step_ns, expand_repeat=False)
            # ``to_sequence`` returns only the pulse EDGES, so a trailing all-low period (a gap the
            # table defines AFTER the last edge) is invisible in the edge list -- its duration would
            # be dropped, shortening the fired program by that tail.  The program must run the WHOLE
            # cycle the table defines (the edges PLUS any trailing gap), so pin the sequence period to
            # the table's ONE-CYCLE duration.  Sum the RAW periods (NOT expanded_periods): an inner
            # finite-repeat bracket is the streamer's OWN affine loop, not a frame the camera reads,
            # so unrolling it here would inflate the cycle; ``to_sequence(expand_repeat=False)`` keeps
            # the same single-cycle view.  With no trailing gap this is the no-op identity.
            step_ns = payload.time_step_ns
            cycle_steps = sum(period.duration_steps(slots=ref_slots, time_step_ns=step_ns)
                              for period in payload.periods)
            cycle_s = cycle_steps * step_ns * 1e-9
            if cycle_s > sequence.base_duration + 1e-15:
                sequence = sequence.repeated(1, period=cycle_s)
            return sequence
        return payload

    def payload(
        self,
        *,
        slots: Mapping[str, float] | None = None,
        scan_table: Sequence[Sequence[float]] | None = None,
        repeat_forever: bool | None = None,
        scan_repeats: int | None = None,
    ) -> PulseSequence | PulseTableState:
        if isinstance(self.pulse, PulseTableState):
            table = self.scan_table if scan_table is None else scan_table
            merged = dict(self.slots)
            merged.update(slots or {})
            if table is not None and len(table) > 0:   # len(): numpy-array-safe (no truth value)
                data = self.pulse.to_dict()
                data["scan_table"] = [list(row) for row in table]
                payload = PulseTableState.from_dict(data)
            elif merged:
                payload = self.pulse.with_slots_resolved(merged)
            else:
                payload = self.pulse
            # repeat_forever / scan_repeats overrides flow through the dict (the same seam the GUI /
            # save bundle use).  The streamed-scan invariants -- "ANY scan_points => repeat_forever"
            # (0 = sweep forever, K = stop after K whole sweeps) and "a finite K-sweep needs >= 2
            # points" -- are enforced ONCE on the COMPILED program (RuntimeSequenceProgram.__post_init__),
            # which EVERY fire path passes through, INCLUDING the pulse GUI's On Pulse that bypasses
            # this method.  So we only carry the explicit overrides here and never re-derive streaming
            # (no second copy to drift; the GUI gets the same enforcement it used to miss).
            if repeat_forever is not None or scan_repeats is not None:
                data = payload.to_dict()
                if repeat_forever is not None:
                    data["repeat_forever"] = bool(repeat_forever)
                if scan_repeats is not None:
                    data["scan_repeats"] = max(0, int(scan_repeats))
                payload = PulseTableState.from_dict(data)
            return payload
        if repeat_forever is not None:
            data = self.pulse.to_dict()
            data["repeat_forever"] = bool(repeat_forever)
            return PulseSequence.from_dict(data)
        return self.pulse

    def prepare(
        self,
        *,
        scan_table: Sequence[Sequence[float]] | None = None,
        repeat_forever: bool | None = None,
        scan_repeats: int | None = None,
    ) -> RuntimeSequenceProgram:
        self.last_program = self._sequencer.prepare(
            self.payload(scan_table=scan_table, repeat_forever=repeat_forever, scan_repeats=scan_repeats))
        return self.last_program

    def on_pulse(
        self,
        *,
        wait: bool = False,
        timeout: float | None = None,
        scan_table: Sequence[Sequence[float]] | None = None,
        repeat_forever: bool | None = None,
        scan_repeats: int | None = None,
    ) -> RuntimeSequenceProgram:
        payload = self.payload(scan_table=scan_table, repeat_forever=repeat_forever, scan_repeats=scan_repeats)
        # A FINITE scan-repeat (scan_repeats>0) streams (repeat_forever) but DOES finish after K
        # sweeps, so it is exempt from the "cannot wait for a repeat_forever pulse" guard -- only a
        # truly endless repeat_forever (scan_repeats==0) needs a timeout to wait on.
        endless = bool(getattr(payload, "repeat_forever", False)) and int(getattr(payload, "scan_repeats", 0)) == 0
        if wait and timeout is None and endless:
            raise RuntimeError(
                "pulse.on_pulse(wait=True) cannot wait for a repeat_forever pulse without a timeout. "
                "Use pulse.on_pulse(wait=False, repeat_forever=True) for continuous scope output, "
                "or pulse.on_pulse(wait=True, repeat_forever=False) for a finite shot, "
                "or pulse.on_pulse(wait=True, scan_repeats=K) for a finite K-sweep scan."
            )
        # On Pulse ALWAYS = off_pulse -> prepare -> fire, unconditionally: a fresh run stops whatever
        # is currently playing, then uploads THIS payload and starts.  This is what makes On Pulse
        # order-independent -- switching a running scan (new slot / new points / new binding) is just
        # another On Pulse, never a "already running, reuse the old scan" short-circuit.  The stop also
        # clears any cached prepared program (set_safe_state drops prepared_program), so the next
        # prepare is guaranteed to re-upload rather than hit the cache.  Notebook, GUI and real all go
        # through this one seam, so the API and the GUI cannot drift into different run semantics.
        self.stop()
        self.last_program = self._sequencer.prepare(payload)
        program = self.last_program
        self._sequencer.fire()
        if wait:
            if not self._sequencer.wait_done(timeout=timeout):
                raise TimeoutError(f"sequencer did not report done for pulse {program.sequence_name!r}.")
        return program

    def wait_done(self, timeout: float | None = None, *, stop=None) -> bool:
        return bool(self._sequencer.wait_done(timeout=timeout, stop=stop))

    def stop(self) -> None:
        if hasattr(self._sequencer, "set_safe_state"):
            self._sequencer.set_safe_state()
        elif hasattr(self._sequencer, "abort"):
            self._sequencer.abort()

    def off_pulse(self) -> None:
        """Stop playback and drive the safe state (alias of :meth:`stop`).

        The natural opposite of :meth:`on_pulse` -- a delay-calibration loop
        reads ``pulse.set_channel_delay(ch, d).on_pulse(); ...; pulse.off_pulse()``."""
        self.stop()

    # --- pulse-table editing conveniences (channel delay calibration etc.) -----
    def set_channel_delay(self, channel: str, delay_ns: float) -> "PulseController":
        """Set one channel's fixed output delay (ns, may be negative) on the bound
        :class:`PulseTableState`.  The next :meth:`prepare`/:meth:`on_pulse` compiles
        and uploads the new delay -- the core primitive of a delay-calibration loop::

            for d in np.linspace(-200, 200, 21):
                pulse.set_channel_delay("ch11", d).on_pulse()
                counts.append(measure())
                pulse.off_pulse()
        """
        if not isinstance(self.pulse, PulseTableState):
            raise TypeError("set_channel_delay needs a PulseTableState pulse (not a raw PulseSequence).")
        channel = str(channel)
        if channel not in self.pulse.port_catalog.raw_lanes:
            raise ValueError(
                f"unknown channel {channel!r}; choices: "
                f"{list(self.pulse.port_catalog.raw_lanes)}")
        # Only the real TTL outputs have a delay line; a delay on a bus-member bit or a
        # da_clk pin has no hardware effect, so reject it here (the RTL also gates it to a
        # passthrough).  Eligible = hardware position < the eligible count.
        if float(delay_ns) != 0.0 and channel.startswith("ch") and channel[2:].isdigit():
            from .fpga_pulse_streamer import DEFAULT_FPGA_CHANNEL_COUNT, delay_eligible_channel_count
            if int(channel[2:]) >= delay_eligible_channel_count(DEFAULT_FPGA_CHANNEL_COUNT):
                raise ValueError(
                    f"channel {channel!r} is not delay-eligible (it is a bus-member bit or a "
                    f"da_clk pin, which has no delay line); only the real TTL outputs can be delayed.")
        self.pulse.delays[channel] = float(delay_ns)
        self.pulse.delay_units[channel] = "ns"
        self.pulse.validate()
        return self

    def get_channel_delay(self, channel: str) -> float:
        """Return one channel's fixed output delay in ns."""
        if not isinstance(self.pulse, PulseTableState):
            raise TypeError("get_channel_delay needs a PulseTableState pulse.")
        unit = str(self.pulse.delay_units.get(str(channel), "ns"))
        factor = UNITS_TO_NS.get(unit, 1.0)
        return float(self.pulse.delays.get(str(channel), 0.0)) * factor

    def load_pulse(self, path: str | Path) -> "PulseController":
        """Replace the bound pulse with a saved pulse-table JSON (GUI Save format)."""
        self.pulse = PulseTableState.load(path)
        self.slots = {}
        self.scan_table = list(self.pulse.scan_table or [])
        return self

    def save_pulse(self, path: str | Path) -> Path:
        """Save the bound PulseTableState as JSON (loadable by the GUI and load_pulse)."""
        if not isinstance(self.pulse, PulseTableState):
            raise TypeError("save_pulse needs a PulseTableState pulse.")
        path = Path(path)
        self.pulse.save(path)
        return path

    def synced_state(self) -> PulseTableState | None:
        """Return the PulseTableState actually APPLIED on the sequencer, or None.

        Reads the sequencer snapshot's ``last_payload_json`` (recorded by the
        service at every successful prepare -- whether it came from this
        controller, the GUI, or any other raw-API caller).  This is the same
        source the GUI's "Sync" button uses, so notebook and GUI always agree
        on what is running."""
        snap = self._sequencer.snapshot() if hasattr(self._sequencer, "snapshot") else {}
        payload = snap.get("last_payload_json")
        if not payload:
            return None
        data = json.loads(payload)
        if isinstance(data, Mapping) and "periods" in data:
            return PulseTableState.from_dict(data)
        return None

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-safe summary for notebook/debug display."""

        last = None
        if self.last_program is not None:
            last = {
                "sequence_name": self.last_program.sequence_name,
                "channels": list(self.last_program.channels),
                "edge_count": len(self.last_program.ticks),
                "duration": float(self.last_program.duration),
                "repeat_forever": bool(self.last_program.repeat_forever),
                "loop_count": int(self.last_program.loop_count),
            }
        return {
            "type": type(self).__name__,
            "pulse_type": type(self.pulse).__name__,
            "slots": dict(self.slots),
            "scan_table": [list(row) for row in self.scan_table],
            "sequencer_type": type(self._sequencer).__name__,
            "sequencer_channels": list(getattr(self._sequencer, "channels", [])),
            "clock_hz": float(getattr(self._sequencer, "clock_hz", 0.0)),
            "last_program": last,
        }


def bind_pulse(sequencer: SequencerDevice, pulse: PulseSequence | PulseTableState) -> PulseController:
    """Return a ``PulseController`` for concise notebook pulse scans."""

    return PulseController(sequencer, pulse)




# RPyC-transport array fields.  A large numeric list (the scan_table forward, the compiled
# scan_points / scan_point_durations on return) crosses the RPyC wire as ONE raw little-endian
# ndarray buffer instead of a per-number JSON list -- removing the O(N) json.dumps/loads over the
# tens of thousands of scan numbers (measured ~40 ms round-trip at 20000 points).  The rebuilt value
# is a NATIVE Python list identical to what json.loads would have produced, so the compiled program
# is byte-for-byte the same as the all-JSON path (proven by an in-process RPyC-loopback hash check).
_WIRE_ARRAY_FIELDS_PAYLOAD = ("scan_table",)
_WIRE_ARRAY_FIELDS_PROGRAM = ("scan_points", "scan_point_durations")


























































def serve_runtime_sequencer(
    service: SequencerService,
    *,
    host: str = "0.0.0.0",
    port: int = 18861,
    start: bool = True,
):
    """Expose ``SequencerService`` over RPyC on the FPGA/Vivado computer."""

    try:
        import rpyc
        from rpyc.utils.server import ThreadedServer
    except ImportError as exc:  # pragma: no cover - depends on lab install
        raise RuntimeError("serve_runtime_sequencer requires `rpyc` on the FPGA computer.") from exc

    class RPyCSequencerService(rpyc.Service):
        def on_disconnect(self, conn):
            # The control client dropped (GUI closed / crashed / link lost).  A
            # repeat_forever streamed program would otherwise keep driving the
            # FPGA outputs with nobody watching -- leave hardware in the safe
            # state on disconnect.  Best-effort: nothing to propagate to here.
            try:
                service.set_safe_state()
            except Exception:
                pass

        def exposed_prepare(self, head_json, blobs=None):
            payload = decode_wire_payload(head_json, blobs or {})
            return encode_wire_payload(service.prepare(payload), _WIRE_ARRAY_FIELDS_PROGRAM)

        def exposed_fire(self, head_json=None, blobs=None):
            payload = None if head_json is None else decode_wire_payload(head_json, blobs or {})
            return json.dumps(service.fire(payload))

        def exposed_wait_done(self, timeout=None):
            return service.wait_done(timeout)

        def exposed_abort(self):
            return service.abort()

        def exposed_set_safe_state(self):
            return service.set_safe_state()

        def exposed_scan_progress(self):
            return service.scan_progress()

        def exposed_snapshot(self):
            return service.snapshot()

    server = ThreadedServer(
        RPyCSequencerService,
        hostname=host,
        port=int(port),
        protocol_config={"allow_public_attrs": True, "allow_pickle": True, "sync_request_timeout": None},
    )
    if start:
        server.start()
    return server


__all__ = [
    "ManualSequencer",
    "PulseController",
    "RemoteSequencer",
    "RuntimeBusDelay",
    "RuntimeBusSegment",
    "RuntimeSequenceProgram",
    "SequencerService",
    "bind_pulse",
    "compile_pulse_table_runtime_program",
    "compile_pulse_table_scan_runtime_program",
    "compile_runtime_program",
    "compile_runtime_program_for_payload",
    "serve_runtime_sequencer",
]

# The domain's PulseTableState.compile/compile_scan reach the runtime compilers through a
# registry (the domain package may not import the device layer).  This module IS the device
# layer, so loading it arms the registry -- the same inversion the frontend uses for the
# solve-thread guard.
# (registry import removed: pulse_table now imports its sibling compiler directly) as _register_runtime_compilers



