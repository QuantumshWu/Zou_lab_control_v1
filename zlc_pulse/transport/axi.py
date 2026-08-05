"""Persistent Vivado/JTAG word transport for the frozen pulse streamer."""

from __future__ import annotations

import os
import math
import queue
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Sequence

from fpga.pulse_streamer.host.image import CtrlWords

from .session import TransportAborted


_AXI_BURST_BOUNDARY_BYTES = 4096


def _default_vivado() -> str:
    return os.environ.get("ZLC_PS_VIVADO_BIN", "vivado")


def _default_probes() -> str | None:
    for name in ("ZLC_PS_VIVADO_LTX", "ZLC_PS_LTX"):
        value = os.environ.get(name)
        if value:
            return value
    root = os.environ.get("ZLC_PS_PROJECT_DIR")
    if not root:
        return None
    candidate = Path(root) / "ps.runs" / "impl_1" / "zlc_pulse_streamer_top.ltx"
    return str(candidate) if candidate.exists() else None


class VivadoAxiRegisterTransport:
    """Ordered 32-bit register/BRAM access through one persistent Vivado Tcl owner."""

    transport_id = "vivado-axi"
    def __init__(
        self,
        *,
        state_dir: str | Path,
        vivado: str | None = None,
        probes: str | None = None,
        hw_server_url: str | None = None,
        startup_timeout: float = 180.0,
        action_timeout: float = 120.0,
        write_batch: int = 200,
        burst_max: int = 256,
        tcl_executor: Callable[[Sequence[str], str, float | None], str] | None = None,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.vivado = vivado or _default_vivado()
        self.probes = probes if probes is not None else _default_probes()
        self.hw_server_url = (
            hw_server_url
            or os.environ.get("ZLC_PS_HW_SERVER_URL")
            or os.environ.get("ZLC_HW_SERVER_URL")
            or ""
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in (startup_timeout, action_timeout)
        ):
            raise ValueError("AXI transport timeouts must be finite and positive")
        self.startup_timeout = float(startup_timeout)
        self.action_timeout = float(action_timeout)
        self.write_batch = max(1, int(write_batch))
        self.burst_max = max(1, min(256, int(burst_max)))
        self._external_executor = tcl_executor
        self._io_lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._counter = 0
        self._closed = False
        self._log_path = self.state_dir / "vivado_axi_transport.log"

    def start(self) -> None:
        if self._external_executor is not None:
            self._closed = False
            return
        if self._process is not None:
            return
        self._closed = False
        try:
            process = subprocess.Popen(
                [self.vivado, "-mode", "tcl", "-nolog", "-nojournal"],
                cwd=self.state_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as error:
            self.record_diagnostic(
                "start_failure",
                f"Vivado executable was not found: {self.vivado!r}",
            )
            raise RuntimeError("persistent Vivado transport could not start") from error
        self._process = process
        generation_queue: queue.Queue[str | None] = queue.Queue()
        self._queue = generation_queue
        self._reader = threading.Thread(
            target=self._read_stdout,
            args=(process, generation_queue),
            name="zlc-vivado-axi-reader",
            daemon=True,
        )
        self._reader.start()
        self._run_tcl(
            self._init_tcl(),
            action="transport_start",
            deadline=time.monotonic() + self.startup_timeout,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write(
                    "catch {close_hw_target}\n"
                    "catch {disconnect_hw_server}\n"
                    "exit\n"
                )
                process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
        self._process = None

    def write_words(
        self,
        rows: Sequence[tuple[int, int]],
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> None:
        absolute_deadline = self._effective_deadline(deadline)
        pending = tuple(
            (int(word_offset) * 4, int(value) & 0xFFFFFFFF)
            for word_offset, value in rows
        )
        lines: list[str] = []
        bursts = 0
        for base, values in self._burst_runs(pending):
            lines.extend(self._write_burst_tcl(base, values))
            bursts += 1
            if bursts >= self.write_batch:
                self._run_tcl(
                    lines,
                    action="axi_write",
                    deadline=absolute_deadline,
                    stop=stop,
                )
                lines = []
                bursts = 0
        if lines:
            self._run_tcl(
                lines,
                action="axi_write",
                deadline=absolute_deadline,
                stop=stop,
            )

    def rewrite_scan_bank(
        self,
        *,
        unarmed_bank_ready: int,
        bank_words: Sequence[tuple[int, int]],
        chunk_word: int,
        chunk_index: int,
        rearmed_bank_ready: int,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> None:
        """Commit one bank in a single ordered Vivado action."""

        if chunk_word not in (CtrlWords.BANK0_CHUNK, CtrlWords.BANK1_CHUNK):
            raise ValueError("scan-bank rewrite has an invalid chunk register")
        self.write_words(
            (
                (CtrlWords.BANK_READY, unarmed_bank_ready),
                *tuple(bank_words),
                (chunk_word, chunk_index),
                (CtrlWords.BANK_READY, rearmed_bank_ready),
            ),
            stop=stop,
            deadline=deadline,
        )

    def read_word(
        self,
        word_offset: int,
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> int:
        absolute_deadline = self._effective_deadline(deadline)
        marker = "ZLCDATA"
        output = self._run_tcl(
            self._read_txn_tcl(int(word_offset) * 4, marker),
            action="axi_read",
            deadline=absolute_deadline,
            stop=stop,
        )
        return self._parse_read(output, marker)

    def read_words(
        self,
        word_offsets: Sequence[int],
        *,
        stop: threading.Event | None = None,
        deadline: float | None = None,
    ) -> tuple[int, ...]:
        """Read several proof words in one persistent-Vivado action.

        Each address remains an independent AXI read transaction; only the costly
        Python/Vivado stdin-marker round trip is shared.  External executors retain
        the older one-read contract used by hardware models and therefore fall back
        to ``read_word``.
        """

        offsets = tuple(int(value) for value in word_offsets)
        if not offsets:
            return ()
        if self._external_executor is not None:
            return tuple(
                self.read_word(address, stop=stop, deadline=deadline)
                for address in offsets
            )
        absolute_deadline = self._effective_deadline(deadline)
        lines: list[str] = []
        markers: list[str] = []
        for index, address in enumerate(offsets):
            marker = f"ZLCBATCH_{index:04d}"
            markers.append(marker)
            lines.extend(self._read_txn_tcl(address * 4, marker))
        output = self._run_tcl(
            lines,
            action="axi_read_batch",
            deadline=absolute_deadline,
            stop=stop,
        )
        return tuple(self._parse_read(output, marker) for marker in markers)

    def record_diagnostic(self, name: str, text: str) -> None:
        try:
            (self.state_dir / f"{name}.log").write_text(
                text,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            pass

    def _effective_deadline(self, value: float | None) -> float:
        deadline = (
            time.monotonic() + self.action_timeout
            if value is None
            else float(value)
        )
        if not math.isfinite(deadline):
            raise ValueError("AXI transaction deadline must be finite")
        if deadline <= time.monotonic():
            raise TimeoutError("AXI transaction deadline expired before issue")
        return deadline

    def _init_tcl(self) -> list[str]:
        lines = [
            "if {[llength [info commands load_features]]} { catch {load_features labtools} }",
            "if {[llength [info commands open_hw_manager]]} { open_hw_manager } elseif {[llength [info commands open_hw]]} { open_hw }",
        ]
        if self.hw_server_url:
            lines.append(f"connect_hw_server -url {self.hw_server_url}")
        else:
            lines.append("if {[catch {connect_hw_server}]} { connect_hw_server }")
        lines.extend(
            [
                "catch {refresh_hw_server}",
                "set zlc_targets [get_hw_targets]",
                'if {$zlc_targets eq ""} { error "No Vivado hardware target" }',
                "current_hw_target [lindex $zlc_targets 0]",
                "if {[catch {open_hw_target}]} { catch {close_hw_target}; after 2000; open_hw_target }",
                "current_hw_device [lindex [get_hw_devices] 0]",
            ]
        )
        if self.probes:
            lines.append(
                f'set_property PROBES.FILE {{{self.probes}}} [current_hw_device]'
            )
            lines.append(
                f'set_property FULL_PROBES.FILE {{{self.probes}}} [current_hw_device]'
            )
        # Frozen deployment: this owner never selects or programs a bitstream.
        lines.extend(
            [
                "refresh_hw_device [current_hw_device]",
                "set zlc_axi [get_hw_axis]",
                'if {$zlc_axi eq ""} { error "No JTAG-to-AXI core found" }',
            ]
        )
        return lines

    def _burst_runs(
        self,
        pending: Sequence[tuple[int, int]],
    ) -> list[tuple[int, list[int]]]:
        """Coalesce only insertion-order contiguous writes; never reorder commands."""

        runs: list[tuple[int, list[int]]] = []
        index = 0
        while index < len(pending):
            base, value = pending[index]
            values = [value]
            cursor = index + 1
            while (
                cursor < len(pending)
                and len(values) < self.burst_max
                and pending[cursor][0] == base + 4 * len(values)
                and (base + 4 * len(values)) // _AXI_BURST_BOUNDARY_BYTES
                == base // _AXI_BURST_BOUNDARY_BYTES
            ):
                values.append(pending[cursor][1])
                cursor += 1
            runs.append((base, values))
            index = cursor
        return runs

    @staticmethod
    def _write_burst_tcl(byte_address: int, values: Sequence[int]) -> list[str]:
        address = f"{byte_address & 0xFFFFFFFF:08X}"
        data = "".join(
            f"{int(value) & 0xFFFFFFFF:08X}" for value in reversed(values)
        )
        return [
            f"create_hw_axi_txn zlc_w [get_hw_axis] -address {address} -data {data} -len {len(values)} -type write -burst INCR -force",
            "run_hw_axi zlc_w",
            "delete_hw_axi_txn zlc_w",
        ]

    @staticmethod
    def _read_txn_tcl(byte_address: int, marker: str) -> list[str]:
        address = f"{byte_address & 0xFFFFFFFF:08X}"
        return [
            f"create_hw_axi_txn zlc_r [get_hw_axis] -address {address} -len 1 -type read -force",
            "run_hw_axi zlc_r",
            f'puts "{marker} [get_property DATA [get_hw_axi_txns zlc_r]]"',
            "delete_hw_axi_txn zlc_r",
        ]

    @staticmethod
    def _parse_read(output: str, marker: str) -> int:
        token = ""
        for line in output.splitlines():
            if marker in line:
                fields = line.split(marker, 1)[1].strip().split()
                token = fields[-1] if fields else ""
        if not token:
            raise RuntimeError("hw_axi read returned no DATA")
        if re.fullmatch(r"(?:0[xX])?[0-9a-fA-F]{1,8}", token) is None:
            raise RuntimeError(f"hw_axi read returned non-binary DATA {token!r}")
        return int(token, 16) & 0xFFFFFFFF

    def _run_tcl(
        self,
        lines: Sequence[str],
        *,
        action: str,
        deadline: float,
        stop: threading.Event | None = None,
    ) -> str:
        remaining = self._remaining(deadline, action)
        if not self._io_lock.acquire(timeout=remaining):
            raise TimeoutError(f"{action} timed out waiting for the AXI I/O owner")
        try:
            if self._closed:
                raise RuntimeError(
                    "Vivado AXI transport is closed; call start() before another transaction"
                )
            if stop is not None and stop.is_set():
                raise TransportAborted(f"{action} aborted before issue")
            if self._external_executor is not None:
                return self._external_executor(
                    list(lines),
                    action,
                    self._remaining(deadline, action),
                )
            return self._execute(
                lines,
                action=action,
                deadline=deadline,
                stop=stop,
            )
        finally:
            self._io_lock.release()

    def _execute(
        self,
        lines: Sequence[str],
        *,
        action: str,
        deadline: float,
        stop: threading.Event | None,
    ) -> str:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError(
                "persistent Vivado transport is not running; call start() explicitly"
            )
        self._remaining(deadline, action)
        self._counter += 1
        marker = f"ZLC_AXI_{self._counter:06d}"
        try:
            process.stdin.write(self._wrap_tcl(lines, marker))
            process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self.close()
            raise RuntimeError(f"Vivado stopped before {action}") from error
        output = self._read_until_marker(marker, deadline=deadline, stop=stop)
        self.record_diagnostic(action, output)
        if f"{marker}_ERROR" in output:
            raise RuntimeError(
                f"persistent Vivado {action} failed: "
                f"{self._error_detail(output, marker, action)}"
            )
        return output

    def _error_detail(self, output: str, marker: str, action: str) -> str:
        """Return the Tcl result the wrapper already printed beside the marker.

        Vivado prefixes its own faults with ``ERROR:``, but a Tcl ``error``
        raised by this transport's bring-up script does not -- and those are
        exactly the two operator-facing causes ("No Vivado hardware target",
        "No JTAG-to-AXI core found").  Reporting only that an error occurred
        discarded the one fact that says which repair to make.
        """

        token = f"{marker}_ERROR"
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        for line in lines:
            if line.startswith(token):
                detail = line[len(token):].strip()
                if detail:
                    return detail
        for line in lines:
            if line.startswith("ERROR:"):
                return line
        return (
            "Vivado Tcl action returned an error; full output in "
            f"{self.state_dir / f'{action}.log'}"
        )

    def _kill_process(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            try:
                process.kill()
            except Exception:
                pass

    @staticmethod
    def _wrap_tcl(lines: Sequence[str], marker: str) -> str:
        body = "\n".join(lines)
        return (
            f'puts "{marker}_BEGIN"\n'
            # A failed run_hw_axi skips the transaction helper's normal delete.
            # Clear only our private fixed names before and after every action so
            # one Xicom fault cannot poison the later physical-SAFE attempt.
            "catch {delete_hw_axi_txn -quiet [get_hw_axi_txns -quiet zlc_w]}\n"
            "catch {delete_hw_axi_txn -quiet [get_hw_axi_txns -quiet zlc_r]}\n"
            "set zlc_axi_code [catch {\n"
            f"{body}\n"
            "} zlc_axi_result zlc_axi_options]\n"
            "catch {delete_hw_axi_txn -quiet [get_hw_axi_txns -quiet zlc_w]}\n"
            "catch {delete_hw_axi_txn -quiet [get_hw_axi_txns -quiet zlc_r]}\n"
            "if {$zlc_axi_code} {\n"
            f'    puts "{marker}_ERROR $zlc_axi_result"\n'
            "    if {[dict exists $zlc_axi_options -errorinfo]} { puts [dict get $zlc_axi_options -errorinfo] }\n"
            "} else {\n"
            f'    puts "{marker}_OK"\n'
            "}\n"
            f'puts "{marker}_END"\n'
            "flush stdout\n"
        )

    def _read_until_marker(
        self,
        marker: str,
        *,
        deadline: float,
        stop: threading.Event | None,
    ) -> str:
        lines: list[str] = []
        while True:
            if stop is not None and stop.is_set():
                raise TransportAborted(f"read aborted waiting for {marker}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._closed = True
                self._kill_process()
                raise TimeoutError(f"Vivado action timed out waiting for {marker}")
            slice_seconds = min(0.2, remaining)
            try:
                item = self._queue.get(timeout=slice_seconds)
            except queue.Empty:
                continue
            if item is None:
                raise RuntimeError("persistent Vivado process exited unexpectedly")
            lines.append(item)
            if f"{marker}_END" in item:
                return "".join(lines)

    @staticmethod
    def _remaining(deadline: float, action: str) -> float:
        remaining = float(deadline) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"{action} exceeded its absolute deadline")
        return remaining

    def _read_stdout(
        self,
        process: subprocess.Popen,
        generation_queue: queue.Queue[str | None],
    ) -> None:
        stdout = process.stdout
        if stdout is None:
            generation_queue.put(None)
            return
        with self._log_path.open("a", encoding="utf-8", errors="replace") as log:
            for line in stdout:
                log.write(line)
                log.flush()
                generation_queue.put(line)
        generation_queue.put(None)


__all__ = ["VivadoAxiRegisterTransport"]
