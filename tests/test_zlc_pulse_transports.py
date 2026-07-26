"""Current transports preserve order and finite scan continuity contracts."""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from fpga.pulse_streamer.host.image import (
    CMD_FIRE,
    CMD_LOAD,
    CMD_SAFE,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_LOADED,
    STATUS_RUNNING,
    STATUS_UNDERFLOW,
    CtrlWords,
    StreamerParams,
    build_fingerprint,
)
from fpga.pulse_streamer.host import uart_frame as framing
from zlc_pulse import (
    OutputDelay,
    PulseExecutionForm,
    PulseExecutionService,
    compile_pulse_artifact,
    freeze_scan_table,
    load_pulse_document,
    pulse_target_manifest_from_lanes,
)
from zlc_pulse.transport import (
    DeployedStreamerSession,
    InterprocessDeviceLease,
    TransportAborted,
    UartError,
    UartRegisterTransport,
    VivadoAxiRegisterTransport,
)


ROOT = Path(__file__).parents[1]


class MemoryDeviceLease:
    def __init__(self):
        self.acquired = False

    def acquire(self):
        if self.acquired:
            raise RuntimeError("test device lease is already held")
        self.acquired = True

    def release(self):
        self.acquired = False


class TraceRegisterTransport:
    transport_id = "trace"

    def __init__(self, params):
        self.params = params
        self.words = {CtrlWords.LAYOUT_ID: build_fingerprint(params)}
        self.write_batches = []
        self.read_addresses = []
        self.status = 0
        self.cursor = 0
        self.closed = False

    def start(self):
        pass

    def close(self):
        self.closed = True

    def write_words(self, rows, *, stop=None, deadline=None):
        if stop is not None and stop.is_set():
            from zlc_pulse.transport import TransportAborted

            raise TransportAborted("cancelled trace write")
        rows = tuple(rows)
        self.write_batches.append(rows)
        for address, value in rows:
            self.words[address] = value
            if address == CtrlWords.STATUS:
                self.status = value
            if address == CtrlWords.COMMAND and value:
                if value & CMD_SAFE:
                    self.status = 0
                elif value & CMD_LOAD:
                    self.status = STATUS_LOADED
                elif value & CMD_FIRE:
                    self.status = STATUS_RUNNING

    def read_word(self, address, *, stop=None, deadline=None):
        self.read_addresses.append(address)
        if stop is not None and stop.is_set():
            from zlc_pulse.transport import TransportAborted

            raise TransportAborted("cancelled trace read")
        if address == CtrlWords.STATUS:
            return self.status
        if address == CtrlWords.CURSOR:
            return self.cursor
        return self.words.get(address, 0)

    def record_diagnostic(self, name, text):
        pass


def _scan_artifact(params, count=5):
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    rows = tuple((float(index), 0.0, 0.0) for index in range(count))
    table, _report = freeze_scan_table(
        document,
        ("da_x", "da_y", "da_z"),
        rows,
    )
    document = replace(document, scan_table=table)
    return document, compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        trigger_channels=("ch11",),
        params=params,
    )


def _session(document, params, transport):
    return DeployedStreamerSession(
        transport,
        device_lease=MemoryDeviceLease(),
        deployed_target=document.target,
        params=params,
        clock_hz=50e6,
        terminal_poll_interval=0.0001,
    ).start()


def test_finite_scan_refills_freed_ping_pong_bank_in_the_fire_owned_worker():
    params = replace(StreamerParams(), bank_size=2)
    document, artifact = _scan_artifact(params, count=5)
    transport = TraceRegisterTransport(params)
    session = _session(document, params, transport)

    session.prepare(artifact)
    session.fire(artifact)
    transport.cursor = 2
    deadline = time.monotonic() + 1.0
    refill = None
    while time.monotonic() < deadline:
        refill = next(
            (
                batch
                for batch in transport.write_batches
                if (CtrlWords.BANK0_CHUNK, 2) in batch
            ),
            None,
        )
        if refill is not None:
            break
        time.sleep(0.001)
    assert refill is not None
    assert refill[0] == (CtrlWords.BANK_READY, 0b10)
    assert refill[-2:] == (
        (CtrlWords.BANK0_CHUNK, 2),
        (CtrlWords.BANK_READY, 0b11),
    )

    transport.cursor = 4
    transport.status = STATUS_DONE
    assert session.await_completion(artifact, timeout=1.0) is not None
    assert session.snapshot()["next_monotonic_scan_chunk"] == 3


def test_scan_refill_rejects_a_boundary_crossing_hidden_by_rearm():
    class CrossingDuringRewriteTransport(TraceRegisterTransport):
        def write_words(self, rows, *, stop=None, deadline=None):
            rows = tuple(rows)
            super().write_words(rows, stop=stop, deadline=deadline)
            if (CtrlWords.BANK0_CHUNK, 2) in rows:
                # Model the exact non-sticky RTL race: playback reached the bank
                # boundary while the host write was in flight, then advanced as
                # soon as the final READY write cleared the transient stall.
                self.cursor = 4

    params = replace(StreamerParams(), bank_size=2)
    document, artifact = _scan_artifact(params, count=5)
    transport = CrossingDuringRewriteTransport(params)
    session = _session(document, params, transport)

    session.prepare(artifact)
    session.fire(artifact)
    transport.cursor = 2

    with pytest.raises(RuntimeError, match="boundary while its next bank refill"):
        session.await_completion(artifact, timeout=1.0)
    assert session.snapshot()["state"] == "FAILED"


def test_public_service_accepts_scan_larger_than_the_two_resident_banks():
    params = replace(StreamerParams(), bank_size=2)
    document, artifact = _scan_artifact(params, count=5)
    transport = TraceRegisterTransport(params)
    session = _session(document, params, transport)
    service = PulseExecutionService(
        pulse_target_manifest_from_lanes(document.target),
        clock_hz=50e6,
        backend=session,
        params=params,
    )

    reference = service.prepare(artifact)

    assert reference.artifact_digest == artifact.fingerprint
    assert service.snapshot()["state"] == "PREPARED"
    service.safe_state()


def test_resident_finite_terminal_owner_starts_at_fire_before_await():
    params = replace(StreamerParams(), bank_size=2)
    document, artifact = _scan_artifact(params, count=4)
    transport = TraceRegisterTransport(params)
    session = _session(document, params, transport)
    session.prepare(artifact)

    session.fire(artifact)
    transport.cursor = 3
    transport.status = STATUS_DONE
    completion = session.await_completion(artifact, timeout=1.0)
    assert completion is not None
    terminal = session.snapshot()["terminal"]
    assert terminal["cursor_first"] == terminal["cursor_second"] == 3
    assert terminal["schema"] == "zlc_pulse.AutonomousTableTerminalEvidence"
    assert "expected_final_cursor" not in terminal
    assert "logical_done" not in terminal
    assert completion.post_terminal_tail.terminal_evidence_digest == (
        completion.hardware_terminal.fingerprint
    )


def test_continuous_scan_observer_publishes_only_sampled_hardware_cursor():
    params = replace(StreamerParams(), bank_size=2)
    document, _finite = _scan_artifact(params, count=4)
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
        params=params,
    )
    transport = TraceRegisterTransport(params)
    session = _session(document, params, transport)
    session.prepare(artifact)

    before_fire = session.snapshot()
    assert before_fire["cursor_sample_count"] == 0
    session.fire(artifact)
    transport.cursor = 2

    deadline = time.monotonic() + 1.0
    while True:
        observed = session.snapshot()
        if (
            observed["cursor_sample_count"] > 0
            and observed["last_confirmed_cursor"] == 2
        ):
            break
        if time.monotonic() >= deadline:
            raise AssertionError("continuous scan cursor was not sampled")
        time.sleep(0.001)

    assert observed["state"] == "RUNNING"
    assert observed["terminal"] is None
    assert observed["underflow_observed"] is False
    session.safe_state()


def test_static_terminal_evidence_never_reads_semantically_empty_cursor():
    params = StreamerParams()
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    artifact = compile_pulse_artifact(
        document,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
        params=params,
    )
    transport = TraceRegisterTransport(params)
    session = _session(document, params, transport)
    session.prepare(artifact)
    session.fire(artifact)
    transport.read_addresses.clear()
    transport.status = STATUS_DONE

    completion = session.await_completion(artifact, timeout=1.0)

    assert completion is not None
    assert CtrlWords.CURSOR not in transport.read_addresses
    assert session.snapshot()["terminal"]["schema"] == (
        "zlc_pulse.StaticOnceTerminalEvidence"
    )


def test_autonomous_terminal_reads_the_frozen_register_recipe_exactly():
    params = replace(StreamerParams(), bank_size=2)
    document, artifact = _scan_artifact(params, count=4)

    class ImmediateDoneTransport(TraceRegisterTransport):
        def write_words(self, rows, *, stop=None, deadline=None):
            rows = tuple(rows)
            super().write_words(rows, stop=stop, deadline=deadline)
            if any(
                address == CtrlWords.COMMAND and value & CMD_FIRE
                for address, value in rows
            ):
                self.status = STATUS_DONE
                self.cursor = 3

    transport = ImmediateDoneTransport(params)
    session = _session(document, params, transport)
    session.prepare(artifact)
    transport.read_addresses.clear()

    session.fire(artifact)
    completion = session.await_completion(artifact, timeout=1.0)

    assert completion is not None
    assert transport.read_addresses == [
        CtrlWords.STATUS,
        CtrlWords.CURSOR,
        CtrlWords.STATUS,
        CtrlWords.CURSOR,
    ]


def test_safe_during_deployment_validation_cannot_revive_prepared_state(monkeypatch):
    params = StreamerParams()
    document, artifact = _scan_artifact(params, count=2)
    transport = TraceRegisterTransport(params)
    session = _session(document, params, transport)
    session.check_register_layout()
    entered = threading.Event()
    release = threading.Event()

    def blocked_validation(*args, **kwargs):
        entered.set()
        assert release.wait(1.0)

    monkeypatch.setattr(
        "zlc_pulse.transport.session._validate_artifact_against_bound_deployment",
        blocked_validation,
    )
    errors = []

    def prepare():
        try:
            session.prepare(artifact)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=prepare)
    worker.start()
    assert entered.wait(1.0)
    session.safe_state()
    release.set()
    worker.join(1.0)

    assert not worker.is_alive()
    assert errors and isinstance(errors[0], TransportAborted)
    assert session.state == "SAFE"
    assert session.snapshot()["prepared_artifact_digest"] is None


def test_safe_after_fire_command_commit_cannot_be_overwritten_by_late_fire():
    class BlockingFireTransport(TraceRegisterTransport):
        def __init__(self, params):
            super().__init__(params)
            self.fire_committed = threading.Event()
            self.release_fire = threading.Event()

        def write_words(self, rows, *, stop=None, deadline=None):
            rows = tuple(rows)
            super().write_words(rows, stop=stop, deadline=deadline)
            if (CtrlWords.COMMAND, CMD_FIRE) in rows:
                self.fire_committed.set()
                assert self.release_fire.wait(1.0)

    params = StreamerParams()
    document, artifact = _scan_artifact(params, count=2)
    transport = BlockingFireTransport(params)
    session = _session(document, params, transport)
    session.prepare(artifact)
    errors = []

    def fire():
        try:
            session.fire(artifact)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=fire)
    worker.start()
    assert transport.fire_committed.wait(1.0)
    session.safe_state()
    transport.release_fire.set()
    worker.join(1.0)

    assert not worker.is_alive()
    assert errors and isinstance(errors[0], TransportAborted)
    assert session.state == "SAFE"
    assert session.snapshot()["prepared_artifact_digest"] is None


def test_safe_interrupts_tail_wait_without_republishing_a_drain_deadline():
    params = StreamerParams()
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    delayed = replace(document, delays=(OutputDelay("ch11", 200, "ms"),))
    artifact = compile_pulse_artifact(
        delayed,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        params=params,
    )
    transport = TraceRegisterTransport(params)
    session = _session(delayed, params, transport)
    session.prepare(artifact)
    session.fire(artifact)
    transport.status = STATUS_DONE
    errors = []

    def wait():
        try:
            session.await_completion(artifact, timeout=1.0)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=wait)
    worker.start()
    deadline = time.monotonic() + 1.0
    while session.state != "DONE" and time.monotonic() < deadline:
        time.sleep(0.001)
    assert session.state == "DONE"
    session.safe_state()
    worker.join(1.0)

    assert not worker.is_alive()
    assert errors and isinstance(errors[0], TransportAborted)
    assert session.state == "SAFE"


def test_any_observed_underflow_invalidates_the_whole_formal_run():
    params = replace(StreamerParams(), bank_size=2)
    document, artifact = _scan_artifact(params, count=4)
    transport = TraceRegisterTransport(params)
    session = _session(document, params, transport)
    session.prepare(artifact)
    session.fire(artifact)
    transport.status = STATUS_RUNNING | STATUS_UNDERFLOW

    with pytest.raises(RuntimeError, match="underflowed"):
        session.await_completion(artifact, timeout=1.0)
    assert session.state == "FAILED"
    assert any(
        address == CtrlWords.COMMAND and value == CMD_SAFE
        for batch in transport.write_batches
        for address, value in batch
    )


def test_terminal_cursor_mismatch_is_rejected_even_when_done_is_set():
    params = replace(StreamerParams(), bank_size=4)
    document, artifact = _scan_artifact(params, count=3)
    transport = TraceRegisterTransport(params)
    session = _session(document, params, transport)
    session.prepare(artifact)
    session.fire(artifact)
    transport.cursor = 1
    transport.status = STATUS_DONE

    with pytest.raises(ValueError, match="terminal CURSOR"):
        session.await_completion(artifact, timeout=1.0)


def test_short_wait_does_not_allow_next_prepare_to_cut_off_delay_tail():
    params = StreamerParams()
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    delayed = replace(document, delays=(OutputDelay("ch11", 20, "ms"),))
    artifact = compile_pulse_artifact(
        delayed,
        clock_hz=50e6,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        params=params,
    )
    transport = TraceRegisterTransport(params)
    session = _session(delayed, params, transport)
    session.prepare(artifact)
    session.fire(artifact)
    transport.status = STATUS_DONE

    assert session.await_completion(artifact, timeout=0.001) is None
    started = time.monotonic()
    session.prepare(artifact)
    assert time.monotonic() - started >= 0.012


def _axi_runs(text):
    return [
        (int(address, 16) // 4, int(length))
        for address, length in re.findall(
            r"-address ([0-9A-Fa-f]+) .*?-len (\d+) -type write",
            text,
        )
    ]


def test_axi_transport_preserves_repeated_addresses_and_splits_4kb_boundaries(tmp_path):
    calls = []

    def execute(lines, action, timeout):
        calls.append("\n".join(lines))
        return "ok\n"

    transport = VivadoAxiRegisterTransport(
        state_dir=tmp_path,
        tcl_executor=execute,
        burst_max=256,
    )
    transport.write_words(
        (
            (1022, 1),
            (1023, 2),
            (1024, 3),
            (CtrlWords.COMMAND, 0),
            (CtrlWords.COMMAND, CMD_FIRE),
        )
    )
    assert _axi_runs("\n".join(calls)) == [
        (1022, 2),
        (1024, 1),
        (CtrlWords.COMMAND, 1),
        (CtrlWords.COMMAND, 1),
    ]


def test_axi_absolute_deadline_includes_waiting_for_the_io_owner(tmp_path):
    transport = VivadoAxiRegisterTransport(
        state_dir=tmp_path,
        tcl_executor=lambda lines, action, timeout: "ok\n",
    )
    held = threading.Event()
    release = threading.Event()

    def hold_io_owner():
        with transport._io_lock:
            held.set()
            assert release.wait(1.0)

    owner = threading.Thread(target=hold_io_owner)
    owner.start()
    assert held.wait(1.0)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="I/O owner"):
            transport.read_word(1, deadline=started + 0.02)
    finally:
        release.set()
        owner.join(1.0)

    assert not owner.is_alive()
    assert time.monotonic() - started < 0.08


def test_safe_absolute_deadline_includes_waiting_for_an_older_safe():
    params = StreamerParams()
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    transport = TraceRegisterTransport(params)
    session = DeployedStreamerSession(
        transport,
        device_lease=MemoryDeviceLease(),
        deployed_target=document.target,
        params=params,
        clock_hz=50e6,
        action_timeout=0.02,
    ).start()
    session.check_register_layout()
    held = threading.Event()
    release = threading.Event()

    def hold_safe_owner():
        with session._safe_lock:
            held.set()
            assert release.wait(1.0)

    owner = threading.Thread(target=hold_safe_owner)
    owner.start()
    assert held.wait(1.0)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="safety owner"):
            session.safe_state()
    finally:
        release.set()
        owner.join(1.0)

    assert not owner.is_alive()
    assert time.monotonic() - started < 0.08
    session.close()


def test_clear_host_config_rechecks_layout_before_its_first_write():
    params = StreamerParams()
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    transport = TraceRegisterTransport(params)
    transport.words[CtrlWords.LAYOUT_ID] ^= 1
    session = _session(document, params, transport)

    with pytest.raises(RuntimeError, match="geometry/layout mismatch"):
        session.clear_host_config()
    assert transport.write_batches == []


def test_layout_mismatch_close_revokes_without_geometry_dependent_writes():
    params = StreamerParams()
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    transport = TraceRegisterTransport(params)
    transport.words[CtrlWords.LAYOUT_ID] ^= 1
    lease = MemoryDeviceLease()
    session = DeployedStreamerSession(
        transport,
        device_lease=lease,
        deployed_target=document.target,
        params=params,
        clock_hz=50e6,
    ).start()

    with pytest.raises(RuntimeError, match="geometry/layout mismatch"):
        session.check_register_layout()
    session.close()

    assert transport.write_batches == []
    assert not lease.acquired


def test_bringup_enters_acknowledged_safe_before_clearing_live_configuration():
    params = StreamerParams()
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    transport = TraceRegisterTransport(params)
    session = _session(document, params, transport)

    session.clear_host_config()

    assert transport.write_batches[0] == (
        (CtrlWords.STATUS, STATUS_ERROR),
        (CtrlWords.COMMAND, 0),
        (CtrlWords.COMMAND, CMD_SAFE),
    )
    assert session.state == "SAFE"


def test_safe_requires_a_bounded_stable_status_acknowledgement():
    class StuckSafeTransport(TraceRegisterTransport):
        def write_words(self, rows, *, stop=None, deadline=None):
            rows = tuple(rows)
            for address, value in rows:
                self.words[address] = value
                if address == CtrlWords.STATUS:
                    self.status = value
            self.write_batches.append(rows)

    params = StreamerParams()
    document = load_pulse_document(ROOT / "pulses" / "imaging_template.json")
    transport = StuckSafeTransport(params)
    session = DeployedStreamerSession(
        transport,
        device_lease=MemoryDeviceLease(),
        deployed_target=document.target,
        params=params,
        clock_hz=50e6,
        action_timeout=0.01,
        terminal_poll_interval=0.0001,
    ).start()
    session.check_register_layout()

    with pytest.raises(TimeoutError, match="acknowledge SAFE"):
        session.safe_state()
    assert session.state == "SAFE_FAILED"


def test_failed_safe_command_never_claims_safe_state():
    class FailingSafeTransport(TraceRegisterTransport):
        fail_safe = False

        def write_words(self, rows, *, stop=None, deadline=None):
            rows = tuple(rows)
            if self.fail_safe and (CtrlWords.COMMAND, CMD_SAFE) in rows:
                raise OSError("SAFE wire write failed")
            super().write_words(rows, stop=stop, deadline=deadline)

    params = StreamerParams()
    document, artifact = _scan_artifact(params, count=2)
    transport = FailingSafeTransport(params)
    session = _session(document, params, transport)
    session.prepare(artifact)
    session.fire(artifact)
    transport.fail_safe = True

    with pytest.raises(OSError, match="SAFE wire write failed"):
        session.safe_state()
    assert session.state == "SAFE_FAILED"
    assert session.snapshot()["prepared_artifact_digest"] is None


def test_closed_axi_transport_cannot_implicitly_restart(tmp_path):
    calls = []

    def execute(lines, action, timeout):
        calls.append(action)
        return "ok\n"

    transport = VivadoAxiRegisterTransport(
        state_dir=tmp_path,
        tcl_executor=execute,
    )
    transport.write_words(((1, 2),))
    transport.close()
    before = tuple(calls)

    with pytest.raises(RuntimeError, match="transport is closed"):
        transport.write_words(((1, 3),))
    assert tuple(calls) == before


def test_closed_session_revokes_prepare_without_hardware_io():
    params = StreamerParams()
    document, artifact = _scan_artifact(params, count=2)
    transport = TraceRegisterTransport(params)
    session = _session(document, params, transport)
    session.close()
    before = tuple(transport.write_batches)

    with pytest.raises(RuntimeError, match="session is closed"):
        session.prepare(artifact)
    assert tuple(transport.write_batches) == before


def test_axi_and_uart_processes_share_one_exclusive_device_lease(tmp_path):
    path = tmp_path / "one-physical-streamer.lock"
    first = InterprocessDeviceLease(path)
    second = InterprocessDeviceLease(path)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="already owned"):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()


def test_device_lease_is_exclusive_across_processes(tmp_path):
    path = tmp_path / "subprocess-streamer.lock"
    lease = InterprocessDeviceLease(path)
    script = (
        "import sys\n"
        "from zlc_pulse.transport import InterprocessDeviceLease\n"
        "lease = InterprocessDeviceLease(sys.argv[1])\n"
        "try:\n"
        "    lease.acquire()\n"
        "except RuntimeError:\n"
        "    raise SystemExit(23)\n"
        "lease.release()\n"
    )
    lease.acquire()
    try:
        blocked = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            cwd=ROOT,
            check=False,
            timeout=5.0,
        )
        assert blocked.returncode == 23
    finally:
        lease.release()
    admitted = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=ROOT,
        check=False,
        timeout=5.0,
    )
    assert admitted.returncode == 0


@pytest.mark.parametrize("token", ("XXXXXXXX", "0x12XZ", "123456789", ""))
def test_axi_read_rejects_unknown_or_malformed_logic_values(token):
    with pytest.raises(RuntimeError, match="no DATA|non-binary DATA"):
        VivadoAxiRegisterTransport._parse_read(f"ZLCDATA {token}\n", "ZLCDATA")


class WrongSequenceUartLink:
    def open(self):
        pass

    def close(self):
        pass

    @staticmethod
    def exchange(request, *, deadline, stop=None):
        return framing.encode_reply((request[3] + 1) & 0xFF, framing.ST_OK, (7,))

    @staticmethod
    def write_batch(requests, *, deadline, stop=None):
        return [
            framing.encode_reply((request[3] + 1) & 0xFF, framing.ST_OK)
            for request in requests
        ]


class RecordingUartLink:
    def __init__(self):
        self.batches = []

    def open(self):
        pass

    def close(self):
        pass

    @staticmethod
    def exchange(request, *, deadline, stop=None):
        return framing.encode_reply(request[3], framing.ST_OK, (0,))

    def write_batch(self, requests, *, deadline, stop=None):
        requests = tuple(requests)
        self.batches.append(requests)
        return [
            framing.encode_reply(request[3], framing.ST_OK)
            for request in requests
        ]


def test_uart_scan_bank_rearm_is_a_separate_acknowledged_phase(tmp_path):
    link = RecordingUartLink()
    transport = UartRegisterTransport(state_dir=tmp_path, link=link)

    transport.rewrite_scan_bank(
        unarmed_bank_ready=0b10,
        bank_words=((100, 7), (101, 8)),
        chunk_word=CtrlWords.BANK0_CHUNK,
        chunk_index=2,
        rearmed_bank_ready=0b11,
    )

    assert len(link.batches) == 4
    addresses = [
        int.from_bytes(batch[0][4:8], "little")
        for batch in link.batches
    ]
    assert addresses == [
        CtrlWords.BANK_READY,
        100,
        CtrlWords.BANK0_CHUNK,
        CtrlWords.BANK_READY,
    ]


def test_uart_transport_rejects_stale_or_reordered_sequence_replies(tmp_path):
    transport = UartRegisterTransport(
        state_dir=tmp_path,
        link=WrongSequenceUartLink(),
    )
    with pytest.raises(UartError, match="write reply sequence"):
        transport.write_words(((1, 2),))
    with pytest.raises(UartError, match="read reply sequence"):
        transport.read_word(1)


def test_uart_absolute_deadline_includes_waiting_for_the_io_owner(tmp_path):
    transport = UartRegisterTransport(
        state_dir=tmp_path,
        link=WrongSequenceUartLink(),
    )
    held = threading.Event()
    release = threading.Event()

    def hold_io_owner():
        with transport._lock:
            held.set()
            assert release.wait(1.0)

    owner = threading.Thread(target=hold_io_owner)
    owner.start()
    assert held.wait(1.0)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="I/O owner"):
            transport.read_word(1, deadline=started + 0.02)
    finally:
        release.set()
        owner.join(1.0)

    assert not owner.is_alive()
    assert time.monotonic() - started < 0.08


def test_transport_modules_contain_no_legacy_or_bitstream_programming_surface():
    transport_root = ROOT / "zlc_pulse" / "transport"
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in transport_root.glob("*.py")
    )
    for forbidden in (
        "program_on_start",
        "program_hw_devices",
        "RuntimeSequenceProgram",
        "pack_program(",
        "scan_repeats",
        "supports_stream_refill",
        "UartBridgeModel",
        "Zou_lab_control",
    ):
        assert forbidden not in text
