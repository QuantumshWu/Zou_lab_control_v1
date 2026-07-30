from __future__ import annotations

from dataclasses import replace
import threading
import time

import pytest

from Zou_lab_control.api import PulseRunResult, WorkspacePaths, connect
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.devices.sequencer.application import PulseRunRequest
from zlc_neutral_atom.runtime.run import CancelOutcome, RunState
from zlc_pulse import (
    DAC_OFFSET_BINARY,
    FIELD_DURATION,
    PORT_CLOCK,
    PORT_DAC,
    PORT_DIGITAL,
    AnalogStep,
    ApiParameter,
    FrozenScanTable,
    OutputDelay,
    PulseDocument,
    PulseExecutionForm,
    PulseFieldRef,
    PulsePeriod,
    PulsePortSpec,
    PulseTarget,
    RepeatRegion,
    ScanParameter,
    save_pulse_document,
)
from zlc_workbench.pulse_editor.session import PulseEditorSession


def _target() -> PulseTarget:
    return PulseTarget(
        ("ttl_lane", "dac_bit_0", "dac_bit_1", "dac_clock_lane"),
        (
            PulsePortSpec(
                "ttl",
                PORT_DIGITAL,
                ("ttl_lane",),
                "TTL",
                None,
                1,
                "binary",
                0,
                None,
            ),
            PulsePortSpec(
                "dac",
                PORT_DAC,
                ("dac_bit_0", "dac_bit_1"),
                "DAC",
                0,
                2,
                DAC_OFFSET_BINARY,
                2,
                "dac_clock",
            ),
            PulsePortSpec(
                "dac_clock",
                PORT_CLOCK,
                ("dac_clock_lane",),
                "DAC clock",
                None,
                1,
                "binary",
                0,
                None,
            ),
        ),
    )


def _document() -> PulseDocument:
    target = _target()
    return PulseDocument(
        "timeline proof",
        target,
        10.0,
        (
            PulsePeriod(
                "on",
                100,
                "ns",
                "on",
                (1, 0, 0, 0),
                (AnalogStep("dac", "edge", 1),),
            ),
            PulsePeriod(
                "ramp",
                100,
                "ns",
                "ramp",
                (0, 0, 0, 0),
                (AnalogStep("dac", "ramp", -1),),
            ),
        ),
        visible_ports=("ttl", "dac"),
        delays=(OutputDelay("ttl", 20, "ns"),),
        repeat=RepeatRegion("on", "ramp", 2),
    )


def _rekeyed_target(target: PulseTarget) -> PulseTarget:
    keys = {port.key: f"online_{port.key}" for port in target.ports}
    return PulseTarget(
        target.raw_lanes,
        tuple(
            replace(
                port,
                key=keys[port.key],
                label=f"Online {port.label}",
                latch_clock=(
                    None if port.latch_clock is None else keys[port.latch_clock]
                ),
            )
            for port in target.ports
        ),
    )


def test_preview_is_exact_for_authored_pass_repeat_bracket_and_dac_ramp():
    document = _document()
    _revision, timeline = PulseEditorSession(document).preview()

    # The editor projects one authored pass.  Repeat remains explicit as an
    # annotation instead of duplicating periods and waveforms in the picture.
    assert timeline.logical_duration_ticks == 20
    assert timeline.duration_ticks == 22
    assert timeline.reference_label == "compiled static pulse"
    digital, dac = timeline.rows
    assert [
        (item.start_tick, item.stop_tick, item.start_value, item.stop_value)
        for item in digital.segments
    ] == [
        (0, 2, 0, 0),
        (2, 12, 1, 1),
        (12, 22, 0, 0),
    ]
    assert [
        (item.start_tick, item.stop_tick, item.start_value, item.stop_value)
        for item in dac.segments
    ] == [
        (0, 16, 1, 1),
        (16, 22, 0, 0),
    ]
    assert [item.label for item in timeline.annotations if item.kind == "period"] == [
        "on",
        "ramp",
    ]
    assert [item.label for item in timeline.annotations if item.kind == "repeat"] == [
        "×2"
    ]


def test_scan_preview_uses_visible_nominal_values_not_the_first_scan_row():
    base = _document()
    parameter = ScanParameter(
        "on_duration",
        PulseFieldRef(FIELD_DURATION, "on"),
        "on duration",
        "ns",
    )
    document = replace(
        base,
        scan_parameters=(parameter,),
        scan_table=FrozenScanTable(("on_duration",), ((200,),)),
    )

    _revision, timeline = PulseEditorSession(document).preview()

    assert timeline.reference_label == "nominal scan/API reference"
    assert timeline.logical_duration_ticks == 20


def test_hardware_run_retains_authored_api_intent_and_explicit_values():
    document = _document()
    duration = PulseFieldRef(FIELD_DURATION, "on")
    unresolved = replace(
        document,
        api_parameters=(ApiParameter("on_duration", duration, "ns"),),
    )
    reference = DeviceRef("installation", "runtime", "sequencer")

    with pytest.raises(ValueError, match="exactly cover"):
        PulseRunRequest(
            unresolved,
            PulseExecutionForm.STATIC_ONCE,
            reference,
            1.0,
        )

    request = PulseRunRequest(
        unresolved,
        PulseExecutionForm.STATIC_ONCE,
        reference,
        1.0,
        (("on_duration", 150),),
    )
    assert request.document is unresolved
    assert request.document.api_parameters == unresolved.api_parameters
    assert request.api_values == (("on_duration", 150),)


def test_editor_save_detects_an_external_current_document_change(tmp_path):
    document = _document()
    session = PulseEditorSession(document)
    path = session.save(tmp_path / "pulse.json")
    session.replace_document(replace(document, name="local edit"))
    save_pulse_document(replace(document, name="external edit"), path)

    with pytest.raises(RuntimeError, match="changed on disk"):
        session.save()
    path.unlink()
    with pytest.raises(RuntimeError, match="changed on disk"):
        session.save()


def test_editor_normalizes_the_checked_and_written_save_path(tmp_path):
    original = _document()
    path = save_pulse_document(original, tmp_path / "collision")
    before = path.read_bytes()
    session = PulseEditorSession(replace(original, name="must not overwrite"))

    with pytest.raises(FileExistsError, match="already exists"):
        session.save(tmp_path / "collision")

    assert path.read_bytes() == before


def test_online_target_rebind_is_dirty_against_the_real_disk_baseline(tmp_path):
    path = save_pulse_document(_document(), tmp_path / "rebind")
    session = PulseEditorSession.load(path)
    assert not session.dirty

    session.bind_target(_rekeyed_target(session.document.target))
    assert session.dirty
    session.save()

    assert not session.dirty
    assert PulseEditorSession.load(path).document.target == session.document.target


def test_editor_serializes_concurrent_saves(
    tmp_path,
    monkeypatch,
):
    import zlc_workbench.pulse_editor.session as pulse_module

    session = PulseEditorSession(_document())
    entered = threading.Event()
    release = threading.Event()
    counter_lock = threading.Lock()
    active = 0
    maximum_active = 0
    calls = 0
    original_save = pulse_module.save_pulse_document

    def blocking_save(document, path):
        nonlocal active, maximum_active, calls
        with counter_lock:
            active += 1
            calls += 1
            maximum_active = max(maximum_active, active)
            call = calls
        try:
            if call == 1:
                entered.set()
                assert release.wait(2.0)
            return original_save(document, path)
        finally:
            with counter_lock:
                active -= 1

    monkeypatch.setattr(pulse_module, "save_pulse_document", blocking_save)
    errors = []

    def save():
        try:
            session.save(tmp_path / "serialized", overwrite=True)
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=save)
    second = threading.Thread(target=save)
    first.start()
    assert entered.wait(2.0)
    second.start()
    time.sleep(0.05)
    assert calls == 1 and maximum_active == 1
    release.set()
    first.join(2.0)
    second.join(2.0)

    assert not first.is_alive() and not second.is_alive() and not errors
    assert maximum_active == 1
    assert not session.dirty


def test_edit_during_save_remains_dirty(tmp_path, monkeypatch):
    import zlc_workbench.pulse_editor.session as pulse_module

    session = PulseEditorSession(_document())
    entered = threading.Event()
    release = threading.Event()
    original_save = pulse_module.save_pulse_document

    def blocking_save(document, path):
        entered.set()
        assert release.wait(2.0)
        return original_save(document, path)

    monkeypatch.setattr(pulse_module, "save_pulse_document", blocking_save)
    errors = []

    def save():
        try:
            session.save(tmp_path / "mid-edit")
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=save)
    worker.start()
    assert entered.wait(2.0)
    session.replace_document(replace(session.document, name="edited during save"))
    release.set()
    worker.join(2.0)

    assert not worker.is_alive() and not errors
    assert session.dirty


def test_notebook_pulse_run_and_cancelled_hold_share_the_runtime_safe_path(tmp_path):
    with connect(
        "virtual",
        workspace=WorkspacePaths.for_workspace(
            (tmp_path / "experiment").resolve()
        ),
    ) as experiment:
        target = experiment.pulse.target
        document = PulseDocument(
            "runtime pulse",
            target.target,
            target.time_step_ns,
            (
                PulsePeriod(
                    "idle",
                    100,
                    "ns",
                    "idle",
                    tuple(0 for _ in target.target.raw_lanes),
                ),
            ),
            visible_ports=tuple(
                port.key
                for port in target.target.ports
                if port.kind in (PORT_DIGITAL, PORT_DAC)
            ),
        )

        finite = experiment.pulse.request(document)
        assert finite.timeout_seconds is None
        descriptor = experiment.pulse.inspect(finite)
        assert descriptor.timeout_seconds == 30.0
        result = experiment.pulse.run(finite)
        assert isinstance(result, PulseRunResult)
        assert result.artifact_digest == descriptor.artifact_digest
        assert result.execution_form is PulseExecutionForm.STATIC_ONCE

        scan_parameter = ScanParameter(
            "idle_duration",
            PulseFieldRef(FIELD_DURATION, "idle"),
            "idle duration",
            "ns",
        )
        scan_document = replace(
            document,
            scan_parameters=(scan_parameter,),
            scan_table=FrozenScanTable(
                (scan_parameter.parameter_id,),
                ((100,), (200,)),
            ),
        )
        scan_result = experiment.pulse.run(
            experiment.pulse.request(
                scan_document,
                PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
            )
        )
        assert scan_result.execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_ONCE

        hold = experiment.pulse.start(
            experiment.pulse.request(
                document,
                PulseExecutionForm.CONTINUOUS_MONITOR,
            )
        )
        deadline = time.monotonic() + 2.0
        while hold.snapshot().phase != "holding-pulse":
            if time.monotonic() >= deadline:
                raise AssertionError("continuous pulse never reached its HOLD phase")
            time.sleep(0.005)
        assert hold.cancel("test Stop Pulse") is CancelOutcome.REQUESTED
        terminal = hold.wait(5.0)

    assert terminal.state is RunState.CANCELLED
