"""Active E0 trigger-path qualification for one real camera connection.

The digest returned here is derived from one actually executed frozen pulse,
camera records, SDK terminal readback, and remote hardware terminal evidence.
Configuration text alone can never mint this capability.
"""

from __future__ import annotations

import math

from zlc_neutral_atom.devices.camera.contract import CameraAdapter, CameraFrameRecord
from zlc_pulse import (
    PORT_DIGITAL,
    PulseDocument,
    PulseExecutionForm,
    PulsePeriod,
    RemotePulseExecutionClient,
    compile_pulse_artifact,
)
from zlc_storage import canonical_digest, canonical_text


E0_TRIGGER_COUNT = 4
E0_TRIGGER_HIGH_SECONDS = 10e-6
E0_INTERVAL_MARGIN = 1.25


def _qualification_document(
    *,
    client: RemotePulseExecutionClient,
    trigger_lane: str,
    required_interval_seconds: float,
) -> PulseDocument:
    snapshot = client.snapshot()
    target = snapshot.target
    lane = canonical_text(trigger_lane, "E0 trigger lane")
    try:
        lane_index = target.raw_lanes.index(lane)
    except ValueError as exc:
        raise ValueError(f"E0 trigger lane {lane!r} is absent from the live target") from exc
    owner = next(port for port in target.ports if lane in port.lanes)
    if owner.kind != PORT_DIGITAL:
        raise ValueError("E0 trigger lane does not belong to a digital port")
    if owner.lanes != (lane,):
        raise ValueError("E0 qualification requires a one-lane camera trigger port")
    clock_hz = snapshot.clock_hz
    tick_ns = 1e9 / clock_hz
    high_ticks = max(1, math.ceil(E0_TRIGGER_HIGH_SECONDS * clock_hz))
    interval_ticks = max(
        high_ticks + 1,
        math.ceil(required_interval_seconds * E0_INTERVAL_MARGIN * clock_hz),
    )
    low_ticks = interval_ticks - high_ticks
    low = tuple(0 for _ in target.raw_lanes)
    high_values = list(low)
    high_values[lane_index] = 1
    high = tuple(high_values)
    periods: list[PulsePeriod] = [
        PulsePeriod("e0_initial_safe", tick_ns, "ns", "safe", low)
    ]
    for index in range(E0_TRIGGER_COUNT):
        periods.append(
            PulsePeriod(
                f"e0_trigger_{index}",
                high_ticks * tick_ns,
                "ns",
                "camera trigger",
                high,
            )
        )
        periods.append(
            PulsePeriod(
                f"e0_safe_{index}",
                low_ticks * tick_ns,
                "ns",
                "safe interval",
                low,
            )
        )
    return PulseDocument(
        name=f"E0 {lane} active trigger qualification",
        target=target,
        time_step_ns=tick_ns,
        periods=tuple(periods),
        visible_ports=(owner.key,),
    )


def _record_evidence(
    records: list[CameraFrameRecord],
    *,
    expected_shape: tuple[int, int],
    expected_dtype,
) -> list[dict[str, object]]:
    if len(records) != E0_TRIGGER_COUNT:
        raise RuntimeError("E0 camera drain count differs from the trigger schedule")
    if [record.source_ordinal for record in records] != list(range(E0_TRIGGER_COUNT)):
        raise RuntimeError("E0 camera records are missing or reordered")
    if any(
        record.image.shape != expected_shape or record.image.dtype != expected_dtype
        for record in records
    ):
        raise RuntimeError("E0 camera payload differs from its frozen working point")
    produced = [record.produced_count for record in records]
    if not all(value is None for value in produced):
        if produced != list(range(1, E0_TRIGGER_COUNT + 1)):
            raise RuntimeError("E0 camera produced counts contain a gap or reordering")
    for field in ("frame_stamp", "camera_stamp"):
        values = [getattr(record, field) for record in records]
        if all(value is None for value in values):
            continue
        if any(value is None for value in values):
            raise RuntimeError(f"E0 camera {field} availability changed during the run")
        if any(
            right != left + 1
            for left, right in zip(values[:-1], values[1:], strict=True)
        ):
            raise RuntimeError(f"E0 camera {field} has a gap or reordering")
    timestamps = [
        None
        if record.timestamp_seconds is None
        else record.timestamp_seconds * 1_000_000 + record.timestamp_microseconds
        for record in records
    ]
    if not all(value is None for value in timestamps):
        if any(value is None for value in timestamps):
            raise RuntimeError("E0 camera timestamp availability changed during the run")
        if any(
            right <= left
            for left, right in zip(timestamps[:-1], timestamps[1:], strict=True)
        ):
            raise RuntimeError("E0 camera timestamps are not strictly increasing")
    if not any(
        all(getattr(record, field) is not None for record in records)
        for field in ("frame_stamp", "camera_stamp")
    ):
        raise RuntimeError(
            "E0 exact qualification requires a hardware frame or camera stamp"
        )
    return [
        {
            "source_ordinal": record.source_ordinal,
            "produced_count": record.produced_count,
            "frame_stamp": record.frame_stamp,
            "camera_stamp": record.camera_stamp,
            "timestamp_seconds": record.timestamp_seconds,
            "timestamp_microseconds": record.timestamp_microseconds,
            "host_received_at_ns": record.host_received_at_ns,
            "shape": record.image.shape,
            "dtype": record.image.dtype.str,
        }
        for record in records
    ]


def qualify_external_trigger_path(
    *,
    client: RemotePulseExecutionClient,
    camera: CameraAdapter,
    trigger_lane: str,
) -> str:
    """Run one frozen FPGA program and return its runtime-derived digest."""

    if not isinstance(client, RemotePulseExecutionClient):
        raise TypeError("client must be RemotePulseExecutionClient")
    if not isinstance(camera, CameraAdapter):
        raise TypeError("camera must implement CameraAdapter")
    working_point = camera.capture_working_point()
    if working_point.capture_trigger_channels != (trigger_lane,):
        raise RuntimeError("camera working point is wired to another trigger lane")
    required_interval = working_point.required_external_trigger_interval_seconds
    if required_interval is None:
        raise RuntimeError(
            "camera has no hardware-read external trigger interval; E0 cannot qualify it"
        )
    document = _qualification_document(
        client=client,
        trigger_lane=trigger_lane,
        required_interval_seconds=required_interval,
    )
    snapshot = client.snapshot()
    artifact = compile_pulse_artifact(
        document,
        clock_hz=snapshot.clock_hz,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=(trigger_lane,),
        live_target=snapshot.target,
    )
    trigger_index = snapshot.target.raw_lanes.index(trigger_lane)
    if any(
        period.analog_steps
        or any(
            state and lane_index != trigger_index
            for lane_index, state in enumerate(period.states)
        )
        for period in document.periods
    ):
        raise RuntimeError("E0 source would disturb a non-trigger output")
    expected_bus_safe = tuple(
        port.safe_value
        for port in sorted(
            (port for port in snapshot.target.ports if port.kind == "dac"),
            key=lambda port: port.bus_index,
        )
    )
    if artifact.target_ir.bus_segments or artifact.target_ir.bus_safe_values != expected_bus_safe:
        raise RuntimeError("E0 artifact does not preserve every DAC safe value")
    schedule = artifact.trigger_schedules[0]
    if schedule.total != E0_TRIGGER_COUNT:
        raise RuntimeError("E0 compiled trigger count differs from its fixed request")
    minimum_ticks = schedule.minimum_interval_ticks
    if minimum_ticks is None or minimum_ticks / snapshot.clock_hz < required_interval:
        raise RuntimeError("E0 compiled trigger interval is below camera readback")
    reference = None
    armed = False
    terminal = None
    completion = None
    records: list[CameraFrameRecord] = []
    try:
        reference = client.prepare(artifact)
        camera.arm(
            E0_TRIGGER_COUNT,
            source_group_sizes=(E0_TRIGGER_COUNT,),
            buffer_frame_count=E0_TRIGGER_COUNT,
            timeout=camera.timeout,
        )
        armed = True
        client.fire(reference)
        records = list(
            camera.read_frame_records(
                E0_TRIGGER_COUNT,
                timeout=max(camera.timeout, E0_TRIGGER_COUNT * required_interval + 1.0),
                exact=True,
            )
        )
        completion = client.complete(
            reference,
            timeout=client.transport_timeout_seconds * 0.8,
        )
        terminal = camera.finish_record_capture()
        armed = False
    except BaseException as primary:
        if armed:
            try:
                camera.finish_record_capture()
            except BaseException as secondary:
                primary.add_note(f"E0 camera terminalization also failed: {secondary}")
        try:
            client.safe_state(timeout=client.transport_timeout_seconds * 0.8)
        except BaseException as secondary:
            primary.add_note(f"E0 sequencer SAFE also failed: {secondary}")
        raise
    assert completion is not None and terminal is not None
    expected_counts = dict(completion.expected_trigger_counts_from_completed_schedule)
    if expected_counts != {trigger_lane: E0_TRIGGER_COUNT}:
        raise RuntimeError("E0 remote completion trigger counts differ from the schedule")
    if terminal.produced_count != E0_TRIGGER_COUNT or not (
        terminal.source_stopped and terminal.no_more_frames and terminal.joined
    ):
        raise RuntimeError("E0 camera terminal record did not reconcile the full run")
    record_rows = _record_evidence(
        records,
        expected_shape=working_point.frame_shape_yx,
        expected_dtype=working_point.dtype,
    )
    return canonical_digest(
        {
            "contract": "zlc.real-camera-active-e0",
            "connection_generation": snapshot.connection_generation,
            "manifest_fingerprint": snapshot.manifest.fingerprint,
            "artifact_fingerprint": artifact.fingerprint,
            "trigger_schedule_fingerprint": schedule.fingerprint,
            "camera_settings_fingerprint": working_point.settings_fingerprint,
            "camera_records": record_rows,
            "camera_terminal": {
                "produced_count": terminal.produced_count,
                "source_stopped": terminal.source_stopped,
                "no_more_frames": terminal.no_more_frames,
                "joined": terminal.joined,
            },
            "pulse_terminal_fingerprint": completion.hardware_terminal.fingerprint,
            "pulse_tail_fingerprint": completion.post_terminal_tail.fingerprint,
            "expected_trigger_counts": completion.expected_trigger_counts_from_completed_schedule,
        }
    )


__all__ = ["qualify_external_trigger_path"]
