"""Current API_SLOT segmented scan contract and product oracles."""

from __future__ import annotations

from dataclasses import replace
import copy
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest

import Zou_lab_control.notebook as zlc
from fpga.pulse_streamer.host.image import DEFAULT_CLOCK_HZ
from zlc_data import ComponentValidity
from zlc_neutral_atom.bootstrap._sequencer_endpoint import (
    VirtualSequencerExecutionEndpoint,
)
from zlc_neutral_atom.bootstrap._virtual_hardware import (
    VirtualAtomArray,
    VirtualCamera,
    VirtualSequencer,
)
from zlc_neutral_atom.readout.sitemap import load_packaged_sitemap_pulse
from zlc_neutral_atom.runtime.ports import (
    DeviceBroker,
    SafetyOperation,
    SessionCloseCommand,
)
from zlc_neutral_atom.runtime.resources import (
    DeviceIdentityEvidenceKind,
    PhysicalDeviceIdentity,
    ResourceKey,
)
from zlc_neutral_atom.runtime.run import RunFailed
from zlc_neutral_atom.scan import (
    ApiSegmentTable,
    ApiSegmentedScanExecution,
    ApiSlotSegmentedProgram,
    AutonomousScanExecution,
    pulse_scan_program_from_tree,
    pulse_scan_program_to_tree,
)
from zlc_neutral_atom.scan.repository import ScanRepository
from zlc_neutral_atom.scan.lineage import (
    api_segmented_cell_schedule,
    execution_compiled_artifacts,
    pulse_scan_execution_from_tree,
    pulse_scan_execution_to_tree,
)
from zlc_neutral_atom.timing.pulse import (
    CompletePulseCommand,
    FinitePulseExecutionRequest,
    FirePulseCommand,
    PreparePulseCommand,
)
from zlc_pulse import (
    FrozenScanTable,
    PlaybackPulse,
    PulseExecutionForm,
    PulsePlayback,
    RepeatRegion,
    ScanParameter,
    compile_pulse_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
_SEGMENTATION_RATIONALE = (
    "Each point is a physically independent finite exposure; host gaps are allowed."
)
from zlc_neutral_atom.timing.segmented import (
    _host_boundary_delay_seconds,
    _wait_for_host_boundary,
)


def _api_document():
    """One readout edge, a nominal SCAN declaration, and two API columns."""

    document = load_packaged_sitemap_pulse()
    camera_port = next(
        port for port in document.target.ports if port.label == "emCCD"
    )
    trigger_index = document.target.raw_lanes.index(camera_port.lanes[0])
    segment = -1
    previous = 0
    periods = []
    for period in document.periods:
        states = list(period.states)
        current = int(states[trigger_index])
        if current and not previous:
            segment += 1
        states[trigger_index] = int(bool(current and segment == 1))
        periods.append(replace(period, states=tuple(states)))
        previous = current

    nominal = document.api_parameters[0]
    nominal_value = document.field_value(nominal.field)[0]
    nominal_scan = ScanParameter(
        "nominal_reference_before",
        nominal.field,
        "nominal reference before",
        nominal.unit,
    )
    return replace(
        document,
        name="api-segmented-readout",
        periods=tuple(periods),
        api_parameters=document.api_parameters[1:],
        scan_parameters=(nominal_scan,),
        scan_table=FrozenScanTable(
            (nominal_scan.parameter_id,),
            ((nominal_value,), (nominal_value + 1e-6,)),
        ),
        repeat=RepeatRegion(periods[0].period_id, periods[-1].period_id, 2),
    )


def _api_table(document, *, points: int = 3) -> ApiSegmentTable:
    columns = tuple(item.parameter_id for item in document.api_parameters)
    baseline = [
        document.field_value(parameter.field)[0]
        for parameter in document.api_parameters
    ]
    rows = []
    for index in range(points):
        row = list(baseline)
        # Vary only the post-readout reference duration.  This leaves the
        # calibrated readout event's physical context unchanged.
        row[-1] = float(baseline[-1]) + index * 1.2e-7
        rows.append(tuple(row))
    return ApiSegmentTable(columns, tuple(rows))


def _program(*, points: int = 3) -> ApiSlotSegmentedProgram:
    document = _api_document()
    return ApiSlotSegmentedProgram(
        document,
        _api_table(document, points=points),
        _SEGMENTATION_RATIONALE,
    )


def test_program_codec_is_strict_lossless_and_strips_nominal_scan_execution():
    program = _program()
    tree = pulse_scan_program_to_tree(program)

    assert pulse_scan_program_from_tree(tree) == program
    assert program.table.rows[1][-1] == 0.02000012
    nominal = program.document.scan_parameters[0]
    nominal_value = program.document.field_value(nominal.field)[0]
    point_documents = program.resolved_point_documents
    assert point_documents is program.resolved_point_documents
    for point_index, resolved in enumerate(point_documents):
        assert resolved.api_parameters == ()
        assert resolved.scan_parameters == ()
        assert resolved.scan_table is None
        assert resolved.scan_recipe is None
        assert resolved.repeat is None
        assert resolved.field_value(nominal.field)[0] == nominal_value
        varied = program.document.api_parameter_by_id[program.table.columns[-1]]
        assert resolved.field_value(varied.field)[0] == program.table.rows[point_index][-1]

    extra = copy.deepcopy(tree)
    extra["legacy_mode"] = "API_SLOT"
    with pytest.raises(ValueError, match="must contain exactly"):
        pulse_scan_program_from_tree(extra)
    rounded = copy.deepcopy(tree)
    rounded["table"]["rows"][1][-1] = round(program.table.rows[1][-1], 7)
    decoded = pulse_scan_program_from_tree(rounded)
    assert decoded.table.rows[1][-1] != program.table.rows[1][-1]
    assert pulse_scan_program_to_tree(decoded) == rounded


def _bind_endpoint(program: ApiSlotSegmentedProgram):
    document = program.resolved_point_documents[0]
    sequencer = VirtualSequencer(
        document.target,
        clock_hz=DEFAULT_CLOCK_HZ,
        sleep_scale=0,
    )
    endpoint = VirtualSequencerExecutionEndpoint(sequencer)
    broker = DeviceBroker()
    identity = PhysicalDeviceIdentity(
        "w3f-endpoint-sequencer",
        DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
        "w3f-endpoint-evidence",
        "w3f-endpoint-assets",
    )
    proof = broker.verify_identity(lambda: identity)
    binding = None

    def current_binding():
        assert binding is not None
        return binding

    binding = broker.bind(
        key=ResourceKey.parse("device/sequencer/w3f"),
        identity=proof,
        execute_command=lambda command: endpoint.execute_command(
            current_binding(), command
        ),
        capability_probe=lambda: endpoint.capability_probe(current_binding()),
        close_session=lambda command: endpoint.close_session(
            current_binding(), command
        ),
        interrupt_operations={SafetyOperation.SAFE_STATE: endpoint.interrupt},
    )
    capability = broker.verify_capability(binding).snapshot
    artifact = compile_pulse_artifact(
        document,
        clock_hz=DEFAULT_CLOCK_HZ,
        execution_form=PulseExecutionForm.STATIC_ONCE,
        trigger_channels=("ch11",),
        live_target=document.target,
    )
    request = FinitePulseExecutionRequest(document, artifact)
    return sequencer, endpoint, broker, binding, capability, request


def _camera_with_one_trigger_source(source, *, timeout: float = 0.05):
    """Arm one target-owned camera around a deliberately controlled iterator."""

    atoms = object.__new__(VirtualAtomArray)
    atoms.iter_frames = source
    sequencer = VirtualSequencer(
        _api_document().target,
        clock_hz=DEFAULT_CLOCK_HZ,
        sleep_scale=0,
    )
    camera = VirtualCamera(
        atoms,
        sequencer,
        capture_trigger_channels=("ch11",),
        exposure=1e-6,
        timeout=timeout,
    )
    playback = PulsePlayback(
        "one-camera-trigger",
        (PlaybackPulse("ch11", 0.0, 1e-6),),
        1e-6,
        1e-6,
        False,
        trigger_channels=("ch11",),
    )
    camera.arm(1, max_inflight_frames=1)
    camera._on_fire(playback)
    return camera, sequencer


def _frame() -> np.ndarray:
    return np.zeros((2, 2), dtype=np.uint16)


def test_virtual_camera_rejects_source_error_after_expected_final_frame():
    def source(*_args, **_kwargs):
        yield _frame()
        raise RuntimeError("late finite-source failure")

    camera, sequencer = _camera_with_one_trigger_source(source)
    try:
        with pytest.raises(RuntimeError, match="virtual camera source failed") as caught:
            camera.read_frame_records(1, timeout=0.5, exact=True)
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert str(caught.value.__cause__) == "late finite-source failure"
    finally:
        camera.close()
        sequencer.close()


def test_virtual_camera_rejects_extra_frame_after_expected_final_frame():
    def source(*_args, **_kwargs):
        yield _frame()
        yield _frame()

    camera, sequencer = _camera_with_one_trigger_source(source)
    try:
        with pytest.raises(RuntimeError, match="virtual camera source failed") as caught:
            camera.read_frame_records(1, timeout=0.5, exact=True)
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert "exceeded the expected trigger count" in str(caught.value.__cause__)
    finally:
        camera.close()
        sequencer.close()


def test_virtual_camera_post_frame_validation_honors_timeout_and_never_hangs():
    release = threading.Event()

    def source(*_args, **_kwargs):
        yield _frame()
        release.wait()

    camera, sequencer = _camera_with_one_trigger_source(source, timeout=0.05)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="source validation timed out"):
            camera.read_frame_records(1, timeout=0.03, exact=True)
        assert time.monotonic() - started < 0.5
        with pytest.raises(TimeoutError, match="producer did not join"):
            camera.finish_record_capture()
    finally:
        release.set()
        worker = camera._worker
        if worker is not None:
            worker.join(0.5)
        camera.finish_record_capture()
        camera.close()
        sequencer.close()


def test_virtual_camera_post_frame_validation_honors_cancellation():
    release = threading.Event()
    cancelled = threading.Event()

    def source(*_args, **_kwargs):
        yield _frame()
        release.wait()

    camera, sequencer = _camera_with_one_trigger_source(source)
    timer = threading.Timer(0.01, cancelled.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="source validation was cancelled"):
            camera.read_frame_records(
                1,
                timeout=0.5,
                stop=cancelled,
                exact=True,
            )
        assert time.monotonic() - started < 0.25
    finally:
        timer.cancel()
        release.set()
        worker = camera._worker
        if worker is not None:
            worker.join(0.5)
        camera.finish_record_capture()
        camera.close()
        sequencer.close()


def test_endpoint_allows_completed_same_run_successor_but_fences_cross_run():
    sequencer, endpoint, broker, binding, capability, request = _bind_endpoint(
        _program(points=2)
    )

    def complete(session_id: str, run_id: str):
        endpoint.execute_command(
            binding,
            PreparePulseCommand(
                session_id,
                run_id,
                request,
                capability.capability_fingerprint,
                2.0,
            ),
        )
        endpoint.execute_command(
            binding,
            FirePulseCommand(session_id, request.artifact_digest),
        )
        return endpoint.execute_command(
            binding,
            CompletePulseCommand(session_id, request.artifact_digest, 2.0),
        )

    complete("segment-0", "flat-run")
    complete("segment-1", "flat-run")
    with pytest.raises(RuntimeError, match="cannot cross runs before cleanup"):
        endpoint.execute_command(
            binding,
            PreparePulseCommand(
                "other-run-segment",
                "other-run",
                request,
                capability.capability_fingerprint,
                2.0,
            ),
        )
    closed = endpoint.close_session(
        binding,
        SessionCloseCommand("segment-1", 2.0),
    )
    assert closed.is_terminal
    assert sequencer.snapshot()["state"] == "safe"
    broker.shutdown()
    sequencer.close()


def test_camera_boundary_delay_is_derived_before_fire():
    _sequencer, _endpoint, broker, _binding, _capability, request = _bind_endpoint(
        _program(points=2)
    )
    try:
        artifact = request.artifact
        schedule = next(
            item for item in artifact.trigger_schedules if item.channel == "ch11"
        )
        pulse_binding = type(
            "BoundaryBinding",
            (),
            {"compiled_artifact": artifact, "trigger_schedule": schedule},
        )()
        descriptor = type(
            "BoundaryDescriptor",
            (),
            {"pulse_binding": pulse_binding},
        )()
        hardware_elapsed = (
            artifact.target_ir.duration_seconds
            - int(schedule.ticks_from_run_start[0]) / artifact.target_ir.clock_hz
            + artifact.max_configured_output_delay_ticks
            / artifact.target_ir.clock_hz
            + int(schedule.ticks_from_run_start[0]) / artifact.target_ir.clock_hz
        )
        assert _host_boundary_delay_seconds(
            descriptor,
            descriptor,
            hardware_elapsed + 0.01,
        ) == pytest.approx(0.01)

        class BoundaryContext:
            deadline = time.monotonic() + 1.0
            checkpoints = 0

            def checkpoint(self):
                self.checkpoints += 1

        context = BoundaryContext()
        started = time.monotonic()
        _wait_for_host_boundary(context, started + 0.015)
        assert time.monotonic() - started >= 0.014
        assert context.checkpoints >= 2
    finally:
        broker.shutdown()
        _sequencer.close()


def _run_api_segmented_virtual_product(workspace: Path) -> None:
    from zlc_neutral_atom.bootstrap._installation import _InstallationRuntime

    program = _program()
    run_names: list[str] = []
    armed_frames: list[int] = []
    prepared_forms: list[PulseExecutionForm] = []
    prepared_sources: list[str] = []
    fires: list[str] = []

    real_start = _InstallationRuntime.start
    real_arm = VirtualCamera.arm
    real_prepare = VirtualSequencer.prepare_compiled_playback
    real_fire = VirtualSequencer.fire_compiled_playback

    def record_start(runtime, plan):
        run_names.append(plan.name)
        return real_start(runtime, plan)

    def record_arm(camera, frames, **kwargs):
        armed_frames.append(frames)
        return real_arm(camera, frames, **kwargs)

    def record_prepare(sequencer, artifact, playback):
        prepared_forms.append(artifact.execution_form)
        prepared_sources.append(artifact.source_document_digest)
        return real_prepare(sequencer, artifact, playback)

    def record_fire(sequencer, artifact_digest):
        fires.append(artifact_digest)
        return real_fire(sequencer, artifact_digest)

    with (
        patch.object(_InstallationRuntime, "start", record_start),
        patch.object(VirtualCamera, "arm", record_arm),
        patch.object(VirtualSequencer, "prepare_compiled_playback", record_prepare),
        patch.object(VirtualSequencer, "fire_compiled_playback", record_fire),
    ):
        real_repository_init = ScanRepository.__init__
        scan_repositories: list[ScanRepository] = []

        def recording_repository_init(repository, root, **kwargs):
            real_repository_init(repository, root, **kwargs)
            scan_repositories.append(repository)

        with patch.object(ScanRepository, "__init__", recording_repository_init):
            exp = zlc.connect("virtual", repository=workspace)
        try:
            assert len(scan_repositories) == 1
            request = exp.readout.api_scan_request(
                program.document,
                api_table=program.table,
                segmentation_rationale=program.segmentation_rationale,
                timeout_seconds=20.0,
            )
            reference = exp.scan(request)
            artifact = exp.readout.load_scan(reference)
            data = exp.readout.materialize_scan(reference)
            expected_cells = program.repeat_count * len(program.table.rows)

            assert len(run_names) == 1
            assert armed_frames == [expected_cells]
            assert prepared_forms == [PulseExecutionForm.STATIC_ONCE] * expected_cells
            expected_point_sources = tuple(
                document.fingerprint
                for document in program.resolved_point_documents
            )
            assert tuple(prepared_sources) == expected_point_sources * program.repeat_count
            assert len(fires) == expected_cells
            assert data.values.shape == (2, 3, 96, 128)
            assert data.schema.point_axes == program.point_table.point_axes
            assert isinstance(artifact.execution, ApiSegmentedScanExecution)
            assert tuple(
                (segment.repeat_index, segment.point_storage_index)
                for segment in artifact.execution.segments
            ) == tuple(
                (repeat_index, point_index)
                for repeat_index in range(2)
                for point_index in range(3)
            )
            assert all(
                segment.evidence.compiled_artifact.execution_form
                is PulseExecutionForm.STATIC_ONCE
                for segment in artifact.execution.segments
            )
            assert len(
                {
                    segment.evidence.terminal.session_id
                    for segment in artifact.execution.segments
                }
            ) == expected_cells
            camera = artifact.execution.camera
            assert camera.event_count == expected_cells
            assert camera.terminal.session_id
            assert camera.terminal.produced_count == expected_cells
            assert camera.terminal.drained_count == expected_cells
            assert camera.arm_spec.digest == camera.terminal.capture_spec_fingerprint
            assert camera.capability.fingerprint == camera.terminal.capability_fingerprint
            camera.validate_dataset_provenance(artifact.provenance)
            with pytest.raises(ValueError, match="pulse terminal acknowledgement"):
                replace(
                    artifact.execution.segments[0].evidence.terminal,
                    session_id="x" * 9_000,
                )
            with pytest.raises(ValueError, match="capture terminal acknowledgement"):
                replace(camera.terminal, session_id="x" * 5_000)

            execution_tree = pulse_scan_execution_to_tree(artifact.execution)
            assert pulse_scan_execution_from_tree(
                execution_tree,
                artifact.execution.program,
                execution_compiled_artifacts(artifact.execution),
            ) == artifact.execution
            forged_count = copy.deepcopy(execution_tree)
            forged_count["camera"]["terminal"]["drained_count"] -= 1
            with pytest.raises(
                ValueError,
                match="camera terminal does not prove exact stop, drain, and join",
            ):
                pulse_scan_execution_from_tree(
                    forged_count,
                    artifact.execution.program,
                    execution_compiled_artifacts(artifact.execution),
                )
            for terminal_field, message in (
                ("capability_fingerprint", "camera capability lineage"),
                ("capture_spec_fingerprint", "camera arm-spec lineage"),
                ("settings_fingerprint", "camera settings lineage"),
            ):
                forged_digest = copy.deepcopy(execution_tree)
                forged_digest["camera"]["terminal"][terminal_field] = "0" * 64
                with pytest.raises(ValueError, match=message):
                    pulse_scan_execution_from_tree(
                        forged_digest,
                        artifact.execution.program,
                        execution_compiled_artifacts(artifact.execution),
                    )
            forged_schema = copy.deepcopy(execution_tree)
            forged_schema["camera"]["source_schema_fingerprint"] = "0" * 64
            forged_schema_execution = pulse_scan_execution_from_tree(
                forged_schema,
                artifact.execution.program,
                execution_compiled_artifacts(artifact.execution),
            )
            with pytest.raises(ValueError, match="camera source schema differs"):
                forged_schema_execution.camera.validate_source_schema(
                    artifact.source_dataset_schema
                )
            forged_schedule = copy.deepcopy(execution_tree)
            forged_schedule["camera"]["source_schedule_digest"] = "0" * 64
            forged_execution = pulse_scan_execution_from_tree(
                forged_schedule,
                artifact.execution.program,
                execution_compiled_artifacts(artifact.execution),
            )
            camera_schema = forged_execution.camera.validate_source_schema(
                artifact.source_dataset_schema
            )
            with pytest.raises(
                ValueError,
                match="camera source schedule differs from pulse execution",
            ):
                forged_execution.camera.require_schedule(
                    api_segmented_cell_schedule(
                        artifact.execution.program,
                        camera_schema,
                    ),
                    camera_schema,
                )

            run_names.clear()
            armed_frames.clear()
            prepared_forms.clear()
            prepared_sources.clear()
            fires.clear()
            autonomous_values = {
                parameter.parameter_id: program.document.field_value(parameter.field)[0]
                for parameter in program.document.api_parameters
            }
            autonomous_ref = exp.scan(
                exp.readout.scan_request(
                    program.document,
                    api_values=autonomous_values,
                    timeout_seconds=20.0,
                )
            )
            autonomous = exp.readout.load_scan(autonomous_ref)
            assert len(run_names) == 1
            assert armed_frames == [4]
            assert prepared_forms == [PulseExecutionForm.AUTONOMOUS_SCAN_ONCE]
            assert len(fires) == 1
            assert isinstance(autonomous.execution, AutonomousScanExecution)

            calibration_ref = exp.readout.sitemap(frames=6)
            occupancy_request = exp.readout.api_occupancy_scan_request(
                program.document,
                api_table=program.table,
                segmentation_rationale=program.segmentation_rationale,
                calibration_ref=calibration_ref,
                timeout_seconds=20.0,
            )
            from Zou_lab_control.notebook.facade import (
                _prepare_occupancy_scan_for_workbench,
            )
            from zlc_neutral_atom.runtime.pipeline import ExactDatasetPreviewSpec
            from zlc_workbench.progressive_scan import ExactDatasetLiveSlot

            rejected = _prepare_occupancy_scan_for_workbench(exp, occupancy_request)
            preview = ExactDatasetLiveSlot(
                ExactDatasetPreviewSpec(
                    rejected.source_schema.fingerprint,
                )
            )
            with pytest.raises(ValueError, match="FINAL-only"):
                rejected.start(preview)
            assert preview.terminal

            occupancy_ref = exp.scan(occupancy_request)
            occupancy = exp.readout.materialize_scan(occupancy_ref)
            occupancy_artifact = exp.readout.load_scan(occupancy_ref)
            assert occupancy.values.shape[:2] == (2, 3)
            assert occupancy.values.ndim == 3
            assert isinstance(occupancy.validity, ComponentValidity)
            assert occupancy.validity.mask.shape == occupancy.values.shape
            assert isinstance(
                occupancy_artifact.execution,
                ApiSegmentedScanExecution,
            )
            occupancy_camera = occupancy_artifact.execution.camera
            assert occupancy_artifact.provenance.derivation is not None
            assert (
                occupancy_artifact.provenance.derivation.root_input_span
                == occupancy_camera.source_event_span
            )
            occupancy_camera.validate_dataset_provenance(
                occupancy_artifact.provenance
            )

            fire_calls = 0
            final_commit_calls = 0

            def fail_second_fire(sequencer, artifact_digest):
                nonlocal fire_calls
                fire_calls += 1
                if fire_calls == 2:
                    raise RuntimeError("injected API segment FIRE failure")
                return real_fire(sequencer, artifact_digest)

            def forbidden_final_commit(*_args, **_kwargs):
                nonlocal final_commit_calls
                final_commit_calls += 1
                raise AssertionError("failed segmented run reached FINAL commit")

            with (
                patch.object(
                    VirtualSequencer,
                    "fire_compiled_playback",
                    fail_second_fire,
                ),
                patch.object(ScanRepository, "final_commit", forbidden_final_commit),
            ):
                with pytest.raises(RunFailed) as caught:
                    exp.scan(request)
            assert "injected API segment FIRE failure" in str(caught.value)
            assert final_commit_calls == 0
        finally:
            exp.close()


def test_api_segmented_virtual_direct_occupancy_failure_and_autonomous_product(
    tmp_path,
):
    code = (
        "from pathlib import Path; import runpy, sys; "
        "ns=runpy.run_path(sys.argv[1]); "
        "ns['_run_api_segmented_virtual_product'](Path(sys.argv[2]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=ROOT,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, (
        f"API segmented product subprocess failed ({completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
