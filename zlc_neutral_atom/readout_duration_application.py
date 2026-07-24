"""Hardware-timed fidelity acquisition with one explicit API boundary per point."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import uuid
from typing import Callable

import numpy as np

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
    dataset_revision_ref_to_tree,
    expand_value_validity,
)
from zlc_neutral_atom.acquisition import (
    CameraAcquisitionMode,
    CameraCaptureSpec,
    CameraSample,
    freeze_camera_capture_spec,
)
from zlc_neutral_atom.readout.analysis import fit_bimodal
from zlc_neutral_atom.readout.calibration import (
    ReadoutModelKind,
    ResolvedCalibration,
    extract_readout_features,
)
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.coupled_measurements import (
    BoundReadoutDurationFidelity,
)
from zlc_neutral_atom.runtime.capture import (
    CameraExposureConfiguredAck,
    CapturePreparedAck,
    CaptureStartedAck,
    CaptureTerminalAck,
    CapturedPayloadAck,
    CompleteCaptureCommand,
    ConfigureCameraExposureCommand,
    PrepareCaptureCommand,
    ReadCaptureCommand,
    StartCaptureCommand,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.run import PostSafetyContext, RunContext, RunHandle, RunPlan
from zlc_neutral_atom.timing._coordination import run_cleanup_steps
from zlc_neutral_atom.timing.pulse import PulseSession, PulseTerminalAck
from zlc_storage import canonical_digest, sha256_text


@dataclass(frozen=True, slots=True)
class ReadoutDurationFidelityResult:
    """Fidelity curve plus the terminal evidence that admitted every point."""

    run_id: str
    snapshot: OwnedSnapshot
    calibration_ref: CalibrationArtifactRef
    model_kind: ReadoutModelKind
    site: int | None
    program_fingerprint: str
    capture_terminals: tuple[CaptureTerminalAck, ...]
    pulse_terminals: tuple[PulseTerminalAck, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind")
        sha256_text(self.program_fingerprint, "program_fingerprint")
        point_count = self.snapshot.block.schema.point_layout.storage_size
        values = (tuple(self.capture_terminals), tuple(self.pulse_terminals))
        if any(len(value) != point_count for value in values):
            raise ValueError("result evidence does not cover every duration point")
        if any(not isinstance(value, CaptureTerminalAck) for value in values[0]):
            raise TypeError("capture_terminals contain another value type")
        if any(not isinstance(value, PulseTerminalAck) for value in values[1]):
            raise TypeError("pulse_terminals contain another value type")
        object.__setattr__(self, "capture_terminals", values[0])
        object.__setattr__(self, "pulse_terminals", values[1])

    @property
    def identity(self) -> str:
        return canonical_digest(
            {
                "owner": "zlc_neutral_atom.readout-duration-fidelity",
                "run_id": self.run_id,
                "dataset_ref": dataset_revision_ref_to_tree(self.snapshot.ref),
                "calibration": self.calibration_ref.target_ref,
                "model_kind": self.model_kind.value,
                "site": self.site,
                "program": self.program_fingerprint,
            }
        )


@dataclass(slots=True)
class _PreparedReadoutDuration:
    exposure_lease_id: str
    exposure_lease_open: bool = False
    current_camera_session_id: str | None = None
    current_pulse: PulseSession | None = None


class PreparedReadoutDurationFidelity:
    """One immutable start command for the coupled duration Measurement."""

    __slots__ = ("_plan", "_start_run", "_started", "_lock")

    def __init__(self, plan: RunPlan, start_run: Callable[[RunPlan], RunHandle]) -> None:
        if not isinstance(plan, RunPlan):
            raise TypeError("plan must be RunPlan")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        self._plan = plan
        self._start_run = start_run
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> RunHandle:
        with self._lock:
            if self._started:
                raise RuntimeError("PreparedReadoutDurationFidelity is one-shot")
            self._started = True
        return self._start_run(self._plan)


def _validate_ack(ack: object, expected_type: type, session_id: str, binding) -> None:
    if not isinstance(ack, expected_type):
        raise TypeError(
            f"camera returned {type(ack).__name__}, expected {expected_type.__name__}"
        )
    if ack.session_id != session_id:
        raise RuntimeError("camera acknowledgement belongs to another session")
    if ack.binding_instance_id != binding.binding_instance_id:
        raise RuntimeError("camera acknowledgement belongs to another binding")


def _point_samples(model, site: int | None, frames: list[CameraSample]) -> np.ndarray:
    samples: list[np.ndarray] = []
    usable_sites = np.asarray(model.usable_sites.mask, dtype=bool)
    for frame in frames:
        signals = extract_readout_features(model.feature, frame.image)
        valid = np.asarray(
            expand_value_validity(signals.validity, signals.schema), dtype=bool
        ) & usable_sites
        values = np.asarray(signals.values, dtype=float)
        if site is None:
            samples.append(values[valid])
        elif valid[site]:
            samples.append(values[site : site + 1])
    return np.concatenate(samples) if samples else np.empty((0,), dtype=float)


def _require_hardware_trigger_spacing(
    pulse_request,
    trigger_channel: str,
    required_interval_seconds: float,
) -> None:
    schedules = tuple(
        value
        for value in pulse_request.artifact.trigger_schedules
        if value.channel == trigger_channel
    )
    if len(schedules) != 1:
        raise RuntimeError("frozen point lost its single camera trigger schedule")
    ticks = schedules[0].ticks_from_run_start
    clock_hz = pulse_request.artifact.target_ir.clock_hz
    if any(
        (following - previous) / clock_hz + 1e-12
        < required_interval_seconds
        for previous, following in zip(ticks, ticks[1:])
    ):
        raise RuntimeError(
            "compiled hardware trigger spacing is shorter than the camera "
            "working-point readback permits"
        )


def _result_snapshot(
    run_id: str,
    program_fingerprint: str,
    durations: tuple[float, ...],
    fidelities: tuple[float, ...],
    validity: tuple[bool, ...],
) -> OwnedSnapshot:
    point_axis = AxisSpec(
        AxisId("readout_duration.duration"),
        "Detection time",
        SCAN_POINT,
        len(durations),
        durations,
        "s",
    )
    schema = DatasetSchema(
        AxisSpec(AxisId("readout_duration.repeat"), "repeat", REPEAT, 1, (0,)),
        (point_axis,),
        PointLayout.rect_c((len(durations),)),
        ValueSchema((), ValidityContract.value(), np.dtype("<f8"), "fidelity"),
    )
    identity = canonical_digest(
        {
            "owner": "zlc_neutral_atom.readout-duration-fidelity-result",
            "run_id": run_id,
            "program": program_fingerprint,
            "duration_seconds": durations,
        }
    )
    block = DataBlock(
        BlockId(f"readout-duration-{identity[:20]}"),
        DatasetRevision(0),
        np.asarray(fidelities, dtype="<f8").reshape((1, len(durations))),
        CellValidity(np.asarray(validity, dtype=bool).reshape((1, len(durations)))),
        schema,
    )
    generation = StreamGenerationId(f"readout-duration-{identity}")
    return OwnedSnapshot(block.ref(generation), block)


def prepare_readout_duration_fidelity(
    bound: BoundReadoutDurationFidelity,
    calibration: ResolvedCalibration,
    *,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedReadoutDurationFidelity:
    if not isinstance(bound, BoundReadoutDurationFidelity):
        raise TypeError("bound must be BoundReadoutDurationFidelity")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    calibration._require_authority()
    if calibration.reference != bound.request.calibration_ref:
        raise ValueError("calibration differs from the bound request")
    model = calibration.artifact.select_model(bound.request.model_kind)
    pulse_port = bound.pulse_port
    camera_port = bound.camera_port
    camera_timeout = camera_port.capability.max_blocking_call_seconds
    camera_device = camera_port.device

    def preflight(_context: RunContext) -> _PreparedReadoutDuration:
        return _PreparedReadoutDuration(uuid.uuid4().hex)

    def execute(
        context: RunContext,
        prepared: _PreparedReadoutDuration,
    ) -> ReadoutDurationFidelityResult:
        device = context.device(camera_device.key)
        applied_durations: list[float] = []
        fidelities: list[float] = []
        valid_cells: list[bool] = []
        capture_terminals: list[CaptureTerminalAck] = []
        pulse_terminals: list[PulseTerminalAck] = []

        for requested, pulse_request in zip(
            bound.request.duration_seconds,
            bound.point_requests,
            strict=True,
        ):
            context.checkpoint()
            prepared.exposure_lease_open = True
            applied = device.execute(
                ConfigureCameraExposureCommand(
                    prepared.exposure_lease_id,
                    requested,
                    camera_port.capability.settings_fingerprint,
                )
            )
            _validate_ack(
                applied,
                CameraExposureConfiguredAck,
                prepared.exposure_lease_id,
                camera_device,
            )
            if not np.isclose(
                applied.applied_exposure_seconds,
                requested,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "camera exposure readback differs from the frozen API row"
                )
            _require_hardware_trigger_spacing(
                pulse_request,
                bound.trigger_channel,
                applied.required_external_trigger_interval_seconds,
            )

            capture_session_id = uuid.uuid4().hex
            prepared.current_camera_session_id = capture_session_id
            capture_spec = freeze_camera_capture_spec(
                CameraCaptureSpec(
                    CameraAcquisitionMode.EXTERNAL_TRIGGERED,
                    bound.request.shots,
                    tuple(1 for _ in range(bound.request.shots)),
                    applied.settings_fingerprint,
                )
            )
            ack = device.execute(
                PrepareCaptureCommand(
                    capture_session_id,
                    capture_spec.payload,
                    capture_spec.owner_fingerprint,
                    capture_spec.digest,
                    applied.capability_fingerprint,
                    applied.settings_fingerprint,
                    bound.request.shots,
                    camera_timeout,
                )
            )
            _validate_ack(ack, CapturePreparedAck, capture_session_id, camera_device)
            ack = device.execute(StartCaptureCommand(capture_session_id, camera_timeout))
            _validate_ack(ack, CaptureStartedAck, capture_session_id, camera_device)

            pulse = pulse_port.open_session(pulse_request)
            prepared.current_pulse = pulse
            pulse.prepare(context)
            pulse.fire(context)
            frames: list[CameraSample] = []
            for _ in range(bound.request.shots):
                context.checkpoint()
                read = device.execute(
                    ReadCaptureCommand(capture_session_id, camera_timeout)
                )
                _validate_ack(
                    read, CapturedPayloadAck, capture_session_id, camera_device
                )
                if not isinstance(read.payload, CameraSample):
                    raise TypeError("readout-duration camera returned another payload")
                frames.append(read.payload)
            pulse_terminal = pulse.complete(context)
            capture_terminal = device.execute(
                CompleteCaptureCommand(
                    capture_session_id,
                    bound.request.shots,
                    camera_timeout,
                )
            )
            _validate_ack(
                capture_terminal,
                CaptureTerminalAck,
                capture_session_id,
                camera_device,
            )
            prepared.current_camera_session_id = None

            samples = _point_samples(model, bound.request.site, frames)
            fit = fit_bimodal(samples)
            valid = bool(fit.ok and math.isfinite(float(fit.fidelity)))
            applied_durations.append(applied.applied_exposure_seconds)
            fidelities.append(float(fit.fidelity) if valid else float("nan"))
            valid_cells.append(valid)
            capture_terminals.append(capture_terminal)
            pulse_terminals.append(pulse_terminal)

        return ReadoutDurationFidelityResult(
            context.run_id.value,
            _result_snapshot(
                context.run_id.value,
                bound.program.fingerprint,
                tuple(applied_durations),
                tuple(fidelities),
                tuple(valid_cells),
            ),
            calibration.reference,
            model.kind,
            bound.request.site,
            bound.program.fingerprint,
            tuple(capture_terminals),
            tuple(pulse_terminals),
        )

    def cleanup(
        context: RunContext,
        prepared: _PreparedReadoutDuration | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is None:
            return run_cleanup_steps(
                lambda: pulse_port.verify_idle(context),
                lambda: camera_port.verify_idle(context),
            )
        steps = []
        if prepared.current_pulse is not None:
            steps.append(lambda: prepared.current_pulse.cleanup(context))
        if prepared.current_camera_session_id is not None:
            steps.append(
                lambda: camera_port.cleanup(
                    context, prepared.current_camera_session_id
                )
            )
        if prepared.exposure_lease_open:
            steps.append(
                lambda: camera_port.cleanup(context, prepared.exposure_lease_id)
            )
        return run_cleanup_steps(*steps) if steps else CleanupReport.complete()

    def finalize(
        context: PostSafetyContext,
        result: ReadoutDurationFidelityResult,
    ) -> ReadoutDurationFidelityResult:
        if not isinstance(result, ReadoutDurationFidelityResult):
            raise TypeError("readout-duration finalize received another result")
        if result.run_id != context.run_id.value:
            raise ValueError("readout-duration result belongs to another Run")
        context.checkpoint()
        return result

    plan = RunPlan(
        name=f"Fidelity vs duration {bound.request.pulse_document.name}",
        resource_claims=(pulse_port.resource_claim, camera_port.resource_claim),
        bound_devices=(pulse_port.device, camera_port.device),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=(
            *pulse_port.interrupt_operations,
            *camera_port.interrupt_operations,
        ),
        requires_final_commit=False,
    )
    return PreparedReadoutDurationFidelity(plan, start_run)


__all__ = [
    "PreparedReadoutDurationFidelity",
    "ReadoutDurationFidelityResult",
    "prepare_readout_duration_fidelity",
]
