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
    PointColumn,
    PointTable,
    StreamGenerationId,
    ValueSchema,
)
from zlc_data.codec import dataset_revision_ref_to_tree
from zlc_data.value import dataset_cell_value, expand_value_validity
from zlc_neutral_atom.capture.binding import (
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.capture.coordination import execute_autonomous_single_fire
from zlc_neutral_atom.capture.pipeline import (
    ExactCaptureTransaction,
    MinimalPipelineSpec,
    open_exact_capture_transaction,
)
from zlc_neutral_atom.capture.triggered import (
    TriggeredCaptureSpec,
    finalize_triggered_pipeline_result,
)
from zlc_neutral_atom.dataset_output import (
    FinalDatasetOutput,
    final_dataset_join_digest,
)
from zlc_neutral_atom.logic_nodes.readout.bimodal import fit_bimodal
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    ResolvedCalibration,
    extract_readout_features,
)
from zlc_neutral_atom.logic_nodes.readout.model_contract import ReadoutModelKind
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import CalibrationArtifactRef
from zlc_neutral_atom.logic_nodes.readout.duration_fidelity.measurement import (
    BoundReadoutDurationFidelity,
    READOUT_DURATION_FIDELITY_OUTPUT_DECLARATIONS,
    ReadoutDurationFidelityRequest,
    bind_readout_duration_fidelity,
)
from zlc_neutral_atom.devices.camera.capture_port import (
    BoundCapturePort,
    CaptureTerminalAck,
    capture_terminal_ack_to_tree,
    configure_camera_exposure,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport, run_cleanup_steps
from zlc_neutral_atom.runtime.run import PostSafetyContext, RunContext, RunHandle, RunPlan
from zlc_neutral_atom.devices.sequencer.port import (
    BoundPulsePort,
    PulseSession,
    PulseTerminalAck,
    pulse_terminal_ack_to_tree,
)
from zlc_storage import canonical_digest, sha256_text
from zlc_pulse import PulseExecutionForm


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
        point_count = self.snapshot.block.schema.point_table.row_count
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
                "owner": "zlc_neutral_atom.logic_nodes.readout-duration-fidelity",
                "run_id": self.run_id,
                "dataset_ref": dataset_revision_ref_to_tree(self.snapshot.ref),
                "calibration": self.calibration_ref.target_ref,
                "model_kind": self.model_kind.value,
                "site": self.site,
                "program": self.program_fingerprint,
                "point_terminals": tuple(
                    {
                        "capture": capture_terminal_ack_to_tree(capture),
                        "pulse": pulse_terminal_ack_to_tree(pulse),
                    }
                    for capture, pulse in zip(
                        self.capture_terminals,
                        self.pulse_terminals,
                        strict=True,
                    )
                ),
            }
        )


@dataclass(slots=True)
class _PreparedReadoutDuration:
    exposure_lease_id: str
    exposure_lease_open: bool = False
    current_capture: ExactCaptureTransaction | None = None
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
        return self._start_run(
            self._plan.with_lifecycle(
                owner=self,
                preemptible=False,
            )
        )

    def final_dataset_outputs(
        self,
        result: ReadoutDurationFidelityResult,
    ) -> dict[str, FinalDatasetOutput]:
        if not isinstance(result, ReadoutDurationFidelityResult):
            raise TypeError("result must be ReadoutDurationFidelityResult")
        declaration = READOUT_DURATION_FIDELITY_OUTPUT_DECLARATIONS[0]
        output = FinalDatasetOutput(
            declaration,
            result.snapshot,
            final_dataset_join_digest(
                owner="readout-duration-fidelity",
                declaration=declaration,
                source_identity=result.identity,
                snapshot=result.snapshot,
            ),
        )
        return {output.name: output}


def _point_samples(model, site: int | None, block: DataBlock) -> np.ndarray:
    """Extract calibrated samples from one exact, fully sealed point capture."""

    samples: list[np.ndarray] = []
    usable_sites = np.asarray(model.usable_sites.mask, dtype=bool)
    if block.schema.point_table.row_count != 1:
        raise ValueError("readout-duration point capture must have one point cell")
    for repeat_index in range(block.schema.repeat_axis.size):
        frame = dataset_cell_value(block, repeat_index, 0)
        signals = extract_readout_features(model.feature, frame)
        valid = np.asarray(
            expand_value_validity(signals.validity, signals.schema), dtype=bool
        ) & usable_sites
        values = np.asarray(signals.values, dtype=float)
        if site is None:
            samples.append(values[valid])
        elif valid[site]:
            samples.append(values[site : site + 1])
    return np.concatenate(samples) if samples else np.empty((0,), dtype=float)


def _result_snapshot(
    run_id: str,
    program_fingerprint: str,
    durations: tuple[float, ...],
    fidelities: tuple[float, ...],
    validity: tuple[bool, ...],
) -> OwnedSnapshot:
    duration_column = PointColumn(
        AxisId("readout_duration.duration"),
        "Detection time",
        SCAN_POINT,
        PointColumn.NUMERIC,
        durations,
        "s",
    )
    schema = DatasetSchema(
        AxisSpec(AxisId("readout_duration.repeat"), "repeat", REPEAT, 1, (0,)),
        PointTable(len(durations), (duration_column,)),
        None,
        ValueSchema.scalar(np.dtype("<f8"), "fidelity"),
    )
    identity = canonical_digest(
        {
            "owner": "zlc_neutral_atom.logic_nodes.readout-duration-fidelity-result",
            "run_id": run_id,
            "program": program_fingerprint,
            "duration_seconds": durations,
        }
    )
    block = DataBlock(
        BlockId(f"readout-duration-{identity[:20]}"),
        DatasetRevision(0),
        np.asarray(fidelities, dtype="<f8").reshape((1, len(durations), 1)),
        CellValidity(np.asarray(validity, dtype=bool).reshape((1, len(durations)))),
        schema,
    )
    generation = StreamGenerationId(f"readout-duration-{identity}")
    return OwnedSnapshot(block.ref(generation), block)


def prepare_readout_duration_fidelity(
    request: ReadoutDurationFidelityRequest,
    calibration: ResolvedCalibration,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedReadoutDurationFidelity:
    if not isinstance(request, ReadoutDurationFidelityRequest):
        raise TypeError("request must be ReadoutDurationFidelityRequest")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    calibration._require_authority()
    bound = bind_readout_duration_fidelity(
        request,
        calibration,
        pulse_port=pulse_port,
        camera_port=camera_port,
    )
    if not isinstance(bound, BoundReadoutDurationFidelity):
        raise RuntimeError("readout-duration binding returned another domain value")
    if calibration.reference != bound.request.calibration_ref:
        raise ValueError("calibration differs from the bound request")
    model = calibration.artifact.select_model(bound.request.model_kind)
    pulse_port = bound.pulse_port
    camera_port = bound.camera_port

    def preflight(_context: RunContext) -> _PreparedReadoutDuration:
        return _PreparedReadoutDuration(uuid.uuid4().hex)

    def execute(
        context: RunContext,
        prepared: _PreparedReadoutDuration,
    ) -> ReadoutDurationFidelityResult:
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
            configured_camera = configure_camera_exposure(
                context,
                camera_port,
                prepared.exposure_lease_id,
                requested,
            )
            applied_exposure = (
                configured_camera.capability.camera_physical_facts.exposure_seconds
            )
            point_binding = bind_triggered_camera_acquisition(
                pulse_port,
                configured_camera,
                pulse_document=pulse_request.document,
                execution_form=PulseExecutionForm.STATIC_ONCE,
                trigger_channel=bound.trigger_channel,
                layout=TriggeredCameraLayout(
                    repeat_axis=AxisSpec(
                        AxisId("readout-duration.capture-repeat"),
                        "capture repeat",
                        REPEAT,
                        bound.request.shots,
                        tuple(range(bound.request.shots)),
                    ),
                    ordinal_scan_axis_id=AxisId(
                        "readout-duration.capture-point"
                    ),
                    readout_event_axis_id=AxisId(
                        "readout-duration.capture-event"
                    ),
                    readout_events_per_repeat=1,
                ),
            )
            if (
                point_binding.pulse_request.artifact.fingerprint
                != pulse_request.artifact.fingerprint
            ):
                raise RuntimeError(
                    "point capture recompiled a different frozen pulse artifact"
                )
            capture_spec = MinimalPipelineSpec(
                f"Readout duration point {len(applied_durations)}",
                point_binding.capture,
                BlockId(
                    f"readout-duration-{context.run_id.value}-"
                    f"{len(applied_durations)}"
                ),
            )
            triggered_spec = TriggeredCaptureSpec(
                capture_spec,
                point_binding.pulse_port,
                point_binding.pulse_request,
                point_binding.trigger_channel,
                point_binding.cell_plan,
            )
            capture = open_exact_capture_transaction(capture_spec, context)
            prepared.current_capture = capture
            pulse = point_binding.pulse_port.open_session(
                point_binding.pulse_request
            )
            prepared.current_pulse = pulse
            capture_result, pulse_terminal = execute_autonomous_single_fire(
                context,
                pulse=pulse,
                capture=capture,
            )
            triggered_result = finalize_triggered_pipeline_result(
                triggered_spec,
                capture_result,
                pulse_terminal,
            )
            point_block = triggered_result.capture.dataset.block
            if point_block.schema.repeat_axis.size != bound.request.shots:
                raise RuntimeError(
                    "point CaptureCompletion covers another shot cardinality"
                )
            samples = _point_samples(model, bound.request.site, point_block)
            fit = fit_bimodal(samples)
            valid = bool(fit.ok and math.isfinite(float(fit.fidelity)))
            capture.release_completed_software(triggered_result.capture)

            # Each point proves hardware completion before the next point is
            # admitted, but only the Run cleanup phase owns physical session
            # closure.  Retain the latest completed camera/pulse sessions so
            # cleanup closes the endpoint state left by the final point.  A
            # following point replaces these references only after installing
            # its own sessions; clearing them here silently orphaned the final
            # endpoint session and made the next Run fail admission.

            applied_durations.append(applied_exposure)
            fidelities.append(float(fit.fidelity) if valid else float("nan"))
            valid_cells.append(valid)
            capture_terminals.append(triggered_result.capture.capture_terminal)
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
        if prepared.current_capture is not None:
            steps.append(lambda: prepared.current_capture.cleanup(context))
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
