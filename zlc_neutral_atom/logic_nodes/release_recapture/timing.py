"""One autonomous FPGA-triggered two-readout survival run."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from zlc_neutral_atom.logic_nodes.release_recapture.pipeline import (
    ExactReleaseRecaptureTransaction,
    ReleaseRecapturePipelineSpec,
    open_exact_release_recapture,
)
from zlc_neutral_atom.capture.pipeline import PipelineResult
from zlc_neutral_atom.runtime._failure import record_secondary_failure
from zlc_neutral_atom.runtime.cleanup import CleanupReport, run_cleanup_steps
from zlc_neutral_atom.runtime.run import PostSafetyContext, RunContext, RunPlan
from zlc_neutral_atom.devices.rf import BoundRfTablePort, RfDetuningTable, RfTableTerminal
from zlc_neutral_atom.capture.coordination import (
    execute_autonomous_single_fire,
    validate_single_trigger_capture_binding,
)
from zlc_neutral_atom.timing.lineage import PulseCaptureBinding, PulseCaptureLineage
from zlc_neutral_atom.devices.sequencer.port import (
    PulseSession,
    PulseTerminalAck,
)


@dataclass(frozen=True, slots=True)
class TriggeredReleaseRecaptureResult:
    pipeline: PipelineResult
    lineage: PulseCaptureLineage
    rf_terminal: RfTableTerminal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline, PipelineResult):
            raise TypeError("pipeline must be PipelineResult")
        if not isinstance(self.lineage, PulseCaptureLineage):
            raise TypeError("lineage must be PulseCaptureLineage")
        if self.rf_terminal is not None and not isinstance(
            self.rf_terminal, RfTableTerminal
        ):
            raise TypeError("rf_terminal must be RfTableTerminal or None")
        self.lineage.cell_plan.validate_dataset_schema(
            self.pipeline.source_dataset_schema
        )
        if not self.pipeline.source_cell_schedule.same_order_as(
            self.lineage.cell_plan.cell_schedule
        ):
            raise ValueError(
                "release-recapture camera order differs from pulse cell plan"
            )
        evidence = self.pipeline.camera_capability_evidence
        evidence.physical_facts.require_single_capture_trigger_channel(
            self.lineage.trigger_channel
        )
        expected = self.lineage.expected_trigger_count
        terminal = self.pipeline.capture_terminal
        if not (
            expected
            == terminal.produced_count
            == terminal.drained_count
            == self.pipeline.source_event_span.end_sequence
            - self.pipeline.source_event_span.start_sequence
        ):
            raise RuntimeError(
                "pulse, camera, and release-recapture source counts differ"
            )
        if self.pipeline.dataset.coverage.total_cells * 2 != expected:
            raise RuntimeError(
                "release-recapture output is not an exact two-frame reduction"
            )

    @property
    def survival(self):
        return self.pipeline.dataset.snapshot


def _execute_release_recapture_with_rf_table(
    context: RunContext,
    *,
    pulse: PulseSession,
    capture: ExactReleaseRecaptureTransaction,
    rf_port: BoundRfTablePort,
    rf_session_id: str,
    rf_table: RfDetuningTable,
) -> tuple[PipelineResult, PulseTerminalAck, RfTableTerminal]:
    """Preload the RF table, then let the sequencer clock one exact capture."""

    if not isinstance(rf_port, BoundRfTablePort):
        raise TypeError("rf_port must be BoundRfTablePort")
    if not isinstance(rf_table, RfDetuningTable):
        raise TypeError("rf_table must be RfDetuningTable")
    try:
        pulse.prepare(context)
        rf_port.prepare(context, rf_session_id, rf_table)
        capture.start(context)
        pulse.fire(context)
        capture.capture_all(context)
        terminal = pulse.complete(context)
        if not pulse.owns_terminal(terminal):
            raise PermissionError("pulse terminal was not minted by this session")
        rf_terminal = rf_port.complete(
            context,
            rf_session_id,
            rf_table,
        )
        return capture.complete(context), terminal, rf_terminal
    except BaseException as primary:
        for label, fail in (
            ("capture poison", lambda: capture.fail(primary)),
            ("pulse poison", pulse.fail),
        ):
            try:
                fail()
            except BaseException as secondary:
                record_secondary_failure(
                    primary,
                    f"{label} also failed",
                    secondary,
                )
        raise


def compile_triggered_release_recapture_pipeline(
    spec: ReleaseRecapturePipelineSpec,
) -> RunPlan:
    """Compile ready-all -> arm camera -> one hardware FIRE -> exact pairs."""

    if not isinstance(spec, ReleaseRecapturePipelineSpec):
        raise TypeError("spec must be ReleaseRecapturePipelineSpec")
    camera_binding = spec.camera_binding
    camera_port = camera_binding.capture.capture_port
    pulse_port = camera_binding.pulse_port
    if camera_port.device.key == pulse_port.device.key:
        raise ValueError("camera and sequencer must be distinct physical resources")
    pulse_binding = PulseCaptureBinding(
        camera_binding.pulse_request.artifact,
        camera_binding.trigger_channel,
        camera_binding.cell_plan,
    )
    validate_single_trigger_capture_binding(
        capture_spec=camera_binding.capture.capture_spec,
        contract=camera_binding.capture.capture_contract,
        pulse_binding=pulse_binding,
    )
    grouping = pulse_binding.cell_plan.join_contract.within_point_grouping
    if {event for _repeat, event in grouping} != {0, 1}:
        raise ValueError(
            "release-recapture requires physical readout events 0 and 1"
        )

    def preflight(
        context: RunContext,
    ) -> tuple[ExactReleaseRecaptureTransaction, PulseSession, str | None]:
        pulse = pulse_port.open_session(camera_binding.pulse_request)
        try:
            reducer = open_exact_release_recapture(
                spec,
                context,
            )
        except BaseException as primary:
            try:
                pulse.fail()
            except BaseException as secondary:
                record_secondary_failure(
                    primary,
                    "pulse preflight poison also failed",
                    secondary,
                )
            raise
        return (
            reducer,
            pulse,
            None if spec.rf_port is None else uuid.uuid4().hex,
        )

    def execute(
        context: RunContext,
        prepared: tuple[
            ExactReleaseRecaptureTransaction,
            PulseSession,
            str | None,
        ],
    ) -> TriggeredReleaseRecaptureResult:
        reducer, pulse, rf_session_id = prepared
        rf_terminal = None
        if spec.rf_port is None:
            result, terminal = execute_autonomous_single_fire(
                context,
                pulse=pulse,
                capture=reducer,
            )
        else:
            assert spec.rf_table is not None
            assert rf_session_id is not None
            result, terminal, rf_terminal = (
                _execute_release_recapture_with_rf_table(
                    context,
                    pulse=pulse,
                    capture=reducer,
                    rf_port=spec.rf_port,
                    rf_session_id=rf_session_id,
                    rf_table=spec.rf_table,
                )
            )
        if not isinstance(result, PipelineResult):
            raise TypeError(
                "release-recapture capture returned another result type"
            )
        return TriggeredReleaseRecaptureResult(
            result,
            PulseCaptureLineage(pulse_binding, terminal),
            rf_terminal,
        )

    def cleanup(
        context: RunContext,
        prepared: tuple[
            ExactReleaseRecaptureTransaction,
            PulseSession,
            str | None,
        ] | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is None:
            steps = [
                lambda: pulse_port.verify_idle(context),
                lambda: camera_port.verify_idle(context),
            ]
            if spec.rf_port is not None:
                steps.append(lambda: spec.rf_port.verify_idle(context))
            return run_cleanup_steps(*steps)
        steps = [
            lambda: prepared[1].cleanup(context),
            lambda: prepared[0].cleanup(context),
        ]
        if spec.rf_port is not None:
            assert prepared[2] is not None
            steps.append(
                lambda: spec.rf_port.cleanup(context, prepared[2])
            )
        return run_cleanup_steps(*steps)

    def finalize(
        context: PostSafetyContext,
        executed: TriggeredReleaseRecaptureResult,
    ) -> TriggeredReleaseRecaptureResult:
        if not isinstance(executed, TriggeredReleaseRecaptureResult):
            raise TypeError(
                "release-recapture finalize received another value"
            )
        if executed.pipeline.run_id != context.run_id.value:
            raise ValueError(
                "release-recapture result belongs to another Run"
            )
        context.checkpoint()
        return executed

    rf_claims = () if spec.rf_port is None else (spec.rf_port.resource_claim,)
    rf_devices = () if spec.rf_port is None else (spec.rf_port.device,)
    rf_interrupts = () if spec.rf_port is None else spec.rf_port.interrupt_operations
    return RunPlan(
        name=spec.name,
        resource_claims=(
            pulse_port.resource_claim,
            camera_port.resource_claim,
            *rf_claims,
        ),
        bound_devices=(pulse_port.device, camera_port.device, *rf_devices),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=(
            *pulse_port.interrupt_operations,
            *camera_port.interrupt_operations,
            *rf_interrupts,
        ),
        requires_final_commit=False,
    )


__all__ = [
    "TriggeredReleaseRecaptureResult",
    "compile_triggered_release_recapture_pipeline",
]
