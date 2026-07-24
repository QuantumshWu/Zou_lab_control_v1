"""Validated pulse-to-camera binding and terminal lineage."""

from __future__ import annotations

from dataclasses import dataclass, field

from zlc_data import DatasetSchema
from zlc_pulse import (
    CompiledPulseArtifact,
    DigitalTriggerSchedule,
)
from zlc_storage import canonical_text, exact_mapping

from .capture_plan import (
    CaptureCellJoinContract,
    CompiledCaptureCellPlan,
    capture_cell_join_contract_from_tree,
    capture_cell_join_contract_to_tree,
)
from zlc_neutral_atom.devices.sequencer.port import (
    PulseTerminalAck,
    pulse_terminal_ack_from_tree,
    pulse_terminal_ack_to_tree,
    validate_pulse_terminal_for_artifact,
)


@dataclass(frozen=True)
class PulseCaptureBinding:
    """One validated compiled-pulse to exact-cell-plan association."""

    compiled_artifact: CompiledPulseArtifact
    trigger_channel: str
    cell_plan: CompiledCaptureCellPlan
    _trigger_schedule: DigitalTriggerSchedule = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.compiled_artifact, CompiledPulseArtifact):
            raise TypeError("compiled_artifact must be CompiledPulseArtifact")
        canonical_text(self.trigger_channel, "trigger_channel")
        if not isinstance(self.cell_plan, CompiledCaptureCellPlan):
            raise TypeError("cell_plan must be CompiledCaptureCellPlan")
        schedules = tuple(
            schedule
            for schedule in self.compiled_artifact.trigger_schedules
            if schedule.channel == self.trigger_channel
        )
        if len(schedules) != 1:
            raise ValueError(
                "pulse capture binding requires exactly one trigger-channel schedule"
            )
        schedule = schedules[0]
        if (
            self.cell_plan.compiled_pulse_artifact_digest
            != self.compiled_artifact.fingerprint
        ):
            raise ValueError("capture cell plan belongs to another compiled artifact")
        if self.cell_plan.execution_form is not self.compiled_artifact.execution_form:
            raise ValueError("capture cell plan execution form differs from lineage")
        if self.cell_plan.trigger_channel != self.trigger_channel:
            raise ValueError("capture cell plan trigger channel differs from lineage")
        if self.cell_plan.trigger_schedule_digest != schedule.fingerprint:
            raise ValueError("capture cell plan trigger schedule digest differs")
        object.__setattr__(self, "_trigger_schedule", schedule)

    @property
    def trigger_schedule(self) -> DigitalTriggerSchedule:
        return self._trigger_schedule

    @property
    def expected_trigger_count(self) -> int:
        return self.cell_plan.total_events


@dataclass(frozen=True)
class PulseCaptureLineage:
    """A validated pulse/cell binding plus its completed terminal receipt."""

    binding: PulseCaptureBinding
    terminal: PulseTerminalAck

    def __post_init__(self) -> None:
        if not isinstance(self.binding, PulseCaptureBinding):
            raise TypeError("binding must be PulseCaptureBinding")
        if not isinstance(self.terminal, PulseTerminalAck):
            raise TypeError("terminal must be PulseTerminalAck")
        validate_pulse_terminal_for_artifact(
            self.terminal,
            self.binding.compiled_artifact,
        )
        counts = dict(
            self.terminal.expected_trigger_counts_from_completed_schedule
        )
        if self.binding.trigger_channel not in counts:
            raise ValueError("pulse terminal omits the capture trigger channel")
        if counts[self.binding.trigger_channel] != self.binding.expected_trigger_count:
            raise ValueError("pulse terminal count differs from capture cell plan")

    @property
    def compiled_artifact(self) -> CompiledPulseArtifact:
        return self.binding.compiled_artifact

    @property
    def trigger_channel(self) -> str:
        return self.binding.trigger_channel

    @property
    def cell_plan(self) -> CompiledCaptureCellPlan:
        return self.binding.cell_plan

    @property
    def expected_trigger_count(self) -> int:
        return self.binding.expected_trigger_count

    def evidence(self) -> "PulseCaptureEvidence":
        """Discard the execution plan while retaining its exact cell identity."""

        return PulseCaptureEvidence(
            self.compiled_artifact,
            self.trigger_channel,
            self.terminal,
            self.binding.cell_plan.join_contract,
        )


@dataclass(frozen=True)
class PulseCaptureEvidence:
    """Execution-after evidence without the O(N) pre-execution cell plan."""

    compiled_artifact: CompiledPulseArtifact
    trigger_channel: str
    terminal: PulseTerminalAck
    join_contract: CaptureCellJoinContract
    _trigger_schedule: DigitalTriggerSchedule = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.compiled_artifact, CompiledPulseArtifact):
            raise TypeError("compiled_artifact must be CompiledPulseArtifact")
        canonical_text(self.trigger_channel, "trigger_channel")
        if not isinstance(self.terminal, PulseTerminalAck):
            raise TypeError("terminal must be PulseTerminalAck")
        if not isinstance(self.join_contract, CaptureCellJoinContract):
            raise TypeError("join_contract must be CaptureCellJoinContract")
        schedules = tuple(
            schedule
            for schedule in self.compiled_artifact.trigger_schedules
            if schedule.channel == self.trigger_channel
        )
        if len(schedules) != 1:
            raise ValueError(
                "pulse capture evidence requires exactly one trigger-channel schedule"
            )
        schedule = schedules[0]
        validate_pulse_terminal_for_artifact(
            self.terminal,
            self.compiled_artifact,
        )
        counts = dict(
            self.terminal.expected_trigger_counts_from_completed_schedule
        )
        if counts.get(self.trigger_channel) != schedule.total:
            raise ValueError(
                "pulse terminal count differs from capture trigger schedule"
            )
        object.__setattr__(self, "_trigger_schedule", schedule)

    @property
    def trigger_schedule(self) -> DigitalTriggerSchedule:
        return self._trigger_schedule

    @property
    def expected_trigger_count(self) -> int:
        return self._trigger_schedule.total

    def expected_cell_schedule_digest(self, schema: DatasetSchema) -> str:
        return self.join_contract.expected_cell_schedule_digest(
            self._trigger_schedule,
            schema,
        )


def pulse_capture_evidence_to_tree(
    value: PulseCaptureEvidence | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, PulseCaptureEvidence):
        raise TypeError("value must be PulseCaptureEvidence or None")
    return {
        "trigger_channel": value.trigger_channel,
        "terminal": pulse_terminal_ack_to_tree(value.terminal),
        "join_contract": capture_cell_join_contract_to_tree(
            value.join_contract
        ),
    }


def pulse_capture_evidence_from_tree(
    tree: object,
    compiled_artifact: CompiledPulseArtifact | None,
) -> PulseCaptureEvidence | None:
    if tree is None:
        if compiled_artifact is not None:
            raise ValueError("compiled pulse exists without pulse capture evidence")
        return None
    if compiled_artifact is None:
        raise ValueError("pulse capture evidence omits its compiled pulse")
    data = exact_mapping(
        tree,
        {"trigger_channel", "terminal", "join_contract"},
        "pulse capture evidence",
        discriminator=None,
    )
    return PulseCaptureEvidence(
        compiled_artifact,
        data["trigger_channel"],
        pulse_terminal_ack_from_tree(data["terminal"]),
        capture_cell_join_contract_from_tree(data["join_contract"]),
    )


__all__ = [
    "PulseCaptureBinding",
    "PulseCaptureEvidence",
    "PulseCaptureLineage",
    "pulse_capture_evidence_from_tree",
    "pulse_capture_evidence_to_tree",
]
