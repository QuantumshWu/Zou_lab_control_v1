"""Validated pulse-to-camera binding and terminal lineage."""

from __future__ import annotations

from dataclasses import dataclass, field

from zlc_pulse import (
    CompiledPulseArtifact,
    DigitalTriggerSchedule,
    digital_trigger_schedule_to_tree,
)
from zlc_storage import canonical_digest, canonical_text

from .capture_plan import CompiledCaptureCellPlan
from .pulse import (
    PulseTerminalAck,
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
        if self.cell_plan.trigger_schedule_digest != canonical_digest(
            digital_trigger_schedule_to_tree(schedule)
        ):
            raise ValueError("capture cell plan trigger schedule digest differs")
        if (
            self.cell_plan.scan_point_count != schedule.point_count
            or self.cell_plan.total_events != schedule.total
        ):
            raise ValueError("capture cell plan cardinality differs from trigger schedule")
        scheduled_edges = tuple(
            (edge.trigger_ordinal, edge.point_index, edge.point_trigger_ordinal)
            for edge in schedule.edges
        )
        planned_edges = tuple(
            (
                item.trigger_ordinal,
                item.pulse_point_index,
                item.point_trigger_ordinal,
            )
            for item in self.cell_plan.assignments
        )
        if planned_edges != scheduled_edges:
            raise ValueError("capture cell plan edge association differs from trigger schedule")
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


__all__ = ["PulseCaptureBinding", "PulseCaptureLineage"]
