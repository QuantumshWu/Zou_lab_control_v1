"""Typed terminal evidence for the frozen pulse-streamer runtime.

Raw register observations, host tail waiting, and compiled expectations are
deliberately separate.  The frozen RTL does not expose an artifact digest,
emitted-trigger count, per-edge tag, physical-done receipt, or tail-idle bit;
none of those claims may be manufactured here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fpga.pulse_streamer.host.image import (
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_UNDERFLOW,
)
from zlc_storage import canonical_digest

from .artifact import CompiledPulseArtifact, PulseExecutionForm


STATIC_TERMINAL_EVIDENCE_SCHEMA = (
    "zlc_pulse.StaticOnceTerminalEvidence/v1"
)
AUTONOMOUS_TERMINAL_EVIDENCE_SCHEMA = (
    "zlc_pulse.AutonomousTableTerminalEvidence/v1"
)
POST_TERMINAL_TAIL_EVIDENCE_SCHEMA = (
    "zlc_pulse.PostTerminalTailEvidence/v1"
)

STATIC_STATUS_READ_RECIPE = "STATUS_THEN_STATUS/v1"
AUTONOMOUS_TABLE_READ_RECIPE = "STATUS_CURSOR_STATUS_CURSOR/v1"
POST_TERMINAL_TAIL_WAIT_RECIPE = "HOST_MONOTONIC_AFTER_TERMINAL/v1"


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _u32(value: object, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 0xFFFFFFFF
    ):
        raise ValueError(f"{field} must be an unsigned 32-bit integer")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class StaticOnceTerminalEvidence:
    """Two raw stable STATUS samples for a finite non-table execution."""

    read_recipe_id: str
    transport_id: str
    status_first: int
    status_second: int
    underflow_observed: bool
    status_sample_count: int

    def __post_init__(self) -> None:
        if self.read_recipe_id != STATIC_STATUS_READ_RECIPE:
            raise ValueError("static terminal evidence uses an unknown read recipe")
        _text(self.transport_id, "transport_id")
        _u32(self.status_first, "status_first")
        _u32(self.status_second, "status_second")
        if type(self.underflow_observed) is not bool:
            raise TypeError("underflow_observed must be bool")
        samples = _nonnegative_int(self.status_sample_count, "status_sample_count")
        if samples < 2:
            raise ValueError("terminal evidence requires at least two STATUS samples")

    @property
    def fingerprint(self) -> str:
        return canonical_digest(hardware_terminal_evidence_to_tree(self))


@dataclass(frozen=True)
class AutonomousTableTerminalEvidence:
    """Raw stable STATUS/CURSOR samples for one resident autonomous table."""

    read_recipe_id: str
    transport_id: str
    status_first: int
    cursor_first: int
    status_second: int
    cursor_second: int
    underflow_observed: bool
    status_sample_count: int

    def __post_init__(self) -> None:
        if self.read_recipe_id != AUTONOMOUS_TABLE_READ_RECIPE:
            raise ValueError("autonomous terminal evidence uses an unknown read recipe")
        _text(self.transport_id, "transport_id")
        for field in (
            "status_first",
            "cursor_first",
            "status_second",
            "cursor_second",
        ):
            _u32(getattr(self, field), field)
        if type(self.underflow_observed) is not bool:
            raise TypeError("underflow_observed must be bool")
        samples = _nonnegative_int(self.status_sample_count, "status_sample_count")
        if samples < 2:
            raise ValueError("terminal evidence requires at least two STATUS samples")

    @property
    def fingerprint(self) -> str:
        return canonical_digest(hardware_terminal_evidence_to_tree(self))


PulseHardwareTerminalEvidence = (
    StaticOnceTerminalEvidence | AutonomousTableTerminalEvidence
)


@dataclass(frozen=True)
class PostTerminalTailEvidence:
    """Host-monotonic proof that configured output-delay tail time elapsed."""

    terminal_evidence_digest: str
    wait_recipe_id: str
    required_tail_ticks: int
    clock_hz: float
    elapsed_ns: int

    def __post_init__(self) -> None:
        _sha256(self.terminal_evidence_digest, "terminal_evidence_digest")
        if self.wait_recipe_id != POST_TERMINAL_TAIL_WAIT_RECIPE:
            raise ValueError("post-terminal tail uses an unknown wait recipe")
        _nonnegative_int(self.required_tail_ticks, "required_tail_ticks")
        if (
            isinstance(self.clock_hz, bool)
            or not isinstance(self.clock_hz, (int, float))
            or not math.isfinite(float(self.clock_hz))
            or float(self.clock_hz) <= 0
        ):
            raise ValueError("clock_hz must be finite and positive")
        object.__setattr__(self, "clock_hz", float(self.clock_hz))
        _nonnegative_int(self.elapsed_ns, "elapsed_ns")

    @property
    def fingerprint(self) -> str:
        return canonical_digest(post_terminal_tail_evidence_to_tree(self))


@dataclass(frozen=True)
class PulseBackendCompletion:
    """Backend observations before compiled-artifact validation by the service."""

    hardware_terminal: PulseHardwareTerminalEvidence
    post_terminal_tail: PostTerminalTailEvidence

    def __post_init__(self) -> None:
        if not isinstance(
            self.hardware_terminal,
            (StaticOnceTerminalEvidence, AutonomousTableTerminalEvidence),
        ):
            raise TypeError("hardware_terminal has an unknown evidence variant")
        if not isinstance(self.post_terminal_tail, PostTerminalTailEvidence):
            raise TypeError("post_terminal_tail must be PostTerminalTailEvidence")
        if (
            self.post_terminal_tail.terminal_evidence_digest
            != self.hardware_terminal.fingerprint
        ):
            raise ValueError("tail evidence belongs to another hardware terminal")


def validate_terminal_for_artifact(
    evidence: PulseHardwareTerminalEvidence,
    artifact: CompiledPulseArtifact,
) -> None:
    """Validate raw observations using compiled expectations without mixing them."""

    if not isinstance(artifact, CompiledPulseArtifact):
        raise TypeError("artifact must be CompiledPulseArtifact")
    if artifact.execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
        raise ValueError("continuous monitor has no finite terminal evidence")
    if artifact.execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
        if not isinstance(evidence, AutonomousTableTerminalEvidence):
            raise ValueError("autonomous table execution requires cursor evidence")
    elif not isinstance(evidence, StaticOnceTerminalEvidence):
        raise ValueError("static execution forbids cursor evidence")
    _validate_terminal_success(evidence)
    if isinstance(evidence, AutonomousTableTerminalEvidence):
        expected = len(artifact.target_ir.scan_points) - 1
        if expected < 0 or evidence.cursor_first != expected:
            raise ValueError("terminal CURSOR differs from compiled final table row")


def _validate_terminal_success(
    evidence: PulseHardwareTerminalEvidence,
) -> None:
    """Validate success facts that do not require a compiled artifact."""

    if evidence.status_first != evidence.status_second:
        raise ValueError("terminal STATUS samples are not stable")
    forbidden = STATUS_ERROR | STATUS_UNDERFLOW
    if not evidence.status_first & STATUS_DONE:
        raise ValueError("terminal STATUS does not report DONE")
    if evidence.status_first & forbidden or evidence.underflow_observed:
        raise ValueError("terminal STATUS reports error or observed underflow")
    if isinstance(evidence, AutonomousTableTerminalEvidence):
        if evidence.cursor_first != evidence.cursor_second:
            raise ValueError("terminal CURSOR samples are not stable")


def validate_backend_completion_intrinsic(
    completion: PulseBackendCompletion,
) -> None:
    """Validate the self-contained success claims of a backend completion."""

    if not isinstance(completion, PulseBackendCompletion):
        raise TypeError("completion must be PulseBackendCompletion")
    _validate_terminal_success(completion.hardware_terminal)
    tail = completion.post_terminal_tail
    if tail.elapsed_ns * tail.clock_hz < tail.required_tail_ticks * 1_000_000_000:
        raise ValueError("post-terminal output-delay tail is incomplete")


def validate_backend_completion_for_artifact(
    completion: PulseBackendCompletion,
    artifact: CompiledPulseArtifact,
) -> None:
    if not isinstance(completion, PulseBackendCompletion):
        raise TypeError("completion must be PulseBackendCompletion")
    validate_backend_completion_intrinsic(completion)
    validate_terminal_for_artifact(completion.hardware_terminal, artifact)
    tail = completion.post_terminal_tail
    if tail.required_tail_ticks != artifact.max_configured_output_delay_ticks:
        raise ValueError("tail evidence tick bound differs from compiled artifact")
    if tail.clock_hz != artifact.target_ir.clock_hz:
        raise ValueError("tail evidence clock differs from compiled artifact")


def hardware_terminal_evidence_to_tree(
    value: PulseHardwareTerminalEvidence,
) -> dict[str, object]:
    common = {
        "read_recipe_id": value.read_recipe_id,
        "transport_id": value.transport_id,
        "status_first": value.status_first,
        "status_second": value.status_second,
        "underflow_observed": value.underflow_observed,
        "status_sample_count": value.status_sample_count,
    }
    if isinstance(value, StaticOnceTerminalEvidence):
        return {"schema": STATIC_TERMINAL_EVIDENCE_SCHEMA, **common}
    if isinstance(value, AutonomousTableTerminalEvidence):
        return {
            "schema": AUTONOMOUS_TERMINAL_EVIDENCE_SCHEMA,
            **common,
            "cursor_first": value.cursor_first,
            "cursor_second": value.cursor_second,
        }
    raise TypeError("value has an unknown hardware terminal evidence variant")


def hardware_terminal_evidence_from_tree(
    tree: object,
) -> PulseHardwareTerminalEvidence:
    if not isinstance(tree, dict):
        raise TypeError("hardware terminal evidence must be an object")
    schema = tree.get("schema")
    common = {
        "schema",
        "read_recipe_id",
        "transport_id",
        "status_first",
        "status_second",
        "underflow_observed",
        "status_sample_count",
    }
    if schema == STATIC_TERMINAL_EVIDENCE_SCHEMA:
        if set(tree) != common:
            raise ValueError("StaticOnceTerminalEvidence has an unknown field set")
        return StaticOnceTerminalEvidence(
            tree["read_recipe_id"],
            tree["transport_id"],
            tree["status_first"],
            tree["status_second"],
            tree["underflow_observed"],
            tree["status_sample_count"],
        )
    if schema == AUTONOMOUS_TERMINAL_EVIDENCE_SCHEMA:
        if set(tree) != common | {"cursor_first", "cursor_second"}:
            raise ValueError("AutonomousTableTerminalEvidence has an unknown field set")
        return AutonomousTableTerminalEvidence(
            tree["read_recipe_id"],
            tree["transport_id"],
            tree["status_first"],
            tree["cursor_first"],
            tree["status_second"],
            tree["cursor_second"],
            tree["underflow_observed"],
            tree["status_sample_count"],
        )
    raise ValueError("hardware terminal evidence schema differs")


def post_terminal_tail_evidence_to_tree(
    value: PostTerminalTailEvidence,
) -> dict[str, object]:
    if not isinstance(value, PostTerminalTailEvidence):
        raise TypeError("value must be PostTerminalTailEvidence")
    return {
        "schema": POST_TERMINAL_TAIL_EVIDENCE_SCHEMA,
        "terminal_evidence_digest": value.terminal_evidence_digest,
        "wait_recipe_id": value.wait_recipe_id,
        "required_tail_ticks": value.required_tail_ticks,
        "clock_hz": value.clock_hz,
        "elapsed_ns": value.elapsed_ns,
    }


def post_terminal_tail_evidence_from_tree(tree: object) -> PostTerminalTailEvidence:
    fields = {
        "schema",
        "terminal_evidence_digest",
        "wait_recipe_id",
        "required_tail_ticks",
        "clock_hz",
        "elapsed_ns",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("PostTerminalTailEvidence has an unknown field set")
    if tree["schema"] != POST_TERMINAL_TAIL_EVIDENCE_SCHEMA:
        raise ValueError("PostTerminalTailEvidence schema differs")
    return PostTerminalTailEvidence(
        tree["terminal_evidence_digest"],
        tree["wait_recipe_id"],
        tree["required_tail_ticks"],
        tree["clock_hz"],
        tree["elapsed_ns"],
    )


__all__ = [
    "AUTONOMOUS_TABLE_READ_RECIPE",
    "AUTONOMOUS_TERMINAL_EVIDENCE_SCHEMA",
    "AutonomousTableTerminalEvidence",
    "POST_TERMINAL_TAIL_EVIDENCE_SCHEMA",
    "POST_TERMINAL_TAIL_WAIT_RECIPE",
    "PostTerminalTailEvidence",
    "PulseBackendCompletion",
    "PulseHardwareTerminalEvidence",
    "STATIC_STATUS_READ_RECIPE",
    "STATIC_TERMINAL_EVIDENCE_SCHEMA",
    "StaticOnceTerminalEvidence",
    "hardware_terminal_evidence_from_tree",
    "hardware_terminal_evidence_to_tree",
    "post_terminal_tail_evidence_from_tree",
    "post_terminal_tail_evidence_to_tree",
    "validate_backend_completion_for_artifact",
    "validate_backend_completion_intrinsic",
    "validate_terminal_for_artifact",
]
