"""Current compiled pulse artifact joining source, TargetIR, wire, and triggers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from zlc_storage import (
    canonical_text as _text,
    decode,
    encode,
    exact_mapping,
    sha256_digest,
    sha256_text as _sha256,
)

from .fpga import (
    PulseWireImage,
    pulse_wire_image_from_tree,
    pulse_wire_image_to_tree,
)
from .ir import TargetIR, target_ir_from_tree, target_ir_to_tree
from .schedule import (
    DigitalTriggerSchedule,
    MAX_MATERIALIZED_TRIGGER_EDGES,
    _same_physical_digital_trigger_schedules,
    digital_trigger_schedule_from_tree,
    digital_trigger_schedule_to_tree,
)


COMPILED_PULSE_ARTIFACT_SCHEMA = "zlc_pulse.CompiledPulseArtifact"
MAX_COMPILED_PULSE_ARTIFACT_BYTES = 128 * 1024 * 1024
_COMPILED_PULSE_RETAINED_FIXED_BYTES = 256 * 1024
_COMPILED_PULSE_LOGICAL_ITEM_BYTES = 256
_COMPILED_PULSE_DECODE_MULTIPLIER = 16


class PulseExecutionForm(str, Enum):
    STATIC_ONCE = "STATIC_ONCE"
    STATIC_REFERENCE_POINT = "STATIC_REFERENCE_POINT"
    CONTINUOUS_MONITOR = "CONTINUOUS_MONITOR"
    AUTONOMOUS_SCAN_ONCE = "AUTONOMOUS_SCAN_ONCE"
    AUTONOMOUS_SCAN_CONTINUOUS = "AUTONOMOUS_SCAN_CONTINUOUS"


@dataclass(frozen=True, slots=True)
class CompiledPulseRuntimeSummary:
    """Small resource facts recomputed from a fully decoded pulse artifact.

    Artifact decode/retention bounds apply to every valid pulse.  The physical
    readout-window bounds are an optional capability: raw capture/playback may
    legitimately use a cyclic pulse, DAC ramp, or compact repeated DAC program
    that the current readout-context model cannot represent.
    """

    retained_upper_bound_bytes: int
    decode_peak_upper_bound_bytes: int
    readout_physical_context_bounds: tuple[int, int] | None

    def __post_init__(self) -> None:
        for field in (
            "retained_upper_bound_bytes",
            "decode_peak_upper_bound_bytes",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.decode_peak_upper_bound_bytes < self.retained_upper_bound_bytes:
            raise ValueError("pulse decode bound is smaller than retained state")
        bounds = self.readout_physical_context_bounds
        if bounds is not None:
            try:
                bounds = tuple(bounds)
            except TypeError as exc:
                raise TypeError(
                    "readout_physical_context_bounds must be a pair or None"
                ) from exc
            if len(bounds) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in bounds
            ):
                raise ValueError(
                    "readout physical-context bounds must be two non-negative integers"
                )
            object.__setattr__(self, "readout_physical_context_bounds", bounds)

    @property
    def supports_readout_physical_context(self) -> bool:
        return self.readout_physical_context_bounds is not None

    def require_readout_physical_context_bounds(self) -> tuple[int, int]:
        """Return bounds or fail closed at the readout capability boundary."""

        bounds = self.readout_physical_context_bounds
        if bounds is None:
            from .physical import PhysicalReadoutContextUnsupportedError

            raise PhysicalReadoutContextUnsupportedError(
                "compiled pulse has no exact readout physical-context capability"
            )
        return bounds

    @property
    def physical_index_peak_upper_bound_bytes(self) -> int:
        """Required readout bound; absent capability fails closed."""

        return self.require_readout_physical_context_bounds()[0]

    @property
    def physical_projection_peak_upper_bound_bytes(self) -> int:
        """Required readout bound; absent capability fails closed."""

        return self.require_readout_physical_context_bounds()[1]

    def require_encoded_size(self, encoded_size: int) -> None:
        """Bind fail-fast resource facts to one persisted canonical blob size."""

        size = admit_compiled_pulse_payload_size(encoded_size)
        expected = (
            _COMPILED_PULSE_DECODE_MULTIPLIER * size
            + 2 * self.retained_upper_bound_bytes
        )
        if self.decode_peak_upper_bound_bytes != expected:
            raise ValueError(
                "compiled pulse runtime summary differs from its blob size"
            )


@dataclass(frozen=True)
class CompiledPulseArtifact:
    source_document_digest: str
    compiler_id: str
    execution_form: PulseExecutionForm
    target_ir: TargetIR
    wire_image: PulseWireImage
    trigger_schedules: tuple[DigitalTriggerSchedule, ...] = ()
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _sha256(self.source_document_digest, "source_document_digest")
        object.__setattr__(self, "compiler_id", _text(self.compiler_id, "compiler_id"))
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        if not isinstance(self.target_ir, TargetIR):
            raise TypeError("target_ir must be TargetIR")
        if not isinstance(self.wire_image, PulseWireImage):
            raise TypeError("wire_image must be PulseWireImage")
        if self.wire_image.source_ir_digest != self.target_ir.fingerprint:
            raise ValueError("wire image is not bound to target_ir")
        schedules = tuple(self.trigger_schedules)
        if any(not isinstance(item, DigitalTriggerSchedule) for item in schedules):
            raise TypeError("trigger_schedules must contain DigitalTriggerSchedule values")
        if len({item.channel for item in schedules}) != len(schedules):
            raise ValueError("trigger schedule channels must be unique")
        if sum(item.total for item in schedules) > MAX_MATERIALIZED_TRIGGER_EDGES:
            raise ValueError(
                "compiled pulse exceeds the materialization limit of "
                f"{MAX_MATERIALIZED_TRIGGER_EDGES} trigger edges"
            )
        object.__setattr__(self, "trigger_schedules", schedules)
        scan = self.target_ir.scan_enabled
        if self.execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_ONCE:
            if not scan or self.target_ir.repeat_forever:
                raise ValueError("AUTONOMOUS_SCAN_ONCE requires finite scan TargetIR")
        elif self.execution_form is PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS:
            if not scan or not self.target_ir.repeat_forever:
                raise ValueError(
                    "AUTONOMOUS_SCAN_CONTINUOUS requires cyclic scan TargetIR"
                )
        elif scan:
            raise ValueError("only autonomous scan forms may carry scan points")
        if self.execution_form in (
            PulseExecutionForm.CONTINUOUS_MONITOR,
            PulseExecutionForm.AUTONOMOUS_SCAN_CONTINUOUS,
        ):
            if not self.target_ir.repeat_forever or schedules:
                raise ValueError(
                    "continuous execution must be cyclic and has no finite trigger schedule"
                )
        elif self.target_ir.repeat_forever:
            raise ValueError("finite execution form cannot carry cyclic TargetIR")
        expected_points = len(self.target_ir.scan_points) if scan else 1
        if any(item.point_count != expected_points for item in schedules):
            raise ValueError("trigger schedule point count differs from TargetIR")
        if (
            not self.target_ir.repeat_forever
            and bool(schedules)
            and not _same_physical_digital_trigger_schedules(
                self.target_ir,
                schedules,
            )
        ):
            raise ValueError("trigger schedules are not the deterministic TargetIR expansion")
        payload = encode(compiled_pulse_artifact_to_tree(self))
        admit_compiled_pulse_payload_size(len(payload))
        object.__setattr__(self, "_fingerprint", sha256_digest(payload))

    @property
    def target_abi_fingerprint(self) -> str:
        return self.target_ir.target_abi_fingerprint

    @property
    def max_configured_output_delay_ticks(self) -> int:
        return max(
            (0, *self.target_ir.channel_delays, *(item.delay_ticks for item in self.target_ir.bus_delays))
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

def compiled_pulse_artifact_to_tree(value: CompiledPulseArtifact) -> dict[str, object]:
    if not isinstance(value, CompiledPulseArtifact):
        raise TypeError("value must be CompiledPulseArtifact")
    return {
        "schema": COMPILED_PULSE_ARTIFACT_SCHEMA,
        "source_document_digest": value.source_document_digest,
        "compiler_id": value.compiler_id,
        "execution_form": value.execution_form.value,
        "target_ir": target_ir_to_tree(value.target_ir),
        "wire_image": pulse_wire_image_to_tree(value.wire_image),
        "trigger_schedules": [
            digital_trigger_schedule_to_tree(item) for item in value.trigger_schedules
        ],
    }


def compiled_pulse_artifact_from_tree(tree: object) -> CompiledPulseArtifact:
    fields = {
        "schema",
        "source_document_digest",
        "compiler_id",
        "execution_form",
        "target_ir",
        "wire_image",
        "trigger_schedules",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("CompiledPulseArtifact has an unknown field set")
    if tree["schema"] != COMPILED_PULSE_ARTIFACT_SCHEMA:
        raise ValueError("CompiledPulseArtifact schema differs")
    if not isinstance(tree["trigger_schedules"], list):
        raise TypeError("trigger_schedules must be a list")
    return CompiledPulseArtifact(
        tree["source_document_digest"],
        tree["compiler_id"],
        PulseExecutionForm(tree["execution_form"]),
        target_ir_from_tree(tree["target_ir"]),
        pulse_wire_image_from_tree(tree["wire_image"]),
        tuple(
            digital_trigger_schedule_from_tree(item)
            for item in tree["trigger_schedules"]
        ),
    )


def encode_compiled_pulse_artifact(value: CompiledPulseArtifact) -> bytes:
    payload = encode(compiled_pulse_artifact_to_tree(value))
    admit_compiled_pulse_payload_size(len(payload))
    return payload


def decode_compiled_pulse_artifact(
    payload: bytes | bytearray | memoryview,
) -> CompiledPulseArtifact:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("compiled pulse payload must be bytes-like")
    admit_compiled_pulse_payload_size(memoryview(payload).nbytes)
    raw = bytes(payload)
    value = compiled_pulse_artifact_from_tree(decode(raw))
    if encode_compiled_pulse_artifact(value) != raw:
        raise ValueError("CompiledPulseArtifact payload is not canonical")
    return value


def compiled_pulse_retained_upper_bound_bytes(
    value: CompiledPulseArtifact,
) -> int:
    """Bound the decoded immutable object graph from owner cardinalities."""

    if not isinstance(value, CompiledPulseArtifact):
        raise TypeError("value must be CompiledPulseArtifact")
    ir = value.target_ir
    slot_count = ir.slot_count
    logical_items = (
        len(ir.channels)
        + 2 * len(ir.ticks)
        + len(ir.tick_slot_coeffs) * slot_count
        + len(ir.scan_points) * slot_count
        + len(ir.scan_point_durations)
        + len(ir.slot_kinds)
        + len(ir.loop_end_slot_coeffs)
        + len(ir.channel_delays)
        + 2 * len(ir.logical_digital_outputs)
        + len(ir.bus_names)
        + len(ir.bus_safe_values)
        + len(ir.bus_delays) * 3
        + sum(12 + 2 * slot_count for _item in ir.bus_segments)
        + 2 * len(value.wire_image.words)
        + sum(8 + schedule.total for schedule in value.trigger_schedules)
    )
    packed_schedule_bytes = sum(
        schedule.point_indices.nbytes
        + schedule.loop_iterations.nbytes
        + schedule.ticks_from_run_start.nbytes
        for schedule in value.trigger_schedules
    )
    names = (
        value.compiler_id,
        *ir.channels,
        *ir.slot_kinds,
        *ir.bus_names,
        *(key for key, _lane in ir.logical_digital_outputs),
        *(lane for _key, lane in ir.logical_digital_outputs),
        *(schedule.channel for schedule in value.trigger_schedules),
    )
    name_bytes = sum(len(item.encode("utf-8")) for item in names)
    return int(
        _COMPILED_PULSE_RETAINED_FIXED_BYTES
        + _COMPILED_PULSE_LOGICAL_ITEM_BYTES * logical_items
        + 2 * packed_schedule_bytes
        + name_bytes
    )


def compiled_pulse_decode_peak_upper_bound_bytes(
    value: CompiledPulseArtifact,
    encoded_size: int,
) -> int:
    """Bound canonical JSON/array decode plus the retained pulse graph."""

    if isinstance(encoded_size, bool) or not isinstance(encoded_size, int):
        raise TypeError("encoded_size must be an integer")
    admit_compiled_pulse_payload_size(encoded_size)
    retained = compiled_pulse_retained_upper_bound_bytes(value)
    return int(_COMPILED_PULSE_DECODE_MULTIPLIER * encoded_size + 2 * retained)


def compiled_pulse_runtime_summary(
    value: CompiledPulseArtifact,
    *,
    encoded_size: int,
) -> CompiledPulseRuntimeSummary:
    """Build the canonical fail-fast summary for one encoded artifact."""

    from .physical import (
        PhysicalReadoutContextUnsupportedError,
        estimate_physical_waveform_index_peak_bytes,
        estimate_physical_window_projection_peak_bytes,
    )

    retained = compiled_pulse_retained_upper_bound_bytes(value)
    try:
        physical_index_peak = estimate_physical_waveform_index_peak_bytes(
            value.target_ir
        )
        physical_projection_peak = estimate_physical_window_projection_peak_bytes(
            value.target_ir
        )
    except PhysicalReadoutContextUnsupportedError:
        physical_index_peak = None
        physical_projection_peak = None
    return CompiledPulseRuntimeSummary(
        retained,
        compiled_pulse_decode_peak_upper_bound_bytes(value, encoded_size),
        (
            None
            if physical_index_peak is None or physical_projection_peak is None
            else (physical_index_peak, physical_projection_peak)
        ),
    )


def compiled_pulse_runtime_summary_to_tree(
    value: CompiledPulseRuntimeSummary,
) -> dict[str, object]:
    if not isinstance(value, CompiledPulseRuntimeSummary):
        raise TypeError("value must be CompiledPulseRuntimeSummary")
    bounds = value.readout_physical_context_bounds
    return {
        "retained_upper_bound_bytes": value.retained_upper_bound_bytes,
        "decode_peak_upper_bound_bytes": value.decode_peak_upper_bound_bytes,
        "physical_index_peak_upper_bound_bytes": (
            None if bounds is None else bounds[0]
        ),
        "physical_projection_peak_upper_bound_bytes": (
            None if bounds is None else bounds[1]
        ),
    }


def compiled_pulse_runtime_summary_from_tree(
    tree: object,
) -> CompiledPulseRuntimeSummary:
    data = exact_mapping(
        tree,
        {
            "retained_upper_bound_bytes",
            "decode_peak_upper_bound_bytes",
            "physical_index_peak_upper_bound_bytes",
            "physical_projection_peak_upper_bound_bytes",
        },
        "CompiledPulseRuntimeSummary",
        discriminator=None,
    )
    index = data["physical_index_peak_upper_bound_bytes"]
    projection = data["physical_projection_peak_upper_bound_bytes"]
    if (index is None) != (projection is None):
        raise ValueError(
            "CompiledPulseRuntimeSummary physical bounds must both be present or absent"
        )
    return CompiledPulseRuntimeSummary(
        data["retained_upper_bound_bytes"],
        data["decode_peak_upper_bound_bytes"],
        None if index is None else (index, projection),
    )


def admit_compiled_pulse_payload_size(size: int) -> int:
    """Own the current compiled-artifact byte ceiling across RPC and storage."""

    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise TypeError("compiled pulse payload size must be a non-negative integer")
    if size > MAX_COMPILED_PULSE_ARTIFACT_BYTES:
        raise ValueError(
            "compiled pulse artifact exceeds the payload limit of "
            f"{MAX_COMPILED_PULSE_ARTIFACT_BYTES} bytes"
        )
    return size


__all__ = [
    "COMPILED_PULSE_ARTIFACT_SCHEMA",
    "MAX_COMPILED_PULSE_ARTIFACT_BYTES",
    "CompiledPulseArtifact",
    "CompiledPulseRuntimeSummary",
    "PulseExecutionForm",
    "compiled_pulse_artifact_from_tree",
    "compiled_pulse_artifact_to_tree",
    "compiled_pulse_decode_peak_upper_bound_bytes",
    "compiled_pulse_retained_upper_bound_bytes",
    "compiled_pulse_runtime_summary",
    "compiled_pulse_runtime_summary_from_tree",
    "compiled_pulse_runtime_summary_to_tree",
    "admit_compiled_pulse_payload_size",
    "decode_compiled_pulse_artifact",
    "encode_compiled_pulse_artifact",
]
