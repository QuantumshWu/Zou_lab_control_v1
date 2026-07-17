"""Current TaskConsole values for one autonomous SCAN_SLOT product slice.

This module owns authored UI intent, optimistic edit revisioning, and its strict
current-only workspace codec.  It deliberately does not own device resolution,
Run lifecycle, Qt widgets, or a generic processor graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import threading
from uuid import uuid4

from zlc_data import (
    CoordinateRangeSelection,
    DataTransformSpec,
    IndexRangeSelection,
    IndexSelection,
    ReductionSpec,
    Selection,
    data_transform_spec_from_tree,
    data_transform_spec_to_tree,
)
from zlc_neutral_atom.catalog import (
    DefinitionCatalog,
    DefinitionKey,
    MeasurementDefinition,
    StreamProcessorDefinition,
    TaskDefinition,
    definition_key_from_tree,
    definition_key_to_tree,
)
from zlc_neutral_atom.acquisition import (
    CAMERA_MEASUREMENT_DEFINITIONS,
    CAMERA_MEASUREMENT_KEY,
)
from zlc_neutral_atom.readout.calibration import ReadoutModelKind
from zlc_neutral_atom.readout.calibration_reference import (
    CalibrationArtifactRef,
    calibration_artifact_ref_from_tree,
    calibration_artifact_ref_to_tree,
)
from zlc_neutral_atom.readout.occupancy import (
    OCCUPANCY_STREAM_PROCESSOR_DEFINITIONS,
    OCCUPANCY_STREAM_PROCESSOR_KEY,
)
from zlc_neutral_atom.scan.contracts import (
    AUTONOMOUS_SCAN_SLOT_TASK_KEY,
    SCAN_TASK_DEFINITIONS,
    ScanPointTable,
)
from zlc_pulse import (
    PulseDocument,
    pulse_document_from_tree,
    pulse_document_to_tree,
    resolve_api_parameters,
)
from zlc_storage import (
    canonical_text,
    decode,
    encode,
    exact_mapping,
    positive_integer,
    positive_real,
)

from .progressive_scan import ScanDisplayIntent


TASK_CONSOLE_SCAN_INTENT_FORMAT = "zlc_workbench.TaskConsoleScanIntent"


@dataclass(frozen=True, slots=True)
class TaskConsoleCatalogItem:
    """Workbench-only grouping for one explicitly projected domain Definition."""

    key: DefinitionKey
    group: str
    title: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, DefinitionKey):
            raise TypeError("key must be DefinitionKey")
        object.__setattr__(self, "group", canonical_text(self.group, "group"))
        object.__setattr__(self, "title", canonical_text(self.title, "title"))


def compose_task_console_catalog() -> DefinitionCatalog:
    """Compose exactly the three current capabilities used by this product."""

    return DefinitionCatalog.compose(
        SCAN_TASK_DEFINITIONS,
        CAMERA_MEASUREMENT_DEFINITIONS,
        OCCUPANCY_STREAM_PROCESSOR_DEFINITIONS,
    )


def task_console_catalog_items(
    catalog: DefinitionCatalog,
) -> tuple[TaskConsoleCatalogItem, ...]:
    """Project every supplied Definition, rejecting silent catalog omissions."""

    if not isinstance(catalog, DefinitionCatalog):
        raise TypeError("catalog must be DefinitionCatalog")
    expected = {
        AUTONOMOUS_SCAN_SLOT_TASK_KEY,
        CAMERA_MEASUREMENT_KEY,
        OCCUPANCY_STREAM_PROCESSOR_KEY,
    }
    actual = set(catalog.by_key)
    if actual != expected:
        raise ValueError(
            "TaskConsole catalog projection is incomplete; "
            f"missing={sorted(map(str, expected - actual))}, "
            f"unexpected={sorted(map(str, actual - expected))}"
        )
    task = catalog.resolve(AUTONOMOUS_SCAN_SLOT_TASK_KEY, TaskDefinition)
    measurement = catalog.resolve(CAMERA_MEASUREMENT_KEY, MeasurementDefinition)
    processor = catalog.resolve(
        OCCUPANCY_STREAM_PROCESSOR_KEY,
        StreamProcessorDefinition,
    )
    return (
        TaskConsoleCatalogItem(task.key, "Task", task.title),
        TaskConsoleCatalogItem(measurement.key, "Measurement", measurement.title),
        TaskConsoleCatalogItem(processor.key, "Processor", processor.title),
    )


def _number(value: object, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be an int or float")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    return value


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    return canonical_text(value, field)


def describe_authoritative_transform(
    spec: DataTransformSpec | None,
) -> str:
    """Describe every persisted authority operation without inferring axes."""

    if spec is None:
        return "None · no user-authored Select/Reduce"
    if not isinstance(spec, DataTransformSpec):
        raise TypeError("spec must be DataTransformSpec or None")
    operations: list[str] = []
    for operation in spec.operations:
        if isinstance(operation, Selection):
            terms: list[str] = []
            for term in operation.terms:
                axis = term.axis_id.value
                if isinstance(term, IndexSelection):
                    terms.append(f"{axis}=index[{term.index}]")
                elif isinstance(term, IndexRangeSelection):
                    terms.append(f"{axis}=indices[{term.start}:{term.stop}]")
                elif isinstance(term, CoordinateRangeSelection):
                    frame = (
                        ""
                        if term.coordinate_frame is None
                        else f"@{term.coordinate_frame.value}"
                    )
                    terms.append(
                        f"{axis}=coordinates[{term.lower},{term.upper}]{frame}"
                    )
                else:  # pragma: no cover - Selection owns the closed term union.
                    raise TypeError("Selection contains an unsupported term")
            operations.append("select(" + ", ".join(terms) + ")")
        elif isinstance(operation, ReductionSpec):
            axes = ",".join(axis_id.value for axis_id in operation.axis_ids)
            minimum = (
                ""
                if operation.minimum_valid_count is None
                else f"/min={operation.minimum_valid_count}"
            )
            operations.append(
                f"reduce({axes})={operation.method.value}"
                f"/{operation.missing_policy.value}"
                f"/{operation.validity_policy.value}"
                f"{minimum}"
            )
        else:  # pragma: no cover - DataTransformSpec owns the closed union.
            raise TypeError("DataTransformSpec contains an unsupported operation")
    return "AUTHORITATIVE · " + " → ".join(operations)


@dataclass(frozen=True, slots=True)
class TaskConsoleScanIntent:
    """Authored SCAN_SLOT intent; runtime DeviceRefs never enter this value."""

    task_key: DefinitionKey
    measurement_key: DefinitionKey
    processor_key: DefinitionKey | None
    pulse_document: PulseDocument
    api_values: tuple[tuple[str, int | float], ...]
    camera_role: str
    sequencer_role: str
    trigger_channel: str | None = None
    calibration_ref: CalibrationArtifactRef | None = None
    model_kind: ReadoutModelKind | None = None
    output_transform_spec: DataTransformSpec | None = None
    display_intent: ScanDisplayIntent = ScanDisplayIntent()
    transport_memory_limit_bytes: int = 64 << 20
    memory_limit_bytes: int = 512 << 20
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.task_key, DefinitionKey):
            raise TypeError("task_key must be DefinitionKey")
        if not isinstance(self.measurement_key, DefinitionKey):
            raise TypeError("measurement_key must be DefinitionKey")
        if self.processor_key is not None and not isinstance(
            self.processor_key,
            DefinitionKey,
        ):
            raise TypeError("processor_key must be DefinitionKey or None")
        if self.task_key != AUTONOMOUS_SCAN_SLOT_TASK_KEY:
            raise ValueError("TaskConsole currently supports only Autonomous SCAN_SLOT")
        if self.measurement_key != CAMERA_MEASUREMENT_KEY:
            raise ValueError("TaskConsole scan source must be the camera measurement")
        if self.processor_key not in (None, OCCUPANCY_STREAM_PROCESSOR_KEY):
            raise ValueError("TaskConsole scan processor must be occupancy or absent")
        if not isinstance(self.pulse_document, PulseDocument):
            raise TypeError("pulse_document must be PulseDocument")
        # This validates the complete named point layout without flattening it.
        ScanPointTable.from_pulse_document(self.pulse_document)

        supplied = tuple(self.api_values)
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            for item in supplied
        ):
            raise TypeError("api_values must contain (parameter_id, value) tuples")
        supplied_map: dict[str, int | float] = {}
        for parameter_id, value in supplied:
            key = canonical_text(parameter_id, "API parameter_id")
            if key in supplied_map:
                raise ValueError(f"duplicate API value {key!r}")
            supplied_map[key] = _number(value, f"API value {key!r}")
        expected = tuple(
            parameter.parameter_id for parameter in self.pulse_document.api_parameters
        )
        if set(supplied_map) != set(expected):
            missing = tuple(key for key in expected if key not in supplied_map)
            extra = tuple(key for key in supplied_map if key not in set(expected))
            raise ValueError(
                "SCAN_SLOT requires exactly one whole-run value for every API "
                f"parameter; missing={missing}, extra={extra}"
            )
        resolved = resolve_api_parameters(self.pulse_document, supplied_map)
        if resolved.api_parameters:
            raise AssertionError("pulse owner left declared API parameters unresolved")
        object.__setattr__(
            self,
            "api_values",
            tuple((key, supplied_map[key]) for key in expected),
        )

        object.__setattr__(
            self,
            "camera_role",
            canonical_text(self.camera_role, "camera_role"),
        )
        object.__setattr__(
            self,
            "sequencer_role",
            canonical_text(self.sequencer_role, "sequencer_role"),
        )
        object.__setattr__(
            self,
            "trigger_channel",
            _optional_text(self.trigger_channel, "trigger_channel"),
        )
        if self.processor_key is None:
            if self.calibration_ref is not None or self.model_kind is not None:
                raise ValueError(
                    "direct-camera intent cannot carry occupancy calibration/model"
                )
        else:
            if not isinstance(self.calibration_ref, CalibrationArtifactRef):
                raise TypeError(
                    "processor intent requires a CalibrationArtifactRef"
                )
            if self.model_kind is not None and not isinstance(
                self.model_kind,
                ReadoutModelKind,
            ):
                raise TypeError("model_kind must be ReadoutModelKind or None")
        if self.output_transform_spec is not None:
            if not isinstance(self.output_transform_spec, DataTransformSpec):
                raise TypeError(
                    "output_transform_spec must be DataTransformSpec or None"
                )
            if not self.output_transform_spec.operations:
                raise ValueError("empty output_transform_spec must be None")
        if not isinstance(self.display_intent, ScanDisplayIntent):
            raise TypeError("display_intent must be ScanDisplayIntent")
        if self.processor_key is None and self.display_intent != ScanDisplayIntent():
            raise ValueError("direct-camera intent has no SITE display choice")
        object.__setattr__(
            self,
            "transport_memory_limit_bytes",
            positive_integer(
                self.transport_memory_limit_bytes,
                "transport_memory_limit_bytes",
            ),
        )
        object.__setattr__(
            self,
            "memory_limit_bytes",
            positive_integer(self.memory_limit_bytes, "memory_limit_bytes"),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            positive_real(self.timeout_seconds, "timeout_seconds"),
        )

    @property
    def fixed_api_values(self) -> dict[str, int | float]:
        return dict(self.api_values)


@dataclass(frozen=True, slots=True)
class ScanEditSnapshot:
    revision: int
    intent: TaskConsoleScanIntent

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision must be an integer")
        if self.revision < 0:
            raise ValueError("revision must be nonnegative")
        if not isinstance(self.intent, TaskConsoleScanIntent):
            raise TypeError("intent must be TaskConsoleScanIntent")


@dataclass(frozen=True, slots=True)
class ScanEditDraft:
    base_revision: int
    intent: TaskConsoleScanIntent

    def __post_init__(self) -> None:
        if isinstance(self.base_revision, bool) or not isinstance(
            self.base_revision,
            int,
        ):
            raise TypeError("base_revision must be an integer")
        if self.base_revision < 0:
            raise ValueError("base_revision must be nonnegative")
        if not isinstance(self.intent, TaskConsoleScanIntent):
            raise TypeError("intent must be TaskConsoleScanIntent")


class ScanEditConflict(RuntimeError):
    """An editor attempted last-write-wins against a newer applied revision."""


class ScanEditorSession:
    """Owner-thread optimistic edit session shared by Setting and Edit views."""

    __slots__ = ("_owner_thread", "_revision", "_intent")

    def __init__(self, intent: TaskConsoleScanIntent) -> None:
        if not isinstance(intent, TaskConsoleScanIntent):
            raise TypeError("intent must be TaskConsoleScanIntent")
        self._owner_thread = threading.get_ident()
        self._revision = 0
        self._intent = intent

    def snapshot(self) -> ScanEditSnapshot:
        self._require_owner()
        return ScanEditSnapshot(self._revision, self._intent)

    def begin(self) -> ScanEditDraft:
        snapshot = self.snapshot()
        return ScanEditDraft(snapshot.revision, snapshot.intent)

    def apply(self, draft: ScanEditDraft) -> ScanEditSnapshot:
        self._require_owner()
        if not isinstance(draft, ScanEditDraft):
            raise TypeError("draft must be ScanEditDraft")
        if draft.base_revision != self._revision:
            raise ScanEditConflict(
                f"edit base revision {draft.base_revision} is stale; "
                f"current revision is {self._revision}"
            )
        self._intent = draft.intent
        self._revision += 1
        return self.snapshot()

    def cancel(self, draft: ScanEditDraft) -> ScanEditSnapshot:
        self._require_owner()
        if not isinstance(draft, ScanEditDraft):
            raise TypeError("draft must be ScanEditDraft")
        return self.snapshot()

    def _require_owner(self) -> None:
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("ScanEditorSession methods require its owner thread")


def task_console_scan_intent_to_tree(
    intent: TaskConsoleScanIntent,
) -> dict[str, object]:
    if not isinstance(intent, TaskConsoleScanIntent):
        raise TypeError("intent must be TaskConsoleScanIntent")
    return {
        "schema": TASK_CONSOLE_SCAN_INTENT_FORMAT,
        "task_key": definition_key_to_tree(intent.task_key),
        "measurement_key": definition_key_to_tree(intent.measurement_key),
        "processor_key": (
            None
            if intent.processor_key is None
            else definition_key_to_tree(intent.processor_key)
        ),
        "pulse_document": pulse_document_to_tree(intent.pulse_document),
        "api_values": [
            {"parameter_id": key, "value": value}
            for key, value in intent.api_values
        ],
        "camera_role": intent.camera_role,
        "sequencer_role": intent.sequencer_role,
        "trigger_channel": intent.trigger_channel,
        "calibration_ref": (
            None
            if intent.calibration_ref is None
            else calibration_artifact_ref_to_tree(intent.calibration_ref)
        ),
        "model_kind": (
            None if intent.model_kind is None else intent.model_kind.value
        ),
        "output_transform_spec": (
            None
            if intent.output_transform_spec is None
            else data_transform_spec_to_tree(intent.output_transform_spec)
        ),
        "display_intent": {
            "site_mode": intent.display_intent.site_mode,
            "site_index": intent.display_intent.site_index,
        },
        "transport_memory_limit_bytes": intent.transport_memory_limit_bytes,
        "memory_limit_bytes": intent.memory_limit_bytes,
        "timeout_seconds": intent.timeout_seconds,
    }


def task_console_scan_intent_from_tree(tree: object) -> TaskConsoleScanIntent:
    data = exact_mapping(
        tree,
        {
            "schema",
            "task_key",
            "measurement_key",
            "processor_key",
            "pulse_document",
            "api_values",
            "camera_role",
            "sequencer_role",
            "trigger_channel",
            "calibration_ref",
            "model_kind",
            "output_transform_spec",
            "display_intent",
            "transport_memory_limit_bytes",
            "memory_limit_bytes",
            "timeout_seconds",
        },
        TASK_CONSOLE_SCAN_INTENT_FORMAT,
    )
    if not isinstance(data["api_values"], list):
        raise TypeError("api_values must be a list")
    api_values: list[tuple[str, int | float]] = []
    for raw in data["api_values"]:
        item = exact_mapping(
            raw,
            {"parameter_id", "value"},
            "API value",
            discriminator=None,
        )
        api_values.append((item["parameter_id"], item["value"]))
    display = exact_mapping(
        data["display_intent"],
        {"site_mode", "site_index"},
        "ScanDisplayIntent",
        discriminator=None,
    )
    value = TaskConsoleScanIntent(
        task_key=definition_key_from_tree(data["task_key"]),
        measurement_key=definition_key_from_tree(data["measurement_key"]),
        processor_key=(
            None
            if data["processor_key"] is None
            else definition_key_from_tree(data["processor_key"])
        ),
        pulse_document=pulse_document_from_tree(data["pulse_document"]),
        api_values=tuple(api_values),
        camera_role=data["camera_role"],
        sequencer_role=data["sequencer_role"],
        trigger_channel=data["trigger_channel"],
        calibration_ref=(
            None
            if data["calibration_ref"] is None
            else calibration_artifact_ref_from_tree(data["calibration_ref"])
        ),
        model_kind=(
            None
            if data["model_kind"] is None
            else ReadoutModelKind(data["model_kind"])
        ),
        output_transform_spec=(
            None
            if data["output_transform_spec"] is None
            else data_transform_spec_from_tree(data["output_transform_spec"])
        ),
        display_intent=ScanDisplayIntent(
            display["site_mode"],
            display["site_index"],
        ),
        transport_memory_limit_bytes=data["transport_memory_limit_bytes"],
        memory_limit_bytes=data["memory_limit_bytes"],
        timeout_seconds=data["timeout_seconds"],
    )
    if task_console_scan_intent_to_tree(value) != tree:
        raise ValueError("TaskConsoleScanIntent tree is typed but non-canonical")
    return value


def encode_task_console_scan_intent(intent: TaskConsoleScanIntent) -> bytes:
    return encode(task_console_scan_intent_to_tree(intent))


def decode_task_console_scan_intent(payload: bytes | bytearray | memoryview):
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("TaskConsoleScanIntent payload must be bytes-like")
    raw = bytes(payload)
    value = task_console_scan_intent_from_tree(decode(raw))
    if encode_task_console_scan_intent(value) != raw:
        raise ValueError("TaskConsoleScanIntent payload is typed but non-canonical")
    return value


def save_task_console_scan_intent(
    intent: TaskConsoleScanIntent,
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser().resolve()
    payload = encode_task_console_scan_intent(intent)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{uuid4().hex}"
    )
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_task_console_scan_intent(path: str | Path) -> TaskConsoleScanIntent:
    return decode_task_console_scan_intent(Path(path).expanduser().resolve().read_bytes())


__all__ = [
    "ScanDisplayIntent",
    "ScanEditConflict",
    "ScanEditDraft",
    "ScanEditSnapshot",
    "ScanEditorSession",
    "TASK_CONSOLE_SCAN_INTENT_FORMAT",
    "TaskConsoleCatalogItem",
    "TaskConsoleScanIntent",
    "compose_task_console_catalog",
    "decode_task_console_scan_intent",
    "describe_authoritative_transform",
    "encode_task_console_scan_intent",
    "load_task_console_scan_intent",
    "save_task_console_scan_intent",
    "task_console_catalog_items",
    "task_console_scan_intent_from_tree",
    "task_console_scan_intent_to_tree",
]
