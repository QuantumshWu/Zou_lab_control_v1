"""Current TaskConsole values for one typed pulse-scan product slice.

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
    DataTransformSpec,
    data_transform_spec_from_tree,
    data_transform_spec_to_tree,
)
from zlc_frontend import FormChoice, FormFieldProps, FormSpec
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
    PULSE_SCAN_TASK_KEY,
    SCAN_TASK_DEFINITIONS,
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    pulse_scan_program_from_tree,
    pulse_scan_program_to_tree,
)
from zlc_storage import (
    canonical_text,
    decode,
    encode,
    exact_mapping,
    positive_real,
)

# One package deeper than it used to live: the siblings stayed in ``zlc_workbench``.
from ..progressive_scan import ScanDisplayIntent


TASK_CONSOLE_SCAN_INTENT_FORMAT = "zlc_workbench.TaskConsoleScanIntent"
SCAN_INTENT_DEFAULT_CAMERA_ROLE = "camera"
SCAN_INTENT_DEFAULT_SEQUENCER_ROLE = "sequencer"
SCAN_INTENT_DEFAULT_TIMEOUT_SECONDS = 30.0


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


_CATALOG_GROUP_BY_DEFINITION = (
    (TaskDefinition, "Task"),
    (MeasurementDefinition, "Measurement"),
    (StreamProcessorDefinition, "Processor"),
)


def task_console_catalog_items(
    catalog: DefinitionCatalog,
) -> tuple[TaskConsoleCatalogItem, ...]:
    """Project every supplied Definition, rejecting silent catalog omissions.

    The projection is driven by each Definition's own type, never by a pinned
    key set, so a newly registered Definition reaches this product without a
    GUI edit.  A Definition kind this product cannot group is still refused
    rather than dropped, which is the omission this projection exists to catch.
    """

    if not isinstance(catalog, DefinitionCatalog):
        raise TypeError("catalog must be DefinitionCatalog")
    items = []
    for definition in catalog.definitions:
        group = next(
            (
                name
                for kind, name in _CATALOG_GROUP_BY_DEFINITION
                if type(definition) is kind
            ),
            None,
        )
        if group is None:
            raise TypeError(
                "TaskConsole cannot group definition "
                f"{getattr(definition, 'key', definition)} of type "
                f"{type(definition).__name__}"
            )
        items.append(
            TaskConsoleCatalogItem(definition.key, group, definition.title)
        )
    return tuple(items)


def _role_form_field(
    roles: tuple[str, ...],
    *,
    key: str,
    label: str,
    domain: str,
    preferred: str | None,
) -> FormFieldProps:
    """A role picker.  ``preferred=None`` means: leave it to the installation.

    Some requests carry their own role-resolution rule in the domain (a camera
    MONITOR prefers the free-running camera, which a capture does not).  Naming a
    default here would copy that rule into the form, where it would silently rot
    the day the domain's preference changes -- so an unset field submits nothing
    and the installation resolves the role from its own single source.
    """

    values = tuple(roles)
    if not values:
        raise ValueError(f"{domain} roles must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{domain} roles must be unique")
    for role in values:
        canonical_text(role, f"{domain} role")
    if preferred is None:
        return FormFieldProps(
            key,
            "choice",
            label,
            default=None,
            required=False,
            choices=tuple(FormChoice(role, role) for role in values),
            description=f"{domain} role; leave blank to let the installation resolve it",
        )
    default = preferred if preferred in values else values[0]
    return FormFieldProps(
        key,
        "choice",
        label,
        default=default,
        required=True,
        choices=tuple(FormChoice(role, role) for role in values),
        description=f"Frozen {domain} role resolved by the current installation",
    )


def task_console_scan_binding_form_spec(
    camera_roles: tuple[str, ...],
    sequencer_roles: tuple[str, ...],
) -> FormSpec:
    """Project owner facts plus one frozen installation role snapshot."""

    return FormSpec(
        (
            _role_form_field(
                camera_roles,
                key="camera_role",
                label="Camera role",
                domain="camera",
                preferred=SCAN_INTENT_DEFAULT_CAMERA_ROLE,
            ),
            _role_form_field(
                sequencer_roles,
                key="sequencer_role",
                label="Sequencer role",
                domain="sequencer",
                preferred=SCAN_INTENT_DEFAULT_SEQUENCER_ROLE,
            ),
            FormFieldProps(
                "trigger_channel",
                "text",
                "Trigger channel",
                default=None,
                description="Optional explicit camera-trigger channel",
            ),
        )
    )


def task_console_scan_runtime_form_spec() -> FormSpec:
    """Single UI projection of the whole-run deadline."""

    return FormSpec(
        (
            FormFieldProps(
                "timeout_seconds",
                "float",
                "Timeout seconds",
                default=SCAN_INTENT_DEFAULT_TIMEOUT_SECONDS,
                required=True,
                minimum=math.nextafter(0.0, math.inf),
                description="Positive whole-run deadline",
            ),
        )
    )


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    return canonical_text(value, field)


@dataclass(frozen=True, slots=True)
class TaskConsoleScanIntent:
    """Authored pulse-scan intent; runtime DeviceRefs never enter this value."""

    task_key: DefinitionKey
    measurement_key: DefinitionKey
    processor_key: DefinitionKey | None
    program: AutonomousScanSlotProgram | ApiSlotSegmentedProgram
    camera_role: str = SCAN_INTENT_DEFAULT_CAMERA_ROLE
    sequencer_role: str = SCAN_INTENT_DEFAULT_SEQUENCER_ROLE
    trigger_channel: str | None = None
    calibration_ref: CalibrationArtifactRef | None = None
    model_kind: ReadoutModelKind | None = None
    output_transform_spec: DataTransformSpec | None = None
    display_intent: ScanDisplayIntent = ScanDisplayIntent()
    timeout_seconds: float = SCAN_INTENT_DEFAULT_TIMEOUT_SECONDS

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
        if self.task_key != PULSE_SCAN_TASK_KEY:
            raise ValueError("TaskConsole currently supports only the Pulse scan Task")
        if self.measurement_key != CAMERA_MEASUREMENT_KEY:
            raise ValueError("TaskConsole scan source must be the camera measurement")
        if self.processor_key not in (None, OCCUPANCY_STREAM_PROCESSOR_KEY):
            raise ValueError("TaskConsole scan processor must be occupancy or absent")
        if not isinstance(
            self.program,
            (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
        ):
            raise TypeError("program must be a current pulse-scan program")

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
            "timeout_seconds",
            positive_real(self.timeout_seconds, "timeout_seconds"),
        )

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
        "program": pulse_scan_program_to_tree(intent.program),
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
            "program",
            "camera_role",
            "sequencer_role",
            "trigger_channel",
            "calibration_ref",
            "model_kind",
            "output_transform_spec",
            "display_intent",
            "timeout_seconds",
        },
        TASK_CONSOLE_SCAN_INTENT_FORMAT,
    )
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
        program=pulse_scan_program_from_tree(data["program"]),
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
    "SCAN_INTENT_DEFAULT_CAMERA_ROLE",
    "SCAN_INTENT_DEFAULT_SEQUENCER_ROLE",
    "SCAN_INTENT_DEFAULT_TIMEOUT_SECONDS",
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
    "encode_task_console_scan_intent",
    "load_task_console_scan_intent",
    "save_task_console_scan_intent",
    "task_console_scan_binding_form_spec",
    "task_console_scan_runtime_form_spec",
    "task_console_catalog_items",
    "task_console_scan_intent_from_tree",
    "task_console_scan_intent_to_tree",
]
