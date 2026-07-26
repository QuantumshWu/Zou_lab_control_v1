"""Definition, request, and output semantics for Camera Measurement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from zlc_data import (
    MONITOR_HISTORY,
    READOUT_EVENT,
    BlockId,
    DatasetRevisionRef,
    DatasetSchema,
    IndexSelection,
    OwnedSnapshot,
    Selection,
    dataset_revision_ref_to_tree,
    materialize_dataset_selection,
)
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.catalog import DefinitionKey, MeasurementDefinition
from zlc_neutral_atom.logic_node_declaration import (
    DynamicChoicePresentation,
    LogicNodeDeclaration,
    OutputPresentation,
)
from zlc_neutral_atom.node_input import bind_no_node_inputs
from zlc_neutral_atom.capture.reference import (
    CaptureArtifactRef,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
    final_dataset_join_digest,
)
from zlc_neutral_atom.runtime.dataset import (
    MonitorCoverage,
    MonitorDatasetSnapshot,
)
from zlc_neutral_atom.runtime.streams import event_ref_to_tree
from zlc_storage import canonical_digest, canonical_text, positive_integer
from zlc_storage import positive_real

from zlc_neutral_atom.authoring import (
    MINIMUM_POSITIVE_FLOAT,
    AuthoringChoice,
    AuthoringField,
    AuthoringSchema,
)


DEFAULT_CAMERA_MEASUREMENT_ROLE = "camera"
CAMERA_MEASUREMENT_ROLE_ORDER = ("camera", "mot_camera")
DEFAULT_CAMERA_MEASUREMENT_REPEAT = 0
DEFAULT_CAMERA_FRAMES_PER_CYCLE = 1
DEFAULT_CAMERA_MONITOR_HISTORY_CYCLES = 8
MINIMUM_CAMERA_MEASUREMENT_REPEAT = 0
MINIMUM_CAMERA_FRAMES_PER_CYCLE = 1
CAMERA_FRAME_OUTPUT_CONTRACT_ID = (
    "zlc_neutral_atom.camera-measurement.current-frame"
)
CAMERA_MEASUREMENT_KEY = DefinitionKey(
    "zlc_neutral_atom.logic_nodes.camera_measurement",
    "camera-measurement",
)
CAMERA_MEASUREMENT_DEFINITION = MeasurementDefinition(
    CAMERA_MEASUREMENT_KEY,
    "Camera",
    "zlc.camera-measurement-request",
    "zlc.camera-measurement-binding",
)


_CAMERA_MEASUREMENT_AUTHORING_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "camera_role",
            "choice",
            "Camera",
            required=True,
            dynamic_choices=True,
            description="Frozen camera role from the current installation",
        ),
        AuthoringField(
            "frames_per_cycle",
            "int",
            "Frames per cycle",
            default=DEFAULT_CAMERA_FRAMES_PER_CYCLE,
            required=True,
            minimum=MINIMUM_CAMERA_FRAMES_PER_CYCLE,
            allow_blank=False,
            description=(
                "Ordered camera frames retained on an explicit READOUT_EVENT axis"
            ),
        ),
        AuthoringField(
            "exposure",
            "float",
            "Exposure",
            default=None,
            required=False,
            unit="s",
            minimum=MINIMUM_POSITIVE_FLOAT,
            allow_blank=True,
            description=(
                "Exposure applied and read back for this Camera run; blank keeps "
                "the selected camera's installed working point"
            ),
        ),
        AuthoringField(
            "repeat",
            "int",
            "Repeat",
            default=DEFAULT_CAMERA_MEASUREMENT_REPEAT,
            required=True,
            minimum=MINIMUM_CAMERA_MEASUREMENT_REPEAT,
            allow_blank=False,
            description=(
                "0 keeps this installed camera live; a positive value performs "
                "that many exact finite capture cycles"
            ),
        ),
    )
)


def camera_measurement_authoring_schema() -> AuthoringSchema:
    """Return the one scalar authoring declaration beside the typed request."""

    return _CAMERA_MEASUREMENT_AUTHORING_SCHEMA


def camera_measurement_roles(installed_roles) -> tuple[str, ...]:
    """Filter installed Camera roles into the product's declared visible order."""

    roles = tuple(installed_roles)
    if len(set(roles)) != len(roles):
        raise ValueError("installed camera roles must be unique")
    for role in roles:
        canonical_text(role, "installed camera role")
    installed = set(roles)
    return tuple(role for role in CAMERA_MEASUREMENT_ROLE_ORDER if role in installed)


def camera_measurement_default_role(available_roles) -> str | None:
    """Select the request owner's default from one frozen role projection."""

    roles = tuple(available_roles)
    if len(set(roles)) != len(roles):
        raise ValueError("available camera roles must be unique")
    for role in roles:
        canonical_text(role, "available camera role")
    if DEFAULT_CAMERA_MEASUREMENT_ROLE in roles:
        return DEFAULT_CAMERA_MEASUREMENT_ROLE
    return roles[0] if roles else None


@dataclass(frozen=True, slots=True)
class CameraMeasurementIntent:
    """Device-independent Camera authoring frozen before installation binding."""

    camera_role: str
    frames_per_cycle: int = DEFAULT_CAMERA_FRAMES_PER_CYCLE
    exposure_seconds: float | None = None
    repeat: int = DEFAULT_CAMERA_MEASUREMENT_REPEAT

    def __post_init__(self) -> None:
        canonical_text(self.camera_role, "camera_role")
        object.__setattr__(
            self,
            "frames_per_cycle",
            positive_integer(self.frames_per_cycle, "frames_per_cycle"),
        )
        if isinstance(self.repeat, bool) or not isinstance(self.repeat, int):
            raise TypeError("repeat must be an integer")
        if self.repeat < MINIMUM_CAMERA_MEASUREMENT_REPEAT:
            raise ValueError("repeat must be non-negative")
        if self.exposure_seconds is not None:
            object.__setattr__(
                self,
                "exposure_seconds",
                positive_real(self.exposure_seconds, "exposure_seconds"),
            )

    @property
    def output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return camera_frame_output_declarations(self.frames_per_cycle)


def build_camera_measurement_intent_from_authoring(
    values: Mapping[str, object],
) -> CameraMeasurementIntent:
    """Freeze Camera leaves without resolving an installed DeviceRef."""

    authored = camera_measurement_authoring_schema().freeze(values)
    return CameraMeasurementIntent(
        camera_role=authored["camera_role"],
        frames_per_cycle=authored["frames_per_cycle"],
        exposure_seconds=authored["exposure"],
        repeat=authored["repeat"],
    )


def camera_frame_output_index(output_name: str) -> int:
    """Parse one canonical ``frame_i`` name without accepting aliases."""

    name = canonical_text(output_name, "camera frame output name")
    prefix = "frame_"
    if not name.startswith(prefix):
        raise ValueError("camera frame output name must start with 'frame_'")
    token = name[len(prefix) :]
    if not token.isascii() or not token.isdecimal():
        raise ValueError("camera frame output index must be a decimal integer")
    index = int(token)
    if token != str(index):
        raise ValueError("camera frame output index must use canonical decimal text")
    return index


def camera_frame_output_declarations(
    frames_per_cycle: int,
) -> tuple[DatasetOutputDeclaration, ...]:
    """Return owner-paired public outputs for one camera cycle.

    A camera cycle is stored atomically as one Dataset whose named
    ``READOUT_EVENT`` axis has ``frames_per_cycle`` cells.  Presentation may
    expose those cells independently, but their names are owned here beside
    the request that defines the cycle -- never guessed from an ndarray shape
    and never supplemented with a lossy ``frame`` alias.
    """

    count = positive_integer(frames_per_cycle, "frames_per_cycle")
    return tuple(
        DatasetOutputDeclaration(
            f"frame_{index}",
            CAMERA_FRAME_OUTPUT_CONTRACT_ID,
        )
        for index in range(count)
    )


def _camera_frame_ref(
    source: OwnedSnapshot,
    event_axis_id: str,
    event_index: int,
    output_schema: DatasetSchema,
) -> DatasetRevisionRef:
    identity = canonical_digest(
        {
            "owner": "zlc_neutral_atom.camera-measurement.frame-output",
            "source_block_id": source.ref.block_id.value,
            "readout_event_axis_id": event_axis_id,
            "readout_event_index": event_index,
        }
    )
    return DatasetRevisionRef(
        BlockId(f"camera-frame-{identity[:32]}"),
        source.ref.stream_generation,
        output_schema.fingerprint,
        source.ref.revision,
    )


def project_camera_measurement_outputs(
    source: OwnedSnapshot,
    request: "CameraMeasurementRequest",
) -> dict[str, OwnedSnapshot]:
    """Materialize the Camera Measurement's declared ``frame_i`` outputs.

    The measurement contract, rather than a GUI, owns how one atomic camera
    cycle maps its named ``READOUT_EVENT`` cells to public outputs.  Repeat,
    remaining point axes, trailing data axes, component validity, revision and
    stream generation are retained exactly.
    """

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("camera frame source must be OwnedSnapshot")
    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError("camera output projection requires CameraMeasurementRequest")
    schema = source.block.schema
    event_axes = tuple(
        axis for axis in schema.point_axes if axis.role == READOUT_EVENT
    )
    if len(event_axes) != 1:
        raise ValueError(
            "camera Dataset must contain exactly one READOUT_EVENT point axis"
        )
    if any(axis.role == READOUT_EVENT for axis in schema.cell_schema.data_axes):
        raise ValueError("READOUT_EVENT cannot be a trailing camera data axis")
    event_axis = event_axes[0]
    if event_axis.size != request.frames_per_cycle:
        raise ValueError(
            "camera Dataset READOUT_EVENT size differs from frames_per_cycle"
        )

    expected_point_axes = tuple(
        axis for axis in schema.point_axes if axis.axis_id != event_axis.axis_id
    )
    projected: dict[str, OwnedSnapshot] = {}
    for event_index, output_name in enumerate(request.output_names):
        snapshot = materialize_dataset_selection(
            source,
            Selection.index(event_axis.axis_id, event_index),
            reference_for=lambda output_schema, index=event_index: _camera_frame_ref(
                source,
                event_axis.axis_id.value,
                index,
                output_schema,
            ),
        )
        output_schema = snapshot.block.schema
        if output_schema.repeat_axis != schema.repeat_axis:
            raise RuntimeError("camera output projection changed the repeat axis")
        if output_schema.point_axes != expected_point_axes:
            raise RuntimeError("camera output projection changed another point axis")
        if output_schema.cell_schema != schema.cell_schema:
            raise RuntimeError("camera output projection changed trailing data axes")
        if snapshot.ref.stream_generation != source.ref.stream_generation:
            raise RuntimeError("camera output projection changed stream generation")
        projected[output_name] = snapshot

    if tuple(projected) != request.output_names:
        raise RuntimeError("camera output projection order is inconsistent")
    return projected


def camera_measurement_final_outputs(
    reference: CaptureArtifactRef,
    source: OwnedSnapshot,
    request: "CameraMeasurementRequest",
) -> dict[str, FinalDatasetOutput]:
    """Publish every declared Camera frame from one admitted FINAL capture."""

    if not isinstance(reference, CaptureArtifactRef):
        raise TypeError("Camera FINAL reference must be CaptureArtifactRef")
    snapshots = project_camera_measurement_outputs(source, request)
    source_identity = capture_artifact_ref_to_tree(reference)
    declarations = request.output_declarations
    return {
        declaration.name: FinalDatasetOutput(
            declaration,
            snapshot,
            final_dataset_join_digest(
                owner="camera-measurement",
                declaration=declaration,
                source_identity=source_identity,
                snapshot=snapshot,
            ),
        )
        for declaration, snapshot in zip(
            declarations,
            snapshots.values(),
            strict=True,
        )
    }


def project_camera_monitor_outputs(
    source: MonitorDatasetSnapshot,
    request: "CameraMeasurementRequest",
) -> dict[str, LiveDatasetOutput]:
    """Publish the newest atomic monitor cycle as declared ``frame_i`` values.

    ``MONITOR_HISTORY`` is private rolling-storage geometry, not a physical
    point axis of a public Camera signal.  The frozen monitor orders that axis
    newest-first, so every output selects history index zero together with its
    declared ``READOUT_EVENT`` index.  The resulting public signal therefore
    retains the universal ``(R=1, P=1, *data_shape)`` contract regardless of
    the configured history capacity.
    """

    if not isinstance(source, MonitorDatasetSnapshot):
        raise TypeError("source must be MonitorDatasetSnapshot")
    source_schema = source.snapshot.block.schema
    history_axes = tuple(
        axis for axis in source_schema.point_axes if axis.role == MONITOR_HISTORY
    )
    event_axes = tuple(
        axis for axis in source_schema.point_axes if axis.role == READOUT_EVENT
    )
    if len(history_axes) != 1:
        raise ValueError("Camera monitor must contain one MONITOR_HISTORY axis")
    if len(event_axes) != 1:
        raise ValueError("Camera monitor must contain one READOUT_EVENT axis")
    if any(
        axis.role not in {MONITOR_HISTORY, READOUT_EVENT}
        for axis in source_schema.point_axes
    ):
        raise ValueError("Camera monitor contains an unsupported public point axis")
    history_axis = history_axes[0]
    event_axis = event_axes[0]
    if event_axis.size != request.frames_per_cycle:
        raise ValueError(
            "Camera monitor READOUT_EVENT size differs from frames_per_cycle"
        )

    projected: dict[str, LiveDatasetOutput] = {}
    for declaration in request.output_declarations:
        output_name = declaration.name
        event_index = camera_frame_output_index(output_name)
        snapshot = materialize_dataset_selection(
            source.snapshot,
            Selection(
                (
                    IndexSelection(history_axis.axis_id, 0),
                    IndexSelection(event_axis.axis_id, event_index),
                )
            ),
            reference_for=lambda output_schema, index=event_index: _camera_frame_ref(
                source.snapshot,
                event_axis.axis_id.value,
                index,
                output_schema,
            ),
        )
        output_schema = snapshot.block.schema
        if output_schema.point_axes:
            raise RuntimeError("Camera monitor storage axes leaked into a public frame")
        source_logical_point = tuple(
            0 if axis.role == MONITOR_HISTORY else event_index
            for axis in source_schema.point_axes
        )
        source_storage_index = source_schema.point_layout.storage_index(
            source_logical_point
        )
        selected_refs = tuple(
            source.event_refs[
                repeat_index * source_schema.point_layout.storage_size
                + source_storage_index
            ]
            for repeat_index in range(output_schema.repeat_axis.size)
        )
        total = (
            output_schema.repeat_axis.size
            * output_schema.point_layout.storage_size
        )
        if len(selected_refs) != total:
            raise RuntimeError("Camera monitor projection lost output cells")
        coverage = MonitorCoverage(
            written_cells=sum(ref is not None for ref in selected_refs),
            total_cells=total,
            missed_events=source.coverage.missed_events,
            current_gap=source.coverage.current_gap,
        )
        join_digest = canonical_digest(
            {
                "owner": "zlc_neutral_atom.camera-monitor-output",
                "source": dataset_revision_ref_to_tree(source.snapshot.ref),
                "output_name": output_name,
                "events": tuple(
                    None if ref is None else event_ref_to_tree(ref)
                    for ref in selected_refs
                ),
            }
        )
        projected[output_name] = LiveDatasetOutput(
            declaration,
            snapshot,
            coverage,
            join_digest,
        )
    return projected


@dataclass(frozen=True)
class CameraMeasurementRequest:
    """Read raw camera cycles: ``repeat=0`` live, ``repeat=K`` finite.

    The request owns only the selected camera.  Trigger timing belongs to
    independently running hardware; Camera Measurement never acquires pulse or
    sequencer authority.
    """

    camera_ref: DeviceRef
    repeat: int = DEFAULT_CAMERA_MEASUREMENT_REPEAT
    history_cycles: int = DEFAULT_CAMERA_MONITOR_HISTORY_CYCLES
    frames_per_cycle: int = DEFAULT_CAMERA_FRAMES_PER_CYCLE
    exposure_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.camera_ref, DeviceRef):
            raise TypeError("camera_ref must be DeviceRef")
        if isinstance(self.repeat, bool) or not isinstance(self.repeat, int):
            raise TypeError("repeat must be an integer")
        if self.repeat < MINIMUM_CAMERA_MEASUREMENT_REPEAT:
            raise ValueError("repeat must be non-negative")
        object.__setattr__(
            self,
            "history_cycles",
            positive_integer(self.history_cycles, "history_cycles"),
        )
        object.__setattr__(
            self,
            "frames_per_cycle",
            positive_integer(self.frames_per_cycle, "frames_per_cycle"),
        )
        exposure = self.exposure_seconds
        if exposure is not None:
            object.__setattr__(
                self,
                "exposure_seconds",
                positive_real(exposure, "exposure_seconds"),
            )

    @property
    def output_names(self) -> tuple[str, ...]:
        """One signal per declared readout event, in cycle order."""

        return tuple(output.name for output in self.output_declarations)

    @property
    def output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return camera_frame_output_declarations(self.frames_per_cycle)


@dataclass(frozen=True)
class CameraMeasurementDescriptor:
    name: str
    camera_role: str
    output_schema: DatasetSchema
    resource_claim: str

    def __post_init__(self) -> None:
        canonical_text(self.name, "camera measurement name")
        canonical_text(self.camera_role, "camera role")
        if not isinstance(self.output_schema, DatasetSchema):
            raise TypeError("output_schema must be DatasetSchema")
        canonical_text(self.resource_claim, "resource_claim")


def _camera_request_outputs(
    request: object,
) -> tuple[OutputPresentation, ...]:
    if not isinstance(request, (CameraMeasurementIntent, CameraMeasurementRequest)):
        raise TypeError("Camera output owner requires an authored or bound request")
    return tuple(
        OutputPresentation(
            declaration,
            declaration.name,
            "Counts",
            "ordered camera readout event; repeat, point, and trailing data axes "
            "are preserved",
        )
        for declaration in request.output_declarations
    )


def _camera_role_choices(context: object) -> tuple[DynamicChoicePresentation, ...]:
    if not isinstance(context, tuple):
        raise TypeError("Camera dynamic choice context must be a role tuple")
    roles = camera_measurement_roles(context)
    return (
        DynamicChoicePresentation(
            "camera_role",
            tuple(AuthoringChoice(role, role) for role in roles),
            camera_measurement_default_role(roles),
            "Camera Measurement requires an installed camera role" if not roles else "",
        ),
    )


CAMERA_MEASUREMENT_LOGIC_NODE = LogicNodeDeclaration(
    definition=CAMERA_MEASUREMENT_DEFINITION,
    description="Acquire camera frames as a live or finite Measurement",
    authoring_schema=_CAMERA_MEASUREMENT_AUTHORING_SCHEMA,
    input_specs=(),
    outputs=(),
    build_request=build_camera_measurement_intent_from_authoring,
    bind_request=bind_no_node_inputs,
    resolve_outputs=_camera_request_outputs,
    resolve_dynamic_choices=_camera_role_choices,
)


__all__ = [
    "CAMERA_MEASUREMENT_ROLE_ORDER",
    "CAMERA_MEASUREMENT_DEFINITION",
    "CAMERA_MEASUREMENT_KEY",
    "CAMERA_MEASUREMENT_LOGIC_NODE",
    "CameraMeasurementDescriptor",
    "CameraMeasurementIntent",
    "CameraMeasurementRequest",
    "CAMERA_FRAME_OUTPUT_CONTRACT_ID",
    "DEFAULT_CAMERA_FRAMES_PER_CYCLE",
    "DEFAULT_CAMERA_MEASUREMENT_REPEAT",
    "DEFAULT_CAMERA_MEASUREMENT_ROLE",
    "DEFAULT_CAMERA_MONITOR_HISTORY_CYCLES",
    "MINIMUM_CAMERA_FRAMES_PER_CYCLE",
    "MINIMUM_CAMERA_MEASUREMENT_REPEAT",
    "camera_measurement_authoring_schema",
    "build_camera_measurement_intent_from_authoring",
    "camera_measurement_default_role",
    "camera_measurement_roles",
    "camera_frame_output_index",
    "camera_frame_output_declarations",
    "project_camera_measurement_outputs",
    "camera_measurement_final_outputs",
    "project_camera_monitor_outputs",
]
