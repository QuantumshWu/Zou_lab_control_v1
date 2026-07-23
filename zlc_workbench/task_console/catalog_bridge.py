"""CATALOG seam: DefinitionCatalog -> the console skeleton's Add-Panel/Logic vocabulary.

Seam 1 of the composition root's rewiring contract (``app.py``).  The domain
catalog stays the single source of capability identity
(:func:`_compose_catalog` -- plain imports, duplicate keys fail at compose);
this module only PROJECTS it into the
node vocabulary the ORIGINAL console UI consumes: a kind, a title, a parameter
form, the declared output names, and a ``build_request`` closure that freezes the
form values into the owning facade's typed request.

Headless by construction: no Qt, no matplotlib, no signal hub -- the projection
is a read-only view plus request construction, and every runtime behaviour
(start/stop/monitor) lives in the RUN/MONITOR seams, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from zlc_data import ReductionMethod
from zlc_data.param_decl import ParamDecl
from zlc_frontend.form import FormChoice, FormFieldProps, FormSpec
from zlc_neutral_atom.acquisition import CAMERA_MEASUREMENT_KEY
from zlc_neutral_atom.acquisition import CAMERA_MEASUREMENT_DEFINITIONS
from zlc_neutral_atom.catalog import (
    DefinitionCatalog,
    DefinitionKey,
    MeasurementDefinition,
    StreamProcessorDefinition,
    TaskDefinition,
)
from zlc_neutral_atom.mot_field import (
    MOT_FIELD_TASK_DEFINITIONS,
    MOT_FIELD_TASK_KEY,
)
from zlc_neutral_atom.monitor_application import (
    CAMERA_MONITOR_MEASUREMENT_DEFINITIONS,
    CAMERA_MONITOR_MEASUREMENT_KEY,
)
from zlc_neutral_atom.pulse_programs import DEFAULT_PROBE_PULSE_PATH
from zlc_neutral_atom.readout.occupancy import (
    OCCUPANCY_STREAM_PROCESSOR_DEFINITIONS,
    OCCUPANCY_STREAM_PROCESSOR_KEY,
)
from zlc_neutral_atom.readout.coupled_measurements import (
    COUPLED_MEASUREMENT_DEFINITIONS,
    GREY_MOLASSES_DETUNING_KEY,
    READOUT_DURATION_FIDELITY_KEY,
    TEMPERATURE_RELEASE_RECAPTURE_KEY,
)
from zlc_neutral_atom.readout.sitemap import (
    SITEMAP_CALIBRATION_TASK_DEFINITIONS,
    SITEMAP_CALIBRATION_TASK_KEY,
)
from zlc_neutral_atom.scan import PULSE_SCAN_TASK_KEY, SCAN_TASK_DEFINITIONS
from zlc_storage import canonical_text

from .mot_field_task import build_mot_field_request, mot_field_params
from .occupancy_binding import OccupancyBindingIntent
from .coupled_measurement_presenter import (
    build_grey_molasses_detuning_intent,
    build_readout_duration_fidelity_intent,
    build_temperature_release_recapture_intent,
    grey_molasses_detuning_params,
    readout_duration_fidelity_params,
    temperature_release_recapture_params,
)

SCAN_INTENT_DEFAULT_CAMERA_ROLE = "camera"
SCAN_INTENT_DEFAULT_SEQUENCER_ROLE = "sequencer"


@dataclass(frozen=True, slots=True)
class _CatalogItem:
    key: DefinitionKey
    group: str
    title: str


_CATALOG_GROUP_BY_DEFINITION = (
    (TaskDefinition, "Task"),
    (MeasurementDefinition, "Measurement"),
    (StreamProcessorDefinition, "Processor"),
)

__all__ = ["ConsoleCatalogView", "ConsoleNodeSpec", "ConsoleSignalDecl"]


@dataclass(frozen=True)
class ConsoleSignalDecl:
    """One output's routing, ready-to-render label, and explanatory text."""

    name: str
    short: str
    axis_label: str
    unit: str
    description: str = ""


@dataclass(frozen=True)
class ConsoleNodeSpec:
    """One Add-Panel/Logic entry: catalog identity + form + request freezing."""

    key: DefinitionKey
    kind: str                     # "camera" | "measurement" | "processor" | "task"
    title: str
    description: str
    params: tuple[ParamDecl, ...]
    declared_outputs: tuple[ConsoleSignalDecl, ...]
    build_request: Callable[[Mapping[str, object]], object]

    @property
    def name(self) -> str:
        """The skeleton addresses specs by title (its historical lookup key)."""

        return self.title

_GROUP_TO_KIND = {"Task": "task", "Measurement": "measurement", "Processor": "processor"}


def _compose_catalog() -> DefinitionCatalog:
    """Compose the exact capabilities this TaskConsole currently implements."""

    return DefinitionCatalog.compose(
        SCAN_TASK_DEFINITIONS,
        SITEMAP_CALIBRATION_TASK_DEFINITIONS,
        MOT_FIELD_TASK_DEFINITIONS,
        CAMERA_MEASUREMENT_DEFINITIONS,
        CAMERA_MONITOR_MEASUREMENT_DEFINITIONS,
        COUPLED_MEASUREMENT_DEFINITIONS,
        OCCUPANCY_STREAM_PROCESSOR_DEFINITIONS,
    )


def _catalog_items(catalog: DefinitionCatalog) -> tuple[_CatalogItem, ...]:
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
                "TaskConsole cannot place definition "
                f"{definition.key} of type {type(definition).__name__}"
            )
        items.append(_CatalogItem(definition.key, group, definition.title))
    return tuple(items)


def _role_form_field(
    roles: tuple[str, ...],
    *,
    key: str,
    label: str,
    domain: str,
    preferred: str | None,
) -> FormFieldProps:
    """Present an installation-owned role without copying runtime resolution."""

    values = tuple(roles)
    if not values:
        raise ValueError(f"{domain} roles must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{domain} roles must be unique")
    for role in values:
        canonical_text(role, f"{domain} role")
    choices = tuple(FormChoice(role, role) for role in values)
    if preferred is None:
        return FormFieldProps(
            key,
            "choice",
            label,
            default=None,
            required=False,
            choices=choices,
            description=(
                f"{domain} role; leave blank to let the installation resolve it"
            ),
        )
    return FormFieldProps(
        key,
        "choice",
        label,
        default=preferred if preferred in values else values[0],
        required=True,
        choices=choices,
        description=f"Frozen {domain} role from the current installation",
    )


def _scan_binding_form(
    camera_roles: tuple[str, ...],
    sequencer_roles: tuple[str, ...],
) -> FormSpec:
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


def _short_title(key: DefinitionKey) -> str:
    """A menu-length label derived from the definition's STABLE id.

    A ``Definition.title`` is free prose -- some read as labels ("Pulse scan"),
    one is a full sentence describing what the processor does -- and a menu row
    cannot hold a sentence.  Deriving from ``stable_definition_id`` gives every
    entry the same shape, and ties the label the operator sees to the same
    string the saved board persists, so a title reword cannot orphan a layout.
    The prose title survives as the entry's ``description`` (its tooltip).
    """

    words = str(key.stable_definition_id).replace("_", "-").split("-")
    return " ".join(words).capitalize()


def _params_from_form(form: FormSpec) -> tuple[ParamDecl, ...]:
    """Project simple scalar presentation fields into the shared Qt vocabulary."""

    return tuple(
        ParamDecl(
            key=field.key,
            label=field.label,
            kind="choice" if field.choices else str(field.kind),
            default=field.default,
            unit=field.unit or "",
            required=bool(field.required),
            choices=tuple(str(choice.value) for choice in field.choices),
            tooltip=field.description or "",
        )
        for field in form.fields
    )


def _camera_monitor_params(camera_roles: tuple[str, ...]) -> tuple[ParamDecl, ...]:
    choices = camera_roles or (SCAN_INTENT_DEFAULT_CAMERA_ROLE,)
    return (
        *_params_from_form(
            FormSpec((
                # No preferred default: a monitor wants the FREE-RUNNING camera, and
                # which role that is belongs to the installation's own resolution.
                _role_form_field(
                    choices, key="camera_role", label="Camera role",
                    domain="camera", preferred=None,
                ),
                FormFieldProps(
                    "history_capacity", "int", "Frame history", default=8,
                    required=True, minimum=1,
                    description="Frames retained in the monitor history axis",
                ),
                FormFieldProps(
                    "scalar_history_capacity", "int", "Scalar history", default=300,
                    required=True, minimum=1,
                    description="ROI-scalar samples retained for the rolling trace",
                ),
            ))
        ),
        ParamDecl(
            "roi_reduction",
            "ROI reducer",
            "choice",
            default=ReductionMethod.MEAN.value,
            required=True,
            choices=(
                ReductionMethod.MEAN.value,
                ReductionMethod.SUM.value,
                ReductionMethod.MAX.value,
            ),
            tooltip="Reducer applied to the selector ROI for roi_value",
        ),
    )


def _camera_capture_params(
    camera_roles: tuple[str, ...],
    sequencer_roles: tuple[str, ...],
) -> tuple[ParamDecl, ...]:
    return (
        ParamDecl(
            "pulse",
            "Pulse",
            "path",
            default=DEFAULT_PROBE_PULSE_PATH,
            required=True,
            path_mode="file",
            base_dir="pulses",
            file_filter="Pulse program (*.json);;All files (*)",
            tooltip="Finite PulseDocument that emits the camera trigger",
        ),
        *_params_from_form(
            _scan_binding_form(camera_roles, sequencer_roles)
        ),
        ParamDecl(
            "repeat_count",
            "Repeats",
            "int",
            default=1,
            lo=1,
            hi=1_000_000,
            required=True,
            optional=False,
            tooltip="Exact capture repeats",
        ),
        ParamDecl(
            "readout_events_per_repeat",
            "Readout events / repeat",
            "int",
            default=1,
            lo=1,
            hi=1_000_000,
            required=True,
            optional=False,
            tooltip=(
                "Expected camera-triggered frames in each repeat. A one-cell "
                "capture also has a live preview; every other valid layout is "
                "preserved in the complete FINAL capture without reduction"
            ),
        ),
    )


def _pulse_scan_params(camera_roles: tuple[str, ...],
                       sequencer_roles: tuple[str, ...]) -> tuple[ParamDecl, ...]:
    binding = _scan_binding_form(camera_roles, sequencer_roles)
    return (
        ParamDecl(
            "pulse",
            "Pulse template",
            "path",
            default=DEFAULT_PROBE_PULSE_PATH,
            required=True,
            path_mode="file",
            base_dir="pulses",
            file_filter="Pulse program (*.json);;All files (*)",
            tooltip="Current PulseDocument whose declared scan/API slots are edited below",
        ),
        ParamDecl(
            "pulse_slots",
            "Slots",
            "pulse_slots",
            default={},
            required=True,
            depends_on="pulse",
            tooltip=(
                "Choose SCAN_SLOT or API_SLOT and author the complete numeric "
                "scan_table program"
            ),
        ),
        *_params_from_form(binding),
    )


def _calibration_params(camera_roles: tuple[str, ...]) -> tuple[ParamDecl, ...]:
    role = _params_from_form(FormSpec((
        _role_form_field(
            camera_roles,
            key="camera_role",
            label="Camera role",
            domain="camera",
            preferred=SCAN_INTENT_DEFAULT_CAMERA_ROLE,
        ),
    )))
    return (
        ParamDecl(
            "frames",
            "Reference groups",
            "int",
            default=20,
            lo=1,
            hi=1_000_000,
            required=True,
            optional=False,
            tooltip=(
                "Complete long/readout/long groups captured before calibration"
            ),
        ),
        *role,
    )


def _occupancy_params() -> tuple[ParamDecl, ...]:
    return (
        ParamDecl(
            "camera_frame",
            "Capture recipe",
            "signal",
            required=True,
            tooltip=(
                "Select the frame output only to identify a Camera capture "
                "Measurement row. Occupancy reruns that row's frozen capture "
                "recipe with calibration; it does not consume a previously "
                "published frame"
            ),
        ),
        ParamDecl(
            "calibration",
            "Calibration",
            "signal",
            required=True,
            tooltip=(
                "FINAL calibration output of a successful Calibrate readout "
                "Task row in this TaskConsole"
            ),
        ),
    )


def _form_values(values: Mapping[str, object], *keys: str) -> dict:
    picked = {}
    for key in keys:
        value = values.get(key)
        if value not in (None, ""):
            picked[key] = value
    return picked


class ConsoleCatalogView:
    """Read-only projection of the composed DefinitionCatalog for ONE experiment.

    No registration, no package scanning: a new capability enters through the
    explicit tuple in :func:`_compose_catalog`, and an unknown definition type
    is refused here rather than silently dropped.
    """

    def __init__(self, experiment) -> None:
        self._experiment = experiment
        self._catalog = _compose_catalog()
        items = _catalog_items(self._catalog)
        specs: list[ConsoleNodeSpec] = []
        for item in items:
            specs.append(self._project(item))
        by_name = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"duplicate console spec title {spec.name!r}")
            by_name[spec.name] = spec
        self._specs = tuple(specs)
        self._by_name = by_name

    # ------------------------------------------------------------ projection
    def _project(self, item) -> ConsoleNodeSpec:
        experiment = self._experiment
        if item.key == CAMERA_MONITOR_MEASUREMENT_KEY:
            roles = self.camera_roles()

            def build_camera_monitor(values: Mapping[str, object]):
                reduction_value = values.get(
                    "roi_reduction",
                    ReductionMethod.MEAN.value,
                )
                try:
                    reduction = (
                        reduction_value
                        if isinstance(reduction_value, ReductionMethod)
                        else ReductionMethod(
                            str(reduction_value).strip().upper()
                        )
                    )
                except ValueError as error:
                    raise ValueError(
                        "camera monitor ROI reducer must be MEAN, SUM, or MAX"
                    ) from error
                return experiment.readout.camera_monitor_request(
                    **_form_values(values, "camera_role", "history_capacity",
                                   "scalar_history_capacity"),
                    roi_reduction=reduction,
                )

            return ConsoleNodeSpec(
                key=item.key, kind="camera", title=_short_title(item.key),
                description=item.title,
                params=_camera_monitor_params(roles),
                declared_outputs=(
                    ConsoleSignalDecl(
                        "frame",
                        "frame",
                        "Counts",
                        "counts",
                        "camera frame",
                    ),
                    ConsoleSignalDecl(
                        "roi_value",
                        "ROI value",
                        "ROI value",
                        "counts",
                        "ROI scalar history",
                    ),
                ),
                build_request=build_camera_monitor,
            )
        if item.key == CAMERA_MEASUREMENT_KEY:

            def build_camera_capture(values: Mapping[str, object]):
                pulse = values.get("pulse")
                if not pulse:
                    raise ValueError("camera capture needs a PulseDocument path")
                return experiment.readout.capture_request(
                    pulse,
                    **_form_values(
                        values,
                        "camera_role",
                        "sequencer_role",
                        "trigger_channel",
                        "repeat_count",
                        "readout_events_per_repeat",
                    ),
                )

            return ConsoleNodeSpec(
                key=item.key,
                kind="camera",
                title=_short_title(item.key),
                description=item.title,
                params=_camera_capture_params(
                    self.camera_roles(),
                    self.sequencer_roles(),
                ),
                declared_outputs=(
                    ConsoleSignalDecl(
                        "frame",
                        "frame",
                        "Counts",
                        "counts",
                        "complete exact capture dataset (R, P, *data_shape)",
                    ),
                ),
                build_request=build_camera_capture,
            )
        if item.key == TEMPERATURE_RELEASE_RECAPTURE_KEY:
            return ConsoleNodeSpec(
                key=item.key,
                kind="measurement",
                title=item.title,
                description=(
                    "Autonomous hardware scan with two exact camera events per "
                    "cell; publishes calibrated survival without dropping the "
                    "repeat or scan axes"
                ),
                params=temperature_release_recapture_params(
                    self.camera_roles(),
                    self.sequencer_roles(),
                ),
                declared_outputs=(
                    ConsoleSignalDecl(
                        "survival",
                        "survival",
                        "Survival",
                        "survival",
                        "release-recapture survival",
                    ),
                ),
                build_request=build_temperature_release_recapture_intent,
            )
        if item.key == READOUT_DURATION_FIDELITY_KEY:
            return ConsoleNodeSpec(
                key=item.key,
                kind="measurement",
                title=item.title,
                description=(
                    "Visible current Measurement intent; Start rejects until "
                    "the camera Port can configure and read back exposure before "
                    "each exact API-slot point group"
                ),
                params=readout_duration_fidelity_params(
                    self.camera_roles(),
                    self.sequencer_roles(),
                ),
                declared_outputs=(
                    ConsoleSignalDecl(
                        "fidelity",
                        "fidelity",
                        "Fidelity",
                        "",
                        "readout fidelity",
                    ),
                ),
                build_request=build_readout_duration_fidelity_intent,
            )
        if item.key == GREY_MOLASSES_DETUNING_KEY:
            return ConsoleNodeSpec(
                key=item.key,
                kind="measurement",
                title=item.title,
                description=(
                    "Visible current Measurement intent; Start rejects until "
                    "the installation exposes an RF Port with a preloaded "
                    "hardware-synchronized detuning table"
                ),
                params=grey_molasses_detuning_params(
                    self.camera_roles(),
                    self.sequencer_roles(),
                    self.rf_roles(),
                ),
                declared_outputs=(
                    ConsoleSignalDecl(
                        "survival",
                        "survival",
                        "Survival",
                        "survival",
                        "grey-molasses survival",
                    ),
                ),
                build_request=build_grey_molasses_detuning_intent,
            )
        kind = _GROUP_TO_KIND.get(item.group)
        if kind is None:
            raise TypeError(f"console cannot place catalog group {item.group!r}")
        if item.key == SITEMAP_CALIBRATION_TASK_KEY:

            def build_calibration(values: Mapping[str, object]):
                frames = values.get("frames", 20)
                if not isinstance(frames, int) or isinstance(frames, bool):
                    raise TypeError("reference groups must be an integer")
                return experiment.readout.sitemap_request(
                    frames=frames,
                    **_form_values(values, "camera_role"),
                )

            return ConsoleNodeSpec(
                key=item.key,
                kind="task",
                title=_short_title(item.key),
                description=item.title,
                params=_calibration_params(self.camera_roles()),
                declared_outputs=(
                    ConsoleSignalDecl(
                        "calibration",
                        "calibration",
                        "Calibration",
                        "",
                        "FINAL calibration artifact",
                    ),
                ),
                build_request=build_calibration,
            )
        if item.key == MOT_FIELD_TASK_KEY:
            return ConsoleNodeSpec(
                key=item.key,
                kind="task",
                title=item.title,
                description=(
                    "Sweep da_x/da_y/da_z in one autonomous hardware scan, "
                    "measure MOT fluorescence, and report the refined optimum"
                ),
                params=mot_field_params(
                    self.camera_roles(),
                    self.sequencer_roles(),
                ),
                declared_outputs=(
                    ConsoleSignalDecl(
                        "mot_field",
                        "MOT field",
                        "Counts",
                        "counts",
                        "FINAL optimum + 3-D intensity",
                    ),
                    ConsoleSignalDecl(
                        "scan",
                        "scan",
                        "Signal",
                        "",
                        "exact source scan artifact",
                    ),
                ),
                build_request=lambda values: build_mot_field_request(
                    experiment,
                    values,
                ),
            )
        if item.key == PULSE_SCAN_TASK_KEY:

            def build_scan(values: Mapping[str, object]):
                from zlc_data.vocabulary import SWEEP_API_SLOT, SWEEP_SCAN_SLOT
                from zlc_neutral_atom.scan import ApiSegmentTable
                from zlc_pulse import load_pulse_document
                from zlc_workbench.pulse_editor.scan_workspace import (
                    commit_scan_candidate,
                    execute_numeric_table_program,
                    execute_scan_program,
                )

                pulse = values.get("pulse")
                if not pulse:
                    raise ValueError("pulse scan needs a PulseDocument path")
                document = load_pulse_document(pulse)
                slots = dict(values.get("pulse_slots") or {})
                sweep_kind = str(slots.get("sweep_kind") or "")
                source = str(slots.get("program") or "")
                binding = _form_values(
                    values,
                    "camera_role",
                    "sequencer_role",
                    "trigger_channel",
                )
                if sweep_kind == SWEEP_SCAN_SLOT:
                    candidate = execute_scan_program(document, source)
                    committed = commit_scan_candidate(
                        document,
                        candidate.candidate,
                        "generated",
                    )
                    return experiment.readout.scan_request(
                        committed,
                        api_values=dict(slots.get("api") or {}),
                        **binding,
                    )
                if sweep_kind == SWEEP_API_SLOT:
                    columns = tuple(
                        parameter.parameter_id
                        for parameter in document.api_parameters
                    )
                    rows = execute_numeric_table_program(
                        source,
                        width=len(columns),
                    )
                    return experiment.readout.api_scan_request(
                        document,
                        api_table=ApiSegmentTable(columns, rows),
                        segmentation_rationale=(
                            "Explicit API-slot sweep authored in TaskConsole"
                        ),
                        **binding,
                    )
                raise ValueError("choose a SCAN_SLOT or API_SLOT sweep")

            return ConsoleNodeSpec(
                key=item.key, kind=kind, title=_short_title(item.key),
                description=item.title,
                params=_pulse_scan_params(self.camera_roles(), self.sequencer_roles()),
                declared_outputs=(
                    ConsoleSignalDecl(
                        "scan",
                        "scan",
                        "Signal",
                        "",
                        "scan result",
                    ),
                ),
                build_request=build_scan,
            )
        if item.key == OCCUPANCY_STREAM_PROCESSOR_KEY:

            def build_occupancy(values: Mapping[str, object]):
                camera_frame = values.get("camera_frame")
                calibration = values.get("calibration")
                if not isinstance(camera_frame, str) or not camera_frame.strip():
                    raise ValueError(
                        "occupancy requires a Camera capture frame output"
                    )
                if not isinstance(calibration, str) or not calibration.strip():
                    raise ValueError(
                        "occupancy requires a successful Calibration FINAL output"
                    )
                return OccupancyBindingIntent(
                    camera_frame.strip(),
                    calibration.strip(),
                )

            return ConsoleNodeSpec(
                key=item.key,
                kind=kind,
                title=_short_title(item.key),
                description=item.title,
                params=_occupancy_params(),
                declared_outputs=(
                    ConsoleSignalDecl(
                        "occupied",
                        "occupied",
                        "Occupancy",
                        "",
                        "site occupancy",
                    ),
                    ConsoleSignalDecl(
                        "counts",
                        "counts",
                        "Counts",
                        "counts",
                        "site counts",
                    ),
                ),
                build_request=build_occupancy,
            )
        if kind == "task":
            raise TypeError(
                f"TaskConsole has no explicit task presenter for {item.key}"
            )
        raise TypeError(
            "TaskConsole has no explicit "
            f"{kind} presenter for {item.key}"
        )

    # -------------------------------------------------------------- queries
    def specs(self, kind: str | None = None) -> tuple[ConsoleNodeSpec, ...]:
        if kind is None:
            return self._specs
        return tuple(spec for spec in self._specs if spec.kind == kind)

    def spec_named(self, name: str) -> ConsoleNodeSpec | None:
        return self._by_name.get(str(name))

    def camera_roles(self) -> tuple[str, ...]:
        return tuple(self._experiment.device_catalog.roles("camera"))

    def sequencer_roles(self) -> tuple[str, ...]:
        return tuple(self._experiment.device_catalog.roles("sequencer"))

    def rf_roles(self) -> tuple[str, ...]:
        return tuple(self._experiment.device_catalog.roles("rf"))
