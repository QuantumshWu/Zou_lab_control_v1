"""CATALOG seam: DefinitionCatalog -> TaskConsole Add-Panel/Logic vocabulary.

Catalog seam of the composition root (``app.py``).  The domain
catalog stays the single source of capability identity
(:func:`_compose_catalog` -- plain imports, duplicate keys fail at compose);
this module only projects it into the
node vocabulary the console consumes: a kind, a title, a parameter
form, the declared output names, and a ``build_request`` closure that freezes the
form values into the owning application's typed request.

Headless by construction: no Qt, no matplotlib, no runtime data plane -- the projection
is a read-only view plus request construction, and every runtime behaviour
(start/stop/monitor) lives in the RUN/MONITOR seams, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from types import MappingProxyType
from typing import Callable, Mapping

from zlc_frontend.form import FormSpec
from zlc_neutral_atom.acquisition import CAMERA_MEASUREMENT_KEY
from zlc_neutral_atom.acquisition import CAMERA_MEASUREMENT_DEFINITIONS
from zlc_neutral_atom.camera_measurement import CameraMeasurementRequest
from zlc_neutral_atom.catalog import (
    DefinitionCatalog,
    DefinitionKey,
    MeasurementDefinition,
    StreamProcessorDefinition,
    TaskDefinition,
    definition_key_from_tree,
    definition_key_to_tree,
)
from zlc_neutral_atom.mot_field import (
    MOT_FIELD_FINAL_OUTPUT_NAMES,
    MOT_FIELD_TASK_DEFINITIONS,
    MOT_FIELD_TASK_KEY,
)
from zlc_neutral_atom.mot_field_live import MOT_FIELD_LIVE_OUTPUT_NAMES
from zlc_neutral_atom.readout.occupancy import (
    OCCUPANCY_LIVE_OUTPUT_NAMES,
    OCCUPANCY_STREAM_PROCESSOR_DEFINITIONS,
    OCCUPANCY_STREAM_PROCESSOR_KEY,
)
from zlc_neutral_atom.readout.coupled_measurements import (
    COUPLED_MEASUREMENT_DEFINITIONS,
    GREY_MOLASSES_DETUNING_KEY,
    READOUT_DURATION_FIDELITY_KEY,
    TEMPERATURE_RELEASE_RECAPTURE_KEY,
    GREY_MOLASSES_DETUNING_OUTPUT_NAMES,
    READOUT_DURATION_FIDELITY_OUTPUT_NAMES,
    TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_NAMES,
)
from zlc_neutral_atom.readout.calibration_projection import (
    CALIBRATION_FINAL_OUTPUT_NAMES,
)
from zlc_neutral_atom.readout.calibration_task import (
    CALIBRATION_LIVE_OUTPUT_NAMES,
)
from zlc_neutral_atom.readout.sitemap import (
    SITEMAP_CALIBRATION_TASK_DEFINITIONS,
    SITEMAP_CALIBRATION_TASK_KEY,
)
from zlc_neutral_atom.scan import (
    PULSE_SCAN_FINAL_OUTPUT_NAMES,
    PULSE_SCAN_MEASUREMENT_KEY,
    SCAN_MEASUREMENT_DEFINITIONS,
)
from zlc_storage import canonical_text

from .calibration_task import (
    build_calibration_task_intent,
    calibration_task_params,
)
from .mot_field_task import build_mot_field_intent, mot_field_params
from .camera_measurement_form import (
    build_camera_measurement_request,
    camera_measurement_form,
    camera_measurement_roles,
)
from .coupled_measurement_forms import (
    build_grey_molasses_detuning_binding,
    build_readout_duration_fidelity_binding,
    build_temperature_release_recapture_binding,
    grey_molasses_detuning_params,
    readout_duration_fidelity_params,
    temperature_release_recapture_params,
)
from .occupancy_form import build_occupancy_binding, occupancy_form
from .pulse_scan_form import (
    PulseScanFormSpec,
    build_pulse_scan_binding,
    pulse_scan_form,
)


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

_MOT_FIELD_GRID_PANEL_PARAMS = MappingProxyType(
    {
        "default_grid_intent": "IMAGE",
        "default_grid_facet_axis": "scan.parameter.da_z",
    }
)

__all__ = [
    "ConsoleCatalogView",
    "ConsoleDefaultPanel",
    "ConsoleNodeSpec",
    "ConsoleSignalDecl",
]


@dataclass(frozen=True)
class ConsoleSignalDecl:
    """Presentation metadata for one domain-declared output name.

    The declaration lets the Workbench label and offer a signal in pickers.  It
    deliberately carries no RUN/FINAL routing fact: the concrete
    ``LiveDatasetOutput`` or ``FinalDatasetOutput`` published by the application
    is the authority for that boundary.
    """

    name: str
    short: str
    axis_label: str
    description: str = ""


@dataclass(frozen=True)
class ConsoleDefaultPanel:
    """One catalog-selected view admitted only when its typed value exists."""

    output_name: str
    kind: str
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        output_name = canonical_text(self.output_name, "default panel output name")
        kind = canonical_text(self.kind, "default panel kind")
        if not isinstance(self.params, Mapping):
            raise TypeError("default panel params must be a mapping")
        object.__setattr__(self, "output_name", output_name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True)
class ConsoleNodeSpec:
    """One Add-Panel/Logic entry: catalog identity + form + request freezing."""

    key: DefinitionKey
    kind: str                     # "camera" | "measurement" | "processor" | "task"
    title: str
    description: str
    form: FormSpec | PulseScanFormSpec
    declared_outputs: tuple[ConsoleSignalDecl, ...]
    build_request: Callable[[Mapping[str, object]], object]
    default_panels: tuple[ConsoleDefaultPanel, ...] = ()
    request_output_axis_label: str | None = None
    request_output_description: str = ""

    @property
    def name(self) -> str:
        """Human-facing catalog label; never a persisted capability identity."""

        return self.title

    @property
    def definition_tree(self) -> dict[str, object]:
        return definition_key_to_tree(self.key)

    def outputs_for(
        self,
        request: object,
    ) -> tuple[ConsoleSignalDecl, ...]:
        """Project one frozen request to its exact, ordered public outputs.

        Static definitions retain their catalog declaration.  The sole dynamic
        form repeats one presentation template over the exact ``output_names``
        frozen by its domain request.  There is deliberately no arbitrary
        projector callback on this composition record: physical output
        materialization remains an application-owner operation.
        """

        if self.request_output_axis_label is None:
            outputs = self.declared_outputs
        else:
            if self.declared_outputs:
                raise ValueError(
                    "request-declared and static console outputs are mutually exclusive"
                )
            if self.key != CAMERA_MEASUREMENT_KEY or not isinstance(
                request,
                CameraMeasurementRequest,
            ):
                raise TypeError(
                    "dynamic console outputs are the closed Camera Measurement product"
                )
            names = request.output_names
            outputs = tuple(
                ConsoleSignalDecl(
                    name,
                    name,
                    self.request_output_axis_label,
                    self.request_output_description,
                )
                for name in names
            )
        outputs = tuple(outputs)
        if any(not isinstance(output, ConsoleSignalDecl) for output in outputs):
            raise TypeError("console outputs must contain ConsoleSignalDecl values")
        names = tuple(output.name for output in outputs)
        for name in names:
            canonical_text(name, "console output name")
        if len(set(names)) != len(names):
            raise ValueError("console output names must be unique")
        return outputs

_GROUP_TO_KIND = {"Task": "task", "Measurement": "measurement", "Processor": "processor"}


def _compose_catalog() -> DefinitionCatalog:
    """Compose the exact capabilities this TaskConsole currently implements."""

    return DefinitionCatalog.compose(
        SCAN_MEASUREMENT_DEFINITIONS,
        SITEMAP_CALIBRATION_TASK_DEFINITIONS,
        MOT_FIELD_TASK_DEFINITIONS,
        CAMERA_MEASUREMENT_DEFINITIONS,
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


def _short_title(key: DefinitionKey) -> str:
    """A menu-length label derived from the definition's STABLE id.

    A ``Definition.title`` is free prose -- some read as labels ("Pulse scan"),
    one is a full sentence describing what the processor does -- and a menu row
    cannot hold a sentence.  Deriving from ``stable_definition_id`` gives every
    entry the same shape.  Persistence uses the owner-encoded DefinitionKey;
    neither this label nor the prose title participates in identity.  The prose
    title remains the entry's ``description`` (its tooltip).
    """

    words = str(key.stable_definition_id).replace("_", "-").split("-")
    return " ".join(words).capitalize()


class ConsoleCatalogView:
    """Read-only projection of the composed catalog and installation facts.

    No registration, no package scanning: a new capability enters through the
    explicit tuple in :func:`_compose_catalog`, and an unknown definition type
    is refused here rather than silently dropped.
    """

    def __init__(
        self,
        *,
        installed_camera_roles: tuple[str, ...],
        sitemap_camera_roles: tuple[str, ...],
        installed_rf_roles: tuple[str, ...],
        camera_request_builder,
    ) -> None:
        self._installed_camera_roles = tuple(installed_camera_roles)
        self._sitemap_camera_roles = tuple(sitemap_camera_roles)
        self._installed_rf_roles = tuple(installed_rf_roles)
        if not callable(camera_request_builder):
            raise TypeError("camera_request_builder must be callable")
        self._camera_request_builder = camera_request_builder
        self._catalog = _compose_catalog()
        items = _catalog_items(self._catalog)
        specs: list[ConsoleNodeSpec] = []
        for item in items:
            specs.append(self._project(item))
        by_key = {}
        for spec in specs:
            if spec.key in by_key:
                raise ValueError(f"duplicate console DefinitionKey {spec.key}")
            by_key[spec.key] = spec
        self._specs = tuple(specs)
        self._by_key = by_key

    # ------------------------------------------------------------ projection
    def _project(self, item) -> ConsoleNodeSpec:
        if item.key == CAMERA_MEASUREMENT_KEY:
            return ConsoleNodeSpec(
                key=item.key,
                kind="camera",
                title=item.title,
                description=item.title,
                form=camera_measurement_form(self.camera_roles()),
                # Camera output names are request-owned, so this definition has
                # no static fallback vocabulary.
                declared_outputs=(),
                build_request=partial(
                    build_camera_measurement_request,
                    self._camera_request_builder,
                ),
                request_output_axis_label="Counts",
                request_output_description=(
                    "ordered camera readout event; repeat, point, and trailing "
                    "data axes are preserved"
                ),
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
                form=temperature_release_recapture_params(),
                declared_outputs=(
                    ConsoleSignalDecl(
                        TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_NAMES[0],
                        "survival",
                        "Survival",
                        "release-recapture survival",
                    ),
                ),
                build_request=build_temperature_release_recapture_binding,
            )
        if item.key == READOUT_DURATION_FIDELITY_KEY:
            return ConsoleNodeSpec(
                key=item.key,
                kind="measurement",
                title=item.title,
                description=(
                    "Coupled API-slot measurement: each point applies and reads "
                    "back the camera integration time, then hardware-timed "
                    "shots publish one calibrated Otsu fidelity value"
                ),
                form=readout_duration_fidelity_params(),
                declared_outputs=(
                    ConsoleSignalDecl(
                        READOUT_DURATION_FIDELITY_OUTPUT_NAMES[0],
                        "fidelity",
                        "Fidelity",
                        "readout fidelity",
                    ),
                ),
                build_request=build_readout_duration_fidelity_binding,
            )
        if item.key == GREY_MOLASSES_DETUNING_KEY:
            return ConsoleNodeSpec(
                key=item.key,
                kind="measurement",
                title=item.title,
                description=(
                    "Autonomous release-recapture scan whose two-photon "
                    "detuning table advances from the same hardware scan "
                    "clock; Start names the missing capability when no "
                    "synchronized RF Port is installed"
                ),
                form=grey_molasses_detuning_params(self.rf_roles()),
                declared_outputs=(
                    ConsoleSignalDecl(
                        GREY_MOLASSES_DETUNING_OUTPUT_NAMES[0],
                        "recapture",
                        "Recapture rate",
                        "grey-molasses recapture rate",
                    ),
                ),
                build_request=build_grey_molasses_detuning_binding,
            )
        kind = _GROUP_TO_KIND.get(item.group)
        if kind is None:
            raise TypeError(f"console cannot place catalog group {item.group!r}")
        if item.key == SITEMAP_CALIBRATION_TASK_KEY:
            return ConsoleNodeSpec(
                key=item.key,
                kind="task",
                title=_short_title(item.key),
                description=item.title,
                form=calibration_task_params(self.sitemap_camera_roles()),
                declared_outputs=(
                    ConsoleSignalDecl(
                        CALIBRATION_LIVE_OUTPUT_NAMES[0],
                        "reference frame",
                        "Counts",
                        "exact capture frame while calibration is running",
                    ),
                    ConsoleSignalDecl(
                        CALIBRATION_FINAL_OUTPUT_NAMES[0],
                        "calibration",
                        "Calibration",
                        "FINAL calibration artifact",
                    ),
                    ConsoleSignalDecl(
                        CALIBRATION_FINAL_OUTPUT_NAMES[1],
                        "site fidelity",
                        "Readout fidelity",
                        (
                            "held-out balanced fidelity for each canonical site "
                            "from the FINAL calibration's default model"
                        ),
                    ),
                    ConsoleSignalDecl(
                        CALIBRATION_FINAL_OUTPUT_NAMES[2],
                        "site threshold",
                        "Readout threshold",
                        (
                            "trained per-site threshold from the FINAL calibration "
                            "report's default model"
                        ),
                    ),
                    ConsoleSignalDecl(
                        CALIBRATION_FINAL_OUTPUT_NAMES[3],
                        "site centres",
                        "Site centre",
                        "calibrated x/y centre for each canonical site",
                    ),
                    ConsoleSignalDecl(
                        CALIBRATION_FINAL_OUTPUT_NAMES[4],
                        "aggregate fidelity",
                        "Aggregate fidelity",
                        "held-out balanced fidelity using per-site thresholds",
                    ),
                    ConsoleSignalDecl(
                        CALIBRATION_FINAL_OUTPUT_NAMES[5],
                        "global fidelity",
                        "Global fidelity",
                        "held-out balanced fidelity using one shared threshold",
                    ),
                ),
                build_request=build_calibration_task_intent,
                default_panels=(
                    ConsoleDefaultPanel(CALIBRATION_LIVE_OUTPUT_NAMES[0], "2d"),
                    ConsoleDefaultPanel(CALIBRATION_FINAL_OUTPUT_NAMES[0], "sites"),
                ),
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
                form=mot_field_params(self.camera_roles()),
                declared_outputs=(
                    ConsoleSignalDecl(
                        MOT_FIELD_LIVE_OUTPUT_NAMES[0],
                        "MOT intensity grid",
                        "Counts",
                        "provisional Bx/By/Bz intensity while the scan runs",
                    ),
                    ConsoleSignalDecl(
                        MOT_FIELD_FINAL_OUTPUT_NAMES[0],
                        "MOT field",
                        "Counts",
                        "FINAL optimum + 3-D intensity",
                    ),
                    ConsoleSignalDecl(
                        MOT_FIELD_FINAL_OUTPUT_NAMES[1],
                        "scan",
                        "Signal",
                        "exact source scan artifact",
                    ),
                ),
                build_request=build_mot_field_intent,
                default_panels=(
                    ConsoleDefaultPanel(
                        MOT_FIELD_LIVE_OUTPUT_NAMES[0],
                        "grid",
                        _MOT_FIELD_GRID_PANEL_PARAMS,
                    ),
                    ConsoleDefaultPanel(
                        MOT_FIELD_FINAL_OUTPUT_NAMES[0],
                        "grid",
                        _MOT_FIELD_GRID_PANEL_PARAMS,
                    ),
                ),
            )
        if item.key == PULSE_SCAN_MEASUREMENT_KEY:
            return ConsoleNodeSpec(
                key=item.key, kind=kind, title=_short_title(item.key),
                description=item.title,
                form=pulse_scan_form(),
                declared_outputs=(
                    ConsoleSignalDecl(
                        PULSE_SCAN_FINAL_OUTPUT_NAMES[0],
                        "scan",
                        "Signal",
                        "scan result",
                    ),
                ),
                build_request=build_pulse_scan_binding,
            )
        if item.key == OCCUPANCY_STREAM_PROCESSOR_KEY:
            return ConsoleNodeSpec(
                key=item.key,
                kind=kind,
                title="Judge occupancy",
                description=item.title,
                form=occupancy_form(),
                declared_outputs=(
                    ConsoleSignalDecl(
                        OCCUPANCY_LIVE_OUTPUT_NAMES[0],
                        "counts",
                        "Counts",
                        "site counts",
                    ),
                    ConsoleSignalDecl(
                        OCCUPANCY_LIVE_OUTPUT_NAMES[1],
                        "occupied",
                        "Occupancy",
                        "site occupancy",
                    ),
                    ConsoleSignalDecl(
                        OCCUPANCY_LIVE_OUTPUT_NAMES[2],
                        "rate",
                        "Loading rate",
                        "valid-site occupancy fraction for each repeat/point cell",
                    ),
                ),
                build_request=build_occupancy_binding,
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

    def spec_for_definition(
        self,
        tree: Mapping[str, object],
    ) -> ConsoleNodeSpec | None:
        return self._by_key.get(definition_key_from_tree(tree))

    def spec_for_key(self, key: DefinitionKey) -> ConsoleNodeSpec | None:
        if not isinstance(key, DefinitionKey):
            return None
        return self._by_key.get(key)

    def camera_roles(self) -> tuple[str, ...]:
        return camera_measurement_roles(self._installed_camera_roles)

    def sitemap_camera_roles(self) -> tuple[str, ...]:
        return self._sitemap_camera_roles

    def rf_roles(self) -> tuple[str, ...]:
        return self._installed_rf_roles
