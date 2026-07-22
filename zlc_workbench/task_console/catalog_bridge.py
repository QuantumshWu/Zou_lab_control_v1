"""CATALOG seam: DefinitionCatalog -> the console skeleton's Add-Panel/Logic vocabulary.

Seam 1 of the composition root's rewiring contract (``app.py``).  The domain
catalog stays the single source of capability identity
(:func:`zlc_workbench.task_console.intent.compose_task_console_catalog` -- plain
imports, duplicate keys fail at compose); this module only PROJECTS it into the
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

from zlc_frontend.form import FormFieldProps, FormSpec
from zlc_neutral_atom.acquisition import CAMERA_MEASUREMENT_KEY
from zlc_neutral_atom.catalog import DefinitionKey

from .intent import (
    SCAN_INTENT_DEFAULT_CAMERA_ROLE,
    _role_form_field,
    compose_task_console_catalog,
    task_console_catalog_items,
    task_console_scan_binding_form_spec,
    task_console_scan_runtime_form_spec,
)

__all__ = ["ConsoleCatalogView", "ConsoleNodeSpec", "ConsoleSignalDecl"]


@dataclass(frozen=True)
class ConsoleSignalDecl:
    """One declared output of a node, in the skeleton's signal vocabulary."""

    name: str
    short: str
    axis_label: str
    unit: str


@dataclass(frozen=True)
class ConsoleNodeSpec:
    """One Add-Panel/Logic entry: catalog identity + form + request freezing."""

    key: DefinitionKey
    kind: str                     # "camera" | "measurement" | "processor" | "task"
    title: str
    description: str
    form_spec: FormSpec
    declared_outputs: tuple[ConsoleSignalDecl, ...]
    build_request: Callable[[Mapping[str, object]], object]

    @property
    def name(self) -> str:
        """The skeleton addresses specs by title (its historical lookup key)."""

        return self.title

    @property
    def params(self) -> tuple:
        """The form as the skeleton's own param vocabulary (ParamDecl).

        The definition declares its parameters once, as a FormSpec; the console's
        editor renders ParamDecls.  Projecting here keeps that ONE declaration
        authoritative -- an editor-side copy of the field list would be a second
        place for a parameter to exist, and the two would drift the first time a
        definition gained a knob.
        """

        from zlc_data.param_decl import ParamDecl

        decls = []
        for field in self.form_spec.fields:
            kind = str(getattr(field.kind, "value", field.kind))
            decls.append(ParamDecl(
                key=field.key,
                label=field.label,
                kind="choice" if field.choices else kind,
                default=field.default,
                unit=field.unit or "",
                required=bool(field.required),
                choices=tuple(str(c.value) for c in field.choices),
                tooltip=field.description or "",
            ))
        return tuple(decls)


_GROUP_TO_KIND = {"Task": "task", "Measurement": "measurement", "Processor": "processor"}


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


def _camera_monitor_form(camera_roles: tuple[str, ...]) -> FormSpec:
    choices = camera_roles or (SCAN_INTENT_DEFAULT_CAMERA_ROLE,)
    return FormSpec(
        (
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
            FormFieldProps(
                "io_timeout_seconds", "float", "IO timeout s", default=2.0,
                required=True, minimum=1e-3,
                description="Per-read camera IO deadline",
            ),
        )
    )


def _pulse_scan_form(camera_roles: tuple[str, ...],
                     sequencer_roles: tuple[str, ...]) -> FormSpec:
    binding = task_console_scan_binding_form_spec(camera_roles, sequencer_roles)
    runtime = task_console_scan_runtime_form_spec()
    pulse = FormFieldProps(
        "pulse", "text", "Pulse document", default=None, required=True,
        description="PulseDocument path (pulses/<name>.json) driving the scan",
    )
    return FormSpec((pulse, *binding.fields, *runtime.fields))


def _occupancy_form() -> FormSpec:
    return FormSpec(
        (
            FormFieldProps(
                "source", "text", "Capture artifact", default=None, required=True,
                description="FINAL CaptureArtifactRef id to classify per frame",
            ),
            FormFieldProps(
                "calibration", "text", "Calibration artifact", default=None,
                required=True,
                description="CalibrationArtifactRef id supplying the site model",
            ),
        )
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

    No registration, no package scanning: a new capability enters by being added
    to the compose tuple in :mod:`.intent`, and an unknown definition type is
    refused there rather than silently dropped.
    """

    def __init__(self, experiment) -> None:
        self._experiment = experiment
        self._catalog = compose_task_console_catalog()
        items = task_console_catalog_items(self._catalog)
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
        if item.key == CAMERA_MEASUREMENT_KEY:
            roles = self.camera_roles()

            def build_camera(values: Mapping[str, object]):
                return experiment.readout.camera_monitor_request(
                    **_form_values(values, "camera_role", "history_capacity",
                                   "scalar_history_capacity", "io_timeout_seconds"))

            return ConsoleNodeSpec(
                key=item.key, kind="camera", title=_short_title(item.key),
                description=item.title,
                form_spec=_camera_monitor_form(roles),
                declared_outputs=(
                    ConsoleSignalDecl("frame", "frame", "camera frame", "counts"),
                    ConsoleSignalDecl("roi_value", "roi_value", "ROI reduction", "counts"),
                ),
                build_request=build_camera,
            )
        kind = _GROUP_TO_KIND.get(item.group)
        if kind is None:
            raise TypeError(f"console cannot place catalog group {item.group!r}")
        if kind == "task":

            def build_scan(values: Mapping[str, object]):
                pulse = values.get("pulse")
                if not pulse:
                    raise ValueError("pulse scan needs a PulseDocument path")
                return experiment.readout.scan_request(
                    pulse,
                    **_form_values(values, "camera_role", "sequencer_role",
                                   "trigger_channel", "timeout_seconds"))

            return ConsoleNodeSpec(
                key=item.key, kind=kind, title=_short_title(item.key),
                description=item.title,
                form_spec=_pulse_scan_form(self.camera_roles(), self.sequencer_roles()),
                declared_outputs=(
                    ConsoleSignalDecl("scan", "scan", "scan result", ""),
                ),
                build_request=build_scan,
            )

        def build_detection(values: Mapping[str, object]):
            raise NotImplementedError(
                f"{item.title}: occupancy detection binds to FINAL capture + "
                "calibration artifact refs; the artifact picker lands with the "
                "RUN seam (contract 2)")

        return ConsoleNodeSpec(
            key=item.key, kind=kind, title=_short_title(item.key),
            description=item.title,
            form_spec=_occupancy_form(),
            declared_outputs=(
                ConsoleSignalDecl("occupied", "occupied", "site occupancy", ""),
                ConsoleSignalDecl("counts", "counts", "site counts", "counts"),
            ),
            build_request=build_detection,
        )

    # -------------------------------------------------------------- queries
    @property
    def experiment(self):
        """The one session this view projects.

        The skeleton reaches the domain ONLY through this view, so the window
        never imports ``Zou_lab_control.notebook`` itself; the RUN seam derives
        its prepare/start closures from the same object.
        """

        return self._experiment

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
