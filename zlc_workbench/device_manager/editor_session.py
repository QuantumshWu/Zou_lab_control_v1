"""Headless ordered-device draft for the Device Manager.

The editor deliberately keeps temporarily invalid text (for example an empty
or duplicate role) local to the draft.  Domain values are materialized and the
whole requirement graph is validated only at the exact Save/Apply boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import re

from zlc_frontend.form import FormChoice, FormSpec, project_authoring_form
from zlc_neutral_atom.device_types import (
    device_type,
    discover_device_types,
    validate_installation_graph,
)
from zlc_neutral_atom.installation_config import (
    DeviceInstanceConfig,
    InstallationConfigDocument,
    installation_template,
)


_NON_WORD = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True)
class _DraftDevice:
    instance_id: str
    role: str
    type_id: str
    parameters: dict[str, object]

    @classmethod
    def from_value(cls, value: DeviceInstanceConfig) -> "_DraftDevice":
        return cls(
            value.instance_id,
            value.role,
            value.type_id,
            dict(value.parameters),
        )

    def copy(self) -> "_DraftDevice":
        return _DraftDevice(
            self.instance_id,
            self.role,
            self.type_id,
            dict(self.parameters),
        )


class DeviceConfigEditorSession:
    """One mutable graph draft with explicit disk and runtime baselines."""

    __slots__ = (
        "_rows",
        "_baseline",
        "_active",
        "_path",
    )

    def __init__(
        self,
        document: InstallationConfigDocument,
        active_document: InstallationConfigDocument | None = None,
        path: str | os.PathLike[str] | None = None,
    ) -> None:
        _require_document(document, "document")
        validate_installation_graph(document)
        if active_document is not None:
            _require_document(active_document, "active_document")
            validate_installation_graph(active_document)
        self._rows = [_DraftDevice.from_value(value) for value in document.devices]
        self._baseline = _document_state(document)
        self._active = (
            None if active_document is None else _document_state(active_document)
        )
        self._path = _path_or_none(path)

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def instance_ids(self) -> tuple[str, ...]:
        return tuple(row.instance_id for row in self._rows)

    @property
    def dirty(self) -> bool:
        return self._draft_state() != self._baseline

    @property
    def differs_from_active(self) -> bool:
        return self._active is not None and self._draft_state() != self._active

    def row(self, instance_id: str) -> _DraftDevice:
        return self._row(instance_id).copy()

    def form_spec(self, instance_id: str) -> FormSpec:
        """Project leaf fields plus graph-derived stable-ID requirements.

        Requirement semantics remain owned by ``DeviceTypeDescriptor``.  The
        manager merely renders every requirement as the compatible instances
        already present in this draft, so users never author an untyped role or
        Python class name.  A missing/incompatible current reference remains
        visible for repair but still fails graph validation at Save/Apply.
        """

        row = self._row(instance_id)
        descriptor = device_type(row.type_id)
        base = project_authoring_form(descriptor.authoring_schema)
        required_capabilities = dict(descriptor.requirements)
        if not required_capabilities:
            return base

        projected = []
        for field in base.fields:
            capability = required_capabilities.get(field.key)
            if capability is None:
                projected.append(field)
                continue
            current = row.parameters[field.key]
            choices = [
                FormChoice(
                    f"{candidate.instance_id} · {device_type(candidate.type_id).label}",
                    candidate.instance_id,
                )
                for candidate in self._rows
                if candidate.instance_id != row.instance_id
                and capability in device_type(candidate.type_id).capabilities
            ]
            if isinstance(current, str) and current and all(
                choice.value != current for choice in choices
            ):
                choices.append(FormChoice(f"{current} · missing/incompatible", current))
            projected.append(
                replace(
                    field,
                    kind="choice",
                    default=current,
                    choices=tuple(choices),
                    description=(
                        f"Stable device instance providing {capability}"
                    ),
                )
            )
        return FormSpec(tuple(projected))

    def add(self, domain: str, type_id: str | None = None) -> _DraftDevice:
        descriptors = tuple(
            descriptor
            for descriptor in discover_device_types()
            if descriptor.domain == domain
        )
        if not descriptors:
            raise ValueError(f"device domain {domain!r} has no declared types")
        descriptor = descriptors[0] if type_id is None else device_type(type_id)
        if descriptor.domain != domain:
            raise ValueError(
                f"device type {descriptor.type_id!r} does not belong to {domain!r}"
            )
        occupied_ids = set(self.instance_ids)
        occupied_roles = {row.role for row in self._rows}
        base = _slug(descriptor.label) or _slug(descriptor.type_id) or domain
        row = _DraftDevice(
            _unique(base, occupied_ids),
            _unique(domain, occupied_roles),
            descriptor.type_id,
            dict(descriptor.defaults),
        )
        self._rows.append(row)
        return row.copy()

    def add_instance(self, value: DeviceInstanceConfig) -> _DraftDevice:
        if not isinstance(value, DeviceInstanceConfig):
            raise TypeError("discovered device must be DeviceInstanceConfig")
        if value.instance_id in set(self.instance_ids):
            raise ValueError(
                f"device instance {value.instance_id!r} is already configured"
            )
        descriptor = device_type(value.type_id)
        parameters = descriptor.authoring_schema.freeze(value.parameters)
        row = _DraftDevice(
            value.instance_id,
            value.role,
            value.type_id,
            dict(parameters),
        )
        self._rows.append(row)
        return row.copy()

    def remove(self, instance_id: str) -> None:
        index = self._index(instance_id)
        del self._rows[index]

    def set_role(self, instance_id: str, role: str) -> bool:
        if not isinstance(role, str):
            raise TypeError("device role draft must be text")
        row = self._row(instance_id)
        if row.role == role:
            return False
        row.role = role
        return True

    def set_parameter(self, instance_id: str, key: str, value: object) -> bool:
        if not isinstance(key, str):
            raise TypeError("device parameter key must be text")
        row = self._row(instance_id)
        descriptor = device_type(row.type_id)
        if key not in descriptor.authoring_schema.keys:
            raise KeyError(
                f"parameter {key!r} does not belong to {row.type_id!r}"
            )
        previous = row.parameters[key]
        if _typed_equal(previous, value):
            return False
        row.parameters[key] = value
        return True

    def retype(self, instance_id: str, type_id: str) -> bool:
        row = self._row(instance_id)
        previous = device_type(row.type_id)
        replacement = device_type(type_id)
        if replacement.domain != previous.domain:
            raise ValueError("Retype cannot move a device to another domain")
        if replacement.type_id == previous.type_id:
            return False
        row.type_id = replacement.type_id
        row.parameters = dict(replacement.defaults)
        return True

    def replace_new(self, template: str) -> None:
        document = installation_template(template)
        validate_installation_graph(document)
        self._replace(document)
        self._baseline = _document_state(document)
        self._path = None

    def replace_loaded(
        self,
        document: InstallationConfigDocument,
        *,
        path: str | os.PathLike[str],
    ) -> None:
        _require_document(document, "document")
        validate_installation_graph(document)
        self._replace(document)
        self._baseline = _document_state(document)
        self._path = _path_or_none(path)

    def candidate(self) -> InstallationConfigDocument:
        devices = []
        for row in self._rows:
            descriptor = device_type(row.type_id)
            parameters = descriptor.authoring_schema.freeze(row.parameters)
            devices.append(
                DeviceInstanceConfig(
                    row.instance_id,
                    row.role,
                    row.type_id,
                    parameters,
                )
            )
        document = InstallationConfigDocument(tuple(devices))
        validate_installation_graph(document)
        return document

    def mark_saved(
        self,
        path: str | os.PathLike[str],
        document: InstallationConfigDocument,
    ) -> None:
        _require_document(document, "document")
        self._path = _path_or_none(path)
        self._baseline = _document_state(document)

    def set_active_document(
        self,
        document: InstallationConfigDocument | None,
    ) -> None:
        if document is None:
            self._active = None
            return
        _require_document(document, "document")
        self._active = _document_state(document)

    def _replace(self, document: InstallationConfigDocument) -> None:
        self._rows = [
            _DraftDevice.from_value(value) for value in document.devices
        ]

    def _row(self, instance_id: str) -> _DraftDevice:
        return self._rows[self._index(instance_id)]

    def _index(self, instance_id: str) -> int:
        if not isinstance(instance_id, str):
            raise TypeError("device instance id must be text")
        for index, row in enumerate(self._rows):
            if row.instance_id == instance_id:
                return index
        raise KeyError(f"unknown device instance {instance_id!r}")

    def _draft_state(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (row.instance_id, row.role, row.type_id, dict(row.parameters))
            for row in self._rows
        )


def _require_document(value: object, field: str) -> None:
    if not isinstance(value, InstallationConfigDocument):
        raise TypeError(f"{field} must be InstallationConfigDocument")


def _document_state(
    document: InstallationConfigDocument,
) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            value.instance_id,
            value.role,
            value.type_id,
            dict(value.parameters),
        )
        for value in document.devices
    )


def _path_or_none(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _slug(value: str) -> str:
    return _NON_WORD.sub("_", value.strip().lower()).strip("_")


def _unique(base: str, occupied: set[str]) -> str:
    if base not in occupied:
        return base
    index = 2
    while f"{base}_{index}" in occupied:
        index += 1
    return f"{base}_{index}"


def _typed_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and bool(left == right)


__all__ = ["DeviceConfigEditorSession"]
