"""Narrow DeviceManager commands and event-driven Qt coordination."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PyQt5 import QtCore

from zlc_neutral_atom.device_types import discover_device_types
from zlc_neutral_atom.installation import DeviceCatalogView
from zlc_neutral_atom.installation_config import (
    DeviceInstanceConfig,
    InstallationConfigDocument,
    load_installation_config,
    save_installation_config,
)
from zlc_neutral_atom.logic_node import SelectionParameterPatch

from .editor_session import DeviceConfigEditorSession


@dataclass(frozen=True, slots=True)
class DeviceAdminState:
    """Capability-free observation of the application-owned installation."""

    active_config: InstallationConfigDocument | None
    catalog: DeviceCatalogView | None
    runtime_instance_id: str | None
    can_apply: bool
    closed: bool = False

    def __post_init__(self) -> None:
        if type(self.can_apply) is not bool or type(self.closed) is not bool:
            raise TypeError("can_apply and closed must be bool")
        if self.active_config is None:
            if self.catalog is not None or self.runtime_instance_id is not None:
                raise ValueError("an inactive admin state cannot expose a runtime")
        else:
            if not isinstance(self.active_config, InstallationConfigDocument):
                raise TypeError("active_config must be InstallationConfigDocument")
            if not isinstance(self.catalog, DeviceCatalogView):
                raise TypeError("active admin state requires DeviceCatalogView")
            if self.runtime_instance_id != self.catalog.runtime_instance_id:
                raise ValueError("runtime id differs from the catalog generation")
        if self.closed and self.can_apply:
            raise ValueError("a closed device authority cannot apply a graph")


class DeviceAdminPort(Protocol):
    """The complete application-root authority needed by DeviceManager."""

    def state(self) -> DeviceAdminState: ...

    def apply(
        self,
        candidate: InstallationConfigDocument,
    ) -> DeviceAdminState: ...

    def dispose(self) -> None: ...


class DeviceManagerController(QtCore.QObject):
    """Coordinate one local graph draft and one atomic application command."""

    draft_changed = QtCore.pyqtSignal(str, str)
    device_added = QtCore.pyqtSignal(str)
    device_removed = QtCore.pyqtSignal(str)
    device_retyped = QtCore.pyqtSignal(str)
    document_replaced = QtCore.pyqtSignal()
    discovery_changed = QtCore.pyqtSignal(object)
    runtime_changed = QtCore.pyqtSignal(object)
    busy_changed = QtCore.pyqtSignal(bool, str)
    operation_finished = QtCore.pyqtSignal(str, bool)
    status_changed = QtCore.pyqtSignal(str, str)
    parameter_patch_staged = QtCore.pyqtSignal(str)
    _completed = QtCore.pyqtSignal(object)

    def __init__(
        self,
        editor: DeviceConfigEditorSession,
        admin: DeviceAdminPort,
        parent=None,
    ) -> None:
        if not isinstance(editor, DeviceConfigEditorSession):
            raise TypeError("editor must be DeviceConfigEditorSession")
        super().__init__(parent)
        self.editor = editor
        self._admin = admin
        self._state = _admin_state(admin.state())
        self._busy = False
        self._disposed = False
        self._field_errors: dict[tuple[str, str], str] = {}
        self._discovered: tuple[DeviceInstanceConfig, ...] = ()
        self._completed.connect(self._finish_operation)

    @property
    def state(self) -> DeviceAdminState:
        return self._state

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def field_errors(self) -> dict[tuple[str, str], str]:
        return dict(self._field_errors)

    @property
    def discovered(self) -> tuple[DeviceInstanceConfig, ...]:
        return self._discovered

    def set_role(self, instance_id: str, role: str) -> None:
        self._field_errors.pop((instance_id, "role"), None)
        if self.editor.set_role(instance_id, role):
            self.draft_changed.emit(instance_id, "role")

    def set_parameter(self, instance_id: str, key: str, value: object) -> None:
        self._field_errors.pop((instance_id, key), None)
        if self.editor.set_parameter(instance_id, key, value):
            self.draft_changed.emit(instance_id, key)

    def stage_parameter_patch(self, patch: SelectionParameterPatch) -> bool:
        """Stage a leaf-declared patch without applying hardware changes."""

        self._require_idle()
        if not isinstance(patch, SelectionParameterPatch):
            raise TypeError("patch must be SelectionParameterPatch")
        changed = self.editor.stage_parameter_patch(
            patch.target_instance_id,
            dict(patch.values),
        )
        for key in changed:
            self._field_errors.pop((patch.target_instance_id, key), None)
            self.draft_changed.emit(patch.target_instance_id, key)
        if changed:
            self.parameter_patch_staged.emit(patch.target_instance_id)
            self.status_changed.emit(
                patch.description or "selection parameters staged",
                "info",
            )
        return bool(changed)

    def set_field_error(
        self,
        instance_id: str,
        key: str,
        error: BaseException | str,
    ) -> None:
        self._field_errors[(instance_id, key)] = str(error)
        self.draft_changed.emit(instance_id, key)

    def add(self, domain: str, type_id: str | None = None) -> str:
        self._require_idle()
        row = self.editor.add(domain, type_id)
        self.device_added.emit(row.instance_id)
        self.status_changed.emit(f"added {row.role}", "info")
        return row.instance_id

    def add_discovered(self, value: DeviceInstanceConfig) -> str:
        self._require_idle()
        row = self.editor.add_instance(value)
        self.device_added.emit(row.instance_id)
        self.status_changed.emit(f"added discovered {row.role}", "info")
        return row.instance_id

    def remove(self, instance_id: str) -> None:
        self._require_idle()
        self.editor.remove(instance_id)
        self._drop_errors(instance_id)
        self.device_removed.emit(instance_id)
        self.status_changed.emit("removed device", "info")

    def retype(self, instance_id: str, type_id: str) -> None:
        self._require_idle()
        if not self.editor.retype(instance_id, type_id):
            return
        self._drop_errors(instance_id)
        self.device_retyped.emit(instance_id)
        self.status_changed.emit("reset parameters for the selected type", "info")

    def replace_new(self, template: str) -> None:
        self._require_idle()
        self.editor.replace_new(template)
        self._field_errors.clear()
        self.document_replaced.emit()
        self.status_changed.emit(f"new {template} installation", "info")

    def load_file(self, path: str | Path) -> None:
        self._require_idle()
        resolved = Path(path).expanduser().resolve()
        document = load_installation_config(resolved)
        self.editor.replace_loaded(document, path=resolved)
        self._field_errors.clear()
        self.document_replaced.emit()
        self.status_changed.emit(f"loaded {resolved.name}", "info")

    def save_file(self, path: str | Path | None = None) -> Path:
        self._require_idle()
        self._require_valid_fields()
        target = self.editor.path if path is None else Path(path).expanduser().resolve()
        if target is None:
            raise ValueError("choose a config file with Save as")
        candidate = self.editor.candidate()
        save_installation_config(target, candidate)
        self.editor.mark_saved(target, candidate)
        self.draft_changed.emit("", "")
        self.status_changed.emit(f"saved {target.name}", "info")
        return target

    def apply(self) -> None:
        self._require_idle()
        self._require_valid_fields()
        if not self._state.can_apply:
            raise RuntimeError("the application does not currently permit Apply")
        candidate = self.editor.candidate()
        self._start_operation(
            "applying installation",
            "apply",
            lambda: _admin_state(self._admin.apply(candidate)),
        )

    def discover(self) -> None:
        self._require_idle()

        def scan() -> tuple[DeviceInstanceConfig, ...]:
            found: list[DeviceInstanceConfig] = []
            owners: dict[str, str] = {}
            for descriptor in discover_device_types():
                operation = descriptor.discover
                if operation is None:
                    continue
                values = tuple(operation())
                for value in values:
                    if not isinstance(value, DeviceInstanceConfig):
                        raise TypeError(
                            f"{descriptor.type_id} discovery returned a non-device value"
                        )
                    if value.type_id != descriptor.type_id:
                        raise ValueError(
                            f"{descriptor.type_id} discovery returned {value.type_id!r}"
                        )
                    prior = owners.setdefault(value.instance_id, descriptor.type_id)
                    if prior != descriptor.type_id or any(
                        item.instance_id == value.instance_id for item in found
                    ):
                        raise ValueError(
                            f"duplicate discovered instance id {value.instance_id!r}"
                        )
                    found.append(value)
            return tuple(found)

        self._start_operation("discovering hardware", "discover", scan)

    def close(self) -> None:
        if self._disposed:
            return
        self._require_idle()
        self._start_operation(
            "closing device authority",
            "dispose",
            self._admin.dispose,
        )

    def _drop_errors(self, instance_id: str) -> None:
        self._field_errors = {
            key: value
            for key, value in self._field_errors.items()
            if key[0] != instance_id
        }

    def _require_valid_fields(self) -> None:
        if not self._field_errors:
            return
        details = "; ".join(
            f"{instance}.{key}: {value}"
            for (instance, key), value in sorted(self._field_errors.items())
        )
        raise ValueError(details)

    def _require_idle(self) -> None:
        if self._busy:
            raise RuntimeError("a device administration operation is already running")
        if self._disposed:
            raise RuntimeError("Device manager is closed")

    def _start_operation(self, label: str, kind: str, operation) -> None:
        self._require_idle()
        self._busy = True
        self.busy_changed.emit(True, label)
        self.status_changed.emit(label, "task")

        def run() -> None:
            try:
                value = operation()
            except BaseException as error:
                self._completed.emit((kind, None, error))
            else:
                self._completed.emit((kind, value, None))

        threading.Thread(
            target=run,
            name=f"zlc-device-{kind}",
            daemon=True,
        ).start()

    @QtCore.pyqtSlot(object)
    def _finish_operation(self, completion) -> None:
        kind, value, error = completion
        self._busy = False
        self.busy_changed.emit(False, "")
        if error is not None:
            if kind == "apply":
                self._refresh_admin_state_after_failure()
            self.status_changed.emit(
                f"{type(error).__name__}: {error}",
                "error",
            )
            self.operation_finished.emit(kind, False)
            return
        if kind == "dispose":
            self._disposed = True
            self.status_changed.emit("device authority closed", "info")
            self.operation_finished.emit(kind, True)
            return
        if kind == "discover":
            self._discovered = tuple(value)
            self.discovery_changed.emit(self._discovered)
            self.status_changed.emit(
                f"discovered {len(self._discovered)} device(s)",
                "info",
            )
            self.operation_finished.emit(kind, True)
            return
        if kind != "apply":
            self.status_changed.emit("unknown device operation", "error")
            self.operation_finished.emit(kind, False)
            return
        state = _admin_state(value)
        self._state = state
        self.editor.set_active_document(state.active_config)
        self.runtime_changed.emit(state)
        self.status_changed.emit("installation applied", "info")
        self.operation_finished.emit(kind, True)

    def _refresh_admin_state_after_failure(self) -> None:
        try:
            state = _admin_state(self._admin.state())
        except BaseException:
            return
        self._state = state
        self.editor.set_active_document(state.active_config)
        self.runtime_changed.emit(state)


def _admin_state(value: object) -> DeviceAdminState:
    if not isinstance(value, DeviceAdminState):
        raise TypeError("device admin authority returned an invalid state")
    return value


__all__ = [
    "DeviceAdminPort",
    "DeviceAdminState",
    "DeviceManagerController",
]
