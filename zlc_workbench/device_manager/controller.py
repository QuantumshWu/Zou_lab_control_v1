"""Narrow DeviceManager commands and event-driven Qt coordination."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PyQt5 import QtCore

from zlc_neutral_atom.installation import DeviceCatalogView
from zlc_neutral_atom.installation_config import (
    InstallationConfigDocument,
    load_installation_config,
    save_installation_config,
)

from .editor_session import DeviceConfigEditorSession


@dataclass(frozen=True, slots=True)
class DeviceAdminState:
    """Capability-free observation of the one application-owned installation."""

    active_config: InstallationConfigDocument | None
    catalog: DeviceCatalogView | None
    runtime_instance_id: str | None
    can_initialize: bool
    closed: bool = False

    def __post_init__(self) -> None:
        if self.active_config is None:
            if self.catalog is not None or self.runtime_instance_id is not None:
                raise ValueError("an inactive admin state cannot expose a runtime")
        else:
            if not isinstance(self.catalog, DeviceCatalogView):
                raise TypeError("active admin state requires DeviceCatalogView")
            if self.runtime_instance_id != self.catalog.runtime_instance_id:
                raise ValueError("runtime id differs from the catalog generation")


@dataclass(frozen=True, slots=True)
class ConfigChange:
    candidate_digest: str
    active_digest: str | None
    initialization_required: bool
    restart_required: bool


@dataclass(frozen=True, slots=True)
class ShutdownReport:
    runtime_instance_id: str
    closed: bool
    diagnostics: tuple[str, ...] = ()


class DeviceAdminPort(Protocol):
    """The exact application-root operations earned by DeviceManager."""

    def state(self) -> DeviceAdminState: ...

    def assess(self, candidate: InstallationConfigDocument) -> ConfigChange: ...

    def initialize_once(
        self, candidate: InstallationConfigDocument
    ) -> DeviceAdminState: ...

    def shutdown_for_restart(
        self, expected_runtime_instance_id: str
    ) -> ShutdownReport: ...

    def dispose(self) -> None: ...


class DeviceManagerController(QtCore.QObject):
    """Local config draft plus two hardware-bearing commands.

    Field edits are keyed deltas.  A whole immutable config is constructed only
    at Load/Save/Init, and runtime observations are emitted only after an actual
    initialize/shutdown transition.  There is no timer or whole-window snapshot.
    """

    draft_changed = QtCore.pyqtSignal(str)
    document_replaced = QtCore.pyqtSignal()
    runtime_changed = QtCore.pyqtSignal(object)
    busy_changed = QtCore.pyqtSignal(bool, str)
    status_changed = QtCore.pyqtSignal(str, str)
    _completed = QtCore.pyqtSignal(object)

    def __init__(
        self,
        editor: DeviceConfigEditorSession,
        admin: DeviceAdminPort,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.editor = editor
        self._admin = admin
        self._state = admin.state()
        self._busy = False
        self._disposed = False
        self._field_errors: dict[str, str] = {}
        self._completed.connect(self._finish_operation)

    @property
    def state(self) -> DeviceAdminState:
        return self._state

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def field_errors(self) -> dict[str, str]:
        return dict(self._field_errors)

    def set_field(self, key: str, value: object) -> None:
        self._field_errors.pop(str(key), None)
        self.editor.set_field(str(key), value)
        self.draft_changed.emit(str(key))

    def set_field_error(self, key: str, error: BaseException | str) -> None:
        self._field_errors[str(key)] = str(error)
        self.draft_changed.emit(str(key))

    def switch_backend(self, backend: str) -> None:
        self._field_errors.clear()
        self.editor.switch_backend(backend)
        self.document_replaced.emit()

    def replace_new(self, backend: str) -> None:
        """Start one new draft for a domain-declared backend."""

        self._field_errors.clear()
        self.editor.replace_new(backend)
        self.document_replaced.emit()
        self.status_changed.emit("new installation draft", "info")

    def load_file(self, path: str | Path) -> None:
        document = load_installation_config(path)
        resolved = Path(path).expanduser().resolve()
        self._field_errors.clear()
        self.editor.replace_loaded(
            document,
            path=resolved,
            digest=document.content_digest,
        )
        self.document_replaced.emit()
        self.status_changed.emit(f"loaded {resolved.name}", "info")

    def save_file(self, path: str | Path | None = None) -> Path:
        self._require_valid_fields()
        target = self.editor.path if path is None else Path(path).expanduser().resolve()
        if target is None:
            raise ValueError("choose a config file with Save as")
        candidate = self.editor.candidate()
        expected = (
            self.editor.baseline_digest
            if self.editor.path is not None and target == self.editor.path
            else None
        )
        digest = save_installation_config(
            target,
            candidate,
            expected_digest=expected,
        )
        self.editor.mark_saved(target, digest)
        self.draft_changed.emit("")
        self.status_changed.emit(f"saved {target.name}", "info")
        return target

    def cancel(self) -> None:
        self._field_errors.clear()
        self.editor.cancel()
        self.document_replaced.emit()
        self.status_changed.emit("discarded unsaved config edits", "info")

    def initialize(self) -> None:
        self._require_idle()
        self._require_valid_fields()
        candidate = self.editor.candidate()
        change = self._admin.assess(candidate)
        if not change.initialization_required:
            raise RuntimeError("an installation is already active")
        self._start_operation(
            "initializing devices",
            lambda: ("initialize", self._admin.initialize_once(candidate)),
        )

    def shutdown_for_restart(self) -> None:
        self._require_idle()
        runtime_id = self._state.runtime_instance_id
        if runtime_id is None:
            raise RuntimeError("no installation is active")
        self._start_operation(
            "shutting down for restart",
            lambda: (
                "shutdown",
                self._admin.shutdown_for_restart(runtime_id),
            ),
        )

    def close(self) -> None:
        if self._disposed:
            return
        if self._busy:
            raise RuntimeError(
                "cannot close Device manager while a device operation is active"
            )
        self._disposed = True
        self._admin.dispose()

    def _require_valid_fields(self) -> None:
        if self._field_errors:
            details = "; ".join(
                f"{key}: {value}"
                for key, value in sorted(self._field_errors.items())
            )
            raise ValueError(details)

    def _require_idle(self) -> None:
        if self._busy:
            raise RuntimeError("a device administration operation is already running")
        if self._disposed:
            raise RuntimeError("Device manager is closed")

    def _start_operation(self, label: str, operation) -> None:
        self._require_idle()
        self._busy = True
        self.busy_changed.emit(True, label)
        self.status_changed.emit(label, "task")

        def run() -> None:
            try:
                outcome = operation()
            except BaseException as error:
                self._completed.emit((None, error))
            else:
                self._completed.emit((outcome, None))

        threading.Thread(
            target=run,
            name="zlc-device-admin",
            daemon=True,
        ).start()

    @QtCore.pyqtSlot(object)
    def _finish_operation(self, completion) -> None:
        outcome, error = completion
        self._busy = False
        self.busy_changed.emit(False, "")
        if error is not None:
            self.status_changed.emit(
                f"{type(error).__name__}: {error}",
                "error",
            )
            return
        kind, value = outcome
        if kind == "initialize":
            if not isinstance(value, DeviceAdminState):
                self.status_changed.emit(
                    "device authority returned an invalid state",
                    "error",
                )
                return
            self._state = value
            self.editor.set_active_document(value.active_config)
            self.runtime_changed.emit(value)
            self.status_changed.emit("devices initialized", "info")
            return
        if kind == "shutdown":
            if not isinstance(value, ShutdownReport):
                self.status_changed.emit(
                    "device authority returned an invalid shutdown report",
                    "error",
                )
                return
            if not value.closed:
                details = "; ".join(value.diagnostics) or "shutdown incomplete"
                self.status_changed.emit(details, "error")
                return
            self._state = DeviceAdminState(None, None, None, False, closed=True)
            self.editor.set_active_document(None)
            self.runtime_changed.emit(self._state)
            self.status_changed.emit(
                "installation closed; start a new process to load the saved config",
                "info",
            )


__all__ = [
    "ConfigChange",
    "DeviceAdminPort",
    "DeviceAdminState",
    "DeviceManagerController",
    "ShutdownReport",
]
