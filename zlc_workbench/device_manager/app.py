"""Composition root for the current DeviceManager product."""

from __future__ import annotations

import threading
from pathlib import Path

from zlc_neutral_atom.installation_config import InstallationConfigDocument

from .controller import (
    ConfigChange,
    DeviceAdminState,
    DeviceManagerController,
    ShutdownReport,
)
from .editor_session import DeviceConfigEditorSession

__all__ = ["open_device_manager"]


class _DeviceManagerAuthority:
    """Hide Experiment ownership behind the four DeviceAdmin operations."""

    def __init__(
        self,
        *,
        experiment=None,
        repository: Path,
        name: str,
    ) -> None:
        self._lock = threading.RLock()
        self._experiment = experiment
        self._repository = repository
        self._name = str(name)
        self._owns_experiment = experiment is None
        self._ever_initialized = experiment is not None
        self._initializing = False
        self._closing = False
        self._disposed = False

    @property
    def experiment(self):
        with self._lock:
            return self._experiment

    def state(self) -> DeviceAdminState:
        with self._lock:
            experiment = self._experiment
            if experiment is None:
                return DeviceAdminState(
                    None,
                    None,
                    None,
                    not self._ever_initialized and not self._disposed,
                    closed=self._ever_initialized,
                )
            catalog = experiment.device_catalog
            return DeviceAdminState(
                experiment.installation_config,
                catalog,
                catalog.runtime_instance_id,
                False,
            )

    def assess(self, candidate: InstallationConfigDocument) -> ConfigChange:
        if not isinstance(candidate, InstallationConfigDocument):
            raise TypeError("candidate must be InstallationConfigDocument")
        state = self.state()
        active = state.active_config
        return ConfigChange(
            candidate.content_digest,
            None if active is None else active.content_digest,
            initialization_required=active is None and state.can_initialize,
            restart_required=(
                active is not None
                and active.content_digest != candidate.content_digest
            ),
        )

    def initialize_once(
        self,
        candidate: InstallationConfigDocument,
    ) -> DeviceAdminState:
        if not isinstance(candidate, InstallationConfigDocument):
            raise TypeError("candidate must be InstallationConfigDocument")
        with self._lock:
            if self._disposed:
                raise RuntimeError("Device manager is closed")
            if self._experiment is not None or self._ever_initialized:
                raise RuntimeError(
                    "this process has already published an installation"
                )
            if self._initializing:
                raise RuntimeError("installation initialization is already running")
            self._initializing = True
        try:
            from Zou_lab_control.notebook.facade import connect

            experiment = connect(
                candidate,
                repository=self._repository,
                name=self._name,
            )
        except BaseException:
            with self._lock:
                self._initializing = False
            raise
        with self._lock:
            self._initializing = False
            if self._disposed:
                close_after_publish = True
            else:
                self._experiment = experiment
                self._ever_initialized = True
                close_after_publish = False
        if close_after_publish:
            experiment.close()
            raise RuntimeError("Device manager closed during initialization")
        return self.state()

    def shutdown_for_restart(
        self,
        expected_runtime_instance_id: str,
    ) -> ShutdownReport:
        with self._lock:
            experiment = self._experiment
            if experiment is None:
                raise RuntimeError("no installation is active")
            actual = experiment.device_catalog.runtime_instance_id
            if str(expected_runtime_instance_id) != actual:
                raise RuntimeError("installation generation changed before shutdown")
            if self._closing:
                raise RuntimeError("installation shutdown is already running")
            self._closing = True
        try:
            experiment.close()
        except BaseException as error:
            with self._lock:
                self._closing = False
            return ShutdownReport(
                actual,
                False,
                (f"{type(error).__name__}: {error}",),
            )
        with self._lock:
            self._closing = False
            if self._experiment is experiment:
                self._experiment = None
            self._ever_initialized = True
        return ShutdownReport(actual, True)

    def dispose(self) -> None:
        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            experiment = self._experiment if self._owns_experiment else None
            if experiment is not None:
                self._experiment = None
        if experiment is not None:
            experiment.close()


def _workspace(path: str | Path | None) -> Path:
    if path is None:
        result = Path.home() / ".zlc" / "device-manager"
    else:
        result = Path(path).expanduser()
    result = result.resolve()
    result.mkdir(parents=True, exist_ok=True)
    return result


def open_device_manager(
    experiment=None,
    *,
    document: InstallationConfigDocument | None = None,
    repository: str | Path | None = None,
    name: str = "neutral_atom",
    on_initialized=None,
):
    """Open the formal config/admin window on one narrow authority."""

    # DeviceManager may be the first formal window in a process (in
    # particular, the standalone TaskConsole now enters through it).  Resolve
    # the same process-global Fluent scale used by TaskConsole/PulseGUI before
    # constructing any QWidget; doing this after the body exists leaves every
    # fixed metric in that body at an incorrect 1.0 scale on high-DPI screens.
    from zlc_frontend.qt_widgets import ensure_qt_app, set_fluent_scale

    ensure_qt_app()
    set_fluent_scale(None)

    if experiment is not None and document is not None:
        raise ValueError("a bound Experiment already supplies its installation config")
    if on_initialized is not None and not callable(on_initialized):
        raise TypeError("on_initialized must be callable or None")
    if experiment is not None:
        document = experiment.installation_config
    elif document is None:
        document = InstallationConfigDocument.virtual()
    if not isinstance(document, InstallationConfigDocument):
        raise TypeError("document must be InstallationConfigDocument")

    authority = _DeviceManagerAuthority(
        experiment=experiment,
        repository=_workspace(repository),
        name=name,
    )
    active = authority.state().active_config
    editor = DeviceConfigEditorSession(document, active_document=active)
    controller = DeviceManagerController(editor, authority)
    if on_initialized is not None:
        def publish_initialized(state: DeviceAdminState) -> None:
            if state.active_config is None:
                return
            initialized = authority.experiment
            if initialized is None:
                raise RuntimeError(
                    "initialized DeviceManager state has no owning Experiment"
                )
            on_initialized(initialized)

        controller.runtime_changed.connect(publish_initialized)

    from .window import launch_device_manager_window

    body = launch_device_manager_window(
        controller,
        hide_on_close=experiment is not None,
    )
    if experiment is not None and on_initialized is not None:
        from PyQt5 import QtCore

        QtCore.QTimer.singleShot(0, lambda: on_initialized(experiment))
    return body
