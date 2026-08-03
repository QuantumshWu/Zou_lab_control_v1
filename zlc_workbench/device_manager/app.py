"""Qt composition for DeviceManager over one application authority."""

from __future__ import annotations

from pathlib import Path

from zlc_neutral_atom.device_types import validate_installation_graph
from zlc_neutral_atom.installation_config import (
    InstallationConfigDocument,
    installation_template,
    load_installation_config,
)

from .controller import DeviceAdminPort, DeviceAdminState, DeviceManagerController
from .editor_session import DeviceConfigEditorSession

__all__ = ["open_device_manager"]


def _initial_document(
    admin: DeviceAdminPort,
    document: InstallationConfigDocument | None,
    config_path: Path | None,
) -> tuple[InstallationConfigDocument, DeviceAdminState]:
    state = admin.state()
    if not isinstance(state, DeviceAdminState):
        raise TypeError("device admin authority returned an invalid state")
    if state.active_config is not None:
        if document is not None or config_path is not None:
            raise ValueError(
                "an active device authority already supplies its installation config"
            )
        return state.active_config, state
    if config_path is not None:
        loaded = load_installation_config(config_path)
        if document is not None and document != loaded:
            raise ValueError(
                "document differs from the installation config at config_path"
            )
        document = loaded
    elif document is None:
        document = installation_template("virtual")
    if not isinstance(document, InstallationConfigDocument):
        raise TypeError("document must be InstallationConfigDocument")
    validate_installation_graph(document)
    return document, state


def open_device_manager(
    admin: DeviceAdminPort,
    *,
    document: InstallationConfigDocument | None = None,
    config_path: str | Path | None = None,
    on_runtime_changed=None,
    hide_on_close: bool | None = None,
):
    """Open the graph editor without implicitly constructing an Experiment."""

    if on_runtime_changed is not None and not callable(on_runtime_changed):
        raise TypeError("on_runtime_changed must be callable or None")
    resolved_path = (
        None if config_path is None else Path(config_path).expanduser().resolve()
    )
    initial, state = _initial_document(admin, document, resolved_path)

    from zlc_frontend.qt_widgets import ensure_qt_app, set_fluent_scale

    ensure_qt_app()
    set_fluent_scale(None)

    editor = DeviceConfigEditorSession(
        initial,
        active_document=state.active_config,
        path=resolved_path,
    )
    controller = DeviceManagerController(editor, admin)
    if on_runtime_changed is not None:
        controller.runtime_changed.connect(on_runtime_changed)

    from .window import launch_device_manager_window

    return launch_device_manager_window(
        controller,
        hide_on_close=(
            state.active_config is not None
            if hide_on_close is None
            else bool(hide_on_close)
        ),
    )
