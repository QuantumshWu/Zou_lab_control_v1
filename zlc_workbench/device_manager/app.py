"""Qt composition for DeviceManager over an explicit application authority.

This module owns only the Workbench surface.  Installation construction and
the lifetime of the object that implements :class:`DeviceAdminPort` belong to
the embedding application composition root; importing or opening this window
can never create an installation implicitly.
"""

from __future__ import annotations

from pathlib import Path

from zlc_neutral_atom.installation_config import (
    InstallationConfigDocument,
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
        from zlc_neutral_atom.installation_package import (
            default_installation_package,
        )

        document = InstallationConfigDocument.from_parameters(
            default_installation_package().backend,
            {},
        )
    if not isinstance(document, InstallationConfigDocument):
        raise TypeError("document must be InstallationConfigDocument")
    return document, state


def open_device_manager(
    admin: DeviceAdminPort,
    *,
    document: InstallationConfigDocument | None = None,
    config_path: str | Path | None = None,
    on_runtime_changed=None,
    hide_on_close: bool | None = None,
    shutdown_on_owner_close: bool | None = None,
):
    """Open DeviceManager on one authority supplied by the application root.

    ``admin`` is the sole capability-bearing input.  The optional callback
    receives only :class:`DeviceAdminState`; a higher composition root may map
    that state back to its own application object without leaking that object
    into Workbench.
    """

    if on_runtime_changed is not None and not callable(on_runtime_changed):
        raise TypeError("on_runtime_changed must be callable or None")
    resolved_path = (
        None if config_path is None else Path(config_path).expanduser().resolve()
    )
    document, state = _initial_document(admin, document, resolved_path)

    from zlc_frontend.qt_widgets import ensure_qt_app, set_fluent_scale

    ensure_qt_app()
    set_fluent_scale(None)

    editor = DeviceConfigEditorSession(
        document,
        active_document=state.active_config,
        path=resolved_path,
        baseline_digest=(
            None if resolved_path is None else document.content_digest
        ),
    )
    controller = DeviceManagerController(editor, admin)
    if on_runtime_changed is not None:
        controller.runtime_changed.connect(on_runtime_changed)

    bound = state.active_config is not None
    from .window import launch_device_manager_window

    return launch_device_manager_window(
        controller,
        hide_on_close=(bound if hide_on_close is None else bool(hide_on_close)),
        shutdown_on_owner_close=(
            not bound
            if shutdown_on_owner_close is None
            else bool(shutdown_on_owner_close)
        ),
    )
