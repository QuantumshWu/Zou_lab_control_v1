"""Composition root for the one formal Pulse GUI.

The visible Edit/Preview/Scan product is fixed.  This module only supplies its
current document/application authorities: an existing Experiment is borrowed;
a standalone editor starts offline and may compose one owned virtual or remote
Experiment from the existing Connection controls.
"""

from __future__ import annotations

import os
from pathlib import Path

from zlc_pulse import (
    PulseDocument,
    load_deployed_pulse_target,
    pulse_target_manifest_from_lanes,
    restrict_pulse_document_to_manifest,
    validate_pulse_document_clock_grid,
)
from zlc_workbench.pulse import PulseEditorSession

from .controller import OwnedPulseConnection, PulseEditorController
from .window import launch_pulse_editor_window

__all__ = ["open_pulse_editor"]


def _standalone_workspace(value: str | Path | None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    configured = os.environ.get("ZLC_PULSE_WORKSPACE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".zlc" / "pulse-workbench").resolve()


def _editor_session(
    *,
    document: PulseDocument | None,
    path: str | Path | None,
    target,
    time_step_ns: float,
) -> PulseEditorSession:
    if document is not None and path is not None:
        raise ValueError("provide document or path, not both")
    if document is not None:
        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument or None")
        return PulseEditorSession(document)
    if path is not None:
        return PulseEditorSession.load(path)
    return PulseEditorSession.new(target, time_step_ns=time_step_ns)


def _managed_connection_mode(experiment, descriptor) -> str:
    """Project an immutable installation fact into the existing combo choice."""

    try:
        info = experiment.device_catalog[descriptor.sequencer_ref.role]
        adapter = str(info.adapter_kind).lower()
    except (AttributeError, KeyError, TypeError):
        adapter = ""
    return "virtual" if "virtualsequencer" in adapter else "remote"


def _standalone_connection_factory(workspace: Path):
    def compose(
        mode: str,
        host: str | None,
        port: int | None,
        required_document: PulseDocument,
    ) -> OwnedPulseConnection:
        # Import lazily so offline authoring does not construct or import a
        # runtime installation.  Qt receives only the narrow facade below.
        from Zou_lab_control.notebook.facade import connect

        if mode == "virtual":
            experiment = connect("virtual", repository=workspace)
        elif mode == "remote":
            if host is None or port is None:
                raise ValueError("remote Pulse connection requires host and port")
            from zlc_neutral_atom.installation_config import (
                InstallationConfigDocument,
            )

            experiment = connect(
                InstallationConfigDocument.remote_pulse(
                    host=host,
                    port=port,
                ),
                repository=workspace,
                required_pulse_document=required_document,
            )
        else:
            raise ValueError("Pulse connection mode must be virtual or remote")
        try:
            pulse = experiment.pulse
            descriptor = pulse.target
            return OwnedPulseConnection(pulse, descriptor, experiment.close)
        except BaseException:
            experiment.close()
            raise

    return compose


def open_pulse_editor(
    experiment=None,
    *,
    document: PulseDocument | None = None,
    path: str | Path | None = None,
    remote_endpoint: str | None = None,
    repository: str | Path | None = None,
):
    """Open the unchanged formal surface on current Pulse authorities."""

    if document is not None and not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument or None")
    if document is not None and path is not None:
        raise ValueError("provide document or path, not both")
    if experiment is not None and remote_endpoint is not None:
        raise ValueError("an existing Experiment already owns its Pulse connection")

    if experiment is not None:
        pulse = getattr(experiment, "pulse", None)
        if pulse is None:
            raise TypeError("experiment must expose its current Pulse facade")
        descriptor = pulse.target
        session = _editor_session(
            document=document,
            path=path,
            target=descriptor.target,
            time_step_ns=descriptor.time_step_ns,
        )
        session.bind_target(descriptor.target)
        session.replace_document(
            restrict_pulse_document_to_manifest(
                session.document,
                descriptor.manifest,
            )
        )
        validate_pulse_document_clock_grid(session.document, descriptor.clock_hz)
        controller = PulseEditorController(
            session,
            pulse=pulse,
            descriptor=descriptor,
            authoring_manifest=descriptor.manifest,
            initial_connection_mode=_managed_connection_mode(experiment, descriptor),
        )
        return launch_pulse_editor_window(controller, hide_on_close=True)

    target = load_deployed_pulse_target()
    from zlc_neutral_atom.timing.clock import default_time_step_ns

    session = _editor_session(
        document=document,
        path=path,
        target=target,
        time_step_ns=default_time_step_ns(),
    )
    authoring_manifest = pulse_target_manifest_from_lanes(
        session.document.target
    )
    workspace = _standalone_workspace(repository)
    controller = PulseEditorController(
        session,
        authoring_manifest=authoring_manifest,
        connection_factory=_standalone_connection_factory(workspace),
        initial_connection_mode="offline",
    )
    body = launch_pulse_editor_window(controller)
    if remote_endpoint is not None:
        controller.connect("remote", remote_endpoint)
    return body
