"""Qt composition for the formal Pulse editor over explicit authorities.

The Workbench owns Pulse authoring and presentation only.  Runtime installation
construction belongs to the caller: a bound editor receives a Pulse facade and
its immutable target descriptor, while a standalone editor may receive a
connection factory.  This module never imports ``connect`` and never owns or
stores the outer application authority.
"""

from __future__ import annotations

from pathlib import Path

from zlc_neutral_atom.devices.sequencer.application import PulseTargetDescriptor
from zlc_pulse import (
    PulseDocument,
    load_deployed_pulse_target,
    load_deployed_geometry_facts,
    pulse_target_manifest_from_lanes,
    restrict_pulse_document_to_manifest,
    validate_pulse_document_clock_grid,
)
from zlc_workbench.pulse_editor.session import PulseEditorSession
from zlc_storage.paths import resolve_under

from .controller import PulseEditorController, PulseRunFacade
from .window import launch_pulse_editor_window

__all__ = ["open_pulse_editor"]


def _editor_session(
    *,
    document: PulseDocument | None,
    path: str | Path | None,
    target,
    time_step_ns: float,
    pulses_root: Path,
) -> PulseEditorSession:
    if document is not None and path is not None:
        raise ValueError("provide document or path, not both")
    if document is not None:
        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument or None")
        return PulseEditorSession(document)
    if path is not None:
        return PulseEditorSession.load(resolve_under(pulses_root, path))
    return PulseEditorSession.new(target, time_step_ns=time_step_ns)


def open_pulse_editor(
    *,
    pulse: PulseRunFacade | None = None,
    descriptor: PulseTargetDescriptor | None = None,
    connection_factory=None,
    initial_connection_mode: str | None = None,
    document: PulseDocument | None = None,
    path: str | Path | None = None,
    remote_endpoint: str | None = None,
    pulses_root: Path,
    output_root: Path,
):
    """Open the unchanged Pulse surface on explicitly composed capabilities."""

    if document is not None and not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument or None")
    if document is not None and path is not None:
        raise ValueError("provide document or path, not both")
    if (pulse is None) != (descriptor is None):
        raise ValueError("pulse facade and target descriptor must appear together")
    if descriptor is not None and not isinstance(descriptor, PulseTargetDescriptor):
        raise TypeError("descriptor must be PulseTargetDescriptor")
    if connection_factory is not None and not callable(connection_factory):
        raise TypeError("connection_factory must be callable or None")

    if pulse is not None:
        if connection_factory is not None:
            raise ValueError("a bound Pulse authority cannot be replaced")
        if remote_endpoint is not None:
            raise ValueError("a bound Pulse authority already owns its endpoint")
        if initial_connection_mode not in ("virtual", "remote"):
            raise ValueError(
                "a bound Pulse authority requires its explicit virtual/remote mode"
            )
        session = _editor_session(
            document=document,
            path=path,
            target=descriptor.target,
            time_step_ns=descriptor.time_step_ns,
            pulses_root=pulses_root,
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
            initial_connection_mode=initial_connection_mode,
        )
        return launch_pulse_editor_window(
            controller,
            pulses_root=pulses_root,
            output_root=output_root,
            hide_on_close=True,
        )

    if initial_connection_mode not in (None, "offline"):
        raise ValueError("an unbound Pulse editor must initially be offline")
    if remote_endpoint is not None and connection_factory is None:
        raise ValueError("remote_endpoint requires an explicit connection_factory")

    target = load_deployed_pulse_target()
    geometry = load_deployed_geometry_facts()

    session = _editor_session(
        document=document,
        path=path,
        target=target,
        time_step_ns=1_000_000_000.0 / geometry.clock_hz,
        pulses_root=pulses_root,
    )
    controller = PulseEditorController(
        session,
        authoring_manifest=pulse_target_manifest_from_lanes(
            session.document.target
        ),
        connection_factory=connection_factory,
        initial_connection_mode="offline",
    )
    body = launch_pulse_editor_window(
        controller,
        pulses_root=pulses_root,
        output_root=output_root,
    )
    if remote_endpoint is not None:
        controller.connect("remote", remote_endpoint)
    return body
