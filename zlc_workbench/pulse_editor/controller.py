"""Qt-free current Pulse GUI controller.

The formal Qt surface submits semantic intents and observes immutable snapshots.
This owner alone coordinates editor revision, remote installation composition,
RunHandle lifecycle, worker completion, and standalone SAFE close.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Callable, Protocol
from uuid import uuid4

from zlc_neutral_atom.pulse_application import (
    AppliedPulseSnapshot,
    PulseRunObservation,
    PulseRunRequest,
    PulseTargetDescriptor,
)
from zlc_neutral_atom.runtime.run import RunHandle, RunSnapshot
from zlc_neutral_atom.timing.pulse import PulseScanProgress
from zlc_pulse import (
    FIELD_DAC,
    FIELD_DELAY,
    FIELD_DURATION,
    PORT_DAC,
    PORT_DIGITAL,
    AnalogStep,
    ApiParameter,
    OutputDelay,
    PulseDocument,
    PulseFieldRef,
    PulseExecutionForm,
    PulseTargetEditResult,
    PulseTargetManifest,
    PulseTimelineDocument,
    RepeatRegion,
    ScanParameter,
    bind_pulse_document_target,
    cycle_field_binding,
    insert_period,
    move_period,
    new_period,
    rename_port_label,
    remove_period,
    resolve_scan_point,
    replace_field_binding,
    replace_pulse_field,
    replace_pulse_document_target,
    restrict_pulse_document_to_manifest,
    set_analog_action,
    set_digital_output,
    set_output_delay,
    pulse_target_manifest_from_lanes,
    validate_pulse_document_clock_grid,
)
from zlc_workbench.pulse import PulseEditorSession, project_pulse_preview
from zlc_pulse.authoring import clear_port as clear_document_port
from zlc_pulse.authoring import clear_pulse_schedule
from .preview_projection import (
    pulse_preview_status,
    pulse_timeline_render_kwargs,
)
from .scan_workspace import (
    ScanCandidateResult,
    ScanCandidateSource,
    ScanWorkspaceCandidate,
    ScanWorkspaceSnapshot,
    candidate_from_document,
    candidate_snapshot,
    commit_scan_candidate,
    default_scan_program,
    execute_scan_program,
    format_scan_slots,
    format_scan_table,
    load_scan_array as read_scan_array,
    load_scan_program_source,
    save_scan_array as write_scan_array,
    scan_column_specs,
    scan_slot_schema,
)


class PulseRunFacade(Protocol):
    """The existing application facade shape consumed by the Workbench.

    This is structural only: it neither wraps nor owns a second run state.
    """

    @property
    def target(self) -> PulseTargetDescriptor: ...

    def request(
        self,
        document: PulseDocument,
        execution_form: PulseExecutionForm,
        *,
        api_values: dict[str, int | float] | None = None,
        scan_sweep_count: int = 1,
    ) -> PulseRunRequest: ...

    def start(self, request: PulseRunRequest) -> RunHandle: ...

    def snapshot(self) -> AppliedPulseSnapshot | None: ...

    def observe_active(self) -> PulseRunObservation | None: ...

    def cancel_active(self, reason: str = ...) -> object | None: ...

    def observe_scan_progress(self) -> PulseScanProgress | None: ...


@dataclass(frozen=True)
class OwnedPulseConnection:
    """One composed standalone authority, without exposing its Experiment object."""

    pulse: PulseRunFacade
    descriptor: PulseTargetDescriptor
    close: Callable[[], None]

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, PulseTargetDescriptor):
            raise TypeError("connection descriptor must be PulseTargetDescriptor")
        required = (
            "request",
            "start",
            "snapshot",
            "observe_active",
            "cancel_active",
        )
        if any(not callable(getattr(self.pulse, name, None)) for name in required):
            raise TypeError("connection pulse must provide the current application facade")
        if not callable(self.close):
            raise TypeError("connection close must be callable")


@dataclass(frozen=True)
class RenderedPulsePreview:
    """One worker-rendered front bound to an exact document and view revision."""

    document_generation: int
    editor_revision: int
    presentation_revision: int
    timeline: PulseTimelineDocument
    raster: object
    payload: object
    size: str
    status: str
    pixel_ratio: float

    def __post_init__(self) -> None:
        from zlc_frontend.render import PulsePanelPayload, RasterBuffer

        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.document_generation,
                self.editor_revision,
                self.presentation_revision,
            )
        ):
            raise ValueError("preview revisions must be non-negative integers")
        if not isinstance(self.timeline, PulseTimelineDocument):
            raise TypeError("preview timeline must be PulseTimelineDocument")
        if not isinstance(self.raster, RasterBuffer):
            raise TypeError("preview raster must be RasterBuffer")
        if not isinstance(self.payload, PulsePanelPayload):
            raise TypeError("preview payload must be PulsePanelPayload")
        if self.payload.document_input.document_revision != self.editor_revision:
            raise ValueError("preview payload belongs to another editor revision")
        if self.payload.viewport.display_revision != self.presentation_revision:
            raise ValueError("preview payload belongs to another presentation revision")
        if not isinstance(self.size, str) or not self.size:
            raise ValueError("preview size must be non-empty text")
        if not isinstance(self.status, str):
            raise TypeError("preview status must be text")
        ratio = float(self.pixel_ratio)
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("preview pixel_ratio must be finite and positive")
        object.__setattr__(self, "pixel_ratio", ratio)


def _cancel_and_wait_detached_run(handle: RunHandle) -> RunSnapshot:
    """Reap an impossible/stale admission without abandoning its SAFE owner."""

    handle.cancel("stale PulseGUI start result")
    return handle.wait()


def _applied_authoring_document(snapshot: AppliedPulseSnapshot) -> PulseDocument:
    """Show exact applied API values while retaining their authored bindings."""

    document = snapshot.source_document
    values = dict(snapshot.api_values)
    for parameter in document.api_parameters:
        document = replace_pulse_field(
            document,
            parameter.field,
            values[parameter.parameter_id],
            unit=parameter.unit,
        )
    return document


def _changed_analog_cells(
    before: PulseDocument,
    after: PulseDocument,
    ports: tuple[str, ...] | None = None,
) -> tuple[tuple[str, str], ...]:
    """Identify only Edge/Ramp/Hold presentations changed by one edit."""

    selected = (
        tuple(port.key for port in after.target.ports if port.kind == PORT_DAC)
        if ports is None
        else ports
    )

    def values(document: PulseDocument) -> dict[tuple[str, str], tuple[str, int]]:
        carried = {port: 0 for port in selected}
        result = {}
        for period in document.periods:
            actions = {step.port: step for step in period.analog_steps}
            for port in selected:
                action = actions.get(port)
                if action is None:
                    result[(period.period_id, port)] = ("hold", carried[port])
                else:
                    carried[port] = action.value
                    result[(period.period_id, port)] = (action.mode, action.value)
        return result

    old = values(before)
    return tuple(
        cell for cell, value in values(after).items()
        if cell in old and old[cell] != value
    )


def parse_remote_endpoint(value: str) -> tuple[str, int]:
    """Parse ``host[:port]`` and bracketed IPv6 without guessing a blank host."""

    text = str(value).strip()
    if not text:
        raise ValueError("remote pulse endpoint must be host or host:port")
    if text.startswith("["):
        close = text.find("]")
        if close < 0:
            raise ValueError("bracketed IPv6 endpoint is missing ']'")
        host = text[1:close].strip()
        suffix = text[close + 1 :]
        if suffix and not suffix.startswith(":"):
            raise ValueError("bracketed IPv6 endpoint has trailing text")
        port_text = suffix[1:] if suffix else "18861"
    elif text.count(":") == 1:
        host, port_text = text.rsplit(":", 1)
        host = host.strip()
    else:
        host, port_text = text, "18861"
    if not host:
        raise ValueError("remote pulse endpoint requires a host")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError("remote pulse endpoint port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("remote pulse endpoint port must be between 1 and 65535")
    return host, port


@dataclass(frozen=True)
class PulseEditorControllerSnapshot:
    document: PulseDocument
    document_generation: int
    editor_revision: int
    path: Path | None
    file_state: str
    dirty: bool
    connection_state: str
    connection_mode: str
    connection_endpoint: str
    target_descriptor: PulseTargetDescriptor | None
    target_manifest: PulseTargetManifest
    display_visible_ports: tuple[str, ...]
    run_snapshot: RunSnapshot | None
    run_generation: int | None
    run_revision: int | None
    preview_revision: int | None
    preview_generation: int | None
    preview: PulseTimelineDocument | None
    rendered_preview: RenderedPulsePreview | None
    preview_error: str
    preview_notice: str
    scan_workspace: ScanWorkspaceSnapshot
    applied_snapshot: AppliedPulseSnapshot | None
    scan_progress: PulseScanProgress | None
    held_scan_point: tuple[
        int,
        int,
        tuple[tuple[str, int | float], ...],
    ] | None
    diagnostic: str
    file_busy: bool
    run_busy: bool
    close_requested: bool
    close_complete: bool


@dataclass(frozen=True)
class PulseScanProgressUpdate:
    """One changed scan cursor front, independent of the Scan editor draft."""

    scan_progress: PulseScanProgress | None
    applied_snapshot: AppliedPulseSnapshot | None
    held_scan_point: tuple[
        int,
        int,
        tuple[tuple[str, int | float], ...],
    ] | None


@dataclass(frozen=True)
class PulseRuntimeUpdate:
    """Changed Run/connection facts without an editor or Scan projection."""

    connection_state: str
    connection_mode: str
    connection_endpoint: str
    target_descriptor: PulseTargetDescriptor | None
    run_snapshot: RunSnapshot | None
    run_generation: int | None
    run_revision: int | None
    applied_snapshot: AppliedPulseSnapshot | None
    scan_progress: PulseScanProgress | None
    held_scan_point: tuple[
        int,
        int,
        tuple[tuple[str, int | float], ...],
    ] | None
    diagnostic: str
    file_busy: bool
    run_busy: bool
    close_requested: bool
    close_complete: bool


@dataclass(frozen=True)
class PulseEditorProjection:
    """Only the editor revision and facts derived from that revision.

    This is the synchronous Qt edit boundary.  It deliberately excludes Run,
    connection, worker, preview-front, and close state so an ordinary unit or
    value commit cannot manufacture an application-wide snapshot.
    """

    document: PulseDocument
    document_generation: int
    editor_revision: int
    path: Path | None
    file_state: str
    dirty: bool
    target_manifest: PulseTargetManifest
    display_visible_ports: tuple[str, ...]
    scan_workspace: ScanWorkspaceSnapshot


@dataclass(frozen=True)
class PulseEditorLocalDelta:
    """One same-thread edit result, not a projected application snapshot.

    ``document`` is the already-committed immutable domain object (one borrowed
    reference, no copy/serialization).  The stable ids say which widgets may
    read from it; the view never retains it or traverses unrelated controls.
    """

    document_generation: int
    base_revision: int
    editor_revision: int
    document: PulseDocument
    document_name: str | None = None
    period_ids: tuple[str, ...] = ()
    port_ids: tuple[str, ...] = ()
    digital_cells: tuple[tuple[str, str], ...] = ()
    fields: tuple[PulseFieldRef, ...] = ()
    analog_cells: tuple[tuple[str, str], ...] = ()
    period_structure: bool = False
    refresh_bindings: bool = False
    display_visible_ports: tuple[str, ...] | None = None
    scan_workspace: ScanWorkspaceSnapshot | None = None
    scan_slots_text: str | None = None
    scan_sweep_count: int | None = None


class PulseEditorController:
    """Single mutable application owner behind the frozen formal Pulse UI."""

    def __init__(
        self,
        editor: PulseEditorSession,
        *,
        pulse: PulseRunFacade | None = None,
        descriptor: PulseTargetDescriptor | None = None,
        authoring_manifest: PulseTargetManifest | None = None,
        connection_factory: Callable[
            [str, str | None, int | None, PulseDocument], OwnedPulseConnection
        ]
        | None = None,
        initial_connection_mode: str | None = None,
        initial_connection_endpoint: str = "",
        notify: Callable[[], None] | None = None,
    ) -> None:
        if not isinstance(editor, PulseEditorSession):
            raise TypeError("editor must be PulseEditorSession")
        if (pulse is None) != (descriptor is None):
            raise ValueError("pulse facade and descriptor must appear together")
        if pulse is not None and connection_factory is not None:
            raise ValueError("an installation-managed editor cannot replace its authority")
        if authoring_manifest is None:
            authoring_manifest = (
                descriptor.manifest
                if descriptor is not None
                else pulse_target_manifest_from_lanes(editor.document.target)
            )
        if not isinstance(authoring_manifest, PulseTargetManifest):
            raise TypeError("authoring_manifest must be PulseTargetManifest")
        if (
            authoring_manifest.target.abi_fingerprint
            != editor.document.target.abi_fingerprint
        ):
            raise ValueError("authoring manifest target differs from editor document")
        if pulse is None:
            mode = "offline" if initial_connection_mode is None else str(initial_connection_mode)
            if mode != "offline":
                raise ValueError("an offline editor must initially display Offline")
        else:
            mode = "virtual" if initial_connection_mode is None else str(initial_connection_mode)
            if mode not in ("virtual", "remote"):
                raise ValueError("an online Pulse authority must be virtual or remote")
        self._editor = editor
        self._pulse = pulse
        self._descriptor = descriptor
        self._authoring_manifest = authoring_manifest
        self._connection_factory = connection_factory
        self._owned_connection: OwnedPulseConnection | None = None
        self._notify = notify or (lambda: None)
        self._pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="zlc-pulse-editor"
        )
        self._lock = threading.Lock()
        self._tracked: set[Future] = set()
        self._results: list[tuple[str, object, Future]] = []
        self._editor_generation = 0
        self._operation_generation = 0
        self._preview_inflight = None
        self._preview_requested = None
        self._preview_revision: int | None = None
        self._preview_generation: int | None = None
        self._preview: PulseTimelineDocument | None = None
        self._rendered_preview: RenderedPulsePreview | None = None
        self._preview_error = ""
        self._preview_notice = ""
        self._preview_export_inflight = False
        self._preview_document_namespace = f"pulse-editor-{uuid4().hex}"
        self._preview_include_off = False
        self._preview_size: str | None = None
        self._preview_pixel_ratio = 1.0
        self._preview_x_limits: tuple[float, float] | None = None
        self._preview_presentation_revision = 0
        self._scan_operation_generation = 0
        self._scan_busy_operation: str | None = None
        self._scan_diagnostic = ""
        self._scan_selected: ScanCandidateSource = "generated"
        self._scan_generated: ScanWorkspaceCandidate | None = None
        self._scan_loaded: ScanWorkspaceCandidate | None = None
        self._scan_source_text = ""
        self._scan_source_baseline: str | None = None
        self._scan_auto_template = ""
        self._scan_source_revision = 0
        self._scan_workspace_cache_key: tuple[object, ...] | None = None
        self._scan_workspace_cache: ScanWorkspaceSnapshot | None = None
        self._scan_workspace_cache_generated: object | None = None
        self._scan_workspace_cache_loaded: object | None = None
        self._scan_progress: PulseScanProgress | None = None
        self._scan_progress_inflight = False
        self._held_scan_source: PulseDocument | None = None
        self._held_scan_index: int | None = None
        self._pending_hold: tuple[PulseDocument, int, str] | None = None
        self._save_inflight = False
        self._load_inflight = False
        self._connect_inflight = False
        self._run_starting = False
        self._cancel_when_started = False
        self._pending_start: tuple[
            PulseDocument,
            PulseExecutionForm,
            tuple[tuple[str, int | float], ...],
            int,
            int | None,
            int | None,
        ] | None = None
        self._handle: RunHandle | None = None
        self._run_snapshot: RunSnapshot | None = None
        self._applied_snapshot: AppliedPulseSnapshot | None = None
        self._applied_authoring_cache_source: AppliedPulseSnapshot | None = None
        self._applied_authoring_cache_fingerprint: str | None = None
        self._run_generation: int | None = None
        self._run_revision: int | None = None
        self._owner_reaped = True
        self._reap_inflight = False
        self._owned_close_inflight = False
        self._close_requested = False
        self._close_complete = False
        self._borrowed_authority_retire = threading.Event()
        self._borrowed_authority_retired = False
        self._pool_closed = False
        self._diagnostic = ""
        self._connection_state = "ready" if pulse is not None else "offline"
        self._connection_mode = mode
        self._connection_endpoint = str(initial_connection_endpoint)
        self._active_manifest_mode = mode if descriptor is not None else "offline"
        self._display_ports_by_mode: dict[str, tuple[str, ...]] = {
            self._active_manifest_mode: (
                editor.document.visible_ports
                if self._active_manifest_mode == "offline"
                else descriptor.manifest.available_port_keys
            )
        }
        self._pending_connection_request: tuple[
            str,
            str | None,
            int | None,
            str,
        ] | None = None
        self._reset_scan_workspace(editor.document)

    def set_notify(self, notify: Callable[[], None]) -> None:
        if not callable(notify):
            raise TypeError("notify must be callable")
        self._notify = notify

    @property
    def worker_idle(self) -> bool:
        with self._lock:
            return not self._tracked and not self._results

    @property
    def current_document(self) -> PulseDocument:
        """Return the owner-held immutable document without deriving GUI state."""

        return self._editor.document

    @property
    def dirty(self) -> bool:
        return self._editor.dirty

    def editor_projection(self) -> PulseEditorProjection:
        """Project one committed editor revision, without runtime state."""

        revision, document = self._editor.snapshot()
        source_manifest = (
            self._descriptor.manifest
            if self._descriptor is not None
            else self._authoring_manifest
        )
        target_manifest = source_manifest.with_target(document.target)
        display_visible_ports = self._display_ports_for(
            self._active_manifest_mode,
            target_manifest,
            document=document,
        )
        return PulseEditorProjection(
            document=document,
            document_generation=self._editor_generation,
            editor_revision=revision,
            path=self._editor.path,
            file_state=self._editor.file_state,
            dirty=self._editor.dirty,
            target_manifest=target_manifest,
            display_visible_ports=display_visible_ports,
            scan_workspace=self._scan_workspace_snapshot(document),
        )

    def _presentation_state_key(
        self,
        *,
        include_scan_progress: bool = True,
    ) -> tuple[object, ...]:
        """Return cheap owner facts without constructing any GUI projection.

        Worker completion is a reason to wake the owner, not proof that a
        visible fact changed.  This key deliberately uses immutable values and
        stable object identities; it never formats the scan workspace, rebuilds
        an effective PulseDocument, or snapshots a RunHandle.
        """

        editor = self._editor
        return (
            self._editor_generation,
            editor.revision,
            id(editor.document),
            editor.path,
            editor.file_state,
            editor.dirty,
            self._connection_state,
            self._connection_mode,
            self._connection_endpoint,
            self._active_manifest_mode,
            id(self._descriptor),
            tuple(sorted(self._display_ports_by_mode.items())),
            self._run_snapshot,
            self._applied_snapshot,
            self._run_generation,
            self._run_revision,
            self._run_starting,
            self._pending_start is not None,
            id(self._handle),
            self._owner_reaped,
            self._reap_inflight,
            self._preview_revision,
            self._preview_generation,
            id(self._preview),
            id(self._rendered_preview),
            self._preview_error,
            self._preview_notice,
            self._preview_export_inflight,
            self._scan_source_revision,
            self._scan_source_baseline,
            self._scan_selected,
            id(self._scan_generated),
            id(self._scan_loaded),
            self._scan_busy_operation,
            self._scan_diagnostic,
            self._scan_progress if include_scan_progress else None,
            id(self._held_scan_source),
            self._held_scan_index,
            self._save_inflight,
            self._load_inflight,
            self._connect_inflight,
            self._owned_close_inflight,
            self._diagnostic,
            self._close_requested,
            self._close_complete,
        )

    def _full_projection_state_key(self) -> tuple[object, ...]:
        """Facts that actually require rebuilding a full GUI projection."""

        editor = self._editor
        return (
            self._editor_generation,
            editor.revision,
            id(editor.document),
            editor.path,
            editor.file_state,
            editor.dirty,
            self._active_manifest_mode,
            id(self._descriptor),
            tuple(sorted(self._display_ports_by_mode.items())),
            self._preview_revision,
            self._preview_generation,
            id(self._preview),
            id(self._rendered_preview),
            self._preview_error,
            self._preview_notice,
            self._scan_source_revision,
            self._scan_source_baseline,
            self._scan_selected,
            id(self._scan_generated),
            id(self._scan_loaded),
            self._scan_busy_operation,
            self._scan_diagnostic,
        )

    def runtime_update(self) -> PulseRuntimeUpdate:
        """Project current narrow runtime facts without polling or editor work."""

        return PulseRuntimeUpdate(
            connection_state=self._connection_state,
            connection_mode=self._connection_mode,
            connection_endpoint=self._connection_endpoint,
            target_descriptor=self._descriptor,
            run_snapshot=self._run_snapshot,
            run_generation=self._run_generation,
            run_revision=self._run_revision,
            applied_snapshot=self._applied_snapshot,
            scan_progress=self._scan_progress,
            held_scan_point=self._held_scan_snapshot(),
            diagnostic=self._diagnostic,
            file_busy=(
                self._save_inflight
                or self._load_inflight
                or self._preview_export_inflight
            ),
            run_busy=self._run_busy(),
            close_requested=self._close_requested,
            close_complete=self._close_complete,
        )

    def snapshot(self) -> PulseEditorControllerSnapshot:
        editor = self.editor_projection()
        return PulseEditorControllerSnapshot(
            document=editor.document,
            document_generation=editor.document_generation,
            editor_revision=editor.editor_revision,
            path=editor.path,
            file_state=editor.file_state,
            dirty=editor.dirty,
            connection_state=self._connection_state,
            connection_mode=self._connection_mode,
            connection_endpoint=self._connection_endpoint,
            target_descriptor=self._descriptor,
            target_manifest=editor.target_manifest,
            display_visible_ports=editor.display_visible_ports,
            run_snapshot=self._run_snapshot,
            run_generation=self._run_generation,
            run_revision=self._run_revision,
            preview_revision=self._preview_revision,
            preview_generation=self._preview_generation,
            preview=self._preview,
            rendered_preview=self._rendered_preview,
            preview_error=self._preview_error,
            preview_notice=self._preview_notice,
            scan_workspace=editor.scan_workspace,
            applied_snapshot=self._applied_snapshot,
            scan_progress=self._scan_progress,
            held_scan_point=self._held_scan_snapshot(),
            diagnostic=self._diagnostic,
            file_busy=(
                self._save_inflight
                or self._load_inflight
                or self._preview_export_inflight
            ),
            run_busy=self._run_busy(),
            close_requested=self._close_requested,
            close_complete=self._close_complete,
        )

    def _display_ports_for(
        self,
        mode: str,
        manifest: PulseTargetManifest,
        *,
        document: PulseDocument | None = None,
    ) -> tuple[str, ...]:
        source = self._editor.document if document is None else document
        saved = self._display_ports_by_mode.get(mode)
        if saved is None:
            saved = (
                source.visible_ports
                if mode == "offline"
                else manifest.available_port_keys
            )
        selected = set(saved)
        return tuple(
            key for key in manifest.available_port_keys if key in selected
        )

    def replace_document(self, document: PulseDocument) -> int:
        self._require_authoring_available()
        previous = self._editor.document
        if document == previous:
            return self._editor.revision
        if self._descriptor is not None:
            document = bind_pulse_document_target(document, self._descriptor.target)
            document = restrict_pulse_document_to_manifest(
                document,
                self._descriptor.manifest,
            )
            validate_pulse_document_clock_grid(document, self._descriptor.clock_hz)
        elif (
            document.target.abi_fingerprint
            != self._authoring_manifest.target.abi_fingerprint
        ):
            self._authoring_manifest = pulse_target_manifest_from_lanes(
                document.target
            )
        document = self._invalidate_changed_scan_schema(previous, document)
        revision = self._editor.replace_document(document)
        if self._active_manifest_mode == "offline":
            self._display_ports_by_mode["offline"] = document.visible_ports
        self._observe_document_schema_change(previous, document)
        self._invalidate_document_preview()
        return revision

    def _invalidate_document_preview(self) -> None:
        """Retire a stale front without rendering a hidden Preview tab.

        The Preview tab explicitly requests the current revision when opened.
        An already-running older render is allowed to finish and be discarded;
        it must not chain a render for every intermediate editor revision.
        """

        self._preview = None
        self._rendered_preview = None
        self._preview_error = ""
        self._preview_revision = None
        self._preview_generation = None

    def _invalidate_scan_workspace_projection(self) -> None:
        self._scan_workspace_cache_key = None
        self._scan_workspace_cache = None
        self._scan_workspace_cache_generated = None
        self._scan_workspace_cache_loaded = None

    def _local_delta(
        self,
        base_revision: int,
        *,
        period_ids: tuple[str, ...] = (),
        port_ids: tuple[str, ...] = (),
        digital_cells: tuple[tuple[str, str], ...] = (),
        fields: tuple[PulseFieldRef, ...] = (),
        analog_cells: tuple[tuple[str, str], ...] = (),
        include_document_name: bool = False,
        period_structure: bool = False,
        refresh_bindings: bool = False,
        include_visible_ports: bool = False,
        include_scan_workspace: bool = False,
        include_scan_slots: bool = False,
        include_scan_sweep_count: bool = False,
    ) -> PulseEditorLocalDelta:
        """Describe the committed local change without projecting the document."""

        document = self._editor.document
        revision = self._editor.revision
        workspace = None
        if include_scan_workspace:
            self._invalidate_scan_workspace_projection()
            workspace = self._scan_workspace_snapshot(document)
        return PulseEditorLocalDelta(
            document_generation=self._editor_generation,
            base_revision=base_revision,
            editor_revision=revision,
            document=document,
            document_name=(document.name if include_document_name else None),
            period_ids=period_ids,
            port_ids=port_ids,
            digital_cells=digital_cells,
            fields=fields,
            analog_cells=analog_cells,
            period_structure=period_structure,
            refresh_bindings=refresh_bindings,
            display_visible_ports=(
                self._display_ports_for(
                    self._active_manifest_mode,
                    (
                        self._descriptor.manifest.with_target(document.target)
                        if self._descriptor is not None
                        else self._authoring_manifest.with_target(document.target)
                    ),
                    document=document,
                )
                if include_visible_ports
                else None
            ),
            scan_workspace=workspace,
            scan_slots_text=(
                format_scan_slots(document) if include_scan_slots else None
            ),
            scan_sweep_count=(
                document.scan_sweep_count if include_scan_sweep_count else None
            ),
        )

    def apply_target_manifest(
        self,
        manifest: PulseTargetManifest,
        *,
        cascade: bool = False,
    ) -> PulseTargetEditResult:
        """Commit one Offline Target draft and document remap as one owner turn."""

        self._require_authoring_available()
        if self._descriptor is not None or self._connection_mode != "offline":
            raise RuntimeError("Target topology is editable only in Offline mode")
        previous = self._editor.document
        result = replace_pulse_document_target(
            previous,
            manifest,
            cascade=cascade,
        )
        document = self._invalidate_changed_scan_schema(previous, result.document)
        self._editor.replace_document(document)
        self._authoring_manifest = result.manifest.with_target(document.target)
        self._display_ports_by_mode["offline"] = document.visible_ports
        self._observe_document_schema_change(previous, document)
        self._invalidate_document_preview()
        return PulseTargetEditResult(document, self._authoring_manifest, result.impact)

    def rename_document(self, name: str) -> PulseEditorLocalDelta | None:
        """Commit the visible pulse name without making the view an authority."""

        document = self._editor.document
        normalized = str(name)
        if normalized == document.name:
            return None
        base_revision = self._editor.revision
        self.replace_document(replace(document, name=normalized))
        return self._local_delta(base_revision, include_document_name=True)

    def set_scan_sweep_count(self, count: int) -> PulseEditorLocalDelta | None:
        """Commit the saved operator default; runtime remains request-explicit."""

        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("scan sweep count must be an integer")
        if count < 0:
            raise ValueError("scan sweep count must be non-negative")
        document = self._editor.document
        if count == document.scan_sweep_count:
            return None
        base_revision = self._editor.revision
        self.replace_document(
            replace(document, scan_sweep_count=count)
        )
        return self._local_delta(
            base_revision,
            include_scan_sweep_count=True,
        )

    def rename_period(
        self,
        period_id: str,
        name: str,
    ) -> PulseEditorLocalDelta | None:
        document = self._editor.document
        if period_id not in document.period_by_id:
            raise KeyError(f"unknown period {period_id!r}")
        normalized = str(name)
        if document.period_by_id[period_id].name == normalized:
            return None
        base_revision = self._editor.revision
        periods = tuple(
            replace(period, name=normalized) if period.period_id == period_id else period
            for period in document.periods
        )
        self.replace_document(replace(document, periods=periods))
        has_scan_display = any(
            parameter.field.period_id == period_id
            for parameter in document.scan_parameters
        )
        return self._local_delta(
            base_revision,
            period_ids=(period_id,),
            include_scan_workspace=has_scan_display,
        )

    def rename_port(
        self,
        port: str,
        label: str,
    ) -> PulseEditorLocalDelta | None:
        document = self._editor.document
        normalized = str(label).strip() or str(port)
        if document.target.by_key[str(port)].label == normalized:
            return None
        base_revision = self._editor.revision
        self.replace_document(rename_port_label(document, port, normalized))
        has_scan_display = any(
            parameter.field.port == str(port)
            for parameter in document.scan_parameters
        )
        return self._local_delta(
            base_revision,
            port_ids=(str(port),),
            include_scan_workspace=has_scan_display,
        )

    def set_period_duration(
        self,
        period_id: str,
        value: int | float,
        unit: str,
    ) -> PulseEditorLocalDelta | None:
        document = self._editor.document
        field = PulseFieldRef(FIELD_DURATION, period_id)
        current_value, current_unit = document.field_value(field)
        if current_value == value and current_unit == str(unit):
            return None
        base_revision = self._editor.revision
        self.replace_document(
            replace_pulse_field(document, field, value, unit=unit)
        )
        was_bound = any(
            parameter.field == field
            for parameter in (*document.scan_parameters, *document.api_parameters)
        )
        return self._local_delta(
            base_revision,
            fields=(field,),
            include_scan_slots=was_bound,
        )

    def set_digital(
        self,
        period_id: str,
        port: str,
        high: bool,
    ) -> PulseEditorLocalDelta | None:
        document = self._editor.document
        target_port = document.target.by_key[str(port)]
        lane_index = document.target.raw_lanes.index(target_port.lanes[0])
        if bool(document.period_by_id[period_id].states[lane_index]) == bool(high):
            return None
        base_revision = self._editor.revision
        self.replace_document(set_digital_output(document, period_id, port, high))
        return self._local_delta(
            base_revision,
            digital_cells=((period_id, str(port)),),
        )

    def set_analog(
        self,
        period_id: str,
        port: str,
        mode: str,
        value: int | float | None,
        *,
        cascade: bool = False,
    ) -> PulseEditorLocalDelta | None:
        document = self._editor.document
        base_revision = self._editor.revision
        normalized = str(mode).strip().lower()
        action = None
        if normalized != "hold":
            if normalized not in {"edge", "ramp"}:
                raise ValueError("DAC mode must be Edge, Ramp, or Hold")
            if value is None:
                raise ValueError("Edge and Ramp require a DAC value")
            action = AnalogStep(port, normalized, value)
        result = set_analog_action(
            document,
            period_id,
            port,
            action,
            cascade=cascade,
        )
        self.replace_document(result.document)
        if self._editor.revision == base_revision:
            return None
        after = self._editor.document
        edited_field = PulseFieldRef(FIELD_DAC, period_id, str(port))
        binding_changed = (
            document.scan_parameters,
            document.api_parameters,
        ) != (after.scan_parameters, after.api_parameters)
        was_bound = any(
            parameter.field == edited_field
            for parameter in (*document.scan_parameters, *document.api_parameters)
        )
        return self._local_delta(
            base_revision,
            fields=(edited_field,),
            analog_cells=_changed_analog_cells(
                document,
                after,
                (str(port),),
            ),
            refresh_bindings=binding_changed,
            include_scan_workspace=binding_changed,
            include_scan_slots=was_bound and not binding_changed,
        )

    def set_delay(
        self,
        port: str,
        value: int | float | None,
        unit: str = "ns",
        *,
        cascade: bool = False,
    ) -> PulseEditorLocalDelta | None:
        document = self._editor.document
        base_revision = self._editor.revision
        delay = None if value is None else OutputDelay(port, value, unit)
        result = set_output_delay(
            document,
            port,
            delay,
            cascade=cascade,
        )
        self.replace_document(result.document)
        if self._editor.revision == base_revision:
            return None
        after = self._editor.document
        binding_changed = (
            document.scan_parameters,
            document.api_parameters,
        ) != (after.scan_parameters, after.api_parameters)
        return self._local_delta(
            base_revision,
            fields=(PulseFieldRef(FIELD_DELAY, None, str(port)),),
            refresh_bindings=binding_changed,
            include_scan_workspace=binding_changed,
        )

    def set_visible_ports(
        self,
        ports: tuple[str, ...] | list[str],
    ) -> PulseEditorLocalDelta | None:
        document = self._editor.document
        requested = tuple(str(port) for port in ports)
        manifest = (
            self._descriptor.manifest.with_target(document.target)
            if self._descriptor is not None
            else self._authoring_manifest.with_target(document.target)
        )
        supported = set(manifest.available_port_keys)
        unknown = set(requested) - supported
        if unknown:
            raise KeyError(f"unknown visible pulse ports: {sorted(unknown)}")
        if len(set(requested)) != len(requested):
            raise ValueError("visible pulse ports must be unique")
        selected = set(requested)
        ordered = tuple(
            key for key in manifest.available_port_keys if key in selected
        )
        if ordered == self._display_ports_for(
            self._active_manifest_mode,
            manifest,
            document=document,
        ):
            return None
        base_revision = self._editor.revision
        self._display_ports_by_mode[self._active_manifest_mode] = ordered
        if self._active_manifest_mode == "offline":
            self.replace_document(replace(document, visible_ports=ordered))
        return self._local_delta(
            base_revision,
            include_visible_ports=True,
        )

    def clear_port(self, port: str) -> PulseEditorLocalDelta | None:
        """Apply the formal row-clear action through the domain owner."""

        document = self._editor.document
        base_revision = self._editor.revision
        result = clear_document_port(
            document,
            str(port),
            cascade=True,
        )
        self.replace_document(result.document)
        if self._editor.revision == base_revision:
            return None
        after = self._editor.document
        target_port = after.target.by_key[str(port)]
        binding_changed = (
            document.scan_parameters,
            document.api_parameters,
        ) != (after.scan_parameters, after.api_parameters)
        return self._local_delta(
            base_revision,
            digital_cells=(
                tuple((period.period_id, str(port)) for period in after.periods)
                if target_port.kind == PORT_DIGITAL
                else ()
            ),
            analog_cells=(
                _changed_analog_cells(document, after, (str(port),))
                if target_port.kind == PORT_DAC
                else ()
            ),
            refresh_bindings=binding_changed,
            include_scan_workspace=binding_changed,
        )

    def clear_all(self) -> PulseEditorLocalDelta | None:
        """Apply the formal destructive schedule reset as one document revision."""

        base_revision = self._editor.revision
        result = clear_pulse_schedule(self._editor.document, cascade=True)
        self.replace_document(result.document)
        if self._editor.revision == base_revision:
            return None
        return self._local_delta(
            base_revision,
            period_structure=True,
            refresh_bindings=True,
            include_scan_workspace=True,
        )

    def add_period(
        self,
        *,
        before_period_id: str | None = None,
        duration: int | float = 1000,
        unit: str = "ns",
        name: str = "",
        cascade: bool = False,
    ) -> PulseEditorLocalDelta:
        document = self._editor.document
        base_revision = self._editor.revision
        period = new_period(document, duration=duration, unit=unit, name=name)
        result = insert_period(
            document,
            period=period,
            before=before_period_id,
            cascade=cascade,
        )
        self.replace_document(result.document)
        return self._local_delta(
            base_revision,
            period_ids=(period.period_id,),
            period_structure=True,
            include_scan_workspace=bool(
                document.scan_parameters or document.api_parameters
            ),
        )

    def move_period(
        self,
        period_id: str,
        before_period_id: str | None,
        *,
        cascade: bool = False,
    ) -> PulseEditorLocalDelta | None:
        document = self._editor.document
        base_revision = self._editor.revision
        result = move_period(
            document,
            period_id=period_id,
            before=before_period_id,
            cascade=cascade,
        )
        self.replace_document(result.document)
        if self._editor.revision == base_revision:
            return None
        return self._local_delta(
            base_revision,
            analog_cells=_changed_analog_cells(document, self._editor.document),
            period_structure=True,
            include_scan_workspace=bool(
                document.scan_parameters or document.api_parameters
            ),
        )

    def remove_period(
        self,
        period_id: str,
        *,
        cascade: bool = False,
    ) -> PulseEditorLocalDelta | None:
        document = self._editor.document
        base_revision = self._editor.revision
        result = remove_period(
            document,
            period_id,
            cascade=cascade,
        )
        self.replace_document(result.document)
        if self._editor.revision == base_revision:
            return None
        return self._local_delta(
            base_revision,
            analog_cells=_changed_analog_cells(document, self._editor.document),
            period_structure=True,
            refresh_bindings=True,
            include_scan_workspace=bool(
                document.scan_parameters or document.api_parameters
            ),
        )

    def set_repeat(
        self,
        start_period_id: str | None,
        end_period_id: str | None = None,
        count: int = 2,
    ) -> PulseEditorLocalDelta | None:
        document = self._editor.document
        if start_period_id is None:
            if end_period_id is not None:
                raise ValueError("repeat end requires a repeat start")
            repeat = None
        else:
            if end_period_id is None:
                raise ValueError("repeat start requires a repeat end")
            repeat = RepeatRegion(start_period_id, end_period_id, count)
        if repeat == document.repeat:
            return None
        base_revision = self._editor.revision
        self.replace_document(replace(document, repeat=repeat))
        return self._local_delta(
            base_revision,
            period_structure=True,
        )

    def replace_binding(
        self,
        field: PulseFieldRef,
        kind: str | None,
        *,
        parameter_id: str | None = None,
        label: str = "",
        unit: str | None = None,
        cascade: bool = False,
    ) -> int:
        document = self._editor.document
        _value, current_unit = document.field_value(field)
        binding = None
        if kind is not None:
            normalized = str(kind).strip().lower()
            if parameter_id is None:
                raise ValueError("a bound pulse field requires a stable ParameterId")
            binding_unit = current_unit if unit is None else str(unit)
            if normalized == "scan":
                binding = ScanParameter(
                    str(parameter_id),
                    field,
                    str(label),
                    binding_unit,
                )
            elif normalized == "api":
                binding = ApiParameter(str(parameter_id), field, binding_unit)
            else:
                raise ValueError("binding kind must be Scan, API, or None")
        result = replace_field_binding(
            document,
            field,
            binding,
            cascade=cascade,
        )
        return self.replace_document(result.document)

    def cycle_binding(self, field: PulseFieldRef) -> PulseEditorLocalDelta | None:
        document = self._editor.document
        base_revision = self._editor.revision
        result = cycle_field_binding(
            document,
            field,
            cascade=True,
        )
        self.replace_document(result.document)
        if self._editor.revision == base_revision:
            return None
        return self._local_delta(
            base_revision,
            fields=(field,),
            refresh_bindings=True,
            include_scan_workspace=True,
        )

    def set_scan_template(self, kind: str) -> PulseEditorLocalDelta | None:
        """Replace the code buffer with the current typed slot template."""

        self._require_not_closing()
        source = default_scan_program(self._editor.document, kind)
        if source == self._scan_source_text:
            return None
        self._scan_source_text = source
        self._scan_source_revision += 1
        return self._local_delta(
            self._editor.revision,
            include_scan_workspace=True,
        )

    def generate_scan_source(self, source: str | None = None) -> None:
        """Run trusted local Scan-tab Python on the controller worker."""

        self._require_scan_operation_available()
        if source is not None:
            submitted = str(source)
            if submitted != self._scan_source_text:
                self._scan_source_text = submitted
                self._scan_source_revision += 1
        document = self._editor.document
        if not document.scan_parameters:
            raise ValueError("bind at least one field to a scan parameter first")
        program = self._scan_source_text
        self._scan_operation_generation += 1
        operation = self._scan_operation_generation
        token = (operation, scan_slot_schema(document), program)
        self._scan_busy_operation = "generate"
        self._scan_diagnostic = ""
        self._submit(
            "scan-generate",
            token,
            lambda: execute_scan_program(document, program),
        )

    def load_scan_program(self, path: str | Path) -> None:
        """Apply the formal Load Program button's extension-specific semantics."""

        resolved = Path(path).expanduser().resolve()
        if resolved.suffix.lower() in (".npy", ".csv", ".json"):
            self.load_scan_array(resolved)
            return
        self._require_scan_operation_available()
        self._scan_operation_generation += 1
        operation = self._scan_operation_generation
        self._scan_busy_operation = "load-program"
        self._scan_diagnostic = ""
        self._submit(
            "scan-load-program",
            (operation, resolved),
            lambda: load_scan_program_source(resolved),
        )

    def load_scan_array(self, path: str | Path) -> None:
        """Load and strictly freeze the existing Edit-panel array source."""

        self._require_scan_operation_available()
        document = self._editor.document
        if not document.scan_parameters:
            raise ValueError("bind at least one field to a scan parameter first")
        resolved = Path(path).expanduser().resolve()
        self._scan_operation_generation += 1
        operation = self._scan_operation_generation
        token = (operation, scan_slot_schema(document), resolved)
        self._scan_busy_operation = "load-array"
        self._scan_diagnostic = ""
        self._submit(
            "scan-load-array",
            token,
            lambda: read_scan_array(document, resolved),
        )

    def select_scan_source(
        self,
        source: ScanCandidateSource | str,
    ) -> PulseEditorLocalDelta | None:
        """Apply the switch-selected candidate without fallback or reshaping."""

        self._require_scan_operation_available()
        normalized = str(source).strip().lower()
        if normalized not in ("generated", "loaded"):
            raise ValueError("scan source must be generated or loaded")
        candidate = (
            self._scan_generated if normalized == "generated" else self._scan_loaded
        )
        if candidate is None:
            raise ValueError(
                "no loaded scan array is available"
                if normalized == "loaded"
                else "no generated scan table is available; Run code first"
            )
        document = self._editor.document
        base_revision = self._editor.revision
        previous_selected = self._scan_selected
        self._scan_selected = normalized  # type: ignore[assignment]
        if candidate.schema != scan_slot_schema(self._editor.document):
            self._clear_active_scan_table()
            self._scan_diagnostic = (
                "Scan table is stale because the parameter schema changed; "
                + ("reload the array" if normalized == "loaded" else "Run code again")
            )
            return self._local_delta(
                base_revision,
                include_scan_workspace=True,
            )
        self._commit_scan_candidate(candidate, self._scan_selected)
        if (
            self._editor.revision == base_revision
            and previous_selected == self._scan_selected
        ):
            return None
        return self._local_delta(
            base_revision,
            include_scan_workspace=True,
        )

    def save_scan_array(self, path: str | Path) -> None:
        """Export exactly the selected compatible frozen candidate on a worker."""

        self._require_scan_operation_available()
        candidate = self._selected_scan_candidate()
        if candidate is None:
            raise ValueError("no selected scan table is available to save")
        if candidate.schema != scan_slot_schema(self._editor.document):
            raise ValueError("selected scan table is stale; regenerate or reload it")
        resolved = Path(path).expanduser().resolve()
        self._scan_operation_generation += 1
        operation = self._scan_operation_generation
        self._scan_busy_operation = "save-array"
        self._scan_diagnostic = ""
        self._submit(
            "scan-save-array",
            (operation, resolved),
            lambda: write_scan_array(candidate.table, resolved),
        )

    def new_document(self) -> None:
        self._require_authoring_available()
        current = self._editor.document
        target = current.target
        time_step_ns = current.time_step_ns
        if self._descriptor is not None:
            target = self._descriptor.target
            time_step_ns = self._descriptor.time_step_ns
        self._editor = PulseEditorSession.new(
            target, time_step_ns=time_step_ns
        )
        self._editor_generation += 1
        self._reset_scan_workspace(self._editor.document)
        self._invalidate_document_preview()

    def open_path(self, path: str | Path) -> None:
        self._require_not_closing()
        if (
            self._save_inflight
            or self._load_inflight
            or self._connect_inflight
            or self._run_busy()
        ):
            raise RuntimeError("cannot load a pulse while another operation is active")
        resolved = Path(path).expanduser().resolve()
        self._load_inflight = True
        self._submit("load", resolved, lambda: PulseEditorSession.load(resolved))

    def save(self, path: str | Path | None = None, *, overwrite: bool = False) -> None:
        self._require_not_closing()
        if self._save_inflight or self._load_inflight:
            raise RuntimeError("a pulse file operation is already active")
        session = self._editor
        token = (self._editor_generation, session)
        self._save_inflight = True
        self._submit(
            "save", token, lambda: session.save(path, overwrite=overwrite)
        )

    def save_preview(self, path: str | Path) -> None:
        """Export the same frozen drawing as Preview, at the formal 600-DPI setting."""

        self._require_not_closing()
        if self._preview_export_inflight:
            raise RuntimeError("a Pulse preview export is already active")
        target = Path(path).expanduser().resolve()
        if not target.suffix:
            target = target.with_suffix(".png")
        revision, document = self._editor.snapshot()
        generation = self._editor_generation
        include_off = self._preview_include_off
        size = self._preview_size
        self._preview_export_inflight = True
        self._preview_notice = ""

        def export() -> Path:
            from zlc_frontend.matplotlib_render import render_pulse_timeline_png

            timeline = project_pulse_preview(document)
            kwargs = pulse_timeline_render_kwargs(
                timeline,
                include_off_rows=include_off,
                size=size,
            )
            image = render_pulse_timeline_png(**kwargs, export=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_name: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=target.parent,
                    delete=False,
                ) as stream:
                    temporary_name = stream.name
                    stream.write(image)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, target)
                temporary_name = None
            finally:
                if temporary_name is not None:
                    try:
                        Path(temporary_name).unlink()
                    except FileNotFoundError:
                        pass
            return target

        self._submit(
            "save-preview",
            (generation, revision, target),
            export,
        )

    def set_preview_include_off(self, include: bool) -> None:
        if type(include) is not bool:
            raise TypeError("include-off preview state must be bool")
        if include == self._preview_include_off and self._preview_size is None:
            return
        self._preview_include_off = include
        # This is the formal UI's established behaviour: changing the row set
        # returns Size to its content-derived default, while preserving the
        # current time viewport.
        self._preview_size = None
        self._preview_presentation_revision += 1
        self.request_preview()

    def set_preview_size(self, size: str) -> None:
        from zlc_data.panel_size import PANEL_SIZES

        normalized = str(size)
        if normalized not in PANEL_SIZES:
            raise ValueError(f"unknown pulse preview size {normalized!r}")
        if normalized == self._preview_size:
            return
        self._preview_size = normalized
        self._preview_presentation_revision += 1
        self.request_preview()

    def reset_preview_size(self) -> None:
        """Clear the formal Preview tab's transient manual size selection."""

        if self._preview_size is None:
            return
        self._preview_size = None
        self._preview_presentation_revision += 1
        self.request_preview()

    def set_preview_pixel_ratio(self, ratio: int | float) -> None:
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise TypeError("preview pixel ratio must be numeric")
        normalized = float(ratio)
        if not math.isfinite(normalized) or normalized <= 0:
            raise ValueError("preview pixel ratio must be finite and positive")
        if normalized == self._preview_pixel_ratio:
            return
        self._preview_pixel_ratio = normalized
        self._preview_presentation_revision += 1
        self.request_preview()

    def commit_preview_view(
        self,
        x_limits: tuple[float, float] | None,
        *,
        presentation_revision: int,
    ) -> None:
        if (
            isinstance(presentation_revision, bool)
            or not isinstance(presentation_revision, int)
            or presentation_revision <= self._preview_presentation_revision
        ):
            raise ValueError("preview presentation revision must advance monotonically")
        normalized = None
        if x_limits is not None:
            values = tuple(float(value) for value in x_limits)
            if (
                len(values) != 2
                or not all(math.isfinite(value) for value in values)
                or values[1] <= values[0]
            ):
                raise ValueError("preview x limits must be one finite increasing pair")
            normalized = values
        self._preview_x_limits = normalized
        self._preview_presentation_revision = presentation_revision
        self.request_preview()

    def request_preview(self) -> None:
        if self._close_requested:
            return
        self._preview_notice = ""
        revision, document = self._editor.snapshot()
        document_generation = self._editor_generation
        presentation_revision = self._preview_presentation_revision
        token = (document_generation, revision, presentation_revision)
        self._preview_requested = token
        rendered = self._rendered_preview
        # A presentation-only request (live pan/zoom, DPR, size, row filter)
        # must not erase the last completed front.  The worker is intentionally
        # latest-only, but a continuously moving pointer can advance faster
        # than Agg; clearing here made every completed intermediate revision
        # invisible until motion stopped.  A changed DOCUMENT identity remains
        # strict and clears immediately.
        if rendered is None or (
            rendered.document_generation,
            rendered.editor_revision,
        ) != (document_generation, revision):
            self._preview = None
            self._rendered_preview = None
            self._preview_error = ""
        if self._preview_inflight is not None:
            return
        self._preview_inflight = token

        include_off = self._preview_include_off
        size = self._preview_size
        pixel_ratio = self._preview_pixel_ratio
        x_limits = self._preview_x_limits
        document_id = f"{self._preview_document_namespace}-g{document_generation}"

        def render_preview() -> RenderedPulsePreview:
            from zlc_frontend.matplotlib_render import render_pulse_timeline_panel
            from zlc_frontend.render import DocumentInputIdentity

            timeline = project_pulse_preview(document)
            kwargs = pulse_timeline_render_kwargs(
                timeline,
                include_off_rows=include_off,
                size=size,
            )
            effective_size = str(kwargs["size"])
            document_input = DocumentInputIdentity(
                document_id,
                revision,
                timeline.fingerprint,
            )
            raster, payload = render_pulse_timeline_panel(
                **kwargs,
                document_input=document_input,
                display_revision=presentation_revision,
                pixel_ratio=pixel_ratio,
                x_limits=x_limits,
            )
            return RenderedPulsePreview(
                document_generation=document_generation,
                editor_revision=revision,
                presentation_revision=presentation_revision,
                timeline=timeline,
                raster=raster,
                payload=payload,
                size=effective_size,
                status=pulse_preview_status(
                    timeline,
                    include_off_rows=include_off,
                ),
                pixel_ratio=pixel_ratio,
            )

        self._submit(
            "preview", token, render_preview
        )

    def connect(self, mode: str, endpoint: str = "") -> None:
        if self._connection_factory is None:
            raise RuntimeError("this Pulse editor cannot change its installation")
        if self._connect_inflight or self._close_requested:
            return
        if self._load_inflight or self._save_inflight or self._run_busy():
            raise RuntimeError("connection requires an idle Pulse editor")
        normalized_mode = str(mode).strip().lower()
        if normalized_mode == "remote":
            host, port = parse_remote_endpoint(endpoint)
            endpoint_label = f"{host}:{port}"
        elif normalized_mode == "virtual":
            host, port = None, None
            endpoint_label = "local virtual"
        elif normalized_mode == "offline":
            host, port, endpoint_label = None, None, ""
        else:
            raise ValueError("Pulse connection mode must be virtual, remote, or offline")

        current_ports = self._display_ports_for(
            self._active_manifest_mode,
            (
                self._descriptor.manifest.with_target(self._editor.document.target)
                if self._descriptor is not None
                else self._authoring_manifest.with_target(self._editor.document.target)
            ),
        )
        self._display_ports_by_mode[self._active_manifest_mode] = current_ports
        request = (normalized_mode, host, port, endpoint_label)

        if self._owned_connection is not None:
            if (
                normalized_mode == self._active_manifest_mode
                and endpoint_label == self._connection_endpoint
            ):
                return
            self._pending_connection_request = request
            self._connect_inflight = True
            self._owned_close_inflight = True
            self._connection_state = "switching"
            self._connection_mode = normalized_mode
            self._connection_endpoint = endpoint_label
            connection = self._owned_connection
            self._detach_pulse_authority()
            self._submit("switch-close", connection, connection.close)
            return

        if normalized_mode == "offline":
            self._activate_offline()
            return
        self._begin_connection(request)

    def _begin_connection(
        self,
        request: tuple[str, str | None, int | None, str],
    ) -> None:
        normalized_mode, host, port, endpoint_label = request
        connector = self._connection_factory
        if connector is None:
            raise RuntimeError("this Pulse editor cannot establish an installation")
        required_document = self._editor.document
        self._connect_inflight = True
        self._connection_state = "connecting"
        self._connection_mode = normalized_mode
        self._connection_endpoint = endpoint_label

        def connect_owned():
            connection = connector(
                normalized_mode,
                host,
                port,
                required_document,
            )
            if not isinstance(connection, OwnedPulseConnection):
                raise TypeError("Pulse connection factory returned the wrong type")
            return connection

        self._submit(
            "connect",
            (
                normalized_mode,
                host,
                port,
                self._editor_generation,
                self._editor.revision,
            ),
            connect_owned,
        )

    def _activate_offline(self) -> None:
        self._detach_pulse_authority()
        self._active_manifest_mode = "offline"
        self._connection_state = "offline"
        self._connection_mode = "offline"
        self._connection_endpoint = ""
        self._diagnostic = ""

    def _detach_pulse_authority(self) -> None:
        """Stop all reads through an authority before its asynchronous close starts."""

        self._owned_connection = None
        self._pulse = None
        self._descriptor = None
        self._run_snapshot = None
        self._applied_snapshot = None
        self._applied_authoring_cache_source = None
        self._applied_authoring_cache_fingerprint = None
        self._run_generation = None
        self._run_revision = None
        self._scan_progress = None

    def _applied_authoring_fingerprint(
        self,
        snapshot: AppliedPulseSnapshot,
    ) -> str:
        """Resolve execution-time API values once for one applied identity."""

        if snapshot is not self._applied_authoring_cache_source:
            self._applied_authoring_cache_source = snapshot
            self._applied_authoring_cache_fingerprint = (
                _applied_authoring_document(snapshot).fingerprint
            )
        fingerprint = self._applied_authoring_cache_fingerprint
        if fingerprint is None:
            raise RuntimeError("applied Pulse authoring fingerprint was not cached")
        return fingerprint

    def start(
        self,
        form: PulseExecutionForm,
        *,
        api_values: dict[str, int | float] | None = None,
        nominal_reference: bool = False,
        scan_sweep_count: int = 1,
    ) -> None:
        if not isinstance(form, PulseExecutionForm):
            raise TypeError("form must be PulseExecutionForm")
        if form is PulseExecutionForm.STATIC_REFERENCE_POINT:
            raise ValueError("STATIC_REFERENCE_POINT is preview-only")
        if self._pulse is None:
            raise RuntimeError("Pulse editor is offline")
        if self._load_inflight or self._connect_inflight or self._close_requested:
            raise RuntimeError("Pulse execution is unavailable")
        document = self._editor.document
        if self._descriptor is not None:
            rebound = bind_pulse_document_target(document, self._descriptor.target)
            rebound = restrict_pulse_document_to_manifest(
                rebound,
                self._descriptor.manifest,
            )
            validate_pulse_document_clock_grid(rebound, self._descriptor.clock_hz)
            if rebound != document:
                self.replace_document(rebound)
                document = rebound
        if form in (
            PulseExecutionForm.STATIC_ONCE,
            PulseExecutionForm.CONTINUOUS_MONITOR,
        ) and document.scan_parameters:
            if not nominal_reference:
                raise ValueError("explicit nominal scan reference is required")
            document = replace(
                document,
                scan_parameters=(),
                scan_table=None,
                scan_recipe=None,
            )
        supplied = (
            {
                parameter.parameter_id: document.field_value(parameter.field)[0]
                for parameter in document.api_parameters
            }
            if api_values is None
            else dict(api_values)
        )
        ordered_values = tuple(
            (parameter.parameter_id, supplied[parameter.parameter_id])
            for parameter in document.api_parameters
            if parameter.parameter_id in supplied
        )
        expected = {parameter.parameter_id for parameter in document.api_parameters}
        if set(supplied) != expected:
            raise ValueError(
                "Pulse execution requires exactly every API parameter; "
                f"missing={tuple(sorted(expected - set(supplied)))}, "
                f"unknown={tuple(sorted(set(supplied) - expected))}"
            )
        pending = (
            document,
            form,
            ordered_values,
            scan_sweep_count,
            self._editor_generation,
            self._editor.revision,
        )
        self._held_scan_source = None
        self._held_scan_index = None
        self._pending_hold = None
        self._replace_run(pending)

    def _replace_run(
        self,
        pending: tuple[
            PulseDocument,
            PulseExecutionForm,
            tuple[tuple[str, int | float], ...],
            int,
            int | None,
            int | None,
        ],
    ) -> None:
        pulse = self._pulse
        if pulse is None:
            raise RuntimeError("Pulse editor is offline")
        active = self._pulse.observe_active()
        local_not_reaped = self._handle is not None and not self._owner_reaped
        if (
            self._run_starting
            or local_not_reaped
            or (active is not None and not active.run.state.terminal)
        ):
            # The formal On Pulse action is a replacement: request SAFE first,
            # then admit the frozen new intent only after the previous owner is
            # terminal.  No pulse edge is scheduled by this GUI state machine.
            self._pending_start = pending
            self._diagnostic = ""
            if self._run_starting:
                self._cancel_when_started = True
            else:
                self._pulse.cancel_active("PulseGUI replacing the applied pulse")
            return
        self._begin_start(pending)

    def _begin_start(
        self,
        pending: tuple[
            PulseDocument,
            PulseExecutionForm,
            tuple[tuple[str, int | float], ...],
            int,
            int | None,
            int | None,
        ],
    ) -> None:
        document, form, api_values, sweep_count, editor_generation, revision = pending
        pulse = self._pulse
        if pulse is None:
            raise RuntimeError("Pulse editor is offline")
        self._operation_generation += 1
        generation = self._operation_generation
        self._run_starting = True
        self._cancel_when_started = False
        self._owner_reaped = False
        self._handle = None
        self._run_snapshot = None
        self._applied_snapshot = None
        self._applied_authoring_cache_source = None
        self._applied_authoring_cache_fingerprint = None
        self._run_generation = editor_generation
        self._run_revision = revision

        def start_run() -> RunHandle:
            return pulse.start(
                pulse.request(
                    document,
                    form,
                    api_values=dict(api_values),
                    scan_sweep_count=sweep_count,
                )
            )

        self._submit("start", (generation, revision), start_run)

    def request_scan_progress(self) -> None:
        """Schedule one read-only observation; never perform transport I/O on Qt."""

        pulse = self._pulse
        run = self._run_snapshot
        if (
            pulse is None
            or self._scan_progress_inflight
            or self._close_requested
        ):
            return
        if run is None or run.state.terminal:
            self._scan_progress = None
            return
        self._scan_progress_inflight = True
        self._submit(
            "scan-progress",
            run.run_id.value,
            pulse.observe_scan_progress,
        )

    def _active_scan_point(self) -> tuple[PulseDocument, int]:
        pulse = self._pulse
        progress = self._scan_progress
        if pulse is None:
            raise RuntimeError("No installation is attached.")
        active = pulse.observe_active()
        if active is None or active.run.state.terminal:
            raise RuntimeError("No running scan point is available to hold.")
        if (
            progress is None
            or progress.run_id != active.run.run_id.value
            or not progress.available
        ):
            reason = (
                "scan progress has not been observed yet"
                if progress is None
                else progress.unavailable_reason
            )
            raise RuntimeError(
                "The current hardware scan point is unavailable"
                + (f": {reason}" if reason else ".")
            )
        applied = active.applied
        if applied is None:
            raise RuntimeError("The running scan has not reached hardware prepare.")
        source = _applied_authoring_document(applied)
        table = source.scan_table
        if table is None or not table.rows:
            raise RuntimeError("No scan table is applied on the sequencer.")
        assert progress.current_point_index is not None
        return source, progress.current_point_index % len(table.rows)

    def _queue_held_scan_point(self, source: PulseDocument, index: int) -> None:
        table = source.scan_table
        if table is None or not table.rows:
            raise RuntimeError("No scan table to hold -- run a scan first.")
        point = max(0, min(int(index), len(table.rows) - 1))
        if self._held_scan_source is source and self._held_scan_index == point:
            return
        frozen = resolve_scan_point(source, point)
        api_values = tuple(
            (
                parameter.parameter_id,
                frozen.field_value(parameter.field)[0],
            )
            for parameter in frozen.api_parameters
        )
        self._pending_hold = (source, point, frozen.fingerprint)
        self._replace_run(
            (
                frozen,
                PulseExecutionForm.CONTINUOUS_MONITOR,
                api_values,
                1,
                None,
                None,
            )
        )

    def hold_scan_point(self) -> None:
        source, index = self._active_scan_point()
        self._queue_held_scan_point(source, index)

    def step_scan_point(self, delta: int) -> None:
        if isinstance(delta, bool) or not isinstance(delta, int):
            raise TypeError("scan step delta must be an integer")
        if delta == 0:
            return
        if self._held_scan_source is not None and self._held_scan_index is not None:
            source, index = self._held_scan_source, self._held_scan_index
        elif self._pending_hold is not None:
            source, index, _fingerprint = self._pending_hold
        else:
            source, index = self._active_scan_point()
        self._queue_held_scan_point(source, index + delta)

    def cancel(self, reason: str = "PulseGUI Stop Pulse") -> None:
        self._pending_start = None
        self._pending_hold = None
        self._held_scan_source = None
        self._held_scan_index = None
        self._scan_progress = None
        if self._run_starting and self._handle is None:
            self._cancel_when_started = True
        pulse = self._pulse
        if pulse is not None:
            pulse.cancel_active(str(reason))
        handle = self._handle
        run = self._run_snapshot
        if handle is not None and (run is None or not run.state.terminal):
            handle.cancel(str(reason))

    def sync_applied(self) -> int:
        """Load the exact last applied authoring intent and effective API values."""

        pulse = self._pulse
        if pulse is None:
            raise RuntimeError("Offline: nothing to sync from.")
        applied = pulse.snapshot()
        if applied is None:
            raise RuntimeError(
                "The sequencer has no applied pulse yet (nothing was prepared)."
            )
        document = _applied_authoring_document(applied)
        revision = self.replace_document(document)
        self._applied_snapshot = applied
        self._run_generation = self._editor_generation
        self._run_revision = revision
        self._diagnostic = "Synced from device."
        return revision

    def request_close(self) -> None:
        if self._close_complete:
            return
        self._close_requested = True
        self.cancel("PulseGUI is closing")
        self._advance_close()

    def retire_borrowed_authority(self) -> None:
        """Thread-safe retirement requested by the owning Experiment.

        The Experiment runtime owns hardware SAFE during its own shutdown.  This
        path therefore only wakes the controller so its next owner turn detaches
        the borrowed facade, drains local work, and permanently closes.
        """

        if self._owned_connection is not None or self._connection_factory is not None:
            raise RuntimeError("only an Experiment-borrowed Pulse editor may retire here")
        self._borrowed_authority_retire.set()
        self._notify()

    def pump(
        self,
        *,
        force: bool = False,
    ) -> (
        PulseEditorControllerSnapshot
        | PulseRuntimeUpdate
        | PulseScanProgressUpdate
        | None
    ):
        """Consume one owner wake and publish only an actual visible change.

        ``force`` is reserved for a synchronous semantic command whose state
        changed before this owner turn began.  Worker wakes are change-gated,
        so stale completions, equal progress observations, and a coalesced
        replay cannot manufacture an application snapshot.
        """

        before = None if force else self._presentation_state_key()
        before_without_progress = (
            None
            if force
            else self._presentation_state_key(include_scan_progress=False)
        )
        before_full_projection = (
            None if force else self._full_projection_state_key()
        )
        self._apply_borrowed_authority_retirement()
        self._drain_results()
        self._poll_run()
        self._advance_close()
        if before is not None:
            after = self._presentation_state_key()
            if after == before:
                return None
            if (
                before_without_progress
                == self._presentation_state_key(include_scan_progress=False)
            ):
                return PulseScanProgressUpdate(
                    scan_progress=self._scan_progress,
                    applied_snapshot=self._applied_snapshot,
                    held_scan_point=self._held_scan_snapshot(),
                )
            if before_full_projection == self._full_projection_state_key():
                return self.runtime_update()
        return self.snapshot()

    def poll_runtime_change(self) -> PulseRuntimeUpdate | None:
        """Poll an active Run without manufacturing unchanged GUI snapshots.

        This is the narrow compatibility seam for ``RunHandle``, whose current
        API is observed rather than pushed.  Idle editors do no work.  Active
        editors publish only when a runtime fact changes; the 40 ms Qt timer is
        therefore no longer an application-wide snapshot clock.
        """

        # The Experiment owner can retire the borrowed facade from another
        # thread.  Detach it before even deciding whether the compatibility
        # timer has Run work; otherwise a timer already queued in Qt can call
        # observe_active() after Experiment.close().
        self._apply_borrowed_authority_retirement()
        if not self._runtime_poll_required():
            return None
        before = self._runtime_presentation_key()
        self._poll_run()
        self._advance_close()
        if self._runtime_presentation_key() == before:
            return None
        return self.runtime_update()

    def _runtime_poll_required(self) -> bool:
        snapshot = self._run_snapshot
        return bool(
            self._run_starting
            or self._pending_start is not None
            or self._handle is not None
            or self._reap_inflight
            or (snapshot is not None and not snapshot.state.terminal)
            or (self._close_requested and not self._close_complete)
        )

    @property
    def runtime_poll_required(self) -> bool:
        """Return whether the compatibility RunHandle watcher has work.

        Reading this flag does not poll hardware and does not build a GUI
        snapshot.  The Qt owner uses it to keep its timer stopped while idle.
        """

        return self._runtime_poll_required()

    @property
    def scan_progress_poll_required(self) -> bool:
        """Whether a non-terminal applied scan can produce progress facts."""

        run = self._run_snapshot
        applied = self._applied_snapshot
        return bool(
            self._pulse is not None
            and run is not None
            and not run.state.terminal
            and applied is not None
            and applied.source_document.scan_table is not None
        )

    def _runtime_presentation_key(self) -> tuple[object, ...]:
        return (
            self._connection_state,
            self._connection_mode,
            self._connection_endpoint,
            id(self._descriptor),
            self._run_snapshot,
            self._applied_snapshot,
            self._run_generation,
            self._run_revision,
            self._scan_progress,
            self._held_scan_index,
            None
            if self._held_scan_source is None
            else self._held_scan_source.fingerprint,
            self._run_starting,
            self._pending_start is not None,
            self._owner_reaped,
            self._reap_inflight,
            self._save_inflight,
            self._load_inflight,
            self._preview_export_inflight,
            self._diagnostic,
            self._close_requested,
            self._close_complete,
        )

    def _apply_borrowed_authority_retirement(self) -> None:
        if (
            not self._borrowed_authority_retire.is_set()
            or self._borrowed_authority_retired
        ):
            return
        self._borrowed_authority_retired = True
        self._close_requested = True
        self._pending_start = None
        self._pending_hold = None
        self._held_scan_source = None
        self._held_scan_index = None
        self._scan_progress = None
        self._pulse = None
        self._descriptor = None
        if self._handle is None:
            self._run_snapshot = None
            self._applied_snapshot = None
            self._applied_authoring_cache_source = None
            self._applied_authoring_cache_fingerprint = None
            self._run_generation = None
            self._run_revision = None
        self._connection_state = "closing"

    def _submit(self, kind: str, token: object, work) -> None:
        if self._pool_closed:
            raise RuntimeError("Pulse editor worker is closed")
        future = self._pool.submit(work)
        with self._lock:
            self._tracked.add(future)

        def done(completed: Future) -> None:
            with self._lock:
                self._tracked.discard(completed)
                self._results.append((kind, token, completed))
            self._notify()

        future.add_done_callback(done)

    def _drain_results(self) -> None:
        with self._lock:
            pending, self._results = self._results, []
        for kind, token, future in pending:
            if kind == "scan-progress":
                self._scan_progress_inflight = False
                try:
                    progress = future.result()
                    if progress is not None and not isinstance(
                        progress,
                        PulseScanProgress,
                    ):
                        raise TypeError("scan progress reader returned another type")
                except BaseException:
                    progress = None
                run = self._run_snapshot
                if run is not None and run.run_id.value == token:
                    if progress == self._scan_progress:
                        continue
                    self._scan_progress = progress
                continue
            if kind == "scan-generate":
                operation, schema, source = token
                if operation == self._scan_operation_generation:
                    self._scan_busy_operation = None
                try:
                    result = future.result()
                    if not isinstance(result, ScanCandidateResult):
                        raise TypeError("scan worker returned the wrong result type")
                except BaseException as error:
                    if operation == self._scan_operation_generation:
                        self._scan_diagnostic = (
                            f"Scan code error: {type(error).__name__}: {error}"
                        )
                    continue
                if operation != self._scan_operation_generation:
                    continue
                current_schema = scan_slot_schema(self._editor.document)
                if self._close_requested:
                    continue
                if schema != current_schema:
                    self._scan_generated = result.candidate
                    self._scan_selected = "generated"
                    self._scan_diagnostic = (
                        "Generated scan table is stale because the parameter "
                        "schema changed; Run code again"
                    )
                    continue
                if source != self._scan_source_text:
                    self._scan_diagnostic = (
                        "Generated scan table belongs to an older code buffer; "
                        "Run the current code"
                    )
                    continue
                self._scan_generated = result.candidate
                self._scan_selected = "generated"
                self._scan_source_baseline = source
                try:
                    self._commit_scan_candidate(result.candidate, "generated")
                except BaseException as error:
                    self._scan_source_baseline = None
                    self._scan_diagnostic = (
                        f"Apply generated scan failed: {type(error).__name__}: {error}"
                    )
                    continue
                adjusted = result.normalization.adjusted_cells
                self._scan_diagnostic = (
                    f"Generated {len(result.candidate.table.rows)} scan points"
                    + (f"; snapped {adjusted} value(s)" if adjusted else "")
                )
                continue
            if kind == "scan-load-array":
                operation, schema, resolved = token
                if operation == self._scan_operation_generation:
                    self._scan_busy_operation = None
                try:
                    result = future.result()
                    if not isinstance(result, ScanCandidateResult):
                        raise TypeError("scan worker returned the wrong result type")
                except BaseException as error:
                    if operation == self._scan_operation_generation:
                        self._scan_diagnostic = (
                            f"Load scan array failed: {type(error).__name__}: {error}"
                        )
                    continue
                if operation != self._scan_operation_generation:
                    continue
                self._scan_loaded = result.candidate
                self._scan_selected = "loaded"
                if self._close_requested:
                    continue
                if schema != scan_slot_schema(self._editor.document):
                    self._scan_diagnostic = (
                        "Loaded scan array is stale because the parameter schema "
                        "changed; load it again"
                    )
                    continue
                try:
                    self._commit_scan_candidate(result.candidate, "loaded")
                except BaseException as error:
                    self._scan_diagnostic = (
                        f"Apply loaded scan failed: {type(error).__name__}: {error}"
                    )
                    continue
                adjusted = result.normalization.adjusted_cells
                self._scan_diagnostic = (
                    f"Loaded {len(result.candidate.table.rows)} scan points from "
                    f"{Path(resolved).name}"
                    + (f"; snapped {adjusted} value(s)" if adjusted else "")
                )
                continue
            if kind == "scan-load-program":
                operation, resolved = token
                if operation == self._scan_operation_generation:
                    self._scan_busy_operation = None
                try:
                    loaded_path, source = future.result()
                except BaseException as error:
                    if operation == self._scan_operation_generation:
                        self._scan_diagnostic = (
                            f"Load scan program failed: {type(error).__name__}: {error}"
                        )
                    continue
                if operation == self._scan_operation_generation and not self._close_requested:
                    self._scan_source_text = source
                    self._scan_source_baseline = None
                    self._scan_source_revision += 1
                    self._scan_diagnostic = f"Loaded scan program: {loaded_path.name}"
                continue
            if kind == "scan-save-array":
                operation, _resolved = token
                if operation == self._scan_operation_generation:
                    self._scan_busy_operation = None
                try:
                    saved = future.result()
                except BaseException as error:
                    if operation == self._scan_operation_generation:
                        self._scan_diagnostic = (
                            f"Save scan array failed: {type(error).__name__}: {error}"
                        )
                else:
                    if operation == self._scan_operation_generation:
                        self._scan_diagnostic = f"Saved scan array: {saved.name}"
                continue
            if kind == "preview":
                self._preview_inflight = None
                current = (
                    self._editor_generation,
                    self._editor.revision,
                    self._preview_presentation_revision,
                )
                same_document = token[:2] == current[:2]
                if same_document and not self._close_requested:
                    try:
                        rendered = future.result()
                        if not isinstance(rendered, RenderedPulsePreview):
                            raise TypeError(
                                "preview worker returned the wrong result type"
                            )
                    except BaseException as error:
                        # Only the latest requested presentation may replace a
                        # healthy front with an error.  An older presentation
                        # can fail while a newer coalesced request is already
                        # waiting; surfacing that obsolete failure would make
                        # a live drag flicker unavailable.
                        if token == current:
                            self._preview = None
                            self._rendered_preview = None
                            self._preview_revision = token[1]
                            self._preview_generation = token[0]
                            self._preview_error = (
                                f"{type(error).__name__}: {error}"
                            )
                    else:
                        shown = self._rendered_preview
                        shown_revision = (
                            -1
                            if shown is None
                            or (
                                shown.document_generation,
                                shown.editor_revision,
                            ) != token[:2]
                            else shown.presentation_revision
                        )
                        if token[2] > shown_revision:
                            # Same immutable document, monotonic presentation:
                            # publish the completed intermediate frame now and
                            # immediately continue toward _preview_requested.
                            self._rendered_preview = rendered
                            self._preview = rendered.timeline
                            self._preview_revision = token[1]
                            self._preview_generation = token[0]
                            self._preview_error = ""
                if self._preview_requested != token and not self._close_requested:
                    self.request_preview()
                continue
            if kind == "load":
                self._load_inflight = False
                try:
                    loaded = future.result()
                    if self._descriptor is not None:
                        loaded.bind_target(self._descriptor.target)
                        loaded.replace_document(
                            restrict_pulse_document_to_manifest(
                                loaded.document,
                                self._descriptor.manifest,
                            )
                        )
                        validate_pulse_document_clock_grid(
                            loaded.document,
                            self._descriptor.clock_hz,
                        )
                except BaseException as error:
                    self._diagnostic = f"Open failed: {type(error).__name__}: {error}"
                else:
                    if not self._close_requested and not self._run_busy():
                        self._editor = loaded
                        if self._descriptor is None:
                            if (
                                loaded.document.target.abi_fingerprint
                                == self._authoring_manifest.target.abi_fingerprint
                            ):
                                self._authoring_manifest = (
                                    self._authoring_manifest.with_target(
                                        loaded.document.target
                                    )
                                )
                            else:
                                self._authoring_manifest = (
                                    pulse_target_manifest_from_lanes(
                                        loaded.document.target
                                    )
                                )
                        self._display_ports_by_mode["offline"] = (
                            loaded.document.visible_ports
                        )
                        self._editor_generation += 1
                        self._reset_scan_workspace(loaded.document)
                        self._diagnostic = f"Opened {loaded.path}"
                        self._invalidate_document_preview()
                continue
            if kind == "save":
                self._save_inflight = False
                try:
                    saved = future.result()
                except BaseException as error:
                    self._diagnostic = f"Save failed: {type(error).__name__}: {error}"
                else:
                    if token == (self._editor_generation, self._editor):
                        self._diagnostic = f"Saved {saved}"
                continue
            if kind == "save-preview":
                self._preview_export_inflight = False
                try:
                    saved = future.result()
                except BaseException as error:
                    self._diagnostic = (
                        f"Save figure failed: {type(error).__name__}: {error}"
                    )
                else:
                    self._preview_notice = f"Saved figure: {saved.name}"
                continue
            if kind == "connect":
                self._connect_inflight = False
                try:
                    connection = future.result()
                except BaseException as error:
                    self._connection_state = "offline"
                    self._diagnostic = (
                        f"Connection failed: {type(error).__name__}: {error}"
                    )
                    continue
                if self._pulse is not None or self._owned_connection is not None:
                    self._submit(
                        "discard-connection",
                        connection,
                        connection.close,
                    )
                    continue
                descriptor = connection.descriptor
                try:
                    rebound = bind_pulse_document_target(
                        self._editor.document,
                        descriptor.target,
                    )
                    rebound = restrict_pulse_document_to_manifest(
                        rebound,
                        descriptor.manifest,
                    )
                    validate_pulse_document_clock_grid(
                        rebound,
                        descriptor.clock_hz,
                    )
                except BaseException as error:
                    self._retire_claimed(
                        connection,
                        f"Remote target rejected: {type(error).__name__}: {error}",
                    )
                    continue
                previous = self._editor.document
                rebound = self._invalidate_changed_scan_schema(previous, rebound)
                self._editor.replace_document(rebound)
                self._observe_document_schema_change(previous, rebound)
                self._owned_connection = connection
                self._pulse = connection.pulse
                self._descriptor = descriptor
                self._active_manifest_mode = str(token[0])
                if self._active_manifest_mode not in self._display_ports_by_mode:
                    self._display_ports_by_mode[self._active_manifest_mode] = (
                        descriptor.manifest.available_port_keys
                    )
                self._connection_state = "ready"
                self._diagnostic = ""
                self._invalidate_document_preview()
                continue
            if kind == "switch-close":
                self._owned_close_inflight = False
                request = self._pending_connection_request
                self._pending_connection_request = None
                close_error: BaseException | None = None
                try:
                    future.result()
                except BaseException as error:
                    close_error = error
                self._activate_offline()
                self._connect_inflight = False
                if (
                    request is not None
                    and request[0] != "offline"
                    and not self._close_requested
                ):
                    self._begin_connection(request)
                elif close_error is not None:
                    self._diagnostic = (
                        "Previous Pulse connection close failed: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
                continue
            if kind == "discard-connection":
                try:
                    future.result()
                except BaseException as error:
                    self._diagnostic = f"Discarded connection close failed: {error}"
                continue
            if kind == "start":
                generation, revision = token
                self._run_starting = False
                try:
                    handle = future.result()
                except BaseException as error:
                    if generation == self._operation_generation:
                        self._owner_reaped = True
                        self._handle = None
                        self._pending_hold = None
                        self._run_generation = None
                        self._run_revision = None
                        self._diagnostic = (
                            f"Pulse start failed: {type(error).__name__}: {error}"
                        )
                        self._maybe_start_pending()
                    continue
                if generation != self._operation_generation:
                    self._submit(
                        "discard-run",
                        handle,
                        lambda: _cancel_and_wait_detached_run(handle),
                    )
                    continue
                self._handle = handle
                self._run_snapshot = handle.snapshot()
                if (
                    (self._close_requested or self._cancel_when_started)
                    and not self._borrowed_authority_retired
                ):
                    handle.cancel("PulseGUI stop requested before admission returned")
                continue
            if kind == "discard-run":
                try:
                    future.result()
                except BaseException as error:
                    self._diagnostic = f"Stale Pulse owner reap failed: {error}"
                continue
            if kind == "reap":
                self._reap_inflight = False
                try:
                    future.result()
                except BaseException as error:
                    self._diagnostic = f"Pulse owner reap failed: {error}"
                else:
                    self._owner_reaped = True
                    self._handle = None
                    self._maybe_start_pending()
                continue
            if kind == "close-owned":
                self._owned_close_inflight = False
                try:
                    future.result()
                except BaseException as error:
                    self._diagnostic = f"Remote installation close failed: {error}"
                continue

    def _reset_scan_workspace(self, document: PulseDocument) -> None:
        # Any result still queued for a previous document becomes detached.
        self._scan_operation_generation += 1
        self._scan_auto_template = default_scan_program(document)
        self._scan_generated = candidate_from_document(document)
        self._scan_loaded = None
        self._scan_selected = "generated"
        if document.scan_recipe is not None:
            self._scan_source_text = document.scan_recipe.source
            self._scan_source_baseline = document.scan_recipe.source
        else:
            self._scan_source_text = self._scan_auto_template
            self._scan_source_baseline = self._scan_auto_template
        self._scan_source_revision += 1
        self._scan_busy_operation = None
        self._scan_diagnostic = ""

    def _scan_workspace_snapshot(
        self,
        document: PulseDocument,
    ) -> ScanWorkspaceSnapshot:
        scan_field_state = tuple(
            (parameter, document.field_value(parameter.field))
            for parameter in document.scan_parameters
        )
        api_field_state = tuple(
            (parameter, document.field_value(parameter.field))
            for parameter in document.api_parameters
        )
        cache_key = (
            self._scan_source_revision,
            self._scan_source_baseline == self._scan_source_text,
            self._scan_selected,
            scan_field_state,
            api_field_state,
            tuple(period.period_id for period in document.periods),
            document.time_step_ns,
            self._scan_busy_operation,
            self._scan_diagnostic,
        )
        if (
            cache_key == self._scan_workspace_cache_key
            and self._scan_workspace_cache is not None
            and self._scan_workspace_cache_generated is self._scan_generated
            and self._scan_workspace_cache_loaded is self._scan_loaded
        ):
            return self._scan_workspace_cache
        generated = candidate_snapshot(self._scan_generated, document)
        loaded = candidate_snapshot(self._scan_loaded, document)
        selected = generated if self._scan_selected == "generated" else loaded
        selected_table = None if selected is None else selected.table
        compatible = selected is not None and selected.compatible
        table_stale = selected is not None and not compatible
        snapshot = ScanWorkspaceSnapshot(
            source_text=self._scan_source_text,
            source_revision=self._scan_source_revision,
            default_template=self._scan_auto_template,
            source_dirty=(
                self._scan_source_baseline is None
                or self._scan_source_text != self._scan_source_baseline
            ),
            selected_source=self._scan_selected,
            generated=generated,
            loaded=loaded,
            selected_table=selected_table,
            selected_compatible=compatible,
            loaded_path=(None if loaded is None else loaded.origin_path),
            table_text=format_scan_table(
                selected_table,
                column_names=(
                    None
                    if table_stale
                    else tuple(
                        spec.name for spec in scan_column_specs(document)
                    )
                ),
                stale=table_stale,
            ),
            slots_text=format_scan_slots(document),
            busy_operation=self._scan_busy_operation,
            diagnostic=self._scan_diagnostic,
        )
        self._scan_workspace_cache_key = cache_key
        self._scan_workspace_cache = snapshot
        self._scan_workspace_cache_generated = self._scan_generated
        self._scan_workspace_cache_loaded = self._scan_loaded
        return snapshot

    def _held_scan_snapshot(
        self,
    ) -> tuple[int, int, tuple[tuple[str, int | float], ...]] | None:
        source = self._held_scan_source
        index = self._held_scan_index
        if source is None or index is None or source.scan_table is None:
            return None
        table = source.scan_table
        if not 0 <= index < len(table.rows):
            return None
        display_names = tuple(spec.name for spec in scan_column_specs(source))
        return (
            index,
            len(table.rows),
            tuple(zip(display_names, table.rows[index], strict=True)),
        )

    def _invalidate_changed_scan_schema(
        self,
        previous: PulseDocument,
        candidate: PulseDocument,
    ) -> PulseDocument:
        if scan_slot_schema(previous) == scan_slot_schema(candidate):
            return candidate
        if candidate.scan_table is None and candidate.scan_recipe is None:
            return candidate
        return replace(candidate, scan_table=None, scan_recipe=None)

    def _observe_document_schema_change(
        self,
        previous: PulseDocument,
        document: PulseDocument,
    ) -> None:
        if scan_slot_schema(previous) == scan_slot_schema(document):
            return
        old_auto_template = self._scan_auto_template
        source_was_auto = (
            self._scan_source_text == old_auto_template
            and self._scan_source_baseline == old_auto_template
        )
        self._scan_auto_template = default_scan_program(document)
        if source_was_auto:
            self._scan_source_text = self._scan_auto_template
            self._scan_source_revision += 1
        self._scan_source_baseline = (
            self._scan_auto_template
            if source_was_auto and self._scan_generated is None
            else None
        )
        if self._scan_generated is not None or self._scan_loaded is not None:
            self._scan_diagnostic = (
                "Scan parameters changed; existing generated and loaded tables "
                "are stale and were not reshaped"
            )

    def _selected_scan_candidate(self) -> ScanWorkspaceCandidate | None:
        return (
            self._scan_generated
            if self._scan_selected == "generated"
            else self._scan_loaded
        )

    def _commit_scan_candidate(
        self,
        candidate: ScanWorkspaceCandidate,
        source: ScanCandidateSource,
    ) -> None:
        committed = commit_scan_candidate(self._editor.document, candidate, source)
        self.replace_document(committed)

    def _clear_active_scan_table(self) -> None:
        document = self._editor.document
        if document.scan_table is None and document.scan_recipe is None:
            return
        self.replace_document(replace(document, scan_table=None, scan_recipe=None))

    def _require_scan_operation_available(self) -> None:
        self._require_authoring_available()
        if self._save_inflight:
            raise RuntimeError("scan file operations require the Pulse file owner to be idle")
        if self._scan_busy_operation is not None:
            raise RuntimeError(
                f"scan workspace operation {self._scan_busy_operation!r} is active"
            )

    def _retire_claimed(
        self,
        connection: OwnedPulseConnection,
        diagnostic: str,
    ) -> None:
        # A rejected connection is already unusable.  Remove it from every
        # read path before its asynchronous close begins; a failed close is a
        # fact about that connection, not process-global state.
        self._detach_pulse_authority()
        self._connection_state = "offline"
        self._diagnostic = diagnostic
        self._owned_close_inflight = True
        self._submit("close-owned", connection, connection.close)

    def _run_busy(self) -> bool:
        if self._run_starting or self._pending_start is not None:
            return True
        if self._run_snapshot is not None and not self._run_snapshot.state.terminal:
            return True
        # ``_poll_run`` owns the sole RunSnapshot read for an observation turn.
        # A retained local handle remains busy until its owner has been reaped,
        # including the terminal-but-not-yet-joined interval.
        return self._handle is not None and not self._owner_reaped

    def _require_authoring_available(self) -> None:
        self._require_not_closing()
        if self._load_inflight:
            raise RuntimeError("Pulse authoring is unavailable while Open is active")
        if self._connect_inflight:
            raise RuntimeError("Pulse authoring is unavailable while Connect is active")

    def _require_not_closing(self) -> None:
        if self._close_requested or self._close_complete:
            raise RuntimeError("Pulse editor is closing")

    def _poll_run(self) -> None:
        pulse = self._pulse
        observation = None if pulse is None else pulse.observe_active()
        handle = self._handle
        local_run = (
            observation.run
            if (
                observation is not None
                and handle is not None
                and observation.run.run_id == handle.run_id
            )
            else None
        )
        if observation is not None:
            if (
                self._scan_progress is not None
                and self._scan_progress.run_id != observation.run.run_id.value
            ):
                self._scan_progress = None
            self._run_snapshot = observation.run
            self._applied_snapshot = observation.applied
            if observation.applied is None:
                self._applied_authoring_cache_source = None
                self._applied_authoring_cache_fingerprint = None
                self._run_generation = None
                self._run_revision = None
            else:
                if (
                    self._applied_authoring_fingerprint(observation.applied)
                    == self._editor.document.fingerprint
                ):
                    self._run_generation = self._editor_generation
                    self._run_revision = self._editor.revision
                else:
                    self._run_generation = None
                    self._run_revision = None
                pending_hold = self._pending_hold
                if (
                    pending_hold is not None
                    and observation.applied.source_document.fingerprint
                    == pending_hold[2]
                ):
                    if observation.run.state.terminal:
                        self._pending_hold = None
                        self._held_scan_source = None
                        self._held_scan_index = None
                    else:
                        self._held_scan_source = pending_hold[0]
                        self._held_scan_index = pending_hold[1]
                        self._pending_hold = None
                elif (
                    self._held_scan_source is not None
                    and observation.run.state.terminal
                    and not observation.request.document.scan_parameters
                ):
                    self._held_scan_source = None
                    self._held_scan_index = None
        else:
            self._scan_progress = None
        if handle is not None and observation is None:
            local_run = handle.snapshot()
            self._run_snapshot = local_run
        elif handle is not None and local_run is None:
            # Another application owner may have replaced the active Run.  The
            # GUI still has to reap its own handle, so this is the only case
            # where two distinct handles are observed in one turn.
            local_run = handle.snapshot()
        if (
            handle is not None
            and local_run is not None
            and local_run.state.terminal
            and not self._owner_reaped
            and not self._reap_inflight
        ):
            self._reap_inflight = True
            self._submit("reap", self._operation_generation, handle.wait)
        self._maybe_start_pending()

    def _maybe_start_pending(self) -> None:
        pending = self._pending_start
        if pending is None or self._close_requested or self._run_starting:
            return
        if self._handle is not None and not self._owner_reaped:
            return
        pulse = self._pulse
        if pulse is None:
            self._pending_start = None
            self._diagnostic = "Pulse replacement failed: editor is offline"
            return
        active = pulse.observe_active()
        if active is not None and not active.run.state.terminal:
            return
        self._pending_start = None
        self._begin_start(pending)

    def _advance_close(self) -> None:
        if not self._close_requested or self._close_complete:
            return
        if self._run_starting:
            return
        if self._run_snapshot is not None and not self._run_snapshot.state.terminal:
            return
        handle = self._handle
        if handle is not None:
            snapshot = self._run_snapshot
            if snapshot is None:
                return
            if not snapshot.state.terminal:
                return
            if not self._owner_reaped:
                if not self._reap_inflight:
                    self._reap_inflight = True
                    self._submit("reap", self._operation_generation, handle.wait)
                return
        with self._lock:
            if self._tracked or self._results:
                return
        if self._owned_connection is not None:
            if not self._owned_close_inflight:
                self._owned_close_inflight = True
                connection = self._owned_connection
                self._detach_pulse_authority()
                self._submit("close-owned", connection, connection.close)
            return
        if not self._pool_closed:
            # There are no tracked futures/results at this boundary, so the
            # executor is idle and joining it cannot wait on product work.
            # Publishing close_complete after shutdown(wait=False) left worker
            # threads unwinding while Python destroyed Qt/font resources at
            # process exit, producing intermittent 0xC0000409 crashes.
            self._pool.shutdown(wait=True)
            self._pool_closed = True
        self._connection_state = "closed"
        self._close_complete = True


__all__ = [
    "OwnedPulseConnection",
    "PulseEditorController",
    "PulseEditorControllerSnapshot",
    "PulseEditorLocalDelta",
    "PulseEditorProjection",
    "PulseRuntimeUpdate",
    "PulseScanProgressUpdate",
    "PulseRunFacade",
    "RenderedPulsePreview",
    "parse_remote_endpoint",
]
