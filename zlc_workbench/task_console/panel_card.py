"""One TaskConsole plot card over the sole ``zlc_plot`` runtime."""

from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass
import threading
from typing import Mapping

from PyQt5 import QtCore, QtWidgets

from zlc_data import OwnedSnapshot
from zlc_frontend.qt_widgets import (
    ACCENT,
    CARD_PAD,
    GREY,
    ORANGE,
    RED,
    FluentButton,
    FluentComboBox,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentPlotFitPanel,
    FluentPlotParameterPanel,
    FluentPlotSpecPanel,
    FluentPopup,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingsPopupAnchor,
    FluentSettingRow,
    FluentTreeComboBox,
    QtOwnerWake,
    fluent_scrollbar_thickness,
    popup_gap,
    scaled_px,
    setting_label_width,
    signals_blocked,
)
from zlc_neutral_atom.processing.signal_plane import SignalPublication, SignalValue
from zlc_plot import (
    DEFAULTS,
    FitEvent,
    PlotKind,
    PlotSpec,
    PlotSessionConfig,
    RasterFront,
    RasterOperation,
    RasterPlotHost,
    SelectionChange,
    SelectionData,
    SelectionEvent,
    SelectorKind,
    default_plot_spec,
)
from zlc_plot import Qt5PlotWidget

from .console_records import (
    DEFAULT_UPDATE_MS,
    PANEL_KINDS,
    UPDATE_INTERVALS,
    PanelConfig,
)
from .panel_board import card_size


_PLOT_PARAMETERS = "plot_parameters"
_UPDATE_MS = "update_ms"
_PANEL_PARAM_KEYS = frozenset((_PLOT_PARAMETERS, _UPDATE_MS))


@dataclass(frozen=True, slots=True)
class PanelSurfaceUpdate:
    """One staged worker operation awaiting board-coherent presentation."""

    panel_id: str
    serial: int
    host: RasterPlotHost
    publication: SignalPublication
    value: SignalValue
    future: Future
    replacement: bool

    def __post_init__(self) -> None:
        if not isinstance(self.host, RasterPlotHost):
            raise TypeError("surface update host must be RasterPlotHost")
        if not isinstance(self.publication, SignalPublication):
            raise TypeError("surface update requires SignalPublication")
        if self.publication.value(self.value.name) is not self.value:
            raise ValueError("surface update value is not owned by its publication")
        if not isinstance(self.future, Future):
            raise TypeError("surface update future must be Future")


class PanelCard(FluentGroupBox):
    """Stable Fluent chrome around one worker-owned raster surface.

    The card owns routing and widget lifetime only.  Projection, selectors,
    fitting, artists, rasterization, style, size/DPR semantics and export all
    remain inside its ``RasterPlotHost``.
    """

    changed = QtCore.pyqtSignal()
    layout_changed = QtCore.pyqtSignal()
    dropped = QtCore.pyqtSignal(object)
    update_interval_changed = QtCore.pyqtSignal()
    remove_requested = QtCore.pyqtSignal(object)
    edit_requested = QtCore.pyqtSignal(object)
    front_presented = QtCore.pyqtSignal()
    selectors_enabled_changed = QtCore.pyqtSignal(bool)
    selection_ready = QtCore.pyqtSignal(object, object)
    fit_ready = QtCore.pyqtSignal(object, object)

    @staticmethod
    def validate_config(config: PanelConfig) -> None:
        if not isinstance(config, PanelConfig):
            raise TypeError("panel card config must be PanelConfig")
        config.update_ms
        unknown = set(config.params) - _PANEL_PARAM_KEYS
        if unknown:
            raise ValueError(
                "panel config contains retired plot fields: "
                + ", ".join(sorted(map(str, unknown)))
            )
        parameters = config.params.get(_PLOT_PARAMETERS, {})
        if not isinstance(parameters, Mapping):
            raise TypeError("plot_parameters must be a mapping")

    def __init__(
        self,
        config: PanelConfig,
        parent=None,
        *,
        signal_groups_provider,
    ) -> None:
        self.validate_config(config)
        if not callable(signal_groups_provider):
            raise TypeError("signal_groups_provider must be callable")
        super().__init__(PANEL_KINDS[config.kind], parent)
        self.config = config
        self.panel_id = str(config.panel_id)
        self.signal_groups_provider = signal_groups_provider
        self._host: RasterPlotHost | None = None
        self._pending_host: RasterPlotHost | None = None
        self._plot_widget: QtWidgets.QWidget | None = None
        self._source_generation = None
        self._pending_generation = None
        self._source_schema_fingerprint: str | None = None
        self._pending_schema_fingerprint: str | None = None
        self._presented_value: SignalValue | None = None
        self._presented_publication: SignalPublication | None = None
        self._requested_publication: SignalPublication | None = None
        self._request_serial = 0
        self._publication_by_host_revision: dict[
            str, dict[int, SignalPublication]
        ] = {}
        self._unresolved_revisions_by_host: dict[str, set[int]] = {}
        self._latest_host_revisions: dict[str, tuple[int, ...]] = {}
        self._latest_host_sequence: dict[str, int] = {}
        self._presented_revisions_by_host: dict[str, tuple[int, ...]] = {}
        self._active_selector_kinds: set[SelectorKind] = set()
        self._selector_materializations: dict[Future, tuple[RasterPlotHost, SelectorKind]] = {}
        self._configuration_futures: set[Future] = set()
        self._subscriptions: dict[str, list[object]] = {}
        self._retiring_hosts: set[RasterPlotHost] = set()
        self._worker_events: deque[tuple[str, RasterPlotHost, object]] = deque()
        self._event_lock = threading.Lock()
        self._closing = False
        self._selectors_on = False
        self._drag_offset: QtCore.QPoint | None = None
        self._status_text = ""
        self._signal_info = ""

        holder = QtWidgets.QVBoxLayout(self)
        holder.setContentsMargins(CARD_PAD, scaled_px(2), CARD_PAD, CARD_PAD)
        holder.setSpacing(0)
        self.canvas_holder = holder
        self._placeholder = FluentLabel("Pick a signal in Setting")
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        holder.addWidget(self._placeholder)

        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)
        self._build_settings()
        self.setting_button = FluentButton("Setting", color=GREY)
        self.setting_button.setParent(self)
        self.setting_button.setFixedSize(
            scaled_px(74, minimum=64), scaled_px(26, minimum=22)
        )
        self.setting_button.clicked.connect(self._open_settings)
        self._settings_anchor = FluentSettingsPopupAnchor(
            self.settings_popup, self.setting_button
        )
        self.setCursor(QtCore.Qt.OpenHandCursor)
        self._apply_fixed_size()
        self.set_status("waiting for data…", error=False)

    @property
    def host(self) -> RasterPlotHost | None:
        return self._host

    @property
    def plot_widget(self):
        return self._plot_widget

    @property
    def presented_value(self) -> SignalValue | None:
        return self._presented_value

    @property
    def presented_publication(self) -> SignalPublication | None:
        return self._presented_publication

    @property
    def presented_snapshot(self) -> OwnedSnapshot | None:
        value = self._presented_value
        return None if value is None else value.snapshot

    def current_plot_config(self) -> tuple[object, dict[str, object]] | None:
        plot = self.config.plot
        if isinstance(plot, PlotKind):
            return None
        return (
            plot,
            dict(self.config.params.get(_PLOT_PARAMETERS, {})),
        )

    def _spec_for(self, snapshot: OwnedSnapshot):
        plot = self.config.plot
        if not isinstance(plot, PlotKind):
            return plot
        try:
            return default_plot_spec(snapshot.block.schema, plot)
        except ValueError as error:
            self._install_unresolved_spec_controls(snapshot)
            self.set_status(f"Choose plot axes in Setting: {error}", error=False)
            return None

    def _track_surface_submission(
        self,
        host: RasterPlotHost,
        revision: int,
        publication: SignalPublication,
    ) -> None:
        revision = int(revision)
        by_revision = self._publication_by_host_revision.setdefault(host.host_id, {})
        existing = by_revision.get(revision)
        if existing is not None and existing is not publication:
            raise ValueError("one plot revision cannot identify two publications")
        by_revision[revision] = publication
        self._unresolved_revisions_by_host.setdefault(host.host_id, set()).add(
            revision
        )
        self._prune_publications(host.host_id)

    @staticmethod
    def _surface_update_revision(update: PanelSurfaceUpdate) -> int:
        return int(update.value.snapshot.ref.revision.value)

    def _prune_publications(self, host_id: str) -> None:
        by_revision = self._publication_by_host_revision.get(host_id)
        if by_revision is None:
            return
        keep = set(self._presented_revisions_by_host.get(host_id, ()))
        keep.update(self._latest_host_revisions.get(host_id, ()))
        keep.update(self._unresolved_revisions_by_host.get(host_id, ()))
        for revision in tuple(by_revision):
            if revision not in keep:
                by_revision.pop(revision, None)
        if not by_revision:
            self._publication_by_host_revision.pop(host_id, None)

    def _observe_host_front(self, host: RasterPlotHost, front: RasterFront) -> None:
        if front.identity.host_id != host.host_id:
            raise ValueError("raster front belongs to another plot host")
        sequence = int(front.identity.sequence)
        if sequence < self._latest_host_sequence.get(host.host_id, -1):
            # The caller has already resolved this submission.  Its stale
            # worker front must not replace the newest Rolling window, but it
            # may have been the last reason an out-of-window publication was
            # retained.
            self._prune_publications(host.host_id)
            return
        self._latest_host_sequence[host.host_id] = sequence
        self._latest_host_revisions[host.host_id] = tuple(front.source_revisions)
        self._prune_publications(host.host_id)

    def observe_surface_result(
        self,
        update: PanelSurfaceUpdate,
        operation: RasterOperation,
    ) -> None:
        """Record one successful worker front without claiming it was presented."""

        if (
            update.panel_id != self.panel_id
            or operation.front.identity.host_id != update.host.host_id
        ):
            raise ValueError("surface result belongs to another panel or host")
        revision = self._surface_update_revision(update)
        if operation.front.identity.data_revision != revision:
            raise ValueError("surface result revision differs from its publication")
        unresolved = self._unresolved_revisions_by_host.get(update.host.host_id)
        if unresolved is not None:
            unresolved.discard(revision)
            if not unresolved:
                self._unresolved_revisions_by_host.pop(update.host.host_id, None)
        self._observe_host_front(update.host, operation.front)

    def reject_surface_update(self, update: PanelSurfaceUpdate) -> None:
        """Release a submission that never changed the worker-side plot front."""

        revision = self._surface_update_revision(update)
        unresolved = self._unresolved_revisions_by_host.get(update.host.host_id)
        if unresolved is not None:
            unresolved.discard(revision)
            if not unresolved:
                self._unresolved_revisions_by_host.pop(update.host.host_id, None)
        self._prune_publications(update.host.host_id)

    def finish_unpresented_surface_update(self, update: PanelSurfaceUpdate) -> None:
        """Finish an abandoned coherent batch without losing worker provenance."""

        update.future.add_done_callback(
            lambda completed, current=update: self._enqueue_worker_event(
                "unpresented-surface", current.host, (current, completed)
            )
        )

    def prepare_surface_update(
        self,
        value: SignalValue,
        publication: SignalPublication,
    ) -> PanelSurfaceUpdate | None:
        if self._closing:
            return None
        if not isinstance(value, SignalValue) or not isinstance(
            publication, SignalPublication
        ):
            raise TypeError("panel update requires SignalValue and SignalPublication")
        if publication.value(value.name) is not value:
            raise ValueError("panel update value/publication mismatch")
        if value.name != self.config.signal:
            raise ValueError("panel update belongs to another signal")
        if self._requested_publication is publication:
            return None

        snapshot = value.snapshot
        generation = publication.event_ref.generation
        schema_fingerprint = snapshot.ref.schema_fingerprint
        host = self._pending_host or self._host
        replacement = (
            host is None
            or (
                self._pending_host is not None
                and (
                    self._pending_generation != generation
                    or self._pending_schema_fingerprint != schema_fingerprint
                )
            )
            or (
                self._pending_host is None
                and (
                    self._source_generation != generation
                    or self._source_schema_fingerprint != schema_fingerprint
                )
            )
        )
        if replacement:
            self._retire_pending_host()
            spec = self._spec_for(snapshot)
            if spec is None:
                return None
            parameters = dict(self.config.params.get(_PLOT_PARAMETERS, {}))
            host = RasterPlotHost.from_plot(
                snapshot,
                spec,
                size=self.config.size,
                parameters=parameters,
            )
            self._pending_host = host
            self._pending_generation = generation
            self._pending_schema_fingerprint = schema_fingerprint
            self._install_host_subscriptions(host)
            future = host.dispatch(lambda: None)
        else:
            assert host is not None
            future = host.update_data(snapshot)
        self._request_serial += 1
        self._requested_publication = publication
        revision = snapshot.ref.revision.value
        self._track_surface_submission(host, revision, publication)
        return PanelSurfaceUpdate(
            self.panel_id,
            self._request_serial,
            host,
            publication,
            value,
            future,
            self._pending_host is host,
        )

    def can_accept_surface_update(
        self,
        update: PanelSurfaceUpdate,
        operation: RasterOperation,
    ) -> bool:
        return (
            not self._closing
            and update.panel_id == self.panel_id
            and update.serial == self._request_serial
            and update.publication is self._requested_publication
            and isinstance(operation, RasterOperation)
            and operation.front.identity.host_id == update.host.host_id
        )

    def accept_surface_update(
        self,
        update: PanelSurfaceUpdate,
        operation: RasterOperation,
    ) -> bool:
        if not self.can_accept_surface_update(update, operation):
            return False
        self._observe_host_front(update.host, operation.front)
        if self._publications_for(update.host, operation.front.source_revisions) is None:
            self.set_status("display source revisions are no longer available", error=True)
            return False
        if update.host is self._pending_host:
            self._activate_pending_host(operation.front)
        else:
            widget = self._plot_widget
            if widget is None or widget.host is not update.host:
                return False
            if not widget.present_front(operation.front):
                return False
        self._presented_revisions_by_host[update.host.host_id] = tuple(
            operation.front.source_revisions
        )
        self._prune_publications(update.host.host_id)
        self._presented_value = update.value
        self._presented_publication = update.publication
        self.set_status("ok", error=False)
        self.front_presented.emit()
        self._request_active_selector_values()
        return True

    def _activate_pending_host(self, front: RasterFront) -> None:
        host = self._pending_host
        if host is None or front.identity.host_id != host.host_id:
            raise ValueError("pending raster front belongs to another host")
        widget = Qt5PlotWidget(host, self, auto_present=False)
        widget.set_interaction_enabled(self._selectors_on)
        widget.errorOccurred.connect(lambda text: self.set_status(text, error=True))
        if not widget.present_front(front):
            widget.close_adapter()
            widget.deleteLater()
            raise RuntimeError("Qt rejected the prepared raster front")
        old_widget = self._plot_widget
        old_host = self._host
        if old_widget is not None:
            self.canvas_holder.removeWidget(old_widget)
            old_widget.hide()
            old_widget.close_adapter()
            old_widget.deleteLater()
        self._placeholder.hide()
        self.canvas_holder.insertWidget(0, widget)
        self._plot_widget = widget
        self._host = host
        self._source_generation = self._pending_generation
        self._source_schema_fingerprint = self._pending_schema_fingerprint
        self._pending_host = None
        self._pending_generation = None
        self._pending_schema_fingerprint = None
        if old_host is not None and old_host is not host:
            self._retire_host(old_host)
        self._install_host_controls(host)
        self._request_configuration(host)

    def _install_host_subscriptions(self, host: RasterPlotHost) -> None:
        card_ref = self

        def selection(event: SelectionEvent) -> None:
            card_ref._enqueue_worker_event("selection", host, event)

        def fit(event: FitEvent) -> None:
            card_ref._enqueue_worker_event("fit", host, event)

        futures = (
            host.subscribe_selection(selection),
            host.subscribe_fit(fit),
        )
        self._subscriptions[host.host_id] = list(futures)
        for future in futures:
            future.add_done_callback(
                lambda _future, current=host: self._enqueue_worker_event(
                    "subscription", current, _future
                )
            )

    def _enqueue_worker_event(
        self,
        kind: str,
        host: RasterPlotHost,
        value: object,
    ) -> None:
        with self._event_lock:
            if self._closing:
                return
            self._worker_events.append((kind, host, value))
        self._wake.request_owner_wake()

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        self._poll_retiring_hosts()
        if self._closing:
            return
        with self._event_lock:
            events = tuple(self._worker_events)
            self._worker_events.clear()
        for kind, host, value in events:
            if kind == "selection":
                self._accept_selection_event(host, value)
            elif kind == "fit":
                self._accept_fit_event(host, value)
            elif kind == "selector-data":
                self._accept_selector_future(host, value)
            elif kind == "configuration":
                self._accept_configuration_future(host, value)
            elif kind == "subscription":
                self._accept_subscription_future(host, value)
            elif kind == "local-front":
                self._accept_local_front(host, value)
            elif kind == "unpresented-surface":
                self._accept_unpresented_surface_result(host, value)

    def _accept_unpresented_surface_result(
        self,
        host: RasterPlotHost,
        value: object,
    ) -> None:
        try:
            update, future = value
        except (TypeError, ValueError):
            return
        if (
            not isinstance(update, PanelSurfaceUpdate)
            or update.host is not host
            or not isinstance(future, Future)
        ):
            return
        try:
            operation = future.result()
            if not isinstance(operation, RasterOperation):
                raise TypeError("plot worker returned another operation type")
        except (CancelledError, RuntimeError, TypeError, ValueError):
            self.reject_surface_update(update)
            return
        self.observe_surface_result(update, operation)

    def _accept_local_front(self, host: RasterPlotHost, future: object) -> None:
        if host is not self._host or not isinstance(future, Future):
            return
        try:
            operation = future.result()
            if not isinstance(operation, RasterOperation):
                raise TypeError("plot control returned another operation type")
        except CancelledError:
            return
        except BaseException as error:
            self.set_status(str(error), error=True)
            return
        self._present_local_front(operation.front)

    def _accept_subscription_future(self, host: RasterPlotHost, future: Future) -> None:
        try:
            operation = future.result()
            unsubscribe = operation.value
            if not callable(unsubscribe):
                raise TypeError("plot event subscription returned no release callable")
        except CancelledError:
            return
        except BaseException as error:
            if host in (self._host, self._pending_host):
                self.set_status(str(error), error=True)
            return
        self._subscriptions.setdefault(host.host_id, []).append(unsubscribe)

    def _publications_for(
        self,
        host: RasterPlotHost,
        revisions: tuple[int, ...],
    ) -> tuple[SignalPublication, ...] | None:
        by_revision = self._publication_by_host_revision.get(host.host_id, {})
        try:
            publications = tuple(by_revision[int(revision)] for revision in revisions)
        except KeyError:
            return None
        unique: list[SignalPublication] = []
        for publication in publications:
            if publication not in unique:
                unique.append(publication)
        return tuple(unique)

    def _accept_selection_event(
        self,
        host: RasterPlotHost,
        event: object,
    ) -> None:
        if host is not self._host or not isinstance(event, SelectionEvent):
            return
        kind = event.selector.kind
        if event.change is SelectionChange.REMOVED:
            self._active_selector_kinds.discard(kind)
            self.selection_ready.emit(kind, None)
            return
        if event.change is not SelectionChange.COMMITTED:
            return
        data = event.data
        if not isinstance(data, SelectionData):
            return
        self._active_selector_kinds.add(kind)
        publications = self._publications_for(host, data.source_revisions)
        if publications is None:
            self.set_status("selector source revision is no longer displayed", error=True)
            return
        self.selection_ready.emit(data, publications)

    def _request_active_selector_values(self) -> None:
        host = self._host
        if host is None:
            return
        for kind in tuple(self._active_selector_kinds):
            future = host.selector_data(kind)
            self._selector_materializations[future] = (host, kind)
            future.add_done_callback(
                lambda completed, current=host: self._enqueue_worker_event(
                    "selector-data", current, completed
                )
            )

    def _accept_selector_future(self, host: RasterPlotHost, future: object) -> None:
        if not isinstance(future, Future):
            return
        binding = self._selector_materializations.pop(future, None)
        if binding is None or binding[0] is not host or host is not self._host:
            return
        try:
            operation = future.result()
            data = operation.value
            if not isinstance(data, SelectionData):
                return
        except CancelledError:
            return
        except BaseException as error:
            self.set_status(str(error), error=True)
            return
        publications = self._publications_for(host, data.source_revisions)
        if publications is not None:
            self.selection_ready.emit(data, publications)

    def _accept_fit_event(self, host: RasterPlotHost, event: object) -> None:
        if host is not self._host or not isinstance(event, FitEvent):
            return
        selections = (
            event.selection
            if isinstance(event.selection, tuple)
            else (event.selection,)
        )
        revisions: list[int] = []
        for selection in selections:
            for revision in selection.source_revisions:
                if revision not in revisions:
                    revisions.append(revision)
        publications = self._publications_for(
            host, tuple(revisions)
        )
        if publications is None:
            return
        self.fit_ready.emit(event, publications)

    def _clear_plot_controls(self) -> None:
        for widget in tuple(self._host_controls):
            self._display_layout.removeWidget(widget)
            widget.hide()
            widget.deleteLater()
        self._host_controls.clear()
        self._unresolved_spec_identity = None

    def _install_unresolved_spec_controls(self, snapshot: OwnedSnapshot) -> None:
        identity = (snapshot.ref.schema_fingerprint, self.config.kind)
        if self._unresolved_spec_identity == identity:
            return
        self._clear_plot_controls()
        panel = FluentPlotSpecPanel(
            None,
            snapshot.block.schema,
            self.settings_popup,
            kind=self.config.kind,
        )
        panel.specAccepted.connect(self._accept_authored_spec)
        panel.specRejected.connect(
            lambda error: self.set_status(str(error), error=False)
        )
        self._display_layout.addWidget(panel)
        self._host_controls.append(panel)
        self._unresolved_spec_identity = identity

    @QtCore.pyqtSlot(object)
    def _accept_authored_spec(self, spec: object) -> None:
        if not isinstance(spec, PlotSpec) or spec.kind is not self.config.kind:
            self.set_status("PlotSpec authoring returned another plot kind", error=True)
            return
        if self.config.plot != spec:
            self.config.plot = spec
            self._requested_publication = None
            self.changed.emit()
        self.set_status("plot axes ready", error=False)

    def _install_host_controls(self, host: RasterPlotHost) -> None:
        self._clear_plot_controls()
        snapshot = self.presented_snapshot
        if snapshot is None and self._requested_publication is not None:
            value = self._requested_publication.value(self.config.signal)
            snapshot = None if value is None else value.snapshot
        if snapshot is None:
            return
        spec = FluentPlotSpecPanel(host, snapshot.block.schema, self.settings_popup)
        parameters = FluentPlotParameterPanel(host, self.settings_popup)
        fit = FluentPlotFitPanel(host, live=True, parent=self.settings_popup)
        for widget in (spec, parameters, fit):
            self._display_layout.addWidget(widget)
            self._host_controls.append(widget)
        spec.frontReady.connect(self._present_local_front)
        spec.specAccepted.connect(self._accept_host_configuration)
        spec.specRejected.connect(lambda error: self.set_status(str(error), error=True))
        parameters.frontReady.connect(self._present_local_front)
        parameters.parameterRejected.connect(
            lambda _name, error: self.set_status(str(error), error=True)
        )
        fit.frontReady.connect(self._present_local_front)
        fit.fitRejected.connect(
            lambda _action, error: self.set_status(str(error), error=True)
        )

    @QtCore.pyqtSlot(object)
    def _present_local_front(self, front: object) -> None:
        widget = self._plot_widget
        if widget is None or not isinstance(front, RasterFront):
            return
        host = widget.host
        try:
            self._observe_host_front(host, front)
            if self._publications_for(host, front.source_revisions) is None:
                raise RuntimeError(
                    "display source revisions are no longer available"
                )
            if not widget.present_front(front):
                return
        except (RuntimeError, TypeError, ValueError) as error:
            self.set_status(str(error), error=True)
            return
        self._presented_revisions_by_host[host.host_id] = tuple(
            front.source_revisions
        )
        self._prune_publications(host.host_id)
        self._request_configuration(host)

    def _request_configuration(self, host: RasterPlotHost) -> None:
        future = host.configuration()
        self._configuration_futures.add(future)
        future.add_done_callback(
            lambda completed, current=host: self._enqueue_worker_event(
                "configuration", current, completed
            )
        )

    def _accept_configuration_future(
        self,
        host: RasterPlotHost,
        future: object,
    ) -> None:
        if not isinstance(future, Future):
            return
        self._configuration_futures.discard(future)
        if host is not self._host:
            return
        try:
            operation = future.result()
            config = operation.value
            if not isinstance(config, PlotSessionConfig):
                raise TypeError("plot host returned another configuration value")
        except CancelledError:
            return
        except BaseException as error:
            self.set_status(str(error), error=True)
            return
        self._accept_host_configuration(config)

    @QtCore.pyqtSlot(object)
    def _accept_host_configuration(self, config: object) -> None:
        if not isinstance(config, PlotSessionConfig):
            return
        changed = False
        parameters = dict(config.parameters)
        if self.config.plot != config.spec:
            self.config.plot = config.spec
            changed = True
        if self.config.params.get(_PLOT_PARAMETERS) != parameters:
            self.config.params[_PLOT_PARAMETERS] = parameters
            changed = True
        if config.size != self.config.size:
            self.config.size = config.size
            self._apply_fixed_size(sync_host=False)
            self.layout_changed.emit()
            changed = True
        if changed:
            self.changed.emit()

    def set_selectors_enabled(self, enabled: bool) -> None:
        selected = bool(enabled)
        if selected == self._selectors_on:
            return
        self._selectors_on = selected
        widget = self._plot_widget
        if widget is not None:
            widget.set_interaction_enabled(selected)
        self.selectors_enabled_changed.emit(selected)

    @property
    def selectors_enabled(self) -> bool:
        return self._selectors_on

    def _apply_fixed_size(
        self,
        size_name: str | None = None,
        *,
        sync_host: bool = False,
    ) -> None:
        selected = self.config.size if size_name is None else str(size_name)
        self.setFixedSize(*card_size(selected))
        self._place_setting_button()
        if sync_host and self._host is not None:
            future = self._host.set_size(selected)
            future.add_done_callback(
                lambda completed, current=self._host: self._enqueue_worker_event(
                    "local-front", current, completed
                )
            )

    def _place_setting_button(self) -> None:
        button = getattr(self, "setting_button", None)
        if button is not None:
            button.move(
                self.width() - button.width() - scaled_px(8), scaled_px(4)
            )
            button.raise_()

    def _build_settings(self) -> None:
        popup = FluentPopup(self)
        outer = QtWidgets.QVBoxLayout(popup)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = FluentScrollArea()
        content = QtWidgets.QWidget()
        content.setStyleSheet("background: transparent;")
        column = QtWidgets.QVBoxLayout(content)
        pad = scaled_px(10)
        column.setContentsMargins(
            pad,
            pad,
            pad + fluent_scrollbar_thickness() + scaled_px(4),
            pad,
        )
        column.setSpacing(scaled_px(10, minimum=6))
        scroll.set_width_bounded_widget(content)
        outer.addWidget(scroll)
        self.settings_popup = popup
        self._settings_scroll = scroll
        self._settings_col = column
        self._settings_h_hwm = 0
        label_width = setting_label_width(
            ("Signal", "Size", "Update", "Title", "Facet", "Reduction")
        )
        popup.setFixedWidth(
            label_width
            + scaled_px(360, minimum=320)
            + 2 * pad
            + fluent_scrollbar_thickness()
            + scaled_px(4)
        )

        def section(title: str) -> QtWidgets.QVBoxLayout:
            column.addWidget(FluentSectionLabel(title))
            layout = QtWidgets.QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(scaled_px(6, minimum=4))
            column.addLayout(layout)
            return layout

        source = section("Source")
        self.signal_combo = FluentTreeComboBox()
        self.signal_combo.currentIndexChanged.connect(self._on_signal_pick)
        source.addWidget(
            FluentSettingRow(
                "Signal", self.signal_combo, label_width=label_width, parent=content
            )
        )
        self.status = FluentLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        source.addWidget(self.status)

        display = section("Display")
        self.size_combo = FluentComboBox()
        for preset in DEFAULTS.layout.size_names:
            self.size_combo.addItem(preset, preset)
        self.size_combo.setCurrentIndex(self.size_combo.findData(self.config.size))
        self.size_combo.activated.connect(self._on_size_pick)
        display.addWidget(
            FluentSettingRow(
                "Size", self.size_combo, label_width=label_width, parent=content
            )
        )
        self.update_combo = FluentComboBox()
        for interval in UPDATE_INTERVALS:
            self.update_combo.addItem(f"{interval} ms", interval)
        self.update_combo.setCurrentIndex(
            self.update_combo.findData(self.config.update_ms)
        )
        self.update_combo.currentIndexChanged.connect(self._on_update_interval)
        display.addWidget(
            FluentSettingRow(
                "Update", self.update_combo, label_width=label_width, parent=content
            )
        )
        self._display_layout = display
        self._host_controls: list[QtWidgets.QWidget] = []
        self._unresolved_spec_identity: tuple[str, PlotKind] | None = None

        panel = section("Panel")
        self.title_edit = FluentLineEdit(self.config.title)
        self.title_edit.editingFinished.connect(self._commit_title)
        panel.addWidget(
            FluentSettingRow(
                "Title", self.title_edit, label_width=label_width, parent=content
            )
        )
        actions = QtWidgets.QHBoxLayout()
        remove = FluentButton("Remove", color=ORANGE)
        edit = FluentButton("Edit…", color=ACCENT)
        remove.clicked.connect(self._remove_from_settings)
        edit.clicked.connect(self._edit_from_settings)
        actions.addWidget(remove)
        actions.addWidget(edit)
        actions.addStretch(1)
        panel.addLayout(actions)
        column.addStretch(1)
        self._refresh_signal_combo()

    def _refresh_signal_combo(self) -> None:
        groups = self.signal_groups_provider(str(self.config.signal or ""))
        with signals_blocked(self.signal_combo):
            self.signal_combo.set_signal_tree(
                groups,
                current=str(self.config.signal or ""),
                none_label="(none)",
            )

    def refresh_open_signal_metadata(self) -> bool:
        if not self.settings_popup.isVisible():
            return False
        self._refresh_signal_combo()
        return True

    def refresh_open_signal_topology(self) -> bool:
        return self.refresh_open_signal_metadata()

    def _on_signal_pick(self, _index: int) -> None:
        selected = self.signal_combo.currentData()
        value = "" if selected is None else str(selected)
        if value == self.config.signal:
            return
        self.config.signal = value
        self.config.plot = self.config.kind
        self.config.params.pop(_PLOT_PARAMETERS, None)
        self._retire_all_surfaces()
        self.changed.emit()
        self.set_status("waiting for data…" if value else "pick a signal", error=False)

    def _on_size_pick(self, _index: int) -> None:
        selected = str(self.size_combo.currentData() or self.config.size)
        if selected == self.config.size:
            return
        DEFAULTS.layout.validate_preset(selected)
        self.config.size = selected
        self._apply_fixed_size(sync_host=True)
        self.layout_changed.emit()
        self.changed.emit()

    def _on_update_interval(self, _index: int) -> None:
        value = int(self.update_combo.currentData())
        if value == self.config.update_ms:
            return
        self.config.params[_UPDATE_MS] = value
        self.update_interval_changed.emit()
        self.changed.emit()

    def _commit_title(self) -> None:
        title = self.title_edit.text().strip()
        if title == self.config.title:
            return
        self.config.title = title
        self._refresh_title()
        self.changed.emit()

    def _refresh_title(self) -> None:
        title = PANEL_KINDS[self.config.kind]
        if self.config.title:
            title += f" · {self.config.title}"
        if self._signal_info:
            title += f" · {self._signal_info}"
        self.setTitle(title)

    def set_signal_info(self, info: str) -> None:
        value = str(info)
        if value != self._signal_info:
            self._signal_info = value
            self._refresh_title()

    def set_status(self, text: str, *, error: bool) -> None:
        self._status_text = str(text)
        if hasattr(self, "status"):
            self.status.setText(self._status_text)
            self.status.setStyleSheet(
                f"color: {RED if error else GREY}; background: transparent; border: none;"
            )
        if hasattr(self, "setting_button"):
            self.setting_button.setToolTip(self._status_text)

    def _open_settings(self) -> None:
        self._settings_anchor.toggle(
            self._settings_scroll.widget(),
            prepare=self._prepare_settings_popup,
            present=self._present_settings_popup,
        )

    def _prepare_settings_popup(self) -> None:
        self._refresh_signal_combo()
        for widget in self._host_controls:
            reconcile = getattr(widget, "reconcile", None)
            if callable(reconcile):
                reconcile()

    def _present_settings_popup(self) -> None:
        popup = self.settings_popup
        anchor = self.setting_button.mapToGlobal(
            QtCore.QPoint(self.setting_button.width(), self.setting_button.height())
        )
        self._size_settings_popup()
        top_y = anchor.y() + popup_gap()
        screen = QtWidgets.QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        x = anchor.x() - popup.width()
        if available is not None:
            x = max(available.left(), min(x, available.right() - popup.width()))
        popup.move(x, top_y)
        popup.show()
        popup.raise_()

    def _size_settings_popup(self) -> None:
        popup = self.settings_popup
        popup.adjustSize()
        anchor_y = self.setting_button.mapToGlobal(
            QtCore.QPoint(0, self.setting_button.height())
        ).y()
        top_y = anchor_y + popup_gap()
        content = self._settings_scroll.widget()
        content_h = content.sizeHint().height() + 2 * scaled_px(10)
        panel_bottom = self.mapToGlobal(QtCore.QPoint(0, self.height())).y()
        cap = max(scaled_px(140), panel_bottom - top_y)
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            cap = min(cap, screen.availableGeometry().bottom() - top_y)
        wanted = min(content_h, cap)
        self._settings_h_hwm = max(self._settings_h_hwm, int(wanted))
        popup.setMaximumHeight(int(cap))
        popup.resize(
            popup.width(), max(scaled_px(140), min(self._settings_h_hwm, cap))
        )

    def _remove_from_settings(self) -> None:
        self.settings_popup.hide()
        self.remove_requested.emit(self)

    def _edit_from_settings(self) -> None:
        self.settings_popup.hide()
        self.edit_requested.emit(self)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_offset = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_offset is not None and event.buttons() & QtCore.Qt.LeftButton:
            self.move(self.mapToParent(event.pos() - self._drag_offset))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton and self._drag_offset is not None:
            self._drag_offset = None
            self.setCursor(QtCore.Qt.OpenHandCursor)
            self.dropped.emit(self)
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._place_setting_button()

    def _retire_host(self, host: RasterPlotHost) -> None:
        entries = self._subscriptions.pop(host.host_id, [])
        for entry in entries:
            if callable(entry):
                try:
                    entry()
                except BaseException:
                    pass
            elif isinstance(entry, Future):
                entry.cancel()
        self._publication_by_host_revision.pop(host.host_id, None)
        self._unresolved_revisions_by_host.pop(host.host_id, None)
        self._latest_host_revisions.pop(host.host_id, None)
        self._latest_host_sequence.pop(host.host_id, None)
        self._presented_revisions_by_host.pop(host.host_id, None)
        if host.close(timeout=0.0):
            self._retiring_hosts.discard(host)
        else:
            self._retiring_hosts.add(host)
            if not self._closing:
                QtCore.QTimer.singleShot(25, self._wake.request_owner_wake)

    def _poll_retiring_hosts(self) -> bool:
        for host in tuple(self._retiring_hosts):
            if host.close(timeout=0.0):
                self._retiring_hosts.discard(host)
        if self._retiring_hosts and not self._closing:
            QtCore.QTimer.singleShot(25, self._wake.request_owner_wake)
        return not self._retiring_hosts

    def _retire_pending_host(self) -> None:
        host = self._pending_host
        self._pending_host = None
        self._pending_generation = None
        self._pending_schema_fingerprint = None
        if host is not None:
            self._retire_host(host)

    def _retire_all_surfaces(self) -> None:
        self._request_serial += 1
        widget = self._plot_widget
        self._plot_widget = None
        if widget is not None:
            self.canvas_holder.removeWidget(widget)
            widget.hide()
            widget.close_adapter()
            widget.deleteLater()
        host = self._host
        self._host = None
        if host is not None:
            self._retire_host(host)
        self._retire_pending_host()
        self._placeholder.show()
        self._presented_value = None
        self._presented_publication = None
        self._requested_publication = None
        self._source_generation = None
        self._source_schema_fingerprint = None
        self._active_selector_kinds.clear()

    def retire_source_generation(self) -> None:
        self._retire_all_surfaces()

    def shutdown(self) -> bool:
        if not self._closing:
            self._closing = True
            self._wake.detach()
            self.settings_popup.hide()
            self._retire_all_surfaces()
        return self._poll_retiring_hosts()


__all__ = ["PanelCard", "PanelSurfaceUpdate"]
