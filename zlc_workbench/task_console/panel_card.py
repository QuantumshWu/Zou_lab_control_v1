"""The console's panel card: its raster surface, its Setting, and the card geometry.

A card owns NO plotting object.  Its surface is a raster board painted from
immutable bytes, and every picture on it was composed on a worker from one
frozen snapshot (:mod:`zlc_frontend.panel_render`).  That is what keeps a
megapixel frame off the thread that also has to stay responsive, and it is why
the display knobs here are stored FACTS (``config.params``) rather than pushes
into a live figure: the knobs are read back on the next compose, so what is
stored and what is drawn cannot drift.

"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import threading
from typing import Mapping
from PyQt5 import QtCore, QtWidgets

import zlc_frontend.qt_widgets as _qt_widgets
from zlc_frontend.qt_widgets import (
    ACCENT,
    CARD_PAD,
    FluentButton,
    FluentComboBox,
    FluentGroupBox,
    FitAuthoringPane,
    FluentLabel,
    FluentLineEdit,
    FluentPopup,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingsPopupAnchor,
    FluentSettingRow,
    FluentTreeComboBox,
    FigureOutputAuthority,
    FigureSurfaceContext,
    FigureSurfaceHost,
    FigureSurfaceRenderRequest as _PanelRenderRequest,
    ViewSpecEditor,
    GREY,
    ORANGE,
    RED,
    fluent_scrollbar_thickness,
    popup_gap,
    scaled_px,
    setting_label_width as _setting_label_width,
    signals_blocked as _signals_blocked,
)
from zlc_frontend.form import FormFieldProps
from zlc_frontend.render_style import panel_display_size
from zlc_frontend.render import HistogramPanelPayload
from zlc_frontend.display_range import RelimMode
from zlc_frontend.panel_params import (
    panel_display_form_labels,
    panel_display_form_spec,
    panel_display_form_values,
    panel_display_form_values_from_tree,
    panel_display_form_values_to_tree,
    panel_display_param_keys,
    panel_display_state_from_form,
    panel_display_value_range_keys,
)
from zlc_frontend import (
    FigureIntent,
    ViewIntent,
)
from zlc_frontend.plot_panel import (
    HISTOGRAM_CELL_THRESHOLDS_PARAM as _HISTOGRAM_CELL_THRESHOLDS_PARAM,
    HISTOGRAM_THRESHOLDS_PARAM as _HISTOGRAM_THRESHOLDS_PARAM,
    VIEW_SPEC_PARAM as _VIEW_SPEC_PARAM,
)
from .console_records import (
    DEFAULT_UPDATE_MS,
    PANEL_KINDS,
    PanelConfig,
    UPDATE_INTERVALS,
)
from zlc_frontend.panel_size import PANEL_SIZES
from zlc_frontend.plot_kind import PlotKind
from zlc_neutral_atom.processing.signal_plane import SignalPublication

from .panel_board import card_size as _card_size


_FIT_SPEC_PARAM = "figure_fit_spec"

# qt_widgets submodules are reached as ATTRIBUTES of the one facade binding: their names are
# deliberately absent from the facade __all__, and the package forbids outside deep imports.
FORM_WIDGET_HANDLERS = _qt_widgets.FORM_WIDGET_HANDLERS


@dataclass(frozen=True, slots=True)
class _PointerInteractionPin:
    """One pressed Figure surface and the exact semantic front it displays."""

    host: object
    origin: object
    value: object
    publication: SignalPublication
    surface_id: str


@dataclass(frozen=True, slots=True)
class _PanelFitRequest:
    """One exact surface-local Fit operation retained by composition only."""

    panel_id: str
    surface_id: str
    request_revision: int
    source: object
    publication: SignalPublication
    figure: object
    overlay_figure: object | None
    source_frame: object | None
    spec: object
    cached_result: object | None = None
    cancelled: threading.Event = field(
        default_factory=threading.Event,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        from zlc_data import FitSpec, OwnedSnapshot

        for name in ("panel_id", "surface_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)
        if (
            isinstance(self.request_revision, bool)
            or not isinstance(self.request_revision, int)
            or self.request_revision <= 0
        ):
            raise ValueError("Fit request revision must be positive")
        snapshot = getattr(self.source, "snapshot", None)
        if not isinstance(snapshot, OwnedSnapshot):
            raise TypeError("Figure Fit requires an immutable source snapshot")
        if not isinstance(self.spec, FitSpec):
            raise TypeError("Figure Fit requires FitSpec")
        from zlc_frontend import DataFigure

        if not isinstance(self.figure, DataFigure):
            raise TypeError("Figure Fit requires an immutable DataFigure")
        entries = tuple(self.figure.datasets.entries)
        if len(entries) != 1 or entries[0].snapshot.ref != snapshot.ref:
            raise ValueError("Figure Fit source and Figure are not one front")
        if self.overlay_figure is not None:
            if not isinstance(self.overlay_figure, DataFigure):
                raise TypeError("Fit overlay_figure must be DataFigure or None")
            overlay_entries = tuple(self.overlay_figure.datasets.entries)
            if (
                len(overlay_entries) != 1
                or overlay_entries[0].snapshot.ref != snapshot.ref
            ):
                raise ValueError("Fit overlay Figure belongs to another source")
        if self.source_frame is not None:
            from zlc_frontend.render import BoardFrame

            if not isinstance(self.source_frame, BoardFrame):
                raise TypeError("Figure Fit source_frame must be BoardFrame or None")
            panels = tuple(self.source_frame.panels)
            if len(panels) != 1:
                raise ValueError("Figure Fit source_frame must contain one panel")
            evaluated_input = getattr(
                panels[0].display_payload,
                "evaluated_input",
                None,
            )
            if evaluated_input is not None and evaluated_input.ref != snapshot.ref:
                raise ValueError("Figure Fit source_frame belongs to another revision")
        if (
            self.spec.committed_transform.source_schema_fingerprint
            != snapshot.ref.schema_fingerprint
        ):
            raise ValueError("Figure Fit spec belongs to another source schema")
        if self.cached_result is not None:
            from zlc_data import FitResultBatch

            if not isinstance(self.cached_result, FitResultBatch):
                raise TypeError("cached_result must be FitResultBatch or None")
            if (
                self.cached_result.source_ref != snapshot.ref
                or self.cached_result.spec != self.spec
            ):
                raise ValueError("cached Fit result belongs to another operation")
        _require_publication_value(self.publication, self.source)


@dataclass(slots=True)
class _FitSurfaceState:
    """Private authoring/result state for exactly one Figure surface."""

    pane: FitAuthoringPane
    surface_id: str
    context_provider: object | None
    host_provider: object | None
    spec: object | None = None
    source: object | None = None
    publication: SignalPublication | None = None
    figure: object | None = None
    overlay_figure: object | None = None
    source_frame: object | None = None
    result: object | None = None
    authority_key: object | None = None
    authoring_key: object | None = None
    authoring_options: tuple = ()
    request_revision: int = 0
    requested_ref: object | None = None
    requested_overlay_key: object | None = None

    def __post_init__(self) -> None:
        self.surface_id = str(self.surface_id).strip()
        if not self.surface_id:
            raise ValueError("Fit surface_id must not be empty")
        if self.context_provider is not None and not callable(self.context_provider):
            raise TypeError("Fit context_provider must be callable or None")
        if self.host_provider is not None and not callable(self.host_provider):
            raise TypeError("Fit host_provider must be callable or None")


def _require_publication_value(
    publication: SignalPublication,
    value,
) -> SignalPublication:
    """Validate one private Workbench publication/value pair.

    Frontend requests deliberately carry no neutral transaction object.  The
    composition layer therefore retains this exact pair and verifies identity
    whenever a render, gesture, Fit, or Edit snapshot crosses an operation
    boundary.
    """

    if not isinstance(publication, SignalPublication):
        raise TypeError("Figure source requires an exact SignalPublication")
    name = str(getattr(value, "name", ""))
    if not name or publication.value(name) is not value:
        raise ValueError("SignalPublication does not own the Figure source value")
    return publication



# A fresh plot panel is BLANK: a pure view is fully decoupled from acquisition, so
# it shows nothing until the user picks a declared signal in its Setting
# -- it must NOT auto-bind to any node's signal.  An empty source is the blank
# state; ``refresh`` treats it (and a source that produces None) as "pick a signal"
# rather than an error, so a blank panel sits quietly until wired.


# Board layout (raw px).  The board is a pure PIXEL plane of card AABBs -- there is NO column
# grid.  WIDTH and HEIGHT wrap the exact FigureSpec logical panel size plus Fluent chrome;
# the card is exactly large enough for its figure, with NO stretch or blank
# padding below (every size hugs its content).  ``PanelConfig.col`` is the card's pixel X and
# ``row`` is the card's pixel Y; :func:`pack` is the order-driven TOP-LEFT GRAVITY packer that places
# every card at the first free NW slot in list order.  The CARD'S FORMAT (rounded corners, shadow, grey title strip,
# content padding) belongs to the FluentGroupBox COMPONENT (qt_widgets.CARD_PAD / CARD_TITLE_PX,
# the single source); this module only lays cards out.
class PanelCard(FluentGroupBox):
    """One dashboard panel: a TITLED frame (title strip = the panel KIND + the signal-source
    legend, top-left) holding the frontend canvas, and a text
    "Setting" button on the title strip (top-right).  The frame border is the DRAG
    HANDLE (the board keeps all its own pointer interactions); the card
    spans whole layout slots -- a 2-row card is exactly two
    1-row cards plus the gap."""

    changed = QtCore.pyqtSignal()          # any config edit (console marks dirty)
    layout_changed = QtCore.pyqtSignal()   # size/slot change (console re-arranges)
    dropped = QtCore.pyqtSignal(object)    # drag-release ONLY (console snaps the drop to its nearest anchor)
    update_interval_changed = QtCore.pyqtSignal()  # per-panel refresh rate change (console re-bases the timer)
    remove_requested = QtCore.pyqtSignal(object)
    edit_requested = QtCore.pyqtSignal(object)   # "Edit…" -> open the panel's Edit tab
    front_presented = QtCore.pyqtSignal()
    selectors_enabled_changed = QtCore.pyqtSignal(bool)
    figure_outputs_changed = QtCore.pyqtSignal()
    fit_requested = QtCore.pyqtSignal(object)
    fit_cancel_requested = QtCore.pyqtSignal(str)
    fit_output_clear_requested = QtCore.pyqtSignal(str)

    @staticmethod
    def validate_config(config: PanelConfig):
        """Validate every owner-coded value consumed during card construction.

        This is intentionally QWidget-free so a whole TaskConsole layout can be
        checked before the currently running console is stopped.
        """

        if not isinstance(config, PanelConfig):
            raise TypeError("panel card config must be PanelConfig")
        config.update_ms
        fit_spec = None
        raw_fit_spec = config.params.get(_FIT_SPEC_PARAM)
        if raw_fit_spec is not None:
            from zlc_data.fit import fit_spec_from_tree

            fit_spec = fit_spec_from_tree(raw_fit_spec)
        from zlc_frontend.plot_panel import plot_panel_view_from_params

        plot_panel_view_from_params(config.kind, config.params)
        return fit_spec

    def __init__(self, config: PanelConfig, parent=None, *,
                 signal_groups_provider=None,
                 render_request=None,
                 presentation_provider=None,
                 initial_grid_size_pending: bool = False):
        fit_spec = self.validate_config(config)
        if type(initial_grid_size_pending) is not bool:
            raise TypeError("initial_grid_size_pending must be bool")
        if initial_grid_size_pending and config.kind is not PlotKind.GRID:
            raise ValueError(
                "only a fresh Grid panel may await an initial size recommendation"
            )
        # Titled frame: the title strip carries the panel KIND (top-left) and the
        # Setting button (top-right), so the card is delineated like the rest.
        super().__init__(PANEL_KINDS[config.kind], parent)
        self.config = config
        # Runtime-only admission fact supplied by the composition root.  It is
        # never encoded in PanelConfig: a loaded size and a user-authored size
        # are already authoritative.  A fresh Grid consumes this exactly once,
        # when the first complete schema/ViewSpec raster is presented.
        self._initial_grid_size_pending = initial_grid_size_pending
        # One call returns the complete frontend picker tree for the current
        # topology.  The composition root builds it from one topology
        # projection, so opening Setting never scans the same producers once
        # for names, again for sources, again for labels, and again for shapes.
        if not callable(signal_groups_provider):
            raise TypeError("signal_groups_provider must be callable")
        self.signal_groups_provider = signal_groups_provider
        # callable(card, force=False) -> enqueue one latest-only worker compose.
        # The callback receives no mutable render state: ``render_request`` asks
        # this card to freeze a request first, then the worker owns every
        # PanelComposer/Agg object and Qt only presents its immutable result.
        if not callable(render_request):
            raise TypeError("render_request must be callable")
        self._render_request = render_request
        if not callable(presentation_provider):
            raise TypeError("presentation_provider must be callable")
        self._presentation_provider = presentation_provider
        # Figure-owned derived signals.  Area is an authoritative named-axis
        # Selection promoted from a completed gesture; Cross is a completed
        # right-click coordinate.  Neither is a Measurement parameter and
        # neither opens another window.
        self._figure_output_authority = FigureOutputAuthority(self)
        # Each Figure surface owns one private Fit scope.  The live Setting
        # scope follows presented fronts; every Edit scope remains bound to its
        # own explicit snapshot.  They share only immutable FitSpec seeds.
        self._fit_surfaces: dict[FitAuthoringPane, _FitSurfaceState] = {}
        self._live_fit_pane: FitAuthoringPane | None = None
        self._persisted_fit_spec = fit_spec
        # The card's display surface is an immutable-bytes raster board.
        # The panel's stable identity: the board, its composer and every frame
        # they exchange are keyed on it, so a presented frame can only ever land
        # on the panel it was composed for.
        self.panel_id = str(config.panel_id)
        self.board = None
        self._pending_frame = None    # composed front awaiting its present pass
        self._pending_faceted_result = None
        # The newest already-immutable data-plane value may be newer than the
        # front on screen.  It is input to worker composition only; visible
        # schema, Setting/Edit controls, selectors and Fit always derive from
        # ``_presented_value`` until the matching raster is committed.
        self._candidate_value = None
        self._candidate_publication = None
        # Worker completion and visible presentation are different facts.
        # Promote this group only after the Qt board accepts its matching
        # immutable frame in ``present()``.
        self._pending_figure = None
        self._pending_display = None
        self._pending_contract = None
        self._pending_value = None
        self._pending_publication = None
        self._pending_render_request_revision = None
        self._presented_figure = None
        self._presented_display = None
        self._presented_contract = None
        self._presented_value = None
        self._presented_publication = None
        self._presented_render_request_revision = None
        self._grid_focus = None
        self._render_request_revision = 0
        self._requested_signature = None
        self._latest_requested_source_ref = None
        self._latest_requested_source_key = None
        self._latest_requested_display_revision = None
        # Qt paints the card in logical pixels, while the worker raster is
        # authored at the physical-pixel ratio of the screen containing the
        # TaskConsole.  The console owns screen observation and updates this
        # value; a DPR change is therefore a render-key change even when the
        # displayed dataset revision did not advance.
        self._raster_pixel_ratio = 1.0
        self._pending_interaction_origin = None
        self._pending_interaction_host = None
        # The newest display revision authored by a wheel/pan/rail gesture.
        # An older worker front may still be useful while a button is held,
        # but it cannot settle this intent.  Pulse Preview uses the identical
        # revision-owned answer rule.
        self._pending_interaction_revision = None
        # Pointer-down is itself a presentation transaction boundary.  The
        # board freezes the exact painted origin before motion; this card pins
        # the matching value/Figure/component until release so a live camera
        # cannot advance the semantic owner underneath a drag.
        self._pointer_interaction_pin: _PointerInteractionPin | None = None
        # Bumped by every display-knob edit.  The renderer reads it to tell a
        # genuinely new display from a repeat of the same one.
        self._display_revision = 0
        # The console header's "Selectors" switch state for THIS card (set via
        # ``set_selectors_enabled``; default OFF keeps Monitor display-only).
        # Every plotter (re)build parks its selector layer to this flag (``_apply_selectors_state``),
        # so a fresh figure always inherits the switch instead of coming up live.
        self._selectors_on = False
        # The source-frame key at this panel's last render -- the per-panel multi-rate refresh
        # (see TaskConsole._tick) skips a panel on its beat when nothing new was published
        # since, so a slow panel does not redraw stale data and a fast one only when needed.
        self._render_version = -1
        self._drag_offset: QtCore.QPoint | None = None
        self.setCursor(QtCore.Qt.OpenHandCursor)   # the frame border drags

        holder = QtWidgets.QVBoxLayout(self)
        # The card's content padding is the component-library token CARD_PAD (L/R + bottom); the
        # grey title strip is above (the FluentGroupBox padding-top), and the bottom pad makes the
        # height proportional (see _card_size).  No footer -- the signal source lives in the title.
        holder.setContentsMargins(CARD_PAD, scaled_px(2), CARD_PAD, CARD_PAD)
        holder.setSpacing(0)
        self.canvas_holder = holder
        # The transient status (the Setting tooltip / button colour) and the persistent SIGNAL
        # legend (which node each read comes from) -- the legend goes into the frame TITLE; the
        # per-shot status no longer takes panel space.
        self._status_text = ""
        self._signal_info = ""

        self._build_settings()

        self.setting_button = FluentButton("Setting", color=GREY)
        self.setting_button.setParent(self)
        self.setting_button.setFixedSize(scaled_px(74, minimum=64), scaled_px(26, minimum=22))
        self.setting_button.setToolTip("Panel settings")
        self._settings_anchor = FluentSettingsPopupAnchor(
            self.settings_popup,
            self.setting_button,
        )
        self.setting_button.clicked.connect(self._open_settings)

        self._apply_fixed_size()
        self.set_status("waiting for data…", error=False)

    # ------------------------------------------------------------- geometry
    def _apply_fixed_size(
        self,
        size_name: str | None = None,
        *,
        sync_board: bool = True,
    ) -> None:
        """Apply one size fact to both Fluent chrome and the plot host."""

        resolved_size = self.config.size if size_name is None else str(size_name)
        logical_size = tuple(
            int(value) for value in panel_display_size(resolved_size)
        )
        if sync_board and self.board is not None:
            self.board.set_logical_size(logical_size)
        self.setFixedSize(*_card_size(resolved_size))
        self._place_setting_button()

    def set_raster_pixel_ratio(self, ratio: float) -> bool:
        """Set the Qt-owner screen ratio used by the next worker request."""

        ratio = float(ratio)
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("raster pixel ratio must be positive and finite")
        if ratio == self._raster_pixel_ratio:
            return False
        self._raster_pixel_ratio = ratio
        self._pending_frame = None
        self._pending_faceted_result = None
        self._pending_figure = None
        self._pending_display = None
        self._pending_contract = None
        self._pending_value = None
        self._pending_publication = None
        self._pending_render_request_revision = None
        if self.board is not None:
            self.board.clear()
        return True

    @property
    def raster_pixel_ratio(self) -> float:
        return self._raster_pixel_ratio

    def _place_setting_button(self) -> None:
        if hasattr(self, "setting_button"):
            # top-right, on the title strip (the title kind sits top-left).
            self.setting_button.move(
                self.width() - self.setting_button.width() - scaled_px(8),
                scaled_px(4))
            self.setting_button.raise_()

    # ------------------------------------------------------------- Figure Fit
    def _fit_capable_kind(self) -> bool:
        """Whether this panel can expose the named-axis Figure Fit editor."""

        return self.config.kind in {
            PlotKind.CURVE,
            PlotKind.ROLLING,
            PlotKind.IMAGE,
            PlotKind.HISTOGRAM,
            PlotKind.GRID,
        }

    def make_fit_authoring_pane(
        self,
        parent=None,
        *,
        label_width: int | None = None,
        context_provider=None,
        host_provider=None,
        surface_id: str | None = None,
    ) -> FitAuthoringPane:
        """Create one editor whose execution/result scope is this surface only."""

        if not self._fit_capable_kind():
            raise ValueError("this panel kind does not support Figure Fit")
        live = context_provider is None
        identity = self.panel_id if live else str(surface_id or "").strip()
        if not live and not identity:
            raise ValueError("snapshot Fit requires an explicit surface_id")
        if any(state.surface_id == identity for state in self._fit_surfaces.values()):
            raise ValueError("Fit surface_id is already registered")
        pane = FitAuthoringPane(parent, label_width=label_width)
        seed = self._persisted_fit_spec
        if not live and self._live_fit_pane is not None:
            live_state = self._fit_surfaces.get(self._live_fit_pane)
            if live_state is not None and live_state.spec is not None:
                seed = live_state.spec
        state = _FitSurfaceState(
            pane,
            identity,
            context_provider,
            host_provider,
            spec=seed,
        )
        self._fit_surfaces[pane] = state
        if live:
            if self._live_fit_pane is not None:
                raise RuntimeError("panel already owns its live Fit surface")
            self._live_fit_pane = pane
        pane.fitRequested.connect(
            lambda _revision, spec, owner=pane: self._accept_fit_request(owner, spec)
        )
        pane.fitRequestRejected.connect(
            lambda diagnostic: self.set_status(
                f"Fit request invalid: {diagnostic}",
                error=True,
            )
        )
        pane.clearRequested.connect(lambda owner=pane: self.clear_fit(owner))
        if not self.refresh_fit_authoring_pane(pane):
            pane.set_busy("prepare", draft_ready=False)
        return pane

    def release_fit_authoring_pane(self, pane: FitAuthoringPane) -> None:
        """Retire one snapshot surface without touching any other Fit surface."""

        state = self._fit_surfaces.pop(pane, None)
        if state is None:
            return
        live = pane is self._live_fit_pane
        self.fit_cancel_requested.emit(state.surface_id)
        if live:
            self.fit_output_clear_requested.emit(state.surface_id)
        self._clear_fit_surface_overlay(state)
        if live:
            self._live_fit_pane = None

    def _fit_state(self, surface=None) -> _FitSurfaceState | None:
        if isinstance(surface, FitAuthoringPane):
            return self._fit_surfaces.get(surface)
        if surface is None:
            return self._fit_surfaces.get(self._live_fit_pane)
        identity = str(surface).strip()
        return next(
            (
                state
                for state in self._fit_surfaces.values()
                if state.surface_id == identity
            ),
            None,
        )

    def _live_fit_panes(self) -> tuple[FitAuthoringPane, ...]:
        pane = self._live_fit_pane
        return () if pane is None else (pane,)

    def _fit_surface_host(self, state: _FitSurfaceState):
        if state.pane is self._live_fit_pane:
            return self.board
        provider = state.host_provider
        return None if provider is None else provider()

    def _clear_fit_surface_overlay(self, state: _FitSurfaceState) -> None:
        host = self._fit_surface_host(state)
        if host is not None:
            host.clear_fit_overlays()

    def _present_fit_surface_overlay(
        self,
        state: _FitSurfaceState,
        request: _PanelFitRequest,
        overlays,
    ) -> str | None:
        """Install optional overlay primitives without failing a Fit result."""

        host = self._fit_surface_host(state)
        if host is None:
            return None
        try:
            if overlays is None:
                host.clear_fit_overlays()
                return None
            if request.overlay_figure is None or request.source_frame is None:
                host.clear_fit_overlays()
                return "overlay has no single-panel source front"
            status = host.install_fit_overlays(
                request.overlay_figure,
                request.source_frame,
                overlays,
            )
        except Exception as error:
            host.clear_fit_overlays()
            detail = " ".join(str(error).split()) or type(error).__name__
            return f"{type(error).__name__}: {detail}"
        return None if status != "INCOMPATIBLE" else "surface projection changed"

    @staticmethod
    def _fit_overlay_key(overlay_figure, source_frame):
        """Return the exact single-panel projection key, or no overlay scope."""

        if overlay_figure is None or source_frame is None:
            return None
        try:
            return overlay_figure._transient_fit_projection_key(source_frame)
        except Exception:
            return None

    def _fit_authority_key(
        self,
        value,
        figure,
        histogram_projection,
    ):
        """Small semantic identity whose change invalidates a Fit scope.

        Source revision and viewport are deliberately absent: Monitor Fit is
        live/latest and a viewport is presentation-only.  View bindings,
        resolved selectors, Area authority, schema, and exact Histogram edges
        do change the physical Fit problem and therefore end the old scope.
        """

        layer = figure.document.layers[0]
        evaluated = figure.evaluated.layers[0]
        histogram_key = None
        if histogram_projection is not None:
            histogram_key = (
                int(histogram_projection.requested_bin_count),
                tuple(float(value) for value in histogram_projection.bin_edges),
            )
        return (
            value.snapshot.ref.schema_fingerprint,
            layer.view,
            tuple(
                (resolution.source, resolution.selector, resolution.index)
                for resolution in evaluated.resolutions
            ),
            self._fit_selection_for_value(value),
            histogram_key,
        )

    def _fit_context_for_pane(self, pane: FitAuthoringPane):
        """Return one exact publication/Figure pair owned by this surface."""

        state = self._fit_surfaces.get(pane)
        if state is None:
            return None
        provider = state.context_provider
        if provider is None:
            value = self._presented_value
            figure = self._presented_figure
            publication = self._presented_publication
            payload = self.frozen_render_payload()
        else:
            context = provider()
            if context is None:
                return None
            if not isinstance(context, tuple) or len(context) not in (3, 4):
                raise TypeError(
                    "Fit context provider must return "
                    "(value, figure, publication[, rendered payload])"
                )
            value, figure, publication = context[:3]
            payload = context[3] if len(context) == 4 else None
        snapshot = getattr(value, "snapshot", None)
        if snapshot is None or figure is None:
            return None
        _require_publication_value(publication, value)
        from zlc_frontend import DataFigure

        if not isinstance(figure, DataFigure):
            raise TypeError("Fit context must carry one DataFigure")
        entries = tuple(figure.datasets.entries)
        if len(entries) != 1 or entries[0].snapshot.ref != snapshot.ref:
            raise ValueError("Fit context Figure belongs to another source revision")
        histogram_projection = None
        if figure.document.layers[0].view.intent is ViewIntent.HISTOGRAM:
            if not isinstance(payload, HistogramPanelPayload):
                raise ValueError(
                    "Histogram Fit requires the exact rendered bin projection"
                )
            histogram_projection = payload.bin_projection
        host = self._fit_surface_host(state)
        source_frame = None if host is None else host.front_frame
        surface_context = None if host is None else host.context
        overlay_figure = (
            figure
            if surface_context is None or surface_context.selector_figure is None
            else surface_context.selector_figure
        )
        return (
            value,
            figure,
            overlay_figure,
            publication,
            histogram_projection,
            source_frame,
        )

    def _prepared_fit_options(
        self,
        state: _FitSurfaceState,
        *,
        seed,
    ):
        context = self._fit_context_for_pane(state.pane)
        if context is None:
            raise RuntimeError("Fit source is not currently available")
        (
            value,
            figure,
            _overlay_figure,
            _publication,
            histogram_projection,
            _source_frame,
        ) = context
        if (
            seed is not None
            and seed.committed_transform.source_schema_fingerprint
            != value.snapshot.ref.schema_fingerprint
        ):
            seed = None
        authority_key = self._fit_authority_key(
            value,
            figure,
            histogram_projection,
        )
        authoring_key = (authority_key, seed)
        if state.authoring_key == authoring_key and state.authoring_options:
            return context, state.authoring_options, seed, authority_key
        from zlc_frontend import prepare_fit_authoring_options

        options = prepare_fit_authoring_options(
            figure,
            self._fit_selection_for_value(value),
            seed_spec=seed,
            histogram_projection=histogram_projection,
        )
        state.authoring_key = authoring_key
        state.authoring_options = options
        return context, options, seed, authority_key

    def refresh_fit_authoring_pane(self, pane: FitAuthoringPane) -> bool:
        """Reconcile one stable editor against only its own exact surface."""

        state = self._fit_surfaces.get(pane)
        if state is None:
            return False
        try:
            context, options, seed, authority_key = self._prepared_fit_options(
                state,
                seed=state.spec,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            if pane.fit_models:
                pane.clear_options()
            pane.set_busy("prepare", draft_ready=False)
            if hasattr(self, "setting_button"):
                self.set_status(f"Fit unavailable: {error}", error=True)
            return False
        value = context[0]
        previous_snapshot = getattr(getattr(state.source, "snapshot", None), "ref", None)
        if (
            pane is not self._live_fit_pane
            and previous_snapshot is not None
            and previous_snapshot != value.snapshot.ref
        ):
            self._cancel_fit_state(
                state,
                drop_spec=False,
                clear_options=False,
            )
        if state.authority_key is not None and state.authority_key != authority_key:
            self._cancel_fit_state(
                state,
                drop_spec=False,
                clear_options=False,
            )
        selected = None if seed is None else seed.model_id
        if selected not in {option.spec.model_id for option in options}:
            selected = None
        pane.reconcile_options(options, selected_model=selected)
        result = state.result
        pane.set_busy(
            None,
            draft_ready=bool(
                result is not None and result.source_ref == value.snapshot.ref
            ),
        )
        return True

    def _fit_selection_for_value(self, value):
        """Return the Area authority only when it belongs to this exact value."""

        from zlc_data import Selection

        commit = self._figure_output_authority.area_commit
        if (
            commit is not None
            and isinstance(commit.authority, Selection)
            and self._figure_commit_matches_value(commit, value)
        ):
            return commit.authority
        return None

    def _fit_selection_changed(self) -> None:
        """Refresh only visible authoring panes after an Area commit."""

        for state in tuple(self._fit_surfaces.values()):
            if state.pane.isVisible():
                self.refresh_fit_authoring_pane(state.pane)

    def _sync_fit_authoring_from_presented(
        self,
        *,
        prepare_authoring: bool,
    ) -> None:
        """Refresh the live editor; never retarget or pause its running scope."""

        if not isinstance(prepare_authoring, bool):
            raise TypeError("prepare_authoring must be bool")
        state = self._fit_state()
        if state is not None and prepare_authoring:
            self.refresh_fit_authoring_pane(state.pane)

    def _accept_fit_request(self, pane: FitAuthoringPane, spec) -> None:
        """Promote one pane draft to an exact surface-local Fit command."""

        from zlc_data import FitSpec
        from zlc_data.fit import fit_spec_to_tree

        state = self._fit_surfaces.get(pane)
        if state is None or not isinstance(spec, FitSpec):
            return
        try:
            context, options, _seed, authority_key = self._prepared_fit_options(
                state,
                seed=spec,
            )
            option = next(
                item for item in options if item.spec.model_id == spec.model_id
            )
            from zlc_frontend import fit_spec_from_arguments

            exact_spec = fit_spec_from_arguments(option, pane.arguments_text)
        except (StopIteration, TypeError, ValueError, RuntimeError) as error:
            self.set_status(f"Fit request invalid: {error}", error=True)
            return
        (
            source,
            figure,
            overlay_figure,
            publication,
            _histogram_projection,
            source_frame,
        ) = context
        state.request_revision += 1
        self.fit_cancel_requested.emit(state.surface_id)
        # An explicit command replaces this surface's committed operation.
        # Its old EVENT_RESULT must not remain labelled as the new model/spec.
        # Ordinary live/latest revision advance does not cross this edge and
        # therefore may retain the last exact, provenance-bearing result until
        # the same authority produces its replacement.
        if pane is self._live_fit_pane:
            self.fit_output_clear_requested.emit(state.surface_id)
        self._clear_fit_surface_overlay(state)
        state.spec = exact_spec
        state.source = source
        state.publication = publication
        state.figure = figure
        state.overlay_figure = overlay_figure
        state.source_frame = source_frame
        state.result = None
        state.authority_key = authority_key
        state.requested_ref = None
        state.requested_overlay_key = None
        if pane is self._live_fit_pane:
            self._persisted_fit_spec = exact_spec
            self._commit_persisted_params(
                {_FIT_SPEC_PARAM: fit_spec_to_tree(exact_spec)}
            )
        pane.set_busy("fit", draft_ready=False)
        self.fit_requested.emit(pane)

    def _queue_live_fit(self) -> None:
        """Submit the latest presented front without pinning the base raster."""

        state = self._fit_state()
        if state is None or state.spec is None:
            return
        try:
            context = self._fit_context_for_pane(state.pane)
        except (TypeError, ValueError, RuntimeError) as error:
            self.clear_fit(state.pane)
            self.set_status(f"Fit stopped: {error}", error=True)
            return
        if context is None:
            return
        (
            source,
            figure,
            overlay_figure,
            publication,
            histogram_projection,
            source_frame,
        ) = context
        snapshot = source.snapshot
        if (
            state.spec.committed_transform.source_schema_fingerprint
            != snapshot.ref.schema_fingerprint
        ):
            self.clear_fit(state.pane)
            return
        _require_publication_value(publication, source)
        authority_key = self._fit_authority_key(
            source,
            figure,
            histogram_projection,
        )
        if state.authority_key != authority_key:
            self.clear_fit(state.pane)
            self.set_status(
                "Fit stopped because its data/view authority changed",
                error=False,
            )
            return
        overlay_key = self._fit_overlay_key(overlay_figure, source_frame)
        state.source = source
        state.publication = publication
        state.figure = figure
        state.overlay_figure = overlay_figure
        state.source_frame = source_frame
        if state.requested_ref == snapshot.ref:
            if state.requested_overlay_key == overlay_key:
                return
            if state.result is None or state.result.source_ref != snapshot.ref:
                return
        state.pane.set_busy("fit", draft_ready=state.result is not None)
        self.fit_requested.emit(state.pane)

    def freeze_fit_request(self, surface=None) -> _PanelFitRequest | None:
        """Freeze one exact immutable operation for the composition scheduler."""

        state = self._fit_state(surface)
        if state is None:
            return None
        source = state.source
        snapshot = None if source is None else getattr(source, "snapshot", None)
        if (
            state.spec is None
            or snapshot is None
            or state.figure is None
            or state.publication is None
            or state.spec.committed_transform.source_schema_fingerprint
            != snapshot.ref.schema_fingerprint
        ):
            return None
        overlay_key = self._fit_overlay_key(
            state.overlay_figure,
            state.source_frame,
        )
        cached_result = (
            state.result
            if state.result is not None
            and state.result.source_ref == snapshot.ref
            and state.result.spec == state.spec
            else None
        )
        state.request_revision += 1
        request = _PanelFitRequest(
            self.panel_id,
            state.surface_id,
            state.request_revision,
            source,
            state.publication,
            state.figure,
            state.overlay_figure,
            state.source_frame,
            state.spec,
            cached_result,
        )
        state.requested_ref = snapshot.ref
        state.requested_overlay_key = overlay_key
        return request

    def accept_fit_completion(
        self,
        request: _PanelFitRequest,
        result,
        overlays,
        error: str | None,
        overlay_error: str | None = None,
        *,
        summary: str | None = None,
    ) -> bool:
        """Accept a result only into the surface that submitted its exact pair."""

        if not self._fit_completion_is_current(request):
            return False
        state = self._fit_state(request.surface_id)
        if state is None:
            return False
        from zlc_data import FitResultBatch

        if error is not None:
            if request.request_revision == state.request_revision:
                state.requested_ref = None
                state.requested_overlay_key = None
                state.pane.set_busy(None, draft_ready=state.result is not None)
            self.set_status(f"Fit failed: {error}", error=True)
            return True
        if not isinstance(result, FitResultBatch) or result.source_ref != request.source.snapshot.ref:
            self.set_status("Fit worker returned another source revision", error=True)
            return True
        state.result = result
        if request.request_revision == state.request_revision:
            state.pane.set_busy(None, draft_ready=True)
        presentation_error = self._present_fit_surface_overlay(
            state,
            request,
            overlays,
        )
        fault = overlay_error or presentation_error
        if not isinstance(summary, str) or not summary.strip():
            self.set_status("Fit worker returned no result summary", error=True)
            return True
        message = summary
        if fault:
            message += f"; overlay unavailable ({fault})"
        self.set_status(message, error=False)
        if (
            request.request_revision == state.request_revision
            and state.pane is self._live_fit_pane
        ):
            QtCore.QTimer.singleShot(0, self._queue_live_fit)
        return True

    def _fit_completion_is_current(self, request: _PanelFitRequest) -> bool:
        """Whether composition may commit this exact surface operation."""

        if not isinstance(request, _PanelFitRequest):
            return False
        state = self._fit_state(request.surface_id)
        return bool(
            state is not None
            and request.panel_id == self.panel_id
            and not request.cancelled.is_set()
            and state.spec == request.spec
        )

    def _cancel_fit_state(
        self,
        state: _FitSurfaceState,
        *,
        drop_spec: bool,
        clear_options: bool,
        notify_output: bool = True,
    ) -> bool:
        had_state = state.spec is not None or state.result is not None
        state.request_revision += 1
        self.fit_cancel_requested.emit(state.surface_id)
        if notify_output and state.pane is self._live_fit_pane:
            self.fit_output_clear_requested.emit(state.surface_id)
        self._clear_fit_surface_overlay(state)
        if drop_spec:
            state.spec = None
        state.source = None
        state.publication = None
        state.figure = None
        state.overlay_figure = None
        state.source_frame = None
        state.result = None
        state.authority_key = None
        state.requested_ref = None
        state.requested_overlay_key = None
        if drop_spec or clear_options:
            state.authoring_key = None
            state.authoring_options = ()
        if clear_options and state.pane.fit_models:
            state.pane.clear_options()
        state.pane.set_busy(
            "prepare" if clear_options else None,
            draft_ready=False,
        )
        return had_state

    def _retire_all_fits(
        self,
        *,
        emit_changed: bool,
        notify_output: bool = True,
    ) -> bool:
        """Retire every surface when the bound signal generation disappears."""

        had_state = False
        for state in tuple(self._fit_surfaces.values()):
            had_state = self._cancel_fit_state(
                state,
                drop_spec=True,
                clear_options=True,
                notify_output=notify_output,
            ) or had_state
        self._persisted_fit_spec = None
        self._commit_persisted_params(
            remove=(_FIT_SPEC_PARAM,),
            emit_changed=emit_changed,
        )
        return had_state

    def clear_fit(self, pane: FitAuthoringPane | None = None) -> None:
        """Clear only the requesting surface; other Fit scopes keep running."""

        state = self._fit_state(pane)
        if state is None:
            return
        had_state = self._cancel_fit_state(
            state,
            drop_spec=True,
            clear_options=False,
        )
        if state.pane is self._live_fit_pane:
            self._persisted_fit_spec = None
            self._commit_persisted_params(remove=(_FIT_SPEC_PARAM,))
        if had_state:
            self.set_status("Fit cleared", error=False)

    # ------------------------------------------------------------- settings UI

    def _make_param_widget(
        self,
        spec: FormFieldProps,
        current,
        *,
        apply,
    ) -> QtWidgets.QWidget:
        """One widget per declarative form field with a semantic commit edge.

        ``apply`` overrides where the edit goes (default ``self._set_param``); the
        Edit tab passes its own callback.  Choice/toggle activation is already a
        complete user command.  Numeric spins disable keyboard tracking so an
        in-progress token remains local until Qt commits it.  Free text is a
        local draft until ``editingFinished``; no character can start a render.
        """

        handler = FORM_WIDGET_HANDLERS[spec.kind]
        holder = {}

        def apply_current() -> None:
            widget = holder.get("widget")
            if widget is None:
                return
            try:
                value = handler.read(spec, widget)
            except (TypeError, ValueError) as exc:
                self.set_status(
                    f"Invalid {spec.label}: {exc}",
                    error=True,
                )
                return
            try:
                apply(spec.key, value)
            except (TypeError, ValueError, RuntimeError) as exc:
                self.set_status(
                    f"Invalid {spec.label}: {exc}",
                    error=True,
                )

        if spec.kind == "text":
            widget = handler.build(spec, current, lambda: None)
            holder["widget"] = widget
            widget.editingFinished.connect(apply_current)
            return widget
        widget = handler.build(spec, current, apply_current)
        holder["widget"] = widget
        if isinstance(widget, QtWidgets.QAbstractSpinBox):
            widget.setKeyboardTracking(False)
        return widget

    def _emit_param_rows(
        self,
        specs,
        values,
        add,
        apply,
        label_w,
        *,
        parent,
    ) -> dict:
        """Render each declarative field in ``specs`` as a ``[label | control]`` row through the
        same frontend form-handler path the measurement form uses, appending it via the
        ``add`` callback.  Returns ``{key: widget}`` so a caller can keep a named back-reference.  BOTH
        the Setting popup AND the Edit tab call this for a plot's display knobs, so adding a plot
        field shows up in both surfaces with no hand-wiring."""
        out = {}
        for spec in specs:
            widget = self._make_param_widget(
                spec,
                values[spec.key],
                apply=apply,
            )
            out[spec.key] = widget
            add(
                FluentSettingRow(
                    spec.label,
                    widget,
                    label_width=label_w,
                    parent=parent,
                )
            )
        return out

    def _display_contract_for_authoring(self):
        contract = self._presented_contract or self._pending_contract
        if contract is not None and self.config.kind is not PlotKind.GRID:
            return contract
        view = None
        if self.config.kind is not PlotKind.SITE_MAP:
            from zlc_frontend.plot_panel import plot_panel_view_from_params

            view = plot_panel_view_from_params(
                self.config.kind,
                self.config.params,
            )
            if self.config.kind is PlotKind.GRID and view is None:
                return contract
        if contract is not None and contract.figure.view == view:
            return contract
        return self._plot_panel_contract(view, value_name="Signal")

    def _display_form_spec(self):
        contract = self._display_contract_for_authoring()
        if contract is None:
            return None
        return panel_display_form_spec(
            self.config.kind,
            cell_intent=(
                contract.figure.view_intent
                if self.config.kind is PlotKind.GRID
                else None
            ),
        )

    def _display_form_values(self):
        contract = self._display_contract_for_authoring()
        if contract is None:
            return {}
        state = self._display_state(contract)
        return panel_display_form_values(
            self.config.kind,
            state,
            rolling_distribution=contract.figure.rolling_distribution,
        )

    def _make_display_form_surface(self, apply, label_w, *, parent):
        """Place one canonical frontend display form in a stable surface."""

        surface = QtWidgets.QWidget(parent)
        layout = QtWidgets.QVBoxLayout(surface)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(scaled_px(6, minimum=4))
        surface._zlc_display_apply = apply
        surface._zlc_display_label_width = label_w
        surface._zlc_display_signature = None
        surface._zlc_display_fields = {}
        surface._zlc_display_widgets = {}
        self._reconcile_display_form_surface(surface)
        return surface

    def _reconcile_display_form_surface(self, surface) -> None:
        """Replace only the form subtree when a Grid changes cell intent."""

        if surface is None:
            return
        spec = self._display_form_spec()
        signature = None if spec is None else spec.fields
        if signature != surface._zlc_display_signature:
            layout = surface.layout()
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            surface._zlc_display_signature = signature
            surface._zlc_display_fields = (
                {} if spec is None else {field.key: field for field in spec.fields}
            )
            surface._zlc_display_widgets = {}
            if spec is not None:
                surface._zlc_display_widgets = self._emit_param_rows(
                    spec.fields,
                    self._display_form_values(),
                    layout.addWidget,
                    surface._zlc_display_apply,
                    surface._zlc_display_label_width,
                    parent=surface,
                )
            surface.setVisible(spec is not None)
            surface.updateGeometry()
        self._refresh_display_form_surface(surface)

    def _refresh_display_form_surface(self, surface) -> None:
        if surface is None or surface._zlc_display_signature is None:
            return
        values = self._display_form_values()
        for key, widget in surface._zlc_display_widgets.items():
            field = surface._zlc_display_fields[key]
            with _signals_blocked(widget):
                FORM_WIDGET_HANDLERS[field.kind].write(
                    field,
                    widget,
                    values[key],
                )

    def _make_unit_readout_row(self, label_w, *, parent):
        """Build the shared read-only producer-unit row for Setting and Edit."""

        label = FluentLabel(self._current_unit_text())
        label.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        label.setToolTip(
            "Declared by the producer schema. Display panels do not rewrite units."
        )
        row = FluentSettingRow(
            "unit",
            label,
            label_width=label_w,
            parent=parent,
        )
        return row, label

    def refresh_on_show(self) -> None:
        """Re-seed every Setting control from ``config.params`` -- the SINGLE source of truth for a
        panel's params -- so the Setting popup shows the CURRENT values whenever it opens, even if they
        were changed elsewhere (the Edit tab writes the same config.params).  Each widget is re-seeded
        through its form handler's ``write`` (one entry point, no per-key handwiring), with its
        change signals blocked so re-seeding does not re-fire ``_set_param`` (which would enqueue a
        duplicate compose).  A control is a view of config.params, refreshed on show, never a
        private copy that drifts from the other surface."""
        self._reconcile_display_form_surface(self.display_form_surface)
        self._refresh_view_spec_controls()
        self._refresh_unit_readout()

    def _build_plot(self) -> None:
        """Give this card its raster surface.

        A panel shows PIXELS the worker produced -- an already-coherent
        BoardFrame handed to the frontend-owned ``FigureSurfaceHost``, which
        paints from immutable bytes and owns the exact Figure context.  The card
        therefore holds no Matplotlib object at all: rendering happens off the
        GUI thread and arrives here already rasterised.  The host is the ONE
        selector owner; ``FrozenRasterView`` stays a frozen-report presenter.
        The console card's selector switch and completed gestures therefore go
        through the same binding as every other interactive Figure window.
        """

        if self.board is not None:
            return
        self.board = FigureSurfaceHost(
            self.panel_id,
            faceted=self.config.kind is PlotKind.GRID,
            empty_text=(
                "choose a facet axis in Setting"
                if self.config.kind is PlotKind.GRID
                else "waiting for data"
            ),
            output_authority=self._figure_output_authority,
        )
        if self.config.kind is PlotKind.GRID:
            self.board.focusRequested.connect(self._focus_grid_cell)
            self.board.overviewRequested.connect(self._return_to_grid_overview)
        self.board.figureOutputsChanged.connect(
            self._figure_surface_outputs_changed
        )
        self.board.interactionRejected.connect(
            lambda detail: self.set_status(detail, error=True)
        )
        self._apply_fixed_size()
        # The pulse-preview answer protocol, verbatim: a wheel-zoom / pan /
        # double-middle commit is answered by re-composing THIS card at the
        # candidate's view under the candidate's revision; a clim-rail commit
        # routes through the ONE fixed-limits writer the Setting inputs use.
        self.board.viewCommitted.connect(self._on_view_committed)
        self.board.colorLimitsCommitted.connect(self._on_color_limits_committed)
        self.board.thresholdsCommitted.connect(
            self._on_histogram_thresholds_committed
        )
        self.board.interactionStarted.connect(
            lambda origin, host=self.board: self._begin_pointer_interaction(
                host,
                origin,
                value=self._presented_value,
                publication=self._presented_publication,
                surface_id=self.panel_id,
            )
        )
        self.board.interactionFinished.connect(
            lambda host=self.board: self._finish_pointer_interaction(host)
        )
        # The console switch may have been armed before this card received its
        # first frame.  Replay the card-owned state onto the newly-created host;
        # otherwise the visible switch says ON while this first surface stays
        # inert until the operator toggles it twice.
        self._apply_selectors_state()
        self.canvas_holder.addWidget(self.board)

    def _figure_surface_outputs_changed(self) -> None:
        """Publish the one frontend-owned Area/Cross command state."""

        self.figure_outputs_changed.emit()
        self._fit_selection_changed()

    def _clear_figure_outputs(self, *, notify: bool) -> None:
        self._figure_output_authority.clear(notify=notify)
        if not notify:
            self._fit_selection_changed()

    @staticmethod
    def _figure_commit_matches_value(commit, value) -> bool:
        if commit is None or value is None:
            return False
        snapshot = getattr(value, "snapshot", None)
        if snapshot is None:
            return False
        from zlc_frontend.figure_outputs import source_identity_matches_snapshot

        return source_identity_matches_snapshot(commit.source_identity, snapshot)

    def frozen_figure_output_state(self):
        """Return selector intents with their exact visible publication/value.

        Area and Cross are independent continuous routes.  This method exposes
        only their frozen declarations and the exact neutral publication that
        owns the painted value; it neither evaluates nor bundles their arrays.
        """

        value = self._presented_value
        publication = self._presented_publication
        if value is not None:
            _require_publication_value(publication, value)
        authority = self._figure_output_authority
        area_commit = (
            authority.area_commit
            if self._figure_commit_matches_value(authority.area_commit, value)
            else None
        )
        cross_commit = (
            authority.cross_commit
            if self._figure_commit_matches_value(authority.cross_commit, value)
            else None
        )
        return publication, value, area_commit, cross_commit

    def has_live_selector_outputs(self) -> bool:
        """Whether a generation-scoped selector must follow source revisions."""

        return (
            self._figure_output_authority.area_commit is not None
            or self._figure_output_authority.cross_commit is not None
        )

    def _focus_grid_cell(self, panel_index: int, address) -> None:
        """Show one exact cell from the currently painted coherent overview."""

        if self.config.kind is not PlotKind.GRID:
            return
        from zlc_frontend.panel_render import FacetedPanelFocus

        focus = FacetedPanelFocus(int(panel_index), address)
        if focus == self._grid_focus:
            return
        self._grid_focus = focus
        self._clear_display_view(request_render=False)
        self._request_display_render()

    def _return_to_grid_overview(self) -> None:
        """Return to the same typed grid without replacing its Qt host."""

        if self.config.kind is not PlotKind.GRID or self._grid_focus is None:
            return
        self._grid_focus = None
        self._clear_display_view(request_render=False)
        self._request_display_render()

    def freeze_render_request(
        self,
        snapshot,
        frame_key,
        *,
        publication: SignalPublication,
        force: bool = False,
        axis_labels=None,
        short_labels=None,
    ) -> _PanelRenderRequest | None:
        """Freeze one worker request without exposing mutable Qt/card state.

        Repeated timer ticks for the same source/display signature are folded
        here, before they can become executor work.  ``force`` is reserved for
        the visible Refresh action and still creates exactly one request.
        """

        name = self.config.signal
        if not name:
            self.set_status("pick a signal in Setting", error=False)
            return None
        value = None if snapshot is None else snapshot.value(name)
        if value is None or getattr(value, "snapshot", None) is None:
            self.set_status(f"waiting for {name}", error=False)
            return None
        _require_publication_value(publication, value)
        if self._live_surface_interaction_pending():
            # One selector transaction is defined on the exact immutable
            # front the operator saw.  Remember the newest live value for the
            # next ordinary tick, but never let it replace the interaction's
            # pinned render request or splice a new input revision into a drag.
            self._remember_candidate_value(
                value,
                publication=publication,
            )
            return None
        return self._freeze_value_render_request(
            value,
            frame_key,
            force=force,
            axis_labels=axis_labels,
            short_labels=short_labels,
            publication=publication,
        )

    def _remember_candidate_value(
        self,
        value,
        *,
        publication: SignalPublication,
    ) -> None:
        """Retain the newest exact publication/value pair as one fact."""

        _require_publication_value(publication, value)
        candidate = self._candidate_publication
        advances = (
            candidate is None
            or candidate.event_ref.stream_id != publication.event_ref.stream_id
            or candidate.event_ref.generation != publication.event_ref.generation
            or candidate.event_ref.sequence <= publication.event_ref.sequence
        )
        if advances:
            self._candidate_value = value
            self._candidate_publication = publication

    def _retained_publication_for_value(self, value) -> SignalPublication | None:
        """Return the exact private publication retained beside ``value``."""

        pairs = (
            (self._candidate_value, self._candidate_publication),
            (self._pending_value, self._pending_publication),
            (self._presented_value, self._presented_publication),
        )
        pin = self._pointer_interaction_pin
        if pin is not None:
            pairs += ((pin.value, pin.publication),)
        for retained_value, publication in pairs:
            if retained_value is value and publication is not None:
                return _require_publication_value(publication, value)
        return None

    def freeze_current_view_request(
        self,
        *,
        force: bool = False,
        axis_labels=None,
        short_labels=None,
    ) -> _PanelRenderRequest | None:
        """Freeze a pure view edit against the already accepted data front.

        A selector/display/title/size commit is not a data-acquisition boundary.
        It must not advance ``SignalDataPlane`` merely because the operator
        moved a control.  A source rebind whose selected name differs from the
        accepted value waits for the next base tick or explicit Refresh.
        """

        name = self.config.signal
        if self._live_surface_interaction_pending():
            # A held pointer gesture edits the exact data front the operator
            # can still see.  A newer live-camera completion may already be in
            # ``_candidate_value`` while its raster is queued or while the held
            # front deliberately remains painted.  Advancing to that value
            # here would splice two input identities into one gesture.
            value = self._presented_value
        elif force and self._candidate_value is not None:
            value = self._candidate_value
        else:
            value = self._presented_value
        if (
            not name
            or value is None
            or str(getattr(value, "name", "")) != str(name)
            or getattr(value, "snapshot", None) is None
        ):
            if name:
                self.set_status(f"waiting for {name}", error=False)
            else:
                self.set_status("pick a signal in Setting", error=False)
            return None
        publication = self._retained_publication_for_value(value)
        if publication is None:
            self.set_status(
                "painted Figure source has no exact signal publication",
                error=True,
            )
            return None
        return self._freeze_value_render_request(
            value,
            self._render_version,
            force=force,
            axis_labels=axis_labels,
            short_labels=short_labels,
            publication=publication,
        )

    def _freeze_value_render_request(
        self,
        value,
        frame_key,
        *,
        publication: SignalPublication,
        force: bool,
        axis_labels=None,
        short_labels=None,
    ) -> _PanelRenderRequest | None:
        """Freeze one immutable value/display pair for the raster worker."""

        schema = value.snapshot.block.schema
        try:
            leaf_figure, source = self._plot_panel_source(
                value.snapshot,
                value,
                publication,
            )
        except (TypeError, ValueError) as error:
            self.set_status(str(error), error=True)
            return None
        previous_schema = self._current_schema()
        self._remember_candidate_value(
            value,
            publication=publication,
        )
        schema_transition = (
            previous_schema is not None
            and previous_schema.fingerprint != schema.fingerprint
        )
        if previous_schema is None or schema_transition:
            # The candidate is now the only schema owner for an initially
            # blank card.  Reconcile the one canonical Setting editor before
            # resolving the view so an already-open popup exposes the same
            # typed choices as a popup opened after the first publication.
            self._refresh_view_spec_controls()
        declared_default = (
            None
            if leaf_figure is None or self.config.kind is PlotKind.SITE_MAP
            else leaf_figure.view
        )
        view = self._effective_view_spec(
            schema,
            declared_default=declared_default,
        )
        if self.config.kind is not PlotKind.SITE_MAP and view is None:
            self.set_status(
                "the declared axes cannot form a complete typed view",
                error=False,
            )
            return None
        size_name = self.config.size
        if self._initial_grid_size_pending:
            from zlc_frontend.plot_layout import optimal_grid_size_for_view

            try:
                size_name = optimal_grid_size_for_view(schema, view)
            except (TypeError, ValueError) as error:
                self.set_status(
                    f"Grid initial size unavailable: {error}",
                    error=True,
                )
                return None
        contract = self._plot_panel_contract(
            view,
            value_name=str(value.name),
            axis_labels=axis_labels,
            short_labels=short_labels,
            size_name=size_name,
            pixel_ratio=self._raster_pixel_ratio,
            leaf_figure=leaf_figure,
        )
        display = self._display_state(
            contract,
            reset_view=schema_transition,
        )
        focus = (
            self._grid_focus
            if self.config.kind is PlotKind.GRID and not schema_transition
            else None
        )
        next_revision = self._render_request_revision + 1
        request = self._build_surface_render_request(
            value,
            frame_key,
            request_revision=next_revision,
            display=display,
            contract=contract,
            focus=focus,
            source=source,
        )
        if not force and request.signature == self._requested_signature:
            return None
        self._render_request_revision = next_revision
        self._requested_signature = request.signature
        self._latest_requested_source_ref = value.snapshot.ref
        self._latest_requested_source_key = request.source_key
        self._latest_requested_display_revision = int(display.revision)
        return request

    def freeze_surface_request(
        self,
        value,
        *,
        surface_id: str,
        request_revision: int,
        frame_key,
        axis_labels=None,
        short_labels=None,
        display=None,
        publication: SignalPublication,
    ) -> _PanelRenderRequest:
        """Freeze an additional presentation of one already accepted value.

        The Edit tab uses this seam for viewport-only recomposition.  All
        Figure semantics still come from this card's one declared panel/view
        state, while the caller owns only its immutable input identity and its
        surface-local request ordering.  This method never advances the live
        card's candidate/presented source.
        """

        if not surface_id or str(surface_id) == self.panel_id:
            raise ValueError("secondary render surface needs a distinct id")
        if int(request_revision) < 1:
            raise ValueError("render request revision must be positive")
        snapshot = getattr(value, "snapshot", None)
        if snapshot is None:
            raise TypeError("render surface value must carry an OwnedSnapshot")
        _require_publication_value(publication, value)
        schema = snapshot.block.schema
        leaf_figure, source = self._plot_panel_source(
            snapshot,
            value,
            publication,
        )
        declared_default = (
            None
            if leaf_figure is None or self.config.kind is PlotKind.SITE_MAP
            else leaf_figure.view
        )
        view = self._effective_view_spec(
            schema,
            declared_default=declared_default,
        )
        if self.config.kind is not PlotKind.SITE_MAP and view is None:
            raise ValueError("dataset surface needs a complete typed view")
        contract = self._plot_panel_contract(
            view,
            value_name=str(value.name),
            axis_labels=axis_labels,
            short_labels=short_labels,
            size_name=self.config.size,
            pixel_ratio=self._raster_pixel_ratio,
            leaf_figure=leaf_figure,
        )
        if display is None:
            display = self._display_state(contract)
        return self._build_surface_render_request(
            value,
            frame_key,
            request_revision=int(request_revision),
            display=display,
            contract=contract,
            focus=self._grid_focus if self.config.kind is PlotKind.GRID else None,
            source=source,
            surface_id=str(surface_id),
        )

    def _build_surface_render_request(
        self,
        value,
        frame_key,
        *,
        request_revision: int,
        display,
        contract,
        focus,
        source,
        surface_id: str | None = None,
    ) -> _PanelRenderRequest:
        """Freeze one frontend-owned presentation for either Qt surface."""

        source_key = (
            str(value.name),
            contract.session_identity,
            source.session_identity,
        )
        signature = (
            frame_key,
            source_key,
            display,
            focus,
        )
        return _PanelRenderRequest(
            self.panel_id,
            int(request_revision),
            signature,
            source_key,
            frame_key,
            value,
            contract,
            source,
            display,
            focus,
            surface_id=surface_id,
        )

    def accept_render_result(
        self,
        request: _PanelRenderRequest,
        *,
        frame=None,
        faceted_result=None,
        figure=None,
        error: str | None = None,
    ) -> bool:
        """Accept any useful completed front from the current source generation.

        Source and display revisions remain the presentation authority.  The
        surface-local request revision orders distinct answers that deliberately
        share both of those identities.  A completed immutable source revision remains a real front
        while later work is waiting in the capacity-one lane, so show it unless
        an equal/newer answer is already pending or painted.  Structural rebinds
        and generation replacement still reject a superseded result.
        """

        publication = self._retained_publication_for_value(request.value)
        if publication is None:
            return False
        source_ref = request.value.snapshot.ref
        latest_ref = self._latest_requested_source_ref
        if (
            latest_ref is None
            or request.contract.pixel_ratio != self._raster_pixel_ratio
            or request.source_key != self._latest_requested_source_key
            or source_ref.block_id != latest_ref.block_id
            or source_ref.stream_generation != latest_ref.stream_generation
            or source_ref.schema_fingerprint != latest_ref.schema_fingerprint
        ):
            return False
        pointer_pin = self._pointer_interaction_pin
        pinned_interaction = (
            pointer_pin.origin
            if pointer_pin is not None and pointer_pin.host is self.board
            else (
                self._pending_interaction_origin
                if self._pending_interaction_host is self.board
                else None
            )
        )
        if pinned_interaction is not None:
            pending_ref = getattr(
                pinned_interaction.input_identity,
                "ref",
                None,
            )
            if pending_ref is None or source_ref != pending_ref:
                self._settle_pending_interaction_through(
                    request.display.revision,
                    failed=True,
                    answer_host=self.board,
                )
                return False
        if error is not None and (
            source_ref.revision != latest_ref.revision
            or int(request.display.revision)
            != self._latest_requested_display_revision
        ):
            return False
        for value, display, render_revision in (
            (
                self._pending_value,
                self._pending_display,
                self._pending_render_request_revision,
            ),
            (
                self._presented_value,
                self._presented_display,
                self._presented_render_request_revision,
            ),
        ):
            existing_ref = getattr(getattr(value, "snapshot", None), "ref", None)
            if (
                existing_ref is None
                or existing_ref.stream_generation != source_ref.stream_generation
            ):
                continue
            if existing_ref.revision > source_ref.revision:
                return False
            if (
                existing_ref.revision == source_ref.revision
                and display is not None
            ):
                existing_display_revision = int(display.revision)
                requested_display_revision = int(request.display.revision)
                if existing_display_revision > requested_display_revision:
                    return False
                if (
                    existing_display_revision == requested_display_revision
                    and render_revision is not None
                    and int(render_revision) >= int(request.request_revision)
                ):
                    return False
        if error is not None:
            self._render_version = request.frame_key
            self._settle_pending_interaction_through(
                request.display.revision,
                failed=True,
                answer_host=self.board,
            )
            self.set_status(error, error=True)
            return True
        pending_frame = None
        pending_faceted = None
        if request.contract.figure.faceted:
            from zlc_frontend.panel_render import FacetedPanelResult

            if not isinstance(faceted_result, FacetedPanelResult):
                self._settle_pending_interaction_through(
                    request.display.revision,
                    failed=True,
                    answer_host=self.board,
                )
                self.set_status(
                    "render worker returned no complete faceted front",
                    error=True,
                )
                return False
            pending_figure = faceted_result.figure
            if figure is not pending_figure:
                self._settle_pending_interaction_through(
                    request.display.revision,
                    failed=True,
                    answer_host=self.board,
                )
                self.set_status(
                    "faceted worker result lost its exact DataFigure",
                    error=True,
                )
                return False
            pending_faceted = faceted_result
            pending_display = request.display
            if faceted_result.focus is not None:
                from zlc_frontend.histogram_display import (
                    FacetedHistogramDisplayState,
                )

                if isinstance(pending_display, FacetedHistogramDisplayState):
                    pending_display = pending_display.display_for(
                        faceted_result.focus.address
                    )
        elif frame is None or (
            self.config.kind is not PlotKind.SITE_MAP and figure is None
        ):
            self._settle_pending_interaction_through(
                request.display.revision,
                failed=True,
                answer_host=self.board,
            )
            self.set_status("render worker returned no complete front", error=True)
            return False
        else:
            pending_figure = None
            if self.config.kind is not PlotKind.SITE_MAP:
                from zlc_frontend import DataFigure

                if not isinstance(figure, DataFigure):
                    self._settle_pending_interaction_through(
                        request.display.revision,
                        failed=True,
                        answer_host=self.board,
                    )
                    self.set_status(
                        "render worker returned no exact DataFigure",
                        error=True,
                    )
                    return False
                pending_figure = figure
            pending_frame = frame
            pending_display = request.display
        self._remember_candidate_value(
            request.value,
            publication=publication,
        )
        self._pending_frame = pending_frame
        self._pending_faceted_result = pending_faceted
        self._pending_figure = pending_figure
        self._pending_display = pending_display
        self._pending_contract = request.contract
        self._pending_value = request.value
        self._pending_publication = publication
        self._pending_render_request_revision = int(request.request_revision)
        self._render_version = request.frame_key
        self.set_status("ok", error=False)
        return True

    def _settle_pending_interaction_through(
        self,
        presentation_revision: int,
        *,
        failed: bool,
        answer_host=None,
    ) -> None:
        """Settle only the display intent reached by this worker answer.

        Discard may synchronously promote the board mailbox's queued human
        intent.  The captured receipt is therefore cleared by identity, never
        by a broad revision comparison after a re-entrant callback.
        """

        pending_revision = self._pending_interaction_revision
        if (
            pending_revision is None
            or int(presentation_revision) < pending_revision
        ):
            return
        origin = self._pending_interaction_origin
        host = self._pending_interaction_host
        if answer_host is not None and host is not answer_host:
            return
        if failed and origin is not None and host is not None:
            host.discard_pending_interaction(origin)
        if (
            self._pending_interaction_origin != origin
            or self._pending_interaction_host is not host
            or self._pending_interaction_revision != pending_revision
        ):
            return
        self._pending_interaction_origin = None
        self._pending_interaction_host = None
        self._pending_interaction_revision = None
        self._resume_latest_candidate_after_interaction()

    def _clear_pending_render_result(self) -> None:
        """Discard one unpainted worker answer without touching the stable host."""

        self._pending_frame = None
        self._pending_faceted_result = None
        self._pending_figure = None
        self._pending_display = None
        self._pending_contract = None
        self._pending_value = None
        self._pending_publication = None
        self._pending_render_request_revision = None

    def _has_staged_render(self, request: _PanelRenderRequest) -> bool:
        return bool(
            self._pending_render_request_revision == request.request_revision
            and self._pending_value is request.value
            and (
                self._pending_frame is not None
                or self._pending_faceted_result is not None
            )
        )

    def _discard_staged_render(self, request: _PanelRenderRequest) -> None:
        if self._has_staged_render(request):
            self._clear_pending_render_result()

    def _begin_pointer_interaction(
        self,
        host,
        origin,
        *,
        value,
        publication: SignalPublication,
        surface_id: str,
    ) -> None:
        """Pin one surface's exact publication/value at pointer press."""

        if host is None or origin != host.visible_interaction_origin():
            return
        if value is None:
            self.set_status(
                "painted Figure front has no immutable data value",
                error=True,
            )
            return
        try:
            _require_publication_value(publication, value)
        except (TypeError, ValueError) as error:
            self.set_status(str(error), error=True)
            return
        visible_ref = getattr(getattr(value, "snapshot", None), "ref", None)
        origin_ref = getattr(origin.input_identity, "ref", None)
        if visible_ref is None or origin_ref != visible_ref:
            self.set_status(
                "interaction origin differs from the painted data front",
                error=True,
            )
            return
        self._pointer_interaction_pin = _PointerInteractionPin(
            host,
            origin,
            value,
            publication,
            str(surface_id),
        )
        if host is not self.board:
            return
        pending_ref = getattr(
            getattr(self._pending_value, "snapshot", None),
            "ref",
            None,
        )
        if pending_ref is not None and pending_ref != origin_ref:
            self._clear_pending_render_result()

    def _finish_pointer_interaction(self, host) -> None:
        """Release the press-time pin after the frontend has delivered its intent."""

        pin = self._pointer_interaction_pin
        if pin is None or host is not pin.host:
            return
        self._pointer_interaction_pin = None
        self._resume_latest_candidate_after_interaction()

    def _live_surface_interaction_pending(self) -> bool:
        """Whether the live card, rather than a frozen Edit surface, is pinned."""

        pin = self._pointer_interaction_pin
        return (
            (pin is not None and pin.host is self.board)
            or (
                self._pending_interaction_origin is not None
                and self._pending_interaction_host is self.board
            )
        )

    def _resume_latest_candidate_after_interaction(self) -> None:
        """Catch the live surface up once no gesture or render receipt owns it."""

        if (
            self._live_surface_interaction_pending()
            or self._candidate_value is None
        ):
            return
        candidate_ref = self._candidate_value.snapshot.ref
        presented_ref = getattr(
            getattr(self._presented_value, "snapshot", None),
            "ref",
            None,
        )
        if candidate_ref != presented_ref:
            self._request_current_render(force=True)

    def _continues_pending_interaction(self, host, origin) -> bool:
        """Whether ``origin`` is a newer front of this host's same gesture.

        An intermediate worker answer advances the held front's sequence and
        painted display revision.  Exact origin equality would therefore reject
        the next motion of the same drag.  Host identity plus monotonic exact
        exact front lineage admits that advance while preventing the Edit
        tab's second host from taking over another host's pending command.
        """

        pending = self._pending_interaction_origin
        if pending is None or host is not self._pending_interaction_host:
            return False
        return (
            origin.panel_id == pending.panel_id
            and origin.board_id == pending.board_id
            and origin.layout_generation == pending.layout_generation
            and origin.source_identity == pending.source_identity
            and origin.input_identity == pending.input_identity
            and origin.sequence >= pending.sequence
            and origin.painted_revision >= pending.painted_revision
        )

    def view_intent(self):
        """Return the intent resolved by the frontend Plot Panel contract."""

        if self.config.kind is PlotKind.SITE_MAP:
            raise ValueError(
                "Site map is an exact composite payload, not a dataset ViewIntent"
            )
        from zlc_frontend.plot_panel import plot_panel_view_from_params

        return FigureIntent(
            self.config.kind,
            "",
            "",
            plot_panel_view_from_params(
                self.config.kind,
                self.config.params,
            ),
        ).view_intent

    def _plot_panel_contract(
        self,
        view,
        *,
        value_name: str,
        axis_labels=None,
        short_labels=None,
        size_name: str | None = None,
        pixel_ratio: float | None = None,
        leaf_figure: FigureIntent | None = None,
    ):
        """Compose authored card facts into the frontend's sole contract."""

        from zlc_frontend import PlotPanelContract
        from zlc_frontend.plot_panel import plot_panel_value_label

        if leaf_figure is not None and not isinstance(leaf_figure, FigureIntent):
            raise TypeError("leaf Figure intent must be FigureIntent")
        if leaf_figure is None or self.config.kind is not PlotKind.SITE_MAP:
            figure = FigureIntent(
                self.config.kind,
                str(self.config.title or value_name),
                (
                    leaf_figure.value_label
                    if leaf_figure is not None
                    else plot_panel_value_label(
                        str(value_name),
                        axis_labels,
                        short_labels,
                    )
                ),
                view=view,
                rolling_distribution=(
                    self.config.kind is PlotKind.ROLLING
                    and bool(
                        panel_display_form_values_from_tree(
                            PlotKind.ROLLING,
                            self.config.params,
                        )["show_dist"]
                    )
                ),
            )
        else:
            if leaf_figure.kind is not PlotKind.SITE_MAP or view is not None:
                raise ValueError("SiteMap attachment supplied another Figure intent")
            figure = leaf_figure
        return PlotPanelContract(
            self.panel_id,
            figure,
            size_name=self.config.size if size_name is None else str(size_name),
            pixel_ratio=(
                self._raster_pixel_ratio
                if pixel_ratio is None
                else float(pixel_ratio)
            ),
        )

    def _plot_panel_source(self, snapshot, value, publication):
        """Bind one exact source and an optional leaf-authored Figure intent.

        A leaf may resolve an otherwise ambiguous standard view, while all
        rendering and display state still belong to the shared frontend.  A
        SiteMap additionally supplies its composite presentation payload.
        Monitor and Edit call this same normalizer, so neither reconstructs
        leaf semantics independently.
        """

        from zlc_frontend.plot_panel import plot_panel_input

        attached = self._presentation_provider(value, publication)
        if self.config.kind is not PlotKind.SITE_MAP:
            source = plot_panel_input(self.config.kind, snapshot)
            if attached is None:
                return None, source
            if (
                isinstance(attached, tuple)
                and len(attached) == 2
                and isinstance(attached[0], FigureIntent)
                and attached[0].kind is PlotKind.SITE_MAP
            ):
                # A composite SiteMap attachment is irrelevant when the user
                # deliberately chooses an ordinary Dataset presentation.
                return None, source
            if not isinstance(attached, FigureIntent):
                raise TypeError("Dataset presentation attachment must be FigureIntent")
            if attached.kind is not self.config.kind:
                return None, source
            if attached.view is None:
                raise ValueError("Dataset Figure intent needs a complete typed view")
            if attached.view.schema_fingerprint != snapshot.block.schema.fingerprint:
                raise ValueError("Dataset Figure intent belongs to another schema")
            return attached, source
        if not isinstance(attached, tuple) or len(attached) != 2:
            raise TypeError(
                "SiteMap output requires (FigureIntent, SiteMapPresentation)"
            )
        figure, presentation = attached
        if not isinstance(figure, FigureIntent):
            raise TypeError("SiteMap attachment lost its FigureIntent")
        if figure.kind is not PlotKind.SITE_MAP:
            raise ValueError("SiteMap attachment supplied another plot kind")
        return figure, plot_panel_input(
            PlotKind.SITE_MAP,
            snapshot,
            presentation,
        )

    def _display_state(self, contract, *, reset_view: bool = False):
        """Resolve the one frontend-owned display contract for this card.

        A candidate from another schema/generation composes at its home view;
        it cannot consume the viewport or Grid focus authored against the
        still-visible schema.  The stored view is retired only if that
        candidate is later presented.
        """

        from zlc_frontend import plot_panel_display_state
        if not isinstance(reset_view, bool):
            raise TypeError("reset_view must be bool")
        return plot_panel_display_state(
            contract,
            self.config.params,
            revision=self._display_revision,
            focus=self._grid_focus,
            home_view=reset_view,
        )

    def frozen_data_figure(self):
        """Return the exact typed figure behind the currently displayed panel.

        This is a projection of the already-owned immutable monitor revision,
        not another acquisition and not a GUI snapshot.  The same
        frontend Figure session supplies the document used to draw the card,
        so saving cannot re-guess axes or plot kind from an array shape.
        """

        if self.config.kind is PlotKind.SITE_MAP:
            raise RuntimeError(
                "Site map is an exact composite front, not a single-dataset DataFigure"
            )
        figure = self._presented_figure
        if figure is None:
            raise RuntimeError("the panel has no presented typed figure to save")

        # Grid overview is an encoded image whose exact DataFigure is promoted
        # in the same Qt present transaction.  Focused and ordinary panels also
        # carry an EvaluatedInput, so verify that visible provenance before
        # exposing their figure to Fit or Save.
        board = self.board
        if board is not None and getattr(board, "front_frame", None) is None:
            if not bool(getattr(board, "showing_overview", False)):
                raise RuntimeError("front replacement in progress")
            return figure
        visible_input = self._visible_evaluated_input()
        entries = tuple(figure.datasets.entries)
        if (
            len(entries) != 1
            or entries[0].dataset_id != visible_input.dataset_id
            or entries[0].snapshot.ref != visible_input.ref
        ):
            raise RuntimeError("front replacement in progress")
        return figure

    def frozen_display_state(self):
        """Return the display state promoted with the visible immutable front."""

        if self._presented_display is None:
            raise RuntimeError("the panel has no presented display state")
        return self._presented_display

    def frozen_plot_panel_contract(self):
        """Return the exact frontend contract promoted with the visible front."""

        contract = self._presented_contract
        if contract is None:
            raise RuntimeError("the panel has no presented PlotPanel contract")
        return contract

    def frozen_render_value(self):
        """Return the immutable data-plane value promoted with the visible front."""

        value = self._presented_value
        if value is None:
            raise RuntimeError("the panel has no presented data value")
        return value

    def frozen_render_publication(self) -> SignalPublication:
        """Return the exact neutral publication paired with the painted value."""

        value = self._presented_value
        if value is None:
            raise RuntimeError("the panel has no presented data value")
        return _require_publication_value(self._presented_publication, value)

    def frozen_render_payload(self):
        """Return the exact typed payload painted by the current focused front."""

        board = self.board
        front = None if board is None else board.front_frame
        if front is None:
            return None
        if len(front.panels) != 1:
            raise RuntimeError("front replacement in progress")
        return front.panels[0].display_payload

    def _visible_evaluated_input(self):
        """Read the exact typed input painted by the one current board front."""

        board = self.board
        front = None if board is None else board.front_frame
        if front is None or len(front.panels) != 1:
            raise RuntimeError("front replacement in progress")
        visible_input = getattr(
            front.panels[0].display_payload,
            "evaluated_input",
            None,
        )
        if visible_input is None:
            raise RuntimeError("front replacement in progress")
        return visible_input

    def present(self) -> None:
        """Flush this card's composed front to the screen.  GUI thread only.

        Phase 2 of the board's two-phase render: the board composes every panel
        of a tick, then presents them together, so the screen never shows a torn
        mix of instants.
        """

        frame = self._pending_frame
        faceted = self._pending_faceted_result
        if frame is None and faceted is None:
            return
        self._pending_frame = None
        self._pending_faceted_result = None
        pending_figure = self._pending_figure
        pending_display = self._pending_display
        pending_contract = self._pending_contract
        pending_value = self._pending_value
        pending_publication = self._pending_publication
        pending_render_request_revision = self._pending_render_request_revision
        visible_schema = self._current_schema()
        self._pending_figure = None
        self._pending_display = None
        self._pending_contract = None
        self._pending_value = None
        self._pending_publication = None
        self._pending_render_request_revision = None
        if pending_contract is None:
            raise RuntimeError("pending raster has no PlotPanel contract")
        pending_size_name = str(pending_contract.size_name)
        initial_grid_size_commit = (
            self._initial_grid_size_pending
            and self.config.kind is PlotKind.GRID
        )
        geometry_changes = (
            pending_size_name != self.config.size
            or (
                self._presented_contract is not None
                and pending_size_name != self._presented_contract.size_name
            )
        )
        initial_size_changed = False
        if geometry_changes:
            self.setUpdatesEnabled(False)
        try:
            self._build_plot()
            logical_size = pending_contract.logical_size
            selector_figure = None
            if faceted is not None and faceted.focus is not None:
                intent = pending_contract.figure.view_intent
                if pending_figure is None or intent is None:
                    raise RuntimeError(
                        "focused grid front lost its typed Figure context"
                    )
                selector_figure = pending_figure.focused_typed_panel(
                    faceted.focus.panel_index,
                    expected_address=faceted.focus.address,
                    expected_intent=intent,
                )
            if faceted is not None:
                context = (
                    FigureSurfaceContext.for_figure(
                        pending_figure,
                        display=pending_display,
                        contract=pending_contract,
                    )
                    if faceted.overview is not None
                    else FigureSurfaceContext.for_frame(
                        faceted.frame,
                        figure=pending_figure,
                        display=pending_display,
                        contract=pending_contract,
                        selector_figure=selector_figure,
                    )
                )
                self.board.present_faceted(
                    faceted,
                    context=context,
                    logical_size=logical_size,
                )
            else:
                context = FigureSurfaceContext.for_frame(
                    frame,
                    figure=pending_figure,
                    display=pending_display,
                    contract=pending_contract,
                )
                self.board.present_frame(
                    frame,
                    context=context,
                    logical_size=logical_size,
                )
            self._presented_figure = pending_figure
            self._presented_display = pending_display
            self._presented_contract = pending_contract
            self._presented_value = pending_value
            self._presented_publication = _require_publication_value(
                pending_publication,
                pending_value,
            )
            self._presented_render_request_revision = (
                pending_render_request_revision
            )
            if initial_grid_size_commit:
                self._initial_grid_size_pending = False
                if self.config.size != pending_size_name:
                    self.config.size = pending_size_name
                    initial_size_changed = True
                    combo = getattr(self, "size_combo", None)
                    if combo is not None:
                        with _signals_blocked(combo):
                            combo.setCurrentIndex(
                                combo.findData(pending_size_name)
                            )
            presented_schema = pending_value.snapshot.block.schema
            schema_changed = (
                visible_schema is not None
                and visible_schema.fingerprint != presented_schema.fingerprint
            )
            if schema_changed:
                # Viewport and focused-cell state belong to the old visible
                # schema.  The candidate already rendered at home; retire the
                # old state only now that its matching raster has replaced the
                # old front.
                self._clear_display_view(request_render=False)
                self._grid_focus = None
            self._reconcile_persisted_view_for_schema(presented_schema)
            self._refresh_grid_control_surface(self)
            self._refresh_view_spec_controls()
            self._refresh_unit_readout()
            self._settle_pending_interaction_through(
                pending_display.revision,
                failed=False,
                answer_host=self.board,
            )
            if geometry_changes:
                # Promote raster and geometry as one visible fact.  Until this
                # exact-size answer arrived, the accepted raster stayed at its authored
                # authored extent and was never stretched into the new size.
                self._apply_fixed_size(
                    pending_size_name,
                    sync_board=False,
                )
                if (
                    getattr(self, "settings_popup", None) is not None
                    and self.settings_popup.isVisible()
                ):
                    self._size_settings_popup()
                self.layout_changed.emit()
        finally:
            if geometry_changes:
                self.setUpdatesEnabled(True)
        if initial_size_changed:
            self.changed.emit()
        self._sync_fit_authoring_from_presented(
            prepare_authoring=any(
                pane.isVisible() for pane in self._live_fit_panes()
            )
        )
        self._queue_live_fit()
        self.front_presented.emit()

    def setting_label_width(self, _metrics) -> int:
        """One label column for Setting and Edit, independent of live text."""

        labels = {
            "signal",
            "size",
            "sub plot",
            "facet",
            "repeat",
            "unit",
            "update",
            "title",
            "x range",
            "y range",
            "colour range",
            "path",
            "name",
            "file",
        }
        labels.update(panel_display_form_labels(self.config.kind))
        # Delegate the actual typography/scale rule to the component library;
        # TaskConsole contributes only the product's field inventory.
        return _setting_label_width(labels, minimum=80)

    def _build_settings(self) -> None:
        """Build the main-UI flat Setting surface over current typed state."""

        popup = FluentPopup(self)
        outer = QtWidgets.QVBoxLayout(popup)
        outer.setContentsMargins(0, 0, 0, 0)
        self._settings_scroll = FluentScrollArea()
        content = QtWidgets.QWidget()
        content.setStyleSheet("background: transparent;")
        self._settings_col = QtWidgets.QVBoxLayout(content)
        pad = scaled_px(10)
        self._settings_col.setContentsMargins(
            pad,
            pad,
            pad + fluent_scrollbar_thickness() + scaled_px(4),
            pad,
        )
        self._settings_col.setSpacing(scaled_px(10, minimum=6))
        self._settings_scroll.set_width_bounded_widget(content)
        outer.addWidget(self._settings_scroll)
        self.settings_popup = popup
        self._settings_h_hwm = 0

        label_w = self.setting_label_width(self.fontMetrics())
        # The Setting surface owns one stable width.  Descendant combo/path/error
        # size hints are content, not a request to widen the outer popup/window.
        popup.setFixedWidth(
            label_w
            + scaled_px(360, minimum=320)
            + 2 * pad
            + fluent_scrollbar_thickness()
            + scaled_px(4)
        )

        def section(title):
            self._settings_col.addWidget(FluentSectionLabel(title))
            layout = QtWidgets.QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(scaled_px(6, minimum=4))
            self._settings_col.addLayout(layout)
            return layout

        # ---- Source: one typed dataset.  Combining producers belongs to a
        # Processor or explicit join, never an independent-latest GUI expression.
        source = section("Source")
        input_format = self.config.kind.input_format
        if input_format:
            accepts = FluentLabel(f"accepts {input_format}")
            accepts.setWordWrap(True)
            accepts.setStyleSheet(
                f"color: {GREY}; background: transparent; border: none;"
            )
            source.addWidget(accepts)
        self.signal_combo = FluentTreeComboBox()
        self.signal_combo.setToolTip(
            "The typed dataset this panel displays, grouped by its producing node."
        )
        self.signal_combo.currentIndexChanged.connect(self._on_signal_pick)
        source.addWidget(
            FluentSettingRow("signal", self.signal_combo, label_width=label_w)
        )
        self.status = FluentLabel(self._status_text)
        self.status.setWordWrap(True)
        self.status.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        self.status.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        source.addWidget(self.status)

        # ---- Display: the declared view knobs for this kind, emitted through the SHARED
        # row builder, so a kind that gains a knob shows it in both surfaces with no
        # wiring here.
        display = section("Display")
        self.size_combo = FluentComboBox()
        for preset in PANEL_SIZES:
            self.size_combo.addItem(preset, preset)
        index = self.size_combo.findData(self.config.size)
        if index >= 0:
            self.size_combo.setCurrentIndex(index)
        self.size_combo.setToolTip("Panel size preset (height × width half-units)")
        # ``activated`` is the operator-authoring boundary.  It also fires
        # when the operator explicitly re-selects the already visible preset,
        # which must cancel a still-pending one-shot Grid recommendation.
        # Programmatic reconciliation is presentation, not authoring.
        self.size_combo.activated.connect(
            lambda _i: self._on_size(
                str(self.size_combo.currentData() or self.config.size)
            )
        )
        display.addWidget(
            FluentSettingRow("size", self.size_combo, label_width=label_w)
        )
        self.display_form_surface = self._make_display_form_surface(
            self._set_param,
            label_w,
            parent=content,
        )
        display.addWidget(self.display_form_surface)
        self.view_spec_editor = ViewSpecEditor(
            label_width=label_w,
            parent=content,
        )
        self.view_spec_editor.viewChanged.connect(self._commit_view_spec)
        display.addWidget(self.view_spec_editor)
        unit_row, self.unit_label = self._make_unit_readout_row(
            label_w,
            parent=content,
        )
        display.addWidget(unit_row)
        self.update_combo = FluentComboBox()
        for interval in UPDATE_INTERVALS:
            self.update_combo.addItem(f"{interval} ms", interval)
        index = self.update_combo.findData(
            int(
                self.config.params.get("update_ms", DEFAULT_UPDATE_MS)
                or DEFAULT_UPDATE_MS
            )
        )
        if index >= 0:
            self.update_combo.setCurrentIndex(index)
        self.update_combo.setToolTip(
            "How often this panel redraws; acquisition is unaffected."
        )
        self.update_combo.currentIndexChanged.connect(self._on_update_interval)
        display.addWidget(
            FluentSettingRow("update", self.update_combo, label_width=label_w)
        )

        # ---- Fit: one Figure-owned editor.  The Edit tab builds a
        # second view through the same factory; both reconcile against the
        # card's one request/result state and neither opens a DataFigure window.
        self.fit_authoring_pane = None
        if self._fit_capable_kind():
            fit_section = section("Fit")
            self.fit_authoring_pane = self.make_fit_authoring_pane(
                popup,
                label_width=label_w,
            )
            fit_section.addWidget(self.fit_authoring_pane)

        # ---- Panel: card identity and the two standard panel actions.
        panel = section("Panel")
        self.title_edit = FluentLineEdit(self.config.title)
        self.title_edit.setPlaceholderText("panel title…")
        self.title_edit.setToolTip("Rename this panel (also the default save name)")
        self.title_edit.editingFinished.connect(self._commit_title)
        panel.addWidget(FluentSettingRow("title", self.title_edit, label_width=label_w))
        self.remove_button = FluentButton("Remove", color=ORANGE)
        self.remove_button.setFixedWidth(scaled_px(72, minimum=58))
        self.remove_button.clicked.connect(self._remove_from_settings)
        self.edit_button = FluentButton("Edit…", color=ACCENT)
        self.edit_button.setFixedWidth(scaled_px(64, minimum=52))
        self.edit_button.setToolTip("Open this panel's full Edit tab")
        self.edit_button.clicked.connect(self._edit_from_settings)
        actions = QtWidgets.QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(scaled_px(6, minimum=4))
        actions.addWidget(self.remove_button)
        actions.addWidget(self.edit_button)
        actions.addStretch(1)
        panel.addLayout(actions)

        self._settings_col.addStretch(1)

        self._refresh_signal_combo()
        self._refresh_view_spec_controls()

    def _remove_from_settings(self) -> None:
        self.settings_popup.hide()
        self.remove_requested.emit(self)

    def _edit_from_settings(self) -> None:
        self.settings_popup.hide()
        self.edit_requested.emit(self)

    def _open_settings(self) -> None:
        self._settings_anchor.toggle(
            self._settings_scroll.widget(),
            prepare=self._prepare_settings_popup,
            present=self._present_settings_popup,
        )

    def _prepare_settings_popup(self) -> None:
        self.refresh_on_show()
        self._refresh_signal_combo()
        self._refresh_grid_control_surface(self)
        self._refresh_view_spec_controls()
        self._sync_fit_authoring_from_presented(prepare_authoring=True)

    def _present_settings_popup(self) -> None:
        popup = self.settings_popup
        anchor = self.setting_button.mapToGlobal(
            QtCore.QPoint(self.setting_button.width(), self.setting_button.height()))
        self._size_settings_popup()                        # height: show-all, grow-not-shrink
        screen = QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        top_y = anchor.y() + popup_gap()   # the ONE below-anchor Fluent popup gap (combo / overflow share it)
        x = anchor.x() - popup.width()
        if avail is not None:
            x = max(avail.left(), min(x, avail.right() - popup.width()))
        popup.move(x, top_y)
        popup.show()
        popup.raise_()

    def _size_settings_popup(self) -> None:
        """Size the Setting frame: EXPAND to show all its content, UNTIL it reaches the PLOT PANEL's
        own bottom edge -- then the FluentScrollArea scrolls the overflow.  So the cap is the panel
        boundary (`panel_bottom - top_y`), NOT the screen: a tall panel gets a tall frame, a short
        panel scrolls, and either way the frame stays within the panel.  GROW with the panel size,
        clamped to the live cap (so shrinking the panel re-clamps the frame to the smaller panel)."""
        popup = getattr(self, "settings_popup", None)
        if popup is None:
            return
        popup.adjustSize()
        screen = QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        anchor_y = self.setting_button.mapToGlobal(QtCore.QPoint(0, self.setting_button.height())).y()
        top_y = anchor_y + popup_gap()      # match the same gap the open path (above) uses to place it
        content = self._settings_scroll.widget()
        content_h = (content.sizeHint().height() if content is not None else popup.height()) + 2 * scaled_px(10)
        # The cap is the PLOT PANEL's own bottom edge (the popup opens just below the gear near the
        # panel top and grows DOWN); content past it scrolls.
        panel_bottom = self.mapToGlobal(QtCore.QPoint(0, self.height())).y()
        cap = max(scaled_px(140), panel_bottom - top_y)
        if avail is not None:                              # last-resort: never escape the physical screen
            cap = min(cap, avail.bottom() - top_y)
        want = min(content_h, cap)                         # grow to the content, capped at the panel bottom
        self._settings_h_hwm = max(int(self._settings_h_hwm), int(want))   # grow within a session...
        h = min(self._settings_h_hwm, cap)                 # ...but always re-clamp to the LIVE panel cap
        popup.setMaximumHeight(int(cap))                   # content beyond the panel bottom scrolls
        popup.resize(popup.width(), max(scaled_px(140), int(h)))

    def _fill_slot_combo(self, combo, current: str) -> None:
        """Populate one tree from one already-coherent topology projection."""

        groups = self.signal_groups_provider(str(current or ""))
        with _signals_blocked(combo):
            combo.set_signal_tree(
                groups,
                current=str(current or ""),
                none_label="(none)",
            )

    def _refresh_signal_combo(self) -> None:
        """Refresh the one dataset picker without rebuilding the popup."""

        combo = getattr(self, "signal_combo", None)
        if combo is not None:
            self._fill_slot_combo(combo, self.config.signal)

    def refresh_open_signal_metadata(self) -> bool:
        """Refresh one open Setting picker's leaf metadata in place.

        Signal topology is owned by the console's explicit add/remove/start/stop
        paths.  A newly published value or a schema change is not topology, so it
        may only change the existing leaves' shape/readiness text.
        """

        popup = getattr(self, "settings_popup", None)
        combo = getattr(self, "signal_combo", None)
        if (
            popup is None
            or combo is None
            or not popup.isVisible()
        ):
            return False
        groups = self.signal_groups_provider(str(self.config.signal or ""))
        return combo.reconcile_signal_tree_metadata(groups)

    def refresh_open_signal_topology(self) -> bool:
        """Reconcile an explicit provider add/remove in one open picker."""

        popup = getattr(self, "settings_popup", None)
        combo = getattr(self, "signal_combo", None)
        if popup is None or combo is None or not popup.isVisible():
            return False
        self._refresh_signal_combo()
        return True

    def _current_schema(self):
        """Schema of the visible front, or the first front awaiting presentation.

        A candidate may seed an initially blank card's Setting because no older
        pixels or controls exist to contradict it.  Once any front is visible,
        only that exact front owns Setting until its replacement is presented.
        """

        value = self._presented_value or self._candidate_value
        snapshot = None if value is None else getattr(value, "snapshot", None)
        block = None if snapshot is None else getattr(snapshot, "block", None)
        return None if block is None else getattr(block, "schema", None)

    def _refresh_grid_control_surface(self, owner) -> None:
        """Reconcile the canonical Grid cell FormSpec after intent changes."""

        if self.config.kind is not PlotKind.GRID:
            return
        surface = getattr(owner, "display_form_surface", None)
        self._reconcile_display_form_surface(surface)
        popup = getattr(self, "settings_popup", None)
        if owner is self and popup is not None and popup.isVisible():
            self._size_settings_popup()

    def _saved_view_spec(self, schema):
        """Purely read the current owner-coded presentation value, if valid."""

        if self.config.kind is PlotKind.SITE_MAP:
            return None
        from zlc_frontend.plot_panel import plot_panel_view_for_schema

        return plot_panel_view_for_schema(
            self.config.kind,
            self.config.params,
            schema,
        )

    def _effective_view_spec(self, schema, *, declared_default=None):
        """Resolve the exact auto-or-authored frontend view used for display."""

        if self.config.kind is PlotKind.SITE_MAP:
            return None
        saved = self._saved_view_spec(schema)
        if saved is not None:
            return saved
        if declared_default is not None:
            if declared_default.schema_fingerprint != schema.fingerprint:
                raise ValueError("declared default view belongs to another schema")
            return declared_default
        if self.config.kind is PlotKind.GRID:
            from zlc_frontend.figure import suggest_default_grid_view

            return suggest_default_grid_view(schema).spec
        from zlc_frontend.figure import suggest_view

        return suggest_view(schema, self.view_intent()).spec

    def _refresh_view_spec_control_surface(self, owner) -> None:
        editor = getattr(owner, "view_spec_editor", None)
        if editor is None:
            return
        if self.config.kind in {PlotKind.SITE_MAP, PlotKind.PULSE}:
            editor.reconcile(None, None)
            return
        schema = self._current_schema()
        contract = self._presented_contract or self._pending_contract
        declared_default = (
            None
            if contract is None or contract.figure.kind is PlotKind.SITE_MAP
            else contract.figure.view
        )
        view = (
            None
            if schema is None
            else self._effective_view_spec(
                schema,
                declared_default=declared_default,
            )
        )
        editor.reconcile(
            schema,
            view,
            intent=(
                None
                if schema is None or self.config.kind is PlotKind.GRID
                else self.view_intent()
            ),
            faceted=self.config.kind is PlotKind.GRID,
        )

    def _refresh_view_spec_controls(self) -> None:
        self._refresh_view_spec_control_surface(self)

    def _commit_view_spec(self, candidate) -> bool:
        """Persist one frontend-authored display view without changing data."""

        from zlc_frontend.figure import ViewSpec, validate_view_spec, view_spec_to_tree
        from zlc_frontend.plot_panel import plot_panel_view_from_params

        schema = self._current_schema()
        if schema is None:
            return False
        if not isinstance(candidate, ViewSpec):
            raise TypeError("view editor must emit ViewSpec")
        if candidate.schema_fingerprint != schema.fingerprint:
            raise ValueError("view editor emitted another schema generation")
        validate_view_spec(schema, candidate)
        previous = plot_panel_view_from_params(
            self.config.kind,
            self.config.params,
        )
        if self.config.kind is PlotKind.GRID:
            self._grid_focus = None
        intent_changed = (
            previous is not None and previous.intent is not candidate.intent
        )
        if intent_changed:
            changed = bool(
                self._commit_persisted_params(
                    {_VIEW_SPEC_PARAM: view_spec_to_tree(candidate)},
                    remove=panel_display_param_keys(self.config.kind),
                    request_render=True,
                )
            )
        else:
            changed = self._set_params(
                {_VIEW_SPEC_PARAM: view_spec_to_tree(candidate)}
            )
        self._refresh_grid_control_surface(self)
        self._refresh_view_spec_controls()
        return changed

    def _reconcile_persisted_view_for_schema(self, schema) -> None:
        """Retire prior-generation view state at the schema commit boundary."""

        if self.config.kind is PlotKind.SITE_MAP:
            return
        from zlc_frontend.plot_panel import plot_panel_view_from_params

        view = plot_panel_view_from_params(
            self.config.kind,
            self.config.params,
        )
        if view is None or view.schema_fingerprint == schema.fingerprint:
            return
        self._commit_persisted_params(
            remove=(
                _VIEW_SPEC_PARAM,
                _HISTOGRAM_CELL_THRESHOLDS_PARAM,
                _HISTOGRAM_THRESHOLDS_PARAM,
            )
        )

    def _on_signal_pick(self, _index: int) -> None:
        """Commit one card-local dataset binding and request one compose."""

        name = str(self.signal_combo.currentData() or "")
        if name == self.config.signal:
            return
        self.config.signal = name
        # A Fit result is exact-revision state of the old Figure, not a panel
        # preference that may survive a dataset rebind.  Retire it through the
        # same authority edge as Clear, but let the new binding below issue the
        # sole render request for this transaction.
        self._retire_all_fits(emit_changed=False)
        self._commit_persisted_params(
            remove=(
                _VIEW_SPEC_PARAM,
                _HISTOGRAM_CELL_THRESHOLDS_PARAM,
                _HISTOGRAM_THRESHOLDS_PARAM,
            ),
            emit_changed=False,
        )
        self._candidate_value = None
        self._candidate_publication = None
        self._grid_focus = None
        self._clear_display_view(request_render=False, emit_changed=False)
        self._invalidate_render_binding()
        self._presented_figure = None
        self._presented_display = None
        self._presented_contract = None
        self._presented_value = None
        self._presented_publication = None
        self._presented_render_request_revision = None
        if self.board is None:
            self._clear_figure_outputs(notify=True)
        else:
            # A binding change is not a request to rebuild the stable widget;
            # it atomically retires the old pixels, Figure context and selector
            # outputs before a request for the newly selected route is issued.
            self.board.clear()
        self._refresh_grid_control_surface(self)
        self._refresh_view_spec_controls()
        self._refresh_unit_readout()
        self._render_version = -1
        self._request_display_render()
        self.changed.emit()

    def _current_unit_text(self) -> str:
        """The visible signal's producer-declared unit."""

        value = self._presented_value or self._candidate_value
        if value is None:
            return "—"
        return value.unit or "dimensionless"

    def _refresh_unit_readout(self) -> None:
        label = getattr(self, "unit_label", None)
        if label is not None:
            label.setText(self._current_unit_text())

    def _on_view_committed(self, commit) -> None:
        self.accept_view_commit_from(self.board, commit)

    def accept_view_commit_from(self, host, commit) -> None:
        """CAS and answer one exact-front zoom/pan commit.

        ``FigureSurfaceHost`` deliberately forwards the whole typed commit.  A
        delayed gesture from an older front therefore cannot rewrite this
        card's current view, and a failed compose releases only that pending
        origin instead of leaving the selector permanently wedged.
        """

        from zlc_frontend.selector import (
            CurveViewportCommit,
            HistogramViewportCommit,
            ImageViewportCommit,
        )

        if not isinstance(
            commit,
            (CurveViewportCommit, HistogramViewportCommit, ImageViewportCommit),
        ):
            raise TypeError("view commit must retain its typed exact origin")
        if host is None:
            return
        if commit.origin != host.visible_interaction_origin():
            host.discard_pending_interaction(commit.origin)
            return
        candidate = commit.viewport
        viewport_revision = getattr(candidate, "viewport_revision", None)
        if viewport_revision is not None:      # image-family transform
            views = candidate.optional_coordinate_views_for_normalized_bounds()
            pin = views if any(view is not None for view in views) else None
            revision = int(viewport_revision)
        else:                                  # curve/histogram transform
            span = tuple(float(value) for value in candidate.x_limits)
            home = tuple(float(value) for value in candidate.home_x_limits)
            pin = None if span == home else (span, None)
            revision = int(candidate.display_revision)
        pending_origin = self._pending_interaction_origin
        if pending_origin is None:
            if self._display_revision != commit.origin.painted_revision:
                host.discard_pending_interaction(commit.origin)
                return
        elif not self._continues_pending_interaction(host, commit.origin):
            host.discard_pending_interaction(commit.origin)
            return
        if revision <= self._display_revision:
            host.discard_pending_interaction(commit.origin)
            return
        values = self._display_form_values()
        if isinstance(commit, ImageViewportCommit):
            x_view, y_view = pin or (None, None)
            values.update(
                {
                    "x_min": None if x_view is None else x_view[0],
                    "x_max": None if x_view is None else x_view[1],
                    "y_min": None if y_view is None else y_view[0],
                    "y_max": None if y_view is None else y_view[1],
                }
            )
        else:
            x_view = None if pin is None else pin[0]
            values.update(
                {
                    "x_min": None if x_view is None else x_view[0],
                    "x_max": None if x_view is None else x_view[1],
                }
            )
        if not self._commit_display_form_values(
            values,
            current_value_limits=self._shown_limits(),
            request_render=False,
            emit_changed=True,
        ):
            host.discard_pending_interaction(commit.origin)
            return
        if self._display_revision > revision:
            raise RuntimeError("display form revision advanced beyond viewport commit")
        self._pending_interaction_origin = commit.origin
        self._pending_interaction_host = host
        self._pending_interaction_revision = self._display_revision
        # The commit changes only the card-owned display state.  The worker
        # answers it from the already accepted immutable data revision; Qt
        # never composes or waits for that answer.
        self._request_current_render(surface=host)

    def _on_color_limits_committed(self, commit) -> None:
        self.accept_color_limits_from(self.board, commit)

    def accept_color_limits_from(self, host, commit) -> None:
        """CAS one clim-rail commit into the shared fixed-limits fact."""

        from zlc_frontend.selector import ImageColorLimitsCommit

        if not isinstance(commit, ImageColorLimitsCommit):
            raise TypeError("color-limit commit must retain its typed exact origin")
        if host is None:
            return
        if commit.origin != host.visible_interaction_origin():
            host.discard_pending_interaction(commit.origin)
            return
        if (
            (
                self._pending_interaction_origin is None
                and self._display_revision
                != commit.origin.painted_revision
            )
            or (
                self._pending_interaction_origin is not None
                and not self._continues_pending_interaction(
                    host,
                    commit.origin,
                )
            )
        ):
            host.discard_pending_interaction(commit.origin)
            return

        lo, hi = (float(value) for value in commit.color_limits)
        if not self._store_fixed_lims(lo, hi):
            host.discard_pending_interaction(commit.origin)
            return
        self._pending_interaction_origin = commit.origin
        self._pending_interaction_host = host
        self._pending_interaction_revision = self._display_revision
        self._request_current_render(surface=host)

    def _on_histogram_thresholds_committed(self, commit) -> None:
        self.accept_thresholds_from(self.board, commit)

    def accept_thresholds_from(self, host, commit) -> None:
        """CAS one drag step into the exact visible histogram cell."""

        from zlc_frontend.histogram_display import (
            FacetedHistogramDisplayState,
            faceted_histogram_display_with_thresholds,
            histogram_cell_thresholds_to_tree,
            histogram_display_with_thresholds,
        )
        from zlc_frontend.selector import HistogramThresholdCommit

        if not isinstance(commit, HistogramThresholdCommit):
            raise TypeError(
                "threshold commit must retain its typed exact origin"
            )
        if host is None:
            return
        if commit.origin != host.visible_interaction_origin():
            host.discard_pending_interaction(commit.origin)
            return
        if (
            (
                self._pending_interaction_origin is None
                and self._display_revision
                != commit.origin.painted_revision
            )
            or (
                self._pending_interaction_origin is not None
                and not self._continues_pending_interaction(
                    host,
                    commit.origin,
                )
            )
        ):
            host.discard_pending_interaction(commit.origin)
            return

        contract = self._presented_contract
        if contract is None:
            host.discard_pending_interaction(commit.origin)
            return
        display = self._display_state(contract)
        if self.config.kind is PlotKind.GRID:
            focus = self._grid_focus
            if (
                focus is None
                or not isinstance(display, FacetedHistogramDisplayState)
            ):
                host.discard_pending_interaction(commit.origin)
                return
            candidate = faceted_histogram_display_with_thresholds(
                display,
                focus.address,
                commit.thresholds,
            )
            if candidate == display:
                host.discard_pending_interaction(commit.origin)
                return
            self._commit_persisted_params(
                {
                    _HISTOGRAM_CELL_THRESHOLDS_PARAM:
                        histogram_cell_thresholds_to_tree(
                            candidate.cell_thresholds
                        )
                }
            )
            self._display_revision = candidate.revision
        else:
            candidate = histogram_display_with_thresholds(
                display,
                commit.thresholds,
            )
            if candidate == display:
                host.discard_pending_interaction(commit.origin)
                return
            self._commit_persisted_params(
                {
                    _HISTOGRAM_THRESHOLDS_PARAM: list(candidate.thresholds)
                }
            )
            self._display_revision = candidate.revision
        self._pending_interaction_origin = commit.origin
        self._pending_interaction_host = host
        self._pending_interaction_revision = self._display_revision
        self._request_current_render(surface=host)

    def _store_fixed_lims(self, lo: float, hi: float) -> bool:
        """Commit a colour-rail gesture through the canonical display handler."""

        state = self._display_state(self._display_contract_for_authoring())
        keys = panel_display_value_range_keys(state)
        if keys is None:
            raise ValueError("this PlotKind has no authored value range")
        values = self._display_form_values()
        values["relim_mode"] = RelimMode.FIXED
        values[keys[0]], values[keys[1]] = float(lo), float(hi)
        return self._commit_display_form_values(
            values,
            current_value_limits=(float(lo), float(hi)),
            request_render=False,
            emit_changed=True,
        )

    def _on_update_interval(self, _idx: int) -> None:
        """Persist THIS panel's display refresh interval (``config.params["update_ms"]``,
        one of :data:`UPDATE_INTERVALS`) and ask the console to re-base its timer so the new
        rate co-aligns with the others.  No plot rebuild -- only the refresh cadence changes."""
        ms = int(self.update_combo.currentData() or DEFAULT_UPDATE_MS)
        if ms == int(self.config.params.get("update_ms", DEFAULT_UPDATE_MS) or DEFAULT_UPDATE_MS):
            return
        self._commit_persisted_params({"update_ms": ms})
        self.update_interval_changed.emit()    # console re-bases the shared timer

    def _refresh_title(self) -> None:
        """Compose the grey frame TITLE: the panel KIND + WHERE its signal comes from (the
        legend the console computes), e.g. ``1D vector — value ← Fit``.  This is
        the ordinary QGroupBox title -- the grey chip, alignment and font are the frame's own."""
        head = PANEL_KINDS[self.config.kind]
        info = " · ".join(p for p in self._signal_info.splitlines() if p.strip())
        self.setTitle(f"{head} — {info}" if info else head)

    def set_signal_info(self, info: str) -> None:
        """Set the signal legend (computed by the console: which node each read comes from).
        Shown in the frame title (the grey strip)."""
        info = str(info or "")
        if info == self._signal_info:
            return
        self._signal_info = info
        self._refresh_title()

    def set_status(self, text: str, *, error: bool) -> None:
        # No per-shot status line in the panel any more (it needed a footer, which broke the
        # height proportion).  The status lives in the Setting popup + the Setting-button
        # tooltip; an error turns the Setting button red.  Restyle only on the ok<->error edge.
        # Render workers return string-only outcomes.  This funnel is therefore
        # Qt-owner-only; no worker ever reaches a widget or parks a mutable
        # exception on the card.
        if QtCore.QThread.currentThread() is not self.thread():
            raise RuntimeError("panel status is Qt-owner-only")
        self._status_text = str(text)
        if hasattr(self, "status"):
            self.status.setText(str(text))
        self.setting_button.setToolTip(f"Panel settings — {text}" if text else "Panel settings")
        if error is not getattr(self, "_status_error", None):
            self._status_error = bool(error)
            colour = RED if error else GREY
            if hasattr(self, "status"):
                self.status.setStyleSheet(f"color: {colour}; background: transparent; border: none;")
            self.setting_button.set_color(colour)

    # ------------------------------------------------------------- drag to grid
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # only the border frame starts a drag; the canvas consumes its own events
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_offset = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self.raise_()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_offset is not None:
            new_pos = self.mapToParent(event.pos() - self._drag_offset)
            self.move(max(0, new_pos.x()), max(0, new_pos.y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_offset is not None:
            self._drag_offset = None
            self.setCursor(QtCore.Qt.OpenHandCursor)
            # Record the raw drop pixel as this card's (col, row); the console then REORDERS the card
            # to the ORDER position nearest that drop (:func:`drop_index` via ``dropped`` -- a drop onto
            # a card's slot displaces it, a drop past the last card appends to the bottom), and
            # :func:`pack` recomputes every pixel top-left from the new order.
            col, row = max(0, self.x()), max(0, self.y())
            if (col, row) != (self.config.col, self.config.row):
                self.config.col, self.config.row = col, row
                self.changed.emit()
            self.dropped.emit(self)         # drag-release ONLY: the console snaps the drop seed
            self.layout_changed.emit()      # re-pack (even when dropped back near the same spot)
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._place_setting_button()

    def _commit_title(self) -> None:
        """Commit the locally edited title once focus/Return finishes it.

        One string: the frame header shows it and the composer carries it as the
        dataset label, so the picture and the card can never be captioned
        differently.
        """

        text = str(self.title_edit.text())
        if text == self.config.title:
            return
        self.config.title = text
        self._invalidate_render_binding()
        self._request_display_render()
        self.changed.emit()

    def _on_size(self, size: str) -> None:
        # The first explicit operator choice wins even if it happens to equal
        # the stock 2x2 value.  No schema-driven recommendation may later
        # overwrite a user-authored size.
        had_pending_recommendation = self._initial_grid_size_pending
        self._initial_grid_size_pending = False
        size = str(size)
        if size == self.config.size:
            if had_pending_recommendation:
                # Supersede any recommended-size answer already in flight.
                # The replacement request is authored from the explicit 2x2
                # config, so the old 4x4 contract cannot be admitted/presented.
                self._invalidate_render_binding()
                self._request_display_render(force=True)
            return
        self.config.size = size
        self._invalidate_render_binding()
        if self._presented_contract is None:
            # With no painted front there is nothing to stretch.  Keep initial
            # empty-card geometry responsive while waiting for first data.
            self._apply_fixed_size()
            if (
                getattr(self, "settings_popup", None) is not None
                and self.settings_popup.isVisible()
            ):
                self._size_settings_popup()
            self.layout_changed.emit()
        # Once a front exists, geometry is presented only with the matching
        # worker raster in ``present``.  The accepted surface therefore remains
        # exactly unchanged while this latest-only request is in flight.
        self._request_display_render(force=self._presented_value is None)
        self.changed.emit()

    def _set_param(self, key: str, value) -> bool:
        key = str(key)
        spec = self._display_form_spec()
        if spec is not None and key in spec.keys:
            values = self._display_form_values()
            values[key] = value
            return self._commit_display_form_values(
                values,
                current_value_limits=self._shown_limits(),
            )
        return self._set_params({key: value})

    def _commit_display_form_values(
        self,
        values: Mapping[str, object],
        *,
        current_value_limits: tuple[float, float] | None,
        request_render: bool = True,
        emit_changed: bool = True,
    ) -> bool:
        """Apply one complete canonical frontend display form transaction."""

        contract = self._display_contract_for_authoring()
        if contract is None:
            raise RuntimeError("Grid display form has no resolved cell intent")
        base = self._display_state(contract)
        candidate, rolling_distribution = panel_display_state_from_form(
            self.config.kind,
            base,
            values,
            current_value_limits=current_value_limits,
            rolling_distribution=contract.figure.rolling_distribution,
        )
        rolling_changed = (
            rolling_distribution != contract.figure.rolling_distribution
        )
        if rolling_changed and candidate == base:
            candidate = replace(candidate, revision=base.revision + 1)
        cell_intent = (
            contract.figure.view_intent
            if self.config.kind is PlotKind.GRID
            else None
        )
        canonical_values = panel_display_form_values(
            self.config.kind,
            candidate,
            rolling_distribution=rolling_distribution,
        )
        encoded = panel_display_form_values_to_tree(
            self.config.kind,
            canonical_values,
            cell_intent=cell_intent,
        )
        obsolete = panel_display_param_keys(self.config.kind).difference(encoded)
        changed = self._commit_persisted_params(
            encoded,
            remove=obsolete,
            emit_changed=False,
        )
        if not changed and candidate == base and not rolling_changed:
            return False
        self._display_revision = candidate.revision
        if rolling_changed:
            self._invalidate_render_binding()
        surface = getattr(self, "display_form_surface", None)
        if surface is not None:
            self._reconcile_display_form_surface(surface)
        if request_render:
            self._request_current_render()
        if emit_changed:
            self.changed.emit()
        return True

    def _clear_display_view(
        self,
        *,
        request_render: bool,
        emit_changed: bool = True,
    ) -> bool:
        """Return the canonical viewport fields to home without touching style."""

        contract = self._display_contract_for_authoring()
        if contract is None or contract.figure.view_intent is ViewIntent.METER:
            return False
        values = self._display_form_values()
        keys = ["x_min", "x_max"]
        if contract.figure.view_intent is ViewIntent.IMAGE:
            keys.extend(("y_min", "y_max"))
        if all(values.get(key) is None for key in keys):
            return False
        for key in keys:
            values[key] = None
        return self._commit_display_form_values(
            values,
            current_value_limits=self._shown_limits(),
            request_render=request_render,
            emit_changed=emit_changed,
        )

    def _commit_persisted_params(
        self,
        updates: Mapping[str, object] | None = None,
        *,
        remove=(),
        request_render: bool = False,
        emit_changed: bool = True,
    ) -> frozenset[str]:
        """The only writer for persisted panel parameters.

        Callers supply one complete semantic transaction.  This method owns
        removals, equality checks, and the two observable
        effects of a persisted edit (optional render request and dirty signal).
        Pure readers must never call it.
        """

        if not isinstance(request_render, bool) or not isinstance(emit_changed, bool):
            raise TypeError("persisted parameter effects must be bool")
        incoming = {} if updates is None else dict(updates)
        removals = tuple(str(key) for key in remove)
        overlap = set(incoming).intersection(removals)
        if overlap:
            raise ValueError(
                "persisted parameter transaction both updates and removes "
                f"{tuple(sorted(overlap))}"
            )
        missing = object()
        encoded = {str(raw_key): value for raw_key, value in incoming.items()}
        changed = {
            key for key, value in encoded.items()
            if self.config.params.get(key, missing) != value
        }
        changed.update(key for key in removals if key in self.config.params)
        if not changed:
            return frozenset()
        for key in removals:
            self.config.params.pop(key, None)
        self.config.params.update(encoded)
        if request_render:
            self._request_display_render()
        if emit_changed:
            self.changed.emit()
        return frozenset(changed)

    def _set_params(self, updates: Mapping[str, object]) -> bool:
        """Commit one semantic parameter transaction and request one compose.

        Display fields are diverted through their frontend state handler.  This
        writer remains for view binding, cadence, thresholds, and other
        non-display panel facts.
        """

        effective = {str(key): value for key, value in dict(updates).items()}
        display_keys = panel_display_param_keys(self.config.kind)
        overlap = display_keys.intersection(effective)
        if overlap:
            spec = self._display_form_spec()
            if spec is None or not set(effective).issubset(spec.keys):
                raise ValueError(
                    "display transaction targets another PlotKind form: "
                    f"{tuple(sorted(overlap))}"
                )
            values = self._display_form_values()
            values.update(effective)
            return self._commit_display_form_values(
                values,
                current_value_limits=self._shown_limits(),
            )
        changed = self._commit_persisted_params(
            effective,
            request_render=True,
        )
        return bool(changed)

    def _shown_limits(self):
        """The value window the last composed front actually used, or None.

        Read off the front rather than recomputed: the whole point of pinning is
        to freeze WHAT IS ON SCREEN, and a freshly derived envelope is a
        different number the moment the data moved.
        """

        payload = getattr(self._pending_frame, "panels", None)
        front = getattr(self.board, "front_frame", None)
        panels = payload or getattr(front, "panels", ())
        for panel in panels or ():
            display = panel.display_payload
            for attr in ("color_limits",):
                limits = getattr(display, attr, None)
                if limits:
                    return (float(limits[0]), float(limits[1]))
            viewport = getattr(display, "viewport", None)
            for attr in ("y_limits", "count_limits"):
                limits = getattr(viewport, attr, None)
                if limits:
                    return (float(limits[0]), float(limits[1]))
        return None

    def _request_display_render(self, *, force: bool = False) -> None:
        """Commit one display revision and enqueue one latest-only compose.

        This method never rasterises and never waits.  The console freezes the
        latest immutable source value and coalesces requests while its one
        render worker is busy.
        """

        self._display_revision += 1
        self._request_current_render(force=force)

    def _request_current_render(
        self,
        *,
        force: bool = False,
        surface=None,
    ) -> None:
        # ``surface`` is an additional Edit-host route, not part of the
        # established live-card callback contract.  Keep ordinary display
        # commits source-compatible with any composition-root callback that
        # accepts only ``(card, force=...)``; only an actual secondary host
        # needs the explicit route.
        if surface is None:
            self._render_request(self, force=bool(force))
        else:
            self._render_request(
                self,
                force=bool(force),
                surface=surface,
            )

    def set_selectors_enabled(self, on: bool) -> None:
        """The console header's "Selectors" switch for THIS card: remember the desired state and
        gate the CURRENT plotter now (in place -- no rebuild, no flash).  Every later rebind /
        focus swap re-applies it through ``_apply_selectors_state``, so a fresh figure always
        inherits the switch."""
        enabled = bool(on)
        changed = enabled != self._selectors_on
        self._selectors_on = enabled
        self._apply_selectors_state()
        if changed:
            self.selectors_enabled_changed.emit(enabled)

    @property
    def selectors_enabled(self) -> bool:
        """The one console-authored selector switch state for this panel."""

        return bool(self._selectors_on)

    def _apply_selectors_state(self) -> None:
        """Carry the board header's Selectors switch onto this card's surface.

        The switch is the card's state, not the surface's: it is stored here so
        a surface bound later comes up matching the switch instead
        of coming up live behind the operator's back.
        """

        board = self.board
        if board is not None and hasattr(board, "set_selectors_enabled"):
            board.set_selectors_enabled(bool(self._selectors_on))

    def retire_source_generation(self) -> None:
        """Retire every exact-revision state owned by the current live source.

        A panel binding (the authored signal name, size and display choices)
        survives a producer restart.  Its accepted pixels, selector commits,
        Fit command/result and worker identities do not: all of them name one
        immutable producer generation.  Keeping any one of those facts would
        republish retired ancestry on the replacement generation.

        This is the card-side half of the composition root's causal retirement
        transaction.  It mutates only the stable card/host in place; no widget
        or renderer is rebuilt, and publication notification is deliberately
        suppressed because the data/presentation owners withdraw the complete
        causal closure atomically in the surrounding transaction.
        """

        pending_origin = self._pending_interaction_origin
        pending_host = self._pending_interaction_host
        if pending_origin is not None and pending_host is not None:
            pending_host.discard_pending_interaction(pending_origin)
        self._pending_interaction_origin = None
        self._pending_interaction_host = None
        self._pending_interaction_revision = None
        self._pointer_interaction_pin = None

        self._retire_all_fits(
            emit_changed=False,
            notify_output=False,
        )
        self._invalidate_render_binding()
        self._candidate_value = None
        self._candidate_publication = None
        self._grid_focus = None
        self._presented_figure = None
        self._presented_display = None
        self._presented_contract = None
        self._presented_value = None
        self._presented_publication = None
        self._presented_render_request_revision = None
        self._clear_figure_outputs(notify=False)

        board = self.board
        if board is not None:
            board.clear()
        self._render_version = -1

    # ------------------------------------------------------------- plot lifecycle
    def _invalidate_render_binding(self) -> None:
        """Invalidate only the worker request identity, never the Qt surface.

        Source/title/size changes cause the worker to replace its composer from
        the next request's frozen ``source_key``.  The accepted front stays in
        place until that replacement is ready, so an edit cannot flash an empty
        card or rebuild a widget subtree.
        """

        self._pending_frame = None
        self._pending_faceted_result = None
        self._pending_figure = None
        self._pending_display = None
        self._pending_contract = None
        self._pending_value = None
        self._pending_publication = None
        self._pending_render_request_revision = None
        self._render_request_revision += 1
        self._requested_signature = None
        self._latest_requested_source_ref = None
        self._latest_requested_source_key = None
        self._latest_requested_display_revision = None

    def _teardown_plot(self) -> None:
        """Drop this card's surface, leaving nothing painted behind it.

        Hide synchronously, then keep the host parented until deferred deletion.
        Reparenting a QWidget to ``None`` would temporarily make it a top-level
        window, which is never a legal Figure lifecycle state.
        """

        board = self.board
        if (
            self._pending_interaction_origin is not None
            and self._pending_interaction_host is not None
        ):
            self._pending_interaction_host.discard_pending_interaction(
                self._pending_interaction_origin
            )
        self._pending_interaction_origin = None
        self._pending_interaction_host = None
        self._pending_interaction_revision = None
        self._pointer_interaction_pin = None
        self.board = None
        self._pending_frame = None
        self._pending_faceted_result = None
        self._pending_figure = None
        self._pending_display = None
        self._pending_contract = None
        self._pending_value = None
        self._pending_publication = None
        self._pending_render_request_revision = None
        self._candidate_value = None
        self._candidate_publication = None
        self._grid_focus = None
        self._presented_figure = None
        self._presented_display = None
        self._presented_contract = None
        self._presented_value = None
        self._presented_publication = None
        self._presented_render_request_revision = None
        self._clear_figure_outputs(notify=False)
        self._requested_signature = None
        self._latest_requested_source_ref = None
        self._latest_requested_source_key = None
        self._latest_requested_display_revision = None
        if board is not None:
            self.canvas_holder.removeWidget(board)
            board.hide()
            board.deleteLater()

    def shutdown(self) -> None:
        """Release this card's Qt surface.

        Worker requests contain no card/QWidget reference and are rejected by
        panel identity after removal, so teardown never waits on raster work.
        """

        for state in tuple(self._fit_surfaces.values()):
            self.fit_cancel_requested.emit(state.surface_id)
            self.fit_output_clear_requested.emit(state.surface_id)
            self._clear_fit_surface_overlay(state)
        self._fit_surfaces.clear()
        self._live_fit_pane = None
        self._teardown_plot()
