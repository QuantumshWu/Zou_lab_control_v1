"""The TaskConsole window: tabs, cards, logic rows, and lifecycle wiring.

Panel widgets, board geometry, editing, and worker raster ownership each live in
their own module; this file owns only the application window and their wiring.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, Future
import inspect
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence

from PyQt5 import QtCore, QtWidgets

import zlc_frontend.qt_widgets as _qt_widgets
from zlc_frontend.qt_widgets import (
    ACCENT,
    FluentButton,
    FluentComboBox,
    FluentFrame,
    FluentLabel,
    FluentLineEdit,
    FluentScrollArea,
    FluentStatusDot,
    FluentStatusStrip,
    FluentSwitch,
    FluentTabWidget,
    FormRuntimeContext,
    GREEN,
    GREY,
    ORANGE,
    QtOwnerWake,
    WINDOW_SCREEN_FRACTION,
    YELLOW,
    ensure_qt_app,
    error_summary,
    fluent_message,
    fluent_widget_stylesheet,
    launch_fluent_window,
    release_window,
    scaled_px,
    screen_fit_window_size,
    set_fluent_scale,
    wait_for_owner_retirement,
    window_pad,
)
from .console_state import (
    TaskConsoleState,
    default_console_state,
)
from .console_records import (
    DEFAULT_UPDATE_MS,
    LogicNodeConfig,
    PANEL_KINDS,
    PanelConfig,
    UPDATE_INTERVALS,
    console_signal_key,
    panel_signal_key,
)
from zlc_frontend.shape_text import describe_dataset_shape, indexed_unique_name
from zlc_data import (
    BlockId,
    DataTransformSpec,
    DatasetRevisionRef,
    StreamGenerationId,
    materialize_value_dataset,
)
from zlc_plot import FitEvent, PlotKind, RasterOperation, SelectionData, SelectorKind
from .panel_board import (
    GAP,
    PanelBoard,
    board_width,
    drop_index,
    opaque_white_composite,
    pack,
)
from zlc_neutral_atom.processing.signal_plane import (
    DerivedSignalOutput,
    SignalDataPlane,
    SignalPublication,
)
from zlc_neutral_atom.runtime.signal_source import (
    authoritative_signal_event_source,
)
from zlc_neutral_atom.artifact_output import ArtifactOutputDeclaration
from zlc_neutral_atom.catalog import definition_key_from_tree, definition_key_to_tree
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.input_spec import ArtifactInputSpec, DatasetInputSpec
from zlc_neutral_atom.installation import DeviceCatalogView
from zlc_neutral_atom.logic_node import (
    ArtifactOutputSpec,
    DatasetOutputSpec,
    LogicNodeDescriptor,
)
from zlc_neutral_atom.runtime.hosted_run import LogicNodeHost
from .panel_card import PanelCard, PanelSurfaceUpdate
from .panel_editor import PanelEditor
from .logic_node_editor import LogicNodeEditor
from .logic_node_row import LogicNodeRow
from .published_signal_row import PublishedSignalRow
from .layout_repository import (
    load_task_console_state,
    resolve_task_state,
    save_task_console_state,
    task_files_dir as _task_files_dir,
)


_ConsoleNode = LogicNodeHost
_AREA_DATA_OUTPUT = "area.data"
_CROSS_DATA_OUTPUT = "cross.data"
_FIT_OUTPUT_PREFIX = "fit."
_CROSS_CONTRACT_ID = "zlc.plot.cross-value"
_FIT_CONTRACT_ID = "zlc.plot.fit-parameter"
_SignalTopologyState = Literal[
    "running",
    "declared-not-started",
    "retained-final",
    "retained-view",
]


@dataclass(frozen=True, slots=True)
class _SignalTopologyEntry:
    """One exact producer/output entry in the console-owned topology."""

    state: _SignalTopologyState
    label: str
    node: _ConsoleNode | PanelCard | None
    declaration: DatasetOutputSpec | ArtifactOutputSpec
    kind: str


@dataclass(frozen=True, slots=True)
class _SignalTopologyProjection:
    """One event-owned immutable projection of the signal inventory.

    Signal topology is structural UI state.  It changes only at the explicit
    mutation boundaries that mark ``_signal_info_dirty``; camera frames and
    selector gestures merely consume it.  Keeping the topology and both label
    maps in one value prevents a render tick from rediscovering every logic
    node, card, and Figure output, and prevents independently refreshed label
    caches from disagreeing.
    """

    topology: Mapping[str, _SignalTopologyEntry]
    axis_labels: Mapping[str, str]
    short_names: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _PanelSelectionRoute:
    """The one routing fact retained for an active panel selector."""

    owner_id: str
    output_name: str
    source_name: str
    source_generation: StreamGenerationId
    source_contract_id: str
    output_schema: object
    selection: object | None
    event_source: object | None
    event_output_name: str | None
    generation: StreamGenerationId


# ====================================================================== console
class TaskConsole(QtWidgets.QWidget):
    """The dashboard window body: header bar + drag-and-snap panel board."""

    def __init__(
        self,
        *,
        state: TaskConsoleState | None = None,
        descriptors: tuple[LogicNodeDescriptor, ...] = (),
        device_catalog: DeviceCatalogView,
        host_factory=None,
        data_plane: SignalDataPlane,
        project_root: Path,
        pulses_root: Path,
        tasks_root: Path,
        figures_root: Path,
        scale: float | None = None,
        window_ratio: float = WINDOW_SCREEN_FRACTION,
        window_px: tuple[int, int] | None = None,
        embedded: bool = False,
    ):
        ensure_qt_app()
        set_fluent_scale(scale)
        super().__init__()
        roots = tuple(
            Path(value).expanduser()
            for value in (project_root, pulses_root, tasks_root, figures_root)
        )
        if any(not value.is_absolute() for value in roots):
            raise ValueError("TaskConsole roots must be absolute")
        (
            self._project_root,
            self._pulses_root,
            self._tasks_root,
            self._figures_root,
        ) = tuple(value.resolve() for value in roots)
        # One queued edge owns every worker/data-plane -> QWidget transition.
        # Display cadence is deliberately separate and never drains lifecycle
        # mailboxes.
        self._owner_wake = QtOwnerWake(self)
        self._owner_wake.bind(self._owner_cycle)
        self._owner_event_lock = threading.Lock()
        self._lifecycle_wake_pending = False
        self._data_wake_pending = False
        self._surface_wake_pending = False
        self._accept_data_wake = False
        self._surface_batches: deque[tuple[PanelSurfaceUpdate, ...]] = deque()
        self._selection_routes: dict[str, _PanelSelectionRoute] = {}
        self._fit_output_names: dict[str, tuple[str, ...]] = {}
        self._owner_close_lock = threading.Lock()
        self._owner_retiring = False
        self._permanently_closed = False
        self._owner_closed = threading.Event()
        self._display_suspended = False
        # MONITOR SEAM: the board reads one immutable aggregate
        # front per tick.
        # The plane owns the live slots the RUN seam's start closures register;
        # each producer contributes its own latest atomic transaction.  Sharing
        # this front prevents intra-producer tearing but makes no same-shot claim
        # across independent runs.
        if not isinstance(data_plane, SignalDataPlane):
            raise TypeError("TaskConsole requires the Experiment SignalDataPlane")
        self._data = data_plane
        self._data_wake_token = None
        self._tick_data = self._data.freeze()
        self._signal_metadata_front = self._tick_data
        self._signal_schemas = self._front_schemas(
            self._tick_data
        )
        # EMBEDDED mode (the figure viewer hosts a whole TaskConsole in one pane): a stripped,
        # RESIZABLE console -- no Logic tab, no whole-board Pause/Save image/Save/Load buttons (a
        # loaded static figure has nothing to acquire, freeze, or persist as a layout) -- and, crucially,
        # NOT size-frozen: the parent layout stretches it so the gravity board reads the REAL viewport
        # width and reflows into 2+ columns.  Standalone mode owns a fixed top-level
        # size.  Omitted embedded controls are represented by None.
        self.embedded = bool(embedded)
        if not isinstance(device_catalog, DeviceCatalogView):
            raise TypeError("device_catalog must be DeviceCatalogView")
        if host_factory is not None and not callable(host_factory):
            raise TypeError("host_factory must be callable or None")
        self._device_catalog = device_catalog
        self._host_factory = host_factory
        self._panel_teardown_phases: dict[int, set[str]] = {}
        descriptor_values = tuple(descriptors)
        if any(
            not isinstance(value, LogicNodeDescriptor)
            for value in descriptor_values
        ):
            raise TypeError("descriptors must contain LogicNodeDescriptor values")
        self._descriptors = descriptor_values
        self._descriptor_by_key = {
            value.definition.key: value for value in descriptor_values
        }
        if len(self._descriptor_by_key) != len(descriptor_values):
            raise ValueError("TaskConsole descriptor keys must be unique")
        self.state = state or default_console_state()
        self.window_ratio = float(window_ratio)
        self._window_px = window_px
        self.cards: list[PanelCard] = []
        # The signal legend is a projection of two event-owned facts: provider
        # topology and each card's binding.  Mutations below mark it dirty and
        # reconcile it once; the refresh timer never rediscovers topology by
        # enumerating every provider/card merely to learn that nothing changed.
        self._signal_info_dirty = True
        self._signal_projection: _SignalTopologyProjection | None = None
        self._card_topology_identities: dict[int, tuple[str, str, str]] = {}
        # The Logic-tab nodes.  Each entry maps a LogicNodeRow -> the live state for
        # that node: its built node (None until Started) + its Edit tab.
        self.logic_nodes: list[LogicNodeRow] = []
        self._passive_publisher_rows: dict[str, PublishedSignalRow] = {}
        self._logic_nodes: dict[int, _ConsoleNode | None] = {}
        # id(row) -> the LAST built node, kept after Stop so the retained final
        # display front still has its producer identity.  Cleared on restart or
        # row removal.
        self._last_node: dict[int, _ConsoleNode] = {}
        self._logic_editors: dict[int, "LogicNodeEditor"] = {}  # id(row) -> Edit tab
        self._building = False
        self._address: str | None = None
        # The folder the LAST panel "Save Fig" wrote into -- so a panel's Edit-tab save
        # picker reopens at the same place across panels and across reopens (remembered
        # for the life of the process / Jupyter kernel, like the pulse GUI's save dir).
        # None until the first save -> the picker defaults to the unified
        # operator-output tree, never the reloadable tasks/ layout folder.
        self._last_save_dir: str | None = None
        # Per-panel editors: one PanelEditor per opened PLOT panel, hosted as a
        # closable tab (keyed by id(card)).
        self._panel_editors: dict[int, "PanelEditor"] = {}
        self._retiring_panel_editors: set["PanelEditor"] = set()
        # Default live-result panels are ordinary signal consumers.  Their
        # transient lifetime is tracked by panel identity, while each Logic row
        # remains independently editable and stoppable.  Resource admission is
        # owned exclusively by the Experiment application.
        self._transient_panel_ids: set[str] = set()
        self._window = None
        self._pending_window_action: str | None = None
        self._window_retry_scheduled = False

        # Multi-rate refresh: the timer ticks at the BASE interval (the smallest panel
        # update_ms, which divides every other so the rates co-align); each panel redraws
        # every update_ms // base ticks.  _tick_count counts base ticks since the last re-base.
        self._tick_count = 0
        self._base_interval_ms = int(self.state.interval_ms)
        # Display reads each signal at its own latest value.  A producer that owns
        # related view signals publishes them atomically, so companion values remain
        # shot-coherent without console-side joins.

        self._build_ui()
        self.load_state(self.state)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._terminal_timer = QtCore.QTimer(self)
        self._terminal_timer.setInterval(min(UPDATE_INTERVALS))
        self._terminal_timer.timeout.connect(self.request_owner_wake)
        self._paused = False                     # display-only board freeze
        self._recompute_tick_interval()          # base = min panel update_ms (sets the interval)
        self._sync_terminal_poll_timer()

    # ------------------------------------------------------------------ UI
    def _target_console_size(self) -> QtCore.QSize:
        """Window size from the primary screen (the shared GUI sizing rule)."""

        if self._window_px is not None:
            return QtCore.QSize(int(self._window_px[0]), int(self._window_px[1]))
        return screen_fit_window_size(self.window_ratio)

    def _build_ui(self) -> None:
        self.setWindowTitle("TaskConsole@Zou lab")
        self.setStyleSheet(fluent_widget_stylesheet())
        # Standalone: a fixed window (the shared GUI sizing rule).  EMBEDDED: the parent owns
        # the size and this is only one pane of it, so the console expands into whatever the
        # layout gives and demands no minimum of its own.  The shared rule sizes
        # only the standalone window; an embedded body is divided by its host.
        if self.embedded:
            self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        else:
            self.setFixedSize(self._target_console_size())
        root = QtWidgets.QVBoxLayout(self)
        # SINGLE SOURCE for every window-edge inset: ``window_pad(1)`` (== ``scaled_px(WINDOW_PAD)``).
        # The window title (qt_widgets draws it at ``scaled_px(TITLE_LEFT_INSET)`` == ``scaled_px(WINDOW_PAD)``)
        # pins to this SAME left column, so the body's left edge lines up under the "task_console@zoulab"
        # title text -- one shared left edge.
        margin = window_pad(1)
        # EMBEDDED: no top/bottom inset -- the host (the figure viewer) already frames the console with
        # its own root margin, so the console's FIRST card (the header) sits flush at the pane top and its
        # visible top/bottom edges line up with the Info card beside it.  STANDALONE: the top AND bottom
        # insets are the SAME ``window_pad(1)`` as left/right, so the padding to all four window edges is
        # identical (the bottom gap now matches the sides).
        v_margin = 0 if self.embedded else window_pad(1)
        # The tab card carries a flat 1 px border (no drop shadow), so no bottom-bleed headroom is
        # reserved -- top and bottom insets are both the plain ``v_margin`` (0 embedded, so the header
        # lines up flush with the Info card beside it).
        root.setContentsMargins(margin, v_margin, margin, v_margin)
        # A clear GAP separates the three rows -- the header card, the (hidden) task banner and
        # the tab card -- so they read as DISTINCT rounded cards on the grey window background.
        # The gap is HALF the window pad (``window_pad(0.5)``), so every spacing on screen is a clean
        # multiple of the ONE ``WINDOW_PAD`` unit.  The header is flat (no drop shadow) and the tab bar
        # draws no top base line, so this gap reads as clean card separation rather than a hard line.
        root.setSpacing(window_pad(0.5))

        # FLAT header (no drop shadow): the shadow's soft bottom edge cast a thin grey line
        # into the gap right above the tab strip (the "line above the tabs").  The tab widget
        # below carries its own card, so the header needs no elevation -- it is a plain top bar.
        header_frame = FluentFrame(bordered=False)
        header_frame.setFixedHeight(scaled_px(48, minimum=38))
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(scaled_px(12), scaled_px(6), scaled_px(12), scaled_px(6))
        header.setSpacing(scaled_px(7, minimum=4))

        self.status_dot = FluentStatusDot(size=16)
        self.status_dot.set_color(GREEN)
        self.name_edit = FluentLineEdit(self.state.name)
        self.name_edit.setPlaceholderText("task name")
        self.name_edit.setFixedWidth(scaled_px(150, minimum=110))
        self.name_edit.textChanged.connect(self._mark_dirty)
        # Persistent telemetry belongs to the header: it is orthogonal to the event/status
        # channel below and therefore never disappears behind a task message or node error.
        self.summary = FluentLabel("")
        self.summary.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.kind_combo = FluentComboBox()
        # Add Panel offers EXACTLY the four kinds the user designs with -- nothing
        # invented, no composite:
        #   * "Plot: X"        -- a pure VIEW; shows nothing until you select a
        #                         signal in its Setting (it NEVER starts anything);
        #   * "Measurement: X" -- an acquisition logic node (the camera live stream,
        #                         or a swept measurement) added to the LOGIC tab;
        #   * "Processor: Y"   -- a reactive transform logic node (the "func" layer);
        #   * "Task: Z"        -- a one-shot orchestration logic node (e.g. calibrate).
        # A logic node (measurement/processor/task) is added STOPPED to the Logic tab;
        # you Start/Stop it from its own Edit.  A plot is added to the Monitor board.
        # The dropdown offers only current TaskConsole panel kinds: each has an
        # end-to-end typed live payload and renderer.  Static/document figure
        # kinds are not persisted as panel records.
        for key, label in PANEL_KINDS.items():
            self.kind_combo.addItem(f"Plot: {label}", key)
        # The node layers come directly from the catalog view.  Every entry
        # carries its own display label and DefinitionKey; Camera is an ordinary
        # catalog entry rather than a window-owned special case.
        for kind, layer in (("measurement", "Measurement"),
                            ("processor", "Processor"), ("task", "Task")):
            for descriptor in self._catalog_specs(kind):
                self.kind_combo.addItem(
                    f"{layer}: {descriptor.definition.title}",
                    definition_key_to_tree(descriptor.definition.key),
                )
                # The definition's prose title is the row's tooltip: the menu stays
                # scannable while the full sentence stays one hover away.
                self.kind_combo.setItemData(self.kind_combo.count() - 1,
                                            descriptor.description, QtCore.Qt.ToolTipRole)
        self.kind_combo.setFixedWidth(scaled_px(170, minimum=130))
        add_button = FluentButton("Add Panel", color=ACCENT)
        add_button.clicked.connect(self._add_panel)
        # "Selectors" switch: the Monitor board is display-only BY DEFAULT (every panel builds
        # its selector layer but parks it inactive; the wheel scrolls the board).
        # Flip ON to arm the full selector layer (zoom/pan, area, cross,
        # draggable threshold/clim lines) on every dashboard panel IN PLACE -- no rebuild, and a
        # panel added while ON inherits it (see _attach_card).  A DISPLAY control like Pause, so
        # it remains independent of Run lifecycle.
        self.selectors_switch = FluentSwitch("Selectors")
        self.selectors_switch.setChecked(False)
        self.selectors_switch.setToolTip(
            "OFF: panels are display-only (wheel scrolls the board).\n"
            "ON: zoom / area / cross / draggable lines work on every panel\n"
            "(the wheel zooms inside a plot, like the Edit tab).")
        self.selectors_switch.toggled.connect(self._toggle_selectors)
        # Whole-console layout + monitor controls: Save/Load persist the LAYOUT (json); Pause/Resume
        # freezes every plot at once (a display freeze -- acquisition keeps running); Save image grabs
        # the WHOLE board region to a PNG.  EMBEDDED (figure-viewer host) has nothing to acquire, freeze
        # or persist as a standalone layout, so all four are OMITTED -- set to None so every reference
        # (``_mark_dirty``, ``_toggle_pause`` ...) guards on existence.
        if self.embedded:
            self.save_button = None
            load_button = None
            self.pause_button = None
            self.save_image_button = None
        else:
            self.save_button = FluentButton("Save", color=ACCENT)
            self.save_button.clicked.connect(self.save_to_file)
            load_button = FluentButton("Load", color=ORANGE)
            load_button.clicked.connect(self.load_from_file)
            self.pause_button = FluentButton("Pause", color=ORANGE)
            self.pause_button.clicked.connect(self._toggle_pause)
            self.save_image_button = FluentButton("Save image", color=ACCENT)
            self.save_image_button.clicked.connect(self._save_board_image)

        for widget in (self.status_dot, self.name_edit):
            header.addWidget(widget)
        header.addWidget(self.summary, 1)
        # Add Panel stays even when embedded (a loaded figure is re-wired + extra panels added), and so
        # does the Selectors switch (inspecting a loaded figure is exactly when the selectors help);
        # the four whole-console buttons only when they exist.
        for widget in (self.kind_combo, add_button, self.selectors_switch,
                       self.pause_button, self.save_image_button, self.save_button, load_button):
            if widget is not None:
                header.addWidget(widget)
        root.addWidget(header_frame)

        # PERSISTENT status strip -- the ONE always-visible line between the header and the
        # tabs.  Its content switches by PRIORITY (node error > display-behind
        # advisory; idle is empty, see _update_summary).  Run progress and Stop
        # stay row-local, so the board never acquires a second admission mode.
        # Its fixed height keeps status changes from moving the layout.
        self.status_strip = FluentStatusStrip()
        root.addWidget(self.status_strip)

        # TWO permanent tabs:
        #   * Monitor -- the live drag-and-snap board of PLOT panels (pure views);
        #   * Logic   -- the list of LOGIC NODES (measurement / processor / task),
        #               each a row with a status dot + Edit.
        # A plot panel's Edit and a logic node's Edit each open their OWN closable
        # tab beside these two.
        self.tabs = FluentTabWidget()
        dash_tab = QtWidgets.QWidget()
        dash_tab.setStyleSheet("background: transparent;")
        dash_layout = QtWidgets.QVBoxLayout(dash_tab)
        dash_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll = FluentScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.board = PanelBoard()
        self.scroll.setWidget(self.board)
        dash_layout.addWidget(self.scroll, 1)
        # Re-pack the gravity board when the viewport WIDTH changes (window resized): cards wrap at
        # the live viewport width, so narrowing the window must reflow them into fewer columns (else
        # they would just clip behind a horizontal scrollbar).  Watch the viewport's resize via an
        # event filter and re-arrange only on an actual width change (no-op recursion guard).
        self._board_view_w = 0
        self.scroll.viewport().installEventFilter(self)
        self.tabs.add_permanent_tab(dash_tab, "Monitor")
        # A one-shot re-arrange AFTER the first show, when the scroll viewport finally has its real
        # width (0 during construction): so an EMBEDDED console (hosted in the figure viewer) packs
        # its board against the true pane width immediately, not a stale 0.
        QtCore.QTimer.singleShot(0, self._arrange_if_cards)

        # Logic tab: a scrolled top-packed column of LogicNodeRow cards.  OMITTED when embedded (the
        # figure viewer hosts pure VIEWS of a loaded static figure -- no acquisition logic to run), so
        # ``logic_scroll`` / ``logic_layout`` / ``logic_hint`` stay None and their few users guard.
        if self.embedded:
            self.logic_scroll = None
            self.logic_layout = None
            self.logic_hint = None
        else:
            logic_tab = QtWidgets.QWidget()
            logic_tab.setStyleSheet("background: transparent;")
            logic_outer = QtWidgets.QVBoxLayout(logic_tab)
            logic_outer.setContentsMargins(0, 0, 0, 0)
            self.logic_scroll = FluentScrollArea()
            logic_body = QtWidgets.QWidget()
            logic_body.setStyleSheet("background: transparent;")
            self.logic_layout = QtWidgets.QVBoxLayout(logic_body)
            lm = scaled_px(10, minimum=6)
            self.logic_layout.setContentsMargins(lm, lm, lm, lm)
            self.logic_layout.setSpacing(scaled_px(8, minimum=5))
            self.logic_hint = FluentLabel(
                "No logic nodes yet.  Add a Measurement / Processor / Task from the header "
                "(it starts STOPPED); open its Edit to set parameters and Start it.")
            self.logic_hint.setWordWrap(True)
            self.logic_hint.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            self.logic_layout.addWidget(self.logic_hint)
            self.logic_layout.addStretch(1)
            self.logic_scroll.set_width_bounded_widget(logic_body)
            logic_outer.addWidget(self.logic_scroll, 1)
            self.tabs.add_permanent_tab(logic_tab, "Logic")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.tab_close_requested.connect(self._on_editor_tab_closed)
        # The tab card (Monitor / Logic / Edit) carries a flat 1 px border (no drop shadow), so the row
        # spacing above it is the whole gap -- no extra top-bleed headroom is topped up.
        root.addWidget(self.tabs, 1)

    # ------------------------------------------------------------------ state
    def read_state(self) -> TaskConsoleState:
        # Flush every OPEN logic-node Edit form into its config first, so Save captures
        # the current parameter values even when the node was never started (a
        # layout persists Edit params, not just geometry).  Plot params already write
        # through to ``card.config.params`` live, so panels need no flush.
        drafts = []
        for row in self.logic_nodes:
            editor = self._logic_editors.get(id(row))
            if editor is None:
                continue
            # Validate every open draft before mutating any authored config, so
            # one invalid tab cannot leave a half-committed Save attempt.
            drafts.append((row, editor.collect_binding()))
        for row, (authored, inputs) in drafts:
            self._commit_logic_authored_values(
                row,
                authored=authored,
                inputs=inputs,
            )
        return TaskConsoleState(
            name=self.name_edit.text().strip() or "task",
            interval_ms=self.state.interval_ms,
            panels=[
                card.config
                for card in self.cards
                if card.panel_id not in self._transient_panel_ids
            ],
            logic=[row.node for row in self.logic_nodes],
        )

    def load_state(self, state: TaskConsoleState) -> None:
        if not isinstance(state, TaskConsoleState):
            raise TypeError("state must be TaskConsoleState")
        # Loading consumes no caller-owned objects.  Teardown intentionally mutates
        # the live PanelConfig/LogicNodeConfig graph, so freeze a private desired
        # graph before the first stop/remove and install only that graph.
        desired_state = TaskConsoleState.from_dict(state.to_dict())
        for config in desired_state.panels:
            PanelCard.validate_config(config)
        for node in desired_state.logic:
            spec = self._spec_for_logic(node)
            if spec is None:
                raise ValueError(
                    "TaskConsole layout references an unknown or wrong-kind "
                    "DefinitionKey"
                )
            unknown_authored = set(node.authored) - set(spec.authoring_schema.keys)
            unknown_inputs = set(node.inputs) - {
                value.key for value in spec.input_specs
            }
            if unknown_authored or unknown_inputs:
                raise ValueError(
                    f"{node.title!r} contains fields absent from its descriptor; "
                    f"authored={tuple(sorted(map(str, unknown_authored)))}, "
                    f"inputs={tuple(sorted(map(str, unknown_inputs)))}"
                )
        self._building = True
        try:
            for row in list(self.logic_nodes):
                if not self._stop_logic_node(
                    row,
                    _silent=True,
                ):
                    raise RuntimeError(
                        "logic-node stop is pending; load the console state after "
                        "the owner reaches terminal"
                    )
            for row in list(self.logic_nodes):
                if not self._remove_logic_node(row, _state_change=False):
                    raise RuntimeError("stopped logic node could not be removed")
            for card in list(self.cards):
                if not self._remove_panel(card):
                    raise RuntimeError(
                        "cannot remove a panel while loading the new console state"
                    )
            self.state = desired_state
            self.name_edit.setText(desired_state.name)
            for config in desired_state.panels:
                self._attach_card(self._new_panel_card(config))
            for node in desired_state.logic:
                self._attach_logic_node(node)    # always STOPPED -- Start is manual
            self._arrange()
        finally:
            self._building = False
        self._refresh_signal_info()
        self._recompute_tick_interval()    # the loaded panels' rates set the timer base
        self._update_summary()

    def _new_panel_card(
        self,
        config: PanelConfig,
    ) -> PanelCard:
        """Build the thin card around the sole ``zlc_plot`` surface owner."""
        card = PanelCard(
            config, parent=self.board,
            signal_groups_provider=self._panel_signal_groups,
        )
        return card

    def _attach_card(self, card: PanelCard) -> None:
        card.setParent(self.board)
        card.show()
        card.changed.connect(self._mark_dirty)
        # ``changed`` also covers size/display edits.  Compare the exact small
        # structural identity consumed by signal topology; ordinary viewport
        # edits never enumerate providers, while title/kind changes cannot
        # leave Figure-output labels stale.
        card.changed.connect(
            lambda c=card: self._card_topology_identity_changed(c)
        )
        card.dropped.connect(self._snap_dropped_card)   # drag-release only: snap BEFORE the re-pack
        card.layout_changed.connect(self._arrange)
        card.update_interval_changed.connect(self._recompute_tick_interval)
        card.remove_requested.connect(self._remove_panel)
        card.edit_requested.connect(self._edit_card)
        card.selection_ready.connect(
            lambda result, parents, c=card: self._accept_card_selection(
                c, result, parents
            )
        )
        card.fit_ready.connect(
            lambda event, parents, c=card: self._accept_card_fit(c, event, parents)
        )
        # a panel added (or loaded) while the header's "Selectors" switch is ON inherits it --
        # the guard covers construction order (state panels may attach before the header exists).
        switch = getattr(self, "selectors_switch", None)
        if switch is not None:
            card.set_selectors_enabled(switch.isChecked())
        self.cards.append(card)
        self._card_topology_identities[id(card)] = self._card_topology_identity(card)
        self._signal_topology_changed()
        self._recompute_tick_interval()        # a new panel's rate may change the timer base

    @staticmethod
    def _figure_route_owner(card: PanelCard, output_name: str) -> str:
        return f"figure/{card.panel_id}/{output_name}"

    @staticmethod
    def _selector_output_name(kind: SelectorKind) -> str | None:
        if kind is SelectorKind.CROSSHAIR:
            return _CROSS_DATA_OUTPUT
        if kind in {SelectorKind.AREA, SelectorKind.X_RANGE}:
            return _AREA_DATA_OUTPUT
        return None

    @staticmethod
    def _panel_materialized_ref(
        card: PanelCard,
        output_name: str,
        source_ref,
        schema,
    ) -> DatasetRevisionRef:
        identity = f"{card.panel_id}/{output_name}"
        return DatasetRevisionRef(
            BlockId(f"panel-materialized/{identity}"),
            StreamGenerationId(f"panel-materialized/{identity}"),
            schema.fingerprint,
            source_ref.revision,
        )

    def _withdraw_selection_route(
        self,
        card: PanelCard,
        output_name: str,
        *,
        refresh: bool = True,
    ) -> bool:
        owner_id = self._figure_route_owner(card, output_name)
        existed = self._selection_routes.pop(owner_id, None) is not None
        self._data.withdraw_derived(owner_id)
        if existed and refresh:
            self._promote_data_front(self._data.freeze())
            self._signal_topology_changed()
        return existed

    def _accept_card_selection(
        self,
        card: PanelCard,
        result: object,
        publications: object,
    ) -> None:
        """Publish one committed zlc_plot selection with all exact parents."""

        if card not in self.cards:
            return
        if publications is None:
            if isinstance(result, SelectorKind):
                output_name = self._selector_output_name(result)
                if output_name is not None:
                    self._withdraw_selection_route(card, output_name)
            return
        if not isinstance(result, SelectionData):
            return
        output_name = self._selector_output_name(result.selector.kind)
        if output_name is None:
            return
        parents = tuple(publications)
        source_name = str(card.config.signal or "").strip()
        if not source_name or not parents or any(
            not isinstance(parent, SignalPublication)
            or parent.value(source_name) is None
            for parent in parents
        ):
            card.set_status("selector lost its exact source publications", error=True)
            return
        source_generation = parents[-1].event_ref.generation
        if any(parent.event_ref.generation != source_generation for parent in parents):
            card.set_status("selector spans retired source generations", error=True)
            return
        source_entry = self._current_signal_projection().topology.get(source_name)
        if source_entry is None:
            card.set_status("selector source is no longer declared", error=True)
            return
        source_ref = parents[-1].value(source_name).snapshot.ref
        try:
            if output_name == _CROSS_DATA_OUTPUT:
                if result.selected_value is None:
                    raise ValueError("Cross selection has no painted data value")
                snapshot = materialize_value_dataset(
                    source_ref,
                    result.selected_value,
                    reference_for=lambda schema: self._panel_materialized_ref(
                        card, output_name, source_ref, schema
                    ),
                )
                contract_id = _CROSS_CONTRACT_ID
            else:
                snapshot = result.materialize(
                    reference_for=lambda schema: self._panel_materialized_ref(
                        card, output_name, source_ref, schema
                    )
                )
                contract_id = source_entry.declaration.contract_id
        except (TypeError, ValueError, RuntimeError) as error:
            card.set_status(f"Selector output: {error_summary(error)}", error=True)
            return

        owner_id = self._figure_route_owner(card, output_name)
        existing = self._selection_routes.get(owner_id)
        same_route = (
            existing is not None
            and existing.source_name == source_name
            and existing.source_generation == source_generation
            and existing.source_contract_id == contract_id
            and existing.output_schema == snapshot.block.schema
            and existing.selection == result.selection
        )
        if same_route:
            assert existing is not None
            generation = existing.generation
            event_source = existing.event_source
            event_output_name = existing.event_output_name
        else:
            self._data.withdraw_derived(owner_id)
            event_source = None
            event_output_name = None
            if result.selection is not None:
                try:
                    _source_generation, upstream, upstream_output = (
                        self._data.signal_event_binding(source_name)
                    )
                except (KeyError, LookupError, RuntimeError, TypeError, ValueError):
                    pass
                else:
                    event_source = authoritative_signal_event_source(
                        upstream,
                        upstream_output,
                        DataTransformSpec((result.selection,)),
                    )
                    event_output_name = upstream_output
            generation = self._data.bind_continuous_derived(
                owner_id,
                source_name=source_name,
                expected_source_generation=source_generation,
                output_names=(panel_signal_key(card.panel_id, output_name),),
                event_source=event_source,
                event_output_name=event_output_name,
            )
            self._selection_routes[owner_id] = _PanelSelectionRoute(
                owner_id,
                output_name,
                source_name,
                source_generation,
                contract_id,
                snapshot.block.schema,
                result.selection,
                event_source,
                event_output_name,
                generation,
            )
        qualified = panel_signal_key(card.panel_id, output_name)
        try:
            accepted = self._data.publish_continuous_derived(
                owner_id,
                generation,
                parents,
                {qualified: DerivedSignalOutput(snapshot)},
            )
        except (KeyError, LookupError, RuntimeError, TypeError, ValueError) as error:
            card.set_status(f"Selector output: {error_summary(error)}", error=True)
            return
        if accepted:
            self._promote_data_front(self._data.freeze())
            if not same_route:
                self._signal_topology_changed()
            card.set_status("ok", error=False)

    def _accept_card_fit(
        self,
        card: PanelCard,
        event: object,
        publications: object,
    ) -> None:
        """Publish painted Fit parameters; zlc_plot owns compute and overlay."""

        if card not in self.cards or not isinstance(event, FitEvent):
            return
        parents = tuple(publications or ())
        source_name = str(card.config.signal or "").strip()
        if not source_name or not parents or any(
            not isinstance(parent, SignalPublication)
            or parent.value(source_name) is None
            for parent in parents
        ):
            card.set_status("Fit lost its exact source publications", error=True)
            return
        source_ref = parents[-1].value(source_name).snapshot.ref
        try:
            materialized = event.materialize_parameters(
                lambda name, schema: self._panel_materialized_ref(
                    card,
                    f"{_FIT_OUTPUT_PREFIX}{name}",
                    source_ref,
                    schema,
                )
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            card.set_status(f"Fit output: {error_summary(error)}", error=True)
            return
        output_names = tuple(
            f"{_FIT_OUTPUT_PREFIX}{name}" for name in materialized
        )
        snapshots = {
            output_name: materialized[name]
            for output_name, name in zip(
                output_names,
                materialized,
                strict=True,
            )
        }
        owner_id = self._figure_route_owner(card, "fit")
        qualified = {
            panel_signal_key(card.panel_id, name): DerivedSignalOutput(snapshot)
            for name, snapshot in snapshots.items()
        }
        try:
            generation = self._data.bind_event_derived(
                owner_id,
                source_name=source_name,
                source_publications=parents,
                output_names=tuple(qualified),
            )
            published = self._data.publish_event_derived(
                owner_id,
                generation,
                parents,
                qualified,
            )
        except (KeyError, LookupError, RuntimeError, TypeError, ValueError) as error:
            self._data.withdraw_derived(owner_id)
            self._fit_output_names.pop(card.panel_id, None)
            card.set_status(f"Fit output: {error_summary(error)}", error=True)
            self._promote_data_front(self._data.freeze())
            self._signal_topology_changed()
            return
        if published is None:
            return
        changed = self._fit_output_names.get(card.panel_id) != output_names
        self._fit_output_names[card.panel_id] = output_names
        self._promote_data_front(self._data.freeze())
        if changed:
            self._signal_topology_changed()

    def _withdraw_panel_outputs(self, panel_id: str) -> None:
        identity = str(panel_id)
        for owner_id, route in tuple(self._selection_routes.items()):
            if owner_id.startswith(f"figure/{identity}/"):
                self._data.withdraw_derived(owner_id)
                self._selection_routes.pop(owner_id, None)
        self._data.withdraw_derived(f"figure/{identity}/fit")
        self._fit_output_names.pop(identity, None)

    def _card_reads(self, card: "PanelCard") -> set:
        """The panel's one typed dataset binding."""

        return {card.config.signal} if card.config.signal else set()

    def _commit_logic_authored_values(
        self,
        row: "LogicNodeRow",
        *,
        editor: "LogicNodeEditor | None" = None,
        authored: dict | None = None,
        inputs: dict | None = None,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Commit one complete descriptor form without building a runtime."""

        if row not in self.logic_nodes:
            raise ValueError("Logic row is no longer attached to this console")
        explicit = authored is not None or inputs is not None
        if explicit and (authored is None or inputs is None):
            raise ValueError("authored values and inputs must be committed together")
        if editor is not None and explicit:
            raise ValueError("supply either a Logic editor or explicit binding")
        active_editor = self._logic_editors.get(id(row))
        source_editor = active_editor if editor is None else editor
        if editor is not None and active_editor is not editor:
            raise ValueError("Logic editor is not attached to this row")
        if explicit:
            candidate_authored = dict(authored or {})
            candidate_inputs = dict(inputs or {})
        elif source_editor is not None:
            candidate_authored, candidate_inputs = source_editor.collect_binding()
        else:
            candidate_authored = dict(row.node.authored)
            candidate_inputs = dict(row.node.inputs)
        descriptor = self._spec_for_logic(row.node)
        if descriptor is None:
            raise RuntimeError("Logic row lost its descriptor")
        candidate_authored = descriptor.authoring_schema.freeze(candidate_authored)
        expected_inputs = {value.key for value in descriptor.input_specs}
        unknown_inputs = set(candidate_inputs).difference(expected_inputs)
        if unknown_inputs:
            raise ValueError(
                "Logic input refs contain undeclared keys: "
                f"{tuple(sorted(unknown_inputs))}"
            )
        changed = (
            candidate_authored != dict(row.node.authored)
            or candidate_inputs != dict(row.node.inputs)
        )
        if changed:
            previous_outputs = (
                *self._declared_signal_keys(row),
                *self._declared_artifact_keys(row),
            )
            row.node.authored = dict(candidate_authored)
            row.node.inputs = dict(candidate_inputs)
            self._mark_dirty()
            current_outputs = (
                *self._declared_signal_keys(row),
                *self._declared_artifact_keys(row),
            )
            if current_outputs != previous_outputs:
                self._signal_topology_changed()
        if active_editor is not None and (
            explicit or active_editor is not source_editor
        ):
            active_editor.form.seed_binding(candidate_authored, candidate_inputs)
        return candidate_authored, candidate_inputs

    def _node_label(self, node: _ConsoleNode) -> str:
        """Short LAYER name for a node (camera / detect / calibrate / a
        measurement's curve), NOT the Python class name -- the dashboard speaks in
        the architecture's layers, so no class name ever leaks into the UI."""

        for row in self.logic_nodes:
            if (
                self._logic_nodes.get(id(row)) is node
                or self._last_node.get(id(row)) is node
            ):
                return str(row.node.title)
        return str(node.instance_id)

    def _signal_topology(self) -> dict[str, _SignalTopologyEntry]:
        """Project every signal into one exact catalog-owned producer state.

        A valid key is always ``console_signal_key(row.node_id, output.name)``.
        Runtime nodes may prove that key is running or retained, but may never
        rename it.  A data-plane value without such a declaration is not a
        picker candidate; an existing card that still names it is rendered
        explicitly as unbound because it has no topology entry.
        """

        topology: dict[str, _SignalTopologyEntry] = {}
        published_names = set(map(str, self._tick_data.names()))
        title_counts: dict[str, int] = {}
        for row in self.logic_nodes:
            title = str(row.node.title)
            title_counts[title] = title_counts.get(title, 0) + 1

        def add(
            key: str,
            *,
            state: _SignalTopologyState,
            label: str,
            node: _ConsoleNode | PanelCard | None,
            declaration: DatasetOutputSpec | ArtifactOutputSpec,
            kind: str,
        ) -> None:
            if key in topology:
                raise RuntimeError(
                    f"duplicate exact TaskConsole signal declaration {key!r}"
                )
            topology[key] = _SignalTopologyEntry(
                state=state,
                label=label,
                node=node,
                declaration=declaration,
                kind=kind,
            )

        for row in self.logic_nodes:
            descriptor = self._spec_for_logic(row.node)
            if descriptor is None:
                continue
            title = str(row.node.title)
            display_label = (
                title
                if title_counts[title] == 1
                else f"{title} · {row.node.node_id[-8:]}"
            )
            outputs = self._outputs_for_row(row)
            keys = tuple(self._declared_signal_keys(row))
            live = self._logic_nodes.get(id(row))
            retained = self._last_node.get(id(row))
            node = live if live is not None else retained
            if node is not None:
                actual = tuple(node.published_signals())
                if actual != keys:
                    raise RuntimeError(
                        f"{row.node.title!r} runtime outputs differ from its "
                        "catalog-declared exact keys"
                    )
            if live is not None:
                state = "running"
            elif (
                retained is not None
                and retained.final_result_resolved
            ):
                state = "retained-final"
            elif retained is not None and any(key in published_names for key in keys):
                state = "retained-view"
            else:
                state = "declared-not-started"
                node = None
            for key, output in zip(keys, outputs, strict=True):
                add(
                    key,
                    state=state,
                    label=display_label,
                    node=node,
                    declaration=output,
                    kind=descriptor.definition.kind,
                )

        from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration

        cards_by_id = {card.panel_id: card for card in self.cards}
        for route in self._selection_routes.values():
            parts = route.owner_id.split("/")
            panel_id = parts[1] if len(parts) >= 3 else ""
            card = cards_by_id.get(panel_id)
            if card is None:
                continue
            key = panel_signal_key(panel_id, route.output_name)
            if self._data.latest_publication(key) is None:
                continue
            short = "area" if route.output_name == _AREA_DATA_OUTPUT else "value"
            add(
                key,
                state="retained-view",
                label=str(card.config.title or PANEL_KINDS[card.config.kind]),
                node=card,
                declaration=DatasetOutputSpec(
                    DatasetOutputDeclaration(route.output_name, route.source_contract_id),
                    short,
                    short,
                    "Committed plot selection data.",
                ),
                kind="figure",
            )
        for panel_id, output_names in self._fit_output_names.items():
            card = cards_by_id.get(panel_id)
            if card is None:
                continue
            for output_name in output_names:
                key = panel_signal_key(panel_id, output_name)
                if self._data.latest_publication(key) is None:
                    continue
                short = output_name.removeprefix(_FIT_OUTPUT_PREFIX)
                add(
                    key,
                    state="retained-view",
                    label=str(card.config.title or PANEL_KINDS[card.config.kind]),
                    node=card,
                    declaration=DatasetOutputSpec(
                        DatasetOutputDeclaration(output_name, _FIT_CONTRACT_ID),
                        short,
                        short,
                        "Painted plot Fit parameter.",
                    ),
                    kind="figure",
                )

        return topology

    def _artifact_topology(self) -> dict[str, _SignalTopologyEntry]:
        """Project FINAL artifact outputs without pretending they are Datasets."""

        topology: dict[str, _SignalTopologyEntry] = {}
        title_counts: dict[str, int] = {}
        for row in self.logic_nodes:
            title = str(row.node.title)
            title_counts[title] = title_counts.get(title, 0) + 1
        for row in self.logic_nodes:
            descriptor = self._spec_for_logic(row.node)
            if descriptor is None:
                continue
            outputs = self._artifacts_for_row(row)
            if not outputs:
                continue
            keys = tuple(self._declared_artifact_keys(row))
            live = self._logic_nodes.get(id(row))
            retained = self._last_node.get(id(row))
            node = live if live is not None else retained
            if node is not None:
                actual = tuple(node.published_artifacts())
                if actual != keys:
                    raise RuntimeError(
                        f"{row.node.title!r} runtime Artifact outputs differ "
                        "from its owner declaration"
                    )
            if live is not None:
                state = "running"
            elif (
                retained is not None
                and retained.final_result_resolved
            ):
                state = "retained-final"
            else:
                state = "declared-not-started"
                node = None
            title = str(row.node.title)
            display_label = (
                title
                if title_counts[title] == 1
                else f"{title} · {row.node.node_id[-8:]}"
            )
            for key, output in zip(keys, outputs, strict=True):
                if key in topology:
                    raise RuntimeError(
                        f"duplicate exact TaskConsole Artifact declaration {key!r}"
                    )
                topology[key] = _SignalTopologyEntry(
                    state=state,
                    label=display_label,
                    node=node,
                    declaration=output,
                    kind=descriptor.definition.kind,
                )
        return topology

    def _signal_providers(self) -> dict:
        """Exact key -> its one producer group, annotated with lifecycle state."""

        return self._providers_from_topology({
            **self._current_signal_projection().topology,
            **self._artifact_topology(),
        })

    @staticmethod
    def _providers_from_topology(
        topology: Mapping[str, _SignalTopologyEntry],
    ) -> dict[str, list[str]]:
        """Project one already-built topology into picker group labels."""

        state_labels = {
            "running": "running",
            "declared-not-started": "declared · not started",
            "retained-final": "retained FINAL",
            "retained-view": "retained view",
        }
        providers: dict[str, list[str]] = {}
        for key, entry in topology.items():
            state = entry.state
            providers[key] = [
                f"{entry.label}  [{state_labels[state]}]"
            ]
        return providers

    def _signal_names(self) -> list[str]:
        """Only exact catalog declarations are offered as picker candidates."""

        return sorted(self._current_signal_projection().topology)

    def _input_signal_names(
        self,
        input_spec,
        *,
        excluded: frozenset[str] = frozenset(),
    ) -> list[str]:
        """Filter candidates by declared contract and required capability.

        ``excluded`` is applied before capability resolution.  In particular,
        a node's own outputs must never be resolved as prospective producers
        while its editor is still being constructed: doing so asks an
        incomplete consumer draft to prove a producer capability and turns a
        valid graph invariant into an authoring-order dependency.
        """

        from zlc_neutral_atom.input_spec import ArtifactInputSpec

        artifact_input = isinstance(input_spec, ArtifactInputSpec)
        topology = (
            self._artifact_topology()
            if artifact_input
            else self._current_signal_projection().topology
        )
        return sorted(
            key
            for key, entry in topology.items()
            if key not in excluded
            and input_spec.accepts(
                entry.declaration.contract_id
            )
            and (
                not getattr(input_spec, "requires_event_association", False)
                or self._signal_event_association_available(key)
            )
        )

    def _signal_event_association_available(
        self,
        signal_key: str,
    ) -> bool:
        """Return whether one current signal has producer-owned association."""

        key = str(signal_key).strip()
        if not key:
            return False
        return self._data.has_event_association(key)

    def _signal_formats(self) -> dict:
        """Describe current values for every declared signal."""

        return self._signal_formats_for_names(self._signal_names())

    def _signal_formats_for_names(self, names) -> dict[str, str]:
        """Describe current values for one already-chosen signal inventory."""

        out: dict[str, str] = {}
        for name in names:
            value = self._tick_data.value(str(name))
            if value is None:
                continue
            out[str(name)] = describe_dataset_shape(
                value.schema,
                value.values,
            )
        return out

    def _signal_short_names(self) -> dict:
        """Map exact producer/output keys to catalog-owned short labels."""

        projection = self._current_signal_projection()
        return {
            **projection.short_names,
            **self._short_names_from_topology(self._artifact_topology()),
        }

    @staticmethod
    def _short_names_from_topology(topology) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, entry in topology.items():
            declaration = entry.declaration
            out[key] = str(declaration.label or declaration.name)
        return out

    @staticmethod
    def _axis_labels_from_topology(topology) -> dict[str, str]:
        labels: dict[str, str] = {}
        for key, entry in topology.items():
            declaration = entry.declaration
            label = str(declaration.axis_label or "").strip()
            labels[key] = label or str(
                declaration.label or declaration.name
            )
        return labels

    def _panel_signal_groups(self, current: str = ""):
        """Build one Plot Panel picker tree from the event-owned topology."""

        projection = self._current_signal_projection()
        topology = projection.topology
        names = sorted(topology)
        current = str(current or "")
        if current and current not in topology:
            names.append(current)
        return _qt_widgets.signal_tree_groups(
            names,
            self._providers_from_topology(topology),
            self._signal_formats_for_names(names),
            projection.short_names,
        )

    def _panel_render_labels(self):
        """Return labels without scanning topology on a render hot path."""

        projection = self._current_signal_projection()
        return projection.axis_labels, projection.short_names

    @staticmethod
    def _front_schemas(front) -> dict[str, object | None]:
        """Exact present-signal keys and their typed schemas for one front."""

        schemas: dict[str, object | None] = {}
        for name in front.names():
            key = str(name)
            value = front.value(key)
            schemas[key] = None if value is None else value.schema
        return schemas

    def _promote_data_front(self, front) -> set[str]:
        """Promote one immutable front and reconcile only metadata transitions.

        Ordinary revisions with an unchanged schema do no Qt metadata work.
        Presence changes and typed-schema changes update the affected
        Logic legend plus any Setting picker that is currently visible, without
        invoking the topology owner or rebuilding a picker model.
        """

        self._tick_data = front
        if front is self._signal_metadata_front:
            return set()
        previous = self._signal_schemas
        current = self._front_schemas(front)
        self._signal_metadata_front = front
        self._signal_schemas = current
        changed = {
            key
            for key in set(previous).union(current)
            if previous.get(key) != current.get(key)
            or (key in previous) != (key in current)
        }
        if not changed:
            return set()

        for row in self.logic_nodes:
            if changed.intersection(self._declared_signal_keys(row)):
                self._update_row_publishes(row)
                # A transient default panel appears on the first real typed
                # value, not at Start.  This rule is per row and applies to any
                # leaf declaration with a default view.
                self._ensure_task_preview_panels(row)
        for card in self.cards:
            card.refresh_open_signal_metadata()
        return changed

    def _live_node_formats(
        self,
        row: "LogicNodeRow",
        node: _ConsoleNode,
    ) -> list[tuple[str, str, str]]:
        """``[(name, shape, description)]`` for a RUNNING node -- one ROW per output, each
        shape read off a real value via
        ``zlc_frontend.shape_text.describe_dataset_shape``
        (auto, never hand-typed)
        and each description from the node's ``output_specs`` (what the signal MEANS).
        Measurement and processor nodes publish to the data plane under their prefix.
        A declaration has no RUN/FINAL authority; that fact appears only when a
        typed value is admitted by the data plane."""
        declarations = self._outputs_for_row(row)
        if not declarations:
            raise RuntimeError("running node has no frozen output declarations")
        published = tuple(node.published_signals())
        rows: list[tuple[str, str, str]] = []
        for full, declaration in zip(published, declarations, strict=True):
            short = str(declaration.label or declaration.name)
            # Logic rows and signal pickers share this one schema-driven owner.
            # The field is a read-only physical tensor contract, not an axis-summary
            # editor and not catalog-authored display text.
            shape = describe_dataset_shape(
                self._signal_schema(full),
                self._signal_values(full),
            )
            rows.append(
                (
                    short,
                    shape,
                    str(declaration.description or declaration.axis_label),
                )
            )
        return rows

    def _update_row_publishes(self, row: "LogicNodeRow") -> None:
        """Fill a Logic-tab row's "publishes:" legend (ONE signal per line: name, shape,
        meaning).  Shapes are AUTO-EXTRACTED from the real published VALUES
        (``zlc_frontend.shape_text.describe_dataset_shape``) and the meaning
        from the node's
        ``output_specs`` --
        never a hand-typed map.  Running node: live data-plane shapes (measurement /
        processor).  A task does not predict RUN/FINAL from its catalog row; once
        a typed value exists it displays that value through the same schema
        formatter.  A
        stopped data-plane node likewise keeps the exact schema of any retained
        view or FINAL value; only an output that has never published is shown as
        ``—``."""
        descriptor = self._spec_for_logic(row.node)
        if descriptor is None:
            row.set_publishes(())
            return
        if descriptor.definition.kind == "task":
            outputs = self._outputs_for_row(row)
            keys = tuple(self._declared_signal_keys(row))
            published = [
                    (
                        str(output.label or output.name),
                        (
                            "—"
                            if (value := self._tick_data.value(key)) is None
                            else describe_dataset_shape(
                                value.schema,
                                value.values,
                            )
                        ),
                        str(output.description or output.axis_label),
                    )
                    for key, output in zip(keys, outputs, strict=True)
                ]
            retained = self._logic_nodes.get(id(row)) or self._last_node.get(id(row))
            artifact_ready = bool(
                retained is not None
                and retained.final_result_resolved
            )
            published.extend(
                (
                    str(output.label or output.name),
                    "FINAL artifact" if artifact_ready else "artifact",
                    str(output.description),
                )
                for output in self._artifacts_for_row(row)
            )
            row.set_publishes(published)
            return
        node = self._logic_nodes.get(id(row))
        if node is not None and node.running:
            row.set_publishes(self._live_node_formats(row, node))
            return
        outputs = self._outputs_for_row(row)
        keys = tuple(self._declared_signal_keys(row))
        row.set_publishes(
            [
                (
                    str(output.label or output.name),
                    (
                        "—"
                        if (value := self._tick_data.value(key)) is None
                        else describe_dataset_shape(
                            value.schema,
                            value.values,
                        )
                    ),
                    str(output.description or output.axis_label),
                )
                for key, output in zip(keys, outputs, strict=True)
            ]
        )

    def _declared_signal_keys(self, row: "LogicNodeRow") -> list[str]:
        """Exact keys a row will publish or commit, before it is started."""

        return [
            console_signal_key(row.node.node_id, output.name)
            for output in self._outputs_for_row(row)
        ]

    def _declared_artifact_keys(self, row: "LogicNodeRow") -> list[str]:
        """Exact non-Dataset output keys owned by one row."""

        return [
            console_signal_key(row.node.node_id, output.name)
            for output in self._artifacts_for_row(row)
        ]

    def _outputs_for_row(self, row: "LogicNodeRow") -> tuple:
        """Resolve one row's output contract from its exact frozen request.

        A live/retained node owns the declarations frozen with that run.  A row
        that has never run first freezes any request-dependent vocabulary through
        the same ``build_request`` boundary.  No picker or legend may derive
        Camera cardinality from raw form values or Dataset shape.
        """

        descriptor = self._spec_for_logic(row.node)
        if descriptor is None:
            return ()
        runtime = self._logic_nodes.get(id(row)) or self._last_node.get(id(row))
        if runtime is not None and runtime.definition_key == descriptor.definition.key:
            outputs = tuple(descriptor.outputs_for(runtime.request))
            declarations = tuple(output.declaration for output in outputs)
            if declarations != runtime.dataset_output_declarations:
                raise RuntimeError(
                    "producer runtime Dataset declarations differ from its "
                    "frozen request"
                )
            return outputs
        if descriptor.resolve_outputs is None:
            return tuple(descriptor.outputs)
        try:
            request = descriptor.build_request(
                descriptor.authoring_schema.freeze(row.node.authored)
            )
        except (TypeError, ValueError):
            return ()
        return tuple(descriptor.outputs_for(request))

    def _artifacts_for_row(self, row: "LogicNodeRow") -> tuple:
        """Return the row's static typed Artifact vocabulary."""

        descriptor = self._spec_for_logic(row.node)
        if descriptor is None:
            return ()
        artifacts = tuple(descriptor.artifact_outputs)
        runtime = self._logic_nodes.get(id(row)) or self._last_node.get(id(row))
        if runtime is not None and runtime.definition_key == descriptor.definition.key:
            declarations = tuple(output.declaration for output in artifacts)
            if declarations != runtime.artifact_output_declarations:
                raise RuntimeError(
                    "producer runtime Artifact declarations differ from its "
                    "frozen request"
                )
        return artifacts

    def _artifact_value(self, output_key: str, spec: ArtifactInputSpec) -> object:
        """Resolve one exact Task output to its already-typed FINAL ArtifactRef."""

        matches = []
        for row in self.logic_nodes:
            descriptor = self._spec_for_logic(row.node)
            if descriptor is None:
                continue
            for key, output in zip(
                self._declared_artifact_keys(row),
                self._artifacts_for_row(row),
                strict=True,
            ):
                if key == output_key:
                    matches.append((row, descriptor, output))
        if len(matches) != 1:
            raise LookupError(
                f"Artifact input {output_key!r} must name one TaskConsole output"
            )
        row, descriptor, output = matches[0]
        if not spec.accepts(output.contract_id):
            raise ValueError(
                f"{spec.label} rejects Artifact contract {output.contract_id!r}"
            )
        host = self._logic_nodes.get(id(row)) or self._last_node.get(id(row))
        if host is None or host.running or not host.final_result_resolved:
            raise RuntimeError(
                f"{row.node.title!r} has no completed FINAL Artifact"
            )
        if (
            host.definition_key != descriptor.definition.key
            or host.instance_id != row.node.node_id
        ):
            raise RuntimeError("retained Artifact host differs from its saved row")
        result = host.final_result
        if isinstance(result, Mapping):
            try:
                return result[output.name]
            except KeyError as error:
                raise RuntimeError("FINAL Artifact result omitted its output") from error
        if len(descriptor.artifact_outputs) != 1:
            raise RuntimeError(
                "a multi-Artifact descriptor must return a name-keyed mapping"
            )
        return result

    def resolve_node_inputs(
        self,
        descriptor: LogicNodeDescriptor,
        values: Mapping[str, object],
        *,
        own_keys: frozenset[str] = frozenset(),
    ) -> dict[str, object]:
        """Resolve saved refs directly to the values accepted by host_factory."""

        if not isinstance(descriptor, LogicNodeDescriptor):
            raise TypeError("descriptor must be LogicNodeDescriptor")
        supplied = dict(values)
        declared = {value.key: value for value in descriptor.input_specs}
        unknown = set(supplied).difference(declared)
        if unknown:
            raise ValueError(
                f"Logic inputs contain undeclared keys: {tuple(sorted(unknown))}"
            )
        resolved: dict[str, object] = {}
        for key, input_spec in declared.items():
            selected = supplied.get(key)
            if isinstance(input_spec, DatasetInputSpec):
                if not isinstance(selected, str) or not selected.strip():
                    raise ValueError(f"select {input_spec.label}")
                signal_key = selected.strip()
                if signal_key in own_keys:
                    raise ValueError("a Logic node cannot consume its own output")
                entry = self._current_signal_projection().topology.get(signal_key)
                if entry is None or not input_spec.accepts(entry.declaration.contract_id):
                    raise ValueError(
                        f"{input_spec.label} does not accept {signal_key!r}"
                    )
                if (
                    input_spec.requires_event_association
                    and not self._signal_event_association_available(signal_key)
                ):
                    raise ValueError(
                        f"{input_spec.label} requires producer-associated events"
                    )
                resolved[key] = signal_key
                continue
            if not isinstance(input_spec, ArtifactInputSpec):
                raise TypeError("descriptor contains another input kind")
            if selected is None and input_spec.allow_saved_reference:
                continue
            if isinstance(selected, str) and selected.startswith("@logic/"):
                resolved[key] = self._artifact_value(selected, input_spec)
                continue
            if not input_spec.allow_saved_reference:
                raise ValueError(f"select {input_spec.label}")
            if not isinstance(selected, (str, Path)):
                raise TypeError(f"{input_spec.label} path must be text")
            resolved[key] = resolve_under(self._project_root, selected)
        return resolved

    @staticmethod
    def _card_topology_identity(card: "PanelCard") -> tuple[str, str, str]:
        """Card facts consumed by signal and Figure-output topology."""

        kind = card.config.kind
        return (
            str(card.config.signal or ""),
            kind.value,
            str(card.config.title or PANEL_KINDS[kind]),
        )

    def _card_topology_identity_changed(self, card: "PanelCard") -> None:
        """Consume one card-local structural delta, not a board snapshot."""

        if card not in self.cards:
            return
        current = self._card_topology_identity(card)
        previous = self._card_topology_identities.get(id(card))
        if previous == current:
            return
        if previous is not None and previous[:2] != current[:2]:
            self._withdraw_panel_outputs(card.panel_id)
            self._promote_data_front(self._data.freeze())
        self._card_topology_identities[id(card)] = current
        self._signal_topology_changed()

    def _signal_topology_changed(self) -> None:
        """Reconcile legends after an explicit provider/card topology mutation."""

        self._data.set_front_signals(
            frozenset(
                str(card.config.signal).strip()
                for card in self.cards
                if str(card.config.signal or "").strip()
            )
        )
        self._signal_info_dirty = True
        if not self._building:
            self._refresh_signal_info()
            for card in self.cards:
                card.refresh_open_signal_topology()
            for editor in self._logic_editors.values():
                if editor.isVisible():
                    editor.refresh_on_show()

    def _current_signal_projection(self) -> _SignalTopologyProjection:
        """Return the sole projection, reconciling one explicit dirty edge."""

        if self._signal_info_dirty or self._signal_projection is None:
            self._refresh_signal_info()
        projection = self._signal_projection
        if projection is None:
            raise RuntimeError("signal topology projection was not initialized")
        return projection

    def _refresh_signal_info(self) -> None:
        """Give every panel a frame-title legend naming which node and layer
        produces the panel's one bound signal -- e.g.
        ``signal ← producer [processor]``.  It does not list the producing node's whole
        output set, so the title answers exactly "this plot's value comes from which
        measurement/processor".  A signal
        published by more than one running node is flagged ambiguous.  The
        caller is an explicit topology/card-binding mutation; a timer tick with
        no such mutation does not enter this method's provider enumeration."""
        if not self._signal_info_dirty:
            return
        topology = self._signal_topology()
        short_names = self._short_names_from_topology(topology)
        axis_labels = self._axis_labels_from_topology(topology)
        self._refresh_passive_publisher_rows(topology)
        for card in self.cards:
            reads = sorted(self._card_reads(card))
            parts: list[str] = []
            for name in reads:
                entry = topology.get(name)
                if entry is None:
                    parts.append(
                        f"{name} ← truly unbound "
                        "(no catalog producer/output)"
                    )
                else:
                    state = entry.state
                    state_text = {
                        "running": "running",
                        "declared-not-started": "declared, not started",
                        "retained-final": "retained FINAL",
                        "retained-view": "retained view",
                    }[state]
                    label = entry.label
                    layer = entry.kind
                    tag = f"{label} [{layer}]" if layer else label
                    short = short_names.get(name, name)
                    parts.append(
                        f"{short} ← {label} [Figure]"
                        if layer == "figure"
                        else f"{short} ← {tag} · {state_text}"
                    )
            if not reads:
                parts.append("(no signal set — pick one in Setting: value = <signal>)")
            # one read per line: "<signal> ← <node> [layer]" -- the value's origin only,
            # not the producing node's full output list.
            card.set_signal_info("\n".join(parts))
        frozen_topology = MappingProxyType(dict(topology))
        self._signal_projection = _SignalTopologyProjection(
            frozen_topology,
            MappingProxyType(dict(axis_labels)),
            MappingProxyType(dict(short_names)),
        )
        self._signal_info_dirty = False

    def _refresh_passive_publisher_rows(
        self,
        topology: Mapping[str, _SignalTopologyEntry],
    ) -> None:
        """Reconcile non-LogicNode publishers into the ordinary Logic list."""

        if self.logic_layout is None:
            return
        projected: dict[str, tuple[str, str, str, list[tuple[str, str, str]]]] = {}
        for card in self.cards:
            rows: list[tuple[str, str, str]] = []
            for key, entry in topology.items():
                if entry.kind != "figure" or entry.node is not card:
                    continue
                declaration = entry.declaration
                value = self._tick_data.value(key)
                shape = (
                    "—"
                    if value is None
                    else describe_dataset_shape(value.schema, value.values)
                )
                description = str(declaration.description).strip()
                detail = str(key) if not description else f"{description} · {key}"
                rows.append((str(declaration.label), shape, detail))
            if rows:
                projected[card.panel_id] = (
                    str(card.config.title or PANEL_KINDS[card.config.kind]),
                    "figure",
                    "retained view",
                    sorted(rows),
                )

        for publisher_id in tuple(self._passive_publisher_rows):
            if publisher_id in projected:
                continue
            row = self._passive_publisher_rows.pop(publisher_id)
            self.logic_layout.removeWidget(row)
            row.hide()
            row.deleteLater()

        for publisher_id, (title, kind, state, rows) in projected.items():
            row = self._passive_publisher_rows.get(publisher_id)
            if row is None:
                row = PublishedSignalRow(title=title, kind=kind, state=state)
                self._passive_publisher_rows[publisher_id] = row
                self.logic_layout.insertWidget(self.logic_layout.count() - 1, row)
            else:
                row.set_identity(title=title, kind=kind, state=state)
            row.set_publishes(rows)
        self._refresh_logic_hint_visibility()

    def _refresh_logic_hint_visibility(self) -> None:
        if self.logic_hint is None:
            return
        self.logic_hint.setVisible(
            not self.logic_nodes and not self._passive_publisher_rows
        )

    def _start_node_owner(self, node: _ConsoleNode) -> None:
        """Start the one common host; it owns reservation and execution policy."""

        node.start()

    def _stop_node_owner(self, node: _ConsoleNode) -> bool:
        """Ask a hosted node to stop and report whether its owner is terminal.

        Cancellation is a request, not a join: the console never blocks the GUI
        thread waiting for hardware to let go.  An unterminated owner keeps its
        registration so the next tick (or the close path) can finish the teardown
        instead of losing track of a run that is still holding a device.
        """

        node.cancel()
        # The Qt owner performs exactly one observation.  Future timer turns
        # continue the same state transition; no GUI callback sleeps or joins a
        # hardware/worker owner.
        observation = node.poll()
        if not observation.terminal:
            return False
        if not node.worker_idle:
            return False
        node.shutdown()
        return True

    def _edit_card(self, card: "PanelCard") -> None:
        """Open (or focus) this PLOT panel's OWN closable Edit tab.

        Opens even BEFORE the panel has data -- the panel's plot params /
        acquisition / fit / limits are editable straight away.  The snapshot
        section just shows "waiting for data" until a plot exists (the data is
        produced by a Logic-tab node, not by this Edit)."""
        if card is None:
            return
        existing = self._panel_editors.get(id(card))
        if existing is not None:
            self.tabs.setCurrentWidget(existing)
            return
        editor = PanelEditor(card, self, self.tabs)
        editor.shutdownFinished.connect(self.request_owner_wake)
        self._panel_editors[id(card)] = editor
        title = (card.config.title or PANEL_KINDS[card.config.kind]).strip() or "panel"
        # Freeze the accepted front while the editor is already parented but
        # still hidden in the tab stack.  Adding/focusing the tab then reveals
        # one complete page; no late child show and no transient top-level.
        editor.refresh_snapshot()
        self.tabs.add_closable_tab(editor, title)

    def _close_panel_editor(self, card: "PanelCard") -> None:
        """Close a card's Edit tab if open (called when the card is removed)."""
        editor = self._panel_editors.pop(id(card), None)
        if editor is None:
            return
        index = self.tabs.indexOf(editor)
        if index >= 0:
            self.tabs.removeTab(index)
        # ``removeTab`` leaves the page owned by the tab stack until deferred
        # deletion.  Keep that parent: reparenting a QWidget to ``None`` turns
        # it into a native top-level between this callback and DeferredDelete.
        editor.hide()
        self._retiring_panel_editors.add(editor)
        editor.shutdownFinished.connect(
            lambda current=editor: self._finish_panel_editor_retirement(current)
        )
        if editor.teardown():
            self._finish_panel_editor_retirement(editor)

    def _finish_panel_editor_retirement(self, editor: "PanelEditor") -> None:
        if editor not in self._retiring_panel_editors:
            return
        self._retiring_panel_editors.discard(editor)
        editor.hide()
        editor.deleteLater()
        self.request_owner_wake()

    def _on_editor_tab_closed(self, widget) -> None:
        """X on a PanelEditor / LogicNodeEditor tab: tear it down + drop it from the
        registry.  The permanent Monitor / Logic tabs carry no X, so they never
        arrive here.  A logic node's Edit only closes the TAB -- the node keeps
        running and stays in the Logic list (reopen its Edit from its row)."""
        logic_entry = next(
            (
                (key, editor)
                for key, editor in self._logic_editors.items()
                if editor is widget
            ),
            None,
        )
        if logic_entry is not None:
            key, editor = logic_entry
            row = next((item for item in self.logic_nodes if id(item) == key), None)
            if row is None:
                raise RuntimeError("Logic editor lost its authored row")
            try:
                self._commit_logic_authored_values(row, editor=editor)
            except Exception as error:
                message = f"cannot close: invalid form values: {error}"
                editor.set_status(message, error=True)
                self.status_strip.show_message(message, severity="error")
                self.tabs.setCurrentWidget(editor)
                return

        index = self.tabs.indexOf(widget)
        if index >= 0:
            self.tabs.removeTab(index)
        for key, editor in list(self._panel_editors.items()):
            if editor is widget:
                del self._panel_editors[key]
                self._retiring_panel_editors.add(editor)
                editor.shutdownFinished.connect(
                    lambda current=editor: self._finish_panel_editor_retirement(
                        current
                    )
                )
        for key, editor in list(self._logic_editors.items()):
            if editor is widget:
                del self._logic_editors[key]
        closed = True
        if hasattr(widget, "teardown"):
            result = widget.teardown()
            closed = result is not False
        # A retired Edit page has no legal top-level state.  Hiding it while the
        # tab stack remains its owner makes teardown atomic from the operator's
        # point of view and prevents the historical one-frame popup.
        widget.hide()
        if closed:
            if widget in self._retiring_panel_editors:
                self._finish_panel_editor_retirement(widget)
            else:
                widget.deleteLater()

    def _on_tab_changed(self, _index: int) -> None:
        # ONE rule: when a tab becomes visible, give whatever's inside a chance to refresh its
        # dynamic content -- any widget that implements ``refresh_on_show()`` runs it.  No
        # special-casing per tab type (PanelEditor / LogicNodeEditor / parameter forms all
        # honour the same hook), so a brand new editor automatically participates by
        # implementing ``refresh_on_show`` without adding a type branch here.
        widget = self.tabs.currentWidget()
        hook = getattr(widget, "refresh_on_show", None)
        if callable(hook):
            try:
                hook()
            except Exception as exc:
                self.status_strip.show_message(
                    f"Tab refresh failed: {type(exc).__name__}: {exc}",
                    severity="error",
                )

    def _arrange_if_cards(self) -> None:
        """Re-pack the board (deferred one-shot on show) ONLY when it holds cards -- a no-op on an
        empty board, so an embedded console with nothing loaded yet does not thrash."""
        if getattr(self, "cards", None):
            self._arrange()

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        # Re-pack the gravity board when its scroll viewport WIDTH changes (window resized): the pack
        # width is demand-driven (``_pack_width`` = the viewport width grown to the furthest card's
        # reach), so WIDENING lets more columns fit and NARROWING re-flows cards that now fit while any
        # card the user parked past the edge keeps the board wide + a horizontal scrollbar.  Guard on an
        # actual width change (no redundant packs).
        if (getattr(self, "scroll", None) is not None and obj is self.scroll.viewport()
                and event.type() == QtCore.QEvent.Resize):
            w = self.scroll.viewport().width()
            if w != getattr(self, "_board_view_w", 0) and getattr(self, "cards", None):
                self._board_view_w = w
                self._arrange()
        return super().eventFilter(obj, event)

    def _snap_dropped_card(self, card: PanelCard) -> None:
        """Reorder a JUST-DROPPED card to the ORDER position nearest its raw drop point
        (:func:`drop_index`): dropping ONTO a card's slot DISPLACES it (insert before it, so it and
        everything after shift down and re-pack), dropping past the last card appends to the bottom.
        Connected to ``PanelCard.dropped`` -- the drag-release path ONLY; the ``layout_changed`` the
        card emits right after re-packs the new order (:func:`pack` recomputes every pixel), so resize
        / Add never reorder.  The ORDER (``self.cards``) is the layout's single source of truth."""
        if card not in self.cards:
            return
        others = [c.config for c in self.cards if c is not card]
        idx = drop_index(card.config, others, self._pack_width([c.config for c in self.cards]))
        self.cards.remove(card)
        self.cards.insert(idx, card)

    def _arrange(self) -> None:
        # ONE order-driven north-west pack (:func:`pack`): each card, IN ``self.cards`` ORDER, floats
        # to the first free top-left slot.  A drag has already reordered ``self.cards`` (via
        # _snap_dropped_card) so the drop is honoured, while gravity stays strictly top-left and
        # deterministic -- a resize/click re-packs to the SAME arrangement (no surprise reflow) and an
        # Add (appended last) lands in the next bottom slot, never a middle hole.
        board_w = self._pack_width([c.config for c in self.cards])
        if pack([c.config for c in self.cards], board_w):
            self._mark_dirty()
        # Repack as one atomic frame: board.arrange moves EVERY card (card.move) + resizes the board,
        # so without a freeze each card's move paints on its own and a multi-card reflow reads as a
        # cascade of little jumps.  The SAME updates-disabled primitive the resized card uses in
        # _on_size, applied one level up so the whole reflow composites in a single flush.
        self.board.setUpdatesEnabled(False)
        try:
            self.board.arrange(self.cards)
        finally:
            self.board.setUpdatesEnabled(True)
        self._update_summary()

    def _pack_width(self, configs: Sequence["PanelConfig"]) -> int:
        """The width :func:`pack` wraps at = the live scroll-viewport width (pack clamps it up to
        one-card-wide for a card wider than the viewport).  Strict NW gravity means the board only ever
        grows DOWN (vertical scroll) -- it never parks a card in an off-screen column, so there is no
        demand-grown horizontal extent to track any more (that was the parking the user asked us to
        stop.  No viewport yet (pre-show) -> the headless two-wide fallback."""
        vw = self.scroll.viewport().width() if hasattr(self, "scroll") else 0
        return vw if vw else board_width(configs)

    # ------------------------------------------------------------------ actions
    def _add_panel(self) -> None:
        """Header "Add Panel": add either a BLANK plot panel (a plot kind) or a
        STOPPED logic node (measurement / processor / task)."""
        data = self.kind_combo.currentData()
        # A node LAYER -> a STOPPED logic node on the Logic tab.  It publishes
        # nothing until Started.
        if isinstance(data, Mapping):
            definition_key = definition_key_from_tree(data)
            descriptor = self._descriptor_by_key.get(definition_key)
            if descriptor is None:
                raise RuntimeError("Add Panel selection differs from discovery")
            self._add_logic_node(
                LogicNodeConfig(
                    definition_key=definition_key_to_tree(definition_key),
                    title=descriptor.definition.title,
                    authored=self._default_authored(descriptor),
                ),
                focus=True,
            )
            return
        # Otherwise a PLOT kind -> a BLANK pure-view panel on the Monitor board
        # (decoupled: it shows nothing until a signal is picked in its Setting).
        kind = data or PlotKind.CURVE
        if not isinstance(kind, PlotKind):
            raise TypeError("Add Panel plot selection must be PlotKind")
        # Every new panel gets a unique "<kind> #N" title so two defaults never share
        # one name in the card header / Edit tab / frame title.
        title = indexed_unique_name(PANEL_KINDS[kind], {c.config.title for c in self.cards})
        # PanelConfig owns the ordinary stock 2x2 default.  Do not restate a
        # shell-specific size here; Pulse and a resolved Grid are the only
        # plot families with topology-derived initial-size policies.
        config = PanelConfig(plot=kind, title=title, row=GAP, col=GAP)
        # APPEND the new card LAST in order (``_attach_card`` adds it to the end of ``self.cards``);
        # the order-driven :func:`pack` in ``_arrange`` then lands it in the next free BOTTOM slot,
        # never a middle hole.  No pixel seed needed -- pack recomputes every card's position from
        # the order alone.
        card = self._new_panel_card(config)
        self._attach_card(card)
        self._arrange()
        # Adding a plot is one visible navigation action: reveal the Monitor
        # surface and the newly appended card before the operator can open its
        # Setting popup.  Otherwise a wrapped board can leave the new card below
        # the viewport while programmatic focus and popup anchors point at an
        # object the operator cannot actually see.
        self.tabs.setCurrentWidget(self.tabs.widget(0))
        self.scroll.ensureWidgetVisible(card, GAP, GAP)
        self._mark_dirty()

    def _remove_panel(
        self,
        card: PanelCard,
        *,
        _state_change: bool = True,
    ) -> bool:
        phases = self._panel_teardown_phases.setdefault(id(card), set())
        if "detached" in phases:
            return True
        if card not in self.cards and "removed" not in phases:
            return False
        self._forget_card_render(card)
        if "editor" not in phases:
            card.settings_popup.hide()
            self._close_panel_editor(card)     # drop this card's Edit tab too
            phases.add("editor")
        # Shutdown is the acknowledgement that the card no longer owns render/selector resources.
        # It must succeed before the card disappears from the registry; otherwise a retry would see
        # "not in cards" and falsely report completion while teardown never happened.
        if "shutdown" not in phases:
            if not card.shutdown():
                QtCore.QTimer.singleShot(
                    25,
                    lambda current=card, state_change=_state_change: (
                        self._remove_panel(
                            current,
                            _state_change=state_change,
                        )
                    ),
                )
                return False
            phases.add("shutdown")
        if "removed" not in phases:
            self.cards.remove(card)
            self._card_topology_identities.pop(id(card), None)
            self._withdraw_panel_outputs(card.panel_id)
            self._transient_panel_ids.discard(card.panel_id)
            # Panel removal retires its neutral routes before any picker or
            # topology projection can observe the structural change.
            self._promote_data_front(self._data.freeze())
            self._signal_topology_changed()
            phases.add("removed")
        if "qt_detached" not in phases:
            # PanelBoard remains the QObject owner through DeferredDelete.
            # ``setParent(None)`` would briefly promote the removed card to a
            # top-level QWidget on Windows.
            card.hide()
            card.deleteLater()
            phases.add("qt_detached")
        if "arranged" not in phases:
            self._arrange()
            self._recompute_tick_interval()    # removing the fastest panel can slow the base
            if _state_change:
                self._mark_dirty()
            phases.add("arranged")
        phases.add("detached")
        self._panel_teardown_phases.pop(id(card), None)
        return True

    # ====================================================================== logic nodes
    def _catalog_specs(self, kind: str) -> tuple:
        """Discovered descriptors of one owner-declared kind."""

        return tuple(
            value
            for value in self._descriptors
            if value.definition.kind == kind
        )

    def _spec_for_logic(self, node: LogicNodeConfig):
        """The catalog entry a logic node builds from.

        It carries the node's parameter form (the Edit auto-form renders it), its
        declared outputs and the typed-request builder the RUN seam starts from.
        One lookup for every layer uses the owner-encoded stable DefinitionKey;
        the row title remains presentation-only.
        """

        try:
            key = definition_key_from_tree(node.definition_key)
        except (TypeError, ValueError):
            return None
        return self._descriptor_by_key.get(key)

    def _default_authored(
        self,
        descriptor: LogicNodeDescriptor,
    ) -> dict[str, object]:
        """Seed owner defaults plus an unambiguous installed device choice."""

        requirements = dict(descriptor.device_requirements)
        values: dict[str, object] = {}
        for field in descriptor.authoring_schema.fields:
            if field.dynamic_choices:
                capabilities = requirements[field.key]
                devices = tuple(
                    info
                    for info in self._device_catalog.values()
                    if all(value in info.capabilities for value in capabilities)
                )
                if len(devices) == 1:
                    values[field.key] = devices[0].instance_id
                continue
            if field.default is not None or not field.required:
                values[field.key] = field.default
        return values

    def _unique_logic_title(self, title: str) -> str:
        """Choose a readable default label for a newly added Logic row."""

        return indexed_unique_name(title, {str(r.node.title) for r in self.logic_nodes})

    def _attach_logic_node(self, node: LogicNodeConfig, *, focus: bool = False) -> "LogicNodeRow | None":
        """Add a STOPPED logic-node row to the Logic tab (no node built yet).  A NO-OP when embedded:
        the figure viewer has no Logic tab (``logic_layout`` is None), so there is nowhere to host the
        row -- the console is a pure view host, its acquisition is whatever produced the loaded figure."""
        if self.logic_layout is None:                        # embedded: no Logic tab to attach to
            return None
        if self._spec_for_logic(node) is None:
            raise ValueError(
                "TaskConsole layout references an unknown or wrong-kind DefinitionKey"
            )
        if any(row.node.node_id == node.node_id for row in self.logic_nodes):
            raise ValueError(
                f"duplicate TaskConsole logic node_id {node.node_id!r}"
            )
        descriptor = self._spec_for_logic(node)
        assert descriptor is not None
        row = LogicNodeRow(node, descriptor.definition.kind)
        row.edit_requested.connect(self._edit_logic_node)
        row.remove_requested.connect(self._remove_logic_node)
        row.start_requested.connect(self._start_logic_node)
        row.stop_requested.connect(self._stop_logic_node)
        # insert ABOVE the trailing stretch (the hint + stretch are the last 2 items)
        self.logic_layout.insertWidget(self.logic_layout.count() - 1, row)
        self.logic_nodes.append(row)
        self._logic_nodes[id(row)] = None
        self._signal_topology_changed()
        self._refresh_logic_hint_visibility()
        self._update_row_publishes(row)                       # show its outputs + shapes up front
        if focus and hasattr(self.tabs, "setCurrentWidget"):
            self._edit_logic_node(row)
        return row

    def _add_logic_node(self, node: LogicNodeConfig, *, focus: bool = True) -> "LogicNodeRow":
        node.title = self._unique_logic_title(node.title)
        row = self._attach_logic_node(node, focus=focus)
        self._mark_dirty()
        return row

    def _edit_logic_node(self, row: "LogicNodeRow") -> None:
        """Open (or focus) a logic node's OWN closable Edit tab (param form + Start/Stop)."""
        existing = self._logic_editors.get(id(row))
        if existing is not None:
            self.tabs.setCurrentWidget(existing)
            return
        descriptor = self._spec_for_logic(row.node)
        if descriptor is None:
            raise RuntimeError("Logic row lost its descriptor")
        editor = LogicNodeEditor(
            title=row.node.title,
            descriptor=descriptor,
            authored=row.node.authored,
            inputs=row.node.inputs,
            device_catalog=self._device_catalog,
            project_root=self._project_root,
            pulses_root=self._pulses_root,
            runtime=self.form_runtime_for_logic(row),
            parent=self.tabs,
        )
        editor.start_requested.connect(
            lambda current=row: self._start_logic_node(current)
        )
        editor.stop_requested.connect(
            lambda current=row: self._stop_logic_node(current)
        )
        # A draft makes the saved layout stale, but stays widget-local until a
        # semantic commit boundary; typing never rebuilds a node or snapshots data.
        editor.draft_changed.connect(self._mark_dirty)
        # reflect the live run state on the form (a Started node reopened keeps Stop enabled)
        node = self._logic_nodes.get(id(row))
        editor.set_running(node is not None and node.running)
        self._logic_editors[id(row)] = editor
        self.tabs.add_closable_tab(editor, row.node.title)

    def form_runtime_for_logic(
        self,
        row: "LogicNodeRow",
    ) -> FormRuntimeContext:
        """Return the explicit dynamic-form capabilities for one Logic row."""

        descriptor = self._spec_for_logic(row.node)
        if descriptor is None:
            raise RuntimeError("Logic row lost its descriptor")
        field_input_specs = {
            field_key: spec
            for spec in descriptor.input_specs
            for field_key in spec.field_keys
            if (
                isinstance(spec, DatasetInputSpec)
                or field_key == spec.producer_key
            )
        }
        # No LogicNode may bind one of its own declared outputs as an input.
        # This is a graph invariant, not a Processor-specific UI policy:
        # Measurements such as PulseScan also publish outputs, and admitting
        # those keys here creates an impossible self-cycle before the node has
        # produced anything.
        own = {
            str(key)
            for key in (
                *self._declared_signal_keys(row),
                *self._declared_artifact_keys(row),
            )
        }

        def names(key: str):
            input_spec = field_input_specs.get(str(key))
            if input_spec is None:
                values = self._signal_names()
            else:
                values = self._input_signal_names(input_spec, excluded=frozenset(own))
            return tuple(str(value) for value in values if str(value) not in own)

        return FormRuntimeContext(
            signal_names=names,
            signal_sources=self._signal_providers,
            signal_formats=self._signal_formats,
            signal_labels=self._signal_short_names,
        )

    def _start_logic_node(self, row: "LogicNodeRow") -> None:
        """Build the node FROM the node's current param-form values with display
        suppressed (it publishes to the data plane and never opens another plot),
        install it as this row's sole active runtime, and start it (data-paced).
        Sets the node's status dot green; on build/run error -> red + the error
        on the status line.  Reuses the SAME node-build paths the real readout /
        notebook use."""
        editor = self._logic_editors.get(id(row))
        try:
            authored, inputs = self._commit_logic_authored_values(row, editor=editor)
        except Exception as error:
            current = self._logic_nodes.get(id(row))
            still_running = bool(
                current is not None
                and current.running
            )
            action = "restart" if still_running else "start"
            status = f"{action} rejected: invalid form values: {error}"
            row.set_state(
                "running" if still_running else "error",
                status=status,
            )
            if editor is not None:
                editor.set_running(still_running)
                editor.set_status(status, error=True)
            return
        # Build the replacement before stopping the current run.  Invalid form
        # values therefore cannot destroy a valid running node.
        try:
            node = self._build_logic_node(row, authored, inputs)
        except Exception as exc:
            row.set_state("error", status=f"build failed: {exc}")
            if editor is not None:
                editor.set_status(f"build failed: {exc}", error=True)
            return
        previous = (
            self._logic_nodes.get(id(row))
            or self._last_node.get(id(row))
        )
        # A row never stacks two generations of itself.  Conflicts with another
        # row are not guessed or retried here: the Experiment application is the
        # sole admission and retirement authority.
        if not self._stop_logic_node(row, _silent=True):
            row.set_state("running", status="restart blocked: previous owner still active")
            if editor is not None:
                editor.set_running(True)
                editor.set_status(
                    "restart blocked: previous owner thread did not terminate", error=True
                )
            return
        if previous is not None:
            # A replacement generation cannot begin while the previous
            # generation still owns its data-plane route.  Retire the complete
            # value/presentation front first; the new node reuses the same
            # catalog namespace but publishes a new immutable generation.
            self._retire_logic_node_publications(previous)
        try:
            self._start_node_owner(node)
        except Exception as exc:
            self._signal_topology_changed()
            row.set_state("error", status=f"start failed: {exc}")
            if editor is not None:
                editor.set_running(False)
                editor.set_status(f"start failed: {exc}", error=True)
            return
        # Commit the replacement to the row only after start submission succeeds.
        self._logic_nodes[id(row)] = node
        self._last_node[id(row)] = node           # survives Stop, for signal-source labelling
        self._sync_terminal_poll_timer()
        self._signal_topology_changed()
        # Submission is not hardware admission.  Keep the visible lifecycle at
        # ``starting`` until the hosted owner exposes its admitted lifecycle; this
        # prevents a dependent Task from treating an optimistic GUI label as a
        # physical owner that is already safe to preempt.
        row.set_state("running", status="starting")
        self._update_row_publishes(row)            # now show the LIVE node's published shapes
        if editor is not None:
            editor.set_running(True)
            editor.set_status("starting", error=False)
        self.status_dot.set_color(GREEN)
        self._mark_dirty()
        self._ensure_task_preview_panels(row)

    def _ensure_task_preview_panels(self, row: "LogicNodeRow") -> None:
        """Open a Task's explicitly declared previews exactly once.

        The catalog chooses only an output and an existing plot kind.  This
        creates no task-specific viewer and does not publish data.  A
        transient panel is admitted only after its first typed live value exists;
        a retained panel may wait for the task's actual FINAL output.
        """

        spec = self._spec_for_logic(row.node)
        previews = () if spec is None else tuple(spec.task_previews)
        if not previews:
            return
        declarations = {item.name: item for item in self._outputs_for_row(row)}
        for preview in previews:
            output_name = preview.output_name
            key = console_signal_key(row.node.node_id, output_name)
            if any(card.config.signal == key for card in self.cards):
                continue
            declaration = declarations.get(output_name)
            if declaration is None:
                raise RuntimeError(
                    f"default panel output {output_name!r} is not declared"
                )
            value = self._tick_data.value(key)
            # A default panel represents a real typed result, never a promise
            # made at Start.  Transient and FINAL lifecycle comes only from the
            # admitted value, not from catalog stage metadata.
            if value is None:
                continue
            card = self._new_panel_card(
                PanelConfig(
                    plot=preview.plot,
                    title=str(declaration.label or declaration.name),
                    row=GAP,
                    col=GAP,
                    signal=key,
                ),
            )
            self._attach_card(card)
            if value.transient:
                self._transient_panel_ids.add(card.panel_id)
            self._arrange()
            if not value.transient:
                self._mark_dirty()

    def _remove_transient_result_panels(self, row: "LogicNodeRow") -> None:
        """Remove only transient default panels sourced by ``row``.

        The panel id set records presentation lifetime, while the frozen signal
        declarations identify ownership.  No board-global task/session state is
        required.
        """

        owned_signals = self._declared_signal_keys(row)
        panels = tuple(
            card
            for card in self.cards
            if card.panel_id in self._transient_panel_ids
            and card.config.signal in owned_signals
        )
        for panel in panels:
            self._remove_panel(panel, _state_change=False)

    def _build_logic_node(
        self,
        row: LogicNodeRow,
        authored: Mapping[str, object],
        inputs: Mapping[str, object],
    ) -> LogicNodeHost:
        """Freeze one row through the application's sole host factory."""

        node = row.node
        descriptor = self._spec_for_logic(node)
        if descriptor is None:
            raise RuntimeError(
                "no descriptor for the saved DefinitionKey"
            )
        if self._host_factory is None:
            raise RuntimeError(
                "this console was opened without a Logic-node host factory"
            )
        own_keys = frozenset(
            (*self._declared_signal_keys(row), *self._declared_artifact_keys(row))
        )
        resolved = self.resolve_node_inputs(
            descriptor,
            inputs,
            own_keys=own_keys,
        )
        result = self._host_factory(
            descriptor,
            dict(authored),
            resolved,
            node.node_id,
            self.request_owner_wake,
        )
        if not isinstance(result, LogicNodeHost):
            raise TypeError("host_factory must return LogicNodeHost")
        return result

    def _enqueue_surface_batch(self, cards, front) -> bool:
        """Prepare one exact continuous group without presenting a partial board."""

        inputs = []
        for card in cards:
            signal_name = str(card.config.signal or "").strip()
            value = front.value(signal_name)
            publication = front.publication(signal_name)
            if value is None or publication is None:
                card.set_status(
                    f"waiting for {signal_name}" if signal_name else "pick a signal",
                    error=False,
                )
                return False
            inputs.append((card, value, publication))

        updates: list[PanelSurfaceUpdate] = []
        for card, value, publication in inputs:
            try:
                update = card.prepare_surface_update(value, publication)
            except (KeyError, LookupError, RuntimeError, TypeError, ValueError) as error:
                card.set_status(f"Display: {error_summary(error)}", error=True)
                for submitted in updates:
                    owner = next(
                        (
                            item
                            for item, _value, _publication in inputs
                            if item.panel_id == submitted.panel_id
                        ),
                        None,
                    )
                    if owner is not None:
                        owner.finish_unpresented_surface_update(submitted)
                return False
            if update is not None:
                updates.append(update)
        if not updates:
            return False
        batch = tuple(updates)
        self._surface_batches.append(batch)
        for update in batch:
            update.future.add_done_callback(self._surface_future_done)
        return True

    def _surface_future_done(self, _future: Future) -> None:
        with self._owner_event_lock:
            if self._permanently_closed:
                return
            self._surface_wake_pending = True
        self._owner_wake.request_owner_wake()

    def _drain_surface_batches(self) -> None:
        pending: deque[tuple[PanelSurfaceUpdate, ...]] = deque()
        while self._surface_batches:
            batch = self._surface_batches.popleft()
            if not all(update.future.done() for update in batch):
                pending.append(batch)
                continue
            cards = [
                next(
                    (item for item in self.cards if item.panel_id == update.panel_id),
                    None,
                )
                for update in batch
            ]
            operations: list[RasterOperation | None] = []
            for card, update in zip(cards, batch, strict=True):
                try:
                    operation = update.future.result()
                    if not isinstance(operation, RasterOperation):
                        raise TypeError("plot worker returned another operation type")
                    if card is not None:
                        card.observe_surface_result(update, operation)
                    operations.append(operation)
                except CancelledError:
                    if card is not None:
                        card.reject_surface_update(update)
                    operations.append(None)
                except BaseException as error:
                    if card is not None:
                        card.reject_surface_update(update)
                        card.set_status(
                            f"Display: {error_summary(error)}",
                            error=True,
                        )
                    operations.append(None)
            if (
                any(card is None for card in cards)
                or any(operation is None for operation in operations)
                or not all(
                card.can_accept_surface_update(update, operation)
                for card, update, operation in zip(
                    cards, batch, operations, strict=True
                )
                )
            ):
                continue
            for card, update, operation in zip(cards, batch, operations, strict=True):
                assert card is not None and operation is not None
                card.accept_surface_update(update, operation)
        self._surface_batches.extend(pending)

    def _forget_card_render(self, card: PanelCard) -> None:
        """Invalidate this card's pending immutable worker results."""

        card.retire_source_generation()

    def _shutdown_presentation_workers(self) -> bool:
        """Retire every zlc_plot owner before application resources disappear."""

        for batch in self._surface_batches:
            for update in batch:
                update.future.cancel()
        self._surface_batches.clear()
        closed = True
        for editor in tuple(self._panel_editors.values()):
            closed = editor.teardown() and closed
        for editor in tuple(self._retiring_panel_editors):
            closed = editor.teardown() and closed
        for card in self.cards:
            closed = card.shutdown() and closed
        if not closed:
            QtCore.QTimer.singleShot(25, self.request_owner_wake)
        return closed

    def _retire_logic_node_publications(
        self,
        node: _ConsoleNode,
    ) -> None:
        """Retire one generation and its complete neutral descendant closure.

        Retirement covers the complete causal descendant closure, not merely
        the node's declared routes.  Otherwise a retained selector or
        Processor publication could re-introduce the old generation while the
        coherence gate builds its next front.  A topology reader must never
        run between that retirement and promotion of the matching immutable
        front.  This is the single path shared by restart and row removal.
        """

        retired_names = self._data.retire(node)
        for card in self.cards:
            output_prefix = f"@panel/{card.panel_id}/"
            owns_retired_output = any(
                name.startswith(output_prefix) for name in retired_names
            )
            if self._card_reads(card).intersection(retired_names) or owns_retired_output:
                self._withdraw_panel_outputs(card.panel_id)
                self._forget_card_render(card)
        self._promote_data_front(self._data.freeze())

    def _stop_logic_node(
        self,
        row: "LogicNodeRow",
        *,
        _silent: bool = False,
    ) -> bool:
        """Cancel a logic row's hosted owner and grey its dot."""
        node = self._logic_nodes.get(id(row))
        if node is not None:
            if not self._stop_node_owner(node):
                editor = self._logic_editors.get(id(row))
                if not _silent:
                    row.set_state("running", status="stop pending: node owner not terminal")
                    if editor is not None:
                        editor.set_running(True)
                        editor.set_status(
                            "stop pending: node owner not terminal", error=True
                        )
                return False
        self._remove_transient_result_panels(row)
        self._logic_nodes[id(row)] = None
        self._sync_terminal_poll_timer()
        # Running, retained FINAL/view, and declared-not-started are
        # intentionally different picker/legend states.
        self._signal_topology_changed()
        editor = self._logic_editors.get(id(row))
        if not _silent:
            row.set_state("stopped", status="stopped")
            self._update_row_publishes(row)        # back to the spec-declared outputs
            if editor is not None:
                editor.set_running(False)
                editor.set_status("stopped", error=False)
        return True

    def _remove_logic_node(
        self,
        row: "LogicNodeRow",
        *,
        _state_change: bool = True,
    ) -> bool:
        """Stop + remove a logic node (its node is stopped, its row + Edit drop)."""
        # Removing a row also retires its retained terminal display source.
        gone = (
            self._logic_nodes.get(id(row))
            or self._last_node.get(id(row))
        )
        if not self._stop_logic_node(row, _silent=True):
            return False
        editor = self._logic_editors.pop(id(row), None)
        if editor is not None:
            idx = self.tabs.indexOf(editor)
            if idx >= 0:
                self.tabs.removeTab(idx)
            editor.teardown()
            editor.hide()
            editor.deleteLater()
        self._logic_nodes.pop(id(row), None)
        self._last_node.pop(id(row), None)
        if gone is not None:
            self._retire_logic_node_publications(gone)
        if row in self.logic_nodes:
            self.logic_nodes.remove(row)
        self._signal_topology_changed()
        self.logic_layout.removeWidget(row)
        # The Logic page owns the retired row until DeferredDelete; there is no
        # intermediate unparented-widget state.
        row.hide()
        row.deleteLater()
        self._refresh_logic_hint_visibility()
        if _state_change:
            self._mark_dirty()
        return True

    def _poll_logic_nodes(self) -> None:
        """Project the same LogicNodeObservation for every descriptor kind."""

        self._promote_data_front(self._data.freeze())
        for row in self.logic_nodes:
            node = self._logic_nodes.get(id(row))
            if node is None:
                continue
            editor = self._logic_editors.get(id(row))
            observation = node.poll()
            warnings = tuple(observation.warnings)
            warning_text = "" if not warnings else "; warning: " + "; ".join(warnings)
            if not observation.terminal:
                status = observation.phase + warning_text
                row.set_state("running", status=status)
                if editor is not None:
                    editor.set_running(True)
                    editor.set_status(status, error=False)
                continue

            error = observation.error
            status = (
                f"error: {error}"
                if error is not None
                else f"{observation.phase}{warning_text}"
            )
            row.set_state("error" if error else "stopped", status=status)
            if editor is not None:
                editor.set_running(False)
                editor.set_status(status, error=error is not None)
            if not node.worker_idle:
                self.request_owner_wake()
                continue
            succeeded = error is None and observation.phase == "done"
            self._stop_logic_node(row, _silent=True)
            self._promote_data_front(self._data.freeze())
            if succeeded:
                self._ensure_task_preview_panels(row)
        self._sync_terminal_poll_timer()

    def _mark_dirty(self, *_args) -> None:
        if self._building:
            return
        if self.save_button is not None:            # embedded: no layout Save button to flag dirty
            self.save_button.set_dirty(True, dirty_color=YELLOW)
        self._update_summary()

    # ------------------------------------------------------------------ refresh
    def request_owner_wake(self) -> None:
        """Thread-safe request for one coalesced TaskConsole owner turn."""

        with self._owner_event_lock:
            if self._permanently_closed:
                return
            self._lifecycle_wake_pending = True
        self._owner_wake.request_owner_wake()

    def _request_data_owner_wake(self) -> None:
        """Wake reactive data routing without polling unrelated Run owners."""

        with self._owner_event_lock:
            if self._permanently_closed or not self._accept_data_wake:
                return
            self._data_wake_pending = True
        self._owner_wake.request_owner_wake()

    def _activate_data_owner_wake(self) -> None:
        """Commit the shared-plane wake borrow after window composition succeeds."""

        if self._data_wake_token is not None:
            raise RuntimeError("TaskConsole data wake is already active")
        with self._owner_event_lock:
            self._accept_data_wake = True
        try:
            token = self._data.bind_owner_wake(self._request_data_owner_wake)
        except BaseException:
            with self._owner_event_lock:
                self._accept_data_wake = False
            raise
        self._data_wake_token = token

    def _owner_cycle(self) -> None:
        """Admit lifecycle/data-plane events on the sole Qt owner."""

        if self._permanently_closed:
            return
        with self._owner_event_lock:
            lifecycle = self._lifecycle_wake_pending
            data = self._data_wake_pending
            surface = self._surface_wake_pending
            self._lifecycle_wake_pending = False
            self._data_wake_pending = False
            self._surface_wake_pending = False
        if surface:
            self._drain_surface_batches()
        if lifecycle:
            self._poll_logic_nodes()
        if (
            (lifecycle or data)
            and not getattr(self, "_shut", False)
            and not self._display_suspended
        ):
            # A producer revision may have arrived with no visible panel.  Freeze
            # here so reactive Processors advance from the event itself; the
            # display timer still decides when a panel is actually recomposed.
            self._promote_data_front(self._data.freeze())
            self._update_summary()
        if self._owner_retiring:
            self._advance_owner_retirement()
        elif self._pending_window_action is not None:
            # Window retirement waits on node, raster, selector, and Fit work.
            # Any one of those owner completions must be able to advance the
            # same close transition; tying the retry only to Run polling can
            # strand a window after the final Fit completion.
            self._continue_pending_window_action()

    # ------------------------------------------------- reading the frozen tick
    def _signal_values(self, name):
        """This signal's array as of the current tick, or None if it has none yet."""

        value = self._tick_data.value(str(name))
        return None if value is None else value.values

    def _signal_schema(self, name):
        """This signal's schema as of the current tick, or None if unpublished."""

        value = self._tick_data.value(str(name))
        return None if value is None else value.schema

    def _recompute_tick_interval(self) -> None:
        """Re-base the shared timer to the SMALLEST panel ``update_ms`` (which divides every
        other rate in :data:`UPDATE_INTERVALS`, so the rates co-align) and reset the tick
        phase.  Called on add / remove / per-panel rate change, so the timer ticks no faster
        than the fastest panel actually needs."""
        base = min((c.config.update_ms for c in self.cards), default=DEFAULT_UPDATE_MS)
        self._base_interval_ms = max(min(UPDATE_INTERVALS), int(base))
        self._tick_count = 0                      # re-sync the phase to the new base
        if getattr(self, "_timer", None) is not None:
            self._timer.setInterval(self._base_interval_ms)
            self._sync_display_timer()

    def _sync_display_timer(self) -> None:
        """Run presentation cadence only while a visible board can consume it."""

        timer = getattr(self, "_timer", None)
        if timer is None:
            return
        active = bool(self.cards) and not (
            self._paused
            or self._display_suspended
            or self._permanently_closed
            or getattr(self, "_shut", False)
        )
        if active and not timer.isActive():
            timer.start()
        elif not active and timer.isActive():
            timer.stop()

    def _sync_terminal_poll_timer(self) -> None:
        """Keep only the narrow fallback needed by Runs without terminal events."""

        timer = getattr(self, "_terminal_timer", None)
        if timer is None:
            return
        active = any(node is not None for node in self._logic_nodes.values()) and not (
            self._permanently_closed or getattr(self, "_shut", False)
        )
        if active and not timer.isActive():
            timer.start()
        elif not active and timer.isActive():
            timer.stop()

    def _toggle_pause(self) -> None:
        """Pause / Resume the WHOLE monitor: freeze or unfreeze every plot's live display at once.
        This is a DISPLAY freeze -- acquisition keeps running (Stop a Logic node to halt that)."""
        self._paused = not self._paused
        self.pause_button.setText("Resume" if self._paused else "Pause")
        self.pause_button.set_color(GREEN if self._paused else ORANGE)
        # A paused display freezes every panel at its last accepted front.  Those
        # fronts may belong to independent producers and therefore carry no
        # board-wide same-shot claim.  On Resume each stale card catches up from
        # its own source revisions on its next update_ms beat.
        self._sync_display_timer()
        if not self._paused and self.cards:
            self._tick()

    def _toggle_selectors(self, on: bool) -> None:
        """Header "Selectors" switch: arm (ON) or park (OFF) the selector layer of EVERY dashboard
        panel in place (``PanelCard.set_selectors_enabled`` -> ``BaseLivePlot.set_selectors_active``)
        -- a pure display gate, no rebuild, no effect on acquisition or the Edit tabs."""
        for card in self.cards:
            card.set_selectors_enabled(bool(on))

    def _save_board_image(self) -> None:
        """Save the WHOLE monitor board (every panel, composited in its laid-out position) to a PNG.
        Unlike a per-plot save (data png+npz), this is a raster of the board region, so it is a plain
        image.  Reuses ``_last_save_dir`` so it shares the per-panel save's remembered folder."""
        default_dir = self._last_save_dir or str(
            self._figures_root / "task-console"
        )
        default = str(Path(default_dir) / time.strftime("monitor_%Y_%m_%d_%H_%M_%S.png", time.localtime()))
        path, _sel = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save monitor image", default, "PNG image (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        # Grab ONLY the MINIMAL rectangle that bounds the laid-out cards (union of their geometries +
        # one GAP margin), NOT the whole board widget -- the board is sized to fill the scroll viewport
        # (setWidgetResizable), so grabbing it would bake in a big empty plot area below/right of the
        # panels.  Clamp to the board so the sub-rect is valid; empty board -> the whole (tiny) board.
        # Save the currently accepted immutable fronts.  A worker may be
        # preparing a newer batch, but Qt never waits for it or reads its Agg
        # state; the saved board is exactly what the operator currently sees.
        cards = list(self.cards)
        if cards:
            rect = cards[0].geometry()
            for card in cards[1:]:
                rect = rect.united(card.geometry())
            rect = rect.adjusted(-GAP, -GAP, GAP, GAP).intersected(self.board.rect())
            pm = self.board.grab(rect)
        else:
            pm = self.board.grab()
        # PanelBoard is transparent; flatten the exported board onto white.
        opaque_white_composite(pm).save(path)
        self._last_save_dir = str(Path(path).parent)
        self._update_summary()

    def _panel_render_groups(self, front) -> tuple[tuple[PanelCard, ...], ...]:
        """Group live surfaces by the plane-resolved continuous frontier."""

        grouped: dict[object, list[PanelCard]] = {}
        for card in self.cards:
            signal_name = str(card.config.signal or "").strip()
            continuous = front.continuous_group(signal_name)
            key = (
                ("continuous", continuous)
                if continuous
                else ("unbound", card.panel_id)
            )
            grouped.setdefault(key, []).append(card)
        return tuple(tuple(cards) for cards in grouped.values())

    def _tick(self) -> None:
        # This timer owns presentation cadence only.  Worker/data-plane events
        # use ``QtOwnerWake``; Runs without a terminal event use the separate
        # narrow terminal timer.  An idle board therefore performs no tree scan.
        if self._paused or not self.cards or self._permanently_closed:
            self._sync_display_timer()
            return
        self._tick_count += 1
        # One immutable front for this GUI pass.  It is a per-producer-latest
        # observation, not a board-wide physical-shot join.
        self._promote_data_front(self._data.freeze())
        elapsed = self._tick_count * self._base_interval_ms
        # Connected continuous panels share one causal group and therefore one
        # display beat.  Independent producers remain independent groups.  A
        # due group always freezes every member from this exact SignalFront;
        # a held/invalid member keeps the prior complete group visible.
        for cards in self._panel_render_groups(self._tick_data):
            if all(
                card.presented_publication
                is self._tick_data.publication(str(card.config.signal or ""))
                for card in cards
            ):
                continue
            if elapsed % max(card.config.update_ms for card in cards) != 0:
                continue
            self._enqueue_surface_batch(cards, self._tick_data)
        self._update_summary()

    def _note_display_drops(self) -> int:
        """Events the monitor taps dropped because the display fell behind.

        A tap overwrites rather than back-pressuring acquisition -- which is
        deliberately never rate-capped -- and reports the loss per signal.  The
        advisory sums those: there is no global shot clock to subtract against,
        and one would compare runs that advance independently.
        """

        return sum(value.behind for value in self._tick_data.signals.values())

    def _update_summary(self) -> None:
        n_signals = len(self._tick_data.names())
        telemetry = f"{len(self.cards)} panels | {n_signals} signals"
        if telemetry != getattr(self, "_summary_text", None):
            self._summary_text = telemetry
            self.summary.setText(telemetry)
        dropped = self._note_display_drops()
        # The persistent strip's ONE priority ladder (every tick): a wedged node must never fail
        # silently, so a red node error outranks the
        # display-behind advisory is amber (the RUN is unaffected -- acquisition is unthrottled
        # and always shows latest); idle shows the board summary.  The strip itself change-gates
        # text/style, so this per-tick call never repolishes an unchanged state.
        faulted = [
            node
            for node in self._logic_nodes.values()
            if node is not None and node.last_error is not None
        ]
        if faulted:
            node = faulted[0]
            who = self._node_label(node).rstrip("_:")
            self.status_strip.show_message(
                f"⚠ NODE ERROR ({who}): {node.last_error}", severity="error")
        elif dropped:
            self.status_strip.show_message(
                f"⚠ display behind: the monitor taps dropped {dropped} event(s) -- "
                "acquisition unaffected", severity="warning")
        else:
            # Idle telemetry already lives in the header.  Keep the status surface empty (but at its
            # fixed height, so layout never jumps) until an event worth the operator's attention exists.
            self.status_strip.show_message("", severity="info")

    # ------------------------------------------------------------------ files
    def save_to_file(self) -> None:
        try:
            state = self.read_state()
            start = self._address or str(
                _task_files_dir(self._tasks_root) / f"{state.name}.json"
            )
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save task layout", start, "Task layout (*.json)")
            if not path:
                return
            saved_path = save_task_console_state(state, path)
            self._address = str(saved_path)
            if self.save_button is not None:
                self.save_button.set_dirty(False)
            self._message(f"Saved: {path}")
        except Exception as exc:
            self._message(f"Save failed: {exc}")

    def load_from_file(self) -> None:
        try:
            start = (
                str(Path(self._address).parent)
                if self._address
                else str(_task_files_dir(self._tasks_root))
            )
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load task layout", start, "Task layout (*.json)")
            if not path:
                return
            state = load_task_console_state(path)
            self._address = path
            self.load_state(state)
            if self.save_button is not None:
                self.save_button.set_dirty(False)
        except Exception as exc:
            self._message(f"Load failed: {exc}")

    def _message(self, text: str) -> None:
        if os.environ.get("QT_QPA_PLATFORM", "") == "offscreen":
            self.status_strip.show_message(str(text), severity="info")
            return
        fluent_message(self, "Task console", str(text), kind="info")  # pragma: no cover

    # ------------------------------------------------------------------ teardown
    def stop_all_nodes(self) -> bool:
        """Stop every running node through THE lifecycle endpoint (``_stop_logic_node``) but KEEP
        the panels + editors intact.  This is the notebook close = "hide" path: closing the window
        halts all running processes, yet the layout is preserved so reopening the session-bound
        console (``exp.task_console()``) restores the SAME interface rather than a blank new one.
        Distinct from :meth:`shutdown`, which also tears the cards/editors down.

        Going row-by-row through ``_stop_logic_node`` is load-bearing: the row map
        is the sole active-node owner, so terminal cleanup and row presentation
        cannot drift into two independently-maintained registries."""
        if QtCore.QThread.currentThread() is not self.thread():
            raise RuntimeError("node shutdown must run on the TaskConsole Qt owner thread")
        stopped = True
        for row in list(self.logic_nodes):
            if self._logic_nodes.get(id(row)) is not None:
                stopped = self._stop_logic_node(row) and stopped
        if not stopped:
            self.status_strip.show_message(
                "Close delayed: a logic-node owner thread is still active",
                severity="error",
            )
        self._sync_terminal_poll_timer()
        return stopped

    def _attach_window(self, window) -> None:
        """Attach the sole Fluent wrapper used to continue an async close."""

        if self._window is not None and self._window is not window:
            raise RuntimeError("TaskConsole already belongs to another window")
        self._window = window

    def request_window_hide(self) -> bool:
        """Stop node owners without blocking, then let the wrapper hide."""

        if self._permanently_closed:
            return True
        self._display_suspended = True
        self._sync_display_timer()
        self._pending_window_action = "hide"
        self._window_retry_scheduled = False
        ready = self.stop_all_nodes()
        if ready:
            self._pending_window_action = None
        return ready

    def request_window_close(self) -> bool:
        """Begin full teardown and complete it on later Qt owner turns."""

        if self._permanently_closed:
            return True
        self._display_suspended = True
        self._sync_display_timer()
        self._pending_window_action = "close"
        self._window_retry_scheduled = False
        ready = self.shutdown()
        if ready:
            self._pending_window_action = None
            self._commit_permanent_close(delete_window=False)
        return ready

    @property
    def permanently_closed(self) -> bool:
        """Whether this Experiment-owned handle can no longer be restored."""

        return self._permanently_closed

    def restore_window(self) -> None:
        """Restore the same bound hide-on-close console and authored layout."""

        if QtCore.QThread.currentThread() is not self.thread():
            raise RuntimeError("TaskConsole restore must run on its Qt owner thread")
        if self._owner_retiring or self._permanently_closed:
            raise RuntimeError("TaskConsole is closing with its application")
        window = self._window
        if window is None:
            raise RuntimeError("TaskConsole window is unavailable")
        self._display_suspended = False
        window.showNormal()
        window.raise_()
        window.activateWindow()
        self._sync_display_timer()
        self.request_owner_wake()

    def wait_owner_closed(self, timeout: float) -> bool:
        """Wait for the worker-acknowledged permanent close."""

        return wait_for_owner_retirement(
            self,
            self._owner_closed,
            timeout=timeout,
        )

    def request_owner_close(self) -> None:
        """Thread-safely retire this borrowed GUI when Experiment closes."""

        with self._owner_close_lock:
            if self._permanently_closed:
                return
            self._owner_retiring = True
        if QtCore.QThread.currentThread() is self.thread():
            # Already on the owner: advance immediately so Experiment.close()
            # cannot return with an idle window still retained.  A foreign
            # caller takes the queued branch below and never touches QWidget.
            self._advance_owner_retirement()
        else:
            self._owner_wake.request_owner_wake()

    def _advance_owner_retirement(self) -> None:
        """Advance application-owned teardown on the Qt owner without blocking."""

        if self._permanently_closed:
            return
        self._display_suspended = True
        self._sync_display_timer()
        window = self._window
        if window is not None and window.isVisible():
            window.hide()
        self._pending_window_action = "owner-close"
        if self.shutdown():
            self._pending_window_action = None
            self._commit_permanent_close(delete_window=True)

    def _commit_permanent_close(self, *, delete_window: bool) -> None:
        """Commit the one irreversible Workbench-handle transition."""

        if self._permanently_closed:
            return
        self._permanently_closed = True
        self._display_suspended = True
        self._timer.stop()
        self._terminal_timer.stop()
        self._owner_wake.detach()
        window = self._window
        self._window = None
        if window is None:
            self._owner_closed.set()
            return
        release_window(window)
        if delete_window:
            window.hide()
            window.deleteLater()
        self._owner_closed.set()

    def _continue_pending_window_action(self) -> None:
        """Retry an ignored X action after node terminal truth advances."""

        action = self._pending_window_action
        window = self._window
        if action is None or window is None or self._window_retry_scheduled:
            return
        ready = self.stop_all_nodes() if action == "hide" else self.shutdown()
        if not ready:
            return
        self._pending_window_action = None
        if action == "owner-close":
            self._commit_permanent_close(delete_window=True)
            return
        if action == "close":
            self._commit_permanent_close(delete_window=False)
        self._window_retry_scheduled = True
        QtCore.QTimer.singleShot(0, window.close)

    def shutdown(self) -> bool:
        """Stop the refresh timer and every running node's owner thread, then release
        the editors/cards.  IDEMPOTENT -- it is reached from both the window close
        (``show_task_console`` installs it as the pre-close guard) and an explicit
        ``with show_task_console(...)`` / re-run, which can both fire.

        Stopping a node sets its cooperative-cancel event (M5), so a node thread
        blocked in ``camera.acquire`` unwinds and the camera is released -- this is
        what keeps a closed dashboard from leaving a live acquire thread (and a held
        camera / RPyC connection) behind.
        Raster work owns no QWidget, but its frontend sessions must retire on
        their creating worker before data/UI ownership ends.  True means node,
        renderer, resource and Qt teardown completed; False keeps the window
        alive for a completion-driven owner turn."""
        if QtCore.QThread.currentThread() is not self.thread():
            raise RuntimeError("TaskConsole shutdown must run on its Qt owner thread")
        state = getattr(self, "_shutdown_state", "RUNNING")
        if state == "TERMINATED" or getattr(self, "_shut", False):
            return True
        if state in {
            "STOPPING_NODES",
            "CLOSING_RESOURCES",
            "TEARING_DOWN_UI",
        }:
            return False
        if state in {"RUNNING", "BLOCKED_NODE_OWNERSHIP"}:
            self._shutdown_state = "STOPPING_NODES"
            if not self.stop_all_nodes():
                self._shutdown_state = "BLOCKED_NODE_OWNERSHIP"
                return False
            # A permanently closed surface must not leave Area/Cross/Fit
            # generations in the Experiment-owned SignalPlane.  Retire every
            # card-owned route while the selector/render owner is still alive,
            # then promote the one resulting neutral front.  Normal panel
            # removal uses the same route owner path.
            for card in tuple(self.cards):
                self._withdraw_panel_outputs(card.panel_id)
            self._promote_data_front(self._data.freeze())
            self._timer.stop()
            self._terminal_timer.stop()
            self._shutdown_state = "WAITING_RENDER"
        if state == "WAITING_RENDER" or self._shutdown_state == "WAITING_RENDER":
            if not self._shutdown_presentation_workers():
                return False
        with self._owner_event_lock:
            self._accept_data_wake = False
            self._data_wake_pending = False
        if self._data_wake_token is not None:
            self._data.unbind_owner_wake(self._data_wake_token)
            self._data_wake_token = None
        if not getattr(self, "_on_close_done", False):
            self._shutdown_state = "CLOSING_RESOURCES"
            on_close = getattr(self, "_on_close", None)
            if on_close is not None:
                try:
                    close_result = on_close()
                    if close_result is False:
                        raise RuntimeError("resource close reported unresolved ownership")
                except Exception as exc:
                    self._shutdown_error = exc
                    self._shutdown_state = "BLOCKED_RESOURCE_CLOSE"
                    self.status_strip.show_message(
                        f"Close delayed: device/resource release failed: {exc}",
                        severity="error",
                    )
                    return False
            self._on_close_done = True
        self._shutdown_state = "TEARING_DOWN_UI"
        try:
            for key, editor in list(self._panel_editors.items()):
                editor.teardown()
                self._panel_editors.pop(key, None)
            for key, editor in list(self._logic_editors.items()):
                editor.teardown()
                self._logic_editors.pop(key, None)
            completed_cards = getattr(self, "_shutdown_cards_done", set())
            self._shutdown_cards_done = completed_cards
            for card in self.cards:
                if id(card) in completed_cards:
                    continue
                card.shutdown()
                completed_cards.add(id(card))
        except Exception as exc:
            self._shutdown_error = exc
            self._shutdown_state = "BLOCKED_UI_TEARDOWN"
            self.status_strip.show_message(
                f"Close delayed: UI teardown failed: {exc}", severity="error"
            )
            return False
        self._shutdown_state = "TERMINATED"
        self._shut = True
        return True

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if not self.request_window_close():
            event.ignore()
            return
        super().closeEvent(event)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self.shutdown():
            raise RuntimeError("TaskConsole shutdown did not complete")
        self._commit_permanent_close(delete_window=True)
        return False

def show_task_console(
    *,
    descriptors: tuple[LogicNodeDescriptor, ...],
    device_catalog: DeviceCatalogView,
    host_factory,
    data_plane: SignalDataPlane,
    project_root: Path,
    pulses_root: Path,
    tasks_root: Path,
    figures_root: Path,
    state: TaskConsoleState | None = None,
    task: str | None = None,
    scale: float | None = None,
    window_ratio: float = WINDOW_SCREEN_FRACTION,
    title: str = "TaskConsole@Zou lab",
    on_close=None,
    hide_on_close: bool = False,
):
    """Open the console in a Fluent window (mirrors ``launch_pulse_editor_window``: the body
    sizes itself from the primary screen; the window wraps it exactly).

    ``task`` loads a layout YOU saved (``tasks/<name>.json``) or a JSON path;
    without one the console opens empty.

    Logic rows are projected directly from discovered descriptors.  The one
    injected host factory binds all three kinds through the same lifecycle.

    Teardown: closing the window stops the refresh timer and every running node's owner
    thread (the cooperative-cancel path), so no acquire thread is left holding the
    camera -- closing the dashboard genuinely releases it.  Pass ``on_close`` (e.g.
    ``exp.close``) to ALSO disconnect/safe the devices the caller owns when the
    window closes.  In a notebook the returned console is also a context manager
    (``with show_task_console(...) as console:``) and exposes ``console.shutdown()``
    for deterministic teardown on a cell re-run."""

    ensure_qt_app()          # the console is a QWidget: the app must exist BEFORE its ctor
    if state is None and task is not None:
        state = resolve_task_state(task, tasks_root=tasks_root)
    console = TaskConsole(
        state=state,
        descriptors=descriptors,
        device_catalog=device_catalog,
        host_factory=host_factory,
        data_plane=data_plane,
        project_root=project_root,
        pulses_root=pulses_root,
        tasks_root=tasks_root,
        figures_root=figures_root,
        scale=scale,
        window_ratio=window_ratio,
    )
    console._on_close = on_close
    # Closing the window must stop the node owner threads (else they keep running, blocked in
    # camera.acquire holding the camera / RPyC link, wedging the kernel).  The console is a CHILD
    # of the window so its own closeEvent never fires on a window close -- we wire the window's
    # signals instead.  Minimising NEVER stops the nodes (only the X / a genuine close does).
    def _wire_close(window) -> None:
        console._attach_window(window)
        if hide_on_close:
            # Session-bound (notebook) console: the X HIDES the window (keeps the panel layout so a
            # later exp.task_console() restores the SAME interface) and stops every running node so
            # the devices are released.  The hide guard runs on X, never on minimize.
            window.set_hide_guard(console.request_window_hide)
        else:
            # Standalone window (.bat / explicit): the X fully tears the console down.
            window.set_close_guard(console.request_window_close)
    # ONE launcher sequence (launch_fluent_window: wrap -> wire -> size -> centre -> show ->
    # retain), shared with every other show_* GUI so the steps cannot drift per-launcher.
    launch_fluent_window(console, title=title, hide_on_close=hide_on_close, wire=_wire_close)
    try:
        console._activate_data_owner_wake()
    except BaseException:
        if console.shutdown():
            console._commit_permanent_close(delete_window=True)
        raise
    return console
