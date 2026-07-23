"""The task console window itself: tabs, board, logic rows, render bridge, lifecycle.

This is the composition CORE salvaged out of the legacy shell -- it holds the panel cards
(plot_bridge objects), the background render bridge and the whole window lifecycle, so its
transitional home is the plot_bridge zone.  The render-purification pass dissolves it: the
wiring sinks into ``app.py`` and the widgets graduate into ``qt_widgets``.

Every import names a TRUE owner -- nothing here touches the legacy tree.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import inspect
import os
import threading
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
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
    GREEN,
    GREY,
    LogicNodeEditor,
    QtOwnerWake,
    LogicNodeRow,
    ORANGE,
    WINDOW_SCREEN_FRACTION,
    YELLOW,
    ensure_qt_app,
    fluent_message,
    fluent_widget_stylesheet,
    launch_fluent_window,
    scaled_px,
    screen_fit_window_size,
    set_fluent_scale,
    window_pad,
)
from zlc_frontend.console_state import (
    TaskConsoleState,
    default_console_state,
    resolve_task_state,
    task_files_dir as _task_files_dir,
)
from zlc_data.console_records import (
    DEFAULT_UPDATE_MS,
    LOGIC_KINDS,
    LogicNodeConfig,
    PANEL_KINDS,
    PanelConfig,
    UPDATE_INTERVALS,
)
from zlc_data.shape_text import indexed_unique_name, strip_node_prefix
from .plot_bridge import (
    GAP,
    PanelCard,
    _PanelRenderRequest,
    _PanelBoard,
    _board_width,
    _opaque_white_composite,
    drop_index,
    pack,
)
from .data_plane import ConsoleDataPlane
from .plot_bridge_editor import PanelEditor


# ====================================================================== console
class TaskConsole(QtWidgets.QWidget):
    """The dashboard window body: header bar + drag-and-snap panel board."""

    def __init__(
        self,
        *,
        state: TaskConsoleState | None = None,
        running_nodes: Sequence[object] = (),
        catalog_view: object | None = None,
        run_factory=None,
        data_plane=None,
        fit_window_factory=None,
        scale: float | None = None,
        window_ratio: float = WINDOW_SCREEN_FRACTION,
        window_px: tuple[int, int] | None = None,
        embedded: bool = False,
    ):
        ensure_qt_app()
        set_fluent_scale(scale)
        super().__init__()
        # MONITOR SEAM (contract 3): the board reads ONE frozen snapshot per tick.
        # The plane owns the live slots the RUN seam's start closures register;
        # ``self._tick_data`` is the snapshot the current tick is drawing from, so
        # every widget in one tick describes the same instant.
        self._data = data_plane if data_plane is not None else ConsoleDataPlane()
        self._tick_data = self._data.freeze()
        # EMBEDDED mode (the figure viewer hosts a whole TaskConsole in one pane): a stripped,
        # RESIZABLE console -- no Logic tab, no whole-board Pause/Save image/Save/Load buttons (a
        # loaded static figure has nothing to acquire, freeze, or persist as a layout) -- and, crucially,
        # NOT size-frozen: the parent layout stretches it so the gravity board reads the REAL viewport
        # width and reflows into 2+ columns.  Standalone (embedded=False) behaviour is byte-for-byte
        # unchanged.  The four omitted buttons are set to None so every reference guards on existence.
        self.embedded = bool(embedded)
        # Nodes that are CURRENTLY running (their owner thread is publishing to
        # the hub).  Populated on Start (``_start_logic_node``) and drained on Stop;
        # an externally-supplied, already-running node may be adopted here too.
        # This is DISTINCT from ``self.logic_nodes`` (the Logic-tab ROWS): a row is a
        # declaration that exists whether or not its node is running.
        self.running_nodes = list(running_nodes)
        # RUN SEAM (contract 2): the ONE way this window starts anything.  The
        # composition root injects a factory that turns a catalog spec plus the
        # row's form values into a :class:`ConsoleRunNode` -- a frozen typed
        # request whose prepare/start/cancel all happen on its own worker.  With
        # no factory the window is a layout you can open and edit, but Start says
        # so plainly rather than reaching for the domain itself.
        if run_factory is not None and not callable(run_factory):
            raise TypeError("run_factory must be callable or None")
        self._run_factory = run_factory
        if fit_window_factory is not None and not callable(fit_window_factory):
            raise TypeError("fit_window_factory must be callable or None")
        self._fit_window_factory = fit_window_factory
        self._analysis_window = None
        self._analysis_source = None
        self._panel_teardown_phases: dict[int, set[str]] = {}
        # CATALOG SEAM (contract 1): the ONE capability vocabulary this window offers.
        # ``ConsoleCatalogView`` projects the domain DefinitionCatalog into addable
        # entries -- camera / measurement / processor / task -- each carrying its own
        # parameter form, declared outputs and typed-request builder.  Without a view
        # the window is plot-kinds only (a layout you can open with no session).
        self._catalog = catalog_view
        self.state = state or default_console_state()
        self.window_ratio = float(window_ratio)
        self._window_px = window_px
        self.cards: list[PanelCard] = []
        # The signal legend is a projection of two event-owned facts: provider
        # topology and each card's binding.  Mutations below mark it dirty and
        # reconcile it once; the refresh timer never rediscovers topology by
        # enumerating every provider/card merely to learn that nothing changed.
        self._signal_info_dirty = True
        self._card_signal_bindings: dict[int, tuple[str, tuple[str, ...]]] = {}
        # The Logic-tab nodes.  Each entry maps a LogicNodeRow -> the live state for
        # that node: its built node (None until Started) + its Edit tab.
        self.logic_nodes: list[LogicNodeRow] = []
        self._logic_nodes: dict[int, object] = {}      # id(row) -> node (or None when stopped)
        # id(row) -> the LAST built node, kept even after Stop (unlike _logic_nodes, which
        # is None'd on stop) -- so a finished/stopped node's signals that LINGER in the hub
        # still show WHICH node produced them in the signal picker (a stopped scan's
        # `readout_fidelity`, a stopped camera's `frame`).  Cleared only on row removal.
        self._last_node: dict[int, object] = {}
        self._logic_editors: dict[int, "LogicNodeEditor"] = {}  # id(row) -> Edit tab
        self._building = False
        self._address: str | None = None
        # The folder the LAST panel "Save Fig" wrote into -- so a panel's Edit-tab save
        # picker reopens at the same place across panels and across reopens (remembered
        # for the life of the process / Jupyter kernel, like the pulse GUI's save dir).
        # None until the first save -> the picker defaults to the tasks/ folder.
        self._last_save_dir: str | None = None
        # No-measurement / no-processor sentinels kept so older callers / tests that
        # probe "is there a global measurement launcher" still read None (there is
        # no global form -- a measurement is a Logic node now).
        self.measurement_panel = None
        self.measurement_group = None
        # Per-panel editors: one PanelEditor per opened PLOT panel, hosted as a
        # closable tab (keyed by id(card)).
        self._panel_editors: dict[int, "PanelEditor"] = {}
        # A running TASK takes over the console: the status strip reports its
        # lifecycle and all other actions remain locked until it finishes or is
        # stopped.  Live task data belongs to the formal monitor seam; the console
        # never invents an off-hub dataset or a blank transient plot for it.
        self._running_task_row: "LogicNodeRow | None" = None
        self._task_status_text: str | None = None
        self._task_locked = False              # True while a task runs -> all other actions blocked

        # Multi-rate refresh: the timer ticks at the BASE interval (the smallest panel
        # update_ms, which divides every other so the rates co-align); each panel redraws
        # every update_ms // base ticks.  _tick_count counts base ticks since the last re-base.
        self._tick_count = 0
        self._base_interval_ms = int(self.state.interval_ms)
        # Display reads each signal at its own latest value.  A producer that owns
        # related view signals publishes them atomically, so companion values remain
        # shot-coherent without console-side joins.

        # One thread-affine worker owns every PanelComposer/Agg object.  Qt only
        # freezes immutable requests and presents immutable BoardFrames.  While
        # one batch is running, newer requests replace the not-yet-started
        # request for that panel; the next batch is not dispatched until the Qt
        # owner has accepted/rejected the current result, so render continuity
        # is never determined by executor timing.
        self._render_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="zlc-task-console-raster",
        )
        self._render_lock = threading.Lock()
        self._render_future: Future | None = None
        self._render_completion = None
        self._render_pending: dict[str, _PanelRenderRequest] = {}
        self._render_reset_pending: set[str] = set()
        self._render_worker_composers: dict[str, tuple[object, object]] = {}
        self._render_closing = False
        self._render_wake = QtOwnerWake(self)
        self._render_wake.bind(self._render_owner_cycle)

        self._build_ui()
        self.load_state(self.state)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._recompute_tick_interval()          # base = min panel update_ms (sets the interval)
        self._timer.start()
        self._paused = False                     # Pause button: freeze EVERY plot's display at once

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
        # layout gives and demands no minimum of its own.  It used to ask for the WHOLE-window
        # size as its minimum, which made any host that put anything beside it wider than every
        # other GUI -- the shared rule sets the window, and the window is divided up inside it.
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
        # Persistent telemetry belongs to the original header: it is orthogonal to the event/status
        # channel below and therefore never disappears behind a task message or node error.
        self.summary = FluentLabel("")
        self.summary.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.kind_combo = FluentComboBox()
        # Add Panel offers EXACTLY the four kinds the user designs with -- nothing
        # invented, no composite:
        #   * "Plot: X"        -- a pure VIEW; shows nothing until you wire a hub
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
        # The node LAYERS, straight off the catalog view: every entry carries its own
        # display title, so a renamed domain Definition reaches the dropdown AND the row
        # title without a second literal.  The camera is NOT special-cased any more -- it
        # is one catalog entry among the others (the old ``("camera", "live")`` singleton
        # placeholder encoded an assumption the DefinitionCatalog does not make).
        for kind, layer in (("camera", "Measurement"), ("measurement", "Measurement"),
                            ("processor", "Processor"), ("task", "Task")):
            for spec in self._catalog_specs(kind):
                self.kind_combo.addItem(f"{layer}: {spec.name}", (kind, spec.name))
                # The definition's prose title is the row's tooltip: the menu stays
                # scannable while the full sentence stays one hover away.
                self.kind_combo.setItemData(self.kind_combo.count() - 1,
                                            spec.description, QtCore.Qt.ToolTipRole)
        self.kind_combo.setFixedWidth(scaled_px(170, minimum=130))
        add_button = FluentButton("Add Panel", color=ACCENT)
        add_button.clicked.connect(self._add_panel)
        # "Selectors" switch: the Monitor board is display-only BY DEFAULT (every panel builds
        # its selector layer but parks it inactive; the wheel scrolls the board) -- the original
        # anti-misclick rule.  Flip ON to arm the full selector layer (zoom/pan, area, cross,
        # draggable threshold/clim lines) on every dashboard panel IN PLACE -- no rebuild, and a
        # panel added while ON inherits it (see _attach_card).  A DISPLAY control like Pause, so
        # it stays OUT of the running-task lockout.
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
        # (``_lockable_header``, ``_mark_dirty``, ``_toggle_pause`` ...) guards on existence.
        if self.embedded:
            self.save_button = None
            load_button = None
            self.pause_button = None
            self.save_image_button = None
            self.devices_button = None
        else:
            self.save_button = FluentButton("Save", color=ACCENT)
            self.save_button.clicked.connect(self.save_to_file)
            load_button = FluentButton("Load", color=ORANGE)
            load_button.clicked.connect(self.load_from_file)
            self.pause_button = FluentButton("Pause", color=ORANGE)
            self.pause_button.clicked.connect(self._toggle_pause)
            self.save_image_button = FluentButton("Save image", color=ACCENT)
            self.save_image_button.clicked.connect(self._save_board_image)
            # The READ-ONLY device viewer: one tab per loaded device, no config editing -- the
            # console must never let an operator mutate the running device set.  The viewer has not
            # been rebuilt on the current data plane, so the button is DISABLED and says why: an
            # enabled button whose handler resolves to nothing reads as a broken window, which is
            # worse than one that admits the view is missing.
            self.devices_button = FluentButton("Devices", color=GREY)
            self.devices_button.setEnabled(False)
            self.devices_button.setToolTip("The read-only device viewer has not been rebuilt on this "
                                           "data plane yet.  The device list is available from a "
                                           "notebook as exp.device_catalog.")
            self.devices_button.clicked.connect(self._open_device_viewer)

        for widget in (self.status_dot, self.name_edit):
            header.addWidget(widget)
        header.addWidget(self.summary, 1)
        # Add Panel stays even when embedded (a loaded figure is re-wired + extra panels added), and so
        # does the Selectors switch (inspecting a loaded figure is exactly when the selectors help);
        # the four whole-console buttons only when they exist.
        for widget in (self.kind_combo, add_button, self.selectors_switch, self.devices_button,
                       self.pause_button, self.save_image_button, self.save_button, load_button):
            if widget is not None:
                header.addWidget(widget)
        # header controls disabled while a task runs (the lockout, #5); kept as a group
        # so ``_set_task_running`` flips them all at once.  Pause stays OUT of the lockout (you may
        # want to freeze the display even mid-task); Save image is locked like the other mutators.
        # None entries (embedded: the button was never made) are dropped so the lockout loop is safe.
        self._lockable_header = tuple(w for w in (self.kind_combo, add_button, self.save_image_button,
                                                  self.save_button, load_button, self.name_edit)
                                      if w is not None)
        root.addWidget(header_frame)

        # PERSISTENT status strip -- the ONE always-visible line between the header and the
        # tabs.  Its content switches by PRIORITY (node error > running task's progress >
        # display-behind advisory; idle is empty, see _update_summary); the Stop-task
        # action shows only while a task owns the console.  This replaces BOTH old mechanisms:
        # the transient orange task banner (its show/hide shifted the whole layout under the
        # pointer).  Header telemetry remains visible independently above it.
        self.status_strip = FluentStatusStrip(action_text="Stop task")
        self.status_strip.action_clicked.connect(self._stop_running_task)
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
        self.board = _PanelBoard()
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
            self.logic_scroll.setWidgetResizable(True)
            self.logic_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)   # #2: never scroll sideways
            self.logic_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
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
            self.logic_scroll.setWidget(logic_body)
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
        # the CURRENT parameter values even when the node was never Started (#4: a
        # layout persists Edit params, not just geometry).  Plot params already write
        # through to ``card.config.params`` live, so panels need no flush.
        for row in self.logic_nodes:
            editor = self._logic_editors.get(id(row))
            if editor is None:
                continue
            try:
                row.node.values = editor.collect_values()
            except Exception:
                pass
        return TaskConsoleState(
            name=self.name_edit.text().strip() or "task",
            interval_ms=self.state.interval_ms,
            panels=[card.config for card in self.cards],
            logic=[row.node for row in self.logic_nodes],
        )

    def load_state(
        self,
        state: TaskConsoleState,
        *,
        timeout: float = 5.0,
        _replacement_running_nodes: Sequence[object] | None = None,
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
            raise ValueError("state-load timeout must be a non-negative number")
        if not isinstance(state, TaskConsoleState):
            raise TypeError("state must be TaskConsoleState")
        # Loading consumes no caller-owned objects.  Teardown intentionally mutates
        # the live PanelConfig/LogicNodeConfig graph, so freeze a private desired
        # graph before the first stop/remove and install only that graph.
        desired_state = TaskConsoleState.from_dict(state.to_dict())
        replacement_nodes = None
        if _replacement_running_nodes is not None:
            replacement_nodes = list(_replacement_running_nodes)
        deadline = time.monotonic() + float(timeout)
        self._building = True
        try:
            for row in list(self.logic_nodes):
                if not self._stop_logic_node(
                    row,
                    _silent=True,
                    timeout=max(0.0, deadline - time.monotonic()),
                ):
                    raise RuntimeError(
                        "cannot load a new console state while a logic-node owner is still active"
                    )
            for row in list(self.logic_nodes):
                if not self._remove_logic_node(row, _rebuild=False):
                    raise RuntimeError("stopped logic node could not be removed")
            for card in list(self.cards):
                if not self._remove_panel(card):
                    raise RuntimeError(
                        "cannot remove a panel while loading the new console state"
                    )
            if replacement_nodes is not None:
                self.running_nodes = replacement_nodes
                self._signal_topology_changed()
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
        for card in self.cards:            # force every panel to redraw on its next beat
            card._render_version = -1
        self._recompute_tick_interval()    # the loaded panels' rates set the timer base
        self._sync_fit_analysis_entries()
        self._update_summary()

    def reseed(self, state: TaskConsoleState, *, running_nodes: Sequence[object] = ()) -> None:
        """Reload the board to a NEW state IN PLACE: reuse this console's WHOLE widget tree (tabs, board,
        header, hub, refresh timer) and rebuild only its PANELS (:meth:`load_state`), swapping the
        producing-node set.  The single reseed entry for a host that shows one console and re-points it
        at different data -- the figure_viewer's Browse re-load: swapping the seeded panel must NOT tear
        the console down and construct a fresh one, which re-realizes the entire Qt widget tree (~0.35 s
        of the perceived load).  The caller owns STOPPING the previous nodes (the console only reads the
        hub); a reused hub's stale signals are overwritten by same-named republishes or GC'd as orphans."""
        self.load_state(state, _replacement_running_nodes=running_nodes)

    def _new_panel_card(self, config: PanelConfig) -> PanelCard:
        """Build a PanelCard wired to the console's signal providers -- the ONE place the
        provider block lives, so adding/renaming a provider is a single edit here instead of three
        parallel edits at every PanelCard construction site (load_state / _add_panel / a task run) (#A1)."""
        return PanelCard(
            config, parent=self.board,
            names_provider=self._signal_names, sources_provider=self._signal_providers,
            formats_provider=self._signal_formats,
            short_names_provider=self._signal_short_names,
            render_request=self._request_card_render,
            fit_analysis_sink=self._open_panel_fit)

    def _attach_card(self, card: PanelCard) -> None:
        card.setParent(self.board)
        card.show()
        card.changed.connect(self._mark_dirty)
        card.changed.connect(self._sync_fit_analysis_entries)
        # ``changed`` also covers title/size/display edits.  Read only this
        # card's tiny binding delta and rebuild legends only when its signal
        # actually changed; ordinary display edits never enumerate providers.
        card.changed.connect(
            lambda c=card: self._card_signal_binding_changed(c)
        )
        card.dropped.connect(self._snap_dropped_card)   # drag-release only: snap BEFORE the re-pack
        card.layout_changed.connect(self._arrange)
        card.update_interval_changed.connect(self._recompute_tick_interval)
        card.remove_requested.connect(self._remove_panel)
        card.edit_requested.connect(self._edit_card)
        # a panel added (or loaded) while the header's "Selectors" switch is ON inherits it --
        # the guard covers construction order (state panels may attach before the header exists).
        switch = getattr(self, "selectors_switch", None)
        if switch is not None:
            card.set_selectors_enabled(switch.isChecked())
        self.cards.append(card)
        self._card_signal_bindings[id(card)] = self._card_signal_binding(card)
        self._signal_topology_changed()
        self._sync_fit_analysis_entries()
        self._recompute_tick_interval()        # a new panel's rate may change the timer base

    # ----------------------------------------------------------------- control
    # ---- producing-node discovery: a panel's data comes from a logic node; its Edit
    # exposes THAT node's acquisition parameters (e.g. a raw-frame panel is
    # produced by the camera measurement -> exposure / roi).
    def _card_reads(self, card: "PanelCard") -> set:
        """The panel's one typed dataset binding."""

        return {card.config.signal} if card.config.signal else set()

    def _producing_node(self, card: "PanelCard"):
        """The running node that publishes the panel's bound dataset."""
        refs = self._card_reads(card)
        if not refs:
            return None
        for node in self.running_nodes:
            published = node.published_signals() if hasattr(node, "published_signals") else frozenset()
            if refs & set(published):
                return node
        return None

    def _producing_row(self, card: "PanelCard"):
        """The Logic-tab ROW whose RUNNING node produces this panel's signal (None if
        the producing node has no row -- e.g. a node passed via ``running_nodes=`` from
        a notebook).  Lets a plot's Edit show the SOURCE measurement/processor's full
        parameter form (#2)."""
        node = self._producing_node(card)
        if node is None:
            return None
        for row in self.logic_nodes:
            if self._logic_nodes.get(id(row)) is node:
                return row
        return None

    def _apply_source_params(self, row: "LogicNodeRow", values: dict) -> None:
        """Apply edited SOURCE parameters (from a plot Edit's source form, #2) to the
        producing node: live where the node accepts it (a camera's exposure/ROI), else
        rebuild + restart the node with the new values.  Keeps the node's own Logic-tab
        Edit form in sync so the two never diverge.  Returns False if a task lock
        dropped the edit (so the caller does not falsely report success)."""
        if self._task_locked:
            return False
        row.node.values = dict(values)
        editor = self._logic_editors.get(id(row))
        if editor is not None:
            editor.form.seed_values(values)               # keep the node's own Edit consistent
        node = self._logic_nodes.get(id(row))
        if node is None:
            return True                                    # not running -> values remembered for next Start
        try:
            live_keys = set(node.acquisition_parameters())
        except Exception:
            live_keys = set()
        if any(k not in live_keys for k in values) or not live_keys:
            self._start_logic_node(row)                   # rebuild + restart with the new values
        elif hasattr(node, "apply_acquisition_parameters"):
            node.apply_acquisition_parameters(**{k: v for k, v in values.items() if k in live_keys})
        return True

    @staticmethod
    def _node_params(node) -> list:
        """``[(name, current_value)]`` for the editable parameters of the data
        SOURCE behind ``node`` -- a camera's exposure/ROI, or the node's own
        analysis settings.  The node declares them via ``acquisition_parameters()``
        (the source decides what is tunable, not __init__ reflection)."""
        if node is None or not hasattr(node, "acquisition_parameters"):
            return []
        return list(node.acquisition_parameters().items())

    @staticmethod
    def _node_label(node) -> str:
        """Short LAYER name for a node (camera / detect / calibrate / a
        measurement's curve), NOT the Python class name -- the dashboard speaks in
        the architecture's layers, so no class name ever leaks into the UI."""
        label = getattr(node, "display_label", None)
        if label:
            return str(label)
        return str(getattr(node, "prefix", "") or type(node).__name__)

    def _provider_nodes(self) -> list:
        """Every node that can PROVIDE data or signals to a panel -- the RUNNING nodes plus the last
        build of each Logic-tab node (``_last_node``), kept past Stop.  The ONE source of "which nodes
        count": signal resolution (:meth:`_node_for_signal`) and the picker
        (:meth:`_signal_providers`) both read this, so a stopped node's panel keeps rendering its
        lingering state instead of erroring "needs a producing node"."""
        return [*self.running_nodes, *self._last_node.values()]

    def _signal_providers(self) -> dict:
        """``name -> [node labels]`` for every signal a node produces, so the picker shows
        WHICH measurement / processor / camera each signal comes from.

        Covers RUNNING nodes AND the last build of every Logic-tab node (``_last_node``,
        kept past Stop): a finished scan's ``readout_fidelity`` or a stopped camera's
        ``frame`` LINGER in the hub, and the picker must still name their source node rather
        than show a bare signal.  Running nodes are listed first; a name carried by more
        than one node lists every source node (so an ambiguous pick can be flagged)."""
        providers: dict[str, list] = {}
        seen: set[int] = set()
        for node in self._provider_nodes():
            if node is None or id(node) in seen or not hasattr(node, "published_signals"):
                continue
            seen.add(id(node))
            label = self._node_label(node)
            for name in node.published_signals():
                bucket = providers.setdefault(str(name), [])
                if label not in bucket:
                    bucket.append(label)
        # DECLARED (not-yet-started) Logic-tab nodes: list the signals they WILL publish too,
        # tagged by their node title, so a Monitor can be wired to a node's output BEFORE that
        # node is started (#6: connect first, start later).  Skip rows already covered above.
        for row in self.logic_nodes:
            node = self._logic_nodes.get(id(row))
            if node is not None and getattr(node, "running", False):
                continue
            label = str(getattr(row.node, "title", "") or getattr(row.node, "name", "") or row.node.kind)
            keys = list(self._declared_signal_keys(row))
            if row.node.kind == "task":
                spec = self._spec_for_logic(row.node)
                keys.extend(
                    str(output.name)
                    for output in getattr(spec, "declared_outputs", ()) or ()
                )
            for key in keys:
                bucket = providers.setdefault(str(key), [])
                if label not in bucket:
                    bucket.append(label)
        return providers

    def _signal_names(self) -> list[str]:
        """Published datasets plus catalog-declared outputs not yet produced.

        The data plane is the runtime source; the catalog declaration is what
        lets an operator wire a panel before pressing Start.  A FINAL task
        artifact is deliberately not republished through a mutable signal hub,
        but its declared output name remains the stable card-to-task binding
        used by the exact Fit entrance.
        """

        names = {str(name) for name in self._tick_data.names()}
        names.update(self._signal_providers().keys())
        for row in self.logic_nodes:
            spec = self._spec_for_logic(row.node)
            for output in getattr(spec, "declared_outputs", ()) or ():
                names.add(str(output.name))
        return sorted(names)

    def _signal_formats(self) -> dict:
        """``name -> standardized array shape`` for every LIVE hub signal, read straight
        off the most recent published VALUE (``shape_text.describe_shape``) -- AUTO from real
        data, never a hand-typed name->format map that could drift from what a node
        actually emits.  Lets the signal picker show each signal's SHAPE, not just its
        name (e.g. ``occupied  [(35,)]``)."""
        from zlc_data.shape_text import describe_shape
        out: dict[str, str] = {}
        for name in self._signal_names():
            try:
                st = self._signal_structure(name) or {}
                out[str(name)] = describe_shape(self._signal_values(name), points_shape=st.get("points_shape"),
                                                data_shape=st.get("data_shape"), grid_shape=st.get("grid_shape"))
            except Exception:
                continue
        return out

    def _signal_short_names(self) -> dict:
        """``{full hub signal: SHORT name}`` = each PROVIDER node's published signals with that node's
        prefix stripped (``temperature_survival`` -> ``survival``, ``frame`` -> ``frame``).  The picker
        nest binds this so the leaf shows the short name -- the SAME rule the Logic tab uses
        (``strip_node_prefix``), never the verbose SignalSpec axis label.  Reads the SAME
        ``_provider_nodes()`` source as every other signal-resolution site so a stopped node's lingering
        signal keeps its short name instead of showing the verbose full name."""
        out: dict[str, str] = {}
        for node in self._provider_nodes():
            pfx = str(getattr(node, "prefix", "") or "")
            try:
                for full in node.published_signals():
                    out[str(full)] = strip_node_prefix(str(full), pfx)
            except Exception:
                continue
        return out

    def _live_node_formats(self, node) -> list[tuple[str, str, str]]:
        """``[(name, shape, description)]`` for a RUNNING node -- one ROW per output, each
        shape read off a real value via ``shape_text.describe_shape`` (auto, never hand-typed)
        and each description from the node's ``output_specs`` (what the signal MEANS).
        Measurement and processor nodes publish to the data plane under their prefix.
        Task outputs are FINAL artifact declarations and are rendered by the spec-only
        branch in :meth:`_update_row_publishes`, never fabricated as live signals."""
        from zlc_data.shape_text import describe_shape
        specs = {s.name: s for s in node.output_specs()} if hasattr(node, "output_specs") else {}

        def desc(name: str) -> str:
            spec = specs.get(name)
            return spec.description if spec is not None else ""

        rows: list[tuple[str, str, str]] = []
        # published_signals() are HUB names (incl. the node's disambiguating prefix when two
        # nodes would collide).  Show the SHORT natural name (strip the prefix) because the
        # Logic row is already titled by the node.  ``output_specs`` (and so ``desc``) is
        # keyed by the FULL published name: look descriptions up by ``full``.
        pfx = str(getattr(node, "prefix", "") or "")
        for full in sorted(node.published_signals()):
            short = strip_node_prefix(full, pfx)             # ONE rule, shared with the picker nest
            try:
                # SAME schema-driven formatter as the task branch (#12): a hub signal and a task
                # output of the same logical shape render byte-identically -- no drift possible.
                # ZERO-COPY: read the tensor's ``.data`` (a reference -- only ``.shape`` is used) rather
                # than ``hub.latest`` (which .copy()s the whole 2.3 MB frame every tick just to format a
                # shape string, #4-E).
                shape = self._describe_from_schema(self._signal_values(full), self._signal_schema(full))
            except Exception:
                shape = "—"
            rows.append((short, shape, desc(full)))
        return rows

    def _update_row_publishes(self, row: "LogicNodeRow") -> None:
        """Fill a Logic-tab row's "publishes:" legend (ONE signal per line: name, shape,
        meaning).  Shapes are AUTO-EXTRACTED from the real published VALUES
        (``shape_text.describe_shape``) and the meaning from the node's ``output_specs`` --
        never a hand-typed map.  Running node: live shapes off the hub (measurement /
        processor).  A task lists its declared FINAL artifact names at every
        lifecycle state; those names never pretend to be live data-plane signals.
        Stopped data-plane node: the NAMES it will publish (shape ``—`` until it runs)."""
        if row.node.kind == "task":
            spec = self._spec_for_logic(row.node)
            outputs = getattr(spec, "declared_outputs", ()) or ()
            row.set_publishes(
                [(str(output.short or output.name), "FINAL", "") for output in outputs]
            )
            return
        node = self._logic_nodes.get(id(row))
        if node is not None and getattr(node, "running", False):
            row.set_publishes(self._live_node_formats(node))
            return
        # The legend shows the SHORT name (strip the node prefix), exactly like the running path
        # (_live_node_formats); _declared_signal_keys now returns the FULL published names (#prebind).
        pfx = self._declared_node_prefix(row)
        row.set_publishes([(strip_node_prefix(k, pfx), "—", "") for k in self._declared_signal_keys(row)])

    def _declared_node_prefix(self, row: "LogicNodeRow") -> str:
        """The hub-signal PREFIX the row's node publishes under -- so a declared name == the
        published name (a binding made before Start, or restored from a saved layout, re-attaches
        the instant the producer starts, #prebind).

        The prefix is allocated ONCE, at Start, by the shared per-instance rule
        (``_logic_node_prefix``) and then STICKS as the instance's identity: a row that has (ever)
        been built reads its node's own ``prefix`` back -- re-running the collision rule later
        would drift with hub state (a sibling stopped by the device exclusion before it ever
        published leaves no trace, so a recomputation would 'un-collide' a name that IS published).
        Only a never-started row PREDICTS with the same rule.  A task is off the
        data plane and therefore has no prefix."""
        if row.node.kind == "task":
            return ""
        built = self._logic_nodes.get(id(row)) or (getattr(self, "_last_node", {}) or {}).get(id(row))
        if built is not None and hasattr(built, "prefix"):
            return str(built.prefix or "")
        return self._logic_node_prefix(row.node)

    def _declared_signal_keys(self, row: "LogicNodeRow") -> list[str]:
        """The FULL hub names a STOPPED Logic-tab node WILL publish once started -- IDENTICAL to the
        running node's ``published_signals()`` (same node prefix, #prebind), so the picker offers, and
        a Monitor binding stores, the SAME name the node later emits.  That makes a "connect first,
        start later" binding (and a save->load of one) re-attach automatically when the producer
        publishes that exact name.  The bare keys come from the ONE kind ladder
        (:meth:`_node_bare_keys` -- shared with the prefix collision check, so what the picker
        declares is exactly what is checked and published).  A task publishes no live signal;
        its declared FINAL artifact names are added separately for the exact Analysis entry."""
        if row.node.kind == "task":
            return []
        pfx = self._declared_node_prefix(row)              # prepend the node prefix -> == published_signals()
        return [f"{pfx}{k}" for k in self._node_bare_keys(row.node)]

    def _node_for_signal(self, name: str):
        """The producing node for signal ``name``: a RUNNING node first, else the last build of a
        Logic-tab node (``_last_node``, kept past Stop) so a stopped node's lingering signal still
        resolves.  None if none publishes it."""
        for node in self._provider_nodes():
            if node is not None and hasattr(node, "published_signals"):
                try:
                    if name in node.published_signals():
                        return node
                except Exception:
                    continue
        return None

    @staticmethod
    def _schema_structure(schema) -> dict[str, object]:
        """Project one authoritative ``SignalSchema`` into the plot structure mapping."""

        ps = tuple(schema.point_shape)
        ds = tuple(schema.data_shape)
        metadata = dict(schema.metadata)
        grid = tuple(metadata.get("grid_shape", ()))
        if not grid and len(ps) == 2:
            grid = ps
        return {
            "points_shape": ps,
            "data_shape": ds,
            "grid_shape": grid,
            "ring": int(schema.repeat_capacity or 1),
            "metadata": metadata,
        }

    def _describe_from_schema(self, value, schema) -> str:
        """The ONE declared-shape string for any legend/picker row.  Projects a ``SignalSchema``
        through :meth:`_schema_structure` (the single grid rule) and renders it in the canonical
        ``R × P × (data)`` grammar, so a TASK output, a hub signal, a running node and a
        not-yet-published declared signal ALL read identically -- never the raw ``(R×P×data)``
        one-outer-paren spelling ``describe_shape`` falls back to when a caller forgets the schema
        (issue #12: calibration / mot-field task rows).  A canonical block's leading axis is R;
        with no value yet, R is the schema's declared repeat capacity.  No schema -> the raw
        value-only ``describe_shape`` (a scalar result still reads ``scalar``)."""
        from zlc_data.shape_text import contract_shape_label, describe_shape
        if schema is None:
            return describe_shape(value)
        st = self._schema_structure(schema)
        ps, ds, gs = st["points_shape"], st["data_shape"], st["grid_shape"]
        if value is not None:
            return describe_shape(value, points_shape=ps, data_shape=ds, grid_shape=gs)
        return contract_shape_label(int(st["ring"] or 1), ps, ds, gs)

    def _signal_structure(self, name: str):
        """Read the authoritative data-plane ``SignalSchema`` for plotting."""

        name = str(name)
        try:
            schema = self._signal_schema(name)
        except KeyError:
            return None
        result = self._schema_structure(schema)
        metadata = dict(schema.metadata)
        coordinate_names = tuple(metadata.get("coordinate_signals", ()))
        if coordinate_names:
            arrays = []
            for coordinate_name in coordinate_names:
                coordinate = np.asarray(self._signal_values(str(coordinate_name)), dtype=float)
                coordinate_schema = self._signal_schema(str(coordinate_name))
                expected = (coordinate.shape[0], coordinate_schema.point_count, 1)
                if coordinate_schema.data_shape != (1,) or tuple(coordinate.shape) != expected:
                    raise ValueError(
                        f"coordinate signal {coordinate_name!r} must be canonical (R,P,1); "
                        f"got {coordinate.shape}")
                arrays.append(coordinate[-1, :, 0].tolist())
            result["param_names"] = [str(value) for value in
                                     metadata.get("axis_order", coordinate_names)]
            result["points_coords"] = arrays
        return result

    @staticmethod
    def _card_signal_binding(card: "PanelCard") -> str:
        """The only card fields that can change its signal-source legend."""

        return str(card.config.signal or "")

    def _card_signal_binding_changed(self, card: "PanelCard") -> None:
        """Consume one card-local binding delta, not a whole-board snapshot."""

        if card not in self.cards:
            return
        current = self._card_signal_binding(card)
        if self._card_signal_bindings.get(id(card)) == current:
            return
        self._card_signal_bindings[id(card)] = current
        self._signal_topology_changed()

    def _signal_topology_changed(self) -> None:
        """Reconcile legends after an explicit provider/card topology mutation."""

        self._signal_info_dirty = True
        if not self._building:
            self._refresh_signal_info()

    def _refresh_signal_info(self) -> None:
        """Give every panel a legend (shown in its frame TITLE -- the grey strip, the old footer
        was removed) naming which node + layer produces the panel's one bound signal -- e.g.
        ``occupied ← occupancy [processor]``.  It does not list the producing node's whole
        output set, so the title answers exactly "this plot's value comes from which
        measurement/processor".  A signal
        published by more than one running node is flagged ambiguous.  The
        caller is an explicit topology/card-binding mutation; a timer tick with
        no such mutation does not enter this method's provider enumeration."""
        if not self._signal_info_dirty:
            return
        providers = self._signal_providers()
        # GC the hub of ORPHANED signals (#5 unbound): the AUTHORITATIVE invariant -- a hub signal that NO
        # live producer still publishes is stale and must leave, else it piles up as an "(unbound)" picker
        # entry run-after-run.  ``providers`` is the single source of "who publishes what" (running nodes +
        # stopped-but-kept rows via _last_node + declared rows); a hub name absent from it has no owner ->
        # purge.  A STOPPED node's signals are KEPT (its row is still a provider via _last_node -- a finished
        # scan stays plottable).  (A node error is surfaced via instance attrs + the banner, never as a hub
        # signal, so there is no reserved health channel to exempt here.)  This runs only when the provider
        # map / card sources CHANGED (the guard above), which a SWITCH does even with no rebuild: a live
        # ``frames_per_cycle`` 3->1 shrinks a running camera's published set {frame_0,1,2}->{frame_0},
        # dropping frame_1/frame_2 from ``providers`` -> they are caught here (the eager purges in
        # _start/_remove_logic_node cover the rebuild / remove paths synchronously; THIS catches every
        # remaining path, switches included).
        for card in self.cards:
            reads = sorted(self._card_reads(card))
            parts: list[str] = []
            for name in reads:
                src = self._node_for_signal(name)
                if src is not None:
                    layer = str(getattr(src, "layer", ""))
                    tag = self._node_label(src)
                    tag = f"{tag} [{layer}]" if layer and layer != "node" else tag
                    note = "  ⚠ also from another node" if len(providers.get(name, [])) > 1 else ""
                    # The producing node is named to the RIGHT of the arrow, so the signal need
                    # not repeat its prefix -- the ONE strip_node_prefix rule the nested combo uses.
                    short = strip_node_prefix(name, getattr(src, "prefix", ""))
                    parts.append(f"{short} ← {tag}{note}")
                else:
                    parts.append(f"{name} ← (no running source)")
            if not reads:
                parts.append("(no signal set — pick one in Setting: value = <signal>)")
            # one read per line: "<signal> ← <node> [layer]" -- the value's origin only,
            # not the producing node's full output list.
            card.set_signal_info("\n".join(parts))
        self._signal_info_dirty = False

    def _restart_node(self, node, new_params: dict):
        """Apply edited acquisition parameters to the producing node so the
        Monitor re-acquires under them.  This goes through the node's SAFE entry
        ``apply_acquisition_parameters``: while the acquisition loop runs it owns
        the source, so the edit is queued and the loop applies it BETWEEN shots
        (the source re-arms in its owner thread -- a streaming camera picks up the
        new ROI/exposure -- with no GUI-thread stall and no second ``acquire()``
        racing on one camera; an in-place reconfigure of a running stream would be
        ignored, while stopping/starting the thread from here would block the GUI
        and could deadlock).  An idle node is reconfigured and stepped once.
        Returns the node.  Raises if the node has no editable acquisition params."""
        if node is None or not hasattr(node, "apply_acquisition_parameters"):
            raise RuntimeError("this panel's data source exposes no editable acquisition parameters")
        node.apply_acquisition_parameters(**new_params)
        # Edit's Apply makes the source LIVE: if its acquisition loop is not
        # running (a fresh node, or one the user stopped), start it so the Monitor
        # streams under the new params.  start() is idempotent, so a node that is
        # already running just keeps its loop (the edit was queued above).
        if not getattr(node, "running", False):
            self._begin_run(node)
        return node

    def _begin_run(self, node) -> None:
        """Start one ConsoleRunNode and register it as running.

        The node already knows HOW it starts -- the composition root bound that
        when it built the node, because knowing a camera monitor needs a live
        view is the root's business, not the board's.  Submission returns at
        once (the RUN seam owns the worker), so the board stays interactive
        while a camera opens; the tick's ``poll`` turns the pending start into a
        live handle or a reported failure.
        """

        node.start()
        self._sync_fit_analysis_entries()
        if node not in self.running_nodes:
            self.running_nodes.append(node)

    def _stop_run(self, node, *, timeout: float = 2.0) -> bool:
        """Ask a ConsoleRunNode to stop and report whether it reached terminal.

        Cancellation is a request, not a join: the console never blocks the GUI
        thread waiting for hardware to let go.  An un-terminated run keeps its
        registration so the next tick (or the close path) can finish the teardown
        instead of losing track of a run that is still holding a device.
        """

        node.cancel()
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            snapshot = node.poll()
            if snapshot is not None and snapshot.state.terminal:
                if (
                    snapshot.state.name == "SUCCEEDED"
                    and not node.final_result_resolved
                ):
                    if time.monotonic() >= deadline:
                        return False
                    time.sleep(0.01)
                    continue
                break
            if node.last_error is not None:
                break
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        if node in self.running_nodes:
            self.running_nodes.remove(node)
            # A stopped Logic row remains represented by ``_last_node`` and
            # therefore keeps the same provider legend.  Row-less injected
            # nodes have no retained provider and are a real topology change.
            if not any(value is node for value in self._last_node.values()):
                self._signal_topology_changed()
        return True

    def _edit_card(self, card: "PanelCard") -> None:
        """Open (or focus) this PLOT panel's OWN closable Edit tab (a PanelEditor); a
        running task locks every other action, so this no-ops then.

        Opens even BEFORE the panel has data -- the panel's plot params /
        acquisition / fit / limits are editable straight away.  The snapshot
        section just shows "waiting for data" until a plot exists (the data is
        produced by a Logic-tab node, not by this Edit)."""
        if card is None or self._task_locked:
            return
        existing = self._panel_editors.get(id(card))
        if existing is not None:
            self.tabs.setCurrentWidget(existing)
            existing.rebuild()
            return
        editor = PanelEditor(card, self)
        self._panel_editors[id(card)] = editor
        title = (card.config.title or PANEL_KINDS[card.config.kind]).strip() or "panel"
        self.tabs.add_closable_tab(editor, title)

    def _close_panel_editor(self, card: "PanelCard") -> None:
        """Close a card's Edit tab if open (called when the card is removed)."""
        editor = self._panel_editors.pop(id(card), None)
        if editor is None:
            return
        index = self.tabs.indexOf(editor)
        if index >= 0:
            self.tabs.removeTab(index)
        editor.teardown()
        editor.setParent(None)
        editor.deleteLater()

    def _on_editor_tab_closed(self, widget) -> None:
        """X on a PanelEditor / LogicNodeEditor tab: tear it down + drop it from the
        registry.  The permanent Monitor / Logic tabs carry no X, so they never
        arrive here.  A logic node's Edit only closes the TAB -- the node keeps
        running and stays in the Logic list (reopen its Edit from its row)."""
        index = self.tabs.indexOf(widget)
        if index >= 0:
            self.tabs.removeTab(index)
        for key, editor in list(self._panel_editors.items()):
            if editor is widget:
                del self._panel_editors[key]
        for key, editor in list(self._logic_editors.items()):
            if editor is widget:
                del self._logic_editors[key]
        if hasattr(widget, "teardown"):
            widget.teardown()
        widget.setParent(None)
        widget.deleteLater()

    def _on_tab_changed(self, _index: int) -> None:
        # ONE rule: when a tab becomes visible, give whatever's inside a chance to refresh its
        # dynamic content -- any widget that implements ``refresh_on_show()`` runs it.  No
        # special-casing per tab type (PanelEditor / LogicNodeEditor / MeasurementPanel all
        # honour the same hook), so a brand new editor automatically participates by
        # implementing ``refresh_on_show`` -- nothing here changes.  This kills the bug where
        # a processor's source dropdown stayed empty after switching tabs.
        widget = self.tabs.currentWidget()
        hook = getattr(widget, "refresh_on_show", None)
        if callable(hook):
            try:
                hook()
            except Exception:
                pass

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
        # Add (appended last) lands in the next bottom slot, never a middle hole (#2).
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
        stop, #2).  No viewport yet (pre-show) -> the headless two-wide fallback."""
        vw = self.scroll.viewport().width() if hasattr(self, "scroll") else 0
        return vw if vw else _board_width(configs)

    # ------------------------------------------------------------------ actions
    def _add_panel(self) -> None:
        """Header "Add Panel": add either a BLANK plot panel (a plot kind) or a
        STOPPED logic node (camera / measurement / processor / task)."""
        if self._task_locked:
            return
        data = self.kind_combo.currentData()
        # A node LAYER -> a STOPPED logic node on the Logic tab (camera /
        # measurement / processor / task).  It publishes nothing until Started.
        if isinstance(data, tuple) and len(data) == 2 and data[0] in LOGIC_KINDS:
            kind, name = data
            # Every layer's dropdown name IS its catalog entry's title -- ONE source for the
            # dropdown label, the Logic row title and the spec lookup.
            title = str(name)
            self._add_logic_node(LogicNodeConfig(kind=kind, name=name, title=title), focus=True)
            return
        # Otherwise a PLOT kind -> a BLANK pure-view panel on the Monitor board
        # (decoupled: it shows nothing until a signal is picked in its Setting).
        kind = data or "1d"
        # Every panel gets a unique "<kind> #N" title (G1) so two of the same kind never share
        # one name in the card header / Edit tab / frame title.
        title = indexed_unique_name(PANEL_KINDS[str(kind)], {c.config.title for c in self.cards})
        config = PanelConfig(kind=str(kind), title=title, row=GAP, col=GAP, size="1x2")
        # APPEND the new card LAST in order (``_attach_card`` adds it to the end of ``self.cards``);
        # the order-driven :func:`pack` in ``_arrange`` then lands it in the next free BOTTOM slot,
        # never a middle hole (#2).  No pixel seed needed -- pack recomputes every card's position from
        # the order alone.
        card = self._new_panel_card(config)
        self._attach_card(card)
        self._arrange()
        self._mark_dirty()

    def _remove_panel(
        self,
        card: PanelCard,
    ) -> bool:
        # User removal is blocked while a task owns the console.
        if self._task_locked:
            return False
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
            card.shutdown()
            phases.add("shutdown")
        if "removed" not in phases:
            self.cards.remove(card)
            self._card_signal_bindings.pop(id(card), None)
            self._signal_topology_changed()
            phases.add("removed")
        if "qt_detached" not in phases:
            card.setParent(None)
            card.deleteLater()
            phases.add("qt_detached")
        if "arranged" not in phases:
            self._arrange()
            self._recompute_tick_interval()    # removing the fastest panel can slow the base
            self._mark_dirty()
            phases.add("arranged")
        phases.add("detached")
        self._panel_teardown_phases.pop(id(card), None)
        self._sync_fit_analysis_entries()
        return True

    @property
    def experiment(self):
        """The session behind the catalog view, or None for a catalog-less window.

        The skeleton reaches the domain ONLY through the catalog seam, so this is
        the single accessor every panel/editor asks -- no second session field.
        """

        return self._catalog.experiment if self._catalog is not None else None

    # ====================================================================== logic nodes
    def _catalog_specs(self, kind: str) -> tuple:
        """The catalog entries of ONE layer, or empty without a catalog view."""

        if self._catalog is None:
            return ()
        return tuple(self._catalog.specs(kind))

    def _spec_for_logic(self, node: LogicNodeConfig):
        """The catalog entry a logic node builds from.

        It carries the node's parameter form (the Edit auto-form renders it), its
        declared outputs and the typed-request builder the RUN seam starts from.
        One lookup for every layer: the entry's own title is its identity.
        """

        if self._catalog is None:
            return None
        spec = self._catalog.spec_named(node.name)
        return spec if spec is not None and spec.kind == node.kind else None

    def _unique_logic_title(self, title: str) -> str:
        """Make a logic-node title ``"<base> #N"`` and UNIQUE among the existing Logic rows
        (G1) -- so two same-kind nodes are told apart in the Logic
        rows AND get distinct per-instance signal prefixes (see _logic_node_prefix).  Re-indexes
        a loaded title's root, so an already-clean saved layout round-trips."""
        return indexed_unique_name(title, {str(r.node.title) for r in self.logic_nodes})

    def _node_bare_keys(self, node: LogicNodeConfig) -> list[str]:
        """The SHORT (un-prefixed) output names a logic node emits -- the ONE derivation shared by
        the prefix collision check (:meth:`_logic_node_prefix`) and the declared picker names
        (:meth:`_declared_signal_keys`), so what is checked for collision is exactly what will be
        published.  Processor: the spec's ``result_keys``; measurement: its x/y curve keys (or, when
        the spec exposes a ``declared_keys`` deriver, the REAL published names for its form values --
        pulse-scan overrides ``x_key="param"`` with the semantic scan coordinate, so the deriver keeps
        declared == published); camera: ``frame_i`` per emCCD event (``frames_per_cycle``, the same
        helper ``CameraMeasurement.published_signals`` uses so they can never drift)."""
        spec = self._spec_for_logic(node)
        declared = (getattr(spec, "metadata", None) or {}).get("declared_keys")
        if callable(declared):
            try:
                return [str(k) for k in declared(node.values or {})]
            except Exception:
                pass                                       # fall back to the static spec keys below
        keys = [str(k) for k in (getattr(spec, "result_keys", ()) or [])]
        if not keys and node.kind == "measurement":
            keys = [k for k in (getattr(spec, "x_key", ""), getattr(spec, "y_key", "")) if k]
        if not keys and node.kind == "camera":
            from zlc_data.shape_text import camera_frame_keys
            keys = camera_frame_keys((node.values or {}).get("frames_per_cycle", 1))
        return keys

    def _logic_node_base_prefix(self, node: LogicNodeConfig) -> str:
        """The KIND-semantic default prefix a logic node publishes under when nothing collides.
        Measurement: ``f"{spec.key}_"`` so every signal self-describes its quantity
        (``temperature_t_off`` -- one name, derived, #H3r).  Camera / processor: ``""`` -- the short
        natural names (``frame_0`` / ``occupied``); the PRODUCER is shown by the signal-flow legend
        and never baked into the signal (which camera INSTANCE, let alone which DEVICE, is not the
        signal's business -- a consumer binds an instance's output, not a piece of hardware)."""
        if node.kind == "measurement":
            spec = self._spec_for_logic(node)
            return f"{spec.key}_" if spec is not None else ""
        return ""

    def _logic_node_prefix(self, node: LogicNodeConfig) -> str:
        """The hub-signal prefix for a logic node -- the ONE rule for EVERY kind (camera /
        measurement / processor): the kind-semantic base prefix (:meth:`_logic_node_base_prefix`)
        by default, upgraded to a per-INSTANCE slug (from the row's unique title, ``occupancy_2_``)
        only when the base-prefixed names would COLLIDE with another node's signals.  A signal's
        namespace is the logic-node INSTANCE that produces it: two rows of the same kind can never
        overwrite each other, and no device / backend identity ever leaks into a signal name."""
        keys = self._node_bare_keys(node)
        base = self._logic_node_base_prefix(node)
        # Collision is checked against EVERY signal live in the hub -- running nodes AND a STOPPED
        # node's lingering signals -- not just running_nodes (#2): otherwise a new same-kind node added
        # after an earlier one STOPPED (its signals deliberately linger) would see no running collision,
        # take the empty prefix, and CLOBBER the stopped node's lingering data on the hub.
        running: set[str] = set(self._tick_data.versions())
        for n in self.running_nodes:
            try:
                running.update(str(s) for s in n.published_signals())  # just-started, maybe not in hub yet
            except Exception:
                pass
        # A RESTART reclaims its OWN lingering signals -- they are not a collision (#issue-1): without
        # this, restarting a node sees its own STOPPED outputs still in the hub, takes a fresh
        # numbered prefix, and every panel bound to the original names goes UNBOUND.  Its
        # prior built node survives Stop in _last_node tagged with instance_label == this node's title.
        # (The same rule is what lets a row SWITCH its camera device and restart: the row keeps its
        # signal names, the new device's frames simply flow under them -- panels follow seamlessly.)
        for prev in (getattr(self, "_last_node", {}) or {}).values():
            if prev is not None and str(getattr(prev, "instance_label", "")) == str(node.title):
                try:
                    running.difference_update(str(s) for s in prev.published_signals())
                except Exception:
                    pass
        if not keys or not any((base + key) in running for key in keys):
            return base                              # no collision (incl. own restart) -> the base names
        from zlc_data.shape_text import measurement_slug
        slug = measurement_slug(node.title or node.name) or str(node.kind) or "node"
        prefix, k = f"{slug}_", 2
        while prefix in {getattr(n, "prefix", "") for n in self.running_nodes} \
                or any((prefix + key) in running for key in keys):
            prefix = f"{slug}_{k}_"
            k += 1
        return prefix

    def _attach_logic_node(self, node: LogicNodeConfig, *, focus: bool = False) -> "LogicNodeRow | None":
        """Add a STOPPED logic-node row to the Logic tab (no node built yet).  A NO-OP when embedded:
        the figure viewer has no Logic tab (``logic_layout`` is None), so there is nowhere to host the
        row -- the console is a pure view host, its acquisition is whatever produced the loaded figure."""
        if self.logic_layout is None:                        # embedded: no Logic tab to attach to
            return None
        node.title = self._unique_logic_title(node.title)   # distinct rows + distinct signal prefixes
        row = LogicNodeRow(node)
        row.edit_requested.connect(self._edit_logic_node)
        row.remove_requested.connect(self._remove_logic_node)
        row.start_requested.connect(self._start_logic_node)   # Start/Stop act on the row itself (#5)
        row.stop_requested.connect(self._stop_logic_node)
        # insert ABOVE the trailing stretch (the hint + stretch are the last 2 items)
        self.logic_layout.insertWidget(self.logic_layout.count() - 1, row)
        self.logic_nodes.append(row)
        self._logic_nodes[id(row)] = None
        self._signal_topology_changed()
        self.logic_hint.hide()
        self._update_row_publishes(row)                       # show its outputs + shapes up front
        self._sync_fit_analysis_entries()
        if focus and hasattr(self.tabs, "setCurrentWidget"):
            self._edit_logic_node(row)
        return row

    def _add_logic_node(self, node: LogicNodeConfig, *, focus: bool = True) -> "LogicNodeRow":
        row = self._attach_logic_node(node, focus=focus)
        self._mark_dirty()
        return row

    # --------------------------------------------------------- FINAL analysis
    def _fit_source_for_card(self, card: "PanelCard"):
        """Resolve one card binding to exactly one FINAL scan artifact.

        A card stores only declared dataset names, so two task rows declaring
        the same name are ambiguous and must not be guessed.  With exactly one
        matching row, the successful Run result is the authority; no display
        snapshot, selector state or parallel fit draft is consulted.
        """

        if self._fit_window_factory is None:
            return None
        reads = self._card_reads(card)
        if not reads:
            return None
        matched_nodes = []
        for row in self.logic_nodes:
            spec = self._spec_for_logic(row.node)
            if spec is None or str(getattr(spec, "kind", "")) != "task":
                continue
            outputs = {
                str(output.name)
                for output in getattr(spec, "declared_outputs", ()) or ()
            }
            if reads.isdisjoint(outputs):
                continue
            matched_nodes.append(self._last_node.get(id(row)))
        if len(matched_nodes) != 1:
            return None
        node = matched_nodes[0]
        if node is None:
            return None
        from zlc_neutral_atom.scan import ScanArtifactRef

        result = node.final_result
        return result if isinstance(result, ScanArtifactRef) else None

    def _sync_fit_analysis_entries(self, *_args) -> None:
        """Project current exact-source availability onto every card button."""

        for card in tuple(self.cards):
            card.set_fit_analysis_available(
                self._fit_source_for_card(card) is not None
            )

    def _open_panel_fit(self, card: "PanelCard") -> None:
        """Open or focus the shared Fit host for the card's current FINAL ref."""

        source = self._fit_source_for_card(card)
        if source is None:
            card.set_fit_analysis_available(False)
            card.set_status(
                "Fit requires one unambiguous current FINAL scan artifact",
                error=True,
            )
            return
        current = self._analysis_window
        if (
            current is not None
            and self._analysis_source == source
            and not bool(getattr(current, "closed", False))
        ):
            restore = getattr(current, "restore_window", None)
            if callable(restore):
                restore()
            else:
                current.show()
                current.raise_()
                current.activateWindow()
            return
        factory = self._fit_window_factory
        if factory is None:
            raise RuntimeError("this TaskConsole has no Fit window capability")
        window = factory(source)
        self._analysis_window = window
        self._analysis_source = source
        card.set_status("opened exact Fit Analysis for the FINAL scan", error=False)

    def _edit_logic_node(self, row: "LogicNodeRow") -> None:
        """Open (or focus) a logic node's OWN closable Edit tab (param form + Start/Stop)."""
        if self._task_locked:
            return
        existing = self._logic_editors.get(id(row))
        if existing is not None:
            self.tabs.setCurrentWidget(existing)
            return
        spec = self._spec_for_logic(row.node)
        editor = LogicNodeEditor(row, self, spec)
        # reflect the live run state on the form (a Started node reopened keeps Stop enabled)
        node = self._logic_nodes.get(id(row))
        editor.set_running(bool(getattr(node, "running", False)))
        self._logic_editors[id(row)] = editor
        self.tabs.add_closable_tab(editor, row.node.title)

    def _start_logic_node(self, row: "LogicNodeRow") -> None:
        """Build the node FROM the node's current param-form values with display
        SUPPRESSED (it only publishes to the hub -- never opens a matplotlib plot),
        register it in ``self.running_nodes``, and ``node.start()`` (data-paced).  Sets
        the node's status dot green; on build/run error -> red + the error on the status
        line.  Reuses the SAME node-build paths the real readout / notebook use."""
        if self._task_locked:
            return                                 # a task owns the console -- no other Start
        editor = self._logic_editors.get(id(row))
        try:
            values = editor.collect_values() if editor is not None else dict(row.node.values)
        except Exception:
            values = dict(row.node.values)
        if editor is not None:
            # The form's collect_values returns only its DECLARED widgets, so a stored value the form
            # does not own (a drag-set ROI ``selection`` is DATA, not a form field) would be dropped on
            # an Edit-triggered restart.  Preserve those: form values WIN for keys it owns, stored fills
            # the rest -- the same "don't lose a param the UI can't show" rule the retarget seam relies on.
            values = {**dict(row.node.values or {}), **values}
        # BUILD-VALIDATE-COMMIT (#A2): build the NEW node FIRST -- a bad param edit must never
        # kill the run that is already going (the old order stopped the running node, then failed
        # the build, leaving nothing running); and nothing is registered until start() succeeds,
        # so a failed start can never leave a half-started ghost in running_nodes / the hub.
        try:
            node = self._build_logic_node(row.node, values)
        except Exception as exc:
            row.set_state("error", status=f"build failed: {exc}")
            if editor is not None:
                editor.set_status(f"build failed: {exc}", error=True)
            return
        row.node.values = dict(values)            # remember for the next Edit reopen + save
        # Label the built node with its ROW TITLE so its provider label MATCHES the declared row's
        # (the DEFAULT camera has prefix="" -> display_label would otherwise fall back to
        # node_label="camera", which differs from the row title "Camera (live frames)", listing
        # `frame` under TWO sources = the "two cameras" bug).  One label per node => one entry in
        # the signal picker (#H3n).
        node.instance_label = str(getattr(row.node, "title", "") or getattr(node, "instance_label", ""))
        # The build is good -- NOW stand the previous run of THIS node down (never pile up), and
        # apply the device-OCCUPANCY mutual exclusion over ALL running nodes (#A3): each node
        # declares the hardware INSTANCES it drives (``occupied_devices``, from its own
        # ``_occupies`` attribute names), and two nodes conflict iff those sets intersect by
        # identity.  Only the conflicting nodes stop; everyone on disjoint hardware keeps
        # running -- the monitor camera's live view stays up while the main camera's
        # calibration or measurement starts.  Declared on the NODE, so a notebook-injected
        # ``running_nodes=`` node (which has no row) obeys the SAME rule as a GUI row; a
        # reactive processor occupies nothing and is never stopped.
        if not self._stop_logic_node(row, _silent=True):
            row.set_state("running", status="restart blocked: previous owner still active")
            if editor is not None:
                editor.set_running(True)
                editor.set_status(
                    "restart blocked: previous owner thread did not terminate", error=True
                )
            return
        if not self._timer.isActive():
            self._timer.start()
        try:
            # A restart of THIS logical row keeps its signal names.  Hand the stopped instance's
            # schema-ownership proof to the replacement before its worker can publish: if switching
            # devices/ROI changes data_shape, the replacement's first frame then takes the normal
            # explicit schema-version path (which drops incompatible history) instead of looking like
            # an unrelated producer trying to clobber the lingering signal.  The node base validates
            # same hub + stopped owner + exact current definition; the console never relaxes shape
            # compatibility and never reaches into the hub's schema internals.
            self._begin_run(node)
        except Exception as exc:
            row.set_state("error", status=f"start failed: {exc}")
            if editor is not None:
                editor.set_running(False)
                editor.set_status(f"start failed: {exc}", error=True)
            return
        # COMMIT: the node is genuinely running -- only now does it enter the registries, and only
        # now are the previous build's orphan signals unlinked (a failed start leaves the old
        # signals in the hub untouched, exactly like a plain Stop, so nothing is lost).
        # #5: unlink any signal the PREVIOUS build published that this new build no longer does AND no
        # other running node owns -> a switched/rebuilt node leaves NO orphan "(unbound)" signal behind.
        self._logic_nodes[id(row)] = node
        self._last_node[id(row)] = node           # survives Stop, for signal-source labelling
        self._signal_topology_changed()
        self._sync_fit_analysis_entries()         # a new generation revoked the previous FINAL ref
        row.set_state("running", status="running")
        self._update_row_publishes(row)            # now show the LIVE node's published shapes
        if editor is not None:
            editor.set_running(True)
            editor.set_status("running", error=False)
        self.status_dot.set_color(GREEN)
        self._mark_dirty()
        # A TASK (one-shot orchestration) takes over the console while it runs.
        # Status is honest lifecycle text; live plots appear only when a task
        # exposes a formal monitor dataset through the ordinary data plane.
        if self._declared_layer(node) == "task":
            self._set_task_running(row)

    @staticmethod
    def _declared_layer(node) -> str:
        """Which catalog layer a run node belongs to -- read off its spec.

        The layer is a CATALOG fact (a definition's group), not something to ask a
        live object about: a task locks the console whether or not its run has
        reached the point of admitting so.
        """

        spec = getattr(node, "spec", None)
        return str(getattr(spec, "kind", "") or "")

    def _set_task_running(self, row: "LogicNodeRow") -> None:
        """Engage task-run mode without manufacturing a display dataset.

        The Run lifecycle and its FINAL artifact are real; a mid-run plot exists
        only when a formal monitor producer has been admitted to the ordinary
        data plane.  Until then the persistent status strip is the complete,
        truthful UI for a running task.
        """

        self._running_task_row = row
        self._update_task_status_text()
        self._apply_task_lock(True)

    def _update_task_status_text(self) -> None:
        """Assemble the running task's honest lifecycle status line."""

        row = self._running_task_row
        if row is None:
            return
        self._task_status_text = f"⏳  Task running: {row.node.title}"

    def _apply_task_lock(self, locked: bool) -> None:
        """Disable / re-enable every mutating control while a task runs.  The strip's Stop
        action stays enabled (it lives outside the locked groups); the strip itself is
        PERSISTENT -- only its content/action flip, so the layout never jumps."""
        self._task_locked = bool(locked)
        if not locked:
            self._task_status_text = None
        self.status_strip.set_action_visible(bool(locked))
        for widget in self._lockable_header:
            widget.setEnabled(not locked)
        self._update_summary()                       # flip the strip's line NOW, not next tick

    def _stop_running_task(self) -> None:
        """Banner 'Stop task' button: cooperatively stop the running task."""
        row = self._running_task_row
        if row is not None:
            self._stop_logic_node(row)

    def _clear_task_running(self) -> None:
        """Leave task-run mode after either completion or an explicit Stop."""

        self._running_task_row = None
        self._apply_task_lock(False)

    def _build_logic_node(self, node: LogicNodeConfig, values: dict):
        """Freeze this row into a ConsoleRunNode -- the RUN seam's unit of work.

        The row names a catalog entry; the CATALOG seam's spec turns the form
        values into that definition's own typed request, and the composition
        root's ``run_factory`` wraps it with the prepare/start closures for this
        session.  The console itself constructs nothing domain-shaped: it holds a
        title, a form, and whatever the factory hands back.
        """

        spec = self._spec_for_logic(node)
        if spec is None:
            raise RuntimeError(f"no catalog entry named {node.name!r} for a {node.kind} node")
        if self._run_factory is None:
            raise RuntimeError(
                "this console was opened without a runtime: Start needs the "
                "composition root's run factory (open it through app.open_task_console)"
            )
        return self._run_factory(spec, dict(values))

    @property
    def render_worker_idle(self) -> bool:
        with self._render_lock:
            return (
                self._render_future is None
                and self._render_completion is None
                and not self._render_pending
            )

    def _request_card_render(self, card: PanelCard, *, force: bool = False) -> bool:
        """Re-render one card from its already accepted immutable data front.

        This entry is used by Setting/selector view commits.  Advancing the
        data plane belongs only to the base tick or explicit full Refresh; a
        mouse/key event cannot materialize producer data as a side effect.
        """

        if self._render_closing or card not in self.cards:
            return False
        request = card.freeze_current_view_request(force=bool(force))
        if request is None:
            return False
        self._enqueue_render_requests((request,))
        return True

    def _enqueue_render_batch(self, batch, snapshot, *, force: bool = False) -> None:
        requests = []
        for card, key in batch:
            request = card.freeze_render_request(snapshot, key, force=force)
            if request is not None:
                requests.append(request)
        self._enqueue_render_requests(tuple(requests))

    def _enqueue_render_requests(
        self,
        requests: tuple[_PanelRenderRequest, ...],
    ) -> None:
        if not requests or self._render_closing:
            return
        with self._render_lock:
            if self._render_future is not None or self._render_completion is not None:
                for request in requests:
                    self._render_pending[request.panel_id] = request
                return
        self._start_render_batch(requests, ())

    def _start_render_batch(
        self,
        requests: tuple[_PanelRenderRequest, ...],
        reset_panel_ids: tuple[str, ...],
    ) -> None:
        if self._render_closing:
            return
        future = self._render_pool.submit(
            self._compose_render_requests,
            requests,
            reset_panel_ids,
        )
        with self._render_lock:
            if self._render_future is not None:
                raise RuntimeError("TaskConsole render lane admitted overlapping batches")
            self._render_future = future
        future.add_done_callback(self._render_batch_finished)

    def _compose_render_requests(
        self,
        requests: tuple[_PanelRenderRequest, ...],
        reset_panel_ids: tuple[str, ...],
    ):
        """Worker-only PanelComposer/Agg owner."""

        from zlc_frontend.panel_render import PanelComposer, PanelRenderError

        for panel_id in reset_panel_ids:
            owned = self._render_worker_composers.pop(panel_id, None)
            if owned is not None:
                owned[1].close()
        results = []
        for request in requests:
            owned = self._render_worker_composers.get(request.panel_id)
            if owned is None or owned[0] != request.source_key:
                if owned is not None:
                    owned[1].close()
                composer = PanelComposer(
                    request.panel_id,
                    intent=request.intent,
                    size=request.size,
                    label=request.label,
                    view=request.view,
                )
                self._render_worker_composers[request.panel_id] = (
                    request.source_key,
                    composer,
                )
            else:
                composer = owned[1]
            try:
                frame = composer.compose(
                    request.value.snapshot,
                    display=request.display,
                    provenance=request.provenance,
                )
                document = composer.document_for(
                    request.value.snapshot.block.schema
                )
            except PanelRenderError as error:
                results.append((request, None, None, str(error)))
            except BaseException as error:
                # Never retain the raw exception/traceback: it may own the
                # entire frozen dataset and Agg graph.
                results.append(
                    (
                        request,
                        None,
                        None,
                        f"{type(error).__name__}: {error}",
                    )
                )
            else:
                results.append((request, frame, document, None))
        return tuple(results)

    def _render_batch_finished(self, future: Future) -> None:
        try:
            completion = future.result()
        except BaseException as error:
            completion = f"{type(error).__name__}: {error}"
        with self._render_lock:
            if self._render_closing:
                return
            self._render_completion = completion
        self._render_wake.request_owner_wake()

    def _render_owner_cycle(self) -> None:
        """Qt-only accept/present point, then release the next latest batch."""

        if self._render_closing:
            return
        with self._render_lock:
            completion = self._render_completion
            if completion is None:
                return
            self._render_completion = None
            self._render_future = None
        reset = set()
        to_present = []
        if isinstance(completion, str):
            self.status_strip.show_message(
                f"Render failed: {completion}", severity="error"
            )
        else:
            by_id = {card.panel_id: card for card in self.cards}
            for request, frame, document, error in completion:
                card = by_id.get(request.panel_id)
                if card is None or not card.accept_render_result(
                    request,
                    frame=frame,
                    document=document,
                    error=error,
                ):
                    reset.add(request.panel_id)
                    continue
                if error is None:
                    to_present.append(card)
            # Every accepted card swaps only after the whole worker batch has
            # been examined, so Qt never composes and a batch cannot be half
            # accepted because one sibling raised.
            for card in to_present:
                card.present()
        with self._render_lock:
            reset.update(self._render_reset_pending)
            self._render_reset_pending.clear()
            pending = tuple(self._render_pending.values())
            self._render_pending.clear()
        if pending or reset:
            self._start_render_batch(pending, tuple(sorted(reset)))

    def _forget_card_render(self, card: PanelCard) -> None:
        """Revoke queued results without waiting for immutable worker work."""

        card._render_request_revision += 1
        card._requested_signature = None
        start_reset = False
        with self._render_lock:
            self._render_pending.pop(card.panel_id, None)
            if self._render_future is None and self._render_completion is None:
                start_reset = True
            else:
                self._render_reset_pending.add(card.panel_id)
        if start_reset:
            # No completion will arrive to drain ``_render_reset_pending``.
            # Queue the disposal directly on the same worker that owns Agg.
            self._start_render_batch((), (card.panel_id,))

    def _shutdown_render_lane(self) -> None:
        if self._render_closing:
            return
        self._render_closing = True
        self._render_wake.detach()
        with self._render_lock:
            self._render_pending.clear()
            self._render_completion = None

        def release_worker_state() -> None:
            for _key, composer in tuple(self._render_worker_composers.values()):
                composer.close()
            self._render_worker_composers.clear()

        # The executor is serial: cleanup runs after an already-started compose
        # on the same owner thread.  Qt does not block while that finite work
        # drains, and the closing gate refuses every late completion.
        self._render_pool.submit(release_worker_state)
        self._render_pool.shutdown(wait=False)

    def _stop_logic_node(
        self,
        row: "LogicNodeRow",
        *,
        _silent: bool = False,
        timeout: float = 2.0,
    ) -> bool:
        """Cancel a logic row's Run and grey its dot."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        node = self._logic_nodes.get(id(row))
        if node is not None:
            if not self._stop_run(node, timeout=max(0.0, deadline - time.monotonic())):
                editor = self._logic_editors.get(id(row))
                if not _silent:
                    row.set_state("running", status="stop pending: the run has not gone terminal")
                    if editor is not None:
                        editor.set_running(True)
                        editor.set_status(
                            "stop pending: the run has not gone terminal", error=True
                        )
                return False
        if row is self._running_task_row:
            self._clear_task_running()
        self._logic_nodes[id(row)] = None
        editor = self._logic_editors.get(id(row))
        if not _silent:
            row.set_state("stopped", status="stopped")
            self._update_row_publishes(row)        # back to the spec-declared outputs
            if editor is not None:
                editor.set_running(False)
                editor.set_status("stopped", error=False)
        return True

    def _remove_logic_node(self, row: "LogicNodeRow", *, _rebuild: bool = True) -> bool:
        """Stop + remove a logic node (its node is stopped, its row + Edit drop)."""
        # a running task locks the console: the per-row Remove button must no-op too
        # (every other mutating entry guards on this).  Internal teardown (load_state)
        # passes _rebuild=False and is never reached while locked.
        if self._task_locked and _rebuild:
            return False
        # Capture this node's published signals so REMOVE can PURGE them from the hub -- a removed
        # node's signals are stale and must leave, else they pile up run-after-run as "多余 signal" in
        # every picker (#2).  STOPPING keeps them (a finished scan stays plottable / a panel can be
        # wired before the next run); only REMOVING purges.  Use ``_last_node`` (the last built node,
        # retained THROUGH stop) not ``_logic_nodes`` (None'd on stop): the common flow is STOP-then-
        # REMOVE, where the live ref is already gone but the lingering hub signals are precisely the
        # ones to purge.
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
            editor.setParent(None)
            editor.deleteLater()
        self._logic_nodes.pop(id(row), None)
        self._last_node.pop(id(row), None)
        if row in self.logic_nodes:
            self.logic_nodes.remove(row)
        self._signal_topology_changed()
        self.logic_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        if not self.logic_nodes:
            self.logic_hint.show()
        self._sync_fit_analysis_entries()
        if _rebuild:
            self._mark_dirty()
        return True

    def _poll_logic_nodes(self) -> None:
        """Each tick: reflect every running node's Run state on its row + Edit.

        ``poll`` is what turns a submitted start into a live handle, so this is
        also where a start that failed on the worker surfaces -- the GUI thread
        learns of it by draining, never by waiting.
        """

        for row in self.logic_nodes:
            node = self._logic_nodes.get(id(row))
            if node is None:
                continue
            editor = self._logic_editors.get(id(row))
            snapshot = node.poll()
            error = node.last_error
            if error:
                row.set_state("error", status=f"error: {error}")
                if editor is not None:
                    editor.set_running(False)
                    editor.set_status(f"error: {error}", error=True)
                self._stop_logic_node(row, _silent=True, timeout=0.0)
                continue
            if snapshot is None:
                # Submitted, not yet acknowledged: the run exists as a request only.
                row.set_state("running", status="starting")
                if editor is not None:
                    editor.set_running(True)
                    editor.set_status("starting", error=False)
                continue
            state = snapshot.state.name
            if state == "CANCELLING":
                row.set_state("running", status="stop pending: cleanup/safety not complete")
                if editor is not None:
                    editor.set_running(True)
                    editor.set_status("stop pending: cleanup/safety not complete", error=True)
                continue
            if snapshot.state.terminal:
                if state == "FAILED":
                    message = snapshot.primary_error or "run failed"
                    row.set_state("error", status=f"error: {message}")
                    if editor is not None:
                        editor.set_running(False)
                        editor.set_status(f"error: {message}", error=True)
                elif state == "SUCCEEDED":
                    if not node.final_result_resolved:
                        row.set_state("running", status="finishing")
                        if editor is not None:
                            editor.set_running(True)
                            editor.set_status("finishing", error=False)
                        continue
                    row.set_state("stopped", status="done")
                    if editor is not None:
                        editor.set_running(False)
                        editor.set_status("done", error=False)
                else:
                    row.set_state("stopped", status="stopped")
                    if editor is not None:
                        editor.set_running(False)
                        editor.set_status("stopped", error=False)
                # A run that ended on its own releases the console exactly like Stop:
                # one teardown path, so a finished task can never leave the lock on.
                self._stop_logic_node(row, _silent=True, timeout=0.0)
                continue
            row.set_state("running", status="running")
            if editor is not None:
                editor.set_running(True)
                editor.set_status("running", error=False)
        self._sync_fit_analysis_entries()

    def _mark_dirty(self, *_args) -> None:
        if self._building:
            return
        if self.save_button is not None:            # embedded: no layout Save button to flag dirty
            self.save_button.set_dirty(True, dirty_color=YELLOW)
        self._update_summary()

    # ------------------------------------------------------------------ refresh
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
        if not self._paused:
            self._tick()

    def _toggle_selectors(self, on: bool) -> None:
        """Header "Selectors" switch: arm (ON) or park (OFF) the selector layer of EVERY dashboard
        panel in place (``PanelCard.set_selectors_enabled`` -> ``BaseLivePlot.set_selectors_active``)
        -- a pure display gate, no rebuild, no effect on acquisition or the Edit tabs."""
        for card in self.cards:
            card.set_selectors_enabled(bool(on))

    def _open_device_viewer(self) -> None:
        """Header "Devices" button -- currently disabled, and this says why in the status line.

        The viewer used to open through a session facade (``device_viewer()`` ->
        ``show_device_viewer``); neither that facade nor this console's ``session``
        attribute exists on the current data plane, so the old body resolved to
        nothing and returned silently.  A button that quietly does nothing is
        read as a broken window, so the button is disabled at build time and
        this states the position if it is ever reached programmatically."""

        self.status_strip.show_message(
            "The read-only device viewer has not been rebuilt on this data plane yet -- "
            "the device list is available from a notebook as exp.device_catalog.",
            severity="warning")

    def _save_board_image(self) -> None:
        """Save the WHOLE monitor board (every panel, composited in its laid-out position) to a PNG.
        Unlike a per-plot save (data png+npz), this is a raster of the board region, so it is a plain
        image.  Reuses ``_last_save_dir`` so it shares the per-panel save's remembered folder."""
        default_dir = self._last_save_dir or str(_task_files_dir())
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
        # _PanelBoard is transparent -> composite onto an opaque white canvas so the PNG isn't see-through.
        _opaque_white_composite(pm).save(path)
        self._last_save_dir = str(Path(path).parent)
        self._update_summary()

    def _panel_frame_key(self, card, versions: Mapping[str, int]):
        """The identity of the frame this panel would draw from the current tick.

        A panel is stale exactly when one of the signals it reads has advanced.
        The key is deliberately per-card/per-signal: sharing a present cycle
        never asserts that unrelated producers emitted the same physical shot.
        """

        return tuple(
            (str(name), int(versions.get(str(name), 0)))
            for name in sorted(self._card_reads(card))
        )

    def _tick(self) -> None:
        # poll the logic nodes EVERY base tick (even when no new signal arrived) so a
        # one-shot node's run-complete / error transition is never missed if the node
        # self-stops between version bumps.
        self._poll_logic_nodes()
        # PAUSE = freeze the board: skip the render below (no card advances its
        # _render_version) while node polling/banners stay alive.  On Resume each
        # card compares only the revisions of the producers it actually reads.
        if self._paused:
            self._update_summary()
            return
        self._tick_count += 1
        # One immutable front for this GUI pass.  It is a per-producer-latest
        # observation, not a board-wide physical-shot join.
        self._tick_data = self._data.freeze()
        versions = self._tick_data.versions()
        elapsed = self._tick_count * self._base_interval_ms
        # COLLECT every panel whose own source revisions changed and whose beat is
        # due (or owed).  ``update_ms`` gates when that card recomposes; batching
        # only makes presentation efficient and carries no cross-producer
        # same-shot assertion.  A mid-drag panel is skipped whole (a live recompose under a drag stomps
        # the widget blit backgrounds); its beat is owed too, so it catches up right on release.
        batch = []
        for card in self.cards:
            key = self._panel_frame_key(card, versions)
            if key == card._render_version:
                continue                       # nothing this panel shows changed
            if elapsed % card.config.update_ms != 0 and not card._beat_owed:
                continue                       # beat not due (and none owed)
            # A card mid-gesture is skipped whole: recomposing under a drag would
            # replace the very front the pointer is measuring against.
            if bool(getattr(card, "_interacting", False)):
                card._beat_owed = True         # due but unservable this tick -> next idle tick serves it
                continue
            card._beat_owed = False
            batch.append((card, key))
        if batch:
            self._enqueue_render_batch(batch, self._tick_data)
        # keep the visible Edit tab's 'now:' acquisition references live, so a queued
        # parameter edit shows as applied once the loop picks it up.
        editor = self.tabs.currentWidget()
        if isinstance(editor, PanelEditor):
            editor.refresh_node_now_labels()
            editor.refresh_limit_hints()  # #3a: tick updates only the grey placeholder hint, never the text
        self._update_summary()

    def refresh_once(self) -> None:
        """Request one unconditional worker refresh of every current card."""

        self._poll_logic_nodes()
        self._refresh_signal_info()
        self._tick_data = self._data.freeze()
        versions = self._tick_data.versions()
        self._enqueue_render_batch(
            tuple(
                (card, self._panel_frame_key(card, versions))
                for card in self.cards
            ),
            self._tick_data,
            force=True,
        )
        editor = self.tabs.currentWidget()
        if isinstance(editor, PanelEditor):
            editor.refresh_node_now_labels()
            editor.refresh_limit_hints()  # #3a: tick updates only the grey placeholder hint, never the text
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
        # silently, so a red node error outranks even the running task's progress line; the
        # display-behind advisory is amber (the RUN is unaffected -- acquisition is unthrottled
        # and always shows latest); idle shows the board summary.  The strip itself change-gates
        # text/style, so this per-tick call never repolishes an unchanged state.
        faulted = [n for n in self.running_nodes if getattr(n, "last_error", None)]
        if faulted:
            node = faulted[0]
            who = (getattr(node, "prefix", "") or self._node_label(node) or "node").rstrip("_:")
            n = int(getattr(node, "consecutive_errors", 1))
            self.status_strip.show_message(
                f"⚠ NODE ERROR ({who}, ×{n}): {node.last_error}", severity="error")
        elif getattr(self, "_task_status_text", None):
            self.status_strip.show_message(self._task_status_text, severity="task")
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
        if self._task_locked:
            return
        try:
            state = self.read_state()
            start = self._address or str(_task_files_dir() / f"{state.name}.json")
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save task layout", start, "Task layout (*.json)")
            if not path:
                return
            state.save(path)
            self._address = path
            if self.save_button is not None:
                self.save_button.set_dirty(False)
            self._message(f"Saved: {path}")
        except Exception as exc:
            self._message(f"Save failed: {exc}")

    def load_from_file(self) -> None:
        if self._task_locked:
            return
        try:
            start = str(Path(self._address).parent) if self._address else str(_task_files_dir())
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load task layout", start, "Task layout (*.json)")
            if not path:
                return
            state = TaskConsoleState.load(path)
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
    def stop_all_nodes(self, timeout: float = 2.0) -> bool:
        """Stop every running node through THE lifecycle endpoint (``_stop_logic_node``) but KEEP
        the panels + editors intact.  This is the notebook close = "hide" path: closing the window
        halts all running processes, yet the layout is preserved so reopening the session-bound
        console (``exp.task_console()``) restores the SAME interface rather than a blank new one.
        Distinct from :meth:`shutdown`, which also tears the cards/editors down.

        Going row-by-row through ``_stop_logic_node`` (never bare ``node.stop()``) is load-bearing:
        a bare stop leaves the dead node in ``running_nodes`` and the row painted
        "running", so the reopened window exposes a false live owner even though
        that producer can no longer advance."""
        if QtCore.QThread.currentThread() is not self.thread():
            raise RuntimeError("node shutdown must run on the TaskConsole Qt owner thread")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
            raise ValueError("node shutdown timeout must be a non-negative number")
        deadline = time.monotonic() + float(timeout)
        stopped = True
        row_node_ids = {
            id(node)
            for row in self.logic_nodes
            for node in (
                self._logic_nodes.get(id(row)),
            )
            if node is not None
        }
        for row in list(self.logic_nodes):
            if (
                self._logic_nodes.get(id(row)) is not None
                is not None
            ):
                remaining = max(0.0, deadline - time.monotonic())
                stopped = self._stop_logic_node(row, timeout=remaining) and stopped
        # Row-less injected nodes (show_task_console(running_nodes=[...])) have
        # no row to repaint but must still leave the provider topology.
        for node in list(self.running_nodes):
            if id(node) in row_node_ids:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            confirmed = self._stop_run(node, timeout=remaining)
            stopped = confirmed and stopped
            if confirmed and node in self.running_nodes:
                self.running_nodes.remove(node)
        if not stopped:
            self.status_strip.show_message(
                "Close delayed: a logic-node owner thread is still active",
                severity="error",
            )
        return stopped

    def stop_nodes_using(self, affected_ids, timeout: float = 2.0) -> bool:
        """Stop exactly the running nodes that reference one of the ``affected_ids`` devices --
        the session's fine-grained device-change hook (``load_config`` swapping specific device
        INSTANCES).  A node is affected iff its :meth:`~LogicNode.referenced_devices` (EXCLUSIVE
        drivers AND OBSERVE records, unwrapped to real identity) intersect the swapped set, so
        reinitialising the camera stops every camera view / occupancy path riding it while a scan
        on the untouched sequencer keeps running.  Goes through the SAME
        ``_stop_logic_node`` endpoint as every other stop, so no dead provider is
        left advertised as live after close/reopen."""
        if QtCore.QThread.currentThread() is not self.thread():
            raise RuntimeError("node shutdown must run on the TaskConsole Qt owner thread")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout < 0:
            raise ValueError("node shutdown timeout must be a non-negative number")
        deadline = time.monotonic() + float(timeout)
        affected = {int(i) for i in affected_ids}
        if not affected:
            return True

        def _touches(node) -> bool:
            references = tuple(node.referenced_devices()) + tuple(
                getattr(node, "lifecycle_devices", lambda: ())()
            )
            return any(id(d) in affected for d in references)

        stopped = True
        row_node_ids = {
            id(node)
            for row in self.logic_nodes
            for node in (
                self._logic_nodes.get(id(row)),
            )
            if node is not None
        }
        for row in list(self.logic_nodes):
            node = self._logic_nodes.get(id(row))
            if node is not None:
                try:
                    touches = _touches(node)
                except BaseException:
                    stopped = False
                    continue
                if touches:
                    remaining = max(0.0, deadline - time.monotonic())
                    stopped = self._stop_logic_node(row, timeout=remaining) and stopped
        for node in list(self.running_nodes):        # row-less injected nodes
            if id(node) in row_node_ids:
                continue
            try:
                touches = _touches(node)
            except BaseException:
                stopped = False
                continue
            if touches:
                remaining = max(0.0, deadline - time.monotonic())
                confirmed = self._stop_run(node, timeout=remaining)
                stopped = confirmed and stopped
                if confirmed and node in self.running_nodes:
                    self.running_nodes.remove(node)
        return stopped

    def shutdown(self, timeout: float = 5.0) -> bool:
        """Stop the refresh timer and every running node's owner thread, then release
        the editors/cards.  IDEMPOTENT -- it is reached from both the window close
        (``show_task_console`` installs it as the pre-close guard) and an explicit
        ``with show_task_console(...)`` / re-run, which can both fire.

        Stopping a node sets its cooperative-cancel event (M5), so a node thread
        blocked in ``camera.acquire`` unwinds and the camera is released -- this is
        what keeps a closed dashboard from leaving a live acquire thread (and a held
        camera / RPyC connection) behind, which previously wedged the kernel.
        Raster work owns no QWidget and is detached on close; True means node,
        resource and Qt teardown completed, while False keeps the window alive
        so a later close can retry the unresolved external owner."""
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("shutdown timeout must be positive")
        if QtCore.QThread.currentThread() is not self.thread():
            raise RuntimeError("TaskConsole shutdown must run on its Qt owner thread")
        state = getattr(self, "_shutdown_state", "RUNNING")
        if state == "TERMINATED" or getattr(self, "_shut", False):
            return True
        if state in {
            "STOPPING_NODES",
            "WAITING_RENDER",
            "CLOSING_RESOURCES",
            "TEARING_DOWN_UI",
        }:
            return False
        deadline = time.monotonic() + float(timeout)
        self._timer.stop()
        if state in {"RUNNING", "BLOCKED_NODE_OWNERSHIP"}:
            self._shutdown_state = "STOPPING_NODES"
            if not self.stop_all_nodes(timeout=max(0.0, deadline - time.monotonic())):
                self._shutdown_state = "BLOCKED_NODE_OWNERSHIP"
                return False
        # Stop publication before tearing down Qt.  An already-started compose
        # may finish in the background, but its result has no QWidget reference
        # and the closing gate discards it.
        self._shutdown_render_lane()
        if not getattr(self, "_on_close_done", False):
            self._shutdown_state = "CLOSING_RESOURCES"
            on_close = getattr(self, "_on_close", None)
            if on_close is not None:
                try:
                    close_result = on_close()
                    if close_result is False:
                        raise RuntimeError("resource close reported unresolved ownership")
                except BaseException as exc:
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
        except BaseException as exc:
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
        if not self.shutdown():
            event.ignore()
            return
        super().closeEvent(event)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self.shutdown():
            raise RuntimeError("TaskConsole shutdown did not complete")
        return False

def show_task_console(
    *,
    state: TaskConsoleState | None = None,
    task: str | None = None,
    running_nodes: Sequence[object] = (),
    catalog_view: object | None = None,
    run_factory=None,
    data_plane=None,
    fit_window_factory=None,
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

    ``catalog_view`` is the CATALOG seam (contract 1): a
    :class:`~zlc_workbench.task_console.catalog_bridge.ConsoleCatalogView`
    projecting the domain DefinitionCatalog into the Add-Panel dropdown -- every
    entry brings its own parameter form, declared outputs and typed-request
    builder.  Omit it and the dropdown carries only plot kinds.

    Teardown: closing the window stops the refresh timer and every running node's owner
    thread (the cooperative-cancel path), so no acquire thread is left holding the
    camera -- closing the dashboard genuinely releases it.  Pass ``on_close`` (e.g.
    ``exp.close``) to ALSO disconnect/safe the devices the caller owns when the
    window closes.  In a notebook the returned console is also a context manager
    (``with show_task_console(...) as console:``) and exposes ``console.shutdown()``
    for deterministic teardown on a cell re-run."""

    ensure_qt_app()          # the console is a QWidget: the app must exist BEFORE its ctor
    if state is None and task is not None:
        state = resolve_task_state(task)
    console = TaskConsole(state=state, running_nodes=running_nodes,
                          catalog_view=catalog_view, run_factory=run_factory,
                          data_plane=data_plane,
                          fit_window_factory=fit_window_factory, scale=scale,
                          window_ratio=window_ratio)
    console._on_close = on_close
    # Closing the window must stop the node owner threads (else they keep running, blocked in
    # camera.acquire holding the camera / RPyC link, wedging the kernel).  The console is a CHILD
    # of the window so its own closeEvent never fires on a window close -- we wire the window's
    # signals instead.  Minimising NEVER stops the nodes (only the X / a genuine close does).
    def _wire_close(window) -> None:
        if hide_on_close:
            # Session-bound (notebook) console: the X HIDES the window (keeps the panel layout so a
            # later exp.task_console() restores the SAME interface) and stops every running node so
            # the devices are released.  The hide guard runs on X, never on minimize.
            window.set_hide_guard(console.stop_all_nodes)
        else:
            # Standalone window (.bat / explicit): the X fully tears the console down.
            window.set_close_guard(console.shutdown)
    # ONE launcher sequence (launch_fluent_window: wrap -> wire -> size -> centre -> show ->
    # retain), shared with every other show_* GUI so the steps cannot drift per-launcher.
    launch_fluent_window(console, title=title, hide_on_close=hide_on_close, wire=_wire_close)
    return console
