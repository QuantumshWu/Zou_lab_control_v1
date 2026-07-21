"""The panel Edit tab: a snapshot of the card plus its full parameter surface.

Qt x matplotlib, so its home is the task console's plot_bridge zone: the editor rebuilds
figures, owns preview canvases and reads the card's live plotter.  It empties into
qt_widgets with the rest of the zone at render purification.

Every import names a TRUE owner -- nothing here touches the legacy tree.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

import zlc_frontend.qt_widgets as _qt_widgets
from zlc_frontend.qt_widgets import (
    ACCENT,
    FluentButton,
    FluentComboBox,
    FluentLabel,
    FluentLineEdit,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    GREEN,
    GREY,
    ORANGE,
    RED,
    MeasurementPanel,
    fluent_text_width,
    scaled_px,
    signals_blocked as _signals_blocked,
)
from zlc_frontend.console_state import TaskConsoleState as _TaskConsoleState
from zlc_frontend.form import lenient_float as _safe_float
from zlc_frontend.live_plot.live import (
    coerce_panel_value,
    panel_display_size,
    panel_plot,
    region_binding,
)
from zlc_frontend.panel_params import (
    PANEL_PARAMS,
    panel_display_decls as _panel_display_decls,
    resolved_cmap as _resolved_cmap,
    resolved_param as _resolved_param,
)
from zlc_data.console_records import (
    BLANK_SOURCE as _BLANK_SOURCE,
    PANEL_KINDS,
    PanelConfig,
    panel_allows_multi_slot,
    panel_input_slots,
)
from zlc_storage.paths import display_path

from .plot_bridge import (
    _RELIM_PARAM,
    _card_y_is_view_axis,
    _general_fit_models_for_kind,
    _unit_df_for,
)

AnalysisControls = _qt_widgets.analysis_controls.AnalysisControls
PARAM_WIDGETS = _qt_widgets.param_widgets.PARAM_WIDGETS
ParamWidgetContext = _qt_widgets.param_widgets.ParamWidgetContext
fill_grouped_signal_combo = _qt_widgets.param_widgets.fill_grouped_signal_combo

try:
    import matplotlib.pyplot as plt
    from .plot_bridge_canvas import panel_canvas
except Exception:  # pragma: no cover - depends on the local matplotlib install
    panel_canvas = None


class PanelEditor(QtWidgets.QWidget):
    """One panel's processing tab (confocal per-plot control card).

    Opened from a panel's ``Edit...`` button as its OWN closable tab, so several
    panels can be edited side by side.  Holds a FROZEN snapshot of that panel's
    current data plus the heavy controls that do NOT belong in the lightweight
    Setting popup AND do NOT duplicate it (Setting owns source / size / colormap
    / relim / unit):

      * curve fit -- a structured model request gated by the plot family,
        shared with the live overlay and :class:`DataFigure`;
      * manual x/y limits + Save Fig;
      * for a panel that came from a measurement, that measurement's
        AUTO-generated parameter form (the same ParamDecl form the launcher
        uses, bound to the one spec) whose Start RE-RUNS the measurement into
        this very Monitor panel with the edited params.

    The whole page lives in a scroll area, so the snapshot never pushes the
    fit/limits row off-screen (the old single-editor cutoff)."""

    def __init__(self, card: "PanelCard", console: "TaskConsole", parent=None):
        super().__init__(parent)
        self.card = card
        self.console = console
        self.setStyleSheet("background: transparent;")
        # Which kind's PANEL_PARAMS this page baked its rows from -- a grid's RESOLVED per-cell
        # kind can change later (a facet / sub-plot pick), and refresh_on_show then rebuilds this
        # page so the rows never lie (the Edit mirror of _sync_settings_param_rows).
        self._param_kind_built = card._param_kind() if card is not None else None
        self._plotter = None
        self._canvas = None
        self._df = None
        # A plot panel's Edit NEVER carries a measurement/processor param form or
        # Start/Stop -- a plot is a pure VIEW, decoupled from acquisition (the
        # node lives on the Logic tab).  meas_panel stays None so any leftover
        # callers no-op.
        self.meas_panel = None
        self._node = None                       # the node that produces this panel's data
        self._node_widgets: dict = {}           # acquisition-param name -> editable field
        self._node_now_labels: dict = {}        # acquisition-param name -> "now: X" reference
        self.fit_combo = None
        self.xmin = self.xmax = self.ymin = self.ymax = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        page = QtWidgets.QWidget()
        page.setStyleSheet("background: transparent;")
        scroll.setWidget(page)
        col = QtWidgets.QVBoxLayout(page)
        m = scaled_px(10, minimum=6)
        col.setContentsMargins(m, m, m, m)
        col.setSpacing(scaled_px(6, minimum=4))

        def section(text):
            col.addWidget(FluentSectionLabel(text))   # the ONE section style (bold dark), like everywhere

        def labeled(text):
            lab = FluentLabel(text)
            lab.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            return lab

        # A plot panel is a PURE VIEW: its Edit carries ONLY the plotter controls
        # (acquisition of whatever node already publishes the signal it reads --
        # read-only-ish, snapshot, fit, manual limits, save).  It NEVER builds or
        # starts a node -- that lives on the Logic tab.

        # ---- Panel: rename this panel right here in the Edit (kept in sync with the Setting
        # popup's title field; both go through the sealed title API).  Saves a trip back to
        # the Setting popup just to relabel.
        section("Panel")
        self.title_edit = FluentLineEdit(card.config.title)
        self.title_edit.setPlaceholderText("panel title…")
        self.title_edit.setToolTip("Rename this panel (also the default save name).")
        self.title_edit.textChanged.connect(self._edit_title)
        col.addWidget(FluentSettingRow("title", self.title_edit, label_width=scaled_px(96, minimum=72)))

        # ---- Acquisition: the editable parameters of the DATA SOURCE behind this
        # panel.  A panel is a VIEW; the LOGIC NODE whose published signals it reads
        # declares what its source exposes via acquisition_parameters() -- a
        # raw-frame panel reads the camera Measurement (exposure / ROI), while a
        # derived panel reads its processor.  Each field is PREFILLED with
        # the CURRENT value (with a "now: X" reference); Apply pushes the edit to
        # that source IN PLACE (it does not start anything -- the node is started
        # from its own Logic-tab Edit).
        self._node = console._producing_node(card)
        for name, current in console._node_params(self._node):
            if not self._node_widgets:
                section("Acquisition")
            edit = FluentLineEdit(_py_to_text(current))
            # EXPAND to fill the row (stretch=1), not a fixed 110 px: a value
            # like a 4-int ROI [1648, 64, 1144, 64] was clipped while the right
            # half of the page sat empty.  The edit takes the slack; the live
            # "now:" reference trails it.  (cutoff is a core rule.)
            edit.setMinimumWidth(scaled_px(150, minimum=120))
            self._node_widgets[name] = edit
            now = labeled(f"now: {_py_to_text(current)}")
            self._node_now_labels[name] = now
            holder = QtWidgets.QWidget()
            hl = QtWidgets.QHBoxLayout(holder)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(scaled_px(6, minimum=4))
            hl.addWidget(edit, 1)
            hl.addWidget(now, 0)
            # widest node-param labels ("threshold_method" / "readout_exposure", 16 chars)
            # need ~170 px; 150 clipped a trailing char, so 170 fits the longest name in full.
            col.addWidget(FluentSettingRow(name, holder, label_width=scaled_px(170, minimum=140)))
        if self._node_widgets:
            self.node_apply_button = FluentButton("Apply", color=ACCENT)
            self.node_apply_button.setToolTip(
                "Apply the edited acquisition parameters to the data source in place\n"
                "(reconfigure the camera live) -- the panel keeps streaming.")
            self.node_apply_button.clicked.connect(self._restart_node)
            col.addWidget(self.node_apply_button)

        # ---- Source: a plot's signal ALWAYS comes from a measurement/processor (#2),
        # so show THAT node's full declarative parameter form here -- prefilled from its
        # current values (defaults via the spec).  Shown when the node exposes no live
        # acquisition_parameters above (a scanned measurement / a processor); Apply
        # rebuilds + restarts the source node with the edited params (it is Started /
        # Stopped from its OWN Logic-tab Edit, so this form carries no Start/Stop).
        self._source_row = None
        self.source_form = None
        if not self._node_widgets:
            self._source_row = console._producing_row(card)
            source_spec = (console._spec_for_logic(self._source_row.node)
                           if self._source_row is not None else None)
            if source_spec is not None:
                section(f"Source: {source_spec.name}")
                # pass the signal providers so a processor's source parameter renders
                # the SAME nested-by-producer picker the logic-node Edit uses --
                # not a flat/empty combo (every signal picker is the nested form, everywhere).
                self.source_form = MeasurementPanel(
                    [source_spec], single=True, controls=False,
                    signals_provider=getattr(console, "_signal_names", None),
                    sources_provider=getattr(console, "_signal_providers", None),
                    formats_provider=getattr(console, "_signal_formats", None),
                    short_names_provider=getattr(console, "_signal_short_names", None))
                self.source_form.seed_values(self._source_row.node.values or {})
                col.addWidget(self.source_form)
                self.source_apply_button = FluentButton("Apply", color=ACCENT)
                self.source_apply_button.setToolTip(
                    "Apply these parameters to the source node (rebuild + restart it);\n"
                    "the plot keeps reading its published signal.")
                self.source_apply_button.clicked.connect(self._apply_source_form)
                col.addWidget(self.source_apply_button)

        # ---- Parameters: the PLOT's own tunable API params as GUI controls,
        # auto-discovered from the kind's declarative specs.  Each edit re-renders
        # the LIVE panel AND this snapshot.  This is where "the params the plot
        # call exposes" live -- NOT in the basic Setting popup (which keeps only
        # source / size / colormap / relim / unit), so the two never duplicate.
        functional = [s for s in _panel_display_decls(card.config.kind, card._param_kind()) if not s.display]
        if functional:
            section("Parameters")
            for s in functional:
                widget = card._make_param_widget(s, apply=self._edit_param)
                col.addWidget(FluentSettingRow(s.label, widget, label_width=scaled_px(96, minimum=72)))

        # ---- Display: the plot's DataFigure VIEW knobs (colormap / relim / unit) -- the
        # SAME controls as the Setting popup, present HERE too so the Edit tab is the FULL
        # data_figure UI (whatever DataFigure can do has a control here).  They write the
        # SAME config.params keys and drive the SAME live card via the card's handlers
        # (single source -- Setting and Edit never drift), then re-snapshot this Edit canvas.
        self.ed_cmap = self.ed_relim = self.ed_unit_button = self.ed_fixed_row = None
        self.ed_params: dict[str, QtWidgets.QWidget] = {}
        section("Display")
        disp_lw = scaled_px(96, minimum=72)
        # The SAME declarative display knobs as the Setting popup -- the per-kind colormap / toggles
        # PLUS the relim chooser -- rendered through the card's SHARED _emit_param_rows, so the Edit
        # tab auto-exposes EVERY plot display ParamDecl with no hand-wiring (#H3v-4b).  Each edit
        # routes via _edit_param -> the live card (config.params + re-render) THEN re-snapshots here.
        # relim AND repeat_mode: the SAME declarative display knobs the Setting popup renders (relim
        # via _RELIM_PARAM, repeat_mode via _repeat_param_specs) are present HERE too, so the Edit tab
        # is the FULL data_figure UI (whatever the Setting can tune, the Edit can too).  Both route
        # through _edit_param -> card._set_param -- the ONE writer -- so Setting and Edit never drift.
        display_specs = ([s for s in _panel_display_decls(card.config.kind, card._param_kind()) if s.display]
                         + [_RELIM_PARAM] + list(card._repeat_param_specs()))
        self.ed_params = card._emit_param_rows(display_specs, col.addWidget, self._edit_param, disp_lw)
        self.ed_cmap = self.ed_params.get("cmap")        # named back-refs (kept for tests / clarity)
        self.ed_relim = self.ed_params.get("relim")
        # fixed lo/hi: the IDENTICAL bespoke [lo | hi] row the Setting popup builds (shared helper),
        # shown only when relim == "fixed"; _edit_param toggles it when relim changes.  EXCEPTION --
        # a 2d/sites panel's value axis IS the COLOUR limit (clim), not a y-axis: its manual pin
        # belongs with the ranges in the Limits section as an always-editable "colour range" row
        # (built below), NOT here.  Building it in BOTH places would put two inputs on the ONE
        # fixed_lo/hi source = exactly the drift the single-source rule forbids, so for an image kind
        # the Display fixed row is absent and self.ed_fixed_* stay None (every reader is None-guarded);
        # the relim chooser (tight/normal/fixed clim MODE) still lives here.
        if card.config.kind in ("2d", "sites"):
            self.ed_fixed_row = self.ed_fixed_lo = self.ed_fixed_hi = None
        else:
            self.ed_fixed_row, self.ed_fixed_lo, self.ed_fixed_hi = \
                card._make_fixed_lim_row(self._edit_fixed_lim, disp_lw)
            col.addWidget(self.ed_fixed_row)
            # The lo/hi row stays PERMANENTLY in the layout; only its inputs enable when relim ==
            # "fixed".  A setVisible toggle on a row that sits ABOVE the snapshot canvas in this shared
            # scroll page reflowed the unit row + the whole canvas DOWN by the row's height on every
            # relim change -- the reported Edit-tab jump.  Greying in place leaves every widget put.
            self._sync_fixed_lim_enabled(card._relim())
        # x-axis unit cycle -- the IDENTICAL row the Setting popup builds (shared helper), minus the
        # Setting-only current-unit label; the callback re-routes through the card's _on_unit_cycle.
        ed_unit_row, self.ed_unit_button, _ = \
            card._make_unit_cycle_row(self._edit_unit_cycle, disp_lw, with_label=False)
        col.addWidget(ed_unit_row)

        # ---- Processing: frozen snapshot + Refresh (Save is its own row below) -------------
        section("Processing")
        head = QtWidgets.QHBoxLayout()
        head.addWidget(labeled("frozen snapshot of current data"), 1)
        self.refresh_button = FluentButton("Refresh", color=GREY)
        self.refresh_button.setToolTip("Re-snapshot the panel's current data")
        self.refresh_button.clicked.connect(self.rebuild)
        head.addWidget(self.refresh_button)
        col.addLayout(head)

        self.canvas_holder = QtWidgets.QVBoxLayout()
        self.canvas_holder.setContentsMargins(0, 0, 0, 0)
        col.addLayout(self.canvas_holder)

        # Fit + manual limits are a panel concern, so they always build here.  A LOGIC NODE's Edit (on the
        # Logic tab, a LogicNodeEditor) deliberately carries no curve fit -- fitting
        # a curve is a plotter action, so you Add a Plot panel pointed at the
        # signals the node publishes for that.  A PLOT keeps the FULL DataFigure
        # model set (#176).
        # DataFigure picks the 1D / 2D path from the snapshot, so a 2D image
        # fits the 2D-Gaussian "2D center" model and lines fit the rest.  Same
        # [label | control] row idiom as the sections above (aligned label
        # column), so Fit / limits don't read as a cramped one-line jumble.
        proc_lw = scaled_px(96, minimum=72)

        def _inline(*widgets, trailing=None):
            host = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(host)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(scaled_px(6, minimum=4))
            for w in widgets:
                row.addWidget(w, 0)
            row.addStretch(1)
            if trailing is not None:
                row.addWidget(trailing, 0)
            return host

        # ---- Analysis: the SAME control set the Setting popup builds, through the ONE
        # :class:`AnalysisControls` builder (``surface='edit'`` -- adds the explicit Fit / Clear
        # buttons + gains the action combo incl. ROI + a result row).  Every action routes through the
        # ONE card mutator (card.set_fit_request), so the Edit and Setting surfaces are both views of
        # config.params['fit_request'] and can never disagree (#8).  ``fit_combo`` / ``ed_fix_seed``
        # stay the attribute names tests key off; ``do_fit`` / ``clear_fit`` delegate to the composite.
        self.fit_combo = None
        self.ed_fix_seed = None
        self._analysis_controls = None
        if self.card is not None:
            controls = AnalysisControls(self.card, surface="edit", label_w=proc_lw)
            if controls.empty:
                controls.deleteLater()
            else:
                section("Analysis")
                col.addWidget(controls)
                self._analysis_controls = controls
                self.fit_combo = controls.model_combo
                self.ed_fix_seed = controls.fix_seed

        # x/y-range pins (#3): the VIEW window, applied to the LIVE panel AND every grid cell (global,
        # not the snapshot only).  The boxes EDIT THE STORED PIN (``view_xlim``/``view_ylim``), NOT the
        # live range -- they hold the pin value (or are empty = autoscale) and are only re-seeded on
        # build / tab-show / Clear, so typing is NEVER clobbered by the refresh tick.  The live window
        # is shown as the grey PLACEHOLDER (``refresh_limit_hints``), a non-destructive hint Qt draws
        # only while a box is empty.  A y-range row exists ONLY where y is a VIEW axis
        # (``_card_y_is_view_axis``: an image, whose x AND y are pixel coordinates -- pinning both is
        # what makes an ROI crop real, since the image keeps aspect='equal' and an x-only pin would
        # letterbox).  On a 1d/dis panel the y VALUE axis is already owned by the relim family
        # (tight/normal/fixed in the Display section), so it gets no y row.  An image's VALUE axis is
        # the COLOUR limit -- a genuinely distinct quantity -- pinned by its own "colour range" row.
        section("Limits")
        self.xmin = FluentLineEdit(""); self.xmax = FluentLineEdit("")
        self.ymin = self.ymax = None
        if _card_y_is_view_axis(card):
            self.ymin = FluentLineEdit(""); self.ymax = FluentLineEdit("")
        # an image's value axis IS the colour limit: give 2d/sites an always-editable clim pin (built
        # into the Limits row below); every other kind has none (its value axis = the relim y).
        self.clo = self.chi = None
        if card.config.kind in ("2d", "sites"):
            self.clo = FluentLineEdit(""); self.chi = FluentLineEdit("")
        for w in (self.xmin, self.xmax) + ((self.ymin, self.ymax) if self.ymin is not None else ()):
            w.setFixedWidth(scaled_px(88, minimum=68))   # wide enough not to clip "-0.4960"
            w.returnPressed.connect(self.apply_limits)
        apply_btn = FluentButton("Apply lim", color=ACCENT)
        apply_btn.clicked.connect(self.apply_limits)
        # Clear releases the window pins back to autoscale -- the Limits counterpart of the Fit/Clear
        # pair above (without it a pinned window could never be undone: clearing the boxes + Apply
        # only errored 'bad limits', so the operator was stuck at whatever range they once set).
        clear_lim_btn = FluentButton("Clear", color=GREY)
        clear_lim_btn.clicked.connect(self.clear_limits)
        lim_row = _inline(self.xmin, self.xmax, trailing=apply_btn)
        lim_row.layout().addWidget(clear_lim_btn, 0)
        col.addWidget(FluentSettingRow("x range", lim_row, label_width=proc_lw))
        if self.ymin is not None:
            # ONE Apply/Clear pair drives BOTH rows (x + y are one view window, applied together).
            col.addWidget(FluentSettingRow("y range", _inline(self.ymin, self.ymax), label_width=proc_lw))

        # colour range (2d/sites only): an image's value axis is its COLOUR limit, so it gets an
        # always-editable clim pin here, mirroring the x-range row.  Apply writes the ONE clim source
        # (relim="fixed" + the card's apply_fixed_lims -> live card + snapshot + save); Auto releases
        # it to the autoscaled clim (relim="normal").  So this row and the Display relim chooser never
        # hold two copies of the pin -- they are two faces of the same fixed_lo/hi source.
        if self.clo is not None:
            for w in (self.clo, self.chi):
                w.setFixedWidth(scaled_px(88, minimum=68))
                w.returnPressed.connect(self.apply_clim)
            clim_apply = FluentButton("Apply", color=ACCENT)
            clim_apply.clicked.connect(self.apply_clim)
            clim_auto = FluentButton("Auto", color=GREY)
            clim_auto.clicked.connect(self.clear_clim)
            clim_row = _inline(self.clo, self.chi, trailing=clim_apply)
            clim_row.layout().addWidget(clim_auto, 0)
            col.addWidget(FluentSettingRow("colour range", clim_row, label_width=proc_lw))

        # ---- Command: a one-line REPL on the panel's DataFigure (confocal runs the same
        # `data_figure.<fn>(...)` form).  Type e.g. `data_figure.xlim(0, 10)`,
        # `df.fit('gaussian')`, `ax.set_title('x')` -- it runs on the snapshot and
        # the result/exception shows below.  Trusted-local-tool posture (same as the Scan
        # tab / fit args): only run code you wrote.
        section("Command")
        self.cmd_input = FluentLineEdit("")
        self.cmd_input.setPlaceholderText("data_figure.xlim(0, 10)")
        self.cmd_input.setStyleSheet(self.cmd_input.styleSheet() + " QLineEdit { font-family: Consolas, monospace; }")
        self.cmd_input.setToolTip(
            "Run a line of Python on this panel's figure.  Names: data_figure / df, fig, ax,\n"
            "plotter, np.  e.g. df.fit('gaussian') ; ax.set_title('x') ; data_figure.xlim(0,10)")
        cmd_run = FluentButton("Run", color=ACCENT)
        cmd_run.clicked.connect(self._run_command)
        self.cmd_input.returnPressed.connect(self._run_command)
        # the input FILLS the row (stretch 1) so it is not a tiny box; Run trails it.
        run_host = QtWidgets.QWidget()
        run_row = QtWidgets.QHBoxLayout(run_host)
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(scaled_px(6, minimum=4))
        run_row.addWidget(self.cmd_input, 1)
        run_row.addWidget(cmd_run, 0)
        col.addWidget(FluentSettingRow("run", run_host, label_width=proc_lw))
        # result: a read-only-but-COPYABLE field (FluentReadoutEdit) so you can select +
        # Ctrl-C the value/error -- a plain label can't be copied.
        self.cmd_result = FluentReadoutEdit("")
        self.cmd_result.setPlaceholderText("result / error appears here (select to copy)")
        self.cmd_result.setToolTip("The command's result or error — read-only, but select + Ctrl-C to copy.")
        col.addWidget(FluentSettingRow("result", self.cmd_result, label_width=proc_lw))

        # ---- Save: the ONE place to save from now (Setting no longer has Save).  The path
        # field is FULL WIDTH (its own row) so a long path is never cut off -- the reusable
        # FluentPathEdit (Browse picks a folder), prefilled with the LAST place a panel saved
        # (remembered on the console for the kernel session) or tasks/.  An "auto-name" SWITCH
        # decides the filename: ON (default) appends ``_<plot-kind>_<timestamp>`` so saves are
        # unique; OFF writes the path VERBATIM (you control the exact name, overwrites).  A
        # read-only preview shows the actual file that will be written -- the full name, not
        # just the folder.
        section("Save")
        self.save_dir_edit = FluentPathEdit(
            self.console._last_save_dir or str(_task_files_dir()),
            mode="dir", caption="Choose where to save", base_dir=str(_task_files_dir()))
        self.save_dir_edit.setToolTip(
            "Where to save (folder, or a full path base).  Remembered across saves this\n"
            "kernel session.  With auto-name OFF this is the exact output path.")
        col.addWidget(FluentSettingRow("path", self.save_dir_edit, label_width=proc_lw))
        # trailing spaces: FluentSwitch.sizeHint reserves the track width but paints the
        # label past track+gap, clipping the last few px -- the pad absorbs that (scales
        # with DPR since it is text), so the real label stays whole.
        self.save_autoname = FluentSwitch("auto-name (type + time)   ")
        self.save_autoname.setChecked(True)
        self.save_autoname.setToolTip(
            "ON: append _<plot-kind>_<timestamp> to the path (unique files).\n"
            "OFF: write the path verbatim (you set the exact name; overwrites).")
        # image FORMAT: a DATA-layer choice of the output CONTAINER (png / pdf / jpg), NOT an art
        # knob -- the figure's geometry/dpi/typography are unchanged, only the file it lands in.
        # It drives ``DataFigure.save(image_ext=...)`` (the ONE save path) and the file suffix;
        # the .npz payload is format-independent (same data + info for every choice).  Lowercase
        # extensions match confocal's ``save_type`` convention (jpg/png).
        self.save_format_combo = FluentComboBox()
        self.save_format_combo.addItems(list(SAVE_IMAGE_FORMATS))
        self.save_format_combo.setCurrentText(SAVE_IMAGE_FORMATS[0])
        self.save_format_combo.setFixedWidth(scaled_px(72, minimum=56))
        self.save_format_combo.setToolTip(
            "Image container for the saved figure (png / pdf / jpg).\n"
            "The matching .npz data file is the same for every format.")
        self.save_button = FluentButton("Save Fig", color=ACCENT)
        self.save_button.setToolTip("Save the edited figure (png / pdf / jpg) + matching data (npz).")
        col.addWidget(FluentSettingRow(
            "name", _inline(self.save_autoname, self.save_format_combo, trailing=self.save_button),
            label_width=proc_lw))
        # The previewed file path is a read-only-but-COPYABLE field (NOT a word-wrap label):
        # a long absolute path has no spaces to wrap on, so a word-wrap label would force its
        # own width to the full path and DRAG the whole Edit panel wider (the bug).  A
        # FluentReadoutEdit is a QLineEdit -- its size hint is content-INDEPENDENT (it scrolls
        # a long path instead of widening), so the panel width never tracks the path length,
        # and the resolved name stays selectable + Ctrl-C-able.
        self.save_preview = FluentReadoutEdit("")
        self.save_preview.setToolTip("The exact file that will be written — read-only, select to copy.")
        self.save_preview.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        col.addWidget(FluentSettingRow("file", self.save_preview, label_width=proc_lw))
        self.save_dir_edit.changed.connect(lambda *_: self._update_save_preview())
        self.save_autoname.toggled.connect(lambda *_: self._update_save_preview())
        self.save_format_combo.currentTextChanged.connect(lambda *_: self._update_save_preview())
        self.save_button.clicked.connect(self.save)
        self._update_save_preview()

        self.status = FluentLabel("")
        self.status.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        # A status line must NEVER drive the page WIDTH: a QLabel's sizeHint width tracks its text, so a
        # long message (e.g. "saved … → C:\very\long\path") would balloon the Edit column into a
        # horizontal scroll.  Ignored horizontal policy makes the label take the available width and CLIP
        # instead -- the layout width is owned by the canvas/rows, never by a log string (#layout-fixed).
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        col.addWidget(self.status)
        col.addStretch(1)

        self.rebuild()

    # ------------------------------------------------------------- snapshot
    def teardown(self) -> None:
        if self._canvas is not None:
            self.canvas_holder.removeWidget(self._canvas)
            self._canvas.deleteLater()
        if self._plotter is not None and plt is not None and self._plotter.fig is not None:
            plt.close(self._plotter.fig)
        self._canvas = None
        self._plotter = None
        self._df = None

    def rebuild(self) -> None:
        """Snapshot the bound card's CURRENT data, MIRRORING the Monitor frame.

        The snapshot uses the card's OWN size (not a forced 2x4) so it looks
        exactly like the Monitor panel and can never overflow the Edit page (the
        Monitor card already fits) -- and unlike the Monitor card it is built
        INTERACTIVE (default selectors on), so zoom / region-select work here."""
        card = self.card
        if card is None or panel_canvas is None:
            self.status.setText("open the panel with data first")
            return
        # Snapshot the MOST RECENT frame: pull the latest hub data into the Monitor
        # card first (one synchronous tick), so Refresh mirrors what the camera just
        # produced -- not the last timer-tick render (which can lag by a beat).  This
        # refresh is BEST-EFFORT: a mid-session signal that has no retained state at the
        # current display provenance raises SignalHistoryGap, and that must NOT abort the
        # Edit tab's construction -- opening the tab must never depend on a clean hub tick
        # (#1: 'edit tab 点不开', intermittent).  On failure we snapshot whatever the
        # plotter already holds; a later timer tick refreshes it.
        try:
            self.console.refresh_once()
        except Exception:
            pass
        if card.plotter is None:
            self.status.setText("open the panel with data first")
            return
        src = card.plotter
        kind = card.config.kind
        size = card.config.size          # mirror the Monitor frame, never force 2x4
        title = card.config.title or PANEL_KINDS[kind]
        view = card._view_kwargs(kind)   # ONE source: relim mode + fixed lo/hi -- for sites too (#4)
        cmap = _resolved_cmap(kind, card.config.params)   # operator pick, else the kind default (ONE resolver)
        try:
            if kind == "pulse":
                # A pulse panel's snapshot rebuilds through the SAME renderer the live card uses
                # (``build_pulse_preview_plot`` via the console's ``_pulse_state`` provider) -- NEVER the
                # else (1d/monitor) branch, which reads ``src.data_x``/``data_y`` a PulseSequenceFigure does
                # not carry (that was the blank-Edit-tab bug).  ``include_always_off`` comes from the card's
                # own params (same source as the live card), falling back to the node's recorded value.
                from zlc_frontend.live_plot.live import build_pulse_preview_plot
                resolved = self.console._pulse_state(card.config.inputs[0]) if card.config.inputs else None
                if resolved is None:
                    raise ValueError("pulse panel has no producing node to snapshot")
                state, node_include_off = resolved
                include_off = bool(card.config.params.get("include_always_off", node_include_off))
                new_plotter, _channels, _repeat = build_pulse_preview_plot(
                    state, include_always_off=include_off, size=size)   # Edit tab: interactive (default)
            elif kind == "grid":
                # A grid panel's snapshot rebuilds through the SAME builders the live card uses --
                # never the array else branch.  A FACET grid re-slices the card's last good data
                # through the ONE shared builder (_build_facet_plotter -- interactive here, like
                # every Edit snapshot); a recipe grid rebuilds via the _grid_recipe provider.
                if card._facet():
                    if card._last_namespace is None:
                        raise ValueError("facet grid has no data yet to snapshot")
                    new_plotter = card._build_facet_plotter(
                        card._signal_then_repeat(card._last_namespace), interactions=True)
                else:
                    from zlc_frontend.live_plot.live import build_grid_figure
                    recipe = self.console._grid_recipe(card.config.inputs[0]) if card.config.inputs else None
                    if recipe is None:
                        raise ValueError("grid panel has no producing node to snapshot")
                    recipe = card._grid_recipe_with_params(recipe)   # fold the panel's live display knobs in
                    new_plotter = build_grid_figure(recipe, interactions=True, size=size, display=False)
            elif kind == "2d":
                new_plotter = panel_plot(np.array(src.data_x, dtype=float),
                                         np.array(src.data_y[:, 0], dtype=float), kind="2d",
                                         size=size, cmap=cmap, **view,
                                         labels=tuple(src.labels), title=title)
            elif kind == "sites":
                new_plotter = panel_plot(np.array(src.data_x[:, :2], dtype=float),
                                         np.array(src.data_y[:, 0], dtype=float), kind="sites",
                                         size=size, image=getattr(src, "background", None),
                                         roi_radius=getattr(src, "roi_radius", 3.0), cmap=cmap, **view,
                                         labels=tuple(src.labels), title=title)
            elif kind == "hist":
                # Same declared-default resolver as the live build (_resolved_param) -- the Edit
                # snapshot can never render a different default than the Monitor card.
                new_plotter = panel_plot(np.array(src.values, dtype=float), kind="hist", size=size,
                                         bins=int(_resolved_param(kind, card.config.params, "bins")),
                                         fit=str(_resolved_param(kind, card.config.params, "fit")),
                                         ylog=bool(_resolved_param(kind, card.config.params, "ylog")), **view,
                                         labels=tuple(src.labels), title=title)
            else:  # 1d / monitor -> a line snapshot (monitor honours its show_dist toggle)
                extra = ({"show_dist": bool(_resolved_param(kind, card.config.params, "show_dist"))}
                         if kind == "monitor" else {})
                # Pass the live plotter's FULL data_y (all columns), not just column 0: a 1d ``create``
                # repeat_mode collapses to a ``(points, R)`` array = one line per repeat, and the snapshot
                # must draw EVERY column exactly like the live card (Live1D plots one line per column) --
                # else the Edit-tab preview would ignore repeat_mode and always show a single line.
                new_plotter = panel_plot(np.array(src.data_x[:, 0], dtype=float),
                                         np.array(src.data_y, dtype=float),
                                         kind=kind, size=size, **view, **extra,
                                         labels=tuple(src.labels), title=title)
        except Exception as exc:
            self.status.setText(f"could not snapshot: {str(exc).splitlines()[0][:120]}")
            return                       # build failed -> keep the LAST good snapshot, never blank (#4 "图有时消失")
        # Build-then-swap: the new plotter built cleanly, so ONLY NOW tear down the old snapshot and
        # install the new -- a failed rebuild above left the previous figure intact.
        # A grid snapshot may be showing an ENLARGED cell (GridPlot.focus): re-enlarge the SAME cell
        # on the fresh snapshot, so a knob that genuinely needs a re-snapshot still returns the user
        # to the zoomed view -- the generic mirror of the card's own refocus-across-rebuild rule.
        refocus_k = getattr(self._plotter, "_focused", None)
        self.teardown()
        self._plotter = new_plotter
        if refocus_k is not None and hasattr(self._plotter, "focus"):
            self._plotter.focus(int(refocus_k))
        self._canvas = panel_canvas(self._plotter.fig)
        # The Edit page lives in a SCROLL AREA: the page SCROLLS when the figure is taller than the
        # viewport, and the snapshot must never squish (clip) NOR grow.  The canvas already setFixedSize's
        # itself to its DPR-invariant design size at construction (and re-asserts it on every resync), so
        # pin_size() is now idempotent -- the deferred _zlc_resync / showEvent can no longer re-open the
        # size and balloon the figure across param edits (that was the setMinimumSize(sizeHint()) race).
        self._canvas.pin_size()
        self.canvas_holder.addWidget(self._canvas, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self._canvas.draw()      # SYNCHRONOUS: the snapshot shows its frame at once (never a blank)
        self._df = None
        # carry the persisted x-axis unit cycle onto the fresh snapshot (relim + cmap are
        # already baked into panel_plot above; unit is applied post-build, like the live card).
        self._apply_snapshot_unit()
        self._seed_limit_boxes()        # boxes = the stored pin (or empty=auto); grey hint = live range
        self.refresh_limit_hints()
        # confocal-style auto-range: interacting with this (interactive) Edit plot
        # writes the region the user marked back to the source's parameters.  The
        # selector / zoom is a GENERIC interface -- it only ever yields the
        # rectangle (endpoints) / view limits in plot coords; the SOURCE converts
        # that to its own format (a measurement -> scan range; a camera node ->
        # ROI; a 2-D scan -> axis ranges), so no device shape leaks into the GUI.
        # The SAME callback is bound to BOTH the zoom/scroll AND the area selector:
        # ZOOM/PAN updates from the view limits, the area selector OVERRIDES when
        # drawn (precedence in the writeback).  (Wiring only the area selector --
        # the earlier bug -- meant zoom did nothing.)
        area = getattr(self._plotter, "area", None)
        zoom = getattr(self._plotter, "zoom", None)
        # A plot is a pure view that only ever yields the marked rectangle (endpoints) in plot coords;
        # the PRODUCING node converts that to its own parameter -- so no device/measurement shape leaks
        # into the GUI, and the SAME rule serves every source:
        #   * a 2D IMAGE  -> the upstream camera node's ROI (endpoints -> sub-array), on zoom AND select;
        #   * a 1D CURVE of a scanning measurement -> that measurement's scan x-range (its axis_range
        #     param), on a DELIBERATE drag-select only (a mere zoom must not restage the scan).
        # The 1-D case is gated on the node DECLARING a scan range (``_node_scan_range_key``), so EVERY
        # measurement with an axis_range param gets the linkage by construction (never wired per node).
        if kind == "2d" and self._node is not None:
            if area is not None:
                area.callback = self._read_region
            if zoom is not None:
                zoom.callback = self._read_region
        elif self._node is not None and self.console._node_scan_range_key(self._node) is not None:
            if area is not None:
                area.callback = self._read_x_range
        self.status.setText("snapshot of current data — fit / set limits / save are frozen here"
                            if self.fit_combo is not None else
                            "snapshot of current data — Save Fig is frozen here")

    def _edit_param(self, key: str, value) -> None:
        """A plot-param edit from the Edit tab (declarative display knob OR functional param): apply to
        the LIVE panel via the card's ONE _set_param (config.params + live re-render -- single source,
        identical to the Setting popup), reveal the Edit tab's OWN fixed lo/hi row when relim flips to
        ``fixed``, then re-snapshot this canvas so the change shows in both surfaces (#H3v-4b)."""
        if self.card is not None:
            self.card._set_param(key, value)
        if key == "relim" and getattr(self, "ed_fixed_row", None) is not None:
            self._sync_fixed_lim_enabled(str(value))   # enable lo/hi in fixed WITHOUT moving the page
            if str(value) == "fixed" and self.card is not None:
                # mirror the freeze-current-view seed _set_param just wrote (config.params is the
                # one source) into THIS tab's lo/hi inputs -- setText does not re-fire editingFinished
                for edit, pkey in ((self.ed_fixed_lo, "fixed_lo"), (self.ed_fixed_hi, "fixed_hi")):
                    if edit is not None and pkey in self.card.config.params:
                        edit.setText(f"{float(self.card.config.params[pkey]):g}")
        if key == "relim":
            # a 2d/sites image has no Display fixed row (its clim lives in the Limits colour-range row):
            # re-seed THOSE boxes here so picking relim in the chooser fills/empties them to match the
            # pin -- runs even when ed_fixed_row is None, unlike the block above (#2 colour range).
            self._seed_clim_boxes()
        # EVERY knob first tries the snapshot's own in-place apply (BaseLivePlot.apply_param handles
        # the relim family for every kind; a GridPlot stores + forwards to its focused cell and NEVER
        # asks for a rebuild) -- rebuild() is only the fallback for a knob the snapshot truly cannot
        # apply, because it tears down + recreates the whole snapshot (which would throw away a
        # focused grid cell = the "changing lim bounces the enlarged cell back to the grid" bug).
        snap = getattr(self, "_plotter", None)
        if snap is not None and snap.apply_param(key, value):
            if getattr(self, "_canvas", None) is not None:
                self._canvas.draw_idle()
            return
        self.rebuild()

    def _sync_fixed_lim_enabled(self, relim: str) -> None:
        """The Edit tab's fixed lo/hi row is ALWAYS in the layout -- only its INPUTS enable when
        ``relim == "fixed"``.  A ``setVisible`` toggle on a row that sits above the snapshot canvas in
        the shared scroll page reflowed the unit row + the whole canvas down by the row's height on
        every relim change (the reported Edit-tab jump); greying the inputs in place leaves every
        widget put.  (The Setting popup keeps its reveal-and-grow-down: it is a compact floating card
        with nothing beneath the row, so growing downward moves nothing.)"""
        if self.ed_fixed_row is not None:
            # ALWAYS shown here (the shared _make_fixed_lim_row hides it by default for the Setting
            # popup's reveal-on-fixed) -- so the footprint is constant and the canvas never moves.
            self.ed_fixed_row.setVisible(True)
        fixed = str(relim) == "fixed"
        for w in (self.ed_fixed_lo, self.ed_fixed_hi):
            if w is not None:
                w.setEnabled(fixed)

    def _edit_fixed_lim(self) -> None:
        """The Edit tab's fixed lo/hi committed (#H2): apply to the LIVE card through its ONE
        ``apply_fixed_lims`` path (config.params + in-place apply_param; a zoomed grid also stores
        onto the parked grid), then push onto THIS tab's SNAPSHOT in place too (the same apply_param
        relim family; a GridPlot forwards to its focused cell) -- never a re-snapshot, which would
        bounce a focused grid cell back to the grid."""
        if self.card is None:
            return
        lo = _safe_float(self.ed_fixed_lo.text(), 0.0)
        hi = _safe_float(self.ed_fixed_hi.text(), 1.0)
        self.card.apply_fixed_lims(lo, hi)
        snap = getattr(self, "_plotter", None)
        if snap is not None and snap.apply_param("fixed_lo", lo) and snap.apply_param("fixed_hi", hi):
            if getattr(self, "_canvas", None) is not None:
                self._canvas.draw_idle()
            return
        self.rebuild()

    def _edit_unit_cycle(self) -> None:
        if self.card is not None:
            self.card._on_unit_cycle()            # config.params["unit_index"] + live cycle
        self.rebuild()

    def _edit_title(self, text: str) -> None:
        """Rename the panel from the Edit tab: route through the card's sealed title handler
        (config.title + live plot title via style.apply_title) and keep the Setting popup's
        title field in sync, then re-snapshot + refresh the save preview."""
        if self.card is None:
            return
        self.card._on_title(str(text))            # config.title + sealed live title
        edit = getattr(self.card, "title_edit", None)
        if edit is not None and edit.text() != text:
            with _signals_blocked(edit):
                edit.setText(str(text))
        self._update_save_preview()               # default save name follows the title
        self.rebuild()

    def _run_command(self) -> None:
        """A one-line REPL on this panel's DataFigure (confocal's ``data_figure.<fn>(...)``).
        Names exposed: ``data_figure``/``df``, ``fig``, ``ax``, ``plotter``, ``np``.  An
        expression shows its repr; a statement shows ``ok``; an error shows its message.
        SECURITY: runs arbitrary local Python -- trusted-input tool, like the Scan tab."""
        if self._plotter is None or not hasattr(self, "cmd_input"):
            return
        text = self.cmd_input.text().strip()
        if not text:
            return
        try:
            df = self._df_for()
            ns = {"data_figure": df, "df": df, "fig": self._plotter.fig,
                  "ax": getattr(self._plotter, "ax", None), "plotter": self._plotter, "np": np}
            try:
                value = eval(text, ns)            # noqa: S307 - local experiment tool, trusted input
                msg = "ok" if value is None else repr(value)
            except SyntaxError:
                exec(text, ns)                    # noqa: S102 - a statement, not an expression
                msg = "ok"
            if self._canvas is not None:
                self._canvas.draw_idle()
            self.cmd_result.setText(str(msg)[:300])
        except Exception as exc:
            self.cmd_result.setText(f"error: {str(exc).splitlines()[0][:200]}")

    def _apply_snapshot_unit(self) -> None:
        """Cycle the Edit snapshot's x-axis unit ``unit_index`` times from its original,
        mirroring the live card's :meth:`PanelCard._apply_unit` (shared :func:`_unit_df_for`).
        Called on every rebuild so the persisted unit survives a Refresh / re-snapshot."""
        if self._plotter is None or self.card is None:
            return
        index = int(self.card.config.params.get("unit_index", 0) or 0)
        if index <= 0:
            return
        try:
            df = _unit_df_for(self._plotter)
            if not getattr(df, "conversion_map", None):
                return
            for _ in range(index % max(1, len(df.conversion_map))):
                df.change_unit()
            if self._canvas is not None:
                self._canvas.draw_idle()
        except Exception:
            pass

    def _selected_rect(self):
        """The area rectangle if one is drawn, ELSE the current view box (x AND y
        view limits).  Returns (xlo, xhi, ylo, yhi) sorted, or all None -- so
        scroll-zoom alone sets the region, with a drag-rectangle overriding it."""
        plotter = self._plotter
        if plotter is None or plotter.ax is None:
            return (None, None, None, None)
        area = getattr(plotter, "area", None)
        if area is not None and area.range[0] is not None:
            x1, x2, y1, y2 = (float(v) for v in area.range)
            return (min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2))
        xlo, xhi = sorted(float(v) for v in plotter.ax.get_xlim())
        ylo, yhi = sorted(float(v) for v in plotter.ax.get_ylim())
        return (xlo, xhi, ylo, yhi)

    def _read_region(self) -> None:
        """Confocal ``_read_range`` for a 2D panel, GENERIC over the source.

        ZOOM/PAN or an area SELECTION yields the marked rectangle as endpoints
        (x_min, x_max, y_min, y_max) in the panel's axis coordinates (the area
        selection overrides the view box when drawn).  Those ENDPOINTS -- the only
        thing the selector knows -- are handed to the producing source via
        ``region_to_acquisition_parameters``; the SOURCE converts them to its own
        Acquisition parameters (a camera node -> a ROI rectangle; a 2-D scan ->
        axis ranges).  Whatever fields it names are filled in the Edit form, then
        Apply pushes them.  The frontend encodes NO device-specific shape."""
        if self._node is None or self._plotter is None:
            return
        convert = getattr(self._node, "region_to_acquisition_parameters", None)
        if convert is None:
            return
        x1, x2, y1, y2 = self._selected_rect()
        if x1 is None:
            return
        params = convert(x1, x2, y1, y2) or {}
        filled = {}
        for name, value in params.items():
            edit = self._node_widgets.get(name)
            if edit is not None:
                edit.setText(_py_to_text(value))
                filled[name] = value
        if filled:
            self.status.setText(f"region from view: {filled} — Apply to use it")

    def _read_x_range(self) -> None:
        """Confocal ``_read_range`` for a 1-D CURVE panel, GENERIC over the source.

        A drag-selected x-interval on a scanning measurement's plot becomes that measurement's NEXT
        scan x-range.  The selector knows only the endpoints (plot coords); they are staged onto the
        PRODUCING measurement's OWN Logic-tab Edit form (its first ``axis_range`` param, via
        ``MeasurementPanel.set_axis_range``), so ANY measurement that declares a scan range gets this by
        construction and the plot stays a pure view with no measurement internals.  ``Start`` on that
        form then re-runs the scan over the marked range (confocal's mark-then-rerun)."""
        if self._node is None:                     # the plotter-dependence lives in _selected_rect (None-safe)
            return
        x1, x2, y1, y2 = self._selected_rect()
        if x1 is None:
            return
        form = self.console._form_for_node(self._node)
        if form is not None and form.set_axis_range(x1, x2):
            self.status.setText(f"scan range from view: [{x1:g}, {x2:g}] — Start to use it")

    def refresh_node_now_labels(self) -> None:
        """Update the 'now: <value>' references beside each Acquisition field to the
        source's CURRENT values.  The console calls this each tick for the visible
        Edit tab (one general hook, not a per-field signal), so after the loop
        applies a queued edit the references catch up on their own -- no manual
        wiring per parameter, and the frozen snapshot / Refresh / Fit controls are
        untouched."""
        if self._node is None or not self._node_now_labels:
            return
        for name, current in self.console._node_params(self._node):
            label = self._node_now_labels.get(name)
            if label is not None:
                label.setText(f"now: {_py_to_text(current)}")

    def _restart_node(self) -> None:
        """Apply the edited Acquisition params to the data source, routed through the
        node's safe entry (queued + applied BETWEEN shots in the loop's own thread,
        never a GUI-thread stop/start); Apply also STARTs an idle source so it goes
        live.

        Re-snapshot timing is the subtle part.  An IDLE node applies + publishes the
        new-param frame SYNCHRONOUSLY (inside ``apply_acquisition_parameters``), so we
        refresh + re-snapshot right away.  A RUNNING node only publishes the new-param
        frame on its NEXT loop iteration, so reading the hub now would snapshot the
        STALE pre-edit frame -- so we wait for the node's acquisition epoch to advance
        (the first frame computed with the new params is on the hub) and re-snapshot
        THAT, polled off the GUI critical path (no acquire from this thread)."""
        if self._node is None:
            return
        new_params = {name: _text_to_py(w.text()) for name, w in self._node_widgets.items()}
        running = bool(getattr(self._node, "running", False))
        epoch0 = int(getattr(self._node, "acquisition_epoch", lambda: 0)())
        try:
            self.console._restart_node(self._node, new_params)
        except Exception as exc:
            self.status.setText(f"apply failed: {str(exc).splitlines()[0][:120]}")
            return
        self.refresh_node_now_labels()        # reuse the same refresh the tick uses
        if running:
            # queued: the loop has not yet produced a new-param frame -- defer the snapshot
            self.status.setText("acquisition parameters queued — Monitor updates on the next frame")
            self._await_fresh_frame(self._node, epoch0)
        else:
            # idle: apply published a fresh frame synchronously, so it is already here
            self.console.refresh_once()
            self.status.setText("acquisition started with the new parameters")
            self.rebuild()

    def _apply_source_form(self) -> None:
        """Apply the SOURCE node's edited parameter form (#2) to the producing node --
        rebuild + restart it (or live where it accepts), then re-snapshot."""
        if self._source_row is None or self.source_form is None:
            return
        try:
            values = self.source_form.collect_values()
        except Exception as exc:
            self.status.setText(f"apply failed: {str(exc).splitlines()[0][:120]}")
            return
        if not self.console._apply_source_params(self._source_row, values):
            self.status.setText("locked: a task is running — Stop it first")
            return
        self.console.refresh_once()
        self.status.setText("source parameters applied")
        self.rebuild()

    def _await_fresh_frame(self, node, epoch0: int, *, _tries: int = 0) -> None:
        """Poll (off the GUI critical path) until the running node has published its
        FIRST frame computed with the just-queued params -- detected by its
        ``acquisition_epoch`` advancing past ``epoch0`` -- then refresh + re-snapshot
        the Edit panel.  Bounded so a wedged/slow node never spins forever (it falls
        back to a refresh after the cap).  Aborts if the editor moved to another node
        or was torn down."""
        if self._node is not node or node is None:
            return
        epoch = int(getattr(node, "acquisition_epoch", lambda: epoch0 + 1)())
        if epoch > epoch0 or _tries >= 600:    # ~600 * 25 ms = 15 s safety cap
            self.console.refresh_once()
            self.refresh_node_now_labels()
            self.rebuild()
            return
        from PyQt5 import QtCore
        QtCore.QTimer.singleShot(25, lambda: self._await_fresh_frame(node, epoch0, _tries=_tries + 1))

    def _df_for(self):
        # Route through the plotter's OWN to_data_figure() -- the SINGLE source that already knows a
        # GridPlot is N per-cell DataFigures (returns a _GridData composite) while every other kind is a
        # flat DataFigure.  Building DataFigure(self._plotter) directly would collapse a grid to cell-0's
        # axes over placeholder arrays, so fit/limits would touch one subplot with garbage; going through
        # the override makes fit_targets()/xlim()/ylim() fan out over every cell.  Cached so repeated
        # Fit/Clear reuse the SAME per-cell handles (clear_fit clears this instance, not a fresh one).
        if self._df is None:
            self._df = self._plotter.to_data_figure()
        return self._df

    def do_fit(self) -> None:
        """Apply a curve fit from the Edit tab through the ONE :class:`AnalysisControls` composite (which
        funnels into card.set_fit_request), so the live overlay, the hub node, and the Setting popup's
        Analysis controls all follow the same config.params['fit_request'] state.  No-op when the panel
        family offers no fit (kept as a public method the tests + Fit button call)."""
        controls = getattr(self, "_analysis_controls", None)
        if controls is None:
            return
        controls.do_fit()
        if self.fit_combo is not None and self.fit_combo.currentData():
            self.status.setText(f"fit {self.fit_combo.currentText()}: applied (see panel)")

    def clear_fit(self) -> None:
        controls = getattr(self, "_analysis_controls", None)
        if controls is not None:
            controls.clear_fit()
        elif self.card is not None:
            self.card.set_fit_request(None)
        self.status.setText("fit cleared")

    def _refresh_edit_analysis(self) -> None:
        """Re-derive the Edit tab's Analysis controls from the card's stored state -- so the Edit surface
        always shows the SAME fit the Setting popup set, the two being pure views of the single source (#8)."""
        controls = getattr(self, "_analysis_controls", None)
        if controls is not None:
            controls.derive()

    def refresh_on_show(self) -> None:
        """When this panel's Edit tab becomes visible, refresh anything that may have changed
        since it was last shown: re-seed the display knobs from config.params, snap the manual-limit
        fields to current view, re-read the producing source's 'now' acquisition values, and refresh
        any embedded source form's dynamic combos (the ONE hook the tab-switch handler calls -- nothing
        special-cases PanelEditor versus LogicNodeEditor)."""
        # A grid's param rows follow its RESOLVED per-cell kind: when a facet / sub-plot pick changed
        # it since this page was built, rebuild the page in place (fresh rows + snapshot) -- the Edit
        # mirror of the Setting popup's _sync_settings_param_rows.
        card = self.card
        if card is not None and card.config.kind == "grid" \
                and card._param_kind() != self._param_kind_built:
            self.console._refresh_panel_editor(card)
            return
        self._refresh_display_params()
        self._refresh_edit_analysis()   # re-seed the fit model + fix/seed from config.params['fit_request']
        self._seed_limit_boxes()        # re-seed the pin from config.params (may have changed in Setting)
        self.refresh_limit_hints()
        self.refresh_node_now_labels()
        hook = getattr(getattr(self, "source_form", None), "refresh_on_show", None)
        if callable(hook):
            hook()

    def _refresh_display_params(self) -> None:
        """Re-seed the Edit tab's display-knob controls (``ed_params``) from the live card's
        ``config.params`` -- the SINGLE source of truth -- so switching back to this tab shows the
        CURRENT values even when they were changed in the Setting popup (which writes the same
        config.params).  Each control is re-seeded through its kind's ``PARAM_WIDGETS.write`` (the card
        records the kinds while building both surfaces' rows), signals blocked so re-seeding does not
        re-fire ``_edit_param``.  The #6 mirror of ``PanelCard.refresh_on_show`` -- both surfaces are a
        VIEW of config.params, refreshed on show, never private copies that drift."""
        if self.card is None:
            return
        kinds = getattr(self.card, "_param_kinds", {})
        params = self.card.config.params
        for key, widget in self.ed_params.items():
            kind = kinds.get(key)
            if kind is None or key not in params:
                continue
            with _signals_blocked(widget):
                try:
                    PARAM_WIDGETS[kind].write(widget, params[key])
                except (TypeError, ValueError):
                    continue

    def _limits_ax(self):
        """The axes the x-range boxes are a live VIEW of (#3): the LIVE card's plotter when it exists (so a
        zoom / pan on the panel reflects here), else the Edit-tab snapshot.  For a grid the x-window is
        shared across cells, so cell-0's axes is representative."""
        for plotter in (getattr(self.card, "plotter", None) if self.card is not None else None, self._plotter):
            if plotter is None:
                continue
            ax = getattr(plotter, "ax", None)
            if ax is None:
                axes = getattr(plotter, "site_axes", None)     # a grid: cells share one x-window
                ax = axes[0] if axes else None
            if ax is not None:
                return ax
        return None

    def _seed_limit_boxes(self) -> None:
        """Put the STORED x-window pin (``view_xlim`` in ``config.params``) into the boxes.  The boxes
        EDIT THE PIN, never the live autoscaled range, so their text never wanders on its own: empty
        boxes mean 'no pin' (autoscale), a value means 'pinned there'.  Called on build / tab-show /
        after Clear -- NEVER on the refresh tick (the root cause of the old 'I type and it rewrites'
        bug was ``fill_limits`` doing ``setText`` every tick).  The live range is shown separately as a
        non-destructive grey hint (:meth:`refresh_limit_hints`)."""
        if self.xmin is None:
            return                          # no Limits controls on this editor instance
        for key, lo_box, hi_box in self._limit_axes():
            pin = self.card.config.params.get(key) if self.card is not None else None
            lo = hi = ""
            if pin is not None:
                try:
                    lo, hi = f"{float(pin[0]):.6g}", f"{float(pin[1]):.6g}"
                except (TypeError, ValueError, IndexError):
                    lo = hi = ""
            with _signals_blocked(lo_box, hi_box):
                lo_box.setText(lo); hi_box.setText(hi)
        self._seed_clim_boxes()          # keep the 2d/sites colour-range boxes in step with the clim pin

    def refresh_limit_hints(self) -> None:
        """Refresh ONLY the grey PLACEHOLDER of the x-range boxes to the panel's current x-window -- a
        non-destructive live reference.  Qt shows a placeholder ONLY while the box is empty, so this can
        never overwrite a pinned value or the operator's in-progress typing.  This is the tick-safe
        successor of the old ``fill_limits`` (whose per-tick ``setText`` clobbered input): the tick calls
        THIS, so the boxes stay a live VIEW of the x-window (unpinned) while remaining freely editable."""
        if self.xmin is None:
            return                          # no Limits controls on this editor instance
        # colour-range (2d/sites) live hint: show the current clim as the grey PLACEHOLDER so an empty
        # box (= Auto) still tells the operator what range the image is using -- the clim counterpart of
        # the x-window hint below.  Non-destructive: Qt draws a placeholder only while the box is empty,
        # so a pinned/typed value is never overwritten.
        if getattr(self, "clo", None) is not None and self.card is not None \
                and self.card.plotter is not None and hasattr(self.card.plotter, "current_lims"):
            try:
                clo, chi = self.card.plotter.current_lims()
                self.clo.setPlaceholderText(f"{clo:.6g}"); self.chi.setPlaceholderText(f"{chi:.6g}")
            except Exception:
                pass
        ax = self._limits_ax()
        if ax is None:
            return
        xlo, xhi = ax.get_xlim()
        self.xmin.setPlaceholderText(f"{xlo:.6g}"); self.xmax.setPlaceholderText(f"{xhi:.6g}")
        if self.ymin is not None:           # y hint only where y is a view axis (an image family)
            ylo, yhi = ax.get_ylim()
            self.ymin.setPlaceholderText(f"{ylo:.6g}"); self.ymax.setPlaceholderText(f"{yhi:.6g}")

    def _limit_axes(self) -> list[tuple[str, object, object]]:
        """The (param key, lo box, hi box) rows this Edit's Limits section carries -- x always, y only
        on an image family (the y row exists iff ``_card_y_is_view_axis`` at build).  The ONE list
        apply/clear/seed/hints iterate, so adding an axis row can never miss a handler."""
        axes = [("view_xlim", self.xmin, self.xmax)]
        if self.ymin is not None:
            axes.append(("view_ylim", self.ymin, self.ymax))
        return axes

    def apply_limits(self) -> None:
        """Apply the typed view-window pins -- PER AXIS: a filled pair pins that axis, an empty pair
        releases it (so 'clear the y boxes + Apply' un-pins y while keeping x).  Each pin is an ORDINARY
        display knob (``view_xlim``/``view_ylim``) applied through the SAME ``_edit_param`` entry as
        bins / fit / relim -> the LIVE card (whose ``apply_param`` fans it to EVERY cell and re-asserts
        it after each tick's autoscale) + the snapshot + ``config.params`` + save.  So Apply reaches the
        RUNNING grid and STICKS (#3), not a one-shot on a throwaway ``_GridData`` the next redraw
        autoscaled away."""
        if self.xmin is None:
            return
        applied, cleared = [], []
        for key, lo_box, hi_box in self._limit_axes():
            lo_text, hi_text = lo_box.text().strip(), hi_box.text().strip()
            if not lo_text and not hi_text:
                self._edit_param(key, None)          # empty pair = release THIS axis's pin
                cleared.append(key[5])               # 'x' / 'y'
                continue
            try:
                lo, hi = float(lo_text), float(hi_text)
            except ValueError as exc:
                self.status.setText(f"bad limits: {str(exc).splitlines()[0][:100]}")
                return
            self._edit_param(key, (lo, hi))
            applied.append(key[5])
        parts = ([f"{'/'.join(applied)} range applied"] if applied else []) + \
                ([f"{'/'.join(cleared)} range cleared"] if cleared else [])
        self.status.setText("; ".join(parts) + " (all subplots)")

    def clear_limits(self) -> None:
        """Release EVERY view-window pin -> autoscale.  ``None`` is the SAME stored display knob
        ``apply_limits`` writes, routed through the ONE ``_edit_param`` entry: BaseLivePlot.apply_param /
        GridCell.consume_param already read ``None`` as 'no pin', so the live card, the snapshot, and the
        save all drop the pin.  EMPTIES the boxes (empty = no pin) so the grey placeholder hint takes
        over showing the now-auto range -- the Limits counterpart of :meth:`clear_fit`."""
        if self.xmin is None:
            return
        for key, lo_box, hi_box in self._limit_axes():
            self._edit_param(key, None)
            with _signals_blocked(lo_box, hi_box):
                lo_box.setText(""); hi_box.setText("")
        self.refresh_limit_hints()
        self.status.setText("view range cleared (auto)")

    def _sync_relim_combo(self, value: str) -> None:
        """Reflect a PROGRAMMATIC relim change (the colour-range Apply/Auto) in the Display relim combo
        WITHOUT re-firing its handler, so the chooser and the colour-range row always agree on whether
        the clim is pinned.  No-op when the combo is absent (a non-image Edit that has no relim row)."""
        combo = self.ed_params.get("relim") if getattr(self, "ed_params", None) else None
        if combo is None:
            return
        idx = combo.findText(value)
        if idx >= 0:
            with _signals_blocked(combo):
                combo.setCurrentIndex(idx)

    def apply_clim(self) -> None:
        """Pin a 2d/sites panel's COLOUR limit to the typed lo/hi.  Routes through the ONE clim source the
        relim family owns (relim="fixed" + the card's ``apply_fixed_lims``), so the live card (every cell,
        re-asserted each tick), this tab's snapshot, ``config.params`` and Save all move together -- there
        is no second hand-copied clim path.  Both boxes empty = Auto (release the pin)."""
        if self.clo is None:
            return
        if not self.clo.text().strip() and not self.chi.text().strip():
            self.clear_clim()
            return
        try:
            lo, hi = float(self.clo.text()), float(self.chi.text())
        except ValueError as exc:
            self.status.setText(f"bad colour range: {str(exc).splitlines()[0][:80]}")
            return
        self._edit_param("relim", "fixed")          # make the fixed clim take effect (seeds from view) ...
        if self.card is not None:
            self.card.apply_fixed_lims(lo, hi)      # ... then overwrite with the typed lo/hi (final state)
        snap = getattr(self, "_plotter", None)      # push onto THIS tab's snapshot too (as _edit_fixed_lim)
        if snap is not None:
            snap.apply_param("fixed_lo", lo); snap.apply_param("fixed_hi", hi)
            if getattr(self, "_canvas", None) is not None:
                self._canvas.draw_idle()
        self._sync_relim_combo("fixed")
        # _edit_param("relim","fixed") re-seeded the boxes from the FROZEN current view; re-seed now that
        # apply_fixed_lims has written the TYPED lo/hi so the boxes show what was actually pinned.
        self._seed_clim_boxes()
        self.status.setText("colour range applied")

    def clear_clim(self) -> None:
        """Release the colour-limit pin back to the autoscaled clim (relim="normal") -- the image
        counterpart of :meth:`clear_limits`.  Empties the boxes so the grey placeholder hint takes over
        showing the now-auto clim."""
        if self.clo is None:
            return
        self._edit_param("relim", "normal")
        with _signals_blocked(self.clo, self.chi):
            self.clo.setText(""); self.chi.setText("")
        self._sync_relim_combo("normal")
        self.refresh_limit_hints()
        self.status.setText("colour range cleared (auto)")

    def _seed_clim_boxes(self) -> None:
        """Put the STORED clim pin (``fixed_lo/hi``, in force only while relim=="fixed") into the
        colour-range boxes.  Like :meth:`_seed_limit_boxes` for x: the boxes edit the PIN, so empty = Auto
        and a value = pinned; re-seeded on build / tab-show / relim change, NEVER on the tick."""
        if getattr(self, "clo", None) is None:
            return
        p = self.card.config.params if self.card is not None else {}
        lo = hi = ""
        if str(p.get("relim")) == "fixed":
            try:
                lo, hi = f"{float(p.get('fixed_lo')):.6g}", f"{float(p.get('fixed_hi')):.6g}"
            except (TypeError, ValueError):
                lo = hi = ""
        with _signals_blocked(self.clo, self.chi):
            self.clo.setText(lo); self.chi.setText(hi)

    def _save_stem(self, timestamp: str | None) -> Path:
        """The output file stem (no extension) the Save section resolves to.

        ``path`` (the picker) is the base.  With auto-name ON the file is
        ``<base-or-base/title>_<plot-kind>_<timestamp>`` (unique); a ``timestamp`` of None
        yields the literal ``<time>`` placeholder for the read-only preview.  With auto-name
        OFF the base is used VERBATIM (its extension stripped), so the operator sets the exact
        name.  ONE resolver, shared by :meth:`save` and :meth:`_update_save_preview`."""
        title = (self.card.config.title or self.card.config.kind).strip() or "panel"
        kind = self.card.config.kind
        text = self.save_dir_edit.text().strip() if hasattr(self, "save_dir_edit") else ""
        base = Path(text) if text else Path(self.console._last_save_dir or _task_files_dir())
        # a bare folder (blank default / trailing sep / an existing dir) -> the file is
        # <folder>/<title>; otherwise the path already names the file stem.
        if not text or str(base).endswith(("/", "\\")) or (base.is_dir() and not base.suffix):
            base = base / title
        base = base.with_suffix("")               # we always append our own .png / .npz
        if hasattr(self, "save_autoname") and self.save_autoname.isChecked():
            return base.parent / f"{base.name}_{kind}_{timestamp or '<time>'}"   # unique
        return base                                # verbatim (operator-set name; overwrites)

    def _save_image_ext(self) -> str:
        """The image container the next Save writes (``png`` / ``pdf`` / ``jpg``).

        ONE reader of the format picker, shared by :meth:`save` and :meth:`_update_save_preview`, so the
        previewed suffix and the file actually written never disagree.  Falls back to the first
        :data:`SAVE_IMAGE_FORMATS` entry when the picker is absent (e.g. a non-plot editor) or empty."""
        combo = getattr(self, "save_format_combo", None)
        ext = combo.currentText().strip().lower() if combo is not None else ""
        return ext or SAVE_IMAGE_FORMATS[0]

    def _update_save_preview(self) -> None:
        """Show the actual file (full path) the next Save writes -- not just the folder."""
        if not hasattr(self, "save_preview"):
            return
        self.save_preview.setText(
            f"{display_path(str(self._save_stem(None)))}.{self._save_image_ext()} + .npz")

    def _save_view_state(self) -> dict[str, object]:
        """The panel's DISPLAY knobs, DERIVED from the ONE source (``config.params`` keyed by the ONE
        ``PANEL_PARAMS`` catalog), folded into saved ``info['view']`` so ``load_figure(...).plot()`` and
        the figure viewer reopen the figure AS SEEN.  Captures BOTH:

        * the cross-kind view knobs that live OUTSIDE ``PANEL_PARAMS`` (relim mode + fixed lo/hi from
          ``card._relim`` / the fixed bounds, unit index, repeat mode);
        * EVERY kind-specific knob the panel actually renders from -- looped from
          ``PANEL_PARAMS[_param_kind()]`` (a hist(-grid)'s bins/fit/ylog, a 2d/sites(-grid)'s colormap,
          a monitor's length/show_dist), the SAME declarative catalog the render path (``_build_plot`` /
          ``_build_facet_plotter``) and the Setting/Edit UI read.  Each stores the panel's EFFECTIVE
          value (operator's pick else the declared default), so the npz is self-describing and reopens
          exactly as drawn.

        This kills the old divergence where the save HARD-LISTED only 6 keys and silently dropped every
        other rendered knob (bins/fit/ylog/length/show_dist) -- a knob added to ``PANEL_PARAMS`` is now
        saved automatically, so *display == reopened figure* by construction.  ``_param_kind()`` yields a
        grid's per-cell ``sub_plot_kind``, so a grid captures its cells' knobs through the very same loop
        (the same thing ``_grid_recipe_with_params`` already does).  ``cmap`` is one of these declared
        keys: ``params.get('cmap', decl.default)`` is exactly the operator-pick-else-kind-default the
        ``_resolved_cmap`` resolver gives, so an image/site-map save still records the real colormap name
        it drew with; a kind with no colormap param simply omits the key (never a stray ``''``)."""
        params = self.card.config.params
        view: dict[str, object] = {
            "relim": self.card._relim(),
            "fixed_lo": _safe_float(str(params.get("fixed_lo", 0.0)), 0.0),
            "fixed_hi": _safe_float(str(params.get("fixed_hi", 1.0)), 1.0),
            "unit_index": int(params.get("unit_index", 0) or 0),
            "repeat_mode": self.card._repeat_mode_value(),
        }
        # the view-window pins (#3) are cross-kind view knobs too: persist them so a reopened figure keeps
        # the operator's window (only when actually set -- a never-pinned panel omits them, staying
        # autoscale; view_ylim only ever exists on the image family, whose store gate wrote it).
        for key in ("view_xlim", "view_ylim"):
            pin = params.get(key)
            if pin:
                view[key] = [float(pin[0]), float(pin[1])]
        # The serialized request is the one cross-surface fit state.  It contains
        # model + Selection + typed options and is safe to round-trip as data.
        if params.get("fit_request"):
            view["fit_request"] = dict(params["fit_request"])
        for decl in _panel_display_decls(self.card.config.kind, self.card._param_kind()):
            view[decl.key] = params.get(decl.key, decl.default)
        return view

    def save(self) -> None:
        if self._plotter is None:
            return
        try:
            df = self._df_for()
            # BIND the figure to its data SOURCE, then let ``DataFigure.save`` capture ``info['signals']``
            # + ``info['provenance']`` through the ONE frontend-neutral core (the SAME path a notebook
            # ``p.save()`` runs) -- the console only supplies its resolvers (a stopped node's lingering
            # signal still resolves via ``_node_for_signal``, incl. the ``_last_node`` fallback).  The
            # GUI-only display blocks (source/kind/size/view) stay ``extra_info`` (they are console state,
            # not signal/provenance).
            inputs = list(self.card.config.inputs or [])
            value_node = self.console._node_for_signal(inputs[0]) if inputs else None
            df.bind_source(self.console.hub, value_node, inputs=inputs,
                           resolve_node=self.console._node_for_signal, session=self.console.session)
            stem = self._save_stem(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            stem.parent.mkdir(parents=True, exist_ok=True)
            # the operator-chosen container (png / pdf / jpg) drives BOTH the file suffix and
            # ``DataFigure.save(image_ext=...)`` -- the ONE format reader keeps them in lockstep.
            ext = self._save_image_ext()
            # a path WITH a suffix makes DataFigure.save write it VERBATIM (no extra timestamp),
            # so this resolver is the single source of the output name.
            out = df.save(stem.with_suffix(f".{ext}"), image_ext=ext,
                          extra_info={"source": self.card.config.source,
                                      "kind": self.card.config.kind,
                                      "size": self.card.config.size,
                                      "view": self._save_view_state()})
            self.console._last_save_dir = str(stem.parent)   # remember where (kernel session)
            self._update_save_preview()
            # Show the LEAF folder, not the full absolute path (the full path is one click away in the
            # tooltip + the save-preview field) -- a short message that fits, with the width-safe status
            # policy as the backstop so even a long folder name can never widen the page.
            self.status.setText(f"saved {out['figure'].name} + {out['data'].name} → …/{stem.parent.name}")
            self.status.setToolTip(str(stem.parent))
        except Exception as exc:
            self.status.setText(f"save failed: {str(exc).splitlines()[0][:120]}")
