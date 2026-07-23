"""The panel Edit tab: a snapshot of the card plus its full parameter surface.

Qt presentation belongs in the task console's plot_bridge zone: the editor owns
only a frozen raster view and copies the card's accepted immutable front.  It
never evaluates data or owns a Matplotlib composer.

Every import names a TRUE owner -- nothing here touches the legacy tree.
"""

from __future__ import annotations

from pathlib import Path
import time

from PyQt5 import QtCore, QtGui, QtWidgets

import zlc_frontend.qt_widgets as _qt_widgets
from zlc_frontend.qt_widgets import (
    ACCENT,
    FluentButton,
    FluentComboBox,
    FluentLabel,
    FluentLineEdit,
    FluentPathEdit,
    FluentReadoutEdit,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentSwitch,
    GREEN,
    GREY,
    ORANGE,
    RED,
    MeasurementPanel,
    fluent_text_width,
    scaled_px,
    signals_blocked as _signals_blocked,
)
from zlc_frontend.console_state import (
    TaskConsoleState as _TaskConsoleState,
    task_files_dir as _task_files_dir,
)
from zlc_frontend.form import (
    lenient_float as _safe_float,
    python_to_text as _py_to_text,
    text_to_python as _text_to_py,
)
from zlc_frontend.panel_params import panel_param_decls as _panel_param_decls
from .plot_bridge import _RELIM_PARAM
from zlc_storage.paths import display_path


PARAM_WIDGETS = _qt_widgets.param_widgets.PARAM_WIDGETS
ParamWidgetContext = _qt_widgets.param_widgets.ParamWidgetContext
fill_grouped_signal_combo = _qt_widgets.param_widgets.fill_grouped_signal_combo

#: Containers the Save row offers.  A container is a DATA-layer choice (which file
#: the same picture lands in), never an art knob: geometry, dpi and typography are
#: identical whichever is picked.
SAVE_IMAGE_FORMATS = ("png", "jpg")



def _front_qimage(frame):
    """The presented front as one Qt image, or None.

    Reads the panel's own raster rather than re-rendering: what gets written is
    the exact front the operator was looking at, palette and colour window
    included.
    """

    panels = tuple(getattr(frame, "panels", ()) or ())
    if not panels:
        return None
    raster = panels[0].raster
    fmt = (QtGui.QImage.Format_Indexed8
           if str(raster.pixel_format.value).lower().startswith("indexed")
           else QtGui.QImage.Format_RGB32)
    image = QtGui.QImage(bytes(raster.pixels), raster.width, raster.height,
                         raster.stride_bytes, fmt)
    payload = panels[0].display_payload
    palette = tuple(getattr(payload, "base_palette", ()) or ())
    if palette and fmt == QtGui.QImage.Format_Indexed8:
        image.setColorTable(list(palette))
    return image.copy()          # own the bytes: the raster is not ours to keep


class PanelEditor(QtWidgets.QWidget):
    """One panel's processing tab (confocal per-plot control card).

    Opened from a panel's ``Edit...`` button as its OWN closable tab, so several
    panels can be edited side by side.  Holds a FROZEN snapshot of that panel's
    current data plus the heavy controls that do NOT belong in the lightweight
    Setting popup AND do NOT duplicate it (Setting owns source / size / colormap
    / relim / unit):

      * the panel's exact FINAL-artifact Fit entrance, with all fit authoring
        delegated to the one shared :class:`DataFigure` host;
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
        self._board = None
        self._displayed_figure = None
        self._displayed_display = None
        self._follow_next_front = False
        card.front_presented.connect(self._on_card_front_presented)
        # A plot panel's Edit never carries a measurement form or Start/Stop: a plot is a
        # pure VIEW, and the node that produces its data lives on the Logic tab.
        self.meas_panel = None
        self._node = None                       # the node that produces this panel's data
        self._node_widgets = {}                 # acquisition-param name -> editable field
        self._node_now_labels = {}              # acquisition-param name -> "now: X" reference
        self.xmin = self.xmax = self.ymin = self.ymax = None
        self.clo = self.chi = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        page = QtWidgets.QWidget()
        page.setStyleSheet("background: transparent;")
        scroll.setWidget(page)
        col = QtWidgets.QVBoxLayout(page)
        margin = scaled_px(10, minimum=6)
        col.setContentsMargins(margin, margin, margin, margin)
        col.setSpacing(scaled_px(6, minimum=4))

        def section(text):
            col.addWidget(FluentSectionLabel(text))

        def labeled(text):
            label = FluentLabel(text)
            label.setStyleSheet("color: %s; background: transparent; border: none;" % GREY)
            return label

        def inline(*widgets, trailing=None):
            host = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(host)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(scaled_px(6, minimum=4))
            for widget in widgets:
                row.addWidget(widget, 0)
            row.addStretch(1)
            if trailing is not None:
                row.addWidget(trailing, 0)
            return host

        label_w = scaled_px(96, minimum=72)

        # ---- Panel: rename here as well as in the Setting popup; both go through the
        # card's one title handler, so the two surfaces stay views of one string.
        section("Panel")
        self.title_edit = FluentLineEdit(card.config.title)
        self.title_edit.setPlaceholderText("panel title...")
        self.title_edit.setToolTip("Rename this panel (also the default save name).")
        self.title_edit.editingFinished.connect(self._commit_title)
        col.addWidget(FluentSettingRow("title", self.title_edit, label_width=label_w))

        # ---- Acquisition: the editable parameters of the SOURCE behind this panel,
        # prefilled with the current value and trailed by a live "now:" reference.
        # Apply pushes them to that source in place; it starts nothing.
        self._node = console._producing_node(card)
        for name, current in console._node_params(self._node):
            if not self._node_widgets:
                section("Acquisition")
            edit = FluentLineEdit(_py_to_text(current))
            edit.setMinimumWidth(scaled_px(150, minimum=120))
            self._node_widgets[name] = edit
            now = labeled("now: %s" % _py_to_text(current))
            self._node_now_labels[name] = now
            holder = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(holder)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(scaled_px(6, minimum=4))
            row.addWidget(edit, 1)
            row.addWidget(now, 0)
            col.addWidget(FluentSettingRow(name, holder,
                                           label_width=scaled_px(170, minimum=140)))
        if self._node_widgets:
            self.node_apply_button = FluentButton("Apply", color=ACCENT)
            self.node_apply_button.setToolTip(
                "Apply the edited acquisition parameters to the data source in place -- "
                "the panel keeps streaming.")
            self.node_apply_button.clicked.connect(self._restart_node)
            col.addWidget(self.node_apply_button)

        # ---- Source: the producing node's own declarative form, when it exposes no live
        # acquisition parameters above.  Apply rebuilds + restarts that node; it is
        # started and stopped from its OWN Logic-tab Edit, so no Start/Stop here.
        self._source_row = None
        self.source_form = None
        if not self._node_widgets:
            self._source_row = console._producing_row(card)
            source_spec = (console._spec_for_logic(self._source_row.node)
                           if self._source_row is not None else None)
            if source_spec is not None:
                section("Source: %s" % source_spec.name)
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
                    "Apply these parameters to the source node (rebuild + restart it); "
                    "the plot keeps reading its published signal.")
                self.source_apply_button.clicked.connect(self._apply_source_form)
                col.addWidget(self.source_apply_button)

        # ---- Parameters: the plot's own functional params, auto-discovered from the
        # kind's declarations, so a kind that gains a knob shows it with no wiring here.
        functional = [spec for spec in _panel_param_decls(card.config.kind)
                      if not spec.display]
        if functional:
            section("Parameters")
            for spec in functional:
                widget = card._make_param_widget(spec, apply=self._edit_param)
                col.addWidget(FluentSettingRow(spec.label, widget, label_width=label_w))

        # ---- Display: the same view knobs the Setting popup renders, through the card's
        # SHARED row emitter and writing the SAME config.params through the card's one
        # writer -- the two surfaces are views of one state and cannot drift.
        self.ed_cmap = self.ed_relim = self.ed_unit_button = self.ed_fixed_row = None
        self.ed_fixed_lo = self.ed_fixed_hi = None
        self.ed_params = {}
        section("Display")
        display_specs = ([spec for spec in _panel_param_decls(card.config.kind)
                          if spec.display] + [_RELIM_PARAM])
        self.ed_params = card._emit_param_rows(display_specs, col.addWidget, self._edit_param, label_w)
        self.repeat_mode_row, self.repeat_mode_combo = card._make_repeat_mode_row(
            self._edit_repeat_mode,
            label_w,
        )
        col.addWidget(self.repeat_mode_row)
        self.ed_cmap = self.ed_params.get("colormap")
        self.ed_relim = self.ed_params.get("relim")
        # An image's VALUE axis is its colour limit, pinned by the "colour range" row in
        # Limits; a second fixed lo/hi here would put two inputs on one source.
        if card.config.kind != "2d":
            self.ed_fixed_row, self.ed_fixed_lo, self.ed_fixed_hi = card._make_fixed_lim_row(
                self._edit_fixed_lim, label_w)
            col.addWidget(self.ed_fixed_row)
            # The row stays PERMANENTLY in the layout; only its inputs enable in fixed
            # mode.  A visibility toggle above the snapshot reflowed everything below it
            # by the row's height on every relim change -- the reported Edit-tab jump.
            self._sync_fixed_lim_enabled(card._relim())
        unit_row, self.ed_unit_button, _ = card._make_unit_cycle_row(
            self._edit_unit_cycle, label_w, with_label=False)
        col.addWidget(unit_row)

        # ---- Processing: the frozen snapshot and its Refresh.
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

        # ---- Analysis: another view of the card's one FINAL-artifact command.
        # Model/axis/constraint authoring and the result overlay live in the
        # shared DataFigure host; this Edit tab does not create a parallel
        # parallel fit draft or worker.
        if self.card is not None and self.card.fit_analysis_sink is not None:
            section("Analysis")
            col.addWidget(self.card.make_fit_analysis_button())

        # ---- Limits: the view-window pins.  A box holds the STORED pin (empty =
        # autoscale) and is re-seeded only on build / show / Clear, so the refresh tick
        # can never clobber typing; the live window shows as the grey placeholder.
        section("Limits")
        self.xmin = FluentLineEdit("")
        self.xmax = FluentLineEdit("")
        if card.config.kind == "2d":
            # An image's x AND y are pixel coordinates: pinning both is what makes a crop
            # real.  A curve's y is owned by the relim family instead, so it gets no row.
            self.ymin = FluentLineEdit("")
            self.ymax = FluentLineEdit("")
            self.clo = FluentLineEdit("")
            self.chi = FluentLineEdit("")
        boxes = (self.xmin, self.xmax) + ((self.ymin, self.ymax) if self.ymin is not None else ())
        for widget in boxes:
            widget.setFixedWidth(scaled_px(88, minimum=68))
            widget.returnPressed.connect(self.apply_limits)
        apply_button = FluentButton("Apply lim", color=ACCENT)
        apply_button.clicked.connect(self.apply_limits)
        clear_button = FluentButton("Clear", color=GREY)
        clear_button.clicked.connect(self.clear_limits)
        lim_row = inline(self.xmin, self.xmax, trailing=apply_button)
        lim_row.layout().addWidget(clear_button, 0)
        col.addWidget(FluentSettingRow("x range", lim_row, label_width=label_w))
        if self.ymin is not None:
            col.addWidget(FluentSettingRow("y range", inline(self.ymin, self.ymax),
                                           label_width=label_w))
        if self.clo is not None:
            for widget in (self.clo, self.chi):
                widget.setFixedWidth(scaled_px(88, minimum=68))
                widget.returnPressed.connect(self.apply_clim)
            clim_apply = FluentButton("Apply", color=ACCENT)
            clim_apply.clicked.connect(self.apply_clim)
            clim_auto = FluentButton("Auto", color=GREY)
            clim_auto.clicked.connect(self.clear_clim)
            clim_row = inline(self.clo, self.chi, trailing=clim_apply)
            clim_row.layout().addWidget(clim_auto, 0)
            col.addWidget(FluentSettingRow("colour range", clim_row, label_width=label_w))

        # ---- Save: the figure this panel is showing.  Only the picture is written here:
        # the DATA behind it is already owned by the run's repository, and a second copy
        # written by the GUI would be a second answer to what was measured.
        section("Save")
        self.save_dir_edit = FluentPathEdit(
            self.console._last_save_dir or str(_task_files_dir()),
            mode="dir", caption="Choose where to save", base_dir=str(_task_files_dir()))
        self.save_dir_edit.setToolTip(
            "Where to save (folder, or a full path base).  Remembered across saves this "
            "session.  With auto-name OFF this is the exact output path.")
        col.addWidget(FluentSettingRow("path", self.save_dir_edit, label_width=label_w))
        self.save_autoname = FluentSwitch("auto-name (type + time)   ")
        self.save_autoname.setChecked(True)
        self.save_autoname.setToolTip(
            "ON: append _<plot-kind>_<timestamp> to the path (unique files).  "
            "OFF: write the path verbatim (you set the exact name; overwrites).")
        self.save_format_combo = FluentComboBox()
        self.save_format_combo.addItems(list(SAVE_IMAGE_FORMATS))
        self.save_format_combo.setCurrentText(SAVE_IMAGE_FORMATS[0])
        self.save_format_combo.setFixedWidth(scaled_px(72, minimum=56))
        self.save_format_combo.setToolTip("Image container for the saved figure.")
        self.save_button = FluentButton("Save Fig", color=ACCENT)
        self.save_button.setToolTip("Save exactly the picture this snapshot is showing.")
        col.addWidget(FluentSettingRow(
            "name", inline(self.save_autoname, self.save_format_combo, trailing=self.save_button),
            label_width=label_w))
        # A read-only-but-copyable field, not a wrapping label: a long absolute path has
        # nothing to wrap on, so a label would drag the whole page wider.
        self.save_preview = FluentReadoutEdit("")
        self.save_preview.setToolTip("The exact file that will be written -- select to copy.")
        self.save_preview.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        col.addWidget(FluentSettingRow("file", self.save_preview, label_width=label_w))
        self.save_dir_edit.changed.connect(lambda *_: self._update_save_preview())
        self.save_autoname.toggled.connect(lambda *_: self._update_save_preview())
        self.save_format_combo.currentTextChanged.connect(lambda *_: self._update_save_preview())
        self.save_button.clicked.connect(self.save)
        self._update_save_preview()

        self.status = FluentLabel("")
        self.status.setStyleSheet("color: %s; background: transparent; border: none;" % GREY)
        # A status line must never drive the page WIDTH: a label's size hint tracks its
        # text, so a long message would balloon the column into a horizontal scroll.
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        col.addWidget(self.status)
        col.addStretch(1)

        self.rebuild()

    def rebuild(self) -> None:
        """Copy the card's accepted immutable front into the frozen view.

        Refresh is a presentation copy, not a second render.  The live card's
        worker compose is the only evaluation/rasterization for a semantic
        commit, so the Edit snapshot cannot disagree with it or double the
        cost of every parameter change.
        """

        from zlc_frontend.qt_widgets import FrozenRasterView

        card = self.card
        front = (
            None
            if card is None or card.board is None
            else card.board.front_frame
        )
        if front is None:
            self.status.setText("open the panel with data first")
            return
        if self._board is None:
            self._board = FrozenRasterView(card.panel_id, empty_text="no snapshot yet",
                                       zoomable=True)
            self.canvas_holder.addWidget(self._board)
        try:
            figure = card.frozen_data_figure()
        except Exception as error:
            self.status.setText("%s: %s" % (type(error).__name__, error))
            return
        self._board.present(front)
        self._displayed_figure = figure
        self._displayed_display = card._last_display
        self.status.setText("")

    def _on_card_front_presented(self) -> None:
        if not self._follow_next_front:
            return
        self._follow_next_front = False
        self.rebuild()

    def teardown(self) -> None:
        """Release this tab's frozen Qt presentation surface."""

        if self._board is not None:
            self.canvas_holder.removeWidget(self._board)
            self._board.setParent(None)
            self._board.deleteLater()
        self._board = None
        if self.card is not None:
            try:
                self.card.front_presented.disconnect(self._on_card_front_presented)
            except (TypeError, RuntimeError):
                pass

    def _edit_param(self, key: str, value) -> None:
        """Commit one authored parameter to the card's sole render request."""
        if self.card is not None:
            self._follow_next_front = True
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
            # a 2D image has no Display fixed row (its clim lives in the Limits colour-range row):
            # re-seed THOSE boxes here so picking relim in the chooser fills/empties them to match the
            # pin -- runs even when ed_fixed_row is None, unlike the block above (#2 colour range).
            self._seed_clim_boxes()
        # ``front_presented`` copies the accepted immutable raster into this
        # frozen view.  There is deliberately no Edit-side composer.

    def _edit_repeat_mode(self, mode) -> None:
        if self.card is None:
            return
        self._follow_next_front = True
        self.card._commit_repeat_mode(mode)

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
        ``apply_fixed_lims`` path (config.params + local display commit), then let the next accepted
        front update this tab.  It never re-reads acquisition merely because a limit changed."""
        if self.card is None:
            return
        lo = _safe_float(self.ed_fixed_lo.text(), 0.0)
        hi = _safe_float(self.ed_fixed_hi.text(), 1.0)
        self._follow_next_front = True
        self.card.apply_fixed_lims(lo, hi)

    def _edit_unit_cycle(self) -> None:
        if self.card is not None:
            self.card._on_unit_cycle()

    def _commit_title(self) -> None:
        """Commit the Edit tab's local title draft once."""
        if self.card is None:
            return
        text = str(self.title_edit.text())
        if text == self.card.config.title:
            return
        self._follow_next_front = True
        edit = getattr(self.card, "title_edit", None)
        if edit is not None:
            with _signals_blocked(edit):
                edit.setText(text)
        self.card._commit_title()
        self._update_save_preview()               # default save name follows the title

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
            self.status.setText(f"apply failed: {exc}")
            return
        self.refresh_node_now_labels()        # reuse the same refresh the tick uses
        if running:
            # queued: the loop has not yet produced a new-param frame -- defer the snapshot
            self.status.setText("acquisition parameters queued — Monitor updates on the next frame")
            self._await_fresh_frame(self._node, epoch0)
        else:
            # idle: apply published a fresh frame synchronously, so it is already here
            self._follow_next_front = True
            self.console.refresh_once()
            self.status.setText("acquisition started with the new parameters")

    def _apply_source_form(self) -> None:
        """Apply the SOURCE node's edited parameter form (#2) to the producing node --
        rebuild + restart it (or live where it accepts), then re-snapshot."""
        if self._source_row is None or self.source_form is None:
            return
        try:
            values = self.source_form.collect_values()
        except Exception as exc:
            self.status.setText(f"apply failed: {exc}")
            return
        if not self.console._apply_source_params(self._source_row, values):
            self.status.setText("locked: a task is running — Stop it first")
            return
        self._follow_next_front = True
        self.console.refresh_once()
        self.status.setText("source parameters applied")

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
            self._follow_next_front = True
            self.console.refresh_once()
            self.refresh_node_now_labels()
            return
        from PyQt5 import QtCore
        QtCore.QTimer.singleShot(25, lambda: self._await_fresh_frame(node, epoch0, _tries=_tries + 1))

    def refresh_on_show(self) -> None:
        """When this panel's Edit tab becomes visible, refresh anything that may have changed
        since it was last shown: re-seed the display knobs from config.params, snap the manual-limit
        fields to current view, re-read the producing source's 'now' acquisition values, and refresh
        any embedded source form's dynamic combos (the ONE hook the tab-switch handler calls -- nothing
        special-cases PanelEditor versus LogicNodeEditor)."""
        card = self.card
        self._refresh_display_params()
        if card is not None:
            card._seed_repeat_mode_control(
                self.repeat_mode_combo,
                self.repeat_mode_row,
            )
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

    def _limit_axes(self):
        """The view-window rows this editor built, as ``(param key, lo box, hi box)``.

        One list drives seeding, applying and clearing, so a kind that has no y
        row simply contributes no y triple instead of every reader repeating the
        same "does this kind have y?" test.
        """

        rows = [("view_xlim", self.xmin, self.xmax)]
        if self.ymin is not None:
            rows.append(("view_ylim", self.ymin, self.ymax))
        return tuple(rows)

    def _front_view_bounds(self):
        """The window the composed front is showing, as (xlo, xhi, ylo, yhi).

        Read off the front's own viewport: it is the transform that maps this
        picture's pixels back onto the declared axes, so the hint describes the
        picture rather than a range derived beside it.
        """

        front = None if self._board is None else self._board.front_frame
        panels = tuple(getattr(front, "panels", ()) or ())
        if not panels:
            return None
        viewport = getattr(panels[0].display_payload, "viewport", None)
        axes = tuple(getattr(viewport, "axes", ()) or ())
        bounds = tuple(getattr(viewport, "visible_bounds", ()) or ())
        if len(axes) != 2 or len(bounds) != 4:
            return None
        y_axis, x_axis = axes
        x_size = float(getattr(x_axis, "size", 0) or 0)
        y_size = float(getattr(y_axis, "size", 0) or 0)
        left, top, right, bottom = (float(value) for value in bounds)
        return (left * x_size, right * x_size, top * y_size, bottom * y_size)

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
        self._seed_clim_boxes()          # keep the 2D colour-range boxes in step with the clim pin

    def refresh_limit_hints(self) -> None:
        """Refresh ONLY the grey PLACEHOLDER of the x-range boxes to the panel's current x-window -- a
        non-destructive live reference.  Qt shows a placeholder ONLY while the box is empty, so this can
        never overwrite a pinned value or the operator's in-progress typing.  This is the tick-safe
        successor of the old ``fill_limits`` (whose per-tick ``setText`` clobbered input): the tick calls
        THIS, so the boxes stay a live VIEW of the x-window (unpinned) while remaining freely editable."""
        if self.xmin is None:
            return                          # no Limits controls on this editor instance
        # Colour-range live hint: show the current clim as the grey placeholder so an empty
        # box (= Auto) still tells the operator what range the image is using -- the clim counterpart of
        # the x-window hint below.  Non-destructive: Qt draws a placeholder only while the box is empty,
        # so a pinned/typed value is never overwritten.
        if getattr(self, "clo", None) is not None and self.card is not None:
            shown = self.card._shown_limits()      # None until something has been composed
            if shown is not None:
                self.clo.setPlaceholderText(f"{shown[0]:.6g}")
                self.chi.setPlaceholderText(f"{shown[1]:.6g}")
        bounds = self._front_view_bounds()
        if bounds is None:
            return
        xlo, xhi, ylo, yhi = bounds
        self.xmin.setPlaceholderText(f"{xlo:.6g}"); self.xmax.setPlaceholderText(f"{xhi:.6g}")
        if self.ymin is not None:           # y hint only where y is a view axis (an image family)
            self.ymin.setPlaceholderText(f"{ylo:.6g}"); self.ymax.setPlaceholderText(f"{yhi:.6g}")


    def apply_limits(self) -> None:
        """Apply the typed view-window pins -- PER AXIS: a filled pair pins that axis, an empty pair
        releases it (so 'clear the y boxes + Apply' un-pins y while keeping x).  Each pin is an ORDINARY
        display knob (``view_xlim``/``view_ylim``) applied through the SAME ``_edit_param`` entry as
        bins / relim -> the live card + accepted front + ``config.params`` + save."""
        if self.xmin is None:
            return
        applied, cleared, updates = [], [], {}
        for key, lo_box, hi_box in self._limit_axes():
            lo_text, hi_text = lo_box.text().strip(), hi_box.text().strip()
            if not lo_text and not hi_text:
                updates[key] = None                  # empty pair = release THIS axis's pin
                cleared.append(key[5])               # 'x' / 'y'
                continue
            try:
                lo, hi = float(lo_text), float(hi_text)
            except ValueError as exc:
                self.status.setText(f"bad limits: {exc}")
                return
            updates[key] = (lo, hi)
            applied.append(key[5])
        if self.card is not None:
            self._follow_next_front = True
            self.card._set_params(updates)
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
        updates = {}
        for key, lo_box, hi_box in self._limit_axes():
            updates[key] = None
            with _signals_blocked(lo_box, hi_box):
                lo_box.setText(""); hi_box.setText("")
        if self.card is not None:
            self._follow_next_front = True
            self.card._set_params(updates)
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
        """Pin a 2D panel's colour limit to the typed lo/hi.  Routes through the ONE clim source the
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
            self.status.setText(f"bad colour range: {exc}")
            return
        if self.card is not None:
            self._follow_next_front = True
            self.card.apply_fixed_lims(lo, hi)
        self._sync_relim_combo("fixed")
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

    def save(self) -> None:
        """Write the exact visible pixels and their exact typed data revision."""

        front = None if self._board is None else self._board.front_frame
        if front is None:
            self.status.setText("no snapshot to save")
            return
        try:
            stem = self._save_stem(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            stem.parent.mkdir(parents=True, exist_ok=True)
            target = stem.with_suffix(".%s" % self._save_image_ext())
            image = _front_qimage(front)
            if image is None or not image.save(str(target)):
                raise RuntimeError("Qt refused to write %s" % target.name)
            figure = self._displayed_figure
            display = self._displayed_display
            if figure is None or display is None:
                raise RuntimeError("the editor has no exact typed figure to save")
            figure.save_archive(
                stem.with_suffix(".npz"),
                display=display,
                metadata={
                    "panel_kind": str(self.card.config.kind),
                    "title": str(self.card.config.title or ""),
                },
            )
            self.console._last_save_dir = str(stem.parent)
            self._update_save_preview()
            self.status.setText("saved %s + %s" % (target.name, stem.with_suffix(".npz").name))
            self.status.setToolTip(str(stem.parent))
        except Exception as error:
            self.status.setText("save failed: %s" % error)

