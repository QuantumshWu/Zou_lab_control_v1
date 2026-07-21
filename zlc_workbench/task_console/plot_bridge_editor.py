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


AnalysisControls = _qt_widgets.analysis_controls.AnalysisControls
PARAM_WIDGETS = _qt_widgets.param_widgets.PARAM_WIDGETS
ParamWidgetContext = _qt_widgets.param_widgets.ParamWidgetContext
fill_grouped_signal_combo = _qt_widgets.param_widgets.fill_grouped_signal_combo



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
