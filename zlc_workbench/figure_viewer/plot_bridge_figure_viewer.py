"""Reopen a saved figure ``.npz`` INTO the Task console -- ``exp.figure_viewer()``.

A saved panel / overnight scan writes a current-schema ``<name>_<time>.npz`` artifact
next to its ``.png`` (see :meth:`~.data_figure.DataFigure.save`).  This window is the GUI
counterpart of the notebook one-liner ``na.load_figure('scan.npz')`` -- but instead of a bespoke
viewer, the loaded figure becomes ONE hub SIGNAL and the whole reuse is the Task console board:

* a :class:`LoadedFigureNode` publishes the saved data as static hub signals (``fig_value`` plus, for a
  1-D save its companion ``fig_x``; for a site map its ``fig_centers`` and ``fig_frame`` underlay),
  declaring the SAME ``output_specs`` / ``published_signals`` / ``x_signal`` / ``sitemap_*`` a live
  producer does -- so a panel wired to it reads the right axis label, unit, x-coordinates AND the
  site-map background frame.  Every save records the full producer blocks under ``info['signals']`` (each
  a native ``(repeat, *points, *data)`` array + ``points_shape`` / ``data_shape`` + a ``role``), which
  the node re-publishes VERBATIM (a faithful round-trip); missing or malformed typed blocks reject the
  artifact instead of being reconstructed from ndarray rank;
* the window SEEDS one panel with the SAVED ``kind`` + ``view`` (``PanelConfig(kind=sf.kind,
  source="value = fig_value", params=sf.info["view"])``) so it opens reproducing the original figure;
* the panel lives on a real :class:`~.task_console.TaskConsole`, so the user gets the board, Add Panel,
  the signal picker, re-wiring and the light processing for free -- add MORE panels reading the same
  ``fig_value`` under a different kind, or fit / relim / re-save through the panel's own Edit tab.

A PULSE figure (one whose save carries an ``info['figure_recipe']`` of kind ``pulse`` -- a timeline
rendered from a period table + analog buses, which cannot be expressed as ``data_x`` / ``data_y``) takes
the EXACT SAME path as every other kind, no special case: ``LoadedFigureNode`` publishes its reproduction
OBJECT (a ``PulseTableState``, resolved via :meth:`~.data_figure.SavedFigure.pulse_state`) as ``fig_value``
-- a hub value may be any object, not just an array -- and :func:`_seed_state` seeds a ``kind="pulse"``
panel bound to it, so the SAME :class:`~.task_console.PanelCard` reproduces the full timeline through its
``pulse`` branch (every digital channel / analog trace / bracket).  ``pulse`` is ``panel=False`` ONLY in
the sense that it is not offered in the live Add-Panel dropdown (you do not add a blank pulse panel and
wire it) -- it is a full panel kind on this SEED path.  So the loaded pulse card is a normal PanelCard on
the same board, and the Monitor tab / Add Panel / re-wire reuse is identical to a hist or a site map.

A read-only **Info** column on the left lists EVERY key the npz stored, grouped into tabs -- **Plot**
(name / kind / labels / unit / saved / the view sub-dict / any fit), **Measurement** (source / data_x /
data_y shapes / points / repeat / the stored signal blocks), **Device** (the run's provenance expanded
per device), **Flow** (the upstream DAG of how the data was produced -- raw data / a measurement signal /
through which processor(s), drawn as a branching node-link tree, since one panel can consume several
signals each from a different upstream source) and **Raw** (the whole ``info`` dict verbatim, multi-line
+ scrollable) -- so "what is in this file" is always fully visible next to the board.  Browse (or typing a valid path) loads it
automatically -- there is no separate Load button.  Browse lists the saved-figure IMAGES (``.png`` /
``.jpg``) alongside the ``.npz``, so the operator can eye-ball the thumbnail to find the right run and
picking the image loads its SIBLING ``.npz`` data (the save writes the pair under one ``<name>_<time>``
base); picking the ``.npz`` directly still works.

It is session-INDEPENDENT: it only reads a file, no hardware, no acquisition -- the ``LoadedFigureNode``
re-publishes the same stored arrays every shot.  The window chrome (Fluent frameless shell, one shared
display-scale rule, centred on the primary screen) mirrors ``show_pulse_gui`` / ``show_task_console``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from PyQt5 import QtCore, QtWidgets

from zlc_storage.paths import display_path

from zlc_frontend.qt_widgets import FlowGraphView
from zlc_data.console_records import PanelConfig
from zlc_frontend.console_state import TaskConsoleState
from zlc_workbench.task_console.plot_bridge_console import TaskConsole

from zlc_frontend.qt_widgets import (
    CARD_PAD,
    GREY,
    FluentCodeEdit,
    FluentFrame,
    FluentLabel,
    FluentPathEdit,
    FluentReadoutMultiline,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentStatusStrip,
    FluentTabWidget,
    WINDOW_SCREEN_FRACTION,
    ensure_qt_app,
    launch_fluent_window,
    scaled_px,
    window_pad,
    screen_fit_window_size,
    set_fluent_scale,
    setting_label_width,
    signals_blocked as _signals_blocked,
)


#: The hub signal name the loaded figure's PRIMARY data is published under -- the panel the window
#: seeds is wired to it (``value = fig_value``), and any further panel the user adds picks it too.
FIG_VALUE_KEY = "value"
#: The companion x-axis signal (1-D saves) -- lets a 1-D panel draw its curve vs the saved data_x with
#: the saved x-axis label/unit (the console's ``curve_x_provider`` resolves it from THIS node).
FIG_X_KEY = "x"
#: The per-tweezer centres signal (site-map saves) -- a site-map panel resolves its ring centres from
#: this node via ``sitemap_centers_key``.
FIG_CENTERS_KEY = "centers"
#: The camera-frame underlay signal for site-map/2-D saves.  A site-map
#: panel resolves it from the producer through ``sitemap_image_key``.
FIG_FRAME_KEY = "frame"

#: The node prefix -- so the published hub names are ``fig_value`` / ``fig_x`` / ``fig_centers`` /
#: ``fig_frame`` and the seeded panel's ``inputs`` name ``fig_value``.
FIG_PREFIX = "fig_"

#: The saved-figure IMAGE suffixes ``DataFigure.save`` may write beside the data ``.npz`` (its default is
#: ``.png``; ``image_ext=`` can pick ``.jpg`` / ``.jpeg``).  Browsing/typing one of these loads the
#: SIBLING ``<image>.with_suffix('.npz')`` -- the save writes the pair under the same ``<name>_<time>``
#: base, so the mapping is exact.  Kept next to the data param, not baked into ``FluentPathEdit`` (which
#: stays a generic path field -- the sibling mapping is the viewer's job).
FIGURE_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def _resolve_npz(path: Path) -> Path:
    """Map a picked path to the ``.npz`` to load: a saved-figure IMAGE (``.png`` / ``.jpg`` / ``.jpeg``)
    resolves to its SIBLING ``<image>.with_suffix('.npz')`` (the save writes the image + npz as a same-base
    pair); a ``.npz`` (or anything else) is returned unchanged.  The caller checks the result exists."""
    if path.suffix.lower() in FIGURE_IMAGE_SUFFIXES:
        return path.with_suffix(".npz")
    return path


def _stored_shape(entry: Mapping, key: str, signal_name: str) -> tuple[int, ...]:
    """Read one mandatory non-empty stored tensor shape.

    ``info['signals']`` is the faithful typed path, so a present entry is never allowed to fall back to
    ndarray-rank inference.  Array-only saves use the separate ``_build_from_arrays`` boundary; a typed
    entry with a missing shape is malformed and must say so explicitly.
    """
    raw = entry.get(key)
    if raw is None:
        raise ValueError(f"saved signal {signal_name!r} is missing required {key}.")
    shape = tuple(int(n) for n in raw)
    if not shape or any(n < 1 for n in shape):
        raise ValueError(
            f"saved signal {signal_name!r} has invalid {key}={raw!r}; "
            "stored tensor shapes must be non-empty and positive.")
    return shape


def _kind_label(key: str | None) -> str:
    """The human plot-kind label from the ONE vocabulary (never a hand-typed name), so the
    Info panel reads 'Distribution' not 'hist'; falls back to the raw key for an unknown kind."""
    if not key:
        return ""
    from zlc_data.plot_kind import PLOT_KIND_SPEC_BY_KEY
    spec = PLOT_KIND_SPEC_BY_KEY.get(str(key))
    return spec.label if spec is not None else str(key)






class FigureViewer(QtWidgets.QWidget):
    """The reopen-into-the-console body (wrapped in a :class:`FluentWindow` by :func:`show_figure_viewer`).

    Left:  a read-only **Info** column -- a path field (Browse auto-loads a picked ``.npz``, no Load
           button) over a TAB set that groups the stored facts: **Plot** (how it draws) / **Measurement**
           (its data shapes / source / stored signal blocks) / **Device** (the run's provenance) / **Raw**
           (the whole ``info`` dict, multi-line + scrollable), so the entire npz is visible next to the
           board without one crushed column.
    Right: a live embedded :class:`~.task_console.TaskConsole` (size-Expanding, so it FILLS the pane and
           reflows into 2+ columns) whose seeded panel reproduces the saved figure and on which the user
           can Add more panels / re-wire / process the loaded ``fig_value`` signal.  The Info card and the
           console share one root layout with equal margins + spacing, so their outer edges line up.
    """


    # ------------------------------------------------------------------ layout
    def _build_info_column(self) -> QtWidgets.QWidget:
        # The Info column MIRRORS the console beside it: a plain header bar (the path picker) ABOVE a
        # flat-bordered tab card -- TWO separate SIBLING fluent cards, NOT a tab card nested inside an
        # outer frame (nesting a bordered card inside another bordered card would double the edge).  The
        # column's margins / header height / header<->tab gap match the console's (window_pad(1) / 48 /
        # window_pad(0.5)) so the two header bars and the two tab cards line up row-for-row across the
        # divider -- and the column's L/R window pad matches every other GUI's, title-aligned.
        col = QtWidgets.QWidget()
        col.setStyleSheet("background: transparent;")
        col.setFixedWidth(self._info_col_w)
        lay = QtWidgets.QVBoxLayout(col)
        lay.setContentsMargins(window_pad(1), 0, window_pad(1), 0)   # == console root margins -> aligned
        lay.setSpacing(window_pad(0.5))                              # == console header<->tab gap

        # --- File header card: FLAT (like the console header), a single bar "File" + path picker --------
        # Browse (or typing a valid .npz) AUTO-LOADS -- no separate Load button: FluentPathEdit.changed
        # fires when Browse picks a file (or the user finishes typing one) and _on_path_changed loads it.
        # Browse lists the saved-figure IMAGES (png / jpg) too, so the operator can eye-ball a thumbnail;
        # picking the image loads its SIBLING ``<image>.with_suffix('.npz')`` data (the save writes the
        # pair).  A FLAT header (no shadow) mirrors the console header, which is flat so its shadow's soft
        # bottom edge never draws a thin line into the gap above the tab strip.
        header_frame = FluentFrame(bordered=False)
        header_frame.setFixedHeight(scaled_px(48, minimum=38))
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(scaled_px(12), scaled_px(6), scaled_px(12), scaled_px(6))
        header.setSpacing(scaled_px(8, minimum=5))
        header.addWidget(FluentSectionLabel("File"))
        self.path_edit = FluentPathEdit(
            "", mode="file", caption="Open a saved figure (image or .npz)",
            file_filter="Saved figures (*.png *.jpg *.jpeg *.npz);;All files (*)")
        self.path_edit.setToolTip("A saved figure -- Browse (or type a full path).  Pick the image "
                                  "(.png / .jpg) or the .npz and it loads onto the board automatically.")
        self.path_edit.changed.connect(self._on_path_changed)
        header.addWidget(self.path_edit, 1)
        lay.addWidget(header_frame)

        # --- Info tabs card: its OWN flat-bordered card --------------------------------------------------
        # Plot (how it draws) | Measurement (data shapes / source) | Device (run provenance) | Flow (the
        # upstream DAG of how the data was produced) | Raw (the whole dict).  The tab card carries a flat
        # 1 px border (no drop shadow), so no top-bleed headroom is reserved -- the row spacing is the gap.
        self.info_tabs = FluentTabWidget()
        self.info_tabs.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.plot_layout = self._add_rows_tab("Plot")
        self.meas_layout = self._add_rows_tab("Measurement")
        # ``info_layout`` is the DEVICE tab body -- the provenance expansion lands here (kept under this
        # name so _fill_provenance / _add_info_row + the provenance tests read one Info container).
        self.info_layout = self._add_rows_tab("Device")
        # The Flow tab: the upstream DAG of how the data was produced (raw / measurement / processor chain,
        # BRANCHING upward), drawn by the reusable FlowGraphView inside a scroll area (a large graph scrolls).
        self.flow_view = self._add_flow_tab("Flow")
        self.raw_info = self._add_raw_tab("Raw")
        lay.addWidget(self.info_tabs, 1)

        # The SAME persistent status surface the console mounts (FluentStatusStrip): one
        # always-visible line, severity-coloured, eliding with the full text in the tooltip --
        # a load failure turns it red instead of hiding as grey prose.
        self.status = FluentStatusStrip()
        self.status.show_message("Load a saved figure (.npz) to view it on the board.")
        lay.addWidget(self.status)
        return col

    def _add_rows_tab(self, title: str) -> QtWidgets.QVBoxLayout:
        """A permanent tab whose body is a top-aligned rows column inside a vertical FluentScrollArea (so
        a long list of facts scrolls) -- returns the body layout for the filler to add rows to."""
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        body.setStyleSheet("background: transparent;")
        vbox = QtWidgets.QVBoxLayout(body)
        # Body inset == the component-library card padding (CARD_PAD), so the Info tab card's content
        # sits off its edge by the SAME margin as the console's own tab bodies (the Logic / Edit tabs use
        # scaled_px(CARD_PAD, minimum=6)) -- the left and right tab cards read identically, not with the
        # Info column crammed tighter against its edge.
        m = scaled_px(CARD_PAD, minimum=6)
        vbox.setContentsMargins(m, m, m, m)
        vbox.setSpacing(scaled_px(3, minimum=2))
        vbox.setAlignment(QtCore.Qt.AlignTop)
        scroll.setWidget(body)
        self.info_tabs.add_permanent_tab(scroll, title)
        return vbox

    def _add_flow_tab(self, title: str) -> FlowGraphView:
        """A permanent tab holding the reusable :class:`FlowGraphView` inside a horizontal+vertical
        FluentScrollArea -- a branching provenance graph that outgrows the column simply scrolls (the view
        reports its natural size, so ``setWidgetResizable(True)`` lets it grow past the viewport)."""
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        view = FlowGraphView()
        view.setToolTip("How this figure's data was produced: raw data / a measurement signal / through "
                        "which processor(s).  The tree branches upward -- a panel can consume several "
                        "signals, each from a different upstream source.")
        scroll.setWidget(view)
        self.info_tabs.add_permanent_tab(scroll, title)
        return view

    def _add_raw_tab(self, title: str) -> FluentCodeEdit:
        """A permanent tab holding a read-only multi-line code editor (its own scrollbars) that shows the
        WHOLE stored info dict verbatim -- so a long / nested value is fully readable, never truncated.

        The editor lives in a transparent body inset by the SAME component-library card padding as the
        rows tabs (:meth:`_add_rows_tab`), so the Raw tab card's content sits off its edge by CARD_PAD
        exactly like every other Info / console tab -- not flush against the card edge."""
        body = QtWidgets.QWidget()
        body.setStyleSheet("background: transparent;")
        vbox = QtWidgets.QVBoxLayout(body)
        m = scaled_px(CARD_PAD, minimum=6)
        vbox.setContentsMargins(m, m, m, m)
        raw = FluentCodeEdit("", read_only=True)
        raw.setToolTip("The full info dict stored in the npz -- read-only, select to copy.")
        vbox.addWidget(raw)
        self.info_tabs.add_permanent_tab(body, title)
        return raw

    # -------------------------------------------------------------- public API
    def window(self):
        return getattr(self, "_zlc_window", None)

    def open_path(self, path: str | Path) -> None:
        """Load a saved figure: publish its data as ``fig_value`` and seed a console reproducing it.

        ``path`` may be the data ``.npz`` OR its sibling IMAGE (``.png`` / ``.jpg`` / ``.jpeg``) -- picking
        the image loads the same-base ``.npz`` beside it (see :func:`_resolve_npz`).  A picked image with
        no sibling ``.npz`` is reported in the status line (nothing is loaded), never a crash."""
        p = Path(str(path).strip())
        if str(p) in ("", "."):
            return
        npz = _resolve_npz(p)
        if p.suffix.lower() in FIGURE_IMAGE_SUFFIXES and not npz.is_file():
            self.status.show_message(f"no matching .npz data next to {display_path(str(p))} "
                                     f"(expected {npz.name})", severity="warning")
            return
        self._load_npz(npz)

    # ------------------------------------------------------------- file / load
    def _on_path_changed(self, text: str) -> None:
        """Browse (or typed) path -> AUTO-LOAD.  ``changed`` fires on every keystroke too, so we only act
        on a real EXISTING file (mid-type garbage / a folder is ignored, no error spam) whose suffix is a
        saved-figure one -- a ``.npz`` loads directly; a saved-figure IMAGE (``.png`` / ``.jpg`` /
        ``.jpeg``) loads its SIBLING ``.npz`` (see :func:`_resolve_npz`).  Browse always yields a real
        file, so picking the image OR the npz loads immediately; an image with no sibling npz reports it in
        the status (never a crash)."""
        p = Path(str(text).strip())
        suffix = p.suffix.lower()
        if str(p) in ("", ".") or not p.is_file():
            return
        if suffix != ".npz" and suffix not in FIGURE_IMAGE_SUFFIXES:
            return                                       # mid-type / unrelated file -- no error spam
        npz = _resolve_npz(p)
        if not npz.is_file():
            self.status.show_message(f"no matching .npz data next to {display_path(str(p))} "
                                     f"(expected {npz.name})", severity="warning")
            return
        if self._current_path is not None and npz == self._current_path:
            return                                       # already loaded (the setText echo below re-fires)
        self._load_npz(npz)


    def _clear_layout(self, layout: QtWidgets.QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _clear_info(self) -> None:
        for layout in (self.plot_layout, self.meas_layout, self.info_layout):
            self._clear_layout(layout)
        self.flow_view.set_graph(None)
        self.raw_info.clear()
        self.raw_info.setToolTip("")


    def _fill_rows(self, layout: QtWidgets.QVBoxLayout, rows: list[tuple[str, object]]) -> None:
        for key, value in rows:
            if value is None:
                continue
            layout.addWidget(FluentSettingRow(key, self._readout_field(value),
                                              label_width=self._label_w))

    def _add_info_row(self, key: str, value: object) -> None:
        """One read-only ``key | value`` row in the DEVICE tab (the shared row primitive)."""
        self.info_layout.addWidget(FluentSettingRow(key, self._readout_field(value),
                                                    label_width=self._label_w))

    @staticmethod
    def _readout_field(value: object) -> FluentReadoutMultiline:
        """The ONE read-only value control for an Info row: a :class:`FluentReadoutMultiline` that SOFT-
        WRAPS a long value (a resolved path, a device-metadata blob, a data shape) over as many lines as
        it needs instead of a single-line edit clipping it.  Ignored horizontal policy so the row/column
        drives the width; the field auto-sizes its OWN height to the wrapped content (up to its cap).

        A collection is rendered HUMAN-readably here (the ONE place, so no row can leak a Python repr): a
        list -> comma-joined items, a dict -> ``k=v`` pairs, an ndarray -> ``array(shape) dtype`` -- NOT
        ``['a', 'b']`` / ``{'a': 1}`` with quotes and braces.  A plain TUPLE stays as-is because a shape
        (``(1920, 1080)``) already reads cleanly; a scalar is just ``str``."""
        if isinstance(value, Mapping):
            text = ", ".join(f"{k}={v}" for k, v in value.items())
        elif isinstance(value, list):
            text = ", ".join(str(v) for v in value)
        elif isinstance(value, np.ndarray):
            text = f"array{tuple(value.shape)} {value.dtype}"
        else:
            text = str(value)
        field = FluentReadoutMultiline(text)
        field.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        return field

    def _fill_provenance(self, provenance: object) -> None:
        """Expand the saved ``info['provenance']`` (the producing node's device snapshot) into the
        Info column, easy to read: a 'Provenance' section header, then the top-level scalar facts
        (node / layer / captured_at / calibration fingerprint) as rows, then ONE sub-section per held
        device (``camera`` / ``sequencer``) listing its snapshot keys one per row.  ``provenance`` may
        be ``None`` (old npz, or nothing to record) -> nothing is added; a non-dict is shown verbatim."""
        if not provenance:
            return
        self.info_layout.addWidget(FluentSectionLabel("Provenance"))
        if not isinstance(provenance, Mapping):
            self._add_info_row("provenance", provenance)
            return
        devices = provenance.get("devices") if isinstance(provenance.get("devices"), Mapping) else {}
        # Top-level scalar facts first (skip the nested ``devices`` -- it gets its own sub-sections).
        for key, value in provenance.items():
            if key == "devices":
                continue
            if isinstance(value, Mapping):       # e.g. acquisition_parameters / calibration_fingerprint
                self.info_layout.addWidget(FluentSectionLabel(str(key)))
                for sub_key, sub_val in value.items():
                    self._add_info_row(str(sub_key), sub_val)
            else:
                self._add_info_row(str(key), value)
        # One sub-section per device, its snapshot keys one row each.
        for role, snap in devices.items():
            self.info_layout.addWidget(FluentSectionLabel(str(role)))
            if isinstance(snap, Mapping):
                for sub_key, sub_val in snap.items():
                    self._add_info_row(str(sub_key), sub_val)
            else:
                self._add_info_row(str(role), snap)

    # -------------------------------------------------------------- console

    def _teardown_console(self) -> bool:
        if self.console is not None:
            try:
                if self.console.shutdown() is not True:
                    return False
            except BaseException:
                return False
            self._console_holder.removeWidget(self.console)
            self.console.deleteLater()
            self.console = None
        if self.node is not None:
            try:
                if self.node.stop(timeout=5.0) is not True or self.node.running:
                    return False
            except BaseException:
                return False
            self.node = None
        return True

    def teardown(self) -> bool:
        return self._teardown_console()

    # ---------------------------------------------------------------- sizing
    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API name
        # The window opens at the SAME screen fraction as the other two top-level GUIs -- the task
        # console (``TaskConsole.sizeHint``) and the pulse editor both return ``screen_fit_window_size``
        # VERBATIM from their ``sizeHint``.  The size is the screen BUDGET, never the content: the inner
        # layout (fixed Info column + a console board sized to fill the rest) fits inside it, so the
        # window can't collapse to a bare Info strip whether or not a figure is loaded.  A Python
        # exception from this Qt-virtual would crash the C++ adjustSize, so self-swallow.
        try:
            return screen_fit_window_size(self.window_ratio)
        except Exception:
            return super().sizeHint()


def show_figure_viewer(path: str | Path | None = None, *, scale: float | None = None,
                       window_ratio: float = WINDOW_SCREEN_FRACTION,
                       hide_on_close: bool = False) -> FigureViewer:
    """Open the saved-figure viewer in a Fluent window (mirrors ``show_pulse_gui`` /
    ``show_task_console``: the body sizes from the primary screen, the window wraps it, and the shared
    auto-scale rule owns the on-screen control size).  ``path`` loads a ``.npz`` on launch -- its data
    becomes the ``fig_value`` hub signal and a panel of the saved kind opens reproducing it on a real
    Task console board; omit it to open empty and Browse from inside.  Closing the window tears the
    embedded console down (its refresh timer + the loaded-figure node)."""
    ensure_qt_app()          # the viewer is a QWidget: the app must exist BEFORE its ctor
    viewer = FigureViewer(path, scale=scale, window_ratio=window_ratio)
    # ONE launcher sequence (launch_fluent_window: wrap -> wire -> size -> centre -> show ->
    # retain), shared with every other show_* GUI so the steps cannot drift per-launcher.
    def _wire(window):
        # ``closed`` is a post-commit notification and cannot veto destruction.  A genuine close
        # therefore uses the acknowledgement-bearing guard; hide-on-close intentionally preserves
        # the static viewer exactly as before.
        if not hide_on_close:
            window.set_close_guard(viewer.teardown)

    window = launch_fluent_window(
        viewer,
        title="FigureViewer@Zou lab",
        hide_on_close=hide_on_close,
        wire=_wire,
    )
    # The embedded console's scroll viewport only has its REAL width AFTER the window is shown (0 during
    # construction).  Re-pack its board now so it lays out against the true pane width immediately,
    # rather than waiting for the first resize event.
    if viewer.console is not None:
        viewer.console._arrange_if_cards()
    viewer._zlc_window = window
    return viewer


__all__ = ["FigureViewer", "LoadedFigureNode", "show_figure_viewer"]
