"""Standalone Fluent window for reopening a saved figure ``.npz`` -- ``exp.figure_viewer()``.

A saved panel / overnight scan writes a ``<name>_<time>.npz`` (``data_x`` / ``data_y`` / ``info``)
next to its ``.png`` (see :meth:`~.data_figure.DataFigure.save`).  This window is the GUI
counterpart of the notebook one-liner ``na.load_figure('scan.npz')``: pick a ``.npz`` (or a folder
full of them -- a calibration run drops one per site), see WHAT it holds (the ``info`` panel), RE-VIEW
it as any :meth:`~.data_figure.SavedFigure.compatible_kinds` plot kind, tune the same DataFigure knobs
the Task console's Edit tab exposes (relim / fixed lo-hi / unit / fit / manual limits) and RE-SAVE.

It is a PURE VIEWER: no hardware, no session, no acquisition.  Every control drives the ONE
:class:`~.data_figure.DataFigure` the reopened figure renders through -- the same reusable
fit / unit / re-save stack the live panels use -- so the viewer adds NO new plotting art.  The
window chrome (Fluent frameless shell, one shared display-scale rule, centred on the primary
screen) mirrors ``show_pulse_gui`` / ``show_task_console`` exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import time

from PyQt5 import QtCore, QtWidgets

from .data_figure import DataFigure, SavedFigure, VALID_FIT_FUNCS, load_figure
from .qt_fluent import (
    ACCENT,
    GREY,
    ORANGE,
    FluentButton,
    FluentComboBox,
    FluentLabel,
    FluentLineEdit,
    FluentPathEdit,
    FluentReadoutEdit,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentWindow,
    WINDOW_SCREEN_FRACTION,
    center_window_on_primary_screen,
    ensure_qt_app,
    scaled_px,
    screen_fit_window_size,
    set_fluent_scale,
    setting_label_width,
    signals_blocked as _signals_blocked,
)

try:  # matplotlib is a frontend dependency, but keep import failures tidy (headless-safe).
    import matplotlib.pyplot as plt
    from .qt_canvas import panel_canvas
except Exception:  # pragma: no cover - depends on the local desktop environment
    plt = None
    panel_canvas = None

from Zou_lab_control._paths import display_path


#: relim modes -- the SAME confocal ``combo_relim`` vocabulary the Task console uses (tight / normal /
#: fixed).  "fixed" pins the y-axis / colour limit to the operator-set lo/hi; the others autoscale.
_RELIM_MODES = ("tight", "normal", "fixed")

#: Human curve-fit model name -> the DataFigure method it calls -- the SAME set the Task console's Edit
#: tab offers (:data:`~.data_figure.VALID_FIT_FUNCS`), just given readable menu labels.  Read-only view:
#: a 2-D save fits "2D center", a 1-D save fits the rest (DataFigure gates by its own plot_type).
_FIT_MODELS: dict[str, str] = {
    "Lorentzian": "lorent",
    "Gaussian": "gaussian",
    "Lorentzian (Zeeman)": "lorent_zeeman",
    "Rabi": "rabi",
    "Exp decay": "decay",
    "2D center": "center",
}
assert set(_FIT_MODELS.values()) == set(VALID_FIT_FUNCS), "fit model map must cover VALID_FIT_FUNCS"


def _safe_float(text, default: float = 0.0) -> float:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return default


def _kind_label(key: str | None) -> str:
    """The human plot-kind label from the ONE ``PLOT_KINDS`` table (never a hand-typed name), so the
    kind combo reads 'Distribution' not 'hist'; falls back to the raw key for an unknown kind."""
    if not key:
        return ""
    from .live import PLOT_KIND_BY_KEY
    pk = PLOT_KIND_BY_KEY.get(str(key))
    return pk.label if pk is not None else str(key)


class FigureViewer(QtWidgets.QWidget):
    """The reopen-a-saved-figure body (wrapped in a :class:`FluentWindow` by :func:`show_figure_viewer`).

    Left:  a folder GALLERY (every ``.npz`` beside the opened one, so a calibration folder is browsable
           one site at a time) above the ``Info`` panel (what the file holds, one row per fact).
    Centre: the reopened figure on a :func:`panel_canvas` (interactive: zoom / area select).
    Right: the SAME DataFigure knobs as the Task console Edit tab -- plot kind, relim + fixed lo/hi,
           unit cycle, curve fit, manual limits, and re-save -- all driving the ONE reopened DataFigure.
    """

    def __init__(self, path: str | Path | None = None, *, scale: float | None = None,
                 window_ratio: float = WINDOW_SCREEN_FRACTION, parent=None):
        super().__init__(parent)
        set_fluent_scale(scale)                   # the ONE shared GUI-scale rule (mirrors the other GUIs)
        self.window_ratio = float(window_ratio)
        self.setStyleSheet("background: transparent;")

        self._saved: SavedFigure | None = None    # the loaded npz handle
        self._df: DataFigure | None = None        # the reopened DataFigure the controls drive
        self._canvas = None                       # the embedded panel_canvas
        self._gallery_dir: Path | None = None     # the folder whose npz files fill the gallery
        self._current_path: Path | None = None    # the npz currently shown
        self._last_save_dir: str | None = None    # remember where the last re-save landed (this session)

        label_w = setting_label_width(
            ["name", "source", "kind", "labels", "unit", "data_x", "data_y",
             "points", "saved", "path", "plot kind", "relim", "lo / hi", "unit", "model", "args"])
        self._label_w = label_w

        root = QtWidgets.QHBoxLayout(self)
        m = scaled_px(10, minimum=6)
        root.setContentsMargins(m, m, m, m)
        root.setSpacing(scaled_px(10, minimum=6))

        root.addWidget(self._build_left_column(), 0)
        root.addWidget(self._build_centre_column(), 1)
        root.addWidget(self._build_right_column(), 0)

        if path is not None:
            self.open_path(path)

    # ------------------------------------------------------------------ layout
    def _column_width(self) -> int:
        """The fixed width of each side column: the shared label column + room for the widest control
        (a path field + Browse, a combo), derived from ``setting_label_width`` so it SCALES with the
        GUI and never clips a row (a hard ``setFixedWidth`` narrower than the content was the clip)."""
        return self._label_w + scaled_px(200, minimum=150)

    def _build_left_column(self) -> QtWidgets.QWidget:
        col = QtWidgets.QWidget()
        col.setStyleSheet("background: transparent;")
        col.setFixedWidth(self._column_width())
        lay = QtWidgets.QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(scaled_px(6, minimum=4))

        # --- top: path field + Load (pick a .npz or a folder) ------------------
        lay.addWidget(FluentSectionLabel("File"))
        self.path_edit = FluentPathEdit(
            "", mode="file", caption="Open a saved figure (.npz) or its folder",
            file_filter="Saved figures (*.npz);;All files (*)")
        self.path_edit.setToolTip("A saved figure .npz (or a folder of them) -- type a path or Browse.")
        self.path_edit.changed.connect(self._on_path_changed)
        lay.addWidget(FluentSettingRow("path", self.path_edit, label_width=self._label_w))
        load_row = QtWidgets.QHBoxLayout()
        load_row.setContentsMargins(0, 0, 0, 0)
        load_row.setSpacing(scaled_px(6, minimum=4))
        self.load_button = FluentButton("Load", color=ACCENT)
        self.load_button.setToolTip("Open the .npz (or the first .npz in the chosen folder).")
        self.load_button.clicked.connect(self._on_load_clicked)
        folder_button = FluentButton("Folder…", color=GREY)
        folder_button.setToolTip("Pick a FOLDER; its .npz files fill the gallery below.")
        folder_button.clicked.connect(self._pick_folder)
        load_row.addWidget(self.load_button, 0)
        load_row.addWidget(folder_button, 0)
        load_row.addStretch(1)
        lay.addLayout(load_row)

        # --- gallery: one button per .npz in the folder ------------------------
        lay.addWidget(FluentSectionLabel("Gallery"))
        self.gallery_scroll = FluentScrollArea()
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_scroll.setFixedHeight(scaled_px(150, minimum=108))
        gallery_body = QtWidgets.QWidget()
        gallery_body.setStyleSheet("background: transparent;")
        self.gallery_layout = QtWidgets.QVBoxLayout(gallery_body)
        self.gallery_layout.setContentsMargins(0, 0, 0, 0)
        self.gallery_layout.setSpacing(scaled_px(3, minimum=2))
        self.gallery_layout.setAlignment(QtCore.Qt.AlignTop)
        self.gallery_scroll.setWidget(gallery_body)
        lay.addWidget(self.gallery_scroll)
        self._gallery_buttons: list[FluentButton] = []

        # --- info panel: what the file holds ----------------------------------
        lay.addWidget(FluentSectionLabel("Info"))
        info_scroll = FluentScrollArea()
        info_scroll.setWidgetResizable(True)
        info_body = QtWidgets.QWidget()
        info_body.setStyleSheet("background: transparent;")
        self.info_layout = QtWidgets.QVBoxLayout(info_body)
        self.info_layout.setContentsMargins(0, 0, 0, 0)
        self.info_layout.setSpacing(scaled_px(3, minimum=2))
        self.info_layout.setAlignment(QtCore.Qt.AlignTop)
        info_scroll.setWidget(info_body)
        lay.addWidget(info_scroll, 1)

        # Raw info: EVERY key in the npz info dict, verbatim (answers "what is actually stored").
        self.raw_info = FluentReadoutEdit("")
        self.raw_info.setToolTip("The full info dict stored in the npz -- read-only, select to copy.")
        self.raw_info.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        lay.addWidget(FluentSettingRow("raw info", self.raw_info, label_width=self._label_w))
        return col

    def _build_centre_column(self) -> QtWidgets.QWidget:
        col = QtWidgets.QWidget()
        col.setStyleSheet("background: transparent;")
        lay = QtWidgets.QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(scaled_px(4, minimum=3))
        self.canvas_holder = QtWidgets.QVBoxLayout()      # FIXED holder: rebuild swaps the canvas in place
        self.canvas_holder.setContentsMargins(0, 0, 0, 0)
        self.canvas_holder.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        lay.addLayout(self.canvas_holder)
        lay.addStretch(1)
        self.placeholder = FluentLabel("Load a saved figure (.npz) to view it here.")
        self.placeholder.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas_holder.addWidget(self.placeholder)
        return col

    def _build_right_column(self) -> QtWidgets.QWidget:
        col = QtWidgets.QWidget()
        col.setStyleSheet("background: transparent;")
        col.setFixedWidth(self._column_width())
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        page = QtWidgets.QWidget()
        page.setStyleSheet("background: transparent;")
        scroll.setWidget(page)
        outer = QtWidgets.QVBoxLayout(col)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        c = QtWidgets.QVBoxLayout(page)
        c.setContentsMargins(0, 0, 0, 0)
        c.setSpacing(scaled_px(6, minimum=4))
        lw = self._label_w

        # ---- Display: plot kind + relim + fixed lo/hi + unit cycle ------------
        c.addWidget(FluentSectionLabel("Display"))
        self.kind_combo = FluentComboBox()
        self.kind_combo.setToolTip("View the SAME data as another compatible plot kind (the saved kind is first).")
        self.kind_combo.currentIndexChanged.connect(self._on_kind)
        c.addWidget(FluentSettingRow("plot kind", self.kind_combo, label_width=lw))

        self.relim_combo = FluentComboBox()
        self.relim_combo.addItems(list(_RELIM_MODES))
        self.relim_combo.setToolTip(
            "Relim mode (confocal combo_relim naming):\n"
            "  tight  = autoscale hugs the data\n"
            "  normal = autoscale with the matplotlib default margin\n"
            "  fixed  = pin the y-axis / colour-limit to the lo/hi below")
        self.relim_combo.currentTextChanged.connect(self._on_relim)
        c.addWidget(FluentSettingRow("relim", self.relim_combo, label_width=lw))

        # fixed lo/hi: ONE [lo | hi] row that is ALWAYS in the layout -- only its inputs enable when
        # relim == "fixed" (a setVisible toggle above the canvas would reflow the whole page on every
        # relim change; greying in place leaves every widget put -- the same rule as the Edit tab).
        self.fixed_lo = FluentLineEdit("0.0")
        self.fixed_hi = FluentLineEdit("1.0")
        for ed in (self.fixed_lo, self.fixed_hi):
            ed.setMinimumWidth(scaled_px(64, minimum=52))
            ed.editingFinished.connect(self._on_fixed_lim)
        self.fixed_lo.setToolTip("Fixed lower limit (used only when relim = fixed)")
        self.fixed_hi.setToolTip("Fixed upper limit (used only when relim = fixed)")
        fixed_host = QtWidgets.QWidget()
        fl = QtWidgets.QHBoxLayout(fixed_host)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(scaled_px(6, minimum=4))
        fl.addWidget(self.fixed_lo, 1)
        fl.addWidget(self.fixed_hi, 1)
        self.fixed_row = FluentSettingRow("lo / hi", fixed_host, label_width=lw)
        c.addWidget(self.fixed_row)

        unit_host = QtWidgets.QWidget()
        ul = QtWidgets.QHBoxLayout(unit_host)
        ul.setContentsMargins(0, 0, 0, 0)
        ul.setSpacing(scaled_px(6, minimum=4))
        self.unit_button = FluentButton("Unit", color=GREY)
        self.unit_button.setFixedWidth(scaled_px(70, minimum=56))
        self.unit_button.setToolTip(
            "Cycle the x-axis unit (GHz/nm/MHz or ns/us/ms) where the axis label\n"
            "declares a convertible unit; otherwise a no-op")
        self.unit_button.clicked.connect(self._on_unit_cycle)
        self.unit_label = FluentLabel("")
        self.unit_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.unit_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        ul.addWidget(self.unit_button, 0)
        ul.addStretch(1)
        ul.addWidget(self.unit_label, 0)
        c.addWidget(FluentSettingRow("unit", unit_host, label_width=lw))

        # ---- Fit: model + args + Fit / Clear ---------------------------------
        c.addWidget(FluentSectionLabel("Fit"))
        self.fit_combo = FluentComboBox()
        self.fit_combo.addItems(list(_FIT_MODELS))
        self.fit_combo.setToolTip(
            "Curve-fit model (the full DataFigure set).  '2D center' fits a 2D image\n"
            "(2D Gaussian); the others fit the 1D trace.")
        fit_btn = FluentButton("Fit", color=ACCENT)
        fit_btn.clicked.connect(self.do_fit)
        clear_btn = FluentButton("Clear", color=GREY)
        clear_btn.clicked.connect(self.clear_fit)
        model_host = QtWidgets.QWidget()
        mh = QtWidgets.QHBoxLayout(model_host)
        mh.setContentsMargins(0, 0, 0, 0)
        mh.setSpacing(scaled_px(6, minimum=4))
        mh.addWidget(self.fit_combo, 1)
        mh.addWidget(fit_btn, 0)
        mh.addWidget(clear_btn, 0)
        c.addWidget(FluentSettingRow("model", model_host, label_width=lw))
        self.fit_args = FluentLineEdit("")
        self.fit_args.setPlaceholderText("p0=[1,0,1,0], B=0.1, is_fit=False")
        self.fit_args.setToolTip(
            "Optional fit arguments injected into the call (trusted local code):\n"
            "p0=[...] initial guess; NAME=value fixes a named parameter; is_fit=False just overlays p0.")
        self.fit_args.returnPressed.connect(self.do_fit)
        c.addWidget(FluentSettingRow("args", self.fit_args, label_width=lw))

        # ---- Limits: manual x / y ranges -------------------------------------
        c.addWidget(FluentSectionLabel("Limits"))
        self.xmin = FluentLineEdit(""); self.xmax = FluentLineEdit("")
        self.ymin = FluentLineEdit(""); self.ymax = FluentLineEdit("")
        for w in (self.xmin, self.xmax, self.ymin, self.ymax):
            w.setFixedWidth(scaled_px(88, minimum=68))
            w.returnPressed.connect(self.apply_limits)
        apply_btn = FluentButton("Apply lim", color=ACCENT)
        apply_btn.clicked.connect(self.apply_limits)
        c.addWidget(FluentSettingRow("x range", self._inline(self.xmin, self.xmax), label_width=lw))
        c.addWidget(FluentSettingRow("y range", self._inline(self.ymin, self.ymax, trailing=apply_btn),
                                     label_width=lw))

        # ---- Save: re-save the (edited) figure -------------------------------
        c.addWidget(FluentSectionLabel("Save"))
        self.save_dir_edit = FluentPathEdit(
            "", mode="dir", caption="Choose where to save", base_dir="")
        self.save_dir_edit.setToolTip(
            "Where to re-save (folder, or a full path base).  Defaults beside the opened file.")
        self.save_dir_edit.changed.connect(lambda *_: self._update_save_preview())
        c.addWidget(FluentSettingRow("path", self.save_dir_edit, label_width=lw))
        self.save_button = FluentButton("Save Fig", color=ACCENT)
        self.save_button.setToolTip("Save the current view as a new png + matching npz.")
        self.save_button.clicked.connect(self.save)
        c.addWidget(FluentSettingRow("name", self._inline(self.save_button), label_width=lw))
        self.save_preview = FluentReadoutEdit("")
        self.save_preview.setToolTip("The file the next Save writes -- read-only, select to copy.")
        self.save_preview.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        c.addWidget(FluentSettingRow("file", self.save_preview, label_width=lw))

        self.status = FluentLabel("")
        self.status.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        c.addWidget(self.status)
        c.addStretch(1)
        return col

    def _inline(self, *widgets, trailing=None) -> QtWidgets.QWidget:
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

    # -------------------------------------------------------------- public API
    def window(self):
        return getattr(self, "_zlc_window", None)

    def open_path(self, path: str | Path) -> None:
        """Open a ``.npz`` (render it) OR a folder (fill the gallery + open its first ``.npz``)."""
        p = Path(str(path).strip())
        if str(p) in ("", "."):
            return
        if p.is_dir():
            self._populate_gallery(p, select_first=True)
            return
        # a bare .npz: its siblings fill the gallery, and it is the selected one
        if p.suffix.lower() == ".npz" and p.parent.is_dir():
            self._populate_gallery(p.parent, select=p)
        else:
            self._load_npz(p)

    # ------------------------------------------------------------- file / load
    def _on_path_changed(self, text: str) -> None:
        # typing/browsing a path only ARMS Load; nothing renders until Load (mirrors confocal's load()).
        pass

    def _on_load_clicked(self) -> None:
        text = self.path_edit.text().strip()
        if not text:
            self.status.setText("choose a .npz file or a folder first")
            return
        self.open_path(text)

    def _pick_folder(self) -> None:
        start = str(self._gallery_dir or Path.cwd())
        picked = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose a folder of saved figures", start)
        if picked:
            self.open_path(picked)

    def _populate_gallery(self, folder: Path, *, select: Path | None = None,
                          select_first: bool = False) -> None:
        """List every ``.npz`` in ``folder`` as a clickable row; select ``select`` (or the first)."""
        folder = Path(folder)
        self._gallery_dir = folder
        for btn in self._gallery_buttons:
            self.gallery_layout.removeWidget(btn)
            btn.deleteLater()
        self._gallery_buttons = []
        npz_files = sorted(folder.glob("*.npz"))
        if not npz_files:
            self.status.setText(f"no .npz files in {display_path(str(folder))}")
            return
        for npz in npz_files:
            btn = FluentButton(npz.name, color=GREY)
            btn.setToolTip(display_path(str(npz)))
            btn.clicked.connect(lambda _=False, path=npz: self._select_gallery(path))
            self.gallery_layout.addWidget(btn)
            self._gallery_buttons.append(btn)
        target = select if select in npz_files else (npz_files[0] if (select_first or select is None) else None)
        if target is not None:
            self._select_gallery(target)

    def _select_gallery(self, path: Path) -> None:
        path = Path(path)
        # highlight the active row (accent) so the gallery shows which figure is on screen
        for btn in self._gallery_buttons:
            active = btn.toolTip() == display_path(str(path))
            btn.set_color(ACCENT if active else GREY)
        self._load_npz(path)

    def _load_npz(self, path: Path) -> None:
        path = Path(path)
        try:
            saved = load_figure(path)
        except Exception as exc:
            self.status.setText(f"could not load: {str(exc).splitlines()[0][:140]}")
            return
        self._saved = saved
        self._current_path = path
        with _signals_blocked(self.path_edit.edit):
            self.path_edit.setText(str(path))
        if not self.save_dir_edit.text().strip():
            with _signals_blocked(self.save_dir_edit.edit):
                self.save_dir_edit.setText(str(path.parent))
        self._fill_info(saved)
        self._fill_kind_combo(saved)
        self._seed_view_controls(saved)
        self.rebuild()                            # render at the saved (or currently chosen) kind
        self._update_save_preview()

    def _fill_info(self, saved: SavedFigure) -> None:
        # clear the old info rows
        while self.info_layout.count():
            item = self.info_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        rows: list[tuple[str, object]] = [
            ("name", saved.name),
            ("source", saved.info.get("source")),
            ("kind", _kind_label(saved.kind) or saved.kind),
            ("labels", saved.labels),
            ("unit", saved.unit),
            ("data_x", tuple(saved.data_x.shape)),
            ("data_y", tuple(saved.data_y.shape)),
            ("points", saved.info.get("points_done")),
            ("saved", saved.saved_at),
            ("path", display_path(str(self._current_path)) if self._current_path is not None else None),
        ]
        fit = saved.info.get("fit")
        if isinstance(fit, dict) and fit.get("func"):
            pairs = ", ".join(f"{n}={_safe_float(v):.5g}"
                              for n, v in zip(fit.get("names", []), fit.get("popt", [])))
            rows.append(("fit", f"{fit['func']}: {pairs}" if pairs else str(fit["func"])))
        for key, value in rows:
            if value is None:
                continue
            field = FluentReadoutEdit(str(value))
            field.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            self.info_layout.addWidget(FluentSettingRow(key, field, label_width=self._label_w))
        # Raw info: the WHOLE dict verbatim (every key the npz stored) -- answers "what is in the file".
        self.raw_info.setText(repr(dict(saved.info)))
        self.raw_info.setToolTip(repr(dict(saved.info)))

    def _fill_kind_combo(self, saved: SavedFigure) -> None:
        kinds = saved.compatible_kinds()          # saved kind first, then the other compatible kinds
        with _signals_blocked(self.kind_combo):
            self.kind_combo.clear()
            for key in kinds:
                self.kind_combo.addItem(_kind_label(key), key)
            self.kind_combo.setCurrentIndex(0)    # default = saved kind (reproduces the original)

    def _seed_view_controls(self, saved: SavedFigure) -> None:
        """Restore the saved ``info['view']`` (relim / fixed lo-hi) onto the controls so a reopened
        figure comes up AS SAVED; an old npz with no view falls back to the defaults."""
        view = saved.view or {}
        relim = str(view.get("relim", view.get("relim_mode", "tight")) or "tight")
        if relim not in _RELIM_MODES:
            relim = "tight"
        with _signals_blocked(self.relim_combo):
            self.relim_combo.setCurrentText(relim)
        with _signals_blocked(self.fixed_lo, self.fixed_hi):
            self.fixed_lo.setText(str(view.get("fixed_lo", 0.0)))
            self.fixed_hi.setText(str(view.get("fixed_hi", 1.0)))
        self._sync_fixed_lim_enabled(relim)

    def _current_kind(self) -> str | None:
        data = self.kind_combo.currentData()
        return str(data) if data is not None else (self._saved.kind if self._saved else None)

    def _view_kwargs(self) -> dict:
        """The relim view kwargs for ``SavedFigure.plot`` -- relim mode always, fixed lo/hi only in
        ``fixed`` (mirroring the live panel builder, which omits them otherwise so autoscale stays)."""
        relim = self.relim_combo.currentText()
        out: dict[str, object] = {"relim_mode": relim}
        if relim == "fixed":
            out["fixed_lo"] = _safe_float(self.fixed_lo.text(), 0.0)
            out["fixed_hi"] = _safe_float(self.fixed_hi.text(), 1.0)
        return out

    # ------------------------------------------------------------- render / swap
    def teardown(self) -> None:
        if self._canvas is not None:
            self.canvas_holder.removeWidget(self._canvas)
            self._canvas.deleteLater()
        if self._df is not None and plt is not None and getattr(self._df, "fig", None) is not None:
            plt.close(self._df.fig)
        self._canvas = None
        self._df = None

    def rebuild(self) -> None:
        """Re-render the reopened figure at the CURRENT kind + view, then swap the canvas in place.

        Build-then-swap: the new DataFigure is built (through the SAME ``SavedFigure.plot`` factory the
        notebook uses) BEFORE the old one is torn down, so a failed re-view keeps the last good figure
        (never a blank).  ``panel_canvas`` embeds it interactive (zoom / area select), exactly like the
        Task console Edit snapshot."""
        if self._saved is None or panel_canvas is None:
            return
        kind = self._current_kind()
        try:
            new_df = self._saved.plot(kind=kind, **self._view_kwargs())
        except Exception as exc:
            self.status.setText(f"could not render: {str(exc).splitlines()[0][:140]}")
            return
        self.teardown()
        if self.placeholder is not None:
            self.canvas_holder.removeWidget(self.placeholder)
            self.placeholder.deleteLater()
            self.placeholder = None
        self._df = new_df
        self._canvas = panel_canvas(new_df.fig, isolate_wheel=True)
        self._canvas.pin_size()
        self.canvas_holder.addWidget(self._canvas, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self._canvas.draw()
        self._refresh_unit_label()
        self.fill_limits()
        self.status.setText(f"viewing {kind} — fit / relim / limits / save act on this figure")

    # -------------------------------------------------------------- Display cbs
    def _on_kind(self, *_a) -> None:
        self.rebuild()

    def _on_relim(self, mode: str) -> None:
        self._sync_fixed_lim_enabled(str(mode))
        self.rebuild()

    def _on_fixed_lim(self) -> None:
        if self.relim_combo.currentText() == "fixed":
            self.rebuild()

    def _sync_fixed_lim_enabled(self, relim: str) -> None:
        """The lo/hi row stays ALWAYS in the layout (constant footprint -> the canvas never jumps); only
        its inputs enable when ``relim == "fixed"`` (the same no-jitter rule as the Edit tab)."""
        self.fixed_row.setVisible(True)
        fixed = str(relim) == "fixed"
        for w in (self.fixed_lo, self.fixed_hi):
            w.setEnabled(fixed)

    def _on_unit_cycle(self) -> None:
        if self._df is None:
            return
        try:
            self._df.change_unit()
            if self._canvas is not None:
                self._canvas.draw_idle()
        except Exception as exc:
            self.status.setText(f"unit cycle failed: {str(exc).splitlines()[0][:100]}")
            return
        self._refresh_unit_label()
        self.fill_limits()

    def _refresh_unit_label(self) -> None:
        unit = getattr(self._df, "unit", None) if self._df is not None else None
        self.unit_label.setText(str(unit) if unit and str(unit) != "1" else "")

    # ------------------------------------------------------------------- Fit
    def do_fit(self) -> None:
        if self._df is None:
            return
        method = _FIT_MODELS.get(self.fit_combo.currentText())
        if method is None:
            return
        cmd = self.fit_args.text().strip()
        try:
            if self._df.fit is not None:
                self._df.clear()
            call = f"_df.{method}({cmd})" if cmd else f"_df.{method}(is_display=True)"
            import numpy as np
            result, _ = eval(call, {"_df": self._df, "np": np})  # noqa: S307 - local experiment tool
            if self._canvas is not None:
                self._canvas.draw_idle()
            popt = getattr(result, "popt", None)
            names = getattr(result, "names", None) or []
            if popt is None:
                self.status.setText(f"fit {self.fit_combo.currentText()}: did not converge")
                return
            self.status.setText(f"fit {self.fit_combo.currentText()}: "
                                + ", ".join(f"{n}={v:.4g}" for n, v in zip(names, popt)))
        except Exception as exc:
            self.status.setText(f"fit failed: {str(exc).splitlines()[0][:140]}")

    def clear_fit(self) -> None:
        if self._df is None:
            return
        try:
            self._df.clear()
        except Exception:
            pass
        if self._canvas is not None:
            self._canvas.draw_idle()
        self.status.setText("fit cleared")

    # ---------------------------------------------------------------- Limits
    def fill_limits(self) -> None:
        if self._df is None:
            return
        ax = getattr(self._df, "_ax", None)
        if ax is None:
            return
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        with _signals_blocked(self.xmin, self.xmax, self.ymin, self.ymax):
            self.xmin.setText(f"{xlo:.6g}"); self.xmax.setText(f"{xhi:.6g}")
            self.ymin.setText(f"{ylo:.6g}"); self.ymax.setText(f"{yhi:.6g}")

    def apply_limits(self) -> None:
        if self._df is None:
            return
        try:
            self._df.xlim(float(self.xmin.text()), float(self.xmax.text()))
            self._df.ylim(float(self.ymin.text()), float(self.ymax.text()))
            if self._canvas is not None:
                self._canvas.draw_idle()
            self.status.setText("limits applied")
        except Exception as exc:
            self.status.setText(f"bad limits: {str(exc).splitlines()[0][:100]}")

    # ------------------------------------------------------------------ Save
    def _save_stem(self, timestamp: str | None) -> Path:
        """The output stem (no extension) the next Save writes: ``<folder>/<name>_<time>``.

        ``timestamp`` of None yields the ``<time>`` placeholder for the read-only preview; a real
        timestamp makes the file unique.  The folder is the Save picker (defaulting beside the opened
        file)."""
        name = (self._saved.name if self._saved else None) or "figure"
        text = self.save_dir_edit.text().strip()
        base = Path(text) if text else (self._current_path.parent if self._current_path else Path.cwd())
        if base.suffix.lower() in (".npz", ".png"):
            base = base.with_suffix("")
            return base.parent / f"{base.name}_{timestamp or '<time>'}"
        return base / f"{name}_{timestamp or '<time>'}"

    def _update_save_preview(self) -> None:
        if not hasattr(self, "save_preview"):
            return
        self.save_preview.setText(f"{display_path(str(self._save_stem(None)))}.png + .npz")

    def save(self) -> None:
        if self._df is None:
            return
        try:
            stem = self._save_stem(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            stem.parent.mkdir(parents=True, exist_ok=True)
            # a stem WITH a suffix makes DataFigure.save write it verbatim (no extra timestamp),
            # so this resolver is the single source of the output name.
            out = self._df.save(stem.with_suffix(".png"))
            self._last_save_dir = str(stem.parent)
            self._update_save_preview()
            self.status.setText(f"saved {out['figure'].name} + {out['data'].name} → …/{stem.parent.name}")
            self.status.setToolTip(str(stem.parent))
        except Exception as exc:
            self.status.setText(f"save failed: {str(exc).splitlines()[0][:120]}")

    # ---------------------------------------------------------------- sizing
    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API name
        # The window fits its content EXACTLY.  WIDTH is the natural layout size (Qt summing the two
        # fixed side columns + the fixed-inch canvas + gaps) so the fixed-width columns are NEVER
        # clipped -- clamping the width to the screen was what cut the right column off.  HEIGHT is
        # clamped to the primary screen (the shared qt_fluent rule) so a tall content still opens a
        # usable window; each column already scrolls internally.  A Python exception from this
        # Qt-virtual would crash the C++ adjustSize, so keep it total.
        try:
            natural = super().sizeHint()
            fit = screen_fit_window_size(self.window_ratio)
            h = min(natural.height(), fit.height()) if fit.height() > 0 else natural.height()
            return QtCore.QSize(natural.width(), max(h, 1))
        except Exception:
            return super().sizeHint()


def show_figure_viewer(path: str | Path | None = None, *, scale: float | None = None,
                       window_ratio: float = WINDOW_SCREEN_FRACTION,
                       hide_on_close: bool = False) -> FigureViewer:
    """Open the saved-figure viewer in a Fluent window (mirrors ``show_pulse_gui`` /
    ``show_task_console``: the body sizes from the primary screen, the window wraps it, and the shared
    auto-scale rule owns the on-screen control size).  ``path`` opens a ``.npz`` (or a folder of them)
    on launch; omit it to open empty and Browse from inside."""
    app = ensure_qt_app()
    viewer = FigureViewer(path, scale=scale, window_ratio=window_ratio)
    window = FluentWindow(widget=viewer, title="FigureViewer@Zou lab", hide_on_close=hide_on_close)
    window.adjustSize()
    window.setFixedSize(window.size())
    center_window_on_primary_screen(window, app)
    window.show()
    viewer._zlc_window = window
    if not hasattr(app, "_zlc_figure_windows"):
        app._zlc_figure_windows = []
    app._zlc_figure_windows.extend([window, viewer])
    return viewer


__all__ = ["FigureViewer", "show_figure_viewer"]
