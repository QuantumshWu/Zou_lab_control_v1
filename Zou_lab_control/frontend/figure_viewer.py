"""Reopen a saved figure ``.npz`` INTO the Task console -- ``exp.figure_viewer()``.

A saved panel / overnight scan writes a ``<name>_<time>.npz`` (``data_x`` / ``data_y`` / ``info``)
next to its ``.png`` (see :meth:`~.data_figure.DataFigure.save`).  This window is the GUI
counterpart of the notebook one-liner ``na.load_figure('scan.npz')`` -- but instead of a bespoke
viewer, the loaded figure becomes ONE hub SIGNAL and the whole reuse is the Task console board:

* a :class:`LoadedFigureNode` publishes the saved arrays as a static hub signal (``fig_value`` plus,
  for a 1-D save, its companion ``fig_x``; for a site map, its ``fig_centers``), declaring the SAME
  ``output_specs`` / ``published_signals`` / ``x_signal`` / ``sitemap_*`` a live producer does -- so a
  panel wired to it reads the right axis label, unit and x-coordinates;
* the window SEEDS one panel with the SAVED ``kind`` + ``view`` (``PanelConfig(kind=sf.kind,
  source="value = fig_value", params=sf.info["view"])``) so it opens reproducing the original figure;
* the panel lives on a real :class:`~.task_console.TaskConsole`, so the user gets the board, Add Panel,
  the signal picker, re-wiring and the light processing for free -- add MORE panels reading the same
  ``fig_value`` under a different kind, or fit / relim / re-save through the panel's own Edit tab.

A read-only **Info** column on the left lists EVERY key the npz stored (name / source / kind / labels /
unit / shapes / points / saved / the view sub-dict / any fit) plus the raw ``info`` dict verbatim, so
"what is in this file" is always fully visible next to the board.

It is session-INDEPENDENT: it only reads a file, no hardware, no acquisition -- the ``LoadedFigureNode``
re-publishes the same stored arrays every shot.  The window chrome (Fluent frameless shell, one shared
display-scale rule, centred on the primary screen) mirrors ``show_pulse_gui`` / ``show_task_console``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from PyQt5 import QtCore, QtWidgets

from Zou_lab_control._paths import display_path
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.logic import LogicNode, SignalSpec

from .data_figure import SavedFigure, load_figure
from .task_console import (
    PanelConfig,
    TaskConsole,
    TaskConsoleState,
)
from .qt_fluent import (
    GREY,
    FluentButton,
    FluentLabel,
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


#: The hub signal name the loaded figure's PRIMARY data is published under -- the panel the window
#: seeds is wired to it (``value = fig_value``), and any further panel the user adds picks it too.
FIG_VALUE_KEY = "value"
#: The companion x-axis signal (1-D saves) -- lets a 1-D panel draw its curve vs the saved data_x with
#: the saved x-axis label/unit (the console's ``curve_x_provider`` resolves it from THIS node).
FIG_X_KEY = "x"
#: The per-tweezer centres signal (site-map saves) -- a site-map panel resolves its ring centres from
#: this node via ``sitemap_centers_key``.
FIG_CENTERS_KEY = "centers"

#: The node prefix -- so the published hub names are ``fig_value`` / ``fig_x`` / ``fig_centers`` and the
#: seeded panel's ``inputs`` name ``fig_value``.
FIG_PREFIX = "fig_"


def _kind_label(key: str | None) -> str:
    """The human plot-kind label from the ONE ``PLOT_KINDS`` table (never a hand-typed name), so the
    Info panel reads 'Distribution' not 'hist'; falls back to the raw key for an unknown kind."""
    if not key:
        return ""
    from .live import PLOT_KIND_BY_KEY
    pk = PLOT_KIND_BY_KEY.get(str(key))
    return pk.label if pk is not None else str(key)


class LoadedFigureNode(LogicNode):
    """A static logic node that re-publishes a :class:`SavedFigure`'s stored arrays as hub signals.

    It is a SOURCE like any live producer -- so a Task-console panel binds to it exactly as it binds
    to a camera / measurement -- except its ``shot()`` returns the SAME saved data every time (a file,
    not hardware).  The shape it publishes is chosen so the saved ``kind`` REPRODUCES faithfully on a
    panel wired to ``value``:

    * 1-D families (line / rolling trace / distribution): ``value`` = the 1-D ``data_y`` column, plus a
      companion ``x`` = the 1-D ``data_x`` column, with ``x_signal`` / ``y_signal`` / ``output_specs``
      set so a 1-D panel draws the curve vs the saved x with the saved x-axis label + unit;
    * 2-D image: ``value`` = the ``(H, W)`` frame reshaped from ``data_y`` over the ``(N, 2)`` scan grid;
    * site map: ``value`` = the per-site ``data_y`` vector and ``centers`` = the ``(N, 2)`` ``data_x``
      centres, with ``sitemap_centers_key`` so the panel resolves the rings from this node.

    Every output carries ``points_shape``/``data_shape`` = ``None`` (no repeat contract): the saved
    figure is a single static block, so a panel plots it verbatim (no repeat-axis reduction)."""

    layer = "figure"

    def __init__(self, hub: SignalHub, saved: SavedFigure, *, prefix: str = FIG_PREFIX):
        super().__init__(hub, prefix=prefix)
        self.saved = saved
        self.node_label = str(saved.name or "figure")
        self.instance_label = self.node_label

        data_x = np.asarray(saved.data_x, dtype=float)
        data_y = np.asarray(saved.data_y, dtype=float)
        kind = self.kind = str(saved.kind or ("2d" if (data_x.ndim == 2 and data_x.shape[1] >= 2) else "1d"))
        # The saved ``labels`` are the FULL matplotlib axis labels (the unit is baked in-line, e.g.
        # "Detuning (GHz)"), so the SignalSpec keeps unit="" -- appending saved.unit again would double
        # the "(unit)" suffix.  The display unit is carried on the panel's view / DataFigure separately.
        labels = saved.labels or ["X", "Y", "Z"]
        self._xlabel = str(labels[0]) if len(labels) > 0 else "X"
        self._ylabel = str(labels[1]) if len(labels) > 1 else "Y"

        # Resolve the arrays each kind's panel consumes.
        self._value: np.ndarray
        self._x: np.ndarray | None = None
        self._centers: np.ndarray | None = None
        if kind in ("2d",):
            self._value = self._as_image(data_x, data_y)
        elif kind in ("sites",):
            self._value = self._first_column(data_y)
            self._centers = data_x[:, :2] if (data_x.ndim == 2 and data_x.shape[1] >= 2) else None
        else:  # 1d / monitor / hist and any 1-D family
            self._value = self._first_column(data_y)
            self._x = self._first_column(data_x)

    # ---------------------------------------------------------------- shaping
    @staticmethod
    def _first_column(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        return a[:, 0] if a.ndim == 2 else a.reshape(-1)

    @staticmethod
    def _as_image(data_x: np.ndarray, data_y: np.ndarray) -> np.ndarray:
        """The ``(H, W)`` image a 2-D save stored as an ``(N, 2)`` grid of ``data_x`` + flat ``data_y``.
        Recover the grid from the unique x / y coordinates (the SAME reshape ``Live2DDis`` uses); fall
        back to a square-ish reshape when data_x is not a clean 2-column grid."""
        y = np.asarray(data_y, dtype=float).reshape(-1)
        x = np.asarray(data_x, dtype=float)
        if x.ndim == 2 and x.shape[1] >= 2 and x.shape[0] == y.size:
            nx = np.unique(x[:, 0]).size
            ny = np.unique(x[:, 1]).size
            if nx * ny == y.size and nx > 1 and ny > 1:
                return y.reshape(ny, nx)
        side = int(round(np.sqrt(y.size)))
        if side * side == y.size and side > 1:
            return y.reshape(side, side)
        return y.reshape(1, -1)

    # -------------------------------------------------------- sites wiring
    @property
    def sitemap_centers_key(self) -> str:
        """The BARE centres key a site-map panel resolves from this node (the console prepends the
        node prefix, exactly as it does for a Judge-occupancy processor's ``"centers"``) -- present
        only for a site-map save, else blank so the console skips this node for a site-map underlay."""
        return FIG_CENTERS_KEY if self._centers is not None else ""

    #: A saved figure carries no camera frame underlay (the npz stores only data_x/data_y), so the
    #: site map shows rings + occupancy with no background image.
    sitemap_image_key = ""

    # --------------------------------------------------------- scan-curve wiring
    @property
    def x_signal(self) -> str:
        """The companion x-axis signal a 1-D panel draws its curve against (the console's
        ``curve_x_provider`` matches ``y_signal`` -> ``x_signal``).  Blank when there is no x
        (a 2-D / site-map save)."""
        return (self.prefix + FIG_X_KEY) if self._x is not None else ""

    @property
    def y_signal(self) -> str:
        return self.prefix + FIG_VALUE_KEY

    # ------------------------------------------------------------- hub contract
    def published_signals(self) -> frozenset:
        keys = [self.prefix + FIG_VALUE_KEY]
        if self._x is not None:
            keys.append(self.prefix + FIG_X_KEY)
        if self._centers is not None:
            keys.append(self.prefix + FIG_CENTERS_KEY)
        return frozenset(keys)

    def output_specs(self) -> tuple[SignalSpec, ...]:
        # points_shape/data_shape=None: a static, single-block figure carries NO repeat axis, so a
        # panel plots ``value`` verbatim (the identity ``value = signal`` short-circuit).
        specs = [SignalSpec(self.prefix + FIG_VALUE_KEY, self._ylabel, "",
                            "the saved figure's primary data (data_y)")]
        if self._x is not None:
            specs.append(SignalSpec(self.prefix + FIG_X_KEY, self._xlabel, "",
                                    "the saved figure's x axis (data_x)"))
        if self._centers is not None:
            specs.append(SignalSpec(self.prefix + FIG_CENTERS_KEY, "site centres", "px",
                                    "per-tweezer (N, 2) camera-pixel centres (data_x)"))
        return tuple(specs)

    def shot(self) -> dict[str, object]:
        out: dict[str, object] = {FIG_VALUE_KEY: np.array(self._value, dtype=float)}
        if self._x is not None:
            out[FIG_X_KEY] = np.array(self._x, dtype=float)
        if self._centers is not None:
            out[FIG_CENTERS_KEY] = np.array(self._centers, dtype=float)
        return out


def _seed_state(saved: SavedFigure) -> TaskConsoleState:
    """The console layout that REPRODUCES the saved figure: ONE panel of the saved ``kind`` wired to
    ``fig_value`` (``value = fig_value``) with the saved ``info['view']`` restored as its params, so it
    opens exactly as saved.  A save with no recorded kind falls back to shape inference; an unknown/
    non-panel kind falls back to a 1-D vector panel so the window still opens with a reproduction."""
    from .live import PLOT_KIND_BY_KEY

    kind = str(saved.kind or "")
    pk = PLOT_KIND_BY_KEY.get(kind)
    if pk is None or not pk.panel:
        kind = "2d" if (np.asarray(saved.data_x).ndim == 2 and np.asarray(saved.data_x).shape[1] >= 2) else "1d"
    view = dict(saved.view or {})
    panel = PanelConfig(
        kind=kind,
        title=str(saved.name or "figure"),
        source=f"value = {FIG_PREFIX}{FIG_VALUE_KEY}",
        inputs=[FIG_PREFIX + FIG_VALUE_KEY],
        params=view,
    )
    return TaskConsoleState(name=str(saved.name or "figure"), panels=[panel])


class FigureViewer(QtWidgets.QWidget):
    """The reopen-into-the-console body (wrapped in a :class:`FluentWindow` by :func:`show_figure_viewer`).

    Left:  a read-only **Info** column -- a path field + Load, then one row per stored fact and the raw
           ``info`` dict, so the whole npz is visible next to the board.
    Right: a live :class:`~.task_console.TaskConsole` whose seeded panel reproduces the saved figure and
           on which the user can Add more panels / re-wire / process the loaded ``fig_value`` signal.
    """

    def __init__(self, path: str | Path | None = None, *, scale: float | None = None,
                 window_ratio: float = WINDOW_SCREEN_FRACTION, parent=None):
        super().__init__(parent)
        ensure_qt_app()
        set_fluent_scale(scale)                    # the ONE shared GUI-scale rule (mirrors the other GUIs)
        self._scale = scale
        self.window_ratio = float(window_ratio)
        self.setStyleSheet("background: transparent;")

        self.saved: SavedFigure | None = None
        self.hub = SignalHub()
        self.node: LoadedFigureNode | None = None
        self.console: TaskConsole | None = None
        self._current_path: Path | None = None

        self._label_w = setting_label_width(
            ["name", "source", "kind", "labels", "unit", "data_x", "data_y",
             "points", "repeat", "saved", "view", "fit", "path"])
        # The fixed Info-column width; the console is sized to fill the REST of the screen budget beside
        # it (so the whole window stays within the shared screen-fraction rule, no board clipping).
        self._info_col_w = self._label_w + scaled_px(230, minimum=170)
        self._root_margin = scaled_px(10, minimum=6)

        root = QtWidgets.QHBoxLayout(self)
        m = self._root_margin
        root.setContentsMargins(m, m, m, m)
        root.setSpacing(self._root_margin)
        root.addWidget(self._build_info_column(), 0)
        self._console_holder = QtWidgets.QVBoxLayout()
        self._console_holder.setContentsMargins(0, 0, 0, 0)
        holder_host = QtWidgets.QWidget()
        holder_host.setStyleSheet("background: transparent;")
        holder_host.setLayout(self._console_holder)
        root.addWidget(holder_host, 1)

        if path is not None:
            self.open_path(path)

    # ------------------------------------------------------------------ layout
    def _build_info_column(self) -> QtWidgets.QWidget:
        col = QtWidgets.QWidget()
        col.setStyleSheet("background: transparent;")
        col.setFixedWidth(self._info_col_w)
        lay = QtWidgets.QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(scaled_px(6, minimum=4))

        # --- File: path field + Load ------------------------------------------
        lay.addWidget(FluentSectionLabel("File"))
        self.path_edit = FluentPathEdit(
            "", mode="file", caption="Open a saved figure (.npz)",
            file_filter="Saved figures (*.npz);;All files (*)")
        self.path_edit.setToolTip("A saved figure .npz -- type a path or Browse, then Load.")
        lay.addWidget(FluentSettingRow("path", self.path_edit, label_width=self._label_w))
        self.load_button = FluentButton("Load", color=GREY)
        self.load_button.setToolTip("Load the .npz into the console (its data becomes the fig_value signal).")
        self.load_button.clicked.connect(self._on_load_clicked)
        load_row = QtWidgets.QHBoxLayout()
        load_row.setContentsMargins(0, 0, 0, 0)
        load_row.addWidget(self.load_button, 0)
        load_row.addStretch(1)
        lay.addLayout(load_row)

        # --- Info: one row per stored fact ------------------------------------
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

        # Raw info: EVERY key the npz stored, verbatim.
        lay.addWidget(FluentSectionLabel("Raw info"))
        self.raw_info = FluentReadoutEdit("")
        self.raw_info.setToolTip("The full info dict stored in the npz -- read-only, select to copy.")
        self.raw_info.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        lay.addWidget(self.raw_info)

        self.status = FluentLabel("Load a saved figure (.npz) to view it on the board.")
        self.status.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        lay.addWidget(self.status)
        return col

    # -------------------------------------------------------------- public API
    def window(self):
        return getattr(self, "_zlc_window", None)

    def open_path(self, path: str | Path) -> None:
        """Load a ``.npz``: publish its data as ``fig_value`` and seed a console reproducing it."""
        p = Path(str(path).strip())
        if str(p) in ("", "."):
            return
        self._load_npz(p)

    # ------------------------------------------------------------- file / load
    def _on_load_clicked(self) -> None:
        text = self.path_edit.text().strip()
        if not text:
            self.status.setText("choose a .npz file first")
            return
        self.open_path(text)

    def _load_npz(self, path: Path) -> None:
        path = Path(path)
        try:
            saved = load_figure(path)
        except Exception as exc:
            self.status.setText(f"could not load: {str(exc).splitlines()[0][:160]}")
            return
        self.saved = saved
        self._current_path = path
        with _signals_blocked(self.path_edit.edit):
            self.path_edit.setText(str(path))
        self._fill_info(saved)
        self._build_console(saved)
        self.status.setText(
            f"loaded {display_path(str(path))} -> fig_value signal; the seeded {saved.kind or 'figure'} "
            "panel reproduces it -- Add Panel to view it another way.")

    def _fill_info(self, saved: SavedFigure) -> None:
        while self.info_layout.count():
            item = self.info_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        data_x = np.asarray(saved.data_x)
        data_y = np.asarray(saved.data_y)
        rows: list[tuple[str, object]] = [
            ("name", saved.name),
            ("source", saved.info.get("source")),
            ("kind", _kind_label(saved.kind) or saved.kind),
            ("labels", saved.labels),
            ("unit", saved.unit),
            ("data_x", tuple(data_x.shape)),
            ("data_y", tuple(data_y.shape)),
            ("points", saved.info.get("points_done")),
            ("repeat", saved.info.get("repeat_cur")),
            ("saved", saved.saved_at),
        ]
        if saved.info.get("size"):
            rows.append(("size", saved.info.get("size")))
        if saved.view:
            view_bits = ", ".join(f"{k}={v}" for k, v in saved.view.items() if v is not None)
            rows.append(("view", view_bits or "(defaults)"))
        fit = saved.info.get("fit")
        if isinstance(fit, Mapping) and fit.get("func"):
            pairs = ", ".join(f"{n}={float(v):.5g}"
                              for n, v in zip(fit.get("names", []), fit.get("popt", [])))
            rows.append(("fit", f"{fit['func']}: {pairs}" if pairs else str(fit["func"])))
        if self._current_path is not None:
            rows.append(("path", display_path(str(self._current_path))))
        for key, value in rows:
            if value is None:
                continue
            field = FluentReadoutEdit(str(value))
            field.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            self.info_layout.addWidget(FluentSettingRow(key, field, label_width=self._label_w))
        # Raw info: the WHOLE dict verbatim (every key the npz stored).
        self.raw_info.setText(repr(dict(saved.info)))
        self.raw_info.setToolTip(repr(dict(saved.info)))

    # -------------------------------------------------------------- console
    def _build_console(self, saved: SavedFigure) -> None:
        """Publish the loaded figure as a hub signal + seed a fresh TaskConsole reproducing it, and
        swap it into the right column (a re-Load rebuilds everything on a fresh hub)."""
        self._teardown_console()
        self.hub = SignalHub()
        self.node = LoadedFigureNode(self.hub, saved)
        self.node.step()                       # publish once so the signal is on the hub immediately
        state = _seed_state(saved)
        # Size the console to fill the screen budget BESIDE the Info column, so the whole window stays
        # within the shared screen-fraction rule (the console alone would demand the full screen width,
        # pushing the Info column off-screen / clipping the board).  ``window_px`` overrides the console's
        # own screen-fit; the height matches the fit so the board never clips vertically either.
        fit = screen_fit_window_size(self.window_ratio)
        console_w = max(scaled_px(520, minimum=380),
                        fit.width() - self._info_col_w - 3 * self._root_margin)
        self.console = TaskConsole(hub=self.hub, state=state, running_nodes=[self.node],
                                   session=None, scale=self._scale,
                                   window_px=(console_w, fit.height()))
        self._console_holder.addWidget(self.console)

    def _teardown_console(self) -> None:
        if self.console is not None:
            try:
                self.console.shutdown()
            except Exception:
                pass
            self._console_holder.removeWidget(self.console)
            self.console.deleteLater()
            self.console = None
        if self.node is not None:
            try:
                self.node.stop()
            except Exception:
                pass
            self.node = None

    def teardown(self) -> None:
        self._teardown_console()

    # ---------------------------------------------------------------- sizing
    def sizeHint(self) -> QtCore.QSize:  # noqa: N802 - Qt API name
        # The window fits its content: WIDTH is the natural layout size (the fixed Info column + the
        # TaskConsole's own fixed size) so nothing clips; HEIGHT is clamped to the primary screen (the
        # shared qt_fluent rule).  A Python exception from this Qt-virtual would crash the C++
        # adjustSize, so keep it total (self-swallow).
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
    auto-scale rule owns the on-screen control size).  ``path`` loads a ``.npz`` on launch -- its data
    becomes the ``fig_value`` hub signal and a panel of the saved kind opens reproducing it on a real
    Task console board; omit it to open empty and Browse from inside.  Closing the window tears the
    embedded console down (its refresh timer + the loaded-figure node)."""
    app = ensure_qt_app()
    viewer = FigureViewer(path, scale=scale, window_ratio=window_ratio)
    window = FluentWindow(widget=viewer, title="FigureViewer@Zou lab", hide_on_close=hide_on_close)
    window.closed.connect(viewer.teardown)
    window.adjustSize()
    window.setFixedSize(window.size())
    center_window_on_primary_screen(window, app)
    window.show()
    viewer._zlc_window = window
    if not hasattr(app, "_zlc_figure_windows"):
        app._zlc_figure_windows = []
    app._zlc_figure_windows.extend([window, viewer])
    return viewer


__all__ = ["FigureViewer", "LoadedFigureNode", "show_figure_viewer"]
