"""The saved-figure viewer window -- ``figure_viewer.py`` / ``open_figure_viewer``.

A read-only **Info** column on the left, a board on the right.  The Info column
groups everything known about what is on the board into tabs -- **Plot** (how it
draws), **Measurement** (its shapes and source), **Device** (the run's
provenance), **Flow** (the upstream graph of how the data was produced) and
**Raw** (the whole record verbatim) -- so "what is this?" is always answerable
next to the picture rather than in a separate dialog.

**Opening a stored figure is not connected yet, and the window says so.**  It
used to read a saved ``.npz``, publish it as hub signals and seed a panel bound
to them.  All three of those are gone: nothing writes those files, the hub was
deleted, and a figure is now a view PROJECTED from a data artifact
(``Experiment.figure(ref)``) rather than a document reopened from disk.

Restoring it is a product decision rather than a wiring one, and the design note
already places it: the current step must not pre-build an artifact
catalog/browser, and extending catalog/browse/open over arbitrary stored
artifacts is a later step of the migration.  Building that browser here would be
building the thing the plan says not to build yet, so the File field reports
where it stands instead of quietly doing nothing -- which would read as "this
file is empty".

The window chrome (Fluent frameless shell, one shared display-scale rule, centred
on the primary screen) mirrors ``launch_pulse_editor_window`` / ``show_task_console``.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from zlc_storage.paths import display_path

from zlc_frontend.qt_widgets import FlowGraphView
from zlc_workbench.task_console.plot_bridge_console import TaskConsole

from zlc_frontend.qt_widgets import (
    CARD_PAD,
    FluentCodeEdit,
    FluentFrame,
    FluentPathEdit,
    FluentScrollArea,
    FluentSectionLabel,
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
)




class FigureViewer(QtWidgets.QWidget):
    """The viewer body (wrapped in a :class:`FluentWindow` by :func:`show_figure_viewer`).

    Left:  a read-only **Info** column -- a path field (Browse, no separate Load button) over a
           TAB set that groups what is known about the figure on the board: **Plot** (how it
           draws) / **Measurement** (its shapes and source) / **Device** (the run's provenance) /
           **Flow** (how the data was produced) / **Raw** (the whole record verbatim), so the
           facts sit beside the picture instead of in one crushed column.
    Right: an embedded :class:`~.task_console.TaskConsole` (size-Expanding, so it FILLS the pane
           and reflows into 2+ columns), where the operator gets the board, Add Panel, the signal
           picker and re-wiring without this window growing its own.  The Info card and the
           console share one root layout with equal margins + spacing, so their outer edges line up.

    Opening a STORED figure is not connected on the current data plane -- see the module
    docstring for why that is a deferred product decision rather than missing wiring.
    """


    def __init__(self, path: str | Path | None = None, *, scale: float | None = None,
                 window_ratio: float = WINDOW_SCREEN_FRACTION, parent=None) -> None:
        """Build the two-pane body: the Info column beside an embedded console board.

        The scale is set BEFORE any widget is constructed -- every Fluent metric
        is read at build time, so a later change would leave this window's
        controls sized against a different rule than the rest of the GUI.

        The right pane is a ``TaskConsole(embedded=True)``: that mode exists for
        this window (see its constructor), which is why a loaded figure gets the
        board, Add Panel, the picker and re-wiring for free instead of this file
        growing a second display of its own.
        """

        ensure_qt_app()          # metrics below need a QApplication to measure against
        set_fluent_scale(scale)
        super().__init__(parent)
        self.window_ratio = float(window_ratio)
        self._current_path: Path | None = None
        self.node = None
        # One label column for every Info row, from the widest label any filler writes
        # -- so the rows align across all three rows-tabs rather than per-tab.
        self._label_w = setting_label_width(
            ("calibration", "data_shape", "points_shape", "captured_at"))
        # The Info column is FIXED-width -- the board beside it absorbs a resize -- and it
        # is the LABEL column plus room for the value beside it.  Picking a flat number
        # instead leaves out the label half: the column came out too narrow for its own
        # tab strip and all five tab names ("Plot", "Measurement", ...) elided to "...".
        self._info_col_w = self._label_w + scaled_px(320, minimum=240)

        # The window IS the shared screen-fit size -- the same statement the task console
        # makes standalone -- and everything inside is divided out of it.  Leaving it to a
        # size hint is not the same thing: the hint gets clamped to the desktop and this
        # window came out smaller than the other two, which is exactly the drift the shared
        # rule exists to prevent.
        self.setFixedSize(screen_fit_window_size(self.window_ratio))

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, window_pad(1), 0, window_pad(1))
        root.setSpacing(0)
        root.addWidget(self._build_info_column(), 0)

        # The console lives in its own layout so a reload can swap it out (see
        # ``_teardown_console``) without disturbing the Info column beside it.
        holder = QtWidgets.QWidget()
        holder.setStyleSheet("background: transparent;")
        self._console_holder = QtWidgets.QVBoxLayout(holder)
        self._console_holder.setContentsMargins(0, 0, 0, 0)
        self._console_holder.setSpacing(0)
        self.console = TaskConsole(embedded=True, window_ratio=window_ratio)
        self._console_holder.addWidget(self.console)
        root.addWidget(holder, 1)

        if path is not None:
            self.open_path(path)

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
        # Browse (or a typed path) acts immediately -- no separate Load button: FluentPathEdit.changed
        # fires when Browse picks a file, or when the user finishes typing one, and _on_path_changed
        # takes it from there.  A FLAT header (no shadow) mirrors the console header, which is flat so
        # its shadow's soft bottom edge never draws a thin line into the gap above the tab strip.
        header_frame = FluentFrame(bordered=False)
        header_frame.setFixedHeight(scaled_px(48, minimum=38))
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(scaled_px(12), scaled_px(6), scaled_px(12), scaled_px(6))
        header.setSpacing(scaled_px(8, minimum=5))
        header.addWidget(FluentSectionLabel("File"))
        self.path_edit = FluentPathEdit(
            "", mode="file", caption="Open a stored figure",
            file_filter="All files (*)")
        self.path_edit.setToolTip("Browse to a stored figure, or type a full path.  Opening one is "
                                  "not connected yet -- the status line below says so when you pick.")
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

        # Widen the column so no tab name is cut.  The width comes from MEASURING the tab
        # names at the current scale, not from a number I picked and not from the tab
        # bar's own size hint -- that hint is already the elided width, which is why
        # "Measurement" still read as "Measur..." after two attempts.  Every tab gets the
        # widest name's width, since a tab whose name is cut cannot be identified.

        # The SAME persistent status surface the console mounts (FluentStatusStrip): one
        # always-visible line, severity-coloured, eliding with the full text in the tooltip --
        # a load failure turns it red instead of hiding as grey prose.
        self.status = FluentStatusStrip()
        self.status.show_message("Opening a stored figure is not connected on this data plane yet.")
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
        raw.setToolTip("Everything recorded about this figure, verbatim -- read-only, select to copy.")
        vbox.addWidget(raw)
        self.info_tabs.add_permanent_tab(body, title)
        return raw

    # -------------------------------------------------------------- public API
    def window(self):
        return getattr(self, "_zlc_window", None)

    def open_path(self, path: str | Path) -> None:
        """Show ``path`` in the File field and try to put it on the board.

        A blank path is a cleared field, not an error, so it is simply ignored.
        Anything else goes to :meth:`_open_stored_figure`, which reports that
        opening stored figures is not connected on the current data plane."""

        p = Path(str(path).strip())
        if str(p) in ("", "."):
            return
        self._open_stored_figure(p)

    def _open_stored_figure(self, picked: Path) -> None:
        """Put a picked file on the board.

        NOT YET WIRED, and it says so rather than looking like it worked.  The
        old path published the file as hub signals; the hub is gone, nothing
        writes those files any more, and a figure is now a view projected from a
        data artifact (``Experiment.figure(ref)``) rather than a document
        reopened from disk.

        What replaces it is a product decision the plan has already placed: the
        current step must not pre-build an artifact catalog/browser, and
        catalog/browse/open over arbitrary stored artifacts is a later step.
        Building it here would be building what the plan says to defer, so this
        reports where it stands.  Silently doing nothing would read as "this
        file is empty", which is worse than an admitted gap.
        """

        self._current_path = picked
        self.status.show_message(
            f"{display_path(str(picked))}: opening stored figures is not connected yet -- "
            "a figure is now projected from a run artifact, not reopened from a file.",
            severity="warning")

    # ------------------------------------------------------------- file / load
    def _on_path_changed(self, text: str) -> None:
        """Browse (or a typed path) -> open it, with no separate Load button.

        ``changed`` fires on every keystroke, so this only acts on a path that
        is a real EXISTING file: half-typed text and folders are ignored rather
        than reported, which is what keeps a person typing from being told off
        once per character.  Re-reporting the file already shown is skipped too,
        since setting the field echoes back through this same signal."""

        p = Path(str(text).strip())
        if str(p) in ("", ".") or not p.is_file():
            return
        if self._current_path is not None and p == self._current_path:
            return
        self._open_stored_figure(p)


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
        # VERBATIM from their ``sizeHint``.  The size is the screen envelope, never the content: the inner
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
    """Open the figure viewer in a Fluent window.

    Mirrors ``launch_pulse_editor_window`` / ``show_task_console``: the body sizes from the primary screen, the
    window wraps it, and the shared auto-scale rule owns the on-screen control size.  ``path``
    prefills the File field on launch; opening a stored figure is not connected on the current data
    plane, so the window reports that rather than loading (see the module docstring).  Closing the
    window tears the embedded console down."""
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


__all__ = ["FigureViewer", "show_figure_viewer"]
