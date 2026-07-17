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

from Zou_lab_control._paths import display_path
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.core.signal_tensor import canonical_physical_shape
from Zou_lab_control.neutral_atom.operations.logic import LogicNode, SignalSpec

from .data_figure import SavedFigure, load_figure
from .flow_graph_view import FlowGraphView
from .task_console import (
    PanelConfig,
    TaskConsole,
    TaskConsoleState,
)

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
    not hardware).

    A save records the FULL producer
    blocks it plotted: for each named signal a ``block`` (its native ``(repeat, *points, *data)`` array),
    its ``points_shape`` / ``data_shape`` and a ``role`` (``value`` / ``x`` / ``centers`` / ``frame``).
    This node re-publishes each block **verbatim** under its own bare name and declares the SAME
    ``SignalSpec`` (with the stored contract), then wires by role exactly as a live producer:

    * ``value`` -> the primary signal (published under ``value`` so a panel seeded ``value = fig_value``
      binds to it) + ``y_signal`` -- because the block declares its structure, the panel's identity
      short-circuit reduces the repeat axis correctly (no bespoke reshape here);
    * ``x`` -> the companion 1-D curve x (``x_signal``);
    * ``centers`` -> ``sitemap_centers_key`` (the rings);
    * ``frame`` -> ``sitemap_image_key`` -- so a site-map save gets its BACKGROUND camera frame back
      (the underlay the live occupancy panel had), not a bare ring plot."""

    layer = "figure"

    def __init__(self, hub: SignalHub, saved: SavedFigure, *, prefix: str = FIG_PREFIX):
        super().__init__(hub, prefix=prefix)
        self.saved = saved
        self.node_label = str(saved.name or "figure")
        self.instance_label = self.node_label

        kind = self.kind = saved.kind
        # The saved ``labels`` are the FULL matplotlib axis labels (the unit is baked in-line, e.g.
        # "Detuning (GHz)"), so the SignalSpec keeps unit="" -- appending saved.unit again would double
        # the "(unit)" suffix.  The display unit is carried on the panel's view / DataFigure separately.
        # ``_xlabel`` / ``_ylabel`` are the x / value signal specs' declared axis labels (legend / picker
        # text, and the 1-D panel's fallback axes).  The panel's RENDERED axes -- x, y and a 2-D colour
        # bar -- come from the saved labels seeded into the panel params (``_seed_state`` reads the ONE
        # ``saved.axis_labels()`` source), so there is no separate z field to carry here.
        labels = saved.labels
        self._xlabel = str(labels[0]) if len(labels) > 0 else "X"
        self._ylabel = str(labels[1]) if len(labels) > 1 else "Y"

        # bare key -> (block, SignalSpec) for every signal this node publishes, in publish order.  The
        # ``_role_key`` maps each explicitly stored role to the fixed bare key it lands on.
        self._blocks: dict[str, np.ndarray] = {}
        self._specs: dict[str, SignalSpec] = {}
        self._role_key: dict[str, str] = {}
        # A pulse figure carries its reproduction OBJECT here (the hub is float-only); the console's
        # pulse_state_provider reads it off this node.  None for every non-pulse figure.
        self.pulse_state = None
        self.pulse_include_always_off = True
        # A grid figure carries its replay RECIPE (a dict) here the SAME way -- the console's
        # grid_recipe_provider reads it off this node.  None for every non-grid figure.
        self.grid_recipe = None

        # A STRUCTURED figure (a pulse timeline / a per-site grid) publishes its reproduction PAYLOAD
        # (a PulseTableState object, or a grid recipe dict) carried ON THIS NODE -- a hub value may only be
        # a float array, so the panel resolves the payload off the producing node (the SAME "aux data from
        # the producing node" wiring the site map uses for its centres/frame).  The seeded ``pulse`` /
        # ``grid`` panel binds to ``fig_value`` exactly like a hist panel binds to its sample vector, and
        # PanelCard's matching branch renders from the resolved payload.  This is the SAME load path every
        # kind uses.
        if kind == "pulse":
            self._build_pulse(saved)
        elif kind == "grid":
            self._build_grid(saved)
        else:
            self._build_from_signals(saved.info["signals"])

    # ------------------------------------------------------- faithful (info['signals'])
    def _build_from_signals(self, stored: Mapping) -> None:
        """Re-publish each stored signal's ``block`` VERBATIM under a role-fixed bare key
        (``value`` / ``x`` / ``centers`` / ``frame``) with its stored ``points_shape`` / ``data_shape``.

        The role -> bare-key mapping is fixed (``value`` -> ``value`` so a panel seeded ``value =
        fig_value`` binds regardless of the original signal's name), so the seeded panel + the sitemap /
        curve wiring resolve off THIS node exactly as they do off the live producer that made the save."""
        role_to_key = {"value": FIG_VALUE_KEY, "x": FIG_X_KEY,
                       "centers": FIG_CENTERS_KEY, "frame": FIG_FRAME_KEY}
        for name, entry in stored.items():
            if not isinstance(entry, Mapping):
                raise TypeError(f"saved signal {name!r} must be a mapping.")
            role = entry["role"]
            if role not in role_to_key:
                raise ValueError(f"saved signal {name!r} has unsupported role {role!r}.")
            bare = role_to_key[role]
            if bare in self._blocks:
                raise ValueError(f"saved figure has more than one signal with role {role!r}.")
            block = np.asarray(entry.get("block"))
            # The value block's declared axis label (legend / picker) falls back to ``_ylabel``; the panel's
            # RENDERED axes -- incl. a 2D colour bar -- come from the saved labels seeded into the panel
            # params (see _seed_state -> PanelCard._panel_labels), so this per-signal label is never the
            # colour-bar source and needs no 2D special case.
            label = entry["label"]
            unit = entry["unit"]
            ps = _stored_shape(entry, "points_shape", str(name))
            ds = _stored_shape(entry, "data_shape", str(name))
            if block.ndim < 3:
                raise ValueError(
                    f"saved signal {name!r} must be a canonical (R,P,*data_shape) tensor; "
                    f"got {block.shape}.")
            expected = canonical_physical_shape(block.shape[0], ps, ds)
            if tuple(block.shape) != expected:
                raise ValueError(
                    f"saved signal {name!r} must be canonical (R,P,*data_shape) {expected}; "
                    f"got {block.shape}.")
            self._blocks[bare] = block
            self._specs[bare] = SignalSpec(bare, label, unit,
                                           f"the saved figure's {role} block ({name})",
                                           points_shape=ps, data_shape=ds,
                                           dtype=block.dtype, repeat_capacity=int(block.shape[0]))
            self._role_key[role] = bare
        if FIG_VALUE_KEY not in self._blocks:
            raise ValueError("saved array figure is missing its required value-role signal.")

    # ------------------------------------------------------- structured (pulse)
    def _build_pulse(self, saved: SavedFigure) -> None:
        """A pulse figure's reproduction source is a ``PulseTableState`` (an OBJECT, not an array).  The
        hub carries only float arrays, so it cannot hold the state -- instead the node CARRIES the state
        as an attribute (``pulse_state`` / ``pulse_include_always_off``) and the seeded pulse panel reads
        it off THIS producing node via the console's ``pulse_state_provider``, exactly as a site-map panel
        reads its centres/frame off ITS producing node (auxiliary data resolved from the node, not the
        bare hub value).  The hub ``fig_value`` is a numeric PLACEHOLDER (the period count) so the shot
        clock / version gate still tick; the pulse panel renders from the resolved state, not this value."""
        resolved = saved.pulse_state()
        if resolved is None:
            raise ValueError("saved pulse figure is missing its validated pulse recipe.")
        self.pulse_state, self.pulse_include_always_off = resolved
        n_periods = float(len(self.pulse_state.periods))
        self._blocks[FIG_VALUE_KEY] = np.array([n_periods], dtype=float)   # numeric placeholder for the hub
        self._specs[FIG_VALUE_KEY] = SignalSpec(
            FIG_VALUE_KEY, str(saved.name or "pulse"), "",
            "the saved pulse figure (its PulseTableState is carried on the node, read via the provider)")
        self._role_key["value"] = FIG_VALUE_KEY

    # ------------------------------------------------------- structured (grid)
    def _build_grid(self, saved: SavedFigure) -> None:
        """A grid figure's reproduction source is a RECIPE dict (per-cell distributions / kernels), carried
        on the node exactly as a pulse figure carries its ``PulseTableState`` -- the seeded grid panel reads
        it off THIS producing node via the console's ``grid_recipe_provider``.  The hub ``fig_value`` is a
        numeric PLACEHOLDER (the cell count) so the shot clock / version gate still tick; the grid panel
        renders from the resolved recipe, not this value (mirrors :meth:`_build_pulse`)."""
        self.grid_recipe = saved.grid_recipe()
        if self.grid_recipe is None:
            raise ValueError("saved grid figure is missing its validated grid recipe.")
        n_cells = float(len(self.grid_recipe["per_cell"]))
        self._blocks[FIG_VALUE_KEY] = np.array([n_cells], dtype=float)    # numeric placeholder for the hub
        self._specs[FIG_VALUE_KEY] = SignalSpec(
            FIG_VALUE_KEY, str(saved.name or "grid"), "",
            "the saved grid figure (its replay recipe is carried on the node, read via the provider)")
        self._role_key["value"] = FIG_VALUE_KEY

    # -------------------------------------------------------- sites wiring
    @property
    def sitemap_centers_key(self) -> str:
        """The bare centres key a site-map panel resolves from this node (the console prepends the
        node prefix) -- present
        only when a centres block was stored, else blank so the console skips this node for rings."""
        return self._role_key.get("centers", "")

    @property
    def sitemap_image_key(self) -> str:
        """The bare camera-frame key a site-map panel resolves its background underlay from (the console
        prepends the prefix) --
        present only when a frame block was stored (new npz), else blank so an OLD site-map save shows
        rings + occupancy with no background image (nothing was stored to draw)."""
        return self._role_key.get("frame", "")

    # --------------------------------------------------------- scan-curve wiring
    @property
    def x_signal(self) -> str:
        """The companion x-axis signal a 1-D panel draws its curve against (the console's
        ``curve_x_provider`` matches ``y_signal`` -> ``x_signal``).  Blank when there is no x
        (a 2-D / site-map save)."""
        xk = self._role_key.get("x")
        return (self.prefix + xk) if xk else ""

    @property
    def y_signal(self) -> str:
        return self.prefix + self._role_key.get("value", FIG_VALUE_KEY)

    # ------------------------------------------------------------- hub contract
    def _bare_published_signals(self) -> frozenset:
        return frozenset(self._blocks)

    def _bare_output_specs(self) -> tuple[SignalSpec, ...]:
        # BARE-name specs; the base's output_specs re-keys them by the one prefix rule, so a
        # consumer's signal_spec(full fig_* name) lookup matches (the old override returned the
        # bare-name specs unprefixed -- a full-name lookup could never hit them).
        return tuple(self._specs[bare] for bare in self._blocks)

    def shot(self) -> dict[str, object]:
        # Publish each stored block VERBATIM (the faithful path relies on the panel's identity
        # short-circuit reading the declared points_shape/data_shape, not a reshape here).  Every block is
        # a float array (a pulse figure's block is the numeric placeholder; its PulseTableState is carried
        # on the node and read via the provider, not published to the float-only hub).
        return {bare: np.array(block) for bare, block in self._blocks.items()}


def _seed_state(saved: SavedFigure, node: LoadedFigureNode) -> TaskConsoleState:
    """The console layout that REPRODUCES the saved figure: ONE panel of the saved ``kind`` wired to
    ``node.y_signal`` (``fig_value``) with the saved ``info['view']`` restored as its params, so it opens
    exactly as saved.  The node is the single owner of the explicit value-role binding.
    EVERY kind takes this ONE path -- a pulse figure seeds a ``kind="pulse"`` panel exactly as a hist
    figure seeds a ``kind="hist"`` panel (``PanelCard`` then renders each by its kind).  ``pulse`` is
    ``panel=False`` (not offered in the live Add-Panel dropdown) but it IS a real panel kind on this
    SEED path, so it is NOT downgraded."""
    from .live import PLOT_KIND_BY_KEY, default_pulse_size

    kind = saved.kind
    pk = PLOT_KIND_BY_KEY.get(kind)
    if pk is None:
        raise ValueError(f"saved figure kind {kind!r} is not registered.")
    view = dict(saved.view)
    recipe = saved.figure_recipe or {}
    # The panel opens at the size it was SAVED at, so a reopened figure looks like it did when saved (the
    # operator can still change it via Setting).  A structured figure (pulse / grid) records ``panel_size``
    # in its recipe -> use it; a pulse with no recorded size falls back to the SAME default the preview
    # uses (default_pulse_size for its drawn content); every other kind keeps the stock 2x2 default.
    size = "2x2"
    if kind == "pulse":
        # A pulse panel carries its ``include_always_off`` (how the reproduction was drawn) as a param so
        # the reproduction re-renders with the saved "show off rows" choice.
        resolved = saved.pulse_state()
        include_off = bool(resolved[1]) if resolved is not None else False
        if resolved is not None:
            view.setdefault("include_always_off", include_off)
        recorded = recipe.get("panel_size")
        if recorded:
            size = str(recorded)
        elif resolved is not None:
            size = default_pulse_size(resolved[0], include_always_off=include_off)
    elif kind == "grid":
        # A grid panel opens at its SAVED size (recipe['panel_size']) if recorded, else the shape-driven
        # optimal_grid_size default (the SAME source the grid factory uses), and restores its saved DISPLAY
        # knobs (bins / fit / ylog) as panel params so the reopened grid draws exactly as it did when saved.
        recorded = recipe.get("panel_size")
        if recorded:
            size = str(recorded)
        else:
            shape = recipe.get("grid_shape") or (1, 1)
            from .live import optimal_grid_size
            size = optimal_grid_size(int(shape[0]), int(shape[1]))
        for key, val in dict(recipe.get("display_params") or {}).items():
            view.setdefault(str(key), val)               # saved display knobs seed the panel params
        # cmap (a 2d cell) / bins (a hist cell) are display knobs that are ALSO construction params, so the
        # recipe records them at TOP level (grid_recipe_from_cells), NOT in display_params -- seed those into
        # the panel params too, or the Setting combo falls back to the kind DEFAULT while build_grid_figure
        # renders the recorded value (UI != actual on load).  Which top-level keys count is the resolved
        # kind's declared PANEL_PARAMS (the ONE param spec), so this never hard-codes ``cmap``/``bins``.
        from .task_console import PANEL_PARAMS
        for decl in PANEL_PARAMS.get(str(recipe.get("sub_plot_kind") or ""), ()):
            if decl.key in recipe:
                view.setdefault(str(decl.key), recipe[decl.key])
    # A reproduced panel draws the SAVED axes: seed the stored labels as the panel's xlabel / ylabel
    # (and, for a colour-bar kind, zlabel) so PanelCard._panel_labels renders them instead of the kind's
    # LIVE defaults (a hist's hard-coded "Value"/"Shots", a 2D panel's ROI-derived "X (px)") -- the
    # reopened figure's axes then MATCH the data it was saved from.  ``saved.axis_labels()`` is the ONE
    # kind-aware source (drops a hist's phantom z, reads a grid's recipe axes), and its ("x label",
    # "y label", "colour bar") names map to the panel param keys.  ``setdefault`` so a saved ``view`` that
    # already pinned a label wins.
    for axis_name, label in saved.axis_labels():
        view.setdefault({"x label": "xlabel", "y label": "ylabel", "colour bar": "zlabel"}[axis_name], label)
    # An identity source expression (a bare signal name) -- signal_expr.is_identity_source keeps the
    # block's structure through the panel's short-circuit, exactly as the fig_value literal did.
    panel = PanelConfig(
        kind=kind,
        title=str(saved.name or "figure"),
        size=size,
        source=f"value = {node.y_signal}",
        inputs=[node.y_signal],
        params=view,
    )
    return TaskConsoleState(name=str(saved.name or "figure"), panels=[panel])


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
        # The right pane ALWAYS holds a real console board.  EVERY figure -- array OR pulse -- seeds a
        # normal PanelCard of its saved kind on it (a pulse figure seeds a ``kind="pulse"`` panel exactly
        # as a hist seeds ``kind="hist"``); the console / Monitor tab are never replaced.
        self._current_path: Path | None = None

        self._label_w = setting_label_width(
            ["name", "source", "kind", "x label", "y label", "z label", "unit", "data_x", "data_y",
             "points", "repeat", "saved", "view", "fit", "path", "signals"])
        # The fixed Info-column width; the console is sized to fill the REST of the screen budget beside
        # it (so the whole window stays within the shared screen-fraction rule, no board clipping).  The
        # column is WIDER than a single-row form (the tabbed facts + multi-line raw dict need room to
        # breathe) so a long label/value is not crushed into an ellipsis.
        self._info_col_w = self._label_w + scaled_px(320, minimum=240)
        # WINDOW PADDING single source: each PANE (the Info column and the embedded console) frames its
        # own L/R with ``window_pad(1)`` internally, so their card edges sit exactly ``window_pad(1)`` from
        # the window sides -- lined up under the title text, the SAME window pad as every other GUI.  The
        # root therefore adds NO L/R inset (that would double it); it supplies only the top/bottom window
        # pad and the divider between the two panes (HALF a unit, ``window_pad(0.5)``), so every gap on
        # screen is a clean multiple of the ONE WINDOW_PAD unit.
        self._pane_pad = window_pad(1)
        self._pane_gap = window_pad(0.5)

        # ONE root QHBoxLayout holding the Info card and the console holder.  Both panes carry the SAME
        # internal L/R pad and 0 top/bottom (the root's top/bottom pad frames them), so the two frames'
        # outer edges (top / bottom / sides) line up exactly -- the core alignment rule.
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, self._pane_pad, 0, self._pane_pad)
        root.setSpacing(self._pane_gap)
        root.addWidget(self._build_info_column(), 0)
        # The console holder STRETCHES (stretch = 1): the embedded TaskConsole is size-Expanding, so it
        # fills the whole right pane and its gravity board reads the real viewport width -> reflows into
        # 2+ columns (vs a frozen-width console that stacks every card in one column).
        self._console_holder = QtWidgets.QVBoxLayout()
        # The embedded TaskConsole reserves its OWN tab-card drop-shadow bleed inside its root layout (a host
        # margin here sits OUTSIDE the console and cannot un-clip a shadow already clipped inside its tree --
        # that was the real root cause of the "bottom shadow cut off"), so the holder itself adds no margin.
        self._console_holder.setContentsMargins(0, 0, 0, 0)
        holder_host = QtWidgets.QWidget()
        holder_host.setStyleSheet("background: transparent;")
        holder_host.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        holder_host.setLayout(self._console_holder)
        root.addWidget(holder_host, 1)

        # The right column ALWAYS holds a real console board -- empty until a figure is loaded -- so the
        # window opens at its full screen-fit size beside the Info column (never a bare, crammed strip).
        # A given ``path`` then reseeds it; a bad path leaves the empty board + an error in the status.
        self._build_console(None)
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

    def _load_npz(self, path: Path) -> None:
        path = Path(path)
        try:
            saved = load_figure(path)
        except Exception as exc:
            self.status.show_message(f"could not load: {str(exc).splitlines()[0][:160]}",
                                     severity="error")
            return
        # EVERY figure -- array OR pulse -- takes the SAME path: publish as ``fig_value``, seed ONE panel
        # of the saved kind that reproduces it.  A pulse figure seeds a ``kind="pulse"`` panel exactly as
        # a hist seeds a ``kind="hist"`` panel; the console / Monitor tab are never replaced.
        try:
            self._build_console(saved)
        except BaseException as exc:
            activation_lost = (
                self.node is None
                and self.console is not None
                and self.console.state.name == "figure-load-failed"
            )
            if activation_lost:
                self.saved = None
                self._current_path = None
                self._clear_info()
            with _signals_blocked(self.path_edit.edit):
                self.path_edit.setText(
                    "" if self._current_path is None else str(self._current_path)
                )
            self.status.show_message(
                f"could not activate figure: {str(exc).splitlines()[0][:160]}",
                severity="error",
            )
            return
        # Metadata and the info/path UI are part of the same generation commit.  They move only after
        # node/Hub/console activation succeeds; a render-barrier veto leaves the complete old viewer.
        self.saved = saved
        self._current_path = path
        with _signals_blocked(self.path_edit.edit):
            self.path_edit.setText(str(path))
        self._fill_info(saved)
        self.status.show_message(
            f"loaded {display_path(str(path))} -> fig_value signal; the seeded {saved.kind or 'figure'} "
            "panel reproduces it -- Add Panel to view it another way.")

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

    def _fill_info(self, saved: SavedFigure) -> None:
        for layout in (self.plot_layout, self.meas_layout, self.info_layout):
            self._clear_layout(layout)
        data_x = np.asarray(saved.data_x)
        data_y = np.asarray(saved.data_y)

        # --- Plot tab: how the figure DRAWS (kind / labels / unit / view / fit / when saved) ---
        plot_rows: list[tuple[str, object]] = [
            ("name", saved.name),
            ("kind", _kind_label(saved.kind) or saved.kind),
        ]
        # Show each axis the figure ACTUALLY draws on its own row (never the whole tuple's list repr,
        # which is not a label anything renders).  ``saved.axis_labels()`` is the ONE kind-aware source:
        # a 2D image / site map yields x, y AND a colour bar; a 1D-family kind (incl. the DISTRIBUTION)
        # yields only x, y (no phantom z); a grid yields its recipe's axes -- so the info tab matches the
        # rendered figure for every kind.
        plot_rows.extend(saved.axis_labels())
        plot_rows += [
            ("unit", saved.unit),
            ("saved", saved.saved_at),
        ]
        if saved.view:
            view_bits = ", ".join(f"{k}={v}" for k, v in saved.view.items() if v is not None)
            plot_rows.append(("view", view_bits or "(defaults)"))
        fit = saved.info.get("fit")
        if isinstance(fit, Mapping) and fit.get("func"):
            pairs = ", ".join(f"{n}={float(v):.5g}"
                              for n, v in zip(fit.get("names", []), fit.get("popt", [])))
            plot_rows.append(("fit", f"{fit['func']}: {pairs}" if pairs else str(fit["func"])))
        self._fill_rows(self.plot_layout, plot_rows)

        # --- Measurement tab: the DATA (source / shapes / points / repeat / stored signal blocks) ---
        meas_rows: list[tuple[str, object]] = [
            ("source", saved.info.get("source")),
            ("data_x", tuple(data_x.shape)),
            ("data_y", tuple(data_y.shape)),
            ("points", saved.info.get("points_done")),
            ("repeat", saved.info.get("repeat_cur")),
        ]
        if saved.info.get("size"):
            meas_rows.append(("size", saved.info.get("size")))
        signals = saved.info.get("signals")
        if isinstance(signals, Mapping) and signals:
            bits = ", ".join(
                f"{name}[{entry.get('role', '?')}]"
                for name, entry in signals.items() if isinstance(entry, Mapping))
            meas_rows.append(("signals", bits or sorted(signals)))   # a list -> _readout_field joins it (no list repr)
        if self._current_path is not None:
            meas_rows.append(("path", display_path(str(self._current_path))))
        self._fill_rows(self.meas_layout, meas_rows)

        # --- Device tab: provenance ("what the apparatus was doing"), expanded READABLY --
        # each held device gets its own small section (its snapshot keys, one row each), so the
        # operator sees the device state at a glance instead of one crushed line.  Absent / old npz:
        # a single placeholder line (no Provenance section), so the tab is never blank.
        self._fill_provenance(saved.info.get("provenance"))
        if self.info_layout.count() == 0:
            self._add_info_row("provenance", "(none recorded)")

        # --- Flow tab: the upstream DAG stored by the required provenance record.
        provenance = saved.info["provenance"]
        self.flow_view.set_graph(provenance["flow_graph"])

        # --- Raw tab: the WHOLE dict verbatim (every key the npz stored), multi-line + scrollable ---
        text = repr(dict(saved.info))
        self.raw_info.setPlainText(text)
        self.raw_info.setToolTip("The full info dict stored in the npz -- read-only, select to copy.")

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
    def _build_console(self, saved: SavedFigure | None = None) -> None:
        """(Re)build the embedded TaskConsole filling the right column -- the right pane ALWAYS holds a
        real console board (Monitor tab, Add Panel), never a replacement.

        EVERY figure takes the SAME path: :class:`LoadedFigureNode` publishes its data as the ``fig_value``
        hub signal and :func:`_seed_state` seeds ONE panel of the saved ``kind`` bound to it, so
        ``PanelCard`` reproduces it by its kind (a 2-D image, a site map, a histogram, a 1-D curve, a PULSE
        timeline -- the pulse panel's value is a ``PulseTableState`` object, published + rendered by the
        same node / seed / PanelCard machinery).  ``None`` (nothing loaded yet) is an EMPTY board, so the
        window ALWAYS shows a real console beside the Info column and Browse simply RESEEDS it (in
        place -- the console widget tree is built ONCE and only its seeded panel is swapped per load)."""
        # The hub is created ONCE (``__init__``) and REUSED for every load, so the loaded-figure slot is
        # ONE logical producer that publishes the fixed ``fig_value``/... names each load -- exactly the
        # "same signal names, new geometry" case the schema-ownership contract handles.  A reload is a
        # producer REBIND, identical to a Task-console row restart: stop the previous instance, hand its
        # schema-ownership proof to the replacement (so a kind/shape change -- 2-D image -> grid --> takes
        # the legitimate ``replace=True`` schema-version path instead of being rejected as an unrelated
        # producer clobbering ``fig_value``), then PURGE the signals the previous figure published that
        # this one does not (a prior site-map's ``fig_centers``/``fig_frame`` must not linger on a bare
        # 1-D curve).  The console reuses the same hub, so it follows the new schema version seamlessly.
        prev = self.node
        candidate = None
        if saved is not None:
            # Construct and validate the replacement without touching the shared Hub.  Publication is
            # the generation commit and therefore happens only after the old panel's render ownership
            # has been acknowledged by ``console.reseed`` below.
            candidate = LoadedFigureNode(self.hub, saved)
            state = _seed_state(saved, candidate)
            running: list = [candidate]
        else:
            state = TaskConsoleState(name="figure", panels=[])
            running = []
        if self.console is None:
            # FIRST build only: construct the embedded console once.  It is size-Expanding, so the
            # stretching holder makes it FILL the right pane and its gravity board reflows into 2+
            # columns.  ``window_px`` is only its MINIMUM (embedded mode setMinimumSize's it): the
            # budget BESIDE the Info column (window minus the fixed Info column and the ONE inter-pane
            # divider gap; height minus the top+bottom window pad), so it never starts smaller but grows.
            fit = screen_fit_window_size(self.window_ratio)
            console_w = max(scaled_px(520, minimum=380),
                            fit.width() - self._info_col_w - self._pane_gap)
            console_h = max(scaled_px(360, minimum=280), fit.height() - 2 * self._pane_pad)
            if candidate is not None:
                candidate.step()
            self.console = TaskConsole(
                hub=self.hub,
                state=state,
                running_nodes=running,
                session=None,
                scale=self._scale,
                window_px=(console_w, console_h),
                embedded=True,
            )
            self._console_holder.addWidget(self.console)
        else:
            # RESEED in place: reuse the WHOLE console widget tree (chrome / board / hub / refresh
            # timer) and rebuild only the seeded panel.  Browse-reloading used to tear the console down
            # and construct a fresh one every load -- re-realizing the entire Qt widget tree (~0.35 s of
            # the perceived load) despite this docstring always claiming it "reseeds".  This IS that
            # reseed, so a re-load only pays for the ONE panel changing, not the console chrome.
            if prev is not None:
                try:
                    stopped = prev.stop(timeout=5.0) is True and not prev.running
                except BaseException:
                    stopped = False
                if not stopped:
                    raise RuntimeError(
                        "cannot replace the loaded figure while its source node is still active"
                    )
            if candidate is not None and prev is not None:
                candidate._inherit_output_schema_ownership(prev)
            # This is the prepare boundary: barrier/teardown failure leaves the old registry, Hub and
            # ``self.node`` untouched.  The candidate has not published anything yet.
            self.console.reseed(state, running_nodes=running)
            try:
                if candidate is not None:
                    candidate.step()
                if prev is not None and hasattr(prev, "published_signals"):
                    previous = set(prev.published_signals())
                    current = set(candidate.published_signals()) if candidate is not None else set()
                    self.hub.remove_signals(previous - current)
            except BaseException:
                # The old panels are already safely torn down, so rollback means an explicit empty
                # diagnostic board rather than re-exposing old UI against a partially changed Hub.
                self.console.reseed(
                    TaskConsoleState(name="figure-load-failed", panels=[]), running_nodes=[]
                )
                failed_names = set(candidate.published_signals()) if candidate is not None else set()
                if prev is not None and hasattr(prev, "published_signals"):
                    failed_names.update(prev.published_signals())
                if failed_names:
                    self.hub.remove_signals(failed_names)
                self.node = None
                raise
        self.node = candidate
        if saved is not None:
            # A LOADED figure is STATIC: render its seeded panel NOW (synchronously) rather than waiting
            # for the console's first refresh-timer beat (~DEFAULT_UPDATE_MS) to first-draw it -- that
            # dead wait is why the panel sat BLANK for ~0.4 s after Load before the figure appeared.
            self.console.refresh_once()

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
