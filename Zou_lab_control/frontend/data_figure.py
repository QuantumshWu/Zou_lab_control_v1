"""Post-processing, fitting, unit conversion, and saving for front-end figures."""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..neutral_atom.core.fitting import (
    FitRequest,
    FitResult,
    fit_data,
    fit_image,
    fit_model,
)
from ..neutral_atom.core.raster import RegularRaster
from ..neutral_atom.core.selection import Selection, SelectedData, select_rows
from ..neutral_atom.core.signal_tensor import canonical_physical_shape

from zlc_frontend.render_style import FIT_DIM_ALPHA, PALETTE, small_fontsize

# 2D "center" fit overlay sizing (a centre DOT + the fitted-radius RING -- the pairing is fixed).  The
# dot is an ABSOLUTE-size locator (scatter ``s`` is points^2, independent of the axes scale), so it must
# be SMALL or it blots the atom out on a ~100px grid thumbnail; the ring's radius is in DATA coords (it
# shrinks with the cell), so it must be BOLD + OPAQUE to stay the readable element.  Tuned once here so a
# tiny thumbnail and the full focus view share one primitive (data_figure fit is the single source).
_CENTER_DOT_SIZE = 8       # scatter ``s`` (points^2): a small centre locator, never a signal-blotting blob
_CENTER_RING_LW = 1.8      # ring linewidth: bold enough to read the fitted radius over noise / the spot
_CENTER_RING_ALPHA = 0.9   # ring opacity: the ring is the PRIMARY readable overlay, so nearly solid


def _as_2d_y(data_y) -> np.ndarray:
    y = np.asarray(data_y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    return y


def resolve_save_base(path, stem: str) -> Path:
    """Turn a save ``path`` + ``stem`` into the extension-less base path, and mkdir its parent --
    the ONE place a figure+npz pair's location is resolved (DataFigure.save and the grid save both
    call it).  Empty / ``"."`` -> the bare ``stem``; a directory or trailing-separator path -> the
    stem inside it; a path that already has a suffix -> that path sans suffix; anything else ->
    ``<path>_<stem>`` (#C4)."""
    p = Path(path)
    if str(p) in ("", "."):
        base = Path(stem)
    elif str(p).endswith(("/", "\\")) or p.is_dir():
        base = p / stem
    elif p.suffix:
        base = p.with_suffix("")
    else:
        base = Path(f"{p}_{stem}")
    base.parent.mkdir(parents=True, exist_ok=True)
    return base


# Saved figures are an artifact boundary, not an append-only bag of optional
# ``info`` fields.  The npz has one exact current-schema envelope.
_SAVED_FIGURE_SCHEMA = "zou_lab_control.saved_figure"
_SAVED_FIGURE_KEYS = frozenset(
    {"schema", "data_x", "data_y", "plot", "signals", "recipe", "provenance", "metadata"}
)
_SAVED_PLOT_KEYS = frozenset(
    {"name", "kind", "labels", "unit", "points_done", "repeat_cur", "view", "fit", "size"}
)
_SAVED_SIGNAL_KEYS = frozenset(
    {"block", "points_shape", "data_shape", "label", "unit", "role"}
)
_SAVED_SIGNAL_ROLES = frozenset({"value", "x", "centers", "frame"})
_STRUCTURED_FIGURE_KINDS = frozenset({"pulse", "grid"})


def _object_scalar(value: object) -> np.ndarray:
    """Store one Python object as a scalar object array, never a rank inferred from its contents."""
    out = np.empty((), dtype=object)
    out[()] = value
    return out


def _required_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{where} keys must be exactly {sorted(expected)}; missing={missing}, extra={extra}.")


def _strict_int(value: object, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{where} must be an integer, got {type(value).__name__}.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{where} must be >= {minimum}, got {result}.")
    return result


def _strict_shape(value: object, where: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{where} must be a list of positive integers.")
    shape = [_strict_int(item, f"{where}[{index}]", minimum=1) for index, item in enumerate(value)]
    if not shape:
        raise ValueError(f"{where} must be non-empty.")
    return shape


def _strict_numeric_array(value: object, where: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "biufc":
        raise TypeError(f"{where} must be a numeric ndarray, got dtype {array.dtype}.")
    if array.ndim < 1:
        raise ValueError(f"{where} must have at least one axis, got shape {array.shape}.")
    return array


def _validate_plot_record(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("saved figure plot must be a mapping.")
    plot = dict(value)
    _required_keys(plot, _SAVED_PLOT_KEYS, "saved figure plot")
    for key in ("name", "kind", "unit", "size"):
        if not isinstance(plot[key], str) or (key in ("name", "kind", "size") and not plot[key]):
            raise TypeError(f"saved figure plot.{key} must be a non-empty string.")
    if not isinstance(plot["labels"], list) or len(plot["labels"]) < 2 \
            or any(not isinstance(label, str) for label in plot["labels"]):
        raise TypeError("saved figure plot.labels must be a list of at least two strings.")
    plot["points_done"] = _strict_int(plot["points_done"], "saved figure plot.points_done")
    plot["repeat_cur"] = _strict_int(plot["repeat_cur"], "saved figure plot.repeat_cur")
    if not isinstance(plot["view"], Mapping) or any(not isinstance(key, str) for key in plot["view"]):
        raise TypeError("saved figure plot.view must be a string-keyed mapping.")
    plot["view"] = dict(plot["view"])
    fit = plot["fit"]
    if fit is not None:
        if not isinstance(fit, Mapping):
            raise TypeError("saved figure plot.fit must be a mapping or None.")
        _required_keys(dict(fit), frozenset({"func", "names", "popt"}), "saved figure plot.fit")
        if not isinstance(fit["func"], str) or not isinstance(fit["names"], list) \
                or any(not isinstance(name, str) for name in fit["names"]) \
                or not isinstance(fit["popt"], list):
            raise TypeError("saved figure plot.fit must contain func:str, names:list[str], popt:list.")
        plot["fit"] = dict(fit)
    from .live import PLOT_KIND_BY_KEY
    if plot["kind"] not in PLOT_KIND_BY_KEY:
        raise ValueError(
            f"saved figure plot.kind {plot['kind']!r} is not registered; "
            f"choose from {sorted(PLOT_KIND_BY_KEY)}.")
    return plot


def _validate_signals(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError("saved figure signals must be a mapping.")
    signals: dict[str, dict[str, Any]] = {}
    roles: set[str] = set()
    for signal_name, raw in value.items():
        if not isinstance(signal_name, str) or not signal_name:
            raise TypeError("saved figure signal names must be non-empty strings.")
        if not isinstance(raw, Mapping):
            raise TypeError(f"saved signal {signal_name!r} must be a mapping.")
        entry = dict(raw)
        _required_keys(entry, _SAVED_SIGNAL_KEYS, f"saved signal {signal_name!r}")
        role = entry["role"]
        if not isinstance(role, str) or role not in _SAVED_SIGNAL_ROLES:
            raise ValueError(
                f"saved signal {signal_name!r}.role must be one of {sorted(_SAVED_SIGNAL_ROLES)}.")
        if role in roles:
            raise ValueError(f"saved figure has more than one signal with role {role!r}.")
        roles.add(role)
        if not isinstance(entry["label"], str) or not isinstance(entry["unit"], str):
            raise TypeError(f"saved signal {signal_name!r} label and unit must be strings.")
        points_shape = _strict_shape(entry["points_shape"], f"saved signal {signal_name!r}.points_shape")
        data_shape = _strict_shape(entry["data_shape"], f"saved signal {signal_name!r}.data_shape")
        block = _strict_numeric_array(entry["block"], f"saved signal {signal_name!r}.block")
        expected = canonical_physical_shape(block.shape[0], points_shape, data_shape)
        if tuple(block.shape) != expected:
            raise ValueError(
                f"saved signal {signal_name!r}.block must be canonical (R,P,*data_shape) {expected}; "
                f"got {block.shape}.")
        signals[signal_name] = {
            "block": block,
            "points_shape": points_shape,
            "data_shape": data_shape,
            "label": entry["label"],
            "unit": entry["unit"],
            "role": role,
        }
    return signals


def _validate_recipe(value: object, kind: str) -> dict[str, Any] | None:
    if kind not in _STRUCTURED_FIGURE_KINDS:
        if value is not None:
            raise ValueError(f"array figure kind {kind!r} must store recipe=None.")
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"structured figure kind {kind!r} requires a replay recipe.")
    recipe = dict(value)
    if recipe.get("kind") != kind:
        raise ValueError(
            f"saved figure kind {kind!r} requires recipe.kind={kind!r}, got {recipe.get('kind')!r}.")
    if kind == "pulse":
        expected = frozenset({"kind", "pulse_state", "include_always_off", "panel_size"})
        _required_keys(recipe, expected, "pulse replay recipe")
        if not isinstance(recipe["pulse_state"], Mapping):
            raise TypeError("pulse replay recipe.pulse_state must be a mapping.")
        if not isinstance(recipe["include_always_off"], (bool, np.bool_)):
            raise TypeError("pulse replay recipe.include_always_off must be bool.")
        if not isinstance(recipe["panel_size"], str) or not recipe["panel_size"]:
            raise TypeError("pulse replay recipe.panel_size must be a non-empty string.")
        recipe["pulse_state"] = dict(recipe["pulse_state"])
        recipe["include_always_off"] = bool(recipe["include_always_off"])
        return recipe

    common = {
        "kind", "labels", "title", "grid_shape", "panel_size", "focused_cell_index",
        "display_params", "sub_plot_kind", "per_cell",
    }
    sub_kind = recipe.get("sub_plot_kind")
    family_keys = {
        "hist": {"bins", "thresholds", "fidelities", "occupied"},
        "2d": {"cmap"},
        "1d": {"x"},
    }
    if sub_kind not in family_keys:
        raise ValueError(f"grid replay recipe.sub_plot_kind must be one of {sorted(family_keys)}.")
    _required_keys(recipe, frozenset(common | family_keys[sub_kind]), "grid replay recipe")
    if not isinstance(recipe["labels"], list) or len(recipe["labels"]) < 2 \
            or any(not isinstance(label, str) for label in recipe["labels"]):
        raise TypeError("grid replay recipe.labels must be a list of at least two strings.")
    recipe["grid_shape"] = _strict_shape(recipe["grid_shape"], "grid replay recipe.grid_shape")
    if len(recipe["grid_shape"]) != 2:
        raise ValueError("grid replay recipe.grid_shape must contain [rows, columns].")
    if not isinstance(recipe["panel_size"], str) or not recipe["panel_size"]:
        raise TypeError("grid replay recipe.panel_size must be a non-empty string.")
    focused = recipe["focused_cell_index"]
    if focused is not None:
        recipe["focused_cell_index"] = _strict_int(focused, "grid replay recipe.focused_cell_index")
    if not isinstance(recipe["display_params"], Mapping):
        raise TypeError("grid replay recipe.display_params must be a mapping.")
    recipe["display_params"] = dict(recipe["display_params"])
    if not isinstance(recipe["per_cell"], list) or not recipe["per_cell"]:
        raise ValueError("grid replay recipe.per_cell must be a non-empty list.")
    if int(np.prod(recipe["grid_shape"], dtype=np.int64)) < len(recipe["per_cell"]):
        raise ValueError("grid replay recipe.grid_shape cannot contain all per_cell entries.")
    if sub_kind == "hist":
        recipe["bins"] = _strict_int(recipe["bins"], "grid replay recipe.bins", minimum=1)
        for key in ("thresholds", "fidelities", "occupied"):
            if recipe[key] is not None and not isinstance(recipe[key], list):
                raise TypeError(f"grid replay recipe.{key} must be a list or None.")
    elif sub_kind == "2d":
        if not isinstance(recipe["cmap"], str) or not recipe["cmap"]:
            raise TypeError("grid replay recipe.cmap must be a non-empty string.")
    elif recipe["x"] is not None and not isinstance(recipe["x"], list):
        raise TypeError("grid replay recipe.x must be a list or None.")
    return recipe


def _validate_provenance(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("saved figure provenance must be a mapping.")
    provenance = dict(value)
    flow = provenance.get("flow_graph")
    if not isinstance(flow, Mapping):
        raise ValueError("saved figure provenance requires a flow_graph mapping.")
    _required_keys(dict(flow), frozenset({"nodes", "edges"}), "saved figure provenance.flow_graph")
    if not isinstance(flow["nodes"], list) or not isinstance(flow["edges"], list):
        raise TypeError("saved figure provenance.flow_graph nodes and edges must be lists.")
    if any(not isinstance(item, Mapping) for item in (*flow["nodes"], *flow["edges"])):
        raise TypeError("saved figure provenance.flow_graph entries must be mappings.")
    provenance["flow_graph"] = {"nodes": list(flow["nodes"]), "edges": list(flow["edges"])}
    return provenance


def _artifact_parts(info: Mapping[str, Any]) -> tuple[dict, dict, dict | None, dict, dict]:
    """Split the internal flat save metadata into the exact persisted records."""
    raw = dict(info)
    plot = {
        "name": str(raw.pop("name")),
        "kind": str(raw.pop("kind")),
        "labels": [str(label) for label in raw.pop("labels")],
        "unit": str(raw.pop("unit")),
        "points_done": raw.pop("points_done"),
        "repeat_cur": raw.pop("repeat_cur"),
        "view": dict(raw.pop("view")),
        "fit": raw.pop("fit"),
        "size": str(raw.pop("size")),
    }
    signals = raw.pop("signals")
    recipe = raw.pop("figure_recipe")
    provenance = raw.pop("provenance")
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise TypeError("saved figure metadata must be a string-keyed mapping.")
    return (
        _validate_plot_record(plot),
        _validate_signals(signals),
        _validate_recipe(recipe, plot["kind"]),
        _validate_provenance(provenance),
        raw,
    )


def _write_saved_npz(path: str | Path, *, data_x, data_y, info: Mapping[str, Any]) -> None:
    """Write the one current saved-figure envelope after validating every required record."""
    x = _strict_numeric_array(data_x, "saved figure data_x")
    y = _strict_numeric_array(data_y, "saved figure data_y")
    plot, signals, recipe, provenance, metadata = _artifact_parts(info)
    np.savez(
        path,
        schema=np.asarray(_SAVED_FIGURE_SCHEMA),
        data_x=x,
        data_y=y,
        plot=_object_scalar(plot),
        signals=_object_scalar(signals),
        recipe=_object_scalar(recipe),
        provenance=_object_scalar(provenance),
        metadata=_object_scalar(metadata),
    )


class DataFigure:
    """Data and post-processing handle for a front-end figure.

    A DataFigure can be created from a ``Live1D``/``Live2DDis``/``HistogramFigure``
    object, or from explicit ``fig``, ``data_x`` and ``data_y`` handles.  Pass an
    explicit ``ax`` to bind the fitting/post-processing to ONE axes of a
    multi-axes figure (e.g. a single cell of a site-histogram grid), so the same
    reusable stack works per cell instead of only on ``fig.axes[0]``.
    """

    def __init__(
        self,
        live_plot=None,
        *,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        data_x=None,
        data_y=None,
        labels: Sequence[str] | None = None,
        tools=None,
        info: Mapping[str, Any] | None = None,
        name: str | None = None,
        unit: str | None = None,
        coordinates: Mapping[str, object] | None = None,
        selection_values=None,
        selection_mode: str | None = None,
        selection_frame: str = "data",
        selection_scope: Sequence[str] = (),
    ):
        self.live_plot = live_plot
        if live_plot is not None:
            fig = live_plot.fig
            data_x = live_plot.data_x
            data_y = live_plot.data_y
            labels = getattr(live_plot, "labels", labels)
            tools = getattr(live_plot, "tools", tools)
            info = getattr(live_plot, "info", info)
            name = getattr(live_plot, "name", name)
            unit = getattr(live_plot, "unit", unit)

        if fig is None or data_x is None or data_y is None:
            raise ValueError("DataFigure needs either live_plot or fig/data_x/data_y.")

        self.fig = fig
        # The single axes this DataFigure draws on / reads limits from.  Defaults
        # to the figure's first axes (single-axes plots); an explicit ``ax`` binds
        # it to one cell of a multi-axes figure so the stack is reusable per cell.
        self._ax = ax if ax is not None else self.fig.axes[0]
        self.raster_grid = data_x if isinstance(data_x, RegularRaster) else None
        if self.raster_grid is None:
            self.data_x = np.asarray(data_x, dtype=float)
            if self.data_x.ndim == 1:
                self.data_x = self.data_x[:, None]
        else:
            self.data_x = self.raster_grid
        # Unit conversion mutates only 1-D x coordinates.  A 2-D image can carry
        # millions of (x, y) rows, so duplicating that immutable coordinate table
        # here used another ~35 MiB for a 1920x1200 frame for no purpose.
        self.data_x_original = (
            self.data_x.copy()
            if self.raster_grid is None and self.data_x.shape[1] == 1
            else self.data_x
        )
        self.data_y = _as_2d_y(data_y)
        self.labels = list(labels) if labels is not None else ["X", "Y", "Z"]
        self.info = dict(info or {})
        self.name = name or self.info.get("name") or self.info.get("class_name") or "figure"

        self.area = getattr(tools, "area", None)
        self.zoom = getattr(tools, "zoom", None)
        self.cross = getattr(tools, "cross", None)
        if live_plot is not None:
            self.area = getattr(live_plot, "area", self.area)
            self.zoom = getattr(live_plot, "zoom", self.zoom)
            self.cross = getattr(live_plot, "cross", self.cross)

        first_ax = self._ax
        # The fitting FAMILY ("1D" / "2D") comes from the plot's DECLARED
        # ``render_family`` (the single-source PLOT_KINDS table in live.py), not
        # re-derived from matplotlib artists.  Fall back to the legacy artist
        # heuristic (an image axes => 2D) when there is nothing to ask -- no
        # live_plot (a GridPlot per-cell DataFigure built with ``fig=``/``ax=``, or
        # a bare externally-constructed DataFigure) OR a plot that declares the
        # ``"auto"`` sentinel (the site map, whose family is image-only when a
        # background frame is supplied, so it stays artist-derived per figure).
        declared = getattr(live_plot, "render_family", None) if live_plot is not None else None
        self.plot_type = declared if declared in ("1D", "2D") else ("2D" if first_ax.images else "1D")

        # Plot -> named coordinates/value is one explicit adapter.  A live plot
        # owns the representation (histograms use value/bin coordinates; images
        # use x/y; traces use x), while an explicit DataFigure gets the same
        # derivation.  Selection and fitting both consume these exact rows.
        adapted = None
        if live_plot is not None:
            adapter = getattr(live_plot, "selection_rows", None)
            if callable(adapter):
                adapted = dict(adapter() or {})
        if adapted is not None:
            adapted_raster = adapted.get("raster")
            if adapted_raster is not None:
                if not isinstance(adapted_raster, RegularRaster):
                    raise TypeError("plot selection adapter raster must be RegularRaster")
                self.raster_grid = adapted_raster
            coordinates = adapted.get("coordinates", coordinates)
            selection_values = adapted.get("values", selection_values)
            selection_mode = adapted.get("mode", selection_mode)
            selection_frame = str(adapted.get("frame", selection_frame))
            selection_scope = tuple(adapted.get("scope", selection_scope))
        mode = str(selection_mode or ("xy" if self.plot_type == "2D" else "x")).lower()
        if coordinates is None and self.raster_grid is None:
            if mode == "xy" and self.data_x.shape[1] >= 2:
                coordinates = {"x": self.data_x[:, 0], "y": self.data_x[:, 1]}
            elif mode == "value":
                coordinates = {"x": self.data_x[:, 0], "value": self.data_x[:, 0]}
            else:
                coordinates = {"x": self.data_x[:, 0]}
        self._selection_coordinates = {
            str(key): np.asarray(value).reshape(-1)
            for key, value in dict(coordinates or {}).items()
        }
        if selection_values is None and self.raster_grid is not None:
            selection_values = self.data_y[:, 0].reshape(self.raster_grid.shape)
        self._selection_values = self.data_y if selection_values is None else np.asarray(selection_values)
        self._selection_mode = mode
        self._selection_frame = str(selection_frame or "data")
        self._selection_scope = tuple(str(value) for value in selection_scope)
        self.ylabel_original = self.labels[1] if len(self.labels) > 1 else first_ax.get_ylabel()
        self.unit = unit or self._infer_unit(first_ax.get_xlabel())
        self.unit_original = self.info.get("unit", self.unit)
        self._load_unit_conversion()

        self.p0 = None
        self.popt = None
        self._fit_artists = None
        self.fit_func = None
        self.text = None
        self.last_fit_result: FitResult | None = None
        self._scatter_list = []
        # The SOURCE binding (hub + producing node + wired inputs), copied from the live plot when this
        # DataFigure wraps one, so ``save`` can write the RICH npz (``info['signals']`` +
        # ``info['provenance']``) through the ONE core capture -- the SAME logic the console panel uses.
        # ``None`` (a bare array DataFigure) => ``save`` writes the basic figure+npz (old behaviour).
        self._figure_source = getattr(live_plot, "_figure_source", None)

    @staticmethod
    def _infer_unit(label: str) -> str:
        match = re.search(r"\((.+)\)$", label or "")
        return match.group(1) if match else "1"

    def _load_unit_conversion(self) -> None:
        if self.unit in ["GHz", "nm", "MHz"]:
            spl = 299792458
            self.conversion_map = {
                "nm": ("GHz", lambda x: spl / x),
                "GHz": ("MHz", lambda x: x * 1e3),
                "MHz": ("nm", lambda x: spl / (x / 1e3)),
            }
        elif self.unit in ["ns", "us", "ms"]:
            self.conversion_map = {
                "ms": ("ns", lambda x: x * 1e6),
                "ns": ("us", lambda x: x / 1e3),
                "us": ("ms", lambda x: x / 1e3),
            }
        else:
            self.conversion_map = None
        if self.conversion_map is None or self.unit_original not in self.conversion_map:
            self.unit_original = self.unit
        self._update_transform_back()

    def fit_targets(self):
        """The per-axes fit handles this figure represents.  A plain :class:`DataFigure` IS one target
        (itself), so the Edit-tab fit / clear stack loops UNIFORMLY over ``fit_targets()`` regardless of
        whether the panel is a single plot or a grid.  :class:`~.live._GridData` overrides this to yield
        its N per-cell DataFigures, so a grid fit fans out over EVERY subplot through this identical
        DataFigure primitive -- one fit conception, the multiplicity owned by the grid handle."""
        return (self,)

    def xlim(self, x_min: float, x_max: float) -> None:
        self._ax.set_xlim(x_min, x_max)
        self.fig.canvas.draw_idle()

    def ylim(self, y_min: float, y_max: float) -> None:
        self._ax.set_ylim(y_min, y_max)
        self.fig.canvas.draw_idle()

    def bind_source(self, hub, node, *, inputs, resolve_node=None, session=None):
        """Stamp WHERE this figure's data came from so :meth:`save` writes the RICH npz -- the DataFigure
        counterpart of :meth:`live.BaseLivePlot.bind_source`, for when a caller holds the DataFigure
        directly.  Returns ``self`` for chaining."""
        self._figure_source = {"hub": hub, "node": node, "inputs": list(inputs or []),
                               "resolve_node": resolve_node, "session": session}
        return self

    def _rich_capture(self) -> dict[str, Any]:
        """The ``signals`` + ``provenance`` blocks a RICH save folds into ``info``, captured through the
        ONE frontend-neutral composition point (``figure_capture.capture_rich_info``) -- the SAME call the
        grid's save and the console panel's Save route through, so every surface writes byte-identical
        rich npz.  For a BARE array plot (no source binding) the ``signals`` / device ``provenance`` are
        empty, but the save STILL folds a minimal ``provenance['flow_graph']`` (``raw data -> plot``), so
        the Flow tab of even a plain ``plot(arr).save()`` has a tree to draw.  No blanket ``except`` here:
        soft-fail is each capture's own documented contract, so a raising capture is contract drift that
        must surface (a swallowed one once saved every grid npz with NO flow graph)."""
        from zlc_data.figure_capture import capture_rich_info
        return capture_rich_info(self._figure_source)

    def save(
        self,
        path: str | Path = "",
        *,
        extra_info: Mapping[str, Any] | None = None,
        image_ext: str = "png",
    ) -> dict[str, Path]:
        """Save the image plus the one current-schema saved-figure artifact.

        The artifact always names its plot kind, typed signal blocks and replay recipe (``None`` for an
        array figure).  A source-bound figure must capture its value signal; a structured figure must
        carry its complete recipe.  Neither case silently degrades to flattened arrays."""
        current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        stem = "_".join(p for p in (self.name, current_time) if p)
        base = resolve_save_base(path, stem)
        image_path = base.with_suffix(f".{image_ext}")
        data_path = base.with_suffix(".npz")
        info = {
            **self.info,
            **self._rich_capture(),
            **dict(extra_info or {}),
            "labels": self.labels,
            "name": self.name,
            "unit": self.unit_original,
            "points_done": getattr(self.live_plot, "points_done", len(self.data_x)),
            "repeat_cur": getattr(self.live_plot, "repeat_cur", 1),
        }
        info.setdefault("view", {})
        info.setdefault("fit", self._saved_fit_info())
        if "kind" not in info and self.live_plot is not None:
            from .live import kind_for_plotter
            info["kind"] = kind_for_plotter(self.live_plot)
        if not info.get("kind"):
            raise ValueError("saving an explicit DataFigure requires info['kind']; plotter kind cannot be inferred.")
        recipe = info.get("figure_recipe")
        info.setdefault("figure_recipe", None)
        info.setdefault(
            "size",
            str((recipe or {}).get("panel_size") or getattr(self.live_plot, "_size", None) or "2x2"),
        )
        if "provenance" not in info:
            raise ValueError("saved figure capture must provide explicit provenance.")

        kind = str(info["kind"])
        signals = info.get("signals")
        if kind in _STRUCTURED_FIGURE_KINDS:
            info["signals"] = dict(signals or {})
        elif signals:
            info["signals"] = dict(signals)
        elif self._figure_source is not None:
            raise ValueError("source-bound figure save could not capture its required value signal.")
        elif kind == "sites":
            info["signals"] = self._sites_underlay_signals()
        else:
            info["signals"] = self._array_signals(kind)

        if kind not in _STRUCTURED_FIGURE_KINDS \
                and not any(entry.get("role") == "value" for entry in info["signals"].values()):
            raise ValueError(f"saved {kind!r} figure requires exactly one explicit value signal.")
        self.fig.savefig(image_path, bbox_inches="tight")
        _write_saved_npz(data_path, data_x=np.asarray(self.data_x_original), data_y=self.data_y, info=info)
        return {"figure": image_path, "data": data_path}

    def _array_signals(self, kind: str) -> dict[str, Any]:
        """Typed signal records for an unbound array figure, preserving every trailing data axis."""
        x = np.asarray(self.data_x_original)
        y = np.asarray(self.data_y)
        if kind == "2d":
            if self.raster_grid is not None:
                image_shape = tuple(self.raster_grid.shape)
            elif x.ndim == 2 and x.shape[1] >= 2 and x.shape[0] == y.shape[0]:
                image_shape = (int(np.unique(x[:, 1]).size), int(np.unique(x[:, 0]).size))
                if int(np.prod(image_shape, dtype=np.int64)) != y.shape[0]:
                    raise ValueError("2d figure coordinates do not describe one complete regular raster.")
            else:
                raise ValueError("2d figure save requires an explicit RegularRaster or complete (x,y) grid.")
            if y.ndim != 2 or y.shape[1] != 1:
                raise ValueError(f"2d figure data_y must be one scalar per raster point, got {y.shape}.")
            image = y[:, 0].reshape(image_shape)
            return {
                "value": {
                    "block": image.reshape(1, 1, *image_shape),
                    "points_shape": [1],
                    "data_shape": list(image_shape),
                    "label": str(self.labels[2] if len(self.labels) > 2 else self.labels[1]),
                    "unit": "",
                    "role": "value",
                }
            }

        def one(name: str, array: np.ndarray, label: str, role: str) -> tuple[str, dict[str, Any]]:
            shape = list(array.shape) if array.ndim else [1]
            value = array if array.ndim else array.reshape(1)
            return name, {
                "block": value.reshape(1, 1, *shape),
                "points_shape": [1],
                "data_shape": shape,
                "label": str(label),
                "unit": "",
                "role": role,
            }

        return dict((
            one("value", y, self.labels[1] if len(self.labels) > 1 else "Y", "value"),
            one("x", x, self.labels[0] if self.labels else "X", "x"),
        ))

    def _sites_underlay_signals(self) -> dict[str, Any]:
        """The self-contained ``info['signals']`` blocks a bare SITE-MAP save folds in so a reopen rebuilds
        the rings AND the background camera frame WITHOUT any device provenance -- the ``value`` occupancy,
        the ``(N, 2)`` centres and (crucially) the underlay ``frame`` -- in the SAME schema the rich
        capture (``operations.figure_capture``) writes.  The entry KEYS are the NEUTRAL role vocabulary
        (``value`` / ``centers`` / ``frame`` -- the loader binds by each entry's ``role``, never by key):
        a rich save's keys are its producer's own bare signal names, but a bare save has NO producer, so
        it must not hand-copy a live processor's private output names.  Empty when
        there is nothing to store (no bound site-map plotter / no background frame), so a save is never
        blocked.  The blocks carry the native ``(repeat, *points, *data)`` axes (occupancy ``(1, 1, N)``,
        frame ``(1, 1, H, W)``) so the reloader declares the SAME shape contract a live producer did."""
        lp = self.live_plot
        centers = getattr(lp, "data_x", None)
        background = getattr(lp, "background", None)
        if lp is None or centers is None:
            return {}
        centers = np.asarray(centers, dtype=float)[:, :2]
        n = centers.shape[0]
        occ = np.asarray(getattr(lp, "data_y", self.data_y), dtype=float).reshape(-1)[:n].reshape(1, 1, n)
        ylabel = self.labels[1] if len(self.labels) > 1 else "occupancy"
        signals: dict[str, Any] = {
            "value": {"block": occ, "points_shape": [1], "data_shape": [n],
                      "label": str(ylabel), "unit": "", "role": "value"},
            # ``centers`` is auxiliary external data, not a scan axis: one logical point whose
            # complete (N, 2) table is the datum.  Store its canonical tensor and full schema instead
            # of making the loader infer meaning from the rank.
            "centers": {"block": centers.reshape(1, 1, n, 2),
                        "points_shape": [1], "data_shape": [n, 2],
                        "label": "site centre", "unit": "px", "role": "centers"},
        }
        if background is not None:
            frame = np.asarray(background, dtype=float)
            h, w = frame.shape
            signals["frame"] = {"block": frame.reshape(1, 1, h, w), "points_shape": [1],
                                "data_shape": [h, w], "label": "camera image", "unit": "counts",
                                "role": "frame"}
        return signals

    def _saved_fit_info(self) -> dict[str, Any] | None:
        """The applied fit as a JSON-friendly dict (``func`` + ``names`` + ``popt``) for the
        saved ``info``, or ``None`` when no fit has been drawn.  ``popt`` becomes a plain list
        (never a raw ndarray) so ``np.savez`` round-trips it cleanly and a reader can read it
        without unpacking an object array."""
        if not self.fit_func or self.popt is None:
            return None
        names = list(getattr(self, "popt_str", []) or [])
        return {"func": str(self.fit_func), "names": names,
                "popt": [float(v) for v in np.asarray(self.popt, dtype=float).ravel()]}

    def _align_to_grid(self, value: float, axis: str) -> float:
        if self.plot_type != "2D" or self.live_plot is None:
            return value
        if not hasattr(self, "grid_center"):
            self.grid_center = self.data_x[0]
            x_array = np.asarray(getattr(self.live_plot, "x_array"))
            y_array = np.asarray(getattr(self.live_plot, "y_array"))
            self.step_x = abs(x_array[1] - x_array[0]) if len(x_array) > 1 else 1
            self.step_y = abs(y_array[1] - y_array[0]) if len(y_array) > 1 else 1
        if axis == "x":
            return round((value - self.grid_center[0]) / self.step_x) * self.step_x + self.grid_center[0]
        return round((value - self.grid_center[1]) / self.step_y) * self.step_y + self.grid_center[1]

    def selection(self) -> Selection:
        """Return the current semantic selection in plot coordinates.

        Every plot uses the same named-axis contract: traces/monitors select
        ``x``; distributions select ``value`` (their x axis); images/site maps
        select ``x`` and ``y``.  A drawn area wins, otherwise the current view
        limits are the selection.  Grid cells carry their immutable scope so a
        cell-local selection can never be mistaken for a global one.
        """

        area = getattr(self.area, "range", [None, None, None, None])
        if area is not None and len(area) >= 4 and area[0] is not None:
            xl, xh, yl, yh = (float(value) for value in area[:4])
        else:
            xl, xh = (float(value) for value in self._ax.get_xlim())
            yl, yh = (float(value) for value in self._ax.get_ylim())
        if self._selection_mode == "xy":
            xl, xh = (self._align_to_grid(value, "x") for value in sorted((xl, xh)))
            yl, yh = (self._align_to_grid(value, "y") for value in sorted((yl, yh)))
            return Selection.rectangle(
                xl, xh, yl, yh, frame=self._selection_frame, scope=self._selection_scope)
        if self._selection_mode == "value":
            return Selection.value(
                xl, xh, frame=self._selection_frame, scope=self._selection_scope)
        return Selection.x(xl, xh, frame=self._selection_frame, scope=self._selection_scope)

    def selected_data(self, selection: Selection | None = None) -> SelectedData:
        """Vectorized selected rows shared by fitting and external data actions.

        Missing axes or an undersized selection remain explicit; this method
        never falls back to the full dataset merely because the chosen region is
        small.
        """

        selected = selection or self.selection()
        if self.raster_grid is not None:
            # Materialising coordinate rows is reserved for this explicit data-extraction action;
            # live fitting takes the compact fit_image path below.
            return select_rows(
                self.raster_grid.coordinate_mapping(),
                np.asarray(self._selection_values).reshape(-1),
                selected,
                finite=True,
            )
        return select_rows(
            self._selection_coordinates, self._selection_values, selected, finite=True)

    def _place_text(self, ax: plt.Axes, text) -> None:
        candidates = [
            (0.025, 0.85, "left", "top"),
            (0.975, 0.85, "right", "top"),
            (0.025, 0.025, "left", "bottom"),
            (0.975, 0.025, "right", "bottom"),
            (0.5, 0.025, "center", "bottom"),
            (0.5, 0.85, "center", "top"),
        ]
        renderer = ax.figure.canvas.get_renderer()
        best = candidates[0]
        best_overlap = float("inf")
        if self.plot_type == "1D":
            pts = np.column_stack([self.data_x[:, 0], self.data_y[:, 0]])
            pts = pts[np.isfinite(pts).all(axis=1)]
            pts_disp = ax.transData.transform(pts[:: max(1, len(pts) // 1000)]) if len(pts) else np.empty((0, 2))
        else:
            pts_disp = np.empty((0, 2))

        for cand in candidates:
            text.set_position(cand[:2])
            text.set_ha(cand[2])
            text.set_va(cand[3])
            # Measure the candidate off the already-fetched ``renderer``: ``set_position`` marks the
            # text stale so ``get_window_extent`` recomputes its layout on the next call.  A full
            # ``canvas.draw()`` per candidate (6x 300dpi rasters just to size a text box, on the
            # every-tick live-fit path) was pure waste -- the renderer + transAxes already carry
            # everything the bbox needs.
            bbox = text.get_window_extent(renderer).expanded(1.05, 1.1)
            if len(pts_disp) == 0:
                overlap = 0
            else:
                overlap = int(
                    np.sum(
                        (pts_disp[:, 0] >= bbox.x0)
                        & (pts_disp[:, 0] <= bbox.x1)
                        & (pts_disp[:, 1] >= bbox.y0)
                        & (pts_disp[:, 1] <= bbox.y1)
                    )
                )
            if overlap < best_overlap:
                best_overlap = overlap
                best = cand
        text.set_position(best[:2])
        text.set_ha(best[2])
        text.set_va(best[3])

    def _display_popt(self, popt, names: Sequence[str], is_display: bool = True) -> None:
        formatted = []
        for name, value in zip(names, popt):
            formatted.append(f"{name}={float(value):.5g}")
        result = f"{self.formula_str}\n" + "\n".join(formatted)

        if is_display:
            if self.text is None:
                self.text = self._ax.text(
                    0.5,
                    0.5,
                    result,
                    transform=self._ax.transAxes,
                    color=PALETTE["fit_text"],
                    ha="center",
                    va="center",
                    fontsize=small_fontsize(),
                )
            else:
                self.text.set_text(result)
            self._place_text(self._ax, self.text)
        elif self.text is not None:
            self.text.remove()
            self.text = None

        # DIM the data lines ONLY on the 1-D line families (the FIT_DIM_ALPHA overlay of #99a4757).
        # An IMAGE family (2D / site map) has no data LINE to fade -- its fit is a dot/ring overlay on
        # the frame -- so dimming there would only touch unrelated artists (a side-distribution line);
        # gate it off so the fit-center overlay never changes the image's appearance (#11).
        if self.plot_type not in ("2D", "SITES"):
            self._dim_data_lines()
            if self.plot_type == "1D" and len(self.data_y) < 2000:
                self._line_to_scatter()
        # No draw here: the rendering adapter places the fit curve immediately after and
        # issues the single terminal draw_idle.  Drawing here too was a premature + redundant full
        # repaint -- it painted the annotation BEFORE the curve existed, then the whole figure again.

    def _draw_fit_result(self, result: FitResult, *, is_display: bool) -> None:
        """Thin Matplotlib adapter for the headless core result."""

        self.last_fit_result = result
        self.fit_func = result.model
        self.popt = result.popt
        self.popt_str = list(result.names)
        if not result.valid:
            # A failed live fit must remove the previous overlay just as the
            # processor publishes NaN/valid=0; stale success is never displayed.
            self.clear()
            self.fit_func = result.model
            self.last_fit_result = result
            return

        spec = fit_model(result.model)
        self.formula_str = spec.formula
        self._display_popt(result.parameters, result.names, is_display)
        ax = self._ax
        if spec.family == "1d":
            x = np.asarray(self._selection_coordinates["x"], dtype=float)
            order = np.argsort(x)
            yfit = spec.function(x[order], *result.parameters)
            self._fit_artists = ax.plot(
                x[order], yfit, color=PALETTE["fit_right"], linestyle="-",
                linewidth=2, alpha=0.5)
        else:
            # #11: the centre dot + radius ring are a READOUT overlay on the image, NOT new data.
            # Snapshot the view -- xlim/ylim, both autoscale flags, and the image clim -- and restore it
            # after adding the artists, so a scatter/patch's dataLim can never expand the 2-D image's
            # displayed range or drop its colour limit.  The artists are clipped to the axes so a ring
            # partly outside the frame never draws beyond it.
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            auto_x, auto_y = ax.get_autoscalex_on(), ax.get_autoscaley_on()
            clims = [(im, im.get_clim()) for im in ax.get_images()]
            params = result.parameter_map()
            x0, y0, radius = params["x0"], params["y0"], abs(params["radius"])
            dot = ax.scatter(x0, y0, color=PALETTE["fit_right"], s=_CENTER_DOT_SIZE, clip_on=True)
            circle = matplotlib.patches.Circle(
                (x0, y0), radius=radius, edgecolor=PALETTE["fit_right"], facecolor="none",
                linewidth=_CENTER_RING_LW, alpha=_CENTER_RING_ALPHA, clip_on=True)
            ax.add_patch(circle)
            self._fit_artists = [dot, circle]
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_autoscalex_on(auto_x)
            ax.set_autoscaley_on(auto_y)
            for image, clim in clims:
                image.set_clim(*clim)
        self.fig.canvas.draw_idle()

    def fit_model(
        self,
        model: str,
        *,
        request: FitRequest | None = None,
        selection: Selection | None = None,
        initial: Sequence[float] | None = None,
        fixed: Mapping[str, float] | None = None,
        is_display: bool = True,
        is_fit: bool = True,
        sample_budget: int = 12_000,
        max_evaluations: int = 4_000,
    ) -> FitResult:
        """Fit this plot through the single headless fitting engine."""

        spec = fit_model(model)
        if request is None:
            request = FitRequest(
                spec.key,
                selection=selection or self.selection(),
                initial=None if initial is None else tuple(float(value) for value in initial),
                fixed=dict(fixed or {}),
                sample_budget=int(sample_budget),
                max_evaluations=int(max_evaluations),
                coordinate_frame=self._selection_frame,
                solve=bool(is_fit),
            )
        actual_family = "2d" if self.raster_grid is not None \
            or "y" in self._selection_coordinates else "1d"
        if spec.family != actual_family:
            result = FitResult.invalid(
                spec,
                f"{spec.key} is a {spec.family} model but this plot supplies {actual_family} data",
                coordinate_frame=request.coordinate_frame,
            )
        elif self.raster_grid is not None:
            result = fit_image(
                np.asarray(self._selection_values).reshape(self.raster_grid.shape),
                request,
                grid=self.raster_grid,
            )
        else:
            result = fit_data(self._selection_coordinates, self._selection_values, request)
        # Exactly one overlay at a time.  Remove the old artists without changing
        # the current semantic selection, then draw the new result if valid.
        if self._fit_artists is not None or self.text is not None:
            self.clear()
        self._draw_fit_result(result, is_display=is_display)
        return result

    def fit(self, model: str, **kwargs) -> FitResult:
        """The sole public fit entry: choose a registered model by key."""

        return self.fit_model(model, **kwargs)

    def clear(self) -> None:
        if self.text is not None:
            self.text.remove()
            self.text = None
        if self._fit_artists is not None:
            for artist in self._fit_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            self._fit_artists = None
        self._scatter_to_line()
        self._restore_data_lines()
        self.fig.canvas.draw_idle()

    def _fit_overlay_lines(self):
        """The persistent data Line2D objects the fit overlay dims/restores (the plot's live lines,
        or this figure's own axis lines for a standalone DataFigure) -- the ONE list both halves use."""
        return getattr(self.live_plot, "lines", None) if self.live_plot is not None else self._ax.lines

    def _dim_data_lines(self) -> None:
        """DIM half of the fit overlay: fade the data lines to ``style.FIT_DIM_ALPHA`` so the fit curve
        reads clearly, recording each line's ORIGINAL alpha ONCE (a per-tick re-dim must not capture the
        already-dimmed value as the 'original')."""
        for line in self._fit_overlay_lines():
            if not hasattr(line, "set_alpha"):
                continue
            if not hasattr(line, "_zlc_fit_pre_alpha"):
                line._zlc_fit_pre_alpha = line.get_alpha()
            line.set_alpha(FIT_DIM_ALPHA)

    def _restore_data_lines(self) -> None:
        """RESTORE half: put each data line back to the ORIGINAL alpha the dim recorded (the user's exact
        opacity), then drop the sentinel.  Same owner as the dim, so a fit clear can never leave data dimmed."""
        for line in self._fit_overlay_lines():
            if not hasattr(line, "set_alpha"):
                continue
            original = getattr(line, "_zlc_fit_pre_alpha", None)
            line.set_alpha(1.0 if original is None else original)
            if hasattr(line, "_zlc_fit_pre_alpha"):
                del line._zlc_fit_pre_alpha

    def _line_to_scatter(self) -> None:
        if self.plot_type != "1D" or self._scatter_list:
            return
        ax = self._ax
        line = self._ax.lines[0] if self._ax.lines else None
        if line is None:
            return
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata())
        sc = ax.scatter(x, y, s=20, color=PALETTE["data_scatter"], edgecolors="none")
        self._scatter_list.append(sc)
        line.set_visible(False)

    def _scatter_to_line(self) -> None:
        for sc in self._scatter_list:
            try:
                sc.remove()
            except Exception:
                pass
        self._scatter_list = []
        if self.fig.axes and self._ax.lines:
            self._ax.lines[0].set_visible(True)

    def _update_transform_back(self) -> None:
        transforms: list[Callable[[Any], Any]] = []
        temp_unit = self.unit
        while self.conversion_map is not None and temp_unit != self.unit_original:
            try:
                next_unit, conv_func = self.conversion_map[temp_unit]
            except KeyError:
                break
            transforms.append(conv_func)
            temp_unit = next_unit

        def _identity(x):
            return x

        def _composed(x):
            out = x
            for func in transforms:
                out = func(out)
            return out

        self.transform_back = _composed if transforms else _identity

    def _update_unit(self, transform: Callable[[Any], Any]) -> None:
        ax = self._ax
        for line in ax.lines:
            data_x = np.asarray(line.get_xdata())
            if data_x.size == 2 and np.array_equal(data_x, np.array([0, 1])):
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                line.set_xdata(np.where(data_x != 0, transform(data_x), np.inf))
        if ax.lines:
            self.data_x = np.asarray(ax.lines[0].get_xdata()).reshape(-1, 1)
        xlim = ax.get_xlim()
        ax.set_xlim(transform(xlim[0]), transform(xlim[1]))

        if self.area is not None and self.area.range[0] is not None:
            self.area.range[0] = transform(self.area.range[0])
            self.area.range[1] = transform(self.area.range[1])
            try:
                self.area.selector.extents = tuple(self.area.range)
            except Exception:
                pass
        if self.cross is not None and self.cross.xy is not None:
            new_x = transform(self.cross.xy[0])
            self.cross.xy[0] = new_x
            if getattr(self.cross, "vline", None) is not None:
                self.cross.vline.set_xdata([new_x, new_x])
            if getattr(self.cross, "point", None) is not None:
                self.cross.point.set_xdata([new_x])
        if self._fit_artists is not None and self.fit_func:
            prev_fit = self.fit_func
            self.clear()
            try:
                self.fit(prev_fit, is_display=True)
            except Exception:
                pass

    def change_unit(self) -> None:
        """Cycle wavelength/frequency or time units for 1D plots."""
        if self.plot_type == "2D" or self.conversion_map is None:
            return
        new_unit, conversion_func = self.conversion_map[self.unit]
        ax = self._ax
        old_xlabel = ax.get_xlabel()
        if re.search(r"\((.+)\)$", old_xlabel):
            ax.set_xlabel(re.sub(r"\((.+)\)$", f"({new_unit})", old_xlabel))
        else:
            ax.set_xlabel(f"{old_xlabel} ({new_unit})")
        self.unit = new_unit
        self._update_transform_back()
        self._update_unit(conversion_func)
        self.fig.canvas.draw_idle()

    def change_cmap(self, cmap: str) -> None:
        """Change image colormap and matching 2D selector colors."""
        if self.plot_type != "2D":
            return
        try:
            base_cmap = matplotlib.colormaps[cmap]
        except Exception:
            base_cmap = plt.get_cmap(cmap)
        new_cmap = base_cmap.copy()
        bad_color = getattr(self.live_plot, "bad_color", "white")
        new_cmap.set_bad(bad_color)

        ax0 = self._ax
        if not ax0.images:
            return
        mappable = ax0.images[0]
        mappable.set_cmap(new_cmap)
        cbar = getattr(self.live_plot, "cbar", None)
        if cbar is not None:
            cbar.update_normal(mappable)
        for attr, value in (("line_l", new_cmap(0.0)), ("line_h", new_cmap(0.95))):
            line = getattr(self.live_plot, attr, None)
            if line is not None:
                line.set_color(value)
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Reopening a saved figure (the read-back counterpart of ``DataFigure.save``).
# ---------------------------------------------------------------------------

#: The view-state keys ``SavedFigure`` restores and that ``plot()`` accepts as DATA options.
#: An override only reaches ``plot()`` when it is in THIS set (a saved layout may carry a
#: knob that a re-interpreted kind does not take, e.g. ``cmap`` on a 1-D plot -- it is simply
#: dropped rather than raising).  Geometry / dpi / typography are NOT here: they are frontend-
#: owned (the sealed-API contract), so a saved figure can never smuggle them back in.
_VIEW_PLOT_KWARGS: tuple[str, ...] = (
    "relim_mode", "fixed_lo", "fixed_hi", "cmap", "bins", "thresholds", "labels",
)

#: ``PanelEditor`` stores its Setting/Edit view knobs under this sub-dict of ``info`` (so the
#: raw payload stays tidy); ``relim`` there is confocal naming for ``plot()``'s ``relim_mode``.
_VIEW_INFO_KEY = "view"


def _view_to_plot_kwargs(view: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a saved ``info['view']`` dict into the DATA kwargs ``plot()`` accepts.

    The stored ``relim`` (confocal naming) maps to ``plot()``'s ``relim_mode``, and ``fixed_lo``/
    ``fixed_hi`` are only forwarded when the mode is actually ``fixed`` (mirroring the live panel
    builder, which omits them otherwise so the plotter keeps its autoscale).  ``unit`` / ``size`` /
    ``repeat_mode`` live in ``info`` (not ``plot()`` kwargs) and are handled by the caller."""
    out: dict[str, Any] = {}
    relim = view.get("relim", view.get("relim_mode"))
    if relim:
        out["relim_mode"] = str(relim)
    if str(relim) == "fixed":
        for k in ("fixed_lo", "fixed_hi"):
            if view.get(k) is not None:
                out[k] = float(view[k])
    cmap = view.get("cmap")
    if cmap:
        out["cmap"] = str(cmap)
    return out


class SavedFigure:
    """A lightweight, hardware-free handle for a ``.npz`` written by :meth:`DataFigure.save`.

    It answers "what was saved" (:meth:`info_summary`), lists the plot kinds the saved data can
    be viewed as (:meth:`compatible_kinds`), and re-renders it (:meth:`plot`) through the SAME
    :func:`~.live.plot` factory the live figure used -- so the saved kind reproduces the original
    figure and any other compatible kind re-interprets the SAME ``data_x`` / ``data_y`` with a
    different plotter.  Build one with :func:`load_figure`."""

    def __init__(self, *, data_x, data_y, info: Mapping[str, Any], path=None):
        self.path = Path(path) if path is not None else None
        self.data_x = np.asarray(data_x)
        self.data_y = np.asarray(data_y)
        self.info = dict(info)
        required = _SAVED_PLOT_KEYS | {"signals", "figure_recipe", "provenance"}
        missing = sorted(required - set(self.info))
        if missing:
            raise ValueError(f"SavedFigure requires the validated current schema; missing={missing}.")
        self.view = dict(self.info[_VIEW_INFO_KEY])

    # -- validated artifact accessors ----------------------------------------------------------
    @property
    def kind(self) -> str:
        return self.info["kind"]

    @property
    def figure_recipe(self) -> Mapping[str, Any] | None:
        """The REPLAY RECIPE a structured figure stores so it can be re-rendered FAITHFULLY (not from
        ``data_x`` / ``data_y``, which for a structured figure are only a lossy fallback).  A recipe is
        ``{"kind": <family>, ...}`` -- e.g. a pulse figure stores ``{"kind": "pulse", "pulse_state":
        <PulseTableState.to_dict()>, "include_always_off": ...}``.  ``None`` for an ordinary array figure
        (a scan / hist / site map), whose ``data_x`` / ``data_y`` ARE the faithful source.  The dispatch
        is by ``recipe['kind']``, so a new structured-figure family plugs in without touching this seam."""
        return self.info["figure_recipe"]

    @property
    def name(self) -> str:
        return self.info["name"]

    @property
    def labels(self) -> list[str]:
        return list(self.info["labels"])

    @property
    def unit(self) -> str:
        return self.info["unit"]

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        return {"data_x": tuple(self.data_x.shape), "data_y": tuple(self.data_y.shape)}

    @property
    def saved_at(self) -> str | None:
        """The save timestamp, recovered from the ``<name>_<YYYY_MM_DD_HH_MM_SS>`` file name
        (the format :meth:`DataFigure.save` writes) or ``None`` if the name does not carry one."""
        if self.path is None:
            return None
        m = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", self.path.stem)
        return m.group(1) if m else None

    def compatible_kinds(self) -> list[str]:
        """The panel plot kinds this saved data can be viewed as, derived from the data SHAPE and
        the ``PLOT_KINDS`` ``render_family`` (never a hard-coded list): a 2-column ``data_x`` (a 2-D
        scan) offers the ``"2D"`` families (image / site map); a 1-column ``data_x`` offers the
        ``"1D"`` families (line / rolling trace / distribution).  The saved kind is always first so
        it reproduces the original figure.

        A STRUCTURED figure (one carrying a ``figure_recipe``) offers ONLY its recipe kind: its
        ``data_x`` / ``data_y`` are a lossy fallback, so re-interpreting them as a line / hist would
        NOT be a faithful view -- the recipe kind is the one true way to draw it."""
        from .live import PLOT_KINDS

        recipe = self.figure_recipe
        if recipe is not None:
            return [str(recipe.get("kind"))]

        is_2d = self.data_x.ndim == 2 and self.data_x.shape[1] >= 2
        want = "2D" if is_2d else "1D"
        kinds: list[str] = []
        for pk in PLOT_KINDS:
            if not pk.panel:
                continue
            # The site map declares ``render_family="auto"`` (image-only when it has a frame), but it
            # is a 2-D VIEW -- it needs (N, 2) centres as data_x -- so it belongs with the 2-D family.
            fam = "2D" if pk.render_family == "auto" else pk.render_family
            if fam == want:
                kinds.append(pk.key)
        saved = self.kind
        if saved in kinds:                       # saved kind FIRST (reproduces the original figure)
            kinds = [saved] + [k for k in kinds if k != saved]
        elif saved:
            kinds = [saved] + kinds
        return kinds

    def axis_labels(self) -> list[tuple[str, str]]:
        """The per-axis labels THIS figure actually DRAWS, kind-aware -- ``[(axis_name, label), ...]``.

        A 2D image / site map draws x, y AND a colour bar (its z quantity = ``labels[2]``); every
        1D-family kind (line / rolling trace / DISTRIBUTION) draws only x and y -- a stored 3rd label is
        the dormant ``BaseLivePlot`` ``("X","Y","Z")`` default (or a distribution's redundant "Population"),
        never rendered, so it is dropped.  A grid carries its axis labels in the replay recipe
        (``build_grid_figure`` reads them), NOT the top-level ``labels``.  The colour-bar test reads the
        ONE ``PLOT_KINDS`` ``render_family`` source (``"2D"`` / ``"auto"``), never a hard-coded kind list --
        so a hist never shows a phantom "z" and a grid shows its recipe's axes instead of nothing."""
        from .live import PLOT_KIND_BY_KEY

        recipe = self.figure_recipe
        source = (recipe.get("labels") if recipe is not None else self.labels) or ()
        labels = [str(x) for x in source]
        names = ["x label", "y label"]
        pk = PLOT_KIND_BY_KEY.get(str(self.kind or ""))
        if pk is not None and pk.render_family in ("2D", "auto"):
            names.append("colour bar")
        return [(n, labels[i]) for i, n in enumerate(names) if i < len(labels) and labels[i].strip()]

    def info_summary(self) -> str:
        """A human-readable multi-line answer to "what is in this file": name, source, kind, labels,
        unit, array shapes, points/repeat progress, the saved view state and any applied fit."""
        rows: list[tuple[str, Any]] = [
            ("name", self.name),
            ("source", self.info.get("source")),
            ("kind", self.kind),
            ("labels", self.labels),
            ("unit", self.unit),
            ("data_x shape", tuple(self.data_x.shape)),
            ("data_y shape", tuple(self.data_y.shape)),
            ("points_done", self.info.get("points_done")),
            ("repeat_cur", self.info.get("repeat_cur")),
            ("saved_at", self.saved_at),
        ]
        if self.info.get("size"):
            rows.append(("size", self.info.get("size")))
        if self.view:
            view_bits = ", ".join(f"{k}={v}" for k, v in self.view.items() if v is not None)
            rows.append(("view", view_bits or "(defaults)"))
        fit = self.info.get("fit")
        if isinstance(fit, Mapping) and fit.get("func"):
            pairs = ", ".join(f"{n}={float(v):.5g}"
                              for n, v in zip(fit.get("names", []), fit.get("popt", [])))
            rows.append(("fit", f"{fit['func']}: {pairs}" if pairs else str(fit["func"])))
        width = max((len(k) for k, _ in rows), default=0)
        header = f"SavedFigure({self.path.name})" if self.path is not None else "SavedFigure"
        lines = [header] + [f"  {k.ljust(width)} : {v}" for k, v in rows if v is not None]
        return "\n".join(lines)

    def plot(self, kind: str | None = None, size: str | None = None, **overrides):
        """Re-render this saved figure as a :class:`DataFigure` -- DATA-viewing semantics: the SAME
        ``data_x`` / ``data_y`` are handed to :func:`~.live.plot` under ``kind`` (the saved kind by
        default, which reproduces the original figure; any :meth:`compatible_kinds` value instead
        re-interprets the data with a different plotter).  The saved view state (relim / fixed lo-hi /
        cmap / labels) seeds the plotter and ``overrides`` win over it.  Only DATA kwargs
        (:data:`_VIEW_PLOT_KWARGS`) reach ``plot()`` -- geometry / dpi / typography stay frontend-owned,
        and a saved knob a re-interpreted kind does not accept is dropped, never raised.

        When this figure carries a ``figure_recipe`` (a STRUCTURED figure such as a pulse timeline),
        the reproduction goes through :meth:`_replay_recipe` instead: it re-renders from the recipe's
        own source (e.g. rebuilds the ``PulseTableState`` and re-draws the pulse figure), so the reopened
        figure is FAITHFUL (all channels / analog traces / brackets), not a flattened line off the
        fallback arrays.  ``kind`` / ``size`` / ``overrides`` are ignored for a recipe figure -- the
        recipe is self-describing."""
        recipe = self.figure_recipe
        if recipe is not None:
            if kind is not None and kind != recipe["kind"]:
                raise ValueError(
                    f"structured figure kind {recipe['kind']!r} cannot be reinterpreted as {kind!r}.")
            if size is not None or overrides:
                raise ValueError("structured figure view is fixed by its replay recipe.")
            return self._replay_recipe(recipe)
        if kind is not None and kind not in self.compatible_kinds():
            raise ValueError(f"kind {kind!r} is not compatible with saved kind {self.kind!r}.")
        return self._plot_from_arrays(kind, size, **overrides)

    def _replay_recipe(self, recipe: Mapping[str, Any]):
        """Re-render a STRUCTURED figure from its ``figure_recipe`` -- a faithful reproduction from the
        recipe's OWN source, not ``data_x`` / ``data_y``.  Dispatch is by ``recipe['kind']``,
        so each structured-figure family owns its rebuild here (and a new family adds a branch).  Returns the
        family's figure handle (a single-axes :class:`DataFigure` for pulse; a composite grid handle for
        grid), each carrying ``.fig`` + ``.save``.

        ``pulse`` -> rebuild the ``PulseTableState`` and re-draw the pulse figure through the SAME
        :func:`~.live.build_pulse_preview_plot` the plot layer owns, so every digital channel / analog
        bus trace / repeat bracket comes back exactly as saved.  The renderer lives in the PLOT LAYER
        (``live.build_pulse_preview_plot`` / ``default_pulse_size``), NOT the pulse_gui app -- ``_replay_pulse``
        imports it LAZILY from ``.live`` (only when a pulse recipe is actually reopened) so the ordinary
        array-figure reload path stays free of that import.  The data layer never imports pulse_gui (a
        contract test forbids it)."""
        rkind = str(recipe.get("kind") or "")
        if rkind == "pulse":
            return self._replay_pulse(recipe)
        if rkind == "grid":
            return self._replay_grid(recipe)
        raise ValueError(f"unsupported structured figure recipe kind {rkind!r}.")

    def _pulse_state_dict(self, recipe: Mapping[str, Any]) -> dict:
        """The ``PulseTableState.to_dict()`` payload to rebuild the pulse figure from -- resolved from ONE
        serialization format (``PulseTableState.to_dict``) with provenance PREFERRED over the recipe copy.

        A fired pulse's provenance already carries the applied state: the sequencer's ``.snapshot()``
        records ``last_payload_json = json.dumps(PulseTableState.to_dict())`` (the exact same format), and
        the rich save folds that under ``info['provenance']['devices']['sequencer']``.  So when the save
        HAS provenance with a sequencer payload, THAT is the reproduction source (single source with the
        device state the run was taken under -- no second copy needed); the recipe's own ``pulse_state``
        is the FALLBACK for a pure preview that was never fired (the standalone editor has no device, so
        the editor's current ``PulseTableState`` is the only truth for what the preview drew)."""
        prov = self.info.get("provenance")
        devices = prov.get("devices") if isinstance(prov, Mapping) else None
        seq = devices.get("sequencer") if isinstance(devices, Mapping) else None
        payload = seq.get("last_payload_json") if isinstance(seq, Mapping) else None
        if payload is not None:
            import json
            data = json.loads(payload) if isinstance(payload, str) else payload
            if not isinstance(data, Mapping):
                raise TypeError("sequencer provenance last_payload_json must decode to a mapping.")
            return dict(data)                                # the device-applied state (provenance-sourced)
        return dict(recipe["pulse_state"])                  # pure preview: editor state is the source

    def pulse_state(self):
        """The reproduced pulse ``(PulseTableState, include_always_off)`` for a ``kind="pulse"`` save --
        the single object the WHOLE reproduction uses: the notebook ``.plot()`` builds its figure from it,
        AND the Task console seeds a pulse PANEL bound to it (published as ``fig_value``), so both draw the
        SAME faithful timeline through the ONE renderer.  Provenance (a fired pulse) is preferred over the
        recipe's own copy (see :meth:`_pulse_state_dict`).  ``None`` when this is not a pulse figure."""
        recipe = self.figure_recipe
        if recipe is None or str(recipe.get("kind") or "") != "pulse":
            return None
        # The renderer may not import the pulse compiler (the import DAG makes
        # zlc_frontend a leaf over zlc_data), so the composition root hands the
        # constructor down; see zlc_frontend.domain_ports.
        from zlc_frontend.domain_ports import pulse_state_from_dict

        state = pulse_state_from_dict(self._pulse_state_dict(recipe))
        return state, bool(recipe.get("include_always_off", True))

    def _replay_pulse(self, recipe: Mapping[str, Any]) -> "DataFigure":
        """Rebuild the pulse figure: :meth:`pulse_state` (provenance-preferred source) ->
        :func:`~.live.build_pulse_preview_plot` -> its :class:`DataFigure`.  The pulse RENDER lives in the
        PLOT LAYER (live.py), never the pulse_gui app -- so notebook replay, the editor preview and a
        seeded console panel all draw through the ONE renderer, identical code."""
        state, include_always_off = self.pulse_state()
        from .live import build_pulse_preview_plot, default_pulse_size

        # The reproduction opens at the size it was SAVED at (recipe['panel_size']), else the SAME optimal
        # default the preview / the console seed use -- so notebook replay matches the reopened panel.
        recorded = recipe.get("panel_size")
        size = str(recorded) if recorded else default_pulse_size(state, include_always_off=include_always_off)
        plotter, _channels, _repeat = build_pulse_preview_plot(
            state, include_always_off=include_always_off, size=size)
        df = plotter.to_data_figure()
        df.info = {**df.info, **{k: v for k, v in self.info.items() if k != "labels"}}
        df.name = self.name or df.name
        return df

    def grid_recipe(self) -> Mapping[str, Any] | None:
        """The per-site GRID replay recipe for a ``kind="grid"`` save -- the single object the WHOLE grid
        reproduction uses (the notebook ``.plot()`` rebuilds its figure from it, AND the Task console seeds
        a ``kind="grid"`` PANEL bound to it, published as ``fig_value``), so both redraw the SAME faithful
        grid through the ONE :func:`~.live.build_grid_figure` builder.  This is the grid counterpart of
        :meth:`pulse_state`.  ``None`` when this is not a grid figure."""
        recipe = self.figure_recipe
        if recipe is None or str(recipe.get("kind") or "") != "grid":
            return None
        return recipe

    def _replay_grid(self, recipe: Mapping[str, Any]):
        """Rebuild the per-site grid figure: the stored recipe -> :func:`~.live.build_grid_figure` -> its
        composite grid ``DataFigure`` (a :class:`~.live._GridData`).  The grid RENDER lives in the PLOT
        LAYER (live.py), so notebook replay and a seeded console panel both draw through the ONE builder,
        identical code (the grid counterpart of :meth:`_replay_pulse`).

        Returns the grid's composite handle (it carries the whole-grid ``.fig`` for display + a per-cell
        :class:`DataFigure` via ``.cell(k)`` + its own recipe-aware ``.save``), not a single-axes
        ``DataFigure`` -- a grid is intrinsically multi-axes, so there is nothing to flatten to one axes."""
        from .live import build_grid_figure

        # Open at the saved size if the recipe recorded one, else the stock default -- matching the panel seed.
        recorded = recipe.get("panel_size")
        size = str(recorded) if recorded else "2x2"
        plotter = build_grid_figure(recipe, size=size, display=False)
        return plotter.to_data_figure()

    def _plot_from_arrays(self, kind: str | None = None, size: str | None = None, **overrides):
        """The ordinary array-figure reproduction (``data_x`` / ``data_y`` -> :func:`~.live.plot`) -- the
        ONE array path, used by :meth:`plot` for a non-recipe figure."""
        from .live import panel_plot_spec, plot as _plot

        use_kind = kind or self.kind
        use_size = size or self.info["size"]
        kwargs = _view_to_plot_kwargs(self.view)
        if self.labels is not None:
            kwargs.setdefault("labels", self.labels)
        kwargs.update(overrides)
        kwargs = {k: v for k, v in kwargs.items() if k in _VIEW_PLOT_KWARGS}
        spec = panel_plot_spec(use_size)
        plotter = _plot(self.data_x, self.data_y, kind=use_kind, _spec=spec,
                        display=False, data_figure=True, update=False, **kwargs)
        df = getattr(plotter, "data_figure", None) or plotter.to_data_figure()
        df.info = {**df.info, **{k: v for k, v in self.info.items() if k != "labels"}}
        if self.name:
            df.name = self.name
        return df


def _load_saved_npz(path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read and validate the one current saved-figure artifact envelope.

    SECURITY: object records are trusted local experiment artifacts and therefore use pickle.  The
    exact schema/key/type checks happen before a :class:`SavedFigure` is constructed."""
    with np.load(str(path), allow_pickle=True) as data:  # noqa: S301 - local experiment tool, trusted input only
        actual = frozenset(data.files)
        if actual != _SAVED_FIGURE_KEYS:
            missing = sorted(_SAVED_FIGURE_KEYS - actual)
            extra = sorted(actual - _SAVED_FIGURE_KEYS)
            raise ValueError(
                f"{path} is not a current saved figure; envelope keys must be exactly "
                f"{sorted(_SAVED_FIGURE_KEYS)}, missing={missing}, extra={extra}.")
        schema_array = data["schema"]
        if schema_array.shape != () or schema_array.dtype.kind not in "US" \
                or str(schema_array.item()) != _SAVED_FIGURE_SCHEMA:
            raise ValueError(
                f"{path} has invalid saved-figure schema; expected {_SAVED_FIGURE_SCHEMA!r}.")

        def record(key: str, expected_type):
            array = data[key]
            if array.shape != () or array.dtype != object:
                raise TypeError(f"saved figure {key} must be a scalar object record.")
            value = array.item()
            if not isinstance(value, expected_type):
                expected_name = getattr(expected_type, "__name__", str(expected_type))
                raise TypeError(f"saved figure {key} must be {expected_name}, got {type(value).__name__}.")
            return value

        x = _strict_numeric_array(data["data_x"], "saved figure data_x").copy()
        y = _strict_numeric_array(data["data_y"], "saved figure data_y").copy()
        plot = _validate_plot_record(record("plot", Mapping))
        signals = _validate_signals(record("signals", Mapping))
        recipe_record = data["recipe"]
        if recipe_record.shape != () or recipe_record.dtype != object:
            raise TypeError("saved figure recipe must be a scalar object record.")
        recipe = _validate_recipe(recipe_record.item(), plot["kind"])
        provenance = _validate_provenance(record("provenance", Mapping))
        metadata = dict(record("metadata", Mapping))
        if any(not isinstance(key, str) for key in metadata):
            raise TypeError("saved figure metadata must be a string-keyed mapping.")
        collisions = set(metadata) & (_SAVED_PLOT_KEYS | {"signals", "figure_recipe", "provenance"})
        if collisions:
            raise ValueError(f"saved figure metadata duplicates reserved fields: {sorted(collisions)}.")
        if plot["kind"] not in _STRUCTURED_FIGURE_KINDS \
                and sum(entry["role"] == "value" for entry in signals.values()) != 1:
            raise ValueError(f"saved {plot['kind']!r} figure requires exactly one value signal.")

    info = {
        **metadata,
        **plot,
        "signals": signals,
        "figure_recipe": recipe,
        "provenance": provenance,
    }
    return x, y, info


def load_figure(path) -> SavedFigure:
    """Load a ``.npz`` saved by :meth:`DataFigure.save` into a :class:`SavedFigure`.

    The read-back counterpart of save: with no hardware or session, reopen an overnight scan /
    a saved panel to inspect what it holds (``.info_summary()``), list how it can be viewed
    (``.compatible_kinds()``) or re-render it (``.plot()`` / ``.plot(kind=...)``)."""
    path = Path(path)
    data_x, data_y, info = _load_saved_npz(path)
    return SavedFigure(data_x=data_x, data_y=data_y, info=info, path=path)


__all__ = ["DataFigure", "FitResult", "SavedFigure", "load_figure"]

