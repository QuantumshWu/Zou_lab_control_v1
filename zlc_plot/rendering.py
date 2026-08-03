"""Persistent Matplotlib artists for every public ZLC plot kind.

This module owns no GUI toolkit and never calls ``pyplot``.  A surface can be
attached to Agg, ipympl, or a Qt5 canvas.  Data/display edits mutate persistent
artists; fixed-size changes rebuild layout within the same Figure.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
import math
from numbers import Real
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ._image_raster import PreparedImageFront, prepare_image_front
from ._fit_scene import FitOverlay, FitPolyline
from ._rendering.pulse import update_pulse_timeline
from ._pulse_time import pulse_time_scale
from ._selector_scene import (
    ColorLimitCandidate,
    ColorLimitState,
    SceneKind,
    SelectorItemContext,
    SelectorLine,
    SelectorMarkers,
    SelectorPrimitive,
    SelectorScene,
    SelectorSceneOwner,
    SelectorSceneKind,
    SelectorSceneStyle,
    SelectorTarget,
)
from .layout import SurfacePlan, facet_focus_box
from .parameters import RenderEffect
from .primitives import ImagePointOverlay, PointStatus, PulseTimelineData
from .selectors import (
    CrosshairPoint,
    NumericRange,
    SelectorKind,
    SelectorSnapshot,
    SelectorState,
)
from .specs import (
    CurvePlot,
    FacetGridPlot,
    HistogramPlot,
    ImagePlot,
    PlotSpec,
    PulseTimelinePlot,
    RollingPlot,
)
from .state import DisplayState
from .style import PlotStyleConfig, style_context
from .ticks import apply_smart_ticks
from ._validation import readonly_copy


@dataclass(frozen=True, slots=True)
class _PreparedSeries:
    x: np.ndarray
    y: np.ndarray
    valid: np.ndarray
    label: str


@contextmanager
def _hidden_artists(artists: Sequence[Any]) -> Iterator[None]:
    visibility = tuple(bool(artist.get_visible()) for artist in artists)
    try:
        for artist in artists:
            artist.set_visible(False)
        yield
    finally:
        for artist, visible in zip(artists, visibility, strict=True):
            artist.set_visible(visible)


def _display_array(value: Any) -> np.ndarray:
    raw = getattr(value, "display", value)
    return np.asarray(raw)


def _valid_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    raw = getattr(value, "valid", None)
    if raw is None:
        return np.ones(shape, dtype=bool)
    return np.asarray(raw, dtype=bool)


def _unit_symbol(value: Any) -> str:
    unit = getattr(value, "display_unit", None)
    return "" if unit is None else str(getattr(unit, "symbol", unit))


_EXPLICIT_UNIT_SUFFIX = re.compile(r"(?:\[[^\[\]]+\]|\([^()]+\))\s*$")
def _quantity_label(value: Any, fallback: str, explicit: str | None = None) -> str:
    if explicit is not None:
        label = str(explicit)
        if not label:
            return ""
    else:
        label = str(getattr(value, "label", fallback) or fallback)
    unit = _unit_symbol(value)
    if not unit or unit == "1":
        return label
    if explicit is not None and _EXPLICIT_UNIT_SUFFIX.search(label):
        return label
    suffix = f"({unit})"
    return label if label.rstrip().endswith(suffix) else f"{label} {suffix}"


def _state_label(
    state: DisplayState,
    name: str,
    fallback: str | None,
) -> str | None:
    value = state.values.get(name)
    return fallback if value is None else str(value)


def _curve_x_limits(values: np.ndarray) -> tuple[float, float] | None:
    """Return the authored curve coordinate span without autoscale padding."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    low = float(np.min(finite))
    high = float(np.max(finite))
    if low == high:
        half_span = max(0.5, 0.05 * abs(low))
        return low - half_span, high + half_span
    return low, high


def _data_limits(values: np.ndarray) -> tuple[float, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.min(finite)), float(np.max(finite))


def _deadband_image_limits(
    data_range: tuple[float, float],
    current: tuple[float, float] | None,
    *,
    padding_fraction: float,
    deadband_fraction: float,
    mode: str = "tight",
    force: bool = False,
) -> tuple[float, float]:
    low, high = map(float, data_range)
    if mode == "normal" and low >= 0.0:
        target = (0.0, high * 1.2 if high else 1.0)
    else:
        data_span = (high - low) or (abs(high) or 1.0)
        padding = padding_fraction * data_span
        target = low - padding, high + padding
    if current is None or force:
        return target
    current_low, current_high = map(float, current)
    if mode == "normal" and low >= 0.0:
        if (
            current_low == 0.0
            and current_high > 0.0
            and 0.7 * current_high <= high <= current_high
        ):
            return current_low, current_high
        return target
    span = current_high - current_low
    if span <= 0.0:
        return target
    clips = low < current_low or high > current_high
    too_empty = (
        high < current_high - deadband_fraction * span
        or low > current_low + deadband_fraction * span
    )
    return target if clips or too_empty else (current_low, current_high)


def _select_display_limits(
    mode: str,
    automatic: tuple[float, float] | None,
    state: DisplayState,
    quantity: str,
    *,
    allow_partial: bool = False,
) -> tuple[float, float]:
    low_name, high_name = f"{quantity}_min", f"{quantity}_max"
    # Ordinary x/y bounds are viewport state, not authored Plot parameters.
    # A kind without a dedicated range control therefore has no range keys;
    # its base limits remain the renderer's automatic limits.
    authored_low = state.values.get(low_name)
    authored_high = state.values.get(high_name)
    if mode == "fixed" and (authored_low is None or authored_high is None):
        raise RuntimeError("committed fixed limits are incomplete")
    if mode != "fixed" and not allow_partial:
        return tuple(map(float, automatic or (0.0, 1.0)))
    base = (0.0, 1.0) if automatic is None else automatic
    low = base[0] if authored_low is None else float(authored_low)
    high = base[1] if authored_high is None else float(authored_high)
    if low >= high:
        if authored_low is not None and authored_high is not None:
            raise RuntimeError("committed display limits are not increasing")
        span = max(abs(low), abs(high), 1.0) * 1e-12
        low, high = (low, low + span) if authored_low is not None else (
            high - span,
            high,
        )
    return float(low), float(high)


def _image_data_range(
    values: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float] | None:
    """Return the exact finite range without promoting a megapixel image."""

    values = np.asarray(values)
    validity = np.broadcast_to(np.asarray(valid, dtype=np.bool_), values.shape)
    all_valid = bool(np.all(validity))
    if values.dtype.kind in "biu":
        if not all_valid and not bool(np.any(validity)):
            return None
        if all_valid:
            native_low = np.min(values)
            native_high = np.max(values)
        elif values.dtype.kind == "b":
            native_low = np.min(values, where=validity, initial=True)
            native_high = np.max(values, where=validity, initial=False)
        else:
            info = np.iinfo(values.dtype)
            native_low = np.min(values, where=validity, initial=info.max)
            native_high = np.max(values, where=validity, initial=info.min)
        low, high = float(native_low), float(native_high)
        if native_low != native_high and low == high:
            raise TypeError(
                "integer image range is not distinguishable in float display space"
            )
        return low, high

    # Keep temporary finite masks bounded.  This matters for float camera
    # frames whose validity is sparse, while the common all-valid path still
    # scans contiguous source chunks without selecting/copying the image.
    low = math.inf
    high = -math.inf
    chunk_size = 262_144
    operands = (values,) if all_valid else (values, validity)
    iterator = np.nditer(
        operands,
        flags=("external_loop", "buffered", "zerosize_ok"),
        op_flags=tuple(("readonly",) for _operand in operands),
        order="K",
        buffersize=chunk_size,
    )
    for chunks in iterator:
        if all_valid:
            chunk = np.asarray(chunks)
            valid_chunk = None
        else:
            value_chunk, mask_chunk = chunks
            chunk = np.asarray(value_chunk)
            valid_chunk = np.asarray(mask_chunk, dtype=np.bool_)
        finite = np.isfinite(chunk)
        if valid_chunk is not None:
            np.logical_and(finite, valid_chunk, out=finite)
        if not bool(np.any(finite)):
            continue
        low = min(low, float(np.min(chunk, where=finite, initial=np.inf)))
        high = max(high, float(np.max(chunk, where=finite, initial=-np.inf)))
    return (low, high) if math.isfinite(low) and math.isfinite(high) else None


def _image_arrays(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    z = np.asarray(z)
    if z.ndim != 2 or z.dtype.kind not in "biuf":
        raise TypeError("image values must be a real numeric two-dimensional array")
    if z.shape != (y.size, x.size):
        raise ValueError("image values must match the y/x coordinate grid")
    validity = (
        np.broadcast_to(np.asarray(True, dtype=np.bool_), z.shape)
        if valid is None
        else np.asarray(valid, dtype=np.bool_)
    )
    try:
        validity = np.broadcast_to(validity, z.shape)
    except ValueError as error:
        raise ValueError("image validity cannot broadcast to its values") from error
    x0, x1, y0, y1 = _centers_extent(x, y)
    return z, validity, (x0, x1, y1, y0)


def _bounded_image_distribution_values(
    values: np.ndarray,
    valid: np.ndarray,
    sample_target: int,
) -> np.ndarray:
    values = np.asarray(values)
    valid = np.broadcast_to(np.asarray(valid, dtype=bool), values.shape)
    order = "F" if values.flags.f_contiguous and not values.flags.c_contiguous else "C"
    flat_values = np.ravel(values, order=order)
    flat_valid = np.ravel(valid, order=order)
    if flat_values.size > sample_target:
        step = flat_values.size // sample_target + 1
        sampled = flat_values[::step]
        sampled_valid = flat_valid[::step]
        if sampled.dtype.kind == "f":
            sampled_valid = sampled_valid & np.isfinite(sampled)
        if bool(np.any(sampled_valid)):
            return np.asarray(sampled[sampled_valid], dtype=float)
    full_valid = flat_valid
    if flat_values.dtype.kind == "f":
        full_valid = full_valid & np.isfinite(flat_values)
    if bool(np.all(full_valid)):
        return np.asarray(flat_values, dtype=float)
    return np.asarray(flat_values[full_valid], dtype=float)


def _point_ring_radius(
    points: np.ndarray,
    *,
    fraction: float,
    fallback: float,
) -> float:
    points = np.asarray(points, dtype=float)
    if len(points) < 2 or not bool(np.all(np.isfinite(points))):
        return float(fallback)
    from scipy.spatial import cKDTree

    distances, _indices = cKDTree(points).query(points, k=2)
    nearest = np.asarray(distances[:, 1], dtype=float)
    nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    if nearest.size == 0:
        return float(fallback)
    return float(fraction) * float(np.median(nearest))


def _centers_extent(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    def edge(values: np.ndarray) -> tuple[float, float]:
        values = np.asarray(values, dtype=float).reshape(-1)
        if not values.size or not bool(np.all(np.isfinite(values))):
            raise ValueError("image coordinates must be non-empty and finite")
        if values.size == 1:
            return (float(values[0] - 0.5), float(values[0] + 0.5))
        steps = np.diff(values)
        if not (bool(np.all(steps > 0.0)) or bool(np.all(steps < 0.0))):
            raise ValueError("image coordinates must be strictly monotonic")
        return (
            float(values[0] - steps[0] / 2.0),
            float(values[-1] + steps[-1] / 2.0),
        )

    x0, x1 = edge(x)
    y0, y1 = edge(y)
    return (x0, x1, y0, y1)


def _square_image_limits(
    extent: tuple[float, float, float, float],
    *,
    coordinate_aspect: float = 1.0,
) -> tuple[float, float, float, float]:
    """Pad the shorter physical span without stretching image pixels."""

    left, right, upper, lower = map(float, extent)
    x_span = abs(right - left)
    y_span = abs(lower - upper) * float(coordinate_aspect)
    if (
        x_span <= 0.0
        or y_span <= 0.0
        or not math.isfinite(float(coordinate_aspect))
        or float(coordinate_aspect) <= 0.0
    ):
        raise ValueError("image extent spans must be positive")
    if x_span > y_span:
        padding = (x_span - y_span) / (2.0 * float(coordinate_aspect))
        direction = 1.0 if lower > upper else -1.0
        upper -= direction * padding
        lower += direction * padding
    elif y_span > x_span:
        padding = (y_span - x_span) / 2.0
        direction = 1.0 if right > left else -1.0
        left -= direction * padding
        right += direction * padding
    return left, right, upper, lower


def _image_coordinate_aspect(x: Any, y: Any) -> float:
    """Return screen y/x scale that preserves canonical physical distance."""

    x_unit = getattr(x, "display_unit", None)
    y_unit = getattr(y, "display_unit", None)
    if (
        x_unit is None
        or y_unit is None
        or not x_unit.compatible_with(y_unit)
    ):
        return 1.0
    return abs(float(y_unit.scale) / float(x_unit.scale))


def _histogram_vertices(edges: np.ndarray, counts: np.ndarray) -> np.ndarray:
    edges = np.asarray(edges, dtype=float).reshape(-1)
    counts = np.asarray(counts, dtype=float).reshape(-1)
    if edges.size != counts.size + 1:
        raise ValueError("histogram edges must contain one more value than counts")
    vertices = np.empty((counts.size, 4, 2), dtype=float)
    vertices[:, 0] = np.column_stack((edges[:-1], np.zeros(counts.size)))
    vertices[:, 1] = np.column_stack((edges[:-1], counts))
    vertices[:, 2] = np.column_stack((edges[1:], counts))
    vertices[:, 3] = np.column_stack((edges[1:], np.zeros(counts.size)))
    return vertices


def _compact_engineering(value: float, *, length: int | None = None) -> str:
    if not math.isfinite(float(value)):
        return "nan"
    numeric = float(value)

    def normalized(text: str) -> str:
        if "e" not in text.lower():
            return text
        mantissa, exponent = text.lower().split("e")
        return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"

    if length is None:
        general = f"{numeric:.4g}"
        return general if "e" not in general.lower() else normalized(f"{numeric:.1e}")
    maximum = max(1, int(length))
    for significant in range(4, 0, -1):
        general = normalized(f"{numeric:.{significant}g}")
        if len(general) <= maximum:
            return general
        scientific = normalized(f"{numeric:.{significant - 1}e}")
        if len(scientific) <= maximum:
            return scientific
    return normalized(f"{numeric:.0e}")


def _facet_cell_title(
    cell: Any,
    fallback: int,
    *,
    compact: bool,
) -> str:
    """Use a full focused title and a value-only small-multiple title.

    Repeating a long facet-axis label in every overview cell consumes the
    data box and eventually overlaps adjacent cells.  The overview therefore
    identifies each cell by its projected value (including any unit already
    carried by the label), while the focused cell restores the complete axis
    label.  This is one renderer policy for every FacetGrid consumer.
    """

    label = getattr(cell, "label", None)
    if label is None:
        label = str(getattr(cell, "facet_value_display", fallback))
    title = str(label)
    if compact and "=" in title:
        _axis_label, _separator, value_label = title.partition("=")
        title = value_label.strip() or title
    value = getattr(cell, "facet_value_display", None)
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric = float(value)
        if math.isfinite(numeric):
            raw = str(value)
            if raw in title:
                title = title.replace(raw, _compact_engineering(numeric), 1)
    return title


@dataclass(frozen=True, slots=True)
class RenderFrame:
    """Complete immutable input for one renderer presentation transaction."""

    payload: Any
    state: DisplayState
    effects: RenderEffect
    data_revision: int | None = None
    fit_overlay: FitOverlay | tuple[FitOverlay, ...] | None = None
    facet_thresholds: tuple[float | None, ...] = ()
    image_overlay: ImagePointOverlay | None = None
    selectors: SelectorSnapshot = SelectorSnapshot(())
    facet_index: int | None = None
    facet_focus_index: int | None = None
    view_limits: tuple[tuple[float, float], tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, DisplayState):
            raise TypeError("RenderFrame.state must be DisplayState")
        if not isinstance(self.effects, RenderEffect):
            raise TypeError("RenderFrame.effects must be RenderEffect")
        if not isinstance(self.selectors, SelectorSnapshot):
            raise TypeError("RenderFrame.selectors must be SelectorSnapshot")
        fit_overlay = self.fit_overlay
        if isinstance(fit_overlay, tuple):
            if any(not isinstance(item, FitOverlay) for item in fit_overlay):
                raise TypeError("RenderFrame fit_overlay tuple must contain FitOverlay")
        elif fit_overlay is not None and not isinstance(fit_overlay, FitOverlay):
            raise TypeError(
                "RenderFrame fit_overlay must be FitOverlay, a tuple, or None"
            )
        thresholds = tuple(self.facet_thresholds)
        for value in thresholds:
            if value is not None and not math.isfinite(float(value)):
                raise ValueError("facet thresholds must be finite or None")
        object.__setattr__(self, "facet_thresholds", thresholds)
        if self.image_overlay is not None and not isinstance(
            self.image_overlay, ImagePointOverlay
        ):
            raise TypeError("RenderFrame image_overlay must be ImagePointOverlay or None")
        if self.view_limits is not None:
            x_limits, y_limits = self.view_limits
            prepared = tuple(
                tuple(float(value) for value in limits)
                for limits in (x_limits, y_limits)
            )
            if any(len(limits) != 2 for limits in prepared):
                raise ValueError("RenderFrame view limits must contain two pairs")
            if not all(math.isfinite(value) for limits in prepared for value in limits):
                raise ValueError("RenderFrame view limits must be finite")
            if any(limits[0] == limits[1] for limits in prepared):
                raise ValueError("RenderFrame view spans must be non-degenerate")
            object.__setattr__(self, "view_limits", prepared)


class MatplotlibRenderer:
    """One fixed-layout Figure with persistent artists and selector overlays."""

    def __init__(
        self,
        spec: PlotSpec,
        plan: SurfacePlan,
        *,
        style: PlotStyleConfig,
    ) -> None:
        if not isinstance(plan, SurfacePlan):
            raise TypeError("plan must be SurfacePlan")
        if not isinstance(style, PlotStyleConfig):
            raise TypeError("style must be PlotStyleConfig")
        self.spec = spec
        self.plan = plan
        self.style = style
        self._figure: Any = None
        self._axes: dict[str, list[Any]] = {}
        self._artists: dict[str, Any] = {}
        self._selector_artists: dict[SceneKind, tuple[Any, ...]] = {}
        self._selector_topologies: dict[SceneKind, tuple[object, ...]] = {}
        self._selector_candidate: SelectorState | None = None
        self._color_limit_candidate: ColorLimitCandidate | None = None
        self._selector_scene_exclusion_kind: SceneKind | None = None
        self._last_selectors = SelectorSnapshot(())
        self._fit_artists: list[Any] = []
        self._fit_slots: dict[str, Any] = {}
        # A fit accepted while a selector gesture is using a cached blit
        # background changes the artists beneath that background.  Remember the
        # last complete fit front so the next presentation can recompose once
        # and recapture a coherent gesture background.
        self._last_fit_overlay: FitOverlay | None = None
        self._fit_source_scatter: Any | None = None
        self._fit_hidden_source_lines: tuple[tuple[Any, bool], ...] = ()
        self._fit_axis: Any | None = None
        self._fit_family: str | None = None
        self._fit_batch_identity: tuple[int, ...] | None = None
        self._facet_threshold_artists: dict[int, tuple[Any, Any]] = {}
        self._image_overlay: ImagePointOverlay | None = None
        self._image_overlay_signature: tuple[object, ...] | None = None
        self._figure_background: Any | None = None
        self._axis_backgrounds: dict[int, Any] = {}
        self._local_chrome_underlay: Any | None = None
        self._semantic_text_background: Any | None = None
        self._selector_gesture_kind: SceneKind | None = None
        self._selector_gesture_backgrounds: tuple[tuple[Any, Any], ...] = ()
        self._selector_gesture_limits: tuple[
            tuple[Any, tuple[float, float], tuple[float, float]], ...
        ] = ()
        self._data_revision: int | None = None
        self._last_payload: Any = None
        self._last_state: DisplayState | None = None
        self._home_limits: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {}
        self._requested_view_limits: tuple[
            tuple[float, float], tuple[float, float]
        ] | None = None
        self._chrome_dirty_axes: set[Any] = set()
        self._focused_facet_index: int | None = None
        self._facet_focus_index: int | None = None
        self._visible_facet_count = 0
        self._compose_figure()

    @property
    def figure(self) -> Any:
        return self._figure

    @property
    def axes(self) -> Mapping[str, tuple[Any, ...]]:
        return MappingProxyType({key: tuple(value) for key, value in self._axes.items()})

    @property
    def primary_axes(self) -> Any:
        if isinstance(self.spec, FacetGridPlot):
            values = self._axes.get("facet_cell", [])
            if values:
                index = self._focused_facet_index or 0
                return values[min(index, len(values) - 1)]
        for role in ("main", "history", "image", "facet_cell"):
            values = self._axes.get(role)
            if values:
                return values[0]
        return next(iter(self._axes.values()))[0]

    def facet_index_for_axes(self, axes: Any) -> int | None:
        if not isinstance(self.spec, FacetGridPlot):
            return None
        for index, candidate in enumerate(self._axes.get("facet_cell", ())):
            if axes is candidate and index < self._visible_facet_count:
                return index
        return None

    def interactive_axes_at(self, event: Any) -> Any | None:
        """Resolve pointer ownership from visible geometry, not backend hints."""

        candidate = getattr(event, "inaxes", None)
        x = getattr(event, "x", None)
        y = getattr(event, "y", None)
        visible = tuple(axis for axis in self._figure.axes if axis.get_visible())
        if x is None or y is None:
            return candidate if candidate in visible else None
        point = (float(x), float(y))
        if candidate in visible and bool(candidate.bbox.contains(*point)):
            return candidate
        # Focused Facet axes overlap the hidden overview geometry exactly.
        # Resolve against current visibility and put the semantic primary axes
        # first so a stale ipympl ``inaxes`` value cannot steal the event.
        ordered = (self.primary_axes, *reversed(visible))
        for axis in dict.fromkeys(ordered):
            if axis in visible and bool(axis.bbox.contains(*point)):
                return axis
        return None

    def _consume_facet_presentation(
        self,
        *,
        facet_index: int | None,
        facet_focus_index: int | None,
        visible_count: int | None = None,
    ) -> None:
        """Mirror the session-owned facet state for the current render pass."""

        for value, name in (
            (facet_index, "facet_index"),
            (facet_focus_index, "facet_focus_index"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{name} must be an integer or None")
        if not isinstance(self.spec, FacetGridPlot):
            if facet_index is not None or facet_focus_index is not None:
                raise TypeError("facet presentation state requires FacetGridPlot")
            self._focused_facet_index = None
            self._facet_focus_index = None
            self._visible_facet_count = 0
            return
        count = (
            len(self._axes.get("facet_cell", ()))
            if visible_count is None
            else visible_count
        )
        if count < 0 or count > len(self._axes.get("facet_cell", ())):
            raise ValueError("visible facet count is outside the rendered grid")
        if facet_index is None:
            if facet_focus_index is not None:
                raise ValueError("an open facet requires a selected facet")
            if visible_count is not None and count:
                raise ValueError("a non-empty FacetGrid requires a selected facet")
        else:
            if facet_index < 0 or facet_index >= count:
                raise IndexError("facet index is outside the current grid")
            if facet_focus_index is not None and facet_focus_index != facet_index:
                raise ValueError("the open facet must equal the selected facet")
        self._focused_facet_index = facet_index
        self._facet_focus_index = facet_focus_index
        self._visible_facet_count = count

    def _compose_figure(self) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        with style_context(
            self.style,
            {
                "figure.dpi": self.plan.logical_dpi,
                "figure.figsize": self.plan.figure_size_inches,
            },
        ):
            figure = Figure(
                figsize=self.plan.figure_size_inches,
                dpi=self.plan.logical_dpi,
                layout=None,
            )
            FigureCanvasAgg(figure)
            # Matplotlib native canvases derive physical DPI from this logical
            # baseline.  Materialise the Agg front at the requested screen DPR
            # without allowing a later ipympl/Qt canvas to multiply it again.
            figure._original_dpi = self.plan.logical_dpi
            figure._set_dpi(self.plan.dpi, forward=False)
            self._figure = figure
            self._axes = self._create_axes(figure, self.plan)

    @staticmethod
    def _create_axes(figure: Any, plan: SurfacePlan) -> dict[str, list[Any]]:
        axes: dict[str, list[Any]] = {}
        for axes_plan in plan.axes:
            axis = figure.add_axes(axes_plan.box.matplotlib_bounds())
            axis.set_gid(
                f"{axes_plan.role}:{axes_plan.cell_index}"
                if axes_plan.cell_index is not None
                else axes_plan.role
            )
            axes.setdefault(axes_plan.role, []).append(axis)
        return axes

    def relayout(
        self,
        plan: SurfacePlan,
        *,
        facet_index: int | None = None,
        facet_focus_index: int | None = None,
    ) -> None:
        """Rebuild a named layout while preserving the attached canvas model."""

        if not isinstance(plan, SurfacePlan):
            raise TypeError("plan must be SurfacePlan")
        figure = self._figure
        with style_context(
            self.style,
            {
                "figure.dpi": plan.logical_dpi,
                "figure.figsize": plan.figure_size_inches,
            },
        ):
            figure.clear()
            figure.set_size_inches(plan.figure_size_inches, forward=False)
            figure._original_dpi = plan.logical_dpi
            figure._set_dpi(plan.dpi, forward=False)
            self.plan = plan
            self._axes = self._create_axes(figure, plan)
        self._artists.clear()
        self._selector_artists.clear()
        self._selector_topologies.clear()
        self._selector_candidate = None
        self._color_limit_candidate = None
        self._selector_scene_exclusion_kind = None
        self._last_selectors = SelectorSnapshot(())
        self._fit_artists.clear()
        self._fit_slots.clear()
        self._last_fit_overlay = None
        self._fit_source_scatter = None
        self._fit_hidden_source_lines = ()
        self._fit_axis = None
        self._fit_family = None
        self._fit_batch_identity = None
        self._facet_threshold_artists.clear()
        self._image_overlay = None
        self._image_overlay_signature = None
        self._invalidate_layer_backgrounds()
        self._selector_gesture_kind = None
        self._selector_gesture_backgrounds = ()
        self._selector_gesture_limits = ()
        self._data_revision = None
        self._last_payload = None
        self._last_state = None
        self._home_limits.clear()
        self._requested_view_limits = None
        self._chrome_dirty_axes.clear()
        self._consume_facet_presentation(
            facet_index=facet_index,
            facet_focus_index=facet_focus_index,
        )

    def present(self, frame: RenderFrame) -> None:
        """Mutate all layers and publish exactly one complete canvas front."""

        if not isinstance(frame, RenderFrame):
            raise TypeError("frame must be RenderFrame")
        payload = frame.payload
        state = frame.state
        selected_effects = frame.effects
        previous_payload = self._last_payload
        previous_state = self._last_state
        previous_data_revision = self._data_revision
        previous_fit_overlay = self._last_fit_overlay
        state_changed = previous_state is None or state.revision != previous_state.revision
        payload_changed = (
            payload is not previous_payload
            or frame.data_revision != previous_data_revision
        )
        visible_facet_count = (
            len(tuple(getattr(payload, "cells", ())))
            if isinstance(self.spec, FacetGridPlot)
            else None
        )
        facet_presentation_unchanged = (
            frame.facet_index == self._focused_facet_index
            and frame.facet_focus_index == self._facet_focus_index
            and (
                visible_facet_count is None
                or visible_facet_count == self._visible_facet_count
            )
        )
        pure_text = (
            selected_effects == RenderEffect.TEXT
            and previous_state is not None
            and state_changed
            and not payload_changed
            and facet_presentation_unchanged
        )
        if pure_text:
            # TEXT is a complete transaction contract, not a hint.  Labels and
            # titles mutate in place without replaying view, data, selector,
            # fit, or point-overlay presentation work.
            with style_context(self.style):
                self._last_state = state
                self._update_text_artists(payload, state)
                if self._try_blit(semantic_text_only=True):
                    return
                self.draw()
            return
        full_draw_effects = (
            RenderEffect.AXIS_TRANSFORM
            | RenderEffect.CHROME
            | RenderEffect.LAYOUT
        )
        if selected_effects & full_draw_effects:
            # Axis visibility, bounds, and locators invalidate every cached
            # pixel transform.  Discard them before mutating the axes so no
            # nested overlay work can observe the previous presentation.
            self._invalidate_layer_backgrounds()
            self._selector_gesture_backgrounds = ()
        self._consume_facet_presentation(
            facet_index=frame.facet_index,
            facet_focus_index=frame.facet_focus_index,
            visible_count=visible_facet_count,
        )
        base_effects = (
            RenderEffect.VIEW_PROJECTION
            | RenderEffect.PAYLOAD_PROJECTION
            | RenderEffect.BASE_GEOMETRY
            | RenderEffect.LAYOUT
        )
        base_changed = payload_changed or bool(selected_effects & base_effects)
        if (
            state_changed
            and selected_effects & RenderEffect.AXIS_TRANSFORM
        ):
            base_changed = True
        semantic = self.spec.cell if isinstance(self.spec, FacetGridPlot) else self.spec
        if (
            selected_effects & RenderEffect.AXIS_TRANSFORM
            and isinstance(semantic, ImagePlot)
        ):
            # The display front is viewport-sized, so a view edit changes its
            # crop even though the immutable source payload is unchanged.
            base_changed = True
        style_only = (
            not base_changed
            and state_changed
            and bool(selected_effects & RenderEffect.BASE_STYLE)
        )
        overview = isinstance(self.spec, FacetGridPlot) and self._facet_focus_index is None
        painted_selectors = SelectorSnapshot(()) if overview else frame.selectors
        if isinstance(frame.fit_overlay, tuple):
            painted_fit = (
                frame.fit_overlay
                if overview
                else tuple(
                    overlay
                    for overlay in frame.fit_overlay
                    if overlay.facet_index == self._focused_facet_index
                )
            )
        else:
            painted_fit = None if overview else frame.fit_overlay
        fit_changed = painted_fit is not previous_fit_overlay
        with style_context(self.style):
            self._requested_view_limits = frame.view_limits
            self._last_payload = payload
            self._last_state = state
            self._data_revision = frame.data_revision
            self._last_fit_overlay = painted_fit
            if base_changed:
                self._update_plot(payload, state)
                self._capture_home_limits(self.primary_axes)
            elif style_only:
                self._update_base_style(state)
            if state_changed and selected_effects & RenderEffect.TEXT:
                self._update_text_artists(payload, state)
            if state_changed and selected_effects & RenderEffect.CHROME:
                self._update_chrome_artists(state)
            self._apply_requested_view(self.primary_axes, frame.view_limits)
            self._restore_selector_gesture_limits()
            self._last_selectors = painted_selectors
            self._update_selectors(self._last_selectors)
            self._set_fit_mode(isinstance(painted_fit, FitOverlay))
            self._update_fit(painted_fit)
            self._update_facet_thresholds(frame.facet_thresholds)
            self._update_image_point_overlay(payload, frame.image_overlay, state)
            # A native selector gesture blits against a cached scene.  If a
            # newly accepted fit changes that scene, recompose once and let the
            # next pointer frame capture the new base; never expose stale fit
            # pixels or a partially erased data layer.
            if fit_changed and self._selector_gesture_kind is not None:
                self._invalidate_layer_backgrounds()
                self._selector_gesture_backgrounds = ()
                self.draw()
                return
            partial_axes = (
                self._image_layer_axes()
                if not selected_effects & (full_draw_effects | RenderEffect.TEXT)
                else ()
            )
            if partial_axes and self._chrome_dirty_axes:
                local_chrome_axes = self._local_chrome_axes()
                local_axes = set(local_chrome_axes)
                if (
                    self._chrome_dirty_axes
                    and self._chrome_dirty_axes <= local_axes
                    and self._refresh_local_chrome_backgrounds()
                ):
                    partial_axes = tuple(dict.fromkeys(
                        (*partial_axes, *local_chrome_axes)
                    ))
                    self._chrome_dirty_axes.clear()
            if not selected_effects & full_draw_effects and not self._chrome_dirty_axes:
                painted = (
                    self._try_blit(axes=partial_axes)
                    if partial_axes
                    else self._try_blit()
                )
                if painted:
                    self._refresh_selector_gesture_background()
                    return
            self.draw()

    def _capture_home_limits(self, axis: Any) -> None:
        semantic = (
            self.spec.cell if isinstance(self.spec, FacetGridPlot) else self.spec
        )
        if isinstance(semantic, ImagePlot) and id(axis) in self._home_limits:
            # Image mutation registers the full source extent before applying
            # the frame viewport.  Never replace it with a zoomed front.
            return
        self._home_limits[id(axis)] = (
            tuple(map(float, axis.get_xlim())),
            tuple(map(float, axis.get_ylim())),
        )

    def _apply_requested_view(
        self,
        axis: Any,
        requested: tuple[tuple[float, float], tuple[float, float]] | None,
    ) -> None:
        if requested is None:
            requested = self._home_limits.get(id(axis))
        if requested is None:
            return
        self._set_xlim(axis, *requested[0])
        self._set_ylim(axis, *requested[1])

    def _update_base_style(self, state: DisplayState) -> None:
        """Apply property-only edits without rebuilding data geometry."""

        # Plot-specific mapping edits are still in-place artist mutations; the
        # plot updater owns their one canonical implementation.
        self._update_plot(self._last_payload, state)
        self._capture_home_limits(self.primary_axes)

    def _set_xlim(self, axis: Any, low: float, high: float) -> None:
        previous = np.asarray(axis.get_xlim(), dtype=float)
        wanted = np.asarray((low, high), dtype=float)
        if not np.allclose(previous, wanted, rtol=1e-12, atol=1e-15):
            axis.set_xlim(float(low), float(high))
            self._mark_axes_chrome_dirty(axis)

    def _set_ylim(self, axis: Any, low: float, high: float) -> None:
        previous = np.asarray(axis.get_ylim(), dtype=float)
        wanted = np.asarray((low, high), dtype=float)
        if not np.allclose(previous, wanted, rtol=1e-12, atol=1e-15):
            axis.set_ylim(float(low), float(high))
            self._mark_axes_chrome_dirty(axis)

    def _mark_axes_chrome_dirty(self, *axes: Any) -> None:
        self._chrome_dirty_axes.update(axes)

    def _invalidate_layer_backgrounds(self) -> None:
        self._figure_background = None
        self._axis_backgrounds.clear()
        self._local_chrome_underlay = None
        self._semantic_text_background = None

    def _capture_layer_backgrounds(self, canvas: Any) -> None:
        self._figure_background = canvas.copy_from_bbox(self._figure.bbox)
        self._axis_backgrounds = {
            id(axis): canvas.copy_from_bbox(self._pixel_aligned_bbox(axis.bbox))
            for axis in self._figure.axes
            if axis.get_visible()
        }

    def _image_layer_axes(self) -> tuple[Any, ...]:
        semantic = self.spec.cell if isinstance(self.spec, FacetGridPlot) else self.spec
        if not isinstance(semantic, ImagePlot):
            return ()
        roles = (
            ("facet_cell",)
            if isinstance(self.spec, FacetGridPlot)
            else ("image", "main", "distribution")
        )
        return tuple(dict.fromkeys(
            axis
            for role in roles
            for axis in self._axes.get(role, ())
            if axis.get_visible()
        ))

    def _local_chrome_axes(self) -> tuple[Any, ...]:
        return tuple(
            axis
            for role in ("distribution", "colorbar")
            for axis in self._axes.get(role, ())
            if axis.get_visible()
        )

    def draw(self) -> None:
        with style_context(self.style):
            canvas = self._figure.canvas
            supports_blit = bool(getattr(canvas, "supports_blit", False))
            self._set_dynamic_animated(supports_blit)
            self._invalidate_layer_backgrounds()
            semantic_text = self._semantic_text_artists()
            local_chrome_axes = self._local_chrome_axes() if supports_blit else ()
            if supports_blit:
                for axis in local_chrome_axes:
                    axis.set_animated(True)
                try:
                    with _hidden_artists(semantic_text):
                        self._native_draw(canvas)
                        self._local_chrome_underlay = canvas.copy_from_bbox(
                            self._figure.bbox
                        )
                        renderer = canvas.get_renderer()
                        for axis in local_chrome_axes:
                            axis.set_animated(False)
                            axis.draw(renderer)
                        self._capture_layer_backgrounds(canvas)
                finally:
                    for axis in local_chrome_axes:
                        axis.set_animated(False)
            else:
                self._native_draw(canvas)
            if supports_blit:
                if self._figure_background is not None and self._try_blit():
                    self._chrome_dirty_axes.clear()
                    self._refresh_selector_gesture_background()
                    return
                self._invalidate_layer_backgrounds()
                self._set_dynamic_animated(False)
                self._native_draw(canvas)
            self._chrome_dirty_axes.clear()

    @staticmethod
    def _native_draw(canvas: Any) -> None:
        """Draw through the raw canvas channel owned by an optional host.

        Notebook backends route their public ``draw`` method back through the
        session so a browser never receives a low-level partial frame.  The
        renderer's own composition must bypass that callback exactly once.
        """

        raw_draw = getattr(canvas, "_zlc_plot_raw_draw", None)
        draw = raw_draw if callable(raw_draw) else getattr(canvas, "draw", None)
        if not callable(draw):
            raise RuntimeError("the native canvas has no callable draw method")
        draw()

    @contextmanager
    def raster_transaction(self) -> Iterator[None]:
        """Expose at most one complete raster through an attached host."""

        canvas = self._figure.canvas
        begin = getattr(canvas, "_zlc_plot_begin_raster_transaction", None)
        end = getattr(canvas, "_zlc_plot_end_raster_transaction", None)
        if not callable(begin) or not callable(end):
            yield
            return
        begin()
        try:
            yield
        except BaseException:
            end(False)
            raise
        else:
            end(True)

    def _pixel_aligned_bbox(self, bounds: Any) -> Any:
        from matplotlib.transforms import Bbox

        x0, y0, x1, y1 = map(float, bounds.extents)
        aligned = Bbox.from_extents(
            math.floor(x0),
            math.floor(y0),
            math.ceil(x1),
            math.ceil(y1),
        )
        clipped = Bbox.intersection(aligned, self._figure.bbox)
        return (aligned if clipped is None else clipped).frozen()

    def _refresh_local_chrome_backgrounds(self) -> bool:
        """Rebuild disjoint distribution/colorbar chrome in place."""

        canvas = self._figure.canvas
        underlay = self._local_chrome_underlay
        if (
            underlay is None
            or not bool(getattr(canvas, "supports_blit", False))
        ):
            return False
        local_axes = self._local_chrome_axes()
        if not local_axes:
            return False
        semantic_text = self._semantic_text_artists()
        renderer = canvas.get_renderer()
        try:
            self._set_dynamic_animated(True)
            with _hidden_artists(semantic_text):
                canvas.restore_region(underlay)
                for axis in local_axes:
                    axis.draw(renderer)
                self._capture_layer_backgrounds(canvas)
        except Exception:
            self._invalidate_layer_backgrounds()
            return False
        self._draw_semantic_text_artists(semantic_text)
        self._semantic_text_background = None
        return True

    def _semantic_text_artists(self) -> tuple[Any, ...]:
        result = [
            artist
            for axis in self._figure.axes
            for artist in (axis.title, axis.xaxis.label, axis.yaxis.label)
        ]
        result.extend(
            self._artists[key]
            for key in ("figure:title", "facet:outer_x", "facet:outer_y")
            if key in self._artists
        )
        return tuple(dict.fromkeys(result))

    def _dynamic_artists(self) -> tuple[Any, ...]:
        from matplotlib.artist import Artist
        from matplotlib.colorbar import Colorbar

        result: list[Any] = []

        def collect(value: Any) -> None:
            if isinstance(value, Artist):
                if (
                    getattr(value, "axes", None) is not None
                    or getattr(value, "figure", None) is self._figure
                ):
                    result.append(value)
            elif isinstance(value, Colorbar):
                # A colorbar is an opaque axis-owned static composite.  Its
                # mappable callback may replace children, so cache/repaint the
                # entire owning axis when mapping changes rather than scanning
                # a transient child topology every image frame.
                return
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        for value in self._artists.values():
            collect(value)
        for values in self._selector_artists.values():
            collect(values)
        for values in self._facet_threshold_artists.values():
            collect(values)
        collect(self._fit_artists)
        collect(self._fit_source_scatter)
        # Axis-owned semantic text is mutated in place by the TEXT lane.  Keep
        # it out of the cached figure background just like data and overlays;
        # otherwise a label edit blits the new glyphs over the old ones.  Tick
        # labels intentionally remain static because locator/tick mutations are
        # AXIS_TRANSFORM or CHROME effects and require a complete draw.
        for artist in self._semantic_text_artists():
            collect(artist)
        return tuple(dict.fromkeys(result))

    def _set_dynamic_animated(self, animated: bool) -> None:
        for artist in self._dynamic_artists():
            artist.set_animated(animated)

    def _draw_semantic_text_artists(self, artists: Sequence[Any]) -> None:
        for artist in sorted(artists, key=lambda item: float(item.get_zorder())):
            if not artist.get_visible():
                continue
            axes = getattr(artist, "axes", None)
            if axes is not None:
                if not axes.get_visible():
                    continue
                axes.draw_artist(artist)
            else:
                self._figure.draw_artist(artist)

    def _try_blit(
        self,
        *,
        axes: Sequence[Any] | None = None,
        semantic_text_only: bool = False,
        cache_semantic_background: bool = True,
    ) -> bool:
        canvas = self._figure.canvas
        if not bool(getattr(canvas, "supports_blit", False)):
            return False
        semantic_text = self._semantic_text_artists()
        try:
            if semantic_text_only:
                background = self._semantic_text_background
                if background is None:
                    return False
                for artist in semantic_text:
                    artist.set_animated(True)
                canvas.restore_region(background)
                self._draw_semantic_text_artists(semantic_text)
                canvas.blit(self._figure.bbox)
                return True
            self._set_dynamic_animated(True)
            if axes is not None:
                from matplotlib.transforms import Bbox

                selected = tuple(dict.fromkeys(
                    axis for axis in axes if axis.get_visible()
                ))
                if not selected or any(
                    id(axis) not in self._axis_backgrounds for axis in selected
                ):
                    return False
                semantic_set = set(semantic_text)
                painted_bounds = []
                for axis in selected:
                    canvas.restore_region(self._axis_backgrounds[id(axis)])
                    for artist in sorted(
                        axis.get_children(),
                        key=lambda item: float(item.get_zorder()),
                    ):
                        if (
                            artist not in semantic_set
                            and artist.get_visible()
                            and artist.get_animated()
                        ):
                            axis.draw_artist(artist)
                    painted_bounds.append(self._pixel_aligned_bbox(axis.bbox))
                canvas.blit(Bbox.union(painted_bounds))
                self._semantic_text_background = None
                return True
            background = self._figure_background
            if background is None:
                return False
            canvas.restore_region(background)
            with _hidden_artists(semantic_text):
                # A dynamic Axis can paint its label as a child operation.
                # Hide all semantic text while composing the reusable data and
                # overlay front, then add current text exactly once.
                for artist in semantic_text:
                    artist.set_visible(False)
                for axes in self._figure.axes:
                    if not axes.get_visible():
                        continue
                    for artist in sorted(
                        axes.get_children(),
                        key=lambda item: float(item.get_zorder()),
                    ):
                        if artist.get_visible() and artist.get_animated():
                            axes.draw_artist(artist)
                figure_artists = tuple(
                    artist
                    for artist in self._dynamic_artists()
                    if getattr(artist, "axes", None) is None
                    and getattr(artist, "figure", None) is self._figure
                )
                for artist in sorted(
                    figure_artists,
                    key=lambda item: float(item.get_zorder()),
                ):
                    if artist.get_visible() and artist.get_animated():
                        self._figure.draw_artist(artist)
                if cache_semantic_background:
                    self._semantic_text_background = canvas.copy_from_bbox(
                        self._figure.bbox
                    )
            self._draw_semantic_text_artists(semantic_text)
            canvas.blit(self._figure.bbox)
            return True
        except Exception:
            if cache_semantic_background:
                self._semantic_text_background = None
            return False

    def begin_selector_gesture(self, kind: SelectorKind) -> bool:
        """Cache the painted plot beneath one selector for motion-only blits."""

        if not isinstance(kind, SelectorKind):
            raise TypeError("kind must be SelectorKind")
        self._selector_gesture_kind = kind
        self._selector_candidate = None
        with style_context(self.style):
            self._update_selectors(self._last_selectors)
        self._capture_selector_gesture_limits()
        return self._refresh_selector_gesture_background()

    def begin_color_limit_gesture(self, candidate: ColorLimitCandidate) -> bool:
        """Cache one color-scale front without entering the data-selector model."""

        if not isinstance(candidate, ColorLimitCandidate):
            raise TypeError("candidate must be ColorLimitCandidate")
        self._selector_gesture_kind = SelectorSceneKind.COLOR_LIMITS
        self._color_limit_candidate = candidate
        with style_context(self.style):
            self._update_selectors(self._last_selectors)
        self._capture_selector_gesture_limits()
        return self._refresh_selector_gesture_background()

    def begin_external_selector_gesture(
        self,
        kind: SceneKind,
    ) -> None:
        """Exclude one selector kind for an external raster compositor.

        The pointer transform is press-time input state; selector geometry and
        labels are rebuilt from each current session snapshot.  No scene owner
        is frozen here, so a live data/fit front cannot become stale during a
        drag.
        """

        if not isinstance(kind, (SelectorKind, SelectorSceneKind)):
            raise TypeError("kind must be a selector scene kind")
        self._selector_scene_exclusion_kind = kind

    def preview_selector(self, state: SelectorState) -> bool:
        """Paint one in-progress selector without redrawing data artists."""

        if not isinstance(state, SelectorState):
            raise TypeError("state must be SelectorState")
        if state.kind is not self._selector_gesture_kind:
            return False
        self._selector_candidate = state
        with style_context(self.style):
            self._update_selectors(self._last_selectors)
        if self._blit_selector_gesture():
            return True
        draw_idle = getattr(self._figure.canvas, "draw_idle", None)
        if callable(draw_idle):
            draw_idle()
            return True
        return False

    def preview_color_limit_candidate(self, candidate: ColorLimitCandidate) -> bool:
        """Paint one color-scale drag candidate without mutating image data."""

        if not isinstance(candidate, ColorLimitCandidate):
            raise TypeError("candidate must be ColorLimitCandidate")
        if self._selector_gesture_kind is not SelectorSceneKind.COLOR_LIMITS:
            return False
        self._color_limit_candidate = candidate
        with style_context(self.style):
            self._update_selectors(self._last_selectors)
        if self._blit_selector_gesture():
            return True
        draw_idle = getattr(self._figure.canvas, "draw_idle", None)
        if callable(draw_idle):
            draw_idle()
            return True
        return False

    def end_selector_gesture(self) -> None:
        was_color = self._selector_gesture_kind is SelectorSceneKind.COLOR_LIMITS
        self._selector_gesture_kind = None
        self._selector_candidate = None
        try:
            setattr(self._figure.canvas, "_zlc_plot_force_full_next", True)
        except (AttributeError, TypeError):
            pass
        if was_color:
            self._color_limit_candidate = None
        self._selector_gesture_backgrounds = ()
        self._selector_gesture_limits = ()

    def _capture_selector_gesture_limits(self) -> None:
        """Freeze native gesture coordinates while live artists keep updating."""

        kind = self._selector_gesture_kind
        artists = () if kind is None else self._selector_artists.get(kind, ())
        axes = tuple(
            dict.fromkeys(
                artist.axes
                for artist in artists
                if getattr(artist, "axes", None) is not None
            )
        )
        self._selector_gesture_limits = tuple(
            (
                axis,
                tuple(map(float, axis.get_xlim())),
                tuple(map(float, axis.get_ylim())),
            )
            for axis in axes
        )

    def _restore_selector_gesture_limits(self) -> None:
        for axis, x_limits, y_limits in self._selector_gesture_limits:
            if axis not in self._figure.axes or not axis.get_visible():
                continue
            self._set_xlim(axis, *x_limits)
            self._set_ylim(axis, *y_limits)

    def _refresh_selector_gesture_background(self) -> bool:
        kind = self._selector_gesture_kind
        if kind is None:
            return False
        self._selector_gesture_backgrounds = ()
        artists = self._selector_artists.get(kind, ())
        axes = tuple(
            dict.fromkeys(
                artist.axes
                for artist in artists
                if getattr(artist, "axes", None) is not None
            )
        )
        canvas = self._figure.canvas
        if (
            not axes
            or not bool(getattr(canvas, "supports_blit", False))
            or any(id(axis) not in self._axis_backgrounds for axis in axes)
        ):
            return False
        visible = tuple(bool(artist.get_visible()) for artist in artists)
        try:
            for artist in artists:
                artist.set_visible(False)
            if not self._try_blit(
                axes=axes,
                cache_semantic_background=False,
            ):
                return False
            backgrounds = tuple(
                (axis, canvas.copy_from_bbox(axis.bbox)) for axis in axes
            )
        except Exception:
            return False
        finally:
            for artist, was_visible in zip(artists, visible, strict=True):
                artist.set_visible(was_visible)
        self._selector_gesture_backgrounds = backgrounds
        return self._blit_selector_gesture()

    def _blit_selector_gesture(self) -> bool:
        kind = self._selector_gesture_kind
        backgrounds = self._selector_gesture_backgrounds
        if kind is None or not backgrounds:
            return False
        canvas = self._figure.canvas
        try:
            for _axis, background in backgrounds:
                canvas.restore_region(background)
            for axis, _background in backgrounds:
                for artist in sorted(
                    self._selector_artists.get(kind, ()),
                    key=lambda item: float(item.get_zorder()),
                ):
                    if artist.get_visible() and getattr(artist, "axes", None) is axis:
                        axis.draw_artist(artist)
            from matplotlib.transforms import Bbox

            canvas.blit(Bbox.union([axis.bbox for axis, _ in backgrounds]))
            # The cached semantic-text base would contain an earlier selector
            # position.  A concurrent TEXT transaction must rebuild instead of
            # restoring stale interaction pixels.
            self._semantic_text_background = None
            return True
        except Exception:
            self._selector_gesture_backgrounds = ()
            return False

    def _update_plot(self, payload: Any, state: DisplayState) -> None:
        if isinstance(self.spec, CurvePlot):
            self._update_curve(self.primary_axes, payload, state, "curve")
        elif isinstance(self.spec, HistogramPlot):
            self._update_histogram(self.primary_axes, payload, state, "histogram")
        elif isinstance(self.spec, ImagePlot):
            self._update_image(payload, state, "image")
        elif isinstance(self.spec, RollingPlot):
            self._update_rolling(payload, state)
        elif isinstance(self.spec, FacetGridPlot):
            self._update_facets(payload, state)
        elif isinstance(self.spec, PulseTimelinePlot):
            self._update_pulse_timeline(payload, state)
        else:
            raise TypeError(f"unsupported plot specification {type(self.spec).__name__}")
        self._apply_common_axes(state)

    def _effective_labels(
        self,
        payload: Any,
        state: DisplayState,
    ) -> tuple[str, str, str | None]:
        semantic = self.spec.cell if isinstance(self.spec, FacetGridPlot) else self.spec
        if isinstance(self.spec, FacetGridPlot):
            cells = tuple(getattr(payload, "cells", ()))
            if cells:
                index = self._focused_facet_index or 0
                index = min(max(index, 0), len(cells) - 1)
                payload = getattr(cells[index], "payload", cells[index])
        labels = getattr(semantic, "labels", None)
        if isinstance(semantic, CurvePlot):
            source = self._series(payload)
            explicit_x = _state_label(
                state,
                "x_label",
                labels.x if labels and labels.x else None,
            )
            explicit_y = _state_label(state, "y_label", None)
            if explicit_y is None:
                explicit_y = _state_label(
                    state,
                    "value_label",
                    None if labels is None else labels.y or labels.value,
                )
            x_label = (
                _quantity_label(source[0].x, "x", explicit_x)
                if source
                else ("x" if explicit_x is None else explicit_x)
            )
            y_label = (
                _quantity_label(source[0].y, "value", explicit_y)
                if source
                else ("value" if explicit_y is None else explicit_y)
            )
            return x_label, y_label, None
        if isinstance(semantic, HistogramPlot):
            explicit_x = _state_label(state, "x_label", None)
            if explicit_x is None:
                explicit_x = _state_label(
                    state,
                    "value_label",
                    None if labels is None else labels.x or labels.value,
                )
            quantity = getattr(payload, "edges", None)
            if quantity is None:
                quantity = getattr(payload, "values", payload)
            y_label = _state_label(
                state,
                "y_label",
                labels.y if labels and labels.y else None,
            )
            if y_label is None:
                y_label = "density" if bool(state["density"]) else "Shots"
            return _quantity_label(quantity, "value", explicit_x), y_label, None
        if isinstance(semantic, ImagePlot):
            explicit_x = _state_label(
                state,
                "x_label",
                labels.x if labels and labels.x else None,
            )
            explicit_y = _state_label(
                state,
                "y_label",
                labels.y if labels and labels.y else None,
            )
            explicit_value = _state_label(
                state,
                "value_label",
                labels.value if labels and labels.value else None,
            )
            value_label = (
                explicit_value
                if explicit_value and _EXPLICIT_UNIT_SUFFIX.search(explicit_value)
                else _quantity_label(payload.z, "value", explicit_value)
            )
            return (
                _quantity_label(payload.x, "x", explicit_x),
                _quantity_label(payload.y, "y", explicit_y),
                value_label,
            )
        if isinstance(semantic, RollingPlot):
            source = self._series(payload)
            explicit_x = _state_label(state, "x_label", labels.x or "Shots ago")
            explicit_y = _state_label(state, "y_label", None)
            if explicit_y is None:
                explicit_y = _state_label(
                    state,
                    "value_label",
                    labels.y or labels.value,
                )
            y_label = (
                _quantity_label(source[0].y, "value", explicit_y)
                if source
                else ("value" if explicit_y is None else explicit_y)
            )
            return "Shots ago" if explicit_x is None else explicit_x, y_label, None
        if isinstance(semantic, PulseTimelinePlot):
            _factor, unit = pulse_time_scale(payload, state["x_display_unit"])
            x_label = _state_label(state, "x_label", f"Time ({unit})")
            if x_label and not _EXPLICIT_UNIT_SUFFIX.search(x_label):
                x_label = f"{x_label} ({unit})"
            y_label = _state_label(state, "y_label", "")
            return x_label or "", y_label or "", None
        raise TypeError(f"unsupported plot specification {type(semantic).__name__}")

    def _update_text_artists(self, payload: Any, state: DisplayState) -> None:
        """Update semantic labels and titles without touching data artists."""

        x_label, y_label, value_label = self._effective_labels(payload, state)
        if isinstance(self.spec, FacetGridPlot):
            for name, value in (("x", x_label), ("y", y_label)):
                artist = self._artists.get(f"facet:outer_{name}")
                if artist is not None:
                    artist.set_text(value)
            if self._facet_focus_index is not None:
                axis = self._axes["facet_cell"][self._facet_focus_index]
                axis.set_xlabel(x_label, fontsize=self.style.fonts.axis_label_pt)
                axis.set_ylabel(y_label, fontsize=self.style.fonts.axis_label_pt)
        else:
            if isinstance(self.spec, ImagePlot):
                axis = self._axes.get("image", [self.primary_axes])[0]
            elif isinstance(self.spec, RollingPlot):
                axis = self._axes.get("history", [self.primary_axes])[0]
            else:
                axis = self.primary_axes
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
        if value_label is not None:
            for key, value in self._artists.items():
                if key.endswith(":colorbar") and hasattr(value, "set_label"):
                    value.set_label(value_label)
        self._update_title_artist(state)

    def _update_chrome_artists(self, state: DisplayState) -> None:
        """Apply visibility/grid chrome without revisiting semantic text."""

        self._apply_colorbar_visibility(state)
        self._apply_grid(state)

    def _apply_colorbar_visibility(self, state: DisplayState) -> None:
        if "show_colorbar" not in state.values:
            return
        visible = bool(state["show_colorbar"])
        for axis in self._axes.get("colorbar", ()):
            if bool(axis.get_visible()) != visible:
                axis.set_visible(visible)
                self._mark_axes_chrome_dirty(axis)

    def _series(self, payload: Any) -> tuple[Any, ...]:
        values = getattr(payload, "series", None)
        if values is None:
            raise TypeError("curve payload must expose a series tuple")
        return tuple(values)

    @staticmethod
    def _prepare_curve_series(source: Sequence[Any]) -> tuple[_PreparedSeries, ...]:
        prepared = []
        for item in source:
            x = np.asarray(_display_array(item.x), dtype=float).reshape(-1)
            y = np.asarray(_display_array(item.y), dtype=float).reshape(-1)
            valid = _valid_array(item, x.shape) & np.isfinite(x) & np.isfinite(y)
            label = getattr(item, "label", None)
            if label is None:
                group_key = getattr(item, "group_key", ())
                label = ", ".join(str(value) for value in group_key) if group_key else ""
            prepared.append(_PreparedSeries(x, y, valid, str(label)))
        return tuple(prepared)

    def _ensure_lines(self, axes: Any, count: int, key: str) -> list[Any]:
        lines: list[Any] = self._artists.setdefault(key, [])
        while len(lines) < count:
            index = len(lines)
            (line,) = axes.plot(
                [],
                [],
                color=self.style.palette.line_color(index),
                linewidth=self.style.artists.curve.linewidth,
                alpha=self.style.artists.curve.alpha,
                linestyle=self.style.artists.curve.linestyle,
                marker=self.style.artists.curve.marker,
            )
            lines.append(line)
        for index, line in enumerate(lines):
            line.set_visible(index < count)
        return lines

    def _update_curve(
        self,
        axes: Any,
        payload: Any,
        state: DisplayState,
        key: str,
        *,
        limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
        prepared_series: tuple[_PreparedSeries, ...] | None = None,
    ) -> None:
        source = self._series(payload)
        series = (
            self._prepare_curve_series(source)
            if prepared_series is None
            else prepared_series
        )
        semantic = self.spec.cell if isinstance(self.spec, FacetGridPlot) else self.spec
        labels = getattr(semantic, "labels", None)
        explicit_x = _state_label(
            state,
            "x_label",
            labels.x if labels and labels.x else None,
        )
        explicit_y = _state_label(state, "y_label", None)
        if explicit_y is None:
            explicit_y = _state_label(
                state,
                "value_label",
                None if labels is None else labels.y or labels.value,
            )
        x_label = (
            _quantity_label(source[0].x, "x", explicit_x)
            if source
            else ("x" if explicit_x is None else explicit_x)
        )
        y_label = (
            _quantity_label(source[0].y, "value", explicit_y)
            if source
            else ("value" if explicit_y is None else explicit_y)
        )
        self._mutate_series_artists(
            axes,
            series,
            state,
            key,
            x_label=x_label,
            y_label=y_label,
            limits=limits,
        )

    def _mutate_series_artists(
        self,
        axes: Any,
        series: tuple[_PreparedSeries, ...],
        state: DisplayState,
        key: str,
        *,
        x_label: str,
        y_label: str,
        limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        lines = self._ensure_lines(axes, len(series), key)
        curve_style = self.style.artists.curve
        width = curve_style.linewidth
        marker_size = self.style.artists.curve_marker_size_pt
        marker = "None" if curve_style.marker is None else curve_style.marker
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        for index, item in enumerate(series):
            # NaNs preserve invalid runs as gaps instead of joining neighbours.
            plotted_x = np.where(item.valid, item.x, np.nan)
            plotted_y = np.where(item.valid, item.y, np.nan)
            lines[index].set_data(plotted_x, plotted_y)
            lines[index].set_linewidth(width)
            lines[index].set_alpha(self.style.artists.curve.alpha)
            lines[index].set_markersize(marker_size)
            lines[index].set_marker(marker)
            lines[index].set_label(item.label)
            all_x.append(item.x[item.valid])
            all_y.append(item.y[item.valid])
        if limits is not None:
            self._set_xlim(axes, *limits[0])
            self._set_ylim(axes, *limits[1])
        else:
            usable_x = tuple(value for value in all_x if value.size)
            usable_y = tuple(value for value in all_y if value.size)
            xlim = (
                _curve_x_limits(np.concatenate(usable_x))
                if usable_x
                else None
            )
            y_range = (
                _data_limits(np.concatenate(usable_y))
                if usable_y
                else None
            )
            if xlim is not None:
                self._set_xlim(axes, *xlim)
            self._set_ylim(
                axes,
                *self._resolve_curve_y_limits(key, y_range, state),
            )
        if axes.get_legend() is not None:
            axes.get_legend().remove()
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        apply_smart_ticks(axes)

    def _histogram_arrays(
        self, payload: Any, state: DisplayState
    ) -> tuple[np.ndarray, np.ndarray]:
        edges = getattr(payload, "edges", None)
        counts = getattr(payload, "counts", None)
        if edges is not None and counts is not None:
            edge_values = np.asarray(_display_array(edges), dtype=float).reshape(-1)
            count_values = np.asarray(counts, dtype=float).reshape(-1)
        else:
            values = getattr(payload, "values", payload)
            values = np.asarray(_display_array(values), dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            count_values, edge_values = np.histogram(
                values,
                bins=int(state["bin_count"]),
            )
            count_values = count_values.astype(float)
        density = bool(state["density"])
        cumulative = bool(state["cumulative"])
        if cumulative:
            count_values = np.cumsum(count_values)
            if density and count_values.size and count_values[-1] > 0.0:
                count_values = count_values / count_values[-1]
        elif density:
            total = float(np.sum(count_values))
            widths = np.diff(edge_values)
            if total > 0:
                count_values = count_values / (total * widths)
        return edge_values, count_values

    def _update_histogram(
        self,
        axes: Any,
        payload: Any,
        state: DisplayState,
        key: str,
        *,
        arrays: tuple[np.ndarray, np.ndarray] | None = None,
        limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> None:
        from matplotlib.collections import PolyCollection

        edges, counts = self._histogram_arrays(payload, state) if arrays is None else arrays
        collection = self._artists.get(key)
        alpha = self.style.artists.histogram_fill_alpha
        if collection is None:
            collection = PolyCollection(
                _histogram_vertices(edges, counts),
                facecolors=self.style.palette.hist_fill,
                edgecolors="none",
                alpha=alpha,
            )
            axes.add_collection(collection)
            self._artists[key] = collection
        else:
            collection.set_verts(_histogram_vertices(edges, counts))
            collection.set_alpha(alpha)
        self._artists[f"{key}:projection"] = (edges, counts)
        if limits is not None:
            selected_x = limits[0]
            selected_y = limits[1]
        else:
            selected_x = (
                (float(edges[0]), float(edges[-1]))
                if edges.size >= 2
                else None
            )
            selected_y = self._resolve_histogram_y_limits(key, counts, state)

        scale = "log" if bool(state["log_y"]) else "linear"
        if scale == "log" and selected_y[0] <= 0.0:
            raise RuntimeError("resolved logarithmic limits are not positive")
        if axes.get_yscale() != scale:
            if scale == "log":
                old_low, _old_high = sorted(map(float, axes.get_ylim()))
                if old_low <= 0.0:
                    axes.set_ylim(
                        float(selected_y[0]),
                        max(float(selected_y[1]), float(selected_y[0]) * 10.0),
                    )
            axes.set_yscale(scale)
            self._mark_axes_chrome_dirty(axes)
        if limits is not None:
            self._set_xlim(axes, *selected_x)
            self._set_ylim(axes, *selected_y)
        else:
            if selected_x is not None:
                self._set_xlim(axes, *selected_x)
            self._set_ylim(axes, *selected_y)
        semantic = self.spec.cell if isinstance(self.spec, FacetGridPlot) else self.spec
        labels = getattr(semantic, "labels", None)
        explicit_x = _state_label(state, "x_label", None)
        if explicit_x is None:
            explicit_x = _state_label(
                state,
                "value_label",
                None if labels is None else labels.x or labels.value,
            )
        quantity = getattr(payload, "edges", None)
        if quantity is None:
            quantity = getattr(payload, "values", payload)
        axes.set_xlabel(_quantity_label(quantity, "value", explicit_x))
        y_label = _state_label(
            state,
            "y_label",
            labels.y if labels and labels.y else None,
        )
        if y_label is None:
            y_label = "density" if bool(state["density"]) else "Shots"
        axes.set_ylabel(y_label)
        apply_smart_ticks(axes)

    def _resolve_histogram_y_limits(
        self,
        key: str,
        counts: np.ndarray,
        state: DisplayState,
    ) -> tuple[float, float]:
        mode = str(state["relim_mode"])
        logarithmic = bool(state["log_y"])
        shrink_ratio = self.style.render.distribution_count_shrink_ratio
        state_key = f"{key}:histogram_limits"
        mode_key = f"{key}:histogram_mode"
        previous = self._artists.get(state_key)
        previous_mode = self._artists.get(mode_key)
        if mode == "fixed":
            automatic = None
        else:
            finite = np.asarray(counts, dtype=float).reshape(-1)
            finite = finite[np.isfinite(finite)]
            if logarithmic:
                positive = finite[finite > 0.0]
                if positive.size:
                    target = (
                        max(float(np.min(positive)) * 0.8, np.finfo(float).tiny),
                        max(float(np.max(positive)) * 1.2, 1.0),
                    )
                else:
                    target = (0.8, 1.2)
            else:
                peak = float(np.max(finite)) if finite.size else 0.0
                target = (0.0, max(1.0, peak * 1.08))
            force = (
                previous is None
                or previous_mode != (mode, logarithmic)
                or mode == "tight"
            )
            if force:
                automatic = target
            else:
                prior_low, prior_high = tuple(map(float, previous))
                expand = target[0] < prior_low or target[1] > prior_high
                shrink = (
                    target[1] < shrink_ratio * prior_high
                    or (
                        logarithmic
                        and target[0]
                        > prior_low / shrink_ratio
                    )
                )
                automatic = target if expand or shrink else (prior_low, prior_high)
        selected = _select_display_limits(mode, automatic, state, "y")
        if logarithmic and selected[0] <= 0.0:
            raise RuntimeError("resolved logarithmic limits are not positive")
        self._artists[state_key] = selected
        self._artists[mode_key] = (mode, logarithmic)
        return selected

    def _resolve_image_limits(
        self,
        key: str,
        data_range: tuple[float, float] | None,
        state: DisplayState,
    ) -> tuple[float, float]:
        return self._resolve_data_limits(
            key, data_range, state, "color", allow_partial=True
        )

    def _resolve_data_limits(
        self,
        key: str,
        data_range: tuple[float, float] | None,
        state: DisplayState,
        quantity: str,
        *,
        allow_partial: bool = False,
        cache_selected: bool = False,
    ) -> tuple[float, float]:
        limits_key = f"{key}:automatic_{quantity}_limits"
        mode_key = f"{key}:relim_mode"
        mode = str(state["relim_mode"])
        current = self._artists.get(limits_key)
        if mode == "fixed":
            automatic = None
        elif data_range is None:
            automatic = current if current is not None else (0.0, 1.0)
        else:
            policy = self.style.render
            automatic = _deadband_image_limits(
                data_range,
                current,
                padding_fraction=policy.image_color_padding_fraction,
                deadband_fraction=policy.image_color_deadband_fraction,
                mode=mode,
                force=self._artists.get(mode_key) != mode,
            )
        selected = _select_display_limits(
            mode, automatic, state, quantity, allow_partial=allow_partial
        )
        cached = selected if cache_selected else automatic
        if cached is not None:
            self._artists[limits_key] = cached
        self._artists[mode_key] = mode
        return selected

    def _resolve_distribution_limits(
        self,
        key: str,
        data_range: tuple[float, float] | None,
        color_limits: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Resolve histogram and rail domains without clipping the color handles."""

        limits_key = f"{key}:distribution_data_limits"
        current = self._artists.get(limits_key)
        if data_range is None:
            selected = current if current is not None else (0.0, 1.0)
        else:
            policy = self.style.render
            selected = _deadband_image_limits(
                data_range,
                current,
                padding_fraction=policy.image_color_padding_fraction,
                deadband_fraction=policy.image_color_deadband_fraction,
                mode="tight",
                force=current is None,
            )
        histogram_limits = tuple(map(float, selected))
        self._artists[limits_key] = histogram_limits
        rail_limits = (
            min(histogram_limits[0], float(color_limits[0])),
            max(histogram_limits[1], float(color_limits[1])),
        )
        return histogram_limits, rail_limits

    def _resolve_curve_y_limits(
        self,
        key: str,
        data_range: tuple[float, float] | None,
        state: DisplayState,
    ) -> tuple[float, float]:
        return self._resolve_data_limits(
            key, data_range, state, "y", cache_selected=True
        )

    def _update_image_artist(
        self,
        axes: Any,
        values: np.ndarray,
        valid: np.ndarray,
        extent: tuple[float, float, float, float],
        state: DisplayState,
        key: str,
        color_limits: tuple[float, float] | None,
        *,
        square_view: bool,
        coordinate_aspect: float,
    ) -> tuple[Any, Any]:
        import matplotlib

        policy = self.style.render
        cmap_name = str(state["colormap"])
        cmap = matplotlib.colormaps[cmap_name].copy()
        cmap.set_bad(self.style.palette.bad)
        interpolation = str(state["interpolation"])
        mapping_state = (cmap_name, interpolation, color_limits)
        mapping_key = f"{key}:mapping_state"
        previous_mapping = self._artists.get(mapping_key)

        home_extent = (
            _square_image_limits(
                extent,
                coordinate_aspect=coordinate_aspect,
            )
            if square_view
            else extent
        )
        self._home_limits[id(axes)] = (
            (float(home_extent[0]), float(home_extent[1])),
            (float(home_extent[2]), float(home_extent[3])),
        )
        requested = (
            self._requested_view_limits
            if axes is self.primary_axes
            else None
        )
        if requested is None:
            x_limits = (home_extent[0], home_extent[1])
            y_limits = (home_extent[2], home_extent[3])
        else:
            x_limits, y_limits = requested
        self._set_xlim(axes, *x_limits)
        self._set_ylim(axes, *y_limits)
        axes.set_anchor(policy.image_anchor)
        axes.set_aspect(coordinate_aspect, adjustable="box")
        # Equal aspect changes the actual drawable box.  Resolve that box
        # before choosing a source reduction so one prepared sample maps to at
        # roughly one physical output pixel at the current DPR.
        axes.apply_aspect()
        display_pixel_shape = (
            max(1, round(float(axes.bbox.width))),
            max(1, round(float(axes.bbox.height))),
        )
        front_key = (
            self._data_revision,
            id(values),
            values.shape,
            values.strides,
            values.dtype.str,
            tuple(map(float, extent)),
            tuple(map(float, x_limits)),
            tuple(map(float, y_limits)),
            display_pixel_shape,
            policy.image_front,
        )
        front_cache_key = f"{key}:prepared_front"
        cached_front = self._artists.get(front_cache_key)
        if cached_front is not None and cached_front[0] == front_key:
            prepared: PreparedImageFront = cached_front[1]
        else:
            prepared = prepare_image_front(
                values,
                valid,
                extent,
                x_limits=x_limits,
                y_limits=y_limits,
                display_pixel_shape=display_pixel_shape,
                policy=policy.image_front,
            )
            self._artists[front_cache_key] = (front_key, prepared)

        image = self._artists.get(key)
        if image is None:
            image = axes.imshow(
                prepared.values,
                origin=policy.image_origin,
                aspect=coordinate_aspect,
                extent=prepared.extent,
                interpolation=interpolation,
                interpolation_stage=prepared.interpolation_stage,
                cmap=cmap,
                vmin=None if color_limits is None else color_limits[0],
                vmax=None if color_limits is None else color_limits[1],
            )
            self._artists[key] = image
        else:
            image.set_data(prepared.values)
            image.set_extent(prepared.extent)
            if previous_mapping is None or previous_mapping[0] != cmap_name:
                image.set_cmap(cmap)
            if previous_mapping is None or previous_mapping[1] != interpolation:
                image.set_interpolation(interpolation)
            image.set_interpolation_stage(prepared.interpolation_stage)
            if (
                color_limits is not None
                and (
                    previous_mapping is None
                    or previous_mapping[2] != color_limits
                    or not np.allclose(
                        np.asarray(image.get_clim(), dtype=float),
                        np.asarray(color_limits, dtype=float),
                        rtol=1.0e-12,
                        atol=1.0e-15,
                    )
                )
            ):
                image.set_clim(*color_limits)
        self._artists[mapping_key] = mapping_state
        # ``imshow``/``set_extent`` may autoscale a new artist.  Reassert the
        # transaction's final transform after mutating the scalar front.
        self._set_xlim(axes, *x_limits)
        self._set_ylim(axes, *y_limits)
        return image, cmap

    def _update_horizontal_histogram(
        self,
        axes: Any,
        key: str,
        edges: np.ndarray,
        counts: np.ndarray,
        y_limits: tuple[float, float],
        *,
        projection_changed: bool,
        tick_profile: str,
    ) -> None:
        from matplotlib.collections import PolyCollection
        from matplotlib.ticker import MaxNLocator, ScalarFormatter

        policy = self.style.render
        collection = self._artists.get(key)
        if collection is None:
            collection = PolyCollection(
                _histogram_vertices(edges, counts)[..., ::-1],
                facecolors=self.style.palette.hist_fill,
                edgecolors="none",
                alpha=policy.side_distribution_fill_alpha,
            )
            axes.add_collection(collection)
            self._artists[key] = collection
            axes.xaxis.set_major_locator(
                MaxNLocator(nbins=policy.distribution_axis_max_ticks, prune="both")
            )
            if tick_profile == "image":
                axes.xaxis.set_major_formatter(ScalarFormatter())
                axes.tick_params(
                    axis="y", left=True, right=False, labelleft=False, labelright=False
                )
            else:
                axes.set_xlabel("")
                axes.set_ylabel("")
                axes.tick_params(
                    axis="y",
                    which="both",
                    left=False,
                    right=False,
                    labelleft=False,
                )
                axes.tick_params(
                    axis="both",
                    which="both",
                    bottom=False,
                    top=False,
                )
        elif projection_changed:
            collection.set_verts(_histogram_vertices(edges, counts)[..., ::-1])
        peak = float(np.max(counts)) if counts.size else 0.0
        wanted = float(
            max(
                policy.distribution_count_floor,
                int(
                    max(
                        peak + policy.distribution_count_headroom,
                        peak * policy.distribution_count_growth,
                    )
                ),
            )
        )
        ceiling_key = f"{key}:count_ceiling"
        ceiling = float(self._artists.get(ceiling_key, 0.0))
        if (
            ceiling <= 0.0
            or wanted > ceiling
            or wanted < policy.distribution_count_shrink_ratio * ceiling
        ):
            ceiling = wanted
            self._artists[ceiling_key] = ceiling
        self._set_xlim(axes, 0.0, ceiling)
        self._set_ylim(axes, *y_limits)

    def _update_image(
        self,
        payload: Any,
        state: DisplayState,
        key: str,
    ) -> None:
        labels = getattr(self.spec, "labels", None)
        explicit_x = _state_label(
            state,
            "x_label",
            labels.x if labels and labels.x else None,
        )
        explicit_y = _state_label(
            state,
            "y_label",
            labels.y if labels and labels.y else None,
        )
        explicit_value = _state_label(
            state,
            "value_label",
            labels.value if labels and labels.value else None,
        )
        self._mutate_image_artists(
            np.asarray(_display_array(payload.x)),
            np.asarray(_display_array(payload.y)),
            np.asarray(_display_array(payload.z)),
            getattr(payload, "valid", None),
            state,
            key,
            x_label=_quantity_label(payload.x, "x", explicit_x),
            y_label=_quantity_label(payload.y, "y", explicit_y),
            value_label=(
                explicit_value
                if explicit_value and _EXPLICIT_UNIT_SUFFIX.search(explicit_value)
                else _quantity_label(payload.z, "value", explicit_value)
            ),
            coordinate_aspect=_image_coordinate_aspect(payload.x, payload.y),
        )

    def _mutate_image_artists(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        valid: np.ndarray | None,
        state: DisplayState,
        key: str,
        *,
        x_label: str,
        y_label: str,
        value_label: str,
        coordinate_aspect: float,
    ) -> None:
        policy = self.style.render
        axes = self._axes["image"][0] if "image" in self._axes else self.primary_axes
        z, valid, extent = _image_arrays(x, y, z, valid)
        data_range = _image_data_range(z, valid)
        vmin, vmax = self._resolve_image_limits(key, data_range, state)
        if self._color_limit_candidate is not None:
            vmin = self._color_limit_candidate.value.low
            vmax = self._color_limit_candidate.value.high
        histogram_limits, distribution_limits = self._resolve_distribution_limits(
            key,
            data_range,
            (vmin, vmax),
        )
        image, cmap = self._update_image_artist(
            axes,
            z,
            valid,
            extent,
            state,
            key,
            (vmin, vmax),
            square_view=True,
            coordinate_aspect=coordinate_aspect,
        )
        cmap_name = str(state["colormap"])
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)

        distribution_axes = self._axes.get("distribution", [])
        if distribution_axes:
            distribution = distribution_axes[0]
            bin_count = max(
                policy.image_distribution_min_bins,
                min(
                    max(int(z.size), 1) // policy.distribution_bin_divisor,
                    policy.distribution_max_bins,
                ),
            )
            cache_key = (
                self._data_revision,
                bin_count,
                histogram_limits,
            )
            cache_name = f"{key}:distribution_cache"
            cached = self._artists.get(cache_name)
            if cached is not None and cached[0] == cache_key:
                counts, edges = cached[1], cached[2]
                distribution_changed = False
            else:
                samples = _bounded_image_distribution_values(
                    z,
                    valid,
                    policy.image_distribution_sample_target,
                )
                if not samples.size:
                    samples = np.asarray([histogram_limits[0]], dtype=float)
                counts, edges = np.histogram(
                    samples,
                    bins=bin_count,
                    range=histogram_limits,
                )
                self._artists[cache_name] = (cache_key, counts, edges)
                distribution_changed = True

            distribution_key = f"{key}:distribution"
            self._update_horizontal_histogram(
                distribution,
                distribution_key,
                edges,
                counts,
                distribution_limits,
                projection_changed=distribution_changed,
                tick_profile="image",
            )

            guide_key = f"{key}:guides"
            guides = self._artists.get(guide_key)
            if guides is None:
                guides = tuple(
                    distribution.axhline(
                        value,
                        color=self.style.palette.guide,
                        linewidth=self.style.artists.side_distribution_linewidth,
                        alpha=policy.distribution_guide_alpha,
                    )
                    for value in (vmin, vmax)
                )
                self._artists[guide_key] = guides
            show_guides = data_range is not None
            for index, guide in enumerate(guides):
                guide.set_visible(show_guides)
                if show_guides:
                    assert data_range is not None
                    value = data_range[index]
                    guide.set_ydata((value, value))

        colorbar_axes = self._axes.get("colorbar", [])
        if colorbar_axes:
            colorbar_key = f"{key}:colorbar"
            colorbar = self._artists.get(colorbar_key)
            if colorbar is None:
                colorbar = self._figure.colorbar(image, cax=colorbar_axes[0])
                self._artists[colorbar_key] = colorbar
            colorbar_state = (
                cmap_name,
                (vmin, vmax),
                value_label,
            )
            state_key = f"{key}:colorbar_state"
            previous_colorbar_state = self._artists.get(state_key)
            if colorbar_state != previous_colorbar_state:
                colorbar.set_label(
                    value_label,
                    labelpad=policy.colorbar_endpoint_label_pad_pt,
                )
                colorbar.set_ticks((vmin, vmax))
                label_chars = policy.colorbar_endpoint_label_chars
                colorbar.set_ticklabels(
                    (
                        _compact_engineering(vmin, length=label_chars),
                        _compact_engineering(vmax, length=label_chars),
                    )
                )
                self._artists[state_key] = colorbar_state
                self._mark_axes_chrome_dirty(colorbar_axes[0])
        self._apply_colorbar_visibility(state)
        apply_smart_ticks(
            axes,
            max_ticks_x=policy.image_spatial_max_ticks,
            max_ticks_y=policy.image_spatial_max_ticks,
        )

    def _update_rolling(self, payload: Any, state: DisplayState) -> None:
        series = self._series(payload)
        sliced: list[_PreparedSeries] = []
        for item in series:
            y_values = np.asarray(_display_array(item.y), dtype=float).reshape(-1)
            plot_x = getattr(item, "plot_x", None)
            if plot_x is None:
                raise ValueError("rolling payload requires projected plot_x")
            x_values = np.asarray(plot_x, dtype=float).reshape(-1)
            valid = (
                _valid_array(
                    item,
                    _display_array(item.x).reshape(-1).shape,
                )
                & np.isfinite(x_values)
                & np.isfinite(y_values)
            )
            label = getattr(item, "label", "")
            if label is None:
                label = ""
            sliced.append(_PreparedSeries(x_values, y_values, valid, str(label)))

        labels = self.spec.labels
        explicit_y = _state_label(state, "y_label", None)
        if explicit_y is None:
            explicit_y = _state_label(
                state,
                "value_label",
                labels.y or labels.value,
            )
        explicit_x = _state_label(state, "x_label", labels.x or "Shots ago")
        y_label = (
            _quantity_label(series[0].y, "value", explicit_y)
            if series
            else ("value" if explicit_y is None else explicit_y)
        )
        history = self._axes.get("history", [self.primary_axes])[0]
        self._mutate_series_artists(
            history,
            tuple(sliced),
            state,
            "rolling:history",
            x_label="Shots ago" if explicit_x is None else explicit_x,
            y_label=y_label,
        )
        latest = None
        if sliced:
            usable = sliced[0].y[sliced[0].valid]
            if usable.size:
                latest = float(usable[-1])
        latest_text = self._artists.get("rolling:latest")
        if latest_text is None:
            latest_text = history.text(
                0.97,
                0.95,
                "",
                transform=history.transAxes,
                color=self.style.palette.readout,
                ha="right",
                va="top",
                fontsize=self.style.fonts.annotation_pt,
            )
            self._artists["rolling:latest"] = latest_text
        latest_text.set_text("" if latest is None else f"{latest:.6g}")

        distribution_axes = self._axes.get("distribution", [])
        if distribution_axes:
            policy = self.style.render
            samples = [item.y[item.valid] for item in sliced]
            values = np.concatenate(samples) if samples else np.asarray([], dtype=float)
            history_length = max((item.size for item in samples), default=0)
            requested_bins = int(state["bin_count"])
            bin_count = max(
                policy.rolling_distribution_min_bins,
                min(
                    requested_bins,
                    max(
                        history_length // policy.distribution_bin_divisor,
                        policy.rolling_distribution_min_bins,
                    ),
                    policy.distribution_max_bins,
                ),
            )
            y_limits = tuple(float(value) for value in history.get_ylim())
            counts, edges = np.histogram(
                values[np.isfinite(values)] if values.size else np.asarray([y_limits[0]]),
                bins=bin_count,
                range=tuple(sorted(y_limits)),
            )
            side = distribution_axes[0]
            self._update_horizontal_histogram(
                side,
                "rolling:distribution",
                edges,
                counts,
                y_limits,
                projection_changed=True,
                tick_profile="rolling",
            )

    def _position_facet_axes_for_frame(self, axes: Sequence[Any]) -> None:
        """Apply final cell boxes before artists resolve pixel-dependent work."""

        if self._facet_focus_index is None:
            for index, axis in enumerate(axes):
                axis.set_position(self.plan.axes[index].box.matplotlib_bounds())
                axis.set_visible(index < self._visible_facet_count)
            return
        selected_index = self._facet_focus_index
        bounds = facet_focus_box(self.plan).matplotlib_bounds()
        for index, axis in enumerate(axes):
            axis.set_visible(index == selected_index)
        axes[selected_index].set_position(bounds)

    def _update_facets(self, payload: Any, state: DisplayState) -> None:
        from matplotlib.ticker import FixedLocator, MaxNLocator, ScalarFormatter

        cells = tuple(getattr(payload, "cells", ()))
        axes = self._axes.get("facet_cell", [])
        if self._visible_facet_count != len(cells):
            raise RuntimeError("FacetGrid payload and rendered topology are inconsistent")
        curve_series: tuple[tuple[_PreparedSeries, ...], ...] = ()
        curve_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        histogram_arrays: tuple[tuple[np.ndarray, np.ndarray], ...] = ()
        histogram_limits: tuple[tuple[float, float], tuple[float, float]] | None = None
        image_arrays: tuple[Any, ...] = ()
        image_limits: tuple[float, float] | None = None
        focused = self._facet_focus_index is not None
        # Facet focus changes the physical cell box.  Establish the complete
        # frame geometry before an image cell chooses its display raster.
        self._position_facet_axes_for_frame(axes)

        if isinstance(self.spec.cell, CurvePlot) and not focused:
            curve_series = tuple(
                self._prepare_curve_series(
                    self._series(getattr(cell, "payload", cell))
                )
                for cell in cells
            )
            x_groups: list[np.ndarray] = []
            y_groups: list[np.ndarray] = []
            for series in curve_series:
                for item in series:
                    if bool(np.any(item.valid)):
                        x_groups.append(item.x[item.valid])
                        y_groups.append(item.y[item.valid])
            if x_groups:
                x_target = _curve_x_limits(np.concatenate(x_groups))
                y_range = _data_limits(np.concatenate(y_groups))
                assert x_target is not None
                curve_limits = (
                    x_target,
                    self._resolve_curve_y_limits(
                        "facet:curve",
                        y_range,
                        state,
                    ),
                )
        elif isinstance(self.spec.cell, HistogramPlot):
            histogram_arrays = tuple(
                self._histogram_arrays(getattr(cell, "payload", cell), state)
                if not focused or index == self._facet_focus_index
                else (np.asarray([], dtype=float), np.asarray([], dtype=float))
                for index, cell in enumerate(cells)
            )
            usable = (
                tuple(item for item in histogram_arrays if item[0].size >= 2)
                if not focused
                else ()
            )
            if usable:
                shared_edges = usable[0][0]
                if any(
                    edges.shape != shared_edges.shape
                    or not np.array_equal(edges, shared_edges)
                    for edges, _counts in usable[1:]
                ):
                    raise ValueError("FacetGrid histogram cells must share one bin projection")
                x_limits = (
                    float(shared_edges[0]),
                    float(shared_edges[-1]),
                )
                pooled_counts = np.concatenate(
                    tuple(np.asarray(counts, dtype=float) for _edges, counts in usable)
                )
                histogram_limits = (
                    x_limits,
                    self._resolve_histogram_y_limits(
                        "facet:histogram",
                        pooled_counts,
                        state,
                    ),
                )
        elif isinstance(self.spec.cell, ImagePlot):
            prepared_images = []
            for index, cell in enumerate(cells):
                if focused and index != self._facet_focus_index:
                    prepared_images.append(None)
                    continue
                cell_payload = getattr(cell, "payload", cell)
                prepared_images.append(
                    _image_arrays(
                        np.asarray(_display_array(cell_payload.x)),
                        np.asarray(_display_array(cell_payload.y)),
                        np.asarray(_display_array(cell_payload.z)),
                        getattr(cell_payload, "valid", None),
                    )
                )
            image_arrays = tuple(prepared_images)
            if not focused:
                pooled_low: float | None = None
                pooled_high: float | None = None
                for prepared in image_arrays:
                    if prepared is None:
                        continue
                    values, valid, _extent = prepared
                    cell_range = _image_data_range(values, valid)
                    if cell_range is not None:
                        pooled_low = (
                            cell_range[0]
                            if pooled_low is None
                            else min(pooled_low, cell_range[0])
                        )
                        pooled_high = (
                            cell_range[1]
                            if pooled_high is None
                            else max(pooled_high, cell_range[1])
                        )
                pooled_range = (
                    None
                    if pooled_low is None or pooled_high is None
                    else (pooled_low, pooled_high)
                )
                image_limits = self._resolve_image_limits(
                    "facet:image", pooled_range, state
                )

        outer_x = ""
        outer_y = ""
        visible_axes: list[tuple[int, Any]] = []
        for index, axis in enumerate(axes):
            visible = index < len(cells) and (
                not focused or index == self._facet_focus_index
            )
            axis.set_visible(visible)
            if not visible:
                continue
            cell = cells[index]
            cell_payload = getattr(cell, "payload", cell)
            if isinstance(self.spec.cell, CurvePlot):
                self._update_curve(
                    axis,
                    cell_payload,
                    state,
                    f"facet:{index}:curve",
                    limits=curve_limits,
                    prepared_series=(
                        curve_series[index] if curve_series else None
                    ),
                )
                outer_x = outer_x or axis.get_xlabel()
                outer_y = outer_y or axis.get_ylabel()
            elif isinstance(self.spec.cell, HistogramPlot):
                arrays = histogram_arrays[index] if index < len(histogram_arrays) else None
                self._update_histogram(
                    axis,
                    cell_payload,
                    state,
                    f"facet:{index}:histogram",
                    arrays=arrays,
                    limits=histogram_limits,
                )
                outer_x = outer_x or axis.get_xlabel()
                outer_y = outer_y or axis.get_ylabel()
            elif isinstance(self.spec.cell, ImagePlot):
                prepared_image = image_arrays[index]
                if prepared_image is None:
                    raise RuntimeError("focused facet image was not prepared")
                values, valid, extent = prepared_image
                key = f"facet:{index}:image"
                cell_image_limits = (
                    self._resolve_image_limits(
                        key,
                        _image_data_range(values, valid),
                        state,
                    )
                    if focused
                    else image_limits
                )
                self._update_image_artist(
                    axis,
                    values,
                    valid,
                    extent,
                    state,
                    key,
                    cell_image_limits,
                    square_view=False,
                    coordinate_aspect=_image_coordinate_aspect(
                        cell_payload.x,
                        cell_payload.y,
                    ),
                )
                cell_labels = getattr(self.spec.cell, "labels", None)
                explicit_x = _state_label(
                    state,
                    "x_label",
                    cell_labels.x if cell_labels and cell_labels.x else None,
                )
                explicit_y = _state_label(
                    state,
                    "y_label",
                    cell_labels.y if cell_labels and cell_labels.y else None,
                )
                outer_x = outer_x or _quantity_label(
                    cell_payload.x,
                    "x",
                    explicit_x,
                )
                outer_y = outer_y or _quantity_label(
                    cell_payload.y,
                    "y",
                    explicit_y,
                )
            axis.set_xlabel("")
            axis.set_ylabel("")
            visible_axes.append((index, axis))
        typography = self.plan.facet_typography
        rows, columns = self.plan.facet_shape or (1, max(len(cells), 1))
        for index, axis in visible_axes:
            cell = cells[index]
            label = _facet_cell_title(
                cell,
                index,
                compact=(self._facet_focus_index is None and len(cells) > 1),
            )
            axis.set_title(
                str(label),
                fontsize=(
                    typography.cell_title_pt
                    if typography is not None
                    else self.style.fonts.tick_pt
                ),
                pad=self.style.render.compact_axes_title_pad_pt,
            )
            axis.tick_params(
                axis="both",
                labelsize=(
                    typography.tick_pt
                    if typography is not None
                    else self.style.fonts.tick_pt
                ),
                length=2,
            )
            row, column = divmod(index, columns)
            if column:
                axis.set_yticks([])
            else:
                axis.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
                axis.yaxis.set_major_formatter(ScalarFormatter())
                axis.tick_params(axis="y", labelleft=True)
            if row < rows - 1 and index + columns < len(cells):
                axis.set_xticks([])
            else:
                low, high = axis.get_xlim()
                interior = tuple(
                    value
                    for value in MaxNLocator(nbins=4).tick_values(low, high)
                    if min(low, high) < value < max(low, high)
                )
                picked = (
                    (
                        min(
                            interior,
                            key=lambda value: abs(value - (low + high) / 2.0),
                        ),
                    )
                    if interior
                    else ()
                )
                axis.xaxis.set_major_locator(FixedLocator(picked))
                axis.xaxis.set_major_formatter(ScalarFormatter())
                axis.tick_params(axis="x", labelbottom=True)

        outer_labels = (("x", outer_x, 0.5, 0.012, 0.0), ("y", outer_y, 0.008, 0.5, 90.0))
        for name, value, x_pos, y_pos, rotation in outer_labels:
            artist_key = f"facet:outer_{name}"
            artist = self._artists.get(artist_key)
            if artist is None:
                artist = self._figure.text(
                    x_pos,
                    y_pos,
                    "",
                    ha="center" if name == "x" else "left",
                    va="bottom" if name == "x" else "center",
                    rotation=rotation,
                    fontsize=(
                        typography.outer_axis_label_pt
                        if typography is not None
                        else self.style.fonts.axis_label_pt
                    ),
                )
                self._artists[artist_key] = artist
            artist.set_text(value)
            artist.set_visible(self._facet_focus_index is None)

        if self._facet_focus_index is None:
            return

        selected_index = self._facet_focus_index
        selected = axes[selected_index]
        selected.set_xlabel(outer_x, fontsize=self.style.fonts.axis_label_pt)
        selected.set_ylabel(outer_y, fontsize=self.style.fonts.axis_label_pt)
        selected.set_title(
            selected.get_title(),
            fontsize=self.style.fonts.figure_title_pt,
            pad=self.style.render.compact_axes_title_pad_pt,
        )
        apply_smart_ticks(selected)
        selected.tick_params(
            axis="both",
            labelsize=self.style.fonts.tick_pt,
            labelbottom=True,
            labelleft=True,
        )

    def _update_image_point_overlay(
        self,
        payload: Any,
        overlay: ImagePointOverlay | None,
        state: DisplayState,
    ) -> None:
        """Mutate the optional Image point layer independently of its raster."""

        if not isinstance(self.spec, ImagePlot):
            return
        x_quantity = getattr(payload, "x", None)
        y_quantity = getattr(payload, "y", None)
        if x_quantity is None or y_quantity is None:
            raise TypeError("Image point overlays require image coordinate quantities")
        signature = (
            None if overlay is None else overlay.revision,
            None if overlay is None else id(overlay),
            bool(state["show_point_labels"]),
            str(getattr(x_quantity, "display_unit", "")),
            str(getattr(y_quantity, "display_unit", "")),
        )
        if signature == self._image_overlay_signature:
            return
        self._image_overlay = overlay
        self._image_overlay_signature = signature
        axis = self._axes.get("image", [self.primary_axes])[0]
        collection = self._artists.get("image:points")
        labels: list[Any] = self._artists.setdefault("image:point-labels", [])
        if overlay is None or overlay.count == 0:
            if collection is not None:
                collection.set_visible(False)
            for label in labels:
                label.set_visible(False)
            return

        from matplotlib.collections import EllipseCollection
        from matplotlib.colors import to_rgba

        canonical = np.asarray(overlay.coordinates, dtype=float)
        x_display = np.asarray(
            x_quantity.canonical_unit.convert_value_to(
                canonical[:, 0], x_quantity.display_unit
            ),
            dtype=float,
        )
        y_display = np.asarray(
            y_quantity.canonical_unit.convert_value_to(
                canonical[:, 1], y_quantity.display_unit
            ),
            dtype=float,
        )
        points = np.column_stack((x_display, y_display))
        x_domain = np.asarray(getattr(x_quantity, "canonical", ()), dtype=float)
        y_domain = np.asarray(getattr(y_quantity, "canonical", ()), dtype=float)
        spans = tuple(
            float(np.ptp(values[np.isfinite(values)]))
            for values in (x_domain, y_domain)
            if bool(np.any(np.isfinite(values)))
        )
        finite_spans = tuple(span for span in spans if span > 0.0)
        singleton_radius = (
            min(finite_spans) * self.style.artists.point_single_radius_fraction
            if finite_spans
            else self.style.artists.point_single_radius_fraction
        )
        canonical_radius = _point_ring_radius(
            canonical,
            fraction=self.style.artists.point_auto_radius_fraction,
            fallback=singleton_radius,
        )

        def display_radius(quantity: Any) -> float:
            converted = np.asarray(
                quantity.canonical_unit.convert_value_to(
                    (0.0, canonical_radius), quantity.display_unit
                ),
                dtype=float,
            )
            return abs(float(converted[1] - converted[0]))

        radius_x = display_radius(x_quantity)
        radius_y = display_radius(y_quantity)
        statuses = overlay.statuses or (PointStatus.UNKNOWN,) * overlay.count
        tokens = {
            PointStatus.UNKNOWN: self.style.artists.point_unknown,
            PointStatus.EMPTY: self.style.artists.point_empty,
            PointStatus.OCCUPIED: self.style.artists.point_occupied,
            PointStatus.INVALID: self.style.artists.point_invalid,
        }
        edgecolors = tuple(
            to_rgba(tokens[status].color, tokens[status].alpha)
            for status in statuses
        )
        linewidths = tuple(tokens[status].linewidth for status in statuses)
        linestyles = tuple(
            "--" if status is PointStatus.INVALID else "-" for status in statuses
        )
        if collection is None:
            collection = EllipseCollection(
                widths=np.full(overlay.count, 2.0 * radius_x),
                heights=np.full(overlay.count, 2.0 * radius_y),
                angles=np.zeros(overlay.count),
                units="xy",
                offsets=points,
                transOffset=axis.transData,
                facecolors="none",
                edgecolors=edgecolors,
                linewidths=linewidths,
                linestyles=linestyles,
                zorder=self.style.artists.point_zorder,
                clip_on=True,
            )
            axis.add_collection(collection)
            self._artists["image:points"] = collection
        else:
            collection.set_offsets(points)
            collection.set_widths(np.full(overlay.count, 2.0 * radius_x))
            collection.set_heights(np.full(overlay.count, 2.0 * radius_y))
            collection.set_angles(np.zeros(overlay.count))
            collection.set_edgecolors(edgecolors)
            collection.set_linewidths(linewidths)
            collection.set_linestyles(linestyles)
            collection.set_visible(True)

        while len(labels) < overlay.count:
            labels.append(
                axis.text(
                    0.0,
                    0.0,
                    "",
                    ha="center",
                    va="bottom",
                    fontsize=self.style.fonts.annotation_pt,
                    zorder=self.style.artists.point_label_zorder,
                    clip_on=True,
                )
            )
        show_labels = bool(state["show_point_labels"])
        point_ids = overlay.point_ids
        point_labels = overlay.labels
        for index, label in enumerate(labels):
            visible = show_labels and index < overlay.count
            label.set_visible(visible)
            if not visible:
                continue
            label.set_position(tuple(points[index]))
            explicit = None if point_labels is None else point_labels[index]
            point_id = None if point_ids is None else point_ids[index]
            label.set_text(explicit or point_id or "")

    def _update_pulse_timeline(self, payload: Any, state: DisplayState) -> None:
        update_pulse_timeline(
            self.primary_axes,
            payload,
            state,
            self.style,
            self._artists,
        )
        x_label = _state_label(state, "x_label", None)
        if x_label is not None:
            if x_label:
                _factor, unit = pulse_time_scale(
                    payload,
                    state["x_display_unit"],
                )
                if not _EXPLICIT_UNIT_SUFFIX.search(x_label):
                    x_label = f"{x_label} ({unit})"
            self.primary_axes.set_xlabel(x_label)
        y_label = _state_label(state, "y_label", None)
        if y_label is not None:
            self.primary_axes.set_ylabel(y_label)
        signature = (
            payload.time_unit,
            state["x_display_unit"],
            bool(state["show_grid"]),
            tuple((channel.channel_id, channel.label) for channel in payload.channels),
            tuple((trace.name, trace.label) for trace in payload.analog_traces),
            tuple(self.primary_axes.get_xlim()),
            tuple(self.primary_axes.get_ylim()),
        )
        previous = self._artists.get("pulse:chrome_signature")
        self._artists["pulse:chrome_signature"] = signature
        if previous != signature:
            self._mark_axes_chrome_dirty(self.primary_axes)

    def _update_title_artist(self, state: DisplayState) -> None:
        title = str(state["title"])
        if isinstance(self.spec, FacetGridPlot):
            title_artist = self._artists.get("figure:title")
            if title_artist is None:
                title_artist = self._figure.text(
                    0.5,
                    self.style.render.figure_title_y,
                    "",
                    ha="center",
                    va="top",
                    fontsize=self.style.fonts.figure_title_pt,
                )
                self._artists["figure:title"] = title_artist
            title_artist.set_text(title)
            title_artist.set_visible(self._facet_focus_index is None and bool(title))
        else:
            owner = self.primary_axes
            owner.set_title(
                title,
                fontsize=self.style.fonts.figure_title_pt,
                pad=self.style.render.axes_title_pad_pt,
                y=1.0,
            )

    def _apply_grid(self, state: DisplayState) -> None:
        show_grid = bool(state["show_grid"])
        data_axes = (
            self._axes.get("facet_cell", [])
            if isinstance(self.spec, FacetGridPlot)
            else [self.primary_axes]
        )
        for axis in data_axes:
            if isinstance(self.spec, PulseTimelinePlot):
                if show_grid:
                    axis.grid(
                        True,
                        axis="x",
                        color=self.style.palette.pulse_grid,
                        linewidth=self.style.pulse.grid_linewidth,
                    )
                else:
                    axis.grid(False, axis="x")
            else:
                axis.grid(show_grid)

    def _apply_common_axes(self, state: DisplayState) -> None:
        self._update_title_artist(state)
        self._apply_grid(state)

    def _remove_artists(self, artists: Iterable[Any]) -> None:
        for artist in tuple(artists):
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                artist.set_visible(False)

    def _selector_axis(self, state: SelectorState) -> Any:
        if isinstance(self.spec, FacetGridPlot):
            axes = self._axes.get("facet_cell", [])
            index = self._focused_facet_index if state.facet_index is None else state.facet_index
            if index is None or index < 0 or index >= self._visible_facet_count:
                raise IndexError("selector facet index is outside the current grid")
            return axes[index]
        return self.primary_axes

    def _image_selector_artist(self, state: SelectorState) -> Any | None:
        if isinstance(self.spec, ImagePlot):
            return self._artists.get("image")
        if isinstance(self.spec, FacetGridPlot) and isinstance(self.spec.cell, ImagePlot):
            index = self._focused_facet_index if state.facet_index is None else state.facet_index
            return self._artists.get(f"facet:{index}:image")
        return None

    def _active_image_artist(self) -> Any | None:
        image = self._artists.get("image")
        if image is None and isinstance(self.spec, FacetGridPlot) and isinstance(
            self.spec.cell, ImagePlot
        ):
            index = 0 if self._focused_facet_index is None else self._focused_facet_index
            image = self._artists.get(f"facet:{index}:image")
        return image

    def resolved_color_limits(self) -> tuple[float, float]:
        """Return the clim painted by the active image artist."""

        image = self._active_image_artist()
        if image is None:
            raise TypeError("color limits require an active Image artist")
        return tuple(map(float, image.get_clim()))

    def _image_selector_payload(self, state: SelectorState) -> Any | None:
        if isinstance(self.spec, ImagePlot):
            return self._last_payload
        if isinstance(self.spec, FacetGridPlot) and isinstance(self.spec.cell, ImagePlot):
            cells = tuple(getattr(self._last_payload, "cells", ()))
            index = self._focused_facet_index if state.facet_index is None else state.facet_index
            if index is not None and 0 <= index < len(cells):
                return getattr(cells[index], "payload", cells[index])
        return None

    def _image_cross_value(self, state: SelectorState) -> float | str | None:
        if not isinstance(state.value, CrosshairPoint):
            return None
        payload = self._image_selector_payload(state)
        if payload is None:
            return None
        x = np.asarray(_display_array(payload.x), dtype=float).reshape(-1)
        y = np.asarray(_display_array(payload.y), dtype=float).reshape(-1)
        z = np.asarray(_display_array(payload.z))
        if not x.size or not y.size or z.shape != (y.size, x.size):
            return None
        x_index = int(np.argmin(np.abs(x - state.value.x)))
        y_index = int(np.argmin(np.abs(y - state.value.y)))
        valid = np.asarray(
            getattr(payload, "valid", np.ones(z.shape, dtype=bool)), dtype=bool
        )
        valid = np.broadcast_to(valid, z.shape)
        value = z[y_index, x_index]
        if not bool(valid[y_index, x_index]) or not bool(np.isfinite(value)):
            return "NaN"
        return float(value)

    def _histogram_threshold_text(self, state: SelectorState) -> str:
        """Describe one effective threshold from the already-painted bins."""

        if state.kind is not SelectorKind.THRESHOLD:
            return ""
        semantic = self.spec.cell if isinstance(self.spec, FacetGridPlot) else self.spec
        if not isinstance(semantic, HistogramPlot):
            return ""
        payload = self._last_payload
        if isinstance(self.spec, FacetGridPlot):
            cells = tuple(getattr(payload, "cells", ()))
            index = self._focused_facet_index if state.facet_index is None else state.facet_index
            if index is None or index < 0 or index >= len(cells):
                return ""
            payload = getattr(cells[index], "payload", cells[index])
        edges = getattr(getattr(payload, "edges", None), "display", None)
        counts = getattr(payload, "counts", None)
        if edges is None or counts is None:
            return ""
        edges = np.asarray(edges, dtype=float).reshape(-1)
        weights = np.asarray(counts, dtype=float).reshape(-1)
        if edges.size != weights.size + 1 or not weights.size:
            return ""
        total = float(np.sum(weights))
        threshold = float(state.value)
        if total <= 0.0:
            left_fraction = 0.0
        else:
            index = int(np.clip(
                np.searchsorted(edges, threshold, side="right") - 1,
                -1,
                len(weights),
            ))
            if index < 0:
                left_fraction = 0.0
            elif index >= len(weights):
                left_fraction = 1.0
            else:
                width = max(float(edges[index + 1] - edges[index]), 1.0e-300)
                fraction = float(np.clip(
                    (threshold - edges[index]) / width,
                    0.0,
                    1.0,
                ))
                left_fraction = float(
                    (np.sum(weights[:index]) + weights[index] * fraction) / total
                )
        return (
            f"th={_compact_engineering(threshold)}\n"
            f"L/R={100.0 * left_fraction:.1f}%/"
            f"{100.0 * (1.0 - left_fraction):.1f}%"
        )

    def _threshold_uses_x_axis(self) -> bool:
        semantic = self.spec.cell if isinstance(self.spec, FacetGridPlot) else self.spec
        return isinstance(semantic, HistogramPlot)

    def set_selector_scene_exclusion(self, kind: SceneKind | None) -> None:
        """Keep one externally composited candidate out of the canvas raster.

        The immutable selector scene remains complete.  Only its Matplotlib
        artists are excluded, allowing a raster frontend to paint the same
        scene primitives exactly once without a second background redraw.
        """

        if kind is not None and not isinstance(
            kind, (SelectorKind, SelectorSceneKind)
        ):
            raise TypeError("kind must be a selector scene kind or None")
        self._selector_scene_exclusion_kind = kind

    def preview_color_limits(self, low: float, high: float) -> None:
        """Preview image normalization without committing display state."""

        selected = np.asarray((low, high), dtype=float)
        if not np.all(np.isfinite(selected)) or selected[0] >= selected[1]:
            raise ValueError("preview color limits must be finite and increasing")
        key = "image" if isinstance(self.spec, ImagePlot) else None
        image = None if key is None else self._artists.get(key)
        if image is None:
            raise TypeError("color-limit preview requires an Image")
        with style_context(self.style):
            image.set_clim(float(selected[0]), float(selected[1]))
            colorbar = self._artists.get(f"{key}:colorbar")
            if colorbar is not None:
                low, high = map(float, selected)
                colorbar.set_ticks((low, high))
                label_chars = self.style.render.colorbar_endpoint_label_chars
                colorbar.set_ticklabels(
                    (
                        _compact_engineering(low, length=label_chars),
                        _compact_engineering(high, length=label_chars),
                    )
                )
                self._artists.pop(f"{key}:colorbar_state", None)
        self._mark_axes_chrome_dirty(*self._axes.get("colorbar", ()))

    @staticmethod
    def _selector_target_for_axis(axis: Any) -> SelectorTarget:
        role, separator, suffix = str(axis.get_gid() or "main").partition(":")
        return SelectorTarget(
            role or "main",
            int(suffix) if separator and suffix.isdigit() else None,
        )

    def _axis_for_selector_target(self, target: SelectorTarget) -> Any:
        for axis in self._figure.axes:
            if self._selector_target_for_axis(axis) == target:
                return axis
        raise KeyError(f"selector target is not present: {target!r}")

    def selector_scene(
        self,
        snapshot: SelectorSnapshot,
        *,
        color_candidate: ColorLimitCandidate | None = None,
    ) -> SelectorScene:
        """Resolve an exact transient scene without mutating plot artists."""

        if not isinstance(snapshot, SelectorSnapshot):
            raise TypeError("snapshot must be SelectorSnapshot")
        # Rebuild the owner from each transient snapshot.  The press-time
        # transform belongs to the input adapter, not to a frozen scene owner;
        # this keeps labels/geometry synchronized with the current live front.
        owner = self._make_selector_scene_owner(snapshot)
        return owner.build(
            snapshot,
            color_state=(
                color_candidate
                if color_candidate is not None
                else self._resolved_color_limit_state()
            ),
        )

    def _resolved_color_limit_state(self) -> ColorLimitState | None:
        if not self._axes.get("distribution"):
            return None
        image = self._active_image_artist()
        if image is None:
            return None
        low, high = sorted(map(float, image.get_clim()))
        if not np.all(np.isfinite((low, high))) or low >= high:
            return None
        return ColorLimitState(NumericRange(low, high))

    def _make_selector_scene_owner(
        self,
        snapshot: SelectorSnapshot,
    ) -> SelectorSceneOwner:
        from matplotlib.colors import to_rgba

        if not isinstance(snapshot, SelectorSnapshot):
            raise TypeError("snapshot must be SelectorSnapshot")

        policy = self.style.render
        pulse_factor = 1.0
        if isinstance(self.spec, PulseTimelinePlot) and isinstance(
            self._last_payload, PulseTimelineData
        ):
            pulse_factor, _unit = pulse_time_scale(
                self._last_payload,
                self._last_state["x_display_unit"],
            )
        threshold = self.style.artists.threshold_line
        threshold_rgba = tuple(map(float, to_rgba(threshold.color, threshold.alpha)))
        threshold_label_rgba = tuple(map(float, to_rgba(self.style.palette.threshold)))
        contexts = []
        for state in snapshot.states:
            axis = self._selector_axis(state)
            image = self._image_selector_artist(state)
            cmap = None if image is None else image.get_cmap()
            selector_rgba = tuple(map(float, to_rgba(
                self.style.palette.line_single
                if cmap is None
                else cmap(policy.colormap_high_fraction)
            )))
            sample_value = self._image_cross_value(state) if image is not None else None
            sample_axis = (
                self._axes["distribution"][0]
                if isinstance(self.spec, ImagePlot)
                and state.kind is SelectorKind.CROSSHAIR
                and self._axes.get("distribution")
                else None
            )
            clim = None if image is None else tuple(map(float, image.get_clim()))
            contexts.append(SelectorItemContext(
                kind=state.kind,
                target=self._selector_target_for_axis(axis),
                label_target=self._selector_target_for_axis(axis),
                x_limits=tuple(map(float, axis.get_xlim())),
                y_limits=tuple(map(float, axis.get_ylim())),
                selector_rgba=selector_rgba,
                color_limit_rgba=(selector_rgba, selector_rgba),
                threshold_rgba=threshold_rgba,
                threshold_label_rgba=threshold_label_rgba,
                threshold_uses_x=self._threshold_uses_x_axis(),
                threshold_text=self._histogram_threshold_text(state),
                x_label_factor=pulse_factor,
                sample_value=sample_value,
                sample_span=(None if clim is None else abs(clim[1] - clim[0])),
                sample_target=(
                    None if sample_axis is None else self._selector_target_for_axis(sample_axis)
                ),
                sample_x_limits=(
                    None if sample_axis is None else tuple(map(float, sample_axis.get_xlim()))
                ),
                sample_rgba=(
                    None
                    if cmap is None or sample_axis is None
                    else tuple(map(float, to_rgba(
                        cmap(policy.selector_sample_colormap_fraction),
                        policy.selector_sample_alpha,
                    )))
                ),
            ))
        color_context = None
        distribution = self._axes.get("distribution", ())
        image = self._active_image_artist()
        if distribution and image is not None:
            color_axis = distribution[0]
            cmap = image.get_cmap()
            color_limit_rgba = tuple(
                tuple(map(float, to_rgba(color)))
                for color in (
                    cmap(policy.colormap_low_fraction),
                    cmap(policy.colormap_high_fraction),
                )
            )
            label_axis = self._axes.get("image", [self.primary_axes])[0]
            color_context = SelectorItemContext(
                kind=SelectorSceneKind.COLOR_LIMITS,
                target=self._selector_target_for_axis(color_axis),
                label_target=self._selector_target_for_axis(label_axis),
                x_limits=tuple(map(float, color_axis.get_xlim())),
                y_limits=tuple(map(float, color_axis.get_ylim())),
                selector_rgba=tuple(map(float, to_rgba(self.style.palette.line_single))),
                color_limit_rgba=color_limit_rgba,
                threshold_rgba=threshold_rgba,
                threshold_label_rgba=threshold_label_rgba,
                threshold_uses_x=False,
            )
        visual = self.style.artists
        return SelectorSceneOwner(
            tuple(contexts),
            SelectorSceneStyle(
                line_width_pt=visual.selector_line_width,
                line_alpha=visual.selector_alpha,
                handle_size_pt=visual.selector_handle_size_pt,
                handle_edge_width_pt=visual.selector_handle_edge_width,
                crosshair_size_pt=visual.crosshair_marker_size_pt,
                annotation_pt=self.style.fonts.annotation_pt,
                side_distribution_width_pt=visual.side_distribution_linewidth,
                selector_zorder=visual.selector_zorder,
                font_family=self.style.fonts.family,
                label_inset_fraction=policy.axes_text_inset_fraction,
                label_line_fraction=policy.selector_label_line_fraction,
                threshold_width_pt=threshold.linewidth,
                threshold_linestyle=threshold.linestyle,
            ),
            color_context,
        )

    @staticmethod
    def _selector_primitive_topology(primitive: SelectorPrimitive) -> object:
        return type(primitive), primitive.target, getattr(primitive, "shape", None)

    def _new_selector_artist(self, primitive: SelectorPrimitive) -> Any:
        axis = self._axis_for_selector_target(primitive.target)
        if isinstance(primitive, SelectorLine):
            (artist,) = axis.plot((), (), clip_on=True)
            return artist
        if isinstance(primitive, SelectorMarkers):
            return axis.scatter(
                (),
                (),
                marker={"square": "s", "circle": "o"}[primitive.shape],
                clip_on=True,
            )
        return axis.text(
            0.0,
            0.0,
            "",
            transform=axis.transAxes,
            va="top",
            clip_on=True,
        )

    @staticmethod
    def _mutate_selector_artist(artist: Any, primitive: SelectorPrimitive) -> None:
        if isinstance(primitive, SelectorLine):
            points = np.asarray(primitive.points, dtype=float)
            artist.set_data(points[:, 0], points[:, 1])
            artist.set_color(primitive.color)
            artist.set_linewidth(primitive.width_pt)
            artist.set_linestyle(primitive.linestyle)
        elif isinstance(primitive, SelectorMarkers):
            artist.set_offsets(np.asarray(primitive.points, dtype=float))
            artist.set_sizes(np.full(len(primitive.points), primitive.size_pt**2))
            artist.set_facecolors((primitive.facecolor,))
            artist.set_edgecolors(
                "none" if primitive.edgecolor is None else (primitive.edgecolor,)
            )
            artist.set_linewidths(primitive.edge_width_pt)
        else:
            artist.set_text(primitive.text)
            artist.set_position(primitive.position)
            artist.set_horizontalalignment(primitive.horizontal_alignment)
            artist.set_color(primitive.color)
            artist.set_fontfamily(primitive.font_family)
            artist.set_fontsize(primitive.font_size_pt)
        artist.set_clip_on(True)
        artist.set_visible(True)
        artist.set_zorder(primitive.zorder)

    def _update_selectors(self, snapshot: SelectorSnapshot) -> None:
        if not isinstance(snapshot, SelectorSnapshot):
            raise TypeError("selector update requires SelectorSnapshot")
        if self._selector_candidate is not None:
            snapshot = SelectorSnapshot(
                snapshot.committed,
                self._selector_candidate,
            )
        owner = self._make_selector_scene_owner(snapshot)
        scene = owner.build(
            snapshot,
            color_state=(
                self._color_limit_candidate
                if self._color_limit_candidate is not None
                else self._resolved_color_limit_state()
            ),
        )
        active_kinds = set(scene.selector_kinds)
        for kind in tuple(self._selector_artists):
            if kind in active_kinds:
                continue
            self._remove_artists(self._selector_artists.pop(kind))
            self._selector_topologies.pop(kind, None)
        for kind, primitives in scene.groups:
            topology = tuple(map(self._selector_primitive_topology, primitives))
            artists = self._selector_artists.get(kind)
            if artists is None or self._selector_topologies.get(kind) != topology:
                if artists is not None:
                    self._remove_artists(artists)
                artists = tuple(self._new_selector_artist(item) for item in primitives)
                self._selector_artists[kind] = artists
                self._selector_topologies[kind] = topology
            for artist, primitive in zip(artists, primitives, strict=True):
                self._mutate_selector_artist(artist, primitive)
            if kind is self._selector_scene_exclusion_kind:
                for artist in artists:
                    artist.set_visible(False)

    def _fit_annotation_text(self, overlay: FitOverlay) -> str:
        lines: list[str] = []
        if overlay.formula:
            lines.append(overlay.formula)
        for parameter in overlay.parameter_display:
            uncertainty = (
                "n/a"
                if parameter.standard_error is None
                else _compact_engineering(parameter.standard_error)
            )
            unit = f" {parameter.unit}" if parameter.unit else ""
            lines.append(
                f"{parameter.label} = {_compact_engineering(parameter.value)} "
                f"± {uncertainty}{unit}"
            )
        return "\n".join(lines)

    def _restore_fit_source_lines(self) -> None:
        for line, was_visible in self._fit_hidden_source_lines:
            if getattr(line, "axes", None) is not None:
                line.set_visible(was_visible)
        self._fit_hidden_source_lines = ()
        if self._fit_source_scatter is not None:
            self._fit_source_scatter.set_offsets(np.empty((0, 2), dtype=float))
            self._fit_source_scatter.set_visible(False)

    def _show_fit_source_scatter(
        self,
        axis: Any,
        source_lines: Sequence[Any],
    ) -> None:
        hidden = tuple(
            (line, was_visible)
            for line, was_visible in self._fit_hidden_source_lines
            if getattr(line, "axes", None) is axis
            and any(line is candidate for candidate in source_lines)
        )
        visible = tuple(
            line
            for line in source_lines
            if getattr(line, "axes", None) is axis and line.get_visible()
        )
        if visible:
            active_lines = visible
            self._fit_hidden_source_lines = tuple((line, True) for line in visible)
        else:
            active_lines = tuple(line for line, was_visible in hidden if was_visible)
        point_groups: list[np.ndarray] = []
        point_count = 0
        for line in active_lines:
            x = np.asarray(line.get_xdata(), dtype=float).reshape(-1)
            y = np.asarray(line.get_ydata(), dtype=float).reshape(-1)
            if x.shape != y.shape:
                continue
            finite = np.isfinite(x) & np.isfinite(y)
            if bool(np.any(finite)):
                points = np.column_stack((x[finite], y[finite]))
                point_groups.append(points)
                point_count += len(points)
        if (
            not point_groups
            or point_count >= self.style.render.fit_source_scatter_max_points
        ):
            self._restore_fit_source_lines()
            return
        scatter = self._fit_source_scatter
        if scatter is not None and getattr(scatter, "axes", None) is not axis:
            self._remove_artists((scatter,))
            scatter = None
        if scatter is None:
            scatter = axis.scatter(
                (),
                (),
                s=self.style.artists.fit_source_scatter_area_pt2,
                color=self.style.palette.data_scatter,
                edgecolors="none",
                clip_on=True,
                zorder=self.style.artists.fit_source_scatter_zorder,
            )
            self._fit_source_scatter = scatter
        scatter.set_offsets(np.concatenate(point_groups, axis=0))
        scatter.set_visible(True)
        for line, _was_visible in self._fit_hidden_source_lines:
            line.set_visible(False)

    def _set_fit_mode(self, active: bool) -> None:
        """Keep source presentation stable while fit-result topology changes."""

        if not active:
            self._restore_fit_source_lines()
            return
        semantic = self.spec.cell if isinstance(self.spec, FacetGridPlot) else self.spec
        if not isinstance(semantic, (CurvePlot, RollingPlot)):
            self._restore_fit_source_lines()
            return
        if isinstance(self.spec, FacetGridPlot):
            index = self._focused_facet_index
            axes = self._axes.get("facet_cell", ())
            if index is None or index < 0 or index >= len(axes):
                self._restore_fit_source_lines()
                return
            axis = axes[index]
        else:
            axis = self.primary_axes
        source_lines: list[Any] = []
        for key, value in self._artists.items():
            if key not in {"curve", "rolling:history"} and not key.startswith("facet:"):
                continue
            if isinstance(value, list):
                source_lines.extend(
                    item
                    for item in value
                    if getattr(item, "axes", None) is axis
                    and hasattr(item, "get_xdata")
                )
        self._show_fit_source_scatter(axis, source_lines[:1])

    def _fit_target(self, overlay: FitOverlay) -> tuple[Any, PlotSpec]:
        if isinstance(self.spec, FacetGridPlot):
            index = (
                self._focused_facet_index
                if overlay.facet_index is None
                else overlay.facet_index
            )
            if index is None or index < 0 or index >= self._visible_facet_count:
                raise IndexError("fit facet index is outside the current grid")
            return self._axes["facet_cell"][index], self.spec.cell
        return self.primary_axes, self.spec

    @staticmethod
    def _fit_family_for(semantic: PlotSpec, overlay: FitOverlay) -> str:
        if not overlay.success:
            return "failure"
        if overlay.radial_glyph is not None:
            return "glyph"
        if any(polyline.role == "component" for polyline in overlay.polylines):
            return f"components:{len(overlay.polylines)}"
        if overlay.polylines:
            return "histogram" if isinstance(semantic, HistogramPlot) else "curve"
        return "annotation"

    def _clear_fit_topology(self) -> None:
        self._remove_artists(self._fit_artists)
        self._fit_artists.clear()
        self._fit_slots.clear()
        self._fit_axis = None
        self._fit_family = None
        self._fit_batch_identity = None

    def _fit_polyline_token(self, semantic: PlotSpec, polyline: FitPolyline) -> Any:
        if polyline.role == "component":
            component_tokens = self.style.artists.bimodal_fit_lines[:2]
            return component_tokens[polyline.component_index % len(component_tokens)]
        if polyline.role == "total" or isinstance(semantic, HistogramPlot):
            return self.style.artists.bimodal_fit_lines[2]
        return self.style.artists.curve_fit_line

    def _build_fit_topology(
        self,
        axis: Any,
        family: str,
        semantic: PlotSpec,
        overlay: FitOverlay,
    ) -> None:
        from matplotlib.patches import Ellipse

        self._fit_axis = axis
        self._fit_family = family
        if family == "failure":
            diagnostic = axis.text(
                0.02,
                0.98,
                "",
                transform=axis.transAxes,
                va="top",
                color=self.style.artists.fit_failure_color,
                fontsize=self.style.fonts.fit_annotation_pt,
                clip_on=True,
                zorder=self.style.artists.fit_annotation_zorder,
            )
            self._fit_slots["diagnostic"] = diagnostic
            self._fit_artists.append(diagnostic)
            return
        if family == "glyph":
            center = axis.scatter(
                (),
                (),
                color=self.style.artists.fit_radial_color,
                s=self.style.artists.fit_radial_center_area_pt2,
                clip_on=True,
                zorder=self.style.artists.fit_radial_zorder,
            )
            ring = Ellipse(
                (0.0, 0.0),
                width=0.0,
                height=0.0,
                edgecolor=self.style.artists.fit_radial_color,
                facecolor="none",
                linewidth=self.style.artists.fit_radial_ring_linewidth,
                alpha=self.style.artists.fit_radial_ring_alpha,
                clip_on=True,
                zorder=self.style.artists.fit_radial_zorder,
            )
            center.set_visible(False)
            ring.set_visible(False)
            axis.add_patch(ring)
            self._fit_slots.update(center=center, ring=ring)
            self._fit_artists.extend((center, ring))
        elif family.startswith("components:"):
            lines = []
            for polyline in overlay.polylines:
                token = self._fit_polyline_token(semantic, polyline)
                (line,) = axis.plot((), (), **token.kwargs())
                lines.append(line)
            self._fit_slots["lines"] = tuple(lines)
            self._fit_artists.extend(lines)
        elif family in {"histogram", "curve"}:
            token = self._fit_polyline_token(semantic, overlay.polylines[0])
            (line,) = axis.plot((), (), **token.kwargs())
            self._fit_slots["line"] = line
            self._fit_artists.append(line)
        if isinstance(semantic, HistogramPlot):
            token = self.style.artists.threshold_line
            suggested = token.kwargs()
            suggested["linestyle"] = "--"
            suggested["alpha"] = min(float(token.alpha), 0.72)
            threshold = axis.axvline(0.0, **suggested)
            threshold.set_visible(False)
            self._fit_slots["suggested_threshold"] = threshold
            self._fit_artists.append(threshold)
        annotation = axis.text(
            self.style.render.axes_text_inset_fraction,
            1.0 - self.style.render.axes_text_inset_fraction,
            "",
            transform=axis.transAxes,
            ha="left",
            va="top",
            clip_on=True,
            zorder=self.style.artists.fit_annotation_zorder,
            fontsize=self.style.fonts.fit_annotation_pt,
            color=self.style.palette.fit_text,
        )
        annotation.set_visible(False)
        self._fit_slots["annotation"] = annotation
        self._fit_artists.append(annotation)

    def _update_fit_annotation(
        self,
        overlay: FitOverlay,
    ) -> None:
        annotation = self._fit_slots["annotation"]
        content = self._fit_annotation_text(overlay)
        annotation.set_text(content)
        annotation.set_visible(bool(content))

    @staticmethod
    def _set_fit_line(line: Any, polyline: FitPolyline) -> None:
        order = np.argsort(polyline.x)
        line.set_data(polyline.x[order], polyline.y[order])
        line.set_visible(True)

    def _update_fit_primitives(self, family: str, overlay: FitOverlay) -> None:
        if family == "glyph":
            center = self._fit_slots["center"]
            ring = self._fit_slots["ring"]
            glyph = overlay.radial_glyph
            if glyph is None:
                raise RuntimeError("glyph fit family requires one radial glyph")
            center.set_offsets(
                np.asarray(((glyph.center_x, glyph.center_y),), dtype=float)
            )
            ring.set_center((glyph.center_x, glyph.center_y))
            ring.set_width(2.0 * glyph.radius_x)
            ring.set_height(2.0 * glyph.radius_y)
            center.set_visible(True)
            ring.set_visible(True)
            return
        if family.startswith("components:"):
            for line, polyline in zip(
                self._fit_slots["lines"],
                overlay.polylines,
                strict=True,
            ):
                self._set_fit_line(line, polyline)
            return
        if family in {"histogram", "curve"}:
            self._set_fit_line(self._fit_slots["line"], overlay.polylines[0])

    def _update_fit_threshold(self, overlay: FitOverlay) -> None:
        threshold = self._fit_slots.get("suggested_threshold")
        if threshold is None:
            return
        value = overlay.suggested_threshold
        if value is None:
            threshold.set_visible(False)
            return
        threshold.set_xdata((value, value))
        threshold.set_visible(True)

    def _build_static_fit_overlay(
        self,
        overlay: FitOverlay,
        *,
        include_annotation: bool,
    ) -> tuple[Any, ...]:
        """Create one immutable batch-cell overlay with the canonical tokens."""

        from matplotlib.patches import Ellipse

        axis, semantic = self._fit_target(overlay)
        family = self._fit_family_for(semantic, overlay)
        artists: list[Any] = []
        if family == "failure":
            if not include_annotation:
                return ()
            diagnostic_text = overlay.diagnostic.strip() or "fit failed"
            if len(diagnostic_text) > 72:
                diagnostic_text = diagnostic_text[:69] + "..."
            artists.append(axis.text(
                0.02,
                0.98,
                f"fit: {diagnostic_text}",
                transform=axis.transAxes,
                va="top",
                color=self.style.artists.fit_failure_color,
                fontsize=self.style.fonts.fit_annotation_pt,
                clip_on=True,
                zorder=self.style.artists.fit_annotation_zorder,
            ))
            return tuple(artists)
        if family == "glyph":
            glyph = overlay.radial_glyph
            if glyph is None:
                raise RuntimeError("glyph fit family requires one radial glyph")
            center = axis.scatter(
                (glyph.center_x,),
                (glyph.center_y,),
                color=self.style.artists.fit_radial_color,
                s=self.style.artists.fit_radial_center_area_pt2,
                clip_on=True,
                zorder=self.style.artists.fit_radial_zorder,
            )
            ring = Ellipse(
                (glyph.center_x, glyph.center_y),
                width=2.0 * glyph.radius_x,
                height=2.0 * glyph.radius_y,
                edgecolor=self.style.artists.fit_radial_color,
                facecolor="none",
                linewidth=self.style.artists.fit_radial_ring_linewidth,
                alpha=self.style.artists.fit_radial_ring_alpha,
                clip_on=True,
                zorder=self.style.artists.fit_radial_zorder,
            )
            axis.add_patch(ring)
            artists.extend((center, ring))
        else:
            for polyline in overlay.polylines:
                token = self._fit_polyline_token(semantic, polyline)
                order = np.argsort(polyline.x)
                (line,) = axis.plot(
                    polyline.x[order],
                    polyline.y[order],
                    **token.kwargs(),
                )
                artists.append(line)
        if overlay.suggested_threshold is not None and isinstance(
            semantic,
            HistogramPlot,
        ):
            token = self.style.artists.threshold_line
            suggested = token.kwargs()
            suggested["linestyle"] = "--"
            suggested["alpha"] = min(float(token.alpha), 0.72)
            artists.append(axis.axvline(
                float(overlay.suggested_threshold),
                **suggested,
            ))
        content = self._fit_annotation_text(overlay) if include_annotation else ""
        if content:
            artists.append(axis.text(
                self.style.render.axes_text_inset_fraction,
                1.0 - self.style.render.axes_text_inset_fraction,
                content,
                transform=axis.transAxes,
                ha="left",
                va="top",
                clip_on=True,
                zorder=self.style.artists.fit_annotation_zorder,
                fontsize=self.style.fonts.fit_annotation_pt,
                color=self.style.palette.fit_text,
            ))
        return tuple(artists)

    def _update_facet_thresholds(
        self,
        thresholds: tuple[float | None, ...],
    ) -> None:
        """Paint authoritative per-cell Histogram annotations in place."""

        if not isinstance(self.spec, FacetGridPlot) or not isinstance(
            self.spec.cell,
            HistogramPlot,
        ):
            thresholds = ()
        active = {
            index: float(value)
            for index, value in enumerate(thresholds)
            if value is not None and index < self._visible_facet_count
        }
        for index in tuple(self._facet_threshold_artists):
            if index in active:
                continue
            self._remove_artists(self._facet_threshold_artists.pop(index))
        token = self.style.artists.threshold_line
        axes = self._axes.get("facet_cell", ())
        for index, value in active.items():
            artists = self._facet_threshold_artists.get(index)
            if artists is None:
                axis = axes[index]
                line = axis.axvline(value, **token.kwargs())
                label = axis.text(
                    1.0 - self.style.render.axes_text_inset_fraction,
                    1.0 - self.style.render.axes_text_inset_fraction,
                    "",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    clip_on=True,
                    zorder=self.style.artists.fit_annotation_zorder,
                    fontsize=self.style.fonts.annotation_pt,
                    color=self.style.palette.threshold,
                )
                artists = (line, label)
                self._facet_threshold_artists[index] = artists
            line, label = artists
            line.set_xdata((value, value))
            line.set_visible(True)
            label.set_text(f"th={_compact_engineering(value)}")
            label.set_visible(
                self._visible_facet_count == 1
                or self._facet_focus_index == index
            )

    def _update_fit(
        self,
        overlay: FitOverlay | tuple[FitOverlay, ...] | None,
    ) -> None:
        if overlay is None:
            self._clear_fit_topology()
            return
        if isinstance(overlay, tuple):
            identity = tuple(id(item) for item in overlay)
            if self._fit_batch_identity == identity:
                return
            self._clear_fit_topology()
            for item in overlay:
                include_annotation = self._visible_facet_count == 1 or (
                    self._facet_focus_index is not None
                    and item.facet_index == self._facet_focus_index
                )
                self._fit_artists.extend(self._build_static_fit_overlay(
                    item,
                    include_annotation=include_annotation,
                ))
            self._fit_batch_identity = identity
            self._fit_family = "facet-batch"
            return
        axis, semantic = self._fit_target(overlay)
        family = self._fit_family_for(semantic, overlay)
        if self._fit_axis is not axis or self._fit_family != family:
            self._clear_fit_topology()
            self._build_fit_topology(axis, family, semantic, overlay)
        if not overlay.success:
            diagnostic_text = overlay.diagnostic.strip() or "fit failed"
            if len(diagnostic_text) > 72:
                diagnostic_text = diagnostic_text[:69] + "..."
            diagnostic = self._fit_slots["diagnostic"]
            diagnostic.set_text(f"fit: {diagnostic_text}")
            diagnostic.set_visible(True)
            return

        self._update_fit_primitives(family, overlay)
        self._update_fit_threshold(overlay)
        self._update_fit_annotation(overlay)

    def _rgba_buffer(self) -> np.ndarray:
        """Return the already-painted canvas buffer in the surface contract."""

        source = np.asarray(self._figure.canvas.buffer_rgba(), dtype=np.uint8)
        target_width, target_height = self.plan.raster_size
        actual_height, actual_width = source.shape[:2]
        if (actual_width, actual_height) == (target_width, target_height):
            return readonly_copy(source)

        # Fractional DPR can put Agg's floor allocation one trailing pixel away
        # from the rounded frontend contract.  Preserve artist transforms and
        # adjust only the right and bottom handoff edges.
        pixels = np.empty((target_height, target_width, 4), dtype=np.uint8)
        background = np.rint(
            np.clip(np.asarray(self._figure.get_facecolor()), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        pixels[...] = background
        copy_width = min(target_width, actual_width)
        copy_height = min(target_height, actual_height)
        pixels[:copy_height, :copy_width] = source[:copy_height, :copy_width]
        pixels.setflags(write=False)
        return pixels

    def capture_rgba(
        self,
        *,
        redraw: bool = False,
    ) -> np.ndarray:
        """Capture the already composed front without changing artist state."""

        if redraw:
            self.draw()
        return self._rgba_buffer()

    def rgba(self) -> np.ndarray:
        """Draw the current scene and return an immutable RGBA snapshot."""

        self.draw()
        return self._rgba_buffer()

    def save(self, path: str | Path | BytesIO, *, dpi: float | None = None, **kwargs: Any) -> None:
        dynamic = self._dynamic_artists()
        animated = tuple(artist.get_animated() for artist in dynamic)
        for artist in dynamic:
            artist.set_animated(False)
        try:
            with style_context(self.style):
                self._figure.savefig(path, dpi=dpi or self.plan.dpi, **kwargs)
        finally:
            for artist, previous in zip(dynamic, animated):
                artist.set_animated(previous)
            self.draw()


__all__ = ["MatplotlibRenderer", "RenderFrame"]
