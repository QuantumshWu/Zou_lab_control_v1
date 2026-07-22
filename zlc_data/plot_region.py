"""Plot-independent data-selection contracts.

Selectors live in a frontend, but the meaning of a selection does not.  A
rectangle on an image, an x range on a trace, a value range on a distribution,
and a rectangle around trap sites are all the same operation: bounds over named
coordinates, optionally scoped to one grid cell.  This module is deliberately
free of Qt and Matplotlib so processors and saved-figure tools consume the exact
same selection as a live plot.

NAME WARNING: this module's ``Selection`` is NOT ``zlc_data.Selection``.
This one is a plot REGION - a set of per-axis ranges plus the row selection
and mask bindings a live panel builds from a rubber-band drag.  The
top-level ``zlc_data.Selection`` is an index/coordinate selection over a
dataset.  Both are legitimate and neither is a superset of the other, so the
module is named for the concept and is deliberately absent from the package
``__all__``: an importer must say ``from zlc_data.plot_region import
Selection`` and thereby say which one it means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AxisRange:
    """Inclusive finite bounds for one named coordinate axis."""

    axis: str
    low: float
    high: float

    def __post_init__(self) -> None:
        axis = str(self.axis).strip()
        if not axis:
            raise ValueError("selection axis name must not be empty")
        low, high = sorted((float(self.low), float(self.high)))
        if not np.isfinite([low, high]).all():
            raise ValueError(f"selection bounds for {axis!r} must be finite")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def mask(self, values) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return np.isfinite(array) & (array >= self.low) & (array <= self.high)


@dataclass(frozen=True)
class Selection:
    """Bounds in a named coordinate frame, optionally scoped to a grid cell.

    ``scope`` is an immutable path such as ``("facet:repeat", "cell:2")``.  It
    prevents a cell-local drag from being misapplied to every subplot while
    remaining independent of any concrete grid widget.
    """

    ranges: tuple[AxisRange, ...] = ()
    frame: str = "data"
    scope: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        ranges = tuple(self.ranges)
        names = [item.axis for item in ranges]
        if len(set(names)) != len(names):
            raise ValueError(f"selection axes must be unique, got {names}")
        object.__setattr__(self, "ranges", ranges)
        object.__setattr__(self, "frame", str(self.frame or "data"))
        object.__setattr__(self, "scope", tuple(str(v) for v in self.scope))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @classmethod
    def x(cls, low: float, high: float, **kwargs) -> "Selection":
        return cls((AxisRange("x", low, high),), **kwargs)

    @classmethod
    def value(cls, low: float, high: float, **kwargs) -> "Selection":
        return cls((AxisRange("value", low, high),), **kwargs)

    @classmethod
    def rectangle(cls, x0: float, x1: float, y0: float, y1: float, **kwargs) -> "Selection":
        return cls((AxisRange("x", x0, x1), AxisRange("y", y0, y1)), **kwargs)

    @property
    def empty(self) -> bool:
        return not self.ranges

    def bounds(self, axis: str) -> tuple[float, float] | None:
        for item in self.ranges:
            if item.axis == str(axis):
                return (item.low, item.high)
        return None

    def mask(self, coordinates: Mapping[str, object], *, length: int | None = None) -> np.ndarray:
        """Vectorized mask over coordinate arrays.

        Missing coordinates are a contract error, never a silent fallback to all
        data.  An empty selection intentionally selects every finite row.
        """

        arrays = {str(name): np.asarray(value) for name, value in coordinates.items()}
        sizes = {int(array.size) for array in arrays.values() if array.ndim > 0}
        if length is None:
            if len(sizes) > 1:
                raise ValueError(f"selection coordinate sizes disagree: {sorted(sizes)}")
            length = next(iter(sizes), 1)
        length = int(length)
        if length < 0:
            raise ValueError("selection length must be non-negative")
        out = np.ones(length, dtype=bool)
        for item in self.ranges:
            if item.axis not in arrays:
                raise KeyError(
                    f"selection needs coordinate {item.axis!r}; available: {sorted(arrays)}")
            values = np.asarray(arrays[item.axis], dtype=float).reshape(-1)
            if values.size != length:
                raise ValueError(
                    f"selection coordinate {item.axis!r} has {values.size} values, expected {length}")
            out &= item.mask(values)
        return out

    def to_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "scope": list(self.scope),
            "ranges": [
                {"axis": item.axis, "low": item.low, "high": item.high}
                for item in self.ranges
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "Selection":
        payload = dict(payload or {})
        return cls(
            tuple(AxisRange(str(item["axis"]), float(item["low"]), float(item["high"]))
                  for item in payload.get("ranges", ())),
            frame=str(payload.get("frame", "data")),
            scope=tuple(str(v) for v in payload.get("scope", ())),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class SelectedData:
    """The concrete rows selected from one dataset."""

    coordinates: Mapping[str, np.ndarray]
    values: np.ndarray
    indices: np.ndarray
    selection: Selection = Selection()
    value_shape: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim == 0:
            values = values.reshape(1, 1)
        elif values.ndim == 1:
            values = values[:, None]
        coords = {str(key): np.asarray(value).reshape(-1)
                  for key, value in self.coordinates.items()}
        indices = np.asarray(self.indices, dtype=np.int64).reshape(-1)
        n = int(values.shape[0])
        if indices.size != n:
            raise ValueError(f"selected indices have {indices.size} rows, values have {n}")
        bad = {key: value.size for key, value in coords.items() if value.size != n}
        if bad:
            raise ValueError(f"selected coordinate lengths do not match values: {bad}, values={n}")
        object.__setattr__(self, "coordinates", coords)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "value_shape", tuple(int(v) for v in self.value_shape) or (1,))

    @property
    def count(self) -> int:
        return int(self.values.shape[0])


def select_rows(
    coordinates: Mapping[str, object],
    values,
    selection: Selection | None = None,
    *,
    finite: bool = True,
    value_shape: Sequence[int] = (1,),
) -> SelectedData:
    """Apply one selection to coordinates and values with no plot-specific branch."""

    selection = selection or Selection()
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        array = array[:, None]
    n = int(array.shape[0])
    coords = {str(key): np.asarray(value).reshape(-1) for key, value in coordinates.items()}
    mask = selection.mask(coords, length=n)
    if finite:
        mask &= np.isfinite(array.reshape(n, -1)).all(axis=1)
        for value in coords.values():
            mask &= np.isfinite(np.asarray(value, dtype=float))
    indices = np.flatnonzero(mask)
    return SelectedData(
        {key: value[indices] for key, value in coords.items()},
        array[indices],
        indices,
        selection,
        tuple(int(v) for v in value_shape),
    )


# --------------------------------------------------------------------------- region bindings
# A region REDUCER (RoiProcessor) applies one Selection to a raw canonical ``(R, P, *data_shape)``
# block with ZERO per-kind branch of its own.  What the block's cells MEAN -- image pixels, a 1-D
# index, a distribution's sample value, trap-site centres -- is per-plot-kind knowledge the FRONTEND
# owns (``live.region_binding``, the inverse of ``coerce_panel_value``).  It ships that knowledge to
# the reducer as PLAIN DATA carried on ``Selection.metadata["binding"]`` (serializable, so it survives
# save/load and never makes ``neutral_atom`` import ``frontend``).  These three constructors are the
# ONE definition of that binding vocabulary, shared by the frontend producer, the reducer, and tests.

def axis_crop_binding(axes: Sequence[tuple[str, int, float, float]]) -> dict[str, object]:
    """Binding for a CONTIGUOUS axis crop (a fixed grid): each entry maps ONE selection-axis name to
    ONE block axis whose per-cell coordinate is the linear ``origin + step * index`` (image pixels; a
    1-D index/point axis).  The reducer slices those axes to the selection's index span, so ``roi_frame``
    is a fixed-shape native sub-array.  ``axes`` items are ``(select_name, block_axis, origin, step)``
    (``block_axis`` may be negative -- resolved against the live block's rank)."""
    return {"mode": "axis-crop",
            "axes": [{"select": str(select), "axis": int(axis), "origin": float(origin), "step": float(step)}
                     for (select, axis, origin, step) in axes]}


def value_mask_binding() -> dict[str, object]:
    """Binding for a VALUE mask (a distribution's count-range): every sample cell's coordinate IS its
    own value, so there is NO fixed sub-frame.  The reducer masks out-of-range cells to NaN (a stable
    passthrough for ``roi_frame``) and reduces the rest per repeat -- ``roi_value`` is the meaningful
    output.  The selection ranges the value axis (``Selection.value``)."""
    return {"mode": "value-mask"}


def scatter_mask_binding(axis: int, coordinates: Mapping[str, object]) -> dict[str, object]:
    """Binding for a SCATTER position mask: the selection axes bind to EXPLICIT per-cell coordinate
    arrays over ONE block axis (trap-site centres for a site map; a 1-D curve's own non-index x
    samples).  These positions are not a contiguous grid, so ``roi_frame`` is a NaN passthrough and
    ``roi_value`` reduces the in-region cells.  ``coordinates`` maps each selection-axis name to a
    length-``block.shape[axis]`` array."""
    return {"mode": "scatter-mask", "axis": int(axis),
            "coordinates": {str(name): [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]
                            for name, values in dict(coordinates).items()}}


# --------------------------------------------------------------------------- region as a signal
# A plot PANEL publishes its drawn Selection as a per-panel hub CONTROL signal (``<slug>_region``) so an
# analysis processor can CONSUME it (reacting to a re-drag) and so the region itself is reusable.  The
# WHOLE selection (bounds + binding + frame + scope + origin + the panel's ``bins``) rides IN THE VALUE
# as one fixed-size JSON payload (a zero-padded uint8 vector), so the SignalSchema is BYTE-STABLE across
# every re-drag / retarget -- a republish can never change the schema fingerprint, never needs
# ``replace=True``, and never opens a ``SignalHistoryGap`` under a running consumer.  The schema's
# metadata carries exactly ONE fact: ``role='control'`` -- the single-source classification the console
# uses to keep control-plane signals out of the bindable data pool and out of the orphan GC.  Dependency
# free: the frontend producer, the ``neutral_atom`` consumer and the tests share this ONE definition, so
# ``neutral_atom`` never imports ``frontend`` (the sealed seam) and a saved region round-trips.

#: The schema-metadata ROLE marking a control-plane signal (a panel's drawn region): filtered out of
#: every signal picker and exempt from the console's orphan GC -- by this ONE classification, never by
#: a parallel name list.
CONTROL_ROLE = "control"
#: Fixed region payload size: the JSON-encoded Selection, zero-padded.  Fixed so the schema never
#: changes shape between drags; generous enough for a scatter-mask binding carrying per-site centres.
REGION_PAYLOAD_BYTES = 16384


def region_doc(selection: "Selection", *, bins: int | None = None) -> dict:
    """The serializable region document -- ``selection.to_dict()`` with the panel's ``bins`` (a hist
    fit's single binning source) folded into the metadata.  The ONE payload shape shared by the region
    signal (:func:`region_to_payload`), the panel config's persisted ``params['region']``, and the tests."""
    doc = selection.to_dict()
    if bins is not None:
        doc["metadata"] = {**dict(doc.get("metadata") or {}), "bins": int(bins)}
    return doc


def region_to_payload(selection: "Selection", *, bins: int | None = None) -> np.ndarray:
    """Encode a :class:`Selection` as the region signal's fixed-size uint8 JSON payload.

    Named ``*_payload`` rather than ``encode_``/``decode_`` on purpose: those
    prefixes are reserved in ``zlc_data`` for canonical artifact serialization, which
    ``zlc_data.codec`` owns and which every persisted-artifact decoder must
    delegate to.  This pair never crosses a storage boundary - it moves a
    region across the in-process signal hub, written and read by the same
    process in the same run - so it must not wear a name that claims
    otherwise.
    """
    blob = json.dumps(region_doc(selection, bins=bins)).encode("utf-8")
    if len(blob) > REGION_PAYLOAD_BYTES:
        raise ValueError(
            f"region selection serializes to {len(blob)} bytes; the fixed region payload is "
            f"{REGION_PAYLOAD_BYTES} bytes.")
    values = np.zeros(REGION_PAYLOAD_BYTES, dtype=np.uint8)
    values[: len(blob)] = np.frombuffer(blob, dtype=np.uint8)
    return values


def region_from_payload(values) -> "Selection":
    """Rebuild a :class:`Selection` from a region signal's payload -- the exact inverse of
    :func:`encode_region`.  An all-zero / empty payload is the empty selection (the whole frame)."""
    raw = bytes(np.asarray(values, dtype=np.uint8).reshape(-1)).rstrip(b"\x00")
    if not raw:
        return Selection()
    return Selection.from_dict(json.loads(raw.decode("utf-8")))


def region_bins(selection: "Selection | None") -> int | None:
    """The panel's ``bins`` carried on a decoded region selection, or ``None`` (not a hist region)."""
    if selection is None:
        return None
    bins = dict(selection.metadata or {}).get("bins")
    return None if bins is None else int(bins)


def region_tensor(selection: "Selection", *, bins: int | None = None):
    """The complete, publish-ready region signal value: the fixed-shape uint8 payload wrapped in a
    :class:`~.signal_tensor.SignalTensor` whose BYTE-STABLE schema carries ``role='control'``.  The ONE
    builder every region publisher (the console's drag path, a notebook, the tests) uses -- so the
    schema can never fork per call site and a republish is always same-definition (no replace)."""
    from .signal_tensor import SignalSchema, SignalTensor

    values = region_to_payload(selection, bins=bins)
    schema = SignalSchema(point_shape=(1,), data_shape=(REGION_PAYLOAD_BYTES,), dtype=np.uint8,
                          repeat_capacity=1, label="region",
                          description="per-panel drawn selection (control-plane)",
                          metadata={"role": CONTROL_ROLE})
    return SignalTensor(values.reshape(1, 1, REGION_PAYLOAD_BYTES), schema)


__all__ = [
    "AxisRange", "Selection", "SelectedData", "select_rows",
    "axis_crop_binding", "value_mask_binding", "scatter_mask_binding",
    "region_to_payload", "region_from_payload", "region_bins", "region_doc", "region_tensor",
    "CONTROL_ROLE", "REGION_PAYLOAD_BYTES",
]
