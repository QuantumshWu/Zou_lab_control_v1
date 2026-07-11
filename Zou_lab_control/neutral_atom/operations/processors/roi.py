"""ROI = a region reducer -- the generic "look at a piece of the data" func node.

Consumes a signal block PLUS its panel's ``<slug>_region`` control signal (a :class:`RegionProcessor`)
and republishes two DERIVED signals per shot, GENERIC over EVERY plot kind:

* ``roi_value`` -- ONE scalar per acquisition (the selected cells reduced by ``mean`` / ``sum`` /
  ``max``): an image rectangle, a 1-D x-range, a distribution's count-range, a site-centre rectangle
  ALL collapse to this one number.  Bind it to a rolling ``monitor``, or use it as a pulse-scan y --
  e.g. the MOT loss signal for a coil scan -- with no bespoke task needed.
* ``roi_frame`` -- a live view of just the region: a fixed-shape native CROP for a contiguous axis
  selection (image data axes, or a 1-D contiguous index axis), OR -- for a VALUE / scatter selection
  that has no fixed sub-frame (a distribution's value range; scattered site centres) -- the
  passthrough block with out-of-region cells set to NaN (a STABLE-shape signal, since the
  SignalSchema contract forbids a variable-length republish).

The region is the panel's own :class:`~...core.selection.Selection`, read LATEST-WINS from the region
signal each shot (a re-drag republishes it), so retarget is just a republish -- there is NO
console-side ``set_selection`` plumbing.  WHAT the consumed block's cells mean (image pixels, a 1-D
index, a sample value, site centres) is per-kind knowledge the FRONTEND owns (``live.region_binding``,
the inverse of ``coerce_panel_value``); it ships that knowledge as PLAIN DATA on the Selection's
``metadata["binding"]``, so a drag on ANY panel kind writes the region back through the region signal,
and ``neutral_atom`` never imports ``frontend``.

Reactive and decoupled exactly like Judge occupancy: it runs beside whatever
measurement publishes the block (virtual == real: only the data differ).
"""

from __future__ import annotations

import warnings

import numpy as np

from ...core.signal_tensor import SignalTensor
from ..logic import IMAGE_STREAM_HISTORY, SignalSpec
from ._region import RegionProcessor

#: The reduce verbs an ROI offers, in dropdown order -- ONE source for the node's dispatch, the
#: ParamDecl choices and the tests, so a verb added here appears everywhere at once.
ROI_REDUCERS = {"mean": np.mean, "sum": np.sum, "max": np.max}
#: The NaN-aware twins used on the value-/scatter-mask path (out-of-region cells are NaN), keyed by the
#: SAME verbs so ``ROI_REDUCERS`` stays the single source of the vocabulary.
_ROI_NAN_REDUCERS = {"mean": np.nanmean, "sum": np.nansum, "max": np.nanmax}


class RoiProcessor(RegionProcessor):
    """Reduce a region of a canonical signal block to one scalar, and republish the region itself --
    generic over plot kind because it carries a Selection (from the region signal), not pixels."""

    provides = ("roi_frame", "roi_value")

    def __init__(self, hub, *, source_expr=None, region=None, selection=None,
                 reduce: str = "mean", prefix: str = ""):
        super().__init__(hub, source_expr=source_expr, region=region, selection=selection, prefix=prefix)
        if str(reduce) not in ROI_REDUCERS:
            raise ValueError(f"ROI reduce must be one of {tuple(ROI_REDUCERS)}; got {reduce!r}.")
        self.reduce = str(reduce)
        self._init_roi_state(hub)

    def _init_roi_state(self, hub) -> None:
        """Seed the roi half's output-shape state (shared verbatim by AnalysisProcessor, whose merged
        __init__ cannot run this class's __init__ chain).  Output shapes are learned from the first
        block (and re-learned on a retarget, which the hub accepts because THIS node owns the schema --
        replace=True); whole-frame defaults come from the source schema so the output specs are valid
        before any frame arrives."""
        self._source_schema = None
        self._roi_frame_point: tuple[int, ...] = (1,)
        self._roi_frame_data: tuple[int, ...] = (1,)
        self._roi_frame_dtype = None
        self._roi_value_point: tuple[int, ...] = (1,)
        self._roi_region = None
        schemas = []
        for name in self.source_names:
            try:
                schemas.append(hub.schema(name))
            except KeyError:
                pass
        if schemas:
            first = schemas[0]
            if all(schema.point_shape == first.point_shape
                   and schema.data_shape == first.data_shape
                   and schema.repeat_capacity == first.repeat_capacity for schema in schemas):
                self._adopt_schema(first)

    def _adopt_schema(self, schema) -> None:
        """Learn the source schema + seed the whole-frame default output shapes (roi_frame = the whole
        block, roi_value = one scalar per (repeat, point) -- what the empty selection reduces to)."""
        self._source_schema = schema
        self._roi_frame_point = tuple(schema.point_shape)
        self._roi_frame_data = tuple(schema.data_shape)
        self._roi_value_point = tuple(schema.point_shape)

    def acquisition_parameters(self) -> dict[str, object]:
        """The editable source knobs (Edit tab) -- the reduce verb -- plus the declared spatial
        ``region`` of ``roi_frame`` (its source-pixel extent, for an image crop), so a roi_frame 2-D
        panel draws real pixel axes and a FURTHER drag on it round-trips through the same coordinates
        (the ROI-of-ROI case).  The Selection itself is NOT a param here: it lives on the region signal."""
        out: dict[str, object] = {"reduce": self.reduce}
        if self._roi_region is not None:
            out["region"] = list(self._roi_region)
        return out

    def set_acquisition_parameters(self, **values) -> None:
        if "reduce" in values:
            reduce = str(values["reduce"])
            if reduce not in ROI_REDUCERS:
                raise ValueError(f"ROI reduce must be one of {tuple(ROI_REDUCERS)}; got {reduce!r}.")
            self.reduce = reduce

    # ------------------------------------------------------------------ transform
    def transform(self, inputs: dict[str, object]) -> dict[str, object]:
        from ..signal_expr import hub_namespace

        value = self.source_expr.evaluate(hub_namespace(self.hub, inputs))
        block = np.asarray(value)
        if not self._input_tensors:
            raise ValueError("ROI requires a registered signal with a SignalSchema.")
        schemas = [tensor.schema for tensor in self._input_tensors.values()]
        schema = schemas[0]
        for other in schemas[1:]:
            schema.assert_compatible(other, check_repeat=True)
        expected = (block.shape[0], schema.point_count, *schema.data_shape)
        if block.ndim != 2 + len(schema.data_shape) or tuple(block.shape) != expected:
            raise ValueError(
                "ROI expression must preserve canonical (R,P,*data_shape) axes; "
                f"expected {expected}, got {block.shape}.")
        self._adopt_schema(schema)
        input_valid = self.input_validity(block.shape[:2])

        frame, frame_valid, reduced, reduce_axes, region = self._reduce_region(block, input_valid)
        r = int(block.shape[0])
        roi_value = np.asarray(reduced, dtype=np.float64).reshape(r, -1, 1)
        # Keep repeats separate ALWAYS; collapse the point axis only when the region reduced over it.
        base_valid = input_valid.any(axis=1, keepdims=True) if 1 in reduce_axes else input_valid
        value_valid = base_valid & np.isfinite(roi_value[..., 0])

        # Publish-time output shapes (re-learned every shot; a retarget's new crop shape is accepted
        # because this node owns the schema -- register_signal replace=True).
        self._roi_frame_point = (int(frame.shape[1]),)
        self._roi_frame_data = tuple(int(s) for s in frame.shape[2:]) or (1,)
        self._roi_frame_dtype = frame.dtype
        self._roi_value_point = (int(roi_value.shape[1]),)
        self._roi_region = region

        specs = {spec.name: spec for spec in self._bare_output_specs()}
        return {
            "roi_frame": SignalTensor(
                frame, specs["roi_frame"].to_schema(dtype=frame.dtype), valid=frame_valid),
            "roi_value": SignalTensor(
                roi_value, specs["roi_value"].to_schema(dtype=roi_value.dtype), valid=value_valid),
        }

    def _reduce_region(self, block, input_valid):
        """Apply the carried Selection (from the region signal) to the raw ``(R,P,*data)`` block through
        the ONE mode-driven machinery -- returns ``(roi_frame, frame_valid, reduced, reduce_axes,
        region)``.  There is NO per-kind branch here: the plot-side binding names WHICH axes the
        selection spans (``mode`` + axis descriptors), so an image rectangle, a 1-D index range, a
        distribution value-range and a site-centre rectangle all resolve identically."""
        selection = self.current_region()
        ndim = block.ndim
        binding = dict(selection.metadata.get("binding") or {})
        mode = str(binding.get("mode") or "axis-crop")
        reducer = ROI_REDUCERS[self.reduce]
        nan_reducer = _ROI_NAN_REDUCERS[self.reduce]

        if mode == "value-mask":
            # The coordinate of every sample cell IS its value: mask out-of-range cells to NaN and pool
            # every sample per repeat.  roi_frame is the stable-shape NaN passthrough (P unchanged).
            vb = selection.bounds("value")
            work = np.asarray(block, dtype=np.float64)
            if vb is not None:
                keep = np.isfinite(work) & (work >= vb[0]) & (work <= vb[1])
                work = np.where(keep, work, np.nan)
            reduce_axes = tuple(range(1, ndim))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slice -> NaN (invalid), by design
                reduced = nan_reducer(work, axis=reduce_axes)
            return work, input_valid, reduced, reduce_axes, None

        if mode == "scatter-mask":
            # Explicit per-cell positions over ONE block axis (site centres; a 1-D curve's own x): mask
            # cells outside the region to NaN.  roi_frame is the NaN passthrough (positions are not a grid).
            axis = int(binding.get("axis", -1)) % ndim
            n = int(block.shape[axis])
            coords = {name: np.asarray(vals, dtype=float)
                      for name, vals in (binding.get("coordinates") or {}).items()}
            keep_cells = (selection.mask(coords, length=n) if coords
                          else np.ones(n, dtype=bool))
            broadcast_shape = [1] * ndim
            broadcast_shape[axis] = n
            keep = np.broadcast_to(keep_cells.reshape(broadcast_shape), block.shape)
            work = np.where(keep & np.isfinite(np.asarray(block, dtype=np.float64)),
                            np.asarray(block, dtype=np.float64), np.nan)
            reduce_axes = (axis,)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                reduced = nan_reducer(work, axis=reduce_axes)
            return work, input_valid, reduced, reduce_axes, None

        # axis-crop (the default / whole-frame case): a CONTIGUOUS index span per axis -> a native crop.
        entries = list(binding.get("axes") or ())
        if not entries:
            # No binding: reduce the WHOLE frame over its data axes (the image-crop default) -- a fresh
            # node with an empty selection reduces every valid cell.
            data_axes = tuple(range(2, ndim)) or (ndim - 1,)
            entries = [{"select": None, "axis": axis, "origin": 0.0, "step": 1.0} for axis in data_axes]
        spans: dict[int, tuple[int, int]] = {}
        endpoints: dict[str, tuple[float, float, int, int]] = {}
        for entry in entries:
            axis = int(entry["axis"]) % ndim
            size = int(block.shape[axis])
            origin = float(entry.get("origin", 0.0))
            step = float(entry.get("step", 1.0)) or 1.0
            bounds = selection.bounds(entry.get("select")) if entry.get("select") else None
            if bounds is None:
                i0, i1 = 0, size
            else:
                lo = (bounds[0] - origin) / step
                hi = (bounds[1] - origin) / step
                i0 = int(np.clip(round(min(lo, hi)), 0, max(size - 1, 0)))
                i1 = int(np.clip(round(max(lo, hi)), i0 + 1, size))
            spans[axis] = (i0, i1)
            if entry.get("select") in ("x", "y"):
                endpoints[str(entry["select"])] = (origin, step, i0, i1)
        slicer = [slice(None)] * ndim
        for axis, (i0, i1) in spans.items():
            slicer[axis] = slice(i0, i1)
        crop = block[tuple(slicer)]
        # The crop's (R,P) validity: slice the point axis by its own span iff the point axis was cropped
        # (a 1-D point-axis ROI shrinks P), else the source mask stands (image / data-axis crop keeps P).
        frame_valid = input_valid[:, slice(*spans[1])] if 1 in spans else input_valid
        reduce_axes = tuple(sorted(spans))
        reduced = reducer(np.asarray(crop, dtype=np.float64), axis=reduce_axes)
        region = None
        if "x" in endpoints and "y" in endpoints:
            ox, sx, ix0, ix1 = endpoints["x"]
            oy, sy, iy0, iy1 = endpoints["y"]
            region = [int(round(ox + ix0 * sx)), int(round(ox + ix1 * sx)),
                      int(round(oy + iy0 * sy)), int(round(oy + iy1 * sy))]
        return crop, frame_valid, reduced, reduce_axes, region

    def _bare_output_specs(self) -> tuple[SignalSpec, ...]:
        schema = self._source_schema
        repeat_capacity = schema.repeat_capacity if schema is not None else 1
        return (
            SignalSpec("roi_frame", "ROI region", "counts",
                       "The selected region for every valid (R,P) cell: a native fixed-shape crop for a "
                       "contiguous axis selection (image / 1-D index), or the NaN-masked passthrough for a "
                       "value / scatter selection (distribution / site centres) -- bind to a 2d panel "
                       "(zoomed image) or a hist panel (distribution of exactly these cells).",
                       points_shape=self._roi_frame_point, data_shape=self._roi_frame_data,
                       repeat_capacity=repeat_capacity, history=IMAGE_STREAM_HISTORY),
            SignalSpec("roi_value", "ROI value", "counts",
                       "Every region reduced to one scalar per acquisition (mean/sum/max over the "
                       "selected cells).",
                       points_shape=self._roi_value_point, data_shape=(1,),
                       repeat_capacity=repeat_capacity, dtype=np.float64),
        )


# The ROI catalog entry lives in processors/analysis.py: the ONE "Analysis" spec whose ``action``
# choice dispatches to this class as its roi strategy.  This module owns only the compute.
