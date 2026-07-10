"""ROI = a Selection-driven region reducer -- the generic "look at a piece of the data" func node.

Consumes a signal block and republishes two DERIVED signals per shot, GENERIC over EVERY plot kind:

* ``roi_value`` -- ONE scalar per acquisition (the selected cells reduced by ``mean`` / ``sum`` /
  ``max``): an image rectangle, a 1-D x-range, a distribution's count-range, a site-centre rectangle
  ALL collapse to this one number.  Bind it to a rolling ``monitor``, or use it as a pulse-scan y --
  e.g. the MOT loss signal for a coil scan -- with no bespoke task needed.
* ``roi_frame`` -- a live view of just the region: a fixed-shape native CROP for a contiguous axis
  selection (image data axes, or a 1-D contiguous index axis), OR -- for a VALUE / scatter selection
  that has no fixed sub-frame (a distribution's value range; scattered site centres) -- the
  passthrough block with out-of-region cells set to NaN (a STABLE-shape signal, since the
  SignalSchema contract forbids a variable-length republish).

The ROI carries a :class:`~...core.selection.Selection` + a ``reduce`` verb -- exactly as a
:class:`FitProcessor` carries a ``FitRequest`` -- so it is generic over plot kind WITHOUT any per-kind
branch of its own.  WHAT the consumed block's cells mean (image pixels, a 1-D index, a sample value,
site centres) is per-kind knowledge the FRONTEND owns (``live.region_binding``, the inverse of
``coerce_panel_value``); it ships that knowledge to this node as PLAIN DATA on the Selection's
``metadata["binding"]`` (a serializable axis-binding dict), so a drag on ANY panel kind writes the
region straight back through :meth:`set_selection`, and ``neutral_atom`` never imports ``frontend``.

Reactive and decoupled exactly like Judge occupancy / MOT intensity: it runs beside whatever
measurement publishes the block (virtual == real: only the data differ).
"""

from __future__ import annotations

import warnings

import numpy as np

from ...core.params import ParamDecl
from ...core.selection import Selection
from ...core.signal_tensor import SignalTensor
from ..logic import DEFAULT_SOURCE, FRAME_0, IMAGE_STREAM_HISTORY, Processor, SignalExpr, SignalSpec
from ..processor import ProcessorSpec
from ..processor_registry import processor
from ..signal_expr import hub_namespace

#: The reduce verbs an ROI offers, in dropdown order -- ONE source for the node's dispatch, the
#: ParamDecl choices and the tests, so a verb added here appears everywhere at once.
ROI_REDUCERS = {"mean": np.mean, "sum": np.sum, "max": np.max}
#: The NaN-aware twins used on the value-/scatter-mask path (out-of-region cells are NaN), keyed by the
#: SAME verbs so ``ROI_REDUCERS`` stays the single source of the vocabulary.
_ROI_NAN_REDUCERS = {"mean": np.nanmean, "sum": np.nansum, "max": np.nanmax}


class RoiProcessor(Processor):
    """Reduce a Selection-selected region of a canonical signal block to one scalar, and republish
    the region itself -- generic over plot kind because it carries a Selection, not pixels."""

    provides = ("roi_frame", "roi_value")

    def __init__(self, hub, *, source_expr=None, selection=None, reduce: str = "mean", prefix: str = ""):
        expr = source_expr if isinstance(source_expr, SignalExpr) else SignalExpr.from_value(source_expr)
        if not expr.inputs:
            expr = SignalExpr([FRAME_0], DEFAULT_SOURCE)
        super().__init__(hub, consumes=tuple(expr.inputs), prefix=prefix)
        self.source_expr = expr
        # The region is ONE Selection in the consumed block's own coordinate frame (the SAME contract
        # fit + DataFigure.selected_data use).  An EMPTY selection (no binding) means "the whole
        # frame" -- a fresh node reduces every valid cell before any region is drawn.
        self.set_selection(selection)
        if str(reduce) not in ROI_REDUCERS:
            raise ValueError(f"ROI reduce must be one of {tuple(ROI_REDUCERS)}; got {reduce!r}.")
        self.reduce = str(reduce)
        # Output shapes are learned from the first block (and re-learned on a retarget, which the hub
        # accepts because THIS node owns the schema -- replace=True).  Seed sensible whole-frame
        # defaults from the source schema so the output specs are valid before any frame arrives.
        self._source_schema = None
        self._roi_frame_point: tuple[int, ...] = (1,)
        self._roi_frame_data: tuple[int, ...] = (1,)
        self._roi_frame_dtype = None
        self._roi_value_point: tuple[int, ...] = (1,)
        self._region = None
        schemas = []
        for name in self.consumes:
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

    # ------------------------------------------------------------------ region (the ONE mutator)
    def set_selection(self, selection) -> None:
        """Atomically install the region as a :class:`Selection` (mirrors FitProcessor.set_fit_request):
        a plot-independent set of bounds in the consumed block's coordinate frame, whose
        ``metadata['binding']`` (supplied by the frontend) tells this node which block axes the
        selection spans and how each coordinate is computed.  ``None`` / a dict / a Selection are all
        accepted, so the console can hand it fresh from a drag OR replay it from a saved config."""
        if selection is None:
            self.selection = Selection()
        elif isinstance(selection, Selection):
            self.selection = selection
        else:
            self.selection = Selection.from_dict(selection)

    def acquisition_parameters(self) -> dict[str, object]:
        """The editable source knobs (Edit tab) -- the region as a serializable Selection + the reduce
        verb -- plus the declared spatial ``region`` of ``roi_frame`` (its source-pixel extent, for an
        image crop), so a roi_frame 2-D panel draws real pixel axes and a FURTHER drag on it round-trips
        through the same coordinates (the ROI-of-ROI case)."""
        out: dict[str, object] = {"selection": self.selection.to_dict(), "reduce": self.reduce}
        if self._region is not None:
            out["region"] = list(self._region)
        return out

    def set_acquisition_parameters(self, **values) -> None:
        if "selection" in values:
            self.set_selection(values["selection"])
        if "reduce" in values:
            reduce = str(values["reduce"])
            if reduce not in ROI_REDUCERS:
                raise ValueError(f"ROI reduce must be one of {tuple(ROI_REDUCERS)}; got {reduce!r}.")
            self.reduce = reduce

    # ------------------------------------------------------------------ transform
    def transform(self, inputs: dict[str, object]) -> dict[str, object]:
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
        self._region = region

        specs = {spec.name: spec for spec in self._bare_output_specs()}
        return {
            "roi_frame": SignalTensor(
                frame, specs["roi_frame"].to_schema(dtype=frame.dtype), valid=frame_valid),
            "roi_value": SignalTensor(
                roi_value, specs["roi_value"].to_schema(dtype=roi_value.dtype), valid=value_valid),
        }

    def _reduce_region(self, block, input_valid):
        """Apply the carried Selection to the raw ``(R,P,*data)`` block through the ONE mode-driven
        machinery -- returns ``(roi_frame, frame_valid, reduced, reduce_axes, region)``.  There is NO
        per-kind branch here: the plot-side binding names WHICH axes the selection spans (``mode`` +
        axis descriptors), so an image rectangle, a 1-D index range, a distribution value-range and a
        site-centre rectangle all resolve identically."""
        ndim = block.ndim
        binding = dict(self.selection.metadata.get("binding") or {})
        mode = str(binding.get("mode") or "axis-crop")
        reducer = ROI_REDUCERS[self.reduce]
        nan_reducer = _ROI_NAN_REDUCERS[self.reduce]

        if mode == "value-mask":
            # The coordinate of every sample cell IS its value: mask out-of-range cells to NaN and pool
            # every sample per repeat.  roi_frame is the stable-shape NaN passthrough (P unchanged).
            vb = self.selection.bounds("value")
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
            keep_cells = (self.selection.mask(coords, length=n) if coords
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
            bounds = self.selection.bounds(entry.get("select")) if entry.get("select") else None
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


#: The catalog/spec display name -- the ONE string a caller that creates a ROI row
#: programmatically (the console's draw-a-box-on-a-live-panel gesture) resolves the spec by.
ROI_SPEC_NAME = "ROI crop"


@processor(order=25)
def roi(readout) -> ProcessorSpec:
    """Reduce a Selection-selected region of a live block to one scalar per shot + republish the region."""

    params = (
        ParamDecl("source", "Signal source", "signal_expr",
                  default={"inputs": [FRAME_0], "source": "value = signal"},
                  tooltip="The block to reduce a region of (a camera frame, a 1-D scan curve, a "
                          "distribution's samples, a site-map vector).  Draw a rectangle / range on a "
                          "panel showing this node's output to set the region."),
        ParamDecl("reduce", "Reduce", "choice", default="mean", choices=tuple(ROI_REDUCERS),
                  tooltip="How roi_value collapses the selected cells each shot: mean / sum "
                          "(total counts, e.g. a MOT loss signal) / max."),
    )

    def make_node(hub, *, prefix: str = "", **values):
        return RoiProcessor(
            hub, source_expr=values.get("source"),
            selection=values.get("selection"),
            reduce=str(values.get("reduce", "mean") or "mean"),
            prefix=prefix)

    return ProcessorSpec(
        name=ROI_SPEC_NAME,
        params=params,
        make_node=make_node,
        # SINGLE SOURCE: the published keys live once on the node class (its ``provides``).
        result_keys=RoiProcessor.provides,
        default_kind="2d",               # the region itself is the natural first view
    )
