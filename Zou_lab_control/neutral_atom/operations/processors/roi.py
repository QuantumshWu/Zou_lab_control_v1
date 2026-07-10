"""ROI crop + reduce -- the generic "look at a piece of the frame" func node.

Consumes a camera frame signal and republishes two DERIVED signals per shot:

* ``roi_frame`` -- the rectangular crop itself (native dtype), a live 2-D view of just the
  region: bind it to a ``2d`` panel for a zoomed live image, or to a ``hist`` panel for the
  distribution of exactly those pixels;
* ``roi_value`` -- ONE scalar per acquisition (the crop reduced by ``sum`` / ``mean`` / ``max``):
  bind it to a rolling ``monitor``, or use it as a pulse-scan y -- e.g. the MOT loss signal for a
  coil scan -- with no bespoke task needed.

This is the building block that lets an operator ASSEMBLE "select a region -> watch its
distribution / total counts" from stock parts (camera measurement -> ROI processor -> panels)
instead of needing a dedicated task.  The region is FOUR pixel endpoints in the consumed frame's
own pixel coordinates -- exactly what the 2-D panel's rectangle selector yields, so a drag on the
plot writes the region straight back (``region_to_acquisition_parameters``), and the ``roi_frame``
panel's axes show the true source pixels (``acquisition_parameters()['region']``).

Reactive and decoupled exactly like Judge occupancy / MOT intensity: it runs beside whatever
camera measurement publishes the frame (virtual == real: only the frames differ).
"""

from __future__ import annotations

import numpy as np

from ...core.params import ParamDecl
from ...core.signal_tensor import SignalTensor
from ..logic import DEFAULT_SOURCE, FRAME_0, IMAGE_STREAM_HISTORY, Processor, SignalExpr, SignalSpec
from ..processor import ProcessorSpec
from ..processor_registry import processor
from ..signal_expr import hub_namespace

#: The reduce verbs an ROI offers, in dropdown order -- ONE source for the node's dispatch, the
#: ParamDecl choices and the tests, so a verb added here appears everywhere at once.
ROI_REDUCERS = {"mean": np.mean, "sum": np.sum, "max": np.max}


class RoiProcessor(Processor):
    """Crop and reduce every valid cell of a canonical image tensor."""

    provides = ("roi_frame", "roi_value")

    def __init__(self, hub, *, source_expr=None, x_min: float = 0.0, x_max: float = 0.0,
                 y_min: float = 0.0, y_max: float = 0.0, reduce: str = "mean", prefix: str = ""):
        expr = source_expr if isinstance(source_expr, SignalExpr) else SignalExpr.from_value(source_expr)
        if not expr.inputs:
            expr = SignalExpr([FRAME_0], DEFAULT_SOURCE)
        super().__init__(hub, consumes=tuple(expr.inputs), prefix=prefix)
        self.source_expr = expr
        # Region endpoints in the CONSUMED frame's pixel coordinates (the same numbers the 2-D
        # panel's axes show).  max <= min on an axis means "to the frame edge", so the all-zero
        # default is simply the whole frame -- a fresh node works before any region is drawn.
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        if str(reduce) not in ROI_REDUCERS:
            raise ValueError(f"ROI reduce must be one of {tuple(ROI_REDUCERS)}; got {reduce!r}.")
        self.reduce = str(reduce)
        self._frame_shape: tuple[int, int] | None = None   # learned from the first frame (for region)
        self._source_schema = None
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
                self._source_schema = first
                if len(first.data_shape) == 2:
                    self._frame_shape = tuple(first.data_shape)

    # ------------------------------------------------------------------ region plumbing
    def _bounds(self, h: int, w: int) -> tuple[int, int, int, int]:
        """The clamped integer crop window ``(x0, x1, y0, y1)`` for an ``h x w`` frame -- at least
        one pixel on each axis, so a degenerate drag can never produce an empty crop."""
        x0 = int(np.clip(round(self.x_min), 0, w - 1))
        y0 = int(np.clip(round(self.y_min), 0, h - 1))
        x1 = int(np.clip(round(self.x_max), x0 + 1, w)) if self.x_max > self.x_min else w
        y1 = int(np.clip(round(self.y_max), y0 + 1, h)) if self.y_max > self.y_min else h
        return x0, x1, y0, y1

    def acquisition_parameters(self) -> dict[str, object]:
        """The editable source knobs (Edit tab) + the declared spatial ``region`` of ``roi_frame``
        (the crop window in source pixels), so its 2-D panel draws real pixel axes and a further
        selection on it round-trips through the same coordinates."""
        out: dict[str, object] = {"x_min": self.x_min, "x_max": self.x_max,
                                  "y_min": self.y_min, "y_max": self.y_max, "reduce": self.reduce}
        if self._frame_shape is not None:
            h, w = self._frame_shape
            x0, x1, y0, y1 = self._bounds(h, w)
            out["region"] = [x0, x1, y0, y1]
        return out

    def set_acquisition_parameters(self, **values) -> None:
        for key in ("x_min", "x_max", "y_min", "y_max"):
            if key in values:
                setattr(self, key, float(values[key]))
        if "reduce" in values:
            reduce = str(values["reduce"])
            if reduce not in ROI_REDUCERS:
                raise ValueError(f"ROI reduce must be one of {tuple(ROI_REDUCERS)}; got {reduce!r}.")
            self.reduce = reduce

    def region_to_acquisition_parameters(self, x_min, x_max, y_min, y_max) -> dict[str, object]:
        """A rectangle dragged on a panel showing this node's output IS the ROI -- the selector's
        four endpoints (already in the consumed frame's pixel coordinates) map 1:1 to the region
        params, so 'draw a box -> watch that box' needs no typing."""
        return region_values(x_min, x_max, y_min, y_max)

    # ------------------------------------------------------------------ transform
    def transform(self, inputs: dict[str, object]) -> dict[str, object]:
        value = self.source_expr.evaluate(hub_namespace(self.hub, inputs))
        block = np.asarray(value)
        if not self._input_tensors:
            raise ValueError("ROI requires a registered image signal with SignalSchema.")
        schemas = [tensor.schema for tensor in self._input_tensors.values()]
        schema = schemas[0]
        for other in schemas[1:]:
            schema.assert_compatible(other, check_repeat=True)
        if len(schema.data_shape) != 2:
            raise ValueError(
                f"ROI requires schema data_shape=(H,W); got {schema.data_shape}.")
        expected = (block.shape[0], schema.point_count, *schema.data_shape)
        if block.ndim != 4 or tuple(block.shape) != expected:
            raise ValueError(
                "ROI expression must preserve canonical (R,P,H,W) axes; "
                f"expected {expected}, got {block.shape}.")
        input_valid = self.input_validity(block.shape[:2])
        self._source_schema = schema
        h, w = schema.data_shape
        self._frame_shape = (int(h), int(w))
        x0, x1, y0, y1 = self._bounds(h, w)
        crop = block[..., y0:y1, x0:x1]
        reducer = ROI_REDUCERS[self.reduce]
        reduced = reducer(np.asarray(crop, dtype=float), axis=(-2, -1))[..., None]
        value_valid = input_valid & np.isfinite(reduced[..., 0])
        specs = {spec.name: spec for spec in self._bare_output_specs()}
        return {
            "roi_frame": SignalTensor(
                crop, specs["roi_frame"].to_schema(dtype=crop.dtype), valid=input_valid),
            "roi_value": SignalTensor(
                reduced, specs["roi_value"].to_schema(dtype=reduced.dtype), valid=value_valid),
        }

    def _bare_output_specs(self) -> tuple[SignalSpec, ...]:
        schema = self._source_schema
        points = tuple(schema.point_shape) if schema is not None else (1,)
        repeat_capacity = schema.repeat_capacity if schema is not None else 1
        if self._frame_shape is None:
            crop_shape = (1, 1)
        else:
            h, w = self._frame_shape
            x0, x1, y0, y1 = self._bounds(h, w)
            crop_shape = (y1 - y0, x1 - x0)
        return (
            SignalSpec("roi_frame", "ROI image", "counts",
                       "The rectangular crop for every valid (R,P) image cell (native dtype) -- a live 2-D view "
                       "of just the region; bind to a 2d panel (zoomed image) or a hist panel "
                       "(distribution of exactly these pixels).",
                       points_shape=points, data_shape=crop_shape,
                       repeat_capacity=repeat_capacity, history=IMAGE_STREAM_HISTORY),
            SignalSpec("roi_value", "ROI value", "counts",
                       "Each (R,P) crop reduced to one scalar (mean/sum/max over pixels).",
                       points_shape=points, data_shape=(1,),
                       repeat_capacity=repeat_capacity, dtype=np.float64),
        )


#: The catalog/spec display name -- the ONE string a caller that creates a ROI row
#: programmatically (the console's draw-a-box-on-a-live-panel gesture) resolves the spec by.
ROI_SPEC_NAME = "ROI crop"


def region_values(x_min, x_max, y_min, y_max) -> dict[str, float]:
    """Selector-rectangle endpoints -> the region PARAM dict ({x_min, x_max, y_min, y_max},
    sorted).  The ONE mapping shared by the live retarget (``region_to_acquisition_parameters``)
    and the create-from-a-drag seed (the console fills a fresh row's values with exactly this)."""
    return {"x_min": float(min(x_min, x_max)), "x_max": float(max(x_min, x_max)),
            "y_min": float(min(y_min, y_max)), "y_max": float(max(y_min, y_max))}


@processor(order=25)
def roi(readout) -> ProcessorSpec:
    """Crop a rectangular region of a live frame + reduce it to one scalar per shot."""

    params = (
        ParamDecl("source", "Frame source", "signal_expr",
                  default={"inputs": [FRAME_0], "source": "value = signal"},
                  tooltip="The camera frame to crop (pick the camera measurement's frame signal; "
                          "the value must be ONE (H, W) frame or the camera's (repeat, 1, H, W) "
                          "block)."),
        ParamDecl("x_min", "Region x min (px)", "float", default=0.0,
                  tooltip="Left edge of the region, source-frame pixels.  Drawing a rectangle on "
                          "a panel showing this node's output fills all four endpoints."),
        ParamDecl("x_max", "Region x max (px)", "float", default=0.0,
                  tooltip="Right edge of the region (<= x min means: to the frame edge)."),
        ParamDecl("y_min", "Region y min (px)", "float", default=0.0,
                  tooltip="Top edge of the region, source-frame pixels."),
        ParamDecl("y_max", "Region y max (px)", "float", default=0.0,
                  tooltip="Bottom edge of the region (<= y min means: to the frame edge)."),
        ParamDecl("reduce", "Reduce", "choice", default="mean", choices=tuple(ROI_REDUCERS),
                  tooltip="How roi_value collapses the region's pixels each shot: mean / sum "
                          "(total counts, e.g. a MOT loss signal) / max."),
    )

    def make_node(hub, *, prefix: str = "", **values):
        return RoiProcessor(
            hub, source_expr=values.get("source"),
            x_min=float(values.get("x_min", 0.0) or 0.0),
            x_max=float(values.get("x_max", 0.0) or 0.0),
            y_min=float(values.get("y_min", 0.0) or 0.0),
            y_max=float(values.get("y_max", 0.0) or 0.0),
            reduce=str(values.get("reduce", "mean") or "mean"),
            prefix=prefix)

    return ProcessorSpec(
        name=ROI_SPEC_NAME,
        params=params,
        make_node=make_node,
        # SINGLE SOURCE: the published keys live once on the node class (its ``provides``).
        result_keys=RoiProcessor.provides,
        default_kind="2d",               # the crop itself is the natural first view
    )
