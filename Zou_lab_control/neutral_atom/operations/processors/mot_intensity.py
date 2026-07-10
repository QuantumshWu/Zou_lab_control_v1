"""MOT fluorescence intensity -- the reactive monitor-camera "func" node.

Consumes a MONITOR-camera frame signal and republishes the background-subtracted MOT spot
intensity as one scalar per frame: the mean count inside a circular ROI minus the mean of the
surrounding annulus (local background), so drifts in ambient light / camera offset cancel.
This is the live y-signal for a coil-field pulse-scan ("sweep the coils, watch the MOT") and
the number the Optimize-MOT-field task maximises -- BOTH consume the same
:func:`mot_roi_intensity` primitive, so the live view and the optimiser can never disagree.

Reactive and decoupled exactly like Judge occupancy: it runs beside whatever camera
measurement publishes the frame (virtual == real: only the frames differ).
"""

from __future__ import annotations

import numpy as np

# FRAME_0 -- the ONE bare default frame name (derived in logic from camera_frame_keys):
# a single camera measurement instance publishes the BARE ``frame_0`` (the signal
# namespace is the INSTANCE; a device identity is never baked into a name), so a consumer
# default must never assume a producer prefix.  In the documented single-monitor-camera
# flow this makes the processor work out of the box; a second camera instance's slugged
# frame name is picked in the form.
from ...core.params import ParamDecl
from ..logic import FRAME_0, Processor, SignalExpr, SignalSpec, DEFAULT_SOURCE
from ...core.signal_tensor import SignalTensor
from ..processor import ProcessorSpec
from ..processor_registry import processor
from ..signal_expr import hub_namespace


def mot_roi_intensity(frame, cx: float, cy: float, radius: float) -> float:
    """The ONE MOT-intensity rule: mean counts inside the circular ROI at ``(cx, cy)`` minus the
    mean of the surrounding annulus (``radius``..``2*radius`` -- the local background), so the
    number reads the MOT fluorescence itself, not the ambient/offset level.  Shared by the live
    processor and the Optimize-MOT-field task (single source)."""
    a = np.asarray(frame, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"mot_roi_intensity takes one (H, W) frame; got shape {a.shape}.")
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2
    disc = r2 <= float(radius) ** 2
    ring = (r2 > float(radius) ** 2) & (r2 <= (2.0 * float(radius)) ** 2)
    if not disc.any():
        raise ValueError(f"the MOT ROI (cx={cx}, cy={cy}, r={radius}) lies outside the {h}x{w} frame.")
    background = float(np.nanmean(a[ring])) if ring.any() else 0.0
    return float(np.nanmean(a[disc]) - background)


class MotIntensityProcessor(Processor):
    """Per-cell MOT intensity from a canonical monitor-camera tensor.

    The transform maps every valid ``(R,P)`` image cell to one scalar and
    therefore publishes ``(R,P,1)`` without collapsing repeat or point axes."""

    provides = ("mot_intensity",)

    def __init__(self, hub, *, source_expr=None, roi_cx: float = 0.0, roi_cy: float = 0.0,
                 roi_radius: float = 8.0, prefix: str = ""):
        expr = source_expr if isinstance(source_expr, SignalExpr) else SignalExpr.from_value(source_expr)
        if not expr.inputs:
            expr = SignalExpr([FRAME_0], DEFAULT_SOURCE)
        super().__init__(hub, consumes=tuple(expr.inputs), prefix=prefix)
        self.source_expr = expr
        # ROI centre: 0 = the frame centre (the virtual MOT spot sits there; a real setup types
        # the spot's pixel position or leaves 0 for a centred view).
        self.roi_cx = float(roi_cx)
        self.roi_cy = float(roi_cy)
        self.roi_radius = float(roi_radius)
        self._point_shape = (1,)
        self._repeat_capacity = 1

    def transform(self, inputs: dict[str, object]) -> dict[str, object]:
        # Coherent inputs + the SHARED expression helpers (np/numpy/math/history/...): the one
        # hub_namespace builder, so this source field honours the same SIGNAL_EXPR_HELP contract
        # every other source (panel / pulse-scan y / occupancy) does.
        frames = np.asarray(
            self.source_expr.evaluate(hub_namespace(self.hub, inputs)), dtype=float)
        if frames.ndim != 4:
            raise ValueError(
                f"MOT intensity source must preserve canonical (R,P,H,W) axes; got {frames.shape}.")
        repeats, points, h, w = (int(n) for n in frames.shape)
        valid = self.input_validity((repeats, points))
        cx = self.roi_cx if self.roi_cx > 0 else w / 2.0
        cy = self.roi_cy if self.roi_cy > 0 else h / 2.0
        intensity = np.full((repeats, points, 1), np.nan, dtype=float)
        for repeat_index in range(repeats):
            for point_index in range(points):
                if valid[repeat_index, point_index]:
                    intensity[repeat_index, point_index, 0] = mot_roi_intensity(
                        frames[repeat_index, point_index], cx, cy, self.roi_radius)
        if not np.any(valid):
            return {}
        self._point_shape = self.input_point_shape(points)
        self._repeat_capacity = self.input_repeat_capacity()
        spec = self._bare_output_specs()[0]
        return {
            "mot_intensity": SignalTensor(
                intensity, spec.to_schema(dtype=np.float64), valid=valid)
        }

    def _bare_output_specs(self) -> tuple[SignalSpec, ...]:
        return (
            SignalSpec(
                "mot_intensity", "MOT intensity", "counts",
                "background-subtracted circular-ROI intensity for each repeat/point cell",
                points_shape=self._point_shape,
                data_shape=(1,),
                dtype=np.float64,
                repeat_capacity=self._repeat_capacity,
            ),
        )


@processor(order=20)
def mot_intensity(readout) -> ProcessorSpec:
    """MOT spot intensity from the monitor camera -- the live y for a coil-field scan."""

    params = (
        ParamDecl("source", "Frame source", "signal_expr",
                  default={"inputs": [FRAME_0], "source": "value = signal"},
                  tooltip="The monitor-camera frame to measure (pick a canonical "
                          "(R,P,*data_shape) signal with data_shape=(H,W))."),
        ParamDecl("roi_cx", "ROI centre x (px)", "float", default=0.0,
                  tooltip="MOT spot centre, x pixel (0 = the frame centre)."),
        ParamDecl("roi_cy", "ROI centre y (px)", "float", default=0.0,
                  tooltip="MOT spot centre, y pixel (0 = the frame centre)."),
        ParamDecl("roi_radius", "ROI radius (px)", "float", default=8.0,
                  tooltip="Circular ROI radius; the 1x..2x annulus around it is the local "
                          "background that gets subtracted."),
    )

    def make_node(hub, *, prefix: str = "", **values):
        return MotIntensityProcessor(
            hub, source_expr=values.get("source"),
            roi_cx=float(values.get("roi_cx", 0.0) or 0.0),
            roi_cy=float(values.get("roi_cy", 0.0) or 0.0),
            roi_radius=float(values.get("roi_radius", 8.0) or 8.0),
            prefix=prefix)

    return ProcessorSpec(
        name="MOT intensity",
        params=params,
        make_node=make_node,
        # SINGLE SOURCE: the published key lives once on the node class (its ``provides``).
        result_keys=MotIntensityProcessor.provides,
        default_kind="monitor",          # a scalar over time reads as a rolling trace
    )
