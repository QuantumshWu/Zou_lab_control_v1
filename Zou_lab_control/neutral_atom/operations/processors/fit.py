"""Fit center -- live 2-D Gaussian-centre fitting as a func node: FIT RESULTS ON THE HUB.

Consumes a camera frame signal and republishes the fitted centre-model parameters as
SCALAR signals every shot -- the "fit results become signals" half of the selector/fit
chain: bind ``fit_x0``/``fit_y0`` to rolling monitors to watch the MOT drift live, or use
them (or ``fit_amplitude``) as a pulse-scan y / loss signal, exactly like ``roi_value``.

The model is the ONE :func:`Zou_lab_control._readout_math.gaussian2d_center` definition the
display-side ``DataFigure.center`` fit uses (popt names A/B/R/x0/y0).  Coordinates are the
CONSUMED frame's OWN pixels (origin = the frame's corner): when the producing node declares
a spatial ``region``, a panel overlay fit reports the shifted AXIS pixels instead -- the two
differ by exactly that region origin.  The frame is stride-decimated to a
small grid before ``curve_fit`` (moment-method seeds), so a 2.3 MP stream fits in ~ms on the
node's own thread -- the GUI never runs this.

Reactive and decoupled exactly like ROI crop / Judge occupancy: it runs beside whatever
camera measurement publishes the frame (virtual == real: only the frames differ).
"""

from __future__ import annotations

import numpy as np

from Zou_lab_control._readout_math import gaussian2d_center
from ...core.params import ParamDecl
from ..logic import DEFAULT_SOURCE, FRAME_0, Processor, SignalExpr, SignalSpec
from ..processor import ProcessorSpec
from ..processor_registry import processor

#: The catalog/spec display name (single source, like ROI_SPEC_NAME).
FIT_SPEC_NAME = "Fit center"

#: Longest side of the stride-decimated grid the fit runs on.  A centre/width estimate does
#: not need full sensor resolution -- 96 px keeps curve_fit ~ms even for a 2.3 MP stream while
#: resolving the centre to a fraction of the decimation step (sub-pixel from the model fit).
_FIT_GRID_MAX = 96


def fit_frame_center(frame) -> dict[str, float] | None:
    """Fit the ONE shared centre model to ``frame`` -> {fit_x0, fit_y0, fit_amplitude,
    fit_size, fit_offset} in THIS frame's own pixel coordinates (never any upstream
    region/axis frame -- the processor only sees the array), or None when the fit fails
    (flat frame / no convergence).  Moment-method seeds + stride decimation, then
    ``scipy.optimize.curve_fit`` on :func:`gaussian2d_center`."""
    from scipy.optimize import curve_fit

    img = np.asarray(frame, dtype=float)
    if img.ndim != 2 or img.size == 0 or not np.isfinite(img).all():
        return None
    step = max(1, int(np.ceil(max(img.shape) / _FIT_GRID_MAX)))
    small = img[::step, ::step]
    ys, xs = np.mgrid[0:small.shape[0], 0:small.shape[1]]
    offset = float(np.median(small))
    weights = np.clip(small - offset, 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        return None                                   # flat frame: nothing to centre on
    x0 = float((weights * xs).sum() / total)
    y0 = float((weights * ys).sum() / total)
    size = float(np.sqrt(((weights * ((xs - x0) ** 2 + (ys - y0) ** 2)).sum() / total)) or 1.0)
    amp = float(small.max() - offset) or 1.0
    coord = (xs.ravel(), ys.ravel())
    try:
        popt, _ = curve_fit(gaussian2d_center, coord, small.ravel().astype(float),
                            p0=[amp, offset, size, x0, y0], maxfev=200)
    except (RuntimeError, ValueError):
        return None
    amplitude, offset, size, x0, y0 = (float(v) for v in popt)
    return {"fit_x0": x0 * step, "fit_y0": y0 * step, "fit_amplitude": amplitude,
            "fit_size": abs(size) * step, "fit_offset": offset}


class FitProcessor(Processor):
    """Per-shot 2-D centre fit of a consumed frame -> fitted params as hub scalars."""

    node_label = "fit"
    provides = ("fit_x0", "fit_y0", "fit_amplitude", "fit_size", "fit_offset")

    def __init__(self, hub, *, source_expr=None, prefix: str = ""):
        expr = source_expr if isinstance(source_expr, SignalExpr) else SignalExpr.from_value(source_expr)
        if not expr.inputs:
            expr = SignalExpr([FRAME_0], DEFAULT_SOURCE)
        super().__init__(hub, consumes=tuple(expr.inputs), prefix=prefix)
        self.source_expr = expr

    def transform(self, inputs: dict[str, object]) -> dict[str, object]:
        from ..signal_expr import hub_namespace
        value = self.source_expr.evaluate(hub_namespace(self.hub, inputs))
        block = np.asarray(value)
        # Accept the camera's uniform (repeat, 1, H, W) block OR one bare (H, W) frame; the
        # NEWEST slice is what a live tracker wants.
        if block.ndim == 2:
            frame = block
        elif block.ndim == 4 and block.shape[1] == 1:
            frame = block[-1, 0]
        elif block.ndim == 3:
            frame = block[-1]
        else:
            raise ValueError(f"Fit center expects (H, W) or (repeat, 1, H, W); got shape {block.shape}.")
        result = fit_frame_center(frame)
        return dict(result) if result else {}         # a failed fit is a clean no-op shot

    def _bare_output_specs(self) -> tuple[SignalSpec, ...]:
        return (
            SignalSpec("fit_x0", "fit x0", "px",
                       "Fitted Gaussian centre x (frame pixels) -- a rolling monitor of the "
                       "cloud/spot position, or a scan loss signal."),
            SignalSpec("fit_y0", "fit y0", "px", "Fitted Gaussian centre y (frame pixels)."),
            SignalSpec("fit_amplitude", "fit amplitude", "counts",
                       "Fitted peak height above the offset -- a brightness/loading signal."),
            SignalSpec("fit_size", "fit size", "px", "Fitted 1/e radius R of the Gaussian."),
            SignalSpec("fit_offset", "fit offset", "counts", "Fitted background level B."),
        )


@processor(order=26)
def fit_center(readout) -> ProcessorSpec:
    """Fit the shared 2-D Gaussian centre model to a live frame each shot."""

    params = (
        ParamDecl("source", "Frame source", "signal_expr",
                  default={"inputs": [FRAME_0], "source": "value = signal"},
                  tooltip="The camera frame to fit (ONE (H, W) frame or the camera's "
                          "(repeat, 1, H, W) block; the newest slice is fitted)."),
    )

    def make_node(hub, *, prefix: str = "", **values):
        return FitProcessor(hub, source_expr=values.get("source"), prefix=prefix)

    return ProcessorSpec(
        name=FIT_SPEC_NAME,
        params=params,
        make_node=make_node,
        result_keys=FitProcessor.provides,
        default_kind="monitor",          # the natural first view: watch the centre drift
        default_value_key="fit_x0",
    )
