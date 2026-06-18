"""Per-frame occupancy judgement -- the REACTIVE live-readout "func" node.

Given a calibration (site centers + per-site thresholds [+ PSF weights]) this consumes
each camera ``frame`` signal and republishes per-site occupancy / counts + a rolling
loading rate, through the SAME ``calibration.detect`` contract the notebook / real
readout uses (it re-implements no detection math).  The calibration is LOADED from a
saved file (a Calibrate-readout task's artifact) or, when blank, taken from the current
session calibration -- so this node is decoupled from HOW the calibration was produced.

Reactive: it runs beside the camera measurement that publishes ``frame``, emitting only
when a new frame arrives.  Its output is a PROCESSOR signal on the hub (virtual==real:
only the camera frames differ; no simulation ground truth is read).
"""

from __future__ import annotations

from ..logic import OccupancyProcessor
from ..processor import ParamDecl, ProcessorSpec
from ..processor_registry import processor


@processor(order=5)   # occupancy is the primary live-readout processor
def judge_occupancy(readout) -> ProcessorSpec:
    """Judge per-site occupancy from each live ``frame`` using a loaded calibration.

    Publishes ``occupied`` (N,), ``counts`` (N,), ``rate`` (scalar EMA loading rate),
    ``rate_sites`` (N,), ``rate_grid`` (grid map), ``centers`` (N, 2) and ``thresholds``
    (N,); the default view is the per-site 'sites' atom map coloured by occupancy."""

    # User-facing readout-method names -> the calibration's method keys.  ONE calibration
    # carries all methods (box / per-site PSF / uniform PSF); the READOUT method is chosen
    # HERE, at the processor, not when the calibration was made (cali once, read many ways).
    method_labels = {"box": "box", "per-site PSF": "psf", "uniform PSF": "uniform_psf"}

    params = (
        ParamDecl("calibration", "Calibration file", "path", default="", path_mode="file",
                  file_filter="Calibration (*.json *.npz);;All files (*)", base_dir="calibrations",
                  tooltip="A saved calibration (.npz/.json: site centers + per-site thresholds "
                          "[+ PSF kernels]) -- e.g. a Calibrate-readout task's saved artifact.  "
                          "Blank = use the CURRENT session calibration."),
        ParamDecl("source", "Frame signal", "signal", default="frame",
                  tooltip="Hub signal carrying each camera frame to judge (a processor's input, "
                          "picked like a plot's signal)."),
        ParamDecl("method", "Readout method", "choice", default="box",
                  choices=tuple(method_labels),
                  tooltip="How to turn each frame into per-site signal: box = square ROI; "
                          "per-site PSF = one matched filter per site; uniform PSF = one shared "
                          "kernel.  The calibration must carry this method (the Calibrate task "
                          "computes all of them)."),
        ParamDecl("ema", "Rate smoothing (EMA)", "float", default=0.05, lo=0.0, hi=1.0,
                  tooltip="Exponential-moving-average weight for the rolling loading rate "
                          "(0 = freeze the first value, 1 = no smoothing)."),
    )

    def make_node(hub, *, prefix: str = "", **values):
        # Reactive node reuses the real readout pipeline (calibration.detect); the
        # calibration is LOADED here (saved file) or DEFERRED to the session calibration
        # -- the console never re-implements detection (single readout contract).
        from ...core.calibration import TrapCalibration

        cal_path = str(values.get("calibration", "")).strip()
        if cal_path:
            calibration = TrapCalibration.load(cal_path)
            calibration_source = None
        else:
            calibration = None
            # lazy: pick up the session calibration once it exists (a Calibrate-readout
            # task may still be running) -- the detector no-ops until then.
            calibration_source = lambda: readout.current
        try:
            grid = readout._session._grid_shape(None)
        except Exception:
            grid = None
        method = method_labels.get(str(values.get("method", "box")), "box")
        return OccupancyProcessor(
            hub, calibration=calibration, calibration_source=calibration_source,
            source=str(values.get("source", "frame")), ema=float(values.get("ema", 0.05)),
            method=method, grid_shape=grid, prefix=prefix)

    return ProcessorSpec(
        name="Judge occupancy",
        params=params,
        make_node=make_node,
        consumes=("frame",),
        result_keys=("occupied", "counts", "rate", "rate_sites", "rate_grid", "centers",
                     "thresholds", "frame_judged"),
        default_kind="sites",            # per-site atom map (live frame underlay + circles)
        default_value_key="occupied",
        # The site map auto-resolves its centres + underlay from THIS producing node: centres =
        # ``centers``, underlay = ``frame_judged`` (the exact frame the occupancy was judged
        # from -> rings + image are always the same shot).  So the user picks ONE signal.  The
        # key NAMES come from the node class (single source -- never re-typed here).
        metadata={"centers_key": OccupancyProcessor.sitemap_centers_key,
                  "image_key": OccupancyProcessor.sitemap_image_key},
    )
