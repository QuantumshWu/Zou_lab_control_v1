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

from ..processor import ParamDecl, ProcessorSpec
from ..processor_registry import processor


@processor(order=5)   # occupancy is the primary live-readout processor
def judge_occupancy(readout) -> ProcessorSpec:
    """Judge per-site occupancy from each live ``frame`` using a loaded calibration.

    Publishes ``occupied`` (N,), ``counts`` (N,), ``rate`` (scalar EMA loading rate),
    ``rate_sites`` (N,), ``rate_grid`` (grid map), ``centers`` (N, 2) and ``thresholds``
    (N,); the default view is the per-site 'sites' atom map coloured by occupancy."""

    params = (
        ParamDecl("calibration", "Calibration file", "text", default="",
                  tooltip="Path to a saved calibration (.npz/.json: site centers + per-site "
                          "thresholds [+ PSF weights]) -- e.g. a Calibrate-readout task's saved "
                          "artifact.  Blank = use the CURRENT session calibration."),
        ParamDecl("source", "Frame signal", "text", default="frame",
                  tooltip="Hub signal carrying each camera frame to judge."),
        ParamDecl("ema", "Rate smoothing (EMA)", "float", default=0.05, lo=0.0, hi=1.0,
                  tooltip="Exponential-moving-average weight for the rolling loading rate "
                          "(0 = freeze the first value, 1 = no smoothing)."),
    )

    def make_node(hub, *, prefix: str = "", **values):
        # Reactive node reuses the real readout pipeline (calibration.detect); the
        # calibration is LOADED here (saved file) or DEFERRED to the session calibration
        # -- the console never re-implements detection (single readout contract).
        from ..logic import DetectProcessor
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
        return DetectProcessor(
            hub, calibration=calibration, calibration_source=calibration_source,
            source=str(values.get("source", "frame")), ema=float(values.get("ema", 0.05)),
            grid_shape=grid, prefix=prefix)

    return ProcessorSpec(
        name="Judge occupancy",
        params=params,
        make_node=make_node,
        consumes=("frame",),
        result_keys=("occupied", "counts", "rate", "rate_sites", "rate_grid", "centers", "thresholds"),
        default_kind="sites",            # per-site atom map (live frame underlay + circles)
        default_value_key="occupied",
        metadata={"centers_key": "centers", "image_key": "frame"},
    )
