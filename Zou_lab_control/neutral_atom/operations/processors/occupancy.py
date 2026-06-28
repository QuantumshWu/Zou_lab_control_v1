"""Per-frame occupancy judgement -- the REACTIVE live-readout "func" node.

Given a calibration (site centers + per-site thresholds [+ PSF weights]) this consumes
each camera ``frame`` signal and republishes per-site occupancy / counts + the cumulative
loading fraction, through the SAME ``calibration.detect`` contract the notebook / real
readout uses (it re-implements no detection math).  The calibration is an EXPLICIT, always-
named FILE (a Calibrate-readout task's artifact) -- the field defaults to the canonical
``calibrations/calibration.json`` the Calibrate task writes, so a calibrate-then-judge flow
needs no path typed, yet the file in use is always shown (never a blank mystery, matching the
Rb87 reference's explicit-file model).  Until that file is on disk the detector falls back to
the current session calibration, so a notebook-calibrated or mid-calibration session also
judges immediately.

Reactive: it runs beside the camera measurement that publishes ``frame``, emitting only
when a new frame arrives.  Its output is a PROCESSOR signal on the hub (virtual==real:
only the camera frames differ; no simulation ground truth is read).
"""

from __future__ import annotations

from Zou_lab_control._paths import CALIBRATION_DIR, DEFAULT_CALIBRATION_FILE
from ..logic import OccupancyProcessor
from ..processor import ParamDecl, ProcessorSpec
from ..processor_registry import processor

# The canonical calibration file the Calibrate-readout task writes (its latest result), and the
# detector's default input -- so calibrate-then-judge wires up with no path typed, while the file in
# use is always explicitly named (never a blank "current" mystery).  Single-sourced in _paths
# (DEFAULT_CALIBRATION_FILE under _output/calibrations/) so the WRITE side (calibrate task) and this
# READ side cannot drift.


@processor(order=5)   # occupancy is the primary live-readout processor
def judge_occupancy(readout) -> ProcessorSpec:
    """Judge per-site occupancy from each live ``frame`` BLOCK using a loaded calibration.

    Judges EVERY repeat slice (repeat_contract='preserve', #H3q), so it publishes
    ``occupied`` and ``counts`` as CLEAN ``(repeat, n_sites)`` blocks (a leading repeat axis,
    no vestigial middle 1, #H3s-F3), ``frame_judged`` as ``(repeat, H, W)``, the scalar
    ``rate`` (this block's loading fraction, for a rolling monitor / scan y), and ``centers``
    (N, 2) / ``thresholds`` (N,) static calibration geometry.  The default view is the 'sites'
    atom map coloured by occupancy; set its ``repeat_mode=average`` to colour by the per-site
    LOADING PROBABILITY (averaging the camera's ``repeat`` shots recovers every site)."""

    # User-facing readout-method names -> the calibration's method keys.  ONE calibration
    # carries all methods (box / per-site PSF / uniform PSF); the READOUT method is chosen
    # HERE, at the processor, not when the calibration was made (cali once, read many ways).
    method_labels = {"box": "box", "per-site PSF": "psf", "uniform PSF": "uniform_psf"}

    params = (
        ParamDecl("calibration", "Calibration file", "path", default=DEFAULT_CALIBRATION_FILE,
                  path_mode="file", file_filter="Calibration (*.json *.npz);;All files (*)",
                  base_dir=CALIBRATION_DIR, required=True,
                  tooltip="The calibration file the detector LOADS (.json/.npz: site centers + "
                          "per-site thresholds [+ PSF kernels]).  Defaults to the canonical file "
                          "the Calibrate-readout task writes (calibrations/calibration.json), so "
                          "calibrate-then-judge needs no path typed; Browse to use another.  Until "
                          "this file exists the detector uses the current session calibration."),
        ParamDecl("source", "Frame source", "signal_expr",
                  default={"inputs": ["frame"], "source": "value = signal"},
                  tooltip="The camera frame to judge: pick one or more hub signals and combine via "
                          "value = ... (default = the single `frame` signal; the value must be ONE "
                          "(H×W) frame -- e.g. `value = (signal[0] + signal[1]) / 2` averages two)."),
        ParamDecl("method", "Readout method", "choice", default="box",
                  choices=tuple(method_labels),
                  tooltip="How to turn each frame into per-site signal: box = square ROI; "
                          "per-site PSF = one matched filter per site; uniform PSF = one shared "
                          "kernel.  The calibration must carry this method (the Calibrate task "
                          "computes all of them)."),
    )

    def make_node(hub, *, prefix: str = "", **values):
        # Reactive node reuses the real readout pipeline (calibration.detect); the console never
        # re-implements detection.  The calibration field is an EXPLICIT, always-named FILE (the
        # canonical calibration.json by default -- never a blank "current" mystery).  Loading is
        # lazy: prefer that file; while it is not on disk yet (a Calibrate task may be mid-run,
        # about to write it, or the session was calibrated in a notebook without saving) fall back
        # to the in-memory session calibration, and keep retrying until the named file appears.
        from ...core.calibration import TrapCalibration
        from Zou_lab_control._paths import resolve_under_project

        cal_path = resolve_under_project(values.get("calibration", "") or DEFAULT_CALIBRATION_FILE)

        def _load_calibration():
            if cal_path.is_file():
                return TrapCalibration.load(cal_path)
            return readout.current        # graceful fallback until the named file exists

        calibration = _load_calibration()
        calibration_source = None if calibration is not None else _load_calibration
        try:
            grid = readout._session._grid_shape(None)
        except Exception:
            grid = None
        method = method_labels.get(str(values.get("method", "box")), "box")
        # ``source`` is a signal_expr value ({"inputs": [...], "source": "value = ..."}) -- the
        # same universal multi-source picker every source field uses; the node builds the shared
        # SignalExpr and judges the resulting (H×W) frame.  ``session_calibration`` is the live
        # in-memory calibration (built from THIS camera's ROI, so it always matches the live
        # frame): the node falls back to it if the loaded FILE calibration's ROI does not match
        # the frame (a stale calibration.json from a different camera shape would otherwise wedge
        # the whole readout) -- the operator still sees the file's name, but a mismatched file
        # never silently freezes every panel.
        return OccupancyProcessor(
            hub, calibration=calibration, calibration_source=calibration_source,
            session_calibration=lambda: readout.current,
            source_expr=values.get("source"),
            method=method, grid_shape=grid, prefix=prefix)

    return ProcessorSpec(
        name="Judge occupancy",
        params=params,
        make_node=make_node,
        consumes=("frame",),
        # SINGLE SOURCE: the published key names live ONCE on the node class (its `provides`),
        # the spec derives them -- so the spec's result_keys and the node's published_signals
        # can never drift (#H3r-F3).
        result_keys=OccupancyProcessor.provides,
        default_kind="sites",            # per-site atom map (live frame underlay + circles)
        default_value_key="occupied",
        # The site map auto-resolves its centres + underlay from THIS producing node: centres =
        # ``centers``, underlay = ``frame_judged`` (the exact frame the occupancy was judged
        # from -> rings + image are always the same shot).  So the user picks ONE signal.  The
        # key NAMES come from the node class (single source -- never re-typed here).
        metadata={"centers_key": OccupancyProcessor.sitemap_centers_key,
                  "image_key": OccupancyProcessor.sitemap_image_key},
    )
