"""Per-frame occupancy judgement -- the REACTIVE live-readout "func" node.

Given a calibration (site centers + per-site thresholds [+ PSF weights]) this consumes
each camera ``frame`` signal and republishes per-site occupancy / counts + the cumulative
loading fraction, through the SAME ``calibration.detect`` contract the notebook / real
readout uses (it re-implements no detection math).  Calibration ownership is explicit:
``session`` means the current in-memory calibration; ``file`` means the named,
versioned artifact.  Neither mode falls back to the other.  The file field defaults
to ``calibrations/calibration.json`` for a deliberate file-backed flow.

Reactive: it runs beside the camera measurement that publishes ``frame``, emitting only
when a new frame arrives.  Its output is a PROCESSOR signal on the hub (virtual==real:
only the camera frames differ; no simulation ground truth is read).
"""

from __future__ import annotations

from Zou_lab_control._paths import CALIBRATION_DIR, DEFAULT_CALIBRATION_FILE
from ..calibration import ALL_READOUT_METHODS
from ...core.params import ParamDecl
from ..logic import FRAME_0, OccupancyProcessor
from ..processor import ProcessorSpec
from ..processor_registry import processor

# The canonical calibration file the Calibrate-readout task writes (its latest result), and the
# detector's default input -- so calibrate-then-judge wires up with no path typed, while the file in
# use is always explicitly named (never a blank "current" mystery).  Single-sourced in _paths
# (DEFAULT_CALIBRATION_FILE under _output/calibrations/) so the WRITE side (calibrate task) and this
# READ side cannot drift.

# User-facing readout-method names -> the calibration's method keys.  ONE calibration
# carries all methods (box / per-site PSF / uniform PSF); the READOUT method is chosen
# HERE, at the processor, not when the calibration was made (cali once, read many ways).
# The human LABELS are a GUI concern (hand-written); the method KEYS they map to are the
# calibration layer's -- guarded below against the ONE allowlist (ALL_READOUT_METHODS),
# ORDER INCLUDED (dict order is the dropdown order), so a method added to the cali either
# appears in this dropdown or fails at import, never drifts silently.
METHOD_LABELS = {"box": "box", "per-site PSF": "psf", "uniform PSF": "uniform_psf"}
if tuple(METHOD_LABELS.values()) != tuple(ALL_READOUT_METHODS):
    raise RuntimeError(
        f"judge_occupancy METHOD_LABELS maps {tuple(METHOD_LABELS.values())} but the one "
        f"readout-method allowlist (calibration.ALL_READOUT_METHODS) is {ALL_READOUT_METHODS} -- "
        "give every method a label here, in the same order.")


@processor(order=5)   # occupancy is the primary live-readout processor
def judge_occupancy(readout) -> ProcessorSpec:
    """Judge per-site occupancy from each live ``frame`` BLOCK using a loaded calibration.

    Judges every valid physical ``(R,P)`` cell and preserves the canonical
    ``(R,P,*data_shape)`` layout.  ``occupied``/``counts`` declare ``data_shape=(N,)``,
    ``rate`` declares ``(1,)``, and ``frame_judged`` retains image ``(H,W)``.  Static calibration geometry is
    canonical too: ``centers`` is ``(1,1,N,2)`` and ``thresholds`` is ``(1,1,N)``.  Logical
    multi-dimensional point geometry remains in each signal's ``SignalSchema.point_shape``;
    only the physical P axis is flattened.  The default sites view is coloured by occupancy;
    ``repeat_mode=average`` displays per-site loading probability."""

    params = (
        ParamDecl("calibration_origin", "Calibration source", "choice", default="session",
                  choices=("session", "file"),
                  tooltip="session = use exactly the experiment's current calibration; "
                          "file = load exactly the named calibration artifact. No fallback."),
        ParamDecl("calibration", "Calibration file", "path", default=DEFAULT_CALIBRATION_FILE,
                  path_mode="file", file_filter="Calibration (*.json *.npz);;All files (*)",
                  base_dir=CALIBRATION_DIR, required=True,
                  tooltip="The calibration file the detector LOADS (.json/.npz: site centers + "
                          "per-site thresholds [+ PSF kernels]).  Defaults to the canonical file "
                          "the Calibrate-readout task writes (calibrations/calibration.json), so "
                          "Select Calibration source = file to use it; it must exist and match "
                          "the camera frame contract."),
        ParamDecl("source", "Frame source", "signal_expr",
                  default={"inputs": [FRAME_0], "source": "value = signal"},
                  tooltip="The camera frame to judge: pick one emCCD event's signal (`frame_0`, "
                          "`frame_1`, … one per trigger of the cycle) and optionally combine via "
                          "value = ... (default = `frame_0`, the cycle's first emCCD event; the value "
                          "must be ONE (H×W) frame -- e.g. `value = (signal[0]+signal[1])/2` averages two)."),
        ParamDecl("method", "Readout method", "choice", default="box",
                  choices=tuple(METHOD_LABELS),
                  tooltip="How to turn each frame into per-site signal: box = square ROI; "
                          "per-site PSF = one matched filter per site; uniform PSF = one shared "
                          "kernel.  The calibration must carry this method (the Calibrate task "
                          "computes all of them)."),
    )

    def make_node(hub, *, prefix: str = "", **values):
        # Reactive node reuses the real readout pipeline (calibration.detect); the console never
        # re-implements detection.  Source selection is tagged and exhaustive.
        from ...core.calibration import TrapCalibration
        from Zou_lab_control._paths import resolve_under_project

        cal_path = resolve_under_project(values.get("calibration", "") or DEFAULT_CALIBRATION_FILE)

        def _load_calibration():
            if not cal_path.is_file():
                raise FileNotFoundError(
                    f"Judge occupancy calibration does not exist: {cal_path}. "
                    "Run Calibrate readout or select an existing calibration file.")
            try:
                return TrapCalibration.load(cal_path)
            except Exception as exc:
                raise ValueError(
                    f"Judge occupancy: cannot load calibration file {cal_path.name}: {exc}. "
                    "Re-run Calibrate readout, or pick a current-version calibration.") from exc

        origin = str(values.get("calibration_origin", "session"))
        if origin == "session":
            calibration = readout.current
            if calibration is None:
                raise ValueError(
                    "Judge occupancy selected the session calibration, but the session has none. "
                    "Run Calibrate readout or select Calibration source = file.")
        elif origin == "file":
            calibration = _load_calibration()
        else:
            raise ValueError(f"unknown calibration source {origin!r}; expected 'session' or 'file'.")
        try:
            grid = readout._session.resolve_grid_shape(None)
        except Exception:
            grid = None
        method = METHOD_LABELS.get(str(values.get("method", "box")), "box")
        # ``source`` is a signal_expr value ({"inputs": [...], "source": "value = ..."}) -- the
        # same universal multi-source picker every source field uses; the node builds the shared
        # SignalExpr and judges the resulting (H×W) frame.  A mismatch propagates as a node error;
        # substituting another calibration would make the UI label disagree with the math used.
        return OccupancyProcessor(
            hub, calibration=calibration,
            source_expr=values.get("source"),
            method=method, grid_shape=grid, prefix=prefix)

    return ProcessorSpec(
        name="Judge occupancy",
        params=params,
        make_node=make_node,
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
