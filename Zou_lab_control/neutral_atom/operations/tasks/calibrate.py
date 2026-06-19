"""Readout calibration as a first-class orchestration TASK (auto-discovered).

The sitemap + per-site threshold calibration -- the notebook's
``exp.readout.sitemap()`` / ``thresholds()`` flow -- surfaced as a
:class:`~..task.TaskSpec` so the task console can run it from a panel: it streams
its template frames to a dedicated MID-RUN panel as it goes and produces a
``TrapCalibration`` an OccupancyProcessor can then consume.  ``build`` captures the
readout subsystem and routes through ``readout.calibrate_task`` (the SAME
:class:`~..logic.CalibrateReadoutTask` the loading-readout composite uses), so GUI
and notebook cannot drift, and it imports no concrete backend / reads no simulation
ground truth -- a virtual run traverses the identical contract path a real run does.
"""

from __future__ import annotations

from ..measurement import ParamDecl
from ..task import TaskSpec
from ..task_registry import task


# The calibrate-readout task's tunable parameters, declared ONCE (the single source of
# truth a GUI form and the build closure both derive from).  ``source`` picks acquire-now
# (live: camera + pulse template) vs reuse-saved (folder); ``pulse_template`` is the REAL
# imaging program loaded each pass (its 'image' window's duration set to the exposure --
# no opaque "built-in"); ``threshold_method`` is otsu vs bimodal.  The cali no longer
# chooses a READOUT method: it computes box / per-site PSF / uniform PSF and the
# OccupancyProcessor picks one.  EVERY parameter carries a REAL default (no blank fields):
# ``folder`` (a real data dir) + ``pulse_template`` (the shipped imaging program) are both
# Browse-able paths the experimenter sees up front.
DEFAULT_DATA_DIR = "calibrations"          # data + report folder (timestamped run subfolders)
# Point at the REAL shipped file under pulses/ so the GUI field shows its full project path
# (the form resolves it to <project>/pulses/imaging_template.json -- not a bare name).
DEFAULT_PULSE_TEMPLATE = "pulses/imaging_template.json"   # the shipped, inspectable imaging program

CALIBRATE_PARAMS = (
    ParamDecl("source", "source", "choice", default="live",
              choices=("live", "saved frames"),
              tooltip="WHERE the frames to calibrate from come from.  "
                      "live = acquire now (camera + pulse template) and write the calibration; "
                      "saved frames = build the calibration from raw frames already on disk in "
                      "`folder` (named img1.npy, img2.npy, ... -- e.g. a prior real run, or "
                      "na.write_virtual_run).  (To REUSE a finished calibration you do NOT "
                      "calibrate again: point the Judge-occupancy processor at its calibration.json.)"),
    ParamDecl("folder", "folder", "path", default=DEFAULT_DATA_DIR, required=True, path_mode="dir",
              base_dir=DEFAULT_DATA_DIR,
              tooltip="The ONE data directory.  source=live WRITES at the root: calibration.json "
                      "+ calibration.npz + summary.json + the (png + npz) figure pairs; if "
                      "'save frames' is on, the raw camera frames go into a clean `frames/` "
                      "sub-folder (img1.npy, ...) -- the cali root stays uncluttered.  "
                      "source=saved frames READS those raw frames (.npy OR raw camera .tif/.tiff) "
                      "from the same `frames/` sub-folder (or, for compatibility, a flat root of "
                      "img<n> files).  Re-running overwrites; use a different folder to keep a "
                      "run.  Defaults to `calibrations/`."),
    ParamDecl("save_frames", "save frames (live)", "bool", default=True,
              tooltip="When source=live, also SAVE the acquired raw frames to `folder` (img1.npy, "
                      "...) so you can re-calibrate later with source=saved frames WITHOUT "
                      "re-acquiring.  Off = only write the calibration + report, not the frames."),
    ParamDecl("pulse_template", "pulse template", "path", default=DEFAULT_PULSE_TEMPLATE,
              path_mode="file", file_filter="Pulse program (*.json);;All files (*)", base_dir="pulses",
              tooltip="The imaging pulse program to LOAD (a real PulseTableState .json from the pulse "
                      "GUI); each pass sets its exposure on the 'image' window.  Defaults to the shipped "
                      "imaging template -- Browse to load your own."),
    ParamDecl("threshold_method", "threshold", "choice", default="otsu", choices=("otsu", "bimodal"),
              tooltip="otsu = single split; bimodal = dark/bright Gaussian-core fit per site.  (The "
                      "READOUT method box / per-site PSF / uniform PSF is chosen on the OccupancyProcessor "
                      "-- the cali computes all of them.)"),
    ParamDecl("sitemap_exposure", "sitemap exposure", "float", default=0.05, unit="s", lo=0.0, hi=10.0,
              tooltip="LONGER imaging duration for the site-finding pass (more photons -> cleaner "
                      "centroids/PSF); set on the template's 'image' window for this pass."),
    ParamDecl("readout_exposure", "readout exposure", "float", default=0.02, unit="s", lo=0.0, hi=10.0,
              tooltip="ACTUAL readout duration for the threshold pass (thresholds learnt under real "
                      "readout conditions); set on the template's 'image' window for this pass."),
    ParamDecl("calibration_frames", "reference frames", "int", default=30, lo=1, hi=4000,
              tooltip="Reference (long-exposure) frames averaged into the site-finding template."),
    ParamDecl("threshold_frames", "readout frames", "int", default=100, lo=2, hi=20000,
              tooltip="Short-readout frames whose per-site count distribution sets the thresholds."),
    ParamDecl("roi_radius", "ROI radius", "int", default=1, lo=1, hi=64,
              tooltip="Per-site square ROI half-width in pixels (box counting / detection geometry)."),
)


@task(order=10)
def calibrate_readout(readout) -> TaskSpec:
    """The readout-calibration task (sitemap + per-site thresholds).

    Its tunable parameters (source / folder / save_frames / pulse_template / threshold /
    frame counts) are declared in :data:`CALIBRATE_PARAMS` -- EVERY one with a real default
    (no blank fields): one ``folder`` (default ``calibrations``) is the data + report
    directory (everything lands there, no hidden sub-folder), and ``source`` decides
    live-acquire vs saved-frames (there is NO "saved calibration" source: reusing a finished
    calibration just loads its calibration.json in the Judge-occupancy processor).  They are
    threaded into the built :class:`~..logic.CalibrateReadoutTask`; mid-run it streams the
    template frame to its dedicated panel under the ``cal_`` namespace (``cal_frame``)."""

    def build(hub, *, prefix: str = "cal_", **param_values):
        return readout.calibrate_task(hub, prefix=prefix, **param_values)

    return TaskSpec(name="Calibrate readout", build=build, params=CALIBRATE_PARAMS,
                    mid_run_key="frame", default_kind="2d", prefix="cal_")
