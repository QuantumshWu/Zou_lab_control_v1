"""Readout calibration as a first-class orchestration TASK (auto-discovered).

The sitemap + per-site threshold calibration -- the notebook's
``exp.readout.sitemap()`` / ``thresholds()`` flow -- surfaced as a
:class:`~..task.TaskSpec` so the task console can run it from a panel: it streams
its template frames to a dedicated MID-RUN panel as it goes and produces a
``TrapCalibration`` a DetectProcessor can then consume.  ``build`` captures the
readout subsystem and routes through ``readout.calibrate_task`` (the SAME
:class:`~..logic.CalibrateReadoutTask` the loading-readout composite uses), so GUI
and notebook cannot drift, and it imports no concrete backend / reads no simulation
ground truth -- a virtual run traverses the identical contract path a real run does.
"""

from __future__ import annotations

from ..measurement import ParamDecl
from ..task import TaskSpec
from ..task_registry import task


# The calibrate-readout task's tunable parameters, declared ONCE (the single
# source of truth a GUI form and the build closure both derive from).  ``source``
# picks acquire-now (live: camera + pulse) vs reuse-saved (folder); ``mode`` is the
# sitemap readout model (box / per-site PSF / uniform PSF -> one shared kernel);
# ``threshold_method`` is otsu vs bimodal; ``save_path`` / ``load_path`` persist or
# restore the resulting centers+thresholds (TrapCalibration.save/load).
CALIBRATE_PARAMS = (
    ParamDecl("source", "source", "choice", default="live", choices=("live", "folder"),
              tooltip="live = acquire now (camera + pulse); folder = use saved images from a folder."),
    ParamDecl("data_dir", "data folder", "text", default="",
              tooltip="Folder of saved frames (used when source = folder)."),
    ParamDecl("mode", "mode", "choice", default="box",
              choices=("box", "per-site PSF", "uniform PSF"),
              tooltip="box = square ROI; per-site PSF = one matched filter per site; "
                      "uniform PSF = one shared kernel for all sites."),
    ParamDecl("threshold_method", "threshold", "choice", default="otsu", choices=("otsu", "bimodal"),
              tooltip="otsu = single split; bimodal = dark/bright Gaussian-core fit per site."),
    ParamDecl("sitemap_exposure", "sitemap exposure", "float", default=0.05, unit="s", lo=0.0, hi=10.0,
              tooltip="LONGER readout duration for the site + PSF calibration pass (more photons "
                      "-> cleaner centroids/PSF).  Used when no sitemap pulse is given."),
    ParamDecl("sitemap_pulse", "sitemap pulse", "text", default="",
              tooltip="Optional saved pulse program (a PulseTableState .json from the pulse GUI) for "
                      "the SITE/PSF acquisition (the long readout).  Blank = default imaging "
                      "sequence at the sitemap exposure."),
    ParamDecl("readout_exposure", "readout exposure", "float", default=0.02, unit="s", lo=0.0, hi=10.0,
              tooltip="ACTUAL readout duration for the threshold pass (thresholds are learnt under "
                      "the real readout conditions).  Used when no readout pulse is given."),
    ParamDecl("readout_pulse", "readout pulse", "text", default="",
              tooltip="Optional saved pulse program for the ACTUAL-READOUT acquisition (thresholds). "
                      "Blank = default imaging sequence at the readout exposure."),
    ParamDecl("calibration_frames", "sitemap frames", "int", default=4, lo=1, hi=1000,
              tooltip="Frames averaged into the all-sites sitemap template."),
    ParamDecl("threshold_frames", "threshold frames", "int", default=24, lo=2, hi=10000,
              tooltip="Frames used to learn the per-site thresholds."),
    ParamDecl("roi_radius", "ROI radius", "int", default=1, lo=1, hi=64,
              tooltip="Per-site square ROI half-width in pixels (box counting / detection geometry)."),
    ParamDecl("save_path", "save to", "text", default="",
              tooltip="Save the calibration (centers + thresholds) to this .npz/.json path."),
    ParamDecl("load_path", "load from", "text", default="",
              tooltip="Load an existing calibration from this path INSTEAD of acquiring."),
)


@task(order=10)
def calibrate_readout(readout) -> TaskSpec:
    """The readout-calibration task (sitemap + per-site thresholds).

    Its tunable parameters (source / mode / threshold / frame counts / save / load)
    are declared in :data:`CALIBRATE_PARAMS` and threaded into the built
    :class:`~..logic.CalibrateReadoutTask`; mid-run it streams the template frame to
    its dedicated panel under the ``cal_`` namespace (``cal_frame``)."""

    def build(hub, *, prefix: str = "cal_", **param_values):
        return readout.calibrate_task(hub, prefix=prefix, **param_values)

    return TaskSpec(name="Calibrate readout", build=build, params=CALIBRATE_PARAMS,
                    mid_run_key="frame", default_kind="2d", prefix="cal_")
