"""Readout calibration as a first-class orchestration TASK (auto-discovered).

The sitemap + per-site threshold calibration -- the notebook's
``exp.readout.sitemap()`` / ``thresholds()`` flow -- surfaced as a
:class:`~..task.TaskSpec` so the task console can run it from a panel: it streams
its template frames to a dedicated MID-RUN panel as it goes and produces a
``TrapCalibration`` a DetectProcessor can then consume.  ``build`` captures the
readout subsystem and routes through ``readout.calibrate_task`` (the SAME
:class:`~..feeds.CalibrateReadoutTask` the loading-readout composite uses), so GUI
and notebook cannot drift, and it imports no concrete backend / reads no simulation
ground truth -- a virtual run traverses the identical contract path a real run does.
"""

from __future__ import annotations

from ..task import TaskSpec
from ..task_registry import task


@task(order=10)
def calibrate_readout(readout) -> TaskSpec:
    """The readout-calibration task (sitemap + per-site thresholds).

    Its tunable parameters (grid / exposure / roi_radius / frame counts / method)
    and its Run come from the built :class:`~..feeds.CalibrateReadoutTask` itself;
    mid-run it streams the template frame to its dedicated panel under the ``cal_``
    namespace (``cal_frame``)."""

    def build(hub, *, prefix: str = "cal_"):
        return readout.calibrate_task(hub, prefix=prefix)

    return TaskSpec(name="Calibrate readout", build=build,
                    mid_run_key="frame", default_kind="2d", prefix="cal_")
