"""AOD atom rearrangement as a first-class orchestration TASK (auto-discovered).

Surfaces ``exp.rearrange.execute()`` -- image -> detect occupancy -> plan moves -> drive the AOD ->
re-image -> report fill fraction -- as a :class:`~..task.TaskSpec` so the task console runs it from a
panel: it streams the before/after frames + occupancy + progress to a dedicated MID-RUN panel and produces
the fill-fraction result.  ``build`` captures the readout subsystem and routes through
``readout.session.rearrange.task`` (the SAME :class:`~..logic.RearrangeTask` the notebook one-liner uses),
so GUI and notebook cannot drift, and it imports no concrete backend / reads no simulation ground truth --
a virtual run traverses the identical contract path a real run does (only the camera frames + the AOD
backend, virtual occupancy vs real RF chirp, differ).
"""

from __future__ import annotations

from ..measurement import ParamDecl
from ..task import TaskSpec
from ..task_registry import task


# The rearrangement task's tunable parameters, declared ONCE (single source for the GUI form + the build
# closure).  EVERY one has a real default (no blank fields): how many central sites to assemble, the
# defect-free layout, the readout exposure each image is taken at, and the per-move transit survival
# (-1 = use the AOD device's own default).
REARRANGE_PARAMS = (
    ParamDecl("target_count", "target sites", "int", default=0, lo=0, hi=100000,
              tooltip="How many of the most-central tweezer sites to assemble into the defect-free "
                      "block.  0 = half the array (what a ~50 %-loaded array can typically fill)."),
    ParamDecl("target_layout", "layout", "choice", default="center", choices=("center", "all"),
              tooltip="center = the target_count sites closest to the array centre (a filled central "
                      "block); all = every site (only feasible at high loading)."),
    ParamDecl("readout_exposure", "readout exposure", "float", default=0.020, unit="s", lo=0.0, hi=10.0,
              tooltip="Camera exposure for the before / after images that detect occupancy (through the "
                      "current calibration's thresholds)."),
    ParamDecl("move_survival", "move survival (-1=device)", "float", default=-1.0, lo=-1.0, hi=1.0,
              tooltip="Per-move atom transit survival probability used by the actuator.  -1 = use the "
                      "AOD device's own default (≈0.98 on the virtual AOD; the real loss is physical)."),
)


@task(order=20)
def rearrange_array(readout) -> TaskSpec:
    """The AOD rearrangement task (image -> plan -> move -> re-image -> fill fraction).

    Its tunable parameters (:data:`REARRANGE_PARAMS`) are threaded into the built
    :class:`~..logic.RearrangeTask`; mid-run it streams the before/after frames to its dedicated panel
    under the ``rearrange_`` namespace.  Requires a thresholded calibration (run Calibrate readout first)
    + an ``aod`` device on the connection."""

    def build(hub, *, prefix: str = "rearrange_", **param_values):
        return readout.session.rearrange.task(hub, prefix=prefix, **param_values)

    return TaskSpec(name="Rearrange array", build=build, params=REARRANGE_PARAMS,
                    mid_run_key="frame_after", default_kind="2d", prefix="rearrange_")
