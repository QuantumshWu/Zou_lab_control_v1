"""AOD atom-rearrangement subsystem -- assemble a defect-free array (``exp.rearrange.*``).

The orchestration facade for rearrangement: image the stochastic loading, detect occupancy through the
session calibration, plan the moves that fill a target defect-free set (:mod:`operations.rearrangement`),
drive the AOD to execute them, re-image, report the fill fraction.  It owns NO state -- it wires the
session's camera + AOD devices + imaging path + calibration into a :class:`~..operations.logic.RearrangeTask`
(the GUI task console runs the SAME task), so the notebook one-liner and the GUI button cannot drift, and
a virtual run traverses the identical contract path a real run does (only the camera frames + the AOD
backend differ).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..operations.logic import RearrangeTask
from .base import ExperimentSubsystem

if TYPE_CHECKING:  # pragma: no cover
    from ..session import NeutralAtomSession


class RearrangeSubsystem(ExperimentSubsystem):
    """Defect-free array assembly via the crossed AOD (see module docstring)."""

    _session: "NeutralAtomSession"

    def _frame_provider(self):
        """A closure that images ONE frame through the readout's imaging path (the SAME sequence the
        sitemap / detect calls fire) -- so the rearrange task images identically to the rest of readout.
        ``load`` fires the cooling/MOT load first: True for the initial image (a fresh array), False for
        the post-rearrange image (image the assembled atoms WITHOUT reloading -- on real hardware a
        reload would cool a new random array on top of the rearranged one)."""
        s = self._session

        def image_once(exposure: float, load: bool = True):
            sequence = s._imaging_sequence(exposure=float(exposure), load=bool(load), name="rearrange")
            frames = s.devices.camera.acquire(1, sequence=sequence, sequencer=getattr(s.devices, "sequencer", None))
            return frames[-1]

        return image_once

    def task(self, hub, *, prefix: str = "rearrange_", grid_shape=None, **params) -> RearrangeTask:
        """Build (UNRUN) the rearrangement TASK over the session camera + AOD.

        Run it from the console (Add Panel -> "Task: Rearrange array") or the notebook (``.step()``):
        mid-run it streams before/after frames + occupancy + progress to a dedicated panel, and produces
        the fill-fraction result.  Wires the frame-imaging path + the session calibration so the task
        stays backend-free; requires an ``aod`` device on the connection (virtual always has one)."""
        s = self._session
        grid_shape = s._grid_shape(grid_shape)

        def current_calibration():
            return s.require_calibration(require_thresholds=True)

        return RearrangeTask(
            hub, s.devices.camera, s.devices.aod,
            sequencer=getattr(s.devices, "sequencer", None), grid_shape=grid_shape,
            frame_provider=self._frame_provider(), calibration_provider=current_calibration,
            prefix=prefix, **params,
        )

    def execute(self, **params):
        """Notebook one-liner: run ONE rearrangement pass now and return its result dict
        (fill_fraction / n_filled / n_target / n_loaded / n_moves / occupancy_before/after / moves).

        Requires a thresholded calibration (``exp.readout.sitemap()`` + ``.thresholds()`` first) and an
        ``aod`` device.  Runs the SAME :class:`~..operations.logic.RearrangeTask` the GUI runs, headless."""
        from ..core.signals import SignalHub
        task = self.task(SignalHub(), **params)            # a task publishes nothing to the hub
        task.run_to_completion()
        result = dict(task.result)
        self._session.history.append(result)
        return result


__all__ = ["RearrangeSubsystem"]
