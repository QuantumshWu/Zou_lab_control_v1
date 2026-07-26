"""Narrow application-host seam for Calibration's optional live output."""

from __future__ import annotations

from .task import PreparedCalibrationTask


def start_calibration_task_command(
    command: PreparedCalibrationTask,
    live_output_host,
):
    """Choose Calibration's declared live/saved start shape, and nothing else."""

    if not isinstance(command, PreparedCalibrationTask):
        raise TypeError("Calibration preparer returned another command type")
    if not callable(getattr(live_output_host, "open_live_dataset", None)):
        raise TypeError("Calibration start requires a live-output host")
    return command.start(live_output_host if command.has_live_output else None)


__all__ = ["start_calibration_task_command"]
