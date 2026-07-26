"""Narrow application-host seam for MOT-field's existing live output."""

from __future__ import annotations

from .mot_field_task import PreparedMotFieldTask


def start_mot_field_task_command(
    command: PreparedMotFieldTask,
    live_output_host,
):
    """Attach the task-owned source, then invoke its domain start method."""

    if not isinstance(command, PreparedMotFieldTask):
        raise TypeError("MOT-field preparer returned another command type")
    attach_live_output = getattr(live_output_host, "attach_live_output", None)
    if not callable(attach_live_output):
        raise TypeError("MOT-field start requires a live-output host")
    attach_live_output(command.live_output)
    return command.start()


__all__ = ["start_mot_field_task_command"]
