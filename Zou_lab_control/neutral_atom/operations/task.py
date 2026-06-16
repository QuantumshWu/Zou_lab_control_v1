"""Declarative TASK catalog entry (the orchestration layer's catalog source).

A task is the orchestration tier of the five layers (device / measurement /
processor / task / plot): a one-shot flow that may drive several devices /
measurements, save frames, derive an artifact (a calibration, an npz), and stream
MID-RUN output to its own dedicated panel -- confocal's "task" idea.

A task is contributed as a FACTORY ``build(readout) -> TaskSpec`` and auto-discovered
exactly like a measurement / processor (see :mod:`task_registry`).  The spec is a
thin declarative record the (decoupled) GUI consumes: it names the task and carries
a ``build(hub) -> Task`` closure (capturing the readout subsystem, so the console
never holds the session) plus which MID-RUN signal its dedicated panel shows.  The
task's editable parameters + Run come from the :class:`~.feeds.Task` itself (its
``acquisition_parameters`` / one-shot ``start``), so there is ONE param mechanism --
no parallel form engine.

Imports no concrete backend and never touches the frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class TaskSpec:
    """A named orchestration task + how to build it + its mid-run output binding.

    ``build(hub, *, prefix=...) -> Task`` returns an UNRUN :class:`~.feeds.Task`
    over the session (the closure captures the readout subsystem).  ``mid_run_key``
    is the task's mid-run output signal the dedicated panel shows while it runs
    (e.g. ``frame`` -> the template frame), ``default_kind`` is that panel's plot
    kind, and ``prefix`` is the task's hub namespace (so its signals never clobber a
    live stream).  The task's tunable parameters + Run come from the built Task's own
    ``acquisition_parameters`` (single param mechanism)."""

    name: str
    build: Callable[..., Any]          # build(hub, *, prefix=...) -> Task
    mid_run_key: str = "frame"
    default_kind: str = "2d"
    prefix: str = "cal_"
    metadata: dict[str, Any] = field(default_factory=dict)

    def mid_run_signal(self) -> str:
        """The fully-namespaced mid-run signal a dedicated panel reads (``cal_frame``)."""
        return f"{self.prefix}{self.mid_run_key}"


__all__ = ["TaskSpec"]
