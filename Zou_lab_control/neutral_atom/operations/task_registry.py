"""Open registry + auto-discovery for orchestration TASKS (the catalog source).

Mirrors :mod:`processor_registry` one tier over.  A task is contributed as a
FACTORY ``build(readout) -> TaskSpec``: it receives the readout subsystem so its
``build(hub)`` closure can capture the session (camera / sequencer / calibration).
Two ways to make one appear in ``exp.readout.task_specs()`` (and so in the task
console's Add-Panel "Task" group):

* drop a module into the ``operations/tasks/`` package with a factory decorated
  ``@task`` -- it is AUTO-DISCOVERED (no hardcoded catalog to edit);
* or call :func:`register_task` from a notebook for an ad-hoc one.

``discovered_task_specs(readout)`` imports the package once, builds every registered
spec, de-duplicates by ``name`` (later wins), and FAILS LOUD if two tasks share a
hub ``prefix`` -- their namespaced signals would clobber each other on the shared
SignalHub.

The mechanics are the SHARED :class:`._open_registry.OpenRegistry` (identical to the
measurement and processor catalogs); this module just binds the task-named public
API to one instance and injects the ``prefix`` collision check.  Imports no concrete
backend and never touches the frontend.
"""

from __future__ import annotations

from ._open_registry import DEFAULT_ORDER, OpenRegistry

_REGISTRY = OpenRegistry(
    noun="task",
    package=f"{__package__}.tasks",
    collision_keys=lambda spec: (getattr(spec, "prefix", ""),),
    collision_advice="give each task a unique prefix so their signals do not collide.",
)

register_task = _REGISTRY.register
task = _REGISTRY.decorator
unregister_task = _REGISTRY.unregister
registered_tasks = _REGISTRY.registered
discovered_task_specs = _REGISTRY.discovered_specs


__all__ = [
    "DEFAULT_ORDER",
    "discovered_task_specs",
    "register_task",
    "registered_tasks",
    "task",
    "unregister_task",
]
