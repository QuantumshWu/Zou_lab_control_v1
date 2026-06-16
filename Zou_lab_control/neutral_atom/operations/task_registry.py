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

Imports no concrete backend and never touches the frontend; the catalog is
assembled session-side and handed to the (decoupled) GUI as a plain list.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Callable

_REGISTERED: list[tuple[int, int, Callable]] = []
_SEQ = 0
_DISCOVERED = False

DEFAULT_ORDER = 100


def register_task(factory: Callable, *, order: int = DEFAULT_ORDER) -> Callable:
    """Register ``factory(readout) -> TaskSpec`` (idempotent by identity).

    Returns the factory unchanged so it doubles as a decorator.  ``order`` sets
    catalog position (lower = earlier); built-ins use small orders."""

    global _SEQ
    if not callable(factory):
        raise TypeError(f"task factory must be callable, got {factory!r}.")
    if any(entry[2] is factory for entry in _REGISTERED):
        return factory
    _REGISTERED.append((int(order), _SEQ, factory))
    _SEQ += 1
    return factory


def task(factory: Callable | None = None, *, order: int = DEFAULT_ORDER):
    """Decorator form of :func:`register_task`.

    Use bare (``@task``) or with an order (``@task(order=10)``) on a
    ``def my_task(readout) -> TaskSpec`` factory."""

    if factory is None:
        return lambda fn: register_task(fn, order=order)
    return register_task(factory, order=order)


def unregister_task(factory: Callable) -> bool:
    """Remove a previously registered factory.  Returns True if one was removed."""

    before = len(_REGISTERED)
    _REGISTERED[:] = [entry for entry in _REGISTERED if entry[2] is not factory]
    return len(_REGISTERED) != before


def registered_tasks() -> tuple[Callable, ...]:
    """The registered factories, in catalog order (built-ins first)."""

    _autodiscover()
    return tuple(entry[2] for entry in sorted(_REGISTERED, key=lambda e: (e[0], e[1])))


def _autodiscover() -> None:
    """Import every module in ``operations/tasks/`` exactly once so each ``@task``
    factory registers.  A module that fails to import is skipped with a warning --
    one bad file must not break the whole catalog."""

    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    from . import tasks as _pkg

    for info in pkgutil.iter_modules(_pkg.__path__):
        if info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{_pkg.__name__}.{info.name}")
        except Exception as exc:  # pragma: no cover - defensive: skip a broken file
            import warnings

            warnings.warn(f"task module {info.name!r} failed to import: {exc}")


def discovered_task_specs(readout):
    """Build every registered task's :class:`TaskSpec` for ``readout``.

    Auto-discovers built-ins, calls each factory with the readout subsystem, and
    returns specs ordered by ``order`` then registration, de-duplicated by ``name``
    (later registration wins, keeping the first position).  Raises if two tasks share
    a hub ``prefix`` -- their namespaced signals would overwrite each other on the
    shared SignalHub, so each must pick a unique prefix."""

    _autodiscover()
    out: list = []
    pos: dict[str, int] = {}
    for _, _, factory in sorted(_REGISTERED, key=lambda e: (e[0], e[1])):
        spec = factory(readout)
        if spec is None:
            continue
        name = spec.name
        if name in pos:
            out[pos[name]] = spec
        else:
            pos[name] = len(out)
            out.append(spec)
    seen: dict[str, str] = {}
    for spec in out:
        prefix = getattr(spec, "prefix", "")
        if not prefix:
            continue
        if prefix in seen and seen[prefix] != spec.name:
            raise ValueError(
                f"tasks {seen[prefix]!r} and {spec.name!r} share hub prefix {prefix!r}; "
                "give each task a unique prefix so their signals do not collide.")
        seen[prefix] = spec.name
    return out


__all__ = [
    "DEFAULT_ORDER",
    "discovered_task_specs",
    "register_task",
    "registered_tasks",
    "task",
    "unregister_task",
]
