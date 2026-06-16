"""Open registry + auto-discovery for data-processing actions (the catalog source).

Mirrors :mod:`measurement_registry` one tier over.  A processor is contributed as
a FACTORY ``build(readout) -> ProcessorSpec``: it receives the readout subsystem so
its ``run`` closure can capture the session (devices/calibration) and DRIVE the
subsystem's analysis.  Two ways to make one appear in
``exp.readout.processor_specs()`` (and so in the task console's Add-Panel
"Data processing" group):

* drop a module into the ``operations/processors/`` package with a factory
  decorated ``@processor`` -- it is AUTO-DISCOVERED (no hardcoded catalog to edit);
* or call :func:`register_processor` from a notebook for an ad-hoc one.

``discovered_processor_specs(readout)`` imports the package once, builds every
registered spec, de-duplicates by ``name`` (later wins), and FAILS LOUD if two
processors publish the same ``result_keys`` signal -- they would clobber each other
on the shared SignalHub.

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


def register_processor(factory: Callable, *, order: int = DEFAULT_ORDER) -> Callable:
    """Register ``factory(readout) -> ProcessorSpec`` (idempotent by identity).

    Returns the factory unchanged so it doubles as a decorator.  ``order`` sets
    catalog position (lower = earlier); built-ins use small orders."""

    global _SEQ
    if not callable(factory):
        raise TypeError(f"processor factory must be callable, got {factory!r}.")
    if any(entry[2] is factory for entry in _REGISTERED):
        return factory
    _REGISTERED.append((int(order), _SEQ, factory))
    _SEQ += 1
    return factory


def processor(factory: Callable | None = None, *, order: int = DEFAULT_ORDER):
    """Decorator form of :func:`register_processor`.

    Use bare (``@processor``) or with an order (``@processor(order=10)``) on a
    ``def my_processor(readout) -> ProcessorSpec`` factory."""

    if factory is None:
        return lambda fn: register_processor(fn, order=order)
    return register_processor(factory, order=order)


def unregister_processor(factory: Callable) -> bool:
    """Remove a previously registered factory.  Returns True if one was removed."""

    before = len(_REGISTERED)
    _REGISTERED[:] = [entry for entry in _REGISTERED if entry[2] is not factory]
    return len(_REGISTERED) != before


def registered_processors() -> tuple[Callable, ...]:
    """The registered factories, in catalog order (built-ins first)."""

    _autodiscover()
    return tuple(entry[2] for entry in sorted(_REGISTERED, key=lambda e: (e[0], e[1])))


def _autodiscover() -> None:
    """Import every module in ``operations/processors/`` exactly once so each
    ``@processor`` factory registers.  A module that fails to import is skipped with
    a warning -- one bad file must not break the whole catalog."""

    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    from . import processors as _pkg

    for info in pkgutil.iter_modules(_pkg.__path__):
        if info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{_pkg.__name__}.{info.name}")
        except Exception as exc:  # pragma: no cover - defensive: skip a broken file
            import warnings

            warnings.warn(f"processor module {info.name!r} failed to import: {exc}")


def discovered_processor_specs(readout):
    """Build every registered processor's :class:`ProcessorSpec` for ``readout``.

    Auto-discovers built-ins, calls each factory with the readout subsystem, and
    returns specs ordered by ``order`` then registration, de-duplicated by ``name``
    (later registration wins, keeping the first position).  Raises if two processors
    publish the same ``result_keys`` signal -- they would overwrite each other on the
    shared SignalHub, so each must pick unique output names."""

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
        for key in getattr(spec, "result_keys", ()):
            if not key:
                continue
            if key in seen and seen[key] != spec.name:
                raise ValueError(
                    f"processors {seen[key]!r} and {spec.name!r} both publish signal {key!r}; "
                    "give each processor unique result_keys (e.g. a per-processor prefix).")
            seen[key] = spec.name
    return out


__all__ = [
    "DEFAULT_ORDER",
    "discovered_processor_specs",
    "processor",
    "register_processor",
    "registered_processors",
    "unregister_processor",
]
