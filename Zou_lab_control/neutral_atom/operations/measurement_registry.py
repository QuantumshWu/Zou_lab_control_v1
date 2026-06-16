"""Open registry + auto-discovery for measurements (the catalog single source).

A measurement is contributed as a FACTORY ``build(readout) -> MeasurementSpec``:
it receives the readout subsystem so its ``build`` closure can capture the
session (devices/calibration) and reuse the subsystem's scan builders.  Two ways
to make one appear in ``exp.readout.measurement_specs()`` (and therefore in the
task console's Add-Panel list):

* drop a module into the ``operations/measurements/`` package with a factory
  decorated ``@measurement`` -- it is AUTO-DISCOVERED (no hardcoded catalog list
  to edit); the built-ins live there too, so a new measurement is exactly as
  first-class as ``temperature``;
* or call :func:`register_measurement` from a notebook for an ad-hoc one.

``discovered_specs(readout)`` imports the package once, then calls every
registered factory with ``readout`` and returns the freshly-built specs ordered
by ``order`` (built-ins first) then registration, de-duplicated by ``name``
(later wins, so a notebook can override a built-in by re-using its name).

This module imports no concrete backend and never touches the frontend; the
catalog is assembled session-side and handed to the (decoupled) GUI as a plain
list of :class:`MeasurementSpec`.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Callable

# Each entry is (order, sequence, factory).  ``sequence`` is a monotonic tie-break
# so equal-order factories keep registration order; ``order`` lets built-ins sort
# ahead of ad-hoc ones.  A module-level list (not a dict) keeps duplicates out via
# identity while preserving order.
_REGISTERED: list[tuple[int, int, Callable]] = []
_SEQ = 0
_DISCOVERED = False

# Default order for user/ad-hoc measurements; built-ins pass a smaller order so
# they list first.  Plenty of headroom on either side.
DEFAULT_ORDER = 100


def register_measurement(factory: Callable, *, order: int = DEFAULT_ORDER) -> Callable:
    """Register ``factory(readout) -> MeasurementSpec`` (idempotent by identity).

    Returns the factory unchanged so it doubles as a decorator.  ``order`` sets
    catalog position (lower = earlier); built-ins use small orders."""

    global _SEQ
    if not callable(factory):
        raise TypeError(f"measurement factory must be callable, got {factory!r}.")
    if any(entry[2] is factory for entry in _REGISTERED):
        return factory
    _REGISTERED.append((int(order), _SEQ, factory))
    _SEQ += 1
    return factory


def measurement(factory: Callable | None = None, *, order: int = DEFAULT_ORDER):
    """Decorator form of :func:`register_measurement`.

    Use bare (``@measurement``) or with an order (``@measurement(order=10)``) on a
    ``def my_measurement(readout) -> MeasurementSpec`` factory."""

    if factory is None:
        return lambda fn: register_measurement(fn, order=order)
    return register_measurement(factory, order=order)


def unregister_measurement(factory: Callable) -> bool:
    """Remove a previously registered factory.  Returns True if one was removed.

    Mainly for notebooks (drop a measurement you added) and test isolation."""

    before = len(_REGISTERED)
    _REGISTERED[:] = [entry for entry in _REGISTERED if entry[2] is not factory]
    return len(_REGISTERED) != before


def registered_measurements() -> tuple[Callable, ...]:
    """The registered factories, in catalog order (built-ins first)."""

    _autodiscover()
    return tuple(entry[2] for entry in sorted(_REGISTERED, key=lambda e: (e[0], e[1])))


def _autodiscover() -> None:
    """Import every module in ``operations/measurements/`` exactly once so each
    ``@measurement`` factory registers.  A module that fails to import is skipped
    with a warning -- one bad measurement file must not break the whole catalog."""

    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    from . import measurements as _pkg

    for info in pkgutil.iter_modules(_pkg.__path__):
        if info.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{_pkg.__name__}.{info.name}")
        except Exception as exc:  # pragma: no cover - defensive: skip a broken file
            import warnings

            warnings.warn(f"measurement module {info.name!r} failed to import: {exc}")


def discovered_specs(readout):
    """Build every registered measurement's :class:`MeasurementSpec` for ``readout``.

    Auto-discovers built-ins, calls each factory with the readout subsystem (so
    its ``build`` closure captures the session), and returns specs ordered by
    ``order`` then registration, de-duplicated by ``name`` (later registration
    wins, keeping the first position) -- ``name`` is the GUI's lookup key, so it
    must be unique."""

    _autodiscover()
    out: list = []
    pos: dict[str, int] = {}
    for _, _, factory in sorted(_REGISTERED, key=lambda e: (e[0], e[1])):
        spec = factory(readout)
        if spec is None:
            continue
        name = spec.name
        if name in pos:
            out[pos[name]] = spec  # later registration overrides, same slot
        else:
            pos[name] = len(out)
            out.append(spec)
    # Two specs publishing the same x_key/y_key would overwrite each other on the
    # shared SignalHub (and their _sites/_grid derivatives) -- a silent data mix-up.
    # Fail loud at discovery so a new measurement must pick a unique signal key.
    seen: dict[str, str] = {}
    for spec in out:
        for key in (getattr(spec, "x_key", None), getattr(spec, "y_key", None)):
            if not key:
                continue
            if key in seen and seen[key] != spec.name:
                raise ValueError(
                    f"measurements {seen[key]!r} and {spec.name!r} both publish signal {key!r}; "
                    "give each measurement a unique x_key/y_key (e.g. a per-measurement prefix).")
            seen[key] = spec.name
    return out


__all__ = [
    "DEFAULT_ORDER",
    "discovered_specs",
    "measurement",
    "register_measurement",
    "registered_measurements",
    "unregister_measurement",
]
