"""Shared open-registry + auto-discovery machinery (the catalog single source).

The measurement / processor / task catalogs are the SAME pattern -- a factory
``build(readout) -> Spec`` is contributed either by dropping a module into a
built-ins package (auto-discovered, no hardcoded list) or by a notebook calling
``register_*``; ``discovered_*_specs(readout)`` then imports the package once,
calls every factory with ``readout`` and returns the specs ordered by ``order``
(built-ins first) then registration, de-duplicated by ``name`` (later wins).

That whole mechanism used to be spelled THREE times (one near-identical module per
kind).  It lives here ONCE now: each catalog is one :class:`OpenRegistry` instance,
and the per-kind modules are thin shells that bind the public names
(``register_measurement`` etc.) to its methods.  The only thing that genuinely
differs per kind is which spec fields would COLLIDE on the shared SignalHub -- so
that is injected as ``collision_keys`` (+ a human ``collision_advice``); everything
else is shared.

This module imports no concrete backend and never touches the frontend.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from typing import Callable

# Default order for user/ad-hoc factories; built-ins pass a smaller order so they
# list first.  Plenty of headroom on either side.
DEFAULT_ORDER = 100


class OpenRegistry:
    """One auto-discovered catalog: register factories, build their specs in order.

    ``noun`` names the kind for messages ("measurement") and ``package`` is the dotted
    path of the built-ins package scanned on first use.  The de-dup rule lives on the
    SPEC, not here: discovery calls ``spec.collision_key()`` (the hub keys two specs
    must not share) and raises loud with the spec's ``collision_advice`` appended -- so
    every catalog's collision check has a single source (the :class:`CatalogSpec`
    subclass), not a per-registry lambda."""

    def __init__(self, *, noun: str, package: str) -> None:
        self._noun = noun
        self._package = package
        # Each entry is (order, sequence, factory).  ``sequence`` is a monotonic
        # tie-break so equal-order factories keep registration order; a list (not a
        # dict) keeps duplicates out via identity while preserving order.
        self._registered: list[tuple[int, int, Callable]] = []
        self._seq = 0
        self._discovered = False

    def register(self, factory: Callable, *, order: int = DEFAULT_ORDER) -> Callable:
        """Register ``factory(readout) -> Spec`` (idempotent by identity).

        Returns the factory unchanged so it doubles as a decorator.  ``order`` sets
        catalog position (lower = earlier); built-ins use small orders."""

        if not callable(factory):
            raise TypeError(f"{self._noun} factory must be callable, got {factory!r}.")
        if any(entry[2] is factory for entry in self._registered):
            return factory
        self._registered.append((int(order), self._seq, factory))
        self._seq += 1
        return factory

    def decorator(self, factory: Callable | None = None, *, order: int = DEFAULT_ORDER):
        """Decorator form of :meth:`register` -- bare or with an order."""

        if factory is None:
            return lambda fn: self.register(fn, order=order)
        return self.register(factory, order=order)

    def unregister(self, factory: Callable) -> bool:
        """Remove a previously registered factory.  Returns True if one was removed."""

        before = len(self._registered)
        self._registered[:] = [entry for entry in self._registered if entry[2] is not factory]
        return len(self._registered) != before

    def registered(self) -> tuple[Callable, ...]:
        """The registered factories, in catalog order (built-ins first)."""

        self._autodiscover()
        return tuple(entry[2] for entry in sorted(self._registered, key=lambda e: (e[0], e[1])))

    def reset_discovery(self) -> None:
        """Force the next call to re-scan the built-ins package.

        A TEST artifact: the catalog is discovered once per process; a test that
        writes a new module into the package mid-run calls this so the next
        ``discovered_specs`` re-imports it (production drops a file and RESTARTS)."""

        self._discovered = False

    def _autodiscover(self) -> None:
        """Import every module in the built-ins package exactly once so each
        decorated factory registers.  A module that fails to import is skipped with
        a warning -- one bad file must not break the whole catalog."""

        if self._discovered:
            return
        self._discovered = True
        pkg = importlib.import_module(self._package)
        for info in pkgutil.iter_modules(pkg.__path__):
            if info.name.startswith("_"):
                continue
            try:
                importlib.import_module(f"{pkg.__name__}.{info.name}")
            except Exception as exc:  # pragma: no cover - defensive: skip a broken file
                warnings.warn(f"{self._noun} module {info.name!r} failed to import: {exc}")

    def discovered_specs(self, readout) -> list:
        """Build every registered factory's spec for ``readout``.

        Auto-discovers built-ins, calls each factory with the readout subsystem (so
        its ``build`` closure captures the session), returns specs ordered by
        ``order`` then registration and de-duplicated by ``name`` (later registration
        overrides in the first slot).  Raises if two specs collide on a hub key."""

        self._autodiscover()
        out: list = []
        pos: dict[str, int] = {}
        for _, _, factory in sorted(self._registered, key=lambda e: (e[0], e[1])):
            spec = factory(readout)
            if spec is None:
                continue
            name = spec.name
            if name in pos:
                out[pos[name]] = spec  # later registration overrides, same slot
            else:
                pos[name] = len(out)
                out.append(spec)
        # Two specs sharing a hub key would overwrite each other on the shared
        # SignalHub -- a silent data mix-up.  Fail loud at discovery so a new
        # contributor must pick a unique key.
        seen: dict[str, str] = {}
        for spec in out:
            for key in spec.collision_key():
                if not key:
                    continue
                if key in seen and seen[key] != spec.name:
                    raise ValueError(
                        f"{self._noun}s {seen[key]!r} and {spec.name!r} both publish signal {key!r}; "
                        + type(spec).collision_advice)
                seen[key] = spec.name
        return out


__all__ = ["DEFAULT_ORDER", "OpenRegistry"]
