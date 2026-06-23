"""The ONE multi-slot signal + value-expression evaluator.

A "source" anywhere in the system is the SAME thing: a small list of picked hub-signal
names (the *slots*, read as ``signal`` / ``signal[i]`` in an expression) plus a one-line
Python expression that assigns its result to ``value``.  A plot panel's data source, a
processor's input, and a pulse-scan's y all share this shape.  This module owns the data
model + its evaluation so there is exactly ONE implementation -- the slot-packing rule
(scalar for one slot, list for many) and the ``value = ...`` contract live here, never
re-rolled per call site.

Dependency-free (no frontend, no backend, reads no simulation ground truth), so both the
analysis layer (``OccupancyProcessor``, ``PulseScanNode``) and the GUI (the panel Setting
and the measurement param form) import this one definition.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable, Mapping

import numpy as np

# Shown both as the inline source field's tooltip and the floating editor's prompt, and as
# the pulse-scan / processor source tooltip -- so every "source" field describes itself the
# same way.  ONE description of the expression namespace.
SIGNAL_EXPR_HELP = (
    "Data source: one line of Python evaluated against the live signals.\n"
    "Assign the result to `value`.  `signal` is the picked signal (one slot); with more\n"
    "than one slot it is a list, so `value = signal[0] - signal[1]` combines them.\n"
    "Namespace: every signal name (latest value), history(name, n), latest(name),\n"
    "names(), shot, np, math.")

#: The canonical source for a single picked signal: the picked signal IS ``signal``.
DEFAULT_SOURCE = "value = signal"


@lru_cache(maxsize=512)
def _compile_source(source: str):
    """Compile a source expression to a code object, cached across instances/ticks (the
    live panel evaluates the same source every tick -- compile once)."""
    return compile(str(source), "<signal-expr>", "exec")


def seed_source_for_slots(n_slots: int, current: str = "") -> str:
    """The canonical source for ``n_slots`` picked signals: ``value = signal`` for one slot,
    ``value = signal[0] - signal[1]`` (and so on) seeded for more -- but a real custom
    expression the operator already typed is kept (only the default is re-seeded)."""

    cur = str(current or "").strip()
    if int(n_slots) <= 1:
        return DEFAULT_SOURCE
    if cur and cur != DEFAULT_SOURCE:
        return cur                                   # keep a genuine custom expression
    return "value = " + " - ".join(f"signal[{i}]" for i in range(int(n_slots)))


class SignalExpr:
    """A list of picked signal names (slots) + a ``value = ...`` expression over them.

    ``inputs`` are the picked hub-signal names in slot order; ``source`` is the one-line
    expression.  ``signal`` in the expression is the single slot's value for one input, a
    list of slot values for several (so ``value = signal[0] - signal[1]`` combines them).
    """

    def __init__(self, inputs=None, source: str = DEFAULT_SOURCE):
        self.inputs = [str(n) for n in (inputs or []) if str(n).strip()]
        self.source = str(source or DEFAULT_SOURCE)

    @classmethod
    def from_value(cls, value) -> "SignalExpr":
        """Build from a GUI/saved value: a ``{"inputs": [...], "source": "..."}`` dict, an
        existing :class:`SignalExpr`, a bare expression string, or a list of names."""
        if isinstance(value, SignalExpr):
            return value
        if isinstance(value, Mapping):
            return cls(value.get("inputs"), value.get("source", DEFAULT_SOURCE))
        if isinstance(value, str):
            return cls([], value)                    # a bare expression, no slots
        if value is None:
            return cls([], DEFAULT_SOURCE)
        try:
            return cls(list(value), DEFAULT_SOURCE)   # a list of slot names
        except TypeError:
            return cls([], DEFAULT_SOURCE)

    def as_value(self) -> dict:
        """The persistable ``{"inputs": [...], "source": "..."}`` form (GUI value)."""
        return {"inputs": list(self.inputs), "source": self.source}

    def signal_for(self, namespace: Mapping[str, object], *, resolve: Callable[[str], str] | None = None):
        """The ``signal`` value for ``namespace``: the single slot's value (scalar) for one
        input, else the list of slot values.  ``resolve`` optionally rewrites a name before
        lookup (the frame-coherence hook -- a frontend concern injected in, never baked here)."""

        def _get(name: str):
            if name and resolve is not None:
                try:
                    name = str(resolve(name)) or name
                except Exception:
                    pass
            return namespace.get(name) if name else None

        resolved = [_get(n) for n in self.inputs]
        return (resolved[0] if resolved else None) if len(resolved) <= 1 else resolved

    def co_names(self) -> frozenset:
        """The hub-signal names this expression reads: the identifiers it names directly PLUS
        the picked slot inputs (the default ``value = signal`` references the pseudo ``signal``,
        not a real name, so the inputs must be folded in for version-gating)."""
        try:
            names = set(_compile_source(self.source).co_names)
        except Exception:
            names = set()
        names.update(n for n in self.inputs if n)
        return frozenset(names)

    def exec_in(self, namespace: dict) -> object:
        """Execute the source IN ``namespace`` (which must already carry ``signal``) and return
        the assigned ``value``.  SECURITY: runs the operator-entered snippet as arbitrary Python
        -- same trusted-local-tool posture as the pulse GUI Scan tab (run only layouts you trust)."""
        exec(_compile_source(self.source), namespace)   # noqa: S102 - local experiment tool, trusted input
        if "value" not in namespace:
            raise ValueError("assign the result to a `value = ...` variable")
        return namespace["value"]

    def evaluate(self, namespace: Mapping[str, object], *, resolve: Callable[[str], str] | None = None) -> object:
        """One-shot: copy ``namespace``, inject ``signal``, execute, return ``value``."""
        ns = dict(namespace)
        ns["signal"] = self.signal_for(ns, resolve=resolve)
        return self.exec_in(ns)


def hub_namespace(hub) -> dict:
    """The expression namespace for a :class:`~..core.signals.SignalHub`: every latest signal
    value plus the helpers (``history``/``latest``/``names``/``shot``/``np``/``math``).  The
    node-side single source -- the console's GUI namespace layers its view-only keys on top."""

    ns = dict(hub.snapshot_latest())
    ns["history"] = lambda name, n=None: hub.history(name, n)
    ns["latest"] = lambda name: hub.latest(name)
    ns["names"] = lambda: hub.names()
    ns["shot"] = hub.shot
    ns["np"] = np
    ns["math"] = math
    return ns


__all__ = ["SignalExpr", "SIGNAL_EXPR_HELP", "DEFAULT_SOURCE", "seed_source_for_slots", "hub_namespace"]
