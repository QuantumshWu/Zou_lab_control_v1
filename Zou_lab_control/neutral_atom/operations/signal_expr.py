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
import re
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
    "Every signal value is canonical (R, P, *data_shape).\n"
    "Namespace: every signal name (latest canonical value), history(name, n),\n"
    "history_logical(name, n), latest(name),\n"
    "tensor(name), logical(name), valid(name), schema(name), names(), shot, np (alias numpy), math.\n"
    "Reactive processors evaluate only their cursor-selected input snapshot: they retain\n"
    "logical/valid/schema/names/shot plus np/numpy/math, but expose no live/latest/history/tensor lookup.")

#: The helper names EVERY expression namespace injects -- the SINGLE spelling.  The GUI's
#: reference-exclusion set (which identifiers in a source are NOT hub signals) and the help
#: text above derive from this, and :func:`namespace_helpers` binds exactly these names, so a
#: helper added in one place can never silently miss the others (contract-tested).
NAMESPACE_HELPERS: tuple[str, ...] = (
    "history", "history_logical", "latest", "tensor", "logical", "valid", "schema",
    "names", "shot", "np", "numpy", "math",
)

#: Helpers that are safe inside a Processor's one-update transaction.  Every
#: data-bearing helper resolves only against the exact immutable tensors selected
#: by that processor's Hub subscription; none can reach a newer Hub value or an
#: unconsumed dependency.  In particular, history/latest/tensor are intentionally
#: absent rather than silently redirected to live state.
PROCESSOR_NAMESPACE_HELPERS: tuple[str, ...] = (
    "logical", "valid", "schema", "names", "shot", "np", "numpy", "math",
)

#: The canonical source for a single picked signal: the picked signal IS ``signal``.
DEFAULT_SOURCE = "value = signal"

#: A bare ``value = <hub signal name>`` source: the picked signal named DIRECTLY instead of via the
#: pseudo ``signal``.  ONE pattern, shared by :func:`is_identity_source` (does this source pass the
#: signal through unchanged?) and the panel-config input reflection (a bare name IS the input slot),
#: so those two never disagree on what "just names a signal" means.
IDENTITY_SOURCE_RE = re.compile(r"\s*value\s*=\s*([A-Za-z_]\w*)\s*")


def is_identity_source(source, inputs=()) -> bool:
    """True when ``source`` is a pure PASS-THROUGH of a single picked signal -- the canonical
    ``value = signal`` OR ``value = <the one input's name>`` (naming the slot directly).  Such a
    binding does NOT rewrite the core shape, so a producing node's DECLARED structure still describes
    the value (the reshape / facet layer relies on this to keep a 2-D frame a 2-D frame, a scan grid a
    scan grid).  A multi-slot or transforming expression (``value = signal[0] - signal[1]``,
    ``value = np.log(f)``) returns ``False`` -- there the declared structure no longer applies."""
    src = str(source or "").strip()
    if src == DEFAULT_SOURCE:
        return True
    m = IDENTITY_SOURCE_RE.fullmatch(src)
    names = [str(n) for n in (inputs or []) if str(n).strip()]
    return bool(m and m.group(1) != "signal" and len(names) == 1 and m.group(1) == names[0])


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

    def direct_names(self) -> frozenset:
        """ONLY the identifiers the source TEXT itself names (the compiled ``co_names``,
        no folded-in slot inputs).  This is the raw-signal detector's set: a consumer that
        must slice per repeat exactly the blocks the expression READS BY NAME must not
        mistake the bound slot (fed through ``signal``) for a directly-named block --
        that mistake made the default identity source lose its zero-copy path and
        float64-materialise every camera frame per tick (W-round regression)."""
        try:
            return frozenset(_compile_source(self.source).co_names)
        except Exception:
            return frozenset()

    def co_names(self) -> frozenset:
        """The hub-signal names this expression reads: the identifiers it names directly
        (:meth:`direct_names`) PLUS the picked slot inputs (the default ``value = signal``
        references the pseudo ``signal``, not a real name, so the inputs must be folded in
        for version-gating)."""
        names = set(self.direct_names())
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


class ProcessorSignalSnapshot(dict):
    """One Processor cursor transaction, represented as canonical arrays.

    ``tensors`` are the exact historical :class:`SignalTensor` states returned
    together by :meth:`SignalHub.next_coherent_update`.  The dict values are
    independent canonical ``(R,P,*data_shape)`` copies for ordinary expressions;
    the immutable tensors remain attached so snapshot-safe schema/valid/logical
    helpers never consult the live Hub.  This marker also lets :func:`hub_namespace`
    enforce the narrower Processor namespace without every concrete Processor
    reimplementing the policy.
    """

    def __init__(self, tensors: Mapping[str, object], *, provenance: int):
        selected = {str(name): tensor for name, tensor in tensors.items()}
        super().__init__({name: tensor.data.copy() for name, tensor in selected.items()})
        self.tensors = selected
        self.provenance = int(provenance)


def processor_namespace_helpers(snapshot: ProcessorSignalSnapshot) -> dict:
    """Snapshot-bound helpers for a Processor SignalExpr.

    Name lookup is intentionally strict: asking for a signal that was not part
    of ``consumes`` raises ``KeyError`` instead of reading it from the Hub.
    """

    def _tensor(name):
        return snapshot.tensors[str(name)]

    return {
        "logical": lambda name: _tensor(name).logical(),
        "valid": lambda name: _tensor(name).valid.copy(),
        "schema": lambda name: _tensor(name).schema,
        "names": lambda: sorted(snapshot.tensors),
        "shot": snapshot.provenance,
        "np": np,
        "numpy": np,
        "math": math,
    }


def namespace_helpers(hub) -> dict:
    """The helper bindings (:data:`NAMESPACE_HELPERS` name -> value) every expression namespace
    carries for ``hub``.  The ONE builder: the console panel, the pulse-scan y, and a reactive
    processor's source all inject exactly this set, so an expression has the SAME capabilities
    everywhere (``numpy`` is a plain alias of ``np`` -- the same alias the pulse-table scan
    namespace ships)."""

    return {
        # Canonical arrays are always explicit physical axes.  ``logical`` is
        # the equally-explicit schema reshape for an image/grid processor; no
        # helper examines ndarray rank to decide what an axis means.
        "history": lambda name, n=None: hub.history(name, n),
        "history_logical": lambda name, n=None: hub.history_logical(name, n),
        "latest": lambda name: hub.latest(name),
        "tensor": lambda name: hub.latest_tensor(name),
        "logical": lambda name: hub.latest_logical(name),
        "valid": lambda name: hub.latest_valid(name),
        "schema": lambda name: hub.schema(name),
        "names": lambda: hub.names(),
        "shot": hub.shot,
        "np": np,
        "numpy": np,
        "math": math,
    }


def hub_namespace(hub, snapshot: Mapping[str, object] | None = None) -> dict:
    """The expression namespace for a :class:`~..core.signals.SignalHub`: a signal snapshot
    (``snapshot`` if given -- e.g. the console's shot-coherent ``snapshot_at`` view or a
    processor's coherent inputs -- else every latest value) with the shared helpers
    (:func:`namespace_helpers`) layered ON TOP.  The single source -- the console's GUI
    namespace only adds its view-only reserved keys on top of this."""

    ns = dict(hub.snapshot_latest() if snapshot is None else snapshot)
    if isinstance(snapshot, ProcessorSignalSnapshot):
        ns.update(processor_namespace_helpers(snapshot))
    else:
        ns.update(namespace_helpers(hub))
    return ns


__all__ = ["SignalExpr", "SIGNAL_EXPR_HELP", "DEFAULT_SOURCE", "NAMESPACE_HELPERS",
           "PROCESSOR_NAMESPACE_HELPERS", "ProcessorSignalSnapshot",
           "seed_source_for_slots", "namespace_helpers", "processor_namespace_helpers",
           "hub_namespace"]
