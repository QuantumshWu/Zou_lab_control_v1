"""Parameter declaration via function signatures -- the single source a GUI form and
the API default both derive from.

A measurement / processor / task is a plain callable whose KEYWORD-ONLY parameters
carry typed annotations (``Param`` / ``Choice`` / ``ScanArray`` / ``SignalRef`` /
``PulseScan``).  ``params_from_signature(fn)`` introspects the signature into a list
of :class:`ParamField` records; an auto-generated GUI form renders one widget per
field, and the API call uses the SAME defaults -- so the two can never drift, exactly
like Confocal_GUIv2 derives its form from the ``caller`` signature.

This module is backend-neutral (pure ``inspect`` + dataclasses, no device/frontend
imports).  The frontend reads ``ParamField`` attributes by duck typing, so it never
imports this module at load time -- the decoupling contract stays intact.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


# ---- annotation spec objects (used AS a parameter's annotation in a signature) -----
@dataclass(frozen=True)
class Param:
    """A scalar parameter: ``kind`` is ``float`` / ``int`` / ``bool`` / ``str``."""
    kind: type = float
    unit: str = ""
    lo: float | None = None
    hi: float | None = None
    label: str = ""


@dataclass(frozen=True)
class Choice:
    """A one-of-many parameter rendered as a dropdown."""
    options: tuple = ()
    label: str = ""


@dataclass(frozen=True)
class ScanArray:
    """A 1-D scan array (start/stop/step or loaded), e.g. a measurement sweep axis."""
    unit: str = ""
    label: str = ""


@dataclass(frozen=True)
class SignalRef:
    """A reference to a hub signal name (rendered as a signal picker)."""
    label: str = ""


@dataclass(frozen=True)
class PulseScan:
    """Bind a swept parameter to a pulse slot/target (rendered as pulse + range)."""
    pulse: str = ""
    target: str = ""
    unit: str = ""
    label: str = ""


# ---- the rendered field record (what the GUI form / API both consume) --------------
@dataclass(frozen=True)
class ParamField:
    name: str
    kind: str                       # float|int|bool|str|choice|array|signal|pulse_scan
    default: Any = None
    unit: str = ""
    choices: tuple = ()
    lo: float | None = None
    hi: float | None = None
    target: str = ""                # pulse_scan: "<pulse>.<target>"
    label: str = ""
    meta: dict = field(default_factory=dict)


_TYPE_KIND = {float: "float", int: "int", bool: "bool", str: "str"}
# Parameter names every producer takes by INJECTION (devices / wiring), never a UI field.
INJECTED = frozenset({"self", "cls", "hub", "camera", "sequencer", "out", "calibration", "prefix"})


def _coerce_default(spec_kind: str, default: Any) -> Any:
    if default is None:
        return None
    try:
        if spec_kind == "int":
            return int(default)
        if spec_kind == "float":
            return float(default)
        if spec_kind == "bool":
            return bool(default)
    except (TypeError, ValueError):
        return default
    return default


def _field_from(name: str, annotation: Any, default: Any) -> ParamField | None:
    """Build a ParamField from one parameter's annotation + default, or None to skip."""
    label = ""
    if isinstance(annotation, Param):
        kind = _TYPE_KIND.get(annotation.kind, "float")
        return ParamField(name, kind, _coerce_default(kind, default), unit=annotation.unit,
                          lo=annotation.lo, hi=annotation.hi, label=annotation.label or name)
    if isinstance(annotation, Choice):
        options = tuple(annotation.options)
        chosen = default if default is not None else (options[0] if options else None)
        return ParamField(name, "choice", chosen, choices=options, label=annotation.label or name)
    if isinstance(annotation, ScanArray):
        return ParamField(name, "array", default, unit=annotation.unit, label=annotation.label or name)
    if isinstance(annotation, SignalRef):
        return ParamField(name, "signal", "" if default is None else str(default), label=annotation.label or name)
    if isinstance(annotation, PulseScan):
        tgt = f"{annotation.pulse}.{annotation.target}" if annotation.pulse else annotation.target
        return ParamField(name, "pulse_scan", default, unit=annotation.unit, target=tgt,
                          label=annotation.label or name)
    # plain type annotation (int/float/bool/str)
    if annotation in _TYPE_KIND:
        kind = _TYPE_KIND[annotation]
        return ParamField(name, kind, _coerce_default(kind, default), label=name)
    # no/unknown annotation: infer from the default's type when there is one
    if default is not None:
        kind = _TYPE_KIND.get(type(default))
        if kind is not None:
            return ParamField(name, kind, default, label=name)
    return None   # an injected positional (device/hub) -- not a UI field


def params_from_signature(fn: Callable, *, skip: tuple = ()) -> list[ParamField]:
    """Introspect a callable's KEYWORD parameters into ``ParamField`` records.

    Positional injected args (``hub`` / ``camera`` / ``sequencer`` / ``out`` / ...)
    and ``*args`` / ``**kwargs`` are skipped, so what remains is exactly the user-
    tunable parameters -- the same set the API caller passes and the GUI form shows."""
    skip_all = INJECTED | set(skip)
    fields: list[ParamField] = []
    # The codebase uses ``from __future__ import annotations`` (PEP 563), so raw
    # annotations are STRINGS; eval_str resolves the spec objects in the callable's
    # own module globals.  Fall back to string annotations (defaults-only inference)
    # on older Pythons or an unresolvable annotation.
    try:
        signature = inspect.signature(fn, eval_str=True)
    except Exception:
        signature = inspect.signature(fn)
    for name, parameter in signature.parameters.items():
        if name in skip_all or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        default = None if parameter.default is inspect._empty else parameter.default
        built = _field_from(name, parameter.annotation, default)
        if built is not None:
            fields.append(built)
    return fields


__all__ = [
    "Param", "Choice", "ScanArray", "SignalRef", "PulseScan",
    "ParamField", "params_from_signature", "INJECTED",
]
