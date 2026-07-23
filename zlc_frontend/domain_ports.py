"""What the render layer needs the DOMAIN to build for it, and nothing else.

The import DAG makes ``zlc_frontend`` a leaf over ``zlc_data``: it may never
reach ``zlc_pulse``, ``zlc_neutral_atom`` or the legacy tree.  That is the right
rule - a renderer should not be able to reach the hardware - but a few render
paths genuinely need a live domain OBJECT, not a description of one.

Reopening a saved pulse figure is the case that forced this module.  The npz
holds the editor's state as a plain dict; drawing the timeline needs a real
``PulseTableState``, because the whole point of the replay is that the notebook,
the editor preview and a seeded console panel all go through the ONE renderer
with the SAME object.  The renderer cannot construct it and must not import the
compiler that can.

So the dependency is inverted: the composition root - the layer that already
imports both sides - hands the constructor down at startup, and the render layer
calls it through this module.  Registration is a one-line call, deliberately
explicit: an unregistered process gets a typed refusal that names what to do,
never a silent half-drawn figure.

Adding a port here is a decision, not a convenience.  A seam belongs in this
module only when the render layer needs a LIVE object it cannot build.  Anything
that is a pure function, a constant, or a description moves into ``zlc_data``
instead - see ``test_u05_shell_salvage`` for that route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

__all__ = [
    "PulseReplayUnavailable",
    "PulseTemplateRows",
    "PulseTemplateUnavailable",
    "pulse_state_from_dict",
    "pulse_state_factory_is_registered",
    "pulse_template_reader_is_registered",
    "pulse_template_rows",
    "register_pulse_state_factory",
    "register_pulse_template_reader",
]


class PulseReplayUnavailable(RuntimeError):
    """Raised when a pulse figure is replayed with no pulse domain wired in."""


_REFUSAL = (
    "reopening a saved pulse figure needs the pulse domain, which the render "
    "layer is not allowed to import.  The composition root must call "
    "zlc_frontend.domain_ports.register_pulse_state_factory(...) with a "
    "callable that turns a saved pulse-state mapping into a PulseTableState "
    "(PulseTableState.from_dict).  Every other figure kind replays without it."
)

_PULSE_STATE_FACTORY: Callable[[Mapping[str, Any]], Any] | None = None


def register_pulse_state_factory(factory: Callable[[Mapping[str, Any]], Any]) -> None:
    """Hand the render layer its pulse-state constructor.

    Idempotent by intent - a composition root may be imported more than once -
    but a SECOND, DIFFERENT factory is refused: two constructors would mean two
    answers to "what did this saved pulse look like", which is exactly the
    drift this port exists to prevent.
    """

    global _PULSE_STATE_FACTORY
    if not callable(factory):
        raise TypeError("the pulse-state factory must be callable")
    if _PULSE_STATE_FACTORY is not None and _PULSE_STATE_FACTORY is not factory:
        raise RuntimeError(
            "a different pulse-state factory is already registered; the "
            "reproduction of a saved pulse has ONE source"
        )
    _PULSE_STATE_FACTORY = factory


def pulse_state_factory_is_registered() -> bool:
    """Whether a pulse figure can be replayed in this process."""

    return _PULSE_STATE_FACTORY is not None


def pulse_state_from_dict(data: Mapping[str, Any]) -> Any:
    """Rebuild the editor state a saved pulse figure was drawn from."""

    if _PULSE_STATE_FACTORY is None:
        raise PulseReplayUnavailable(_REFUSAL)
    if not isinstance(data, Mapping):
        raise TypeError("a saved pulse state must be a mapping")
    return _PULSE_STATE_FACTORY(data)


# ------------------------------------------------- the pulse-template row port

@dataclass(frozen=True)
class PulseTemplateRows:
    """Everything the pulse-slots form needs to draw ONE saved template.

    Plain tuples and strings only.  The console used to reach into a live
    ``PulseTableState`` for this - ``api_slots``, ``scan_slots``, the private
    ``_read_api_field``, ``scan_code``, ``scan_table``, ``to_dict`` - which put
    the whole pulse vocabulary inside the render layer.  The derivation now runs
    domain-side and the GUI receives a DESCRIPTION, so the form can be drawn by a
    process that has never imported the pulse compiler.

    ``api_rows``  : (handle, coordinate, kind, target, unit, current)
    ``scan_rows`` : (coordinate, kind, target, unit, stored_label)
    ``*_columns`` : the finished ``ScanColumnSpec`` per slot.  Building one needs
                    the bus signed range, the unit table and the clock tick, so it
                    is derived DOMAIN-side; the form receives the columns and only
                    hands them to the (pure) template writer.
    ``program``   : the hardware scan program text, or the array literal a stored
                    scan table falls back to.
    ``program_id``: a stable digest of the resolved source-document identity, so
                    content edits can reconcile surviving slot ids while a
                    different source cannot inherit their drafts.
    """

    api_rows: tuple = ()
    scan_rows: tuple = ()
    api_columns: tuple = ()
    scan_columns: tuple = ()
    program: str = ""
    program_id: str = ""


class PulseTemplateUnavailable(RuntimeError):
    """Raised when a pulse template is read with no pulse domain wired in."""


_TEMPLATE_REFUSAL = (
    "reading a saved pulse template needs the pulse domain, which the render "
    "layer is not allowed to import.  The composition root must call "
    "zlc_frontend.domain_ports.register_pulse_template_reader(...) with a "
    "callable that turns a template path into a PulseTemplateRows."
)

_PULSE_TEMPLATE_READER: Callable[[str], PulseTemplateRows] | None = None


def register_pulse_template_reader(reader: Callable[[str], PulseTemplateRows]) -> None:
    """Hand the render layer its pulse-template reader.

    Same single-source rule as the state factory: idempotent for the SAME
    callable, and a second DIFFERENT reader is refused, because two readers would
    mean two answers to "what is in this template".
    """

    global _PULSE_TEMPLATE_READER
    if not callable(reader):
        raise TypeError("the pulse-template reader must be callable")
    if _PULSE_TEMPLATE_READER is not None and _PULSE_TEMPLATE_READER is not reader:
        raise RuntimeError(
            "a different pulse-template reader is already registered; a saved "
            "template has ONE reading"
        )
    _PULSE_TEMPLATE_READER = reader


def pulse_template_reader_is_registered() -> bool:
    """Whether a pulse template can be read in this process."""

    return _PULSE_TEMPLATE_READER is not None


def pulse_template_rows(path) -> PulseTemplateRows:
    """The rows a saved pulse template contributes to the slots form."""

    if _PULSE_TEMPLATE_READER is None:
        raise PulseTemplateUnavailable(_TEMPLATE_REFUSAL)
    rows = _PULSE_TEMPLATE_READER(str(path or ""))
    if not isinstance(rows, PulseTemplateRows):
        raise TypeError("the pulse-template reader must return PulseTemplateRows")
    return rows
