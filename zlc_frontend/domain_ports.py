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

from typing import Any, Callable, Mapping

__all__ = [
    "PulseReplayUnavailable",
    "pulse_state_from_dict",
    "pulse_state_factory_is_registered",
    "register_pulse_state_factory",
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
