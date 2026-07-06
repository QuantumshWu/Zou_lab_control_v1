"""Pulse-payload persistence: load a saved pulse JSON back into its owning class.

Both persistable pulse models -- :class:`PulseSequence` (the physical timing
truth) and :class:`PulseTableState` (the GUI compile model) -- stamp a
class-owned ``schema`` string into ``to_dict``.  This module is the ONE reader
of that on-disk contract: it dispatches a payload on the same class attribute
the writers use, so the schema string keeps a single home (the class) and the
data layer owns its own persistence dispatch, right next to the two classes.
"""

from __future__ import annotations

import json
from pathlib import Path

from .pulse_table import PulseTableState
from .sequence import PulseSequence


def load_pulse_payload(pulse: PulseSequence | PulseTableState | str | Path) -> PulseSequence | PulseTableState:
    """Return an in-memory pulse payload as-is, or load + dispatch a pulse JSON path."""

    if isinstance(pulse, (PulseSequence, PulseTableState)):
        return pulse
    path = Path(pulse)
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema", "")
    # BOTH payload kinds dispatch on their class-owned ``schema`` attribute (the single
    # source each class also writes in ``to_dict``) -- never a retyped literal here.
    if schema == PulseTableState.schema:
        return PulseTableState.from_dict(payload)
    if schema == PulseSequence.schema:
        return PulseSequence.from_dict(payload)
    raise ValueError(f"unsupported pulse JSON schema in {path}: {schema!r}")


__all__ = ["load_pulse_payload"]
