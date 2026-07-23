"""The words the domain and the render layer must spell identically.

Every name here was already a single source inside the legacy tree - each one
carries a comment saying so, some of them emphatically ("the frontend imports
THIS, never a retyped copy").  The problem was never duplication; it was
PLACEMENT.  They lived beside the domain objects that happen to use them, so a
form that wanted to label a sweep kind, or a Setting popup that wanted to list
the analysis actions, had to reach into ``neutral_atom.operations`` and drag the
measurement stack in behind it.

A string is not a domain object.  These are pure vocabulary - literals with no
behaviour, no imports and no state - so they sink to the bottom of the DAG where
both sides can read them without either side importing the other.  The domain
modules that used to define them now import them from here, which keeps the
single source exactly where it always was: one definition, one spelling.

The sink/port rule that put them here: a pure function, constant or description
moves into ``zlc_data``; a LIVE object the render layer cannot construct gets an
inverted port in ``zlc_frontend.domain_ports`` instead.  Nothing in this module
may ever grow a behaviour, an import or a mutable default - the moment it needs
one it is not vocabulary and it does not belong here.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_MID_RUN_KEY",
    "NO_LINEAGE",
    "PULSE_SWEEP_KINDS",
    "SWEEP_API_SLOT",
    "SWEEP_SCAN_SLOT",
]


#: A value that joins no physical acquisition lineage (static metadata or a
#: free-running external value).  Data-plane producers should normally provide
#: a real source-shot id; the sentinel remains explicit rather than inferred.
NO_LINEAGE = -1

#: The two execution semantics supported by PulseScan.  A hardware scan slot is
#: uploaded as one complete table; an API slot is resolved and submitted as one
#: finite pulse per point.  One discriminator for the measurement factory, the
#: logic node and the frontend, instead of parallel mode vocabularies.
SWEEP_SCAN_SLOT = "scan_slot"
SWEEP_API_SLOT = "api_slot"
PULSE_SWEEP_KINDS: tuple[str, str] = (SWEEP_SCAN_SLOT, SWEEP_API_SLOT)

#: The default mid-run buffer key a task streams (``TaskSpec.mid_run_key``'s
#: default) -- the ONE spelling of ``"frame"``.  A consumer that must fall back
#: without a spec in hand imports this instead of re-typing the literal.
DEFAULT_MID_RUN_KEY = "frame"
