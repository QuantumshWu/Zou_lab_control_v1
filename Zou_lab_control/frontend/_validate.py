"""Frontend-private scalar validators -- the SINGLE source for the tiny "reject a
bad public scalar" checks the notebook-facing entry points share.

``plot()`` / ``ArrayWatcher`` / the acquisition ``Session`` each take a few public
scalars (a refresh interval, a flag, a count) that must be rejected the SAME way
everywhere: a bool is never a number, a float must be finite and > 0, an int must
be a true integer.  Keeping one copy here means changing the rule (loosen 0, edit
a message) touches one file, not three -- the byte-for-byte copies that used to
live in ``session.py`` / ``live.py`` / ``_watcher.py`` could (and would) drift.

Dependency-free (stdlib + numpy only) so every ``frontend`` module imports it
without coupling -- the SAME seam pattern as ``_paths`` / ``_readout_math`` /
``_viewer_registry``.  It never reaches across the sealed ``frontend`` <-> ``neutral_atom``
boundary (the analysis layer has its OWN validators in ``core/analysis.py``).
"""

from __future__ import annotations

import numpy as np


def _strict_bool(value, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise TypeError(f"{name} must be a boolean.")


def _positive_float(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be finite, not a boolean.")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and > 0.")
    return result


def _positive_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer, not a boolean.")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _non_negative_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a non-negative integer, not a boolean.")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result
