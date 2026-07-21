"""MOVED to :mod:`zlc_neutral_atom.timing.pulse_table` (backend clearance; this shim dies at Z0).

Every name - including underscore names - resolves to the SAME objects as the moved
module, so legacy imports and isinstance checks keep working unchanged.
"""

import zlc_neutral_atom.timing.pulse_table as _moved


def __getattr__(name):
    """Resolve every name LIVE from the moved module (PEP 562 covers ``from`` imports)."""

    return getattr(_moved, name)


def __dir__():
    """Keep ``dir()``/tab-completion alive: the module dict is now nearly empty."""

    return sorted(set(dir(_moved)) | set(globals()))
