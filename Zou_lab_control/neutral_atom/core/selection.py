"""MOVED to :mod:`zlc_data.plot_region` (GUI shell salvage; this shim dies at Z0).

Renamed on the way: this is the PLOT REGION selection, distinct from the
dataset-level ``zlc_data.Selection``.  See the moved module's docstring.

Every name - including underscore names - resolves to the SAME objects as
the moved module, so legacy imports and isinstance checks keep working.
"""

import zlc_data.plot_region as _moved

def __getattr__(name):
    """Resolve every name LIVE from the moved module (PEP 562 covers ``from`` imports).

    NOT ``globals().update(vars(_moved))``: that copies BINDINGS once at import, so a
    global the moved module later rebinds leaves this file frozen -- which is how
    ``core/fitting._solve_thread_guard`` stayed ``None`` after the frontend armed the
    real guard.  Names assigned in this module's body still shadow this hook.
    """

    return getattr(_moved, name)


def __dir__():
    """Keep ``dir()``/tab-completion alive: the module dict is now nearly empty."""

    return sorted(set(dir(_moved)) | set(globals()))

# zlc_data reserves the ``decode_``/``encode_`` prefixes for canonical byte
# admission, so the moved module names this transport pair ``*_payload``.
# Legacy callers keep the old names, bound to the SAME objects.
encode_region = _moved.region_to_payload
decode_region = _moved.region_from_payload
