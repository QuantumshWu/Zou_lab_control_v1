"""MOVED to :mod:`zlc_frontend.live_plot._watcher` (GUI shell salvage; dies at Z0).

The background thread that polls a bound array and republishes it to a live
plot: stdlib threading over the validators it shares with the rest of the
live-plot family, so it moves with them.

Every name - including underscore names - resolves to the SAME objects as
the moved module, so legacy imports and isinstance checks keep working.
"""

import zlc_frontend.live_plot._watcher as _moved

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
