"""MOVED to :mod:`zlc_frontend.live_plot.live` (shell salvage; dies at Z0).

The live plot classes and the plot-kind registry - the biggest single piece of
the old display layer.  It could only cross once its domain tendrils were gone
(H1a-H1c) AND the two frontend modules it imports had crossed with it (H1d):
a copy in ``zlc_frontend`` that reached back into ``Zou_lab_control`` would
break the same import-DAG rule the tendrils were cut for.

Every name - including underscore names - resolves to the SAME objects as
the moved module, so legacy imports and isinstance checks keep working.
"""

import zlc_frontend.live_plot.live as _moved

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
