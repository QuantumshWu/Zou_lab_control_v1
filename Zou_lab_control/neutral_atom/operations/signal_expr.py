"""MOVED to :mod:`zlc_data.signal_expr` (shell salvage; this shim dies at Z0).

The value-expression language a panel or a logic node writes over published
signals - parsing, the identity fast path, the helper namespace and the help
text.  272 lines of stdlib: it describes and evaluates expressions over whatever
mapping it is handed, and never touches a device, a hub implementation or a
plot.  A describer belongs in zlc_data, not under ``operations``.

Every name - including underscore names - resolves to the SAME objects as
the moved module, so legacy imports and isinstance checks keep working.
"""

import zlc_data.signal_expr as _moved

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
