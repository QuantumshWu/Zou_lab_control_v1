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

globals().update(
    {k: v for k, v in vars(_moved).items() if k not in {"__name__", "__file__", "__loader__", "__spec__", "__package__", "__builtins__", "__cached__", "__doc__"}}
)
