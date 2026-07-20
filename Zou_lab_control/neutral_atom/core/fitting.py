"""MOVED to :mod:`zlc_data.curve_fitting` (GUI shell salvage; this shim dies at Z0).

The curve/histogram fit engine.  Qt knows nothing about it; the Qt-thread
solve guard is still REGISTERED into it from the legacy render_loop shim.

Every name - including underscore names - resolves to the SAME objects as
the moved module, so legacy imports and isinstance checks keep working.
"""

import zlc_data.curve_fitting as _moved

globals().update(
    {k: v for k, v in vars(_moved).items() if k not in {"__name__", "__file__", "__loader__", "__spec__", "__package__", "__builtins__", "__cached__", "__doc__"}}
)
