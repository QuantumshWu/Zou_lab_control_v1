"""MOVED to :mod:`zlc_storage.paths` (GUI shell salvage; this shim dies at Z0).

Every name - including underscore names - resolves to the SAME objects as the
moved module, so legacy imports and isinstance checks keep working unchanged.
"""

import zlc_storage.paths as _moved

globals().update(
    {k: v for k, v in vars(_moved).items() if k not in {"__name__", "__file__", "__loader__", "__spec__", "__package__", "__builtins__", "__cached__", "__doc__"}}
)
