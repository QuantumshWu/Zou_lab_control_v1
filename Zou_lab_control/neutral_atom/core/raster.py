"""MOVED to :mod:`zlc_data.raster` (GUI shell salvage; this shim dies at Z0).

A regular sampling raster: stdlib + numpy only, consumed from both the
domain and the display side.

Every name - including underscore names - resolves to the SAME objects as
the moved module, so legacy imports and isinstance checks keep working.
"""

import zlc_data.raster as _moved

globals().update(
    {k: v for k, v in vars(_moved).items() if k not in {"__name__", "__file__", "__loader__", "__spec__", "__package__", "__builtins__", "__cached__", "__doc__"}}
)
