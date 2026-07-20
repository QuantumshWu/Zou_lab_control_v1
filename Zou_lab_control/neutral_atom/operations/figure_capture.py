"""MOVED to :mod:`zlc_data.figure_capture` (GUI shell salvage; this shim dies at Z0).

What a figure save folds into ``info`` - the signal blocks, the device
provenance and the flow graph - described purely from the duck-typed
hub / node / resolver it is handed.  stdlib + numpy only, and already
guarded frontend-neutral, so it is a data-plane description, not an
operation: it belongs beside the other pure describers in ``zlc_data``.

Every name - including underscore names - resolves to the SAME objects as
the moved module, so legacy imports and isinstance checks keep working.
"""

import zlc_data.figure_capture as _moved

globals().update(
    {k: v for k, v in vars(_moved).items() if k not in {"__name__", "__file__", "__loader__", "__spec__", "__package__", "__builtins__", "__cached__", "__doc__"}}
)
