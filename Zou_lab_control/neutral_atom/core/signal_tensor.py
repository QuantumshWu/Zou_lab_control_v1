"""MOVED to :mod:`zlc_data.signal_tensor` (GUI shell salvage; this shim dies at Z0).

Pure numeric/declarative code with no package dependencies, so it belongs
in the data layer that both ``zlc_neutral_atom`` and ``zlc_frontend`` may
import.  Every name - including underscore names - resolves to the SAME
objects as the moved module, so legacy imports and isinstance checks keep
working unchanged.
"""

import zlc_data.signal_tensor as _moved

globals().update(
    {k: v for k, v in vars(_moved).items() if k not in {"__name__", "__file__", "__loader__", "__spec__", "__package__", "__builtins__", "__cached__", "__doc__"}}
)
