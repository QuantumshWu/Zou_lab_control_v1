"""MOVED to :mod:`zlc_data.plot_region` (GUI shell salvage; this shim dies at Z0).

Renamed on the way: this is the PLOT REGION selection, distinct from the
dataset-level ``zlc_data.Selection``.  See the moved module's docstring.

Every name - including underscore names - resolves to the SAME objects as
the moved module, so legacy imports and isinstance checks keep working.
"""

import zlc_data.plot_region as _moved

globals().update(
    {k: v for k, v in vars(_moved).items() if k not in {"__name__", "__file__", "__loader__", "__spec__", "__package__", "__builtins__", "__cached__", "__doc__"}}
)

# zlc_data reserves the ``decode_``/``encode_`` prefixes for canonical byte
# admission, so the moved module names this transport pair ``*_payload``.
# Legacy callers keep the old names, bound to the SAME objects.
encode_region = _moved.region_to_payload
decode_region = _moved.region_from_payload
