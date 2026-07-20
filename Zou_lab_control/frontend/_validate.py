"""MOVED to :mod:`zlc_frontend.live_plot._validate` (GUI shell salvage; this shim dies at Z0).

Every name - including underscore names - resolves to the SAME objects as the
moved module, so legacy imports and isinstance checks keep working unchanged.
"""

import zlc_frontend.live_plot._validate as _moved

globals().update(
    {k: v for k, v in vars(_moved).items() if k not in {"__name__", "__file__", "__loader__", "__spec__", "__package__", "__builtins__", "__cached__", "__doc__"}}
)

# The import-DAG guard reserves ``_positive_float``/``_positive_int`` for
# ``zlc_storage.canonical``, so the moved module renamed them; legacy callers
# keep the old names here, bound to the SAME objects.
_positive_float = _moved.positive_float
_positive_int = _moved.positive_int
