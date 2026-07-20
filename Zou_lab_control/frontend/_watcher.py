"""MOVED to :mod:`zlc_frontend.live_plot._watcher` (GUI shell salvage; dies at Z0).

The background thread that polls a bound array and republishes it to a live
plot: stdlib threading over the validators it shares with the rest of the
live-plot family, so it moves with them.

Every name - including underscore names - resolves to the SAME objects as
the moved module, so legacy imports and isinstance checks keep working.
"""

import zlc_frontend.live_plot._watcher as _moved

globals().update(
    {k: v for k, v in vars(_moved).items() if k not in {"__name__", "__file__", "__loader__", "__spec__", "__package__", "__builtins__", "__cached__", "__doc__"}}
)
