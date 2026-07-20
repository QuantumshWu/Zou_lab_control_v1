"""MOVED to :mod:`zlc_frontend.live_plot.plot_figure` (shell salvage; dies at Z0).

Renamed on the way: ``zlc_frontend.data_figure`` already exists and means the
STATIC facade over one frozen evaluation, while this module is the legacy
MUTABLE plot figure and its npz save / load / replay.  The moved copy is named
for what it holds; this path keeps the old name for legacy importers.

Every name - including underscore names - resolves to the SAME objects as
the moved module, so legacy imports and isinstance checks keep working.
"""

import zlc_frontend.live_plot.plot_figure as _moved

globals().update(
    {k: v for k, v in vars(_moved).items() if k not in {"__name__", "__file__", "__loader__", "__spec__", "__package__", "__builtins__", "__cached__", "__doc__"}}
)
