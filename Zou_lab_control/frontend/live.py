"""MOVED to :mod:`zlc_frontend.live_plot.live` (shell salvage; dies at Z0).

The live plot classes and the plot-kind registry - the biggest single piece of
the old display layer.  It could only cross once its domain tendrils were gone
(H1a-H1c) AND the two frontend modules it imports had crossed with it (H1d):
a copy in ``zlc_frontend`` that reached back into ``Zou_lab_control`` would
break the same import-DAG rule the tendrils were cut for.

Every name - including underscore names - resolves to the SAME objects as
the moved module, so legacy imports and isinstance checks keep working.
"""

import zlc_frontend.live_plot.live as _moved

globals().update(
    {k: v for k, v in vars(_moved).items() if k not in {"__name__", "__file__", "__loader__", "__spec__", "__package__", "__builtins__", "__cached__", "__doc__"}}
)
