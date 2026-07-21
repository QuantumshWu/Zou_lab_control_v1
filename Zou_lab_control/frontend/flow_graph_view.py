"""MOVED to :mod:`zlc_workbench.figure_viewer.flow_graph_view` (window W3 decomposition; this shim dies at Z0).

Every name resolves LIVE to the SAME objects as the moved module.
"""

import zlc_workbench.figure_viewer.flow_graph_view as _moved


def __getattr__(name):
    return getattr(_moved, name)


def __dir__():
    return sorted(set(dir(_moved)) | set(globals()))
