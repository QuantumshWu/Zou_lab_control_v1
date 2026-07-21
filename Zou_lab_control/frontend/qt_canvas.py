"""MOVED to :mod:`zlc_workbench.task_console.plot_bridge_canvas` (shell salvage; dies at Z0).

The embedded matplotlib canvas is the definitive Qt x matplotlib marriage, so its home is the
task console's plot_bridge zone.  This path keeps the old name for legacy importers.

Every name - including underscore names - resolves to the SAME objects as the moved module,
so legacy imports and isinstance checks keep working.
"""

import zlc_workbench.task_console.plot_bridge_canvas as _moved


def __getattr__(name):
    """Resolve every name LIVE from the moved module (PEP 562 covers ``from`` imports)."""

    return getattr(_moved, name)


def __dir__():
    """Keep ``dir()``/tab-completion alive: the module dict is now nearly empty."""

    return sorted(set(dir(_moved)) | set(globals()))
