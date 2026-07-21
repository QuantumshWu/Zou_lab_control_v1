"""The task console application: headless documents here, the window in :mod:`.app`.

Importing this package must stay Qt-free (C13).  ``app`` is where PyQt enters, and callers
reach it explicitly -- ``from zlc_workbench.task_console.app import open_task_console`` -- so
a notebook doing analysis never drags a GUI toolkit in behind its back.

The star-import is deliberate: this facade publishes EXACTLY what the document module
publishes, and spelling the names out again would be a second list to drift.  ``import *``
honours ``intent.__all__``, so there is one place that decides.
"""

from __future__ import annotations

from .intent import *          # noqa: F401,F403 -- re-export, governed by intent.__all__
from .intent import __all__    # noqa: F401 -- and the same list decides ours
