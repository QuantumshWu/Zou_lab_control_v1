"""The pulse editor application: the window lives in :mod:`.app`.

Importing this package stays Qt-free (C13).  Callers reach the window explicitly --
``from zlc_workbench.pulse_editor.app import open_pulse_editor`` -- so nothing headless drags
a GUI toolkit in behind its back.
"""

from __future__ import annotations
