"""Contract: importing the package enables Qt high-DPI scaling via the environment variable.

Root cause it guards (verified on a real 2.5x screen): in a Jupyter kernel, ipykernel's ``%gui qt``
integration creates the QApplication BEFORE any of our code runs, so the ``AA_EnableHighDpiScaling``
attribute we set inside ``ensure_qt_app`` is too late (it must be set pre-construction).  On a
fractional-DPI screen the app then comes up in physical-pixel mode (devicePixelRatio collapses to
1.0): the window balloons to the whole screen while the fixed-inch live plots stay small, so a
task's Monitor image renders tiny.  ``QT_ENABLE_HIGHDPI_SCALING`` is honoured by Qt at QApplication
construction no matter WHO constructs it, so importing the package first (the notebook always does)
yields correctly scaled GUIs -- matching the standalone ``.bat``.  This pins the line in
``Zou_lab_control/__init__`` so it can't be silently dropped.
"""

from __future__ import annotations

import os


def test_package_import_enables_qt_high_dpi_scaling():
    import Zou_lab_control  # noqa: F401 -- the import runs the package __init__ that sets it

    assert os.environ.get("QT_ENABLE_HIGHDPI_SCALING") == "1"
