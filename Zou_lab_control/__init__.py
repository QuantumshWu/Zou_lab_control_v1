"""Zou lab control package."""

# High-DPI scaling must be enabled BEFORE the first QApplication is constructed.  In a Jupyter
# kernel that app is frequently created by ipykernel's ``%gui qt`` integration -- BEFORE any of
# our code runs -- so setting the ``AA_EnableHighDpiScaling`` attribute inside ``ensure_qt_app``
# is too late and has no effect.  This env var is honoured by Qt at QApplication construction no
# matter WHO constructs it, so importing this package first (the notebook always does) yields
# correctly scaled GUIs -- matching the standalone ``.bat``.  Without it, on a fractional-DPI
# screen (e.g. 2.5x) Qt falls back to physical-pixel mode: the window balloons to the full screen
# while the fixed-inch live plots stay small, so a task's Monitor image looks tiny.  ``setdefault``
# never overrides a value the user set deliberately.
import os as _os
_os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
