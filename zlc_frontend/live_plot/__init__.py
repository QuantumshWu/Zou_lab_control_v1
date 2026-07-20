"""Matplotlib plot infrastructure salvaged verbatim from the legacy frontend.

Canvas geometry, selector overlays, smart ticks and the scalar validators -
the Qt-free part of the old GUI display layer - moved here unchanged, so the
legacy windows and the migrated windows run on ONE copy.  The legacy paths
under ``Zou_lab_control/frontend/`` are import shims onto this package and
die at Z0.

Deliberately NOT here: ``render_loop.py``/``qt_canvas.py`` (the in-process
QtAgg render model).  zlc_frontend has two mechanical axioms - qt_widgets is
the ONLY Qt owner, and qt_widgets is matplotlib-free - so a Qt+matplotlib
marriage has no legal home in this package.  That is by design: the target
render model is worker-raster (S12.5), and those two stay legacy-side until
the shells adopt it.  ``live.py`` (the plot classes) joins this package once
its domain tendrils are ported - see tests/test_u05_shell_salvage.py.
"""
