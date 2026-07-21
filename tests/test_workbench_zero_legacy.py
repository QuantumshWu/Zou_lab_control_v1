"""The user's acceptance criterion, mechanised: opening a window loads ZERO legacy.

Directive 2026-07-21: the GUI keeps only the UI skeleton (``zlc_frontend.qt_widgets``);
every data plane is the current zlc_* stack; the legacy trees are dead NOW.  The proof
is runtime, not AST: each composition root is imported in a FRESH interpreter and the
loaded-module table must contain nothing from the legacy trees.  An AST scan cannot see
a transitive pull-in through a helper; ``sys.modules`` sees everything that actually
loaded, which is exactly what "no legacy" means to the person testing the window.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

#: Every window UI BODY that has been rewired (the kept ORIGINAL UI skeletons on the
#: current data plane), plus the retained entry facades.  figure_viewer's body joins
#: this roster the commit its domain imports are replaced -- it may not ship before.
ROOTS = (
    "zlc_workbench.task_console.plot_bridge_console",
    "zlc_workbench.pulse_editor.plot_bridge_pulse_gui",
    "zlc_workbench.task_console.app",
    "zlc_workbench.pulse_editor.app",
    "Zou_lab_control.workbench",
    "Zou_lab_control.notebook",
)

#: What "legacy" means: the two dead subtrees and the dead top-level seams.  The root
#: package itself and its two retained entry facades (notebook/, workbench/) stay.
LEGACY_PREFIXES = (
    "Zou_lab_control.frontend",
    "Zou_lab_control.neutral_atom",
    "Zou_lab_control._clock",
    "Zou_lab_control._paths",
    "Zou_lab_control._readout_math",
    "Zou_lab_control._streamer_geometry",
    "Zou_lab_control._viewer_registry",
)


@pytest.mark.parametrize("module", ROOTS)
def test_importing_a_composition_root_loads_zero_legacy_modules(module):
    code = (
        "import os, sys, json\n"
        "os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')\n"
        "os.environ.setdefault('MPLBACKEND', 'Agg')\n"
        f"import {module}\n"
        f"prefixes = {LEGACY_PREFIXES!r}\n"
        "loaded = sorted(m for m in sys.modules\n"
        "                if any(m == p or m.startswith(p + '.') for p in prefixes))\n"
        "print(json.dumps(loaded))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=REPO)
    assert result.returncode == 0, f"{module} failed to import:\n{result.stderr[-1500:]}"
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == [], (
        f"{module} pulled {len(loaded)} legacy modules back in:\n  "
        + "\n  ".join(loaded)
    )
