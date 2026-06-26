"""Contract: a design param panel never shows a file/folder field you cannot interpret.

Every ``kind="path"`` field renders an UNAMBIGUOUS, project-anchored path -- an absolute
path, never a bare CWD-relative name like ``calibrations`` and never a blank mystery box
(the occupancy processor's calibration file defaults to the canonical calibration.json the
Calibrate task writes, so the file in use is always named).  This pins the user's two
requirements: (1) explicit, readable defaults in every file param field; (2) one shared
path helper (:mod:`Zou_lab_control._paths`) so display + on-disk resolution agree.

Offscreen Qt + virtual backend (the same contract path real hardware takes).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


def test_paths_helper_resolves_under_project_and_displays_absolute():
    from Zou_lab_control._paths import (PROJECT_ROOT, display_path, project_path,
                                        resolve_under_project)
    assert (PROJECT_ROOT / "Zou_lab_control").is_dir()        # the package sits under the root
    # a RELATIVE path is anchored to the PROJECT root and shown absolute (never CWD-relative)
    d = display_path("calibrations")
    assert Path(d).is_absolute() and Path(d) == PROJECT_ROOT / "calibrations"
    assert resolve_under_project("calibrations") == PROJECT_ROOT / "calibrations"
    # blank in -> blank out (an intentionally empty field, e.g. "use session calibration")
    assert display_path("") == "" and display_path(None) == ""
    # an absolute path (even outside the project) is preserved verbatim
    abs_in = os.path.abspath(os.path.join(os.sep, "data", "run"))
    assert display_path(abs_in) == str(Path(abs_in))
    assert project_path("pulses", "x.json") == PROJECT_ROOT / "pulses" / "x.json"


@pytest.fixture(autouse=True)
def _offscreen(monkeypatch):
    pytest.importorskip("PyQt5")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def _console():
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.task_console import TaskConsole, default_console_state
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (48, 60)})
    exp.readout.sitemap(method="box", frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    console = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                          measurements=exp.readout.measurement_specs(),
                          processors=exp.readout.processor_specs(),
                          tasks=exp.readout.task_specs(), window_px=(900, 600))
    console._timer.stop()
    return exp, console


def _open_editor(console, kind_key):
    kc = console.kind_combo
    i = next(j for j in range(kc.count()) if kc.itemData(j) == kind_key)
    kc.setCurrentIndex(i)
    console._add_panel()
    row = console.logic_nodes[-1]
    console._edit_logic_node(row)
    return console._logic_editors[id(row)]


def test_calibrate_task_path_fields_show_absolute_project_paths():
    """The Calibrate-readout task's ``folder`` + ``pulse template`` fields open prefilled
    with the FULL project path (so the operator sees exactly which folder/json), not a bare
    name resolved against whatever CWD Python was launched from."""
    from Zou_lab_control._paths import PROJECT_ROOT

    exp, console = _console()
    try:
        spec = exp.readout.task_specs()[0]
        editor = _open_editor(console, ("task", spec.name))
        folder = editor.form._widgets["folder"].text()
        template = editor.form._widgets["pulse_template"].text()
        assert Path(folder).is_absolute() and Path(folder) == PROJECT_ROOT / "calibrations"
        assert folder != "calibrations"                       # never the bare ambiguous name
        assert Path(template).is_absolute()
        assert Path(template) == PROJECT_ROOT / "pulses" / "imaging_template.json"
        assert "pulses" in template.replace("\\", "/")        # points at the real shipped file
    finally:
        console.shutdown()
        exp.close()


def test_occupancy_calibration_field_names_the_canonical_file_not_blank():
    """The judge-occupancy processor's ``Calibration file`` is never a blank mystery box: it
    opens prefilled with the canonical ``calibrations/calibration.json`` (absolute, project-
    anchored) the Calibrate task writes -- so the operator always sees which json is loaded,
    matching the Rb87 reference's explicit-file model (no implicit 'current' calibration)."""
    from Zou_lab_control._paths import PROJECT_ROOT

    exp, console = _console()
    try:
        spec = next(s for s in exp.readout.processor_specs() if "occup" in s.name.lower())
        editor = _open_editor(console, ("processor", spec.name))
        text = editor.form._widgets["calibration"].text()
        assert text != ""                                          # NOT blank
        p = Path(text)
        assert p.is_absolute() and p == PROJECT_ROOT / "calibrations" / "calibration.json"
        # there is no invented `ema` smoothing param on this processor
        assert "ema" not in editor.form._widgets
    finally:
        console.shutdown()
        exp.close()
