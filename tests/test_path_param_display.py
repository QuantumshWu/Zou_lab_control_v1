"""Contract: a design param panel never shows a file/folder field you cannot interpret.

Every ``kind="path"`` field renders an UNAMBIGUOUS, project-anchored path -- an absolute
path, never a bare CWD-relative name like ``calibrations``.  Intentional empty values
remain empty.  Display and on-disk resolution share :mod:`Zou_lab_control._paths`.

Offscreen Qt + virtual backend (the same contract path real hardware takes).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

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
