"""Where the project root is, said once -- not re-derived by each shell from its own depth.

``zlc_storage/paths.py`` owns it, and its own docstring already names the folders it
anchors: "the folder that holds the packages alongside ``pulses/`` ``calibrations/``
``tasks/`` ``docs/``".  Yet both GUI shells computed it a SECOND time, from the depth of
their own file:

    task_console.py   Path(__file__).resolve().parents[2] / "tasks"
    pulse_gui.py      Path(__file__).resolve().parents[2] / "pulses"

Those agreed with the seam only for as long as both files stayed exactly two levels down
-- and moving those files is precisely what this migration does.  The failure would not
be an import error either: the GUI would quietly start saving layouts and pulse programs
into a different folder, and the operator would find their work missing rather than see a
crash.  ``parents[3]`` in ``fpga_pulse_streamer`` is deliberately NOT swept in: it looks
for an in-repo resource, tries the CWD first and returns None when absent, which is a
different question from "where is the project root".

Capturing the golden also turned up a real defect on the very line being changed: with
``ZLC_TASK_DIR`` set to whitespace, ``_task_files_dir`` raised FileNotFoundError from
``Path("   ").mkdir()``, while its sibling ``_pulse_files_dir`` stripped and fell back.
A blank environment variable is a human typo, not a directory; the two now agree.

The tests that cover these helpers (test_frontend_smoke.py) sit outside
``tests/migration_active_tests.txt`` and are frozen, so the cases below were captured from
the shells immediately before the change.
"""

from __future__ import annotations

#: C41 -- these specific tests guard legacy artifacts and die with them; the rest of the
#: file guards the NEW structure and is permanent (swept by test_design_charter).

import ast
import os
import pathlib

import pytest

from zlc_storage.paths import PROJECT_ROOT, project_path

REPO = pathlib.Path(__file__).resolve().parents[1]

#: (module path, helper name, env var name, folder) for the two shells that keep a
#: user-facing folder of saved files.
#: The tasks entry named ``task_console._task_files_dir`` until S5-shell(w) moved the whole
#: saved-layout cluster into ``zlc_frontend.console_state``.  The claim is unchanged -- ask
#: the seam, never count your own parents -- so it follows the function rather than being
#: deleted along with the shell that used to hold it.
SHELLS = [
    ("zlc_frontend/console_state.py", "task_files_dir", "TASK_FILES_ENV", "tasks"),
    # The formal Pulse editor owns this one filesystem-facing helper; split
    # views/controllers never derive a second project root.
    ("zlc_workbench/pulse_editor/window.py", "_pulse_files_dir", "_PULSE_FILES_ENV", "pulses"),
]


def _shell(relative):
    import importlib

    return importlib.import_module(relative[:-3].replace("/", "."))


@pytest.mark.parametrize("relative, helper, env_name, folder", SHELLS)
def test_the_default_folder_is_the_one_the_storage_seam_resolves(relative, helper, env_name,
                                                                 folder, monkeypatch):
    module = _shell(relative)
    monkeypatch.delenv(getattr(module, env_name), raising=False)
    assert getattr(module, helper)() == project_path(folder)
    assert project_path(folder).parent == PROJECT_ROOT


@pytest.mark.parametrize("relative, helper, env_name, folder", SHELLS)
def test_an_explicit_override_still_wins(relative, helper, env_name, folder, monkeypatch, tmp_path):
    module = _shell(relative)
    target = tmp_path / folder
    monkeypatch.setenv(getattr(module, env_name), str(target))
    assert getattr(module, helper)() == target
    assert target.is_dir()          # the helper creates it, as it always did


@pytest.mark.parametrize("relative, helper, env_name, folder", SHELLS)
def test_a_blank_override_falls_back_instead_of_crashing(relative, helper, env_name, folder,
                                                         monkeypatch):
    """The defect the golden capture found: ``Path("   ").mkdir()`` raised FileNotFoundError.

    A whitespace-only environment variable is a typo, and the two sibling helpers used to
    disagree about it -- one stripped, one did not."""

    module = _shell(relative)
    monkeypatch.setenv(getattr(module, env_name), "   ")
    assert getattr(module, helper)() == project_path(folder)


@pytest.mark.parametrize("relative, helper, env_name, folder", SHELLS)
def test_neither_shell_re_derives_the_project_root_from_its_own_depth(relative, helper, env_name,
                                                                      folder):
    """The load-bearing assertion: the two values AGREE today, so only the source can
    distinguish a shell that asks the seam from one that counts its own parents."""

    tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
    func = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == helper)
    # Anchor on ``__file__``, not on the word "parents": these helpers legitimately call
    # ``mkdir(parents=True)``, and matching that keyword made this assertion fail on
    # correct code.  A re-derivation always STARTS from the module's own file.
    assert "__file__" not in ast.dump(func), (
        f"{relative}:{func.lineno} still derives a path from its own file location")
    called = {node.func.id for node in ast.walk(func)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "project_path" in called

    imported = {alias.name for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module == "zlc_storage.paths"
                for alias in node.names}
    assert "project_path" in imported


def test_the_seam_is_the_only_definition_of_the_project_root():
    """A sweep, so the next shell that invents its own copy is caught rather than joining a
    habit.  ``fpga_pulse_streamer`` is exempt with its reason: it resolves an in-repo
    RESOURCE (board.xdc), preferring the CWD and returning None when it is not there."""

    exempt = {"Zou_lab_control/neutral_atom/devices/fpga_pulse_streamer.py"}
    offenders = []
    for path in (REPO / "Zou_lab_control").rglob("*.py"):
        relative = path.relative_to(REPO).as_posix()
        if relative in exempt:
            continue
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if "__file__" in line and "parents[" in line:
                offenders.append(f"{relative}:{number}")
    assert not offenders, (
        "these re-derive a project-anchored path from their own file depth instead of "
        "asking zlc_storage.paths:\n" + "\n".join(offenders))




def test_the_storage_seam_still_needs_nothing_but_pathlib():
    """It is imported by both shells at module scope, so it must stay dependency-free."""

    tree = ast.parse((REPO / "zlc_storage" / "paths.py").read_text(encoding="utf-8"))
    modules = {node.module for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.module}
    modules |= {alias.name for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    assert modules <= {"__future__", "pathlib"}, modules
