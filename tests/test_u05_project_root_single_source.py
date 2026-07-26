"""Where the project root is, said once -- not re-derived by each shell from its depth.

``zlc_storage/paths.py`` owns the folders alongside ``pulses/``, ``calibrations/``,
``tasks/``, and ``docs/``. GUI shells must ask that seam instead of deriving a path
from their own source depth; otherwise a file move silently changes where operator
work is stored.

Capturing the golden also turned up a real defect on the very line being changed: with
``ZLC_TASK_DIR`` set to whitespace, ``_task_files_dir`` raised FileNotFoundError from
``Path("   ").mkdir()``, while its sibling ``_pulse_files_dir`` stripped and fell back.
A blank environment variable is a human typo, not a directory; the two now agree.

The cases below directly preserve the current storage-seam behavior.
"""

from __future__ import annotations

import ast
import os
import pathlib

import pytest

from zlc_storage.paths import PROJECT_ROOT, project_path, user_output_path

REPO = pathlib.Path(__file__).resolve().parents[1]

#: (module path, helper name, env var name, folder) for the two shells that keep a
#: user-facing folder of saved files. Each must ask the storage seam rather than
#: count parents from its own source path.
SHELLS = [
    (
        "zlc_workbench/task_console/layout_repository.py",
        "task_files_dir",
        "TASK_FILES_ENV",
        "tasks",
    ),
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


def test_generated_user_files_have_one_non_input_output_root():
    assert user_output_path("figures", "pulses") == project_path(
        "_output",
        "figures",
        "pulses",
    )
    with pytest.raises(ValueError, match="plain path components"):
        user_output_path("../pulses")




def test_the_storage_seam_still_needs_nothing_but_pathlib():
    """It is imported by both shells at module scope, so it must stay dependency-free."""

    tree = ast.parse((REPO / "zlc_storage" / "paths.py").read_text(encoding="utf-8"))
    modules = {node.module for node in ast.walk(tree)
               if isinstance(node, ast.ImportFrom) and node.module}
    modules |= {alias.name for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    assert modules <= {"__future__", "pathlib"}, modules
