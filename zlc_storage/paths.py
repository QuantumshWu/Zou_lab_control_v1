"""Project-anchored filesystem path normalization primitives.

This is the storage-neutral owner for resolving a caller-supplied relative path
against the installed project root instead of volatile process CWD.  It performs
no domain repository selection or I/O; upper contexts retain those decisions.
"""

from __future__ import annotations

from pathlib import Path

# The repository root: the folder that holds the packages alongside ``pulses/``
# ``calibrations/`` ``tasks/`` ``docs/``.  This file sits at
# ``zlc_storage/paths.py``, so two parents up is that root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts: str) -> Path:
    """Absolute path under the project root for ``parts`` -- e.g.
    ``project_path("pulses", "mot_field_template.json")``."""
    return PROJECT_ROOT.joinpath(*(str(part) for part in parts)).resolve()


def resolve_under_project(path) -> Path:
    """Resolve ``path`` to an absolute path: an absolute (or ``~``) path is taken as-is;
    a RELATIVE path is anchored to the PROJECT ROOT (never the process CWD), so a bare
    ``calibrations`` always means ``<project>/calibrations`` wherever Python was started."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def display_path(path) -> str:
    """A clear, unambiguous path STRING for a GUI field / on-disk record: the absolute
    path (relative inputs anchored under the project root).  Empty in -> empty out (an
    intentionally blank field).  Use this everywhere a path is SHOWN so the operator
    always sees exactly which file/folder, never a bare CWD-relative name."""
    if path in (None, ""):
        return ""
    return str(resolve_under_project(path))


__all__ = ["PROJECT_ROOT", "project_path", "resolve_under_project", "display_path"]
