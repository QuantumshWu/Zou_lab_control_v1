"""Project-anchored path helper -- the SINGLE source for turning a path into an
unambiguous, readable string and for resolving a relative path the SAME way
everywhere (never against the volatile process CWD).

A file/folder parameter must read as either an ABSOLUTE path or one clearly anchored
to the project, never a bare name like ``calibrations`` whose meaning depends on where
Python happened to be launched.  GUI param fields call :func:`display_path` so the
operator sees exactly which file/folder is loaded; the analysis/task layer calls
:func:`resolve_under_project` so a relative path resolves to the same location
regardless of CWD.  Dependency-free (pathlib only) so both ``neutral_atom`` and
``frontend`` import it without coupling -- the same seam pattern as
``_readout_math`` / ``_viewer_registry``.
"""

from __future__ import annotations

from pathlib import Path

# The repository root: the folder that holds the ``Zou_lab_control`` package alongside
# ``pulses/`` ``calibrations/`` ``tasks/`` ``docs/``.  This file sits at
# ``Zou_lab_control/_paths.py``, so two parents up is that root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Regenerable output uses one gitignored root rather than scattering folders
# across the repository.  Domain artifact repositories own their own durable
# locations; this module only supplies paths still consumed by the legacy shell.
OUTPUT_ROOT = "_output"
GENERATED_SEQUENCES_DIR = f"{OUTPUT_ROOT}/generated_sequences"
RESULTS_DIR = f"{OUTPUT_ROOT}/results"


def project_path(*parts: str) -> Path:
    """Absolute path under the project root for ``parts`` -- e.g.
    ``project_path("pulses", "mot_field_template.json")``."""
    return PROJECT_ROOT.joinpath(*[str(p) for p in parts])


def resolve_under_project(path) -> Path:
    """Resolve ``path`` to an absolute path: an absolute (or ``~``) path is taken as-is;
    a RELATIVE path is anchored to the PROJECT ROOT (never the process CWD), so a bare
    ``calibrations`` always means ``<project>/calibrations`` wherever Python was started."""
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def display_path(path) -> str:
    """A clear, unambiguous path STRING for a GUI field / on-disk record: the absolute
    path (relative inputs anchored under the project root).  Empty in -> empty out (an
    intentionally blank field).  Use this everywhere a path is SHOWN so the operator
    always sees exactly which file/folder, never a bare CWD-relative name."""
    if path in (None, ""):
        return ""
    return str(resolve_under_project(path))


__all__ = ["PROJECT_ROOT", "project_path", "resolve_under_project", "display_path"]
