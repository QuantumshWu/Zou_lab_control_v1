"""Confined filesystem path resolution against one caller-owned root."""

from __future__ import annotations

from pathlib import Path


def resolve_under(root: str | Path, path: str | Path) -> Path:
    """Resolve one relative path without permitting escape from ``root``.

    Selecting the root is an application-composition decision.  Authored paths
    must remain relative, and their resolved target must stay beneath that root
    even when ``..`` components or existing symlinks are involved.
    """

    base = Path(root).expanduser()
    if not base.is_absolute():
        raise ValueError("path root must be absolute")
    base = base.resolve()
    value = Path(path).expanduser()
    if value.is_absolute():
        raise ValueError("path below root must be relative")
    resolved = (base / value).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError("path escapes its root") from exc
    return resolved


__all__ = [
    "resolve_under",
]
