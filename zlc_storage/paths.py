"""Pure filesystem path resolution against one caller-owned root."""

from __future__ import annotations

from pathlib import Path

def resolve_under(root: str | Path, path: str | Path) -> Path:
    """Resolve ``path`` against an explicit absolute root.

    Absolute authored paths remain absolute.  Relative paths are anchored to
    ``root`` and never to the process CWD or a package location.  Selecting the
    root is an application-composition decision, not storage policy.
    """

    base = Path(root).expanduser()
    if not base.is_absolute():
        raise ValueError("path root must be absolute")
    value = Path(path).expanduser()
    return (value if value.is_absolute() else base / value).resolve()


__all__ = [
    "resolve_under",
]
