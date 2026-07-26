"""One naming invariant shared by typed Logic-node outputs."""

from __future__ import annotations

from zlc_storage import canonical_text


def bare_output_name(value: str, *, kind: str = "output") -> str:
    """Return one canonical unqualified output name."""

    name = canonical_text(value, f"{kind} name")
    if "/" in name or name.startswith("@"):
        raise ValueError(f"{kind} name must be bare, not namespaced")
    return name


__all__ = ["bare_output_name"]
