"""Access to redistributable package-owned rendering assets."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


HELVETICA_LIGHT_ASSET = "helvetica-light-587ebe5a59211.ttf"
HELVETICA_LIGHT_FAMILY = "Helvetica Light"


def helvetica_light_path() -> Path:
    """Return the installed font path for Matplotlib's ``addfont`` API."""

    resource = resources.files(__package__).joinpath(HELVETICA_LIGHT_ASSET)
    candidate = Path(str(resource))
    if not candidate.is_file():
        raise FileNotFoundError(
            "Helvetica Light is not available as an installed filesystem asset"
        )
    return candidate


def register_helvetica_light() -> str:
    """Register and verify the canonical font, importing Matplotlib lazily."""

    from matplotlib import font_manager

    path = helvetica_light_path()
    font_manager.fontManager.addfont(str(path))
    family = font_manager.FontProperties(fname=str(path)).get_name()
    if family != HELVETICA_LIGHT_FAMILY:
        raise RuntimeError(
            f"canonical font asset identifies as {family!r}, "
            f"not {HELVETICA_LIGHT_FAMILY!r}"
        )
    return family


__all__ = [
    "HELVETICA_LIGHT_ASSET",
    "HELVETICA_LIGHT_FAMILY",
    "helvetica_light_path",
    "register_helvetica_light",
]
