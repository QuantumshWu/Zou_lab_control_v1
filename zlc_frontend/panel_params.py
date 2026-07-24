"""Current TaskConsole panel parameters that directly affect rendering.

This catalog is deliberately small.  A declaration belongs here only when the
current panel renderer consumes the same key.  Fit authoring is the typed
``FitSpec`` flow exposed by DataFigure, not a histogram display toggle, and
unused legacy controls are not preserved as saved-layout vocabulary.
"""

from __future__ import annotations

from typing import Mapping

from zlc_data.param_decl import ParamDecl

from .render_style import PALETTE

__all__ = ["CMAPS", "PANEL_PARAMS", "panel_param_decls", "resolved_panel_param"]


CMAPS = ("inferno", "viridis", "magma", "plasma", "gray", "coolwarm")


PANEL_PARAMS: dict[str, tuple[ParamDecl, ...]] = {
    "2d": (
        ParamDecl(
            key="colormap",
            label="colormap",
            kind="choice",
            default=PALETTE["cmap_scan"],
            choices=CMAPS,
            tooltip="Image colormap",
            display=True,
        ),
    ),
    "sites": (
        ParamDecl(
            key="colormap",
            label="colormap",
            kind="choice",
            default=PALETTE["cmap_camera"],
            choices=CMAPS,
            tooltip="Colormap for the exact same-shot camera underlay",
            display=True,
        ),
    ),
    "monitor": (
        ParamDecl(
            key="show_dist",
            label="side distribution",
            kind="bool",
            default=True,
            display=True,
            tooltip="Show the side distribution histogram beside the rolling trace",
        ),
    ),
    "hist": (
        ParamDecl(
            key="bins",
            label="bins",
            kind="int",
            default=60,
            lo=5,
            hi=500,
            display=True,
            tooltip="Histogram bins",
        ),
        ParamDecl(
            key="ylog",
            label="log count",
            kind="bool",
            default=False,
            display=True,
            tooltip="Use a logarithmic count axis",
        ),
        ParamDecl(
            key="fit",
            label="fit",
            kind="choice",
            default="double",
            choices=("none", "single", "double"),
            display=True,
            tooltip="Bounded presentation fit drawn over the distribution",
        ),
    ),
}


def panel_param_decls(param_kind: str) -> tuple[ParamDecl, ...]:
    """Return the exact render-consumed parameter declarations for one kind."""

    return PANEL_PARAMS.get(str(param_kind), ())


def resolved_panel_param(
    param_kind: str,
    params: Mapping[str, object],
    key: str,
) -> object:
    """Resolve one declared key from persisted values or its declaration."""

    declarations = panel_param_decls(param_kind)
    try:
        declaration = next(item for item in declarations if item.key == key)
    except StopIteration as error:
        raise KeyError(f"panel kind {param_kind!r} has no parameter {key!r}") from error
    return params[key] if key in params else declaration.default
