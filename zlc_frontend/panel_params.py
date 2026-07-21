"""What knobs a plot panel of each kind carries, and what one resolves to.

A panel kind declares its knobs as real :class:`~zlc_data.param_decl.ParamDecl` records -- the
SAME declarative record a measurement form uses -- so both surfaces render through one widget
registry and are validated by one kind whitelist.  Four questions get asked of that catalog, and
each has exactly one answer here rather than at every consume site:

* the FULL decl list a panel enumerates (its kind's, plus the grid-only per-cell title knob),
* the DECLARED default of one knob,
* what a panel actually renders for a knob (stored value if present, else the declaration),
* and the same for the colormap specifically, which has a "this kind draws no image" answer.

This catalog reads three things it does not own, which is why it sits in ``zlc_frontend``
rather than a layer below: the record type from ``zlc_data``, the DEFAULT colormaps from
``zlc_frontend.render_style`` (L327 permits the dependency; the palette is "the ONE render
colour owner" and a default colormap is a colour decision), and the histogram-fit verbs from
``zlc_data.curve_fitting``.  Sinking it into ``zlc_data`` instead would mean copying a render
fact down, which L303 -- "只拥有领域中立、无头、可序列化的数据语义和值上的纯算法" -- does
not permit and single-source forbids anyway.

The fit verbs are worth a note.  ``choices`` used to restate ``("none", "single", "double")``
here while the default it opens on lived in the render module as ``DEFAULT_HIST_FIT``, and the
solver in ``zlc_data.curve_fitting`` validated against a set literal written twice more.  Four
copies of one vocabulary, across three layers.  The verbs and their default now belong to the
solver's own module and every surface cites them.
"""

from __future__ import annotations

from typing import Mapping

from zlc_data.curve_fitting import DEFAULT_HISTOGRAM_FIT, HISTOGRAM_FIT_MODES
from zlc_data.param_decl import ParamDecl

from .render_style import PALETTE

__all__ = [
    "CMAPS", "GRID_TITLE_PARAMS", "PANEL_PARAMS",
    "panel_display_decls", "panel_param_default", "resolved_cmap", "resolved_param",
]


#: The colormaps a panel's chooser offers.  A vocabulary, not a palette: which NAMES an operator
#: may pick, while ``PALETTE`` owns which one each kind STARTS on.
CMAPS = ("inferno", "viridis", "magma", "plasma", "gray", "coolwarm")


# A plot panel's per-kind params are REAL ``ParamDecl``s -- the SAME declarative record the
# measurement form uses -- so they render through the SAME PARAM_WIDGETS registry and are
# validated by the SAME kind whitelist (a typo'd kind raises at construction, instead of a
# parallel ParamSpec silently degrading to a text box).  ``display`` (a ParamDecl DATA flag,
# not an art knob) places the param: True = a basic display knob in the Setting popup (the
# colormap chooser); False = a functional plot-API param in the panel's Edit tab -- so the two
# surfaces never duplicate.  Adding a panel param is ONE ParamDecl here; adding a KIND is one
# handler in param_widgets + one whitelist entry on ParamDecl.
PANEL_PARAMS: dict[str, tuple[ParamDecl, ...]] = {
    "2d": (
        ParamDecl(key="cmap", label="colormap", kind="choice", default=PALETTE["cmap_scan"], choices=CMAPS,
                  tooltip="Image colormap", display=True),
    ),
    "sites": (
        # A site map is a binary occupancy OVERLAY (faint ring = empty, bold ring =
        # occupied) on the camera FRAME.  The colormap applies to that frame underlay
        # (its counts colorbar); the rings carry no scale.  It takes ONE signal input --
        # the per-site occupancy (PANEL_INPUT_SLOTS["sites"], picked in the Setting's Source
        # section); its ring CENTRES and the frame UNDERLAY auto-resolve from that signal's
        # producing node, so they are NOT extra slots or params here.
        ParamDecl(key="cmap", label="colormap", kind="choice", default=PALETTE["cmap_camera"], choices=CMAPS,
                  tooltip="Colormap for the camera-frame underlay", display=True),
    ),
    # Pure DISPLAY knobs (history / bins / fit / log axis / colormap) live in the lightweight
    # Setting popup (display=True): they only change how the SAME data is drawn, so they belong with
    # size / relim where an operator reaches for them.  Only acquisition / measurement-API params
    # (none on these display-only kinds) would be display=False and live in the Edit tab.
    "1d": (),
    "monitor": (
        ParamDecl(key="length", label="history", kind="int", default=300, lo=20, hi=10_000,
                  display=True, tooltip="Rolling history length (shots kept on screen)"),
        # The side distribution is ONE plot kind's toggle (not a separate "no-dist" kind):
        # ON shows the histogram band beside the trace, OFF gives the bare rolling line.
        ParamDecl(key="show_dist", label="side distribution", kind="bool", default=True, display=True,
                  tooltip="Show the side distribution histogram beside the rolling trace"),
    ),
    "hist": (
        ParamDecl(key="bins", label="bins", kind="int", default=60, lo=5, hi=500, display=True,
                  tooltip="Histogram bins"),
        # The fit is a confocal-style capsule tri-toggle (none / single / double), NOT a forced default:
        # the operator picks which fit to draw on whatever data the source provides.  "double" is the
        # dark/bright readout convention.  ``segmented=True`` renders it as the TriStateToggleSwitch
        # capsule (sliding thumb) instead of a combo box.  Verbs and default both cite the solver's
        # module -- a fifth fit verb is added there, and this chooser offers it without being touched.
        ParamDecl(key="fit", label="fit", kind="choice", choices=HISTOGRAM_FIT_MODES,
                  default=DEFAULT_HISTOGRAM_FIT, segmented=True, display=True,
                  tooltip="Distribution fit (drives the display directly -- no auto-decision):\n"
                          "  none   = no fit curve\n"
                          "  single = one Gaussian\n"
                          "  double = the dark/bright two-Gaussian readout (fidelity stat shown only "
                          "when the two peaks cleanly separate, else 'fit F=N/A')"),
        # A log count axis makes a SPARSE bright tail (rare high occupancy) visible -- on a linear
        # axis a handful of bright shots vanish under the dark peak.  Default OFF (linear).
        ParamDecl(key="ylog", label="log count axis", kind="bool", default=False, display=True,
                  tooltip="Log-scale the count (y) axis -- reveals a sparse bright tail"),
    ),
    # A pulse panel (seeded from a saved pulse figure) has ONE display knob: whether to draw the
    # always-off channel rows.  The seed restores the saved value; toggling it re-renders the timeline.
    "pulse": (
        ParamDecl(key="include_always_off", label="show off rows", kind="bool", default=True, display=True,
                  tooltip="Draw channel rows that stay OFF the whole sequence (and idle DAC buses)"),
    ),
    # NOTE: there is DELIBERATELY no ``"grid"`` entry.  A grid panel's params are its per-site
    # ``sub_plot_kind``'s params (a hist grid -> the ``"hist"`` bins/fit/ylog, a 2d grid -> the ``"2d"``
    # colormap), resolved dynamically by ``PanelCard._param_kind`` -- so the Setting/Edit UI ALWAYS matches
    # what each cell actually is, instead of a hard-coded hist set that lied for a kernel grid (#4).
}


# Grid-ONLY per-cell title knob (#5): a grid panel ADDS this to its sub_plot_kind's ``PANEL_PARAMS`` so the
# operator can edit the per-cell title TEMPLATE from the Edit tab.  ``display=False`` => the Edit tab (a
# functional knob), not the lightweight Setting popup.  It flows through the SAME ``store_display_param`` ->
# ``GridCell.consume_param`` path every grid display knob uses, and round-trips through the saved view -- so
# ``{id}`` (the facet-aware identifier), ``{popt[i]}`` (a fit param), ``{fid}`` (readout fidelity) are all
# reachable.  (There is no font-SIZE knob: the cell title auto-tracks the xy tick-label size -- _cell_title_pt.)
GRID_TITLE_PARAMS: tuple[ParamDecl, ...] = (
    ParamDecl(key="title_template", label="cell title", kind="text", default="{id}", display=False,
              tooltip="Per-cell title template.  {id}=facet identifier (site / repeat / scan value); "
                      "{k}=cell index; {popt[i]}=a fit parameter; {fid}=readout fidelity.  "
                      "e.g. '{id}  F={popt[2]:.2f}'"),
)


def panel_display_decls(kind: str, param_kind: str) -> tuple[ParamDecl, ...]:
    """The FULL ParamDecl list a panel's Setting / Edit UI + save / recipe enumerate: the kind's own
    ``PANEL_PARAMS`` plus, for a GRID, the grid-generic per-cell title knobs (#5).  The ONE place the two
    are combined, so every enumeration site shows the SAME set and a grid's title template / size are
    edited, applied, saved and reopened through the very same path as bins / cmap."""
    decls = PANEL_PARAMS.get(param_kind, ())
    return decls + GRID_TITLE_PARAMS if kind == "grid" else decls


def panel_param_default(kind: str, key: str) -> object:
    """The declared default of a panel kind's param, from the ONE ``PANEL_PARAMS`` catalog -- so a
    kind's colormap default (``2d`` -> ``inferno``, ``sites`` -> ``gray``) has a SINGLE source and is
    never hand-typed at a consume site.  Returns ``None`` for a kind/key with no declared param."""
    for decl in PANEL_PARAMS.get(str(kind), ()):  # noqa: SIM110 - explicit loop is clearer than any()
        if decl.key == key:
            return decl.default
    return None


def resolved_param(kind: str, params: Mapping[str, object], key: str) -> object:
    """The value a panel of ``kind`` actually renders for ``key``: the operator's stored
    ``params[key]`` when PRESENT, else the kind's declared default from ``PANEL_PARAMS``
    (:func:`panel_param_default`) -- so a consume site (plot build / Edit snapshot) never
    hand-types a declared default, and changing a declaration changes the render AND the
    Setting/Edit UI together (they read the same decl).  Presence is ``key in params``,
    never a truthiness test: ``False`` / ``0`` are legal stored values for a bool/int knob."""
    store = params or {}
    return store[key] if key in store else panel_param_default(kind, key)


def resolved_cmap(kind: str, params: Mapping[str, object]) -> str:
    """The colormap a panel of ``kind`` actually draws with: the operator's picked ``params['cmap']``
    if set, else the kind's declared default from ``PANEL_PARAMS`` (``panel_param_default``).  Returns
    an empty string for a kind that declares no cmap param (1-D / hist / monitor draw no image), so a
    caller can store ``''`` for "no colormap" and a colormap-drawing kind always resolves a real name.
    This is the SINGLE resolver for both the plot-build sites and the save's recorded view state."""
    picked = str((params or {}).get("cmap") or "").strip()
    if picked:
        return picked
    default = panel_param_default(kind, "cmap")
    return str(default) if default else ""
