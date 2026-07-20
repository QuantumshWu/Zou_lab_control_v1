"""The plot-kind VOCABULARY: which kinds exist, and what each one accepts.

A plot kind is described by two quite different things.  Eight of them are WORDS -- the
canonical key, the human label, the fitting family, whether it is offered in the live
Add-Panel menu, the accepted ``value`` shape, the starting signal slots, whether it takes
exactly one signal, and which repeat verbs its menu offers.  The ninth is a live
Matplotlib class.  Only the ninth is rendering; the other eight are what a caller needs
in order to KNOW what a kind is, and they were locked inside the figure module together
with the class.

That mattered because the words have more than one surface: the console builds its
Add-Panel menu, its per-kind input-format text, its slot table and its repeat combo from
them; the saved-figure loader validates a stored ``kind`` against them and decides which
kinds a file can be re-viewed as; the figure viewer prints the human label.  Every one of
those had to import the whole render stack merely to read a string.

The class binding stays in the render layer, paired with a spec there.  See
``zlc_frontend.live_plot.live.PlotKind``, which delegates all eight fields to its spec so
a row cannot be built that disagrees with the vocabulary.

``repeat_modes`` is read back from :mod:`zlc_data.repeat_modes` rather than spelled out
per kind -- those tuples are shared BETWEEN kinds (a 2-D frame and a site map offer the
same three verbs), so they are their own definition and this table cites it.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data.repeat_modes import (
    HIST_REPEAT_MODES,
    IMAGE_REPEAT_MODES,
    ROLLING_REPEAT_MODES,
    TRACE_REPEAT_MODES,
)

__all__ = ["PLOT_KIND_SPECS", "PLOT_KIND_SPEC_BY_KEY", "PlotKindSpec"]


@dataclass(frozen=True)
class PlotKindSpec:
    """ONE declarative record per plot kind -- everything about a kind EXCEPT its renderer.

    Fields
    ------
    key            the canonical kind string (``"1d"``, ``"2d"``, ...).
    label          the human Add-Panel / panel-title label.
    render_family  the DataFigure fitting family ("1D" / "2D"), mirroring
                   ``cls.render_family``.  The sentinel ``"auto"`` means "resolve
                   per-figure from the artists" (the site map: image-family only
                   when a background frame is supplied) -- DataFigure then keeps
                   its legacy artist heuristic for that kind.
    panel          True if this kind is offered in the live console ADD-PANEL
                   dropdown (you add a blank panel of it and wire it to a signal).
                   ``pulse`` is ``panel=False`` -- you do not add a blank pulse
                   panel live -- but it is STILL a full panel kind: a saved pulse
                   figure SEEDS a ``kind="pulse"`` PanelCard (the seed path does not
                   gate on this flag), rendered by PanelCard's ``pulse`` branch.
                   So the flag means "addable live", NOT "can be a panel".
    input_format   one-line description of the accepted ``value`` shape (shown in
                   the Setting; ``""`` for a non-panel kind).
    input_slots    the STARTING signal slot(s) a fresh panel opens with --
                   ``((label, default_signal, tooltip), ...)``; empty = the
                   universal single ``signal`` slot.
    single_slot    True if the kind takes EXACTLY ONE signal (no +/- slot growing).
    repeat_modes   the repeat-DISPLAY modes that are MEANINGFUL for this kind, in menu order
                   (the first is the default).  A TRACE/IMAGE reduces its repeats
                   (``average``/``add``/``replace``/``roll``, plus ``create`` = one line per repeat
                   for 1-D); a DISTRIBUTION instead chooses how to BIN the repeats' samples
                   (``pool`` = all repeats in one histogram -- the only distribution mode).
                   Empty = a non-repeat kind (no repeat_mode control).  This is what stops a
                   trace verb like ``roll`` being offered on a histogram (where it is meaningless)
                   and stops a histogram silently ignoring the control (#issue-1).
    """

    key: str
    label: str
    render_family: str = "1D"
    panel: bool = True
    input_format: str = ""
    input_slots: tuple[tuple[str, str, str], ...] = ()
    single_slot: bool = False
    repeat_modes: tuple[str, ...] = ()


#: The ONE plot-kind table, in Add-Panel MENU order.  ``monitor`` names the DEFAULT rolling
#: variant (the side distribution is a toggle on the same kind, not a second kind).
#:
#: This literal is COMPLETE: every kind is here, ``grid`` included.  The render layer pairs
#: each spec with a class, and does so in two steps only because ``GridPlot`` is defined
#: later in that module than the table -- a fact about Python's execution order in ONE file,
#: which is why the deferral lives there and the vocabulary here stays whole.
PLOT_KIND_SPECS: tuple[PlotKindSpec, ...] = (
    PlotKindSpec(
        key="2d", label="2D image", render_family="2D",
        input_format="value must be a 2D array / camera frame (H×W)",
        repeat_modes=IMAGE_REPEAT_MODES,
    ),
    PlotKindSpec(
        key="sites", label="Site map", render_family="auto",
        single_slot=True, repeat_modes=IMAGE_REPEAT_MODES,
        input_format=(
            "value must be a per-site (N,) vector -- one number per tweezer (e.g. occupancy "
            "0/1 or loading rate); signal[0]'s producing node also supplies the ring centres "
            "+ frame underlay"),
        input_slots=(
            # BLANK default (like every other plot) -- a fresh site-map panel must NOT auto-bind
            # to a running "occupied" signal on open; the user picks the occupancy signal in the
            # Setting, and only THEN do the centres + frame underlay auto-resolve from that signal's
            # producing node (_sites_inputs).  A non-blank default here was the "opens already
            # connected" bug.
            ("occupancy", "", "per-site (N,) occupancy vector (signal[0]) -- colours the rings; its "
                              "producing node also supplies the centres + frame underlay"),
        ),
    ),
    PlotKindSpec(
        key="1d", label="1D vector", render_family="1D",
        input_format="value must be a 1D vector (N,) or per-site array",
        repeat_modes=TRACE_REPEAT_MODES,
    ),
    PlotKindSpec(
        key="monitor", label="Rolling trace", render_family="1D",
        input_format="value must be a scalar per shot (rolling trace)",
        repeat_modes=ROLLING_REPEAT_MODES,                  # rolling trace is the ONLY kind with 'roll'
    ),
    PlotKindSpec(
        key="hist", label="Distribution", render_family="1D",
        input_format="value must be a 1D sample vector",
        repeat_modes=HIST_REPEAT_MODES,
    ),
    # Notebook-only static timing diagram -- NOT a console Add-Panel kind.
    PlotKindSpec(key="pulse", label="Pulse sequence", render_family="1D", panel=False),
    # A grid is a live Add-Panel kind like any other: bind a measurement block signal and pick
    # the facet axis in Setting (the saved-recipe path is the non-faceted case).  Its per-site
    # cell content lives in the saved ``figure_recipe``, not here.
    PlotKindSpec(key="grid", label="Site grid", render_family="1D", panel=True),
)

#: ``key -> PlotKindSpec`` for O(1) lookup.  Insertion order = menu order.
PLOT_KIND_SPEC_BY_KEY: dict[str, PlotKindSpec] = {spec.key: spec for spec in PLOT_KIND_SPECS}
