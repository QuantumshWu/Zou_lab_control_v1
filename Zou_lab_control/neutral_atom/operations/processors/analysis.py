"""Analysis -- the ONE first-class per-panel analysis processor (ROI reduce XOR curve fit).

A panel's drag-analysis and the manual Add-processor catalog both build THIS node: a
:class:`RegionProcessor` consuming a data SOURCE (coherently) plus a per-panel REGION (latest-wins --
a ``<slug>_region`` control signal from a drag, or an INLINE Selection typed into the manual form's
bounds).  The ``action`` choice picks WHAT the region does:

* ``roi``  -- reduce the region to one scalar per shot + republish the region
  (:class:`~.roi.RoiProcessor`'s compute: ``roi_value`` / ``roi_frame``);
* ``fit``  -- fit the request's model over the region on the WORKER
  (:class:`~.fit.FitProcessor`'s compute: ``fit_<param>``/... per the model).

Those two classes are the INTERNAL STRATEGIES: :class:`AnalysisProcessor` inherits both and
dispatches ``transform`` / ``output_keys`` / ``_bare_output_specs`` on ``self.action`` -- every
computation exists exactly once, and switching a live panel's analysis from ROI to fit is ONE
``set_acquisition_parameters(action=...)`` on the SAME node (same row, same region, same prefix),
never a delete-and-recreate.  :data:`ANALYSIS_ACTIONS` is the single action vocabulary the panel's
Analysis combo reuses (frontend imports neutral_atom; the sealed seam holds).
"""

from __future__ import annotations

from ...core.fitting import FitRequest, fit_models
from ...core.params import ParamDecl
from ...core.selection import Selection, axis_crop_binding, value_mask_binding
from ..logic import FRAME_0
from ..processor import ProcessorSpec
from ..processor_registry import processor
from ._region import RegionProcessor
from .fit import FitProcessor, _DEFAULT_MODEL, _provided_keys
from .roi import ROI_REDUCERS, RoiProcessor


#: The catalog/spec display name -- the ONE string the console resolves the analysis spec by.
ANALYSIS_SPEC_NAME = "Analysis"
#: The action vocabulary -- single source for the node's dispatch, the ParamDecl choices AND the
#: panel Setting's Analysis combo (the frontend imports THIS, never a retyped copy).
ANALYSIS_ACTIONS = ("roi", "fit")
#: Manual-form region kinds -> the binding constructors (axes = a contiguous axis crop on the last
#: two block axes; value = a distribution's sample-value mask).
ANALYSIS_REGION_KINDS = ("axes", "value")


class AnalysisProcessor(FitProcessor, RoiProcessor):
    """One per-panel analysis node: ROI reduce XOR model fit over the same (source, region) pair.

    Inherits BOTH strategies' compute and dispatches on ``self.action``.  The merged ``__init__``
    seeds both halves' state directly (each half's ``_init_*_state`` -- the classes' own seeders) and
    runs the :class:`RegionProcessor` chain ONCE, so there is exactly one hub subscription."""

    node_label = "analysis"
    provides = ()                                  # dynamic -- see output_keys()

    def __init__(self, hub, *, action: str = "roi", source_expr=None, region=None, selection=None,
                 reduce: str = "mean", fit_request=None, prefix: str = "",
                 facet=None, sub_plot_kind=None, repeat_mode=None, points_shape=None):
        if str(action) not in ANALYSIS_ACTIONS:
            raise ValueError(f"analysis action must be one of {ANALYSIS_ACTIONS}; got {action!r}.")
        self.action = str(action)
        if str(reduce) not in ROI_REDUCERS:
            raise ValueError(f"ROI reduce must be one of {tuple(ROI_REDUCERS)}; got {reduce!r}.")
        self.reduce = str(reduce)
        self._init_fit_state(fit_request)          # before the base chain: the self-loop guard reads
        # A facet grid panel fits per-cell (the fit half slices with the SAME rule GridPlot displays).
        self.set_facet(facet=facet, sub_plot_kind=sub_plot_kind,
                       repeat_mode=repeat_mode, points_shape=points_shape)
        RegionProcessor.__init__(                  # output_keys() -> action/fit_request
            self, hub, source_expr=source_expr, region=region, selection=selection, prefix=prefix)
        self._init_roi_state(hub)

    # ------------------------------------------------------------------ action dispatch
    def _strategy(self):
        return RoiProcessor if self.action == "roi" else FitProcessor

    def output_keys(self):
        return RoiProcessor.provides if self.action == "roi" else _provided_keys(self.fit_request.model)

    def transform(self, inputs):
        return self._strategy().transform(self, inputs)

    def _bare_output_specs(self):
        return self._strategy()._bare_output_specs(self)

    def acquisition_parameters(self) -> dict[str, object]:
        out: dict[str, object] = {"action": self.action}
        out.update(RoiProcessor.acquisition_parameters(self) if self.action == "roi"
                   else FitProcessor.acquisition_parameters(self))
        return out

    def set_acquisition_parameters(self, **values) -> None:
        """Both halves' knobs plus the LIVE action switch -- applied between shots on the worker, so
        the panel's fit<->ROI toggle re-uses the SAME node/row/region (no delete-and-recreate)."""
        if "action" in values:
            action = str(values["action"])
            if action not in ANALYSIS_ACTIONS:
                raise ValueError(f"analysis action must be one of {ANALYSIS_ACTIONS}; got {action!r}.")
            self.action = action
        RoiProcessor.set_acquisition_parameters(self, **{k: v for k, v in values.items() if k == "reduce"})
        FitProcessor.set_acquisition_parameters(self, **{
            k: v for k, v in values.items()
            if k in ("fit_request", "facet", "sub_plot_kind", "repeat_mode", "points_shape")})


def _declared_keys(values) -> tuple[str, ...]:
    """The bare output names for a row's CURRENT values -- the metadata deriver the console reads
    (declared == published, per action + fit model)."""
    values = dict(values or {})
    if str(values.get("action") or "roi") == "roi":
        return RoiProcessor.provides
    request = values.get("fit_request")
    model = (request or {}).get("model") if isinstance(request, dict) else None
    return _provided_keys(str(model or values.get("model") or _DEFAULT_MODEL))


def _inline_selection(values) -> Selection | None:
    """A Selection from the manual form's typed bounds (``region_kind`` + x/y min/max; blank =
    unbounded), built through the SAME binding vocabulary a panel drag uses -- ``None`` when no bound
    was typed (the whole-frame default)."""
    def _pair(lo_key, hi_key):
        lo, hi = values.get(lo_key), values.get(hi_key)
        return None if lo in (None, "") and hi in (None, "") else (
            float(lo if lo not in (None, "") else "-inf"),
            float(hi if hi not in (None, "") else "inf"))

    x_pair, y_pair = _pair("x_min", "x_max"), _pair("y_min", "y_max")
    if x_pair is None and y_pair is None:
        return None
    # AxisRange requires FINITE bounds: a half-open pair keeps only the typed side by clamping the
    # open side to a huge finite sentinel (the crop clamps to the block anyway).
    big = 1e30
    def _finite(pair):
        return (max(pair[0], -big), min(pair[1], big))

    if str(values.get("region_kind") or "axes") == "value":
        lo, hi = _finite(x_pair or y_pair)
        return Selection.value(lo, hi, metadata={"binding": value_mask_binding()})
    ranges = []
    from ...core.selection import AxisRange
    if x_pair is not None:
        lo, hi = _finite(x_pair)
        ranges.append(AxisRange("x", lo, hi))
    if y_pair is not None:
        lo, hi = _finite(y_pair)
        ranges.append(AxisRange("y", lo, hi))
    binding = axis_crop_binding([("x", -1, 0.0, 1.0), ("y", -2, 0.0, 1.0)])
    return Selection(tuple(ranges), metadata={"binding": binding})


@processor(order=25)
def analysis(readout) -> ProcessorSpec:
    """ROI-reduce or model-fit a region of a live block -- the per-panel drag analysis, also fully
    buildable by hand (typed bounds, no panel needed)."""

    model_choices = tuple(model.key for model in fit_models())
    params = (
        ParamDecl("source", "Signal source", "signal_expr",
                  default={"inputs": [FRAME_0], "source": "value = signal"},
                  tooltip="The block to analyse (a camera frame, a 1-D scan curve, a distribution's "
                          "samples, a site-map vector)."),
        ParamDecl("action", "Action", "choice", default="roi", choices=ANALYSIS_ACTIONS,
                  tooltip="roi = reduce the region to one scalar per shot (roi_value + roi_frame);\n"
                          "fit = solve the chosen model over the region on the worker (fit_*)."),
        ParamDecl("reduce", "Reduce", "choice", default="mean", choices=tuple(ROI_REDUCERS),
                  tooltip="roi action: how roi_value collapses the selected cells each shot."),
        ParamDecl("model", "Fit model", "choice", default=_DEFAULT_MODEL, choices=model_choices,
                  tooltip="fit action: the model to solve (2-D 'center' for an image; the 1-D "
                          "peak/decay family for a curve or a binned distribution)."),
        ParamDecl("fixed", "Fixed params", "json", default=None,
                  tooltip="fit action, optional: {name: value} parameters pinned during the solve."),
        ParamDecl("initial", "Initial guess", "json", default=None,
                  tooltip="fit action, optional: [v1, v2, ...] full initial parameter vector."),
        ParamDecl("region", "Region signal", "text", default="",
                  tooltip="A <slug>_region hub signal carrying a panel's drawn selection (a panel drag "
                          "fills this).  Leave BLANK to type the bounds below instead."),
        ParamDecl("region_kind", "Region kind", "choice", default="axes", choices=ANALYSIS_REGION_KINDS,
                  tooltip="Typed bounds mode: axes = crop the block's x/y axes; value = keep samples "
                          "whose VALUE lies in [x min, x max] (a distribution band)."),
        ParamDecl("x_min", "x min", "float", default=None, lo=-1e12, hi=1e12, optional=True,
                  tooltip="Typed region bound (blank = unbounded)."),
        ParamDecl("x_max", "x max", "float", default=None, lo=-1e12, hi=1e12, optional=True,
                  tooltip="Typed region bound (blank = unbounded)."),
        ParamDecl("y_min", "y min", "float", default=None, lo=-1e12, hi=1e12, optional=True,
                  tooltip="Typed region bound, axes mode (blank = unbounded)."),
        ParamDecl("y_max", "y max", "float", default=None, lo=-1e12, hi=1e12, optional=True,
                  tooltip="Typed region bound, axes mode (blank = unbounded)."),
        # The structured FitRequest a panel's Analysis section round-trips (model + fixed/initial as
        # ONE payload).  Not a form field: the model/fixed/initial fields above are the manual surface.
        ParamDecl("fit_request", "Fit request", "json", default=None, display=False,
                  tooltip="Structured FitRequest shared with the plot selector."),
    )

    def make_node(hub, *, prefix: str = "", **values):
        action = str(values.get("action") or "roi")
        region_name = str(values.get("region") or "").strip()
        fit_request = values.get("fit_request")
        if not fit_request:
            request = FitRequest(str(values.get("model") or _DEFAULT_MODEL),
                                 fixed=dict(values.get("fixed") or {}),
                                 initial=(tuple(values["initial"]) if values.get("initial") else None))
            fit_request = request.to_dict()
        return AnalysisProcessor(
            hub, action=action, source_expr=values.get("source"),
            region=region_name or None,
            selection=None if region_name else _inline_selection(values),
            reduce=str(values.get("reduce") or "mean"),
            fit_request=fit_request, prefix=prefix,
            facet=values.get("facet"), sub_plot_kind=values.get("sub_plot_kind"),
            repeat_mode=values.get("repeat_mode"), points_shape=values.get("points_shape"))

    return ProcessorSpec(
        name=ANALYSIS_SPEC_NAME,
        params=params,
        make_node=make_node,
        # The static union (discovery collision key + pre-values legend); the console's live
        # declaration reads the ``declared_keys`` deriver (declared == published per action/model).
        result_keys=tuple(dict.fromkeys((*RoiProcessor.provides, *_provided_keys(_DEFAULT_MODEL)))),
        default_kind="2d",               # the roi region view is the natural first view
        metadata={"declared_keys": _declared_keys},
    )


def _model_choice_is_valid() -> None:
    # Import-time sanity: the default model must be in the derived choice list (fit_models is the
    # single model catalogue; a rename there must fail loudly here, not silently in a form).
    if _DEFAULT_MODEL not in {m.key for m in fit_models()}:
        raise RuntimeError(f"analysis default model {_DEFAULT_MODEL!r} is not in fit_models().")


_model_choice_is_valid()

__all__ = ["ANALYSIS_ACTIONS", "ANALYSIS_REGION_KINDS", "ANALYSIS_SPEC_NAME", "AnalysisProcessor", "analysis"]
