"""Live fitting processor backed by the same headless engine as ``DataFigure``.

The processor is a per-panel :class:`RegionProcessor`: it consumes the SOURCE block plus the panel's
``<slug>_region`` signal (the drawn selection + the panel's bins), and publishes complete scalar fit
result tensors -- so EVERY fit (2-D image centre, 1-D curve, hist) runs on the node's WORKER thread,
never the Qt event loop.  A failed fit publishes NaN parameters plus explicit validity/quality/status
fields; it never leaves stale values on the hub.

The node is model-AGNOSTIC and family-AGNOSTIC:

* a 2-D image model fits EACH ``(R,P)`` image cell (``fit_image``) and publishes one result per cell;
* a 1-D model over a DISTRIBUTION source (the region's ``value-mask`` binding) fits the BINNED
  ``(bin_centers, counts)`` -- the SAME bins the panel displays, carried on the region -- NEVER the raw
  per-pixel samples, so a hist fit is O(bins), instant;
* a 1-D model over a CURVE source reconstructs ``x`` from the region binding and fits the per-point
  reduced curve.

Either way the published parameter keys derive from the solved model's ``names`` (the single source),
so the node name and its signals never lie.  The ``fit_`` prefix only namespaces the keys on the hub so
a fitted parameter never collides with a raw signal.
"""

from __future__ import annotations

import warnings
from dataclasses import replace as _dc_replace
from threading import Lock
from typing import Mapping

import numpy as np

from ...core.fitting import FitRequest, FitResult, fit_data, fit_image, fit_model
from ..logic import SignalSpec
from ._region import RegionProcessor


FIT_STATUS_INVALID = np.int8(0)
FIT_STATUS_OK = np.int8(1)

#: The default fit model.  ``center`` is the default 2-D image model; the class-level ``provides``
#: derives from it, and a per-panel node's ACTUAL keys derive from its own request's model.
_DEFAULT_MODEL = "center"
#: Fixed quality/validity keys every fit publishes alongside the model's own parameters.
_QUALITY_KEYS: tuple[str, ...] = ("fit_valid", "fit_rmse", "fit_r2", "fit_status", "fit_points")


def _param_keys(model) -> tuple[str, ...]:
    """The ``fit_<name>`` hub keys for a model's parameters -- derived from ``FitModel.names``
    (the single source), never a hand-typed layout."""
    return tuple(f"fit_{name}" for name in fit_model(model).names)


def _provided_keys(model) -> tuple[str, ...]:
    return _param_keys(model) + _QUALITY_KEYS


def _request(value=None) -> FitRequest:
    """Parse a fit request from a model key / dict / FitRequest -- ANY family (2-D image, 1-D curve,
    hist), since every fit is now a per-panel worker node."""
    if value is None or value == "":
        return FitRequest(_DEFAULT_MODEL)
    if isinstance(value, FitRequest):
        return value
    if isinstance(value, str):
        return FitRequest(value)
    if isinstance(value, Mapping):
        return FitRequest.from_dict(value)
    raise TypeError("fit request must be FitRequest, mapping, model key, or None")


class FitProcessor(RegionProcessor):
    """Fit the request's model over a per-panel region of a canonical signal block, on the worker."""

    node_label = "fit"
    #: Class-level default (the ``center`` model); the INSTANCE derives its actual keys from the
    #: request's model so ``output_keys`` / ``_bare_output_specs`` / ``provides`` are one source.
    provides = _provided_keys(_DEFAULT_MODEL)

    def __init__(self, hub, *, source_expr=None, region=None, selection=None,
                 fit_request=None, prefix: str = ""):
        self._init_fit_state(fit_request)
        super().__init__(hub, source_expr=source_expr, region=region, selection=selection, prefix=prefix)

    def _init_fit_state(self, fit_request) -> None:
        """Seed the fit half's state (shared verbatim by AnalysisProcessor, whose merged __init__
        cannot run this class's __init__ chain).  Runs BEFORE the LogicNode __init__ chain: the base's
        self-loop guard calls output_keys() -> fit_request, so the model must be known first."""
        self._request_lock = Lock()
        self._fit_request = _request(fit_request)
        self.last_results: tuple[FitResult, ...] = ()
        # FACET-AWARE state (a grid panel fits per-cell -- the SAME facet slice GridPlot displays, so the
        # node solves every cell on its worker and the panel only draws the published params, #6b).  A
        # non-grid panel leaves these unset (``_facet is None``) and fits once as before.  Set by the
        # console via acquisition params (facet / sub_plot_kind / repeat_mode / points_shape).
        self._facet: str | None = None
        self._facet_sub_plot_kind: str = "1d"
        self._facet_repeat_mode: str = "average"
        self._facet_points_shape: tuple[int, ...] = ()
        self._facet_n_cells: int = 1               # the last-fit cell count -> the published (1,1,N) width

    def set_facet(self, *, facet=None, sub_plot_kind=None, repeat_mode=None, points_shape=None) -> None:
        """Configure the per-cell facet fit between shots (a grid panel's Analysis node).  ``facet=None``
        (or ``''``) restores the single whole-region fit.  The slice rule is the ONE shared
        ``core.facet.facet_cells`` GridPlot uses, so the fit cells match the displayed cells exactly."""
        with self._request_lock:
            if facet is not None:
                self._facet = str(facet) or None
            if sub_plot_kind is not None:
                self._facet_sub_plot_kind = str(sub_plot_kind or "1d")
            if repeat_mode is not None:
                self._facet_repeat_mode = str(repeat_mode or "average")
            if points_shape is not None:
                self._facet_points_shape = tuple(int(n) for n in points_shape)

    @property
    def fit_request(self) -> FitRequest:
        with self._request_lock:
            return self._fit_request

    def set_fit_request(self, request) -> None:
        """Atomically replace the structured request (model + fixed/initial) between shots.  The
        SELECTION is not stored here -- it rides on the region signal, merged in per shot."""
        parsed = _request(request)
        with self._request_lock:
            self._fit_request = parsed

    def acquisition_parameters(self) -> dict[str, object]:
        return {"fit_request": self.fit_request.to_dict()}

    def set_acquisition_parameters(self, **values) -> None:
        if "fit_request" in values:
            self.set_fit_request(values["fit_request"])
        if any(k in values for k in ("facet", "sub_plot_kind", "repeat_mode", "points_shape")):
            self.set_facet(facet=values.get("facet"), sub_plot_kind=values.get("sub_plot_kind"),
                           repeat_mode=values.get("repeat_mode"), points_shape=values.get("points_shape"))

    def output_keys(self) -> tuple[str, ...]:
        """The bare published names derive from THIS node's model (the single source), so a gaussian
        panel publishes ``fit_sigma`` and a centre panel ``fit_radius`` -- never a fixed layout."""
        return _provided_keys(self.fit_request.model)

    # ------------------------------------------------------------------ region -> request
    def _request_with_region(self) -> tuple[FitRequest, object]:
        """Merge the config request (model + fixed/initial) with the panel's LATEST region selection,
        and return ``(request, model)``.  The region's frame is the fit's coordinate frame."""
        base = self.fit_request
        selection = self.current_region()
        request = _dc_replace(
            base, selection=selection,
            coordinate_frame=(selection.frame or base.coordinate_frame))
        return request, fit_model(request.model)

    def _result_shape(self) -> tuple[tuple[int, ...], tuple[int, ...], int | None]:
        """The output (point_shape, data_shape, repeat_capacity).  A FACET fit publishes one param value
        PER CELL as ``(1, 1, N)`` (N==1 degenerates to today's scalar shape, so existing fit_x0 consumers
        are unaffected -- they read cell 0).  A 2-D image fit is PER-(R,P) cell (the source's geometry);
        a 1-D / hist fit is ONE result for the whole curve/distribution."""
        if self._facet:
            return (1,), (max(1, int(self._facet_n_cells)),), 1
        if fit_model(self.fit_request.model).family != "2d":
            return (1,), (1,), 1
        if not self._input_tensors:
            return (1,), (1,), 1
        tensor = next(iter(self._input_tensors.values()))
        return tuple(tensor.schema.point_shape), (1,), tensor.schema.repeat_capacity

    # ------------------------------------------------------------------ transform
    def transform(self, inputs: dict[str, object]) -> dict[str, object]:
        from ..signal_expr import hub_namespace

        value = self.source_expr.evaluate(hub_namespace(self.hub, inputs))
        block = np.asarray(value)
        request, model = self._request_with_region()
        if self._facet:
            return self._fit_facets(block, request, model)
        if model.family == "2d":
            return self._fit_2d(block, request, model)
        return self._fit_1d(block, request, model)

    def _fit_facets(self, block, request, model) -> dict[str, object]:
        """Fit EACH facet cell of the block on the worker and publish per-cell parameter vectors
        ``(1, 1, N)`` -- the grid counterpart of the single fit, using the ONE shared slice rule
        (``core.facet.facet_cells``) GridPlot displays, so the fit cells match the drawn cells."""
        from ...core.facet import facet_cells
        with self._request_lock:
            facet = self._facet
            sub = self._facet_sub_plot_kind
            repeat_mode = self._facet_repeat_mode
            points_shape = self._facet_points_shape
        cells = facet_cells(block, facet, sub_plot_kind=str(sub),
                            points_shape=points_shape, repeat_mode=str(repeat_mode))
        n = max(1, len(cells))
        self._facet_n_cells = n                                # drives the (1,1,N) publish schema this shot
        outputs = self._empty_outputs((1, 1, n), request)
        results: list[FitResult] = []
        for k, cell in enumerate(cells):
            result = self._fit_one_cell(np.asarray(cell), request, model, str(sub))
            results.append(result)
            self._store(outputs, (0, 0, k), model, result)
        self.last_results = tuple(results)
        return outputs

    def _fit_one_cell(self, cell, request, model, sub_plot_kind: str) -> FitResult:
        """Fit ONE facet cell by its kind, over the SAME domain the cell's display uses (a 1-D cell over
        the point index -- ``LineCell`` draws over ``arange`` too, so the published centre lands right)."""
        if sub_plot_kind == "2d" or model.family == "2d":
            arr = np.asarray(cell, dtype=float)
            if arr.ndim != 2:
                return FitResult.invalid(
                    request.model, f"a 2-D facet cell needs a 2-D frame; got shape {arr.shape}",
                    coordinate_frame=request.coordinate_frame)
            return fit_image(arr, request)
        if sub_plot_kind == "hist":
            return self._fit_histogram(np.asarray(cell), request)
        y = np.asarray(cell, dtype=np.float64).reshape(-1)
        x = np.arange(y.size, dtype=np.float64)
        return fit_data({"x": x}, y, request)

    def _empty_outputs(self, shape, request) -> dict[str, object]:
        outputs = {
            name: np.full(shape, np.nan, dtype=np.float64)
            for name in (*_param_keys(request.model), "fit_rmse", "fit_r2", "fit_points")
        }
        outputs["fit_valid"] = np.zeros(shape, dtype=np.bool_)
        outputs["fit_status"] = np.full(shape, FIT_STATUS_INVALID, dtype=np.int8)
        return outputs

    @staticmethod
    def _store(outputs, index, model, result) -> None:
        params = result.parameter_map()
        for name in model.names:
            outputs[f"fit_{name}"][index] = params.get(name, np.nan)
        outputs["fit_valid"][index] = result.valid
        outputs["fit_rmse"][index] = result.quality.get("rmse", np.nan)
        outputs["fit_r2"][index] = result.quality.get("r2", np.nan)
        outputs["fit_status"][index] = FIT_STATUS_OK if result.valid else FIT_STATUS_INVALID
        outputs["fit_points"][index] = float(result.n_points)

    def _fit_2d(self, block, request, model) -> dict[str, object]:
        """Fit the 2-D image model to EVERY valid ``(R,P)`` image cell (``fit_image``)."""
        if not self._input_tensors:
            raise ValueError("a 2-D fit requires a registered canonical image source.")
        tensor = next(iter(self._input_tensors.values()))
        schema = tensor.schema
        if len(schema.data_shape) != 2 or tuple(schema.data_shape) != tuple(block.shape[2:]) \
                or block.ndim != 4 or tuple(block.shape[:2]) != tuple(tensor.valid.shape):
            raise ValueError(
                "a 2-D fit requires a canonical (R,P,H,W) image with schema data_shape=(H,W); "
                f"got shape {block.shape}, data_shape={schema.data_shape}.")
        input_valid = np.asarray(tensor.valid, dtype=bool)
        shape = (*block.shape[:2], 1)
        outputs = self._empty_outputs(shape, request)
        results: list[FitResult] = []
        for repeat, point in np.ndindex(block.shape[:2]):
            if input_valid[repeat, point]:
                result = fit_image(block[repeat, point], request)
            else:
                result = FitResult.invalid(
                    request.model, "input tensor cell is invalid",
                    coordinate_frame=request.coordinate_frame)
            results.append(result)
            self._store(outputs, (repeat, point, 0), model, result)
        self.last_results = tuple(results)
        return outputs

    def _fit_1d(self, block, request, model) -> dict[str, object]:
        """Fit a 1-D model ONCE over the region's reconstructed domain: a distribution's binned
        ``(centers, counts)`` (value-mask) or a curve's ``(x, reduced y)``."""
        binding = dict(self.current_region().metadata.get("binding") or {})
        if str(binding.get("mode")) == "value-mask":
            result = self._fit_histogram(block, request)
        else:
            x, y = self._curve_xy(block, binding)
            result = fit_data({"x": x}, y, request)
        self.last_results = (result,)
        outputs = self._empty_outputs((1, 1, 1), request)
        self._store(outputs, (0, 0, 0), model, result)
        return outputs

    def _fit_histogram(self, block, request) -> FitResult:
        """Bin the pooled source samples with the panel's bins (single source, from the region) and fit
        the 1-D model on ``(bin_centers, counts)`` -- NEVER the raw per-pixel samples (O(bins), instant).
        The value-range selection restricts the fit to the selected band, exactly as the panel shows."""
        samples = np.asarray(block, dtype=np.float64).reshape(-1)
        finite = samples[np.isfinite(samples)]
        if finite.size < 2:
            return FitResult.invalid(
                request.model, "distribution has too few finite samples",
                coordinate_frame=request.coordinate_frame)
        bins = self.current_region_bins() or 50
        counts, edges = np.histogram(finite, bins=int(bins))
        centers = (edges[:-1] + edges[1:]) / 2.0
        # The value selection filters bin centres (matching the panel's displayed value-range fit); "x"
        # and "value" are the SAME coordinate for a distribution (its x axis IS the sample value).
        return fit_data({"x": centers, "value": centers}, counts.astype(np.float64), request)

    def _curve_xy(self, block, binding) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct a 1-D curve ``(x, y)`` from the block + the region binding: ``x`` over the curve
        axis (a contiguous index for an axis-crop, or the explicit scatter coordinates), ``y`` the
        block reduced (nan-mean) over every other axis -- the curve shape the panel displays."""
        work = np.asarray(block, dtype=np.float64)
        if str(binding.get("mode")) == "scatter-mask":
            axis = int(binding.get("axis", -1)) % work.ndim
            coords = binding.get("coordinates") or {}
            x = (np.asarray(coords["x"], dtype=np.float64).reshape(-1)
                 if "x" in coords else None)
        else:  # axis-crop
            entry = (list(binding.get("axes") or ()) or [{}])[0]
            axis = int(entry.get("axis", -1)) % work.ndim
            origin = float(entry.get("origin", 0.0))
            step = float(entry.get("step", 1.0)) or 1.0
            x = origin + step * np.arange(work.shape[axis], dtype=np.float64)
        n = int(work.shape[axis])
        if x is None or x.size != n:
            x = np.arange(n, dtype=np.float64)
        other = tuple(a for a in range(work.ndim) if a != axis)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN column -> NaN, dropped by the fit
            y = np.nanmean(work, axis=other) if other else work
        return x, np.asarray(y, dtype=np.float64).reshape(-1)

    def _bare_output_specs(self) -> tuple[SignalSpec, ...]:
        points, data_shape, repeat_capacity = self._result_shape()
        common = {"points_shape": points, "data_shape": data_shape,
                  "repeat_capacity": repeat_capacity}
        model = fit_model(self.fit_request.model)
        # One SignalSpec per fitted parameter -- name + description derive from the model (single
        # source), so the published signals describe whatever model is fit.
        param_specs = [
            SignalSpec(f"fit_{name}", f"fit {name}", "",
                       f"Fitted {model.formula} parameter {name}.", dtype=np.float64, **common)
            for name in model.names
        ]
        return (
            *param_specs,
            SignalSpec("fit_valid", "fit valid", "", "1 when the selected fit converged.",
                       dtype=np.bool_, **common),
            SignalSpec("fit_rmse", "fit RMSE", "", "Root mean squared residual.",
                       dtype=np.float64, **common),
            SignalSpec("fit_r2", "fit R²", "", "Coefficient of determination.",
                       dtype=np.float64, **common),
            SignalSpec("fit_status", "fit status", "", "1=ok, 0=invalid; text is on last_results.",
                       dtype=np.int8, metadata={"codes": {"0": "invalid", "1": "ok"}}, **common),
            SignalSpec("fit_points", "fit points", "", "Finite selected points used by the solver.",
                       dtype=np.float64, **common),
        )


# The fit catalog entry lives in processors/analysis.py: the ONE "Analysis" spec whose ``action``
# choice dispatches to this class as its fit strategy.  This module owns only the compute.

__all__ = [
    "FIT_STATUS_INVALID", "FIT_STATUS_OK", "FitProcessor",
]
