"""Live fitting processor backed by the same headless engine as ``DataFigure``.

The processor is an adapter only: it preserves the input tensor's physical
``(R, P)`` cells, hands each image datum to :mod:`neutral_atom.core.fitting`,
and publishes complete scalar result tensors.  A failed fit publishes NaN
parameters plus explicit validity/quality/status fields; it never leaves stale
values on the hub.

The node is model-AGNOSTIC: it fits whatever 2-D model the request names and
publishes THAT model's own parameters (``fit_<name>``) plus a fixed
quality/validity suffix.  The published parameter keys therefore derive from the
solved model's ``names`` (the single source), so the node name and its signals
never lie -- a non-centre 2-D model would publish its own parameters, not a
hardwired ``x0/y0/size`` centre layout.  The ``fit_`` prefix only namespaces the
keys on the hub so a fitted parameter never collides with a raw signal.
"""

from __future__ import annotations

from threading import Lock
from typing import Mapping

import numpy as np

from ...core.fitting import FitRequest, FitResult, fit_image, fit_model
from ...core.params import ParamDecl
from ..logic import DEFAULT_SOURCE, FRAME_0, Processor, SignalExpr, SignalSpec
from ..processor import ProcessorSpec
from ..processor_registry import processor


FIT_SPEC_NAME = "Frame fit"
FIT_STATUS_INVALID = np.int8(0)
FIT_STATUS_OK = np.int8(1)

#: The default image model.  Only a 2-D-family fit becomes a hub node, and ``center`` is the sole
#: image model today, so the class-level ``provides`` derives from it.  A future 2-D model would
#: publish ITS parameters through the same single-source derivation.
_DEFAULT_MODEL = "center"
#: Fixed quality/validity keys every frame fit publishes alongside the model's own parameters.
_QUALITY_KEYS: tuple[str, ...] = ("fit_valid", "fit_rmse", "fit_r2", "fit_status", "fit_points")


def _param_keys(model) -> tuple[str, ...]:
    """The ``fit_<name>`` hub keys for a model's parameters -- derived from ``FitModel.names``
    (the single source), never a hand-typed centre layout."""
    return tuple(f"fit_{name}" for name in fit_model(model).names)


def _provided_keys(model) -> tuple[str, ...]:
    return _param_keys(model) + _QUALITY_KEYS


def _request(value=None) -> FitRequest:
    if value is None or value == "":
        return FitRequest(_DEFAULT_MODEL)
    if isinstance(value, FitRequest):
        request = value
    elif isinstance(value, str):
        request = FitRequest(value)
    elif isinstance(value, Mapping):
        request = FitRequest.from_dict(value)
    else:
        raise TypeError("fit request must be FitRequest, mapping, model key, or None")
    if fit_model(request.model).family != "2d":
        raise ValueError("Frame fit consumes image data and therefore requires a 2D fit model")
    return request


class FitProcessor(Processor):
    """Fit every valid image cell in a canonical ``(R,P,H,W)`` signal."""

    node_label = "frame fit"
    #: Class-level default (the ``center`` model); the instance derives its actual keys from the
    #: request's model so ``transform`` / ``_bare_output_specs`` / ``provides`` are one source.
    provides = _provided_keys(_DEFAULT_MODEL)

    def __init__(self, hub, *, source_expr=None, fit_request=None, prefix: str = ""):
        expr = source_expr if isinstance(source_expr, SignalExpr) else SignalExpr.from_value(source_expr)
        if not expr.inputs:
            expr = SignalExpr([FRAME_0], DEFAULT_SOURCE)
        super().__init__(hub, consumes=tuple(expr.inputs), prefix=prefix)
        self.source_expr = expr
        self._request_lock = Lock()
        self._fit_request = _request(fit_request)
        self.last_results: tuple[FitResult, ...] = ()

    @property
    def fit_request(self) -> FitRequest:
        with self._request_lock:
            return self._fit_request

    def set_fit_request(self, request) -> None:
        """Atomically replace the structured request between processor shots."""

        parsed = _request(request)
        with self._request_lock:
            self._fit_request = parsed

    def _input_contract(self, value) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], int | None]:
        array = np.asarray(value)
        if len(self._input_tensors) != 1:
            raise ValueError(
                "Fit center requires exactly one registered canonical image source.")
        tensor = next(iter(self._input_tensors.values()))
        schema = tensor.schema
        if tuple(schema.data_shape) != tuple(array.shape[2:]) or len(schema.data_shape) != 2:
            raise ValueError(
                "Fit center requires schema data_shape=(H,W); "
                f"got data_shape={schema.data_shape}, evaluated shape={array.shape}.")
        if array.ndim != 4 or tuple(array.shape[:2]) != tuple(tensor.valid.shape):
            raise ValueError(
                "Fit expression must preserve canonical (R,P,H,W) axes; "
                f"got {array.shape} for validity {tensor.valid.shape}.")
        return array, np.asarray(tensor.valid, dtype=bool), schema.point_shape, schema.repeat_capacity

    def transform(self, inputs: dict[str, object]) -> dict[str, object]:
        from ..signal_expr import hub_namespace

        value = self.source_expr.evaluate(hub_namespace(self.hub, inputs))
        block, input_valid, _point_shape, _repeat_capacity = self._input_contract(value)
        shape = (*block.shape[:2], 1)
        request = self.fit_request
        names = fit_model(request.model).names        # the ONE source for the published parameter keys
        outputs = {
            name: np.full(shape, np.nan, dtype=np.float64)
            for name in (*_param_keys(request.model), "fit_rmse", "fit_r2", "fit_points")
        }
        outputs["fit_valid"] = np.zeros(shape, dtype=np.bool_)
        outputs["fit_status"] = np.full(shape, FIT_STATUS_INVALID, dtype=np.int8)

        results: list[FitResult] = []
        for repeat, point in np.ndindex(block.shape[:2]):
            if input_valid[repeat, point]:
                result = fit_image(block[repeat, point], request)
            else:
                result = FitResult.invalid(
                    request.model, "input tensor cell is invalid",
                    coordinate_frame=request.coordinate_frame)
            results.append(result)
            params = result.parameter_map()
            for name in names:                          # publish each fitted parameter under fit_<name>
                outputs[f"fit_{name}"][repeat, point, 0] = params.get(name, np.nan)
            outputs["fit_valid"][repeat, point, 0] = result.valid
            outputs["fit_rmse"][repeat, point, 0] = result.quality.get("rmse", np.nan)
            outputs["fit_r2"][repeat, point, 0] = result.quality.get("r2", np.nan)
            outputs["fit_status"][repeat, point, 0] = FIT_STATUS_OK if result.valid else FIT_STATUS_INVALID
            outputs["fit_points"][repeat, point, 0] = float(result.n_points)
        self.last_results = tuple(results)
        return outputs

    def _result_shape(self) -> tuple[tuple[int, ...], int | None]:
        if not self._input_tensors:
            return (1,), 1
        tensor = next(iter(self._input_tensors.values()))
        return tuple(tensor.schema.point_shape), tensor.schema.repeat_capacity

    def _bare_output_specs(self) -> tuple[SignalSpec, ...]:
        points, repeat_capacity = self._result_shape()
        common = {"points_shape": points, "data_shape": (1,),
                  "repeat_capacity": repeat_capacity}
        model = fit_model(self.fit_request.model)
        # One SignalSpec per fitted parameter -- name + description derive from the model (single
        # source), so the published signals describe whatever model is fit, not a fixed centre layout.
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


@processor(order=26)
def frame_fit(readout) -> ProcessorSpec:
    """Fit the request's 2-D model (default: Gaussian centre) to every image tensor cell."""

    default_request = FitRequest(_DEFAULT_MODEL).to_dict()
    params = (
        ParamDecl("source", "Frame source", "signal_expr",
                  default={"inputs": [FRAME_0], "source": "value = signal"},
                  tooltip="One registered image signal with data_shape=(H,W)."),
        ParamDecl("fit_request", "Fit request", "json", default=default_request,
                  tooltip="Structured FitRequest shared with the plot selector.", display=False),
    )

    def make_node(hub, *, prefix: str = "", **values):
        return FitProcessor(
            hub, source_expr=values.get("source"), fit_request=values.get("fit_request"), prefix=prefix)

    return ProcessorSpec(
        name=FIT_SPEC_NAME,
        params=params,
        make_node=make_node,
        result_keys=FitProcessor.provides,
        default_kind="monitor",
        default_value_key="fit_x0",   # the default ``center`` model's fitted centre x
    )


__all__ = [
    "FIT_SPEC_NAME", "FIT_STATUS_INVALID", "FIT_STATUS_OK", "FitProcessor", "frame_fit",
]
