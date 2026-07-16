"""Static notebook rendering facade over one frozen frontend evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from io import BytesIO
from pathlib import Path

from zlc_data import FitResultBatch, validate_fit_result_source_binding

from .figure import (
    AxisViewRole,
    FigureDocument,
    FigureEvaluator,
    ResolvedDatasetMap,
    ViewIntent,
)


class DataFigure:
    """Own one immutable, already-resolved notebook figure.

    ``DataFigure`` never resolves repositories, sessions, devices, or live
    streams.  Construction evaluates the supplied frozen snapshots once and
    releases them; later renders consume only immutable presentation DTOs.
    """

    __slots__ = ("_document", "_evaluated", "_fit_results")

    def __init__(
        self,
        document: FigureDocument,
        datasets: ResolvedDatasetMap,
        *,
        fit_results: Mapping[str, FitResultBatch] | None = None,
    ) -> None:
        if not isinstance(document, FigureDocument):
            raise TypeError("document must be FigureDocument")
        if not isinstance(datasets, ResolvedDatasetMap):
            raise TypeError("datasets must be ResolvedDatasetMap")
        supplied = {} if fit_results is None else dict(fit_results)
        if any(not isinstance(key, str) or not key for key in supplied):
            raise TypeError("fit_results keys must be non-empty layer ids")
        if any(not isinstance(value, FitResultBatch) for value in supplied.values()):
            raise TypeError("fit_results values must be FitResultBatch")

        layers = {layer.layer_id: layer for layer in document.layers}
        fit_layers = {}
        for layer_id, result in supplied.items():
            try:
                layer = layers[layer_id]
            except KeyError as exc:
                raise ValueError(
                    f"fit overlay references unknown layer {layer_id!r}"
                ) from exc
            if result.spec.committed_transform is not None:
                raise ValueError(
                    "a transformed fit cannot be overlaid on a raw FigureDocument"
                )
            snapshot = datasets.resolve(layer.dataset_id)
            validate_fit_result_source_binding(result, snapshot.ref, snapshot.block.schema)
            fit_layers[layer_id] = (layer, result)

        evaluated = FigureEvaluator().evaluate(document, datasets)
        allowed_batch_roles = {
            AxisViewRole.BATCH,
            AxisViewRole.FACET,
            AxisViewRole.SELECTED,
            AxisViewRole.SLIDER,
        }
        for layer, result in fit_layers.values():
            fit_axes = result.fit_axis_specs
            if len(fit_axes) == 1:
                if (
                    layer.view.intent is not ViewIntent.CURVE
                    or layer.view.binding(fit_axes[0].axis_id).role is not AxisViewRole.X
                ):
                    raise ValueError("one-axis fit overlay requires its fitted axis as curve x")
            elif len(fit_axes) == 2:
                if (
                    layer.view.intent is not ViewIntent.IMAGE
                    or layer.view.binding(fit_axes[0].axis_id).role
                    is not AxisViewRole.IMAGE_X
                    or layer.view.binding(fit_axes[1].axis_id).role
                    is not AxisViewRole.IMAGE_Y
                ):
                    raise ValueError(
                        "two-axis fit overlay requires its fitted axes as image x/y"
                    )
            else:
                raise ValueError("only one- and two-axis fit overlays are supported")
            for axis in result.batch_axis_specs:
                if layer.view.binding(axis.axis_id).role not in allowed_batch_roles:
                    raise ValueError(
                        f"fit batch axis {axis.axis_id} is not uniquely displayed or selected"
                    )

        self._document = document
        self._evaluated = evaluated
        self._fit_results = tuple(sorted(supplied.items()))

    @property
    def document(self) -> FigureDocument:
        return self._document

    @property
    def evaluated(self):
        return self._evaluated

    def render(self):
        """Create a new caller-owned Matplotlib Figure using the OO Agg path."""

        from ._matplotlib_render import render_evaluated_figure

        return render_evaluated_figure(
            self._document,
            self._evaluated,
            dict(self._fit_results),
        )

    def to_png_bytes(self, *, dpi: float = 100.0) -> bytes:
        output = BytesIO()
        self.render().savefig(output, format="png", dpi=dpi)
        return output.getvalue()

    def _repr_png_(self) -> bytes:
        return self.to_png_bytes()

    def export(
        self,
        path: str | Path,
        *,
        image_format: str | None = None,
        dpi: float = 100.0,
    ) -> Path:
        target = Path(path)
        if image_format is None:
            image_format = target.suffix.lstrip(".") or "png"
        if not target.suffix:
            target = target.with_suffix(f".{image_format}")
        self.render().savefig(target, format=image_format, dpi=dpi)
        return target


__all__ = ["DataFigure"]
