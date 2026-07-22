"""Static notebook rendering facade over one frozen frontend evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from io import BytesIO
import math
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from zlc_data import (
    DatasetSchema,
    FitResultBatch,
    Selection,
    dataset_revision_ref_to_tree,
    validate_fit_result_source_binding,
)
from zlc_storage import canonical_digest, canonical_text

from .figure import (
    AxisResolution,
    AxisViewBinding,
    AxisViewRole,
    DatasetId,
    EvaluatedCell,
    EvaluatedCurve,
    EvaluatedFigureData,
    EvaluatedHistogram,
    EvaluatedLayer,
    EvaluatedMeter,
    FigureDocument,
    FigureEvaluator,
    FigureLayer,
    FixedIndex,
    ResolvedDatasetMap,
    ViewIntent,
)
from .figure.contract import _validate_selection_fit_view
from .curve_display import numeric_curve_coordinates

if TYPE_CHECKING:
    from .fit_curve_projection import CurveFitOverlayPlan
    from .fit_image_projection import RadialGaussianImageFitPanel
    from .render import RadialGaussianImageFitOverlay


@dataclass(frozen=True, slots=True)
class FigurePanelRegion:
    """One display-only panel hit target in normalized raster coordinates."""

    key: str
    selection: Selection | None
    fit_storage_index: int | None
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        canonical_text(self.key, "figure panel key")
        if self.selection is not None and not isinstance(self.selection, Selection):
            raise TypeError("panel selection must be Selection or None")
        if self.fit_storage_index is not None and (
            isinstance(self.fit_storage_index, bool)
            or not isinstance(self.fit_storage_index, Integral)
            or self.fit_storage_index < 0
        ):
            raise ValueError("fit_storage_index must be non-negative or None")
        if self.fit_storage_index is not None:
            object.__setattr__(self, "fit_storage_index", int(self.fit_storage_index))
        bounds = tuple(float(value) for value in (
            self.left,
            self.top,
            self.right,
            self.bottom,
        ))
        if any(not math.isfinite(value) for value in bounds):
            raise ValueError("panel bounds must be finite")
        left, top, right, bottom = bounds
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise ValueError("panel bounds must be an ordered normalized rectangle")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "top", top)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "bottom", bottom)

    def contains(self, x: float, y: float) -> bool:
        x_value, y_value = float(x), float(y)
        return (
            self.left <= x_value <= self.right
            and self.top <= y_value <= self.bottom
        )


def _validated_fit_result_mapping(
    document: FigureDocument,
    evaluated,
    source_schemas: tuple[tuple[DatasetId, DatasetSchema], ...],
    fit_results: Mapping[str, FitResultBatch] | None,
    *,
    source_bindings_validated: bool = False,
    selection_views_validated: bool = False,
) -> tuple[tuple[str, FitResultBatch], ...]:
    """Validate overlays entirely from frozen schema/ref/evaluation facts."""

    supplied = {} if fit_results is None else dict(fit_results)
    if any(not isinstance(key, str) or not key for key in supplied):
        raise TypeError("fit_results keys must be non-empty layer ids")
    if any(not isinstance(value, FitResultBatch) for value in supplied.values()):
        raise TypeError("fit_results values must be FitResultBatch")
    if (
        document.document_id != evaluated.document_id
        or document.revision != evaluated.document_revision
    ):
        raise ValueError("document and evaluated data identities differ")

    layers = {layer.layer_id: layer for layer in document.layers}
    evaluated_inputs = {item.dataset_id: item for item in evaluated.inputs}
    schemas = dict(source_schemas)
    fit_layers = {}
    for layer_id, result in supplied.items():
        try:
            layer = layers[layer_id]
        except KeyError as exc:
            raise ValueError(
                f"fit overlay references unknown layer {layer_id!r}"
            ) from exc
        try:
            evaluated_input = evaluated_inputs[layer.dataset_id]
            source_schema = schemas[layer.dataset_id]
        except KeyError as exc:
            raise ValueError("fit layer source metadata is absent") from exc
        if (
            result.spec.committed_transform is not None
            and not selection_views_validated
        ):
            try:
                _validate_selection_fit_view(
                    source_schema,
                    result,
                    layer.view,
                )
            except ValueError as exc:
                raise ValueError(
                    "transformed fit overlay is not faithfully displayable: "
                    f"{exc}"
                ) from exc
        if not source_bindings_validated:
            validate_fit_result_source_binding(
                result,
                evaluated_input.ref,
                source_schema,
            )
        fit_layers[layer_id] = (layer, result)

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
    return tuple(sorted(supplied.items()))


def _validate_fit_result_sources_before_evaluation(
    document: FigureDocument,
    source_refs,
    source_schemas: tuple[tuple[DatasetId, DatasetSchema], ...],
    fit_results: Mapping[str, FitResultBatch] | None,
) -> None:
    """Validate sparse source lineage before Figure evaluation."""

    supplied = {} if fit_results is None else dict(fit_results)
    if not supplied:
        return
    layers = {layer.layer_id: layer for layer in document.layers}
    refs = dict(source_refs)
    schemas = dict(source_schemas)
    for layer_id, result in supplied.items():
        if not isinstance(layer_id, str) or not layer_id:
            raise TypeError("fit_results keys must be non-empty layer ids")
        if not isinstance(result, FitResultBatch):
            raise TypeError("fit_results values must be FitResultBatch")
        try:
            layer = layers[layer_id]
            source_ref = refs[layer.dataset_id]
            source_schema = schemas[layer.dataset_id]
        except KeyError as exc:
            raise ValueError("fit layer source metadata is absent") from exc
        validate_fit_result_source_binding(result, source_ref, source_schema)
        if result.spec.committed_transform is not None:
            try:
                _validate_selection_fit_view(source_schema, result, layer.view)
            except ValueError as exc:
                raise ValueError(
                    "transformed fit overlay is not faithfully displayable: "
                    f"{exc}"
                ) from exc


class DataFigure:
    """Own one immutable, already-resolved notebook figure.

    ``DataFigure`` never resolves repositories, sessions, devices, or live
    streams.  Construction evaluates the supplied frozen snapshots once and
    releases them; later renders consume only immutable presentation DTOs.
    """

    __slots__ = (
        "_document",
        "_evaluated",
        "_fit_results",
        "_source_schemas",
    )

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
        source_schemas = tuple(
            (
                descriptor.dataset_id,
                datasets.resolve(descriptor.dataset_id).block.schema,
            )
            for descriptor in document.datasets
        )

        _validate_fit_result_sources_before_evaluation(
            document,
            tuple(
                (
                    descriptor.dataset_id,
                    datasets.resolve(descriptor.dataset_id).ref,
                )
                for descriptor in document.datasets
            ),
            source_schemas,
            fit_results,
        )
        evaluated = FigureEvaluator().evaluate(document, datasets)
        validated_fit_results = _validated_fit_result_mapping(
            document,
            evaluated,
            source_schemas,
            fit_results,
            source_bindings_validated=True,
            selection_views_validated=True,
        )

        self._document = document
        self._evaluated = evaluated
        self._fit_results = validated_fit_results
        self._source_schemas = source_schemas

    @property
    def document(self) -> FigureDocument:
        return self._document

    @property
    def evaluated(self):
        return self._evaluated

    @property
    def has_fit_overlays(self) -> bool:
        """Whether this immutable figure carries an exact saved or draft fit."""
        return bool(self._fit_results)

    def with_fit_results(
        self,
        fit_results: Mapping[str, FitResultBatch] | None,
    ) -> DataFigure:
        """Clone this frozen figure with another exact fit-result mapping.

        Source materialization and view evaluation are intentionally *not*
        repeated.  The clone reuses the identical immutable evaluated arrays,
        while complete source-ref/schema, transform/view, fit-axis and sparse
        batch-layout validation is repeated against the replacement result.
        This is the only fit replay mutation seam: no repository or analysis
        authority becomes reachable through ``DataFigure``.
        """

        _validate_fit_result_sources_before_evaluation(
            self._document,
            tuple((item.dataset_id, item.ref) for item in self._evaluated.inputs),
            self._source_schemas,
            fit_results,
        )
        validated = _validated_fit_result_mapping(
            self._document,
            self._evaluated,
            self._source_schemas,
            fit_results,
            source_bindings_validated=True,
            selection_views_validated=True,
        )
        clone = object.__new__(type(self))
        clone._document = self._document
        clone._evaluated = self._evaluated
        clone._fit_results = validated
        clone._source_schemas = self._source_schemas
        return clone

    def _typed_focus_parts(
        self,
        panel_index: int,
        expected_intent: ViewIntent,
    ):
        if isinstance(panel_index, bool) or not isinstance(panel_index, Integral):
            raise TypeError("panel_index must be a non-negative integer")
        panel_index = int(panel_index)
        if panel_index < 0:
            raise ValueError("panel_index must be a non-negative integer")
        if not isinstance(expected_intent, ViewIntent):
            raise TypeError("expected_intent must be ViewIntent")
        data_type = {
            ViewIntent.CURVE: EvaluatedCurve,
            ViewIntent.METER: EvaluatedMeter,
            ViewIntent.HISTOGRAM: EvaluatedHistogram,
        }.get(expected_intent)
        if data_type is None:
            raise ValueError(
                "focused typed panels currently support CURVE, METER, or HISTOGRAM"
            )
        if self._fit_results:
            raise ValueError("focused typed display does not accept fit overlays")
        if (
            len(self._document.layers) != 1
            or len(self._evaluated.layers) != 1
            or len(self._evaluated.inputs) != 1
        ):
            raise ValueError("focused typed display requires one layer and input")
        source_layer = self._document.layers[0]
        layer = self._evaluated.layers[0]
        if (
            source_layer.layer_id != layer.layer_id
            or source_layer.dataset_id != layer.dataset_id
            or source_layer.view.intent is not expected_intent
        ):
            raise RuntimeError("document/evaluated typed layer identity differs")
        if panel_index >= len(layer.cells):
            raise IndexError("panel_index is outside the frozen figure")
        cell = layer.cells[panel_index]
        if not cell.series or any(
            not isinstance(series.data, data_type) for series in cell.series
        ):
            raise ValueError(
                f"focused {expected_intent.value} display requires one homogeneous panel"
            )
        value_units = {series.data.value_unit for series in cell.series}
        if len(value_units) != 1:
            raise ValueError("focused typed panel mixes value units")
        if expected_intent is ViewIntent.CURVE:
            first_axis = cell.series[0].data.x_axis
            numeric_curve_coordinates(first_axis)
            if any(series.data.x_axis != first_axis for series in cell.series[1:]):
                raise ValueError("focused CURVE series do not share one exact x axis")
        source_schemas = tuple(
            item for item in self._source_schemas if item[0] == layer.dataset_id
        )
        if len(source_schemas) != 1:
            raise RuntimeError("focused typed source schema is absent or ambiguous")
        return panel_index, source_layer, layer, cell, source_schemas

    def focused_typed_panel(
        self,
        panel_index: int,
        *,
        expected_selection: Selection | None,
        expected_intent: ViewIntent,
    ) -> DataFigure:
        """Derive one exact display-only typed panel from this frozen figure.

        The panel order and selection are the same canonical facts used by the
        whole-figure renderer.  No dataset is resolved or evaluated again, and
        no display selection is promoted to an authority transform.
        """

        if expected_selection is not None and not isinstance(
            expected_selection,
            Selection,
        ):
            raise TypeError("expected_selection must be Selection or None")
        panel_index, source_layer, layer, cell, source_schemas = (
            self._typed_focus_parts(panel_index, expected_intent)
        )
        # Imported lazily because fit_image_projection also imports the public
        # FigurePanelRegion DTO from this module.
        from .fit_image_projection import fit_panel_selection

        series_group = cell.series
        actual_selection = fit_panel_selection(layer, cell, series_group, None)
        if actual_selection != expected_selection:
            raise ValueError("panel selection differs from the frozen overview region")

        facet_by_axis = {address.axis_id: address for address in cell.facet_address}
        if len(facet_by_axis) != len(cell.facet_address):
            raise RuntimeError("focused typed facet address repeats an axis")
        facet_bindings = {
            binding.axis_id
            for binding in source_layer.view.axis_bindings
            if binding.role is AxisViewRole.FACET
        }
        if facet_bindings != set(facet_by_axis):
            raise RuntimeError("focused typed facet addresses do not match its view")
        bindings = tuple(
            AxisViewBinding(
                binding.axis_id,
                AxisViewRole.SELECTED,
                selector=FixedIndex(facet_by_axis[binding.axis_id].index),
            )
            if binding.role is AxisViewRole.FACET
            else binding
            for binding in source_layer.view.axis_bindings
        )

        resolutions = {item.axis_id: item for item in layer.resolutions}
        if len(resolutions) != len(layer.resolutions):
            raise RuntimeError("focused typed layer repeats a resolution axis")
        for address in cell.facet_address:
            candidate = AxisResolution(
                address.axis_id,
                "FIXED_INDEX",
                address.index,
                address.coordinate,
            )
            incumbent = resolutions.setdefault(address.axis_id, candidate)
            if incumbent != candidate:
                raise RuntimeError("focused typed resolution conflicts with its facet")

        identity = canonical_digest(
            {
                "schema": "zlc_frontend.FocusedTypedPanel",
                "source_document_id": self._document.document_id,
                "source_document_revision": self._document.revision,
                "dataset_id": layer.dataset_id.value,
                "intent": expected_intent.value,
                "dataset_revision_ref": dataset_revision_ref_to_tree(
                    self._evaluated.inputs[0].ref
                ),
                "layer_id": layer.layer_id,
                "panel_index": panel_index,
                "facet_indices": tuple(
                    (address.axis_id.value, address.index)
                    for address in sorted(
                        cell.facet_address,
                        key=lambda item: item.axis_id.value,
                    )
                ),
            }
        )
        document_id = f"typed-focus-{identity}"
        descriptor = self._document.descriptor(layer.dataset_id)
        focused_document = FigureDocument(
            document_id,
            0,
            (descriptor,),
            (
                FigureLayer(
                    source_layer.layer_id,
                    source_layer.dataset_id,
                    replace(source_layer.view, axis_bindings=bindings),
                ),
            ),
        )
        focused_evaluated = EvaluatedFigureData(
            document_id,
            0,
            self._evaluated.inputs,
            (
                EvaluatedLayer(
                    layer.layer_id,
                    layer.dataset_id,
                    (EvaluatedCell((), tuple(series_group)),),
                    tuple(
                        resolutions[axis_id]
                        for axis_id in sorted(
                            resolutions,
                            key=lambda item: item.value,
                        )
                    ),
                ),
            ),
        )
        clone = object.__new__(type(self))
        clone._document = focused_document
        clone._evaluated = focused_evaluated
        clone._fit_results = ()
        clone._source_schemas = source_schemas
        return clone

    def render(
        self,
        *,
        dpi: float = 100.0,
    ):
        """Create a caller-owned Figure with canonical artist styles frozen in.

        The caller owns later Matplotlib mutations and draws.  Product-controlled PNG/export
        paths use the render owner's serialized compose API instead.
        """

        from .matplotlib_render import (
            render_evaluated_figure,
        )

        return render_evaluated_figure(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            dpi=dpi,
        )

    def to_png_bytes(
        self,
        *,
        dpi: float = 100.0,
    ) -> bytes:
        from .matplotlib_render import save_evaluated_figure

        output = BytesIO()
        save_evaluated_figure(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            output,
            image_format="png",
            dpi=dpi,
        )
        return output.getvalue()

    def to_png_bytes_with_panel_regions(
        self,
        *,
        dpi: float = 100.0,
    ) -> tuple[bytes, tuple[FigurePanelRegion, ...]]:
        """Encode the same frozen figure plus exact display-panel hit regions."""

        from .matplotlib_render import encode_evaluated_figure_with_panel_regions

        payload, regions = encode_evaluated_figure_with_panel_regions(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            dpi=dpi,
        )
        return payload, regions

    def radial_gaussian_image_fit_panels(
        self,
        layer_id: str,
        *,
        artifact_identity: str,
    ) -> tuple[RadialGaussianImageFitPanel, ...]:
        """Return typed saved-fit IMAGE panels without exposing fit authority.

        The immutable projections retain exact source/artifact identity, sparse
        logical holes, authoritative axis metadata, focus summaries, and only
        the published centre/radius annotation.  No solver or predicted image
        is evaluated on this path.
        """

        from .fit_image_projection import radial_gaussian_image_fit_panels

        result = self._fit_result_for_layer(layer_id)

        return radial_gaussian_image_fit_panels(
            self._document,
            self._evaluated,
            result,
            layer_id,
            artifact_identity=artifact_identity,
        )

    def _fit_result_for_layer(self, layer_id: str) -> FitResultBatch:
        resolved = canonical_text(layer_id, "fit layer_id")
        for candidate, result in self._fit_results:
            if candidate == resolved:
                return result
        raise ValueError(f"layer {resolved!r} has no saved fit result")

    def single_panel_curve_fit_overlay_plan(
        self,
        *,
        result_identity: str,
    ) -> CurveFitOverlayPlan:
        """Freeze canonical CURVE overlay work without evaluating a model."""

        from .fit_curve_projection import single_panel_curve_fit_overlay_plan

        return single_panel_curve_fit_overlay_plan(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            self._single_panel_source_schema(),
            result_identity=result_identity,
        )

    def transient_single_panel_curve_fit_overlay_plan(
        self,
        result: FitResultBatch,
        *,
        result_identity: str,
    ) -> CurveFitOverlayPlan:
        """Freeze transient CURVE overlay work without evaluating a model."""

        from .fit_curve_projection import (
            transient_single_panel_curve_fit_overlay_plan,
        )

        return transient_single_panel_curve_fit_overlay_plan(
            self._document,
            self._evaluated,
            self._single_panel_source_schema(),
            result,
            result_identity=result_identity,
        )

    def single_panel_radial_fit_overlay(
        self,
        *,
        result_identity: str,
    ) -> RadialGaussianImageFitOverlay:
        """Return one exact radial IMAGE annotation for typed replay.

        Generic two-dimensional model contours remain outside the typed IMAGE
        contract.  This seam accepts only the existing named radial-Gaussian
        projection and exactly one logical panel.
        """

        if len(self._document.layers) != 1 or len(self._evaluated.layers) != 1:
            raise ValueError("typed radial fit projection requires exactly one layer")
        if len(self._evaluated.inputs) != 1:
            raise ValueError("typed radial fit projection requires exactly one input")
        layer = self._document.layers[0]
        if set(dict(self._fit_results)) != {layer.layer_id}:
            raise ValueError("typed radial fit projection requires one exact layer result")
        panels = self.radial_gaussian_image_fit_panels(
            layer.layer_id,
            artifact_identity=result_identity,
        )
        if len(panels) != 1:
            raise ValueError("typed radial fit projection requires exactly one IMAGE panel")
        return panels[0].fit_overlay

    def transient_single_panel_radial_fit_overlay(
        self,
        result: FitResultBatch,
        *,
        result_identity: str,
        check_cancelled=None,
    ) -> RadialGaussianImageFitOverlay:
        """Project one draft radial annotation over the unchanged full image."""

        from .fit_image_projection import transient_single_panel_radial_fit_overlay

        return transient_single_panel_radial_fit_overlay(
            self._document,
            self._evaluated,
            self._single_panel_source_schema(),
            result,
            result_identity=result_identity,
            check_cancelled=check_cancelled,
        )

    def _single_panel_source_schema(self) -> DatasetSchema:
        if len(self._document.layers) != 1:
            raise ValueError("typed fit projection requires exactly one layer")
        dataset_id = self._document.layers[0].dataset_id
        try:
            return dict(self._source_schemas)[dataset_id]
        except KeyError as exc:  # pragma: no cover - constructor closes this
            raise RuntimeError("typed fit source schema is absent") from exc

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
        from .matplotlib_render import save_evaluated_figure

        save_evaluated_figure(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            target,
            image_format=image_format,
            dpi=dpi,
        )
        return target


__all__ = [
    "DataFigure",
    "FigurePanelRegion",
]
