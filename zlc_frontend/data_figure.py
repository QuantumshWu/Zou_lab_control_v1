"""Static notebook rendering facade over one frozen frontend evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from dataclasses import fields, is_dataclass
from enum import Enum
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
    dataset_schema_retained_upper_bound_nbytes,
    fit_result_source_validation_additional_peak_upper_bound_nbytes,
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
    FigureEvaluationPolicy,
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
    *,
    validation_memory_limit_bytes: int | None,
) -> None:
    """Gate and validate sparse source lineage before Figure evaluation."""

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
        peak = fit_result_source_validation_additional_peak_upper_bound_nbytes(
            result,
            source_schema,
        )
        if (
            validation_memory_limit_bytes is not None
            and peak > validation_memory_limit_bytes
        ):
            raise MemoryError(
                f"fit source validation requires {peak} bytes; limit is "
                f"{validation_memory_limit_bytes}"
            )
        validate_fit_result_source_binding(result, source_ref, source_schema)
        if result.spec.committed_transform is not None:
            try:
                _validate_selection_fit_view(source_schema, result, layer.view)
            except ValueError as exc:
                raise ValueError(
                    "transformed fit overlay is not faithfully displayable: "
                    f"{exc}"
                ) from exc


def _metadata_retained_upper_bound_nbytes(value: object) -> int:
    """Conservatively charge immutable Python metadata, excluding ndarrays."""

    if isinstance(value, np.ndarray):
        return 0
    if is_dataclass(value) and not isinstance(value, type):
        return 512 + sum(
            _metadata_retained_upper_bound_nbytes(getattr(value, item.name))
            for item in fields(value)
        )
    if isinstance(value, Mapping):
        return 256 + sum(
            _metadata_retained_upper_bound_nbytes(key)
            + _metadata_retained_upper_bound_nbytes(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return 128 + sum(_metadata_retained_upper_bound_nbytes(item) for item in value)
    if isinstance(value, str):
        return 128 + 4 * len(value)
    if isinstance(value, (bytes, bytearray)):
        return 128 + len(value)
    if isinstance(value, Enum):
        return 128
    if value is None or isinstance(value, (bool, int, float, complex)):
        return 64
    # Axis ids and other tiny canonical wrappers intentionally receive a
    # generous fixed allowance without depending on their implementation.
    return 256


def figure_document_retained_upper_bound_nbytes(
    document: FigureDocument,
) -> int:
    """Bound the immutable document graph before evaluator/source allocation."""

    if not isinstance(document, FigureDocument):
        raise TypeError("document must be FigureDocument")
    return int(64 * 1024 + _metadata_retained_upper_bound_nbytes(document))


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
        "_render_memory_limit_bytes",
        "_source_schemas",
    )

    def __init__(
        self,
        document: FigureDocument,
        datasets: ResolvedDatasetMap,
        *,
        fit_results: Mapping[str, FitResultBatch] | None = None,
        evaluation_memory_limit_bytes: int | None = None,
        render_memory_limit_bytes: int | None = None,
    ) -> None:
        if not isinstance(document, FigureDocument):
            raise TypeError("document must be FigureDocument")
        if not isinstance(datasets, ResolvedDatasetMap):
            raise TypeError("datasets must be ResolvedDatasetMap")
        render_limit = self._validated_memory_limit(
            render_memory_limit_bytes,
            "render_memory_limit_bytes",
        )
        source_schemas = tuple(
            (
                descriptor.dataset_id,
                datasets.resolve(descriptor.dataset_id).block.schema,
            )
            for descriptor in document.datasets
        )

        if evaluation_memory_limit_bytes is None:
            policy = FigureEvaluationPolicy()
        else:
            if (
                isinstance(evaluation_memory_limit_bytes, bool)
                or not isinstance(evaluation_memory_limit_bytes, Integral)
                or evaluation_memory_limit_bytes <= 0
            ):
                raise ValueError(
                    "evaluation_memory_limit_bytes must be a positive integer or None"
                )
            policy = replace(
                FigureEvaluationPolicy(),
                max_live_nbytes=int(evaluation_memory_limit_bytes),
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
            validation_memory_limit_bytes=policy.max_live_nbytes,
        )
        evaluated = FigureEvaluator(policy).evaluate(document, datasets)
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
        self._render_memory_limit_bytes = render_limit
        self._source_schemas = source_schemas

    @property
    def document(self) -> FigureDocument:
        return self._document

    @property
    def evaluated(self):
        return self._evaluated

    @property
    def render_memory_limit_bytes(self) -> int | None:
        """Frozen default admission limit for every later render/export."""
        return self._render_memory_limit_bytes

    @property
    def has_fit_overlays(self) -> bool:
        """Whether this immutable figure carries an exact saved or draft fit."""
        return bool(self._fit_results)

    @property
    def fit_results_retained_upper_bound_nbytes(self) -> int:
        """Bytes retained only by the immutable Fit-result mapping."""

        from zlc_data import fit_result_retained_upper_bound_nbytes

        return int(
            sum(
                fit_result_retained_upper_bound_nbytes(result)
                for _layer_id, result in self._fit_results
            )
        )

    @property
    def retained_upper_bound_nbytes(self) -> int:
        """Conservative bytes strongly retained by this frozen figure.

        Composition roots subtract this value from the *one* operation budget
        before admitting a solver or another render front.  It is not a second
        independent allowance.
        """

        from .matplotlib_render import evaluated_figure_array_nbytes

        metadata = _metadata_retained_upper_bound_nbytes(
            (self._document, self._evaluated)
        ) + sum(
            dataset_schema_retained_upper_bound_nbytes(schema)
            for _dataset_id, schema in self._source_schemas
        )
        fit_results = self.fit_results_retained_upper_bound_nbytes
        return int(
            evaluated_figure_array_nbytes(self._evaluated)
            + metadata
            + fit_results
        )

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
            validation_memory_limit_bytes=self._render_memory_limit_bytes,
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
        clone._render_memory_limit_bytes = self._render_memory_limit_bytes
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

    def _typed_focus_retained_bound(
        self,
        source_layer,
        layer,
        cell,
        source_schemas,
    ) -> int:
        array_bytes = sum(
            int(series.data.samples.nbytes)
            for series in cell.series
            if isinstance(series.data, EvaluatedHistogram)
        )
        array_bytes += sum(
            int(series.data.values.nbytes + series.data.validity.nbytes)
            for series in cell.series
            if isinstance(series.data, EvaluatedCurve)
        )
        metadata_seed = (
            self._document.descriptor(layer.dataset_id),
            source_layer,
            self._evaluated.inputs,
            layer.layer_id,
            layer.dataset_id,
            cell,
            layer.resolutions,
        )
        return int(
            64 * 1024
            + array_bytes
            + _metadata_retained_upper_bound_nbytes(metadata_seed)
            + sum(
                dataset_schema_retained_upper_bound_nbytes(schema)
                for _dataset_id, schema in source_schemas
            )
        )

    def focused_typed_panel_retained_upper_bound_nbytes(
        self,
        panel_index: int,
        *,
        expected_intent: ViewIntent,
    ) -> int:
        """Bound one focus derivation without allocating its Figure DTO graph."""

        _index, source_layer, layer, cell, source_schemas = (
            self._typed_focus_parts(panel_index, expected_intent)
        )
        return self._typed_focus_retained_bound(
            source_layer,
            layer,
            cell,
            source_schemas,
        )

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
        retained_bound = self._typed_focus_retained_bound(
            source_layer,
            layer,
            cell,
            source_schemas,
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
        clone._render_memory_limit_bytes = self._render_memory_limit_bytes
        clone._source_schemas = source_schemas
        if clone.retained_upper_bound_nbytes > retained_bound:
            raise RuntimeError("focused typed DTO exceeded its preflight bound")
        return clone

    def render(
        self,
        *,
        dpi: float = 100.0,
        memory_limit_bytes: int | None = None,
    ):
        """Create a caller-owned Figure with canonical artist styles frozen in.

        The caller owns later Matplotlib mutations and draws.  Product-controlled PNG/export
        paths use the render owner's serialized compose API instead.
        """

        from .matplotlib_render import (
            render_evaluated_figure,
        )

        self._check_render_budget(dpi, memory_limit_bytes)

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
        memory_limit_bytes: int | None = None,
    ) -> bytes:
        from .matplotlib_render import save_evaluated_figure

        effective_limit = self._check_render_budget(dpi, memory_limit_bytes)
        output = BytesIO()
        save_evaluated_figure(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            output,
            image_format="png",
            dpi=dpi,
        )
        payload = output.getvalue()
        if effective_limit is not None and len(payload) > effective_limit:
            raise MemoryError("PNG payload exceeds figure render memory limit")
        return payload

    def to_png_bytes_with_panel_regions(
        self,
        *,
        dpi: float = 100.0,
        memory_limit_bytes: int | None = None,
    ) -> tuple[bytes, tuple[FigurePanelRegion, ...]]:
        """Encode the same frozen figure plus exact display-panel hit regions."""

        from .matplotlib_render import encode_evaluated_figure_with_panel_regions

        effective_limit = self._check_render_budget(dpi, memory_limit_bytes)
        payload, regions = encode_evaluated_figure_with_panel_regions(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            dpi=dpi,
        )
        if effective_limit is not None and len(payload) > effective_limit:
            raise MemoryError("PNG payload exceeds figure render memory limit")
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

    def radial_gaussian_image_fit_panels_preflight_nbytes(
        self,
        layer_id: str,
        *,
        artifact_identity: str,
    ) -> int:
        """Bound typed IMAGE projection before allocating panel DTOs or labels."""

        from .fit_image_projection import (
            radial_gaussian_image_fit_panels_additional_peak_upper_bound_nbytes,
        )

        result = self._fit_result_for_layer(layer_id)
        return radial_gaussian_image_fit_panels_additional_peak_upper_bound_nbytes(
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

    def single_panel_fit_overlay_preflight_nbytes(
        self,
        result: FitResultBatch | None = None,
        *,
        result_identity: str,
    ) -> tuple[int, int, int, int]:
        """Return validation, retained, prediction, and projection-peak bounds.

        This method performs only bounded metadata reads and integer arithmetic.
        The Workbench calls it before any sparse source validation, overlay plan,
        selection-index expansion, model evaluation, or raster allocation.
        """

        identity = canonical_text(result_identity, "fit result identity")
        if len(identity) > 4096:
            raise ValueError("fit result identity exceeds its display bound")
        if result is None:
            if len(self._fit_results) != 1:
                raise ValueError("canonical fit preflight requires one layer result")
            result = self._fit_results[0][1]
        elif not isinstance(result, FitResultBatch):
            raise TypeError("result must be FitResultBatch or None")
        source_schema = self._single_panel_source_schema()
        validation = (
            fit_result_source_validation_additional_peak_upper_bound_nbytes(
                result,
                source_schema,
            )
        )
        if len(self._document.layers) != 1 or len(self._evaluated.layers) != 1:
            raise ValueError("typed fit preflight requires one layer")
        layer = self._evaluated.layers[0]
        if len(layer.cells) != 1:
            raise ValueError("typed fit preflight requires one displayed cell")
        cell = layer.cells[0]
        intent = self._document.layers[0].view.intent
        identity_bytes = 4 * len(identity)
        if intent is ViewIntent.CURVE:
            from .figure import EvaluatedCurve

            prediction_bytes = 0
            retained = 4096
            for series in cell.series:
                if not isinstance(series.data, EvaluatedCurve):
                    raise ValueError("CURVE fit preflight found another series type")
                prediction_bytes += int(series.data.values.size) * 8
                retained += (
                    2048
                    + identity_bytes
                    + 4 * (512 + 32)
                    + 256 * len(series.batch_address)
                )
            if not cell.series:
                raise ValueError("CURVE fit preflight found no series")
            retained += prediction_bytes
            return (
                validation,
                retained,
                prediction_bytes,
                retained + 2 * 1024 * 1024,
            )
        if intent is ViewIntent.IMAGE:
            from .figure import EvaluatedImage

            if len(cell.series) != 1 or not isinstance(
                cell.series[0].data,
                EvaluatedImage,
            ):
                raise ValueError("IMAGE fit preflight requires one image series")
            descriptor = self._document.descriptor(
                self._document.layers[0].dataset_id
            )
            text_characters = len(descriptor.label)
            addresses = (
                *cell.facet_address,
                *cell.series[0].batch_address,
                *layer.resolutions,
            )
            for address in addresses:
                text_characters += len(address.axis_id.value) + 4
                coordinate = address.coordinate
                text_characters += len(coordinate) if isinstance(coordinate, str) else 64
            for reduction in cell.series[0].reductions:
                text_characters += len(reduction.method.value) + 32
                text_characters += sum(
                    len(axis_id.value) + 2 for axis_id in reduction.axis_ids
                )
                # Contributor counters are policy-bounded integers; 128 chars
                # covers both decimal renderings plus punctuation.
                text_characters += 128
            metadata_bytes = 4 * text_characters + 512 * (
                1 + len(cell.series[0].reductions)
            )
            retained = (
                64 * 1024
                + identity_bytes
                + metadata_bytes
                + 4 * (512 + 32)
            )
            # ``figure_panel_title`` first owns address/reduction fragments,
            # then their joins, then successive immutable title copies.  Four
            # simultaneous Unicode-sized copies conservatively cover that
            # construction before the bounded overlay becomes the retained one.
            projection_peak = retained + 16 * text_characters + 256 * 1024
            return validation, retained, 0, projection_peak
        raise ValueError("typed Fit overlay requires CURVE or IMAGE")

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
        memory_limit_bytes: int | None = None,
    ) -> Path:
        target = Path(path)
        if image_format is None:
            image_format = target.suffix.lstrip(".") or "png"
        if not target.suffix:
            target = target.with_suffix(f".{image_format}")
        from .matplotlib_render import save_evaluated_figure

        self._check_render_budget(dpi, memory_limit_bytes)
        save_evaluated_figure(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            target,
            image_format=image_format,
            dpi=dpi,
        )
        return target

    def _check_render_budget(
        self,
        dpi: float,
        memory_limit_bytes: int | None,
    ) -> int | None:
        requested = self._validated_memory_limit(
            memory_limit_bytes,
            "memory_limit_bytes",
        )
        frozen = self._render_memory_limit_bytes
        if frozen is not None and requested is not None and requested > frozen:
            raise ValueError(
                "memory_limit_bytes cannot weaken the DataFigure render limit"
            )
        effective = frozen if requested is None else requested
        if effective is None:
            return None
        from .matplotlib_render import estimate_render_peak_nbytes

        required = estimate_render_peak_nbytes(self._evaluated, dpi=dpi)
        if required > effective:
            raise MemoryError(
                f"figure render peak {required} exceeds limit {effective}"
            )
        return effective

    @staticmethod
    def _validated_memory_limit(value: int | None, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer or None")
        return int(value)


__all__ = [
    "DataFigure",
    "FigurePanelRegion",
    "figure_document_retained_upper_bound_nbytes",
]
