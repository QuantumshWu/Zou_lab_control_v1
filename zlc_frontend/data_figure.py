"""Static Figure facade over one frozen frontend evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from numbers import Integral
from typing import TYPE_CHECKING

from zlc_data import (
    AxisSourceRef,
    DatasetSchema,
    FitResultBatch,
    HISTOGRAM_BIN,
    HistogramSpec,
    axis_source_ref_to_tree,
    validate_fit_result_source_binding,
)
from zlc_storage import canonical_digest, canonical_text

from .figure import (
    AxisResolution,
    AxisViewRole,
    DatasetId,
    EvaluatedCell,
    EvaluatedCurve,
    EvaluatedFigureData,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedLayer,
    EvaluatedMeter,
    FigureDocument,
    FigureEvaluator,
    FigureLayer,
    FixedIndex,
    ResolvedDatasetMap,
    SourceViewBinding,
    ViewIntent,
    validate_view_spec,
    view_spec_to_tree,
)
from .curve_display import numeric_curve_coordinates
from .figure.contract import _fit_display_selection_indices
from .render import PanelPresentationIdentity, RasterBuffer

if TYPE_CHECKING:
    from .fit_curve_projection import CurveFitOverlayPlan
    from .render import RadialGaussianImageFitOverlay


@dataclass(frozen=True, slots=True)
class FigurePanelRegion:
    """One panel hit target with source-aware focus and Fit storage identity.

    ``focus_address`` excludes dynamic evaluation resolutions so the same
    logical cell survives a newer live snapshot. ``fit_storage_index`` points
    directly into the attached immutable Fit result when one exists.
    """

    key: str
    focus_address: tuple[tuple[AxisSourceRef, int], ...]
    fit_storage_index: int | None
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        canonical_text(self.key, "figure panel key")
        from .fit_projection import canonical_panel_focus_address

        object.__setattr__(
            self,
            "focus_address",
            canonical_panel_focus_address(self.focus_address),
        )
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
    source_refs=(),
    source_and_view_validated: bool = False,
) -> tuple[tuple[str, FitResultBatch], ...]:
    """Validate overlays entirely from frozen schema/ref/evaluation facts."""

    supplied = {} if fit_results is None else dict(fit_results)
    if any(not isinstance(key, str) or not key for key in supplied):
        raise TypeError("fit_results keys must be non-empty layer ids")
    if any(not isinstance(value, FitResultBatch) for value in supplied.values()):
        raise TypeError("fit_results values must be FitResultBatch")
    if evaluated is not None and (
        document.document_id != evaluated.document_id
        or document.revision != evaluated.document_revision
    ):
        raise ValueError("document and evaluated data identities differ")

    layers = {layer.layer_id: layer for layer in document.layers}
    evaluated_inputs = (
        {}
        if evaluated is None
        else {item.dataset_id: item for item in evaluated.inputs}
    )
    refs = dict(source_refs)
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
            source_schema = schemas[layer.dataset_id]
            source_ref = (
                refs[layer.dataset_id]
                if evaluated is None
                else evaluated_inputs[layer.dataset_id].ref
            )
        except KeyError as exc:
            raise ValueError("fit layer source metadata is absent") from exc
        if not source_and_view_validated:
            validate_fit_result_source_binding(result, source_ref, source_schema)
        if not source_and_view_validated:
            transform = result.spec.committed_transform
            operations = tuple(transform.spec.operations)
            view_point_ordinals = tuple(
                range(source_schema.point_table.row_count)
                if layer.view.point_ordinals is None
                else layer.view.point_ordinals
            )
            if view_point_ordinals != transform.exact_point_ordinals:
                raise ValueError(
                    "Fit authority and Figure view select different point rows"
                )
            histogram = operations[-1] if operations else None
            if isinstance(histogram, HistogramSpec):
                if layer.view.intent is not ViewIntent.HISTOGRAM:
                    raise ValueError(
                        "histogram Fit authority requires a HISTOGRAM Figure"
                    )
                sample_sources = {
                    binding.source
                    for binding in layer.view.source_bindings
                    if binding.role is AxisViewRole.SAMPLE
                }
                if set(histogram.sources) != sample_sources:
                    raise ValueError(
                        "histogram Fit sample axes differ from the Figure view"
                    )
            else:
                _fit_display_selection_indices(source_schema, result)
        fit_layers[layer_id] = (layer, result)

    if evaluated is None:
        return tuple(sorted(supplied.items()))

    allowed_batch_roles = {
        AxisViewRole.BATCH,
        AxisViewRole.FACET,
        AxisViewRole.SELECTED,
    }
    for layer, result in fit_layers.values():
        fit_axes = result.fit_axis_specs
        if len(fit_axes) == 1:
            if layer.view.intent is ViewIntent.HISTOGRAM:
                transform = result.spec.committed_transform
                operations = () if transform is None else transform.spec.operations
                if (
                    fit_axes[0].role != HISTOGRAM_BIN
                    or not operations
                    or not isinstance(operations[-1], HistogramSpec)
                    or operations[-1].bin_axis_id != fit_axes[0].axis_id
                ):
                    raise ValueError(
                        "histogram Fit overlay requires its committed bin axis"
                    )
            elif (
                layer.view.intent is not ViewIntent.CURVE
                or layer.view.binding(result.spec.independent_sources[0]).role
                is not AxisViewRole.X
            ):
                raise ValueError("one-axis fit overlay requires its fitted axis as curve x")
        elif len(fit_axes) == 2:
            if (
                layer.view.intent is not ViewIntent.IMAGE
                or layer.view.binding(result.spec.independent_sources[0]).role
                is not AxisViewRole.IMAGE_X
                or layer.view.binding(result.spec.independent_sources[1]).role
                is not AxisViewRole.IMAGE_Y
            ):
                raise ValueError(
                    "two-axis fit overlay requires its fitted axes as image x/y"
                )
        else:
            raise ValueError("only one- and two-axis fit overlays are supported")
        for source, axis in zip(
            result.spec.batch_sources,
            result.batch_axis_specs,
            strict=True,
        ):
            role = layer.view.binding(source).role
            if role not in allowed_batch_roles and not (
                role is AxisViewRole.REDUCED and axis.size == 1
            ):
                raise ValueError(
                    f"fit batch axis {axis.axis_id} is not uniquely displayed or selected"
                )
    return tuple(sorted(supplied.items()))


class DataFigure:
    """Own one immutable, already-resolved Figure session.

    ``DataFigure`` never resolves repositories, sessions, devices, or live
    streams.  Construction evaluates the supplied frozen snapshots once and
    retains those same immutable references for exact archive export; it never
    copies them or turns them into a live source.  Later renders consume only
    immutable presentation DTOs.
    """

    __slots__ = (
        "_datasets",
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
        sources = tuple(
            (descriptor.dataset_id, datasets.resolve(descriptor.dataset_id))
            for descriptor in document.datasets
        )
        source_schemas = tuple(
            (dataset_id, snapshot.block.schema) for dataset_id, snapshot in sources
        )
        source_refs = tuple(
            (dataset_id, snapshot.ref) for dataset_id, snapshot in sources
        )
        _validated_fit_result_mapping(
            document,
            None,
            source_schemas,
            fit_results,
            source_refs=source_refs,
        )
        evaluated = FigureEvaluator().evaluate(document, datasets)
        validated_fit_results = _validated_fit_result_mapping(
            document,
            evaluated,
            source_schemas,
            fit_results,
            source_and_view_validated=True,
        )

        self._datasets = datasets
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
    def datasets(self) -> ResolvedDatasetMap:
        """Return the exact immutable source revisions used by this figure."""

        return self._datasets

    @property
    def fit_results(self) -> Mapping[str, FitResultBatch]:
        """Return a read-only mapping of the exact fit overlays."""

        from types import MappingProxyType

        return MappingProxyType(dict(self._fit_results))

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

        source_refs = tuple(
            (item.dataset_id, item.ref) for item in self._evaluated.inputs
        )
        _validated_fit_result_mapping(
            self._document,
            None,
            self._source_schemas,
            fit_results,
            source_refs=source_refs,
        )
        validated = _validated_fit_result_mapping(
            self._document,
            self._evaluated,
            self._source_schemas,
            fit_results,
            source_and_view_validated=True,
        )
        clone = object.__new__(type(self))
        clone._datasets = self._datasets
        clone._document = self._document
        clone._evaluated = self._evaluated
        clone._fit_results = validated
        clone._source_schemas = self._source_schemas
        return clone

    def _typed_focus_parts(
        self,
        panel_index: int,
        expected_intent: ViewIntent,
        series_index: int | None = None,
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
            ViewIntent.HISTOGRAM: EvaluatedHistogram,
            ViewIntent.IMAGE: EvaluatedImage,
            ViewIntent.METER: EvaluatedMeter,
        }.get(expected_intent)
        if data_type is None:
            raise ValueError(
                "focused typed panels require CURVE, HISTOGRAM, IMAGE, or METER"
            )
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
        from .fit_projection import evaluated_figure_panels

        panels = evaluated_figure_panels(self._evaluated)
        if panel_index >= len(panels):
            raise IndexError("panel_index is outside the frozen figure")
        panel_layer, cell, series_group = panels[panel_index]
        if series_index is not None:
            if isinstance(series_index, bool) or not isinstance(
                series_index,
                Integral,
            ):
                raise TypeError("series_index must be a non-negative integer or None")
            series_index = int(series_index)
            if not 0 <= series_index < len(series_group):
                raise IndexError("series_index is outside the focused panel")
            series_group = (series_group[series_index],)
        if panel_layer is not layer:
            raise RuntimeError("focused panel belongs to another evaluated layer")
        if not series_group or any(
            not isinstance(series.data, data_type) for series in series_group
        ):
            raise ValueError(
                f"focused {expected_intent.value} display requires one homogeneous panel"
            )
        value_units = {series.data.value_unit for series in series_group}
        if len(value_units) != 1:
            raise ValueError("focused typed panel mixes value units")
        if expected_intent is ViewIntent.CURVE:
            first_axis = series_group[0].data.x_axis
            numeric_curve_coordinates(first_axis)
            if any(series.data.x_axis != first_axis for series in series_group[1:]):
                raise ValueError("focused CURVE series do not share one exact x axis")
        if expected_intent is ViewIntent.IMAGE and len(series_group) != 1:
            raise ValueError("focused IMAGE display requires exactly one image")
        source_schemas = tuple(
            item for item in self._source_schemas if item[0] == layer.dataset_id
        )
        if len(source_schemas) != 1:
            raise RuntimeError("focused typed source schema is absent or ambiguous")
        return (
            panel_index,
            source_layer,
            layer,
            cell,
            tuple(series_group),
            source_schemas,
        )

    def focused_typed_panel(
        self,
        panel_index: int,
        *,
        expected_address: tuple[tuple[AxisSourceRef, int], ...],
        expected_intent: ViewIntent,
        series_index: int | None = None,
    ) -> DataFigure:
        """Derive one exact display-only typed panel from this frozen figure.

        The panel order and address are the same canonical facts used by the
        whole-figure renderer.  No dataset is resolved or evaluated again, and
        no display selection is promoted to an authority transform.
        """

        from .fit_projection import (
            canonical_panel_focus_address,
            panel_focus_address,
        )

        expected_address = canonical_panel_focus_address(expected_address)
        panel_index, source_layer, layer, cell, series_group, source_schemas = (
            self._typed_focus_parts(panel_index, expected_intent, series_index)
        )
        actual_address = panel_focus_address(layer, cell, series_group)
        if actual_address != expected_address:
            raise ValueError("panel address differs from the frozen overview region")

        focus_by_source = dict(actual_address)
        expected_sources = {
            binding.source
            for binding in source_layer.view.source_bindings
            if binding.role is AxisViewRole.FACET
            or (len(series_group) == 1 and binding.role is AxisViewRole.BATCH)
        }
        if expected_sources != set(focus_by_source):
            raise RuntimeError("focused panel address does not match its source view")

        schema = source_schemas[0][1]
        raw_focus = {
            source: index
            for source, index in actual_address
            if source.kind
            in {AxisSourceRef.POINT_ROWS, AxisSourceRef.POINT_COORDINATE}
        }
        point_ordinals = source_layer.view.point_ordinals
        if raw_focus:
            from .figure.contract import _resolved_point_group_records

            members = []
            for facet, batch, group_members, _group_index in (
                _resolved_point_group_records(schema, source_layer.view)
            ):
                addresses = (*facet, *(batch if len(series_group) == 1 else ()))
                record = {item.source: item.index for item in addresses}
                if all(record.get(source) == index for source, index in raw_focus.items()):
                    members.extend(group_members)
            point_ordinals = tuple(sorted(set(members)))
            if not point_ordinals:
                raise RuntimeError("focused raw point address resolved no physical row")

        selected_sources = {
            source
            for source in focus_by_source
            if source.kind in {AxisSourceRef.TENSOR, AxisSourceRef.GRID_DIMENSION}
        }
        bindings = tuple(
            replace(
                binding,
                role=AxisViewRole.SELECTED,
                selector=FixedIndex(focus_by_source[binding.source]),
                reduction=None,
            )
            if binding.source in selected_sources
            else binding
            for binding in source_layer.view.source_bindings
        )
        focused_view = replace(
            source_layer.view,
            source_bindings=bindings,
            point_ordinals=point_ordinals,
        )
        validate_view_spec(schema, focused_view)

        resolutions = {item.source: item for item in layer.resolutions}
        if len(resolutions) != len(layer.resolutions):
            raise RuntimeError("focused typed layer repeats a resolution source")
        focus_addresses = [*cell.facet_address]
        if len(series_group) == 1:
            focus_addresses.extend(series_group[0].batch_address)
        for address in focus_addresses:
            if address.source not in selected_sources:
                continue
            candidate = AxisResolution(
                address.source,
                "FIXED_INDEX",
                address.index,
                address.coordinate,
            )
            incumbent = resolutions.setdefault(address.source, candidate)
            if incumbent != candidate:
                raise RuntimeError("focused typed resolution conflicts with its facet")

        descriptor = self._document.descriptor(layer.dataset_id)
        identity = canonical_digest(
            {
                "schema": "zlc_frontend.FocusedTypedPanel",
                "source_document_id": self._document.document_id,
                "source_document_revision": self._document.revision,
                "dataset_id": layer.dataset_id.value,
                "schema_fingerprint": descriptor.schema_fingerprint,
                "intent": expected_intent.value,
                "layer_id": layer.layer_id,
                "view": view_spec_to_tree(focused_view),
                "panel_index": panel_index,
                "focus_address": tuple(
                    (axis_source_ref_to_tree(source), index)
                    for source, index in actual_address
                ),
            }
        )
        document_id = f"typed-focus-{identity}"
        focused_document = FigureDocument(
            document_id,
            0,
            (descriptor,),
            (
                FigureLayer(
                    source_layer.layer_id,
                    source_layer.dataset_id,
                    focused_view,
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
                    (
                        EvaluatedCell(
                            tuple(
                                address
                                for address in cell.facet_address
                                if address.source not in selected_sources
                            ),
                            tuple(
                                replace(
                                    series,
                                    batch_address=tuple(
                                        address
                                        for address in series.batch_address
                                        if address.source not in selected_sources
                                    ),
                                )
                                for series in series_group
                            ),
                        ),
                    ),
                    tuple(
                        resolutions[source]
                        for source in sorted(
                            resolutions,
                        )
                    ),
                ),
            ),
        )
        clone = object.__new__(type(self))
        clone._datasets = self._datasets
        clone._document = focused_document
        clone._evaluated = focused_evaluated
        # A focused panel is a display-only projection of the same immutable
        # Figure, not a new Fit result.  Retain the exact result owner so
        # the focused typed host can resolve the matching batch row from the
        # fixed resolutions above; never recompute or slice the Fit result.
        clone._fit_results = self._fit_results
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
        from .histogram_display import HistogramDisplayState

        _state, projection, overlays = self._histogram_render_values(
            HistogramDisplayState()
        )

        return render_evaluated_figure(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            dpi=dpi,
            histogram_projection=projection,
            histogram_fit_overlays=overlays,
        )

    def to_png_bytes(
        self,
        *,
        dpi: float = 100.0,
    ) -> bytes:
        return self.to_bytes(image_format="png", dpi=dpi)

    def to_bytes(
        self,
        *,
        image_format: str,
        dpi: float = 100.0,
    ) -> bytes:
        """Encode one owned figure payload without choosing a filesystem path."""

        if not isinstance(image_format, str):
            raise TypeError("image_format must be str")
        image_format = image_format.strip().lower()
        if image_format not in {"png", "pdf", "svg", "jpg", "jpeg"}:
            raise ValueError("image_format must be png, pdf, svg, jpg, or jpeg")
        from .matplotlib_render import encode_evaluated_figure
        from .histogram_display import HistogramDisplayState

        _state, projection, overlays = self._histogram_render_values(
            HistogramDisplayState()
        )

        return encode_evaluated_figure(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            image_format=image_format,
            dpi=dpi,
            histogram_projection=projection,
            histogram_fit_overlays=overlays,
        )

    def to_png_bytes_with_panel_regions(
        self,
        *,
        dpi: float = 100.0,
    ) -> tuple[bytes, tuple[FigurePanelRegion, ...]]:
        """Encode the same frozen figure plus exact display-panel hit regions."""

        from .matplotlib_render import encode_evaluated_figure_with_panel_regions
        from .histogram_display import HistogramDisplayState

        _state, projection, overlays = self._histogram_render_values(
            HistogramDisplayState()
        )

        payload, regions = encode_evaluated_figure_with_panel_regions(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            dpi=dpi,
            histogram_projection=projection,
            histogram_fit_overlays=overlays,
        )
        return payload, regions

    def to_panel_png_bytes_with_panel_regions(
        self,
        *,
        size: str,
        width: int,
        height: int,
        dpi: float,
        display_state: object,
        title: str,
        value_label: str,
    ) -> tuple[bytes, tuple[FigurePanelRegion, ...]]:
        """Encode a live grid into one named panel's fixed raster geometry.

        The ordinary ``to_png_bytes_with_panel_regions`` remains the archival
        page renderer.  A TaskConsole grid instead has a pre-existing panel
        size: every cell must subdivide that exact data box and the returned
        hit regions must come from the very same draw.
        """

        from .matplotlib_render import (
            encode_evaluated_panel_with_regions,
        )
        display_state, projection, overlays = self._histogram_render_values(
            display_state
        )

        return encode_evaluated_panel_with_regions(
            self._document,
            self._evaluated,
            dict(self._fit_results),
            size=size,
            width=width,
            height=height,
            dpi=dpi,
            display_state=display_state,
            title=title,
            value_label=value_label,
            histogram_projection=projection,
            histogram_fit_overlays=overlays,
        )

    def transient_fit_to_panel_png_bytes_with_panel_regions(
        self,
        result: FitResultBatch,
        *,
        size: str,
        width: int,
        height: int,
        dpi: float,
        display_state: object,
        title: str,
        value_label: str,
    ) -> tuple[bytes, tuple[FigurePanelRegion, ...]]:
        """Encode one draft Fit grid without attaching it to this DataFigure."""

        if not isinstance(result, FitResultBatch):
            raise TypeError("transient grid fit must be FitResultBatch")
        if len(self._document.layers) != 1:
            raise ValueError("transient grid fit requires exactly one layer")
        layer = self._document.layers[0]
        validated = dict(
            _validated_fit_result_mapping(
                self._document,
                self._evaluated,
                self._source_schemas,
                {layer.layer_id: result},
            )
        )
        from .matplotlib_render import encode_evaluated_panel_with_regions

        projection = None
        overlays = ()
        if self._document.layers[0].view.intent is ViewIntent.HISTOGRAM:
            display_state, projection, overlays = self._histogram_fit_presentation(
                result,
                result_identity=f"transient-fit:{id(result):x}",
                display_state=display_state,
            )

        return encode_evaluated_panel_with_regions(
            self._document,
            self._evaluated,
            validated,
            size=size,
            width=width,
            height=height,
            dpi=dpi,
            display_state=display_state,
            title=title,
            value_label=value_label,
            histogram_projection=projection,
            histogram_fit_overlays=overlays,
        )

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

    def _histogram_fit_presentation(
        self,
        result: FitResultBatch,
        *,
        result_identity: str,
        display_state,
        check_cancelled=None,
    ):
        """Build the exact committed Histogram projection and cell overlays."""

        from .fit_histogram_projection import (
            _histogram_fit_presentation,
        )

        return _histogram_fit_presentation(
            self,
            result,
            result_identity=result_identity,
            display_state=display_state,
            check_cancelled=check_cancelled,
        )

    def _histogram_render_values(self, display_state):
        """Return optional exact Histogram replay values for generic export."""

        if (
            not self._fit_results
            or len(self._document.layers) != 1
            or self._document.layers[0].view.intent is not ViewIntent.HISTOGRAM
        ):
            return display_state, None, ()
        result = dict(self._fit_results)[self._document.layers[0].layer_id]
        return self._histogram_fit_presentation(
            result,
            result_identity=f"embedded-fit:{id(result):x}",
            display_state=display_state,
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


@dataclass(frozen=True, slots=True)
class FacetedOverviewArtifact:
    """One indivisible rendered overview of one frozen ``DataFigure``."""

    figure: DataFigure
    raster: RasterBuffer
    regions: tuple[FigurePanelRegion, ...]
    logical_size: tuple[int, int]
    presentation: PanelPresentationIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.figure, DataFigure):
            raise TypeError("faceted overview figure must be DataFigure")
        if not isinstance(self.raster, RasterBuffer):
            raise TypeError("faceted overview raster must be RasterBuffer")
        regions = tuple(self.regions)
        if len(regions) <= 1 or any(
            not isinstance(item, FigurePanelRegion) for item in regions
        ):
            raise ValueError(
                "faceted overview requires multiple exact FigurePanelRegion values"
            )
        if len({item.key for item in regions}) != len(regions):
            raise ValueError("faceted overview region keys must be unique")
        if len({item.focus_address for item in regions}) != len(regions):
            raise ValueError("faceted overview regions require unique focus addresses")
        logical_size = tuple(self.logical_size)
        if len(logical_size) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in logical_size
        ):
            raise ValueError(
                "faceted overview logical_size must be two positive integers"
            )
        if not isinstance(self.presentation, PanelPresentationIdentity):
            raise TypeError(
                "faceted overview presentation must be PanelPresentationIdentity"
            )
        document = self.figure.document
        if (
            self.presentation.document_id != document.document_id
            or self.presentation.document_revision != document.revision
        ):
            raise ValueError(
                "faceted overview presentation belongs to another Figure"
            )
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "logical_size", logical_size)


__all__ = [
    "DataFigure",
    "FacetedOverviewArtifact",
    "FigurePanelRegion",
]
