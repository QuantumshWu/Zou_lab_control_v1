"""Headless render hand-off values owned by the target frontend package.

The renderer may use Matplotlib, Qt, or neither, but the worker/GUI boundary is
always an immutable :class:`BoardFrame`.  No live Figure, Artist, mutable or
aliased ndarray view, or QImage storage crosses this module's boundary.  The
exact image, curve, and typed site-map interaction payloads are allowed
because their evaluated arrays are intrinsically backed by owned immutable
bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from zlc_data import (
    AxisSpec,
    BlockId,
    CoordinateFrameId,
    DatasetRevisionRef,
    FitBatchStatus,
    SITE,
    StreamGenerationId,
)
from zlc_storage import (
    canonical_text as _text,
    nonnegative_integer as _nonnegative,
    sha256_text,
)

from .curve_display import CurveViewportTransform, NumericViewportTransform
from .display_range import validated_display_range
from .histogram_display import HistogramBinProjection, HistogramViewportTransform
from .image_display import ImageColormap
from .figure import (
    AxisAddress,
    DatasetId,
    EvaluatedCurve,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedInput,
    EvaluatedMeter,
    EvaluatedSeries,
)
from .image_view import ImageViewportTransform
from .site_map import immutable_site_state


def detached_render_fault(error: BaseException) -> RuntimeError:
    """Return string-only diagnostics without retaining a failed render stack."""

    return RuntimeError(f"{type(error).__name__}: {error}")


@dataclass(frozen=True)
class SourceIdentity:
    """Identity of the dataset producer rendered by one panel.

    This value deliberately contains no run/shot position.  Producer identity
    changes when the dataset, schema, or producer generation changes; display
    coherence is represented independently by :class:`CoherenceStamp`.
    """

    dataset_id: DatasetId
    block_id: BlockId
    stream_generation: StreamGenerationId
    schema_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, DatasetId):
            raise TypeError("dataset_id must be DatasetId")
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(self.stream_generation, StreamGenerationId):
            raise TypeError("stream_generation must be StreamGenerationId")
        object.__setattr__(
            self,
            "schema_fingerprint",
            sha256_text(self.schema_fingerprint, "schema_fingerprint"),
        )


@dataclass(frozen=True, slots=True)
class DocumentInputIdentity:
    """Exact immutable document revision rendered by a presentation surface.

    A document is not a dataset producer.  Its identity therefore contains no
    run, join, block, stream-generation, or schema fields.  ``content_digest``
    binds the revision label to the exact content that was rasterized.
    """

    document_id: str
    document_revision: int
    content_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_id",
            _text(self.document_id, "document_id"),
        )
        object.__setattr__(
            self,
            "document_revision",
            _nonnegative(self.document_revision, "document_revision"),
        )
        object.__setattr__(
            self,
            "content_digest",
            sha256_text(self.content_digest, "content_digest"),
        )


@dataclass(frozen=True)
class CoherenceStamp:
    """Exact evaluated inputs shared by one coherent dataset panel group."""

    inputs: tuple[EvaluatedInput, ...]

    def __post_init__(self) -> None:
        inputs = tuple(self.inputs)
        if not inputs or any(not isinstance(value, EvaluatedInput) for value in inputs):
            raise ValueError("inputs must contain at least one EvaluatedInput")
        input_ids = tuple(value.dataset_id for value in inputs)
        if len(set(input_ids)) != len(input_ids):
            raise ValueError("CoherenceStamp input dataset ids must be unique")
        object.__setattr__(
            self,
            "inputs",
            tuple(sorted(inputs, key=lambda value: value.dataset_id.value)),
        )


@dataclass(frozen=True)
class RasterBuffer:
    """One tight owned immutable RGBA8888 raster."""

    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        width = _nonnegative(self.width, "width")
        height = _nonnegative(self.height, "height")
        if width == 0 or height == 0:
            raise ValueError("raster width and height must be positive")
        if not isinstance(self.pixels, bytes):
            raise TypeError("pixels must be owned immutable bytes")
        if len(self.pixels) != width * height * 4:
            raise ValueError("RGBA8888 pixels length must equal width * height * 4")

    @classmethod
    def from_agg_rgba(cls, width: int, height: int, buffer) -> "RasterBuffer":
        """Own a copy of one Agg RGBA buffer as a tight-stride RGBA8888 raster.

        Every Agg surface hands over the same three facts -- the layout is
        ``RGBA8888``, the stride is tight, and the worker's live buffer must be
        COPIED rather than aliased -- so they are stated here once.  A caller that
        re-typed them could drift on any of the three, and the third is the one
        that corrupts pixels rather than raising: ``memoryview`` and ``bytearray``
        both satisfy a length check while still aliasing the buffer the worker is
        about to overwrite.
        """

        return cls(width, height, bytes(buffer))


@dataclass(frozen=True, slots=True)
class RadialGaussianImageFitOverlay:
    """Exact saved-fit annotation for one radial-Gaussian IMAGE cell.

    The fit result remains owned by the analysis/artifact layer.  This small
    presentation value carries only the already-published centre/radius facts,
    their exact source/artifact identity, and a status diagnostic.  It
    can therefore be painted by Agg or Qt without either renderer seeing a
    ``FitResultBatch`` or re-running a solver.
    """

    source_ref: DatasetRevisionRef
    artifact_identity: str
    batch_storage_index: int | None
    status: FitBatchStatus | None
    coordinate_frame: CoordinateFrameId
    caption: str
    diagnostic: str
    center_xy: tuple[float, float] | None = None
    one_over_e_radius: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, DatasetRevisionRef):
            raise TypeError("fit overlay source_ref must be DatasetRevisionRef")
        object.__setattr__(
            self,
            "artifact_identity",
            _text(self.artifact_identity, "fit overlay artifact_identity"),
        )
        storage = self.batch_storage_index
        if storage is not None:
            storage = _nonnegative(storage, "fit overlay batch_storage_index")
            object.__setattr__(self, "batch_storage_index", storage)
        status = self.status
        if status is not None and not isinstance(status, FitBatchStatus):
            raise TypeError("fit overlay status must be FitBatchStatus or None")
        if storage is None:
            if status is not None:
                raise ValueError("an absent fit cell cannot carry a fit status")
        elif status is None:
            raise ValueError("a stored fit cell requires a fit status")
        if not isinstance(self.coordinate_frame, CoordinateFrameId):
            raise TypeError("fit overlay coordinate_frame must be CoordinateFrameId")
        object.__setattr__(self, "caption", _text(self.caption, "fit overlay caption"))
        if not isinstance(self.diagnostic, str):
            raise TypeError("fit overlay diagnostic must be str")
        center = self.center_xy
        radius = self.one_over_e_radius
        if status is FitBatchStatus.CONVERGED:
            if not isinstance(center, tuple) or len(center) != 2:
                raise TypeError("converged radial fit requires center_xy")
            checked_center = tuple(float(value) for value in center)
            if any(not np.isfinite(value) for value in checked_center):
                raise ValueError("fit overlay center must be finite")
            if (
                isinstance(radius, bool)
                or not isinstance(radius, (int, float, np.number))
                or not np.isfinite(float(radius))
                or float(radius) <= 0.0
            ):
                raise ValueError("converged radial fit radius must be finite and positive")
            object.__setattr__(self, "center_xy", checked_center)
            object.__setattr__(self, "one_over_e_radius", float(radius))
        elif center is not None or radius is not None:
            raise ValueError("non-converged or absent fit cells cannot carry geometry")

    @property
    def result_identity(self) -> str:
        """Return the exact saved or draft result identity.

        ``artifact_identity`` is retained as the stored field for the existing
        saved-fit codec/UI boundary.  Typed replay also admits an unsaved draft
        identity, so consumers use this neutral alias and never infer whether
        the result has been published.
        """

        return self.artifact_identity


@dataclass(frozen=True, slots=True, eq=False)
class CurveFitOverlay:
    """One immutable fitted prediction bound to one exact CURVE series.

    The DTO deliberately carries no model, solver, repository, or mutable fit
    result.  ``source_sample_span`` names the contiguous authority ROI inside
    the unchanged cached curve.  ``coordinates`` may carry a worker-bounded
    paint sampling of that span; otherwise ``predicted_y`` covers the full
    span.  Failed and sparse rows own an empty prediction and no coordinates.
    """

    source_ref: DatasetRevisionRef
    result_identity: str
    series_batch_address: tuple[AxisAddress, ...]
    source_sample_span: tuple[int, int]
    batch_storage_index: int | None
    status: FitBatchStatus | None
    diagnostic: str
    predicted_y: np.ndarray
    coordinates: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, DatasetRevisionRef):
            raise TypeError("curve fit overlay source_ref must be DatasetRevisionRef")
        identity = _text(self.result_identity, "curve fit result_identity")
        object.__setattr__(self, "result_identity", identity)

        address = tuple(self.series_batch_address)
        if any(not isinstance(item, AxisAddress) for item in address):
            raise TypeError("curve fit series_batch_address requires AxisAddress values")
        object.__setattr__(self, "series_batch_address", address)

        span = tuple(self.source_sample_span)
        if len(span) != 2:
            raise ValueError("curve fit source_sample_span must contain start and stop")
        start = _nonnegative(span[0], "curve fit source span start")
        stop = _nonnegative(span[1], "curve fit source span stop")
        if stop <= start:
            raise ValueError("curve fit source_sample_span must be nonempty")
        object.__setattr__(self, "source_sample_span", (start, stop))

        storage = self.batch_storage_index
        if storage is not None:
            storage = _nonnegative(storage, "curve fit batch_storage_index")
            object.__setattr__(self, "batch_storage_index", storage)
        status = self.status
        if status is not None and not isinstance(status, FitBatchStatus):
            raise TypeError("curve fit overlay status must be FitBatchStatus or None")
        if storage is None:
            if status is not None:
                raise ValueError("an absent curve fit cell cannot carry a status")
        elif status is None:
            raise ValueError("a stored curve fit cell requires a status")

        if not isinstance(self.diagnostic, str):
            raise TypeError("curve fit overlay diagnostic must be str")

        predicted = np.asarray(self.predicted_y)
        if predicted.dtype != np.dtype("<f8") or predicted.ndim != 1:
            raise TypeError("curve fit predicted_y must be a one-dimensional float64 array")
        # Immutable bytes are the ownership boundary.  Merely setting
        # writeable=False on a caller-owned ndarray would still permit its base
        # buffer to be changed behind this front.
        owned = np.frombuffer(predicted.tobytes(order="C"), dtype=np.dtype("<f8"))
        owned.setflags(write=False)
        object.__setattr__(self, "predicted_y", owned)
        coordinates = self.coordinates
        owned_coordinates = None
        if coordinates is not None:
            coordinates = np.asarray(coordinates)
            if coordinates.dtype != np.dtype("<f8") or coordinates.ndim != 1:
                raise TypeError(
                    "curve fit coordinates must be a one-dimensional float64 array"
                )
            owned_coordinates = np.frombuffer(
                coordinates.tobytes(order="C"),
                dtype=np.dtype("<f8"),
            )
            owned_coordinates.setflags(write=False)
            object.__setattr__(self, "coordinates", owned_coordinates)
        if status is FitBatchStatus.CONVERGED:
            expected_size = (
                stop - start
                if owned_coordinates is None
                else owned_coordinates.size
            )
            if (
                expected_size == 0
                or expected_size > stop - start
                or owned.size != expected_size
                or not bool(np.all(np.isfinite(owned)))
                or (
                    owned_coordinates is not None
                    and not bool(np.all(np.isfinite(owned_coordinates)))
                )
            ):
                raise ValueError(
                    "converged curve fit primitive must align with its span and be finite"
                )
        elif owned.size or owned_coordinates is not None:
            raise ValueError(
                "failed or absent curve fit cells cannot retain a prediction primitive"
            )


@dataclass(frozen=True, slots=True, eq=False)
class HistogramFitOverlay:
    """One formal Fit prediction bound to one exact Histogram series."""

    source_ref: DatasetRevisionRef
    result_identity: str
    series_batch_address: tuple[AxisAddress, ...]
    batch_storage_index: int | None
    status: FitBatchStatus | None
    diagnostic: str
    coordinates: np.ndarray
    component_predictions: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, DatasetRevisionRef):
            raise TypeError("histogram fit overlay source_ref must be DatasetRevisionRef")
        object.__setattr__(
            self,
            "result_identity",
            _text(self.result_identity, "histogram fit result_identity"),
        )
        address = tuple(self.series_batch_address)
        if any(not isinstance(item, AxisAddress) for item in address):
            raise TypeError(
                "histogram fit series_batch_address requires AxisAddress values"
            )
        object.__setattr__(self, "series_batch_address", address)
        storage = self.batch_storage_index
        if storage is not None:
            storage = _nonnegative(storage, "histogram fit batch_storage_index")
            object.__setattr__(self, "batch_storage_index", storage)
        status = self.status
        if status is not None and not isinstance(status, FitBatchStatus):
            raise TypeError("histogram fit overlay status must be FitBatchStatus or None")
        if (storage is None) != (status is None):
            raise ValueError("histogram fit storage and status must be present together")
        if not isinstance(self.diagnostic, str):
            raise TypeError("histogram fit overlay diagnostic must be str")

        coordinates = np.asarray(self.coordinates, dtype=np.dtype("<f8"))
        components = tuple(
            np.asarray(values, dtype=np.dtype("<f8"))
            for values in self.component_predictions
        )
        if coordinates.ndim != 1 or any(values.shape != coordinates.shape for values in components):
            raise ValueError("histogram fit coordinates/components must be aligned vectors")
        if status is FitBatchStatus.CONVERGED:
            if not components or coordinates.size < 2:
                raise ValueError("converged histogram fit requires prediction components")
            if not bool(np.all(np.isfinite(coordinates))) or any(
                not bool(np.all(np.isfinite(values))) for values in components
            ):
                raise ValueError("converged histogram fit prediction must be finite")
        elif coordinates.size or components:
            raise ValueError("failed histogram fit cells cannot retain predictions")
        frozen_coordinates = np.frombuffer(
            coordinates.tobytes(order="C"),
            dtype=np.dtype("<f8"),
        )
        frozen_coordinates.setflags(write=False)
        frozen_components = []
        for values in components:
            owned = np.frombuffer(values.tobytes(order="C"), dtype=np.dtype("<f8"))
            owned.setflags(write=False)
            frozen_components.append(owned)
        object.__setattr__(self, "coordinates", frozen_coordinates)
        object.__setattr__(self, "component_predictions", tuple(frozen_components))


@dataclass(frozen=True, slots=True)
class ImagePanelRasterGeometry:
    """Exact top-origin axes boxes inside a worker-composed image panel raster."""

    image_bounds: tuple[float, float, float, float]
    distribution_bounds: tuple[float, float, float, float]
    colorbar_bounds: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        for name in (
            "image_bounds",
            "distribution_bounds",
            "colorbar_bounds",
        ):
            values = tuple(float(value) for value in getattr(self, name))
            if (
                len(values) != 4
                or any(not math.isfinite(value) for value in values)
                or not 0.0 <= values[0] < values[2] <= 1.0
                or not 0.0 <= values[1] < values[3] <= 1.0
            ):
                raise ValueError(
                    f"{name} must be a non-degenerate normalized rectangle"
                )
            object.__setattr__(self, name, values)


@dataclass(frozen=True, slots=True, eq=False)
class ImagePanelPayload:
    """Exact immutable samples and display mapping for one IMAGE raster front.

    ``data_range`` retains the full observed span for exact diagnostics and
    in-window guide lines even when the painted colour domain is narrower.
    ``colormap`` is the typed identity used by both Agg and transient Qt
    overlays; no second palette or quantized image plane crosses the boundary.
    """

    image: EvaluatedImage
    evaluated_input: EvaluatedInput
    viewport: ImageViewportTransform
    data_range: tuple[float, float] | None
    colormap: ImageColormap
    color_limits: tuple[float, float]
    raster_geometry: ImagePanelRasterGeometry
    fit_overlay: RadialGaussianImageFitOverlay | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.image, EvaluatedImage):
            raise TypeError("image payload requires EvaluatedImage")
        if not isinstance(self.evaluated_input, EvaluatedInput):
            raise TypeError("image payload requires one EvaluatedInput")
        if not isinstance(self.viewport, ImageViewportTransform):
            raise TypeError("image payload requires ImageViewportTransform")
        expected_shape = self.viewport.raster_shape
        if self.image.values.shape != expected_shape:
            raise ValueError("image payload values do not match viewport geometry")
        for evaluated, axis, name in (
            (self.image.x_axis, self.viewport.x_axis, "x"),
            (self.image.y_axis, self.viewport.y_axis, "y"),
        ):
            if evaluated.axis_id != axis.axis_id:
                raise ValueError(f"image payload {name} axis identity changed")
            if evaluated.role != axis.role:
                raise ValueError(f"image payload {name} axis role changed")
            if (
                len(evaluated.indices) != axis.size
                or len(set(evaluated.indices)) != axis.size
            ):
                raise ValueError(
                    f"image payload {name} axis does not cover the projected raster"
                )
            expected_coordinates = (
                axis.coordinates
                if axis.coordinates is not None
                else tuple(
                    axis.index_origin + index for index in range(axis.size)
                )
            )
            if evaluated.coordinates != expected_coordinates:
                raise ValueError(f"image payload {name} coordinates changed")

        data_range = self.data_range
        if data_range is not None:
            data_range = validated_display_range(
                data_range,
                "image data_range",
                allow_degenerate=True,
            )
        object.__setattr__(self, "data_range", data_range)
        object.__setattr__(
            self,
            "color_limits",
            validated_display_range(self.color_limits, "image color_limits"),
        )

        if not isinstance(self.colormap, ImageColormap):
            raise TypeError("image colormap must be ImageColormap")
        overlay = self.fit_overlay
        if overlay is not None:
            if not isinstance(overlay, RadialGaussianImageFitOverlay):
                raise TypeError(
                    "image fit_overlay must be RadialGaussianImageFitOverlay or None"
                )
            if overlay.source_ref != self.evaluated_input.ref:
                raise ValueError("image fit overlay belongs to another evaluated input")
            if overlay.coordinate_frame != self.viewport.coordinate_frame:
                raise ValueError("image fit overlay belongs to another coordinate frame")
        if not isinstance(
            self.raster_geometry,
            ImagePanelRasterGeometry,
        ):
            raise TypeError("image raster_geometry must be ImagePanelRasterGeometry")

    @property
    def value_unit(self) -> str | None:
        """Return the evaluator-owned image value unit without copying it."""

        return self.image.value_unit


@dataclass(frozen=True, slots=True)
class CurvePanelPayload:
    """Exact immutable samples and draw mapping for one CURVE raster front."""

    evaluated_input: EvaluatedInput
    viewport: CurveViewportTransform
    series: tuple[EvaluatedSeries, ...]
    series_labels: tuple[str, ...]
    fit_overlays: tuple[CurveFitOverlay, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.evaluated_input, EvaluatedInput):
            raise TypeError("curve payload requires one EvaluatedInput")
        if not isinstance(self.viewport, CurveViewportTransform):
            raise TypeError("curve payload requires CurveViewportTransform")
        series = tuple(self.series)
        if not series or any(not isinstance(item, EvaluatedSeries) for item in series):
            raise ValueError("curve payload requires EvaluatedSeries values")
        curves = tuple(item.data for item in series)
        if any(not isinstance(item, EvaluatedCurve) for item in curves):
            raise TypeError("curve payload series must all contain EvaluatedCurve")
        first_axis = curves[0].x_axis
        if any(curve.x_axis != first_axis for curve in curves[1:]):
            raise ValueError("curve payload series must share one exact x axis")
        if self.viewport.x_axis != first_axis:
            raise ValueError("curve payload viewport x axis differs from its series")
        if any(curve.value_unit != curves[0].value_unit for curve in curves[1:]):
            raise ValueError("curve payload series must share value_unit")
        labels = tuple(
            _text(label, f"curve series label {index}")
            for index, label in enumerate(self.series_labels)
        )
        if len(labels) != len(series):
            raise ValueError("curve series labels must align with series")
        overlays = _validated_curve_fit_overlays(
            self.evaluated_input,
            series,
            self.fit_overlays,
        )
        object.__setattr__(self, "series", series)
        object.__setattr__(self, "series_labels", labels)
        object.__setattr__(self, "fit_overlays", overlays)

    @property
    def value_unit(self) -> str | None:
        """Return the unit already owned by every validated curve series."""

        curve = self.series[0].data
        assert isinstance(curve, EvaluatedCurve)
        return curve.value_unit


def _validated_curve_fit_overlays(
    evaluated_input: EvaluatedInput,
    series: tuple[EvaluatedSeries, ...],
    overlays: tuple[CurveFitOverlay, ...],
) -> tuple[CurveFitOverlay, ...]:
    """Validate the exact series/result join used by payloads and Agg.

    An empty tuple means that no fit result is currently attached.  A concrete
    result is total over the displayed series: sparse and failed rows still
    have one DTO, so omitting one row can never shift another row into it.
    """

    if not isinstance(evaluated_input, EvaluatedInput):
        raise TypeError("curve overlay validation requires one EvaluatedInput")
    series = tuple(series)
    if not series or any(not isinstance(item, EvaluatedSeries) for item in series):
        raise ValueError("curve overlay validation requires EvaluatedSeries values")
    overlays = tuple(overlays)
    if not overlays:
        return ()
    if len(overlays) != len(series) or any(
        not isinstance(item, CurveFitOverlay) for item in overlays
    ):
        raise ValueError("curve fit overlays must align one-for-one with series")
    identity = overlays[0].result_identity
    source_ref = overlays[0].source_ref
    source_span = overlays[0].source_sample_span
    stored_indices = []
    for position, (overlay, item) in enumerate(zip(overlays, series, strict=True)):
        curve = item.data
        if not isinstance(curve, EvaluatedCurve):
            raise TypeError("curve fit overlays require EvaluatedCurve series")
        if overlay.source_ref != evaluated_input.ref or overlay.source_ref != source_ref:
            raise ValueError("curve fit overlay belongs to another source revision")
        if overlay.result_identity != identity:
            raise ValueError("curve payload cannot mix fit result identities")
        if overlay.source_sample_span != source_span:
            raise ValueError("curve payload cannot mix fit authority spans")
        if overlay.series_batch_address != item.batch_address:
            raise ValueError(
                f"curve fit overlay {position} belongs to another series address"
            )
        start, stop = overlay.source_sample_span
        if stop > curve.values.size:
            raise ValueError("curve fit source span exceeds source samples")
        if overlay.status is FitBatchStatus.CONVERGED:
            expected_shape = (
                (stop - start,)
                if overlay.coordinates is None
                else overlay.coordinates.shape
            )
            if (
                overlay.predicted_y.shape != expected_shape
                or expected_shape[0] > stop - start
            ):
                raise ValueError("curve fit prediction does not align with source span")
        elif overlay.predicted_y.size:
            # The DTO already closes this invariant; keep it explicit at the
            # join boundary so a future alternate DTO cannot weaken it.
            raise ValueError("failed curve fit overlay retained stale prediction data")
        if overlay.batch_storage_index is not None:
            stored_indices.append(overlay.batch_storage_index)
    if len(stored_indices) != len(set(stored_indices)):
        raise ValueError("two displayed curve series map to one fit storage row")
    return overlays


@dataclass(frozen=True, slots=True)
class PulsePanelPayload:
    """Draw-frozen mapping for one PULSE timeline raster front.

    The pulse preview is an x-only interactive surface: gestures select or zoom
    along TIME while the row axis stays pinned, so the payload reuses the curve
    family's viewport transform for its widget<->data mapping and adds only the
    drawn row identities (digital channels then analog bus keys, top to bottom)
    with their display labels for the authored row chrome.
    """

    document_input: DocumentInputIdentity
    viewport: NumericViewportTransform
    row_keys: tuple[str, ...]
    row_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.document_input, DocumentInputIdentity):
            raise TypeError("pulse payload requires one DocumentInputIdentity")
        if type(self.viewport) is not NumericViewportTransform:
            raise TypeError("pulse payload requires NumericViewportTransform")
        keys = tuple(_text(key, "pulse row key") for key in self.row_keys)
        labels = tuple(
            _text(label, "pulse row label") for label in self.row_labels)
        if len(keys) != len(labels):
            raise ValueError("pulse row keys and labels must align")
        object.__setattr__(self, "row_keys", keys)
        object.__setattr__(self, "row_labels", labels)


@dataclass(frozen=True, slots=True)
class HistogramPanelPayload:
    """Exact samples plus one shared, immutable display bin projection.

    Ordinary binning is presentation-only.  ``series`` retains the evaluator's
    exact samples, sample-axis coordinates, component-validity filtering and
    dropped counts.  Only an explicit Fit command may promote the named SAMPLE
    axes plus these exact edges into a terminal authoritative ``HistogramSpec``.
    """

    evaluated_input: EvaluatedInput
    viewport: HistogramViewportTransform
    series: tuple[EvaluatedSeries, ...]
    series_labels: tuple[str, ...]
    bin_projection: HistogramBinProjection
    # The drawn threshold cut lines (display state echoed by the renderer so
    # the board can grab them in the value coordinate).
    thresholds: tuple[float, ...] = ()
    fit_overlays: tuple[HistogramFitOverlay, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "thresholds",
            tuple(float(value) for value in self.thresholds),
        )
        if any(not math.isfinite(value) for value in self.thresholds):
            raise ValueError("histogram payload thresholds must be finite")
        fit_overlays = tuple(self.fit_overlays)
        if any(not isinstance(item, HistogramFitOverlay) for item in fit_overlays):
            raise TypeError("histogram payload fit_overlays require HistogramFitOverlay values")
        object.__setattr__(self, "fit_overlays", fit_overlays)
        if not isinstance(self.evaluated_input, EvaluatedInput):
            raise TypeError("histogram payload requires one EvaluatedInput")
        if not isinstance(self.viewport, HistogramViewportTransform):
            raise TypeError(
                "histogram payload requires HistogramViewportTransform"
            )
        series = tuple(self.series)
        if not series or any(not isinstance(item, EvaluatedSeries) for item in series):
            raise ValueError("histogram payload requires EvaluatedSeries values")
        histograms = tuple(item.data for item in series)
        if any(not isinstance(item, EvaluatedHistogram) for item in histograms):
            raise TypeError(
                "histogram payload series must all contain EvaluatedHistogram"
            )
        value_unit = histograms[0].value_unit
        if any(item.value_unit != value_unit for item in histograms[1:]):
            raise ValueError("histogram payload series must share value_unit")
        projection = self.bin_projection
        if not isinstance(projection, HistogramBinProjection):
            raise TypeError(
                "histogram payload requires one computed HistogramBinProjection"
            )
        if len(projection.series_samples) != len(histograms) or any(
            projected is not histogram.samples
            for projected, histogram in zip(
                projection.series_samples,
                histograms,
                strict=True,
            )
        ):
            raise ValueError(
                "histogram display projection is not bound to these exact samples"
            )
        if projection.requested_bin_count != self.viewport.bin_count:
            raise ValueError(
                "histogram projection bin count differs from its authored viewport"
            )

        labels = tuple(
            _text(label, f"histogram series label {index}")
            for index, label in enumerate(self.series_labels)
        )
        if len(labels) != len(series):
            raise ValueError("histogram series labels must align with series")

        edge_source = np.asarray(projection.bin_edges)
        if edge_source.ndim != 1 or edge_source.size < 2:
            raise ValueError("histogram bin_edges must contain at least two edges")
        if edge_source.dtype.kind not in "biuf" or not bool(
            np.all(np.isfinite(edge_source))
        ):
            raise ValueError("histogram bin_edges must be finite real values")
        edges = projection.bin_edges
        if edges.dtype != np.dtype("<f8") or edges.flags.writeable:
            raise RuntimeError("histogram projection edges lost immutable <f8 form")
        if not bool(np.all(np.diff(edges) > 0.0)):
            raise ValueError("histogram bin_edges must be strictly increasing")

        for index, (value, histogram) in enumerate(
            zip(projection.bin_counts, histograms, strict=True)
        ):
            source = np.asarray(value)
            if source.ndim != 1 or source.shape != (len(edges) - 1,):
                raise ValueError(
                    f"histogram bin_counts[{index}] does not align with edges"
                )
            if source.dtype.kind not in "iu" or bool(np.any(source < 0)):
                raise ValueError("histogram bin counts must be nonnegative integers")
            counts = value
            if counts.dtype != np.dtype("<i8") or counts.flags.writeable:
                raise RuntimeError(
                    "histogram projection counts lost immutable <i8 form"
                )
            if int(np.sum(counts, dtype=np.int64)) != len(histogram.samples):
                raise ValueError(
                    "histogram display bins silently lost or invented valid samples"
                )
        if len(projection.bin_counts) != len(series):
            raise ValueError("histogram bin_counts must align with series")
        if fit_overlays:
            if len(fit_overlays) != len(series):
                raise ValueError("histogram Fit overlays must align with series")
            if len({overlay.result_identity for overlay in fit_overlays}) != 1:
                raise ValueError("histogram payload cannot mix Fit result identities")
            if any(
                overlay.source_ref != self.evaluated_input.ref
                or overlay.series_batch_address != item.batch_address
                for overlay, item in zip(fit_overlays, series, strict=True)
            ):
                raise ValueError("histogram Fit overlay belongs to another exact series")
            stored = tuple(
                overlay.batch_storage_index
                for overlay in fit_overlays
                if overlay.batch_storage_index is not None
            )
            if len(stored) != len(set(stored)):
                raise ValueError("two histogram series map to one Fit storage row")
        if self.viewport.home_x_limits != (float(edges[0]), float(edges[-1])):
            raise ValueError(
                "histogram viewport home range differs from shared bin edges"
        )
        object.__setattr__(self, "series", series)
        object.__setattr__(self, "series_labels", labels)

    @property
    def value_unit(self) -> str | None:
        histogram = self.series[0].data
        assert isinstance(histogram, EvaluatedHistogram)
        return histogram.value_unit

    @property
    def bin_counts(self) -> tuple[np.ndarray, ...]:
        return self.bin_projection.bin_counts

    @property
    def bin_edges(self) -> np.ndarray:
        return self.bin_projection.bin_edges


@dataclass(frozen=True, slots=True)
class MeterPanelPayload:
    """Exact scalar and source identity paired with one METER raster front."""

    evaluated_input: EvaluatedInput
    display_revision: int
    series: tuple[EvaluatedSeries, ...]
    series_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.evaluated_input, EvaluatedInput):
            raise TypeError("meter payload requires one EvaluatedInput")
        object.__setattr__(
            self,
            "display_revision",
            _nonnegative(self.display_revision, "meter display revision"),
        )
        series = tuple(self.series)
        if not series or any(not isinstance(item, EvaluatedSeries) for item in series):
            raise ValueError("meter payload requires EvaluatedSeries values")
        meters = tuple(item.data for item in series)
        if any(not isinstance(item, EvaluatedMeter) for item in meters):
            raise TypeError("meter payload series must all contain EvaluatedMeter")
        value_unit = meters[0].value_unit
        if any(item.value_unit != value_unit for item in meters[1:]):
            raise ValueError("meter payload series must share value_unit")
        labels = tuple(
            _text(label, f"meter series label {index}")
            for index, label in enumerate(self.series_labels)
        )
        if len(labels) != len(series):
            raise ValueError("meter series labels must align with its exact series")
        object.__setattr__(self, "series", series)
        object.__setattr__(self, "series_labels", labels)

    @property
    def value_unit(self) -> str | None:
        meter = self.series[0].data
        assert isinstance(meter, EvaluatedMeter)
        return meter.value_unit

@dataclass(frozen=True, slots=True, eq=False)
class SiteMapPanelPayload:
    """One IMAGE background plus declared sites and optional boolean state.

    Site coordinates carry an explicit frame and are never inferred from array
    shape.  Site state and background remain distinct evaluated inputs so a
    :class:`CoherenceStamp` can prove their exact joined revisions.
    """

    background: ImagePanelPayload
    site_state_input: EvaluatedInput
    site_axis: AxisSpec
    coordinate_frame: CoordinateFrameId
    centers_xy: np.ndarray
    site_state: np.ndarray | None
    site_validity: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.background, ImagePanelPayload):
            raise TypeError("site-map background must be ImagePanelPayload")
        if not isinstance(self.site_state_input, EvaluatedInput):
            raise TypeError("site-map site_state_input must be EvaluatedInput")
        if not isinstance(self.site_axis, AxisSpec) or self.site_axis.role != SITE:
            raise ValueError("site-map site_axis must be an AxisSpec with role SITE")
        if not isinstance(self.coordinate_frame, CoordinateFrameId):
            raise TypeError("site-map coordinate_frame must be CoordinateFrameId")
        if self.coordinate_frame != self.background.viewport.coordinate_frame:
            raise ValueError(
                "site-map coordinate_frame differs from its background viewport"
            )
        if (
            self.site_state_input.dataset_id
            == self.background.evaluated_input.dataset_id
        ):
            raise ValueError(
                "site-map state and background require distinct dataset ids"
            )

        site_count = self.site_axis.size
        state_present = self.site_state is not None
        centers, site_state, validity = immutable_site_state(
            self.centers_xy,
            (
                self.site_state
                if state_present
                else np.zeros(site_count, dtype=np.bool_)
            ),
            self.site_validity,
            site_count=site_count,
        )
        try:
            self.background.viewport.full_points_for_coordinates(
                centers,
                coordinate_frame=self.coordinate_frame,
            )
        except ValueError as exc:
            raise ValueError(
                "site-map geometry cannot be painted on its background viewport"
            ) from exc
        object.__setattr__(self, "centers_xy", centers)
        object.__setattr__(
            self,
            "site_state",
            site_state if state_present else None,
        )
        object.__setattr__(self, "site_validity", validity)


DisplayPayload = (
    ImagePanelPayload
    | CurvePanelPayload
    | HistogramPanelPayload
    | MeterPanelPayload
    | PulsePanelPayload
    | SiteMapPanelPayload
)


@dataclass(frozen=True)
class PanelFrame:
    panel_id: str
    coherence_group: str
    source_identity: SourceIdentity | DocumentInputIdentity
    coherence_stamp: CoherenceStamp | None
    raster: RasterBuffer
    display_payload: DisplayPayload | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", _text(self.panel_id, "panel_id"))
        object.__setattr__(
            self,
            "coherence_group",
            _text(self.coherence_group, "coherence_group"),
        )
        source_is_document = isinstance(
            self.source_identity, DocumentInputIdentity
        )
        if not isinstance(
            self.source_identity, (SourceIdentity, DocumentInputIdentity)
        ):
            raise TypeError(
                "source_identity must be SourceIdentity or DocumentInputIdentity"
            )
        if not isinstance(self.raster, RasterBuffer):
            raise TypeError("raster must be RasterBuffer")
        payload = self.display_payload
        if source_is_document:
            if self.coherence_stamp is not None:
                raise TypeError(
                    "document-backed panels do not carry a dataset coherence stamp"
                )
            if not isinstance(payload, PulsePanelPayload):
                raise TypeError(
                    "document-backed panels require PulsePanelPayload"
                )
            if payload.document_input != self.source_identity:
                raise ValueError(
                    "pulse payload differs from its document source identity"
                )
            return
        if not isinstance(self.coherence_stamp, CoherenceStamp):
            raise TypeError("dataset-backed panels require CoherenceStamp")
        if isinstance(payload, PulsePanelPayload):
            raise TypeError(
                "PulsePanelPayload requires a document source identity"
            )
        if payload is not None:
            if not isinstance(
                payload,
                (
                    ImagePanelPayload,
                    CurvePanelPayload,
                    HistogramPanelPayload,
                    MeterPanelPayload,
                    PulsePanelPayload,
                    SiteMapPanelPayload,
                ),
            ):
                raise TypeError(
                    "display_payload must be ImagePanelPayload, "
                    "CurvePanelPayload, HistogramPanelPayload, MeterPanelPayload, "
                    "PulsePanelPayload, SiteMapPanelPayload, or None"
                )
            if isinstance(payload, ImagePanelPayload):
                source_input = payload.evaluated_input
            elif isinstance(payload, CurvePanelPayload):
                source_input = payload.evaluated_input
            elif isinstance(payload, HistogramPanelPayload):
                source_input = payload.evaluated_input
            elif isinstance(payload, MeterPanelPayload):
                source_input = payload.evaluated_input
            else:
                background = payload.background
                source_input = payload.site_state_input
                try:
                    background_ref = next(
                        value.ref
                        for value in self.coherence_stamp.inputs
                        if value.dataset_id
                        == background.evaluated_input.dataset_id
                    )
                except StopIteration as exc:
                    raise ValueError(
                        "site-map background is absent from its coherence stamp"
                    ) from exc
                if background.evaluated_input.ref != background_ref:
                    raise ValueError(
                        "site-map background differs from its frozen coherence input"
                    )
            try:
                expected_ref = next(
                    value.ref
                    for value in self.coherence_stamp.inputs
                    if value.dataset_id == self.source_identity.dataset_id
                )
            except StopIteration as exc:
                raise ValueError(
                    "display payload source is absent from its coherence stamp"
                ) from exc
            if (
                source_input.dataset_id != self.source_identity.dataset_id
                or source_input.ref != expected_ref
            ):
                raise ValueError(
                    "display payload input differs from its frozen coherence input"
                )
            if isinstance(payload, SiteMapPanelPayload) and (
                self.source_identity.block_id != source_input.ref.block_id
                or self.source_identity.stream_generation
                != source_input.ref.stream_generation
                or self.source_identity.schema_fingerprint
                != source_input.ref.schema_fingerprint
            ):
                raise ValueError(
                    "site-map source identity differs from its site-state input"
                )


@dataclass(frozen=True)
class BoardFrame:
    """One atomic presentation transaction for a complete board layout.

    Coherence is scoped to each explicit ``coherence_group``.  Different
    groups may intentionally carry independent producer revisions; merely
    presenting them in one board transaction never means "the same shot".
    """

    board_id: str
    layout_generation: int
    sequence: int
    panels: tuple[PanelFrame, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "board_id", _text(self.board_id, "board_id"))
        object.__setattr__(
            self,
            "layout_generation",
            _nonnegative(self.layout_generation, "layout_generation"),
        )
        object.__setattr__(self, "sequence", _nonnegative(self.sequence, "sequence"))
        panels = tuple(self.panels)
        if not panels:
            raise ValueError("BoardFrame must contain at least one panel")
        if any(not isinstance(panel, PanelFrame) for panel in panels):
            raise TypeError("panels must contain PanelFrame values")
        ids = tuple(panel.panel_id for panel in panels)
        if len(set(ids)) != len(ids):
            raise ValueError("BoardFrame panel ids must be unique")
        panels_by_group: dict[str, list[PanelFrame]] = {}
        for panel in panels:
            panels_by_group.setdefault(panel.coherence_group, []).append(panel)
        for group_panels in panels_by_group.values():
            first = group_panels[0]
            if isinstance(first.source_identity, DocumentInputIdentity):
                for panel in group_panels:
                    if (
                        not isinstance(panel.source_identity, DocumentInputIdentity)
                        or panel.source_identity != first.source_identity
                        or panel.coherence_stamp is not None
                        or not isinstance(panel.display_payload, PulsePanelPayload)
                    ):
                        raise ValueError(
                            "document presentation group mixes dataset and document "
                            "panel families"
                    )
                continue
            stamp = first.coherence_stamp
            if not isinstance(stamp, CoherenceStamp):
                raise TypeError("dataset coherence group requires CoherenceStamp")
            if any(
                panel.coherence_stamp != stamp
                for panel in group_panels[1:]
            ):
                raise ValueError(
                    "panels in one coherence group must carry one exact CoherenceStamp"
                )
            inputs = {value.dataset_id: value.ref for value in stamp.inputs}
            for panel in group_panels:
                if not isinstance(panel.source_identity, SourceIdentity):
                    raise ValueError(
                        "dataset coherence group contains a document source"
                    )
                try:
                    source_ref = inputs[panel.source_identity.dataset_id]
                except KeyError as exc:
                    raise ValueError(
                        "CoherenceStamp does not freeze a panel source input"
                    ) from exc
                if (
                    source_ref.block_id != panel.source_identity.block_id
                    or source_ref.stream_generation
                    != panel.source_identity.stream_generation
                    or source_ref.schema_fingerprint
                    != panel.source_identity.schema_fingerprint
                ):
                    raise ValueError(
                        "panel source identity differs from its frozen input revision"
                    )
        object.__setattr__(self, "panels", panels)


__all__ = [
    "BoardFrame",
    "CoherenceStamp",
    "CurveFitOverlay",
    "CurvePanelPayload",
    "DocumentInputIdentity",
    "HistogramPanelPayload",
    "HistogramFitOverlay",
    "MeterPanelPayload",
    "detached_render_fault",
    "DisplayPayload",
    "SourceIdentity",
    "PanelFrame",
    "ImagePanelPayload",
    "ImagePanelRasterGeometry",
    "RadialGaussianImageFitOverlay",
    "SiteMapPanelPayload",
    "PulsePanelPayload",
    "RasterBuffer",
]
