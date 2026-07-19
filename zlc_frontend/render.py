"""Headless render hand-off values owned by the target frontend package.

The renderer may use Matplotlib, Qt, or neither, but the worker/GUI boundary is
always an immutable :class:`BoardFrame`.  No live Figure, Artist, mutable or
aliased ndarray view, or QImage storage crosses this module's boundary.  The
exact image, curve, and calibrated site-map interaction payloads are allowed
because their evaluated arrays are intrinsically backed by owned immutable
bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
import threading
from typing import Protocol, runtime_checkable

import numpy as np

from zlc_data import (
    AxisSpec,
    BlockId,
    CoordinateFrameId,
    DatasetRevisionRef,
    FitBatchStatus,
    SITE,
    StreamGenerationId,
    axis_to_tree,
    dataset_revision_ref_to_tree,
)
from zlc_storage import (
    canonical_digest,
    canonical_text as _text,
    nonnegative_integer as _nonnegative,
    sha256_digest,
    sha256_text,
)

from .curve_display import CurveViewportTransform
from .display_range import validated_display_range
from .histogram_display import HistogramBinProjection, HistogramViewportTransform
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
from .site_map import immutable_site_state, site_ring_radius


SITE_MAP_JOIN_SCHEMA_DIGEST = canonical_digest(
    {
        "schema": "zlc_frontend.SiteMapJoinSchema",
        "value_schema": "zlc_frontend.SiteMapJoin",
        "fields": (
            "cell_identity",
            "occupancy.dataset_id",
            "occupancy.ref[zlc_data.DatasetRevisionRef]",
            "background.dataset_id",
            "background.ref[zlc_data.DatasetRevisionRef]",
            "geometry_identity",
        ),
    }
)


def detached_render_fault(error: BaseException) -> RuntimeError:
    """Return string-only diagnostics without retaining a failed render stack."""

    return RuntimeError(f"{type(error).__name__}: {error}")


class RenderSurface(Enum):
    """The three deliberately supported rendering ownership modes."""

    GUI_ARTIST = "gui-artist"
    WORKER_RASTER_LIVE = "worker-raster-live"
    WORKER_HEADLESS_EXPORT = "worker-headless-export"


class PixelFormat(Enum):
    """Canonical owned raster layouts accepted at the presentation boundary."""

    RGBA8888 = "rgba8888"
    RGB888 = "rgb888"
    GRAY8 = "gray8"
    INDEXED8 = "indexed8"

    @property
    def channels(self) -> int:
        return {
            PixelFormat.RGBA8888: 4,
            PixelFormat.RGB888: 3,
            PixelFormat.GRAY8: 1,
            PixelFormat.INDEXED8: 1,
        }[self]


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


@dataclass(frozen=True)
class CoherenceStamp:
    """Frozen evaluation identity shared by panels from one causation point.

    The evaluator mints this value only after freezing the typed join key, every
    input dataset revision, and the presentation intent.  Equality therefore
    means more than matching two bare counters from unrelated producers.
    """

    run_id: str
    provenance_epoch_id: str
    join_key_type: str
    join_key_schema_fingerprint: str
    join_key_digest: str
    inputs: tuple[EvaluatedInput, ...]
    presentations: tuple["PanelPresentationIdentity", ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "provenance_epoch_id",
            _text(self.provenance_epoch_id, "provenance_epoch_id"),
        )
        object.__setattr__(
            self,
            "join_key_type",
            _text(self.join_key_type, "join_key_type"),
        )
        object.__setattr__(
            self,
            "join_key_schema_fingerprint",
            sha256_text(
                self.join_key_schema_fingerprint,
                "join_key_schema_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "join_key_digest",
            sha256_text(self.join_key_digest, "join_key_digest"),
        )
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
        presentations = tuple(self.presentations)
        if not presentations or any(
            not isinstance(value, PanelPresentationIdentity)
            for value in presentations
        ):
            raise ValueError(
                "presentations must contain at least one PanelPresentationIdentity"
            )
        panel_ids = tuple(value.panel_id for value in presentations)
        if len(set(panel_ids)) != len(panel_ids):
            raise ValueError("presentation panel ids must be unique")
        object.__setattr__(
            self,
            "presentations",
            tuple(sorted(presentations, key=lambda value: value.panel_id)),
        )


@dataclass(frozen=True)
class PanelPresentationIdentity:
    """Exact per-panel view intent frozen before raster work is admitted."""

    panel_id: str
    document_id: str
    document_revision: int
    selection_revision: int
    panel_revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", _text(self.panel_id, "panel_id"))
        object.__setattr__(
            self,
            "document_id",
            _text(self.document_id, "document_id"),
        )
        for field in (
            "document_revision",
            "selection_revision",
            "panel_revision",
        ):
            object.__setattr__(
                self,
                field,
                _nonnegative(getattr(self, field), field),
            )


@dataclass(frozen=True)
class RasterBuffer:
    """An owned immutable raster; ``pixels`` can never alias a worker buffer."""

    width: int
    height: int
    stride_bytes: int
    pixel_format: PixelFormat
    pixels: bytes

    def __post_init__(self) -> None:
        width = _nonnegative(self.width, "width")
        height = _nonnegative(self.height, "height")
        stride = _nonnegative(self.stride_bytes, "stride_bytes")
        if width == 0 or height == 0:
            raise ValueError("raster width and height must be positive")
        if not isinstance(self.pixel_format, PixelFormat):
            raise TypeError("pixel_format must be PixelFormat")
        minimum_stride = width * self.pixel_format.channels
        if stride < minimum_stride:
            raise ValueError("stride_bytes is too small for width and pixel format")
        if not isinstance(self.pixels, bytes):
            raise TypeError("pixels must be owned immutable bytes")
        if len(self.pixels) != stride * height:
            raise ValueError("pixels length must equal stride_bytes * height")


@dataclass(frozen=True, slots=True)
class RadialGaussianImageFitOverlay:
    """Exact saved-fit annotation for one radial-Gaussian IMAGE cell.

    The fit result remains owned by the analysis/artifact layer.  This small
    presentation value carries only the already-published centre/radius facts,
    their exact source/artifact identity, and a bounded status diagnostic.  It
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
        if len(self.artifact_identity) > 4096:
            raise ValueError("fit overlay artifact_identity exceeds display bound")
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
        if len(self.caption) > 8192:
            raise ValueError("fit overlay caption exceeds display bound")
        if not isinstance(self.diagnostic, str):
            raise TypeError("fit overlay diagnostic must be str")
        if len(self.diagnostic) > 1024:
            raise ValueError("fit overlay diagnostic exceeds display bound")
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
    the unchanged cached curve; ``predicted_y`` exists only for a converged
    stored row and only over that span.  Failed and sparse ``NOT_PRESENT`` rows
    own the canonical empty array, which makes a newer failed result clear any
    previously painted successful prediction.
    """

    source_ref: DatasetRevisionRef
    result_identity: str
    series_batch_address: tuple[AxisAddress, ...]
    source_sample_span: tuple[int, int]
    batch_storage_index: int | None
    status: FitBatchStatus | None
    diagnostic: str
    predicted_y: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, DatasetRevisionRef):
            raise TypeError("curve fit overlay source_ref must be DatasetRevisionRef")
        identity = _text(self.result_identity, "curve fit result_identity")
        if len(identity) > 4096:
            raise ValueError("curve fit result_identity exceeds its display bound")
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
        if len(self.diagnostic) > 1024:
            raise ValueError("curve fit overlay diagnostic exceeds its display bound")

        predicted = np.asarray(self.predicted_y)
        if predicted.dtype != np.dtype("<f8") or predicted.ndim != 1:
            raise TypeError("curve fit predicted_y must be a one-dimensional float64 array")
        # Immutable bytes are the ownership boundary.  Merely setting
        # writeable=False on a caller-owned ndarray would still permit its base
        # buffer to be changed behind this front.
        owned = np.frombuffer(predicted.tobytes(order="C"), dtype=np.dtype("<f8"))
        owned.setflags(write=False)
        object.__setattr__(self, "predicted_y", owned)
        if status is FitBatchStatus.CONVERGED:
            if owned.size != stop - start or not bool(np.all(np.isfinite(owned))):
                raise ValueError(
                    "converged curve fit prediction must align with its span and be finite"
                )
        elif owned.size:
            raise ValueError(
                "failed or absent curve fit cells cannot retain a prediction"
            )


@dataclass(frozen=True, slots=True, eq=False)
class ImagePanelPayload:
    """Exact immutable samples and display mapping for one IMAGE raster front.

    Codes 1..255 always span ``color_limits``.  ``data_range`` retains the
    full observed span for exact diagnostics and in-window guide lines even
    when the painted/interactive colour domain is much narrower.
    """

    image: EvaluatedImage
    evaluated_input: EvaluatedInput
    viewport: ImageViewportTransform
    data_range: tuple[float, float] | None
    histogram_counts: tuple[int, ...]
    base_palette: tuple[int, ...]
    color_limits: tuple[float, float]
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
            if len(evaluated.coordinates) != axis.size or any(
                actual != axis.coordinate_at(index)
                for index, actual in enumerate(evaluated.coordinates)
            ):
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

        counts = tuple(self.histogram_counts)
        if len(counts) != 255:
            raise ValueError("image histogram_counts must contain 255 scalar codes")
        if any(
            isinstance(count, bool)
            or not isinstance(count, Integral)
            or int(count) < 0
            for count in counts
        ):
            raise ValueError("image histogram counts must be nonnegative integers")
        counts = tuple(int(count) for count in counts)
        valid_count = sum(counts)
        if valid_count > self.image.values.size:
            raise ValueError("image histogram count exceeds raster cardinality")
        if data_range is None and valid_count:
            raise ValueError("image histogram cannot contain samples without data_range")
        object.__setattr__(self, "histogram_counts", counts)

        palette = tuple(self.base_palette)
        if len(palette) != 256 or any(
            isinstance(color, bool)
            or not isinstance(color, Integral)
            or not 0 <= int(color) <= 0xFFFFFFFF
            for color in palette
        ):
            raise ValueError("image base_palette must contain 256 unsigned ARGB32 values")
        object.__setattr__(self, "base_palette", tuple(int(color) for color in palette))
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
            if overlay.predicted_y.shape != (stop - start,):
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
class HistogramPanelPayload:
    """Exact samples plus one shared, immutable display bin projection.

    Binning is presentation-only.  ``series`` retains the evaluator's exact
    samples, sample-axis coordinates, component-validity filtering and dropped
    counts; the shared counts/edges never become a fit or threshold authority.
    """

    evaluated_input: EvaluatedInput
    viewport: HistogramViewportTransform
    series: tuple[EvaluatedSeries, ...]
    series_labels: tuple[str, ...]
    bin_projection: HistogramBinProjection

    def __post_init__(self) -> None:
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
    """One IMAGE background plus calibrated, exact per-site occupancy state.

    Site coordinates carry an explicit frame and are never inferred from array
    shape.  Occupancy and background remain distinct evaluated inputs so a
    :class:`CoherenceStamp` can prove their exact joined revisions.
    """

    background: ImagePanelPayload
    occupancy_input: EvaluatedInput
    site_axis: AxisSpec
    coordinate_frame: CoordinateFrameId
    centers_xy: np.ndarray
    occupied: np.ndarray
    site_validity: np.ndarray
    calibration_identity: str
    cell_identity: str
    _full_normalized_centers_xy: np.ndarray = field(init=False, repr=False)
    _visible_ring_span: tuple[float, float] = field(init=False, repr=False)
    _geometry_identity: str = field(init=False, repr=False)
    _join_key_digest: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.background, ImagePanelPayload):
            raise TypeError("site-map background must be ImagePanelPayload")
        if not isinstance(self.occupancy_input, EvaluatedInput):
            raise TypeError("site-map occupancy_input must be EvaluatedInput")
        if not isinstance(self.site_axis, AxisSpec) or self.site_axis.role != SITE:
            raise ValueError("site-map site_axis must be an AxisSpec with role SITE")
        if not isinstance(self.coordinate_frame, CoordinateFrameId):
            raise TypeError("site-map coordinate_frame must be CoordinateFrameId")
        if self.coordinate_frame != self.background.viewport.coordinate_frame:
            raise ValueError(
                "site-map coordinate_frame differs from its background viewport"
            )
        if (
            self.occupancy_input.dataset_id
            == self.background.evaluated_input.dataset_id
        ):
            raise ValueError(
                "site-map occupancy and background require distinct dataset ids"
            )

        site_count = self.site_axis.size
        centers, occupied, validity = immutable_site_state(
            self.centers_xy,
            self.occupied,
            self.site_validity,
            site_count=site_count,
        )
        radius = site_ring_radius(centers)
        viewport = self.background.viewport
        try:
            normalized_centers = viewport.full_points_for_coordinates(
                centers,
                coordinate_frame=self.coordinate_frame,
            )
            visible_ring_span = viewport.visible_span_for_coordinate_span(
                (2.0 * radius, 2.0 * radius),
                coordinate_frame=self.coordinate_frame,
            )
        except ValueError as exc:
            raise ValueError(
                "site-map geometry cannot be painted on its background viewport"
            ) from exc
        object.__setattr__(self, "centers_xy", centers)
        object.__setattr__(
            self,
            "_full_normalized_centers_xy",
            normalized_centers,
        )
        object.__setattr__(self, "_visible_ring_span", visible_ring_span)
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "site_validity", validity)
        calibration_identity = _text(
            self.calibration_identity,
            "site-map calibration_identity",
        )
        cell_identity = _text(self.cell_identity, "site-map cell_identity")
        object.__setattr__(self, "calibration_identity", calibration_identity)
        object.__setattr__(self, "cell_identity", cell_identity)
        axis = self.site_axis
        geometry_identity = canonical_digest(
            {
                "schema": "zlc_frontend.SiteMapGeometry",
                "calibration_identity": calibration_identity,
                "coordinate_frame": self.coordinate_frame.value,
                "site_axis": axis_to_tree(axis),
                "centers_sha256": sha256_digest(memoryview(centers).cast("B")),
                "ring_radius": radius,
            }
        )

        input_trees = {}
        for name, value in (
            ("occupancy", self.occupancy_input),
            ("background", self.background.evaluated_input),
        ):
            ref = value.ref
            input_trees[name] = {
                "dataset_id": value.dataset_id.value,
                "ref": dataset_revision_ref_to_tree(ref),
            }

        object.__setattr__(self, "_geometry_identity", geometry_identity)
        object.__setattr__(
            self,
            "_join_key_digest",
            canonical_digest(
                {
                    "schema": "zlc_frontend.SiteMapJoin",
                    "cell_identity": cell_identity,
                    "occupancy": input_trees["occupancy"],
                    "background": input_trees["background"],
                    "geometry_identity": geometry_identity,
                }
            ),
        )

    @property
    def full_normalized_centers_xy(self) -> np.ndarray:
        """Return immutable full-raster points used by the Qt paint hot path."""

        return self._full_normalized_centers_xy

    @property
    def visible_ring_span(self) -> tuple[float, float]:
        """Return the current viewport-normalized ring width and height."""

        return self._visible_ring_span

    @property
    def visible_coordinate_aspect_ratio(self) -> float:
        """Return target raster width/height for isotropic physical coordinates."""

        width, height = self._visible_ring_span
        return height / width

    @property
    def geometry_identity(self) -> str:
        """Digest calibration-owned geometry used for hold/topology checks."""

        return self._geometry_identity

    @property
    def join_key_digest(self) -> str:
        """Digest the exact cell, both data revisions, and calibration geometry."""

        return self._join_key_digest


DisplayPayload = (
    ImagePanelPayload
    | CurvePanelPayload
    | HistogramPanelPayload
    | MeterPanelPayload
    | SiteMapPanelPayload
)


@dataclass(frozen=True)
class PanelFrame:
    panel_id: str
    coherence_group: str
    source_identity: SourceIdentity
    coherence_stamp: CoherenceStamp
    raster: RasterBuffer
    display_payload: DisplayPayload | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "panel_id", _text(self.panel_id, "panel_id"))
        object.__setattr__(
            self,
            "coherence_group",
            _text(self.coherence_group, "coherence_group"),
        )
        if not isinstance(self.source_identity, SourceIdentity):
            raise TypeError("source_identity must be SourceIdentity")
        if not isinstance(self.coherence_stamp, CoherenceStamp):
            raise TypeError("coherence_stamp must be CoherenceStamp")
        if not isinstance(self.raster, RasterBuffer):
            raise TypeError("raster must be RasterBuffer")
        payload = self.display_payload
        if payload is not None:
            if not isinstance(
                payload,
                (
                    ImagePanelPayload,
                    CurvePanelPayload,
                    HistogramPanelPayload,
                    MeterPanelPayload,
                    SiteMapPanelPayload,
                ),
            ):
                raise TypeError(
                    "display_payload must be ImagePanelPayload, "
                    "CurvePanelPayload, HistogramPanelPayload, MeterPanelPayload, "
                    "SiteMapPanelPayload, or None"
                )
            presentations = tuple(
                presentation
                for presentation in self.coherence_stamp.presentations
                if presentation.panel_id == self.panel_id
            )
            if len(presentations) != 1:
                raise ValueError("payload panel has no unique presentation identity")
            if isinstance(payload, ImagePanelPayload):
                if self.raster.pixel_format is not PixelFormat.INDEXED8:
                    raise ValueError("image payload requires an INDEXED8 raster")
                if payload.viewport.raster_shape != (
                    self.raster.height,
                    self.raster.width,
                ):
                    raise ValueError("image payload and raster geometry differ")
                payload_revision = payload.viewport.viewport_revision
                source_input = payload.evaluated_input
            elif isinstance(payload, CurvePanelPayload):
                if self.raster.pixel_format is not PixelFormat.RGBA8888:
                    raise ValueError("curve payload requires an RGBA8888 raster")
                payload_revision = payload.viewport.display_revision
                source_input = payload.evaluated_input
            elif isinstance(payload, HistogramPanelPayload):
                if self.raster.pixel_format is not PixelFormat.RGBA8888:
                    raise ValueError("histogram payload requires an RGBA8888 raster")
                payload_revision = payload.viewport.display_revision
                source_input = payload.evaluated_input
            elif isinstance(payload, MeterPanelPayload):
                if self.raster.pixel_format is not PixelFormat.RGBA8888:
                    raise ValueError("meter payload requires an RGBA8888 raster")
                payload_revision = payload.display_revision
                source_input = payload.evaluated_input
            else:
                background = payload.background
                if self.raster.pixel_format is not PixelFormat.INDEXED8:
                    raise ValueError("site-map payload requires an INDEXED8 raster")
                if background.viewport.raster_shape != (
                    self.raster.height,
                    self.raster.width,
                ):
                    raise ValueError("site-map background and raster geometry differ")
                payload_revision = background.viewport.viewport_revision
                source_input = payload.occupancy_input
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
                if self.coherence_stamp.join_key_digest != payload.join_key_digest:
                    raise ValueError(
                        "site-map coherence digest omits or changes its exact cell, "
                        "inputs, or calibration geometry"
                    )
            if presentations[0].panel_revision != payload_revision:
                raise ValueError(
                    "display payload revision differs from panel presentation"
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
                    "site-map source identity differs from its occupancy input"
                )


@dataclass(frozen=True)
class BoardFrame:
    """One atomic, shot-coherent presentation for a complete board layout."""

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
            stamp = group_panels[0].coherence_stamp
            if any(panel.coherence_stamp != stamp for panel in group_panels[1:]):
                raise ValueError(
                    "panels in one coherence group must carry one exact CoherenceStamp"
                )
            expected_panel_ids = tuple(sorted(panel.panel_id for panel in group_panels))
            if (
                tuple(value.panel_id for value in stamp.presentations)
                != expected_panel_ids
            ):
                raise ValueError(
                    "CoherenceStamp presentations must cover its coherence group exactly"
                )
            inputs = {value.dataset_id: value.ref for value in stamp.inputs}
            for panel in group_panels:
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


@runtime_checkable
class BoardPresenter(Protocol):
    """GUI-side sink; one call presents the entire board coherently."""

    def present(self, frame: BoardFrame) -> None: ...

    def clear(self) -> None: ...


class AtomicBoardFront:
    """Concrete old-or-new front mapping read by a GUI board paint adapter.

    The swap is atomic at the model transaction boundary.  Separate native widgets
    can still be painted by the OS at different instants and must not advertise
    pixel-clock simultaneity; they all nevertheless read one immutable BoardFrame.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: BoardFrame | None = None

    def present(self, frame: BoardFrame) -> None:
        if not isinstance(frame, BoardFrame):
            raise TypeError("frame must be BoardFrame")
        with self._lock:
            self._current = frame

    def current(self) -> BoardFrame | None:
        with self._lock:
            return self._current

    def clear(self) -> None:
        with self._lock:
            self._current = None


__all__ = [
    "BoardFrame",
    "BoardPresenter",
    "AtomicBoardFront",
    "CoherenceStamp",
    "CurveFitOverlay",
    "CurvePanelPayload",
    "HistogramPanelPayload",
    "MeterPanelPayload",
    "detached_render_fault",
    "DisplayPayload",
    "PanelPresentationIdentity",
    "SourceIdentity",
    "PanelFrame",
    "ImagePanelPayload",
    "RadialGaussianImageFitOverlay",
    "SiteMapPanelPayload",
    "PixelFormat",
    "RasterBuffer",
    "RenderSurface",
    "SITE_MAP_JOIN_SCHEMA_DIGEST",
]
