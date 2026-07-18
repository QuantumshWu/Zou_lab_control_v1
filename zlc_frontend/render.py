"""Headless render hand-off values owned by the target frontend package.

The renderer may use Matplotlib, Qt, or neither, but the worker/GUI boundary is
always an immutable :class:`BoardFrame`.  No live Figure, Artist, mutable or
aliased ndarray view, or QImage storage crosses this module's boundary.  The
exact IMAGE/CURVE interaction payloads are allowed because their evaluated
arrays are intrinsically backed by owned immutable bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
import threading
from typing import Protocol, runtime_checkable

from zlc_data import BlockId, StreamGenerationId
from zlc_storage import (
    canonical_text as _text,
    nonnegative_integer as _nonnegative,
    sha256_text,
)

from .curve_display import CurveViewportTransform
from .display_range import validated_display_range
from .figure import (
    DatasetId,
    EvaluatedCurve,
    EvaluatedImage,
    EvaluatedInput,
    EvaluatedSeries,
)
from .image_view import ImageViewportTransform


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
            if len(evaluated.indices) != axis.size or any(
                actual != expected
                for expected, actual in enumerate(evaluated.indices)
            ):
                raise ValueError(f"image payload {name} axis is not the full raster")
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


@dataclass(frozen=True, slots=True)
class CurvePanelPayload:
    """Exact immutable samples and draw mapping for one CURVE raster front."""

    evaluated_input: EvaluatedInput
    viewport: CurveViewportTransform
    series: tuple[EvaluatedSeries, ...]
    series_labels: tuple[str, ...]

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
        object.__setattr__(self, "series", series)
        object.__setattr__(self, "series_labels", labels)

    @property
    def value_unit(self) -> str | None:
        """Return the unit already owned by every validated curve series."""

        curve = self.series[0].data
        assert isinstance(curve, EvaluatedCurve)
        return curve.value_unit


DisplayPayload = ImagePanelPayload | CurvePanelPayload


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
            if not isinstance(payload, (ImagePanelPayload, CurvePanelPayload)):
                raise TypeError(
                    "display_payload must be ImagePanelPayload, "
                    "CurvePanelPayload, or None"
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
            else:
                if self.raster.pixel_format is not PixelFormat.RGBA8888:
                    raise ValueError("curve payload requires an RGBA8888 raster")
                payload_revision = payload.viewport.display_revision
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
                payload.evaluated_input.dataset_id
                != self.source_identity.dataset_id
                or payload.evaluated_input.ref != expected_ref
            ):
                raise ValueError(
                    "display payload input differs from its frozen coherence input"
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
    "CurvePanelPayload",
    "detached_render_fault",
    "DisplayPayload",
    "PanelPresentationIdentity",
    "SourceIdentity",
    "PanelFrame",
    "ImagePanelPayload",
    "PixelFormat",
    "RasterBuffer",
    "RenderSurface",
]
