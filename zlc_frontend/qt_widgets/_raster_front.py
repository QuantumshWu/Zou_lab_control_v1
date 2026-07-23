"""Immutable raster-front preparation, layout, and identity helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from PyQt5 import QtCore, QtGui

from zlc_storage import canonical_text

from ..plot_layout import NormalizedBox
from ..render import (
    BoardFrame,
    CurvePanelPayload,
    DisplayPayload,
    DocumentInputIdentity,
    HistogramPanelPayload,
    ImagePanelPayload,
    MeterPanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    PulsePanelPayload,
    SiteMapPanelPayload,
    SourceIdentity,
)
from ..selector import ImageViewportTransform, PanelInteractionOrigin


def _prepared_qimage(panel_or_raster) -> tuple[bytes, QtGui.QImage]:
    """Wrap one immutable tight RGBA8888 visible raster."""

    if isinstance(panel_or_raster, PanelFrame):
        raster = panel_or_raster.raster
        payload = panel_or_raster.display_payload
        if isinstance(payload, SiteMapPanelPayload):
            payload = payload.background
        elif not isinstance(payload, ImagePanelPayload):
            payload = None
    else:
        raster = panel_or_raster
        payload = None
    image = QtGui.QImage(
        raster.pixels,
        raster.width,
        raster.height,
        raster.width * 4,
        QtGui.QImage.Format_RGBA8888,
    )
    if image.isNull():
        raise RuntimeError("Qt rejected the prepared immutable raster front")
    return raster.pixels, image


def _image_source_rect(
    image: QtGui.QImage,
    viewport: ImageViewportTransform | None,
) -> QtCore.QRectF:
    if viewport is None:
        return QtCore.QRectF(0.0, 0.0, float(image.width()), float(image.height()))
    left, top, right, bottom = viewport.visible_bounds
    return QtCore.QRectF(
        left * image.width(),
        top * image.height(),
        (right - left) * image.width(),
        (bottom - top) * image.height(),
    )


def _aspect_target_for_source(
    bounds: QtCore.QRect,
    source: QtCore.QRectF,
    *,
    aspect_ratio: float | None = None,
    anchor_west: bool = False,
) -> QtCore.QRect:
    if source.width() <= 0.0 or source.height() <= 0.0:
        raise ValueError("image source viewport must have positive geometry")
    source_ratio = (
        source.width() / source.height()
        if aspect_ratio is None
        else float(aspect_ratio)
    )
    if not math.isfinite(source_ratio) or source_ratio <= 0.0:
        raise ValueError("image aspect ratio must be finite and positive")
    bounds_ratio = bounds.width() / max(1, bounds.height())
    if bounds_ratio > source_ratio:
        height = bounds.height()
        width = max(1, int(round(height * source_ratio)))
    else:
        width = bounds.width()
        height = max(1, int(round(width / source_ratio)))
    return QtCore.QRect(
        bounds.x()
        if anchor_west
        else bounds.x() + (bounds.width() - width) // 2,
        bounds.y() + (bounds.height() - height) // 2,
        width,
        height,
    )


def _map_panel_box(bounds: QtCore.QRect, box: NormalizedBox) -> QtCore.QRect:
    left = bounds.x() + round(box.left * bounds.width())
    top = bounds.y() + round(box.top * bounds.height())
    right = bounds.x() + round(box.right * bounds.width())
    bottom = bounds.y() + round(box.bottom * bounds.height())
    return QtCore.QRect(
        left,
        top,
        max(1, right - left),
        max(1, bottom - top),
    )


@dataclass(frozen=True, slots=True)
class _ImagePanelGeometry:
    target: QtCore.QRect
    source: QtCore.QRectF
    distribution: QtCore.QRect | None
    colorbar: QtCore.QRect | None


def _panel_image_geometry(
    bounds: QtCore.QRect,
    image: QtGui.QImage,
    payload: ImagePanelPayload | None,
    *,
    site_map_payload: SiteMapPanelPayload | None = None,
) -> _ImagePanelGeometry:
    source = _image_source_rect(
        image,
        None if payload is None else payload.viewport,
    )
    if payload is None:
        return _ImagePanelGeometry(
            _aspect_target_for_source(bounds, source),
            source,
            None,
            None,
        )
    composed = payload.raster_geometry
    return _ImagePanelGeometry(
        _map_panel_box(bounds, NormalizedBox(*composed.image_bounds)),
        QtCore.QRectF(
            0.0,
            0.0,
            float(image.width()),
            float(image.height()),
        ),
        _map_panel_box(
            bounds,
            NormalizedBox(*composed.distribution_bounds),
        ),
        _map_panel_box(
            bounds,
            NormalizedBox(*composed.colorbar_bounds),
        ),
    )


def _panel_bounds(
    bounds: QtCore.QRect,
    *,
    index: int,
    count: int,
    columns: int,
) -> QtCore.QRect:
    """Return the one grid cell used by both raster paint and hit testing."""

    rows = math.ceil(count / columns)
    cell_width = bounds.width() // columns
    cell_height = bounds.height() // rows
    row, column = divmod(index, columns)
    right = (
        bounds.right() + 1
        if column == columns - 1
        else bounds.x() + (column + 1) * cell_width
    )
    bottom = (
        bounds.bottom() + 1
        if row == rows - 1
        else bounds.y() + (row + 1) * cell_height
    )
    return QtCore.QRect(
        bounds.x() + column * cell_width,
        bounds.y() + row * cell_height,
        right - (bounds.x() + column * cell_width),
        bottom - (bounds.y() + row * cell_height),
    )


def _validated_panel_layout(
    panel_ids: tuple[str, ...],
    columns: int,
) -> tuple[tuple[str, ...], int]:
    ids = tuple(canonical_text(value, "panel_id") for value in panel_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("QtRasterBoard requires unique panel ids")
    if isinstance(columns, bool) or not isinstance(columns, int) or columns <= 0:
        raise ValueError("columns must be a positive integer")
    return ids, min(columns, len(ids))


def _panel_presentation(panel: PanelFrame) -> PanelPresentationIdentity:
    matches = tuple(
        value
        for value in panel.coherence_stamp.presentations
        if value.panel_id == panel.panel_id
    )
    if len(matches) != 1:
        raise RuntimeError("panel coherence stamp has no unique presentation identity")
    return matches[0]


def _raster_geometry(panel: PanelFrame) -> tuple[int, int]:
    raster = panel.raster
    return raster.width, raster.height


def _image_payload(
    panel_or_hold: PanelFrame | _HeldPanelFront,
) -> ImagePanelPayload | None:
    payload = panel_or_hold.display_payload
    if isinstance(payload, SiteMapPanelPayload):
        return payload.background
    return payload if isinstance(payload, ImagePanelPayload) else None


def _site_map_payload(
    panel_or_hold: PanelFrame | _HeldPanelFront,
) -> SiteMapPanelPayload | None:
    payload = panel_or_hold.display_payload
    return payload if isinstance(payload, SiteMapPanelPayload) else None


def _payload_input(
    payload: (
        ImagePanelPayload
        | CurvePanelPayload
        | HistogramPanelPayload
        | MeterPanelPayload
        | PulsePanelPayload
        | SiteMapPanelPayload
    ),
):
    if isinstance(payload, SiteMapPanelPayload):
        return payload.site_state_input
    if isinstance(payload, PulsePanelPayload):
        return payload.document_input
    return payload.evaluated_input


def _input_structure(evaluated_input) -> tuple[object, object, object, str]:
    """Return producer structure while deliberately excluding normal revisions."""

    ref = evaluated_input.ref
    return (
        evaluated_input.dataset_id,
        ref.block_id,
        ref.stream_generation,
        ref.schema_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class _HeldPanelFront:
    """One GUI-owned display overlay; it is never an authoritative BoardFrame."""

    panel_id: str
    board_id: str
    layout_generation: int
    sequence: int
    coherence_group: str
    source_identity: SourceIdentity | DocumentInputIdentity
    presentation: PanelPresentationIdentity
    raster_geometry: tuple[int, int]
    prepared: tuple[bytes, QtGui.QImage]
    display_payload: (
        ImagePanelPayload
        | CurvePanelPayload
        | HistogramPanelPayload
        | PulsePanelPayload
        | SiteMapPanelPayload
        | None
    )

    @property
    def gesture_identity(
        self,
    ) -> tuple[str, int, int, SourceIdentity | DocumentInputIdentity]:
        return (
            self.board_id,
            self.layout_generation,
            self.sequence,
            self.source_identity,
        )


def _presented_revision_state(
    pending_revision: int | None,
    candidate_revision: int | None,
    held_revision: int | None,
) -> tuple[bool, bool]:
    """Classify one worker answer as final or useful intermediate."""

    if pending_revision is None or candidate_revision is None:
        return (False, False)
    if candidate_revision >= pending_revision:
        return (True, False)
    return (
        False,
        held_revision is not None and candidate_revision > held_revision,
    )


def _advance_held_front(
    hold: _HeldPanelFront,
    frame: BoardFrame,
    panel: PanelFrame,
    prepared: tuple[bytes, QtGui.QImage],
) -> _HeldPanelFront:
    """Absorb one same-gesture raster answer into the painted hold."""

    return replace(
        hold,
        sequence=frame.sequence,
        source_identity=panel.source_identity,
        presentation=_panel_presentation(panel),
        raster_geometry=_raster_geometry(panel),
        prepared=prepared,
        display_payload=panel.display_payload,
    )


def _visible_display(
    panel_id: str | None,
    payload_type: type | tuple[type, ...],
    *,
    front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None,
    panel_ids: tuple[str, ...],
    hold: _HeldPanelFront | None,
) -> tuple[DisplayPayload | None, PanelInteractionOrigin | None]:
    """Resolve one typed payload and origin from the exact painted panel."""

    if hold is not None and hold.panel_id == panel_id:
        payload = hold.display_payload
        if not isinstance(payload, payload_type):
            return None, None
        return payload, PanelInteractionOrigin(
            hold.panel_id,
            hold.board_id,
            hold.layout_generation,
            hold.sequence,
            hold.source_identity,
            hold.presentation,
            _payload_input(payload),
        )
    if front is None or panel_id is None or panel_id not in panel_ids:
        return None, None
    panel = front[0].panels[panel_ids.index(panel_id)]
    payload = panel.display_payload
    if not isinstance(payload, payload_type):
        return None, None
    return payload, PanelInteractionOrigin(
        panel.panel_id,
        front[0].board_id,
        front[0].layout_generation,
        front[0].sequence,
        panel.source_identity,
        _panel_presentation(panel),
        _payload_input(payload),
    )


def _panel_semantics_changed(old: PanelFrame, new: PanelFrame) -> bool:
    """Whether a retained data-space interaction no longer has the same meaning."""

    old_presentation = _panel_presentation(old)
    new_presentation = _panel_presentation(new)
    old_payload = old.display_payload
    new_payload = new.display_payload

    def interaction_geometry(payload):
        if isinstance(payload, ImagePanelPayload):
            return (ImagePanelPayload, payload.viewport.axes)
        if isinstance(payload, CurvePanelPayload):
            return (CurvePanelPayload, payload.viewport.x_axis)
        if isinstance(payload, HistogramPanelPayload):
            return (
                HistogramPanelPayload,
                payload.value_unit,
                payload.series_labels,
            )
        if isinstance(payload, MeterPanelPayload):
            return (
                MeterPanelPayload,
                payload.value_unit,
                payload.series_labels,
            )
        if isinstance(payload, PulsePanelPayload):
            return (
                PulsePanelPayload,
                payload.viewport.x_axis,
                payload.row_keys,
            )
        if isinstance(payload, SiteMapPanelPayload):
            return (
                SiteMapPanelPayload,
                payload.background.viewport.axes,
                _input_structure(payload.background.evaluated_input),
                payload.site_axis,
                payload.coordinate_frame,
                payload.geometry_identity,
            )
        return (None,)

    return (
        old.panel_id != new.panel_id
        or old.coherence_group != new.coherence_group
        or old.source_identity != new.source_identity
        or old_presentation.panel_id != new_presentation.panel_id
        or old_presentation.document_id != new_presentation.document_id
        or old_presentation.document_revision != new_presentation.document_revision
        or old_presentation.selection_revision != new_presentation.selection_revision
        or interaction_geometry(old_payload) != interaction_geometry(new_payload)
    )


def _hold_matches_frame(
    hold: _HeldPanelFront,
    frame: BoardFrame,
    *,
    panel_ids: tuple[str, ...],
) -> bool:
    if (
        frame.board_id != hold.board_id
        or frame.layout_generation != hold.layout_generation
        or hold.panel_id not in panel_ids
    ):
        return False
    index = panel_ids.index(hold.panel_id)
    panel = frame.panels[index]
    held_payload = hold.display_payload
    current_payload = panel.display_payload
    if isinstance(held_payload, ImagePanelPayload):
        payload_matches = (
            isinstance(current_payload, ImagePanelPayload)
            and current_payload.viewport.axes == held_payload.viewport.axes
        )
    elif isinstance(held_payload, CurvePanelPayload):
        payload_matches = (
            isinstance(current_payload, CurvePanelPayload)
            and current_payload.viewport.x_axis == held_payload.viewport.x_axis
        )
    elif isinstance(held_payload, HistogramPanelPayload):
        payload_matches = (
            isinstance(current_payload, HistogramPanelPayload)
            and current_payload.value_unit == held_payload.value_unit
            and current_payload.series_labels == held_payload.series_labels
        )
    elif isinstance(held_payload, PulsePanelPayload):
        payload_matches = (
            isinstance(current_payload, PulsePanelPayload)
            and current_payload.viewport.x_axis == held_payload.viewport.x_axis
            and current_payload.row_keys == held_payload.row_keys
        )
    elif isinstance(held_payload, SiteMapPanelPayload):
        payload_matches = (
            isinstance(current_payload, SiteMapPanelPayload)
            and current_payload.background.viewport.axes
            == held_payload.background.viewport.axes
            and _input_structure(current_payload.background.evaluated_input)
            == _input_structure(held_payload.background.evaluated_input)
            and current_payload.site_axis == held_payload.site_axis
            and current_payload.coordinate_frame == held_payload.coordinate_frame
            and current_payload.geometry_identity == held_payload.geometry_identity
        )
    else:
        payload_matches = current_payload is None
    current = _panel_presentation(panel)
    presentation_matches = (
        current.panel_id == hold.presentation.panel_id
        and current.document_id == hold.presentation.document_id
        and current.selection_revision == hold.presentation.selection_revision
        and current.document_revision >= hold.presentation.document_revision
        and current.panel_revision >= hold.presentation.panel_revision
    )
    return (
        panel.panel_id == hold.panel_id
        and panel.coherence_group == hold.coherence_group
        and panel.source_identity == hold.source_identity
        and presentation_matches
        and _raster_geometry(panel) == hold.raster_geometry
        and payload_matches
    )
