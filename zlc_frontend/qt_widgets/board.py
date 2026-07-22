"""Interactive Qt presenter for immutable semantic board fronts."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Callable

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_data import FitBatchStatus, Selection
from zlc_storage import canonical_text, nonnegative_integer

from ..curve_display import CurveViewportTransform, NumericViewportTransform
from ..display_range import RelimMode, validated_display_range
from ..histogram_display import HistogramViewportTransform
from ..image_raster import indexed8_code_for_value
from ..image_view import validate_normalized_rectangle
from ..render import (
    BoardFrame,
    CurvePanelPayload,
    DisplayPayload,
    HistogramPanelPayload,
    ImagePanelPayload,
    MeterPanelPayload,
    PanelFrame,
    PulsePanelPayload,
    RadialGaussianImageFitOverlay,
    SiteMapPanelPayload,
    detached_render_fault,
)
from ..site_map import (
    SITE_EMPTY_ALPHA,
    SITE_EMPTY_COLOR,
    SITE_EMPTY_LINEWIDTH,
    SITE_INVALID_ALPHA,
    SITE_INVALID_COLOR,
    SITE_INVALID_LINEWIDTH,
    SITE_OCCUPIED_ALPHA,
    SITE_OCCUPIED_COLOR,
    SITE_OCCUPIED_LINEWIDTH,
)
from ..selector import (
    CurveInteractionIntent,
    CurveRangeGesture,
    CurveViewportCommit,
    HistogramInteractionIntent,
    HistogramRangeGesture,
    HistogramThresholdCommit,
    HistogramViewportCommit,
    ImageColorLimitsCommit,
    ImageInteractionCommit,
    ImageViewportTransform,
    ImageViewportCommit,
    NormalizedRectangle,
    PanelInteractionOrigin,
    RectangleGesture,
)
from ._raster_board_support import (
    _CurveSample,
    _HeldPanelFront,
    _HistogramBinSample,
    _ImagePanelBinding,
    _ImageSample,
    _NUMERIC_PAYLOAD_TYPES,
    _NumericCross,
    _NumericIntent,
    _NumericKind,
    _NumericPanelBinding,
    _NumericTarget,
    _NumericViewport,
    _advance_held_front,
    _image_payload,
    _input_structure,
    _numeric_payload,
    _numeric_plot_geometry,
    _panel_bounds,
    _panel_image_geometry,
    _panel_presentation,
    _payload_input,
    _prepared_qimage,
    _presented_revision_state,
    _raster_geometry,
    _selector_pen_color,
    _selector_precision,
    _site_map_payload,
    _validated_panel_layout,
)
from .style import (
    BG,
    GREEN,
    ORANGE,
    SELECTOR_DOT_PX,
    SELECTOR_FONT_PX,
    SELECTOR_HANDLE_PX,
    SELECTOR_LINE_PX,
)


class QtRasterBoard(QtWidgets.QWidget):
    """Atomic multi-panel presenter for immutable worker-owned raster fronts."""

    imagePanelLeftDoubleClicked = QtCore.pyqtSignal(str)

    def __init__(
        self,
        panel_ids: tuple[str, ...],
        parent: QtWidgets.QWidget | None = None,
        *,
        columns: int = 2,
        empty_text: str = "",
    ) -> None:
        super().__init__(parent)
        self._panel_ids, self._columns = _validated_panel_layout(panel_ids, columns)
        self._active_layout_identity: tuple[str, int] | None = None
        self._staged_layout: tuple[str, int, tuple[str, ...], int] | None = None
        self._empty_text = str(empty_text)
        self._front: tuple[BoardFrame, tuple[tuple[bytes, QtGui.QImage], ...]] | None = None
        self._selector_enabled = False
        self._selector_hold: _HeldPanelFront | None = None
        self._image_bindings: dict[str, _ImagePanelBinding] = {}
        self._numeric_bindings: dict[str, _NumericPanelBinding] = {}
        self._closed = False
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setMinimumSize(128, 64)

    @property
    def panel_ids(self) -> tuple[str, ...]:
        """Panel order of the currently visible/active layout."""

        self._require_owner()
        return self._panel_ids

    @property
    def columns(self) -> int:
        self._require_owner()
        return self._columns

    def stage_layout(
        self,
        panel_ids: tuple[str, ...],
        *,
        board_id: str,
        layout_generation: int,
        columns: int = 2,
    ) -> None:
        """Admit one newer layout without disturbing the currently painted front."""

        self._require_owner()
        self._ensure_open()
        ids, column_count = _validated_panel_layout(panel_ids, columns)
        identity = (
            canonical_text(board_id, "board_id"),
            nonnegative_integer(layout_generation, "layout_generation"),
        )
        current = self._staged_layout
        floor = (
            self._active_layout_identity
            if current is None
            else (current[0], current[1])
        )
        if floor is not None:
            if identity[0] != floor[0]:
                raise ValueError("QtRasterBoard cannot change board identity")
            if identity[1] <= floor[1]:
                raise ValueError("staged layout_generation must increase")
        self._staged_layout = (*identity, ids, column_count)
        if self._selector_hold is not None:
            self._cancel_active_gesture(
                clear_image_draft=True,
                clear_numeric_spans=True,
            )
            self.update()

    def discard_staged_layout(
        self,
        *,
        board_id: str,
        layout_generation: int,
    ) -> bool:
        """Discard only the named unpresented layout, preserving the old front."""

        self._require_owner()
        if self._closed:
            return False
        identity = (
            canonical_text(board_id, "board_id"),
            nonnegative_integer(layout_generation, "layout_generation"),
        )
        staged = self._staged_layout
        if staged is None or (staged[0], staged[1]) != identity:
            return False
        self._staged_layout = None
        self.update()
        return True

    def present(self, frame: BoardFrame) -> None:
        self._require_owner()
        self._ensure_open()
        interaction_was_active = self._selector_hold is not None
        promoting = False
        cancel_interaction = False
        target_panel_ids = self._panel_ids
        target_columns = self._columns
        target_identity = self._active_layout_identity
        target_image_viewports: dict[str, ImageViewportTransform] = {}
        target_numeric_viewports: dict[str, _NumericViewport] = {}
        try:
            if not isinstance(frame, BoardFrame):
                raise TypeError("frame must be BoardFrame")
            frame_identity = (frame.board_id, frame.layout_generation)
            frame_panel_ids = tuple(panel.panel_id for panel in frame.panels)
            staged = self._staged_layout
            if staged is not None:
                staged_identity = (staged[0], staged[1])
                if frame_identity != staged_identity or frame_panel_ids != staged[2]:
                    raise ValueError(
                        "QtRasterBoard frame does not match its staged layout identity"
                    )
                promoting = True
                target_identity = staged_identity
                target_panel_ids = staged[2]
                target_columns = staged[3]
            elif frame_panel_ids != self._panel_ids:
                raise ValueError(
                    "QtRasterBoard frame does not match its configured panel order"
                )
            elif (
                self._active_layout_identity is not None
                and frame_identity != self._active_layout_identity
            ):
                raise ValueError(
                    "QtRasterBoard frame does not match its active layout identity"
                )
            else:
                target_identity = frame_identity
            for panel_id, binding in self._image_bindings.items():
                if panel_id not in target_panel_ids:
                    continue
                target_viewport = self._viewport_for_presented_panel(
                    binding,
                    frame,
                    panel_ids=target_panel_ids,
                )
                self._validate_selector_binding(
                    panel_id,
                    target_viewport,
                    frame,
                    panel_ids=target_panel_ids,
                )
                target_image_viewports[panel_id] = target_viewport
                pending_origin = binding.pending_origin
                pending_limits = binding.pending_color_limits
                target_panel = frame.panels[
                    target_panel_ids.index(panel_id)
                ]
                if (
                    binding.interaction_callback is not None
                    and _image_payload(target_panel) is None
                ):
                    raise ValueError(
                        "image interaction callback requires exact ImagePanelPayload"
                    )
                if (
                    pending_origin is not None
                    and pending_limits is not None
                    and _image_payload(target_panel) is not None
                    and target_viewport.viewport_revision
                    == pending_origin.presentation.panel_revision + 1
                    and _image_payload(target_panel).color_limits != pending_limits
                ):
                    raise ValueError(
                        "pending image color-limit revision returned conflicting limits"
                    )
            for panel_id, binding in self._numeric_bindings.items():
                if panel_id in target_panel_ids:
                    target_numeric_viewports[panel_id] = (
                        self._numeric_viewport_for_presented_panel(
                            binding,
                            frame,
                            panel_ids=target_panel_ids,
                        )
                    )
            if interaction_was_active:
                hold = self._selector_hold
                if hold is None:
                    raise RuntimeError(
                        "active rectangle interaction has no held panel front"
                    )
                cancel_interaction = not self._hold_matches_frame(
                    hold,
                    frame,
                    panel_ids=target_panel_ids,
                )
            # Only after all cheap identity/revision checks pass may INDEXED8
            # setColorTable detach a potentially multi-megapixel QImage plane.
            prepared = tuple(_prepared_qimage(panel) for panel in frame.panels)
        except BaseException:
            if promoting:
                self._staged_layout = None
            if interaction_was_active:
                self._cancel_active_gesture(
                    clear_image_draft=True,
                    clear_numeric_spans=True,
                )
                self.update()
            raise
        previous = self._front
        for panel_id, binding in tuple(self._image_bindings.items()):
            if panel_id not in target_panel_ids:
                self._reset_image_binding(panel_id)
            elif previous is not None and panel_id in self._panel_ids:
                old_index = self._panel_ids.index(panel_id)
                new_index = target_panel_ids.index(panel_id)
                old_panel = previous[0].panels[old_index]
                new_panel = frame.panels[new_index]
                if self._panel_semantics_changed(old_panel, new_panel):
                    self._clear_image_transient(
                        binding,
                        clear_applied_bounds=True,
                        clear_pending=True,
                    )
                    target_viewport = target_image_viewports.get(panel_id)
                    if target_viewport is not None:
                        binding.revision_floor = (
                            target_viewport.viewport_revision
                        )
                    cancel_interaction = cancel_interaction or (
                        interaction_was_active
                        and self._selector_hold is not None
                        and self._selector_hold.panel_id == panel_id
                    )
        for panel_id, binding in tuple(self._numeric_bindings.items()):
            if panel_id not in target_panel_ids:
                self._reset_numeric_binding(panel_id)
            elif previous is not None and panel_id in self._panel_ids:
                old_panel = previous[0].panels[self._panel_ids.index(panel_id)]
                new_panel = frame.panels[target_panel_ids.index(panel_id)]
                if self._panel_semantics_changed(old_panel, new_panel):
                    self._clear_numeric_transient(
                        binding,
                        clear_applied_span=True,
                        clear_pending=True,
                    )
                    target_viewport = target_numeric_viewports.get(panel_id)
                    if target_viewport is not None:
                        binding.revision_floor = (
                            target_viewport.display_revision
                        )
                    cancel_interaction = cancel_interaction or (
                        interaction_was_active
                        and self._selector_hold is not None
                        and self._selector_hold.panel_id == panel_id
                    )
        if cancel_interaction:
            self._cancel_active_gesture(
                clear_image_draft=True,
                clear_numeric_spans=True,
            )
        if promoting:
            self._panel_ids = target_panel_ids
            self._columns = target_columns
            self._staged_layout = None
        self._active_layout_identity = target_identity
        for panel_id, viewport in target_image_viewports.items():
            binding = self._image_bindings.get(panel_id)
            if binding is not None:
                binding.viewport = viewport
                binding.revision_floor = max(
                    binding.revision_floor, viewport.viewport_revision)
        for panel_id, viewport in target_numeric_viewports.items():
            binding = self._numeric_bindings.get(panel_id)
            if binding is not None:
                binding.viewport = viewport
                binding.revision_floor = max(
                    binding.revision_floor, viewport.display_revision)
        for panel_id, binding in self._image_bindings.items():
            target_viewport = target_image_viewports.get(panel_id)
            pending = binding.pending_viewport
            hold = self._selector_hold
            answered, intermediate_answer = _presented_revision_state(
                None if pending is None else pending.viewport_revision,
                None if target_viewport is None else target_viewport.viewport_revision,
                (
                    hold.presentation.panel_revision
                    if hold is not None and hold.panel_id == panel_id
                    else None
                ),
            )
            if answered:
                binding.pending_viewport = None
            pending_origin = binding.pending_origin
            if (
                binding.pending_color_limits is not None
                and target_viewport is not None
                and pending_origin is not None
                and target_viewport.viewport_revision
                > pending_origin.presentation.panel_revision
            ):
                binding.pending_color_limits = None
                answered = True
            if not self._image_interaction_is_pending(binding):
                binding.pending_origin = None
            if (
                (answered or intermediate_answer)
                and hold is not None
                and hold.panel_id == panel_id
            ):
                index = target_panel_ids.index(panel_id)
                panel = frame.panels[index]
                self._selector_hold = _advance_held_front(
                    hold, frame, panel, prepared[index]
                )
        for panel_id, binding in self._numeric_bindings.items():
            pending = binding.pending_viewport
            candidate = target_numeric_viewports.get(panel_id)
            hold = self._selector_hold
            answered, intermediate_answer = _presented_revision_state(
                None if pending is None else pending.display_revision,
                None if candidate is None else candidate.display_revision,
                (
                    hold.presentation.panel_revision
                    if hold is not None and hold.panel_id == panel_id
                    else None
                ),
            )
            if answered:
                binding.pending_viewport = None
            if (
                binding.threshold_pending_revision is not None
                and candidate is not None
                and candidate.display_revision
                >= binding.threshold_pending_revision
            ):
                binding.threshold_pending_revision = None
                binding.threshold_pending_origin = None
                answered = True
            if answered or intermediate_answer:
                if hold is not None and hold.panel_id == panel_id:
                    index = target_panel_ids.index(panel_id)
                    panel = frame.panels[index]
                    self._selector_hold = _advance_held_front(
                        hold, frame, panel, prepared[index]
                    )
            if binding.pending_viewport is None:
                binding.pending_origin = None
        image_hover_positions = {
            panel_id: binding.hover_position
            for panel_id, binding in self._image_bindings.items()
        }
        numeric_hover_positions = {
            panel_id: binding.hover_position
            for panel_id, binding in self._numeric_bindings.items()
        }
        self._front = (frame, prepared)
        for panel_id, binding in self._image_bindings.items():
            hover_position = image_hover_positions.get(panel_id)
            target = self._selector_target(binding)
            sample = None
            if (
                self._image_interaction_armed(binding)
                and not self._image_interaction_is_pending(binding)
                and hover_position is not None
                and target is not None
                and target[0].contains(hover_position.toPoint())
            ):
                sample = self._sample_for_target(target, hover_position)
            if sample is not None and hover_position is not None:
                binding.hover_position = QtCore.QPointF(hover_position)
            self._set_hover_sample(binding, sample)
        for panel_id, binding in self._numeric_bindings.items():
            position = numeric_hover_positions.get(panel_id)
            target = self._numeric_target(binding)
            sample = None
            if (
                self._numeric_interaction_armed(binding)
                and binding.pending_viewport is None
                and position is not None
                and target is not None
                and target.plot.contains(position)
            ):
                sample = self._numeric_sample_for_target(target, position)
            if sample is not None and position is not None:
                binding.hover_position = QtCore.QPointF(position)
            self._set_numeric_hover(binding, sample)
        self.update()

    def clear(self) -> None:
        self._require_owner()
        self._front = None
        self._active_layout_identity = None
        self._staged_layout = None
        self._cancel_active_gesture(
            clear_image_draft=True,
            clear_numeric_spans=True,
        )
        for binding in self._image_bindings.values():
            self._clear_image_transient(
                binding,
                clear_applied_bounds=True,
                clear_pending=True,
            )
        for binding in self._numeric_bindings.values():
            self._clear_numeric_transient(
                binding,
                clear_applied_span=True,
                clear_pending=True,
            )
        self.update()

    @property
    def has_front(self) -> bool:
        return self._front is not None

    @property
    def front_frame(self) -> BoardFrame | None:
        return None if self._front is None else self._front[0]

    @property
    def selector_fault(self) -> RuntimeError | None:
        self._require_owner()
        binding = self._image_binding()
        return None if binding is None else binding.fault

    def image_selector_fault(
        self,
        panel_id: str | None = None,
    ) -> RuntimeError | None:
        """Return one image panel's isolated interaction fault."""

        self._require_owner()
        binding = self._image_binding(panel_id)
        return None if binding is None else binding.fault

    @property
    def curve_selector_fault(self) -> RuntimeError | None:
        self._require_owner()
        binding = self._numeric_binding_for_kind("curve")
        return None if binding is None else binding.fault

    @property
    def histogram_selector_fault(self) -> RuntimeError | None:
        self._require_owner()
        binding = self._numeric_binding_for_kind("histogram")
        return None if binding is None else binding.fault

    @property
    def pulse_selector_fault(self) -> RuntimeError | None:
        self._require_owner()
        binding = self._numeric_binding_for_kind("pulse")
        return None if binding is None else binding.fault

    @property
    def selectors_enabled(self) -> bool:
        """Return the effective board-wide interaction intent."""

        self._require_owner()
        return self._selector_enabled

    def _visible_display(
        self,
        panel_id: str | None,
        payload_type: type | tuple[type, ...],
    ) -> tuple[
        DisplayPayload | None,
        PanelInteractionOrigin | None,
    ]:
        """Resolve one typed payload and origin from the exact painted panel."""

        hold = self._selector_hold
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
        front = self._front
        if front is None or panel_id is None or panel_id not in self._panel_ids:
            return None, None
        panel = front[0].panels[self._panel_ids.index(panel_id)]
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

    def visible_image_payload(
        self,
        panel_id: str | None = None,
    ) -> ImagePanelPayload | None:
        """Return the exact samples paired with the currently painted IMAGE.

        During A/pan interaction this is the held target payload, not the
        advancing board front.  Setting/Edit can therefore freeze FIXED limits
        from exactly what the operator sees without retaining a BoardFrame.
        """

        self._require_owner()
        binding = self._image_binding(panel_id)
        payload, _origin = self._visible_display(
            None if binding is None else binding.panel_id,
            (ImagePanelPayload, SiteMapPanelPayload),
        )
        if isinstance(payload, SiteMapPanelPayload):
            return payload.background
        return payload if isinstance(payload, ImagePanelPayload) else None

    def visible_image_origin(
        self,
        panel_id: str | None = None,
    ) -> PanelInteractionOrigin | None:
        """Return provenance for the exact held/current IMAGE being painted."""

        self._require_owner()
        binding = self._image_binding(panel_id)
        _payload, origin = self._visible_display(
            None if binding is None else binding.panel_id,
            (ImagePanelPayload, SiteMapPanelPayload),
        )
        return origin

    def visible_site_map_payload(
        self,
        panel_id: str | None = None,
    ) -> SiteMapPanelPayload | None:
        """Return the exact composite payload painted by the image-family panel."""

        self._require_owner()
        binding = self._image_binding(panel_id)
        payload, _origin = self._visible_display(
            None if binding is None else binding.panel_id,
            SiteMapPanelPayload,
        )
        return payload if isinstance(payload, SiteMapPanelPayload) else None

    def discard_pending_image_interaction(
        self,
        origin: PanelInteractionOrigin,
    ) -> bool:
        """Release only one exact failed display intent.

        The owner calls this after an asynchronously accepted reconfigure ends
        in a terminal render fault.  A delayed failure cannot clear a newer
        pending command because sequence, source, presentation revision, and
        exact evaluated input all participate in ``origin`` equality.
        """

        self._require_owner()
        if not isinstance(origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        binding = self._image_bindings.get(origin.panel_id)
        if (
            binding is None
            or not self._image_interaction_is_pending(binding)
            or origin != binding.pending_origin
        ):
            return False
        binding.pending_viewport = None
        binding.pending_color_limits = None
        binding.pending_origin = None
        self.update()
        return True

    def visible_curve_payload(
        self,
        panel_id: str | None = None,
    ) -> CurvePanelPayload | None:
        """Return the exact held/current CURVE payload currently painted."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("curve", panel_id=panel_id)
        payload, _origin = self._visible_display(
            None if binding is None else binding.panel_id, CurvePanelPayload
        )
        return payload if isinstance(payload, CurvePanelPayload) else None

    def visible_curve_origin(
        self,
        panel_id: str | None = None,
    ) -> PanelInteractionOrigin | None:
        """Return provenance for the exact held/current CURVE being painted."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("curve", panel_id=panel_id)
        _payload, origin = self._visible_display(
            None if binding is None else binding.panel_id, CurvePanelPayload
        )
        return origin

    def visible_histogram_payload(
        self,
        panel_id: str | None = None,
    ) -> HistogramPanelPayload | None:
        """Return the exact held/current HISTOGRAM payload currently painted."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("histogram", panel_id=panel_id)
        payload, _origin = self._visible_display(
            None if binding is None else binding.panel_id, HistogramPanelPayload
        )
        return payload if isinstance(payload, HistogramPanelPayload) else None

    def visible_meter_payload(
        self,
        panel_id: str | None = None,
    ) -> MeterPanelPayload | None:
        """Return the exact display-only METER payload currently painted."""

        self._require_owner()
        if panel_id is None:
            frame = None if self._front is None else self._front[0]
            if frame is None:
                return None
            matches = tuple(
                panel.panel_id
                for panel in frame.panels
                if isinstance(panel.display_payload, MeterPanelPayload)
            )
            if len(matches) != 1:
                return None
            panel_id = matches[0]
        payload, _origin = self._visible_display(panel_id, MeterPanelPayload)
        return payload if isinstance(payload, MeterPanelPayload) else None

    def visible_histogram_origin(
        self,
        panel_id: str | None = None,
    ) -> PanelInteractionOrigin | None:
        """Return provenance for the exact held/current HISTOGRAM front."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("histogram", panel_id=panel_id)
        _payload, origin = self._visible_display(
            None if binding is None else binding.panel_id, HistogramPanelPayload
        )
        return origin

    def visible_pulse_payload(
        self,
        panel_id: str | None = None,
    ) -> PulsePanelPayload | None:
        """Return the exact held/current PULSE payload currently painted."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("pulse", panel_id=panel_id)
        payload, _origin = self._visible_display(
            None if binding is None else binding.panel_id, PulsePanelPayload
        )
        return payload if isinstance(payload, PulsePanelPayload) else None

    def visible_pulse_origin(
        self,
        panel_id: str | None = None,
    ) -> PanelInteractionOrigin | None:
        """Return provenance for the exact held/current PULSE front."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("pulse", panel_id=panel_id)
        _payload, origin = self._visible_display(
            None if binding is None else binding.panel_id, PulsePanelPayload
        )
        return origin

    def discard_pending_curve_interaction(
        self,
        origin: PanelInteractionOrigin,
    ) -> bool:
        """Discard only the exact failed CURVE display intent."""

        self._require_owner()
        if not isinstance(origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        binding = self._numeric_bindings.get(origin.panel_id)
        if (
            binding is None
            or binding.kind != "curve"
            or binding.pending_viewport is None
            or origin != binding.pending_origin
        ):
            return False
        binding.pending_viewport = None
        binding.pending_origin = None
        self.update()
        return True

    def discard_pending_histogram_interaction(
        self,
        origin: PanelInteractionOrigin,
    ) -> bool:
        """Discard only the exact failed HISTOGRAM display intent."""

        self._require_owner()
        if not isinstance(origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        binding = self._numeric_bindings.get(origin.panel_id)
        if binding is None or binding.kind != "histogram":
            return False
        discarded = False
        if binding.pending_viewport is not None and origin == binding.pending_origin:
            binding.pending_viewport = None
            binding.pending_origin = None
            discarded = True
        if (
            binding.threshold_pending_revision is not None
            and origin == binding.threshold_pending_origin
        ):
            binding.threshold_pending_revision = None
            binding.threshold_pending_origin = None
            discarded = True
        if not discarded:
            return False
        self.update()
        return True

    def discard_pending_pulse_interaction(
        self,
        origin: PanelInteractionOrigin,
    ) -> bool:
        """Discard only the exact failed PULSE display intent."""

        self._require_owner()
        if not isinstance(origin, PanelInteractionOrigin):
            raise TypeError("origin must be PanelInteractionOrigin")
        binding = self._numeric_bindings.get(origin.panel_id)
        if (
            binding is None
            or binding.kind != "pulse"
            or binding.pending_viewport is None
            or origin != binding.pending_origin
        ):
            return False
        binding.pending_viewport = None
        binding.pending_origin = None
        self.update()
        return True

    def selection_for_rectangle_gesture(self, gesture: RectangleGesture) -> Selection:
        """Resolve a gesture only while its exact display-only origin is held."""

        self._require_owner()
        if not isinstance(gesture, RectangleGesture):
            raise TypeError("gesture must be RectangleGesture")
        hold = self._selector_hold
        binding = self._image_bindings.get(gesture.panel_id)
        if (
            hold is None
            or binding is None
            or binding.drag_anchor is not None
            or binding.pan_anchor is not None
        ):
            raise RuntimeError("rectangle gesture has no completed held origin")
        if gesture.panel_id != hold.panel_id or (
            gesture.board_id,
            gesture.layout_generation,
            gesture.sequence,
            gesture.source_identity,
        ) != hold.gesture_identity:
            raise RuntimeError("rectangle gesture differs from its held panel origin")
        if isinstance(hold.display_payload, SiteMapPanelPayload):
            raise RuntimeError(
                "site-map rectangles are display-only candidates; "
                "a spatial box cannot be promoted to authoritative SITE selection"
            )
        viewport = self._require_selector_viewport(binding)
        if viewport.viewport_revision != gesture.viewport_revision:
            raise RuntimeError("rectangle gesture viewport changed before dispatch")
        front = self._front
        if front is None or not self._hold_matches_frame(
            hold,
            front[0],
            panel_ids=self._panel_ids,
        ):
            raise RuntimeError("rectangle gesture origin is stale for this panel binding")
        return viewport.selection_for_normalized_bounds(gesture.normalized_bounds)

    def selection_for_curve_range_gesture(
        self,
        gesture: CurveRangeGesture,
    ) -> Selection:
        """Resolve one painted CURVE span while its exact origin is held.

        Axis identity and coordinate frame come from the immutable payload that
        was actually under the pointer.  The helper never infers an axis from
        array rank, current zoom, or a later front, and a cleared span is not an
        authority candidate.
        """

        self._require_owner()
        if not isinstance(gesture, CurveRangeGesture):
            raise TypeError("gesture must be CurveRangeGesture")
        if gesture.x_span is None:
            raise ValueError("a cleared curve span has no fit Selection")
        hold = self._selector_hold
        binding = self._numeric_bindings.get(gesture.origin.panel_id)
        if (
            hold is None
            or binding is None
            or binding.kind != "curve"
            or binding.span_anchor is not None
            or binding.pan_anchor is not None
        ):
            if binding is not None and binding.kind == "pulse":
                raise RuntimeError(
                    "pulse ranges are display-only and cannot become Selection"
                )
            raise RuntimeError("curve range gesture has no completed held origin")
        origin = self._numeric_interaction_origin(binding, hold=hold)
        if gesture.origin != origin:
            raise RuntimeError("curve range gesture differs from its held panel origin")
        payload = hold.display_payload
        if not isinstance(payload, CurvePanelPayload):
            raise RuntimeError("curve range gesture lost its exact curve payload")
        axis = payload.viewport.x_axis
        return Selection.coordinate_range(
            axis.axis_id,
            gesture.x_span[0],
            gesture.x_span[1],
            coordinate_frame=axis.coordinate_frame,
        )

    def bind_rectangle_selector(
        self,
        panel_id: str,
        viewport: ImageViewportTransform,
        callback: Callable[[RectangleGesture], object],
        *,
        enabled: bool = True,
        interaction_callback: Callable[[ImageInteractionCommit], object] | None = None,
    ) -> None:
        """Bind one image panel without giving the widget a runtime control sink."""

        self._require_owner()
        panel_id = canonical_text(panel_id, "selector panel_id")
        if panel_id not in self._panel_ids:
            raise ValueError("selector panel_id is absent from this board")
        if not isinstance(viewport, ImageViewportTransform):
            raise TypeError("viewport must be ImageViewportTransform")
        if not callable(callback):
            raise TypeError("selector callback must be callable")
        if interaction_callback is not None and not callable(interaction_callback):
            raise TypeError("interaction_callback must be callable or None")
        if not isinstance(enabled, bool):
            raise TypeError("selector enabled must be bool")
        has_other_binding = bool(
            self._numeric_bindings
            or any(bound_id != panel_id for bound_id in self._image_bindings)
        )
        if has_other_binding and enabled != self._selector_enabled:
            raise ValueError(
                "a second selector family must match the board-wide enabled state; "
                "call set_selectors_enabled explicitly"
            )
        if self._front is not None:
            self._validate_selector_binding(
                panel_id,
                viewport,
                self._front[0],
            )
            if interaction_callback is not None:
                index = self._panel_ids.index(panel_id)
                if _image_payload(self._front[0].panels[index]) is None:
                    raise ValueError(
                        "image interaction callback requires exact ImagePanelPayload"
                    )
        if panel_id in self._image_bindings:
            self._reset_image_binding(panel_id)
        self._image_bindings[panel_id] = _ImagePanelBinding(
            panel_id,
            viewport,
            callback,
            interaction_callback=interaction_callback,
            revision_floor=viewport.viewport_revision,
            interaction_ready=enabled,
        )
        if not has_other_binding:
            self._selector_enabled = enabled
        self.update()

    def set_interaction_readiness(
        self,
        *,
        image: bool,
        curve: bool,
        histogram: bool = False,
        pulse: bool = False,
    ) -> None:
        """Arm only panel families whose painted provenance is current.

        ``set_selectors_enabled`` carries the operator's board-wide intent.
        Readiness is a separate presentation fact supplied by the owner after
        comparing each painted payload with its current semantic state.  A
        stale sibling therefore cannot emit an intent merely because another
        panel on the same board is current.
        """

        self._require_owner()
        if (
            not isinstance(image, bool)
            or not isinstance(curve, bool)
            or not isinstance(histogram, bool)
            or not isinstance(pulse, bool)
        ):
            raise TypeError("interaction readiness values must be bool")
        for binding in self._image_bindings.values():
            if not image and binding.interaction_ready:
                self._cancel_image_gesture(binding, clear_draft=True)
                self._set_hover_sample(binding, None)
            binding.interaction_ready = image
        readiness = {"curve": curve, "histogram": histogram, "pulse": pulse}
        for binding in self._numeric_bindings.values():
            ready = readiness[binding.kind]
            if not ready and binding.interaction_ready:
                self._cancel_numeric_gesture(binding, clear_span=True)
                self._set_numeric_hover(binding, None)
            binding.interaction_ready = ready
        self.update()

    def set_selectors_enabled(self, enabled: bool) -> None:
        """Park or arm all healthy bound selector families without rebuilding."""

        self._require_owner()
        if not isinstance(enabled, bool):
            raise TypeError("selector enabled must be bool")
        healthy_image = any(
            binding.binding_enabled
            and binding.interaction_ready
            and binding.fault is None
            for binding in self._image_bindings.values()
        )
        healthy_numeric = any(
            binding.binding_enabled
            and binding.interaction_ready
            and binding.viewport is not None
            and binding.fault is None
            for binding in self._numeric_bindings.values()
        )
        if enabled and not (healthy_image or healthy_numeric):
            raise RuntimeError("no healthy selector binding is available")
        self._selector_enabled = enabled
        if not enabled:
            self._cancel_active_gesture(
                clear_image_draft=True,
                clear_numeric_spans=True,
            )
            for binding in self._image_bindings.values():
                self._set_hover_sample(binding, None)
            for binding in self._numeric_bindings.values():
                self._set_numeric_hover(binding, None)
        self.update()

    def _image_interaction_armed(self, binding: _ImagePanelBinding) -> bool:
        return (
            self._selector_enabled
            and binding.binding_enabled
            and binding.interaction_ready
            and binding.fault is None
        )

    def set_image_interaction_readiness(
        self,
        panel_id: str,
        ready: bool,
    ) -> None:
        """Set readiness for exactly one bound image-family panel."""

        self._require_owner()
        if not isinstance(ready, bool):
            raise TypeError("image interaction readiness must be bool")
        binding = self._require_image_binding(panel_id)
        if not ready and binding.interaction_ready:
            self._cancel_image_gesture(binding, clear_draft=True)
            self._set_hover_sample(binding, None)
        binding.interaction_ready = ready
        self.update()

    def _numeric_interaction_armed(self, binding: _NumericPanelBinding) -> bool:
        return (
            self._selector_enabled
            and binding.binding_enabled
            and binding.interaction_ready
            and binding.viewport is not None
            and binding.fault is None
        )

    def bind_curve_interaction(
        self,
        panel_id: str,
        callback: Callable[[CurveInteractionIntent], object],
        *,
        enabled: bool = True,
    ) -> None:
        """Bind one CURVE panel to display-only typed intents."""

        self._bind_numeric_interaction(
            "curve", panel_id, callback, enabled=enabled
        )

    def bind_histogram_interaction(
        self,
        panel_id: str,
        callback: Callable[[HistogramInteractionIntent], object],
        *,
        enabled: bool = True,
    ) -> None:
        """Bind one HISTOGRAM panel to display-only typed intents."""

        self._bind_numeric_interaction(
            "histogram", panel_id, callback, enabled=enabled
        )

    def bind_pulse_interaction(
        self,
        panel_id: str,
        callback: Callable[[CurveInteractionIntent], object],
        *,
        enabled: bool = True,
    ) -> None:
        """Bind one PULSE timeline panel to display-only typed intents.

        The pulse timeline is an x-only interactive surface, so it speaks the
        CURVE intent vocabulary (``CurveRangeGesture`` for area,
        ``CurveViewportCommit`` for wheel zoom / middle-drag pan) over its
        :class:`PulsePanelPayload` front -- one gesture owner, no second
        selector family.
        """

        self._bind_numeric_interaction(
            "pulse", panel_id, callback, enabled=enabled
        )

    def set_selector_applied_selection(
        self,
        selection: Selection | None,
        *,
        panel_id: str | None = None,
    ) -> None:
        self._require_owner()
        binding = self._require_image_binding(panel_id)
        viewport = self._require_selector_viewport(binding)
        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("applied selection must be zlc_data.Selection or None")
        binding.applied_bounds = (
            None
            if selection is None
            else viewport.normalized_bounds_for_selection(selection)
        )
        self.update()

    def set_selector_draft_selection(
        self,
        selection: Selection | None,
        *,
        panel_id: str | None = None,
    ) -> None:
        self._require_owner()
        binding = self._require_image_binding(panel_id)
        viewport = self._require_selector_viewport(binding)
        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("draft selection must be zlc_data.Selection or None")
        binding.draft_bounds = (
            None
            if selection is None
            else viewport.normalized_bounds_for_selection(selection)
        )
        self._cancel_image_gesture(binding, clear_draft=False)
        self.update()

    def set_image_rectangle_candidate(
        self,
        bounds: NormalizedRectangle | None,
        *,
        panel_id: str | None = None,
    ) -> None:
        """Retain one image-family rectangle without forging data authority.

        The candidate is complete-raster normalized display state.  It is
        suitable for IMAGE and Sites zoom-to-area UX, but deliberately never
        becomes :class:`zlc_data.Selection` inside the Qt leaf.
        """

        self._require_owner()
        binding = self._require_image_binding(panel_id)
        if self.visible_image_payload(binding.panel_id) is None:
            raise RuntimeError("no exact image-family payload is currently painted")
        binding.applied_bounds = (
            None if bounds is None else validate_normalized_rectangle(bounds)
        )
        self._cancel_image_gesture(binding, clear_draft=True)
        self.update()

    def set_curve_range_candidate(
        self,
        x_span: tuple[float, float] | None,
        *,
        panel_id: str | None = None,
    ) -> None:
        """Project the Workbench-owned display-only CURVE range candidate."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("curve", panel_id=panel_id)
        if binding is None:
            if x_span is None:
                return
            raise RuntimeError("no curve panel is bound")
        binding.applied_span = (
            None
            if x_span is None
            else validated_display_range(x_span, "curve range candidate")
        )
        self.update()

    def set_histogram_range_candidate(
        self,
        x_span: tuple[float, float] | None,
        *,
        panel_id: str | None = None,
    ) -> None:
        """Project one display-only HISTOGRAM value-range candidate."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("histogram", panel_id=panel_id)
        if binding is None:
            if x_span is None:
                return
            raise RuntimeError("no histogram panel is bound")
        binding.applied_span = (
            None
            if x_span is None
            else validated_display_range(x_span, "histogram range candidate")
        )
        self.update()

    def set_pulse_range_candidate(
        self,
        x_span: tuple[float, float] | None,
        *,
        panel_id: str | None = None,
    ) -> None:
        """Project the owner-applied display-only PULSE time-range candidate."""

        self._require_owner()
        binding = self._numeric_binding_for_kind("pulse", panel_id=panel_id)
        if binding is None:
            if x_span is None:
                return
            raise RuntimeError("no pulse panel is bound")
        binding.applied_span = (
            None
            if x_span is None
            else validated_display_range(x_span, "pulse range candidate")
        )
        self.update()

    def unbind_rectangle_selector(self, panel_id: str | None = None) -> None:
        self._require_owner()
        binding = self._image_binding(panel_id)
        if binding is not None:
            self._reset_image_binding(binding.panel_id)
        self.update()

    def unbind_curve_interaction(self, panel_id: str | None = None) -> None:
        self._require_owner()
        binding = self._numeric_binding_for_kind("curve", panel_id=panel_id)
        if binding is not None:
            self._reset_numeric_binding(binding.panel_id)
        self.update()

    def unbind_histogram_interaction(self, panel_id: str | None = None) -> None:
        self._require_owner()
        binding = self._numeric_binding_for_kind("histogram", panel_id=panel_id)
        if binding is not None:
            self._reset_numeric_binding(binding.panel_id)
        self.update()

    def unbind_pulse_interaction(self, panel_id: str | None = None) -> None:
        self._require_owner()
        binding = self._numeric_binding_for_kind("pulse", panel_id=panel_id)
        if binding is not None:
            self._reset_numeric_binding(binding.panel_id)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        del event
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.black)
        front = self._front
        if front is None:
            if self._empty_text:
                painter.setPen(QtGui.QColor(BG))
                painter.drawText(self.rect(), QtCore.Qt.AlignCenter, self._empty_text)
            return
        images = front[1]
        hold = self._selector_hold
        if hold is not None and not self._hold_matches_frame(
            hold,
            front[0],
            panel_ids=self._panel_ids,
        ):
            hold = None
        for index, (_pixels, latest_image) in enumerate(images):
            panel_id = self._panel_ids[index]
            panel = front[0].panels[index]
            image = (
                hold.prepared[1]
                if hold is not None and hold.panel_id == panel_id
                else latest_image
            )
            payload = (
                hold.display_payload
                if hold is not None and hold.panel_id == panel_id
                else panel.display_payload
            )
            bounds = _panel_bounds(
                self.rect(),
                index=index,
                count=len(images),
                columns=self._columns,
            )
            image_payload = None
            if isinstance(
                payload,
                (CurvePanelPayload, HistogramPanelPayload, MeterPanelPayload),
            ):
                target = bounds
                source = QtCore.QRectF(
                    0.0,
                    0.0,
                    float(image.width()),
                    float(image.height()),
                )
                rail = None
            else:
                image_payload = (
                    payload.background
                    if isinstance(payload, SiteMapPanelPayload)
                    else payload
                    if isinstance(payload, ImagePanelPayload)
                    else None
                )
                target, source, rail = _panel_image_geometry(
                    bounds,
                    image,
                    image_payload,
                    site_map_payload=(
                        payload
                        if isinstance(payload, SiteMapPanelPayload)
                        else None
                    ),
                )
            painter.drawImage(QtCore.QRectF(target), image, source)
            if isinstance(payload, ImagePanelPayload):
                self._paint_radial_fit_overlay(painter, payload, target)
            if isinstance(payload, SiteMapPanelPayload):
                self._paint_site_map_rings(painter, payload, target)
            if image_payload is not None and rail is not None:
                self._paint_color_rail(
                    painter,
                    image_payload,
                    rail,
                    self._image_bindings.get(panel_id),
                )
        self._paint_selector_overlays(painter)
        self._paint_numeric_overlays(painter)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._selector_enabled:
            super().mousePressEvent(event)
            return
        numeric_target = self._numeric_target_at(event.localPos())
        if (
            numeric_target is not None
            and self._numeric_interaction_armed(numeric_target.binding)
        ):
            binding = numeric_target.binding
            if (
                binding.pending_viewport is not None
                or self._selector_hold is not None
            ):
                event.accept()
                return
            point = self._numeric_normalized_point(
                numeric_target, event.localPos()
            )
            viewport = numeric_target.payload.viewport
            if event.button() == QtCore.Qt.RightButton:
                x, y = viewport.widget_normalized_to_data(*point)
                binding.cross = _NumericCross(x, y)
                self._set_numeric_hover(binding, None)
                self.update()
                event.accept()
                return
            if event.button() == QtCore.Qt.MiddleButton:
                self._selector_hold = self._held_panel_from_numeric_target(
                    numeric_target
                )
                binding.pan_anchor = point[0]
                binding.pan_origin = viewport
                binding.pan_candidate = viewport.x_limits
                self._set_numeric_hover(binding, None)
                self.update()
                event.accept()
                return
            if event.button() == QtCore.Qt.LeftButton:
                # The reference's DragVLine takes PRIORITY over the area
                # selector: a left press within 2% of the x span of an
                # authored threshold line grabs THAT line, and the area
                # machinery stays untouched for the whole drag (the exclusive
                # arrangement the design's histogram row freezes).
                grabbed = self._threshold_line_hit(
                    numeric_target, event.localPos())
                if grabbed is not None:
                    self._selector_hold = (
                        self._held_panel_from_numeric_target(numeric_target))
                    binding.threshold_drag = grabbed
                    binding.threshold_candidate = tuple(
                        numeric_target.payload.thresholds)
                    self._set_numeric_hover(binding, None)
                    self.update()
                    event.accept()
                    return
                # The reference's RectangleSelector on a STANDING box: grab the
                # centre to move it, a corner/edge handle to resize it, and a
                # press anywhere else discards it and pulls a fresh box.  The
                # box lives in DATA coordinates (the reference's artists do),
                # so it follows the data through any zoom/pan between
                # gestures; hit-testing maps it to widget pixels on demand.
                standing = (
                    binding.span_rect if binding.span_anchor is None else None
                )
                handle = None
                if standing is not None:
                    xs_px, ys_px = self._span_rect_widget_extents(
                        numeric_target, standing)
                    handle = self._span_handle_hit(
                        xs_px, ys_px, event.localPos())
                self._selector_hold = self._held_panel_from_numeric_target(
                    numeric_target
                )
                pressed = viewport.widget_normalized_to_data(*point)
                if handle is not None:
                    xs = sorted((standing[0], standing[2]))
                    ys = sorted((standing[1], standing[3]))
                    if handle == "C":
                        binding.span_rect = (xs[0], ys[0], xs[1], ys[1])
                        binding.span_move_grab = (
                            pressed[0] - xs[0], pressed[1] - ys[0])
                    elif handle in ("N", "S"):
                        # A y-edge resize never changes the x span.
                        fixed_y = ys[1] if handle == "N" else ys[0]
                        binding.span_rect = (
                            xs[0], fixed_y, xs[1], pressed[1])
                        binding.span_anchor = xs[0]
                        binding.span_resize_lock = "y"
                    else:
                        anchor_x = xs[1] if "W" in handle else xs[0]
                        if handle in ("W", "E"):
                            binding.span_rect = (
                                anchor_x, ys[0], pressed[0], ys[1])
                            binding.span_resize_lock = "x"
                        else:
                            anchor_y = (
                                ys[1] if handle in ("NW", "NE") else ys[0])
                            binding.span_rect = (
                                anchor_x, anchor_y, pressed[0], pressed[1])
                        binding.span_anchor = anchor_x
                    binding.span_candidate = self._span_data_candidate(
                        binding.span_rect[0], binding.span_rect[2])
                else:
                    binding.span_anchor = pressed[0]
                    binding.span_candidate = None
                    binding.span_rect = (
                        pressed[0], pressed[1], pressed[0], pressed[1])
                self._set_numeric_hover(binding, None)
                self.update()
                event.accept()
                return
            super().mousePressEvent(event)
            return
        image_hit = self._image_target_at(event.localPos(), include_rail=True)
        if image_hit is None:
            super().mousePressEvent(event)
            return
        binding, target, rail_target = image_hit
        if not self._image_interaction_armed(binding):
            super().mousePressEvent(event)
            return
        hits_image = target is not None and target[0].contains(event.pos())
        hits_rail = rail_target is not None and rail_target[0].contains(event.pos())
        if self._image_interaction_is_pending(binding) or self._selector_hold is not None:
            if hits_image or hits_rail:
                event.accept()
            else:
                super().mousePressEvent(event)
            return
        if (
            target is not None
            and rail_target is not None
            and event.button() == QtCore.Qt.LeftButton
            and binding.interaction_callback is not None
            and rail_target[0].contains(event.pos())
        ):
            handle = self._clim_handle_at(event.pos(), rail_target[0], rail_target[4])
            if handle is not None:
                self._selector_hold = self._held_panel_from_target(target)
                binding.clim_drag = handle
                binding.clim_origin_limits = rail_target[4].color_limits
                binding.clim_candidate = rail_target[4].color_limits
                binding.clim_domain = self._color_rail_domain(rail_target[4])
                self._set_hover_sample(binding, None)
                self.update()
                event.accept()
                return
        if not hits_image:
            super().mousePressEvent(event)
            return
        if event.button() == QtCore.Qt.RightButton:
            sample = self._sample_for_target(target, event.localPos())
            if sample is None:
                super().mousePressEvent(event)
                return
            self._set_cross_sample(binding, sample)
            self._set_hover_sample(binding, None)
            self.update()
            event.accept()
            return
        if event.button() == QtCore.Qt.MiddleButton:
            if binding.interaction_callback is None:
                super().mousePressEvent(event)
                return
            self._selector_hold = self._held_panel_from_target(target)
            binding.pan_anchor = QtCore.QPointF(event.localPos())
            binding.pan_origin = self._viewport_for_target(binding, target)
            binding.pan_target_size = (
                max(1, target[0].width()),
                max(1, target[0].height()),
            )
            binding.pan_candidate = binding.pan_origin
            self._set_hover_sample(binding, None)
            self.update()
            event.accept()
            return
        if event.button() != QtCore.Qt.LeftButton:
            super().mousePressEvent(event)
            return
        image_target = target[0]
        point = self._normalized_point(event.localPos(), image_target, clamp=False)
        bounds = binding.draft_bounds or binding.applied_bounds
        handle = (
            None
            if bounds is None
            else self._hit_corner_handle(binding, event.pos(), bounds, image_target)
        )
        binding.drag_prior_draft = binding.draft_bounds
        binding.drag_anchor = (
            point
            if handle is None
            else self._opposite_corner_anchor(binding, bounds, handle)
        )
        self._selector_hold = self._held_panel_from_target(target)
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        numeric_binding = self._active_numeric_binding()
        if (
            numeric_binding is not None
            and numeric_binding.threshold_drag is not None
        ):
            target = self._numeric_target(numeric_binding)
            if target is not None:
                viewport = target.payload.viewport
                point = self._numeric_normalized_point(
                    target,
                    event.localPos(),
                    clamp_to_plot=True,
                )
                moved = viewport.widget_normalized_to_data(*point)
                base = list(
                    numeric_binding.threshold_candidate
                    or target.payload.thresholds
                )
                index = numeric_binding.threshold_drag
                if 0 <= index < len(base) and math.isfinite(moved[0]):
                    base[index] = float(moved[0])
                    candidate = tuple(base)
                    # The reference's DragVLine calls back on EVERY motion.
                    if candidate != numeric_binding.threshold_candidate:
                        numeric_binding.threshold_candidate = candidate
                        self._commit_histogram_thresholds(
                            numeric_binding,
                            candidate,
                            hold=self._selector_hold,
                        )
                self.update()
            event.accept()
            return
        if numeric_binding is not None and (
            numeric_binding.span_anchor is not None
            or numeric_binding.span_move_grab is not None
        ):
            target = self._numeric_target(numeric_binding)
            if target is not None:
                viewport = target.payload.viewport
                point = self._numeric_normalized_point(
                    target,
                    event.localPos(),
                    clamp_to_plot=True,
                )
                moved = viewport.widget_normalized_to_data(*point)
                rect = numeric_binding.span_rect
                grab = numeric_binding.span_move_grab
                if grab is not None and rect is not None:
                    # Centre-handle grab: the whole box follows the pointer.
                    width = rect[2] - rect[0]
                    height = rect[3] - rect[1]
                    x0 = moved[0] - grab[0]
                    y0 = moved[1] - grab[1]
                    numeric_binding.span_rect = (
                        x0, y0, x0 + width, y0 + height)
                    numeric_binding.span_candidate = (
                        self._span_data_candidate(x0, x0 + width))
                elif numeric_binding.span_resize_lock == "y":
                    # A y-edge resize reshapes the box only -- the x span (and
                    # so the candidate fixed at press) never changes.
                    if rect is not None:
                        numeric_binding.span_rect = (
                            rect[0], rect[1], rect[2], moved[1])
                else:
                    numeric_binding.span_candidate = self._span_data_candidate(
                        numeric_binding.span_anchor,
                        moved[0],
                    )
                    if rect is not None:
                        numeric_binding.span_rect = (
                            (rect[0], rect[1], moved[0], rect[3])
                            if numeric_binding.span_resize_lock == "x"
                            else (rect[0], rect[1], moved[0], moved[1])
                        )
                self.update()
            event.accept()
            return
        if (
            numeric_binding is not None
            and numeric_binding.pan_anchor is not None
            and numeric_binding.pan_origin is not None
        ):
            target = self._numeric_target(numeric_binding)
            if target is not None:
                point = self._numeric_normalized_point(target, event.localPos())
                try:
                    numeric_binding.pan_candidate = (
                        numeric_binding.pan_origin.panned_x_limits(
                        numeric_binding.pan_anchor,
                        point[0],
                        start_x_limits=numeric_binding.pan_origin.x_limits,
                        )
                    )
                except ValueError:
                    numeric_binding.pan_candidate = None
                # The reference pans LIVE: every motion lands the new limits
                # immediately (mpl's on_motion -> set_xlim -> draw), so the
                # view follows the pointer instead of jumping at release.
                if numeric_binding.pan_candidate is not None:
                    self._commit_numeric_viewport(
                        numeric_binding,
                        numeric_binding.pan_candidate,
                        hold=self._selector_hold,
                    )
                self.update()
            event.accept()
            return
        image_binding = self._active_image_binding()
        if image_binding is not None and image_binding.clim_drag is not None:
            rail_target = self._clim_rail_target(image_binding)
            if (
                rail_target is not None
                and image_binding.clim_origin_limits is not None
                and image_binding.clim_domain is not None
            ):
                value = self._rail_value(
                    float(event.localPos().y()),
                    image_binding.clim_domain,
                    rail_target[0],
                )
                low, high = image_binding.clim_origin_limits
                if image_binding.clim_drag == "low":
                    low = min(value, math.nextafter(high, -math.inf))
                else:
                    high = max(value, math.nextafter(low, math.inf))
                image_binding.clim_candidate = (low, high)
                # The reference's DragHLine recolors on EVERY motion.
                if self._selector_hold is not None:
                    self._commit_color_limits(
                        image_binding,
                        image_binding.clim_candidate,
                        hold=self._selector_hold,
                    )
                self.update()
            event.accept()
            return
        anchor = None if image_binding is None else image_binding.drag_anchor
        if anchor is not None:
            target = self._selector_target(image_binding)
            if target is None:
                event.accept()
                return
            point = self._normalized_point(event.localPos(), target[0], clamp=True)
            if point[0] == anchor[0] or point[1] == anchor[1]:
                image_binding.draft_bounds = image_binding.drag_prior_draft
            else:
                image_binding.draft_bounds = self._require_selector_viewport(
                    image_binding
                ).snapped_bounds_for_drag(
                    anchor,
                    point,
                )
            self.update()
            event.accept()
            return

        pan_anchor = None if image_binding is None else image_binding.pan_anchor
        pan_origin = None if image_binding is None else image_binding.pan_origin
        pan_size = None if image_binding is None else image_binding.pan_target_size
        if pan_anchor is not None and pan_origin is not None and pan_size is not None:
            delta = (
                float(event.localPos().x() - pan_anchor.x()),
                float(event.localPos().y() - pan_anchor.y()),
            )
            image_binding.pan_candidate = pan_origin.panned_by_pixels(delta, pan_size)
            # The reference pans LIVE: every motion lands the new window (the
            # commit rebases the candidate's revision monotonically).
            if (
                image_binding.pan_candidate is not None
                and self._selector_hold is not None
            ):
                self._commit_viewport(
                    image_binding,
                    image_binding.pan_candidate,
                    hold=self._selector_hold,
                )
            self.update()
            event.accept()
            return

        numeric_target = self._numeric_target_at(event.localPos())
        hovered_numeric = None
        if (
            numeric_target is not None
            and self._numeric_interaction_armed(numeric_target.binding)
            and numeric_target.binding.pending_viewport is None
        ):
            hovered_numeric = numeric_target.binding
            sample = self._numeric_sample_for_target(
                numeric_target,
                event.localPos(),
            )
            hovered_numeric.hover_position = (
                None if sample is None else QtCore.QPointF(event.localPos())
            )
            self._set_numeric_hover(hovered_numeric, sample)
            for binding in self._image_bindings.values():
                self._set_hover_sample(binding, None)
        for binding in self._numeric_bindings.values():
            if binding is not hovered_numeric:
                self._set_numeric_hover(binding, None)
        if hovered_numeric is not None:
            self.update()
            super().mouseMoveEvent(event)
            return
        hovered_image = self._image_target_at(event.localPos())
        if hovered_image is not None:
            binding, target, _rail = hovered_image
            sample = (
                None
                if not self._image_interaction_armed(binding)
                or self._image_interaction_is_pending(binding)
                or target is None
                or not target[0].contains(event.pos())
                else self._sample_for_target(target, event.localPos())
            )
            binding.hover_position = (
                None if sample is None else QtCore.QPointF(event.localPos())
            )
            self._set_hover_sample(binding, sample)
            for other in self._image_bindings.values():
                if other is not binding:
                    self._set_hover_sample(other, None)
            self.update()
        else:
            changed = False
            for binding in self._image_bindings.values():
                changed = changed or binding.hover is not None
                self._set_hover_sample(binding, None)
            if changed:
                self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        numeric_binding = self._active_numeric_binding()
        if (
            numeric_binding is not None
            and numeric_binding.threshold_drag is not None
            and event.button() == QtCore.Qt.LeftButton
        ):
            # Every motion already committed its step; release only ends the
            # drag (the reference's DragVLine on_release re-arms and stops).
            self._cancel_active_gesture(
                clear_image_draft=False,
                clear_numeric_spans=False,
            )
            self.update()
            event.accept()
            return
        if (
            numeric_binding is not None
            and numeric_binding.pan_anchor is not None
            and event.button() == QtCore.Qt.MiddleButton
        ):
            # Every motion already committed its candidate (the reference's
            # live pan); release only ends the gesture.
            self._cancel_active_gesture(
                clear_image_draft=False,
                clear_numeric_spans=False,
            )
            self.update()
            event.accept()
            return
        if (
            numeric_binding is not None
            and (
                numeric_binding.span_anchor is not None
                or numeric_binding.span_move_grab is not None
            )
            and event.button() == QtCore.Qt.LeftButton
        ):
            candidate = numeric_binding.span_candidate
            hold = self._selector_hold
            numeric_binding.span_anchor = None
            numeric_binding.span_move_grab = None
            numeric_binding.span_resize_lock = None
            if candidate is None:
                # A degenerate click clears the standing box + label, exactly
                # like the reference's empty-extents onselect.
                numeric_binding.span_rect = None
            try:
                if hold is not None:
                    origin = self._numeric_interaction_origin(
                        numeric_binding,
                        hold=hold,
                    )
                    # PULSE speaks the CURVE intent vocabulary (x-only surface).
                    gesture: _NumericIntent = (
                        HistogramRangeGesture(origin, candidate)
                        if numeric_binding.kind == "histogram"
                        else CurveRangeGesture(origin, candidate)
                    )
                    numeric_binding.callback(gesture)
            except BaseException as error:
                if numeric_binding.fault is None:
                    numeric_binding.fault = detached_render_fault(error)
                numeric_binding.binding_enabled = False
            finally:
                self._cancel_active_gesture(
                    clear_image_draft=False,
                    clear_numeric_spans=True,
                )
                self.update()
            event.accept()
            return
        image_binding = self._active_image_binding()
        if (
            image_binding is not None
            and image_binding.clim_drag is not None
            and event.button() == QtCore.Qt.LeftButton
        ):
            # Every motion already committed its candidate (the reference's
            # DragHLine recolors live); release only ends the gesture.
            self._cancel_image_gesture(image_binding, clear_draft=False)
            self.update()
            event.accept()
            return
        if (
            image_binding is not None
            and image_binding.pan_anchor is not None
            and event.button() == QtCore.Qt.MiddleButton
        ):
            # Live pan committed per motion; release only ends the gesture.
            self._cancel_image_gesture(image_binding, clear_draft=False)
            self.update()
            event.accept()
            return
        anchor = None if image_binding is None else image_binding.drag_anchor
        if anchor is None or event.button() != QtCore.Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        target = self._selector_target(image_binding)
        completed_bounds: NormalizedRectangle | None = None
        if target is not None:
            point = self._normalized_point(event.localPos(), target[0], clamp=True)
            if point[0] == anchor[0] or point[1] == anchor[1]:
                image_binding.draft_bounds = image_binding.drag_prior_draft
            else:
                completed_bounds = self._require_selector_viewport(
                    image_binding
                ).snapped_bounds_for_drag(
                    anchor,
                    point,
                )
                image_binding.draft_bounds = completed_bounds
        hold = self._selector_hold
        bounds = completed_bounds
        callback = image_binding.selection_callback
        delivered = False
        # Geometry is complete before the consumer callback runs, but the
        # synchronous held origin remains alive until that callback returns.
        # A re-entrant PREPARING/disable transition must therefore preserve
        # this completed draft rather than classify it as a partial drag.
        image_binding.drag_anchor = None
        try:
            if bounds is not None and hold is not None and callback is not None:
                gesture = RectangleGesture(
                    panel_id=hold.panel_id,
                    board_id=hold.board_id,
                    layout_generation=hold.layout_generation,
                    sequence=hold.sequence,
                    source_identity=hold.source_identity,
                    normalized_bounds=bounds,
                    viewport_revision=self._require_selector_viewport(
                        image_binding
                    ).viewport_revision,
                )
                callback(gesture)
                delivered = True
        except BaseException as error:
            if image_binding.fault is None:
                image_binding.fault = detached_render_fault(error)
            image_binding.binding_enabled = False
        finally:
            self._cancel_image_gesture(
                image_binding,
                clear_draft=(bounds is not None and not delivered)
            )
            self.update()
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if not self._selector_enabled:
            super().wheelEvent(event)
            return
        numeric_target = self._numeric_target_at(event.posF())
        if (
            numeric_target is not None
            and self._numeric_interaction_armed(numeric_target.binding)
        ):
            binding = numeric_target.binding
            if (
                binding.pending_viewport is not None
                or self._selector_hold is not None
            ):
                event.accept()
                return
            delta = event.angleDelta().y()
            if delta == 0:
                super().wheelEvent(event)
                return
            point = self._numeric_normalized_point(numeric_target, event.posF())
            viewport = numeric_target.payload.viewport
            anchor_x = viewport.widget_normalized_to_data(*point)[0]
            factor = 1.0 / 1.1 if delta < 0 else 1.1
            try:
                candidate = viewport.zoomed_x_limits(anchor_x, factor)
            except ValueError:
                candidate = None
            if candidate is not None:
                self._commit_numeric_viewport(binding, candidate)
            self._set_numeric_hover(binding, None)
            self.update()
            event.accept()
            return
        image_hit = self._image_target_at(event.posF(), include_rail=True)
        if image_hit is None:
            super().wheelEvent(event)
            return
        binding, target, rail_target = image_hit
        if (
            not self._image_interaction_armed(binding)
            or binding.interaction_callback is None
        ):
            super().wheelEvent(event)
            return
        position = event.pos()
        hits_image = target is not None and target[0].contains(position)
        hits_rail = rail_target is not None and rail_target[0].contains(position)
        if self._image_interaction_is_pending(binding) or self._selector_hold is not None:
            if hits_image or hits_rail:
                event.accept()
            else:
                super().wheelEvent(event)
            return
        if not hits_image:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        point = self._normalized_point(event.posF(), target[0], clamp=False)
        # Preserve the established lab convention: wheel DOWN zooms in and
        # wheel UP zooms out.
        scale = 1.0 / 1.1 if delta < 0 else 1.1
        candidate = self._viewport_for_target(binding, target).centered_zoom(
            point,
            scale,
        )
        self._commit_viewport(binding, candidate)
        self._set_hover_sample(binding, None)
        self.update()
        event.accept()

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            panel_id = self._painted_image_panel_id_at(event.localPos())
            if panel_id is not None:
                binding = self._image_bindings.get(panel_id)
                if (
                    binding is not None
                    and self._selector_hold is not None
                    and self._selector_hold.panel_id == panel_id
                ):
                    # Qt sends a press before the double-click event.  Release
                    # only that incomplete gesture; the already-authored area,
                    # color limits, cross and viewport remain untouched.
                    self._cancel_image_gesture(binding, clear_draft=False)
                self.imagePanelLeftDoubleClicked.emit(panel_id)
                self.update()
                event.accept()
                return
        if not self._selector_enabled:
            super().mouseDoubleClickEvent(event)
            return
        numeric_target = self._numeric_target_at(event.localPos())
        if (
            numeric_target is not None
            and self._numeric_interaction_armed(numeric_target.binding)
        ):
            binding = numeric_target.binding
            if event.button() == QtCore.Qt.MiddleButton:
                # Qt delivers a press BEFORE the double-click, and that press
                # already began a middle pan (hold + possibly a live pan
                # commit).  The reference's double-middle always lands -- so
                # release that incomplete gesture and act, instead of letting
                # the hold/pending gate swallow the double-click.
                if binding.pan_anchor is not None:
                    self._cancel_numeric_gesture(binding, clear_span=False)
                viewport = numeric_target.payload.viewport
                self._commit_numeric_viewport(
                    binding,
                    viewport.home_x_limits
                    if binding.applied_span is None
                    else binding.applied_span,
                )
                self._set_numeric_hover(binding, None)
                self.update()
                event.accept()
                return
            if (
                binding.pending_viewport is not None
                or self._selector_hold is not None
            ):
                event.accept()
                return
            if event.button() == QtCore.Qt.RightButton:
                binding.cross = None
                self._set_numeric_hover(binding, None)
                self.update()
                event.accept()
                return
            super().mouseDoubleClickEvent(event)
            return
        image_hit = self._image_target_at(event.localPos(), include_rail=True)
        if image_hit is None:
            super().mouseDoubleClickEvent(event)
            return
        binding, target, rail_target = image_hit
        if not self._image_interaction_armed(binding):
            super().mouseDoubleClickEvent(event)
            return
        hits_image = target is not None and target[0].contains(event.pos())
        hits_rail = rail_target is not None and rail_target[0].contains(event.pos())
        if (
            event.button() == QtCore.Qt.MiddleButton
            and binding.interaction_callback is not None
            and hits_image
        ):
            # Qt delivers a press BEFORE the double-click and that press
            # already began a middle pan -- release the incomplete gesture so
            # the double-middle always lands (zoom-to-area or home).
            if binding.pan_anchor is not None:
                self._cancel_image_gesture(binding, clear_draft=False)
            viewport = self._viewport_for_target(binding, target)
            area = binding.draft_bounds or binding.applied_bounds
            candidate = viewport.with_visible_bounds(
                (0.0, 0.0, 1.0, 1.0) if area is None else area
            )
            self._commit_viewport(binding, candidate)
            self._set_hover_sample(binding, None)
            self.update()
            event.accept()
            return
        if self._image_interaction_is_pending(binding) or self._selector_hold is not None:
            if hits_image or hits_rail:
                event.accept()
            else:
                super().mouseDoubleClickEvent(event)
            return
        if not hits_image:
            super().mouseDoubleClickEvent(event)
            return
        if event.button() == QtCore.Qt.RightButton:
            self._set_cross_sample(binding, None)
            self._set_hover_sample(binding, None)
            self.update()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        for binding in self._image_bindings.values():
            self._set_hover_sample(binding, None)
        for binding in self._numeric_bindings.values():
            self._set_numeric_hover(binding, None)
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape and self._selector_hold is not None:
            self._cancel_active_gesture(
                clear_image_draft=True,
                clear_numeric_spans=True,
            )
            self.update()
            event.accept()
            return
        if event.key() == QtCore.Qt.Key_Escape and self._selector_enabled:
            # The reference's escape also clears a STANDING selection box
            # (RectangleSelector's 'clear' key hides the artists silently --
            # no onselect fires and the applied range is left as-is).
            cleared = False
            for binding in self._numeric_bindings.values():
                if binding.span_rect is not None:
                    binding.span_rect = None
                    cleared = True
            if cleared:
                self.update()
                event.accept()
                return
        super().keyPressEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        if self._selector_hold is not None:
            self._cancel_active_gesture(
                clear_image_draft=True,
                clear_numeric_spans=True,
            )
        for binding in self._image_bindings.values():
            self._set_hover_sample(binding, None)
        for binding in self._numeric_bindings.values():
            self._set_numeric_hover(binding, None)
        super().resizeEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._reset_all_image_bindings()
        self._reset_all_numeric_bindings()
        self._front = None
        self._active_layout_identity = None
        self._staged_layout = None
        self._closed = True
        super().closeEvent(event)

    def event(self, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.DeferredDelete:
            self._reset_all_image_bindings()
            self._reset_all_numeric_bindings()
            self._front = None
            self._active_layout_identity = None
            self._staged_layout = None
            self._closed = True
        elif event.type() in (
            QtCore.QEvent.Hide,
            QtCore.QEvent.WindowDeactivate,
            QtCore.QEvent.UngrabMouse,
        ):
            changed = getattr(self, "_selector_hold", None) is not None
            if changed:
                self._cancel_active_gesture(
                    clear_image_draft=True,
                    clear_numeric_spans=True,
                )
            for binding in getattr(self, "_image_bindings", {}).values():
                if binding.hover is not None:
                    self._set_hover_sample(binding, None)
                    changed = True
            for binding in getattr(self, "_numeric_bindings", {}).values():
                if binding.hover is not None:
                    self._set_numeric_hover(binding, None)
                    changed = True
            if changed:
                self.update()
        return super().event(event)

    def _image_binding(
        self,
        panel_id: str | None = None,
    ) -> _ImagePanelBinding | None:
        if panel_id is not None:
            panel_id = canonical_text(panel_id, "image panel_id")
            return self._image_bindings.get(panel_id)
        matches = tuple(self._image_bindings.values())
        if len(matches) > 1:
            raise ValueError("multiple image panels are bound; panel_id is required")
        return None if not matches else matches[0]

    def _require_image_binding(
        self,
        panel_id: str | None = None,
    ) -> _ImagePanelBinding:
        binding = self._image_binding(panel_id)
        if binding is None:
            raise RuntimeError("image panel is not bound")
        return binding

    @staticmethod
    def _require_selector_viewport(
        binding: _ImagePanelBinding,
    ) -> ImageViewportTransform:
        return binding.viewport

    def _validate_selector_binding(
        self,
        panel_id: str,
        viewport: ImageViewportTransform,
        frame,
        *,
        panel_ids: tuple[str, ...] | None = None,
    ) -> None:
        configured_ids = self._panel_ids if panel_ids is None else panel_ids
        index = configured_ids.index(panel_id)
        raster = frame.panels[index].raster
        expected_height, expected_width = viewport.raster_shape
        if raster.width != expected_width or raster.height != expected_height:
            raise ValueError(
                "selector viewport axes do not match the selected panel raster geometry"
            )
        if frame.panels[index].panel_id != panel_id:
            raise ValueError("selector panel identity changed")
        payload = _image_payload(frame.panels[index])
        if payload is not None and payload.viewport != viewport:
            raise ValueError(
                "selector viewport differs from the exact image payload viewport"
            )

    def _viewport_for_presented_panel(
        self,
        binding: _ImagePanelBinding,
        frame: BoardFrame,
        *,
        panel_ids: tuple[str, ...],
    ) -> ImageViewportTransform:
        panel_id = binding.panel_id
        current = binding.viewport
        index = panel_ids.index(panel_id)
        panel = frame.panels[index]
        payload = _image_payload(panel)
        if payload is None:
            return current
        candidate = payload.viewport
        previous = self._front
        structurally_new = previous is None or panel_id not in self._panel_ids
        if not structurally_new and previous is not None:
            old_panel = previous[0].panels[self._panel_ids.index(panel_id)]
            structurally_new = self._panel_semantics_changed(old_panel, panel)
        if structurally_new:
            return candidate
        if candidate.axes != current.axes:
            raise ValueError("image viewport axes changed without panel structure change")
        if candidate.viewport_revision < current.viewport_revision:
            raise ValueError("stale image viewport revision cannot replace the visible front")
        pending = binding.pending_viewport
        if (
            pending is not None
            and candidate.viewport_revision == pending.viewport_revision
            and candidate != pending
        ):
            raise ValueError("pending image viewport revision returned conflicting bounds")
        if (
            candidate.viewport_revision == current.viewport_revision
            and candidate != current
        ):
            raise ValueError("one image viewport revision describes conflicting bounds")
        return candidate

    def _numeric_binding_for_kind(
        self,
        kind: _NumericKind,
        *,
        panel_id: str | None = None,
    ) -> _NumericPanelBinding | None:
        if panel_id is not None:
            panel_id = canonical_text(panel_id, f"{kind} panel_id")
            binding = self._numeric_bindings.get(panel_id)
            return binding if binding is not None and binding.kind == kind else None
        matches = tuple(
            binding
            for binding in self._numeric_bindings.values()
            if binding.kind == kind
        )
        if len(matches) > 1:
            raise ValueError(f"multiple {kind} panels are bound; panel_id is required")
        return None if not matches else matches[0]

    def _bind_numeric_interaction(
        self,
        kind: _NumericKind,
        panel_id: str,
        callback: Callable[[_NumericIntent], object],
        *,
        enabled: bool,
    ) -> None:
        self._require_owner()
        panel_id = canonical_text(panel_id, f"{kind} panel_id")
        if panel_id not in self._panel_ids:
            raise ValueError(f"{kind} panel_id is absent from this board")
        if not callable(callback):
            raise TypeError(f"{kind} callback must be callable")
        if not isinstance(enabled, bool):
            raise TypeError("selector enabled must be bool")
        has_other_family = (
            bool(self._image_bindings)
            or any(value.panel_id != panel_id for value in self._numeric_bindings.values())
        )
        if has_other_family and enabled != self._selector_enabled:
            raise ValueError(
                "a second selector family must match the board-wide enabled state; "
                "call set_selectors_enabled explicitly"
            )
        viewport = None
        if self._front is not None:
            panel = self._front[0].panels[self._panel_ids.index(panel_id)]
            payload = _numeric_payload(panel, kind)
            if payload is None:
                raise ValueError(
                    f"{kind} interaction requires exact {kind.title()}PanelPayload"
                )
            if _panel_presentation(panel).panel_revision != payload.viewport.display_revision:
                raise ValueError(
                    f"{kind} payload viewport revision differs from its presentation"
                )
            viewport = payload.viewport
        if panel_id in self._numeric_bindings:
            self._reset_numeric_binding(panel_id)
        self._numeric_bindings[panel_id] = _NumericPanelBinding(
            kind,
            panel_id,
            callback,
            viewport=viewport,
            revision_floor=(
                0 if viewport is None else viewport.display_revision
            ),
            interaction_ready=enabled,
        )
        if not has_other_family:
            self._selector_enabled = enabled
        self.update()

    def _numeric_viewport_for_presented_panel(
        self,
        binding: _NumericPanelBinding,
        frame: BoardFrame,
        *,
        panel_ids: tuple[str, ...],
    ) -> _NumericViewport:
        panel = frame.panels[panel_ids.index(binding.panel_id)]
        payload = _numeric_payload(panel, binding.kind)
        if payload is None:
            raise ValueError(
                f"{binding.kind} interaction requires its exact typed payload"
            )
        candidate = payload.viewport
        if _panel_presentation(panel).panel_revision != candidate.display_revision:
            raise ValueError(
                f"{binding.kind} viewport revision differs from its presentation"
            )
        current = binding.viewport
        previous = self._front
        structurally_new = previous is None or binding.panel_id not in self._panel_ids
        if not structurally_new and previous is not None:
            old_panel = previous[0].panels[
                self._panel_ids.index(binding.panel_id)
            ]
            structurally_new = self._panel_semantics_changed(old_panel, panel)
        if current is None or structurally_new:
            return candidate
        if type(candidate) is not type(current):
            raise ValueError("numeric viewport type changed without panel structure change")
        if (
            isinstance(candidate, NumericViewportTransform)
            and isinstance(current, NumericViewportTransform)
            and candidate.x_axis != current.x_axis
        ):
            raise ValueError("curve x axis changed without panel structure change")
        if candidate.display_revision < current.display_revision:
            raise ValueError(
                f"stale {binding.kind} display revision cannot replace the visible front"
            )
        pending = binding.pending_viewport
        if (
            pending is not None
            and candidate.display_revision == pending.display_revision
            and candidate.x_limits != pending.x_limits
        ):
            raise ValueError(
                f"pending {binding.kind} viewport returned conflicting x bounds"
            )
        if (
            isinstance(candidate, HistogramViewportTransform)
            and isinstance(pending, HistogramViewportTransform)
            and candidate.display_revision == pending.display_revision
            and (
                candidate.count_scale is not pending.count_scale
                or candidate.relim_mode is not pending.relim_mode
                or candidate.x_limits_are_auto != pending.x_limits_are_auto
                or candidate.bin_count != pending.bin_count
                or (
                    candidate.relim_mode is RelimMode.FIXED
                    and candidate.count_limits != pending.count_limits
                )
            )
        ):
            raise ValueError(
                "pending histogram viewport returned conflicting authored state"
            )
        if candidate.display_revision == current.display_revision:
            if isinstance(candidate, NumericViewportTransform) and (
                candidate.x_limits != current.x_limits
                or candidate.home_x_limits != current.home_x_limits
            ):
                raise ValueError(
                    "one curve display revision describes conflicting x bounds"
                )
            if (
                isinstance(candidate, HistogramViewportTransform)
                and isinstance(current, HistogramViewportTransform)
                and (
                    candidate.count_scale is not current.count_scale
                    or candidate.relim_mode is not current.relim_mode
                    or candidate.x_limits_are_auto != current.x_limits_are_auto
                    or candidate.bin_count != current.bin_count
                    or (
                        not candidate.x_limits_are_auto
                        and candidate.x_limits != current.x_limits
                    )
                    or (
                        candidate.relim_mode is RelimMode.FIXED
                        and candidate.count_limits != current.count_limits
                    )
                )
            ):
                raise ValueError(
                    "one histogram display revision describes conflicting authored state"
                )
            # Histogram home/x/count limits are data-derived in AUTO modes and
            # may legitimately advance at one authored display revision.
        return candidate

    def _selector_target(self, binding: _ImagePanelBinding | None = None):
        front = self._front
        if binding is None:
            binding = self._image_binding()
        if front is None or binding is None:
            return None
        panel_id = binding.panel_id
        if panel_id not in self._panel_ids:
            return None
        index = self._panel_ids.index(panel_id)
        hold = self._selector_hold
        prepared = (
            hold.prepared
            if hold is not None and hold.panel_id == panel_id
            else front[1][index]
        )
        image = prepared[1]
        bounds = _panel_bounds(
            self.rect(),
            index=index,
            count=len(front[1]),
            columns=self._columns,
        )
        composite = (
            _site_map_payload(hold)
            if hold is not None and hold.panel_id == panel_id
            else _site_map_payload(front[0].panels[index])
        )
        payload = (
            _image_payload(hold)
            if hold is not None and hold.panel_id == panel_id
            else _image_payload(front[0].panels[index])
        )
        target, _source, _rail = _panel_image_geometry(
            bounds,
            image,
            payload,
            site_map_payload=composite,
        )
        return target, front[0], front[0].panels[index], prepared

    def _image_target_at(
        self,
        point: QtCore.QPointF,
        *,
        include_rail: bool = False,
    ):
        for binding in self._image_bindings.values():
            target = self._selector_target(binding)
            if target is None:
                continue
            rail_target = self._clim_rail_target(binding)
            integer_point = point.toPoint()
            if target[0].contains(integer_point) or (
                include_rail
                and rail_target is not None
                and rail_target[0].contains(integer_point)
            ):
                return binding, target, rail_target
        return None

    def _painted_image_panel_id_at(self, point: QtCore.QPointF) -> str | None:
        front = self._front
        if front is None:
            return None
        for index, panel in enumerate(front[0].panels):
            payload = panel.display_payload
            if not isinstance(payload, (ImagePanelPayload, SiteMapPanelPayload)):
                continue
            bounds = _panel_bounds(
                self.rect(),
                index=index,
                count=len(front[0].panels),
                columns=self._columns,
            )
            image = front[1][index][1]
            image_payload = (
                payload.background
                if isinstance(payload, SiteMapPanelPayload)
                else payload
            )
            target, _source, _rail = _panel_image_geometry(
                bounds,
                image,
                image_payload,
                site_map_payload=(
                    payload if isinstance(payload, SiteMapPanelPayload) else None
                ),
            )
            if target.contains(point.toPoint()):
                return panel.panel_id
        return None

    def _numeric_target(
        self,
        binding: _NumericPanelBinding,
    ) -> _NumericTarget | None:
        front = self._front
        panel_id = binding.panel_id
        if front is None or panel_id is None or panel_id not in self._panel_ids:
            return None
        index = self._panel_ids.index(panel_id)
        hold = self._selector_hold
        prepared = (
            hold.prepared
            if hold is not None and hold.panel_id == panel_id
            else front[1][index]
        )
        panel = front[0].panels[index]
        payload = (
            _numeric_payload(hold, binding.kind)
            if hold is not None and hold.panel_id == panel_id
            else _numeric_payload(panel, binding.kind)
        )
        if payload is None:
            return None
        bounds = _panel_bounds(
            self.rect(),
            index=index,
            count=len(front[1]),
            columns=self._columns,
        )
        plot = _numeric_plot_geometry(bounds, payload.viewport)
        return _NumericTarget(plot, front[0], panel, prepared, payload, bounds, binding)

    def _numeric_target_at(self, point: QtCore.QPointF) -> _NumericTarget | None:
        for binding in self._numeric_bindings.values():
            target = self._numeric_target(binding)
            if target is not None and target.plot.contains(point):
                return target
        return None

    @staticmethod
    def _threshold_line_hit(
        target: _NumericTarget,
        pos: QtCore.QPointF,
    ) -> int | None:
        """Index of the authored threshold line a left press grabs, or None.

        The reference's DragVLine tolerance verbatim: within 2% of the
        current x span of the line's position (measured here as 2% of the
        plot's pixel width, the same quantity in widget space)."""

        payload = target.payload
        if not isinstance(payload, HistogramPanelPayload):
            return None
        thresholds = payload.thresholds
        if not thresholds:
            return None
        viewport = payload.viewport
        bounds = target.bounds
        counts_mid = 0.5 * (
            viewport.count_limits[0] + viewport.count_limits[1])
        tolerance = 0.02 * target.plot.width()
        pressed_x = float(pos.x())
        best_index: int | None = None
        best_distance: float | None = None
        for index, threshold in enumerate(thresholds):
            normalized = viewport.data_to_widget_normalized(
                float(threshold), counts_mid)
            line_x = bounds.x() + normalized[0] * bounds.width()
            distance = abs(pressed_x - line_x)
            if distance <= tolerance and (
                best_distance is None or distance < best_distance
            ):
                best_index, best_distance = index, distance
        return best_index

    @staticmethod
    def _span_rect_widget_extents(
        target: _NumericTarget,
        rect: tuple[float, float, float, float],
    ) -> tuple[list[float], list[float]]:
        """A DATA-coordinate span box as sorted widget-pixel x/y extents,
        mapped through the panel's CURRENT viewport."""

        viewport = target.payload.viewport
        bounds = target.bounds
        first = viewport.data_to_widget_normalized(rect[0], rect[1])
        second = viewport.data_to_widget_normalized(rect[2], rect[3])
        xs = sorted((bounds.x() + first[0] * bounds.width(),
                     bounds.x() + second[0] * bounds.width()))
        ys = sorted((bounds.y() + first[1] * bounds.height(),
                     bounds.y() + second[1] * bounds.height()))
        return xs, ys

    @staticmethod
    def _span_data_candidate(
        first: float | None,
        second: float | None,
    ) -> tuple[float, float] | None:
        """Sorted DATA-coordinate x span; None for a degenerate pull (the
        reference's empty-extents onselect)."""

        if first is None or second is None or first == second:
            return None
        low, high = sorted((float(first), float(second)))
        if not (math.isfinite(low) and math.isfinite(high)):
            return None
        return (low, high)

    @staticmethod
    def _span_handle_hit(
        xs: list[float],
        ys: list[float],
        pos: QtCore.QPointF,
    ) -> str | None:
        """Which handle of a STANDING selection box a press grabs.

        The reference's RectangleSelector rules verbatim: the box centre
        within twice the grab range moves the whole box (``"C"``, priority);
        otherwise the nearest of the eight corner/edge handles within the
        grab range resizes; anything else returns None (press elsewhere
        discards the box and pulls a fresh one).  ``xs``/``ys`` are the box's
        sorted widget-pixel extents."""

        grab = 10.0
        centre_x = (xs[0] + xs[1]) / 2.0
        centre_y = (ys[0] + ys[1]) / 2.0
        px, py = float(pos.x()), float(pos.y())
        if math.hypot(px - centre_x, py - centre_y) < 2.0 * grab:
            return "C"
        handles = (
            ("NW", xs[0], ys[0]), ("N", centre_x, ys[0]), ("NE", xs[1], ys[0]),
            ("W", xs[0], centre_y), ("E", xs[1], centre_y),
            ("SW", xs[0], ys[1]), ("S", centre_x, ys[1]), ("SE", xs[1], ys[1]),
        )
        name, best = None, None
        for key, hx, hy in handles:
            distance = math.hypot(px - hx, py - hy)
            if best is None or distance < best:
                name, best = key, distance
        return name if best is not None and best <= grab else None

    @staticmethod
    def _numeric_normalized_point(
        target: _NumericTarget,
        point: QtCore.QPointF,
        *,
        clamp_to_plot: bool = False,
    ) -> tuple[float, float]:
        bounds = target.bounds
        x = (float(point.x()) - bounds.x()) / max(1, bounds.width())
        y = (float(point.y()) - bounds.y()) / max(1, bounds.height())
        if clamp_to_plot:
            left, top, right, bottom = target.payload.viewport.plot_bounds
            x = min(right, max(left, x))
            y = min(bottom, max(top, y))
        return x, y

    def _numeric_sample_for_target(
        self,
        target: _NumericTarget,
        point: QtCore.QPointF,
    ) -> _CurveSample | _HistogramBinSample | None:
        if isinstance(target.payload, HistogramPanelPayload):
            return self._histogram_sample_for_target(target, point)
        if isinstance(target.payload, PulsePanelPayload):
            # A pulse timeline has rows, not sampled series -- there is nothing
            # to snap a hover to.  Cross/area/zoom stay live via the viewport.
            return None
        return self._curve_sample_for_numeric_target(target, point)

    def _curve_sample_for_numeric_target(
        self,
        target: _NumericTarget,
        point: QtCore.QPointF,
    ) -> _CurveSample | None:
        payload = target.payload
        assert isinstance(payload, CurvePanelPayload)
        viewport = payload.viewport
        bounds = target.bounds
        best: tuple[float, int, int, _CurveSample] | None = None
        coordinates = np.asarray(
            payload.series[0].data.x_axis.coordinates,
            dtype=np.float64,
        )
        x_low, x_high = viewport.x_limits
        y_low, y_high = viewport.y_limits
        left, top, right, bottom = viewport.plot_bounds
        x_widget = bounds.x() + (
            left
            + (coordinates - x_low) / (x_high - x_low) * (right - left)
        ) * bounds.width()
        for series_index, (series, label) in enumerate(
            zip(payload.series, payload.series_labels)
        ):
            curve = series.data
            values = np.asarray(curve.values, dtype=np.float64)
            valid = np.asarray(curve.validity, dtype=bool)
            visible = (
                valid
                & np.isfinite(values)
                & (coordinates >= x_low)
                & (coordinates <= x_high)
                & (values >= y_low)
                & (values <= y_high)
            )
            sample_indices = np.flatnonzero(visible)
            if not sample_indices.size:
                continue
            visible_values = values[sample_indices]
            y_widget = bounds.y() + (
                top
                + (y_high - visible_values) / (y_high - y_low) * (bottom - top)
            ) * bounds.height()
            distances = (
                (x_widget[sample_indices] - point.x()) ** 2
                + (y_widget - point.y()) ** 2
            )
            local_index = int(np.argmin(distances))
            sample_index = int(sample_indices[local_index])
            sample = _CurveSample(
                label,
                float(coordinates[sample_index]),
                float(values[sample_index]),
            )
            candidate = (
                float(distances[local_index]),
                series_index,
                sample_index,
                sample,
            )
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        return None if best is None else best[3]

    def _histogram_sample_for_target(
        self,
        target: _NumericTarget,
        point: QtCore.QPointF,
    ) -> _HistogramBinSample | None:
        payload = target.payload
        assert isinstance(payload, HistogramPanelPayload)
        viewport = payload.viewport
        normalized = self._numeric_normalized_point(target, point)
        x_value, _count_value = viewport.widget_normalized_to_data(*normalized)
        edges = np.asarray(payload.bin_edges, dtype=np.float64)
        index = int(np.searchsorted(edges, x_value, side="right") - 1)
        if x_value == float(edges[-1]):
            index = len(edges) - 2
        if not 0 <= index < len(edges) - 1:
            return None
        best: tuple[float, int, _HistogramBinSample] | None = None
        for series_index, (counts, label) in enumerate(
            zip(payload.bin_counts, payload.series_labels, strict=True)
        ):
            count = int(counts[index])
            if viewport.count_scale.value == "log" and count <= 0:
                continue
            if not viewport.count_limits[0] <= count <= viewport.count_limits[1]:
                continue
            sample = _HistogramBinSample(
                label,
                float(edges[index]),
                float(edges[index + 1]),
                count,
                index == len(edges) - 2,
            )
            widget = viewport.data_to_widget_normalized(sample.x, sample.y)
            widget_y = target.bounds.y() + widget[1] * target.bounds.height()
            candidate = (abs(widget_y - point.y()), series_index, sample)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        return None if best is None else best[2]

    def _clim_rail_target(self, binding: _ImagePanelBinding):
        front = self._front
        panel_id = binding.panel_id
        if front is None or panel_id not in self._panel_ids:
            return None
        index = self._panel_ids.index(panel_id)
        hold = self._selector_hold
        prepared = (
            hold.prepared
            if hold is not None and hold.panel_id == panel_id
            else front[1][index]
        )
        panel = front[0].panels[index]
        payload = (
            _image_payload(hold)
            if hold is not None and hold.panel_id == panel_id
            else _image_payload(panel)
        )
        composite = (
            _site_map_payload(hold)
            if hold is not None and hold.panel_id == panel_id
            else _site_map_payload(panel)
        )
        if payload is None:
            return None
        bounds = _panel_bounds(
            self.rect(),
            index=index,
            count=len(front[1]),
            columns=self._columns,
        )
        _target, _source, rail = _panel_image_geometry(
            bounds,
            prepared[1],
            payload,
            site_map_payload=composite,
        )
        if rail is None:
            return None
        return rail, front[0], panel, prepared, payload

    def _viewport_for_target(
        self,
        binding: _ImagePanelBinding,
        target,
    ) -> ImageViewportTransform:
        hold = self._selector_hold
        if hold is not None and hold.panel_id == target[2].panel_id:
            payload = _image_payload(hold)
            if payload is not None:
                return payload.viewport
        payload = _image_payload(target[2])
        if payload is not None:
            return payload.viewport
        return self._require_selector_viewport(binding)

    def _sample_for_target(
        self,
        target,
        point: QtCore.QPointF,
    ) -> _ImageSample | None:
        image_target, frame, panel = target[0], target[1], target[2]
        hold = self._selector_hold
        if hold is not None and hold.panel_id == panel.panel_id:
            payload = _image_payload(hold)
            presentation = hold.presentation
        else:
            payload = _image_payload(panel)
            presentation = _panel_presentation(panel)
        if payload is None:
            return None
        viewport = payload.viewport
        if presentation.panel_revision != viewport.viewport_revision:
            return None
        normalized = self._normalized_point(point, image_target, clamp=False)
        y_index, x_index = viewport.sample_indices_for_visible_point(normalized)
        value = payload.image.values[y_index, x_index]
        if hasattr(value, "item"):
            value = value.item()
        valid = payload.image.validity[y_index, x_index]
        if hasattr(valid, "item"):
            valid = valid.item()
        try:
            finite_value = math.isfinite(value)
        except TypeError:
            finite_value = False
        x_coordinate = viewport.x_axis.coordinate_at(x_index)
        y_coordinate = viewport.y_axis.coordinate_at(y_index)
        return _ImageSample(
            x_index=x_index,
            y_index=y_index,
            x_coordinate=x_coordinate,
            y_coordinate=y_coordinate,
            value=value,
            valid=bool(valid) and finite_value,
        )

    def _commit_viewport(
        self,
        binding: _ImagePanelBinding,
        candidate: ImageViewportTransform,
        *,
        hold: _HeldPanelFront | None = None,
    ) -> bool:
        current = self._require_selector_viewport(binding)
        if candidate == current:
            return False
        if candidate.axes != current.axes:
            raise ValueError("viewport commit cannot change image axes")
        # Failed/pending motion revisions stay consumed and never alias.
        base_revision = current.viewport_revision
        if binding.viewport is not None:
            base_revision = max(base_revision, binding.viewport.viewport_revision)
        base_revision = max(base_revision, binding.revision_floor)
        if binding.pending_viewport is not None:
            base_revision = max(
                base_revision, binding.pending_viewport.viewport_revision)
        if candidate.viewport_revision <= base_revision:
            candidate = candidate.with_visible_bounds(
                candidate.visible_bounds,
                viewport_revision=base_revision + 1,
            )
        if candidate.viewport_revision <= current.viewport_revision:
            raise ValueError("viewport commit revision must increase")
        front = self._front
        if front is None:
            return False
        if hold is not None and not self._hold_matches_frame(
            hold,
            front[0],
            panel_ids=self._panel_ids,
        ):
            return False
        callback = binding.interaction_callback
        if callback is None:
            return False
        origin = self._interaction_origin(binding, hold=hold)
        command = ImageViewportCommit(origin, candidate)
        binding.revision_floor = max(
            binding.revision_floor,
            candidate.viewport_revision,
        )
        binding.pending_viewport = candidate
        binding.pending_origin = origin
        try:
            callback(command)
        except BaseException as error:
            binding.pending_viewport = None
            binding.pending_origin = None
            if binding.fault is None:
                binding.fault = detached_render_fault(error)
            binding.binding_enabled = False
            self._set_hover_sample(binding, None)
            return False
        return True

    def _commit_histogram_thresholds(
        self,
        binding: _NumericPanelBinding,
        thresholds: tuple[float, ...],
        *,
        hold: _HeldPanelFront | None = None,
    ) -> bool:
        """Author one threshold motion and reserve its display revision."""

        payload = (
            _numeric_payload(hold, binding.kind)
            if hold is not None
            else self._visible_display(
                binding.panel_id,
                _NUMERIC_PAYLOAD_TYPES[binding.kind],
            )[0]
        )
        if payload is None or tuple(thresholds) == tuple(payload.thresholds):
            return False
        front = self._front
        if front is None:
            return False
        if hold is not None and not self._hold_matches_frame(
            hold,
            front[0],
            panel_ids=self._panel_ids,
        ):
            return False
        origin = self._numeric_interaction_origin(binding, hold=hold)
        command = HistogramThresholdCommit(origin, tuple(thresholds))
        expected = payload.viewport.display_revision
        if binding.viewport is not None:
            expected = max(expected, binding.viewport.display_revision)
        expected = max(expected, binding.revision_floor)
        if binding.threshold_pending_revision is not None:
            expected = max(expected, binding.threshold_pending_revision)
        binding.threshold_pending_revision = expected + 1
        binding.revision_floor = binding.threshold_pending_revision
        binding.threshold_pending_origin = origin
        try:
            binding.callback(command)
        except BaseException as error:
            binding.threshold_pending_revision = None
            binding.threshold_pending_origin = None
            if binding.fault is None:
                binding.fault = detached_render_fault(error)
            binding.binding_enabled = False
            self._set_numeric_hover(binding, None)
            return False
        return True

    def _commit_numeric_viewport(
        self,
        binding: _NumericPanelBinding,
        x_limits: tuple[float, float],
        *,
        hold: _HeldPanelFront | None = None,
    ) -> bool:
        payload = (
            _numeric_payload(hold, binding.kind)
            if hold is not None
            else self._visible_display(
                binding.panel_id,
                _NUMERIC_PAYLOAD_TYPES[binding.kind],
            )[0]
        )
        if payload is None or x_limits == payload.viewport.x_limits:
            return False
        assert isinstance(payload, _NUMERIC_PAYLOAD_TYPES[binding.kind])
        # Allocate beyond visible, pending and failed motion revisions.
        base_revision = payload.viewport.display_revision
        if binding.viewport is not None:
            base_revision = max(
                base_revision, binding.viewport.display_revision)
        base_revision = max(base_revision, binding.revision_floor)
        if binding.pending_viewport is not None:
            base_revision = max(
                base_revision, binding.pending_viewport.display_revision)
        front = self._front
        if front is None:
            return False
        if hold is not None and not self._hold_matches_frame(
            hold,
            front[0],
            panel_ids=self._panel_ids,
        ):
            return False
        origin = self._numeric_interaction_origin(binding, hold=hold)
        candidate = replace(
            payload.viewport,
            display_revision=base_revision + 1,
            x_limits=x_limits,
            **(
                {"x_limits_are_auto": False}
                if isinstance(payload, HistogramPanelPayload)
                else {}
            ),
        )
        command: _NumericIntent = (
            HistogramViewportCommit(origin, candidate)
            if binding.kind == "histogram"
            else CurveViewportCommit(origin, candidate)
        )
        binding.revision_floor = candidate.display_revision
        binding.pending_viewport = candidate
        binding.pending_origin = origin
        try:
            binding.callback(command)
        except BaseException as error:
            binding.pending_viewport = None
            binding.pending_origin = None
            if binding.fault is None:
                binding.fault = detached_render_fault(error)
            binding.binding_enabled = False
            self._set_numeric_hover(binding, None)
            return False
        return True

    def _clim_handle_at(
        self,
        point: QtCore.QPoint,
        rail: QtCore.QRect,
        payload: ImagePanelPayload,
    ) -> str | None:
        domain = self._color_rail_domain(payload)
        candidates = (
            (abs(point.y() - self._rail_y(payload.color_limits[0], domain, rail)), "low"),
            (abs(point.y() - self._rail_y(payload.color_limits[1], domain, rail)), "high"),
        )
        distance, handle = min(candidates)
        return handle if distance <= 7.0 else None

    def _commit_color_limits(
        self,
        binding: _ImagePanelBinding,
        limits: tuple[float, float],
        *,
        hold: _HeldPanelFront,
    ) -> bool:
        payload = _image_payload(hold)
        if payload is None or limits == payload.color_limits:
            return False
        # A live clim drag commits once per motion (the reference's DragHLine
        # calls back on every mouse move); a newer candidate REPLACES an
        # in-flight one instead of being refused.
        front = self._front
        if front is None or not self._hold_matches_frame(
            hold,
            front[0],
            panel_ids=self._panel_ids,
        ):
            return False
        callback = binding.interaction_callback
        if callback is None:
            return False
        origin = self._interaction_origin(binding, hold=hold)
        command = ImageColorLimitsCommit(origin, limits)
        binding.pending_color_limits = command.color_limits
        binding.pending_origin = origin
        try:
            callback(command)
        except BaseException as error:
            binding.pending_color_limits = None
            binding.pending_origin = None
            if binding.fault is None:
                binding.fault = detached_render_fault(error)
            binding.binding_enabled = False
            self._set_hover_sample(binding, None)
            return False
        return True

    @staticmethod
    def _image_interaction_is_pending(binding: _ImagePanelBinding) -> bool:
        return (
            binding.pending_viewport is not None
            or binding.pending_color_limits is not None
        )

    def _interaction_origin(
        self,
        binding: _ImagePanelBinding,
        *,
        hold: _HeldPanelFront | None = None,
    ) -> PanelInteractionOrigin:
        return self._require_interaction_origin(
            panel_id=binding.panel_id,
            payload_type=(ImagePanelPayload, SiteMapPanelPayload),
            hold=hold,
            kind="image",
        )

    def _numeric_interaction_origin(
        self,
        binding: _NumericPanelBinding,
        *,
        hold: _HeldPanelFront | None = None,
    ) -> PanelInteractionOrigin:
        return self._require_interaction_origin(
            panel_id=binding.panel_id,
            payload_type=_NUMERIC_PAYLOAD_TYPES[binding.kind],
            hold=hold,
            kind=binding.kind,
        )

    def _require_interaction_origin(
        self,
        *,
        panel_id: str | None,
        payload_type: type | tuple[type, ...],
        hold: _HeldPanelFront | None,
        kind: str,
    ) -> PanelInteractionOrigin:
        if hold is not None and hold is not self._selector_hold:
            raise RuntimeError(f"{kind} interaction hold is no longer painted")
        _payload, origin = self._visible_display(panel_id, payload_type)
        if origin is None:
            raise RuntimeError(f"{kind} interaction origin has no exact payload")
        return origin

    @staticmethod
    def _set_cross_sample(
        binding: _ImagePanelBinding,
        sample: _ImageSample | None,
    ) -> None:
        if sample is binding.cross:
            return
        binding.cross = sample

    @staticmethod
    def _set_hover_sample(
        binding: _ImagePanelBinding,
        sample: _ImageSample | None,
    ) -> None:
        if sample is binding.hover:
            return
        binding.hover = sample
        if sample is None:
            binding.hover_position = None

    def _set_numeric_hover(
        self,
        binding: _NumericPanelBinding,
        sample: _CurveSample | _HistogramBinSample | None,
    ) -> None:
        binding.hover = sample
        if sample is None:
            binding.hover_position = None

    def _active_numeric_binding(self) -> _NumericPanelBinding | None:
        hold = self._selector_hold
        if hold is None:
            return None
        return self._numeric_bindings.get(hold.panel_id)

    def _active_image_binding(self) -> _ImagePanelBinding | None:
        hold = self._selector_hold
        if hold is None:
            return None
        return self._image_bindings.get(hold.panel_id)

    def _held_panel_from_target(self, target) -> _HeldPanelFront:
        frame, panel, prepared = target[1], target[2], target[3]
        return _HeldPanelFront(
            panel_id=panel.panel_id,
            board_id=frame.board_id,
            layout_generation=frame.layout_generation,
            sequence=frame.sequence,
            coherence_group=panel.coherence_group,
            source_identity=panel.source_identity,
            presentation=_panel_presentation(panel),
            raster_geometry=_raster_geometry(panel),
            prepared=prepared,
            display_payload=(
                target[4]
                if len(target) > 4
                else panel.display_payload
            ),
        )

    def _held_panel_from_numeric_target(
        self,
        target: _NumericTarget,
    ) -> _HeldPanelFront:
        return _HeldPanelFront(
            panel_id=target.panel.panel_id,
            board_id=target.frame.board_id,
            layout_generation=target.frame.layout_generation,
            sequence=target.frame.sequence,
            coherence_group=target.panel.coherence_group,
            source_identity=target.panel.source_identity,
            presentation=_panel_presentation(target.panel),
            raster_geometry=_raster_geometry(target.panel),
            prepared=target.prepared,
            display_payload=target.payload,
        )

    @staticmethod
    def _panel_semantics_changed(old: PanelFrame, new: PanelFrame) -> bool:
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
            or old_presentation.selection_revision
            != new_presentation.selection_revision
            or _raster_geometry(old) != _raster_geometry(new)
            or interaction_geometry(old_payload) != interaction_geometry(new_payload)
        )

    def _hold_matches_frame(
        self,
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
        # The hold survives ANY same-identity present -- its own live pan/zoom
        # answers and fresh data frames alike advance the revisions while the
        # gesture keeps running on the frozen press frame (the design's hold
        # semantics; the reference never kills a drag on a redraw).  Only an
        # identity change (panel, document, selection) is a real mismatch.
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

    @staticmethod
    def _paint_site_map_rings(
        painter: QtGui.QPainter,
        payload: SiteMapPanelPayload,
        target: QtCore.QRect,
    ) -> None:
        """Paint calibrated rings over the exact background front in Qt."""

        viewport = payload.background.viewport
        width, height = payload.visible_ring_span
        ring_width = width * target.width()
        ring_height = height * target.height()
        left, top, right, bottom = viewport.visible_bounds
        occupied = payload.site_validity & payload.occupied
        empty = payload.site_validity & ~payload.occupied
        invalid = ~payload.site_validity
        styles = (
            (
                empty,
                SITE_EMPTY_COLOR,
                SITE_EMPTY_ALPHA,
                SITE_EMPTY_LINEWIDTH,
                False,
            ),
            (
                occupied,
                SITE_OCCUPIED_COLOR,
                SITE_OCCUPIED_ALPHA,
                SITE_OCCUPIED_LINEWIDTH,
                False,
            ),
            (
                invalid,
                SITE_INVALID_COLOR,
                SITE_INVALID_ALPHA,
                SITE_INVALID_LINEWIDTH,
                True,
            ),
        )
        painter.save()
        try:
            painter.setClipRect(target)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setBrush(QtCore.Qt.NoBrush)
            for mask, color_name, alpha, linewidth, dashed in styles:
                color = QtGui.QColor(color_name)
                color.setAlphaF(alpha)
                pen = QtGui.QPen(color, linewidth)
                if dashed:
                    pen.setStyle(QtCore.Qt.DashLine)
                painter.setPen(pen)
                for full_x, full_y in payload.full_normalized_centers_xy[mask]:
                    x = (float(full_x) - left) / (right - left)
                    y = (float(full_y) - top) / (bottom - top)
                    painter.drawEllipse(
                        QtCore.QRectF(
                            target.x() + x * target.width() - ring_width / 2.0,
                            target.y() + y * target.height() - ring_height / 2.0,
                            ring_width,
                            ring_height,
                        )
                    )
        finally:
            painter.restore()

    @staticmethod
    def _radial_fit_overlay_geometry(
        payload: ImagePanelPayload,
        target: QtCore.QRect,
    ) -> tuple[QtCore.QPointF, QtCore.QRectF] | None:
        """Project saved radial-fit geometry into one exact image target.

        The centre deliberately remains unbounded.  The painter clip, not a
        coordinate clamp, decides whether an off-view centre or ring is
        visible.  This preserves the physical location when the image is
        zoomed or either declared spatial axis is descending.
        """

        overlay = payload.fit_overlay
        if overlay is None or overlay.status is not FitBatchStatus.CONVERGED:
            return None
        center = overlay.center_xy
        radius = overlay.one_over_e_radius
        if center is None or radius is None:
            raise RuntimeError("converged radial fit overlay lost its geometry")
        visible_center = payload.viewport.unbounded_visible_point_for_coordinate(
            center,
            coordinate_frame=overlay.coordinate_frame,
        )
        visible_diameter = payload.viewport.visible_span_for_coordinate_span(
            (2.0 * radius, 2.0 * radius),
            coordinate_frame=overlay.coordinate_frame,
        )
        center_point = QtCore.QPointF(
            target.left() + visible_center[0] * target.width(),
            target.top() + visible_center[1] * target.height(),
        )
        ring_width = visible_diameter[0] * target.width()
        ring_height = visible_diameter[1] * target.height()
        return center_point, QtCore.QRectF(
            center_point.x() - ring_width / 2.0,
            center_point.y() - ring_height / 2.0,
            ring_width,
            ring_height,
        )

    @staticmethod
    def _radial_fit_caption_status(
        overlay: RadialGaussianImageFitOverlay,
    ) -> str:
        status = (
            "NOT_PRESENT"
            if overlay.status is None
            else overlay.status.value
        )
        return f"{overlay.caption} · {status}"

    @classmethod
    def _paint_radial_fit_overlay(
        cls,
        painter: QtGui.QPainter,
        payload: ImagePanelPayload,
        target: QtCore.QRect,
    ) -> None:
        """Paint one saved fit as a vector dot/ring plus compact status.

        No fitted image is evaluated here.  Sparse and failed cells retain
        their caption/status but cannot acquire geometry by substitution.
        """

        overlay = payload.fit_overlay
        if overlay is None:
            return
        painter.save()
        try:
            painter.setClipRect(target)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            geometry = cls._radial_fit_overlay_geometry(payload, target)
            if geometry is not None:
                center, ring = geometry
                ring_color = QtGui.QColor(ORANGE)
                ring_color.setAlphaF(0.9)
                painter.setPen(QtGui.QPen(ring_color, 1.8))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(ring)
                center_color = QtGui.QColor(ORANGE)
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QBrush(center_color))
                painter.drawEllipse(center, 2.25, 2.25)

            metrics = painter.fontMetrics()
            available_width = max(1, target.width() - 18)
            label = metrics.elidedText(
                cls._radial_fit_caption_status(overlay),
                QtCore.Qt.ElideMiddle,
                available_width,
            )
            label_bounds = metrics.boundingRect(label).adjusted(-5, -2, 5, 2)
            label_bounds.moveTopLeft(target.topLeft() + QtCore.QPoint(4, 4))
            painter.fillRect(label_bounds, QtGui.QColor(0, 0, 0, 190))
            painter.setPen(QtGui.QColor(ORANGE))
            painter.drawText(label_bounds, QtCore.Qt.AlignCenter, label)
        finally:
            painter.restore()

    @staticmethod
    def _color_rail_domain(payload: ImagePanelPayload) -> tuple[float, float]:
        low, high = payload.color_limits
        span = high - low
        padding = max(
            span * 0.08,
            math.ulp(max(1.0, abs(low), abs(high))) * 16.0,
        )
        return low - padding, high + padding

    @staticmethod
    def _rail_y(value: float, domain: tuple[float, float], rail: QtCore.QRect) -> float:
        low, high = domain
        fraction = (value - low) / (high - low)
        return rail.bottom() - min(1.0, max(0.0, fraction)) * rail.height()

    @staticmethod
    def _rail_value(y: float, domain: tuple[float, float], rail: QtCore.QRect) -> float:
        fraction = (rail.bottom() - y) / max(1, rail.height())
        low, high = domain
        return low + min(1.0, max(0.0, fraction)) * (high - low)

    def _paint_color_rail(
        self,
        painter: QtGui.QPainter,
        payload: ImagePanelPayload,
        rail: QtCore.QRect,
        binding: _ImagePanelBinding | None,
    ) -> None:
        painter.save()
        try:
            painter.setClipRect(rail)
            painter.fillRect(rail, QtGui.QColor(12, 12, 12, 230))
            gradient_left = rail.right() - min(9, max(5, rail.width() // 3)) + 1
            denominator = max(1, rail.height() - 1)
            domain = self._color_rail_domain(payload)
            domain_low, domain_high = domain
            for offset in range(rail.height()):
                fraction = 1.0 - offset / denominator
                value = domain_low + fraction * (domain_high - domain_low)
                painter.setPen(
                    QtGui.QColor.fromRgba(
                        self._color_rail_argb(payload, value)
                    )
                )
                y = rail.top() + offset
                painter.drawLine(gradient_left, y, rail.right(), y)

            if any(payload.histogram_counts):
                maximum = max(payload.histogram_counts)
                histogram_width = max(1, gradient_left - rail.left() - 2)
                value_low, value_high = payload.color_limits
                for code, count in enumerate(payload.histogram_counts, start=1):
                    if count == 0:
                        continue
                    value = (
                        value_low
                        if value_high == value_low
                        else value_low + (value_high - value_low) * (code - 1) / 254.0
                    )
                    y = int(round(self._rail_y(value, domain, rail)))
                    width = max(1, int(round(histogram_width * count / maximum)))
                    painter.fillRect(
                        QtCore.QRect(gradient_left - width - 1, y, width, 1),
                        QtGui.QColor(210, 210, 210, 150),
                    )

            if payload.data_range is not None:
                guide_pen = QtGui.QPen(QtGui.QColor(180, 180, 180, 150), 1.0)
                guide_pen.setStyle(QtCore.Qt.DashLine)
                painter.setPen(guide_pen)
                for value in payload.data_range:
                    if not payload.color_limits[0] <= value <= payload.color_limits[1]:
                        continue
                    y = self._rail_y(value, domain, rail)
                    painter.drawLine(
                        QtCore.QPointF(rail.left(), y),
                        QtCore.QPointF(rail.right(), y),
                    )

            limits = (
                binding.clim_candidate
                if self._selector_hold is not None
                and _image_payload(self._selector_hold) is payload
                and binding is not None
                and binding.clim_candidate is not None
                else payload.color_limits
            )
            for value in limits:
                y = self._rail_y(value, domain, rail)
                painter.setPen(QtGui.QPen(QtGui.QColor(ORANGE), 2.0))
                painter.drawLine(
                    QtCore.QPointF(rail.left(), y),
                    QtCore.QPointF(rail.right(), y),
                )
        finally:
            painter.restore()

    @staticmethod
    def _color_rail_argb(payload: ImagePanelPayload, value: float) -> int:
        """Map one physical rail value through the painted image's clim."""

        index = indexed8_code_for_value(value, payload.color_limits)
        return payload.base_palette[index]

    @staticmethod
    def _normalized_point(
        point: QtCore.QPointF,
        target: QtCore.QRect,
        *,
        clamp: bool,
    ) -> tuple[float, float]:
        x = (float(point.x()) - target.x()) / max(1, target.width())
        y = (float(point.y()) - target.y()) / max(1, target.height())
        if clamp:
            return min(1.0, max(0.0, x)), min(1.0, max(0.0, y))
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("pointer lies outside the selected image viewport")
        return x, y

    def _opposite_corner_anchor(
        self,
        binding: _ImagePanelBinding,
        bounds: NormalizedRectangle,
        handle: int,
    ) -> tuple[float, float]:
        viewport = self._require_selector_viewport(binding)
        left, top, right, bottom = viewport.visible_bounds_for_full_bounds(bounds)
        visible_left, visible_top, visible_right, visible_bottom = (
            viewport.visible_bounds
        )
        x_half_cell = 0.5 / (
            viewport.x_axis.size * (visible_right - visible_left)
        )
        y_half_cell = 0.5 / (
            viewport.y_axis.size * (visible_bottom - visible_top)
        )
        return (
            (
                (right - x_half_cell, bottom - y_half_cell),
                (left + x_half_cell, bottom - y_half_cell),
                (right - x_half_cell, top + y_half_cell),
                (left + x_half_cell, top + y_half_cell),
            )[handle]
        )

    @staticmethod
    def _overlay_rect(
        bounds: NormalizedRectangle,
        target: QtCore.QRect,
    ) -> QtCore.QRectF:
        left, top, right, bottom = bounds
        return QtCore.QRectF(
            target.x() + left * target.width(),
            target.y() + top * target.height(),
            (right - left) * target.width(),
            (bottom - top) * target.height(),
        )

    def _hit_corner_handle(
        self,
        binding: _ImagePanelBinding,
        point: QtCore.QPoint,
        bounds: NormalizedRectangle,
        target: QtCore.QRect,
    ) -> int | None:
        try:
            visible_bounds = self._require_selector_viewport(
                binding
            ).visible_bounds_for_full_bounds(
                bounds
            )
        except ValueError:
            return None
        rectangle = self._overlay_rect(visible_bounds, target)
        corners = (
            rectangle.topLeft(),
            rectangle.topRight(),
            rectangle.bottomLeft(),
            rectangle.bottomRight(),
        )
        radius = 7.0
        for index, corner in enumerate(corners):
            if (
                abs(point.x() - corner.x()) <= radius
                and abs(point.y() - corner.y()) <= radius
            ):
                return index
        return None

    def _paint_selector_overlays(self, painter: QtGui.QPainter) -> None:
        for binding in self._image_bindings.values():
            self._paint_image_binding_overlays(painter, binding)

    def _paint_image_binding_overlays(
        self,
        painter: QtGui.QPainter,
        binding: _ImagePanelBinding,
    ) -> None:
        target = self._selector_target(binding)
        if target is None:
            return
        image_target = target[0]
        painter.save()
        painter.setClipRect(image_target)
        if binding.applied_bounds is not None:
            visible = self._require_selector_viewport(
                binding
            ).clipped_visible_bounds_for_full_bounds(
                binding.applied_bounds
            )
            if visible is not None:
                self._paint_selector_rectangle(
                    painter,
                    visible,
                    image_target,
                    QtGui.QColor(GREEN),
                    binding=binding,
                    dashed=False,
                    handles=(
                        self._image_interaction_armed(binding)
                        and binding.draft_bounds is None
                        and self._rectangle_fully_visible(
                            binding,
                            binding.applied_bounds,
                        )
                    ),
                    endpoint_bounds=(
                        binding.applied_bounds
                        if binding.draft_bounds is None
                        else None
                    ),
                )
        if binding.draft_bounds is not None:
            visible = self._require_selector_viewport(
                binding
            ).clipped_visible_bounds_for_full_bounds(
                binding.draft_bounds
            )
            if visible is not None:
                self._paint_selector_rectangle(
                    painter,
                    visible,
                    image_target,
                    QtGui.QColor(ORANGE),
                    binding=binding,
                    dashed=True,
                    handles=(
                        self._image_interaction_armed(binding)
                        and self._rectangle_fully_visible(
                            binding,
                            binding.draft_bounds,
                        )
                    ),
                    endpoint_bounds=binding.draft_bounds,
                )
        hold = self._selector_hold
        held_image_payload = None if hold is None else _image_payload(hold)
        if (
            held_image_payload is not None
            and hold is not None
            and hold.panel_id == binding.panel_id
            and binding.clim_candidate is not None
        ):
            self._paint_clim_candidate_label(
                painter,
                binding,
                held_image_payload,
                image_target,
            )
        if binding.cross is not None:
            self._paint_cross_sample(
                painter,
                binding,
                binding.cross,
                image_target,
            )
        if binding.hover is not None and binding.hover_position is not None:
            self._paint_hover_sample(
                painter,
                binding,
                binding.hover,
                binding.hover_position,
                image_target,
            )
        painter.restore()

    def _paint_numeric_overlays(self, painter: QtGui.QPainter) -> None:
        for binding in self._numeric_bindings.values():
            self._paint_numeric_binding_overlay(painter, binding)

    def _paint_numeric_binding_overlay(
        self,
        painter: QtGui.QPainter,
        binding: _NumericPanelBinding,
    ) -> None:
        target = self._numeric_target(binding)
        if target is None:
            return
        plot, payload, bounds = target.plot, target.payload, target.bounds
        viewport = payload.viewport
        x_unit_value = (
            viewport.x_axis.unit
            if isinstance(viewport, NumericViewportTransform)
            else payload.value_unit
        )
        x_unit = "" if x_unit_value is None else f" {x_unit_value}"
        # A pulse timeline's y is the unit-less row position; histogram y is a
        # bare count.  Only the curve family carries a value unit to print.
        y_unit = (
            ""
            if isinstance(payload, (HistogramPanelPayload, PulsePanelPayload))
            else "" if payload.value_unit is None else f" {payload.value_unit}"
        )

        def widget_point(x: float, y: float) -> QtCore.QPointF:
            normalized = viewport.data_to_widget_normalized(x, y)
            return QtCore.QPointF(
                bounds.x() + normalized[0] * bounds.width(),
                bounds.y() + normalized[1] * bounds.height(),
            )

        painter.save()
        try:
            painter.setClipRect(plot)
            # The reference's AreaSelector, verbatim: a grey SOLID rectangle at
            # alpha 0.8, NO fill, white square handles, and (after release) an
            # unboxed two-line coordinate label in the top-left corner.
            rect_norm = binding.span_rect
            if rect_norm is not None:
                # The box is stored in DATA coordinates (the reference keeps
                # its selector artists in data space), so it follows the data
                # through zoom/pan; map it through the CURRENT viewport here.
                selector_color = _selector_pen_color()
                xs_px, ys_px = self._span_rect_widget_extents(
                    target, rect_norm)
                rectangle = QtCore.QRectF(
                    QtCore.QPointF(xs_px[0], ys_px[0]),
                    QtCore.QPointF(xs_px[1], ys_px[1]),
                )
                painter.setPen(QtGui.QPen(selector_color, SELECTOR_LINE_PX))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawRect(rectangle)
                handle_pen = QtGui.QPen(
                    selector_color, SELECTOR_LINE_PX / 2.0)
                painter.setPen(handle_pen)
                painter.setBrush(QtGui.QBrush(QtGui.QColor("white")))
                half = SELECTOR_HANDLE_PX / 2.0
                centre = rectangle.center()
                for hx, hy in (
                    (rectangle.left(), rectangle.top()),
                    (centre.x(), rectangle.top()),
                    (rectangle.right(), rectangle.top()),
                    (rectangle.left(), centre.y()),
                    (rectangle.right(), centre.y()),
                    (rectangle.left(), rectangle.bottom()),
                    (centre.x(), rectangle.bottom()),
                    (rectangle.right(), rectangle.bottom()),
                ):
                    painter.drawRect(QtCore.QRectF(
                        hx - half, hy - half,
                        SELECTOR_HANDLE_PX, SELECTOR_HANDLE_PX))
                if binding.span_anchor is None:
                    # Selection complete: the reference prints the SORTED extents
                    # "(xmin, ymin)\n(xmax, ymax)" top-left (its selector's
                    # ``extents`` order, whatever the drag direction), precision
                    # 1/1000 of the visible span, no box.  The box IS data
                    # coordinates -- print it directly.
                    lo_x, hi_x = sorted((rect_norm[0], rect_norm[2]))
                    lo_y, hi_y = sorted((rect_norm[1], rect_norm[3]))
                    dx = _selector_precision(
                        viewport.x_limits[1] - viewport.x_limits[0])
                    y_span = (
                        viewport.y_limits
                        if isinstance(viewport, NumericViewportTransform)
                        else viewport.count_limits
                    )
                    dy = _selector_precision(y_span[1] - y_span[0])
                    label = (f"({lo_x:.{dx}f}, {lo_y:.{dy}f})\n"
                             f"({hi_x:.{dx}f}, {hi_y:.{dy}f})")
                    self._paint_selector_text(
                        painter, label, plot, selector_color, corner="top_left")

            cross = binding.cross
            if cross is not None:
                selector_color = _selector_pen_color()
                point = widget_point(cross.x, cross.y)
                if plot.contains(point):
                    painter.setPen(QtGui.QPen(selector_color, SELECTOR_LINE_PX))
                    painter.drawLine(
                        QtCore.QPointF(point.x(), plot.top()),
                        QtCore.QPointF(point.x(), plot.bottom()),
                    )
                    painter.drawLine(
                        QtCore.QPointF(plot.left(), point.y()),
                        QtCore.QPointF(plot.right(), point.y()),
                    )
                    painter.setBrush(QtGui.QBrush(selector_color))
                    painter.setPen(QtCore.Qt.NoPen)
                    painter.drawEllipse(
                        point, SELECTOR_DOT_PX / 2.0, SELECTOR_DOT_PX / 2.0)
                dx = _selector_precision(
                    viewport.x_limits[1] - viewport.x_limits[0])
                y_span = (
                    viewport.y_limits
                    if isinstance(viewport, NumericViewportTransform)
                    else viewport.count_limits
                )
                dy = _selector_precision(y_span[1] - y_span[0])
                self._paint_selector_text(
                    painter,
                    f"({cross.x:.{dx}f}, {cross.y:.{dy}f})",
                    plot,
                    selector_color,
                    corner="top_right",
                )

            sample = binding.hover
            position = binding.hover_position
            if sample is not None and position is not None:
                point = None
                try:
                    point = widget_point(sample.x, sample.y)
                except ValueError:
                    pass
                if point is not None and plot.contains(point):
                    painter.setPen(QtGui.QPen(QtGui.QColor(ORANGE), 1.5))
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(ORANGE)))
                    painter.drawEllipse(point, 3.5, 3.5)
                label = (
                    (
                        f"{sample.series_label}  "
                        f"[{sample.left:.6g}, {sample.right:.6g}"
                        f"{']' if sample.right_closed else ')'}{x_unit}  "
                        f"count={sample.count}"
                    )
                    if isinstance(sample, _HistogramBinSample)
                    else (
                        f"{sample.series_label}  x={sample.x:.6g}{x_unit}  "
                        f"y={sample.y:.6g}{y_unit}"
                    )
                )
                self._paint_curve_label(
                    painter,
                    label,
                    plot,
                    QtGui.QColor(ORANGE),
                    anchor=position,
                )
        finally:
            painter.restore()

    @staticmethod
    def _paint_selector_text(
        painter: QtGui.QPainter,
        label: str,
        plot: QtCore.QRectF,
        color: QtGui.QColor,
        *,
        corner: str,
    ) -> None:
        """The reference's selector coordinate text, verbatim: UNBOXED, in the
        selector's own colour at legend-fontsize, inset 2.5% from the plot
        corner (``ax.text(0.025/0.975, 0.975, ...)``)."""

        painter.save()
        try:
            font = painter.font()
            font.setPixelSize(SELECTOR_FONT_PX)
            painter.setFont(font)
            painter.setPen(color)
            inset_x = 0.025 * plot.width()
            inset_y = 0.025 * plot.height()
            area = QtCore.QRectF(
                plot.left() + inset_x,
                plot.top() + inset_y,
                plot.width() - 2 * inset_x,
                plot.height() - 2 * inset_y,
            )
            flags = QtCore.Qt.AlignTop | (
                QtCore.Qt.AlignRight if corner == "top_right"
                else QtCore.Qt.AlignLeft)
            painter.drawText(area, flags, label)
        finally:
            painter.restore()

    @staticmethod
    def _paint_curve_label(
        painter: QtGui.QPainter,
        label: str,
        plot: QtCore.QRectF,
        color: QtGui.QColor,
        *,
        anchor: QtCore.QPointF | None = None,
        top_right: bool = False,
    ) -> None:
        metrics = painter.fontMetrics()
        label_bounds = metrics.boundingRect(label).adjusted(-5, -2, 5, 2)
        if top_right:
            label_bounds.moveTopRight(plot.topRight().toPoint() + QtCore.QPoint(-5, 5))
        else:
            if anchor is None:
                anchor = plot.topLeft()
            x = min(int(plot.right()) - label_bounds.width(), int(anchor.x()) + 12)
            y = min(int(plot.bottom()) - label_bounds.height(), int(anchor.y()) + 12)
            label_bounds.moveTopLeft(
                QtCore.QPoint(max(int(plot.left()), x), max(int(plot.top()), y))
            )
        painter.fillRect(label_bounds, QtGui.QColor(0, 0, 0, 190))
        painter.setPen(color)
        painter.drawText(label_bounds, QtCore.Qt.AlignCenter, label)

    def _rectangle_fully_visible(
        self,
        binding: _ImagePanelBinding,
        bounds: NormalizedRectangle,
    ) -> bool:
        try:
            self._require_selector_viewport(binding).visible_bounds_for_full_bounds(
                bounds
            )
        except ValueError:
            return False
        return True

    def _visible_point_for_sample(
        self,
        binding: _ImagePanelBinding,
        sample: _ImageSample,
    ) -> tuple[float, float] | None:
        viewport = self._require_selector_viewport(binding)
        if (
            sample.x_index >= viewport.x_axis.size
            or sample.y_index >= viewport.y_axis.size
        ):
            return None
        full = (
            (sample.x_index + 0.5) / viewport.x_axis.size,
            (sample.y_index + 0.5) / viewport.y_axis.size,
        )
        try:
            return viewport.visible_point_for_full_point(full)
        except ValueError:
            return None

    def _paint_cross_sample(
        self,
        painter: QtGui.QPainter,
        binding: _ImagePanelBinding,
        sample: _ImageSample,
        target: QtCore.QRect,
    ) -> None:
        point = self._visible_point_for_sample(binding, sample)
        color = QtGui.QColor(GREEN)
        if point is not None:
            x = target.x() + point[0] * target.width()
            y = target.y() + point[1] * target.height()
            painter.setPen(QtGui.QPen(color, 1.5))
            painter.drawLine(
                QtCore.QPointF(x, target.top()),
                QtCore.QPointF(x, target.bottom()),
            )
            painter.drawLine(
                QtCore.QPointF(target.left(), y),
                QtCore.QPointF(target.right(), y),
            )
            painter.setBrush(QtGui.QBrush(color))
            painter.drawEllipse(QtCore.QPointF(x, y), 3.5, 3.5)
        value = self._formatted_sample_value(sample)
        suffix = " · off-view" if point is None else ""
        label = (
            f"({sample.x_coordinate}, {sample.y_coordinate}){suffix}"
            if self.visible_site_map_payload(binding.panel_id) is not None
            else f"({sample.x_coordinate}, {sample.y_coordinate}, {value}){suffix}"
        )
        metrics = painter.fontMetrics()
        bounds = metrics.boundingRect(label).adjusted(-5, -2, 5, 2)
        bounds.moveTopRight(target.topRight() + QtCore.QPoint(-5, 5))
        painter.fillRect(bounds, QtGui.QColor(0, 0, 0, 190))
        painter.setPen(color)
        painter.drawText(bounds, QtCore.Qt.AlignCenter, label)

    def _paint_hover_sample(
        self,
        painter: QtGui.QPainter,
        binding: _ImagePanelBinding,
        sample: _ImageSample,
        position: QtCore.QPointF,
        target: QtCore.QRect,
    ) -> None:
        site_map = self.visible_site_map_payload(binding.panel_id)
        if site_map is None:
            label = (
                f"x={sample.x_coordinate}  y={sample.y_coordinate}  "
                f"z={self._formatted_sample_value(sample)}"
            )
        else:
            point = np.asarray(
                (float(sample.x_coordinate), float(sample.y_coordinate)),
                dtype=np.float64,
            )
            distances = np.sum(np.square(site_map.centers_xy - point), axis=1)
            site_index = int(np.argmin(distances))
            state = (
                "invalid"
                if not site_map.site_validity[site_index]
                else "occupied"
                if site_map.occupied[site_index]
                else "empty"
            )
            site_label = site_map.site_axis.coordinate_at(site_index)
            label = (
                f"x={sample.x_coordinate}  y={sample.y_coordinate}  "
                f"z={self._formatted_sample_value(sample)}  "
                f"nearest={site_label} ({state})"
            )
        metrics = painter.fontMetrics()
        bounds = metrics.boundingRect(label).adjusted(-5, -2, 5, 2)
        x = min(target.right() - bounds.width(), int(position.x()) + 12)
        y = min(target.bottom() - bounds.height(), int(position.y()) + 12)
        bounds.moveTopLeft(QtCore.QPoint(max(target.left(), x), max(target.top(), y)))
        painter.fillRect(bounds, QtGui.QColor(0, 0, 0, 190))
        painter.setPen(QtGui.QColor(ORANGE))
        painter.drawText(bounds, QtCore.Qt.AlignCenter, label)

    def _paint_clim_candidate_label(
        self,
        painter: QtGui.QPainter,
        binding: _ImagePanelBinding,
        payload: ImagePanelPayload,
        target: QtCore.QRect,
    ) -> None:
        label = self._clim_candidate_label(binding, payload)
        metrics = painter.fontMetrics()
        bounds = metrics.boundingRect(label).adjusted(-6, -3, 6, 3)
        bounds.moveBottomLeft(target.bottomLeft() + QtCore.QPoint(7, -7))
        painter.fillRect(bounds, QtGui.QColor(0, 0, 0, 190))
        painter.setPen(QtGui.QColor(ORANGE))
        painter.drawText(bounds, QtCore.Qt.AlignCenter, label)

    def _clim_candidate_label(
        self,
        binding: _ImagePanelBinding,
        payload: ImagePanelPayload,
    ) -> str:
        limits = binding.clim_candidate
        if limits is None:
            raise RuntimeError("H candidate label requires an active limit draft")
        low, high = self._color_rail_domain(payload)
        span = high - low
        gap = span / 1000.0 if span else 0.01
        precision = max(0, -int(math.ceil(math.log10(gap))))

        def formatted(value: float) -> str:
            return (
                f"{value:.{precision}f}"
                if precision <= 6 and abs(value) < 1.0e9
                else f"{value:.6g}"
            )

        return f"H low={formatted(limits[0])}  high={formatted(limits[1])}"

    @staticmethod
    def _formatted_sample_value(sample: _ImageSample) -> str:
        if not sample.valid:
            return "invalid"
        value = sample.value
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _paint_selector_rectangle(
        self,
        painter: QtGui.QPainter,
        bounds: NormalizedRectangle,
        target: QtCore.QRect,
        color: QtGui.QColor,
        *,
        binding: _ImagePanelBinding,
        dashed: bool,
        handles: bool,
        endpoint_bounds: NormalizedRectangle | None,
    ) -> None:
        rectangle = self._overlay_rect(bounds, target)
        pen = QtGui.QPen(color, 2.0)
        if dashed:
            pen.setStyle(QtCore.Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(rectangle)
        if endpoint_bounds is not None:
            label = self._selection_endpoint_label(binding, endpoint_bounds)
            flags = QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
            label_area = target.adjusted(7, 7, -7, -7)
            label_bounds = painter.fontMetrics().boundingRect(
                label_area,
                flags,
                label,
            ).adjusted(-5, -3, 5, 3)
            painter.fillRect(label_bounds, QtGui.QColor(0, 0, 0, 190))
            painter.setPen(color)
            painter.drawText(label_bounds.adjusted(5, 3, -5, -3), flags, label)
        if handles:
            painter.setPen(QtGui.QPen(color, 1.0))
            painter.setBrush(QtGui.QBrush(QtCore.Qt.white))
            size = 8.0
            for corner in (
                rectangle.topLeft(),
                rectangle.topRight(),
                rectangle.bottomLeft(),
                rectangle.bottomRight(),
            ):
                painter.drawRect(
                    QtCore.QRectF(
                        corner.x() - size / 2,
                        corner.y() - size / 2,
                        size,
                        size,
                    )
                )

    def _selection_endpoint_label(
        self,
        binding: _ImagePanelBinding,
        bounds: NormalizedRectangle,
    ) -> str:
        viewport = self._require_selector_viewport(binding)
        selected_terms = {
            term.axis_id: term
            for term in viewport.selection_for_normalized_bounds(bounds).terms
        }
        visible_cell_bounds = viewport.snapped_bounds_for_drag(
            (0.0, 0.0),
            (1.0, 1.0),
        )
        visible_terms = {
            term.axis_id: term
            for term in viewport.selection_for_normalized_bounds(
                visible_cell_bounds
            ).terms
        }
        selected_x = selected_terms[viewport.x_axis.axis_id]
        selected_y = selected_terms[viewport.y_axis.axis_id]
        visible_x = visible_terms[viewport.x_axis.axis_id]
        visible_y = visible_terms[viewport.y_axis.axis_id]

        def precision(span: float) -> int:
            gap = abs(float(span)) / 1000.0 if span else 0.01
            return max(0, -int(math.ceil(math.log10(gap))))

        x_precision = precision(float(visible_x.upper) - float(visible_x.lower))
        y_precision = precision(float(visible_y.upper) - float(visible_y.lower))
        return (
            f"({float(selected_x.lower):.{x_precision}f}, "
            f"{float(selected_y.lower):.{y_precision}f})\n"
            f"({float(selected_x.upper):.{x_precision}f}, "
            f"{float(selected_y.upper):.{y_precision}f})"
        )

    def _cancel_image_gesture(
        self,
        binding: _ImagePanelBinding,
        *,
        clear_draft: bool,
    ) -> None:
        binding.drag_anchor = None
        binding.drag_prior_draft = None
        binding.pan_anchor = None
        binding.pan_origin = None
        binding.pan_target_size = None
        binding.pan_candidate = None
        binding.clim_drag = None
        binding.clim_origin_limits = None
        binding.clim_candidate = None
        binding.clim_domain = None
        if (
            self._selector_hold is not None
            and self._selector_hold.panel_id == binding.panel_id
        ):
            self._selector_hold = None
        if clear_draft:
            binding.draft_bounds = None

    def _clear_image_transient(
        self,
        binding: _ImagePanelBinding,
        *,
        clear_applied_bounds: bool,
        clear_pending: bool,
    ) -> None:
        self._cancel_image_gesture(binding, clear_draft=True)
        if clear_applied_bounds:
            binding.applied_bounds = None
        if clear_pending:
            binding.pending_viewport = None
            binding.pending_color_limits = None
            binding.pending_origin = None
        binding.cross = None
        self._set_hover_sample(binding, None)

    def _cancel_numeric_gesture(
        self,
        binding: _NumericPanelBinding,
        *,
        clear_span: bool,
    ) -> None:
        # A gesture cancelled MID-DRAG (the anchor is still armed) loses its
        # half-drawn rectangle; a completed selection keeps its box + label
        # (the reference's RectangleSelector leaves them standing).
        if clear_span and (
            binding.span_anchor is not None
            or binding.span_move_grab is not None
        ):
            binding.span_rect = None
        binding.span_anchor = None
        binding.span_move_grab = None
        binding.span_resize_lock = None
        binding.threshold_drag = None
        binding.threshold_candidate = None
        binding.threshold_pending_revision = None
        binding.threshold_pending_origin = None
        binding.pan_anchor = None
        binding.pan_origin = None
        binding.pan_candidate = None
        if (
            self._selector_hold is not None
            and self._selector_hold.panel_id == binding.panel_id
        ):
            self._selector_hold = None
        if clear_span:
            binding.span_candidate = None

    def _clear_numeric_transient(
        self,
        binding: _NumericPanelBinding,
        *,
        clear_applied_span: bool,
        clear_pending: bool,
    ) -> None:
        self._cancel_numeric_gesture(binding, clear_span=True)
        if clear_applied_span:
            binding.applied_span = None
        if clear_pending:
            binding.pending_viewport = None
            binding.pending_origin = None
        binding.cross = None
        self._set_numeric_hover(binding, None)

    def _cancel_active_gesture(
        self,
        *,
        clear_image_draft: bool,
        clear_numeric_spans: bool,
    ) -> None:
        for binding in self._image_bindings.values():
            self._cancel_image_gesture(
                binding,
                clear_draft=clear_image_draft,
            )
        for binding in self._numeric_bindings.values():
            self._cancel_numeric_gesture(
                binding,
                clear_span=clear_numeric_spans,
            )
        self._selector_hold = None

    def _reset_image_binding(self, panel_id: str) -> None:
        binding = self._image_bindings.get(panel_id)
        if binding is None:
            return
        self._clear_image_transient(
            binding,
            clear_applied_bounds=True,
            clear_pending=True,
        )
        binding.binding_enabled = False
        binding.interaction_ready = False
        del self._image_bindings[panel_id]
        if not self._image_bindings and not self._numeric_bindings:
            self._selector_enabled = False

    def _reset_all_image_bindings(self) -> None:
        for panel_id in tuple(self._image_bindings):
            self._reset_image_binding(panel_id)

    def _reset_numeric_binding(self, panel_id: str) -> None:
        binding = self._numeric_bindings.get(panel_id)
        if binding is None:
            return
        self._clear_numeric_transient(
            binding,
            clear_applied_span=True,
            clear_pending=True,
        )
        binding.binding_enabled = False
        binding.interaction_ready = False
        del self._numeric_bindings[panel_id]
        if not self._image_bindings and not self._numeric_bindings:
            self._selector_enabled = False

    def _reset_all_numeric_bindings(self) -> None:
        for panel_id in tuple(self._numeric_bindings):
            self._reset_numeric_binding(panel_id)

    def _require_owner(self) -> None:
        if QtCore.QThread.currentThread() != self.thread():
            raise RuntimeError("QtRasterBoard presentation is GUI-thread affine")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("QtRasterBoard is closed")


__all__ = ["QtRasterBoard"]
