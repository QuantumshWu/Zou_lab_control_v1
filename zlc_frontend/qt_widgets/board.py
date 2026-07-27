"""Interactive Qt presenter for immutable semantic board fronts."""

from __future__ import annotations

import math
from typing import Callable, Literal

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_data import Selection
from zlc_storage import canonical_text, nonnegative_integer

from ..display_range import validated_display_range
from ..image_view import validate_normalized_rectangle
from ..render import (
    BoardFrame,
    CurvePanelPayload,
    DisplayPayload,
    HistogramPanelPayload,
    ImagePanelPayload,
    MeterPanelPayload,
    PulsePanelPayload,
    SiteMapPanelPayload,
    detached_render_fault,
)
from ..selector import (
    CrossGesture,
    CurveInteractionIntent,
    CurveRangeGesture,
    HistogramInteractionIntent,
    HistogramRangeGesture,
    ImageInteractionCommit,
    ImageViewportTransform,
    NormalizedRectangle,
    PanelInteractionOrigin,
    RectangleGesture,
)
from ._raster_front import (
    _HeldPanelFront,
    _advance_held_front,
    _image_payload,
    _hold_matches_frame,
    _panel_bounds,
    _panel_image_geometry,
    _panel_presentation,
    _panel_semantics_changed,
    _prepared_qimage,
    _validated_panel_layout,
    _visible_display as _resolved_visible_display,
)
from ._raster_image_interaction import (
    _ImagePanelBinding,
    _ImageSample,
    _active_image_binding,
    _cancel_image_gesture,
    _clear_image_transient,
    _clim_handle_at,
    _color_rail_domain,
    _commit_color_limits as _commit_image_color_limits,
    _commit_viewport as _commit_image_viewport,
    _held_panel_from_target,
    _image_bounds_for_rectangle_drag,
    _image_interaction_armed,
    _image_interaction_is_pending,
    _image_target_at,
    _normalized_point,
    _overlay_rect,
    _paint_clim_draft_lines,
    _paint_image_overlays,
    _painted_image_panel_id_at,
    _rail_value,
    _selector_target,
    _sample_for_target,
    _clim_rail_target,
    _viewport_for_target,
    _viewport_for_presented_panel,
    _validate_selector_binding,
)
from ._raster_numeric_interaction import (
    _NUMERIC_PAYLOAD_TYPES,
    _NumericIntent,
    _NumericKind,
    _NumericPanelBinding,
    _NumericTarget,
    _NumericViewport,
    _active_numeric_binding,
    _cancel_numeric_gesture,
    _clear_numeric_transient,
    _commit_histogram_thresholds as _commit_numeric_thresholds,
    _commit_numeric_viewport as _commit_numeric_x_viewport,
    _numeric_payload,
    _held_panel_from_numeric_target,
    _paint_numeric_overlays,
    _numeric_target,
    _numeric_target_at,
    _numeric_normalized_point,
    _numeric_interaction_armed,
    _numeric_interaction_is_pending,
    _numeric_viewport_for_presented_panel,
    _span_data_candidate,
    _span_rect_widget_extents,
    _threshold_line_hit,
)
from ._rectangle_selector import (
    RectangleDrag,
    hit_rectangle_handle,
)
from .style import BG


class QtRasterBoard(QtWidgets.QWidget):
    """Atomic multi-panel presenter for immutable worker-owned raster fronts.

    Ordinary pointer motion has no plot meaning.  Data readout is the explicit
    right-click Cross gesture; motion is consumed only while a pressed selector,
    pan, or draggable line already owns the pointer.
    """

    imagePanelLeftDoubleClicked = QtCore.pyqtSignal(str)
    crossSelected = QtCore.pyqtSignal(object)
    interactionStarted = QtCore.pyqtSignal(object)
    interactionFinished = QtCore.pyqtSignal()

    def __init__(
        self,
        panel_ids: tuple[str, ...],
        parent: QtWidgets.QWidget | None = None,
        *,
        columns: int = 2,
        empty_text: str = "",
    ) -> None:
        super().__init__(parent)
        # Product contract: plots do not sample/publish/repaint data merely
        # because the pointer crossed them.  Pressed gestures still receive
        # ordinary Qt mouse-move events without mouse tracking.
        self.setMouseTracking(False)
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
        self.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.setMinimumSize(128, 64)

    def _begin_interaction_hold(self, hold: _HeldPanelFront) -> None:
        """Pin the exact painted panel before any pointer motion can race live data."""

        if not isinstance(hold, _HeldPanelFront):
            raise TypeError("interaction hold must retain one painted panel front")
        if self._selector_hold is not None:
            raise RuntimeError("another pointer interaction is already active")
        self._selector_hold = hold
        _payload, origin = self._visible_display(
            hold.panel_id,
            (
                ImagePanelPayload,
                SiteMapPanelPayload,
                CurvePanelPayload,
                HistogramPanelPayload,
                PulsePanelPayload,
            ),
        )
        if origin is None:
            self._selector_hold = None
            raise RuntimeError("interaction hold has no exact painted origin")
        self.interactionStarted.emit(origin)

    def _finish_interaction_hold(self) -> None:
        """Release exactly one pointer pin and notify its presentation owner."""

        if self._selector_hold is None:
            return
        self._selector_hold = None
        self.interactionFinished.emit()

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
        """Validate, prepare, then atomically install one ordinary front."""

        self._present(frame, interaction_transition=None)

    def present_with_single_panel_interaction(
        self,
        frame: BoardFrame,
        *,
        panel_id: str,
        kind: Literal["image", "curve", "histogram", "pulse"] | None,
        interaction_callback: Callable[[object], object],
        rectangle_callback: Callable[[RectangleGesture], object],
        selectors_enabled: bool,
    ) -> None:
        """Atomically replace one front and its interaction family.

        The candidate binding is completely typed and validated before the old
        binding or old front is touched.  This is the only safe family-change
        path for :class:`SinglePanelHost`: an invalid new raster/payload cannot
        retire a still-working selector binding.
        """

        self._require_owner()
        self._ensure_open()
        if not isinstance(frame, BoardFrame):
            raise TypeError("frame must be BoardFrame")
        panel_id = canonical_text(panel_id, "panel_id")
        if len(frame.panels) != 1 or frame.panels[0].panel_id != panel_id:
            raise ValueError(
                "single-panel interaction transition requires its one panel"
            )
        if kind not in (None, "image", "curve", "histogram", "pulse"):
            raise ValueError("unsupported single-panel interaction kind")
        if not callable(interaction_callback) or not callable(rectangle_callback):
            raise TypeError("single-panel interaction callbacks must be callable")
        if not isinstance(selectors_enabled, bool):
            raise TypeError("selectors_enabled must be bool")
        payload = frame.panels[0].display_payload
        candidate_image = None
        candidate_numeric = None
        if kind == "image":
            image_payload = (
                payload.background
                if isinstance(payload, SiteMapPanelPayload)
                else payload
            )
            if not isinstance(image_payload, ImagePanelPayload):
                raise TypeError("image interaction requires ImagePanelPayload")
            candidate_image = _ImagePanelBinding(
                panel_id,
                image_payload.viewport,
                rectangle_callback,
                interaction_callback=interaction_callback,
                revision_floor=image_payload.viewport.viewport_revision,
                interaction_ready=True,
            )
        elif kind is not None:
            payload_type = _NUMERIC_PAYLOAD_TYPES[kind]
            if not isinstance(payload, payload_type):
                raise TypeError(
                    f"{kind} interaction requires {payload_type.__name__}"
                )
            candidate_numeric = _NumericPanelBinding(
                kind,
                panel_id,
                interaction_callback,
                viewport=payload.viewport,
                revision_floor=payload.viewport.display_revision,
                interaction_ready=True,
            )
        self._present(
            frame,
            interaction_transition=(
                panel_id,
                candidate_image,
                candidate_numeric,
                selectors_enabled,
            ),
        )

    def _present(self, frame: BoardFrame, *, interaction_transition) -> None:
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
        queued_image_viewports: list[
            tuple[_ImagePanelBinding, NormalizedRectangle]
        ] = []
        queued_image_colors: list[
            tuple[_ImagePanelBinding, tuple[float, float]]
        ] = []
        queued_numeric_viewports: list[
            tuple[_NumericPanelBinding, tuple[float, float]]
        ] = []
        queued_histogram_thresholds: list[
            tuple[_NumericPanelBinding, tuple[float, ...]]
        ] = []
        validation_image_bindings = self._image_bindings
        validation_numeric_bindings = self._numeric_bindings
        if interaction_transition is not None:
            (
                transition_panel_id,
                candidate_image,
                candidate_numeric,
                _transition_selectors_enabled,
            ) = interaction_transition
            validation_image_bindings = dict(self._image_bindings)
            validation_numeric_bindings = dict(self._numeric_bindings)
            validation_image_bindings.pop(transition_panel_id, None)
            validation_numeric_bindings.pop(transition_panel_id, None)
            if candidate_image is not None:
                validation_image_bindings[transition_panel_id] = candidate_image
            if candidate_numeric is not None:
                validation_numeric_bindings[transition_panel_id] = candidate_numeric
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
            for panel_id, binding in validation_image_bindings.items():
                if panel_id not in target_panel_ids:
                    continue
                target_viewport = _viewport_for_presented_panel(
                    binding,
                    frame,
                    panel_ids=target_panel_ids,
                    previous=self._front,
                    previous_panel_ids=self._panel_ids,
                )
                _validate_selector_binding(
                    panel_id,
                    target_viewport,
                    frame,
                    panel_ids=target_panel_ids,
                )
                target_image_viewports[panel_id] = target_viewport
                pending_color = binding.pending_color_answer
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
                    pending_color is not None
                    and _image_payload(target_panel) is not None
                    and _panel_presentation(target_panel).panel_revision
                    == pending_color.display_revision
                    and not pending_color.matches(target_panel)
                ):
                    raise ValueError(
                        "pending image color-limit revision returned conflicting limits"
                    )
            for panel_id, binding in validation_numeric_bindings.items():
                if panel_id in target_panel_ids:
                    target_numeric_viewports[panel_id] = (
                        _numeric_viewport_for_presented_panel(
                            binding,
                            frame,
                            panel_ids=target_panel_ids,
                            previous=self._front,
                            previous_panel_ids=self._panel_ids,
                        )
                    )
            if interaction_was_active:
                hold = self._selector_hold
                if hold is None:
                    raise RuntimeError(
                        "active rectangle interaction has no held panel front"
                    )
                cancel_interaction = not _hold_matches_frame(
                    hold,
                    frame,
                    panel_ids=target_panel_ids,
                )
            # Only after all cheap identity/revision checks pass do we wrap the
            # worker-owned immutable raster in a QImage view.
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
        # The candidate frame, QImages, and candidate binding all passed.  The
        # family swap begins only here; no validation below can reject it.
        if interaction_transition is not None:
            (
                transition_panel_id,
                candidate_image,
                candidate_numeric,
                transition_selectors_enabled,
            ) = interaction_transition
            if transition_panel_id in self._image_bindings:
                self._reset_image_binding(transition_panel_id)
            if transition_panel_id in self._numeric_bindings:
                self._reset_numeric_binding(transition_panel_id)
            if candidate_image is not None:
                self._image_bindings[transition_panel_id] = candidate_image
            if candidate_numeric is not None:
                self._numeric_bindings[transition_panel_id] = candidate_numeric
            self._selector_enabled = transition_selectors_enabled
        previous = self._front
        for panel_id, binding in tuple(self._image_bindings.items()):
            if panel_id not in target_panel_ids:
                self._reset_image_binding(panel_id)
            elif previous is not None and panel_id in self._panel_ids:
                old_index = self._panel_ids.index(panel_id)
                new_index = target_panel_ids.index(panel_id)
                old_panel = previous[0].panels[old_index]
                new_panel = frame.panels[new_index]
                if _panel_semantics_changed(old_panel, new_panel):
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
                if _panel_semantics_changed(old_panel, new_panel):
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
            hold = self._selector_hold
            panel = frame.panels[target_panel_ids.index(panel_id)]
            viewport_answered = (
                binding.pending_viewport_answer is not None
                and binding.pending_viewport_answer.matches(panel)
            )
            color_answered = (
                binding.pending_color_answer is not None
                and binding.pending_color_answer.matches(panel)
            )
            if viewport_answered:
                binding.pending_viewport_answer = None
                queued = binding.queued_viewport_bounds
                binding.queued_viewport_bounds = None
                if queued is not None:
                    queued_image_viewports.append((binding, queued))
            if color_answered:
                binding.pending_color_answer = None
                queued = binding.queued_color_limits
                binding.queued_color_limits = None
                if queued is not None:
                    queued_image_colors.append((binding, queued))
            if (
                (viewport_answered or color_answered)
                and hold is not None
                and hold.panel_id == panel_id
                # Area is a Qt-only gesture against the exact raster held at
                # press time.  A render-backed answer may be admitted in the
                # background, but advancing this held front would change the
                # coordinate transform halfway through the drag.  Keep the
                # old raster/viewport until release; the newly admitted front
                # becomes visible as soon as the hold is cleared.
                and binding.rectangle_drag is None
            ):
                index = target_panel_ids.index(panel_id)
                self._selector_hold = _advance_held_front(
                    hold, frame, panel, prepared[index]
                )
        for panel_id, binding in self._numeric_bindings.items():
            hold = self._selector_hold
            index = target_panel_ids.index(panel_id)
            panel = frame.panels[index]
            viewport_answered = (
                binding.pending_viewport_answer is not None
                and binding.pending_viewport_answer.matches(panel)
            )
            threshold_answered = (
                binding.threshold_pending_answer is not None
                and binding.threshold_pending_answer.matches(panel)
            )
            if viewport_answered:
                binding.pending_viewport_answer = None
                queued = binding.queued_viewport_limits
                binding.queued_viewport_limits = None
                if queued is not None:
                    queued_numeric_viewports.append((binding, queued))
            if threshold_answered:
                binding.threshold_pending_answer = None
                queued = binding.queued_thresholds
                binding.queued_thresholds = None
                if queued is not None:
                    queued_histogram_thresholds.append((binding, queued))
            if viewport_answered or threshold_answered:
                if hold is not None and hold.panel_id == panel_id:
                    self._selector_hold = _advance_held_front(
                        hold, frame, panel, prepared[index]
                    )
        self._front = (frame, prepared)
        # Pointer input is allowed to outrun Agg, but semantic answers are not
        # guessed or accepted by revision alone.  Each binding keeps one exact
        # in-flight answer plus one latest desired state.  Once that answer is
        # installed as the real front, author the coalesced state from this
        # new exact origin.  This yields live render-rate feedback without an
        # unbounded request queue or fake Qt-side raster scaling.
        for binding, bounds in queued_image_viewports:
            self._author_queued_image_viewport(binding, bounds)
        for binding, limits in queued_image_colors:
            self._author_queued_image_color(binding, limits)
        for binding, limits in queued_numeric_viewports:
            self._author_queued_numeric_viewport(binding, limits)
        for binding, thresholds in queued_histogram_thresholds:
            self._author_queued_histogram_thresholds(binding, thresholds)
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

        return _resolved_visible_display(
            panel_id,
            payload_type,
            front=self._front,
            panel_ids=self._panel_ids,
            hold=self._selector_hold,
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
            or not _image_interaction_is_pending(binding)
            or not any(
                answer is not None and answer.origin == origin
                for answer in (
                    binding.pending_viewport_answer,
                    binding.pending_color_answer,
                )
            )
        ):
            return False
        if (
            binding.pending_viewport_answer is not None
            and binding.pending_viewport_answer.origin == origin
        ):
            queued_viewport = binding.queued_viewport_bounds
            binding.pending_viewport_answer = None
            binding.queued_viewport_bounds = None
        else:
            queued_viewport = None
        if (
            binding.pending_color_answer is not None
            and binding.pending_color_answer.origin == origin
        ):
            queued_color = binding.queued_color_limits
            binding.pending_color_answer = None
            binding.queued_color_limits = None
        else:
            queued_color = None
        if queued_viewport is not None:
            self._author_queued_image_viewport(binding, queued_viewport)
        if queued_color is not None:
            self._author_queued_image_color(binding, queued_color)
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
            or binding.pending_viewport_answer is None
            or origin != binding.pending_viewport_answer.origin
        ):
            return False
        queued = binding.queued_viewport_limits
        binding.pending_viewport_answer = None
        binding.queued_viewport_limits = None
        if queued is not None:
            self._author_queued_numeric_viewport(binding, queued)
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
        if (
            binding.pending_viewport_answer is not None
            and origin == binding.pending_viewport_answer.origin
        ):
            queued_viewport = binding.queued_viewport_limits
            binding.pending_viewport_answer = None
            binding.queued_viewport_limits = None
            discarded = True
        else:
            queued_viewport = None
        if (
            binding.threshold_pending_answer is not None
            and origin == binding.threshold_pending_answer.origin
        ):
            queued_thresholds = binding.queued_thresholds
            binding.threshold_pending_answer = None
            binding.queued_thresholds = None
            discarded = True
        else:
            queued_thresholds = None
        if not discarded:
            return False
        if queued_viewport is not None:
            self._author_queued_numeric_viewport(binding, queued_viewport)
        if queued_thresholds is not None:
            self._author_queued_histogram_thresholds(
                binding,
                queued_thresholds,
            )
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
            or binding.pending_viewport_answer is None
            or origin != binding.pending_viewport_answer.origin
        ):
            return False
        queued = binding.queued_viewport_limits
        binding.pending_viewport_answer = None
        binding.queued_viewport_limits = None
        if queued is not None:
            self._author_queued_numeric_viewport(binding, queued)
        self.update()
        return True

    def selection_for_rectangle_gesture(self, gesture: RectangleGesture) -> Selection:
        """Resolve a gesture only while its exact display-only origin is held."""

        self._require_owner()
        if not isinstance(gesture, RectangleGesture):
            raise TypeError("gesture must be RectangleGesture")
        if gesture.normalized_bounds is None:
            raise ValueError("a cleared image rectangle has no Selection")
        hold = self._selector_hold
        binding = self._image_bindings.get(gesture.panel_id)
        if (
            hold is None
            or binding is None
            or binding.rectangle_drag is not None
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
        viewport = self._require_selector_viewport(binding)
        if viewport.viewport_revision != gesture.viewport_revision:
            raise RuntimeError("rectangle gesture viewport changed before dispatch")
        front = self._front
        if front is None or not _hold_matches_frame(
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
            or binding.rectangle_drag is not None
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
            _validate_selector_binding(
                panel_id,
                viewport,
                self._front[0],
                panel_ids=self._panel_ids,
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
            binding.interaction_ready = image
        readiness = {"curve": curve, "histogram": histogram, "pulse": pulse}
        for binding in self._numeric_bindings.values():
            ready = readiness[binding.kind]
            if not ready and binding.interaction_ready:
                self._cancel_numeric_gesture(binding, clear_span=True)
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
        self.update()

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
        binding.interaction_ready = ready
        self.update()

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
        if hold is not None and not _hold_matches_frame(
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
                (
                    CurvePanelPayload,
                    HistogramPanelPayload,
                    MeterPanelPayload,
                    PulsePanelPayload,
                ),
            ):
                target = bounds
                source = QtCore.QRectF(
                    0.0,
                    0.0,
                    float(image.width()),
                    float(image.height()),
                )
            else:
                image_payload = (
                    payload.background
                    if isinstance(payload, SiteMapPanelPayload)
                    else payload
                    if isinstance(payload, ImagePanelPayload)
                    else None
                )
                geometry = _panel_image_geometry(
                    bounds,
                    image,
                    image_payload,
                    site_map_payload=(
                        payload
                        if isinstance(payload, SiteMapPanelPayload)
                        else None
                    ),
                )
                if image_payload is not None:
                    target = bounds
                    source = QtCore.QRectF(
                        0.0,
                        0.0,
                        float(image.width()),
                        float(image.height()),
                    )
                else:
                    target = geometry.target
                    source = geometry.source
                if image_payload is not None:
                    painter.fillRect(bounds, QtGui.QColor("white"))
            painter.drawImage(QtCore.QRectF(target), image, source)
            if (
                image_payload is not None
                and geometry.distribution is not None
            ):
                _paint_clim_draft_lines(
                    painter,
                    image_payload,
                    geometry.distribution,
                    self._image_bindings.get(panel_id),
                    hold=self._selector_hold,
                )
        _paint_image_overlays(
            painter,
            selector_enabled=self._selector_enabled,
            widget_rect=self.rect(),
            panel_ids=self._panel_ids,
            columns=self._columns,
            front=self._front,
            hold=self._selector_hold,
            bindings=self._image_bindings,
        )
        _paint_numeric_overlays(
            painter,
            widget_rect=self.rect(),
            panel_ids=self._panel_ids,
            columns=self._columns,
            front=self._front,
            hold=self._selector_hold,
            bindings=self._numeric_bindings,
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._selector_enabled:
            super().mousePressEvent(event)
            return
        numeric_target = self._numeric_target_at(event.localPos())
        if (
            numeric_target is not None
            and _numeric_interaction_armed(self._selector_enabled, numeric_target.binding)
        ):
            binding = numeric_target.binding
            if self._selector_hold is not None:
                event.accept()
                return
            if _numeric_interaction_is_pending(binding):
                event.accept()
                return
            point = _numeric_normalized_point(
                numeric_target, event.localPos()
            )
            viewport = numeric_target.payload.viewport
            if event.button() == QtCore.Qt.RightButton:
                x, y = viewport.widget_normalized_to_data(*point)
                binding.cross = (x, y)
                self.crossSelected.emit(
                    CrossGesture(
                        self._numeric_interaction_origin(binding),
                        (x, y),
                    )
                )
                self.update()
                event.accept()
                return
            if event.button() == QtCore.Qt.MiddleButton:
                self._begin_interaction_hold(
                    _held_panel_from_numeric_target(numeric_target)
                )
                binding.pan_anchor = point[0]
                binding.pan_origin = viewport
                binding.pan_candidate = viewport.x_limits
                self.update()
                event.accept()
                return
            if event.button() == QtCore.Qt.LeftButton:
                # The reference's DragVLine takes PRIORITY over the area
                # selector: a left press within 2% of the x span of an
                # authored threshold line grabs THAT line, and the area
                # machinery stays untouched for the whole drag (the exclusive
                # arrangement the design's histogram row freezes).
                grabbed = _threshold_line_hit(
                    numeric_target, event.localPos())
                if grabbed is not None:
                    self._begin_interaction_hold(
                        _held_panel_from_numeric_target(numeric_target)
                    )
                    binding.threshold_drag = grabbed
                    binding.threshold_candidate = tuple(
                        numeric_target.payload.thresholds)
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
                    binding.span_rect
                    if binding.rectangle_drag is None
                    else None
                )
                handle = None
                display_rectangle = None
                if standing is not None:
                    first = viewport.data_to_widget_normalized(
                        standing[0], standing[1])
                    second = viewport.data_to_widget_normalized(
                        standing[2], standing[3])
                    display_rectangle = (
                        first[0], first[1], second[0], second[1])
                    xs_px, ys_px = _span_rect_widget_extents(
                        numeric_target, standing)
                    handle = hit_rectangle_handle(
                        QtCore.QRectF(
                            QtCore.QPointF(xs_px[0], ys_px[0]),
                            QtCore.QPointF(xs_px[1], ys_px[1]),
                        ),
                        event.localPos(),
                    )
                self._begin_interaction_hold(
                    _held_panel_from_numeric_target(numeric_target)
                )
                if handle is not None and display_rectangle is not None:
                    binding.rectangle_drag = RectangleDrag.begin(
                        display_rectangle,
                        handle,
                        point,
                    )
                    binding.span_candidate = _span_data_candidate(
                        standing[0], standing[2])
                else:
                    binding.rectangle_drag = RectangleDrag.fresh(point)
                    binding.span_candidate = None
                    pressed = viewport.widget_normalized_to_data(*point)
                    binding.span_rect = (
                        pressed[0], pressed[1], pressed[0], pressed[1])
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
        if not _image_interaction_armed(self._selector_enabled, binding):
            super().mousePressEvent(event)
            return
        hits_image = target is not None and target[0].contains(event.pos())
        hits_rail = rail_target is not None and rail_target[0].contains(event.pos())
        # Area and Cross do not author a raster answer: both can safely consume
        # the exact panel that is still painted while a viewport/clim answer is
        # pending.  Pan and clim remain serialized because they author another
        # render-backed transaction.  Silently consuming every press here made
        # a quick Area/Cross action disappear whenever Agg was answering a
        # preceding wheel step.
        if self._selector_hold is not None:
            if hits_image or hits_rail:
                event.accept()
            else:
                super().mousePressEvent(event)
            return
        if (
            _image_interaction_is_pending(binding)
            and not (
                hits_image
                and event.button() in (
                    QtCore.Qt.LeftButton,
                    QtCore.Qt.RightButton,
                )
            )
        ):
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
            handle = _clim_handle_at(event.pos(), rail_target[0], rail_target[4])
            if handle is not None:
                self._begin_interaction_hold(_held_panel_from_target(target))
                binding.clim_drag = handle
                binding.clim_origin_limits = rail_target[4].color_limits
                binding.clim_candidate = rail_target[4].color_limits
                binding.clim_domain = _color_rail_domain(rail_target[4])
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
            binding.cross = sample
            self.crossSelected.emit(
                CrossGesture(
                    self._require_interaction_origin(
                        panel_id=binding.panel_id,
                        payload_type=(ImagePanelPayload, SiteMapPanelPayload),
                        hold=None,
                        kind="image cross",
                    ),
                    (
                        float(sample.x_coordinate),
                        float(sample.y_coordinate),
                    ),
                )
            )
            self.update()
            event.accept()
            return
        if event.button() == QtCore.Qt.MiddleButton:
            if binding.interaction_callback is None:
                super().mousePressEvent(event)
                return
            self._begin_interaction_hold(_held_panel_from_target(target))
            binding.pan_anchor = QtCore.QPointF(event.localPos())
            binding.pan_origin = self._viewport_for_target(binding, target)
            binding.pan_target_size = (
                max(1, target[0].width()),
                max(1, target[0].height()),
            )
            binding.pan_candidate = binding.pan_origin
            self.update()
            event.accept()
            return
        if event.button() != QtCore.Qt.LeftButton:
            super().mousePressEvent(event)
            return
        image_target = target[0]
        point = _normalized_point(event.localPos(), image_target, clamp=False)
        bounds = binding.draft_bounds or binding.applied_bounds
        visible_bounds = None
        if bounds is not None:
            try:
                visible_bounds = self._require_selector_viewport(
                    binding
                ).visible_bounds_for_full_bounds(bounds)
            except ValueError:
                pass
        handle = (
            None
            if visible_bounds is None
            else hit_rectangle_handle(
                _overlay_rect(visible_bounds, image_target),
                event.localPos(),
            )
        )
        binding.drag_prior_draft = binding.draft_bounds
        binding.rectangle_drag = (
            RectangleDrag.fresh(point)
            if handle is None or visible_bounds is None
            else RectangleDrag.begin(visible_bounds, handle, point)
        )
        self._begin_interaction_hold(_held_panel_from_target(target))
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        # No hover/data-cursor semantics.  Button state, not retained gesture
        # bookkeeping, is authoritative: an ordinary move is always inert.
        # The branches below therefore run only while a mouse button is held.
        if event.buttons() == QtCore.Qt.NoButton:
            event.accept()
            return
        numeric_binding = _active_numeric_binding(
            self._numeric_bindings, self._selector_hold
        )
        if (
            numeric_binding is not None
            and numeric_binding.threshold_drag is not None
        ):
            self._move_histogram_threshold(
                numeric_binding,
                event.localPos(),
            )
            self.update()
            event.accept()
            return
        if (
            numeric_binding is not None
            and numeric_binding.rectangle_drag is not None
        ):
            target = self._numeric_target(numeric_binding)
            if target is not None:
                viewport = target.payload.viewport
                point = _numeric_normalized_point(
                    target,
                    event.localPos(),
                    clamp_to_plot=True,
                )
                display_rect = numeric_binding.rectangle_drag.moved(
                    point,
                    clamp=viewport.plot_bounds,
                )
                first = viewport.widget_normalized_to_data(
                    display_rect[0],
                    display_rect[1],
                )
                second = viewport.widget_normalized_to_data(
                    display_rect[2],
                    display_rect[3],
                )
                numeric_binding.span_rect = (
                    first[0], first[1], second[0], second[1])
                numeric_binding.span_candidate = _span_data_candidate(
                    first[0], second[0])
                self.update()
            event.accept()
            return
        if (
            numeric_binding is not None
            and numeric_binding.pan_anchor is not None
            and numeric_binding.pan_origin is not None
        ):
            self._move_numeric_pan(numeric_binding, event.localPos())
            self.update()
            event.accept()
            return
        image_binding = _active_image_binding(
            self._image_bindings, self._selector_hold
        )
        if image_binding is not None and image_binding.clim_drag is not None:
            self._move_image_color_limit(image_binding, event.localPos())
            self.update()
            event.accept()
            return
        rectangle_drag = (
            None if image_binding is None else image_binding.rectangle_drag
        )
        if rectangle_drag is not None:
            target = self._selector_target(image_binding)
            if target is None:
                event.accept()
                return
            point = _normalized_point(event.localPos(), target[0], clamp=True)
            visible_bounds = rectangle_drag.moved(
                point,
                clamp=(0.0, 0.0, 1.0, 1.0),
            )
            if (
                visible_bounds[0] == visible_bounds[2]
                or visible_bounds[1] == visible_bounds[3]
            ):
                image_binding.draft_bounds = image_binding.drag_prior_draft
            else:
                image_binding.draft_bounds = _image_bounds_for_rectangle_drag(
                    self._require_selector_viewport(image_binding),
                    visible_bounds,
                )
            self.update()
            event.accept()
            return

        pan_anchor = None if image_binding is None else image_binding.pan_anchor
        pan_origin = None if image_binding is None else image_binding.pan_origin
        pan_size = None if image_binding is None else image_binding.pan_target_size
        if pan_anchor is not None and pan_origin is not None and pan_size is not None:
            self._move_image_pan(image_binding, event.localPos())
            self.update()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        numeric_binding = _active_numeric_binding(
            self._numeric_bindings, self._selector_hold
        )
        if (
            numeric_binding is not None
            and numeric_binding.threshold_drag is not None
            and event.button() == QtCore.Qt.LeftButton
        ):
            # Qt may deliver a release position not preceded by a motion.
            # Reuse the exact motion path so that final coordinate is never
            # lost; an unchanged candidate remains a no-op.
            self._move_histogram_threshold(
                numeric_binding,
                event.localPos(),
            )
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
            self._move_numeric_pan(numeric_binding, event.localPos())
            self._cancel_active_gesture(
                clear_image_draft=False,
                clear_numeric_spans=False,
            )
            self.update()
            event.accept()
            return
        if (
            numeric_binding is not None
            and numeric_binding.rectangle_drag is not None
            and event.button() == QtCore.Qt.LeftButton
        ):
            candidate = numeric_binding.span_candidate
            hold = self._selector_hold
            numeric_binding.rectangle_drag = None
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
        image_binding = _active_image_binding(
            self._image_bindings, self._selector_hold
        )
        if (
            image_binding is not None
            and image_binding.clim_drag is not None
            and event.button() == QtCore.Qt.LeftButton
        ):
            self._move_image_color_limit(image_binding, event.localPos())
            self._cancel_image_gesture(image_binding, clear_draft=False)
            self.update()
            event.accept()
            return
        if (
            image_binding is not None
            and image_binding.pan_anchor is not None
            and event.button() == QtCore.Qt.MiddleButton
        ):
            self._move_image_pan(image_binding, event.localPos())
            self._cancel_image_gesture(image_binding, clear_draft=False)
            self.update()
            event.accept()
            return
        rectangle_drag = (
            None if image_binding is None else image_binding.rectangle_drag
        )
        if rectangle_drag is None or event.button() != QtCore.Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        target = self._selector_target(image_binding)
        completed_bounds: NormalizedRectangle | None = None
        # ``RectangleDrag.fresh`` is the only drag whose initial rectangle is
        # degenerate.  That fact distinguishes a blank left click (clear Area)
        # from an unmoved click on a standing rectangle's handle/centre.
        fresh_drag = (
            rectangle_drag.initial[0] == rectangle_drag.initial[2]
            or rectangle_drag.initial[1] == rectangle_drag.initial[3]
        )
        if target is not None:
            point = _normalized_point(event.localPos(), target[0], clamp=True)
            visible_bounds = rectangle_drag.moved(
                point,
                clamp=(0.0, 0.0, 1.0, 1.0),
            )
            if (
                visible_bounds[0] == visible_bounds[2]
                or visible_bounds[1] == visible_bounds[3]
            ):
                image_binding.draft_bounds = (
                    None
                    if fresh_drag
                    else image_binding.drag_prior_draft
                )
            else:
                completed_bounds = _image_bounds_for_rectangle_drag(
                    self._require_selector_viewport(image_binding),
                    visible_bounds,
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
        image_binding.rectangle_drag = None
        try:
            if (
                (bounds is not None or fresh_drag)
                and hold is not None
                and callback is not None
            ):
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
            and _numeric_interaction_armed(self._selector_enabled, numeric_target.binding)
        ):
            binding = numeric_target.binding
            if self._selector_hold is not None:
                event.accept()
                return
            if binding.threshold_pending_answer is not None:
                event.accept()
                return
            delta = event.angleDelta().y()
            if delta == 0:
                super().wheelEvent(event)
                return
            point = _numeric_normalized_point(numeric_target, event.posF())
            # The painted front remains the exact command origin, while rapid
            # wheel steps accumulate on the newest authored viewport.  Making
            # the wheel wait for each Agg answer is what made zoom feel
            # proportional to render latency.
            viewport = binding.authored_viewport or numeric_target.payload.viewport
            anchor_x = viewport.widget_normalized_to_data(*point)[0]
            factor = 1.0 / 1.1 if delta < 0 else 1.1
            try:
                candidate = viewport.zoomed_x_limits(anchor_x, factor)
            except ValueError:
                candidate = None
            if candidate is not None:
                self._commit_numeric_viewport(binding, candidate)
            self.update()
            event.accept()
            return
        image_hit = self._image_target_at(event.posF(), include_rail=True)
        if image_hit is None:
            super().wheelEvent(event)
            return
        binding, target, rail_target = image_hit
        if (
            not _image_interaction_armed(self._selector_enabled, binding)
            or binding.interaction_callback is None
        ):
            super().wheelEvent(event)
            return
        position = event.pos()
        hits_image = target is not None and target[0].contains(position)
        hits_rail = rail_target is not None and rail_target[0].contains(position)
        # A colour-limit commit and an active drag are different authored
        # transactions, so keep those serialized.  A pending *viewport* is
        # deliberately replaceable: the next wheel step accumulates on it and
        # the latest-only render lane answers the newest intent.
        if binding.pending_color_limits is not None or self._selector_hold is not None:
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
        point = _normalized_point(event.posF(), target[0], clamp=False)
        # Preserve the established lab convention: wheel DOWN zooms in and
        # wheel UP zooms out.
        scale = 1.0 / 1.1 if delta < 0 else 1.1
        candidate = self._viewport_for_target(binding, target).centered_zoom(
            point,
            scale,
        )
        self._commit_viewport(binding, candidate)
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
            and _numeric_interaction_armed(self._selector_enabled, numeric_target.binding)
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
                self.update()
                event.accept()
                return
            if (
                _numeric_interaction_is_pending(binding)
                or self._selector_hold is not None
            ):
                event.accept()
                return
            if event.button() == QtCore.Qt.RightButton:
                binding.cross = None
                self.crossSelected.emit(
                    CrossGesture(
                        self._numeric_interaction_origin(binding),
                        None,
                    )
                )
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
        if not _image_interaction_armed(self._selector_enabled, binding):
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
            candidate = viewport.home() if area is None else viewport.with_visible_bounds(area)
            self._commit_viewport(binding, candidate)
            self.update()
            event.accept()
            return
        if _image_interaction_is_pending(binding) or self._selector_hold is not None:
            if hits_image or hits_rail:
                event.accept()
            else:
                super().mouseDoubleClickEvent(event)
            return
        if not hits_image:
            super().mouseDoubleClickEvent(event)
            return
        if event.button() == QtCore.Qt.RightButton:
            binding.cross = None
            self.crossSelected.emit(
                CrossGesture(
                    self._require_interaction_origin(
                        panel_id=binding.panel_id,
                        payload_type=(ImagePanelPayload, SiteMapPanelPayload),
                        hold=None,
                        kind="image cross",
                    ),
                    None,
                )
            )
            self.update()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

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

    def _require_selector_viewport(
        self,
        binding: _ImagePanelBinding,
    ) -> ImageViewportTransform:
        """Return the viewport belonging to the exact painted image front.

        A board front may advance while one selector gesture keeps its panel
        payload held.  ``binding.viewport`` tracks the admitted front and is
        therefore only authoritative for untyped raster panels, which have
        no typed viewport in either the held or current painted payload.
        """

        payload, _origin = self._visible_display(
            binding.panel_id,
            (ImagePanelPayload, SiteMapPanelPayload),
        )
        if isinstance(payload, SiteMapPanelPayload):
            return payload.background.viewport
        if isinstance(payload, ImagePanelPayload):
            return payload.viewport
        return binding.viewport

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
    def _selector_target(self, binding: _ImagePanelBinding | None = None):
        if binding is None:
            binding = self._image_binding()
        return _selector_target(
            binding,
            widget_rect=self.rect(),
            panel_ids=self._panel_ids,
            columns=self._columns,
            front=self._front,
            hold=self._selector_hold,
        )

    def _image_target_at(
        self,
        point: QtCore.QPointF,
        *,
        include_rail: bool = False,
    ):
        return _image_target_at(
            point,
            include_rail=include_rail,
            widget_rect=self.rect(),
            panel_ids=self._panel_ids,
            columns=self._columns,
            front=self._front,
            hold=self._selector_hold,
            bindings=self._image_bindings,
        )

    def _painted_image_panel_id_at(self, point: QtCore.QPointF) -> str | None:
        return _painted_image_panel_id_at(
            point,
            widget_rect=self.rect(),
            columns=self._columns,
            front=self._front,
        )

    def _numeric_target(
        self,
        binding: _NumericPanelBinding,
    ) -> _NumericTarget | None:
        return _numeric_target(
            widget_rect=self.rect(),
            panel_ids=self._panel_ids,
            columns=self._columns,
            front=self._front,
            hold=self._selector_hold,
            binding=binding,
        )

    def _numeric_target_at(self, point: QtCore.QPointF) -> _NumericTarget | None:
        return _numeric_target_at(
            point,
            widget_rect=self.rect(),
            panel_ids=self._panel_ids,
            columns=self._columns,
            front=self._front,
            hold=self._selector_hold,
            bindings=self._numeric_bindings,
        )

    def _clim_rail_target(self, binding: _ImagePanelBinding):
        return _clim_rail_target(
            binding,
            widget_rect=self.rect(),
            panel_ids=self._panel_ids,
            columns=self._columns,
            front=self._front,
            hold=self._selector_hold,
        )

    def _viewport_for_target(
        self,
        binding: _ImagePanelBinding,
        target,
    ) -> ImageViewportTransform:
        return _viewport_for_target(binding, target, self._selector_hold)

    def _sample_for_target(
        self,
        target,
        point: QtCore.QPointF,
    ) -> _ImageSample | None:
        return _sample_for_target(
            target,
            point,
            hold=self._selector_hold,
        )

    def _move_histogram_threshold(
        self,
        binding: _NumericPanelBinding,
        position: QtCore.QPointF,
    ) -> None:
        """Advance one live threshold drag through the shared exact front."""

        target = self._numeric_target(binding)
        if target is None or binding.threshold_drag is None:
            return
        viewport = target.payload.viewport
        point = _numeric_normalized_point(
            target,
            position,
            clamp_to_plot=True,
        )
        moved = viewport.widget_normalized_to_data(*point)
        base = list(binding.threshold_candidate or target.payload.thresholds)
        index = binding.threshold_drag
        if not (0 <= index < len(base) and math.isfinite(moved[0])):
            return
        base[index] = float(moved[0])
        candidate = tuple(base)
        if candidate == binding.threshold_candidate:
            return
        binding.threshold_candidate = candidate
        self._commit_histogram_thresholds(
            binding,
            candidate,
            hold=self._selector_hold,
        )

    def _move_numeric_pan(
        self,
        binding: _NumericPanelBinding,
        position: QtCore.QPointF,
    ) -> None:
        """Advance one live numeric pan from its immutable press transform."""

        target = self._numeric_target(binding)
        if (
            target is None
            or binding.pan_anchor is None
            or binding.pan_origin is None
        ):
            return
        point = _numeric_normalized_point(target, position)
        try:
            candidate = binding.pan_origin.panned_x_limits(
                binding.pan_anchor,
                point[0],
                start_x_limits=binding.pan_origin.x_limits,
            )
        except ValueError:
            binding.pan_candidate = None
            return
        binding.pan_candidate = candidate
        self._commit_numeric_viewport(
            binding,
            candidate,
            hold=self._selector_hold,
        )

    def _move_image_color_limit(
        self,
        binding: _ImagePanelBinding,
        position: QtCore.QPointF,
    ) -> None:
        """Advance one live colour-rail drag through the same panel owner."""

        rail_target = self._clim_rail_target(binding)
        if (
            rail_target is None
            or binding.clim_drag is None
            or binding.clim_origin_limits is None
            or binding.clim_domain is None
        ):
            return
        value = _rail_value(
            float(position.y()),
            binding.clim_domain,
            rail_target[0],
        )
        low, high = binding.clim_origin_limits
        if binding.clim_drag == "low":
            low = min(value, math.nextafter(high, -math.inf))
        else:
            high = max(value, math.nextafter(low, math.inf))
        binding.clim_candidate = (low, high)
        self._commit_color_limits(
            binding,
            binding.clim_candidate,
            hold=self._selector_hold,
        )

    def _move_image_pan(
        self,
        binding: _ImagePanelBinding,
        position: QtCore.QPointF,
    ) -> None:
        """Advance one live image pan from its immutable press transform."""

        anchor = binding.pan_anchor
        origin = binding.pan_origin
        target_size = binding.pan_target_size
        if anchor is None or origin is None or target_size is None:
            return
        delta = (
            float(position.x() - anchor.x()),
            float(position.y() - anchor.y()),
        )
        candidate = origin.panned_by_pixels(delta, target_size)
        binding.pan_candidate = candidate
        self._commit_viewport(
            binding,
            candidate,
            hold=self._selector_hold,
        )

    def _held_origin_for_panel(self, panel_id: str) -> _HeldPanelFront | None:
        hold = self._selector_hold
        return (
            hold
            if hold is not None and hold.panel_id == panel_id
            else None
        )

    def _author_queued_image_viewport(
        self,
        binding: _ImagePanelBinding,
        bounds: NormalizedRectangle,
    ) -> bool:
        return self._commit_viewport(
            binding,
            binding.viewport.with_visible_bounds(bounds),
            hold=self._held_origin_for_panel(binding.panel_id),
        )

    def _author_queued_image_color(
        self,
        binding: _ImagePanelBinding,
        limits: tuple[float, float],
    ) -> bool:
        return self._commit_color_limits(
            binding,
            limits,
            hold=self._held_origin_for_panel(binding.panel_id),
        )

    def _author_queued_numeric_viewport(
        self,
        binding: _NumericPanelBinding,
        limits: tuple[float, float],
    ) -> bool:
        return self._commit_numeric_viewport(
            binding,
            limits,
            hold=self._held_origin_for_panel(binding.panel_id),
        )

    def _author_queued_histogram_thresholds(
        self,
        binding: _NumericPanelBinding,
        thresholds: tuple[float, ...],
    ) -> bool:
        return self._commit_histogram_thresholds(
            binding,
            thresholds,
            hold=self._held_origin_for_panel(binding.panel_id),
        )

    def _commit_viewport(
        self,
        binding: _ImagePanelBinding,
        candidate: ImageViewportTransform,
        *,
        hold: _HeldPanelFront | None = None,
    ) -> bool:
        return _commit_image_viewport(
            binding,
            candidate,
            front=self._front,
            panel_ids=self._panel_ids,
            hold=hold,
            painted_hold=self._selector_hold,
        )

    def _commit_histogram_thresholds(
        self,
        binding: _NumericPanelBinding,
        thresholds: tuple[float, ...],
        *,
        hold: _HeldPanelFront | None = None,
    ) -> bool:
        """Author one threshold motion and reserve its display revision."""

        return _commit_numeric_thresholds(
            binding,
            thresholds,
            front=self._front,
            panel_ids=self._panel_ids,
            hold=hold,
            painted_hold=self._selector_hold,
        )

    def _commit_numeric_viewport(
        self,
        binding: _NumericPanelBinding,
        x_limits: tuple[float, float],
        *,
        hold: _HeldPanelFront | None = None,
    ) -> bool:
        return _commit_numeric_x_viewport(
            binding,
            x_limits,
            front=self._front,
            panel_ids=self._panel_ids,
            hold=hold,
            painted_hold=self._selector_hold,
        )

    def _commit_color_limits(
        self,
        binding: _ImagePanelBinding,
        limits: tuple[float, float],
        *,
        hold: _HeldPanelFront | None,
    ) -> bool:
        return _commit_image_color_limits(
            binding,
            limits,
            front=self._front,
            panel_ids=self._panel_ids,
            hold=hold,
            painted_hold=self._selector_hold,
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
    def _cancel_image_gesture(
        self,
        binding: _ImagePanelBinding,
        *,
        clear_draft: bool,
    ) -> None:
        _cancel_image_gesture(binding, clear_draft=clear_draft)
        if (
            self._selector_hold is not None
            and self._selector_hold.panel_id == binding.panel_id
        ):
            self._finish_interaction_hold()

    def _clear_image_transient(
        self,
        binding: _ImagePanelBinding,
        *,
        clear_applied_bounds: bool,
        clear_pending: bool,
    ) -> None:
        _clear_image_transient(
            binding,
            clear_applied_bounds=clear_applied_bounds,
            clear_pending=clear_pending,
        )
        if (
            self._selector_hold is not None
            and self._selector_hold.panel_id == binding.panel_id
        ):
            self._finish_interaction_hold()

    def _cancel_numeric_gesture(
        self,
        binding: _NumericPanelBinding,
        *,
        clear_span: bool,
    ) -> None:
        _cancel_numeric_gesture(binding, clear_span=clear_span)
        if (
            self._selector_hold is not None
            and self._selector_hold.panel_id == binding.panel_id
        ):
            self._finish_interaction_hold()

    def _clear_numeric_transient(
        self,
        binding: _NumericPanelBinding,
        *,
        clear_applied_span: bool,
        clear_pending: bool,
    ) -> None:
        _clear_numeric_transient(
            binding,
            clear_applied_span=clear_applied_span,
            clear_pending=clear_pending,
        )
        if (
            self._selector_hold is not None
            and self._selector_hold.panel_id == binding.panel_id
        ):
            self._finish_interaction_hold()

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
        self._finish_interaction_hold()

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
