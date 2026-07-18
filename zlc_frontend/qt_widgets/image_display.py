"""Reusable Qt editor for one authored image-display draft.

The widget deliberately stops at the GUI boundary.  It owns neither an
``ImageDisplayState`` nor a commit workflow: the owner loads an immutable
state, receives an exact form draft tagged with its base revision, and decides
whether and how to commit it.  Runtime colour limits are presentation hints
only and never replace authored text.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from ..image_display import (
    ImageDisplayState,
    ImageRange,
    image_display_form_spec,
    image_display_form_values,
    validated_image_range,
)
from .form import FluentParameterForm
from .fluent import (
    FluentButton,
    FluentScrollArea,
    FluentStatusStrip,
    GREEN,
    GREY,
    scaled_px,
)


def _validated_runtime_limits(value: ImageRange | None) -> ImageRange | None:
    if value is None:
        return None
    return validated_image_range(value, "runtime_color_limits")


class FluentImageDisplayEditor(QtWidgets.QWidget):
    """Scrollable exact-key image display editor with an optimistic base tag.

    ``load`` replaces the complete form while the draft is clean.  If a newer
    external revision arrives during an edit, every authored widget value is
    left untouched and the draft is marked stale.  Apply still emits the
    original base revision so the owner remains the sole concurrency/commit
    authority.
    """

    applyRequested = QtCore.pyqtSignal(int, object)
    cancelRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._base_revision: int | None = None
        self._loaded_state: ImageDisplayState | None = None
        self._dirty = False
        self._stale = False
        self._reload_after_cancel = False
        self._runtime_color_limits: ImageRange | None = None

        spec = image_display_form_spec()
        self._form = FluentParameterForm(spec)
        self._form.changed.connect(self._on_form_changed)

        self._scroll = FluentScrollArea(self)
        self._scroll.setWidget(self._form)

        self._status = FluentStatusStrip(self)
        self._status.show_message("No image display state loaded.", severity="info")

        self._cancel_button = FluentButton("Cancel", self, color=GREY)
        self._apply_button = FluentButton("Apply", self, color=GREEN)
        self._cancel_button.clicked.connect(self._request_cancel)
        self._apply_button.clicked.connect(self._request_apply)
        self._apply_button.setEnabled(False)

        footer = QtWidgets.QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(scaled_px(8, minimum=6))
        footer.addStretch(1)
        footer.addWidget(self._cancel_button)
        footer.addWidget(self._apply_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(scaled_px(8, minimum=6))
        layout.addWidget(self._scroll, 1)
        layout.addWidget(self._status)
        layout.addLayout(footer)

    @property
    def form(self) -> FluentParameterForm:
        """The shared exact-key form projection, exposed for host composition."""

        return self._form

    @property
    def base_revision(self) -> int | None:
        return self._base_revision

    @property
    def dirty(self) -> bool:
        return self._dirty

    @property
    def stale(self) -> bool:
        return self._stale

    @property
    def status_text(self) -> str:
        return self._status.message.text()

    @property
    def status_severity(self) -> str:
        return self._status.severity

    @property
    def apply_button(self) -> FluentButton:
        return self._apply_button

    @property
    def cancel_button(self) -> FluentButton:
        return self._cancel_button

    def read_all(self) -> dict[str, object]:
        """Read the exact draft without interpreting or committing it."""

        return self._form.read_all()

    def load(
        self,
        state: ImageDisplayState,
        runtime_color_limits: ImageRange | None = None,
    ) -> None:
        """Load an owner-supplied state or observe a newer external revision.

        The first load, any clean load, and the load explicitly requested by a
        Cancel replace all fields atomically.  A normal load received while a
        draft is dirty never overwrites the draft.  Advancing revisions mark it
        stale while retaining the original optimistic base revision.
        """

        if not isinstance(state, ImageDisplayState):
            raise TypeError("state must be ImageDisplayState")
        limits = _validated_runtime_limits(runtime_color_limits)
        if (
            self._base_revision is not None
            and state.revision < self._base_revision
        ):
            raise ValueError("image display revision cannot move backwards")
        if (
            self._base_revision == state.revision
            and self._loaded_state is not None
            and self._loaded_state != state
        ):
            raise ValueError("one image display revision has conflicting state")

        replace_draft = (
            self._base_revision is None
            or self._reload_after_cancel
            or (not self._dirty and state.revision > self._base_revision)
        )
        if replace_draft:
            self._replace_draft(state)
        elif state.revision > self._base_revision:
            self._stale = True
            self._status.show_message(
                "Image display changed externally; this draft is stale.",
                severity="warning",
            )

        self.mark_runtime_color_limits(limits)

    def accept_commit(
        self,
        base_revision: int,
        state: ImageDisplayState,
        runtime_color_limits: ImageRange | None = None,
    ) -> None:
        """Acknowledge the exact draft this editor emitted.

        Ordinary :meth:`load` intentionally preserves a dirty draft when an
        external revision arrives.  The owner needs one distinct operation
        after it has accepted *this* editor's optimistic commit; otherwise the
        submitting surface would incorrectly mark its own successful edit as
        stale.  A semantic no-op keeps the base revision, while a real display
        edit advances it exactly once.
        """

        if isinstance(base_revision, bool) or not isinstance(base_revision, int):
            raise TypeError("base_revision must be an integer")
        if not isinstance(state, ImageDisplayState):
            raise TypeError("state must be ImageDisplayState")
        limits = _validated_runtime_limits(runtime_color_limits)
        if self._base_revision != base_revision:
            raise ValueError("accepted image display commit has another base revision")
        if state.revision not in (base_revision, base_revision + 1):
            raise ValueError(
                "accepted image display commit must be a no-op or advance once"
            )
        if (
            state.revision == base_revision
            and self._loaded_state is not None
            and state != self._loaded_state
        ):
            raise ValueError("image display no-op conflicts with the loaded state")
        self._replace_draft(state)
        self.mark_runtime_color_limits(limits)

    def _replace_draft(self, state: ImageDisplayState) -> None:
        self._form.populate(image_display_form_values(state))
        self._base_revision = state.revision
        self._loaded_state = state
        self._dirty = False
        self._stale = False
        self._reload_after_cancel = False
        self._apply_button.setEnabled(True)
        self._status.show_message(
            f"Loaded image display revision {state.revision}.",
            severity="info",
        )

    def mark_runtime_color_limits(self, limits: ImageRange | None) -> None:
        """Show runtime limits only as placeholders in empty colour fields."""

        self._runtime_color_limits = _validated_runtime_limits(limits)
        self._refresh_color_placeholders()

    def _refresh_color_placeholders(self) -> None:
        placeholders = (
            ("(optional)", "(optional)")
            if self._runtime_color_limits is None
            else tuple(repr(value) for value in self._runtime_color_limits)
        )
        for key, placeholder in zip(
            ("color_min", "color_max"), placeholders, strict=True
        ):
            widget = self._form.widget_for(key)
            if not isinstance(widget, QtWidgets.QLineEdit):
                raise TypeError(f"image display field {key!r} must be a line edit")
            if not widget.text():
                widget.setPlaceholderText(placeholder)

    def _on_form_changed(self, key: str) -> None:
        self._dirty = True
        self._reload_after_cancel = False
        self._refresh_color_placeholders()
        if self._stale:
            self._status.show_message(
                "Image display changed externally; this draft is stale.",
                severity="warning",
            )
        else:
            self._status.show_message(
                f"Edited {key}; changes are not applied.", severity="info"
            )

    def _request_apply(self) -> None:
        if self._base_revision is None:
            self._status.show_message(
                "Load an image display state before applying.", severity="error"
            )
            return
        try:
            values = self._form.read_all()
        except (TypeError, ValueError) as exc:
            self._status.show_message(str(exc), severity="error")
            return
        self._status.show_message(
            f"Apply requested from revision {self._base_revision}.",
            severity="warning" if self._stale else "info",
        )
        self.applyRequested.emit(self._base_revision, values)

    def _request_cancel(self) -> None:
        # The owner remains responsible for choosing and loading the state.  The
        # flag merely lets that explicit response replace a dirty/stale draft.
        self._reload_after_cancel = True
        self.cancelRequested.emit()


__all__ = ["FluentImageDisplayEditor"]
