"""Reusable Qt lifecycle for one revisioned exact-key form draft.

The host supplies the semantic identity, revision, complete form projection,
and optional runtime placeholder text.  This widget deliberately interprets
none of them: it owns only the two-surface draft lifecycle and emits an
optimistic Apply request back to the host.
"""

from __future__ import annotations

from collections.abc import Mapping

from PyQt5 import QtCore, QtWidgets

from ..form import FormSpec
from .form import FluentParameterForm
from .fluent import (
    FluentButton,
    FluentScrollArea,
    FluentStatusStrip,
    GREEN,
    GREY,
    scaled_px,
)


_UNSET = object()


def _revision(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _semantic_equal(left: object, right: object) -> bool:
    """Compare host identities without accepting ambiguous array equality."""

    if type(left) is not type(right):
        return False
    result = left == right
    if not isinstance(result, bool):
        raise TypeError("semantic identity equality must return bool")
    return result


class FluentRevisionedFormEditor(QtWidgets.QWidget):
    """One directly parameterized optimistic form editor.

    ``base_revision`` always names the owner state from which the visible draft
    was created.  A newer owner revision replaces a clean draft, but leaves a
    dirty draft untouched and marks it stale.  Cancel only asks the host to
    reload; this widget never chooses or commits semantic state itself.
    """

    applyRequested = QtCore.pyqtSignal(int, object)
    cancelRequested = QtCore.pyqtSignal()

    def __init__(
        self,
        spec: FormSpec,
        surface_noun: str,
        *,
        runtime_placeholder_fields: tuple[str, ...] = (),
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if not isinstance(spec, FormSpec):
            raise TypeError("spec must be FormSpec")
        if not isinstance(surface_noun, str) or not surface_noun:
            raise ValueError("surface_noun must be a non-empty string")
        if surface_noun[0].isspace() or surface_noun[-1].isspace():
            raise ValueError("surface_noun cannot have surrounding whitespace")
        noun = surface_noun
        fields = tuple(runtime_placeholder_fields)
        if any(not isinstance(key, str) or not key for key in fields):
            raise TypeError("runtime placeholder field names must be non-empty strings")
        if len(fields) != len(set(fields)):
            raise ValueError("runtime placeholder field names must be unique")
        unknown = tuple(key for key in fields if key not in spec.keys)
        if unknown:
            raise ValueError(f"runtime placeholder fields are absent from form: {unknown!r}")

        super().__init__(parent)
        self._surface_noun = noun
        self._surface_label = noun[0].upper() + noun[1:]
        self._base_revision: int | None = None
        self._base_semantic_identity: object = _UNSET
        self._latest_revision: int | None = None
        self._latest_semantic_identity: object = _UNSET
        self._dirty = False
        self._stale = False
        self._reload_after_cancel = False

        self._form = FluentParameterForm(spec)
        self._form.changed.connect(self._on_form_changed)
        self._runtime_placeholder_fields = fields
        self._default_placeholders: dict[str, str] = {}
        for key in fields:
            widget = self._form.widget_for(key)
            if not isinstance(widget, QtWidgets.QLineEdit):
                raise TypeError(
                    f"runtime placeholder field {key!r} must project to QLineEdit"
                )
            self._default_placeholders[key] = widget.placeholderText()
        self._runtime_placeholders = dict(self._default_placeholders)

        self._scroll = FluentScrollArea(self)
        self._scroll.setWidget(self._form)

        self._status = FluentStatusStrip(self)
        self._status.show_message(
            f"No {self._surface_noun} state loaded.",
            severity="info",
        )

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
    def base_revision(self) -> int | None:
        return self._base_revision

    def load(
        self,
        *,
        revision: int,
        semantic_identity: object,
        values: Mapping[str, object],
        runtime_placeholders: Mapping[str, str] | None = None,
    ) -> None:
        """Observe one owner state and project it when the draft may be replaced."""

        revision = _revision(revision, "revision")
        values = self._exact_values(values)
        placeholders = self._prepare_runtime_placeholders(runtime_placeholders)
        self._validate_observation(revision, semantic_identity)

        replace_draft = (
            self._base_revision is None
            or self._reload_after_cancel
            or (not self._dirty and revision > self._base_revision)
        )
        if replace_draft:
            self._replace_draft(revision, semantic_identity, values)
        else:
            self._record_observation(revision, semantic_identity)
            assert self._base_revision is not None
            if revision > self._base_revision:
                self._stale = True
                self._show_stale()

        self._install_runtime_placeholders(placeholders)

    def accept_commit(
        self,
        *,
        base_revision: int,
        revision: int,
        semantic_identity: object,
        values: Mapping[str, object],
        runtime_placeholders: Mapping[str, str] | None = None,
    ) -> None:
        """Acknowledge an exact no-op or one-step commit accepted by the host."""

        base_revision = _revision(base_revision, "base_revision")
        revision = _revision(revision, "revision")
        values = self._exact_values(values)
        placeholders = self._prepare_runtime_placeholders(runtime_placeholders)
        if self._base_revision != base_revision:
            raise ValueError("accepted commit has another base revision")
        if revision not in (base_revision, base_revision + 1):
            raise ValueError("accepted commit must be a no-op or advance once")
        if revision == base_revision:
            if self._base_semantic_identity is _UNSET or not _semantic_equal(
                semantic_identity,
                self._base_semantic_identity,
            ):
                raise ValueError("accepted no-op conflicts with the loaded semantic state")
        self._validate_observation(revision, semantic_identity)
        self._replace_draft(revision, semantic_identity, values)
        self._install_runtime_placeholders(placeholders)

    def _replace_draft(
        self,
        revision: int,
        semantic_identity: object,
        values: Mapping[str, object],
    ) -> None:
        # FluentParameterForm.populate validates every exact key before changing
        # the first widget, so a rejected owner projection cannot partially load.
        self._form.populate(values)
        self._record_observation(revision, semantic_identity)
        self._base_revision = revision
        self._base_semantic_identity = semantic_identity
        self._dirty = False
        self._stale = False
        self._reload_after_cancel = False
        self._apply_button.setEnabled(True)
        self._status.show_message(
            f"Loaded {self._surface_noun} revision {revision}.",
            severity="info",
        )

    def _validate_observation(
        self,
        revision: int,
        semantic_identity: object,
    ) -> None:
        latest = self._latest_revision
        if latest is None:
            return
        if revision < latest:
            raise ValueError(f"{self._surface_noun} revision cannot move backwards")
        if revision == latest:
            if self._latest_semantic_identity is _UNSET or not _semantic_equal(
                semantic_identity,
                self._latest_semantic_identity,
            ):
                raise ValueError(
                    f"one {self._surface_noun} revision has conflicting semantic state"
                )

    def _record_observation(
        self,
        revision: int,
        semantic_identity: object,
    ) -> None:
        if self._latest_revision is None or revision > self._latest_revision:
            self._latest_revision = revision
            self._latest_semantic_identity = semantic_identity

    def _prepare_runtime_placeholders(
        self,
        runtime_placeholders: Mapping[str, str] | None,
    ) -> dict[str, str]:
        prepared = dict(self._default_placeholders)
        if runtime_placeholders is None:
            return prepared
        if not isinstance(runtime_placeholders, Mapping):
            raise TypeError("runtime_placeholders must be a mapping or None")
        unknown = tuple(
            key
            for key in runtime_placeholders
            if key not in self._runtime_placeholder_fields
        )
        if unknown:
            raise ValueError(f"runtime placeholder keys are not admitted: {unknown!r}")
        for key, placeholder in runtime_placeholders.items():
            if not isinstance(placeholder, str):
                raise TypeError(f"runtime placeholder {key!r} must be str")
            prepared[key] = placeholder
        return prepared

    def _exact_values(
        self,
        values: Mapping[str, object],
    ) -> dict[str, object]:
        if not isinstance(values, Mapping):
            raise TypeError("form values must be a mapping")
        supplied = set(values)
        expected = set(self._form.keys)
        if supplied != expected:
            missing = sorted(repr(key) for key in expected - supplied)
            extra = sorted(repr(key) for key in supplied - expected)
            raise ValueError(
                "form values must have exact keys; "
                f"missing={missing}, extra={extra}"
            )
        return {key: values[key] for key in self._form.keys}

    def _install_runtime_placeholders(self, placeholders: dict[str, str]) -> None:
        self._runtime_placeholders = placeholders
        self._refresh_runtime_placeholders()

    def _refresh_runtime_placeholders(self) -> None:
        for key, placeholder in self._runtime_placeholders.items():
            widget = self._form.widget_for(key)
            assert isinstance(widget, QtWidgets.QLineEdit)
            if not widget.text():
                widget.setPlaceholderText(placeholder)

    def _show_stale(self) -> None:
        self._status.show_message(
            f"{self._surface_label} changed externally; this draft is stale.",
            severity="warning",
        )

    def _on_form_changed(self, key: str) -> None:
        self._dirty = True
        self._reload_after_cancel = False
        self._refresh_runtime_placeholders()
        if self._stale:
            self._show_stale()
        else:
            self._status.show_message(
                f"Edited {key}; {self._surface_noun} changes are not applied.",
                severity="info",
            )

    def _request_apply(self) -> None:
        if self._base_revision is None:
            self._status.show_message(
                f"Load a {self._surface_noun} state before applying.",
                severity="error",
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
        self._reload_after_cancel = True
        self.cancelRequested.emit()


__all__ = ["FluentRevisionedFormEditor"]
