"""Headless installation-config editing for the current DeviceManager.

The session owns only an operator's local draft and its disk baseline.  It does
not open devices, observe a runtime, poll, or turn ordinary field edits into a
whole-application projection.
"""

from __future__ import annotations

import os
from pathlib import Path

from zlc_frontend.form import FormSpec, project_authoring_form
from zlc_neutral_atom.installation_config import (
    InstallationConfigDocument,
    default_installation_authoring_schema,
    installation_authoring_schema,
)
from zlc_storage import sha256_text

def form_spec(
    document_or_backend: InstallationConfigDocument | str,
) -> FormSpec:
    """Project one current config topology to the shared headless form contract."""

    if isinstance(document_or_backend, InstallationConfigDocument):
        schema = installation_authoring_schema(document_or_backend)
    elif isinstance(document_or_backend, str):
        schema = default_installation_authoring_schema(document_or_backend)
    else:
        raise TypeError(
            "document_or_backend must be InstallationConfigDocument or backend text"
        )

    return project_authoring_form(schema)


class DeviceConfigEditorSession:
    """One local config draft with an explicit saved and active baseline."""

    __slots__ = (
        "_backend",
        "_values",
        "_baseline_backend",
        "_baseline_values",
        "_active_backend",
        "_active_values",
        "_path",
        "_baseline_digest",
    )

    def __init__(
        self,
        document: InstallationConfigDocument,
        active_document: InstallationConfigDocument | None = None,
        path: str | os.PathLike[str] | None = None,
        baseline_digest: str | None = None,
    ) -> None:
        _require_document(document, "document")
        if active_document is not None:
            _require_document(active_document, "active_document")
        digest = _digest_or_none(baseline_digest)
        if digest is not None and digest != document.content_digest:
            raise ValueError("baseline_digest differs from the supplied document")

        backend, values = _document_state(document)
        self._backend = backend
        self._values = values
        self._baseline_backend = backend
        self._baseline_values = dict(values)
        if active_document is None:
            self._active_backend = None
            self._active_values = None
        else:
            active_backend, active_values = _document_state(active_document)
            self._active_backend = active_backend
            self._active_values = active_values
        self._path = _path_or_none(path)
        self._baseline_digest = digest

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def values(self) -> dict[str, object]:
        return dict(self._values)

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def baseline_digest(self) -> str | None:
        return self._baseline_digest

    @property
    def dirty(self) -> bool:
        return not _same_state(
            self._backend,
            self._values,
            self._baseline_backend,
            self._baseline_values,
        )

    @property
    def restart_required(self) -> bool:
        if self._active_backend is None or self._active_values is None:
            return False
        return not _same_state(
            self._backend,
            self._values,
            self._active_backend,
            self._active_values,
        )

    def set_field(self, key: str, value: object) -> bool:
        """Apply one local field delta without constructing a domain document."""

        if not isinstance(key, str):
            raise TypeError("config field key must be text")
        if key not in self._values:
            raise KeyError(
                f"field {key!r} does not belong to backend {self._backend!r}"
            )
        previous = self._values[key]
        if _typed_equal(previous, value):
            return False
        self._values[key] = value
        return True

    def switch_backend(self, backend: str) -> bool:
        """Explicitly replace the config topology and seed its declared defaults."""

        form = form_spec(backend)
        if backend == self._backend:
            return False
        self._backend = backend
        self._values = form.default_values()
        return True

    def replace_new(self, backend: str) -> None:
        """Start one untitled editor generation for ``backend``.

        ``New`` is a document boundary, not a backend-field edit.  It therefore
        forgets the previous file/CAS baseline even when the requested backend
        is unchanged.  The active installation baseline is deliberately kept:
        it is an independent runtime fact used only for restart-required
        presentation.
        """

        values = form_spec(backend).default_values()
        self._backend = backend
        self._values = values
        self._baseline_backend = backend
        self._baseline_values = dict(values)
        self._path = None
        self._baseline_digest = None

    def candidate(self) -> InstallationConfigDocument:
        """Construct and authoritatively validate the current draft."""

        return InstallationConfigDocument.from_parameters(
            self._backend,
            self._values,
        )

    def replace_loaded(
        self,
        document: InstallationConfigDocument,
        path: str | os.PathLike[str],
        digest: str,
    ) -> None:
        """Replace the editor generation with one successfully loaded document."""

        _require_document(document, "document")
        checked_digest = _matching_digest(document, digest)
        backend, values = _document_state(document)
        self._backend = backend
        self._values = values
        self._baseline_backend = backend
        self._baseline_values = dict(values)
        self._path = _path(path)
        self._baseline_digest = checked_digest

    def mark_saved(
        self,
        path: str | os.PathLike[str],
        digest: str,
    ) -> None:
        """Accept a successful atomic save as the new disk baseline."""

        candidate = self.candidate()
        checked_digest = _matching_digest(candidate, digest)
        self._baseline_backend = self._backend
        self._baseline_values = dict(self._values)
        self._path = _path(path)
        self._baseline_digest = checked_digest

    def set_active_document(
        self,
        document: InstallationConfigDocument | None,
    ) -> None:
        """Record the installation generation after a real lifecycle change."""

        if document is None:
            self._active_backend = None
            self._active_values = None
            return
        _require_document(document, "document")
        backend, values = _document_state(document)
        self._active_backend = backend
        self._active_values = values

    def cancel(self) -> None:
        """Discard the local draft and restore the exact saved/loaded baseline."""

        self._backend = self._baseline_backend
        self._values = dict(self._baseline_values)


def _document_state(
    document: InstallationConfigDocument,
) -> tuple[str, dict[str, object]]:
    return document.backend, document.parameters


def _require_document(value: object, field: str) -> None:
    if not isinstance(value, InstallationConfigDocument):
        raise TypeError(f"{field} must be InstallationConfigDocument")


def _digest_or_none(value: str | None) -> str | None:
    return None if value is None else sha256_text(value, "baseline_digest")


def _matching_digest(
    document: InstallationConfigDocument,
    value: str,
) -> str:
    digest = sha256_text(value, "digest")
    assert digest is not None
    if digest != document.content_digest:
        raise ValueError("digest differs from the current config document")
    return digest


def _path_or_none(
    value: str | os.PathLike[str] | None,
) -> Path | None:
    return None if value is None else _path(value)


def _path(value: str | os.PathLike[str]) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError("config path must be path-like")
    return Path(value).expanduser().resolve()


def _typed_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and bool(left == right)


def _same_state(
    left_backend: str,
    left_values: dict[str, object],
    right_backend: str,
    right_values: dict[str, object],
) -> bool:
    if left_backend != right_backend or set(left_values) != set(right_values):
        return False
    return all(
        _typed_equal(left_values[key], right_values[key]) for key in left_values
    )


__all__ = [
    "DeviceConfigEditorSession",
    "form_spec",
]
