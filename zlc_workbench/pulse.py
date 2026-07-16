"""Headless PulseGUI editor state and exact preview orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading

from zlc_pulse import (
    PulseDocument,
    PulseExecutionForm,
    PulseTarget,
    PulseTimelineDocument,
    bind_pulse_document_target,
    build_pulse_timeline,
    compile_pulse_artifact,
    load_pulse_document,
    new_pulse_document,
    pulse_document_path,
    save_pulse_document,
)


class PulseEditorSession:
    """The only mutable owner of one editor's current immutable document."""

    __slots__ = (
        "_base_fingerprint",
        "_document",
        "_lock",
        "_path",
        "_revision",
        "_save_lock",
    )

    def __init__(
        self,
        document: PulseDocument,
    ) -> None:
        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        self._document = document
        self._path: Path | None = None
        self._base_fingerprint: str | None = None
        self._revision = 0
        self._lock = threading.RLock()
        self._save_lock = threading.Lock()

    @classmethod
    def new(
        cls,
        target: PulseTarget,
        *,
        time_step_ns: int | float,
        name: str = "Untitled pulse",
    ) -> "PulseEditorSession":
        return cls(
            new_pulse_document(
                target,
                time_step_ns=time_step_ns,
                name=name,
            ),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "PulseEditorSession":
        resolved = pulse_document_path(path)
        document = load_pulse_document(resolved)
        session = cls(document)
        session._path = resolved
        session._base_fingerprint = document.fingerprint
        return session

    @property
    def document(self) -> PulseDocument:
        with self._lock:
            return self._document

    @property
    def path(self) -> Path | None:
        with self._lock:
            return self._path

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def dirty(self) -> bool:
        with self._lock:
            return (
                self._base_fingerprint is None
                or self._document.fingerprint != self._base_fingerprint
            )

    def snapshot(self) -> tuple[int, PulseDocument]:
        with self._lock:
            return self._revision, self._document

    def replace_document(self, document: PulseDocument) -> int:
        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        with self._lock:
            if document == self._document:
                return self._revision
            self._document = document
            self._revision += 1
            return self._revision

    def bind_target(self, target: PulseTarget) -> int:
        """Explicitly rebind to one online target; a changed document stays dirty."""

        if not isinstance(target, PulseTarget):
            raise TypeError("target must be PulseTarget")
        with self._lock:
            document = self._document
        return self.replace_document(bind_pulse_document_target(document, target))

    def nominal_reference_document(self) -> PulseDocument:
        """Explicitly remove scan intent while retaining authored nominal values."""

        with self._lock:
            document = self._document
        if not document.scan_parameters:
            return document
        return replace(
            document,
            scan_parameters=(),
            scan_table=None,
            scan_recipe=None,
        )

    def preview(
        self,
        *,
        max_timeline_items: int = 50_000,
    ) -> tuple[int, PulseTimelineDocument]:
        revision, document = self.snapshot()
        timeline = project_pulse_preview(
            document,
            max_timeline_items=max_timeline_items,
        )
        return revision, timeline

    def save(
        self,
        path: str | Path | None = None,
        *,
        overwrite: bool = False,
    ) -> Path:
        with self._save_lock:
            with self._lock:
                document = self._document
                current_path = self._path
                base_fingerprint = self._base_fingerprint
            if path is None:
                if current_path is None:
                    raise ValueError("an unsaved pulse requires an explicit path")
                target = current_path
            else:
                target = pulse_document_path(path)
            same_file = current_path is not None and target == current_path
            if same_file and base_fingerprint is not None and not target.exists():
                raise RuntimeError("pulse document changed on disk since it was loaded")
            if target.exists():
                existing = load_pulse_document(target).fingerprint
                if same_file:
                    if base_fingerprint is None or existing != base_fingerprint:
                        raise RuntimeError("pulse document changed on disk since it was loaded")
                elif not overwrite:
                    raise FileExistsError(f"pulse document already exists: {target}")
            saved = save_pulse_document(document, target)
            with self._lock:
                self._path = saved
                self._base_fingerprint = document.fingerprint
            return saved


def project_pulse_preview(
    document: PulseDocument,
    *,
    max_timeline_items: int = 50_000,
) -> PulseTimelineDocument:
    """Compile the current static or nominal-reference view, never a scan row."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    reference = bool(document.scan_parameters)
    execution_form = (
        PulseExecutionForm.STATIC_REFERENCE_POINT
        if reference
        else PulseExecutionForm.STATIC_ONCE
    )
    artifact = compile_pulse_artifact(
        document,
        clock_hz=1e9 / document.time_step_ns,
        execution_form=execution_form,
        live_target=document.target,
    )
    label = (
        "nominal scan/API reference"
        if reference or document.api_parameters
        else "compiled static pulse"
    )
    return build_pulse_timeline(
        document,
        artifact,
        reference_label=label,
        max_timeline_items=max_timeline_items,
    )


__all__ = ["PulseEditorSession", "project_pulse_preview"]
