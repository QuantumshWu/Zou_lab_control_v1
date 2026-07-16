"""Headless PulseGUI editor state and exact preview orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading

from zlc_neutral_atom.pulse_application import PulseTargetDescriptor
from zlc_pulse import (
    PulseDocument,
    PulseExecutionForm,
    PulseTimelineDocument,
    bind_pulse_document_target,
    build_pulse_timeline,
    compile_pulse_artifact,
    load_pulse_document,
    new_pulse_document,
    save_pulse_document,
)


class PulseEditorSession:
    """The only mutable owner of one editor's current immutable document."""

    __slots__ = (
        "_base_fingerprint",
        "_descriptor",
        "_document",
        "_lock",
        "_path",
        "_revision",
    )

    def __init__(
        self,
        descriptor: PulseTargetDescriptor,
        document: PulseDocument,
        *,
        path: str | Path | None = None,
    ) -> None:
        if not isinstance(descriptor, PulseTargetDescriptor):
            raise TypeError("descriptor must be PulseTargetDescriptor")
        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        self._descriptor = descriptor
        self._document = bind_pulse_document_target(document, descriptor.target)
        self._path = None if path is None else Path(path).expanduser().resolve()
        self._base_fingerprint = (
            None if self._path is None else self._document.fingerprint
        )
        self._revision = 0
        self._lock = threading.RLock()

    @classmethod
    def new(
        cls,
        descriptor: PulseTargetDescriptor,
        *,
        name: str = "Untitled pulse",
    ) -> "PulseEditorSession":
        return cls(
            descriptor,
            new_pulse_document(
                descriptor.target,
                time_step_ns=descriptor.time_step_ns,
                name=name,
            ),
        )

    @classmethod
    def load(
        cls,
        descriptor: PulseTargetDescriptor,
        path: str | Path,
    ) -> "PulseEditorSession":
        resolved = Path(path).expanduser().resolve()
        return cls(descriptor, load_pulse_document(resolved), path=resolved)

    @property
    def descriptor(self) -> PulseTargetDescriptor:
        return self._descriptor

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
        bound = bind_pulse_document_target(document, self._descriptor.target)
        with self._lock:
            if bound == self._document:
                return self._revision
            self._document = bound
            self._revision += 1
            return self._revision

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
            self._descriptor,
            max_timeline_items=max_timeline_items,
        )
        return revision, timeline

    def save(
        self,
        path: str | Path | None = None,
        *,
        overwrite: bool = False,
    ) -> Path:
        with self._lock:
            document = self._document
            revision = self._revision
            current_path = self._path
            base_fingerprint = self._base_fingerprint
        if path is None:
            if current_path is None:
                raise ValueError("an unsaved pulse requires an explicit path")
            target = current_path
        else:
            target = Path(path).expanduser().resolve()
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
            if self._revision == revision and self._document is document:
                return saved
        return saved


def project_pulse_preview(
    document: PulseDocument,
    descriptor: PulseTargetDescriptor,
    *,
    max_timeline_items: int = 50_000,
) -> PulseTimelineDocument:
    """Compile the current static or nominal-reference view, never a scan row."""

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    if not isinstance(descriptor, PulseTargetDescriptor):
        raise TypeError("descriptor must be PulseTargetDescriptor")
    document = bind_pulse_document_target(document, descriptor.target)
    reference = bool(document.scan_parameters)
    execution_form = (
        PulseExecutionForm.STATIC_REFERENCE_POINT
        if reference
        else PulseExecutionForm.STATIC_ONCE
    )
    artifact = compile_pulse_artifact(
        document,
        clock_hz=descriptor.clock_hz,
        execution_form=execution_form,
        live_target=descriptor.target,
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
