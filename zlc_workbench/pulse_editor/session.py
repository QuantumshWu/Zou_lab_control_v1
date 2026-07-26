"""PulseEditor-owned state and exact preview orchestration."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
import threading

from zlc_pulse import (
    PORT_DAC,
    PORT_DIGITAL,
    PulseDocument,
    PulseExecutionForm,
    PulseTarget,
    PulseTimelineDocument,
    bind_pulse_document_target,
    build_authored_pulse_timeline,
    compile_pulse_document,
    insert_period,
    load_pulse_document,
    new_period,
    new_pulse_document,
    nominal_scan_reference,
    pulse_document_path,
    save_pulse_document,
    set_digital_output,
)


class PulseEditorSession:
    """The only mutable owner of one editor's current immutable document."""

    __slots__ = (
        "_base_fingerprint",
        "_dirty",
        "_document",
        "_file_state",
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
        self._dirty = True
        self._file_state = "new"
        self._revision = 0
        self._lock = threading.RLock()
        self._save_lock = threading.Lock()

    @classmethod
    def new(
        cls,
        target: PulseTarget,
        *,
        time_step_ns: int | float,
        name: str | None = None,
    ) -> "PulseEditorSession":
        # Preserve the established PulseGUI's familiar initial schedule while
        # keeping ``zlc_pulse.new_pulse_document`` as the neutral one-safe-period
        # domain primitive.  A new editor opens with two 1 us period cards: the
        # first programmable digital output high in P1, then an all-safe P2.
        # Clear All remains the distinct one-blank-period operation.
        document = new_pulse_document(
            target,
            time_step_ns=time_step_ns,
            name=(
                str(name)
                if name is not None
                else "pulse_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            ),
        )
        familiar_visible = tuple(
            port.key
            for port in target.ports
            if port.kind in (PORT_DIGITAL, PORT_DAC)
        )[:4]
        document = replace(document, visible_ports=familiar_visible)
        first_digital = next(
            (port.key for port in target.ports if port.kind == PORT_DIGITAL),
            None,
        )
        if first_digital is not None:
            document = set_digital_output(
                document,
                document.periods[0].period_id,
                first_digital,
                True,
            )
        document = insert_period(
            document,
            period=new_period(document),
        ).document
        return cls(document)

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
        session._dirty = False
        session._file_state = "loaded"
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
    def file_state(self) -> str:
        """Return the formal title state: new, loaded, or saved."""

        with self._lock:
            return self._file_state

    @property
    def dirty(self) -> bool:
        with self._lock:
            return self._dirty

    def capture_document(self) -> tuple[int, PulseDocument]:
        """Freeze one immutable document reference for worker submission."""

        with self._lock:
            return self._revision, self._document

    def replace_document(self, document: PulseDocument) -> int:
        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        with self._lock:
            if document == self._document:
                return self._revision
            self._document = document
            self._dirty = True
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
        return nominal_scan_reference(document)

    def preview(self) -> tuple[int, PulseTimelineDocument]:
        revision, document = self.capture_document()
        timeline = project_pulse_preview(document)
        return revision, timeline

    def save(
        self,
        path: str | Path | None = None,
        *,
        overwrite: bool = False,
    ) -> Path:
        with self._save_lock:
            with self._lock:
                revision = self._revision
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
                self._file_state = "saved"
                self._dirty = not (
                    self._revision == revision and self._document is document
                )
            return saved


def project_pulse_preview(
    document: PulseDocument,
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
    target_ir = compile_pulse_document(
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
    return build_authored_pulse_timeline(
        document,
        target_ir,
        execution_form=execution_form,
        reference_label=label,
    )


__all__ = ["PulseEditorSession", "project_pulse_preview"]
