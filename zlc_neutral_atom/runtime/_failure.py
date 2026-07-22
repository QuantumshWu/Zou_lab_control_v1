"""String-only failure evidence for runtime ownership boundaries."""

from __future__ import annotations

import traceback


class DetachedFailure(RuntimeError):
    """Diagnostic evidence that cannot retain traceback frames or capabilities."""

    __slots__ = ("original_type", "detail", "locations", "notes")

    def __init__(
        self,
        original_type: str,
        detail: str,
        *,
        locations: tuple[str, ...] = (),
        notes: tuple[str, ...] = (),
    ) -> None:
        self.original_type = original_type
        self.detail = detail
        self.locations = tuple(locations)
        self.notes = tuple(notes)
        super().__init__(f"{original_type}: {detail}")

    @property
    def summary(self) -> str:
        return f"{self.original_type}: {self.detail}"


def _safe_text(value: object) -> str:
    try:
        return str(value)
    except BaseException:
        return "<unprintable>"


def safe_error_summary(error: BaseException) -> str:
    """Return complete string diagnostics without raising from formatting."""

    if isinstance(error, DetachedFailure):
        return error.summary
    try:
        error_type = type(error).__name__
    except BaseException:
        error_type = "BaseException"
    return f"{error_type}: {_safe_text(error)}"


def detach_failure(
    error: BaseException | None,
    *,
    note_prefix: str,
) -> DetachedFailure | None:
    """Replace an exception graph with a complete string-only value."""

    if error is None:
        return None
    if isinstance(error, DetachedFailure):
        return error
    summary = safe_error_summary(error)
    original_type, _, detail = summary.partition(": ")
    try:
        extracted = traceback.extract_tb(error.__traceback__)
        locations = tuple(
            f"{frame.filename}:{frame.lineno}:{frame.name}"
            for frame in extracted
        )
    except BaseException:
        locations = ()
    try:
        raw_notes = getattr(error, "__notes__", ())
        notes = tuple(
            _safe_text(note)
            for note in raw_notes
            if isinstance(note, str)
        )
    except BaseException:
        notes = ()
    try:
        cause = error.__cause__ or error.__context__
        if isinstance(cause, BaseException) and cause is not error:
            notes = (*notes, "caused by: " + safe_error_summary(cause))
    except BaseException:
        pass
    if locations:
        notes = (f"{note_prefix}: " + " <- ".join(locations), *notes)
    detached = DetachedFailure(
        original_type,
        detail,
        locations=locations,
        notes=notes,
    )
    try:
        error.__traceback__ = None
        error.__cause__ = None
        error.__context__ = None
    except BaseException:
        pass
    return detached


def record_secondary_failure(
    primary: BaseException | None,
    operation: str,
    secondary: BaseException,
) -> None:
    """Best-effort note attachment; safety cleanup must never be interrupted."""

    if primary is None:
        return
    note = f"{operation}: {safe_error_summary(secondary)}"
    try:
        primary.add_note(note)
    except BaseException:
        pass
