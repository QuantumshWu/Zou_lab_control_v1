"""No-fail compaction of exception frame graphs at ownership boundaries."""

from __future__ import annotations

from collections.abc import Callable, Iterable


_MAX_EXCEPTION_NODES = 32
_MAX_LOCATION_FRAMES = 8
_ARGS_DESCRIPTOR = BaseException.__dict__["args"]
_TRACEBACK_DESCRIPTOR = BaseException.__dict__["__traceback__"]
_CAUSE_DESCRIPTOR = BaseException.__dict__["__cause__"]
_CONTEXT_DESCRIPTOR = BaseException.__dict__["__context__"]
_EXCEPTION_DICT_DESCRIPTOR = BaseException.__dict__["__dict__"]
try:
    _BASE_EXCEPTION_GROUP = BaseExceptionGroup
except NameError:  # pragma: no cover - Python 3.9/3.10 compatibility
    _BASE_EXCEPTION_GROUP = None
_GROUP_EXCEPTIONS_DESCRIPTOR = (
    None
    if _BASE_EXCEPTION_GROUP is None
    else _BASE_EXCEPTION_GROUP.__dict__["exceptions"]
)


class DetachedExceptionGraph(RuntimeError):
    """String-only failure evidence that cannot retain live runtime authority."""

    def __init__(
        self,
        original_type: str,
        detail: str,
        *,
        truncated: bool,
        related_summaries: tuple[str, ...] = (),
    ) -> None:
        self.original_type = original_type
        self.detail = detail
        self.truncated = bool(truncated)
        self.related_summaries = tuple(
            summary for summary in related_summaries if type(summary) is str
        )[:_MAX_EXCEPTION_NODES]
        super().__init__(self.summary)

    @property
    def summary(self) -> str:
        suffix = (
            f" [exception graph exceeded {_MAX_EXCEPTION_NODES} nodes]"
            if self.truncated
            else ""
        )
        return f"{self.original_type}: {self.detail}{suffix}"


def _safe_type_name(value: object) -> str:
    try:
        return type.__getattribute__(type(value), "__name__")
    except BaseException:
        return "object"


def _safe_argument_text(value: object, *, max_text: int = 256) -> str:
    value_type = type(value)
    if value_type is str:
        return value if len(value) <= max_text else value[: max_text - 3] + "..."
    if value is None:
        return "None"
    if value_type is bool:
        return "True" if value else "False"
    if value_type is int:
        if value.bit_length() <= 128:
            return int.__str__(value)
        return f"<int:{value.bit_length()} bits>"
    if value_type is float:
        return float.__repr__(value)
    if value_type is bytes:
        return f"<bytes:{len(value)}>"
    return f"<{_safe_type_name(value)}>"


def _safe_error_identity(error: BaseException) -> tuple[str, str]:
    error_type = _safe_type_name(error)
    try:
        args = _ARGS_DESCRIPTOR.__get__(error, BaseException)
    except BaseException:
        return error_type, "<unprintable>"
    if type(args) is not tuple:
        return error_type, "<unprintable>"
    if not args:
        return error_type, ""
    rendered = tuple(_safe_argument_text(value) for value in args[:4])
    if len(args) == 1:
        detail = rendered[0]
    else:
        suffix = ", ..." if len(args) > 4 else ""
        detail = "(" + ", ".join(rendered) + suffix + ")"
    return error_type, detail


def clear_exception_traceback(error: BaseException) -> None:
    """Clear the base traceback slot without invoking subclass descriptors."""

    _TRACEBACK_DESCRIPTOR.__set__(error, None)


def _safe_exception_notes(error: BaseException) -> tuple[str, ...]:
    try:
        attributes = _EXCEPTION_DICT_DESCRIPTOR.__get__(error, BaseException)
    except BaseException:
        return ()
    if type(attributes) is not dict:
        return ()
    notes = dict.get(attributes, "__notes__", ())
    if type(notes) is not list:
        return ()
    return tuple(
        note if len(note) <= 512 else note[:509] + "..."
        for note in notes[:16]
        if type(note) is str
    )


def _append_safe_note(error: BaseException, note: str) -> None:
    """Write the base exception dict directly; never invoke subclass hooks."""

    if type(note) is not str:
        return
    try:
        attributes = _EXCEPTION_DICT_DESCRIPTOR.__get__(error, BaseException)
    except BaseException:
        return
    if type(attributes) is not dict:
        return
    notes = dict.get(attributes, "__notes__")
    if notes is None:
        dict.__setitem__(attributes, "__notes__", [note])
        return
    if type(notes) is list and len(notes) < 16:
        duplicate = any(
            type(existing) is str and str.__eq__(existing, note)
            for existing in notes
        )
        if not duplicate:
            list.append(notes, note)


def safe_error_summary(error: BaseException, *, max_detail: int = 512) -> str:
    """Format an exception without trusting subclass ``str``/``repr`` hooks."""

    if type(error) is DetachedExceptionGraph:
        original_type = object.__getattribute__(error, "original_type")
        detail = object.__getattribute__(error, "detail")
        truncated = object.__getattribute__(error, "truncated")
        if type(original_type) is str and type(detail) is str:
            suffix = (
                f" [exception graph exceeded {_MAX_EXCEPTION_NODES} nodes]"
                if truncated is True
                else ""
            )
            return f"{original_type}: {detail}{suffix}"
    error_type, detail = _safe_error_identity(error)
    if len(detail) > max_detail:
        detail = detail[: max(0, max_detail - 3)] + "..."
    return f"{error_type}: {detail}"


def record_secondary_failure(
    primary: BaseException | None,
    operation: str,
    secondary: BaseException,
) -> None:
    """Attach diagnostics without allowing them to interrupt safety cleanup."""

    if primary is None:
        return
    _append_safe_note(
        primary,
        f"{operation}: {safe_error_summary(secondary)}",
    )


def _traceback_locations(saved_traceback: object | None) -> tuple[str, ...]:
    locations: list[str] = []
    cursor = saved_traceback
    try:
        while cursor is not None:
            frame = cursor.tb_frame
            locations.append(
                f"{frame.f_code.co_filename}:{cursor.tb_lineno}:"
                f"{frame.f_code.co_name}"
            )
            if len(locations) > _MAX_LOCATION_FRAMES:
                locations.pop(0)
            cursor = cursor.tb_next
    except BaseException:
        return ()
    finally:
        cursor = None
    return tuple(locations)


def detach_exception_graph(
    error: BaseException | None,
    *,
    note_prefix: str,
    sever_chaining: bool,
    replace_with_evidence: bool = False,
    nested_errors: Callable[[BaseException], Iterable[BaseException]] | None = None,
) -> BaseException | None:
    """Drop traceback frames while retaining bounded immutable location evidence.

    ``sever_chaining`` is true at a terminal Run boundary, where the exception
    object is durable diagnostic evidence rather than a live control-flow
    object.  Processor workers keep ``__cause__``/``__context__`` so their
    immediate ``raise_if_failed`` API can still expose the original cause.

    Root traceback ownership is cleared before allocating the traversal work
    list.  BaseException descriptors are called directly so a hostile subclass
    cannot override ``add_note`` or attribute access and prevent cleanup.
    """

    if error is None:
        return None
    prior_truncated = False
    if type(error) is DetachedExceptionGraph:
        candidate_type = object.__getattribute__(error, "original_type")
        candidate_detail = object.__getattribute__(error, "detail")
        candidate_truncated = object.__getattribute__(error, "truncated")
        if type(candidate_type) is str and type(candidate_detail) is str:
            root_type, root_detail = candidate_type, candidate_detail
            prior_truncated = candidate_truncated is True
        else:
            root_type, root_detail = _safe_error_identity(error)
    else:
        root_type, root_detail = _safe_error_identity(error)
    root_notes = _safe_exception_notes(error)
    related_summaries: list[str] = []
    if type(error) is DetachedExceptionGraph:
        prior_related = object.__getattribute__(error, "related_summaries")
        if type(prior_related) is tuple:
            related_summaries.extend(
                summary
                for summary in prior_related[:_MAX_EXCEPTION_NODES]
                if type(summary) is str
            )
    try:
        root_traceback = _TRACEBACK_DESCRIPTOR.__get__(error, BaseException)
        _TRACEBACK_DESCRIPTOR.__set__(error, None)
    except BaseException:
        # This should be unreachable for a real BaseException, but compaction
        # must never block terminal publication.
        return error

    pending: list[tuple[BaseException, object | None]] = [(error, root_traceback)]
    visited: set[int] = set()
    scheduled: set[int] = {id(error)}
    overflow = False
    root_locations: tuple[str, ...] = ()
    while pending and len(visited) < _MAX_EXCEPTION_NODES:
        current, saved_traceback = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        current_summary = safe_error_summary(current)
        if current_summary not in related_summaries:
            related_summaries.append(current_summary)

        try:
            cause = _CAUSE_DESCRIPTOR.__get__(current, BaseException)
        except BaseException:
            cause = None
        try:
            context = _CONTEXT_DESCRIPTOR.__get__(current, BaseException)
        except BaseException:
            context = None

        if sever_chaining:
            try:
                _CAUSE_DESCRIPTOR.__set__(current, None)
            except BaseException:
                pass
            try:
                _CONTEXT_DESCRIPTOR.__set__(current, None)
            except BaseException:
                pass

        linked: list[Iterable[BaseException]] = []
        direct = tuple(
            candidate
            for candidate in (cause, context if context is not cause else None)
            if isinstance(candidate, BaseException)
        )
        linked.append(direct)
        if _BASE_EXCEPTION_GROUP is not None and isinstance(current, _BASE_EXCEPTION_GROUP):
            try:
                group_errors = _GROUP_EXCEPTIONS_DESCRIPTOR.__get__(
                    current, _BASE_EXCEPTION_GROUP
                )
            except BaseException:
                group_errors = ()
            linked.append(group_errors)
        if nested_errors is not None:
            try:
                linked.append(nested_errors(current))
            except BaseException:
                pass

        for candidates in linked:
            try:
                iterator = iter(candidates)
            except BaseException:
                continue
            while True:
                try:
                    candidate = next(iterator)
                except StopIteration:
                    break
                except BaseException:
                    break
                if not isinstance(candidate, BaseException):
                    continue
                candidate_identity = id(candidate)
                if candidate_identity in scheduled:
                    continue
                try:
                    candidate_traceback = _TRACEBACK_DESCRIPTOR.__get__(
                        candidate, BaseException
                    )
                    _TRACEBACK_DESCRIPTOR.__set__(candidate, None)
                except BaseException:
                    candidate_traceback = None
                if len(scheduled) >= _MAX_EXCEPTION_NODES:
                    overflow = True
                    break
                scheduled.add(candidate_identity)
                pending.append((candidate, candidate_traceback))
            if overflow:
                break

        locations = _traceback_locations(saved_traceback)
        saved_traceback = None
        if current is error:
            root_locations = locations
        if not locations:
            continue
        note = f"{note_prefix}: " + " <- ".join(locations)
        _append_safe_note(current, note)
    if pending:
        overflow = True
    if not overflow and not replace_with_evidence:
        return error

    detached = DetachedExceptionGraph(
        root_type,
        root_detail,
        truncated=prior_truncated or overflow,
        related_summaries=tuple(related_summaries),
    )
    if root_locations:
        _append_safe_note(
            detached,
            f"{note_prefix}: " + " <- ".join(root_locations),
        )
    for note in root_notes:
        _append_safe_note(detached, note)
    return detached
