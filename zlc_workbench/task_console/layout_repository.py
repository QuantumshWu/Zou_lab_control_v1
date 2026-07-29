"""TaskConsole layout paths and atomic JSON publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

from .console_state import TaskConsoleState
from zlc_storage import durable_makedirs, flush_directory


def _require_current_state(state: TaskConsoleState) -> TaskConsoleState:
    if not isinstance(state, TaskConsoleState):
        raise TypeError("state must be TaskConsoleState")
    return state


def task_files_dir(tasks_root: Path) -> Path:
    """Return the explicit TaskConsole layout/export directory."""

    root = Path(tasks_root).expanduser()
    if not root.is_absolute():
        raise ValueError("TaskConsole tasks_root must be absolute")
    return durable_makedirs(root.resolve())


def save_task_console_state(
    state: TaskConsoleState,
    path: str | Path,
) -> Path:
    """Atomically publish one current TaskConsole JSON layout."""

    _require_current_state(state)
    target = Path(path).expanduser().resolve()
    durable_makedirs(target.parent)
    payload = json.dumps(
        state.to_dict(),
        indent=2,
        ensure_ascii=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        flush_directory(target.parent)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def load_task_console_state(path: str | Path) -> TaskConsoleState:
    """Decode one current TaskConsole JSON layout."""

    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    return _require_current_state(TaskConsoleState.from_dict(payload))


def resolve_task_state(task: str, *, tasks_root: Path) -> TaskConsoleState:
    """Resolve an existing JSON path or one simple name under the task directory."""

    text = str(task).strip()
    if not text:
        raise ValueError("task name or path must be non-empty")
    supplied = Path(text).expanduser()
    if supplied.suffix.lower() == ".json" and supplied.is_file():
        return load_task_console_state(supplied)
    if supplied.parent != Path("."):
        raise ValueError(f"task layout path does not exist: {supplied}")
    filename = supplied.name if supplied.suffix.lower() == ".json" else f"{text}.json"
    saved = task_files_dir(tasks_root) / filename
    if saved.is_file():
        return load_task_console_state(saved)
    raise ValueError(
        f"unknown task {task!r}: not an existing JSON layout or a saved task name"
    )


__all__ = [
    "load_task_console_state",
    "resolve_task_state",
    "save_task_console_state",
    "task_files_dir",
]
