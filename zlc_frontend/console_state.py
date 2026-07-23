"""The saved console layout: the record, its JSON file, and how a name resolves to one.

Its FORMAT declaration -- the field spec and the persisted ``schema`` discriminator -- sank
to :mod:`zlc_data.console_records` beside the panel and logic-node records.  The class
itself could not follow, and the design document says why rather than taste:

  L303  ``zlc_data`` 只容纳领域中立、headless、可序列化的数据语义和值上的纯算法
  L320  zlc_data -> zlc_storage.canonical(只允许纯 canonical 模块,不允许 repository/I/O)

``save``/``load`` read and write a file, which is not a pure algorithm on a value.  Nor
does this belong in ``zlc_storage``: L315 gives that package canonical primitives and
bytes/blob/manifest publication and says it "不定义 universal ArtifactRef、领域 schema 或
artifact kind" -- while a console-layout reader constructs a domain record.  What is left
is ``zlc_frontend``, which L327 explicitly allows to depend on storage.  So the record
lives here, its vocabulary in ``zlc_data``, and the project root it anchors to comes from
``zlc_storage.paths``.

``save`` writes in place rather than staging and renaming.  That is the behaviour this
module INHERITED and it is preserved deliberately: the only atomic publication in the
repository lives inside ``ContentAddressedStore`` for content-addressed blobs, so making
layout writes crash-safe means designing a new storage primitive, not moving code.  The
gap is recorded rather than quietly widened.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from zlc_data.console_records import (
    CONSOLE_STATE_SCHEMA,
    LogicNodeConfig,
    PanelConfig,
    TASK_CONSOLE_STATE_FIELDS,
    layout_record,
)
from zlc_storage.paths import project_path

__all__ = ["TASK_FILES_ENV", "TaskConsoleState", "default_console_state",
           "resolve_task_state", "task_files_dir"]

#: Environment override for where saved layouts live.
TASK_FILES_ENV = "ZLC_TASK_DIR"


def task_files_dir() -> Path:
    """The folder saved console layouts live in: ``$ZLC_TASK_DIR`` if set, else the project's
    ``tasks/``.  The default comes from :func:`zlc_storage.paths.project_path`, the one owner of
    "where the project root is".

    A whitespace-only override falls back rather than crashing: it is a human typo in an
    environment variable, and ``Path("   ").mkdir()`` raised FileNotFoundError."""
    root = os.environ.get(TASK_FILES_ENV, "").strip()
    path = Path(root).expanduser() if root else project_path("tasks")
    path.mkdir(parents=True, exist_ok=True)
    return path


class TaskConsoleState:
    """The whole console layout: serialised as ONE machine-portable JSON file."""

    #: The persisted semantic discriminator, defined with the record shape in
    #: :mod:`zlc_data.console_records`; it is intentionally independent of module paths.
    schema = CONSOLE_STATE_SCHEMA

    def __init__(
        self,
        *,
        name: str = "task",
        interval_ms: int = 400,
        panels: Sequence[PanelConfig | Mapping[str, object]] | None = None,
        logic: Sequence[LogicNodeConfig | Mapping[str, object]] | None = None,
    ):
        self.name = str(name)
        self.interval_ms = max(50, int(interval_ms))
        self.panels = [
            PanelConfig.from_dict(
                copy.deepcopy(
                    panel.to_dict()
                    if isinstance(panel, PanelConfig)
                    else panel
                )
            )
            for panel in (panels or [])
        ]
        panel_ids = tuple(panel.panel_id for panel in self.panels)
        if len(panel_ids) != len(set(panel_ids)):
            raise ValueError("TaskConsoleState contains duplicate panel_id values")
        # The Logic-tab nodes (measurement / processor / task), saved alongside the
        # plot panels so a layout restores the whole dashboard -- nodes always come
        # back STOPPED (the layout records what to build, not a running thread).
        self.logic = [
            LogicNodeConfig.from_dict(
                copy.deepcopy(
                    node.to_dict()
                    if isinstance(node, LogicNodeConfig)
                    else node
                )
            )
            for node in (logic or [])
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "name": self.name,
            "interval_ms": self.interval_ms,
            "panels": [panel.to_dict() for panel in self.panels],
            "logic": [node.to_dict() for node in self.logic],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TaskConsoleState":
        data = layout_record(payload, TASK_CONSOLE_STATE_FIELDS, cls.schema)
        if data["interval_ms"] < 50:
            raise ValueError("interval_ms must be at least 50")
        return cls(
            name=data["name"],
            interval_ms=data["interval_ms"],
            panels=data["panels"],
            logic=data["logic"],
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TaskConsoleState":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def default_console_state() -> TaskConsoleState:
    """The console opens EMPTY.  You build the dashboard yourself (Add Panel) and
    Save it to a file; you reuse it by Loading that file back (the Load button /
    ``--task`` / ``show_task_console(task=)``)."""

    return TaskConsoleState(name="task", panels=[])


def resolve_task_state(task: str) -> TaskConsoleState:
    """Resolve a task layout by FILE PATH or a saved tasks/<name>.json."""

    text = str(task).strip()
    path = Path(text)
    if path.suffix.lower() == ".json" and path.exists():
        return TaskConsoleState.load(path)
    saved = task_files_dir() / f"{text}.json"
    if saved.exists():
        return TaskConsoleState.load(saved)
    raise ValueError(
        f"unknown task {task!r}: not a layout file and not a saved layout in {task_files_dir()}.")
