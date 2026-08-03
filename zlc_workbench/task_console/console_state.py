"""Renderer-free product value and codec for one TaskConsole layout."""

from __future__ import annotations

import copy
from typing import Mapping, Sequence

from .console_records import (
    CONSOLE_STATE_SCHEMA,
    LogicNodeConfig,
    PanelConfig,
    TASK_CONSOLE_STATE_FIELDS,
    layout_record,
)
__all__ = ["TaskConsoleState", "default_console_state"]


class TaskConsoleState:
    """The whole console layout: serialised as ONE machine-portable JSON file."""

    #: The persisted semantic discriminator, defined with the record shape in
    #: :mod:`zlc_workbench.task_console.console_records`; it is intentionally
    #: independent of module paths.
    schema = CONSOLE_STATE_SCHEMA

    def __init__(
        self,
        *,
        name: str = "task",
        interval_ms: int = 400,
        panels: Sequence[PanelConfig | Mapping[str, object]] | None = None,
        logic: Sequence[LogicNodeConfig | Mapping[str, object]] | None = None,
    ):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("TaskConsole name must be non-empty str")
        if (
            isinstance(interval_ms, bool)
            or not isinstance(interval_ms, int)
            or interval_ms < 50
        ):
            raise ValueError("interval_ms must be an int of at least 50")
        if panels is not None and not isinstance(panels, Sequence):
            raise TypeError("panels must be a sequence or None")
        if logic is not None and not isinstance(logic, Sequence):
            raise TypeError("logic must be a sequence or None")
        self.name = name
        self.interval_ms = interval_ms
        self.panels = [
            copy.deepcopy(panel)
            if isinstance(panel, PanelConfig)
            else PanelConfig.from_dict(copy.deepcopy(panel))
            for panel in (panels or [])
        ]
        panel_ids = tuple(panel.panel_id for panel in self.panels)
        if len(panel_ids) != len(set(panel_ids)):
            raise ValueError("TaskConsoleState contains duplicate panel_id values")
        # The Logic-tab nodes (measurement / processor / task), saved alongside the
        # plot panels so a layout restores the whole dashboard -- nodes always come
        # back STOPPED (the layout records what to build, not a running thread).
        self.logic = [
            copy.deepcopy(node)
            if isinstance(node, LogicNodeConfig)
            else LogicNodeConfig.from_dict(copy.deepcopy(node))
            for node in (logic or [])
        ]
        node_ids = tuple(node.node_id for node in self.logic)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("TaskConsoleState contains duplicate logic node_id values")

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
        result = cls(
            name=data["name"],
            interval_ms=data["interval_ms"],
            panels=data["panels"],
            logic=data["logic"],
        )
        if result.to_dict() != data:
            raise ValueError("TaskConsoleState is not in current canonical form")
        return result

def default_console_state() -> TaskConsoleState:
    """The console opens EMPTY.  You build the dashboard yourself (Add Panel) and
    Save it to a file; you reuse it by Loading that file back (the Load button /
    ``--task`` / ``show_task_console(task=)``)."""

    return TaskConsoleState(name="task", panels=[])
