"""The logic-node record: which node, and the values it was built with.

A logic node lives on the console's Logic tab and is the thing that PRODUCES data.
This module holds only the record and its codec - no Qt, no renderer - so the row
card that displays it, the workspace writer that persists it and any future reader
all share one definition instead of agreeing by convention.

``layout_record`` came along because it is the shared validator every console record
uses (panel, logic node, console state): exact key set, exact field types, and no
silent coercion.  Keeping it beside the first record that moved would have split it
from its other two callers, so it sinks here and the shell imports it back.
"""

from __future__ import annotations

from typing import Mapping

from zlc_storage.canonical import exact_mapping

__all__ = ["LOGIC_KINDS", "LOGIC_NODE_CONFIG_FIELDS", "LogicNodeConfig", "layout_record"]


#: The four node families the Logic tab can add.
LOGIC_KINDS = ("camera", "measurement", "processor", "task")

LOGIC_NODE_CONFIG_FIELDS = {"kind": str, "name": str, "title": str, "values": dict}


def layout_record(
    payload: Mapping[str, object],
    fields: Mapping[str, type],
    name: str,
    *,
    discriminator: str | None = "schema",
) -> dict[str, object]:
    data = exact_mapping(payload, set(fields), name, discriminator=discriminator)
    for field, expected_type in fields.items():
        value = data[field]
        if type(value) is not expected_type:
            raise TypeError(
                f"{field} must be {expected_type.__name__}, got {type(value).__name__}"
            )
    return data


class LogicNodeConfig:
    """One LOGIC NODE: which node it is + the param values to build it with.

    A logic node lives on the Logic tab, NOT the Monitor board, and is the thing
    that PRODUCES data.  ``kind`` is one of :data:`LOGIC_KINDS` (camera /
    measurement / processor / task); ``name`` is the catalog spec's name (the
    camera's is ``"live"``; its display TITLE comes from ``readout.camera_spec().name``).
    ``values`` is the last param-form ``{key: value}`` it was built / run with, so
    reopening its Edit restores them.  A node is always added STOPPED -- nothing
    runs until Start in its Edit."""

    def __init__(self, *, kind: str, name: str, title: str = "",
                 values: Mapping[str, object] | None = None):
        if kind not in LOGIC_KINDS:
            raise ValueError(f"unknown logic kind {kind!r}; choose from {list(LOGIC_KINDS)}.")
        self.kind = str(kind)
        self.name = str(name)
        self.title = str(title) or str(name)
        self.values = dict(values or {})

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "name": self.name, "title": self.title,
                "values": dict(self.values)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LogicNodeConfig":
        data = layout_record(
            payload,
            LOGIC_NODE_CONFIG_FIELDS,
            "LogicNodeConfig",
            discriminator=None,
        )
        result = cls(**data)
        if result.title != data["title"]:
            raise ValueError("LogicNodeConfig is not in current canonical form")
        return result
