"""The console's saved records: what a panel is, what a logic node is, and their codecs.

These are the two things a console layout is made of.  A PANEL is a view -- a plot kind,
a size preset, a pixel position, and the one typed dataset it reads.  A LOGIC NODE is what
PRODUCES data on the Logic tab.  Both are plain values with a JSON codec, and neither
needs Qt or a renderer to say what it is, so both live here and the windows that display
them, the workspace writer that persists them and any future reader share one definition
instead of agreeing by convention.

``layout_record`` is the validator all three console records use (panel, logic node,
console state): exact key set, exact field types, no silent coercion.

The panel VOCABULARY below -- which kinds exist and what shape each accepts -- is derived
from :mod:`zlc_data.plot_kind` rather than restated.
That is what lets ``PanelConfig`` validate its own ``kind`` here: refusing an unknown kind
is a question about the vocabulary, not about drawing.  Until the plot-kind table split,
that single ``if kind not in PANEL_KINDS`` line was enough to tie this record to the whole
Matplotlib stack.
"""

from __future__ import annotations

from typing import Mapping

from zlc_data.panel_size import panel_size_cells
from zlc_data.plot_kind import PLOT_KIND_SPECS
from zlc_storage.canonical import exact_mapping

__all__ = ["DEFAULT_UPDATE_MS", "LOGIC_KINDS", "LOGIC_NODE_CONFIG_FIELDS",
           "LogicNodeConfig", "PANEL_CONFIG_FIELDS", "PANEL_KINDS",
           "CONSOLE_STATE_SCHEMA", "PanelConfig", "TASK_CONSOLE_STATE_FIELDS",
           "UPDATE_INTERVALS", "layout_record"]


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


# ------------------------------------------------------------------ panel vocabulary
#: ``key -> label`` for current TaskConsole panels.  The complete figure
#: vocabulary remains in :mod:`zlc_data.plot_kind`; this projection contains
#: only kinds with an end-to-end typed live payload and renderer.  A
#: ``PanelConfig`` therefore cannot persist an advertised-but-unimplemented
#: pulse/site/grid bridge.
PANEL_KINDS: dict[str, str] = {
    spec.key: spec.label for spec in PLOT_KIND_SPECS if spec.panel
}


#: Per-panel display refresh intervals (ms) the operator can pick from.  The fixed harmonic set
#: lets one lightweight base timer schedule every card without one timer per widget.  It controls
#: presentation cadence only: each card reads its own producer revision, and cards sharing a timer
#: beat do not thereby claim a common shot or coherence identity.
UPDATE_INTERVALS = (100, 200, 400, 800)
DEFAULT_UPDATE_MS = 400

PANEL_CONFIG_FIELDS = {
    "kind": str,
    "title": str,
    "row": int,
    "col": int,
    "size": str,
    "signal": str,
    "params": dict,
}


class PanelConfig:
    """One panel: kind + a size PRESET + its pixel top-left on the board (``col`` = pixel x,
    ``row`` = pixel y).  The board packer re-packs these top-left under gravity (no column grid)."""

    def __init__(
        self,
        *,
        kind: str,
        title: str = "",
        row: int = 0,
        col: int = 0,
        size: str = "2x2",
        signal: str = "",
        params: Mapping[str, object] | None = None,
    ):
        if kind not in PANEL_KINDS:
            raise ValueError(f"unknown panel kind {kind!r}; choose from {sorted(PANEL_KINDS)}.")
        panel_size_cells(size)              # validate against the limited preset list
        self.kind = str(kind)
        self.title = str(title)
        self.row = max(0, int(row))    # pixel y of the card top-left (no column grid)
        self.col = max(0, int(col))    # pixel x of the card top-left (no column grid)
        self.size = str(size)
        # A panel is a view of exactly one typed dataset.  Combining producers is a
        # Processor/join concern; the GUI must not invent an independent-latest
        # expression and then present it as one coherent source.
        self.signal = str(signal)
        self.params = dict(params or {})

    @property
    def update_ms(self) -> int:
        """This panel's display refresh interval (ms), one of :data:`UPDATE_INTERVALS`.
        Stored in ``params`` (so it round-trips with the saved layout); an out-of-set value
        falls back to :data:`DEFAULT_UPDATE_MS` so the timer base stays harmonic."""
        ms = int(self.params.get("update_ms", DEFAULT_UPDATE_MS) or DEFAULT_UPDATE_MS)
        return ms if ms in UPDATE_INTERVALS else DEFAULT_UPDATE_MS

    def to_dict(self) -> dict[str, object]:
        # ``row``/``col`` are the card's pixel top-left (no column grid).  They are only a SEED for
        # the gravity packer, which re-packs the whole board on load, so a layout's reading order
        # (top-to-bottom, left-to-right) is what round-trips -- exact pixels are recomputed.
        return {
            "kind": self.kind,
            "title": self.title,
            "row": self.row,
            "col": self.col,
            "size": self.size,
            "signal": self.signal,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PanelConfig":
        data = layout_record(
            payload,
            PANEL_CONFIG_FIELDS,
            "PanelConfig",
            discriminator=None,
        )
        if data["row"] < 0 or data["col"] < 0:
            raise ValueError("panel row and col must be non-negative")
        result = cls(**data)
        return result


# ---------------------------------------------------------------- console-state format
#: The persisted DISCRIMINATOR of a saved console layout, written into every layout file as
#: its ``schema`` key and required to match exactly on load.
#:
#: It is a semantic format name, not a Python module path.  The current reader accepts exactly
#: this current shape; incompatible predecessor records are rejected rather than converted.
CONSOLE_STATE_SCHEMA = "zlc.task_console.layout"

#: The third console record's field spec, beside the other two.  ``schema`` is a field here
#: (not just a check) because it round-trips: a layout file carries it, and the exact-key rule
#: means a payload without it is refused rather than defaulted.
TASK_CONSOLE_STATE_FIELDS = {
    "schema": str,
    "name": str,
    "interval_ms": int,
    "panels": list,
    "logic": list,
}
