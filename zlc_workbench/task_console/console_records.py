"""TaskConsole saved records: panels, logic rows, and their current codecs.

These are the two things a console layout is made of.  A PANEL is a view -- a plot kind,
a size preset, a pixel position, and the one typed dataset it reads.  A LOGIC NODE is what
PRODUCES data on the Logic tab.  Both are plain values with a JSON codec, and neither
needs Qt or a renderer to say what it is, so both live here and the windows that display
them, the workspace writer that persists them and any future reader share one definition
instead of agreeing by convention.

``layout_record`` is the validator all three console records use (panel, logic node,
console state): exact key set, exact field types, no silent coercion.

The panel VOCABULARY below -- which kinds exist and what shape each accepts -- is derived
from :mod:`zlc_frontend.plot_kind` rather than restated.
That is what lets ``PanelConfig`` validate its own ``kind`` here: refusing an unknown kind
is a question about the vocabulary, not about drawing, so this record does not
import the Matplotlib renderer.
"""

from __future__ import annotations

import copy
from typing import Mapping
from uuid import uuid4

from zlc_frontend.panel_size import panel_size_cells
from zlc_frontend.plot_kind import PLOT_KIND_SPECS
from zlc_storage.canonical import canonical_text, exact_mapping, normalized_text

__all__ = ["DEFAULT_UPDATE_MS", "LOGIC_KINDS", "LOGIC_NODE_CONFIG_FIELDS",
           "LogicNodeConfig", "PANEL_CONFIG_FIELDS", "PANEL_KINDS",
           "CONSOLE_STATE_SCHEMA", "PanelConfig", "TASK_CONSOLE_STATE_FIELDS",
           "UPDATE_INTERVALS", "console_signal_key", "layout_record",
           "panel_signal_key"]


#: The three domain node families the Logic tab can add.
LOGIC_KINDS = ("measurement", "processor", "task")

LOGIC_NODE_CONFIG_FIELDS = {
    "node_id": str,
    "kind": str,
    "definition_key": dict,
    "title": str,
    "values": dict,
}


def console_signal_key(producer_id: str, output_name: str) -> str:
    """Return the persisted identity of one console-node output.

    Output names such as ``frame`` describe a quantity, not a producer
    instance.  ``producer_id`` is the immutable saved LogicNode id; its editable
    human title is never part of identity.  The picker still presents
    ``title -> output`` while this opaque key is saved by :class:`PanelConfig`
    and used by the live data plane.
    """

    producer = canonical_text(producer_id, "console signal producer id")
    output = canonical_text(output_name, "console signal output name")
    return f"@logic/{producer}/{output}"


def panel_signal_key(panel_id: str, output_name: str) -> str:
    """Return one panel-derived signal's stable, deliberately invisible key.

    A panel title is editable presentation text and therefore cannot identify
    downstream bindings.  ``panel_id`` is persisted with the current layout;
    the picker projects the human title and short output name separately.
    """

    identity = canonical_text(panel_id, "panel signal identity")
    output = canonical_text(output_name, "panel signal output name")
    return f"@panel/{identity}/{output}"


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
    that PRODUCES data.  ``kind`` is one of :data:`LOGIC_KINDS`
    (measurement / processor / task). ``node_id`` is the immutable saved row
    identity used to namespace its outputs; the presented ``title`` never
    participates in identity. ``definition_key`` is the opaque tree
    emitted by the owning catalog's DefinitionKey codec; presentation titles
    never identify a capability.
    ``values`` is the last param-form ``{key: value}`` it was built / run with, so
    reopening its Edit restores them.  A node is always added STOPPED -- nothing
    runs until Start in its Edit."""

    def __init__(
        self,
        *,
        node_id: str | None = None,
        kind: str,
        definition_key: Mapping[str, object],
        title: str,
        values: Mapping[str, object] | None = None,
    ):
        if not isinstance(kind, str):
            raise TypeError("logic kind must be str")
        if kind not in LOGIC_KINDS:
            raise ValueError(f"unknown logic kind {kind!r}; choose from {list(LOGIC_KINDS)}.")
        identity = (
            f"logic_{uuid4().hex}"
            if node_id is None
            else canonical_text(node_id, "logic node_id")
        )
        if not isinstance(definition_key, Mapping) or not definition_key:
            raise TypeError("definition_key must be a non-empty owner codec tree")
        if values is not None and not isinstance(values, Mapping):
            raise TypeError("logic node values must be a mapping or None")
        display_title = normalized_text(title, "logic node title")
        self.node_id = identity
        self.kind = kind
        self.definition_key = copy.deepcopy(dict(definition_key))
        self.title = display_title
        self.values = {} if values is None else dict(values)

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "definition_key": copy.deepcopy(self.definition_key),
            "title": self.title,
            "values": dict(self.values),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LogicNodeConfig":
        data = layout_record(
            payload,
            LOGIC_NODE_CONFIG_FIELDS,
            "LogicNodeConfig",
            discriminator=None,
        )
        result = cls(**data)
        if result.to_dict() != data:
            raise ValueError("LogicNodeConfig is not in current canonical form")
        return result


# ------------------------------------------------------------------ panel vocabulary
#: ``key -> label`` for current TaskConsole panels.  The complete figure
#: vocabulary remains in :mod:`zlc_frontend.plot_kind`; this projection contains
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
    "panel_id": str,
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
        panel_id: str | None = None,
        kind: str,
        title: str = "",
        row: int = 0,
        col: int = 0,
        size: str = "2x2",
        signal: str = "",
        params: Mapping[str, object] | None = None,
    ):
        if not isinstance(kind, str):
            raise TypeError("panel kind must be str")
        if kind not in PANEL_KINDS:
            raise ValueError(f"unknown panel kind {kind!r}; choose from {sorted(PANEL_KINDS)}.")
        if not isinstance(size, str):
            raise TypeError("panel size must be str")
        panel_size_cells(size)              # validate against the limited preset list
        identity = (
            f"panel_{uuid4().hex}"
            if panel_id is None
            else canonical_text(panel_id, "panel_id")
        )
        if not isinstance(title, str):
            raise TypeError("panel title must be str")
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or row < 0
            or isinstance(col, bool)
            or not isinstance(col, int)
            or col < 0
        ):
            raise ValueError("panel row and col must be non-negative int values")
        if not isinstance(signal, str):
            raise TypeError("panel signal must be str")
        if params is not None and not isinstance(params, Mapping):
            raise TypeError("panel params must be a mapping or None")
        self.panel_id = identity
        self.kind = kind
        self.title = title
        self.row = row    # pixel y of the card top-left (no column grid)
        self.col = col    # pixel x of the card top-left (no column grid)
        self.size = size
        # A panel is a view of exactly one typed dataset.  Combining producers is a
        # Processor/join concern; the GUI must not invent an independent-latest
        # expression and then present it as one coherent source.
        self.signal = signal
        self.params = {} if params is None else dict(params)

    @property
    def update_ms(self) -> int:
        """This panel's display refresh interval (ms), one of :data:`UPDATE_INTERVALS`.
        Stored in ``params`` so it round-trips with the saved layout."""

        ms = self.params.get("update_ms", DEFAULT_UPDATE_MS)
        if isinstance(ms, bool) or not isinstance(ms, int) or ms not in UPDATE_INTERVALS:
            raise ValueError(
                f"panel update_ms must be one of {UPDATE_INTERVALS}, got {ms!r}"
            )
        return ms

    def to_dict(self) -> dict[str, object]:
        # ``row``/``col`` are the card's pixel top-left (no column grid).  They are only a SEED for
        # the gravity packer, which re-packs the whole board on load, so a layout's reading order
        # (top-to-bottom, left-to-right) is what round-trips -- exact pixels are recomputed.
        return {
            "panel_id": self.panel_id,
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
        if result.to_dict() != data:
            raise ValueError("PanelConfig is not in current canonical form")
        return result


# ---------------------------------------------------------------- console-state format
#: The persisted DISCRIMINATOR of a saved console layout, written into every layout file as
#: its ``schema`` key and required to match exactly on load.
#:
#: It is a semantic format name, not a Python module path.  Only this exact
#: current record shape is accepted.
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
