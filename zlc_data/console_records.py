"""The console's saved records: what a panel is, what a logic node is, and their codecs.

These are the two things a console layout is made of.  A PANEL is a view -- a plot kind,
a size preset, a pixel position, and the hub signals it reads.  A LOGIC NODE is what
PRODUCES data on the Logic tab.  Both are plain values with a JSON codec, and neither
needs Qt or a renderer to say what it is, so both live here and the windows that display
them, the workspace writer that persists them and any future reader share one definition
instead of agreeing by convention.

``layout_record`` is the validator all three console records use (panel, logic node,
console state): exact key set, exact field types, no silent coercion.

The panel VOCABULARY below -- which kinds exist, what shape each accepts, how many signal
slots it opens with -- is derived from :mod:`zlc_data.plot_kind` rather than restated.
That is what lets ``PanelConfig`` validate its own ``kind`` here: refusing an unknown kind
is a question about the vocabulary, not about drawing.  Until the plot-kind table split,
that single ``if kind not in PANEL_KINDS`` line was enough to tie this record to the whole
Matplotlib stack.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from zlc_data.panel_size import panel_size_cells
from zlc_data.plot_kind import PLOT_KIND_SPEC_BY_KEY, PLOT_KIND_SPECS
from zlc_data.repeat_modes import IMAGE_REPEAT_MODES
from zlc_storage.canonical import exact_mapping

__all__ = ["ADDABLE_PANEL_KINDS", "BLANK_SOURCE", "DEFAULT_INPUT_SLOTS", "DEFAULT_UPDATE_MS",
           "LOGIC_KINDS", "LOGIC_NODE_CONFIG_FIELDS", "LogicNodeConfig",
           "PANEL_CONFIG_FIELDS", "PANEL_INPUT_FORMAT", "PANEL_INPUT_SLOTS", "PANEL_KINDS",
           "CONSOLE_STATE_SCHEMA", "PANEL_SINGLE_SLOT_KINDS", "PanelConfig",
           "TASK_CONSOLE_STATE_FIELDS", "UPDATE_INTERVALS", "layout_record",
           "panel_allows_multi_slot", "panel_input_slots", "repeat_mode_for_kind",
           "repeat_modes_for_kind"]


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
# EVERY plot kind in the ONE vocabulary ``zlc_data.plot_kind`` is a console panel kind -- it
# renders through the SAME ``PanelCard`` (which dispatches on the kind: a 2D frame, a site map,
# a histogram, a 1-D curve, a pulse timeline ...), so a saved figure of ANY kind seeds a normal
# panel and reads its ``value`` off a hub signal.  The ``panel`` flag is NOT "can this be a
# panel"; it is ONLY "is this offered in the live Add-Panel dropdown" (see ADDABLE_PANEL_KINDS)
# -- ``pulse`` is ``panel=False`` because you do not add a blank pulse panel live (it is
# reproduced from a saved recipe / a fired sequence), but it IS a real panel kind that seeds +
# renders exactly like every other.  All the per-kind tables below are derived from the WHOLE
# table so pulse (and any future kind) works on the seed path with no parallel literal.

#: ``key -> label`` for EVERY console panel kind -- the panel/card/frame title + ``PanelConfig``
#: kind validation (so a saved ``pulse`` figure seeds a normal panel).  Insertion order = table order.
PANEL_KINDS: dict[str, str] = {spec.key: spec.label for spec in PLOT_KIND_SPECS}

#: The subset offered in the live Add-Panel dropdown (``panel=True``): the kinds you add a BLANK
#: panel of and then wire to a signal.  ``pulse`` is excluded (you do not add a blank pulse panel
#: live), but it is still a full panel kind on the SEED path.  Insertion order = menu order.
ADDABLE_PANEL_KINDS: dict[str, str] = {spec.key: spec.label
                                       for spec in PLOT_KIND_SPECS if spec.panel}

#: What signal SHAPE each plot kind expects as its ``value`` -- shown in the panel's Setting so it
#: is clear which signals fit (a Site map wants a per-site vector, a 2D image wants a frame).  The
#: per-kind contract is declared in the vocabulary; everything else about the source (the
#: multi-slot picker + ``value = ...`` expression) is universal and kind-agnostic.
PANEL_INPUT_FORMAT: dict[str, str] = {spec.key: spec.input_format for spec in PLOT_KIND_SPECS}

#: The STARTING slot(s) each plot kind opens with, ``(label, default-signal, tooltip)``.  Every kind
#: uses the SAME source MECHANISM (a signal picker + a ``value = ...`` expression box); whether a
#: kind can GROW extra slots is declared by PANEL_SINGLE_SLOT_KINDS below.  A plot reads its picked
#: input(s) as ``signal`` / ``signal[i]``.  The DEFAULT single blank slot lives here; the per-kind
#: overrides (e.g. the site map's "occupancy" slot) come from the vocabulary.
DEFAULT_INPUT_SLOTS = (("signal", "", "the hub signal to plot"),)
PANEL_INPUT_SLOTS: dict[str, tuple[tuple[str, str, str], ...]] = {
    spec.key: spec.input_slots for spec in PLOT_KIND_SPECS if spec.input_slots
}

#: Which plot kinds take EXACTLY ONE signal (no +signal / -signal slot-growing).  The
#: signal-expression MECHANISM is universal -- every kind has it -- but a SINGLE-slot kind cannot
#: add more slots because its auxiliary data is resolved from signal[0]: the site map pulls its
#: ring centres + frame underlay from signal[0]'s producing node, so a 2nd slot would be meaningless.
PANEL_SINGLE_SLOT_KINDS: frozenset = frozenset(spec.key for spec in PLOT_KIND_SPECS
                                               if spec.single_slot)


def panel_input_slots(kind: str) -> tuple[tuple[str, str, str], ...]:
    """The input slots for a plot kind -- ``[(label, default_signal, tooltip)]``.  The
    SINGLE source of how many signals a plot consumes and what each means."""
    return PANEL_INPUT_SLOTS.get(str(kind), DEFAULT_INPUT_SLOTS)


def panel_allows_multi_slot(kind: str) -> bool:
    """Whether a plot kind can grow extra signal slots (+signal / -signal).  Data-driven from
    ``PANEL_SINGLE_SLOT_KINDS`` -- the site map is single-slot (its centres/underlay come from
    signal[0]); every other kind is multi-slot.  Read by PanelCard so the slot UI is declared
    in ONE place, never an inline per-kind check scattered through the widget."""
    return str(kind) not in PANEL_SINGLE_SLOT_KINDS


def repeat_modes_for_kind(kind: str) -> tuple[str, ...]:
    spec = PLOT_KIND_SPEC_BY_KEY.get(str(kind))
    return tuple(spec.repeat_modes) if spec and spec.repeat_modes else tuple(IMAGE_REPEAT_MODES)


_MISSING_REPEAT_MODE = object()


def repeat_mode_for_kind(kind: str, value: object = _MISSING_REPEAT_MODE) -> str:
    modes = repeat_modes_for_kind(kind)
    if value is _MISSING_REPEAT_MODE:
        return modes[0]
    if not isinstance(value, str) or value not in modes:
        raise ValueError(
            f"invalid repeat_mode {value!r} for panel kind {kind!r}; "
            f"choose from {list(modes)}"
        )
    return value


#: Per-panel display refresh intervals (ms) the operator can pick from.  A FIXED, harmonic set
#: (100 x {1,2,4,8}) so the SMALLEST selected interval divides every other -- the console timer runs
#: at that base (the GCD) and each panel refreshes every ``update_ms // base`` ticks.  The payoff is
#: PHASE ALIGNMENT: panels that share a beat fire on the SAME tick and read the SAME hub snapshot,
#: so a 2-D frame and its site-map stay shot-coherent; a fast panel (100 ms) just refreshes more
#: often in between.  Limiting the choices to this set is what makes the synchronisation exact --
#: arbitrary per-panel rates could never co-align.
UPDATE_INTERVALS = (100, 200, 400, 800)
DEFAULT_UPDATE_MS = 400

#: A fresh plot panel is BLANK: a pure view is fully decoupled from acquisition, so it shows nothing
#: until the user picks a hub signal in its Setting -- it must NOT auto-bind to any node's signal.
#: An empty source is the blank state; the refresh path treats it (and a source that produces None)
#: as "pick a signal" rather than an error, so a blank panel sits quietly until wired.
BLANK_SOURCE = ""

PANEL_CONFIG_FIELDS = {
    "kind": str,
    "title": str,
    "row": int,
    "col": int,
    "size": str,
    "source": str,
    "params": dict,
    "inputs": list,
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
        source: str | None = None,
        params: Mapping[str, object] | None = None,
        inputs: Sequence[str] | None = None,
    ):
        if kind not in PANEL_KINDS:
            raise ValueError(f"unknown panel kind {kind!r}; choose from {sorted(PANEL_KINDS)}.")
        panel_size_cells(size)              # validate against the limited preset list
        self.kind = str(kind)
        self.title = str(title)
        self.row = max(0, int(row))    # pixel y of the card top-left (no column grid)
        self.col = max(0, int(col))    # pixel x of the card top-left (no column grid)
        self.size = str(size)
        # The per-slot signal names (signal[0], signal[1], ...): one hub signal per input
        # slot of this plot kind.  Defaults to each slot's default signal so a freshly
        # added panel already names what it wants; a saved layout restores its picks.
        slots = panel_input_slots(self.kind)
        if inputs is None:
            self.inputs = [d for _, d, _ in slots]
        else:
            self.inputs = [str(s) for s in inputs]
            if len(self.inputs) < len(slots):       # pad to the kind's slot count
                self.inputs += [d for _, d, _ in slots[len(self.inputs):]]
        # A pure-view plot is BLANK until a signal is picked (decoupled from acquisition).
        # When the input already names a signal, the default source is ``value = signal``;
        # an empty input leaves it blank ("pick a signal").  A saved layout keeps its
        # stored source verbatim.
        if source is None:
            source = "value = signal" if self.inputs and self.inputs[0] else BLANK_SOURCE
        self.set_source(source)
        self.params = dict(params or {})
        if "repeat_mode" in self.params:
            self.params["repeat_mode"] = repeat_mode_for_kind(
                self.kind,
                self.params["repeat_mode"],
            )

    def set_source(self, source: str) -> None:
        """Set the expression and keep a single-slot bare-name binding canonical.

        ``value = <hub signal>`` names the sole input, whereas ``value = signal``
        reads the existing picker binding and multi-slot/custom expressions leave
        the inputs untouched.  Constructor, GUI edit and slot mutations all pass
        through this owner so the layout writer cannot emit a record its reader
        would normalize differently.
        """
        from zlc_data.signal_expr import IDENTITY_SOURCE_RE

        self.source = str(source)
        match = IDENTITY_SOURCE_RE.fullmatch(self.source.strip())
        if match and match.group(1) != "signal" and len(self.inputs) == 1:
            self.inputs[0] = match.group(1)

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
            "source": self.source,
            "params": dict(self.params),
            "inputs": list(self.inputs),
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
        if any(type(name) is not str for name in data["inputs"]):
            raise TypeError("panel inputs must contain strings")
        result = cls(**data)
        if result.source != data["source"] or result.inputs != data["inputs"]:
            raise ValueError("PanelConfig is not in current canonical form")
        return result


# ---------------------------------------------------------------- console-state format
#: The persisted DISCRIMINATOR of a saved console layout, written into every layout file as
#: its ``schema`` key and required to match exactly on load.
#:
#: It reads like a module path and it is NOT one.  It is a FORMAT NAME that happens to have
#: been minted from the class's location at the time, and it must NOT follow the class as the
#: class moves: every layout a user has already saved carries this exact string, and
#: ``exact_mapping`` refuses a payload whose discriminator differs.  Re-deriving it from the
#: new module path would make every saved dashboard unopenable, and would do so silently at
#: the moment the operator tries to load their work rather than at any point a test would
#: notice.  Pinned by test_u05_console_state_format so a later tidy-up cannot quietly do it.
CONSOLE_STATE_SCHEMA = "Zou_lab_control.frontend.TaskConsoleState"

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
