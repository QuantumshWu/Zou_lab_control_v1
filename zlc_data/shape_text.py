"""The ONE spelling of a signal's shape, a camera's frame names, and a bound pulse field.

Both surfaces have to say the same thing about the same signal.  The domain
publishes ``frame_0 .. frame_{N-1}`` and the console's picker offers those names
before any node has started; a running block reads ``5 x 8 x (3)`` and the
schema-driven declared path must read it identically before a value is buffered.
Every one of those sentences is produced HERE, so a name or a shape can have only
one spelling in the project.

These are pure text derivations over shapes and counts - stdlib and numpy only -
which is why they SINK into ``zlc_data`` rather than staying behind a port: the
render layer needs the STRING, not a live domain object.  They were extracted
from ``operations/logic.py``, which cannot move (it reaches devices and the signal
hub); the legacy module imports them back, so every existing caller keeps
resolving the SAME function objects.

Admission rule: a pure derivation of TEXT THAT MORE THAN ONE SURFACE MUST AGREE
ON.  That started as domain-publishes / GUI-reproduces (a shape's spelling, a
published signal's name) and now also covers text two SHELLS must spell alike:
``slot_label`` names a bound pulse field and both the pulse editor and the task
console's pulse-scan form show it, so it belongs at the bottom rather than inside
whichever shell happened to define it first.  No knowledge of who is asking.
Anything needing a running node belongs in ``zlc_frontend.domain_ports`` instead.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = [
    "indexed_unique_name",
    "strip_node_prefix",
    "contract_shape_label",
    "describe_shape",
    "format_dims",
    "measurement_slug",
    "slot_label",
]


def format_dims(dims) -> str:
    """The ONE spelling of a dims tuple for the GUI: axes joined by the ``×`` glyph
    (``40×20``), and ``"1"`` only when the caller needs the multiplicative size of
    an empty axis domain (for example the mandatory single point of a no-scan
    dataset).  An empty ``data_shape`` is *not* a length-one data axis;
    :func:`contract_shape_label` omits that group entirely.  EVERY surface that
    turns a signal's shape into a display string routes through this module, so
    ``(40, 20)`` (numpy-tuple spelling) can never appear beside ``40×20`` again."""
    parts = tuple(str(int(n)) for n in (dims or ()))
    return "×".join(parts) if parts else "1"


def contract_shape_label(repeat, point_count, data_shape) -> str:
    """Spell the physical ``R × P [× (*data_shape)]`` tensor contract.

    ``P`` is one storage dimension, even when :class:`PointLayout` maps it to
    several logical scan axes.  Logical topology belongs in separate axis
    metadata; splicing it into this string would make an ``(R, P)`` scalar
    dataset look like a higher-rank ndarray.  The schema-driven and value-driven
    paths share this owner so their physical shape can never drift.
    """
    p = int(point_count)
    if p <= 0:
        raise ValueError("point_count must be positive")
    ds = tuple(int(n) for n in (data_shape or ()))
    cell_domain = f"{int(repeat)} × {p}"
    return cell_domain if not ds else f"{cell_domain} × ({format_dims(ds)})"


def describe_shape(value, *, point_count=None, data_shape=None) -> str:
    """A standardized shape string read straight from a published VALUE -- the SINGLE
    way the GUI says what a signal looks like, AUTO-EXTRACTED from real data rather
    than a hand-typed name->format map (which silently drifts from what a node really
    emits).  ``scalar`` for a 0-d / Python number; ``None`` -> ``"—"`` (no value yet).

    When the value is a registered signal tensor (its shape matches the declared
    physical ``(repeat, P, *data_shape)``) it is shown in contract form
    ``repeat × P [× (data)]``.  P is always the single physical point-storage
    axis: a scalar 7×7×7 logical scan is ``1 × 343`` (never a fabricated
    ``(1)`` and never a fake rank-five tensor), while a per-site 1-D scan is
    ``5 × 8 × (3)``.  Logical point-axis names and sizes are displayed as
    metadata rather than being spliced into the ndarray shape.  A no-scan
    single-point camera reads ``5 × 1 × (96×128)`` and per-site occupancy
    reads ``5 × 1 × (35)``; the site layout is a presentation concern, not
    a point-grid alias.  Otherwise the raw numpy shape is shown directly."""
    if value is None:
        return "—"
    shape = tuple(int(n) for n in np.shape(value))
    if shape == ():
        return "scalar"
    dsh = tuple(int(n) for n in (data_shape or ()))
    if point_count is not None and len(shape) == 2 + len(dsh) \
            and shape[1] == int(point_count) \
            and tuple(shape[2:]) == dsh:
        return contract_shape_label(shape[0], point_count, dsh)
    # Raw shape (schema unknown): the SAME ``×`` spelling as the contract form and the flow-graph
    # labels -- never the numpy-tuple ``(96, 128)`` that made the same signal read two different ways.
    return f"({format_dims(shape)})"


def measurement_slug(name: str) -> str:
    """Canonical machine token for a measurement, derived from its display ``name``
    (lower-case, non-alphanumeric runs -> single ``_``, trimmed).  ONE source: the
    node prefix + every published signal name derive from this, so the measurement is
    called the same thing in the Add-Panel list, the signal-flow legend, and the hub
    signal names -- never a separately hand-typed abbreviation that drifts."""
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", str(name).lower())).strip("_")


def slot_label(kind: str, target: str, *, base_1: bool = True) -> str:
    """The STATE-FREE, INDEX-based label for a bound pulse field from (kind, target) ALONE.

    The raw ``target`` is an INTERNAL handle -- a 0-based period index (``duration``),
    ``"<bus>@<period_index>"`` (``dac``), or a channel/bus name (``delay``) -- meaningless to
    show verbatim (the user's "a1  duration @ 1" complaint: "what is 1?").  This names the PERIOD
    by its 1-based INDEX (``Period 3``, matching the 'Period N/M' on the card) / channel / bus
    WITHOUT a ``PulseTableState``, so it works on the flat row tuples the pulse editor + the
    task-console pulse-scan form carry where no state object is in hand.  The COMPLEMENT is
    ``timing.pulse_table.scan_target_label`` -- the STATE-FUL, NAME-based label (``probe duration``)
    for callers that DO hold a state.  The two are NOT duplicates: same question, different input
    (index vs name)."""

    target = str(target)
    off = 1 if base_1 else 0
    if kind == "duration":
        try:
            return f"Period {int(target) + off} duration"
        except ValueError:
            return f"Period {target} duration"
    if kind == "dac":
        bus, sep, period = target.partition("@")
        if sep:
            try:
                return f"{bus} (Period {int(period) + off})"
            except ValueError:
                return f"{bus} (Period {period})"
        return f"{bus} DAC"
    if kind == "delay":
        return f"{target} delay"            # the channel / bus name is the information
    return target


#: A trailing ``" #N"`` index, so re-indexing a name strips the old number instead of nesting.
_INDEX_SUFFIX_RE = re.compile(r"^(.*?)\s*#\d+$")


def indexed_unique_name(base: str, taken) -> str:
    """``"<root> #N"`` with the smallest ``N >= 1`` not already in ``taken``.  Any ``#k`` already
    on ``base`` is stripped first, so re-indexing a loaded ``"1D vector #2"`` re-derives a clean
    number rather than nesting (idempotent for an already-clean layout).

    Admission: panel titles, logic-node titles and Edit-tab titles must agree on this spelling --
    three surfaces, one rule.
    """
    text = str(base or "panel").strip() or "panel"
    m = _INDEX_SUFFIX_RE.match(text)
    root = (m.group(1).strip() if m else text) or "panel"
    taken = set(taken)
    n = 1
    while f"{root} #{n}" in taken:
        n += 1
    return f"{root} #{n}"


def strip_node_prefix(full: str, prefix: str) -> str:
    """The SHORT signal name = the hub name minus its producing node's disambiguating prefix
    (``analysis_rate`` -> ``rate``, ``temperature_survival`` -> ``survival``, ``frame`` ->
    ``frame``).  The ONE rule the Logic tab AND the signal picker share, so the nest leaf is ALWAYS the
    short name -- never the full prefixed key, never the verbose axis label."""
    full = str(full)
    prefix = str(prefix or "")
    return full[len(prefix):] if (prefix and full.startswith(prefix) and len(full) > len(prefix)) else full
