"""The grouped-signal picker's data layer: names -> tree groups -> a filled combo.

`FluentTreeComboBox` migrated into this package but its data helpers did not -
they stayed in the old monolith's `frontend/param_widgets.py`, so a new consumer
could get the widget and then had to reach back into the old tree to fill it.
That is the reverse of this migration's package direction, and it is why no
window under `Zou_lab_control/workbench/` used the picker at all.

Pure Qt-combo utilities: only this package's own widgets plus plain data.  No
domain types, no hub, no `ParamDecl` - which is exactly why the cluster could
move as a unit without dragging the old tree with it.
"""

from __future__ import annotations

from .fluent import FluentTreeComboBox, signals_blocked as _signals_blocked


# The grouped-signal-picker helper cluster LIVES HERE (the leaf every form module depends on;
# task_console / figure_viewer forward-import it).  It used to live in task_console with lazy
# back-imports from this module -- a cycle that contradicted this file's leaf contract.  Pure
# Qt-combo utilities: only qt_widgets (FluentTreeComboBox, signals_blocked) + plain data.


def _common_token_prefix(names) -> str:
    """The longest common UNDERSCORE-token prefix of ``names`` -- e.g. both
    ``analysis_rate`` and ``analysis_score`` share ``analysis_``.
    Empty for fewer than two names or no shared leading token.  Used to strip the producer
    prefix the hub prepends from a grouped signal picker's labels (the producer is the group
    header, so its name need not repeat in every signal)."""
    import os.path as _op
    names = [str(n) for n in names]
    if len(names) < 2:
        return ""
    common = _op.commonprefix(names)
    cut = common.rfind("_")
    return common[: cut + 1] if cut >= 0 else ""


def signal_state(name, formats) -> str:
    """A signal has exactly TWO states (G3): "ready" when it is PUBLISHED on the hub right now
    (so it has a live shape in ``formats``), else "waiting" -- it is declared by a node that has
    not started / not produced yet.  No more none/unbound/error/mid-run/unpublished clutter."""
    return "ready" if formats.get(str(name)) else "waiting"


def _signal_short_label(name, group, labels) -> str:
    """The leaf label for ``name`` under its producer node in the picker nest: the SHORT signal name --
    the producing node's prefix stripped (``temperature_survival`` -> ``survival``, ``frame`` ->
    ``frame``), passed in via ``labels`` (the ``short_names_provider`` map, built from each running
    node's prefix; #design: the nest already names the producer, so the leaf is the short NAME, NOT the
    verbose SignalSpec axis label like ``camera image``).  For a signal with no mapped short name (a
    declared-but-not-running node), fall back to the shared-token prefix stripped from the group, else
    the bare name."""
    short = (labels or {}).get(str(name))
    if short:
        return str(short)
    strip = _common_token_prefix(group)
    if strip and name.startswith(strip) and len(name) > len(strip):
        return name[len(strip):]
    return str(name)


def grouped_signal_items(names, sources, formats, labels=None) -> list:
    """``[(display, bare_name | None)]`` for a signal picker, GROUPED by producing node: a
    non-selectable bold header per node (``bare_name`` is ``None``), then that node's signals
    indented beneath it -- shown by their HUMAN label (the SignalSpec the producing node declares),
    with the SHAPE and the two-state tag (``    Loading rate  [(35,)] ready`` / ``    Survival
    waiting``).  ``data`` stays the BARE signal name (the binding key); only the DISPLAY is humanised.
    The ONE source every signal picker shares (plot panel AND logic-node source)."""
    names = sorted(str(n) for n in (names or []))
    sources = dict(sources or {})
    formats = dict(formats or {})
    labels = dict(labels or {})
    by_producer: dict[str, list[str]] = {}
    for name in names:
        for p in ([str(p) for p in (sources.get(name) or [])] or ["(unbound)"]):
            by_producer.setdefault(p, []).append(name)
    items: list[tuple[str, str | None]] = []
    for producer in sorted(by_producer, key=lambda p: (p == "(unbound)", p.lower())):
        group = by_producer[producer]
        items.append((producer, None))            # group header (rendered disabled + bold)
        for name in group:
            short = _signal_short_label(name, group, labels)
            fmt = formats.get(name)
            state = signal_state(name, formats)
            shape = f"  [{fmt}]" if fmt else ""
            items.append((f"    {short}{shape}  {state}", name))
    return items


def signal_tree_groups(names, sources, formats, labels=None) -> list:
    """``[(producer, [(leaf_label, bare_name, full_label)])]`` for the COLLAPSIBLE tree picker
    (G2): one expandable group per producing node; each leaf's ``leaf_label`` shows the HUMAN signal
    label + shape + ready/waiting state (in the tree), and its ``full_label`` is the producer-
    qualified ``"<producer> · <label>"`` painted when the combo is COLLAPSED (frame-title aligned,
    G3).  Built from the same producer grouping + ``_signal_short_label`` as
    :func:`grouped_signal_items` -- ONE source, so neither ever shows a raw ``temperature_survival``."""
    names = sorted(str(n) for n in (names or []))
    sources = dict(sources or {})
    formats = dict(formats or {})
    labels = dict(labels or {})
    by_producer: dict[str, list[str]] = {}
    for name in names:
        for p in ([str(p) for p in (sources.get(name) or [])] or ["(unbound)"]):
            by_producer.setdefault(p, []).append(name)
    groups: list = []
    for producer in sorted(by_producer, key=lambda p: (p == "(unbound)", p.lower())):
        group = by_producer[producer]
        leaves = []
        for name in group:
            short = _signal_short_label(name, group, labels)
            fmt = formats.get(name)
            shape = f"  [{fmt}]" if fmt else ""
            leaf_label = f"{short}{shape}  {signal_state(name, formats)}"
            leaves.append((leaf_label, name, f"{producer} · {short}"))
        groups.append((producer, leaves))
    return groups


def fill_grouped_signal_combo(combo, *, names, sources, formats, current, none_label=None, labels=None) -> None:
    """Populate ``combo`` with every live hub signal GROUPED by producing node (via
    :func:`grouped_signal_items`): bold non-selectable headers, indented signals (data = the
    BARE name).  ``none_label`` adds a leading empty choice; a not-yet-published ``current``
    is kept selectable.  Read the pick back with ``currentData()`` (the bare name) -- the
    visible label is indented.  Shared by the plot panel's slot picker and the logic-node
    source field, so the nested picker is identical everywhere."""
    cur = str(current or "")
    # A configured input may NAME a signal that is declared but not published yet -- for example a
    # PulseScan y supplied by a not-yet-started external producer.  The
    # binding is by NAME, resolved at RUN time, so keep such a name in the pool: BOTH the tree and the
    # flat picker then render it as a "waiting" leaf AND read it back.  Single-sources the docstring's
    # "kept selectable" promise across both branches -- the tree branch used to drop a not-listed name,
    # so ``read_editable_combo`` returned '' and the configured input vanished (e.g. a Start that then
    # built the node with an empty y-expression input -> every point NaN).  ``signal_state`` renders the
    # added name honestly as "waiting" (it has no live shape in ``formats``).
    names = list(names or [])
    if cur and cur not in {str(n) for n in names}:
        names = [*names, cur]
    if isinstance(combo, FluentTreeComboBox):
        # The collapsible-tree picker (G2): one expandable producer group, leaves = signals.
        with _signals_blocked(combo):
            combo.set_signal_tree(signal_tree_groups(names, sources, formats, labels),
                                  current=cur, none_label=none_label)
        return
    with _signals_blocked(combo):
        combo.clear()
        if none_label is not None:
            combo.addItem(none_label, "")
        items = grouped_signal_items(names, sources, formats, labels)
        for label, name in items:
            if name is None:                      # group header: visible but not selectable
                combo.addItem(label, None)
                item = combo.model().item(combo.count() - 1)
                if item is not None:
                    item.setEnabled(False)
                    font = item.font(); font.setBold(True); item.setFont(font)
                continue
            combo.addItem(label, name)            # indented signal; data is the bare name
        idx = combo.findData(cur)
        # No match: select the leading none-row if there is one, else leave it BLANK (index -1)
        # -- never auto-land on a disabled group HEADER (data None), whose label would otherwise
        # read back as if it were the chosen signal.
        if idx < 0 and none_label is not None:
            idx = 0
        combo.setCurrentIndex(idx)


def read_editable_combo(combo) -> str:
    """Read an EDITABLE combo that pairs a display LABEL with a bare-value ``data`` (the grouped
    signal picker, the pulse-param picker).  Returns the selected item's data when the visible
    text still matches that item (a real pick) -- else the typed text (a not-yet-published custom
    name).  A plain ``currentData()`` would return the STALE previously-selected data after the
    user types a new name into the line edit (Qt does not move currentIndex on free text), so the
    fresh name would be silently dropped; a disabled header (data ``None``) falls through to ''."""
    if isinstance(combo, FluentTreeComboBox):
        return combo.current_signal()             # the tree picker stores the bare name on the leaf
    idx = combo.currentIndex()
    text = combo.currentText()
    if idx >= 0 and text == combo.itemText(idx):
        data = combo.itemData(idx)
        if data is not None:
            return str(data).strip()
    return text.strip()


def coerce_short_labels(provider) -> dict:
    """Normalise a ``{full hub name -> short name}`` callback into the ``labels`` map every grouped
    signal picker feeds ``fill_grouped_signal_combo``: callable-guard, ``str()`` both ends, drop empty
    short names, swallow any provider exception to ``{}``.  The ONE source the signal_expr / plot
    Setting slot / form signal pickers share so they render IDENTICALLY (#combo-parity) instead of four
    hand-copied dict comprehensions (the 4th of which had already dropped the try/except)."""
    if not callable(provider):
        return {}
    try:
        return {str(n): str(s) for n, s in dict(provider()).items() if s}
    except Exception:
        return {}




__all__ = [
    "coerce_short_labels",
    "fill_grouped_signal_combo",
    "grouped_signal_items",
    "read_editable_combo",
    "signal_state",
    "signal_tree_groups",
]
