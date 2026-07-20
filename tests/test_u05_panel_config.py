"""The panel record sinks, and the module that already held one console record is renamed.

``PanelConfig`` is what a saved console layout is made of: a plot kind, a size preset, a
pixel position and the hub signals it reads.  Nothing about that needs Qt or Matplotlib --
but one line, ``if kind not in PANEL_KINDS``, tied it to the whole render stack, because
the panel vocabulary was derived from the plot-kind table inside the figure module.  The
previous slice moved that table; this one moves the record and the derivation with it.

``zlc_data/logic_node.py`` becomes ``zlc_data/console_records.py``.  A module named after
one of the two records it holds would have been a small lie that costs nothing today and
misleads every later reader.

The golden below was captured from the legacy class immediately before the move and is
deliberately weighted towards the branches that DIVERGE: every kind's default record is
identical (blank input, blank source), so a defaults-only golden would have proved almost
nothing.  What discriminates is the eight rejection paths, the per-kind repeat_mode
vocabulary, the update_ms fallback, and the bare-name canonicalisation -- including the
detail that ``value = fit/x0`` is NOT a bare name (a slash is not an identifier) while
``value = zzz`` is.

Every test that covers this record -- test_console_state_roundtrip.py,
test_panel_hug_layout.py, test_board_gravity.py -- sits outside
``tests/migration_active_tests.txt`` and is frozen: not run, not modified, not evidence.
This file is the oracle.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from zlc_data.console_records import (
    ADDABLE_PANEL_KINDS,
    BLANK_SOURCE,
    DEFAULT_UPDATE_MS,
    PANEL_KINDS,
    UPDATE_INTERVALS,
    PanelConfig,
    panel_allows_multi_slot,
    panel_input_slots,
    repeat_mode_for_kind,
)

REPO = pathlib.Path(__file__).resolve().parents[1]

KINDS = ("2d", "sites", "1d", "monitor", "hist", "pulse", "grid")

#: A default record is the SAME for every kind -- stated once, so the tests below can spend
#: themselves on what actually differs.
DEFAULT = {"title": "", "row": 0, "col": 0, "size": "2x2", "source": "",
           "params": {}, "inputs": [""]}


def test_a_default_record_is_identical_for_every_kind():
    """The blank panel: no signal picked, so no source either.  A fresh panel must NOT
    auto-bind to whatever is running -- that was a real bug, and it is why the site map's
    'occupancy' slot carries a BLANK default rather than a signal name."""

    for kind in KINDS:
        assert PanelConfig(kind=kind).to_dict() == {"kind": kind, **DEFAULT}, kind


@pytest.mark.parametrize("kwargs, expected", [
    # Naming an input is what turns the source on.
    ({"kind": "1d", "inputs": ["cam/frame"]},
     {"source": "value = signal", "inputs": ["cam/frame"]}),
    # A bare name in the source binds the sole input ...
    ({"kind": "1d", "source": "value = zzz"},
     {"source": "value = zzz", "inputs": ["zzz"]}),
    # ... but a slashed hub name is not a bare name, so the input stays blank.
    ({"kind": "1d", "source": "value = fit/x0"},
     {"source": "value = fit/x0", "inputs": [""]}),
    # Two inputs: a multi-slot expression leaves them alone.
    ({"kind": "1d", "inputs": ["a", "b"], "source": "value = signal[0]-signal[1]"},
     {"source": "value = signal[0]-signal[1]", "inputs": ["a", "b"]}),
    # Too-few inputs pad to the kind's slot count.
    ({"kind": "sites", "inputs": []}, {"source": "", "inputs": [""]}),
    # Negative pixels clamp rather than raise -- they are only a packer seed.
    ({"kind": "1d", "row": -5, "col": -3}, {"row": 0, "col": 0}),
    ({"kind": "1d", "title": "T", "row": 7, "col": 9, "size": "1x2"},
     {"title": "T", "row": 7, "col": 9, "size": "1x2"}),
])
def test_the_captured_construction_cases_are_reproduced(kwargs, expected):
    record = PanelConfig(**kwargs).to_dict()
    for key, value in expected.items():
        assert record[key] == value, key


@pytest.mark.parametrize("kind, mode, ok", [
    ("hist", "pool", True),        # a distribution's default and only pooling verb
    ("hist", "roll", False),       # 'roll' is a trace verb; a histogram would ignore it
    ("2d", "create", False),       # a frame has no per-repeat lines to create
    ("monitor", "roll", True),     # the rolling trace is the ONLY kind that rolls
])
def test_repeat_mode_is_validated_against_the_kinds_own_menu(kind, mode, ok):
    if ok:
        assert PanelConfig(kind=kind, params={"repeat_mode": mode}).params["repeat_mode"] == mode
    else:
        with pytest.raises(ValueError):
            PanelConfig(kind=kind, params={"repeat_mode": mode})


@pytest.mark.parametrize("kind, default", [("hist", "pool"), ("2d", "average"),
                                           ("monitor", "average"), ("bogus", "average")])
def test_an_unasked_repeat_mode_is_the_kinds_first_menu_entry(kind, default):
    """Order in the vocabulary IS the default; an unknown kind falls back to the image set."""

    assert repeat_mode_for_kind(kind) == default


@pytest.mark.parametrize("stored, effective", [(200, 200), (999, DEFAULT_UPDATE_MS),
                                               (0, DEFAULT_UPDATE_MS)])
def test_an_out_of_set_refresh_interval_falls_back_to_the_harmonic_default(stored, effective):
    """Off-set rates are not an error but they must not survive, or the console timer base
    stops dividing every panel's rate and shot-coherence quietly breaks."""

    assert PanelConfig(kind="1d", params={"update_ms": stored}).update_ms == effective
    assert effective in UPDATE_INTERVALS


def test_the_record_round_trips_through_its_codec():
    payload = PanelConfig(kind="1d", inputs=["cam/frame"]).to_dict()
    assert PanelConfig.from_dict(payload).to_dict() == payload


@pytest.mark.parametrize("mutate, error", [
    (lambda d: d.pop("size"), ValueError),                    # exact key set
    (lambda d: d.update(row="x"), TypeError),                 # exact field types
    (lambda d: d.update(row=-1), ValueError),                 # a STORED negative is a broken file
    (lambda d: d.update(col=-1), ValueError),
    (lambda d: d.update(inputs=[1]), TypeError),
    (lambda d: d.update(kind="zzz"), ValueError),
    (lambda d: d.update(source="value = other"), ValueError),  # would re-normalise on read
])
def test_every_captured_rejection_path_still_refuses(mutate, error):
    """The reader refuses rather than coercing.  The last case is the subtle one: the record
    is well-formed, but reading it would rewrite ``inputs``, so a writer that emitted it and a
    reader that loaded it would disagree -- caught at the boundary instead of silently."""

    payload = PanelConfig(kind="1d", inputs=["cam/frame"]).to_dict()
    mutate(payload)
    with pytest.raises(error):
        PanelConfig.from_dict(payload)


def test_construction_clamps_a_negative_position_but_loading_refuses_one():
    """Deliberately asymmetric, and worth keeping honest: a caller passing -5 means 'top
    left', while a stored -1 means the file is wrong."""

    assert PanelConfig(kind="1d", row=-5).row == 0
    payload = PanelConfig(kind="1d").to_dict()
    payload["row"] = -1
    with pytest.raises(ValueError):
        PanelConfig.from_dict(payload)


def test_the_slot_helpers_answer_for_every_kind_and_for_none():
    assert panel_input_slots("sites")[0][0] == "occupancy"
    assert panel_allows_multi_slot("sites") is False
    for kind in set(KINDS) - {"sites"}:
        assert panel_input_slots(kind) == (("signal", "", "the hub signal to plot"),), kind
        assert panel_allows_multi_slot(kind) is True, kind
    # An unknown kind gets the universal single slot rather than an exception: the Setting
    # must still render something for a record the vocabulary no longer knows.
    assert panel_input_slots("bogus") == (("signal", "", "the hub signal to plot"),)


def test_the_addable_menu_is_the_panel_kinds_minus_pulse():
    """``panel=False`` means 'not addable live', NOT 'cannot be a panel' -- a saved pulse
    figure still seeds a normal panel, which is why PANEL_KINDS keeps it."""

    assert set(PANEL_KINDS) - set(ADDABLE_PANEL_KINDS) == {"pulse"}
    assert list(ADDABLE_PANEL_KINDS) == [k for k in PANEL_KINDS if k != "pulse"]
    assert BLANK_SOURCE == ""


def test_the_record_module_reaches_for_no_toolkit_and_no_renderer():
    """The whole reason it could sink, asserted rather than assumed."""

    import zlc_data.console_records as records

    tree = ast.parse(pathlib.Path(records.__file__).read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    roots = {name.split(".")[0] for name in modules} - {"__future__", "typing"}
    assert roots <= {"zlc_data", "zlc_storage"}, roots


def test_the_shell_reads_both_records_and_the_panel_vocabulary_from_the_new_home():
    """Structural, so this file keeps no legacy-tree dependency of its own.

    A shell that quietly kept its own copy of any of these would pass every behaviour test
    above -- only the import graph can see it."""

    tree = ast.parse((REPO / "Zou_lab_control" / "frontend" / "task_console.py")
                     .read_text(encoding="utf-8"))
    sources: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                sources.setdefault(alias.name, set()).add(node.module)
    for name in ("PanelConfig", "LogicNodeConfig", "PANEL_KINDS", "ADDABLE_PANEL_KINDS",
                 "PANEL_INPUT_FORMAT", "PANEL_INPUT_SLOTS", "PANEL_SINGLE_SLOT_KINDS",
                 "UPDATE_INTERVALS", "DEFAULT_UPDATE_MS", "layout_record",
                 "panel_input_slots", "panel_allows_multi_slot"):
        assert sources.get(name) == {"zlc_data.console_records"}, (name, sources.get(name))


def test_the_legacy_name_is_the_same_class():
    """The shell alias must BE the moved class, not a lookalike left behind -- the frontend
    package re-exports ``PanelConfig``, and devtools / figure_viewer construct it."""

    from Zou_lab_control.frontend.task_console import PanelConfig as shell_name

    assert shell_name is PanelConfig
