"""Contract: the console's per-panel/per-node facts each live in ONE source.

Audit batch (console): the declared-keys kind ladder is ``_node_bare_keys`` alone (#19); a
panel's declared PANEL_PARAMS defaults are resolved by ``_resolved_param`` -- never a re-typed
consume-site literal (#20); the expression namespace helpers come from ``signal_expr``'s ONE
builder, GUI == node (#21); the camera row's display name comes from ``readout.camera_spec()``
(#23); the off-hub display pipeline's reserved words (``TASK_FRAME_KEY`` / ``MID_RUN_TAG`` /
``DEFAULT_MID_RUN_KEY``) are module constants shared by every writer and reader (#24).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import Zou_lab_control.neutral_atom as na
from conftest import add_logic_row, make_console


def test_resolved_param_reads_declared_defaults_and_respects_explicit_false():
    """#20: absent key -> the kind's DECLARED default (derived from PANEL_PARAMS, no literal
    here); present key wins even when falsy (False/0 are legal stored values for bool/int
    knobs -- a truthiness test would silently revert them to the default)."""
    from Zou_lab_control.frontend.task_console import PANEL_PARAMS, _resolved_param

    for kind in ("hist", "monitor"):
        for decl in PANEL_PARAMS[kind]:
            assert _resolved_param(kind, {}, decl.key) == decl.default        # declared, not re-typed
            assert _resolved_param(kind, {decl.key: decl.default}, decl.key) == decl.default
    assert _resolved_param("hist", {"ylog": False}, "ylog") is False          # explicit False sticks
    assert _resolved_param("monitor", {"show_dist": False}, "show_dist") is False
    assert _resolved_param("hist", {"bins": 0}, "bins") == 0                  # explicit 0 sticks


def test_gui_namespace_layers_the_one_helper_set_and_excludes_it_from_references():
    """#21 + #24: the console namespace == signal_expr's helper set layered on the hub view
    plus the reserved keys (each a module constant); the reference scanner excludes exactly
    the shared helper names, so a numpy/np expression maps to its REAL hub inputs only."""
    from Zou_lab_control.frontend.task_console import (
        COORD_FRAMES_KEY, SIG_VERSIONS_KEY, TASK_FRAME_KEY, TaskConsole)
    from Zou_lab_control.neutral_atom.operations.signal_expr import NAMESPACE_HELPERS

    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        ns = con._expression_namespace()
        for key in NAMESPACE_HELPERS:                     # GUI capability == node capability
            assert key in ns
        assert ns["numpy"] is ns["np"]                    # the alias exists on BOTH sides now
        for key in (TASK_FRAME_KEY, SIG_VERSIONS_KEY, COORD_FRAMES_KEY):
            assert key in ns                              # reserved view keys, spelled once
        refs = TaskConsole._referenced_signals("value = np.log(numpy.mean(x)) + math.pi")
        assert refs == {"x"}                              # helpers excluded via the ONE constant
        assert TaskConsole._referenced_signals("value = signal") == set()
    finally:
        con.shutdown()
        exp.close()


def test_task_frame_source_and_injection_share_one_constant():
    """#24: the task mid-run panel's source expression and the namespace injection read the
    SAME reserved key -- a drift between them would blank the panel silently."""
    from Zou_lab_control.frontend.task_console import TASK_FRAME_KEY, MID_RUN_TAG

    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        taskrow = add_logic_row(con, ("task", "Calibrate readout"))
        node = con._build_logic_node(taskrow.node, dict(taskrow.node.values))  # build only, no run
        spec = con._spec_for_logic(taskrow.node)
        cfg = con._task_mid_run_config(spec, node, title="Task: t")
        assert cfg.source == f"value = {TASK_FRAME_KEY}"                       # writer side
        assert TASK_FRAME_KEY in con._expression_namespace()                   # reader side
        # the declared picker entry for a task = its spec's mid_run_key + the ONE display tag
        keys = con._declared_signal_keys(taskrow)
        assert keys == [f"{spec.mid_run_key}{MID_RUN_TAG}"]
    finally:
        con.shutdown()
        exp.close()


def test_camera_row_display_name_comes_from_the_spec():
    """#23: the Add-Panel dropdown label AND the added row's title both read
    ``readout.camera_spec().name`` -- the authoritative spec, never a re-typed literal."""
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        spec_name = exp.readout.camera_spec().name
        kc = con.kind_combo
        i = next(j for j in range(kc.count()) if kc.itemData(j) == ("camera", "live"))
        assert kc.itemText(i) == f"Measurement: {spec_name}"
        row = add_logic_row(con, ("camera", "live"))
        # rows get "<base> #N" uniqueness (Round G); the BASE display name is the spec's.
        assert row.node.title.startswith(spec_name)
    finally:
        con.shutdown()
        exp.close()


def test_declared_keys_ride_the_one_bare_key_ladder():
    """#19: for every hub-publishing kind, the picker's declared names == the collision
    check's bare keys + the shared prefix rule -- one ladder, so what is checked is exactly
    what is offered and later published."""
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        rows = [add_logic_row(con, ("camera", "live")),
                add_logic_row(con, ("processor", "Readout fidelity"))]
        for row in rows:
            pfx = con._declared_node_prefix(row)
            expected = [f"{pfx}{k}" for k in con._node_bare_keys(row.node)]
            assert con._declared_signal_keys(row) == expected
            assert expected                                   # the ladder yields real names
    finally:
        con.shutdown()
        exp.close()
