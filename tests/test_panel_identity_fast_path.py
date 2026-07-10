# -*- coding: utf-8 -*-
"""Contract: a panel bound to a camera signal with the default ``value = signal`` source
takes the IDENTITY zero-copy path every tick -- no float64 materialisation of the frame.

W-round regression this pins: ``SignalExpr.co_names()`` folds the bound slot inputs in
(for version-gating), and the raw-signal detector in ``_eval_signal_per_slice`` used it,
so the bound signal itself was mistaken for a directly-named raw block -> the identity
short-circuit NEVER fired and every bound panel stacked a float64 copy of the 2.3 MP
frame per tick (~52 ms/panel).  The detector must use ``direct_names()`` (the source
TEXT's own identifiers only).
"""
import numpy as np
import pytest

from Zou_lab_control.frontend.live import reduce_repeat
from Zou_lab_control.neutral_atom.operations.signal_expr import DEFAULT_SOURCE, SignalExpr

from conftest import add_logic_row, make_console


def test_direct_names_excludes_bound_inputs_co_names_includes_them():
    expr = SignalExpr(["frame_0"], DEFAULT_SOURCE)
    assert "frame_0" not in expr.direct_names()   # the source text never names frame_0
    assert "frame_0" in expr.co_names()           # version-gating still watches the binding
    named = SignalExpr(["frame_0"], "value = frame_0 * 2")
    assert "frame_0" in named.direct_names()      # the text DOES name it -> raw slicing path


def test_reduce_repeat_single_integer_repeat_is_a_zero_copy_view():
    blk = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(1, 2, 3, 4)
    for mode in ("average", "add", "replace", "roll", "pool"):
        out = reduce_repeat(blk, mode, hist=(mode == "pool"))
        assert np.shares_memory(out, blk), mode
        assert out.dtype == np.uint8, mode
        ref = blk[0].reshape(-1) if mode == "pool" else blk[0]
        assert np.array_equal(np.asarray(out).reshape(-1), ref.reshape(-1)), mode
    # ``create`` may only turn repeats and a one-dimensional data_shape into
    # trace columns; it must not flatten a multidimensional data tensor.
    with pytest.raises(ValueError, match="multidimensional data_shape"):
        reduce_repeat(blk, "create")

    trace = np.arange(2 * 3, dtype=np.uint8).reshape(1, 2, 3)
    created = reduce_repeat(trace, "create")
    assert created.shape == (2, 3)


def test_reduce_repeat_float_single_repeat_keeps_nan_semantics():
    blk = np.full((1, 2, 2), np.nan)
    blk[0, 0, 0] = 3.0
    # 'add' nansums a NaN gap to 0.0 -- the float path must stay byte-identical
    out = reduce_repeat(blk, "add")
    assert out[0, 0] == 3.0 and out[0, 1] == 0.0


def test_reduce_repeat_multi_repeat_integer_mean_unchanged():
    blk = np.stack([np.full((1, 2, 2), 10, dtype=np.uint8),
                    np.full((1, 2, 2), 20, dtype=np.uint8)])
    out = reduce_repeat(blk, "average")
    assert np.allclose(out, 15.0)


def test_bound_camera_panel_tick_value_is_native_and_zero_copy():
    """The USER path: add a 2D panel bound to the monitor camera's frame signal and tick --
    the composed value must be the hub block itself (uint8, shared memory), never a float64
    re-stack."""
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        row = add_logic_row(con, ("camera", "live"))
        con._logic_editors[id(row)].form.seed_values({"camera": "monitor_camera"})
        con._start_logic_node(row)
        node = con._logic_nodes[id(row)]
        node.step()
        sig = sorted(node.published_signals())[0]
        kc = con.kind_combo
        kc.setCurrentIndex(next(i for i in range(kc.count())
                                if kc.itemText(i) == "Plot: 2D image"))
        con._add_panel()
        card = con.cards[-1]
        card.config.inputs = [sig]
        card.source_edit.setText(DEFAULT_SOURCE)
        card._apply_source()
        con.refresh_once()
        node.step()
        ns = con._expression_namespace()
        value = card._signal_then_repeat(ns)
        assert np.asarray(value).dtype == np.uint8
        assert np.shares_memory(np.asarray(value), np.asarray(ns[sig]))
    finally:
        con.shutdown()
        exp.close()
