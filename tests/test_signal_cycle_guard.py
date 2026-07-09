# -*- coding: utf-8 -*-
"""Contract: reactive signal loops are rejected LOUD, at the right scope.

* DIRECT self-loop (a processor consuming its own output) -- rejected at the BASE
  (``Processor.__init__``), covering the notebook path and the console alike.  Unguarded it
  is either permanent silent starvation (fresh hub) or a full-CPU self-sustained republish
  spiral whose inherited source-shot id freezes the display clock (primed hub).
* INDIRECT ring (A consumes B's output while B consumes A's) -- rejected by the console's
  start-time graph walk (``_reactive_ring``), the one place that sees every row.
* The processor's source PICKER never offers the node's own declared outputs, so the
  misclick cannot happen in the GUI at all.
"""
import pytest

from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.processors.roi import RoiProcessor

from conftest import add_logic_row, make_console


def test_processor_rejects_consuming_its_own_output():
    hub = SignalHub()
    with pytest.raises(ValueError, match="its own output"):
        RoiProcessor(hub, source_expr={"inputs": ["roi_frame"], "source": "value = signal"})


def test_console_rejects_an_indirect_reactive_ring():
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        row_a = add_logic_row(con, ("processor", "ROI crop"))
        row_b = add_logic_row(con, ("processor", "MOT intensity"))
        a_out = [k for k in con._declared_signal_keys(row_a) if k.endswith("roi_frame")][0]
        b_out = con._declared_signal_keys(row_b)[0]
        assert a_out != b_out
        con._logic_editors[id(row_a)].form.seed_values(
            {"source": {"inputs": [b_out], "source": "value = signal"}})
        con._start_logic_node(row_a)                 # fine: B is not (yet) consuming A
        assert con._logic_nodes.get(id(row_a)) is not None
        con._logic_editors[id(row_b)].form.seed_values(
            {"source": {"inputs": [a_out], "source": "value = signal"}})
        con._start_logic_node(row_b)                 # closes the ring -> rejected before commit
        assert con._logic_nodes.get(id(row_b)) is None
    finally:
        con.shutdown()
        exp.close()


def test_processor_source_picker_excludes_own_outputs():
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual")
    con = make_console(exp)
    try:
        row = add_logic_row(con, ("processor", "ROI crop"))
        editor = con._logic_editors[id(row)]
        own = set(con._declared_signal_keys(row))
        offered = set(editor.form._signals_provider())
        assert own and not (own & offered)
    finally:
        con.shutdown()
        exp.close()
