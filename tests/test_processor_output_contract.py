"""Contract (#H3r-F3): a reactive processor's output keys are declared ONCE and enforced.

- ``provides`` (a class fact on the :class:`Processor` node) is the SINGLE source of the published
  key names; ``output_keys()`` returns it, ``published_signals()`` = prefix+it, and a spec's
  ``result_keys`` DERIVES from it (occupancy) -- so spec and node can never drift.
- Publish-time conformance: a processor that emits a key it did NOT declare in ``provides`` raises
  loud at the boundary, instead of leaking a silent, unlegended hub signal.
"""

from __future__ import annotations

import pytest

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.logic import OccupancyProcessor, Processor


def test_occupancy_spec_result_keys_derive_from_node_provides():
    exp = na.connect("virtual")
    try:
        specs = {s.name: s for s in exp.readout.processor_specs()}
        # the spec's de-dup/output keys are exactly the node's declared provides (single source)
        assert tuple(specs["Judge occupancy"].result_keys) == tuple(OccupancyProcessor.provides)
    finally:
        exp.close()


def test_output_keys_published_signals_derive_from_provides():
    class _P(Processor):
        provides = ("a", "b")

        def transform(self, inputs):  # pragma: no cover - not exercised here
            return {}

    node = _P(SignalHub(), consumes=("x",), prefix="p_")   # a Processor now requires >=1 consumed signal
    assert node.output_keys() == ("a", "b")
    assert node.published_signals() == frozenset({"p_a", "p_b"})


def test_processor_publishing_undeclared_signal_raises():
    hub = SignalHub()

    class _Rogue(Processor):
        provides = ("good",)

        def transform(self, inputs):
            return {"good": 1.0, "rogue": 2.0}    # 'rogue' is NOT declared in provides

    node = _Rogue(hub, consumes=("x",))
    hub.publish({"x": 1.0})    # publish AFTER the node subscribes at construction (reactive replay)
    with pytest.raises(ValueError, match="undeclared"):
        node.shot()


def test_processor_publishing_only_declared_signals_is_fine():
    hub = SignalHub()

    class _Good(Processor):
        provides = ("good",)

        def transform(self, inputs):
            return {"good": 42.0}

    node = _Good(hub, consumes=("x",))
    hub.publish({"x": 1.0})    # publish AFTER the node subscribes at construction (reactive replay)
    assert node.shot() == {"good": 42.0}
