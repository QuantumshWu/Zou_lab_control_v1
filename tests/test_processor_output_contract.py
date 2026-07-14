"""Contract (#H3r-F3): a reactive processor's output keys are declared ONCE and enforced.

- ``provides`` (a class fact on the :class:`Processor` node) is the single source
  of published key names; ``output_keys()`` and ``published_signals()`` derive from it.
- Publish-time conformance: a processor that emits a key it did NOT declare in ``provides`` raises
  loud at the boundary, instead of leaking a silent, unlegended hub signal.
"""

from __future__ import annotations

import pytest

from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.logic import Processor




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
