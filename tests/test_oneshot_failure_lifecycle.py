"""One-shot task and processor-run termination contracts."""

from __future__ import annotations


def test_oneshot_task_marks_finished_even_when_run_raises():
    """A one-shot Task whose ``run`` RAISES must still end terminated: ``finished``
    means 'this one-shot has RUN ONCE' (success OR failure), so it is set in a
    ``finally`` alongside the stop event.  Otherwise a failed task stays
    ``finished=False`` forever and the console keeps the dashboard locked (finding 7).
    The exception still propagates to a headless ``step()`` caller (finally must not
    swallow it); failure is expressed by ``result`` staying empty."""
    import pytest

    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import Task

    class _BoomTask(Task):
        def run(self, out):
            raise RuntimeError("boom")

    hub = SignalHub()
    node = _BoomTask(hub)
    with pytest.raises(RuntimeError, match="boom"):
        node.shot()
    assert node.finished is True            # terminated despite the failure -> console can unlock
    assert node._stop.is_set()              # one-shot: the loop will not retry
    assert node.result == {}                # failure expressed by the empty result


def test_oneshot_processor_marks_finished_even_when_run_raises():
    """Same lifecycle invariant for the discrete ProcessorRun sibling: a ``spec.run``
    that RAISES sets ``finished=True`` in a ``finally`` (it ran once, no retry), the
    exception propagates, and the per-shot publish never runs (so nothing partial is
    pushed onto the hub)."""
    import pytest

    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import ProcessorRun

    class _BoomSpec:
        name = "boom"
        result_keys = ("value",)

        def run(self, ctx):
            raise RuntimeError("kaboom")

    hub = SignalHub()
    node = ProcessorRun(hub, _BoomSpec(), readout=None)
    with pytest.raises(RuntimeError, match="kaboom"):
        node.shot()
    assert node.finished is True            # terminated despite the failure -> console can unlock
    assert node._stop.is_set()
    assert hub.names() == []                # the failed shot published nothing partial
