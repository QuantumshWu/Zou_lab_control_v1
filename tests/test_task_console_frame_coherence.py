"""MECHANICAL guard for cross-panel SHOT COHERENCE on the task-console board (the '三个 emCCD frame 不
同步' regression).

The board renders every panel from ONE coherent display shot (``_display_shot``) and, when that shot
advances, recomposes EVERY co-displayed panel in the SAME tick (the two-phase compose-all-then-present-
all render keyed on the display shot).  So three 2d panels bound to frame_0/1/2 of one pulse can NEVER
freeze on different shots.

The earlier per-signal render gate broke this: a panel redrew only when ITS OWN signal republished, so
when a SLOWER co-displayed producer (a reactive occupancy) advanced the clock at a tick where the frame
panels' own signal had not moved, those panels were skipped -> they lagged a shot behind (visible on
Pause).  This test recreates exactly that timing and asserts the three frames stay one coherent shot.
"""
import numpy as np
import pytest

import Zou_lab_control.frontend as zf
from conftest import tick
from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
from Zou_lab_control.frontend.task_console import TaskConsole
from Zou_lab_control.neutral_atom.core.signals import SignalHub


class _Spec:
    def __init__(self, data_shape):
        self.has_structure = True
        self.points_shape = ()
        self.data_shape = data_shape


class _FakeProducer:
    """Declares the live signals + their structure (frames are 2-D images, occupancy a scalar); the test
    publishes to the hub manually so it can control each signal's source-shot provenance."""
    last_error = None

    def published_signals(self):
        return ["frame_0", "frame_1", "frame_2", "occupancy"]

    def signal_spec(self, name):
        return _Spec(() if name == "occupancy" else (8, 8))


def _img(shot):
    return np.full((1, 8, 8), shot * 100.0, dtype=float)     # (repeat, H, W) camera contract


def _frame_shot(card):
    return int(round(float(np.nanmean(card.plotter.data_y)))) // 100


def test_three_emccd_frames_never_split_when_clock_advances_via_a_lagging_coproducer():
    ensure_qt_app()
    hub = SignalHub()
    state = zf.TaskConsoleState(name="coh", panels=[
        zf.PanelConfig(kind="2d", title="f0", size="2x2", source="value = frame_0"),
        zf.PanelConfig(kind="2d", title="f1", size="2x2", source="value = frame_1"),
        zf.PanelConfig(kind="2d", title="f2", size="2x2", source="value = frame_2"),
        zf.PanelConfig(kind="monitor", title="occ", size="2x2", source="value = occupancy"),
    ])
    console = TaskConsole(hub=hub, state=state, running_nodes=[_FakeProducer()])
    console._timer.stop()
    console._poll_logic_nodes = lambda: None      # isolate the RENDER path
    console._refresh_signal_info = lambda: None
    console._refresh_task_panel = lambda: None
    try:
        hub.publish({"frame_0": _img(0), "frame_1": _img(0), "frame_2": _img(0)}, provenance=0)
        hub.publish({"occupancy": 0.0}, provenance=0)
        tick(console)
        assert all(c.plotter is not None for c in console.cards)

        clock_advances = 0
        for S in range(1, 7):
            # tick A: the three frames for shot S arrive together; occupancy still at S-1 -> the coherent
            # display shot is HELD BACK to S-1, so all three frame panels must show S-1 (and agree).
            hub.publish({"frame_0": _img(S), "frame_1": _img(S), "frame_2": _img(S)}, provenance=S)
            tick(console)
            disp_a = console._display_shot()
            shots_a = [_frame_shot(console.cards[k]) for k in range(3)]
            assert len(set(shots_a)) == 1 and shots_a[0] == disp_a, (S, "A", shots_a, disp_a)

            # tick B: occupancy CATCHES UP to shot S -> the clock advances S-1 -> S at a tick where the
            # frame panels' OWN signal did NOT republish.  The per-signal gate skipped them here (desync);
            # the display-shot gate recomposes all three together -> still one coherent shot.
            hub.publish({"occupancy": float(S)}, provenance=S)
            tick(console)
            disp_b = console._display_shot()
            shots_b = [_frame_shot(console.cards[k]) for k in range(3)]
            if disp_b != disp_a:
                clock_advances += 1
            assert len(set(shots_b)) == 1 and shots_b[0] == disp_b, (S, "B", shots_b, disp_b)

        # the scenario must actually exercise the clock-advance-without-own-republish path every round,
        # else the test proves nothing.
        assert clock_advances == 6, clock_advances
    finally:
        console.shutdown()


def test_render_lifecycle_is_split_into_compose_and_present():
    """The board's coherence rests on a two-phase render: BaseLivePlot must expose compose() (write the
    offscreen buffer, no present) and present() (flush the buffer), with draw() = compose()+present() for
    single-figure callers.  A guard so the phases are never collapsed back into one draw()."""
    from Zou_lab_control.frontend.live import BaseLivePlot
    for name in ("compose", "present", "draw"):
        assert callable(getattr(BaseLivePlot, name, None)), name
