"""Contracts for the S0.5 headless workbench host and migration fences."""

from __future__ import annotations

import threading
from types import SimpleNamespace
import subprocess
import sys

import pytest

from zlc_frontend import (
    AtomicBoardFront,
    BoardFrame,
    FrameIdentity,
    PanelFrame,
    PixelFormat,
    RasterBuffer,
    RenderSurface,
)
from zlc_workbench import (
    BoardController,
    BoardModel,
    CatalogEntry,
    CatalogRoute,
    CatalogRouter,
    CoherenceSourceBinding,
    LegacyHandoffTimeout,
    PanelSlot,
    RunHandleStatusBinding,
    SerializedLegacyAggBridge,
    WorkspaceModel,
)


SCHEMA_A = "a" * 64
SCHEMA_B = "b" * 64


def _identity(revision: int = 1) -> FrameIdentity:
    return FrameIdentity("run-1", "data-1", 3, SCHEMA_A, revision, f"shot-{revision}")


def _raster(value: int = 0) -> RasterBuffer:
    return RasterBuffer(2, 1, 8, PixelFormat.RGBA8888, bytes([value]) * 8)


def _frame(sequence: int, *, generation: int = 0, revision: int = 1) -> BoardFrame:
    identity = _identity(revision)
    return BoardFrame(
        "board",
        generation,
        sequence,
        (
            PanelFrame("a", "shot", identity, _raster(sequence)),
            PanelFrame("b", "shot", identity, _raster(sequence)),
        ),
    )


def _board(generation: int = 0) -> BoardModel:
    return BoardModel(
        "board",
        generation,
        RenderSurface.WORKER_RASTER_LIVE,
        (
            PanelSlot("a", "image", "shot"),
            PanelSlot("b", "curve", "shot"),
        ),
    )


class _Presenter:
    def __init__(self) -> None:
        self.frames = []

    def present(self, frame: BoardFrame) -> None:
        self.frames.append(frame)


def _controller(presenter=None):
    callbacks = []
    controller = BoardController(_board(), presenter or _Presenter(), callbacks.append)
    port = controller.open_publish_port(
        (CoherenceSourceBinding("shot", 3, SCHEMA_A),)
    )
    return controller, port, callbacks


def test_board_frame_rejects_mixed_identity_and_mutable_pixels():
    identity = _identity()
    with pytest.raises(TypeError, match="owned immutable bytes"):
        RasterBuffer(1, 1, 4, PixelFormat.RGBA8888, bytearray(4))
    with pytest.raises(ValueError, match="exact FrameIdentity"):
        BoardFrame(
            "board",
            0,
            1,
            (
                PanelFrame("a", "shot", identity, _raster()),
                PanelFrame("b", "shot", _identity(2), _raster()),
            ),
        )


def test_frame_and_source_machine_identities_reject_normalized_or_fake_values():
    with pytest.raises(ValueError, match="canonical"):
        FrameIdentity(" run-1 ", "data-1", 3, SCHEMA_A, 1, "shot-1")
    with pytest.raises(ValueError, match="SHA-256"):
        FrameIdentity("run-1", "data-1", 3, "schema-a", 1, "shot-1")
    with pytest.raises(ValueError, match="SHA-256"):
        CoherenceSourceBinding("shot", 3, "schema-a")


def test_board_frame_allows_explicitly_independent_coherence_groups():
    first = _identity(1)
    second = FrameIdentity("run-2", "data-2", 8, SCHEMA_B, 9, "temperature-9")
    frame = BoardFrame(
        "board",
        0,
        1,
        (
            PanelFrame("a", "camera", first, _raster()),
            PanelFrame("b", "temperature", second, _raster()),
        ),
    )
    assert frame.coherence_stamps == (("camera", first), ("temperature", second))


def test_board_controller_replaces_pending_but_presents_one_coherent_board():
    presenter = _Presenter()
    controller, port, callbacks = _controller(presenter)
    assert port.publish(_frame(1))
    assert port.publish(_frame(2, revision=2))
    assert len(callbacks) == 1
    callbacks.pop()()
    assert [frame.sequence for frame in presenter.frames] == [2]
    assert len(presenter.frames[0].coherence_stamps) == 1
    assert not controller.present_pending()


def test_atomic_board_front_swaps_the_complete_mapping_old_or_new():
    front = AtomicBoardFront()
    first, second = _frame(1), _frame(2, revision=2)
    front.present(first)
    assert front.current() is first
    front.present(second)
    assert front.current() is second
    assert tuple(panel.panel_id for panel in front.current().panels) == ("a", "b")


def test_board_controller_rejects_stale_layout_and_wrong_thread_present():
    callbacks = []
    controller = BoardController(_board(), _Presenter(), callbacks.append)
    controller.reconfigure(_board(1))
    old_port = controller.open_publish_port(
        (CoherenceSourceBinding("shot", 3, SCHEMA_A),)
    )
    controller.reconfigure(_board(2))
    assert not old_port.publish(_frame(1, generation=1))
    port = controller.open_publish_port(
        (CoherenceSourceBinding("shot", 3, SCHEMA_A),)
    )
    assert port.publish(_frame(1, generation=2))
    errors = []

    def wrong_thread() -> None:
        try:
            controller.present_pending()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert "owner-thread affine" in str(errors[0])


def test_present_inflight_cannot_accept_an_older_sequence():
    started = threading.Event()
    release = threading.Event()

    class _BlockingPresenter(_Presenter):
        def present(self, frame):
            started.set()
            assert release.wait(5.0)
            super().present(frame)

    presenter = _BlockingPresenter()
    controller, port, callbacks = _controller(presenter)
    assert port.publish(_frame(5, revision=5))
    accepted = []

    def publish_during_present():
        assert started.wait(5.0)
        accepted.append(port.publish(_frame(4, revision=4)))
        accepted.append(port.publish(_frame(6, revision=6)))
        release.set()

    publisher = threading.Thread(target=publish_during_present)
    publisher.start()
    callbacks.pop(0)()
    publisher.join()
    assert accepted == [False, True]
    assert presenter.frames[0].sequence == 5
    assert callbacks
    callbacks.pop(0)()
    assert [frame.sequence for frame in presenter.frames] == [5, 6]


def test_publish_port_is_revoked_on_close_and_rejects_wrong_source_generation():
    controller, port, _callbacks = _controller()
    wrong = FrameIdentity("run-1", "data-1", 4, SCHEMA_A, 1, "shot-1")
    frame = BoardFrame(
        "board",
        0,
        1,
        (
            PanelFrame("a", "shot", wrong, _raster()),
            PanelFrame("b", "shot", wrong, _raster()),
        ),
    )
    assert not port.publish(frame)
    controller.close()
    assert not port.publish(_frame(2))


def test_workspace_updates_are_revisioned_values():
    workspace = WorkspaceModel("workspace", 0, (_board(),))
    changed = workspace.replace_board(_board().replace_panels(_board().panels[:1]))
    assert workspace.revision == 0 and changed.revision == 1
    assert workspace.boards[0].layout_generation == 0
    assert changed.boards[0].layout_generation == 1


def test_catalog_router_requires_one_explicit_route_per_use_case():
    with pytest.raises(ValueError, match="exactly one"):
        CatalogRouter(
            ("camera",),
            (
                CatalogEntry("camera", CatalogRoute.LEGACY),
                CatalogEntry("camera", CatalogRoute.TARGET),
            )
        )
    with pytest.raises(ValueError, match="missing=pulse"):
        CatalogRouter(
            ("camera", "pulse"),
            (CatalogEntry("camera", CatalogRoute.TARGET),),
        )
    router = CatalogRouter(
        ("camera",), (CatalogEntry("camera", CatalogRoute.TARGET),)
    )
    assert router.resolve("camera").route is CatalogRoute.TARGET
    with pytest.raises(KeyError, match="no explicit catalog route"):
        router.resolve("pulse")


class _LegacyLoop:
    def __init__(self, barrier_result=True, stop_result=True) -> None:
        self.barrier_result = barrier_result
        self.stop_result = stop_result
        self.stopped = False

    def submit(self, _job):
        return True

    def barrier(self, _timeout):
        return self.barrier_result

    def stop(self, _timeout):
        self.stopped = True
        return self.stop_result


class _RaisingLegacyLoop(_LegacyLoop):
    def __init__(self, *, raise_barrier=False, raise_stop=False):
        super().__init__()
        self.raise_barrier = raise_barrier
        self.raise_stop = raise_stop

    def barrier(self, _timeout):
        if self.raise_barrier:
            raise RuntimeError("barrier exploded")
        return True

    def stop(self, _timeout):
        if self.raise_stop:
            raise RuntimeError("stop exploded")
        return True


def test_serialized_legacy_agg_handoff_timeout_poison_is_fail_closed():
    bridge = SerializedLegacyAggBridge(_LegacyLoop(False))
    with pytest.raises(LegacyHandoffTimeout):
        bridge.settle(0.1)
    assert bridge.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        bridge.submit(lambda: None)


def test_serialized_legacy_agg_close_requires_confirmed_worker_join():
    loop = _LegacyLoop(stop_result=False)
    bridge = SerializedLegacyAggBridge(loop)
    with pytest.raises(LegacyHandoffTimeout, match="did not terminate"):
        bridge.close(0.1)
    assert loop.stopped and bridge.poisoned and not bridge.closed
    loop.stop_result = True
    bridge.close(0.1)
    assert bridge.closed


def test_serialized_legacy_agg_exceptions_poison_and_block_future_access():
    bridge = SerializedLegacyAggBridge(_RaisingLegacyLoop(raise_barrier=True))
    with pytest.raises(LegacyHandoffTimeout, match="handoff failed"):
        bridge.settle(0.1)
    assert bridge.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        bridge.submit(lambda: None)

    bridge = SerializedLegacyAggBridge(_RaisingLegacyLoop(raise_stop=True))
    with pytest.raises(LegacyHandoffTimeout, match="stop failed"):
        bridge.close(0.1)
    assert bridge.poisoned and not bridge.closed


def test_run_status_binding_projects_but_does_not_own_lifecycle():
    class _Handle:
        def __init__(self) -> None:
            self.reason = None

        def snapshot(self):
            return SimpleNamespace(
                run_id="run-1",
                revision=SimpleNamespace(value=7),
                state=SimpleNamespace(value="CANCELLING"),
                phase="draining",
                final_committed=False,
                primary_error=None,
                cleanup_errors=("late frame",),
            )

        def cancel(self, reason):
            self.reason = reason
            return "REQUESTED"

    handle = _Handle()
    binding = RunHandleStatusBinding(handle)
    assert binding.snapshot().state == "CANCELLING"
    assert binding.cancel("stop") == "REQUESTED"
    assert handle.reason == "stop"


def test_target_frontend_and_workbench_import_without_qt_side_effects():
    code = (
        "import sys; import zlc_frontend, zlc_workbench; "
        "assert not any(name == 'PyQt5' or name.startswith('PyQt5.') for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
