"""Current headless coherent-board host and bounded legacy-island contracts."""

from __future__ import annotations

import subprocess
import sys
import threading

import pytest

from zlc_data import (
    BlockId,
    DatasetRevision,
    DatasetRevisionRef,
    StreamGenerationId,
)
from zlc_frontend.figure import DatasetId, EvaluatedInput
from zlc_frontend.render import (
    AtomicBoardFront,
    BoardFrame,
    CoherenceStamp,
    PanelFrame,
    PanelPresentationIdentity,
    PixelFormat,
    RasterBuffer,
    RenderSurface,
    SourceIdentity,
)
from zlc_workbench.legacy import (
    CatalogEntry,
    CatalogRoute,
    CatalogRouter,
    LegacyHandoffTimeout,
    SerializedLegacyAggBridge,
)
from zlc_workbench.workspace import (
    BoardController,
    BoardModel,
    PanelSlot,
    PanelSourceBinding,
    WorkspaceModel,
)


SCHEMA_A = "a" * 64
SCHEMA_B = "b" * 64
JOIN_A = "c" * 64
JOIN_B = "d" * 64


def _source(
    *,
    dataset: str = "data-1",
    block: str = "block-1",
    generation: str = "generation-3",
    schema: str = SCHEMA_A,
    revision: int = 1,
) -> tuple[SourceIdentity, EvaluatedInput]:
    dataset_id = DatasetId(dataset)
    ref = DatasetRevisionRef(
        BlockId(block),
        StreamGenerationId(generation),
        schema,
        DatasetRevision(revision),
    )
    return (
        SourceIdentity(dataset_id, ref.block_id, ref.stream_generation, schema),
        EvaluatedInput(dataset_id, ref),
    )


def _presentation(
    panel_id: str,
    *,
    document_revision: int = 1,
    selection_revision: int = 0,
    panel_revision: int = 1,
) -> PanelPresentationIdentity:
    return PanelPresentationIdentity(
        panel_id,
        "figure-document",
        document_revision,
        selection_revision,
        panel_revision,
    )


def _stamp(
    panel_ids: tuple[str, ...],
    *,
    evaluated_input: EvaluatedInput,
    run_id: str = "run-1",
    epoch: str = "epoch-1",
    join_digest: str = JOIN_A,
    document_revision: int = 1,
) -> CoherenceStamp:
    return CoherenceStamp(
        run_id,
        epoch,
        "shot-ordinal",
        SCHEMA_A,
        join_digest,
        (evaluated_input,),
        tuple(
            _presentation(panel_id, document_revision=document_revision)
            for panel_id in panel_ids
        ),
    )


def _raster(value: int = 0) -> RasterBuffer:
    return RasterBuffer(2, 1, 8, PixelFormat.RGBA8888, bytes([value]) * 8)


def _frame(
    sequence: int,
    *,
    generation: int = 0,
    revision: int = 1,
) -> tuple[BoardFrame, SourceIdentity, CoherenceStamp]:
    source, evaluated = _source(revision=revision)
    stamp = _stamp(
        ("a", "b"),
        evaluated_input=evaluated,
        epoch=f"epoch-{revision}",
        join_digest=(f"{revision:x}" * 64)[:64],
        document_revision=1,
    )
    return (
        BoardFrame(
            "board",
            generation,
            sequence,
            (
                PanelFrame("a", "shot", source, stamp, _raster(sequence)),
                PanelFrame("b", "shot", source, stamp, _raster(sequence)),
            ),
        ),
        source,
        stamp,
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
        self.frames: list[BoardFrame] = []
        self.clear_count = 0

    def present(self, frame: BoardFrame) -> None:
        self.frames.append(frame)

    def clear(self) -> None:
        self.clear_count += 1


def _controller(presenter: _Presenter | None = None):
    callbacks = []
    presenter = presenter or _Presenter()
    controller = None

    def request_owner_wake() -> None:
        assert controller is not None
        callbacks.append(controller.present_pending)

    controller = BoardController(_board(), presenter, request_owner_wake)
    frame, source, stamp = _frame(1)
    bindings = tuple(
        PanelSourceBinding(source, presentation)
        for presentation in stamp.presentations
    )
    port = controller.open_publish_port(bindings)
    return controller, port, callbacks, presenter, frame, stamp


def test_board_frame_rejects_mixed_coherence_and_mutable_pixels() -> None:
    with pytest.raises(TypeError, match="owned immutable bytes"):
        RasterBuffer(1, 1, 4, PixelFormat.RGBA8888, bytearray(4))

    first, source, stamp = _frame(1)
    _second, _source_value, later = _frame(2, revision=2)
    with pytest.raises(ValueError, match="one exact CoherenceStamp"):
        BoardFrame(
            "board",
            0,
            1,
            (
                first.panels[0],
                PanelFrame("b", "shot", source, later, _raster()),
            ),
        )
    assert stamp != later


def test_source_identity_and_coherence_are_separate_typed_values() -> None:
    source, evaluated = _source()
    stamp = _stamp(("a",), evaluated_input=evaluated)

    assert source.dataset_id == evaluated.dataset_id
    assert stamp.inputs == (evaluated,)
    assert not hasattr(source, "run_id")
    assert not hasattr(source, "revision")
    with pytest.raises(ValueError, match="SHA-256"):
        SourceIdentity(
            source.dataset_id,
            source.block_id,
            source.stream_generation,
            "not-a-schema-digest",
        )


def test_board_frame_allows_explicitly_independent_coherence_groups() -> None:
    camera_source, camera_input = _source()
    temperature_source, temperature_input = _source(
        dataset="temperature",
        block="temperature-block",
        generation="temperature-generation",
        schema=SCHEMA_B,
        revision=9,
    )
    camera_stamp = _stamp(("a",), evaluated_input=camera_input)
    temperature_stamp = CoherenceStamp(
        "run-2",
        "temperature-epoch",
        "sample-ordinal",
        SCHEMA_B,
        JOIN_B,
        (temperature_input,),
        (_presentation("b", document_revision=9),),
    )

    frame = BoardFrame(
        "board",
        0,
        1,
        (
            PanelFrame("a", "camera", camera_source, camera_stamp, _raster()),
            PanelFrame(
                "b",
                "temperature",
                temperature_source,
                temperature_stamp,
                _raster(),
            ),
        ),
    )
    assert frame.panels[0].coherence_stamp != frame.panels[1].coherence_stamp


def test_board_controller_admits_then_publishes_one_coherent_board() -> None:
    controller, port, callbacks, presenter, frame, stamp = _controller()
    work = port.admit(frame.sequence, (("shot", stamp),))

    assert port.publish(work, frame)
    assert len(callbacks) == 1
    callbacks.pop()()
    assert presenter.frames == [frame]
    assert not controller.present_pending()


def test_newer_admission_revokes_an_older_worker_result() -> None:
    controller, port, callbacks, presenter, first, first_stamp = _controller()
    second, _source_value, second_stamp = _frame(2, revision=2)
    first_work = port.admit(1, (("shot", first_stamp),))
    second_work = port.admit(2, (("shot", second_stamp),))

    assert not port.publish(first_work, first)
    assert port.publish(second_work, second)
    assert len(callbacks) == 1
    callbacks.pop()()
    assert [frame.sequence for frame in presenter.frames] == [2]
    assert not controller.present_pending()


def test_atomic_board_front_swaps_the_complete_mapping_old_or_new() -> None:
    front = AtomicBoardFront()
    first, _source_value, _stamp_value = _frame(1)
    second, _source_value, _stamp_value = _frame(2, revision=2)

    front.present(first)
    assert front.current() is first
    front.present(second)
    assert front.current() is second
    assert tuple(panel.panel_id for panel in front.current().panels) == ("a", "b")


def test_board_controller_rejects_stale_layout_and_wrong_thread_present() -> None:
    controller, old_port, callbacks, _presenter, frame, stamp = _controller()
    controller.reconfigure(_board(1))
    with pytest.raises(RuntimeError, match="revoked"):
        old_port.admit(1, (("shot", stamp),))

    current, source, current_stamp = _frame(1, generation=1)
    port = controller.open_publish_port(
        tuple(
            PanelSourceBinding(source, presentation)
            for presentation in current_stamp.presentations
        )
    )
    work = port.admit(1, (("shot", current_stamp),))
    assert port.publish(work, current)

    errors = []

    def wrong_thread() -> None:
        try:
            controller.present_pending()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=wrong_thread)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert "owner-thread affine" in str(errors[0])
    assert callbacks


def test_publish_port_is_revoked_on_close() -> None:
    controller, port, _callbacks, _presenter, frame, stamp = _controller()
    work = port.admit(frame.sequence, (("shot", stamp),))
    controller.close()

    assert not port.publish(work, frame)
    with pytest.raises(RuntimeError, match="revoked"):
        port.admit(2, (("shot", stamp),))


def test_workspace_updates_are_revisioned_values() -> None:
    workspace = WorkspaceModel("workspace", 0, (_board(),))
    changed = workspace.replace_board(_board().replace_panels(_board().panels[:1]))

    assert workspace.revision == 0 and changed.revision == 1
    assert workspace.boards[0].layout_generation == 0
    assert changed.boards[0].layout_generation == 1


def test_catalog_router_requires_one_explicit_route_per_use_case() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        CatalogRouter(
            ("camera",),
            (
                CatalogEntry("camera", CatalogRoute.LEGACY),
                CatalogEntry("camera", CatalogRoute.TARGET),
            ),
        )
    with pytest.raises(ValueError, match="missing=pulse"):
        CatalogRouter(
            ("camera", "pulse"),
            (CatalogEntry("camera", CatalogRoute.TARGET),),
        )
    router = CatalogRouter(
        ("camera",),
        (CatalogEntry("camera", CatalogRoute.TARGET),),
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


def test_serialized_legacy_agg_handoff_timeout_poison_is_fail_closed() -> None:
    bridge = SerializedLegacyAggBridge(_LegacyLoop(False))
    with pytest.raises(LegacyHandoffTimeout):
        bridge.settle(0.1)
    assert bridge.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        bridge.submit(lambda: None)


def test_serialized_legacy_agg_close_requires_confirmed_worker_join() -> None:
    loop = _LegacyLoop(stop_result=False)
    bridge = SerializedLegacyAggBridge(loop)
    with pytest.raises(LegacyHandoffTimeout, match="did not terminate"):
        bridge.close(0.1)
    assert loop.stopped and bridge.poisoned and not bridge.closed
    loop.stop_result = True
    bridge.close(0.1)
    assert bridge.closed


def test_serialized_legacy_agg_exceptions_poison_and_block_future_access() -> None:
    bridge = SerializedLegacyAggBridge(_RaisingLegacyLoop(raise_barrier=True))
    with pytest.raises(LegacyHandoffTimeout, match="handoff failed"):
        bridge.settle(0.1)
    assert bridge.poisoned

    bridge = SerializedLegacyAggBridge(_RaisingLegacyLoop(raise_stop=True))
    with pytest.raises(LegacyHandoffTimeout, match="stop failed"):
        bridge.close(0.1)
    assert bridge.poisoned and not bridge.closed


def test_target_frontend_and_workbench_import_without_qt_side_effects() -> None:
    code = (
        "import sys; import zlc_frontend, zlc_workbench; "
        "assert not any(name == 'PyQt5' or name.startswith('PyQt5.') "
        "for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
