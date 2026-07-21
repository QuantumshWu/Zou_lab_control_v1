"""Current headless coherent-board host contracts."""

from __future__ import annotations

import subprocess
import sys
import threading

import numpy as np
import pytest

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    CoordinateFrameId,
    DatasetRevision,
    DatasetRevisionRef,
    SPATIAL_X,
    SPATIAL_Y,
    StreamGenerationId,
)
from zlc_frontend.figure import (
    DatasetId,
    EvaluatedAxis,
    EvaluatedImage,
    EvaluatedInput,
)
from zlc_frontend.image_view import ImageViewportTransform
from zlc_frontend.image_display import (
    ImageDisplayState,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.render import (
    AtomicBoardFront,
    BoardFrame,
    CoherenceStamp,
    ImagePanelPayload,
    PanelFrame,
    PanelPresentationIdentity,
    PixelFormat,
    RasterBuffer,
    RenderSurface,
    SourceIdentity,
)
from zlc_workbench.workspace import (
    BoardController,
    BoardModel,
    PanelSlot,
    PanelSourceBinding,
    WorkspaceModel,
)
from zlc_workbench.live import _accepted_relim_mode


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
    panel_ids: tuple[str, ...] = ("a", "b"),
) -> tuple[BoardFrame, SourceIdentity, CoherenceStamp]:
    source, evaluated = _source(revision=revision)
    stamp = _stamp(
        panel_ids,
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
            tuple(
                PanelFrame(panel_id, "shot", source, stamp, _raster(sequence))
                for panel_id in panel_ids
            ),
        ),
        source,
        stamp,
    )


def _board(
    generation: int = 0,
    panel_ids: tuple[str, ...] = ("a", "b"),
) -> BoardModel:
    return BoardModel(
        "board",
        generation,
        RenderSurface.WORKER_RASTER_LIVE,
        tuple(
            PanelSlot(panel_id, f"controller-{panel_id}", "shot")
            for panel_id in panel_ids
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


def test_image_panel_payload_carries_exact_samples_and_matching_view_revision() -> None:
    frame_id = CoordinateFrameId("camera-frame")
    x_axis = AxisSpec(
        AxisId("camera.x"),
        "x",
        SPATIAL_X,
        2,
        (10, 11),
        unit="pixel",
        coordinate_frame=frame_id,
    )
    y_axis = AxisSpec(
        AxisId("camera.y"),
        "y",
        SPATIAL_Y,
        1,
        (20,),
        unit="pixel",
        coordinate_frame=frame_id,
    )
    image = EvaluatedImage(
        EvaluatedAxis(
            x_axis.axis_id,
            "x",
            x_axis.role,
            "pixel",
            (0, 1),
            (10, 11),
        ),
        EvaluatedAxis(
            y_axis.axis_id,
            "y",
            y_axis.role,
            "pixel",
            (0,),
            (20,),
        ),
        np.asarray([[2, 9]], dtype=np.uint16),
        np.asarray([[True, True]], dtype=bool),
    )
    viewport = ImageViewportTransform((y_axis, x_axis), viewport_revision=1)
    histogram = [0] * 255
    histogram[0] = histogram[-1] = 1
    source, evaluated = _source()
    payload = ImagePanelPayload(
        image,
        evaluated,
        viewport,
        (2.0, 9.0),
        tuple(histogram),
        tuple(0xFF000000 | index for index in range(256)),
        (1.3, 9.7),
    )
    stamp = _stamp(("a",), evaluated_input=evaluated)
    raster = RasterBuffer(2, 1, 2, PixelFormat.INDEXED8, bytes((1, 255)))
    panel = PanelFrame("a", "shot", source, stamp, raster, payload)

    assert panel.display_payload is payload
    assert int(payload.image.values[0, 1]) == 9
    with pytest.raises(ValueError):
        payload.image.values.setflags(write=True)

    stale_stamp = CoherenceStamp(
        stamp.run_id,
        stamp.provenance_epoch_id,
        stamp.join_key_type,
        stamp.join_key_schema_fingerprint,
        stamp.join_key_digest,
        stamp.inputs,
        (_presentation("a", panel_revision=2),),
    )
    with pytest.raises(ValueError, match="payload revision"):
        PanelFrame("a", "shot", source, stale_stamp, raster, payload)
    with pytest.raises(ValueError, match="INDEXED8"):
        PanelFrame("a", "shot", source, stamp, _raster(), payload)
    other_source, other_input = _source(dataset="other-image")
    del other_source
    mismatched_payload = ImagePanelPayload(
        image,
        other_input,
        viewport,
        (2.0, 9.0),
        tuple(histogram),
        tuple(0xFF000000 | index for index in range(256)),
        (1.3, 9.7),
    )
    with pytest.raises(ValueError, match="frozen coherence input"):
        PanelFrame("a", "shot", source, stamp, raster, mismatched_payload)


def test_invalid_fronts_do_not_consume_or_confuse_the_next_valid_relim() -> None:
    tight = ImageDisplayState(relim_mode=RelimMode.TIGHT)
    fixed = ImageDisplayState(
        revision=1,
        relim_mode=RelimMode.FIXED,
        fixed_color_limits=(20.0, 40.0),
    )
    back_to_tight = ImageDisplayState(revision=2, relim_mode=RelimMode.TIGHT)

    mode = _accepted_relim_mode(None, tight.relim_mode, (1.0, 10.0))
    assert mode is RelimMode.TIGHT
    mode = _accepted_relim_mode(mode, fixed.relim_mode, None)
    assert mode is RelimMode.FIXED
    mode = _accepted_relim_mode(mode, back_to_tight.relim_mode, None)
    assert mode is RelimMode.FIXED
    assert mode is not back_to_tight.relim_mode  # next valid front must force relim
    mode = _accepted_relim_mode(mode, back_to_tight.relim_mode, (2.0, 3.0))
    assert mode is RelimMode.TIGHT


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


def test_freeze_front_revokes_inflight_and_pending_publication_without_clearing_front() -> None:
    controller, port, callbacks, presenter, first, stamp = _controller()
    first_work = port.admit(first.sequence, (("shot", stamp),))
    assert port.publish(first_work, first)
    callbacks.pop()()
    assert presenter.frames == [first]

    pending, _source, pending_stamp = _frame(2, revision=2)
    pending_work = port.admit(pending.sequence, (("shot", pending_stamp),))
    assert port.publish(pending_work, pending)
    assert len(callbacks) == 1
    controller.freeze_front()

    assert callbacks.pop()() is False
    assert presenter.frames == [first]
    assert presenter.clear_count == 0
    assert not controller.present_pending()
    assert not port.publish(pending_work, pending)
    with pytest.raises(RuntimeError, match="revoked"):
        port.admit(3, (("shot", pending_stamp),))

    # A worker that passed its live-side currency check before freeze still
    # holds only this revoked port/work capability and cannot publish later.
    stale_controller, stale_port, _callbacks, stale_presenter, stale, stale_stamp = (
        _controller()
    )
    stale_work = stale_port.admit(stale.sequence, (("shot", stale_stamp),))
    stale_controller.freeze_front()
    assert not stale_port.publish(stale_work, stale)
    assert stale_presenter.frames == []


def test_pending_publication_can_be_revoked_from_a_view_worker() -> None:
    controller, port, _callbacks, presenter, frame, stamp = _controller()
    work = port.admit(frame.sequence, (("shot", stamp),))
    errors = []

    def revoke() -> None:
        try:
            controller.revoke_pending_publication()
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=revoke)
    worker.start()
    worker.join()

    assert not errors
    assert not port.publish(work, frame)
    assert not controller.present_pending()
    assert presenter.frames == []


@pytest.mark.parametrize(
    ("before_revoke", "error_type"),
    (
        (lambda: None, TypeError),
        (lambda: (_ for _ in ()).throw(RuntimeError("broken state fence")), RuntimeError),
    ),
)
def test_broken_revocation_state_fence_still_revokes_pending(
    before_revoke,
    error_type,
) -> None:
    controller, port, callbacks, presenter, frame, stamp = _controller()
    work = port.admit(frame.sequence, (("shot", stamp),))
    assert port.publish(work, frame)

    with pytest.raises(error_type):
        controller.revoke_pending_publication(before_revoke)

    assert callbacks.pop()() is False
    assert presenter.frames == []
    assert not port.publish(work, frame)


def test_revocation_serializes_with_a_frame_already_dequeued_for_present() -> None:
    entered = threading.Event()
    attempting_revoke = threading.Event()
    fact_installed = threading.Event()
    release_present = threading.Event()
    revoke_returned = threading.Event()

    class BlockingPresenter(_Presenter):
        def present(self, frame: BoardFrame) -> None:
            entered.set()
            if not release_present.wait(2.0):
                raise TimeoutError("test did not release blocked board present")
            super().present(frame)

    presenter = BlockingPresenter()
    controller, port, _callbacks, _value, frame, stamp = _controller(presenter)
    work = port.admit(frame.sequence, (("shot", stamp),))
    assert port.publish(work, frame)

    def revoke() -> None:
        assert entered.wait(2.0)
        attempting_revoke.set()
        controller.revoke_pending_publication(
            lambda: fact_installed.set() is None
        )
        revoke_returned.set()

    def release() -> None:
        assert attempting_revoke.wait(2.0)
        assert not revoke_returned.wait(0.05)
        assert not fact_installed.is_set()
        release_present.set()

    revoker = threading.Thread(target=revoke)
    releaser = threading.Thread(target=release)
    revoker.start()
    releaser.start()
    assert controller.present_pending()
    revoker.join()
    releaser.join()

    assert revoke_returned.is_set()
    assert fact_installed.is_set()
    assert presenter.frames == [frame]


def test_atomic_board_front_swaps_the_complete_mapping_old_or_new() -> None:
    front = AtomicBoardFront()
    first, _source_value, _stamp_value = _frame(1)
    second, _source_value, _stamp_value = _frame(2, revision=2)

    front.present(first)
    assert front.current() is first
    front.present(second)
    assert front.current() is second
    assert tuple(panel.panel_id for panel in front.current().panels) == ("a", "b")


def test_board_controller_promotes_new_topology_only_with_its_complete_front() -> None:
    controller, old_port, callbacks, presenter, frame, stamp = _controller()
    old_work = old_port.admit(frame.sequence, (("shot", stamp),))
    assert old_port.publish(old_work, frame)
    callbacks.pop()()
    assert presenter.frames == [frame]

    target_panels = ("a", "b", "c")
    controller.reconfigure(_board(1, target_panels))
    assert presenter.frames == [frame]
    assert presenter.clear_count == 0
    with pytest.raises(RuntimeError, match="revoked"):
        old_port.admit(2, (("shot", stamp),))
    assert not old_port.publish(old_work, frame)

    current, source, current_stamp = _frame(
        1,
        generation=1,
        panel_ids=target_panels,
    )
    port = controller.open_publish_port(
        tuple(
            PanelSourceBinding(source, presentation)
            for presentation in current_stamp.presentations
        )
    )
    assert presenter.frames == [frame]
    assert presenter.clear_count == 0
    work = port.admit(1, (("shot", current_stamp),))
    stale, _stale_source, _stale_stamp = _frame(
        1,
        generation=0,
        panel_ids=target_panels,
    )
    assert not port.publish(work, stale)
    assert presenter.frames == [frame]
    assert port.publish(work, current)
    assert presenter.frames == [frame]

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
    assert len(callbacks) == 1
    callbacks.pop()()
    assert presenter.frames == [frame, current]


def test_failed_staged_present_revokes_authority_without_erasing_old_front() -> None:
    class FailingPresenter(_Presenter):
        def present(self, frame: BoardFrame) -> None:
            if frame.layout_generation == 1:
                raise RuntimeError("target widget rejected frame")
            super().present(frame)

    presenter = FailingPresenter()
    controller, old_port, callbacks, _value, old, old_stamp = _controller(presenter)
    old_work = old_port.admit(old.sequence, (("shot", old_stamp),))
    assert old_port.publish(old_work, old)
    callbacks.pop()()

    controller.reconfigure(_board(1))
    target, source, target_stamp = _frame(1, generation=1)
    port = controller.open_publish_port(
        tuple(
            PanelSourceBinding(source, presentation)
            for presentation in target_stamp.presentations
        )
    )
    work = port.admit(target.sequence, (("shot", target_stamp),))
    assert port.publish(work, target)
    with pytest.raises(RuntimeError, match="target widget rejected frame"):
        callbacks.pop()()

    assert presenter.frames == [old]
    assert presenter.clear_count == 0
    assert controller.fault is not None
    assert not port.publish(work, target)
    with pytest.raises(RuntimeError, match="revoked"):
        port.admit(2, (("shot", target_stamp),))


def test_staged_model_rollback_keeps_front_and_can_reopen_the_active_layout() -> None:
    controller, old_port, callbacks, presenter, old, old_stamp = _controller()
    old_work = old_port.admit(old.sequence, (("shot", old_stamp),))
    assert old_port.publish(old_work, old)
    callbacks.pop()()

    target_model = _board(1, ("a", "b", "c"))
    controller.reconfigure(target_model)
    assert controller.model is target_model
    assert controller.discard_staged_model(target_model)
    assert not controller.discard_staged_model(target_model)
    assert controller.model.layout_generation == 0
    assert presenter.frames == [old]
    assert presenter.clear_count == 0
    with pytest.raises(RuntimeError, match="revoked"):
        old_port.admit(2, (("shot", old_stamp),))

    resumed, source, resumed_stamp = _frame(2, generation=0)
    port = controller.open_publish_port(
        tuple(
            PanelSourceBinding(source, presentation)
            for presentation in resumed_stamp.presentations
        )
    )
    work = port.admit(resumed.sequence, (("shot", resumed_stamp),))
    assert port.publish(work, resumed)
    callbacks.pop()()
    assert presenter.frames == [old, resumed]


def test_publish_port_is_revoked_on_close() -> None:
    controller, port, _callbacks, _presenter, frame, stamp = _controller()
    work = port.admit(frame.sequence, (("shot", stamp),))
    controller.reconfigure(_board(1))
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


def test_target_frontend_and_workbench_import_without_qt_side_effects() -> None:
    code = (
        "import sys; import zlc_frontend, zlc_workbench; "
        "assert not any(name == 'PyQt5' or name.startswith('PyQt5.') "
        "for name in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
