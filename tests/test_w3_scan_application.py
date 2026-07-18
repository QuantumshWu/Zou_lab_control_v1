"""W3b scan-owned snapshot and notebook figure product oracles."""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import copy
import gc
import hashlib
from pathlib import Path
import subprocess
import sys
import threading
import time
import weakref

import numpy as np
import pytest

import Zou_lab_control.notebook as zlc
from Zou_lab_control.notebook.facade import _prepare_occupancy_scan_for_workbench
from zlc_data import (
    READOUT_EVENT,
    REPEAT,
    COMPONENT,
    ReductionMethod,
    ReductionSpec,
    SCAN_POINT,
    SITE,
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    DataBlock,
    DataTransformSpec,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    Selection,
    StreamGenerationId,
    VALID,
    Valid,
    ValidityContract,
    ValidityPolicy,
    ValueSchema,
    commit_transform,
    materialize_transformed_snapshot,
)
from zlc_frontend.matplotlib_render import (
    SinglePanelAggRenderer,
    estimate_live_panel_raster_peak_nbytes,
)
from zlc_frontend.curve_display import (
    CurveDisplayState,
    curve_display_with_x_view,
)
from zlc_frontend.render import AtomicBoardFront, CurvePanelPayload
from zlc_frontend.figure import (
    AxisViewRole,
    CURVE_CONTRACT,
    FigureEvaluationPolicy,
    FigureEvaluator,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetCoverage,
    DatasetPreviewSnapshot,
    dataset_storage_nbytes,
)
from zlc_neutral_atom.runtime.pipeline import ExactDatasetPreviewSpec
from zlc_neutral_atom.runtime.run import (
    CancelOutcome,
    RunFailed,
    RunId,
    RunSnapshot,
    RunState,
)
from zlc_neutral_atom.scan import (
    AutonomousScanExecution,
    AutonomousScanSlotProgram,
    ScanOutputContract,
    ScanPointTable,
)
from zlc_neutral_atom.scan.repository import ScanRepository
from zlc_neutral_atom.readout.calibration_reference import (
    calibration_artifact_input_ref,
)
from zlc_neutral_atom.readout.sitemap import load_packaged_sitemap_pulse
from zlc_pulse import (
    FrozenScanTable,
    RepeatRegion,
    ScanParameter,
    load_pulse_document,
)
from zlc_workbench.progressive_scan import (
    ExactDatasetLiveSlot,
    ProgressiveScanPreview,
    ScanDisplayIntent,
    build_occupancy_progressive_spec,
)
from zlc_workbench.scan import (
    FinalScanPresentation,
    PreparedScanPanelRun,
    ScanPanelController,
)


ROOT = Path(__file__).resolve().parents[1]


class _CountingExactDatasetLiveSlot(ExactDatasetLiveSlot):
    def __init__(self, spec: ExactDatasetPreviewSpec) -> None:
        super().__init__(spec)
        self.bind_calls = 0
        self.source_terminal_calls = 0

    def bind(self, reader, *, run_id, causation_domain_id) -> None:
        self.bind_calls += 1
        super().bind(
            reader,
            run_id=run_id,
            causation_domain_id=causation_domain_id,
        )

    def source_terminal(self) -> None:
        self.source_terminal_calls += 1
        super().source_terminal()


def _axis(name, role, size, coordinates, unit=None):
    return AxisSpec(
        AxisId(name),
        name,
        role,
        size,
        tuple(coordinates),
        unit,
        None,
    )


def _component_snapshot_case(*, x_coordinates=(-1.0, 1.0)):
    repeat = _axis("repeat", REPEAT, 2, (0, 1))
    x = _axis("scan.x", SCAN_POINT, 2, x_coordinates, "MHz")
    y = _axis("scan.y", SCAN_POINT, 2, (10.0, 20.0))
    event = _axis("readout.event", READOUT_EVENT, 1, ("image",))
    site = _axis("site", SITE, 3, ("left", "middle", "right"))
    layout = PointLayout.from_mapping((2, 2), ((0, 0), (1, 0), (1, 1)))
    raw_schema = DatasetSchema(
        repeat,
        (x, y),
        layout,
        ValueSchema(
            (event, site),
            ValidityContract.components(site.axis_id),
            np.dtype("<i2"),
            "count",
        ),
    )
    values = np.arange(np.prod(raw_schema.physical_shape), dtype="<i2").reshape(
        raw_schema.physical_shape
    )
    valid = np.asarray(
        (
            ((True, False, True), (True, True, False), (False, True, True)),
            ((True, True, True), (False, False, True), (True, False, True)),
        )
    )
    raw_block = DataBlock(
        BlockId("raw-component-scan"),
        DatasetRevision(7),
        values,
        ComponentValidity((site.axis_id,), valid),
        raw_schema,
    )
    source = OwnedSnapshot(
        raw_block.ref(StreamGenerationId("component-generation")),
        raw_block,
    )
    transform = commit_transform(
        raw_schema,
        DataTransformSpec((Selection.index(event.axis_id, 0),)),
    )
    output_schema = DatasetSchema(
        repeat,
        (x, y),
        layout,
        ValueSchema(
            (site,),
            ValidityContract.components(site.axis_id),
            np.dtype("<i2"),
            "count",
        ),
    )
    output_ref = DatasetRevisionRef(
        BlockId("derived-component-scan"),
        source.ref.stream_generation,
        output_schema.fingerprint,
        source.ref.revision,
    )
    return source, transform, output_schema, output_ref, values, valid


def _sparse_scan_document():
    document = load_pulse_document(ROOT / "pulses" / "mot_field_template.json")
    columns = tuple(item.parameter_id for item in document.scan_parameters)
    return replace(
        document,
        scan_table=FrozenScanTable(
            columns,
            ((0, 0, 0), (1, 0, 1), (0, 1, 1)),
        ),
        repeat=RepeatRegion(
            document.periods[0].period_id,
            document.periods[-1].period_id,
            2,
        ),
    )


def _occupancy_scan_document():
    """Turn the proven sitemap readout event into a two-point SCAN_SLOT."""

    document = load_packaged_sitemap_pulse()
    camera_port = next(
        port for port in document.target.ports if port.label == "emCCD"
    )
    assert len(camera_port.lanes) == 1
    trigger_index = document.target.raw_lanes.index(camera_port.lanes[0])

    segment = -1
    previous = 0
    periods = []
    for period in document.periods:
        states = list(period.states)
        current = int(states[trigger_index])
        if current and not previous:
            segment += 1
        states[trigger_index] = int(bool(current and segment == 1))
        periods.append(replace(period, states=tuple(states)))
        previous = current

    scanned_api = document.api_parameters[0]
    scanned_period = next(
        period
        for period in periods
        if period.period_id == scanned_api.field.period_id
    )
    scan_parameter = ScanParameter(
        "reference_settle",
        scanned_api.field,
        "reference settle",
        scanned_api.unit,
    )
    start = scanned_period.duration
    step = 1 if isinstance(start, int) else 1e-6
    return replace(
        document,
        name="occupancy-scan-slot",
        periods=tuple(periods),
        api_parameters=tuple(
            parameter
            for parameter in document.api_parameters
            if parameter is not scanned_api
        ),
        scan_parameters=(scan_parameter,),
        scan_table=FrozenScanTable(
            (scan_parameter.parameter_id,),
            ((start,), (start + step,)),
        ),
        repeat=RepeatRegion(
            periods[0].period_id,
            periods[-1].period_id,
            2,
        ),
    )


def _fixed_api_values(document):
    return {
        parameter.parameter_id: document.field_value(parameter.field)[0]
        for parameter in document.api_parameters
    }


def test_transform_owner_freezes_once_and_preserves_component_validity(monkeypatch):
    source, transform, schema, output_ref, values, valid = _component_snapshot_case()
    output = materialize_transformed_snapshot(
        source,
        transform,
        output_ref=output_ref,
        output_schema=schema,
        memory_limit_bytes=64 << 20,
    )
    assert output.ref == output_ref
    assert output.block.values.shape == (2, 3, 3)
    np.testing.assert_array_equal(output.block.values, values[:, :, 0, :])
    assert isinstance(output.block.validity, ComponentValidity)
    assert output.block.validity.axis_ids == (AxisId("site"),)
    np.testing.assert_array_equal(output.block.validity.mask, valid)
    assert not output.block.values.flags.writeable

    import zlc_data.transform as transform_module

    executed = False

    def forbidden_execute(*_args, **_kwargs):
        nonlocal executed
        executed = True
        raise AssertionError("transform executed below its admitted peak")

    monkeypatch.setattr(transform_module, "_execute_transform", forbidden_execute)
    with pytest.raises(MemoryError, match="transformed snapshot peak"):
        materialize_transformed_snapshot(
            source,
            transform,
            output_ref=output_ref,
            output_schema=schema,
            memory_limit_bytes=1,
        )
    assert not executed


def test_progressive_renderer_reuses_artists_and_updates_component_validity(monkeypatch):
    source, transform, schema, output_ref, _values, valid = (
        _component_snapshot_case()
    )
    output = materialize_transformed_snapshot(
        source,
        transform,
        output_ref=output_ref,
        output_schema=schema,
        memory_limit_bytes=64 << 20,
    )
    contract = ScanOutputContract(transform, schema)
    progressive = build_occupancy_progressive_spec(
        source.block.schema,
        contract,
        identity="renderer-update",
    )

    def revision(number, mask):
        block = DataBlock(
            output.block.block_id,
            DatasetRevision(number),
            output.block.values,
            ComponentValidity((AxisId("site"),), mask),
            output.block.schema,
        )
        return OwnedSnapshot(block.ref(output.ref.stream_generation), block)

    partial_valid = valid.copy()
    partial_valid[1, :, :] = False
    snapshots = (
        revision(1, partial_valid),
        revision(2, valid),
    )
    evaluator = FigureEvaluator(
        FigureEvaluationPolicy(max_live_nbytes=progressive.evaluation_peak_bytes)
    )

    def evaluate(snapshot):
        return evaluator.evaluate(
            progressive.document,
            ResolvedDatasetMap(
                (ResolvedDataset(progressive.dataset_id, snapshot),)
            ),
        )

    renderer = SinglePanelAggRenderer(
        progressive.document,
        width=360,
        height=240,
    )
    first, first_payload = renderer.render_interactive_curve(
        evaluate(snapshots[0]),
        CurveDisplayState(),
        current_y_limits=None,
        previous_relim_mode=None,
    )
    assert isinstance(first_payload, CurvePanelPayload)
    assert first_payload.viewport.x_axis.role == SCAN_POINT
    assert first_payload.viewport.x_axis.unit == "MHz"
    assert first_payload.value_unit == "count"
    assert len(first_payload.series) == 3
    assert tuple(
        label.split(" | ", 1)[0] for label in first_payload.series_labels
    ) == ("site=left", "site=middle", "site=right")
    assert all(series.data.values.shape == (2,) for series in first_payload.series)
    assert tuple(
        series.data.validity.tolist() for series in first_payload.series
    ) == ([True, True], [False, True], [True, False])
    figure_id = id(renderer._figure)
    axis_id = id(renderer._axis)
    line_ids = tuple(map(id, renderer._artists))
    first_legend = tuple(
        text.get_text() for text in renderer._axis.get_legend().get_texts()
    )

    second, second_payload = renderer.render_interactive_curve(
        evaluate(snapshots[1]),
        CurveDisplayState(),
        current_y_limits=first_payload.viewport.y_limits,
        previous_relim_mode=CurveDisplayState().relim_mode,
    )
    assert len(second_payload.series) == 3
    assert id(renderer._figure) == figure_id
    assert id(renderer._axis) == axis_id
    assert tuple(map(id, renderer._artists)) == line_ids
    second_legend = tuple(
        text.get_text() for text in renderer._axis.get_legend().get_texts()
    )
    assert second_legend != first_legend
    assert second.pixels != first.pixels

    figure_ref = weakref.ref(renderer._figure)
    canvas_ref = weakref.ref(renderer._figure.canvas)
    collection_was_enabled = gc.isenabled()
    gc.disable()
    try:
        renderer.close()
        assert figure_ref() is None
        assert canvas_ref() is None
        renderer.close()
    finally:
        if collection_was_enabled:
            gc.enable()
    with pytest.raises(RuntimeError, match="closed"):
        renderer.render_interactive_curve(
            evaluate(snapshots[1]),
            CurveDisplayState(),
            current_y_limits=None,
            previous_relim_mode=None,
        )

    from matplotlib.figure import Figure

    partial_canvases = []

    def failed_subplots(self, *_args, **_kwargs):
        partial_canvases.append(weakref.ref(self.canvas))
        raise RuntimeError("injected renderer construction failure")

    collection_was_enabled = gc.isenabled()
    gc.disable()
    try:
        with monkeypatch.context() as failure_patch:
            failure_patch.setattr(Figure, "subplots", failed_subplots)
            with pytest.raises(
                RuntimeError,
                match="injected renderer construction failure",
            ):
                SinglePanelAggRenderer(
                    progressive.document,
                    width=360,
                    height=240,
                )
        assert partial_canvases and all(ref() is None for ref in partial_canvases)
    finally:
        if collection_was_enabled:
            gc.enable()


def test_progressive_display_repaints_same_exact_source_without_restarting():
    source, transform, schema, _output_ref, _values, _valid = (
        _component_snapshot_case()
    )
    progressive = build_occupancy_progressive_spec(
        source.block.schema,
        ScanOutputContract(transform, schema),
        identity="same-source-display-repaint",
    )
    assert progressive.interactive_curve
    slot = ExactDatasetLiveSlot(progressive.preview_spec)
    presenter = AtomicBoardFront()
    wake = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    futures = []

    def submit(work):
        future = executor.submit(work)
        futures.append(future)
        return future

    preview = ProgressiveScanPreview(
        slot,
        progressive,
        presenter,
        curve_display=CurveDisplayState(),
        submit_worker=submit,
        request_owner_wake=wake.set,
    )
    total_cells = (
        source.block.schema.repeat_axis.size
        * source.block.schema.point_layout.storage_size
    )
    terminal = DatasetPreviewSnapshot(
        source,
        DatasetCoverage(total_cells, total_cells),
        (None,) * total_cells,
    )
    with slot._lock:
        slot._run_id = "same-run"
        slot._causation_domain_id = source.ref.stream_generation.value
        slot._terminal_snapshot = terminal
        slot._terminal = True
        listener = slot._notify_locked()
    assert listener is not None
    listener()

    def present_until(revision: int):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            wake.wait(0.05)
            wake.clear()
            preview.owner_cycle()
            frame = presenter.current()
            if (
                frame is not None
                and frame.panels[0].coherence_stamp.presentations[0].panel_revision
                == revision
            ):
                return frame
        raise AssertionError(f"progressive display r{revision} was not presented")

    try:
        first = present_until(0)
        first_panel = first.panels[0]
        first_input = first_panel.coherence_stamp.inputs[0].ref
        first_payload = first_panel.display_payload
        assert isinstance(first_payload, CurvePanelPayload)
        assert first_payload.evaluated_input.ref == first_input

        state = curve_display_with_x_view(
            CurveDisplayState(),
            (-0.5, 0.5),
        )
        preview.reconfigure_curve_display(state)
        second = present_until(1)
        second_panel = second.panels[0]
        second_payload = second_panel.display_payload
        assert isinstance(second_payload, CurvePanelPayload)
        assert second_panel.coherence_stamp.run_id == "same-run"
        assert second_panel.coherence_stamp.inputs[0].ref == first_input
        assert second_payload.evaluated_input.ref == first_input
        assert second_payload.viewport.x_limits == pytest.approx((-0.5, 0.5))
        assert preview.curve_display == state
        assert preview.terminal
    finally:
        preview.close()
        for future in futures:
            future.result(timeout=5.0)
        executor.shutdown(wait=True)
    assert preview.worker_done
    assert preview.retired


def test_progressive_candidate_remains_capacity_one_through_owner_present():
    source, transform, schema, _output_ref, values, valid = (
        _component_snapshot_case()
    )
    progressive = build_occupancy_progressive_spec(
        source.block.schema,
        ScanOutputContract(transform, schema),
        identity="capacity-one-owner-present",
    )
    second_block = DataBlock(
        source.block.block_id,
        DatasetRevision(source.ref.revision.value + 1),
        values + 1,
        ComponentValidity((AxisId("site"),), valid),
        source.block.schema,
    )
    second_source = OwnedSnapshot(
        second_block.ref(source.ref.stream_generation),
        second_block,
    )
    total_cells = (
        source.block.schema.repeat_axis.size
        * source.block.schema.point_layout.storage_size
    )
    snapshots = tuple(
        DatasetPreviewSnapshot(
            item,
            DatasetCoverage(total_cells, total_cells),
            (None,) * total_cells,
        )
        for item in (source, second_source)
    )

    class QueuedSlot(ExactDatasetLiveSlot):
        def __init__(self):
            super().__init__(progressive.preview_spec)
            self._queue_lock = threading.Lock()
            self._snapshots = list(snapshots)
            self.returned = 0
            self.second_requested = threading.Event()

        @property
        def terminal(self):
            with self._queue_lock:
                return not self._snapshots

        @property
        def failure(self):
            return None

        def set_change_listener(self, listener):
            listener()

        def wait_and_freeze(self, after, *, timeout):
            del timeout
            with self._queue_lock:
                if not self._snapshots:
                    return None
                candidate = self._snapshots[0]
                if candidate.ref.revision <= after:
                    return None
                self._snapshots.pop(0)
                self.returned += 1
                if self.returned == 2:
                    self.second_requested.set()
            return (
                "capacity-one-run",
                source.ref.stream_generation.value,
                candidate,
            )

    slot = QueuedSlot()

    class BlockingPresenter:
        def __init__(self):
            self.frames = []

        def present(self, frame):
            if not self.frames:
                assert not slot.second_requested.wait(0.2), (
                    "render worker consumed N+1 before N crossed the owner "
                    "present boundary"
                )
            self.frames.append(frame)

        def clear(self):
            self.frames.clear()

    presenter = BlockingPresenter()
    wake = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    futures = []

    def submit(work):
        future = executor.submit(work)
        futures.append(future)
        return future

    preview = ProgressiveScanPreview(
        slot,
        progressive,
        presenter,
        curve_display=CurveDisplayState(),
        submit_worker=submit,
        request_owner_wake=wake.set,
    )
    try:
        deadline = time.monotonic() + 5.0
        while not presenter.frames and time.monotonic() < deadline:
            wake.wait(0.05)
            wake.clear()
            preview.owner_cycle()
        assert presenter.frames
        assert slot.second_requested.wait(2.0)
    finally:
        preview.close()
        for future in futures:
            future.result(timeout=5.0)
        executor.shutdown(wait=True)


def test_progressive_watcher_cannot_starve_single_worker_terminal_result():
    source, transform, schema, _output_ref, _values, _valid = (
        _component_snapshot_case()
    )
    progressive = build_occupancy_progressive_spec(
        source.block.schema,
        ScanOutputContract(transform, schema),
        identity="single-general-worker",
    )
    reference = zlc.ScanArtifactRef("single-worker-repository", "a" * 64)
    run_id = RunId("single-worker-run")
    terminal_snapshot = RunSnapshot(
        run_id,
        RunState.SUCCEEDED,
        "terminal",
        True,
        None,
        None,
        None,
        (),
        None,
    )

    class TerminalHandle:
        def __init__(self):
            self.run_id = run_id

        def snapshot(self):
            return terminal_snapshot

        def cancel(self, reason="cancel"):
            del reason
            return CancelOutcome.ALREADY_TERMINAL

        def wait(self, timeout=None):
            del timeout
            return terminal_snapshot

        def result(self, timeout=None):
            del timeout
            return reference

    total_cells = (
        source.block.schema.repeat_axis.size
        * source.block.schema.point_layout.storage_size
    )
    exact_terminal = DatasetPreviewSnapshot(
        source,
        DatasetCoverage(total_cells, total_cells),
        (None,) * total_cells,
    )

    class Application:
        def prepare(self):
            def start(slot):
                assert isinstance(slot, ExactDatasetLiveSlot)
                with slot._lock:
                    slot._run_id = run_id.value
                    slot._causation_domain_id = source.ref.stream_generation.value
                    slot._terminal_snapshot = exact_terminal
                    slot._terminal = True
                    listener = slot._notify_locked()
                assert listener is not None
                listener()
                return TerminalHandle()

            return PreparedScanPanelRun(progressive, start)

        def project_final(self, source_ref, *, memory_limit_bytes):
            assert source_ref == reference
            assert memory_limit_bytes > 0
            png = (
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR"
                b"\x00\x00\x00\x01\x00\x00\x00\x01"
            )
            return FinalScanPresentation(source_ref, png, "terminal projection")

    general = ThreadPoolExecutor(max_workers=1)
    wake = threading.Event()
    controller = ScanPanelController(
        Application(),
        wake.set,
        executor=general,
        preview_presenter=AtomicBoardFront(),
    )
    try:
        controller.start()
        deadline = time.monotonic() + 8.0
        while (
            controller.view_model.presentation is None
            and time.monotonic() < deadline
        ):
            wake.wait(0.05)
            wake.clear()
            controller.owner_cycle()
        assert controller.view_model.artifact_ref == reference
        assert controller.view_model.presentation is not None
        assert controller.view_model.status == "FINAL"
    finally:
        controller.close()
        deadline = time.monotonic() + 5.0
        while not controller.closed and time.monotonic() < deadline:
            wake.wait(0.05)
            wake.clear()
            controller.owner_cycle()
        general.shutdown(wait=True)
    assert controller.closed


def test_nonmonotonic_scan_axis_uses_explicit_static_progressive_fallback():
    source, transform, schema, output_ref, _values, _valid = (
        _component_snapshot_case(x_coordinates=(0.0, 0.0))
    )
    progressive = build_occupancy_progressive_spec(
        source.block.schema,
        ScanOutputContract(transform, schema),
        identity="nonmonotonic-static-fallback",
    )
    assert not progressive.interactive_curve
    assert "strictly monotonic" in (
        progressive.interaction_unavailable_reason or ""
    )
    assert "static curve" in progressive.projection_summary

    output = materialize_transformed_snapshot(
        source,
        transform,
        output_ref=output_ref,
        output_schema=schema,
        memory_limit_bytes=64 << 20,
    )
    evaluated = FigureEvaluator(
        FigureEvaluationPolicy(max_live_nbytes=progressive.evaluation_peak_bytes)
    ).evaluate(
        progressive.document,
        ResolvedDatasetMap((ResolvedDataset(progressive.dataset_id, output),)),
    )
    renderer = SinglePanelAggRenderer(progressive.document, width=360, height=240)
    try:
        assert renderer.render(evaluated).pixels
    finally:
        renderer.close()


def test_progressive_site_batch_uses_the_curve_contract_exact_boundary():
    limit = CURVE_CONTRACT.maximum_batch_series

    def build(site_count: int, *, mode: str = "auto"):
        repeat = _axis(f"repeat.{site_count}", REPEAT, 1, (0,))
        scan = _axis(f"scan.{site_count}", SCAN_POINT, 2, (0.0, 1.0))
        event = _axis(
            f"event.{site_count}",
            READOUT_EVENT,
            1,
            ("image",),
        )
        site = _axis(
            f"site.{site_count}",
            SITE,
            site_count,
            tuple(range(site_count)),
        )
        source_schema = DatasetSchema(
            repeat,
            (scan,),
            PointLayout.rect_c((2,)),
            ValueSchema(
                (event, site),
                ValidityContract.components(site.axis_id),
                np.dtype("<f8"),
                "count",
            ),
        )
        transform = commit_transform(
            source_schema,
            DataTransformSpec((Selection.index(event.axis_id, 0),)),
        )
        output_schema = DatasetSchema(
            repeat,
            (scan,),
            PointLayout.rect_c((2,)),
            ValueSchema(
                (site,),
                ValidityContract.components(site.axis_id),
                np.dtype("<f8"),
                "count",
            ),
        )
        return build_occupancy_progressive_spec(
            source_schema,
            ScanOutputContract(transform, output_schema),
            identity=f"site-boundary-{site_count}-{mode}",
            display_intent=ScanDisplayIntent(site_mode=mode),
        ), site

    at_limit, at_limit_site = build(limit)
    assert (
        at_limit.document.layers[0].view.binding(at_limit_site.axis_id).role
        is AxisViewRole.BATCH
    )
    over_limit, over_limit_site = build(limit + 1)
    over_binding = over_limit.document.layers[0].view.binding(
        over_limit_site.axis_id
    )
    assert over_binding.role is AxisViewRole.SELECTED
    assert over_binding.selector.index == 0
    with pytest.raises(ValueError, match=f"2 and {limit} sites"):
        build(limit + 1, mode="batch")


def test_progressive_pointer_hold_budget_has_an_exact_one_byte_admission_edge():
    source, transform, schema, _output_ref, _values, _valid = (
        _component_snapshot_case()
    )
    contract = ScanOutputContract(transform, schema)
    progressive = build_occupancy_progressive_spec(
        source.block.schema,
        contract,
        identity="pointer-hold-memory-edge",
    )
    base_raster = estimate_live_panel_raster_peak_nbytes(
        800,
        520,
        evaluated_data_upper_bound_bytes=progressive.evaluation_peak_bytes,
    )
    held_raster = estimate_live_panel_raster_peak_nbytes(
        800,
        520,
        evaluated_data_upper_bound_bytes=progressive.evaluation_peak_bytes,
        extra_retained_fronts=1,
        extra_retained_evaluated_data_bytes=(
            progressive.evaluation_peak_bytes
        ),
    )
    assert held_raster - base_raster == (
        800 * 520 * 4 + progressive.evaluation_peak_bytes
    )
    assert progressive.preview_spec.downstream_peak_bytes == (
        progressive.transform_peak_bytes
        + progressive.evaluation_peak_bytes
        + held_raster
        + dataset_storage_nbytes(source.block.schema)
    )

    from zlc_neutral_atom.scan.application import _admit_final_data_limit

    exact_limit = (
        progressive.transform_peak_bytes
        + progressive.preview_spec.downstream_peak_bytes
    )
    with pytest.raises(MemoryError, match="scan final data-plane peak"):
        _admit_final_data_limit(
            source.block.schema,
            contract,
            memory_limit_bytes=exact_limit - 1,
            retained_overhead_bytes=(
                progressive.preview_spec.downstream_peak_bytes
            ),
        )
    assert _admit_final_data_limit(
        source.block.schema,
        contract,
        memory_limit_bytes=exact_limit,
        retained_overhead_bytes=progressive.preview_spec.downstream_peak_bytes,
    ) == progressive.transform_peak_bytes


def test_progressive_curve_preserves_multidimensional_data_and_component_validity():
    repeat = _axis("repeat.multi", REPEAT, 2, (0, 1))
    scan = _axis("scan.multi", SCAN_POINT, 3, (-1.0, 0.0, 1.0), "MHz")
    event = _axis("event.multi", READOUT_EVENT, 1, ("image",))
    site = _axis("site.multi", SITE, 3, ("left", "middle", "right"))
    component = _axis("component.multi", COMPONENT, 2, ("signal", "aux"))
    raw_schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((3,)),
        ValueSchema(
            (event, site, component),
            ValidityContract.components(site.axis_id, component.axis_id),
            np.dtype("<f8"),
            "count",
        ),
    )
    values = np.arange(
        np.prod(raw_schema.physical_shape),
        dtype=np.float64,
    ).reshape(raw_schema.physical_shape)
    validity = np.ones((2, 3, 3, 2), dtype=bool)
    validity[:, 1, 1, 0] = False
    validity[0, 2, 2, 0] = False
    validity[:, :, :, 1] = False
    raw_block = DataBlock(
        BlockId("raw-multidimensional-scan"),
        DatasetRevision(1),
        values,
        ComponentValidity((site.axis_id, component.axis_id), validity),
        raw_schema,
    )
    source = OwnedSnapshot(
        raw_block.ref(StreamGenerationId("multi-data-generation")),
        raw_block,
    )
    transform = commit_transform(
        raw_schema,
        DataTransformSpec((Selection.index(event.axis_id, 0),)),
    )
    output_schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((3,)),
        ValueSchema(
            (site, component),
            ValidityContract.components(site.axis_id, component.axis_id),
            np.dtype("<f8"),
            "count",
        ),
    )
    output_ref = DatasetRevisionRef(
        BlockId("derived-multidimensional-scan"),
        source.ref.stream_generation,
        output_schema.fingerprint,
        source.ref.revision,
    )
    output = materialize_transformed_snapshot(
        source,
        transform,
        output_ref=output_ref,
        output_schema=output_schema,
        memory_limit_bytes=64 << 20,
    )
    assert output.block.values.shape == (2, 3, 3, 2)
    assert output.block.validity.mask.shape == (2, 3, 3, 2)
    assert output.block.schema.cell_schema.data_axes == (site, component)

    progressive = build_occupancy_progressive_spec(
        raw_schema,
        ScanOutputContract(transform, output_schema),
        identity="multidimensional-data-axis",
    )
    view = progressive.document.layers[0].view
    assert view.binding(site.axis_id).role is AxisViewRole.BATCH
    component_binding = view.binding(component.axis_id)
    assert component_binding.role is AxisViewRole.SELECTED
    assert component_binding.selector.index == 0
    assert "component.multi=signal" in progressive.projection_summary

    evaluated = FigureEvaluator(
        FigureEvaluationPolicy(max_live_nbytes=progressive.evaluation_peak_bytes)
    ).evaluate(
        progressive.document,
        ResolvedDatasetMap((ResolvedDataset(progressive.dataset_id, output),)),
    )
    renderer = SinglePanelAggRenderer(progressive.document, width=360, height=240)
    try:
        _raster, payload = renderer.render_interactive_curve(
            evaluated,
            CurveDisplayState(),
            current_y_limits=None,
            previous_relim_mode=None,
        )
    finally:
        renderer.close()
    assert len(payload.series) == 3
    for site_index, series in enumerate(payload.series):
        source_values = values[:, :, 0, site_index, 0]
        source_valid = validity[:, :, site_index, 0]
        expected_valid = source_valid.any(axis=0)
        expected = np.zeros(3, dtype=np.float64)
        for point_index in range(3):
            contributors = source_values[:, point_index][
                source_valid[:, point_index]
            ]
            if contributors.size:
                expected[point_index] = contributors.mean()
        np.testing.assert_array_equal(series.data.validity, expected_valid)
        np.testing.assert_allclose(
            series.data.values[expected_valid],
            expected[expected_valid],
        )


def test_bounded_snapshot_rejects_cell_reduction():
    repeat = _axis("repeat", REPEAT, 1, (0,))
    point = _axis("scan.point", SCAN_POINT, 2, (0, 1))
    source_schema = DatasetSchema(
        repeat,
        (point,),
        PointLayout.rect_c((2,)),
        ValueSchema((), ValidityContract.value(), np.dtype("<i2")),
    )
    block = DataBlock(
        BlockId("cell-reduction-source"),
        DatasetRevision(0),
        np.asarray(((1, 2),), dtype="<i2"),
        VALID,
        source_schema,
    )
    source = OwnedSnapshot(
        block.ref(StreamGenerationId("cell-reduction-generation")), block
    )
    cell_reduction = commit_transform(
        source_schema,
        DataTransformSpec(
            (ReductionSpec((point.axis_id,), ReductionMethod.MEAN),)
        ),
    )
    output_schema = DatasetSchema(
        repeat,
        (),
        PointLayout.rect_c(()),
        ValueSchema((), ValidityContract.value(), np.dtype("<f8")),
    )
    output_ref = DatasetRevisionRef(
        BlockId("cell-reduction-output"),
        source.ref.stream_generation,
        output_schema.fingerprint,
        source.ref.revision,
    )
    with pytest.raises(ValueError, match="do not reduce repeat/point axes"):
        materialize_transformed_snapshot(
            source,
            cell_reduction,
            output_ref=output_ref,
            output_schema=output_schema,
            memory_limit_bytes=64 << 20,
        )


def test_bounded_snapshot_reduces_only_the_named_trailing_axis():
    source, _transform, _schema, _output_ref, _values, _valid = (
        _component_snapshot_case()
    )
    source_schema = source.block.schema
    transform = commit_transform(
        source_schema,
        DataTransformSpec(
            (
                Selection.index(AxisId("readout.event"), 0),
                ReductionSpec(
                    (AxisId("site"),),
                    ReductionMethod.MEAN,
                    validity_policy=ValidityPolicy.OMIT_INVALID,
                ),
            )
        ),
    )
    output_schema = DatasetSchema(
        source_schema.repeat_axis,
        source_schema.point_axes,
        source_schema.point_layout,
        ValueSchema((), ValidityContract.value(), np.dtype("<f8"), "count"),
    )
    output_ref = DatasetRevisionRef(
        BlockId("derived-scalar-scan"),
        source.ref.stream_generation,
        output_schema.fingerprint,
        source.ref.revision,
    )
    output = materialize_transformed_snapshot(
        source,
        transform,
        output_ref=output_ref,
        output_schema=output_schema,
        memory_limit_bytes=64 << 20,
    )
    assert output.block.values.shape == (2, 3)
    np.testing.assert_allclose(
        output.block.values,
        ((1.0, 3.5, 7.5), (10.0, 14.0, 16.0)),
    )
    assert isinstance(output.block.validity, Valid)


def test_public_sparse_scan_reopens_with_stable_identity_and_data_figure(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    document = _sparse_scan_document()
    expected_points = ScanPointTable.from_pulse_document(document)

    with zlc.connect("virtual", repository=workspace) as exp:
        request = exp.readout.scan_request(document, timeout_seconds=15.0)

        def forbidden_stage(*_args, **_kwargs):
            raise AssertionError("inspect_scan must not stage repository blobs")

        with monkeypatch.context() as patch:
            patch.setattr(
                ScanRepository,
                "_stage_static_lineage",
                forbidden_stage,
            )
            descriptor = exp.inspect_scan(request)
        assert descriptor.expected_frames == 6
        with pytest.raises(MemoryError, match="scan final data-plane peak"):
            exp.scan(
                exp.readout.scan_request(
                    document,
                    memory_limit_bytes=1,
                    timeout_seconds=15.0,
                )
            )
        import zlc_neutral_atom.scan.application as scan_application

        base_compiled = False

        def forbidden_base_compile(*_args, **_kwargs):
            nonlocal base_compiled
            base_compiled = True
            raise AssertionError("hardware plan compiled below static-lineage admission")

        with monkeypatch.context() as patch:
            patch.setattr(
                scan_application,
                "compile_triggered_pipeline",
                forbidden_base_compile,
            )
            with pytest.raises(MemoryError, match="scan static-lineage peak"):
                exp.scan(
                    exp.readout.scan_request(
                        document,
                        memory_limit_bytes=1 << 20,
                        timeout_seconds=15.0,
                    )
                )
        assert not base_compiled
        scan_ref = exp.scan(
            exp.readout.scan_request(document, timeout_seconds=15.0)
        )
        with pytest.raises(MemoryError):
            exp.readout.materialize_scan(scan_ref, memory_limit_bytes=1)
        data = exp.readout.materialize_scan(scan_ref)
        artifact = exp.readout.load_scan(scan_ref)

        assert data.artifact_ref == artifact.ref
        assert data.source_dataset_ref == artifact.source_dataset_ref
        assert data.snapshot.ref == artifact.output_dataset_ref
        assert data.snapshot.ref != artifact.source_dataset_ref
        assert data.snapshot.ref.block_id.value.startswith("scan-output-")
        assert data.values.shape == (2, 3, 96, 128)
        assert data.schema.repeat_axis.size == 2
        assert data.schema.point_axes == expected_points.point_axes
        assert data.schema.point_layout == expected_points.point_layout
        assert (
            data.schema.cell_schema.data_axes
            == artifact.source_dataset_schema.cell_schema.data_axes
        )
        assert any(
            axis.role == READOUT_EVENT
            for axis in artifact.source_dataset_schema.point_axes
        )
        assert all(axis.role != READOUT_EVENT for axis in data.schema.point_axes)
        assert artifact.provenance.derivation is None
        assert isinstance(artifact.execution, AutonomousScanExecution)
        assert artifact.execution.evidence.expected_trigger_count == 6
        camera = artifact.execution.camera
        assert camera.event_count == 6
        assert camera.terminal.session_id
        assert camera.terminal.produced_count == 6
        assert camera.terminal.drained_count == 6
        assert camera.arm_spec.digest == camera.terminal.capture_spec_fingerprint
        assert camera.capability.fingerprint == camera.terminal.capability_fingerprint
        assert (
            camera.source_schema_fingerprint
            == artifact.source_dataset_schema.fingerprint
        )
        camera.validate_dataset_provenance(artifact.provenance)

        # Exercise the durable reload boundary, not just the live PipelineResult:
        # a well-typed forged aggregate digest must still be rejected against
        # the independently persisted DatasetSealProvenance.
        import zlc_neutral_atom.scan.repository as scan_repository

        decode_index = scan_repository._decode_metadata_index

        def forged_camera_metadata(payload):
            index = decode_index(payload)
            execution_tree = copy.deepcopy(index.execution_tree)
            execution_tree["camera"]["terminal"][
                "ordered_metadata_digest"
            ] = "0" * 64
            return replace(index, execution_tree=execution_tree)

        with monkeypatch.context() as patch:
            patch.setattr(
                scan_repository,
                "_decode_metadata_index",
                forged_camera_metadata,
            )
            with pytest.raises(
                ValueError,
                match="raw dataset provenance differs from camera aggregate evidence",
            ):
                exp.readout.load_scan(scan_ref)
        assert np.all(np.isfinite(data.values))
        assert not data.values.flags.writeable
        assert exp.readout.materialize_scan(scan_ref).snapshot.ref == data.snapshot.ref

        def forbidden_heavy_read(*_args, **_kwargs):
            raise AssertionError("metadata-only inspection decoded heavy lineage/data")

        with monkeypatch.context() as patch:
            patch.setattr(ScanRepository, "materialize", forbidden_heavy_read)
            patch.setattr(
                "zlc_neutral_atom.scan.repository.decode_compiled_pulse_artifact",
                forbidden_heavy_read,
            )
            patch.setattr(
                "zlc_neutral_atom.scan.repository._decode_program",
                forbidden_heavy_read,
            )
            figure_document = exp.figure_document(scan_ref)
        assert figure_document.datasets[0].schema_fingerprint == data.schema.fingerprint
        assert figure_document.layers[0].view.intent is ViewIntent.IMAGE

        figure = exp.figure(scan_ref)
        assert figure.document.datasets == figure_document.datasets
        with pytest.raises(MemoryError, match="figure render peak"):
            figure.to_png_bytes(memory_limit_bytes=1)
        assert figure.to_png_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        with pytest.raises(TypeError, match="CaptureArtifactRef"):
            exp.fit(scan_ref, model="gaussian_offset")

        _assert_public_occupancy_scan(exp, monkeypatch)
        _assert_scan_window(exp, document, monkeypatch)

    digest = hashlib.sha256(data.values.tobytes()).hexdigest()
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import hashlib,sys; import Zou_lab_control.notebook as zlc; "
                "ref=zlc.ScanArtifactRef(sys.argv[2],sys.argv[3]); "
                "exp=zlc.connect('virtual',repository=sys.argv[1]); "
                "data=exp.readout.materialize_scan(ref); "
                "artifact=exp.readout.load_scan(ref); "
                "assert artifact.execution.camera.terminal.produced_count==6; "
                "assert artifact.execution.camera.terminal.drained_count==6; "
                "artifact.execution.camera.validate_dataset_provenance(artifact.provenance); "
                "assert data.snapshot.ref.block_id.value==sys.argv[4]; "
                "assert data.snapshot.ref.schema_fingerprint==sys.argv[5]; "
                "assert data.snapshot.ref.stream_generation.value==sys.argv[6]; "
                "assert str(data.snapshot.ref.revision.value)==sys.argv[7]; "
                "assert hashlib.sha256(data.values.tobytes()).hexdigest()==sys.argv[8]; "
                "exp.close()"
            ),
            str(workspace),
            scan_ref.repository_id,
            scan_ref.manifest_digest,
            data.snapshot.ref.block_id.value,
            data.snapshot.ref.schema_fingerprint,
            data.snapshot.ref.stream_generation.value,
            str(data.snapshot.ref.revision.value),
            digest,
        ],
        cwd=ROOT,
        check=True,
        timeout=30,
    )


def _assert_public_occupancy_scan(exp, monkeypatch):
    document = _occupancy_scan_document()
    points = ScanPointTable.from_pulse_document(document)
    values = _fixed_api_values(document)
    original_parameters = document.api_parameters
    original_table = document.scan_table
    with pytest.raises(ValueError, match="missing"):
        exp.readout.scan_request(document)
    with pytest.raises(ValueError, match="missing=.*extra=.*not-an-api"):
        exp.readout.scan_request(document, api_values={"not-an-api": 1})
    direct_request = exp.readout.scan_request(document, api_values=values)
    assert isinstance(direct_request.program, AutonomousScanSlotProgram)
    assert direct_request.program.document == document
    assert direct_request.program.execution_document.api_parameters == ()
    assert (
        direct_request.program.execution_document.scan_parameters
        == document.scan_parameters
    )
    assert direct_request.program.execution_document.scan_table == original_table
    assert document.api_parameters == original_parameters
    with pytest.raises(ValueError, match="missing=.*extra=.*da_x"):
        exp.readout.scan_request(
            _sparse_scan_document(),
            api_values={"da_x": 0},
        )
    calibration_ref = exp.readout.sitemap(frames=6)
    request = exp.readout.occupancy_scan_request(
        document,
        calibration_ref=calibration_ref,
        api_values=values,
        timeout_seconds=20.0,
    )
    assert isinstance(request.program, AutonomousScanSlotProgram)
    assert request.program.execution_document.api_parameters == ()
    guarded = _prepare_occupancy_scan_for_workbench(exp, request)

    @contextmanager
    def closed_guard(_token):
        raise RuntimeError("Experiment is closed")
        yield

    with monkeypatch.context() as patch:
        patch.setattr(
            "Zou_lab_control.notebook.facade._service_guard",
            closed_guard,
        )
        with pytest.raises(RuntimeError, match="Experiment is closed"):
            guarded.start()

    malformed_prepared = _prepare_occupancy_scan_for_workbench(exp, request)
    malformed = ExactDatasetLiveSlot(
        ExactDatasetPreviewSpec(malformed_prepared.source_schema.fingerprint, 1)
    )
    with pytest.raises(MemoryError, match="frozen source snapshot"):
        malformed_prepared.start(malformed)
    assert malformed.terminal
    assert "frozen source snapshot" in (malformed.failure or "")

    # This port is individually valid and can retain its complete exact
    # source.  Only its additional aggregate footprint exceeds the request's
    # science budget, so the neutral compiler—not the facade—must reject the
    # optional branch and admit the same Run FINAL-only.
    rejected = _prepare_occupancy_scan_for_workbench(exp, request)
    capacity_only = ExactDatasetLiveSlot(
        ExactDatasetPreviewSpec(
            rejected.source_schema.fingerprint,
            request.memory_limit_bytes,
        )
    )
    import zlc_neutral_atom.scan.application as scan_application

    admitted_bases = []
    original_admit_preview = scan_application._admit_optional_preview_data_limit

    def record_preview_admission(*args, **kwargs):
        admitted_bases.append(kwargs["retained_base_bytes"])
        return original_admit_preview(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            scan_application,
            "_admit_optional_preview_data_limit",
            record_preview_admission,
        )
        fallback_handle = rejected.start(capacity_only)
    fallback_reference = fallback_handle.result(timeout=30.0)
    assert isinstance(fallback_reference, zlc.ScanArtifactRef)
    assert len(admitted_bases) == 1
    assert admitted_bases[0] > 0
    assert capacity_only.terminal
    assert "scan final retained overhead" in (capacity_only.failure or "")
    second_start = ExactDatasetLiveSlot(capacity_only.spec)
    with pytest.raises(RuntimeError, match="one-shot"):
        rejected.start(second_start)
    assert second_start.terminal
    assert "one-shot" in (second_start.failure or "")

    # FINAL-transform admission is not the only phase that retains the
    # optional preview.  The bound camera→processor preflight owns a larger
    # independent peak formula and must drop only that branch when its science
    # baseline still fits—before camera prepare/FIRE—rather than failing Run.
    pipeline_rejected = _prepare_occupancy_scan_for_workbench(exp, request)
    pipeline_progressive = build_occupancy_progressive_spec(
        pipeline_rejected.source_schema,
        pipeline_rejected.output_contract,
        identity="w3-pipeline-preview-capacity",
    )
    pipeline_slot = _CountingExactDatasetLiveSlot(
        pipeline_progressive.preview_spec
    )
    import zlc_neutral_atom.readout.occupancy_pipeline as occupancy_pipeline

    original_pipeline_peak = occupancy_pipeline._estimate_peak_bytes
    pipeline_peak_calls = []

    def preview_exceeds_pipeline_only(
        spec,
        bound,
        preview_spec=None,
        *,
        retained_overhead_bytes=0,
    ):
        actual = original_pipeline_peak(
            spec,
            bound,
            preview_spec,
            retained_overhead_bytes=retained_overhead_bytes,
        )
        pipeline_peak_calls.append(preview_spec is not None)
        if preview_spec is not None:
            actual_baseline = original_pipeline_peak(
                spec,
                bound,
                None,
                retained_overhead_bytes=retained_overhead_bytes,
            )
            assert actual - actual_baseline == preview_spec.downstream_peak_bytes
            return spec.memory_limit_bytes + 1
        assert actual <= spec.memory_limit_bytes
        return actual

    with monkeypatch.context() as patch:
        patch.setattr(
            occupancy_pipeline,
            "_estimate_peak_bytes",
            preview_exceeds_pipeline_only,
        )
        pipeline_handle = pipeline_rejected.start(pipeline_slot)
        pipeline_reference = pipeline_handle.result(timeout=30.0)
    assert isinstance(pipeline_reference, zlc.ScanArtifactRef)
    assert pipeline_peak_calls == [True, False]
    assert pipeline_slot.bind_calls == 0
    assert pipeline_slot.terminal
    assert "optional preview" in (pipeline_slot.failure or "")

    import zlc_neutral_atom.timing.occupancy as timing_occupancy

    failed_prepared = _prepare_occupancy_scan_for_workbench(exp, request)
    failed_progressive = build_occupancy_progressive_spec(
        failed_prepared.source_schema,
        failed_prepared.output_contract,
        identity="w3-post-safety-failure",
    )
    failed_slot = ExactDatasetLiveSlot(failed_progressive.preview_spec)

    def reject_post_safety(*_args, **_kwargs):
        raise RuntimeError("post-safety occupancy finalization rejected")

    with monkeypatch.context() as patch:
        patch.setattr(
            timing_occupancy,
            "finalize_occupancy_result",
            reject_post_safety,
        )
        failed_handle = failed_prepared.start(failed_slot)
        with pytest.raises(RunFailed, match="post-safety occupancy finalization"):
            failed_handle.result(timeout=30.0)
    assert failed_slot.terminal
    assert "post-safety occupancy finalization" in (failed_slot.failure or "")

    prepared = _prepare_occupancy_scan_for_workbench(exp, request)
    progressive = build_occupancy_progressive_spec(
        prepared.source_schema,
        prepared.output_contract,
        identity="w3-occupancy",
    )
    site_axis = prepared.output_contract.output_dataset_schema.cell_schema.data_axes[0]
    site_binding = next(
        binding
        for binding in progressive.document.layers[0].view.axis_bindings
        if binding.axis_id == site_axis.axis_id
    )
    assert site_axis.size == 35
    assert site_axis.size > CURVE_CONTRACT.maximum_batch_series
    assert site_binding.role is AxisViewRole.SELECTED
    assert site_binding.selector.index == 0
    assert f"{site_axis.name}={site_axis.coordinate_at(0)}" in (
        progressive.projection_summary
    )
    slot = _CountingExactDatasetLiveSlot(progressive.preview_spec)
    handle = prepared.start(slot)
    scan_ref = handle.result(timeout=30.0)
    assert slot.terminal
    assert slot.failure is None
    assert slot.source_terminal_calls == 1
    provisional = slot.wait_and_freeze(DatasetRevision(0), timeout=0)
    assert provisional is not None
    _run_id, _causation, preview = provisional
    assert preview.coverage.complete
    assert preview.block.schema.fingerprint == prepared.source_schema.fingerprint
    slot.close()
    artifact = exp.readout.load_scan(scan_ref)
    data = exp.readout.materialize_scan(scan_ref)

    assert data.schema.repeat_axis.size == 2
    assert data.schema.point_axes == points.point_axes
    assert data.schema.point_layout == points.point_layout
    assert data.values.shape[:2] == (2, 2)
    assert len(data.values.shape) == 3
    assert len(data.schema.cell_schema.data_axes) == 1
    site_axis = data.schema.cell_schema.data_axes[0]
    assert site_axis.role == SITE
    assert data.values.shape[2] == site_axis.size
    assert isinstance(data.validity, ComponentValidity)
    assert data.validity.axis_ids == (site_axis.axis_id,)
    assert data.validity.mask.shape == data.values.shape

    derivation = artifact.provenance.derivation
    assert derivation is not None
    assert len(derivation.stages) == 1
    assert calibration_artifact_input_ref(calibration_ref) in (
        derivation.artifact_inputs
    )
    assert artifact.source_dataset_ref == data.source_dataset_ref
    assert artifact.output_dataset_ref == data.snapshot.ref
    assert any(
        axis.role == READOUT_EVENT
        for axis in artifact.source_dataset_schema.point_axes
    )
    assert all(axis.role != READOUT_EVENT for axis in data.schema.point_axes)
    assert isinstance(artifact.execution, AutonomousScanExecution)
    assert artifact.execution.evidence.expected_trigger_count == 4
    _assert_occupancy_scan_window(exp, request, monkeypatch)


def _assert_occupancy_scan_window(exp, request, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtCore, QtGui, QtTest, QtWidgets
    from zlc_frontend.qt_widgets import (
        FluentPopup,
        FluentRevisionedFormEditor,
        QtRasterBoard,
    )
    import zlc_workbench.progressive_scan as progressive_scan
    import zlc_neutral_atom.timing.occupancy as timing_occupancy

    owner_thread = threading.get_ident()
    renderer_construct_threads = []
    raster_threads = []
    renderer_ids = []
    renderer_close_threads = []
    present_threads = []
    presented_frames = []
    finalization_reached = threading.Event()
    allow_finalization = threading.Event()
    original_init = progressive_scan.SinglePanelAggRenderer.__init__
    original_render = (
        progressive_scan.SinglePanelAggRenderer.render_interactive_curve
    )
    original_close = progressive_scan.SinglePanelAggRenderer.close
    original_present = QtRasterBoard.present
    original_finalize = timing_occupancy.finalize_occupancy_result

    def record_init(renderer, *args, **kwargs):
        renderer_construct_threads.append(threading.get_ident())
        return original_init(renderer, *args, **kwargs)

    def record_render(renderer, evaluated, state, **kwargs):
        raster_threads.append(threading.get_ident())
        renderer_ids.append(id(renderer))
        return original_render(renderer, evaluated, state, **kwargs)

    def record_close(renderer):
        renderer_close_threads.append(threading.get_ident())
        return original_close(renderer)

    def record_present(board, frame):
        present_threads.append(threading.get_ident())
        result = original_present(board, frame)
        presented_frames.append(frame)
        return result

    def hold_finalization(*args, **kwargs):
        finalization_reached.set()
        if not allow_finalization.wait(10.0):
            raise RuntimeError("W3 interaction oracle did not release finalization")
        return original_finalize(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            progressive_scan.SinglePanelAggRenderer,
            "__init__",
            record_init,
        )
        patch.setattr(
            progressive_scan.SinglePanelAggRenderer,
            "render_interactive_curve",
            record_render,
        )
        patch.setattr(
            progressive_scan.SinglePanelAggRenderer,
            "close",
            record_close,
        )
        patch.setattr(QtRasterBoard, "present", record_present)
        patch.setattr(
            timing_occupancy,
            "finalize_occupancy_result",
            hold_finalization,
        )
        window = exp.scan_gui(request)
        application = QtWidgets.QApplication.instance()
        assert application is not None
        assert "PROVISIONAL OCCUPANCY" in window.findChild(
            QtWidgets.QLabel,
            "scanMode",
        ).text()
        start = window.findChild(QtWidgets.QPushButton, "startScanButton")
        assert start is not None and start.isEnabled()
        start.click()

        board = window.findChild(QtRasterBoard, "scanProvisionalBoard")
        selector = window.findChild(
            QtWidgets.QAbstractButton,
            "scanSelectorSwitch",
        )
        setting = window.findChild(
            QtWidgets.QPushButton,
            "scanDisplaySettingButton",
        )
        assert board is not None
        assert selector is not None
        assert setting is not None

        # Hold only post-safety finalization.  The real virtual camera,
        # processor, exact preview slot, renderer worker, Qt board, and Run all
        # execute normally; this leaves a deterministic interval in which the
        # actual W3 product must accept interaction before the canonical FINAL
        # swap.
        deadline = time.monotonic() + 10.0
        while (
            (
                not finalization_reached.is_set()
                or not presented_frames
                or not selector.isEnabled()
                or board.visible_curve_payload() is None
            )
            and time.monotonic() < deadline
        ):
            application.processEvents()
            time.sleep(0.01)
        assert finalization_reached.is_set()
        assert presented_frames
        assert selector.isEnabled()
        assert setting.isEnabled()

        initial_frame = presented_frames[-1]
        initial_panel = initial_frame.panels[0]
        initial_payload = initial_panel.display_payload
        assert isinstance(initial_payload, CurvePanelPayload)
        assert initial_payload.viewport.x_axis.role == SCAN_POINT
        assert initial_payload.evaluated_input in initial_panel.coherence_stamp.inputs
        initial_run_id = initial_panel.coherence_stamp.run_id
        initial_source_ref = initial_payload.evaluated_input.ref
        assert window.final_reference is None

        try:
            selector.setChecked(True)
            application.processEvents()
            plot = board._curve_target()[0]
            center = QtCore.QPoint(
                int(round(plot.center().x())),
                int(round(plot.center().y())),
            )

            # H/C/A are exercised on the exact painted real-product payload.
            QtTest.QTest.mouseMove(board, center)
            assert board._curve_hover is not None
            QtTest.QTest.mouseClick(board, QtCore.Qt.RightButton, pos=center)
            assert board._curve_cross is not None
            left = QtCore.QPoint(
                int(round(plot.left() + 0.25 * plot.width())),
                center.y(),
            )
            right = QtCore.QPoint(
                int(round(plot.left() + 0.75 * plot.width())),
                center.y(),
            )
            QtTest.QTest.mousePress(board, QtCore.Qt.LeftButton, pos=left)
            board.mouseMoveEvent(
                QtGui.QMouseEvent(
                    QtCore.QEvent.MouseMove,
                    QtCore.QPointF(right),
                    QtCore.Qt.NoButton,
                    QtCore.Qt.LeftButton,
                    QtCore.Qt.NoModifier,
                )
            )
            QtTest.QTest.mouseRelease(board, QtCore.Qt.LeftButton, pos=right)
            assert window._curve_range_candidate is not None

            # Z commits one DISPLAY-only revision.  The live source may advance
            # while the worker repaints, but the same renderer and source
            # identity must remain attached without restarting or changing the
            # authoritative Run/artifact path.  The terminal same-revision
            # repaint is covered independently above.
            wheel = QtGui.QWheelEvent(
                QtCore.QPointF(center),
                QtCore.QPointF(board.mapToGlobal(center)),
                QtCore.QPoint(),
                QtCore.QPoint(0, -120),
                QtCore.Qt.NoButton,
                QtCore.Qt.NoModifier,
                QtCore.Qt.ScrollUpdate,
                False,
            )
            board.wheelEvent(wheel)
            assert wheel.isAccepted()
            assert window._curve_range_candidate is None

            deadline = time.monotonic() + 5.0
            revised_frame = None
            while time.monotonic() < deadline:
                application.processEvents()
                for candidate in reversed(presented_frames):
                    presentation = candidate.panels[0].coherence_stamp.presentations[0]
                    if presentation.panel_revision == 1:
                        revised_frame = candidate
                        break
                if revised_frame is not None:
                    break
                time.sleep(0.01)
            assert revised_frame is not None
            revised_panel = revised_frame.panels[0]
            revised_payload = revised_panel.display_payload
            assert isinstance(revised_payload, CurvePanelPayload)
            assert revised_panel.coherence_stamp.run_id == initial_run_id
            revised_source_ref = revised_payload.evaluated_input.ref
            assert revised_source_ref.block_id == initial_source_ref.block_id
            assert (
                revised_source_ref.stream_generation
                == initial_source_ref.stream_generation
            )
            assert (
                revised_source_ref.schema_fingerprint
                == initial_source_ref.schema_fingerprint
            )
            assert revised_source_ref.revision >= initial_source_ref.revision
            assert window._controller.view_model.run_id == initial_run_id
            assert window.final_reference is None
            edit_editor = window.findChild(
                FluentRevisionedFormEditor,
                "scanCurveEditEditor",
            )
            setting_editor = window.findChild(
                FluentRevisionedFormEditor,
                "scanCurveSettingEditor",
            )
            assert edit_editor is not None and edit_editor.base_revision == 1
            assert setting_editor is not None and setting_editor.base_revision == 1
            setting.click()
            application.processEvents()
            popup = window.findChild(FluentPopup, "scanDisplaySettingsPopup")
            assert popup is not None and popup.isVisible()

            # A board callback fault is detached locally instead of escaping
            # through Qt.  The W3 surface must consume that latched fact on
            # its next owner cycle: no checked-but-dead selector, no stale
            # popup, and a visible operator diagnostic.
            def reject_curve_intent(_command):
                raise RuntimeError("W3 callback fault oracle")

            board._curve_callback = reject_curve_intent
            fault_wheel = QtGui.QWheelEvent(
                QtCore.QPointF(center),
                QtCore.QPointF(board.mapToGlobal(center)),
                QtCore.QPoint(),
                QtCore.QPoint(0, -120),
                QtCore.Qt.NoButton,
                QtCore.Qt.NoModifier,
                QtCore.Qt.ScrollUpdate,
                False,
            )
            board.wheelEvent(fault_wheel)
            assert board.curve_selector_fault is not None
            window._owner_cycle()
            diagnostics = window.findChild(QtWidgets.QLabel, "scanDiagnostics")
            assert diagnostics is not None
            assert "W3 callback fault oracle" in diagnostics.text()
            assert not selector.isEnabled()
            assert not selector.isChecked()
            assert not setting.isEnabled()
            assert not popup.isVisible()
            assert "callback failure" in selector.toolTip()
        finally:
            allow_finalization.set()

        deadline = time.monotonic() + 20.0
        while (
            (
                window.final_reference is None
                or not raster_threads
                or not present_threads
                or not window.can_reconfigure
            )
            and time.monotonic() < deadline
        ):
            application.processEvents()
            time.sleep(0.01)
        assert window.final_reference is not None
        assert raster_threads
        assert renderer_construct_threads
        assert len(set(renderer_ids)) == 1
        assert renderer_close_threads
        assert present_threads
        assert all(thread != owner_thread for thread in raster_threads)
        assert set(renderer_construct_threads) == set(raster_threads)
        assert set(renderer_close_threads) == set(raster_threads)
        assert set(present_threads) == {owner_thread}
        assert not selector.isEnabled()
        assert not selector.isChecked()
        assert not setting.isEnabled()
        assert not popup.isVisible()

        # Replacing the complete scan application starts a new revision
        # domain.  Both shared Setting/Edit surfaces must discard the old r1
        # draft and accept the new owner's r0 default without a backwards-
        # revision exception or partial reconfigure.
        assert window.can_reconfigure
        window.reconfigure(request)
        assert window._curve_display == CurveDisplayState()
        assert edit_editor.base_revision == 0
        assert setting_editor.base_revision == 0
        assert edit_editor._form.read_all() == setting_editor._form.read_all()
        assert not edit_editor._dirty and not setting_editor._dirty

        window.close()
        deadline = time.monotonic() + 5.0
        while window.isVisible() and time.monotonic() < deadline:
            application.processEvents()
            time.sleep(0.01)
        assert not window.isVisible()
        assert window.worker_idle
        assert window not in getattr(application, "_zlc_retained_windows", ())


def _assert_scan_window(exp, document, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from zlc_frontend.qt_widgets import QtImageBoard

    request = exp.readout.scan_request(document, timeout_seconds=15.0)
    window = exp.scan_gui(request)
    application = QtWidgets.QApplication.instance()
    assert application is not None
    assert application.primaryScreen().availableGeometry().contains(
        window.frameGeometry()
    )
    assert "FINAL-ONLY" in window.findChild(
        QtWidgets.QLabel,
        "scanMode",
    ).text()
    start = window.findChild(QtWidgets.QPushButton, "startScanButton")
    assert start is not None and start.isEnabled()
    start.click()

    raster = window.findChild(QtImageBoard, "scanRaster")
    assert raster is not None
    deadline = time.monotonic() + 15.0
    while (
        (
            window.final_reference is None
            or not raster.has_front
        )
        and time.monotonic() < deadline
    ):
        application.processEvents()
        time.sleep(0.01)
    assert window.final_reference is not None
    assert raster.has_front
    assert application.primaryScreen().availableGeometry().contains(
        window.frameGeometry()
    )

    assert start.isEnabled()
    decode_calls = []

    def reject_final_png(_board, _payload, *, image_format):
        decode_calls.append(image_format)
        raise ValueError("injected Qt PNG decode rejection")

    with monkeypatch.context() as patch:
        patch.setattr(QtImageBoard, "present_encoded", reject_final_png)
        start.click()
        assert not raster.has_front
        diagnostics = window.findChild(QtWidgets.QLabel, "scanDiagnostics")
        assert diagnostics is not None
        deadline = time.monotonic() + 15.0
        while (
            "injected Qt PNG decode rejection" not in diagnostics.text()
            and time.monotonic() < deadline
        ):
            application.processEvents()
            time.sleep(0.01)
        assert "injected Qt PNG decode rejection" in diagnostics.text()
        for _ in range(8):
            application.processEvents()
            time.sleep(0.02)
        assert decode_calls == ["PNG"]

    window.close()
    deadline = time.monotonic() + 5.0
    while window.isVisible() and time.monotonic() < deadline:
        application.processEvents()
        time.sleep(0.01)
    assert not window.isVisible()
    assert window.worker_idle
    assert window not in getattr(application, "_zlc_retained_windows", ())
