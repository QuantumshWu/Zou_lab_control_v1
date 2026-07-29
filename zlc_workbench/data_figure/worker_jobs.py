"""DataFigure worker jobs for snapshot Fit, raster work, and atomic export.

All Figure classification, renderer choice, payload construction, visual defaults,
and front validation belong to :mod:`zlc_frontend.data_figure_render`.  This leaf
owns only pure worker calls around application-supplied capabilities and committing
encoded bytes to disk.  Scheduling belongs to the Workbench composition owner.
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass
from pathlib import Path
import threading
import time

from zlc_data import (
    FitDeadlineExceeded,
    FitResultBatch,
    FitSpec,
    Selection,
)
from zlc_frontend import (
    BoardFrame,
    DataFigure,
    FitAuthoringOption,
    HistogramBinProjection,
    validate_fit_authoring_options,
)
from zlc_frontend.data_figure_presentation import (
    DataFigureDisplayState,
    fit_result_draft_summary,
)
from zlc_frontend.data_figure_render import encode_data_figure_front_png
from zlc_frontend.plot_layout import PanelSurfaceGeometry
from zlc_workbench.window_runtime import stage_and_replace_export


@dataclass(frozen=True, slots=True)
class DataFigureSurfaceResult:
    """One worker result tied to the exact Qt raster-surface revision."""

    surface_revision: int
    payload: object

    def __post_init__(self) -> None:
        if (
            isinstance(self.surface_revision, bool)
            or not isinstance(self.surface_revision, int)
            or self.surface_revision < 0
        ):
            raise ValueError("surface_revision must be a non-negative integer")


def _execute_surface_work(
    function,
    args: tuple[object, ...],
    geometry: PanelSurfaceGeometry,
    surface_revision: int,
) -> DataFigureSurfaceResult:
    """Execute a render against one immutable, request-frozen surface."""

    if not callable(function):
        raise TypeError("surface worker must be callable")
    if not isinstance(args, tuple):
        raise TypeError("surface worker args must be a tuple")
    if not isinstance(geometry, PanelSurfaceGeometry):
        raise TypeError("surface worker geometry must be PanelSurfaceGeometry")
    return DataFigureSurfaceResult(
        surface_revision,
        function(*args, geometry),
    )


def _require_not_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise CancelledError()


def _prepare_fit_options(
    prepare,
    figure: DataFigure,
    selection: Selection | None,
    histogram_projection: HistogramBinProjection | None,
) -> tuple[FitAuthoringOption, ...]:
    if not isinstance(figure, DataFigure):
        raise TypeError("Fit preparation requires the exact visible DataFigure")
    if histogram_projection is not None and not isinstance(
        histogram_projection,
        HistogramBinProjection,
    ):
        raise TypeError("histogram_projection must be HistogramBinProjection or None")
    options = tuple(prepare(figure, selection, histogram_projection))
    if not options or any(
        not isinstance(option, FitAuthoringOption) for option in options
    ):
        raise ValueError("Fit preparation produced no FitAuthoringOption")
    models = tuple(option.spec.model_id for option in options)
    if len(models) != len(set(models)):
        raise ValueError("Fit options require unique models")
    return validate_fit_authoring_options(
        options,
        figure=figure,
        selection=selection,
        histogram_projection=histogram_projection,
    )


def _execute_snapshot_fit(
    execute,
    result_of,
    figure: DataFigure,
    source_frame: BoardFrame,
    result_identity: str,
    spec: FitSpec,
    deadline_monotonic: float,
    window_cancelled: threading.Event,
    analysis_cancelled: threading.Event,
) -> tuple[object, FitResultBatch, str, DataFigure, BoardFrame, object | None]:
    def cancelled() -> bool:
        return window_cancelled.is_set() or analysis_cancelled.is_set()

    def check_cancelled() -> None:
        if cancelled():
            raise CancelledError()

    if cancelled():
        raise CancelledError()
    if time.monotonic() >= deadline_monotonic:
        raise FitDeadlineExceeded("fit expired while waiting for compute")
    execution = execute(spec, cancelled, deadline_monotonic)
    if execution is None:
        raise TypeError("fit executor returned no opaque execution")
    result = result_of(execution)
    if not isinstance(result, FitResultBatch):
        raise TypeError("fit result capability returned another result type")
    check_cancelled()
    summary = fit_result_draft_summary(
        result,
        check_cancelled=check_cancelled,
    )
    overlays = figure.materialize_transient_fit_overlays(
        result,
        source_frame,
        result_identity=result_identity,
        check_cancelled=check_cancelled,
    )
    return execution, result, summary, figure, source_frame, overlays


def _export_typed_png(
    frame: BoardFrame,
    state: DataFigureDisplayState,
    destination: Path,
    revision: int,
    cancelled: threading.Event,
    commit_lock: threading.Lock,
) -> tuple[int, Path]:
    encoded = encode_data_figure_front_png(
        frame,
        state,
        check_cancelled=lambda: _require_not_cancelled(cancelled),
    )

    def write_staged(path: Path) -> None:
        _require_not_cancelled(cancelled)
        path.write_bytes(encoded)
        _require_not_cancelled(cancelled)

    result = stage_and_replace_export(
        Path(destination),
        write_staged=write_staged,
        cancelled=cancelled,
        commit_lock=commit_lock,
    )
    return revision, result


def _export_encoded_png(
    payload: bytes,
    destination: Path,
    revision: int,
    cancelled: threading.Event,
    commit_lock: threading.Lock,
) -> tuple[int, Path]:
    if not isinstance(payload, bytes):
        raise TypeError("encoded export requires owned immutable bytes")

    def write_staged(path: Path) -> None:
        _require_not_cancelled(cancelled)
        path.write_bytes(payload)
        _require_not_cancelled(cancelled)

    result = stage_and_replace_export(
        Path(destination),
        write_staged=write_staged,
        cancelled=cancelled,
        commit_lock=commit_lock,
    )
    return revision, result


__all__ = []
