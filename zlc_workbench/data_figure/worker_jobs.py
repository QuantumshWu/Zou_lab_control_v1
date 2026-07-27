"""DataFigure worker scheduling, Fit lifecycle, and atomic export jobs.

All Figure classification, renderer choice, payload construction, visual defaults,
and front validation belong to :mod:`zlc_frontend.data_figure_render`.  This leaf
owns only Workbench execution concerns: cancellation events, the capacity-one Fit
lane, application-supplied Fit capabilities, and committing encoded bytes to disk.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import threading
import time

from zlc_data import (
    AxisId,
    FitDeadlineExceeded,
    FitSpec,
    Selection,
)
from zlc_frontend import (
    BoardFrame,
    DataFigure,
    FitAuthoringOption,
    HistogramBinProjection,
    histogram_fit_transform,
    validate_fit_authoring_options,
)
from zlc_frontend.data_figure_presentation import (
    DataFigureDisplayState,
    fit_result_draft_summary,
)
from zlc_frontend.data_figure_render import encode_data_figure_front_png
from zlc_frontend.figure import AxisViewRole
from zlc_frontend.plot_layout import PanelSurfaceGeometry
from zlc_workbench.data_figure.fit_draft import FitDraftAuthority, FitDraftResult
from zlc_workbench.window_runtime import stage_and_replace_export


_FIT_WORK_EXECUTOR = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="zlc-data-figure-fit",
)


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
    fit_axis_ids: tuple[AxisId, ...],
    axis_roles: tuple[tuple[AxisId, AxisViewRole], ...],
    selection: Selection | None,
    histogram_projection: HistogramBinProjection | None,
    allow_prepared_transform: bool = False,
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
    schemas = {option.spec.input_schema_fingerprint for option in options}
    models = tuple(option.spec.model_id for option in options)
    if len(schemas) != 1 or len(models) != len(set(models)):
        raise ValueError("Fit options require one source schema and unique models")
    if any(option.spec.fit_axis_ids != fit_axis_ids for option in options):
        raise ValueError("Fit option axes differ from the exact displayed axes")
    validated_allow_prepared_transform = bool(allow_prepared_transform)
    if histogram_projection is not None:
        required_transform = histogram_fit_transform(
            figure,
            histogram_projection,
        )
        if any(
            option.spec.committed_transform != required_transform
            for option in options
        ):
            raise ValueError(
                "Histogram Fit options differ from the exact visible sample/bin authority"
            )
        # This is not permission for an injected arbitrary transform: the
        # exact committed transform was reconstructed above from this Figure's
        # named view plus the painted bin projection and matched byte-for-byte.
        validated_allow_prepared_transform = True
    return validate_fit_authoring_options(
        options,
        fit_axis_ids=fit_axis_ids,
        axis_roles=axis_roles,
        selection=selection,
        allow_prepared_transform=validated_allow_prepared_transform,
    )


def _execute_fit_draft(
    authority: FitDraftAuthority,
    spec: FitSpec,
    deadline_monotonic: float,
    window_cancelled: threading.Event,
    analysis_cancelled: threading.Event,
) -> tuple[FitDraftResult, str]:
    def cancelled() -> bool:
        return window_cancelled.is_set() or analysis_cancelled.is_set()

    def check_cancelled() -> None:
        if cancelled():
            raise CancelledError()

    if cancelled():
        raise CancelledError()
    if time.monotonic() >= deadline_monotonic:
        raise FitDeadlineExceeded("fit expired while waiting for its worker lane")
    draft = authority.execute(spec, cancelled, deadline_monotonic)
    try:
        return draft, fit_result_draft_summary(
            draft.result,
            check_cancelled=check_cancelled,
        )
    except BaseException:
        # The authority already installed this live draft.  Presentation failure
        # must release that exact generation so later submissions cannot deadlock.
        authority.discard(draft)
        raise


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
