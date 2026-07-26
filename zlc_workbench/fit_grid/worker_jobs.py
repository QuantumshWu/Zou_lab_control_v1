"""Saved-fit grid load, reraster, and export worker jobs."""

from __future__ import annotations

from concurrent.futures import CancelledError
from pathlib import Path
import threading

from zlc_data import Selection
from zlc_frontend import (
    ImageDisplayState,
    RadialGaussianImageFitPanel,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.fit_grid_render import (
    FitGridRenderSession,
    encode_fit_image_grid,
    encode_loaded_fit_grid,
)
from zlc_frontend.plot_layout import PanelSurfaceGeometry
from zlc_neutral_atom.artifacts.fit_reference import FitResultArtifactRef
from zlc_workbench.window_runtime import stage_and_replace_export

def _require_not_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise CancelledError()

def _rerasterize_grid_view(
    render_session: FitGridRenderSession,
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    current_color_limits: tuple[float, float] | None,
    previous_relim_mode: RelimMode | None,
    layout_generation: int,
    revision: int,
    cancelled: threading.Event,
    surface_geometry: PanelSurfaceGeometry | None = None,
    surface_revision: int = 0,
):
    _require_not_cancelled(cancelled)
    if surface_geometry is not None:
        render_session.use_surface_geometry(surface_geometry)
    front = render_session.build_front(
        panels,
        display,
        current_color_limits=current_color_limits,
        previous_relim_mode=previous_relim_mode,
        layout_generation=layout_generation,
        sequence=revision,
        check_cancelled=lambda: _require_not_cancelled(cancelled),
    )
    return revision, surface_revision, panels, display, front

def _load_grid_view(
    render_session: FitGridRenderSession,
    view_loader,
    reference: FitResultArtifactRef,
    page_address: tuple[int, ...] | None,
    cell_selection: Selection | None,
    revision: int,
    return_model: bool,
    display: ImageDisplayState,
    current_color_limits: tuple[float, float] | None,
    previous_relim_mode: RelimMode | None,
    layout_generation: int,
    cancelled: threading.Event,
    surface_geometry: PanelSurfaceGeometry | None = None,
    surface_revision: int = 0,
):
    _require_not_cancelled(cancelled)
    if surface_geometry is not None:
        render_session.use_surface_geometry(surface_geometry)
    loaded = view_loader(
        reference,
        page_address=page_address,
        cell_selection=cell_selection,
    )
    (
        model,
        page,
        cell_summary,
        resolved_selection,
        summary,
        projection,
    ) = render_session.project_loaded(
        loaded,
        artifact_identity=reference.target_ref,
        page_address=page_address,
        cell_selection=cell_selection,
        display=display,
        current_color_limits=current_color_limits,
        previous_relim_mode=previous_relim_mode,
        layout_generation=layout_generation,
        sequence=revision,
        check_cancelled=lambda: _require_not_cancelled(cancelled),
    )
    model_identity = model.identity
    returned_model = model if return_model else None
    return (
        revision,
        surface_revision,
        returned_model,
        model_identity,
        page,
        cell_summary,
        resolved_selection,
        summary,
        projection,
    )

def _export_grid_view(
    view_loader,
    reference: FitResultArtifactRef,
    expected_model_identity: object,
    page_address: tuple[int, ...] | None,
    cell_selection: Selection | None,
    destination: Path,
    revision: int,
    cancelled: threading.Event,
    commit_lock: threading.Lock,
):
    _require_not_cancelled(cancelled)
    loaded = view_loader(
        reference,
        page_address=page_address,
        cell_selection=cell_selection,
    )
    target = Path(destination)
    image_format = target.suffix.lstrip(".") or "png"
    if not target.suffix:
        target = target.with_suffix(f".{image_format}")
    encoded = encode_loaded_fit_grid(
        loaded,
        artifact_identity=reference.target_ref,
        expected_model_identity=expected_model_identity,
        page_address=page_address,
        cell_selection=cell_selection,
        image_format=image_format,
        check_cancelled=lambda: _require_not_cancelled(cancelled),
    )

    def write_staged(temporary: Path) -> None:
        _require_not_cancelled(cancelled)
        temporary.write_bytes(encoded)
        _require_not_cancelled(cancelled)

    committed = stage_and_replace_export(
        target,
        write_staged=write_staged,
        cancelled=cancelled,
        commit_lock=commit_lock,
    )
    return revision, committed

def _export_typed_grid_view(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    color_limits: tuple[float, float],
    columns: int,
    expected_join_key_digest: str,
    destination: Path,
    revision: int,
    cancelled: threading.Event,
    commit_lock: threading.Lock,
):
    """Export one frozen typed front without reloading data or fit authority."""

    target = Path(destination)
    image_format = target.suffix.lstrip(".").lower() or "png"
    if not target.suffix:
        target = target.with_suffix(f".{image_format}")
    encoded = encode_fit_image_grid(
        panels,
        display,
        color_limits,
        columns=columns,
        expected_join_key_digest=expected_join_key_digest,
        image_format=image_format,
        check_cancelled=lambda: _require_not_cancelled(cancelled),
    )

    def write_staged(temporary: Path) -> None:
        _require_not_cancelled(cancelled)
        temporary.write_bytes(encoded)
        _require_not_cancelled(cancelled)

    committed = stage_and_replace_export(
        target,
        write_staged=write_staged,
        cancelled=cancelled,
        commit_lock=commit_lock,
    )
    return revision, committed
