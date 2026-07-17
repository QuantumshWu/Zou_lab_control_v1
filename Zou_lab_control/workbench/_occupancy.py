"""Read-only Workbench for one exact committed occupancy/camera cell."""

from __future__ import annotations

from concurrent.futures import CancelledError
import threading

from zlc_data import Selection
from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_neutral_atom.readout.occupancy_reference import OccupancyArtifactRef
from zlc_storage import positive_integer

from ._frozen_raster import FrozenRasterWindow, open_frozen_raster_window


_DEFAULT_OCCUPANCY_CELL_GUI_MEMORY_LIMIT_BYTES = 512 << 20


def _require_not_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise CancelledError()


def _render_occupancy_cell(
    loader,
    reference: OccupancyArtifactRef,
    selection: Selection | None,
    memory_limit_bytes: int,
    cancelled: threading.Event,
) -> EncodedRasterDocument:
    if cancelled.is_set():
        raise CancelledError()
    view, retained_upper_bound = loader(
        reference,
        selection,
        memory_limit_bytes=memory_limit_bytes,
    )
    if cancelled.is_set():
        raise CancelledError()
    from zlc_frontend.occupancy_render import render_occupancy_cell

    result = render_occupancy_cell(
        view,
        memory_limit_bytes=memory_limit_bytes,
        source_retained_upper_bound_bytes=retained_upper_bound,
        checkpoint=lambda: _require_not_cancelled(cancelled),
    )
    if cancelled.is_set():
        raise CancelledError()
    return result


def open_occupancy_cell_workbench(
    cell_loader,
    reference: OccupancyArtifactRef,
    *,
    selection: Selection | None = None,
    memory_limit_bytes: int = _DEFAULT_OCCUPANCY_CELL_GUI_MEMORY_LIMIT_BYTES,
) -> FrozenRasterWindow:
    """Load and display one exact same-shot site map on the shared raster lane."""

    if not callable(cell_loader):
        raise TypeError("cell_loader must be callable")
    if not isinstance(reference, OccupancyArtifactRef):
        raise TypeError("reference must be OccupancyArtifactRef")
    if selection is not None and not isinstance(selection, Selection):
        raise TypeError("selection must be Selection or None")
    limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
    return open_frozen_raster_window(
        lambda cancelled: _render_occupancy_cell(
            cell_loader,
            reference,
            selection,
            limit,
            cancelled,
        ),
        window_title="Occupancy Cell",
        mode_text="EXACT OCCUPANCY CELL · SAME-SHOT FRAME · DISPLAY ONLY",
        loading_summary=f"Resolving {reference.target_ref}…",
        object_prefix="occupancyCell",
        subject="OCCUPANCY CELL",
        memory_limit_bytes=limit,
    )


__all__ = ["open_occupancy_cell_workbench"]
