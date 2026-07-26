"""Occupancy-owned loading and raster jobs for its exact-cell viewer."""

from __future__ import annotations

from concurrent.futures import CancelledError
import threading

from zlc_frontend.site_map_render import compose_site_map_front
from zlc_frontend.site_map_view import SiteMapView
from zlc_neutral_atom.logic_nodes.readout.occupancy.cell import OccupancyCellDomain


_PANEL_ID = "sites"
_BOARD_ID = "occupancy-cell"


def _cancel_point(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise CancelledError()


def _load_navigation(loader, reference, cancelled):
    _cancel_point(cancelled)
    result = loader(reference)
    if not isinstance(result, OccupancyCellDomain):
        raise TypeError("navigation loader must return OccupancyCellDomain")
    if result.artifact_identity != reference.target_ref:
        raise ValueError("occupancy navigation names a different artifact")
    _cancel_point(cancelled)
    return result


def _build_front(
    view,
    display,
    color_limits,
    previous_relim,
    cell_revision,
    sequence,
    cancelled,
):
    if not isinstance(view, SiteMapView):
        raise TypeError("cell loader must return SiteMapView")
    _cancel_point(cancelled)
    frame, _effective_limits = compose_site_map_front(
        view,
        display,
        panel_id=_PANEL_ID,
        board_id=_BOARD_ID,
        sequence=sequence,
        selection_revision=cell_revision,
        current_color_limits=color_limits,
        previous_relim_mode=previous_relim,
    )
    _cancel_point(cancelled)
    return frame


def _cell_job(
    loader,
    reference,
    selection,
    navigation,
    loaded_view,
    display,
    color_limits,
    previous_relim,
    cell_revision,
    sequence,
    cancelled,
):
    _cancel_point(cancelled)
    if loaded_view is None:
        loaded_view = loader(
            reference,
            selection,
            expected_navigation=navigation,
        )
    repeat, _storage, logical = navigation.resolve_selection(selection)
    expected_selection = navigation.selection_for_indices(repeat, logical)
    if (
        not isinstance(loaded_view, SiteMapView)
        or loaded_view.cell_selection != expected_selection
    ):
        raise ValueError("cell loader returned a different exact selection")
    frame = _build_front(
        loaded_view,
        display,
        color_limits,
        previous_relim,
        cell_revision,
        sequence,
        cancelled,
    )
    return (
        navigation.identity,
        selection,
        cell_revision,
        display.revision,
        loaded_view,
        frame,
        display.relim_mode,
    )
