"""Occupancy-owned loading jobs for its exact-cell navigator."""

from __future__ import annotations

from concurrent.futures import CancelledError
import threading

from zlc_frontend.figure_source import FigureSource
from zlc_frontend.plot_panel import FigureIntent
from zlc_frontend.site_map_view import SiteMapView
from zlc_neutral_atom.logic_nodes.readout.occupancy.cell import (
    OccupancyCellDomain,
    _occupancy_cell_coherence_identity,
)


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


def _load_cell_figure(
    loader,
    reference,
    address,
    navigation,
    cancelled,
):
    """Load one address and validate its generic Figure source before Qt sees it."""

    _cancel_point(cancelled)
    result = loader(
        reference,
        address,
        expected_navigation=navigation,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("cell loader must return (FigureIntent, FigureSource)")
    figure, source = result
    if not isinstance(figure, FigureIntent) or not isinstance(source, FigureSource):
        raise TypeError("cell loader returned another Figure contract")
    view = source.site_map
    navigation.resolve_address(address)
    if (
        not isinstance(view, SiteMapView)
        or view.coherence_identity
        != _occupancy_cell_coherence_identity(
            navigation.artifact_identity,
            address,
        )
    ):
        raise ValueError("cell loader returned a different exact address")
    _cancel_point(cancelled)
    return navigation.identity, address, figure, source
