"""Headless saved-fit grid layout, identity, and display projection."""

from __future__ import annotations

from concurrent.futures import CancelledError
import math
import threading

from zlc_data import Selection, dataset_revision_ref_to_tree, selection_to_tree
from zlc_frontend import (
    FigurePanelRegion,
    ImageDisplayState,
    RadialGaussianImageFitPanel,
)
from zlc_frontend.image_display import evaluated_image_data_range
from zlc_frontend.display_range import RelimMode, deadband_display_range
from zlc_storage import canonical_digest, positive_integer

_BOARD_ID = "saved-fit-grid"

_FIT_GRID_JOIN_SCHEMA_DIGEST = canonical_digest(
    {
        "schema": "zlc_frontend.SavedFitGridJoin",
        "fields": (
            "artifact_identity",
            "source_inputs",
            "logical_panel_selection",
            "fit_storage_index",
        ),
    }
)


def _require_not_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise CancelledError()

def _fit_panel_id(panel: RadialGaussianImageFitPanel) -> str:
    if not isinstance(panel, RadialGaussianImageFitPanel):
        raise TypeError("saved-fit grid panel has the wrong type")
    selection = (
        None if panel.selection is None else selection_to_tree(panel.selection)
    )
    identity = canonical_digest(
        {
            "selection": selection,
            "fit_storage_index": panel.fit_storage_index,
        }
    )
    return f"fit-cell-{identity[:20]}"

def _grid_columns(count: int) -> int:
    count = positive_integer(count, "saved-fit grid panel count")
    return min(6, max(1, math.ceil(math.sqrt(count))))

def _validated_regions(
    model: FitGridModel,
    regions: tuple[FigurePanelRegion, ...],
) -> tuple[FigurePanelRegion, ...]:
    """Validate the still-consumed generic 1D saved-fit hit map."""

    prepared = tuple(regions)
    if not prepared or any(
        not isinstance(region, FigurePanelRegion) for region in prepared
    ):
        raise TypeError("saved-fit generic renderer must return panel regions")
    if len(prepared) > 36:
        raise ValueError("saved-fit grid page exceeded 36 display panels")
    if len({region.key for region in prepared}) != len(prepared):
        raise ValueError("saved-fit grid panel keys must be unique")
    for region in prepared:
        if region.fit_storage_index != model.storage_index_or_none(
            region.fit_selection
        ):
            raise ValueError(
                "saved-fit panel hit map disagrees with the exact batch layout"
            )
    return prepared

def _image_data_range(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    cancelled: threading.Event,
) -> tuple[float, float] | None:
    """Pool exact valid ranges without concatenating cell-sized arrays."""

    def images():
        for panel in panels:
            _require_not_cancelled(cancelled)
            yield panel.image

    return evaluated_image_data_range(images())

def _shared_color_limits(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    display: ImageDisplayState,
    current_color_limits: tuple[float, float] | None,
    previous_relim_mode: RelimMode | None,
    cancelled: threading.Event,
) -> tuple[float, float]:
    data_range = _image_data_range(panels, cancelled)
    if data_range is None:
        if display.relim_mode is RelimMode.FIXED:
            assert display.fixed_color_limits is not None
            return display.fixed_color_limits
        return (0.0, 1.0) if current_color_limits is None else current_color_limits
    return deadband_display_range(
        display.relim_mode,
        current_color_limits,
        data_range[0],
        data_range[1],
        fixed_range=display.fixed_color_limits,
        force=(
            previous_relim_mode is None
            or previous_relim_mode is not display.relim_mode
        ),
    )

def _fit_grid_join_identity(
    panels: tuple[RadialGaussianImageFitPanel, ...],
    panel_ids: tuple[str, ...],
) -> tuple[str, tuple[object, ...], str]:
    """Freeze the exact logical cells represented by one board frame."""

    if not panels or len(panels) != len(panel_ids):
        raise ValueError("saved-fit join requires one id per logical panel")
    if len(set(panel_ids)) != len(panel_ids):
        raise ValueError("saved-fit join panel ids must be unique")
    input_by_id = {}
    for panel in panels:
        incumbent = input_by_id.setdefault(
            panel.evaluated_input.dataset_id,
            panel.evaluated_input,
        )
        if incumbent != panel.evaluated_input:
            raise ValueError("saved-fit grid panel inputs disagree by dataset id")
    inputs = tuple(input_by_id.values())
    artifact_identities = {
        panel.fit_overlay.artifact_identity for panel in panels
    }
    if len(artifact_identities) != 1:
        raise ValueError("saved-fit grid panels belong to different artifacts")
    artifact_identity = next(iter(artifact_identities))
    digest = canonical_digest(
        {
            "schema": "zlc_frontend.SavedFitGridJoin",
            "artifact_identity": artifact_identity,
            "inputs": tuple(
                {
                    "dataset_id": item.dataset_id.value,
                    "ref": dataset_revision_ref_to_tree(item.ref),
                }
                for item in inputs
            ),
            "panels": tuple(
                {
                    "panel_id": panel_id,
                    "selection": (
                        None
                        if panel.selection is None
                        else selection_to_tree(panel.selection)
                    ),
                    "fit_storage_index": panel.fit_storage_index,
                }
                for panel_id, panel in zip(panel_ids, panels, strict=True)
            ),
        }
    )
    return artifact_identity, inputs, digest
