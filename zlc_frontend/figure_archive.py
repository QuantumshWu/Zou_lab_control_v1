"""Canonical payload for one exact, current ``DataFigure`` archive.

The payload stores source dataset revisions, the headless figure document,
exact fit results, and the complete authored presentation contract.  It deliberately does
not persist evaluated x/y/image projections: those are reproducibly derived
presentation data, whereas the source ``DataBlock`` values and validity are
the facts that must survive reopening.

This module owns canonical payload bytes only.  The desktop repository is the
sole owner of the NPZ filesystem envelope.  There is no numeric version,
compatibility reader, or upgrade path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias

import numpy as np

from zlc_data import (
    DataBlock,
    OwnedSnapshot,
    dataset_revision_ref_from_tree,
    dataset_revision_ref_to_tree,
    dataset_schema_from_tree,
    dataset_schema_to_tree,
    decode_fit_result_batch,
    encode_fit_result_batch,
    selection_from_tree,
    selection_to_tree,
    validity_from_tree,
    validity_to_tree,
)
from zlc_storage import (
    decode,
    encode,
    exact_mapping,
    sha256_digest,
    sha256_text,
)

from .curve_display import CurveDisplayState
from .data_figure import DataFigure
from .display_range import RelimMode, validated_display_range
from .figure import (
    DATASET_VIEW_INTENTS,
    DatasetId,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
)
from .figure.codec import decode_figure_document, encode_figure_document
from .histogram_display import (
    FacetedHistogramDisplayState,
    HistogramCountScale,
    HistogramDisplayState,
    histogram_cell_thresholds_from_tree,
    histogram_cell_thresholds_to_tree,
)
from .image_display import ImageColormap, ImageDisplayState
from .meter_display import MeterDisplayState
from .panel_size import panel_size_cells


FIGURE_ARCHIVE_SCHEMA = "zlc_frontend.DataFigureArchive"
_DATASET_FIELDS = frozenset(
    {"dataset_id", "ref", "schema", "validity", "values"}
)
_FIT_RESULT_FIELDS = frozenset({"layer_id", "payload"})

FigureDisplayState: TypeAlias = (
    CurveDisplayState
    | ImageDisplayState
    | HistogramDisplayState
    | FacetedHistogramDisplayState
    | MeterDisplayState
)

_FACETED_HISTOGRAM_DISPLAY_KIND = "FACETED_HISTOGRAM"


def _display_intent(state: FigureDisplayState) -> ViewIntent:
    if isinstance(state, CurveDisplayState):
        return ViewIntent.CURVE
    if isinstance(state, ImageDisplayState):
        return ViewIntent.IMAGE
    if isinstance(state, (HistogramDisplayState, FacetedHistogramDisplayState)):
        return ViewIntent.HISTOGRAM
    if isinstance(state, MeterDisplayState):
        return ViewIntent.METER
    raise TypeError("display has another Figure display-state type")


@dataclass(frozen=True, slots=True)
class FigurePresentationContract:
    """Everything needed to reproduce one archived interactive surface.

    This is deliberately frontend vocabulary rather than TaskConsole metadata.
    The already-frozen :class:`DataFigure` owns data/view/fit facts; this value
    owns only reproducible presentation intent.  The current screen DPR is a
    Qt-surface fact and is deliberately not archived.
    """

    intent: ViewIntent
    faceted: bool
    rolling_trace: bool
    rolling_distribution: bool
    title: str
    value_label: str
    size_name: str
    display: FigureDisplayState

    @classmethod
    def from_plot_panel(
        cls,
        contract,
        display: FigureDisplayState,
    ) -> "FigurePresentationContract":
        """Freeze the archive projection of frontend's live panel contract."""

        from .plot_panel import PlotPanelContract

        if not isinstance(contract, PlotPanelContract):
            raise TypeError("contract must be PlotPanelContract")
        intent = contract.intent
        if intent is None:
            raise ValueError("SiteMap does not have a DataFigure archive")
        return cls(
            intent=intent,
            faceted=contract.faceted,
            rolling_trace=contract.kind == "monitor",
            rolling_distribution=contract.rolling_distribution,
            title=contract.title,
            value_label=contract.value_label,
            size_name=contract.size_name,
            display=display,
        )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.intent, ViewIntent)
            or self.intent not in DATASET_VIEW_INTENTS
        ):
            raise ValueError("figure presentation requires a dataset ViewIntent")
        for name in ("faceted", "rolling_trace", "rolling_distribution"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")
        if self.rolling_trace and (
            self.intent is not ViewIntent.CURVE or self.faceted
        ):
            raise ValueError("rolling trace requires one ordinary CURVE surface")
        if self.rolling_distribution and not self.rolling_trace:
            raise ValueError("rolling distribution requires rolling_trace")
        if _display_intent(self.display) is not self.intent:
            raise ValueError("presentation display and intent disagree")
        if isinstance(self.display, FacetedHistogramDisplayState) and not self.faceted:
            raise ValueError("per-cell histogram display requires a faceted surface")
        if not isinstance(self.title, str) or not isinstance(self.value_label, str):
            raise TypeError("figure presentation labels must be strings")
        panel_size_cells(self.size_name)
        object.__setattr__(self, "size_name", str(self.size_name))

    def validate_figure(self, figure: DataFigure) -> None:
        if not isinstance(figure, DataFigure):
            raise TypeError("figure presentation requires DataFigure")
        if (
            len(figure.document.layers) != 1
            or len(figure.evaluated.layers) != 1
            or len(figure.evaluated.inputs) != 1
        ):
            raise ValueError(
                "archived interactive presentation requires one layer and input"
            )
        if figure.document.layers[0].view.intent is not self.intent:
            raise ValueError("figure presentation intent differs from its document")
        cell_count = len(figure.evaluated.layers[0].cells)
        if self.faceted != (cell_count > 1):
            raise ValueError(
                "figure presentation faceting differs from evaluated panel topology"
            )


def _presentation_to_tree(
    presentation: FigurePresentationContract,
) -> dict[str, Any]:
    if not isinstance(presentation, FigurePresentationContract):
        raise TypeError("presentation must be FigurePresentationContract")
    return {
        "intent": presentation.intent.value,
        "faceted": presentation.faceted,
        "rolling_trace": presentation.rolling_trace,
        "rolling_distribution": presentation.rolling_distribution,
        "title": presentation.title,
        "value_label": presentation.value_label,
        "size_name": presentation.size_name,
        "display": _display_state_to_tree(presentation.display),
    }


def _presentation_from_tree(tree: Any) -> FigurePresentationContract:
    data = exact_mapping(
        tree,
        {
            "intent",
            "faceted",
            "rolling_trace",
            "rolling_distribution",
            "title",
            "value_label",
            "size_name",
            "display",
        },
        "figure presentation",
        discriminator=None,
    )
    display = _display_state_from_tree(data["display"])
    if display is None:
        raise ValueError("figure presentation requires authored display state")
    return FigurePresentationContract(
        intent=ViewIntent(data["intent"]),
        faceted=data["faceted"],
        rolling_trace=data["rolling_trace"],
        rolling_distribution=data["rolling_distribution"],
        title=data["title"],
        value_label=data["value_label"],
        size_name=data["size_name"],
        display=display,
    )


def _optional_range_to_tree(value) -> list[float] | None:
    return None if value is None else [value[0], value[1]]


def _optional_range_from_tree(value: Any, field: str):
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{field} must be a two-item list or null")
    return validated_display_range(tuple(value), field)


def _display_state_to_tree(
    state: FigureDisplayState | None,
) -> dict[str, Any] | None:
    if state is None:
        return None
    if isinstance(state, CurveDisplayState):
        return {
            "kind": ViewIntent.CURVE.value,
            "revision": state.revision,
            "relim_mode": state.relim_mode.value,
            "x_view": _optional_range_to_tree(state.x_view),
            "fixed_y_limits": _optional_range_to_tree(state.fixed_y_limits),
        }
    if isinstance(state, ImageDisplayState):
        return {
            "kind": ViewIntent.IMAGE.value,
            "revision": state.revision,
            "relim_mode": state.relim_mode.value,
            "colormap": state.colormap.value,
            "x_view": _optional_range_to_tree(state.x_view),
            "y_view": _optional_range_to_tree(state.y_view),
            "fixed_color_limits": _optional_range_to_tree(
                state.fixed_color_limits
            ),
        }
    if isinstance(state, FacetedHistogramDisplayState):
        return {
            "kind": _FACETED_HISTOGRAM_DISPLAY_KIND,
            "display": _display_state_to_tree(state.display),
            "cell_thresholds": histogram_cell_thresholds_to_tree(
                state.cell_thresholds
            ),
        }
    if isinstance(state, HistogramDisplayState):
        return {
            "kind": ViewIntent.HISTOGRAM.value,
            "revision": state.revision,
            "relim_mode": state.relim_mode.value,
            "count_scale": state.count_scale.value,
            "bin_count": state.bin_count,
            "x_view": _optional_range_to_tree(state.x_view),
            "fixed_count_limits": _optional_range_to_tree(
                state.fixed_count_limits
            ),
            "thresholds": list(state.thresholds),
        }
    if isinstance(state, MeterDisplayState):
        return {
            "kind": ViewIntent.METER.value,
            "revision": state.revision,
            "panel_index": state.panel_index,
            "expected_selection": None
            if state.expected_selection is None
            else selection_to_tree(state.expected_selection),
        }
    raise TypeError(
        "display must be CurveDisplayState, ImageDisplayState, "
        "HistogramDisplayState, FacetedHistogramDisplayState, "
        "MeterDisplayState, or None"
    )


def _display_state_from_tree(tree: Any) -> FigureDisplayState | None:
    if tree is None:
        return None
    if not isinstance(tree, dict) or not isinstance(tree.get("kind"), str):
        raise ValueError("figure archive display must be a tagged map or null")
    kind = tree["kind"]
    if kind == ViewIntent.CURVE.value:
        data = exact_mapping(
            tree,
            {
                "kind",
                "revision",
                "relim_mode",
                "x_view",
                "fixed_y_limits",
            },
            ViewIntent.CURVE.value,
            discriminator="kind",
        )
        return CurveDisplayState(
            revision=data["revision"],
            relim_mode=RelimMode(data["relim_mode"]),
            x_view=_optional_range_from_tree(data["x_view"], "curve x_view"),
            fixed_y_limits=_optional_range_from_tree(
                data["fixed_y_limits"],
                "curve fixed_y_limits",
            ),
        )
    if kind == ViewIntent.IMAGE.value:
        data = exact_mapping(
            tree,
            {
                "kind",
                "revision",
                "relim_mode",
                "colormap",
                "x_view",
                "y_view",
                "fixed_color_limits",
            },
            ViewIntent.IMAGE.value,
            discriminator="kind",
        )
        return ImageDisplayState(
            revision=data["revision"],
            relim_mode=RelimMode(data["relim_mode"]),
            colormap=ImageColormap(data["colormap"]),
            x_view=_optional_range_from_tree(data["x_view"], "image x_view"),
            y_view=_optional_range_from_tree(data["y_view"], "image y_view"),
            fixed_color_limits=_optional_range_from_tree(
                data["fixed_color_limits"],
                "image fixed_color_limits",
            ),
        )
    if kind == _FACETED_HISTOGRAM_DISPLAY_KIND:
        data = exact_mapping(
            tree,
            {"kind", "display", "cell_thresholds"},
            _FACETED_HISTOGRAM_DISPLAY_KIND,
            discriminator="kind",
        )
        display = _display_state_from_tree(data["display"])
        if not isinstance(display, HistogramDisplayState):
            raise ValueError(
                "faceted histogram archive requires a histogram display"
            )
        return FacetedHistogramDisplayState(
            display,
            histogram_cell_thresholds_from_tree(data["cell_thresholds"]),
        )
    if kind == ViewIntent.HISTOGRAM.value:
        data = exact_mapping(
            tree,
            {
                "kind",
                "revision",
                "relim_mode",
                "count_scale",
                "bin_count",
                "x_view",
                "fixed_count_limits",
                "thresholds",
            },
            ViewIntent.HISTOGRAM.value,
            discriminator="kind",
        )
        if not isinstance(data["thresholds"], list):
            raise ValueError("histogram thresholds must be a list")
        return HistogramDisplayState(
            revision=data["revision"],
            relim_mode=RelimMode(data["relim_mode"]),
            count_scale=HistogramCountScale(data["count_scale"]),
            bin_count=data["bin_count"],
            x_view=_optional_range_from_tree(
                data["x_view"],
                "histogram x_view",
            ),
            fixed_count_limits=_optional_range_from_tree(
                data["fixed_count_limits"],
                "histogram fixed_count_limits",
            ),
            thresholds=tuple(data["thresholds"]),
        )
    if kind == ViewIntent.METER.value:
        data = exact_mapping(
            tree,
            {"kind", "revision", "panel_index", "expected_selection"},
            ViewIntent.METER.value,
            discriminator="kind",
        )
        selection = data["expected_selection"]
        return MeterDisplayState(
            panel_index=data["panel_index"],
            expected_selection=(
                None if selection is None else selection_from_tree(selection)
            ),
            revision=data["revision"],
        )
    raise ValueError(f"unknown figure archive display kind {kind!r}")


def _metadata_for_tree(
    metadata: Mapping[str, object] | None,
) -> dict[str, object]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("figure archive metadata must be a mapping or None")
    if any(not isinstance(key, str) for key in metadata):
        raise TypeError("figure archive metadata keys must be strings")
    return dict(metadata)


def _archive_tree(
    figure: DataFigure,
    *,
    presentation: FigurePresentationContract,
    metadata: Mapping[str, object] | None,
) -> dict[str, Any]:
    if not isinstance(figure, DataFigure):
        raise TypeError("figure must be DataFigure")
    if not isinstance(presentation, FigurePresentationContract):
        raise TypeError("presentation must be FigurePresentationContract")
    presentation.validate_figure(figure)
    datasets = []
    for entry in sorted(
        figure.datasets.entries,
        key=lambda item: item.dataset_id.value,
    ):
        snapshot = entry.snapshot
        datasets.append(
            {
                "dataset_id": entry.dataset_id.value,
                "ref": dataset_revision_ref_to_tree(snapshot.ref),
                "schema": dataset_schema_to_tree(snapshot.block.schema),
                "validity": validity_to_tree(snapshot.block.validity),
                "values": snapshot.block.values,
            }
        )
    return {
        "schema": FIGURE_ARCHIVE_SCHEMA,
        "document": encode_figure_document(figure.document),
        "datasets": datasets,
        "fit_results": [
            {
                "layer_id": layer_id,
                "payload": encode_fit_result_batch(result),
            }
            for layer_id, result in sorted(figure.fit_results.items())
        ],
        "presentation": _presentation_to_tree(presentation),
        "metadata": _metadata_for_tree(metadata),
    }


def _decode_archive_payload(
    payload: bytes,
) -> tuple[DataFigure, FigurePresentationContract, dict[str, object]]:
    tree = decode(payload)
    data = exact_mapping(
        tree,
        {
            "schema",
            "document",
            "datasets",
            "fit_results",
            "presentation",
            "metadata",
        },
        FIGURE_ARCHIVE_SCHEMA,
    )
    if not isinstance(data["document"], bytes):
        raise ValueError("figure archive document must be canonical bytes")
    document = decode_figure_document(data["document"])

    raw_datasets = data["datasets"]
    if not isinstance(raw_datasets, list):
        raise ValueError("figure archive datasets must be a list")
    entries = []
    for raw in raw_datasets:
        item = exact_mapping(
            raw,
            _DATASET_FIELDS,
            "figure archive dataset",
            discriminator=None,
        )
        if not isinstance(item["values"], np.ndarray):
            raise ValueError("figure archive dataset values must be an ndarray")
        dataset_id = DatasetId(item["dataset_id"])
        ref = dataset_revision_ref_from_tree(item["ref"])
        schema = dataset_schema_from_tree(item["schema"])
        validity = validity_from_tree(item["validity"])
        block = DataBlock(
            ref.block_id,
            ref.revision,
            item["values"],
            validity,
            schema,
        )
        entries.append(
            ResolvedDataset(dataset_id, OwnedSnapshot(ref, block))
        )
    datasets = ResolvedDatasetMap(tuple(entries))

    raw_fit_results = data["fit_results"]
    if not isinstance(raw_fit_results, list):
        raise ValueError("figure archive fit_results must be a list")
    fit_results = {}
    for raw in raw_fit_results:
        item = exact_mapping(
            raw,
            _FIT_RESULT_FIELDS,
            "figure archive fit result",
            discriminator=None,
        )
        layer_id = item["layer_id"]
        result_payload = item["payload"]
        if not isinstance(layer_id, str) or not layer_id:
            raise ValueError("figure archive fit layer_id must be non-empty text")
        if layer_id in fit_results:
            raise ValueError("figure archive fit layer ids must be unique")
        if not isinstance(result_payload, bytes):
            raise ValueError("figure archive fit payload must be canonical bytes")
        fit_results[layer_id] = decode_fit_result_batch(result_payload)

    presentation = _presentation_from_tree(data["presentation"])
    metadata = data["metadata"]
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) for key in metadata
    ):
        raise ValueError("figure archive metadata must be a string-keyed map")
    figure = DataFigure(document, datasets, fit_results=fit_results)
    presentation.validate_figure(figure)

    # A canonical primitive payload can still encode a typed value in a
    # non-canonical field order/normal form.  Re-project through every owner and
    # require exact bytes so there is only one admitted current representation.
    rebuilt = encode(
        _archive_tree(
            figure,
            presentation=presentation,
            metadata=metadata,
        )
    )
    if rebuilt != payload:
        raise ValueError(
            "figure archive payload uses a non-canonical typed representation"
        )
    return figure, presentation, metadata


def _freeze_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_metadata_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_metadata_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class FigureArchive:
    """Decoded current archive value, independent of any repository path."""

    figure: DataFigure
    presentation: FigurePresentationContract
    metadata: Mapping[str, object]
    payload_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.figure, DataFigure):
            raise TypeError("figure must be DataFigure")
        if not isinstance(self.presentation, FigurePresentationContract):
            raise TypeError("presentation must be FigurePresentationContract")
        self.presentation.validate_figure(self.figure)
        if not isinstance(self.metadata, Mapping) or any(
            not isinstance(key, str) for key in self.metadata
        ):
            raise TypeError("metadata must be a string-keyed mapping")
        # Canonicalize caller-created instances as well as loader-created ones,
        # then recursively freeze containers exposed by this frozen DTO.
        normalized_metadata = decode(encode(dict(self.metadata)))
        object.__setattr__(
            self,
            "metadata",
            _freeze_metadata_value(normalized_metadata),
        )
        object.__setattr__(
            self,
            "payload_digest",
            sha256_text(self.payload_digest, "figure archive payload digest"),
        )


def encode_figure_archive_payload(
    figure: DataFigure,
    *,
    presentation: FigurePresentationContract,
    metadata: Mapping[str, object] | None = None,
) -> bytes:
    """Encode one exact current archive payload without doing filesystem I/O."""

    return encode(
        _archive_tree(
            figure,
            presentation=presentation,
            metadata=metadata,
        )
    )


def decode_figure_archive_payload(payload: bytes) -> FigureArchive:
    """Decode the sole current canonical payload without repository concerns."""

    if not isinstance(payload, bytes):
        raise TypeError("figure archive payload must be bytes")
    figure, presentation, metadata = _decode_archive_payload(payload)
    return FigureArchive(
        figure=figure,
        presentation=presentation,
        metadata=metadata,
        payload_digest=sha256_digest(payload),
    )


__all__ = [
    "FIGURE_ARCHIVE_SCHEMA",
    "FigureArchive",
    "FigureDisplayState",
    "FigurePresentationContract",
    "decode_figure_archive_payload",
    "encode_figure_archive_payload",
]
