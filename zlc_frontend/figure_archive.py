"""Current-only typed ``DataFigure`` NPZ persistence.

The archive stores source dataset revisions, the headless figure document,
exact fit results, and optional authored display state.  It deliberately does
not persist evaluated x/y/image projections: those are reproducibly derived
presentation data, whereas the source ``DataBlock`` values and validity are
the facts that must survive reopening.

The NPZ envelope has exactly two non-object arrays, ``schema`` and ``payload``.
``payload`` is one canonical byte string assembled exclusively through the
owning figure/data/fit codecs.  There is no numeric version, compatibility
reader, or upgrade path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias
import uuid

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
from .figure import DatasetId, ResolvedDataset, ResolvedDatasetMap, ViewIntent
from .figure.codec import decode_figure_document, encode_figure_document
from .histogram_display import HistogramCountScale, HistogramDisplayState
from .image_display import ImageColormap, ImageDisplayState
from .meter_display import MeterDisplayState


FIGURE_ARCHIVE_SCHEMA = "zlc_frontend.DataFigureArchive"
_NPZ_FIELDS = ("payload", "schema")
_DATASET_FIELDS = {"dataset_id", "ref", "schema", "validity", "values"}
_FIT_RESULT_FIELDS = {"layer_id", "payload"}

FigureDisplayState: TypeAlias = (
    CurveDisplayState
    | ImageDisplayState
    | HistogramDisplayState
    | MeterDisplayState
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
        "HistogramDisplayState, MeterDisplayState, or None"
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
    display: FigureDisplayState | None,
    metadata: Mapping[str, object] | None,
) -> dict[str, Any]:
    if not isinstance(figure, DataFigure):
        raise TypeError("figure must be DataFigure")
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
        "display": _display_state_to_tree(display),
        "metadata": _metadata_for_tree(metadata),
    }


def _decode_archive_payload(
    payload: bytes,
) -> tuple[DataFigure, FigureDisplayState | None, dict[str, object]]:
    tree = decode(payload)
    data = exact_mapping(
        tree,
        {
            "schema",
            "document",
            "datasets",
            "fit_results",
            "display",
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

    display = _display_state_from_tree(data["display"])
    metadata = data["metadata"]
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) for key in metadata
    ):
        raise ValueError("figure archive metadata must be a string-keyed map")
    figure = DataFigure(document, datasets, fit_results=fit_results)

    # A canonical primitive payload can still encode a typed value in a
    # non-canonical field order/normal form.  Re-project through every owner and
    # require exact bytes so there is only one admitted current representation.
    rebuilt = encode(
        _archive_tree(
            figure,
            display=display,
            metadata=metadata,
        )
    )
    if rebuilt != payload:
        raise ValueError(
            "figure archive payload uses a non-canonical typed representation"
        )
    return figure, display, metadata


def _freeze_metadata_value(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_metadata_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_metadata_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class LoadedFigureArchive:
    """One reopened current archive plus its display-only replay state."""

    figure: DataFigure
    display: FigureDisplayState | None
    metadata: Mapping[str, object]
    path: Path
    payload_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.figure, DataFigure):
            raise TypeError("figure must be DataFigure")
        _display_state_to_tree(self.display)
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
        object.__setattr__(self, "path", Path(self.path).resolve())
        object.__setattr__(
            self,
            "payload_digest",
            sha256_text(self.payload_digest, "figure archive payload digest"),
        )


def _target_path(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    if not target.name:
        raise ValueError("figure archive path must name a file")
    if not target.suffix:
        target = target.with_suffix(".npz")
    elif target.suffix.lower() != ".npz":
        raise ValueError("figure archive path must use the .npz suffix")
    return target


def save_figure_archive(
    figure: DataFigure,
    path: str | os.PathLike[str],
    *,
    display: FigureDisplayState | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically replace ``path`` with the sole current archive envelope."""

    target = _target_path(path)
    payload = encode(
        _archive_tree(
            figure,
            display=display,
            metadata=metadata,
        )
    )
    schema_array = np.frombuffer(
        FIGURE_ARCHIVE_SCHEMA.encode("ascii"),
        dtype=np.uint8,
    )
    payload_array = np.frombuffer(payload, dtype=np.uint8)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                schema=schema_array,
                payload=payload_array,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def load_figure_archive(
    path: str | os.PathLike[str],
) -> LoadedFigureArchive:
    """Load only the exact two-field current archive with pickle disabled."""

    target = _target_path(path)
    with np.load(target, allow_pickle=False) as archive:
        if tuple(sorted(archive.files)) != _NPZ_FIELDS:
            raise ValueError(
                "figure archive NPZ must contain exactly ['payload', 'schema']"
            )
        schema = archive["schema"]
        encoded_payload = archive["payload"]
        if schema.dtype != np.dtype(np.uint8) or schema.ndim != 1:
            raise ValueError("figure archive schema must be a uint8 vector")
        if encoded_payload.dtype != np.dtype(np.uint8) or encoded_payload.ndim != 1:
            raise ValueError("figure archive payload must be a uint8 vector")
        if schema.tobytes(order="C") != FIGURE_ARCHIVE_SCHEMA.encode("ascii"):
            raise ValueError("unsupported figure archive schema")
        payload = encoded_payload.tobytes(order="C")

    figure, display, metadata = _decode_archive_payload(payload)
    return LoadedFigureArchive(
        figure=figure,
        display=display,
        metadata=metadata,
        path=target,
        payload_digest=sha256_digest(payload),
    )


__all__ = [
    "FIGURE_ARCHIVE_SCHEMA",
    "FigureDisplayState",
    "LoadedFigureArchive",
    "load_figure_archive",
    "save_figure_archive",
]
