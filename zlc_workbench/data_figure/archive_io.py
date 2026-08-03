"""Small, reloadable archives for one frozen ``zlc_plot`` surface.

The archive is deliberately not another Figure model.  ``zlc_data`` owns the
snapshot schema and validity codecs, while ``zlc_plot`` owns the PlotSpec codec.
This module only places those owner-projected facts beside the original ndarray
in an atomic NPZ envelope.  Fit overlays are not encoded here: a fitted visual
is exported through the same ``RasterPlotHost`` that painted it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np

from zlc_data import (
    BlockId,
    DataBlock,
    DatasetRevision,
    OwnedSnapshot,
    StreamGenerationId,
)
from zlc_data.codec import (
    dataset_schema_from_tree,
    dataset_schema_to_tree,
    validity_from_tree,
    validity_to_tree,
)
from zlc_plot import DEFAULTS, PlotSpec, plot_spec_from_tree, plot_spec_to_tree
from zlc_storage import atomic_write_file


_FORMAT = "zlc.figure"
_VERSION = 1
_BASE_FIELDS = frozenset({"manifest", "values"})
_MASK_FIELD = "validity_mask"


def _json_mapping(value: Mapping[str, object] | None, field: str) -> Mapping[str, object]:
    source = {} if value is None else dict(value)
    if any(not isinstance(key, str) for key in source):
        raise TypeError(f"{field} keys must be strings")
    try:
        encoded = json.dumps(source, allow_nan=False, separators=(",", ":"))
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field} must contain ordinary JSON values") from error
    if not isinstance(decoded, dict):  # pragma: no cover - source is a dict
        raise TypeError(f"{field} must be a mapping")
    return MappingProxyType(decoded)


@dataclass(frozen=True, slots=True)
class FigureArchive:
    """The complete reloadable state of one standalone plot surface."""

    snapshot: OwnedSnapshot
    spec: PlotSpec
    size: str
    parameters: Mapping[str, object] = MappingProxyType({})
    metadata: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")
        # Round-tripping through the owner validates the closed PlotSpec union
        # without duplicating its concrete type list in Workbench.
        projected = plot_spec_to_tree(self.spec)
        if plot_spec_from_tree(projected) != self.spec:
            raise ValueError("PlotSpec owner codec did not round-trip the spec")
        object.__setattr__(self, "size", DEFAULTS.layout.validate_preset(self.size))
        object.__setattr__(
            self,
            "parameters",
            _json_mapping(self.parameters, "parameters"),
        )
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "metadata"))


@dataclass(frozen=True, slots=True)
class LoadedFigureArchive:
    """A decoded current archive and its resolved source path."""

    path: Path
    archive: FigureArchive

    def __post_init__(self) -> None:
        if not isinstance(self.archive, FigureArchive):
            raise TypeError("archive must be FigureArchive")
        object.__setattr__(self, "path", Path(self.path).resolve())


def _target_path(path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    if not target.name:
        raise ValueError("figure archive path must name a file")
    if not target.suffix:
        target = target.with_suffix(".npz")
    elif target.suffix.lower() != ".npz":
        raise ValueError("figure archive path must use the .npz suffix")
    return target


def _validity_manifest(validity: object) -> tuple[dict[str, object], np.ndarray | None]:
    tree = dict(validity_to_tree(validity))
    mask = tree.pop("mask", None)
    if mask is None:
        return tree, None
    if not isinstance(mask, np.ndarray):
        raise TypeError("zlc_data validity codec returned a non-array mask")
    return tree, mask


def _manifest(archive: FigureArchive) -> tuple[dict[str, object], np.ndarray | None]:
    snapshot = archive.snapshot
    validity, mask = _validity_manifest(snapshot.block.validity)
    return (
        {
            "format": _FORMAT,
            "version": _VERSION,
            "source": {
                "block_id": snapshot.block.block_id.value,
                "stream_generation": snapshot.ref.stream_generation.value,
                "revision": snapshot.block.revision.value,
            },
            "dataset_schema": dataset_schema_to_tree(snapshot.block.schema),
            "validity": validity,
            "plot_spec": plot_spec_to_tree(archive.spec),
            "size": archive.size,
            "parameters": dict(archive.parameters),
            "metadata": dict(archive.metadata),
        },
        mask,
    )


def save_figure_archive(
    archive: FigureArchive,
    path: str | os.PathLike[str],
) -> Path:
    """Atomically replace one current Figure archive."""

    if not isinstance(archive, FigureArchive):
        raise TypeError("archive must be FigureArchive")
    target = _target_path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(f"figure archive directory does not exist: {target.parent}")
    manifest, mask = _manifest(archive)
    manifest_text = json.dumps(
        manifest,
        allow_nan=False,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    fields: dict[str, np.ndarray] = {
        "manifest": np.asarray(manifest_text),
        "values": archive.snapshot.block.values,
    }
    if mask is not None:
        fields[_MASK_FIELD] = mask
    return atomic_write_file(
        target,
        lambda stream: np.savez_compressed(stream, **fields),
    )


def _decode_manifest(value: np.ndarray) -> dict[str, object]:
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError("figure archive manifest must be one Unicode scalar")
    try:
        manifest = json.loads(str(value.item()))
    except (TypeError, ValueError) as error:
        raise ValueError("figure archive manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("figure archive manifest must be a mapping")
    expected = {
        "format",
        "version",
        "source",
        "dataset_schema",
        "validity",
        "plot_spec",
        "size",
        "parameters",
        "metadata",
    }
    if set(manifest) != expected:
        raise ValueError("figure archive manifest has an unexpected field set")
    if manifest["format"] != _FORMAT or manifest["version"] != _VERSION:
        raise ValueError("unsupported figure archive format")
    return manifest


def load_figure_archive(path: str | os.PathLike[str]) -> LoadedFigureArchive:
    """Load only the current archive; no historical reader or fallback exists."""

    target = _target_path(path)
    with np.load(target, allow_pickle=False) as envelope:
        fields = frozenset(envelope.files)
        if not _BASE_FIELDS.issubset(fields) or fields - (_BASE_FIELDS | {_MASK_FIELD}):
            raise ValueError("figure archive NPZ has an unexpected field set")
        manifest = _decode_manifest(envelope["manifest"])
        schema = dataset_schema_from_tree(manifest["dataset_schema"])
        source = manifest["source"]
        if not isinstance(source, dict) or set(source) != {
            "block_id",
            "stream_generation",
            "revision",
        }:
            raise ValueError("figure archive source identity is invalid")
        validity_tree = manifest["validity"]
        if not isinstance(validity_tree, dict):
            raise ValueError("figure archive validity must be a mapping")
        validity_tree = dict(validity_tree)
        requires_mask = validity_tree.get("kind") in {
            "cell",
            "component",
            "dataset_component",
        }
        if requires_mask != (_MASK_FIELD in fields):
            raise ValueError("figure archive validity mask field is inconsistent")
        if requires_mask:
            validity_tree["mask"] = envelope[_MASK_FIELD]
        validity = validity_from_tree(validity_tree)
        revision = DatasetRevision(source["revision"])
        block_id = BlockId(source["block_id"])
        block = DataBlock(
            block_id=block_id,
            revision=revision,
            values=envelope["values"],
            validity=validity,
            schema=schema,
        )
        snapshot = OwnedSnapshot(
            block.ref(StreamGenerationId(source["stream_generation"])),
            block,
        )
        archive = FigureArchive(
            snapshot=snapshot,
            spec=plot_spec_from_tree(manifest["plot_spec"]),
            size=manifest["size"],
            parameters=manifest["parameters"],
            metadata=manifest["metadata"],
        )
    return LoadedFigureArchive(target.resolve(), archive)


__all__ = [
    "FigureArchive",
    "LoadedFigureArchive",
    "load_figure_archive",
    "save_figure_archive",
]
