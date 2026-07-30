"""Direct filesystem I/O for current typed Figure archives.

The frontend owns the archive value and canonical payload codec.  This module
alone owns path normalization, the NPZ envelope, atomic replacement, and the
path-bearing load receipt used by desktop composition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import uuid

import numpy as np

from zlc_frontend import (
    DataFigure,
    FigureArchive,
)
from zlc_frontend.figure_archive import (
    FIGURE_ARCHIVE_SCHEMA,
    decode_figure_archive_payload,
    encode_figure_archive_payload,
)
from zlc_frontend.figure_archive import FigureDisplayState
from zlc_frontend.plot_panel import FigureIntent
from zlc_storage import flush_directory


_NPZ_FIELDS = ("payload", "schema")


@dataclass(frozen=True, slots=True)
class LoadedFigureArchive:
    """One archive path and the exact frontend value decoded from it."""

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


def save_figure_archive(
    figure: DataFigure,
    path: str | os.PathLike[str],
    *,
    figure_intent: FigureIntent,
    size_name: str,
    display: FigureDisplayState,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Atomically replace one exact current Figure archive."""

    target = _target_path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(f"figure archive directory does not exist: {target.parent}")
    payload = encode_figure_archive_payload(
        figure,
        figure_intent=figure_intent,
        size_name=size_name,
        display=display,
        metadata=metadata,
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
        flush_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def load_figure_archive(
    path: str | os.PathLike[str],
) -> LoadedFigureArchive:
    """Load the exact two-field NPZ envelope with pickle disabled."""

    target = _target_path(path)
    with np.load(target, allow_pickle=False) as envelope:
        if tuple(sorted(envelope.files)) != _NPZ_FIELDS:
            raise ValueError(
                "figure archive NPZ must contain exactly ['payload', 'schema']"
            )
        schema = envelope["schema"]
        encoded_payload = envelope["payload"]
        if schema.dtype != np.dtype(np.uint8) or schema.ndim != 1:
            raise ValueError("figure archive schema must be a uint8 vector")
        if encoded_payload.dtype != np.dtype(np.uint8) or encoded_payload.ndim != 1:
            raise ValueError("figure archive payload must be a uint8 vector")
        if schema.tobytes(order="C") != FIGURE_ARCHIVE_SCHEMA.encode("ascii"):
            raise ValueError("unsupported figure archive schema")
        payload = encoded_payload.tobytes(order="C")
    return LoadedFigureArchive(
        target,
        decode_figure_archive_payload(payload),
    )


__all__ = [
    "LoadedFigureArchive",
    "load_figure_archive",
    "save_figure_archive",
]
