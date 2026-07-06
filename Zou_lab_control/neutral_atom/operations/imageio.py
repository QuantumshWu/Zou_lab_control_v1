"""Raw camera-frame folder I/O for the file-based readout workflow.

This mirrors how the REAL experiment stores data (see references rb87_readout):
the camera/DAQ writes one raw frame per shot to a folder as ``PREFIX<number>``
files, grouped ``shots_per_group`` frames per atom loading.  The analysis points
at that folder, indexes the frames into groups, and reads them back -- the EXACT
same code whether a real qCMOS wrote the files or the virtual backend did.  Only
the data SOURCE (who wrote the files) differs between virtual and real; see
``devices.virtual.write_virtual_run`` for the virtual writer.

Pure file I/O + numpy (no device backend, no session): the analysis layer stays
backend-agnostic, enforced by tests/test_virtual_equals_real_contract.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from ..core.analysis import positive_int


_IMAGE_SUFFIXES = (".npy", ".tif", ".tiff", ".png")


def load_frame(path: str | Path, dtype=np.float64) -> np.ndarray:
    """Load one raw frame (``.npy``/``.tif``/``.tiff``/``.png``) as a 2D array."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix in (".tif", ".tiff", ".png"):
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - depends on optional Pillow
            raise RuntimeError(
                f"reading {suffix} frames needs Pillow (pip install pillow); "
                "or write/read frames as .npy."
            ) from exc
        with Image.open(path) as im:
            arr = np.array(im)
    else:
        raise ValueError(f"unsupported frame suffix {suffix!r}; expected one of {_IMAGE_SUFFIXES}.")
    arr = np.asarray(arr, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"frame {path.name} must be 2D, got shape {arr.shape}.")
    return arr


def save_frame(path: str | Path, frame: np.ndarray) -> Path:
    """Write one raw frame to disk (``.npy`` by default; ``.tif``/``.png`` via Pillow)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    arr = np.asarray(frame)
    if suffix == ".npy":
        np.save(path, arr)
    elif suffix in (".tif", ".tiff", ".png"):
        try:
            from PIL import Image
        except Exception as exc:  # pragma: no cover - depends on optional Pillow
            raise RuntimeError(f"writing {suffix} frames needs Pillow; use .npy otherwise.") from exc
        Image.fromarray(arr).save(path)
    else:
        raise ValueError(f"unsupported frame suffix {suffix!r}; expected one of {_IMAGE_SUFFIXES}.")
    return path


def _frame_number(path: Path, prefix: str) -> int | None:
    """Parse the integer ``N`` from a ``PREFIX<N>.<suffix>`` filename, else ``None``.

    Rejects decorated names (e.g. an ``PREFIX1_ave_4`` average) so only raw
    per-shot frames are indexed."""

    if path.suffix.lower() not in _IMAGE_SUFFIXES or not path.name.startswith(prefix):
        return None
    stem = path.name[len(prefix):path.name.rfind(path.suffix)]
    return int(stem) if re.fullmatch(r"\d+", stem) else None


def frame_files(folder: str | Path, prefix: str) -> dict[int, Path]:
    """Map ``{frame_number: path}`` for every ``PREFIX<number>`` raw frame in ``folder``.

    A frame's DATA is its ``.npy``; a same-numbered ``.png``/``.tif`` beside it is a VISUAL
    companion (the cali save path writes ``img<n>.npy`` data + an ``img<n>.png`` picture so the
    operator can eyeball it).  When both exist for one number, the ``.npy`` WINS -- the picture
    companion must never shadow the round-trip data (re-reading the png as a frame would fail: it
    is RGBA, not a 2D count map).  A run with only image-suffix frames (a real qCMOS that wrote
    ``.tif``, no companion) still reads those -- the rule only disambiguates a genuine collision."""

    folder = Path(folder).expanduser()
    if not folder.exists():
        raise FileNotFoundError(f"data folder does not exist: {folder}")
    out: dict[int, Path] = {}
    for p in sorted(folder.iterdir()):
        n = _frame_number(p, prefix) if p.is_file() else None
        if n is None:
            continue
        prev = out.get(n)
        if prev is not None and prev.suffix.lower() == ".npy" and p.suffix.lower() != ".npy":
            continue                                   # keep the .npy data, skip its picture companion
        out[n] = p
    return out


# The Rb87 4-shot reference-bracket layout -- ONE source for every reader/writer of a calibration
# run (this file's RunIndex + index_run, and the readout/fidelity entry points that default to it):
# ``shots_per_group`` frames per atom loading; the ``short_shot`` frame is the short readout under
# test; ``ref_shots`` are the high-SNR frames that vote the ground-truth label (#F3, single-source).
DEFAULT_SHOTS_PER_GROUP = 4
DEFAULT_SHORT_SHOT = 3
DEFAULT_REF_SHOTS = (1, 2, 4)
# Shot indices are 1-based: ``index_run`` coerces ``short_shot`` via ``positive_int`` (>= 1),
# so any form offering a shot index must bound it from HERE, never a re-typed 0/1.
SHOT_INDEX_MIN = 1


@dataclass
class RunIndex:
    """Lazy index of grouped raw frames in a folder (no full stack in memory).

    ``shots_per_group`` frames belong to one atom loading; within a group the
    ``short_shot`` frame is the short readout being characterized and ``ref_shots``
    are the high-SNR reference frames that vote the ground-truth label."""

    folder: Path
    prefix: str
    group_paths: list[list[Path]]
    image_shape: tuple[int, int]
    shots_per_group: int = DEFAULT_SHOTS_PER_GROUP
    short_shot: int = DEFAULT_SHORT_SHOT
    ref_shots: tuple[int, ...] = DEFAULT_REF_SHOTS

    @property
    def n_groups(self) -> int:
        return len(self.group_paths)

    def _shot_paths(self, shot_number: int) -> list[Path]:
        shot = int(shot_number)
        if shot < 1 or shot > self.shots_per_group:
            raise ValueError(f"shot_number={shot} outside 1..{self.shots_per_group}.")
        return [g[shot - 1] for g in self.group_paths]

    def short_frames(self):
        """Iterate the short-readout frame of every group (one per loading)."""
        for p in self._shot_paths(self.short_shot):
            yield load_frame(p)

    def reference_frames(self):
        """Iterate ``(group, ref_shot)`` reference frames in group-major order."""
        for g in self.group_paths:
            for shot in self.ref_shots:
                yield load_frame(g[int(shot) - 1])

    def template_frames(self):
        """Iterate all reference frames (used to build the all-sites template)."""
        return self.reference_frames()


def index_run(
    folder: str | Path,
    prefix: str,
    *,
    shots_per_group: int = DEFAULT_SHOTS_PER_GROUP,
    short_shot: int = DEFAULT_SHORT_SHOT,
    ref_shots=DEFAULT_REF_SHOTS,
    max_groups: int | None = None,
) -> RunIndex:
    """Index ``PREFIX<number>`` raw frames in ``folder`` into ``shots_per_group`` groups.

    Trailing frames that do not complete a group are ignored; ``max_groups`` caps
    the number of groups.  Identical for real and virtual data -- it only reads
    files."""

    folder = Path(folder).expanduser()
    files = frame_files(folder, prefix)
    if not files:
        raise FileNotFoundError(f"no raw frames matching {prefix}<number> in {folder}")
    spg = positive_int(shots_per_group, "shots_per_group")
    short_shot = positive_int(short_shot, "short_shot")
    refs = tuple(int(s) for s in ref_shots)
    if short_shot > spg or any(s < 1 or s > spg for s in refs):
        raise ValueError(f"short_shot/ref_shots must be within 1..{spg}.")
    numbers = sorted(files)
    n_complete = len(numbers) // spg
    if n_complete < 1:
        raise ValueError(f"need at least {spg} frames for one group; found {len(numbers)}.")
    if max_groups is not None:
        n_complete = min(n_complete, positive_int(max_groups, "max_groups"))
    used = numbers[: n_complete * spg]
    group_paths = [[files[used[g * spg + i]] for i in range(spg)] for g in range(n_complete)]
    image_shape = tuple(load_frame(group_paths[0][0]).shape)
    return RunIndex(
        folder=folder, prefix=str(prefix), group_paths=group_paths, image_shape=image_shape,
        shots_per_group=spg, short_shot=short_shot, ref_shots=refs,
    )


__all__ = ["RunIndex", "SHOT_INDEX_MIN", "frame_files", "index_run", "load_frame", "save_frame"]
