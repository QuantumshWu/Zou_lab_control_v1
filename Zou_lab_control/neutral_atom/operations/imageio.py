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
import hashlib
import json
from pathlib import Path
import re
from typing import Any, ClassVar, Mapping

import numpy as np

from ..core.analysis import positive_int
from ..core.calibration import FrameContract
from .._serialization import (
    require_array,
    require_exact_fields,
    require_int,
    require_object,
    require_string,
)


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


@dataclass(frozen=True)
class RunManifest:
    """Versioned grouping/geometry authority for one saved frame run."""

    schema: ClassVar[str] = "Zou_lab_control.neutral_atom.FrameRun"
    version: ClassVar[int] = 1

    prefix: str
    shots_per_group: int
    short_shot: int
    ref_shots: tuple[int, ...]
    n_groups: int
    frames_subdir: str
    frame_contract: FrameContract

    def __post_init__(self) -> None:
        prefix = str(self.prefix)
        if not prefix or any(ch in prefix for ch in "/\\"):
            raise ValueError("frame-run prefix must be non-empty and contain no path separator.")
        object.__setattr__(self, "prefix", prefix)
        spg = positive_int(self.shots_per_group, "shots_per_group")
        short = positive_int(self.short_shot, "short_shot")
        refs = tuple(int(v) for v in self.ref_shots)
        if short > spg or any(v < 1 or v > spg for v in refs):
            raise ValueError(f"short_shot/ref_shots must be within 1..{spg}.")
        if not refs or len(set(refs)) != len(refs) or short in refs:
            raise ValueError("ref_shots must be non-empty, unique, and exclude short_shot.")
        object.__setattr__(self, "shots_per_group", spg)
        object.__setattr__(self, "short_shot", short)
        object.__setattr__(self, "ref_shots", refs)
        object.__setattr__(self, "n_groups", positive_int(self.n_groups, "n_groups"))
        subdir = str(self.frames_subdir)
        subpath = Path(subdir) if subdir else Path(".")
        if subpath.is_absolute() or ".." in subpath.parts:
            raise ValueError("frames_subdir must stay inside the manifest folder.")
        object.__setattr__(self, "frames_subdir", "" if subpath == Path(".") else subpath.as_posix())
        if not isinstance(self.frame_contract, FrameContract):
            object.__setattr__(self, "frame_contract", FrameContract.from_dict(
                require_object(self.frame_contract, what="FrameRun.frame_contract")))

    @property
    def n_frames(self) -> int:
        return self.n_groups * self.shots_per_group

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(include_fingerprint=False), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "version": self.version,
            "prefix": self.prefix,
            "shots_per_group": self.shots_per_group,
            "short_shot": self.short_shot,
            "ref_shots": list(self.ref_shots),
            "n_groups": self.n_groups,
            "frames_subdir": self.frames_subdir,
            "frame_contract": self.frame_contract.to_dict(),
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        fields = {
            "schema", "version", "prefix", "shots_per_group", "short_shot",
            "ref_shots", "n_groups", "frames_subdir", "frame_contract", "fingerprint",
        }
        require_exact_fields(payload, fields, what="frame-run manifest")
        schema = require_string(payload["schema"], what="FrameRun.schema")
        version = require_int(payload["version"], what="FrameRun.version")
        if schema != cls.schema or version != cls.version:
            raise ValueError(
                f"unsupported frame-run manifest {schema!r} version {version}; "
                f"expected {cls.schema!r} version {cls.version}.")
        manifest = cls(
            prefix=require_string(payload["prefix"], what="FrameRun.prefix"),
            shots_per_group=require_int(payload["shots_per_group"], what="FrameRun.shots_per_group"),
            short_shot=require_int(payload["short_shot"], what="FrameRun.short_shot"),
            ref_shots=tuple(require_int(v, what="FrameRun.ref_shots item") for v in
                            require_array(payload["ref_shots"], what="FrameRun.ref_shots")),
            n_groups=require_int(payload["n_groups"], what="FrameRun.n_groups"),
            frames_subdir=require_string(payload["frames_subdir"], what="FrameRun.frames_subdir"),
            frame_contract=FrameContract.from_dict(require_object(
                payload["frame_contract"], what="FrameRun.frame_contract")),
        )
        if require_string(payload["fingerprint"], what="FrameRun.fingerprint") != manifest.fingerprint:
            raise ValueError("frame-run manifest fingerprint mismatch.")
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        payload = require_object(json.loads(Path(path).read_text(encoding="utf-8")),
                                 what="frame-run manifest file")
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path


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
    frame_contract: FrameContract
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
            yield self._load_checked(p)

    def reference_frames(self):
        """Iterate ``(group, ref_shot)`` reference frames in group-major order."""
        for g in self.group_paths:
            for shot in self.ref_shots:
                yield self._load_checked(g[int(shot) - 1])

    def _load_checked(self, path: Path) -> np.ndarray:
        frame = load_frame(path)
        if tuple(frame.shape) != self.image_shape:
            raise ValueError(
                f"frame {path.name} has shape {tuple(frame.shape)}, expected the run's "
                f"declared shape {self.image_shape}.")
        return frame

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
    frame_contract: FrameContract | None = None,
) -> RunIndex:
    """Index ``PREFIX<number>`` raw frames in ``folder`` into ``shots_per_group`` groups.

    Numbering must be exactly ``1..N`` with no gaps, and ``N`` must contain
    whole groups.  Missing or partial shots are data corruption, not a reason to
    shift every later frame into the wrong loading.  ``max_groups`` deliberately
    caps a valid complete run after those checks.
    """

    folder = Path(folder).expanduser()
    files = frame_files(folder, prefix)
    if not files:
        raise FileNotFoundError(f"no raw frames matching {prefix}<number> in {folder}")
    spg = positive_int(shots_per_group, "shots_per_group")
    short_shot = positive_int(short_shot, "short_shot")
    refs = tuple(int(s) for s in ref_shots)
    if short_shot > spg or any(s < 1 or s > spg for s in refs):
        raise ValueError(f"short_shot/ref_shots must be within 1..{spg}.")
    if not refs or len(set(refs)) != len(refs) or short_shot in refs:
        raise ValueError("ref_shots must be non-empty, unique, and exclude short_shot.")
    numbers = sorted(files)
    expected = list(range(1, numbers[-1] + 1))
    if numbers != expected:
        missing = sorted(set(expected) - set(numbers))
        raise ValueError(
            f"raw frame numbering must be contiguous from 1; found {numbers[0]}..{numbers[-1]} "
            f"with missing frame numbers {missing}.")
    if len(numbers) < spg:
        raise ValueError(f"need at least {spg} frames for one group; found {len(numbers)}.")
    remainder = len(numbers) % spg
    if remainder:
        raise ValueError(
            f"raw run has {len(numbers)} frames for groups of {spg}; the final {remainder} "
            "frame(s) form an incomplete loading and cannot be indexed safely.")
    n_complete = len(numbers) // spg
    if max_groups is not None:
        n_complete = min(n_complete, positive_int(max_groups, "max_groups"))
    used = numbers[: n_complete * spg]
    group_paths = [[files[used[g * spg + i]] for i in range(spg)] for g in range(n_complete)]
    image_shape = tuple(load_frame(group_paths[0][0]).shape)
    contract = frame_contract or FrameContract(image_shape=image_shape)
    if contract.image_shape != image_shape:
        raise ValueError(
            f"frame-run manifest image_shape {contract.image_shape} != first frame {image_shape}.")
    return RunIndex(
        folder=folder, prefix=str(prefix), group_paths=group_paths, image_shape=image_shape,
        frame_contract=contract,
        shots_per_group=spg, short_shot=short_shot, ref_shots=refs,
    )


def index_manifest(folder: str | Path) -> RunIndex:
    """Load the required manifest and verify its exact frame population."""

    root = Path(folder).expanduser()
    manifest = RunManifest.load(root / "run_schema.json")
    data_dir = root / manifest.frames_subdir if manifest.frames_subdir else root
    run = index_run(
        data_dir, manifest.prefix,
        shots_per_group=manifest.shots_per_group,
        short_shot=manifest.short_shot,
        ref_shots=manifest.ref_shots,
        frame_contract=manifest.frame_contract,
    )
    if run.n_groups != manifest.n_groups:
        raise ValueError(
            f"frame-run manifest declares {manifest.n_groups} groups/{manifest.n_frames} frames, "
            f"but the folder contains {run.n_groups} complete groups.")
    return run


__all__ = [
    "RunIndex", "RunManifest", "SHOT_INDEX_MIN", "frame_files", "index_manifest",
    "index_run", "load_frame", "save_frame",
]
