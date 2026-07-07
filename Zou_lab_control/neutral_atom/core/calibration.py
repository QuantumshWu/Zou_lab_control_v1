"""Calibration record for the lightweight neutral-atom session.

A :class:`TrapCalibration` is the single readout contract: it owns the site
centers, the per-site thresholds, and HOW a raw image becomes one scalar per
site.  ``method='box'`` (the default) reduces a square ROI like the original
readout; ``method='psf'`` carries a per-site PSF weight and does matched-filter
extraction (the Rb87 qCMOS readout).  ``signals(image)`` is the single dispatch
point; ``detect(image)`` thresholds it.  Everything downstream (results,
operations, the readout subsystem) is unchanged across methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .analysis import AtomDetection, centers_array, grid_shape_tuple, nonnegative_int, roi_counts, threshold_array
from .bimodal import classify_threshold
from .psf import psf_signals

#: Readout-KIND for every readout method, the EXPLICIT dispatch table that decides how
#: ``signals()`` extracts a method's per-site scalar.  ``"box"`` reduces a square ROI;
#: ``"kernel"`` does matched-filter (PSF) extraction.  This is the single source for the
#: method->kind relation, so dispatch is by declared kind -- NOT by a fragile ``"psf" in m``
#: substring on the method NAME (a future ``matched_filter`` would miss the substring and be
#: read as box, every count silently wrong).  ``operations.ALL_READOUT_METHODS`` validates that
#: every offered method is registered here.  core owns this (it owns ``signals()`` dispatch) and
#: never imports operations, keeping the analysis->backend decoupling direction (AGENTS §2).
READOUT_KINDS = {"box": "box", "psf": "kernel", "uniform_psf": "kernel"}


def readout_kind(method: str) -> str:
    """The readout KIND (``"box"`` square-ROI or ``"kernel"`` matched-filter) for ``method`` --
    the explicit dispatch the readout uses instead of inferring it from the method NAME."""
    m = str(method).lower()
    try:
        return READOUT_KINDS[m]
    except KeyError:
        raise ValueError(
            f"readout method {m!r} has no registered readout kind -- add it to "
            "core.calibration.READOUT_KINDS.") from None


@dataclass(frozen=True)
class TrapCalibration:
    centers: np.ndarray
    thresholds: np.ndarray | float
    grid_shape: tuple[int, int] | None = None
    # roi_radius / reducer are the BOX readout's extraction geometry (square ROI half-width +
    # how it reduces).  They are meaningful ONLY for method='box'; a PSF (kernel) calibration
    # reads through psf_weights/psf_boxes and ignores them entirely, so __post_init__ sets both
    # to None there (no dead box-only state carried on a PSF calibration).
    roi_radius: int | None = 1
    reducer: str | None = "mean"
    method: str = "box"
    psf_weights: np.ndarray | None = None   # (N, h, w) normalized, method='psf'
    psf_boxes: np.ndarray | None = None      # (N, 4) int (x0, y0, w, h), method='psf'
    background: str = "none"                 # extraction background: 'none' | 'annulus'
    # OPTIONAL per-method readout data so ONE calibration supports several readout
    # methods and the downstream OccupancyProcessor picks which to use (cali once, read
    # many ways).  ``{method: {"thresholds": (N,), "psf_weights": (N,h,w)|None,
    # "psf_boxes": (N,4)|None}}`` for every method OTHER than this calibration's own
    # (which lives in the top-level fields).  None / empty = single-method (the default).
    by_method: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        centers = centers_array(self.centers)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "thresholds", threshold_array(self.thresholds, len(centers)))
        if self.grid_shape is not None:
            shape = grid_shape_tuple(self.grid_shape)
            if int(np.prod(shape)) != len(centers):
                raise ValueError("grid_shape product must match number of centers.")
            object.__setattr__(self, "grid_shape", shape)
        method = str(self.method).lower()
        if method not in READOUT_KINDS:                  # the ONE method registry (box / psf / uniform_psf)
            raise ValueError(f"method must be one of {tuple(READOUT_KINDS)}.")
        object.__setattr__(self, "method", method)
        # Normalization dispatches on the method's declared KIND (READOUT_KINDS), never on the
        # method NAME -- 'psf' and 'uniform_psf' are both kernel readouts and normalize the same.
        kind = readout_kind(method)
        # roi_radius / reducer are box-only extraction geometry: validate + keep them for a box
        # calibration, but a PSF (kernel) calibration reads through the kernels and ignores them,
        # so drop both to None there rather than carry meaningless box state (#historical-residue).
        if kind == "box":
            object.__setattr__(self, "roi_radius", nonnegative_int(self.roi_radius, "roi_radius"))
            object.__setattr__(self, "reducer", str(self.reducer))
        else:
            object.__setattr__(self, "roi_radius", None)
            object.__setattr__(self, "reducer", None)
        object.__setattr__(self, "background", str(self.background).lower())
        if kind == "kernel":
            if self.psf_weights is None or self.psf_boxes is None:
                raise ValueError(f"method={method!r} (kernel readout) requires psf_weights and psf_boxes.")
            weights = np.ascontiguousarray(self.psf_weights, dtype=float)
            boxes = np.ascontiguousarray(self.psf_boxes, dtype=int)
            if weights.ndim != 3 or len(weights) != len(centers):
                raise ValueError("psf_weights must be (N_sites, h, w).")
            if boxes.shape != (len(centers), 4):
                raise ValueError("psf_boxes must be (N_sites, 4).")
            object.__setattr__(self, "psf_weights", weights)
            object.__setattr__(self, "psf_boxes", boxes)
        else:
            object.__setattr__(self, "psf_weights", None if self.psf_weights is None else np.asarray(self.psf_weights, dtype=float))
            object.__setattr__(self, "psf_boxes", None if self.psf_boxes is None else np.asarray(self.psf_boxes, dtype=int))
        normalized: dict[str, Any] = {}
        for name, entry in dict(self.by_method or {}).items():
            entry = dict(entry or {})
            thr = entry.get("thresholds")
            normalized[str(name).lower()] = {
                "thresholds": None if thr is None else threshold_array(thr, len(centers)),
                "psf_weights": None if entry.get("psf_weights") is None else np.ascontiguousarray(entry["psf_weights"], dtype=float),
                "psf_boxes": None if entry.get("psf_boxes") is None else np.ascontiguousarray(entry["psf_boxes"], dtype=int),
                # The background model is part of the READOUT: a PSF method subtracts an
                # annulus, box subtracts nothing.  Carry it per method so signals(method=m)
                # reads on the SAME scale its thresholds were calibrated on -- otherwise a
                # psf threshold (annulus-subtracted) is compared to a never-subtracted
                # signal and lands off the distribution (every shot bright; the figure's
                # threshold line + fidelity vanish off-axis).
                "background": entry.get("background"),
            }
        object.__setattr__(self, "by_method", normalized or None)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def n_sites(self) -> int:
        return len(self.centers)

    def methods(self) -> tuple[str, ...]:
        """Every readout method this calibration can read with -- its own plus any extra
        carried in ``by_method`` (so a GUI can offer exactly the available choices)."""
        extra = tuple(sorted(self.by_method)) if self.by_method else ()
        return (self.method, *(m for m in extra if m != self.method))

    def _resolve_method(self, method) -> str:
        m = str(self.method if method in (None, "") else method).lower()
        # Whitelist against the methods this calibration can actually read with
        # (methods(), the single source) -- a misspelled method must fail loud
        # (like analysis.normalize_reducer / operations ALL_READOUT_METHODS),
        # not silently fall back to box because ``"psf" in m`` happened to miss.
        valid = self.methods()
        if m not in valid:
            raise ValueError(
                f"method {m!r} is not one of {', '.join(valid)} -- "
                "recalibrate including it or pick an available method.")
        return m

    def _kernels_for(self, method: str):
        """PSF (weights, boxes) for a psf-type ``method`` -- the top-level fields when it
        is this calibration's own method, else the ``by_method`` entry (raises if absent)."""
        if method == self.method:
            return self.psf_weights, self.psf_boxes
        entry = (self.by_method or {}).get(method)
        if entry is None or entry.get("psf_weights") is None or entry.get("psf_boxes") is None:
            raise ValueError(
                f"calibration carries no '{method}' PSF readout -- recalibrate including it.")
        return entry["psf_weights"], entry["psf_boxes"]

    def _background_for(self, method: str):
        """Background model for ``method`` -- this calibration's own (top-level), else the
        ``by_method`` entry's, falling back to the top-level.  signals(method=m) MUST read on
        the scale m's thresholds were calibrated on, so it uses m's OWN background, not the
        top-level method's (box subtracts nothing; a psf method subtracts an annulus)."""
        if method == self.method:
            return self.background
        entry = (self.by_method or {}).get(method) or {}
        return entry.get("background") if entry.get("background") is not None else self.background

    def readout_exposure(self, fallback: float | None = None) -> float | None:
        """The camera gate time these thresholds were LEARNT at (``threshold_exposure`` on the
        metadata), or ``fallback`` when none was recorded.  This is the ONE authoritative reader
        of that exposure-self-match invariant: every readout that must image at the calibration's
        exposure (``detect``, the live calibrate-task adoption, the temperature survival frames)
        goes through HERE instead of each reaching into ``metadata`` with its own defensive spelling
        -- a threshold is exposure-specific, so a missed/mistyped lookup re-floors occupancy /
        sticks survival at the readout false-positive rate (#issue-2 / #H3v-2).  Callers pass their
        OWN fallback (the camera exposure, or None to mean 'do not force a match')."""
        value = self.metadata.get("threshold_exposure")
        return float(value) if value else fallback

    def thresholds_for(self, method=None) -> np.ndarray:
        """Per-site thresholds for ``method`` (this calibration's own, or a ``by_method``
        entry); falls back to the top-level thresholds when the method has no own set."""
        m = self._resolve_method(method)
        if m == self.method:
            return self.thresholds
        entry = (self.by_method or {}).get(m)
        if entry is not None and entry.get("thresholds") is not None:
            return entry["thresholds"]
        return self.thresholds

    def signals(self, image, *, method=None) -> np.ndarray:
        """One scalar per site for ``image``.  ``method`` (None = this calibration's own)
        picks the readout: ``box`` square-ROI, ``psf``/``uniform_psf`` matched-filter --
        so one calibration can be read several ways (the processor chooses)."""

        # ROI fingerprint: the centers/PSF boxes are absolute camera pixels, so a
        # frame from a DIFFERENT camera ROI (same size but shifted) would silently
        # extract the WRONG pixels -- in-bounds, no error, every count wrong.  When
        # the calibration recorded the image shape it was built on, fail loud on a
        # mismatch instead (raise -> recalibrate) rather than corrupt results.
        expected = self.metadata.get("image_shape")
        if expected is not None:
            got = tuple(int(v) for v in np.shape(image)[:2])
            if got != tuple(int(v) for v in expected):
                raise ValueError(
                    f"image shape {got} does not match the calibration's {tuple(expected)} "
                    "(camera ROI changed since calibration?) -- recalibrate before reading out.")
        m = self._resolve_method(method)
        if readout_kind(m) == "kernel":
            weights, boxes = self._kernels_for(m)
            return psf_signals(image, weights, boxes, background=self._background_for(m) or "annulus")
        # Box (square-ROI) readout.  Its extraction geometry -- roi_radius / reducer -- lives ONLY
        # in the top-level fields and is meaningful ONLY for a box-PRIMARY calibration (a kernel
        # calibration null's them in __post_init__).  ``by_method`` carries thresholds / PSF
        # kernels / background, NEVER box geometry, so box is readable ONLY as this calibration's
        # OWN method.  ``_resolve_method`` already rejects a method absent from ``methods()``; this
        # guards the remaining trap -- a 'box' entry smuggled into a kernel-primary calibration's
        # ``by_method`` -- which would otherwise read with the None'd top-level geometry (wrong
        # pixels, no error).  Fail loud instead.
        if m != self.method:
            raise ValueError(
                f"box readout is only available as this calibration's own method (this "
                f"calibration is {self.method!r}); box extraction geometry (roi_radius/reducer) "
                "is not carried per-method -- recalibrate with method='box' to read this way.")
        return roi_counts(image, self.centers, radius=self.roi_radius, reducer=self.reducer)

    def detect(self, image, *, method=None) -> AtomDetection:
        """Classify occupancy by thresholding the per-site signals (atoms are bright).
        ``method`` selects the readout + its matching thresholds (None = this
        calibration's own)."""

        counts = np.asarray(self.signals(image, method=method), dtype=float)
        thresholds = self.thresholds_for(method)
        # Occupancy classification is the ONE primitive ``bimodal.classify_threshold`` -- shared with
        # the fidelity path -- not a re-inlined ``counts > thresholds`` here.  ``bright_above=True`` is
        # the ENFORCED physical invariant of this fluorescence readout: a loaded site scatters photons
        # and reads HIGHER, so occupancy is always "above threshold".  The training fit's per-site
        # ``bright_above`` is therefore NOT stored as calibration state -- a site fitting bright_above=
        # False means its two Gaussians are degenerate/mislabelled (a dead or contaminated site), a
        # data-quality flag the fidelity report surfaces, never a per-site readout polarity to honour.
        occupied = classify_threshold(counts, thresholds, bright_above=True)
        return AtomDetection(
            counts=counts,
            occupied=occupied,
            occupied_indices=np.flatnonzero(occupied).astype(int).tolist(),
            thresholds=thresholds,
        )

    def with_thresholds(self, thresholds, **metadata) -> "TrapCalibration":
        # dataclasses.replace re-runs __post_init__ (re-normalizes thresholds/by_method), so it is
        # the ONE faithful "copy with one field changed" -- no hand-re-spelling every field (#C2).
        return replace(self, thresholds=thresholds, metadata={**self.metadata, **metadata})

    def with_method_thresholds(self, per_method, **metadata) -> "TrapCalibration":
        """A copy with per-METHOD thresholds replaced.  ``per_method`` maps a method name to
        its ``(n_sites,)`` thresholds: the calibration's OWN method updates the top-level
        thresholds, every other method updates its ``by_method`` entry; a method absent from
        ``per_method`` keeps its current thresholds.  Used to write the reference-bracket
        per-site boundaries back so ``detect`` reads on the TRAINED threshold (not the otsu
        quick split) -- the Rb87 'use the true labels to set the boundary' step."""
        per_method = {str(k).lower(): np.asarray(v, dtype=float).reshape(-1)
                      for k, v in dict(per_method or {}).items()}
        top = per_method.get(self.method, self.thresholds)
        by_method = {}
        for name, entry in (self.by_method or {}).items():
            updated = dict(entry)
            if name in per_method:
                updated["thresholds"] = per_method[name]
            by_method[name] = updated
        # the by_method rebuild above is the real work; replace() carries every OTHER field faithfully (#C2).
        return replace(self, thresholds=top, by_method=by_method or None,
                       metadata={**self.metadata, **metadata})

    @staticmethod
    def _by_method_to_json(by_method) -> dict | None:
        """``by_method`` with numpy arrays -> plain lists (JSON-able)."""
        if not by_method:
            return None
        out = {}
        for name, entry in by_method.items():
            out[name] = {
                k: (None if v is None else np.asarray(v).tolist())
                for k, v in entry.items()
            }
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "centers": self.centers.tolist(),
            "thresholds": self.thresholds.tolist(),
            "grid_shape": None if self.grid_shape is None else list(self.grid_shape),
            "roi_radius": self.roi_radius,
            "reducer": self.reducer,
            "method": self.method,
            "psf_weights": None if self.psf_weights is None else self.psf_weights.tolist(),
            "psf_boxes": None if self.psf_boxes is None else self.psf_boxes.tolist(),
            "background": self.background,
            "by_method": self._by_method_to_json(self.by_method),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrapCalibration":
        return cls(
            payload["centers"],
            payload["thresholds"],
            grid_shape=None if payload.get("grid_shape") is None else tuple(payload["grid_shape"]),
            roi_radius=payload.get("roi_radius", 1),
            reducer=payload.get("reducer", "mean"),
            method=payload.get("method", "box"),
            psf_weights=payload.get("psf_weights"),
            psf_boxes=payload.get("psf_boxes"),
            background=payload.get("background", "none"),
            by_method=payload.get("by_method"),
            metadata=payload.get("metadata", {}),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".npz":
            np.savez(
                path,
                centers=self.centers,
                thresholds=self.thresholds,
                grid_shape=np.asarray([] if self.grid_shape is None else self.grid_shape),
                # roi_radius / reducer are box-only (None on a PSF calibration); store a sentinel
                # (-1 / "") that load() reads back as None so the npz stays allow_pickle=False.
                roi_radius=np.asarray(-1 if self.roi_radius is None else self.roi_radius),
                reducer=np.asarray("" if self.reducer is None else self.reducer),
                method=np.asarray(self.method),
                background=np.asarray(self.background),
                psf_weights=np.asarray([] if self.psf_weights is None else self.psf_weights),
                psf_boxes=np.asarray([] if self.psf_boxes is None else self.psf_boxes),
                by_method_json=np.asarray(json.dumps(self._by_method_to_json(self.by_method), ensure_ascii=False)),
                metadata_json=np.asarray(json.dumps(self.metadata, ensure_ascii=False)),
            )
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrapCalibration":
        path = Path(path)
        if path.suffix.lower() == ".npz":
            data = np.load(path, allow_pickle=False)
            grid = data["grid_shape"]
            metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data.files else {}
            by_method = json.loads(str(data["by_method_json"].item())) if "by_method_json" in data.files else None
            psf_weights = data["psf_weights"] if "psf_weights" in data.files else None
            psf_boxes = data["psf_boxes"] if "psf_boxes" in data.files else None
            # roi_radius / reducer sentinels (-1 / "") restore to None (a PSF calibration carried
            # no box geometry); __post_init__ also drops them for any non-box method either way.
            roi_radius_raw = int(data["roi_radius"].item())
            reducer_raw = str(data["reducer"].item())
            return cls(
                data["centers"],
                data["thresholds"],
                grid_shape=None if grid.size == 0 else tuple(int(v) for v in grid),
                roi_radius=None if roi_radius_raw < 0 else roi_radius_raw,
                reducer=None if reducer_raw == "" else reducer_raw,
                method=str(data["method"].item()) if "method" in data.files else "box",
                psf_weights=None if psf_weights is None or psf_weights.size == 0 else psf_weights,
                psf_boxes=None if psf_boxes is None or psf_boxes.size == 0 else psf_boxes,
                background=str(data["background"].item()) if "background" in data.files else "none",
                by_method=by_method,
                metadata=metadata,
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


__all__ = ["TrapCalibration", "READOUT_KINDS", "readout_kind"]
