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
import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np

from .._serialization import (
    require_array,
    require_exact_fields,
    require_int,
    require_number,
    require_object,
    require_string,
)
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


@dataclass(frozen=True)
class FrameContract:
    """Physical camera geometry/settings on which a calibration is valid.

    ``image_shape`` alone cannot distinguish two equal-sized ROIs at different
    sensor offsets.  The complete contract therefore carries the sensor window,
    exposure and camera identity as independent, fingerprinted facts.
    """

    schema: ClassVar[str] = "Zou_lab_control.neutral_atom.FrameContract"
    version: ClassVar[int] = 1

    image_shape: tuple[int, int]
    sensor_shape: tuple[int, int] | None = None
    roi: tuple[int, int, int, int] | None = None
    exposure_s: float | None = None
    camera_type: str | None = None
    camera_id: str | None = None
    readout_mode: str | None = None

    def __post_init__(self) -> None:
        image = grid_shape_tuple(self.image_shape, "frame contract image_shape")
        object.__setattr__(self, "image_shape", image)
        sensor = None if self.sensor_shape is None else grid_shape_tuple(
            self.sensor_shape, "frame contract sensor_shape")
        object.__setattr__(self, "sensor_shape", sensor)
        if self.roi is not None:
            roi = tuple(int(v) for v in self.roi)
            if len(roi) != 4:
                raise ValueError("frame contract roi must be (x, width, y, height).")
            x, width, y, height = roi
            if x < 0 or y < 0 or width < 1 or height < 1:
                raise ValueError("frame contract roi requires x/y >= 0 and positive width/height.")
            if (height, width) != image:
                raise ValueError(
                    f"frame contract roi size {(height, width)} != image_shape {image}.")
            if sensor is not None and (x + width > sensor[1] or y + height > sensor[0]):
                raise ValueError(f"frame contract roi {roi} lies outside sensor_shape {sensor}.")
            object.__setattr__(self, "roi", roi)
        elif sensor is not None and sensor != image:
            raise ValueError(
                "frame contract without an roi must have image_shape == sensor_shape.")
        if self.exposure_s is not None:
            exposure = float(self.exposure_s)
            if not np.isfinite(exposure) or exposure <= 0:
                raise ValueError("frame contract exposure_s must be finite and > 0.")
            object.__setattr__(self, "exposure_s", exposure)
        for name in ("camera_type", "camera_id", "readout_mode"):
            value = getattr(self, name)
            if value is not None:
                value = str(value).strip()
                if not value:
                    raise ValueError(f"frame contract {name} cannot be blank.")
                object.__setattr__(self, name, value)

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(include_fingerprint=False), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "version": self.version,
            "image_shape": list(self.image_shape),
            "sensor_shape": None if self.sensor_shape is None else list(self.sensor_shape),
            "roi": None if self.roi is None else list(self.roi),
            "exposure_s": self.exposure_s,
            "camera_type": self.camera_type,
            "camera_id": self.camera_id,
            "readout_mode": self.readout_mode,
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrameContract":
        fields = {
            "schema", "version", "image_shape", "sensor_shape", "roi",
            "exposure_s", "camera_type", "camera_id", "readout_mode", "fingerprint",
        }
        require_exact_fields(payload, fields, what="calibration frame contract")
        schema = require_string(payload["schema"], what="FrameContract.schema")
        version = require_int(payload["version"], what="FrameContract.version")
        if schema != cls.schema or version != cls.version:
            raise ValueError(
                f"unsupported frame contract {schema!r} version {version!r}; "
                f"expected {cls.schema!r} version {cls.version}.")
        image_shape = require_array(payload["image_shape"], what="FrameContract.image_shape")
        sensor_shape = (None if payload["sensor_shape"] is None else
                        require_array(payload["sensor_shape"], what="FrameContract.sensor_shape"))
        roi = (None if payload["roi"] is None else
               require_array(payload["roi"], what="FrameContract.roi"))
        exposure = (None if payload["exposure_s"] is None else
                    require_number(payload["exposure_s"], what="FrameContract.exposure_s"))
        strings = {
            name: (None if payload[name] is None else require_string(
                payload[name], what=f"FrameContract.{name}"))
            for name in ("camera_type", "camera_id", "readout_mode")
        }
        contract = cls(
            image_shape=tuple(image_shape), sensor_shape=None if sensor_shape is None else tuple(sensor_shape),
            roi=None if roi is None else tuple(roi), exposure_s=exposure, **strings,
        )
        if require_string(payload["fingerprint"], what="FrameContract.fingerprint") != contract.fingerprint:
            raise ValueError("calibration frame-contract fingerprint mismatch.")
        return contract

    @classmethod
    def from_camera(cls, camera, *, image_shape=None, exposure_s=None) -> "FrameContract":
        """Snapshot a camera without importing a concrete backend."""

        snap = camera.snapshot() if hasattr(camera, "snapshot") else {}
        shape = image_shape or getattr(camera, "frame_shape", None)
        if shape is None:
            raise ValueError("camera frame shape is unknown; acquire a frame before calibration.")
        sensor = getattr(camera, "sensor_shape", None)
        roi = getattr(camera, "roi", None)
        exposure = exposure_s if exposure_s is not None else snap.get(
            "exposure", getattr(camera, "exposure", None))
        identity = snap.get("device_index", snap.get("serial_number", snap.get("serial")))
        mode = snap.get("readout_speed", snap.get("readout_mode"))
        return cls(
            image_shape=tuple(int(v) for v in shape),
            sensor_shape=None if sensor is None else tuple(int(v) for v in sensor),
            roi=None if roi is None else tuple(int(v) for v in roi),
            exposure_s=exposure,
            camera_type=type(camera).__name__,
            camera_id=None if identity is None else str(identity),
            readout_mode=None if mode is None else str(mode),
        )

    def assert_image(self, image) -> None:
        got = tuple(int(v) for v in np.shape(image))
        if got != self.image_shape:
            raise ValueError(
                f"image shape {got} does not match calibration frame contract "
                f"{self.image_shape}; recalibrate for the current camera ROI.")

    def assert_compatible(self, observed: "FrameContract") -> None:
        """Reject any known physical-camera fact that differs."""

        mismatches = []
        for name in ("image_shape", "sensor_shape", "roi", "exposure_s",
                     "camera_type", "camera_id", "readout_mode"):
            expected, actual = getattr(self, name), getattr(observed, name)
            if expected is not None and actual is not None and expected != actual:
                mismatches.append(f"{name}: calibrated={expected!r}, current={actual!r}")
        if mismatches:
            raise ValueError("camera/calibration frame-contract mismatch (" + "; ".join(mismatches) + ").")


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
    schema: ClassVar[str] = "Zou_lab_control.neutral_atom.TrapCalibration"
    version: ClassVar[int] = 1

    centers: np.ndarray
    thresholds: np.ndarray | float
    frame_contract: FrameContract | Mapping[str, Any]
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
    # methods and the downstream detector picks which to use (calibrate once, read
    # many ways).  ``{method: {"thresholds": (N,), "psf_weights": (N,h,w)|None,
    # "psf_boxes": (N,4)|None}}`` for every method OTHER than this calibration's own
    # (which lives in the top-level fields).  None / empty = single-method (the default).
    by_method: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = (self.frame_contract if isinstance(self.frame_contract, FrameContract)
                    else FrameContract.from_dict(require_object(
                        self.frame_contract, what="TrapCalibration.frame_contract")))
        object.__setattr__(self, "frame_contract", contract)
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
        """The camera gate time these thresholds were learned at, or ``fallback``.

        Exposure is a typed field of :class:`FrameContract`, not an unversioned
        metadata spelling.  This is the ONE authoritative reader
        of that exposure-self-match invariant: every readout that must image at the calibration's
        exposure (``detect``, the live calibrate-task adoption, the temperature survival frames)
        goes through HERE instead of reaching into unversioned metadata
        -- a threshold is exposure-specific, so a missed/mistyped lookup re-floors occupancy /
        sticks survival at the readout false-positive rate (#issue-2 / #H3v-2).  Callers pass their
        OWN fallback (the camera exposure, or None to mean 'do not force a match')."""
        value = self.frame_contract.exposure_s
        return float(value) if value is not None else fallback

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

        self.frame_contract.assert_image(image)
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

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(include_fingerprint=False), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "version": self.version,
            "centers": self.centers.tolist(),
            "thresholds": self.thresholds.tolist(),
            "frame_contract": self.frame_contract.to_dict(),
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
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrapCalibration":
        fields = {
            "schema", "version", "centers", "thresholds", "frame_contract",
            "grid_shape", "roi_radius", "reducer", "method", "psf_weights",
            "psf_boxes", "background", "by_method", "metadata", "fingerprint",
        }
        require_exact_fields(payload, fields, what="trap calibration")
        schema = require_string(payload["schema"], what="TrapCalibration.schema")
        version = require_int(payload["version"], what="TrapCalibration.version")
        if schema != cls.schema or version != cls.version:
            raise ValueError(
                f"unsupported trap calibration {schema!r} version {version!r}; "
                f"expected {cls.schema!r} version {cls.version}.")
        centers = require_array(payload["centers"], what="TrapCalibration.centers")
        thresholds = require_array(payload["thresholds"], what="TrapCalibration.thresholds")
        grid = (None if payload["grid_shape"] is None else
                require_array(payload["grid_shape"], what="TrapCalibration.grid_shape"))
        roi_radius = (None if payload["roi_radius"] is None else
                      require_int(payload["roi_radius"], what="TrapCalibration.roi_radius"))
        reducer = (None if payload["reducer"] is None else
                   require_string(payload["reducer"], what="TrapCalibration.reducer"))
        method = require_string(payload["method"], what="TrapCalibration.method")
        background = require_string(payload["background"], what="TrapCalibration.background")
        psf_weights = (None if payload["psf_weights"] is None else
                       require_array(payload["psf_weights"], what="TrapCalibration.psf_weights"))
        psf_boxes = (None if payload["psf_boxes"] is None else
                     require_array(payload["psf_boxes"], what="TrapCalibration.psf_boxes"))
        by_method = (None if payload["by_method"] is None else
                     require_object(payload["by_method"], what="TrapCalibration.by_method"))
        calibration = cls(
            centers,
            thresholds,
            frame_contract=FrameContract.from_dict(require_object(
                payload["frame_contract"], what="TrapCalibration.frame_contract")),
            grid_shape=None if grid is None else tuple(grid),
            roi_radius=roi_radius,
            reducer=reducer,
            method=method,
            psf_weights=psf_weights,
            psf_boxes=psf_boxes,
            background=background,
            by_method=by_method,
            metadata=require_object(payload["metadata"], what="TrapCalibration.metadata"),
        )
        if require_string(payload["fingerprint"], what="TrapCalibration.fingerprint") != calibration.fingerprint:
            raise ValueError("trap-calibration fingerprint mismatch.")
        return calibration

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".npz":
            np.savez(
                path,
                schema=np.asarray(self.schema),
                version=np.asarray(self.version, dtype=np.int64),
                fingerprint=np.asarray(self.fingerprint),
                frame_contract_json=np.asarray(json.dumps(
                    self.frame_contract.to_dict(), ensure_ascii=False, sort_keys=True)),
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
            expected = {
                "schema", "version", "fingerprint", "frame_contract_json",
                "centers", "thresholds", "grid_shape", "roi_radius", "reducer",
                "method", "background", "psf_weights", "psf_boxes",
                "by_method_json", "metadata_json",
            }
            with np.load(path, allow_pickle=False) as data:
                actual = set(data.files)
                if actual != expected:
                    raise ValueError(
                        f"trap calibration npz fields mismatch: missing={sorted(expected - actual)}, "
                        f"unknown={sorted(actual - expected)}.")
                grid = data["grid_shape"]
                psf_weights = data["psf_weights"]
                psf_boxes = data["psf_boxes"]
                roi_radius_raw = int(data["roi_radius"].item())
                reducer_raw = str(data["reducer"].item())
                payload = {
                    "schema": str(data["schema"].item()),
                    "version": int(data["version"].item()),
                    "fingerprint": str(data["fingerprint"].item()),
                    "frame_contract": json.loads(str(data["frame_contract_json"].item())),
                    "centers": data["centers"].tolist(),
                    "thresholds": data["thresholds"].tolist(),
                    "grid_shape": None if grid.size == 0 else [int(v) for v in grid],
                    "roi_radius": None if roi_radius_raw < 0 else roi_radius_raw,
                    "reducer": None if reducer_raw == "" else reducer_raw,
                    "method": str(data["method"].item()),
                    "background": str(data["background"].item()),
                    "psf_weights": None if psf_weights.size == 0 else psf_weights.tolist(),
                    "psf_boxes": None if psf_boxes.size == 0 else psf_boxes.tolist(),
                    "by_method": json.loads(str(data["by_method_json"].item())),
                    "metadata": json.loads(str(data["metadata_json"].item())),
                }
            return cls.from_dict(payload)
        payload = require_object(
            json.loads(path.read_text(encoding="utf-8")), what="trap calibration file")
        return cls.from_dict(payload)


__all__ = ["FrameContract", "TrapCalibration", "READOUT_KINDS", "readout_kind"]
