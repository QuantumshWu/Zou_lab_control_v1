"""Human-readable export of one canonical calibration result.

The Calibration/Capture repositories remain the only machine authority.  This
module writes an operator-facing, explicitly non-authoritative projection of an
already admitted ``CalibrationComputation``: a JSON index, numeric tables and
arrays, and frontend-rendered PNG pages.  None of these files is admitted as a
CalibrationArtifact or consumed by Occupancy.
"""

from __future__ import annotations

from collections.abc import Callable
import csv
import json
import math
from pathlib import Path
import re
from typing import Protocol

import numpy as np

from zlc_storage import canonical_text, sha256_digest

from zlc_neutral_atom.capture.reference import (
    CaptureArtifactRef,
    capture_artifact_ref_to_tree,
)

from .projection import CalibrationReportProjection
from .reference import (
    CalibrationArtifactRef,
    calibration_artifact_ref_to_tree,
)


CALIBRATION_RESULT_BUNDLE_FORMAT = (
    "zlc_neutral_atom.logic_nodes.readout.calibration.result-bundle"
)
_PAGE_KEY = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


class _RasterPage(Protocol):
    key: str
    title: str
    png_bytes: bytes


class _RasterDocument(Protocol):
    summary: str
    pages: tuple[_RasterPage, ...]


def _json_real(value: object) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _json_reals(values: object) -> list[float | None]:
    return [_json_real(value) for value in np.asarray(values).reshape(-1)]


def _write_sites_csv(path: Path, view: CalibrationReportProjection) -> None:
    model_columns: list[str] = []
    for index, _model in enumerate(view.models):
        prefix = f"model_{index}"
        model_columns.extend(
            (
                f"{prefix}_threshold",
                f"{prefix}_threshold_source",
                f"{prefix}_model_fidelity",
                f"{prefix}_heldout_fidelity",
                f"{prefix}_dark_fidelity",
                f"{prefix}_bright_fidelity",
                f"{prefix}_dark_mean",
                f"{prefix}_dark_sigma",
                f"{prefix}_bright_mean",
                f"{prefix}_bright_sigma",
                f"{prefix}_n_test",
                f"{prefix}_n_train_dark",
                f"{prefix}_n_train_bright",
                f"{prefix}_feature_valid",
                f"{prefix}_runtime_usable",
                f"{prefix}_bright_above",
            )
        )
    columns = [
        "site_index",
        "site_label",
        "x_px",
        "y_px",
        "site_valid",
        *model_columns,
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for site, label in enumerate(view.site_labels):
            x_px, y_px = view.actual_centers_xy[site]
            row: dict[str, object] = {
                "site_index": site,
                "site_label": label,
                "x_px": repr(float(x_px)),
                "y_px": repr(float(y_px)),
                "site_valid": bool(view.site_validity[site]),
            }
            for index, model in enumerate(view.models):
                prefix = f"model_{index}"
                row.update(
                    {
                        f"{prefix}_threshold": repr(
                            float(model.runtime_thresholds[site])
                        ),
                        f"{prefix}_threshold_source": (
                            model.runtime_threshold_sources[site]
                        ),
                        f"{prefix}_model_fidelity": repr(
                            float(model.model_fidelity[site])
                        ),
                        f"{prefix}_heldout_fidelity": repr(
                            float(model.heldout_fidelity[site])
                        ),
                        f"{prefix}_dark_fidelity": repr(
                            float(model.dark_fidelity[site])
                        ),
                        f"{prefix}_bright_fidelity": repr(
                            float(model.bright_fidelity[site])
                        ),
                        f"{prefix}_dark_mean": repr(float(model.dark_mean[site])),
                        f"{prefix}_dark_sigma": repr(float(model.dark_sigma[site])),
                        f"{prefix}_bright_mean": repr(
                            float(model.bright_mean[site])
                        ),
                        f"{prefix}_bright_sigma": repr(
                            float(model.bright_sigma[site])
                        ),
                        f"{prefix}_n_test": int(model.n_test[site]),
                        f"{prefix}_n_train_dark": int(
                            model.n_train_dark[site]
                        ),
                        f"{prefix}_n_train_bright": int(
                            model.n_train_bright[site]
                        ),
                        f"{prefix}_feature_valid": bool(
                            model.feature_validity[site]
                        ),
                        f"{prefix}_runtime_usable": bool(
                            model.runtime_usable[site]
                        ),
                        f"{prefix}_bright_above": bool(
                            model.bright_above[site]
                        ),
                    }
                )
            writer.writerow(row)


def _write_diagnostics_npz(
    path: Path,
    view: CalibrationReportProjection,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "reference_average": np.asarray(view.reference_average),
        "reference_average_validity": np.asarray(
            view.reference_average_validity,
            dtype=np.bool_,
        ),
        "actual_centers_xy": np.asarray(view.actual_centers_xy),
        "site_validity": np.asarray(view.site_validity, dtype=np.bool_),
        "default_boxes_xywh": np.asarray(view.default_boxes_xywh),
        "occupied_labels": np.asarray(view.occupied_labels, dtype=np.bool_),
        "dark_labels": np.asarray(view.dark_labels, dtype=np.bool_),
        "label_validity": np.asarray(view.label_validity, dtype=np.bool_),
    }
    if view.expected_centers_xy is not None:
        arrays["expected_centers_xy"] = np.asarray(view.expected_centers_xy)
    if view.psf_kernels is not None:
        arrays["psf_kernels"] = np.asarray(view.psf_kernels)
        arrays["psf_fit_ok"] = np.asarray(view.psf_fit_ok, dtype=np.bool_)
        arrays["psf_sigma_xy"] = np.asarray(view.psf_sigma_xy)
    for index, model in enumerate(view.models):
        prefix = f"model_{index}"
        arrays.update(
            {
                f"{prefix}_signals": np.asarray(model.signals),
                f"{prefix}_signal_validity": np.asarray(
                    model.signal_validity,
                    dtype=np.bool_,
                ),
                f"{prefix}_bin_edges": np.asarray(model.bin_edges),
                f"{prefix}_quick_thresholds": np.asarray(
                    model.quick_thresholds
                ),
                f"{prefix}_formal_thresholds": np.asarray(
                    model.formal_thresholds
                ),
                f"{prefix}_runtime_thresholds": np.asarray(
                    model.runtime_thresholds
                ),
                f"{prefix}_feature_validity": np.asarray(
                    model.feature_validity,
                    dtype=np.bool_,
                ),
                f"{prefix}_runtime_usable": np.asarray(
                    model.runtime_usable,
                    dtype=np.bool_,
                ),
                f"{prefix}_bright_above": np.asarray(
                    model.bright_above,
                    dtype=np.bool_,
                ),
                f"{prefix}_model_fidelity": np.asarray(model.model_fidelity),
                f"{prefix}_heldout_fidelity": np.asarray(
                    model.heldout_fidelity
                ),
                f"{prefix}_dark_fidelity": np.asarray(model.dark_fidelity),
                f"{prefix}_bright_fidelity": np.asarray(model.bright_fidelity),
                f"{prefix}_dark_mean": np.asarray(model.dark_mean),
                f"{prefix}_dark_sigma": np.asarray(model.dark_sigma),
                f"{prefix}_bright_mean": np.asarray(model.bright_mean),
                f"{prefix}_bright_sigma": np.asarray(model.bright_sigma),
                f"{prefix}_n_test": np.asarray(model.n_test),
                f"{prefix}_n_train_dark": np.asarray(model.n_train_dark),
                f"{prefix}_n_train_bright": np.asarray(model.n_train_bright),
                f"{prefix}_ablation_drop_worst_k": np.asarray(
                    model.ablation_drop_worst_k
                ),
                f"{prefix}_ablation_excluded_sites": np.asarray(
                    model.ablation_excluded_sites,
                    dtype=np.bool_,
                ),
                f"{prefix}_ablation_fidelity": np.asarray(
                    model.ablation_fidelity
                ),
                f"{prefix}_ablation_errors": np.asarray(
                    model.ablation_errors
                ),
                f"{prefix}_ablation_n_valid": np.asarray(
                    model.ablation_n_valid
                ),
            }
        )
    np.savez(path, **arrays)


def _model_summary(index: int, model) -> dict[str, object]:
    return {
        "array_prefix": f"model_{index}",
        "label": canonical_text(model.label, "calibration model label"),
        "is_default": bool(model.is_default),
        "runtime_thresholds": _json_reals(model.runtime_thresholds),
        "runtime_threshold_sources": list(model.runtime_threshold_sources),
        "feature_valid": [bool(value) for value in model.feature_validity],
        "runtime_usable": [bool(value) for value in model.runtime_usable],
        "bright_above": [bool(value) for value in model.bright_above],
        "model_fidelity": _json_reals(model.model_fidelity),
        "heldout_fidelity": _json_reals(model.heldout_fidelity),
        "dark_fidelity": _json_reals(model.dark_fidelity),
        "bright_fidelity": _json_reals(model.bright_fidelity),
        "dark_mean": _json_reals(model.dark_mean),
        "dark_sigma": _json_reals(model.dark_sigma),
        "bright_mean": _json_reals(model.bright_mean),
        "bright_sigma": _json_reals(model.bright_sigma),
        "n_test": [int(value) for value in model.n_test],
        "n_train_dark": [int(value) for value in model.n_train_dark],
        "n_train_bright": [int(value) for value in model.n_train_bright],
        "runtime_model_fidelity_mean": _json_real(
            model.runtime_model_fidelity_mean
        ),
        "aggregate_fidelity": _json_real(model.aggregate_fidelity),
        "global_threshold": _json_real(model.global_threshold),
        "global_bright_above": bool(model.global_bright_above),
        "global_fidelity": _json_real(model.global_fidelity),
        "ablation": [
            {
                "drop_worst_k": int(model.ablation_drop_worst_k[index]),
                "excluded_sites": [
                    bool(value)
                    for value in model.ablation_excluded_sites[index]
                ],
                "fidelity": _json_real(model.ablation_fidelity[index]),
                "errors": int(model.ablation_errors[index]),
                "n_valid": int(model.ablation_n_valid[index]),
            }
            for index in range(len(model.ablation_drop_worst_k))
        ],
    }


def write_calibration_result_bundle(
    destination: str | Path,
    view: CalibrationReportProjection,
    calibration_ref: CalibrationArtifactRef,
    source_capture_ref: CaptureArtifactRef,
    *,
    calibration_repository_root: str | Path,
    capture_repository_root: str | Path,
    render_report: Callable[[CalibrationReportProjection], _RasterDocument],
) -> None:
    """Write one complete operator bundle into a new empty directory.

    ``render_report`` is the explicit composition seam: neutral owns all physical
    values and the file index, while frontend owns pixels.  The caller atomically
    installs this fully staged directory before publishing ``calibration_ref.json``.
    """

    if not isinstance(view, CalibrationReportProjection):
        raise TypeError("view must be CalibrationReportProjection")
    if not isinstance(calibration_ref, CalibrationArtifactRef):
        raise TypeError("calibration_ref must be CalibrationArtifactRef")
    if not isinstance(source_capture_ref, CaptureArtifactRef):
        raise TypeError("source_capture_ref must be CaptureArtifactRef")
    if not callable(render_report):
        raise TypeError("render_report must be callable")
    if view.calibration_identity != calibration_ref.target_ref:
        raise ValueError("calibration report projection belongs to another artifact")
    if view.source_capture_identity != source_capture_ref.target_ref:
        raise ValueError("calibration report projection belongs to another capture")
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=False, exist_ok=False)

    document = render_report(view)
    summary_text = canonical_text(document.summary, "calibration report summary")
    pages = tuple(document.pages)
    if not pages:
        raise ValueError("calibration report renderer returned no pages")
    keys: set[str] = set()
    files: dict[str, dict[str, object]] = {}
    for page in pages:
        key = canonical_text(page.key, "calibration report page key")
        if _PAGE_KEY.fullmatch(key) is None or key in keys:
            raise ValueError("calibration report page keys must be unique path-safe names")
        keys.add(key)
        title = canonical_text(page.title, "calibration report page title")
        payload = page.png_bytes
        if not isinstance(payload, bytes) or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("calibration report page must be encoded PNG bytes")
        filename = f"{key}.png"
        page_path = root / filename
        page_path.write_bytes(payload)
        files[filename] = {
            "description": title,
            "sha256": sha256_digest(page_path.read_bytes()),
            "size_bytes": page_path.stat().st_size,
        }

    diagnostics_path = root / "diagnostics.npz"
    _write_diagnostics_npz(diagnostics_path, view)
    files[diagnostics_path.name] = {
        "description": (
            "non-authoritative numeric export of the stored CalibrationReport"
        ),
        "sha256": sha256_digest(diagnostics_path.read_bytes()),
        "size_bytes": diagnostics_path.stat().st_size,
    }
    sites_path = root / "sites.csv"
    _write_sites_csv(sites_path, view)
    files[sites_path.name] = {
        "description": "per-site operator table",
        "sha256": sha256_digest(sites_path.read_bytes()),
        "size_bytes": sites_path.stat().st_size,
    }

    calibration_root = Path(calibration_repository_root).expanduser().resolve()
    capture_root = Path(capture_repository_root).expanduser().resolve()
    summary = {
        "schema": CALIBRATION_RESULT_BUNDLE_FORMAT,
        "authority": {
            "calibration_ref": calibration_artifact_ref_to_tree(calibration_ref),
            "source_capture_ref": capture_artifact_ref_to_tree(source_capture_ref),
            "calibration_repository_root": str(calibration_root),
            "capture_repository_root": str(capture_root),
            "rule": (
                "The repositories and typed refs above are the only machine "
                "authority. This report directory is a human-readable projection."
            ),
        },
        "physical_context": {
            "binding": view.binding,
            "camera_identity": view.camera_identity,
            "roi_shape_yx": list(view.roi_shape_yx),
            "exposure_seconds": view.exposure_seconds,
            "group_count": view.group_count,
        },
        "site_map": {
            "site_count": len(view.site_labels),
            "grid_shape_yx": list(view.grid_shape_yx),
            "labels": list(view.site_labels),
            "valid": [bool(value) for value in view.site_validity],
            "centers_xy": [
                [_json_real(x), _json_real(y)]
                for x, y in view.actual_centers_xy
            ],
        },
        "models": [
            _model_summary(index, model)
            for index, model in enumerate(view.models)
        ],
        "software_lineage": [list(item) for item in view.software_lineage],
        "render_summary": summary_text,
        "files": files,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (root / "README.txt").write_text(
        "Calibration result bundle\n"
        "=========================\n\n"
        "Open summary.json for the result index and canonical repository paths.\n"
        "Open the PNG pages for SiteMap, fidelity, per-site distributions, "
        "pooled populations, and optional PSF diagnostics.\n"
        "sites.csv and diagnostics.npz are convenient, non-authoritative exports.\n\n"
        "Do not use this directory as a calibration database. Occupancy and all\n"
        "machine consumers admit the CalibrationArtifactRef in ../calibration_ref.json\n"
        "from the canonical Calibration/Capture repositories listed in summary.json.\n",
        encoding="utf-8",
    )


__all__ = [
    "CALIBRATION_RESULT_BUNDLE_FORMAT",
    "write_calibration_result_bundle",
]
