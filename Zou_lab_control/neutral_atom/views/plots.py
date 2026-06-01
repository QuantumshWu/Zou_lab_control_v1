"""Plot adapters that keep neutral-atom code on the frontend data contract."""

from __future__ import annotations

from math import ceil, sqrt
from typing import Sequence

from matplotlib.patches import Circle
import numpy as np


def image_to_points(image, *, max_points: int | None = 120_000):
    """Convert a 2D image into ``frontend.plot`` point-table data."""

    img = np.asarray(image, dtype=float)
    if img.ndim != 2 or 0 in img.shape:
        raise ValueError("image must be a non-empty 2D array.")
    if max_points is None:
        stride = 1
    else:
        max_points = positive_int(max_points, "max_points")
        stride = max(1, int(ceil(sqrt(img.size / max_points))))
    yy, xx = np.mgrid[0 : img.shape[0] : stride, 0 : img.shape[1] : stride]
    data_x = np.column_stack([xx.ravel(), yy.ravel()])
    data_y = img[::stride, ::stride].ravel().reshape(-1, 1)
    return data_x, data_y


def _draw_now(fig) -> None:
    fig.canvas.draw_idle()
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
    except Exception:
        pass


def _site_radius(roi_radius: int | float = 1) -> float:
    return max(float(roi_radius) + 3.5, 4.5)


def plot_image(image, *, centers=None, roi_radius: int = 1, labels=("Camera x (px)", "Camera y (px)", "Counts"), display: bool = True, **kwargs):
    """Plot a qCMOS image with optional site-center overlay."""

    from Zou_lab_control import frontend as zf

    data_x, data_y = image_to_points(image, max_points=kwargs.pop("max_points", 120_000))
    plot = zf.plot(data_x, data_y, labels=labels, display=False, **kwargs)
    if centers is not None:
        centers = np.asarray(centers, dtype=float)
        if centers.size:
            radius = _site_radius(roi_radius)
            for x, y in centers[:, :2]:
                plot.ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor="#C37D5A", linewidth=0.65, alpha=0.9, zorder=5))
    if display:
        zf.display_figure(plot.fig)
    else:
        _draw_now(plot.fig)
    return plot


def plot_detection_image(
    image,
    centers,
    occupied,
    *,
    roi_radius: int = 1,
    labels=("Camera x (px)", "Camera y (px)", "Counts"),
    display: bool = True,
    **kwargs,
):
    """Plot raw camera data with faint sitemap rings and occupied-site rings."""

    centers = np.asarray(centers, dtype=float)
    occupied = np.asarray(occupied, dtype=bool).reshape(-1)
    if centers.ndim != 2 or centers.shape[1] < 2:
        raise ValueError("centers must have shape (N, 2).")
    if len(centers) != len(occupied):
        raise ValueError("occupied must have one value per site center.")
    from Zou_lab_control import frontend as zf

    plot = plot_image(image, labels=labels, display=False, **kwargs)
    radius = _site_radius(roi_radius)
    for x, y in centers[:, :2]:
        plot.ax.add_patch(
            Circle((x, y), radius, facecolor="none", edgecolor="#7EA5A3", linewidth=0.45, alpha=0.24, zorder=4)
        )
    for x, y in centers[occupied, :2]:
        plot.ax.add_patch(
            Circle((x, y), radius, facecolor="none", edgecolor="#D07850", linewidth=0.85, alpha=0.94, zorder=5)
        )
    if display:
        zf.display_figure(plot.fig)
    else:
        _draw_now(plot.fig)
    return plot


def plot_site_values(centers, values, *, labels=("Camera x (px)", "Camera y (px)", "Value"), display: bool = True, **kwargs):
    """Plot one scalar per trap site."""

    from Zou_lab_control import frontend as zf

    centers = np.asarray(centers, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    if centers.ndim != 2 or centers.shape[1] < 2:
        raise ValueError("centers must have shape (N, 2).")
    if len(centers) != len(values):
        raise ValueError("centers and values must have the same length.")
    return zf.plot(centers[:, :2], values, labels=labels, display=display, **kwargs)


def plot_threshold_hist(values, *, threshold=None, labels=("ROI counts", "Shots", "Population"), display: bool = True, **kwargs):
    """Plot threshold calibration values as a frontend histogram."""

    from Zou_lab_control import frontend as zf

    thresholds = [] if threshold is None else [float(threshold)]
    return zf.plot(np.asarray(values, dtype=float).reshape(-1), kind="hist", labels=labels, thresholds=thresholds, display=display, **kwargs)


def plot_detection_scan(times, fidelities, *, labels=("Detection time (s)", "Fidelity", "Fidelity"), display: bool = True, **kwargs):
    """Plot detection-time fidelity scan."""

    from Zou_lab_control import frontend as zf

    return zf.plot(np.asarray(times, dtype=float).reshape(-1, 1), np.asarray(fidelities, dtype=float).reshape(-1, 1), labels=labels, display=display, **kwargs)


def positive_int(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    numeric = float(value)
    if not np.isfinite(numeric) or int(numeric) != numeric or numeric <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


__all__ = ["image_to_points", "plot_detection_image", "plot_detection_scan", "plot_image", "plot_site_values", "plot_threshold_hist"]
