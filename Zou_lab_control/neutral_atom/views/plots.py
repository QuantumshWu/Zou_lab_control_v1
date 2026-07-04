"""Plot adapters that keep neutral-atom code on the frontend data contract.

These helpers turn experiment data (images, site centers, per-site values) into
frontend plots WITHOUT importing the frontend.  They route through the viewer
registered in :mod:`Zou_lab_control._viewer_registry`; the frontend registers
itself on import.  When no viewer is registered (headless runs, or the frontend
was never imported) every adapter returns ``None`` and the data on the result
object is still complete.
"""

from __future__ import annotations

from math import ceil, sqrt
from typing import Sequence

from matplotlib.patches import Circle
import numpy as np

from Zou_lab_control._viewer_registry import active_plotter
from ..core.analysis import positive_int


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
    """Plot a qCMOS image with optional site-center overlay (``None`` if headless)."""

    plotter = active_plotter()
    if plotter is None:
        return None

    data_x, data_y = image_to_points(image, max_points=kwargs.pop("max_points", 120_000))
    plot = plotter.plot(data_x, data_y, labels=labels, display=False, **kwargs)
    if centers is not None:
        centers = np.asarray(centers, dtype=float)
        if centers.size:
            radius = _site_radius(roi_radius)
            for x, y in centers[:, :2]:
                plot.ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor="#C37D5A", linewidth=0.65, alpha=0.9, zorder=5))
    if display:
        plotter.display_figure(plot.fig)
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
    """Plot the camera frame with faint (empty) / bold (occupied) site rings.

    The ring art is NOT hard-coded here: this routes through the frontend's sealed ``site_map``
    view, whose rings come from the single ``SITE_OCCUPANCY_STYLE`` source (a ``LiveSiteMap``).
    ``None`` when headless / the registered plotter has no ``site_map``."""

    centers = np.asarray(centers, dtype=float)
    occupied = np.asarray(occupied, dtype=bool).reshape(-1)
    if centers.ndim != 2 or centers.shape[1] < 2:
        raise ValueError("centers must have shape (N, 2).")
    if len(centers) != len(occupied):
        raise ValueError("occupied must have one value per site center.")

    plotter = active_plotter()
    site_map = getattr(plotter, "site_map", None)
    if site_map is None:
        return None
    return site_map(centers[:, :2], occupied, image=image, roi_radius=_site_radius(roi_radius),
                    labels=labels, display=display, **kwargs)


def plot_site_values(centers, values, *, labels=("Camera x (px)", "Camera y (px)", "Value"), display: bool = True, **kwargs):
    """Plot one scalar per trap site (``None`` if headless)."""

    centers = np.asarray(centers, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    if centers.ndim != 2 or centers.shape[1] < 2:
        raise ValueError("centers must have shape (N, 2).")
    if len(centers) != len(values):
        raise ValueError("centers and values must have the same length.")
    plotter = active_plotter()
    if plotter is None:
        return None
    return plotter.plot(centers[:, :2], values, labels=labels, display=display, **kwargs)


def plot_threshold_hist(values, *, threshold=None, labels=("ROI counts", "Shots", "Population"), display: bool = True, **kwargs):
    """Plot threshold calibration values as a frontend histogram (``None`` if headless)."""

    plotter = active_plotter()
    if plotter is None:
        return None
    thresholds = [] if threshold is None else [float(threshold)]
    return plotter.plot(np.asarray(values, dtype=float).reshape(-1), kind="hist", labels=labels, thresholds=thresholds, display=display, **kwargs)


def plot_detection_scan(times, fidelities, *, labels=("Detection time (s)", "Fidelity", "Fidelity"), display: bool = True, **kwargs):
    """Plot detection-time fidelity scan (``None`` if headless)."""

    plotter = active_plotter()
    if plotter is None:
        return None
    return plotter.plot(np.asarray(times, dtype=float).reshape(-1, 1), np.asarray(fidelities, dtype=float).reshape(-1, 1), labels=labels, display=display, **kwargs)


__all__ = ["image_to_points", "plot_detection_image", "plot_detection_scan", "plot_image", "plot_site_values", "plot_threshold_hist"]
