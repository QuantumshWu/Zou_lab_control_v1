"""Frontend manual content loader and example figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

_CONTENT_DIR = Path(__file__).resolve().parent / "manual_templates"


def _save_plot_figure(plot_obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_obj.fig.savefig(path, bbox_inches="tight")


def generate_frontend_manual_figures(asset_dir: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from ..live import plot

    asset_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12)

    x = np.linspace(737.0, 737.2, 220).reshape(-1, 1)
    y = 18 * ((0.018 / 2) ** 2) / ((x[:, 0] - 737.095) ** 2 + (0.018 / 2) ** 2) + 3
    y = (y + rng.normal(0, 0.18, size=len(x))).reshape(-1, 1)
    p1 = plot(x, y, labels=("Wavelength (nm)", "Counts/0.1s", "Counts"), relim_mode="tight", display=False)
    p1.data_figure.lorent(is_display=True)
    one_d = asset_dir / "frontend_1d_fit.pdf"
    _save_plot_figure(p1, one_d)
    plt.close(p1.fig)

    scan_x_axis = np.linspace(-4, 4, 37)
    scan_y_axis = np.linspace(-3, 3, 25)
    sx, sy = np.meshgrid(scan_x_axis, scan_y_axis)
    z = 420 * np.exp(-((sx - 0.7) ** 2 + (sy + 0.4) ** 2) / 5.0) + 30
    z += rng.normal(0, 8, size=z.shape)
    data_x = np.column_stack([sx.ravel(), sy.ravel()])
    data_y = z.ravel().reshape(-1, 1)
    p2 = plot(data_x, data_y, labels=("X (um)", "Y (um)", "Counts"), display=False)
    two_d = asset_dir / "frontend_2d_scan.pdf"
    _save_plot_figure(p2, two_d)
    plt.close(p2.fig)

    dark = rng.normal(20, 4, 260)
    bright = rng.normal(78, 8, 360)
    hist = plot(
        np.r_[dark, bright],
        kind="hist",
        bins=55,
        thresholds=[45],
        labels=("ROI counts", "Shots", "Population"),
        display=False,
    )
    hist_fig = asset_dir / "frontend_histogram.pdf"
    _save_plot_figure(hist, hist_fig)
    plt.close(hist.fig)

    return {"one_d": one_d, "two_d": two_d, "hist": hist_fig}


def frontend_manual_body(figures: Mapping[str, Path]) -> str:
    body = (_CONTENT_DIR / "frontend_manual_zh.texbody").read_text(encoding="utf-8")
    return (
        body.replace("__ONE_D__", figures["one_d"].as_posix())
        .replace("__TWO_D__", figures["two_d"].as_posix())
        .replace("__HIST__", figures["hist"].as_posix())
    )


__all__ = ["frontend_manual_body", "generate_frontend_manual_figures"]
