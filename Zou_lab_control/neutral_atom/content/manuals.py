"""Neutral-atom manual text generation."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Mapping


def _template_text(name: str) -> str:
    path = resources.files("Zou_lab_control.neutral_atom") / "content" / "manual_templates" / name
    return path.read_text(encoding="utf-8")


def _device_placeholder_image(path: Path, caption: str) -> Path:
    """Write a small matplotlib placeholder so the manual still builds even if
    the live virtual-backend render is unavailable."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.axis("off")
    ax.text(0.5, 0.5, caption, ha="center", va="center", wrap=True, fontsize=9)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def _render_threshold_hist(path: Path) -> Path:
    """Render the REAL threshold-calibration histogram the readout tutorial
    produces, using the offline virtual backend (no hardware)."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from Zou_lab_control import neutral_atom as na

        exp = na.connect("virtual")
        exp.readout.sitemap(frames=6, display=False)
        threshold_result = exp.readout.thresholds(frames=120, site=0, display=False)
        plot = threshold_result.plot_site(0, display=False)
        plot.fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(plot.fig)
        return path
    except Exception:  # pragma: no cover - defensive: keep the manual buildable
        return _device_placeholder_image(path, "阈值直方图（暗/亮双峰 + Otsu 阈值）")


def generate_device_manual_figures(asset_dir: str | Path) -> dict[str, Path]:
    """Render the device-manual figures (real tutorial output) into asset_dir."""

    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    return {"threshold_hist": _render_threshold_hist(asset_dir / "device_threshold_hist.png")}


def _threshold_figure_tex(fig_path: str) -> str:
    caption = (
        "虚拟后端\\tfocus{实跑}的 thresholds 标定直方图（Site 0，120 帧）：左峰为\\tfocus{暗}态"
        "（背景+读出噪声），右峰为\\tfocus{亮}态（原子荧光），并叠加亮/暗高斯拟合；图中标注了 Otsu "
        "阈值、拟合保真度与亮/暗占比。\\pyapi{detect} 时把单张图每格点的 ROI 计数与该阈值逐位比较，"
        "得占据布尔；两峰分得越开、保真度越接近 1。本图由读出标定 \\pyapi{thresholds} 的 "
        "\\pyapi{plot_site} 直接产出，而非示意图。"
    )
    return (
        "\\begin{figure}[h]\n\\centering\n"
        f"\\includegraphics[width=0.6\\linewidth]{{{fig_path}}}\n"
        f"\\caption{{{caption}}}\n"
        "\\end{figure}"
    )


def main_manual_body() -> str:
    """Return the main (system-overview) manual body.

    The main manual uses inline TikZ diagrams only, so no figure files are
    required; the template compiles as-is.
    """

    return _template_text("main_manual_zh.texbody")


def fpga_manual_body() -> str:
    """Return the FPGA (pulse-streamer) manual body.

    The FPGA manual uses inline TikZ diagrams only, so no figure files are
    required; the template compiles as-is.
    """

    return _template_text("fpga_manual_zh.texbody")


def device_manual_body(figures: Mapping[str, Path] | None = None) -> str:
    """Return the device & experiment manual body.

    Covers device configuration/loading, camera capture, the camera-readout
    tutorial (sitemap/thresholds/detect) with principles, calibration & result
    objects, and the end-to-end experiment flow.  ``figures`` (from
    :func:`generate_device_manual_figures`) injects the real threshold-histogram
    image; omit it and the placeholder is simply dropped so the text still
    compiles.
    """

    body = _template_text("device_manual_zh.texbody")
    fig_path = None if not figures else figures.get("threshold_hist")
    figure_tex = _threshold_figure_tex(Path(fig_path).as_posix()) if fig_path else ""
    return body.replace("__READOUT_THRESHOLD_FIG__", figure_tex)


__all__ = [
    "device_manual_body",
    "generate_device_manual_figures",
    "fpga_manual_body",
    "main_manual_body",
]
