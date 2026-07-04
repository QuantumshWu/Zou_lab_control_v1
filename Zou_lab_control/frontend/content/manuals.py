"""Frontend manual content loader and example figure generation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import numpy as np

_CONTENT_DIR = Path(__file__).resolve().parent / "manual_templates"


def _save_plot_figure(plot_obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_obj.fig.savefig(path, bbox_inches="tight")


def _placeholder_image(path: Path, caption: str) -> Path:
    """Write a small matplotlib placeholder so the manual still compiles.

    Used when the offscreen Qt screenshot cannot be produced (e.g. PyQt or the
    ``offscreen`` platform plugin is unavailable in a headless build).
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    ax.axis("off")
    ax.text(0.5, 0.5, caption, ha="center", va="center", fontsize=11, wrap=True)
    ax.add_patch(plt.Rectangle((0.01, 0.04), 0.98, 0.92, fill=False, lw=1.0, ec="#999999"))
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def generate_frontend_gui_figures(asset_dir: Path) -> dict[str, Path]:
    """Render real Edit / Preview / Scan screenshots of the pulse editor.

    The pulse editor is a live PyQt widget, so these are genuine GUI grabs (not
    mockups), captured with the ``offscreen`` Qt platform.  Each capture is
    individually guarded: if Qt is unavailable the slot falls back to a labelled
    placeholder so the manual still builds.  Text is rendered via the bundled
    screenshot font so offscreen grabs are not blank.
    """

    asset_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    targets = {
        "gui_edit": ("edit", "Edit 标签页：脉冲表编辑（截图不可用）"),
        "gui_preview": ("preview", "Preview 标签页：未展开时序图（截图不可用）"),
        "gui_scan": ("scan", "Scan 标签页：写代码生成扫描表（截图不可用）"),
    }
    out: dict[str, Path] = {}
    editor = None
    try:
        from .. import devtools as dt

        editor = dt.demo_editor(size=(1480, 900))
    except Exception:
        editor = None
    for key, (tab, caption) in targets.items():
        path = asset_dir / f"frontend_{key}.png"
        if editor is not None:
            try:
                from .. import devtools as dt

                dt.screenshot_tab(editor, tab, path, settle_ms=700)
                out[key] = path
                continue
            except Exception:
                pass
        out[key] = _placeholder_image(path, caption)
    out["gui_dot"] = _scan_dot_figure(asset_dir / "frontend_gui_dot.png")
    out.update(_region_figures(editor, asset_dir))
    return out


def _region_figures(editor, asset_dir: Path) -> dict[str, Path]:
    """Grab cropped close-ups of individual Edit-tab regions for the manual."""

    regions: dict[str, tuple[object, str]] = {}
    if editor is not None:
        try:
            from .. import devtools as dt

            editor.tabs.setCurrentWidget(editor.edit_tab)
            dt.settle(editor, 500)
            cards = editor.drag_container.pulse_cards()
            regions = {
                "gui_names": (getattr(editor, "names_panel", None), "Channel Names 面板"),
                "gui_delay": (getattr(editor, "channel_panel", None), "Delay / Scan 面板"),
                "gui_period": (cards[1] if len(cards) > 1 else None, "一张 period 卡片"),
                "gui_bottombar": (getattr(editor, "button_frame", None), "底部控制条"),
            }
        except Exception:
            regions = {}
    out: dict[str, Path] = {}
    captions = {
        "gui_names": "Channel Names 面板（截图不可用）",
        "gui_delay": "Delay / Scan 面板（截图不可用）",
        "gui_period": "period 卡片（截图不可用）",
        "gui_bottombar": "底部控制条（截图不可用）",
    }
    for key, default_caption in captions.items():
        path = asset_dir / f"frontend_{key}.png"
        widget = regions.get(key, (None, default_caption))[0]
        if widget is not None:
            try:
                from .. import devtools as dt

                dt.settle(widget, 400)
                widget.grab().save(str(path))
                out[key] = path
                continue
            except Exception:
                pass
        out[key] = _placeholder_image(path, default_caption)
    return out


def _scan_dot_figure(path: Path) -> Path:
    """Close-up of an unbound vs. bound scan field (the inline dot button)."""

    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5 import QtWidgets

        from .. import devtools as dt
        from ..qt_fluent import FluentScanLineEdit, Metrics, ensure_qt_app, set_fluent_scale

        ensure_qt_app()
        dt.install_screenshot_font()
        set_fluent_scale(3.0)  # zoom so the dot geometry reads clearly in print
        holder = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(holder)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(20)
        unbound = FluentScanLineEdit("20000")
        bound = FluentScanLineEdit("20000")
        bound.set_scan_bound(True, 2)
        for widget in (unbound, bound):
            widget.setFixedSize(440, Metrics.row_h())
            lay.addWidget(widget)
        holder.show()
        dt.settle(holder, 600)
        holder.grab().save(str(path))
        set_fluent_scale(1.0)
        return path
    except Exception:
        return _placeholder_image(path, "扫描圆点：未绑定（空心）/ 已绑定（橙点+编号）")


def _schematic_axes(figsize):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from .. import style as S
    from ..devtools import install_screenshot_font

    install_screenshot_font()
    S.apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    return plt, fig, ax


def _render_display_chain(path: Path) -> Path:
    """The 2D live-image display chain: full-res array -> crop+block-mean to the axes pixel
    budget -> imshow('auto') -> Agg buffer at the screen's device px (1:1 blit) -> screen.  ONE
    resample; the block-mean is an area average.  English labels; Chinese teaching in the caption."""

    try:
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        plt, fig, ax = _schematic_axes((7.6, 2.6))
        ink, accent, boxfc = "#222222", "#c0563f", "#f0efec"
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 26)

        def box(x, w, text, *, fc=boxfc, ec=ink, tc=ink):
            ax.add_patch(FancyBboxPatch((x - w / 2, 13.75), w, 6.5, boxstyle="round,pad=0.15,rounding_size=0.5",
                                        fc=fc, ec=ec, lw=1.0, zorder=3))
            ax.text(x, 17, text, ha="center", va="center", fontsize=7.0, color=tc, zorder=4, linespacing=1.2)

        xs = [11, 33, 55, 77, 94]
        ws = [17, 19, 15, 17, 10]
        box(xs[0], ws[0], "full-res array\n1920x1200\n(kept in memory)", fc="#eef2f4")
        box(xs[1], ws[1], "crop to view +\nblock-mean to the\naxes pixel budget", fc="#fbf3ee", ec=accent, tc=accent)
        box(xs[2], ws[2], "imshow\n(interpolation\n= 'auto')")
        box(xs[3], ws[3], "Agg buffer\n= screen device px\n(1:1 blit)")
        box(xs[4], ws[4], "screen")
        for a, wa, b, wb in zip(xs[:-1], ws[:-1], xs[1:], ws[1:]):
            ax.add_patch(FancyArrowPatch((a + wa / 2, 17), (b - wb / 2, 17), arrowstyle="-|>",
                                         mutation_scale=11, color=ink, lw=1.2, zorder=5, shrinkA=1, shrinkB=1))
        ax.text(33, 9.5, "zoom re-slices from the full array -> detail returns until 1:1",
                ha="center", va="center", fontsize=6.4, color=accent)
        ax.text(50, 3.0, "ONE resample in the whole chain -- the block-mean is an area average "
                "(it IMPROVES SNR, unlike nearest subsampling)", ha="center", va="center",
                fontsize=6.8, color=ink, bbox=dict(boxstyle="round,pad=0.45", fc="#fbf3ee", ec=accent, lw=0.8))
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return path
    except Exception:  # pragma: no cover - keep the manual buildable
        return _placeholder_image(path, "2D 显示链:全分辨率数组 -> 块平均到轴像素预算 -> 1:1 blit -> 屏幕")


def _render_drag_placement(path: Path) -> Path:
    """The console drag-placement rule: the dragged card's top-left corner competes over
    {existing card corners = displace} vs {free lattice anchors = land} by nearest distance,
    then NW gravity compacts.  English labels; Chinese teaching in the caption."""

    try:
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

        plt, fig, ax = _schematic_axes((7.2, 3.4))
        ink, accent, faint, green = "#222222", "#c0563f", "#8a8a8a", "#3a7d44"
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 44)

        def card(x, y, text):
            ax.add_patch(FancyBboxPatch((x - 11, y - 6), 22, 12, boxstyle="round,pad=0.15,rounding_size=0.6",
                                        fc="#eef2f4", ec=ink, lw=1.0, zorder=3))
            ax.text(x, y, text, ha="center", va="center", fontsize=7.4, color=ink, zorder=4)

        def dashline(x0, y0, x1, y1, color):
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-", color=color, lw=0.8,
                                         ls=(0, (2, 2)), zorder=2, shrinkA=1, shrinkB=1))

        ax.add_patch(Rectangle((6, 4), 88, 36, fill=False, ec=faint, lw=0.8, ls=(0, (4, 3)), zorder=1))
        card(20, 31, "card A")
        card(20, 14, "card B")
        ax.add_patch(FancyBboxPatch((44, 24), 22, 12, boxstyle="round,pad=0.15,rounding_size=0.6",
                                    fc="none", ec=accent, lw=1.4, ls=(0, (3, 2)), zorder=3))
        ax.text(55, 30, "dragged\n(ghost)", ha="center", va="center", fontsize=7.0, color=accent, linespacing=1.2)
        dashline(44, 36, 10.5, 37, ink)
        dashline(44, 36, 68.5, 37, green)
        ax.plot(44, 36, "o", ms=6, color=accent, zorder=6)
        ax.text(40, 41.5, "top-left corner (the judged point)", ha="right", va="bottom",
                fontsize=6.3, color=accent)
        for cx, cy in ((9, 37), (9, 20)):
            ax.plot(cx, cy, "s", ms=5, color=ink, zorder=6)
        ax.text(14, 42.6, "existing corners = DISPLACE that card", ha="center", va="bottom",
                fontsize=6.3, color=ink)
        ax.plot(70, 37, "^", ms=6, color=green, zorder=6)
        ax.text(72, 37, "free lattice anchor\n= LAND here", ha="left", va="center", fontsize=6.3,
                color=green, linespacing=1.1)
        ax.add_patch(FancyArrowPatch((88, 10), (80, 6.5), arrowstyle="-|>", mutation_scale=11,
                                     color=faint, lw=1.2, zorder=5))
        ax.text(88, 12, "then NW gravity compacts\n(never teleports past a blocker)", ha="right",
                va="bottom", fontsize=6.3, color=faint, linespacing=1.1)
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return path
    except Exception:  # pragma: no cover - keep the manual buildable
        return _placeholder_image(path, "拖拽落位:被拖卡左上角对 {已存卡角=挤走} 与 {空缺锚=落位} 比最近距离，再 NW 重力压实")


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

    figures = {"one_d": one_d, "two_d": two_d, "hist": hist_fig}
    figures["display_chain"] = _render_display_chain(asset_dir / "frontend_display_chain.png")
    figures["drag_place"] = _render_drag_placement(asset_dir / "frontend_drag_place.png")
    figures.update(generate_frontend_gui_figures(asset_dir))
    return figures


def frontend_manual_body(figures: Mapping[str, Path]) -> str:
    body = (_CONTENT_DIR / "frontend_manual_zh.texbody").read_text(encoding="utf-8")
    body = (
        body.replace("__ONE_D__", figures["one_d"].as_posix())
        .replace("__TWO_D__", figures["two_d"].as_posix())
        .replace("__HIST__", figures["hist"].as_posix())
    )
    for key, token in (
        ("gui_edit", "__GUI_EDIT__"),
        ("gui_preview", "__GUI_PREVIEW__"),
        ("gui_scan", "__GUI_SCAN__"),
        ("gui_dot", "__GUI_DOT__"),
        ("gui_names", "__GUI_NAMES__"),
        ("gui_delay", "__GUI_DELAY__"),
        ("gui_period", "__GUI_PERIOD__"),
        ("gui_bottombar", "__GUI_BOTTOMBAR__"),
        ("display_chain", "__DISPLAY_CHAIN__"),
        ("drag_place", "__DRAG_PLACE__"),
    ):
        if key in figures:
            body = body.replace(token, figures[key].as_posix())
    return body


__all__ = ["frontend_manual_body", "generate_frontend_manual_figures", "generate_frontend_gui_figures"]
