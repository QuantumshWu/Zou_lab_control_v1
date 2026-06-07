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


def _render_pulse_png(path: Path, sequence, *, channels=None, caption=None, **fig_kwargs) -> Path:
    """Render a real pulse via the FRONTEND pulse plotter (PulseSequenceFigure) to
    a PNG -- the same matplotlib figure the GUI preview draws, not an ASCII/TikZ
    sketch.  Falls back to a placeholder image so the manual still builds."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from Zou_lab_control.frontend.devtools import install_screenshot_font
        from Zou_lab_control.frontend.live import PulseSequenceFigure

        install_screenshot_font()
        fig = PulseSequenceFigure(sequence, channels=channels, **fig_kwargs).show(display=False)
        fig.fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig.fig)
        return path
    except Exception:  # pragma: no cover - keep the manual buildable
        return _device_placeholder_image(path, caption or "脉冲时序")


def generate_fpga_manual_figures(asset_dir: str | Path) -> dict[str, Path]:
    """Render the FPGA-manual TIMING figures as REAL pulses via the frontend plotter
    (PulseSequenceFigure), into asset_dir.  Returns {key: png path}."""

    import numpy as np

    from Zou_lab_control import neutral_atom as na

    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, Path] = {}

    # (1) 20 ns / 1-tick resolution: back-to-back 1-tick edges, one per 20 ns tick.
    tick = na.PulseTableState(
        channels=["ch00", "ch01", "ch02"],
        channel_labels={"ch00": "cooling", "ch01": "probe", "ch02": "trig"},
        visible_channels=["ch00", "ch01", "ch02"],
        time_step_ns=20,
        periods=[na.PulsePeriod(20, (1, 0, 0), unit="ns"), na.PulsePeriod(20, (0, 1, 0), unit="ns"),
                 na.PulsePeriod(20, (0, 0, 1), unit="ns"), na.PulsePeriod(20, (0, 0, 0), unit="ns")],
    )
    figs["tick"] = _render_pulse_png(asset_dir / "fpga_1tick.png", tick.to_sequence(),
                                     channels=["ch00", "ch01", "ch02"], channel_labels=tick.channel_labels,
                                     show_names=True, caption="背靠背 1-tick 脉冲")

    # (2) affine scan: the SAME pulse rendered at two scan points -- the scanned
    # middle period slides the later edge in lockstep.
    def scan_state(mid_ns):
        return na.PulseTableState(
            channels=["ch00", "ch01"], channel_labels={"ch00": "cooling", "ch01": "probe"},
            visible_channels=["ch00", "ch01"], time_step_ns=20,
            periods=[na.PulsePeriod(60, (1, 0), unit="ns"), na.PulsePeriod(mid_ns, (0, 1), unit="ns"),
                     na.PulsePeriod(60, (0, 0), unit="ns")])
    figs["scan_lo"] = _render_pulse_png(asset_dir / "fpga_scan_lo.png", scan_state(40).to_sequence(),
                                        channels=["ch00", "ch01"], show_names=True, caption="scan 点 0")
    figs["scan_hi"] = _render_pulse_png(asset_dir / "fpga_scan_hi.png", scan_state(160).to_sequence(),
                                        channels=["ch00", "ch01"], show_names=True, caption="scan 点 N")

    # (3) hardware loop / repeat-forever: the loop body with a repeat bracket.
    rep = na.PulseTableState(
        channels=["ch00", "ch01"], channel_labels={"ch00": "load", "ch01": "trig"},
        visible_channels=["ch00", "ch01"], time_step_ns=20,
        periods=[na.PulsePeriod(40, (1, 0), unit="ns"), na.PulsePeriod(40, (0, 1), unit="ns"),
                 na.PulsePeriod(40, (0, 0), unit="ns")])
    rep_seq = rep.to_sequence(expand_repeat=False)
    dur = max((float(r["stop"]) for r in __import__("Zou_lab_control.frontend.live", fromlist=["_pulse_rows"])._pulse_rows(rep_seq)), default=120e-9)
    # in-figure text must be ASCII (the bundled DejaVu Sans has no CJK glyphs); the
    # Chinese explanation lives in the LaTeX \caption rendered by xelatex.
    figs["repeat"] = _render_pulse_png(asset_dir / "fpga_repeat.png", rep_seq, channels=["ch00", "ch01"],
                                       show_names=True, repeat_bracket=(0.0, dur, "repeat x N (HW loop, seamless)"),
                                       caption="硬件重复 loop")

    # (4) analog DAC ramp via the frontend analog trace: hold 0 -> ramp 0..1023 -> hold.
    ramp_t = list(np.linspace(1e-6, 3e-6, 21))
    ramp_v = [int(v) for v in np.linspace(0, 1023, 21)]
    starts = [0.0, 1e-6] + ramp_t[1:] + [4e-6]
    values = [0] + ramp_v[1:] + [1023]
    dac_trace = {"name": "da_dipole", "label": "da_dipole (DAC)", "members": [f"d{i}" for i in range(10)],
                 "max": 1023, "starts": starts, "values": values}
    # start + end marker pulses so the timeline x-axis spans the full 0..4 us and the
    # ramp (1..3 us) is on-screen.
    dctx = na.PulseTableState(
        channels=["ch00"], channel_labels={"ch00": "trig"}, visible_channels=["ch00"], time_step_ns=20,
        periods=[na.PulsePeriod(40, (1,), unit="ns"), na.PulsePeriod(3920, (0,), unit="ns"),
                 na.PulsePeriod(40, (1,), unit="ns")])
    figs["dac"] = _render_pulse_png(asset_dir / "fpga_dac.png", dctx.to_sequence(), channels=["ch00"],
                                    show_names=True, analog_traces=[dac_trace], caption="DAC ramp")
    return figs


def _fpga_figure_tex(fig_path: str, caption: str, *, width: float = 0.8) -> str:
    return (
        "\\begin{figure}[h]\n\\centering\n"
        f"\\includegraphics[width={width}\\linewidth]{{{fig_path}}}\n"
        f"\\caption{{{caption}}}\n\\end{{figure}}"
    )


def _fpga_two_figure_tex(fig_a: str, fig_b: str, caption: str) -> str:
    return (
        "\\begin{figure}[h]\n\\centering\n"
        f"\\includegraphics[width=0.48\\linewidth]{{{fig_a}}}\\hfill\n"
        f"\\includegraphics[width=0.48\\linewidth]{{{fig_b}}}\n"
        f"\\caption{{{caption}}}\n\\end{{figure}}"
    )


def main_manual_body() -> str:
    """Return the main (system-overview) manual body.

    The main manual uses inline TikZ diagrams only, so no figure files are
    required; the template compiles as-is.
    """

    return _template_text("main_manual_zh.texbody")


def fpga_manual_body(figures: Mapping[str, Path] | None = None) -> str:
    """Return the FPGA (pulse-streamer) manual body.

    The TIMING diagrams are REAL pulses rendered by the frontend plotter
    (:func:`generate_fpga_manual_figures`).  ``figures`` injects those PNGs; omit
    them and each placeholder is dropped so the text still compiles (block-diagram
    TikZ stays inline).
    """

    body = _template_text("fpga_manual_zh.texbody")
    figures = figures or {}

    def fig(key, caption, width=0.8):
        p = figures.get(key)
        return _fpga_figure_tex(Path(p).as_posix(), caption, width=width) if p else ""

    body = body.replace("__FPGA_FIG_TICK__", fig(
        "tick", "前端脉冲实绘：三路通道在相邻 20\\,ns tick 上背靠背切换——引擎的最小脉宽与分辨率就是 1 个 tick；"
        "预取流水线让这些 1-tick 边沿逐拍打出，中间无空拍。"))
    sa, sb = figures.get("scan_lo"), figures.get("scan_hi")
    body = body.replace("__FPGA_FIG_SCAN__", _fpga_two_figure_tex(
        Path(sa).as_posix(), Path(sb).as_posix(),
        "前端脉冲实绘：同一脉冲在两个 scan 点的渲染。被扫的中间周期把后面的边沿在硬件里 lockstep 平移；"
        "扫描点之间的切换是无缝的（边界影子重装）。") if (sa and sb) else "")
    body = body.replace("__FPGA_FIG_REPEAT__", fig(
        "repeat", "前端脉冲实绘：硬件重复 loop 的循环体（不展开）。\\pyapi{repeat\\_forever} 在硬件里无缝回绕，"
        "重复之间不留缝。"))
    body = body.replace("__FPGA_FIG_DAC__", fig(
        "dac", "前端脉冲实绘：模拟总线 DAC 波形——保持 0、斜坡 0$\\to$1023、保持 1023。引擎在本地按 tick 插值生成 "
        "10-bit 阶梯;双 value\\_select 允许斜坡两端各跟一个 scan slot。"))
    return body


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
    "generate_fpga_manual_figures",
    "fpga_manual_body",
    "main_manual_body",
]
