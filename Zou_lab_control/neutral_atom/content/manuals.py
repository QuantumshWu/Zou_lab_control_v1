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

        # Fixed seed so the manual's figure is REPRODUCIBLE -- a doc rebuild must not churn the
        # committed PNG with a fresh random load pattern (the figure is illustrative, not a result).
        exp = na.connect("virtual", seed=0)
        exp.readout.sitemap(frames=6, display=False)
        threshold_result = exp.readout.thresholds(frames=120, site=0, display=False)
        plot = threshold_result.plot_site(0, display=False)
        plot.fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(plot.fig)
        return path
    except Exception:  # pragma: no cover - defensive: keep the manual buildable
        return _device_placeholder_image(path, "阈值直方图（暗/亮双峰 + Otsu 阈值）")


def _render_grabber_timing(path: Path) -> Path:
    """Schematic of the arm-before-fire contract: three lanes (measurement / camera /
    sequencer) on one timeline.  In-figure labels are English/code terms (Helvetica
    Light carries no CJK glyphs); the Chinese teaching text is in the LaTeX caption."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        from Zou_lab_control.frontend import style as S
        from Zou_lab_control.frontend.devtools import install_screenshot_font

        install_screenshot_font()
        S.apply_style()
        ink, accent, faint, boxfc = "#222222", "#c0563f", "#8a8a8a", "#f0efec"

        fig, ax = plt.subplots(figsize=(7.6, 3.4))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 30)
        ax.axis("off")
        for label, y in (("measurement\n(triggered_frames)", 23.5), ("camera\n(pure grabber)", 15.0),
                         ("sequencer\n(pure streamer)", 6.5)):
            ax.axhline(y, xmin=0.17, xmax=0.99, color=faint, lw=0.7, zorder=1)
            ax.text(1.0, y, label, ha="left", va="center", fontsize=7.6, color=ink, linespacing=1.3)

        def box(x, y, w, h, text, *, fc=boxfc, ec=ink, tc=ink, fs=7.5):
            ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h,
                                        boxstyle="round,pad=0.15,rounding_size=0.6",
                                        fc=fc, ec=ec, lw=1.0, zorder=3))
            ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4)

        def arrow(x0, y0, x1, y1, *, color=ink, style="-|>", lw=1.1, ls="-"):
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=10,
                                         color=color, lw=lw, ls=ls, zorder=5, shrinkA=0, shrinkB=0))

        fx_all = [44, 55, 66, 77]
        box(18, 23.5, 13, 4.2, "arm(N)", fc="#e8eef2")
        box(75, 23.5, 20, 4.2, "read_frames(N)", fc="#e8eef2")
        box(18, 15.0, 13, 3.6, "armed:\nwaiting", fc=boxfc, fs=6.8)
        for i, fx in enumerate(fx_all):
            box(fx, 15.0, 8.2, 3.2, f"frame {i + 1}", fc="#fbf3ee", ec=accent, tc=accent, fs=6.6)
        box(31, 6.5, 14, 3.6, "prepare + fire", fc=boxfc, fs=6.8)
        for fx in fx_all:
            ax.plot([fx, fx], [5.1, 7.9], color=accent, lw=1.6, zorder=4)
        ax.text(fx_all[0], 9.0, "trigger edges", ha="left", va="bottom", fontsize=6.2, color=accent)
        arrow(24.5, 21.4, 21, 16.8, color=ink)
        ax.text(16.5, 19.6, "arm() returns:\nhardware ready", ha="right", va="center",
                fontsize=6.4, color=ink, linespacing=1.2)
        arrow(24.5, 21.4, 31, 8.3, color=faint, style="-|>", ls=(0, (3, 2)))
        ax.text(33, 11.4, "fire always\nafter arm", ha="left", va="center", fontsize=6.4,
                color=faint, linespacing=1.2)
        for fx in fx_all:
            arrow(fx, 8.0, fx, 13.4, color=accent, lw=1.0)
        arrow(fx_all[-1] + 5, 15.0, 69, 21.4, color=ink, style="-|>")
        ax.text(71, 18.6, "lossless buffer:\na late read keeps every frame", ha="left",
                va="center", fontsize=6.4, color=ink, linespacing=1.2)
        ax.text(50, 1.4, "N frames  =  N trigger periods    "
                "(triggered_frames repeats the base cycle to carry N edges)",
                ha="center", va="center", fontsize=7.2, color=accent,
                bbox=dict(boxstyle="round,pad=0.4", fc="#fbf3ee", ec=accent, lw=0.8))
        arrow(17, 29.0, 99, 29.0, color=faint, lw=0.8)
        ax.text(99, 29.0, "time", ha="right", va="bottom", fontsize=6.3, color=faint)

        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return path
    except Exception:  # pragma: no cover - keep the manual buildable
        return _device_placeholder_image(path, "arm -> fire N edges -> read_frames（纯 grabber 时序）")


def _render_trigger_cable(path: Path) -> Path:
    """Virtual==real parity: the measurement/analysis stack is ONE object over both a real
    (sequencer --copper trigger line--> camera) and a virtual (VirtualSequencer --fire notify /
    firing pull--> VirtualCamera, config \"$device:sequencer\") bottom layer.  English labels;
    Chinese teaching in the LaTeX caption."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        from Zou_lab_control.frontend import style as S
        from Zou_lab_control.frontend.devtools import install_screenshot_font

        install_screenshot_font()
        S.apply_style()
        ink, accent, faint, boxfc = "#222222", "#c0563f", "#8a8a8a", "#f0efec"

        fig, ax = plt.subplots(figsize=(7.6, 3.6))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 34)
        ax.axis("off")

        def box(x, y, w, h, text, *, fc=boxfc, ec=ink, tc=ink, fs=7.2):
            ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                        boxstyle="round,pad=0.15,rounding_size=0.6",
                                        fc=fc, ec=ec, lw=1.0, zorder=3))
            ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, zorder=4, linespacing=1.2)

        def arrow(x0, y0, x1, y1, *, color=ink, style="-|>", lw=1.2, ls="-"):
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11,
                                         color=color, lw=lw, ls=ls, zorder=5, shrinkA=1, shrinkB=1))

        ax.add_patch(FancyBboxPatch((14, 27.0), 72, 5.0, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc="#eef2f4", ec=ink, lw=1.1, zorder=2))
        ax.text(50, 29.5, "measurement / analysis  (session, subsystems, operations)\n"
                "-- identical object, same triggered_frames path in both --",
                ha="center", va="center", fontsize=7.2, color=ink, linespacing=1.35)
        ax.text(28, 24.0, "REAL hardware", ha="center", va="center", fontsize=7.8, color=ink, weight="bold")
        ax.text(74, 24.0, "VIRTUAL (only the bottom layer is faked)", ha="center", va="center",
                fontsize=7.8, color=accent, weight="bold")
        ax.axvline(51, ymin=0.03, ymax=0.68, color=faint, lw=0.7, ls=(0, (2, 3)))

        box(16, 15.5, 17, 4.4, "SequencerDevice\n(FPGA streamer)")
        box(40, 15.5, 15, 4.4, "CameraDevice\n(qCMOS / Basler)")
        arrow(24.6, 15.5, 32.4, 15.5, color=accent, lw=1.6)
        ax.text(28.5, 17.3, "copper\ntrigger line", ha="center", va="bottom", fontsize=6.2,
                color=accent, linespacing=1.1)
        arrow(16, 26.9, 16, 17.8, color=faint, style="<->", lw=0.9)
        arrow(40, 26.9, 40, 17.8, color=faint, style="<->", lw=0.9)

        box(63, 15.5, 17, 4.4, "VirtualSequencer")
        box(87, 15.5, 15, 4.4, "VirtualCamera")
        arrow(71.6, 16.6, 79.4, 16.6, color=accent, lw=1.4)
        ax.text(75.5, 18.2, "fire notify /\nfiring pull", ha="center", va="bottom", fontsize=6.2,
                color=accent, linespacing=1.1)
        ax.text(75.5, 12.6, '(config: "$device:sequencer")', ha="center", va="top", fontsize=6.0, color=faint)
        arrow(63, 26.9, 63, 17.8, color=faint, style="<->", lw=0.9)
        arrow(87, 26.9, 87, 17.8, color=faint, style="<->", lw=0.9)

        ax.text(50, 5.2, "The camera images the atoms the sequencer's edges gate -- it never drives the "
                "sequencer.\nThe virtual camera renders exactly the edges the fired program carries "
                "(armed only) -- an unarmed camera misses them, like hardware.",
                ha="center", va="center", fontsize=6.9, color=ink, linespacing=1.4,
                bbox=dict(boxstyle="round,pad=0.5", fc="#fbf3ee", ec=accent, lw=0.8))

        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return path
    except Exception:  # pragma: no cover - keep the manual buildable
        return _device_placeholder_image(path, "虚拟触发电缆:真机铜线 vs 虚拟 fire 通知(上层同栈)")


def generate_device_manual_figures(asset_dir: str | Path) -> dict[str, Path]:
    """Render the device-manual figures (real tutorial output) into asset_dir."""

    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    return {
        "threshold_hist": _render_threshold_hist(asset_dir / "device_threshold_hist.png"),
        "grabber_timing": _render_grabber_timing(asset_dir / "device_grabber_timing.png"),
        "trigger_cable": _render_trigger_cable(asset_dir / "device_trigger_cable.png"),
        "device_manager": _render_device_manager_shot(asset_dir / "device_manager_shot.png"),
    }


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


def _grabber_figure_tex(fig_path: str) -> str:
    caption = (
        "\\tfocus{纯 grabber 的 arm-before-fire 时序}（三泳道：measurement / camera / sequencer，"
        "同一条时间轴）。相机先 \\pyapi{arm(N)}，\\tfocus{返回时硬件已就绪、等外触发}——所以 fire "
        "\\tfocus{永远}发生在 arm 返回之后，第一个触发边沿绝不丢失。随后 measurement 层的 "
        "\\pyapi{triggered\\_frames} 让序列器 \\pyapi{prepare + fire} 一段\\tfocus{携带 N 个触发边沿}"
        "的程序（把成像基周期重复到 N 个边沿）：每个边沿门控一帧进相机\\tfocus{自有的无损缓冲}，"
        "\\pyapi{read\\_frames(N)} 再把它们取走——即使先 fire、后取帧，缓冲也一帧不丢。一句话记住："
        "\\tfocus{N 帧 = N 个触发周期}；相机从不驱动序列器，只按到达的边沿取帧（虚拟后端亦然，"
        "只是把这根触发线也仿真了）。"
    )
    return (
        "\\begin{figure}[h]\n\\centering\n"
        f"\\includegraphics[width=0.92\\linewidth]{{{fig_path}}}\n"
        f"\\caption{{{caption}}}\n"
        "\\end{figure}"
    )


def _cable_figure_tex(fig_path: str) -> str:
    caption = (
        "\\tfocus{虚拟 == 实机：只有最底层被仿真}。上方的 measurement / analysis 栈（session / "
        "subsystems / operations，含唯一编排 \\pyapi{triggered\\_frames}）在真机与虚拟下是\\tfocus{同一个对象、"
        "走同一条代码路径}。真机里序列器用一根\\tfocus{铜触发线}接到相机；虚拟里 \\pyapi{VirtualSequencer} 的 "
        "fire 通知 / firing 拉取这根\\tfocus{被仿真的触发线}驱动 \\pyapi{VirtualCamera}（config 用 "
        "\\pyapi{\"\\$device:sequencer\"} 接线）。两边相机都\\tfocus{只按到达的触发边沿成像、从不驱动序列器}；"
        "虚拟相机严格按 fired 程序在其触发线上\\tfocus{真实携带}的边沿数渲染帧，\\tfocus{未武装就漏掉边沿}——"
        "和真机一模一样。所以换真机只改 \\pyapi{connect()}，上面一行分析代码都不用动。"
    )
    return (
        "\\begin{figure}[h]\n\\centering\n"
        f"\\includegraphics[width=0.92\\linewidth]{{{fig_path}}}\n"
        f"\\caption{{{caption}}}\n"
        "\\end{figure}"
    )


def _render_device_manager_shot(path: Path) -> Path:
    """Real screenshot of the device-manager GUI (``exp.device_manager()``): ONE section per registered
    device domain, driven by the device registry, PLUS the config toolbar (Open devices / Load config /
    Save config) the session wires in.  Built on a real virtual session so it renders with no hardware and
    shows EXACTLY what ``exp.device_manager()`` opens -- same callbacks, same buttons (this doc-build tool
    may reach the frontend, like the other GUI figures here; the runtime analysis path stays sealed)."""
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import Zou_lab_control.neutral_atom as na
        from Zou_lab_control.frontend import devtools as dt
        from Zou_lab_control.frontend.device_manager import DeviceManagerPanel
        from Zou_lab_control.frontend.devtools import install_screenshot_font
        from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
        from Zou_lab_control.neutral_atom.devices.registry import device_config_dir

        ensure_qt_app()
        install_screenshot_font()
        exp = na.connect("virtual")
        try:
            # Wire the SAME session binding exp.device_manager() does, so the figure shows the
            # real editor + Loaded card (not a bare session-less editor).
            from Zou_lab_control.neutral_atom._gui import _session_device_binding

            panel = DeviceManagerPanel(
                exp.devices, session_binding=_session_device_binding(exp),
                config_dir=str(device_config_dir()))
            panel.resize(960, 660)
            panel.show()
            dt.settle(panel, 500)
            panel.grab().save(str(path))
        finally:
            exp.close()
        return path
    except Exception:
        return _device_placeholder_image(
            path, "设备管理器 GUI：按域分区 + Open/Load/Save config 工具栏 + Scan hardware")


def _device_manager_figure_tex(fig_path: str) -> str:
    caption = (
        "\\tfocus{设备管理器 GUI}（\\pyapi{exp.device\\_manager()}）：把 \\pyapi{connect} / "
        "\\pyapi{load\\_devices} 加载出的每个设备\\tfocus{按域分区}列出（Camera / Sequencer / Trap array / "
        "未来的 RF 源——与逐测量的设备下拉读的是\\tfocus{同一个注册表}），底部 \\pyapi{Scan hardware} 按钮"
        "现场探一遍总线并把发现的设备补一张卡。顶部的\\tfocus{配置工具栏}——绿色 \\pyapi{Open devices}（初始化硬件）、"
        "\\pyapi{Load config\\ldots} / \\pyapi{Save config\\ldots}——就是会话 \\pyapi{exp.open\\_devices()} / "
        "\\pyapi{exp.load\\_config()} / \\pyapi{exp.save\\_config()} 的图形面（存一份配置、下次一行连回来）。它是 "
        "\\pyapi{load\\_devices} / \\pyapi{discover\\_devices} 的\\tfocus{GUI 面孔}。两个入口："
        "\\pyapi{exp.device\\_manager()}（\\tfocus{会话绑定}，编辑与换设备）和 \\pyapi{na.device\\_manager(config)}"
        "（还没有 session 时的\\tfocus{初始化}入口：按 \\pyapi{Init devices} 即 \\pyapi{connect}，\\pyapi{window.session} "
        "把新会话交回 notebook）。监控台顶栏的 \\pyapi{Devices} 按钮开的是\\tfocus{只读查看器}"
        "（\\pyapi{exp.device\\_viewer()}，快照 + 运行时读回，\\tfocus{不}编辑、\\tfocus{不}换设备），而非本编辑器——"
        "运行中改设备是危险的。本编辑器天生依赖会话的 \\pyapi{DeviceSet}（frontend 被密封、拿不到设备），"
        "所以它\\tfocus{不}是 \\pyapi{zf} 模块级函数；建 session 的模块级入口是 \\pyapi{na.device\\_manager}。"
    )
    return (
        "\\begin{figure}[h]\n\\centering\n"
        f"\\includegraphics[width=0.5\\linewidth]{{{fig_path}}}\n"
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
    tick_catalog = na.PortCatalog.from_channels(
        ["ch00", "ch01", "ch02"],
        channel_labels={"ch00": "cooling", "ch01": "probe", "ch02": "trig"},
    )
    tick = na.PulseTableState(
        port_catalog=tick_catalog,
        visible_ports=["ch00", "ch01", "ch02"],
        time_step_ns=20,
        periods=[na.PulsePeriod(20, (1, 0, 0), unit="ns"), na.PulsePeriod(20, (0, 1, 0), unit="ns"),
                 na.PulsePeriod(20, (0, 0, 1), unit="ns"), na.PulsePeriod(20, (0, 0, 0), unit="ns")],
    )
    figs["tick"] = _render_pulse_png(asset_dir / "fpga_1tick.png", tick.to_sequence(),
                                     channels=["ch00", "ch01", "ch02"],
                                     channel_labels=tick.port_catalog.channel_labels,
                                     show_names=True, caption="背靠背 1-tick 脉冲")

    # (2) affine scan: the SAME pulse rendered at two scan points -- the scanned
    # middle period slides the later edge in lockstep.
    def scan_state(mid_ns):
        catalog = na.PortCatalog.from_channels(
            ["ch00", "ch01"],
            channel_labels={"ch00": "cooling", "ch01": "probe"},
        )
        return na.PulseTableState(
            port_catalog=catalog, visible_ports=["ch00", "ch01"], time_step_ns=20,
            periods=[na.PulsePeriod(60, (1, 0), unit="ns"), na.PulsePeriod(mid_ns, (0, 1), unit="ns"),
                     na.PulsePeriod(60, (0, 0), unit="ns")])
    figs["scan_lo"] = _render_pulse_png(asset_dir / "fpga_scan_lo.png", scan_state(40).to_sequence(),
                                        channels=["ch00", "ch01"], show_names=True, caption="scan 点 0")
    figs["scan_hi"] = _render_pulse_png(asset_dir / "fpga_scan_hi.png", scan_state(160).to_sequence(),
                                        channels=["ch00", "ch01"], show_names=True, caption="scan 点 N")

    # (3) hardware loop / repeat-forever: the loop body with a repeat bracket.
    rep = na.PulseTableState(
        port_catalog=na.PortCatalog.from_channels(
            ["ch00", "ch01"],
            channel_labels={"ch00": "load", "ch01": "trig"},
        ),
        visible_ports=["ch00", "ch01"], time_step_ns=20,
        periods=[na.PulsePeriod(40, (1, 0), unit="ns"), na.PulsePeriod(40, (0, 1), unit="ns"),
                 na.PulsePeriod(40, (0, 0), unit="ns")])
    rep_seq = rep.to_sequence(expand_repeat=False)
    # Use the sequence's own duration (timing layer) rather than reaching into a
    # private frontend symbol -- the manual generator must not depend on
    # frontend internals (decoupling; see AGENTS.md §2).
    dur = float(getattr(rep_seq, "duration", 0.0)) or 120e-9
    # in-figure text must be ASCII (the bundled DejaVu Sans has no CJK glyphs); the
    # Chinese explanation lives in the LaTeX \caption rendered by xelatex.
    figs["repeat"] = _render_pulse_png(asset_dir / "fpga_repeat.png", rep_seq, channels=["ch00", "ch01"],
                                       show_names=True, repeat_bracket=(0.0, dur, "repeat x N (HW loop, seamless)"),
                                       caption="硬件重复 loop")

    # (4) analog DAC ramp via the frontend analog trace: rest at 0 V -> ramp the SIGNED
    # value -512..+511 -> hold (the user layer is signed LSB; 0 V sits mid-row).
    ramp_t = list(np.linspace(1e-6, 3e-6, 21))
    ramp_v = [int(v) for v in np.linspace(-512, 511, 21)]
    starts = [0.0, 1e-6] + ramp_t[1:] + [4e-6]
    values = [0] + ramp_v[1:] + [511]
    dac_trace = {"name": "da_dipole", "label": "da_dipole (DAC)", "members": [f"d{i}" for i in range(10)],
                 "min": -512, "max": 511, "starts": starts, "values": values}
    # start + end marker pulses so the timeline x-axis spans the full 0..4 us and the
    # ramp (1..3 us) is on-screen.
    dctx = na.PulseTableState(
        port_catalog=na.PortCatalog.from_channels(
            ["ch00"], channel_labels={"ch00": "trig"}),
        visible_ports=["ch00"], time_step_ns=20,
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
    body = body.replace("__READOUT_THRESHOLD_FIG__", figure_tex)
    grab_path = None if not figures else figures.get("grabber_timing")
    grab_tex = _grabber_figure_tex(Path(grab_path).as_posix()) if grab_path else ""
    body = body.replace("__CAMERA_GRABBER_FIG__", grab_tex)
    cable_path = None if not figures else figures.get("trigger_cable")
    cable_tex = _cable_figure_tex(Path(cable_path).as_posix()) if cable_path else ""
    body = body.replace("__CAMERA_CABLE_FIG__", cable_tex)
    dm_path = None if not figures else figures.get("device_manager")
    dm_tex = _device_manager_figure_tex(Path(dm_path).as_posix()) if dm_path else ""
    return body.replace("__DEVICE_MANAGER_FIG__", dm_tex)


__all__ = [
    "device_manual_body",
    "generate_device_manual_figures",
    "generate_fpga_manual_figures",
    "fpga_manual_body",
    "main_manual_body",
]
