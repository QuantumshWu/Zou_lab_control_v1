"""Chinese PDF tutorial for the lightweight neutral-atom session."""

from __future__ import annotations

from datetime import date
from pathlib import Path


from Zou_lab_control.frontend.notes import NotesBuildResult, render_notes_pdf

from .content.manuals import (
    device_manual_body,
    fpga_manual_body,
    generate_device_manual_figures,
    generate_fpga_manual_figures,
    main_manual_body,
)


def build_main_manual(
    output_dir: str | Path = "docs/main_manual",
    *,
    compile_pdf: bool = True,
) -> NotesBuildResult:
    """Generate the main (system-overview) manual.

    The main manual uses inline TikZ diagrams only, so it has no figure
    dependencies and compiles directly from the template.
    """

    return render_notes_pdf(
        Path(output_dir),
        filename="main_manual_zh.tex",
        title="Zou_lab_control 主手册",
        subtitle="notebook-first + PyQt GUI + remote FPGA pulse-streamer 系统总览",
        description="架构、session/devices/timing 分层、sequencer 生命周期、真实硬件 runbook 与 N-slot 扫描模型",
        body=main_manual_body(),
        doc_date=date.today().isoformat(),
        compile_pdf=compile_pdf,
    )


def build_fpga_manual(
    output_dir: str | Path = "docs/fpga_manual",
    *,
    compile_pdf: bool = True,
) -> NotesBuildResult:
    """Generate the FPGA pulse-streamer manual.

    The TIMING diagrams are REAL pulses rendered by the frontend plotter
    (generate_fpga_manual_figures) into ``<output_dir>/assets`` and injected via
    fpga_manual_body; block-diagram TikZ stays inline.
    """

    output_dir = Path(output_dir)
    asset_dir = output_dir / "assets"
    figures = generate_fpga_manual_figures(asset_dir)
    tex_figures = {name: Path("assets") / path.name for name, path in figures.items()}
    return render_notes_pdf(
        output_dir,
        filename="fpga_manual_zh.tex",
        title="ZLC FPGA Pulse Streamer 手册",
        subtitle="Artix-7 35T 边沿流送器 / 1-tick 预取 / 无限流式扫描 / JTAG-to-AXI",
        description="RTL、1-tick 预取、双 bank 流式、仿射扫描引擎、模拟总线 DAC、编译上传流程、资源预算",
        body=fpga_manual_body(tex_figures),
        doc_date=date.today().isoformat(),
        compile_pdf=compile_pdf,
    )


def build_device_manual(
    output_dir: str | Path = "docs/device_manual",
    *,
    compile_pdf: bool = True,
) -> NotesBuildResult:
    """Generate the device & experiment manual.

    Covers device configuration/loading, camera capture, the camera-readout
    tutorial with principles, calibration & result objects, and the experiment
    flow.  Embeds the real threshold-calibration histogram rendered offline via
    the virtual backend (with a placeholder fallback).
    """

    output_dir = Path(output_dir)
    asset_dir = output_dir / "assets"
    figures = generate_device_manual_figures(asset_dir)
    tex_figures = {name: Path("assets") / path.name for name, path in figures.items()}
    return render_notes_pdf(
        output_dir,
        filename="device_manual_zh.tex",
        title="ZLC 设备与实验手册",
        subtitle="设备配置/加载 / 相机读出原理 / 标定与结果对象 / 实验流程",
        description="devices 层契约、camera capture/readout 教程(sitemap/thresholds/detect)、TrapCalibration、虚拟后端与完整实验循环",
        body=device_manual_body(tex_figures),
        doc_date=date.today().isoformat(),
        compile_pdf=compile_pdf,
    )


__all__ = [
    "build_device_manual",
    "build_fpga_manual",
    "build_main_manual",
]
