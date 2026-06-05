"""Neutral-atom manual text and figure generation."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np


def _save_plot(plot_obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_obj.fig.savefig(path, bbox_inches="tight", dpi=180)
    return path


def _save_pulse_streamer_flow_figure(path: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.8, 5.8), dpi=180)
    ax.set_xlim(0, 12.8)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    nodes = [
        ((0.35, 3.35), "Pulse GUI / Notebook", "PulseTableState\nvisible subset"),
        ((2.45, 3.35), "RemoteSequencer", "JSON over RPyC\nprepare / fire"),
        ((4.55, 3.35), "FPGA PC server", "compile full\nXDC-inferred masks"),
        ((6.65, 3.35), "Vivado session", "VIO probes\nchanged rows"),
        ((8.75, 3.35), "Verilog core", "tick_mem + mask_mem\nstate machine"),
        ((10.85, 3.35), "XDC outputs", "emCCD / trap\ncooling / probe"),
    ]
    box_w = 1.65
    box_h = 1.25
    colors = ["#EAF3F8", "#EEF6EA", "#FFF6DE", "#F7ECF1", "#ECECFA", "#F2F2F2"]

    for idx, ((x, y), title, body) in enumerate(nodes):
        box = FancyBboxPatch(
            (x, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#4A6472",
            facecolor=colors[idx],
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, y + 0.86, title, ha="center", va="center", fontsize=9.2, weight="bold", color="#23343D")
        ax.text(x + box_w / 2, y + 0.38, body, ha="center", va="center", fontsize=8.0, color="#3B4A52")

    def arrow(x0, y0, x1, y1, label=None, color="#5D7583"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color=color))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.20, label, ha="center", va="bottom", fontsize=7.8, color=color)

    for i in range(len(nodes) - 1):
        x0, y0 = nodes[i][0]
        x1, y1 = nodes[i + 1][0]
        arrow(x0 + box_w, y0 + box_h / 2, x1, y1 + box_h / 2)

    ax.text(6.4, 5.08, "Runtime pulse-streamer path", ha="center", va="center", fontsize=15, weight="bold", color="#22313A")
    ax.text(
        6.4,
        4.78,
        "GUI/API is only the front-end; the FPGA server owns compilation and hardware upload.",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#53656D",
    )

    lower = [
        (0.65, 1.55, 3.15, "Edit pulse once:\nname, delays, periods, bracket, x"),
        (3.6, 1.55, 3.15, "Prepare before camera arm:\nreset high, upload ticks/masks, release reset"),
        (7.05, 1.55, 2.15, "Fire:\none start pulse"),
        (9.65, 1.55, 2.45, "After start:\nFPGA clock owns edge timing"),
    ]
    for x, y, w, text in lower:
        box = FancyBboxPatch(
            (x, y),
            w,
            0.72,
            boxstyle="round,pad=0.08,rounding_size=0.08",
            linewidth=1.0,
            edgecolor="#B2A469",
            facecolor="#FFF9E8",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + 0.36, text, ha="center", va="center", fontsize=8.0, color="#5C4D19")

    arrow(5.15, 3.35, 5.15, 2.35, "edge table", color="#9A765E")
    arrow(8.0, 2.27, 8.0, 3.35, "VIO", color="#9A765E")
    arrow(8.12, 1.92, 9.08, 3.34, "start", color="#7A6FA4")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _template_text(name: str) -> str:
    path = resources.files("Zou_lab_control.neutral_atom") / "content" / "manual_templates" / name
    return path.read_text(encoding="utf-8")


def generate_hardware_quickstart_figures(asset_dir: str | Path) -> dict[str, Path]:
    """Generate figures used by the hardware quickstart manual."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    import Zou_lab_control.frontend as zf
    import Zou_lab_control.neutral_atom as na

    zf.apply_style()
    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    exp = na.connect(
        "virtual",
        bright_count_rate=3000,
        background_count_rate=8,
        loss_rate=0.1,
        exposure=2e-3,
        sitemap={"grid_shape": (5, 7), "spacing_px": 12.0, "roi_radius": 1, "sitemap_exposure": 0.02},
    )
    seq = exp.timing.configure_imaging(exposure=2e-3, load=True, trigger_width=20e-6, pre_trigger=100e-6)
    expanded = na.sequence_for_frame_count(seq, 5)

    pulse = zf.plot(expanded, kind="pulse", channels=exp.devices.sequencer.channels, title="Five-frame camera-trigger sequence", display=False)
    pulse_path = _save_plot(pulse, asset_dir / "hardware_pulse_sequence.png")
    plt.close(pulse.fig)

    flow_path = _save_pulse_streamer_flow_figure(asset_dir / "hardware_pulse_streamer_flow.png")

    capture = exp.camera.capture(display=False)
    capture_path = _save_plot(capture.plot, asset_dir / "hardware_capture.png")
    plt.close(capture.plot.fig)

    sitemap = exp.readout.sitemap(frames=12, display=False)
    sitemap_path = _save_plot(sitemap.plot, asset_dir / "hardware_sitemap.png")
    plt.close(sitemap.plot.fig)

    threshold = exp.readout.thresholds(frames=80, site=0, display=False)
    threshold_path = _save_plot(threshold.plot, asset_dir / "hardware_threshold.png")
    plt.close(threshold.plot.fig)

    shot = exp.readout.detect(display=False)
    detect_path = _save_plot(shot.plot, asset_dir / "hardware_detect.png")
    plt.close(shot.plot.fig)

    clock_hz = exp.devices.sequencer.clock_hz
    time_ticks = np.linspace(int(round(0.2e-3 * clock_hz)), int(round(8e-3 * clock_hz)), 60, dtype=int)
    times = time_ticks / clock_hz
    scan = exp.readout.detection_time(times, shots=20, live=False, display=False)
    scan_path = _save_plot(scan.plot, asset_dir / "hardware_detection_time.png")
    plt.close(scan.plot.fig)

    return {
        "pulse": pulse_path,
        "pulse_streamer_flow": flow_path,
        "capture": capture_path,
        "sitemap": sitemap_path,
        "threshold": threshold_path,
        "detect": detect_path,
        "scan": scan_path,
    }


def hardware_quickstart_body(figures: dict[str, Path]) -> str:
    """Return the hardware quickstart manual body with figure paths filled in."""

    text = _template_text("hardware_quickstart_zh.texbody")
    for name, path in figures.items():
        text = text.replace(f"__FIG_{name.upper()}__", (Path("assets") / Path(path).name).as_posix())
    return text


__all__ = ["generate_hardware_quickstart_figures", "hardware_quickstart_body"]
