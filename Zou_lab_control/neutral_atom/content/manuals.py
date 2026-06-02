"""Neutral-atom manual text and figure generation."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np


def _save_plot(plot_obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_obj.fig.savefig(path, bbox_inches="tight", dpi=180)
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

    pulse = zf.plot(expanded, kind="pulse", channels=exp.devices.sequencer.channels, title="Five-frame qCMOS trigger sequence", display=False)
    pulse_path = _save_plot(pulse, asset_dir / "hardware_pulse_sequence.png")
    plt.close(pulse.fig)

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
