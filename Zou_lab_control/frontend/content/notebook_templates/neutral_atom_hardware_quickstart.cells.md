<!-- cell:markdown -->
# Neutral atom hardware quickstart

这个 notebook 是控制电脑上的硬件流程：连接 qCMOS 和 sequencer，配置 pulse sequence，拍 raw image，校准 sitemap 和 threshold，detect，最后扫 detection time。

运行前先在 Verilog/FPGA 电脑上启动 sequencer server，可以打开 `tutorials/neutral_atom_fpga_server.ipynb`，也可以运行同等命令行：

```powershell
cd "C:\path\to\Zou_lab_control_v1"
$env:PYTHONPATH = (Get-Location).Path

$env:ZLC_LEGACY_SINGLE_CAMERA_TRIGGER_CONFIRMED = "0"  # set to 1 only after oscilloscope confirmation

python -m Zou_lab_control.neutral_atom.devices.sequencer_server `
  --host 0.0.0.0 `
  --port 18861 `
  --channels trap cooling probe qcm_trigger `
  --trigger-channels qcm_trigger `
  --state-dir D:\zlc_sequencer_state `
  --prepare-command "python -m Zou_lab_control.neutral_atom.devices.legacy_address_switch prepare" `
  --fire-command "python -m Zou_lab_control.neutral_atom.devices.legacy_address_switch fire" `
  --wait-done-command "python -m Zou_lab_control.neutral_atom.devices.legacy_address_switch wait_done" `
  --safe-state-command "python -m Zou_lab_control.neutral_atom.devices.legacy_address_switch safe_state"
```

离线检查 frontend/readout 流程时跑 `neutral_atom_tutorial.ipynb`。

<!-- cell:code -->
{{BOOTSTRAP_CELL}}

<!-- cell:code -->
from pathlib import Path
import numpy as np

import Zou_lab_control.frontend as zf
import Zou_lab_control.neutral_atom as na

try:
    zf.use_widget_backend()
except Exception as exc:
    print(f"Widget backend not enabled here: {exc}")

zf.enable_long_output()
zf.apply_style()

<!-- cell:markdown -->
## Connect hardware

`na.connect(..., open_devices=True)` 会通过 device loader 构造、校验并打开 camera/sequencer。

<!-- cell:code -->
exp = na.connect(
    "remote_template",
    sequencer={"host": "192.168.0.20", "port": 18861},
    open_devices=True,
)

# First-light manual trigger path:
# exp = na.connect("manual_template", open_devices=True)

exp

<!-- cell:markdown -->
## Configure and preflight the imaging sequence

`PulseSequence` 是 hardware 和 notebook 共同使用的时序源。`preflight.raise_if_failed()` 通过之后再拍照。

<!-- cell:code -->
exp.timing.configure_imaging(exposure=2e-3, load=True, trigger_width=20e-6, pre_trigger=100e-6)
pulse_plot = exp.timing.plot_sequence()
preflight = exp.timing.preflight()
preflight.summary()

<!-- cell:code -->
preflight.raise_if_failed()

<!-- cell:markdown -->
## Capture a camera image

`capture` 只显示 raw camera frame；site overlay 只属于 calibration/readout/detect 图。

<!-- cell:code -->
capture = exp.camera.capture(frames=1, display=True)
capture.summary()

<!-- cell:markdown -->
## Calibrate sitemap

hardware config 没有 virtual `trap_array`，所以 sitemap 需要显式给出 site grid。

<!-- cell:code -->
grid_shape = (5, 7)
sitemap = exp.readout.sitemap(frames=20, grid_shape=grid_shape, roi_radius=1, display=True)
sitemap.summary()

<!-- cell:markdown -->
## Calibrate thresholds

threshold calibration 依赖刚刚得到的 sitemap。

<!-- cell:code -->
threshold = exp.readout.thresholds(frames=120, site=0, display=True)
threshold.summary()

<!-- cell:markdown -->
## Detect one shot

`DetectionResult.occupied` 是后续 rearrangement/statistics 可以直接使用的 boolean array。

<!-- cell:code -->
shot = exp.readout.detect(display=True)
occupancy_grid = shot.occupied.reshape(grid_shape)
occupancy_grid, shot.summary()

<!-- cell:markdown -->
## Scan detection time and fidelity

这个 scan 使用 camera images，不使用任何 ground truth。第一次上机默认同步跑完；确认流程稳定后可以改成 `live=True`，再用 `scan.stop()` 结束 live scan。

<!-- cell:code -->
times = np.linspace(0.2e-3, 8e-3, 40)
scan = exp.readout.detection_time(times, shots=30, live=False, display=True)
fit_result, popt = scan.data_figure.decay(is_display=False)
scan.summary(), fit_result, popt

<!-- cell:markdown -->
## Save calibration, status, and Verilog

<!-- cell:code -->
Path("results").mkdir(exist_ok=True)
Path("generated_sequences").mkdir(exist_ok=True)

calibration_path = exp.readout.save("results/neutral_atom_calibration.json")
status_path = exp.save_status("results/neutral_atom_status.json")
verilog_path = exp.timing.write_verilog("generated_sequences")

calibration_path, status_path, verilog_path

<!-- cell:markdown -->
## Close hardware

<!-- cell:code -->
exp.close()
