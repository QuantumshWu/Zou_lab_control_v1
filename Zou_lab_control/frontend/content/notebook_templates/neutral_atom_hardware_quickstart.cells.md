<!-- cell:markdown -->
# Neutral atom real-hardware quickstart

这个 notebook 是控制电脑上跑的真实硬件流程，不是 virtual demo。它会连接真实 `QCMOSCamera` 和真实/远程 `SequencerDevice`，然后按实验顺序拍照、校准 sitemap、校准 threshold、detect，最后扫 detection time。

运行前请先在 Verilog/FPGA 电脑上启动 sequencer server。推荐先打开并运行 `tutorials/neutral_atom_fpga_server.ipynb`，或者使用同等命令行：

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

这个 notebook 不包含模拟采集。如果你只是想离线检查 frontend/readout 流程，请跑 `neutral_atom_tutorial.ipynb`。

<!-- cell:code -->
{{BOOTSTRAP_CELL}}

<!-- cell:code -->
from pathlib import Path
import json
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
## 1. Hardware parameters for this run

这里的参数应该是你这次真实实验要用的值。`DEVICE_CONFIG` 默认使用 `real_remote_template`，也就是控制电脑连接 qCMOS，同时通过 `RemoteSequencer` 连接 Verilog/FPGA 电脑。如果 remote server 还没有准备好，可以临时改成 `real_manual_template` 做 first-light 手动触发。

<!-- cell:code -->
DEVICE_CONFIG = "real_remote_template"   # or "real_manual_template" for first-light manual trigger

GRID_SHAPE = (5, 7)
ROI_RADIUS = 1

IMAGING_EXPOSURE = 2e-3
TRIGGER_WIDTH = 20e-6
PRE_TRIGGER = 100e-6

SITEMAP_FRAMES = 20
THRESHOLD_FRAMES = 120
THRESHOLD_SITE = 0

DETECTION_TIMES = np.linspace(0.2e-3, 8e-3, 40)
DETECTION_SCAN_SHOTS = 30

RESULT_DIR = Path("results_real_hardware")
RESULT_DIR.mkdir(exist_ok=True)

<!-- cell:markdown -->
## 2. Inspect the real device config

这一步只读配置文件，不打开 camera，也不连接 remote sequencer。确认 ROI、host、port、channel 名之后再继续。

<!-- cell:code -->
config_path = na.device_config_dir() / f"{DEVICE_CONFIG}.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
print(config_path)
print(json.dumps(config, indent=2))

<!-- cell:markdown -->
## 3. Connect real devices

这一步开始连接真实硬件：

- `real_remote_template`: `exp.devices.sequencer.open()` 会连接 Verilog/FPGA 电脑上的 RPyC sequencer server。
- `exp.camera.open()` 会 import DCAM wrapper 并打开 qCMOS。

如果这里失败，不要跳过；先修 device config、IP/port、DCAM driver、相机 index 或 Verilog 电脑 server。

<!-- cell:code -->
exp = na.connect(DEVICE_CONFIG)
zf.require_attrs(exp, ["camera", "readout", "timing"], name="exp")

assert isinstance(exp.camera, na.QCMOSCamera), type(exp.camera)
assert isinstance(exp.devices.sequencer, na.SequencerDevice), type(exp.devices.sequencer)

if hasattr(exp.devices.sequencer, "open"):
    exp.devices.sequencer.open()
exp.camera.open()

exp.status()

<!-- cell:markdown -->
## 4. Configure and preflight the imaging sequence

`PulseSequence` 是真实硬件和 notebook 共同使用的时序源。`preflight.raise_if_failed()` 必须通过之后再拍照。pulse plot 只用于检查，不替代 preflight。

<!-- cell:code -->
sequence = exp.timing.configure_imaging(
    exposure=IMAGING_EXPOSURE,
    load=True,
    trigger_width=TRIGGER_WIDTH,
    pre_trigger=PRE_TRIGGER,
)

pulse_plot = exp.timing.plot_sequence()
preflight = exp.timing.preflight()
preflight.summary()

<!-- cell:code -->
preflight.raise_if_failed()

<!-- cell:markdown -->
## 5. First real raw capture

这一步会真实 arm qCMOS、fire sequencer、等待 frame ready，然后画 raw camera frame。`capture` 不应该显示任何 sitemap 圈；如果 raw 图上已经有 site overlay，说明代码层边界错了。

如果使用 `real_manual_template`，执行这个 cell 后按照输出提示手动启动 FPGA/manual trigger。

<!-- cell:code -->
capture = exp.camera.capture(frames=1, display=True)
capture.summary()

<!-- cell:markdown -->
## 6. Calibrate sitemap from real camera images

真实硬件配置没有 `trap_array`，所以必须显式给 `grid_shape`。这个步骤用真实 camera images 找 camera-space site centers。

<!-- cell:code -->
sitemap = exp.readout.sitemap(
    frames=SITEMAP_FRAMES,
    grid_shape=GRID_SHAPE,
    roi_radius=ROI_RADIUS,
    display=True,
)
sitemap.summary()

<!-- cell:markdown -->
## 7. Calibrate thresholds

threshold calibration 依赖刚刚得到的 sitemap。它会真实采集 `THRESHOLD_FRAMES` 张图，按 ROI counts 估计每个 site 的 threshold。

<!-- cell:code -->
threshold = exp.readout.thresholds(
    frames=THRESHOLD_FRAMES,
    site=THRESHOLD_SITE,
    display=True,
)
threshold.summary()

<!-- cell:markdown -->
## 8. Detect one real shot

输出的 `shot.occupied` 是后续 rearrangement/statistics 可以直接使用的 boolean array。图上的浅色圈来自 sitemap calibration，橙色圈只标 occupied sites。

<!-- cell:code -->
shot = exp.readout.detect(display=True)
occupied_grid = shot.occupied.reshape(GRID_SHAPE)
occupied_grid, shot.summary()

<!-- cell:markdown -->
## 9. Scan real detection time and fidelity

这个 scan 仍然使用真实 camera images，不使用任何 ground truth。为了第一次上机可控，这里默认 `live=False`，cell 会等整个 scan 结束后返回。确认流程稳定后可以改成 `live=True`，用 `scan.stop()` 中断长 scan。

<!-- cell:code -->
scan = exp.readout.detection_time(
    DETECTION_TIMES,
    shots=DETECTION_SCAN_SHOTS,
    live=False,
    display=True,
)
fit_result, popt = scan.data_figure.decay(is_display=False)
scan.summary(), fit_result, popt

<!-- cell:markdown -->
## 10. Save real run artifacts

保存 calibration、status 和当前 sequence/Verilog manifest。后续 notebook 可以先 `exp.readout.load(calibration_path)`，再直接 detect 或 scan。

<!-- cell:code -->
calibration_path = exp.readout.save(RESULT_DIR / "neutral_atom_real_calibration.json")
status_path = exp.save_status(RESULT_DIR / "neutral_atom_real_status.json")
verilog_path = exp.timing.write_verilog(RESULT_DIR / "generated_sequences")

calibration_path, status_path, verilog_path

<!-- cell:markdown -->
## 11. Close hardware

实验结束后关闭 camera 和 remote connection。

<!-- cell:code -->
exp.close()
