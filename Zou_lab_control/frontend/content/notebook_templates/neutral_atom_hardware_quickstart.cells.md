<!-- cell:markdown -->
# Neutral atom hardware quickstart

这个 notebook 是控制电脑上的硬件流程：连接 qCMOS 和 sequencer，配置 pulse sequence，拍 raw image，校准 sitemap 和 threshold，detect，最后扫 detection time。

运行前先在 Verilog/FPGA 电脑上启动 sequencer server，可以打开 `tutorials/neutral_atom_fpga_server.ipynb`，也可以运行同等命令行：

```powershell
cd "C:\path\to\Zou_lab_control_v1"
$env:PYTHONPATH = (Get-Location).Path

$env:ZLC_PS_VIVADO_BIN = "C:\Xilinx\Vivado\2019.2\bin\vivado.bat"
$env:ZLC_PS_VIVADO_PROJECT = "D:\time_sequence\zlc_pulse_streamer\zlc_pulse_streamer.xpr"
$env:ZLC_PS_VIVADO_BIT = "D:\time_sequence\zlc_pulse_streamer\zlc_pulse_streamer.runs\impl_1\main.bit"
$env:ZLC_PS_VIVADO_LTX = "D:\time_sequence\zlc_pulse_streamer\zlc_pulse_streamer.runs\impl_1\main.ltx"
$env:ZLC_PS_VIVADO_PROGRAM_ON_RUN = "1"
$env:ZLC_PS_VIO_FILTER = 'CELL_NAME=~"*vio*"'
$env:ZLC_PS_MAX_EDGES = "1024"
$env:ZLC_PS_TICK_WIDTH = "32"
$env:ZLC_PS_CHANNEL_COUNT = "4"

python -m Zou_lab_control.neutral_atom.devices.sequencer_server `
  --host 0.0.0.0 `
  --port 18861 `
  --channels trap cooling probe qcm_trigger `
  --trigger-channels qcm_trigger `
  --clock-hz 100000000 `
  --state-dir D:\zlc_sequencer_state `
  --prepare-command "python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer prepare" `
  --fire-command "python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer fire" `
  --wait-done-command "python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer wait_done" `
  --safe-state-command "python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer safe_state"
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
## Optional: edit pulses with the PyQt pulse GUI

GUI 只是 pulse 的前端。它读取 `exp.devices.sequencer.channels`，编辑 `PulseTableState`，然后在 `Prepare/Fire/Wait/Safe` 按钮里调用同一个 sequencer。新建 pulse 默认名是 `pulse_YYYYMMDD_HHMMSS`。`channels` 是硬件 channel 名和 FPGA bit order，例如 `ch00/ch01/...`；Name 面板左侧固定显示硬件 channel，右侧是可选 display label，GUI 不会自动猜物理含义。右侧 name 改动后，Delay 行、period checkbox 和 Preview y 轴会显示这个 label。左侧 `step (ns)` 来自 `1e9 / exp.devices.sequencer.clock_hz`，所有 duration、delay 和 `x` 都必须是这个 minimal time 的整数倍。40-channel server 也可以用这个入口；默认只显示硬件顺序前 4 路，其它 channel 从下拉框里添加。`X` 会把该 channel 的所有 period 设为 off，但不自动隐藏；`Hide Off` 只看 period 是否为 on，delay 非零也可以隐藏；display name 和 delay 会保留，重新 Add Channel 会按硬件顺序插回原位。全通道展开时，channel name、delay 和 period checkbox 共用整体纵向滚动。Preview 页会画未展开 period table 的 pulse 图，默认隐藏 always-off channel；如果 channel 有 display label，Preview y 轴显示 label。没有 bracket 时是 `repeat ∞`；bracket 覆盖所有 period 时是有限外层 repeat；bracket 在内部时整体仍然是 `repeat ∞`，Preview 会画整段 `∞` 和内部 `xN` 两套不同颜色的 bracket，状态栏显示 `repeat ∞ + Pm-Pn xN`。bracket 画在真实 start/stop 时间节点上，xlim 只负责留显示空间，负时间 tick label 会被隐藏。`Save Pulse` 默认保存到仓库 `pulses/` 目录；`Save Figure` 是 Preview 顶栏最右侧的一行按钮，单独保存 preview PNG。日常 camera preset 在 `pulses/camera_imaging_40ch.json`。小屏幕可以传 `scale=0.82, window_ratio=0.90`。

如果当前环境没有桌面/Qt，跳过这个 cell，继续用 `exp.timing.configure_imaging(...)` 和 API 配置 pulse。

<!-- cell:code -->
# Uncomment on a desktop Python/Qt environment.
# pulse_gui = zf.show_pulse_gui(experiment=exp, scale=0.82, window_ratio=0.90)
# pulse_gui

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
clock_hz = exp.devices.sequencer.clock_hz
time_ticks = np.linspace(int(round(0.2e-3 * clock_hz)), int(round(8e-3 * clock_hz)), 40, dtype=int)
times = time_ticks / clock_hz
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
