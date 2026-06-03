<!-- cell:markdown -->
# Neutral atom hardware quickstart

这个 notebook 是控制电脑上的硬件流程：连接 qCMOS 和 40ch FPGA sequencer，配置 pulse sequence，拍 raw image，校准 sitemap 和 threshold，detect，最后扫 detection time。

运行前先在 Verilog/FPGA 电脑上启动 sequencer server，可以打开 `tutorials/neutral_atom_fpga_server.ipynb`，也可以运行同等命令行：

```powershell
cd "D:\ZLC"
.\fpga\build_and_program.bat --check
.\fpga\build_and_program.bat
.\fpga\run_server.bat
```

如果 Verilog 电脑的 Vivado 不在默认搜索路径，先设置：

```powershell
$env:ZLC_PS_VIVADO_BIN = "C:\Xilinx\Vivado\2019.2\bin\vivado.bat"
```

`fpga\build_and_program.bat` 和 `fpga\run_server.bat` 都是 40ch 入口。GUI 或 API 可以只配置 `ch00..ch03`，但传给 FPGA backend 时会按 server 的完整 `ch00..ch39` channel order 编译成 40-bit mask；没有配置的 channel 全部为 off。40ch 真实 bitstream 需要先填完 `fpga\pulse_streamer\zlc_pulse_streamer_40ch.xdc`，也可以设置 `ZLC_PS_40CH_XDC` 指向别处的板级 XDC；没填真实 pin 前只运行 `.\fpga\build_and_program.bat --check` 做 HDL/VIO 宽度自查，不要 program。

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

GUI 只是 pulse 前端。它读取 `exp.devices.sequencer.channels`，编辑 `PulseTableState`，然后在 `On Pulse/Stop Pulse/Wait Done` 按钮里调用同一个 sequencer。`On Pulse` 会先把当前 pulse state 上传到 sequencer，再立刻 start；GUI 里没有单独的 sync 按钮。

40ch server 的硬件 channel 和 FPGA bit order 是 `ch00...ch39`，trigger 默认是 `ch03`。GUI 默认只显示前 4 路，让简单 pulse 不乱；其它 channel 可以从 Add Channel 下拉框加回来。这个“显示几路”不改变硬件宽度：上传时仍然以 server 的 40 路为准，未显示/未配置的 channel 自动补 0。`pulses/camera_imaging_40ch.json` 只给 `ch00..ch03` 显示 label：`trap/cooling/probe/qcm_trigger`，但保存和上传的硬件名字仍是 `ch00..ch39`。

左侧 `step (ns)` 来自 `1e9 / exp.devices.sequencer.clock_hz`，所有 duration、delay 和 `x` 都必须是这个 minimal time 的整数倍。`X` 会把该 channel 的所有 period 设为 off；`Hide Off` 只看 period 是否为 on，display name 和 delay 会保留，重新 Add Channel 会按硬件顺序插回原位。全通道展开时，channel name、delay 和 period checkbox 共用整体纵向滚动。Preview 页会画未展开 period table 的 pulse 图，默认隐藏 always-off channel，并用 display label 作为 y 轴。`Save Pulse` 默认保存到仓库 `pulses/` 目录；`Save Figure` 单独保存 preview PNG。小屏幕可以传 `scale=0.82, window_ratio=0.90`。

如果当前环境没有桌面/Qt，跳过这个 cell，继续用 `exp.timing.configure_imaging(...)` 和 API 配置 pulse。

<!-- cell:code -->
# Uncomment on a desktop Python/Qt environment.
# pulse_gui = zf.show_pulse_gui(
#     experiment=exp,
#     state=na.PulseTableState.load("pulses/camera_imaging_40ch.json"),
#     scale=0.82,
#     window_ratio=0.90,
# )
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
