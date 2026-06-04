<!-- cell:markdown -->
# Neutral atom hardware quickstart

这个 notebook 是控制电脑上的硬件流程：连接 qCMOS 和 40ch FPGA sequencer，配置 pulse sequence，拍 raw image，校准 sitemap 和 threshold，detect，最后扫 detection time。

运行前先在 Verilog/FPGA 电脑上启动 sequencer server，可以打开 `tutorials/neutral_atom_fpga_server.ipynb`，也可以运行同等命令行：

```powershell
cd "D:\ZLC"
.\fpga\build_and_program.bat --check
.\fpga\build_and_program.bat
.\fpga\run_server.bat --check-config
.\fpga\run_server.bat
```

如果 Verilog 电脑的 Vivado 不在默认搜索路径，先设置：

```powershell
$env:ZLC_PS_VIVADO_BIN = "C:\Xilinx\Vivado\2019.2\bin\vivado.bat"
```

`fpga\build_and_program.bat` 和 `fpga\run_server.bat` 都是 40ch 入口。GUI 或 API 可以只配置 `ch00..ch03`，但传给 FPGA backend 时会按 server 的完整 `ch00..ch39` channel order 编译成 40-bit mask；没有配置的 channel 全部为 off。40ch 真实 bitstream 默认使用已经从旧 `address_switch` pin map 生成的 `fpga\pulse_streamer\zlc_pulse_streamer_40ch.xdc`；如果板卡或转接线不同，可以设置 `ZLC_PS_40CH_XDC` 指向别处的板级 XDC。没确认真实 pin 前只运行 `.\fpga\build_and_program.bat --check` 做 HDL/VIO 宽度自查，不要 program。

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

GUI 只是 pulse 前端。它读取 `exp.devices.sequencer.channels`，编辑 `PulseTableState`，然后在 `On Pulse/Stop Pulse` 按钮里调用同一个 sequencer。`On Pulse` 会先把当前 pulse state 上传到 sequencer，再立刻 start；`Stop Pulse` 调用 safe/reset。GUI 里没有单独的 sync 按钮，也没有 `Wait Done` 按钮；等待 finite shot 完成属于 notebook/API 和 camera acquisition。

40ch server 的硬件 channel 和 FPGA bit order 是 `ch00...ch39`，trigger 默认是 `ch03`。GUI 默认只显示前 4 路，让简单 pulse 不乱；其它 channel 可以从 Add Channel 下拉框加回来。这个“显示几路”不改变硬件宽度：上传时仍然以 server 的 40 路为准，未显示/未配置的 channel 自动补 0。standalone `pulse_gui.bat` 会从 `fpga\pulse_streamer\zlc_pulse_streamer_40ch.xdc` 注释读取默认显示名；`pulses/camera_imaging_40ch.json` 也保存了完整 40 路 display label。保存和上传的硬件名字仍是 `ch00..ch39`。

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
## Bind a pulse for x scans

对于 readout-time 或曝光宽度扫描，可以把一张 `PulseTableState` 绑定到当前 session 的 sequencer。之后只改 `pulse.x`，再调用 `pulse.on_pulse()`；它内部会走同一条 remote `prepare -> fire` 链路。仓库里的 `pulses/camera_imaging_40ch.json` 已经把 `camera_exposure` period 写成 `duration="x", unit="str (ns)"`，默认 `x_ns=19_980_000`，所以 `pulse.x` 就是 probe/readout exposure 的 ns 数。

GUI 不再暴露单独的 `Repeat ∞` 开关；默认整体就是 inf repeat，所以没有内部 bracket 时 Preview 仍然会画覆盖整段的 `∞` bracket。脚本里如果要等待一次 finite shot，调用时写 `repeat_forever=False`。传给 camera acquisition 时，`exp.readout.detection_time(..., pulse=pulse)` 会用同一张 pulse 先拍 long-reference，再为每个扫描点临时生成刚好 `shots` 个 qCMOS trigger 的有限序列，保证相机先 arm，再由同一个 sequencer fire。

server 会缓存上一次已经上传的 `sequence_id`。如果 `pulse.x` 和 pulse 表都没变，再次 `On Pulse` 只会发送 `fire`；只要改了 `x`、period、channel state 或 repeat metadata，就会得到新的 `sequence_id` 并重新上传。

<!-- cell:code -->
pulse = exp.timing.bind_pulse("pulses/camera_imaging_40ch.json")
pulse.snapshot()

# This does not fire hardware; it shows that x controls the finite readout
# sequence duration before you run the scan.
test_widths_ns = [2_000_000, 4_000_000, 8_000_000]
[(width, pulse.frame_sequence(1, x_ns=width).duration) for width in test_widths_ns]

RUN_SINGLE_PULSE_TEST = False

single_program = None
if RUN_SINGLE_PULSE_TEST:
    pulse.x = 2_000_000  # ns
    single_program = pulse.on_pulse(wait=True, timeout=10.0, repeat_forever=False)
single_program

# Free-running output is still explicit when you want it:
# pulse.on_pulse(wait=False, repeat_forever=True)

<!-- cell:markdown -->
## Scan detection time and fidelity

这个 scan 使用 camera images，不使用任何 ground truth。第一次上机默认同步跑完；确认流程稳定后，下一格有一个显式的 live 版本：只把 `RUN_LIVE_READOUT_SCAN` 改成 `True`，其它 API 形状不变，仍然通过同一个 `pulse` 和 remote sequencer。

<!-- cell:code -->
clock_hz = exp.devices.sequencer.clock_hz
time_ticks = np.linspace(int(round(0.2e-3 * clock_hz)), int(round(8e-3 * clock_hz)), 40, dtype=int)
times = time_ticks / clock_hz
scan = exp.readout.detection_time(times, shots=30, live=False, display=True, pulse=pulse)
fit_result, popt = scan.data_figure.decay(is_display=False)
scan.summary(), fit_result, popt

<!-- cell:markdown -->
## Optional live readout-time scan

这个 cell 是控制电脑上最短的 live readout-time/fidelity 工作形状。它不会改 FPGA 电脑的 server，也不需要重新打开 GUI：`pulse.x` 仍然是唯一的 readout exposure 变量，`exp.readout.detection_time(..., live=True, pulse=pulse)` 会在 frontend worker 里逐点更新图。

第一次真实上机建议先保持 `RUN_LIVE_READOUT_SCAN = False`，确认前一格同步 scan 正常后再改成 `True`。live scan 返回后 notebook 不会阻塞等待全部点结束；需要提前停时运行下一格 `live_scan.stop()`。

<!-- cell:code -->
RUN_LIVE_READOUT_SCAN = False

live_scan = None
if RUN_LIVE_READOUT_SCAN:
    live_scan = exp.readout.detection_time(
        times,
        shots=30,
        live=True,
        display=True,
        pulse=pulse,
        update_time=0.2,
    )
live_scan

<!-- cell:markdown -->
## Stop live scan

如果上一格启动了 live scan，运行这一格会请求 acquisition worker 停止并保留已经采到的数据。停止后仍然可以查看 `live_scan.summary()`，或者用 `live_scan.data_figure.decay(...)` 拟合已经完成的点。

<!-- cell:code -->
if live_scan is not None:
    live_scan.stop()
    live_scan.summary()

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
