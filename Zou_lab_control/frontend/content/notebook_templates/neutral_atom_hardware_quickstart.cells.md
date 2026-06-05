<!-- cell:markdown -->
# Neutral atom hardware quickstart

这个 notebook 是控制电脑上的硬件流程：连接 qCMOS 和 FPGA sequencer，配置
pulse sequence，拍 raw image，校准 sitemap 和 threshold，detect，最后扫
detection time。

运行前先在 Verilog/FPGA 电脑上启动 sequencer server：

```powershell
cd "D:\ZLC"
.\fpga\build_and_program.bat --check
.\fpga\build_and_program.bat
.\fpga\run_server.bat --check-config
.\fpga\run_server.bat
```

默认硬件路线是 address-switch pulse streamer。server 从 XDC 推断完整
channel order；GUI 或 API 可以只显示/配置其中几路，但上传时会自动补成
full-width mask，没配置的 channel 全部为 off。默认相机成像子集是：

```text
ch09 trap
ch00 cooling
ch03 probe
ch11 emCCD
```

The same XDC also has `ch06 trig`, but the checked-in camera preset uses
`ch11/emCCD/M13` as the qCMOS/emCCD trigger.

默认 clock 是 50 MHz，也就是 20 ns step。所有 duration、delay 和 scan table
里的 timing 参数都必须对齐到这个 step。

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

`na.connect(..., open_devices=True)` 会通过 device loader 构造、校验并打开
camera/sequencer。把 `host` 改成 FPGA/Vivado 电脑的 IP。

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

`PulseSequence` 是 hardware 和 notebook 共同使用的时序源。address-switch
sequencer 会把 imaging helper 映射到 `ch09/ch00/ch03/ch11`。通过
`preflight.raise_if_failed()` 之后再拍照。

<!-- cell:code -->
exp.timing.configure_imaging(exposure=2e-3, load=True, trigger_width=20e-6, pre_trigger=100e-6)
pulse_plot = exp.timing.plot_sequence()
preflight = exp.timing.preflight()
preflight.summary()

<!-- cell:code -->
preflight.raise_if_failed()

<!-- cell:markdown -->
## Optional: edit pulses with the PyQt pulse GUI

GUI 只是 pulse 前端。它读取 `exp.devices.sequencer.channels`，编辑
`PulseTableState`，然后在 `On Pulse/Stop Pulse` 按钮里调用同一个
sequencer。`On Pulse` 会先把当前 pulse state 上传到 sequencer，再立刻
start；`Stop Pulse` 调用 safe/reset。

如果当前环境没有桌面/Qt，跳过这个 cell，继续用
`exp.timing.configure_imaging(...)` 和 API 配置 pulse。

Pulse GUI 的实际工作方式：

```text
Edit tab
  Channel Names and Duration: display label、total duration、visible count
  Delay and Scan:             Step、active params、scan file、per-channel delay
  Period cards:               duration/unit 和每个 visible channel 的 on/off
  Control Buttons:            On/Stop、Add/Remove Column、Add Bracket、Save/Load
  Channel View:               Add Channel、Hide Off、Show All

Preview tab
  自动画当前 PulseTableState，不需要手动 refresh。
  默认只画 active channel；Show off rows 会显示完整 channel list。
```

Name 面板左侧 raw column 在 address-switch 路线下显示 XDC package pin。例如
camera preset 应该看到 `M17/F15/N15/M13`，对应 `trap/cooling/probe/emCCD`。
`chNN` 硬件 bit 名仍然保存在 tooltip、JSON 和 API state 里。这里特别注意：
XDC 里还有 `ch06/trig/R17`，但当前 camera/qCMOS preset 的 trigger 是
`ch11/emCCD/M13`。

如果只想让 FPGA 自由重复输出给示波器看，GUI 的 `On Pulse` 是合适的；默认
camera preset 是 `repeat_forever=True`。如果要拍有限帧 camera stack，不要让
camera 等一个无限自由循环的 pulse；使用后面的 `exp.readout...` helper，它会
先 arm camera，再为所需帧数生成 finite trigger sequence 并 fire。

<!-- cell:code -->
# Uncomment on a desktop Python/Qt environment.
# pulse_gui = zf.show_pulse_gui(
#     experiment=exp,
#     state=na.PulseTableState.load("pulses/camera_imaging_address_switch.json"),
#     scale=0.82,
#     window_ratio=0.90,
# )
# pulse_gui

<!-- cell:markdown -->
## Pulse API equivalent

GUI 不是单独硬件层；下面的 API 和 GUI `On Pulse` 调的是同一个 sequencer。
这段适合在真正拍照前做软件侧 preflight，或者在示波器上打一发 finite shot。

<!-- cell:code -->
state = na.PulseTableState.load("pulses/camera_imaging_address_switch.json")
program = state.compile(
    clock_hz=exp.devices.sequencer.clock_hz,
    trigger_channels=exp.devices.sequencer.trigger_channels,
    repeat_forever=False,
)
{
    "ticks": program.ticks[:8],
    "masks": program.masks[:8],
    "trigger_count": program.trigger_count,
    "repeat_forever": program.repeat_forever,
}

<!-- cell:markdown -->
To actually fire the finite test pulse, set `RUN_SCOPE_PULSE_TEST = True`.
Keep it `False` while the camera is connected unless you are deliberately doing
scope/debug work.

<!-- cell:code -->
RUN_SCOPE_PULSE_TEST = False

scope_program = None
if RUN_SCOPE_PULSE_TEST:
    scope_program = exp.devices.sequencer.prepare(program)
    exp.devices.sequencer.fire()
scope_program

<!-- cell:markdown -->
## Capture a camera image

`capture` 只显示 raw camera frame；site overlay 只属于
calibration/readout/detect 图。

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

`DetectionResult.occupied` 是后续 rearrangement/statistics 可以直接使用的
boolean array。

<!-- cell:code -->
shot = exp.readout.detect(display=True)
occupancy_grid = shot.occupied.reshape(grid_shape)
occupancy_grid, shot.summary()

<!-- cell:markdown -->
## Bind a pulse for named scan parameters

对于 readout-time 或曝光宽度扫描，可以把一张 `PulseTableState` 绑定到当前
session 的 sequencer。仓库里的
`pulses/camera_imaging_address_switch.json` 已经把 `camera_exposure` period
写成 `duration="camera_exposure_ns", unit="str (ns)"`，默认
`camera_exposure_ns=19_980_000`，所以
`pulse.set_variable("camera_exposure_ns", value_ns)` 就能改变 probe/readout
exposure。

GUI/API 的 scan table 是一个带名字的文件；GUI 左侧列出 active params，并链接
scan file：

```text
# vars: camera_exposure_ns(ns), trig_delay(ns)
1000 0
2000 20
4000 40
```

所有 timing 值是 ns，必须对齐到 20 ns。Preview 不展开所有 scan rows，而是把
包含 `camera_exposure_ns` 或 `100000-camera_exposure_ns` 这类表达式的时间段标出来。

传给 camera acquisition 时，`exp.readout.detection_time(..., pulse=pulse)`
会用同一张 pulse 先拍 long-reference，再为每个扫描点临时生成刚好 `shots`
个外部触发的有限序列，保证相机先 arm，再由同一个 sequencer fire。

<!-- cell:code -->
pulse = exp.timing.bind_pulse("pulses/camera_imaging_address_switch.json")
pulse.snapshot()

# This does not fire hardware; it shows that the named variable controls the finite readout
# sequence duration before you run the scan.
test_widths_ns = [2_000_000, 4_000_000, 8_000_000]
[(width, pulse.frame_sequence(1, variables={"camera_exposure_ns": width}).duration) for width in test_widths_ns]

RUN_SINGLE_PULSE_TEST = False

single_program = None
if RUN_SINGLE_PULSE_TEST:
    pulse.set_variable("camera_exposure_ns", 2_000_000)  # ns
    single_program = pulse.on_pulse(wait=True, timeout=10.0, repeat_forever=False)
single_program

# Free-running output is still explicit when you want it:
# pulse.on_pulse(wait=False, repeat_forever=True)

<!-- cell:markdown -->
## Analog bus notes

The address-switch XDC also contains 10-bit TTL buses such as `da_dipole` and
`da_bias_x/y/z`. The GUI folds each bus into one logical analog row. A bus row
has three modes:

```text
edge: jump to a value at the beginning of the period
ramp: linearly move from the previous value to the target value over the period
hold: keep the current value; no numeric field is shown
```

The numeric field is a line edit, not a spinbox. For a 10-bit bus the GUI clamps
the value to `0..1023`. Preview draws one hollow stair-step line for the bus
value instead of drawing all ten TTL bits. The runtime uploads bus rows through
the FPGA analog-bus segment table, not by expanding every stair step into the
ordinary digital edge table, so the digital edge budget remains available for
lasers, shutters, camera, and trigger TTLs.

<!-- cell:markdown -->
## Scan detection time and fidelity

这个 scan 使用 camera images，不使用任何 ground truth。第一次上机默认同步跑完；
确认流程稳定后，下一格有一个显式的 live 版本：只把
`RUN_LIVE_READOUT_SCAN` 改成 `True`，其它 API 形状不变，仍然通过同一个
`pulse` 和 remote sequencer。

<!-- cell:code -->
clock_hz = exp.devices.sequencer.clock_hz
time_ticks = np.linspace(int(round(0.2e-3 * clock_hz)), int(round(8e-3 * clock_hz)), 40, dtype=int)
times = time_ticks / clock_hz
scan = exp.readout.detection_time(times, shots=30, live=False, display=True, pulse=pulse)
fit_result, popt = scan.data_figure.decay(is_display=False)
scan.summary(), fit_result, popt

<!-- cell:markdown -->
## Optional live readout-time scan

这个 cell 是控制电脑上最短的 live readout-time/fidelity 工作形状。它不会改
FPGA 电脑的 server，也不需要重新打开 GUI。

<!-- cell:code -->
RUN_LIVE_READOUT_SCAN = False

live_scan = None
if RUN_LIVE_READOUT_SCAN:
    live_scan = exp.readout.detection_time(times, shots=30, live=True, display=True, pulse=pulse)
live_scan
