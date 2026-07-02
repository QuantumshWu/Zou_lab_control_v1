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

默认硬件路线是 JTAG-to-AXI edge-table pulse streamer。server 从 board XDC
(`fpga/board_config/board.xdc` 的引脚图) 推断完整 channel order；GUI 或 API
可以只显示/配置其中几路，但上传时会自动补成 full-width mask，没配置的
channel 全部为 off。默认相机成像子集是：

```text
ch09 trap
ch00 cooling
ch03 probe
ch11 emCCD
```

The same XDC also has `ch06 trig`, but the checked-in camera preset uses
`ch11/emCCD/M13` as the qCMOS/emCCD trigger.

默认 clock 是 50 MHz，也就是 20 ns step。所有 duration、delay 和 scan
table 的值都会自动对齐(snap)到这个 step——硬件只能落在整数 tick 上。

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
## Discover attached devices（扫描端口自动发现）

连接前先扫一遍本机总线：`na.discover_devices()` 枚举 Basler (pypylon) 相机和
VISA 资源(每个资源用 `*IDN?` 短超时询问身份)。缺库/空总线不会报错，而是打印
一行提示(confocal 式)。每台相机的行自带 **ready config 片段**——直接塞进
device config 就能用，不用手抄序列号。

<!-- cell:code -->
found = na.discover_devices()
cameras = [d for d in found if d.config is not None]
cameras

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
  Channel Names:  display label、total duration、visible count
  Delay / Scan:   FPGA clock(只读)、per-channel delay(ns/us)+X 清除+clk 按钮
  Period cards:   duration/unit + scan 圆点(绑定 s0..)、DAC bus 行
                  (Edge/Ramp/Hold + 值 + scan 圆点)、每个 visible channel 的 on/off
  Control:        On/Stop、Add/Del Column、Add/Del Bracket、Save/Load
  Channels:       Add Channel、Hide Off、Show All

Preview tab
  自动画当前 PulseTableState，不需要手动 refresh。
  默认只画 active channel；Show off rows 会显示完整 channel list。
  被扫描的字段用透明橙色 band + slot 编号标出，不展开全部扫描点。

Scan tab
  已绑定 slot 列表；scan_table 的两种来源(代码生成 / Load Array 文件)；Run。
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
    repeat_forever=False,
)
# 数相机被触发几次（每次出一帧）是相机层的事：用 count_trigger_pulses，传相机
# 自己持有的 capture_trigger_channels（哪条 TTL 线触发相机是相机的属性，序列器不感知）。
imaging_seq = state.to_sequence()
{
    "ticks": program.ticks[:8],
    "masks": program.masks[:8],
    "trigger_count": na.count_trigger_pulses(
        imaging_seq, trigger_channels=exp.camera.capture_trigger_channels),
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

`capture` 是会话级编排（选相机、写曝光、arm→fire→读帧一条龙），只显示 raw camera
frame；site overlay 只属于 calibration/readout/detect 图。

<!-- cell:code -->
capture = exp.capture(frames=1, display=True)
capture.summary()

<!-- cell:markdown -->
## MOT 监视相机（Basler pylon，如 acA1920-155um）

第二只相机 `monitor_camera` 盯 MOT 荧光斑。`configs/basler_monitor.json` 只声明
**真实存在的硬件**：一台自由跑的 Basler（`serial=""` = 第一台，
`trigger_source="Software"`，插上 USB 就能看图；接好 FPGA 触发线后改 `"Line1"`）。
缺的角色（读出相机 / sequencer / trap）就是缺——session 对缺角色宽容，存在的角色
自然点亮，config 里绝不用假设备凑数。上面 `discover_devices()` 打印的序列号可以
直接填进 config 钉住某一台。

notebook 看图就是同一个 `capture`——真机自由跑时不需要任何脉冲：

```python
exp2 = na.connect("basler_monitor", open_devices=True)
mot = exp2.capture(camera="monitor_camera", frames=1, display=True)
```

task_console 里看图：`task_console.bat --config basler_monitor` 启动，
Add Panel → Measurement → **Camera (live frames)**，把 `Camera` 下拉切到
`monitor_camera`，Start——2d 面板实时显示 Basler 图像。配合 **MOT intensity**
处理器（Frame source 选该测量的 `frame_0`）就得到 MOT 亮度实时曲线；扫线圈找
最优磁场用 **Optimize MOT field** task（见 device manual"第二只相机"一章）。

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

`DetectionResult.occupied` 是后续 statistics 可以直接使用的
boolean array。

<!-- cell:code -->
shot = exp.readout.detect(display=True)
occupancy_grid = shot.occupied.reshape(grid_shape)
occupancy_grid, shot.summary()

<!-- cell:markdown -->
## Bind a pulse for hardware scans

对于 readout-time 或曝光宽度扫描，可以把一张 `PulseTableState` 绑定到当前
session 的 sequencer。仓库里的
`pulses/camera_imaging_address_switch.json` 已经把 `camera_exposure` period
绑定为 scan slot `s0`（`duration="s0", unit="str (ns)"`，nominal
19,980,000 ns），所以这张 pulse 的 exposure 是一个可设置/可扫描的命名量。

扫描用命名 slot + scan table（GUI 里点 duration/DAC 框旁的圆点绑定，
Scan 页提供表；API 里 `bind_field` + `set_scan_table`）：

```text
单点设置:  pulse.set_time(2_000_000)            # 第一个 duration slot, ns
           pulse.set_slot("s0", 2_000_000)      # 任意 slot 按名字设
硬件扫描:  pulse.set_scan_table([[w0], [w1], ...])   # N_points x N_slots, ns
```

所有值是 ns，会自动对齐到 20 ns tick。Preview 不展开所有 scan points，
而是把被扫描的时间段用透明橙色 band + slot 编号标出来。

传给 camera acquisition 时，`exp.readout.detection_time(..., pulse=pulse)`
会用同一张 pulse 先拍 long-reference，再为每个扫描点临时生成刚好 `shots`
个外部触发的有限序列，保证相机先 arm，再由同一个 sequencer fire。

<!-- cell:code -->
pulse = exp.timing.bind_pulse("pulses/camera_imaging_address_switch.json")
pulse.snapshot()

# This does not fire hardware; it shows that the exposure slot (s0) controls
# the finite readout sequence duration before you run the scan.
test_widths_ns = [2_000_000, 4_000_000, 8_000_000]
[(width, pulse.frame_sequence(1, time_ns=width).duration) for width in test_widths_ns]

RUN_SINGLE_PULSE_TEST = False

single_program = None
if RUN_SINGLE_PULSE_TEST:
    pulse.set_time(2_000_000)  # exposure slot s0, ns
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

The numeric field is a line edit, not a spinbox. Values are SIGNED LSB counts:
for a 10-bit bus the GUI clamps to `-512..+511`, and `0` means TRUE 0 V (the
driver is offset-binary; the wire code = value + 512 is produced by the
compiler, and an idle bus rests at 0 V). Preview draws one stair-step line for
the bus value instead of drawing all ten TTL bits, with the 0 V dashed
reference mid-row and negative values below it. The runtime uploads bus rows through
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

<!-- cell:markdown -->
## 实机 live 读出:Task 控制台

上面是脚本式流程。日常上机更常用 **Task 控制台**——一个 GUI,把 device / measurement / processor / task / plot 五层用 `SignalHub` 接好,Add Panel 即可搭出活读出板。它和脚本走**同一批建器**(虚拟 == 实机),所以这里连的是真相机/真 sequencer,操作和 `task_console_tutorial.ipynb` 里虚拟那套一模一样:

1. **Add Panel → Camera**:Edit 里设曝光/ROI,Start → 发 `frame`;再加 **2D plot** 选 `frame` 看活图。
2. **Add Panel → Task: Calibrate readout**:`pulse template` **Browse** 选你自己的成像程序(pulse GUI 存的 `pulses/camera_imaging_address_switch.json`),`folder` 选数据/报告目录,设曝光/帧数,Start。注意 **启动 task 会先停掉相机**(task 直接占用相机+sequencer,避免抢资源卡死),跑完标定自动成为会话标定。
3. **Add Panel → Judge occupancy**:`calibration` 留空用刚标好的,`source` 选 `frame`,`method` 选 box / per-site PSF / uniform PSF,Start → 发 `occupied`/`counts`/`rate`/`centers`/`frame_judged`。
4. **Add Panel → Site map**:只选一个信号 `occupied`(圆心+底图自动来自同一节点,环和图永远同一发)。

关窗(配 `on_close=exp.close`)会停掉所有采集线程并安全断开 device——不会有线程占着相机或 RPyC 链路。

<!-- cell:code -->
# 实机 live 读出板。桌面 Qt 环境里取消注释(本 notebook 其余格用的同一个已连接 exp)。
from Zou_lab_control.neutral_atom.core.signals import SignalHub

# console = zf.show_task_console(
#     hub=SignalHub(),
#     session=exp,
#     measurements=exp.readout.measurement_specs(),
#     processors=exp.readout.processor_specs(),
#     tasks=exp.readout.task_specs(),
#     on_close=exp.close,   # 关窗 = 停所有节点线程 + 安全断开真机 device
# )
# console
