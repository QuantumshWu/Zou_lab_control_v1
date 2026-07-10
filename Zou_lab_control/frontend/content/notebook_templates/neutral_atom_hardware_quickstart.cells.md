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
一行提示(confocal 式)。每台相机的行自带 **ready config 片段**——把 `row.config` 直接给
`na.load_devices({角色: row.config})` / `na.connect(...)` 就能用，闭环 discover→select→connect，
不用手抄序列号。

<!-- cell:code -->
found = na.discover_devices()
cameras = [d for d in found if d.config is not None]
# 闭环:把发现到的 ready config 按角色喂进 loader(有真相机时;这里没接硬件则 cameras 为空)。
if cameras:
    devset = na.load_devices({"monitor_camera": cameras[0].config})
cameras

<!-- cell:markdown -->
## 接入你自己的硬件（自定义设备类 / 发现提供者）

`discover_devices()` 只认识**注册表里、且实现了 `discover()` 的类**（内置的 Basler 相机 + VISA
provider）。你的相机 / DAQ / 线圈电源不在其中时，**不用改本仓库源码**——三个扩展点就够
（和 confocal 的 `lookup_dict` 同源）：

1. **写一个设备类**——继承正确的领域基类（`na.CameraDevice` / `na.SequencerDevice` /
   `na.TrapArrayDevice`），实现它的契约。相机契约 = `exposure` + `configure(...)` + 纯 grabber
   三原语 `arm` / `read_frames` / `disarm`（无损缓冲逻辑在**基类**，子类只把到达的帧喂进
   `self._deliver(frame)`）。可选：加一个 `discover()` classmethod 让它**自报家门**——扫到就
   返回一行带 **ready config** 的 `na.DiscoveredDevice`；缺库 / 空总线用
   `na.discovery_note(kind, "pip install ...")` 报一行、**绝不 raise**（和内置 Basler 的
   `discover()` 一模一样）。
2. **注册 / 引用**——`na.register_device_class("MyCam", MyCam)`（或传可导入路径字符串
   `"my_pkg.cams:MyCam"` 延迟导入），之后任何 config 里 `{"type": "MyCam"}` 按短名引用；只在
   本 notebook **临时**用则 `na.load_devices({...}, lookup=globals())`——把这里定义的类直接拿来
   构造，**零注册、不写全局注册表**（`lookup` 命中是 per-call 的，不留痕）。
3. **class-less 总线**——一台没有任何设备类认领的仪器（一堆串口 / site-specific 机箱），写一个
   零参扫描函数，`na.register_discovery_provider("mybus", scan)` 注册进 `discover_devices()`
   （VISA 就是内置的这种 provider，逐个资源发 `*IDN?`）。

下面这格**不接任何真硬件就能跑**：定义一个占位相机类，注册它，看它出现在扫描里、并被
`load_devices` 按短名 / 按 `globals()` 构造出来。把 `DemoCamera` 换成你自己的相机 SDK 封装，
真机上流程一字不变。**设备只管硬件动作**；标定 / 检测算法属于 `core/`，别塞进设备类。完整的
“编写并注册一个自定义设备”一章见 **device manual**。

<!-- cell:code -->
# 一个占位相机:换成你自己的相机 SDK 封装即可(这里不接硬件,只演示注册/发现/构造的接线)。
class DemoCamera(na.CameraDevice):
    def __init__(self, *, serial="", exposure=3e-3):
        self._serial = str(serial)
        self._exposure = float(exposure)

    @property
    def exposure(self):
        return self._exposure

    @exposure.setter
    def exposure(self, value):
        self.configure(exposure=float(value))     # 唯一硬件写入口

    def configure(self, *, exposure=None, **kwargs):
        self._reject_unknown_configure_keys({"exposure"}, kwargs)   # 未知键大声报错
        if exposure is not None:
            self._exposure = float(exposure)
        # 真机:这里把曝光写进相机;帧到达时子类调用 self._deliver(frame)。

    @classmethod
    def discover(cls):
        # 自报家门:返回带 ready config 的行(缺库/空总线改用 na.discovery_note(...),绝不 raise)。
        return [na.DiscoveredDevice(
            kind="demo", ident="demo-0001", label="DemoCamera (example)",
            config={"type": cls.__name__, "params": {"serial": "demo-0001"}})]


# (1)+(2) 注册短名 -> 它自报家门出现在扫描里(和 Basler/VISA 并列),且带 ready config:
na.register_device_class("DemoCamera", DemoCamera)
demo_rows = [row for row in na.discover_devices(display=False) if row.kind == "demo"]

# (2') 零注册路径:notebook 里定义的类,直接用 lookup=globals() 构造(不写全局注册表):
built = na.load_devices({"monitor_camera": {"type": "DemoCamera", "params": {"serial": "x"}}},
                        lookup=globals())

# (3) class-less 总线:注册一个自定义发现 provider(VISA 是内置的同类)。
def scan_my_bus():
    return [na.discovery_note("mybus", "example provider: 0 instruments on this bus")]

na.register_discovery_provider("mybus", scan_my_bus)
provider_row = [row for row in na.discover_devices(display=False) if row.ident == "mybus"]

# 列出一个 DeviceSet 里某角色类型的设备 = 角色通用的单源 device_names(base_type)（camera_names()
# 只是它的相机薄封装）。平时你不用手列——每个用相机的 measurement 表单里自带 Camera 下拉，
# GUI 里也可 exp.device_manager() 一眼看全。
demo_rows[0].config, built.device_names(na.CameraDevice), provider_row[0].label

<!-- cell:markdown -->
## Connect hardware

`na.connect(..., open_devices=True)` 会通过 device loader 构造、校验并打开
camera/sequencer。把 `host` 改成 FPGA/Vivado 电脑的 IP。

<!-- cell:code -->
exp = na.connect(
    "remote_template",
    sequencer={"host": "FPGA_SERVER_IP", "port": 18861},   # 改成你 FPGA/Vivado 电脑的 IP
    open_devices=True,
)

# First-light manual trigger path:
# exp = na.connect("manual_template", open_devices=True)

exp

<!-- cell:markdown -->
## See your devices（设备管理 GUI）

连上后 `exp.device_manager()` 开一个窗口，按角色类型（Camera / Sequencer / Trap array / 你注册的
RF …）列出这次配置真正载入的每台设备，并有 “Scan hardware” 现场扫总线——就是上面
`na.discover_devices()` / `na.load_devices()` 的图形面。task console 顶栏也有个 “Devices” 按钮开同一
窗口。之后每个用相机的测量表单里都自带 **Camera 下拉**，从这些设备里挑哪台跑该测量（单相机时
用默认，双相机时挑读出相机还是 `monitor_camera`）。

<!-- cell:code -->
exp.device_manager()

<!-- cell:markdown -->
## Save / load an experiment config（设备管理 GUI 里存取）

一次配置（`{角色: {"type", "params"}}` 这张 JSON 表，就是 `na.connect(...)` 吃的东西）往往是
你现场扫总线、填串号、调曝光**试出来**的，值得存下来下次直接复用——不用再从头连一遍。三个动作，
notebook API 和 device manager GUI **同一套**：

- **存**：`exp.save_config("configs/my_experiment.json")` 把当前 DeviceSet 序列化成 JSON
  （`.json` 后缀会自动补）。GUI 里点 **Save config…**，同一个文件对话框。
- **载**：`na.connect("configs/my_experiment.json")` 从文件**新开**一个 session；已经有 session 时用
  `exp.load_config(path)` 就地把设备**换成**该配置（会重建 imaging sequence、清掉旧标定）。GUI 里点
  **Load config…**，设备列表随即刷新成新配置。
- **开**：`exp.open_devices()` 真正连接 / 初始化硬件（等价于 `na.connect(..., open_devices=True)`；
  之前 `open_devices=False` 只构造不打开时用它补上一步）。GUI 里点左上角绿色 **Open devices**。

GUI 的这三个按钮就在 device manager 顶栏，默认目录指向仓库的 `configs/`。下面几格演示 API 侧；
文件路径按你机器改。

<!-- cell:code -->
# 存下这次连接的配置(扫总线 / 填串号 / 调曝光试出来的那套),下次一行连回来。
saved = exp.save_config("configs/my_experiment.json")   # 返回真正写出的 Path(.json 自动补)
print("saved ->", saved)

# 换台机器 / 下次开工:从文件新开一个 session(和 GUI 的 Load config… 等价)。
# exp2 = na.connect("configs/my_experiment.json", open_devices=True)

# 已有 session 时就地换配置(重建 imaging sequence、清旧标定),再按需开硬件:
# exp.load_config("configs/my_experiment.json")
# exp.open_devices()

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

GUI 只是 pulse 前端。它读取 `exp.devices.sequencer.port_catalog`，编辑
引用同一不可变 catalog 的 `PulseTableState`，然后在 `On Pulse/Stop Pulse` 按钮里调用同一个
sequencer。`On Pulse` 会先把当前 pulse state 上传到 sequencer，再立刻
start；`Stop Pulse` 调用 safe/reset。

如果当前环境没有桌面/Qt，跳过这个 cell，继续用
`exp.timing.configure_imaging(...)` 和 API 配置 pulse。

Pulse GUI 的实际工作方式：

```text
Edit tab
  Port Catalog:   只读 logical port/label/kind/lane、fingerprint、total duration
  Delay / Scan:   FPGA clock(只读)、per-port delay(ns/us)+X 清除
  Period cards:   duration/unit + scan 圆点(绑定语义 ScanSlot.name)、DAC bus 行
                  (Edge/Ramp/Hold + 值 + scan 圆点)、每个 visible channel 的 on/off
  Control:        On/Stop、Add/Del Column、Add/Del Bracket、Save/Load
  Ports:          Add visible、Hide Off、Show All（只改变 visible_ports）

Preview tab
  自动画当前 PulseTableState，不需要手动 refresh。
  默认只画 active port；Show off rows 会显示 catalog 的完整可编程 port 集合。
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
绑定为公开名 `exposure` 的 scan slot（内部当前列 token 为 `s0`，nominal
19,980,000 ns），所以这张 pulse 的 exposure 是一个可设置/可扫描的命名量。

扫描用命名 slot + scan table（GUI 里点 duration/DAC 框旁的圆点绑定，
Scan 页提供表；API 里 `bind_field` + `set_scan_table`）：

```text
单点设置:  pulse.set_slot("exposure", 2_000_000) # 公开语义名, ns
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

# This does not fire hardware; it shows that the semantic exposure slot controls
# the finite readout sequence duration before you run the scan.
test_widths_ns = [2_000_000, 4_000_000, 8_000_000]
[(width, pulse.frame_sequence(1, time_ns=width).duration) for width in test_widths_ns]

RUN_SINGLE_PULSE_TEST = False

single_program = None
if RUN_SINGLE_PULSE_TEST:
    pulse.set_slot("exposure", 2_000_000)  # semantic scan slot, ns
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
fit_result = scan.data_figure.fit("decay")
scan.summary(), fit_result, fit_result.popt

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

1. **Add Panel → Camera**:Edit 里设曝光/ROI,Start → 按 pulse 的触发事件发 `frame_0`/`frame_1`/...；单触发 live 源再加 **2D plot** 选 `frame_0` 看活图。
2. **Add Panel → Task: Calibrate readout**:`pulse template` **Browse** 选你自己的成像程序(pulse GUI 存的 `pulses/camera_imaging_address_switch.json`),`folder` 选数据/报告目录,设曝光/帧数,Start。注意 **启动 task 会先停掉相机**(task 直接占用相机+sequencer,避免抢资源卡死),跑完标定自动成为会话标定。
3. **Add Panel → Judge occupancy**:`calibration` 留空用刚标好的,`source` 选对应触发事件（单触发时为 `frame_0`）,`method` 选 box / per-site PSF / uniform PSF,Start → 发 `occupied`/`counts`/`rate`/`centers`/`frame_judged`。
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
