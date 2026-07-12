<!-- cell:code -->
{{BOOTSTRAP_CELL}}

<!-- cell:code -->
import time
import numpy as np
import matplotlib.pyplot as plt

import Zou_lab_control.frontend as zf

try:
    zf.use_widget_backend()
except Exception as exc:
    print(f"Widget backend not enabled here: {exc}")

zf.enable_long_output()
zf.apply_style()

<!-- cell:markdown -->
## 1D plot, title, and fitting

`zf.plot(x, y)` 的输入契约是 `x: (N, coord_dim)` 和 `y: (N, channel_dim)`。静态图和 live 图都用这套契约；差别只是 live 时有 worker 持续写入共享 array。

<!-- cell:code -->
x = np.linspace(737.0, 737.2, 301).reshape(-1, 1)
y = 18 * ((0.018 / 2) ** 2) / ((x[:, 0] - 737.095) ** 2 + (0.018 / 2) ** 2) + 3
y = (y + np.random.default_rng(3).normal(0, 0.25, size=len(x))).reshape(-1, 1)

ple = zf.plot(
    x,
    y,
    labels=("Wavelength (nm)", "Counts/0.1s", "Counts"),
    relim_mode="tight",
    title="Lorentzian scan",
)
fit_result = ple.data_figure.fit("lorent")
fit_result, fit_result.popt

<!-- cell:markdown -->
## 2D plot

对外的 2D plot 始终保持 square view，避免主图、distribution axis 和 colorbar 视觉错位。内部如果要调试非 square，可以直接使用 `Live2DDis`，但 notebook 正常调用不开放 `square=False`。

<!-- cell:code -->
scan_x_axis = np.linspace(-8, 8, 49)
scan_y_axis = np.linspace(-6, 6, 37)
SX, SY = np.meshgrid(scan_x_axis, scan_y_axis)
Z = 1200 * np.exp(-((SX - 1.5) ** 2 + (SY + 0.8) ** 2) / 12) + 80
Z += np.random.default_rng(4).normal(0, 15, size=Z.shape)
Z[3:8, 3:8] = np.nan

map_x = np.column_stack([SX.ravel(), SY.ravel()])
map_y = Z.ravel().reshape(-1, 1)

pl_map = zf.plot(map_x, map_y, labels=("X (um)", "Y (um)", "Counts/50ms"), title="2D count map")
pl_map.data_figure.fit("center")

<!-- cell:markdown -->
## Pulse sequence plot

`kind="pulse"` 使用实心色块显示 on 区间，同时保留每个 channel 的 off baseline。baseline 和 block 使用同一个颜色、同一个 alpha，只是 baseline 是细线，on interval 是从 baseline 向上长出的实心块。y 轴 label 与该 channel 的 pulse 颜色一致，10 个 channel 仍然可读。x 轴会按总时长自动选择 `ns/us/ms/s`，避免时序图上全是很长的科学计数法秒数。

<!-- cell:code -->
channels = ["trap", "cooling", "probe", "trig", "pushout", "microwave", "aod_x", "aod_y", "repump", "camera_gate"]
pulses = [
    {"channel": channel, "start": i * 1.4e-6, "duration": 0.9e-6 + (i % 3) * 0.16e-6, "value": 1, "name": channel}
    for i, channel in enumerate(channels)
]

pulse_plot = zf.plot(
    pulses,
    kind="pulse",
    channels=channels,
    labels=("Time (s)", "", "State"),
    title="10-channel timing check",
)

<!-- cell:markdown -->
## Pulse table model and PyQt pulse GUI

`PulseTableState` 是 pulse GUI 和 notebook 共用的 period-card 模型。GUI 是可选前端；不打开 GUI 时，也可以直接用这个模型生成 `PulseSequence`。它只接收一个不可变 `PortCatalog` 作为拓扑真相：raw FPGA lane 顺序、逻辑 digital/DAC/clock port、只读 label、DAC lane ownership 与 latch clock 都在 catalog 里，并由 fingerprint 锁定。pulse JSON 嵌入同一份 catalog，不保存任何平行拓扑镜像。standalone `pulse_gui.bat` 在设备/XDC 边界构造 catalog；编辑器只读，不允许 pulse 文档改硬件拓扑。`visible_ports` 只是当前视图子集，不创建、删除或重排端口。`time_step_ns` 是 minimal time，连接默认 FPGA server 时是 20 ns；所有 duration、delay 和 scan table 值都要落在它的整数网格上。扫描轴的公开身份是 `ScanSlot.name`；`s0/s1/...` 只是编译器在表内使用的 token。Preview 页保留这些 symbolic binding，不把 scan table 展开成大量 period columns。

Pulse GUI 的 Edit 页可以按这个顺序读：

```text
Port Catalog:    pulse 名字、总时长、只读 port label/kind/lane 与 catalog fingerprint。
Delay / Scan:    FPGA clock(只读显示)、每个可编程 port 的 delay(ns/us)+X 清除按钮。
Period cards:    每个 period 的 duration/unit + scan 圆点(绑定语义 ScanSlot.name)、
                 DAC bus 行(Edge/Ramp/Hold + 值 + scan 圆点)、channel on/off。
Control:         On Pulse/Stop Pulse、Sync、Add Period/Remove、Add Bracket、
                 Save/Load、Collapse。
Ports:           Add visible、Hide Off、Show All 和 visible/hidden 计数；只改变 visible_ports。
Scan tab:        已绑定 slot 列表、代码生成/Load Array 两种 scan_table 来源、Run。
```

Port Catalog 面板显示 catalog 给出的只读 label，例如 `trap/cooling/probe/emCCD`，
tooltip 同时显示 logical key、kind、raw lane 与 XDC package pin。保存、编译和上传都
携带并校验同一 fingerprint，所以不存在 GUI label 覆盖硬件 bit order 的第二份真相。
Preview y 轴使用 logical port label；`Show off rows` 只扩展视图到 catalog 的完整
可编程 port 集合，不会把 DAC 的 raw lanes 重新暴露成可编辑 TTL。

`On Pulse` 的语义和 API 一样：先读取当前 GUI state，按 attached sequencer 的
clock 与 `PortCatalog.raw_lanes` 编译成 full-width edge table，先校验 pulse catalog
fingerprint 与 attached sequencer 完全一致，再 `prepare` 上传和 `fire`。如果 GUI
只显示四个 port，上传仍然是 catalog 的完整硬件宽度；未显示的可编程 port 保持其
明确安全状态。`Stop Pulse` 调用 sequencer safe/reset。`Sync`
把设备上实际生效的脉冲程序拉回编辑器（sequencer 会记录每次成功 prepare 的
PulseTableState 来源，无论它来自这个 GUI 还是 notebook/raw API 调用，比如
`PulseController.on_pulse`）——在 GUI 之外改了设备后点它，GUI 就重新反映设备
状态。等待 finite acquisition 完成属于 notebook/camera API。

<!-- cell:code -->
import Zou_lab_control.neutral_atom as na

catalog = na.PortCatalog.from_channels(
    [f"ch{i:02d}" for i in range(62)],
    channel_labels={"ch09": "trap", "ch00": "cooling", "ch03": "probe", "ch11": "emCCD"},
)
pulse_state = na.PulseTableState(
    port_catalog=catalog,
    visible_ports=["ch09", "ch00", "ch03", "ch11"],
    time_step_ns=20,
)
pulse_state.set_period_state(0, "ch09", 1)
pulse_sequence = pulse_state.to_sequence(time_step_ns=20)
pulse_state.total_duration_steps(time_step_ns=20)

api_pulse_plot = zf.plot(
    pulse_sequence,
    kind="pulse",
    channels=pulse_state.port_catalog.raw_lanes,
    title="PulseTableState API sequence",
)

# Uncomment on a desktop Python/Qt environment:
# pulse_gui = zf.show_pulse_gui(state=pulse_state, scale=0.82, window_ratio=0.90)

<!-- cell:markdown -->
扫描用**有语义名字的 `ScanSlot`**。`bind_field(kind, target, label=...)` 把 duration 或 DAC 字段绑定到一个公开名字；`s0/s1/...` 只是在 period 表达式与 affine 编译器里的列 token，不是 UI、plot 或结果数据的轴名。`set_scan_table` 提供 `N_points x N_slots` 的表；`compile_scan` 把整张表编译成**一个**硬件 program——FPGA 在扫描点之间无缝切换，不逐点重新上传。GUI 里的同一件事是：绑定字段、给 slot 命名，再到 Scan 页生成或 Load Array。下面这个例子不打开 GUI，只用 API 扫 `image width`。

<!-- cell:code -->
scan_catalog = na.PortCatalog.from_channels(
    ["ch09", "ch03", "ch11"],
    channel_labels={"ch09": "trap", "ch03": "probe", "ch11": "emCCD"},
)
scan_state = na.PulseTableState(
    port_catalog=scan_catalog,
    time_step_ns=20,
    periods=[
        na.PulsePeriod(1000, (1, 0, 0), unit="ns", name="pre"),
        na.PulsePeriod(240, (1, 1, 1), unit="ns", name="image"),
        na.PulsePeriod(1000, (0, 0, 0), unit="ns", name="idle"),
    ],
)
scan_state.bind_field(
    "duration", "1", label="image width", name="image_width")
scan_state = scan_state.set_scan_table([[240], [500], [1000], [2000]])  # N_points x N_slots, ns

scan_program = scan_state.compile_scan(clock_hz=50_000_000)
scan_program.scan_points, scan_program.ticks  # 一个 program 携带全部扫描点(tick 单位)

<!-- cell:code -->
# 单点检查：with_slots_resolved 按公开名 image_width 换成具体值，
# 得到一张普通的静态表；compile(slots=...) 等价。
single = scan_state.with_slots_resolved({"image_width": 500})
single_program = single.compile(clock_hz=50_000_000)
[(width, scan_state.with_slots_resolved({"image_width": width})
                   .total_duration_steps(time_step_ns=20))
 for width in [240, 500, 1000, 2000]], single_program.ticks

<!-- cell:markdown -->
For real hardware, do not let the GUI invent hardware. Start the server on the
FPGA/Vivado computer, create one installation-owned experiment session, and open
the editor through that session:

```python
exp = na.connect(
    "remote_template",
    sequencer={"host": "FPGA_SERVER_IP", "port": 18861},
    open_devices=True,
)
# 哪条线触发相机（emCCD/M13）是相机的属性（camera.capture_trigger_channels），
# 不是序列器的——序列器是纯脉冲流送器，不感知谁被它触发。
gui = exp.pulse_gui(
    state=na.PulseTableState.load("pulses/camera_imaging_address_switch.json"),
)
```

In normal camera acquisition, prefer the higher-level readout helper because it
arms qCMOS first and then fires a finite trigger sequence. Free-running
`repeat_forever=True` is useful for scope checks, not for a finite camera stack.

Analog bus rows such as `da_dipole` or `da_bias_x/y/z` are folded views of
10-bit TTL groups. Their GUI value field is a line edit clamped to the SIGNED range `-512..+511` (0 = true 0 V; the offset-binary wire code = value + 512 is produced by the compiler).
Preview draws one hollow stair-step analog trace. The runtime uploads these rows
through the FPGA analog-bus segment table, so a long bus ramp costs one bus
segment instead of one ordinary TTL `prog_mask` edge per stair step.

<!-- cell:markdown -->
## Live 2D scan

`zf.run` 接收采集函数 handle。worker 负责采集，frontend timer 负责刷新图；调用者不需要自己建线程或手动维护 controller。

<!-- cell:code -->
scan_x_axis = np.linspace(-4, 4, 25)
scan_y_axis = np.linspace(-3, 3, 19)
SX, SY = np.meshgrid(scan_x_axis, scan_y_axis)
live_scan_x = np.column_stack([SX.ravel(), SY.ravel()])

def measure_scan(point):
    px, py = point
    time.sleep(0.002)
    return 400 * np.exp(-((px - 0.8) ** 2 + (py + 0.3) ** 2) / 5) + 30

live_scan = zf.run(
    live_scan_x,
    measure_scan,
    labels=("X", "Y", "Counts"),
    update_time=0.05,
)
time.sleep(1.2)
live_scan.stop()
live_scan.points_done

<!-- cell:markdown -->
## Histogram with draggable threshold

右上角显示当前 threshold、双峰 Gaussian fidelity、左右比例和 `fit cut`。`fit cut` 是模型建议的交点，不会覆盖你拖动的实际 threshold。

<!-- cell:code -->
rng = np.random.default_rng(6)
shots = np.r_[rng.normal(20, 4, 250), rng.normal(78, 8, 350)]

hist = zf.plot(
    shots,
    kind="hist",
    bins=55,
    thresholds=[45],
    labels=("ROI counts", "Shots", "Population"),
    title="Threshold calibration",
)
hist.fractions(), hist.stats_text.get_text()

<!-- cell:markdown -->
## Continuous monitor without auto stop

`stop_when_full=False` 用于长期 monitor。实验中不需要 `.wait()`；这里为了 notebook 自动执行，最后会显式 stop。

<!-- cell:code -->
continuous_x = np.arange(200).reshape(-1, 1)
continuous_rng = np.random.default_rng(8)

def read_continuous_count():
    time.sleep(0.002)
    return continuous_rng.poisson(50)

continuous_monitor = zf.run(
    continuous_x,
    read_continuous_count,
    kind="monitor",
    mode="roll",
    stop_when_full=False,
    labels=("Recent shots", "Counts/shot", "Counts"),
    update_time=0.05,
    max_points=80,
)
time.sleep(0.35)
continuous_monitor.stop()
continuous_monitor.points_done

<!-- cell:code -->
for name in ["live_scan", "continuous_monitor"]:
    obj = globals().get(name)
    if obj is not None and hasattr(obj, "stop"):
        obj.stop()

<!-- cell:markdown -->
## 把这些组件接成一台实验:Task 控制台

上面是 frontend 的各个**单件**(plot / pulse GUI / live scan)。把它们按 device → measurement → processor → task → plot 五层接成一台跑读出实验的 GUI,看 **`task_console_tutorial.ipynb`**——`zf.show_task_console(...)` 一行打开,Add Panel 搭出相机活图 + 校准 + 判 occupancy + site map,虚拟和真机用同一套调用。
