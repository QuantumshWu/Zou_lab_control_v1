<!-- cell:markdown -->
# Zou_lab_control.frontend tutorial

这个 notebook 展示统一的 Jupyter 画图接口。第一格直接把 `..` 加入 `sys.path` / `PYTHONPATH`，然后导入 `Zou_lab_control.frontend`，不需要先安装本仓库。

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
fit_result, popt = ple.data_figure.lorent()
fit_result, popt

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
pl_map.data_figure.center()

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

`PulseTableState` 是 pulse GUI 和 notebook 共用的 period-card 模型。GUI 是可选前端；不打开 GUI 时，也可以直接用这个模型生成 `PulseSequence`。新建 pulse 默认名是 `pulse_YYYYMMDD_HHMMSS`。`channels` 是硬件 channel 名和 FPGA bit order，例如 `ch00/ch01/...`；display label 只是前端名字。standalone `pulse_gui.bat` 会从 address-switch XDC 推断完整 channel list、display label 和 package pin；JSON 里保存过的 `channel_labels` 优先。`time_step_ns` 是 minimal time，连接默认 FPGA server 时是 20 ns。所有 duration、delay 和 scan array 的值都要是它的整数倍。扫描用命名 slot：给任意 duration 或 DAC 字段绑定一个 slot(`s0, s1, ...`)，再提供一张 `N_points x N_slots` 的 `scan_table`（channel delay 是固定的每通道值，不能扫描）。GUI 默认只显示常用子集，其它 channel 可以在 GUI 里临时添加或隐藏；隐藏不改变上传宽度，compile/upload 会自动补齐完整硬件 channel order。Preview 页自动调用 `zf.plot(..., kind="pulse")`，默认隐藏 off-only channel，并保留 symbolic slot 标记，不把 scan array 展开成大量 period columns。

Pulse GUI 的 Edit 页可以按这个顺序读：

```text
Channel Names:   pulse 名字、总时长、可见 channel 的 display name。
Delay / Scan:    FPGA clock(只读显示)、每通道 delay(ns/us)+X 清除按钮+clk 按钮。
Period cards:    每个 period 的 duration/unit + scan 圆点(绑定 s0..)、
                 DAC bus 行(Edge/Ramp/Hold + 值 + scan 圆点)、channel on/off。
Control:         Stop Pulse、On Pulse、Add/Del Column、Add/Del Bracket、Save/Load。
Channels:        Add Channel、Hide Off、Show All 和 visible/hidden 计数。
Scan tab:        已绑定 slot 列表、代码生成/Load Array 两种 scan_table 来源、Run。
```

Name 面板左侧 raw column 在 standalone address-switch 路线下显示 XDC package pin，
例如 `M17/F15/N15/M13`，不是 `ch09/ch00/ch03/ch11`。硬件 bit 名仍然保存在
tooltip、JSON 和 API state 中，所以保存、编译和上传不会丢失真正的 channel order。
Preview y 轴显示 display label，例如 `trap/cooling/probe/emCCD`；如果打开
`Show off rows`，它会显示完整硬件 channel list，但 y 轴仍然不显示总标题 `Pulse`。

`On Pulse` 的语义和 API 一样：先读取当前 GUI state，按 attached sequencer 的
clock/channel list 编译成 full-width edge table，`prepare` 上传，再 `fire`。如果
GUI 只显示四路，上传仍然是完整 address-switch channel 宽度；没显示、没配置或被
隐藏的 channel mask bit 都是 0。`Stop Pulse` 调用 sequencer safe/reset。GUI 没有
独立 sync 按钮；等待 finite acquisition 完成属于 notebook/camera API。

<!-- cell:code -->
import Zou_lab_control.neutral_atom as na

pulse_state = na.PulseTableState(
    channels=[f"ch{i:02d}" for i in range(62)],
    visible_channels=["ch09", "ch00", "ch03", "ch11"],
    channel_labels={"ch09": "trap", "ch00": "cooling", "ch03": "probe", "ch11": "emCCD"},
    time_step_ns=20,
)
pulse_state.set_period_state(0, "ch09", 1)
pulse_sequence = pulse_state.to_sequence(time_step_ns=20)
pulse_state.total_duration_steps(time_step_ns=20)

api_pulse_plot = zf.plot(
    pulse_sequence,
    kind="pulse",
    channels=pulse_state.channels,
    title="PulseTableState API sequence",
)

# Uncomment on a desktop Python/Qt environment:
# pulse_gui = zf.show_pulse_gui(state=pulse_state, scale=0.82, window_ratio=0.90)

<!-- cell:markdown -->
扫描用**命名 scan slot**（`s0, s1, ...`）。`bind_field(kind, target)` 把一个字段绑定成下一个 slot：`kind="duration"` 时 `target` 是 period 序号，`kind="dac"` 时是 `"bus@period"`（channel delay 是固定量，不能绑定）。绑定后该字段的值变成 slot 表达式（如 `"s0"`），再用 `set_scan_table` 提供一张 `N_points x N_slots` 的表；`compile_scan` 把整张表编译成**一个**硬件 program——FPGA 在扫描点之间无缝切换（affine tick：`effective_tick = base + (coeff*s0)>>8`），不需要逐点重新上传。GUI 里的同一件事是：点 duration/DAC 输入框右侧的圆点（变橙色、显示 slot 号），再到 Scan 页生成或 Load Array 一张 scan table。下面这个例子不打开 GUI，只用 API 扫 `image` period 的宽度。

<!-- cell:code -->
scan_state = na.PulseTableState(
    channels=["ch09", "ch03", "ch11"],
    channel_labels={"ch09": "trap", "ch03": "probe", "ch11": "emCCD"},
    time_step_ns=20,
    periods=[
        na.PulsePeriod(1000, (1, 0, 0), unit="ns", name="pre"),
        na.PulsePeriod(240, (1, 1, 1), unit="ns", name="image"),
        na.PulsePeriod(1000, (0, 0, 0), unit="ns", name="idle"),
    ],
)
scan_state.bind_field("duration", "1", label="image width")   # period 1 duration -> s0
scan_state = scan_state.set_scan_table([[240], [500], [1000], [2000]])  # N_points x N_slots, ns

scan_program = scan_state.compile_scan(clock_hz=50_000_000, trigger_channels=["ch11"])
scan_program.scan_points, scan_program.ticks  # 一个 program 携带全部扫描点(tick 单位)

<!-- cell:code -->
# 单点检查：with_slots_resolved 把 s0 换成一个具体值(其余 slot 保持 nominal)，
# 得到一张普通的静态表；compile(slots=...) 等价。
single = scan_state.with_slots_resolved({"s0": 500})
single_program = single.compile(clock_hz=50_000_000, trigger_channels=["ch11"])
[(width, scan_state.total_duration_steps(slots={"s0": width}, time_step_ns=20))
 for width in [240, 500, 1000, 2000]], single_program.ticks

<!-- cell:markdown -->
For real hardware, do not let the GUI invent hardware. Start the server on the
FPGA/Vivado computer, then attach the same `RemoteSequencer` from GUI or API:

```python
sequencer = na.RemoteSequencer(
    host="192.168.0.20",
    port=18861,
    channels=[f"ch{i:02d}" for i in range(62)],
    clock_hz=50_000_000,
    trigger_channels=["ch11"],  # emCCD/M13 in the checked-in address-switch XDC
)
gui = zf.show_pulse_gui(
    state=na.PulseTableState.load("pulses/camera_imaging_address_switch.json"),
    sequencer=sequencer,
)
```

The API equivalent of pressing `On Pulse` is:

```python
state = na.PulseTableState.load("pulses/camera_imaging_address_switch.json")
program = state.compile(clock_hz=50_000_000, trigger_channels=["ch11"])
sequencer.prepare(program)
sequencer.fire()
```

In normal camera acquisition, prefer the higher-level readout helper because it
arms qCMOS first and then fires a finite trigger sequence. Free-running
`repeat_forever=True` is useful for scope checks, not for a finite camera stack.

Analog bus rows such as `da_dipole` or `da_bias_x/y/z` are folded views of
10-bit TTL groups. Their GUI value field is a line edit clamped to `0..1023`.
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
