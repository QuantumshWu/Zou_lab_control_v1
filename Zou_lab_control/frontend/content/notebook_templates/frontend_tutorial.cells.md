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
channels = ["trap", "cooling", "probe", "qcm_trigger", "pushout", "microwave", "aod_x", "aod_y", "repump", "camera_gate"]
pulses = [
    {"channel": channel, "start": i * 1.4e-6, "duration": 0.9e-6 + (i % 3) * 0.16e-6, "value": 1, "name": channel}
    for i, channel in enumerate(channels)
]

pulse_plot = zf.plot(
    pulses,
    kind="pulse",
    channels=channels,
    labels=("Time (s)", "Pulse", "State"),
    title="10-channel timing check",
)

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
