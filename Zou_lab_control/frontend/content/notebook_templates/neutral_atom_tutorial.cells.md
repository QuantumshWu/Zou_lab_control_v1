<!-- cell:markdown -->
# Neutral atom quickstart

这个 notebook 展示第一版轻量中性原子实验线路：连接 device，配置 pulse sequence，拍 camera 图，校准 sitemap，校准 threshold，探测 occupancy，最后得到 detection time 和 fidelity 曲线。

第一格直接把 `..` 加入 `sys.path` / `PYTHONPATH`，然后导入 `Zou_lab_control.frontend`，不需要先安装本仓库。

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
## Architecture shape

推荐调用边界：

- `na.BaseDevice` / `na.CameraDevice` / `na.SequencerDevice` / `na.TrapArrayDevice`：硬件契约。真实 camera 至少要满足 `exposure`、`configure(...)`、`acquire(frames, sequence=..., sequencer=...)`。
- `na.load_devices(...)`：只负责按 JSON/dict 构造 device graph，并要求每个 device 继承对应 base class。
- `exp.camera`：真实 camera device 本体，`capture()` 是 camera device 方法。
- `exp.readout`：camera readout subsystem，包含 sitemap、threshold、detect、detection-time fidelity calibration。
- `exp.timing.*`：pulse sequence、preflight、Verilog 生成。

分层原则是：`operations` 里的函数可以 standalone 处理 array；`ReadoutSubsystem` 使用当前 `exp` 的 camera/defaults/calibration 去调度这些 operation；result object 负责把 raw data、plot 和 summary 带回来。

当前源码也按这个边界放置：

```text
neutral_atom/
  core/        # analysis, TrapCalibration, ResultObject
  devices/     # BaseDevice, registry, virtual, qCMOS, sequencer adapters
  timing/      # PulseSequence and Verilog generation
  operations/  # standalone array algorithms
  subsystems/  # exp.readout and exp.timing
  views/       # neutral_atom -> frontend.plot adapters
  session.py   # NeutralAtomSession / connect
```

<!-- cell:markdown -->
## Result objects and `summary()`

每个 subsystem 调用都返回 result object，而不是只返回裸 array。result object 保留 raw data、plot handle 和一个小的 `summary()` dict。`summary()` 是给 notebook 快速查看、GUI 状态栏、JSON log 和测试断言用的轻量状态摘要；真正分析时仍然读 `result.images`、`result.counts`、`result.occupied` 或 `result.calibration`。

<!-- cell:code -->
exp = na.connect(
    "virtual",
    bright_count_rate=3000,
    loss_rate=0.1,
    sitemap={"grid_shape": (5, 7), "spacing_px": 12.0, "roi_radius": 1, "sitemap_exposure": 0.02},
)
zf.require_attrs(exp, ["camera", "readout", "timing"], name="exp")
exp

<!-- cell:code -->
isinstance(exp.camera, na.CameraDevice), isinstance(exp.readout, na.ReadoutSubsystem)

<!-- cell:markdown -->
## Configure and inspect the imaging pulse

`PulseSequence` 用物理时间描述 pulse，而不是直接手写 Verilog。frontend 的 pulse plot 用实心块显示 on 区间，并保留每个 channel 的 off baseline；x 轴会按时长自动切换到 `ns/us/ms/s`。

<!-- cell:code -->
exp.timing.configure_imaging(exposure=2e-3, load=True, trigger_width=20e-6, pre_trigger=100e-6)
pulse_plot = exp.timing.plot_sequence()
preflight = exp.timing.preflight()
preflight.summary()

<!-- cell:markdown -->
## Capture a camera image

`capture` 是 camera device 的方法，所以调用是 `exp.camera.capture()`。它永远只显示 raw camera frame，不自动叠加 sitemap 圈；site overlay 只属于 calibration/readout/detect 图。virtual camera 参考 C15550-22UP 的量级：约 200 counts offset、0.107 electrons/count、0.43 electrons RMS readout noise。

<!-- cell:code -->
capture = exp.camera.capture(display=True)
capture.summary()

<!-- cell:markdown -->
## Calibrate sitemap

`sitemap` 只回答“每个 trap site 在 camera 上在哪里”。输出包含 `centers`、`calibration`、`average_image` 和 frontend plot handle。

<!-- cell:code -->
sitemap = exp.readout.sitemap(frames=12, display=True)
sitemap.summary()

<!-- cell:markdown -->
## Calibrate thresholds

这个步骤依赖 sitemap。histogram 里的 threshold 线可拖动；右上角显示当前 threshold、左右比例、双峰 Gaussian fidelity 和模型交点 `fit cut`。

<!-- cell:code -->
threshold = exp.readout.thresholds(frames=80, site=0, display=True)
threshold.summary()

<!-- cell:code -->
threshold.plot_site(site=10, display=True)
threshold.summary()

<!-- cell:markdown -->
## Detect one shot

detect 图显示 raw camera data：所有 sitemap site 有很浅的背景圆圈，只有判断为 occupied 的 site 画较细的橙色圆圈。`DetectionResult.occupied` 是后续 rearrangement 或 statistics 可以直接使用的 boolean array。

<!-- cell:code -->
shot = exp.readout.detect(display=True)
occupancy_grid = shot.occupied.reshape(exp.devices.trap_array.grid_shape)
occupancy_grid

<!-- cell:markdown -->
## Standalone array analysis

有些算法不应该绑死在 session 上。只给 images 和 calibration，也可以重算 sitemap、threshold 或 detect。

<!-- cell:code -->
standalone_sequence = na.imaging_sequence(exposure=exp.camera.exposure, load=True, name="sitemap")
standalone_images = exp.camera.acquire(4, sequence=standalone_sequence)
standalone_sitemap = na.calibrate_sitemap_from_images(
    standalone_images,
    grid_shape=exp.devices.trap_array.grid_shape,
    display=False,
)
standalone_threshold = na.calibrate_threshold_from_images(
    exp.camera.capture(frames=12, display=False).images,
    standalone_sitemap.calibration,
    display=False,
)
standalone_shot = na.detect_image(capture.image, standalone_threshold.calibration, display=False)
standalone_shot.occupied.shape

<!-- cell:markdown -->
## Scan detection time and fidelity

`detection_time` 不使用 virtual ground truth。它先拍 long-exposure reference images，然后对每个 detection time 的 ROI count distribution 做 threshold 和 Gaussian split fidelity 估计。接口默认 `live=True`；这里保留 live scan，cell 返回后 acquisition worker 和 frontend plot 会继续更新。等图跑完或想提前停止时，运行下一格 `scan.stop()`，再在后面的 cell 里做 decay fit。

<!-- cell:code -->
times = np.linspace(0.2e-3, 10e-3, 100)
scan = exp.readout.detection_time(times, shots=30, live=True, display=True)

<!-- cell:markdown -->
## Stop the live scan

对 notebook 和未来 GUI 来说，外部只需要一个 stop：`scan.stop()`。它转发到 frontend `RunSession.stop()`，这个 session 会请求 acquisition worker/source 停止，并停止 attached plot refresh timer。已经采到的数据仍然留在 `scan.fidelities` 里，可以继续保存、显示 summary，或在下一格做 fit。

内部仍然保留 `scan.measurement`、`scan.plot`、`scan.data_figure` 这三个部件，方便 debug 或 GUI 接管；但普通实验流程不要把 stop 拆成两套 API。

<!-- cell:code -->
scan.stop()
scan.summary()

<!-- cell:markdown -->
## Fit the stopped scan

拟合前要保证 live scan 已经结束或已经运行过 `scan.stop()`。decay fit 直接使用 frontend 的 `DataFigure` fitting 栈。

<!-- cell:code -->
fit_result, popt = scan.data_figure.decay()
scan.summary(), fit_result, popt

<!-- cell:markdown -->
## Save calibration, status, and Verilog

当前 Verilog 生成的是轻量 edge-table module。它已经足够检查 timing/channel/tick，但还不是完整 address_switch register/VIO 体系。

<!-- cell:code -->
Path("results").mkdir(exist_ok=True)
Path("generated_sequences").mkdir(exist_ok=True)

calibration_path = exp.readout.save("results/neutral_atom_quickstart_calibration.json")
status_path = exp.save_status("results/neutral_atom_quickstart_status.json")
verilog_path = exp.timing.write_verilog("generated_sequences")

calibration_path, status_path, verilog_path
