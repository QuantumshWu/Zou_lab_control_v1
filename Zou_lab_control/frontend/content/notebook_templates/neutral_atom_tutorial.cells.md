<!-- cell:markdown -->
# Neutral atom quickstart

这个 notebook 展示第一版轻量中性原子实验线路：连接 device，配置 pulse sequence，拍 camera 图，校准 sitemap，校准 threshold，探测 occupancy，最后得到 detection time 和 fidelity 曲线。

> **虚拟 == 实机（核心准则）。** 下面**每一步都是真机上要跑的完整流程**：从相机图像**提取**每个 site 的位置、从数据**拟合** PSF、从计数分布**学习**每个 site 的阈值、从 reference 帧**推**保真度。**唯一虚拟的是相机数据**——它由一个实现了和真机相机相同 `CameraDevice` 契约的 `VirtualCamera` fake 出来。换到真机时**只改第一格的 `na.connect("virtual", ...)` → `na.connect("qcmos", ...)`**(连一个 JSON 设备图),下面的分析代码一行都不用动。所以在虚拟上跑通 = 在真机上跑通。

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
- `na.load_devices(...)`：按 JSON/dict 构造 device graph，合并本次运行的 device 参数，并要求每个 device 继承对应 base class；需要时也可以统一 open。
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
exp

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
## Calibrate sitemap（从图像**提取**每个 site 的位置）

`sitemap` 回答“每个 trap site 在 camera 上在哪里”。这就是实机第一步:**它没有任何 site 坐标的先验**——它拍一组全亮模板帧,在平均图上用 `core.analysis.find_site_centers`(高斯平滑 + 找局部极大 + 按 trap 网格排序)**从图像里检测**出中心。换真机时这一格不变:真机相机拍的全亮帧(满载模板)走的是同一段检测代码。输出含 `centers`、`calibration`、`average_image` 和 plot handle。

<!-- cell:code -->
sitemap = exp.readout.sitemap(frames=12, display=True)
# centers 是从图像检测出来的(不是设备给的);打印头几个确认它确实定位到了亮斑:
print("detected centers (from the image):", sitemap.calibration.centers[:3].round(1).tolist(), "...")
sitemap.summary()

<!-- cell:markdown -->
## Calibrate thresholds（从计数分布**学习**阈值）

这个步骤依赖 sitemap。它拍一批随机装载帧,用标定的方式逐 site 提取计数,然后**从这堆实验计数的分布里**定阈值(Otsu / 双高斯)——阈值不是手填的常数,而是从数据学出来的,真机同样如此。histogram 里的 threshold 线可拖动;右上角显示当前 threshold、左右比例、双峰 Gaussian fidelity 和模型交点 `fit cut`。

<!-- cell:code -->
threshold = exp.readout.thresholds(frames=80, site=0, display=True)
print("thresholds learned from the count distribution:", threshold.calibration.thresholds[:3].round(1).tolist(), "...")
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
## Rb87 读出：PSF 匹配滤波 + 双高斯（完整实机流程）

这一节就是 **Rb87 的真实读出流程**:位点密集、PSF 边缘交叠、光子数少时,方框计数会糊在一起,换成匹配滤波读出。整条链路全部**从实验数据来**(真机一模一样,只有相机是虚拟的):

1. **从全亮模板图像提取 PSF**(`method="psf"`):`core.psf.fit_site_psfs` 对每个检测到的 site 裁框、扣环形背景、拟合 2D 高斯,得到**逐站点归一化 PSF 权重**;逐发提取改成 PSF 加权点积(已知形状 + 加性噪声下信噪比最优)。权重是**拟合出来的**,不是预设的——下面 `psf_cal.psf_weights.shape` 就是它。
2. **从计数分布定阈值**(`method="bimodal"`):拟合暗/亮双高斯峰核,阈值放在两高斯总错判率最小处,并给出模型保真度。

二者都接在同一套 `TrapCalibration.detect` 契约后面(box/otsu 仍是默认)。下面用 standalone 路径演示,不改动上面 session 的 box 标定。`psf_template_images` 这里由 `exp.camera.acquire(...)` 拿到(虚拟相机产帧);真机上同一行换成真相机即可,`calibrate_sitemap_from_images`/`calibrate_threshold_from_images` 一字不改。

<!-- cell:code -->
psf_template_images = exp.camera.acquire(
    8, sequence=na.imaging_sequence(exposure=exp.camera.exposure, load=True, name="sitemap")
)
psf_sitemap = na.calibrate_sitemap_from_images(
    psf_template_images,
    grid_shape=exp.devices.trap_array.grid_shape,
    method="psf",
    display=False,
)
psf_threshold = na.calibrate_threshold_from_images(
    exp.camera.capture(frames=120, display=False).images,
    psf_sitemap.calibration,
    method="bimodal",
    display=False,
)
psf_cal = psf_threshold.calibration
print("method:", psf_cal.method, "| psf weights:", psf_cal.psf_weights.shape)
na.detect_image(exp.camera.capture(display=False).image, psf_cal, display=True).occupied.sum()

<!-- cell:markdown -->
## Per-site readout fidelity from data — Rb87 文件夹工作流(与真机完全一致)

这是 **Rb87 真机读出流程**,和 `references/rb87_readout_v16` 一样是**文件夹式**:实验把相机原始帧逐张存到**一个数据文件夹**,命名 `PREFIX<n>`;分析**指向那个文件夹**,把帧按 `shots_per_group` 一组(每次装载成像几张,同一批原子)索引,`short_shot` 是待表征的短读出帧、`ref_shots` 是高 SNR 参考帧:

1. 平均参考帧得到全亮模板 → **从模板图像检测站点** + 拟合 PSF;
2. reference 帧双高斯**严格共识**给出每个 `(group, site)` 的真值亮/暗标签;
3. 随机 **train** 划分上**逐站点**训练阈值,**held-out test** 上诚实报告保真度;
4. 全局单阈值对比 + drop-worst-site 消融;结果写到 `<data_dir>_results/`。

**唯一虚拟之处:数据文件夹由 `na.write_virtual_run` 写假帧填充**(真机上这一格删掉——你的实验/相机已经把帧写进文件夹了)。下面**分析两行真机一字不改**,只把第一格的 `na.connect("virtual",...)` 换成 `na.connect("qcmos", <设备配置>)`。

<!-- cell:code -->
# 数据文件夹(真机:相机/DAQ 把 PREFIX<n> 原始帧写到这里)。改成你的实验路径即可。
data_dir = "results/rb87_run01"
# 【仅虚拟】把假原始帧写进该文件夹,模拟真机采集(真机:删除这一格)。
# 每次装载成像 shots_per_group=4 张:short_shot=3 是短读出,ref_shots=(1,2,4) 是高 SNR 参考。
na.write_virtual_run(
    data_dir, prefix="img", groups=150, shots_per_group=4, short_shot=3, ref_shots=(1, 2, 4),
    short_exposure=3e-3, reference_exposure=20e-3,
    grid_shape=exp.devices.trap_array.grid_shape, seed=1,
)

<!-- cell:code -->
# 【真机一字不变】从文件夹读图 → 检测站点+PSF(method="psf") → 逐站点定阈值+保真度。
exp.readout.sitemap_from_dir(data_dir, prefix="img", method="psf")
report = exp.readout.characterize_from_dir(data_dir, prefix="img", train_fraction=0.9, seed=1)
report.summary()

<!-- cell:markdown -->
## The per-site readout histogram grid（通用 N，这里 35 站）

每个站点一张直方图：暗=灰、亮=蓝、橙线=该站**训练出的**阈值，标题给 held-out 保真度；按 trap 网格 `(rows, cols)` 排布。布局由 frontend 拥有——**不重叠、不裁切、对齐**，且通用任意站点数（不写死 35）。

<!-- cell:code -->
values, occupied = report.per_site_arrays()
grid = zf.site_histogram_grid(
    values, occupied=occupied, thresholds=report.thresholds,
    site_fidelities=report.site_fidelities, grid_shape=exp.devices.trap_array.grid_shape,
    labels=("PSF signal (counts)", "Shots"),
    title=f"Per-site readout histograms (held-out F={report.aggregate_fidelity:.3f})",
)

<!-- cell:markdown -->
## Global vs per-site threshold, and the drop-worst-site ablation

逐站点阈值通常优于一个全局阈值；drop-worst-site 消融显示忽略最差的 K 个站点后，held-out 保真度如何回升。

<!-- cell:code -->
print(f"global one-threshold held-out fidelity: {report.global_fidelity:.4f}")
print(f"per-site             held-out fidelity: {report.aggregate_fidelity:.4f}")
for row in report.ablation:
    print(f"  drop worst {row['drop_worst_k']}: F={row['fidelity']:.4f}  kept={row['kept_sites']}  errors={row['errors']}")

<!-- cell:markdown -->
## Scan detection time and fidelity

`detection_time` 不使用 virtual ground truth。它先拍 long-exposure reference images，然后对每个 detection time 的 ROI count distribution 做 threshold 和 Gaussian split fidelity 估计。接口默认 `live=True`；这里保留 live scan，cell 返回后 acquisition worker 和 frontend plot 会继续更新。等图跑完或想提前停止时，运行下一格 `scan.stop()`，再在后面的 cell 里做 decay fit。

<!-- cell:code -->
clock_hz = exp.devices.sequencer.clock_hz
time_ticks = np.linspace(int(round(0.2e-3 * clock_hz)), int(round(10e-3 * clock_hz)), 100, dtype=int)
times = time_ticks / clock_hz
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
## 一键测温：release-recapture（弹道重捕）

测温与上面的读出时长扫描**是同一台通用扫描引擎**的另一个实例——只换扫描轴 / 每点采帧方式 / 约简器：两次成像之间把 trap 关掉 `t_off`，原子按热速度自由飞，再开 trap，看还有多少能被重捕。越热飞得越远，存活随 `t_off` 衰减越快——**survival-vs-`t_off` 曲线编码了温度**。曲线只定 `r_c/sqrt(T)`，所以**必须**从阱几何给出捕获半径 `capture_radius`（米）才能定出 T。

下面是**实机一字不变**的配方（唯一虚拟之处仍是相机）。`build_release_recapture_pulse` 建一个 6 周期双触发序列，trap-off 周期的 duration 绑到 scan slot `s0`=`t_off`（与扫描读出时长同一种可扫量）；`exp.readout.temperature` 每点采两帧、按 `calibration.detect` 算逐点存活；`fit_temperature` 是纯后处理。

> **一键 GUI 路径**：同一测量在 Task 控制台里走 `zf.show_task_console(hub=..., measurements=exp.readout.measurement_specs())`——Control 标签页选 Temperature、填范围/shots/capture_radius、点 Start，Monitor 出 survival 曲线、跑完自动显示 T。GUI 的 Start 与下面这行 API 调**同一个建器**，不会漂移。

<!-- cell:code -->
# 6 周期双触发序列：trap_off 周期的 duration 绑到 scan slot s0(= t_off)。
rr_state = na.build_release_recapture_pulse(channels=list(exp.devices.sequencer.channels))
rr_pulse = na.bind_pulse(exp.devices.sequencer, rr_state)

# 扫 t_off(0..300us, 13 点),每点 shots=16 次装载求均存活。live=False 直接跑完。
t_off_s = np.linspace(0, 300e-6, 13)
temp_scan = exp.readout.temperature(t_off_s, pulse=rr_pulse, shots=16, live=False, display=True)
print("t_off (us):", np.round(temp_scan.x * 1e6, 1).tolist())
print("survival  :", np.round(temp_scan.y, 3).tolist())

<!-- cell:code -->
# 纯后处理拟合温度(capture_radius 必填,米;这里 ~6um 量级的子阱)。
temp_fit = na.fit_temperature(temp_scan.x, temp_scan.y, capture_radius=6e-6)
temp_fit.summary()   # {'temperature_uK': ~44..50, 'capture_radius_m': 6e-06, 'success': True}

<!-- cell:markdown -->
## Task 控制台：从零搭建实时监控（loading rate / 占据 / 站点）

task_console 的设计原则是**自由搭建**——看板开出来是空的，你从 **Add Panel** 一路自己搭。把 `session=exp` 传进去（让看板能用相机建连续生产者）+ `measurements` / `processors`，Add Panel 就分成清晰的几类：

- **Measurement: Camera (live frames)**：连续出帧的相机测量，只发一个信号 `frame`。这是整条 loading 读出链的源头——相机出帧、真流程检测，再没有别的隐藏环节。
- **Processor: Judge occupancy**（判占据，reactive）：消费 `frame`、跑**真** `calibration.detect`，流式发布 `occupied` / `counts` / `rate` / `rate_sites` / `rate_grid` / `centers` / `thresholds`。参数是**从哪载入标定**（site/PSF/阈值，留空用 `session` 当前标定）+ `source` + `ema`。
- **Task: Calibrate readout**（一次性工作流）：在**自己的线程**里跑标定、不卡界面；它**不往 hub 发任何信号**——结果落在 `task.result`，中途帧/进度写进它自己的 `TaskOutput` 缓冲。运行时它占一张**固定 Monitor 面板**看中途模板帧、并锁定其它操作只留 **Stop task**（confocal task 式）。
- 然后**自由加视图**读 measurement / processor 发的信号：Add Panel → `Plot: Site map`，在 Setting 里把 source 写 `value = occupied`（占据图：圈画在相机帧上，centers 自动取 `centers` 信号）；`Plot: 2D` + `value = frame`（原始图）；`Plot: 1D` + `value = rate_sites`（逐站点装载率）。
- **Measurement: …**（扫描）：温度 / 读出时长，默认绑曲线图。

每张面板底部都列出它**读 / 发**了哪些信号（重名会标 ⚠），所以不用猜 hub 里有什么名字（hub 里只有 measurement + processor 的输出，没有 task）。要调相机/判占据的 `grid_shape` / `exposure` / `roi_radius` / `ema`，在那个节点**自己**的 Edit 标签里改 + **Apply**（两帧之间应用，不中断采集）；plot 面板的 Edit 直接给产它信号的那个 measurement/processor 的参数表单。换实机只改 `na.connect("virtual"→"qcmos")`，这一节一字不变。

<!-- cell:code -->
%gui qt
from Zou_lab_control.neutral_atom.core.signals import SignalHub

hub = SignalHub()
# session=exp 让 Add Panel 能从相机建 "Measurement: Camera (live frames)";
# 看板开出来是空的 —— Add Panel -> Camera(发 frame) + Processor: Judge occupancy(真 detect),
# 再 Add Panel -> Plot: Site map (value = occupied) / Plot: 2D (value = frame) 自己搭。
console = zf.show_task_console(hub=hub, session=exp,
                              measurements=exp.readout.measurement_specs(),
                              processors=exp.readout.processor_specs())
console

<!-- cell:markdown -->
## Save calibration, status, and Verilog

`write_verilog` 导出的是一个轻量 edge-table 片段，便于离线检查 timing/channel/tick。真实硬件上传走的是 host 把程序打包成 BRAM image、经 JTAG-to-AXI 写进 `zlc_pulse_streamer_top` 的路径(见 FPGA manual)。

<!-- cell:code -->
Path("results").mkdir(exist_ok=True)
Path("generated_sequences").mkdir(exist_ok=True)

calibration_path = exp.readout.save("results/neutral_atom_quickstart_calibration.json")
status_path = exp.save_status("results/neutral_atom_quickstart_status.json")
verilog_path = exp.timing.write_verilog("generated_sequences")

calibration_path, status_path, verilog_path
