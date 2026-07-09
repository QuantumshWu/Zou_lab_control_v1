<!-- cell:markdown -->
# Neutral atom quickstart

这个 notebook 展示中性原子实验的核心线路：连接 device，配置 pulse sequence，拍 camera 图，校准 sitemap，校准 threshold，探测 occupancy，最后得到 detection time 和 fidelity 曲线。

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

- `na.BaseDevice` / `na.CameraDevice` / `na.SequencerDevice` / `na.TrapArrayDevice`：硬件契约。真实 camera 至少要满足 `exposure`、`configure(...)` 和纯 grabber 三原语 `arm(frames)`（返回即硬件就绪等触发）/ `read_frames(n)`（从设备自有缓冲取帧，武装期间不丢帧）/ `disarm()`；`acquire(frames)` 是三者的免触发便捷组合。相机不感知时序——arm 之后到达的触发边沿产生帧，仅此而已。
- `na.triggered_frames(camera, sequencer, sequence, frames)`：测量层**唯一**的 arm-before-fire 编排（arm 相机 → prepare+fire 序列 → 读回帧）。需要 sequencer 的是测量层，从来不是相机。
- `na.load_devices(...)`：按 JSON/dict 构造 device graph，合并本次运行的 device 参数，并要求每个 device 继承对应 base class；需要时也可以统一 open。**接自己的硬件**：写一个继承 `CameraDevice` / `SequencerDevice` / `TrapArrayDevice` 的类，`na.register_device_class(名, 类)` 注册后按短名引用（或 `load_devices(..., lookup=globals())` 零注册直接用），可选给它一个 `discover()` 让 `na.discover_devices()` 扫到它——完整示例见 hardware quickstart 与 device manual。
- `exp.camera`：默认（读出角色）camera device 本体；一键快照是会话级编排 `exp.capture()`（`exp.capture(camera="名")` 指定某台相机；非读出相机如 MOT 监视相机走它自己的 coil 模板/measurement，不是这个便捷接口）。
- `exp.device_manager()`：**设备管理 GUI（config 编辑器）**——`Config` tab 左栏按角色类型（Camera / Sequencer / Trap array / 未来的 RF …）逐台列出设备：名字、类型下拉（按角色过滤全部已注册设备类）、由设备类自声明的参数表单（`$device:` 交叉引用是下拉、容器参数是 JSON 字面量，永不 eval），可增删设备、改名、New/Load/Save 整份 config；右栏 “Scan hardware” 扫总线（发现行一键 “Add to config”）+ 已载实例列表（Snapshot / “Open devices” 初始化硬件 / 每台一个 “Control” 按钮开它的运行时控制页）。底栏 **Apply** 把编辑中的 config 应用到会话（对应 `exp.load_config(dict)`，坏 config 不伤现场；换设备时**先停引用被换设备的逻辑节点**，跑在没动设备上的节点继续）；改动未应用/未保存分别有状态点与星号提示。
- **`na.device_manager()`：还没连硬件时的 init 入口**——没有会话时直接打开编辑器，配好点 “Init devices” 即 `na.connect(该 config)`；它**返回窗口**，连出的会话经 `window.session` 交回 notebook。所以 notebook 流程是 `mgr = na.device_manager("virtual")` → 编辑 + 点 Init → `exp = mgr.session`。它是 `na.load_devices` / `na.discover_devices` 的图形面。
- **运行时控制（Control tab）**：设备管理器的 `Control` tab 是每台设备 `runtime_controls()` 声明的**运行时可调属性**目录（区别于构造期 config 参数）——相机 `exposure` 可写（经设备 setter 校验），其余（roi / sensor / 序列器 firing / scan 进度 …）是只读读回，每 200 ms 刷新。控件走与 config 表单同一个 `PARAM_WIDGETS`，永不 eval。
- `exp.device_viewer()`：**只读设备查看器**——每台设备一个 tab，显示 snapshot + 运行时读回，**无 config 编辑 / 无增删 / 无 Apply**。task console 顶栏的 “Devices” 按钮开的就是它，方便实验正跑时安全地瞄设备状态而不会误改；要真正编辑/换设备走完整的 `exp.device_manager()` / `na.device_manager()`。
- **按测量选设备**：凡用到相机的 measurement / task（*Pulse scan*、*Camera (live frames)*、*Optimize MOT field*）表单里都自动带一个 **Camera 下拉**——它的 spec 声明了 `devices=["camera"]`，基类就自动追加下拉并把**你选中**的设备注入进去（单相机用默认；双相机时在这里挑 `monitor_camera` 还是读出相机）。加一台新设备域（RF…）无需改任何 spec。读出/存活/保真类测量**故意**锁定读出科学相机（MOT 监视相机无法成像单原子）所以不给下拉。
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

<!-- cell:markdown -->
## Init devices

**默认初始化入口 = `na.device_manager("virtual")`**：打开设备管理器 GUI(config 编辑器),传 `"virtual"` **自动载入虚拟设备图**;点绿色 **Init devices** 就 `na.connect` 这份 config,连出的 session 经窗口交回 `mgr.session`。换实机时在这里把相机类型从 `virtual` 换成 `qcmos`(或 Load 一份实机 config),下面的分析代码一行都不用动。

两条**二选一**的入口(别混用):**A** GUI 交互 —— 开管理器、点 Init、`exp = mgr.session`;**B** 不开 GUI —— 等价的一行 `na.connect("virtual", ...)` 直连。下面用 A;想走 B 就把那一行换成注释里的 `na.connect(...)`。

<!-- cell:code -->
mgr = na.device_manager("virtual")

<!-- cell:code -->
# 路径 A:在上面的窗口里点过绿色 "Init devices" 后,运行本格 —— session 经窗口交回 mgr.session。
# (没点 Init 时 mgr.session 就是 None,一目了然;它不会偷偷替你连一个别的 config。)
#
# 路径 B(二选一,不开 GUI):把下一行整格换成这一行直连,显式带上本教程下游用到的 5×7 sitemap:
#   exp = na.connect("virtual", bright_count_rate=3000, loss_rate=0.1,
#                    sitemap={"grid_shape": (5, 7), "spacing_px": 12.0, "roi_radius": 1, "sitemap_exposure": 0.02})
exp = mgr.session
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

`capture` 是会话级编排（相机本身不感知会话），所以调用是 `exp.capture()`；`camera=` 可指名任意一台相机。它永远只显示 raw camera frame，不自动叠加 sitemap 圈；site overlay 只属于 calibration/readout/detect 图。virtual camera 参考 C15550-22UP 的量级：约 200 counts offset、0.107 electrons/count、0.43 electrons RMS readout noise。

<!-- cell:code -->
capture = exp.capture(display=True)
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

detect 图显示 raw camera data：所有 sitemap site 有很浅的背景圆圈，只有判断为 occupied 的 site 画较细的橙色圆圈。`DetectionResult.occupied` 是后续 statistics 可以直接使用的 boolean array。

<!-- cell:code -->
shot = exp.readout.detect(display=True)
occupancy_grid = shot.occupied.reshape(exp.devices.trap_array.grid_shape)
occupancy_grid

<!-- cell:markdown -->
## Standalone array analysis

有些算法不应该绑死在 session 上。只给 images 和 calibration，也可以重算 sitemap、threshold 或 detect。

<!-- cell:code -->
standalone_sequence = na.imaging_sequence(exposure=exp.camera.exposure, load=True, name="sitemap")
standalone_images = na.triggered_frames(
    exp.devices.camera, exp.devices.sequencer, standalone_sequence, 4)
standalone_sitemap = na.calibrate_sitemap_from_images(
    standalone_images,
    grid_shape=exp.devices.trap_array.grid_shape,
    display=False,
)
standalone_threshold = na.calibrate_threshold_from_images(
    exp.capture(frames=12, display=False).images,
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

二者都接在同一套 `TrapCalibration.detect` 契约后面(box/otsu 仍是默认)。下面用 standalone 路径演示,不改动上面 session 的 box 标定。`psf_template_images` 这里由 `na.triggered_frames(...)` 拿到(armed 相机 + fired 成像序列,虚拟相机经触发线产帧);真机上同一行换成真相机即可,`calibrate_sitemap_from_images`/`calibrate_threshold_from_images` 一字不改。

<!-- cell:code -->
psf_template_images = na.triggered_frames(
    exp.devices.camera, exp.devices.sequencer,
    na.imaging_sequence(exposure=exp.camera.exposure, load=True, name="sitemap"), 8)
psf_sitemap = na.calibrate_sitemap_from_images(
    psf_template_images,
    grid_shape=exp.devices.trap_array.grid_shape,
    method="psf",
    display=False,
)
psf_threshold = na.calibrate_threshold_from_images(
    exp.capture(frames=120, display=False).images,
    psf_sitemap.calibration,
    method="bimodal",
    display=False,
)
psf_cal = psf_threshold.calibration
print("method:", psf_cal.method, "| psf weights:", psf_cal.psf_weights.shape)
na.detect_image(exp.capture(display=False).image, psf_cal, display=True).occupied.sum()

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
# 【仅虚拟】用 *同一台* 虚拟相机(trap_array=exp.devices.trap_array)渲染假原始帧,
# 这样存盘数据与 live 会话的几何/信号完全一致(真机里本就是同一台相机在写盘)。真机:删除这一格。
# 每次装载成像 shots_per_group=4 张:short_shot=3 是短读出,ref_shots=(1,2,4) 是高 SNR 参考。
na.write_virtual_run(
    data_dir, prefix="img", groups=150, shots_per_group=4, short_shot=3, ref_shots=(1, 2, 4),
    short_exposure=3e-3, reference_exposure=20e-3,
    trap_array=exp.devices.trap_array, seed=1,
)

<!-- cell:code -->
# 【真机一字不变】从文件夹读图 → 检测站点+PSF(method="psf") → 逐站点定阈值+保真度。
exp.readout.sitemap_from_dir(data_dir, prefix="img", method="psf")
report = exp.readout.characterize_from_dir(data_dir, prefix="img", train_fraction=0.9, seed=1)
report.summary()

<!-- cell:markdown -->
## The per-site readout grid（`zf.grid(sub_plot_kind=…)`，通用 N，这里 35 站）

所有逐站点网格都走**同一个** `zf.grid(...)`，用 `sub_plot_kind` 声明每格是哪种 plot kind：`"hist"` = 每格一张读出分布直方图，`"2d"` = 每格一张图像（如 PSF 核）。这一个声明同时驱动缩略图、双击放大成该 kind 的标准单图、以及 `exp.figure_viewer()` 里 Setting 显示该 kind 的参数。每格：暗=灰、亮=蓝、橙线=该站**训练出的**阈值，**站号（+保真度）画在每格小标题里、不占图内**；按 trap 网格 `(rows, cols)` 排布，布局由 frontend 拥有（**不重叠、不裁切、对齐**，通用任意站点数）。`zf.site_histogram_grid` / `zf.site_psf_grid` 只是 `sub_plot_kind="hist"` / `"2d"` 的薄壳预设。

<!-- cell:code -->
values, occupied = report.per_site_arrays()
hist_grid = zf.grid(
    values, sub_plot_kind="hist",
    occupied=occupied, thresholds=report.thresholds,
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

`detection_time` 不使用 virtual ground truth：它先拍 long-exposure reference 帧，再对每个 detection time 的 ROI 计数分布做阈值 + 双高斯保真度估计。下面用 `live=False` **一次跑完整条扫描**（确定、可复现），结果里 `scan.times` / `scan.fidelities` 是曲线，`summary()['best']` 给出保真度最高的读出时长。

> 想交互观察：`detection_time(..., live=True)` 会后台采集并实时刷新前端图，`scan.stop()` 提前停止；停止后 `scan.fidelities` 里已采到的数据仍然可用。

<!-- cell:code -->
clock_hz = exp.devices.sequencer.clock_hz
time_ticks = np.linspace(int(round(0.2e-3 * clock_hz)), int(round(10e-3 * clock_hz)), 30, dtype=int)
times = time_ticks / clock_hz
# live=False：一次跑完整条扫描，确定且可复现（live=True 则后台跑、前端实时刷新、scan.stop() 提前停）。
scan = exp.readout.detection_time(times, shots=20, live=False, display=True)
print("times (ms):", np.round(scan.times * 1e3, 2).tolist())
print("fidelity  :", np.round(scan.fidelities, 3).tolist())

<!-- cell:markdown -->
## 读出时长的选择

保真度随读出时长先升后饱和：太短信噪比不够，太长则散射加热、原子丢失、占空比下降。`summary()['best']` 直接给出曲线上保真度最高的那个读出时长——这就是要选的正式成像时间。

<!-- cell:code -->
scan.summary()   # {'best': {'time': ..., 'fidelity': ...}, 'finished': True, ...}

<!-- cell:markdown -->
## 取最优读出时长

`scan.summary()['best']` = `{'time': 秒, 'fidelity': 保真度}`。取拐点附近（略偏右留余量）作为正式成像时间；比它更长只是浪费占空比、徒增加热。

<!-- cell:code -->
best = scan.summary()["best"]
print(f"建议读出时长 = {best['time'] * 1e3:.2f} ms，保真度 ≈ {best['fidelity']:.4f}")
best

<!-- cell:markdown -->
## 一键测温：release-recapture（弹道重捕）

测温与上面的读出时长扫描**是同一台通用扫描引擎**的另一个实例——只换扫描轴 / 每点采帧方式 / 约简器：两次成像之间把 trap 关掉 `t_off`，原子按热速度自由飞，再开 trap，看还有多少能被重捕。越热飞得越远，存活随 `t_off` 衰减越快——**survival-vs-`t_off` 曲线编码了温度**。曲线只定 `r_c/sqrt(T)`，所以**必须**从阱几何给出捕获半径 `capture_radius`（米）才能定出 T。

下面的配方在实机上**基本一字不变**（唯一虚拟之处仍是相机）——只是 `build_release_recapture_pulse` 默认按 `trap`/`probe`/`emCCD` 三个**角色通道名**建序列;实机通道名不同就把角色名作参数传进去（`build_release_recapture_pulse(channels=..., trap_channel="chNN", probe_channel=..., trigger_channel=...)`）。它建一个 6 周期双触发序列，trap-off 周期的 duration 绑到 scan slot `s0`=`t_off`（与扫描读出时长同一种可扫量）；`exp.readout.temperature` 每点采两帧、按 `calibration.detect` 算逐点存活；`fit_temperature` 是纯后处理。

> **一键 GUI 路径**：同一测量在 Task 控制台里走 `zf.show_task_console(hub=SignalHub(), session=exp, measurements=exp.readout.measurement_specs())`——头部 **Add Panel** 选 `Temperature`，它作为一个 Logic 节点加入，在自己的 **Edit** 页填范围 `t_off`/`shots`/`per_site`、点 **Start**，再加一个 Monitor 面板指向 survival 信号看曲线。`capture_radius` **不在采集表单里**——它是把 survival 曲线变成温度的**后处理 fit 入参**（一个已知的阱几何，不改变发什么/读什么），所以拿到 survival 后用下面的 `fit_temperature(..., capture_radius=...)` 定 T。GUI 的 Start 与下面这行 API 调**同一个建器**，不会漂移。完整的控制台跑实验流程见 `task_console_tutorial.ipynb`。

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
- **Processor: Judge occupancy**（判占据，reactive）：消费 `frame`、跑**真** `calibration.detect`，逐 repeat 切片判，流式发布 `occupied` / `counts`（`(repeat, n_sites)` 块，`repeat_mode=average` 即逐站装载概率）/ `rate`（本块装载率标量）/ `centers` / `thresholds` / `frame_judged`。参数是**从哪载入标定**（`calibration`：site/PSF/阈值，默认就是 Calibrate 任务写的 `calibrations/calibration.json`，该文件出现前回退到 `session` 当前标定）+ `source`（要判的 `frame` 信号）+ `method`（box / per-site PSF / uniform PSF）。
- **Task: Calibrate readout**（一次性工作流）：在**自己的线程**里跑标定、不卡界面；它**不往 hub 发任何信号**——结果落在 `task.result`，中途帧/进度写进它自己的 `TaskOutput` 缓冲。运行时它占一张**固定 Monitor 面板**看中途模板帧、并锁定其它操作只留 **Stop task**（confocal task 式）。
- 然后**自由加视图**读 measurement / processor 发的信号：Add Panel → `Plot: Site map`，在 Setting 里把 source 写 `value = occupied`（占据图：圈画在相机帧上，centers 自动取 `centers` 信号）；`Plot: 2D` + `value = frame_0`（原始图）；`Plot: Site map` + `value = occupied` 设 `repeat_mode=average`（逐站点装载概率）。
- **Measurement: …**（扫描）：温度 / 读出时长，默认绑曲线图。

每张面板底部都列出它**读 / 发**了哪些信号（重名会标 ⚠），所以不用猜 hub 里有什么名字（hub 里只有 measurement + processor 的输出，没有 task）。要调相机/判占据的 `grid_shape` / `exposure` / `roi_radius` / `method`，在那个节点**自己**的 Edit 标签里改 + **Apply**（两帧之间应用，不中断采集）；plot 面板的 Edit 直接给产它信号的那个 measurement/processor 的参数表单。换实机只改 `na.connect("virtual"→"qcmos")`，这一节一字不变。

<!-- cell:code -->
# 一键打开 Task 控制台:`exp.task_console()` 自动把本 session 的 hub + 自动发现的
# measurement / processor / task 目录都接好(等价于手写 zf.show_task_console(hub=..., session=exp,
# measurements=..., processors=...),但一行搞定)。看板开出来是空的,从 Add Panel 自己搭
# Camera(发 frame) -> Processor: Judge occupancy(真 detect) -> Plot: Site map (value = occupied) /
# Plot: 2D (value = frame_0)。`exp.task_console(task="<名字>")` 还能直接载入 tasks/<名字>.json 的存盘布局。
# 注意:Qt 实时窗口与本 notebook 上面的 ipympl 内联图是两套事件循环——建议重启 kernel 后,只跑
# 上面那格 `exp = na.connect("virtual", ...)` + 本格(完整的从零搭建流程见 task_console_tutorial.ipynb)。
console = exp.task_console()
console

<!-- cell:markdown -->
## 一键打开 pulse GUI(编辑 / 扫描脉冲)

`exp.pulse_gui()` 打开绑定到本 session 的脉冲编辑器(等价于不带 session 的 `zf.show_pulse_gui()`,但绑了 session 后测量端能读回编辑后的程序)。在 period 卡里改 duration / DAC / delay;给任意 duration / DAC / delay 字段点那个圆点,绑成 **API slot**(`aN`,紫色,按名设值——普通通道 delay 和 **DAC-bus delay 一样能绑**)或 **scan slot**(`sN`,可流式扫描);**Scan** 页给所有 scan slot 写一张 `N×n_slots` 的 `scan_table`(delay 只有 API slot、没有 scan slot)。换实机只改 `na.connect`,这一格不变。

<!-- cell:code -->
# 一键打开脉冲编辑器(绑定本 session,所以 exp.readout.* 测量能读回编辑后的时序):
pulse_win = exp.pulse_gui()
pulse_win

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
