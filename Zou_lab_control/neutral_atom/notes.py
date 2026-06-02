"""Chinese PDF tutorial for the lightweight neutral-atom session."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from Zou_lab_control.frontend.notes import NotesBuildResult, render_notes_pdf

from .content.manuals import generate_hardware_quickstart_figures, hardware_quickstart_body


def _save_plot(plot_obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_obj.fig.savefig(path, bbox_inches="tight")
    return path


def _generate_figures(asset_dir: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    import Zou_lab_control.neutral_atom as na

    asset_dir.mkdir(parents=True, exist_ok=True)
    exp = na.connect("virtual")
    exp.timing.configure_imaging(exposure=2e-3, load=True)

    pulse = exp.timing.plot_sequence(display=False)
    pulse_path = _save_plot(pulse, asset_dir / "pulse_sequence.png")
    plt.close(pulse.fig)

    capture = exp.camera.capture(display=False)
    capture_path = _save_plot(capture.plot, asset_dir / "capture.png")
    plt.close(capture.plot.fig)

    sitemap = exp.readout.sitemap(frames=12, display=False)
    sitemap_path = _save_plot(sitemap.plot, asset_dir / "sitemap.png")
    plt.close(sitemap.plot.fig)

    threshold = exp.readout.thresholds(frames=80, site=0, display=False)
    threshold_path = _save_plot(threshold.plot, asset_dir / "threshold.png")
    plt.close(threshold.plot.fig)

    shot = exp.readout.detect(display=False)
    detect_path = _save_plot(shot.plot, asset_dir / "detect.png")
    plt.close(shot.plot.fig)

    times = np.array([2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3])
    scan = exp.readout.detection_time(times, shots=30, live=False, display=False)
    scan_path = _save_plot(scan.plot, asset_dir / "detection_scan.png")
    plt.close(scan.plot.fig)

    return {
        "pulse": pulse_path,
        "capture": capture_path,
        "sitemap": sitemap_path,
        "threshold": threshold_path,
        "detect": detect_path,
        "scan": scan_path,
    }


def _manual_body(figures: dict[str, Path]) -> str:
    figs = {key: (Path("assets") / value.name).as_posix() for key, value in figures.items()}
    return rf"""
\chapter{{这版要解决什么}}
\section{{先跑通一条真实实验会用的线}}
当前 \pyapi{{Zou_lab_control.neutral_atom}} 只承诺一条最小闭环：

\begin{{codeblock}}[Experiment loop]
connect devices
  -> configure pulse sequence
  -> capture camera image
  -> calibrate sitemap
  -> calibrate threshold
  -> detect occupancy
  -> scan detection time vs fidelity
\end{{codeblock}}

这不是把 Jupyter 当命令行。session 仍然是实验配置的容器，device、timing、analysis、plot、result object 的边界先放对。之后要加真实 qCMOS、FPGA、AOD rearrangement 或 GUI panel，都应该沿着这些边界补实现，而不是把 camera handle、Matplotlib artist、threshold array、Verilog state 又混进一个 cell。

\section{{核心原则}}
\begin{{itemize}}
  \item device 只负责硬件动作，例如 camera acquire、sequencer prepare/fire。
  \item \pyapi{{operations}} 函数只吃 array 和 calibration，可以 standalone 离线运行。
  \item subsystem 负责把当前 \pyapi{{exp}} 的 device、timing、defaults、calibration 串成实验动作。
  \item frontend 只画图和刷新共享 array，不知道硬件细节。
  \item result object 保存 raw data、summary 和 plot handle，方便 notebook 交互与复盘。
\end{{itemize}}

\begin{{notebox}}[为什么不把所有函数都放在 exp 上]
\pyapi{{exp.capture()}} 很短，但长期看会让 session 变成上帝对象。现在推荐 \pyapi{{exp.camera.capture()}} 和 \pyapi{{exp.readout.sitemap()/thresholds()/detect()/detection_time()}}。camera 是 device 本体；readout 是一个有机 subsystem，因为 sitemap、threshold、detect 和 readout fidelity calibration 都共享同一份 \pyapi{{TrapCalibration}} 状态和同一套 camera-readout 假设。
\end{{notebox}}

\chapter{{模块和对象关系}}
\section{{新的 package tree}}
现在 \pyapi{{neutral_atom}} 根目录只保留 session、教程生成器、公开 re-export 和配置文件；真正的实现按架构层放进子包。这个布局的目的不是“看起来整齐”，而是让以后加硬件、算法、GUI 时能一眼看出应该放在哪里。

\begin{{codeblock}}[Package tree]
Zou_lab_control/neutral_atom/
  __init__.py              # public API: import Zou_lab_control.neutral_atom as na
  session.py               # NeutralAtomSession / connect
  notes.py                 # this manual generator
  configs/                 # virtual / real device graph examples

  core/
    analysis.py            # pure image/statistics helpers
    calibration.py         # TrapCalibration data record
    results.py             # ResultObject / CaptureResult / DetectionResult / ...
    utils.py               # small serialization/index helpers

  devices/
    base.py                # BaseDevice / CameraDevice / SequencerDevice / TrapArrayDevice
    registry.py            # load_devices and config dependency resolution
    virtual.py             # VirtualTrapArray / VirtualCamera / VirtualSequencer
    qcmos.py               # real qCMOS adapter boundary
    sequencer.py           # VerilogSequencer adapter boundary

  timing/
    sequence.py            # Pulse / PulseSequence / imaging_sequence
    verilog.py             # edge table -> Verilog bundle

  operations/
    calibration.py         # standalone sitemap / threshold algorithms
    detection.py           # standalone image -> DetectionResult

  subsystems/
    readout.py             # exp.readout: sitemap / threshold / detect / fidelity scan
    timing.py              # exp.timing: configure / preflight / write_verilog

  views/
    plots.py               # neutral atom objects -> frontend.plot adapters
\end{{codeblock}}

\section{{依赖方向}}
最重要的规则是依赖只能往下走，不能绕回去。这样 notebook、GUI 和真实硬件都可以共享同一套底层对象。

\begin{{codeblock}}[Dependency direction]
notebook / GUI
    |
    v
NeutralAtomSession
    |
    +--> devices/*        # hardware actions only
    +--> subsystems/*     # experiment-level organic actions
    +--> timing/*         # pulse and Verilog timing source of truth
    |
    v
operations/*              # standalone array algorithms
    |
    v
core/*                    # data records, result objects, analysis helpers
    |
    v
views/* -> frontend       # plotting adapters only
\end{{codeblock}}

\pyapi{{devices}} 不应该 import \pyapi{{subsystems}}，因为真实 hardware adapter 不应该知道实验动作。 \pyapi{{operations}} 不应该 import \pyapi{{session}}，因为离线重分析不应该要求连接设备。 \pyapi{{views}} 只把 neutral-atom 数据转换成 \pyapi{{frontend.plot}} 的输入，不应该控制 camera 或 timing。这个方向约束比文件名更重要：它防止以后重新长出一个什么都知道的“上帝对象”。

\section{{为什么 camera 是 device}}
\pyapi{{capture}} 的物理语义是“camera 采图”。所以它属于 \pyapi{{CameraDevice}}：真实 qCMOS 要 arm buffer、等 trigger、读 frame；virtual camera 要根据 trap array 渲染图像；两者都返回 image list。它们的共同 contract 是：
\begin{{codeblock}}[Python]
class CameraDevice(BaseDevice):
    @property
    def exposure(self) -> float: ...

    def configure(self, *, exposure=None, **kwargs) -> None: ...

    def acquire(self, frames=1, *, sequence=None, sequencer=None, **kwargs) -> list[np.ndarray]: ...

    def capture(self, *, frames=1, exposure=None, sequence=None, display=True, **kwargs) -> CaptureResult:
        ...
\end{{codeblock}}

\pyapi{{capture}} 比 \pyapi{{acquire}} 多做两件 notebook 友好的事：把最后一张 raw frame 交给 frontend 画出来，并把 raw images、sequence、plot handle 包成 \pyapi{{CaptureResult}}。它永远不画 sitemap/site 圈，即使当前 session 已经做过 calibration。site overlay 只属于 \pyapi{{detect}} 或显式的 calibration/readout 图；\pyapi{{capture}} 不应该知道 site centers、threshold 或 virtual simulator state。

\section{{为什么 readout 是一个 subsystem}}
sitemap、threshold、detect、detection-time fidelity calibration 不是四个彼此无关的独立动作。它们共同依赖 camera readout 模型和同一份 \pyapi{{TrapCalibration}}：
\begin{{itemize}}
  \item \pyapi{{sitemap}} 建立 site centers、grid ordering、ROI radius。
  \item \pyapi{{thresholds}} 在这些 centers 上建立 per-site thresholds。
  \item \pyapi{{detect}} 消费 centers 和 thresholds，输出 occupancy。
  \item \pyapi{{detection_time}} 扫 exposure/readout time，仍然要在同一套 centers/ROI/count distribution 上评价 fidelity。
\end{{itemize}}

所以这些动作被放进一个 \pyapi{{ReadoutSubsystem}}，入口是：
\begin{{codeblock}}[Python]
exp.readout.sitemap(...)
exp.readout.thresholds(...)
exp.readout.detect(...)
exp.readout.detection_time(...)
exp.readout.save(...)
exp.readout.load(...)
exp.readout.clear()
\end{{codeblock}}

这个 subsystem 是“实验层调度器”，不是算法仓库。它调用 \pyapi{{operations}} 里的 standalone 函数，把当前 session 的 camera、sequence、defaults、calibration 和 history 串起来。这样在线实验和离线 array 重分析共用同一份算法。

\section{{session 的职责边界}}
\pyapi{{NeutralAtomSession}} 是实验配置容器，不是所有动作的集合。它只持有：
\begin{{itemize}}
  \item \pyapi{{devices}}：实际 device graph。
  \item \pyapi{{sequence}}：当前 pulse sequence。
  \item \pyapi{{readout}}、\pyapi{{timing}}：有机 subsystem。
  \item \pyapi{{_calibration}}：当前 readout state。
  \item \pyapi{{history}}：返回过的 result objects。
\end{{itemize}}

用户层保持短调用，但语义不混乱：
\begin{{codeblock}}[Python]
exp = na.connect("virtual")
capture = exp.camera.capture()
sitemap = exp.readout.sitemap()
threshold = exp.readout.thresholds()
shot = exp.readout.detect()
scan = exp.readout.detection_time(times)
\end{{codeblock}}

注意这里没有把 capture、calibration、detect、characterization 都挂成 session 顶层属性。那样要么让 session 变胖，要么把同一个 readout state 拆成几个假装独立的对象。

\section{{文件边界}}
\begin{{description}}
  \item[\filepath{{core/analysis.py}}] ROI counts、site finding、threshold、fidelity estimate。这些函数不认识 session，也不碰硬件。
  \item[\filepath{{core/calibration.py}}] \pyapi{{TrapCalibration}} 数据结构与保存读取。
  \item[\filepath{{core/results.py}}] result object、task stop 契约和 notebook HTML summary。
  \item[\filepath{{devices/base.py}}] \pyapi{{BaseDevice}}、\pyapi{{CameraDevice}}、\pyapi{{SequencerDevice}}、\pyapi{{TrapArrayDevice}}。这里定义换硬件时必须继承的接口。
  \item[\filepath{{devices/registry.py}}] 把 JSON/dict 配置解析成 \pyapi{{DeviceSet}}，支持 \pyapi{{"$device:name"}} 依赖引用，并要求每个 device 继承对应 base class。
  \item[\filepath{{devices/virtual.py}}] 离线 virtual trap array、camera、sequencer。
  \item[\filepath{{devices/qcmos.py}}] 真实 qCMOS camera adapter 边界。
  \item[\filepath{{devices/sequencer.py}}] FPGA/Verilog sequencer adapter 边界。
  \item[\filepath{{timing/sequence.py}}] \pyapi{{PulseSequence}}、pulse validation、pulse plot。
  \item[\filepath{{timing/verilog.py}}] pulse edge table 到 Verilog bundle。
  \item[\filepath{{operations/calibration.py}}] standalone sitemap 和 threshold calibration，从 image stack 到 \pyapi{{SitemapResult}} / \pyapi{{ThresholdResult}}。
  \item[\filepath{{operations/detection.py}}] standalone 单张 image detect，从 image + calibration 到 \pyapi{{DetectionResult}}。
  \item[\filepath{{views/plots.py}}] neutral atom data 到 \pyapi{{frontend.plot}} 的适配层。
  \item[\filepath{{subsystems/base.py}}] \pyapi{{ExperimentSubsystem}}，所有 exp-bound subsystem 的共同基类。
  \item[\filepath{{subsystems/readout.py}}] \pyapi{{ReadoutSubsystem}}，一个整体管理 sitemap、threshold、detect 和 readout-fidelity calibration。
  \item[\filepath{{subsystems/timing.py}}] \pyapi{{TimingSubsystem}}，管理当前 sequence、preflight 和 Verilog。
  \item[\filepath{{session.py}}] \pyapi{{NeutralAtomSession}} 和 \pyapi{{connect}}；它绑定 device/subsystem，不承载 standalone 算法和 result class。
\end{{description}}

\section{{device contract}}
真实硬件接入不是“刚好写一个同名方法”。camera adapter 应该继承 \pyapi{{CameraDevice}}，至少满足：
\begin{{codeblock}}[Python]
class MyCamera(na.CameraDevice):
    @property
    def exposure(self) -> float: ...

    def configure(self, *, exposure=None, **kwargs) -> None: ...

    def acquire(self, frames=1, *, sequence=None, sequencer=None, **kwargs) -> list[np.ndarray]: ...
\end{{codeblock}}

\pyapi{{acquire}} 的职责是 arm camera、让 sequencer prepare/fire（如果需要）、读出 frame，并返回 numpy image list。它不应该做 site threshold、fidelity 或保存图；这些属于 \pyapi{{operations}} / result object。sequencer adapter 对应 \pyapi{{SequencerDevice}}，必须提供 \pyapi{{prepare(sequence)}} 和 \pyapi{{fire(sequence)}}。如果一个 camera 类没有实现这些 abstract method，Python 会禁止实例化；如果 JSON 配置里的 \pyapi{{camera}} 没有继承 \pyapi{{CameraDevice}}，\pyapi{{load_devices}} 会直接报错。

\section{{session 里面有什么}}
\begin{{codeblock}}[Python]
exp = na.connect("virtual")

exp.devices      # raw DeviceSet
exp.camera       # actual CameraDevice, e.g. VirtualCamera or QCMOSCamera
exp.readout      # sitemap / threshold / detect / readout-fidelity subsystem
exp.timing       # pulse / preflight / verilog subsystem
exp.readout.current  # current TrapCalibration or None
exp.history      # returned result objects
\end{{codeblock}}

\section{{standalone 算法入口}}
并不是所有事情都需要 session。三类函数可以直接处理 array：
\begin{{codeblock}}[Python]
sitemap = na.calibrate_sitemap_from_images(images, grid_shape=(5, 7))
threshold = na.calibrate_threshold_from_images(images, sitemap.calibration)
shot = na.detect_image(image, threshold.calibration)
\end{{codeblock}}

这条 standalone 路径对真实实验很关键。很多时候你已经有 raw image stack，只想重调 ROI radius、ordering 或 threshold；这时不应该重新连接 camera，也不应该让分析函数依赖 GUI 状态。

\section{{Result object 和 summary()}}
subsystem/device 调用的返回值统一叫 result object，例如 \pyapi{{CaptureResult}}、\pyapi{{SitemapResult}}、\pyapi{{ThresholdResult}}、\pyapi{{DetectionResult}}、\pyapi{{DetectionTimeScanResult}} 和 \pyapi{{PreflightReport}}。它们不是只为打印好看而存在，而是 notebook、GUI 和自动日志之间的稳定边界。

每个 result object 都保留三类信息：
\begin{{itemize}}
  \item raw 或近 raw 数据，例如 \pyapi{{CaptureResult.images}}、\pyapi{{SitemapResult.average_image}}、\pyapi{{ThresholdResult.counts}}、\pyapi{{DetectionResult.image}}、\pyapi{{DetectionResult.occupied}}。
  \item 当前 plot handle，例如 \pyapi{{result.plot}}，用于 notebook 里继续调 axis、保存图，或让 GUI panel 接管显示。
  \item 一个小而稳定的 \pyapi{{summary()}} dict，用于快速查看、保存状态、测试断言和未来 GUI 的状态栏。
\end{{itemize}}

\pyapi{{summary()}} 的设计原则是“只放轻量、JSON-friendly、不会巨大展开的关键信息”。它不替代 raw data：真正要做分析时应该读 \pyapi{{result.images}}、\pyapi{{result.counts}}、\pyapi{{result.occupied}} 或 \pyapi{{result.calibration}}。Jupyter 里直接显示 result object 时，\pyapi{{_repr_html_}} 也是调用 \pyapi{{summary()}} 生成一小块 HTML，所以 notebook 不会因为一个 image stack 或 counts matrix 被整个刷出来。

\begin{{codeblock}}[Python]
capture = exp.camera.capture()
capture.summary()
# {{'frames': 1, 'image_shape': [96, 128], 'sequence': 'imaging', ...}}

shot = exp.readout.detect()
shot.summary()
# {{'loaded_atoms': 18, 'occupied_indices': [0, 3, 4, ...]}}
\end{{codeblock}}

长期架构里，\pyapi{{summary()}} 也是 GUI 和实验调度可以依赖的最小状态接口：按钮回调拿到 result 后可以立刻更新状态栏、写一行 JSON log、或者在失败时给出可读诊断，而不会把大型 numpy array 或 Matplotlib artist 混进控制层。

\begin{{notebox}}[scan-like task 的统一契约]
以后所有 scan / measurement / async task result 都应该继承同一类契约，而不是各自临时发明接口。公共结构是：
\begin{{itemize}}
  \item \pyapi{{result.stop()}}：task 对外唯一 stop，停止 measurement worker/source 和 attached frontend refresh。
  \item \pyapi{{result.measurement}}：采集/执行任务本身，拥有底层 lifecycle 和状态，例如 \pyapi{{points\_done}}、\pyapi{{running}}、\pyapi{{done}}。
  \item \pyapi{{result.plot}}：frontend plot 对象，只负责显示和交互。
  \item \pyapi{{result.data\_figure}}：frontend 后处理对象，负责 fitting、unit conversion、save 等。
  \item result 顶层只保存实验语义数据，例如 \pyapi{{times}}、\pyapi{{fidelities}}、\pyapi{{thresholds}}、\pyapi{{calibration}}；不重复包装 \pyapi{{DataFigure.decay()}} 这类已经存在的方法。
\end{{itemize}}
这样 GUI 以后对外只需要把 Stop 按钮连到 \pyapi{{result.stop()}}；需要 debug 或高级接管时，仍然能看到 \pyapi{{measurement}}、\pyapi{{plot}} 和 \pyapi{{data\_figure}} 三个部件。
\end{{notebox}}

\chapter{{Frontend 更新}}
\section{{统一 plot 契约}}
frontend 的公开契约仍然是 \pyapi{{data_x: (N, coord_dim)}} 和 \pyapi{{data_y: (N, channel_dim)}}。已有数组和 live plot 本质上是同一个 plot：静态时直接画当前 array；live 时 acquisition worker 写共享 array，frontend timer 只刷新 artist。

\section{{标题}}
所有 plot 现在都接受 \pyapi{{title=...}}。标题字号等于 axis label 字号，不会变成 notebook 里的大标题；有标题时默认增加 top margin，避免被图边界截断。

\section{{Pulse sequence 图}}
新增 \pyapi{{zf.plot(sequence, kind="pulse")}}。它用实心低饱和色块表示 TTL pulse，y 轴 label 与该 channel 色块同色，并缩小 tick label 来容纳至少 10 行 channel。x 轴按总时长自动选择 \pyapi{{ns/us/ms/s}}，例如微秒级 sequence 会显示 \pyapi{{Time (us)}}，避免时序图上全是很长的科学计数法秒数。off baseline 和 on block 使用完全相同的颜色和不透明度，只是几何形状不同：baseline 是同色细线，on interval 是从 baseline 向上长出的实心块。这样 pulse 看起来是一条同色信号轨迹，而不是两套叠加图层。pulse plot 用来检查时间结构，不在色块内部强行塞文字，避免短 pulse 上 label 被截断。

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.84\textwidth]{{{figs["pulse"]}}}
  \caption{{当前 imaging sequence 的 pulse plot。}}
\end{{figure}}

\section{{2D 图保持 square 公开接口}}
\pyapi{{frontend.plot(..., kind="2d")}} 对外始终使用 square view。内部 \pyapi{{Live2DDis}} 仍保留非 square 能力给开发调试，但 public API 不允许 \pyapi{{square=False}}。这样 colormap 主图、distribution axis 和 colorbar 的视觉对齐更稳定。

\chapter{{Device 配置与 virtual camera}}
\section{{最小 device graph}}
\begin{{codeblock}}[JSON]
{{
  "trap_array": {{"type": "VirtualTrapArray"}},
  "camera": {{
    "type": "VirtualCamera",
    "params": {{"trap_array": "$device:trap_array"}}
  }},
  "sequencer": {{"type": "VirtualSequencer"}}
}}
\end{{codeblock}}

\pyapi{{load_devices}} 会按依赖关系构造 device 并校验 base-class contract。真实硬件配置只需要把 camera/sequencer class 换掉；只要新 class 继承并实现 \pyapi{{CameraDevice}} / \pyapi{{SequencerDevice}}，subsystem 层仍然调用 \pyapi{{camera.acquire(...)}} 和 \pyapi{{sequencer.fire(...)}}。

\section{{qCMOS-like simulation}}
virtual camera 不再用任意 counts 噪声。它参考 C15550-22UP instruction manual 中的典型量级：
\begin{{itemize}}
  \item dark/baseline offset 约 \pyapi{{200 counts}}。
  \item conversion factor \pyapi{{0.107 electrons/count}}。
  \item standard scan readout noise \pyapi{{0.43 electrons RMS}}。
  \item ultra quiet scan readout noise 手册给出 \pyapi{{0.30 electrons RMS}}，当前默认先模拟 standard scan。
  \item -35 °C dark current 量级 \pyapi{{0.006 electrons/pixel/s}}。
\end{{itemize}}

模拟流程是：先在电子数空间生成 background、dark current 和 Gaussian atom spot；对期望电子数做 Poisson shot noise；再用 conversion factor 转回 counts，加 baseline offset 和 readout noise。这样短 exposure 的分布会自然变差，长 exposure 的分布会变好；readout calibration 只能从这些模拟相机图像里估计结果，不能从 simulator 的内部状态直接取答案。

\begin{{notebox}}[为什么默认不让 lifetime 影响第一版 scan]
真实实验里 detection time 过长确实可能造成 atom loss。但第一版要先校准 camera SNR 对 fidelity 的影响，所以 virtual 默认 detection lifetime 设为很长，避免 fidelity curve 被模拟寿命伪影拖成“时间越长越差”。以后可以把 loss model 做成 scan strategy 的显式参数。
\end{{notebox}}

\chapter{{Timing、preflight 和 Verilog}}
\section{{PulseSequence 的作用}}
\pyapi{{PulseSequence}} 保存物理时间：
\begin{{codeblock}}[Python]
seq = na.imaging_sequence(exposure=2e-3, load=True)
seq = seq.delay("qcm_trigger", 4e-9)
\end{{codeblock}}

默认 imaging sequence 包含 \pyapi{{trap}}、可选 \pyapi{{cooling}}、\pyapi{{probe}} 和 \pyapi{{qcm_trigger}}。sequence 本身不接触 camera，也不直接写 Verilog；它只是可验证的 timing source of truth。

\section{{preflight 检查}}
\pyapi{{exp.timing.preflight()}} 检查：
\begin{{itemize}}
  \item pulse 是否有负时间或非正 duration。
  \item 同一 channel 是否存在 overlap。
  \item pulse 是否短到低于一个 FPGA clock tick。
  \item sequence channel 是否在 sequencer channel list 中。
  \item 可选生成 Verilog edge table 摘要。
\end{{itemize}}

\section{{Verilog 生成}}
\pyapi{{generate_verilog}} 把 pulse 编译成绝对 tick 和 channel mask。生成 module 有 \pyapi{{clk/reset/start/running/done}} 和各 channel 输出。当前没有复刻完整 address-switch register/VIO 体系，这是有意收缩：第一版先把 pulse timing 从手写 cumulative if 变成 Python 可检查的 edge table。

\chapter{{Camera capture}}
\section{{推荐调用}}
\begin{{codeblock}}[Python]
capture = exp.camera.capture()
\end{{codeblock}}

这一步调用当前 \pyapi{{exp.sequence}}。对 virtual camera，它直接渲染模拟图；对真实 camera，应该由 \pyapi{{QCMOSCamera.acquire}} 完成 arm/trigger/readout/release。camera device 只负责采图，不负责 threshold 或 detect。

\begin{{notebox}}[capture 必须是 raw image]
\pyapi{{exp.camera.capture()}} 的图里不允许自动叠加 sitemap 圈。哪怕当前 session 已经有 \pyapi{{TrapCalibration}}，capture 仍然只显示相机 raw data。需要检查 site map 对齐或 occupied atoms 时，使用 \pyapi{{exp.readout.sitemap()}}、\pyapi{{exp.readout.detect()}} 或 \pyapi{{DetectionResult.plot\_occupancy()}}。
\end{{notebox}}

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.72\textwidth]{{{figs["capture"]}}}
  \caption{{一次 virtual qCMOS capture。}}
\end{{figure}}

\section{{CaptureResult}}
\pyapi{{CaptureResult}} 保存 \pyapi{{images}}、\pyapi{{sequence}} 和 \pyapi{{plot}}。如果以后要保存 raw image stack，应在 result object 上加 \pyapi{{save_npz}}，不要让 camera 在 acquire 内部偷偷写文件。

\chapter{{Sitemap calibration}}
\section{{推荐调用}}
\begin{{codeblock}}[Python]
sitemap = exp.readout.sitemap(frames=12, roi_radius=1)
\end{{codeblock}}

sitemap calibration 拍多帧 sitemap 图，平均后找 bright local maxima，再按 ordering 排成稳定 site index。它只回答“site 在哪里”，不回答“occupied threshold 是多少”。

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.76\textwidth]{{{figs["sitemap"]}}}
  \caption{{平均 sitemap 图和自动找到的 site centers。}}
\end{{figure}}

\section{{常见错误}}
\begin{{itemize}}
  \item center 整体偏移：优先检查 camera ROI、image orientation、ordering。
  \item 少找 site：增加 sitemap frames，降低 threshold 或检查是否所有 site 被点亮。
  \item site index 顺序不对：改 \pyapi{{ordering}}，不要在 detect 里临时重排。
\end{{itemize}}

\chapter{{Threshold calibration}}
\section{{推荐调用}}
\begin{{codeblock}}[Python]
threshold = exp.readout.thresholds(frames=80, site=0)
\end{{codeblock}}

threshold calibration 在已知 centers 上重复拍图，得到 \pyapi{{counts: shots x sites}}。每个 site 用 Otsu threshold 做初始阈值，显示时用 frontend histogram。threshold 线可拖动，右上角显示当前 threshold、左右比例和双峰 Gaussian 模型 fidelity。

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.78\textwidth]{{{figs["threshold"]}}}
  \caption{{site threshold histogram。}}
\end{{figure}}

\section{{为什么不能用 simulator truth 标定}}
真实实验没有逐 shot ground truth。virtual device 内部当然有用于生成图像的 hidden occupancy state，但这个状态不能出现在 \pyapi{{CaptureResult}}、\pyapi{{DetectionResult}}、session status 或 fidelity API 里。当前 readout-fidelity calibration 的 fidelity 来自可观测图像分布：threshold 两侧的 Gaussian split、左右比例和 reference exposure。这样同一套逻辑可以迁移到真实 qCMOS。

\chapter{{Detect occupancy}}
\section{{推荐调用}}
\begin{{codeblock}}[Python]
shot = exp.readout.detect()
occupied = shot.occupied
\end{{codeblock}}

\pyapi{{detect}} 要求已经有 sitemap 和 threshold。它拍一张 raw image，提取每个 site 的 ROI count，再和 per-site threshold 比较。输出 \pyapi{{DetectionResult.occupied}} 是 boolean array，可直接传给 rearrangement planner 或统计函数。

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.74\textwidth]{{{figs["detect"]}}}
  \caption{{raw image 上叠加 sitemap faint rings 和 occupied rings。}}
\end{{figure}}

\section{{图像设计}}
detect 图不再只画一个抽象 occupancy heatmap。背景是 raw camera data；每个 sitemap site 有较大的浅青灰色 ring，尺寸与 sitemap-found overlay 保持一致；只有判断为 occupied 的 site 再叠加较细的橙色 ring。这样可以同时检查 raw signal、site map 对齐和最终 binary classification。

\chapter{{Detection-time fidelity scan}}
\section{{推荐调用}}
\begin{{codeblock}}[Python]
clock_hz = exp.devices.sequencer.clock_hz
time_ticks = np.linspace(int(round(0.2e-3 * clock_hz)), int(round(10e-3 * clock_hz)), 100, dtype=int)
times = time_ticks / clock_hz
scan = exp.readout.detection_time(times, shots=30, live=True)
\end{{codeblock}}

\pyapi{{detection_time}} 默认 \pyapi{{live=True}}。cell 会很快返回，scan 在 frontend-owned worker 里继续跑；worker 负责采集并写共享 \pyapi{{data_y}}，plot timer 只刷新图。这样采集不会阻塞 Jupyter UI，也给未来 PyQt 主线程绘图留出边界。教程里保留 live scan：等图跑完或需要提前停止时，先运行 \pyapi{{scan.stop()}}，再在后面的 cell 里做 fit。

\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=0.80\textwidth]{{{figs["scan"]}}}
  \caption{{detection time 与 reference-calibrated model fidelity。}}
\end{{figure}}

\section{{停止 live scan：一个对外 stop}}
长扫描对外只提供一个 stop，完成后或需要提前中断时运行：
\begin{{codeblock}}[Python]
scan.stop()
scan.summary()
\end{{codeblock}}

\pyapi{{scan.stop()}} 来自 task base class，它转发到 frontend \pyapi{{RunSession.stop()}}。这个 session 会设置 stop event、调用 source/device 的 \pyapi{{stop()}}（如果存在）、停止 plot refresh timer，并做一次 final refresh，让已采到的数据留在图和 \pyapi{{scan.fidelities}} 里。真实硬件接入时，device/source 的 \pyapi{{stop()}} 必须能打断阻塞采集，否则 Python 线程只能在当前 shot/acquire 返回后自然退出。

\section{{用 scan.data\_figure 做 decay fit}}
拟合前要保证数据已经采完。交互式实验里可以等 live 图跑完，再先执行上一节的 \pyapi{{scan.stop()}} 做 final refresh；然后在后面的 cell 里执行拟合。

scan result 不应该复制 \pyapi{{DataFigure}} 的方法。它只暴露组成部分：\pyapi{{scan.measurement}} 管 lifecycle，\pyapi{{scan.plot}} 管显示，\pyapi{{scan.data\_figure}} 管 fitting/saving/post-processing。因此 decay fit 直接调用已有的 frontend fitting 栈：
\begin{{codeblock}}[Python]
fit_result, popt = scan.data_figure.decay()
scan.summary(), fit_result, popt
\end{{codeblock}}

\section{{reference exposure 的角色}}
\pyapi{{detection_time}} 先拍 long-exposure reference images，用真实实验也能得到的图像分布建立参考。然后每个 detection time 都独立拍 \pyapi{{shots}} 张，估计该 exposure 下的 threshold 和 Gaussian split fidelity。

当前 result 保存：
\begin{{itemize}}
  \item \pyapi{{scan.fidelities}}：画在图上的 fidelity。
  \item \pyapi{{scan.thresholds}}：每个 detection time 的自动 threshold。
  \item \pyapi{{scan.model_fidelities}}：每个 time 的 Gaussian model fidelity。
  \item \pyapi{{scan.reference_counts}}：long-exposure reference ROI counts。
  \item \pyapi{{scan.reference_threshold}} 与 \pyapi{{scan.reference_fidelity}}。
\end{{itemize}}

\begin{{notebox}}[后续更严格的策略]
真实实验如果需要更严谨的 detection fidelity，可以扩展成 paired reference：每次短 exposure detect 后加一张 long exposure reference，或在同一 loading 条件下做 before/after reference。这个策略应该加在 \pyapi{{ReadoutSubsystem}} 或更高层 experiment protocol，不应该塞进 camera，也不是让 virtual truth 混进 fidelity API。
\end{{notebox}}

\chapter{{保存输出}}
\section{{calibration、status、Verilog}}
\begin{{codeblock}}[Python]
cal_path = exp.readout.save("results/calibration.json")
status_path = exp.save_status("results/status.json")
verilog_path = exp.timing.write_verilog("generated_sequences")
\end{{codeblock}}

\pyapi{{save_status}} 会保存 device snapshot、current sequence、calibration 和 history length。Verilog bundle 会写 \filepath{{.v}} 和 manifest，manifest 里包含 clock、channels、ticks、masks 和 source hash。

\chapter{{以后加功能应该放哪里}}
\section{{加真实 camera}}
真实 camera adapter 放在 \filepath{{devices/}}，继承 \pyapi{{CameraDevice}}。它应该只处理硬件生命周期和 image acquisition：
\begin{{itemize}}
  \item \pyapi{{open/close/stop}} 管资源。
  \item \pyapi{{configure}} 管 exposure、ROI、readout mode 等稳定参数。
  \item \pyapi{{acquire}} 管 arm、trigger、wait frame、readout。
  \item \pyapi{{snapshot}} 返回轻量 metadata，给 status log 和 GUI 状态栏使用。
\end{{itemize}}
不要在 camera adapter 里做 threshold、site finding、plot styling、保存实验结果。这些分别属于 \filepath{{operations/}}、\filepath{{views/}} 和 result object。

\section{{加新的图像算法}}
如果算法只需要 image/counts/calibration，就放在 \filepath{{operations/}} 或 \filepath{{core/analysis.py}}。判断标准是：这个函数能不能在没有 camera、没有 session、没有 GUI 的 Python 脚本里跑。如果能，就不要放进 subsystem。subsystem 只是把当前实验状态喂给算法。

\section{{加新的实验动作}}
如果动作需要多个 device、当前 sequence、defaults、calibration 或 history，就放在 \filepath{{subsystems/}}。例如 readout 相关动作放 \pyapi{{ReadoutSubsystem}}；以后 rearrangement 可以是 \pyapi{{RearrangementSubsystem}}，它消费 \pyapi{{DetectionResult.occupied}}，调用 AOD/PXIE device，输出 move plan 和执行 result。

\section{{加新的硬件角色}}
如果是新的硬件类型，先在 \filepath{{devices/base.py}} 定义一个明确 base class，例如 \pyapi{{AODDevice}} 或 \pyapi{{MicrowaveDevice}}，再让 virtual/real implementation 继承它。这样错误会在类实例化或 \pyapi{{load_devices}} 阶段暴露，而不是实验跑到一半才发现少了方法。

\section{{加 GUI}}
GUI 不应该重新实现实验逻辑。GUI panel 持有一个 \pyapi{{NeutralAtomSession}}，按钮只调用 \pyapi{{exp.camera.capture()}}、\pyapi{{exp.readout.detect()}}、\pyapi{{exp.timing.preflight()}} 这类现成接口。画图层继续接受 result 的 \pyapi{{plot}} 或 \pyapi{{data\_figure}}。如果 PyQt 要求绘图在主线程，采集线程只负责写 queue/shared array，UI timer 只负责读数据并刷新 frontend/Qt artist。

\chapter{{下一步扩展计划}}
\section{{真实 qCMOS}}
\begin{{itemize}}
  \item 增加 DCAM error report 和 timeout diagnostics。
  \item 明确 external trigger mode、trigger polarity、readout direction 和 exposure timing。
  \item 对 ROI/subarray 做硬件约束检查。
  \item 保存每次 capture 的 exposure/readout metadata。
\end{{itemize}}

\section{{sequencer 和 address-switch}}
\begin{{itemize}}
  \item 当前 \filepath{{timing/verilog.py}} 已经能从 pulse 生成 edge table。
  \item 下一步把 address-switch register/VIO/download/start handshake 包成 \pyapi{{VerilogSequencer}} 的真实实现。
  \item notebook 仍然只调用 \pyapi{{exp.timing.write_verilog()}}、\pyapi{{exp.timing.preflight()}} 和 camera/readout subsystem。
\end{{itemize}}

\section{{rearrangement 和 GUI}}
\begin{{itemize}}
  \item rearrangement planner 应消费 \pyapi{{DetectionResult.occupied}}，输出 move plan。
  \item AOD/PXIE device 应该是独立 device class，不应该塞进 detector。
  \item GUI panel 持有 session，按钮调用 device/subsystem method；plot surface 继续用 frontend data contract。
  \item PyQt 画图必须在主线程时，采集 worker 只通过 thread-safe queue/shared array 交数据，UI timer 只读数据并刷新。
\end{{itemize}}
"""


def build_neutral_atom_manual(
    output_dir: str | Path = "docs/neutral_atom_manual",
    *,
    compile_pdf: bool = True,
) -> NotesBuildResult:
    """Generate the focused Chinese quickstart manual."""

    output_dir = Path(output_dir)
    figures = _generate_figures(output_dir / "assets")
    body = _manual_body(figures)
    return render_notes_pdf(
        output_dir,
        filename="neutral_atom_manual_zh.tex",
        title="Zou_lab_control.neutral_atom 轻量教程",
        subtitle="Jupyter-first camera / sitemap / threshold / detection session",
        description="中性原子实验控制最小闭环、frontend 画图接口和后续扩展边界",
        body=body,
        doc_date=date.today().isoformat(),
        compile_pdf=compile_pdf,
    )


def build_neutral_atom_hardware_manual(
    output_dir: str | Path = "docs/neutral_atom_hardware_manual",
    *,
    compile_pdf: bool = True,
) -> NotesBuildResult:
    """Generate the real-hardware quickstart manual."""

    output_dir = Path(output_dir)
    figures = generate_hardware_quickstart_figures(output_dir / "assets")
    body = hardware_quickstart_body(figures)
    return render_notes_pdf(
        output_dir,
        filename="neutral_atom_hardware_quickstart_zh.tex",
        title="Zou_lab_control.neutral_atom 硬件接入教程",
        subtitle="qCMOS / FPGA sequencer / frontend live plotting quickstart",
        description="从 virtual 离线闭环到真实双 PC 触发架构的第一版实验线路",
        body=body,
        doc_date=date.today().isoformat(),
        compile_pdf=compile_pdf,
    )


__all__ = ["build_neutral_atom_hardware_manual", "build_neutral_atom_manual"]
