# task_console 架构 —— 设计规范(权威)

> 这是 task_console 五层架构的**设计规范**(维护者视角,非面向用户手册;用户教学在 `task_console_design_zh.texbody`)。改任何东西前先读仓库 `AGENTS.md` + 本文件。

## 0. 目的与现状病灶

task_console 及其上游分为清晰五层,解决以下结构性约束:
1. 所有逻辑节点共享同一 `LogicNode` 基类与同一套循环/更新机制,而不是各类节点各写一份。
2. 参数声明从函数签名派生(单一真相源),不手列两套。
3. 连续 live 与 data-processing 统一为同一节点模型。
4. 相机缓冲放在相机里,不放在节点层。
5. 有真正的 task 层(detect-site/cali 由 task 编排,不塞进 processor)。
6. **virtual==real**:看 loading 走真机式"相机出帧 → 真流程 cali → detect",由独立节点组合而非单个节点全产。
7. 节点模型不泄漏进 frontend;自动 UI 完整;signal legend 排版整齐。

## 1. 五层总览 + 数据流

```
 device ──(camera.acquire / sequencer.fire)──► Measurement(worker线程, 拥有 repeat 轴 = 填块)
                                                  │ publish (repeat,*points,*data) 块
                                                  ▼
                                              SignalHub(命名信号 + 版本 + 节点登记)
                                                  ▲                         │
                            Processor(纯类型化变换)─┘                        ▼
                              frame→occupied/rate (读 cali)            Plot(纯消费, repeat_mode 合并显示, 订阅信号名)
 Task: 编排 device+measurement+processor+plot, 文件夹流程, 产标定 npz, 中途输出占专用 panel
```

**核心决策**:保留 confocal 的 Measurement(worker 线程 + 签名即参数 + 默认 plotter),但把"controller 直连一个 plot"换成"publish 到已有 `SignalHub`、plot 只订阅信号名"。⇒ 一个 measurement 输出可被多图同时连;SignalHub 天然记录"谁产谁消费"喂流向图;前端保持解耦。**repeat 模型(#H3o/#H3u-2)**:全管线只有两个 repeat 旋钮——measurement 的唯一 `repeat`(0=∞,拍几次、填 `(ring,*points,*data)` 块,块深 `_ring=max(1,repeat)`,无单独 free_run)与 plot 的 `repeat_mode`(怎么合并显示)。Processor 是**纯类型化变换、无用户 mode**(静态类属性 `repeat_contract=reduce|preserve`,不进表单);旧的基类 `update_mode`/`_postprocess`/`repeats` 累积是死的第三套,已删。详见 `MAINTAINER_NOTES.md` 的「Repeat: TWO systems」节。

## 2. 模块布局

`neutral_atom/operations/` 下:
- `logic.py` —— `LogicNode` 共享基类 + 各 KIND(`CameraMeasurement` / `ScannedMeasurementNode` / `Processor` / `OccupancyProcessor` / `Task` / `ProcessorRun` 等)+ update-mode 注册表 + 信号发布。
- registry —— `@measurement` / `@processor` / `@task` 装饰器 + `discovered_*` 发现。
- `params.py` —— 参数注解工具 `Param/Choice/ScanArray/SignalRef/PulseScan` + `params_from_signature(fn)`(签名→ParamField 列表)。
- task —— `Task` 基类 + 中途输出 channel。
- 纯算法(检测/拟合等)与节点解耦,节点参数从函数签名派生而非手列 spec。

`devices/`:`base.py` 统一 `snapshot()`;`CameraDevice` 加 ring-buffer(`arm/latest/drain`);`virtual.py` 虚拟相机只产 qCMOS 帧。

`frontend/`:`task_console.py` 重建(Add Panel 三类 + 自动 UI + 三种 Edit + signal 流向图 + legend 排版);复用 `BaseLivePlot/DataFigure/qt_fluent`。

## 3. Device 层

```python
class BaseDevice:
    name: str
    def connect(self): ...
    def close(self): ...
    def snapshot(self) -> dict: ...          # 单一公共方法:记录可复现状态(存数据时聚合)
class CameraDevice(BaseDevice):
    def acquire(self, frames, *, sequence=None, sequencer=None) -> np.ndarray: ...  # 既有契约,不变
    def arm(self, n_buffer=64): ...          # 起采集线程 + ring(deque maxlen=N),每 trigger 入一帧
    def latest(self): ...                    # live 取最新一帧
    def drain(self): ...                     # 取"上次以来全部新帧"(无损);measurement 用它
```
- 防丢帧缓冲**在相机里**(不在节点层)。virtual camera 用同一 ring,只把"trigger 来帧"换成仿真帧;measurement/processor 全走同一契约 ⇒ virtual==real。

## 4. Measurement 层

```python
@measurement(produces=('frame',), default_plot={'frame': '2d'},
             devices=('camera','sequencer'))
def live_image(camera, sequencer, *, exposure: Param(float, 0.1, unit='s')):
    frame = camera.acquire(1, sequence=...)[-1]   # 只产相机帧;不在这里 detect
    return {'frame': frame}
```
- **参数 = build 函数签名 + 类型注解**(`params_from_signature`)。API 直接调;GUI 自动出表单(§7)。无手列 spec。
- **worker 线程**:`Measurement.start()` 后台跑,每点 publish 一次并推进版本;不直接刷 plot。
- **repeat 轴 = measurement 拥有**:**唯一**一个 `repeat:int` 参数,**0 = ∞**(#H3u-2,控制台自动注入 ParamDecl,无单独 free_run 开关)——`K>0` 保留 K 深块(K 趟/帧均值后停)、`0` 滚动 1 深环不停;块深 `_ring=max(1,repeat)`。measurement 填 `(ring,*points,*data)` 块**整块** publish,**从不**自己合并 repeat 轴。怎么合并显示是 plot 的 `repeat_mode`(§见 plot 层 / MAINTAINER_NOTES「Repeat」)。
- **publish**:返回 `{name: value}` → Hub publish(命名空间见 §8)。
- **pulse 绑定**:扫描类用 `PulseScan('pulse_name','target')` 注解,运行时写 `PulseTableState` scan 表(已硬件化)。
- **plot=True / plot=False 分流**:notebook 里直接 API 调一个 measurement(如 `readout.temperature(...)`)默认 `display=True` → 自动出适配的默认图(`ScannedMeasurement.run(display=True)`)。但在 task console / GUI 里,measurement 作为 logic 节点由 `ScannedMeasurementNode.step()` 逐点驱动、**只 publish 到 hub、从不调 `.run()`** ⇒ 等价 plot=False:不自动出图,用户自己加 Plot 面板按 signal 配。
- **相机 measurement 的可编辑参数 = 相机自己的(`exposure` / `frames_per_cycle` / `region` ROI)**,经 `CameraMeasurement.acquisition_parameters / set_acquisition_parameters → camera.configure(exposure=, roi=)`。`region` 是传感器像素端点 `[x0,x1,y0,y1]`(空=全幅);端点↔设备 ROI 的换算只在 `set_acquisition_parameters` 一处。**虚拟相机与真机同一 `CameraDevice` 契约**,Edit 表单完全一致(`readout.camera_spec()` 声明这三个 ParamDecl,notebook 与 GUI 共用 `camera_spec().build(hub, exposure=, region=)`)。

## 5. Processor 层(= 用户的 func)

```python
@processor(consumes=('frame',), produces=('occupied','rate','centers'),
           needs_calibration=True)
def detect_occupancy(frame, *, calibration: SignalRef|str):
    cal = resolve_calibration(calibration)
    return occupancy_from_image(frame, cal)       # 复用 readout 单一契约,不重实现
```
- 纯变换;无采集循环/无线程/无 fit。两种用法:(a) **live 信号图节点**(订阅 `frame` → 每帧产 `occupied`,这就是 virtual==real 的真 detect 流程,#2);(b) 一次性调用。
- 合并现有 `@processor`。rb87 逐站点判读是典型。

## 6. Task 层

```python
@task(produces=('site_centers','thresholds'))
def calibrate_readout(camera, sequencer, *, n_frames: Param(int,200),
                      folder: Param(str,'cali/run1'), out):   # out = 中途输出通道
    out.stage('collecting', frames_done=0)
    frames = collect_frames(camera, n_frames, folder, progress=out)   # 边采边 publish 到中途 panel
    out.stage('detecting')
    centers = detect_sites(frames); thr = fit_thresholds(frames, centers)
    save_calibration(folder, centers, thr)        # npz + 指纹(复用现 fingerprint)
    return {'site_centers': centers, 'thresholds': thr}
```
- task = 编排 measurement/processor/device + 文件夹流程;输出可为文件 + 少量标量。
- **中途输出**:task 拿一个 `out`(`TaskOutput`)通道(仿 confocal task),把中间帧/进度写进它**自己的缓冲**(`node.output`),由 console 的固定面板读保留键 `__task_frame__` 显示(§10)。**不进 hub**(#6:hub 只承载 measurement+processor 输出);结果/标定留在 `node.result`/`node.calibration`,大产物写文件。

### calibrate-readout task 的完整参数设计(`CalibrateReadoutTask` + `CALIBRATE_PARAMS`)
所有参数声明一次(`operations/tasks/calibrate.py` 的 `CALIBRATE_PARAMS`,GUI 自动表单与 `readout.calibrate_task(**values)` 同读),逐项:
- **source = live / saved frames**:`live` = 现在开相机直接采图并写标定;`saved frames`(`folder`)= 用已存文件夹的原始帧(`index_run` 读 `img<n>`)cali。**没有 "saved calibration" 这个 source**:复用一份已存标定直接让 Judge-occupancy 指它的 `calibration.json`。
- **模板本身就是 long-short-long(file == fired)+ API slot 设曝光**:`pulses/imaging_template.json` 文件本身就是 6 个 period 的 bracket(`load`/`image_0`/`gap_0`/`image_1`/`gap_1`/`image_2`)——一次 cooling cycle,然后**三个连续 emCCD 帧**,帧间只 hold trap 的 gap(让相机落下再升 = 三个独立 trigger;gap 里 cooling/probe/emCCD 全关,**不重做 cooling**,否则原子乱掉、标签失效)。cali **不派生 bracket**(已删 `with_imaging_bracket`),它只 load 模板、用 **API slot** 按名设那两个曝光 duration:`set_api("a1", reference_exposure)`(两长参考帧共用 `a1`,投真值 + 建 site map / PSF)+ `set_api("a2", readout_exposure)`(中间短读出帧,阈值在真实读出条件下学)。两者都是显式 cali 参数;在 pulse GUI 打开模板看到的就是 cali 实际发的 long-short-long。site map 用 bracket 的全部长参考帧(live==saved 单一路径)。
- **readout method = box / per-site PSF / uniform PSF**:cali **不**选 method,它把三种全算进一份标定;OccupancyProcessor 选其一。
- **threshold = otsu / bimodal**;`threshold_frames`(bracket 发数)/`roi_radius`。
- **复用标定**:重跑覆盖 `folder`;复用一份已标好的标定不在 cali 里——让 Judge-occupancy load 它的 `calibration.json`。
- 同一 `TrapCalibration` 契约(`calibrate_sitemap_from_images` / `calibrate_threshold_from_images`),虚拟与真机只差相机帧。守:`tests/test_task_cali_modes_and_plot_split.py`。

## 7. 参数注解 + 自动 UI 引擎

`params.py`:`params_from_signature(fn)` → `[ParamField(name, kind, default, unit, choices, meta)]`。
| 注解 | 控件 |
|---|---|
| `Param(int/float, default, unit, lo, hi)` | 数字输入(validator+单位) |
| `Param(bool, default)` | 勾选框 |
| `Choice(options, default)` | 下拉 |
| `ScanArray(default)` | 扫描数组编辑器(start/stop/step 或 Load) |
| `SignalRef('name')` | 信号选择下拉(把参数接到 hub 信号) |
| `PulseScan(pulse, target)` | 选 pulse/slot + 扫描范围 |

前端 `auto_form.py`:吃 `ParamField` 列表 → 渲染 fluent 控件(复用 qt_fluent);selector 框选写回声明了 `ScanArray`/区域的参数。**密封约束**:只吃 ParamField,不收 data_px/margins_px/spec/dpi。

## 8. plot+controller + SignalHub + 流向图

- plot 纯消费:订阅**信号名**,GUI 定时按版本水位刷新;不绑 measurement。复用 `BaseLivePlot/DataFigure/selector`。
- SignalHub 记录每个信号的产出节点与消费节点;每节点声明 produces/consumes。
- **流向图**:节点 = measurement/processor/task/plot,边 = 信号名;前端拓扑画 DAG(满足"谁到谁、经过了谁")。
- 信号命名空间:`<node_ns>.<name>`,可用户别名;跨节点重名靠 ns 区分。

## 9. virtual == real(#2 真做)

- virtual camera **只产 qCMOS 帧**(经 `acquire`/ring)。
- "看 loading"= `CameraMeasurement`(相机 measurement,产 `frame`)+ `OccupancyProcessor`(processor 逐帧 `frame`→`occupied`/`rate`,跑真 `calibration.detect`)两个独立节点;cali 由 `CalibrateReadoutTask`(task)先跑、写回标定 / 存 npz,OccupancyProcessor 用它。**由独立节点组合而成,没有单个节点全产**。
- 由 `tests/test_virtual_equals_real_contract.py` 守:分析层不 import 后端、不读仿真真值;端到端虚拟读出走同一契约。
- **virtual 帧由脉冲驱动的原子物理模型生成**(`devices/virtual.py` `VirtualTrapArray`):每次 `acquire` 是一次 shot,先 MOT 加载(~50% 伯努利)+ PGC 冷到 `cooled_temperature_K`,再由 fired `PulseSequence` 逐帧演化 per-site **占据 + 温度**——cooling/MOT 脉冲(re)加载、probe 散射成像 + 读出丢失 + recoil 加热、trap-off gap 按**当前温度**做弹道 release-recapture 丢失。于是 loading rate / 温度(release-recapture)/ 读出保真度都从同一物理"涌现",只经相机帧被恢复(虚实同契约不变)。可调常数在 `virtual.py` dataclass 顶部,经 `connect("virtual", sitemap=/params=...)` 调;交互式 launcher 用 `sleep_scale=1.0` 给可感知耗时(测试默认 0,快)。`tests/test_virtual_atom_physics.py` 守这些效应。

## 10. 两个常驻 tab(Monitor / Logic)+ 解耦 + 三种 Edit

- **两个不可关 tab**:`Monitor`(plot 面板的拖拽板)+ `Logic`(measurement/processor/task 逻辑节点的列表)。控制台开局 **空 hub、不自动启动任何东西**。
- **VIEW / LOGIC 解耦**:加一个 **Plot** 面板 = 纯 VIEW,**不会启动任何 measurement**;它一开始空白,只有在 Setting 里配了 signal **且**产出该 signal 的逻辑节点被 Start 之后才显示。空 hub 时 plot 的 signal 选择器只有 `(expression)`,不会凭空冒一堆 signal。
- 加一个 **measurement / processor / task** = 一个 **Logic 节点行**进 Logic tab,**默认停**(行点灰 GREY)。在它**自己的 Edit** 里 Start(行点绿 GREEN)/ Stop;出错红 RED(`LogicNodeRow.STATE_COLORS`)。控制台两套集合:`console.logic_nodes` = Logic 行(声明,不管跑不跑都在);`console.running_nodes` = 当前在跑的已建节点(Start 时 append、Stop 时 remove)。
- **三种 Edit**:
  - **Logic 节点 Edit**(measurement/processor/task/camera):其 ParamDecl 自动表单(§7;相机来自 `readout.camera_spec()`)+ Start/Stop;**无 fit**(fit 是 plotter 的事)。
  - **Plot 面板 Edit**:Source 段 = 产出该图信号的 measurement/processor 节点的**完整参数表单**(预填其当前值;Apply 对 camera 等 live 参数原地下发、否则重建重跑该源节点,#2)+ fit 栈(DataFigure)+ relim/colorset 等 plotter 项 + manual lim;**无 Start/Stop**(节点在它自己的 Logic Edit 起停)。一个 plot 的 signal 总来自某 measurement/processor,所以从 plot 也能看/调它的源参数。
  - measurement 在 GUI 强制 plot=False(见 §4),所以 Plot 与 measurement 解耦于**生命周期**(加 plot 不启动任何东西),但 plot Edit 仍能调它**当前所读源**的参数。
- **没有 preset**:Add Panel 只有 plot / measurement / processor / task 四类(无自造 readout 复合体);唯一可复用的布局是你 Save 出来的文件(Load / `--task` 读回),save 含各 Edit 表单参数(#4),无内置预设。
- **task 运行接管控制台**(#5,confocal 式):Start 一个 task 时,console 在 Monitor 开一张**固定**面板显示它的 `node.output` 中途帧(读保留键 `__task_frame__`,不经 hub,#6),顶部橙色横幅 “Task running …%”,并**禁用其他一切操作**(Add/Save/Load/Edit/其他节点 Start),只留 Stop;task 完成或 Stop 后解锁并撤掉该瞬态面板。`console._running_task_row`/`_task_card`/`_task_locked` + `_refresh_task_panel`(在 `_tick` version-gate 前,因 task 输出不 bump hub version)。

## 11. signal legend 排版(#1)

panel 空白处的"读/发信号"图例:加**左右 padding**,过长**按需分行**(word-wrap + 每类一行:Reads / Provides / ⚠重名),不再贴边挤成一团。owned 常量在 frontend。

## 12. 阶段计划(每阶段独立可验, virtual==real, 删旧 clean-delete)

- **P1 Device**:`snapshot()` 统一 + CameraDevice ring-buffer(arm/latest/drain)+ virtual 出帧。验:snapshot 往返、drain 无损、虚拟帧契约。
- **P2 Measurement 基类**:worker 线程+签名参数+publish;measurement 拥有 repeat 轴(填 `(ring,*points,*data)` 块,唯一 `repeat` 参,0=∞);`live_image` 产 frame、`ScannedMeasurementNode` 走扫描 measurement。验:契约测试、虚拟==实机。
- **P3 Processor**:`detect_occupancy` 等逐帧节点 + 合并现 @processor;cali 文件契约。验:frame→occupied 走真 detect;processor 读 npz。
- **P4 Task**:`calibrate_readout` 文件夹流程 + 中途输出通道 + 产物指纹。验:task 产 npz、中途 panel 收到进度。
- **P5 自动 UI**:`params_from_signature` + `auto_form` + 三种 Edit + panel Setting 清理。验:N 参数→N 控件、selector 写回、Edit 分工正确。
- **P6 console 重建**:Add Panel 三类(plot/measurement+processor/task)、接信号、signal 流向图、legend 排版(#1)、中途输出 panel。验:三档 DPR + 四个例子工作流。
- **P7 全验**:契约测试全绿、虚拟==实机端到端、三档 DPR、四例、文档与 tutorial 同步。

## 13. 决策(§13 答案,已定)

① 总称 `Measurement`(连续/扫描都是它,拥有 repeat 轴 = 填块);② func=`Processor` 合并(纯类型化变换,无用户 mode,静态 `repeat_contract`);③ repeat 全管线只两旋钮:measurement `repeat`(0=∞,单旋钮无 free_run)+ plot `repeat_mode`(`average/add/replace/roll/create`),旧基类 `update_mode` 累积已删(#H3o);④ 参数=签名+注解(单一真相源);⑤ 信号节点命名空间+可别名,Hub 记产出/消费节点;⑥ task 产物=带指纹 npz/run 文件夹;⑦ Processor 进 live 图逐帧;⑧ pulse 绑定注解 `PulseScan`;⑨ 含基础 load-device+snapshot。

## 14. 关键实现事实

- **相机最近帧 ring**:`CameraDevice` 提供 `recent_capacity`/`latest`/`drain`/`recent_frames`/`clear_recent`,`acquire` 末尾经 `_retain(images)` 入 ring(virtual 与 qcmos 共用,qcmos 不 override ⇒ virtual==real);snapshot 含 roi 对齐。
- **节点基类与 KIND**(`operations/logic.py`):所有逻辑节点继承 `LogicNode`。`Measurement` 类节点拥有 repeat 轴(唯一 `repeat` 参,0=∞,块深 `_ring=max(1,repeat)`,填 `(ring,*points,*data)` 块);`CameraMeasurement` 与 `ScannedMeasurementNode` 都是其子类。`Processor` 为 reactive:`new_inputs()` 按 hub 每信号版本只在输入前进时发,`step()` 返回空 dict = no-op(skip-on-empty);静态 `repeat_contract`(`reduce`/`preserve`)声明它对 repeat 轴的关系,**不是用户 mode**。`OccupancyProcessor(Processor)` 是 `preserve`(#H3q/#H3s-F3):**逐 repeat 切片**跑 `calibration.detect` → `occupied`/`counts` 为**干净** `(repeat, n_sites)` 块(前导 repeat 轴,无多余中间 1)、`frame_judged` 为 `(repeat, H, W)` 块、`centers`(N,2)/`thresholds`(N,) 静态、`rate` 标量(本块装载率);删了累积 `rate_sites`/`rate_grid`(= `repeat_mode=average` 的重复)。repeat 折叠**由结构驱动而非 ndim 猜测**:每信号的 `SignalSpec` 声明各自 `points_shape`/`data_shape`,`core_ndim`(occupancy 的 points=()/data=(n_sites,) → 1)告诉 `reduce_repeat(block, mode, core_ndim=...)` 当 `block.ndim == 1 + core_ndim` 时 axis 0 即 repeat;不传 `core_ndim` 时保留旧 ndim≥3 回退(相机/扫描块字节级不变)。`repeat_mode=average` over `occupied` = 逐站装载概率。是**真 detect 流程**:相机 measurement 只发 `frame`,OccupancyProcessor 单独跑 detect,`occupied[r] == cal.detect(frame[r]).occupied`。
- **Task 层**:`Task` 基类是 one-shot——`run(out)` 跑完把结果存上 `self.result` 后自停,`Task.shot()` 向 hub **发 0 信号**、`Task.published_signals()` 为空(#6:task 不进 hub)。`TaskOutput`(`node.output`)是 task 自有缓冲(非 hub),`run` 把中途数值信号(`frame`/`progress`)写进它供固定面板(`__task_frame__`)显示。`CalibrateReadoutTask` 跑真 sitemap+threshold → `self.calibration`,存 npz 产物并中途出帧。完整 loading 读出 = device + task + processor 组合(无单体节点),全程 virtual==real。
- **repeat 行为(#H3o)**:measurement 每 `shot()` 填整块 `(repeat,*points,*data)` 并整块 publish(`LogicNode.step` 不再有合并钩子);怎么把 repeat 轴合并成画面全部交给 plot 的 `repeat_mode`(`live.reduce_repeat`,只对 `ndim>=3` 块生效)。processor 不参与合并——`reduce` 型出无 repeat 轴结果、`preserve` 型出 `>=3-D` 块复用同一 `reduce_repeat`。旧 `Measurement.UPDATE_MODES/_postprocess/update_mode/repeats/_accum` 已删(死的第三套)。
- **参数引擎**(`operations/params.py`):注解 spec(`Param/Choice/ScanArray/SignalRef/PulseScan`)+ `ParamField` 记录 + `params_from_signature(fn)`(用 `inspect.signature(eval_str=True)` 解 PEP 563 字符串注解,跳过 `INJECTED`=hub/camera/sequencer/out/calibration/prefix,保序)。前端 auto_form 鸭子类型读 `ParamField` 属性(不 import,保持解耦)。
- **`auto_form`**(`frontend/auto_form.py`):`AutoForm(fields, current=, signal_names=)` 把 ParamField list 渲染成 fluent 行(float/int/str→LineEdit、bool→CheckBox、choice→Combo、signal→可编辑 Combo、array/pulse_scan→`start:stop:step`/逗号 LineEdit),`values()`/`set_values()` 往返;鸭子类型读 ParamField(不 import operations,保解耦);密封。`parse_scan_text` 解析扫描数组。
- **节点基类复用线程/取消/参数队列**:`LogicNode` 是 worker-loop 生产者 + owner-thread 参数队列(`apply_acquisition_parameters`/`_apply_pending_params`/`acquisition_epoch`)+ 协作取消(`stop` event)+ 错误 banner(`node_error`)。从 build 函数签名派生参数取代手列 `acquisition_parameters`(measurement 再自动注入唯一 `repeat`,0=∞)。loading 读出拆成三节点(§4–§6 代码块里的 `live_image`/`detect_occupancy`/`calibrate_readout` 是示意签名,真实节点类如下):标定(find_site_centers+estimate_thresholds)→ `CalibrateReadoutTask`(task);逐帧 detect(roi_counts>thresholds)→ `OccupancyProcessor`(processor);采帧 → `CameraMeasurement`(measurement,只发 `frame`,多触发用 `drain()`)。`ScannedMeasurementNode` 为扫描型(填 `(repeat,N,dim)` 块)。
- **loading 读出 = 用户用三类节点自己拼**(无 `build_loading_readout` 组合入口、无自造 "Readout: Loading" 复合体):在 Logic tab 加一个 `Measurement: Camera (live frames)`(`readout.camera_measurement`→`CameraMeasurement`,发 `frame`)+ 一个 `Processor`(逐帧 `OccupancyProcessor` 跑真 `calibration.detect`)+ 一个 `Task: Calibrate readout`(先跑、写回标定 / 产 npz),各自 Start,再在 Monitor 加 Plot 面板按 signal 连。launcher(`task_console.py`)`na.connect` 后只 `sitemap`/`thresholds` 自标定让 catalog 能跑,console 开 **空 hub、全停**,不自动建/启任何读出。
- **task 输出不进 hub**(#6):`Task` 自带 `TaskOutput` 缓冲(`node.output`,非 hub),`run(out)` 把中途帧/进度写进它;`Task.shot()` 向 hub 发 **0** 信号、`Task.published_signals()` 为空,结果/标定留在 `node.result`/`node.calibration`。**hub 只承载 measurement + processor 输出**——一次性 task 的瞬态帧永不混进 live 信号或被误当读出。console 起 task 时把 `node.output` 绑到 Monitor 一张**固定**面板(保留键 `__task_frame__`,见 §10),`TaskSpec.mid_run_key` 选缓冲键(如 `frame`)。
- **panel 永远是 plot 视图 + 两类 Edit 各属一套类**(`frontend/task_console.py`):`PanelConfig.role` ∈ `PANEL_ROLES=("plot",)`——board 上的 panel **只有 plot 一种角色**(纯视图),不再有 measurement/task 角色面板。两类 Edit 分属两套类:`PanelEditor`(plot 面板的 Edit,`is_plot` 恒真)= Acquisition(它所读信号的**产出 logic 节点**自报的源参数 + Apply 原地下发)+ Parameters(plot 自己的 API 参数)+ Processing(快照 + 全 fit 栈 + manual limits + Save);`LogicNodeEditor`(Logic tab 上 measurement/processor/task/camera 节点的 Edit)= 该节点 ParamDecl 自动表单 + Start/Stop,**无 fit / 无 limits**(拟合是 plotter 的事,去 Plot 面板做)。`do_fit`/`fill_limits`/`apply_limits` 经 guard(`fit_combo`/`xmin` 预置 None);fit 由 **role(恒 plot)** 门、绝不由 plot kind 门。
- **plot 面板 Edit 的 Acquisition 区 = 数据源(logic 节点)自报参数**(解耦,#4):`PanelEditor` 经 `console._producing_node(card)` 找到产这张图所读信号的 logic 节点,用 `console._node_params(node)`(节点 `acquisition_parameters()` 自报)列出可编辑源参数(相机 = `exposure`/`frames_per_cycle`/`region`),每格带 `now:` 当前值,Apply 经 `_restart_node` 把改动**原地**下发(`camera.configure` 实时重配,不 start——节点从它自己的 Logic-tab Edit start)。plot 面板**绝不**自带某 measurement 的"参数表单 + Start"(`PanelEditor.meas_panel` 恒 None);要重跑一个扫描就去 Logic tab 调那节点。Setting 弹窗只留 source/size/colormap/relim/unit/title/actions。
- **信号流向(谁→谁→谁)**:`_node_chain(node)` 沿 Processor 的 `consumes` 回溯到上游产出节点(camera ▸ occupancy),折进每面板 footer 的 "from …" 行。节点的 `layer`/`node_label`/`display_label` 给 footer 提供层名,footer 形如 `from camera ▸ occupancy [processor]`。
- **`@task` 注册表(对称 `@processor`)**:`TaskSpec`(`name`/`build(hub)`→Task/`params`(ParamDecl 列)/`mid_run_key`/`default_kind`/`prefix`)+ `@task`/`register_task`/`discovered_task_specs` + 内置 `@task calibrate_readout`(build 走 `readout.calibrate_task(hub, prefix="cal_", **values)`)+ `readout.task_specs()`。console 的 `tasks=` 喂 kind_combo 列 `Task: <name>`;选中 Add Panel → `_add_logic_node(LogicNodeConfig(kind="task", name=...))` 加一个**停着的 Logic 行**(与 measurement/processor 同路),Start 时 `_build_logic_node` 走 `spec.build(self.hub, **values)`。task 参数 + Run 走 Task 自己的 `acquisition_parameters`(`CalibrateReadoutTask` 含它 + `mid_run=("frame","progress")`);**task 不向 hub publish**——中途帧在 `node.output` 缓冲,由 console 固定面板显示(§10、#6)。kind_combo 严格四类:`Plot: <kind>` / `Measurement: Camera (live frames)`(`readout.camera_measurement`→`CameraMeasurement`)+ `Measurement: <扫描名>` / `Processor: <名>`(如反应式 `Judge occupancy`)/ `Task: <名>`——无自造 "Readout: Loading" 复合体。
