# task_console 架构 v2 —— 实现规范(权威)

> 这是进行中大重构的**实现规范**(非面向用户手册)。完成并稳定后,面向用户的教学并入 `task_console_design_zh.texbody`,本文件届时可删。改任何东西前先读仓库 `AGENTS.md` + 本文件。

## 0. 目的与现状病灶

把 task_console 及其上游从"补丁式 producer/feed 模型"重做成清晰五层。现状病灶:
1. 三个并列 producer(`LoadingFeed`/`ScannedMeasurementFeed`/`ProcessorFeed`)各写各的循环/更新。
2. 参数声明手列两套(`MeasurementSpec.params` / `ProcessorSpec`),非函数签名派生。
3. 连续 live 是特例 feed、data-processing 另一套 spec,模型不统一。
4. 相机缓冲错放在 feed 层。
5. 无真正 task 层(detect-site/cali 硬塞进 ProcessorSpec)。
6. **virtual 作弊**:一个 `LoadingFeed` 一次产 frame/rate/occupied/centers,没走真机式"相机出帧 → 真流程 cali → detect"。
7. producer 模型泄漏进 frontend;自动 UI 半成品;signal legend 排版乱。

## 1. 五层总览 + 数据流

```
 device ──(camera.acquire / sequencer.fire)──► Measurement(worker线程, update_mode)
                                                  │ publish 命名信号
                                                  ▼
                                              SignalHub(命名信号 + 版本 + producer/consumer 登记)
                                                  ▲                         │
                            Processor(纯变换/逐帧)─┘                         ▼
                              frame→occupied/rate (读 cali)            Plot(纯消费, 订阅信号名, 不绑 measurement)
 Task: 编排 device+measurement+processor+plot, 文件夹流程, 产标定 npz, 中途输出占专用 panel
```

**核心决策**:保留 confocal 的 Measurement(worker 线程 + update_mode + 签名即参数 + 默认 plotter),但把"controller 直连一个 plot"换成"publish 到已有 `SignalHub`、plot 只订阅信号名"。⇒ 一个 measurement 输出可被多图同时连;SignalHub 天然记录"谁产谁消费"喂流向图;前端保持解耦。

## 2. 模块布局

新增/改写(`neutral_atom/operations/` 下):
- `producers/__init__.py` `base.py` —— `Measurement` 基类 + `Processor` 基类 + update-mode 注册表 + 信号发布。
- `producers/registry.py` —— `@measurement` / `@processor` / `@task` 装饰器 + `discovered_*`(取代现 measurement_registry/processor_registry)。
- `params.py` —— 参数注解工具 `Param/Choice/ScanArray/SignalRef/PulseScan` + `params_from_signature(fn)`(签名→ParamField 列表)。
- `task.py` —— `Task` 基类 + 中途输出 channel。
- 删:`feeds.py` 的 `LoadingFeed/ScannedMeasurementFeed/ProcessorFeed`(迁入 producers);`measurement.py`/`processor*.py` 旧 spec(被签名派生取代,保留纯算法)。

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
- 防丢帧缓冲**在相机里**(不在 feed)。virtual camera 用同一 ring,只把"trigger 来帧"换成仿真帧;measurement/processor 全走同一契约 ⇒ virtual==real。

## 4. Measurement 层

```python
@measurement(produces=('frame',), default_plot={'frame': '2d'},
             update_modes=('roll','replace'), devices=('camera','sequencer'))
def live_image(camera, sequencer, *, exposure: Param(float, 0.1, unit='s'),
               update_mode: Choice(('roll','replace'), 'roll')):
    frame = camera.acquire(1, sequence=...)[-1]   # 只产相机帧;不在这里 detect
    return {'frame': frame}
```
- **参数 = build 函数签名 + 类型注解**(`params_from_signature`)。API 直接调;GUI 自动出表单(§7)。无手列 spec。
- **worker 线程**:`Measurement.start()` 后台跑,每点 publish 一次并推进版本;不直接刷 plot。
- **update_mode**:`single/replace/roll/average/repeat`(注册表,可扩展);声明 `update_modes` 限定可选。
- **publish**:返回 `{name: value}` → Hub publish(命名空间见 §8)。
- **pulse 绑定**:扫描类用 `PulseScan('pulse_name','target')` 注解,运行时写 `PulseTableState` scan 表(已硬件化)。

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
- **中途输出**:task 拿一个 `out` 通道(仿 confocal task),把中间帧/进度 publish 到一个**专用 panel**(§10)。能画的 publish 到 hub;不能画的写状态/文件。

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
- SignalHub 扩:`producers[name]` / `consumers[name]` 登记;每节点声明 produces/consumes。
- **流向图**:节点 = measurement/processor/task/plot,边 = 信号名;前端拓扑画 DAG(满足"谁到谁、经过了谁")。
- 信号命名空间:`<producer_ns>.<name>`,可用户别名;跨 producer 重名靠 ns 区分。

## 9. virtual == real(#2 真做)

- virtual camera **只产 qCMOS 帧**(经 `acquire`/ring)。
- "看 loading"= `live_image`(产 frame)+ `detect_occupancy`(processor 逐帧 frame→occupied/rate)两个独立节点;cali 由 `calibrate_readout`(task)先跑、存 npz,processor 读它。**没有一个 feed 全产**。
- 由 `tests/test_virtual_equals_real_contract.py` 守:分析层不 import 后端、不读仿真真值;端到端虚拟读出走同一契约。

## 10. 三种 Edit + panel Setting 清理(#3)

- **Measurement Edit**:只有参数表单(§7)+ Start/Stop/update_mode;**无 fit**(无意义)。
- **Plotter Edit**:fit 栈(DataFigure)+ relim/colorset 等 plotter 项 + **它所连 measurement 的参数表单**(可就地改重启)。
- **Task Edit**:参数表单 + Run + 进度;中途输出 → 专用 panel。
- panel Setting:非 plot 面板**去掉只对 plotter 有用的项**(fit/colorset/relim 只在 plotter Edit 出现)。

## 11. signal legend 排版(#1)

panel 空白处的"读/发信号"图例:加**左右 padding**,过长**按需分行**(word-wrap + 每类一行:Reads / Provides / ⚠重名),不再贴边挤成一团。owned 常量在 frontend。

## 12. 阶段计划(每阶段独立可验, virtual==real, 删旧 clean-delete)

- **P1 Device**:`snapshot()` 统一 + CameraDevice ring-buffer(arm/latest/drain)+ virtual 出帧。验:snapshot 往返、drain 无损、虚拟帧契约。
- **P2 Measurement 基类**:worker 线程+update_mode+签名参数+publish;迁 `LoadingFeed`→`live_image`、`ScannedMeasurement`→扫描 measurement;删旧 feed。验:迁移后契约测试、虚拟==实机。
- **P3 Processor**:`detect_occupancy` 等逐帧节点 + 合并现 @processor;cali 文件契约。验:frame→occupied 走真 detect;processor 读 npz。
- **P4 Task**:`calibrate_readout` 文件夹流程 + 中途输出通道 + 产物指纹。验:task 产 npz、中途 panel 收到进度。
- **P5 自动 UI**:`params_from_signature` + `auto_form` + 三种 Edit + panel Setting 清理。验:N 参数→N 控件、selector 写回、Edit 分工正确。
- **P6 console 重建**:Add Panel 三类(plot/measurement+processor/task)、接信号、signal 流向图、legend 排版(#1)、中途输出 panel。验:三档 DPR + 四个例子工作流。
- **P7 全验**:契约测试全绿、虚拟==实机端到端、三档 DPR、四例、文档与 tutorial 同步。

## 13. 决策(§13 答案,已定)

① 总称 `Measurement`(update_mode 分连续/扫描);② func=`Processor` 合并;③ update_mode=`single/replace/roll/average/repeat`(注册表);④ 参数=签名+注解(单一真相源);⑤ 信号 producer 命名空间+可别名,Hub 记 producer/consumer;⑥ task 产物=带指纹 npz/run 文件夹;⑦ Processor 进 live 图逐帧;⑧ pulse 绑定注解 `PulseScan`;⑨ 含基础 load-device+snapshot;⑩ 旧 churn 已回滚。

## 14. 实现进度 + 关键实现决策(随做随记)

- **P1 ✅ 完成并验证**(commit 待做):`CameraDevice` 加最近帧 ring(`recent_capacity`/`_retain`/`latest`/`drain`/`recent_frames`/`clear_recent`),`acquire` 末尾 `return self._retain(images)`(virtual+qcmos);snapshot 加 roi 对齐。测试 `tests/test_camera_recent_buffer.py`(4 例,含"qcmos 不 override = virtual==real")。虚拟==实机契约 + multitrigger 仍绿。
- **P2/P3 核心 ✅ 已落地并验证**(`operations/feeds.py`):`ExperimentFeed`→`Producer`(基类,改名,4 处 importer 已改,21 测试绿);新增三 KIND——`Measurement`(加 `UPDATE_MODES`+`update_mode`,`CameraFrameFeed`/`ScannedMeasurementFeed` 已 reparent 到它)、`Producer`(基)、`Processor`(reactive:`new_inputs()` 按 hub 每信号版本只在输入前进时发,`step()` 空 dict = no-op)、`DetectProcessor(Processor)`(逐帧 `calibration.detect(frame)`→`occupied/counts/rate/rate_sites/rate_grid/centers/thresholds`,**真 detect 流程**)。`step()` 改 skip-on-empty。测试 `tests/test_producer_split_contract.py`:相机 measurement 只发 `frame`,DetectProcessor 单独跑真 detect,`occupied==cal.detect(frame).occupied`,reactive no-op;virtual==real 解耦契约仍绿。**这就是 #2 的架构核心**。
- **P4 ✅ 已落地并验证**:`Task` 基类(one-shot:`run(out)` 跑完发 result+`task_done` 自停)+ `TaskOutput`(中途数值信号:`frame`/`progress` 到 hub 供专用 panel)+ `CalibrateReadoutTask`(真 sitemap+threshold→`self.calibration`,存 npz 产物,中途出帧)。测试 `test_producer_split_contract.py::test_calibrate_task_produces_calibration_and_feeds_detect`:task 产 calibration+中途 frame+progress=1+npz,再喂 DetectProcessor 逐帧检测——**完整 loading 读出 = device+task+processor 组合,无单体 feed,virtual==real 全程**。
- **update_mode 行为 ✅**:`Producer.step` 加 `_postprocess` 钩子;`Measurement._postprocess` 实现 roll/replace 透传、`single` 发一次自停、`average` 累计均值、`repeat=N` 攒 N 发均值(中途 tick suppress 返回 {})。测试 `test_measurement_update_mode.py` 5 例绿。
- **P5 引擎 ✅ 已落地**:`operations/params.py` —— 注解 spec(`Param/Choice/ScanArray/SignalRef/PulseScan`)+ `ParamField` 记录 + `params_from_signature(fn)`(用 `inspect.signature(eval_str=True)` 破 PEP 563 字符串注解,跳过 `INJECTED`=hub/camera/sequencer/out/calibration/prefix,保序)。前端 auto_form 鸭子类型读 `ParamField` 属性(不 import,保持解耦)。测试 `test_params_from_signature.py` 2 例绿(py3.13)。
- **P5 `auto_form` ✅ 已落地**:`frontend/auto_form.py` —— `AutoForm(fields, current=, signal_names=)` 把 ParamField list 渲染成 fluent 行(float/int/str→LineEdit、bool→CheckBox、choice→Combo、signal→可编辑 Combo、array/pulse_scan→`start:stop:step`/逗号 LineEdit),`values()`/`set_values()` 往返;鸭子类型读 ParamField(不 import operations,保解耦);密封。`parse_scan_text` 解析扫描数组。测试 `test_auto_form.py` 2 例绿。**复用件齐了:params + auto_form + producer kinds(measurement/processor/task)。**
- **下一步(P6,均在 `frontend/task_console.py` ~3200 行内 + 删旧迁移,blast radius 大,宜 compaction 后专注做)**:① 三 Edit 用 `AutoForm` 组装——Measurement Edit=参数表单无 fit / Plotter Edit=fit 栈+relim/colorset + 其所连 measurement 的参数表单 / Task Edit=参数+Run+进度;非 plot 面板 Setting 去 plotter-only 项(#3);② 删 `LoadingFeed`,把 `virtual_loading_feed`/`readout.live_loading_feed`/launcher/console 改产组合(`CalibrateReadoutTask`+`CameraFrameFeed`+`DetectProcessor`),迁 ~10 文件(console/tests/tutorial/docs);③ console 重建 Add Panel 三类 + signal legend **左右 padding + 按需分行**(#1)+ signal 流向图(hub producers/consumers→DAG)+ task 中途输出专用 panel(#3,confocal task 式);④ P7 全验 + 三档 DPR + 四例工作流 + tutorial/docs 同步。
- **P2 关键决策(别从零重写而回归!)**:`operations/feeds.py` 的 `ExperimentFeed` **已经是** worker-loop 生产者 + owner-thread 参数队列(`apply_acquisition_parameters`/`_apply_pending_params`/`acquisition_epoch`)+ 协作取消(`stop` event)+ 错误 banner(`feed_error`)——这些都修过真机 bug,**必须复用**。P2 = 把 `ExperimentFeed` **重构为 `Measurement` 基类**(加 `update_mode` 注册表 + 从 build 函数签名派生参数,取代手列 `acquisition_parameters`),并**拆掉单体 `LoadingFeed`**:`_calibrate()`(find_site_centers+estimate_thresholds)→ `calibrate_readout` **task**(P4);逐发 detect(roi_counts>thresholds)→ `detect_occupancy` **processor**(P3);采帧 → `live_image` **measurement**(只发 `frame`)。`CameraFrameFeed` 的 frame_i 多触发逻辑并入 `live_image`(用 P1 的 `drain()`)。`ScannedMeasurementFeed` → 扫描型 measurement(update_mode='replace')。删旧三 feed 类。这样既得清晰五层,又不丢线程/取消/参数队列的正确性。
- **#2 ✅(LoadingFeed 彻底删除)**:单体 `LoadingFeed` 类 + 全部 export 删净;`virtual_loading_feed`→`virtual_loading_readout`(走 `build_loading_readout`)、`readout.live_loading_readout`、devtools demo、根 launcher 两支、console `_add_live_loading_panel` 全改产组合三节点(calibrate task + camera measurement 发 `frame` + DetectProcessor 跑真 `calibration.detect`)。grep `LoadingFeed` = 0 代码引用。
- **#4 ✅(task 中途输出专用 panel)**:`build_loading_readout` 把 calibrate task 的 `TaskOutput` 命名空间设为 `<prefix>cal_`(`cal_frame`/`cal_progress`,不撞 live `frame`);`_add_live_loading_panel` 加一个 2d "Calibrating (task output)" 面板读 `value = cal_frame`(confocal task 式中途过程)。
- **P6 ✅ 完成并三档 DPR 验收**(`frontend/task_console.py`):新增 **`PanelConfig.role`**(`PANEL_ROLES=("plot","measurement","task")`,持久化,旧 layout 缺省 "plot");role 在 Add Panel 建面板时定——结果面板 measurement→"measurement"、processor 面板→"task"、纯图/loading 视图→"plot"。**三 Edit 按 role 组装**(`PanelEditor.__init__` 的 `is_plot` 门):plot=全套(产它的 measurement 参数表单 + 采集参数 + plot 参数 + 快照 + **全 fit 栈(#176 不回归)** + manual limits);measurement=measurement 参数表单 + 快照(选区→扫描范围回写仍在)**无 fit / 无 limits / 无 plot 参数 / 无采集**;task=data-processing 参数表单 + Run + 快照,同样无 fit。`do_fit`/`fill_limits`/`apply_limits` 已 guard(`fit_combo`/`xmin` 预置 None)。**Setting 去 plotter-only 项**(#3):非 plot role 不建 relim/unit 行(纯轴显示),保留 source/size/colormap/title/actions。**`_spec_for_card` 双路解析**:显式 `params["measurement"]` 或按 source 读到的信号匹配 measurement 的 x/y_key——所以一张 plot-role 图指到某 measurement 的结果信号时,它的 plotter Edit 自动带上那 measurement 的参数表单(#3「plotter edit 也有对应 measurement 的参数」)。**信号流向(谁→谁→谁)**:`_producer_chain(feed)` 沿 Processor 的 `consumes` 回溯到上游生产者(camera ▸ detect),折进每面板 footer 的 "from …" 行——无需在 hub 加 producer/consumer 记账。测试 `tests/test_panel_role_edit.py`(5 例:role 往返+校验 / measurement Edit 无 fit+Setting 去 relim·unit / plot Edit 留全 fit+显示项 / plot 读 measurement 信号→带其参数表单+留 fit / task Edit 无 fit)+ `test_task_console_live_loading.py` 加流向链断言。三档 DPR(1.0/1.25/1.5)真窗口 widget.grab 验收 measurement Edit:Measurement 表单+Start/Stop+Processing 快照,无 Fit/Limits 区,无裁切无重叠。**关键不回归**:fit 由 **role** 门,绝不由 plot kind 门(否则 revert #176)。36 contract 测试(三强制契约+console role/legend/measurement)+ 63 新架构测试合跑全绿。
- **返工轮(用户暴怒)+ `@task` 注册表完成层暴露**:四个最根本诉求我"交付"时全没做(没真开窗口看 + 没回读最初 frame 诉求)。修:① **frame 整体 revert `24edbda`**(用户最初就要回滚的 commit):标题条回左上 + Setting 回右上 + `_CARD_PAD=10`(signal 不再贴边,贴边只是 `_CARD_PAD=0` 的症状)+ 去 `FluentGroupBox.padding_top` 参数;② **Add Panel 按层全暴露**——根因是 `MeasurementSpec` 只支持扫描型、且没 task 注册表,camera 持续流 + calibrate task 都埋在 loading 组合里。现 Add Panel:`Measurement: Camera (live frames)`(`readout.camera_measurement`→CameraFrameFeed)、`Processor: <名>`、`Task: <名>`(来自新 **`@task` 注册表**)、`Readout: Loading(组合)`;③ **去 UI 里的 "Feed" 类名**:Producer 加 `layer`/`node_label`/`display_label`(只 node_label,prefix 是信号命名细节),footer 现 `from camera ▸ detect [processor]`(层名+哪层)。**`@task` 注册表(对称 `@processor`)**:`operations/task.py`(`TaskSpec`:name/build(hub)→Task/mid_run_key/default_kind/prefix)+ `operations/task_registry.py`(`@task`/`register_task`/`discovered_task_specs`,按 prefix 去重防信号撞车)+ `operations/tasks/calibrate.py`(内置 `@task calibrate_readout`,build 走 `readout.calibrate_task`)+ `readout.task_specs()`。console 加 `tasks=` 参数,kind_combo 从 `self.tasks` 列 `Task: <name>`,`_add_task_panel(spec)` 通用(spec.build(hub)+中途 `spec.mid_run_signal()` 面板)。复用关键:task 参数 + Run 走 Task 自己的 `acquisition_parameters`(`CalibrateReadoutTask` 加了它 + `mid_run=("frame","progress")`,`Task.published_signals` 含 mid_run 让中途面板能映射回 task);PanelEditor Acquisition 区 gate 改 `spec is None and proc_spec is None`(camera/task 无 MeasurementSpec 靠它显示参数)。测试 `test_task_registry.py`(发现/注册往返/dup-prefix 报错/build 返 unrun Task)+ `test_panel_role_edit.py` 第 6 例(camera+task 可加+层名干净);三档 DPR 真窗口验收(标题左上+Setting 右上+signal 内缩+Camera/Detect/Calibrate 各 Edit tab+Task Edit Run+无 fit)。**教训重刻**:声称完成前必须真开入口窗口截图看 + 回读用户原始诉求逐条对(见 verify-against-user-acceptance)。
