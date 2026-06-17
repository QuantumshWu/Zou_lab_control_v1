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
 device ──(camera.acquire / sequencer.fire)──► Measurement(worker线程, update_mode)
                                                  │ publish 命名信号
                                                  ▼
                                              SignalHub(命名信号 + 版本 + 节点登记)
                                                  ▲                         │
                            Processor(纯变换/逐帧)─┘                         ▼
                              frame→occupied/rate (读 cali)            Plot(纯消费, 订阅信号名, 不绑 measurement)
 Task: 编排 device+measurement+processor+plot, 文件夹流程, 产标定 npz, 中途输出占专用 panel
```

**核心决策**:保留 confocal 的 Measurement(worker 线程 + update_mode + 签名即参数 + 默认 plotter),但把"controller 直连一个 plot"换成"publish 到已有 `SignalHub`、plot 只订阅信号名"。⇒ 一个 measurement 输出可被多图同时连;SignalHub 天然记录"谁产谁消费"喂流向图;前端保持解耦。

## 2. 模块布局

`neutral_atom/operations/` 下:
- `logic.py` —— `LogicNode` 共享基类 + 各 KIND(`CameraMeasurement` / `ScannedMeasurementNode` / `Processor` / `DetectProcessor` / `Task` / `ProcessorRun` 等)+ update-mode 注册表 + 信号发布。
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
- SignalHub 记录每个信号的产出节点与消费节点;每节点声明 produces/consumes。
- **流向图**:节点 = measurement/processor/task/plot,边 = 信号名;前端拓扑画 DAG(满足"谁到谁、经过了谁")。
- 信号命名空间:`<node_ns>.<name>`,可用户别名;跨节点重名靠 ns 区分。

## 9. virtual == real(#2 真做)

- virtual camera **只产 qCMOS 帧**(经 `acquire`/ring)。
- "看 loading"= `live_image`(产 frame)+ `detect_occupancy`(processor 逐帧 frame→occupied/rate)两个独立节点;cali 由 `calibrate_readout`(task)先跑、存 npz,processor 读它。**由独立节点组合而成,没有单个节点全产**。
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
- **P2 Measurement 基类**:worker 线程+update_mode+签名参数+publish;`live_image` 产 frame、`ScannedMeasurementNode` 走扫描 measurement。验:契约测试、虚拟==实机。
- **P3 Processor**:`detect_occupancy` 等逐帧节点 + 合并现 @processor;cali 文件契约。验:frame→occupied 走真 detect;processor 读 npz。
- **P4 Task**:`calibrate_readout` 文件夹流程 + 中途输出通道 + 产物指纹。验:task 产 npz、中途 panel 收到进度。
- **P5 自动 UI**:`params_from_signature` + `auto_form` + 三种 Edit + panel Setting 清理。验:N 参数→N 控件、selector 写回、Edit 分工正确。
- **P6 console 重建**:Add Panel 三类(plot/measurement+processor/task)、接信号、signal 流向图、legend 排版(#1)、中途输出 panel。验:三档 DPR + 四个例子工作流。
- **P7 全验**:契约测试全绿、虚拟==实机端到端、三档 DPR、四例、文档与 tutorial 同步。

## 13. 决策(§13 答案,已定)

① 总称 `Measurement`(update_mode 分连续/扫描);② func=`Processor` 合并;③ update_mode=`single/replace/roll/average/repeat`(注册表);④ 参数=签名+注解(单一真相源);⑤ 信号节点命名空间+可别名,Hub 记产出/消费节点;⑥ task 产物=带指纹 npz/run 文件夹;⑦ Processor 进 live 图逐帧;⑧ pulse 绑定注解 `PulseScan`;⑨ 含基础 load-device+snapshot。

## 14. 关键实现事实

- **相机最近帧 ring**:`CameraDevice` 提供 `recent_capacity`/`latest`/`drain`/`recent_frames`/`clear_recent`,`acquire` 末尾经 `_retain(images)` 入 ring(virtual 与 qcmos 共用,qcmos 不 override ⇒ virtual==real);snapshot 含 roi 对齐。
- **节点基类与 KIND**(`operations/logic.py`):所有逻辑节点继承 `LogicNode`。`Measurement` 类节点带 `UPDATE_MODES`+`update_mode`;`CameraMeasurement` 与 `ScannedMeasurementNode` 都是其子类。`Processor` 为 reactive:`new_inputs()` 按 hub 每信号版本只在输入前进时发,`step()` 返回空 dict = no-op(skip-on-empty)。`DetectProcessor(Processor)` 逐帧跑 `calibration.detect(frame)` → `occupied/counts/rate/rate_sites/rate_grid/centers/thresholds`,是**真 detect 流程**:相机 measurement 只发 `frame`,DetectProcessor 单独跑 detect,`occupied == cal.detect(frame).occupied`。
- **Task 层**:`Task` 基类是 one-shot——`run(out)` 跑完发 result + `task_done` 自停;`TaskOutput` 把中途数值信号(`frame`/`progress`)推到 hub 供专用 panel。`CalibrateReadoutTask` 跑真 sitemap+threshold → `self.calibration`,存 npz 产物并中途出帧。完整 loading 读出 = device + task + processor 组合(无单体节点),全程 virtual==real。
- **update_mode 行为**:`LogicNode.step` 经 `_postprocess` 钩子分派;`Measurement._postprocess` 实现 roll/replace 透传、`single` 发一次自停、`average` 累计均值、`repeat=N` 攒 N 发均值(中途 tick suppress 返回 `{}`)。
- **参数引擎**(`operations/params.py`):注解 spec(`Param/Choice/ScanArray/SignalRef/PulseScan`)+ `ParamField` 记录 + `params_from_signature(fn)`(用 `inspect.signature(eval_str=True)` 解 PEP 563 字符串注解,跳过 `INJECTED`=hub/camera/sequencer/out/calibration/prefix,保序)。前端 auto_form 鸭子类型读 `ParamField` 属性(不 import,保持解耦)。
- **`auto_form`**(`frontend/auto_form.py`):`AutoForm(fields, current=, signal_names=)` 把 ParamField list 渲染成 fluent 行(float/int/str→LineEdit、bool→CheckBox、choice→Combo、signal→可编辑 Combo、array/pulse_scan→`start:stop:step`/逗号 LineEdit),`values()`/`set_values()` 往返;鸭子类型读 ParamField(不 import operations,保解耦);密封。`parse_scan_text` 解析扫描数组。
- **节点基类复用线程/取消/参数队列**:`LogicNode` 是 worker-loop 生产者 + owner-thread 参数队列(`apply_acquisition_parameters`/`_apply_pending_params`/`acquisition_epoch`)+ 协作取消(`stop` event)+ 错误 banner(`node_error`)。`update_mode` 注册表 + 从 build 函数签名派生参数取代手列 `acquisition_parameters`。loading 读出拆成三节点:标定(find_site_centers+estimate_thresholds)→ `calibrate_readout` task;逐帧 detect(roi_counts>thresholds)→ `detect_occupancy` processor;采帧 → `live_image` measurement(只发 `frame`,多触发用 `drain()`)。`ScannedMeasurementNode` 为扫描型(update_mode='replace')。
- **loading 组合入口**:`build_loading_readout` 产组合三节点(calibrate task + camera measurement 发 `frame` + DetectProcessor 跑真 `calibration.detect`);`virtual_loading_readout` 是其虚拟便利工厂,`readout.live_loading_readout`、launcher、console `_add_live_loading_panel` 均经它。
- **task 中途输出专用 panel**:`build_loading_readout` 把 calibrate task 的 `TaskOutput` 命名空间设为 `<prefix>cal_`(`cal_frame`/`cal_progress`,不撞 live `frame`);`_add_live_loading_panel` 加一个 2d "Calibrating (task output)" 面板读 `value = cal_frame`(confocal task 式中途过程)。
- **panel role 与三 Edit**(`frontend/task_console.py`):`PanelConfig.role` ∈ `PANEL_ROLES=("plot","measurement","task")`(持久化,缺省 "plot"),在 Add Panel 建面板时定——结果面板 measurement→"measurement"、processor 面板→"task"、纯图/loading 视图→"plot"。三 Edit 按 role 组装(`PanelEditor` 的 `is_plot` 门):plot=全套(产它的 measurement 参数表单 + 采集参数 + plot 参数 + 快照 + 全 fit 栈 + manual limits);measurement=measurement 参数表单 + 快照(选区→扫描范围回写)**无 fit / 无 limits / 无 plot 参数 / 无采集**;task=data-processing 参数表单 + Run + 快照,同样无 fit。`do_fit`/`fill_limits`/`apply_limits` 经 guard(`fit_combo`/`xmin` 预置 None)。fit 由 **role** 门、绝不由 plot kind 门。
- **Setting 去 plotter-only 项**:非 plot role 不建 relim/unit 行(纯轴显示),保留 source/size/colormap/title/actions。`_spec_for_card` 双路解析(显式 `params["measurement"]` 或按 source 信号匹配 measurement 的 x/y_key),所以一张 plot-role 图指到某 measurement 的结果信号时,它的 plotter Edit 自动带上那 measurement 的参数表单。
- **信号流向(谁→谁→谁)**:`_node_chain(node)` 沿 Processor 的 `consumes` 回溯到上游产出节点(camera ▸ detect),折进每面板 footer 的 "from …" 行。节点的 `layer`/`node_label`/`display_label` 给 footer 提供层名,footer 形如 `from camera ▸ detect [processor]`。
- **`@task` 注册表(对称 `@processor`)**:`TaskSpec`(name/build(hub)→Task/mid_run_key/default_kind/prefix)+ `@task`/`register_task`/`discovered_task_specs`(按 prefix 去重防信号撞车)+ 内置 `@task calibrate_readout`(build 走 `readout.calibrate_task`)+ `readout.task_specs()`。console 的 `tasks=` 参数喂 kind_combo 列 `Task: <name>`,`_add_task_panel(spec)` 通用(spec.build(hub) + 中途 `spec.mid_run_signal()` 面板)。task 参数 + Run 走 Task 自己的 `acquisition_parameters`(`CalibrateReadoutTask` 含它 + `mid_run=("frame","progress")`,`Task.published_signals` 含 mid_run 让中途面板能映射回 task)。Add Panel 三类直接暴露每层:`Measurement: Camera (live frames)`(`readout.camera_measurement`→`CameraMeasurement`)、`Processor: <名>`、`Task: <名>`、`Readout: Loading(组合)`。PanelEditor Acquisition 区 gate = `spec is None and proc_spec is None`(camera/task 无 MeasurementSpec 时靠它显示参数)。
