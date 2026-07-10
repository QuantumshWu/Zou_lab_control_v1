# task_console 架构 —— 设计规范(权威)

> 这是 task_console 五层架构的**设计规范**(维护者视角,非面向用户手册;用户教学在 `task_console_design_zh.texbody`)。改任何东西前先读仓库 `AGENTS.md` + 本文件。

## 0. 目的与现状病灶

task_console 及其上游分为清晰五层,解决以下结构性约束:
1. 所有逻辑节点共享同一 `LogicNode` 基类与同一套循环/更新机制,而不是各类节点各写一份。
2. 参数由 dependency-free `ParamDecl` 显式声明一次；API 默认、catalog spec 与所有 GUI 表单共读，不从签名/AST 猜。
3. 连续 live 与 data-processing 统一为同一节点模型。
4. 相机缓冲放在相机里,不放在节点层。
5. 有真正的 task 层(detect-site/cali 由 task 编排,不塞进 processor)。
6. **virtual==real**:看 loading 走真机式"相机出帧 → 真流程 cali → detect",由独立节点组合而非单个节点全产。
7. 节点模型不泄漏进 frontend;自动 UI 完整;signal legend 排版整齐。

## 1. 五层总览 + 数据流

```
 device ──(camera.acquire / sequencer.fire)──► Measurement(worker线程, 拥有 repeat 轴 = 填块)
                                                  │ publish SignalTensor (R,P,*data_shape)
                                                  ▼
                                              SignalHub(命名信号 + 版本 + 节点登记)
                                                  ▲                         │
                            Processor(纯类型化变换)─┘                        ▼
                            frame_0→occupied/rate (读 cali)          Plot(纯消费, repeat_mode 合并显示, 订阅信号名)
 Task: 编排 device+measurement+processor+plot, 文件夹流程, 产标定 npz, 中途输出占专用 panel
```

**核心决策**:Measurement/Processor 只向 `SignalHub` 发布命名的 `SignalTensor`，plot 只订阅信号名。Hub 的 `SignalSchema` 是每个信号的轴、dtype、validity 与坐标元数据单一来源：物理形状固定为 `(R,P,*data_shape)`，只把逻辑 `point_shape` 展成 P，任意维 `data_shape` 原样保留。全管线只有 measurement 的 `repeat` 与 plot 的 `repeat_mode` 两个旋钮；Processor 是纯 typed transform，每个输出声明自己的 schema，没有 processor-side repeat mode/contract。

## 2. 模块布局

`neutral_atom/operations/` 下:
- `logic.py` —— `LogicNode` 共享基类 + `Measurement` / `Processor` / `Task` 运行模型与 typed publish。
- `_spec.py` + `_open_registry.py` —— `CatalogSpec` 与唯一 open-registry 机制；measurement / processor / task 只是三种薄 spec/registry。
- `measurements/` / `processors/` / `tasks/` —— 各领域 factory；参数来自 `core/params.py::ParamDecl`。
- `core/signal_tensor.py` / `core/signals.py` —— `SignalSchema`、`SignalTensor`、patch journal、cursor/provenance。
- `core/fitting.py` / `core/selection.py` —— fit/selection 的 headless 单源；节点只做 adapter。

`devices/`:`base.py` 统一 `snapshot()`;`CameraDevice` 加 ring-buffer(`arm/latest/drain`);`virtual.py` 虚拟相机只产 qCMOS 帧。

`frontend/`:`task_console.py` 负责 board/Logic/Setting/Edit；`param_widgets.py::PARAM_WIDGETS` 是所有 `ParamDecl.kind` 的唯一控件分发；复用 `BaseLivePlot/DataFigure/qt_fluent`。

## 3. Device 层

```python
class BaseDevice:
    name: str
    def connect(self): ...
    def close(self): ...
    def snapshot(self) -> dict: ...          # 单一公共方法:记录可复现状态(存数据时聚合)
class CameraDevice(BaseDevice):
    def arm(self, frames=None): ...          # 返回时硬件已就绪等外触发
    def read_frames(self, n=1, *, timeout=None, stop=None) -> list: ...
    def disarm(self): ...                    # 纯 grabber
    def latest(self): ...                    # recent ring 的最新帧
    def drain(self): ...                     # 取并清空 recent ring
```
- 采集所有权在相机设备；测量层唯一编排是 `arm → sequencer prepare/fire → read_frames → disarm`。recent ring 也在相机里，不在节点层；virtual 与 real 共享同一契约。

## 4. Measurement 层

- `MeasurementSpec(CatalogSpec)` 显式携带 `params: tuple[ParamDecl,...]`、输出键、metadata 与 build/make-node 边界；API 默认和 GUI 表单共读这一个 record。
- `CameraMeasurement` 按 pulse 的触发事件预声明 `frame_0/frame_1/...`，每个输出 schema 为 `point_shape=(1), data_shape=(H,W)`，物理 tensor 为 `(R,1,H,W)`；不存在 lumped frame。
- Measurement 拥有唯一 `repeat:int`（0=∞），只填 R；plot 的 `repeat_mode` 才能折叠 R。worker 只 publish typed tensor/patch，不直接操作图。
- `ScannedMeasurementNode` 预注册 coordinate/y stores，逐点 patch `(R,P,*data_shape)`；`point_shape` 可多维，尾部 `data_shape` 从不展平。
- `PulseScanNode` 始终只拥有 sequencer 与外部 y cursor，但支持两种显式策略：`scan_slot` 把完整 `PulseTableState.scan_table` 一次 prepare/fire；`api_slot` 对 program 的每一行调用 `set_api`、编译并发射一个有限 pulse。表单值固定为 `{program_id, api, sweep_kind, program}`。两条路径都发布语义坐标、消费外部 producer 的下一条 lineage-coherent typed 更新，且都不拥有相机或 relay frame。
- notebook 一键 API 可选择自动绘图；task console 中节点只 publish，plot 生命周期完全独立。相机的 exposure/region/frames-per-cycle 也由同一 `ParamDecl` 经 owner-thread 参数队列下发。

## 5. Processor 层(= 用户的 func)

- `ProcessorSpec(CatalogSpec)` 声明 `params/result_keys/make_node`；open registry 负责发现与 collision 检查，GUI 不选择具体类。
- `Processor` 按每个输入 signal 的 cursor 取得下一组 lineage-coherent `SignalTensor`，没新数据就 no-op；它不拥有设备或 repeat knob。
- 每个输出有独立 `SignalSpec/SignalSchema`。cell-wise transform 保留所有有效 `(R,P)`；aggregate 必须显式声明 `(1,1,*data_shape)`，不能靠 squeeze/rank 推断。
- `OccupancyProcessor` 默认消费 `frame_0=(R,P,H,W)`，发布 `occupied/counts=(R,P,N)`、`rate=(R,P,1)`、`frame_judged=(R,P,H,W)`、`centers=(1,1,N,2)`、`thresholds=(1,1,N)`。
- `FitProcessor` 只是 Hub adapter；模型、selection、solver 与 quality metric 只在 `core.fitting`，与 `DataFigure.fit(model, request=FitRequest(...))` 共用。

## 6. Task 层

- `TaskSpec(CatalogSpec)` 用同一组 `ParamDecl` 描述一次性编排的输入、build、mid-run key 与默认视图。
- task 可编排 device/measurement/processor 与文件流程；大产物写文件，最终结果留在 `node.result/node.calibration`。
- **中途输出**:task 拿一个 `out`(`TaskOutput`)通道(仿 confocal task),把中间帧/进度写进它**自己的缓冲**(`node.output`),由 console 的固定面板读保留键 `__task_frame__` 显示(§10)。**不进 hub**(#6:hub 只承载 measurement+processor 输出);结果/标定留在 `node.result`/`node.calibration`,大产物写文件。

### calibrate-readout task 的完整参数设计(`CalibrateReadoutTask` + `CALIBRATE_PARAMS`)
所有参数声明一次(`operations/tasks/calibrate.py` 的 `CALIBRATE_PARAMS`,GUI 自动表单与 `readout.calibrate_task(**values)` 同读),逐项:
- **source = live / saved frames**:`live` = 现在开相机直接采图并写标定;`saved frames`(`folder`)= 用已存文件夹的原始帧(`index_run` 读 `img<n>`)cali。**没有 "saved calibration" 这个 source**:复用一份已存标定直接让 Judge-occupancy 指它的 `calibration.json`。
- **模板本身就是 long-short-long(file == fired)+ API slot 设曝光**:`pulses/imaging_template.json` 文件本身就是 6 个 period 的 bracket(`load`/`image_0`/`gap_0`/`image_1`/`gap_1`/`image_2`)——一次 cooling cycle,然后**三个连续 emCCD 帧**,帧间只 hold trap 的 gap(让相机落下再升 = 三个独立 trigger;gap 里 cooling/probe/emCCD 全关,**不重做 cooling**,否则原子乱掉、标签失效)。cali **不派生 bracket**(已删 `with_imaging_bracket`),它只 load 模板，并按三个唯一 API handle 写三个 exposure cell：`a1` 与 `a3` 分别接收同一个 `reference_exposure`，`a2` 接收 `readout_exposure`。两个参数值都是显式 cali 参数；每个 handle 仍严格只绑定一个 field。在 pulse GUI 打开模板看到的就是 cali 实际发的 long-short-long。site map 用 bracket 的全部长参考帧(live==saved 单一路径)。
- **readout method = box / per-site PSF / uniform PSF**:cali **不**选 method,它把三种全算进一份标定;OccupancyProcessor 选其一。
- **threshold = otsu / bimodal**;`threshold_frames`(bracket 发数)/`roi_radius`。
- **复用标定**:重跑覆盖 `folder`;复用一份已标好的标定不在 cali 里——让 Judge-occupancy load 它的 `calibration.json`。
- 同一 `TrapCalibration` 契约(`calibrate_sitemap_from_images` / `calibrate_threshold_from_images`),虚拟与真机只差相机帧。守:`tests/test_task_cali_modes_and_plot_split.py`。

## 7. ParamDecl + 统一控件注册表

`core/params.py::ParamDecl` 是唯一参数 record；常用 `kind` 包括
`float/int/bool/choice/text/json/path/axis_range/signal/pulse_param/signal_expr/pulse_slots/device_ref`。
declaration 同时携带 default、optional/required、bounds、unit、choices、dependency 与 tooltip。

`frontend/param_widgets.py::PARAM_WIDGETS` 将每种 kind 映射到一个 `ParamWidgetHandler`，统一实现
`build/read/write/is_empty/refresh`。Logic Edit、plot Setting/Edit、device manager 与运行时 controls
都通过同一注册表；增加 kind 只新增一个 handler 与 registry entry。`pulse_slots` widget 的值严格是
`{program_id, api, sweep_kind, program}`，其中 `sweep_kind∈{scan_slot, api_slot}`；不会生成 camera、
frame、delay 或第二套 mode 字段。所有结构化输入用 typed parser/JSON，永不 eval。

## 8. plot+controller + SignalHub + 流向图

- producer 先注册 `SignalSchema`，再发布 `SignalTensor` 或 `TensorPatch`；Hub 原子校验 exact
  `(R,P,*data_shape)`、dtype、`(R,P)` validity、schema version 与 provenance，保存不可变副本。
- `latest_tensor/history_tensor/snapshot_at/cursor` 是 typed 消费接口；有界 patch journal 支持
  reactive processor 与 PulseScan 逐更新读取，而不是轮询一份会被覆盖的 raw dict。
- plot 纯消费信号名/`SignalExpr`，按 schema 选择兼容 view；plot 不绑定 measurement 生命周期。
- Hub 记录 provider/consumer，前端流向图以逻辑节点/plot 为节点、signal 为边。prefix 只是明确的
  public signal key 组成规则，不创建第二层隐式 namespace。

## 9. virtual == real(#2 真做)

- virtual camera **只产 qCMOS 帧**(经 `acquire`/ring)。
- "看 loading"= `CameraMeasurement`(产 `frame_0`)+ `OccupancyProcessor`(`frame_0`→`occupied`/`rate`，跑同一 `calibration.detect`)两个独立节点；cali 由 `CalibrateReadoutTask` 先跑并写文件。没有单个节点全产。
- 由 `tests/test_virtual_equals_real_contract.py` 守:分析层不 import 后端、不读仿真真值;端到端虚拟读出走同一契约。
- **virtual 帧由脉冲驱动的原子物理模型生成**(`devices/virtual.py` `VirtualTrapArray`):每次 `acquire` 是一次 shot,先 MOT 加载(~50% 伯努利)+ PGC 冷到 `cooled_temperature_K`,再由 fired `PulseSequence` 逐帧演化 per-site **占据 + 温度**——cooling/MOT 脉冲(re)加载、probe 散射成像 + 读出丢失 + recoil 加热、trap-off gap 按**当前温度**做弹道 release-recapture 丢失。于是 loading rate / 温度(release-recapture)/ 读出保真度都从同一物理"涌现",只经相机帧被恢复(虚实同契约不变)。可调常数在 `virtual.py` dataclass 顶部,经 `connect("virtual", sitemap=/params=...)` 调;交互式 launcher 用 `sleep_scale=1.0` 给可感知耗时(测试默认 0,快)。`tests/test_virtual_atom_physics.py` 守这些效应。

## 10. 两个常驻 tab(Monitor / Logic)+ 解耦 + 三种 Edit

- **两个不可关 tab**:`Monitor`(plot 面板的拖拽板)+ `Logic`(measurement/processor/task 逻辑节点的列表)。控制台开局 **空 hub、不自动启动任何东西**。
- **VIEW / LOGIC 解耦**:加一个 **Plot** 面板 = 纯 VIEW,**不会启动任何 measurement**;它一开始空白,只有在 Setting 里配了 signal **且**产出该 signal 的逻辑节点被 Start 之后才显示。空 hub 时 plot 的 signal 选择器只有 `(expression)`,不会凭空冒一堆 signal。
- 加一个 **measurement / processor / task** = 一个 **Logic 节点行**进 Logic tab,**默认停**(行点灰 GREY)。在它**自己的 Edit** 里 Start(行点绿 GREEN)/ Stop;出错红 RED(`LogicNodeRow.STATE_COLORS`)。控制台两套集合:`console.logic_nodes` = Logic 行(声明,不管跑不跑都在);`console.running_nodes` = 当前在跑的已建节点(Start 时 append、Stop 时 remove)。
- **三种 Edit**:
  - **Logic 节点 Edit**(measurement/processor/task/camera):其 ParamDecl 自动表单 + Start/Stop；无 plot fit。
  - **Plot Setting / Edit**:两处都通过同一个 `FitRequest(model, selection)` 写 `PanelConfig.params["fit_request"]`；模型按 plot family 过滤，没有自由文本 fit 参数或动态方法调用。Setting 负责 live 操作，Edit 仍提供冻结快照、源参数、limits/command/save；无 Start/Stop。
  - measurement 在 GUI 强制 plot=False(见 §4),所以 Plot 与 measurement 解耦于**生命周期**(加 plot 不启动任何东西),但 plot Edit 仍能调它**当前所读源**的参数。
- **没有 preset**:Add Panel 只有 plot / measurement / processor / task 四类(无自造 readout 复合体);唯一可复用的布局是你 Save 出来的文件(Load / `--task` 读回),save 含各 Edit 表单参数(#4),无内置预设。
- **task 运行接管控制台**(#5,confocal 式):Start 一个 task 时,console 在 Monitor 开一张**固定**面板显示它的 `node.output` 中途帧(读保留键 `__task_frame__`,不经 hub,#6),顶部橙色横幅 “Task running …%”,并**禁用其他一切操作**(Add/Save/Load/Edit/其他节点 Start),只留 Stop;task 完成或 Stop 后解锁并撤掉该瞬态面板。`console._running_task_row`/`_task_card`/`_task_locked` + `_refresh_task_panel`(在 `_tick` version-gate 前,因 task 输出不 bump hub version)。

## 11. signal legend 排版(#1)

panel 空白处的"读/发信号"图例:加**左右 padding**,过长**按需分行**(word-wrap + 每类一行:Reads / Provides / ⚠重名),不再贴边挤成一团。owned 常量在 frontend。

## 12. 阶段计划(每阶段独立可验, virtual==real, 删旧 clean-delete)

- **P1 Device**:`snapshot()` 统一 + CameraDevice ring-buffer(arm/latest/drain)+ virtual 出帧。验:snapshot 往返、drain 无损、虚拟帧契约。
- **P2 Measurement 基类**:worker 线程 + `MeasurementSpec/ParamDecl` + typed publish；measurement 拥有 R，`SignalSpec` 为每个输出声明 `point_shape/data_shape`，Hub 强制 `(R,P,*data_shape)`。
- **P3 Processor**:cursor-driven typed transform + explicit output schema；验:`frame_0→occupied` 走同一 detect，processor 读 calibration 文件。
- **P4 Task**:`calibrate_readout` 文件夹流程 + 中途输出通道 + 产物指纹。验:task 产 npz、中途 panel 收到进度。
- **P5 自动 UI**:`ParamDecl + PARAM_WIDGETS/ParamWidgetHandler` + 三种 Edit；验:N declarations→N controls、typed 往返、Edit 分工正确。
- **P6 console 重建**:Add Panel 四类(plot/measurement/processor/task)、接信号、流向图与 task 中途 panel。验:三档 DPR + 四个例子工作流。
- **P7 全验**:契约测试全绿、虚拟==实机端到端、三档 DPR、四例、文档与 tutorial 同步。

## 13. 决策(§13 答案,已定)

① `Measurement` 拥有 R；② `Processor` 是 schema→schema typed transform，无 repeat mode；③ plot `repeat_mode` 只改变显示；④ `SignalSchema` 是轴/shape/dtype/validity 单源；⑤ 参数只有 `ParamDecl`；⑥ task 产物带 schema/指纹；⑦ PulseScan 只拥有 sequencer 与外部 y cursor，执行策略显式为 `scan_slot` 或 `api_slot`。

## 14. 关键实现事实

- **相机最近帧 ring**:`CameraDevice` 提供 `recent_capacity`/`latest`/`drain`/`recent_frames`/`clear_recent`,`acquire` 末尾经 `_retain(images)` 入 ring(virtual 与 qcmos 共用,qcmos 不 override ⇒ virtual==real);snapshot 含 roi 对齐。
- **节点基类与 KIND**(`operations/logic.py`):所有逻辑节点继承 `LogicNode`。Measurement 注册并发布 typed tensors；Processor 用 cursor 读取下一组 lineage-coherent `SignalTensor`，按输入 validity 变换后发布显式 output schema。`OccupancyProcessor` 对每个有效 `(R,P)` 图像 cell 跑 `calibration.detect`，输出 `occupied/counts=(R,P,N)`、`frame_judged=(R,P,H,W)`、`centers=(1,1,N,2)`、`thresholds=(1,1,N)`。任何多维 `data_shape` 都保留；UI 只从 `hub.schema()` 取结构。
- **Task 层**:`Task` 基类是 one-shot——`run(out)` 跑完把结果存上 `self.result` 后自停,`Task.shot()` 向 hub **发 0 信号**、`Task.published_signals()` 为空(#6:task 不进 hub)。`TaskOutput`(`node.output`)是 task 自有缓冲(非 hub),`run` 把中途数值信号(`frame`/`progress`)写进它供固定面板(`__task_frame__`)显示。`CalibrateReadoutTask` 跑真 sitemap+threshold → `self.calibration`,存 npz 产物并中途出帧。完整 loading 读出 = device + task + processor 组合(无单体节点),全程 virtual==real。
- **repeat 行为(#H3o)**:transport 永远有显式 R/P；plot 把 schema 与 validity 交给 `reduce_repeat`，只折叠 R。是否存在 R 从 schema 得知，绝不靠 `ndim`。Processor 的 aggregate/cell-wise 语义体现在各输出 schema，不另设字符串 contract。
- **参数引擎**:`core/params.py::ParamDecl` 是 dependency-free 唯一声明；`CatalogSpec.params`、API 默认与所有表单共读。`frontend/param_widgets.py::PARAM_WIDGETS` 以 `ParamWidgetHandler` 注册 kind，禁止签名反射/AST/eval 产生第二份 schema。
- **sequencer 拓扑**:`PortCatalog` 是 raw lane、logical digital/DAC/clock port、DAC bus index/encoding/safe value/latch clock 的唯一不可变来源。XDC/device 配置在边界构造一次；`PulseTableState` 只持有 `port_catalog`，pulse JSON 带 catalog + fingerprint，不接受 channels/labels/buses/clocks 平行字段。Pulse GUI 只读显示 port identity，不能改硬件拓扑。
- **PulseScan 双策略、单 owner**:表单 `pulse_slots` 只往返 `{program_id, api, sweep_kind, program}`。`scan_slot` 使用模板的 `scan_code/scan_table` 并整表一次上传/fire；`api_slot` 将 `program` 每一行映射到 semantic API handles，逐点生成有限 pulse。两者都只拥有 sequencer、都从外部 producer 取 y；MOT-field 模板只允许 `scan_slot`。
- **节点基类复用线程/取消/参数队列**:`LogicNode` 是 worker-loop 生产者 + owner-thread 参数队列(`apply_acquisition_parameters`/`_apply_pending_params`/`acquisition_epoch`)+ 协作取消(`stop` event)+ 错误状态。节点参数来自 catalog spec 的 `ParamDecl`；measurement 追加唯一 `repeat`(0=∞)。`ScannedMeasurementNode` 预注册 `SignalSchema` 并填 canonical `(R,P,*data_shape)`。
- **loading 读出 = 三节点显式组合**:`CameraMeasurement(frame_0) → OccupancyProcessor(occupied/counts/rate)`，标定由独立 `CalibrateReadoutTask` 产文件；各自 Start，Monitor plot 只订阅 signal。console 开局空 hub、全停。
- **task 输出不进 hub**(#6):`Task` 自带 `TaskOutput` 缓冲(`node.output`,非 hub),`run(out)` 把中途帧/进度写进它;`Task.shot()` 向 hub 发 **0** 信号、`Task.published_signals()` 为空,结果/标定留在 `node.result`/`node.calibration`。**hub 只承载 measurement + processor 输出**——一次性 task 的瞬态帧永不混进 live 信号或被误当读出。console 起 task 时把 `node.output` 绑到 Monitor 一张**固定**面板(保留键 `__task_frame__`,见 §10),`TaskSpec.mid_run_key` 选缓冲键(如 `frame`)。
- **panel 永远是 plot 视图**(`frontend/task_console.py`):`PanelEditor` 管源参数、冻结快照、limits、command 与 save；`LogicNodeEditor` 管节点参数/生命周期。Selection 是 plot-independent value，`selection_action` 显式选择 none/fit/ROI。Setting 与 Edit 共享 `FitRequest` 和 `core.fitting`，并按 plot family 只提供兼容模型；`FitProcessor` 是 image fit 的 hub adapter，而不是第二套 solver。
- **plot 面板 Edit 的 Acquisition 区 = 数据源(logic 节点)自报参数**(解耦,#4):`PanelEditor` 经 `console._producing_node(card)` 找到产这张图所读信号的 logic 节点,用 `console._node_params(node)`(节点 `acquisition_parameters()` 自报)列出可编辑源参数(相机 = `exposure`/`frames_per_cycle`/`region`),每格带 `now:` 当前值,Apply 经 `_restart_node` 把改动**原地**下发(`camera.configure` 实时重配,不 start——节点从它自己的 Logic-tab Edit start)。plot 面板**绝不**自带某 measurement 的"参数表单 + Start"(`PanelEditor.meas_panel` 恒 None);要重跑一个扫描就去 Logic tab 调那节点。Setting 弹窗只留 source/size/colormap/relim/unit/title/actions。
- **信号来源(谁产的)**:每面板标题由 console 从 Hub provider 计算，形如 `<kind> — <signal> ← <producer>`（例如 `2D — frame_0 ← Camera`）；不靠 Python 类名或手写映射。
- **`@task` 注册表(对称 `@processor`)**:`TaskSpec`(`name`/`build(hub)`→Task/`params`(ParamDecl 列)/`mid_run_key`/`default_kind`/`prefix`)+ `@task`/`register_task`/`discovered_task_specs` + 内置 `@task calibrate_readout`(build 走 `readout.calibrate_task(hub, prefix="cal_", **values)`)+ `readout.task_specs()`。console 的 `tasks=` 喂 kind_combo 列 `Task: <name>`;选中 Add Panel → `_add_logic_node(LogicNodeConfig(kind="task", name=...))` 加一个**停着的 Logic 行**(与 measurement/processor 同路),Start 时 `_build_logic_node` 走 `spec.build(self.hub, **values)`。task 参数 + Run 走 Task 自己的 `acquisition_parameters`(`CalibrateReadoutTask` 含它 + `mid_run=("frame","progress")`);**task 不向 hub publish**——中途帧在 `node.output` 缓冲,由 console 固定面板显示(§10、#6)。kind_combo 严格四类:`Plot: <kind>` / `Measurement: Camera (live frames)`(`readout.camera_measurement`→`CameraMeasurement`)+ `Measurement: <扫描名>` / `Processor: <名>`(如反应式 `Judge occupancy`)/ `Task: <名>`——无自造 "Readout: Loading" 复合体。
