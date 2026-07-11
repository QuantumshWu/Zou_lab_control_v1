# Zou_lab_control 最终系统架构设计

## 1. 文档定位

本文定义 Zou_lab_control 的最终目标架构、运行语义、数据与线程约束、包所有权、实现顺序和验收标准。实现工作应以本文为架构基线；手册负责用户教学，测试负责机械强制，代码中的公开类型负责表达运行时合同。

目标不是把当前类机械搬进新目录，而是建立以下不变量：

1. data kernel、pulse/FPGA、neutral_atom、frontend 和 composition app 的职责与依赖方向唯一。
2. Task、Measurement、StreamProcessor、Analysis 的语义清晰且不能互相冒充。
3. 多维数据、validity、axis、lineage 在采集、处理、拟合、显示和保存过程中不丢失。
4. 正式采集、实时监视、控制状态和运行事件使用不同通信语义。
5. 正式 PulseScan 从触发源到最终 y 都是端到端 exact pipeline。
6. Qt、阻塞硬件 I/O、数值计算和 render 各有唯一线程 owner。
7. calibration 是内建领域能力，不使用 plugin 或动态发现。
8. 通用数据语义/fit 与呈现/交互分层：前者由 headless data kernel 维护一套，后者由 frontend 维护一套。
9. FPGA 的 build、target ABI、host、wire、RTL 和约束均可校验且 fail closed。
10. 最终运行时不保留 alias、fallback、legacy reader、双 registry 或平行实现。
11. 精密 pulse/trigger 时序始终由现有 FPGA/qCMOS 等硬件执行；软件不得用 host sleep 调度微秒/纳秒事件。与此同时，bitstream/RTL 是冻结部署资产：架构不得为了获得更强证明而要求重烧；只有 E0/真机证据发现现有 RTL bug、与既定设计不符或在已批准工作区间确实无法正确运行时，才单独评估硬件修复。

本文同时区分三种东西：终态不变量、当前冻结硬件上的 baseline capability、以及有明确删除点的迁移脚手架。正常 PulseScan baseline 是现有 bitstream 的无缝 `AUTONOMOUS_STREAMED`；唯一允许的非无缝路径是 API-slot 无法在运行中无缝更新时已经存在并被接受的 segmented 路径。`HOST_STEPPED_GROUP`/逐 cell fire-and-wait 不属于设计。逐沿 stamp、额外 ROM attestation、trigger-return 或新 watchdog 等需要重烧的增强只可在真机证据触发后作为独立硬件修复/升级提案，不能成为软件架构的前置要求。

这里的“硬件时序优先”不是把领域 key、工作流或新观测电路塞进 FPGA。当前 baseline 使用现有硬件已经提供的 pulse table执行、completion/progress/build fingerprint回读，以及 qCMOS 外触发、frame counter/stamp/timestamp；neutral runtime 用冻结计划、E0证明的相机确定性属性、preflight时序余量和整 run末端对账映射 TriggerKey/ScanCellKey。它明确弱于逐沿硬件证明，但在 PI 接受的风险边界内 fail closed：任何可见不一致使整个 run INVALID并重跑，不能提交部分或猜测性结果。

## 2. 用户体验兼容目标

架构重构主要发生在内部。对实验用户，日常工作流应保持熟悉：

- 仍通过 `task_console.bat`、`pulse_gui.bat` 和 notebook 启动系统；
- PulseGUI 仍保留 Edit、Preview、Scan 三个主要工作区；
- TaskConsole 仍使用 Add Panel、Setting、Edit、Start、Stop、Save/Load；
- Measurement、Processor、Task 仍可从 catalog 选择并连接；其中 UI 中的 `Processor` 明确对应在线 `StreamProcessor`，完整数据集上的 Fit/Calibration/Report 显示为 `Analysis`；
- live image、rolling plot、histogram、site map、fit overlay 的视觉语言保持一致；
- virtual 与 real 仍只替换最低层设备 adapter；
- pulse prepare/fire/safe、camera arm/read/disarm 的物理顺序不改变。

以下是有意的用户可见变化：

- Fit 成为明确可见的 `Add Analysis -> Fit` 和 Plot card `Analyze -> Fit` 操作；
- 非标量数据会自动得到一个可见、可改的默认视图；当选择或降维要进入 fit、scan y 或派生 artifact 等权威结果时，必须冻结为 CommittedTransform，不再暗中取第 0 项；
- history gap、schema change、缺帧和硬件 mismatch 会明确失败，不再显示拼接结果；
- qCMOS只有在E0证明目标工作点的确定性外触发/按序交付envelope、preflight trigger余量通过且整run EndAttestation一致时才可产出Formal ScanArtifact；否则只能monitor或整run INVALID重跑；
- qCMOS 正式扫描一次 arm并由 FPGA 无缝执行完整冻结 scan table；仅 API-slot 无法无缝更新时沿用既有 segmented 路径，SCAN_SLOT/MOT 不允许退化为逐 cell host stepping；
- Stop 后若设备尚未确认退出，UI 显示 `CANCELLING`，不会提前宣称已停止；
- 设备冲突会提示并等待原 owner 真正退出，而不是静默抢占；
- 保存文件只接受当前 artifact schema；历史数据通过独立离线转换工具处理；
- monitor 可能跳过中间帧以保持实时，但一次画面中的相关信号不会来自不同 shot；
- 重型 grid/多 panel board 由 worker raster 后整板 coherent present，视觉与交互保持，但内部不再让 GUI/worker 无确认共享 Figure；
- Pulse prepare/upload、长 fit 和 calibration 不再阻塞 GUI thread。

因此，用户看到的主要入口、操作结构和视觉风格基本一致；变化集中在功能可发现性、错误提示和安全状态。内部实现则会彻底重构。

## 3. 当前实现的根本问题

### 3.1 包边界相互反向依赖

frontend 当前直接导入 neutral_atom 的 fit、selection、pulse 和 Signal 类型；neutral_atom 又反向导入 frontend；FPGA 的部分生成、server 和 replay 路径依赖 neutral_atom。结果是：

- frontend 不能独立使用；
- FPGA 不能作为完整 headless 产品安装；
- GUI 知道设备与 runtime 私有对象；
- lazy import 隐藏了循环但没有消除循环；
- 公共 API 被大规模 re-export 扩张。

### 3.2 `neutral_atom/core` 没有单一语义

`core` 同时包含 runtime stream、fit、selector、facet、raster、calibration、readout、result、parameter 和 utils。它不是稳定领域层，而是历史依赖汇集点。目标架构中不保留该 namespace。

### 3.3 Task、Measurement、Processor 相互冒充

当前存在以下混合：

- Processor 可以自行订阅 Hub、取 latest、访问 camera/sequencer；
- one-shot Processor 可以占用并驱动两个硬件设备；
- Task 同时承担设备编排、采集、算法、artifact、report 和 UI mid-run buffer；
- Measurement 有时包含 plot 或领域 reduction；
- 通用 LogicNode 基类同时维护线程、设备声明、schema、provenance、参数队列和发布。

这使层名不能可靠说明对象能做什么。

更严重的是当前 `LogicNode.stop()` 在 join timeout 后仍把 `_thread=None`，TaskConsole 随即从 running set 移除并允许冲突节点启动；旧线程可能仍阻塞在硬件调用里，软件却已释放所有权。OneShot 又用同一个 stop event 表示“用户取消”和“finally 已结束”，后台异常可能被外层当正常 stop 吞掉、`last_error=None`。因此 lifecycle truth/ResourceClaim 必须先于普通类拆分建立，不能靠给现有 stop 再加一个 timeout 修补。

### 3.4 多维 shape 保留了，逻辑 axis 仍会丢失

物理数据已经接近正确形式：

```text
(R, P, *data_shape)
```

但 fit、ROI、facet、measurement result 和 plot 仍存在：

- `reshape(-1)`；
- 对其余轴自动 `nanmean`；
- 对 `data_shape=(N,)` 取 `[:,0]`；
- 根据 singleton/rank 猜测 curve/image/hist；
- 改写 point shape 而不记录映射。

此外，目标 DataBlock 还必须显式定义未采点的 validity，不能依赖 NaN 或 0。

validity 不能只停在 `(R,P)`：readout fidelity 已经产生 `(group,site)` 的真实 per-site valid，若通用合同只能表达整 cell 有效，dead/ambiguous site 会在后续 reduce、fit 或 histogram 中被静默算入。

这里还存在一个容易走向另一个极端的问题：禁止隐式降维，不等于要求用户每次打开图都手工选择所有 axis。显示层需要安全的自动视图；真正缺失的是“临时显示建议”和“权威分析变换”之间的类型边界。

把显示得到的完整 concrete transform 复制给 Fit/Scan，会让 `repeat=mean/32` 等 display-only policy 静默升级成物理分析输入。正确边界不是再加确认框，而是让 display step 在类型上不可提交，权威 draft 从 FitSpec/ScanOutputContract 重新派生。

### 3.5 Signal 的正式采集与 monitor 语义仍未完全分开

当前正式 PulseScan 已具备 fire 前 reservation、cursor、共同 lineage 和 gap-fatal，这是必须保留的正确机制。

仍存在的问题：

- TaskConsole gap 后逐 signal 回退 latest，可能混合不同 shot；
- 普通 expression 可独立读取多个 latest；
- fit 结果拆成多个 scalar signal 后由 UI 重组；
- control object 被编码为 numeric tensor；
- TaskOutput 构成第二套 mutable 数据通道；
- final y 的 reservation 不能自动保证上游 camera -> processor 每条边都 exact。

当前 finite CameraMeasurement 每来一帧重新发布累计 `(1..K)` repeat block，TensorStore full publish 又把整个 repeat-capacity current state复制进 journal，OccupancyProcessor 每次从头遍历所有已累计 R/P；实测 journal payload 随 K² 增长。Camera pending queue 还用 list `pop(0)`。根因是 sample event、mutable materializer 与 immutable dataset 被同一个“signal tensor update”冒充；目标实现必须一次只处理一个 sample、builder 私有增量写、UI 按 revision 请求 snapshot，而不是只把 history size 调小。

多 signal `next_coherent_update` 当前会寻找下一个共同 provenance 并推进掉更快流的 unmatched 更新；这对 coherent monitor 可以接受，却不能自动代表 formal EXACT_KEY。JoinPolicy 必须在 pipeline 合同中显式区分“允许跳过并计数”与“缺 key 立即失败”。

真 qCMOS 的 capture `nFrameCount` 是产出帧累计数；仓库 DCAM wrapper 还暴露 framestamp/camerastamp/timestamp，但当前 adapter 丢弃这些 metadata，也没有证明外触发工作区间内“一触发一帧、按序、无漏帧”。若只循环 `read_frames(1)` 并在 host 缓冲中取 latest，中间帧仍会被软件丢弃。近期修复不改变冻结 bitstream：E0 用真实相机和目标 exposure/ROI/readout/触发间距证明确定性外触发 envelope；preflight 用编译后 trigger schedule 与该 envelope 的最小帧间隔+安全余量拒绝过快 scan；运行时一次 arm完整帧预算、顺序保存全部 metadata，整 run结束后用现有 FPGA completion/progress回读、冻结计划的 trigger总数、相机产出数与 frame/camera stamp/timestamp连续性对账。任一不符使整个 run INVALID并重跑。该合同不能像逐沿硬件 tag那样定位具体错误，也不能绝对排除漏一帧同时多一帧的等量抵消；这是冻结硬件约束下明确接受的剩余风险，不能在文档中伪装成同等证明强度。

### 3.6 UI 与 render 所有权不清

TaskConsole、PulseGUI 和 DataFigure 同时承担 view、runtime、设备控制、文件 I/O、analysis、artifact 和线程管理。

当前 RenderLoop 允许 worker 操作与 Qt canvas/selector 共享的 Matplotlib Figure；barrier 或 join 超时后，调用方仍可能继续访问或销毁资源。它确实把既有复合板 compose 从 GUI 热路径移开，不能简单禁掉后把所有 grid/multi-panel 退回 GUI compose；正确处理是先以 fail-closed serialized handoff 隔离旧壳，再迁到 worker-owned board raster/front-buffer。PulseGUI 又在 GUI thread 同步执行 prepare/fire/safe。两者都不能原样成为最终线程模型。

只在“GUI-owned interactive Figure”和“纯 headless export”二选一也不够：高帧率 qCMOS、grid 和多 panel board 需要 worker raster 的吞吐，同时需要 GUI overlay ROI/selector 的交互和 board-coherent present。若没有独立的 worker-owned live raster + immutable front buffer + coordinate transform + Qt overlay 模式，最终要么重新阻塞 GUI compose，要么丢掉选择器/同 shot coherence。

### 3.7 Calibration 的领域模型不完整

一次 readout calibration 实际上从同一批 frames 产生多个 readout model，之后 Occupancy 再选择模型。把 Box、PerSitePsf、UniformPsf 设计成三个互斥 whole artifact 会复制 FrameContract、site map 和 lineage，也不符合真实使用方式。

### 3.8 FPGA 单一来源不完整

当前部分 geometry 可生成，但 clock、UART、slot width、pin mapping、timing constraint、tool/IP 配置和 digest 仍有多处来源；UART oversized frame 与 FIFO overflow 也缺少完整的 RTL fail-closed 行为。

streamed scan 的 inactive bank 目前依赖host写data -> chunk id -> ready的既有协议，没有RTL复算seq/count/CRC；geometry fingerprint也不能区分相同ABI下不同implementation seed/timing结果。这些是冻结bitstream的已知证明边界，不是架构自动要求新RTL的理由。近期baseline用host/model/golden、现有readback、发布资产映射和真机故障注入守住；只有观察到静默bank损坏/回放或证实部署bitstream身份错误等真实问题，才按RTL bug/设计偏离流程评估硬件完整性或attestation修复。

### 3.9 测试保护实现结构多于稳定行为

大量测试直接访问私有属性、类所在模块、继承树或生产源码字符串。它们会阻止正确重构，却不能证明用户能从真实入口完成 fit、calibration、PulseScan 和 shutdown。

### 3.10 notebook-first 的最常用面没有被列为交付物

现有用户依赖 `exp.readout.sitemap(...)`、`detect(...)`、`temperature(...)` 这类短路径。若目标架构只列 create_experiment、RunPlan、PipelineResult 和 DataFigure，却没有明确由谁把它们组合成薄 Experiment 门面，正确的内部边界会以 notebook ceremony 为代价；同时 neutral_atom 不能导入 render，DataFigure 的合法归属也会悬空。门面与 render ownership 必须在 core 迁移前定下，并用少量语句的真实 notebook E2E 守住。

## 4. 最终顶层边界

系统包含四个内聚 bounded context、一个窄基础设施模块和两个 composition 应用面。这里先定义 Python import 与所有权边界，不把“必须拆成几个 wheel/repository”写进架构：desktop 可以捆绑安装；FPGA server 可只安装 pulse+storage；headless experiment 可安装 data+pulse+neutral+storage。

```text
zlc_data           通用 Value/DataBlock、axis/validity、Selection 语义、transform/reduction/fit
zlc_pulse          Pulse 文档、FPGA target/compiler、host、transport、RTL 和 build
zlc_neutral_atom   实验 runtime、设备 port、Task/Measurement/StreamProcessor/领域 Analysis
zlc_frontend       View/Figure/DataFigure、render、selector controller 和 Qt 组件
zlc_storage        canonical bytes/digest + content-addressed blob 与 atomic manifest 存储引擎
zlc_workbench      桌面应用与唯一 Qt composition root
Zou_lab_control.notebook  notebook composition 与薄 Experiment 门面
```

`zlc_data` 不是新的 `common/utils`：它只容纳领域中立、headless、可序列化的数据语义和值上的纯算法。它拥有 Value/DataBlock、Axis/Validity/PointLayout、Selection/CommittedTransform、Reduction、FitSpec/FitProblem/FitResultBatch、model/solver adapter 与 `fit_analysis`。它不知道 Hub、Run、Device、neutral artifact、Figure、Qt 或 Matplotlib。

headless fit 保存的 typed identity 也由数据语义 owner 定义：`zlc_data.FitResultArtifactRef` 与 FitResultBatch canonical codec 属于 zlc_data；实际 Repository I/O 由 notebook/neutral/workbench composition adapter 委托 zlc_storage 完成。`fit.save()` 保存 FitResultBatch 与 input/FitSpec lineage，不要求 frontend，也不把通用 fit 伪装成 neutral 领域 Analysis。Figure 保存仍使用 frontend-owned FigureArtifactRef；二者是不同 artifact kind。

“selector”必须拆开看：`Selection` 是可保存、可供 fit/processor 共同消费的数据语义，属于 zlc_data；鼠标手势、RectangleSelector、handle、overlay 和 interaction state 属于 frontend。`DataFigure` 明确属于 frontend，因为它是 render/public presentation facade。fit editor/overlay 属于 frontend，但它们调用 zlc_data 的唯一 fit 实现，不复制模型和结果 schema。

zlc_data 用 `bind_fit(FitSpec, expected DatasetSchema) -> BoundFit` 冻结并验证 fit/batch axes、CommittedTransform、model 与数值策略，但不捕获尚未产生的数据。`BoundFit.run(frozen DataBlock) -> FitResultBatch` 是 formal、interactive 与 offline 三条路径共享的唯一执行值；neutral runtime 只把 `BoundFit + DatasetInputSlot` 放进无 Fit 语义的通用 `AnalysisStep`。baseline 不再建立 `FitAnalysisDescriptor -> DataAnalysisProgram` 两层通用分析框架；出现第二个确实需要相同跨 context bind/replay 语义的领域中立 Analysis 后再提取。neutral 不得定义 `FitProcessor`、`FitOperator` 或 neutral-owned `FitAnalysisDefinition`，Workbench 只把 zlc_data 的 FitSpec/editor capability 投影进 `Add Analysis`。

`zlc_pulse` 是一个逻辑 bounded context，而不是“为了目录好看必须独立发布的产品”。它内部包含 `model`（PulseDocument/IR）与当前唯一生产 target `fpga`（TargetSpec/compiler/wire/host/RTL/build）。FPGA server、sim/build 和 neutral sequencer adapter 已是独立消费者，所以禁止它反向 import neutral；若未来出现第二硬件 target，再在 pulse 内抽出 target Protocol，baseline 不预建插件系统。

`zlc_storage` 只拥有两类窄基础设施：其一是无领域类型的 canonical primitive encoding/digest（canonical map/list/scalar、ndarray header/bytes、hash 与 framing）；其二是 bytes/blob/manifest 的校验、fsync、原子发布和最小维护。它不定义 universal ArtifactRef、领域 schema 或 artifact kind。frontend、pulse、neutral_atom 与 data 各自拥有 typed Ref/值对象 schema 和 `to_canonical_tree`，但最终 bytes/digest 必须委托同一个 canonical encoder；跨包嵌值对象必须调用 owner codec，不能手写字段顺序。这样避免四份 canonical JSON/float/ndarray/digest 实现，又不建立能收容领域类型的 `common` 包。baseline 只实现经过 probe 的 local filesystem commit；复杂 GC、多后端/分布式锁等出现真实第二用例后再扩展。

### 4.1 依赖方向

```text
zlc_data -> zlc_storage.canonical（只允许纯 canonical 模块，不允许 repository/I/O）

zlc_neutral_atom -> zlc_data
zlc_neutral_atom -> zlc_pulse public API
zlc_neutral_atom -> zlc_storage

zlc_frontend -> zlc_data
zlc_frontend -> zlc_storage
zlc_pulse -> zlc_storage

zlc_workbench -> zlc_frontend
zlc_workbench -> zlc_data
zlc_workbench -> zlc_pulse
zlc_workbench -> zlc_neutral_atom

Zou_lab_control.notebook -> zlc_data
Zou_lab_control.notebook -> zlc_neutral_atom
Zou_lab_control.notebook -> zlc_pulse public API
Zou_lab_control.notebook -> zlc_frontend.figure
Zou_lab_control.notebook[render] -> zlc_frontend.render (optional)
Zou_lab_control.notebook[workbench] -> zlc_workbench (optional GUI launcher)
```

箭头表示依赖。禁止：

- zlc_data 导入 frontend、neutral_atom、pulse、storage repository/backend 或 workbench；它只可导入 zlc_storage.canonical 纯模块；
- frontend 导入 neutral_atom、pulse 或 workbench；
- pulse 导入 data、frontend、neutral_atom 或 workbench；
- neutral_atom domain/runtime 导入 frontend 或 workbench；
- composition root 以外实例化 concrete adapters。

### 4.2 Composition roots

允许三个明确装配入口：

- `zlc_neutral_atom.bootstrap.create_domain_experiment`：headless 领域实验；
- `zlc_pulse.fpga.server.bootstrap`：可独立部署的 FPGA server；
- `zlc_workbench.composition.create_workbench`：desktop/Qt 应用。

另有一个薄的公开 notebook composition module `Zou_lab_control.notebook`：它依赖上述公开 API，提供 `connect -> Experiment`，但不成为领域包之间的新依赖层。顶层 `Zou_lab_control.connect` 可以密封地转发这一入口；禁止把所有 bounded context 的 symbol 做 umbrella re-export。公开名称只使用 `Zou_lab_control.notebook`；若仓库保留 `apps/zlc_notebook` launcher，它只能是导入该 module 的应用入口，不能形成第二套 facade/API 名称。

可复用 library 内禁止通过 FQCN、包扫描或 service locator 动态构造依赖。

DeviceSet 是 composition root 拥有的 concrete connection/lane lifecycle container，不是运行时可随处查询的 service locator。composition root 同时提供 typed `DeviceBindingResolver`，把 request 中的 `DeviceBinding(role/id, required capability)` 原子解析为：

```text
BoundDevice[P]:
  port: P
  resource_key
  thread_affinity_key
  capability_snapshot
  connection_generation
```

Definition.bind 只能通过 resolver 一次取得自己显式声明的 BoundDevice，并把该对象放进 BoundDependencies；Port、claim、affinity、capability 和 generation 不能作为五份平行字段分别传递。execute 不能按字符串回查 DeviceSet，resolver 也不能从全局 registry 隐式挑“第一个相机”。

### 4.3 Data 与 Frontend 内部层次

```text
zlc_data
   ^
   |
frontend.figure <- frontend.render <- frontend.qt
```

所有权：

- `zlc_data`：Axis、Value/DataBlock、Selection、DataTransform、Reduction、Fit；
- `frontend.figure`：ViewIntent、ViewSpec、FigureDocument、FigureEvaluator、FigureArtifactRef、codec；
- `render`：renderer、Matplotlib layer、interaction controller、DataFigure public facade；
- `qt`：Canvas、Qt event adapter、通用 widgets。

neutral_atom 只依赖 `zlc_data`，不依赖 frontend 的任何层。

`zlc_data` base 只依赖 NumPy/必要 solver与 `zlc_storage.canonical`，不加载 repository I/O、Matplotlib/PyQt。`zlc_frontend` 的 headless figure 层依赖 data+storage；Matplotlib backend 放在 `[render]` optional extra，PyQt/Fluent 及 Qt canvas 放在 `[qt]` extra（`qt` 可依赖 `render`）。`zlc_neutral_atom` 依赖 data、storage 与必要的 pulse public API；notebook 的显示 extra 依赖 `zlc_frontend[render]`，其可选 `[workbench]` extra 才懒加载 `zlc_workbench` GUI launcher，workbench 显式依赖 `zlc_frontend[qt,render]`。模块顶层 import 不能让 data/neutral 间接加载 repository backend、Matplotlib backend 或 PyQt。

`zlc_pulse` 的 model/compiler/host/server base import 不探测 Vivado、不导入 Qt；build/sim 工具只在明确命令入口检查外部工具。neutral_atom 导入 Sequencer adapter 所需 public API 时不能触发 build environment 初始化。

#### zlc_data 准入规则

一个类型/函数只有同时满足下列条件才进入 zlc_data：

1. 输入输出只由 zlc_data 值对象、NumPy 数组和标量组成；
2. 不知道实验设备、shot/Hub/Run、neutral artifact、panel/plot kind 或用户 session；
3. 同一语义确实被至少两个上层 context 消费，或它是 DataBlock 正确性不可分割的不变量；
4. 可用纯函数/冻结 spec 表达，并有任意 axis/validity 的 property contract。

因此通用 curve/image fit framework、明确数学模型、batch result 属于 zlc_data；readout threshold decision、PSF calibration、occupancy/fidelity 等带中性原子物理语义的模型属于 neutral Analysis/StreamProcessor，即使内部复用 zlc_data 的数值求解 primitive。领域 model id/quality gate 由 neutral codec 保存，不能塞进 zlc_data 的 built-in fit catalog。

`Selection` 只保存 AxisId、typed geometry/range/index、CoordinateFrameId 与必要的稳定坐标参数；不允许 arbitrary metadata dict、widget scope path、plot-kind binding、ControlTopic payload 或 JSON byte-packing。frontend 的 facet/cell scope、hover/drag state 和 neutral 的 control binding 是各自 adapter state，通过显式映射引用同一个 Selection 值。

例如 ROI processor 使用自己声明的 `RoiBinding(selection, source_axis_ids, coordinate_transform, reducer)`，Figure facet 使用 `PanelSelectionBinding(selection, layer_id, facet_coordinate)`；二者共享 Selection，但不会把 processor reducer 或 panel cell path 写回 Selection。这样保存/复用选择不需要 frontend 与 neutral 约定一个隐藏 metadata vocabulary。

`BoundFit` 是由 FitSpec 与预期 DatasetSchema 确定性绑定的进程内执行值，不进入 artifact、pickle 或 FQCN import。持久化只保存 FitSpec/CommittedTransform、model/algorithm version、numeric policy 与 input lineage；重放时调用当前 zlc_data `bind_fit` 并验证版本/digest。zlc_data 不提供可变全局 model/analysis registry，也不扫描 entry point。

### 4.4 Pulse Preview 边界

Pulse bounded context 不导入 frontend。Preview 路径是：

```text
zlc_pulse.PulseDocument / TargetIR
  -> zlc_workbench.pulse.PulsePreviewProjector
  -> zlc_frontend.figure.FigureDocument
  -> zlc_frontend renderer
```

frontend codec 不认识 PulseDocument。若 workspace 需要可编辑 replay，由 workbench 保存 PulseDocumentRef 与 FigureArtifactRef 的关联。

### 4.5 Notebook-first Experiment 门面

notebook 不是“绕过正式架构的调试入口”，而是一等 composition root。`zlc_neutral_atom.bootstrap` 只装配 headless domain/runtime；`Zou_lab_control.notebook` 提供薄 `Experiment` 门面，把 domain experiment、repositories、RunController 与可选 renderer 显式组合：

```text
Experiment                         # notebook/application facade，不含领域算法
  .readout / .timing / .pulse      # 语义子门面
  .readout.current_calibration_ref(binding?)
                                      # 按 ReadoutBindingKey 保存的可见默认 Ref
  .run(request_or_plan)
  .figure_document(result_or_ref)   # headless projector
  .figure(result_or_ref)            # 需要 zlc_frontend[render]
  .task_console() / .pulse_gui()    # 懒加载 notebook[workbench]，否则入口不存在/给安装提示

neutral domain Result
  typed values/artifact refs
  no FigureDocument/DataFigure/Qt/Matplotlib object
```

`Experiment` 只做参数便利、typed request 构造、结果解包和可选 adapter 调用；它不复制 calibration/fit/scan 算法，也不让 domain object lazy 回查全局 session。唯一允许的便利状态是 `Experiment.readout` 内用户可见、可读取/赋值/清空的 `ReadoutBindingKey -> CalibrationArtifactRef` 映射：ref 只是 immutable 指针，不是 calibration 数据或第二份权威状态。单一默认 readout 时 `.current_calibration_ref` 可保持短属性；存在多个 camera/readout binding 时必须通过 `for_binding(binding)` 或显式 `calibration=` 选择，不能把 camera A 的 ref 猜给 camera B。`sitemap()` 成功可按公开契约只更新本次 binding 的默认 ref；任何依赖 calibration 的 convenience request 在**构造 request 时**把该 ref 解析为显式字段并立即冻结，运行时不再回查 facade。结果、RunPlan 与 artifact lineage 始终记录实际使用的 binding/ref/digest，因此短 notebook 路径不以隐藏物理输入换便利。

`figure_document` 由 notebook composition 的 result projector 生成，避免 neutral_atom 反向依赖 frontend.figure。render extra 不可用时，采集与分析仍完整工作，FigureDocument 可保存或交给其它 renderer。GUI launcher 不是 headless notebook baseline；只有安装 `notebook[workbench]` 时 `.task_console()`/`.pulse_gui()` 才通过单向 optional edge 调用 workbench composition，workbench 不反向 import notebook facade。

日常路径必须保持短而诚实，例如：

```python
exp = zlc.connect("virtual", repository=repo)
cal = exp.readout.sitemap(frames=12)
# 单 readout 时 exp.readout.current_calibration_ref == cal.ref；多 readout 按 binding 保存
scan = exp.readout.detection_time(times).run()
fit = scan.fit(model="decay")
fit_ref = fit.save()  # headless zlc_data FitResultArtifactRef，不要求 renderer
```

同一组代码换成 real adapter 只改 `connect` 参数。契约 E2E 固定“connect virtual -> capture -> 1D fit -> save”和“sitemap -> calibration ref -> detect”均为少量 notebook 语句；不得要求用户手工 bind Port、构造 PipelineSpec、解析 PipelineResult 或 resolve artifact ref。门面在最早 vertical slice 与 RunController/Repository 一起交付，不能拖到剩余 helper 收尾阶段。

### 4.6 顶层运行模型：四个平面、三个边界

最终架构不是一个所有节点都传同一种“大数据对象”的通用 DAG，而是四个语义平面：

```text
外部世界
  -> Measurement
  -> [sample/event plane: Envelope<Value | typed domain record>]
  -> StreamProcessor（可选、逐 event 或明确 key group）
  -> DatasetBuilder
  -> [dataset plane: immutable DataBlock revision / typed ArtifactRef]
  -> Analysis（zlc_data 通用分析或 neutral 领域分析）
  -> [result plane: typed result / immutable artifact]

任一冻结 dataset/result
  -> frontend ViewSpec/FigureDocument/DataFigure
  -> [presentation plane: 可丢弃、可重算、不可反向成为权威输入]
```

三个边界各自只有一个 owner：

1. `Measurement -> event` 由 acquisition runtime 赋 envelope、key、generation 和 provenance；设备 adapter 不发布 Hub event。
2. `event -> dataset` 只由 `DatasetBuilder` 完成；它用显式 repeat/point key 写入 `(R,P,*data_shape)`，因此 StreamProcessor 永远不会把累计 DataBlock 当“最新 signal”。
3. `dataset/result -> presentation` 只由 frontend 完成；ViewSpec 是可重算显示意图，zlc_data FitSpec/CommittedTransform 与 neutral 领域 AnalysisSpec 是独立权威意图，二者没有可隐式升级的继承关系。

这四个平面不是四套框架：它们共享 zlc_data 的 Value/DataBlock/axis/validity 值对象和统一 lineage，但不共享生命周期、背压或失败语义。sample plane 处理实时性与 exact reservation；dataset plane 处理完整性与 revision；result plane 处理算法合同与 artifact commit；presentation plane 只优化交互延迟。这样既避免“所有东西都是 Processor”，也避免为了统一表面形式引入递归工作流引擎。

### 4.7 顶层边界的对抗结论

| 候选 | 否决/采纳原因 |
|---|---|
| fit/DataBlock/Selection 全放 frontend | 否决。neutral headless 必须依赖一个 presentation context，且 UI policy 容易泄漏进权威分析；当前 frontend 反向导入问题也会换方向重现。 |
| 通用 fit/Selection 全放 neutral_atom | 否决。DataFigure/notebook 为复用通用算法必须依赖中性原子领域；calibration/readout 与普通数学 fit 再次混进同一个 `core`。 |
| 为 fit、axis、selection 各拆独立包 | 否决。三者共同维护同一个 DataBlock/validity/transform 不变量，过细拆分会制造 codec、version 和 import ceremony。 |
| 小型 zlc_data kernel | 采纳。它有 frontend 与 neutral 两个真实消费者，内部围绕“具名多维数据上的纯、可复现操作”高度内聚，并能用准入规则防止变成 common。 |
| FitAnalysisDescriptor/DataAnalysisProgram 通用层 | baseline 否决。当前 Fit 只需 FitSpec -> BoundFit；FitResultBatch 因 gridplot/per-site 是真实用例而保留。第二个通用 Analysis 出现后再提取共同 descriptor。 |
| 把 pulse/FPGA 全并入 neutral | 否决逻辑合并。FPGA server、sim/build、GUI preview projector 与 neutral adapter 是不同消费者，反向 import 已证明会污染边界。 |
| pulse 逻辑边界 = 必须独立 wheel | 否决。所有权/DAG 与发布单元是不同决策；先保持稳定 namespace，按部署证据决定是否拆 wheel。 |
| 抽通用 zlc_runtime | 暂不采纳。当前只有 neutral_atom 一个真实 runtime consumer；先留在 neutral，出现第二个生产领域且合同确实相同后再提取。 |
| 窄 zlc_storage | 采纳。frontend Figure、neutral capture/scan、pulse compiled artifact 都需要同一 crash-safe blob/manifest 机制；四个 owner 又必须共享 canonical primitive bytes/digest。storage 仍不拥有任何领域 schema/Ref，canonical 子模块无 I/O。 |
| 四包各自重写 canonical JSON/digest | 否决。owner 只拥有 domain object -> canonical tree；primitive tree -> bytes/digest 由 zlc_storage.canonical 单源并从第一天做 cross-package vectors。 |

这个划分的优化目标不是“包数最少”或“每个名词一个包”，而是让变化原因相同的代码在一起、让不同生命周期/依赖方向的代码无法互相偷用。namespace 数量由这些边界决定，wheel 数量由部署决定。

## 5. 约束如何进入代码

不同约束使用不同机制，不建立万能基类。

| 约束 | 机制 |
|---|---|
| 数据不变量 | frozen value object + constructor validator |
| 设备替换边界 | consumer-owned Protocol |
| Task/Measurement/StreamProcessor/领域 Analysis 元数据 | frozen Definition；通用数据 Analysis 使用 zlc_data-owned descriptor/spec |
| 线程和生命周期 | final concrete runner/state machine |
| 资源排他 | ResourceClaim + ResourceArbiter |
| axis 变换 | typed transform value + validator |
| 包依赖 | architecture tests |
| FPGA 事实 | authoring spec + generated output + resolved manifest |
| adapter 一致性 | parameterized contract kit |

stream generation、block revision、config/control revision、connection generation、document/panel revision 使用不同的强类型值，不共用裸 int alias。每种 revision 只由唯一 owner 单调推进；跨域对应关系记录在 Envelope/ViewModel/EvaluatedFigureData 中，禁止拿不同 domain 做大小比较或互相替代。

### 5.1 Base class 使用条件

只有真实 is-a 关系和不可分割不变量才建立 base class。例如 CameraDevice 可以拥有 arm/read/disarm 与 acquisition-lock 模板，但不能同时拥有 GUI、Hub、artifact 和 task 状态。

禁止：

- 空 hook；
- 大量 optional methods；
- `supports_x` flag 伪装多个 capability；
- protected fields 形成隐式 API；
- 跨包继承 concrete implementation；
- 为复用几行代码建立继承树。

### 5.2 Protocol 使用条件

Protocol 只用于真实替换边界：

- CameraPort；
- SequencerPort；
- PulseTransport；
- typed Repository；
- Renderer；
- 确有多个 solver 实现时的 Solver。

单实现纯算法使用函数或 final concrete implementation，不提前创建 Protocol。

### 5.3 抽象引入门槛

baseline 只实现当前有生产消费者且会守住正确性边界的机制：

| 能力 | baseline | 引入更重抽象的证据门槛 |
|---|---|---|
| fit batch | FitResultBatch 一等支持 | gridplot/site/component fit 已是当前消费者 |
| 单位 | canonical id + 封闭 conversion table | 第二个必须组合量纲/自动推导的生产用例 |
| 坐标 | opaque frame id + 显式 transform | 第二个需要通用 frame graph/复合变换的生产用例 |
| exact | 有限 Run + reservation | 必须连续不可丢、且不能自然切成有限 Run 的生产用例 |
| snapshot ownership | 默认 owned immutable snapshot | profiler 证明 copy 是瓶颈且 adapter 能安全 pin buffer |
| plan compile | 同步纯函数；宿主可投递普通 CPU worker | profiler 证明需要专用 scheduling/QoS，而非仅后台执行 |
| execution | 扁平 RunPlan + 同步状态机 | 本文不预留 child plan/递归 workflow slot |

“未来可能需要”不能单独成为新 Protocol、Service、状态或序列化类型的理由。新抽象必须同时给出第二用例、被消除的重复/风险、生命周期 owner 和 contract test；否则使用现有值对象、函数与 composition boundary。

## 6. 多维数据合同

### 6.1 ValueSchema 与 DatasetSchema

流中的一次事件和收集后的完整数据集使用不同合同：

```text
ValueSchema:
  data_axes: tuple[AxisSpec, ...]
  validity_contract: VALUE | COMPONENTS(axis_ids)
  dtype
  value_unit: canonical unit id | None

DatasetSchema:
  repeat_axis: AxisSpec
  point_axes: tuple[AxisSpec, ...]
  point_layout: PointLayout
  cell_schema: ValueSchema
```

`ValueSchema` 描述一次 Measurement/StreamProcessor event 携带的值，例如一帧 `(H,W)` image、一个 `(site,)` occupancy vector 或一个真正标量。它没有伪造的 R/P leading axes。`DatasetSchema` 描述 DatasetBuilder 把带 key 的事件放入哪些 repeat/point cell 后形成的完整数据集。DataBlock 永远符合 DatasetSchema，AcquisitionStream/MonitorStream 的普通 event 永远不携带累计 DataBlock 或 DataPatch。

一个 domain event 可以是 frozen typed record，例如 `CameraSample(image: Value, frame_metadata)` 或 `OccupancySample(occupied: Value, counts: Value, source_metadata)`；它仍作为一个 Envelope payload 原子发布。record 中每个数值字段使用 zlc_data 的 Value/ValueSchema，record 类型和领域 metadata 由 producer package 拥有。

`AxisSpec` 包含：

- stable AxisId；
- name、role；
- size、coordinates；
- unit: canonical unit id 或 `None`、coordinate_frame: CoordinateFrameId。

AxisId 由 producer Definition 的稳定字段语义派生，在相同 semantic axis 的不同 run/adapter 间保持一致；不能每次构造随机 UUID，也不能用可修改 display name 或 tuple position 充当 identity。Select/Transpose/Rename 保留来源 AxisId；Create/Stack/Reduction 的新 AxisId 由 operation id、输入 AxisIds 与 transform digest 确定性派生，并在 TransformRecord 中记录来源。

单位采用 zlc_data 单源维护的 canonical id（例如 `Hz`、`MHz`、`s`、`count`），display label 与物理单位分开。baseline 不建立量纲代数、单位表达式 AST 或自动推导系统；兼容性只允许“相同 canonical id”或显式列入封闭 `UnitConversionTable` 的转换。未知单位作为 opaque id round-trip，但不能自动换算。CoordinateFrameId 同样是 stable opaque id，只做等值检查；跨 frame 必须提供显式、版本化 CoordinateTransform 及其参数/lineage，不能因 shape、名字或数值范围相似而默认兼容。

UnitConversion/CoordinateTransform 都是 serializable value，只引用 zlc_data 的封闭实现；不携带 callable/FQCN。需要相机畸变标定、物理模型等领域映射时使用带 CalibrationArtifactRef 的 neutral StreamProcessor 或 Analysis，不能把领域 calibration 藏进通用坐标转换。出现第二个确需组合量纲或坐标代数的生产用例后，才评估引入更强模型。

`role: AxisRoleId` 是 producer 声明的 stable、可序列化语义，例如 repeat、scan-point、spatial-x、spatial-y、spectral、site 或 component。built-in role 由 zlc_data 单源定义；领域扩展使用 namespaced id，不注册可变全局对象。不认识的 role 仍能 round-trip，但默认 preserve/select。role 不能从 rank、长度、singleton 或数值内容反推。

ValidityContract 是 ValueSchema 的一部分并进入 fingerprint：VALUE 表示整个 event value 同生同灭；COMPONENTS(axis_ids) 声明 mask 可细化到哪些具名 data axes。producer 不能首帧发 VALUE validity、遇到坏 site 后再未经 generation migration 改成未声明的 component mask。StreamProcessorDefinition 在 bind/preflight 中根据输入 validity contract 声明输出 contract 与传播规则，无法证明时不能进入 formal exact pipeline。

Data schema 不枚举“当前软件允许哪些 projection/reducer”。数据身份与已安装分析功能必须解耦：

- Select/Transpose/Stack 等结构变换由 zlc_data 的显式 DataTransformSpec 定义；
- display-only mean/latest/sample policy 由 frontend.figure 的 ViewContract 定义；
- 权威通用 mean/sum 使用 zlc_data ReductionSpec，用户/AnalysisSpec 必须显式选择并记录 unit/validity rule；
- ROI photon count、occupancy、calibration 等领域 reduction 是 neutral_atom StreamProcessor/Analysis，不伪装成 ValueSchema/DatasetSchema 的内建能力。

因此新增一个 renderer 或 reducer 不改变 ValueSchema/DatasetSchema fingerprint，也不触发无意义的 stream generation migration。

shape 是派生值，不保存第二份可变真相：

```text
logical_point_shape = tuple(axis.size for axis in point_axes)
data_shape          = tuple(axis.size for axis in cell_schema.data_axes)
P                   = point_layout.storage_size
```

### 6.2 PointLayout

`PointLayout` 是 frozen value，不是策略 registry：

```text
PointLayout:
  logical_shape
  mode = RECT_C | RECT_F | EXPLICIT
  storage_size
  storage_to_multi: optional immutable tuple[tuple[int, ...], ...]
```

RECT_C/RECT_F 要求 `storage_size == product(logical_shape)`，映射由 order 唯一派生。EXPLICIT 要求 `storage_size == len(storage_to_multi)`，每个 multi-index 维数正确、范围内且唯一；它可以只覆盖逻辑笛卡尔空间的稀疏子集。serpentine、非规则轨迹或自定义扫描携带显式映射；若多个物理采样落在同一个逻辑 point，必须增加独立 sample/repeat axis，不能在映射表中重复 index 后静默覆盖。

没有 point axis 时 logical_shape=()、storage_size=P=1。将 EXPLICIT 数据 densify 到 logical_shape 必须同时产生 validity/mapping；算法若只需采集顺序就沿 P 和 PointLayout 工作，不能假设 P 等于逻辑尺寸乘积。

### 6.3 DataBlock 与 validity

```text
Value:
  values: ndarray         # (*data_shape)，标量为 shape ()
  validity: ValueValidity
  schema: ValueSchema

DataBlock:
  block_id
  revision
  values: ndarray        # (R, P, *data_shape)
  validity: DatasetValidity
  schema: DatasetSchema

DatasetRevisionRef:
  block_id
  stream_generation
  schema_fingerprint
  revision

OwnedSnapshot:
  ref: DatasetRevisionRef
  values/validity owned or backed by immutable chunks
```

```text
ValueValidity =
    Valid | Invalid
  | ComponentValidity(axis_ids, mask)

DatasetValidity =
    CellValidity(mask: bool array)       # shape (R, P)，整 cell 同生同灭
  | ComponentValidity(
        axis_ids,
        mask: bool array,                # (R,P, *declared component axes)
        broadcast_contract
    )
```

Value 是 stream event 内的 zlc_data 数值值对象；Envelope 的 key/provenance、CameraSample 等领域 record 不属于 Value。CellValidity 表示 Dataset 中一个完整 trailing value 是否已经采集；ComponentValidity 表示同一 value 内不同 site、pixel、bin 或其它 component 可以独立无效。ComponentValidity 的 `axis_ids` 必须是 ValueSchema.data_axes 的有序子集，mask 只能按这些具名 axis 广播；禁止依靠 ndarray 尾部对齐猜语义。这样：

- uint16 image 不必转 float；
- 未采点不被误认为 0；
- partial scan 保持固定 shape；
- fidelity 的 `(group,site)`、dead site、坏 pixel/bin 不会被整 cell 的 valid 掩盖；
- fit/reduce/histogram/meter/image alpha 都消费同一 validity，而不是各自用 `isfinite` 猜。

默认使用紧凑 CellValidity；只有 producer/processor 确实产生 component 级缺失时才使用 ComponentValidity。实现可用只读 broadcast view、packed bitmap 或按 chunk 存储，不能强迫所有完整 image 复制同尺寸 boolean mask；但优化不能改变具名 axis 语义。

Value.validity 与 DataBlock.validity 必须符合 cell_schema.validity_contract；COMPONENTS 合同仍允许用整体 Valid/Invalid 或 CellValidity 表示“本 revision 所有 component 同生同灭”的紧凑特例，但一旦提供 ComponentValidity，其 axis_ids 只能是合同声明集合的子集。VALUE 合同绝不接受 component mask。Select/Transpose/Stack/Reduce 必须同时派生新的 validity_contract，不能只变 values/schema axes 而忘记 mask 语义。

ReductionSpec 必须声明 `validity_policy`（例如 `ALL_REQUIRED`、`ANY_VALID`、`MIN_COUNT(n)` 或所选 reducer 合同自己的规则）。reducer 只在 mask 为真的 component 上运算，并产生新的具名 validity；不能把 NaN 当通用 validity，也不能用 `nanmean` 在未声明策略时悄悄吞掉坏 site。FitProblem 逐 batch cell 过滤无效 observation，并记录有效样本数；不足模型最小点数时只使该 batch result 失败。Histogram 丢弃无效 sample 但记录 dropped count；Meter 在目标 component 无效时显示 invalid，不回退其它 component。

发布后的 DataBlock 是 immutable materialized dataset snapshot：不仅 consumer 不能写，**snapshot 的 bytes 在其整个可见生命周期内也不得因 DatasetBuilder 后续 ingest 而变化**。累计采集由单 owner 的 `DatasetBuilder` 持有不外泄的 mutable preallocated/chunked storage；它根据 typed key + DatasetLayout 定位 cell并原子更新 values+validity+journal。**每个 sample 只产生轻量 `DatasetProgress(block_id, revision, dirty_cells, coverage)`，不得发布/复制完整累计 DataBlock。** Live binding 按 UI 刷新率和明确 `DatasetRevisionRef + SliceSpec` 请求 snapshot；baseline 只能返回 owned materialized slice，或由 immutable sealed chunk/copy-on-write 支撑的 snapshot。禁止把 mutable preallocated ndarray 的 view 仅设 `writeable=False` 后冒充 immutable revision；该 flag 只约束 consumer，不能阻止 builder owner 改写别名。最终 EOS 可把完整 buffer ownership一次转移为冻结 DataBlock 或流式 artifact。无论选择何种实现，单点 ingest、journal 与 progress 必须摊销 O(1)，总数据复制近似 O(final bytes)，不能把现有 SignalHub 的 O(P²D) 搬进 DatasetBuilder。

`SnapshotStore.materialize(ref, SliceSpec) -> OwnedSnapshot` 必须精确解析请求的 revision；FINITE_EXACT revision 在 Run 保留期内必须可解析，ROLLING_MONITOR 的旧 revision 被覆盖时返回 `SnapshotExpired`。它绝不能在请求旧 revision 时悄悄返回 current/latest。Fit、Save、BoardFrame 和 artifact lineage 都记录完整 DatasetRevisionRef，而不是裸 revision 整数。

Measurement/StreamProcessor 不创建 DataBlock/DataPatch，也不读“当前累计 block”来决定下一条输出。它们只处理当前 Envelope payload 或声明的完整 key group。DatasetBuilder 是 sample stream -> dataset 的唯一边界；Fit/Calibration 等 Analysis 消费其冻结的 DataBlock snapshot 或 artifact。DatasetProgress 是状态通知，不是数据输入，consumer 不能从它重建权威值；snapshot store/Repository 才拥有 revision -> bytes 的解析。

所有 AxisId 在一个 DatasetSchema 内唯一；coordinates 长度与 size 相等；`repeat_axis.size == R`；`values.shape[1] == point_layout.storage_size`；cell_schema.data_axes 顺序与 trailing ndarray 顺序完全一致。任何 public consumer 若要从 P 恢复多维 point index，必须调用 PointLayout，不能自行 `reshape` 猜 order。

### 6.4 标量

标量唯一表示：

```text
ValueSchema.data_axes == ()
data_shape == ()
Value.values.shape == ()
values.shape == (R, P)
```

`data_shape == (1,)` 是长度一数据轴，仍需显式 Select 或 Reduce。

### 6.5 DataPatch

```text
DataPatch:
  block_id
  base_revision
  result_revision
  target_cells
  values
  validity_patch
  schema_fingerprint
```

values 与 cell/component validity 原子更新。Patch 只能应用到匹配 block_id/base_revision/schema_fingerprint 的 DatasetBuilder，不能改变 schema 或 validity axis contract；result_revision 必须严格递增。target_cells 在 patch 内唯一，shape/dtype/mask axes 必须与 schema 一致。runtime 另在 Envelope/cursor 层校验 stream_generation，zlc_data DataPatch 不引用 stream lifecycle。正式 exact capture 重复写已 valid component 是 duplicate fatal；只有明确声明 replace semantics 的非权威 monitor builder 才能覆盖。

DataPatch 只在 DatasetBuilder 内部、snapshot store 与持久化 materializer 的受控边界传播，不是 sample stream/StreamProcessor edge 或普通 UI queue 的 payload。Live binding 正常只接 DatasetProgress，再向 snapshot store 请求自己需要的 revision/slice；不得每个 event 把 full DataPatch 或 full snapshot fan-out 给所有 panel。Journal 保存 delta，不能每个点复制整个累计 block，否则 scan 会退化为 O(P²D)。Snapshot 的 revision 与 journal commit 点一致：consumer 要么看到 patch 前完整状态，要么看到 patch 后完整状态，不能看到 values 已更新但 validity 尚未更新的中间态。

### 6.6 Axis transform

允许：

```text
Select(axis_ids, index/range/geometry Selection)
Reduce(input_axis_ids, ReductionSpec)
Transpose(axis_ids)
Stack(axis_ids, new_axis_spec, reversible_mapping)
Unstack(axis_id, original_axis_specs, reversible_mapping)
TransformCoordinates(axis_ids, CoordinateTransform)
ConvertUnit(target_axis_or_values, UnitConversion)
Create(axis_spec)
Rename(axis_id, name)
```

每次变换同时返回 TransformedData 与 TransformRecord，并验证 values、validity、AxisId、coordinates 和 mapping。

Rename 只改 display name，不改变 role/unit/frame；unit conversion 与 frame transform 使用各自显式 operation，不能通过改 metadata 假装数值已转换。

不提供匿名 `flatten()`。`Stack` 必须给出新 AxisSpec、来源 AxisId 和可逆坐标映射；因此它不能把 `(repeat, point, *data_axes)` 偷换成三个无语义长度，也不能被用作“先摊平再猜”。DataBlock 的物理 P 维始终由 PointLayout 映射回完整 point_axes。

## 7. Stream、buffer 与一致性

### 7.1 Envelope

```text
Envelope[T]:
  event_id
  stream_id
  stream_generation
  sequence
  emitted_at
  join_key: optional typed key
  join_key_schema_fingerprint: optional
  trace
  payload

TraceContext:
  run_id
  source_id
  correlation_id
  causation_refs
  config_revision
  control_revision

CausationRef =
  EventRef(stream_id, generation, sequence, event_id)
  | EventSpanRef(stream_id, generation, [start, end), count, ordered_digest)
  | ArtifactInputRef(typed_ref, content_digest)
```

数值/领域数据 Envelope 额外包含 payload contract fingerprint 与 captured timestamp。普通 stream payload 是 Value 或包含 Value 字段的 frozen domain record；DataBlock/DataPatch 只属于 DatasetBuilder/materialization 边界，不能作为“当前累计 signal”反复发布。Provenance 是 causation graph、payload fingerprint 和 TransformRecords 的派生视图，不是另一套含义模糊字段。

JoinKey 是 frozen、可哈希、可序列化的领域值（例如 TriggerKey/ScanCellKey/ShotKey），不是字符串拼接或 payload 私有字段。`EXACT_KEY`/`LATEST_COMPLETE_KEY_MONITOR` 只接受相同 key type + schema fingerprint；key 缺失或类型不同在配置/preflight 阶段失败。TraceContext.correlation_id 只用于追踪，不能代替数据关联 key。

`sequence` 在 `(stream_id, generation)` 内从 0 严格单调且不复用；event_id 在一次创建后不可改变。StreamProcessor 输出创建新 event_id，不能沿用某个输入 id 冒充同一事件。少量 join 使用 EventRef；StreamReducer/DatasetBuilder 的长连续输入使用 EventSpanRef，ordered_digest 覆盖按 sequence 排列的 event_id/payload digest。禁止在每个累计结果里复制全部历史 event_id，避免 provenance 退化为 O(N²)。

### 7.2 四种通信原语

| 原语 | 语义 | 用途 |
|---|---|---|
| AcquisitionStream | ordered、exact、cursor、gap-fatal | 正式 scan/capture |
| MonitorStream | bounded、latest/coherent-latest、missed count | live UI |
| ControlTopic[T] | typed、revisioned、ack | ROI、threshold、run command |
| EventStream[T] | progress/transition notification | UI/headless status |

Artifact Repository 是持久化原语，不是 stream。

ControlTopic 的 ack 明确区分 `ACCEPTED`、`APPLIED(at transaction boundary)`、`REJECTED(reason)`、`SUPERSEDED(by_revision)` 与 `TERMINATED(reason)`；发送成功不等于硬件已经应用。每个被 ACCEPTED 的 revision 最终必须恰好收到 APPLIED、SUPERSEDED、REJECTED 或 TERMINATED 之一，UI 不会永久等待一个被 coalesce 或 owner shutdown 吞掉的 command。有限正式 Run 拒绝 reconfigure 时必须返回 REJECTED，UI 不能先改成本地“已生效”状态。

EventStream 中 progress 可以 coalesce，transition 必须按 run revision 有序；但通知流不是状态真相源。RunHandle 保存可查询的最新 authoritative state/error/phase snapshot，UI 初次连接、漏事件或重连后先读取 snapshot，再订阅后续事件，因此 terminal event 即使 UI 当时阻塞也不会“丢掉终态”。

### 7.3 Exact reservation

Reservation 是对以下区间未 ack 数据的 retention pin：

```text
(stream_id, generation, [start_sequence, end_sequence))
```

状态：

```text
PLANNED -> RESERVED -> ACTIVE -> DRAINING -> COMPLETED -> RELEASED
                         |
                         -> ABORTING -> FAILED/CANCELLED -> RELEASED
```

必须满足：

- fire 前以 payload 最大尺寸上界原子检查 event budget、byte budget 与实际 retention backend capacity；
- cursor 是 opaque subscription state；
- cursor 不跨 generation；
- 下游成功原子发布后才 ack 输入；
- 多 reservation 由最老未 ack watermark 决定 retention；
- exact path 不自动 retry hardware run；
- 无 reservation 的慢消费者获得 typed Gap，不回 latest。

正式 exact run 必须是有限 Run，并有确定的最大 end_sequence；不能用“预计平均帧大小”冒充 byte 上界。但完整性总量与同时 retention 预算是两个不同合同：

```text
total_expected/max_events        # EOS/coverage 完整性
max_inflight_events/bytes        # 未 ack retention admission
max_source_burst + backpressure capability
```

RunController 根据 source 最大 burst/速率、最慢 required consumer、ack 边界和是否可 backpressure，证明一个保守最大 backlog；reservation 只 pin 未 ack 区间，ack 后立即释放。不能因为 run 有 N 点就无条件在 RAM 保留 N 个大帧，也不能在不可 backpressure 的相机上假设 consumer 平均够快。若无法证明 max inflight，必须为最坏 total 分配、选择流式 artifact sink/更慢触发，或 preflight 拒绝。连续 Measurement 在 baseline 中只使用 MonitorStream，不建立 infinite reservation、continuous-exact epoch 或 durable spool。未来只有出现必须连续、不可丢且无法切成普通有限 Run 的第二个生产用例，才单独设计持久 spool/epoch 协议；不能先让所有运行背负该状态机。

stream_generation/payload contract fingerprint 改变时，旧 exact cursor 终止为 typed SchemaChanged。schema-affecting reconfigure 不是“原地改参数”，而是 generation migration：owner 在 transaction boundary 终止旧 generation、对所有 pending Control revision 发 terminal ack，为每个绑定的 DatasetBuilder 创建新 block_id/DatasetSchema/generation，再允许 Monitor 显式 rebind。旧 pending view/fit 结果 stale，CommittedTransform 因 DatasetSchema fingerprint 改变一律失效，不能按 index 偷迁移。稳定 AxisId 只帮助迁移 workspace preference 的候选匹配，仍须完整 schema/coordinate/validity 校验。正式 finite Run 默认拒绝 schema-affecting reconfigure；value-only 且 schema 不变的参数才可按运行合同在边界 APPLIED。

一个 StreamProcessor invocation 只原子发布一个 typed payload；同 shot 多字段装进同一 frozen record，成功 enqueue 后才 ack 输入。baseline 不支持把一次 invocation 拆成多个 exact stream 再实现跨 stream transaction；确有不同 cardinality/key 的结果应拆成独立节点。DatasetBuilder 在 Value 已按 key 原子写入 values+validity+journal 后 ack；Repository sink 的 ack 点是临时 blob fsync/校验完成且 manifest 原子提交之后，不是刚开始写文件。

### 7.4 JoinPolicy

当前实际需要四种：

```text
EXACT_KEY
ZIP_SEQUENCE
LATEST_COMPLETE_KEY_MONITOR
INDEPENDENT_LATEST_MONITOR
```

`LATEST_COMPLETE_KEY_MONITOR` 允许 UI 跳过旧 shot，但一次显示的相关信号必须属于同一个 key；不完整旧 key 被淘汰并计入 missed/incomplete count。它不能用于 fit、scan、calibration 或 artifact。

`EXACT_KEY` 是跨设备/跨 worker 的正式关联方式。`ZIP_SEQUENCE` 只允许用于同一已验证软件 producer 拆出的等长 ordered streams，且合同能证明 sequence 一一对应；不能拿两个独立设备的“第 N 条”推断同一 shot。`INDEPENDENT_LATEST_MONITOR` 只用于互不声称相关的独立 panel，不能用于多输入 expression 或同一 coherent view。

暂不引入 WINDOW/ClockTransform；出现真实 use case 后再设计。

### 7.5 Exact 与 monitor fan-out

一个 physical CaptureSession 是 camera 的唯一 owner。它产生一次 AcquiredSample，broker 分发到：

- exact subscribers：受 reservation/backpressure 保护；
- monitor tap：bounded overwrite，记录 missed，不反压 exact。

二者共享 event_id/trace，但拥有不同 QoS 与 buffer。

broker 只分发 immutable payload/ref；driver 会复用 DMA/frame buffer 时，在第一次发布前复制或转移 ownership。exact retention、monitor current ref 和每个 consumer 各自持有明确 lifetime；monitor overwrite 只能释放自己的引用，不能使 exact consumer 看到被复用的内存。

### 7.6 Buffer sizing

Measurement output contract 声明 `max_payload_bytes`、`max_burst_events`、finite run 的 expected/max total events、生产速率/停顿边界与是否能 backpressure；StreamProcessorDefinition 声明 cardinality、record/Value 输出尺寸上界、最大并发处理与 ack 点；DatasetBuilder/sink 声明 ingest backlog、chunk/flush 策略和最终 DataBlock bytes。RunController 从这些合同派生每条 edge 的 `max_inflight_events/bytes`，而不是把 total events 直接当 history depth。它们是 preflight 预算输入，不是节点自行分配私有 list 的权限。

Monitor subscriber 可以请求小容量（例如 image latest/history=8），其合同就是允许 overwrite 并报告 missed。正式采集不能靠“把 history 设成 full size”保证完整：RunController 必须沿整个 source -> processor -> sink exact chain 汇总每条 edge 的 event/byte budget，并在 fire 前对共享 retention backend 原子 reservation。一个节点的 full-size buffer 不能替代上游/下游 edge reservation，也不能保证变长 payload、多 subscriber 或 processor fan-out 不溢出。

### 7.7 Finite dataset 与 rolling monitor

同一个 DatasetBuilder/materializer contract 明确区分两种模式，不允许 implementation 自行发明无界 live buffer：

```text
FINITE_EXACT
  固定/有界 key coverage，duplicate fatal，EOS 冻结
  revision 在 retention 合同内可重取，可进入 formal Analysis/artifact

ROLLING_MONITOR
  固定 byte/cell capacity，overwrite + missed/expired count
  每个可见 revision 仍是内容稳定的 OwnedSnapshot
  默认不可成为完整 CaptureArtifact 或 Formal Analysis 输入
```

交互 Fit/Save 若要使用 rolling 数据，必须先冻结一个仍可解析的 DatasetRevisionRef/窗口为新的 owned finite input并记录 missed/coverage；不能把“当前 rolling window”冒充从运行开始至今的完整 dataset。monitor progress、dirty cells 与 pending snapshot request 都必须有界/coalesce，Python 引用不能绕过 retention budget让旧大帧无限存活。

## 8. 同步执行与线程托管

核心 runtime 不使用 asyncio。执行原则是：

> synchronous execution semantics, threaded hosting, cooperative cancellation。

### 8.1 线程拓扑

```text
GUI thread
  Qt QObject/Widget/QTimer
  interactive Figure/Canvas/selector

RunController
  所有用户可启动 Run 的 lifecycle
  start() 立即返回 RunHandle
  run-owner 执行 bind 后的 preflight/cleanup，不直接跨 affinity 调 driver

Blocking I/O lane[ThreadAffinityKey]
  每个有线程亲和性的 device/session 串行

StreamProcessorWorker
  ordered reactive transform

Analysis executor
  bounded dataset/artifact analysis
  formal/offline 与 interactive QoS 分队列

View-evaluation executor
  per-panel latest-only/coalescing
  纯 display transform/reduction/FigureEvaluator

Headless render worker
  永久拥有独立 export Figure
```

不使用“每种职责固定一个全局 OS thread”。连续 camera monitor 不能阻塞无冲突设备；同一 thread-affine device 的调用必须串行。

独立 panel latest-only 是逻辑 mailbox（每 panel 最多一个 pending revision）；声明为同一 coherence group 的 panel 使用一个 board mailbox/evaluation revision，不各自挑 latest。它们由少量 bounded workers 消费，不是每个 panel 新建线程。worker 数和队列预算来自 WorkbenchProfile；满载时只替换尚未开始的旧 view/board work。Analysis executor 区分 formal/offline 与 interactive QoS：interactive 同 panel 新 revision 可替换尚未开始的旧 fit；formal/offline/明确保存的 Analysis 不 coalesce，满载时返回 typed Busy 或在 Run deadline 内排队。正式 StreamProcessor event 绝不进入可丢弃 view/interactive 队列。

### 8.2 RunController 与 RunHandle

```text
RunController.run(plan)   -> Result      # 同步，notebook/test
RunController.start(plan) -> RunHandle   # 后台，workbench
```

`RunController` 是所有用户可启动 Run 的唯一 lifecycle owner，包括 one-shot Task、finite/continuous Measurement、FormalPulseScan、DatasetBuilders、StreamProcessorWorkers 和 formal Analyses。每次 `start()` 创建一个 run-owner thread；terminal state 只能在所有 I/O call、CaptureSession、online worker、materializer 和 required Analysis 确认退出后产生。

每种 Definition 只有一个与其语义一致的绑定结果：

```text
TaskDefinition.bind(request, immutable bindings) -> RunPlan[Result]
MeasurementDefinition.bind(request, immutable bindings) -> BoundMeasurement
StreamProcessorDefinition.bind(config) -> BoundStreamProcessor
DomainAnalysisDefinition.bind(config, typed input slots/refs) -> AnalysisStep
zlc_data.bind_fit(FitSpec, expected DatasetSchema) -> BoundFit
neutral.bind_analysis(BoundFit, DatasetInputSlot) -> AnalysisStep
```

Measurement、StreamProcessor 与 AnalysisStep 都不是独立 lifecycle owner，不直接返回 RunPlan；它们由静态 PipelineSpec 编译进一个顶层 RunPlan。用户“单独 Start Measurement”也编译成一个 source + DatasetBuilder/明确 sink 的最小 PipelineSpec，而不是特殊启动路径。这样不会为了统一方法签名而让它们冒充 Task。

bind/compile_pipeline 是无硬件 I/O 的确定性构造，可做 request validation、纯 pulse compile 和静态预算。Notebook 可以在调用线程直接构造；Workbench 把同一个同步函数投递给其普通 application worker，结果再交给 RunController.start。runtime 不定义专用 command/build lane、第二套队列协议或额外 Service；若 profiling 证明某个编译器本身很重，只把该纯函数放入现有 bounded CPU worker，不改变领域合同。RunController.start 只接收已经构造好的 RunPlan，因此不会持有 ResourceClaims 等待纯编译。

`RunPlan` 是扁平静态计划：

```text
RunPlan:
  request
  bound_dependencies
  typed request/output contract
  static ResourceClaims
  preflight(ctx, request, bound_dependencies) -> PreparedRun
  execute(ctx, prepared_run) -> Result
  mode = FINITE_EXACT | CONTINUOUS_MONITOR
  expected/max events/samples/grouping
  deadline: optional monotonic deadline
```

FINITE_EXACT 必须给出可验证 max budget/deadline policy；CONTINUOUS_MONITOR 明确允许 overwrite/missed count，不产生宣称完整的正式 artifact。所有 timeout/deadline 使用 monotonic clock，artifact timestamp 另用 wall clock。

`PreparedRun` 显式携带 request、与 claims 对应的 BoundDependencies、resolved schemas、reservations、cursors 和其它 preflight 结果。`execute` 只能收到该 PreparedRun，不能通过 closure、session、global registry 或 service locator 找回未声明 Port。不包含 child run、递归 DAG 或运行中新增资源。

BoundDependencies 只含 consumer-owned Port/factory、typed Repository 和 immutable config，不含 QWidget、open CaptureSession 或任意线程外可直接调用的 raw driver。Port 调用由 RunController 路由到 owner lane；PreparedRun 中的 session token/handle 也只能由该 lane 消费。

bind 必须从 request/bindings 计算完整或保守 superset ResourceClaims。preflight 可以拒绝 claim 与硬件 capability 不匹配，却不能发现后临时追加资源；若某 adapter 的条件资源无法在 bind 时确定，Definition 必须声明 superset 或拒绝该 request。

真实硬件使用两阶段启动，但仍是单层计划：

```text
bind -> RunPlan
-> acquire_all static claims
-> 在正确 I/O lane 执行 open/configure/query preflight
-> resolve ValueSchema/DatasetSchema、event/sample/byte budget
-> PreparedRun(reservations, cursors, resolved contracts)
-> arm/sources ready
-> fire/execute
```

preflight 或 reservation 失败时不得 arm/fire，并释放已创建 reservation。CaptureSession 固定拥有 disarm；长期 device connection 的 close 属于 DeviceSet/application lifecycle，只有 CaptureSession 自己创建临时 handle 时才负责 close。

device/session 的 create/open/configure/read/disarm/close 必须在其 ThreadAffinityKey 对应 lane 执行；composition root 只能在外部构造不接触 driver 的轻量 adapter/factory。CaptureSession 在 owner lane 创建并在同一 lane 销毁，不能在 run-owner thread 创建后交给 I/O lane 使用。

外部权威状态：

```text
RUNNING -> SUCCEEDED | FAILED
RUNNING -> CANCELLING -> CANCELLED | FAILED
```

waiting resource、arming、capturing、fitting、saving、finalizing 是 phase，不是通用工作流状态。

`run(plan)` 内部也使用同一个 RunHandle。Notebook/test 遇到 KeyboardInterrupt 时先 cancel 该 RunHandle、等待 cleanup acknowledgement，再重新抛出或返回取消结果。若等待超过 join deadline，抛出携带 run_id/RunHandle lookup 的 `RunStillCancelling`，RunController registry 继续持有 handle/claims；不能丢掉 handle 后把 cell 当成已经停止。notebook 可继续 `status()/wait()/recovery_instructions()`。

### 8.3 CancellationToken

- 每个 Run 新建；
- 单调从 active 变 cancelled；
- 绝不 clear/reuse；
- cancel requested 不等于 worker terminated；
- join timeout 后不得清 thread owner、释放资源或允许 restart；
- 每个阻塞 Port 必须有 bounded timeout 或 interrupt contract；
- cancellation 先置 token，再调用 Port 声明为 thread-safe 的 out-of-band `interrupt/abort`，随后由 owner lane 完成正常 cleanup；
- safety-critical Port 不能让 `safe_state` 永久排在可能无限阻塞的同一调用之后：必须有 transport timeout、独立 abort/safe channel 或硬件 watchdog 中至少一种可验证机制；
- 不可中断的 SciPy/NumPy 计算等待返回后丢弃 stale result；
- 需要 hard deadline 的计算使用 disposable subprocess。

当前 RemoteSequencer 的单条同步 RPyC connection 会让 `wait_done` 占住同一请求通道，且 transport backstop 可长达 3600 s；软件重构必须先缩短/分解阻塞调用，使用现有transport支持的timeout、cancel/abort/safe并在故障注入中测量最坏停止时间。第二条RPyC socket若共享backend/`_io_lock`只能改善RPC调度，不能宣称硬件独立；但baseline也不要求因此新增watchdog、SAFE寄存器或重烧bitstream。无法确认safe时Run保持CANCELLING/FAILED且resource quarantine。只有真机证据表明现有safe路径违反既定安全要求，才按bug修复流程评估硬件改变。

### 8.4 Cleanup

优先使用同步 context manager 和 `try/finally`。安全关键 abort 的顺序先消除物理危险，再清理软件对象：

```text
cancel intent
sequencer out-of-band abort/safe -> no-more-trigger acknowledgement
CaptureSession cleanup: camera disarm
workers/builders abort or drain + join
DeviceSet/application shutdown: camera close
temporary handle: only its creating CaptureSession closes it
sequencer safe_state
temporary config restore
reservation release
resource release
```

业务错误保留为 primary error，cleanup/safety failure 作为附加错误。安全清理失败不能报告成功或普通取消。

一旦最后一个硬件 sample 已取得且不再需要设备，正常路径立即退出 CaptureSession、disarm/safe，再进行长时间 fit/calibration/artifact commit；`finally` 是异常兜底，不是把安全动作拖到所有磁盘/CPU 工作之后。硬件 safe acknowledgement 失败时不得提交宣称整个 Run 成功的最终 artifact。

只有 worker/session 已退出且 safe/disarm 得到肯定 acknowledgement，RunController 才释放对应 ResourceClaim。若 join 超时，Run 保持 CANCELLING 且 claim 仍由原 run 持有；若 worker 已退出但 safe/disarm 明确失败，Run 进入 FAILED，ResourceArbiter 将设备标记为 QUARANTINED。QUARANTINED 资源不能被 acquire，只有用户执行 adapter 声明的 recovery action、随后 identity/health/safe 验证通过并显式确认，才可解除；不能在 finally 中无条件 release。

ResourceArbiter 使用 machine/device-installation 级稳定安全目录中的 append-only `QuarantineJournal`，不能放在用户可切换的 artifact RepositoryRoot 中。journal 记录 stable DeviceIdentity、connection generation、run_id、prepared artifact/table digest、原因、required recovery、首次/最近时间和解除证明。每个 hazardous control-lease/run epoch 在**第一次 arm/fire 前**原子追加一次 `HAZARD_ACTIVE` write-ahead record，不为每个 trigger做磁盘fsync。只有整个hazard epoch通过现有safe/status回读、正常completion+保守drain合同或人工recovery验证后才追加`RESOLVED`。这样进程在fire后直接崩溃也会留下未解除事实，又不要求新硬件receipt。FPGA identity使用板卡DNA/serial（若现有接口可读）或稳定部署endpoint+人工资产映射，qCMOS使用model/serial，不能只用device_index或枚举顺序。

每个新 device connection generation 初始为 UNVERIFIED；若同一 DeviceIdentity 有未解除 journal entry，则初始为 QUARANTINED_PENDING_VERIFY，而不是因进程重启回到 AVAILABLE。adapter 完成 identity/layout/fatal-status/health/safe handshake 只能提供解除证明，不能自行删除记录；用户确认 recovery 后追加 RESOLVED。设备确实更换且 stable identity 不同时建立新资源记录，不继承另一个物理设备的 quarantine。journal 的写入失败本身使安全相关 resource 保持 unavailable。

现有硬件若暴露sticky fatal/status，它是最高优先级事实；没有该能力时QuarantineJournal覆盖进程崩溃、driver无持久fatal或远端socket重建等软件边界。诊断日志只用于观察，不能代替持久状态合同，也不能因此要求新增RTL状态位。

该 baseline 不是多装置工作流数据库：只需一个由 zlc_storage 原子追加的本地安全 ledger、按 stable DeviceIdentity 查询未解决记录、追加 RESOLVED，且不承担分布式协调/自动恢复/通用审计 UI。virtual adapter 和纯单元测试可用内存实现；任何能驱动真实安全关键输出的 adapter 必须使用持久实现。不能把它降成仅进程内状态，因为进程崩溃、RPyC断线和stop join timeout会让“硬件是否safe”跨重启未知；现有status/safe/reconnect handshake用于解除记录，不能让重启自动洗白。

### 8.5 Owner-thread command mailbox

长时间 Measurement 的参数修改通过 command mailbox 送到 owner thread，并只在 shot/capture transaction 边界应用。GUI 不跨线程直接 configure driver。有限正式 Run 默认拒绝运行中 reconfigure。每个 accepted control revision 遵守 §7.2 的 terminal ack 合同；同 key 的尚未应用 revision 可被较新 revision SUPERSEDED，但已经开始硬件 transaction 的 revision 不能假装被覆盖。

Blocking I/O lane 的普通 command queue 有界并使用公平调度：同一 owner 的连续 monitor read 以小 transaction 重新排队，不能永久压住已获资源的 finite run/control apply；不同 ResourceKey 使用 round-robin 或等价的 bounded-wait 规则。safety interrupt/abort 不进入普通公平队列，走 §8.3 的 out-of-band 通道。公平只决定已合法排队工作的顺序，不允许绕过 ResourceClaim 或并发调用 thread-affine driver。

## 9. ResourceArbiter

```text
ResourceClaim:
  ResourceKey
  mode = EXCLUSIVE | OBSERVE
```

Run 启动前一次解析全部 claims 并原子 acquire_all。运行中禁止新增 capability 或 lease。

ResourceArbiter 只证明同一 composition root 内的互斥。真实 adapter/server connection 还必须取得窄 `DeviceControlLease`，用 SDK exclusive-open、server-side owner token 或本机 interprocess lock 证明 notebook、standalone launcher、Workbench 和远端 client 之间的物理排他；无法证明时只能开放一个真实控制入口，不能把 EXCLUSIVE 描述成跨进程事实。lease丢失走现有safe/connection-recovery路径并quarantine，而不是让另一个进程静默接管；不因此要求新硬件watchdog。

claims 在 bind 时仍声明完整 superset，但 `PreparedRun` 可以静态标记每个 claim 的最后使用 phase。CaptureSession 已退出且 hardware safe ack 后，RunController 可提前释放 camera/sequencer claim再做长 fit/save；释放后后续 phase不再持有对应 Port capability，且禁止重取或重新 acquire。没有这条静态 phase 证明时，一律持有到 Run terminal。

ResourceArbiter 只返回：

```text
Acquired
ResourceBusy(conflicting_run)
ResourceQuarantined(reason, recovery_action)
```

它不自动停止其它 run。Workbench 可请求停止冲突 owner，但必须等待其 RunHandle 确认 termination 后再重试。

OBSERVE 应尽量使用独立只读 capability Protocol，而不是把同一个控制对象运行时阉割为 read-only wrapper。

OBSERVE 不等于允许第二个 session 并发读同一 driver。对 camera 等单 owner 设备，monitor 通过已有 CaptureSession 的 broker tap 观察 immutable samples；只有硬件/adapter 明确提供可并发只读 capability 时才创建 OBSERVE claim。没有该 capability 就与 EXCLUSIVE 冲突，不能用 wrapper 绕过。

同一 ResourceKey 上多个 OBSERVE 可共存；EXCLUSIVE 与任何其它 claim 冲突。ResourceKey 由 device owner 提供 canonical hierarchy，父资源的 EXCLUSIVE 与子资源 claim 冲突。`acquire_all` 对完整 claim set 一次判定并提交，不逐个等待，因此不依赖调用方排序规避死锁。

## 10. Task、Measurement、StreamProcessor 与 Analysis

### 10.1 Definition 原则

Definition 是 frozen metadata + callable，不需要每类再建立 Handler Protocol 和公共 ABC。

Definition 中 callable 只能是显式 top-level builder/operator 引用，不能 closure 捕获 Device、Session、Repository、GUI 或 mutable config；所有运行依赖必须出现在 request/bindings/BoundDependencies，所有可变参数必须进入 config revision。

只有会出现在 catalog/UI/API 的能力需要 Definition；Task 内部私有算法保持普通函数。

Definition 发现不依赖 global mutable registry、包扫描或 entry point：

```text
DefinitionKey:
  owner_package
  stable_definition_id
  schema_version

DefinitionCatalog:
  definitions: immutable tuple
```

`zlc_neutral_atom` 拥有 DefinitionKey/DefinitionCatalog，各领域模块显式导出 definitions tuple；composition root 通过普通 import 组装 catalog，重复 DefinitionKey 或同 id 冲突时启动失败。Workbench 用本地 adapter 将它们映射为 CatalogView；排序、分组、图标和可见性只存在于 CatalogView，不反向写进领域 Definition。zlc_data/zlc_pulse/frontend 不为了进入 UI catalog 而依赖 neutral_atom 的 Definition 类型，也不建立跨 bounded-context universal Definition base。

Catalog composition 对每个 DefinitionKey 必须产生一个显式 visible mapping 或 hidden reason；未处理 definition 使 architecture/E2E失败，避免新领域能力已经注册却在 UI 静默消失。迁移期 CatalogRouter 用同一规则保证一个 use case只有 legacy 或 new入口可见，不制造双启动按钮。

### 10.2 Task

```text
TaskDefinition[Request, Result]:
  stable DefinitionKey
  parameter/request schema
  bind(request, bindings) -> RunPlan[Result]
```

Task 是 one-shot use case，可以同步组合 CaptureSession、纯 operator 和 typed Repository。它不继承 Measurement/StreamProcessor/Analysis，不发布 measurement signal，不拥有 QWidget。

Task 不一定产生 artifact；普通控制/查询 Task 可返回 immutable result，需要持久化时返回本包 typed ref。

Task 的中途数值/图像显示不重新建立 `TaskOutput`。需要 live frame、3D map 或优化轨迹的 Task，必须在同一 RunPlan 中声明并复用正式 DatasetBuilder 或 ROLLING_MONITOR materializer；RunHandle 只暴露 typed `LiveDatasetSlot -> DatasetRevisionRef`，Workbench panel 通过 SnapshotStore 读取。阶段/progress/warning 仍走 EventStream。这样中途 UI、最终 Analysis 与 artifact 使用同一 builder/revision 真相源，Task 不发布第二份 mutable signal，也不会因删除 `__task_frame__` 丢失现有用户功能。

### 10.3 Measurement

```text
MeasurementDefinition:
  stable DefinitionKey
  request/capture plan builder
  required Device Ports
  output_payload_contract(request, DeviceCapabilitySnapshot)
    -> ValueSchema 或 frozen typed record schema
  delivery class
  expected events/samples/grouping
  bind(request, bindings) -> BoundMeasurement

BoundMeasurement:
  capture_spec
  bound Device Ports
  output/cardinality/budget contracts
  ResourceClaims
```

```python
capture_spec = build_capture_spec(request)  # 整段在 owner I/O lane 中执行
with capture_factory.open(ctx, capture_spec) as capture:
    sample = capture.read_next(timeout)
```

Measurement 从外部世界取数据，可以访问 Device Port；它不 fit、不渲染、不保存 Figure、不管理 Task terminal state。runtime 负责将 AcquiredSample 包装为 Envelope。

DeviceCapabilitySnapshot 是 connection generation health handshake 后得到的 immutable、versioned descriptor。bind/UI 用它纯解析 expected payload/ValueSchema，preflight 再读取硬件实际设置并要求 fingerprint 相符。formal run 不允许 fire 后才发现 shape/axis；无法预先确定 schema 的 adapter 只能提供 monitor 或先执行独立 probe/config Task 后重新 bind。

绑定阶段构造 immutable BoundMeasurement/CaptureSpec；Pipeline Run execute 在 owner I/O lane 打开 CaptureSession。Task 若需要同一种采集，复用 CaptureSpec builder/CaptureSession，而不是启动一个 child Measurement Run。

### 10.4 StreamProcessor

```text
StreamProcessorDefinition:
  stable DefinitionKey
  named input contracts
  JoinPolicy/QoS
  output_payload_contract(input contracts, config)
  cardinality/byte-bound contract
  axis/lineage/join-key transform
  pure operator
  bind(config) -> BoundStreamProcessor
```

```python
output = operator(joined_inputs, config)
```

StreamProcessor 只处理当前 Envelope payload 或声明的完整 key group，不访问设备、Hub 私有状态、latest、累计 DataBlock、Repository 或 QWidget，也不创建 Envelope。`StreamProcessorWorker` 负责 subscription、join、validation 和 publish。

每个 StreamProcessor invocation 原则上发布一个 frozen typed payload。多个同 shot 结果组成一个 record，例如 `OccupancySample(occupied, counts, source_metadata)`；UI/下游通过 field projection 读取字段，不把同一物理结果拆成多个需要分布式原子提交的 signal。只有字段具有不同 cardinality、key 或生命周期时才拆成独立节点。

operator 不读 wall clock、module global config 或 global RNG；需要随机算法时 seed/RNG algorithm 是 immutable config 与 lineage 的一部分。相同 input/config/model version 必须可重放，允许的浮点容差由 operator contract 声明。

StreamProcessorDefinition 必须声明 output join_key 是 pass-through、typed compose 还是 intentionally absent；`StreamProcessorWorker` 按声明生成/验证，operator 不能从 payload 猜 key。Formal exact pipeline 中在最后一次所需 EXACT_KEY join 之前不得丢弃 key。

cardinality contract 明确 `1:1`、固定/有界 fan-out、`group K:1` 或 intentional filter，并给出 EOS completeness 规则和 max output bytes。Pipeline preflight 据此计算预算；FormalPulseScan 通往 ScanCellKey y 的路径不得存在未在 ScanOutputContract 中解释的 filter/drop。

output payload schema 必须只依赖 input contracts/config，不能读第一帧后改变 axis 数量或 record fields。站点发现、模型选择等 data-dependent schema discovery 属于 finite Task/Analysis/artifact 构造；其结果若要进入 formal pipeline，先以 SiteMap/CalibrationArtifact 等 immutable input 固定 schema，再 bind RunPlan。

需要跨 event group 状态的在线算法使用明确 StreamReducer；普通 StreamProcessor 不带可选 start/update/finish/reset。对完整 scan/capture 数据集做 fit、calibration 或 report 的算法属于 Analysis，不使用 StreamReducer 模拟 batch analysis。

StreamReducerDefinition 必须提供 state factory 与 update/finalize contract；每个 `StreamProcessorWorker`/RunHandle 创建独立 state，不能复用 module singleton 或上一次 run 的缓存。state 的 schema/config revision 固定，EOS incomplete/cancel 时不得把 partial finalize 结果发布成成功输出。

当前任何会 camera grab 或 fire sequencer 的 one-shot Processor 都必须重新分类为 Task 或 Measurement。

### 10.5 Analysis

```text
AnalysisStep[Result]:                  # neutral runtime 的已绑定执行值，不是算法 owner
  typed input slots/refs + expected contracts
  frozen config/spec
  deterministic bound operation
  output contract
  resource/compute budget

DomainAnalysisDefinition[Config, Result]:
  stable DefinitionKey
  input contract: DataBlock snapshot | typed ArtifactRef(s)
  output result/artifact contract
  deterministic analysis function
  resource/compute budget
  bind(config) -> AnalysisStep

BoundFit:                              # zlc_data-owned，不含 runtime slot/ref
  frozen FitSpec + expected DatasetSchema fingerprint
  resolved fit/batch axes + model/numeric policy
  run(frozen DataBlock) -> Result
```

Analysis 不访问 Device、Hub/latest、QWidget 或 mutable DatasetBuilder。它消费冻结的 DataBlock revision 或 immutable artifact，产生 FitResultBatch、CalibrationArtifact、report 等 typed result。`FitSpec/BoundFit/fit_analysis/FitResultBatch` 全部由 zlc_data 拥有；Calibration/ReadoutFidelity 等带 neutral 物理语义的算法才使用 `DomainAnalysisDefinition`。Workbench 的 `Add Analysis` 只把明确的 zlc_data Fit capability 与 neutral DomainAnalysisDefinition 合并成只读 CatalogView，neutral 的 DefinitionCatalog 不重新注册或包装通用 fit。

neutral runtime 用一个无 Fit 语义的 `bind_analysis(bound_operation, DatasetInputSlot) -> AnalysisStep` 适配器托管执行；baseline 的 data-owned bound operation 只有 BoundFit。Pipeline 编译时 slot 只有 expected DatasetSchema；DatasetBuilder finalize 后，runtime 把该 slot恰好一次解析为冻结 DataBlock revision，再调用 `BoundFit.run`。它负责 cancellation、compute lane、terminal propagation 与 result handoff，但不根据字符串 analysis id 选择算法，也不定义 zlc_data result schema。interactive/offline frontend 可以把同一个 BoundFit 与当前冻结 snapshot 投递到自己的 executor。依赖方向始终是 neutral/frontend -> zlc_data，不产生新的 fit owner，也不假装未来 dataset 在 bind 时已经存在。只有出现第二个确实共享此绑定合同的通用 Analysis 后，才把 `BoundOperation` 提升成公共泛型/Protocol；当前不为它预建 registry、descriptor hierarchy 或 program DSL。

正式 Analysis 是同一个 flat RunPlan 的 materialization/EOS 之后阶段；interactive plot Analysis 则由 workbench/frontend adapter 对当前冻结 revision 发起，使用不同 QoS，但调用相同函数。两者都不是 StreamProcessor，也不通过伪造一个“累计 DataBlock event”接入 sample stream。

对已保存 Capture/Data artifact 单独运行 calibration/report 时，composition 提供 `compile_analysis_run(step, resolved_inputs) -> flat RunPlan`；它只做输入解析、claim/compute budget和artifact commit hosting，不要求用户伪装成 Task，也不引入 Analysis workflow engine。

### 10.6 Pipeline composition

TaskConsole 中的 Measurement/Processor 连接先形成 immutable dataflow，不让节点自行订阅或开线程：

```text
PipelineSpec:
  bound_measurement_sources
  bound_stream_processors
  typed event edges
  dataset_materializers
  post_materialization AnalysisSteps
  artifact/result sinks
  delivery/QoS per edge
  criticality = REQUIRED | BEST_EFFORT_MONITOR

compile_pipeline(spec, immutable bindings) -> RunPlan[PipelineResult]
```

编译阶段完成：DefinitionKey/descriptor binding、event payload/schema/axis 校验、无环校验、ResourceClaims 并集、exact/monitor edge 分类、buffer budget、JoinPolicy、DatasetBuilder key coverage、AnalysisStep input revision、criticality 和 terminal propagation。当前不支持 feedback data cycle；需要反馈控制时使用 revisioned ControlTopic，不把回路伪装成数据边。

REQUIRED source/processor/sink failure 使整个 Run 失败。BEST_EFFORT_MONITOR 只能出现在 monitor-only 叶子分支；其失败产生 panel/branch error、missed telemetry 和 Run diagnostic warning，但不反向终止仍健康的 formal exact run。required outputs 完整时 Run 仍是 SUCCEEDED，但 Result/Event snapshot 含 structured warnings，且失败 panel 不能标成成功。Compiler 禁止 BEST_EFFORT_MONITOR 输出再连接 exact、fit authority、calibration 或 artifact，也不自动 restart 失败 branch。

编译结果仍是一个扁平顶层 RunPlan：一个 RunHandle 依次拥有 online acquisition graph、DatasetBuilder finalization、post-materialization Analyses、artifact commit 和 cleanup。节点不能嵌套 start Run、动态新增边或各自成为 terminal-state owner。PipelineSpec 是静态阶段合同，不是 child-plan workflow DAG。

PipelineResult 只汇总 required sink 的 typed results/artifact refs、structured warnings、event/missed metrics 和 terminal lineage；不暴露 worker、cursor、mutable buffer 或第二套 TaskOutput。

## 11. Fit、Selection、Projection 与 DataFigure

### 11.1 唯一 owner

zlc_data 拥有：

```text
Selection
DataTransformSpec
ReductionSpec
CommittedTransform
FitSpec
FitProblem
FitResultBatch
FitModel
BoundFit
bind_fit()
fit_analysis()/fit() concrete implementation
```

当前只有一个 solver 时不建立 FitSolver Protocol。

核心 transform 类型是可序列化的值，不含 QWidget、artist、callable 或 live signal：

```text
DataTransformSpec:
  transforms: tuple[AxisTransform, ...]

ReductionSpec:
  reducer_id
  input_axis_ids
  operation
  parameters
  validity_policy
```

`apply_transform(block, spec) -> TransformedData` 是 zlc_data 的纯函数。它验证 AxisId、所选 reducer 的封闭合同、单位、coordinates、validity 和 transform 顺序，并返回派生 schema 与 TransformRecords。DataTransformSpec 只描述“对数据做什么”，不包含 auto/default 或显示 binding。Reducer 能力属于 zlc_data 算法目录，不写入 ValueSchema/DatasetSchema；新增 renderer/analysis 不改变数据 fingerprint。

frontend figure 拥有只服务于呈现的：

```text
ViewIntent                         # suggest_view 的小型 enum 输入，不持久化成另一状态机
ViewContract                       # 每种 figure kind 的静态能力/安全规则
ViewSpec                           # 唯一可保存的 presentation spec
suggest_view(schema, view_intent, selection, preferences)

ViewSpec:
  axis_bindings: tuple[AxisViewBinding, ...]
    # 每根 AxisId 恰好一次：X/IMAGE_X/IMAGE_Y/SAMPLE/BATCH/FACET/SLIDER/SELECTED/REDUCED
    # selector/reducer 若存在就内联为带标签的 frontend canonical record
  display_options

ViewSuggestion:                    # suggest_view 的瞬时返回 DTO，不保存
  spec: optional ViewSpec
  status: RESOLVED | REVIEW_REQUIRED | NEEDS_INPUT
  reasons
  alternatives

EvaluatedFigureData:               # evaluator 的 immutable transient DTO，不持久化
  layer arrays/raster inputs
  input_revision
  resolution_records
```

`zlc_data` 对外只有 `DataTransformSpec` 与 `CommittedTransform` 两种 transform 合同；不拥有 x/image/sample/facet、latest/navigation 等呈现语义，也不提供无上下文的 `default_projection(schema)`。同一个 schema 在 image、curve、histogram、meter 和 fit 中需要不同处理；把自动决策放进 data kernel 会让权威路径误用显示启发式。neutral_atom 只依赖 zlc_data，因此既不能调用显示层 auto policy，也不会看到 ViewSpec。

ViewSpec 是 figure 唯一持久 presentation 类型；它不保存 authority seed、CommittedTransform、FitSpec 或 ScanOutputSpec。FigureEvaluator 根据当前 immutable DataBlock revision/validity 和 Selection snapshot，把 ViewSpec 直接求值为 `EvaluatedFigureData`。latest/navigation 每次解析都有明确 input revision/coordinate record；renderer DTO 不能进入 zlc_data authority path。`ViewSuggestion` 只是 ViewSpec 是否能被安全构造的解释性返回，不复制 axis bucket，不成为第四层 projection，也不进入 artifact。

auto slice、latest、repeat mean 和鼠标刚画出的 ROI 都是 display state/candidate。用户从 Fit/Scan 动作接受某个候选时，Workbench 根据当前 Selection snapshot 在对应 FitSpec/ScanOutputSpec draft 中重新构造 DataTransformSpec，再交给 commit_transform；不存在把 ViewSpec 的 axis binding/display operation cast、unwrap 或复制成权威 transform 的通用函数。已保存的权威复用项是独立 AnalysisPreset/FitSpec/ScanOutputSpec，不藏在 workspace ViewSpec 中。

`suggest_view` 返回轻量 `ViewSuggestion` 供 UI 显示。算法只做三件事：优先把信息轴放入 ViewContract 允许的 display/facet/batch；其余轴给有坐标标签的 slider/select；只有 ViewContract 明确允许的 display reduction 才自动加入并始终显示标签。仍需压掉有物理信息的轴而没有唯一规则时返回 `NEEDS_INPUT`。baseline 因而只有一个可保存 ViewSpec、一个权威 CommittedTransform、一个瞬时 suggestion 和一个 renderer DTO；没有可互相转换的五层 projection 状态机。

LatestValid 必须在应用显式 display selection 后解析为唯一、可记录的 axis index。若剩余多个 cell 的“最新 valid repeat”不同，ViewContract 必须明确声明 per-cell gather 及其输出 schema/标签；否则返回 NEEDS_INPUT 或改用声明过的 reduction，不能拼出每点来自不同 repeat 的曲线却标成同一次 shot。

### 11.2 三层 Projection 语义

自动化按结果用途分层，而不是按数据是否“复杂”分层：

| 层 | 是否自动 | 合同 |
|---|---|---|
| 默认视图 | 是 | 根据声明的 AxisSpec 与 ViewIntent 生成建议；不修改 DataBlock；立即可看、可改 |
| 显示用选择/降维 | 是 | 每一步必须在 panel 上可见，保留原始 DataBlock/snapshot，不能成为物理结果的隐含输入 |
| fit、正式 scan y、calibration 中的有损预处理、保存派生结果 | 不允许未提交的 auto | 一旦执行选择/降维，必须使用与 schema 绑定、带 revision 的 CommittedTransform，完整写入 lineage；直接消费完整 DataBlock 不需要制造空 transform |

“显式”指变换在 typed spec、界面摘要和 artifact lineage 中都明确，不等于每次弹窗让用户重复点击。ViewSpec/Selection 可以预填权威操作的候选，但 display operation/axis binding 绝不能逐字复制。用户点击语义明确的 `Fit`、`Run Scan` 或 `Save Derived Result`，提交的是该领域 draft 明示的 transform，而不是“把当前画面冻结”。已保存且 schema fingerprint 相同的 AnalysisPreset/CommittedTransform 可以直接复用。

### 11.3 ViewIntent 与自动建议

最小 ViewIntent 集合：

```text
IMAGE
CURVE
HISTOGRAM
METER
```

每个 ViewIntent 对应一个 declarative ViewContract，而不是 render 主干中的 plot-kind 分支：

```text
ViewContract:
  allowed/preferred display roles
  allowed x/image/sample/batch/facet bindings + value contract
  maximum visible facets/layers
  permitted presentation-only reductions
  unresolved-axis policy
```

ViewContract 是 frontend figure 的静态值。新 plot kind 必须先声明合同，再复用同一个 suggestion/validation pipeline。用户 preference 只能在合同列出的等价安全选项中选默认项，例如 image repeat 显示 latest 或 mean；preference 必须随 workspace 保存，不能自行开放新的 reduction capability。

自动建议的输入只有 schema/axis metadata、视图意图、已有 Selection 和明确的用户 preference；禁止读取数组值后“猜哪条轴像信号”，禁止根据 rank、singleton、长度或 axis 顺序猜语义。

```text
suggest_view(
  schema,
  view_intent,
  selection?,
  preferences?
) -> ViewSuggestion

ViewSuggestion:
  spec: optional ViewSpec
  status: RESOLVED | REVIEW_REQUIRED | NEEDS_INPUT
  reasons: tuple[DecisionReason, ...]
  alternatives
```

- `RESOLVED`：由 ViewContract 和 axis role 唯一决定，所有有损步骤都有唯一、明确的显示语义；
- `REVIEW_REQUIRED`：可以安全预览，但包含需要用户看到并接受的临时 select/reduce；
- `NEEDS_INPUT`：无法在不压掉有物理信息的 axis 时满足该 ViewIntent，此时 `spec=None`。它可以显示占位说明或另一种无损视图，但不能进入权威路径。

每个输入 AxisId 必须恰好出现在 ViewSpec 的一条 AxisViewBinding 中；UI 没有同时画出的轴也必须是 slider/facet/batch/selected/reduced 之一，不能靠独立 `hidden_axes` 字段成为丢轴通道。summary、displayed/reduced axes 和 lossy steps 全部从 ViewSpec 派生，ViewSuggestion 不保存第二份。若同一优先级有多个同 role axis 且合同不能同时容纳，返回 alternatives/`NEEDS_INPUT`，不能按 tuple 顺序选第一个。

自动选择使用稳定优先级：

1. 把 axis 保留为该视图的 display axis；
2. 保留为 batch axis；
3. 在 ViewContract 容量内 facet；
4. 使用已有、带坐标标签的 Selection；
5. 使用 ViewContract 明确允许的 display-only reduction；
6. 仅为预览建立可见的 slider/current-slice；
7. 返回 `NEEDS_INPUT`。

禁止以 `index=0`、flatten、全局 `nanmean` 作为兜底。临时 current-slice 必须显示实际坐标和“仅预览”，也不能被静默升级为 committed input。

### 11.4 role 与 ViewIntent 的组合规则

role 只说明 axis 是什么；ViewIntent 说明用户现在想怎么看。两者共同决定策略。特别是 repeat 不存在全局 `mean` 默认：

| axis role | IMAGE | CURVE | HISTOGRAM | METER |
|---|---|---|---|---|
| repeat | mean 或 latest，由 image contract 声明并标注 | mean/error-band 或 batch，由 curve contract 声明 | pool 为样本，绝不先 mean | latest 或声明的统计量 |
| scan-point | slider/current point 或 facet | 优先作为 x | batch/facet | 不能静默 reduce |
| spatial-x/y | 显示轴 | 需要 ROI/Selection，不自动平均 | batch/facet，除非明确 pool sites | 需要 ROI 或物理积分 |
| spectral | curve x/facet | 优先作为 x | batch/facet | 需要显式 band/integral |
| site/component | facet/batch/select | facet/batch/select | 默认逐 site；pool 必须显式 | select |

`mean`、`sum`、`integrate` 是不同物理 reduction，不能编码成一个含义模糊的 reducer。通用 `mean/sum` 使用 zlc_data 中封闭、版本化的 reducer 合同，并由用户/analysis spec 显式选择；ROI photon count、相机畸变校正等带设备/物理含义的操作由 neutral 领域 StreamProcessor/Analysis 定义，不能因输入恰好是 image 就由 frontend 自动提出。普通 image 默认只能显示、选择或保留 spatial axes。

histogram 的 repeat 语义是 sample binding，不是 reduction，也不是把轴 flatten 后丢掉身份。ViewSpec 保留 repeat AxisId，Histogram layer 将其声明为 `sample_axes`。这由 `HISTOGRAM` 的 ViewContract 表达，render 主干不允许再出现 `if kind == "hist"` 特例。

### 11.5 显示建议如何产生权威 DataTransform

显示与权威提交使用不同类型，防止布尔标志被漏检：

```text
ViewSpec                         # presentation-only；axis binding/display operation 不可提交

CommittedTransform:
  spec: DataTransformSpec
  input_schema_fingerprint
  transform_digest
  revision
  origin: USER | ACCEPTED_SUGGESTION | SAVED
```

CommittedTransform 中的 select/ROI 必须是坐标系和 Selection revision 已解析的不可变快照，不得保存指向 live ControlTopic、slider 或 mutable FigureSession 的引用。ViewSpec 的 x/image/sample/batch/facet binding 与 display operation 不进入 CommittedTransform；fit axes/batch axes 由 FitProblem 明确表达，scan batch axes 由 ScanOutputContract 表达。

```text
commit_transform(schema, authoritative_spec, revision, origin)
  -> CommittedTransform
```

该 zlc_data 函数只验证并冻结完整 DataTransformSpec，不做建议，并从 canonical serialization 计算 transform_digest。Notebook/headless 用户可以显式构造 DataTransformSpec 后调用它，或加载已保存的 CommittedTransform，不依赖 frontend figure/Qt。

提交规则：

1. Panel 始终显示当前视图摘要，并逐项标 scope，例如 `x=detuning · repeat=mean/32 [display] · ROI=A [candidate] · batch=site`；
2. 打开 Fit/Scan 配置时，从当前 Selection snapshot、DatasetSchema 与明确 AnalysisPreset 构造领域 draft；ViewSpec 只提供“用户正在看什么”的候选提示，任何 display reduction 都不复制；
3. Fit draft 由 authority-side `suggest_fit_draft(schema, FitPolicy, SelectionCandidate)` 派生 repeat reduction、fit axes、batch axes；它返回 FitDraft/DataTransformSpec candidate，与 ViewSuggestion 没有继承、转换或字段复制关系。Scan draft同样从 ScanOutputContract 独立派生 output/reduction。二者都做 axis-total-coverage 验证，不能继承 image/sample/facet binding；
4. 若 status 是 `NEEDS_INPUT`，UI 聚焦缺失的 ROI/axis/reducer，禁止开始权威操作；
5. `RESOLVED` 可由紧邻权威 draft 摘要的正常动作直接提交；`REVIEW_REQUIRED` 必须突出显示有损步骤，用户接受该摘要或编辑后再生成 CommittedTransform。这里的 status 来自 Fit/Scan draft validator，不沿用显示 ViewSuggestion 的 status；
6. 需要 transform 的 RunPlan/AnalysisCommand 字段只接收 CommittedTransform，运行中 UI 改选择会产生新 revision，不能改变已启动 run；
7. schema fingerprint 不匹配时提交失效，重新建议或要求修正，不能按 axis index 迁移；
8. `commit_transform` 的参数类型只接受 DataTransformSpec，frontend.figure 不提供 ViewSpec/display operation -> DataTransformSpec 转换 API；Analysis result、FitResultBatch、ScanArtifact 和派生 artifact 记录 spec、revision、schema fingerprint 与 TransformRecord。

保存 workspace 时保存用户最终选择的 ViewSpec，保证重开后的画面一致；保存权威派生 artifact 时还必须保存 CommittedTransform 与 input lineage。保存视图不等于把显示结果冒充原始数据。

### 11.6 多维示例

输入相机数据：

```text
(repeat=32, detuning=21, height=40, width=20)
```

- IMAGE：height/width 为显示轴；detuning 是带坐标标签的 slider/current point；repeat 按 IMAGE ViewContract 选择 `mean/32` 或 latest，panel 明示。底层四个 axis 完整保留。
- CURVE：detuning 作为 x；height/width 不能自动平均。没有 ROI 时返回 `NEEDS_INPUT`，同时继续显示 image 让用户框 ROI；定义 ROI count 后曲线建议成为 `REVIEW_REQUIRED` 或 `RESOLVED`。
- HISTOGRAM：repeat 作为独立样本 pool；site/spatial 维默认 batch/facet，不把 repeat 先平均，也不默认把所有 site 混成一个分布。
- Fit authority draft：`suggest_fit_draft` 令 detuning 成为 fit axis；repeat 默认 preserve 为 batch，或预填用户已提交的 repeat reduction；剩余 site/spatial axis 继续成为 batch。每个 batch cell 产生一个 FitResult，组成 FitResultBatch，绝不对剩余轴 `nanmean`。该步骤不是 ViewIntent。
- METER：没有已声明 ROI/integral 时为 `NEEDS_INPUT`，不能显示像素 `(0,0)` 冒充物理 signal。

### 11.7 Fit contract

zlc_data 的可复现分析合同是 FitSpec，solver 接收已解析数组的 FitProblem；input ref 属于调用 adapter：

```text
FitSpec:
  input_schema_fingerprint
  committed_transform: optional CommittedTransform
  fit_axes: tuple[AxisId, ...]
  batch_axes: tuple[AxisId, ...]
  model_id + model_version
  initial_parameters
  bounds
  weighting/mask policy

WorkbenchFitRequest / FigureFitRequest:
  input_ref + input_revision
  spec: FitSpec

FitProblem:
  x arrays + coordinates/units
  observations
  validity/weights
  fit_axes
  batch_axes
  resolved model/parameters/bounds

FitResultBatch:
  batch_axis_specs
  batch_layout: RECT_C | RECT_F | EXPLICIT
  parameter_schema
  parameter_values: (B, parameter)
  covariance/uncertainty
  goodness_of_fit
  per_batch_status/error
  input_schema_fingerprint
  optional transform_digest(identity if absent) + model id/version + numeric policy record
```

FitModel 声明 independent-variable/observation/parameter 的 canonical unit id 和允许的 coordinate frame；initial/bounds 先通过封闭 UnitConversionTable 转换/验证再进入 solver，FitResultBatch 参数携带 canonical unit。单位或 frame 不兼容是 request error，不能把裸数值直接拟合后只改 label。

DataTransform 后仍存活的每根 axis 必须恰好属于 fit、batch 或模型明确声明的 observation component；不能留给 solver 猜。FitModel 从首版显式声明 independent-variable arity/roles，支持当前已有的 1D 与 2D model；不能把 2D Gaussian 当作未来功能删除，也不能通过数组 rank 推断 arity。

WorkbenchFitRequest 是 workbench Command DTO，可持 app-local LiveDataBlockRef；FigureFitRequest 是 frontend figure DTO，只持 DatasetId。各 adapter 先解析为 immutable DataBlock revision，再调用 zlc_data `build_fit_problem(block, FitSpec)`；zlc_data 不定义 universal InputRef/FitRequest，也不看到 neutral live ref。artifact 保存 FitSpec 与已解析 input lineage，不保存 application request DTO。

batch cell 独立失败时保留其它成功结果并记录 typed status；输入整体 schema/model 不兼容、transform 无效或 cancellation 才使整个 Fit Analysis 失败。FitResultBatch 不包含 runtime EventRef、LiveDataBlockRef 或 ArtifactRef；formal Analysis/figure repository adapter 在外层附加 input lineage。它不拆成多个 scalar signal，overlay 从同一个 result 与外层 lineage 派生。

FitResultBatch 是当前一等需求，不延后：gridplot、site grid 和任何保留 site/component axis 的 fit 都要求“一组共享 model/parameter schema + 按具名 batch axes 排列的每格结果”。`BatchLayout` 复用 PointLayout 的 RECT_C/RECT_F/EXPLICIT 映射思想；稀疏 batch 只保存实际 B 个 cell，missing coordinate 与 fit failure 是不同状态，不能强行 densify 后混成 NaN。grid 的 cell label/coordinate 由 batch_axis_specs + BatchLayout/axis coordinates 派生，不能用 list index 充当永久 identity。ComponentValidity 在 build_fit_problem 时按 batch cell 切片；某个 site 无效只使对应 per_batch_status 失败，不污染其它 cell，也不允许先对 site 轴平均成一个 FitResult。`build_fit_problem` 是 fit densify/packing 的唯一 owner；若某 solver 只接受 dense layout，它必须显式 materialize mapping+validity或在 bind 时拒绝，不由 renderer/collector 猜 reshape。

BoundFit 对 batch cell 使用确定性迭代顺序，并在 cell 边界检查 cancellation；单次 solver call 必须有 max evaluation/time budget。取消可保留已完成 cell 作为诊断 DTO，但整个 formal Analysis 为 CANCELLED，不提交成功 FitResultArtifact；interactive stale result按 DatasetRevisionRef 丢弃。该最小 seam 不引入 workflow engine。

### 11.8 Formal/权威 Fit Analysis 路径

```text
PipelineSpec post-materialization FitAnalysis(FitSpec)
-> zlc_data bind_fit(FitSpec, expected schema) -> BoundFit
-> neutral binds BoundFit + DatasetInputSlot as AnalysisStep
-> DatasetBuilder finalize -> immutable DataBlock revision
-> neutral resolves the slot and executes BoundFit once
-> resolve FitProblem
-> zlc_data.fit_analysis()/fit()
-> FitResultBatch
-> PipelineResult / optional typed artifact
```

FitSpec 必须包含 input_schema_fingerprint 与显式 fit/batch axes；发生选择/降维时 committed_transform 必须存在，identity path 可以为空。Fit Analysis 只在 DatasetBuilder 完成 EOS/key/validity coverage 并冻结输入 revision 后运行，验证 schema fingerprint、解析 FitProblem 后执行相同的 zlc_data transform/reduction/fit 函数。它不在每个 sample/patch 到达时把累计 DataBlock 重新拟合一遍。

### 11.9 Interactive Fit 路径

```text
Plot card AnalysisCommand[WorkbenchFitRequest]
-> bounded Fit executor
-> resolve immutable input revision + FitProblem
-> 同一个 zlc_data fit program
-> FitResultBatch
-> revision-checked overlay/ViewModel
```

interactive Fit 使用 frontend application adapter 提供的独立 bounded Fit executor；同一 panel 的 stale queued request 可 coalesce、已运行的不可中断 solver 返回后按 revision 丢弃。它执行 zlc_data `bind_fit` 产生的同一个 BoundFit，不创建隐藏 StreamProcessor node、不发布正式 measurement signal，也不占用 exact `StreamProcessorWorker` 或 view-evaluation 队列。用户要让 fit result 进入下游正式 pipeline，必须显式添加 Fit AnalysisStep；保存 interactive 派生结果时按 §16 materialize 输入 revision 与完整 lineage。

### 11.10 Offline/Figure 路径

```text
FigureDocument
-> Fit executor
-> 同一个 zlc_data fit program
-> FitResultBatch
-> overlay
```

三条路径只在执行 adapter/publish QoS 上不同；Selection/DataTransform/Fit 在 zlc_data、Projection/overlay 在 frontend.figure 各有唯一 owner，不复制 solver、result schema 或模型解释。

### 11.11 Selection

Selection 是不可变语义值：AxisId、range/index/geometry、coordinate frame。Matplotlib controller 把鼠标事件转成 Selection；Qt adapter 只传递事件。neutral StreamProcessor/Analysis 看不到 artist、Axes 或 QWidget。

Workbench 另有瞬时 `SelectionCandidate(selection, source DatasetRevisionRef, schema_fingerprint, document/viewport revision, coordinate_resolution_record)`。它不是 zlc_data authority 类型，也不持久化成另一套 Selection；它只证明这个鼠标选择来自哪一版数据和坐标变换。Fit/Scan draft 只有在 candidate 仍与目标 snapshot/schema 匹配时，才可显式重建 CommittedTransform；不匹配则 stale/重新解析，不能把旧 ROI 套到新 camera generation 或新 viewport。

Selection 值及其坐标/geometry 语义属于 zlc_data；frontend selector controller 只把鼠标/键盘手势转换成 `SelectionChanged(Selection)`，不导入 neutral_atom 的 ControlTopic。Workbench 的 PanelController 是唯一中介：它判断该选择只是 display state、analysis candidate，还是用户明确绑定到某个 neutral StreamProcessor/Analysis 的 control；只有最后一种才映射为 revisioned ControlTopic command。结果携带 control revision；旧 revision 结果不覆盖新选择。关闭 panel 时 workbench 对未完成 command 发/等待 terminal ack，不能让 frontend selector 直接持有 runtime sink。

### 11.12 Analysis 不建立 god processor

纯算法只有一层命名：`zlc_data.apply_transform`、`reduce_data`、`build_fit_problem`、`bind_fit`、`fit_analysis`/`fit`。formal、interactive、offline 三条路径都执行同一个 BoundFit；neutral runtime 只执行通用 `AnalysisStep`，不定义任何 Fit-named class。`OccupancyStreamProcessorDefinition` 属于 neutral_atom，因为它包含逐帧领域物理语义；`CalibrationAnalysisDefinition`、`ReadoutFidelityAnalysisDefinition` 属于 neutral_atom，因为它们消费完整 dataset/artifact。简单的、无领域语义的逐 event 变换若确需进入在线图，可由 zlc_data 提供纯函数，neutral pipeline 在 composition 时绑定为普通 StreamProcessor operator，但不复制实现。

zlc_data 的 solver 是同步纯调用，不注册 frontend 提供的 GUI-thread guard、不读取环境变量来判断调用线程，也不持有 executor。是否在 GUI thread 之外执行是 frontend/neutral hosting adapter 的合同，并由真实入口测试证明；把线程策略注入数学 kernel 会形成隐藏全局反向依赖。

### 11.13 DataFigure

```text
FigureDocument    immutable datasets/layers/view specification
FigureSession     transient frontend interaction state
FigureEvaluator   (document, ResolvedDatasetMap) -> EvaluatedFigureData
FigureRenderer    (document, EvaluatedFigureData) -> surface/frame
FigureCodec       current schema only
DataFigure        frontend.render 的 notebook/public render facade
```

FigureDocument 只持有 frontend-owned DatasetId/immutable dataset descriptor、zlc_data Selection 和已解析的 ViewSpec；ViewIntent 只是创建/编辑时调用 suggest_view 的输入，不成为另一份持久状态。权威派生 dataset 另带 zlc_data CommittedTransform/analysis record 与 frontend FigureArtifact digest。FigureDocument 不持有 neutral runtime ref/lineage 类型；Workbench 在 materialize 时把外部 causation 转成 FigureArtifact manifest 的普通 canonical descriptors。Workbench LiveFigureBinding 维护 LiveDataBlockRef -> DatasetId 的临时映射，解析成 zlc_data DataBlock snapshot/ResolvedDatasetMap。

Interactive path 在 per-panel latest-only view-evaluation executor 运行 FigureEvaluator：直接解析 ViewSpec 的 axis bindings/navigation policy，再执行 display transform/reduction/layer data 计算，产生带 document/input revision 与 resolution records 的 immutable EvaluatedFigureData；具体 surface ownership 见 §12.5。Headless renderer 在自己的线程完成 evaluate+render 并永久拥有 Figure。两者都只执行 document 已决定的 ViewSpec，不重新猜 axis；live/persisted binding 的保存规则见 §16.3。

FigureDocument/FigureEvaluator/codec 属于 headless `frontend.figure`；DataFigure 因拥有 renderer/surface/export convenience，属于可选 `frontend.render`，安装 `[render]` 才存在。neutral_atom 只返回领域 Result/ArtifactRef；notebook/workbench projector 把它映射为 FigureDocument，neutral_atom 不导入 figure 或 DataFigure。DataFigure 不主动访问 Hub、Task、Session、PulseDocument 或 Device。

## 12. Workbench 与 UI

### 12.1 最小应用职责

不强制一项职责一个 Service 类。最小组件：

```text
WorkspaceModel
RunCoordinator
PanelController
PulseController
ArtifactController
Presenter / View
```

RunCoordinator 只是 RunController/RunHandle 的 Qt-facing adapter：把 typed command 转成 start/cancel，把 EventStream 转成 ViewModel；它不拥有第二套线程、状态机、resource lease 或 terminal state。PulseController/PanelController 同样只编排 command 和 presentation state，不直接调用 driver。

### 12.2 Command/ViewModel

View 只发送 typed Command，接收 immutable ViewModel/DataRef。Backend 不修改 Widget。

Workbench 拥有 UI Command/ViewModel；neutral_atom 拥有领域 Request、RunPlan/RunHandle/Event；zlc_data 拥有 Selection/DataBlock/Fit 类型；frontend 拥有 Figure/View/interaction 类型；zlc_pulse 拥有 Pulse/compile/transport 类型。Controller 在 composition boundary 显式映射，不建立跨 bounded-context `common.dto`，也不让领域包为了某个按钮新增字段。

Workbench 大图像 ViewModel 使用 app-local LiveDataBlockRef/ReadOnlyArrayView 和 revision，不默认在每个 UI hop 再深拷贝。默认发布边界产生拥有自己内存的 immutable snapshot；若 driver 会复用 buffer，必须在该边界 copy，发布后 producer 不得再修改。该 live ref 经 LiveFigureBinding 解析，不泄漏进 frontend FigureDocument/codec。

baseline 的 `LiveFigureBinding.resolve(DatasetRevisionRef, SnapshotQuery) -> OwnedSnapshot` 只 materialize 当前 ViewSpec 所需 axis slice/chunks，不默认复制完整累计 DataBlock，也不返回 mutable builder alias。SnapshotQuery 只描述所需 slice/revision，显示 reduction 仍由 FigureEvaluator 拥有。只有 profiling 证明大帧发布 copy 是真实瓶颈、且某 adapter 明确提供可 pin 的零拷贝 buffer 时，才启用 opt-in `BorrowedSnapshot`：它把 read-only bytes 与 workbench-owned release token 绑定，token 只存在于 LiveFigureBinding/WorkbenchRenderMessage，frontend 类型和 artifact codec 永远看不到。worker 已产生不再 alias 的 layer/raster 后立即 release；若 front buffer 仍 alias borrow，则 front-buffer replacement、stale-result discard、queued-job cancellation、panel close和shutdown都必须在 `finally` 中 release。Save 先物化 owned bytes。该优化必须有 pin 上限、timeout/quarantine 与 shutdown drain 测试，不能成为所有数据的默认抽象。

### 12.3 Setting 与 Edit

统一的是：

- EditorSession；
- base_revision + draft/apply/cancel；
- validator；
- 基础控件；
- typed command。

普通 scalar、bool、enum、bounded number、unit value、简单 path/list 可以 schema-driven 自动生成。

以下使用显式 presenter/view，不强行自动生成：

- Pulse editor；
- ROI/selector；
- fit axis/batch/reduction；
- calibration workflow；
- device connection；
- resource conflict 和安全确认。

领域 Definition 只拥有 type、required、default、unit、range、choices 和 semantic description。group/order/widget/layout/file-dialog/dynamic-enable 属于 workbench/frontend。

Apply 时必须同时通过领域 request validator 与 `base_revision == current_revision` 检查；后台或其它 editor 已更新配置时返回 typed EditConflict，不能 last-write-wins。Cancel 只丢弃 draft。UI 的 enable/disable 是提示，RunPlan.bind/preflight 仍执行同一个权威 validator，不能信任界面已经挡住非法输入。

axis 编辑器读取 ViewSuggestion，并只从其中的 ViewSpec 派生 display/x/sample/batch/select/reduce/facet 摘要，不让 image、rolling、histogram 和 fit 各自实现一套 shape 猜测。`RESOLVED` 建议无需弹窗；`REVIEW_REQUIRED` 在 panel 摘要中持续突出有损步骤；`NEEDS_INPUT` 才展开最小必要编辑器。

### 12.4 Qt 线程规则

- QObject affinity 由创建线程或 `moveToThread` 决定；
- 不直接跨线程调用 worker method 并假设它会在 worker thread 执行；
- queued signal 只传 immutable DTO、只读数组或明确 copy；
- 终态禁止跨线程传 QWidget、Figure、Canvas、artist、driver/session handle；S0.5 的 allowlisted SerializedLegacyAggBridge 只能做有确认的排他 ownership handoff，不能把对象放进普通 queued signal；
- 所有结果带 run_id/revision；
- window close、cancel 或新 revision 后丢弃旧 result；
- 禁止 BlockingQueuedConnection、嵌套 processEvents/QEventLoop 等待 worker；
- shutdown 后 queued result 仍可能到达，receiver 必须有 shutting-down gate。

### 12.5 Render ownership

终态只允许**单 owner**，但承认三种不同 surface，而不是把所有 live 图逼回 GUI-thread Matplotlib compose：

1. `GUI_ARTIST`：低成本 1D/少量 artist；GUI thread 永久拥有 Figure/canvas/artist/overlay，view-evaluation worker 只返回 immutable layer arrays；
2. `WORKER_RASTER_LIVE`：高成本 2D、grid 和多 panel board；render worker 永久拥有独立 Agg Figure 或纯 raster backend，GUI 只接 immutable front QImage/texture + coordinate transform，并用 Qt overlay 处理 ROI/crosshair/selection/hover；
3. `WORKER_HEADLESS_EXPORT`：export worker 永久拥有独立 Figure，只返回 immutable image/file artifact。

`WORKER_RASTER_LIVE` 是复合板的正式性能路径，不是把每个 panel 退回 GUI compose。worker 可一次 compose 整个 board 或一组共享布局，并通过 double/triple buffer 发布：

```text
BoardFrame:
  board_revision
  immutable front raster(s)
  panel_rects + ViewportTransform(s)
  coherence_groups: group -> CoherenceStamp
  per_panel_status/missed

CoherenceStamp:
  run_id/provenance_epoch_id
  join_key type/schema/digest
  input DatasetRevisionRef(s)
  panel/document/selection revisions
```

同一 coherence group 的 panel 必须从同一个 causation domain 与完整 CoherenceStamp 求值，并在一次 GUI transaction 中 `present(BoardFrame)`；裸 `JoinKey==7` 或裸 `revision==5` 在不同 run/block/generation/schema 间不具有可比性。不能让 per-panel latest-only 各自成功后拼成看似同 shot 的 board。互相独立的 monitor 可以带不同 revision，但 BoardFrame 必须显式标出它们不是 coherent group。强像素级 coherence 使用一个 parent raster/front-buffer 原子换页；多个独立 QWidget surface 最多声明 model transaction coherent，不能声称 OS paint 同一时刻完成。后台慢时丢弃未开始的旧 board revision，不能逐 panel 呈现半新半旧状态。

GUI 不读取 worker-owned Figure/artist；所有命中测试和选择都使用随 front buffer 一起发布的 ViewportTransform。静态 axes/labels/colorbar 可由 worker raster 缓存，动态 overlay 由 Qt 画；export 始终从 FigureDocument + frozen data revision 重画，不保存屏幕 texture。

```text
LiveRasterFrame:
  image
  document_revision + input_revision
  axes_pixel_rect
  ViewportTransform(data <-> logical pixels)
  scale/inversion/coordinate-frame metadata
```

Overlay 的鼠标点先用同一 revision 的 ViewportTransform 转回数据坐标，再产生 Selection Command，经 workbench 转成 ControlTopic 或 analysis candidate。zoom/pan 改 ViewSpec/document revision并请求新 LiveRasterFrame；旧 frame 或旧 transform 的事件一律丢弃。非线性轴必须由 transform 显式支持，不能拿线性比例近似。view-evaluation array worker 不访问 Figure/QWidget；WORKER_RASTER_LIVE 的 render worker 可访问且永久独占自己的 Figure，GUI 不访问该 worker state。导出从 FigureDocument + frozen data revision 在 headless renderer 重画，不把屏幕 texture 当权威数据。

Matplotlib/Agg 使用一个有界、公平的 render lane（或隔离 process），只用 OO Figure/Agg API，不并发修改 pyplot/rcParams 等全局状态；纯 raster backend 才可安全并行。lane 对单个 board job 设最大 compose 时间/分片或等价 bounded-wait，continuous live board 不能饿死 export、其它 board 或 control-related raster；未开始的旧 live job可 coalesce。worker 应缓存未变化 panel tile并在整板完成后原子交换 front index，不能把“整板 coherent”实现成每次 source event 全量重画所有 panel。

当前 console-wide RenderLoop/Matplotlib Figure 不能在 S1 前瞬间删除，所以 S0.5 使用三个互不冒充的迁移桥：`LegacyPanelHost/CatalogRouter` 托管并逐项隐藏旧 panel；`LegacyRuntimeFence` 让所有旧 LogicNode start/stop先取得同一 ResourceArbiter 的保守 ResourceClaims，真实 thread termination + safe ack 前不释放；`SerializedLegacyAggBridge` 只负责旧 Figure ownership handoff。旧 Figure 在 compose 期间由 render worker独占，GUI 只有在成功 handoff 后才可执行 allowlisted artist/selector 操作；非 GUI 线程调用 `draw_idle/update/resize/mpl_connect/mpl_disconnect/Qt selector state` 等 QObject-affine API 必须机械拒绝。若无法证明旧 Figure 已与 QTAgg/QWidget state 解耦，则 worker 必须使用独立 Agg clone/raster。handoff timeout 时禁用交互/延迟 teardown并显示错误，绝不能继续访问。三个 bridge 都只存在 workbench migration adapter；Z0 全部为 0。

因此终态仍禁止 worker 与 GUI 同时或无确认地访问同一个 Figure；这里接纳的是现有性能事实和明确的迁移 handoff，不是把双 owner/barrier 自死锁提升成架构合同。

### 12.6 UI 可见 Fit

提供两个明确入口：

```text
Add Analysis -> Fit
Plot card -> Analyze -> Fit
```

前者用 zlc_data `bind_fit(FitSpec, expected schema) -> BoundFit`，再由 composition/runtime 与 DatasetInputSlot 包成 PipelineSpec 的 post-materialization AnalysisStep，在冻结 Dataset revision 后产生正式 FitResultBatch/result artifact；后者把同一个 FitSpec 与当前 input ref/revision 包成 WorkbenchFitRequest，并执行同一个 BoundFit，只更新当前 panel overlay。二者复用同一个 frontend editor 与 zlc_data solver，但生命周期和发布语义不同，UI 必须明确标注。

UI 明确展示 input、fit axes、batch axes、selection、reduction、model、initial/bounds、status、result 和 overlay。

Fit 面板先调用 authority-side `suggest_fit_draft(schema, FitPolicy, SelectionCandidate)`，不把 Fit 当作 ViewIntent。用户看到的普通 `Fit` 动作就是对该权威 draft 的提交，不额外制造无意义确认框；但存在未解决 axis 时按钮不可执行，并明确指出需要 ROI、选择 axis 或定义物理 reducer。当前 ViewSpec 只帮助用户理解候选 selection，不提供可复制的 display reduction。运行后 selection/reduction 改变会创建新 revision 和新结果，旧结果不能覆盖。

### 12.7 Shutdown

关闭窗口是显式状态流程：

```text
reject new commands
-> mark UI shutting_down
-> terminal-ack pending ControlTopic revisions
-> cancel all RunHandles
-> await termination acknowledgement without blocking GUI event loop
-> stop producers/subscriptions and reject new view/fit jobs
-> cancel queued latest-only work; drain in-flight view evaluation/raster work
-> deliver/discard final revision-checked GUI results
-> close interactive FigureSessions/renderers on GUI thread
-> verify/release all opt-in BorrowedSnapshot tokens
-> stop view-evaluation/Fit/raster workers
-> close DeviceSet on owner I/O lanes
-> stop lanes
-> flush QuarantineJournal/artifact repository diagnostics
-> destroy Qt views
```

存在 CANCELLING、join timeout、QUARANTINED resource 或 safe failure 时，普通 close 不能假装完成；UI 显示阻塞原因并提供继续等待或明确的强制进程退出。强制退出是用户确认的最后手段，不改变日志中的失败/安全不确定状态。queued result 通过 application lifetime token + run/panel revision 双重检查后丢弃，不能更新已销毁或 id 被复用的新 panel。

## 13. Calibration

Calibration 是 `neutral_atom.readout.calibration` 的内建 feature，不使用 plugin、entry point、包扫描或动态 registry 覆盖。

### 13.1 Artifact

```text
CalibrationArtifact:
  frame_contract
  site_map
  models: tuple[ReadoutModel, ...]
  capabilities/stage
  required_model_kinds
  algorithm_version
  input_lineage
  parameters

ReadoutModel =
    BoxReadoutModel
  | PerSitePsfReadoutModel
  | UniformPsfReadoutModel
```

一次 calibration 可产生共享 artifact 中的多种 model。artifact 的 `capabilities/stage` 明确区分 site-map-only、含 threshold、含完整 readout model 等完成态；这是合法的 typed capability，不是 partial-success 模糊状态。每个 Analysis 声明自己需要的 capability/model kind。Occupancy request 可显式选择 model；若用户未指定，只允许按 Definition 声明的稳定 default model policy 在 artifact 内唯一选择，并把实际 model id/version冻结进 request/lineage。没有唯一 default 时构造 request 即提示选择，不能按 tuple 第一项猜，也不能让 notebook 短路径退化成每次手写冗长参数。

```text
FrameContract:
  DatasetSchema/ValueSchema fingerprint
  stable camera/sensor identity + optical/readout path identity
  sensor/ROI/binning geometry
  dtype + count unit
  exposure/gain/readout mode
  coordinate frame

SiteMap:
  stable site AxisSpec
  site coordinates in the same coordinate frame
  detection/selection lineage
```

所有会改变数值解释的采集设置都进入 FrameContract fingerprint。CalibrationRequest 明确列出 required_model_kinds；所有 required model 成功且通过质量 gate 才原子提交 CalibrationArtifact，不能把部分失败 artifact 当成功。每个 model 保存自己的参数 schema、适用 FrameContract/SiteMap fingerprint 和质量指标。

### 13.2 输入

```text
CalibrationInput =
    LiveCalibrationInput(CaptureSpec)
  | CaptureArtifactInput(CaptureArtifactRef)
```

neutral domain/runtime 不接受“执行时读取 session current calibration”、裸 filesystem fallback 或 legacy path search。Notebook `Experiment.readout` 的 binding-keyed calibration ref 只是 composition facade 的可见默认：facade 构造 Occupancy/Detection/Scan request 时必须把具体 ReadoutBindingKey、CalibrationArtifactRef 和 selected/default model id复制为显式字段并冻结；若 ref 为空、binding/FrameContract/capability/model 不适用，request 构造/preflight 失败，运行中不能换成另一个 calibration。Workbench 同样在用户点击 Run 时冻结当前选择，而不是让 processor 回查 mutable session或按 repository 最近文件猜。

### 13.3 执行

```text
CalibrationTask:
  LiveCalibrationInput:
    CaptureSession -> CaptureRepository.atomic_put -> CaptureArtifactRef
  CaptureArtifactInput:
    validate/load CaptureArtifactRef
  -> bind CalibrationAnalysisStep(capture_artifact, parameters)
  -> 同一个 calibrate(capture_artifact, parameters)
  -> CalibrationRepository.atomic_put
  -> CalibrationArtifactRef
```

live 路径先提交原始 CaptureArtifact，再与 offline 路径汇合；算法或模型质量 gate 失败时不产生 CalibrationArtifact，但原始 capture 仍可用于诊断和重跑。virtual/real 只在 CaptureSession adapter 不同，提交后的校准代码完全相同。

不需要 CalibrationService、child Measurement Run、calibration StreamProcessor 或 reducer 包装。普通批量校准就是 neutral-owned `CalibrationAnalysisDefinition` 绑定出的 AnalysisStep，在该 Task 的 run-owner/必要时 disposable compute process 中运行，不占用 view/Fit executor，并受同一 RunHandle cancellation/revision 管理。只有出现一个必须在采集完成前产生控制反馈、且不能保存原始样本后批处理的真实用例，才另行设计领域 StreamReducer；baseline 不为 calibration 预留它。

Occupancy request 携带 CalibrationArtifactRef 和 selected model kind；RunPlan 在绑定 `OccupancyStreamProcessor` 前解析为 immutable CalibrationArtifact，并验证输入 sample/capture 的 FrameContract、SiteMap/coordinate frame、model applicability 和 algorithm schema。任何 mismatch 明确失败，不按相同 shape 猜“应该兼容”。

## 14. PulseScan

### 14.1 两种明确语义

```text
FormalPulseScanRun   权威、完整、端到端 exact、可产生 ScanArtifact
LiveSweepMonitor     非权威、可跳帧、只用于显示
```

它们不是 fallback 或兼容双轨，而是两个不同 use case。LiveSweepMonitor 不能保存为成功 ScanArtifact。

PulseScan 的精密时序由冻结bitstream上的FPGA scan engine与qCMOS外触发硬件执行；host只在run前冻结配置/计划并在run后验证。当前baseline使用现有FPGA completion/progress/build fingerprint和qCMOS sensor/capture counter、framestamp/camerastamp/timestamp，不假定逐沿counter、delay-idle或PHYSICAL_DONE。host wall-clock不调度pulse edge；仅E0测得的camera drain bound可作为末端等待上界，并必须与硬件counter/metadata对账共同使用。

### 14.2 Request 与 ScanPlan

```text
PulseScanRequest:
  compiled_pulse_ref 或可编译 PulseDocumentRef
  point_axes + PointLayout
  repeat_axis
  slot_binding = SCAN_SLOT | API_SLOT
  slot values
  exact source PipelineSpec/config
  ScanOutputContract draft
  deadline/budget policy

ScanPlan:
  CompiledPulseArtifact identity
  immutable point/repeat axes + PointLayout
  ordered expected ScanCellKeys/TriggerKeys
  frozen slot values + trigger schedule
  bound exact source pipeline
  resolved ScanOutputContract
  execution_strategy = AUTONOMOUS_STREAMED | API_SLOT_SEGMENTED_EXISTING
  correlation capability/proof record
  total event/byte/cardinality budgets
```

`PulseScanDefinition.bind(request, bindings) -> RunPlan[ScanArtifactRef]`。bind 完成纯 request/port/claim 绑定；preflight 在正确 I/O lanes 解析硬件 capability、schema、counter mode、compiled pulse compatibility 与全部预算，生成 ScanPlan 并放入 PreparedRun。ScanPlan 一旦生成不可被 GUI/ControlTopic 修改，也不包含 child RunPlan。

point_axes/PointLayout 决定 logical cell 顺序，trigger schedule 明确每个 ScanCellKey 期望的 TriggerKeys。`slot_binding` 是用户/模板的参数语义；`execution_strategy` 是物理执行方式。正常模式固定为现有 bitstream 的 `AUTONOMOUS_STREAMED`：SCAN_SLOT/MOT 将完整冻结 scan table交给 FPGA 一次无缝执行。只有 selected=API_SLOT 且 adapter明确证明该 API值无法在一次自主 sweep中更新时，才允许既有 `API_SLOT_SEGMENTED_EXISTING` 路径；它不能反向成为 SCAN_SLOT fallback。任一数量、slot、camera envelope、schema或 output contract无法在 fire前解析，preflight失败且不 arm。

Formal SCAN_SLOT 的 repeat axis 必须在compile阶段展开进一个完整、有限、冻结的scan table，并使用现有`repeat_forever=False + scan_points` finite single-pass路径；table顺序由ScanPlan的repeat/point axes与PointLayout唯一决定。**禁止使用当前`scan_repeats>0`的cursor-wrap + host `CMD_SAFE`路径**：它由host观察wrap后停止，现实现明确可能已经多发下一sweep的一个point，不能进入Formal Scan。`scan_repeats`只保留给非权威monitor/旧交互路径，不能成为S4执行策略。

### 14.3 ExactAcquisitionPipeline

FormalPulseScanRun 消费一个预先声明、固定有限的 exact source pipeline，而不拥有或选择具体 camera：

```text
Declared ExactSourcePipeline(s)
-> StreamProcessorWorker(s)
-> exact y stream
-> scan DatasetBuilder/collector
```

Camera CaptureSession 只是 ExactSourcePipeline 的一个实现；其它 Measurement/仪器也可以产生 y。PulseScan 本身只拥有 sequencer、scan plan、exact pipeline contract 和 collector，不 import camera 或假定 y 的设备来源。

FormalPulseScanRun 在 fire 前为本次 run 创建并启动整条 dedicated exact chain。不能借用 serving monitor/latest 的 `StreamProcessorWorker`；operator 实现可以复用，但 cursor、reservation、queue 和 worker state 都属于本次 RunHandle。任意 source/worker failure 立即传播给 scan run。

若 continuous camera monitor 正占有同一单-owner device，Workbench 先请求停止其 RunHandle 并等待 termination，再启动 FormalPulseScan；不把旧 CaptureSession 暗中“转交”给新 run。正式 scan 期间的 live image 订阅本次 CaptureSession 的 monitor tap；scan 结束后需要持续监视时再显式启动新的 monitor run。

每条边在 fire 前建立：

- schema/generation；
- expected event/sample/grouping；
- byte/event budget；
- reservation/cursor；
- JoinPolicy；
- error propagation。

因此不仅 final y，所有声明的 source -> processor 上游边也得到 exact 保证。

### 14.4 状态与安全

```text
PREFLIGHT -> RESERVED -> SOURCES_READY -> FIRED -> DRAINING
-> SOURCES_STOPPED -> SAFE_CONFIRMED -> PROVENANCE_VALIDATED
-> COMMITTING -> SUCCEEDED
```

任意 duplicate、out-of-order、typed key mismatch、gap、EOS incomplete、schema change、timeout 或 hardware fatal：

```text
ABORTING -> SAFE_CONFIRMED -> FAILED -> release
         -> SAFE_FAILED -> FAILED + quarantine
```

若错误发生在已 SAFE_CONFIRMED 的 COMMITTING，只需删除未提交 temp、记录 FAILED 并 release；不把保存失败误报成采集成功，也不重复 fire/重开 source。

正常和abort路径都在artifact commit前停止sources并按现有协议请求sequencer safe。abort顺序固定为：设置cancel intent -> 调用现有abort/safe尽快阻止更多trigger -> disarm source -> 等待现有status/completion或保守drain deadline -> abort/drain workers/builders -> join acknowledgement。不能先做长CPU/磁盘工作再请求safe。只有SAFE_CONFIRMED且existing FPGA readback、camera metadata、join、DatasetBuilder coverage与最终EOS全链通过EndAttestation，才允许ScanArtifact commit；无法确认safe则ResourceClaim quarantine。provenance validation失败只产生RunFailureRecord，不保留已显示的provisional rows。

### 14.5 Scan keys

```text
ScanCellKey:
  run_id
  point_index: tuple[int, ...]
  repeat_index

TriggerKey:
  cell_key: ScanCellKey
  trigger_ordinal
```

ScanCellKey 对应最终 DataBlock 的一个 `(R,P)` cell。`point_index` 与 ScanPlan 的 point_axes 一一对应，通过同一个 PointLayout 映射物理 P；即使只有一根 scan axis 也使用长度一 tuple，不能重新引入扁平 `sweep_index`。TriggerKey 区分同一 cell 内的多次硬件触发。

qCMOS adapter在匹配E0 envelope时按冻结schedule为第i个按序frame生成**provisional TriggerKey[i]**；需要多帧/多source的StreamProcessorDefinition声明grouping与join-key transform，在完整输入到达后产生恰好一个provisional ScanCellKey typed result。scan DatasetBuilder只接受ScanCellKey并验证计划内每个cell恰好一次；整个epoch只有EndAttestation后才转VALID，不能在验证前提交，也不能从monitor/latest路径填“下一个cell”。

近期 baseline 的关联模式是 `ORDERED_END_ATTESTED_RUN`，不是逐 cell handshake，也不要求逐沿 FPGA tag：

```text
E0 CameraExternalTriggerEnvelope
  冻结 qCMOS model/firmware、trigger mode/polarity、ROI/binning、
  exposure/global-exposure/readout mode、buffer policy
  在声明的 trigger_interval_min + safety_margin 工作区间内实测：
    每个外触发产生一帧、delivery order稳定、frame/camera stamp单调连续
  保存样本数、持续时间、最坏间隔与实测丢帧/乱序结果

Preflight
  从冻结 CompiledPulse trigger schedule 得到 expected TriggerKeys/总帧数
  从 camera envelope 得到 minimum safe frame interval
  验证每个相邻有效 trigger间距 >= minimum + configured margin
  验证 total frames/bytes <= qCMOS finite buffer/driver/device capacity
  若table超过resident window，验证每chunk playback time >= refill p99 + margin
  冻结 camera settings readback；不满足则不 arm

Run
  camera一次 arm并按整个 scan frame budget配置 capture/buffer
  FPGA以repeat_forever=False一次fire并自主执行展开repeat后的finite scan table
  adapter按 delivery order保存全部 frame及 framestamp/camerastamp/timestamp
  所有数据在末端验证前均为 PROVISIONAL

EndAttestation
  existing FPGA completion/scan-progress readback 与冻结计划一致
  planned/emitted-by-existing-readback total == camera produced total
  frame/camera stamp按E0语义连续，timestamp间隔在E0容差内
  DatasetBuilder/processor/EOS coverage完整
  任一不符 -> 整 run INVALID、丢弃并重跑；全部通过才提交
```

通过 E0 后，`frame[i] -> frozen trigger schedule[i] -> TriggerKey` 是该 qCMOS/工作区间的 adapter contract。这里依赖的是经真机验证的确定性外触发和顺序交付，不是运行时“取 latest”或两个自由流按 N zip；host侧 reservation/cursor仍保证相机已交付的每帧不会在软件缓冲中静默跳过。

当前baseline中的`emitted_total`不伪装成新增逐沿counter：只有现有DONE/status/scan-progress证明完整自主table到达正常terminal时，才可由compiled schedule的有效camera-trigger数得到；任何early stop、status歧义、cursor未达终点或transport error都使EndAttestation失败。timestamp检查按E0实测的“相机timestamp事件定义 + 非均匀trigger schedule + readout容差”比较，不能简单要求固定间隔或拿host wall clock替代。

`HOST_STEPPED_GROUP`、逐 cell arm/fire/wait、single-cell fire gate、per-cell `PHYSICAL_DONE` receipt 均不属于 baseline，也不能作为 qCMOS 首光方案。SCAN_SLOT/MOT 必须使用现有 FPGA 无缝完整 table；API_SLOT 只有在既有 adapter确实不能无缝更新时才使用已接受的 segmented路径。架构不得为了获得逐 cell证明而要求新 bitstream。

`AUTONOMOUS_STREAMED` 已是当前冻结 bitstream 的正式 baseline，不依赖下述增强。只有 E0 在批准工作余量内实测发现丢帧/乱序、末端对账无法满足实验正确性，或发现现有 RTL 真 bug/与既定设计不符时，才可提出证据驱动的 bitstream变更。届时可选增强之一是在真正 camera-trigger输出沿记录：

```text
HardwareTriggerStamp:
  prepare_generation
  emitted_edge_counter
  compiled_trigger_ordinal
  fpga_tick
  scan_linear_cursor_diagnostic_only
```

若未来实施 HardwareTriggerStamp，相机 trigger走 per-channel delay时不能把输出时的当前 `scan_cursor` 当作cell身份；ordinal必须随事件保存或由 emitted counter对冻结 schedule映射。该要求只约束未来增强的正确实现，不是当前 baseline 的隐藏 RTL要求。当前 baseline仍要求一个 CompiledPulseArtifact只声明一个正式 camera-trigger output channel，并由 compiler生成确定性有效沿 schedule。

未来若证据触发逐沿 FIFO方案，其容量、overflow sticky fatal与排空带宽必须共同设计和真机验证，不能先写架构再要求重烧。未触发时不实现、不预留、不把它列为当前 Formal Scan gate；当前 UART/JTAG也不被假定具有尚不存在的高带宽 telemetry能力。

相机 adapter 每帧保留 DCAMBUF_FRAME 的 `framestamp`、`camerastamp`、`timestamp` 与 capture `nFrameCount`，不能像当前 adapter 一样只返回 ndarray；这些字段的语义必须按具体 qCMOS 型号实测，字段存在本身不等于 TAGGED。

ScanArtifact 的 provenance manifest 保存 `AUTONOMOUS_STREAMED`/API segmented mode、CameraExternalTriggerEnvelope id/digest、冻结 camera settings readback、compiled trigger schedule digest、现有 FPGA completion/progress readback、expected/produced total、完整 frame/camera stamp与timestamp range/digest、frame-index -> TriggerKey mapping digest，以及 end-attestation结果。加载时这些记录与 ScanPlan/TriggerKey coverage一起验证；不能只保存 `mode="ordered"` 布尔声明。

运行闭环是：camera一次 arm整个 frame budget -> FPGA一次 fire完整 scan table -> exact queue按序保存所有 frame+metadata -> hardware run完成后等待E0声明的最坏 camera drain deadline -> 停止/读取最终 camera counter -> 完成 processor/DatasetBuilder最终EOS -> 执行 EndAttestation -> VALID后才commit。未验证前每个 Envelope携带 run-scoped provenance_epoch_id且 formal sink只暂存；任一 count、stamp、timestamp、coverage、timeout或hardware error不符使整个 epoch INVALID并丢弃，不能提交前半段。

明确接受的取舍：这是“preflight余量 + per-run末端对账 + reject-and-redo”，不是“per-cell当场fail-closed”。它通常能发现漏帧、乱序、未完成和大间隔异常，但不能定位具体point，也不能数学上排除漏一个触发/帧同时出现一个额外触发/帧且metadata仍落在容差内的等量抵消。PI接受这一剩余风险以换取冻结RTL和无缝扫描；文档、UI和artifact provenance必须如实标记 `ORDERED_END_ATTESTED_RUN`，不得声称拥有逐沿accepted-trigger证明。

INVALID必须形成可查询的RunFailureRecord，保存失败原因、工作点、counter/stamp摘要和累计失败率。系统不得无限或静默自动重试直到“碰巧成功”；重试只能由用户发起，或由request中显式、有限、可审计的RetryPolicy发起，每次attempt具有独立run_id/provenance，最终artifact记录失败attempt refs。这样reject-and-redo不会掩盖硬件不稳定或造成不可见的选择偏差。

只有 E0 在目标工作余量内观察到真实丢帧/乱序，或代码/RTL证据证明现有实现与既定设计不符，才重开bitstream评估；候选可以是bug fix、HardwareTriggerStamp、trigger-return或其它最小修复。触发条件、证据、替代的软件/相机配置方案和重烧风险必须单独评审，不能由本架构自动授权。

### 14.6 非标量 y

正式 scan 的权威 y 必须在 RunPlan preflight 时选择一种：

- 通过 CommittedTransform 显式 Select/Reduce 成 scalar；或
- 作为 batch axes 完整保存。

```text
ScanOutputContract:
  source_output_id
  input_schema_fingerprint
  committed_transform: optional CommittedTransform
  repeat_axis: AxisId                 # authoritative R，必须原样保留
  output_data_axes: tuple[AxisId, ...]
  batch_axes: tuple[AxisId, ...]  # subset of output_data_axes
  output_dataset_schema
```

DataTransform 后仍存活的每根非 scan data axis 必须出现在 output_data_axes；batch_axes 只是其语义子集，不改变 ndarray 中 axis 的存在。repeat 是 Formal Scan Dataset 的权威 R 轴，不属于“可被 y transform 顺手 reduce 的非 scan axis”，ScanOutputContract 必须原样保留其 AxisId/coordinates/validity；若某个后续 Analysis 想对 repeats 求均值，它在冻结 ScanArtifact 上另建显式 ReductionSpec，不能改写原始 scan y。没有任何选择/降维时 committed_transform 可以为空，但 output contract 仍必须逐轴列出。最终 ScanArtifact 保存 scan point_axes、PointLayout、repeat axis和全部 output/batch axes，不能压成 `(repeat, data_points, data_dim)` 三个匿名长度。

Workbench 构造 Scan draft 时从 DatasetSchema、Selection snapshot 与独立 AnalysisPreset/Scan preset 构造 DataTransformSpec；ViewSpec 只提供当前可见 ROI/select 的候选提示，display mean/latest/sample/facet 不能复制。ScanOutputContract 根据目标 output axes、batch axes 与 reducer policy 重新派生并做 axis-total-coverage 校验；用户启动 scan 时才冻结 CommittedTransform、ScanOutputContract 与 schema fingerprint。scan 运行中不能因 latest、slider、ROI 或 panel 切换而改变 y 语义。

scan collector 是 TriggerKey/ScanCellKey -> `(R,P,*data_shape)` 和 PointLayout storage 的唯一 owner；只有它能按 ScanPlan materialize sparse/rectangular layout。`build_fit_problem` 是 fit batch packing/densify 的 owner。frontend、processor 和 artifact loader 都不得自行 reshape P、densify EXPLICIT layout或把 missing cell 与 invalid component混为一谈。

禁止取第 0 项、flatten 或自动平均 trailing axes。该禁令针对权威数据路径；LiveSweepMonitor 可以使用带标签的临时显示投影，但它不能产生成功 ScanArtifact。

### 14.7 Scan binding

```text
ScanContract.allowed = NONE | {SCAN_SLOT} | {API_SLOT} | {SCAN_SLOT, API_SLOT}
PulseScanRequest.selected = SCAN_SLOT | API_SLOT
```

一次 request 只能选择一种，绝不 fallback。MOT template 固定：

```text
allowed = {SCAN_SLOT}
slots = da_x, da_y, da_z
```

正常执行策略固定为 `AUTONOMOUS_STREAMED`，不改变模板语义；MOT/SCAN_SLOT 把 `da_x/da_y/da_z` 等完整 slot table 在 run 前冻结、编译并一次交给现有 FPGA scan engine，禁止逐 cell host mutation 或 API fallback。`API_SLOT_SEGMENTED_EXISTING` 只在 selected=API_SLOT 且设备 API 无法无缝更新时使用既有路径，并在 artifact 中显式记录 segment 边界；不得为了统一实现把 SCAN_SLOT 也切段。

API segmented模式的每个segment分别冻结camera/API settings、frame budget、TriggerKeys并完成segment EndAttestation；全部segment完成后再做aggregate count/coverage/lineage attestation。任一segment INVALID使整个Formal Run失败，不能只丢坏segment、拼接其余segment或自动重试后隐藏失败。

compiler/preflight 必须基于现有 bitstream 支持的 scan table 和实际有效 camera-trigger schedule 工作；不能要求 SINGLE_CELL_SCAN_SLOT、one-shot cell token 或新 RTL 寄存器。若现有实现对合法 SCAN_SLOT table 有 bug，先由 golden/model/真机证据确认，再按“修复既定设计”流程决定是否动 RTL。

## 15. Pulse bounded context 与 FPGA target

### 15.1 所有权

zlc_pulse 拥有：

- PulseDocument；
- logical/target IR；
- compiler；
- CompiledPulseArtifact；
- host/wire ABI；
- AXI/UART transport；
- remote server；
- virtual engine/model；
- RTL/build/sim。

它不拥有 MOT/readout 等实验 template、DeviceBinding、RunPlan、ScanCellKey、Qt editor 或 panel。neutral template 产生 PulseDocument/scan binding request，workbench editor 产生 PulseDocument command，二者都通过 pulse public API；pulse 不为调用方反向增加字段。当前只有 FPGA 一个生产 target，因此 compiler 可以是清晰的 concrete implementation；只有第二个可运行 target 出现且共享 IR 经过验证后，才抽 `PulseTarget` Protocol，不能先建立 target plugin/registry。

### 15.2 Authoring spec 与 resolved manifest

```text
HardwareProfile / BoardSpec
  可配置板级事实

Handwritten design constraints
  CDC、timing exception、设计约束

ResolvedBuildManifest
  本次 build 的完整解析结果与所有输入 digest
```

BoardSpec 生成 board pin/clock XDC、Verilog parameters、host ABI 和 Tcl/IP sizing。复杂设计约束可以手写，但必须有唯一 owner 并进入 resolved manifest digest。

### 15.3 两个派生 digest

从同一个 canonical ResolvedBuildManifest 计算：

```text
build_digest
  board/part/pins/toolchain/IP/RTL/constraints/build inputs

pulse_target_digest
  semantic ports/clock tick/slot widths/capacity/register+wire+compiler ABI
```

当前冻结 bitstream 的近期 runtime identity baseline 继续使用已经实现的 `image.build_fingerprint`/几何与 ABI 指纹握手：host、server image metadata 和 runtime现有回读必须一致，不一致禁止 upload/fire。Repository保存当前已部署 `.bit`、host/compiler版本、现有fingerprint与人工/发布记录的对应关系，但架构不声称旧bitstream暴露了并不存在的ROM字段。

`design_build_id + timing-signoff digest + programmed-bitstream content attestation` 是未来可选增强，因为把新字段放入ROM/USR_ACCESS需要重烧；它不是baseline、不是S4 gate，也不能为架构整洁主动实施。只有证据已经触发合法RTL/bitstream变更时，才随那次修复评估加入；若加入，仍需解决bitstream不能自嵌自身SHA-256的自引用问题，并由server content digest + runtime build id + manifest形成闭环。

canonical serialization 固定字段排序、整数宽度、编码和 line ending，digest 使用 SHA-256 并排除 digest 字段自身。

### 15.4 CompiledPulseArtifact

```text
CompiledPulseArtifact:
  source_digest
  pulse_target_digest
  compiler_version
  target_ir
  wire_image
  execution_forms: FIXED | AUTONOMOUS_SCAN_TABLE | API_SLOT_SEGMENTED_EXISTING
  scan_slot_schema + frozen camera-trigger schedule
```

内容寻址，不通过 sibling 文件名判断新旧，不重复嵌入 source table。

prepare/upload/fire 在冻结 bitstream 的既有协议上增加软件侧显式身份与校验，不要求新寄存器或RTL状态机：

```text
PREPARE(run_id, artifact_digest, frozen full finite scan table)
  host/compiler validate capacity、slot schema、camera-trigger schedule
  expand repeat axis into table order; require repeat_forever=False, scan_repeats=0
  verify existing image.build_fingerprint/geometry/ABI handshake
  upload through the current supported transport/protocol
  server records PreparedProgramRef(
    connection_generation, run_id, artifact_digest, table_digest
  ) in software

FIRE(PreparedProgramRef)
  server rejects stale connection/ref or mismatched currently prepared image
  camera is already armed for the whole scan budget
  current FPGA executes the complete autonomous table once

COMPLETE
  read existing DONE/status/scan-progress/error registers
  apply the already-established host tail/drain contract where required
  these existing facts feed EndAttestation; no per-cell receipt is claimed

SAFE/RESET/connection loss
  software invalidates PreparedProgramRef and follows current safe/reset path
```

PreparedProgramRef 是 host/server软件 guard，不伪装成硬件 one-shot token。它防止明显的旧连接、旧artifact和GUI状态漂移，但不能证明每个物理trigger沿；物理归属仍按§14.5的E0 envelope + frozen schedule + EndAttestation。FIRE前scan-slot/API-slot values和完整table冻结，运行中不从mutable GUI state读取。

expected trigger count/schedule 来自compiler对实际配置的唯一camera output channel、active polarity、clock mux、相邻高段合并、channel delay和全部合法slot values的确定性展开；camera channel不能同时配置为clk_enable。该schedule用于preflight间距与末端映射，不声称是运行时逐沿回读。

超过resident scan window时只使用当前bitstream/host已经支持并经过golden+真机验证的finite stream/ping-pong路径。preflight根据每chunk点数、最短point duration、transport/refill benchmark计算`chunk_playback_time >= refill_p99 + safety_margin`；无法证明时拒绝、减慢scan或缩小table，不能接受runtime underflow stall后仍称“无缝”。软件记录chunk seq/count/digest并在现有可读状态范围内fail closed，但不得假定RTL拥有尚未实现的CRC verifier、BANK_VERIFIED或新sticky fatal。若故障注入发现静默bank损坏/回放，这是现有RTL/协议问题证据，可触发最小修复评估；没有证据不重烧。

任何prepare/upload/identity validation失败都不得调用FIRE。重连改变connection_generation，旧PreparedProgramRef在软件侧失效；即使设备仍保留旧active image也不能由正式路径误触发。API-slot segmented路径每段同样冻结自己的values与artifact lineage。

### 15.5 硬件安全

- baseline先使用host/compiler的typed range/capacity/slot/schedule validation，以及现有bitstream实际暴露的DONE/status/error/fatal/safe/reset回读；contract kit只声明真机证实存在且语义明确的位，不能把目标寄存器写成既有能力；
- upload/fire沿用当前已工作的UART/AXI/JTAG协议。host/server通过PreparedProgramRef、connection generation、artifact/table digest防止软件层旧程序误触发；若transport error或readback异常，禁止提交并按现有safe/reset路径处理；
- RemoteSequencer通过现有软件/transport能力提供bounded timeout、cancel/abort和safe调用；共享backend的第二socket不冒充硬件独立性。无法确认safe时resource quarantine，但baseline不因此要求新增watchdog/SAFE寄存器；
- runtime identity近期只要求现有`image.build_fingerprint`/几何/ABI握手一致；design_build_id、timing-signoff ROM、programmed-bitstream digest不是baseline；
- 逐沿counter/FIFO、per-fire count、PHYSICAL_DONE、BANK_VERIFIED/RTL CRC等均不作为当前合同。只有E0/故障注入证实真实问题或现有RTL偏离既定设计时，才提出证据驱动的最小硬件修复；
- 若未来合法重建bitstream，build仍必须满足unconstrained paths=0、WNS>=0、TNS>=0，并审查generated clocks、CDC、IP property和critical warnings；这约束未来修复质量，不授权为架构偏好重烧。

## 16. Artifact 与持久化

### 16.1 Typed Ref 与 manifest

各 bounded context 不共享 universal ArtifactRef class：

```text
data:          FitResultArtifactRef
frontend:      FigureArtifactRef
pulse:         CompiledPulseRef
neutral_atom:  CaptureArtifactRef、CalibrationArtifactRef、ScanArtifactRef
```

每种 typed Ref 至少包含 kind、schema_id/schema_version、artifact content digest 和 repository namespace；它不是任意 filesystem path，也不直接嵌入 mutable Python object。各 owner context 拥有自己的 Repository/codec 与 canonical manifest。Workbench 通过本地 ArtifactDescriptor adapters 聚合展示，不把 descriptor 反向泄漏给 owner。

repository namespace 在 composition root 由 ExperimentWorkspace 显式绑定到用户可见的 RepositoryRoot；不读 current working directory、session 最近文件或隐式搜索路径。virtual/real 运行使用同一个 Repository API 和目录布局，测试只把 RepositoryRoot 换成临时目录。UI/日志始终能显示 artifact 实际写入的 root/ref，offline 流程要求用户选择 typed Ref 而不是“猜最近文件”。

```text
ArtifactManifest:
  kind + current schema version
  immutable metadata
  typed input/output lineage
  blob descriptors(digest, dtype, shape, byte_length, encoding)
  canonical manifest digest
```

shape、AxisSpec、PointLayout、validity、DataTransform/Fit/Scan contract 和算法/设备 fingerprint 都进入对应 manifest，不能只保存一个 ndarray 和文件名。

当前 artifact codec 不使用 pickle、object ndarray、FQCN import 或任意 callable 序列化；value object 使用显式 tagged schema，数组使用明确 dtype/endianness/order 的 blob encoding。受信任本地环境不等于允许格式不可移植或靠导入旧 Python 类才能读取。

跨包 artifact 若嵌入另一个 package 拥有的值对象，必须调用 owner 公布的 `to_canonical_tree`/`from_canonical_tree` 与 schema id，禁止在调用包重写字段、兼容 reader或 owner object digest。例如 neutral ScanArtifact 嵌 ValueSchema/DatasetSchema/CommittedTransform 时委托 zlc_data codec；Workbench manifest 只包裹 owner canonical bytes + digest。每个值对象到 canonical tree 的映射只有 owner 一个实现。

canonical tree 到 bytes、UTF-8、map key order、整数/float/NaN 表示、ndarray dtype/endianness/C-order、framing 与 digest algorithm 则全部委托 `zlc_storage.canonical`，不能由 data/pulse/frontend/neutral 手工实现四遍。`zlc_storage.canonical` 不认识任何领域 schema/type，也不能 import repository backend；它只是 content-addressed storage 所必需的纯 bytes 规则，不是 universal ArtifactRef/common domain。codec round-trip 必须保持 AxisId、coordinates、dtype、native integer data 和 validity，不允许为统一格式把 uint image 全部转 float。

F0 第一日即建立 cross-package golden/property contract：同一 primitive tree 在四个 owner 包中产生 byte-identical encoding/digest；嵌入 owner value object 时 outer manifest 使用 owner bytes/digest；字段重排、float edge、NaN、unicode、ndarray order/endianness 与版本变化均有向量。golden 不是允许四份实现漂移的补救，而是守卫唯一 encoder 和 owner codec delegation。

### 16.2 Atomic commit 与 load

各 owner context 的 typed Repository 委托 `zlc_storage` 的同一个 `BlobStore/ManifestCommitter` 实现 immutable content-addressed bytes、锁、fsync 与 atomic replace；owner Repository 仍负责 typed Ref、schema、canonical codec、lineage 和 load validation。`zlc_storage` 不 import AxisSpec、FigureArtifactRef、ScanArtifactRef 或任何领域类型，也不提供“万能 artifact repository”。commit point 是最后原子发布的 owner canonical manifest：

```text
write blobs to temporary names
-> fsync/close
-> verify digest + byte length
-> move blobs to content-addressed final names（已存在同 digest 可复用）
-> write/fsync manifest temp
-> atomic replace manifest commit marker
-> return typed Ref
```

temporary 与 final path 必须位于同一 Repository backend/filesystem。LocalFilesystemRepository 在绑定 RepositoryRoot 时 probe atomic replace、fsync/durability 与 file-lock capability；不满足合同的同步盘/网络盘不能承载正式 artifact，除非实现自己带事务语义的 Repository backend。绝不先写系统 temp 再跨盘 move 冒充 atomic。

manifest 出现前的 blob 都不是成功 artifact；baseline maintenance 只识别有 age/lock 证明的 stale temp，并提供 dry-run/list + 显式清理，不自动删除任意 unreferenced content blob。load 先验证 kind/schema/manifest digest，再验证所有 blob digest、shape/dtype/length 和交叉合同，全部通过后才返回 immutable/memory-mapped payload。index/最近文件列表只是可重建缓存，不是 artifact 存在性的权威。

每个 manifest atomic replace 是该 artifact 的 commit linearization point；Run 声明的 final result manifest 是其 SUCCEEDED commit point。CancellationToken 在 final replace 前最后检查：此前取消则删除 final temp、不发布 final manifest 并进入 CANCELLED/FAILED(cleanup error)；replace 成功后 final artifact 已是事实，后到 cancel 返回 `TOO_LATE_ALREADY_COMMITTED`，Run 完成 SUCCEEDED 并可记录 late-cancel warning，不能删除 artifact 后谎报取消。更早独立提交的 upstream artifact 遵守自己的 commit point，不随父 Run 回滚。

content-addressed blob 允许并发 writer 幂等复用；manifest publish 使用 digest/id 冲突检查，不能覆盖不同内容。只有 repository 规模证明 unreferenced blob 回收是实际问题、且所有 owner 能提供已验证 committed-manifest roots 后，才增加 maintenance-lock 下的 mark-and-sweep；storage 不自行解析产品 manifest。这样 baseline 先共享崩溃安全机制和 canonical bytes，不为尚未出现的多 backend/复杂 GC 建一套存储平台。

每种 artifact 只按自己的合同判断 commit：CaptureArtifact 只有在其 expected frames/schema/provenance 完整时提交，ScanArtifact 只有最终 collector/transform/commit 全部成功时提交。父 Run 的下游 processor 后续失败，不回滚已经完整提交的上游 CaptureArtifact；该 capture lineage 必须记录 parent run/result failure，且 UI 不把它显示成成功 ScanArtifact。若 capture 自身发生取消、gap、schema mismatch 或 hardware failure，只写 RunFailureRecord/诊断日志，不能留下名字像成功 artifact 的 partial 文件。Calibration live 路径同理：已完成采集是独立事实，因此后续 calibration 失败仍可保留 CaptureArtifact。

RunFailureRecord 记录 run_id、request/plan digest、最后 phase、primary error、cleanup/safety errors、resource/quarantine 状态、已成功提交的独立 upstream refs 与 event/log spans。它是诊断记录，不满足任何 Capture/Scan/Calibration result Protocol，不能被下游当数据输入。

### 16.3 Live Ref 与 Figure 保存

```text
LiveDataBlockRef:
  application_lifetime_token
  block_id
  revision

ArtifactDataRef:
  frontend FigureArtifactRef/content digest
  dataset path/id within manifest
```

LiveDataBlockRef 只在进程内有效，由 neutral runtime broker/snapshot store 持有明确 lifetime，不能被 frontend codec 序列化，也不能出现在 FigureDocument。Workbench LiveFigureBinding 用本地 LiveDatasetBinding 将它映射到 frontend DatasetId；render 时解析为只读 DataBlock snapshot。执行 Save 时必须把用户看到的确切 block revision materialize 到 frontend Figure Repository 的 immutable blob，再让保存后的 FigureArtifact dataset descriptor 指向 frontend-owned ArtifactDataRef。保存期间 live revision 继续变化不影响已冻结 snapshot，也不能偷偷保存“最新”revision；FigureArtifact 不直接嵌入 CaptureArtifactRef/ScanArtifactRef 等 neutral 类型。

FigureArtifact 保存 ViewSpec、当次 EvaluatedFigureData 的 input revision/resolution records、Selection snapshot、layer/model/fit lineage 和所引用 dataset digests；重开默认复现保存时的 concrete selection，用户明确切回 dynamic latest policy 后才重新解析，不进行新的 axis auto 推断。若用户只保存 workspace layout 而不 materialize live data，文件必须明确标为 session-only workspace，并在数据 lifetime 结束后显示 missing binding，不能假装是自包含 FigureArtifact。

### 16.4 当前 schema 与离线转换

正式 runtime 只接受当前 schema。历史转换使用独立离线工具。

转换工具读取明确指定的旧 artifact，输出新的当前-schema artifact，并记录 source digest、converter id/version 与转换 TransformRecords；它不覆盖原文件，也不把 legacy reader 链接进 runtime/GUI。无法无歧义恢复的 axis、unit 或 provenance 必须停止并要求用户提供映射，不能按 shape 猜。

## 17. 性能约束

### 17.1 Camera exact queue

CaptureSession queue 禁止 list `pop(0)` 的 O(n²) 路径，使用 deque/ring，实现：

- enqueue/dequeue 摊销 O(1)；
- bounded capacity；
- exact overflow/backpressure 显式；
- monitor overwrite/missed count；
- exact 与 monitor fan-out 不复制不必要的大帧。

PerformanceBudget 计算物理保留字节而不是简单 `payload × subscriber`：分别统计稳定 BufferId 去重后的 unique retained buffers、每 edge queue/ref overhead、monitor front buffers、builder chunks、in-flight Owned/BorrowedSnapshot、processor output和render front buffers。共享 immutable frame只计一次 payload但每个引用有自己的生命周期开销；monitor虽不参与 exact ack，也必须有独立上限，不能靠持有 Python ref阻止大帧释放。

### 17.2 Scan compile

使用：

- expression 预编译；
- typed contiguous arrays；
- vectorized expansion；
- 一次性 validation；
- source document 与 wire artifact 分离。

优化后 target IR/wire image 必须保持等价，时间和内存对点数近似线性。

Formal Scan性能预算另包含两项硬门：展开repeat后的`total_frames × frame_bytes`必须同时满足qCMOS finite buffer、driver可分配容量、exact retention/consumer预算；超过resident FPGA scan window时，最坏chunk playback时间必须覆盖实测refill p99+margin。任何一项不能证明就preflight拒绝或降低scan rate，不能依靠运行时stall、内存swap或丢帧维持表面可运行。

### 17.3 UI 与 analysis

- monitor 由 UI refresh rate 限制，不降低 acquisition rate；
- 独立 panel view-evaluation latest-only；同一 coherence group 由 board evaluator 选同一 JoinKey/revision并原子 present，Fit executor 队列 bounded，三者不互相阻塞；
- revision coalescing；
- stale fit/render result 丢弃；
- exact pipeline 的必要 StreamProcessor/DatasetBuilder 不与可丢弃 UI fit 共用一个拥塞队列；
- `suggest_view` 只遍历 axis metadata，复杂度 O(axis count)，不触碰大型 values；
- ViewSuggestion 可按 `(schema fingerprint, ViewIntent, Selection revision, preference revision)` 缓存；
- Select/Transpose/Stack 优先产生只读 view，只有 reduction、driver buffer ownership 或持久化边界才复制；
- 显示用 mean/latest 不复制或覆盖权威 DataBlock，缓存键必须包含 input revision 与 ViewSpec digest；
- EventSpanRef 的 count/ordered_digest 随 exact sequence 增量更新，不为每个累计输出复制历史 event_id。

display decimation/downsampling 只能作为带标签的 render policy，不能改 DataTransformSpec、FitSpec、ScanOutputContract 或 artifact data。用户缩放/导出时 renderer 从同一原始 snapshot 重新取样，不能把屏幕像素缓存当权威数据。

### 17.4 Profiling 与性能 gate

性能结论必须来自相同 workload 的 profiler/benchmark，不用刷新频率下降或数据丢弃掩盖瓶颈。每个优化记录：baseline、调用图/分配热点、改动后结果、数值/视觉等价证明。

固定 benchmark matrix 至少包含：

- camera 不同 frame bytes、repeat、burst 与 exact+monitor fan-out；
- 1D/2D/multi-axis scan 的 P 扩展；
- DataPatch/immutable snapshot 的时间与 peak RSS；
- StreamProcessor chain 深度、fan-out、typed record bytes 与 DatasetBuilder materialization；
- FitResultBatch 的 batch size 与 model cost；
- artifact streaming write/load 与 digest 校验；
- 当前 TaskConsole 与目标 WORKER_RASTER_LIVE 多 panel board 的 ingest-to-visible、compose/present、GUI event latency、coherence mismatch 和 stale queue length；迁移后不得以回到 GUI compose 换结构纯洁。

机械 gate 使用 scaling 与配置预算，而不是拍脑袋的单机绝对秒数：queue/patch 摊销 O(1)，scan compile/journal/artifact bytes 对数据量近似 O(N)，内存不随已 ack history 无界增长；p95/p99 latency 和 peak bytes 必须低于目标环境 PerformanceBudget/WorkbenchProfile 声明预算。任何超预算都保存 profile artifact，先定位 producer、copy、lock、solver 或 render 热点再决定优化层。

## 18. 测试体系

### 18.1 Package tests

各 bounded context 拥有自己的 unit/contract tests，根仓库只保留 architecture、cross-package integration、E2E 和 performance。

### 18.2 必须保留/新增的合同

Data：

- Value event 只携带 `(*data_shape)` 与 ValueSchema；DataBlock 只携带 `(R,P,*data_shape)` 与 DatasetSchema，普通 stream edge 拒绝 DataBlock/DataPatch；
- DatasetBuilder 是唯一 event -> dataset 边界：TriggerKey/ScanCellKey 到 `(R,P)` 映射、duplicate/missing key、ValueSchema mismatch、values+validity journal 原子更新均有合同测试；
- StreamProcessor 每次 invocation 返回一个 frozen typed record；同一 record 的字段共享 key/provenance，字段不同 cardinality/key/lifecycle 时静态拒绝并要求拆节点；
- 任意 point/data axes；
- scalar 与长度一 axis；
- arbitrary-schema property tests：AxisId 唯一、coordinate/size、shape 与 axis coverage 不变量；
- 不同 revision/generation domain 不能互相比较、赋值或通过裸 int 混用；
- 同一 Definition 的 virtual/real/不同 run AxisId 稳定，派生 AxisId 对相同 transform 确定；
- PointLayout RECT_C/RECT_F/EXPLICIT sparse mapping round-trip，public path 不假设 P=product 或自行 reshape P；
- ValueSchema/DatasetSchema fingerprint 分离：前者包含 data axes、dtype/unit、ValidityContract，后者另含 repeat/point axes 与 PointLayout；两者都不包含 renderer、ViewIntent 或已安装 reducer 列表；
- canonical unit conversion、CoordinateFrameId mismatch 与显式 CoordinateTransform lineage；
- 多轴 ROI/integrate contract 同时验证 input axes、output axes、unit 与 validity；
- 未知 reducer id、axis 不兼容或缺少领域 CalibrationArtifact 的 reduction 失败；
- native uint/image + partial validity；
- CellValidity 与按具名 axis 广播的 ComponentValidity；`(group,site)` dead-site mask 在 reduce/fit/histogram/meter 中一致传播；
- validity mask axis/shape 不匹配失败，NaN 不能替代 integer/bool/component validity；
- 发布 DataBlock/array write-protected，driver buffer reuse 不改变已发布 snapshot；
- DataPatch block/base/result revision、schema fingerprint、重复 target、exact 重写 valid cell 与 values/validity 原子性；
- DataPatch/snapshot 近线性且无每点全 block copy；
- 每 sample 只发 DatasetProgress/dirty coverage，不把完整 DataBlock/DataPatch fan-out；snapshot 按请求 revision/slice 解析，EOS final freeze 总复制近似 O(final bytes)；
- 非法隐式 reduce/anonymous flatten 失败；
- DataTransformSpec 不包含显示 binding，ViewSpec 不可传入 neutral runtime；
- 同一 schema + ViewIntent + Selection 得到确定性 ViewSuggestion/ViewSpec；
- suggestion 不读取 values，不按 rank/singleton/axis 顺序猜 role；
- suggestion 不修改 DataBlock，所有有损 binding/operation 均可从 ViewSpec 派生并出现在 panel 摘要；
- latest/navigation binding 每个 input revision 解析为带 coordinate record 的 EvaluatedFigureData，display navigation 不能进入 CommittedTransform；
- `NEEDS_INPUT` 不能转换为 CommittedTransform；
- CommittedTransform 在 schema fingerprint 变化时失效；
- ViewSpec display mean/latest 不能被 commit；Fit/Scan draft 不读取任何 ViewSpec 权威字段，并从各自 spec/policy 重派生 reduction/batch/output axes；
- ViewSpec 类型中不存在 `authority_seed`/CommittedTransform；临时 ROI/select 只有从 Selection snapshot 经领域 draft 显式重建后才可 commit；
- save/load 后 ViewSpec、EvaluatedFigureData resolution records、selection、revision 与 lineage 一致；
- repeat 在 IMAGE/CURVE/HISTOGRAM 等 presentation ViewContract 下遵守各自显示规则；authority-side FitDraft 从 FitPolicy 独立派生，禁止把 display mean 或全局 mean 带入 fit；
- fit 的 surviving axes 必须被 fit/batch/component 完整覆盖，不对 trailing axes `nanmean`；
- fit identity path 只用 schema fingerprint，不制造空 CommittedTransform；
- FitResultBatch 保留 batch AxisSpec + RECT_C/RECT_F/EXPLICIT BatchLayout；稀疏 missing cell 不等于 fit failure，单 batch failure 不破坏其它结果，整体 schema/model 错误明确失败；
- gridplot 的每个可见 cell 与 FitResultBatch batch coordinate 一一对应，component-invalid cell 只产生该 cell 的 typed failure；
- FitModel input/output/parameter unit mismatch 在 solver 前失败，结果参数 unit 正确。

Architecture：

- zlc_data 不导入 frontend/neutral/pulse/workbench 或 storage repository/backend，只可导入 zlc_storage.canonical；AxisSpec/ValueSchema/DatasetSchema 不引用 ViewIntent/ViewContract；
- frontend.figure 可以依赖 zlc_data 的 DataTransformSpec/Selection，但 ViewSpec/suggestion policy 不进入 zlc_data 或 neutral_atom；
- zlc_data 的 FitSpec/FitProblem/FitResultBatch 不包含 application/neutral InputRef/EventRef/ArtifactRef；
- FitSpec/BoundFit/fit_analysis/FitResultBatch/Selection 只在 zlc_data 定义；DataFigure/selector controller 只在 frontend 定义；neutral_atom 中任何 `FitProcessor`、`FitOperator`、`FitAnalysisDefinition` symbol 或复制 solver/model schema 都由 architecture test 拒绝；
- zlc_data `BoundFit` 不引用 neutral `AnalysisStep`/DatasetInputSlot；只有 neutral-side generic adapter 执行 BoundFit+slot -> step 映射。AnalysisStep 对 data fit 与 neutral calibration 使用同一 hosting contract，但算法 id/codec/result owner 仍属于 zlc_data/neutral 各自 bounded context；baseline 不存在 DataAnalysisDescriptor/Program registry；
- zlc_data solver 不含 Qt/thread guard 注册、executor、环境变量线程策略或 callable/FQCN 序列化；frontend/neutral 各自在 hosting contract test 中证明 fit 不运行于 GUI/I/O lane；
- ViewContract 只有一套，plot/render 不复制 role/repeat 判断表；
- DefinitionCatalog 只由显式 imports 组装，重复 id fail，禁止 package scan/global registry；
- catalog Definition callable 无 hidden closure/device/session/global mutable dependency；
- PipelineSpec 编译成唯一顶层 RunPlan，节点不能 start child run 或自行拥有 terminal state；
- bind claim superset 完整，preflight/execute 尝试新增 ResourceKey 失败；
- 同一物理 DeviceIdentity 的 Workbench/notebook/standalone/remote client 只有取得 DeviceControlLease 的一个控制 owner；两个进程各自的 ResourceArbiter 不能同时 Acquired；
- S0.5 legacy start 必须经过 LegacyRuntimeFence；旧 thread 未真实退出/safe 前新 Run无法取得同一 ResourceKey，所有 direct LogicNode.start 入口被机械禁止或限定为无硬件测试；
- StreamProcessor/Analysis callable 不读 global RNG/time/config，显式 seed/config 可重放；
- BEST_EFFORT_MONITOR 只能是 monitor 叶子，其失败不 abort exact，且不能流回 authority/artifact；
- frontend FigureDocument/codec 不引用 neutral LiveDataBlockRef；
- frontend EvaluatedFigureData 不引用 BorrowedSnapshot/release token；opt-in zero-copy 的 lifetime pairing 只在 workbench WorkbenchRenderMessage。

Stream：

- reservation 在 fire 前；
- history=8 + burst=20 在 reservation 下完整消费；
- 无 reservation 产生 Gap；
- schema generation 改变终止旧 cursor；
- monitor rebind 新 generation 创建新 block_id，旧 evaluation/borrow/CommittedTransform 不可复用；
- 多 reservation 独立 release；
- event/byte budget 不足 preflight 拒绝；
- total expected events 与 max-inflight retention 分开计算；ack 后 retained bytes 下降，不可 backpressure burst 使用最坏 backlog 而非平均率；
- stale DeviceCapabilitySnapshot/output schema mismatch 在 arm/fire 前拒绝；
- StreamProcessor cardinality/byte-bound 无法证明或 formal path 含未解释 filter 时 preflight 拒绝；
- StreamProcessor output contract 依赖首帧数值或运行中改变 record fields/axis 时不能编入 formal pipeline；
- continuous Measurement 只能使用 MonitorStream；exact request 必须有限且可完整 reservation；
- ROLLING_MONITOR 固定容量、报告 overwrite/missed/expired；冻结为 formal input时产生新的 owned finite revision并记录 coverage，不能冒充完整 capture；
- 单个 typed output record 的 ack 只在 publish 与所有 required downstream 接收成功后；不同 cardinality/key/lifecycle 的结果必须建独立节点，不能伪装为一次多-output transaction；
- exact/monitor 同源 event_id；
- EventSpanRef digest/count 等价于显式 ordered events，lineage 不随累计输出 O(N²) 增长；
- driver buffer 重用与 monitor overwrite 不破坏 exact payload lifetime；
- coherent monitor 永不混 shot，INDEPENDENT_LATEST_MONITOR 不可用于相关 expression；
- EXACT_KEY/coherent monitor 对 join_key type/schema mismatch 或 missing key fail closed；
- 独立设备 ZIP_SEQUENCE 被拒绝；
- E0 CameraExternalTriggerEnvelope在每个批准的ROI/exposure/readout工作点验证一触发一帧、delivery order、frame/camera stamp连续性、timestamp语义、buffer行为和最小安全trigger间隔；超出envelope不能声明Formal capability；
- E0报告保存样本量、观察loss/reorder率与统计上界；未达到PI批准的样本量/上界/margin不能发布envelope；
- preflight从编译后有效trigger schedule计算所有相邻间距并要求`interval >= camera minimum + configured margin`；边界内/外、delay/polarity、重复trigger edge均有测试；
- repeat axis确定性展开进finite table；Formal program强制`repeat_forever=False, scan_repeats=0`，任何`scan_repeats>0`/cursor-wrap stop在compile/preflight被拒绝；repeat/point/TriggerKey round-trip覆盖多repeat；
- qCMOS一次arm完整scan frame budget，`total_frames/bytes`同时通过camera/driver buffer与exact reservation预算；超容量在arm/fire前拒绝；frame[i]只在匹配E0 envelope时映射frozen TriggerKey[i]；
- 超resident table的chunk playback time与refill p99+margin通过preflight；故意慢refill必须在fire前拒绝，不能以underflow stall通过“无缝”测试；
- EndAttestation比较existing FPGA completion/progress、计划/现有回读trigger总数、camera produced count、frame/camera stamp、timestamp容差、DatasetBuilder coverage和EOS；任一不符整run INVALID且无ScanArtifact；
- 注入drop/reorder/duplicate/counter reset/metadata gap/short read使整epoch失败；系统不声称能定位具体point，也不声称能检测metadata仍合法的等量loss+extra抵消；该剩余风险在artifact proof_class中可见；
- 所有scan数据在EndAttestation前为PROVISIONAL；只有`ORDERED_END_ATTESTED_RUN` VALID后才能commit；
- INVALID attempt保存RunFailureRecord且默认不自动重试；显式RetryPolicy有有限次数、独立run_id/lineage，最终artifact引用所有失败attempt；
- API_SLOT_SEGMENTED_EXISTING每segment单独EndAttestation且全局aggregate attestation；任一segment失败整run不提交；
- TAGGED/TIMELINE若相机现有metadata经E0证明可作为更强软件关联能力可使用，但不要求FPGA逐沿FIFO；HardwareTriggerStamp只在证据触发未来RTL修复后测试。

Thread/UI：

- blocking I/O 中 cancel；
- out-of-band interrupt 可使被占 I/O lane 进入 cleanup；
- join timeout 不允许 restart/release/destroy，safe failure 转 ResourceQuarantined；
- synchronous run 的 RunStillCancelling 保留可查询 RunHandle/claims；
- reset/reconnect 未通过 health/safe check 不能解除 quarantine；
- 新 connection generation 在 UNVERIFIED handshake 完成前不可 acquire，应用重启不洗白 sticky fatal；
- 每个hazardous run epoch首次arm/fire前HAZARD_ACTIVE write-ahead（非逐cell fsync），crash-after-fire-before-safe重启后恢复QUARANTINED_PENDING_VERIFY；只有现有safe/status或recovery验证 + 用户确认追加RESOLVED；切换artifact repository root不洗白machine/device safety ledger；
- schema-affecting reconfigure 建新 generation/block_id，旧 cursor/view/fit terminal；每个 accepted ControlTopic revision 恰有一个 terminal ack；
- I/O lane 满载下 finite/control command bounded-wait，monitor 不能 starvation，safety interrupt 不经普通队列；
- stale queued result 不更新 UI；
- retained revision N 的 OwnedSnapshot 在 builder ingest N+1 后 digest/bytes 不变；mutable builder read-only view被 contract拒绝；SnapshotExpired 不返回 latest；
- 默认 owned snapshot 无 lease；opt-in BorrowedSnapshot 在 stale discard、queued cancellation、panel close、monitor overwrite、artist/front-buffer replacement和shutdown后无 token 泄漏；
- QObject affinity；
- GUI_ARTIST 的 Figure/artist 唯一 GUI owner，WORKER_RASTER_LIVE/HEADLESS 的 Figure 唯一 worker owner；普通 queue 不跨线程传 QWidget/Figure/artist；
- S0.5 SerializedLegacyAggBridge 的 ownership handoff 成功前 GUI 不碰 Figure，timeout fail-closed，shutdown join 前不 destroy；Z0 时该 allowlist 为 0；
- Legacy Agg worker 对 QObject-affine draw/update/connect/selector API 机械拒绝；只允许独立 FigureCanvasAgg/allowlisted Agg-only path；
- interactive transform/reduction 在 FigureEvaluator worker；GUI_ARTIST 消费 revision-matched layer DTO，WORKER_RASTER_LIVE 消费 immutable BoardFrame/front raster；
- 长 interactive fit 不阻塞 view-evaluation，view/Fit 满载不影响 exact StreamProcessorWorker/DatasetBuilder；
- headless raster 不泄漏 Figure；
- worker raster + Qt overlay 的 ViewportTransform round-trip、revision mismatch 丢弃、ROI 事件真实改变 data-space Selection；
- 同一 coherence group 的多 panel 只在完整 CoherenceStamp（run/epoch、typed JoinKey、DatasetRevisionRef、document/selection revisions）一致时 board-atomic present；跨 generation 的相同裸 key/revision 不相等；独立 monitor不伪装 coherent；
- fit 不在 GUI thread；
- Definition.bind/pipeline validation/pulse compile 不在 GUI thread 且不持有 hardware claim；
- notebook Experiment facade 的 virtual connect -> capture -> 1D fit -> save 保持少量语句，headless 无 render extra 仍可完整运行；
- headless fit.save 返回 zlc_data FitResultArtifactRef且不加载 frontend.render；figure_document 只需 frontend.figure，只有 figure()/GUI 需要 render/workbench extra；
- Experiment.readout 的 ReadoutBindingKey -> CalibrationArtifactRef 可见可设；多 camera 不串 ref，convenience request 在构造时冻结 binding/ref/model，运行时清空/切换 facade pointer 不改变已启动 Run；
- panel 的视图摘要与实际 render 的 ViewSpec/EvaluatedFigureData 一致，权威操作摘要与执行的 CommittedTransform/FitProblem/ScanOutputContract 一致；
- 一次 Fit/Run 点击冻结 revision，后续 selector 变化不污染进行中的结果；
- EditorSession base revision 冲突拒绝 last-write-wins；
- shutdown 真实入口等待 RunHandle/worker/device acknowledgement，销毁后 queued result 被 lifetime token 拒绝。

Artifact/Calibration：

- commit 每个步骤 crash injection：manifest commit marker 前 load 永远看不到成功 artifact；
- cancel/manifest replace race 在 commit point 前取消、之后 TOO_LATE/SUCCEEDED 线性一致；
- RepositoryRoot atomic/durability probe 失败或 temp/final 跨 backend 时拒绝正式 commit；
- blob/manifest digest、dtype/shape/length 任一损坏均 fail closed；
- zlc_storage.canonical primitive golden/property vectors跨 data/pulse/frontend/neutral byte-identical；跨包嵌值对象必须委托 owner canonical tree/bytes；
- LiveDataBlockRef codec 拒绝，Figure Save 冻结屏幕对应 revision 而非随后 latest；
- failed/cancelled scan 只产生 RunFailureRecord，不产生 ScanArtifact；
- Task mid-run frame/map 通过正式 LiveDatasetSlot/DatasetRevisionRef 显示；删除 TaskOutput 后不丢功能，也不存在第二 mutable buffer；
- live calibration 先生成 CaptureArtifact，之后与 offline ref 走 byte/contract-equivalent 算法路径；
- FrameContract/SiteMap/model mismatch 拒绝 occupancy；
- required calibration model 任一失败不提交 CalibrationArtifact；
- converter 生成新 artifact 和 lineage，不覆盖旧文件、不进入 runtime。

Pulse/FPGA：

- duplicate/out-of-order/gap/EOS incomplete 均 safe；
- upstream exact edge gap 使正式 scan 失败；
- multi-axis TriggerKey -> ScanCellKey -> PointLayout round-trip，non-scalar y axes 完整且 transform explicit；
- MOT 无 API fallback；
- Pulse preview 不制造 frontend -> FPGA import；
- build/target digest mismatch fail closed；
- 现有`image.build_fingerprint`/几何/ABI handshake mismatch fail closed；测试不得要求当前bitstream不存在的design_build_id/timing ROM；
- partial/oversized upload、host digest/table mismatch、旧connection generation或旧PreparedProgramRef均不能进入正式FIRE；不声称硬件one-shot token；
- resident/streamed scan只测试当前bitstream实际支持的seq/count/status语义；不存在的RTL CRC/BANK_VERIFIED/sticky位不写fake测试；
- reconnect generation使软件PreparedProgramRef失效，SAFE/RESET按现有协议验证；
- 主RPyC wait_done/backend `_io_lock`/transport阻塞时现有timeout/cancel/abort/safe行为有真机故障注入；无法确认safe则resource quarantine，但测试不要求新增watchdog/独立SAFE寄存器；
- host/model/RTL golden byte-identical；
- UART oversized frame 零提交；
- qCMOS contract kit保存nFrameCount/framestamp/camerastamp/timestamp真机语义、工作余量内长时间零丢帧/乱序证据和可复现报告；
- qCMOS Formal preflight readback/freeze trigger source/polarity、sensor/global exposure、ROI/binning、exposure/readout mode，并验证完整schedule margin；整run drain后的pending/late/extra frame阻止commit；
- current DONE/tail行为按现有contract测试；Formal EndAttestation只有在保守drain后读取最终camera/FPGA状态，不把DONE重新定义成不存在的PHYSICAL_DONE；
- 当前`scan_repeats=K`可能多发下一sweep point的路径有回归测试并被Formal compiler明确拒绝；finite single-pass大表走现有同步refill后到STATUS_DONE；
- HardwareTriggerStamp/FIFO/trigger-return/新ROM测试默认不存在；只有证据批准RTL变更后才加入对应package+真机gate；
- Measurement/StreamProcessor/DatasetBuilder 全链传播 provenance_epoch_id；EpochIntegrityState 未 VALID 时 formal sink 即使已有全部 y 也不能 terminal success/commit。

### 18.3 测试边界

Public contract/E2E 不依赖私有结构。Package-internal unit test、RTL/model state-machine test 可以检查必要内部状态，但不能用内部断言代替行为合同。

Correctness tests 使用 deterministic TestClock/Scheduler，不依赖真实 sleep/qWait/processEvents。

### 18.4 安装与文档 gate

- zlc_storage.canonical 可在无 repository backend 环境 isolated import；zlc_data 在无 Qt/Matplotlib/neutral/pulse 环境只依赖该纯模块并 isolated import；frontend.figure 在无 PyQt/Matplotlib backend 环境 isolated import；
- notebook composition 在 headless 安装可 connect/run/save artifact；安装 `[render]` 后才暴露 DataFigure 路径，安装 `[workbench]` 后才暴露 GUI launcher；
- workbench 安装显式拉取 frontend `[qt,render]` extras；
- pulse model/compiler/FPGA host/server isolated headless install，不加载 neutral/frontend/Vivado environment；
- neutral_atom 不依赖 Qt/workbench；
- launchers、notebooks、presets、config 使用新 public surface；
- virtual tutorial 在 CI 执行；
- hardware tutorial compile/lint/dry-run；
- API、schema、enum、hardware tables 从 owner 生成。

Workbench E2E 必须从真实 launcher/composition root 驱动用户路径：Add Panel -> Setting -> 选 Measurement/Processor -> Start -> selector/repeat -> Add Analysis/Fit -> Save/Load -> Stop/Close；不能用 demo fixture、直接 poke controller 内部或手工调用 `_tick` 代替。PulseGUI 同样从真实入口执行 Edit -> Preview -> Scan/prepare -> cancel/safe。三档 DPI 截图与交互事件验证视觉/操作不退化。

virtual 与 real adapter 运行相同 Task/Measurement bind、PipelineSpec/RunPlan、artifact repository 和文件夹流程，只替换最低层 Port；adapter contract kit 使用生产 Port 的真实属性/方法名，并覆盖`ORDERED_END_ATTESTED_RUN`、CameraExternalTriggerEnvelope、DCAM frame metadata、timeout、buffer reuse、disconnect与health recovery。virtual adapter可以注入drop/reorder验证整run invalidation，但不能用fake队列代替E0真机确定性证据。

## 19. 实施路线：证据门、并行轨与纵向切片

实施不按“先写完所有 core，再写 runtime，再写 UI”的横向层次推进。每个纵向切片必须同时完成：

```text
producer/adapter
+ event/runtime/materializer
+ analysis或consumer
+ notebook/UI/artifact
+ contract/E2E/performance evidence
+ 删除该 use case 的旧实现、alias、fallback、测试和文档
```

同一个 use case 不允许双写、双读或自动 fallback；尚未迁移的其它 use case 可以暂时停留在旧实现，但不能通过 bridge 污染新合同。S0.5 只允许 host/catalog/render/resource containment，不允许把旧 Hub/LogicNode/metadata vocabulary 适配成新 event/data/runtime 类型；这叫迁移隔离，不是领域兼容层。旧 panel 可由 LegacyPanelHost 逐项托管，旧 hardware workflow 必须经过 LegacyRuntimeFence 使用同一 ResourceArbiter/DeviceControlLease。共享 primitive 只在首个消费它的切片中建立最小正式版本；后续若无第二用例，不提前泛化。

依赖关系是：

```text
Evidence Gate E0 -> Foundation F0 -> S0.5 Workbench host -> S1 Camera -> S2 Analysis -> S3 Readout
        |                                                                    |
        +------ H1 Pulse boundary + current-bitstream golden/bring-up --------+-> S4 Formal autonomous scan
        |
        +-- only measured camera failure or proven RTL bug --> H2 evidence-driven hardware repair review (optional)

S1/S2/S3/S4 -> S5 remaining use cases -> Z0 zero-residual audit
```

本设计的开工判定分两层，不能把“架构可开工”与“真机 Formal Scan 已 ready”混成一个 GO：

- **GO NOW**：E0 只读证据、import-DAG ratchet、F0 绿地 data/storage/stream/catalog、S0.5 Workbench 三 surface 与三个 legacy containment bridge；这些工作不需要删除现有生产链，可立即开始。
- **GO PER DEPENDENCY-CLOSED SLICE**：S1/S2/S3 只有在 LegacyRuntimeFence 生效、consumer matrix确认且每个切片不提前删除共享 producer时开始。
- **GO FOR S4 AFTER CURRENT-HARDWARE EVIDENCE**：E0 CameraExternalTriggerEnvelope、compiled schedule margin、现有bitstream golden/真机 completion/progress语义、S1 exact acquisition和S3 processor/materializer通过后，直接使用AUTONOMOUS_STREAMED；不等待新RTL。
- **HARDWARE CHANGE NO-GO BY DEFAULT**：HardwareTriggerStamp、新ROM attestation、per-fire counter/PHYSICAL_DONE、trigger-return、watchdog或RTL CRC均不得由路线图自动启动。只有E0测得工作余量内真实丢帧/乱序，或现有RTL bug/设计偏离被证实，才进入H2评审。

### E0：先取得会改变架构选择的证据

在迁移 data kernel/runtime 前完成，只读探测与 benchmark 可以独立提交，但不引入临时 runtime：

1. 真qCMOS contract kit对每个目标ROI/exposure/global-exposure/readout/trigger模式建立CameraExternalTriggerEnvelope：保存`nFrameCount/framestamp/camerastamp/timestamp`语义、reset/rollover、buffer行为；在多轮长scan中测量“一触发一帧、按序、无漏”的工作区间、最小安全trigger间隔和所需margin。观察到任何loss/reorder先收紧工作点/余量并复测；若在批准工作区间仍发生，才产生H2硬件变更候选。
2. 对现有 1D rolling、2D qCMOS live、gridplot/多 panel board 做 ingest-to-visible、GUI event latency、copy、compose 与 board coherence profile；据此确认 GUI_ARTIST、WORKER_RASTER_LIVE 的分界、front-buffer 预算和 S0.5 legacy bridge 的临时覆盖范围。
3. 对目标 RepositoryRoot（包括同步盘实际目录）执行 atomic replace/fsync/lock/crash probe；不满足合同就选择合格本地 root，而不是弱化 commit 语义。
4. 固定 camera queue、journal/patch、fit batch、scan compile、artifact 和 UI benchmark matrix；保存基线 profile artifact。
5. 建 import-DAG ratchet，立即禁止新增 data -> 其它 bounded context、frontend -> neutral/pulse、pulse -> neutral/frontend/data 的反向边。

E0报告是CameraExternalTriggerEnvelope、preflight margin、capability handshake与PerformanceBudget的输入；没有保存的真机报告不能启用Formal qCMOS scan。报告必须包含样本规模、持续时间、工作点、最坏间隔、观察到的loss/reorder率及其统计上界、设备/driver版本；当观察为零时也用样本量给出可解释的upper confidence bound。PI在报告中批准最大可接受率、最小样本量和margin，不能只写“看起来没丢”。

### F0：最小架构脊柱

只建立后续 S1 立即消费的正式能力：

1. safety spine：RunController/RunHandle、ResourceArbiter、跨入口 DeviceControlLease、BoundDevice/owner I/O lane、真实 termination acknowledgement 与 machine/device级 write-ahead quarantine journal；先用阻塞 fake/virtual camera 证明 join timeout 不释放 claim/不允许 restart。
2. zlc_data：AxisId/AxisSpec、ValueSchema/DatasetSchema、Value/DataBlock、Validity、PointLayout 和 canonical codec；它是这些通用数据类型的唯一 owner。
3. zlc_storage：canonical primitive encoder/digest、BlobStore/ManifestCommitter/atomic probe；各 owner 保留 typed Repository/schema codec，并从第一天用 cross-package golden/property test 锁定 canonical bytes。
4. neutral stream：Envelope/journal/cursor/reservation/ack/generation 与 DatasetBuilder/DatasetProgress；不发布累计 DataBlock。
5. explicit DefinitionCatalog、PipelineSpec -> flat RunPlan compiler；此时只支持 S1 所需的 Measurement、DatasetBuilder 与 sink，不预建递归 plan 或通用 workflow DSL。
6. camera exact queue 改为 O(1)，明确 driver buffer ownership。

F0 只有 contract/unit tests，不作为长期“基础设施里程碑”单独宣称产品完成；完成标准是立刻进入 S0.5/S1。

### S0.5：先建立可承载纵向替换的 Workbench 壳

当前 `task_console.py/live.py/pulse_gui.py` 是共享一个 console-wide RenderLoop 的巨壳；如果 WorkspaceModel/PanelHost/render surface 全拖到 S5，S1 无法替换 camera panel 后删除旧路径。S0.5 只建立迁移宿主，不迁全部领域逻辑：

1. 建立最小 Workbench composition、WorkspaceModel、BoardController、PanelHost 与 RunHandle/status binding；不复制 TaskConsole 业务规则。
2. 交付 GUI_ARTIST、WORKER_RASTER_LIVE/BoardFrame 和 headless export surface 的接口与真实性能测试。
3. 建立 `LegacyPanelHost/CatalogRouter`、`LegacyRuntimeFence` 与 `SerializedLegacyAggBridge` 三个窄桥；旧 panel 可逐项隐藏/替换，旧 LogicNode 的所有 start/stop 先走共同 resource claim，Figure handoff timeout fail-closed。若无法证明旧 workflow 的占用设备或真实 thread termination，则同一 ResourceKey 的 legacy/new mode 互斥，不能嵌入并发运行。
4. 后续每个 dependency-closed 纵向切片以新 panel/controller/runtime 替换对应旧岛；已迁 use case 立即删除自己的旧路径，但共享 producer/algorithm 只在最后一个旧 consumer 迁走时删除。三个 bridge 都有删除期限，不是 public API，Z0 必须为 0。

S0.5 解决的是“新切片住在哪里”，不是预先重写 9000 行 UI；Setting/Edit/catalog 的完整迁移仍随实际 panel/use case 发生。

### S1：Camera -> Value event -> Dataset -> live/save/notebook

1. 迁 CameraPort、BoundMeasurement、CaptureSpec/CaptureSession 与 owner I/O lane。
2. 一帧只发布 `CameraSample(image: Value, metadata)`；DatasetBuilder 根据显式 key 写私有 chunk/journal，只发轻量 DatasetProgress，snapshot/materializer 按 revision 解析，禁止每帧 full DataBlock/DataPatch fan-out。
3. 交付 IMAGE ViewContract/ViewSpec/FigureEvaluator、2D live raster+Qt overlay、Workbench LiveDatasetBinding；验证 GUI/worker owner 和 driver buffer reuse。
4. 交付 CaptureArtifact Repository 和 crash-safe commit；live/save 冻结用户所见 revision。
5. 同时交付薄 Experiment：`connect -> capture -> inspect/figure -> save` 保持少量语句。
6. E2E 后只删除 live-image panel 自己的旧累计显示 buffer/latest polling/render 旁路。旧 camera frame producer/LogicNode 仍是 Occupancy/ROI/readout/sitemap 的 reactive 输入，必须保留并由 LegacyRuntimeFence 隔离，直到 S3 迁走最后一个 frame consumer 后与整条旧链一起删除；不能在 S1 让旗舰 readout 链黑屏，也不能同时启动新旧 camera owner。TaskOutput 仍有 CalibrateReadout/OptimizeMotField 等消费者，移到最后一个消费者迁走的 S5 删除。

### H1：Pulse bounded-context 与冻结 bitstream bring-up（与 S0.5-S3 并行）

先建立PulseDocument/TargetIR/CompiledPulseArtifact canonical seam，并以当前已部署bitstream对应的host/model/wire golden bytes、现有xsim/真机回读保护语义。按consumer纵向切换：compiler/server -> neutral Sequencer adapter -> workbench PulsePreviewProjector；每切一个consumer删除其旧timing/compiler/reader，不维持自动fallback。整个H1默认不修改RTL、不生成新bitstream。

H1完成现有`image.build_fingerprint`/几何/ABI握手、PreparedProgramRef软件guard、repeat轴展开的finite autonomous table与camera-trigger schedule digest、当前UART/AXI/JTAG容量/错误行为和existing DONE/status/progress语义的contract kit。Formal compiler明确强制`repeat_forever=False, scan_repeats=0`并拒绝host wrap-stop；超过resident window时保存refill p99/profile并生成可供preflight使用的margin。只测试现有RTL实际提供的能力，不增加ProgramToken/CellFireToken、ROM attestation、CRC verifier、PHYSICAL_DONE或telemetry。preview通过S0.5 workbench projector使用frontend FigureDocument，不制造frontend -> pulse反向边。

### H2：证据驱动的可选硬件修复门

H2默认不排期。只有以下任一证据成立才可创建提案：

1. E0在批准工作余量、正确camera配置和足够软件reservation下仍实测到丢帧/乱序，且无法通过扩大margin/降低trigger rate解决；
2. golden/xsim/真机交叉验证发现现有RTL真实bug或与已经批准的硬件设计不符；包括某安全行为只有在证明根因位于RTL且偏离既定设计时才满足本项，普通软件/transport问题不能借此要求重烧。

提案必须选择修复被证实问题的最小改动，列出不改RTL替代方案、回归范围和重烧风险，经PI/硬件owner批准后才实施。HardwareTriggerStamp、trigger-return、watchdog、ROM attestation、RTL CRC等只是候选，不是默认答案；任何合法重建都重新执行timing/CDC/golden/真机验收。

### S2：Data analysis + Frontend Figure 的完整纵向切片

1. 在 zlc_data 完成 Selection、DataTransform/Reduction/CommittedTransform、FitSpec/BoundFit/FitResultBatch；在 frontend 完成 ViewContract/ViewSpec/ViewSuggestion、FigureDocument/codec、selector controller、DataFigure/render。
2. 以“冻结 Capture/DataBlock -> curve/gridplot -> fit -> overlay -> FigureArtifact”为真实路径，同时验证 scalar、site grid、多 batch axes、component validity 和 per-cell failure。
3. formal 路径由 zlc_data `bind_fit` 产生 BoundFit、再由 neutral generic adapter 与 DatasetInputSlot 包成 AnalysisStep；interactive/offline 路径直接把同一个 BoundFit 与冻结 snapshot 投递到各自 QoS executor，三者调用同一 `fit_analysis`。
4. 验证 display ViewSpec 无 authority 字段，Selection candidate 只有在 FitSpec/CommittedTransform 中重建后才进入结果 lineage。
5. 新路径不再增加旧 `core.selection/fitting/facet/raster` consumer；这些模块、scalar fit signals 和 neutral Fit-named implementation 只在其最后一个旧 frontend/ROI/Analysis consumer 迁走的切片物理删除，通常为 S3/S5/Z0，不能在 S2 提前断开 opaque legacy island。

### S3：StreamProcessor、Calibration 与 Occupancy/readout

1. 在已工作的 camera event 上加入最小 `StreamProcessorWorker`、typed record、join/cardinality/budget 和 exact propagation；不让 StreamProcessor 读取累计 DataBlock。
2. 完成 CaptureArtifact -> CalibrationAnalysis -> CalibrationArtifact 的 live/offline 同路算法，以及 FrameContract/SiteMap/ReadoutModels。
3. 迁 `OccupancyStreamProcessor`，输出单个 `OccupancySample(occupied, counts, metadata)` typed record，并显式绑定 CalibrationArtifactRef/model。
4. DatasetBuilder 把 occupancy events 物化为 dataset；frontend Figure 与 zlc_data Fit 直接消费该冻结 dataset，证明四平面边界贯通。
5. integration 通过后删除旧 camera frame producer/LogicNode 与最后一个 Occupancy/ROI/readout reactive consumer组成的完整旧链，并删除 runtime/session calibration 回查 fallback、filesystem fallback、legacy search、拆散 scalar signals 和会碰硬件的旧 Processor；保留的只有 notebook/workbench composition 在 request 构造时按 ReadoutBindingKey 冻结显式 CalibrationArtifactRef/model 的可见 convenience pointer。

### S4：近期 Formal PulseScan（AUTONOMOUS_STREAMED）

只有E0 CameraExternalTriggerEnvelope、H1冻结bitstream合同、S1 exact acquisition与S3 StreamProcessor/DatasetBuilder全部通过才开始：

1. bind declared ExactSourcePipeline，fire 前建立全链 reservation/cursor/budget/ack；不得借用 monitor worker。
2. repeat轴展开进`repeat_forever=False, scan_repeats=0`的finite table；preflight冻结camera readback与compiled trigger schedule，验证trigger interval、total frame/byte buffer和chunk refill margin；camera一次arm完整帧预算，FPGA一次fire完整SCAN_SLOT table并无缝自主执行。
3. adapter按E0证明的delivery order将frame[i]映射为frozen TriggerKey[i]，全链数据保持PROVISIONAL；run末端比较existing FPGA completion/progress、expected/produced total、frame/camera stamps、timestamp容差、coverage/EOS，全部通过才VALID。
4. 迁scan-slot/API-slot request、ScanOutputContract、multidimensional y和ScanArtifact Repository；MOT只允许SCAN_SLOT/AUTONOMOUS_STREAMED，不加API或host-stepped fallback。API_SLOT无法无缝更新时仅沿用`API_SLOT_SEGMENTED_EXISTING`，每segment及aggregate均对账。
5. 对drop/reorder/duplicate/short read/counter reset/timestamp gap、camera总buffer或refill margin不足、旧`scan_repeats`多发point、schema generation、component invalidity、RemoteSequencer abort与provisional epoch做整runreject-and-redo真机测试；重试默认手动，自动策略必须显式有界并保存失败attempt。
6. E2E 后删除 positional zip、latest fallback、旧 PulseScan 与 neutral key 泄漏进 FPGA 的类型。

### S5：Workbench、其余 use cases 与用户兼容

1. 在 S0.5 宿主上迁剩余 WorkspaceModel/RunCoordinator/controllers；Setting/Edit 使用共享 EditorSession/schema widgets/base-revision conflict，catalog 只做各 bounded context capability/definition 的本地投影。
2. Fit 保持 zlc_data 单一算法 owner，并由 frontend 在 UI 中提供明确 `Add Analysis -> Fit` / `Analyze -> Fit` 编辑与 overlay；Pulse prepare/fire/safe 后台托管。
3. 所有 panel 使用 S1/S2 的 render/evaluation lanes；完成 acknowledgement-driven shutdown、persistent quarantine 和 ControlTopic terminal/superseded ack。
4. 逐条迁 temperature、MOT、readout、device manager 和 notebook convenience；每条都按纵向切片删除自己的旧路径。
5. 真实入口 E2E 覆盖 fit/gridplot、calibration/occupancy、PulseScan、save/load、cancel/quarantine、shutdown 和 virtual/real adapter parity。
6. CalibrateReadout/OptimizeMotField 等最后消费者迁走后删除 TaskOutput；每个删除项由“移走最后一个消费者”的切片负责，而不是由第一个碰到该类型的切片负责。
7. 最后一个 consumer 消失时物理删除 `neutral_atom/core`，不保留空 re-export 包。

### Z0：零残余审计

- legacy path/symbol/fixture/reader、双 registry、双 codec、双 fit owner 为 0；
- reverse import 为 0，FPGA domain key 泄漏为 0，stream 上的累计 DataBlock/DataPatch 为 0；
- giant smoke/source-location/private-structure tests 删除；
- docs/notebooks 只描述当前 public path；
- architecture、contract、E2E、performance、artifact crash 与 hardware gates 全部通过。

## 20. 目标目录结构

```text
packages/
  zlc_data/
    value/
    dataset/
    selection/
    transform/
    reduction/
    fit/
    codec/

  zlc_frontend/
    figure/document/
    figure/view/
    figure/evaluation/
    figure/artifacts/
    render/
    qt/

  zlc_storage/
    canonical/
    blobs/
    manifests/
    atomic/
    maintenance/

  zlc_pulse/
    model/document/
    model/ir/
    fpga/target/
    fpga/compiler/
    fpga/manifest/
    fpga/host/
    fpga/telemetry/
    fpga/transport/
    fpga/server/
    fpga/rtl/
    fpga/sim/

  zlc_neutral_atom/
    catalog/
    runtime/run/
    runtime/pipeline/
    runtime/streams/
    runtime/materialization/
    runtime/analysis/
    runtime/resources/
    acquisition/
    processing/stream/
    analysis/
    artifacts/
    devices/ports/
    devices/adapters/
    scan/
    readout/calibration/
    readout/occupancy/
    experiments/

apps/
  zlc_notebook/
    experiment/
    result_projectors/
    composition/

  zlc_workbench/
    workspace/
    experiment/
    pulse/
    analysis/
    calibration/
    devices/
    figure/
      live_binding/
    composition/
```

这些是 import/ownership namespace，不等于第一天必须发布五个 wheel。仓库可先用一个 distribution 的多个顶层 namespace，并用 architecture tests 保证 DAG；只有 FPGA server、Qt extras 或部署体积确实需要时再拆 wheel，公开 import path 与所有权不变。

不建立新的 `common`、`shared`、`utils` 或跨领域 `core` 杂物包。

## 21. 删除清单

删除由**最后一个真实 consumer 消失的 dependency-closed 切片**负责。不得因为 S1/S2 首次建立替代品就提前删掉仍被 S3/S5 使用的能力，也不得以“还有别的 consumer”为由让已迁 use case 继续双写/双读。至少固定：camera live panel 的旧显示旁路 -> S1；共享旧 camera frame producer/LogicNode + Occupancy/ROI/readout/sitemap reactive chain -> S3；旧 fitting/selection/facet/raster -> 其最后一个 legacy frontend/processor consumer 所在的 S3/S5/Z0；session calibration fallback/旧 Occupancy outputs -> S3；旧 positional/latest-polling PulseScan -> S4；TaskOutput、LegacyPanelHost、LegacyRuntimeFence、SerializedLegacyAggBridge、剩余 TaskConsole god shell -> 最后 consumer 的 S5/Z0；legacy pulse upgrader/旧 compiler consumer -> 对应 H1 consumer切换提交。每项在路线图/PR checklist记录replacement、全部consumers、shared ResourceKeys、first migrated slice、last consumer slice与物理删除证据。

完成态不存在：

- `neutral_atom.core`；
- zlc_data -> frontend/neutral/pulse/workbench import；
- frontend -> neutral_atom/pulse 反向 import；
- pulse/FPGA -> neutral_atom/frontend/data 反向 import；
- async ExecutionEngine、child run、递归 plan；
- continuous-exact epoch/spool 与专用 command/build lane；
- UnitSpec/CoordinateFrame 图代数；只保留 canonical unit id、opaque frame id 与显式转换；
- 默认 SnapshotLease；零拷贝只允许 profiling 驱动的 opt-in BorrowedSnapshot；
- node-owned worker/thread/terminal state 与运行中动态 pipeline edge；
- public Task/Measurement/StreamProcessor/Analysis god base hierarchy；
- TaskOutput 和 `__task_frame__`；
- per-signal gap -> latest fallback；
- 独立设备按 sequence zip、自由运行无 tag 的位置式 trigger ledger；
- FPGA 内的 neutral TriggerKey/ScanCellKey/ScanPlan 类型；
- sample stream edge 上的累计 DataBlock/DataPatch 与把 DatasetBuilder 伪装成 Processor；
- 一个 invocation 的多 scalar output transaction；同 shot 结果必须是一个 typed record；
- scalar fit result signal 集合；
- hidden Fit node inference、neutral `FitProcessor/FitOperator/FitAnalysisDefinition`；
- frontend 内复制的 fit model/solver/result schema，或 zlc_data 内的 Matplotlib/Qt selector controller；
- Selection 中的 arbitrary metadata/plot binding/widget scope/control JSON byte payload；
- ViewSpec 中的 authority seed/CommittedTransform 与 display operation -> authority 通用转换；
- ValueSchema/DatasetSchema 中的 ProjectionCapability/renderer/reducer inventory；
- rank/singleton/index-0/global-nanmean/anonymous-flatten axis inference；
- DataFigure 的 Hub/session/pulse knowledge；
- GUI/worker 无确认共享同一 Figure/artist、barrier timeout 后继续访问，以及 S0.5 SerializedLegacyAggBridge；
- calibration plugin/discovery/session fallback；
- dynamic FQCN task/device import；
- universal ArtifactRef；
- FigureDocument 中的 LiveDataBlockRef、partial success artifact、manifest 前可见的半写文件；
- legacy pulse upgrader/fixtures/readers；
- sibling `_program.json`；
- tracked FPGA fallback literals；
- 文档/host把当前RTL并不存在的CRC/BANK_VERIFIED/逐沿receipt冒充既有能力；
- stale PreparedProgramRef、未冻结完整scan table、运行中读取mutable GUI slot或失败cleanup后无条件release；
- 除密封 notebook `connect/Experiment` 外的 broad umbrella re-export；
- 保护文件位置、继承树和私有 GUI 结构的 public contract tests。

## 22. 最终验收

架构：

- import DAG 通过；
- zlc_data、pulse/FPGA、neutral_atom、frontend 与窄 zlc_storage 可按声明边界 isolated import；部署可同 wheel 或按 server/Qt 需求拆分；
- concrete construction 只在 allowlisted bootstrap；
- DefinitionCatalog 由显式 imports 组装，PipelineSpec 只有一个顶层 Run owner；
- zlc_data 不引用 runtime/presentation 类型，frontend FigureDocument 不引用 neutral runtime live ref；
- Fit/Selection/FitResultBatch/BoundFit 只有 zlc_data owner，DataFigure/selector controller 只有 frontend owner；BoundFit + DatasetInputSlot -> AnalysisStep 的适配只发生在 neutral/composition 侧；
- sample event、materialized dataset、analysis result、presentation 四个平面之间只有声明的三处边界；
- 每个事实有唯一 owner；
- physical device 在 Workbench/notebook/standalone/remote 入口间只有一个经 DeviceControlLease 证明的 owner；迁移期 legacy runtime也不能绕过同一 ResourceArbiter；
- 旧路径、alias、fallback 为 0。

数据与 Signal：

- 任意 trailing axes 不丢；
- sample stream 只传 Value/typed record，不传累计 DataBlock/DataPatch；DatasetBuilder 是唯一 event -> dataset materializer；
- 多维 point axes 只经 PointLayout 映射 P，不退化为匿名 data_points/data_dim；
- cell/component partial validity 沿 reduce/fit/histogram/meter/figure 正确传播；
- scalar 语义明确；
- published DataBlock 内容稳定 immutable，builder 后续 ingest 不改变旧 DatasetRevisionRef；patch revision/duplicate 规则正确；
- exact pin/ack/generation/byte budget 正确；
- qCMOS AUTONOMOUS_STREAMED只在E0 CameraExternalTriggerEnvelope与preflight margin内运行；所有数据在EndAttestation前provisional，不一致整run拒绝；artifact明确记录ORDERED_END_ATTESTED_RUN及其非逐沿剩余风险；
- repeat轴已展开进finite single-pass table，Formal路径不存在`scan_repeats>0` host wrap-stop；
- coherent monitor 不混 shot；
- rolling monitor bounded且不会冒充完整 formal dataset；
- artifact/result 原子提交；
- 自动视图只按 axis metadata + ViewIntent 决策，不读 values 猜语义；
- 显示投影始终可见、可改、可复现，且不修改原始 DataBlock；
- DataTransformSpec/CommittedTransform 与 figure ViewSpec 分层，display-only step 不能进入 fit、正式 scan y、calibration 或派生 artifact；
- histogram pool、image repeat、fit batch 等语义由 ViewContract 声明，没有 render 特例或全局 repeat mean。

执行与线程：

- GUI thread 无阻塞 hardware I/O、plan/pulse compile、transform 或 fit；
- cancellation 不谎报 terminated；
- join timeout 不释放 owner，safe failure quarantine resource；
- device/session/Figure thread affinity 与单 owner 正确；S0.5 legacy handoff bridge 已删除；
- BoardFrame coherence 使用完整 run/key/dataset/generation/schema/revision identity；BorrowedSnapshot 在所有 discard/close 路径释放；
- cleanup failure 可见；
- shutdown 无存活 worker、未释放 opt-in borrow token 或未执行 safe action；
- RunCoordinator 不成为第二 lifecycle owner，EditorSession 不覆盖新 revision。

Artifact/Calibration：

- manifest commit marker 前不存在成功 artifact，load 全量验证 manifest/blob；
- Figure Save 冻结用户所见 revision，frontend artifact 不序列化 LiveDataBlockRef；
- failed/cancelled formal run 不产生成功 artifact；
- live/offline calibration 在 CaptureArtifactRef 后走同一算法路径；
- FrameContract/SiteMap/model applicability 不匹配明确失败。
- calibration 默认按 ReadoutBindingKey 与声明的 model policy选择，site-map-only/完整 model capability 不混淆。

Pulse/FPGA：

- Formal PulseScan在软件reservation/cursor/processor/collector层端到端exact；相机frame↔point按E0证明的ordered external-trigger contract与整run末端对账成立，不声称逐沿硬件证明；
- 精密scan/trigger时序由冻结bitstream和qCMOS硬件执行，host只做run前margin验证和run后attestation；
- 可观察的gap/duplicate/out-of-order/count/stamp/timestamp/EOS不一致使整run INVALID且不提交；accepted equal loss+extra residual risk在provenance可见；
- TriggerKey/ScanCellKey/ScanArtifact 保留多维 point/output axes；
- S4使用现有FPGA完整自主scan table；neutral按frozen schedule + ordered camera frames映射TriggerKey，末端验证后才转VALID；
- camera总frame/byte buffer与超resident chunk refill p99 margin均在fire前证明；运行时underflow stall不算无缝成功；
- MOT只使用SCAN_SLOT + AUTONOMOUS_STREAMED；无API或host-stepped fallback；
- API segmented每segment和aggregate双层EndAttestation；INVALID attempt可见且不会被无限自动重试隐藏；
- build/target digest 闭环；
- runtime现有`image.build_fingerprint`/几何/ABI握手闭环；新ROM attestation不是baseline；
- PreparedProgramRef + connection generation + artifact/table digest使旧软件引用不能进入正式FIRE，完整自主table在fire前冻结；
- current resident/streamed bank按现有RTL真实能力完成golden/真机验证，不要求不存在的RTL CRC；
- RemoteSequencer现有timeout/cancel/abort/safe与故障时quarantine通过真机测试；不要求新watchdog/独立SAFE硬件；
- RTL/bitstream保持冻结；只有E0实测工作区间内loss/reorder或已证实RTL bug/设计偏离，才允许H2硬件修复评审。

性能：

- queue/patch 摊销 O(1)，compile/journal/artifact 对数据量近似 O(N)；
- exact retained bytes、view/Fit queues 与 UI stale work 全部有界；
- profile matrix 的 p95/p99 latency、peak RSS 与输出等价 gate 通过。

用户体验：

- launchers、主要窗口和核心操作流程保持熟悉；
- Fit 可见且可从真实入口完成；
- 非标量数据打开即有合理默认视图；只有真实歧义才要求最小必要输入；
- panel 始终显示当前 axis/selection/reduction/batch 摘要；
- monitor 显示实时且 coherent；
- Stop、冲突和错误状态诚实；
- virtual 与 real 仍走同一路径；
- notebook 的 connect/capture/fit/save 仍保持短路径，不暴露内部 PipelineSpec/Port ceremony；
- UI 视觉和交互能力不因架构拆分退化。

## 23. 核心结论

系统不需要一个通用异步工作流编排器。它需要的是：

```text
同步领域语义
+ 明确线程宿主
+ cooperative cancellation
+ flat resource claims
+ exact stream reservation/cursor
+ immutable typed data
+ sample/dataset/analysis/presentation 四平面
+ 单向 package ownership
+ fail-closed hardware contracts
+ capability-evidence gates（近期可做什么与终态想做什么分开）
```

扩展性来自稳定数据与能力边界、显式组合和机械 contract tests，而不是更多继承层、Protocol、Service 或动态注册。

近期最重要的落地顺序不是先拆完五个namespace，而是先让lifecycle/resource状态诚实、建立Workbench/render宿主、跑通Camera event -> DatasetBuilder -> live/save，并在E0用真qCMOS建立外触发确定性envelope。随后直接在冻结bitstream上运行无缝AUTONOMOUS_STREAMED，以preflight margin + ordered metadata + per-run EndAttestation提供Formal基线。逐沿stamp、新ROM、trigger-return等只有真机证据触发硬件修复时才评估。

终版GO/NO-GO裁决是：顶层架构 **GO**，E0/F0/S0.5可立即开工，S1-S3按dependency-closed cut开工；S4在E0 CameraExternalTriggerEnvelope、H1 current-bitstream contract、schedule margin和软件exact链通过后 **GO**，无需重烧。硬件改变默认 **NO-GO**；唯一解锁条件是E0在批准余量内实测loss/reorder，或现有RTL bug/既定设计偏离被证实，并经PI/硬件owner单独批准。

最终实现应让用户继续使用熟悉的TaskConsole、PulseGUI和notebook工作流；重型board保持下线程raster性能，notebook保持短路径，MOT保持SCAN_SLOT无缝自主扫描。内部消除软件缓冲跳帧、线程竞态、隐式降维和重复算法；相机↔point的物理保证明确采用`ORDERED_END_ATTESTED_RUN`而非逐沿硬件tag，任何不一致整run拒绝重跑，所有迁移bridge在Z0物理删除。
