# Zou_lab_control 最终系统架构设计

## 1. 文档定位

本文定义 Zou_lab_control 的最终架构、实现边界与验收标准。它只保存仍由当前实现和产品合同兑现的终态事实，不保存迁移轮次、临时 checkpoint、过程处置或过期计数。设计一旦被真实产品流、profiling、设备合同或代码依赖证伪，就应从 owner 和物理语义重新推导，并同步修正文档与实现。

全系统必须同时满足以下不变量：

1. `zlc_data`、`zlc_pulse`、`zlc_neutral_atom`、`zlc_frontend`、`zlc_workbench` 与 `Zou_lab_control` 的 owner 和依赖方向唯一。
2. Task、Measurement、Processor 与 Fit 的领域语义不可互相冒充；live、finite、latest-only、exact 只是宿主或交付策略。
3. 数据永久保持 `(R,P,*data_shape)`、具名 axis、PointLayout、validity 与完整 lineage；标量固定为 `(R,P,1)`。
4. 正式采集、连续 monitor、控制状态、GUI front 与运行事件使用不同合同，不能用 displayed/latest 代替实验事实。
5. PulseScan 是“pulse program + 已经运行的外部 `Signal(y)`”：它只取得 sequencer authority，不取得、不启动、不停止信号 producer，也不按 Camera、Processor 或 Figure 类型分支。
6. 普通 future cursor 只保证订阅后的软件顺序。正式 PulseScan 还要求 producer-owned association：FIRE 前冻结下一组事件，FIRE 后绑定 exact `PulseTerminalAck`，完成后由 producer 返回可持久化证据；缺少该能力必须在任何 FIRE 前拒绝。
7. Qt、阻塞设备 I/O、数值计算和 raster compose 各有唯一线程 owner；immutable snapshot 只出现在真实 ownership/revision 边界。
8. Calibration 是内建 readout 领域能力，不使用 plugin；其物理与算法由当前 owner 和独立数据证据共同约束。只有已经独立验证的 calibration/readout 科学算法才可按需把 `main` 的对应实现作为 oracle，不能让旧 UI、生命周期或包结构反向定义当前系统。
9. Fit/selection/transform 的纯数据语义由 `zlc_data` 单一实现，Figure/selector/renderer/交互由 `zlc_frontend` 单一实现；领域节点只发布 typed inputs/outputs，不复制画图系统。
10. installation 与 Logic Node 都在固定内建 namespace 下发现一个 leaf-owned、冻结的 `*.package` 声明，再由 composition root 接线 Port、API 与可选 UI leaf；禁止 mutable registry、entry point、service locator、FQCN 构造和外层字段/输出硬编码。
11. 最终树中不存在 alias、fallback、兼容 reader、双 registry、平行实现、临时 bridge 或零消费者抽象。
12. 精密 pulse/trigger/exposure 时序由现有 FPGA、qCMOS 等硬件执行，host 不用 sleep 调度边沿；bitstream/RTL 冻结，架构偏好本身绝不是重烧理由。

### 1.1 PulseScan 与硬件的最高优先级约束

1. **现有 RTL/bitstream 冻结。** baseline 不生成、不修改、不重烧、不隐式调用 Vivado programming。只有真机或模型证据证明现有 RTL 有 bug、偏离既定设计，或在已批准工作余量内仍发生无法由相机设置、trigger rate、margin 或软件 ownership 修正的 loss/reorder，才可进入单独的硬件修复评审；评审本身不授权修改或烧录。
2. **SCAN_SLOT 正常路径只有自主硬件扫描。** `AutonomousScanSlotProgram` 在 FIRE 前冻结并编译完整有限表，一次 FIRE 后由 FPGA 自主执行全部微观时序；host 不参与逐 cell 发射、等待或边沿调度。
3. **API_SLOT 是已存在且显式的例外。** 只有值确实不能在一次 FPGA sweep 中无缝更新时，`ApiSlotSegmentedProgram` 才按 R-major/P-fast 执行独立 `STATIC_ONCE` pulse session。每段内部全部边沿仍由硬件执行；segment 间 host gap 是该模式公开的物理事实，不能伪装为 autonomous，也不能推广给 SCAN_SLOT/MOT。
4. **PulseScan 不拥有上游生命周期。** 绑定时必须解析到一个已经 RUNNING 的具体 producer instance/generation。scan 的启动、取消和 cleanup 只作用于自己的 sequencer session、collector 与 repository borrow；producer 在 scan 成功或失败后都继续按自己的生命周期运行。
5. **Signal 能被选择，不等于能被正式关联。** 所有声明的 Dataset output 都可进入 picker；只有实现 `SignalEventAssociationSource` 的 producer 才可启动 PulseScan。排序型 `SignalEventSource`、monitor latest、panel raster、display projection、静态 artifact 与自由运行相机均不能自行升级。
6. **关联证明由物理 producer 铸造。** association cursor 必须在 FIRE 前 arm 一个明确 `expected_event_count` 的 group；绑定时核对 exact pulse session/artifact terminal；交付时只返回该 group 的事件；finish 时证明 group 完整，并把 request、terminal digest、source generation 和 producer-specific物理证据写入 canonical evidence。PulseScan 只验证通用合同，不解释 Camera metadata 或仿真类型。
7. **Processor 只能在严格 1:1 时传播关联。** 每一个关联输入必须确定地产生恰一个关联输出；派生 EventRef、直接 input EventRef、processor binding 和 calibration artifact refs 全部进入 lineage。window、drop、aggregate-over-time、latest-only 跳过或多输入 join 不得传播该能力。
8. **virtual 与 real 的证明强度不混用。** virtual readout Camera 可由其唯一 in-process trigger-wire owner记录实际 FIRE、trigger channel、frame ordinal interval 与完整产出，因而可为 Camera 和严格 1:1 Occupancy 提供正式模拟 association。FREE_RUNNING `mot_camera` 没有该能力。真实 qCMOS 只有在 production InstallationConfig/AssetMap、CameraExternalTriggerQualification 与 product E2E 全部成立后才可从 typed NO-GO 放行；virtual 成功不能替真机背书。
9. **真实 qCMOS 的正式路径不要求改 RTL。** 相机经验时序与外部触发资格必须先证明冻结 exposure/ROI/readout 工作点在编译 trigger 间隔内满足“一触发一帧、按序”的经验合同；每次 run 再对账 camera produced/drained count、source ordinal、可用 stamp/timestamp、coverage 与 pulse terminal。任一不符使整个 attempt INVALID并拒绝提交。该保证弱于逐沿硬件 tag，不能声称能定位具体错点或排除所有等量 loss+extra 情形。
10. **需要新 RTL 的能力默认不存在。** HardwareTriggerStamp FIFO、trigger-return、per-edge counter、新 ROM/timing attestation、single-cell gate 与新 watchdog 只是在上述证据触发后可能评审的候选，不是软件 baseline、测试假能力或准入 gate。

当前事实 owner 固定如下：

| 事实 | 唯一 owner | 消费方可以做什么 |
|---|---|---|
| pulse source、scan slots、target IR、compiled trigger schedule | `zlc_pulse` | neutral application 冻结并提交 artifact；不得重算 waveform |
| FIRE、terminal 与硬件执行顺序 | bound sequencer Port/session | PulseScan 保存 exact `PulseTerminalAck`；不得从 GUI progress 猜 terminal |
| 一组信号事件是否属于该 FIRE | 具体 signal producer 的 association authority | PulseScan 只调用 arm/bind/next/finish 并保存 evidence |
| Camera frame ordinal、produced count、stamp/timestamp | Camera adapter/endpoint | Camera Measurement 验证并发布 immutable `CameraSample`；Workbench 不解释 metadata |
| Processor 派生与 calibration 依赖 | 具体 Processor application | 只有严格 1:1 路径传播 association 并追加 lineage |
| `(R,P,*data_shape)`、cell coverage 与 materialization | `zlc_data` + PulseScan collector | ScanRepository 只接受完整 seal；不得 positional zip/latest 补点 |
| panel 的当前 raster、selector 与显示 revision | `zlc_frontend`/Workbench | 仅用于显示与用户意图，永不成为物理 same-shot 证明 |
| 部署身份 | installation 的现有 fingerprint/geometry/ABI 握手与部署记录 | mismatch 时拒绝；不得宣称验证硬件未暴露的 content/timing digest |

一次 PulseScan 的权威闭环只有一套：

```text
Bind
  resolve exact running producer instance/generation/output
  freeze optional authoritative DataTransformSpec -> CommittedTransform
  compile pulse program and exact DatasetCellSchedule
  require producer association capability

Pre-FIRE
  prepare sequencer session
  open producer-associated cursor
  arm exact event group before FIRE

Execute
  autonomous SCAN_SLOT: one finite hardware FIRE for the complete schedule
  API_SLOT exception: one hardware-executed STATIC_ONCE session per cell
  bind each group to its exact PulseTerminalAck
  consume every associated event once into DatasetBuilder

Commit
  producer finishes association evidence
  verify event refs, direct input refs, processor/calibration stages,
         pulse terminal(s), transform authority and complete cell coverage
  seal `(R,P,*data_shape)` dataset
  atomically publish one ScanArtifact
```

任何 association、terminal、schema、lineage、coverage 或 repository 校验失败都使该 scan 失败；不得丢帧、移动 ordinal、补点、拼接成功部分或把显示中的 provisional 数据升级为权威结果。重跑必须是新的 Run/association/repository commit。上游 producer 本身不因 scan 失败被停止或重建。

## 2. 用户体验合同

内部包边界与线程所有权不得改变实验用户的日常工作流：

- 仍通过 `task_console.bat`、`pulse_gui.bat` 和 notebook 启动系统；
- PulseGUI 仍保留 Edit、Preview、Scan 三个主要工作区；
- TaskConsole 仍使用 Add Panel、Setting、Edit、Start、Stop、Save/Load；
- Measurement、Processor、Task 仍可从 catalog 选择并连接；`Processor` 是唯一产品与领域名称，具体capability分别拥有latest snapshot evaluation与可选的association-bearing derived signal source，不建立通用formal processor host。用户可见拟合一律命名为 `Fit`；Calibration/Report 保持各自领域名称，不建立泛化的 `Analysis` 菜单或节点；
- live image、rolling plot、histogram、site map、fit overlay 的视觉语言保持一致；
- virtual 与 real 仍只替换最低层设备 adapter；
- pulse prepare/fire/safe、camera arm/read/terminal-drain/disarm 的用户操作流程保持；Stop 只有在可验证的 terminal recipe 完成后才能 release。

用户可见的终态行为固定为：

- Fit 直接存在于 panel 的 Setting/Edit 和 DataFigure 的 `Fit` tab；只有 model、一个可选参数命令行（如 `center=50, sigma_lower=0`）、`Fit/Clear`，不展开模型全部参数表，不打开第二个 DataFigure；
- 非标量数据自动得到一个可见、可改的默认视图；当选择或降维要进入 fit、scan y 或派生 artifact 等权威结果时，必须冻结为 CommittedTransform，禁止暗中取第 0 项；
- history gap、schema change、缺帧和硬件 mismatch 明确失败，禁止显示拼接结果；
- PulseScan picker允许浏览已声明的外部signal，但正式Start只接受producer-owned association；失败时明确指出缺少哪项能力，禁止改用latest值、停止上游或建立内部Camera capture；
- 正常SCAN_SLOT把完整逻辑scan table在FIRE前冻结，一次FIRE后由FPGA自主执行微观时序；仅API-slot值无法无缝更新时沿用既有segmented路径，并公开segment间host gap。真实qCMOS只有在其自身production composition、CameraExternalTriggerQualification与逐run对账证据完成后才能作为formal producer；virtual通过不替代该真机gate；
- Stop 后若设备尚未确认退出，UI 显示 `CANCELLING`，不会提前宣称已停止；
- 同一个 TaskConsole 内的新 Logic run 若与另一行发生 typed resource conflict，shell 用 runtime 返回的 exact conflicting RunId 停止那一行，等待其真实 terminal 后自动重试原冻结请求；复合 Task 必须把 child admission rejection 作为 typed `ResourceBusy` 随 `RunSnapshot` 原样带回，shell 只归一化 direct/composite 两条 typed 路径，禁止解析错误文本；外部窗口/进程 owner 仍明确拒绝，runtime 错误与 admission 逻辑不被绕过；
- 保存/加载只接受current artifact schema；未知或旧格式明确拒绝，runtime不提供转换工具；
- frontend render 可以用新revision替换尚未开始的display工作以保持实时，但monitor数据交付不跳帧，一次画面中的相关信号也不会来自不同 shot；
- 重型 grid/多 panel board 由 worker raster 后整板 coherent present，视觉与交互保持；GUI/worker禁止共享 Figure；
- Pulse prepare/upload、长 fit 和 calibration 不阻塞 GUI thread。

主要入口、操作结构、视觉风格、功能可发现性、错误提示与安全状态均以本节和 §2.2 为最终合同。

GUI/data 接缝必须满足以下单一合同：

1. `task_console.bat` 先由 DeviceManager 初始化唯一 `Experiment`，然后同时打开 TaskConsole 与 PulseGUI；两窗借用同一个 composition root，TaskConsole 关闭时先关闭其 PulseGUI sibling，再关闭 installation。
2. 真标量的物理表示固定为 `(R,P,1)`，尾轴只能是 `zlc_data.SCALAR_AXIS`；Logic/picker 自动显示 `R × P × (1)`。空 `data_axes`、按 singleton/rank 猜 scalar、把多维 point layout 展开进 shape 字符串全部拒绝。
3. Camera 的 `frames_per_cycle=N` 在 request/catalog seam 唯一声明 `frame_0 … frame_(N-1)`；一次原子 Camera Dataset 只沿声明的 `READOUT_EVENT` 轴投影，各输出保留 repeat、其余 point/data axes、validity、revision 与 generation。所有 picker、Logic legend、live/finite route 使用这份冻结声明，不存在 `frame` 兼容别名。
4. Occupancy 显式选择 Camera `frame_i`，并在 `Task output` 与 `Saved calibration` 二者中选一个 calibration source。保存路径只接受 current `calibration_ref.json` 指针，校验其中 CalibrationArtifactRef 与 source CaptureArtifactRef 的 exact provenance；不搜索 latest、不猜目录、不复制模型。
5. IMAGE viewport 的唯一权威是物理 `x_limits/y_limits`；wheel/pan 直接提交新 limits 给 worker，禁止先 crop/stretch 旧 raster 伪装响应。renderer 固定 `aspect="equal", adjustable="box", anchor="W"`，Divider/size 使用 draw 后真实 axes bbox；相同 source generation 的已完成 front 按数据 revision 接受，GUI 请求序号不冒充 presentation truth。已有 matplotlib chrome/blit cache继续复用。
6. 只有 Monitor board 可因用户摆放卡片而横向超过 viewport，并出现 scroll。Logic 与所有 Edit 页永远不因 status/error/path 文本增宽；文本 elide/wrap，横向 scrollbar 关闭。Edit body 先挂入正式 tab/window 再构建 Figure host，禁止产生瞬时 top-level 小窗。
7. Grid 只 author 一条具名 facet AxisId；其余显示轴由 frontend 的纯 resolver 按 schema 声明顺序补全。repeat 不能暗中形成第二 facet。Setting/Edit 复用同一行 inventory（facet、sub plot、bins、log count、colormap）、同一 label width、一个 outer scroll，不在 Qt callback 猜 AxisId 或维护第二套布局。

### 2.1 UX 行为权威与验收证据

**UX 行为权威是本设计列出的当前产品合同与正式 launcher 的真实表现。** `main` 只在用户明确要求复刻某项交互、或当前行为证据不足时作为按需比较材料；不得把旧树整体当成默认 UI、Grid、projection、lifecycle 或包边界权威。selector 覆盖哪些 plot kind、zoom/pan 手感、live 是否不断流、ROI/threshold 是否热更新、Fit 从哪里触发及何时可见、relim/cmap/limits 控件、Setting/Edit 布局与保存/载入流程，都必须由当前唯一 owner、真实输入事件和产品 E2E 共同证明。内部实现不得因便利而缩窄本节的用户面。

以下条款为**冻结条款清单**，任何局部实现状态、实现困难或测试便利均不能覆盖：

1. §2 的“日常工作流应保持熟悉”和“视觉语言保持一致”；
2. §7.2 的 `ControlTopic[T]`：ROI、threshold 等运行控制是 typed、revisioned、acknowledged，已 ACCEPTED revision 必须在事务边界得到 `APPLIED` 或明确的 terminal ack；
3. §12.5 的 `WORKER_RASTER_LIVE`：worker raster 与 GUI 解耦的同时，Qt overlay 必须承担 Area、锁定 Cross、selection 与 drag handles，`ViewportTransform` 必须承担同 revision 的 zoom/pan 与命中换算；plot 不提供 pointer-motion 数据 hover；
4. §18.4 的真实 launcher/composition-root E2E：必须用真实交互事件证明日常流程和操作时序未退化，不能只证明 controller 内部状态或静态 PNG；
5. §12.6 的用户可见 Fit：普通 Fit 即提交权威 draft，selector/SelectionCandidate 可预填同一 draft，不能用额外确认步骤或独立工具窗口替代主工作流。

**任何 GUI/交互变更在实现前都必须冻结当前行为和目标合同证据。** 只有目标交互需要按需对照旧实现时才只读检查 `main`；不允许每次修改都重新审查整棵旧树：

| 字段 | 必须记录的证据 |
|---|---|
| exact oracle | 当前产品合同、正式入口与唯一 owner；若确需旧实现对照，再记录 `main` 的 exact commit、对应文件/符号与真实 consumer |
| 入口与控件 | 真实 launcher、菜单/按钮/快捷键、Setting/Edit 字段和默认值 |
| 交互覆盖 | Area/locked Cross/zoom/pan/relim/cmap/fit 各覆盖哪些 plot kind；plot pointer-motion hover 一律不存在 |
| 时序 | 按下、拖动、松手、Apply、Stop/Close 后何时可见、何时 authoritative |
| 不可中断项 | 哪个 source、raw front、其它 panel 或硬件 Run 在交互期间继续运行 |
| 即时生效项 | 哪些修改热更新，revision/ack 在何处显示 APPLIED |
| 保存/恢复 | workspace、figure、selection、fit 与控件状态的持久/重开行为 |
| authority | `DISPLAY_STATE / SELECTION_CANDIDATE / REVISIONED_CONTROL / AUTHORITATIVE_DRAFT / COMMITTED_RESULT` 中的边界与跨越动作 |
| 禁止机制 | Hub、共享 Figure、shape 猜测与隐式降维均不得出现 |
| 验收对照 | 实现逐项 PASS/FAIL；FAIL 必须修正，或取得用户批准并登记替代验收，不能改写权威行为或删掉验收项 |

证据状态只可为 `MATCHED / MUST_CLOSE / APPROVED_DEVIATION / NOT_APPLICABLE_WITH_EVIDENCE`，并附真实 launcher E2E、event、Run/frame/revision 证据。`main` 仅是按需的 UX 比较材料和已验证科学算法 oracle；`DeviceSet/Registry/Hub/LogicNode`、旧线程共享方式或旧包结构都不是应保留行为。“实现复杂”永远不构成降低 UX 的理由。

### 2.2 冻结的最终 UX 合同

本节只陈述产品终态，不保存实现过程或完成状态。任何偏离都必须按 §2.1 取得用户批准，并在对应的 current contract/test 中留下可执行证据；不得另建迁移历史作为运行时或测试权威。

1. live plot 的正式交互包含 Area、锁定 Cross、zoom/pan、relim 与 cmap/limits；plot 不提供 pointer-motion 数据 hover。拖动只 hold 目标 panel，source 与其它 panel 继续前进，松手直接显露目标 panel 已到达的最新 front。pointer press 先按具体 surface 固定屏幕 raster、typed Figure/display、exact dataset value 与 producer transaction sidecar；随后每个 drag commit 都只能消费这组 press-time 事实，不能把下一帧 ancestry 接到旧 pixels 上。Live 与冻结 Edit 各自持有 pin，互不暂停或替换。Area 只在完成手势后发布 `area.data` 及具名轴范围，Cross 只在右击锁定后发布坐标。
2. 六种正式 plot kind（`2d/sites/1d/monitor/hist/grid`）共用 frontend 的 `FigureSpec/Divider -> PlotPanelContract/PlotPanelSession -> immutable front -> SinglePanelHost/QtRasterBoard` 链。Meter 不是可添加 panel，只保留为确有内部消费者的静态数值 render primitive。Grid overview 只负责 exact typed cell focus，focus 后复用同一个 host、selector、Fit 和 export；稀疏 hole 保持其 logical address，不暗选第一格。
3. FigureViewer 打开 current `.npz` 后仍是完整 DataFigure：支持适用的 range/cross/zoom/pan/home、relim/cmap/limits、Setting/Edit、Fit/Refit、原子导出和 Save/reload。multi-layer、faceted IMAGE 与各 archive plot kind 不能因加载来源而降级。静态多页报告是唯一无数据坐标交互的例外：`FrozenRasterView` 按原生像素显示，由 `FluentScrollArea` 浏览，不把 bitmap 重采样冒充 zoom/pan。
4. 普通 Fit 在原 panel 的 Setting/Edit 或 DataFigure 的 `Fit` tab 一步生效。TaskConsole 对当前已呈现的 exact snapshot 运行唯一 `BoundFit`，只产生瞬时 overlay 与 `fit.<parameter>` 派生 signal；拟合曲线/中心/半径必须在同一 exact Figure 上使用 frontend 唯一 fit style 直接可见，不能只返回 DTO 或依赖 source line 颜色。它不打开第二个 DataFigure、不创建本地 archive。DataFigure/FigureViewer 的显式 Save/Refit 走自身 artifact/archive 生命周期。Area 只有在用户点击 Fit 时才能成为 authority intent；viewport/relim/cmap 永不进入 FitSpec。
4a. Distribution 无需用户先点 Fit：renderer 对已经冻结的 histogram bin centers/counts 调用 zlc_data model owner 的窄 `analyze_bimodal_distribution`，显示 left/right/total 与仅在既定 separation 合同成立且两均值间只有一个交点时的 threshold。该结果是 `DISPLAY_ONLY`，不修改 `HistogramDisplayState`、不发布参数、不进入 artifact；用户实际拖动 threshold 后才形成普通 authored display commit。显式 Figure Fit 存在时完全覆盖自动显示分析，二者不得叠加或互相冒充 authority。Grid histogram cell 复用同一函数与 renderer。
5. TaskConsole 保留 header、常驻 Monitor/Logic、六种 Add Panel、Measurement/Processor/Task catalog、Start/Stop、统一 Setting/Edit、树形 signal picker、workspace Save/Load、panel size 与 selectors。普通 Measurement/Processor 不自动开图；Task 只可声明零到多个普通 default panel，并且只在对应 typed output 真实出现后建立。picker 以 current immutable front 的 `DatasetSchema` 判定 waiting/ready，并只列 owner 声明的合法 output。
6. PulseScan 的用户语义固定为“一份 pulse program + 一个已经运行的外部 `Signal(y)`”。所有合法 output 都可被选择，只有 producer-owned association 可进入正式 Start；GUI displayed/latest front 永远不是 scan authority。virtual association 不能替真实 qCMOS 资格背书。
7. Calibration、SiteMap 与 Occupancy 的适用交互面复用同一个 frontend Figure/GridPlot owner，提供 exact-cell focus、selector、zoom 与 export，不丢 validity、sample/value、dropped count、unit、exact ref 或 SITE address。Calibration 多页报告只显示已保存的 typed report facts；它按原生像素滚动，不重新拟合、阈值化或伪造普通 DataFigure axes。
8. 非标量数据可获得 role-driven 的低摩擦默认视图；任何进入 fit、scan-y 或派生 artifact 的选择/降维都必须显式冻结为 `CommittedTransform`。当前投影可见且可修改，但 display-only policy 不能静默升级为 authority。
9. gap、schema/hardware mismatch、缺帧、association/terminal/coverage 不完整都明确使 exact attempt 失败；错误保留可理解原因和原始失败证据，monitor 与 formal 状态不得混淆。SCAN_SLOT/MOT 继续使用冻结 bitstream 的 autonomous hardware timing，不存在 host-stepped fallback。
10. Stop 或资源切换不能提前 release。若 exact conflicting RunId 属于同一 TaskConsole 的另一 Logic 行，shell 可请求停止该行，等待真实 terminal 后重试同一冻结 request；外部 owner 明确拒绝，不能靠设备字段猜冲突、同步阻塞 GUI 或抢占。
11. monitor 可以跳过中间展示 revision；同一 neutral publication 的关联 signals 必须先完成物理 causal closure，Workbench linked-front 只让匹配的 presentation revision 整组可见。独立 producer 可以处于不同 revision；whole-board present 永远不能创造跨 producer same-shot。
12. TaskConsole 的一次性普通告知使用常驻 status strip 的最低 `info` 级，不弹逐条确认框；状态优先级保持 error > task > warning > info。任何 render/worker ownership 或 revision 握手未落定的 Edit/Save/Refresh 操作必须 fail closed，放弃该动作并在 status strip 以 `error` 指出动作名，不能继续读取或保存可能撕裂的画面。

## 3. 禁止的失败模式

本节只列架构必须机械排除的终态反例。

### 3.1 反向依赖与聚合命名空间

frontend 不得导入 neutral 的设备、runtime、Pulse 或领域 capability；neutral core 不得反向导入 frontend/Workbench；pulse/server/build 不得依赖 neutral。lazy import、根包 re-export、`core/common/utils` 聚合目录和 service locator 都不能掩盖反向依赖。每个 public symbol 必须能指向 §4 的唯一 owner。

### 3.2 Task、Measurement、Processor 与物理 capture 不得互相冒充

Task 是 one-shot use case，Measurement 描述领域采集语义，Processor 是 typed transform；三者的 Definition 只保存关闭 metadata。Task/Measurement/Analysis capability构造flat RunPlan；Processor capability构造自己的typed prepared application并由真实host消费，不冒充Run lifecycle。物理 camera capture 只有 node-neutral `BoundCameraCapture/FrozenCaptureSpec/CaptureSession` 一套合同，任何 Measurement、MOT、Calibration 或 release-recapture 都只能消费它，不能复制其 schema/cardinality、线程或 terminal owner。

RunHandle 只有在 worker、session 和 interrupt 真正退出后才能发布 terminal 并释放 ResourceClaim。join timeout、GUI row removal、同一个 event 同时表示 cancel/finished，或吞掉后台异常后把 `last_error` 写成空值，均不得改变 lifecycle truth。

### 3.3 轴、validity 与 authority 不得靠 shape 猜测

任何 fit、ROI、facet、measurement result 或 plot 都不得使用 `reshape(-1)`、隐式 trailing mean、按 singleton/rank 猜 kind、取第 0 项或改写 point shape 而不记录映射。显示层可以给出安全建议；Fit/Scan/derived artifact 只能消费明确的具名 Selection/CommittedTransform。

单个 `Value` 的 component mask 是 `ComponentValidity`；完整 `DataBlock` 的 component mask 是带明确 `(R,P)` 前缀的 `DatasetComponentValidity`。二者不能按 rank 或广播巧合互换，NaN/0 也不能替代 validity。

### 3.4 顺序、物化、presentation 与物理因果不得混名

单个 sample event、连续 monitor front、formal exact group 和累计 immutable Dataset 使用不同合同。producer 一次发布一个 immutable event；monitor 可 latest/coalesce；formal consumer 使用 lossless cursor；DatasetBuilder 私有增量写并只在 revision/final seal 时物化 Dataset。FollowTap 的有序交付不证明事件属于某次 FIRE；只有 producer-owned association evidence 可以证明。

Workbench linked-front、一次 whole-board present、GUI request 序号或几个 latest panel 的同时可见都只是 presentation coherence，不能创造 same-shot。多输出 Processor 必须在 neutral owner 内先以 direct input EventRef/join digest 完成 atomic sibling publication。

### 3.5 Figure、Qt、设备和持久化不得形成第二 owner

frontend 唯一拥有 PlotPanel、DataFigure、FitGrid、report、SiteMap presentation、selector、render/style 与 immutable front；Workbench worker/lane 只拥有作业生命周期、取消和 repository/file I/O，不拥有 plot kind、view、layout、composer policy、codec 或 present policy。Workbench Qt host 只原子安装 frontend 返回的 immutable front。Qt 与 worker 不共享可变 Figure/Canvas/artist；没有任何兼容 handoff 或共享-Figure 例外。

public Experiment、Definition、frontend DTO、Widget 和教程都不得可达 raw adapter、SDK handle、BoundDevice/drive-capable Port 或 `prepare/fire/acquire/configure` verb。artifact reader/writer 只接受 owner 的一个 current format；不能用 fallback、并行 codec、按文件名猜版本或 runtime upgrade chain 形成第二真相源。

### 3.6 硬件证明边界不得由软件补造

冻结 RTL/bitstream 只提供现有 wire/status/cursor/target 能力。host fingerprint、GUI progress、最终计数相等或 virtual 成功都不能补造逐沿 tag、per-fire counter、timing ROM、hardware one-shot token 或完整 timing signoff。真实 qCMOS formal association 必须由具体 installation、工作点、trigger margin、produced/drained metadata、coverage 与 exact terminal 逐 run 共同证明；证据不足即 typed NO-GO。

## 4. 最终顶层边界

系统由下列内聚 bounded contexts、领域中立产品骨架与一个不可合并的 application boundary 组成。这里先定义 Python import 与所有权边界，不把“必须拆成几个 wheel/repository”写进架构：desktop 可以捆绑安装；FPGA server 可只安装 pulse+storage；headless experiment 可安装 data+pulse+neutral+storage。

```text
zlc_data           通用 Value/DataBlock、axis/validity、Selection 语义、transform/reduction/fit
zlc_pulse          Pulse 文档、FPGA target/compiler、host、transport、RTL 和 build
zlc_neutral_atom   实验 framework 骨架，以及按 capability 纵向闭合的 devices / logic_nodes 实现
zlc_frontend       不依赖neutral的presentation层：View/Figure/DataFigure、SiteMap view/Area、render、selector与Qt组件
zlc_storage        canonical bytes/digest + content-addressed blob 与 atomic manifest 存储引擎
zlc_workbench      只接显式应用端口的领域中立桌面/Qt host 与产品骨架
Zou_lab_control    唯一稳定 public application API、installation/repository/runtime binding 与 desktop composition adapter
```

`zlc_neutral_atom` 的内部目录不是按历史调用阶段横切，而是固定为“骨架 + devices + 纵向 Logic Node”。骨架只包含 catalog/authoring/input-output contract、installation、generic Run/stream/resource/processing runtime 与跨节点确有复用的最小协议；`devices/` 包含具体设备 Port/adapter/SDK glue/endpoint、真实连接与 virtual physical implementation；`logic_nodes/` 按 capability 闭合具体 Task/Measurement/Processor。framework 可以依赖抽象设备 Port，logic-node core 可以依赖 framework 与设备 Port；logic-node 不得依赖 concrete adapter，设备实现不得反向依赖 readout/calibration/MOT 等 logic-node 语义。

当前 catalog 只有八个一等 capability，而不是一个可扫描插件宇宙：独立叶 `camera_measurement`、`pulse_scan`、`mot_field`，Readout family 下的 `calibration`、`occupancy`、`duration_fidelity`，以及 Release-recapture family 下的 `temperature`、`grey_molasses_detuning`。每个可枚举叶节点必须闭合自己的 Definition、typed Request/Config、专属算法、application、output/schema/materializer 与 artifact，并导出唯一 process-local、headless `LogicNodeDeclaration`。family 根只拥有同族叶节点确实共享的领域机制；该声明同时拥有 description、authoring fields、path presentation hints、dynamic-choice resolver、typed input specs、static/dynamic output declarations、default views、request build/bind；callback-bearing declaration不持久化，durable identity仍是DefinitionKey及owner codec。

普通Logic Node没有per-node TaskConsole form、binding module、presenter或attachment。`zlc_workbench.task_console.declaration_projection`机械消费`LogicNodeDeclaration`，生成同一份form、Setting/Edit、input picker、signal vocabulary与default panels；普通节点新增字段或output只修改declaration/owner，不修改TaskConsole。只有prepared command的真实启动调用形状无法由generic host表达时，capability根下才允许一个headless、可选`workbench_adapter.py`；它只把已准备命令接到generic live/output host，不声明字段、投影、presenter或lifecycle。只有 declaration + generic Figure 无法表达的产品交互才允许可选`ui/**`，其`__init__.py`必须inert；当前有证据的例外为PulseScan scan-table/slot、Calibration多页报告/创建面与Occupancy exact-cell导航面。它们的普通字段仍来自declaration projector，SiteMap/Figure交互仍委托frontend唯一owner；capability根可以re-export headless core，但不得eager import adapter/UI。

Calibration/Occupancy capability独占SiteMap领域事实：site axis、centers、validity、coordinate frame、calibration identity及其source/cell关系都属于neutral immutable value。`zlc_frontend`只独占SiteMap的typed view、Area派生、render与selector，就像它独占其它Figure交互一样；neutral不定义或托管concrete SiteMap presentation/projector。neutral Processor/node必须在发布前验证输入source revision/event、same-shot facts、join digest与全部sibling outputs，原子交付一个causally closed publication；缺任一输入或identity不符直接失败，不把latest frame与latest occupancy交给外层猜配。frontend只从该已闭合typed value建立immutable SiteMap view，Workbench只路由和present，不做物理join。

`logic_nodes/` 只容纳实验领域 capability leaf 与具名 domain family；不得把 family 共享机制伪装成第二个 Logic Node，也不得为了消除 sibling import 将它升成 framework/runtime。Readout family 拥有读出物理值与合同，Release-recapture family 拥有两帧同一次 loading 到 survival 的机制。只有同时跨独立领域族、公开 vocabulary 不含某一实验物理语义、且删除任一具体 capability 后仍有完整职责的机制才能离开 `logic_nodes/`；当前 `capture/` 的 exact camera acquisition/session/artifact 满足该门槛，`pulse_catalog.py` 只拥有跨族仓库 PulseDocument 的稳定位置。所有 owner 都不得退化成 `common/utils/core` 或建立 presentation UI。

`zlc_frontend`是与neutral执行域隔离的presentation层，`zlc_workbench`是domain-neutral产品host：frontend拥有Figure/Divider/Fit/selector以及完整SiteMap view/Area/render，Workbench拥有declaration projection、窗口host、Run/Processor lifecycle、typed input resolution与output routing。Workbench不能按shape或DefinitionKey解释领域数据，frontend也不能回查neutral repository/runtime。不存在按宿主横向聚合concrete capability的presentation/attachment目录。

Figure派生的Area/Cross/Fit signal由frontend一次发布完整typed facts：bare output name、contract id、operator短标签、axis/value label、description，以及确有权威输入语义时的typed source transform。Workbench只给bare name加稳定panel namespace，并机械适配成通用signal declaration/routing；它不得解析`area.*`/`cross.*`/`fit.*`前缀、检查selector/fit metadata subtype、重算contract/label，或按Area名字反推authoritative transform。新增Figure output只修改frontend owner，不修改TaskConsole。

`Processor` 是唯一的 catalog、领域和产品类型。TaskConsole snapshot evaluation 的 `latest-only` 是 host delivery policy；未来事件的 exact 交付只属于具体capability自有的derived `SignalEventSource`，finite artifact则由该capability自己的flat Run提交。三者不得按delivery policy产生第二套Definition/form/binding/node、public hierarchy或通用formal worker。TaskConsole Processor row只以host-local latest-only job调用已经prepared的Processor；host只拥有job lifecycle、cancel与owner-thread handoff，不认识Occupancy或任何领域output。replace-before-start不是pending容量预算，处理器未执行的中间revision也不得被写成producer采集gap。

每个 concrete capability 在自己的 `logic_nodes/<capability>/package.py` 导出唯一冻结 `LogicNodePackage`：它同时指向 leaf-owned declaration、public API binder、TaskConsole binder、显式 API dependencies、artifact capabilities 与可选 close。framework 只在固定 `zlc_neutral_atom.logic_nodes` namespace 下确定性寻找文件名严格为 `package.py` 的内建 leaf，排序导入、校验唯一 API name/DefinitionKey/order、冻结结果；没有注册 API、entry point、FQCN、运行期替换、fallback 或 service locator。composition 不得写任何 field key、default、input/output name、path hint、default-view、SiteMap 领域事实或 UI 策略；这些事实只来自 package 指向的 owner。

任何可作为Figure/Fit输入的FINAL artifact都由自己的capability/repository owner投影成同一个headless `ArtifactDatasetSource(schema, exact DatasetRevisionRef, optional OwnedSnapshot)`；是否只inspect还是materialize由owner projector显式参数决定。Capture、Scan、Occupancy等artifact的内部block字段、generation、repository admission与snapshot构造不能泄漏到`Zou_lab_control`。每个 `LogicNodePackage` 可绑定自己封闭的 `ArtifactCapability`，composition 把冻结集合交给 exact-type `ArtifactDispatch`；dispatch 没有注册、继承匹配或fallback，外层也不得读取`.frame_source/.output_schema/.occupied/.counts`、重复解释output字符串或自行调用repository-specific materializer。saved Fit、直接Fit、Figure与headless FigureDocument必须共用这条dataset-source seam；新增artifact只修改自己的 leaf package 与 owner。

`zlc_workbench/task_console`中的builder/projector只能按declaration机械形成通用Run/Processor host。Camera/MOT/Calibration等若只有普通fields/inputs/outputs，直接投影，不建立专用UI。`Zou_lab_control.api` 从同一冻结 `LogicNodePackage` 集合构造 `exp.nodes.<api_name>`；`Zou_lab_control.workbench._composition` 再调用这些 package 的 TaskConsole binder，把当前 Experiment 的窄 application context 和领域中立 `TaskConsoleProjection` 接起来。composition 不显式导入具体 capability，不拥有字段、物理binding、output schema、materializer、presentation或QWidget。

`Zou_lab_control` 不能删除或合并进其它包：它是唯一稳定的 public application API，同时拥有 application-level installation/repository/runtime binding，并为桌面产品把一个已建立的 Experiment 拆成窄端口。若并入 neutral，领域层将反向拥有 public application facade 与 repository/runtime composition；若并入 workbench，headless API import 将被 Qt 产品依赖污染。这里的“薄”按职责而不是机械行数判定：它可以维护 public convenience 方法、composition 生命周期和对冻结 package contract 的机械投影，但不能定义或重建任何 capability schema、算法、materializer、artifact interpretation 或 presentation；这些必须委托各自 owner。

public surface 位于 `Zou_lab_control.api`。跨 capability 仍有独立职责的 capture/load/materialize 脊柱由 node-neutral `ReadoutFacade` 保留；具体 Camera Measurement、Calibration、Occupancy、PulseScan、MOT 与 release-recapture API 则由各自 `LogicNodePackage.bind_api` 构造，并只通过 `exp.nodes.<capability>` 暴露。`LogicNodeApis` 在构造期按显式 dependency graph 一次建立、之后冻结，不提供字符串 dispatch、注册或替换。顶层 facade 只提供 `_ExperimentServices` 生命周期与 installation/repository/runtime binding，不重述领域实现。

`zlc_data` 不是新的 `common/utils`：它只容纳领域中立、headless、可序列化的数据语义和值上的纯算法。它拥有 Value/DataBlock、Axis/Validity/PointLayout、Selection/CommittedTransform、Reduction、FitSpec/BoundFit/FitResultBatch、closed model catalog 与同步 solver；`FitProblem` 只是包内瞬时 packing 值，唯一公开执行入口是 `BoundFit.run()`。它不知道 Hub、Run、Device、neutral artifact、Figure、Qt 或 Matplotlib。

**Dataset 输出所有权必须沿领域调用链闭合，不能落在 GUI。** `zlc_data`只拥有通用`(R,P,*data_shape)`载体、scalar的`(R,P,1)`物理表示、轴/validity/layout与具名选择/变换的机械 materialization；它不知道“Camera的一次cycle怎样拆成frame_0/frame_1”“Calibration发布哪些site量”“MOT/Occupancy/Scan的y是什么”。每个`zlc_neutral_atom` Measurement/Task/Processor application owner必须把自己的typed request、物理source contract、公开output names、schema/materialization、artifact admission与join lineage冻结在同一个prepared command中，并通过`live_dataset_outputs(...)`或`final_dataset_outputs(result)`发布完整typed outputs。preview的source ordinal、数据`BlockId`与物理分组同样由该application owner持有；Qt只实现typed preview port，不得铸造数据身份。Formal Scan 的 PROVISIONAL 与 FINAL 输出还必须调用同一个 neutral scan materializer：Workbench只能把prepared command返回的`OwnedSnapshot`交给Figure，不得自己调用通用transform重新生成scan输出或另造preview `BlockId`。

`zlc_workbench`/TaskConsole只拥有领域中立host composition：机械投影declaration，把用户选择解析为已有producer/ref，调用prepared command，托管Run/live port，给bare output name加实例namespace，并把已闭合typed outputs路由到frontend。它不得构造`AxisSpec/DatasetSchema/DataBlock/Validity`，不得按shape拆帧或reduce，不能解释artifact、做same-shot join或接收开放`result -> Dataset`projector。live port只持neutral声明的窄freeze/close协议。真正特殊UI由capability inert leaf适配generic host，但host不认识其字段；SiteMap presentation始终由frontend构造。`Zou_lab_control.api`只补installation/repository/runtime binding与public convenience orchestration，facade不得复制request、materializer或presentation。

TaskConsole的linked-front是上述物理边界之后的纯presentation gate：它可在某个已声明source→processor component的descendant revision尚未到达时保留完整上一front，等同一已接纳source identity对应的关联signals齐备后再一起显示；它不能重新验证算法、calibration或same-shot，也不能把两个独立producer连成一个物理事务。whole-board一次freeze/present同样只保证Qt看到一组不可撕裂的GUI fronts，不产生board-wide shot identity。

这里的“composition”只指窗口内部把typed controller、render surface与用户操作接起来，不授权`zlc_workbench`取得整个`Experiment`。每个正式窗口只接自己真实消费的显式端口：TaskConsole接冻结的`TaskConsoleApplicationPorts`，DeviceManager接`DeviceAdminPort`，PulseGUI接`PulseRunFacade + PulseTargetDescriptor`或独立connection factory，Capture/Scan接prepare/project等最小callable。端口字段必须逐项列出真实操作与immutable installation facts，禁止用`experiment/session/services/facade: object`、`__getattr__`、duck-typed service locator或单个generic `call(name, ...)`把整张对象图藏回来。`Zou_lab_control.workbench`是唯一可把已有`Experiment`拆成这些端口的桌面adapter；standalone launcher也必须先在该顶层composition层建立同一authority，再调用相同Qt入口。由此`zlc_workbench`既不能import `Zou_lab_control.api`/`connect`，运行时对象图也不能从window/controller反向到达Experiment、repository、raw adapter或其它未声明能力。

FitResultBatch 的 canonical payload codec 属于 `zlc_data`，但 durable identity 必须由最窄的 artifact owner 持有。现在已经出现两个真实、同构且均为 FINAL dataset artifact 的 consumer：`CaptureArtifact -> FitResultBatch` 与 `ScanArtifact -> FitResultBatch`。因此 neutral 只保留一个 `FitResultArtifactRef/FitResultRepository`，manifest 的 closed tagged source union 只能是 `CaptureArtifactRef | ScanArtifactRef`，并逐种委托 source owner 的 canonical serializer、exact DatasetRevision/schema validation 与 binding；它不是可注册 source 的 generic Analysis repository，也不拥有 Fit 算法、Processor 或 Figure。第三种 source 若不满足相同 FINAL dataset/replay 合同，必须有自己的 adapter，不能向 repository 塞 registry/plugin/fallback。`fit.save()` 不要求 frontend；Figure 保存仍使用 frontend-owned FigureArtifactRef；两者是不同 artifact kind。

“selector”必须拆开看：`Selection` 是可保存、可供 fit/processor 共同消费的数据语义，属于 zlc_data；鼠标手势、RectangleSelector、handle、overlay 和 interaction state 属于 frontend。`DataFigure` 明确属于 frontend，因为它是 render/public presentation facade。fit editor/overlay 属于 frontend，但它们调用 zlc_data 的唯一 fit 实现，不复制模型和结果 schema。editor 只能从 public immutable `fit_model_catalog()`/`fit_model_definition()` 取得模型与参数 metadata，并从 BoundFit 的 `parameter_definitions/parameter_units` 取得绑定后单位；不能导入 fit implementation submodule 或在 frontend 硬编码第二份模型表。`fit_projection_metadata()` 与 `validate_fit_authoring_options()` 是 DataFigure窗口、TaskConsole嵌入panel和Grid共同消费的唯一X/Y/FACET/BATCH投影owner；Workbench窗口之间不得互相导入私有`_...projection` helper，也不得各自按shape/rank解释fit轴。

zlc_data 用 `bind_fit(FitSpec, expected DatasetSchema) -> BoundFit` 冻结并验证 fit/batch axes、CommittedTransform、model 与数值策略，但不捕获尚未产生的数据。`BoundFit.run(OwnedSnapshot) -> FitResultBatch` 是当前 interactive、offline/artifact 与未来确有消费者的 formal 路径共享的唯一执行值；OwnedSnapshot 同时持有 frozen DataBlock 与 exact DatasetRevisionRef，禁止 adapter 只传裸 block 丢掉 lineage。当前 baseline 不建立 `DatasetInputSlot`、generic `AnalysisStep`、`FitAnalysisDescriptor -> DataAnalysisProgram` 或 post-materialization workflow：真实用户需求只是对已提交 FINAL Capture/Scan artifact 明确打开Fit并显式保存结果，现有 `FitResultRepository` 已完整拥有这条边界。只有出现自动/headless preset或正式下游消费者后，才先实现“FINAL dataset artifact -> 独立 flat analysis Run -> 自己的一次 FinalCommit”；只有真实领域要求 scan 与 analysis 成为不可分割的一个提交结果时，才另行设计 composite commit。neutral 不得定义 `FitProcessor`、`FitOperator` 或 neutral-owned `FitAnalysisDefinition`；Workbench 只把 current zlc_data Fit capability 嵌入现有 Figure 的 `Fit` surface。

`zlc_data.codec` 是该 bounded context 内 typed canonical bytes 的唯一 owner；当前只有确有 durable consumer 的 FitSpec/FitResultBatch 暴露 standalone current canonical bytes，并复用这一处 canonical round-trip 判定。Axis/DatasetSchema、Selection 与 CommittedTransform 只公开供外层 artifact 嵌入的 owner tree projector/parser；它们不各自预建无人消费的 bytes wrapper。大型 Value/DataBlock 更不经过通用 JSON/tree codec，真实持久化边界使用 binary chunk/CAS。已排期的 AnalysisPreset/保存 FitSpec 通过 public `fit_spec_to_tree/from_tree` 或 `encode/decode_fit_spec` 委托同一 schema owner；FitResultBatch 的 tree projector 仍为 codec 私有，公开面只给 current canonical bytes，不能顺势恢复 generic result tree/ref codec。各领域类型仍由自己的 projector/parser 负责，primitive bytes 继续委托 `zlc_storage.canonical`。

`zlc_pulse` 是一个逻辑 bounded context，而不是“为了目录好看必须独立发布的产品”。它内部包含 `model`（PulseDocument/IR）与当前唯一生产 target `fpga`（TargetSpec/compiler/wire/host/RTL/build）。FPGA server、sim/build 和 neutral sequencer adapter 已是独立消费者，所以禁止它反向 import neutral；若未来出现第二硬件 target，再在 pulse 内抽出 target Protocol，baseline 不预建插件系统。

`zlc_storage` 只拥有两类窄基础设施：其一是无领域类型的 canonical primitive encoding/digest（canonical map/list/scalar、ndarray header/bytes、hash 与 framing）；其二是 bytes/blob/manifest 的校验、fsync、原子发布和最小维护。它不定义 universal ArtifactRef、领域 schema 或 artifact kind。frontend、pulse、neutral_atom 与 data 各自拥有 typed Ref/值对象 schema 和 `to_canonical_tree`，但最终 bytes/digest 必须委托同一个 canonical encoder；跨包嵌值对象必须调用 owner codec，不能手写字段顺序。canonical/non-empty text、SHA-256 text、integer、finite/positive real 等标量不变量同样只由该 primitive 模块实现；领域构造器调用它而不复制 `isfinite`/type/range 检查。仅明确的人类/外部输入 adapter 可调用单一 `normalized_text` 先 strip，机器 identity 一律使用拒绝空白改写的 `canonical_text`。这样避免四份 canonical JSON/float/ndarray/digest/validator 实现，又不建立能收容领域类型的 `common` 包。baseline 只实现经过 probe 的 local filesystem commit；复杂 GC、多后端/分布式锁等出现真实第二用例后再扩展。

### 4.1 依赖方向

依赖 ratchet 按模块路径检查；special UI叶物理位于`zlc_neutral_atom` namespace不代表neutral core可以反向依赖GUI。下列箭头表示左侧可以导入右侧：

```text
zlc_data -> zlc_storage.canonical（只允许纯 canonical 模块，不允许 repository/I/O）

zlc_neutral_atom framework/runtime/devices/logic-node-core/declarations/workbench_adapter -> zlc_data
zlc_neutral_atom framework/runtime/devices/logic-node-core/declarations/workbench_adapter -> zlc_pulse public API
zlc_neutral_atom framework/runtime/devices/logic-node-core/declarations/workbench_adapter -> zlc_storage

zlc_frontend -> zlc_data
zlc_frontend -> zlc_storage
zlc_pulse -> zlc_storage

zlc_workbench -> zlc_frontend
zlc_workbench -> zlc_data
zlc_workbench -> zlc_pulse
zlc_workbench -> zlc_neutral_atom headless declarations/public protocols

zlc_neutral_atom/logic_nodes/<capability>/ui/<leaf>
  -> its own capability core + generic zlc_frontend/zlc_workbench APIs

Zou_lab_control.api -> zlc_data
Zou_lab_control.api -> zlc_neutral_atom
Zou_lab_control.api -> zlc_pulse public API
Zou_lab_control.api -> zlc_storage
Zou_lab_control.api -> zlc_frontend.figure / zlc_frontend.data_figure
Zou_lab_control.api[render] -> zlc_frontend Matplotlib/render leaves (optional)
Zou_lab_control.api[workbench] -> Zou_lab_control.workbench -> zlc_workbench (optional GUI launcher)
Zou_lab_control.workbench -> frozen LogicNodePackages + installed contexts/preparers/loaders
Zou_lab_control.workbench -> optional capability special UI leaf through package binders
```

箭头表示依赖。禁止：

- zlc_data 导入 frontend、neutral_atom、pulse、storage repository/backend 或 workbench；它只可导入 zlc_storage.canonical 纯模块；
- frontend 导入 neutral_atom、pulse 或 workbench；
- pulse 导入 data、frontend、neutral_atom 或 workbench；
- neutral_atom framework/runtime/devices、LogicNodeDeclaration、capability core或`workbench_adapter.py`导入frontend/workbench/Qt/Matplotlib；
- neutral_atom framework/runtime 导入任一 concrete logic-node 算法或 concrete device adapter；
- `devices/` 导入 `logic_nodes/`，或 `logic_nodes/` 直接导入 DCAM/remote/virtual concrete implementation；logic-node 只能消费设备 Port/descriptor；
- generic frontend/workbench导入任一concrete capability、`workbench_adapter.py`或`logic_nodes/<capability>/ui/` leaf；Workbench只能导入generic `logic_node_declaration`/runtime协议，frontend不得导入neutral；
- capability根`__init__.py` eager import/re-export `workbench_adapter.py`或UI、执行registration；根可re-export自身headless core。`ui/__init__.py`必须inert，普通headless import不得加载frontend/workbench/Qt/Matplotlib；
- neutral或workbench定义/构造SiteMap view、Area materializer、render/selector，或frontend自己从不相关latest inputs建立same-shot claim；
- zlc_workbench 导入 Zou_lab_control.api、调用 connect、接收或保存 Experiment；
- composition root 以外实例化 concrete adapters。

mechanical ratchet至少包含六组：第一，除`logic_nodes/*/ui/**`外的全部neutral graph（包括declaration与`workbench_adapter.py`）禁止frontend/workbench/Qt/Matplotlib import；第二，capability package root不得触达adapter/UI，所有`ui/__init__.py` fresh import保持inert；第三，generic frontend/workbench不得import concrete capability，TaskConsole源码不得出现具名DefinitionKey/field key/output名分支；第四，neutral Calibration/Occupancy capability是SiteMap领域事实唯一owner，frontend是`SiteMapPresentation/SiteMapView`、Area materialization、render与selector唯一owner，workbench不得出现同义领域或presentation projector；第五，每个reactive Processor publication在进入Workbench前必须通过exact source ref/event digest、single join digest与atomic sibling-output closure检查，具体Processor再验证自己的derived siblings共享其声明的revision/generation关系；禁止要求derived stream generation等于source generation，不同stream的same-shot只能由causal envelope证明；第六，Workbench linked-front测试只断言presentation revision gating，并明确覆盖两个独立producer可处于不同revision、一次board present不产生same-shot claim。只扫描顶层包import或只测GUI结果都不足以证明这些边界。

### 4.2 Composition roots

只允许三类明确的 application composition；其中只有普通用户入口公开领域 facade：

- `Zou_lab_control.api.connect(...) -> Experiment`：headless/application 实验根；
- `Zou_lab_control.workbench.open_*(Experiment, ...)`与standalone launcher：desktop adapter，只把一个已建立installation拆成逐窗口显式端口；`zlc_workbench.open_*`仅组装对应Qt产品，不建立installation；
- `zlc_pulse.server_app` 与 FPGA launcher：可独立部署的 current pulse server 根。

`zlc_neutral_atom.installation_package` 在固定 `zlc_neutral_atom.devices` namespace 下确定性发现每个 `package.py` 导出的唯一 `InstallationPackage`，校验 backend、config type、default、device plan 与 authoring schema 后冻结。每个 leaf 自己拥有 config codec、DeviceManager authoring、公开 topology 与 executable composition；`installation_dispatch`只把已验证 document 委托给 exact package，不保存 DCAM/remote/virtual 分支、物理 wiring、Measurement binder 或默认参数副本。machine AssetMap/adapter identity facts 属于更低层的 `installation_assets.py`，connection-lifetime runtime/catalog/graph 与逆序 close 属于 `installation_runtime.py`。当前 `Zou_lab_control.api.connect()` 只在 composition 边界建立这一 authority；不存在第二个 headless session root或顶层转发。

`Zou_lab_control` 是上述前两类 application composition 的共同、不可合并 owner：`api` 保持headless public API和installation/repository/runtime binding；`workbench`把同一Experiment映射成桌面窄端口，并机械消费已冻结的 installation/LogicNode package。它不通过mutable registry发现能力，不在facade中重述field/input/output/default-view、join、materializer或presentation。该包若膨胀，应把具体capability实现移回其owner，而不是删除application boundary或并入neutral/workbench。

可复用 library 内禁止通过 FQCN、entry point、开放包扫描或 service locator 动态构造依赖；唯一允许的扫描是上述两个固定内建 namespace 对严格 `package.py` 名称的确定性、冻结发现。

每个真实或虚拟连接由一个 composition-owned `InstallationRuntime` 管理。它在该次连接建立时构造私有、membership immutable 的 `InstallationDeviceGraph`，并同时拥有 `ResourceArbiter`、`DeviceBroker`、`RunController`、owner lanes、typed `DeviceBindingResolver` 与只读 catalog。`InstallationDeviceGraph` 只是该次活连接内的 exact role -> adapter owner/binding/close-order 图，不是旧 `DeviceSet`、registry 或 service locator；构造完成后不在运行中增删或替换成员。resolver 把 request 中的 `DeviceBinding(role/id, required capability)` 原子解析为：

```text
BoundDevice:
  resource_key
  binding_stamp:
    physical_identity:
      stable_device_identity
      evidence_kind
      evidence_digest
      asset_map_revision
    binding_instance_id
  safety/interrupt/session capability summary
  private broker reference
```

领域 owner/composition 的具名 binder 只能通过 resolver 一次取得请求显式声明的 BoundDevice，并把该对象放进领域私有的 immutable bindings；Definition 本身不执行 bind。Port、claim、affinity、capability 和 identity/generation 不能作为平行字段分别传递。generic runtime 不定义公共的泛型bindings容器；领域 composition 把 typed request/bindings 冻结在计划构造边界，随后只向 `RunPlan` 交付 `BoundDevice` 与阶段 callable。execute 不能按字符串回查 installation graph，resolver 也不能从全局 registry 隐式挑“第一个相机”。

#### 4.2.1 Connection-lifetime InstallationRuntime 与 public DeviceCatalog

`InstallationRuntime` 是单次活连接的 composition authority，不进入 public object graph。它唯一拥有该次连接的硬件图、运行 admission、broker binding 与 terminal shutdown。软件不保存“上次设备是否安全”的跨连接权威事实；新 authority 只能由新连接的 live identity 握手、当前硬件SAFE初始化与能力探测建立。session 不能分别保存 raw graph、binding registry、catalog 与 facade：

```text
InstallationRuntime                    # 单次活连接一个，不公开
  installation_id
  runtime_instance_id                  # 每次进程启动重新生成，不复用
  lifecycle = STARTING | RUNNING | CLOSING | CLOSED
  private immutable InstallationDeviceGraph
  ResourceArbiter / DeviceBroker / RunController
  typed DeviceBindingResolver
  typed domain facades/descriptors
  DeviceCatalogReader -> immutable DeviceCatalogView

InstallationDeviceGraph               # runtime私有，构造后membership不可变
  ordered adapter owners / owner lanes
  role -> BoundDevice
  deterministic reverse close order
  no public lookup / mutation / replacement API
```

连接建立顺序发生在 Run admission 开放之前：

```text
acquire backend/composition physical-owner proof
-> load and canonical-verify AssetMap
-> construct owner lanes and inert adapter owners
-> for each asset: open on its owner lane
   -> live identity readback + AssetMap match
   -> execute that adapter's current-hardware SAFE initialization
   -> DeviceBroker.verify_identity -> bind -> capability probe
-> freeze InstallationDeviceGraph, resolver, descriptors and initial catalog
-> open RunController/public command admission
```

ResourceArbiter 不提供 connection-establishment lease：正常 open/bind 发生时还没有普通 Run admission，互斥由 backend physical-owner proof、owner lane 与 `DeviceBroker.bind` 完成。若某 SDK 的 open 本身会改变输出，adapter 必须在该次启动中以硬件当前回读执行并验证SAFE初始化。任一 open、identity、AssetMap、SAFE初始化、capability 或 graph freeze 失败都不发布 `Experiment` 或 drive facade，composition 按 §12.7 关闭已经建立的子集并使本次连接失败。这个失败不写跨连接禁止状态；后续连接必须重新执行同一组live验证。

Target-owned virtual root 与真机共享同一composition形状：先构造并探测 deterministic in-process atom array、camera、sequencer与broker binding，成功后才发布facade。virtual的SAFE初始化由仿真硬件当前状态验证，而不是从先前软件记录恢复。正常`CLOSED`或启动回滚结束后，同一进程可以新建compose；先前close失败只是对应session的诊断，不能变成新compose的软件历史门禁。真实adapter、remote FPGA与qCMOS每次同样必须按上面的physical-owner/AssetMap/live SAFE顺序启动。

virtual graph直接消费 `zlc_pulse.load_deployed_pulse_target()`，不存在`PortCatalog`投影或第二份拓扑；canonical clock来自同一checked-in FPGA config。标准模拟接线只有实际被物理模型消费的cooling `ch00/ch01`、probe `ch03`、trap `ch09`、camera trigger `ch11`，以及`VirtualMonitorCamera`消费的`da_bias_x/y/z`三根MOT DAC（每根连同自己的latch clock）；其它deployed lane虽仍属于sequencer执行ABI，却不冒充模拟器能力、不进入virtual operator manifest。trap只作为camera背后的私有物理模型，不进入public catalog；public catalog只含camera/sequencer immutable `DeviceInfo`，关闭顺序固定camera→sequencer→trap。

这里“virtual=real”指两者共享`PulseTarget`执行模型与同一Run/compile路径，不指两者暴露相同数量的operator port。`PulseTarget`只拥有stable internal port key到ordered raw lane的执行ABI；该key只服务引用与原子重映射，**不是操作者channel name且不得作为界面首列**。可编辑`label/signal`才是operator-visible逻辑通道名，改名保持stable key从而不错误断开既有pulse引用；正因为它可编辑，领域代码也不得用`label == "probe"`、key字符串或lane序号猜物理角色。具体Measurement必须从冻结request的显式binding，或从已冻结waveform与设备/Calibration事实唯一推导角色并交叉验证，执行时只保留stable key/endpoint identity。`zlc_pulse.PulseTargetManifest`把一个backend实际暴露的port子集绑定到lane-aligned operator endpoint，并作为`SequencerCapabilitySnapshot/PulseTargetDescriptor`的一部分进入capability fingerprint。manifest是唯一endpoint来源，UI不得按rank、lane序号或本机文件猜测。`PulseDocument.visible_ports`只保存Offline authoring偏好；Virtual/Remote各自的当前显示集合是controller-owned view state并受各自manifest约束，Show All/Hide Off不得反写文档、制造dirty或覆盖Offline选择。切换manifest时Edit页一次投影对应集合；channel区域永久预留同一Fluent scrollbar gutter，内容从未溢出变成溢出也不能横向推动整块布局。

Remote server启动时同时读取canonical target与**server-side** XDC，逐lane验证XDC signal、DAC bit order、latch clock与target完全一致，随后在current RPC snapshot发布含package pins的manifest；客户端绝不读取自己的XDC覆盖远端authority，缺失/不一致在socket开放前失败。Virtual manifest只由installation-owned simulator wiring构造，因而精确发布5个digital与3个MOT DAC；它不维护fake XDC。package pin/simulator endpoint不进入PulseDocument或wire ABI，但PulseGUI Edit左列只显示当前manifest endpoint（如`F15`或`SIM:C0`），右列显示文档signal label，内部`chNN`不再冒充operator hardware name；Show All也只能在manifest公开集合内工作。

PulseGUI的同一个Target tab投影该manifest：Remote/Virtual全部只读；只有明确Offline模式可编辑草稿。正式界面复用`FluentScrollArea + FluentGroupBox`，Digital/DAC各是一张紧凑分区卡，内部用共享Fluent输入控件组成对齐行，不显示stable internal key，也不得另造原生`QTableWidget/QScrollArea`视觉体系。Offline可增删Digital，也可增删完整DAC port；一个DAC草稿原子包含signal、width、逐bit data endpoints与配对latch-clock endpoint，不能拆成多条TTL。任何字段变化在Apply前不触碰文档；Apply一次构造candidate `PulseTarget+PulseTargetManifest`，按stable port key重映射全部period states/DAC actions/delay/scan/API引用并发布一个新revision。删除或改宽已被权威内容使用的port先列出精确引用并要求一次显式“Apply and clear”；确认后才清除受影响引用与失效scan provenance。Qt控件树不是第二owner，在线模式也不存在可编辑client signal config。standalone PulseGUI从Virtual/Remote选择Offline并按Connect时，先在worker调用旧installation的领域`close_session`；无论关闭成功还是失败，旧facade都立即从GUI权威中摘除，避免Qt timer继续轮询已关闭Experiment。成功时原子恢复Offline manifest与Offline显示集合；失败只显示该次关闭诊断，不安装进程永久门禁。后续Remote/Virtual连接必须在worker上从零做live identity与当前SAFE初始化，不继承旧连接的软件判定。窗口关闭只有在全部tracked future/result已归零并且idle executor已经`shutdown(wait=True)`完成后才发布`close_complete`；不能让Qt/font资源先销毁而worker线程仍在Python退出路径中解栈。

PulseGUI的编辑态与运行态必须物理分开。scan code键入、尚未Apply的Target字段和其它临时输入只属于稳定Qt editor draft，不调用controller、不render、不读取硬件。unit/name/value/delay/binding/visibility等同线程语义提交由对应Qt handler直接调用领域命令；handler已经知道被编辑的stable id，命令只返回领域结果，窗口据此原位更新该widget及明确dependent，禁止再制造local-delta/snapshot/projection信封重述同一事件。后到的Preview、Run或connection completion只更新各自的窄runtime区域，不得补做一次全树`set_document()`。Scan workspace是Scan tab自己的component front，只在scan worker、schema、candidate或source事实改变时投影；不能因无关编辑重建，也不能与Run/connection/preview/close拼成application snapshot。初始composition分别读取editor、runtime和preview三个窄front；之后只有真实worker、连接、Run/cancel/close ownership边界发布各自的immutable结果。idle时timer关闭，active timer观察不到变化就返回`None`。保留的snapshot必须冻结真实ownership/consistency boundary（immutable dataset revision、Run/ack/capability跨线程观察或同一次presentation-coherent board front）；后一种只表示GUI revision不可撕裂，不证明物理same-shot。不得拿普通Qt编辑事件或周期全应用投影冒充。Preview只在打开tab或显式view intent时请求，编辑隐藏页面不会后台compile/render。Period/port控件按stable id长期存活：标量变化只写现有widget，Add只insert，Remove只delete，Reorder只move，visibility只hide/show；任何路径都不得清空layout再重建整棵Edit树。dirty标志由editor session在提交/保存边界维护，普通编辑不得为标题状态重新序列化整个PulseDocument。

`InstallationDeviceGraph` 只在 composition/runtime owner lane 内可达，也不能通过 debug property、generic resolver、callback closure 或 frontend ViewModel 泄漏。这里的 typed facades 是 runtime-instance-pinned、immutable binding surface/descriptor，不包含用户可变的 calibration convenience pointer或UI state。public `Experiment`只发布`device_catalog`与稳定的领域 convenience facade；每个 facade 操作在一个 composition 临界区恰好取得一次当前 RUNNING runtime snapshot，据此构造并冻结 request/binding stamp，不能分别读取 descriptor 和 runtime 指针。所有依赖标定的请求都显式接收 `CalibrationArtifactRef` 并在构造时与 binding/model 一起冻结；headless domain session 本身不是普通用户的硬件 service locator。

graph 的“immutable”指 role membership、adapter owner、binding membership 与 close order 在该次活连接内不可改；其中 live adapter/connection 当然会在 owner lane 内部改变物理/driver状态，但这些对象不向 graph 读者开放。transport disconnect、device removed、identity mismatch 或 capability invalidation 使当前 binding fail closed；不在一个active Run中透明reconnect。需要重连或改变 connection identity、adapter topology、installation config 时，先摘除当前facade并关闭该次runtime，再在同一进程中建立全新runtime；新runtime必须重做live验证，不复用旧binding事实。

catalog是只读观察值，不是设备容器：

```text
DeviceRef:
  installation_id
  runtime_instance_id
  role

DeviceInfo:
  ref: DeviceRef
  domain
  adapter_kind
  resource_key
  availability
  health

DeviceCatalogView:
  installation_id
  runtime_instance_id
  revision
  devices: tuple[DeviceInfo, ...]
  find(role) -> DeviceInfo | None
  require(role) -> DeviceInfo
  roles() -> tuple[str, ...]
```

这些对象递归 immutable、canonical-serializable，且不含 raw adapter、SDK handle、callable、lazy getter/setter、任意 callback 或 `configure/arm/acquire/prepare/fire/abort/open/close`。`require()` 只取 `DeviceInfo`，绝不解析 control capability。旧 snapshot 是合法历史观察值，但其中旧 runtime instance 的 `DeviceRef` 不能被任何 command facade 执行；authority 必须在触碰 adapter 前以零底层调用拒绝 stale ref。

`runtime_instance_id`、每个 binding 的 `binding_instance_id` 与 `catalog revision` 不混用。runtime instance 每次成功建立活连接都重新生成；broker 为每次成功 live bind mint 一个不可复用的 binding instance id；同一 binding 下纯观察 health 变化只推进 catalog revision。immutable installation graph 内不存在第二个与它一一对应的 local `connection_generation`/`binding_id`。任何在 shutdown 开始前排队但尚未进入合法 Run 的 command 都绑定原 runtime instance，并在 CLOSING 后以零 adapter 调用失败；任何新runtime都绝不接受旧 `DeviceRef`、request、RunPlan、binding 或 capability。Pulse RPC server 自己的 `server connection generation` 是跨进程 transport 事实，继续由 `zlc_pulse` owner 独立维护，不能与本地 binding instance 混名或合并。

generic catalog 只回答“安装中有哪些角色、当前观察状态是什么”。领域事实由具名、冻结的 facade descriptor 单源提供，例如 `Experiment.pulse.target` 的 clock/port/target facts、`Experiment.readout.camera_descriptor(binding)` 的 frame/trigger contract、`Experiment.trap.geometry` 的 site/grid geometry。禁止把这些异质事实重新塞进任意 `snapshot: dict`，也禁止 frontend/Definition 按 role 字符串从 catalog 找到一个对象后调用领域方法；否则 catalog 会退化成新的 service locator。

#### 4.2.2 配置边界：关闭旧连接后建立新runtime

baseline 没有在同一active runtime内替换config/device/virtual-real，也没有 `InstallationCandidate`、available/unavailable union、swap intent、transition generation、partial new graph 或 reconnect coordinator。以下变化都必须先执行 §12.7 关闭当前runtime，再在同一application process中从 canonical config 与 AssetMap 完整建立新runtime：

- AssetMap、physical asset、adapter kind、endpoint 或 topology 改变；
- real/virtual backend 改变；
- 会改变 installation graph、owner lane 或 connection identity 的 machine/device config；
- 需要重新 open/reconnect 已失效 binding 的故障处理。

实验 request、pulse parameter、camera working point、calibration ref 与 panel state 不属于 installation graph，可以按各自 typed contract 在 Run 边界改变。UI 可以在worker请求关闭旧runtime，并在旧facade摘除后请求新连接；Qt callback 不直接执行硬件 close。旧close失败不得使整个进程进入永久reconnect-required状态；新连接能否接管只由它自己的physical-owner、live identity与当前SAFE初始化判定。

同一 RUNNING runtime 内的 catalog 异步通知只描述 health/observation revision，不承担 authority 事实，也不能有“先读、后订阅”的丢失窗口。`DeviceCatalogReader.snapshot()` 与 `watch(after_revision)` 在同一 owner 临界区线性化并返回完整不可变 snapshot；UI 可 coalesce 到最新 revision，检测 gap 时重新读取 current snapshot。shutdown 开始后 reader 只报告 terminal runtime lifecycle，不能发布一张看似可驱动的新 catalog；hardware safety 从不等待 subscriber ACK。

#### 4.2.3 adapter 作者、测试与 simulation 的命名空间

普通 `Zou_lab_control.neutral_atom`/`Zou_lab_control` umbrella 不导出 adapter base、concrete adapter、DeviceSet、loader、server bootstrap 或 raw pulse helper。adapter 作者只使用对应 `zlc_neutral_atom.devices.<kind>.contract` 与 parameterized contract kit；virtual/fault-injection 使用 `zlc_neutral_atom.devices.simulation`/`testing`，真实 server 使用自己的 application/CLI bootstrap。device contract 可以公开最小生命周期/Port 实现合同，但不能成为普通 Experiment 对象图的一部分，也不能提供 `lookup=globals()`、包扫描或运行时任意注册逃生口。真实adapter的构造/open/drive还必须消费composition owner签发的不可伪造owner capability并绑定owner lane；仅从owner module导入类不能得到可运行的真实硬件对象。Python反射不作为恶意安全沙箱，但普通协作代码绕过authority必须在构造或第一次drive前fail closed。

### 4.3 Data 与 Frontend 内部层次

```text
zlc_data <- zlc_frontend.figure / data_figure / plot_panel
                  |                         \
                  v                          v
      zlc_frontend.render DTO       matplotlib_render + render_style [render]
                  ^                          |
                  |                          v
      zlc_frontend.qt_widgets.FigureSurfaceHost/Lane [qt]
```

所有权：

- `zlc_data`：Axis、Value/DataBlock、Selection、DataTransform、Reduction、Fit；
- `frontend.figure`：ViewIntent、ViewSpec、FigureDocument、FigureEvaluator、FigureArtifactRef、codec；
- `zlc_frontend.figure`、`data_figure`、`plot_panel`：`ViewSpec/FigureDocument/DataFigure/PlotPanelContract/PlotPanelSession` 与 Figure output 语义，是所有 plot surface 的唯一高层呈现 owner；
- `zlc_frontend.render`：immutable raster/presentation DTO 与并发中立的 front/presenter 合同，不加载 Matplotlib/Qt；
- `zlc_frontend.matplotlib_render` + `render_style`：Agg renderer、权威字体/几何/palette/publication style、串行 Matplotlib compose lane 与 DataFigure 的可选 render backend；
- `zlc_frontend.qt_widgets`：Qt application/window lifetime、Qt event adapter、immutable raster board、Qt/QPainter style token 与通用 widgets；`FigureSurfaceHost` 原子提升 raster、exact DataFigure/display/contract/source identity 和 Area/Cross authority，`FigureSurfaceLane`单线程拥有 Agg/composer 与尚未开始工作的 latest-only coalesce。`SinglePanelHost`、`FacetedPanelHost`、`QtRasterBoard`只是其内部 presenter/gesture primitives，产品不得直接重建语义事务；
- neutral Calibration/Occupancy capability：SiteMap的site axis/centers/validity/coordinate frame、calibration/source/cell identity与publication前same-shot causal closure的唯一领域owner；
- `zlc_frontend.site_map*` 与 `figure_outputs`：已闭合typed inputs到SiteMapView、Area/Cross derived signals、render/selector的唯一presentation owner；不负责采集、领域SiteMap事实或证明same-shot；

neutral_atom的framework/runtime/devices、LogicNodeDeclaration与capability core只依赖`zlc_data`、`zlc_storage`、必要的`zlc_pulse` public API及自身headless协议，不依赖frontend/workbench。真正特殊的`logic_nodes/<capability>/ui/<explicit_leaf>`才可消费generic frontend/workbench；只有 `package.py` 的惰性 binder 在 composition 时可以明确引入该 leaf，普通 capability import 不得隐式带入。

`zlc_data` base 只依赖 NumPy/必要 solver与 `zlc_storage.canonical`，不加载 repository I/O、Matplotlib/PyQt。`zlc_frontend` 的 headless figure 与 raster/presentation DTO 层依赖 data+storage；Matplotlib backend/style/font 放在 `[render]` optional extra，PyQt/Fluent/Qt board 放在独立 `[qt]` extra。完整 Workbench 同时安装 `[analysis]+[render]+[qt]`。composition消费冻结的LogicNodePackage；只有特殊capability UI leaf消费render/Qt extras。application API 的显示 extra 依赖 `zlc_frontend[render]`，其可选 `[workbench]` extra 才懒加载 `zlc_workbench` GUI launcher和真实特殊UI。`zlc_frontend`、`zlc_frontend.figure`、`zlc_workbench`、`Zou_lab_control.workbench`、普通capability根与`ui/__init__.py`顶层 import 都不能加载 Matplotlib backend、PyQt/qframelesswindow、repository backend 或真实 hardware adapter；调用者必须显式进入具体 render/Qt/UI leaf。

#### 4.3.1 Qt 组件、表单与 presentation 的单一 owner

通用 Qt 组件只属于 `zlc_frontend.qt_widgets`，application/Workbench 只从其 curated public facade 取件，禁止 deep import 或从 `zlc_frontend` 根重导出 Qt symbol。`QtOwnerWake`、语义交互板与 encoded-raster presenter 同属这个 Qt leaf；导入 headless frontend 不得加载 PyQt、qframelesswindow、Matplotlib 或 IPython。

| 需求 | 唯一通用组件 | 边界 |
|---|---|---|
| QApplication、缩放、窗口可达性、保活 | `ensure_qt_app/set_fluent_scale/screen_fit_window_size/center_window_on_primary_screen/retain_window/release_window` | 首次 app 只能在 Python main thread 创建；异步窗口只在 committed close 释放 |
| 普通 Setting/Edit 行 | `FluentSectionLabel + FluentSettingRow + setting_label_width` | 一列标签+一列控件；不得用 `QFormLayout` 建第二种风格 |
| 稠密 authoring grid | `FluentFormGrid/FluentLabeledField/Metrics` | 只用于多列、跨行或统一 row metric，不替代普通 SettingRow |
| 路径、只读值、typed binding | `FluentPathEdit/ReadoutEdit/ScanLineEdit/TreeComboBox/TriStateToggle` | edit buffer 与 committed resource 分离，验证成功才提交 |
| 状态与提示 | `FluentStatusStrip/StatusDot/ScanDot/muted_note_label` | presentation-only，不持有 Run 或领域状态机 |
| 模态消息、确认、单行文本 | `fluent_message/fluent_confirm/fluent_text_prompt` | 只返回临时选择/文本，不直接提交领域对象或替代 validation |
| 容器与滚动 | `FluentGroupBox/TabWidget/ScrollArea/Frame/Popup/Window` | 按真实 lifecycle 组合，不因外形相似强套同一个顶层 Window |
| encoded report page | `FrozenRasterView/QtOwnerWake` | 只 present frontend 已编码页面；按原生像素显示并滚动，不产生 Selection/Fit/ROI 或 bitmap zoom |
| live/typed plot | `FigureSurfaceHost + FigureSurfaceLane` | 原子消费 frontend immutable front、typed context与derived outputs；内部复用`SinglePanelHost/FacetedPanelHost/QtRasterBoard`，唯一拥有即时 Area/locked Cross overlay、zoom/pan、size/DPR promotion 与 exact-origin intent |
| 批更新与 Qt hygiene | `batched_updates/signals_blocked/apply_fluent_scrollbars` | 禁止各窗口复制 signal blocking 或 scrollbar QSS |

声明式字段的数据流固定为：

```text
typed Request/Config schema owner
  （唯一拥有 value type/default/unit/range/required/static choices/description）
        |
        | Workbench use-case 的显式 projector；普通 import，不按 schema-id dispatch
        v
zlc_frontend.form.FormSpec / FormFieldProps
  （headless、immutable、只增加 label/group/order/widget/layout hint）
        |
        v
zlc_frontend.qt_widgets.FluentParameterForm + closed handler mapping
  （normalize/build/read/write/is_empty/refresh/full-state populate；不 Apply）
        |
        | keyed draft values
        v
Workbench EditorSession -> typed Request/Intent constructor -> owner validator
  -> base_revision 检查 -> atomic Apply
```

`TaskDefinition/MeasurementDefinition/ProcessorDefinition`只保存 catalog identity 与 request/config schema identity，不复制字段默认值或 GUI schema。字段语义在 typed Request/Config owner 旁有且只有一份声明；Workbench 的显式 projector 必须机械证明 key 全集、默认值、range 和 choices 与 owner 声明逐项相等。`FormSpec`不持久化、不进入 artifact/config fingerprint、不成为第二 validator。

领域输入同样走这条路径：owner 用关闭的 `DatasetInputSpec` 与 `ArtifactInputSpec`声明稳定 output-contract id、delivery/authority、typed artifact-ref schema 与允许的明确来源；producer owner 为每个 public output 声明稳定 contract id。动态 `frame_i` 共享一个 Camera-frame contract，GUI 不按名字前缀、DefinitionKey、shape 或 rank 猜类型。Workbench 只有一套 typed input projector、兼容 output 过滤器与 resolver。

`FluentParameterForm`按有序 FormSpec 构造现有 SettingRow，保存 `key -> (handler, widget)`，并提供 exact-key 的 `read_all/write_all/is_empty/refresh` 与原子、signal-blocked full-state populate。所有字段必须先 `normalize` 成功才开始改 Qt；未知、遗漏、重复 key 或非法 saved value均 fail closed，不能留下部分新值+部分旧值。Setting 与 Edit 消费同一个 FormSpec 和 committed state，但各自拥有 widget instance 与开始编辑时的 `base_revision`；refresh/populate 不得触发 Apply。

普通必填 `int/float` 使用 `FluentSpinBox/FluentDoubleSpinBox`；缺失的一侧领域范围只采用控件可表示范围，不写回领域合同。只有空白本身合法的 optional numeric 才使用可空文本编辑器。Pulse table、API segment table与普通 form 的自由数值文本共同调用 `zlc_frontend.form.parse_number_text`，保留输入的 `int | float`，统一拒绝表达式、NaN 与 Infinity；不得各自 `float(text)` 或复制 regex。权威 pulse duration/delay 使用 non-quantizing 模式。

复杂结构只豁免行列结构、增删、selection 与整表 commit，不豁免普通叶值。PulseDocument/scan/API table、真正多行 authoring、Figure selector/Fit 与 authoritative DataTransform 可以使用 owner 的显式 presenter；其中普通 path、signal、artifact、choice 和 numeric leaf 仍由 owner schema 注入。CalibrationArtifactRef 是普通 typed artifact input，不因领域名称另建 picker。不存在 generic table/workflow/widget-plugin framework。

动态 option snapshot 必须带 owner revision/generation：仍合法的选择保留，已失效的 committed value 可见但不可提交，旧 revision draft 明确 stale，绝不静默选择第一项。真实 live control 的提交节奏与 teardown 由其 `ControlTopic`合同声明；普通 Qt draft 不进入 worker mailbox。需要 leading+trailing 合并的控件按 key flush，Apply/Close 前必须提交最后一次合法值。

组件与绘图还必须满足以下冻结规则：

1. 控件复用按行为合同而非外形判断，必须保持 value range、float precision、commit/rollback、signal ordering、keyboard/wheel 与 enabled semantics；现有组件无法保持合同时才允许最窄的 raw Qt complex table/value editor。
2. `FluentPathEdit`的值是 edit buffer，不是已提交 PulseDocument/calibration/resource；验证失败必须恢复最后 committed path 与原对象，不能显示“坏路径+旧 authority”。
3. presentation-only 高层 widget 只有在至少两个真实 consumer 具有相同交互、视觉和 lifecycle时才能进入 `qt_widgets`；widget 不得持有 RunHandle、repository、Definition、scan/calibration intent 或领域 revision。
4. 各 application shell 独自拥有 cancel/reap/close；不得因外形相似抽 `GenericRunPanel/RunControlStrip`。launcher 只在窗口完成 cancel/reap/worker shutdown 后释放 retention，不能把可能仅 hide 的 `QWidget.close()`当成销毁完成。
5. 顶层 launcher 在构造 widget 前解析 process-wide Fluent scale，构造后先 screen-fit，`show()`取得 native frame 后再居中。大内容区使用 `FluentScrollArea`，validation/status 与 Apply/Cancel footer 留在滚动区外；最小 `800×600`、Windows 125% DPI 时 `availableGeometry`仍必须包含 `frameGeometry`。
6. 动作颜色固定：`Start/Run/Apply=GREEN`、`Stop/Hold/Load/Paste=ORANGE`、`Cancel/Remove/Clear/secondary navigation=GREY`、普通非危险主操作 `=ACCENT`。颜色只作提示，enabled/Run state仍由领域 owner 决定。
7. Qt chrome/QSS/window metrics/QPainter token 只属于 `[qt]` owner；字体、geometry、palette 与 publication defaults只属于 `[render]` owner，所有公开 nested token递归 immutable。Agg 不从 Qt 读取颜色或 fluent scale，Qt 不从 Matplotlib rcParams决定控件视觉。
8. Matplotlib `rc_context`会改变进程全局状态；frontend 的所有产品 Figure construction/draw/save/clear 使用同一个 re-entrant compose lock，并由同一 frontend session创建、使用和释放 Figure/Canvas/artist graph。Workbench lane只托管调用、取消与文件 I/O，不实例化 composer，也不存在跨 Qt/worker 的共享 Figure 或 ownership handoff 例外。
9. 屏幕几何固定为 `physical raster px = qRound(logical panel px × devicePixelRatioF())`；worker按 `DESIGN_DPI × PANEL_DISPLAY_SCALE × DPR`构图，Qt 一对一 present，禁止先按 DPR=1 渲染再拉伸。`zlc_frontend.panel_size.DEFAULT_PANEL_SIZE = "2x2"` 是普通 panel named logical size 的唯一默认 owner，`PANEL_SIZES` 是唯一 vocabulary，named panel 的 `logical_size/raster_size/dpi`只能由 `zlc_frontend.plot_layout.panel_surface_geometry()`一次派生；`PanelConfig`、`PlotPanelContract/PanelComposer`、普通 PlotReport、普通 DataFigure/FigureViewer 与低层 fallback 只能消费这些owner，不能重写默认、各自校验size、round DPR或从raster反推logical size。fresh Grid 的一次初始建议由 `optimal_grid_size_for_view(schema, view)`单源派生，Pulse preview由自己的 `optimal_pulse_size(channel_count, period_count)`单源派生；用户显式选择或archive中已保存的`size_name`始终权威，后续data revision不得重算。跨屏/DPI变化生成新 presentation revision；导出不携带screen DPR，只在terminal render提高 frontend-owned export pixel ratio。对同一 authored panel，导出提高分辨率禁止把普通 `2x2` 偷换成 `4x4/8x8` 等另一 named logical size；只有 Grid/Pulse 内容布局本身明确要求时才使用自己的初始策略。
10. 所有 plot kind共用 frontend 唯一 `FigureSpec + Divider + artist policy`：size/kind token一次决定 Figure、axes/data box 与 margin，禁止每个 renderer 使用 `subplots_adjust/tight_layout/constrained_layout` 或按当前文字/data反推边界。viewport只修改相应 axis limits；title、row label 与 plot bounds在同一结构 revision的pan/zoom全程固定。live single/faceted/SiteMap session在topology不变时保留同一Figure、Axes与artist graph，只原位更新data、limits、text和overlay；source exact ref只进入evaluated input/coherence stamp，不进入artist-owner identity。Curve/Histogram/Image各自声明的typed display state必须由同一renderer完整消费，遗漏x-view、relim、fixed range、count scale、colormap或viewport均是fail-closed合同错误，不能退回autoscale默认值。
11. `2d / sites / 1d / monitor / hist / grid`及 pulse preview在同平台、字体、logical size、DPR、确定性 evaluated DTO 与 display state 下，静态 raster必须与 UX oracle逐像素一致。Figure/data-box、ticks、labels、legend、font、color、alpha、linewidth、marker、distribution/colorbar/rail、site ring、grid cell/gutter/focus都属于该像素合同；只有 QPainter 与 Agg 在 selector边缘的单像素 fringe 可作为已说明的 rasterizer差异。
12. 完整静态图面只由 frontend painter绘制；Qt只 present 同 DPR front并绘制 Area、locked Cross 与 drag handles等瞬时 overlay。Qt不得重画 axes、label、colorbar、distribution、legend、site rings或Grid layout；plot 不提供 pointer-motion数据 hover。

机械 ratchet 必须证明：production只从 curated `zlc_frontend.qt_widgets` facade取件；`zlc_frontend`中的 PyQt/qframelesswindow import只位于 `qt_widgets/**`；Qt leaf不导入Matplotlib、neutral、pulse、Workbench或 `Zou_lab_control`；render backend/style不导入Qt/IPython；fresh package-root import不加载optional backend；所有 application Qt shell复用上述 application、scale、screen-fit、center、retention、FormSpec、style、renderer与selector owner，不建立同义实现。

#### zlc_data 边界规则

一个类型/函数只有同时满足下列条件才进入 zlc_data：

1. 输入输出只由 zlc_data 值对象、NumPy 数组和标量组成；
2. 不知道实验设备、shot/Hub/Run、neutral artifact、panel/plot kind 或用户 session；
3. 同一语义确实被至少两个上层 context 消费，或它是 DataBlock 正确性不可分割的不变量；
4. 可用纯函数/冻结 spec 表达，并有任意 axis/validity 的 property contract。

因此通用 curve/image fit framework、明确数学模型、batch result 属于 zlc_data；readout threshold decision、PSF calibration、occupancy/fidelity 等带中性原子物理语义的模型属于 neutral Analysis或具体Processor capability，即使内部复用 zlc_data 的数值求解 primitive。领域 model id/quality gate 由 neutral codec 保存，不能塞进 zlc_data 的 built-in fit catalog。

`Selection` 只保存 AxisId、typed geometry/range/index、CoordinateFrameId 与必要的稳定坐标参数；不允许 arbitrary metadata dict、widget scope path、plot-kind binding、ControlTopic payload 或 JSON byte-packing。frontend 的 facet/cell scope与 transient drag state 是展示状态；Workbench只把 committed Area/Cross/Fit投影成panel-owned派生signal，不把Selection反向解释成Measurement控制。

例如 ROI processor 使用自己声明的 `RoiBinding(selection, source_axis_ids, coordinate_transform, reducer)`，Figure facet 使用 `PanelSelectionBinding(selection, layer_id, facet_coordinate)`；二者共享 Selection，但不会把 processor reducer 或 panel cell path 写回 Selection。这样保存/复用选择不需要 frontend 与 neutral 约定一个隐藏 metadata vocabulary。

`BoundFit` 是由 FitSpec 与预期 DatasetSchema 确定性绑定的进程内执行值，不进入 artifact、pickle 或 FQCN import。持久化只保存 FitSpec/CommittedTransform、具有完整数学语义的 `model_id`、constraints、numeric policy、AxisSpec 与 input lineage；initializer/solver 都是当前 closed implementation 的实现细节，不再复制成字符串 identity 或对象 digest。重放时调用当前 zlc_data `bind_fit`；public FitSpec owner codec 只接受当前字段集合，不维护 model/algorithm 版本号、兼容 reader 或升级器。zlc_data 只公开 immutable closed catalog view，不提供可变全局 model/analysis registry，也不扫描 entry point。

### 4.4 Pulse Preview 边界

Pulse bounded context 不导入 frontend。通用 `FigureDocument` 的 image/curve/histogram/meter primitive 无法无损表达同板 mixed digital step、DAC ramp、period、repeat bracket 与 nominal scan reference，因此 Pulse Preview 不再伪装成通用数据 Figure。当前路径是：

```text
zlc_workbench.project_pulse_preview(PulseDocument)
  -> zlc_pulse.compile_pulse_artifact(STATIC_ONCE | STATIC_REFERENCE_POINT)
  -> zlc_pulse.build_pulse_timeline(PulseDocument, CompiledPulseArtifact)
  -> immutable PulseTimelineDocument
  -> Workbench 提取 frontend-owned plain pulse render input
  -> frontend 唯一 pulse renderer -> RasterBuffer + PulsePanelPayload
  -> SinglePanelHost / QtRasterBoard
```

`PulseTimelineDocument` 只是一组 renderer-neutral、进程内 immutable row/segment/annotation 值，不是新的持久格式、版本化 schema 或可编辑真相；保存/加载仍只有 current `PulseDocument`。离线 editor 因此只需文档自带的 target/time grid，不构造假 `DeviceRef`或 `PulseTargetDescriptor`；在线 composition 只在执行边界另外注入 live descriptor/facade，并通过显式 `bind_target()` 将文档重绑到当前 target。projector 只接收携带 `source_document_digest` 的完整 `CompiledPulseArtifact` 并核对源文档，禁止把 A 的 TargetIR 与 B 的标签、visible rows 或 digest 拼接。scan-authored document 的默认预览明确编译 nominal literal reference，绝不把第一行或任意孤立 scan row 冒充具有前序 DAC carry 的精确物理点。API/scan nominal reference 在画面中可见标注；但 nominal API 只允许预览，hardware `PulseRunRequest` 必须先通过 `resolve_api_parameters` 显式提供值并清除已解析声明，仍有任何未解析 API parameter 即拒绝 Run。Workbench 是同时看见 pulse 与 frontend 的 composition seam，只做一次不含语义猜测的 plain render projection；frontend 不导入 pulse。`PULSE_CONTRACT` 的类型边界、pulse renderer 与 `PulsePanelPayload` 共同保证它不进入 DataBlock/ViewSpec/FigureEvaluator/通用 Figure codec，也不把 display-only time span 提升成 zlc_data Selection authority。Qt 只呈现不可变 raster/viewport，不另建第二个 pulse renderer。export 只把同一 timeline drawing 输出为不可变 raster/vector，不反向恢复 authoring document。

Pulse front 不得借 dataset provenance 过桥。`zlc_pulse` 为完整 timeline 内容提供单向 fingerprint（覆盖row activity、segments、annotations、reference与时钟）；Workbench 投影成 frontend-owned `DocumentInputIdentity(document_id, document_revision, content_digest)`，renderer 只携带该输入身份。host 在 panel/layout facts 已知后才补 `DocumentPresentationStamp(source, presentations)`。这是输入冻结与present冻结两个时点所必需的最小两层，不增加run/schema/block字段。`PanelFrame`、interaction origin与coherence group必须同族：dataset和document不能混组；pulse range始终只是带Document origin的display gesture。`DatasetId("pulse.preview")`、假`DatasetRevisionRef`、第二套QPainter timeline与pulse→DataBlock适配器均不得存在。

Pulse Preview 的selector手感以 §2.1 的正式UX oracle为权威：wheel与middle-drag在每个鼠标motion产生display-only viewport intent，不能只在release更新。worker以单worker latest-state方式追赶，但同一immutable document、同一gesture中已经完成且revision单调前进的中间raster必须立即present，不能因其已非“绝对最新”而全部丢掉；最新pending仍保留并最终收敛。失败只撤销exact pending intent，已经消费的display revision不得回退复用，且整个路径不得改变`PulseDocument`、editor revision、dirty、scan request或artifact。Pulse plot使用正式`FigureSpec + Divider`的固定几何语义：size/kind token一次决定data box与margin，viewport只改xlim；禁止每帧用`constrained_layout/tight_layout`让tick文字或scan badge反向移动axes、title与row labels。Pulse的fresh初始size只由完整document topology调用`optimal_pulse_size(channel_count, period_count)`决定；显式用户size随后权威，普通panel的`DEFAULT_PANEL_SIZE`与Grid策略都不得介入。

Preview 的稳定 `SinglePanelHost` 从构造起就归最终 Qt owner，但在第一张完整 raster front 通过校验并原子 present 前必须保持隐藏，页面继续显示正式 placeholder；不得让尚未进入layout的parented child按Qt默认geometry自行露出，也不得把`QtRasterBoard`的空黑底当加载态。首帧提交后只显露同一个长期存活host，后续revision不得替换QWidget树。

Preview 的repeat标注由一个纯presentation policy从authored period span派生，Edit摘要与Preview不得各写一套判断：没有有限bracket时完整物理frame显示外层`×∞`；有限bracket只覆盖部分period时显示外层`×∞`与内层`×N`；有限bracket覆盖完整period table时只显示`×N`。外层frame必须包含延迟输出tail，内层仍严格停在authored period边界。关闭“show off”且全部digital row均为off时，仍保留第一条digital baseline作为空间参考；这是非权威显示fallback，不能把该row标成active、修改`PulseDocument`或影响编译/执行。

### 4.5 Application-first Experiment 门面

脚本、notebook 与桌面入口共享同一个一等 application composition root。`Zou_lab_control.api`提供薄 `Experiment`门面，把私有 `InstallationRuntime`、repositories、RunController 与领域 API 显式组合。一个 `Experiment`、其 `PulseFacade`、node-neutral `ReadoutFacade` 与冻结 `LogicNodeApis` 共享同一个私有 `_ExperimentServices` 生命周期 owner；不存在全局 token registry、service locator 或动态 facade registry。短操作在该 owner 的 state lock 内借用服务，长操作只登记真实在飞 operation 供 close 等待；关闭后所有 facade 由同一 `CLOSED`状态拒绝。

`ReadoutFacade`只保留跨 capability 仍成立的 capture/load/materialize 与 readout binding。每个具体 capability 的短 API 与窄 Host Protocol 留在自己的 logic-node leaf，并由该 leaf 的 `LogicNodePackage.bind_api` 组成 `exp.nodes.<capability>`；package dependency 是冻结的具名 DAG，composition 只实现 Host 所需的 installation/repository/runtime 接线，不解释 request 字段、算法、artifact、Figure 或 UI 策略。

```text
Experiment                                  # public application facade
  .readout / .pulse                         # node-neutral capture / pulse facades
  .nodes.camera_measurement
  .nodes.calibration / .nodes.occupancy
  .nodes.pulse_scan / .nodes.mot_field
  .nodes.temperature / .nodes.grey_molasses_detuning
  .device_catalog                           # immutable observation only
  .pulse.target                             # target descriptor, not raw sequencer
  .run(request) / .start(request) / .inspect(request)
  .fit(capture_or_scan_ref, spec|model=...) -> FitExecution
  .fit_gui(...) / .figure_gui(...)          # frontend DataFigure owner
  .figure_document(...) / .figure(...)      # headless projector / optional render
  .nodes.<capability>.<typed operation>(...)
  .task_console() / .pulse_gui()             # optional workbench lazy import

neutral domain Result
  typed values/artifact refs
  no FigureDocument/DataFigure/Qt/Matplotlib object
```

`connect(config="virtual" | config_path | InstallationConfigDocument, repository=...)`只通过冻结 installation package contract 建立一次完整 authority。remote pulse必须由保存的installation config或 `InstallationConfigDocument.from_parameters("remote_pulse", {...})`组成，并只发布真实 sequencer能力。Camera monitor与finite acquisition不是两个Measurement：`repeat=0`和 `repeat=K`选择同一request/definition的不同 execution form；`frames_per_cycle`的具名输出与schema仍由Camera Measurement owner声明。pulse-only remote连通不能外推为qCMOS qualification。

Pulse Workbench统一 Edit/Preview/Scan/New/Open/Save、Run Once、HOLD、AUTONOMOUS scan与Stop；standalone窗口拥有它创建的Experiment，`exp.pulse_gui()`借用调用者已有Experiment。连接、preview、load/save、start/reap都在窄application worker；Qt只消费 `PulseFacade + PulseTargetDescriptor + RunHandle`，从不到达raw client或FIRE/SAFE verb。

正式GUI的 `Scan repeats`由 `PulseDocument.scan_sweep_count`单一拥有并随同一current document保存/恢复；`0`表示下一次由GUI发起continuous，`K>0`表示有限完整sweep数。窗口、QSettings与sidecar不得另存副本。每次执行仍由 `PulseRunRequest`显式冻结execution form与正整数sweep count；compiler、transport和progress poll都不得把它解释为时序命令。

Notebook Pulse窗口是Experiment-owned singleton：第一次 `exp.pulse_gui()`登记窗口；按X只执行既有hide，不弹未保存确认、不Stop、不关闭controller；下一次无参数调用恢复同一窗口、`PulseEditorSession`、路径、scan code与未保存编辑。窗口已存在时再次传 `document/path`必须拒绝。`Experiment.close()`把第一次调用者确立为该次teardown的唯一owner，并冻结全部Workbench handle；并发调用者只等待同一个close-attempt完成事实，不得重复shutdown、清repository或提前发布`CLOSED`。owner先使handle失效并由runtime完成active Run interrupt/SAFE；每个handle在Qt owner排空worker、释放retention并永久销毁后设置application可等待的close ack。调用者在foreign thread只等待该ack；调用者本身是Qt owner时通过冻结handle的窄等待端口只泵既有QApplication事件直到共享attempt结束，不能裸等application Event而饿死正被foreign owner等待的Qt ack，也不能创建第二event loop/应用或从外线程碰QWidget。任一request/ack或runtime关闭失败时，状态保持`CLOSING`，冻结的handle集合与全部data/repository owner保持存活，后续显式`close()`从同一集合重试；等待同一次失败attempt的调用者得到同一失败结果。只有全部handle ack到齐后才允许关闭repository并原子发布`CLOSED`，不能让异步窗口清理泄漏到下一次cell/test，也不能让未退役worker访问已经关闭的数据资源。standalone窗口继续执行dirty confirmation、cancel/reap与owned Experiment SAFE close，但成功后也必须进入同一个`permanently_closed`提交：body断开wrapper、从retention移除并`deleteLater`，不存在“controller已关、窗口只隐藏、Python wrapper留到解释器退出”的第二终态。offscreen正式流程只请求一次close，等待该事实并消费DeferredDelete；不得轮询重复调用已经提交的close命令。

Figure/Fit产品面只有一套：

- `ScanArtifact`与 `CaptureArtifact`均可作为FitResultRepository、headless `.fit()`、DataFigure、`figure_gui/fit_gui`与saved-grid的exact source。
- TaskConsole每张可Fit panel在既有Setting/Edit中使用frontend `FitAuthoringPane`，从当前已呈现的exact `OwnedSnapshot`冻结FitSpec，调用唯一 `bind_fit -> BoundFit.run`，结果只作为该panel的transient overlay并发布 `fit.<parameter>`。它不打开第二窗口、不创建本地archive、不按artifact kind特判。
- 所有非fit信息轴保留为batch轴；canonical scalar carrier由binder按 `SCALAR` role取唯一index 0。无repeat只合成明确的size-1 repeat dataset轴，repeat×site/grid保留具名轴、稀疏PointLayout与失败cell validity。Grid overview/focus共享完整batch result，focus raster不能冒充single-panel authority。
- `Experiment.figure_gui()`无参数时打开session-independent FigureViewer；`.npz`直接打开，`.png/.jpg/.jpeg`只解析同stem的 `.npz`，绝不从像素或rank反猜数据。typed source仍走同一DataFigure/Grid/Fit分派。
- FigureViewer的File/Info外壳嵌入唯一 `DataFigureWindow`；候选文件在worker完整decode/validate，成功后才替换pane，失败保留上一幅有效Figure。Save原子写current archive并从 `LoadedFigureArchive`重开，不伪造neutral artifact。
- `DISPLAY_ONLY`只表示ViewSpec/viewport/display reducer/export raster不能升级为FitSpec、ScanOutputContract、CommittedTransform或修改source artifact；不表示typed交互图可缺少适用的zoom/pan/selector/refit/export。静态多页report是无数据坐标交互的明确例外。

Calibration report不拥有第二套presentation。repository一次decode并互证 `CalibrationArtifact + CalibrationReport`；neutral calibration owner只提供已保存的site、threshold、fidelity、validity、empirical PSF事实，capability UI leaf选择页面顺序、领域label与overlay intent，随后委托frontend唯一 `PlotReportDocument -> PlotPanelContract/PlotPanelSession -> EncodedRasterDocument`链。Workbench lane只托管worker取消、窗口作业生命周期与文件I/O；Workbench Qt host原子安装frontend返回的multi-page document。Workbench不实例化Matplotlib composer、不重跑拟合/阈值、不把site向量reshape成伪二维Dataset，也不决定style、layout或codec。普通runtime calibration load不导入SciPy、Matplotlib或Qt；只有明确的analysis/report调用加载相应extra。

Occupancy的Figure一次只选择artifact中的一个真实输出块：`occupancy_output=None|"occupied"`默认分类结果，`"counts"`显式选择计数；非occupancy source携带该参数立即拒绝。document、ResolvedDatasetMap与OwnedSnapshot都指向所选块的exact schema/revision/generation，不创建第三个DataBlock、不堆伪COMPONENT轴、不退回source capture。SITE不自动reduce；artifact DataBlock的逐SITE缺失使用 `DatasetComponentValidity`。bool histogram固定 `false/true`两类。

`occupancy_cell_gui(ref, selection=...)`只接受repeat/point具名轴上的exact `IndexSelection`；只有size=1轴可自动取0，非单例缺失、range、未知轴或PointLayout hole都在读数组前拒绝。neutral Occupancy owner用同一 `DatasetCellAddress`读取occupied cell与source frame，并在publication前比较repeat/point axes、PointLayout、revision、generation及artifact/calibration lineage；frontend只消费已经same-shot闭合的frame/SiteMap typed value。导航沿schema-owned `cell_layout`的repeat-major storage order，跳过sparse hole且不wrap；每根非singleton轴保持显式未选择，不暗选first/latest。

DataFigure窗口中未保存execution与 `FitResultBatch`由headless `FitDraftAuthority`独占；Qt只得到不可保存的 `FitDraftResult`与immutable front。只有显式Save才持久化：Capture/Scan发布 `FitResultArtifactRef`并从exact ref重开，本地Figure写archive并从exact `LoadedFigureArchive`重开。TaskConsole的fit job没有Save capability。source/revision改变时撤销stale overlay/output；Clear只撤当前spec、overlay与 `fit.*`。

`camera_monitor_request()`只冻结camera role与数据/视图语义；adapter I/O deadline属于安装时验证的Port capability，不是Measurement参数。`inspect_camera_monitor()`只返回free-running capability、working point与exact output schema，不arm设备。`camera_monitor_gui()`只把one-shot `PreparedCameraMonitor` factory交给Workbench，不交Experiment、RunPlan、Port、resolver或raw camera。每次Start取得新的Run、BlockId与stream generation；Stop/Close只cancel该RunHandle并异步reap，真实terminal后才撤下front。free-running monitor不获得formal association、artifact或authoritative Fit/Save语义。

`Experiment`只做参数便利、typed request构造、结果解包和composition delegation；它不调用raw adapter，不复制calibration/fit/scan算法，也不保存 `current_calibration`或“最近一次”映射。依赖calibration的request必须显式接收 `CalibrationArtifactRef`，构造时冻结 `ReadoutBindingKey/ref/model`，formal preflight再admit并验证source与物理适用性；多camera不猜ref，运行时不回查facade。

`Experiment`、TaskConsole、PulseGUI与launcher不得公开raw Camera/Sequencer、DeviceSet、SDK handle、BoundDevice/drive-capable Port、internal RunPlan或硬件verb。普通硬件动作都转为declarative request/窄command facade，经同一个 `InstallationRuntime -> RunController -> ResourceArbiter -> DeviceBroker -> adapter owner`执行。public `device_catalog`只观察；`DeviceBindingResolver`只在composition/bind调用栈内出现。

`figure_document`及所有SiteMap presentation由frontend owner生成。neutral先发布完成causal/same-shot closure的typed inputs；frontend建立view、Area/Cross派生、render与selector；application composition只注入loader并委托，不解释artifact字段、重做join或选择UI策略。`FigureDocument`不含repository/ref/resolver，evaluate/render另收同revision `ResolvedDatasetMap`。render extra缺失时headless采集、分析与 `figure_document`仍完整。

日常路径必须保持短且诚实：

```python
exp = zlc.connect("virtual", repository=repo)
capture_ref = exp.readout.capture("my_pulse.json")
fit = exp.fit(capture_ref, model="radial_gaussian_center")
fit_ref = fit.save()
saved = exp.load_fit(fit_ref)
```

显式校准同样不暴露Port/RunPlan：用户构造带具名event layout和空间意图的 `CalibrationAnalysisRequest`，由 `exp.nodes.calibration` 的leaf-owned API冻结readout binding并执行；短 `sitemap(...) -> calibration ref -> detect(...)`依次提交raw Capture与Calibration。第一步成功、第二步失败或被 `KeyboardInterrupt`中断时，typed异常保留 `source_capture_ref`；不回滚第一个artifact，也不建立隐式current/latest calibration。

### 4.6 顶层运行模型：四个平面、三个边界

最终架构不是一个所有节点都传同一种“大数据对象”的通用 DAG，而是四个语义平面：

```text
外部世界
  -> Measurement
  -> [sample/event plane: Envelope<Value | typed domain record>]
  -> capability-owned derived SignalEventSource（可选、逐event）
  -> DatasetBuilder（finite exact）| MonitorDataset（live preview）
  -> [dataset plane: immutable DataBlock revision / typed ArtifactRef]
  -> capability-owned Processor snapshot evaluate（可选、typed coverage/event identity）
  -> [dataset plane: typed atomic derived live outputs]
  -> Analysis（zlc_data 通用分析或 neutral 领域分析）
  -> [result plane: typed result / immutable artifact]

任一冻结 dataset/result
  -> frontend ViewSpec/FigureDocument/DataFigure
  -> [presentation plane: 可丢弃、可重算、不可反向成为权威输入]
```

三个边界各自只有一个 owner：

1. `Measurement -> event` 由 acquisition runtime 赋 envelope、key、generation 和 provenance；设备 adapter 不发布 Hub event。
2. `event -> dataset` 只由一个预先绑定的 materializer 完成：finite exact 由 `DatasetBuilder` 按冻结 repeat/point schedule 写入，live 由 `MonitorDataset` 按物理 cycle 或 event sequence 写入；二者都保持 `(R,P,*data_shape)`。capability-owned live Processor若消费dataset，必须显式取得一个immutable `OwnedSnapshot + coverage/event identity`并原子产生新typed outputs；它不能读取mutable builder、从revision通知重建数据或冒充event materializer。
3. `dataset/result -> presentation` 只由 frontend 完成；ViewSpec 是可重算显示意图，zlc_data FitSpec/CommittedTransform 与 neutral 领域 AnalysisSpec 是独立权威意图，二者没有可隐式升级的继承关系。

这四个平面不是四套框架：它们共享 zlc_data 的 Value/DataBlock/axis/validity 值对象和统一 lineage，但不共享生命周期、背压或失败语义。sample plane 处理实时性与 exact reservation；dataset plane 处理完整性与 revision；result plane 处理算法合同与 artifact commit；presentation plane 只优化交互延迟。这样既避免“所有东西都是 Processor”，也避免为了统一表面形式引入递归工作流引擎。

### 4.7 顶层边界的对抗结论

| 候选 | 否决/采纳原因 |
|---|---|
| fit/DataBlock/Selection 全放 frontend | 否决。neutral headless 必须依赖一个 presentation context，且 UI policy 容易泄漏进权威分析；当前 frontend 反向导入问题也会换方向重现。 |
| 通用 fit/Selection 全放 neutral_atom | 否决。DataFigure/notebook 为复用通用算法必须依赖中性原子领域；calibration/readout 与普通数学 fit 再次混进同一个 `core`。 |
| 为 fit、axis、selection 各拆独立包 | 否决。三者共同维护同一个 DataBlock/validity/transform 不变量，过细拆分会制造 codec 和 import ceremony。 |
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
| 设备调用/virtual-real实现边界 | consumer-owned Protocol；实现选择只在新进程composition时发生 |
| Task/Measurement/Processor/领域 Analysis 元数据 | frozen Definition；通用数据 Analysis 使用 zlc_data-owned descriptor/spec |
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

`ValueSchema` 描述一次 Measurement或capability-owned derived signal event 携带的值，例如一帧 `(H,W)` image、一个 `(site,)` occupancy vector 或一个使用 canonical trailing carrier `(1,)` 的真正标量。`data_axes` 永不为空；scalar 必须使用 zlc_data 唯一的 `SCALAR_AXIS`，普通 singleton data axis 不能冒充它。ValueSchema 没有 R/P leading axes。`DatasetSchema` 描述 materializer 把事件放入哪些 repeat/point cell 后形成的完整数据集。DataBlock 永远符合 DatasetSchema，AcquisitionStream/MonitorTap 的普通 event 永远不携带累计 DataBlock。

一个 domain event 可以是 frozen typed record，例如 `CameraSample(image: Value, frame_metadata)` 或 `OccupancySample(occupied: Value, counts: Value, source_metadata)`；它仍作为一个 Envelope payload 原子发布。record 中每个数值字段使用 zlc_data 的 Value/ValueSchema，record 类型和领域 metadata 由 producer package 拥有。

`AxisSpec` 包含：

- stable AxisId；
- name、role；
- size、coordinates；隐式坐标另带 `index_origin`，连续裁剪只移动 origin，不物化整根坐标轴；
- unit: canonical unit id 或 `None`、coordinate_frame: CoordinateFrameId。

AxisId 由 producer Definition 的稳定字段语义派生，在相同 semantic axis 的不同 run/adapter 间保持一致；不能每次构造随机 UUID，也不能用可修改 display name 或 tuple position 充当 identity。baseline 的 Selection 保留被保留轴的 AxisId，Reduction 只移除被约简轴，不创造匿名 replacement axis。唯一例外不是新的信息轴：当变换消费最后一根信息 data axis 时，zlc_data 统一补回同一个 canonical `SCALAR_AXIS` 物理 carrier，使输出保持 trailing scalar dimension。没有真实消费者的 Transpose/Stack/Create/Rename 与逐 operation 历史对象不预建；出现第二个必须创建派生信息轴的生产用例时，再由该 operation owner 定义稳定 AxisId 与 lineage。

单位采用 canonical string（例如 `Hz`、`MHz`、`s`、`count`），display label 与物理单位分开。baseline 不建立量纲代数、单位表达式 AST、UnitConversionTable 或自动换算；只有完全相同的 canonical string 才兼容。未知单位作为 opaque string round-trip。CoordinateFrameId 同样是 stable opaque id，只做等值检查；不同 frame 在 baseline 直接拒绝，不能因 shape、名字或数值范围相似而默认兼容。

需要相机畸变标定、单位换算或其它带物理模型的映射时，先由带 CalibrationArtifactRef 的 neutral Processor capability或Analysis显式产生新值与新 schema；不能把领域 calibration 藏进通用坐标 metadata。只有真实用例证明多个领域需要同一纯转换合同后，才从这些用例中提取 serializable UnitConversion/CoordinateTransform，baseline 不预建。

`role: AxisRoleId` 是 producer 声明的 stable、可序列化语义，例如 repeat、scan-point、monitor-history、spatial-x、spatial-y、spectral、site 或 component。built-in role 由 zlc_data 单源定义；领域扩展使用 namespaced id，不注册可变全局对象。不认识的 role 仍能 round-trip，但默认 preserve/select。role 不能从 rank、长度、singleton 或数值内容反推。`MONITOR_HISTORY` 只表示 live snapshot 内 newest-first 的可见 slot，不是物理 scan point、readout setting 或可直接进入权威 Fit 的自变量。

ValidityContract 是 ValueSchema 的一部分并进入 fingerprint：VALUE 表示整个 event value 同生同灭；COMPONENTS(axis_ids) 声明 mask 可细化到哪些具名 data axes。producer 不能首帧发 VALUE validity、遇到坏 site 后再未经 generation replacement 改成未声明的 component mask。具体 Processor capability 在构造自己的 prepared application 时根据 typed input validity contract 冻结输出 contract 与传播规则；无法证明时prepare直接拒绝，evaluate不能按首个payload改写合同。

Data schema 不枚举“当前软件允许哪些 projection/reducer”。数据身份与已安装分析功能必须解耦：

- Selection 与 Reduction 由 zlc_data 的显式 DataTransformSpec 定义；其它结构变换在出现真实消费者前不进入 baseline；
- display-only mean/latest/sample policy 由 frontend.figure 的 ViewContract 定义；
- 权威通用 mean/sum 使用 zlc_data ReductionSpec，用户/AnalysisSpec 必须显式选择并记录 unit/validity rule；
- ROI photon count、occupancy、calibration 等领域 reduction 是 neutral_atom具体Processor capability或Analysis，不伪装成 ValueSchema/DatasetSchema 的内建能力。

因此新增一个 renderer 或 reducer 不改变 ValueSchema/DatasetSchema fingerprint，也不触发无意义的 stream generation replacement。

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

`DatasetSchema.cell_layout` 是 repeat + PointLayout 合成后的唯一 canonical physical-row mapping owner。它在 schema 构造时只合成一次并缓存；transform、fit 与 frontend evaluator 都直接复用该对象，不能各自重写 C/F/EXPLICIT 组合规则。对于稀疏 PointLayout，PRODUCT factor 必须保留原 immutable PointLayout 实例及其已经建立的反向索引，不能为了消除 subclass equality 差异复制一份 mapping/dict；AxisLayout 的值相等与 hash 按结构而非 specialization class 判断，使 owner tree codec 把 factor 还原为通用 AxisLayout 后仍保持同一 canonical identity。live render 不得每帧重建 layout 或反向索引。

### 6.3 DataBlock 与 validity

```text
Value:
  values: ndarray         # (*data_shape)，标量为 shape (1,)
  validity: Valid | Invalid | ComponentValidity
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
DatasetValidity =
    Valid | Invalid                     # 整个 Dataset 同生同灭
  | CellValidity(mask: bool array)       # shape (R, P)，整 cell 同生同灭
  | DatasetComponentValidity(
        axis_ids,
        mask: bool array,                # (R,P, *declared component axes)
        broadcast_contract
    )
```

Value 是 stream event 内的 zlc_data 数值值对象；Envelope 的 key/provenance、CameraSample 等领域 record 不属于 Value。`ComponentValidity(axis_ids, mask)`只描述单个Value，mask rank恰好等于具名component axis数；`DatasetComponentValidity(axis_ids, mask)`只描述完整DataBlock，前两维明确且只能是物理`(R,P)`。两者是不同类型，不能把Value mask误当Dataset mask，也不能让调用者凭rank猜carrier。CellValidity表示Dataset中一个完整trailing value是否已经采集；DatasetComponentValidity表示每个`(R,P)` cell内不同site、pixel、bin或其它component可以独立无效。两种component validity的`axis_ids`都必须是ValueSchema.data_axes的有序子集，mask只能按这些具名axis广播；禁止依靠ndarray尾部对齐猜语义。这样：

- uint16 image 不必转 float；
- 未采点不被误认为 0；
- partial scan 保持固定 shape；
- fidelity 的 `(group,site)`、dead site、坏 pixel/bin 不会被整 cell 的 valid 掩盖；
- fit/reduce/histogram/meter/image alpha 都消费同一 validity，而不是各自用 `isfinite` 猜。

默认使用紧凑Valid/Invalid或CellValidity；只有producer/processor确实产生component级缺失时，Value使用ComponentValidity、DataBlock使用DatasetComponentValidity。实现可用只读broadcast view、packed bitmap或按chunk存储，不能强迫所有完整image复制同尺寸boolean mask；但优化不能改变具名axis语义。

Value.validity 与 DataBlock.validity 必须符合 cell_schema.validity_contract；COMPONENTS 合同仍允许用整体Valid/Invalid，DataBlock也允许CellValidity表示“本revision所有component同生同灭”的紧凑特例；一旦提供component mask，Value只能接受ComponentValidity、DataBlock只能接受DatasetComponentValidity，其axis_ids只能是合同声明集合的子集。VALUE合同绝不接受component mask。Selection/Reduction必须同时派生新的validity_contract，不能只变values/schema axes而忘记mask语义。

ReductionSpec 必须声明 `validity_policy`（例如 `ALL_REQUIRED`、`ANY_VALID`、`MIN_COUNT(n)` 或所选 reducer 合同自己的规则）。reducer 只在 mask 为真的 component 上运算，并产生新的具名 validity；不能把 NaN 当通用 validity，也不能用 `nanmean` 在未声明策略时悄悄吞掉坏 site。FitProblem 逐 batch cell 过滤无效 observation，并记录有效样本数；不足模型最小点数时只使该 batch result 失败。Histogram 丢弃无效 sample 但记录 dropped count；Meter 在目标 component 无效时显示 invalid，不回退其它 component。

发布后的 DataBlock 是 immutable materialized dataset snapshot：不仅 consumer 不能写，**snapshot 的内容在其整个可见生命周期内也不得因 materializer 后续 ingest 而变化**。finite exact 的 `DatasetBuilder` 持有不外泄的 mutable preallocated/chunked storage，根据冻结 schedule 原子写 values+validity，并只返回轻量 `DatasetProgress(block_id, revision, dirty_cells, coverage)`；旧 ref 请求必须返回 `SnapshotExpired`，绝不能回 latest。live 的 `MonitorDataset` 也只保留 current mutable window，但 ingest 通知只用于 coalesce；controller 必须从同一把锁冻结的 `MonitorDatasetSnapshot(OwnedSnapshot, aligned EventRefs, head, coverage)`读取当前值与 current selection，禁止把旧 progress 的 dirty/head 与后来 snapshot 拼接。任何 owned snapshot 都按UI刷新节奏显式取得，不能每个 event 自动 fan-out 完整 DataBlock。

`DatasetBuilder.materialize(current_ref)` 只产生 **provisional** `DatasetPreviewSnapshot`；只有绑定的 exact reservation 全部 ack、冻结的 `sequence -> DatasetCellAddress` 计划逐项匹配、TraceBinding 一致、source-owner EndOfStream 与 reserved end 相同、coverage 完整时，它才能 mint `SealedDatasetArtifact`。`MonitorDataset` 在类型上没有 `seal()`，只产生 `MonitorDatasetSnapshot`；交互冻结必须另建带 coverage/EventRef 的 finite diagnostic input，不能把 live window 冒充 formal capture。软件 seal 不自动证明事件属于某次pulse；PulseScan还必须保存并验证producer-owned association evidence与exact `PulseTerminalAck`。发布快照必须是 owned copy、immutable sealed chunk 或 copy-on-write；禁止把 mutable ndarray view 仅设 `writeable=False` 后冒充 revision。

DatasetRevisionRef故意不携带Formal runtime provenance，避免zlc_data反向依赖neutral。PulseScan的neutral-owned ScanArtifact manifest把普通DatasetRevisionRef与association evidence、pulse terminal、ordered EventRefs和完整coverage绑定；Workbench/Repository不得只抽出裸DatasetRevisionRef后绕过该artifact合同。

Measurement与capability-owned derived signal source不创建 DataBlock，也不读“当前累计 block”来决定下一条事件输出；它们只发布单个typed event。snapshot Processor可以从一个精确`OwnedSnapshot + coverage/event identity`原子构造自己声明的derived DataBlocks，但不能读取mutable builder、拼接不同revision或修改source block。`DatasetBuilder` 与 `MonitorDataset` 是互斥的两种 sample -> dataset 边界；interactive/display Analysis 可显式消费 provisional snapshot，权威 Fit/Calibration/Repository 只消费完整sealed artifact或由具体producer association证明并由collector完整seal的输入。DatasetProgress/RevisionRef 是状态通知，不是数据输入，consumer 不能从它重建权威值。

两种 materializer 共用一个与 generation-owned PayloadContract、同一 ValueSchema owner 对齐的 frozen `DatasetEventAdapter[T]`，但不复制/反射重建整张 adapter graph。adapter 从一次 frozen payload 投影 `Value`，metadata 由其 `DatasetMetadataContract(snapshot/fingerprint/digest)` owner 冻结；runtime 只拒绝真实可变 metadata alias，不把同进程 `object.__setattr__` 当安全边界。exact 路径在 Delivery/ack 事务中保存 ordered metadata digest；live 路径只保存显示所需 metadata 与 EventRef，不在热路径计算无人消费的第二份 digest。`CameraSample(image, metadata)` 因此不需要 side-channel metadata stream；adapter 不能改变 key、sequence、TraceBinding 或 exact cell schedule。

所有 AxisId 在一个 DatasetSchema 内唯一；coordinates 长度与 size 相等；`repeat_axis.size == R`；`values.shape[1] == point_layout.storage_size`；cell_schema.data_axes 顺序与 trailing ndarray 顺序完全一致。任何 public consumer 若要从 P 恢复多维 point index，必须调用 PointLayout，不能自行 `reshape` 猜 order。

### 6.4 标量

标量唯一表示：

```text
ValueSchema.data_axes == (SCALAR_AXIS,)
SCALAR_AXIS.role == SCALAR
SCALAR_AXIS.size == 1
data_shape == (1,)
Value.values.shape == (1,)
DataBlock.values.shape == (R, P, 1)
```

`SCALAR_AXIS` 只是统一的物理 carrier，不携带额外物理自由度。Fit、Figure 与 transform 只能按声明的 `SCALAR` role消费它，绝不能按 `size == 1`、rank 或 singleton 猜。任意其它 role/AxisId 的长度一 data axis 仍是信息轴，仍需显式 Select、Reduce 或作为 batch 保留。旧的空 `data_axes`/shape `()` 直接拒绝，不保留兼容 reader。

GUI 的 canonical signal dimension 必须忠实显示真实物理 tensor：`R × P × (*data_shape)`，其中 `R` 与 `P=point_layout.storage_size` 各自始终只有一个维度。Logic 行与 signal picker 中该字段都是由当前权威 `DatasetSchema`/value 自动派生的只读结果：不得让用户编辑，不得由 catalog、definition 或 Workbench 手写，不得在维度串内追加 logical point axes/coordinates/PointLayout 等第二种说明；未发布时显示 `—`，已有 value 却缺少 `DatasetSchema` 必须报契约错误，不能退回裸 ndarray shape 猜测。logical point metadata 只供 Grid/facet/axis navigation 的具名控件消费，不能混入 signal dimension。因此 7×7×7 的三维标量扫描在 Logic/picker 中显示 `1 × 343 × (1)`；不能显示为 `1 × 7×7×7 × (1)`。二维/多维 data shape 必须逐轴完整显示。物理维度契约只属于`zlc_data.DatasetSchema`；声明前与已有值后的 Logic、picker、panel 信息统一调用纯展示层`zlc_frontend.shape_text.describe_dataset_shape`消费该schema，任何 Workbench 不得另写shape/rank特判或反向定义维度。

### 6.5 Materializer 的原子提交

materialized value 的全部相关状态由实际拥有 mutable storage 的 materializer 直接负责，不建立无人消费的通用 delta 值对象。`DatasetBuilder` 先完成 payload、validity、exact schedule/key、metadata 与 authority 验证，再在 stream 的 Delivery/ack 事务和 builder 自身锁内一次提交 values、written、validity、metadata、ordered event/metadata hash state、counters 与 revision；所有会按输入拒绝的操作必须在 commit point 前完成，commit point 后只消费已经准入的 typed owner 值并以同一 stream 临界区内的 no-fail ack 收尾。前置验证失败不写入、不推进 revision/ack；进程或基础设施故障只能使本次 run 失败，不能产生 sealed artifact。`MonitorDataset` 在自己的锁内一次提交 values、written、validity、metadata、EventRef、head/counters 与 revision，并从同一临界区冻结 head、coverage、EventRefs 与 owned snapshot，不能把不同 revision 的字段拼在一起。

sample stream、capability-owned derived signal source 与普通 UI queue 只传事件值或 typed record。持久化只接收 sealed artifact/immutable snapshot，不保存一套与 materializer revision 平行的 delta journal。Live binding 只接 coalesced revision 通知，并按刷新节奏请求 current slice/owned snapshot；不得把 full mutation record 或 full snapshot 逐 event fan-out 给所有 panel。只有真实 profile 与已排期 consumer 同时证明 immutable snapshot 成本不可接受时，才从那个 storage owner 内部提取最小增量表示；在此之前不预建 history、revision replay 或跨 owner apply 协议。

### 6.6 Axis transform

baseline 只允许已经被 fit、scan 与 frontend authority draft 消费的两类 operation：

```text
Selection(index / contiguous index range / coordinate range)
ReductionSpec(axis_ids, MEAN | SUM | MIN | MAX, missing_policy, validity_policy)
```

`DataTransformSpec.operations` 直接保存 `Selection | ReductionSpec`，不再为二者各包一层单字段 Select/Reduce。`commit_transform(schema, spec)` 只冻结 `{input_schema_fingerprint, spec, output_schema_fingerprint}`；UI revision/origin 属于 Workbench draft，durable digest 属于外层 artifact/CAS，不能塞回 data authority 形成自证 metadata。`apply_transform(OwnedSnapshot, CommittedTransform)` 返回带 source ref、完整 committed transform、values、validity 与派生 schema 的 TransformedData；不建立无人消费的 preview type 或逐 operation record。

连续隐式坐标 range 用 `index_origin + local_index` 表示，保留原 unit/frame/name/role；不得按逻辑 axis 长度建立 tuple/remap。显式稀疏 PointLayout 只遍历实际 physical rows。任何 operation 都不得匿名 `flatten()`，也不能把 `(repeat, point, *data_axes)` 偷换成三个无语义长度；DataBlock 的物理 P 维始终由 PointLayout 映射回完整 point_axes。

## 7. Stream、buffer 与一致性

### 7.1 Envelope

```text
Envelope[T]:
  event_id
  stream_id
  stream_generation
  sequence
  emitted_at
  join_key: optional owner-snapshotted key
  join_key_contract_fingerprint: optional
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

数值/领域数据 Envelope 额外包含 payload contract fingerprint 与 captured timestamp。payload 的 snapshot/validate/fingerprint/digest 必须由一个 generation-owned `PayloadContract` 单源提供，不能让多个 lambda 分别冻结与验证并漂移；`ValuePayloadContract` 还要求所有 event 共享同一个 ValueSchema 对象，禁止每帧夹带重复 schema/coordinates。普通 stream payload 是 Value 或包含 Value 字段的 frozen domain record；DataBlock 只属于 DatasetBuilder/materialization 边界，不能作为“当前累计 signal”反复发布。Provenance 是 causation graph、payload fingerprint、CommittedTransform 与外层 artifact lineage 的派生视图，不是另一套含义模糊字段。

JoinKey 是 frozen、可序列化的领域值（例如 TriggerKey/ScanCellKey/ShotKey），不是字符串拼接或 payload 私有字段。generation-owned `JoinKeyContract.snapshot(key)` 是唯一验证 owner：它同时验证并返回 owned frozen key，stream 不在下一行重复 validate；fingerprint 绑定其语义。exact DatasetBuilder 另绑定由编译计划独立产生的完整 `sequence -> DatasetCellAddress` schedule，event key 必须逐项相等；仅有合法 key 类型并不足以证明 row 没有对调。keyed live cycle 同样验证物理 schedule；append history 则故意不把 producer join key 当 panel slot，slot 只由 consumer sequence 决定。TraceContext.correlation_id 只用于追踪，不能代替数据关联 key。

`stream_generation` 只能由 broker/factory mint 的不可复用 incarnation identity 产生，调用方不能用可复用字符串为两个 live source 指定同一 generation；否则不同内容可能得到相等 DatasetRevisionRef。`sequence` 在 `(stream_id, generation)` 内从 0 严格单调且不复用，event_id 由 generation+sequence 派生，不维护随 monitor 寿命无界增长的去重集合。capability-owned derived signal source为输出创建新 event_id，不能沿用某个输入 id 冒充同一事件；严格1:1传播另以direct input EventRef记录因果。少量 join 使用 EventRef；capability-owned grouped application或DatasetBuilder的长连续输入使用 EventSpanRef，ordered_digest 覆盖按 sequence 排列的 event_id/payload digest。禁止在每个累计结果里复制全部历史 event_id，避免 provenance 退化为 O(N²)。

### 7.2 四种通信原语

| 原语 | 语义 | 用途 |
|---|---|---|
| AcquisitionStream | ordered、exact、cursor、gap-fatal；未ack事件自然保留 | 正式 scan/capture |
| MonitorTap + MonitorDataset | ordered delivery；Dataset只保存请求声明的rolling window | live UI |
| ControlTopic[T] | typed、revisioned、ack | ROI、threshold、run command |
| EventStream[T] | progress/transition notification | UI/headless status |

Artifact Repository 是持久化原语，不是 stream。

ControlTopic 的 ack 明确区分 `ACCEPTED`、`APPLIED(at transaction boundary)`、`REJECTED(reason)`、`SUPERSEDED(by_revision)` 与 `TERMINATED(reason)`；发送成功不等于硬件已经应用。每个被 ACCEPTED 的 revision 最终必须恰好收到 APPLIED、SUPERSEDED、REJECTED 或 TERMINATED 之一，UI 不会永久等待一个被 coalesce 或 owner shutdown 吞掉的 command。有限正式 Run 拒绝 reconfigure 时必须返回 REJECTED，UI 不能先改成本地“已生效”状态。

monitor 的 ROI/threshold value 更新必须把 **source acquisition 与 downstream analysis 生命周期分开**。相机 source Run、raw stream/tap 和可见 raw front 持续运行；修改一个已经存在的 ROI/threshold processor 只向该 downstream owner 的 `ControlTopic` 发布新 revision，并在该 processor 的事务边界得到 `APPLIED` 后切换语义，不能重启 source、重建 raw history 或制造 running consumer gap。新建/删除 ROI processor 只创建/终止该 downstream stream/generation；source tap topology 仍不变。schema-affecting 的 downstream 变更可以替换该 downstream generation，但不得借此替换 source generation。只有 source 本身的硬件/采集 schema 确实改变时，才按下文 source generation replacement 执行；不能把 whole-Run replacement 当成 monitor retarget 的实现捷径。

EventStream 中 progress 可以 coalesce，transition 必须按 run revision 有序；但通知流不是状态真相源。RunHandle 保存可查询的最新 authoritative state/error/phase snapshot，UI 初次连接、漏事件或重连后先读取 snapshot，再订阅后续事件，因此 terminal event 即使 UI 当时阻塞也不会“丢掉终态”。

### 7.3 Exact reservation

Reservation 是对以下区间的唯一 formal-consumer authority 与 ack watermark；它不是内存配额：

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

- fire前按expected event count、物理采集计划与exact reservation原子验证；
- `AcquisitionProducer` 是 source owner lane 独占的 write/terminal capability；普通 pipeline 只拿 read-side stream/reservation，不能提前 mint EOS；
- cursor、Delivery、EndOfStream 都是 owner-minted opaque capability，不能用公开构造器或“is_exact=True”伪造；
- cursor 不跨 generation；
- baseline 每个 stream generation 只允许一个 formal exact materializer，monitor fan-out 不延长formal未ack事件的所有权；出现第二个真实 required-exact consumer 前不引入多 reservation watermark 机器；
- reservation 绑定 `TraceBinding(run_id, source_id)`；同一区间混入另一 run/source 的 event 在写 DatasetBuilder 前失败；
- DatasetBuilder 构造时独占 claim reservation completion/abort authority；`commit cell + ack` 在同一个 stream authority 临界区完成，失败不能留下可 seal 的半 revision；raw reservation 不再提供与 builder owner 冲突的 context-manager cleanup，统一由 DatasetBuilder/session teardown；
- frozen join schedule、Envelope key、sequence 与 destination cell 四者逐项一致后才写入；
- exact path 不自动 retry hardware run；
- formal cursor逐event有序消费；未ack事件由producer自然持有，内存不足按普通分配失败结束run，不能提前预算、覆盖或伪装成专用容量异常。非formal显示端是否latest-only由host delivery policy单独决定，并把skipped revisions记录为presentation telemetry，不能修改source coverage。

正式exact run必须是有限Run并有确定的`end_sequence/total_expected`，该数字只证明EOS/coverage完整性，不生成任何软件内存配额。producer把每个owned immutable event追加到当前generation；formal consumer ack后即可释放对应前缀，consumer慢时未ack集合自然增长。系统不根据预测速率、frame bytes或“安全预算”预先拒绝合法run，也不以软件ring覆盖未ack数据；真实分配失败明确使run FAILED。连续Measurement使用`MonitorTap -> MonitorDataset`，其中`history_cycles`是可见Dataset的声明轴长度，而非全局内存上限；host的latest-only展示可coalesce尚未开始的工作，但只能记录独立presentation telemetry，不能修改source coverage或倒推改变source对已交付事件的所有权。

软件数据面不定义生产流控、积压毒化或容量异常状态机。物理采集完成后的decode/copy/schema/metadata/key/publish异常仍由CaptureSession统一转成`SourceFailed`并完成设备cleanup；ordinal/stamp gap、driver报告真实覆盖或实际帧数与冻结计划不符同样使当前run失败，但原因是观测到的物理不一致，不是软件预算被用完。

camera adapter 边界使用一种不可变的 `CameraFrameRecord`，而不是一条 ndarray queue 加另一条 metadata queue：

```text
CameraFrameRecord:                       # devices.camera.contract owner，不是 artifact schema
  image: owned, C-contiguous, read-only ndarray
  source_ordinal: non-negative int       # 当前 arm epoch 内的软件交付序号
  produced_count: optional non-negative int
  frame_stamp/camera_stamp: optional int
  timestamp_seconds: optional non-negative int
  timestamp_microseconds: optional int in [0, 1_000_000)
  host_received_at_ns: positive int
  driver_buffer_index: optional non-negative int
```

record 构造时就必须取得图像 bytes 的 ownership 并冻结 metadata；driver 之后复用 ring slot 不得改变已发布 record。`source_ordinal` 在每次 arm epoch 从0连续增加，duplicate/gap/超出expected count 立即失败；它是 host adapter 的排空顺序，不是 FPGA emitted-edge receipt。`produced_count` 是读取该帧时观察到的 source 累计快照，batch drain 时可在多个 record 中相同；禁止把它伪造成逐帧 +1 counter。qCMOS adapter 必须从同一次 `buf_getframedata` 保留 `framestamp/camerastamp/timestamp`，并把同一 drain 观察点的 `cap_transferinfo().nFrameCount` 写入 `produced_count`。

`devices.camera.contract.CameraWorkingPoint`只冻结adapter从硬件/模拟器读取的物理工作点与由adapter唯一计算的settings fingerprint；它不携带、也无权自铸exact qualification。`CameraCaptureEndpoint`只把这些primitive facts一次转换成neutral-owned `ValueSchema/CameraPhysicalFacts`，不认识Virtual/DCAM concrete type；可信installation composition另行注入qualification。当前virtual composition注入与原实现相同的deterministic in-process trigger-wire digest；real composition只能注入由active CameraExternalTriggerQualification authority解析并pin的资格，raw adapter返回任意字符串不能放行FIRE。该结构化SPI只解除concrete-type反向依赖，不证明hardware identity、thread affinity或external-trigger资格。

camera arm只接受物理采集所需的`buffer_frame_count`：finite capture等于冻结的完整`expected_frames`；continuous monitor等于其声明的`history_cycles * frames_per_cycle`。这是DCAM `dcambuf_alloc()`/driver ring的本次采集几何，不是可调内存预算；endpoint不得再施加第二个上限或取较小值。adapter drain得到owned record后source stream自然追加；driver在host取得ownership前发生真实覆盖、stamp/ordinal不连续或finite count越界时返回`CameraBufferOverrun`并使当前run`SourceFailed`。monitor Dataset只在owner已接纳record后按声明的rolling window替换可见slot；这不会删除source的formal未ack事件，也不等于MonitorTap隐式丢数据。

endpoint terminal边界是一个窄的两阶段合同：terminal worker可先调用`finish_record_capture()`解除并发blocked read、stop/drain/join并冻结terminal record，随后arm-owner再次调用以完成owner-affine teardown；两个调用必须线程安全、幂等且返回同一record。adapter只交付`CameraFrameRecord`，CaptureSession在owner lane一次转换为neutral-owned`CameraSample(Value, metadata)`；不存在array-only平行reader。

stream_generation/payload contract fingerprint 改变时，旧 exact cursor 终止为 typed SchemaChanged。schema-affecting reconfigure 不是“原地改参数”，而是 generation replacement：owner 在 transaction boundary 终止旧 generation、对所有 pending Control revision 发 terminal ack，为每个绑定的 exact DatasetBuilder 或 live MonitorDataset 创建新 block_id/DatasetSchema/generation，再在新 generation 首次 publication 前完成 tap/materializer rebind。旧 pending view/fit 结果 stale，CommittedTransform 因 DatasetSchema fingerprint 改变一律失效，不能按 index 偷换。稳定 AxisId 只帮助为 workspace preference 生成候选匹配，仍须完整 schema/coordinate/validity 校验。正式 finite Run 默认拒绝 schema-affecting reconfigure；value-only 且 schema 不变的参数才可按运行合同在边界 APPLIED。

capability-owned derived signal source对每个输入只原子发布一个 typed payload；同 shot 多字段装进同一 frozen record，成功 enqueue 后才 ack 输入。不得把一次classification拆成多个 exact stream再伪造跨stream transaction；确有不同cardinality/key/lifecycle的结果应由capability拆成独立产品节点。DatasetBuilder 在 Value 已按 frozen schedule 原子写入 values+validity 后 ack；storage 与权威领域算法只接受 SealedDatasetArtifact，或其领域合同明确接受且保存了完整association/terminal/coverage的typed artifact，不能退回接受裸 DataBlock/OwnedSnapshot/DatasetPreviewSnapshot。Repository sink 的 ack 点是临时 blob fsync/校验完成且 manifest 原子提交之后，不是刚开始写文件。

payload 内容 digest 的唯一运行时 owner 是 `AcquisitionStream` 发布点：payload contract 先产生 owned immutable snapshot，stream 对该 snapshot 计算一次语义 digest 并铸造唯一 `EventRef`；Envelope 只保存这一份 EventRef。exact consumer、worker ack、DatasetBuilder 和 artifact/finalizer 只验证各自的 authority/sequence/cell/terminal 事实并流式累计小型 EventRef，不重新扫描 payload 或从 DataBlock 重建逐帧 digest。持久化时 ContentStore 对最终 blob bytes 另做一次 CAS hash，读取时按同一 ref 校验；CAS hash 与 EventRef digest 生命周期不同，二者都保留但不得互相重复冒充。同进程 `object.__setattr__`/手工伪造新 manifest 不是安全边界，不得以此为理由把每帧 SHA 放回 consume/load 热路径。

### 7.4 JoinPolicy

当前实际需要四种：

```text
EXACT_KEY
ZIP_SEQUENCE
LATEST_COMPLETE_KEY_MONITOR
INDEPENDENT_LATEST_MONITOR
```

`LATEST_COMPLETE_KEY_MONITOR` 允许 UI 跳过旧 shot，但一次显示的相关信号必须属于同一个 key；不完整旧 key 被淘汰并计入 missed/incomplete count。它不能用于 fit、scan、calibration 或 artifact。

`EXACT_KEY` 是跨设备/跨 worker 的正式关联方式。`ZIP_SEQUENCE` 只允许用于同一已验证软件 producer 拆出的等长 ordered streams，且合同能证明 sequence 一一对应；不能拿两个独立设备的“第 N 条”推断同一 shot。`INDEPENDENT_LATEST_MONITOR` 只用于互不声称相关的独立 panel，不能用于多输入计算或同一 coherent view。TaskConsole 的每张 panel 只持有一个 typed `signal` 绑定，Setting/Edit 不提供任意多信号表达式；需要多个 producer 的计算必须由具体capability拥有typed join/evaluation或成为formal Analysis，再由panel消费它唯一发布的结果。

暂不引入 WINDOW/ClockTransform；出现真实 use case 后再设计。

### 7.5 Exact 与 monitor fan-out

一个 physical CaptureSession 是 camera 的唯一 owner。它产生一次 AcquiredSample，broker 分发到：

- formal materializer：通过唯一 reservation 有序消费并逐项 ack；
- monitor tap：有序消费同一 immutable event，不覆盖或跳过尚未消费的record。

二者共享 event_id/trace 与同一个payload ownership事实。formal路径的未ack record自然保留；monitor tap读取后即可释放自己的引用。只有frontend明确声明的latest-only展示工作可以在开始渲染前用新revision替换旧revision，而且该显示跳过必须作为presentation telemetry报告，不能回写source、fit、scan、calibration或artifact。

broker 只分发 immutable payload/ref；driver 会复用 DMA/frame buffer 时，在第一次发布前复制或转移 ownership。formal未ack record、monitor current ref 和每个 consumer 各自持有明确 lifetime，任何consumer释放自己的引用都不能使其它consumer看到被复用的存储。

### 7.6 Buffer ownership 与采集 cardinality

Measurement output contract 只声明真实数据 cardinality：finite run 的完整`expected_frames`，或continuous monitor的`history_cycles`与`frames_per_cycle`。具体Processor prepared application声明自己的atomic output vocabulary；其capability-owned derived signal source若存在，还必须声明并证明输入/输出cardinality与ack点。DatasetBuilder/sink声明完整schedule与commit边界。运行时不得从生产速率、预计停顿、frame bytes或drain延迟派生另一个软件容量数字。

camera adapter的`buffer_frame_count`只有一条派生规则：finite capture等于完整`expected_frames`；continuous monitor等于`history_cycles * frames_per_cycle`。DCAM按该数值执行本次`dcambuf_alloc()`；virtual adapter遵守同一SPI但不另设固定队列上限。分配失败由SDK/Python原样使本次操作失败，不重试、降采样、丢轴、降低精度或缩短历史。driver在host取得ownership前发生的真实ring覆盖仍是`CameraBufferOverrun`。

正式采集的完整性来自source -> processor -> sink exact chain冻结的事件数量、顺序、ack与EOS，不来自把某个history或queue参数设大。MonitorTap与其唯一materializer必须在首次publication前绑定；execute中或panel打开时不得新增raw tap。可见rolling history只由下游MonitorDataset的声明shape决定，neutral `LiveDatasetPort`只拥有一个materializer lifetime并coalesce无payload通知，绝不能再持第二份window。`LiveDatasetHost`把该port接入host owner mailbox；同一source的多个panel读取`SignalDataPlane`中的同一atomic front，再用ViewSpec显示。

free-running Workbench使用这条固定拓扑：`FREE_RUNNING camera -> AcquisitionStream -> MonitorTap -> MonitorDataset -> LiveDatasetPort/Host -> SignalDataPlane`，首次publication后不能attach新window。Camera 的公开 `frame_i` 输出始终是 `(1,1,*frame_shape)`；`history_cycles`只属于显式 MonitorDataset/history view，不得泄漏为普通 Camera signal 的 P 轴，也不是host内存上限。

当前 finite exact-capture preview 只交付一个更窄的本地合同：`CapturePreviewSpec` 从 exact cell schema 派生 `(R=1, MONITOR_HISTORY=1, *data_shape)`；compiler attach 边界唯一核对它与本次 exact capture 共享同一个 cell-schema owner 和 event-adapter owner，不能把 capture A 的 projection 接到 capture B。运行中 preview bind/ingest/evaluate/raster 失败只撤下 preview，不能改变 exact result、CaptureArtifact 或 hardware cleanup；它不建立第二套资源策略。

### 7.7 Finite dataset 与 rolling monitor

三个 owner 明确分责，不允许一个双模式 builder 同时背 exact seal、stream delivery 与 GUI rolling：

```text
AcquisitionStream -> MonitorTap
  ordered event delivery；next不丢record；不改变formal未ack事件所有权

ExactReservation -> DatasetBuilder
  完整 sequence -> DatasetCellAddress 排列；ack/EOS/seal；只能 finite exact

MonitorTap -> MonitorDataset
  keyed_cycle: 物理 cycle offset 0 或 sequence gap 时先原子清空旧值/validity/metadata
  append_window: materializer 按 event sequence 分配 ring slot，snapshot 统一 newest-first
  两者都只产生 MonitorDatasetSnapshot，绝不 seal
```

`MonitorDatasetSnapshot` 在同一临界区冻结 DataBlock、cell-aligned EventRefs、head 与 `MonitorCoverage`；controller 的 current selection 只来自该 snapshot，不拼接可能过期的 progress。append history 的目标 shape 必须是 `(R=1,P=history_cycles,*data_shape)`，且 P 只能是一条 `MONITOR_HISTORY`、dense `RECT_C`、坐标严格为 `0..history_cycles-1` 的 newest-first slot 轴；slot 0 表示本次 snapshot 中最新的 retained event，遇到真实source sequence gap时slot n不等于物理上的“n shots ago”，真实sequence只能读取aligned EventRef。任意二维/多维 `data_shape` 原样保留，绝不能借用 SCAN_POINT/READOUT_EVENT、塞进匿名 `(repeat,data_points,data_dim)` 三项容器或 `reshape(...)[0]`。按声明history滚出旧slot是数据产品的正常更新，不是source丢帧；`MonitorCoverage`只报告真实source/sequence gap。formal `DatasetCoverage`只保存written/total；exact loss已由reservation sequence、cursor ack、完整schedule与owner EOS fail-closed证明。keyed cycle的complete只描述当前sweep，绝不混入上一sweep的仍valid cell。

交互 Fit/Save 若要使用 live 数据，必须把一个原子 snapshot 冻结为新的 owned finite diagnostic input并记录 event range、head、missed/coverage；不能把“当前 window”冒充从运行开始至今的完整 dataset。同一交互目标至多保留一个尚未开始的请求并以新revision替换旧请求；已完成的 diagnostic input 由其显式用户引用决定生命周期。

finite preview 的唯一顺序是 `capture_next -> exact DatasetBuilder.consume/ack -> MonitorDataset.ingest_latest -> no-payload change notice`；显示永远由 worker 随后直接 `materialize(None)` 冻结当时的原子 snapshot，通知中的 ref 不会被保存后再延迟解析。`CapturePreviewPort.bind()` 原子把 MonitorDataset lifetime 转给 Workbench slot，runtime 此后只留 non-owning ingest handle；失败由 slot 唯一关闭，进入 exact allocation 后即使在 bind 前发生 open/reservation/builder 失败也会终止该 slot。只有整个 direct cleanup 或 pulse+camera aggregate cleanup 没有 primary error、cleanup error 或 UNSAFE decision时才发布正常 source terminal并把最终单帧snapshot保留到 panel close；否则走同一个 `fail -> owner wake -> invalidate/clear`，不能让失败 Run 的最后front继续冒充有效。它不是第二份 finite truth、不会 seal，也不进入 artifact lineage。

产品状态也沿这条authority边界分开：只有`RunSnapshot.state == SUCCEEDED && final_committed`能把Capture标为`FINAL`；preview失败、board present失败或preview close重试都不能阻塞该snapshot的reconcile、result/reap或改变artifact。preview始终标`DISPLAY ONLY / latest rendered raw frame`，不能仅因Run已FINAL就声称当前raster必然是最后采集帧；FAILED/CANCELLED时旧front先退场，若presenter clear暂时失败则保留同一controller重试，而不是丢owner或把旧front重新标成PROVISIONAL。

continuous monitor与finite preview共享数据面但不共享终态语义。配置为`FREE_RUNNING`的独立`monitor_camera`一次arm后由传感器曝光时钟产生frame；其`buffer_frame_count`严格等于声明的`history_cycles * frames_per_cycle`，host循环持续排空adapter交付的record，不用sleep或GUI刷新节奏调度曝光。每个record先进入稳定StreamId的fresh generation，再由tap/materializer原子冻结`(R=1, MONITOR_HISTORY=1, SPATIAL_Y, SPATIAL_X)`、BlockId、head、aligned EventRefs与MonitorCoverage；IMAGE view用具名MONITOR_HISTORY AxisId显式选slot 0，不使用LatestNonempty、flatten或隐式data-axis reduce。该路径没有ExactReservation、seal、artifact、权威Fit/Save或formal capture含义。

driver真实ring覆盖、source ordinal/produced-count不连续或设备worker失败属于pre-broker source failure，必须终止Run并撤下front；record交给broker以后MonitorTap不再覆盖。frontend render coalescing只记录display-skipped revision，不能伪装成source gap或回流权威路径。正常Stop也不保留最后一帧：先cancel并完成session-specific stop/join/SAFE verification，再`source_terminal`撤销publish authority和front；并发真实source error不能被稍后的用户cancel洗成CANCELLED。virtual MOT role的物理合同固定为main oracle的1920×1200 Mono8、free-running sensor clock、读取同一current sequencer的compiled/held三轴DAC输出、SAFE零场与coil-space Gaussian fluorescence；与sequencer状态无关的移动Gaussian source不属于该产品。real Pylon的`LatestImageOnly`若会在record owner之前跳帧，则在adapter能显式报告skip并完成contract qualification前保持NO-GO，不能用virtual严格交付外推真机正确性。

## 8. 同步执行与线程托管

核心 runtime 不使用 asyncio。执行原则是：

> synchronous execution semantics, threaded hosting, cooperative cancellation。

### 8.1 线程拓扑

```text
GUI thread
  Qt QObject/Widget/QTimer
  immutable front present + selector/gesture overlay

RunController
  所有用户可启动 Run 的 lifecycle
  start() 立即返回 RunHandle
  run-owner 执行 bind 后的 preflight/cleanup，不直接跨 affinity 调 driver

Blocking I/O lane[ThreadAffinityKey]
  每个有线程亲和性的 device/session 串行

Capability-owned Processor application
  typed prepare/evaluate；可选的derived SignalEventSource也由同一capability拥有

Analysis executor
  bounded dataset/artifact analysis
  formal/offline 与 interactive QoS 分队列

Workbench application job host
  只托管 worker 生命周期、cancel、repository/file I/O 与 Qt owner wake
  不拥有 view、plot kind、layout、composer policy 或 codec

frontend presentation session（可在上述worker或notebook调用线程执行）
  PlotPanelSession / DataFigure / FitGrid / PlotReportDocument
  唯一拥有 view evaluation、FigureSpec/Divider、Agg composer 与 immutable front/bytes
  live panel 的尚未开始作业可按presentation revision coalesce
```

`Blocking I/O lane[ThreadAffinityKey]` 只有一个职责：在 adapter/composition owner 中串行化真实 SDK/session 的 thread-affine blocking call。它不是公平scheduler，不定义通用队列容量、pending/backlog 预算或预测内存准入。没有真实第二消费者和bounded blocking/interrupt证据时，不抽共享lane或通用调度框架。

不使用“每种职责固定一个全局 OS thread”。连续 camera monitor 不能阻塞无冲突设备；同一 thread-affine device 的调用必须串行。

独立 panel latest-only 是逻辑 mailbox（每 panel 最多一个 pending revision）；声明为同一 coherence group 的 panel 使用一个 board mailbox/evaluation revision，不各自挑 latest。这个“一份”是 replace-before-start 的交付语义：旧pending job在新revision接纳时已不再是待执行工作，不存在容量满、overflow或预算准入结果。它们复用 composition 中已经存在的少量 worker，不是每个 panel 新建线程；新revision只替换尚未开始的同一 view/board work。Analysis executor 区分 formal/offline 与 interactive QoS：interactive 同 panel 新 revision 可替换尚未开始的旧 fit；formal/offline/明确保存的 Analysis 不 coalesce，并服从 Run cancellation/deadline。capability-owned association signal source 的有序事件绝不进入可丢弃 view/interactive 队列；TaskConsole latest-only只调度相同capability prepared application的snapshot evaluate。

单panel live-image host不预建通用scheduler；它以host-local serial gate保证任一时刻只有一份snapshot/evaluation/raster工作。只有上一worker调用栈真正返回、释放相关引用后才提交dirty follow-up；多worker executor因而不会让同一panel两套大帧重叠。这不构成跨panel公平lane、queue policy或全局并发策略。高层presentation state与composer始终属于frontend session，host只决定何时调用它。

### 8.2 RunController 与 RunHandle

```text
RunController.run(plan)   -> Result      # 同步，notebook/test
RunController.start(plan) -> RunHandle   # 后台，workbench
```

这两个是composition内部入口，不是notebook public API。public `Experiment.run/start`只接收declarative Request；composition在同一generation snapshot内bind成internal RunPlan并立即提交给RunController，既不返回plan也不把它挂到RunHandle。RunHandle公开面只有run id、status/wait/cancel/recovery/result/ref等生命周期DTO，不含RunPlan、prepared value、领域bindings、RunDevice/CleanupDevice或drive-capable Port。

`RunController` 是所有用户可启动 Run 的唯一 lifecycle owner，包括 one-shot Task、finite/continuous Measurement（含PulseScan）和formal Analyses；DatasetBuilder只是对应Run拥有的materializer。每次 `start()` 创建一个 run-owner thread；terminal state 只能在所有run-owned I/O call、CaptureSession、producer/collector、materializer和required Analysis确认退出后产生。TaskConsole Processor row由其领域中立host管理row lifecycle并调用capability prepared application；它不是RunPlan、child Run或第二个RunController。

每种 Definition 只有一个与其语义一致的绑定结果：

```text
task owner builder(TaskDefinition key, request, immutable bindings) -> RunPlan[Result]
resolve MeasurementDefinition + domain binder -> domain Prepared/Spec（camera-backed时内含BoundCameraCapture）
processor capability prepare(typed request, admitted inputs) -> Prepared<Capability>Processor
domain analysis owner builder(typed request, immutable artifact refs) -> flat RunPlan[Result]
zlc_data.bind_fit(FitSpec, expected DatasetSchema) -> BoundFit
```

Measurement不是独立 lifecycle owner；每个Measurement capability把自己的typed request/bindings直接编译成一个顶层flat RunPlan。用户“单独 Start Camera Measurement”也使用camera owner的最小`BoundCameraCapture -> DatasetBuilder` compiler，而不是特殊启动路径或通用pipeline DSL。Processor capability返回自己关闭的prepared application，由已有TaskConsole host或capability-owned signal source消费，不编进通用formal processor pipeline。已提交 artifact 上的 calibration/detection 等领域 Analysis 由自己的 typed request 直接编译成一个 flat RunPlan；generic post-materialization AnalysisStep 不是baseline。这样不会为了统一方法签名而让不同语义冒充 Task，也不会为一个UI入口预建第二套生命周期。

MeasurementDefinition只含DefinitionKey、title与request/binding schema id等关闭字段；camera的动态output schema/cardinality只属于物理`BoundCameraCapture.capture_contract`，领域Definition不复制、不冒充该物理合同。ProcessorDefinition同样只含DefinitionKey、title与config schema id；input/output vocabulary、algorithm、artifact inputs、validity与lineage全部属于具体capability的typed request/prepared application，不提升到generic Definition或runtime fingerprint。三个Definition dataclass各自在构造器验证自己的精确字段类型与canonical值，且不存在开放metadata map，所以callback、raw driver、mutable cache没有可写入的字段；不为这些关闭值保留零生产消费者的递归声明式检查器或聚合Catalog class。generic runtime不调用Definition.bind，也不接收任意`request: object/bindings: object`；各领域composition在自己的typed request/typed bindings边界完成纯验证并直接构造Prepared/Bound值，bindings只含Bound Port和immutable config。camera capture compiler是无硬件I/O的确定性构造，可做schema、owner、完整schedule与expected counts校验；Processor prepare只admit它真实消费的typed input/artifact并冻结自己的application。真实分配只在driver prepare/arm时发生。Notebook可以在调用线程直接构造；Workbench把同步构造函数投递给普通application worker。只有RunPlan结果才交给RunController.start；Processor prepared application交给现有Processor row host。runtime不定义专用command/build lane、第二套队列协议或额外Service。

`RunPlan` 是扁平静态计划：

```text
RunPlan[Prepared, Executed, Final]:
  name
  resource_claims: tuple[ResourceClaim, ...]
  bound_devices: tuple[BoundDevice, ...]
  preflight(ctx) -> Prepared
  execute(ctx, prepared) -> Executed
  cleanup(ctx, prepared | None, primary_error | None) -> CleanupReport
  finalize(PostSafetyContext, executed) -> Final
  interrupt_operations: tuple[SafetyInterrupt, ...]
  timeout_seconds: optional finite monotonic duration
  requires_final_commit: bool
```

generic runtime 不保存领域 request/bindings 容器、execution mode 或 event/grouping 等领域字段。有限 exact 与 continuous monitor 的完整性、事件cardinality和rolling数据语义属于 Measurement/Pipeline/Dataset contract；领域 composition 先冻结 typed request/bindings，再用不可变输入构造上述 callbacks。Measurement/Calibration request、Definition、实验保存格式与TaskConsole/CalibrationWorkbench表单一律不含通用`input_timeout/io_timeout/timeout`：阻塞硬件调用的deadline由adapter/installation policy拥有并经Port capability冻结，runtime hosting/等待API若需要deadline也只属于调用生命周期，不能反向传播成物理实验参数。已提交capture上的calibration/detection若需要host execution deadline，只允许各自application模块的私有policy在`prepare_*_plan()`调用repository compiler时注入`RunPlan.timeout_seconds`；它不进入`CalibrationArtifactRequest`、`DetectionRequest`、`SitemapCalibrationRequest`、Notebook facade、可编辑request或FormSpec。所有内部 timeout 必须 finite、非负且只用 monotonic clock；artifact timestamp 才使用 wall clock。

`preflight` 的返回值就是领域私有的 typed prepared value，不再套公共运行包装对象。它可以携带 resolved schemas、reservations、cursors 和其它准备结果；`execute` 只能收到这一个值与 `RunContext`，不能从 session、global registry 或 service locator 找回未声明 Port。不包含 child run、递归 DAG 或运行中新增资源。

每个 `device/...` 的 EXCLUSIVE claim 必须在 `bound_devices` 中恰好出现一次；普通 CPU、repository 和纯只读资源不伪装成 device。`ResourceArbiter` 只持有当前进程的run owner与claim互斥，不从设备claim派生持久风险记录、不把旧run结果当成新连接的硬件状态。

baseline 的一个 `RunPlan` 只能使用同一个 `DeviceBroker`/installation authority；跨机器 endpoint 必须在它自己的 adapter/server 边界提供单一可验证 binding，而不是让一个 plan 拼接多套本地 arbiter。只有出现第二个必须共同驱动且无法归入同一 authority 的真实用例，才另行设计跨 authority 协调；当前只实现一个进程内ResourceLease，不预建分布式设备状态协调器。

stable identity 必须由当前live connection的adapter receipt与installation-owned AssetMap共同建立；普通实验config、role、Python class、device index、枚举顺序或用户填写的字符串都不能自证物理身份。AssetMap不是一个手写revision标签：它必须是machine/device级持久、canonical序列化的`asset_id -> canonical ResourceKey + exact adapter kind + expected live identity/endpoint matcher`，revision取其canonical内容digest。真实runtime缺少AssetMap、adapter kind不符或live readback不匹配时，composition直接NO-GO；普通`load_config`不能创建/覆盖ResourceKey、expected matcher或revision。同role换成另一serial即使重启了进程和broker也必须拒绝；只有离线maintenance明确更新AssetMap并保留旧安全事实后，才允许下一次新进程启动验证该映射。

identity evidence明确分为`HARDWARE_IDENTITY_READBACK`与`INSTALLATION_ASSERTED_ENDPOINT`：前者读取设备serial/DNA等不可混淆硬件标识；后者只在现有接口确实没有硬件标识时，用稳定deployment endpoint + AssetMap revision证明“当前连接到被安装声明占用的控制端点”，不得声称已经读回同一块物理板。`PhysicalDeviceIdentity(stable_device_identity, evidence_kind, evidence_digest, asset_map_revision)` 是跨连接稳定的完整身份；`DeviceBindingStamp(physical_identity, binding_instance_id)` 是一次 live binding 的精确身份，并拥有唯一 canonical tree codec。`VerifiedPhysicalDeviceIdentity` 只是 broker-minted、一次消费的握手结果，成功 bind 后即被 `BoundDevice.binding_stamp` 取代。adapter只返回绑定当前live connection的 identity readback；每次成功live handshake后，由DeviceBroker签发新的binding instance id，adapter不能选择、复用或自报。active Run首次检测到transport断开、device-removed或live-readback failure时，authority使旧binding失效并进入本次run的cleanup；禁止transparent reconnect后继续execute或cleanup。后续重连建立新runtime与binding instance，必须重做live identity与当前SAFE初始化。每次Run start都重新核对完整 physical identity、runtime instance 与 `DeviceBindingStamp`。

领域 composition 的 immutable bindings 只含 consumer-owned Port/factory、typed Repository 和 immutable config，不含 QWidget、open CaptureSession 或任意线程外可直接调用的 raw driver。Port 调用由 RunController 路由到 owner lane；preflight 返回值中的 session token/handle 也只能由该 lane 消费。

bind 必须从 request/bindings 计算完整或保守 superset ResourceClaims。preflight 可以拒绝 claim 与硬件 capability 不匹配，却不能发现后临时追加资源；若某 adapter 的条件资源无法在 bind 时确定，Definition 必须声明 superset 或拒绝该 request。

真实硬件使用两阶段启动，但仍是单层计划：

```text
bind -> RunPlan
-> acquire_all static claims
-> 在正确 I/O lane 使用InstallationDeviceGraph已冻结的verified physical connection，
   创建本run的session/capture handle并执行configure/query preflight；不得reconnect
-> resolve ValueSchema/DatasetSchema 与 expected event/sample cardinality
-> private prepared value(reservations, cursors, resolved contracts)
-> arm/sources ready
-> fire/execute
```

preflight 或 reservation 失败时不得 arm/fire，并释放已创建 reservation。CaptureSession 固定拥有它创建的硬件session及其disarm/close；installation connection 的 close 属于该次`InstallationRuntime` shutdown。两个owner不能对同一物理session重复SAFE。

device/session 的 create/open/configure/read/disarm/close 必须在其 ThreadAffinityKey 对应 lane 执行；composition root 只能在外部构造不接触 driver 的轻量 adapter/factory。真正raw SDK/driver对象只在 allowlisted InstallationDeviceGraph/DeviceBroker owner lane内部创建、保存和销毁；public `bind`/Definition/RunPlan/finalize不得接受或保留任意raw driver callback、bound method或可回调到driver的adapter object。CaptureSession 在 owner lane 创建并在同一 lane 销毁，不能在 run-owner thread 创建后交给 I/O lane 使用。

外部权威状态：

```text
RUNNING -> SUCCEEDED | FAILED
RUNNING -> CANCELLING -> CANCELLED | FAILED
```

waiting resource、arming、capturing、fitting、saving、finalizing、commit-reconciliation-blocked 是 phase，不是通用工作流状态。

由 `RunController.requires_final_commit` 管理的最终 artifact，其可见提交与 cancellation 使用同一个短原子 gate。`finalize` 可以在 gate 外构造和校验临时 artifact；`commit_final(FinalCommit)`只能使用owner Repository的`RepositoryCommitCoordinator`在startup reconciliation成功后铸造的opaque、不可变、单次 `CommitAuthority`。公开authority是无副作用handle：除冻结CommitTarget外不暴露`publish()`、journal、recover或callback；真正的`target/journal/publish/recover`快照只存在coordinator私有registry。普通plan只能携带handle，RunController通过内部consumer token原子pop签发快照；同一authority跨run/commit_id复用直接拒绝。lost-ack重试使用RunController已经持有的快照与稳定commit_id做reconciliation，不重新开放publish capability。随后在该Repository同一durability域持久化`CommitIntent(kind, commit_id, run_id, target, created_at)`。`CommitTarget`至少冻结repository_id、artifact_kind、artifact_format、target_ref与expected_manifest_digest，使重启后无需内存closure即可路由到唯一owner并验证目标内容。repository publish必须返回typed `PublishedManifest(target_ref, manifest_digest, result)`，owner快照逐字段匹配CommitTarget后才允许写COMMITTED，正常成功路径也不能跳过digest验证。返回类型错误、target/digest不符及其它确定性合同违例直接写ABORTED并失败，绝不能调用recover“洗白”；只有Repository明确抛出`PublishVisibilityUnknown`，表示atomic replace后可见性确实无法判定，才进入inspection-only recovery。intent fsync期间cancellation仍可受理。intent完成后在短内存gate内做最后一次CancellationToken cancellation check并关闭cancel gate，随后才允许manifest/rename publish：cancel先取得gate，则把intent幂等标为`ABORTED`、publish调用次数必须为0，run不能产生成功artifact；publish先取得gate，则之后的cancel明确返回`TOO_LATE_ALREADY_COMMITTED`（若run已terminal则为`ALREADY_TERMINAL`），不得把已经可见的成功artifact报成CANCELLED。长时间序列化、blob写入和intent fsync不在不可取消gate内；gate只保护最终可见发布及其结果判定。`FitExecution.save()`是独立的notebook/direct CAS保存面，不携带Run final-commit authority；它只能把同一repository `execute()`铸造的process-local execution交回private `_save_execution`。该路径不加入RunController的lost-ack coordinator：publish acknowledgement丢失时不返回成功ref，可见但未被调用者引用的manifest只算content-addressed orphan；不得把这条较窄保证外推为其它Repository的提交合同。

`CommitTarget` journal只接受current exact field set。startup必须在开放任何新Run admission前枚举并reconcile全部current pending intent；未知schema/field set直接fail closed，不能按字段存在猜版本或把unsupported intent当成已提交/已终止。runtime不提供双reader、在线upgrade或fallback；任何离线管理动作也必须保留原intent和repository可见性证据，不能伪造commit resolution。

manifest atomic replace成功但调用方因I/O/进程故障没有收到确认时，Repository必须把这一特定歧义归类为`PublishVisibilityUnknown`，不能用裸`OSError`把所有错误混成未知，也不能把确定性manifest校验错误送入recovery。每种Repository必须按稳定`commit_id`提供权威、幂等的`recover()`：确认已提交时返回`CommitRecovery(committed=True, PublishedManifest(target_ref, manifest_digest, result))`，RunController再次逐字段匹配冻结CommitTarget后才追加`COMMITTED`并完成SUCCEEDED；确认未提交则追加`ABORTED`并按原publish error失败。错误target/digest、无typed manifest evidence或任意字符串result不能证明恢复成功。Repository或commit journal暂时不可判定时，Run保持非terminal `RUNNING/commit-reconciliation-failed` phase、关闭cancel gate、持有resource claims并给出显式重试指令。`COMMITTED`与`ABORTED`在跨进程文件锁内互斥验证，二者都清除pending；commit marker自身写确认丢失也走同一reconciliation，不能重复发布或提前释放claim。startup在接受新run前枚举所有pending CommitIntent并调用对应owner Repository的reconciler；无法找到owner/schema或仍无法判定时fail closed，不重新fire、不把temp文件当成功artifact。

pending reconciliation必须冻结“事实是否已经确定”，不能每次重试重新询问可变callback：`FORCE_ABORT`用于确定性publish/validation失败或validated recovery已确认未提交，重试只幂等写ABORTED；`RECOVER_VISIBILITY`只用于尚未判定的PublishVisibilityUnknown，只有此态调用recover；`FORCE_COMMIT`用于publish已返回并验证成功或validated recovery已给出匹配manifest，持有已验证result并只幂等写COMMITTED。marker写入/确认失败只重试相同resolution，不得让wrong digest经一次abort-marker故障反转成成功，也不得让已可见artifact经一次commit-marker故障反转成ABORTED。

`run(plan)` 内部也使用同一个 RunHandle。Notebook/test 遇到 KeyboardInterrupt 时先 cancel 该 RunHandle、等待 cleanup acknowledgement，再重新抛出或返回取消结果。若等待超过 join deadline，抛出携带 run_id/RunHandle lookup 的 `RunStillCancelling`，RunController registry 继续持有 handle/claims；不能丢掉 handle 后把 cell 当成已经停止。notebook 可继续 `status()/wait()/diagnostics()`。

RunController registry 强引用所有 active handle，以及已经发布 terminal 但 owner thread 尚未被确认退出的 handle；只有另一个线程完成 join/reap 后才移除。`RunHandle.wait/result` 在返回 terminal 结果前也必须确认 owner thread 已退出，不能把“状态字段已写入”冒充线程终止。handle/snapshot 只保存有界字符串错误摘要与必要结果，不保存 `BaseException`、traceback、plan、prepared value、context 或 raw device graph；owner 收尾时主动断开这些引用。baseline 不另建 terminal-handle archive/`forget_terminal` 状态机；可持久的数据诊断归 artifact 与 commit journal，设备cleanup诊断只属于本次run/session。

### 8.3 CancellationToken

- 每个 Run 由 controller 私有 `_CancellationSource` 新建，只向 plan/worker 暴露不可 clear、不可 request 的只读 `CancellationToken`；
- 单调从 active 变 cancelled；
- 绝不 clear/reuse；
- cancel requested 不等于 worker terminated；
- join timeout 后不得清 thread owner、释放资源或允许 restart；
- 每个阻塞 Port 必须有 bounded timeout 或 interrupt contract；
- cancellation 先置 token，再调用 Port 声明为 thread-safe 的 out-of-band `interrupt/abort`，随后由 owner lane 完成正常 cleanup；
- out-of-band interrupt 只在cancel或框架异常需要尽快停止in-flight硬件调用时启用；正常cleanup不先单独interrupt再重复close；
- interrupt 一旦启动就是 terminal barrier：interrupt call 未返回时不得开始可能与其并发碰硬件的 cleanup、不得发布 terminal、不得释放 claim；其迟到异常必须进入 CleanupReport，不能被后台线程吞掉；
- safety-critical Port 不能让 `safe_state` 永久排在可能无限阻塞的同一调用之后：必须有 transport timeout、独立 abort/safe channel 或硬件 watchdog 中至少一种可验证机制；
- 不可中断的 SciPy/NumPy 计算等待返回后丢弃 stale result；
- 需要 hard deadline 的计算使用 disposable subprocess。

current `RemotePulseExecutionClient` 必须建立两条不同的RPyC connection：control通道执行snapshot/prepare/fire/complete，interrupt通道只执行generation-bound safe-state；endpoint的logical blocking limit必须严格小于transport backstop。server返回的 `PreparedPulseRef(connection_generation, artifact_digest)` 在prepare acknowledgement前原子写入同一private session，后续fire/complete只消费该exact ref；每次操作重验server generation、target、clock与geometry，禁止transparent reconnect。第二条socket保证长complete RPC不会在客户端协议层堵死safe请求，但若两条请求最终共享backend/硬件 `_io_lock`，它仍不能冒充独立硬件中断路径；baseline也不因此新增watchdog、SAFE寄存器或重烧bitstream。

logical deadline必须覆盖endpoint的SAFE single-flight lock等待与interrupt RPC本身，不能在transport backstop返回后才检查时钟。当前client用RPyC timed request消费调用方传入的剩余时限；超时后撤销该client并切断两条本地transport，使迟到ack不能成为当前证据且调用方按logical deadline返回。transport断开会触发server owner-disconnect SAFE，所以真正的重复调用约束由`PulseExecutionService`唯一拥有：generation-bound SAFE、disconnect SAFE和emergency SAFE共用一个single-flight gate；后到者等待正在执行的物理SAFE，若其成功并清空prepared authority则直接复用同一SAFE snapshot，只有前一SAFE失败才允许新的物理重试。这样不依赖client/endpoint/runtime对象寿命，不因GC再发第二次backend SAFE，也不启动detached watchdog或伪装远端调用已经终止。

SAFE还必须与prepare/FIRE/complete在**物理backend边界**线性化，而不只是更新Python state。server先把operation epoch推进到INTERRUPTING并调用声明为out-of-band的`request_interrupt()`，再等待唯一backend-operation gate；已经进入backend的调用先退出，随后同一owner才执行`backend.safe_state()`；尚未进入backend的普通调用取得gate后必须重新校验epoch/state，已被SAFE超越就零硬件调用失败。`request_interrupt()`不取得该gate，否则会失去中断意义；adapter contract必须明确它线程安全、非阻塞且不等待普通I/O owner。普通session cleanup不先执行一遍独立SAFE再close；`close_session`是该路径唯一SAFE owner：它执行物理SAFE/stop、join本地operation并返回领域`SessionClosedAck`，不重复backend SAFE。无法确认本次关闭时Run返回FAILED及诊断；该结果不持久化为新连接的设备状态。只有真机证据表明现有safe路径违反既定安全要求，才按bug修复流程评估硬件改变。

Pulse-only real composition 固定为 `RemotePulseExecutionClient -> RemotePulseExecutionEndpoint -> DeviceBroker -> BoundPulsePort -> RunController`。操作者显式提供 `host:port`；composition 建立两条 RPC 连接并解码 current snapshot，网络输错可重试；进入该次连接 authority 前先使同一 server generation 达到 SAFE，再用“显式 deployment endpoint + AssetMap revision”作为 `INSTALLATION_ASSERTED_ENDPOINT` identity，并在每次 capability/prepare/complete/SAFE 继续校验 server generation、target、clock 与 geometry。PulseGUI 只取得 `PulseFacade + PulseTargetDescriptor`，连接、compile、start、reap 与 owned-installation close 均不阻塞 Qt owner。该 composition 只声明 sequencer 能力，不伪造 camera/qCMOS role，也不构造或包装 `RemoteSequencer` 兼容面。完整 real neutral-atom installation 必须独立提供相机 AssetMap 与 qualification。

### 8.4 Cleanup

普通session/temporary resource使用同步context manager与`try/finally`。正常cleanup的单一所有权规则是：

```text
cancel intent（仅取消路径）
-> 必要时发一次out-of-band interrupt，阻止in-flight硬件调用继续推进
-> join/等待该调用退出
-> 每个领域session只调用一次close_session
   -> 由该session唯一执行本领域的stop/disarm/SAFE、terminal drain与最终readback
   -> 返回SessionClosedAck或明确失败
-> workers/builders abort或drain并join
-> temporary config restore
-> reservation release
-> finalize/commit（仅全部必需session关闭成功且数据合同成立时）
-> terminal publication
-> 释放当前进程内ResourceClaim
```

`close_session`是正常关闭路径唯一的物理SAFE/stop owner。generic runtime不在它前后再调用`safe_state`、`verify_safe_state`或第二个cleanup recipe；DeviceBroker只负责排他binding、转发领域调用和撤销当前session authority。cancel可以在正常close之前使用领域声明为线程安全的out-of-band interrupt；框架发现失控调用时可使用同一emergency interrupt。interrupt只负责让in-flight调用退出，不能替代`close_session`的领域终态确认。

业务错误保留为primary error，cleanup错误作为附加诊断。session关闭失败令本次Run/连接关闭失败，且不得提交宣称该Run完整成功的artifact；但这个失败不写持久设备状态、不改变未来进程admission，也不要求进程级重启。旧facade必须先从UI/controller authority摘除，避免关闭过程中或关闭后继续poll；以后若重新连接，必须重新执行physical-owner取得、live identity握手和当前硬件SAFE初始化，能否连接只由这些当下事实决定。

只有worker/session与in-flight interrupt真正退出后，RunController才发布terminal并释放当前进程claim；join尚未完成时继续持有owner，防止同一进程并发碰同一设备。join或close最终返回失败后可以发布FAILED并释放已经终止的本地owner；不能因为失败历史而永久保留claim。硬件自身的sticky fatal/status若存在，adapter在下一次live握手中必须读取并据此拒绝或复位；软件不能伪造同等事实。

数据持久化与设备cleanup严格解耦。`CommitJournal`、CAS manifest、lost-ack reconciliation只证明artifact可见性与crash consistency，其记录不含设备安全id，也不参与硬件连接admission。反过来，session close acknowledgement也不能证明artifact已提交。
### 8.5 Owner-thread command mailbox

长时间 Measurement 的参数修改通过 command mailbox 送到 owner thread，并只在 shot/capture transaction 边界应用。GUI 不跨线程直接 configure driver。有限正式 Run 默认拒绝运行中 reconfigure。每个 accepted control revision 遵守 §7.2 的 terminal ack 合同；同 key 的尚未应用 revision 可被较新 revision SUPERSEDED，但已经开始硬件 transaction 的 revision 不能假装被覆盖。

普通command mailbox只做线程亲和性交接，不是第二套scheduler，也不设置queue容量、pending上限、backlog预算、最大等待turns或预测内存拒绝。普通Qt draft不进入mailbox；只有用户显式commit的控制revision、已经admit的Run命令与owner继续读取当前物理transaction所需的命令才入队。同一硬件owner若同时服务monitor与finite run，composition必须在admission前停止或交接冲突monitor，等待其真实terminal后再启动finite run，而不是让两类硬件session在一个“公平队列”中长期竞争。

真实adapter仍必须把SDK/transport已经存在的blocking-call timeout、可验证cancel/abort与owner-affine close完整暴露给Run cleanup；普通Python线程的逻辑超时不等于driver已经终止。已经进入driver call后的失败按§8.3/§8.4完成interrupt、真实termination与领域`close_session`，不得并发启动替代调用或提前释放当前claim。此处不预建无实现、无第二消费者的`LaneFairnessPolicy`或隔离process协议；若未来profiling证明一个真实共享owner仍会starvation，再从那个adapter的具体transaction语义设计最小调度机制。

## 9. ResourceArbiter

```text
ResourceClaim:
  exact ResourceKey
```

Run启动前一次解析全部claims并原子`acquire_all`；运行中禁止新增claim。ResourceArbiter的职责只有**当前进程内互斥与owner生命周期**：

```text
Acquired
ResourceBusy(conflicting_run)
```

它不保存跨进程设备历史、不解释上一次cleanup结果、不执行SAFE、不自动停止其它Run。Workbench可请求停止冲突owner，但必须等待其RunHandle确认真实termination后再重试。当前合同只有一种含义：一个Run在存活期间独占它声明的exact ResourceKey；相同key冲突，`acquire_all`对完整集合一次判定并提交。只读界面复用领域已经发布的immutable sample/data tap，不另开driver session，也不进入资源claim。

复合Task不能把child的admission rejection压平成`str(error)`。若child在启动边界收到`ResourceBusy(conflicting_run)`，复合handle的immutable `RunSnapshot`必须携带同一个typed outcome；Workbench把直接Run与复合Run的outcome归一成同一种`ResourceBusy`后，才可依据exact `conflicting_run`执行本地handoff。该字段只报告当前启动尝试的admission事实，不是第二套资源状态、错误缓存或跨run journal；Task重试仍重建prepared command，但必须消费同一份冻结request。

真实adapter/server connection的跨进程物理排他由具体backend/composition使用SDK exclusive-open、server-side owner token或本机interprocess lock证明；无法证明时只能开放一个真实控制入口。这个physical-owner proof与ResourceArbiter是两层不同事实，generic runtime不为其再造平行lease或持久设备状态机。physical-owner失效时当前Run失败并关闭当前binding；后续连接必须重新取得proof、执行live identity与当前SAFE初始化。

claims在bind时声明完整superset，并一直持有到本Run的全部worker、session与interrupt调用退出、terminal发布。当前没有第二个真实消费者证明提前phase release值得新增状态机，因此不在prepared value上标last-use，也不提供运行中re-acquire。cleanup失败本身不会永久持有claim；只要本地硬件调用与owner线程已经终止，就发布本次FAILED并释放当前进程互斥。若线程仍未退出则继续持有，因为那是当下并发事实，不是历史惩罚。

`AssetMap`仍是installation-owned、machine/device级持久配置，只保存`asset_id -> canonical ResourceKey + exact adapter kind + expected physical identity/endpoint matcher`；它描述接线身份，不描述设备是否SAFE。更新AssetMap属于离线maintenance/换机操作；下一次连接重新执行identity与SAFE初始化。启动时必须检查map的当前格式、canonical digest、ResourceKey唯一性、matcher可判定性与所有真实adapter覆盖。

capability probe一次返回完整frozen snapshot，camera/sequencer descriptor只能从该snapshot纯函数投影；probe结束后不存在仍可读取raw connection的裸callback。active Run内binding失效后不透明重连；关闭旧runtime后可以在同一application process建立全新runtime，且必须重新live握手。

## 10. Task、Measurement、Processor 与 Analysis

### 10.1 Definition 原则

Definition 是关闭字段的 frozen metadata，不含 callable、Port、Repository、GUI、mutable config 或 binding generation 事实，也不需要每类再建立 Handler Protocol 和公共 ABC。`TaskDefinition/MeasurementDefinition/ProcessorDefinition` 的构造器分别验证其精确字段；因为没有开放metadata容器，行为对象在类型结构上就无处可写，不需要另建递归对象图检查器。领域 owner 的 builder/operator 仍是具名 top-level 函数，由 composition 通过普通 import 显式调用，不以字符串 dispatch、隐藏 registry 或 Definition field 形成第二套执行真相源。所有运行依赖必须出现在 typed request 与领域私有 immutable bindings，所有可变参数必须进入 config revision。

只有会出现在 catalog/UI/API 的能力需要 Definition；Task 内部私有算法保持普通函数。

能力 composition 不依赖 global mutable registry、entry point或开放扫描；它只消费固定namespace下已校验、冻结的package集合：

```text
DefinitionKey:
  owner_package
  stable_definition_id

LogicNodeDeclaration (process-local, headless):
  definition + description
  authoring fields + path hints + dynamic-choice resolver
  typed inputs + static/dynamic outputs + default views
  request build/bind

Composition wiring (application-only):
  LogicNodePackage
  installed dynamic-choice context + typed application host
  bind_api + bind_task_console + optional artifact/UI leaf
```

`zlc_neutral_atom`拥有DefinitionKey、三个关闭Definition dataclass与唯一`LogicNodeDeclaration`值；每个领域 leaf 从自己的 `LogicNodePackage` 指向具名 declaration。固定 namespace discovery 一次冻结全部 package，public API 与 desktop composition 消费同一集合；`TaskConsoleApplicationPorts`只保存投影完成的immutable运行入口，并让重复`(owner_package, stable_definition_id)`启动即失败。不存在第二份Definitions catalog、per-node form/presenter或mutable registry。Definition没有平行schema版本；declaration字段改变后全套current软件原子部署。Workbench只把declaration机械映射为只读`ConsoleCatalogView`，不反向写领域事实。zlc_data/zlc_pulse/frontend不为了进入UI catalog依赖neutral Definition类型，也不建立跨bounded-context universal Definition base。

composition对每个正式`LogicNodeDeclaration`必须产生一个显式installed wiring或typed unavailable reason；遗漏declaration使architecture/E2E失败，避免领域能力已存在却在UI静默消失。可见性不由第二份名单、设备探针或UI策略决定。

### 10.2 Task

```text
TaskDefinition[Request, Result]:
  stable DefinitionKey
  parameter/request schema
```

Task 是 one-shot use case，可以同步组合 CaptureSession、纯 operator 和 typed Repository。Definition 只声明 catalog identity/request schema；owner/composition 的显式 builder 才把 typed request 与 bindings 构造成 `RunPlan[Result]`。它不继承 Measurement/Processor/Analysis，不发布 measurement signal，不拥有 QWidget。

Task 不一定产生 artifact；普通控制/查询 Task 可返回 immutable result，需要持久化时返回本包 typed ref。

Task的中途数值/图像显示只有一条正式路径。需要live frame、3D map或优化轨迹的Task，必须在同一RunPlan中声明finite exact DatasetBuilder或admitted `MonitorTap -> MonitorDataset -> LiveDatasetPort`；`LiveDatasetHost`把port接到唯一owner mailbox，`SignalDataPlane`从其原子snapshot建立typed fronts。阶段/progress/warning仍走EventStream。中途UI、最终Analysis与artifact因此使用同一materializer/revision真相源；Task不发布第二个task-local数值carrier或mutable signal。

同一个Task的科学analysis只执行一次。MOT field的execute阶段从唯一source生成一个immutable analysis result；FINAL result、run-scoped Dataset outputs、report或artifact都引用/投影这个值，不能为了不同consumer再次调用`analyze_mot_scan`或重新materialize原始scan。Calibration与Duration Fidelity都消费的双峰分布拟合属于Readout family的一个纯数值owner；叶节点只提供各自样本与领域acceptance，不复制初始化、阈值或solver公式。

### 10.3 Measurement

```text
MeasurementDefinition:
  stable DefinitionKey
  request_schema_id / binding_schema_id

BoundCameraCapture（仅物理camera capture；不含任何logic-node Definition）:
  FrozenCaptureSpec(owner fingerprint, canonical bytes, digest)
  bound Device Ports
  output schema/cardinality contracts
  ResourceClaims
```

```python
capture_spec = camera_capture_owner.build_frozen_capture_spec(typed_request)  # 纯函数
with capture_factory.open(ctx, capture_spec) as capture:
    sample = capture.read_next(timeout)
```

Measurement 从外部世界取数据，可以访问 Device Port；它不 fit、不渲染、不保存 Figure、不管理 Task terminal state。runtime 负责将 AcquiredSample 包装为 Envelope。

DeviceCapabilitySnapshot 是 connection generation health handshake 后得到的 immutable、versioned descriptor。bind/UI 用它纯解析 expected payload/ValueSchema，preflight 再读取硬件实际设置并要求 fingerprint 相符。formal run 不允许 fire 后才发现 shape/axis；无法预先确定 schema 的 adapter 只能提供 monitor 或先执行独立 probe/config Task 后重新 bind。

camera capture owner构造 immutable BoundCameraCapture/FrozenCaptureSpec；领域Measurement、MOT、release-recapture等只消费它，不把物理capture重标成自己的Definition。runtime只验证owner fingerprint与canonical bytes SHA-256，不执行任意spec snapshot/validate/digest回调，也不会在session中二次freeze。camera Run preflight只建立software CaptureSession/reservation/materializer，execute才在owner I/O lane发送prepare/start。Task 若需要同一种采集，复用同一FrozenCaptureSpec构造器/CaptureSession，而不是启动一个 child Measurement Run。

### 10.4 Processor

```text
ProcessorDefinition:
  stable DefinitionKey
  title
  config_schema_id

capability-owned prepare_<capability>_processor(
  typed request,
  admitted artifact/input refs,
) -> Prepared<Capability>Processor

Prepared<Capability>Processor:
  frozen request/config/model identity
  closed output declarations
  evaluate(typed immutable input facts) -> typed atomic evaluation
```

Processor Definition 只保存关闭的 catalog/schema metadata。每个具体 capability 在自己的 `processor_application.py` 中拥有 typed request admission、artifact/input resolution、prepared value、输出 vocabulary、算法调用和 lineage；neutral `HostedProcessor`只托管一个已准备processor的latest-only owner-lane lifecycle，并把typed `DerivedSignalOutput`交给`SignalDataPlane`，不拥有算法、字段或presentation。runtime 不定义通用formal processor graph、operator callback、任意stream edge binding或递归config snapshot/fingerprint framework；Workbench也不得从Definition字段重建这些事实。

正式 Occupancy 的具体合同是 `PreparedOccupancyProcessor`。prepare 一次解析并 admit `CalibrationArtifactRef`、冻结 camera input intent、model kind 与 `OCCUPANCY_LIVE_OUTPUT_DECLARATIONS`；随后有两条真实消费路径：

1. `evaluate(OwnedSnapshot, MonitorCoverage, source_event_digest)` 原子返回一个 `OccupancyProcessorEvaluation`，其中 counts、occupied、rate、source revision、coverage、event digest、calibration/model lineage 与 join digest 来自同一次分类；neutral `HostedProcessor`的共享latest-only lane只调度这一方法并路由结果，不拥有算法、schema、subscription或presentation。
2. `start_signal_events(SignalEventSource)` 构造 Occupancy capability 自有的 `RunningOccupancySignalSource`；每个上游事件只原子发布一个 `OccupancySignalValues(counts, occupied, rate)`。只有该owner证明严格1:1、保留direct EventRef并完成association propagation时，它才可作为PulseScan正式signal；普通latest evaluation不能升级为该证明。

finite Occupancy artifact 路径直接复用同一 capability classification primitive和已admit calibration，在完整 `OwnedSnapshot`/artifact context 上生成counts与occupied，再由自己的flat artifact Run提交；它不把累计DataBlock伪装成事件，也不经过通用processor worker。相同source/calibration/model identity必须得到相同分类结果；算法不读wall clock、module global config或global RNG。

所有 Processor capability 都遵守同一最小规则：output schema/vocabulary必须在prepare时由typed input contract与immutable config确定，不能读第一帧后改变axis或record fields；一次evaluation的同shot字段作为一个typed record原子交付，不按字段拆成需要分布式原子性的平行signal；prepared application不接触设备、Repository、QWidget或frontend。需要设备grab/fire的one-shot能力属于Task或Measurement；完整dataset上的fit、calibration或report属于Analysis。

Temperature release-recapture 的相邻 `READOUT_EVENT=0/1`、严格 `2:1` 顺序与EOS完整性由其自己的Measurement/application拥有，输出保留完整 `(R,P)` 与声明的SITE axis；它不为generic reducer或workflow framework提供理由。新增Processor首先提供capability-owned typed prepare/evaluate和真实产品consumer；只有多个独立consumer证明相同hosting合同后，才从它们抽取最小共享host。

### 10.5 Analysis

```text
BoundFit:                              # zlc_data-owned，不含 runtime slot/ref
  frozen FitSpec + expected DatasetSchema fingerprint
  resolved fit/batch axes + model/numeric policy
  run(OwnedSnapshot) -> FitResultBatch

DomainAnalysisRequest[Result]:         # 仅在领域物理语义需要时定义
  exact immutable ArtifactRef inputs
  frozen domain config/policy
  compile -> one flat RunPlan[Result]
```

离线/权威数据计算不访问 Device、Hub/latest、QWidget 或 mutable DatasetBuilder。它消费携 exact DatasetRevisionRef 的 OwnedSnapshot或immutable artifact，产生 FitResultBatch、CalibrationArtifact、report 等 typed result。`FitSpec/BoundFit/FitResultBatch` 与 `BoundFit.run()` 全部由 zlc_data 拥有；Calibration/ReadoutFidelity 等带 neutral 物理语义的算法使用自己的typed request/compiler，不以generic Fit wrapper冒充领域判断。neutral 的 Definition 词汇不重新注册或包装通用Fit；Workbench只在Figure已有的`Fit` surface投影这项capability。

Fit只有两个hosting路径：interactive/offline adapter把已冻结snapshot交给同一个BoundFit；需要durable领域artifact的typed request从已提交输入编译一个flat RunPlan并拥有自己的一次FinalCommit。二者都不是Processor，也不通过伪造“累计DataBlock event”接入sample stream。

`DatasetInputSlot -> AnalysisStep -> post-materialization pipeline`明确是**延后设计**，不是baseline。重开它必须先给出至少一个真实自动/headless或下游consumer、其失败/cancel语义以及artifact原子性需求。默认优先选择“输入FINAL artifact -> 独立flat analysis Run”，因为它复用immutable replay边界且不改scan提交；只有明确要求scan+analysis不可分割成功时，才设计同一FinalCommit可恢复的composite result。不得先建`BoundOperation Protocol`、Analysis registry、descriptor hierarchy、program DSL或child Run。

### 10.6 Pipeline composition

TaskConsole中的Measurement与Processor连接不形成通用pipeline DAG。Measurement在自己的capability application中把typed request和immutable bindings编译成一个flat RunPlan；Processor row则由composition解析`DatasetInputSpec/ArtifactInputSpec`，调用capability-owned prepare取得一个关闭的prepared application，再交给neutral `HostedProcessor`托管lifecycle与latest-only调度。generic runtime不保存processor图、per-edge QoS/criticality或任意sink callback结构。

Camera→Occupancy的真实live路径只有一条：neutral `SignalDataPlane`在owner线程接纳一个明确`SignalValue` revision，`HostedProcessor`把其`OwnedSnapshot + MonitorCoverage + source event digest`交给`PreparedOccupancyProcessor.evaluate()`；capability返回同一revision的完整counts/occupied/rate事务，data-plane owner再原子发布。busy时共享lane可用新revision替换尚未开始的旧evaluation；这是processor delivery policy，不能修改source coverage、event stream、calibration或artifact authority。`HostedProcessor`不持领域算法、schema构造、Repository callback或某个Processor字段分支；Occupancy的Camera/Calibration需求只来自Occupancy owner的typed request和prepare。Workbench只把signal key和exact-revision presentation sidecar送给frontend，不持第二份数据面。

需要lossless未来事件的Occupancy consumer使用同一个prepared application的`start_signal_events()`，得到capability-owned `RunningOccupancySignalSource`；它独立消费上游`SignalEventSource`并发布`OccupancySignalValues`。PulseScan只绑定该producer已经公开的signal/association capability，不编译Camera→Processor通用图，也不在自己的application复制classifier。另一个Processor若要成为scan source，必须先由自己的owner提供真实derived signal source、cardinality/association证据和product consumer；不能靠动态registry、通用worker或任意Processor接口预造。

feedback control使用revisioned ControlTopic；完整dataset上的post-materialization算法使用自己的flat Analysis Run。monitor leaf失败只影响对应panel/Processor row，不反向改写仍健康的Measurement Run；association signal source失败按自身producer合同终止并使依赖它的formal consumer失败。两者不能用一个通用`REQUIRED | BEST_EFFORT_MONITOR` edge policy混合。

最小 camera pipeline compiler 只接受 `1 BoundCameraCapture -> 1 DatasetMaterializerSpec -> opaque in-memory PipelineResult`：没有 processor、analysis、feedback、持久sink callback、可选 child 或通用 node/edge DSL。它在 `RunController.start()` 取得 claim 之前完成FrozenCaptureSpec owner、payload/adapter/schema、完整cell permutation与expected event count校验；RunPlan.preflight只用真实run_id创建software TraceBinding、CaptureSession、唯一exact reservation、cursor和DatasetBuilder，不发送任何device command；真实driver分配在prepare/arm时失败就使本次run失败。execute在prepared state完整返回后才prepare/start。CaptureSession自己从冻结`expected_cells[source_ordinal]`派生join key，不接受execute层传入另一个key；只有该reservation已经ACTIVE且持有绑定schema/adapter/完整schedule的ExactDatasetReadiness后，start才可触达设备。

`BoundCapturePort`只接受DeviceBroker针对当前BoundDevice/binding/generation执行endpoint-owned capability probe后mint的opaque attestation，不能把普通`CaptureCapabilitySnapshot`拼到真实设备上；probe全程持有broker probing token并与Run open、binding invalidation互斥，跨过任何activity epoch的结果不得发布。FrozenCaptureSpec在进入runtime前已由camera capture owner生成canonical bytes，runtime自行重算SHA-256并要求contract/capability/spec owner fingerprint一致，prepare阶段没有替换或回调入口。CaptureSession创建线程就是其owner I/O lane，prepare/start/read/complete/termination/cleanup跨线程调用一律拒绝。普通整数、字符串或任意格式正确的digest不构成物理证明；正常terminal必须同时核对generation、spec/settings/capability binding、全部source ordinal、produced/drained、ordered metadata digest、source stopped、no-more-frame和真实join，才可mint不可伪造的CaptureCompletion。取消后普通execution capability会被撤销，因此BoundCapturePort必须提供thread-safe ABORT/DISARM与有限blocking-call bound；该bound写入每个prepare/start/read/complete/session-close command，adapter必须把它交给SDK wait/poll或自己的有界等待，不能只把它留作描述字段。cleanup phase发送绑定本session的`SessionCloseCommand`；wrong-session、stop/drain/join未知或超时都使当前run失败，不能靠safe-state布尔值跳过join。formal compiler只消费该session拥有的CaptureCompletion，再取其中EOS交给DatasetBuilder seal，并交叉验证sealed artifact与terminal的metadata fingerprint/digest；PipelineResult由compiler私有authority mint并再次核对coverage/count/digest，调用方不能拼接另一个terminal伪造成功。裸EOS不构成pipeline成功。DatasetBuilder是exact reservation teardown的唯一owner：success seal、preflight/execute/cancel失败都在独立finally中close，最终reservation必须RELEASED且registry为空，前一步cleanup失败不能阻止它。finalize阶段的persistent sink只接受storage-owned staged FinalCommit，不接受“任意 callback + requires_commit bool”。新的领域Analysis或复合Measurement必须拥有自己的具体flat RunPlan与真实consumer；不得把这个最小直线扩展成通用processor graph或递归工作流引擎。

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
BoundFit.run()
```

当前只有一个 solver 时不建立 FitSolver Protocol。

核心 transform 类型是可序列化的值，不含 QWidget、artist、callable 或 live signal：

```text
DataTransformSpec:
  operations: tuple[Selection | ReductionSpec, ...]

ReductionSpec:
  axis_ids
  method: MEAN | SUM | MIN | MAX
  missing_policy: REQUIRE_ALL | OMIT_MISSING
  validity_policy

CommittedTransform:
  input_schema_fingerprint
  spec: DataTransformSpec
  output_schema_fingerprint
```

`commit_transform(DatasetSchema, DataTransformSpec) -> CommittedTransform` 在不接触 values 的情况下验证并冻结 authority；`apply_transform(OwnedSnapshot, CommittedTransform) -> TransformedData` 是唯一执行入口。它验证 input/output schema fingerprint、AxisId、reducer 封闭合同、coordinates、validity 和 operation 顺序，并保留 exact DatasetRevisionRef。DataTransformSpec 只描述“对数据做什么”，不包含 auto/default、UI revision/origin 或显示 binding；CommittedTransform 也不保存自嵌 digest、逐 operation record 或 artifact identity。Reducer 能力属于 zlc_data 算法目录，不写入 ValueSchema/DatasetSchema；新增 renderer/analysis 不改变数据 fingerprint。

frontend figure 拥有只服务于呈现的：

```text
ViewIntent                         # frontend 展示意图的封闭词汇，不持久化成另一状态机
ViewContract                       # dataset-fed intent 的静态 axis/reduction 规则
DocumentViewContract               # document-fed intent 的来源类型防火墙
ViewSpec                           # dataset presentation 唯一可保存的 spec
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

ViewSpec 是 figure 唯一持久 presentation 类型；它不保存 authority seed、CommittedTransform、FitSpec 或 ScanOutputContract。FigureEvaluator 根据当前 immutable DataBlock revision/validity 和 Selection snapshot，把 ViewSpec 直接求值为 `EvaluatedFigureData`。每个 `FixedIndex` 与 `LatestNonempty` navigation 都必须在结果中留下 AxisId、实际 index、coordinate 和 input revision resolution record；renderer 不得只看 ViewSpec 猜“当前切片”。renderer DTO 不能进入 zlc_data authority path。`ViewSuggestion` 只是 ViewSpec 是否能被安全构造的解释性返回，不复制 axis bucket，不成为第四层 projection，也不进入 artifact。

auto slice、latest、repeat mean 和鼠标刚画出的 ROI 都是 display state/candidate。用户从 Fit/Scan 动作接受某个候选时，Workbench 根据当前 Selection snapshot 在对应 FitSpec/ScanOutputContract draft 中重新构造 DataTransformSpec，再交给 commit_transform；不存在把 ViewSpec 的 axis binding/display operation cast、unwrap 或复制成权威 transform 的通用函数。已保存的权威复用项是独立 AnalysisPreset/FitSpec/ScanOutputContract，不藏在 workspace ViewSpec 中。

`suggest_view` 返回轻量 `ViewSuggestion` 供 UI 显示。算法优先把信息轴放入 ViewContract 允许的 display/facet/batch，其余轴给有坐标标签的 slider/select；batch/facet 联合容量由一个有界、确定性的局部规划器求解，不能按 dataset tuple 顺序贪心。Selection 只限定允许的 index/coordinate 集合，本身不表示 mean、sum、integrate 或 count；除 repeat 的 ViewContract policy 外，suggestion 不从“框了一个范围”发明 reducer。仍需压掉有物理信息的轴而没有唯一规则时返回 `NEEDS_INPUT`。baseline 因而只有一个可保存 ViewSpec、一个权威 CommittedTransform、一个瞬时 suggestion 和一个 renderer DTO；没有可互相转换的五层 projection 状态机。

`LatestNonempty` 只表示“显式 display selection 内最大的非空**逻辑 repeat index**”，不表示最后发布事件、硬件时序或 provenance current cell；它只能绑定 repeat AxisId。若剩余多个 cell 的最大 valid repeat 不同，ViewContract 也不能做 per-cell latest gather 后拼出一条伪装成同一次 shot 的曲线。Workbench live current 必须读取同一个 `MonitorDatasetSnapshot.head + event_refs + block` 后生成覆盖 repeat 与全部 point axes 的完整显式 Selection；figure 层禁止从 axis role、最大 nonempty index、过期 DatasetProgress 或 PointLayout 猜“最后发布”。

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
PULSE       # authored pulse document-fed；不是 DataBlock projection
```

每个 ViewIntent 都有 declarative contract，但数据集视图与文档视图必须是不同类型，不能为了共用字段而伪造语义：

```text
ViewContract:
  allowed/preferred display roles
  allowed x/image/sample/batch/facet bindings + value contract
  maximum visible facets/layers
  permitted presentation-only reductions
  unresolved-axis policy

DocumentViewContract:
  owner-qualified source schema
  # 无 repeat/axis-role/batch/facet 字段
```

其中 IMAGE/CURVE/HISTOGRAM/METER 是 dataset-evaluated intent，使用 `ViewContract` 并走 `suggest_view -> ViewSpec -> FigureEvaluator`。PULSE 使用 `DocumentViewContract(source_schema="zlc_pulse.PulseTimelineDocument")`；它必须由 authored pulse document 提取 plain render data，再经唯一 pulse renderer 生成 `PulsePanelPayload`，禁止伪造 DataBlock/ViewSpec/DatasetRevisionRef，禁止进入 FigureEvaluator 或通用 Figure codec。它可以复用同一个 `SinglePanelHost` 与 x-only gesture vocabulary，但不能因此把 pulse 当成 CURVE/IMAGE 数据投影，也不能把 display-only time span 升级成 zlc_data Selection。

两类 contract 都是 frontend figure 的静态值。新 plot kind 必须先声明真实来源合同；只有 dataset-fed kind 才复用 suggestion/validation pipeline。用户 preference 只能在 dataset 合同列出的等价安全选项中选默认项，例如 image repeat 显示 latest 或 mean；preference 必须随 workspace 保存，不能自行开放新的 reduction capability。

`ViewIntent`只表示renderer需要哪一种输入/轴合同，不是TaskConsole菜单、输入slot、panel复合布局或产品能力注册表。TaskConsole消费`zlc_frontend.plot_kind`的closed presentation tuple；它自己的current布局、panel与logic-row记录只属于`zlc_workbench.task_console.console_records/console_state`，不得从通用frontend根再导出。SITES是exact composite payload，GRID是board/facet布局，二者都不能伪装成普通IMAGE intent，PULSE也不能作为空dataset panel加入。`zlc_data`不得拥有label、panel尺寸、repeat菜单或render-family词汇。禁止把该closed tuple升级成plugin、registry或class factory。

TaskConsole 的 Setting/Edit 必须把当前 dataset schema 交给同一个 `ViewContract`，只展示该 intent 真实允许的 `RepeatViewMode`；选择结果直接形成 `ViewPreferences`，不能另存 `average/add/replace/create/pool` 字符串表。`roll` 是 Monitor rolling dataset 的采集/历史策略，不是 repeat reduction；`create` 由 CURVE 的 BATCH 表达，histogram pool 由 SAMPLE 表达。GRID 的 facet 选择保存具名 `AxisId` 并由 `FigureEvaluator` 产生 cells，复用同一 DataFigure/SinglePanelHost 的 overview、focus、selector、fit 与 export；不得存在 `points:k/data:k`、`facet_cells`、按 shape/rank 猜轴或第二 Grid renderer。TaskConsole 必须始终提供 `main` 规定的六种 Add Panel 用户面，并由真实 Qt 流程证明 repeat/sites/grid 的端到端行为。

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

- `RESOLVED`：由 ViewContract、axis role 与明确 Selection 唯一决定；可包含始终显示实际 coordinate、可编辑且永不进入权威路径的 presentation-only slider；
- `REVIEW_REQUIRED`：可以安全预览，但包含用户明确提出、必须持续突出显示的临时有损 reducer/sample policy；普通可见 FixedIndex navigation 本身不触发弹窗；
- `NEEDS_INPUT`：无法在不压掉有物理信息的 axis 时满足该 ViewIntent，此时 `spec=None`。它可以显示占位说明或另一种无损视图，但不能进入权威路径。

每个输入 AxisId 必须恰好出现在 ViewSpec 的一条 AxisViewBinding 中；UI 没有同时画出的轴也必须是 slider/facet/batch/selected/reduced 之一，不能靠独立 `hidden_axes` 字段成为丢轴通道。summary、displayed/reduced axes 和 lossy steps 全部从 ViewSpec 派生，ViewSuggestion 不保存第二份。选择 x/image display axis 是物理语义决定：同一优先级有多个同 role 候选且 Selection 未消歧时必须返回 alternatives/`NEEDS_INPUT`。而把其余轴放入 batch/facet/slider 是不丢轴的布局问题，可按 ViewContract role priority + AxisId 稳定求一个满足容量的方案；AxisId 只作相同语义方案的确定性 tie-break，绝不推断物理角色。对 point axes，所有自动 FixedIndex 必须来自同一条实际存在且满足 Selection 的 PointLayout storage row；逐轴拿第一个 index 拼出 EXPLICIT layout 中不存在的 tuple，或 Selection 覆盖不到任何物理 row，都必须 fail closed。

自动选择使用稳定优先级：

1. 规范化 Selection，一次解析每根轴的 allowed indices，并确认 point-axis 限制至少覆盖一条真实 PointLayout row；Selection 不产生 reducer；
2. 选择该视图的 display axis；真实 display-axis 歧义要求最小用户输入，显式 scalar Selection 可消除相应候选；
3. 按 ViewContract 绑定 repeat policy，再冻结用户显式 batch/facet/sample preference；
4. 对剩余信息轴按合同声明的 automatic-role 顺序求满足 batch/facet product 上限的方案；排序只用 role priority + AxisId，不用 ndarray/data-axis tuple 顺序；
5. SLIDER 在 display 层可初始化为 allowed set 中一个带标签的 FixedIndex；point-axis FixedIndex 必须共同取自第 1 步同一条物理 row；
6. 无合法布局或仍需未声明 reduction 时返回 `NEEDS_INPUT`。

禁止在权威路径或无标签显示中以 `index=0`、flatten、全局 `nanmean` 作为降维兜底。presentation-only slider 可以从**第一个 allowed index**初始化（Selection 后不一定是原始 index 0），但 panel 必须显示 axis name、实际 coordinate 与“仅预览”，允许立即编辑，evaluator 必须输出对应 resolution record，并且它不能被静默升级为 committed input。rolling 的 current cell 不是这种初始化值，必须由 runtime provenance 驱动的完整显式 Selection 给出。

### 11.4 role 与 ViewIntent 的组合规则

role 只说明 axis 是什么；ViewIntent 说明用户现在想怎么看。两者共同决定策略。特别是 repeat 不存在全局 `mean` 默认：

| axis role | IMAGE | CURVE | HISTOGRAM | METER |
|---|---|---|---|---|
| repeat | mean 或最大非空逻辑 repeat，由 image contract 声明并标注 | mean/error-band 或 batch，由 curve contract 声明 | pool 为样本，绝不先 mean | 最大非空逻辑 repeat 或声明的统计量；不代表最后发布事件 |
| scan-point | 带实际坐标的 fixed slider 或 facet；rolling current 由显式完整 cell Selection 驱动 | 优先作为 x | batch/facet | 带标签 fixed/显式 current Selection；不能静默 reduce |
| spatial-x/y | 显示轴 | 需要 ROI/Selection，不自动平均 | batch/facet，除非明确 pool sites | 需要 ROI 或物理积分 |
| spectral | curve x/facet/fixed slider；不能把最大波长索引叫 latest | 优先作为 x | batch/facet | 带标签 fixed；band/integral 必须显式 |
| site/component | facet/batch/select | facet/batch/select | 默认逐 site；pool 必须显式 | select |

上表只覆盖四种 dataset-evaluated intent；PULSE 的 channel/time/repeat bracket 来自 authored pulse document，不参与 axis-role 自动建议。

`mean`、`sum`、`integrate` 是不同物理 reduction，不能编码成一个含义模糊的 reducer。通用 `mean/sum/min/max` 使用 zlc_data 中封闭的 current reducer 合同，并由用户/analysis spec 显式选择；ROI photon count、相机畸变校正等带设备/物理含义的操作由 neutral 具体Processor capability或Analysis定义，不能因输入恰好是 image 就由 frontend 自动提出。普通 image 默认只能显示、选择或保留 spatial axes。

histogram 的 repeat 语义是 sample binding，不是 reduction，也不是把轴 flatten 后丢掉身份。ViewSpec 保留 repeat AxisId，Histogram layer 将其声明为 `sample_axes`。这由 `HISTOGRAM` 的 ViewContract 表达，render 主干不允许再出现 `if kind == "hist"` 特例。

### 11.5 显示建议如何产生权威 DataTransform

显示与权威提交使用不同类型，防止布尔标志被漏检：

```text
ViewSpec                         # presentation-only；axis binding/display operation 不可提交

CommittedTransform:
  input_schema_fingerprint
  spec: DataTransformSpec
  output_schema_fingerprint
```

CommittedTransform 中的 selection/ROI 必须是根据当前 Selection snapshot 解析并重建的不可变 authority intent，不得保存 UI revision、origin、live ControlTopic、slider 或 mutable FigureSession 引用。ViewSpec 的 x/image/sample/batch/facet binding 与 display operation 不进入 CommittedTransform；fit axes/batch axes 由 FitProblem 明确表达，scan batch axes 由 ScanOutputContract 表达。

```text
commit_transform(schema, authoritative_spec)
  -> CommittedTransform
```

该 zlc_data 函数只验证并冻结完整 DataTransformSpec，不做建议。CommittedTransform 本身只保存 input/output schema fingerprint 与 spec；需要 durable content identity 时由外层 artifact/CAS 对 owner tree 求 digest。Notebook/headless 用户可以显式构造 DataTransformSpec 后调用它，或从所属 artifact 加载已保存的 CommittedTransform，不依赖 frontend figure/Qt。

提交规则：

1. Panel 始终显示当前视图摘要，并逐项标 scope，例如 `x=detuning · repeat=mean/32 [display] · ROI=A [candidate] · batch=site`；
2. 打开 Fit/Scan 配置时，从当前 Selection snapshot、DatasetSchema 与明确 AnalysisPreset 构造领域 draft；ViewSpec 只提供“用户正在看什么”的候选提示，任何 display reduction 都不复制；
3. Fit draft 由 authority-side `suggest_fit_draft(schema, FitPolicy, SelectionCandidate)` 派生 repeat reduction、fit axes、batch axes；它返回 FitDraft/DataTransformSpec candidate，与 ViewSuggestion 没有继承、转换或字段复制关系。Scan draft同样从 ScanOutputContract 独立派生 output/reduction。二者都做 axis-total-coverage 验证，不能继承 image/sample/facet binding；
4. 若 status 是 `NEEDS_INPUT`，UI 聚焦缺失的 ROI/axis/reducer，禁止开始权威操作；
5. `RESOLVED` 可由紧邻权威 draft 摘要的正常动作直接提交；`REVIEW_REQUIRED` 必须突出显示有损步骤，用户接受该摘要或编辑后再生成 CommittedTransform。这里的 status 来自 Fit/Scan draft validator，不沿用显示 ViewSuggestion 的 status；
6. 需要 transform 的 RunPlan/AnalysisCommand 字段只接收 CommittedTransform，运行中 UI 改选择会产生新 revision，不能改变已启动 run；
7. schema fingerprint 不匹配时提交失效，重新建议或要求修正，不能按 axis index 套用；
8. `commit_transform` 的参数类型只接受 DataTransformSpec，frontend.figure 不提供 ViewSpec/display operation -> DataTransformSpec 转换 API；Analysis result、FitResultBatch、ScanArtifact 和派生 artifact 保存完整 CommittedTransform、input lineage 与 artifact owner digest，不保存 UI revision 或逐 operation 历史对象。

保存 workspace 时保存用户最终选择的 ViewSpec，保证重开后的画面一致；保存权威派生 artifact 时还必须保存 CommittedTransform 与 input lineage。保存视图不等于把显示结果冒充原始数据。

### 11.6 多维示例

输入相机数据：

```text
(repeat=32, detuning=21, height=40, width=20)
```

- IMAGE：height/width 为显示轴；detuning 是带坐标标签的 fixed slider，rolling controller 有 provenance 时可用完整显式 Selection 改到 current cell；repeat 按 IMAGE ViewContract 选择 `mean/32` 或最大非空逻辑 repeat，panel 明示。底层四个 axis 完整保留。
- CURVE：detuning 作为 x；height/width 不能自动平均。没有 ROI 时返回 `NEEDS_INPUT`，同时继续显示 image 让用户框 ROI；定义 ROI count 后曲线建议成为 `REVIEW_REQUIRED` 或 `RESOLVED`。
- HISTOGRAM：repeat 作为独立样本 pool；site/spatial 维默认 batch/facet，不把 repeat 先平均，也不默认把所有 site 混成一个分布。
- Fit authority draft：`suggest_fit_draft` 令 detuning 成为 fit axis；repeat 默认 preserve 为 batch，或预填用户已提交的 repeat reduction；剩余 site/spatial axis 继续成为 batch。每个 batch cell 产生一个 FitResult，组成 FitResultBatch，绝不对剩余轴 `nanmean`。该步骤不是 ViewIntent。
- headless/notebook 的 `fit_spec_for` 使用同一条 role-driven 规则：显式 axes 原样采用；省略时只有 model 声明的 axis role 存在**唯一完整 matching**才生成权威 FitSpec。只要存在第二组物理上合法的 matching（即使 role 不同，例如 scan 与 spectral），就要求用户明确 `fit_axis_ids`；稳定优先级只允许用于 frontend 的显示/草稿建议，不能直接提交成权威 fit。它绝不按 rank/shape/singleton 猜轴，也不自动 Select/Reduce/flatten；repeat、point、site、component 与所有未选中的多维 data axes 逐根保留为具名 batch axes。
- METER：没有已声明 ROI/integral 时为 `NEEDS_INPUT`，不能显示像素 `(0,0)` 冒充物理 signal。

### 11.7 Fit contract

zlc_data 的可复现分析合同是 FitSpec，solver 接收已解析数组的 FitProblem；input ref 属于调用 adapter：

```text
FitSpec:
  input_schema_fingerprint
  committed_transform: optional CommittedTransform  # 终态模型；当前direct-camera实现必有
  fit_axes: tuple[AxisId, ...]
  batch_axes: tuple[AxisId, ...]
  model_id
  constraints: tuple[initial/lower/upper/fixed]
  numeric_policy: max_evaluations + covariance_rcond

WorkbenchFitRequest / FigureFitRequest:
  input_ref + input_revision
  spec: FitSpec

FitProblem:
  private transient packed used coordinates
  packed authoritative observations + per-batch offsets/counts
  fit_axis_specs（coordinate source 由 AxisSpec 唯一派生）
  batch_axis_specs + sparse/rectangular batch layout

BoundFit:
  resolved closed-catalog model + mathematical/user bounds

FitResultBatch:
  batch_axis_specs
  batch_layout: RECT_C | RECT_F | EXPLICIT | PRODUCT(factors)
  parameter_schema/unit（由 FitSpec model_id + fit AxisSpec + value unit 唯一派生）
  parameter_values: (B, parameter)
  covariance/uncertainty
  RSS + R²
  per_batch numeric status/error
  source_ref + FitSpec（包含 input/transform/model/constraints/numeric policy）
  scipy_version（仅 producer lineage）
```

`model_id` 是不可原地改写的完整数学语义身份，覆盖公式、参数顺序/命名/约定、axis requirement 与参数静态 domain；任何一项数学语义改变都使用新的描述性 model id，不能 bump 一个改稿数字。当前只有一个 initializer/solver implementation，分别由 model implementation table 与 `BoundFit.run()` 直接选择，不再保存 `initializer_id`、`solver_contract_id` 或对象 digest 这些镜像真相。SciPy least-squares 的显式 options 由 solver owner 单点测试；`scipy_version` 只是结果的被动 producer lineage，不是 replay gate、环境证明、签名或 attestation。

FitModel 声明 axis role requirement、参数顺序、相对 unit relation 与数学静态 domain；BoundFit 验证有效 axis 的 unit/frame compatibility，FitResultBatch 从实际轴派生参数 unit。当前 `FitParameterConstraint` 的 initial/lower/upper/fixed 数值信任边界是“caller 已提供绑定后有效 axis/value 的 canonical unit”；core/codec 不按 label、数量级或裸数值猜单位。未来若 notebook/UI 接受非 canonical 输入，unit slice 必须先交付 zlc_data owner 的显式转换入口及测试，再允许 adapter 构造 FitSpec；在该入口存在前 UI 不能声称已自动转换。单位或 frame 不兼容是 request error，不能先拟合再只改 label。

initializer 只提供有限 seed，不拥有 hard bound。唯一 hard bound 来源是参数数学 domain（例如 positive、nonnegative、phase 主值区间）和用户显式 constraint；data range、选区 span、观测 contrast 等启发式绝不能变成无法扩宽的物理边界。全部参数都已有 fixed 或 explicit initial 时，执行直接使用 caller seed，不调用 data-derived initializer。最少 observation 也不是 model catalog 常量：每个 bound request 按 `max(2, free_parameter_count + 1)` 派生，固定参数是 caller 提供的 hypothesis，不再要求数据重新识别它。时间模型可用 selection window 改善 seed，但 artifact 参数仍保持 absolute-coordinate 定义。自由约束与静态 domain 必须有至少一个可表示的内部浮点值；phase 使用唯一主值表示 `[-π, π)`，不能同时保存 `+π/-π` 两个等价 artifact。FitProblem 构造器验证最终 packed shape；大输入的性能和临时复制由真实profiling驱动优化。

`FitBatchStatus.CONVERGED` 只表示数值求解完成；generic fit core 不拥有实验域的“科学上可用”判据。结果保留参数、RSS/R²、observation/evaluation counts 与 covariance validity，供 UI 和领域 consumer 判断。active authoritative bound 会使 covariance 明确 invalid/canonical-zero，但不把已经收敛的数值结果伪装成失败。SNR、支持区间、alias prior、目标参数容差或“哪些 batch 必须通过”等 policy 由真正消费这些参数的领域 AnalysisSpec 拥有；baseline 不建立通用 `FitAcceptance`、reason 字段或 model-local quality-gate DSL。执行失败使用其它 typed status、非空 execution error与 canonical-zero 数值，不能把不可辨识或领域不接受伪装成 solver failure。

generic damped-sine 只拟合 catalog 定义的 `baseband_frequency` 数值，不宣称证明无混叠，也不从 coordinate gap、shape 或 rank 猜 Nyquist。formal 物理频率 consumer 若需要无混叠结论，必须在领域 request 中持有采样设计与 band-limit prior（或提高硬件采样率）；软件不能从已 alias 的样本反推出“真实高频”。因此 FitProblem 不再持久或传播 sampling quantum/index-gcd 这类只服务一个推测性 acceptance gate 的字段。

packing 以 declared coordinate 为主排序、logical index 为 duplicate-coordinate tie-break，物理 storage permutation 不能改变入选观测。coordinate-less axis 使用 `index_origin + logical_index` 的 absolute coordinate；若 AxisSpec 声明 unit 就保留该 unit，否则参数 unit 才是 `index`。连续整数坐标必须在 bind 时证明每一点都能被 float64 精确区分，不能只验端点后让中间 x 静默重复。完整 coordinate validation 只在 `BoundFit` 绑定时执行一次并缓存每根 fit axis 的 source；packing 无条件消费该结果，不能在每个 batch、FitProblem 或 property 中重新扫描 declared coordinate。`BoundFit` 只接收 FitSpec 与 expected DatasetSchema，effective schema/model 均在内部单次派生；package-private packing/solver 只接受 exact BoundFit，不能把可覆写 `__post_init__` 的普通子类当成已验证结果。TransformedSchema 的 canonical fingerprint 在同一 immutable 实例首次需要时计算一次并缓存；identity bind 不为未消费的 digest 付出 O(P) 成本。Selection 后重复选择仍保留 absolute coordinate，不重基。packing按canonical order保留全部valid observations，不做抽样、候选筛选或数据依赖截断；valid NaN/Inf 必须进入数值路径并fail closed，invalid nonfinite 不进入。dense qCMOS image 路径不构造全帧rank/value副本；sparse point axis 的坐标 gather 与 canonical row order 只按 present physical rows 分配，绝不能先建立 logical-size coordinate array。后续只有在真实profile证明瓶颈时才优化具体packing实现，不能为假设风险再造索引框架。

时间模型继承当前真机验证过的 absolute-coordinate 语义：decay amplitude 仍表示 x=0 的幅度，damped-sine phase 仍相对 absolute x；Selection/CommittedTransform 只筛选观测，不偷偷用选区最小值重定相位或幅度。若未来确有“从选区起点计时”的物理需求，必须使用显式权威坐标变换或新的描述性 model id。`FitResultBatch.evaluate_batch()` 是 overlay/replay 的唯一结果求值入口，使用相同 absolute coordinates 与 catalog evaluator。damped-sine 将 amplitude 约束为非负、phase 约束在主值区间，消除 `(A, φ) == (-A, φ+π)` 的 artifact 歧义。

领域中立的一维数学模型接受具名有限数值轴，包括 scan、spectral、spatial lineout 与 `histogram-bin`；axis role 用于 UI 推荐，不能让 histogram/lineout 能力消失。二维 radial Gaussian 仍严格要求 spatial-x/spatial-y、共同 unit/frame，并把第三参数明确命名为 `one_over_e_radius`。原 neutral 层“Zeeman”标签对应的通用数学模型在 zlc_data 中命名为 `symmetric_lorentzian_doublet`；neutral UI 可以显示领域标签，但不得让通用公式冒充所有 Zeeman 物理选择规则。

DataTransform 后仍存活的每根 axis 必须恰好属于 fit、batch 或模型明确声明的 observation component；不能留给 solver 猜。FitModel 从首版显式声明 independent-variable arity/roles，支持当前已有的 1D 与 2D model；不能把 2D Gaussian 当作未来功能删除，也不能通过数组 rank 推断 arity。

WorkbenchFitRequest 是 workbench Command DTO，可持 app-local LiveDataBlockRef；FigureFitRequest 是 frontend figure DTO，只持 DatasetId。各 adapter 先解析为 OwnedSnapshot，再调用公开的 `bind_fit(FitSpec, snapshot.block.schema).run(snapshot)`；package-private `build_fit_problem(bound, snapshot)` 只在 `BoundFit.run` 内负责 packing。zlc_data 不定义 universal InputRef/FitRequest，也不看到 neutral live ref。artifact 保存 FitSpec 与已解析 input lineage，不保存 application request DTO。

batch cell 独立执行；预先列举的数值初始化/solver/evaluation-limit/浮点或线性代数失败只使该格产生 typed status，某格失败不破坏其它格结果。输入整体 schema/model 不兼容、transform 无效、host cancellation/deadline，以及未被列为数值失败的实现或资源异常必须中止整个 Fit Analysis，不能被 broad `except Exception` 伪装成单格 solver failure。wall-clock deadline/cancel 是一次 `BoundFit.run()` 的 hosting lifecycle，不写进 FitSpec、不产生 per-cell `TIMEOUT` 状态。FitResultBatch 不包含 runtime EventRef、LiveDataBlockRef 或 ArtifactRef；formal Analysis/figure repository adapter 在外层附加 input lineage。它不拆成多个 scalar signal，overlay 从同一个 result 与外层 lineage 派生。

FitResultBatch 是 compact solver-issued report，不是把原始坐标、observation、Jacobian 重复塞进去的 proof-carrying result。构造器/strict codec 验证状态机、静态 domain/constraint、计数、RSS、R²、covariance 的有限性/对称/PSD/fixed-row与 canonical zero；RMSE、effective schema fingerprint、coordinate source、parameter schema/unit 都由已保存的 FitSpec/AxisSpec/catalog 唯一派生，不在 payload 再存第二份真相。raw codec decode 只能得到 untrusted report；public direct 保存只接受 `FitResultRepository.execute_capture/execute_scan()` 铸造的 process-local `FitExecution`，load 在对应Capture/Scan source binding校验后铸造不可replace/pickle/直接构造的`AdmittedFitResult`。execution capability和repository均final、slotted、普通赋值不可变；每次操作复核root lease与content-store authority。outer manifest只含current format、repository id、owner编码的closed `CaptureArtifactRef | ScanArtifactRef` source与result ContentRef；CAS digest是唯一payload identity。execute_capture物化已验证的CaptureFrameSource，execute_scan委托ScanRepository exact materialization，随后都进入同一个BoundFit；load只读取source FINAL metadata/revision/schema并复算fit/batch/layout/present-count binding，不重跑solver、不从parameters反推历史执行，也不读取source data blob。repository按真实payload直接encode/decode并让分配错误显式失败；load仍必须验证manifest/result blob的实际长度、digest与codec结构。producer signature/journal仍无真实consumer而不预建。这里的信任边界是OS/process root lease排他的本地writer加CAS内容完整性；若外部主体可绕过API直接改filesystem，则没有密钥/签名的本地artifact都必须按untrusted repository处理。

FitResultBatch 是一等需求：gridplot、site grid 和任何保留 site/component axis 的 fit 都要求“一组共享 model/parameter schema + 按具名 batch axes 排列的每格结果”。`BatchLayout` 复用 PointLayout 的 RECT_C/RECT_F/EXPLICIT 映射思想；稀疏 batch 只保存实际 B 个 cell，missing coordinate 与 fit failure 是不同状态，不能强行 densify 后混成 NaN。grid 的 cell label/coordinate 由 batch_axis_specs + BatchLayout/axis coordinates 派生，不能用 list index 充当永久 identity。DatasetComponentValidity 在 build_fit_problem 时按 batch cell 切片；某个 site 无效只使对应 per_batch_status 失败，不污染其它 cell，也不允许先对 site 轴平均成一个 FitResult。`build_fit_problem` 是 fit densify/packing 的唯一 owner；若某 solver 只接受 dense layout，它必须显式 materialize mapping+validity或在 bind 时拒绝，不由 renderer/collector 猜 reshape。

BoundFit 对 batch cell 使用确定性迭代顺序，并在 packing chunk、cell 边界和 model evaluation 间检查 host cancellation/deadline；单次 solver call 只有确定性的max-evaluation数值策略。取消或 deadline 使整个 formal Analysis 失败且不提交成功 artifact；interactive stale result 按 DatasetRevisionRef 丢弃。该最小 seam 不引入 workflow engine。

### 11.8 权威 Fit Analysis 路径

权威路径固定为：

```text
FINAL CaptureArtifactRef | ScanArtifactRef
-> owner repository inspect/materialize exact DatasetRevisionRef + schema
-> zlc_data.bind_fit(FitSpec, expected schema) -> BoundFit
-> resolve FitProblem
-> BoundFit.run(exact OwnedSnapshot) exactly once
-> FitResultBatch / FitExecution
-> explicit FitResultRepository.save -> FitResultArtifactRef
```

FitSpec 必须包含 input_schema_fingerprint 与显式 fit/batch axes；发生选择/降维时 committed_transform 必须存在，identity path 可以为空。Fit只消费已经通过EOS/key/validity coverage并提交的immutable revision，验证schema fingerprint后执行相同的zlc_data transform/reduction/fit函数；它不在每个sample/update到达时把累计DataBlock重新拟合一遍。TaskConsole入口也必须先取得当前card的精确FINAL ScanArtifactRef，再进入同一路径。

若真实consumer要求无人工操作的formal Fit，必须把该FINAL artifact作为独立flat analysis Run的输入，并让Fit artifact由该Run自己的一次FinalCommit提交。Scan Run只有一个ScanRepository FinalCommit；不得在其commit后直接保存第二artifact却仍宣称原子，也不得在recovery时重跑solver。只有确有“scan与fit必须共同成功/共同恢复”的领域consumer时，才引入可持久恢复的composite commit/result；否则不实现DatasetInputSlot或AnalysisStep。

Fit返回完整FitResultBatch；下游参数引用、校准更新、scan决策或“成功物理结论”必须在自己的领域AnalysisSpec中解释numeric status、covariance/counts/RSS/R²并声明哪些具名batch必须满足何种物理policy。generic fit core只报告数值事实，不能替领域consumer发明统一acceptance。

### 11.9 Interactive Fit 路径

```text
Plot card AnalysisCommand[WorkbenchFitRequest]
-> Workbench application job host (lifecycle/cancel only)
-> resolve immutable input revision + FitProblem
-> shared application executor calls 同一个 zlc_data fit program
-> FitResultBatch
-> revision-checked overlay/ViewModel
```

interactive Fit 复用普通application executor；workbench application host只托管job生命周期与cancel，不定义queue/backlog/capacity预算，也不拥有fit算法。frontend.figure只拥有Figure DTO、View求值和overlay投影，不成为executor/lifecycle owner。同一panel的stale queued request可按latest-only语义coalesce、已运行的不可中断solver返回后按revision丢弃。它执行zlc_data `bind_fit`产生的同一个BoundFit，不创建隐藏Processor node、不发布正式measurement signal，也不占用capability signal source或view-evaluation队列。用户要让fit result进入下游权威流程，必须显式保存FitResultArtifact并由下游typed request引用；自动analysis consumer必须按§10.5建立独立flat Run，不能用隐藏AnalysisStep升级当前交互动作。

interactive 只意味着 QoS/入口不同，不降低输入 integrity：若输入只是尚未sealed的live/preview revision，可以为即时观察运行临时fit，但overlay必须带`PROVISIONAL`标记且不能保存为`FitResultArtifact`、不能成为后续authority input。source generation改变或该snapshot失效时，相关queued/running result按绑定revision丢弃并从正常overlay撤销；只有SealedDatasetArtifact或领域typed artifact的完整authority合同通过后，才允许materialize为正式派生结果。

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

Selection 是不可变语义值：AxisId、range/index/geometry、coordinate frame。Matplotlib controller 把鼠标事件转成 Selection；Qt adapter 只传递事件。neutral Processor capability/Analysis 看不到 artist、Axes 或 QWidget。

Workbench 另有瞬时 `SelectionCandidate(selection, source DatasetRevisionRef, schema_fingerprint, document/viewport revision, coordinate_resolution_record)`。它不是 zlc_data authority 类型，也不持久化成另一套 Selection；它只证明这个鼠标选择来自哪一版数据和坐标变换。Fit/Scan draft 只有在 candidate 仍与目标 snapshot/schema 匹配时，才可显式重建 CommittedTransform；不匹配则 stale/重新解析，不能把旧 ROI 套到新 camera generation 或新 viewport。

Selection 值及其坐标/geometry 语义属于 zlc_data；frontend selector controller 只把鼠标/键盘手势转换成 `SelectionChanged(Selection)`，不导入 neutral_atom 的 ControlTopic。Workbench 的 PanelController 是唯一中介：它判断该选择只是 display state、analysis candidate，还是用户明确绑定到某个 neutral Processor capability/Analysis 的 control；只有最后一种才映射为 revisioned ControlTopic command。结果携带 control revision；旧 revision 结果不覆盖新选择。关闭 panel 时 workbench 对未完成 command 发/等待 terminal ack，不能让 frontend selector 直接持有 runtime sink。

selector/board发出的commit必须保留完整`PanelInteractionOrigin`直到Workbench consumer，任何host不得只转发x-span、viewport或clim tuple。consumer先将origin与当前painted/held origin做CAS，再更新唯一display authority；render/reconfigure失败调用family owner按exact origin撤销pending，旧失败不得清掉更新的命令。range/rectangle同样保留origin，只有PanelController能把仍匹配source/schema/viewport的candidate重建为canonical `zlc_data.Selection`，再按明确用户动作送往`bind_fit`或typed ControlTopic；frontend host不决定“这个框是什么意思”。

### 11.12 Analysis 不建立 god processor

纯算法只有一层命名：`zlc_data.apply_transform`、`reduce_data`、private `build_fit_problem`、`bind_fit` 与权威 dataset Fit 的唯一公开执行入口 `BoundFit.run()`。interactive、offline/artifact以及确有consumer的formal Fit 路径都执行同一个BoundFit；neutral不定义generic AnalysisStep或任何Fit-named class。Distribution 的自动双高斯是已经出现的第二种、非权威且无 DatasetSnapshot 的真实输入：它只能调用 zlc_data 的窄 `analyze_bimodal_distribution(bin_centers, bin_counts)`，该函数内部复用同一 model catalog/cell solver 并返回最小 typed display result；不得因此公开 `model_id + arrays` 泛化入口、伪造 snapshot、复制 scipy/阈值公式或发布 FitResult。`OccupancyProcessorDefinition` 属于 neutral_atom，因为它包含逐帧领域物理语义；Calibration/ReadoutFidelity等完整dataset/artifact算法的typed request与compiler属于neutral_atom，因为它们承担领域物理判断。简单、无领域语义的逐event变换若出现真实产品用例，由消费它的capability绑定zlc_data纯函数并拥有typed prepared application；不得为此预建内部streaming operator框架或第二种公开Processor。

zlc_data 的 solver 是同步纯调用，不注册 frontend 提供的 GUI-thread guard、不读取环境变量来判断调用线程，也不持有 executor。是否在 GUI thread 之外执行是 frontend/neutral hosting adapter 的合同，并由真实入口测试证明；把线程策略注入数学 kernel 会形成隐藏全局反向依赖。

### 11.13 DataFigure

```text
FigureDocument    immutable datasets/layers/view specification
FigureSession     transient frontend interaction state
FigureEvaluator   (document, ResolvedDatasetMap) -> EvaluatedFigureData
FigureRenderer    (document, EvaluatedFigureData) -> surface/frame
FigureCodec       current schema only
FigureArchive     exact current typed NPZ（FigureDocument + source revisions + fit + display）
DataFigure        zlc_frontend.data_figure 的 public/headless Figure value
```

FigureDocument 只持有 frontend-owned DatasetId/immutable dataset descriptor、zlc_data Selection 和已解析的 dataset ViewSpec；dataset-fed ViewIntent 只在创建/编辑时作为 suggest_view 输入，不成为另一份持久状态。document-fed PULSE 不能出现在 ViewSpec/FigureDocument/codec 中。权威派生 dataset 另带 zlc_data CommittedTransform/analysis record 与 frontend FigureArtifact digest。FigureDocument 不持有 neutral runtime ref/lineage 类型；Workbench 在 materialize 时把外部 causation 转成 FigureArtifact manifest 的普通 canonical descriptors。Workbench LiveFigureBinding 维护 LiveDataBlockRef -> DatasetId 的临时映射，解析成 zlc_data DataBlock snapshot/ResolvedDatasetMap。

Interactive live path 在 per-panel latest-only view-evaluation executor 运行 FigureEvaluator：直接解析 ViewSpec 的 axis bindings/navigation policy，再执行 display transform/reduction/layer data 计算，产生带 document/input revision 与 resolution records 的 immutable EvaluatedFigureData；具体 surface ownership 见 §12.5。live与one-shot export都由frontend presentation session独占自己的Figure；Workbench job只托管worker/cancel与目标文件I/O，不拥有或构造Figure。冻结的notebook DataFigure仍属于`WORKER_RASTER_LIVE` surface contract，只采用同步one-shot hosting：构造时一次evaluate，并保留已经接收的immutable `ResolvedDatasetMap`引用以支持精确archive；不复制array、不回查repository、也不取得live authority。每次`render/export/_repr_png_`由frontend DataFigure session在调用线程这一非Qt execution context内创建、使用并释放完整OO Agg graph，调用者只取得immutable front/bytes而不拥有Figure；这不是第三种surface，也不需要第二个scheduler或预建render lane。窗口只把既有DataFigure交给共享worker/frontend session，不把one-shot调用提升成周期snapshot源。所有路径都只执行document已决定的ViewSpec，不重新猜axis；live/persisted binding的保存规则见§16.3。

`FigureArchive`的值与current-only NPZ codec只属于`zlc_frontend`，并采用`allow_pickle=False`。外层只有固定schema标识与canonical payload；payload由各领域值对象自己的canonical codec组成，必须精确保留`FigureDocument`、每个`DatasetId + DatasetRevisionRef + DatasetSchema + DataBlock`（包括完整`(R,P,*data_shape)`、coordinates/layout、`DatasetComponentValidity`）、每层`FitResultBatch`和string-keyed metadata。当前display state单独作为`DISPLAY_ONLY`保存，用于重开时还原屏幕选择；它不得修改document、CommittedTransform、FitSpec或scan authority。frontend的archive/render encoder只接typed值并返回bytes，不能接收`Path`、创建目录或写文件；`zlc_workbench.data_figure.archive_repository`与各产品export lane才以临时文件+同目录原子替换提交。decoder先完整校验canonical envelope、schema绑定、source revision与fit绑定，再构造`DataFigure`并计算payload digest作为只读身份。格式不带无消费者的数字version，不提供旧8/9-key reader、converter、shape/rank inference或兼容fallback；不符合current envelope的文件明确拒绝。

FigureViewer只负责文件选择、异步decode、Info投影和pane生命周期；显示、selector、Setting/Edit、fit与export全部委托同一个`DataFigureWindow`。一次成功Open/Load是允许的完整generation替换，旧pane只在新pane构造成功后销毁；失败不能清空旧front。加载之后的普通交互只更新稳定widget和display state，不重新解码archive、不构造全应用snapshot、不重建pane。TaskConsole保存panel时，图像文件必须逐像素转录当前front，而同stem `.npz`由这一archive owner写出exact source+document+display；两者任何一个失败都不能报告“image + data均已保存”。

Calibration report不是一个单独的普通DataFigure，因为它是一组SiteMap、per-site histogram、pooled histogram、fidelity curve与PSF grid页面；但它也绝不能拥有第二套plot style/renderer。neutral calibration owner先成对校验artifact/report，再把每一页的stored facts投影成具名axis、真实dtype与validity完整的typed Dataset/SiteMap输入；这些轴表达真实的site row/column、population、model与image坐标，不为排版伪造物理语义。可选`logic_nodes/readout/calibration/ui`叶只选择页面顺序、领域label与阈值overlay意图，然后交给frontend唯一的`PlotReportDocument -> PlotPanelContract/PlotPanelSession -> EncodedRasterDocument`链。size、DPR、FigureSpec/Divider、字体、颜色、tick、grid chrome、renderer、PNG编码与backend lifecycle全部复用普通plot surface；Calibration、notebook与Workbench都不得直接实例化Matplotlib composer或重算模型。Workbench lane只管理worker取消、窗口作业生命周期与文件I/O；Workbench Qt host原子安装frontend返回的multi-page document。

Occupancy artifact则是普通dataset view，但同一Figure一次只绑定其一个真实输出块。composition通过显式`occupancy_output="occupied"|"counts"`选择artifact已经持久化的exact snapshot，并让FigureDocument descriptor、ResolvedDatasetMap和OwnedSnapshot ref都指向该块的原始schema/revision/generation；不能把两个块堆成伪COMPONENT轴、伪造第三个DataBlock，或回退到source capture冒充occupancy lineage。occupied/counts共享repeat/point/layout/SITE domain与逐SITE DatasetComponentValidity，但仍是两个不同dtype/unit的dataset；frontend只做display投影。该冻结Figure没有Calibration SiteMap的物理XY/GridOrder证据，因此只显示canonical SITE index/facet，不宣称physical grid、paired calibration overlay或fit authority。

精确物理cell overlay不是普通单dataset Figure：它需要同一个occupancy cell、source capture cell与Calibration SiteMap三方事实，但不产生新的权威join artifact。该三方关系必须在neutral Processor/node发布边界先以唯一`DatasetCellAddress`、source ref/event digest与join digest完成same-shot atomic closure；任何一方缺失或错位则整次publication失败，composition与Workbench都不能分别取latest再拼。frontend接收该已闭合typed value，建立自包含SiteMapView；renderer看不到neutral ref/repository，neutral也不导入frontend。worker按同一IMAGE/Sites FigureSpec、Divider和artist policy合成完整不可变RGBA front，同时保留三态site facts、exact image plane、typed colormap、effective clim与ViewportTransform；Qt只present并绘制瞬时Area、locked Cross、Zoom/Pan、clim与drag overlay。矩形只能生成同一exact address/revision与ViewportTransform上的`DISPLAY_ONLY` SelectionCandidate，不能保存成Fit/Scan/Calibration输入或把当前画面升级为权威选择。

FigureDocument/FigureEvaluator/codec 属于 headless `zlc_frontend.figure`；DataFigure 值与其数据/fit/archive语义属于 `zlc_frontend.data_figure`，只有真正调用 render/export/Qt host 时才惰性进入 render/qt leaf。neutral_atom 只返回领域 Result/ArtifactRef；application/workbench projector 把它映射为 FigureDocument，neutral_atom 不导入 figure 或 DataFigure。DataFigure 只接收 FigureDocument、ResolvedDatasetMap 与按 layer id 绑定的 data-owned FitResultBatch，不主动访问 Hub、Task、Session、PulseDocument、repository、ArtifactRef 或 Device。

Figure render可以显示 PROVISIONAL revision，但必须在所有surface持续显示不可被theme/overlay隐藏的状态徽标；普通 Figure Save/Export在输入epoch未VALID时拒绝。唯一例外是用户显式选择“保存诊断快照”，生成`DIAGNOSTIC_PROVISIONAL` artifact并把水印、epoch id、revision与当前状态固化进pixels/manifest；它不能被 FigureArtifact 或任何 source-specific authoritative fit-artifact loader 当作权威输入。epoch INVALID 后，LiveFigureBinding提升lifetime token并清除或标红旧front buffer，避免之前排队的正常BoardFrame覆盖失败状态。

## 12. Workbench 与 UI

### 12.1 最小应用职责

不强制一项职责一个 Service 类。最小组件是：neutral `HostedRun/HostedProcessor/SignalDataPlane`，产品自己的窄 controller，稳定的 `PanelCard`，以及 frontend `FigureSurfaceLane + FigureSurfaceHost`。`SinglePanelHost`/`FacetedPanelHost/QtRasterBoard`只是后者内部的raster/gesture primitives；不存在另一个 `BoardModel`、`BoardController` 或 `BoardPublishPort` 抽象。

Run-owner adapter只是`RunController/RunHandle`的Qt-facing接缝：把typed command转成start/cancel，把实际变化投递为窄更新；它不拥有第二套线程、状态机、resource lease或terminal state。产品controller只编排command与presentation state，不直接调用driver。Panel topology由现有card/layout owner增删移动具体panel；普通数据、display、selector与size revision只在稳定card/host上更新，不能为了不可变领域值而重建Qt树或再引入全局Workspace状态机。

### 12.2 Command/ViewModel

View 只发送 typed Command，接收 immutable ViewModel/DataRef。Backend 不修改 Widget。

这里的 immutable ViewModel 是**变化消息**，不是定时全量状态总线。只有跨
ownership/thread 边界已经产生新 revision、新 worker 结果或新硬件状态时才发布；
没有新事实就不构造 snapshot、不唤醒 presenter。Qt 不得用固定周期调用
`controller.pump()->whole application snapshot`，也不得在 GUI thread 内同步读取 remote/device
状态。必须轮询的 active-run adapter 在非 Qt owner 中执行，只在值实际变化时投递窄的
runtime update；idle 时零轮询、零投影。worker completion 的 owner wake 是 level-triggered
“可能有结果”提示，不是一条必须逐件呈现的消息；同一 queued owner turn 前的多个 wake
必须合并，callback 执行期间新到的 completion 才补一个 replay，drain 后无变化则不发布。

Widget/editor session 拥有尚未提交的本地 draft。`textChanged`、code typing、spin/combo
临时编辑和 selector move 只更新该 draft 与局部视觉反馈；`editingFinished`、Apply、Run、
明确 selection commit 才发 typed command。command 成功后 presenter 只消费字段级 delta 及
它明确列出的 derived dependents；完整 document/ViewModel 投影只用于首次 composition、
Open/Load、Target topology/generation 替换和显式 Cancel/恢复。不可变领域 document 仍可作为
提交后的 authority，但不能因为它不可变就让一次 unit/name/value 修改遍历并回填全部
period×port 控件。稳定 key 的 Add/Remove/Reorder 分别只创建、销毁、移动对应 widget，
scalar 变化永不重建或全量 reconcile。该规则适用于 PulseGUI、TaskConsole、DeviceManager、
FigureViewer 及所有 Workbench GUI。一个机制只有在至少两个真实 consumer 具有相同语义、根因已关闭、且真实 Qt 人类事件链与机械 ratchet 都通过后才可抽成公共组件；不得先复制一个未证明的框架再等待后续修正。

Workbench骨架拥有领域中立UI Command/ViewModel/host；neutral capability core拥有领域Request、RunPlan/RunHandle/Event与完整LogicNodeDeclaration；zlc_data拥有Selection/DataBlock/Fit；frontend拥有Figure/View/interaction及全部SiteMap presentation；zlc_pulse拥有Pulse/compile/transport。普通declaration由generic projector直接变成form、Setting/Edit、signals与default views。只有启动调用形状真实不同才用headless`workbench_adapter.py`；只有声明模型与generic Figure无法表达的特殊产品面才有可选inert UI leaf，当前只有PulseScan scan-table/slot、Calibration多页报告/创建面与Occupancy exact-cell导航面有实现证据。不建立跨bounded-context `common.dto`，也不让领域schema为按钮新增字段。

TaskConsole、PulseGUI、DeviceViewer/DeviceManager的领域中立shell/controller属于`zlc_workbench`，不是`zlc_frontend`；PulseGUI与DeviceManager是独立产品，不因此成为Logic Node UI聚合目录。`zlc_frontend`保留通用Figure/render/selector、SiteMap view/Area及纯widget/presenter；它不导入neutral/pulse类型，也不接收runtime port。Workbench controller不得出现具名capability字段/输出/显示分支；PulseScan等真实特殊UI从inert leaf通过generic host窄接口取件。所有controller、start adapter与special UI factory必须显式列出窄依赖，禁止接收整个Experiment、Session、DeviceSet或返回raw object的provider：

- Workbench TaskConsole controller接收generic declaration projector、已投影的installed node entries、`RunCommandPort`、RunSnapshot reader与`open_device_viewer` action；它不接收concrete capability catalog/presenter、registry、真实node或任何平行启动权限。公开running列表只返回`RunNodeInfo`/RunSnapshot DTO，所有start/cancel/join都经唯一RunCommandPort进入RunController。
- Workbench PulseGUI controller的 editor 只接收 `PulseEditorSession`、当前`PulseTargetManifest`与 pure preview function；`PulseEditorSession` 只拥有 current `PulseDocument/path/revision/disk baseline`，因此 offline authoring/preview 无需任何设备身份。online composition 才额外注入 immutable `PulseTargetDescriptor` 与已有 notebook/application `PulseFacade`，显式 rebind 若改变文档则相对真实磁盘 baseline 保持 dirty；Target tab随descriptor切为backend manifest只读投影。这里不再额外造一层 `PulseCommandPort` wrapper：authority 已用 `PulseRunRequest -> PreparedPulseExecution(one-shot) -> RunHandle` 提供恰好所需的 run-once/hold/scan/start/cancel 面；再包装只会产生第二份状态和验证。纯compile/preview不触碰hardware；authority内部完成prepare/fire/session close/safe，且不暴露raw sequencer或可拆开的public prepare/fire/safe。standalone 的 `Remote server` 控件只把人类输入的host/port交给workbench composition factory；连接成功后窗口持有并最终关闭新建的Experiment authority，Qt仍看不到client/endpoint。已有 `exp.pulse_gui()` 则复用调用者的Experiment且窗口关闭不关闭它；两者都不能自行构造或包装旧RemoteSequencer。
- Workbench DeviceViewer controller接收`DeviceCatalogReader`和只读status DTO；需要操作者控制时只注入具名、审计化的`DeviceControlPort`，不存在`editable=True`后直接调用raw setter。
- Workbench DeviceManager controller接收一个`DeviceAdminPort`并直接调用
  `zlc_neutral_atom.installation_config`唯一的current config codec；当前只有一个磁盘后端，
  不为reader/catalog各包一层单方法interface。capability-free catalog随
  `DeviceAdminState`返回。controller可以校验候选config、显示restart-required差异并请求
  shutdown-for-restart，但不能在进程内Apply/Open/Swap physical graph，也不返回或缓存旧
  `DeviceSet`。

**DeviceManager 产品合同：** 配置模型只描述 composition root
真正能够建立的 closed variant：完整的 `virtual(seed)` installation、sequencer-only 的
`remote_pulse(host, port, transport_timeout_seconds)`，以及当前 `hardware` package声明的
remote FPGA sequencer + DCAM qCMOS + Pylon MOT camera 完整装置。它不是旧
`{role: {type, params}}` 动态类注册表，也不保存 FQCN、`$device:` 字符串引用、constructor
reflection或任意raw topology。`hardware` leaf唯一拥有其config codec、DeviceManager字段、
DevicePlan、adapter composition与E0 qualification入口；软件包可进入真实bring-up，不等于
具体设备、ROI、trigger working point或formal association已经通过资格化。

current-only config codec 使用 exact key set 与 canonical bytes；Load/New 是明确整份 generation
替换，Save 是磁盘提交，二者都不接触硬件。普通 host/port/seed/timeout 输入只更新
`DeviceConfigEditorSession` 的单 key 本地 draft；不会 serialize 整份 config、创建全窗
snapshot、调用全量 reconcile 或重建其它卡。`FluentParameterForm.read_value(key)` 是该局部
路径，`read_all/candidate()` 只在 Save/Init 边界运行。backend 切换才是 topology 变化，
只允许 keyed form reconcile 与 configured-device rows 的结构替换。

`DeviceAdminPort` 只有四个已挣得动作：读取当前 capability-free state、纯比较 candidate、
尚未发布 runtime 的 standalone process 中 initialize once、以及按
`runtime_instance_id` 请求 shutdown-for-restart。只要本进程曾成功发布 runtime，关闭后也
不构造 replacement graph；新配置由新进程重新加载。shutdown 复用唯一
`InstallationRuntime.shutdown`，尽量反向关闭全部 adapter 并返回本次 detached diagnostics；
不增加 persistent journal、quarantine、七态机或 reconnect coordinator。Qt 不执行连接/
关闭：controller 只在明确按钮后启动一个非 Qt worker，并在真实完成时投递一次窄 lifecycle
delta；idle 零轮询。

从保存的 config path 打开时，composition 必须把 exact resolved path 与该 document 的 content digest
交给同一个 `DeviceConfigEditorSession`，使普通 Save 继续对原文件做 CAS；不能把 Load 成功后的文档
降成“无磁盘来源”的新草稿。standalone DeviceManager 拥有它初始化的唯一 Experiment；TaskConsole
只借用这一个 runtime instance。TaskConsole/窗口关闭时同样走上述非 Qt shutdown 命令，成功 state delta
之后才销毁窗口与退出 launcher；失败保留可见诊断，禁止在 Qt close callback 中同步调用
`Experiment.close()` 或让 event loop 先退出而遗留后台清理。

可见面固定为main DeviceManager oracle的永久Config tab、Devices header/status dot、3:2双
FluentScrollArea、New/Load/Save/Save-as/Init、Loaded 紧凑行与常驻 status strip 为 oracle。
Config产品不暴露raw-device操作面：runtime必须在Experiment发布前完成open，运行期readback/write只属于另一个具名的`DeviceControlPort/DeviceViewer`产品闭包。
这些ports不是跨包万能Service。每个port的方法集合必须由单一UI use case挣得；它们接受/返回owner定义的immutable request/result。Workbench controller负责把neutral/pulse/installation对象投影成frontend ViewModel；frontend不复制领域DTO。Selection到neutral `ControlTopic`的转换由Workbench PanelController完成，frontend不导入neutral stream原语；设备role到BoundDevice的解析也只在composition/bind发生，GUI不保存resolver。

Workbench 大图像 ViewModel 使用 app-local LiveDataBlockRef/ReadOnlyArrayView 和 revision，不默认在每个 UI hop 再深拷贝。默认发布边界产生拥有自己内存的 immutable snapshot；若 driver 会复用 buffer，必须在该边界 copy，发布后 producer 不得再修改。该 live ref 经 LiveFigureBinding 解析，不泄漏进 frontend FigureDocument/codec。

baseline 的 `LiveFigureBinding.resolve(DatasetRevisionRef, SnapshotQuery) -> OwnedSnapshot` 只 materialize 当前 ViewSpec 所需 axis slice/chunks，不默认复制完整累计 DataBlock，也不返回 mutable builder alias。SnapshotQuery 只描述所需 slice/revision，显示 reduction 仍由 FigureEvaluator 拥有。当前没有自定义 lease、pin、borrow token 或零拷贝生命周期协议：Python 普通所有权已经足够，额外协议只会把每条 discard/close 路径变成新的正确性负担。

真实大帧profile已经证明“每层重新freeze”会让CameraFrameRecord→Value/DataBlock→EvaluatedImage重复复制，并在任一component invalid时把整张整数图升成float64。窄优化因此固定为：可复用driver ring到bytes-backed immutable snapshot只做一次必要复制；之后只有ultimate backing确为不可变bytes且dtype/shape一致的ndarray slice/transpose/stride view才可直接复用，普通`writeable=False`、owning ndarray、mutable base或外部mapping仍必须复制。all-valid validity使用同一immutable单字节broadcast，不展开整张bool plane；renderer对可见slice与显示分辨率做mask-aware decimation，只允许显示尺寸的reduce结果升浮点，禁止整源uint8/uint16图转float64。该优化仍是普通Python对象所有权，不引入lease、borrow token、预算或可变ring别名。

### 12.3 Setting 与 Edit

统一的是四条互不混名的边界，而不是一个万能SettingsEngine：typed Request/Config owner的字段语义、§4.3.1的headless FormSpec与closed Qt handler、Workbench EditorSession的`base_revision + draft/apply/cancel`、以及最终领域validator/typed command。普通scalar、bool、enum、bounded number、unit value、简单path和静态list可由FormSpec自动生成；Setting与Edit必须消费同一个FormSpec和同一个committed state，但拥有各自widget instance与开始编辑时的base revision。

Definition只标识catalog项及其request/config schema identity；它不复制字段默认值，也不携带FormSpec、Qt hint或projector callback。字段的key、stable operator label、type、required、default、unit、range、static choices和semantic description属于typed Request/Config schema owner。Workbench为已知use case通过普通import调用明确的schema projector，把这些事实逐项投影成`zlc_frontend.form.FormSpec`，并只在必要时增加group/order/widget/layout/file-dialog/dynamic-enable等presentation信息；它不能为了生成普通表单再次逐字段填写label/default/range。frontend不认识Definition，领域包也不导入frontend；schema-id不用于动态查找builder。

以下继续使用显式presenter/view，不强行自动生成：Pulse editor与PulseDocument/API table、ROI/selector、fit axis/batch/reduction、CalibrationArtifactRef与calibration workflow、device connection、authoritative DataTransform、resource conflict和安全确认。显式presenter内部的普通叶字段仍优先取公共handler，但复杂对象的commit、rollback、validation和lifecycle不能交给generic form。`FluentParameterForm`不递归、不持久化、不做domain validation，也不拥有Apply按钮。

Apply必须先由field handler无损读出完整draft，再构造typed Request/Intent并通过领域validator，最后检查`base_revision == current_revision`后原子提交；后台或其它editor已更新配置时返回typed EditConflict，不能last-write-wins。Cancel只丢弃draft并从当前exact snapshot做full-state回填。UI的enable/disable只是提示，hidden/disabled字段仍必须在populate/reset中被覆盖；RunPlan.bind/preflight继续执行同一个权威validator，不能信任界面已经挡住非法输入。refresh或programmatic populate必须exception-safe block signals，既不能重新触发Apply，也不能吞掉非法saved value后保留旧widget状态。

axis 编辑器读取 ViewSuggestion，不让 image、rolling、histogram 和 fit 各自实现shape猜测。控件inventory只包含当前合同仍交给用户决定、且能按具名`AxisId`无损round-trip的有限`FixedIndex`选择；resolver已经决定的display/x/sample/batch/reduced/facet事实只进入紧凑只读summary，不得伪装成disabled ComboBox/SpinBox。`RESOLVED`且没有可编辑choice时整个axis editor隐藏；`NEEDS_INPUT`只建立消除歧义所需的最少具名行。不得无来源地出现`Reduce`、`ROI X`、`ROI Y`等只读假字段；ROI/Selection由统一selector presenter author，权威reduction/transform由自己的显式presenter author。repeat与Grid facet使用各自已有的显式控件，不在普通axis editor复制第二套。

TaskConsole 不建立平行 signal graph 或 workflow editor：composition 显式组装当前 Task、Measurement 与 Processor Definition，并要求每项都被 CatalogView 投影。所有已声明的 Measurement 都在 Add catalog 中可见；安装能力不是 Definition discovery filter，但具体 binding choice 必须由 installation 的已发布 capability 投影，不能列出一项必然无法 bind 的 role。Calibration 因此只列具备完整 sitemap profile 的 camera role；Fidelity/grey-molasses 这类 Definition 本身仍可见，并在缺少整体执行能力时给出具名拒绝。Temperature release-recapture 走上述真实 autonomous exact pipeline；virtual installation 已发布 readout-duration 所需的 camera exposure configure/readback/rearm Port，real adapter 若未发布同一能力则只在用户真正点击 Start 后以具体 capability reason 拒绝；grey-molasses 同理只在缺少同步 RF table Port 时拒绝。Definition 不能消失、构造时崩溃或改写物理实验。readout-duration fidelity 的 main 物理定义是在每个点同时改变 pulse readout-light window 与 qCMOS integration time；固定 20 ms camera integration 而只扫 probe 高电平会改变背景积分与 readout physical context，是另一个实验，禁止拿它冒充可运行实现。readout-duration execution 由 composition-owned camera Port 在每个 API point 的 arm 前 configure exposure 并读取硬件 applied value，随后用 point-group exact coordinator arm camera、执行该点已冻结的 `STATIC_ONCE` API segment/shot group、完整排空并形成该组 terminal，再进入下一点；qCMOS adapter 必须通过这套 configure/readback/rearm contract kit。这里 host 只位于明确获准的 API-slot segment boundary，segment 内 trigger/readout edge 仍由 FPGA 与 camera hardware 决定，因此它是“API_SLOT 无法无缝更新”的既有例外，不是可推广到 SCAN_SLOT 的逐点 host stepping；整 run 单 arm 的 generic API scan 不具备这项能力。Definition 只提供稳定 key/schema metadata；动态 camera schema、Processor output/algorithm/lineage 与 adapter deadlines 在本次 bind 的 capability-owned prepared application中冻结，所有 Measurement Request/FormSpec 都不携通用 timeout。

Setting 与 Edit 使用同一个 `ScanIntentForm` 类和同一个 card-owned `ScanEditorSession`，但各自持有开始编辑时的 `base_revision`；Apply 先构造完整 `TaskConsoleScanIntent` 与 public scan request，再要求 revision 未过期且现有 panel 已完全 stopped/idle，最后原子替换同一个 `ScanPanelController` application。它不另建 Run 状态机、renderer 或第二个 panel owner。request-level/owner codec 约束在 Apply 时完成；依赖真实 calibration、device capability 与输出 schema 的完整 bind/preflight 仍只在 Start 的既有 worker 路径完成，失败保持 NOT FINAL，不能为了“Apply 即全验证”在 GUI 线程读 artifact 或接触硬件。Cancel 只恢复当前 applied revision；过期 Setting/Edit 返回 typed `ScanEditConflict`，不做 last-write-wins。`populate/reset` 必须是覆盖所有intent-owned控件的全状态函数，包括当前disabled/hidden的calibration/model、roles、trigger与SITE字段；occupancy→direct Apply/Load或首次Cancel后不得让已取消的隐藏值在切回occupancy时复活。

可保存 intent 只含稳定 DefinitionKey、owner `PulseDocument`、按声明顺序冻结的全部 whole-run API 常量、角色/trigger、显式 CalibrationArtifactRef、`model_kind=None`（跟随 immutable artifact default）或显式模型、权威 `DataTransformSpec`、独立display-only `ScanDisplayIntent`与deadline。保存/加载委托各 owner 的 current codec，严格拒绝额外字段、旧 schema 和非 canonical bytes；不保存 DeviceRef、BoundDevice、RunHandle、reader/front buffer、provisional revision 或可执行node对象。加载只能在 stopped/idle 时重配并清除旧 FINAL。任何非空权威 transform 必须在 Setting/Edit 中逐 operation、逐 AxisId 显示 Select/Reduce/missing/validity/min-count 语义，并提供明确清除 user-authored transform 的动作；不能把已保存的有损 authority 藏在 form 私有字段里。SITE auto/batch/select 只改变可见 View；底层 `(R,P,*data_shape)`、具名 axes、DatasetComponentValidity 与权威 ScanOutputContract 完全不变。UI 必须把“Calibration default”保留为默认引用语义，不能打开编辑器后静默改成 BOX 或其它显式模型。

### 12.4 Qt 线程规则

- QObject affinity 由创建线程或 `moveToThread` 决定；
- 不直接跨线程调用 worker method 并假设它会在 worker thread 执行；
- queued signal 只传 immutable DTO、只读数组或明确 copy；
- 禁止跨线程传 QWidget、Figure、Canvas、artist、driver/session handle；Figure/Agg graph 必须由同一 frontend presentation session 在其执行线程创建、使用和释放，不存在共享 Figure 的 handoff 例外；
- 所有结果带 run_id/revision；
- window close、cancel 或新 revision 后丢弃旧 result；
- 禁止 BlockingQueuedConnection、嵌套 processEvents/QEventLoop 等待 worker；
- shutdown 后 queued result 仍可能到达，receiver 必须有 shutting-down gate。

### 12.5 Render ownership

产品只保留两种真正有消费者的 surface：

1. `WORKER_RASTER_LIVE`：TaskConsole、Pulse preview、DataFigure/Edit 等交互面。frontend `PlotPanelSession/DataFigure/FitGrid` 在worker调用中永久拥有自己的`PanelComposer`/`SiteMapComposer`与Agg Figure/artist；Workbench只托管worker生命周期和cancel。Qt 只接收 immutable raster、typed display payload 和 viewport geometry，并绘制 Area、locked Cross、drag handles 等瞬时 overlay。
2. `FROZEN_RASTER`：Calibration report、静态导出与没有数据坐标交互的 encoded page。frontend report/DataFigure owner从已冻结的typed report/document合成 immutable bytes/page，Workbench只负责输入/目标I/O与cancel，Qt 的 `FrozenRasterView` 只负责原生像素显示与 Fluent scroll。命名尺寸的report在screen DPR变化时只重做presentation render，绝不重跑calibration、fit、threshold或artifact load；不受命名surface约束的外来encoded artifact保持原生像素并用scroll承载，不冒充可伸缩typed surface。

baseline 不保留无消费者的 GUI-thread Matplotlib artist 模式。若未来一个真实低成本产品必须直接操作 artist，必须先证明它比现有 worker-raster路径更简单且不与Qt/worker共享Figure；不能仅为了架构对称预建第三种surface。

TaskConsole 的正式链路以当前代码类型为准：

```text
SignalDataPlane -> immutable SignalFront/SignalValue
  + ConsolePresentationIndex exact-revision sidecar
  -> PanelCard.freeze_render_request()
  -> FigureSurfaceRenderRequest
  -> FigureSurfaceLane (one worker; owns Agg/composer)
  -> frontend PlotPanelSession / DataFigure / FitGrid / report owner
  -> FigureSurfaceCompletion + FigureSurfaceContext
  -> PanelCard.accept_render_result() revision checks
  -> FigureSurfaceHost atomic promote
  -> internal SinglePanelHost/FacetedPanelHost/QtRasterBoard
```

`FigureSurfaceRenderRequest`只冻结一个presentation job所需的source ref、display revision、view、fit identity、logical size与pixel ratio；它不是全应用snapshot，也不复制整个runtime状态。`FigureSurfaceLane`对每个`surface_id`只保留尚未开始的最新request；同一source topology由frontend session复用自己的composer与blit状态，source key变化才替换该session的composer。coalesce只影响显示工作，绝不能跳过producer stream、改变Dataset revision或变成软件内存/pending/backlog预算。

高层呈现决定不属于任何Workbench render lane。`PlotPanelContract/PlotPanelSession`是普通panel与Calibration report的唯一kind/view/size/DPR/style/composer owner；DataFigure与FitGrid各自的纯frontend presentation owner负责分类、display state、panel identity、join stamp、viewport/color range、payload、整板compose与编码。Workbench lane只提交worker job、传播cancel、加载repository输入，并把frontend返回的immutable front/bytes原子写入目标路径；它不得直接实例化Agg renderer、构造`BoardFrame/PanelFrame`、选择colormap/facet/grid columns或重写export codec。frontend纯函数只接收data/frontend值与`check_cancelled()`回调，不接收Executor、Event、Path、neutral ArtifactRef或Qt对象。

运行时plot、report、SiteMap、Area/Cross/Fit派生输出共同使用唯一`FigureSource(OwnedSnapshot, SiteMapPresentation|None)`；若带SiteMap，其输入revision必须与snapshot exact相等。一次compose的运行因果只使用唯一`PanelProvenance(run_id, epoch_id, join_digest)`，且join digest在构造边界验证。不存在按调用方改名的`PlotPanelInput/FigureOutputSource`或另一份provenance wrapper；Calibration report与TaskConsole必须原样传递这两个frontend值，不能各自重建一个较弱验证的同义DTO。面向notebook的`FrozenFigureSource`是“可先只有schema/ref、随后构造FigureDocument”的应用请求，不进入render/derived-output链，也不能替代`FigureSource`。

Calibration UI leaf只把领域结果投影成frontend-owned `SiteMapPresentation`或`PlotReportDocument`；它不得实例化renderer、composer、FigureSpec/Divider或style。TaskConsole中的Calibration FINAL `site_map`与Calibration overview/report原样消费frontend `FigureSource + SiteMapPresentation + PlotPanelContract`，最终都由`PlotPanelSession` compose。给定相同typed source、ViewSpec/display、labels、named size与pixel ratio，两条产品链必须得到同一raster/pixel contract；report默认只可提高terminal export pixel ratio，不能改变logical size、font、margin、artist policy或另建Calibration style。

Neutral `SignalDataPlane`的`SignalValue`与Workbench `ConsolePresentationIndex`的frontend sidecar保持不同owner，但一次FINAL或Figure-output composition publication必须先分别得到完全验证的prepared replacement；任一侧prepare失败时两侧均零改变。两项replacement-only commit发生在同一Qt owner turn，期间禁止callback、freeze、topology projection或其它可失败工作；两项都提交后才`freeze()`并以exact `signal_revision_identity`执行`reconcile_visible(front.signals)`。presentation route与data plane同样只保留`visible + 至多一个candidate`：N+1 sidecar不能覆盖仍可见的N；只有N+1 exact value进入consumer front时才提升candidate；withdraw先撤candidate，但N的visible sidecar保留到N真正退出front。需要presentation的Figure/SiteMap value若缺少或冲突sidecar，必须在publication prepare阶段整事务失败，不能等signal topology读取时才抛错。

Logic producer generation被替换或row被移除时不能只detach source slot。data-plane owner先冻结transitive causal retirement：从该generation的declared/live/FINAL routes出发，递归包含所有active Processor dependents，以及source edge或captured ancestry与集合相交的Processor/Figure-derived publication，并列出完整retired signal names。Workbench先撤销受影响surface的queued render/Fit completion，再通过PanelCard唯一`retire_source_generation()`撤销pending interaction、Fit pin/result/busy、selector commits与旧live raster；这样该动作随后为仍打开的frozen Edit surface发出的“清除旧Fit overlay”请求不会被第二次forget。composition同时准备完整retired names的presentation withdrawal，随后在无观察者插入的owner transaction中提交presentation/data retirement、关闭依赖processor/slot，最后freeze+reconcile visible。旧source component、candidate、derived publication、Fit参数、sidecar或迟到completion都不得重新注入旧generation；新generation复用同一用户namespace前必须完成该闭包。

worker completion到Qt后必须同时验证：surface仍存在、source的block/generation/schema仍相同、request revision未被更新结果取代、display revision匹配、Fit仍绑定同一source ref。`FigureSurfaceHost`只在同一次Qt owner事务中提升匹配的raster、logical size、DPR与typed context；同一generation的Area/Cross authority随后按新data revision重新materialize，不因live更新失效。失败结果只留下detached string diagnostic，不长期持有traceback、Dataset或Agg graph。关闭时先停止接收与撤销Qt发布：Figure surface lane消费已经开始的compose Future，随后在同一串行worker对全部frontend session执行try-all release；Fit lane取消尚未开始与正在求解的request，并等待active Future真正返回、清除request/snapshot引用。两条lane各自的owner回执都到达Qt后，TaskConsole才关闭SignalDataPlane、application resource和QWidget；等待期间closeEvent保持ignored且GUI event loop继续响应。

交互render不是另一套全局snapshot或无界事件队列。每个panel只有一个render-paced mailbox：一个不可被覆盖的exact in-flight answer，加一个可被后续pointer motion原位替换的latest semantic draft。viewport、threshold与clim共用这条规则；同一panel不同交互family必须串行，不能各自持有一个in-flight槽。没有in-flight时，owner从当前painted/held front冻结完整`PanelInteractionOrigin`并真正分配presentation revision；已有in-flight时只替换latest draft，不提前分配revision。exact answer到达后先原子安装真实front并推进hold，再从这个新exact origin提交latest draft；失败也只按捕获的exact token做CAS清理，并从仍可见的front重新提交尚存latest intent，旧失败不能清掉同步重入产生的新请求。

一次交互始终钉在operator实际看到的immutable input/source ref上。producer与其它panel可以继续前进，同panel的新live candidate也可被记为latest，但在该交互mailbox清空前不能替换其input或被拼进answer；完成后普通live presentation再追到最新合法candidate。mouse release必须用release事件自身坐标重新走与motion相同的纯几何函数，不能假设Qt一定先发最后一次move。这样连续拖动既能按renderer实际帧率看到中间真raster，也不会用假stretch、丢最后坐标或把另一shot的数据嫁接进当前selector。

`BoardFrame`是frontend的不可变**显示事务**，不是新的runtime controller。单panel host把一个已验证的`PanelFrame`包装为单panel `BoardFrame`；faceted/grid host一次呈现完整faceted结果。多个领域输出只有在neutral producer已用同一input EventRef/join digest形成typed atomic transaction时才可称same-shot；把几个独立panel在同一个Qt turn里paint出来，永远不能创造物理因果关系。也不存在文档中另一个`BoardModel/BoardController/BoardPublishPort`层。

Figure的完整静态内容——axes、title、xlabel/ylabel、tick、colorbar、distribution、site rings、grid chrome与data artist——只由同一个frontend FigureSpec/Divider/render_style/renderer owner绘制。Calibration、TaskConsole、DataFigure与FigureViewer对相同typed source、ViewSpec、size和display state必须得到相同像素合同；只有用户明确要求复刻某项交互时才按需对照`main`。Qt不得近似重画这些元素；Qt overlay只画必须跟随鼠标即时变化的selection几何。所有plot kind、Pulse preview、TaskConsole、DataFigure与FigureViewer复用同一selector geometry/host，而不是各自实现第二套Area/Cross/zoom/pan。

交互遵循“即时overlay + 完整新raster”而不是假缩放：

- wheel/pan先基于当前已画front携带的viewport transform计算candidate；
- Qt可立即显示selector/gesture overlay，但不stretch旧raster冒充新图；
- owner提交新的display revision，worker用持久artist/blit路径生成匹配viewport的完整raster；
- 只有匹配candidate revision的新front才能替换旧front；旧结果丢弃；
- pointer press期间只hold目标panel的一份immutable已画front和exact origin；producer与其它panel继续前进，但目标panel在mailbox清空前保持同一input，之后一次追到最新合法candidate，不回放历史队列；
- 不发布pointer-motion hover数据。

Area、Cross、clim、threshold、zoom/pan的hit test和坐标换算只使用同一front payload中的typed viewport/axes geometry。手势结束产生带exact `PanelInteractionOrigin` 的intent；owner先做current-front CAS，再修改唯一display/selection state。任何source、layout、presentation、axis、geometry或revision变化都会使旧gesture失效。normalised rectangle在浮点运算中可暂时出现机器精度越界时，应由唯一rectangle/viewport纯函数在语义边界归一化；paint函数不另写补丁规则。

Image viewport是连续物理坐标窗口，不以data extent或一个sample cell作为zoom/pan硬上限。Main风格的equal-aspect正方形data box、zoom-out与pan都可显示source外的axes background，亚像素zoom也合法；Cross/fit overlay按同一display extent映射。只有Area/selection进入数据authority时，唯一`ImageViewportTransform`才把visible rectangle与完整source raster求交，完全落在background的drag不产生Selection。禁止在Qt paint、selector或测试中重新引入clip-to-data、minimum-one-cell或忽略square padding的第二套公式。

Curve与Histogram的一维numeric viewport共同委托frontend的纯`numeric_viewport`函数完成normalized widget坐标、data x、zoom和pan换算；各display模块只验证自己的typed state，不复制同一套边界公式。Image保留自己明确的二维`ImageViewportTransform`，因为其equal-aspect、origin、像素cell edge和双轴约束不是一维公式的别名。不得为“统一”建立继承层或通用viewport状态机。

size由统一的panel-size token解析为logical pixels，再乘当前screen pixel ratio生成worker raster；host在同一次present transaction中设置logical尺寸并交换front。`size_name`是可保存、可复现的presentation intent，screen DPR是当前Qt surface的runtime事实，禁止写入Figure archive、Calibration artifact或其它持久合同。所有Qt raster host复用唯一`RasterPixelRatioObserver`观察真实顶层window/screen；变化时先递增surface revision并立即清空旧front，再把不可变`PanelSurfaceGeometry`冻结进worker request，只有同revision结果可present。每个window owner在真正提交异步关闭时必须显式、幂等地`detach()`该observer，撤销host event filter、QWindow/QScreen signal与callback；不能依赖Python GC或QApplication退出顺序清理native连接。普通DataFigure/DataFigureWindow无显式或saved `size_name`时使用唯一`DEFAULT_PANEL_SIZE`（当前`2x2`）；Grid在构造时已经拥有exact schema+view才使用`optimal_grid_size_for_view()`作一次初始建议，用户选择或archive值始终权威，异步factory尚无schema/view时先用普通默认且首帧不得跳尺寸；Pulse使用自己的topology策略。不存在“固定800×520 raster + 任意扩张board”的匿名surface；FitGrid的整板logical geometry由frontend按同一panel geometry和grid topology产出，Workbench不得重算。不得先让Qt拉伸旧图再等新图，也不得让card、board和renderer各维护一份size表。DPI只通过统一pixel-ratio/typography token进入composer，不把`dpi=300`当作跨窗口的隐式全局style。

`SinglePanelHost`是单panel identity、present与selector-family的唯一组合widget；`FacetedPanelHost`负责grid/facet overview与focus，但复用同一个`QtRasterBoard`和交互协议。`FrozenRasterView`没有数据坐标、selector、Fit或bitmap zoom；需要这些能力的页面必须进入typed panel host，不能因为已经有PNG而降级。

导出总是从冻结的FigureDocument/DataBlock revision与同一renderer重新合成，不能把任意屏幕texture当权威数据。若用户要求保存“当前所见”，先冻结当前已呈现source/display identity，再分别写图像与frontend archive；两项不能以不同revision成功后拼成一个结果。

### 12.6 UI 可见 Fit

TaskConsole panel 的 Setting 与 Edit 嵌入同一个 Figure-owned `FitAuthoringPane`；DataFigure/FigureViewer 使用同一个 pane 并把 tab 命名为 `Fit`。产品中不存在 `Add Analysis`、`Analyze`、独立拟合窗口或点击后再弹出的 DataFigure。TaskConsole 的结果只作为当前 panel 的瞬时 overlay 并发布 `fit.<parameter>`；DataFigure 的显式 `Save Fit` 才进入其 artifact/archive 生命周期。

pane 只有 model combo、一个 `args` 文本框和 `Fit/Clear`。空文本使用 model owner 的自动初始化与边界；文本只接受逗号分隔的有限数值 keyword，例如 `center=50, sigma_lower=0, sigma_upper=8`。`name=value` 固定参数，`name_initial/name_lower/name_upper`设置solver constraint。parser 只解析受限 AST numeric literal，拒绝 positional argument、表达式、call、`**`、未知/重复参数和 NaN/Inf，绝不 `eval`。model metadata、参数名、默认 seed/bounds、FitSpec 和 solver 仍由唯一 zlc_data catalog/binder拥有，Qt 不复制参数表。

TaskConsole 点击 Fit 时从当前已经接受并画出的 exact `OwnedSnapshot`、具名显示轴和仍有效的 SelectionCandidate 冻结权威 draft；没有可证明的轴完整source时按钮不可执行。它不反推 artifact、不启动第二个窗口、不保存本地archive。DataFigure/FigureViewer 对 FINAL `CaptureArtifactRef | ScanArtifactRef` 或 current figure archive 走自身既有的持久Fit路径。Grid overview/focus raster不能冒充single-panel authority；只有resolver证明的完整具名batch source才允许Fit。

live TaskConsole 提交 Fit 后只把该 panel 的命令surface固定在提交时的 exact source；`SignalDataPlane` 与其candidate仍继续前进，不建立历史队列。成功结果及overlay持续绑定该source，直到用户Clear；source ancestry在worker入队前捕获失败、worker异常、取消、重绑、移除或shutdown等所有非成功terminal都必须携带原`FigureFitRequest`身份进入同一个card清理路径，原子清除pending/busy、释放surface pin并直接呈现最新合法candidate，不能只写status后返回。producer generation retirement是另一条同步terminal：即使solver已成功或pending request被lane直接移除，也必须由PanelCard source-retirement owner立即提升request revision、取消lane、清Fit spec/result/overlay/parameter publication与所有pane busy，并清除旧live front；迟到solver completion因request identity失效而拒绝，同名replacement无需用户手动Clear即可显示。TaskConsole的Clear同时撤销瞬时draft/result/overlay与全部`fit.<parameter>`，并恢复latest live；它不删除DataFigure/FigureViewer已经显式保存的artifact identity。

Histogram 的bars、bin count、x viewport与rolling侧分布首先都是display-only。用户点击Fit时，frontend才从exact Figure的具名SAMPLE/SELECTED/REDUCED轴和完整样本范围冻结唯一terminal `HistogramSpec(sample_axis_ids, bin_edges)`，随后仍走同一个`BoundFit -> FitResultBatch -> overlay`；x zoom不得改变这组权威bin edges。普通Histogram/Grid renderer不得自行求解Gaussian、推断阈值或发布参数；rolling monitor允许复用同一closed model initializer绘制不发布结果的display-only单Gaussian诊断，但它绝不是Fit authority。

耐久路径中，card只用自己声明的数据名去匹配logic row的`declared_outputs`；必须恰好匹配一个row，且该row最近一次Run已经`SUCCEEDED`并从`RunHandle.result()`取得真实Capture/Scan ref，才选择artifact authority。除source ref外，当前可见neutral `SignalValue.run_id`还必须与该`RunHandle.run_id`全等；这样producer已提交新artifact但worker仍画旧front的短窗口不会把新结果套在旧图上。两个row声明同名输出时视为歧义，不能按row顺序、最近时间或当前显示值猜来源。Run start先撤销旧result，terminal snapshot与owner thread退出之间继续非阻塞轮询，直到result已取得才把row转为done；失败、取消、Load、移除row或重绑card都立即撤销该耐久入口。普通panel路径不反推artifact：`frozen_data_figure()`直接比较当前`FigureSurfaceHost` typed context中的exact evaluated input与拟返回DataFigure snapshot；replacement微窗内任一不等都fail closed，不能把旧图/selector与新snapshot拼在一起。正式Fit面只有本节规定的draft、BoundFit与artifact/archive路径，不得并存第二套analysis control/request/processor/region-signal链。

TaskConsole的Qt timer也不是snapshot生产者：neutral `SignalDataPlane.freeze()`只有producer revision或membership真实变化时才构造新的immutable per-producer front；无变化必须返回同一对象。一个present cycle可以携带多个producer各自的latest revision，但绝不因此声称它们属于同一物理shot；只有同一producer transaction内经明确EventRef验证的raw/derived值，或数据面按显式join key产生的结果，才可共享coherence group。卡片只消费这个front及`ConsolePresentationIndex`中exact-revision sidecar，不能因为timer tick、Fit控件状态同步或row状态轮询制造全板数据快照。每个card只在Qt owner上冻结一个窄、不可变`FigureSurfaceRenderRequest`；唯一串行`FigureSurfaceLane`调用frontend session，Qt只接受匹配request revision的结果并由`FigureSurfaceHost`原子present。Setting/Edit文本留在稳定widget中，`editingFinished`/Apply才提交；Edit复用同一surface lane/host，不另建composer或第二次evaluate/rasterize。

TaskConsole不保存另一份LogicNode事实或运行列表。`ConsoleNodeSpec`直接持有领域完整`LogicNodeDeclaration`，只附加机械Form投影、installed binder与真正特殊的UI factory；output、artifact、dynamic choice、path hint与default view原对象委托，不重包为Console专用领域DTO。活动执行的唯一事实来自neutral `HostedRun | HostedProcessor`与其typed snapshot；Workbench只保存row/widget identity和retained presentation。signal topology只由显式add/remove/start/terminal/card-binding边界重算；“unbound”就是key不存在，不能保留一个从未产生的魔法字符串分支。

这不是把formal能力降格为GUI：当前真实consumer包括人对已提交Capture/Scan artifact做可复现分析，也包括人对已画出的单panel immutable dataset显式分析并另存完整DataFigure archive。自动/headless preset或下游consumer出现时按§10.5/§11.8另建以FINAL artifact为输入的flat analysis Run；不能为了一个泛化菜单名称预建DatasetInputSlot、generic Analysis registry或修改Scan的单FinalCommit语义。

Figure Fit composition固定提供prepare/execute/result/save/reload五个窄能力；artifact binding冻结exact `CaptureArtifactRef | ScanArtifactRef`与source schema inspector，本地binding冻结DataFigure已经拥有的唯一`OwnedSnapshot`。DataFigure不取得repository或neutral `FitExecution`保存能力，TaskConsole也不获得另一套执行接口。两条binding都调用`zlc_data`唯一的`suggest_fit_draft/bind_fit/BoundFit.run`；`FitDraftAuthority`仍是唯一未保存opaque execution owner，Qt只持`FitDraftResult`。执行与overlay raster分别使用既有窄lane，不建立async engine；revision/CAS规则保证旧prepare/solver/overlay/reload completion不能覆盖更新后的selector、约束或viewport。

UI 用紧凑summary/tooltip明确展示 input、fit axes、batch axes、selection authority、model、status、result、save identity 和 overlay；initial/bounds/fixed 只在同一个 `args` 文本中可逆author，不展开成参数表。`suggest_fit_draft(schema, model, fit_axis_ids, selection)`只从schema声明的轴role与当前typed panel的具名显示轴生成候选：repeat及所有非fit point/data轴默认完整保留为batch；显示层为得到单panel可以带标签选择一个真实physical cell，但这个display selection绝不能进入FitSpec。不存在按rank/singleton猜role、`flatten`、取第0个权威batch cell或对trailing axes `nanmean`。

用户看到的普通 `Fit` 动作就是提交当前权威draft，不再弹第二个确认框；但未解析真实fit axis时按钮不可执行。1D range与2D box由Qt selector产生`SelectionCandidate`，经完整exact `PanelInteractionOrigin`转换成只含fit axis的`Selection`并预填同一draft：前者只选择当前CURVE x轴，后者显式绑定`SPATIAL_X/SPATIAL_Y`，其它repeat/point/site/data轴继续为batch。候选只在其完整origin仍与visible front全等时有效，不能跨document/frame revision泄漏。已经保存的DataFigure archive初次打开且没有新authoring时精确恢复原FitSpec，包括任意已提交transform，而不是把它重猜成selection-only spec；若原transform恰好是单个Selection，它作为初始candidate显示，用户点击`Use full range`则明确移除它并重新建议full-range spec，不能静默保留旧range。TaskConsole Clear遵守上文的瞬时publication撤销；DataFigure/FigureViewer清除当前未保存draft/overlay时不删除已经保存的artifact identity，二者都保留当前selector candidate。zoom/pan/relim/cmap永远是display-only，不能复制进CommittedTransform。

CURVE overlay按batch逐series求值并在每次物化前检查取消；replace/clear必须及时释放旧prediction引用。IMAGE radial overlay只携exact batch storage identity、center/radius及viewport映射，不生成第二张predicted image；失败/NOT_PRESENT只显示诊断，不伪造曲线或圆环。Save成功identity在线性化点先被接受：artifact路径从exact FitResultArtifactRef reload；本地路径用`DataFigure.with_fit_results`合并目标layer、写archive并从exact LoadedFigureArchive reload。后续decode/render失败或Close都不能吞identity，Save中Close明确defer。headless notebook的`fit.save()`仍保持短cell。

saved-fit archive GridPlot不会因一页只有一个panel就暗选第一格：Overview、hole或未聚焦时Refit禁用，只有用户聚焦一个真实`FitGridCell`后才启用`Fit/Refit`。该cell Selection只决定新DataFigure的display panel；重新author的FitSpec从exact saved ref恢复原model、constraints、numeric policy与range-preserving CommittedTransform，并绑定原Capture/Scan source。打开Fit tab不求解，只有随后明确点击Fit才创建新draft；原artifact不可变。当前权威transform边界仍是一个range-preserving Selection，unsupported transform显式拒绝，不能用显示fallback解释。

### 12.7 Shutdown

关闭窗口、断开连接或切换config/device/virtual-real都使用同一change-driven关闭流程；Qt只发命令并消费结果，不轮询已关闭facade：

```text
reject new commands for the old runtime
-> immediately detach old facade/descriptor from UI authority
-> terminal-ack pending ControlTopic revisions
-> stop producers/subscriptions and reject new view/fit jobs
-> cancel queued latest-only work; drain in-flight evaluation/raster work
-> release interactive frontend FigureSessions/renderers on their owning non-Qt presentation execution threads
-> RunController cancel active RunHandles and join owner/interrupt threads
-> each domain session performs exactly one close_session
-> DeviceBroker invalidates bindings after those sessions terminate
-> close remaining connection-owned adapters in reverse dependency order
-> stop owner lanes and release backend physical-owner proof
-> old InstallationRuntime -> CLOSED (or return close diagnostics)
-> destroy Qt views, or compose a fresh runtime on a later connect command
```

上述顺序是ownership边界，不由GUI线程是否响应来证明。ResourceArbiter、DeviceBroker与RunController在被composition绑定后，child public `shutdown()`必须拒绝；只有InstallationRuntime私有lifecycle capability能推进teardown。不得持composition lifecycle lock跨SDK close或其它物理I/O，等待者必须能按自己的monotonic deadline返回。

正常关闭不执行“generic SAFE + domain close + generic verify”三遍动作。每个领域`close_session`是其硬件stop/SAFE与终态readback的唯一owner；取消中的out-of-band interrupt只用于打断in-flight调用，随后仍由同一个`close_session`收口。adapter close失败时返回本次close诊断，不把runtime写成跨请求sticky失败，也不建立进程级reconnect-required门禁。旧facade已经被摘除，因此timer/controller不能继续调用已关闭Experiment。

新连接不复用旧runtime、binding、capability或软件判定；它重新取得physical-owner proof，重新读取live identity，重新执行当前硬件SAFE初始化并生成新的runtime/binding id。若这些实时步骤失败，本次新连接拒绝。queued result通过application lifetime token + run/panel revision双重检查后丢弃，不能更新已销毁或id被复用的新panel。
## 13. Calibration

Calibration 是 `zlc_neutral_atom.logic_nodes.readout.calibration` 的内建 feature，不使用 plugin、entry point、包扫描或动态 registry 覆盖。

`zlc_neutral_atom.logic_nodes.readout`只容纳至少两个读出Logic node真正共享的窄合同；包根不得重导出contracts、codec、analysis或repository的宽API，也不得让轻量合同import加载SciPy。Calibration与Occupancy的算法、artifact和repository各自从自己的语义owner leaf导入。包根不提供lazy `__getattr__`兼容表或重复出口清单；稳定public用户面由notebook/workbench facade组合，领域实现直接依赖leaf owner。fresh-process import ratchet机械验证这一终态边界。

### 13.1 Artifact

```text
ReadoutFeature =
    BoxFeature
  | PerSitePsfFeature
  | UniformPsfFeature

ReadoutModel:
  feature: ReadoutFeature
  thresholds: ndarray[site]
  usable_sites: ComponentValidity[site]

CalibrationArtifact:
  source_binding: (CaptureArtifactRef, CalibrationCaptureLayout)
  frame_contract
  site_map
  models: tuple[ReadoutModel, ...]  # non-empty, kind unique, canonical order
  default_model_kind: ReadoutModelKind

CalibrationReport:
  request
  software_lineage  # passive numpy/scipy text only
  group_contexts    # 每组完整的 (AxisId, logical index)，不匿名 flatten
  reference_average + reference_average_validity
  reference_box_signals
  labels + split + per-model short_signals/short_validity
  thresholds/polarity/fidelity/ablation + PSF diagnostics

CalibrationAnalysisRequest spatial intent:
  expected_centers_xy: optional ndarray[site, xy]  # preview 可省略，正式提交必需
  maximum_site_residual_px: optional positive float # 与 expected_centers_xy 成对
```

一次 calibration 可产生共享 `SiteMap` 的多种 model。feature 表示训练前已经确定、且训练/运行共同执行的信号提取数学；model 只再绑定阈值和可用 site。这个两层结构有两个真实消费者，不能合并：analysis 必须先提取 short signal 才能学习 threshold，runtime 必须执行同一 feature；但不存在再镜像一遍 feature 字段的 `ReadoutFeatureSpec`。`models` 必须非空、kind 唯一并按 `ReadoutModelKind` 声明顺序排列，`default_model_kind` 必须命中已有 model，不保留 optional/default-policy 包装或 tuple-first 猜测。

所有 ndarray 在领域值构造边界复制为 C-contiguous read-only owner。`CalibrationArtifact` 不保存 fingerprint、FrameContract/SiteMap fingerprint 镜像、model header、quality evidence、generic parameter bag 或资源证明；内容身份只由 durable codec bytes 和 CAS `ContentRef` 拥有。Repository 的 typed decode 构造领域值一次，之后信任 immutable type，不为防 `object.__setattr__`、pickle 或反射攻击在 getter/worker 中反复重编码、重哈希或 replay 科学算法。

Readout contract 的公开序列化面只包含由真实 Capture/Calibration artifact 静态调用的 owner `to_tree/from_tree`；它们不是独立文件、wire union或repository，因此不各自拥有 standalone bytes codec、nested schema discriminator、格式常量或异常层。外层 artifact 的 `format/schema`、canonical decode、完整 typed reconstruction 与全 payload re-encode 是唯一 durable canonical admission；nested parser 只核 exact field set、委托 foreign owner subtree并构造领域类型，领域不变量只由各 contract 构造器验证。`CameraEventReadoutSetting` 的 tree 函数是 descriptor codec 私有实现，不能扩成第二套公共 API。`FrameContract` 同样没有 digest/fingerprint：运行时适用性用结构相等，持久内容身份只归外层 codec bytes/CAS `ContentRef`。这些nested值不定义standalone format，因此不存在独立tag、兼容reader或转换器。

quality、阈值诊断、PSF fit、train/test split 和 drop-worst ablation 只住在 `CalibrationReport`，不复制进 runtime model。report 必须保留 short signal 的 component validity，坏 component 不能在图、histogram 或统计中被当成零。NumPy/SciPy 版本只作为 report 中两条被动文本 lineage 随 blob 保存；读取端不与当前环境比较，它不参与准入、重放、格式选择或模型适用性。不存在 backend schema、定宽版本字段、模拟升级测试或 WorkPlan binding。

Readout model 的选择身份只有 `CalibrationArtifactRef + ReadoutModelKind`。运行时 fluorescence 物理不变量固定为 `occupied = signal > threshold`，不持久化 per-site polarity；short calibration 若拟合出 `bright_above=False`，该 site 在 `usable_sites` 中明确无效，绝不能把 `< threshold` 变成另一种 runtime 模式。feature/site/model 的 `ComponentValidity` 逐层取交集；无效 signal 与 occupied 使用规范 filler，但消费者必须读取 validity，不能把 `False` 解读为 dark atom。

```text
FrameContract:
  full frame ValueSchema (data axes / validity / dtype / unit)
  stable camera/sensor identity + optical/readout path identity
  sensor/ROI/binning geometry
  dtype + count unit
  exposure/gain/readout mode
  coordinate frame

SiteMap:
  stable site AxisSpec
  site coordinates in the same coordinate frame
  component validity
```

Camera geometry、ROI/binning 整除、output shape、spatial axis 与 real-count dtype 是设备物理事实，只由 `zlc_neutral_atom.devices.camera.contract` 拥有。`CameraCaptureDescriptor`、`CameraPhysicalFacts`、capability evidence 与它们的 canonical codec 都在同一设备 owner；读出层的 `FrameContract` 与 `CalibrationCaptureLayout` 只增加 Calibration/Occupancy 所需的光路、事件布局和适用性语义，并委托 Camera owner 序列化嵌入的相机值，不能复制第二套物理公式或让 device/runtime 反向依赖 Logic node。

Calibration layout owner 是 READOUT_EVENT 与全部其它 named logical context 做稀疏 join 的唯一实现；它返回包内 `_CalibrationCaptureJoin`，只保存 point-context 与按 `(reference events..., readout event)` 排列的 physical rows，repeat 只在取帧/生成 report context 时惰性展开。它不是公开 DTO、持久格式或缓存层。FrameContract 先完成廉价的 descriptor/AxisId/schema admission，再调用 layout owner；任一 selected event 缺失或 context 不成套都 fail-closed。formal preflight直接对source执行一次FINAL `admit()`并解析一次`_ResolvedCalibrationSource(source binding + FrameContract + physical context + join)`；不存在先造一份inspection/summary再重复admit的平行读取面。execute只消费这份prepared resolution。`CalibrationAnalysisResult`同时保留exact `AdmittedCapture`与同一resolution，final commit只复核process-local token、artifact与resolution字段、以及join对report contexts的匹配，不重新decode capture、不重建physical index，也不产生第二份join。整个flat RunPlan从preflight到finalize持有capture与calibration repository root borrow；close要么在preflight前获胜，要么明确失败并等待该run释放，不能在lazy frame读取或CAS staging中途使authority失效。不存在只为测试服务的`from_schema`、raw-row diagnostic list、公开bracket type或witnessed-layout wrapper。Workbench mint descriptor后由紧邻的`CaptureStreamContract`构造边界完成一次schema admission，mint helper不提前做相同全表校验。

所有会改变模型数值解释的采集设置都保存在 `FrameContract` provenance 中；artifact 构造时一次验证 SiteMap coordinate frame、site coordinates、feature boxes 和 model axes 与该合同一致。公共API application 在prepare/bind时提交 `CalibrationArtifact + 当前 FrameContract`并比较真正影响模型适用性的事实：camera/sensor/optical path identity、ROI/binning/geometry、spatial axes/coordinate frame、dtype/count unit、gain、readout mode与frame schema。raw `exposure_seconds`和包含它的opaque settings fingerprint仍完整保留并显示诊断，但在尚无typed effective illumination/integration-window contract时不得单独阻断Occupancy，因为真实窗口还取决于pulse probe/readout schedule。绑定后的唯一public数值入口是 `apply_readout_model(model, frame, *, expected_frame_schema=bound_frame_contract.frame_schema)`；它要求`frame.schema`与冻结schema精确相等，再调用共享feature extractor与classifier。该逐帧schema guard不重复验证物理context，也不复制frame/site/model digest；不存在无`expected_frame_schema`的裸二参数operator或另一层wrapper。

`FrameContract` 只回答 camera 如何解释一帧，不能单独证明该帧曝光时原子装置经历了同样的 pulse 条件。因此权威 calibration 还必须保存由 **CaptureArtifact 中已经冻结的 camera physical facts 与 pulse lineage** 派生的 `ReadoutPhysicalContext`；调用者不能提交一个自报 context/digest 来给自己作证。context 绑定 pulse-owned `target_abi_fingerprint`，使 raw lane、logical port 到 lane 的映射、DAC bus index/width/encoding/safe value 与 latch clock 任一改变都会拒绝旧 calibration；它不是 whole-artifact fingerprint，也不把无关 pulse 编译细节误当成适用性。

每个 readout event 先把已经包含 channel delay 的物理 trigger 上升沿作为时间锚，再用 camera-qualified integration-start offset 和实际 exposure 得到严格半开窗口 `[start, end)`。当前实现只有 nullable scalar offset，**尚不存在**名为 `CommonFrameAperture` 的类型级证明；该名字只描述 完成相机经验时序与外部触发资格后才能发布的能力。开放 scalar 权威路径前，相机经验时序与外部触发资格必须对具体 sensor mode、applied global-exposure mode、ROI 与 readout speed 证明全部输出像素共享同一个 integration aperture，并由届时的 typed capability 承载；只读取到一个 global-exposure 枚举值或非空 scalar 不构成证明。若 qCMOS 实际是 rolling/per-row aperture，当前 scalar capability 必须继续为 `None`/NO-GO；必须先引入与 spatial-y/component axis 对齐的 typed aperture model并逐 component 派生适用性，禁止拿平均行、首行或一个经验 offset 代表整帧。

EDGE trigger 下 trigger 只负责锚定，trigger high width/下降沿不作为被测物理条件；context 收集窗口起点的完整状态以及窗口内所有其它 logical digital output 和 decoded DAC value transition。窗口起点 transition 进入初态，恰在 `end` 的 transition 不进入本帧。有限 pulse 在 DONE 时的真实 bus safe 行为同样属于物理 waveform：RTL 在 DONE 边界清除 undelayed bus，registered 输出从下一 tick 可见 safe；每个物理 bus 再按其冻结 delay 后移，因此 safe transition 位于 `DONE + 1 + bus_delay`。若它落入曝光窗就必须写入 context，不能只展开用户编程的 DAC segment。compact repeated DAC 或 live ramp 若无法从当前 TargetIR 无歧义展开则 fail closed，绝不猜中间值。

同一calibration capture中所有被layout选作runtime readout event的repeat/scan cells必须派生完全相同的`ReadoutPhysicalContext`；reference events可以承担不同的制备/标签物理语义，不能被错误要求与readout event同波形。preflight从持久pulse lineage派生一次并写入同一source resolution；triggered occupancy则在任何camera arm/FPGA FIRE之前从当前camera capability、当前compiled pulse/cell plan派生并与artifact比较。`apply_readout_model`只是已完成上述bind后的无副作用数值evaluator：它强制精确ValueSchema等值，但自身没有artifact、physical context或association authority，不能产生或冒充正式occupancy artifact。qCMOS没有edge-to-integration offset资格化证据时，adapter必须发布`None`；此时它可以做诊断capture，但权威calibration/triggered occupancy必须明确拒绝，不能默认猜`0`。VirtualCamera的已声明offset为`0`，可用于离线/E2E验证。

Calibration的计算事实与提交权威是两个不同类型。`CalibrationComputation(artifact, report)`只表示纯计算已经通过artifact/report绑定校验；该构造边界也是“detector centers符合request中spatial intent”的唯一owner，后续边界信任这个不可变类型，不重复计算residual。系统没有public raw-array calibration函数或非权威数组入口；正式flat RunPlan的package-private `_analyze_calibration_resolved(...)` 从preflight已经持有的exact `AdmittedCapture + _ResolvedCalibrationSource`完成计算并铸造closed `CalibrationAnalysisResult`。不存在可被普通调用者直接调用的公开authority constructor。这个结果携带“意图已由CalibrationComputation绑定验证”、同一次process-local source admission与exact resolution。`final_commit`只核对这些held source/resolution facts与逐组context，不重验site residual、不重新admit source、不重建physical waveform。加载已有artifact/report时repository paired decode可以重建`CalibrationComputation`并复用其纯绑定验证，但不能借decode把它升级成提交权威。这个类型级边界消除了SiteMap/feature/threshold与另一份report错绑后被提交的路径；producer内部仍直接核对source layout、grid、frame shape、site count、model kinds/default、逐model threshold、由held-out evidence推导的usable mask，以及request声明的feature类型/box/PSF geometry，作为实现自检；不恢复artifact fingerprint或proof graph。

Repository将runtime `CalibrationArtifact`与diagnostic `CalibrationReport`分成两条真实读取路径：current manifest配对一个typed artifact blob与report metadata；全分辨率`reference_average(<f8)`和validity使用两个raw CAS blob，由metadata的owner-encoded `ContentRef`引用。`load()`本身要求FINAL并只读取manifest+artifact；显式`load_report()`或paired `load_computation()`才materialize diagnostics，后者一次返回已经重新互证的artifact/report而不做两次decode。Repository内部不存在仅为重复字段投影而生的summary、平行metadata facade或兼容inspection对象；但成功的Calibration Task必须从同一`load_computation()`显式生成用户可发现的`report/` bundle：`summary.json`列出typed Calibration/Capture refs、两个CAS根和关键逐model/site结果，`diagnostics.npz`/`sites.csv`是便利副本，frontend只把neutral的`CalibrationReportProjection`渲染为overview、逐model histogram和PSF PNG。该目录逐字声明非权威，Occupancy及所有机器消费者只admit CAS ref；因而人类报告不会成为第二算法或repository真相源。写入端在发布manifest前使用读取端同一个structure/typed decoder做round-trip，不能生成自己无法读取的FINAL。pending recovery对raw arrays使用流式hash，不materialize大图，也不重跑detector、fit或threshold。Repository锁只保护open/commit状态与coordinator线性化点，CAS read/write、report decode和大数组复制全部在锁外；分配失败以当前操作的真实异常返回，不建立另一套准入协议。

这里必须区分canonical repository commit与Calibration Task的operator folder。后者的相对`folder`由`resolve_under_project`锚定项目根，默认`_output/calibrations`，显式绝对路径保持用户选择；writer在同一根下stage新的`report/`与可选`frames/`，把已有目录移到nonce backup，安装新目录，最后才以单文件`os.replace`替换`calibration_ref.json`。若Python进程仍在且任一步抛错，`finally`按已安装清单回滚目录；这只是in-process rollback与pointer-last可见顺序。该operator folder不提供完整file/directory fsync或task-folder recovery journal，因此进程/OS崩溃可能留下stage/backup、先前pointer配新非权威report或replace尚未持久化的状态；禁止称整个folder为crash-atomic/durable transaction。机器authority仍仅是pointer中两条typed refs及其各自具备crash-consistency合同的canonical repositories，report/frames的残余不能被admit为成功artifact。

Occupancy Repository的FINAL metadata以owner codec保存exact counts/occupied `DatasetSchema`，并保存raw values/validity blob的ContentRef与size。公开读取只有一次`admit()`：它验证FINAL与run generation，admit真实Capture/Calibration依赖，从同一binding重新派生schema并与持久metadata exact compare，然后解码counts/occupied/validity并返回一个`ResolvedOccupancy`。Figure、fit或navigator若需要axes，直接消费该已admit artifact；不再为“metadata-only”另造一套inspection DTO、读取生命周期或字段镜像。持久schema只是FINAL索引和早期fail-closed证据，不是第二物理算法authority；domain schema validator与zlc_data schema codec仍各只有一个owner。


### 13.2 算法权威与明确偏离

Calibration/readout 的production物理算法权威是当前 `zlc_neutral_atom.logic_nodes.readout` owner；`main@6c337d49c7086fa0ff21f879cd159bdf0e753f51`只作为这一组已独立验证科学算法的固定oracle，任何旁路归档、非权威样本或旧UI都不能反向定义合同。当前实现保持已验证的 reference frame 平均、Gaussian smooth/local maxima/5×5 subpixel refine、separable lattice repair、四种 grid order、3×3 BOX mean 默认及 mean/sum/median/max、7×7 empirical PSF 与 annulus-median、uniform PSF、96-bin quick Otsu、pooled per-site bimodal strict consensus、per-site/per-class 90/10 seed-0 split、120-bin common edges、empirical balanced threshold、held-out/model/global fidelity 与 drop-worst ablation；训练和 runtime 共同调用唯一 feature extractor。

main strongest-N detector 存在一个不能靠同帧内部规则性消除的信息歧义：真实 site 变暗而出现更亮伪峰时，它可能产出另一个自洽但物理错误的规则格；仅凭同一张图的 peak count、lattice residual 或规则性无法区分真格与假格。因此 exact-main detector 一行不改，但正式 authority 增加独立空间意图 gate：`expected_centers_xy` 必须按当前 ordering/FrameContract 给出粗略逐 site 位置，`maximum_site_residual_px` 给出显式容差；detector 结果不吸附、不重排、不替换，只要任一 site 超限就拒绝。系统没有非权威raw-array calibration入口；committed-capture analysis在启动/昂贵计算前拒绝缺失空间意图。package-private `_analyze_calibration_resolved`成功返回的closed `CalibrationAnalysisResult`成为后续提交边界信任的类型证明，不在authority mint/final commit重做同一residual校验。Workbench可以显示apparatus/已admit同物理FrameContract calibration提供的独立expected centers并让用户明确核对；不能把本次detector输出无提示地自动回填成自己的权威证据。

只保留四个有明确错误依据的偏离，并分别使用同帧/物理不变量测试：

- non-finite/invalid reference observation 不得因为 boolean filler 变成 dark；缺少 required PSF pixel 时整 site invalid，不 renormalize kernel 改变信号尺度；训练 annulus 只消费有效 pixel，uniform kernel 只平均有效 site，invalid placeholder 不得污染其它 site；BOX finite-only reducer 仅在至少有一个有效 pixel 时有效，BOX-only request 不执行或受 PSF geometry 约束；
- short signal 出现反 fluorescence polarity 时 site invalid，runtime 仍只执行 `>`；
- 完全无判别力的 site 即使产生有限 threshold，也只有 chance-level `0.5` held-out balanced fidelity，不能进入 runtime；request 只增加一个显式 `minimum_site_fidelity`（默认 `0.5`，usable 必须严格大于），不恢复 Holm/Clopper/valley 等无 main 依据的统计 gate，也不因单 site 低质量拒绝整件 artifact；
- 模型适用性使用完整 `FrameContract` 和 component validity，不再只按 array shape 猜相机/ROI/setting。

冻结的 `tests/fixtures/main_readout_oracle.npz` 只含 raw synthetic camera frames和由上述 exact main commit 产生的 expected arrays；测试不在运行时 import 另一棵工作树或任何其它算法实现，也不用新实现同款公式生成期望。它完整覆盖 detector、三种 feature、labels/split、quick/formal threshold、fidelity/ablation、PSF diagnostics、runtime occupancy 和 `(R,P,site)` 保形；单独的 intentional-difference tests 才覆盖上述四处纠错。真实 CaptureArtifact E2E 另外锁定相机 `ValidityContract` 接缝：VALUE 只能生成整帧 VALID/INVALID，局部坏像素保守使整帧 invalid；只有 schema 已声明 COMPONENTS 时才沿声明轴保留 component mask，不能伪造 y/x validity。

### 13.3 输入

```text
CalibrationInput =
    LiveCalibrationInput(CaptureSpec)
  | CaptureArtifactInput(CaptureArtifactRef)
```

neutral domain/runtime 不接受“执行时读取 session current calibration”、裸 filesystem fallback 或兼容 path search。Notebook facade 构造 Occupancy/Detection/Scan request 时必须显式接收并 load/admit `CalibrationArtifactRef`，验证 readout binding，解析显式/default/唯一 model，并把具体 ReadoutBindingKey、CalibrationArtifactRef 与最终 `ReadoutModelKind` 冻结进 request。若 ref 缺失、binding/FrameContract/model 不适用或选择歧义，request 构造/preflight 失败；运行中不存在可切换的 facade current pointer。Workbench 同样在用户点击 Run 时冻结用户明确选择的 ref，而不是让 processor 回查 mutable session或按 repository 最近文件猜。

### 13.4 执行

```text
CalibrationTask:
  LiveCalibrationInput:
    CaptureSession -> CaptureRepository.atomic_put -> CaptureArtifactRef
  CaptureArtifactInput:
    CaptureRepository.admit(CaptureArtifactRef) -> AdmittedCapture
  -> resolve once: AdmittedCapture + _ResolvedCalibrationSource
  -> _analyze_calibration_resolved(held source, held resolution, explicit request)
       -> 只从 exact admitted committed frame source 流式执行权威算法
       -> CalibrationComputation(artifact, report)          # closed non-authoritative pure result
  -> CalibrationAnalysisResult(computation + held source/resolution)
  -> CalibrationRepository.final_commit(runtime artifact + diagnostic metadata/raw arrays + manifest)
  -> CalibrationArtifactRef
```

live 路径先提交原始 CaptureArtifact，再与 offline 路径汇合；detector/feature/model 无法构造完整请求结果时不发布 CalibrationArtifact，但原始 capture 仍可诊断和重跑。低 fidelity 本身是 report evidence，不由没有 main 依据的 Holm/Clopper/valley gate 擅自拒绝整个校准；具体坏 site 通过 `usable_sites` fail-closed。virtual/real 只在 CaptureSession adapter 不同，提交后的 calibration 代码完全相同。

qCMOS calibration在下列真机硬件事实与production DCAM composition全部成立前保持typed NO-GO；正式adapter必须兑现以下合同。最终 adapter 必须在完整配置事务结束后一次读取并冻结实际 `EXPOSURETIME`、`TIMING_MINTRIGGERINTERVAL`、readout speed、sensor mode、trigger-global-exposure 以及 trigger source/active/polarity；qCMOS 的 minimum interval 必须严格为有限正数，trigger trio 必须仍是 external/edge/positive，无法读取、后续 ROI 操作改变 trigger mode或配置中途失败都会清除旧 working-point proof。ROI 写入按 `SUBARRAY OFF -> zero positions -> sizes -> final positions -> SUBARRAY ON/readback` 完成。所有 public configure 路径在取得 acquisition lock 前后都检查 arming/armed，关闭同线程 RLock 重入与跨线程 B→A 的 ABA；`cap_start` 后还重新从硬件读取完整 working point，任何 drift 都先 stop/release/disarm并失败，因此 camera endpoint 在 FPGA FIRE 前比较的是 arm 后真实 readback fingerprint，不是配置期缓存。

通用 compiled binder 只有在 adapter 实际发布上述冻结 working point 后，才会对**同一 artifact 内相邻 trigger**做 fail-before-arm 的最小间距检查；当前 virtual 与 production DCAM 软件路径都能消费该合同，但真实 qCMOS 只有跑完E0 contract kit并保存具体工作点证据后才能宣称通过。这也不等于已经证明 arm-ready 到第一沿、最后一沿到 drain/下一 run 第一沿的跨边界余量，后两者仍必须由相机经验时序与外部触发资格给出。更根本的实验 gate 也仍存在：adapter 配置 `TRIGGERACTIVE.EDGE`，一次 arm 只有一个 hardware `EXPOSURETIME`；checked-in calibration pulse 的 20 ms/5 ms/20 ms 只是 FPGA period/probe-window 时长，不能被软件宣称为三种相机曝光。资格证据必须证明 edge-to-integration offset、所选 trigger mode 的曝光语义、每沿一帧/顺序/不漏、arm/first-edge 与 run-boundary margin；若相机只支持 per-arm 固定曝光，bracket 必须改成物理可实现且算法语义正确的协议，不能把 pulse width 冒充 camera exposure。virtual 帧通过、计数最终相等或 GUI 上看见三帧都不能替代该证明；这些事实资格化前 qCMOS 正式 calibration 用户路径仍为 NO-GO。该 gate 优先使用相机与现有 FPGA 的硬件时序，不自动授权 RTL/bitstream 变更。

CaptureArtifact 的大帧面只有一个公共 owner：`frame_source: CaptureFrameSource`。它保存完整 `DatasetSchema`、block/revision、精确 cell schedule、event-order metadata 与raw frame chunks；不再并列暴露 `.block`、`.event_metadata`、`.source_cell_schedule` alias，也不保留旧 whole-DataBlock blob reader。普通 `load()`只验证 manifest/index及 chunk refs，实际 `read/iter_cells` 首次用到某 chunk 时核验其 size+SHA，pending-commit recovery才逐块流式全验。这里的检查证明commit/journal authority与索引可解析，不等于全介质健康扫描；未读取chunk的损坏会在第一次读取/计算时以内容损坏明确拒绝。显式`materialize()`是唯一whole-dataset入口，真实分配错误原样传播。index写入和读取共用同一个size、canonical structure、typed reconstruction和re-encode owner；任何writer自己读不回的index在manifest可见前拒绝。compiled capture plan在任何camera arm/FPGA fire前取得repository root borrow；close若先赢则run在硬件前拒绝，run若先赢则borrow阻止repository在finalize/cleanup前关闭，不能完成硬件后才发现保存根已经失效。

frame bytes 的 invalid/component-invalid/NaN 规范化只由 `zlc_data.canonical_value_array` 拥有：它在 schema-level INVALID 快捷返回前仍验证 dtype/shape/validity，避免任意错误 frame 取得合法 invalid digest；普通 C-contiguous uint16 VALID frame 返回原 view，不为 hash/持久化前检查复制整帧；Capture repository 仅在真正写 CAS 的边界转为 bytes。native/big-endian 等数值等价 dtype 先转换到 schema 的 canonical endian，再对 float/complex 的每个 NaN component 规范 payload；不能用 canonical component dtype 去解释尚未换 endian 的 complex bytes。schema-level INVALID 沿用既有 `canonical-invalid-values` event digest，component-invalid 则以相同 mask 与零 filler 产生相同 identity，不能让两条路径漂移；canonicalization所需临时数组由该owner直接分配，失败就使当前编码失败。

analysis 按 resolved join/context 从 `CaptureFrameSource` 流式消费；不 `np.stack` 原始 frame、不为每帧构造第二份 owned image，也不生成 `(groups, shots, H, W)` 临时栈。reference 阶段允许为“平均图”和“按最终 site feature 提取”各走一次可重复源遍历；short 阶段对每帧只准备一次并同时填入全部 model 的小型 `(model, groups, sites)` signal/validity 数组，禁止每个 model 重读整套 qCMOS frame。reference average 使用一个 float64 image accumulator和按真实 shot 数选择的最小无符号 count image，最终原位除法；空间复杂度是 `O(HW + groups*shots*sites + models*groups*sites)`，不是 `O(groups*shots*HW)`。cell 地址用可重复惰性 generator，不提前构造 repeat-expanded row对象；report 逐组保存原来的 `(AxisId, logical index)` context，repeat、多条 point axis和二维 data axes各自保留语义，绝不能变成匿名 `data_points/data_dim`。

`CalibrationAnalysisRequest.max_drop` 省略时取 `min(5, site_count)`，显式值不得超过 site count：更大的值只会重复“全部 site 已排除”的同一报告，不是新证据。analysis对完整矩形layout直接计算selected-row cardinality，对sparse/product layout流式遍历physical rows，不先保留一份row/context图。大阵列热点以真实profiling定位；发现join、lattice pair、ablation或image workspace形成不必要复制时直接优化对应owner。

runtime feature extractor 保留 camera 原 dtype，只把当前 site 的 BOX/PSF 小窗口转换成 float64；float reducer/weighted sum仍与 main 数值路径一致。annulus-median fallback与numeric operator仍由calibration owner单源实现；pipeline不复制公式或维护镜像scratch字段。

不需要 CalibrationService、child Measurement Run、calibration Processor、recursive execution plan、WorkPlan 或 reducer 包装。`compile_calibration_artifact_plan` 是一个同步 flat `RunPlan` adapter：preflight完成FINAL inspection后一次load/resolve并取得capture+calibration repository borrows；execute调用package-private `_analyze_calibration_resolved`；finalize只消费同一held result并做FINAL commit。

Occupancy request 携带 `ResolvedCalibration(reference, artifact)`、已解析的 `ReadoutModelKind`，以及Camera Measurement在实际reconfigure/readback并开始发布后生成的完整`CameraFrameOutputBinding`。该binding冻结exact output/event index、ReadoutBindingKey、完整Camera working-point/capability evidence、device binding stamp、frame schema及stream id/generation；停止或cleanup必须撤销它，未运行的declaration不能伪造这份物理事实。repository `admit` 负责 FINAL/source 验证，但返回值不冒充不可伪造的 authority token。该轻量Calibration领域值归 `calibration.py` 所有，因此导入 occupancy runtime 不加载 calibration repository、report codec、analysis 或 SciPy。

processor bind必须同时比较Calibration的完整`FrameContract`与当前Camera binding：camera/sensor/optical-path身份、sensor geometry、ROI/binning、axes/frame、dtype/unit、exposure/gain/mode/settings fingerprint、SiteMap/model kind全部一致；同shape从不代表适用。hot snapshot再验证exact stream id/generation与schema，cursor只在同generation内消费；任何producer restart或schema漂移立即失败。hot path每帧调用唯一`apply_readout_model(model, frame, expected_frame_schema=bound_schema)`，原子发布counts、occupied、metadata和相同component validity，processor provenance携带source binding identity。公开API不能暴露绕过binding的裸array evaluator。

## 14. PulseScan

### 14.1 产品语义与 owner

PulseScan 是一个 Measurement，语义固定为：消费一份冻结的有限 pulse-parameter scan program，并把一个**已经运行的外部 signal**在该执行期间发布的、由producer证明属于本次FIRE的事件物化为 `(R,P,*data_shape)` ScanArtifact。`AutonomousScanSlotProgram | ApiSlotSegmentedProgram` 的共享程序语汇由 node-neutral `zlc_neutral_atom.timing.pulse_parameter_scan` 唯一拥有，供 PulseScan 与其它需要相同物理程序的 capability 静态复用。

它只拥有：

- 一条 `ScanSignalBinding`；
- bound sequencer Port/session；
- scan-output contract、collector/materialization、preview与ScanRepository commit；
- pulse terminal、source event、processor/calibration与transform lineage。

它明确不拥有：

- Camera、Processor、Figure或任意其它producer的设备Port；
- producer的start/stop/cancel/claim、buffer、工作线程或artifact repository；
- Camera trigger channel、frames-per-cycle、ROI、calibration model等source-specific字段；
- panel displayed/latest state、selector widget、Fit UI或另一套采集pipeline。

因此 Camera frame、Occupancy counts/occupied、Figure Area-derived Dataset或未来其它typed output都走同一个PulseScan application；差别只在producer是否能为所选output提供正式association。新增producer不需要修改PulseScan。

### 14.2 Request、binding 与 schema

```text
PulseScanBoundRequest:
  program: timing.pulse_parameter_scan.AutonomousScanSlotProgram
         | timing.pulse_parameter_scan.ApiSlotSegmentedProgram
  signal: ScanSignalBinding

ScanSignalBinding:
  producer_definition: DefinitionKey
  output: DatasetOutputDeclaration
  transform: DataTransformSpec | None
```

`DefinitionKey + output declaration`描述可保存的领域意图；Workbench在Start时把它解析到一个明确RUNNING的producer row、lifecycle generation与当前output schema。PulseScan不保存Qt row、标题或临时路由key；artifact从实际事件保存source run/source id/stream generation。若producer停止、重启、generation变化或output schema改变，本次scan失败，不能自动重绑到“同名最新实例”。

输入transform是权威意图，不是display state。bind时由`zlc_data.commit_transform()`在输入`ValueSchema`上唯一冻结为`SignalProjectionAuthority(input schema, CommittedTransform|None, output schema)`；association-capable source经过transform wrapper后必须保留原association能力与原EventRef/processor lineage。display的repeat mean、current facet、latest、临时colormap或panel zoom不能进入这个值。

point truth由program自己的point table/PointLayout拥有，repeat truth由program的repeat count拥有；所选signal只提供每个cell的`ValueSchema`。最终schema唯一构造为：

```text
DatasetSchema(
  repeat_axis = R,
  point_axes + PointLayout = program P,
  cell_schema = transformed signal ValueSchema,
)
```

最终数组永远是 `(R,P,*data_shape)`。P可以由多个具名logical point axes和非平凡PointLayout描述，但物理storage仍只有一个P维；绝不能把logical point axes展开成多个ndarray前缀，也不能把全部data axes压成匿名`data_dim`。标量cell使用canonical scalar axis，所以shape是`(R,P,1)`。

### 14.3 Ordering capability 与 association capability

信号消费分成两层，二者不可互相冒充：

```text
SignalEventSource
  value_schema(output)
  open_signal_cursor(output)          # future-only、lossless、ordered

SignalEventAssociationSource : SignalEventSource
  open_associated_signal_cursor(output)
```

普通`SignalEventCursor`只证明同一stream generation中订阅后的事件按sequence无遗漏交付。它适合live Processor、显示和普通事件驱动计算，但不证明这些事件由某个pulse FIRE导致。

正式scan只使用`SignalEventAssociationCursor`：

```text
arm_signal_association(request) -> token       # 必须在FIRE前
bind_signal_association(token, PulseTerminalAck)
next_associated_signal(token) -> SignalEvent   # 只交付该group
finish_signal_association(token) -> SignalAssociationEvidence
close()
```

`SignalAssociationRequest`包含唯一association id、pulse session/cause id、compiled artifact digest与期望的**所选signal事件数**。producer可把一个output event映射到多个物理records或cycle phase；例如`frame_0`在`frames_per_cycle=3`时，一个selected event对应完整三帧cycle中的phase 0，物理trigger count由Camera producer核对，PulseScan不得假设`one y == one camera edge`。

arm必须冻结“从现在开始的下一组”，并拒绝重叠token或无法保证完整组的状态；bind必须核对terminal的session/artifact；next不得读到组外event；finish必须验证exact count、物理producer完整性与terminal digest。关闭或失败丢弃未完成token，不得把普通cursor读到的值补入。

所有Dataset output都可以出现在picker中，因为它们仍可用于普通Figure/Processor；是否具备association是运行能力而不是Definition发现过滤器。UI可显示当前running producer的能力状态，但最终gate仍在application prepare/FIRE边界。缺能力必须产生具名`SignalAssociationUnavailable`，且没有任何FIRE发生。

### 14.4 执行闭环

#### Autonomous SCAN_SLOT

```text
preflight:
  bind live target and compile one finite AUTONOMOUS_SCAN_ONCE artifact
  create exact collector/DatasetBuilder and optional read-only preview
  open one sequencer session

execute:
  session.prepare()
  open producer associated cursor
  token = arm(expected_event_count = R * P)
  session.fire()                         # FPGA owns all edge timing
  terminal = session.complete()
  cursor.bind(token, terminal)
  for address in DatasetCellSchedule(R-major/P-fast):
      event = cursor.next_associated_signal(token)
      collector.emit(event.value, join_key=address,
                     causation=(event.ref, event.trace.causation_refs...))
      DatasetBuilder.consume()
  evidence = cursor.finish(token)
  seal collector EOS and commit
```

完整SCAN_SLOT表在FIRE前冻结；一次FIRE后由现有FPGA自主执行全部pulse/trigger时序。host只等待terminal、排空producer已经关联的事件并物化数据，不逐point sleep、fire或修改slot。

#### API_SLOT segmented例外

只有program明确是`ApiSlotSegmentedProgram`时，preflight编译P个唯一`STATIC_ONCE` artifacts，并按R-major/P-fast重复使用它们。每个cell打开独立PulseSession，先为一个selected event arm association，再由硬件执行该segment的全部edge，取得该session的terminal，绑定/消费/finish该group后进入下一cell。上游producer始终是同一个running generation；PulseScan不stop/re-arm它。segment间存在不受上限保证的host gap，因此这种program只能用于物理上接受该gap的API-slot实验，不能包装成autonomous或成为SCAN_SLOT fallback。

无论哪种program，scan cancel/cleanup只关闭自己的source cursor、collector、preview、sequencer session和repository borrow。上游producer仍由其原Logic row/RunHandle拥有，scan成功或失败都不能替它调用cancel、stop、close或释放设备claim。

### 14.5 Producer association 产品矩阵

#### Virtual readout Camera

virtual readout Camera是当前第一个production association owner。物理真相必须留在`VirtualCamera`，因为只有它同步观察`VirtualSequencer`的唯一in-process trigger wire，并知道实际收到的playback、camera trigger channel、trigger group、FIRE generation与生成的source ordinal interval。Camera Measurement只把该authority与自己的`frame_i` phase projection组合成signal cursor；它不复制pulse解析或仿真物理。

一份virtual Camera evidence至少绑定：association request、camera binding/capability identity、stream id/generation、output phase/cycle cardinality、trigger channel、实际物理source ordinal起止、pulse artifact digest、exact terminal digest与完整produced count。terminal中“compiled schedule expected counts”必须与VirtualCamera实际观察的playback和frame production同时吻合；只核对terminal字段而没有VirtualCamera记录不足以铸造证据。

FREE_RUNNING `mot_camera`没有pulse因果authority，必须拒绝。它仍可作为普通Camera Measurement/monitor或MOT Task自己的耦合设备，不能因为类型同为`VirtualCamera`就获得association；能力由installation显式注入具体readout role，不按class name或字符串猜测。

#### Occupancy

TaskConsole latest host只调度`PreparedOccupancyProcessor.evaluate()`服务latest display；`RunningOccupancySignalSource`为formal consumer打开独立upstream associated cursor，不能与latest host竞争事件。对每个关联Camera Value调用同一个唯一classifier，恰好生成一个`OccupancySignalValues(counts, occupied, rate)` typed transaction，再按所选output投影一个derived SignalEvent。其Trace直接引用Camera EventRef和Calibration ArtifactInputRef，并追加同一个`ProcessorStageProvenance`。

只有这条严格1:1映射可传播association。未来若Processor做window、drop、merge、one-to-many、many-to-one、异步latest或跨source join，默认失去association；必须由该Processor自己的领域合同证明新的cardinality/causality，不能借用Occupancy实现或只转发上游evidence。

#### Real qCMOS

Real Camera只有在production installation、AssetMap、CameraExternalTriggerQualification和public composition全部成立时才可暴露associated cursor；任一缺失都使PulseScan typed NO-GO。该enablement属于Camera/installation owner，不修改PulseScan：

1. 以production adapter和真实工作点完成相机经验时序与外部触发资格，冻结exposure/ROI/readout、trigger channel、最小安全间隔、counter reset/rollover与stamp/timestamp语义；
2. 在FIRE前建立完整下一组的driver/source baseline并保证host持续排空；
3. 由现有FPGA执行冻结schedule，不引入host edge timing；
4. run末端核对pulse terminal期望trigger total、camera produced/drained delta、source ordinal、可用stamp/timestamp与完整coverage；
5. 任一不一致整组失败，不提交、不移动ordinal、不丢“多余帧”后继续。

这仍是经验资格化的ordered association，不是逐沿硬件tag。bitstream保持冻结；只有§1.1规定的证据触发条件才能另行评估硬件修改。

### 14.6 数据、validity 与 lineage

每个accepted source event保留：

- 自己的exact `EventRef`；
- direct input EventRefs；
- source run/source id、stream generation与captured time；
- 有序`ProcessorStageProvenance`；
- 每个stage的direct ArtifactInputRefs；
- producer association evidence；
- 本cell的DatasetCellAddress。

`SignalEventSequence`对全scan验证stream/generation/source/processor chain恒定、EventRef严格递增、direct causation类型合法、事件数与cell schedule完全相等、association groups完整覆盖全部events。API模式还保存按cell有序的`ApiSegmentEvidence`；autonomous模式保存一个compiled artifact与terminal。repository加载时用owner codec重新计算并交叉验证所有digest，不能只相信manifest里的字符串。

component validity与采集coverage正交：缺cell、duplicate、gap或association不完整使整个scan失败；一个已完整cell内的dead site/bad pixel在source Value中使用ComponentValidity，写入scan DataBlock后使用DatasetComponentValidity原样保存，供后续reduce/Fit逐component消费。不能把坏component写成missing scan cell，也不能把missing event填NaN后宣布成功。

权威transform只有两种结果：显式具名Select/Reduce，或原样保留所有data axes。禁止取第0项、flatten、按singleton/rank猜role或自动平均信息轴。repeat轴和point layout由scan拥有，input transform不得删除/改写它们；后续Fit可把repeat reduce、把其余axes作为batch，但不能回写ScanArtifact。

`ScanOutputContract`只保存最终output DatasetSchema与collector已完成的identity output事实；输入/output schema与CommittedTransform由`SignalProjectionAuthority`唯一保存，不能在contract再复制一份漂移字段。`build_fit_problem`拥有后续fit packing/densify，frontend与artifact loader不得自行reshape P。

### 14.7 Artifact、preview 与失败语义

一次成功Run只发布一个canonical ScanArtifact。ScanRepository保存program、compiled pulse identities、terminal evidence、SignalEventSequence、projection authority、source Dataset seal、final output snapshot与content-addressed values/validity。若processor、association、transform、collector、cleanup或commit失败，不额外发布名字像成功scan的raw CaptureArtifact，也不存在promote/recover历史；需要独立raw capture时由用户另发一个明确Capture/Camera Measurement请求。

preview只读DatasetBuilder已经committed的revision，可显示PROVISIONAL数据，但没有权力改变source、补cell、提交Fit/Scan authority或成为最终artifact。FINAL只来自repository commit；GUI显示到最后一帧、Run terminal label或plot完整都不能替代seal。

失败后：

- sequencer session按自己的cleanup合同回到终态；
- association token/cursor关闭；
- collector与preview失效；
- repository不发布manifest；
- 上游producer保持其原Run状态；
- GUI显示本次scan的具体错误，并允许用户用新Run重试。

资源冲突只描述当前进程中可证明存在的占用：TaskConsole停止并join本窗口的exact conflicting row后重试同一冻结request；连接关闭后不保存推测性隔离状态，也不自动无限重跑。PulseScan不会因为“可能需要source”而停止其输入producer，因为输入producer不是冲突资源。

### 14.8 UI 与可发现性

PulseScan的普通字段由`LogicNodeDeclaration`投影；唯一特殊UI leaf只负责pulse template、SCAN_SLOT/API_SLOT structured table/program editor。TaskConsole shell不认识scan columns、Camera或trigger字段。`y_signal` picker列出运行中producer的typed Dataset outputs，显示完整`R × P × (*data_shape)`、dtype/unit与可读label；选中后冻结具体producer row和output declaration。

PulseScan作为Measurement不会自动打开plot panel。成功后它发布`scan` FINAL signal，用户像其它Measurement一样手动连到1D/2D/grid/monitor panel；Figure Area/Cross/Fit仍由Figure owner发布自己的derived signals。UI不得弹出第二个DataFigure，不得把selector塞入Measurement form，也不得用panel当前画面作为scan数据源。

Scan table列名使用pulse owner声明的稳定、可读parameter id/label；不得把随机UUID/hash暴露为操作者需要编辑的变量名。保存/加载只使用current PulseDocument与PulseScan owner codec，不保留旧scan JSON、兼容reader或upgrade editor。

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

它不拥有 MOT/readout 等实验 template、DeviceBinding、RunPlan、ScanCellKey、Qt editor 或 panel。neutral template 产生 PulseDocument/scan binding request，workbench editor 产生 PulseDocument command，二者都通过 pulse public API；pulse 不为调用方反向增加字段。pulse-specific `PulseTimelineDocument`及`TargetIR -> timeline`解释归 pulse owner，因为digital delay、loop展开、DAC live-state/ramp与offset-binary转换都是TargetIR物理语义；它不含Qt/Matplotlib、widget state或持久codec。Workbench只选择static/nominal-reference compile、管理editor revision并渲染该值。当前只有 FPGA 一个生产 target，因此 compiler 可以是清晰的 concrete implementation；只有第二个可运行 target 出现且共享 IR 经过验证后，才抽 `PulseTarget` Protocol，不能先建立 target plugin/registry。

Pulse authoring 与加载只保留一个当前合同：`schema="zlc_pulse.PulseDocument"`。可编辑 schedule、scan parameters/recipe/table 与 target 都由 `PulseDocument` 的明确字段表达；scan/API column的顺序、field identity、物理range、unit、clock quantum、参数单位换算，以及trusted-local `scan_table`程序的numeric matrix合同、normalization与commit，都由`zlc_pulse`唯一拥有。`zlc_frontend`只把pulse-owned column描述渲染成可编辑starter文本，Workbench只托管文本/文件/candidate状态；PulseGUI、TaskConsole与notebook不得互相导入workspace helper或各自执行/量化scan程序。编译后的 `TargetIR` / `CompiledPulseArtifact` 是不同类型，不能再用一个并不存在的 `kind=table|sequence` 字段把 raw sequence 塞回 authoring document。唯一公开文件入口是 `load_pulse_document()`，tree boundary 是 `pulse_document_from_tree()`；二者只接受当前 exact field set，所有 save 也只写这一格式，compiled artifact 不作为同名 `_program.json` sibling。`pulse_document_path()` 是扩展名/绝对路径归一化的唯一 owner，load、冲突检查与实际 save 必须消费同一路径；Editor 以一把本地 save lock 串行化同一 session 的并发保存，但不为没有用例的跨进程编辑另建锁文件/事务系统。Workbench 提交 save 时冻结当时的 editor session/generation，到完成前对称禁止 New/Open/load 替换 session，且 stale completion 不得更新新 editor UI；保存中途继续编辑会使当前 revision 自然相对已写入 baseline 保持 dirty。仓库中受版本控制的 pulse JSON 资产与当前 codec 同步提交并通过 round-trip/golden；不存在历史 fixture、旧 parser、逐版本 upgrader 或一次性转换器。仓库外旧文件不属于终态产品合同；未知 schema/field set 由该 current owner 以明确 `ValueError` fail closed，不得按字段存在、shape或名字猜测，也不得提示 runtime fallback。

`scan_sweep_count` 是同一 current `PulseDocument` 的普通显式字段，不是第二种 wrapper/schema，也不是旧 `scan_repeats` reader 的兼容入口。它与 authored `RepeatRegion` 正交：前者只冻结 GUI 下一次完整 scan sweep 的默认数量，后者描述每个 scan point 内部的 pulse timeline repeat；任何层都不得把两者相乘、互相推断或按字段缺失回退。

`TargetIR.fingerprint` 是 TargetIR canonical identity 的唯一 owner；packer、artifact 与 repository 只能消费该派生值，不能把 IR 再编码并重复算 digest。固定点 scan timing 的仿射求值只由 `zlc_pulse.ir.evaluate_affine_tick` 定义，duration-to-ticks 与 integral DAC code 的精确转换分别复用 PulseDocument 的单源转换函数；compiler、validator 与 model 不得各自保留等价公式或“更保险”的第二次解释。

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

近期运行时身份判定到此为止：现有 `image.build_fingerprint`、geometry 与 ABI 三者匹配即允许进入当前pulse协议，不再引入软件deployment状态机、activation/pin token或额外ROM字段作为baseline gate。部署目录可以保存供人审计的 `.bit` content digest、build manifest与timing报告，但它们是离线发布记录，不能冒充硬件runtime readback，也不进入PulseScan的每run类型图。

`design_build_id + timing-signoff digest + programmed-bitstream content attestation` 仅是未来可选增强，因为把新字段放入ROM/USR_ACCESS需要重烧。只有证据已经触发合法RTL/bitstream修复时，才随该修复单独评估；绝不为了架构整洁主动实现。

canonical serialization 固定字段排序、整数宽度、编码和 line ending，digest 使用 SHA-256 并排除 digest 字段自身。

### 15.4 CompiledPulseArtifact

```text
CompiledPulseArtifact:
  source_digest
  pulse_target_digest
  compiler_id
  target_ir
  wire_image
  execution_forms: FIXED | AUTONOMOUS_SCAN_TABLE | API_SLOT_SEGMENTED_EXISTING
  scan_slot_schema + frozen output-edge schedule
```

内容寻址，不通过 sibling 文件名判断新旧，不重复嵌入 source table。

`zlc_pulse` 只负责 pulse program、编译产物、上传、FIRE 与 terminal；它不知道相机、processor、ScanArtifact 或 neutral 的 signal association。prepare/upload/fire 在冻结 bitstream 的既有协议上增加软件侧显式身份与校验，不要求新寄存器或RTL状态机：

```text
PREPARE_AUTONOMOUS(run_id, artifact_digest, frozen full finite logical scan table)
  host/compiler validate frozen streamer geometry、slot schema、all output-edge schedules
  expand repeat axis into table order; require repeat_forever=False, scan_repeats=0
  freeze the complete logical table before fire
  precompute every immutable physical chunk and both bank-address variants before fire
  preload the first two chunks into the current RTL ping-pong banks
  server records PreparedProgramRef(
    server_connection_generation, run_id, artifact_digest, table_digest
  ) in software

FIRE_AUTONOMOUS(PreparedProgramRef)
  current FPGA executes the complete logical autonomous table once
  host never chooses point timing after FIRE
  the FIRE-owned transport observer alone reads STATUS/CURSOR and refills each freed
    bank from the already-frozen chunk sequence: clear READY -> write rows/CHUNK -> re-arm
  after every re-arm, read STATUS/CURSOR immediately; an error/underflow or a cursor
    that already crossed the source chunk means seamless timing was not proven
  either condition invalidates the whole run and enters SAFE

COMPLETE_AUTONOMOUS
  single transport owner reads the currently implemented STATUS/CURSOR/error facts
  return PulseTerminalAck bound to PreparedProgramRef and compiled artifact

PREPARE_API_RUN(run_id, P frozen API point values/artifacts, R-major/P-fast cell schedule)
  freeze R*P cell schedule; resolve/compile only P unique point programs
  validate all point settings、STATIC_ONCE schedules and canonical non-empty
    segmentation_rationale before the first FIRE

FOR_EACH_API_CELL(repeat_index, point_storage_index)
  open a new PulseSession for the already-frozen point-indexed artifact
  prepare through the existing API-slot path and verify prior cell terminal
  FIRE_API_SEGMENT(PreparedProgramRef)；hardware executes every edge in this finite segment
  COMPLETE_API_SEGMENT -> unique PulseTerminalAck(CURSOR=N/A)

SAFE/RESET/connection loss
  software invalidates PreparedProgramRef and follows current safe/reset path
```

PreparedProgramRef 是 pulse server 的软件 guard，不是硬件 one-shot token，也不证明某个外部设备接收了某条边。任何 prepare/upload/identity validation 失败都不得调用 FIRE；重连改变 `server_connection_generation`，旧 ref 失效。`PulseTerminalAck` 只证明该 pulse session 的可验证终态，不替 signal producer 制造关联证据。

compiler 可确定性展开每个输出通道的 edge schedule，包括 polarity、clock mux、相邻高段合并、channel delay 与全部 slot values；这些 facts 由外层 producer 在自己的 preflight/association 合同中消费。`zlc_pulse` 不把任一通道硬编码成 Camera trigger，也不解释 frame metadata。

SCAN_SLOT 正常产品路径使用当前 bitstream 的自主流式能力；完整逻辑表在 FIRE 前冻结，一次 FIRE 后所有微观时序由硬件决定。API-slot 只有在值不能无缝更新、且实验明确允许段间 gap 时使用既有 segmented 路径。不存在通用 host-stepped fallback，也不因架构偏好添加新 RTL。只有真机证据证明既定硬件行为有 bug，才进入 §15.5 的硬件变更门。

### 15.5 硬件安全

- baseline先使用host/compiler的typed range/capacity/slot/schedule validation，以及现有bitstream实际暴露的DONE/status/error/fatal/safe/reset回读；contract kit只声明真机证实存在且语义明确的位，不能把目标寄存器写成既有能力；
- upload/fire沿用当前已工作的UART/AXI/JTAG协议。host/server通过PreparedProgramRef、connection generation、artifact/table digest防止软件层旧程序误触发；若transport error或readback异常，禁止提交并按现有safe/reset路径处理；
- RemoteSequencer通过现有软件/transport能力提供bounded timeout、cancel/abort和safe调用；共享backend的第二socket不冒充硬件独立性。无法确认safe时当前run失败，但baseline不因此要求新增watchdog/SAFE寄存器；
- runtime identity近期只要求现有`image.build_fingerprint`/几何/ABI握手一致；installation-owned deployment record可以保存已批准`.bit`文件的content digest与release/timing记录作为SOP provenance，但它不证明endpoint此刻实际运行的内容。需要新RTL才能提供的runtime `design_build_id`、timing-signoff ROM或programmed-bitstream content attestation均不是baseline；
- 逐沿counter/FIFO、per-fire count、PHYSICAL_DONE、BANK_VERIFIED/RTL CRC等均不作为当前合同。只有相机经验时序、外部触发资格与故障注入在已批准工作余量、正确camera配置、finite完整`expected_frames`已分配并按合同排空时证实真实loss/reorder且非硬件替代方案均不能修正，或现有RTL偏离既定设计时，才提出与已证实根因有因果关系的最小硬件修复；
- 若未来合法重建bitstream，build仍必须满足unconstrained paths=0、WNS>=0、TNS>=0，并审查generated clocks、CDC、IP property和critical warnings；这约束未来修复质量，不授权为架构偏好重烧。

**2026-07-25冻结硬件只读审查的已知 NO-GO / 证据缺口：** 对`fpga/`、`zlc_pulse`、XDC、build Tcl、host model与部署geometry逐文件审查，工作树对这些路径保持零修改。以下各项均仍然存在、未在本设计中修复；它们是证明边界，不能被软件架构悄悄“补成已经存在的硬件能力”：

- 当前`fpga/board_config/board.xdc`与`fpga/pulse_streamer/create_project.tcl`未见显式`create_clock`、I/O delay等完整timing constraint；虽然Tcl含检查路径，仍必须由真机/build产物闭合timing约束证据。因此现有冻结bitstream可以继续按既有部署记录运行，但本仓库内容本身不足以宣称完整timing signoff。未来若因被证实RTL bug合法重建，必须先闭合这项约束证据，不能只看实现工具输出的单一WNS摘要。
- `fpga/board_config/streamer_config.json:32`的说明文字仍把formal host上限写成`2 * bank_size`的双bank驻留窗口，而README与当前host/image/RTL实现支持运行中ping-pong refill；正式软件以当前wire/image/RTL行为与host validation为准，不能把那条说明复制成第二个软件容量限制。
- `fpga/pulse_streamer/zlc_pulse_streamer_top.v:264`、`:400`、`:470`仍留有旧dense CTRL delay映射注释，而实际实现与host validation使用`R_DELAY + NUM_DELAY_CH`；旧注释不构成ABI。任何未来修改前必须先以host/model/RTL golden byte对照还原真实映射。
- `fpga/pulse_streamer/host/engine_model.py:109`的`zip(coeffs, slots)`没有strict检查并会静默截断，所以它不能单独证明长度不一致时的硬件行为；正式 compiler 必须在进入 model 前完成精确 cardinality 验证。只有 hardware owner 依据证据批准后，才在同一闭包修 model 与其 golden；冻结的 RTL/bitstream/XDC 上下文保持只读。

这些缺口不授权HOST_STEPPED、软件sleep边沿、伪造trigger stamp/ROM或主动重烧；它们要求发布结论准确区分“当前部署可按既有冻结合同使用”和“仓库已独立证明可重新综合并完成timing signoff”。

## 16. Artifact 与持久化

### 16.1 Typed Ref 与 manifest

各 bounded context 不共享 universal ArtifactRef class：

```text
data:          FitResultBatch payload（无 generic durable Ref）
frontend:      FigureArtifactRef
pulse:         CompiledPulseRef
neutral_atom:  CaptureArtifactRef、ScanArtifactRef、FitResultArtifactRef、CalibrationArtifactRef
```

typed Ref 的 class identity 提供 artifact kind/format 语义；持久 repository ref 至少绑定 repository namespace 与 artifact content/manifest digest，不是任意 filesystem path，也不直接嵌入 mutable Python object。只有 Ref 确实跨 wire、manifest 或独立持久化边界时，值语义 owner 才提供对应 typed codec。正式 lineage 同时验证更高层 source authority时，outer manifest/repository位于能够单向依赖payload与source两侧的最窄adapter；当前 `FitResultRepository` 的source union由Capture/Scan两个真实同构consumer挣得，但不得开放注册、复制payload schema或反向夺走算法所有权。Workbench通过本地ArtifactDescriptor adapters聚合展示，不把descriptor反向泄漏给owner。

repository namespace 在 composition root 由 ExperimentWorkspace 显式绑定到用户可见的 RepositoryRoot；不读 current working directory、session 最近文件或隐式搜索路径。virtual/real 运行使用同一个 Repository API 和目录布局，测试只把 RepositoryRoot 换成临时目录。UI/日志始终能显示 artifact 实际写入的 root/ref，offline 流程要求用户选择 typed Ref 而不是“猜最近文件”。

```text
ArtifactManifest:
  kind + current artifact format
  immutable metadata
  typed input/output lineage
  blob descriptors(digest, dtype, shape, byte_length, encoding)
  canonical manifest digest
```

shape、AxisSpec、PointLayout、validity、DataTransform/Fit/Scan contract 和算法/设备 fingerprint 都进入对应 manifest，不能只保存一个 ndarray 和文件名。

当前 artifact codec 不使用 pickle、object ndarray、FQCN import 或任意 callable 序列化；standalone persistence、独立 wire message 与 union-dispatch boundary 使用显式 format/discriminator，数组使用明确 dtype/endianness/order 的 blob encoding。若 outer artifact 的 exact field 已静态决定 embedded value 类型，该 subtree 不重复携带第二个 tag；outer typed reconstruction 与完整 re-encode负责 canonical admission。受信任本地环境不等于允许格式不可移植或靠导入旧 Python 类才能读取。

跨包 artifact 若嵌入另一个 package 拥有的值对象，必须调用 owner 公布的 canonical projector/parser，禁止在调用包重写字段、兼容 reader或 owner object digest。只有该 owner value 自己是 standalone/union-dispatch boundary 时才同时携带其 schema id；由 outer exact field 静态选型的 embedded subtree 只委托 owner tree映射，不复制 discriminator。例如 neutral ScanArtifact 嵌 ValueSchema/DatasetSchema/CommittedTransform 时委托 zlc_data codec；Workbench manifest 只包裹 owner canonical bytes + digest。每个值对象到 canonical tree 的映射只有 owner 一个实现。

canonical tree 到 bytes、UTF-8、map key order、整数/float/NaN 表示、ndarray dtype/endianness/C-order、framing 与 digest algorithm 则全部委托 `zlc_storage.canonical`，不能由 data/pulse/frontend/neutral 手工实现四遍。`zlc_storage.canonical` 不认识任何领域 schema/type，也不能 import repository backend；它只是 content-addressed storage 所必需的纯 bytes 规则，不是 universal ArtifactRef/common domain。codec round-trip 必须保持 AxisId、coordinates、dtype、native integer data 和 validity，不允许为统一格式把 uint image 全部转 float。

同一条 canonical primitive 约束也只有这里一个实现：canonical/non-empty text、lowercase SHA-256 text、finite real、integer lower bound 与 exact mapping/discriminator admission 由 `zlc_storage.canonical` 提供；领域 constructor 只保留物理/语义约束，codec 只声明字段集合并委托 owner。领域包不得复制 `_text/_sha256/_positive_int/_exact_map` 一类 helper；架构 AST ratchet 机械禁止重新引入。面向用户输入的 UI label normalizer 可以在 presentation/composition 层 strip 文本，但它必须用不同语义、不得承担 persisted canonical value 的权威校验。

无领域语义的 `ContentRef{digest,size}` 及其 schema-free current tree 只由 `zlc_storage.content_store` 拥有；Capture/Calibration 等 manifest 必须调用 owner codec，不能各抄一份 `{digest,size}` parser。领域 typed Ref 仍分别拥有自己的 repository namespace 与 `target_ref` 文法；recovery 从冻结的 expected manifest digest 构造 typed Ref 后比较完整 `target_ref`，不得手工切 prefix/slice。storage owner 的 `identify_blob(payload)` 可以在发布前计算 canonical 内容身份，供 metadata 引用；它不发布、不证明 durability，也不能被领域包用手写 `sha256(payload)+len(payload)` 替代。只有 `put_blob()` 才把 payload staging 到 CAS、核验并确认该 `ContentRef` 已可见。对 writable `bytearray/memoryview`，store 必须写入自己拥有的临时文件并按预计算 ref 重新校验后才能 atomic replace；若 replace 后的验证失败，必须删除目标并 flush parent，使读取保持 fail-closed，而不是留下 digest 与 bytes 不一致的可见对象。manifest 发布前遗留的不可达 blob 是安全 orphan、不是可见 artifact；它不构成自动 GC 或领域自算地址的理由。

必须建立 cross-package golden/property contract：同一 primitive tree 在四个 owner 包中产生 byte-identical encoding/digest；嵌入 owner value object 时 outer manifest 使用 owner bytes/digest；字段重排、float edge、NaN、unicode、ndarray order/endianness 与版本变化均有向量。golden 不是允许四份实现漂移的补救，而是守卫唯一 encoder 和 owner codec delegation。

### 16.2 Atomic commit 与 load

各 owner context 的 typed Repository 委托 `zlc_storage` 的同一个 `BlobStore/ManifestCommitter` 实现 immutable content-addressed bytes、锁、fsync 与 atomic replace；owner Repository 仍负责 typed Ref、schema、canonical codec、lineage 和 load validation。`zlc_storage` 不 import AxisSpec、FigureArtifactRef、ScanArtifactRef 或任何领域类型，也不提供“万能 artifact repository”。commit point 是最后原子发布的 owner canonical manifest：

`CommitJournal`使用`zlc_storage.FramedJournal`保存artifact发布意图与COMMITTED/ABORTED解析。frame使用canonical bytes、稳定record id与SHA-256；append在跨进程文件锁内验证状态、写入并fsync。它只服务artifact visibility与crash consistency，不记录设备状态，也不参与硬件连接admission。

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

每个 manifest atomic replace 是该 artifact 的 commit linearization point；Run 声明的 final result manifest 是其 SUCCEEDED commit point。Repository在replace前已有durable CommitIntent；CancellationToken在intent之后、final replace之前最后检查：此前取消则删除final temp、不发布final manifest并把intent标为ABORTED，进入CANCELLED/FAILED(cleanup error)；replace成功后final artifact已是事实，后到cancel返回`TOO_LATE_ALREADY_COMMITTED`，Run完成SUCCEEDED并可记录late-cancel warning，不能删除artifact后谎报取消。replace返回异常不等于未提交，必须由owner Repository按commit_id检查最终manifest/digest后才能写COMMITTED或ABORTED。更早独立提交的upstream artifact遵守自己的commit point，不随父Run回滚。

最终artifact只有在本Run所有必需领域session都已成功`close_session`、workers已join且数据合同通过后才可发布。`CommitIntent`、manifest replace、COMMITTED-or-ABORTED resolution与Run terminal是有顺序的linearization points；startup按`commit_id + target/manifest digest`执行确定性reconciliation。设备cleanup诊断不进入CommitIntent。

content-addressed blob 允许并发 writer 幂等复用；manifest publish 使用 digest/id 冲突检查，不能覆盖不同内容。只有 repository 规模证明 unreferenced blob 回收是实际问题、且所有 owner 能提供已验证 committed-manifest roots 后，才增加 maintenance-lock 下的 mark-and-sweep；storage 不自行解析产品 manifest。这样 baseline 先共享崩溃安全机制和 canonical bytes，不为尚未出现的多 backend/复杂 GC 建一套存储平台。

每种 artifact 只按自己的合同判断 commit。显式“先采raw、以后再分析”的 Capture/Calibration workflow中，完整CaptureArtifact是独立上游事实，后续calibration失败不回滚它。PulseScan却是另一个用例：已运行producer交付的associated signal group在本Run内只是scan输入，用户请求的唯一成功结果是canonical ScanArtifact；transform、association、pulse terminal或scan commit失败时不额外发布一个名字像成功scan的raw CaptureArtifact，也不创建第二条recover/promotion历史。producer原本独立发布的monitor/processor输出继续遵守自身生命周期，不因scan成败回滚。若用户确实需要独立raw artifact，必须作为另一个显式Capture Run请求，而不是scan内部副作用。

当前Scan application使用一个flat Run完成`producer association -> pulse FIRE/terminal -> committed transform -> ScanRepository FINAL`。ScanRepository是唯一scan dataset authority，manifest保存owner-encoded logical PulseDocument与compiled pulse blobs、exact PulseTerminalAck、producer association evidence、ordered input EventRefs、processor stages、source/output schema、ScanOutputContract、canonical output DatasetRevisionRef及values/validity blobs。output BlockId由logical document、source generation与association identity、output contract共同派生；final ScanArtifactRef由实际values与provenance内容寻址。没有ScanIntent、raw Capture promotion、`promote_scan()`、旧格式reader或两份manifest真相源。

blob staging仍可留下不可达安全orphan，但只有同一Run的`context.commit_final(ScanRepository.final_commit(...))`能发布成功manifest。publish lost-ack由RepositoryCommitCoordinator按稳定commit_id、target和manifest digest reconcile；artifact已经可见则客观返回同一成功ref，未可见才失败，绝不重新FIRE或退回raw promotion。virtual与real使用同一flat commit；差别只在producer-owned association evidence的类型与资格。真实qCMOS把CameraExternalTriggerQualification、工作点和逐run对账封装进Camera evidence，不重建第二套scan repository或workflow engine。

RunFailureRecord 记录 run_id、request/plan digest、最后 phase、primary error、cleanup errors、当前resource claim与已成功提交的独立 upstream refs/event spans。它是诊断记录，不满足任何 Capture/Scan/Calibration result Protocol，不能被下游当数据输入，也不能证明设备在未来连接中的状态。

### 16.3 Live Ref 与 Figure 保存

系统自动选择的用户输出目录使用`zlc_storage.paths.user_output_path(...)`，位于`<project>/_output/`而不是可编辑输入目录；普通Figure使用`user_output_path("figures", <product>)`作为首次Save/Export目录，`pulses/`只保存PulseDocument等输入，不接收Pulse preview PNG。Task自己的可编辑folder字段由领域Authoring声明并统一经`resolve_under_project(...)`解析：Calibration默认`_output/calibrations`，相对路径锚定项目根而不依赖进程CWD，显式绝对路径按用户选择使用。不能把`user_output_path`与`resolve_under_project`写成同一个API或让GUI另算第三套路径。

路径统一不等于durability统一：Calibration Task的`report/frames/calibration_ref.json`属于operator output projection，不进入`CommitJournal`。它只遵守上述pointer-last与同进程rollback；只有Calibration/Capture repository的CAS manifest/ref可以使用本章的crash-safe artifact措辞。

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

FigureArtifact 保存 ViewSpec、当次 EvaluatedFigureData 的 input revision/resolution records、Selection snapshot、layer/model/fit lineage 和所引用 dataset digests；重开默认复现保存时的 concrete selection。用户明确切回 repeat 的 `LatestNonempty` display policy 后，才重新解析最大非空逻辑 repeat；这仍不等于恢复最后发布事件，也不进行新的 axis auto 推断。若用户只保存 workspace layout 而不 materialize live data，文件必须明确标为 session-only workspace，并在数据 lifetime 结束后显示 missing binding，不能假装是自包含 FigureArtifact。

Finite-preview component 只保留进程内 final MonitorDatasetSnapshot，供 panel 关闭前继续显示；该 surface 不提供 Save/FigureArtifact materialization，因此不能把 final slot 当作可重开 artifact，也不能声称用户能够保存 live 所见 revision。

### 16.4 唯一格式名与重跑策略

正式runtime、authoring load/save、wire和全部artifact只接受各owner的一个当前格式名。真正长期落盘、跨会话读取的值保留朴素、无改稿序号的格式名（例如 `zlc_pulse.PulseDocument`、`zlc_neutral_atom.calibration-artifact`）；临时进程内摘要只保留用途明确的 domain separator。未知格式名清晰失败，不存在版本比较、旧 reader、upgrade chain、转换 CLI 或 GUI fallback。

不符合current软件artifact格式的标定、capture或analysis输入不属于产品读取合同；需要相同实验事实时重新采集/重跑，不维护档案转换器。只有已部署且被硬件/外部协议真实消费的wire/ABI结构版本可以保留独立版本号；该例外必须有双端consumer和部署证据，不能由软件改稿次数推导。终态allowlist只有FPGA `LAYOUT_STRUCT_VERSION=3`与已批准部署拓扑的`zlc_pulse.PulseTargetABI/v1` hash domain；普通PulseTarget/PulseDocument/RPyC artifact格式不在例外内。`ZLC-CANONICAL-1\n`字节前缀是生成后者获批digest的冻结canonical hash原语，不是第三个可协商格式身份；它必须随该ABI一起保持逐字节不变。仅改变软件格式名不得重签这些硬件事实。

控制进程与 FPGA server 的当前 RPyC payload 是同一软件 release 的一个协议闭包，必须原子部署；字段或格式不一致时 fail closed，不提供 mixed-release reader、协商或 fallback。该部署约束不把软件 payload 的改稿次数提升为硬件 ABI。

## 17. 性能约束

### 17.1 Camera event ownership

CaptureSession 的有序record容器禁止 list `pop(0)` 的 O(n²) 路径；使用摊销 O(1) 的append/popleft结构，但不设置软件容量上限。formal事件在ack前自然保留，monitor `next()`有序交付且不覆盖；exact与monitor fan-out共享immutable payload/ref，不复制不必要的大帧。内存分配失败使本次run明确FAILED，不能在压力下静默减少数据。

finite driver ring 的`buffer_frame_count`固定为完整`expected_frames`；continuous monitor固定为`history_cycles * frames_per_cycle`。exact transport、DatasetBuilder、monitor与renderer各自明确谁持有引用、何时释放以及失败时如何撤销。真实profile若发现重复快照或不必要的长期引用，应修正ownership/copy路径，但不得据此重新引入软件预算、拒绝或隐式覆盖。

### 17.2 Scan compile

使用：

- expression 预编译；
- typed contiguous arrays；
- vectorized expansion；
- 一次性 validation；
- source document 与 wire artifact 分离。

优化后 target IR/wire image 必须保持等价，时间与实际存储增长对点数近似线性。

Formal Scan在FIRE前冻结完整logical table与全部physical chunk；当前冻结RTL的双bank只是流式执行窗口，不是scan点数上限。FIRE后唯一transport observer按硬件`BANK_READY`/`CURSOR`事实补写已释放bank，FPGA仍独自决定全部edge时序。每次re-arm后立即读取`STATUS/CURSOR`：看见任何`UNDERFLOW`/error，或cursor已经跨出refill开始时所在chunk，都说明无缝边界无法证明，整run作废并进入SAFE，后续DONE不能洗白。qCMOS finite driver ring仍按完整`expected_frames`配置；真实SDK/系统分配失败作为本次具体I/O失败直接传播，不建立另一层预设拒绝策略。

### 17.3 UI 与 analysis

- monitor 由 UI refresh rate 限制，不降低 acquisition rate；
- 独立 panel view-evaluation latest-only；同一 coherence group 由 board evaluator 选同一 JoinKey/revision并原子 present，尚未开始的display/fit请求由更新revision替换而不积累历史队列，三者不互相阻塞；
- revision coalescing；
- stale fit/render result 丢弃；
- exact DatasetBuilder与capability-owned association signal source不与可丢弃UI fit共用一个拥塞队列；
- `suggest_view` 只遍历 axis metadata，复杂度 O(axis count)，不触碰大型 values；
- ViewSuggestion 可按 `(schema fingerprint, ViewIntent, Selection revision, preference revision)` 缓存；
- 连续 Selection 优先产生只读 view，显式稀疏选择只按实际输出 gather；只有 reduction、driver buffer ownership 或持久化边界才复制；
- 显示用 mean/latest 不复制或覆盖权威 DataBlock，缓存键必须包含 input revision 与 ViewSpec digest；
- EventSpanRef 的 count/ordered_digest 随 exact sequence 增量更新，不为每个累计输出复制历史 event_id。

display decimation/downsampling 只能作为带标签的 render policy，不能改 DataTransformSpec、FitSpec、ScanOutputContract 或 artifact data。用户缩放/导出时 renderer 从同一原始 snapshot 重新取样，不能把屏幕像素缓存当权威数据。

### 17.4 Profiling 与性能 gate

性能结论必须来自相同 workload 的 profiler/benchmark，不用刷新频率下降或数据丢弃掩盖瓶颈。每个优化记录：baseline、调用图/分配热点、改动后结果、数值/视觉等价证明。

固定 benchmark matrix 至少包含：

- camera 不同 frame bytes、repeat、burst 与 exact+monitor fan-out；
- 1D/2D/multi-axis scan 的 P 扩展；
- materializer 原子提交与 immutable snapshot 的时间和 peak RSS；
- capability-owned Processor prepare/evaluate、derived signal fan-out、typed record bytes 与 DatasetBuilder materialization；
- FitResultBatch 的 batch size 与 model cost；
- artifact streaming write/load 与 digest 校验；
- TaskConsole `WORKER_RASTER_LIVE` 多 panel board 的 ingest-to-visible、compose/present、GUI event latency、coherence mismatch 和 stale queue length；不得以回到 GUI compose 换取结构简化。

机械 gate 使用 scaling，而不是拍脑袋的单机绝对秒数：queue/materializer commit 摊销 O(1)，scan compile/journal/artifact数据量近似 O(N)，已 ack history 不得被无界保留；p95/p99 latency、peak RSS与copy数量作为profiling观测值记录，不作为运行准入条件。出现回归时保存profile artifact，先定位 producer、copy、lock、solver 或 render 热点再决定优化层。

## 18. 测试体系

### 18.1 Package tests

各 bounded context 拥有自己的 unit/contract tests，根仓库只保留 architecture、cross-package integration、E2E 和 performance。

### 18.2 必须保留/新增的合同

Data：

- Value event 只携带 `(*data_shape)` 与 ValueSchema；DataBlock 只携带 `(R,P,*data_shape)` 与 DatasetSchema，普通 stream edge 拒绝 DataBlock；
- 每条 edge 恰有一个 event -> dataset owner：finite exact DatasetBuilder 验证完整 TriggerKey/ScanCellKey schedule、missing/key mismatch、ValueSchema 与原子写；live MonitorDataset 验证 keyed cycle 或按 sequence 管理 append window，二者不共享 mode/state machine；
- `PreparedOccupancyProcessor.evaluate`返回一个typed atomic evaluation；其counts/occupied/rate共享source revision/event digest与join lineage，`OccupancySignalValues`的三个字段共享direct EventRef/provenance；字段不同cardinality/key/lifecycle时capability静态拒绝并拆产品节点；
- 任意 point/data axes；
- scalar 与长度一 axis；
- arbitrary-schema property tests：AxisId 唯一、coordinate/size、shape 与 axis coverage 不变量；
- 不同 revision/generation domain 不能互相比较、赋值或通过裸 int 混用；
- 同一 Definition 的 virtual/real/不同 run AxisId 稳定；Selection 保留 AxisId，Reduction 只移除 axis，baseline 不制造匿名派生 axis；
- PointLayout RECT_C/RECT_F/EXPLICIT sparse mapping round-trip，public path 不假设 P=product 或自行 reshape P；
- ValueSchema/DatasetSchema fingerprint 分离：前者包含 data axes、dtype/unit、ValidityContract，后者另含 repeat/point axes 与 PointLayout；两者都不包含 renderer、ViewIntent 或已安装 reducer 列表；
- canonical unit string 与 CoordinateFrameId mismatch 必须拒绝；baseline 没有隐式或通用自动换算；
- 多轴 ROI/integrate contract 同时验证 input axes、output axes、unit 与 validity；
- 不支持的 reduction method、axis 不兼容或把需要 CalibrationArtifact 的领域 reduction 伪装成通用 reduction 时失败；
- native uint/image + partial validity；
- Value 的 ComponentValidity、DataBlock 的 CellValidity/DatasetComponentValidity 及其具名 axis 广播；`(group,site)` dead-site mask 在 reduce/fit/histogram/meter 中一致传播；
- validity mask axis/shape 不匹配失败，NaN 不能替代 integer/bool/component validity；
- 发布 DataBlock/array write-protected，driver buffer reuse 不改变已发布 snapshot；
- exact 提交在 metadata/key/admission 失败时不写 cell、不推进 ack/revision，成功时 values/written/validity/metadata/ordered hash state/counters/revision 与 ack 绑定；live snapshot 的 head/coverage/EventRefs/block 来自同一 revision；
- materializer/snapshot 近线性且无每点全 block copy；
- finite exact 每 sample 只发 DatasetProgress/dirty coverage；live 只发可 coalesce 的 revision 通知并按需取得同一原子 MonitorDatasetSnapshot；两者都不把完整 DataBlock fan-out，EOS final freeze 总复制近似 O(final bytes)；
- 非法隐式 reduce/anonymous flatten 失败；
- DataTransformSpec 不包含显示 binding，ViewSpec 不可传入 neutral runtime；
- 同一 schema + ViewIntent + Selection 得到确定性 ViewSuggestion/ViewSpec；只改变等语义 data-axis tuple 排列不改变 AxisId→binding 决策；合同有可行 batch/facet 分配时不能因贪心次序误拒；
- suggestion 不读取 values，不按 rank/singleton/axis 顺序猜 role；Selection range 不自动变成 reducer；
- 多 point-axis 的自动 FixedIndex 来自同一个真实 PointLayout storage row；EXPLICIT hole、Selection 后无物理 row、手写/解码得到的不存在 fixed tuple 均失败；
- suggestion 不修改 DataBlock，所有有损 binding/operation 均可从 ViewSpec 派生并出现在 panel 摘要；
- FixedIndex/LatestNonempty navigation 每个 input revision 都解析为带 coordinate record 的 EvaluatedFigureData；LatestNonempty 只允许 repeat 且只表示最大非空逻辑 index；display navigation 不能进入 CommittedTransform；
- rolling replacement/wrap 的 current cell 由 EventRef/progress 驱动、覆盖 repeat 与全部 point axes 的显式 Selection 给出；最高 nonempty axis index 不得冒充最后发布事件；
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
- FitSpec/BoundFit/FitResultBatch/Selection 与 `BoundFit.run()` 只在 zlc_data 定义；DataFigure/selector controller 只在 frontend 定义；neutral_atom 中任何 `FitProcessor`、`FitOperator`、`FitAnalysisDefinition` symbol 或复制 solver/model schema 都由 architecture test 拒绝；
- zlc_data `BoundFit` 不引用neutral runtime slot或artifact owner；当前不存在neutral-side generic AnalysisStep/slot adapter，artifact Fit由现有repository host，领域Analysis由各自typed request编译flat RunPlan；baseline不存在DataAnalysisDescriptor/Program registry；
- zlc_data solver 不含 Qt/thread guard 注册、executor、环境变量线程策略或 callable/FQCN 序列化；frontend/neutral 各自在 hosting contract test 中证明 fit 不运行于 GUI/I/O lane；
- ViewContract 只有一套，plot/render 不复制 role/repeat 判断表；
- TaskConsole只消费composition root逐项注入installed context/preparer/loader后的`LogicNodeDeclaration`投影；`TaskConsoleApplicationPorts`对重复DefinitionKey启动即失败，禁止第二份definitions、package scan/global registry；fields/path hints/dynamic choices/inputs/outputs/default views全部由同一个generic projector消费，只有真实特殊产品面才允许独立inert UI leaf；
- import-DAG ratchet按路径扫描：除`logic_nodes/*/ui/**`外的neutral graph（含declaration与`workbench_adapter.py`）禁止frontend/workbench/Qt/Matplotlib；capability root不得eager import adapter/UI，`ui/__init__.py`必须inert；generic frontend/workbench不得导入concrete capability，composition不得出现field key/output/default-view/UI-policy literals；
- frontend是SiteMap view/Area/render/selector唯一owner，neutral/workbench出现同义presentation/projector即失败；每个reactive Processor publication在进入Workbench前验证source ref/event digest、generation/revision、single join digest与全部sibling outputs的atomic causal closure；
- `DEFAULT_PANEL_SIZE`只有一个赋值；ordinary constructor/default parameter不得重写`"2x2"`。真实窗口验证普通Panel/PlotPanel/DataFigure默认2x2、Grid在facet边界消费同一topology策略、Pulse只消费document topology策略，且用户或archive size不被后续revision覆盖；
- 真实Calibration FINAL `site_map -> TaskConsole default sites card -> freeze_render_request -> PlotPanelSession`与Calibration report的相同logical contract逐字节比较raster；report更多像素只能来自export pixel ratio，Calibration leaf中不得出现composer/renderer/style owner；
- presentation publication反例覆盖N visible/N+1 candidate、缺失或冲突sidecar的双侧零改变、exact N+1 promotion及withdraw；producer replacement/remove覆盖Processor、多级selector/Fit后代、candidate/source-component/sidecar的传递退休，成功与in-flight Fit均立即解除pin/busy/overlay，打开的Edit也清overlay，迟到completion不能复活旧generation，同名replacement可直接显示；
- Qt Setting在resolved且无可编辑axis时零行并隐藏；有歧义时只出现具名、可编辑AxisId行，disabled `Reduce/ROI X/ROI Y`假字段为零；
- catalog Definition 不含 callable；owner top-level binder/operator 无 hidden closure/device/session/global mutable dependency；
- 每个Measurement/Analysis capability只编译自己的唯一顶层flat RunPlan，节点不能start child run或自行拥有terminal state；TaskConsole Processor prepared application不冒充RunPlan；
- bind claim superset 完整，preflight/execute 尝试新增 ResourceKey 失败；
- 同一PhysicalDeviceIdentity在Workbench/notebook/standalone/remote入口间只有一个installation authority和一份backend可验证physical-owner proof；两个进程各自的ResourceArbiter不能同时把本地EXCLUSIVE冒充成同一物理设备的跨进程所有权；
- TaskConsole、PulseGUI、Experiment/session与standalone real入口均拿不到raw device drive verb；其它owner持有重叠claim时，从每个公开入口尝试camera acquire或sequencer prepare/fire都被同一authority拒绝；
- production中不存在绕过RunController的直接LogicNode.start或平行启动入口；AST/行为测试证明所有硬件Run只经唯一RunCommandPort/RunController取得真实ResourceClaim，thread/session未退出或SAFE未确认前不能发布terminal、释放claim或越过shutdown；
- 改变device/config/virtual-real只产生reconnect-required并请求同一个InstallationRuntime shutdown；与并发start线性化后新start为零adapter调用拒绝，console外handle和target Run同样由唯一RunController cancel/join并走各领域既有close_session，旧connection关闭前claims归零；原进程内不得构造或发布replacement graph；
- DeviceManager从非Qt worker请求shutdown-for-restart，Run终止、领域session收口与adapter close仍完全由InstallationRuntime完成；GUI只在Qt owner thread消费一次完成delta，event loop阻塞、QWidget callback失败或窗口已销毁都不产生跨线程QWidget调用；
- 对startup的journal lock、physical-owner proof、adapter open、identity、AssetMap、broker bind、capability probe与graph freeze逐点故障注入：Run admission始终未开放，已打开的exact owned subset按reverse close order关闭，绝不发布partial Experiment/catalog/drive facade；
- 对shutdown的run join、每个领域`close_session`、broker invalidation、每个adapter close、lane stop与physical-owner release逐点故障注入；旧facade先摘除，失败只形成本次close诊断，下一次连接重新做live SAFE初始化；
- Processor capability/Analysis算法不读 global RNG/time/config，显式 seed/config 可重放；
- BEST_EFFORT_MONITOR 只能是 monitor 叶子，其失败不 abort exact，且不能流回 authority/artifact；
- frontend FigureDocument/codec 不引用 neutral LiveDataBlockRef；
- frontend EvaluatedFigureData 与 workbench render message 都只持普通 owned immutable 值，不携带自定义 release token；

Stream：

- reservation 在 fire 前；
- history=8 + burst=20 在 reservation 下完整消费；
- formal consumer无reservation时在读取前拒绝；
- schema generation 改变终止旧 cursor；
- monitor rebind 新 generation 创建新 block_id，旧 evaluation/borrow/CommittedTransform 不可复用；
- 同一 generation 第二个 formal reservation 被拒；一个 exact DatasetBuilder + monitor fan-out 共享immutable payload/ref而不重复复制；
- finite camera arm把`buffer_frame_count=expected_frames`传给driver；continuous monitor传`history_cycles * frames_per_cycle`；SDK/系统内存真实分配失败使本次操作失败；
- ack 后已释放前缀不再由stream持有；slow consumer的未ack集合自然增长，不触发软件配额拒绝或覆盖；
- candidate Envelope/contract/key/timestamp 验证失败时不先释放已有record，rejected publish 对 stream 状态零副作用；
- broker-minted generation 使两个同 StreamId/BlockId/schema、不同内容的 capture 得到不同 DatasetRevisionRef；
- PayloadContract统一snapshot/validate，ComponentValidity mask与per-event schema metadata均必须接受同一owner校验；
- exact Delivery 必须属于 builder 绑定的具体 source+reservation；同名 source、伪 cursor/Delivery/EOS、跨 tap MonitorUpdate 均被拒；
- frozen sequence->cell schedule 阻止合法 key 的 row swap；TraceBinding 阻止同一 reservation 混入另一 run/source；
- DatasetPreviewSnapshot 不能进入 formal storage/authority processor，只有 SealedDatasetArtifact 或 VALID epoch wrapper 可以；
- DatasetBuilder 异常退出统一 abort+release，不覆盖 body error或泄漏 formal claim；
- stale DeviceCapabilitySnapshot/output schema mismatch 在 arm/fire 前拒绝；
- Processor capability在prepare时无法冻结output vocabulary/validity/lineage时直接拒绝；`start_signal_events`无法证明严格1:1或含未解释filter时不得传播association；
- Processor output contract依赖首帧数值或evaluate期间改变record fields/axis时拒绝；TaskConsole不得补写schema，PulseScan不得把该source当formal association；
- continuous Measurement 只能使用 admitted MonitorTap/MonitorDataset；exact request 必须有限且可完整 reservation；
- MonitorTap `next()`有序交付且不丢record；MonitorDataset按声明`history_cycles`维护rolling data cardinality并只产生provisional atomic snapshot；若要成为formal input，必须显式冻结新的finite diagnostic input或启动finite exact capture，不能给live snapshot改名；
- 单个 typed output record 的 ack 只在 publish 与所有 required downstream 接收成功后；不同 cardinality/key/lifecycle 的结果必须建独立节点，不能伪装为一次多-output transaction；
- exact/monitor 同源 event_id；
- EventSpanRef digest/count 等价于显式 ordered events，lineage 不随累计输出 O(N²) 增长；
- driver buffer 重用与MonitorDataset正常滚动不破坏formal payload lifetime；
- `CameraFrameRecord` 构造即拥有并冻结图像 bytes；driver ring slot、原 ndarray 和 metadata wrapper 随后改写都不改变 record，非整数/bool ordinal、负值、非法 microseconds 和无效 host receive time 在进队前拒绝；
- qCMOS fake/contract kit 验证同一 `buf_getframedata` 的 frame/camera stamp、timestamp、driver buffer index 与同一 drain 观察点的 `nFrameCount` 原样进入 record；同一batch的 `produced_count` 允许重复，不被改写为伪逐帧counter；
- `read_frame_records()` 是唯一armed-session读取入口，不存在array-only平行reader；finite count越过`expected_frames`或发生duplicate/gap时在原子发布前返回`CameraBufferOverrun`，不部分保留该batch；
- neutral JoinPolicy标记为coherent的monitor publication必须按声明key完成物理join且永不混shot，`INDEPENDENT_LATEST_MONITOR`不可用于相关expression；Workbench linked-front仅做已闭合publication之后的presentation revision gating，不能替代或证明这条合同；
- EXACT_KEY/coherent monitor 对 join_key type/schema mismatch 或 missing key fail closed；
- 独立设备 ZIP_SEQUENCE 被拒绝；
- `SignalEventSource` 与 `SignalEventAssociationSource` 的能力必须分开测试：前者只保证有序future事件，不能启动formal scan；后者必须在FIRE前arm、绑定exact `PulseTerminalAck`、交付恰好N个事件并finish为canonical evidence；
- associated source可声明source-neutral的compiled edge/grouping requirements；compiler必须在FIRE前合并并验证这些requirements，既不能漏掉producer观测所需通道，也不能在PulseScan写Camera类型分支；
- PulseScan绑定到已RUNNING且其设备已完成arm acknowledgement的精确producer instance/generation；在该ready边界前打开associated cursor必须typed reject；成功、失败、cancel后producer的node identity、RunId和RUNNING状态不变，scan Stop只关闭自己的sequencer/collector；
- virtual readout Camera的association由simulation apparatus实际观察到的FIRE、trigger channel、trigger group与frame ordinal区间证明；FREE_RUNNING mot camera明确不提供该能力；
- Occupancy只有在其upstream提供association且分类是严格1:1时才透明传播；使用独立upstream associated cursor，不与后台monitor cursor竞争，每个输入恰有一个输出，并把upstream evidence、EventRef、processor stage与CalibrationArtifactRef全部写入lineage；
- Workbench的Run/Processor node只能按协议透明委托association；没有Camera/Occupancy/PulseScan分支，也不得从display revision、panel raster或普通顺序自行铸造证明；
- repeat/point schedule确定性展开成`R×P` logical rows；collector把每个事件写入完整`(*data_shape)`，scalar仍为`(1,)`；Value 的 ComponentValidity 写入 DataBlock 时转换为 DatasetComponentValidity，所有surviving axes原样保存；
- 注入gap、duplicate、short group、wrong terminal、wrong generation、schema变化、证据digest不匹配、processor非1:1和非法transform时，FIRE前能发现的必须FIRE前拒绝，其余整run失败且无ScanArtifact；
- Camera→PulseScan及Camera→Occupancy→PulseScan从正式TaskConsole产品入口执行，且测试不得用test-only association owner帮助成功；自环output、static/display-only/latest与无association能力的signal不得成为可提交binding；
- SCAN_SLOT证明完整表在FIRE前冻结、一次FIRE、硬件自主时序；API_SLOT只测试已经存在并显式允许段间gap的segmented例外；两者都没有通用HOST_STEPPED fallback；
- real qCMOS的CameraExternalTriggerQualification合同独立验证一触发一帧、delivery order、frame/camera stamps、counter reset/rollover、工作余量和逐run produced/drained/coverage对账；未接入production composition或任一证据不符时只拒绝该producer的formal association，不把PulseScan改成Camera owner；
- 系统不声称能检测metadata仍合法的等量loss+extra抵消，也不声称逐沿硬件tag；该剩余风险必须记录在real Camera association evidence中。HardwareTriggerStamp/FIFO/trigger-return/新ROM只有证据批准RTL修复后才出现。

Thread/UI：

- blocking I/O 中 cancel；
- out-of-band interrupt 可使被占 I/O lane 进入 cleanup；
- cancel先置token；若硬件调用已in-flight则调用一次out-of-band interrupt，等待其退出后仍只调用一次领域`close_session`；
- interrupt in-flight是terminal barrier，cleanup不并发碰同device、claim不释放、迟到异常进入CleanupReport；
- join timeout 不允许 restart/release/destroy，safe failure 转 ResourceBusy；
- synchronous run 的 RunStillCancelling 保留可查询 RunHandle/claims；
- active Run内不透明reconnect；旧runtime关闭并摘除后，新runtime必须重新验证live identity、physical owner与当前SAFE，旧close结果不能代替或阻止该验证；
- 新 connection generation 在 UNVERIFIED handshake 完成前不可 acquire，应用重启不洗白 sticky fatal；
- active Run内transport断开不透明reconnect；普通重连要求当前runtime完成shutdown并由新进程产生新generation；旧run cleanup不能借新generation readback冒充旧generation已终止，也不写persistent blocker/quarantine；
- startup open/identity/AssetMap verification/broker bind在Run admission开放前完成，不创建ResourceArbiter connection lease；任一步失败时普通硬件调用次数为零、partial graph不发布且已开子集被安全关闭；
- `VerifiedPhysicalDeviceIdentity`不可变且只能由DeviceBroker握手mint并在bind时一次消费；成功后唯一长期事实是`DeviceBindingStamp(PhysicalDeviceIdentity, binding_instance_id)`。同一握手结果复用、同一PhysicalDeviceIdentity绑定两个ResourceKey、同key二次bind或静默换physical identity全部拒绝；
- identity evidence明确区分HARDWARE_IDENTITY_READBACK与INSTALLATION_ASSERTED_ENDPOINT；后者保存endpoint/AssetMap revision与剩余换板风险，不能在qualification evidence/artifact/UI中显示成硬件serial readback；
- 真实runtime缺失AssetMap、map revision不是canonical内容digest、exact adapter kind/expected matcher不符时composition拒绝；新进程+新broker下把同role换成另一serial仍拒绝，只有旧runtime完成safe shutdown并退出后的显式offline maintenance可更新map；
- 每个领域的`SessionClosedAck`只能由该领域`close_session`产生并绑定当前session；设备A的ack、普通command返回或缓存布尔不能替代设备B的终态readback；
- `safe_requested`、command return或本地cache都不能替代 sequencer 的实际 SAFE/readback；相机则由自己的 stop/drain/no-more-frame/join 合同终结，不强塞成通用SAFE模型；
- Run开始前核对当前`BoundDevice.binding_stamp`并取得进程内claim；cleanup只依赖当前session真实终态，不写跨run设备历史，也不因上一次失败阻止下一次真实preflight；
- cancellation仅为停止in-flight调用使用一次已声明的out-of-band interrupt；正常cleanup不重复interrupt，随后每个领域session恰好进入自己的`close_session`；
- session/worker/in-flight interrupt全部退出后才撤销本Run capability并发布terminal；任何必需session关闭失败都阻止成功artifact、令当前run FAILED并保留本次诊断；
- RunHandle只在worker/session/interrupt真正退出后发布FAILED/CANCELLED/SUCCEEDED并释放当前进程claims；
- cleanup后只把executed facts交给无device Port的finalize上下文；注入旧session/closure或late cancel硬件调用必须得到CapabilityRevoked且调用计数为0；
- raw SDK/driver只在allowlisted owner lane构造和保存；RunPlan/Definition/finalize的对象图、global、container与bound method均不存在driver或可直达driver的callback，验收不以closure introspection冒充隔离；
- cancellation在CommitIntent fsync期间仍可受理；intent后取消写ABORTED且publish调用次数为0；manifest replace确认丢失、COMMITTED marker确认丢失和Repository暂时不可达均保持非terminal/claim且不重复publish；startup用`commit_id + CommitTarget/manifest digest`把pending intent唯一解析为COMMITTED或ABORTED；
- `CommitAuthority`只能由startup-reconciled RepositoryCommitCoordinator签发，是不含public publish/journal/recover的无副作用opaque handle且单次消费；直接发布、替换payload、重复/跨run消费、ephemeral journal生产签发与绕过startup pending gate全部拒绝；错误PublishedManifest类型/target/digest直接ABORTED且recover调用次数为0，只有typed PublishVisibilityUnknown进入recover，recovered PublishedManifest仍须再次匹配target/digest；
- commit reconciliation三态不可反转：wrong digest + abort-marker failure仍FORCE_ABORT且recover为0；visibility recovery已判uncommitted + marker failure仍FORCE_ABORT；validated publish/recovery + commit-marker failure仍FORCE_COMMIT且不再调用recover；
- crash发生在commit intent、artifact manifest与commit resolution相邻边界时，startup只恢复artifact可见性事实，不重新fire、不把temp当成功，也不推断设备当前状态；
- terminal publication与当前进程claim释放对竞争acquire线性化；新进程或新连接重新执行领域live handshake/preflight；
- remote endpoint的物理命令仍由硬件server唯一串行化；client的本地状态不能替代server当前generation与实际readback；
- schema-affecting reconfigure 建新 generation/block_id，旧 cursor/view/fit terminal；每个 accepted ControlTopic revision 恰有一个 terminal ack；
- 同一硬件owner存在monitor时，finite run admission必须先请求冲突owner停止并等待真实terminal；验证普通command mailbox不设置queue/pending/backlog容量策略，Qt draft不入队，driver call超时后不能仍在后台碰硬件，interrupt不经普通command路径；
- stale queued result 不更新 UI；
- retained revision N 的 OwnedSnapshot 在 builder ingest N+1 后 digest/bytes 不变；mutable builder read-only view被 contract拒绝；SnapshotExpired 不返回 latest；
- 所有显示路径都只持普通 owned immutable value，不建立自定义 lease/pin/release 生命周期；
- QObject affinity；
- 只有`WORKER_RASTER_LIVE`与`FROZEN_RASTER`两种surface；所有Agg Figure/artist graph都由同一个非Qt frontend presentation执行线程（application worker或notebook调用线程）创建、使用并释放，普通queue不跨线程传QWidget/Figure/artist；
- production中不存在共享Figure handoff或兼容Agg bridge；GUI不接触worker-owned Figure/Canvas/artist，frontend presentation session必须在同一执行线程创建、使用、释放完整Agg graph，shutdown等待session release退栈；
- frontend presentation session 对 QObject-affine draw/update/connect/selector API 机械拒绝；只允许会话自己创建并在同一执行线程释放的 FigureCanvasAgg/Agg-only path；
- interactive transform/reduction在FigureEvaluator worker；`WORKER_RASTER_LIVE`只向Qt交付revision-matched immutable BoardFrame/front raster，`FROZEN_RASTER`只交付encoded page/bytes；
- 长 interactive fit 不阻塞 view-evaluation，view/Fit 满载不影响capability-owned association signal source或DatasetBuilder；
- headless raster 不泄漏 Figure；
- worker raster + Qt overlay 的 ViewportTransform round-trip、revision mismatch 丢弃、ROI 事件真实改变 data-space Selection；
- 同一 coherence group 的多 panel 只在完整 CoherenceStamp（run/epoch、typed JoinKey、DatasetRevisionRef、document/selection revisions）一致时 board-atomic present；跨 generation 的相同裸 key/revision 不相等；独立 monitor不伪装 coherent；
- fit 不在 GUI thread；
- owner binder/pipeline validation/pulse compile 不在 GUI thread 且不持有 hardware claim；
- notebook Experiment facade 的 virtual connect -> capture -> 1D fit -> save 保持少量语句，headless 无 render extra 仍可完整运行；
- headless `fit.save()` 返回 neutral-owned `FitResultArtifactRef`、使用明确repository且不加载Matplotlib/Qt；figure_document只需frontend.figure/data_figure，只有figure()/GUI需要render/workbench extra；
- Experiment.readout 不保存 current calibration；依赖标定的 convenience request 必须显式接收 ref，并在构造时冻结 binding/ref/model，多 camera 不允许猜测或串用；
- panel 的视图摘要与实际 render 的 ViewSpec/EvaluatedFigureData 一致，权威操作摘要与执行的 CommittedTransform/FitProblem/ScanOutputContract 一致；
- 一次 Fit/Run 点击冻结 revision，后续 selector 变化不污染进行中的结果；
- EditorSession base revision 冲突拒绝 last-write-wins；
- shutdown 真实入口等待 RunHandle/worker/device acknowledgement，销毁后 queued result 被 lifetime token 拒绝。

Public hardware capability boundary：

- 从 Experiment、所有领域 facade、RunHandle、TaskConsole、PulseGUI、DeviceManager/Viewer、DeviceCatalogView/DeviceInfo 作为根递归遍历 public object graph；拒绝 BaseDevice/DeviceSet/SDK handle、BoundDevice/RunDevice/CleanupDevice、drive-capable Port、含设备binding或driver callback的internal RunPlan、raw bound method、resolver与drive verb；
- public Experiment.run/start signature只接受declarative Request；inspect只返回PlanDescriptor DTO。internal RunPlan、领域prepared value和immutable bindings只在composition/RunController私有执行图可达，RunHandle对象图不含plan或Port；
- public GUI constructor signature 不接受 Experiment/Session/DeviceSet、raw camera/sequencer、`devices_provider` 或返回 raw object 的 callback；TaskConsole running nodes 只暴露 DTO；
- `Zou_lab_control` 与 `neutral_atom` umbrella 的 raw symbol deny-list既不在`__all__`，`getattr`也必须AttributeError；frontend import graph不出现device adapter/registry/server module；
- AST drive-owner gate扫描`open/configure/arm/acquire/prepare/fire/abort/safe/close`，只允许 adapter owner、InstallationRuntime startup/shutdown、owner I/O lane bridge 与明确的 recovery implementation；offline maintenance只改canonical配置，不在旧进程触碰driver。教程、frontend、Definition、RunPlan/finalize不在allowlist；
- 从owner submodule直接导入real adapter但不持有composition owner capability时，constructor/open/任一drive verb在零硬件调用前拒绝；owner capability跨lane、跨installation或过期generation复用同样拒绝；
- 任意 DeviceCatalogView/DeviceInfo/DeviceRef可canonical serialize、不可mutation、不含callable/raw object；role顺序和digest稳定，`require()`只返回DeviceInfo；
- 并发读catalog与health变化时，每个读者只看到一个runtime instance的完整snapshot；snapshot/watch之间通知反序、漏失或shutdown时，单调revision+replay/current reread保证UI不回退且authority不等待UI；
- 同binding health变化只推进catalog revision；runtime_instance_id只在新进程生成，connection generation只在startup/recovery bind生成，二者均不可复用；CLOSING前排队但尚未admit的command永远不能迟到执行；
- 所有旧DeviceRef、command facade与pending GUI command在runtime CLOSING或新进程启动后以零adapter调用失败；旧DeviceCatalogView仍可作为历史值显示但不能执行；
- config/device/virtual-real改变不创建inert candidate、replacement graph或进程内transition；旧runtime先完成§12.7并退出，新进程随后从canonical config重建。GUI缺席、卡住或销毁不影响旧runtime的safe shutdown；
- public capture、PulseGUI、TaskConsole、DeviceControlPort与notebook路径在claim conflict、stale runtime/binding和InstallationRuntime CLOSING下全部fail closed；
- adapter contract tests从对应device contract/owner module导入，并由fixture在composition前保留raw spy；runtime/public/GUI tests不得为了断言底层调用从Experiment反向取得raw object；
- 仓库级 fixture 集合只能从 Git tracked set 或显式 committed manifest 枚举，禁止用目录 glob 把开发者本机的 ignored 实验文件变成隐形测试输入；可选私有文件不存在时 skip 的测试不计入回归覆盖，观测事故必须转成 committed 最小 golden 或由独立模型生成的确定性场景；
- docs、notebook templates与生成的notebooks grep禁止`exp.devices` raw、`exp.camera/sequencer`、直接QCMOSCamera/RemoteSequencer构造及直接open/acquire/prepare/fire；关键virtual notebook在CI执行。

Artifact/Calibration：

- commit 每个步骤 crash injection：manifest commit marker 前 load 永远看不到成功 artifact；
- cancel/manifest replace race 在 commit point 前取消、之后 TOO_LATE/SUCCEEDED 线性一致；
- RepositoryRoot atomic/durability probe 失败或 temp/final 跨 backend 时拒绝正式 commit；
- blob/manifest digest、dtype/shape/length 任一损坏均 fail closed；
- zlc_storage.canonical primitive golden/property vectors跨 data/pulse/frontend/neutral byte-identical；跨包嵌值对象必须委托 owner canonical tree/bytes；
- LiveDataBlockRef codec 拒绝，Figure Save 冻结屏幕对应 revision 而非随后 latest；
- Figure/Fit save同时验证输入epoch integrity；PROVISIONAL/INVALID revision不能通过普通typed artifact codec绕过authority gate；
- failed/cancelled scan 只产生 RunFailureRecord，不产生 ScanArtifact；
- Task mid-run frame/map只通过正式LiveDatasetPort/Host、SignalDataPlane与DatasetRevisionRef显示；不存在第二个task-local输出carrier或mutable buffer；
- live calibration 先生成 CaptureArtifact，之后与 offline ref 走 byte/contract-equivalent 算法路径；
- FrameContract/SiteMap/model mismatch 拒绝 occupancy；
- required calibration model 任一失败不提交 CalibrationArtifact；

Pulse/FPGA：

- 仓库内每个tracked pulse JSON均只使用当前`zlc_pulse.PulseDocument`并通过`load_pulse_document()`与当前codec round-trip/golden；未知schema输入由同一owner确定性`ValueError`，package/CLI/GUI中不存在历史parser、fixture、upgrade chain或schema转换器；
- duplicate/out-of-order/gap/EOS incomplete 均 safe；
- upstream exact edge gap 使正式 scan 失败；
- multi-axis TriggerKey -> ScanCellKey -> PointLayout round-trip，non-scalar y axes 完整且 transform explicit；
- MOT 无 API fallback；
- Pulse preview 不制造 frontend -> FPGA import；
- build/target digest mismatch fail closed；
- 现有`image.build_fingerprint`/几何/ABI handshake mismatch fail closed；测试不得要求当前bitstream不存在的design_build_id/timing ROM；
- partial/oversized upload、host digest/table mismatch、旧connection generation或旧PreparedProgramRef均不能进入正式FIRE；不声称硬件one-shot token；
- streamed scan只测试当前bitstream实际支持的bank-ready、chunk marker、cursor、status与error语义；短表和长表走同一个ping-pong协议，不把双bank窗口误写为点数上限；不存在的RTL CRC/BANK_VERIFIED/sticky位不写fake测试；
- reconnect generation使软件PreparedProgramRef失效，SAFE/RESET按现有协议验证；
- 主RPyC wait_done/backend `_io_lock`/transport阻塞时现有timeout/cancel/abort/safe行为有真机故障注入；无法确认safe则当前run失败，但测试不要求新增watchdog/独立SAFE寄存器；
- host/model/RTL golden byte-identical；
- host encoder/coalescer 不生成超过当前 `FRAME_WORDS` 能力的 UART frame，server/upload/PreparedProgramRef guard 对 partial/oversized payload 在发送前拒绝并禁止 FIRE；contract kit如实记录当前RTL收到合法CRC oversized frame时缺少硬件零提交保证的已知边界，测试不得为满足目标合同而假设或要求新RTL。只有golden/真机证据确认该行为是既定RTL设计偏离并经H2批准后，才增加“硬件收到oversized也零提交”的bitstream gate；
- qCMOS contract kit分别保存nFrameCount累计快照与per-frame framestamp/camerastamp/timestamp的位宽、signedness、modulus、reset epoch、rollover语义，以及批准工作点内的触发间隔、loss/reorder观察与可复现报告；任何unwrap多解、未声明reset、stamp duplicate/gap或counter倒退都使该Camera association失败；
- qCMOS association preflight冻结trigger source/polarity、exposure、ROI/binning、readout mode与compiler给出的source-neutral edge schedule；整run结束核对produced/drained count、metadata单调性、coverage和exact PulseTerminalAck，pending/late/extra frame阻止该run提交；
- `zlc_pulse`合同只测试当前DONE/status/cursor与PreparedProgramRef/PulseTerminalAck，不把它们命名成Camera terminal、EndAttestation或不存在的PHYSICAL_DONE；Camera owner负责把pulse terminal和自己的物理观察组合成association evidence；
- 当前`scan_repeats=K`可能多发下一sweep point的路径有回归测试并被formal compiler明确拒绝；正常SCAN_SLOT覆盖完整表冻结、一次FIRE和自主流式完成，API_SLOT只覆盖既有segmented例外；
- HardwareTriggerStamp/FIFO/trigger-return/新ROM测试默认不存在；只有证据批准RTL修复后才加入对应package与真机gate；
- Measurement与capability-owned derived signal source全链传播EventRef、direct causation、source generation和processor stage；只有producer association完成且collector coverage完整时formal sink才能commit。

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

Workbench E2E 必须从真实 launcher/composition root 驱动用户路径：DeviceManager Init -> 同时得到 TaskConsole/PulseGUI -> Add Panel -> Setting -> 选 Measurement/Processor -> Start -> selector/repeat -> Fit -> Save/Load -> Stop/Close；不能用 demo fixture、直接 poke controller 内部或手工调用 `_tick` 代替。PulseGUI 同样从真实入口执行 Edit -> Preview -> Scan/prepare -> cancel/safe。

所有GUI的视觉debug/验收采用同一双轨合同。**快轨就是offscreen正式窗口**：共享test helper必须在QApplication出现前只设置`QT_QPA_PLATFORM=offscreen`，再调用唯一`ensure_qt_app()`，随后只走该产品的正式open/composition与`launch_fluent_window()`，通过真实Qt input点击可见控件、等待事件稳定后抓outer FluentWindow。它不得另建`QApplication`、设置任何DPI/scale、改产品窗口尺寸/样式、直接`setCurrent*`或poke controller，也不得在桌面弹窗。Windows offscreen plugin若给出空font database，唯一application owner可注册产品本来就声明的系统font face；共享契约测试必须直接栅格化文字并验证非背景glyph像素，只有方框/空白的截图无效。**慢轨才是真正桌面人类流程**：从该产品正式根`.py/.bat` launcher启动，用桌面鼠标/键盘逐步操作并取屏幕截图，作为最终或任何视觉争议的裁决。两轨共享完全相同的application owner、composition root、sizing/style与产品操作序列，差别只有Qt平台和驱动层；快轨截图用于高频视觉debug，慢轨证明真实桌面呈现。这个合同覆盖PulseGUI、TaskConsole、DeviceManager、FigureViewer及未来所有GUI；每个应用只定义自己的动作序列，offscreen选择、事件等待、整窗抓取和geometry记录由共享owner实现，禁止再复制一套截图脚本。

这里的“交互事件”必须由真实 Qt input/event 路径覆盖，而不是只断言 controller state：

- 在运行中的 raw camera panel 上创建 ROI、热修改已有 ROI/threshold 并删除下游 processor；逐 revision 观察 `ACCEPTED -> APPLIED/terminal`，同时证明 source Run、raw stream generation、raw front sequence 与 source tap topology 不重启、不回退且无 gap；
- 对每种 live plot kind 执行适用的 Area、locked Cross、zoom/pan、clim/threshold；plot pointer-motion 数据 hover 明确不存在。shown plot 必须返回适用的非空 interaction handle，并用同 revision 的 `ViewportTransform` 验证 raster↔data 命中；
- 拖拽/调整一个 panel 期间，其它 panel 继续 present；被拖 panel 在 release 后补到最新合法 revision，不把拖拽期间的 stale front 当新 front；
- 通过同一 Setting/Edit 路径切换 `normal/tight/fixed` relim、cmap 与 limits，并验证保存/重开；
- 从 panel Setting/Edit 或 DataFigure `Fit` tab 的同一个 model+args pane 一键提交 authority draft，框选直接预填同一 draft，覆盖 1D range 与 2D box ROI fit，并证明后台求解时 live/其它 panel 仍响应、stale completion 不覆盖新 selection；
- 从通用 figure/archive viewer 载入任意已存 figure/artifact，执行 zoom/pan、re-fit 与 export；报告类 frozen multi-page raster 按原生像素呈现并滚动浏览，不能以静态 PNG 冒充通用 viewer 的交互能力。

每条 E2E 都必须同时验证用户可见结果、时序与不中断项；截图只补视觉证据，不能替代行为 oracle。

virtual 与 real adapter 运行相同 Task/Measurement capability bind、flat RunPlan、artifact repository 和文件夹流程，只替换最低层 Port；adapter contract kit 使用生产 Port 的真实属性/方法名，并覆盖`ORDERED_END_ATTESTED_RUN`、CameraExternalTriggerQualification、DCAM frame metadata、timeout、buffer reuse、disconnect与health recovery。virtual adapter可以注入drop/reorder验证整run invalidation与qualification revocation状态机，但不能用fake队列代替真机有限样本、统计上界与工作区间资格化证据。

## 19. 实现符合性协议

本节只规定任何实现或变更如何证明符合本设计，不记录哪一轮完成了什么。完成日期、临时测试数字、迁移轮次和过期处置历史不属于产品文档，也不得成为运行时或测试输入。

### 19.1 Owner 闭包

实现必须始终满足以下闭包：

- 包与 bounded-context 边界以 §20 为准；领域值、持久化、pulse、neutral application、frontend presentation、Workbench host 与 public composition各有唯一 owner。
- Camera finite/live、Calibration、Occupancy、MOT、PulseScan与release-recapture的Definition、request、application、artifact和可选adapter/UI都位于自己的capability/family边界。
- sample/event plane共同使用 `AcquisitionStream/SignalEvent/EventRef/ValueSchema`；dataset presentation plane由neutral `LiveDatasetPort/Host`和`SignalDataPlane/SignalFront/SignalValue`唯一拥有，processor hosting只用`HostedProcessor + DerivedSignalOutput`，Workbench没有同义数据carrier。
- exact关联链由Camera或其它producer保存terminal-bound evidence；只有具体capability-owned derived signal source证明严格1:1时才能透明传播并追加lineage，当前Occupancy使用`PreparedOccupancyProcessor + OccupancySignalValues`。PulseScan只验证evidence并提交ScanArtifact，Workbench不按领域类型分支，也不补造association。
- PlotPanel、DataFigure、FitGrid、FigureViewer中的plot surface与Calibration report都委托frontend `FigureSurfaceHost/Lane`及其headless Figure/Plot contracts；Workbench只提供routing、layout、exact-revision presentation sidecar与repository/file I/O，不复制composer、selector、Fit或viewport owner。
- public capability短方法由leaf-owned `LogicNodePackage.bind_api`提供，application composition形成冻结的`exp.nodes.*`；不建立动态registry或第二套facade实现。
- 真实设备只有在当前installation、Port合同和本次运行证据全部存在时启用；缺失事实必须在prepare/start前具名拒绝。

### 19.2 Dependency-closed 变更

每个实现变更都沿同一条完整产品路径闭合，不横向预建“未来框架”：

```text
正式 launcher / public API
→ declarative request 与 binding
→ capability application + Port / signal capability
→ typed dataset / artifact / frontend presentation
→ 用户路径 E2E、故障反例与资源收尾
→ 同一事实的 producer/consumer/export/test/doc 只有一个 owner
```

移动或删除一个symbol时，最后consumer、import、re-export、fixture、测试与文档引用必须在同一变更中闭合。仍有真实consumer时保留其唯一owner；不得用改名、alias、shim、fallback或双reader/writer冒充完成。

### 19.3 发布证据

每次发布必须同时具备：

1. **真实设备证据**：从installation配置进入实际Camera、sequencer与RF Port，复用与virtual相同的application、association、dataset和artifact路径；读回、时序、terminal、故障与cleanup证据不能由simulation替代。
2. **用户产品链**：正式TaskConsole/notebook可完成Camera → Calibration → Occupancy → PulseScan → DataFigure/save/reload，并用真实Qt输入或public facade覆盖选择、运行、停止、关闭和失败保留。
3. **声明覆盖**：每个 `LogicNodeDeclaration`都有显式installed wiring或typed unavailable reason；MOT、release-recapture、readout、fit、facet/grid/sites等合法能力都接入同一typed output、Figure与artifact owner。
4. **边界与残余检查**：headless/import/install边界、教程、tracked assets、current codecs和公开入口通过机械扫描；alias、fallback、双实现、无consumer symbol与绕过authority的路径为零。
5. **发布资格分离**：software/virtual产品通过不自动放行real qCMOS或硬件改变；每个产品只凭自己列出的evidence判定GO/NO-GO。

新的真实需求若不能放入现有owner，先用具体consumer、语义与依赖证明最小新contract；文档不为可能的需求预建目录、Service或通用协议。

## 20. 最终 bounded-context tree

下列只展开承担边界职责的目录，不把单文件module伪装成独立bounded context：

```text
zlc_data/
zlc_storage/
zlc_pulse/
  transport/

zlc_neutral_atom/
  installation_package.py
  logic_node_package.py
  artifact_dispatch.py
  runtime/
  processing/
  capture/
  devices/
    camera/
    hardware/
    sequencer/
    simulation/
  logic_nodes/
    camera_measurement/
    mot_field/
    pulse_scan/
      ui/
    readout/
      calibration/
        ui/
      duration_fidelity/
      occupancy/
        ui/
    release_recapture/
      grey_molasses_detuning/
      temperature/
  timing/

zlc_frontend/
  figure/
  plot_panel.py
  data_figure.py
  fit_grid.py
  plot_report.py
  qt_widgets/

zlc_workbench/
  task_console/
  data_figure/       # lifecycle/cancel/repository-file I/O host only
  device_manager/
  figure_viewer/
  fit_grid/          # lifecycle/cancel/repository-file I/O host only
  pulse_editor/

Zou_lab_control/
  api/
  workbench/
```

Workbench根只保留确实被多个产品消费的`form_projection`、`frozen_raster`与`window_runtime`。Run/Processor/Signal/live-dataset ownership属于neutral `HostedRun/HostedProcessor/SignalDataPlane/LiveDatasetHost`；DataFigure、PulseEditor、TaskConsole的产品状态分别位于自己的产品子包。不得在Workbench根目录再放一个单消费者owner或用root re-export兼容旧路径；删除最后consumer后目录随源码一起消失。

边界方向保持单向：`zlc_data`与 `zlc_storage`提供纯合同和持久化primitive；`zlc_pulse`独立拥有PulseDocument、target/compiler、transport与冻结RTL资产；`zlc_neutral_atom`拥有installation、runtime、设备Port、node-neutral capture、logic-node application与实验artifact，以及两个固定namespace的冻结package contract；`zlc_neutral_atom.timing.pulse_parameter_scan`拥有跨capability共享的冻结pulse-parameter program vocabulary；`zlc_frontend`拥有PlotPanel/DataFigure/FitGrid/report/SiteMap view与全部presentation；`zlc_workbench`只组成领域中立Qt host并托管lifecycle/I/O；`Zou_lab_control`只提供public facade与确定性composition。设备SDK、Qt widget、repository implementation和public facade不能反向成为领域事实owner。

新增代码首先进入上述owner。只有真实新bounded context同时拥有独立语义、生命周期、持久合同和至少一个真实consumer时，才允许增加同级目录。

## 21. 零残余不变量

- public import、launcher、教程、asset和持久reader/writer只指向唯一current owner；不存在第二棵兼容树、warning alias、fallback proxy、双codec或隐式upgrade chain。
- 同一事实只有一个validator、canonical codec、digest、lifecycle和presentation owner；GUI snapshot、事件顺序或render完成不能补造物理因果、artifact authority或设备状态。
- frontend不持有raw device、SDK handle或drive verb；Workbench只接窄application ports并托管lifecycle/I/O；neutral领域层不依赖Qt/Workbench；virtual与real只在最低层Port implementation分叉。
- `BoundCameraCapture`只属于node-neutral物理capture；Logic-node Definition不得复制其capture contract。
- Value的ComponentValidity与DataBlock的DatasetComponentValidity是不同carrier；任意consumer不得按rank或broadcast巧合互换。
- software stream/mailbox不定义内存、pending、backlog、queue-turn或预测速率预算；真实分配失败、物理ring覆盖和presentation latest-only coalesce各用自己的合同。
- RTL、bitstream、XDC与build输入保持冻结；§15.5的只读审查缺口只能限制声明强度，不能授权软件fallback、伪造hardware capability或主动重烧。
- 过程历史、逐轮计数、临时checkpoint与活动测试命令不进入最终产品文档，也不保留独立迁移台账或测试输入；current contract、代码、artifact与可执行测试才是权威。

## 22. 最终验收

发布只有在以下条件同时成立时才可称为符合本设计：

1. §4与§20的import/owner ratchet全部通过，fresh headless import不加载optional GUI/render、repository backend或concrete hardware adapter。
2. 每个公开Definition、LogicNodePackage/API、Workbench entry和artifact/codec都能指向唯一owner；除固定内建namespace的确定性package发现外，无开放扫描、动态注册、service locator、alias、fallback、平行实现或通用processor DAG。
3. `(R,P,*data_shape)`、PointLayout、具名axes、Value/DataBlock validity carrier和lineage在capture、capability-owned Processor、scan、Fit、archive与Figure全链不丢失；`PreparedOccupancyProcessor.evaluate`与`OccupancySignalValues`分别证明latest snapshot和严格1:1 derived signal路径，scalar固定 `(R,P,1)`。
4. exact cursor、association、terminal、coverage、commit与cleanup故障反例全部fail closed；monitor latest、GUI revision和whole-board present不升级为formal authority。
5. `BoundCameraCapture`、RunController、ResourceClaim、owner I/O lane和terminal publication的线程/生命周期合同通过故障注入；Stop/Close不提前release。
6. frontend唯一拥有PlotPanel/DataFigure/FitGrid/report及其view/style/composer/codec；Workbench lane只托管worker/cancel和I/O，Workbench Qt host只安装frontend返回的immutable front/document。六种plot、Pulse preview、selector、Fit、Setting/Edit与保存/载入按§2及§4.3.1通过真实Qt/像素验收。
7. capability-owned package binders组成冻结`exp.nodes.*`，node-neutral capture留在ReadoutFacade；短脚本/notebook流程不暴露Port/RunPlan/raw hardware，也不复制领域逻辑或presentation。
8. current-only artifact、Figure、PulseDocument与configuration codec通过canonical round-trip、corruption、atomicity与lost-ack测试；未知格式明确拒绝。
9. real qCMOS formal association只有在production composition、CameraExternalTriggerQualification、trigger margin、metadata/coverage对账与product E2E全部通过后放行；virtual通过不能替代。
10. RTL/bitstream/XDC工作树保持零修改；任何硬件改动仍需被证实的RTL bug或既定设计偏离、与根因直接相关的最小方案及hardware owner单独批准。

精确命令、临时计数、签收人与日期不进入本节；发布证据由对应current contract test、artifact与真机qualification记录持有。

## 23. 核心结论

系统不需要通用异步工作流编排器。它需要的是：

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
+ connection-lifetime InstallationRuntime + immutable hardware graph
+ capability-evidence gates（已证明能力与期望能力严格分开）
```

扩展性来自稳定数据与能力边界、显式静态组合和机械contract tests，而不是更多继承层、Protocol、Service或动态注册。

GO/NO-GO按产品与证据分开：software/data/frontend/runtime边界、完整 `hardware` installation composition与E0入口、virtual Camera与capability-owned严格1:1 derived signal association、source-neutral PulseScan及其virtual产品E2E可以独立判定为software-ready。真实qCMOS正式scan仍须在具体机器上完成production device graph、CameraExternalTriggerQualification、trigger schedule margin、produced/drained metadata/coverage对账和相同产品E2E；普通Camera monitor/capture与Pylon MOT资格由各自adapter contract kit裁决，不与PulseScan gate混名。硬件改变默认 **NO-GO**；只有真机证据在已批准工作点仍证明loss/reorder无法由相机设置、trigger rate或margin修正，或证明现有RTL bug/既定设计偏离，才可提出与根因直接相关的最小改动并另行取得hardware owner批准。

最终用户仍使用熟悉的TaskConsole、PulseGUI和notebook流程；重型board保持worker-raster性能，notebook保持短路径，MOT和普通SCAN_SLOT保持一次FIRE后的硬件自主时序。每个producer为自己的物理关联背书；只有capability-owned严格1:1 derived signal source可以保真传播，Workbench只托管生命周期/I/O并安装frontend返回的immutable front，PulseScan只验证通用合同并提交唯一ScanArtifact。
