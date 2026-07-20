# Zou_lab_control 最终系统架构设计

## 1. 文档定位

本文定义 Zou_lab_control 的最终目标架构、运行语义、数据与线程约束、包所有权、实现顺序和验收标准。实现工作应以本文为架构基线；手册负责用户教学，测试负责机械强制，代码中的公开类型负责表达运行时合同。

目标不是把当前类机械搬进新目录，而是建立以下不变量：

1. data kernel、pulse/FPGA、neutral_atom、frontend 和 composition app 的职责与依赖方向唯一。
2. Task、Measurement、StreamProcessor、Analysis 的语义清晰且不能互相冒充。
3. 多维数据、validity、axis、lineage 在采集、处理、拟合、显示和保存过程中不丢失。
4. 正式采集、实时监视、控制状态和运行事件使用不同通信语义。
5. 正式 PulseScan 的软件 reservation/cursor/processor/materializer 链端到端 exact；物理 frame↔trigger 关联由 `ORDERED_END_ATTESTED_RUN` 的 Q0 qualification envelope、冻结 schedule 与整 run 对账成立，并显式保存当前硬件下无法排除的剩余风险。
6. Qt、阻塞硬件 I/O、数值计算和 render 各有唯一线程 owner。
7. calibration 是内建领域能力，不使用 plugin 或动态发现。
8. 通用数据语义/fit 与呈现/交互分层：前者由 headless data kernel 维护一套，后者由 frontend 维护一套。
9. FPGA 的 build、target ABI、host、wire、RTL 和约束在构建/发布记录层可追溯；installation-owned deployment record把endpoint映射到冻结`.bit`/release/timing记录，近期 runtime 对现有 `image.build_fingerprint`、geometry/ABI 握手 fail closed，但前者只是SOP assertion，系统不声称能在运行时验证当前 bitstream 内容或 timing signoff。
10. 最终运行时不保留 alias、fallback、legacy reader、双 registry 或平行实现。
11. 精密 pulse/trigger 时序始终由现有 FPGA/qCMOS 等硬件执行；软件不得用 host sleep 调度微秒/纳秒事件。与此同时，bitstream/RTL 是冻结部署资产：架构不得为了获得更强证明而要求重烧；只有 E0a/Q0/真机证据发现现有 RTL bug、与既定设计不符或在已批准工作区间确实无法正确运行时，才单独评估硬件修复。
12. raw hardware graph 只存在于 installation composition/runtime owner 的封闭内部；普通 Experiment、领域对象、frontend、教程和公共 umbrella 只能取得 immutable observation、typed descriptor 与经 authority 的窄 command facade，不能反向找回 adapter 或旧 `DeviceSet`。

本文同时区分三种东西：终态不变量、当前冻结硬件上的 baseline capability、以及有明确删除点的迁移脚手架。正常 PulseScan 的执行方式族是现有 bitstream 的 `AUTONOMOUS_STREAMED`，其装载方式分为 fire前全部rows resident的`AUTONOMOUS_RESIDENT`与条件性的`AUTONOMOUS_REFILLED`；近期baseline只开放resident，refilled默认拒绝，只有§15.4强证明后才成为条件capability。Formal资格不是装载方式名称的一部分，而是 execution mode、有效Q0 qualification、association proof、软件exact链和本run EndAttestation共同评估的结果。唯一允许的非自主执行例外是 API-slot 值无法在一次自主sweep中无缝更新时已经存在并被接受的 `API_SLOT_SEGMENTED_EXISTING` 路径：整run只建立一次camera arm/exact capture transaction，R-major/P-fast地执行R×P个独立`STATIC_ONCE` pulse session，每个session取得自己的physical pulse terminal后才进入下一cell；segment之间必然存在的host gap由execution mode、canonical `segmentation_rationale`与有序pulse evidence显式表明，不能称为连续或autonomous。它不是通用`HOST_STEPPED_GROUP`，也绝不能成为SCAN_SLOT/MOT的逐cell fallback。逐沿 stamp、额外 ROM attestation、trigger-return 或新 watchdog 等需要重烧的增强只可在§1.1/H2的证据条件、因果证明与独立批准全部满足后作为硬件修复候选，不能成为软件架构的前置要求。

这里的“硬件时序优先”不是把领域 key、工作流或新观测电路塞进 FPGA。当前 baseline 使用现有硬件已经提供的 pulse execution、mode-specific raw terminal/build fingerprint回读，以及 qCMOS 外触发、frame counter/stamp/timestamp；neutral runtime 用冻结计划、Q0资格化的经验性ordered-trigger合同、preflight时序余量和整 run末端对账映射 TriggerKey/ScanCellKey。它明确弱于逐沿硬件证明，但在 PI 接受的风险边界内 fail closed：任何可见不一致使整个 run INVALID并按显式有限策略重跑，不能提交部分或猜测性结果。

### 1.1 硬约束与冲突裁决

以下条款是本文最高优先级的实现约束。若后文示例、类型草图、路线图或未来能力描述与本节冲突，以本节为准，冲突内容必须删除或降级，不能由实现者自行折中：

1. **现有 RTL/bitstream 冻结。** baseline 不生成、不修改、不重烧 bitstream；默认实现、迁移脚本和 CI 也不得隐式调用 Vivado synthesis/implementation/programming。只有 E0a/Q0 在已批准工作余量内实测到真实 loss/reorder 且软件、相机配置和时序余量无法解决，或 golden/model/真机证据证明现有 RTL 有 bug、偏离既定设计时，才允许进入单独的 H2 硬件修复评审；架构想获得更漂亮或更强的证明，不构成改硬件理由。H2 只是新提案的准入门，不自动授权修改、构建或烧录，仍需 PI/硬件 owner 单独批准。
2. **正常 PulseScan 只使用现有 FPGA 的无缝自主执行。** `AUTONOMOUS_STREAMED` 是正常执行方式族，近期装载方式基线是 fire 前全部物理 rows resident 的 `AUTONOMOUS_RESIDENT`。对 SCAN_SLOT/MOT，`HOST_STEPPED_GROUP`、逐 cell fire-and-wait、single-cell gate 和 host sleep edge scheduling 不得作为 baseline、首光方案、容量 fallback 或更强关联证明的替代品。
3. **唯一已接受的非无缝例外是 API-slot 既有 segmented 路径。** 它只适用于 adapter 已证明一次自主 sweep 中无法更新的 API_SLOT 值。整run一次arm相机并冻结R×P总frame预算；host只在显式segment boundary依次prepare/fire一个由硬件执行全部edge的`STATIC_ONCE` pulse session，并等该session的physical terminal与对应camera event后再进入下一cell。这个事实必须如实标记为 `API_SLOT_SEGMENTED_EXISTING`，不得包装成 autonomous execution，也不得伪造每段camera arm、camera terminal或camera EndAttestation。SCAN_SLOT/MOT 不得借此退化为 host stepping。
4. **能由现有硬件确定的精密时序必须由硬件确定。** FPGA 决定 pulse/trigger edge schedule，qCMOS 决定 exposure/readout/frame production；host 只冻结计划、验证工作 envelope、供应已冻结的获准 refill chunk、排空数据并做末端验证，不参与微秒/纳秒时序调度。
5. **当前物理关联保证诚实降级而不伪装。** 没有现存逐沿 emitted/accepted-trigger 回读时，baseline 使用有效 Q0 empirical ordered-trigger qualification、冻结 schedule、模式专用pulse terminal evidence和整 run EndAttestation；任何可见不一致整 run INVALID。API的per-cell PulseTerminalAck只证明有限pulse session物理终止，不是camera accepted-trigger/frame receipt；整个关联合同仍接受有限样本资格化无法绝对排除等量 loss+extra 的剩余风险。
6. **需要新 RTL 的观测增强默认不存在。** HardwareTriggerStamp/FIFO、trigger-return、per-fire counter、`PHYSICAL_DONE`、RTL CRC/`BANK_VERIFIED`、新 watchdog、`design_build_id` 与 timing-signoff ROM 都是证据触发后的可选修复候选，不是当前合同、测试假能力或迁移 gate。
7. **INVALID attempt 不可修补或续跑。** count/stamp/timestamp/coverage/EOS 任一对账失败后，不得丢掉“多余帧”、移动 ordinal、从某个 point 继续、复用该 attempt 的 provisional 数据或把已有行补成成功结果。显式重跑必须重新 arm、建立新的 session counter baseline、run_id/attempt_id、reservation、qualification authorization 与 lineage；失败 attempt 永久作为诊断事实保留，但不能成为成功 artifact 的数据来源。

当前事实的权威来源固定如下，避免“硬件时序优先”被误解成由软件重新制造硬实时保证：

| 事实 | 当前权威来源 | host 的合法职责 |
|---|---|---|
| pulse/trigger edge 的相对时序 | 冻结 bitstream 上的 FPGA scan engine 与 compiled table | 编译、冻结、preflight；不得逐 edge 调度 |
| qCMOS exposure/readout 与 frame production | 冻结 camera settings 下的 qCMOS/driver | autonomous与API segmented均为整run一次arm；API的R×P host boundary期间保持同一个capture session持续drain，并只在整run末端形成一次camera terminal及原始counter/stamp/timestamp对账 |
| 逻辑 TriggerKey/ScanCellKey 顺序 | 冻结 CompiledPulse schedule/PointLayout | 在有效 Q0 qualification envelope 内按序建立 provisional mapping |
| 完整 run 是否可成为权威 artifact | execution-mode-specific pulse evidence + 一个run级CameraRunEvidence + source-specific evidence + exact pipeline coverage 的aggregate EndAttestation | 由唯一I/O owner读取现有原始事实并比对；autonomous使用单个完整table terminal，API使用R-major/P-fast且session-id唯一的R×P个physical pulse terminal；相机两者都只在整run末端terminalize一次。不得把raw DONE当tail-idle、把per-cell pulse receipt伪装成camera attestation、使用UI progress镜像、补点或猜点 |
| 部署身份 | 现有 `image.build_fingerprint`/geometry/ABI 握手 + neutral installation-owned ProgrammedImageDeploymentRecordRef | fingerprint mismatch时所有真实upload/fire均拒绝；record inactive时只拒绝Formal upload/fire，H1前E0a诊断例外见§19；record只断言endpoint到冻结`.bit`/release的SOP映射，不得声称运行时验证旧bitstream未暴露的content/timing digest |

为避免实现者从分散章节拼出不同结论，baseline 的四个判定只有以下一套：

```text
execution_allowed :=
  AUTONOMOUS_RESIDENT
  or (AUTONOMOUS_REFILLED and §15.4 capability 已真实发布)
  or (API_SLOT_SEGMENTED_EXISTING and 该既有路径已被接受
      and adapter 已证明 API value 无法无缝更新
      and canonical非空segmentation_rationale明确说明为何物理实验允许host segment boundary
      and 完整R×P顺序、P个STATIC_ONCE artifacts、整runcamera settings/frame budget已冻结
      and 当前use case物理上接受任意、可变、非负host gap，
          不依赖连续演化、最大gap、精确settle时间或gap-dependent physics)

formal_fire_allowed :=
  execution_allowed
  and current image.build_fingerprint/geometry/ABI handshake 匹配
  and installation-owned ProgrammedImageDeploymentRecordRef仍active并绑定本endpoint、
      现有fingerprint、冻结.bit内容digest与已批准release/timing记录
  and frozen compiled schedule/settings/expected frame budget 完整
  and 每个 formal physical source 的 qualification/capability 精确覆盖本设备、软件版本与冻结设置
      （若 source 是 qCMOS：active Q0 还必须覆盖 camera settings 与 trigger interval margin）
  and source-specific inflight buffer、host exact retention、consumer 与 artifact sink 预算通过
      （若 source 是 qCMOS：driver ring 必须按 max-inflight 而非 total frames 定容）
      （若为API_SLOT_SEGMENTED_EXISTING：先以R×P cardinality floor在point resolve/compile前快速拒绝
          明显不可提交规模；随后在真实program、source block identity/schema、output contract、compiled refs/summaries、
          camera静态事实与逐point trigger/join shape全部冻结后、Run/arm/FIRE前，由具体ScanRepository
          mint一个绑定repository实例、构造期write-once metadata policy、artifact schema及上述fingerprints的process-local admission receipt。
          pulse/camera terminal canonical byte/node上界由各自owner强制；FINAL只能消费该receipt，实际encoder
          超过receipt或任何绑定事实漂移均是pre-FIRE证明被破坏的RuntimeError，不得伪装成跑完后的普通资源拒绝）
  and exact reservation/cursor/owner claims 已建立
  and（若为API_SLOT_SEGMENTED_EXISTING：整run camera authorization已在首次FIRE前固定，
      首cell已满足arm-ready/first-edge margin；后续cell的上一camera event与physical pulse terminal已完成，
      且camera required_external_trigger_interval的保守最小等待已deadline/cancel-aware完成）

formal_commit_allowed :=
  FIRE authority仍可追溯（autonomous为单次FIRE；API为一个run级camera authorization
      加R-major/P-fast的R×P个唯一PulseSession terminal lineage）
  and execution-mode-specific terminal evidence 与冻结 schedule 一致
      （autonomous table使用完整table terminal；API每cell使用独立STATIC_ONCE physical pulse terminal，CURSOR=N/A）
  and compiled/H1 post-terminal output-tail bound 已保守等待并记录
  and 每个 formal source 的run级produced/drained/terminal evidence与冻结计划一致
      （若 source 是 qCMOS：camera_produced_delta、frame/camera stamp、timestamp 容差全部通过）
  and processor/DatasetBuilder/coverage/EOS、source termination/join 与 safety disposition 全部通过

hardware_change_review_allowed :=
  批准余量内真实 loss/reorder 无法由 camera/software/margin 修正
  or 现有 RTL bug/对既定设计的偏离已有 golden/model/真机证据
```

任一 `formal_fire_allowed` 条件不成立都必须在 Formal arm/FIRE 前拒绝。任一 `formal_commit_allowed` 条件不成立都使整个 attempt `INVALID`，不得提交成功 artifact、不得按已有数量补点、不得把显示中的 provisional 数据升级为权威结果。H1前经批准的E0a `DIAGNOSTIC_CHARACTERIZATION`不使用这份Formal authority公式，只受§19明确的诊断合同约束且永不产生Formal/成功实验artifact。`hardware_change_review_allowed` 为假时，任何新寄存器、stamp FIFO、counter、ROM attestation、watchdog 或 bitstream 重建都不在本架构范围内。

## 2. 用户体验兼容目标

架构重构主要发生在内部。对实验用户，日常工作流应保持熟悉：

- 仍通过 `task_console.bat`、`pulse_gui.bat` 和 notebook 启动系统；
- PulseGUI 仍保留 Edit、Preview、Scan 三个主要工作区；
- TaskConsole 仍使用 Add Panel、Setting、Edit、Start、Stop、Save/Load；
- Measurement、Processor、Task 仍可从 catalog 选择并连接；其中 UI 中的 `Processor` 明确对应在线 `StreamProcessor`，完整数据集上的 Fit/Calibration/Report 显示为 `Analysis`；
- live image、rolling plot、histogram、site map、fit overlay 的视觉语言保持一致；
- virtual 与 real 仍只替换最低层设备 adapter；
- pulse prepare/fire/safe、camera arm/read/terminal-drain/disarm 的用户操作流程保持；内部会把旧的“stop后立即release”纠正为可验证的terminal recipe。

以下是有意的用户可见变化：

- Fit 成为明确可见的 `Add Analysis -> Fit` 和 Plot card `Analyze -> Fit` 操作；
- 非标量数据会自动得到一个可见、可改的默认视图；当选择或降维要进入 fit、scan y 或派生 artifact 等权威结果时，必须冻结为 CommittedTransform，不再暗中取第 0 项；
- history gap、schema change、缺帧和硬件 mismatch 会明确失败，不再显示拼接结果；
- qCMOS只有在当前adapter版本的Q0 qualification对目标工作点有效、preflight trigger余量通过且整run EndAttestation一致时才可产出Formal ScanArtifact；否则只能monitor或整run INVALID；
- qCMOS autonomous与API segmented正式扫描都一次arm整个run session、用按max-inflight定容的driver ring持续排空全部帧；正常SCAN_SLOT把完整逻辑scan table在fire前冻结，resident能力在fire前上传全部物理rows，条件refilled能力只供应冻结chunk，二者都由FPGA自主执行微观时序；仅API-slot值无法无缝更新时沿用既有segmented路径，按R-major/P-fast执行独立`STATIC_ONCE` pulse session并在segment间保留显式host gap，但相机不重复arm/stop且只做整run aggregate attestation；SCAN_SLOT/MOT不允许退化为逐cell host stepping；
- Stop 后若设备尚未确认退出，UI 显示 `CANCELLING`，不会提前宣称已停止；
- 设备冲突会提示并等待原 owner 真正退出，而不是静默抢占；
- 保存文件只接受当前 artifact schema；历史数据通过独立离线转换工具处理；
- monitor 可能跳过中间帧以保持实时，但一次画面中的相关信号不会来自不同 shot；
- 重型 grid/多 panel board 由 worker raster 后整板 coherent present，视觉与交互保持，但内部不再让 GUI/worker 无确认共享 Figure；
- Pulse prepare/upload、长 fit 和 calibration 不再阻塞 GUI thread。

因此，用户看到的主要入口、操作结构和视觉风格基本一致；变化集中在功能可发现性、错误提示和安全状态。内部实现则会彻底重构。

### 2.1 UX 行为权威、冻结条款与 salvage gate

**UX 行为权威固定为 `main` 分支旧实现。** 这与规则 4“物理/数值权威为 `main`”同构：selector 覆盖哪些 plot kind、zoom/pan 手感、live 是否不断流、ROI/threshold 是否热更新、fit 从哪里触发及何时可见、relim/cmap/limits 控件、Setting/Edit 布局与保存/载入流程，默认逐项继承 `main` 的真实用户行为。新架构可以彻底替换内部对象、线程、数据合同与包边界，但不能因实现更容易而缩窄用户面；安全或数据正确性要求确实需要偏离时，只能进入 §2.2 的 UX 偏离账本，不能散落在 checkpoint 中被写成永久产品合同。

以下条款为**冻结条款清单**，任何后续 checkpoint、局部 READY、实现困难或测试便利均不能覆盖：

1. §2 的“日常工作流应保持熟悉”和“视觉语言保持一致”；
2. §7.2 的 `ControlTopic[T]`：ROI、threshold 等运行控制是 typed、revisioned、acknowledged，已 ACCEPTED revision 必须在事务边界得到 `APPLIED` 或明确的 terminal ack；
3. §12.5 的 `WORKER_RASTER_LIVE`：worker raster 与 GUI 解耦的同时，Qt overlay 必须承担 ROI/crosshair/selection/hover，`ViewportTransform` 必须承担同 revision 的 zoom/pan 与命中换算；
4. §18.4 的真实 launcher/composition-root E2E：必须用真实交互事件证明日常流程和操作时序未退化，不能只证明 controller 内部状态或静态 PNG；
5. §12.6 的用户可见 Fit：普通 Fit 即提交权威 draft，selector/SelectionCandidate 可预填同一 draft，不能用额外确认步骤或独立工具窗口替代主工作流。

**行为 salvage gate 是每个 GUI/交互切片的开工前置证据。** 实现者必须先只读检查 `main` 对应旧实现，并在该切片 checkpoint 的首段冻结下列清单；未完成这一步不得改目标代码：

| 字段 | 必须记录的证据 |
|---|---|
| exact oracle | `main` 的 exact commit（本轮修宪基线为 `6c337d49c7086fa0ff21f879cd159bdf0e753f51`）、对应文件/符号与真实 consumer |
| 入口与控件 | 真实 launcher、菜单/按钮/快捷键、Setting/Edit 字段和默认值 |
| 交互覆盖 | selector/zoom/pan/hover/crosshair/relim/cmap/fit 各覆盖哪些 plot kind |
| 时序 | 按下、拖动、松手、Apply、Stop/Close 后何时可见、何时 authoritative |
| 不可中断项 | 哪个 source、raw front、其它 panel 或硬件 Run 在交互期间继续运行 |
| 即时生效项 | 哪些修改热更新，revision/ack 在何处显示 APPLIED |
| 保存/恢复 | workspace、figure、selection、fit 与控件状态的持久/重开行为 |
| authority | `DISPLAY_STATE / SELECTION_CANDIDATE / REVISIONED_CONTROL / AUTHORITATIVE_DRAFT / COMMITTED_RESULT` 中的边界与跨越动作 |
| 禁止复制机制 | 对应旧 Hub/共享 Figure/shape 猜测/隐式降维等只属于实现、不得迁入的机制 |
| 收口对照 | 新实现逐项 PASS/FAIL；FAIL 必须登记 §2.2，不能改写旧行为或删掉验收项 |

每一行只可标记 `MATCHED / MUST_CLOSE / LEDGERED_PENDING_APPROVAL / APPROVED_DEVIATION / NOT_APPLICABLE_WITH_EVIDENCE`，并附真实 launcher E2E、event、Run/frame/revision 证据。salvage 只把 `main` 当 UX/物理/算法 oracle，不把旧 `DeviceSet/Registry/Hub/LogicNode`、线程共享方式或包结构迁入目标架构。做不动时只能把能力拆成更小、仍保持行为闭合的纵向切片，或按冻结合同继续实现；“实现复杂”永远不构成降低 UX 的理由。

### 2.2 UX 偏离账本（全部待用户批准）

账本不阻塞后续纠正和迁移，但任何未关闭项都必须在《迁移完成报告》首页列为**待用户裁决**；未得到用户批准的偏离不能被称为终态设计。默认动作是恢复 `main` 行为并关闭账本项，而不是等待批准后才修复。

| id | `main` 旧行为权威 | 当前分支/既有 checkpoint 的偏离 | 形成原因（非正当化） | 状态与必须关闭的证据 |
|---|---|---|---|---|
| `UX-001` | 修改已有 ROI 时 raw camera source 与可见 raw 画面不断流；running downstream 在 drag 完成后直接 retarget | M2d/M2e 以 immutable whole-Run replacement 重启 source/stream/history，且 draft 还需额外 `Apply ROI` | 为避免首个 consumer 前置 ControlTopic 而选了较小实现 | **待用户批准 / 默认必须修复**：drag release 自动提交 revisioned ROI ControlTopic，事务边界 APPLIED 后才显示 applied；新建/删除只替换下游 generation；source Run、raw front、tap topology 不变的 E2E |
| `UX-002` | live plot 的 selector、zoom/pan、十字、hover 读值、relim 与 cmap/limits 覆盖旧 plot contract；拖动只冻结当前 panel，其它 panel 不停流，松手补拍。旧 1D/Monitor 的 Area 实际画二维矩形却只消费 x，纯水平拖会被误清 | U0.2b 已在 free-running camera monitor IMAGE 恢复完整 A/C/Z/H/hover、ViewportTransform、三种 relim、六种 cmap、同源 Setting/Edit 与 per-panel hold；U0.2c 又让 rolling Monitor 与真实 occupancy progressive `SCAN_POINT` 共用原生 x-span、continuous cross、真实 sample hover、x-only zoom/pan/home、三态 y relim、同源 Setting/Edit 与 panel-local hold；U0.2d 已在exact occupancy cell恢复Sites的双轴A/C/Z/H、nearest-site hover、clim、Setting/Edit与display-only rectangle；U0.2e 已在同一camera ROI scalar history的live Histogram恢复A/C/Z/H/pan/hover、linear/log count axis、shared bins、x pin、三态count relim、同源Setting/Edit与panel-local hold；U0.2f 又让finite exact capture IMAGE复用同一IMAGE owner取得完整A/C/Z/H/hover、clim/cmap、Setting/Edit和display-only rectangle。live Sites、generic Distribution/Grid及真实 launcher 的全 plot-kind handle 仍未闭合 | 早期窄 leaf 被误当作可长期冻结产品面；旧 1D selector 的视觉维度与实际 authority 维度不一致 | **待用户批准 / 默认必须继续修复**：剩余live plot kind的非空交互句柄、各自relim/axis语义、grid focus与launcher E2E；已经共用唯一强类型owner的finite/live IMAGE、CURVE/SCAN_POINT、current ROI scalar Histogram与exact Sites行为不得回退，display-only rectangle/range不得静默升级为Analysis authority |
| `UX-003` | figure viewer 载入后是可交互 panel，可 zoom/pan、重新 fit 与导出；GridPlot 可 focus/返回 | U0.3a-h 已把单panel HISTOGRAM/CURVE/IMAGE/METER、fit-bearing CURVE/IMAGE，以及单layer METER/HISTOGRAM/CURVE多cell overview/focus接入同一 typed board；range/cross/zoom/pan/home、relim/cmap/limits、Setting/Edit、原子导出、即时 fit overlay 已恢复。exact saved-fit GridPlot、METER grid、SITE-faceted occupancy counts Histogram与真实ScanArtifact CURVE都不暗选第一格，只在显式命中真实cell后focus，并从缓存返回overview；稀疏hole保持原逻辑位置。CURVE grid focus因其显示曲线已含display-only repeat reduction，不能伪装成轴完整Fit authority，故Analyze保持明确禁用；尚未闭合的是multi-layer DataFigure、faceted IMAGE及非Fit的其它archive plot kind，报告类多页仍是 frozen raster | 继续按真实consumer逐个恢复typed交互面；typed资格不足时保留完整whole-figure renderer，不以丢fit/轴/series换交互，也不先造通用dashboard框架 | **待用户批准 / 默认必须继续修复**：剩余multi-layer、faceted IMAGE与archive plot kind，以及CURVE grid的轴完整Analyze入口；U0.3a-h及W7已恢复的typed交互、explicit-cell focus/Refit与导出不得回退。**报告类多页zoom已闭合**：`QtImageBoard(zoomable=True)` 提供滚轮放大(上限16x)、拖拽平移、放大后双击复位，可见窗口恒被夹在页内；live 面板默认关闭该开关，因为其 zoom 属于同 revision 的 `ViewportTransform`(§12.5 冻结条款)，证据 `tests/test_u04_frozen_report_zoom.py` |
| `UX-004` | plot/panel 内直接发起 fit，框选与 fit selection 联动，普通 Fit 一步生效；2D box ROI fit 可见 | U0.3d 已闭合通用 Figure 的 `Analyze -> Fit`、1D range、2D box ROI、一键authority draft、Capture/Scan exact source、即时CURVE/IMAGE overlay、Save/reopen/Refit和`fit_gui`委托同一host；U0.3e 又让真实 current TaskConsole 在当前 card 产生精确 FINAL `ScanArtifactRef` 后开放 `Add Analysis -> Fit`，委托同一 DataFigure/Fit host，并把根 launcher 从旧 SignalHub/registry UI 切到同一 Workbench | 先交付typed solver/editor与exact artifact，再将独立W5 host物理删除并并入DataFigure；TaskConsole只桥接已提交artifact，不为无人消费的自动分析预建第二生命周期 | **CLOSED**：Figure与TaskConsole两入口、真实launcher、selector预填、双空间轴fit、显式Save/reopen和唯一host均有current E2E；不得回退或把display selection升级为authority |
| `UX-005` | TaskConsole 支持全部 addable plot kind、Monitor board、Logic rows、Start/Stop、统一 Setting/Edit、树形 signal picker、布局与 workspace | 当前 target TaskConsole 仍是 one-card/one-Task 窄产品，不能完成旧 tutorial 日常流程 | 纵向迁移尚未接完，而窄 checkpoint 容易被误读为产品完成 | **待用户批准 / 默认必须修复**：U0#4 全清单与真实 launcher tutorial E2E 零代码走通 |
| `UX-006` | calibration/site-map/occupancy 图沿统一交互 viewer/GridPlot 获得 focus、selector、zoom 与 export | U0.3f-g已让单layer SITE-faceted occupancy METER与真实`OccupancyArtifactRef(counts)` HISTOGRAM复用同一Figure外壳；U0.3h又把真实AUTONOMOUS occupancy `ScanArtifactRef`的35-cell CURVE接入同一same-draw overview、exact-cell focus、standalone Curve交互、缓存返回与原子export。三者都不暗选第一site，不丢ComponentValidity、sample/value、dropped count、单位、exact ref或SITE地址；CURVE overview共享X/Y以可比，focus恢复局部relim。calibration report、site-map图与其它occupancy图仍未闭合，未提交工作树不能作为证据 | 保留已完成的METER/HISTOGRAM/CURVE grid；其余路径先闭合immutable report/exact-address与预算/线程所有权，再按真实surface接入既有selector/viewer | **待用户批准 / 默认必须修复**：报告类多页zoom**已闭合**(见 `UX-003`)；calibration/site-map与剩余occupancy路径按旧surface补适用的focus/selector/export，或逐项提交用户裁决 |
| `UX-007` | 旧实现可能按 first/flatten/trailing mean 直接把非标量送入权威结果 | 新设计只让显示层 role-driven auto；fit/scan-y/derived artifact 必须显式冻结 `CommittedTransform` | 防止静默丢轴和错误物理结论 | **待用户批准；安全边界不可回退**：默认视图低摩擦、当前投影始终可见可改、authority draft 由视图预填但明确提交 |
| `UX-008` | 旧 UI 可能继续显示拼接/latest，或在缺帧、gap/schema/hardware mismatch 后隐式继续 | 新设计明确报错并使 exact attempt 失败 | 防止数据错位和伪成功 | **待用户批准其提示/恢复 UX；正确性不可回退**：错误可理解、原始失败证据可查、monitor 与 formal 状态不混淆 |
| `UX-009` | 旧 qCMOS 路径可在缺少当前资格/末端证明时继续；部分软件路径可能逐 cell host stepping | Formal 必须 Q0+preflight+EndAttestation；SCAN_SLOT/MOT 保持冻结 bitstream 的 autonomous hardware timing，不允许 host-stepped fallback | 物理时序与帧关联正确性 | **待用户批准其 gate/重跑 UX；硬件/正确性边界不可回退**：INVALID 原因、手动 reject-and-redo 与资格状态清楚可见 |
| `UX-010` | Stop/设备切换可能立即 release 或静默抢占 | UI 在真实 terminal/SAFE/reap 前显示 `CANCELLING`，冲突等待原 owner 退出 | 防止双 owner 与未安全停止即重用 | **待用户批准其等待/错误 UX；安全边界不可回退**：真实 launcher cancel/close/conflict E2E |
| `UX-011` | 历史软件 artifact 可能由同一 reader 猜格式或宽松升级 | 普通软件只接受 current plain schema；历史转换为独立离线工具 | 删除升级链和双 reader 历史残余 | **待用户批准其错误/转换 UX；current-only边界不可回退**：拒绝信息与离线转换 runbook |
| `UX-012` | monitor 为实时性可跳中间显示帧，重型操作曾阻塞 GUI 或由 GUI/worker 共享 Figure | 新 monitor 仍允许显示丢帧但同一 board coherent；compose/fit/calibration 在有界 worker，GUI 保持响应 | 保持交互性能同时消除 shot 混合和线程竞态 | **待用户批准 / 默认匹配旧手感**：profile + coherent board + GUI event latency E2E，不能把 monitor skip 泄漏到 formal exact |

| `UX-013` | `main` 的 console `_message()` 在真实 GUI 下把「Saved / Load failed / …」这类一次性告知走 `fluent_message` 弹窗，只有 offscreen 时才写进常驻状态条；状态条本身只承载 error/task/display-behind 三级 | 迁移后的 console 把这类告知作为常驻状态条的**最低一级**（`info`）显示，不弹窗 | 迁移后的 console 每一步（Add/Remove/Save/Load/Analysis）都会产生这类告知，逐条弹窗会把日常操作变成点确认；状态条本身是常驻固定高度，显示它不挤动布局 | **待用户批准**：若用户要求恢复弹窗，改回 `fluent_message` 即可，状态条的 error>task>warning 三级排序与常驻性不受影响 |

| `UX-014` | `main` 的 TaskConsole 窗口:**49** 个 widget;header 九件控件 `Add Panel` / `Selectors` / `Devices` / `Pause` / `Save image` / `Save` / `Load` / `Stop task` / `⋯`;常驻两个不可关 tab `Monitor` 与 `Logic`;Add Panel 下拉列出六种可加 plot kind(`Plot: 2D image`、`Plot: Site map`、`Plot: 1D vector`、`Plot: Rolling trace`、`Plot: Distribution`、`Plot: Site grid`) | 迁移后的 `TaskConsoleWindow`:**20** 个 widget;header 只剩 `Add Panel` / `Save` / `Load`,并新增了旧版没有的 `Add Analysis → Fit`;**完全没有 tab**;Add Panel 下拉只有一项 `Task: Pulse scan`,因此根本无法添加任何 plot 面板 | 新窗口是**从零另画**的,不是把旧窗口搬过来;此前没有任何测试以旧界面为基准,所以整条白名单全绿的同时用户面持续漂移。这不是设计决定,是缺闸导致的累积降级 | **待用户批准 / 默认必须全部恢复**:这是清单4 的实质内容。`tests/test_u04_console_ui_parity.py` 同进程构造新旧两个 console 并逐项 diff,充当双向棘轮——再掉一个控件立即红,恢复一个控件也必须同 commit 删掉本行对应条目。本行清空之时该测试退化为新旧窗口的相等断言 |
| `UX-015` | `main` 的 console 在三处 GUI 路径上调用 `RenderLoop.barrier()` 并**忽略返回值**(Edit 快照 / Save image / 无条件 Refresh):barrier 超时后仍继续读写 Figure | 迁移后这三处改为**fail-closed**——握手未落定即放弃该操作,并在常驻状态条以 `error` 级报出被拒的动作名 | L596 规定未迁出的共享-Figure RenderLoop **只允许**活在 `SerializedLegacyAggBridge` 隔离岛内,而该岛是 fail-closed 的;忽略 barrier 结果意味着 GUI 线程可能在 render worker 仍持有 Figure 时闯入,正是该岛存在的理由。这不是为省事缩窄用户面,而是 L129 所指的**安全/数据正确性驱动**偏离 | **LEDGERED_PENDING_APPROVAL**:正常操作下行为不变,仅在握手超时(故障路径)时可见差异——操作被放弃而不是可能存下一张撕裂的图。按 L3874 该状态允许正交整改继续推进;证据 `tests/test_u05_render_island.py` |

只有用户明确批准的偏离才可把状态改为 `APPROVED`，并必须记录批准范围、日期与替代验收；实现者、测试通过或局部架构审查都无权自行批准。

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

当前 finite CameraMeasurement 每来一帧重新发布累计 `(1..K)` repeat block，TensorStore full publish 又把整个 repeat-capacity current state复制进 journal；已删除的旧 `OccupancyProcessor` 还曾从头遍历所有已累计 R/P，因此实测 journal payload 随 K² 增长。删除旧 processor 并没有消除 camera/materializer 的根因。把 camera pending queue 改成 O(1)、有界且 overrun-fatal 也只能解决 adapter 边界的保留与覆写问题，不会自动消除上层的累计重发布。根因是 sample event、mutable materializer 与 immutable dataset 被同一个“signal tensor update”冒充；目标实现必须一次只处理一个 sample、builder 私有增量写、UI 按 revision 请求 snapshot，而不是只把 history size 调小。

多 signal `next_coherent_update` 当前会寻找下一个共同 provenance 并推进掉更快流的 unmatched 更新；这对 coherent monitor 可以接受，却不能自动代表 formal EXACT_KEY。JoinPolicy 必须在 pipeline 合同中显式区分“允许跳过并计数”与“缺 key 立即失败”。

真 qCMOS 的 capture `nFrameCount` 是产出帧累计数；仓库 DCAM wrapper 还暴露 `nNewestFrameIndex`、framestamp/camerastamp/timestamp。当前 adapter 已完成一块正确的迁移首付：exact 路径中 `cap_transferinfo()` 失败、负数或计数倒退均立即失败；每次先读累计 count，有已存在 backlog 就直接排空，只有 count 与 drained cursor 相等才等待新的 ready event；ring 槽位由同一次 snapshot 的 `nNewestFrameIndex` 反推，不能再假设 `source_ordinal % ring_size`；每帧复制后重新读取 count，若复制期间 ring 已可能覆盖当前 ordinal 则整次采集失败；wait 与 backlog copy 共用同一总 deadline，并在帧间检查 Stop。已报告的一批 frame 全部进入同一个有界 pending record queue，`CameraFrameRecord` 保留 produced count、frame/camera stamp、timestamp 与实际 driver ring index。driver ring只按capability中声明的max-inflight burst定容，完整run保留属于host exact stream/dataset，不再按`total_frames`重复分配同样大的相机ring。

这些整改只关闭host retention、槽位映射和已观察overrun的静默错位，并不自动证明max-inflight数值或外触发工作区间内“一触发一帧、按序、无漏帧”。当前真实 qCMOS 的 qualification 仍为 `None`，所以target exact prepare在arm前明确拒绝；只有virtual deterministic source被放行。旧 `read_frames()/acquire()` array-only consumer仍会在record边界后丢掉metadata；若旧上层继续取latest或累计重发布，软件仍可能形成第二套真相源。近期完成路径不改变冻结bitstream：E0a先用现有系统探索目标 exposure/ROI/readout/触发间距，S1/H1稳定后Q0再用最终adapter、CaptureSession与buffer/drain policy建立可发布的经验性ordered-trigger qualification，并把worst-case outstanding、单帧复制/排空延迟、ring覆盖余量与完整call bound写进同一个envelope；preflight用编译后trigger schedule与该qualification envelope的最小帧间隔+安全余量拒绝过快scan。autonomous与API segmented都只arm一次完整run camera session。autonomous由唯一I/O owner读取现有raw FPGA STATUS/CURSOR证明冻结table logical terminal；API则按R-major/P-fast保存每个独立STATIC_ONCE PulseSession的physical terminal，只有全部R×P cell完成后才terminalize同一个camera capture。两者最终都用一个run级CameraRunEvidence，把expected trigger total与按Q0 reset/rollover语义计算的本session相机产出增量、完整frame/camera stamp/timestamp连续性及exact coverage一次对账。任一不符使整个run INVALID，重跑只能由用户或显式有限RetryPolicy发起。该合同不能像逐沿硬件tag那样定位具体错误，也不能绝对排除漏一帧同时多一帧的等量抵消；这是冻结硬件约束下明确接受的剩余风险，不能在文档中伪装成同等证明强度。

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

### 3.11 public object graph 仍可反向取得 raw hardware capability

当前 `NeutralAtomSession.devices/.camera/.sequencer`、umbrella re-export、PulseGUI 的 raw sequencer fallback、TaskConsole 保存整个 session、DeviceViewer/DeviceManager 接受旧 `DeviceSet`，以及教程直接构造 `QCMOSCamera/RemoteSequencer`，共同形成一条绕过 installation authority 的平行控制面。即使某个调用点目前“只读”，只要对象图里仍能到达 adapter、SDK handle、bound method 或 `prepare/fire/acquire/configure`，它就能绕过 runtime instance、ResourceClaim、quarantine、owner lane 与 safety journal；把旧 `DeviceSet` 包一层 proxy 或从 `__all__` 删除名称都不能形成安全边界。

根因不是缺少更多 wrapper，而是四类受众被一个 API 面混在一起：普通实验用户、adapter 作者、composition/runtime owner 和白盒测试。最终必须分开：实验用户只拿领域 facade 与 immutable catalog；frontend 只拿显式窄 command/view ports；adapter 作者只从独立 adapter SDK 导入合同；composition/runtime 私有持有 raw graph；测试若要观察 spy/raw adapter，必须在 composition 前自行保留引用。旧实现把 config/device 改变当成进程内字段替换，会制造“新目录配旧 authority”或“旧 descriptor 配新 connection”的混合代；目标架构不修补该热替换状态机，而是 safe shutdown 后由新进程完整重建。

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

`zlc_data` 不是新的 `common/utils`：它只容纳领域中立、headless、可序列化的数据语义和值上的纯算法。它拥有 Value/DataBlock、Axis/Validity/PointLayout、Selection/CommittedTransform、Reduction、FitSpec/BoundFit/FitResultBatch、closed model catalog 与同步 solver；`FitProblem` 只是包内瞬时 packing 值，唯一公开执行入口是 `BoundFit.run()`。它不知道 Hub、Run、Device、neutral artifact、Figure、Qt 或 Matplotlib。

FitResultBatch 的 canonical payload codec 属于 `zlc_data`，但 durable identity 必须由最窄的 artifact owner 持有。现在已经出现两个真实、同构且均为 FINAL dataset artifact 的 consumer：`CaptureArtifact -> FitResultBatch` 与 `ScanArtifact -> FitResultBatch`。因此 neutral 只保留一个 `FitResultArtifactRef/FitResultRepository`，manifest 的 closed tagged source union 只能是 `CaptureArtifactRef | ScanArtifactRef`，并逐种委托 source owner 的 canonical serializer、re-admission 与 exact DatasetRevision/schema binding；它不是可注册 source 的 generic Analysis repository，也不拥有 Fit 算法、Processor 或 Figure。第三种 source 若不满足相同 FINAL dataset/replay 合同，必须有自己的 adapter，不能向 repository 塞 registry/plugin/fallback。`fit.save()` 不要求 frontend，并使用 repository 安装级有界默认；GUI 另传当前窗口的精确剩余预算。Figure 保存仍使用 frontend-owned FigureArtifactRef；两者是不同 artifact kind。

“selector”必须拆开看：`Selection` 是可保存、可供 fit/processor 共同消费的数据语义，属于 zlc_data；鼠标手势、RectangleSelector、handle、overlay 和 interaction state 属于 frontend。`DataFigure` 明确属于 frontend，因为它是 render/public presentation facade。fit editor/overlay 属于 frontend，但它们调用 zlc_data 的唯一 fit 实现，不复制模型和结果 schema。editor 只能从 public immutable `fit_model_catalog()`/`fit_model_definition()` 取得模型与参数 metadata，并从 BoundFit 的 `parameter_definitions/parameter_units` 取得绑定后单位；不能导入 fit implementation submodule 或在 frontend 硬编码第二份模型表。

zlc_data 用 `bind_fit(FitSpec, expected DatasetSchema) -> BoundFit` 冻结并验证 fit/batch axes、CommittedTransform、model 与数值策略，但不捕获尚未产生的数据。`BoundFit.run(OwnedSnapshot) -> FitResultBatch` 是当前 interactive、offline/artifact 与未来确有消费者的 formal 路径共享的唯一执行值；OwnedSnapshot 同时持有 frozen DataBlock 与 exact DatasetRevisionRef，禁止 adapter 只传裸 block 丢掉 lineage。当前 baseline 不建立 `DatasetInputSlot`、generic `AnalysisStep`、`FitAnalysisDescriptor -> DataAnalysisProgram` 或 post-materialization workflow：真实用户需求只是对已提交 FINAL Capture/Scan artifact 明确打开Fit并显式保存结果，现有 `FitResultRepository` 已完整拥有这条边界。只有出现自动/headless preset或正式下游消费者后，才先实现“FINAL dataset artifact -> 独立 flat analysis Run -> 自己的一次 FinalCommit”；只有真实领域要求 scan 与 analysis 成为不可分割的一个提交结果时，才另行设计 composite commit。neutral 不得定义 `FitProcessor`、`FitOperator` 或 neutral-owned `FitAnalysisDefinition`，Workbench 的 `Add Analysis -> Fit` 只是 current zlc_data Fit capability 的本地产品入口。

`zlc_data.codec` 是该 bounded context 内 typed canonical-byte admission 的唯一 owner；当前只有确有 durable consumer 的 FitSpec/FitResultBatch 暴露 standalone current canonical bytes，并复用这一处 canonical round-trip 判定。Axis/DatasetSchema、Selection 与 CommittedTransform 只公开供外层 artifact 嵌入的 owner tree projector/parser；它们不各自预建无人消费的 bytes wrapper。大型 Value/DataBlock 更不经过通用 JSON/tree codec，真实持久化边界使用 bounded binary chunk/CAS。已排期的 AnalysisPreset/保存 FitSpec 通过 public `fit_spec_to_tree/from_tree` 或 `encode/decode_fit_spec` 委托同一 schema owner；FitResultBatch 的 tree projector 仍为 codec 私有，公开面只给 current canonical bytes，不能顺势恢复 generic result tree/ref codec。各领域类型仍由自己的 projector/parser 负责，primitive bytes 继续委托 `zlc_storage.canonical`。

`zlc_pulse` 是一个逻辑 bounded context，而不是“为了目录好看必须独立发布的产品”。它内部包含 `model`（PulseDocument/IR）与当前唯一生产 target `fpga`（TargetSpec/compiler/wire/host/RTL/build）。FPGA server、sim/build 和 neutral sequencer adapter 已是独立消费者，所以禁止它反向 import neutral；若未来出现第二硬件 target，再在 pulse 内抽出 target Protocol，baseline 不预建插件系统。

`zlc_storage` 只拥有两类窄基础设施：其一是无领域类型的 canonical primitive encoding/digest（canonical map/list/scalar、ndarray header/bytes、hash 与 framing）；其二是 bytes/blob/manifest 的校验、fsync、原子发布和最小维护。它不定义 universal ArtifactRef、领域 schema 或 artifact kind。frontend、pulse、neutral_atom 与 data 各自拥有 typed Ref/值对象 schema 和 `to_canonical_tree`，但最终 bytes/digest 必须委托同一个 canonical encoder；跨包嵌值对象必须调用 owner codec，不能手写字段顺序。canonical/non-empty text、SHA-256 text、integer、finite/positive real 等标量不变量同样只由该 primitive 模块实现；领域构造器调用它而不复制 `isfinite`/type/range 检查。仅明确的人类/外部输入 adapter 可调用单一 `normalized_text` 先 strip，机器 identity 一律使用拒绝空白改写的 `canonical_text`。这样避免四份 canonical JSON/float/ndarray/digest/validator 实现，又不建立能收容领域类型的 `common` 包。baseline 只实现经过 probe 的 local filesystem commit；复杂 GC、多后端/分布式锁等出现真实第二用例后再扩展。

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
Zou_lab_control.notebook -> zlc_storage
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

只允许三类明确的 application composition；其中只有普通用户入口公开领域 facade：

- `Zou_lab_control.notebook.connect(...) -> Experiment`：headless/notebook 实验根；
- `zlc_workbench.composition.create_workbench(...)`：desktop/Qt 应用根，待 GUI 纵切交付；
- `zlc_pulse.server_app` 与 FPGA launcher：可独立部署的 current pulse server 根。

`zlc_neutral_atom.bootstrap` 是 composition-private implementation package，包根 `__init__` 不导出构造器；当前 notebook root 只在 `connect()` 内惰性调用私有 installation factory。不存在 public `create_domain_experiment`、第二个 headless session root或顶层 `Zou_lab_control.connect` 转发。这样用户只有 `Zou_lab_control.notebook` 一个 canonical connect，领域包之间也不会多出 notebook 依赖层；若仓库保留独立 launcher，它只能调用上述 application root，不能形成第二套 facade/API 名称。

可复用 library 内禁止通过 FQCN、包扫描或 service locator 动态构造依赖。

每个真实或虚拟进程只有一个 composition-owned `InstallationRuntime`。它在启动期一次构造私有、membership immutable 的 `InstallationDeviceGraph`，并同时拥有 `ResourceArbiter`、`DeviceBroker`、`PersistentSafetyJournal`、`RunController`、owner lanes、typed `DeviceBindingResolver` 与只读 catalog。`InstallationDeviceGraph` 只是进程生命周期内的 exact role -> adapter owner/binding/close-order 图，不是旧 `DeviceSet`、registry 或 service locator；构造完成后不能增删、替换或重连成员。resolver 把 request 中的 `DeviceBinding(role/id, required capability)` 原子解析为：

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

#### 4.2.1 Process-lifetime InstallationRuntime 与 public DeviceCatalog

`InstallationRuntime` 是 process-lifetime composition authority，不进入 public object graph。它唯一拥有硬件图、运行 admission、broker binding、SafetyJournal owner lock、recovery coordinator 与 terminal shutdown。普通 runtime/adapter 调用者即使拿到 child 对象，也不能提前关闭 Run admission、broker binding 或 journal owner。quarantine 是 SafetyJournal 中一种 unresolved projection，不是第二个 journal 或第二套 authority。session 不能分别保存 raw graph、binding registry、catalog 与 facade：

```text
InstallationRuntime                    # 一个进程恰好一个，不公开
  installation_id
  runtime_instance_id                  # 每次进程启动重新生成，不复用
  lifecycle = STARTING | RUNNING | CLOSING | CLOSED | FAILED_CLOSED
  private immutable InstallationDeviceGraph
  ResourceArbiter / DeviceBroker / RunController
  PersistentSafetyJournal + owner lock
  typed DeviceBindingResolver
  typed domain facades/descriptors
  DeviceCatalogReader -> immutable DeviceCatalogView

InstallationDeviceGraph               # runtime私有，构造后membership不可变
  ordered adapter owners / owner lanes
  role -> BoundDevice
  deterministic reverse close order
  no public lookup / mutation / replacement API
```

启动顺序是唯一的 normal connection establishment 路径，而且发生在 Run admission 开放之前：

```text
load + exclusively lock PersistentSafetyJournal
-> replay unresolved blockers and construct ResourceArbiter
-> acquire backend/composition physical-owner proof
-> load and canonical-verify AssetMap
-> construct owner lanes and inert adapter owners
-> for each unblocked asset: open on its owner lane
   -> live identity readback + AssetMap match
   -> DeviceBroker.verify_identity -> bind -> capability probe
-> for each durable blocker: ordinary binding/admission remains forbidden;
   only RecoveryController.begin -> recovery-only open/bind -> explicit complete/abort
-> freeze InstallationDeviceGraph, resolver, descriptors and initial catalog
-> open RunController/public command admission
```

ResourceArbiter 不提供 connection-establishment lease：正常 open/bind 发生时还没有普通 Run admission，互斥由 process-level physical-owner proof、owner lane 与 `DeviceBroker.bind` 的 current/active/recovery 检查完成。若某 SDK 的 open 本身会改变危险输出，它必须在 adapter 的启动/恢复安全 recipe 中得到显式 hazard/safe 处理；不能用一个名为“只读连接”的 lease 洗白。任一 open、identity、AssetMap、capability、recovery 或 graph freeze 失败都不发布 `Experiment` 或 drive facade，composition 按 §12.7 安全关闭已经建立的子集并使启动失败。

当前已交付的 target-owned virtual root 是这条真实启动顺序的明确非物理例外，而不是可复制到真机的捷径：它先构造并探测 deterministic in-process atom array、camera、sequencer与broker binding，成功后才取得本地 safety journal authority；这些对象没有外部物理输出或跨进程竞争者。即便如此，composition claim仍是**一次进程生命周期、永不复用**：正常 `CLOSED` 后不能在同一进程重新 compose；startup rollback任一close失败会强引用保留整个partial graph/journal authority并永久拒绝后续compose，直到替换进程。真实adapter、remote FPGA与qCMOS不得使用此例外，仍必须按上面的journal/physical-owner/AssetMap顺序启动。

当前virtual graph直接消费 `zlc_pulse.load_deployed_pulse_target()`，不再把旧 `PortCatalog` 投影成第二份拓扑；canonical clock来自同一checked-in FPGA config。标准物理接线固定为cooling `ch00/ch01`、probe `ch03`、trap `ch09`、camera trigger `ch11`，每条均须是deployed target中的单lane digital port。trap只作为camera背后的私有物理模型，不进入public catalog；public catalog只含camera/sequencer immutable `DeviceInfo`，关闭顺序固定camera→sequencer→trap。

`InstallationDeviceGraph` 只在 composition/runtime owner lane 内可达，也不能通过 debug property、generic resolver、callback closure 或 frontend ViewModel 泄漏。这里的 typed facades 是 runtime-instance-pinned、immutable binding surface/descriptor，不包含用户可变的 calibration convenience pointer或UI state。public `Experiment`只发布`device_catalog`与稳定的领域 convenience facade；每个 facade 操作在一个 composition 临界区恰好取得一次当前 RUNNING runtime snapshot，据此构造并冻结 request/binding stamp，不能分别读取 descriptor 和 runtime 指针。所有依赖标定的请求都显式接收 `CalibrationArtifactRef` 并在构造时与 binding/model 一起冻结；headless domain session 本身不是普通用户的硬件 service locator。

graph 的“immutable”指 role membership、adapter owner、binding membership 与 close order 不可改；其中 live adapter/connection 当然会在 owner lane 内部改变物理/driver状态，但这些对象不向 graph 读者开放。transport disconnect、device removed、identity mismatch 或 capability invalidation 使当前 binding fail closed；baseline 不在同一进程透明 reconnect、替换 binding 或重新开放 admission。需要改变 connection identity、adapter topology 或 installation config 时走 §4.2.2 的安全关闭与新进程启动。

catalog 是观察值，不是换名后的旧设备容器：

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

`runtime_instance_id`、每个 binding 的 `binding_instance_id` 与 `catalog revision` 不混用。runtime instance 每次进程启动重新生成；broker 为每次成功 startup/recovery bind mint 一个不可复用的 binding instance id；同一 binding 下纯观察 health 变化只推进 catalog revision。immutable installation graph 内不存在第二个与它一一对应的 local `connection_generation`/`binding_id`。任何在 shutdown 开始前排队但尚未进入合法 Run 的 command 都绑定原 runtime instance，并在 CLOSING 后以零 adapter 调用失败；新进程绝不接受旧 `DeviceRef`、request、RunPlan、binding 或 capability。Pulse RPC server 自己的 `server connection generation` 是跨进程 transport 事实，继续由 `zlc_pulse` owner 独立维护，不能与本地 binding instance 混名或合并。

generic catalog 只回答“安装中有哪些角色、当前观察状态是什么”。领域事实由具名、冻结的 facade descriptor 单源提供，例如 `Experiment.pulse.target` 的 clock/port/target facts、`Experiment.readout.camera_descriptor(binding)` 的 frame/trigger contract、`Experiment.trap.geometry` 的 site/grid geometry。禁止把这些异质事实重新塞进任意 `snapshot: dict`，也禁止 frontend/Definition 按 role 字符串从 catalog 找到一个对象后调用领域方法；否则 catalog 会退化成新的 service locator。

#### 4.2.2 配置边界：安全关闭后由新进程重建

baseline 没有进程内 config/device/virtual-real hot swap，也没有 `InstallationCandidate`、available/unavailable union、swap intent、transition generation、partial new graph 或 reconnect coordinator。以下变化都必须执行 §12.7 的 safe shutdown、退出当前进程，再由新进程从 canonical config 与 AssetMap 完整重建：

- AssetMap、physical asset、adapter kind、endpoint 或 topology 改变；
- real/virtual backend 改变；
- 会改变 installation graph、owner lane 或 connection identity 的 machine/device config；
- 需要重新 open/reconnect 已失效 binding 的故障处理。

实验 request、pulse parameter、camera working point、calibration ref 与 panel state 不属于 installation graph，可以按各自 typed contract 在 Run 边界改变。普通 config API 只能产生“需要重启”的诊断，不能关闭旧 connection 后在同一进程安装新 graph。UI 可以请求 safe shutdown，并在进程完全退出后由外部 launcher 启动新进程；Qt callback 不执行硬件 close、不确认 safety，也不能把“新窗口已出现”当作旧 runtime 已终止。

同一 RUNNING runtime 内的 catalog 异步通知只描述 health/observation revision，不承担 authority 事实，也不能有“先读、后订阅”的丢失窗口。`DeviceCatalogReader.snapshot()` 与 `watch(after_revision)` 在同一 owner 临界区线性化并返回完整不可变 snapshot；UI 可 coalesce 到最新 revision，检测 gap 时重新读取 current snapshot。shutdown 开始后 reader 只报告 terminal runtime lifecycle，不能发布一张看似可驱动的新 catalog；hardware safety 从不等待 subscriber ACK。

#### 4.2.3 adapter 作者、测试与 simulation 的命名空间

普通 `Zou_lab_control.neutral_atom`/`Zou_lab_control` umbrella 不导出 adapter base、concrete adapter、DeviceSet、loader、server bootstrap 或 raw pulse helper。adapter 作者使用明确的 `zlc_neutral_atom.adapter_sdk` 合同与 parameterized contract kit；virtual/fault-injection 测试使用 `zlc_neutral_atom.testing`/`simulation`；真实 server 使用自己的 application/CLI bootstrap。adapter SDK 可以公开最小生命周期/Port 实现合同，但不能成为普通 Experiment 对象图的一部分，也不能提供 `lookup=globals()`、包扫描或运行时任意注册逃生口。真实adapter的构造/open/drive还必须消费composition owner签发的不可伪造owner capability并绑定owner lane；仅从owner module导入类不能得到可运行的真实硬件对象。Python反射不作为恶意安全沙箱，但普通协作代码绕过authority必须在构造或第一次drive前fail closed。

### 4.3 Data 与 Frontend 内部层次

```text
zlc_data <- zlc_frontend.figure
                  |          \
                  v           v
      zlc_frontend.render   zlc_frontend.matplotlib_render + render_style [render]
                  ^
                  |
      zlc_frontend.qt_widgets [qt]

      zlc_frontend.notebook_integration [notebook, lazy IPython leaf]
```

所有权：

- `zlc_data`：Axis、Value/DataBlock、Selection、DataTransform、Reduction、Fit；
- `frontend.figure`：ViewIntent、ViewSpec、FigureDocument、FigureEvaluator、FigureArtifactRef、codec；
- `zlc_frontend.render`：immutable raster/presentation DTO 与并发中立的 front/presenter 合同，不加载 Matplotlib/Qt；
- `zlc_frontend.matplotlib_render` + `render_style`：Agg renderer、从旧 frontend 完整迁入的字体/几何/palette/publication style、串行 Matplotlib compose lane 与 DataFigure 的可选 render backend；
- `zlc_frontend.qt_widgets`：Qt application/window lifetime、Qt event adapter、immutable raster board、Qt/QPainter style token 与通用 widgets。
- `zlc_frontend.notebook_integration`：`%matplotlib widget`与IPython display hook的惰性adapter；render owner不探测IPython。Qt leaf另可在显式`ensure_qt_app()`时惰性执行`%gui qt`以维持notebook事件循环，但导入`qt_widgets`本身不加载IPython。

neutral_atom 只依赖 `zlc_data`，不依赖 frontend 的任何层。

`zlc_data` base 只依赖 NumPy/必要 solver与 `zlc_storage.canonical`，不加载 repository I/O、Matplotlib/PyQt。`zlc_frontend` 的 headless figure 与 raster/presentation DTO 层依赖 data+storage；Matplotlib backend/style/font 放在 `[render]` optional extra，PyQt/Fluent/Qt board 放在独立 `[qt]` extra，Matplotlib-widget/display类IPython hook只在惰性notebook integration leaf。Qt leaf 可以消费 base `render` DTO，但不得导入 Matplotlib implementation；仅在调用`ensure_qt_app()`时才允许惰性探测IPython的Qt事件循环。完整 Workbench 同时安装 `[analysis]+[render]+[qt]`。`zlc_neutral_atom` 依赖 data、storage 与必要的 pulse public API；notebook 的显示 extra 依赖 `zlc_frontend[render]`，其可选 `[workbench]` extra 才懒加载 `zlc_workbench` GUI launcher。`zlc_frontend`、`zlc_frontend.figure`、`zlc_workbench`、`Zou_lab_control.workbench` 与各 application package root 顶层 import 都不能加载 Matplotlib backend、PyQt/qframelesswindow、repository backend 或真实 hardware adapter；调用者必须显式进入 `zlc_frontend.matplotlib_render`、`zlc_frontend.qt_widgets` 或 notebook integration leaf。

#### 4.3.1 Qt 组件单一 owner 与迁移复用契约

旧 `frontend/qt_fluent.py` 已经是经过真 GUI 使用的完整组件层，不是待重写的样式草稿。迁移必须把它的保留行为整体移入 `zlc_frontend.qt_widgets`，删除旧文件且不留 shim/re-export；application/Workbench 只从 `zlc_frontend.qt_widgets` 的 curated public facade 取件，禁止 deep import 或从 `zlc_frontend` 根重导出 Qt symbol。`QtImageBoard/QtOwnerWake` 同属这个 Qt leaf，不能继续平铺在 headless frontend 根。

先取件表不是新的 widget taxonomy，而是把旧组件已经承担的语义写清：

| 需求 | 先使用 | 边界 |
|---|---|---|
| QApplication、缩放、窗口可达性、保活 | `ensure_qt_app/set_fluent_scale/screen_fit_window_size/center_window_on_primary_screen/retain_window/release_window` | 首次 app 只能在 Python main thread 创建；异步窗口只在 committed close 释放 |
| 普通 Setting/Edit 行 | `FluentSectionLabel + FluentSettingRow + setting_label_width` | 一列标签+一列控件；不得退回 `QFormLayout` 另造风格 |
| 稠密 authoring grid | `FluentFormGrid/FluentLabeledField/Metrics` | 多列、跨行或需要统一 row metric 时使用；不能拿它替代简单 SettingRow |
| 路径、只读值、scan binding | `FluentPathEdit/ReadoutEdit/ScanLineEdit/TreeComboBox/TriStateToggle` | edit buffer 与 committed resource 分离，验证成功才提交 |
| 状态与提示 | `FluentStatusStrip/StatusDot/ScanDot/muted_note_label` | presentation-only，不持有 Run/领域状态机 |
| 模态消息、确认、单行文本 | `fluent_message/fluent_confirm/fluent_text_prompt` | 只返回用户的临时选择/文本；不得直接提交领域对象或替代 validation |
| 容器与滚动 | `FluentGroupBox/TabWidget/ScrollArea/Frame/Popup/Window` | 按真实 lifecycle 组合，不因外形相似强套同一个顶层 Window |
| raster/owner wake | `QtImageBoard/QtOwnerWake` | GUI 只消费 immutable front，不拥有 renderer/Run |
| 批更新与 Qt hygiene | `batched_updates/signals_blocked/apply_fluent_scrollbars` | 禁止每个窗口复制 signal blocking/scrollbar QSS |

**Legacy abstraction salvage gate（每个后续W纵切写代码前必过）：** `qt_fluent.py`不是旧树里唯一已经解决DRY的地方。切片必须先列出相关旧模块的等价物、真实consumer、依赖与行为合同，再分三类处理：dependency-closed且presentation-only的实现整体move到`qt_widgets`或对应frontend owner；绑定旧`ParamDecl/Hub/LogicNode`等领域对象的composite随其current Definition/intent纵切解依赖并迁到领域composition owner；确认无current语义且最后consumer已迁走的才删除。禁止current反向import legacy，也禁止跳过审计后凭外形重写一份。已知但仍须按真实consumer逐切片核验的库存包括`frontend/param_widgets.py`的`ParamWidgetHandler` registry（build/read/write/is_empty/refresh）、`RateLimitedApply`、`RefreshProviders`、signal picker helpers，以及旧Pulse/TaskConsole中的领域composite；列名不是原样保留承诺，行为合同与单一owner才是保留对象。

这里的复用有严格优先级：**MOVE旧实现源码 > 从旧源码做最小ADAPT > RETIRE；从空白重新实现是例外，不是默认。** 只要旧实现的依赖可以通过参数/typed snapshot切断，就必须以旧源码为基线搬迁并在review中展示语义差异；不能因为current类型改名、包名不同或想写得“更干净”就平行造一份。只有旧实现把领域authority、全局状态或已废弃语义焊进控制本身，且机械矩阵逐项说明为何无法dependency-close时，才允许保留行为oracle后重写接缝；新实现还必须列出旧源码中每项已挣得行为是保留、强化还是明确退役。过渡期也不得让legacy与current各自长期拥有一套通用handler：能切断依赖的公共实现迁入current owner后，尚未迁走的legacy consumer应直接调用这个owner或留在明确、有删除切片的adapter island；不得用re-export/shim伪装完成搬迁。

这道门必须产生一张随切片提交的机械矩阵，而不是只写“已参考旧实现”：`legacy symbol -> 当前全部consumer -> import/状态依赖 -> 保留的行为合同 -> MOVE/ADAPT/RETIRE -> current owner -> last-consumer删除切片`。只要一个consumer、依赖或删除切片未知，就不能重写或删除。尤其“新窗口已经能显示”不能证明旧composite可删；必须证明它的参数域、动态刷新、commit/rollback、精度、快捷键、生命周期和全部产品面都已有current替代。

首轮库存不是只有基础控件，后续纵切至少要逐项核验以下已挣得结构：Pulse侧的`PulseStateUIManager/PeriodCard/PulseDragContainer/RepeatBracket/ChannelNamesPanel/ChannelPanel`；TaskConsole侧的`MeasurementPanel` form loop、`LogicNodeEditor`的Setting/Edit同源、`AnalysisControls/_FitFixSeedEditor`、grouped signal tree helpers、`PanelConfig + pack/drop_index + _PanelBoard`以及`PanelCard/PanelEditor`的单writer同步；figure侧的`BaseLivePlot/GridPlot/GridCell`增量artist与grid focus、`selectors.py`的ROI/cross/zoom/disconnect、`DataFigure/SavedFigure`交互与重放、`calibration_report.py`的site histogram/PSF/site-map布局、`figure_viewer.py`的artifact浏览壳，以及DeviceManager的统一handler/readback/限流。这个清单是salvage入口，不是要求整类照搬：纯presentation或纯layout算法优先MOVE；携带旧Hub、LogicNode、mutable config、raw hardware或旧artifact状态的类必须拆出已验证的view/interaction算法，再接到current typed owner与EditorSession。每个未来monitor/rolling/calibration/fit/selector窗口在写新widget前都要先给这张库存追加真实consumer和裁决，不能等窗口写完再做“复用清理”。

**W-UI1声明式表单抢救是后续窗口按consumer分段通过的硬前置，不是末尾清理，也不是一次预造旧全集。** W-UI0只完成Fluent primitive、style、render与window lifecycle owner；它没有完成旧树早已存在的“字段声明 -> 单一handler registry -> Setting/Edit共用”机制。W-UI1a先交付W3已经消费的static scalar/typed choice/exact populate；W-UI1b必须在monitor/Measurement/Processor出现动态stream/device选项前交付typed option snapshot + revision + stale-draft拒绝；W-UI1c必须在DeviceManager或明确live-control出现前原源码迁入`RateLimitedApply`与teardown flush。某一段没有真实current consumer时不得为“完整”提前搬JSON、Hub expression或万能schema，但对应产品纵切也不得先写一份临时控件绕过门。

旧`frontend/param_widgets.py`中已经由多个真consumer挣得的通用内核必须以现有源码和行为oracle为迁移输入：`ParamWidgetHandler`的`build/read/write/is_empty/refresh`五操作封闭合同、统一change wiring、no-`eval` coercion、label/unit/required组合、choice/path/scalar读写、selection-preserving refresh、exception-safe signal blocking，以及`RateLimitedApply`的per-key leading+trailing与teardown flush。current为了保证full-state populate真正原子，在这五项上只允许增加一个公开且abstract的`normalize(field,value)`预变更操作；所有字段先normalize成功才可开始写Qt，不能把它藏成可忘记override的私有helper。禁止在W窗口里重新建立按字段类型分支、独立number parser、第二套populate/reset或逐字段Setting/Edit接线。

终态数据流固定为：

```text
typed Request/Config schema owner
  （唯一拥有value type/default/unit/range/required/static choices/description）
        |
        | Workbench中的use-case显式projector；普通import，不按schema-id动态dispatch
        v
zlc_frontend.form.FormSpec / FormFieldProps
  （headless、immutable、仅presentation projection；增加label/group/order/widget hint）
        |
        v
zlc_frontend.qt_widgets.FluentParameterForm + closed handler mapping
  （build/read/write/is_empty/refresh/full-state populate；不Apply、不持有领域对象）
        |
        | keyed draft values
        v
Workbench EditorSession -> typed Request/Intent constructor -> owner validator
  -> base_revision检查 -> atomic Apply
```

这里以§10的轻量Definition规则为准：`TaskDefinition/MeasurementDefinition/StreamProcessorDefinition`只保存catalog identity与`request/config_schema_id`等稳定身份，不复制字段默认值或GUI schema。字段语义在对应typed Request/Config owner旁边有且只有一份声明；Workbench通过显式owner函数取得并投影，不能凭dataclass signature/AST反射，也不能把schema-id变成service locator。`FormSpec`是一次界面投影，不持久化、不参与artifact/config fingerprint、不成为第二validator；projector必须机械证明key全集、默认值、range和choices与owner声明逐项相等。

旧`ParamDecl`本身不作为跨包终态公共基类，因为它混合了领域语义、`display/segmented/path dialog/depends_on`等presentation选择，以及`$device:`、全局Hub signal、`signal_expr`、旧`pulse_slots`等legacy专用种类。复用按下表裁决：

| 旧能力 | 处理 | current边界 |
|---|---|---|
| scalar `Float/Int/Bool/Choice/Text/Json` handler、五操作ABC、统一wiring/form loop | MOVE/ADAPT，尽量保留既有实现 | current formal数值默认non-quantizing；非法populate必须fail-closed，不能沿用“异常就忽略” |
| `PathHandler` | ADAPT | 复用`FluentPathEdit`，但资源解析、base dir与filter由Workbench注入；Qt owner不导入旧`_paths` |
| `RateLimitedApply` | MOVE，按真实consumer使用 | 只用于preview/display或明确的live-control presenter；正式Task/Scan/fit authority仍经draft+Apply，绝不把它当硬件时序或Run提交 |
| `RefreshProviders`/signal tree交互 | ADAPT到immutable typed option snapshot+revision | 未来由Workbench把`StreamId/generation/schema/label`投影成choice；不恢复全局Hub裸字符串authority |
| `AxisRangeHandler` | 暂留legacy island，出现current同义consumer再迁 | Formal scan继续使用具名`ScanPointTable`，不得退回匿名`(min,max,points)` |
| `DeviceRefHandler`/`$device:` | RETIRE | current设备选择只经typed `DeviceBindingResolver`/role projection，GUI不保存DeviceRef或service locator |
| `SignalExprHandler/_SignalExprWidget` | 不进入formal current y | monitor/display若有真实需求另做typed presenter；scan/fit/存档禁止任意全局Hub表达式 |
| `PulseSlotsHandler/_PulseSlotsWidget` | 不再是generic param kind | PulseDocument、W2 Scan/API authoring与W3 program presenter各自使用current owner contract |
| `MeasurementPanel/PanelCard/PanelEditor/AnalysisControls` | 类不整体搬；提取已验证的form loop/state合同 | 领域Run/Hub依赖留在相应纵切；fit/selector分别适配current `FitSpec/Selection` |

迁移期间为了不复制实现，通用handler内核只允许有一个current owner；尚有真实consumer的legacy专用handler可留在明确的legacy adapter island并依赖该内核，但不得从旧路径re-export形成shim。最后consumer迁走时连adapter island与旧symbol一起物理删除。`zlc_frontend.form`和`qt_widgets`都不得导入neutral/workbench/旧`Zou_lab_control`；Workbench作为唯一双向composition层完成投影。

`FluentParameterForm`只是一层薄的字段集合：按有序`FormSpec`构造现有`FluentSettingRow`，保存`key -> (handler, widget)`，提供exact-key `read_all/write_all/is_empty/refresh`和原子、signal-blocked的full-state populate。它不拥有section workflow、visibility状态机、EditorSession、RunHandle、repository、Definition、artifact或硬件写，也不递归解释任意nested schema。Pulse editor、PulseDocument/API segment table、CalibrationArtifactRef、DataTransform、ROI/selector、fit axis/batch/reduction、device connection、resource conflict和安全确认继续由显式presenter组合；不得为了提高“自动生成率”把它们塞进JSON schema或万能widget plugin。

“显式table/grid presenter”只豁免行列结构、增删、selection与整表commit，不豁免普通叶值。W2 Pulse scan/API table、W3 segmented API table以及form的`number` kind必须共同调用`zlc_frontend.form.parse_number_text`，保留作者输入的`int | float`类型并统一拒绝表达式、NaN与Infinity；不得各自`float(text)`或复制regex。类似地，未来日期、unit value或StreamId一旦有第二个table/form consumer，应抽叶coercion/delegate，而不是抽generic table framework。

W1没有request editor，不强套表单；W2的period/DAC/delay/repeat/scan-table是显式Pulse presenter，只在单字段行为相同时取handler，duration/delay保持non-quantizing；W3保留一个scan-specific `ScanIntentForm`、两个surface实例和同一个card-owned `ScanEditorSession`，但role、budget、deadline、普通choice/text以及PulseDocument动态API常量必须经同一FormSpec/handler路径。Pulse load/mode、segmented table、calibration ref、authoritative transform和SITE axis/display presenter继续显式。Setting/Edit两实例必须消费同一个FormSpec，populate覆盖全部exact keys（包括disabled/hidden），并从同一committed snapshot回填；不能只因复用了同一个Python类就宣称DRY完成。

W1/W2/W3及后续所有current Workbench只能消费current owner：基础Qt行为来自`zlc_frontend.qt_widgets`，字段语义来自typed Request/Config owner，标量叶coercion/handler来自`zlc_frontend.form`与`FluentParameterForm`。legacy adapter只服务尚未迁走且已登记删除切片的旧consumer，不能被current窗口导入；一旦某个current窗口又出现本地scalar parser、默认值/range/choice副本、第二套Setting/Edit wiring或从legacy路径取件，该纵切即未通过复用门，不能以“UI已经可用”宣称闭包。

W-UI1a机械ratchet至少验证：每个closed scalar kind恰有一个完整`normalize/build/read/write/is_empty/refresh` handler；每个owner auto-field恰好投影一次，未知/遗漏/重复key失败；Setting/Edit字段key集合一致且刷新不触发Apply；typed value、required、bounds、static choices、unit与非量化float round-trip；current W form和table cell对已覆盖kind不存在独立parser/default/range/choice literal；`qt_widgets`无neutral/workbench/legacy import。W-UI1b另加：option snapshot带owner revision/generation，仍合法选择被保留，已失效的committed值可见但不可提交，旧revision draft明确stale而不是静默选first。W-UI1c另加per-key leading+trailing与close/Apply flush。旧通用scalar handler在过渡期只能是调用current内核的有删除切片adapter，不能长期保留第二套实现。复杂显式presenter要逐项登记豁免原因，不能用“custom UI”一句话逃过审查。

复用顺序是硬约束：

1. 先使用上表与基础 Fluent controls；Setting/Edit 能由 `FluentSettingRow + setting_label_width` 表达时禁止另起 `QFormLayout` 风格，需要共享label列的稠密表单（如finite repeat的Start/End/Count）直接用`FluentFormGrid`。
2. **可替换性以行为合同为准，不以长相为准。** 控件替换必须保持 value range、float precision、commit/rollback、signal ordering、keyboard/wheel 与 enabled semantics。presentation-oriented `FluentDoubleSpinBox` 默认可按显示长度量化；pulse duration/delay 等权威值必须用其 non-quantizing 模式。若现有组件确实不能保持合同，raw Qt complex table/value editor 可暂留并登记真实缺口，不能为了复用率改变物理值；OS resource picker可继续用`QFileDialog`，普通文本/确认不得退回`QInputDialog/QMessageBox`。
3. `FluentPathEdit` 的 typed/selected path 是 edit buffer，不是已提交的 PulseDocument/calibration/resource。选择后 validation 失败必须恢复最后 committed path 与原 artifact/document；不得显示“坏路径+旧权威对象”。
4. 没有合适组件时，先证明至少两个真实 consumer 与一致的交互/视觉语义，才把新的 presentation-only 高层 widget 加入 `qt_widgets`；W2的三处文本请求与两处确认因此只补了两个薄helper，而不是dialog framework。一个 widget 不得持有 RunHandle、repository、Definition、scan/calibration intent 或领域 revision。
5. 相似外形不等于相同 lifecycle。W1/W2/W3 的 cancel/reap/close 协议仍由各自 application shell 拥有，不能为了统一外壳强套 `FluentWindow` 或抽 `GenericRunPanel/RunControlStrip`；launcher 保活，只有各窗口完成自己的 cancel/reap/worker shutdown 后才在 committed close 释放；不得把普通 `QWidget.close()`（可能只隐藏、不触发 `destroyed`）误当成保活表已经清理。
6. 所有顶层 W launcher 在构造 widget **之前**统一解析 process-wide Fluent scale，构造后先`screen_fit_window_size`，`show()`取得真实native frame后再`center_window_on_primary_screen`；首次 QApplication 创建若不在 Python main thread直接拒绝。Setting/Edit 的大内容区必须使用 `FluentScrollArea`，validation/status与 Apply/Cancel footer留在滚动区外，保证最小支持视口 `800×600` 与 Windows 125% DPI 下动作始终可达；验收看`availableGeometry.contains(frameGeometry)`而不是只看client width/height，fixed `resize(...)` 不是可用性证明。
7. 动作颜色沿用成熟语义，不新增 role enum：`Start/Run/Apply=GREEN`、`Stop/Hold/Load/Paste=ORANGE`、`Cancel/Remove/Clear/secondary navigation=GREY`、普通非危险主操作=`ACCENT`。颜色只是可辨提示，enabled/Run state仍由领域 owner决定。
8. Qt chrome/QSS/window metrics/QPainter token 只在 `[qt]` owner。旧 `frontend/style.py` 的字体资产、geometry、palette与publication defaults整体迁入唯一 `[render]` owner并物理删除旧文件；所有公开 nested token必须 immutable（tuple/mapping proxy），不得只做外层只读。Agg不能从Qt读取颜色或`fluent_scale`，Qt也不能用Matplotlib rcParams决定控件视觉。
9. Matplotlib `rc_context` 修改的是进程全局状态，不是线程局部变量；所有迁入current render owner的Figure construction/draw/save/clear必须持有同一个re-entrant compose lock，DPI必须显式传入Figure且与memory preflight一致。`DataFigure.render()`返回caller-owned Figure后，后续第三方draw/mutation不再宣称受产品lane隔离；产品PNG/export必须回到render owner完成构造+draw，并经同一个release owner断开Figure/Canvas/artist循环。`Experiment.figure(memory_limit_bytes=...)`必须把总render预算冻结进返回的`DataFigure`，`_repr_png_`、默认render/PNG/export都消费该预算；逐次override只能收紧、不能取消或放宽。尚未迁出的旧共享-Figure RenderLoop只允许留在§12.5的`SerializedLegacyAggBridge`隔离岛，不能据此宣称已满足current lane合同；最后consumer迁走时连岛删除。直接第三方Matplotlib在产品draw期间并发修改rcParams不属于支持合同。
10. W/application module 禁止散落 hex、局部公共控件 stylesheet 或复制 QApplication retention。若领域 QPainter 需要新颜色，先给 Qt style 增加语义 token；Agg艺术值进入唯一 render style。

机械 ratchet 同时验证：旧 `qt_fluent.py`、旧 `frontend/style.py`、旧 `_matplotlib_render.py` 与旧 `qt_board.py` 不存在且 production 旧 import 为零；production只能从curated `zlc_frontend.qt_widgets` facade取件，禁止deep import；`zlc_frontend`内PyQt/qframelesswindow import只在`qt_widgets/**`；`qt_widgets/**`不导入Matplotlib、neutral/pulse/workbench或旧`Zou_lab_control`；render backend/style不导入Qt/IPython，notebook integration import本身也不加载IPython；fresh import的所有package roots不加载optional backend。所有现在及未来的`Zou_lab_control/workbench/_*.py` Qt shell对已有Fluent等价控件不得重新构造raw Qt或`QInputDialog/QMessageBox`，必须使用scale/screen-fit/center与retain/release；repo-root current `figure_viewer.py/pulse_gui.py/task_console.py`同样禁止自行构造`QApplication`。真实W launcher关闭后retention registry回到基线。当前用户/维护者手册同样扫描旧owner名，不能让文档继续教人走已删除路径。style并发恢复、明确DPI canvas尺寸、font package-data、权威float round-trip、路径失败回滚、sticky footer、动作颜色、端到端render预算与GC-disabled连续export/compose/renderer-construction fault资源释放都有current focused oracle。历史测试不因迁移旧路径而维护。

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

`BoundFit` 是由 FitSpec 与预期 DatasetSchema 确定性绑定的进程内执行值，不进入 artifact、pickle 或 FQCN import。持久化只保存 FitSpec/CommittedTransform、具有完整数学语义的 `model_id`、constraints、numeric policy、AxisSpec 与 input lineage；initializer/solver 都是当前 closed implementation 的实现细节，不再复制成字符串 identity 或对象 digest。重放时调用当前 zlc_data `bind_fit`；public FitSpec owner codec 只接受当前字段集合，不维护 model/algorithm 版本号、兼容 reader 或升级器。zlc_data 只公开 immutable closed catalog view，不提供可变全局 model/analysis registry，也不扫描 entry point。

### 4.4 Pulse Preview 边界

Pulse bounded context 不导入 frontend。通用 `FigureDocument` 的 image/curve/histogram/meter primitive 无法无损表达同板 mixed digital step、DAC ramp、period、repeat bracket 与 nominal scan reference，因此 Pulse Preview 不再伪装成通用数据 Figure。当前路径是：

```text
zlc_workbench.project_pulse_preview(PulseDocument)
  -> zlc_pulse.compile_pulse_artifact(STATIC_ONCE | STATIC_REFERENCE_POINT)
  -> zlc_pulse.build_pulse_timeline(PulseDocument, CompiledPulseArtifact)
  -> immutable PulseTimelineDocument
  -> Workbench 的窄 timeline renderer / Qt canvas
```

`PulseTimelineDocument` 只是一组 renderer-neutral、进程内 immutable row/segment/annotation 值，不是新的持久格式、版本化 schema 或可编辑真相；保存/加载仍只有 current `PulseDocument`。离线 editor 因此只需文档自带的 target/time grid，不构造假 `DeviceRef`或 `PulseTargetDescriptor`；在线 composition 只在执行边界另外注入 live descriptor/facade，并通过显式 `bind_target()` 将文档重绑到当前 target。projector 只接收携带 `source_document_digest` 的完整 `CompiledPulseArtifact` 并核对源文档，禁止把 A 的 TargetIR 与 B 的标签、visible rows 或 digest 拼接。scan-authored document 的默认预览明确编译 nominal literal reference，绝不把第一行或任意孤立 scan row 冒充具有前序 DAC carry 的精确物理点。API/scan nominal reference 在画面中可见标注；但 nominal API 只允许预览，hardware `PulseRunRequest` 必须先通过 `resolve_api_parameters` 显式提供值并清除已解析声明，仍有任何未解析 API parameter 即拒绝 Run。frontend codec 不认识 PulseDocument；未来 export 只把同一 timeline 渲染为不可变 raster/vector，不反向恢复 authoring document。

### 4.5 Notebook-first Experiment 门面

notebook 不是“绕过正式架构的调试入口”，而是一等 application composition root。`zlc_neutral_atom.bootstrap` 只保存不公开的装配实现；`Zou_lab_control.notebook` 提供薄 `Experiment` 门面，把私有 `InstallationRuntime`、repositories、RunController 与领域 facade 显式组合：

```text
Experiment                         # notebook/application facade，不含领域算法
  .readout / .pulse                # 语义子门面
  .device_catalog                  # immutable DeviceCatalogView，只观察、不驱动
  .trap.geometry                   # typed immutable domain descriptor
  .pulse.target                    # clock/port/target descriptor，不是 raw sequencer
  .readout.camera_descriptor(...)  # frame/trigger/config capability descriptor
  .run(request)                       # public只收declarative Request
  .start(request) -> RunHandle        # RunHandle不暴露plan/Port
  .inspect(request) -> PlanDescriptor # 纯摘要，不含capability
  .fit(capture_or_scan_ref, spec|model=...) -> FitExecution
  .fit_gui(capture_or_scan_ref, model=None, committed_transform=None) # 同一DataFigure Analysis host的直达入口
  .load_fit(FitResultArtifactRef) -> AdmittedFitResult
  .figure_document(result_or_ref, occupancy_output=None) # headless projector
  .figure(result_or_ref, occupancy_output=None)          # 需要 zlc_frontend[render]
  .figure_gui(result_or_ref, occupancy_output=None)      # typed交互查看；Capture/Scan可Analyze->Fit，saved-fit进入可显式Refit的exact GridPlot
  .readout.load_calibration_computation(calibration_ref) # 同次decode返回artifact+report
  .readout.calibration_report_gui(calibration_ref)       # 懒加载冻结标定报告，只读显示
  .readout.calibration_gui(calibration_request)           # 编辑显式request，成功即原子FINAL commit
  .readout.calibration_edit_gui(calibration_ref)          # 从exact ref重开，另存为新artifact
  .readout.occupancy_cell_gui(occupancy_ref, selection=...) # 精确同shot物理位点图与具名cell导航
  .readout.camera_monitor_request(camera_role="monitor_camera") # 声明free-running显示源
  .readout.inspect_camera_monitor(request)               # 只读schema/working-point摘要
  .readout.camera_monitor_gui(request=None, ...)          # 懒加载continuous raw image产品
  .task_console() / .pulse_gui()    # 懒加载 notebook[workbench]，否则入口不存在/给安装提示

neutral domain Result
  typed values/artifact refs
  no FigureDocument/DataFigure/Qt/Matplotlib object
```

当前纵切已实现 `connect(config="virtual", repository=...)`、immutable `device_catalog`、`.pulse.target`、declarative capture request、`start/run/inspect`、capture load、capture-fit save/load，以及三条显式分析链：“已提交 `CaptureArtifactRef` -> `CalibrationArtifactRequest` -> 原子提交/重开 `CalibrationArtifactRef`”、“已提交单event `CaptureArtifactRef` + `CalibrationArtifactRef` -> `DetectionRequest` -> 原子提交/重开 `OccupancyArtifactRef`”和virtual/offline短`exp.readout.sitemap(frames=...) -> CalibrationArtifactRef`。W2a 进一步接通 `.pulse.request/inspect/start/run`：finite pulse 返回轻量 `PulseRunResult`；continuous HOLD 没有伪造 logical terminal，只能 `start()` 后由同一个 `RunHandle.cancel()` 进入已有 interrupt、session close、SAFE verification 与 safety journal 路径，GUI 不得到 public prepare/fire/safe。W2b 已交付 current Qt PulseWorkbench：Edit/Preview/Scan、current-only New/Open/Save、精确 QPainter timeline、static/scan/HOLD 执行与显式 API 输入由同一窗口组合；preview/load/save/start/reap 在有界 worker 上运行，Qt owner 只按 `(editor_generation, revision)` 接受最新结果，并分别显示 editor revision 与正在运行的 immutable revision。online 窗口只接收既有 `PulseFacade + PulseTargetDescriptor`，Stop/Close 只调用同一 `RunHandle.cancel()` 并异步 reap；offline 窗口不伪造安装/设备身份且禁用执行。W3a 已接通 `.figure_document(capture/fit)` 与 `.figure(capture/fit)`：composition 只把 admitted capture 物化为 `OwnedSnapshot`，frontend-owned DataFigure 一次求值后释放 source snapshot；1D curve、2D image、site/event batch gallery、ComponentValidity、per-cell fit failure 与 overlay 都从具名 evaluated address 求值，模型曲线/面只调用 `FitResultBatch.evaluate_batch()`。W4a在同一产品上增加 `.figure_gui(...)`：notebook只把绑定后的`.figure`窄callable、source与显式view/budget参数交给Workbench；viewer module的唯一capacity-one lane顺序完成repository准入、DataFigure求值与一次Agg/PNG compose，Qt owner在decode前同时准入PNG source与当前物理board front，只接收一张immutable PNG做整板present；close立即撤销发布且不等待worker，也不关闭Experiment。该surface明确为`FROZEN / DISPLAY ONLY`，不取得raw runtime、repository resolver、Figure/artist或第二份fit/projection语义。普通 capture 使用 role 驱动默认视图；generic fit-aware view仍要求repeat具名选择且所有fit batch axis必须是facet/batch/selected/slider；W7对未携自定义view参数的exact saved-fit ref改走专用GridPlot explorer，把repeat明确显示为FACET、每页最多36个逻辑panel，并保留EXPLICIT sparse hole。W8a已把现有公开`exp.fit(capture_ref, committed_transform=...) -> figure/save/exact-ref/W7`认作第一个真实transformed-fit显示consumer，但只接受**恰好一个**作用于仍为fit axis的具名`SPATIAL_X/SPATIAL_Y`数据轴、且由`IndexRangeSelection`或`CoordinateRangeSelection`表达的权威Selection；frontend先按transform解析effective metadata来建议视图，再把同一Selection逐项提升到raw `ViewSpec`，FigureEvaluator仍直接读原`OwnedSnapshot`的精确投影，不造derived dataset/ref、不执行第二遍transform。W8b把完全相同的窄合同接进`exp.fit_gui(..., committed_transform=...)`：transform由调用者在开窗时显式冻结，初始与Clear后的source预览只使用其中同一个Selection，constraint form只替换`FitSpec.constraints`，draft/direct fit/save exact ref继续携带逐值相同的CommittedTransform；窗口从不读取viewport、monitor selector或display reducer来铸造权威。任何`ReductionSpec`、多operation、drop-index、非空间轴、repeat/point/cell-axis变换或display reduction继续在Figure/Agg之前fail-closed。ScanArtifact仍等下一Scan application slice由scan owner提供合法output snapshot后接入，notebook不reshape/伪造lineage。短sitemap的installation profile自动预填同一个trap/camera grid registration并冻结camera、sequencer、trigger、target与坐标身份；真实qCMOS profile/qualification、live GUI site-map/gridplot、旧嵌入式 PulseGUI 最后consumer删除仍未交付，不能把virtual闭环改写成这些产品面或整个 S0.5/H1 已完成。导入 `Zou_lab_control.notebook` 不加载Qt、Matplotlib、SciPy、serial/RPyC、旧 `Zou_lab_control.neutral_atom` umbrella或concrete bootstrap；`connect()`才惰性导入virtual composition，compile capture request才惰性导入triggered binding，calibration/occupancy repository按第一次使用惰性构造，SciPy只在真正执行calibration analysis时进入。request构造只读取FINAL manifest、frame schema与小型runtime summary，不解码pulse IR、calibration arrays或raw frame chunks；完整input admission只发生在formal Run preflight。`CalibrationAnalysisRequest`由轻量`readout.calibration`唯一拥有并由notebook重导出，旧`readout.analysis`路径不保留alias。`pyproject.toml`是依赖的唯一真相，base只含NumPy，analysis/render/notebook/workbench/hardware/docs按能力拆成extras；安装脚本直接安装这些extras，删除平行`requirements.txt`。

**当前Fit/figure事实覆盖上一段的历史W4/W5/W7/W8枚举：** ScanArtifact已经是FitResultRepository、headless `.fit()`、DataFigure、`figure_gui/fit_gui`与exact saved-grid的正式source；generic单panelCURVE/IMAGE及fit-bearing replay均为typed interactive，不再是`FROZEN / DISPLAY ONLY`；独立W5 `_fit.py`与Capture-only ref/repository均已物理删除。U0.3e又让TaskConsole的`Add Analysis -> Fit`只在当前card拥有精确FINAL ScanArtifact时委托同一host；它不是尚未实现的formal AnalysisStep，也不保存自动重跑preset。不能把上一段的历史措辞用于新实现。

W4b在这个边界上交付`.readout.load_calibration_computation(ref)`与`.readout.calibration_report_gui(ref)`：repository一次解码并成对返回已经互相校验的`CalibrationArtifact + CalibrationReport`，worker只投影和绘制已经保存的site、threshold、fidelity、validity与empirical PSF事实，Qt只接收多页owned immutable PNG。报告不重新拟合、不重新阈值化、不把canonical site向量reshape成二维数据，也不为了复用DataFigure而伪造`DatasetSchema`。它与W4a共用一个capacity-one frozen-raster executor、同一Qt window lifecycle和同一encoded-page DTO；关闭窗口只撤销未开始/阶段间工作，已进入repository或Agg的调用诚实排空且不发布stale结果。该窗口只是static frozen calibration report；大阵列的交互focus/zoom/export、live site-map/gridplot仍属于后续明确consumer，不能用本纵切的缩放PNG冒充已交付。W6已在同一paired loader/renderer上补交显式request edit/recalibrate，仍不把静态PNG冒充live grid或几何selector。

W4c把current `OccupancyArtifactRef`接入同一个Figure产品面，但一个Figure只选择artifact中一个真实输出块：`occupancy_output=None/"occupied"`默认选择分类结果，`"counts"`显式选择读出计数；非occupancy source携带该参数立即拒绝。`figure_document()`只读取FINAL metadata中的exact DatasetSchema，`figure()`才在同一总预算内完整admit并直接取artifact已有的`occupied_snapshot`或`counts_snapshot`，不创建第三个DataBlock、不堆成伪COMPONENT轴，也不借source capture冒充occupancy lineage。无scan/spectral/history轴的occupied默认是SITE facet的METER，repeat使用可见的display-only mean；有声明x轴时为CURVE；counts沿既有role规则选择HISTOGRAM/CURVE。SITE永不自动reduce，ComponentValidity继续逐component消费；bool histogram固定使用`false/true`两个类别，避免NumPy auto bins把0/1静默合并。GUI只把output/view/budget转发给W4a现有worker，不新增窗口、executor或renderer。该能力是canonical site index上的冻结显示，不包含Calibration SiteMap的物理XY/GridOrder overlay，也不宣称live gridplot/selector/focus/zoom/export已完成。

W4d补上一个更窄但物理完整的冻结产品面：`.readout.occupancy_cell_gui(ref, selection=...)`只显示一个显式`(repeat, logical point...)` cell的同shot原相机帧与占据圆环。Selection只接受具名repeat/point轴上的exact `IndexSelection`；只有size=1的轴可自动取0，任一非单例轴未给定、range选择、未知轴或sparse `PointLayout` hole都在读取数组前拒绝。composition把同一个`DatasetCellAddress`同时用于`occupied[r,p,SITE]`/ComponentValidity与`CaptureFrameSource.read(address)`，并精确比较source/occupancy的repeat轴、全部point轴、PointLayout、revision、generation及artifact/calibration lineage；不调用capture materialize、不取latest、不flatten、不reduce。Calibration只加载本图实际消费的SiteMap artifact以取得已冻结centers/validity，不加载report、不重跑算法；计数仍由W4c的counts Figure显示，本DTO不为未来tooltip预携无消费者字段。

该物理图不是第三个权威DataBlock，也不把三个artifact拼成新的领域结果：composition在一个总内存cap内顺序admit，frontend只接收自包含的frame/pixel-validity/centers/occupied/site-validity view并绘制empty/occupied/invalid三态。三个仍存活的owner inspection retained bound与view reserve先从cap一次扣除，occupancy materialize、calibration decode、capture admit+single-cell read都只使用同一余额。site最近邻半径使用固定128×128 scratch而非`S²`矩阵，圆环由最多三个`EllipseCollection`承载而非逐site artist；1 MiB scratch与512 B/site均进入render budget。该窗口复用唯一frozen-raster lane与immutable PNG/Qt present，只是`DISPLAY ONLY`静态同shot检查；continuous rolling、live overlay、鼠标selector/ROI与zoom/export仍是后续产品，离散cell导航由紧随其后的W4e独立补齐，不能把W4d称为interactive gridplot完成。

W4e只补W4d已有exact-address产品的离散cell导航，不把它扩成live grid或图形selector。窗口先在同一个shared frozen-raster lane异步读取metadata-only `OccupancyCellNavigation`：content-addressed artifact target、occupied schema fingerprint、generation、完整repeat/point `AxisSpec`、`PointLayout`与schema-owned `cell_layout`。这些长期metadata和按轴Qt控件的retained upper bound先从唯一总预算扣除，剩余预算才逐字传给W4d exact-cell load/render/presentation；换cell前先清旧PNG/QImage front。每根非singleton轴保持显式“未选择”，绝不默认first/latest；只有singleton可自动取0。控件只按轴数建立spin/index editor并即时显示coordinate/unit，不按axis size建立item列表。Load把全部repeat/point轴冻结为具名`IndexSelection`；Previous/Next沿`cell_layout`的repeat-major physical storage order移动，因此EXPLICIT sparse hole天然跳过且不wrap。

同一窗口在全局唯一capacity-one lane上至多有一个已提交job和一个可覆盖的latest pending selection；每个结果同时校验UI request revision、artifact/schema/generation identity与canonical Selection，stale success/error/cancellation一律不能改status、summary或front。旧Future/PNG在当前owner callback真正退栈后，下一次queued owner turn才启动pending，避免两次大图生命周期重叠。Close只清pending/front、置cooperative cancellation并立即返回；已进入repository/Agg的工作诚实排空但永不发布。W4e当时只完成离散 exact-address 导航，尚无`ViewportTransform`、鼠标selector/ROI、zoom、export或live source；“永久 `DISPLAY ONLY`”这一产品能力上限现已废止。artifact/source仍不可原地修改，但该查看面必须在U0纠正3或通用viewer切片复用统一interaction owner补齐旧行为。

**术语裁决（覆盖本节以上 W4/W7 现态摘要及后文历史 checkpoint）：** `DISPLAY_ONLY` 以后只允许表示 ViewSpec/viewport/display reducer/export raster 不能静默升级成 FitSpec、ScanOutputContract、CommittedTransform 或修改 source artifact；绝不表示产品 UI 可以永久缺少 zoom/pan/selector/re-fit/export。W4a/W4d/W4e 仍保留已证明的 immutable source、预算、单 render owner、cancel/close 与 revision-check 不变量，但其产品能力上限已撤销。W4b calibration 多页报告是 frozen raster 的合法例外，仍必须补 zoom；其 site/grid focus/export 是否适用按规则 9 salvage。W7 exact-result replay 继续禁止**隐式** solver/rewrite；用户显式 `Analyze -> Fit` 必须从 exact source ref + SelectionCandidate 新建 authority draft/result/ref，绝不原地改旧 artifact。

U0.3d 已把 W5/W8b 的独立 Fit host 物理删除并并入唯一 `DataFigureWindow`。`.fit_gui(CaptureArtifactRef|ScanArtifactRef, ...)`现在只是“打开同一Figure并选中Analysis tab”的便利入口；普通`.figure_gui(source)`也可从typed CURVE/IMAGE直接`Analyze -> Fit`。同一共享form DSL编辑catalog-owned seed/bounds/fixed；`FitSpec`冻结具名fit axes，所有其余repeat/point/data axes原样成为batch。1D range与2D box selector只经显式普通Fit动作提升为authority Selection；viewport、relim、cmap与用于单panel显示的batch-cell selection仍是presentation，不能进入FitSpec。

`FitExecution`与唯一未保存draft由headless `FitDraftAuthority`独占；Qt只得到不可保存的`FitDraftResult`与immutable typed front。draft overlay把结果与窗口冻结的exact Capture/Scan source重新绑定，并在物化前验证source revision/schema/fit axes/batch layout。只有显式Save才通过composition的lifecycle gate发布neutral-owned `FitResultArtifactRef`，随后从该exact ref重开；Clear只撤draft/overlay。Save返回ref以后，decode/render失败或Close都不能吞ref，Save中Close明确defer。长Fit只在开始/结束短持Experiment gate并登记fit-specific active count，不建立workflow/lease/async engine；deadline/cancel在lane等待、solver batch之间和Qt接纳处诚实检查，已进入bounded source decode只排空而不伪称已中止。

M1在同一门面上交付第一个真正continuous的raw monitor产品。`camera_monitor_request()`只冻结显式camera role、总内存上限与adapter I/O timeout；`inspect_camera_monitor()`只返回free-running capability、working point与精确输出schema，均不arm设备。`camera_monitor_gui()`只把one-shot `PreparedCameraMonitor` factory交给Workbench，不把Experiment、RunPlan、Port、DeviceBindingResolver或raw camera交给窗口。窗口每次Start都先完成全拓扑与总内存准入，再由RunController取得排他资源；Stop/Close只cancel同一个RunHandle并异步reap，清理完成且设备证明SAFE后source terminal才撤下front。重新Start必须取得新的BlockId、stream generation和Run identity，不能延续上次“latest”。该入口目前只承诺raw IMAGE：ROI、scalar meter、rolling curve、histogram、multipanel coherence和selector明确留给M2，不能把capacity-one raw view称为完整rolling/gridplot迁移。

上段“只有真正执行calibration analysis时进入SciPy”在W4b之后还包括用户显式调用paired diagnostic load/report GUI，因为`CalibrationReport`的owner就是analysis extra；普通`import Zou_lab_control.notebook/workbench/zlc_frontend`和runtime-only calibration load仍不导入SciPy、Matplotlib或Qt。

`Experiment` 只做参数便利、typed request 构造、结果解包和composition delegation；它不调用raw adapter，不复制 calibration/fit/scan 算法，也不让 domain object lazy 回查全局 session。notebook baseline 不保存 `current_calibration` 指针、revision 或“最近一次”映射：这些状态在当前实现没有业务消费者，还会让短 API 的物理输入变成隐式。所有依赖 calibration 的 convenience request 必须显式接收 `calibration=CalibrationArtifactRef`；构造边界从FINAL runtime summary冻结具体 `ReadoutBindingKey`、event AxisId 与最终model kind，但不把轻量inspection冒充完整authority。formal Run preflight随后一次性admit calibration及其source、重新验证全部结构/物理适用性，并把同一process-local admission保留到analysis与final commit；多 camera 不能猜 ref，运行时也不回查 facade。结果、internal RunPlan 与 artifact lineage 始终记录实际使用的 binding/ref/digest，因此短 notebook 路径不以隐藏物理输入换便利。

`Experiment`、TaskConsole、PulseGUI 与 standalone launcher 都不得公开 raw `CameraDevice`、`SequencerDevice`、旧 `DeviceSet`、SDK handle、BoundDevice/drive-capable Port、internal RunPlan，或可直接执行 `configure/arm/acquire/prepare/fire/abort/safe` 的 adapter。普通实验硬件动作必须转换为 declarative typed request/窄command facade，经同一个 process-lifetime `InstallationRuntime`、RunController、ResourceArbiter、DeviceBroker 与 adapter owner 执行；跨进程物理排他由具体backend/composition提供可验证proof，不在generic runtime中再建一套平行lease框架。普通连接只在开放 admission 前由 composition 启动序列建立；durable blocker 的唯一例外是 §9 的 claim-first `RecoveryAttempt`。workbench 只调用窄 command/recovery facade，不取得第三种 raw adapter capability。`.pulse_gui()` 得到的是 workbench-owned pulse command facade，而不是 `session.sequencer`；`.readout` 得到的是领域 convenience facade，而不是 `session.camera`；`.device_catalog` 返回 immutable 值，而不是 `session.devices` 的兼容代理。若某 standalone 入口无法加入同一 installation runtime，real mode 启动即失败，只可显式运行 virtual/offline 模式，不能以“独立窗口”为由绕过 quarantine、runtime instance 或 active claim。

不存在 public `NeutralAtomSession.devices/.camera/.sequencer`、`Experiment.devices` raw alias、`__getattr__` warning fallback 或“只读时返回 raw、写入时再检查”的 wrapper。`Experiment.device_catalog` 是新语义，不为旧调用保持 duck typing。领域对象、Definition、internal RunPlan 和 frontend 也不得保存 `DeviceBindingResolver`；resolver 只在 composition/bind 边界把已声明的 requirement 解析成领域私有immutable bindings，随后立即退出调用栈。RunPlan与preflight的私有prepared value只存在于composition/RunController执行图，且owner结束时主动断开；`inspect()`只返回claims/schema/budget/summary组成的immutable PlanDescriptor。baseline不提供public延迟Plan对象；若未来真实用例需要，只能由authority签发opaque、generation-bound、one-shot PlanHandle，handle本身不含或暴露Port。

`figure_document` 由 notebook composition 的 result projector 生成，避免 neutral_atom 反向依赖 frontend.figure。FigureDocument 故意不含 repository/ref/resolver，单独只能编码、检查或编辑 presentation；实际 evaluate/render 必须另给同 revision 的 `ResolvedDatasetMap`。`.figure()` 在 composition guard 内准入并物化 source，退出 guard 后才把 document、snapshot map 与可选 FitResultBatch 交给 DataFigure。render extra 不可用时，采集、分析与 `figure_document` 仍完整工作。GUI launcher 不是 headless notebook baseline；只有安装 `notebook[workbench]` 时各GUI入口才通过单向optional edge调用workbench composition，workbench不反向import notebook facade。Figure Fit窗口只接收`.figure` loader、prepare/execute/save/reload四个窄command、exact source identity、初始model/authority Selection与总预算；saved-fit Grid只接收worker-affine exact-ref view loader、显式focused-cell Refit opener、typed ref与总预算。calibration、occupancy、monitor窗口仍各接自己的窄loader/starter。任何窗口都不接收/保存Experiment、repository、Port、RunPlan或raw adapter；关闭后同一Experiment继续可用。

日常路径必须保持短而诚实，例如：

```python
exp = zlc.connect("virtual", repository=repo)
capture_ref = exp.readout.capture("my_pulse.json")
fit = exp.fit(capture_ref, model="radial_gaussian_center")
fit_ref = fit.save()  # neutral-owned FitResultArtifactRef；有界默认且不要求 renderer
saved = exp.load_fit(fit_ref)
```

当前显式校准底座同样不暴露Port/RunPlan：用户可先构造带具名event layout和独立空间意图的`CalibrationAnalysisRequest`，再经`exp.readout.calibration_request(capture_ref, analysis)`冻结实际`ReadoutBindingKey`并调用`calibrate(request)`；它不会按shape/rank猜事件轴，也不会用本次detector输出自证expected centers。已交付的virtual/offline短`exp.readout.sitemap(...)`只在application facade顺序执行“capture提交raw ref -> calibration提交calibration ref”，从installation-owned grid/profile预填空间意图，并让FPGA pulse repeat包围完整long-short-long组；第二步普通失败通过`SitemapCalibrationFailed.source_capture_ref`、notebook中断通过仍属于`KeyboardInterrupt`的`SitemapCalibrationInterrupted.source_capture_ref`返回第一步raw ref。它不嵌套child plan、不回滚第一个artifact，也不引入隐式`current/latest calibration`。

同一组代码换成 real adapter 只改 `connect` 参数。契约 E2E 固定“connect virtual -> capture -> 1D fit -> save”和“sitemap -> calibration ref -> detect”均为少量 notebook 语句；不得要求用户手工 bind Port、构造 PipelineSpec、解析 PipelineResult 或 resolve artifact ref。门面在最早 vertical slice 与 RunController/Repository 一起交付，不能拖到剩余 helper 收尾阶段。

### 4.6 顶层运行模型：四个平面、三个边界

最终架构不是一个所有节点都传同一种“大数据对象”的通用 DAG，而是四个语义平面：

```text
外部世界
  -> Measurement
  -> [sample/event plane: Envelope<Value | typed domain record>]
  -> StreamProcessor（可选、逐 event 或明确 key group）
  -> DatasetBuilder（finite exact）| MonitorDataset（live preview）
  -> [dataset plane: immutable DataBlock revision / typed ArtifactRef]
  -> Analysis（zlc_data 通用分析或 neutral 领域分析）
  -> [result plane: typed result / immutable artifact]

任一冻结 dataset/result
  -> frontend ViewSpec/FigureDocument/DataFigure
  -> [presentation plane: 可丢弃、可重算、不可反向成为权威输入]
```

三个边界各自只有一个 owner：

1. `Measurement -> event` 由 acquisition runtime 赋 envelope、key、generation 和 provenance；设备 adapter 不发布 Hub event。
2. `event -> dataset` 只由一个预先绑定的 materializer 完成：finite exact 由 `DatasetBuilder` 按冻结 repeat/point schedule 写入，live 由 `MonitorDataset` 按物理 cycle 或 event sequence 写入；二者都保持 `(R,P,*data_shape)`，StreamProcessor 永远不会把累计 DataBlock 当“最新 signal”。
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

`ValueSchema` 描述一次 Measurement/StreamProcessor event 携带的值，例如一帧 `(H,W)` image、一个 `(site,)` occupancy vector 或一个真正标量。它没有伪造的 R/P leading axes。`DatasetSchema` 描述 materializer 把事件放入哪些 repeat/point cell 后形成的完整数据集。DataBlock 永远符合 DatasetSchema，AcquisitionStream/MonitorTap 的普通 event 永远不携带累计 DataBlock。

一个 domain event 可以是 frozen typed record，例如 `CameraSample(image: Value, frame_metadata)` 或 `OccupancySample(occupied: Value, counts: Value, source_metadata)`；它仍作为一个 Envelope payload 原子发布。record 中每个数值字段使用 zlc_data 的 Value/ValueSchema，record 类型和领域 metadata 由 producer package 拥有。

`AxisSpec` 包含：

- stable AxisId；
- name、role；
- size、coordinates；隐式坐标另带 `index_origin`，连续裁剪只移动 origin，不物化整根坐标轴；
- unit: canonical unit id 或 `None`、coordinate_frame: CoordinateFrameId。

AxisId 由 producer Definition 的稳定字段语义派生，在相同 semantic axis 的不同 run/adapter 间保持一致；不能每次构造随机 UUID，也不能用可修改 display name 或 tuple position 充当 identity。baseline 的 Selection 保留被保留轴的 AxisId，Reduction 只移除被约简轴，不创造匿名 replacement axis。没有真实消费者的 Transpose/Stack/Create/Rename 与逐 operation 历史对象不预建；出现第二个必须创建派生轴的生产用例时，再由该 operation owner 定义稳定 AxisId 与 lineage。

单位采用 canonical string（例如 `Hz`、`MHz`、`s`、`count`），display label 与物理单位分开。baseline 不建立量纲代数、单位表达式 AST、UnitConversionTable 或自动换算；只有完全相同的 canonical string 才兼容。未知单位作为 opaque string round-trip。CoordinateFrameId 同样是 stable opaque id，只做等值检查；不同 frame 在 baseline 直接拒绝，不能因 shape、名字或数值范围相似而默认兼容。

需要相机畸变标定、单位换算或其它带物理模型的映射时，先由带 CalibrationArtifactRef 的 neutral StreamProcessor/Analysis 显式产生新值与新 schema；不能把领域 calibration 藏进通用坐标 metadata。只有真实用例证明多个领域需要同一纯转换合同后，才从这些用例中提取 serializable UnitConversion/CoordinateTransform，baseline 不预建。

`role: AxisRoleId` 是 producer 声明的 stable、可序列化语义，例如 repeat、scan-point、monitor-history、spatial-x、spatial-y、spectral、site 或 component。built-in role 由 zlc_data 单源定义；领域扩展使用 namespaced id，不注册可变全局对象。不认识的 role 仍能 round-trip，但默认 preserve/select。role 不能从 rank、长度、singleton 或数值内容反推。`MONITOR_HISTORY` 只表示 live snapshot 内 newest-first 的可见 slot，不是物理 scan point、readout setting 或可直接进入权威 Fit 的自变量。

ValidityContract 是 ValueSchema 的一部分并进入 fingerprint：VALUE 表示整个 event value 同生同灭；COMPONENTS(axis_ids) 声明 mask 可细化到哪些具名 data axes。producer 不能首帧发 VALUE validity、遇到坏 site 后再未经 generation migration 改成未声明的 component mask。processor owner binder 在构造 BoundStreamProcessor/preflight 时根据输入 validity contract 冻结输出 contract 与传播规则，无法证明时不能进入 formal exact pipeline。

Data schema 不枚举“当前软件允许哪些 projection/reducer”。数据身份与已安装分析功能必须解耦：

- Selection 与 Reduction 由 zlc_data 的显式 DataTransformSpec 定义；其它结构变换在出现真实消费者前不进入 baseline；
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

`DatasetSchema.cell_layout` 是 repeat + PointLayout 合成后的唯一 canonical physical-row mapping owner。它在 schema 构造时只合成一次并缓存；transform、fit 与 frontend evaluator 都直接复用该对象，不能各自重写 C/F/EXPLICIT 组合规则。对于稀疏 PointLayout，PRODUCT factor 必须保留原 immutable PointLayout 实例及其已经建立的反向索引，不能为了消除 subclass equality 差异复制一份 mapping/dict；AxisLayout 的值相等与 hash 按结构而非 specialization class 判断，使 owner tree codec 把 factor 还原为通用 AxisLayout 后仍保持同一 canonical identity。live render 不得每帧重建 layout 或反向索引。

### 6.3 DataBlock 与 validity

```text
Value:
  values: ndarray         # (*data_shape)，标量为 shape ()
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

Value.validity 与 DataBlock.validity 必须符合 cell_schema.validity_contract；COMPONENTS 合同仍允许用整体 Valid/Invalid 或 CellValidity 表示“本 revision 所有 component 同生同灭”的紧凑特例，但一旦提供 ComponentValidity，其 axis_ids 只能是合同声明集合的子集。VALUE 合同绝不接受 component mask。Selection/Reduction 必须同时派生新的 validity_contract，不能只变 values/schema axes 而忘记 mask 语义。

ReductionSpec 必须声明 `validity_policy`（例如 `ALL_REQUIRED`、`ANY_VALID`、`MIN_COUNT(n)` 或所选 reducer 合同自己的规则）。reducer 只在 mask 为真的 component 上运算，并产生新的具名 validity；不能把 NaN 当通用 validity，也不能用 `nanmean` 在未声明策略时悄悄吞掉坏 site。FitProblem 逐 batch cell 过滤无效 observation，并记录有效样本数；不足模型最小点数时只使该 batch result 失败。Histogram 丢弃无效 sample 但记录 dropped count；Meter 在目标 component 无效时显示 invalid，不回退其它 component。

发布后的 DataBlock 是 immutable materialized dataset snapshot：不仅 consumer 不能写，**snapshot 的 bytes 在其整个可见生命周期内也不得因 materializer 后续 ingest 而变化**。finite exact 的 `DatasetBuilder` 持有不外泄的 mutable preallocated/chunked storage，根据冻结 schedule 原子写 values+validity，并只返回轻量 `DatasetProgress(block_id, revision, dirty_cells, coverage)`；旧 ref 请求必须返回 `SnapshotExpired`，绝不能回 latest。live 的 `MonitorDataset` 也只保留 current mutable window，但 ingest 通知只用于 coalesce；controller 必须从同一把锁冻结的 `MonitorDatasetSnapshot(OwnedSnapshot, aligned EventRefs, head, coverage)`读取当前值与 current selection，禁止把旧 progress 的 dirty/head 与后来 snapshot 拼接。任何 owned snapshot 都按刷新预算显式取得，不能每个 event 自动 fan-out 完整 DataBlock。

`DatasetBuilder.materialize(current_ref)` 只产生 **provisional** `DatasetPreviewSnapshot`；只有绑定的 exact reservation 全部 ack、冻结的 `sequence -> DatasetCellAddress` 计划逐项匹配、TraceBinding 一致、source-owner EndOfStream 与 reserved end 相同、coverage 完整时，它才能 mint `SealedDatasetArtifact`。`MonitorDataset` 在类型上没有 `seal()`，只产生 `MonitorDatasetSnapshot`；交互冻结必须另建带 coverage/EventRef 的 finite diagnostic input，不能把 live window 冒充 formal capture。软件 seal 不自动证明物理 trigger↔frame 真实性；PulseScan 仍须由 §14.5 `EpochValidationRecord` 包装为 VALID。发布快照必须是 owned copy、immutable sealed chunk 或 copy-on-write；禁止把 mutable ndarray view 仅设 `writeable=False` 后冒充 revision。

DatasetRevisionRef故意不携带Formal runtime provenance，避免zlc_data反向依赖neutral。Formal scan由§14.5的neutral-owned `EpochBoundDatasetRef`把普通DatasetRevisionRef与epoch integrity绑定；Workbench/Repository adapter必须保留这个外层wrapper，不能只抽出裸DatasetRevisionRef后绕过authority gate。

Measurement/StreamProcessor 不创建 DataBlock，也不读“当前累计 block”来决定下一条输出。它们只处理当前 Envelope payload 或声明的完整 key group。`DatasetBuilder` 与 `MonitorDataset` 是互斥的两种 sample -> dataset 边界；interactive/display Analysis 可显式消费 provisional snapshot，权威 Fit/Calibration/Repository 只消费 SealedDatasetArtifact 或更强的 VALID EpochBoundDatasetRef。DatasetProgress/RevisionRef 是状态通知，不是数据输入，consumer 不能从它重建权威值。

两种 materializer 共用一个与 generation-owned PayloadContract、同一 ValueSchema owner 对齐的 frozen `DatasetEventAdapter[T]`，但不复制/反射重建整张 adapter graph。adapter 从一次 frozen payload 投影 `Value`，metadata 由其 `DatasetMetadataContract(snapshot/retained_nbytes/max/fingerprint/digest)` owner 冻结；runtime 只拒绝真实可变 metadata alias，不把同进程 `object.__setattr__` 当安全边界。exact 路径在 Delivery/ack 事务中保存 ordered metadata digest；live 路径只保存显示所需 metadata 与 EventRef，不在热路径计算无人消费的第二份 digest。`CameraSample(image, metadata)` 因此不需要 side-channel metadata stream；adapter 不能改变 key、sequence、TraceBinding 或 exact cell schedule。

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

### 6.5 Materializer 的原子提交

materialized value 的全部相关状态由实际拥有 mutable storage 的 materializer 直接负责，不建立无人消费的通用 delta 值对象。`DatasetBuilder` 先完成 payload、validity、exact schedule/key、metadata 与 authority 验证，再在 stream 的 Delivery/ack 事务和 builder 自身锁内一次提交 values、written、validity、metadata、ordered event/metadata hash state、counters 与 revision；所有会按输入拒绝的操作必须在 commit point 前完成，commit point 后只消费已经准入的 typed owner 值并以同一 stream 临界区内的 no-fail ack 收尾。前置验证失败不写入、不推进 revision/ack；进程或基础设施故障只能使本次 run 失败，不能产生 sealed artifact。`MonitorDataset` 在自己的锁内一次提交 values、written、validity、metadata、EventRef、head/counters 与 revision，并从同一临界区冻结 head、coverage、EventRefs 与 owned snapshot，不能把不同 revision 的字段拼在一起。

sample stream、StreamProcessor edge 与普通 UI queue 只传事件值或 typed record。持久化只接收 sealed artifact/immutable snapshot，不保存一套与 materializer revision 平行的 delta journal。Live binding 只接 coalesced revision 通知，并按刷新预算请求 current slice/owned snapshot；不得把 full mutation record 或 full snapshot 逐 event fan-out 给所有 panel。只有真实 profile 与已排期 consumer 同时证明 immutable snapshot 成本不可接受时，才从那个 storage owner 内部提取最小增量表示；在此之前不预建 history、revision replay 或跨 owner apply 协议。

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

数值/领域数据 Envelope 额外包含 payload contract fingerprint 与 captured timestamp。payload 的 snapshot/validate/retained-bytes/max-bytes 必须由一个 generation-owned `PayloadContract` 单源提供，不能让三个 lambda 分别估计并漂移；`ValuePayloadContract` 还要求所有 event 共享同一个 ValueSchema 对象，并把 ComponentValidity mask 的 owned bytes 纳入预算，禁止每帧夹带未计费的重复 schema/coordinates。普通 stream payload 是 Value 或包含 Value 字段的 frozen domain record；DataBlock 只属于 DatasetBuilder/materialization 边界，不能作为“当前累计 signal”反复发布。Provenance 是 causation graph、payload fingerprint、CommittedTransform 与外层 artifact lineage 的派生视图，不是另一套含义模糊字段。

JoinKey 是 frozen、可序列化的领域值（例如 TriggerKey/ScanCellKey/ShotKey），不是字符串拼接或 payload 私有字段。generation-owned `JoinKeyContract.snapshot(key)` 是唯一 admission owner：它同时验证并返回 owned frozen key，stream 不在下一行重复 validate；fingerprint 绑定其语义。exact DatasetBuilder 另绑定由编译计划独立产生的完整 `sequence -> DatasetCellAddress` schedule，event key 必须逐项相等；仅有合法 key 类型并不足以证明 row 没有对调。keyed live cycle 同样验证物理 schedule；append history 则故意不把 producer join key 当 panel slot，slot 只由 consumer sequence 决定。TraceContext.correlation_id 只用于追踪，不能代替数据关联 key。

`stream_generation` 只能由 broker/factory mint 的不可复用 incarnation identity 产生，调用方不能用可复用字符串为两个 live source 指定同一 generation；否则不同内容可能得到相等 DatasetRevisionRef。`sequence` 在 `(stream_id, generation)` 内从 0 严格单调且不复用，event_id 由 generation+sequence 派生，不维护随 monitor 寿命无界增长的去重集合。StreamProcessor 输出创建新 event_id，不能沿用某个输入 id 冒充同一事件。少量 join 使用 EventRef；StreamReducer/DatasetBuilder 的长连续输入使用 EventSpanRef，ordered_digest 覆盖按 sequence 排列的 event_id/payload digest。禁止在每个累计结果里复制全部历史 event_id，避免 provenance 退化为 O(N²)。

### 7.2 四种通信原语

| 原语 | 语义 | 用途 |
|---|---|---|
| AcquisitionStream | ordered、exact、cursor、gap-fatal | 正式 scan/capture |
| MonitorTap + MonitorDataset | bounded backlog、latest/ordered、missed + sequence-owned window | live UI |
| ControlTopic[T] | typed、revisioned、ack | ROI、threshold、run command |
| EventStream[T] | progress/transition notification | UI/headless status |

Artifact Repository 是持久化原语，不是 stream。

ControlTopic 的 ack 明确区分 `ACCEPTED`、`APPLIED(at transaction boundary)`、`REJECTED(reason)`、`SUPERSEDED(by_revision)` 与 `TERMINATED(reason)`；发送成功不等于硬件已经应用。每个被 ACCEPTED 的 revision 最终必须恰好收到 APPLIED、SUPERSEDED、REJECTED 或 TERMINATED 之一，UI 不会永久等待一个被 coalesce 或 owner shutdown 吞掉的 command。有限正式 Run 拒绝 reconfigure 时必须返回 REJECTED，UI 不能先改成本地“已生效”状态。

monitor 的 ROI/threshold value 更新必须把 **source acquisition 与 downstream analysis 生命周期分开**。相机 source Run、raw stream/tap 和可见 raw front 持续运行；修改一个已经存在的 ROI/threshold processor 只向该 downstream owner 的 `ControlTopic` 发布新 revision，并在该 processor 的事务边界得到 `APPLIED` 后切换语义，不能重启 source、重建 raw history 或制造 running consumer gap。新建/删除 ROI processor 只创建/终止该 downstream stream/generation；source tap topology 仍不变。schema-affecting 的 downstream 变更可以迁移该 downstream generation，但不得借此迁移 source generation。只有 source 本身的硬件/采集 schema 确实改变时，才按下文 source generation migration 执行；不能把 M2d/M2e 的 whole-Run replacement 当成 monitor retarget 的实现捷径。

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
- `AcquisitionProducer` 是 source owner lane 独占的 write/terminal capability；普通 pipeline 只拿 read-side stream/reservation，不能提前 mint EOS；
- cursor、Delivery、EndOfStream 都是 owner-minted opaque capability，不能用公开构造器或“is_exact=True”伪造；
- cursor 不跨 generation；
- baseline 每个 stream generation 只允许一个 formal exact materializer，monitor fan-out 不 pin exact retention；出现第二个真实 required-exact consumer 前不引入多 reservation watermark 机器；
- reservation 绑定 `TraceBinding(run_id, source_id)`；同一区间混入另一 run/source 的 event 在写 DatasetBuilder 前失败；
- DatasetBuilder 构造时独占 claim reservation completion/abort authority；`commit cell + ack` 在同一个 stream authority 临界区完成，失败不能留下可 seal 的半 revision；raw reservation 不再提供与 builder owner 冲突的 context-manager cleanup，统一由 DatasetBuilder/session teardown；
- frozen join schedule、Envelope key、sequence 与 destination cell 四者逐项一致后才写入；
- exact path 不自动 retry hardware run；
- 无 reservation 的慢消费者获得 typed Gap，不回 latest。

正式 exact run 必须是有限 Run，并有确定的最大 end_sequence；不能用“预计平均帧大小”冒充 byte 上界。但完整性总量与同时 retention 预算是两个不同合同：

```text
total_expected/max_events        # EOS/coverage 完整性
max_inflight_events/bytes        # 未 ack retention admission
max_source_burst + backpressure capability
```

RunController 根据 source 最大 burst/速率、最慢 required consumer、ack 边界和是否可 backpressure，证明一个保守最大 backlog；reservation 只 pin 未 ack 区间，ack 后立即释放。不能因为 run 有 N 点就无条件在 RAM 保留 N 个大帧，也不能在不可 backpressure 的相机上假设 consumer 平均够快。若无法证明 max inflight，必须为最坏 total 分配、选择流式 artifact sink/更慢触发，或 preflight 拒绝。连续 Measurement 在 baseline 中只使用 admitted `MonitorTap -> MonitorDataset`，不建立 infinite reservation、continuous-exact epoch 或 durable spool。未来只有出现必须连续、不可丢且无法切成普通有限 Run 的第二个生产用例，才单独设计持久 spool/epoch 协议；不能先让所有运行背负该状态机。

source contract 必须声明 `BACKPRESSURE_CAPABLE` 或 `NON_BACKPRESSURE_CAPTURED`。前者的 publish 可在零状态变化后返回 `StreamBackpressure` 并由真实 producer 稍后重试；后者表示调用 publish 时物理帧已经不可撤回，第一次 event/byte retention miss 必须在同一临界区把 generation、reservation 与所有 formal consumer 永久置为 `RetentionOverrun/FAILED`，后续 frame 绝不能占用失败帧原本的 sequence，emit/finish/seal 全部拒绝。qCMOS 属于后者。retention 之外任何发生在物理 capture **之后**的 decode/copy/schema/metadata/key/publish 异常也必须由 CaptureSession 统一转成 `producer.fail(SourceFailed)` 并继续做硬件安全 drain；禁止 catch 后继续 formal collection。

camera adapter 边界使用一种不可变的 `CameraFrameRecord`，而不是一条 ndarray queue 加另一条 metadata queue：

```text
CameraFrameRecord:                       # adapter_sdk owner，不是 artifact schema
  image: owned, C-contiguous, read-only ndarray
  source_ordinal: non-negative int       # 当前 arm epoch 内的软件交付序号
  produced_count: optional non-negative int
  frame_stamp/camera_stamp: optional int
  timestamp_seconds: optional non-negative int
  timestamp_microseconds: optional int in [0, 1_000_000)
  host_received_at_ns: positive int
  driver_buffer_index: optional non-negative int
```

record 构造时就必须取得图像 bytes 的 ownership 并冻结 metadata；driver 之后复用 ring slot 不得改变已发布 record。`source_ordinal` 在每次 arm epoch 从0连续增加，duplicate/gap/out-of-budget 立即失败；它是 host adapter 的排空顺序，不是 FPGA emitted-edge receipt。`produced_count` 是读取该帧时观察到的 source 累计快照，batch drain 时可在多个 record 中相同；禁止把它伪造成逐帧 +1 counter。qCMOS adapter 必须从同一次 `buf_getframedata` 保留 `framestamp/camerastamp/timestamp`，并把同一 drain 观察点的 `cap_transferinfo().nFrameCount` 写入 `produced_count`。

`adapter_sdk.CameraWorkingPoint`只冻结adapter从硬件/模拟器读取的物理工作点与由adapter唯一计算的settings fingerprint；它不携带、也无权自铸exact qualification。`CameraCaptureEndpoint`只把这些primitive facts一次转换成neutral-owned `ValueSchema/CameraPhysicalFacts`，不认识Virtual/DCAM concrete type；可信installation composition另行注入qualification。当前virtual composition注入与原实现相同的deterministic in-process trigger-wire digest；real composition只能注入由active Q0 authority解析并pin的资格，raw adapter返回任意字符串不能放行FIRE。该结构化SPI只解除concrete-type反向依赖，不证明hardware identity、thread affinity或Q0资格。

SPI中的`max_pending_records`是adapter在一次arm内可能持有的owned record硬上界，不是GUI recent/history长度；endpoint的`max_source_burst_events`不得超过它，adapter的`arm(max_inflight_frames=...)`也必须拒绝越界。当前endpoint terminal边界有一个明确而窄的两阶段合同：bounded terminal worker可先调用`finish_record_capture()`解除并发blocked read、stop/drain/join并冻结terminal record，随后arm-owner再次调用以完成owner-affine teardown，两个调用必须线程安全、幂等且返回同一record。VirtualCamera与contract fake满足此合同；不能满足的DCAM/thread-affine adapter必须等待camera owner lane/host交付，不能因Python Protocol结构匹配就宣称real READY。

arm 时冻结 pending retention capacity：finite capture 以声明的 frame budget 为硬上限，continuous capture 以 adapter/profile 证明的 max-inflight 为上限。adapter pending queue 在两种模式下都不 overwrite 尚未被唯一 capture owner 排空的物理帧；monitor 的 overwrite/missed 只发生在 owner 已将 record 转交给 broker 之后的 bounded monitor tap。容量不足、ordinal 不连续或超过 arm budget 返回 typed `CameraBufferOverrun`，formal CaptureSession 将其上升为 `SourceFailed/RetentionOverrun`；monitor run 也必须明确停止/重建 capture session 并报错，不能在 adapter queue 内悄悄丢帧后只增加 UI missed count。迁移期的 `read_frames()` 只能解包同一 record queue 的 `image`，不得维持第二份排队、ordinal 或 metadata 真相源；array-only reader 只在**最后一个 legacy camera consumer 所在的 dependency-closed 切片**删除，不能把时点写死为 S3。当前 consumer matrix 仍包含 ROI、monitor、temperature 与剩余 legacy measurement/UI，预计随 S5 收口。终态 adapter contract 只交付 record。CaptureSession 在 owner lane 把 record 一次转换为neutral-owned `CameraSample(Value, metadata)`；`CameraFrameRecord` 不穿过 bounded-context 边界进入 zlc_data、processor、UI 或持久 artifact。

stream_generation/payload contract fingerprint 改变时，旧 exact cursor 终止为 typed SchemaChanged。schema-affecting reconfigure 不是“原地改参数”，而是 generation migration：owner 在 transaction boundary 终止旧 generation、对所有 pending Control revision 发 terminal ack，为每个绑定的 exact DatasetBuilder 或 live MonitorDataset 创建新 block_id/DatasetSchema/generation，再在新 generation 首次 publication 前完成 tap/materializer rebind。旧 pending view/fit 结果 stale，CommittedTransform 因 DatasetSchema fingerprint 改变一律失效，不能按 index 偷迁移。稳定 AxisId 只帮助迁移 workspace preference 的候选匹配，仍须完整 schema/coordinate/validity 校验。正式 finite Run 默认拒绝 schema-affecting reconfigure；value-only 且 schema 不变的参数才可按运行合同在边界 APPLIED。

一个 StreamProcessor invocation 只原子发布一个 typed payload；同 shot 多字段装进同一 frozen record，成功 enqueue 后才 ack 输入。baseline 不支持把一次 invocation 拆成多个 exact stream 再实现跨 stream transaction；确有不同 cardinality/key 的结果应拆成独立节点。DatasetBuilder 在 Value 已按 frozen schedule 原子写入 values+validity 后 ack；storage 与权威 processor 只接受 SealedDatasetArtifact 或 VALID EpochBoundDatasetRef，不能退回接受裸 DataBlock/OwnedSnapshot/DatasetPreviewSnapshot。Repository sink 的 ack 点是临时 blob fsync/校验完成且 manifest 原子提交之后，不是刚开始写文件。

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

`EXACT_KEY` 是跨设备/跨 worker 的正式关联方式。`ZIP_SEQUENCE` 只允许用于同一已验证软件 producer 拆出的等长 ordered streams，且合同能证明 sequence 一一对应；不能拿两个独立设备的“第 N 条”推断同一 shot。`INDEPENDENT_LATEST_MONITOR` 只用于互不声称相关的独立 panel，不能用于多输入 expression 或同一 coherent view。

暂不引入 WINDOW/ClockTransform；出现真实 use case 后再设计。

### 7.5 Exact 与 monitor fan-out

一个 physical CaptureSession 是 camera 的唯一 owner。它产生一次 AcquiredSample，broker 分发到：

- exact subscribers：受 reservation/backpressure 保护；
- monitor tap：bounded overwrite，记录 missed，不反压 exact。

二者共享 event_id/trace，但拥有不同 QoS 与 buffer。

broker 只分发 immutable payload/ref；driver 会复用 DMA/frame buffer 时，在第一次发布前复制或转移 ownership。exact retention、monitor current ref 和每个 consumer 各自持有明确 lifetime；monitor overwrite 只能释放自己的引用，不能使 exact consumer 看到被复用的内存。

### 7.6 Buffer sizing

Measurement output contract 声明 `max_payload_bytes`、`max_burst_events`、finite run 的 expected/max total events、生产速率/停顿边界与是否能 backpressure；StreamProcessorDefinition 声明 cardinality、输出尺寸上界、最大并发处理与 ack 点；DatasetBuilder/sink 声明 exact backlog、chunk/flush 与最终 DataBlock bytes；`LiveDatasetSlotSpec` 单独声明可见 window capacity。RunController 从这些合同派生 exact edge budget、MonitorTap 暂时来不及 ingest 的 backlog budget，以及 MonitorDataset 已 ingest 可见历史 budget；三者不能用一个 “buffer size” 冒充。

MonitorTap 可以请求小 backlog（例如 image 2–8 帧），其合同就是允许 overwrite 并报告 missed；可见 rolling history长度由下游MonitorDataset决定，LiveDatasetSlot只共享handle/revision并coalesce通知，绝不能再持第二份window/buffer。同一source的多个panel默认共享一个admitted slot，再用ViewSpec取`<= capacity`的窗口，而不是各自动态attach大帧ring。正式采集不能靠“把history设成full size”保证完整：RunController必须沿source -> processor -> sink exact chain汇总event/byte budget并在fire前reservation。MonitorTap与其唯一materializer都必须在首次publication前绑定；execute中或panel打开时不得新增raw tap/window。

M1已把这条规则接到一个真实free-running Workbench，但只对其冻结的单slot拓扑负责：Start前一次合计driver ring、adapter record retention、AcquisitionStream retention、capacity-one MonitorTap、capacity-one mutable MonitorDataset、一个frozen snapshot、metadata、Figure evaluation、INDEXED8 scratch/result、exact sample payload、Qt detached front与U0.2 per-panel hold；不足则在Run和camera arm前拒绝。拓扑固定为`FREE_RUNNING camera -> AcquisitionStream -> MonitorTap -> MonitorDataset.append_window -> LiveDatasetSlot`，首次publication后不能扩容或attach新window。这个准入不冒充未来整个Workspace/process-wide multipanel admission；M2仍须在真实多panel consumer出现时汇总共享slot、各view scratch/front和lane公平性。

当前 finite exact-capture preview 只交付一个更窄的本地合同：`CapturePreviewSpec` 从 exact cell schema 派生 `(R=1, MONITOR_HISTORY=1, *data_shape)`；compiler attach 边界唯一核对它与本次 exact capture 共享同一个 cell-schema owner 和 event-adapter owner，不能把 capture A 的 projection 接到 capture B。随后在 camera `open_session` 前把 exact transport、mutable builder（含独立 written mask）、最终 immutable block、全程 metadata、一个 raw tap payload、一个 MonitorDataset mutable window、一个 frozen snapshot、FigureEvaluator live cap、INDEXED8 scratch/result、palette/Qt detached front与一份retained exact sample payload一次相加。预算不足时 preflight 在任何 preview bind/arm/FIRE 前拒绝；运行中 preview bind/ingest/evaluate/raster 失败只撤下 preview，不能改变 exact result、CaptureArtifact 或 hardware cleanup。这个数字只证明**同一 Run 的一个 capacity-one slot**，不覆盖终态保留的其它 slot/front、并行 Run、整个 Workspace 或共享 executor；S5 process-wide topology admission 与 real qCMOS profile 因此仍为 NO-GO。

W1 的第一个产品 consumer 不再拿 `FigureEvaluationPolicy.max_live_nbytes` 当实际使用量：`zlc_frontend.figure` 的同一个 metadata-only estimator与真实 evaluator共用 `_layer_resource_upper_bound`，composition只把该 view 的实际 evaluation peak 加上INDEXED8 scratch、retained raster/sample payload、palette与Qt detached front owner的公开 estimate，再作为`downstream_peak_bytes`交还capture compiler做一次总准入。这样“大上限=大分配”的旧错误被消除，同时 estimator 也没有在Workbench复制第二套shape公式。该准入仍只覆盖一张finite image及其两个worker，不宣称整个process aggregate bounded。

### 7.7 Finite dataset 与 rolling monitor

三个 owner 明确分责，不允许一个双模式 builder 同时背 exact seal、broker backlog 与 GUI rolling：

```text
AcquisitionStream -> MonitorTap
  bounded event/byte backlog；overwrite/missed；不参与 exact retention/backpressure

ExactReservation -> DatasetBuilder
  完整 sequence -> DatasetCellAddress 排列；ack/EOS/seal；只能 finite exact

MonitorTap -> MonitorDataset
  keyed_cycle: 物理 cycle offset 0 或 sequence gap 时先原子清空旧值/validity/metadata
  append_window: materializer 按 event sequence 分配 ring slot，snapshot 统一 newest-first
  两者都只产生 MonitorDatasetSnapshot，绝不 seal
```

`MonitorDatasetSnapshot` 在同一临界区冻结 DataBlock、cell-aligned EventRefs、head 与 `MonitorCoverage`；controller 的 current selection 只来自该 snapshot，不拼接可能过期的 progress。append history 的目标 shape 必须是 `(R=1,P=history,*data_shape)`，且 P 只能是一条 `MONITOR_HISTORY`、dense `RECT_C`、坐标严格为 `0..capacity-1` 的 newest-first slot 轴；slot 0 表示本次 snapshot 中最新的 retained event，遇到 gap 时 slot n 不等于物理上的“n shots ago”，真实 source sequence 只能读取 aligned EventRef。任意二维/多维 `data_shape` 原样保留，绝不能借用 SCAN_POINT/READOUT_EVENT、塞进匿名 `(repeat,data_points,data_dim)` 三项容器或 `reshape(...)[0]`。正常 ring eviction 不计为 loss；`MonitorCoverage.missed_events` 是 lifetime telemetry，`current_gap` 只描述当前可见窗口，gap 滚出后 coverage 可以恢复 complete。formal `DatasetCoverage` 只保存 written/total；exact loss 已由 reservation sequence、cursor ack、完整 schedule 与 owner EOS fail-closed 证明，不保留一个生产中永远为 0 的旧 monitor 字段。keyed cycle 的 complete 只描述当前 sweep，绝不混入上一 sweep 的仍 valid cell。

交互 Fit/Save 若要使用 live 数据，必须把一个原子 snapshot 冻结为新的 owned finite diagnostic input并记录 event range、head、missed/coverage；不能把“当前 window”冒充从运行开始至今的完整 dataset。pending snapshot request 必须有界/coalesce，Python 引用不能绕过 retention budget让旧大帧无限存活。

finite preview 的唯一顺序是 `capture_next -> exact DatasetBuilder.consume/ack -> MonitorDataset.ingest_latest -> no-payload change notice`；显示永远由 worker 随后直接 `materialize(None)` 冻结当时的原子 snapshot，通知中的 ref 不会被保存后再延迟解析。`CapturePreviewPort.bind()` 原子把 MonitorDataset lifetime 转给 Workbench slot，runtime 此后只留 non-owning ingest handle；失败由 slot 唯一关闭，进入 exact allocation 后即使在 bind 前发生 open/reservation/builder 失败也会终止该 slot。只有整个 direct cleanup 或 pulse+camera aggregate cleanup 没有 primary error、cleanup error 或 UNSAFE decision时才发布正常 source terminal并把最终 capacity-one snapshot保留到 panel close；否则走同一个 `fail -> owner wake -> invalidate/clear`，不能让失败 Run 的最后front继续冒充有效。它不是第二份 finite truth、不会 seal，也不进入 artifact lineage。

产品状态也沿这条authority边界分开：只有`RunSnapshot.state == SUCCEEDED && final_committed`能把Capture标为`FINAL`；preview失败、board present失败或preview close重试都不能阻塞该snapshot的reconcile、result/reap或改变artifact。preview始终标`DISPLAY ONLY / latest rendered raw frame`，不能仅因Run已FINAL就声称当前raster必然是最后采集帧；FAILED/CANCELLED时旧front先退场，若presenter clear暂时失败则保留同一controller重试，而不是丢owner或把旧front重新标成PROVISIONAL。

M1 continuous monitor与finite preview共享数据面但不共享终态语义。配置为`FREE_RUNNING`的独立`monitor_camera`一次arm后由传感器曝光时钟产生frame；host循环只排空有界record queue，不用sleep或GUI刷新节奏调度曝光。每个record先进入稳定StreamId的fresh generation，再由capacity-one tap/materializer原子冻结`(R=1, MONITOR_HISTORY=1, SPATIAL_Y, SPATIAL_X)`、BlockId、head、aligned EventRefs与MonitorCoverage；IMAGE view用具名MONITOR_HISTORY AxisId显式选slot 0，不使用LatestNonempty、flatten或隐式data-axis reduce。该路径没有ExactReservation、seal、artifact、权威Fit/Save或formal capture含义。

adapter queue内的overflow、source ordinal/produced-count不连续或设备worker失败属于pre-broker source failure，必须终止Run并撤下front；只有record已经交给broker以后，MonitorTap overwrite和render coalescing才记为可见`missed_events/current_gap`，不能把前者伪装成UI落后。正常Stop也不保留最后一帧：先cancel并完成session-specific stop/join/SAFE verification，再`source_terminal`撤销publish authority和front；并发真实source error不能被稍后的用户cancel洗成CANCELLED。当前virtual role恢复main的MOT行为oracle：1920×1200 Mono8、free-running sensor clock、读取同一current sequencer的compiled/held三轴DAC输出、SAFE零场以及coil-space Gaussian fluorescence；移动高斯临时源不保留。real Pylon的`LatestImageOnly`若会在record owner之前跳帧，则在adapter能显式报告skip并完成contract qualification前保持NO-GO，不能用virtual严格队列外推真机正确性。

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

上图的 Blocking I/O lane 与 `ThreadAffinityKey` 是后续 adapter-hosting 的**终态迁移门**，不是本 runtime authority 闭包已经兑现的事实。当前 `DeviceBroker` 仍在 run-owner/interrupt 调用路径直接调用已登记callback，因此只有明确声明线程安全且通过contract test的adapter可接入；需要SDK owner-thread affinity的adapter在其composition/adapter owner中串行化，迁完前不得把generic broker描述成已有公平lane scheduler。引入共享lane前必须先有真实第二消费者与bounded blocking/interrupt证据，不能为了图完整预建通用调度框架。

不使用“每种职责固定一个全局 OS thread”。连续 camera monitor 不能阻塞无冲突设备；同一 thread-affine device 的调用必须串行。

独立 panel latest-only 是逻辑 mailbox（每 panel 最多一个 pending revision）；声明为同一 coherence group 的 panel 使用一个 board mailbox/evaluation revision，不各自挑 latest。它们由少量 bounded workers 消费，不是每个 panel 新建线程。worker 数和队列预算来自 WorkbenchProfile；满载时只替换尚未开始的旧 view/board work。Analysis executor 区分 formal/offline 与 interactive QoS：interactive 同 panel 新 revision 可替换尚未开始的旧 fit；formal/offline/明确保存的 Analysis 不 coalesce，满载时返回 typed Busy 或在 Run deadline 内排队。正式 StreamProcessor event 绝不进入可丢弃 view/interactive 队列。

当前单 panel live-image controller 不预建通用 scheduler，却必须兑现其静态“一份 snapshot/evaluation/raster”预算：它在任意注入的 worker executor 外再设一个 controller-local serial gate，并且只在上一 worker 调用栈真正返回、释放 snapshot/evaluated/raster 引用后提交 dirty follow-up。多 worker executor 因而不会让同一 panel 两套大帧重叠；这不构成跨 panel 公平 lane、queue policy 或全局并发预算。

### 8.2 RunController 与 RunHandle

```text
RunController.run(plan)   -> Result      # 同步，notebook/test
RunController.start(plan) -> RunHandle   # 后台，workbench
```

这两个是composition内部入口，不是notebook public API。public `Experiment.run/start`只接收declarative Request；composition在同一generation snapshot内bind成internal RunPlan并立即提交给RunController，既不返回plan也不把它挂到RunHandle。RunHandle公开面只有run id、status/wait/cancel/recovery/result/ref等生命周期DTO，不含RunPlan、prepared value、领域bindings、RunDevice/CleanupDevice或drive-capable Port。

`RunController` 是所有用户可启动 Run 的唯一 lifecycle owner，包括 one-shot Task、finite/continuous Measurement、FormalPulseScan、DatasetBuilders、StreamProcessorWorkers 和 formal Analyses。每次 `start()` 创建一个 run-owner thread；terminal state 只能在所有 I/O call、CaptureSession、online worker、materializer 和 required Analysis 确认退出后产生。

每种 Definition 只有一个与其语义一致的绑定结果：

```text
task owner builder(TaskDefinition key, request, immutable bindings) -> RunPlan[Result]
resolve MeasurementDefinition metadata + domain composition binder -> BoundMeasurement
processor owner binder(StreamProcessorDefinition key, config, contracts) -> BoundStreamProcessor
domain analysis owner builder(typed request, immutable artifact refs) -> flat RunPlan[Result]
zlc_data.bind_fit(FitSpec, expected DatasetSchema) -> BoundFit
```

Measurement 与 StreamProcessor 都不是独立 lifecycle owner，不直接返回 RunPlan；它们由静态 PipelineSpec 编译进一个顶层 RunPlan。用户“单独 Start Measurement”也编译成一个 source + DatasetBuilder/明确 sink 的最小 PipelineSpec，而不是特殊启动路径。已提交 artifact 上的 calibration/detection 等领域 Analysis 由自己的 typed request 直接编译成一个 flat RunPlan；generic post-materialization AnalysisStep 不是当前 baseline。这样不会为了统一方法签名而让不同语义冒充 Task，也不会为一个UI入口预建第二套生命周期。

MeasurementDefinition只含DefinitionKey、title、request/binding schema id与capture-spec owner fingerprint等递归声明式字段；动态 output schema/cardinality/budget 只属于 BoundMeasurement 的 capture contract。StreamProcessorDefinition同样只含DefinitionKey、title与config schema id；input/output/join contracts、operator、deadlines与artifact inputs全部属于BoundStreamProcessor并进入其fingerprint。DefinitionCatalog机械拒绝callback、raw driver、mutable cache或其它非声明式field。generic runtime不调用Definition.bind，也不接收任意`request: object/bindings: object`；各领域composition在自己的typed request/typed bindings边界完成纯验证并直接构造Bound值，bindings只含Bound Port和immutable config。compile_pipeline 是无硬件 I/O 的确定性构造，可做schema、owner、完整schedule和静态预算校验。Notebook 可以在调用线程直接构造；Workbench 把同一个同步函数投递给其普通 application worker，结果再交给 RunController.start。runtime 不定义专用 command/build lane、第二套队列协议或额外 Service；若 profiling 证明某个编译器本身很重，只把该纯函数放入现有 bounded CPU worker，不改变领域合同。RunController.start 只接收已经构造好的 RunPlan，因此不会持有 ResourceClaims 等待纯编译。

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

generic runtime 不保存领域 request/bindings 容器、execution mode 或 event/grouping 等领域字段。有限 exact 与 continuous monitor 的完整性、预算和 overwrite 语义属于 Measurement/Pipeline/Dataset contract；领域 composition 先冻结 typed request/bindings，再用不可变输入构造上述 callbacks。所有 timeout 必须 finite、非负且只用 monotonic clock；artifact timestamp 才使用 wall clock。

`preflight` 的返回值就是领域私有的 typed prepared value，不再套公共运行包装对象。它可以携带 resolved schemas、reservations、cursors 和其它准备结果；`execute` 只能收到这一个值与 `RunContext`，不能从 session、global registry 或 service locator 找回未声明 Port。不包含 child run、递归 DAG 或运行中新增资源。

每个 `device/...` 的 EXCLUSIVE claim 必须在 `bound_devices` 中恰好出现一次；普通 CPU、repository 和纯只读资源不伪装成 device。调用者不再构造第二份 `HazardClaim`：RunController 在完成 live binding identity recheck 后，唯一地从每个 `BoundDevice.binding_stamp` 派生内部 hazard。这样 claim、identity、evidence 与 connection generation 没有平行输入可错配。RunController 取得全部 ResourceClaims 后，必须先把同一 run 的 `HAZARD_ACTIVE` records durable append，才允许 configure、session start、arm、fire、safe、abort 或 interrupt。记录尚未持久化时 cancel 只单调设置 token；journal 写失败时 claims 保持占用且硬件调用次数为零。

baseline 的一个 `RunPlan` 只能使用同一个 `DeviceBroker`/installation authority；跨机器 endpoint 必须在它自己的 adapter/server 边界提供单一可验证 binding，而不是让一个 plan 拼接多套本地 arbiter。只有出现第二个必须共同驱动且无法归入同一 authority 的真实用例，才另行设计跨 authority 协调；当前只实现一个 ResourceLease、一个 SafetyJournal 和一个 run 级 SafetyDispositionBundle，不预建分布式提交器。

stable identity 必须由当前live connection的adapter receipt与installation-owned AssetMap共同建立；普通实验config、role、Python class、device index、枚举顺序或用户填写的字符串都不能自证物理身份。AssetMap不是一个手写revision标签：它必须是machine/device级持久、canonical序列化的`asset_id -> canonical ResourceKey + exact adapter kind + expected live identity/endpoint matcher`，revision取其canonical内容digest。真实runtime缺少AssetMap、adapter kind不符或live readback不匹配时，composition直接NO-GO；普通`load_config`不能创建/覆盖ResourceKey、expected matcher或revision。同role换成另一serial即使重启了进程和broker也必须拒绝；只有离线maintenance明确更新AssetMap并保留旧安全事实后，才允许下一次新进程启动验证该映射。

identity evidence明确分为`HARDWARE_IDENTITY_READBACK`与`INSTALLATION_ASSERTED_ENDPOINT`：前者读取设备serial/DNA等不可混淆硬件标识；后者只在现有接口确实没有硬件标识时，用稳定deployment endpoint + AssetMap revision证明“当前连接到被安装声明占用的控制端点”，不得声称已经读回同一块物理板。`PhysicalDeviceIdentity(stable_device_identity, evidence_kind, evidence_digest, asset_map_revision)` 是跨连接稳定的完整身份；`DeviceBindingStamp(physical_identity, binding_instance_id)` 是一次 live binding 的精确身份，并拥有唯一 canonical tree codec。`VerifiedPhysicalDeviceIdentity` 只是 broker-minted、一次消费的握手结果，成功 bind 后即被 `BoundDevice.binding_stamp` 取代。adapter只返回绑定当前live connection的 identity readback；每次成功startup或claim-first recovery handshake后，由DeviceBroker签发新的binding instance id，adapter不能选择、复用或自报。active Run首次检测到transport断开、device-removed或live-readback failure时，authority使旧binding失效；SAFE verifier仍必须执行自己的live readback，不能只信缓存状态。禁止transparent reconnect后继续execute或cleanup；普通重连要求safe shutdown与新进程，只有已存在durable blocker时可在exclusive RecoveryAttempt内建立recovery-only connection并取得新binding instance。每次Run start与每个safety verifier都重新核对完整 physical identity、runtime instance 与 `DeviceBindingStamp`。

领域 composition 的 immutable bindings 只含 consumer-owned Port/factory、typed Repository 和 immutable config，不含 QWidget、open CaptureSession 或任意线程外可直接调用的 raw driver。Port 调用由 RunController 路由到 owner lane；preflight 返回值中的 session token/handle 也只能由该 lane 消费。

bind 必须从 request/bindings 计算完整或保守 superset ResourceClaims。preflight 可以拒绝 claim 与硬件 capability 不匹配，却不能发现后临时追加资源；若某 adapter 的条件资源无法在 bind 时确定，Definition 必须声明 superset 或拒绝该 request。

真实硬件使用两阶段启动，但仍是单层计划：

```text
bind -> RunPlan
-> acquire_all static claims
-> 在正确 I/O lane 使用InstallationDeviceGraph已冻结的verified physical connection，
   创建本run的session/capture handle并执行configure/query preflight；不得reconnect
-> resolve ValueSchema/DatasetSchema、event/sample/byte budget
-> private prepared value(reservations, cursors, resolved contracts)
-> arm/sources ready
-> fire/execute
```

preflight 或 reservation 失败时不得 arm/fire，并释放已创建 reservation。CaptureSession 固定拥有 disarm；长期 device connection 的 close 属于 process-lifetime InstallationRuntime shutdown，只有 CaptureSession 自己创建临时 handle 时才负责 close。

device/session 的 create/open/configure/read/disarm/close 必须在其 ThreadAffinityKey 对应 lane 执行；composition root 只能在外部构造不接触 driver 的轻量 adapter/factory。真正raw SDK/driver对象只在 allowlisted InstallationDeviceGraph/DeviceBroker owner lane内部创建、保存和销毁；public `bind`/Definition/RunPlan/finalize不得接受或保留任意raw driver callback、bound method或可回调到driver的adapter object。CaptureSession 在 owner lane 创建并在同一 lane 销毁，不能在 run-owner thread 创建后交给 I/O lane 使用。

外部权威状态：

```text
RUNNING -> SUCCEEDED | FAILED
RUNNING -> CANCELLING -> CANCELLED | FAILED
```

waiting resource、arming、capturing、fitting、saving、finalizing、commit-reconciliation-blocked 是 phase，不是通用工作流状态。

由 `RunController.requires_final_commit` 管理的最终 artifact，其可见提交与 cancellation 使用同一个短原子 gate。`finalize` 可以在 gate 外构造和校验临时 artifact；`commit_final(FinalCommit)`只能使用owner Repository的`RepositoryCommitCoordinator`在startup reconciliation成功后铸造的opaque、不可变、单次 `CommitAuthority`。公开authority是无副作用handle：除冻结CommitTarget外不暴露`publish()`、journal、recover或callback；真正的`target/journal/publish/recover`快照只存在coordinator私有registry。普通plan只能携带handle，RunController通过内部consumer token原子pop签发快照；同一authority跨run/commit_id复用直接拒绝。lost-ack重试使用RunController已经持有的快照与稳定commit_id做reconciliation，不重新开放publish capability。随后在该Repository同一durability域持久化`CommitIntent(kind, commit_id, run_id, safety_bundle_id, target, created_at)`；无硬件hazard时`safety_bundle_id=None`。`CommitTarget`至少冻结repository_id、artifact_kind、artifact_format、target_ref与expected_manifest_digest，使重启后无需内存closure即可路由到唯一owner并验证目标内容。repository publish必须返回typed `PublishedManifest(target_ref, manifest_digest, result)`，owner快照逐字段匹配CommitTarget后才允许写COMMITTED，正常成功路径也不能跳过digest验证。返回类型错误、target/digest不符及其它确定性合同违例直接写ABORTED并失败，绝不能调用recover“洗白”；只有Repository明确抛出`PublishVisibilityUnknown`，表示atomic replace后可见性确实无法判定，才进入inspection-only recovery。intent fsync期间cancellation仍可受理。intent完成后在短内存gate内做最后一次CancellationToken checkpoint并关闭cancel gate，随后才允许manifest/rename publish：cancel先取得gate，则把intent幂等标为`ABORTED`、publish调用次数必须为0，run不能产生成功artifact；publish先取得gate，则之后的cancel明确返回`TOO_LATE_ALREADY_COMMITTED`（若run已terminal则为`ALREADY_TERMINAL`），不得把已经可见的成功artifact报成CANCELLED。长时间序列化、blob写入和intent fsync不在不可取消gate内；gate只保护最终可见发布及其结果判定。当前 `FitExecution.save()` 是 notebook/direct CAS 保存面，不携带 Run final-commit authority；它只能把同一 repository `execute()` 铸造的 process-local execution 交回 private `_save_execution`。publish acknowledgement 丢失时不返回成功 ref，可见但未被调用者引用的 manifest 只算 content-addressed orphan。它尚未纳入上述 lost-ack coordinator，不能被本轮 raw Capture/Calibration repository 的闭合结论代称为“全部 capture artifact 已闭合”。

`CommitTarget` journal 的字段名切换是 current-only durable protocol cutover：部署新软件前必须完成 startup reconciliation 并确认没有 pending 旧 intent；本项目不保留双 reader，也不建立在线 upgrade fallback。若现场仍存在 pending intent，先用旧 release 完成或明确终止该提交，再部署新 release。

manifest atomic replace成功但调用方因I/O/进程故障没有收到确认时，Repository必须把这一特定歧义归类为`PublishVisibilityUnknown`，不能用裸`OSError`把所有错误混成未知，也不能把确定性manifest校验错误送入recovery。每种Repository必须按稳定`commit_id`提供权威、幂等的`recover()`：确认已提交时返回`CommitRecovery(committed=True, PublishedManifest(target_ref, manifest_digest, result))`，RunController再次逐字段匹配冻结CommitTarget后才追加`COMMITTED`并完成SUCCEEDED；确认未提交则追加`ABORTED`并按原publish error失败。错误target/digest、无typed manifest evidence或任意字符串result不能证明恢复成功。Repository或commit journal暂时不可判定时，Run保持非terminal `RUNNING/commit-reconciliation-failed` phase、关闭cancel gate、持有resource claims并给出显式重试指令。`COMMITTED`与`ABORTED`在跨进程文件锁内互斥验证，二者都清除pending；commit marker自身写确认丢失也走同一reconciliation，不能重复发布或提前释放claim。startup在接受新run前枚举所有pending CommitIntent并调用对应owner Repository的reconciler；无法找到owner/schema或仍无法判定时fail closed，不重新fire、不把temp文件当成功artifact。

pending reconciliation必须冻结“事实是否已经确定”，不能每次重试重新询问可变callback：`FORCE_ABORT`用于确定性publish/validation失败或validated recovery已确认未提交，重试只幂等写ABORTED；`RECOVER_VISIBILITY`只用于尚未判定的PublishVisibilityUnknown，只有此态调用recover；`FORCE_COMMIT`用于publish已返回并验证成功或validated recovery已给出匹配manifest，持有已验证result并只幂等写COMMITTED。marker写入/确认失败只重试相同resolution，不得让wrong digest经一次abort-marker故障反转成成功，也不得让已可见artifact经一次commit-marker故障反转成ABORTED。

`run(plan)` 内部也使用同一个 RunHandle。Notebook/test 遇到 KeyboardInterrupt 时先 cancel 该 RunHandle、等待 cleanup acknowledgement，再重新抛出或返回取消结果。若等待超过 join deadline，抛出携带 run_id/RunHandle lookup 的 `RunStillCancelling`，RunController registry 继续持有 handle/claims；不能丢掉 handle 后把 cell 当成已经停止。notebook 可继续 `status()/wait()/recovery_instructions()`。

RunController registry 强引用所有 active handle，以及已经发布 terminal 但 owner thread 尚未被确认退出的 handle；只有另一个线程完成 join/reap 后才移除。`RunHandle.wait/result` 在返回 terminal 结果前也必须确认 owner thread 已退出，不能把“状态字段已写入”冒充线程终止。handle/snapshot 只保存有界字符串错误摘要与必要结果，不保存 `BaseException`、traceback、plan、prepared value、context 或 raw device graph；owner 收尾时主动断开这些引用。baseline 不另建 terminal-handle archive/`forget_terminal` 状态机，持久诊断事实归 artifact、commit journal 与 SafetyJournal。

### 8.3 CancellationToken

- 每个 Run 由 controller 私有 `_CancellationSource` 新建，只向 plan/worker 暴露不可 clear、不可 request 的只读 `CancellationToken`；
- 单调从 active 变 cancelled；
- 绝不 clear/reuse；
- cancel requested 不等于 worker terminated；
- join timeout 后不得清 thread owner、释放资源或允许 restart；
- 每个阻塞 Port 必须有 bounded timeout 或 interrupt contract；
- cancellation 先置 token，再调用 Port 声明为 thread-safe 的 out-of-band `interrupt/abort`，随后由 owner lane 完成正常 cleanup；
- hazardous run 的 out-of-band interrupt 只有在对应 `HAZARD_ACTIVE` 已持久化后才 enable；此前 cancel 只置 token；
- interrupt 一旦启动就是 terminal barrier：interrupt call 未返回时不得开始可能与其并发碰硬件的 cleanup、不得发布 terminal、不得释放 claim；其迟到异常必须进入 CleanupReport，不能被后台线程吞掉；
- safety-critical Port 不能让 `safe_state` 永久排在可能无限阻塞的同一调用之后：必须有 transport timeout、独立 abort/safe channel 或硬件 watchdog 中至少一种可验证机制；
- 不可中断的 SciPy/NumPy 计算等待返回后丢弃 stale result；
- 需要 hard deadline 的计算使用 disposable subprocess。

current `RemotePulseExecutionClient` 必须建立两条不同的RPyC connection：control通道执行snapshot/prepare/fire/complete，interrupt通道只执行generation-bound safe-state；endpoint的logical blocking limit必须严格小于transport backstop。server返回的 `PreparedPulseRef(connection_generation, artifact_digest)` 在prepare acknowledgement前原子写入同一private session，后续fire/complete只消费该exact ref；每次操作重验server generation、target、clock与geometry，禁止transparent reconnect。第二条socket保证长complete RPC不会在客户端协议层堵死safe请求，但若两条请求最终共享backend/硬件 `_io_lock`，它仍不能冒充独立硬件中断路径；baseline也不因此新增watchdog、SAFE寄存器或重烧bitstream。

logical deadline必须覆盖endpoint的SAFE single-flight lock等待与interrupt RPC本身，不能在transport backstop返回后才检查时钟。当前client用RPyC timed request消费调用方传入的剩余时限；超时后先永久撤销整个client，再直接切断两条本地transport，使迟到ack永远不能成为当前证据且调用方按logical deadline返回。transport断开会触发server owner-disconnect SAFE，所以真正的重复调用约束由`PulseExecutionService`唯一拥有：所有generation-bound SAFE、disconnect SAFE和failure recovery SAFE共用一个single-flight gate；后到者等待正在执行的物理SAFE，若其成功并清空prepared authority则直接复用同一SAFE snapshot，只有前一SAFE失败才允许新的物理重试。这样不依赖client/endpoint/runtime对象寿命，不因GC再发第二次backend SAFE，也不启动detached watchdog或伪装远端调用已经终止。

SAFE还必须与prepare/FIRE/complete在**物理backend边界**线性化，而不只是更新Python state。server先把operation epoch推进到INTERRUPTING并调用声明为out-of-band的`request_interrupt()`，再等待唯一backend-operation gate；因此已经进入backend的调用会先退出，随后同一owner才真正执行`backend.safe_state()`。尚未进入backend的普通调用在取得gate后必须重新校验epoch/state，已被SAFE超越就零硬件调用失败。普通调用完成backend后也要在发布ack前再校验epoch；这样不存在“软件先宣布SAFE、迟到prepare/FIRE随后又改硬件”的窗口。`request_interrupt()`故意不取得这个gate，否则它会排在正在阻塞的硬件调用之后而失去中断意义；adapter contract必须明确它线程安全、非阻塞且不等待普通I/O owner。service state lock不得跨backend-operation gate等待或阻塞式backend命令；failure recovery也只能在释放该gate后复用同一SAFE owner。只有physical SAFE成功、prepared authority清空后才能发布权威SAFE snapshot；运行中普通snapshot只是观察值，其backend部分必须由thread-safe、non-blocking readback提供，不能冒充terminal proof。普通session cleanup不再先执行一遍独立SAFE再close；`close_session`是该路径唯一SAFE owner：它先触发server的上述物理顺序，若本地operation线程仍在收尾则join后只再次确认同一SAFE snapshot，不重复backend SAFE。无法确认safe时Run保持CANCELLING或内部FINALIZING_SAFETY/SAFETY_JOURNAL_BLOCKED并持有claims；本installation的同一个SafetyDispositionBundle durable、UNSAFE quarantine projection成立后，才发布FAILED。只有真机证据表明现有safe路径违反既定安全要求，才按bug修复流程评估硬件改变。target-native remote endpoint已保留，但真实 `RemotePulseExecutionClient -> DeviceBroker -> BoundPulsePort` composition在AssetMap/identity/SafeStateContract闭合前仍为NO-GO，旧`RemoteSequencer`不得恢复或包装。

### 8.4 Cleanup

普通session/temporary resource优先使用同步 context manager 和 `try/finally`。`RecoveryAttempt`是明确例外：journal acknowledgement 不确定时必须保留同一attempt/bundle重试，因此禁止“异常即自动abort”的context-manager sugar，只能显式complete/abort。安全关键 abort 的顺序先消除物理危险，再清理软件对象：

```text
cancel intent
sequencer out-of-band abort/safe -> logical terminal/safe acknowledgement + H1 post-terminal tail recipe
CaptureSession cleanup: adapter-specific terminal drain -> camera stop/disarm -> stable check -> release/join
workers/builders abort or drain + join
temporary handle: only its creating CaptureSession closes it
temporary config restore
reservation release
live identity + terminal safe verification -> SafetyProof or UNSAFE decision
revoke this Run's broker execution/cleanup lease
append the same SafetyDispositionBundle durably
construct hardware-free PostSafetyContext
finalize/commit
terminal publication + ResourceClaim release as one arbiter-visible boundary
```

长期 raw connection 不属于单次Run cleanup；它只在 process-lifetime InstallationRuntime shutdown 按 §12.7 的 composition 顺序关闭。

业务错误保留为 primary error，cleanup/safety failure 作为附加错误。安全清理失败不能报告成功或普通取消。

一旦adapter的terminal recipe证明最后一个硬件 sample 已取得、trigger source不再产生新工作且设备不再需要，正常路径立即退出 CaptureSession、完成适用的drain/stop/disarm/safe，再进行长时间 fit/calibration/artifact commit；`finally` 是异常兜底，不是把安全动作拖到所有磁盘/CPU 工作之后。对Formal qCMOS，“最后一个sample已取得”必须按§14.5保持camera capturing完成terminal drain、冻结final metadata后才成立，不能以“队列暂时达到expected N”或先`cap_stop`代替。硬件 safe acknowledgement 失败时不得提交宣称整个 Run 成功的最终 artifact。

cleanup command ACK与物理安全证明必须分型。`abort/disarm/read-status/safe-state-command`只产生`CleanupStepAck`，表示该步骤返回，不能直接解除hazard；session termination使用独立`SessionCloseCommand`与typed `SessionClosedAck`，仍只证明本session终止。只有adapter的终态verification recipe完成真实safe-state/no-more-trigger/readback肯定验证后，DeviceBroker私有run lease才返回绑定完整stamp的`SafeReceipt`，随后RunContext用run-scoped私有nonce/registry铸`SafetyProof`；CleanupReport的SAFE分支只接受该proof，公开可构造的receipt或普通step ACK不能提交SAFE disposition。

capability 生命周期不能用一条“默认单次消费”抹平不同语义：`VerifiedPhysicalDeviceIdentity` 是一次握手结果，bind 成功即消费；`BoundDevice` 是不含raw callback的immutable binding reference，直到identity/transport失效、recovery replacement或broker shutdown；`VerifiedDeviceCapability` 是当前 binding generation 的冻结能力事实，不是execution lease，它跨Run有效，直到成功re-probe supersede、binding失效或broker shutdown；真正的执行排他由broker私有 `_DeviceRunLease` 承担；`SafetyProof` 只属于一个run并在cleanup中消费。所有对象都由owner签发并在消费时核对owner、ResourceKey、完整 `DeviceBindingStamp` 与私有nonce/registry事实，但不能为了表面统一给每种事实再包一层一次性lease。

closure introspection、扫描`__closure__`或检查finalize函数签名不是capability confinement。post-safety“不能再碰硬件”由构造边界完成：raw driver从未离开owner，plan只拿RunDevice/CleanupDevice代理；cleanup完成后broker run lease先被不可逆revoke，再构造不含hardware verb的`PostSafetyContext`并执行finalize。若任意application模块仍可持有raw SDK对象并在finalize直接调用，系统就是违反composition contract，必须用import/constructor allowlist与真实入口E2E清零，而不是增加更多proof wrapper掩盖泄漏。

只有 worker/session 与 in-flight interrupt 已退出，且每个device hazard都得到恰好一个 `SAFE(SafetyProof)` 或 `UNSAFE(reason, recovery_action)` 决定后，RunController 才撤销broker run lease。join 超时保持 CANCELLING 与原claims；safe verification失败生成UNSAFE，不能在finally中无条件release。一个run的全部决定由同一个 `SafetyDispositionBundle` 原子覆盖：SAFE record必须引用完全相等的 `DeviceBindingStamp`，UNSAFE record产生quarantine projection；bundle durable以后才进入不含硬件能力的PostSafetyContext，UNSAFE run不提交成功artifact。

ResourceArbiter 使用 machine/device-installation 稳定目录中的 append-only `SafetyJournal`，不能放在用户可切换的 artifact RepositoryRoot 中。它统一记录 `HazardRecord`、`SafetyDispositionBundle` 与 `RecoveryBundle`；quarantine只是UNSAFE disposition产生的未解决projection，不建立第二本日志或第二套authority。每个run在第一次可能改变设备状态前一次性追加hazards，不为每个trigger fsync。HazardRecord携带完整 `DeviceBindingStamp`；SAFE必须匹配同一 physical identity 与同一 connection generation。时间字段只用于人类诊断，不参与admission、重放顺序、幂等或因果判断；因果只来自exclusive claim、append顺序和精确 record id引用。

MemorySafetyJournal 在每个entry完整验证后增量维护 unresolved projection，不能每次append全量重放。PersistentSafetyJournal委托 storage-owned `FramedJournal` exclusive append session：启动时scan/repair一次，正常steady append不重扫历史；lost-ack使缓存未知并至多触发一次refresh。持久实现启动时持有覆盖整个installation lifecycle的owner lock，第二个authority直接拒绝。已绑定ResourceArbiter的journal不能被外部提前close。

Recovery采用claim-first：`RecoveryController.begin(ResourceKey)`先在broker为空也可行的情况下取得该**精确** blocked key的exclusive recovery claim，并冻结唯一 `blocking_record_id + PhysicalDeviceIdentity`。composition随后建立或取得recovery-only live binding，`RecoveryAttempt.complete(binding)`在同一claim内验证live identity与safe state。正常SAFE要求full stamp完全相等；跨重启recovery允许新的connection generation，但 `PhysicalDeviceIdentity` 四字段必须完全相等。调用方不得用会在异常时隐式abort的context-manager sugar：取得attempt后只能显式`complete(binding)`或`abort()`。journal acknowledgement不确定时attempt继续持有claim，retry必须复用同一binding、evidence与bundle id；调用方只能重试同一attempt或保持fail-closed并重启，不能自动abort后新建bundle。历史已resolved时返回既有事实，不复活旧run；显式abort或进程故障保留原blocker。普通startup binding与Run admission都拒绝blocked key。

同一physical identity不能同时绑定两个ResourceKey，同key也不能静默换成另一个physical identity；换机必须新建ResourceKey/AssetMap事实并保留旧blocker。硬件sticky fatal/status若存在优先于软件观察；没有该能力时SafetyJournal仍覆盖进程崩溃、driver无持久fatal和remote socket重建。真实adapter必须使用持久journal；MemorySafetyJournal只用于virtual与测试。

journal append失败时run保持内部`safety-journal-failed`非终态phase并继续持有claims。retry只重交同一个已缓存bundle，不能生成新record id、改变决定或重新运行硬件cleanup。baseline持有所有claims直到safety bundle与terminal publication完成；没有第二个真实last-use消费者与独立证明前，不引入phase-release状态机。

### 8.5 Owner-thread command mailbox

长时间 Measurement 的参数修改通过 command mailbox 送到 owner thread，并只在 shot/capture transaction 边界应用。GUI 不跨线程直接 configure driver。有限正式 Run 默认拒绝运行中 reconfigure。每个 accepted control revision 遵守 §7.2 的 terminal ack 合同；同 key 的尚未应用 revision 可被较新 revision SUPERSEDED，但已经开始硬件 transaction 的 revision 不能假装被覆盖。

Blocking I/O lane 的普通 command queue 有界并使用公平调度：同一 owner 的连续 monitor read 以小 transaction 重新排队，不能永久压住已获资源的 finite run/control apply；不同 ResourceKey 使用 round-robin 或等价的 bounded-wait 规则。safety interrupt/abort 不进入普通公平队列，走 §8.3 的 out-of-band 通道。公平只决定已合法排队工作的顺序，不允许绕过 ResourceClaim 或并发调用 thread-affine driver。

公平不能只是一句 round-robin 意图。每条共享 lane 在 composition 时冻结一个带 `policy_revision` 的 `LaneFairnessPolicy`：`max_monitor_burst` 限制同一 ResourceKey 连续执行的 monitor transaction 数；`transaction_deadline_by_kind` 为 lane 上每种 blocking command 声明从真正进入 driver call 起的有限 deadline；`accepted_finite_max_queue_turns` 与 `accepted_control_max_queue_time` 分别限制已经 admission 的 finite/control command 最多被多少个其它 transaction 越过以及最多等待多久。scheduler 保存 accepted/start/finish monotonic time、实际等待 turns、deadline/timeout 原因和 policy revision，供 RunEvent、ControlTopic terminal ack 与 profile gate 使用。具体数值来自 adapter contract/profile 并随部署配置冻结，不能在运行中由 monitor 或 UI 放宽。

真实 adapter 只有在 SDK/transport timeout、可验证 cancel/abort 或可终止的隔离 process 能让每个已声明 transaction deadline 成立时，才可把该 command kind 放入共享 lane；普通 Python worker thread 不能因“逻辑超时”被视为已经终止。`None`、无限 timeout 或“超时后留下仍可碰硬件的后台调用”不满足合同，preflight 必须拒绝或改用有独立 owner 且具备真实终止合同的隔离 lane/process。queue-wait 上限到期且 command 尚未开始时，scheduler 不执行迟到硬件动作，而是给 control revision 发唯一 `DEADLINE_EXCEEDED` terminal ack，或使 finite run 进入正常 cancellation/cleanup。已经进入 driver call 后超时则按 §8.3/§8.4 的 interrupt、真实 termination、safe proof/quarantine 处理；不得并发启动替代调用或提前释放 claim。

## 9. ResourceArbiter

```text
ResourceClaim:
  ResourceKey
  mode = EXCLUSIVE | OBSERVE
```

Run 启动前一次解析全部 claims 并原子 acquire_all。运行中禁止新增 capability 或 lease。

ResourceArbiter 只证明同一 composition root 内的互斥。真实 adapter/server connection 还必须由具体backend/composition用 SDK exclusive-open、server-side owner token 或本机 interprocess lock 证明 notebook、standalone launcher、Workbench 和远端 client 之间的物理排他；无法证明时只能开放一个真实控制入口，不能把 EXCLUSIVE 描述成跨进程事实。物理owner proof丢失时先停止普通admission并走safe/quarantine；若已有durable blocker只允许claim-first recovery，否则执行safe shutdown并由新进程重建，绝不让另一个进程静默接管。generic runtime不为此新增平行lease类型，也不因此要求新硬件watchdog。

claims 在 bind 时声明完整superset，baseline始终持有到SafetyDispositionBundle durable、finalize/commit结束并线性化发布terminal。当前没有第二个真实消费者证明提前phase release值得新增状态机，因此不在private prepared value上标last-use，也不提供运行中re-acquire。

ResourceArbiter 只返回：

```text
Acquired
ResourceBusy(conflicting_run)
ResourceQuarantined(reason, recovery_action)
```

它不自动停止其它 run。Workbench 可请求停止冲突 owner，但必须等待其 RunHandle 确认 termination 后再重试。

普通 Run 之外只保留一个受 Arbiter 约束的设备安全入口：

```text
attempt = RecoveryController.begin(ResourceKey)
  -> RecoveryAttempt(exact blocking_record_id, PhysicalDeviceIdentity)
  -> composition establishes/obtains recovery-only BoundDevice
  -> attempt.complete(binding) -> durable RecoveryBundle
  -> attempt.abort() -> original blocker remains
```

正常 startup open/bind 不是 ResourceArbiter lease：它只在 public/Run admission 尚未开放时，由同一 `InstallationRuntime` 在 physical-owner proof 和 owner lane 内完成。运行中没有普通 reconnect、replacement 或 connection-establishment API。active binding 失效后，ordinary command fail closed；没有 durable blocker 时执行 safe shutdown 并由新进程重建，有 blocker 时只能进入上述 recovery 路径。

RecoveryAttempt 只能引用一个已经存在的精确 blocker，与所有普通 EXCLUSIVE/OBSERVE claim 互斥。它不是绕过 RunController 的“管理员后门”：composition 只开放 adapter 声明的 identity/status/safe 最小路径和 bounded timeout。`complete(binding)` 在 probe 前后都重验 binding 对象；相同 PhysicalDeviceIdentity 但新的 binding instance 可以跨重启恢复，任一 physical identity 字段不同都拒绝。attempt 不实现异常时自动 abort 的 context-manager 语义；owner 必须显式 complete 或 abort。journal acknowledgement 不确定时保持原 claim、binding、evidence 与 bundle id，重试同一 attempt；一旦存在 pending bundle，`abort` 必须拒绝，因为 durable append 可能已经发生。不能生成新 bundle、解除 blocker 或继续普通启动。进程崩溃与尚未开始持久提交时的显式 abort 都由 durable journal 保留原 blocker。用户确认若产品需要，属于 composition/UI policy，不进入 generic runtime 状态机。

`AssetMap` 是 installation-owned、machine/device级持久配置，只保存 `asset_id -> canonical ResourceKey + exact adapter kind + expected physical identity/endpoint matcher`；revision是完整canonical内容的digest，不能是代码常量、版本昵称或由实验role派生的字符串。它不在实验preset、用户可切换repository或普通`load_config`中，也不是另一套device registry。更新AssetMap属于离线maintenance/换机操作：旧进程先完成safe shutdown并退出，maintenance原子更新map且保留旧hazard/quarantine事实，新进程启动时重新执行identity/recovery验证；普通实验只能引用已有asset_id。启动时必须检查map的当前 plain format name、canonical digest、ResourceKey唯一性、matcher可判定性与所有真实adapter覆盖；缺项、歧义或未知adapter一律在composition阶段拒绝，不能留到某个LogicNode首次使用时才失败。

capability probe一次返回完整frozen snapshot，camera/sequencer descriptor只能从该snapshot纯函数投影；`TargetDeviceEndpoint`没有第二个`describe()`硬件/RPC入口。probe结束后不存在仍可读取raw connection的裸callback。physical/virtual config、设备、adapter或topology改变时不在原runtime内重建probe/binding graph；§12.7关闭全部authority和raw graph后进程退出，下一进程从零建立新的runtime instance与connection generations。

OBSERVE 应尽量使用独立只读 capability Protocol，而不是把同一个控制对象运行时阉割为 read-only wrapper。

OBSERVE 不等于允许第二个 session 并发读同一 driver。对 camera 等单 owner 设备，monitor 通过已有 CaptureSession 的 broker tap 观察 immutable samples；只有硬件/adapter 明确提供可并发只读 capability 时才创建 OBSERVE claim。没有该 capability 就与 EXCLUSIVE 冲突，不能用 wrapper 绕过。

同一 ResourceKey 上多个 OBSERVE 可共存；EXCLUSIVE 与任何其它 claim 冲突。ResourceKey 由 device owner 提供 canonical hierarchy，父资源的 EXCLUSIVE 与子资源 claim 冲突。`acquire_all` 对完整 claim set 一次判定并提交，不逐个等待，因此不依赖调用方排序规避死锁。

## 10. Task、Measurement、StreamProcessor 与 Analysis

### 10.1 Definition 原则

Definition 是递归声明式 frozen metadata，不含 callable、Port、Repository、GUI、mutable config 或 binding generation 事实，也不需要每类再建立 Handler Protocol 和公共 ABC。`DefinitionCatalog` 机械拒绝 callable；领域 owner 的 builder/operator 仍是具名 top-level 函数，由 composition 通过普通 import 显式调用，不以字符串 dispatch、隐藏 registry 或 Definition field 形成第二套执行真相源。所有运行依赖必须出现在 typed request 与领域私有 immutable bindings，所有可变参数必须进入 config revision。

只有会出现在 catalog/UI/API 的能力需要 Definition；Task 内部私有算法保持普通函数。

Definition 发现不依赖 global mutable registry、包扫描或 entry point：

```text
DefinitionKey:
  owner_package
  stable_definition_id

DefinitionCatalog:
  definitions: immutable tuple
```

`zlc_neutral_atom` 拥有 DefinitionKey/DefinitionCatalog，各领域模块显式导出 definitions tuple；composition root 通过普通 import 组装 catalog，重复的 `(owner_package, stable_definition_id)` 启动即失败。Definition 没有平行 schema 版本；定义的声明字段改变后，全套当前软件原子部署。Workbench 用本地 adapter 将它们映射为 CatalogView；排序、分组、图标和可见性只存在于 CatalogView，不反向写进领域 Definition。zlc_data/zlc_pulse/frontend 不为了进入 UI catalog 而依赖 neutral_atom 的 Definition 类型，也不建立跨 bounded-context universal Definition base。

Catalog composition 对每个 DefinitionKey 必须产生一个显式 visible mapping 或 hidden reason；未处理 definition 使 architecture/E2E失败，避免新领域能力已经注册却在 UI 静默消失。迁移期 CatalogRouter 用同一规则保证一个 use case只有 legacy 或 new入口可见，不制造双启动按钮。

### 10.2 Task

```text
TaskDefinition[Request, Result]:
  stable DefinitionKey
  parameter/request schema
```

Task 是 one-shot use case，可以同步组合 CaptureSession、纯 operator 和 typed Repository。Definition 只声明 catalog identity/request schema；owner/composition 的显式 builder 才把 typed request 与 bindings 构造成 `RunPlan[Result]`。它不继承 Measurement/StreamProcessor/Analysis，不发布 measurement signal，不拥有 QWidget。

Task 不一定产生 artifact；普通控制/查询 Task 可返回 immutable result，需要持久化时返回本包 typed ref。

Task 的中途数值/图像显示不重新建立 `TaskOutput`。需要 live frame、3D map 或优化轨迹的 Task，必须在同一 RunPlan 中声明 finite exact DatasetBuilder 或 admitted `MonitorTap -> MonitorDataset -> LiveDatasetSlot`；RunHandle 只暴露 slot 的 coalesced revision 通知，Workbench 从 slot 取得原子 MonitorDatasetSnapshot。阶段/progress/warning 仍走 EventStream。这样中途 UI、最终 Analysis 与 artifact 使用同一 materializer/revision 真相源，Task 不发布第二份 mutable signal，也不会因删除 `__task_frame__` 丢失现有用户功能。

### 10.3 Measurement

```text
MeasurementDefinition:
  stable DefinitionKey
  request_schema_id / binding_schema_id
  capture_spec_owner_fingerprint
  display metadata

BoundMeasurement:
  FrozenCaptureSpec(owner fingerprint, canonical bytes, digest)
  bound Device Ports
  output schema/cardinality/budget contracts
  ResourceClaims
```

```python
capture_spec = domain_build_frozen_capture_spec(typed_request)  # 纯函数，不碰设备
with capture_factory.open(ctx, capture_spec) as capture:
    sample = capture.read_next(timeout)
```

Measurement 从外部世界取数据，可以访问 Device Port；它不 fit、不渲染、不保存 Figure、不管理 Task terminal state。runtime 负责将 AcquiredSample 包装为 Envelope。

DeviceCapabilitySnapshot 是 connection generation health handshake 后得到的 immutable、versioned descriptor。bind/UI 用它纯解析 expected payload/ValueSchema，preflight 再读取硬件实际设置并要求 fingerprint 相符。formal run 不允许 fire 后才发现 shape/axis；无法预先确定 schema 的 adapter 只能提供 monitor 或先执行独立 probe/config Task 后重新 bind。

领域composition构造 immutable BoundMeasurement/FrozenCaptureSpec；runtime只验证owner fingerprint与canonical bytes SHA-256，不执行任意spec snapshot/validate/digest回调，也不会在session中二次freeze。Pipeline Run preflight只建立software CaptureSession/reservation/materializer，execute才在owner I/O lane发送prepare/start。Task 若需要同一种采集，复用同一FrozenCaptureSpec构造器/CaptureSession，而不是启动一个 child Measurement Run。

### 10.4 StreamProcessor

```text
StreamProcessorDefinition:
  stable DefinitionKey
  title
  config_schema_id

BoundStreamProcessor:
  frozen admitted config
  input/output payload contracts
  join-key/cardinality/lineage contract
  pure top-level operator
  operator/terminal deadlines
  output identity + artifact inputs
```

```python
output = operator(joined_inputs, config)
```

StreamProcessor 只处理当前 Envelope payload 或声明的完整 key group，不访问设备、Hub 私有状态、latest、累计 DataBlock、Repository 或 QWidget，也不创建 Envelope。`StreamProcessorWorker` 负责 subscription、join、validation 和 publish。

`BoundStreamProcessor` 在 bind 边界由一个递归 owner 同时完成 config admission 与 owner snapshot，再从已准入值投影一次 canonical tree/fingerprint；之后读取 lineage 只返回缓存 fingerprint，不能为了防反射篡改而在每次属性访问重新遍历/哈希大 ndarray。当前 config dataclass 的获批基线是 stable module-owned、直接继承 `object`、exact frozen、只含声明字段的 plain dataclass；不预建继承/C-extension/container-backed config 体系，`replace()` 重建后必须保持 exact type并复核全部最终字段。Enum 与 operator 的 module+qualname 必须解析回当前 exact class/function。finite float、cycle、canonical string key、显式little-endian dtype、canonical C strides 与 ndarray 都由同一 owner处理；MappingProxy owner-copy按 canonical key order重建，ndarray owner以独立header直接重基到长度恰好相等的终端immutable `bytes`，不能把caller header、可逆readonly view、hidden slice或额外backing带入bound值。snapshot与canonical projection的memo都强持有`(source, result)`，不能用裸整数id让临时validity/array释放后的id复用污染另一个字段；显式typed generation owner identity与dataclass构造器确定建立的derived alias仍保留。

config/fingerprint 的语义域是声明类型、字段值与canonical layout。普通Python `id`、caller alias topology、ndarray base/address以及frozenset的hash-table iteration order都不是配置语义；有顺序需求必须声明tuple，reviewed operator不得从这些表示细节派生结果，也不得修改bound config。Enum member/class、operator函数或module binding的运行中monkeypatch属于软件代码热修改，不是config content lineage；获批运行假设同一进程的软件定义保持不变，operator只使用声明的Enum/字段语义。`ExactStreamProcessorWorker` 在claim input consumer前一次验证bound、reservation/cursor、FrozenDatasetEdge、input/output contract、downstream与deadline/cancellation的真实owner graph；不叠加没有production issuer/consumer的可选callback guard。物理capture authority继续由CaptureSession/`CaptureProcessorInputBinding`领域边界拥有，不能伪装成通用processor hook。

每个 StreamProcessor invocation 原则上发布一个 frozen typed payload。多个同 shot 结果组成一个 record，例如 `OccupancySample(occupied, counts, source_metadata)`；UI/下游通过 field projection 读取字段，不把同一物理结果拆成多个需要分布式原子提交的 signal。只有字段具有不同 cardinality、key 或生命周期时才拆成独立节点。

operator 不读 wall clock、module global config 或 global RNG；需要随机算法时 seed/RNG algorithm 是 immutable config 与 lineage 的一部分。相同 input/config/immutable model identity 必须可重放，允许的浮点容差由 operator contract 声明。

BoundStreamProcessor 必须冻结 output join_key 是 pass-through、typed compose 还是 intentionally absent；`StreamProcessorWorker` 按 bound contract 生成/验证，operator 不能从 payload 猜 key。具体 input/output/join fingerprints 与 deadlines 都是 binding/config generation 事实，必须进入 Bound fingerprint，不能塞回启动期 Definition。Formal exact pipeline 中在最后一次所需 EXACT_KEY join 之前不得丢弃 key。

cardinality contract 明确 `1:1`、固定/有界 fan-out、`group K:1` 或 intentional filter，并给出 EOS completeness 规则和 max output bytes。Pipeline preflight 据此计算预算；FormalPulseScan 通往 ScanCellKey y 的路径不得存在未在 ScanOutputContract 中解释的 filter/drop。

output payload schema 必须只依赖 input contracts/config，不能读第一帧后改变 axis 数量或 record fields。站点发现、模型选择等 data-dependent schema discovery 属于 finite Task/Analysis/artifact 构造；其结果若要进入 formal pipeline，先以 SiteMap/CalibrationArtifact 等 immutable input 固定 schema，再 bind RunPlan。

需要跨 event group 状态的在线算法使用明确 StreamReducer；普通 StreamProcessor 不带可选 start/update/finish/reset。对完整 scan/capture 数据集做 fit、calibration 或 report 的算法属于 Analysis，不使用 StreamReducer 模拟 batch analysis。

StreamReducerDefinition 必须提供 state factory 与 update/finalize contract；每个 `StreamProcessorWorker`/RunHandle 创建独立 state，不能复用 module singleton 或上一次 run 的缓存。state 的 schema/config revision 固定，EOS incomplete/cancel 时不得把 partial finalize 结果发布成成功输出。

当前任何会 camera grab 或 fire sequencer 的 one-shot Processor 都必须重新分类为 Task 或 Measurement。

### 10.5 Analysis

```text
BoundFit:                              # zlc_data-owned，不含 runtime slot/ref
  frozen FitSpec + expected DatasetSchema fingerprint
  resolved fit/batch axes + model/numeric policy
  run(OwnedSnapshot) -> FitResultBatch

DomainAnalysisRequest[Result]:         # 仅在领域物理语义需要时定义
  exact immutable ArtifactRef inputs
  frozen domain config/policy
  resource/compute budget
  compile -> one flat RunPlan[Result]
```

Analysis 不访问 Device、Hub/latest、QWidget 或 mutable DatasetBuilder。它消费携 exact DatasetRevisionRef 的 OwnedSnapshot或immutable artifact，产生 FitResultBatch、CalibrationArtifact、report 等 typed result。`FitSpec/BoundFit/FitResultBatch` 与 `BoundFit.run()` 全部由 zlc_data 拥有；Calibration/ReadoutFidelity 等带 neutral 物理语义的算法使用自己的typed request/compiler，不以generic Fit wrapper冒充领域判断。neutral 的 DefinitionCatalog不重新注册或包装通用Fit；当前Workbench只投影一个本地`Add Analysis -> Fit` capability。

当前有两个挣得起的hosting路径：interactive/offline adapter把已冻结snapshot交给同一个BoundFit；需要durable领域artifact的typed request从已提交输入编译一个flat RunPlan并拥有自己的一次FinalCommit。二者都不是StreamProcessor，也不通过伪造“累计DataBlock event”接入sample stream。

`DatasetInputSlot -> AnalysisStep -> post-materialization pipeline`明确是**延后设计**，不是baseline。重开它必须先给出至少一个真实自动/headless或下游consumer、其失败/cancel语义以及artifact原子性需求。默认优先选择“输入FINAL artifact -> 独立flat analysis Run”，因为它复用immutable replay边界且不改scan提交；只有明确要求scan+analysis不可分割成功时，才设计同一FinalCommit可恢复的composite result。不得先建`BoundOperation Protocol`、Analysis registry、descriptor hierarchy、program DSL或child Run。

### 10.6 Pipeline composition

TaskConsole 中的 Measurement/Processor 连接先形成 immutable dataflow，不让节点自行订阅或开线程：

```text
PipelineSpec:
  bound_measurement_sources
  bound_stream_processors
  typed event edges
  dataset_materializers
  artifact/result sinks
  delivery/QoS per edge
  criticality = REQUIRED | BEST_EFFORT_MONITOR

compile_pipeline(spec, immutable bindings) -> RunPlan[PipelineResult]
```

编译阶段完成：DefinitionKey/descriptor binding、event payload/schema/axis 校验、无环校验、ResourceClaims 并集、exact/monitor edge 分类、buffer budget、JoinPolicy、DatasetBuilder key coverage、criticality 和 terminal propagation。当前不支持 feedback data cycle；需要反馈控制时使用 revisioned ControlTopic，不把回路伪装成数据边。

REQUIRED source/processor/sink failure 使整个 Run 失败。BEST_EFFORT_MONITOR 只能出现在 monitor-only 叶子分支；其失败产生 panel/branch error、missed telemetry 和 Run diagnostic warning，但不反向终止仍健康的 formal exact run。required outputs 完整时 Run 仍是 SUCCEEDED，但 Result/Event snapshot 含 structured warnings，且失败 panel 不能标成成功。Compiler 禁止 BEST_EFFORT_MONITOR 输出再连接 exact、fit authority、calibration 或 artifact，也不自动 restart 失败 branch。

编译结果仍是一个扁平顶层 RunPlan：一个 RunHandle 依次拥有 online acquisition graph、DatasetBuilder finalization、artifact commit 和 cleanup。节点不能嵌套 start Run、动态新增边或各自成为 terminal-state owner。PipelineSpec 是静态阶段合同，不是 child-plan workflow DAG。未来真实consumer若挣得post-materialization Analysis，也必须先解决同一commit结果与ambiguous recovery，而不能把callback塞到finalize之后假装原子。

F0/S1 的首个 compiler 只接受 `1 BoundMeasurement -> 1 DatasetMaterializerSpec -> opaque in-memory PipelineResult`：没有 processor、analysis、feedback、持久sink callback、可选 child 或通用 node/edge DSL。它在 `RunController.start()` 取得 claim 之前完成 DefinitionKey、FrozenCaptureSpec owner、payload/adapter/schema、完整 cell permutation 与保守 event/byte budget 校验；RunPlan.preflight只用真实run_id创建software TraceBinding、CaptureSession、唯一exact reservation、cursor和DatasetBuilder，不发送任何device command；execute在prepared state完整返回后才prepare/start。CaptureSession自己从冻结`expected_cells[source_ordinal]`派生join key，不接受execute层传入另一个key；只有该reservation已经ACTIVE且持有绑定schema/adapter/完整schedule的ExactDatasetReadiness后，start才可触达设备。

`BoundCapturePort`只接受DeviceBroker针对当前BoundDevice/binding/generation执行endpoint-owned capability probe后mint的opaque attestation，不能把普通`CaptureCapabilitySnapshot`拼到真实设备上；probe全程持有broker probing token并与Run open、binding invalidation、recovery互斥，跨过任何activity epoch的结果不得发布。FrozenCaptureSpec在进入runtime前已由领域owner生成canonical bytes，runtime自行重算SHA-256并要求definition/contract/capability/spec owner fingerprint四者一致，prepare阶段没有替换或回调入口。CaptureSession创建线程就是其owner I/O lane，prepare/start/read/complete/termination/cleanup跨线程调用一律拒绝。普通整数、字符串或任意格式正确的digest不构成物理证明；正常terminal必须同时核对generation、spec/settings/capability binding、全部source ordinal、produced/drained、ordered metadata digest、source stopped、no-more-frame和真实join，才可mint不可伪造的CaptureCompletion。取消后普通execution capability会被撤销，因此BoundCapturePort必须提供thread-safe ABORT/DISARM与有限blocking-call bound；该bound写入每个prepare/start/read/complete/session-close command，adapter必须把它交给SDK wait/poll或自己的有界等待，不能只把它留作描述字段。cleanup phase发送绑定本session的`SessionCloseCommand`；wrong-session、stop/drain/join未知或超时都返回UNSAFE/quarantine，不能靠safe-state布尔值跳过join。formal compiler只消费该session拥有的CaptureCompletion，再取其中EOS交给DatasetBuilder seal，并交叉验证sealed artifact与terminal的metadata fingerprint/digest；PipelineResult由compiler私有authority mint并再次核对coverage/count/digest，调用方不能拼接另一个terminal伪造成功。裸EOS不构成pipeline成功。DatasetBuilder是exact reservation teardown的唯一owner：success seal、preflight/execute/cancel失败都在独立finally中close，最终reservation必须RELEASED且registry为空，前一步cleanup失败不能阻止它。未来post-safety persistent sink只接受storage-owned staged FinalCommit，不接受“任意 callback + requires_commit bool”。后续S3/S4只为已存在的processor或领域typed Analysis扩展静态合同；generic post-materialization Analysis仍受§10.5 consumer/commit门槛，不把这个最小直线偷偷演化成递归工作流引擎。

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

`mean`、`sum`、`integrate` 是不同物理 reduction，不能编码成一个含义模糊的 reducer。通用 `mean/sum/min/max` 使用 zlc_data 中封闭的 current reducer 合同，并由用户/analysis spec 显式选择；ROI photon count、相机畸变校正等带设备/物理含义的操作由 neutral 领域 StreamProcessor/Analysis 定义，不能因输入恰好是 image 就由 frontend 自动提出。普通 image 默认只能显示、选择或保留 spatial axes。

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
7. schema fingerprint 不匹配时提交失效，重新建议或要求修正，不能按 axis index 迁移；
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
  numeric_policy: evaluation/batch/per-cell-sample/total-packed/covariance budgets

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

initializer 只提供有限 seed，不拥有 hard bound。唯一 hard bound 来源是参数数学 domain（例如 positive、nonnegative、phase 主值区间）和用户显式 constraint；data range、选区 span、观测 contrast 等启发式绝不能变成无法扩宽的物理边界。全部参数都已有 fixed 或 explicit initial 时，执行直接使用 caller seed，不调用 data-derived initializer。最少 observation 也不是 model catalog 常量：每个 bound request 按 `max(2, free_parameter_count + 1)` 派生，固定参数是 caller 提供的 hypothesis，不再要求数据重新识别它。时间模型可用 selection window 改善 seed，但 artifact 参数仍保持 absolute-coordinate 定义。自由约束与静态 domain 必须有至少一个可表示的内部浮点值；phase 使用唯一主值表示 `[-π, π)`，不能同时保存 `+π/-π` 两个等价 artifact。资源合同同时限制每个 batch cell 的抽样数和整个 FitProblem 的 packed observation 总数；默认总量 2,000,000，在当前最多二维独立变量下把 observation + coordinate 主体限制在约 48 MB。builder 必须在 append/concatenate 大数组前累计拒绝超限请求，FitProblem 构造器验证最终 packed shape；已知有更大内存预算的调用方可在 FitSpec 中显式提高该值，不能让 `max_batch_cells * sample_budget_per_batch` 隐式放大到十亿级观测。

`FitBatchStatus.CONVERGED` 只表示数值求解完成；generic fit core 不拥有实验域的“科学上可用”判据。结果保留参数、RSS/R²、observation/evaluation counts 与 covariance validity，供 UI 和领域 consumer 判断。active authoritative bound 会使 covariance 明确 invalid/canonical-zero，但不把已经收敛的数值结果伪装成失败。SNR、支持区间、alias prior、目标参数容差或“哪些 batch 必须通过”等 policy 由真正消费这些参数的领域 AnalysisSpec 拥有；baseline 不建立通用 `FitAcceptance`、reason 字段或 model-local quality-gate DSL。执行失败使用其它 typed status、非空 execution error与 canonical-zero 数值，不能把不可辨识或领域不接受伪装成 solver failure。

generic damped-sine 只拟合 catalog 定义的 `baseband_frequency` 数值，不宣称证明无混叠，也不从 coordinate gap、shape 或 rank 猜 Nyquist。formal 物理频率 consumer 若需要无混叠结论，必须在领域 request 中持有采样设计与 band-limit prior（或提高硬件采样率）；软件不能从已 alias 的样本反推出“真实高频”。因此 FitProblem 不再持久或传播 sampling quantum/index-gcd 这类只服务一个推测性 acceptance gate 的字段。

packing 以 declared coordinate 为主排序、logical index 为 duplicate-coordinate tie-break，物理 storage permutation 不能改变入选观测。coordinate-less axis 使用 `index_origin + logical_index` 的 absolute coordinate；若 AxisSpec 声明 unit 就保留该 unit，否则参数 unit 才是 `index`。连续整数坐标必须在 bind 时证明每一点都能被 float64 精确区分，不能只验端点后让中间 x 静默重复。完整 coordinate admission 只在 `BoundFit` 绑定时执行一次并缓存每根 fit axis 的 source；packing 无条件消费该 proof，不能在每个 batch、FitProblem 或 property 中重新扫描 declared coordinate。`BoundFit` 只接收 FitSpec 与 expected DatasetSchema，effective schema/model 均在内部单次派生；package-private packing/solver 只接受 exact BoundFit，不能把可覆写 `__post_init__` 的普通子类当成已验证 proof。TransformedSchema 的 canonical fingerprint 在同一 immutable 实例首次需要时计算一次并缓存；identity bind 不为未消费的 digest 付出 O(P) 成本。Selection 后重复选择仍保留 absolute coordinate，不重基。有限 sample budget 使用确定性 Cartesian preferred grid 加小比例 value-feature 候选；max/min 与局部邻域公平交错，剩余额度再按 canonical rank chunk-stream 填充。valid NaN/Inf 必须优先进入样本并使数值路径 fail closed，invalid nonfinite 不进入。dense qCMOS image 路径不构造全帧 rank/value 副本；sparse point axis 的坐标 gather 与 canonical row order 只按 present physical rows 分配，绝不能先建立 logical-size coordinate array。当前 compressed irregular 2D point-layout 的 preferred-grid 仍允许一次 O(N) `np.unique` 工作数组，因为它不是 dense camera 主路径，后续只有在 profile 证明它成为瓶颈时才替换，不能为假设风险再造索引框架。

时间模型继承当前真机验证过的 absolute-coordinate 语义：decay amplitude 仍表示 x=0 的幅度，damped-sine phase 仍相对 absolute x；Selection/CommittedTransform 只筛选观测，不偷偷用选区最小值重定相位或幅度。若未来确有“从选区起点计时”的物理需求，必须使用显式权威坐标变换或新的描述性 model id。`FitResultBatch.evaluate_batch()` 是 overlay/replay 的唯一结果求值入口，使用相同 absolute coordinates 与 catalog evaluator。damped-sine 将 amplitude 约束为非负、phase 约束在主值区间，消除 `(A, φ) == (-A, φ+π)` 的 artifact 歧义。

领域中立的一维数学模型接受具名有限数值轴，包括 scan、spectral、spatial lineout 与 `histogram-bin`；axis role 用于 UI 推荐，不能让迁移后的 histogram/lineout 能力消失。二维 radial Gaussian 仍严格要求 spatial-x/spatial-y、共同 unit/frame，并把第三参数明确命名为 `one_over_e_radius`。原 neutral 层“Zeeman”标签对应的通用数学模型在 zlc_data 中命名为 `symmetric_lorentzian_doublet`；neutral UI 可以显示领域标签，但不得让通用公式冒充所有 Zeeman 物理选择规则。

DataTransform 后仍存活的每根 axis 必须恰好属于 fit、batch 或模型明确声明的 observation component；不能留给 solver 猜。FitModel 从首版显式声明 independent-variable arity/roles，支持当前已有的 1D 与 2D model；不能把 2D Gaussian 当作未来功能删除，也不能通过数组 rank 推断 arity。

WorkbenchFitRequest 是 workbench Command DTO，可持 app-local LiveDataBlockRef；FigureFitRequest 是 frontend figure DTO，只持 DatasetId。各 adapter 先解析为 OwnedSnapshot，再调用公开的 `bind_fit(FitSpec, snapshot.block.schema).run(snapshot)`；package-private `build_fit_problem(bound, snapshot)` 只在 `BoundFit.run` 内负责 packing。zlc_data 不定义 universal InputRef/FitRequest，也不看到 neutral live ref。artifact 保存 FitSpec 与已解析 input lineage，不保存 application request DTO。

batch cell 独立执行；预先列举的数值初始化/solver/evaluation-limit/浮点或线性代数失败只使该格产生 typed status，某格失败不破坏其它格结果。输入整体 schema/model 不兼容、transform 无效、host cancellation/deadline，以及 AssertionError/TypeError/MemoryError 等未知实现或资源异常必须中止整个 Fit Analysis，不能被 broad `except Exception` 伪装成单格 solver failure。wall-clock deadline/cancel 是一次 `BoundFit.run()` 的 hosting lifecycle，不写进 FitSpec、不产生 per-cell `TIMEOUT` 状态。FitResultBatch 不包含 runtime EventRef、LiveDataBlockRef 或 ArtifactRef；formal Analysis/figure repository adapter 在外层附加 input lineage。它不拆成多个 scalar signal，overlay 从同一个 result 与外层 lineage 派生。

FitResultBatch 是 compact solver-issued report，不是把原始坐标、observation、Jacobian 重复塞进去的 proof-carrying result。构造器/strict codec 验证状态机、静态 domain/constraint、计数、RSS、R²、covariance 的有限性/对称/PSD/fixed-row与 canonical zero；RMSE、effective schema fingerprint、coordinate source、parameter schema/unit 都由已保存的 FitSpec/AxisSpec/catalog 唯一派生，不在 payload 再存第二份真相。raw codec decode 只能得到 untrusted report；public direct 保存只接受 `FitResultRepository.execute_capture/execute_scan()` 铸造的 process-local `FitExecution`，load 在对应Capture/Scan source re-admission与binding校验后铸造不可replace/pickle/直接构造的`AdmittedFitResult`。execution/admission capability和repository均final、slotted、普通赋值不可变；每次操作复核root lease与content-store authority。outer manifest只含current format、repository id、owner编码的closed `CaptureArtifactRef | ScanArtifactRef` source与result ContentRef；CAS digest是唯一payload identity。execute_capture物化admitted CaptureFrameSource，execute_scan委托ScanRepository exact materialization，随后都进入同一个BoundFit与packed-observation预算；load只读取source FINAL metadata/revision/schema并复算fit/batch/layout/present-count binding，不重跑solver、不从parameters反推历史执行，也不读取source data blob。save/load在任何result codec或CAS allocation前按encode/decode additional-peak公式做有界preflight；notebook省略save预算时使用repository安装级默认，GUI传其当前front后的精确剩余预算。manifest/result blob另有固定byte上限，producer signature/journal仍无真实consumer而不预建。这里的信任边界是OS/process root lease排他的本地writer加CAS内容完整性；若外部主体可绕过API直接改filesystem，则没有密钥/签名的本地artifact都必须按untrusted repository处理。

FitResultBatch 是当前一等需求，不延后：gridplot、site grid 和任何保留 site/component axis 的 fit 都要求“一组共享 model/parameter schema + 按具名 batch axes 排列的每格结果”。`BatchLayout` 复用 PointLayout 的 RECT_C/RECT_F/EXPLICIT 映射思想；稀疏 batch 只保存实际 B 个 cell，missing coordinate 与 fit failure 是不同状态，不能强行 densify 后混成 NaN。grid 的 cell label/coordinate 由 batch_axis_specs + BatchLayout/axis coordinates 派生，不能用 list index 充当永久 identity。ComponentValidity 在 build_fit_problem 时按 batch cell 切片；某个 site 无效只使对应 per_batch_status 失败，不污染其它 cell，也不允许先对 site 轴平均成一个 FitResult。`build_fit_problem` 是 fit densify/packing 的唯一 owner；若某 solver 只接受 dense layout，它必须显式 materialize mapping+validity或在 bind 时拒绝，不由 renderer/collector 猜 reshape。

BoundFit 对 batch cell 使用确定性迭代顺序，并在 packing chunk、cell 边界和 model evaluation 间检查 host cancellation/deadline；单次 solver call 只有确定性的 max-evaluation/memory/sample budget。取消或 deadline 使整个 formal Analysis 失败且不提交成功 artifact；interactive stale result 按 DatasetRevisionRef 丢弃。该最小 seam 不引入 workflow engine。

### 11.8 权威 Fit Analysis 路径

当前已经实现且有真实consumer的路径是：

```text
FINAL CaptureArtifactRef | ScanArtifactRef
-> owner repository inspect/materialize exact DatasetRevisionRef + schema
-> zlc_data.bind_fit(FitSpec, expected schema) -> BoundFit
-> resolve FitProblem
-> BoundFit.run(exact OwnedSnapshot) exactly once
-> FitResultBatch / FitExecution
-> explicit FitResultRepository.save -> FitResultArtifactRef
```

FitSpec 必须包含 input_schema_fingerprint 与显式 fit/batch axes；发生选择/降维时 committed_transform 必须存在，identity path 可以为空。Fit只消费已经通过EOS/key/validity coverage并提交的immutable revision，验证schema fingerprint后执行相同的zlc_data transform/reduction/fit函数；它不在每个sample/update到达时把累计DataBlock重新拟合一遍。TaskConsole的当前入口也必须先取得当前card的精确FINAL ScanArtifactRef，再进入同一路径。

若未来真实consumer要求无人工操作的formal Fit，优先把该FINAL artifact作为独立flat analysis Run的输入，并让Fit artifact由该Run自己的一次FinalCommit提交。当前Scan Run只有一个ScanRepository FinalCommit；不得在其commit后直接保存第二artifact却仍宣称原子，也不得在recovery时重跑solver。只有确有“scan与fit必须共同成功/共同恢复”的领域consumer时，才引入可持久恢复的composite commit/result；在此之前不实现DatasetInputSlot或AnalysisStep。

Fit返回完整FitResultBatch；下游参数引用、校准更新、scan决策或“成功物理结论”必须在自己的领域AnalysisSpec中解释numeric status、covariance/counts/RSS/R²并声明哪些具名batch必须满足何种物理policy。generic fit core只报告数值事实，不能替领域consumer发明统一acceptance。

### 11.9 Interactive Fit 路径

```text
Plot card AnalysisCommand[WorkbenchFitRequest]
-> bounded Fit executor
-> resolve immutable input revision + FitProblem
-> 同一个 zlc_data fit program
-> FitResultBatch
-> revision-checked overlay/ViewModel
```

interactive Fit 使用 workbench application adapter 提供的独立 bounded Fit executor；frontend.figure 只拥有 Figure DTO、View 求值和 overlay 投影，不成为 executor/lifecycle owner。同一 panel 的 stale queued request 可 coalesce、已运行的不可中断 solver 返回后按 revision 丢弃。它执行 zlc_data `bind_fit` 产生的同一个 BoundFit，不创建隐藏 StreamProcessor node、不发布正式 measurement signal，也不占用 exact `StreamProcessorWorker` 或 view-evaluation 队列。用户要让fit result进入下游权威流程，必须显式保存FitResultArtifact并由下游typed request引用；未来若出现自动analysis consumer，再按§10.5建立独立flat Run，不能用隐藏AnalysisStep升级当前交互动作。

interactive 只意味着 QoS/入口不同，不降低输入 integrity：若输入 DatasetRevisionRef 属于 Formal epoch 且仍为 PROVISIONAL，可以为即时观察运行临时 fit，但 overlay 必须带 `PROVISIONAL` 标记且不能保存为`FitResultArtifact`、不能成为后续 authority input。epoch 转 INVALID 时相关queued/running result按epoch lifetime token丢弃并从正常overlay撤销；只有独立 EpochValidationRecord 证明该revision为 VALID 后，才允许 materialize为正式派生结果。

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

纯算法只有一层命名：`zlc_data.apply_transform`、`reduce_data`、private `build_fit_problem`、`bind_fit` 与唯一公开执行入口 `BoundFit.run()`。interactive、offline/artifact以及未来挣得的formal路径都执行同一个BoundFit；neutral当前不定义generic AnalysisStep或任何Fit-named class。`OccupancyStreamProcessorDefinition` 属于 neutral_atom，因为它包含逐帧领域物理语义；Calibration/ReadoutFidelity等完整dataset/artifact算法的typed request与compiler属于neutral_atom，因为它们承担领域物理判断。简单的、无领域语义的逐event变换若确需进入在线图，可由zlc_data提供纯函数，neutral pipeline在composition时绑定为普通StreamProcessor operator，但不复制实现。

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

Interactive live path 在 per-panel latest-only view-evaluation executor 运行 FigureEvaluator：直接解析 ViewSpec 的 axis bindings/navigation policy，再执行 display transform/reduction/layer data 计算，产生带 document/input revision 与 resolution records 的 immutable EvaluatedFigureData；具体 surface ownership 见 §12.5。Workbench 的 live/headless export job 在 worker 中永久独占自己的 Figure。冻结的 notebook DataFigure 是不同的同步 one-shot surface：构造时一次 evaluate 并释放 source snapshot，每次 `render/export/_repr_png_` 在调用线程用 OO Agg 新建一个 caller-owned Figure；它没有 live mutation、共享 artist 或第二个scheduler，因此不为DataFigure本身预建render lane。W4a窗口只是把既有`.figure()`和一次`to_png_bytes()`作为同一个module-owned capacity-one worker job托管，使这个“调用线程”不是Qt owner；job结束即释放DataFigure/Agg，只留下immutable PNG front，不把one-shot viewer提升为live renderer。所有路径都只执行 document 已决定的 ViewSpec，不重新猜 axis；live/persisted binding 的保存规则见 §16.3。

Calibration report不是普通dataset view：它同时展示runtime artifact、statistical report、component validity、threshold provenance与empirical PSF diagnostics，强塞进DataFigure会要求伪造point/data axes并丢失领域关系。W4b因此只复用相同的frozen-raster hosting，不复用DataFigure语义：neutral owner成对校验artifact/report并给出阈值来源与GridOrder物理位置；Workbench composition投影成frontend-owned immutable report view；frontend renderer只画stored facts。`EncodedRasterDocument/Page`是DataFigure与Calibration两个真实consumer共用的唯一worker→Qt DTO，shared shell拥有唯一capacity-one executor、取消/reap、atomic multi-page present和Qt front预算；它不是第二renderer或通用workflow engine。

Occupancy artifact则是普通dataset view，但同一Figure一次只绑定其一个真实输出块。composition通过显式`occupancy_output="occupied"|"counts"`选择artifact已经持久化的exact snapshot，并让FigureDocument descriptor、ResolvedDatasetMap和OwnedSnapshot ref都指向该块的原始schema/revision/generation；不能把两个块堆成伪COMPONENT轴、伪造第三个DataBlock，或回退到source capture冒充occupancy lineage。occupied/counts共享repeat/point/layout/SITE domain与逐SITE ComponentValidity，但仍是两个不同dtype/unit的dataset；frontend只做display投影。该冻结Figure没有Calibration SiteMap的物理XY/GridOrder证据，因此只显示canonical SITE index/facet，不宣称physical grid、paired calibration overlay或fit authority。

精确物理cell overlay则明确不是普通单dataset Figure：它需要同一个occupancy cell、source capture cell与Calibration SiteMap三方事实，但不产生新的权威join artifact。notebook composition先用具名Selection解析唯一`DatasetCellAddress`，用该地址分别取得occupied/ComponentValidity和同shot frame，再投影成frontend-owned自包含view；renderer看不到neutral ref/repository，neutral也不导入frontend。该view进入interactive exact SiteMap，Qt直接消费immutable INDEXED8 background与三态site facts；A/C/Z/H、hover、clim和Setting/Edit只改变display state。矩形仍只能生成同一exact address/revision与ViewportTransform上的`DISPLAY_ONLY` SelectionCandidate，不能保存成Fit/Scan/Calibration输入或把当前画面升级为权威选择。

FigureDocument/FigureEvaluator/codec 属于 headless `frontend.figure`；DataFigure 因拥有 renderer/surface/export convenience，属于可选 `frontend.render` surface，只有真正调用 render/export 时才惰性导入 Matplotlib。neutral_atom 只返回领域 Result/ArtifactRef；notebook/workbench projector 把它映射为 FigureDocument，neutral_atom 不导入 figure 或 DataFigure。DataFigure 只接收 FigureDocument、ResolvedDatasetMap 与按 layer id 绑定的 data-owned FitResultBatch，不主动访问 Hub、Task、Session、PulseDocument、repository、ArtifactRef 或 Device。

Figure render可以显示 PROVISIONAL revision，但必须在所有surface持续显示不可被theme/overlay隐藏的状态徽标；普通 Figure Save/Export在输入epoch未VALID时拒绝。唯一例外是用户显式选择“保存诊断快照”，生成`DIAGNOSTIC_PROVISIONAL` artifact并把水印、epoch id、revision与当前状态固化进pixels/manifest；它不能被 FigureArtifact 或任何 source-specific authoritative fit-artifact loader 当作权威输入。epoch INVALID 后，LiveFigureBinding提升lifetime token并清除或标红旧front buffer，避免之前排队的正常BoardFrame覆盖失败状态。

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

领域感知的 TaskConsole、PulseGUI、DeviceViewer/DeviceManager shell/controller全部属于`zlc_workbench`，不是`zlc_frontend`。`zlc_frontend`只保留通用Figure/render/selector与纯widget/presenter；它接收workbench-owned、presentation-only ViewModel/widget props并发出UI command，不导入neutral/pulse类型，也不接收runtime port。Workbench controller的构造器仍必须显式列出真正需要的窄依赖，禁止接收整个Experiment、Session、DeviceSet或返回raw object的provider：

- Workbench TaskConsole controller接收definition/readout catalog projector、processor/plan factory、`RunCommandPort`、RunSnapshot reader与`open_device_viewer` action；LegacyRuntimeFence只由workbench composition持有并隐藏在RunCommandPort实现后，console/controller/widget都不保存fence、registry或真实node。公开running列表只返回`RunNodeInfo`/RunSnapshot DTO。
- Workbench PulseGUI controller的 editor 只接收 `PulseEditorSession` 与 pure preview function；`PulseEditorSession` 只拥有 current `PulseDocument/path/revision/disk baseline`，因此 offline authoring/preview 无需任何设备身份。online composition 才额外注入 immutable `PulseTargetDescriptor` 与已有 notebook/application `PulseFacade`，显式 rebind 若改变文档则相对真实磁盘 baseline 保持 dirty。这里不再额外造一层 `PulseCommandPort` wrapper：authority 已用 `PulseRunRequest -> PreparedPulseExecution(one-shot) -> RunHandle` 提供恰好所需的 run-once/hold/scan/start/cancel 面；再包装只会产生第二份状态和验证。纯compile/preview不触碰hardware；authority内部完成prepare/fire/session close/safe，且不暴露raw sequencer或可拆开的public prepare/fire/safe。standalone real mode必须由workbench composition注入同一authority，不能自行构造RemoteSequencer；未装配时只允许offline/virtual。
- Workbench DeviceViewer controller接收`DeviceCatalogReader`和只读status DTO；需要操作者控制时只注入具名、审计化的`DeviceControlPort`，不存在`editable=True`后直接调用raw setter。
- Workbench DeviceManager controller接收config document reader、catalog reader与`DeviceAdminPort`；它可以校验候选config、显示restart-required差异并请求safe shutdown，但不能在进程内Apply/Open/Swap physical graph，也不返回或缓存旧`DeviceSet`。

这些ports不是跨包万能Service。每个port的方法集合必须由单一UI use case挣得；它们接受/返回owner定义的immutable request/result。Workbench controller负责把neutral/pulse/installation对象投影成frontend ViewModel；frontend不复制领域DTO。Selection到neutral `ControlTopic`的转换由Workbench PanelController完成，frontend不导入neutral stream原语；设备role到BoundDevice的解析也只在composition/bind发生，GUI不保存resolver。

Workbench 大图像 ViewModel 使用 app-local LiveDataBlockRef/ReadOnlyArrayView 和 revision，不默认在每个 UI hop 再深拷贝。默认发布边界产生拥有自己内存的 immutable snapshot；若 driver 会复用 buffer，必须在该边界 copy，发布后 producer 不得再修改。该 live ref 经 LiveFigureBinding 解析，不泄漏进 frontend FigureDocument/codec。

baseline 的 `LiveFigureBinding.resolve(DatasetRevisionRef, SnapshotQuery) -> OwnedSnapshot` 只 materialize 当前 ViewSpec 所需 axis slice/chunks，不默认复制完整累计 DataBlock，也不返回 mutable builder alias。SnapshotQuery 只描述所需 slice/revision，显示 reduction 仍由 FigureEvaluator 拥有。只有 profiling 证明大帧发布 copy 是真实瓶颈、且某 adapter 明确提供可 pin 的零拷贝 buffer 时，才启用 opt-in `BorrowedSnapshot`：它把 read-only bytes 与 workbench-owned release token 绑定，token 只存在于 LiveFigureBinding/WorkbenchRenderMessage，frontend 类型和 artifact codec 永远看不到。worker 已产生不再 alias 的 layer/raster 后立即 release；若 front buffer 仍 alias borrow，则 front-buffer replacement、stale-result discard、queued-job cancellation、panel close和shutdown都必须在 `finally` 中 release。Save 先物化 owned bytes。该优化必须有 pin 上限、timeout/quarantine 与 shutdown drain 测试，不能成为所有数据的默认抽象。

### 12.3 Setting 与 Edit

统一的是四条互不混名的边界，而不是一个万能SettingsEngine：typed Request/Config owner的字段语义、§4.3.1的headless FormSpec与closed Qt handler、Workbench EditorSession的`base_revision + draft/apply/cancel`、以及最终领域validator/typed command。普通scalar、bool、enum、bounded number、unit value、简单path和静态list可由FormSpec自动生成；Setting与Edit必须消费同一个FormSpec和同一个committed state，但拥有各自widget instance与开始编辑时的base revision。

Definition只标识catalog项及其request/config schema identity；它不复制字段默认值，也不携带FormSpec、Qt hint或projector callback。字段的type、required、default、unit、range、static choices和semantic description属于typed Request/Config schema owner。Workbench为已知use case通过普通import调用明确的schema projector，把这些事实逐项投影成`zlc_frontend.form.FormSpec`，并只在这里增加group/order/label/widget/layout/file-dialog/dynamic-enable等presentation信息。frontend不认识Definition，领域包也不导入frontend；schema-id不用于动态查找builder。

以下继续使用显式presenter/view，不强行自动生成：Pulse editor与PulseDocument/API table、ROI/selector、fit axis/batch/reduction、CalibrationArtifactRef与calibration workflow、device connection、authoritative DataTransform、resource conflict和安全确认。显式presenter内部的普通叶字段仍优先取公共handler，但复杂对象的commit、rollback、validation和lifecycle不能交给generic form。`FluentParameterForm`不递归、不持久化、不做domain validation，也不拥有Apply按钮。

Apply必须先由field handler无损读出完整draft，再构造typed Request/Intent并通过领域validator，最后检查`base_revision == current_revision`后原子提交；后台或其它editor已更新配置时返回typed EditConflict，不能last-write-wins。Cancel只丢弃draft并从当前exact snapshot做full-state回填。UI的enable/disable只是提示，hidden/disabled字段仍必须在populate/reset中被覆盖；RunPlan.bind/preflight继续执行同一个权威validator，不能信任界面已经挡住非法输入。refresh或programmatic populate必须exception-safe block signals，既不能重新触发Apply，也不能吞掉非法saved value后保留旧widget状态。

axis 编辑器读取 ViewSuggestion，并只从其中的 ViewSpec 派生 display/x/sample/batch/select/reduce/facet 摘要，不让 image、rolling、histogram 和 fit 各自实现一套 shape 猜测。`RESOLVED` 建议无需弹窗；`REVIEW_REQUIRED` 在 panel 摘要中持续突出有损步骤；`NEEDS_INPUT` 才展开最小必要编辑器。

W3e 的 current TaskConsole 先兑现一个真实、封闭的 SCAN_SLOT 纵切，不把 legacy signal graph 换皮成新 workflow editor：composition 显式组装一个静态 Task、camera Measurement 与 occupancy StreamProcessor 三项 Definition，并要求每项都被 CatalogView 投影；UI 当前只显示可添加的 `Autonomous SCAN_SLOT` Task，source 编辑明确只有 direct camera 或专属 camera→occupancy-counts exact pipeline。任意 signal expression、全局 Hub 名称、stopped node 输出、monitor worker 和 arbitrary processor graph 都不进入正式 y。Definition 只提供稳定 key/schema metadata；动态 camera schema、processor contracts、operator 与 deadlines 在本次 bind 的 Bound 值中冻结。

Setting 与 Edit 使用同一个 `ScanIntentForm` 类和同一个 card-owned `ScanEditorSession`，但各自持有开始编辑时的 `base_revision`；Apply 先构造完整 `TaskConsoleScanIntent` 与 public scan request，再要求 revision 未过期且现有 panel 已完全 stopped/idle，最后原子替换同一个 `ScanPanelController` application。它不另建 Run 状态机、renderer 或第二个 panel owner。request-level/owner codec 约束在 Apply 时完成；依赖真实 calibration、device capability 与输出 schema 的完整 bind/preflight 仍只在 Start 的既有 worker 路径完成，失败保持 NOT FINAL，不能为了“Apply 即全验证”在 GUI 线程读 artifact 或接触硬件。Cancel 只恢复当前 applied revision；过期 Setting/Edit 返回 typed `ScanEditConflict`，不做 last-write-wins。`populate/reset` 必须是覆盖所有intent-owned控件的全状态函数，包括当前disabled/hidden的calibration/model、roles、trigger、budgets与SITE字段；occupancy→direct Apply/Load或首次Cancel后不得让已取消的隐藏值在切回occupancy时复活。

可保存 intent 只含稳定 DefinitionKey、owner `PulseDocument`、按声明顺序冻结的全部 whole-run API 常量、角色/trigger、显式 CalibrationArtifactRef、`model_kind=None`（跟随 immutable artifact default）或显式模型、权威 `DataTransformSpec`、独立 display-only `ScanDisplayIntent` 与预算/deadline。保存/加载委托各 owner 的 current codec，严格拒绝额外字段、旧 schema 和非 canonical bytes；不保存 DeviceRef、BoundDevice、RunHandle、reader/front buffer、provisional revision 或 legacy node。加载只能在 stopped/idle 时重配并清除旧 FINAL。任何非空权威 transform 必须在 Setting/Edit 中逐 operation、逐 AxisId 显示 Select/Reduce/missing/validity/min-count 语义，并提供明确清除 user-authored transform 的动作；不能把已保存的有损 authority 藏在 form 私有字段里。SITE auto/batch/select 只改变可见 View；底层 `(R,P,*data_shape)`、具名 axes、ComponentValidity 与权威 ScanOutputContract 完全不变。UI 必须把“Calibration default”保留为默认引用语义，不能打开编辑器后静默改成 BOX 或其它显式模型。

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

W4a冻结Figure窗口是一个**历史线程/内存所有权 checkpoint，而不是终态 UX 合同**：viewer module的唯一capacity-one lane先调用notebook提供的窄`.figure` callable，再用既有串行Agg owner合成整个board PNG；Qt owner在按当前物理board尺寸完成front预算准入后，只做一次`QtImageBoard.present_encoded()`原子换front。它当时没有持续source、latest/coalescing、overlay、ViewportTransform或交互selector；这只能说明该切片尚未完成，不能把缺失能力冻结成永久`DISPLAY ONLY`。§2.2 `UX-003` 与 §19 的纠正 3 要求通用 figure/archive 查看路径恢复 `main` 的 zoom/pan/re-fit/export；只有报告类多页可继续以 frozen raster 为数值呈现载体，并补 zoom。关闭仍须立即置cooperative cancellation并撤销发布，尚未开始及repository/Agg阶段之间的工作直接终止，已经进入不可安全强杀的repository/Agg调用则由共享lane诚实排空，GUI不等待且最终丢弃返回值，绝不发布stale front。

`WORKER_RASTER_LIVE` 是复合板的正式性能路径，不是把每个 panel 退回 GUI compose。worker 可一次 compose 整个 board 或一组共享布局，并通过 double/triple buffer 发布：

```text
BoardFrame:
  board_id + layout_generation + monotonically increasing sequence
  PanelFrame[]:
    panel_id + coherence_group
    SourceIdentity(dataset_id, block_id, stream_generation, schema_fingerprint)
    CoherenceStamp
    immutable RasterBuffer

CoherenceStamp:
  run_id/provenance_epoch_id
  join_key type/schema/digest
  EvaluatedInput(dataset_id -> exact DatasetRevisionRef)[]
  PanelPresentationIdentity(panel_id, document_id,
                            document_revision, selection_revision,
                            panel_revision)[]
```

同一 coherence group 的 panel 必须从同一个 causation domain 与完整 CoherenceStamp 求值，并在一次 GUI transaction 中 `present(BoardFrame)`；裸 `JoinKey==7` 或裸 `revision==5` 在不同 run/block/generation/schema 间不具有可比性。不能让 per-panel latest-only 各自成功后拼成看似同 shot 的 board。互相独立的 monitor 可以带不同 revision，但 BoardFrame 必须显式标出它们不是 coherent group。强像素级 coherence 使用一个 parent raster/front-buffer 原子换页；多个独立 QWidget surface 最多声明 model transaction coherent，不能声称 OS paint 同一时刻完成。后台慢时丢弃未开始的旧 board revision，不能逐 panel 呈现半新半旧状态。

U01交付这些值对象与**整板最终准入闸**；后续 finite-preview 窄切片才在其上接出一个 source、一个 DatasetId/layer、一个 panel/coherence group 的 `MonitorDatasetSnapshot -> FigureEvaluator -> pure GRAY8 -> BoardFrame`，并提供不从根包加载 Qt 的可选 `QtImageBoard/QtOwnerWake` leaf。owner thread仍先以 `BoardPublishPort.admit(sequence, group->exact stamp)` 冻结完整期望向量并签发一次性work token，worker只可用该token提交完全匹配的 `BoardFrame`；新admit、source invalidation、reconfigure、port replacement或close会使旧token失效，首次成功publish即消费token。admit与present都在owner thread线性化，worker只替换一个bounded pending whole-board mailbox。上游多 source/per-group join、tile cache、公平render lane与PanelHost仍由后续Workbench scheduler拥有，不能把这个单 panel coalescer误写成第二个scheduler。

M1是`WORKER_RASTER_LIVE`的第一个continuous单panel产品，不是第四种render模式。Run worker只负责camera record -> stream -> MonitorDataset；live render worker从同一原子snapshot冻结block/head/EventRefs/coverage，FigureEvaluator和GRAY8 raster都不在Qt线程，Qt只接受immutable BoardFrame front。coverage只在对应front成功publish时一起更新，不能把后来的missed计数贴到旧图。source failure或normal terminal都先使publish port失效再clear，因此已排队worker结果不能在撤源后复活旧front。capture与monitor复用同一个窄Qt Run-owner/mailbox管理Future、generation、cancel/reap和close顺序；finite capture的FINAL front保留、monitor Stop撤屏/重新prepare仍留在各产品controller，禁止抽成可配置“成功策略”或通用workflow窗口。

`BoardPresenter.present()`必须是old-or-new原子交换：失败要么在修改front前抛出，要么保留上一张完整front；controller随后进入sticky fault，UI在上一张完整图上覆盖明确错误状态，绝不呈现半张新board。controller持久保存的fault只能是string-only detached diagnostic，不能保存带`__traceback__/__context__`的原始异常而把失败栈中的snapshot/evaluated/raster长期钉住。`clear()`只在reconfigure、publish-port replacement或close的owner thread调用，并释放front raster引用。close在调用clear前先撤销publish/work authority；若clear失败则记录fault、保留同一presenter/front且close不宣称完成，后续close只重试该clear；组合controller也只能在底层clear成功后标记close complete，不能因自己已经停止新work而吞掉底层重试。只有clear成功才丢掉presenter与旧fault引用并使close幂等。普通present异常不把上一张有效图静默清空。

“普通 present 失败保留旧完整 front”与“source 已失效”分开：capture/preview source failure 通过同一个 no-payload owner wake 到达 controller，owner 先 `invalidate()` 撤销 publish/work authority再清空旧 front，防止失败 run 的最后一帧继续伪装为有效。当前 leaf 还没有 PROVISIONAL/FINAL 徽标或错误 overlay，所以只能保守清空；在这些状态 UI 接线前不能暴露为完成的产品 surface。

GUI 不读取 worker-owned Figure/artist；所有命中测试和选择都使用随 front buffer 一起发布的 ViewportTransform。静态 axes/labels/colorbar 可由 worker raster 缓存，动态 overlay 由 Qt 画；export 始终从 FigureDocument + frozen data revision 重画，不保存屏幕 texture。该交互不是可选 enhancement：每个用户可见 `WORKER_RASTER_LIVE` panel 必须按 `main` salvage 清单提供适用的 zoom/pan/crosshair/hover/selector，并使 Setting/Edit 中的 relim、cmap 与 limits 走同一 document/view revision；不适用的 plot kind 必须由旧行为证据明确，而不能返回空交互句柄冒充完成。

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

W3d 的单 CURVE progressive scan 是该规则的第一个窄实现；M2c 又让同一 owner 服务固定的 CURVE/HISTOGRAM/METER scalar panels，但仍不把它扩张为全局 renderer cache：`SinglePanelAggRenderer` 的每个实例只活在一个 preview/live worker 内，只接受一种冻结的单panel homogeneous topology，Figure/Canvas/Axes/artist 的构造、revision update 和 close 全在同一 worker thread。CURVE更新具名X与component-validity mask，HISTOGRAM用统一60-bin纯函数更新step vertices，METER原地更新最新值或明确`invalid`；三者对valid NaN/Inf都fail-closed，不把非有限值偷偷当invalid。GUI始终只接owned RGBA bytes。topology 漂移立即失败，provisional renderer断开Figure↔Canvas并显式收集Matplotlib artist cycle后才发布`worker_done`，FINAL projection必须再等board clear/worker release，不能复用provisional Figure或raster。close/render fault只使display branch失败并留下detached diagnostic，不改变exact Run；renderer budget仍覆盖persistent canvas、旧/新artist arrays、histogram counts/edges/vertices、candidate/front和terminal source freeze，没有因复用而下调。相同virtual `(R=2,P=2,site=35)` W3 profile中exact freeze不超过约`0.14 ms`、transform约`3.3 ms`、evaluate约`1.9 ms`，旧逐revision重建Agg的raster累计约`1.44 s`；worker-local persistent path的一次coalesced raster约`149 ms`，端到端由约`1.59–1.61 s`降至约`1.27 s`。Matplotlib cold import/font成本仍存在；本切片没有用全局预热、常驻跨scan Figure或新调度器掩盖它。

当前 console-wide RenderLoop/Matplotlib Figure 不能在 S1 前瞬间删除，所以 S0.5 使用三个互不冒充的迁移桥：`LegacyPanelHost/CatalogRouter` 托管并逐项隐藏旧 panel；`LegacyRuntimeFence` 让所有旧 LogicNode start/stop先取得同一 ResourceArbiter 的保守 ResourceClaims，真实 thread termination + safe ack 前不释放；`SerializedLegacyAggBridge` 只负责旧 Figure ownership handoff。旧 Figure 在 compose 期间由 render worker独占，GUI 只有在成功 handoff 后才可执行 allowlisted artist/selector 操作；非 GUI 线程调用 `draw_idle/update/resize/mpl_connect/mpl_disconnect/Qt selector state` 等 QObject-affine API 必须机械拒绝。若无法证明旧 Figure 已与 QTAgg/QWidget state 解耦，则 worker 必须使用独立 Agg clone/raster。handoff timeout 时禁用交互/延迟 teardown并显示错误，绝不能继续访问。三个 bridge 都只存在 workbench migration adapter；Z0 全部为 0。

因此终态仍禁止 worker 与 GUI 同时或无确认地访问同一个 Figure；这里接纳的是现有性能事实和明确的迁移 handoff，不是把双 owner/barrier 自死锁提升成架构合同。

底层 Qt leaf 在 W1 历史切片中只画一张 immutable GRAY8、保持声明的 `(Y,X)` 方向并做 aspect-fit；当时尚无 `ViewportTransform`、ROI/crosshair/selection/hover、zoom/pan、固定色阶 control、axes/colorbar、save/export、多 panel compose、多 source join或 continuous/free-running monitor。W1 的 one-event finite exact `CaptureWorkbenchWindow` 只证明 prepare/start/result/reap 与异步 Stop/Close 所有权；它不是终态交互 panel。所有仍缺的用户面必须按 §2.1/§2.2 与 §19 纠正 2/3 继续交付，不能因 leaf 曾局部 READY 而成为合法降级。

### 12.6 UI 可见 Fit

提供两个明确入口：

```text
Add Analysis -> Fit
Plot card -> Analyze -> Fit
```

两个入口现在都落在唯一`DataFigureWindow`。`Plot card -> Analyze -> Fit`由U0.3d交付；U0.3e的TaskConsole按钮只在当前card拥有精确FINAL `ScanArtifactRef`时启用，点击瞬间重新读取该ref并调用同一个`fit_gui(ref)` composition seam。重复点击同一ref只聚焦既有窗口；新Run或Load/reconfigure撤销旧入口，旧窗口仍只是自己冻结artifact的独立viewer。按钮不持FitSpec preset、不加入neutral DefinitionCatalog、不创建FitProcessor/AnalysisStep，也不把TaskConsole的SITE等display selection复制进FitSpec。只有用户在共享Fit pane中明确点击Fit才求解，只有明确Save才产生FitResultArtifactRef。

这不是把formal能力降格为GUI：当前真实consumer就是人对已提交ScanArtifact做交互分析与显式保存，artifact边界已经权威且可复现。自动/headless preset或下游consumer出现时按§10.5/§11.8另建以FINAL artifact为输入的flat analysis Run；不能为了保持“Add Analysis”这个按钮文字而预建DatasetInputSlot、generic Analysis registry或修改Scan的单FinalCommit语义。

Figure Fit 的 composition capability 只冻结 exact `CaptureArtifactRef | ScanArtifactRef`、source schema inspector、prepare/execute/save/reload 四个窄command与总预算；DataFigure不取得repository或`FitExecution`保存能力，TaskConsole也只通过既有Experiment facade发起开窗而不获得另一套执行接口。`FitDraftAuthority`是唯一未保存execution owner，Qt只持`FitDraftResult`。执行与overlay raster分别使用容量一的既有Figure lane和窄Fit lane，不建立async engine；revision/CAS规则保证旧prepare/solver/overlay/reload completion不能覆盖更新后的selector、约束或viewport。

UI 明确展示 input、fit axes、batch axes、selection authority、model、initial/bounds、status、result、save identity 和 overlay。`suggest_fit_draft(schema, model, fit_axis_ids, selection)`只从schema声明的轴role与当前typed panel的具名显示轴生成候选：repeat及所有非fit point/data轴默认完整保留为batch；显示层为得到单panel可以带标签选择一个真实physical cell，但这个display selection绝不能进入FitSpec。不存在按rank/singleton猜role、`flatten`、取第0个权威batch cell或对trailing axes `nanmean`。

用户看到的普通 `Fit` 动作就是提交当前权威draft，不再弹第二个确认框；但未解析真实fit axis时按钮不可执行。1D range与2D box由Qt selector产生`SelectionCandidate`，经exact panel origin/revision转换成只含fit axis的`Selection`并预填同一draft：前者只选择当前CURVE x轴，后者显式绑定`SPATIAL_X/SPATIAL_Y`，其它repeat/point/site/data轴继续为batch。`Use full range`仅移除这条authority Selection；Clear只撤当前draft/overlay并保留已发布ref及当前selector candidate。zoom/pan/relim/cmap永远是display-only，不能复制进CommittedTransform。

CURVE overlay 先建立不求值的plan并在任何`FitResultBatch.evaluate_batch()`或prediction allocation前计算aggregate peak；exact limit通过、少一字节在零batch求值时拒绝。replace/clear同时计入Qt仍持有的旧prediction数组，transient result只按retained事实计一次；每个series物化前检查取消。IMAGE radial overlay只携exact batch storage identity、center/radius及viewport映射，不生成第二张predicted image；失败/NOT_PRESENT只显示诊断，不伪造曲线或圆环。Save先按窗口剩余预算preflight，成功ref在线性化点先被接受，再从该exact ref reload；后续decode/render失败或Close都不能吞ref，Save中Close明确defer。headless notebook的`fit.save()`使用repository配置的有界默认，仍保持短cell。

saved-fit archive GridPlot不会因一页只有一个panel就暗选第一格：Overview、hole或未聚焦时Refit禁用，只有用户聚焦一个真实`FitGridCell`后才启用`Analyze -> Fit/Refit`。该cell Selection只决定新DataFigure的display panel；重新author的FitSpec从exact saved ref恢复原model、constraints、numeric policy与range-preserving CommittedTransform，并绑定原Capture/Scan source。打开Analysis不求解，只有随后明确点击Fit才创建新draft；原artifact不可变。当前权威transform边界仍是一个range-preserving Selection，unsupported transform显式拒绝，不能用显示fallback解释。

### 12.7 Shutdown

关闭窗口或请求切换config/device/virtual-real都进入同一个显式、不可逆的 process shutdown 流程；它不在原进程内构造 replacement runtime：

```text
reject new commands
-> mark UI shutting_down
-> terminal-ack pending ControlTopic revisions
-> stop producers/subscriptions and reject new view/fit jobs
-> cancel queued latest-only work; drain in-flight view evaluation/raster work
-> deliver/discard final revision-checked GUI results
-> close interactive FigureSessions/renderers on GUI thread
-> verify/release all opt-in BorrowedSnapshot tokens
-> stop view-evaluation/Fit/raster workers
-> InstallationRuntime atomically RUNNING -> CLOSING; reject public commands, resolver reads and all new Run/recovery admission
-> RunController owner-shutdown: cancel all RunHandles, join owner+interrupt threads
-> wait until every Run safety bundle is durable and all Run ResourceClaims are released
-> finish each already-started RecoveryAttempt explicitly:
   complete, or abort only when no journal-acknowledgement ambiguity exists;
   ambiguous journal result retries the same attempt/bundle or keeps shutdown FAILED_CLOSED
-> gate/wait DeviceBroker identity/capability probes and session authority
-> DeviceBroker owner-shutdown: irreversibly invalidate every binding/capability
-> on each owner lane close the one InstallationDeviceGraph in deterministic reverse dependency order
   (startup failure closes the exact successfully-opened prefix/subset owned by the composition builder)
-> after every adapter confirms closed, stop adapter lanes
-> ResourceArbiter owner-shutdown releases PersistentSafetyJournal owner lock
-> release backend/process physical-owner proof
-> InstallationRuntime -> CLOSED
-> destroy Qt views
-> process exits; a launcher may now start a new runtime
```

上述顺序是authority边界，不由GUI线程是否仍响应来证明。ResourceArbiter、DeviceBroker与RunController在被composition绑定后，child public `shutdown()`必须拒绝；只有InstallationRuntime私有lifecycle capability能推进terminal teardown。broker必须先于raw close失效，journal owner lock必须晚于最后一个raw close acknowledgement释放，physical-owner proof最后释放。不得持composition lifecycle lock跨SDK close、journal unlock或其它物理I/O，等待者必须能按自己的monotonic deadline返回。

graph membership 在启动完成后不再增长，因此shutdown不需要“current/retained graph union”、ownership transfer或replacement context。adapter close失败时，runtime仍强持有同一个graph、原handle、owner lane与journal authority供同一次shutdown幂等重试；失败对象不能从close manifest消失。存在join timeout、RecoveryAttempt journal歧义、raw close未确认或journal owner close失败时，`shutdown()`不能假装完成，runtime保持CLOSING或sticky `FAILED_CLOSED`、所有admission关闭并在后续调用重报同一原因。UI只显示继续等待、重试或明确强制进程退出；强制退出不是clean shutdown成功。持久quarantine本身不阻止干净进程关闭——它跨重启继续阻止下一次普通admission。queued result通过application lifetime token + run/panel revision双重检查后丢弃，不能更新已销毁或id被复用的新panel。只有runtime达到CLOSED并且旧进程退出后，外部launcher才可使用改变后的config启动新进程。

## 13. Calibration

Calibration 是 `neutral_atom.readout.calibration` 的内建 feature，不使用 plugin、entry point、包扫描或动态 registry 覆盖。

`zlc_neutral_atom.readout` 包根不重导出 contracts、codec、analysis、repository 的宽 API；调用方必须从语义 owner 子模块导入。追溯整改实测旧包根的 122 个 eager re-export 会让单独导入 `readout.contracts` 也加载 SciPy，冷导入约 0.86 s、tracemalloc 峰值约 51.8 MiB。删除包根聚合后，同一探针不再加载 SciPy（当前机器约 0.35 s、10 MiB，主要为 NumPy/zlc_data 基础值）。这不是 lazy `__getattr__` 兼容表：本项目没有需要维护的旧公共格式，重复出口清单只会形成第二个 owner；稳定公共用户面由 notebook/workbench facade 组合，领域实现直接依赖 leaf owner。

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

Readout contract 的公开序列化面只包含由真实 Capture/Calibration artifact 静态调用的 owner `to_tree/from_tree`；它们不是独立文件、wire union或repository，因此不各自拥有 standalone bytes codec、nested schema discriminator、格式常量或异常层。外层 artifact 的 `format/schema`、canonical decode、完整 typed reconstruction 与全 payload re-encode 是唯一 durable canonical admission；nested parser 只核 exact field set、委托 foreign owner subtree并构造领域类型，领域不变量只由各 contract 构造器验证。`CameraEventReadoutSetting` 的 tree 函数是 descriptor codec 私有实现，不能扩成第二套公共 API。`FrameContract` 同样没有 digest/fingerprint：运行时适用性用结构相等，持久内容身份只归外层 codec bytes/CAS `ContentRef`。本项目没有部署过这些被删除的伪独立格式，因此不保留 tag、兼容 reader 或转换器。

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

Camera geometry、ROI/binning 整除、output shape、spatial axis 与 real-count dtype 的纯规则只由 `zlc_neutral_atom.readout.contracts` 拥有；`CameraCaptureDescriptor`、`FrameContract` 与 runtime `CameraPhysicalFacts` 各自仍验证本对象的完整合同，但必须委托同一组规则，不能复制第二套物理公式。

Calibration layout owner 是 READOUT_EVENT 与全部其它 named logical context 做稀疏 join 的唯一实现；它返回包内 `_CalibrationCaptureJoin`，只保存 point-context 与按 `(reference events..., readout event)` 排列的 physical rows，repeat 只在取帧/生成 report context 时惰性展开。它不是公开 DTO、持久格式或缓存层。FrameContract 先完成廉价的 descriptor/AxisId/schema admission，再调用 layout owner；任一 selected event 缺失或 context 不成套都 fail-closed。formal preflight先用FINAL inspection完成限额拒绝，再完整admit source并解析一次`_ResolvedCalibrationSource(source binding + FrameContract + physical context + join)`；execute只消费这份prepared resolution。`CalibrationAnalysisResult`同时保留exact `AdmittedCapture`与同一resolution，final commit只复核process-local token、artifact与resolution字段、以及join对report contexts的匹配，不重新decode capture、不重建physical index，也不产生第二份join。整个flat RunPlan从preflight到finalize持有capture与calibration repository root borrow；close要么在preflight前获胜，要么明确失败并等待该run释放，不能在lazy frame读取或CAS staging中途使authority失效。不存在只为测试服务的`from_schema`、raw-row diagnostic list、公开bracket type或witnessed-layout wrapper。Workbench mint descriptor后由紧邻的`CaptureStreamContract`构造边界完成一次schema admission，mint helper不提前做相同全表校验。

所有会改变数值解释的采集设置都进入 `FrameContract`；artifact 构造时一次验证 SiteMap coordinate frame、site coordinates、feature boxes 和 model axes 与该合同一致。公共 notebook/API application 必须提交 `CalibrationArtifact + 当前 FrameContract + frame`，因此相同 shape 但 exposure、ROI origin、optical path 或 camera identity 不同的 frame 也会拒绝；裸 `ReadoutModel + frame` operator 是 package 内部的已绑定 hot path，不是 public API。Occupancy bind 再把完整 capture `FrameContract` 与 artifact 比较一次，因为这是“当前物理输入适用性”而不是重复验证 artifact 内部结构；绑定后 hot path 信任该事实，不再逐帧复制 frame/site/model digest 或 FrameContract 检查。

`FrameContract` 只回答 camera 如何解释一帧，不能单独证明该帧曝光时原子装置经历了同样的 pulse 条件。因此权威 calibration 还必须保存由 **CaptureArtifact 中已经冻结的 camera physical facts 与 pulse lineage** 派生的 `ReadoutPhysicalContext`；调用者不能提交一个自报 context/digest 来给自己作证。context 绑定 pulse-owned `target_abi_fingerprint`，使 raw lane、logical port 到 lane 的映射、DAC bus index/width/encoding/safe value 与 latch clock 任一改变都会拒绝旧 calibration；它不是 whole-artifact fingerprint，也不把无关 pulse 编译细节误当成适用性。

每个 readout event 先把已经包含 channel delay 的物理 trigger 上升沿作为时间锚，再用 camera-qualified integration-start offset 和实际 exposure 得到严格半开窗口 `[start, end)`。当前实现只有 nullable scalar offset，**尚不存在**名为 `CommonFrameAperture` 的类型级证明；该名字只描述 Q0 以后才能发布的能力。开放 scalar 权威路径前，E0/Q0 必须对具体 sensor mode、applied global-exposure mode、ROI 与 readout speed 证明全部输出像素共享同一个 integration aperture，并由届时的 typed capability 承载；只读取到一个 global-exposure 枚举值或非空 scalar 不构成证明。若 qCMOS 实际是 rolling/per-row aperture，当前 scalar capability 必须继续为 `None`/NO-GO；必须先引入与 spatial-y/component axis 对齐的 typed aperture model并逐 component 派生适用性，禁止拿平均行、首行或一个经验 offset 代表整帧。

EDGE trigger 下 trigger 只负责锚定，trigger high width/下降沿不作为被测物理条件；context 收集窗口起点的完整状态以及窗口内所有其它 logical digital output 和 decoded DAC value transition。窗口起点 transition 进入初态，恰在 `end` 的 transition 不进入本帧。有限 pulse 在 DONE 时的真实 bus safe 行为同样属于物理 waveform：RTL 在 DONE 边界清除 undelayed bus，registered 输出从下一 tick 可见 safe；每个物理 bus 再按其冻结 delay 后移，因此 safe transition 位于 `DONE + 1 + bus_delay`。若它落入曝光窗就必须写入 context，不能只展开用户编程的 DAC segment。compact repeated DAC 或 live ramp 若无法从当前 TargetIR 无歧义展开则 fail closed，绝不猜中间值。

同一calibration capture中所有被layout选作runtime readout event的repeat/scan cells必须派生完全相同的`ReadoutPhysicalContext`；reference events可以承担不同的制备/标签物理语义，不能被错误要求与readout event同波形。preflight从持久pulse lineage派生一次并写入同一source resolution；triggered occupancy则在任何camera arm/FPGA FIRE之前从当前camera capability、当前compiled pulse/cell plan派生并与artifact比较。裸`apply_calibration(artifact, Value)`只保留为明确的非权威纯数值函数：它验证结构/schema，但没有物理lineage，不能产生或冒充正式occupancy artifact。qCMOS当前尚未资格化edge-to-integration offset，adapter必须发布`None`；因此它可以做诊断capture，但权威calibration/triggered occupancy会明确拒绝，不能默认猜`0`。VirtualCamera的已声明offset为`0`，可用于离线/E2E验证。

Calibration的计算事实与提交权威是两个不同类型。`CalibrationComputation(artifact, report)`只表示纯计算已经通过artifact/report绑定校验；该构造边界也是“detector centers符合request中可选spatial intent”的唯一owner，后续边界信任这个不可变类型，不重复计算residual。公共`compute_calibration(CaptureArtifact, request)`和package-private的raw-frame oracle都只返回该非可提交类型。正式flat RunPlan的package-private `_analyze_calibration_resolved(...)` 才能从preflight已经持有的exact `AdmittedCapture + _ResolvedCalibrationSource`铸造closed `CalibrationAnalysisResult`；不存在可被普通调用者直接调用的公开authority constructor。这个结果携带“意图已由CalibrationComputation绑定验证”、同一次process-local source admission、exact resolution和整条run的内存证明。`final_commit`只核对这些held proofs与逐组context，不重验site residual、不重新admit source、不重建physical waveform。加载已有report时可以复用`CalibrationComputation`的纯绑定验证，但不能借decode把它升级成提交权威。这个类型级边界消除了SiteMap/feature/threshold与另一份report错绑后被提交的路径；producer内部仍直接核对source layout、grid、frame shape、site count、model kinds/default、逐model threshold、由held-out evidence推导的usable mask，以及request声明的feature类型/box/PSF geometry，作为实现自检；不恢复artifact fingerprint或proof graph。

Repository将runtime `CalibrationArtifact`与diagnostic `CalibrationReport`分成两条读取面：current manifest配对一个小型typed artifact blob、一个report-metadata blob和一个codec-stable `CalibrationRuntimeSummary`；全分辨率`reference_average(<f8)`与validity使用两个raw CAS blob，由metadata的owner-encoded `ContentRef`引用。summary按每个持久逻辑array字段计数，绝不依赖进程内ndarray alias；`inspect_final()`仅读FINAL manifest即可得到source capture ref、binding、site/model选择和owner报告的inspection/admission上界，完整load后必须重算并exact compare。`load()/admit()`只读取manifest+artifact，绝不为occupancy拉入report、平均图、validity或SciPy；只有显式`load_report()`或paired `load_computation()`才按内存预算materialize diagnostics，后者一次返回已经重新互证的artifact/report而不做两次decode。写入端在编码或复制全分辨率诊断前先按已知shape/dtype做峰值admission，metadata产生后再做精确第二次检查；staging预算区分raw image/mask copy与其它canonical array的normalize/base64/JSON/decode-roundtrip overlap。在发布manifest前还必须用读取端同一个size/structure/typed decoder做round-trip，不能生成自己无法读取的FINAL。pending recovery对raw arrays使用有界流式hash，不materialize大图，也不重跑detector、fit或threshold。Repository锁只保护open/commit状态与coordinator线性化点，CAS read/write、report decode和大数组复制全部在锁外；因此GUI打开report不能阻塞并发runtime admission。

Occupancy Repository的FINAL metadata以各自owner codec保存exact counts/occupied `DatasetSchema`，并保存raw values/validity blob的ContentRef与size。`inspect_final()`只验证FINAL、run generation、两份schema的共同repeat/point/layout/SITE domain及blob size，不admit source capture/calibration，也不materializearray；这使metadata-only FigureDocument可以得到精确axes/dtype/unit，而不是按artifact kind重造schema。完整`admit()`随后从实际Capture+Calibration binding重新派生应有schema并与metadata exact compare，再解码arrays。持久schema只是FINAL索引和早期fail-closed证据，不是第二物理算法authority；domain schema validator与zlc_data schema codec仍各只有一个owner。preflight在admit dependencies前先用已知dependency/analysis峰值早拒绝，只有通过后才做exact metadata/storage第二门，避免一个已经必然超预算的Figure/analysis先物化大依赖。

formal calibration 的资源路径遵守“先便宜inspection、汇总依赖、释放inspection图，再materialize”的顺序。结果存活到FINAL时的权威峰值证明为 `S + C + A + K`：`S`是Capture owner报告的source retained上界，`C = 128 KiB + 2048*trace_count + 384*transition_count`是exact `ReadoutPhysicalContext`语义保留，`A`是analysis及非context staging，`K = 1 MiB + 8*C + 8*escaped_JSON_name_bytes`是calibration codec拥有的tree/JSON/decode round-trip workspace。`K`只在exact physical context解析后计算，不能拿完整pulse waveform的保守summary替代而误拒绝；FINAL在首次encode前用同一owner重算并与held proof比较。encode后的artifact/report大小上限、canonical structure admission和typed round-trip仍分别执行，但不再另造一套局部“post-encode peak”与whole proof重复计费。这里的memory limit约束本次workload/repository产生和保留的分配，不冒充Python解释器、首次SciPy import或不受信任同进程plugin的whole-process RSS硬隔离；需要对hostile plugin给出硬上限时应使用进程隔离，而不是继续给同进程边界叠guard。

### 13.2 算法权威与明确偏离

Calibration/readout 的唯一物理算法权威是 `main@6c337d49c7086fa0ff21f879cd159bdf0e753f51` 的实际代码；任何旁路归档、旧迁移样本或本设计文档都不能反向定义 production 合同。当前实现逐项继承：全部 reference frame 平均、Gaussian smooth/local maxima/5×5 subpixel refine、separable lattice repair、四种 grid order、3×3 BOX mean 默认及 mean/sum/median/max、7×7 empirical PSF 与 annulus-median、uniform PSF、96-bin quick Otsu、pooled per-site bimodal strict consensus、per-site/per-class 90/10 seed-0 split、120-bin common edges、empirical balanced threshold、held-out/model/global fidelity 与 drop-worst ablation。训练和 runtime 共同调用唯一 feature extractor。

main strongest-N detector 存在一个不能靠同帧内部规则性消除的信息歧义：真实 site 变暗而出现更亮伪峰时，它可能产出另一个自洽但物理错误的规则格；仅凭同一张图的 peak count、lattice residual 或规则性无法区分真格与假格。因此 exact-main detector 一行不改，但正式 authority 增加独立空间意图 gate：`expected_centers_xy` 必须按当前 ordering/FrameContract 给出粗略逐 site 位置，`maximum_site_residual_px` 给出显式容差；detector 结果不吸附、不重排、不替换，只要任一 site 超限就拒绝。无该意图的 `compute_calibration` 仍可生成非权威 preview；formal Run在启动/昂贵计算前拒绝缺失意图，package-private `_analyze_calibration_resolved`成功返回的closed `CalibrationAnalysisResult`则成为后续提交边界信任的类型证明，不在authority mint/final commit重做同一residual校验。Workbench 可以显示 detector 建议并让用户明确核对，也可以从先前已 admit 的同物理 FrameContract calibration 预填；不能把同一次 detector 输出无提示地自动回填成自己的权威证据。

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

neutral domain/runtime 不接受“执行时读取 session current calibration”、裸 filesystem fallback 或 legacy path search。Notebook facade 构造 Occupancy/Detection/Scan request 时必须显式接收并 load/admit `CalibrationArtifactRef`，验证 readout binding，解析显式/default/唯一 model，并把具体 ReadoutBindingKey、CalibrationArtifactRef 与最终 `ReadoutModelKind` 冻结进 request。若 ref 缺失、binding/FrameContract/model 不适用或选择歧义，request 构造/preflight 失败；运行中不存在可切换的 facade current pointer。Workbench 同样在用户点击 Run 时冻结用户明确选择的 ref，而不是让 processor 回查 mutable session或按 repository 最近文件猜。

### 13.4 执行

```text
CalibrationTask:
  LiveCalibrationInput:
    CaptureSession -> CaptureRepository.atomic_put -> CaptureArtifactRef
  CaptureArtifactInput:
    CaptureRepository.admit(CaptureArtifactRef) -> AdmittedCapture
  -> resolve once: AdmittedCapture + _ResolvedCalibrationSource
  -> _analyze_calibration_resolved(held source, held resolution, explicit request)
       -> compute_calibration(admitted_capture.artifact, explicit request)
       -> CalibrationComputation(artifact, report)          # non-authoritative pure stage
  -> CalibrationAnalysisResult(computation + held source/resolution + memory proof)
  -> CalibrationRepository.final_commit(runtime artifact + diagnostic metadata/raw arrays + manifest)
  -> CalibrationArtifactRef
```

live 路径先提交原始 CaptureArtifact，再与 offline 路径汇合；detector/feature/model 无法构造完整请求结果时不发布 CalibrationArtifact，但原始 capture 仍可诊断和重跑。低 fidelity 本身是 report evidence，不由没有 main 依据的 Holm/Clopper/valley gate 擅自拒绝整个校准；具体坏 site 通过 `usable_sites` fail-closed。virtual/real 只在 CaptureSession adapter 不同，提交后的 calibration 代码完全相同。

当前 qCMOS 真机 calibration 还有必须由 E0 硬件事实关闭的 gate，但软件侧不能继续把本可读取的事实留成猜测。adapter 已在完整配置事务结束后一次读取并冻结实际 `EXPOSURETIME`、`TIMING_MINTRIGGERINTERVAL`、readout speed、sensor mode、trigger-global-exposure 以及 trigger source/active/polarity；qCMOS 的 minimum interval 必须严格为有限正数，trigger trio 必须仍是 external/edge/positive，无法读取、后续 ROI 操作改变 trigger mode或配置中途失败都会清除旧 working-point proof。ROI 写入按 `SUBARRAY OFF -> zero positions -> sizes -> final positions -> SUBARRAY ON/readback` 完成。所有 public configure 路径在取得 acquisition lock 前后都检查 arming/armed，关闭同线程 RLock 重入与跨线程 B→A 的 ABA；`cap_start` 后还重新从硬件读取完整 working point，任何 drift 都先 stop/release/disarm并失败，因此 camera endpoint 在 FPGA FIRE 前比较的是 arm 后真实 readback fingerprint，不是配置期缓存。

compiled binder 现在会用上述冻结 working point 对**同一 artifact 内相邻 trigger**做 fail-before-arm 的最小间距检查；这不等于已经证明 arm-ready 到第一沿、最后一沿到 drain/下一 run 第一沿的跨边界余量，后两者仍必须由 Q0/E0 qualification 给出。更根本的实验 gate 也仍存在：adapter 配置 `TRIGGERACTIVE.EDGE`，一次 arm 只有一个 hardware `EXPOSURETIME`；checked-in calibration pulse 的 20 ms/5 ms/20 ms 只是 FPGA period/probe-window 时长，不能被软件宣称为三种相机曝光。E0 必须证明 edge-to-integration offset、所选 trigger mode 的曝光语义、每沿一帧/顺序/不漏、arm/first-edge 与 run-boundary margin；若相机只支持 per-arm 固定曝光，bracket 必须改成物理可实现且算法语义正确的协议，不能把 pulse width 冒充 camera exposure。virtual 帧通过、计数最终相等或 GUI 上看见三帧都不能替代该证明；这些事实资格化前 qCMOS 正式 calibration 用户路径仍为 NO-GO。该 gate 优先使用相机与现有 FPGA 的硬件时序，不自动授权 RTL/bitstream 变更。

CaptureArtifact 的大帧面只有一个公共 owner：`frame_source: CaptureFrameSource`。它保存完整 `DatasetSchema`、block/revision、精确 cell schedule、event-order metadata 与固定大小 raw frame chunks；不再并列暴露 `.block`、`.event_metadata`、`.source_cell_schedule` alias，也不保留旧 whole-DataBlock blob reader。chunk 以约 64 MiB 为目标并受 repository policy 上限约束；普通 `load/admit` 只验证 manifest/index及 chunk refs，实际 `read/iter_cells` 首次用到某 chunk 时核验其 size+SHA，pending-commit recovery 才逐块流式全验。这里的 admission 证明 commit/journal authority 与索引可解析，不等于全介质健康扫描；未读取 chunk 的损坏会在第一次读取/计算时以内容损坏明确拒绝。显式 `materialize(memory_limit_bytes=...)` 是唯一 whole-dataset 入口，预算同时包含最终 block、构造 copy、validity 与最大单 chunk scratch。index 写入和读取共用同一个 size、canonical-node/container、typed reconstruction 和 re-encode owner；任何 writer 自己读不回的 index 在 manifest 可见前拒绝。compiled capture plan 在任何 camera arm/FPGA fire 前取得 repository root borrow；close 若先赢则 run 在硬件前拒绝，run 若先赢则 borrow 阻止 repository 在 finalize/cleanup 前关闭，不能完成硬件后才发现保存根已经失效。

frame bytes 的 invalid/component-invalid/NaN 规范化只由 `zlc_data.canonical_value_array` 拥有：它在 schema-level INVALID 快捷返回前仍验证 dtype/shape/validity，避免任意错误 frame 取得合法 invalid digest；普通 C-contiguous uint16 VALID frame 返回原 view，不为 hash/持久化前检查复制整帧；Capture repository 仅在真正写 CAS 的边界转为 bytes。native/big-endian 等数值等价 dtype 先转换到 schema 的 canonical endian，再对 float/complex 的每个 NaN component 规范 payload；不能用 canonical component dtype 去解释尚未换 endian 的 complex bytes。schema-level INVALID 沿用既有 `canonical-invalid-values` event digest，component-invalid 则以相同 mask 与零 filler 产生相同 identity，不能让两条路径漂移。float/complex NaN payload canonicalization 需要的临时 mask 是独立 admitted scratch，不能因为结果最终仍是一块 frame bytes 就漏算峰值。

analysis 按 resolved join/context 从 `CaptureFrameSource` 流式消费；不 `np.stack` 原始 frame、不为每帧构造第二份 owned image，也不生成 `(groups, shots, H, W)` 临时栈。reference 阶段允许为“平均图”和“按最终 site feature 提取”各走一次可重复源遍历；short 阶段对每帧只准备一次并同时填入全部 model 的小型 `(model, groups, sites)` signal/validity 数组，禁止每个 model 重读整套 qCMOS frame。reference average 使用一个 float64 image accumulator和按真实 shot 数选择的最小无符号 count image，最终原位除法；空间复杂度是 `O(HW + groups*shots*sites + models*groups*sites)`，不是 `O(groups*shots*HW)`。cell 地址用可重复惰性 generator，不提前构造 repeat-expanded row对象；report 逐组保存原来的 `(AxisId, logical index)` context，repeat、多条 point axis和二维 data axes各自保留语义，绝不能变成匿名 `data_points/data_dim`。

`CalibrationAnalysisRequest.max_drop` 省略时取 `min(5, site_count)`，显式值不得超过 site count：更大的值只会重复“全部 site 已排除”的同一报告，不是新证据。preflight estimator 与 analysis 数值实现同属一个 owner；对完整矩形 layout 直接计算 selected-row cardinality，对 sparse/product layout 只做不保留 row/context 的 physical 计数，必须在构造 join 前完成预算拒绝。总峰值取三个真实时相的最大值：join 构建临时图；retained compact join + materialized `group_contexts` + source read scratch + science working set；以及 result 已存活并携带held source/resolution时的publication staging。预算还显式计入72 bytes/pixel 的实测保守图像工作集、全部 model short arrays、histogram、PSF、每个 ablation point 的 mask+Python object overhead、每 site/model 的 retained Python result graph，以及 inherited robust lattice 对每条 grid axis 构造的无序 anchor-pair slope workspace，不能把 Python context graph、held source/join retained bytes或细长 grid 的二次项记成零。真实 `1×1000` grid 反例中，旧 estimator 为 `1,533,616 bytes`，权威数值路径的 tracemalloc 峰值为 `24,398,478 bytes`；按每个 unordered pair `64 bytes` 与每 site/model object graph `1024 bytes` 计费后为 `35,549,616 bytes`（约实测 `1.46×`），并由直接 profile inherited `_robust_axis_lattice`、而非重抄估算公式的独立 oracle 固定。2304×2304 qCMOS 下，64 MiB chunk 产生约 81 MiB（VALUE/cell validity）至 91 MiB（full component validity）的读取 scratch，72 bytes/pixel 图像项约 364.5 MiB；紧凑 site/join/context 项另计，任何双 chunk、全帧 float64 copy、限额检查前的 full join或未计费 ablation/lattice pair 都是回归。

runtime feature extractor 保留 camera 原 dtype，只把当前 site 的 BOX/PSF 小窗口转换成 float64；float reducer/weighted sum仍与 main 数值路径一致。2304×2304 uint16 针对性 profile 中，旧整图转换每帧额外 `40.50 MiB`、约 `11.91 ms`，当前四个 3×3 ROI 路径峰值增量约 `0.04 MiB`、约 `0.028 ms`。`readout_runtime_scratch_nbytes` 与 numeric operator 同属 calibration owner并保守覆盖 annulus 全图 median fallback；pipeline 只调用该 estimator，不能复制公式或恢复 `bound.operator_scratch_nbytes` 镜像。

不需要 CalibrationService、child Measurement Run、calibration StreamProcessor、recursive execution plan、WorkPlan 或 reducer 包装。当前 `compile_calibration_artifact_plan` 只是一个同步 flat `RunPlan` adapter：preflight以FINAL inspection先拒绝不可能的预算，随后一次admit/resolve并取得capture+calibration repository borrows；execute调用package-private `_analyze_calibration_resolved`；finalize只消费同一held result/proof并发布FINAL，随后释放borrows。cancel/execute failure不发布manifest，也不存在finalize重新admit source。长计算可由现有 RunController 的普通 worker hosting 执行，不因此发明 calibration 专用 async engine。只有出现一个必须在采集完成前反馈、且不能保存原始样本后批处理的真实用例，才另行设计领域 StreamReducer。

Occupancy request 携带 `ResolvedCalibration(reference, artifact)` 和已解析的 `ReadoutModelKind`；repository `admit` 负责 FINAL/source 验证，但返回值不冒充不可伪造的 authority token。该轻量领域值归 `calibration.py` 所有，因此导入 occupancy runtime 不加载 calibration repository、report codec、analysis 或 SciPy；fresh-process import ratchet机械锁定此边界。processor bind 再比较当前 capture 的完整 FrameContract、SiteMap/model kind 与 axis。任何 mismatch 明确失败，不按相同 shape 猜“应该兼容”；hot path 每帧只执行共享 feature extractor 和 `>` classifier，原子发布 counts、occupied、metadata 和相同 component validity。

## 14. PulseScan

### 14.1 两种明确语义

```text
FormalPulseScanRun   软件链exact、物理关联end-attested、可产生权威ScanArtifact
LiveSweepMonitor     非权威、可跳帧、只用于显示
```

它们不是 fallback 或兼容双轨，而是两个不同 use case。LiveSweepMonitor 不能保存为成功 ScanArtifact。

PulseScan 的精密时序由冻结bitstream上的FPGA scan engine与qCMOS外触发硬件执行。正常qCMOS autonomous baseline由host在run前冻结完整配置/计划并在一次FIRE后只做排空与末端验证；API segmented是唯一例外，host在显式segment boundary选择下一个已冻结API point，但每个`STATIC_ONCE` segment内部的全部edge仍由硬件决定。两种模式都为整run建立一次camera arm/exact capture transaction。autonomous保存一个完整table pulse terminal；API保存R-major/P-fast的R×P个独立、session-id唯一的physical pulse terminal，再在全部segment完成后形成唯一aggregate CameraRunEvidence/EndAttestation。两者都不假定逐沿counter、delay-idle或PHYSICAL_DONE，也不为API伪造per-segment camera terminal。当前RTL的logical DONE可能早于内部delay scheduler排空，CURSOR也不可见该队列，因此raw readback必须经H1定义的mode-specific terminal recipe提升为pulse terminal；任何保守monotonic tail wait都只防止过早进入下一API segment或关闭camera，不选择、移动或调度segment内部的pulse edge，不能被宣传成硬件tail-idle receipt。Q0 qualification中的camera drain bound只在整run末端用于aggregate camera terminal，并必须与完整counter/metadata对账共同使用。

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
  ProgrammedImageDeploymentRecordRef + pinned revision/digest
  immutable point/repeat axes + PointLayout
  ordered expected ScanCellKeys/TriggerKeys
  frozen slot values + trigger schedule
  bound exact source pipeline
  source_association_contracts: tuple[BoundSourceAssociationContract, ...]
  resolved ScanOutputContract
  execution_mode = AUTONOMOUS_RESIDENT | AUTONOMOUS_REFILLED |
                   API_SLOT_SEGMENTED_EXISTING
  pulse_evidence_contract = AutonomousTableTerminalContract |
                            ApiSegmentedPulseEvidenceContract
  camera_evidence_contract = one run-level CameraRunEvidenceContract
  required_association_proof = ORDERED_END_ATTESTED_RUN
  formal_requirements digest
  total event/byte/cardinality budgets
```

`BoundSourceAssociationContract`逐source冻结`source_id`、expected input/output keys与grouping、qualification或capability ref、terminal recipe id/version、required proof class和source-specific budget；它不是插件registry，也不把qCMOS字段强塞给其它Measurement。scan owner 的具名 builder 以静态 TaskDefinition key、typed request和bindings构造`RunPlan[ScanArtifactRef]`，完成纯 request/port/claim 绑定；preflight 在正确 I/O lanes 解析硬件 capability、schema、counter mode、compiled pulse compatibility 与全部预算，返回领域私有的 ScanPlan 作为该RunPlan的prepared value。ScanPlan 一旦生成不可被 GUI/ControlTopic 修改，也不包含 child RunPlan。

point_axes/PointLayout 决定 logical cell 顺序，trigger schedule 明确每个 ScanCellKey 期望的 TriggerKeys。`slot_binding` 是用户/模板的参数语义；`execution_mode` 只描述物理装载/执行方式。`AUTONOMOUS_RESIDENT`和`AUTONOMOUS_REFILLED`共同属于现有bitstream的`AUTONOMOUS_STREAMED`方式族：SCAN_SLOT/MOT 的**完整逻辑 finite table**必须在fire前冻结、编译并digest，FPGA在一次fire后自主决定微观时序。resident模式在fire前上传全部物理table；只有显式通过§15.4强证明的refilled capability才预装初始banks并在运行中按已冻结table的immutable chunks补充，host不得选择下一point或调度edge。只有 selected=API_SLOT 且 adapter明确证明该 API值无法在一次自主 sweep中更新时，才允许既有 `API_SLOT_SEGMENTED_EXISTING` 路径；它在preflight先冻结P个唯一point document/STATIC_ONCE artifact、R×P的R-major/P-fast cell schedule和整runcamera frame budget，只在显式host boundary按该冻结顺序执行，不能反向成为 SCAN_SLOT fallback。任一数量、slot、所需source qualification/capability、schema或 output contract无法在第一次arm前解析，preflight失败且不 arm；R×P control/terminal memory也必须在解析或编译P个point前先准入，不能先构造R×P份document/artifact再事后检查。类型模型允许未来多个source-specific合同，但近期S4 Formal enablement只开放**恰好一个Q0-qualified qCMOS physical source**；多physical source或非camera source在其association/terminal contract、contract kit和真实用例完成前typed NO-GO，不能借source-neutral接口自动获得Formal资格。

`execution_mode`不包含`FORMAL`或`END_ATTESTED`字样，因为装载方式本身不能授予权威资格。ScanPlan中的`required_association_proof`只是本run必须达到的目标等级，不是已获得证明；PROVISIONAL epoch、UI草稿和中间dataset不得把它显示或序列化为achieved。只有EndAttestation成功后，immutable EpochValidationRecord才写入`achieved_association_proof=ORDERED_END_ATTESTED_RUN`。Formal eligibility 由FIRE线性化点有效的Q0 qualification authorization、achieved proof、exact pipeline合同、authority transform与本run成功EndAttestation共同决定；ScanArtifact逐项保存这些事实和最终eligibility record，不能压成一个mode字符串。

Formal SCAN_SLOT 的 repeat axis 必须在compile阶段展开进一个完整、有限、冻结的scan table，并使用现有`repeat_forever=False + scan_points` finite single-pass路径；table顺序由ScanPlan的repeat/point axes与PointLayout唯一决定。**禁止使用当前`scan_repeats>0`的cursor-wrap + host `CMD_SAFE`路径**：它由host观察wrap后停止，现实现明确可能已经多发下一sweep的一个point，不能进入Formal Scan。迁移期尚未替换的非权威交互入口若仍使用该机制，只能归入有删除点的legacy monitor岛；最终`LiveSweepMonitor`也不得依赖cursor-wrap host stop，旧路径在S4迁移完成后删除并由Z0验证为零。

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
-> SOURCES_STOPPED -> SAFE_CONFIRMED -> FINALIZING_SAFETY
-> one SafetyDispositionBundle durable -> hardware-free PostSafetyContext
-> PROVENANCE_VALIDATED
-> COMMITTING -> terminal publish + claim release -> SUCCEEDED
```

任意 duplicate、out-of-order、typed key mismatch、gap、EOS incomplete、schema change、timeout 或 hardware fatal：

```text
ABORTING -> SAFE_CONFIRMED --+
         -> SAFE_FAILED ------+-> FINALIZING_SAFETY
                                  -> one SafetyDispositionBundle durable
                                  -> FAILED + quarantine-unsafe-keys
                                  -> terminal publish + all claim release
```

若错误发生在本run唯一SafetyDispositionBundle已经durable、硬件能力已经撤销后的PROVENANCE_VALIDATED/COMMITTING，不再重复调用硬件safe；删除未提交temp或保留已原子提交manifest这个客观事实，再按§8.4的`safety_bundle_id + commit_id + manifest digest` reconciliation发布FAILED或SUCCEEDED并释放claims。未提交manifest时绝不把保存失败误报成采集成功，已提交manifest时也绝不谎报CANCELLED；两种情况都不重复fire/重开source。正常成功同样必须先用一个bundle durable resolve该run全部hazards，再进入PostSafetyContext执行final artifact publish，最后线性化terminal+release。

正常和abort路径都在artifact commit前终止physical sources并按现有协议请求sequencer safe。abort顺序固定为：设置cancel intent -> 调用现有abort/safe尽快阻止更多trigger并取得logical terminal/safe ack -> 按H1验证的safe/abort tail recipe保守等待可能仍在delay scheduler中的物理输出排空 -> camera仍保持capturing，在Q0合同内保守drain并冻结final metadata -> camera stop/disarm、stable check与buffer release -> abort/drain workers/builders -> join acknowledgement。不能先做长CPU/磁盘工作再请求safe，也不能在camera drain前调用`cap_stop/disarm`。只有SAFE_CONFIRMED且mode-specific sequencer terminal evidence、deployment-bound compiled/H1 post-terminal tail evidence、全部source metadata/terminal recipes、join、DatasetBuilder coverage与最终EOS全链通过EndAttestation，才允许ScanArtifact commit；无法确认safe或tail bound则ResourceClaim quarantine。provenance validation失败只产生RunFailureRecord，不保留已显示的provisional rows。

### 14.5 Scan keys

```text
ScanCellKey:
  run_id
  point_index: tuple[int, ...]
  repeat_index

TriggerKey:
  cell_key: ScanCellKey
  trigger_ordinal

EpochIntegrityState:
  PROVISIONAL | VALID | INVALID

EpochValidationRecord:
  provenance_epoch_id
  validated_dataset_revision
  proof_class + proof_digest
  terminal_state = VALID | INVALID

EpochBoundDatasetRef:
  dataset_revision_ref: zlc_data.DatasetRevisionRef
  provenance_epoch_id
  validation_record_ref: optional until terminal
```

ScanCellKey 对应最终 DataBlock 的一个 `(R,P)` cell。`point_index` 与 ScanPlan 的 point_axes 一一对应，通过同一个 PointLayout 映射物理 P；即使只有一根 scan axis 也使用长度一 tuple，不能重新引入扁平 `sweep_index`。TriggerKey 区分同一 cell 内的多次硬件触发。

这些epoch类型由neutral Formal Scan领域拥有，不进入zlc_data或frontend.figure codec。EndAttestation不能原地把旧DataBlock字段从PROVISIONAL改成VALID，而是原子发布一个独立immutable EpochValidationRecord；Workbench的LiveDatasetBinding把EpochBoundDatasetRef解析成snapshot和presentation-only integrity badge，ArtifactController/Fit input adapter则在调用zlc_data/frontend owner codec之前检查authority eligibility。PROVISIONAL可以带明显状态live显示，但不能作为CommittedTransform的权威输入，不能提交正式/interactive source-specific authoritative fit artifact、FigureArtifact或其它derived artifact；显式排障保存只能生成`DIAGNOSTIC_PROVISIONAL`。INVALID时workbench递增epoch lifetime token，使queued BoardFrame/fit/save stale并清除或持续标红旧视图。API segmented的单segment通过仍是run-level PROVISIONAL，只有aggregate EndAttestation才产生VALID record。

qCMOS adapter在匹配有效Q0 qualification envelope时按冻结schedule为第i个按序frame生成**provisional TriggerKey[i]**；需要多帧/多source的StreamProcessorDefinition声明grouping与join-key transform，在完整输入到达后产生恰好一个provisional ScanCellKey typed result。scan DatasetBuilder只接受ScanCellKey并验证计划内每个cell恰好一次；整个epoch只有EndAttestation后才转VALID，不能在验证前提交，也不能从monitor/latest路径填“下一个cell”。

sequencer terminal evidence按execution mode使用两个互斥值类型，不能用一个全optional结构或UI progress猜测：

```text
AutonomousTableTerminalEvidence:
  PreparedProgramRef + compiled table/schedule digest
  H1 read-recipe revision
  stable raw STATUS + final CURSOR

ApiSegmentEvidence:
  repeat_index + point_storage_index
  point-indexed compiled STATIC_ONCE artifact + one-trigger schedule digest
  unique PulseSession/PulseTerminalAck lineage
  H1 physical-terminal recipe result（若底层raw事实为DONE/STATUS则CURSOR=N/A）
```

`AUTONOMOUS_RESIDENT/AUTONOMOUS_REFILLED`只接受前者；`API_SLOT_SEGMENTED_EXISTING`接受恰好R×P个后者，并要求它们按R-major/P-fast完整覆盖cell schedule、pulse terminal session id互异、同一point跨repeat复用同一compiled artifact identity。两者都只消费现有寄存器/transport事实，不新增RTL；`scan_progress()`镜像、缺失字段的默认值或人为构造的cursor都不能成为terminal evidence。`ApiSegmentEvidence`只证明对应有限pulse session物理终止，不是camera terminal。整个API run另有且只有一个`CameraRunEvidence`，它绑定整run唯一arm spec、source schema/schedule digest、R×P event span和aggregate `CaptureTerminalAck`；只有run级EndAttestation验证二者完全一致后才可VALID。

近期 baseline 的关联模式是 `ORDERED_END_ATTESTED_RUN`，不是逐 cell handshake，也不要求逐沿 FPGA tag：

```text
Q0 CameraExternalTriggerQualification
  冻结 qCMOS model/serial/firmware、SDK/driver/adapter version、
  trigger mode/polarity、ROI/binning、exposure/global-exposure/readout mode、
  buffer policy与counter/stamp/timestamp语义版本、位宽、signedness、modulus、reset epoch，
  arm-ready/status ack语义、arm-ready到第一沿的最小余量、最小active/inactive pulse width、
  最后一沿到driver可见frame的最坏延迟与terminal quiet-window
  在声明的完整物理trigger waveform与trigger_interval_min + safety_margin工作区间内实测：
    每个外触发产生一帧、delivery order稳定、frame/camera stamp单调连续
  保存样本数、持续时间、最坏间隔与实测丢帧/乱序结果

Preflight
  从冻结 CompiledPulse trigger schedule 得到 expected TriggerKeys/总帧数
  从 camera envelope 得到arm/edge/pulse-width/frame-interval/tail-drain限制
  对compiler展开delay、polarity与相邻高段merge后的实际物理waveform逐项验证：
    arm-ready ack成立，arm-ready到第一沿余量通过，active/inactive width通过，
    每个相邻有效 trigger间距 >= minimum + configured margin
  验证driver ring的max_inflight×frame_bytes满足Q0 drain bound；另行验证
  total frames/bytes <= host exact retention/consumer/artifact sink budget
  若table超过resident window，要求§15.4完整AUTONOMOUS_REFILLED capability：单I/O owner、
  refill transaction保守硬上界，以及对每个潜在seam（含无camera edge区段和tail seam）
  足分辨率的硬件时间观测/完整schedule residual attestation；camera timestamp只能作为
  补充证据，不能单独证明transport从未stall；任一条件缺失都在fire前拒绝
  清空host pending并按adapter contract处理driver residual；保存pre_arm_residual_observation，
  但不跨cap_start/reset epoch把旧counter绝对值当作本session baseline
  冻结 camera settings readback；不满足则不 arm

Autonomous Run
  camera一次 arm整个 scan session；capture expected count按总frame budget冻结，
  driver ring只按max_inflight定容并由dedicated drain持续排空
  cap_start/arm-ready后、FIRE前按Q0 reset epoch读取session_counter_baseline；
  若counter或per-frame stamp只在首帧可读，则Q0必须定义implicit initial/first-snapshot
  与first-frame successor rule，否则该工作点不具备Formal capability
  arm后、正式fire前若出现不属于Q0声明arm行为的frame/counter变化立即失败；
  第一个正式frame必须满足该session baseline或first-frame rule
  FPGA以repeat_forever=False一次fire并自主执行展开repeat后的finite scan table
  adapter按 delivery order保存全部 frame及 framestamp/camerastamp/timestamp
  所有数据在末端验证前均为 PROVISIONAL

EndAttestation
  single I/O owner按ScanPlan选择并验证mode-specific pulse evidence：
    autonomous table = 一个完整table terminal；
    API segmented = R-major/P-fast的R×P个独立STATIC_ONCE physical pulse terminal
  pulse evidence无歧义证明对应完整run schedule完成，且
  expected_trigger_total_from_completed_schedule == run级camera_produced_delta
  frame/camera stamp按Q0语义连续，timestamp间隔在Q0容差内
  唯一CameraRunEvidence证明一次arm、aggregate stop/drain/no-more-frames/join，且
  每个BoundSourceAssociationContract的terminal recipe、DatasetBuilder/processor/EOS coverage完整
  任一不符 -> 本 attempt 整体 INVALID并丢弃；是否重跑只由用户或显式有限RetryPolicy决定；全部通过才提交
```

这里的“重跑”始终创建新的 `run_id/attempt_id`，重新执行 preflight、qualification FIRE gate、arm/FIRE、采集与 EndAttestation；失败 attempt 的 RunFailureRecord 和原始诊断 provenance 必须保留。禁止在原 attempt 中从失败位置续接、只补缺失 point、复用旧 authorization，或由 UI/adapter 在未声明 RetryPolicy 时静默重跑。即使 RetryPolicy 允许自动重试，也必须有明确次数/时间预算，且只有某个完整新 attempt 独立通过全部 commit 条件时才产生成功 ScanArtifact。

通过 Q0 后，`frame[i] -> frozen trigger schedule[i] -> TriggerKey` 是该 qCMOS/工作区间的adapter contract。Q0是对一组冻结设备身份、firmware/SDK/driver/adapter、采集设置、buffer policy、arm/pulse/interval/camera-tail envelope以及counter/stamp/timestamp reset/modulus语义的**经验性发布资格**，保存有限样本、统计上界和PI明确接受的残余风险，不要求每个run重做长时间统计实验；每个run只验证自己仍落在该envelope并执行EndAttestation。上述任一身份/版本/语义字段改变、设置超出已批准集合、或归因完成后确认一次camera-envelope合同违例，都使该qualification对相应工作点失效，恢复Formal capability前必须重新Q0 qualification。这里依赖的是经真机资格化的ordered external-trigger contract，不是数学上的确定性证明、运行时“取latest”或两个自由流按N zip；只有frozen compiled schedule本身可称为确定性展开，host侧reservation/cursor则保证相机已交付的每帧不会在软件缓冲中静默跳过。

`CameraExternalTriggerQualification`是neutral camera/scan领域拥有的immutable artifact，blob/manifest由zlc_storage canonical repository保存；它包含qualification id/revision/digest、设备与软件身份、批准工作点集合、统计证据、margin、PI批准和创建时间。installation级`CameraQualificationIndex`是`ACTIVE | SUSPENDED_PENDING_ATTRIBUTION | SUSPENDED_PENDING_RECORD | REVOKED`状态的唯一权威，使用append-only activation/suspension/exoneration/revocation records并跨重启恢复，不能靠覆盖artifact或删除文件撤销。record必须绑定qualification revision、device identity、工作点、effective scope/time和具名evidence；只有`ACTIVE`可进入Formal FIRE gate。

qualification authority与camera运行入口必须加入同一个installation authority及其跨进程物理owner proof，不能创建平行控制面。RunController在任何camera/sequencer configure或arm之前，已经从`RunPlan.bound_devices[*].binding_stamp`内部派生并持久化本run的HAZARD_ACTIVE records；调用者不提交第二份hazard列表。preflight取得camera claim后解析active revision并pin其digest。真正提交FIRE时调用短原子`pin_for_fire` gate，在同一installation线性化边界内复核既有hazard id仍active、identity/generation/settings与qualification revision仍匹配，生成引用该hazard id与revision的immutable `QualificationFireAuthorization`，并把FIRE命令提交给既有transport后才释放gate；它不在arm之后新建或替换HAZARD_ACTIVE。activation、suspension、exoneration、revocation和其它FIRE gate均与它串行，因此不存在“复核后、FIRE前”插入撤销的窗口。真实camera Formal run由EXCLUSIVE claim串行；尚未fire且pin旧revision的run在gate处失败。

同一个`pin_for_fire`还必须复核ScanPlan pin住的`ProgrammedImageDeploymentRecordRef`仍是该sequencer endpoint的active revision；authorization保存该revision/digest。deployment record变化与Q0 activation/suspension使用同一installation线性化gate，不能在复核后、FIRE前换成另一份installation mapping。

已fire run观察到原始camera counter/stamp/timestamp的明确违例时本run立即INVALID，并在解除HAZARD_ACTIVE/释放claim前持久化suspension/revocation。若归因尚不明确但合理可能属于camera envelope，先写`SUSPENDED_PENDING_ATTRIBUTION`暂停该工作点；只有证据排除camera原因后才能用`QualificationExonerationRecord`恢复原revision。若qualification journal写入或ack失败，内存状态进入`SUSPENDED_PENDING_RECORD`、不得继续ACTIVE，且本run不能解析HAZARD_ACTIVE或释放camera claim；进程崩溃后未解析的installation safety record继续阻止下一run，直到恢复流程补齐qualification disposition。processor、DatasetBuilder、EOS、artifact或已明确归因的一般transport失败只产生各自RunFailureRecord，不能直接REVOKE。

历史加载验证artifact保存的`QualificationFireAuthorization`在该run的FIRE linearization point是否有效，而不是要求该qualification今天仍ACTIVE。revocation record的`effective_scope`必须明确为`FUTURE_ONLY | FROM_FIRE_SEQUENCE | ALL_USES_OF_REVISION`：普通现场违例通常从incident run起生效，之前artifact保持“当时有效、后来撤销”的provenance；若发现qualification证据本身无效，可显式追溯覆盖整个revision，旧artifact仍可读取但不再具备authority eligibility。不能用当前index状态无差别洗白或否定全部历史结果。

当前baseline不定义`emitted_total`字段，artifact/UI使用`expected_trigger_total_from_completed_schedule`。自主table模式只有唯一I/O owner按H1冻结的读序/稳定规则取得无歧义的完整table physical terminal时，才可由compiled schedule的有效camera-trigger数得到；API segmented则必须由R-major/P-fast的完整R×P个`ApiSegmentEvidence`证明每个STATIC_ONCE pulse session均取得独立physical terminal，再把每cell one-trigger schedule求和。当前高层`scan_progress()`及其后台轮询维护的`_scan_point/_scan_sweep`只是UI诊断镜像，Formal EndAttestation禁止消费；它可能滞后、漏掉最后跃迁，也不能替代原始寄存器证据。`expected_trigger_total_from_completed_schedule`是“整个run的全部pulse schedule已完成”条件下的推导值，不是逐沿硬件实测counter；任何early stop、raw状态组合歧义、自主模式cursor未达终点、API cell terminal缺失/重复、session id复用、顺序不完整或transport error都使EndAttestation失败。

raw terminal也不能证明内部delay scheduler已经排空。`CompiledPulseArtifact`必须根据冻结channel delay、最后有效edge与当前RTL tick/quantization语义给出`max_physical_output_tail_after_logical_done`；H1 contract kit用golden/xsim/真机观测验证该上界及safe/abort变体，并给出保守余量。唯一I/O owner在观察到mode-specific raw logical terminal/safe ack后记录monotonic起点，camera与dedicated drain保持运行，直到`elapsed >= compiled_tail_bound + h1_margin`才生成`PostTerminalTailEvidence(compiled_digest, h1_contract_revision, programmed_image_deployment_revision, terminal_evidence, required_bound, elapsed)`。该evidence只证明host在对应installation deployment/H1上界之后才继续termination，不是当前硬件不存在的tail-idle receipt或runtime bitstream content attestation；monotonic wait可以保守超时，不能提前返回，也不能用于安排实验edge。用户cancel只能把run置为INVALID，不能取消这段cleanup wait；bound/deployment revision缺失、版本不匹配、进程/transport/时钟异常导致tail recipe无法完成时，Formal epoch INVALID且设备claim保持到quarantine/recovery裁决。

`camera_produced_delta`也不是累计counter的裸绝对值，而是adapter按Q0冻结的位宽、signedness、modulus、reset epoch与rollover语义，从`cap_start`/arm-ready后且FIRE前建立的`session_counter_baseline`到terminal drain后的最终counter计算出的本session增量；禁止跨`cap_start` reset epoch使用`pre_arm_residual_observation`的绝对值。per-frame `framestamp/camerastamp`逐帧验证modular successor并做唯一可逆的unwrap；`nFrameCount`来自`cap_transferinfo()`累计快照，只按Q0语义验证session baseline/final delta，可选中间快照只要求modular monotonic并允许batch jump，不能错误要求每个交付frame对应counter恰好+1。预期范围、完整per-frame metadata序列和起止值必须使wrap次数唯一；若stamp只在首帧出现，则首帧必须满足Q0定义的initial/successor rule。最终counter delta还必须与本session实际保留的frame metadata条数和首末stamp关系交叉一致；任何多解、未声明reset、stamp duplicate/gap、counter倒退、rollover歧义或delta/metadata不符都使epoch INVALID。timestamp检查按Q0实测的“相机timestamp事件定义 + 非均匀trigger schedule + readout容差”比较，不能简单要求固定间隔或拿host wall clock替代。

对 SCAN_SLOT/MOT，`HOST_STEPPED_GROUP`、逐 cell arm/fire/wait、single-cell fire gate、per-cell `PHYSICAL_DONE` receipt 均不属于 baseline，也不能作为 qCMOS 首光、容量或证明 fallback。SCAN_SLOT/MOT 必须使用现有 FPGA 的完整逻辑table自主执行：近期无缝装载方式baseline为`AUTONOMOUS_RESIDENT`全量预装，`AUTONOMOUS_REFILLED`只有经§15.4强证明后才可成为条件execution capability。唯一非自主例外是既有adapter确实不能在一次自主sweep中更新API值时的`API_SLOT_SEGMENTED_EXISTING`；它必须以execution mode、canonical rationale与冻结cell schedule如实表明显式host boundary，且每cell只等待pulse physical terminal与同一armed camera transaction中的下一个exact event，不能称为autonomous，也不能泛化成可供SCAN_SLOT复用的host-stepped mode。Formal eligibility仍由Q0、exact链、association proof与整runaggregate EndAttestation联合决定。架构不得为了获得逐cell证明而要求新bitstream。

`AUTONOMOUS_STREAMED` 是当前冻结bitstream的正式执行方式族，`AUTONOMOUS_RESIDENT`是不依赖下述增强的近期装载方式baseline；refilled的条件门见§15.4。只有 E0a/Q0 在批准工作余量内、正确camera配置和充分软件reservation下仍实测发现丢帧/乱序，且已经证明camera设置、软件保留/排空策略、降低trigger rate与扩大margin均不能修正；或发现现有 RTL 真 bug/与既定设计不符时，才可提出证据驱动的 bitstream变更。仅仅metadata语义不清、样本量不足、一次未归因异常或无法建立qualification时，结果是不开启Formal capability，不构成硬件修改授权。届时可选增强之一是在真正 camera-trigger输出沿记录：

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

相机 adapter 每帧保留 DCAMBUF_FRAME 的 `framestamp`、`camerastamp`、`timestamp`，并把`cap_transferinfo().nFrameCount/nNewestFrameIndex`作为同一读取时刻的累计count与ring位置快照。当前 `CameraFrameRecord` 边界已经做到metadata保留；DCAM读取又已改为count-first、newest-index反推槽位、已报告backlog完整排空、exact count失败/倒退与复制期可能覆盖时fail-closed，并把wait+copy纳入同一deadline。S1仍必须让最终 CameraPort/CaptureSession、exact retention、typed Q0与EndAttestation端到端消费同一record语义，并由Q0实测决定max-inflight与drain/copy余量；当前constructor默认的live `recent_capacity`不能冒充真机资格化结果。会解包成纯ndarray的旧 public consumer只在其最后一个legacy consumer迁走的dependency-closed切片删除，不能无条件写成S3，也不能重新退化为array-only路径或把累计`nFrameCount`伪装成per-frame metadata。这些字段的语义必须按具体qCMOS型号实测，字段存在和host排空正确本身都不等于TAGGED或ordered-trigger qualification。

每个run的camera start boundary也是关联合同的一部分：adapter必须在arm前排空/拒绝旧software pending与driver residual并保存`pre_arm_residual_observation`；随后在`cap_start`/arm-ready后、FIRE前按Q0 reset epoch建立`session_counter_baseline`。若counter或stamp只在首帧存在，Q0必须定义implicit initial、first-snapshot与first-frame successor rule，否则该工作点不具备Formal capability。必须证明arm本身是否可能产生frame，禁止跨cap_start reset把旧epoch绝对值带入delta。任何未声明的pre-fire frame、reset epoch不符、首帧不满足规则或stop后late frame都使整个epoch INVALID；不能只依赖“最终总数恰好相等”来掩盖开头混入旧帧、末尾少一帧的错位。

ScanArtifact 的 provenance manifest 分别保存 `execution_mode`（`AUTONOMOUS_RESIDENT`/`AUTONOMOUS_REFILLED`/`API_SLOT_SEGMENTED_EXISTING`）、EpochValidationRecord的`achieved_association_proof=ORDERED_END_ATTESTED_RUN`、ProgrammedImageDeploymentRecordRef revision/digest、全部BoundSourceAssociationContracts及其qualification/capability refs、formal eligibility record、冻结source settings readback、compiled trigger schedule/tail-bound digest、mode-specific pulse evidence与H1 terminal recipe版本、`expected_trigger_total_from_completed_schedule`、camera pre-arm observation、session baseline/final、counter/stamp width/signedness/modulus/reset/rollover语义与`camera_produced_delta`、owner-minted ordered-metadata digest、source event span、source schedule/frame-index -> TriggerKey mapping digest，以及aggregate end-attestation结果。逐帧metadata与frame payload仍由source DatasetArtifact拥有，ScanArtifact不复制第二份列表。两种模式都只保存一个绑定整run唯一arm spec、source event span、schedule digest与aggregate CaptureTerminalAck的`CameraRunEvidence`。autonomous mode另保存单个run级`QualificationFireAuthorization`与`AutonomousTableTerminalEvidence`；API segmented保存同一个run级camera authorization、canonical `segmentation_rationale`、P个point-indexed compiled artifact identities、R-major/P-fast的R×P个`ApiSegmentEvidence`；execution mode+rationale+有序evidence已经显式表达host boundary，不另存逐boundary时刻，也不保存逐segment camera authorization、camera settings副本、camera EndAttestation或伪造的camera terminal。加载时这些记录与ScanPlan的required proof、TriggerKey coverage、deployment revision和revocation effective scope一起验证；不能只保存 `mode="ordered"`、计划要求值或一个混合Formal资格的执行字符串。

`AUTONOMOUS_RESIDENT/AUTONOMOUS_REFILLED`的正常运行闭环是：camera一次arm整个session并冻结expected total（driver ring仍只按max-inflight定容）-> 等待Q0声明的arm-ready/status ack并验证first-edge margin -> 建立本session counter baseline/first-frame rule -> FPGA一次fire完整逻辑scan table -> exact queue按序保存所有frame+metadata -> 唯一I/O owner读取raw FPGA terminal/cursor并按H1规则确认完整logical table terminal -> **camera仍保持capturing且dedicated drain继续运行**，从观察terminal的monotonic起点完整等待CompiledPulseArtifact/H1给出的保守physical output-tail bound并生成`PostTerminalTailEvidence` -> 再持续到expected metadata齐全并经历Q0-qualified terminal quiet-window -> 冻结final counter/stamps -> `cap_stop` -> 复核capture/transfer状态稳定 -> 最后才release driver buffer -> 完成processor/DatasetBuilder最终EOS -> 执行EndAttestation -> VALID后才commit。raw STATUS/CURSOR自身不证明delayed-output tail settle；这里“停止/disable trigger”只指logical engine已经terminal/safe并且后续tail recipe完成，绝不指在drain前调用camera `cap_stop/disarm`。现有`_disarm()`中`cap_stop`后立即`buf_release`的路径不能复用于Formal CaptureSession termination。H1 output-tail bound和Q0 camera tail latency/drain deadline/quiet-window都是经contract/qualification获得的有限运行合同，不伪装成数学上的逐沿no-more-frame证明，§14.5声明的剩余风险仍然存在。abort路径先用现有abort/safe阻止新logical edge入队，再按H1 safe/abort tail bound保持camera capturing并drain，随后才final metadata -> cap_stop -> stable check -> release；无法确认logical terminal、tail evidence或camera终态时整run INVALID并quarantine。

API segmented不使用上述“一次FIRE完整table”描述，但**仍使用一次arm完整camera run**。它按§14.7逐cell执行独立STATIC_ONCE pulse session，在每个physical pulse terminal后才进入下一segment；camera与dedicated drain从首次FIRE前一直保持同一session，全部R×P完成后才aggregate complete/stop/join并执行一次EndAttestation。未验证前每个Envelope携带run-scoped provenance_epoch_id且formal sink只暂存；任一pulse terminal、count、stamp、timestamp、coverage、timeout或hardware error不符使整个epoch INVALID并丢弃，不能提交前半段。

明确接受的取舍：这是“preflight余量 + per-run末端对账 + reject-and-redo”，不是“per-cell当场fail-closed”。它通常能发现漏帧、乱序、未完成和大间隔异常，但不能定位具体point，也不能数学上排除漏一个触发/帧同时出现一个额外触发/帧且metadata仍落在容差内的等量抵消。PI接受这一剩余风险以保持RTL冻结；自主模式仍保持无缝硬件扫描，API模式则明确接受非连续host boundary。文档、UI和artifact provenance必须如实标记 `ORDERED_END_ATTESTED_RUN`，不得声称拥有逐沿accepted-trigger证明。

INVALID必须形成可查询的RunFailureRecord，保存失败原因、工作点、counter/stamp摘要和累计失败率。系统不得无限或静默自动重试直到“碰巧成功”；重试只能由用户发起，或由request中显式、有限、可审计的RetryPolicy发起，每次attempt具有独立run_id/provenance，最终artifact记录失败attempt refs。这样reject-and-redo不会掩盖硬件不稳定或造成不可见的选择偏差。

只有 E0a/Q0 在目标工作余量、正确camera配置和充分软件reservation下观察到真实丢帧/乱序，且软件保留/排空、camera设置、trigger rate与margin调整均无法修正；或代码/RTL证据证明现有实现与既定设计不符，才重开bitstream评估。候选可以是bug fix、HardwareTriggerStamp、trigger-return或其它最小修复；先证明问题与候选改动之间的因果关系，不能因为相机侧异常就默认修改FPGA。触发条件、证据、替代的软件/相机配置方案和重烧风险必须单独评审，不能由本架构自动授权。

### 14.6 非标量 y

正式 scan 的权威 y 在 bind 时只有两种合法结果：通过 `CommittedTransform` 具名 `Select/Reduce`；或把所有仍存活的 trailing data axes 原样保存。`batch axis` 是后续 Fit/Analysis 如何解释这些轴的语义，不是 ScanArtifact 再保存一份的事实；scan 不为尚未发生的分析提前把 data axis 标成 batch，更不能为了画一条曲线先压成 scalar。

```text
ScanOutputContract:
  committed_transform: CommittedTransform
  output_dataset_schema
```

这两个字段已经包含全部不变量：`CommittedTransform`拥有输入schema fingerprint与具名操作；`output_dataset_schema`拥有repeat、point axes、PointLayout、全部存活data axes、dtype/unit和ValidityContract。exact source ref/schema、完整DatasetSealProvenance与processor/calibration artifact inputs由ScanArtifact保存。禁止再镜像`input_schema_fingerprint/repeat_axis/output_data_axes/batch_axes`，否则同一事实出现第二owner并产生漂移。当前真实producer都需要显式删除一个system singleton `READOUT_EVENT`，所以contract保持非空；等第二个已经是权威scan dataset、确实无需任何变换的producer出现后，再评估optional transform，不预建mode enum。

repeat 是 Scan Dataset 的权威 R 轴，不属于“可被 y transform 顺手 reduce 的非 scan axis”；binder要求它的AxisId/coordinates/layout原样保留。每根未被用户权威transform显式选择或reduce掉的trailing axis都继续存在于output schema。后续Analysis若想对repeat求均值或把site/spectral/spatial axis解释成batch，在冻结ScanArtifact上另建FitSpec/ReductionSpec，不能改写原始scan y。最终数组始终是 `(R,P,*data_shape)` 配具名axes，绝不能压成 `(repeat,data_points,data_dim)` 三个匿名长度。

物理采集完整性与component物理有效性是两条正交规则。missing/duplicate ScanCellKey、frame gap或EOS不完整始终使epoch INVALID；已完整cell中的dead site/bad pixel则按source声明的ComponentValidity原样进入成功ScanArtifact，供后续fit/reduce逐component消费，不能吞成NaN或把一个dead site误报为整run缺帧。当前只有这个真实策略，因此不建立`ValidityAcceptancePolicy`枚举；出现第二个确需“全部component有效才允许artifact成功”的实验request时，才把该物理准入意图加入request/contract，而不是由显示层推断。

Workbench 构造 Scan draft 时从 DatasetSchema、Selection snapshot 与独立 AnalysisPreset/Scan preset 构造 DataTransformSpec；ViewSpec 只提供当前可见 ROI/select 的候选提示，display mean/latest/sample/facet 不能复制。binder从CommittedTransform唯一派生output DatasetSchema，并对repeat/point/layout及每根存活data axis做total-coverage校验；用户启动scan时才冻结两者。scan运行中不能因latest、slider、ROI或panel切换而改变y语义。

final working set也在任何底层capture/processor plan编译和hardware prepare之前准入：zlc_data用同一个schema+CommittedTransform峰值函数同时服务pre-FIRE估算与实际materialize；processed scan再扣除冻结plan仍常驻的calibration array bytes。base pipeline完成后，application只抽取一个权威source snapshot/provenance/evidence tuple，先释放包含occupied sibling/event metadata的opaque joint result，再执行transform和同一Run的FINAL commit。这样既不靠事后OOM判断，也不为两种source建立通用workflow engine。

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

正常执行策略固定为 `AUTONOMOUS_STREAMED`，不改变模板语义；MOT/SCAN_SLOT 把 `da_x/da_y/da_z` 等完整逻辑slot table在run前冻结、编译并digest。resident模式一次上传全部物理rows；条件refilled模式只按冻结顺序补immutable chunks，不能把它解释成host逐point调度。禁止逐cell host mutation或API fallback。`API_SLOT_SEGMENTED_EXISTING`只在selected=API_SLOT且设备API无法无缝更新时使用既有路径；artifact以该execution mode、canonical `segmentation_rationale`及R-major/P-fast的有序cell pulse evidence表达“这里存在host boundary”，不要求或伪造每个boundary的计时记录。它是一个具名、封闭的API物理执行器，不是可被其它scan复用的execution policy。不得为了统一实现把SCAN_SLOT也切段。

API segmented模式不是任意API scan的自动fallback。`ApiSlotSegmentedProgram`必须直接携带canonical、trimmed、非空的`segmentation_rationale: str`，明确说明该实验为何允许host segment boundary。这个字符串是可审计的物理理由，不是新的policy类型、插件或可执行DSL；program/template owner对这个理由负责，preflight只验证它存在、与API-slot执行选择一起冻结且未被篡改，不尝试从自然语言推导物理真值。baseline只允许物理上接受**任意、可变、非负host gap**的实验；依赖连续时间演化、无缝扫参、段间状态不可重建、最大允许gap、精确settle/re-equilibration时间或任何gap-dependent physics的实验一律typed NO-GO。等真实第二种分段语义出现后再为那个具体request增加typed字段，不能提前恢复一个万能Segmentation对象。这项门只判断物理实验语义是否允许现有分段，不能用“每段数据exact”代替。

获准分段后的唯一执行模型如下，不能再派生“每段相机run”变体：

```text
Preflight
  先以R×P cardinality准入control/terminal lineage memory；不解析或编译R×P份对象
  冻结P个唯一API point documents与P个STATIC_ONCE artifacts，每个恰好一个camera trigger
  冻结全局R-major/P-fast DatasetCellSchedule、同一trigger channel、camera settings与R×P frame budget
  取得覆盖整个冻结run的一个camera qualification/authority；未通过则不arm

Execute
  建立一个external-triggered exact camera transaction；只在首个segment FIRE前start/arm一次
  for repeat in R:
    for point in P:
      为该cell打开新的PulseSession，prepare对应point-indexed STATIC_ONCE artifact
      确认上一cell的camera event/physical terminal及run authority
      以deadline/cancel-aware monotonic wait保守满足camera required external-trigger interval
      由硬件执行该segment全部edge
      exact capture_next取得同一camera transaction中的下一个有序event
      PulseSession.complete取得session-id唯一的physical pulse terminal
      终结并撤销该PulseSession的mutable authority，只保留immutable evidence
      记录ApiSegmentEvidence(repeat_index, point_storage_index, pulse evidence)
  全部R×P完成后才camera complete/stop/drain/no-more-frames/join一次

Aggregate
  CameraRunEvidence绑定唯一arm spec、完整source schema/schedule、R×P event span与aggregate terminal
  验证R×P pulse receipts、camera count/metadata、DatasetBuilder/processor/EOS与lineage全部一致
  只执行一次run级EndAttestation；成功后才发布唯一VALID EpochValidationRecord
```

host gap是`segmentation_rationale`明确接受并由program owner承担的API物理语义，不是精密延时工具。相邻segment之间仍必须按camera capability的`required_external_trigger_interval`执行保守的deadline/cancel-aware最小等待；它只保证下一外触发不会早于相机安全下限，可以更长且不承诺上限，绝不承担实验精确settle或edge timing。baseline不持久化每个boundary的host monotonic时刻，也不从相机timestamp反推segment gap；execution mode、rationale与有序pulse/camera aggregate evidence已经足以诚实声明“非连续”，但不能量化或冒充段间精密硬件时序。需要测量gap、限制最大gap或让结果依赖gap的实验当前直接NO-GO，而不是偷偷把这个安全wait升级成权威时间轴。下一segment只可在上一segment的camera event与physical pulse terminal都完成后开始；host不得在segment内部sleep排edge。相机在边界间保持armed/draining，不能stop/re-arm、清空counter或建立新capture epoch。任一pulse terminal、capture event、count/metadata或aggregate terminal失败都会poison同一个capture transaction并使整个run INVALID；不能只丢坏segment、拼接其余segment或自动重试后隐藏失败。

camera qualification/deployment authority同样是run级而不是segment级：首次FIRE前的线性化gate一次固定完整R×P schedule所用revision/digest。每个后续boundary可以对这个已固定authority做side-effect-free的仍有效检查，并在qualification被suspend/revoke或binding generation变化时阻止下一FIRE；它不能mint新authorization、重新pin另一revision或改变相机epoch。已完成segment的PulseTerminalAck仍作为失败run诊断事实保留，但不能因为前半段有效就部分提交。

这套模型有意只缓存P个resolved point documents/compiled artifacts，repeat只复用point-indexed identity；R×P只保留有界cell schedule和terminal lineage。preflight必须在解析P或调用compiler前先完成R×P control-memory admission，编译再按P渐进完成；禁止先构造`resolved_segment_documents[R*P]`、R×P份artifact或R×P个camera session。per-cell pulse terminal是必须的物理证据，但per-segment camera authorization、camera terminal、CameraRunEvidence或EndAttestation全都不存在；若实现为了接口方便制造这些对象，即是在复制同一事实owner。

compiler/preflight 必须基于现有 bitstream 支持的 scan table 和实际有效 camera-trigger schedule 工作；不能要求 SINGLE_CELL_SCAN_SLOT、one-shot cell token 或新 RTL 寄存器。若现有实现对合法 SCAN_SLOT table 有 bug，先由 golden/model/真机证据确认，再按“修复既定设计”流程决定是否动 RTL。

当前迁移中的 **S4a/W3c virtual authority** 只实现上述模型中不依赖真实硬件资格的确定性部分，不能被命名或显示为 Formal。它先把logical PulseDocument绑定到live target，以bound sequencer公布的resident capacity在展开前检查`R*P`；随后由pulse owner把whole-document RepeatRegion确定性展开为repeat-major的有限物理table，使用一次fire、`loop_count=1`和`repeat_forever=False`编译，而ScanPointTable始终从未展开logical document取得，所以physical row duplicate不会污染logical point identity。`inspect_scan`只完成纯绑定/编译/预算验证，不写repository；真正run由一个flat Run执行exact source/processor、transform与ScanRepository final commit，没有ScanIntent、raw Capture promotion或第二份history。该切片没有API-slot、host-stepped、cursor-wrap stop、RTL或bitstream fallback。真实hardware composition、Q0/H1/EndAttestation、Formal eligibility与RunFailureRecord查询面齐全前，hardware ScanArtifact继续typed NO-GO。

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

Pulse authoring 与加载只保留一个当前合同：`schema="zlc_pulse.PulseDocument"`。可编辑 schedule、scan parameters/recipe/table 与 target 都由 `PulseDocument` 的明确字段表达；编译后的 `TargetIR` / `CompiledPulseArtifact` 是不同类型，不能再用一个并不存在的 `kind=table|sequence` 字段把 raw sequence 塞回 authoring document。唯一公开文件入口是 `load_pulse_document()`，tree boundary 是 `pulse_document_from_tree()`；二者只接受当前 exact field set，所有 save 也只写这一格式，compiled artifact 不作为同名 `_program.json` sibling。`pulse_document_path()` 是扩展名/绝对路径归一化的唯一 owner，load、冲突检查与实际 save 必须消费同一路径；Editor 以一把本地 save lock 串行化同一 session 的并发保存，但不为没有用例的跨进程编辑另建锁文件/事务系统。Workbench 提交 save 时冻结当时的 editor session/generation，到完成前对称禁止 New/Open/load 替换 session，且 stale completion 不得更新新 editor UI；保存中途继续编辑会使当前 revision 自然相对已写入 baseline 保持 dirty。仓库中受版本控制的 pulse JSON 资产与当前 codec 同步提交并通过 round-trip/golden；不存在历史 fixture、旧 parser、逐版本 upgrader 或一次性转换器。仓库外旧文件不属于终态产品合同；未知 schema/field set 由该 current owner 以明确 `ValueError` fail closed，不得按字段存在、shape或名字猜测，也不得提示 runtime fallback。

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

这份对应关系必须成为neutral installation-owned、canonical且有revision的`ProgrammedImageDeploymentRecordRef`，而不是README或操作者记忆。`zlc_pulse`只拥有`BitstreamArtifactRef`、`ResolvedBuildManifestRef`、release/timing record refs及其canonical codec；`zlc_neutral_atom`的installation runtime/composition authority拥有`ProgrammedImageDeploymentRecord/Ref/Index`及active/suspend/revoke状态，record通过owner codec嵌入pulse refs，并映射`asset_id + canonical ResourceKey + endpoint matcher -> pulse image/release refs`，blob/manifest委托`zlc_storage`持久化。这样依赖只沿`neutral -> pulse`，pulse不认识AssetMap、ResourceKey、endpoint或RunPlan。record至少绑定endpoint/asset_id、canonical ResourceKey、冻结`.bit`文件content digest、ResolvedBuildManifest/release/timing-signoff record refs、现有`image.build_fingerprint`、programmed/independently-verified time与hardware owner批准。H1 contract activation、Q0 qualification activation、每次Formal QualificationFireAuthorization、PostTerminalTailEvidence和ScanArtifact都pin同一active record revision；endpoint重新program、release映射变化、record撤销或无法把当前installation可信地对应到该`.bit`时，立即使相关H1/Q0 capability suspended并令Formal NO-GO。这是deployment/SOP assertion，只说明installation owner声明并复核过“该endpoint部署了这份冻结image”；它不能证明硬件运行时content、implementation seed或timing token，也不替代现有fingerprint握手。建立/维护此record不要求修改RTL或重烧；若现有部署事实无法可靠重建，只能保持Formal关闭，不能补造记录。

`design_build_id + timing-signoff digest + programmed-bitstream content attestation` 是未来可选增强，因为把新字段放入ROM/USR_ACCESS需要重烧；它不是baseline、不是S4 gate，也不能为架构整洁主动实施。只有证据已经触发合法RTL/bitstream变更时，才随那次修复评估加入；若加入，仍需解决bitstream不能自嵌自身SHA-256的自引用问题，并由server content digest + runtime build id + manifest形成闭环。

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
  scan_slot_schema + frozen camera-trigger schedule
```

内容寻址，不通过 sibling 文件名判断新旧，不重复嵌入 source table。

prepare/upload/fire 在冻结 bitstream 的既有协议上增加软件侧显式身份与校验，不要求新寄存器或RTL状态机：

```text
NEUTRAL_COMMON_FORMAL_PREPARE(run_id, artifact_digest, ProgrammedImageDeploymentRecordRef)
  neutral installation authority verifies active deployment revision and pins it for pin_for_fire
  pulse server verifies existing image.build_fingerprint/geometry/ABI handshake
  pulse server rejects only stale connection/PreparedProgramRef or mismatched currently prepared image;
    it never imports or interprets neutral deployment ref/index state

PREPARE_AUTONOMOUS(run_id, artifact_digest, frozen full finite logical scan table)
  host/compiler validate capacity、slot schema、camera-trigger schedule
  expand repeat axis into table order; require repeat_forever=False, scan_repeats=0
  resident: upload all physical rows before fire
  conditionally-approved refilled: preload initial banks and bind an immutable,
    digest-checked chunk source for the already-frozen remaining rows
  server records PreparedProgramRef(
    server_connection_generation, run_id, artifact_digest, table_digest
  ) in software

FIRE_AUTONOMOUS(PreparedProgramRef)
  camera session is already armed with frozen expected_total；driver ring depth
  remains the separately-proven max_inflight, not total_frames
  current FPGA executes the complete logical autonomous table once；refill若获准
  只供应预定chunk，不决定point、slot value或edge timing

COMPLETE_AUTONOMOUS
  single I/O owner reads raw STATUS/CURSOR/error bits under H1 stable-read rule
  mint AutonomousTableTerminalEvidence(PreparedProgramRef, table/schedule digest,
                                       stable raw STATUS, final CURSOR)

PREPARE_API_RUN(run_id, P frozen API point values/artifacts,
                R-major/P-fast cell schedule, R*P camera budget)
  admit R*P control/terminal memory before resolving or compiling P
  validate all point settings、one-trigger STATIC_ONCE schedules、single trigger channel、
    total budget与canonical non-empty segmentation_rationale；freeze one camera arm spec/authorization
  arm/start one exact camera transaction before the first segment FIRE only

FOR_EACH_API_CELL(repeat_index, point_storage_index)
  open a new PulseSession for the already-frozen point-indexed artifact
  prepare through the existing API-slot path and verify prior cell terminal + run authority
  if not first cell: wait at least the camera required external-trigger interval
                     with deadline/cancel awareness
  FIRE_API_SEGMENT(PreparedProgramRef)；hardware executes every edge in this finite segment
  capture_next from the same armed camera transaction
  COMPLETE_API_SEGMENT -> unique PulseTerminalAck/ApiSegmentEvidence(CURSOR=N/A)
  do not stop/re-arm camera and do not mint camera terminal/EndAttestation here

COMPLETE_API_RUN
  after exactly R*P pulse terminals/events, complete the one camera transaction once
  mint one aggregate CameraRunEvidence and run-level EndAttestation
  verify complete R-major/P-fast pulse lineage、camera count/metadata、coverage/EOS

COMMON_PULSE_TERMINAL(mode-specific pulse evidence)
  raw terminal state proves logical terminal only；camera/drain remain active while host waits
  use deployment-bound CompiledPulseArtifact tail bound + H1 margin where required
  autonomous records one table terminal；API records one physical pulse terminal per cell
  Q0 camera quiet-window/drain and CameraRunEvidence occur only once at whole-run completion
  feed source-specific evidence and exact coverage into one aggregate EndAttestation
  no emitted-edge/per-cell camera receipt is claimed；API aggregate rules remain in §14.7

SAFE/RESET/connection loss
  software invalidates PreparedProgramRef and follows current safe/reset path
```

这里的`NEUTRAL_COMMON_FORMAL_PREPARE`只描述S4 Formal路径；H1前E0a诊断characterization不伪造deployment ref、不进入这条authority路径。deployment active/suspend/revoke与run authority线性化复核始终属于neutral installation authority；pulse server只消费pulse-owned refs/bytes和既有transport状态，不认识neutral record。PreparedProgramRef 是 host/server软件 guard，不伪装成硬件 one-shot token。它防止明显的旧连接、旧artifact和GUI状态漂移；deployment revision由外层run级QualificationFireAuthorization固定，两者都不能证明每个物理trigger沿。物理归属仍按§14.5的有效Q0 qualification + frozen schedule + EndAttestation。首次arm前，autonomous模式冻结完整logical table；API segmented模式冻结P个唯一API point program、R-major/P-fast的R×P schedule、整runcamera settings/budget与lineage。API每cell只新建pulse session/PreparedProgramRef，不重新冻结camera事实。两者运行中都不从mutable GUI state读取，也不能互相伪造table/cursor语义。

expected trigger count/schedule 来自compiler对实际配置的唯一camera output channel、active polarity、clock mux、相邻高段合并、channel delay和全部合法slot values的确定性展开；camera channel不能同时配置为clk_enable。该schedule用于preflight间距与末端映射，不声称是运行时逐沿回读。

当前冻结硬件的近期**无条件 Formal 容量线**只有`AUTONOMOUS_RESIDENT`：table不超过`2 * scan_bank_size`（当前默认4096行），全部物理数据fire前resident，硬件时序不依赖host。这里的“无条件容量线”只裁决何时可以授予Formal资格，不把用户的正常扫描改成host stepping，也不否认当前bitstream已经具备ping-pong refill。`AUTONOMOUS_REFILLED`是同一冻结bitstream上的自主流式执行能力，而不是未来硬件；它默认尚未取得Formal资格，物理执行仍须一次fire完整冻结逻辑table、绝不host-step。它要求一个final `FiniteScanStreamer` I/O owner同时负责status、cursor、bank refill、progress、cancel与completion，删除monitor thread和`wait_done()`争用同一transport的双owner；但“measured worst refill + Windows/Python scheduler allowance”不是确定性上界，当前RTL的UNDERFLOW又会在bank恢复后清零而非保留sticky history，所以仅有平均/p99/worst-observed admission、最终DONE或部分camera timestamp都不足以发布该mode。即使mode已发布，单次run的Formal eligibility仍需独立满足Q0、association proof、exact链和EndAttestation。

只有contract kit证明所有潜在bank seam均有足够分辨率、语义明确且覆盖stall影响区间的硬件时间观测，并能把每个seam与完整compiled schedule做residual attestation，同时证明refill transaction的保守硬上界时，才可为该设备/transport/workload发布`AUTONOMOUS_REFILLED` execution capability。没有camera edge的区段、最后一个trigger后的seam或任何不可观察stall都会使该能力不可发布；preflight返回`FormalScanCapacityExceeded(resident_limit, capability_unavailable_reason)`并拒绝大表。软件记录chunk seq/count/digest并在现有可读状态范围内fail closed，但不得把非sticky UNDERFLOW、DONE、camera局部timestamp或尚不存在的CRC verifier/BANK_VERIFIED当作“从未stall”的证明。真实实验对更大容量或更高性能的需求本身不构成H2解锁条件；仍必须命中E0a/Q0在批准工作余量内发现无法由软件、相机配置和margin修正的真实loss/reorder，或证明现有RTL bug/既定设计偏离，才可按H2评估最小硬件修复。

任何prepare/upload/identity validation失败都不得调用FIRE。Pulse RPC server 重启或重连会改变`server_connection_generation`，旧PreparedProgramRef在软件侧失效；即使设备仍保留旧active image也不能由正式路径误触发。API-slot segmented在run前只冻结P个point-indexed values/artifacts并跨repeat复用其identity；每个R×P cell取得新的PreparedProgramRef/PulseSession terminal lineage，但不复制point document、camera arm或camera terminal。

### 15.5 硬件安全

- baseline先使用host/compiler的typed range/capacity/slot/schedule validation，以及现有bitstream实际暴露的DONE/status/error/fatal/safe/reset回读；contract kit只声明真机证实存在且语义明确的位，不能把目标寄存器写成既有能力；
- upload/fire沿用当前已工作的UART/AXI/JTAG协议。host/server通过PreparedProgramRef、connection generation、artifact/table digest防止软件层旧程序误触发；若transport error或readback异常，禁止提交并按现有safe/reset路径处理；
- RemoteSequencer通过现有软件/transport能力提供bounded timeout、cancel/abort和safe调用；共享backend的第二socket不冒充硬件独立性。无法确认safe时resource quarantine，但baseline不因此要求新增watchdog/SAFE寄存器；
- runtime identity近期只要求现有`image.build_fingerprint`/几何/ABI握手一致；installation-owned deployment record可以保存已批准`.bit`文件的content digest与release/timing记录作为SOP provenance，但它不证明endpoint此刻实际运行的内容。需要新RTL才能提供的runtime `design_build_id`、timing-signoff ROM或programmed-bitstream content attestation均不是baseline；
- 逐沿counter/FIFO、per-fire count、PHYSICAL_DONE、BANK_VERIFIED/RTL CRC等均不作为当前合同。只有E0a/Q0/故障注入在已批准工作余量、正确camera配置和充分软件reservation下证实真实loss/reorder且非硬件替代方案均不能修正，或现有RTL偏离既定设计时，才提出与已证实根因有因果关系的最小硬件修复；
- 若未来合法重建bitstream，build仍必须满足unconstrained paths=0、WNS>=0、TNS>=0，并审查generated clocks、CDC、IP property和critical warnings；这约束未来修复质量，不授权为架构偏好重烧。

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

无领域语义的 `ContentRef{digest,size}` 及其 schema-free current tree 只由 `zlc_storage.content_store` 拥有；Capture/Calibration 等 manifest 必须调用 owner codec，不能各抄一份 `{digest,size}` parser。领域 typed Ref 仍分别拥有自己的 repository namespace 与 `target_ref` 文法；recovery 从冻结的 expected manifest digest 构造 typed Ref 后比较完整 `target_ref`，不得手工切 prefix/slice。storage owner 的 `identify_blob(payload)` 可以在发布前计算 canonical 内容身份，供 metadata 引用与内存 admission 使用；它不发布、不证明 durability，也不能被领域包用手写 `sha256(payload)+len(payload)` 替代。只有 `put_blob()` 才把 payload staging 到 CAS、核验并确认该 `ContentRef` 已可见。对 writable `bytearray/memoryview`，store 必须写入自己拥有的临时文件并按预计算 ref 重新校验后才能 atomic replace；若 replace 后的验证失败，必须删除目标并 flush parent，使读取保持 fail-closed，而不是留下 digest 与 bytes 不一致的可见对象。manifest 发布前遗留的不可达 blob 是安全 orphan、不是可见 artifact；它不构成自动 GC 或领域自算地址的理由。

F0 第一日即建立 cross-package golden/property contract：同一 primitive tree 在四个 owner 包中产生 byte-identical encoding/digest；嵌入 owner value object 时 outer manifest 使用 owner bytes/digest；字段重排、float edge、NaN、unicode、ndarray order/endianness 与版本变化均有向量。golden 不是允许四份实现漂移的补救，而是守卫唯一 encoder 和 owner codec delegation。

### 16.2 Atomic commit 与 load

各 owner context 的 typed Repository 委托 `zlc_storage` 的同一个 `BlobStore/ManifestCommitter` 实现 immutable content-addressed bytes、锁、fsync 与 atomic replace；owner Repository 仍负责 typed Ref、schema、canonical codec、lineage 和 load validation。`zlc_storage` 不 import AxisSpec、FigureArtifactRef、ScanArtifactRef 或任何领域类型，也不提供“万能 artifact repository”。commit point 是最后原子发布的 owner canonical manifest：

SafetyJournal与CommitJournal共享`zlc_storage.FramedJournal`的纯存储机制，但记录schema与状态机仍分别由neutral runtime owner定义。frame使用canonical bytes、稳定record id与SHA-256。普通`FramedJournal.append_checked()`在一次跨进程文件锁内重扫并验证prospective state后append并file fsync；PersistentSafetyJournal则在installation启动时用`FramedJournal.open_exclusive()`取得生命周期排他session，scan/repair一次并缓存已验证record index，steady append只对缓存执行幂等/冲突检查后写入并file fsync，不重新扫描整个历史。journal/lock文件首次创建时才需要同步parent directory；已存在文件的steady append不伪造额外目录变更。Windows必须真实调用可验证的directory-handle `FlushFileBuffers`或在root probe时拒绝需要目录durability的backend，不能把directory durability静默降成no-op。仅允许修复校验明确失败的最后一个torn frame；中间frame损坏、冲突duplicate id或非法COMMITTED/ABORTED、HAZARD/RESOLVED跃迁均fail closed，不能截断历史继续启动。

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

hazardous Run必须先让§8.4中该run唯一的SafetyDispositionBundle durable；任一UNSAFE key禁止发布成功final manifest，全部SAFE时CommitIntent直接记录该bundle id。SafetyDispositionBundle append、CommitIntent、manifest replace、COMMITTED-or-ABORTED resolution与Run terminal是有顺序但不伪装成跨文件原子事务的linearization points，startup按`safety_bundle_id + commit_id + target/manifest digest`执行确定性reconciliation。普通Repository不得跳过这个outer RunController gate直接保存“成功run artifact”。

content-addressed blob 允许并发 writer 幂等复用；manifest publish 使用 digest/id 冲突检查，不能覆盖不同内容。只有 repository 规模证明 unreferenced blob 回收是实际问题、且所有 owner 能提供已验证 committed-manifest roots 后，才增加 maintenance-lock 下的 mark-and-sweep；storage 不自行解析产品 manifest。这样 baseline 先共享崩溃安全机制和 canonical bytes，不为尚未出现的多 backend/复杂 GC 建一套存储平台。

每种 artifact 只按自己的合同判断 commit。显式“先采raw、以后再分析”的 Capture/Calibration workflow中，完整CaptureArtifact是独立上游事实，后续calibration失败不回滚它。PulseScan却是另一个用例：camera/processor exact dataset在本Run内只是provisional source，用户请求的唯一成功结果是canonical ScanArtifact；processor、transform或scan commit失败时不额外发布一个名字像成功scan的raw CaptureArtifact，也不创建第二条recover/promotion历史。若用户确实需要独立raw artifact，必须作为另一个显式Capture Run请求，而不是scan内部副作用。

当前Scan application使用一个flat Run完成`exact source -> optional processor -> committed transform -> ScanRepository FINAL`。ScanRepository是唯一dataset authority，manifest直接保存owner-encoded logical PulseDocument与compiled pulse blobs、PulseCaptureEvidence、exact source DatasetRevisionRef/schema与完整DatasetSealProvenance（processed source含processor stages和calibration ArtifactInputRef）、ScanOutputContract、canonical output DatasetRevisionRef、values/validity blobs及本Run safety_bundle_id。output BlockId由logical document、exact source revision identity和output contract共同派生；final ScanArtifactRef由包含实际values/provenance/safety事实的manifest内容寻址。没有ScanIntent、raw Capture promotion、`promote_scan()`、旧格式reader或两份manifest真相源。

blob staging仍可留下不可达安全orphan，但只有同一Run的`context.commit_final(ScanRepository.final_commit(...))`能发布成功manifest。publish lost-ack由RepositoryCommitCoordinator按稳定commit_id、target和manifest digest reconcile；artifact已经可见则客观返回同一成功ref，未可见才失败，绝不重新FIRE或退回raw promotion。current virtual/offline slice已经走这条flat commit；真实Formal enablement只是在同一边界加入Q0/deployment/EndAttestation与eligibility事实，不重建第二套scan repository或workflow engine。

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

FigureArtifact 保存 ViewSpec、当次 EvaluatedFigureData 的 input revision/resolution records、Selection snapshot、layer/model/fit lineage 和所引用 dataset digests；重开默认复现保存时的 concrete selection。用户明确切回 repeat 的 `LatestNonempty` display policy 后，才重新解析最大非空逻辑 repeat；这仍不等于恢复最后发布事件，也不进行新的 axis auto 推断。若用户只保存 workspace layout 而不 materialize live data，文件必须明确标为 session-only workspace，并在数据 lifetime 结束后显示 missing binding，不能假装是自包含 FigureArtifact。

当前 finite-preview component 只保留进程内 final MonitorDatasetSnapshot 供 panel 关闭前继续显示；尚未交付本节的 Save/FigureArtifact materialization，因此不能把该 final slot 当作可重开 artifact，也不能声称用户已经能保存 live 所见 revision。

### 16.4 当前格式名与重跑策略

正式runtime、authoring load/save、wire和全部artifact只接受各owner的一个当前格式名。真正长期落盘、跨会话读取的值保留朴素、无改稿序号的格式名（例如 `zlc_pulse.PulseDocument`、`zlc_neutral_atom.calibration-artifact`）；临时进程内摘要只保留用途明确的 domain separator。未知格式名清晰失败，不存在版本比较、旧 reader、upgrade chain、转换 CLI 或 GUI fallback。

当前系统尚无必须保存的旧格式生产数据；标定、capture、analysis 等实验 artifact 的策略是重新采集/重跑，不维护档案迁移器。只有已部署且被硬件/外部协议真实消费的 wire/ABI 结构版本可以保留独立版本号；该例外必须有双端 consumer 和部署证据，不能由软件改稿次数推导。终态 allowlist 只有 FPGA `LAYOUT_STRUCT_VERSION=3` 与已批准部署拓扑的 `zlc_pulse.PulseTargetABI/v1` hash domain；普通 PulseTarget/PulseDocument/RPyC artifact 格式不在例外内。`ZLC-CANONICAL-1\n` 字节前缀是生成后者获批 digest 的冻结 canonical hash 原语，不是第三个可协商格式身份；它必须随该 ABI 一起保持逐字节不变。仅改变软件格式名不得重签这些硬件事实。旧树 `RuntimeSequenceProgram` 的 wire version 4 在最后一个 legacy sequencer consumer 删除前属于迁移期部署事实，必须保持原值并 dependency-closed 删除，不能误归入终态 allowlist，也不能提前改写。

控制进程与 FPGA server 的当前 RPyC payload 是同一软件 release 的一个协议闭包，必须原子部署；字段或格式不一致时 fail closed，不提供 mixed-release reader、协商或 fallback。该部署约束不把软件 payload 的改稿次数提升为硬件 ABI。

## 17. 性能约束

### 17.1 Camera exact queue

CaptureSession queue 禁止 list `pop(0)` 的 O(n²) 路径，使用 deque/ring，实现：

- enqueue/dequeue 摊销 O(1)；
- bounded capacity；
- exact overflow/backpressure 显式；
- monitor overwrite/missed count；
- exact 与 monitor fan-out 不复制不必要的大帧。

capture层的预算只覆盖device driver ring、exact transport retention和单event冻结scratch。当前raw payload进入session后先生成owned snapshot，stream publish再生成自己的retained snapshot，因此保守预算至少额外包含`payload_contract.max_retained_nbytes + metadata_contract.max_retained_nbytes`；只有以后增加不公开、由同一contract authority mint的already-frozen emit路径并证明不发生第二份copy，才能删掉这项。DatasetBuilder current storage、immutable result copy与metadata retention由扁平pipeline compiler统一计入。Python Envelope/dict/deque/list/tuple与allocator headroom不接受调用方自报；PipelineMemoryProfile只接受用户选择的总内存上限，固定reserve与per-event conservative minimum由runtime policy拥有。admission在触碰硬件前产生与exact chain绑定的process-local证据，最终结果只保留可诊断的`aggregate_peak_bytes`；Python实现名、微版本、pointer width和policy fingerprint不参与预算裁决，也不进入持久artifact。UI preview/render snapshot属于后续Workbench aggregate profile，不塞回CaptureStreamContract形成反向依赖。所有大小乘法使用Python无界整数并在arm前与实际RAM上限比较，不允许固定宽度乘法溢出后得到较小预算。

PerformanceBudget 计算物理保留字节而不是简单 `payload × subscriber`：分别统计稳定 BufferId 去重后的 unique retained buffers、每 edge queue/ref overhead、monitor front buffers、builder chunks、in-flight Owned/BorrowedSnapshot、processor output和render front buffers。共享 immutable frame只计一次 payload但每个引用有自己的生命周期开销；monitor虽不参与 exact ack，也必须有独立上限，不能靠持有 Python ref阻止大帧释放。

### 17.2 Scan compile

使用：

- expression 预编译；
- typed contiguous arrays；
- vectorized expansion；
- 一次性 validation；
- source document 与 wire artifact 分离。

优化后 target IR/wire image 必须保持等价，时间和内存对点数近似线性。

Formal Scan性能预算另包含两项硬门：展开repeat后的`total_frames × frame_bytes`必须满足host exact retention/consumer与artifact流写预算，qCMOS driver ring则按`max_inflight × frame_bytes`及实测drain latency独立定容，不能把总帧数误当driver buffer数；超过resident FPGA scan window默认拒绝，只有§15.4要求的每个潜在seam全可观察、完整schedule residual与保守refill硬上界全部证明后才开放条件能力。不能依靠运行时stall、非sticky underflow、内存swap或丢帧维持表面可运行。

### 17.3 UI 与 analysis

- monitor 由 UI refresh rate 限制，不降低 acquisition rate；
- 独立 panel view-evaluation latest-only；同一 coherence group 由 board evaluator 选同一 JoinKey/revision并原子 present，Fit executor 队列 bounded，三者不互相阻塞；
- revision coalescing；
- stale fit/render result 丢弃；
- exact pipeline 的必要 StreamProcessor/DatasetBuilder 不与可丢弃 UI fit 共用一个拥塞队列；
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
- StreamProcessor chain 深度、fan-out、typed record bytes 与 DatasetBuilder materialization；
- FitResultBatch 的 batch size 与 model cost；
- artifact streaming write/load 与 digest 校验；
- 当前 TaskConsole 与目标 WORKER_RASTER_LIVE 多 panel board 的 ingest-to-visible、compose/present、GUI event latency、coherence mismatch 和 stale queue length；迁移后不得以回到 GUI compose 换结构纯洁。

机械 gate 使用 scaling 与配置预算，而不是拍脑袋的单机绝对秒数：queue/materializer commit 摊销 O(1)，scan compile/journal/artifact bytes 对数据量近似 O(N)，内存不随已 ack history 无界增长；p95/p99 latency 和 peak bytes 必须低于目标环境 PerformanceBudget/WorkbenchProfile 声明预算。任何超预算都保存 profile artifact，先定位 producer、copy、lock、solver 或 render 热点再决定优化层。

## 18. 测试体系

### 18.1 Package tests

各 bounded context 拥有自己的 unit/contract tests，根仓库只保留 architecture、cross-package integration、E2E 和 performance。

### 18.2 必须保留/新增的合同

Data：

- Value event 只携带 `(*data_shape)` 与 ValueSchema；DataBlock 只携带 `(R,P,*data_shape)` 与 DatasetSchema，普通 stream edge 拒绝 DataBlock；
- 每条 edge 恰有一个 event -> dataset owner：finite exact DatasetBuilder 验证完整 TriggerKey/ScanCellKey schedule、missing/key mismatch、ValueSchema 与原子写；live MonitorDataset 验证 keyed cycle 或按 sequence 管理 append window，二者不共享 mode/state machine；
- StreamProcessor 每次 invocation 返回一个 frozen typed record；同一 record 的字段共享 key/provenance，字段不同 cardinality/key/lifecycle 时静态拒绝并要求拆节点；
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
- CellValidity 与按具名 axis 广播的 ComponentValidity；`(group,site)` dead-site mask 在 reduce/fit/histogram/meter 中一致传播；
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
- rolling overwrite/wrap 的 current cell 由 EventRef/progress 驱动、覆盖 repeat 与全部 point axes 的显式 Selection 给出；最高 nonempty axis index 不得冒充最后发布事件；
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
- DefinitionCatalog 只由显式 imports 组装，重复 id fail，禁止 package scan/global registry；
- catalog Definition 不含 callable；owner top-level binder/operator 无 hidden closure/device/session/global mutable dependency；
- PipelineSpec 编译成唯一顶层 RunPlan，节点不能 start child run 或自行拥有 terminal state；
- bind claim superset 完整，preflight/execute 尝试新增 ResourceKey 失败；
- 同一PhysicalDeviceIdentity在Workbench/notebook/standalone/remote入口间只有一个installation authority和一份backend可验证physical-owner proof；两个进程各自的ResourceArbiter不能同时把本地EXCLUSIVE冒充成同一物理设备的跨进程所有权；
- TaskConsole、PulseGUI、Experiment/session与standalone real入口均拿不到raw device drive verb；quarantine或其它owner持claim时，从每个公开入口尝试camera acquire或sequencer prepare/fire都被同一authority拒绝；
- S0.5 legacy start 必须经过 LegacyRuntimeFence并登记`LegacyRunFootprint(claims, reference_keys)`；claims与实际host读写一致，reference_keys覆盖全部raw connection/lifecycle依赖；旧 thread 未真实退出/safe 前shutdown不能越过对应reference，新 Run只被真实冲突claim阻塞，所有 direct LogicNode.start 入口被机械禁止或限定为无硬件测试；
- 改变device/config/virtual-real只产生restart-required并请求同一个InstallationRuntime shutdown；与并发start线性化后新start为零adapter调用拒绝，console外handle和target Run同样被authority发现、cancel、join并完成durable safety，旧connection关闭前claims归零；原进程内不得构造或发布replacement graph；
- console打开时从非Qt notebook/kernel线程请求safe shutdown，硬件quiescence、safety disposition与close仍完全由InstallationRuntime完成；GUI只在Qt owner thread queued reconcile，event loop阻塞、QWidget callback失败或窗口已销毁都不改变硬件正确性，也不存在跨线程QWidget调用；
- 对startup的journal lock、physical-owner proof、adapter open、identity、AssetMap、broker bind、capability probe与graph freeze逐点故障注入：Run admission始终未开放，已打开的exact owned subset按reverse close order关闭，绝不发布partial Experiment/catalog/drive facade；
- 对shutdown的run join、SafetyDispositionBundle、RecoveryAttempt lost-ack retry、broker invalidation、每个adapter close、lane stop、journal unlock与physical-owner release逐点故障注入；失败保持同一个runtime/graph owner和admission closed，可幂等重试，只有CLOSED+进程退出后新config才能启动；
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
- 同一 generation 第二个 formal reservation 被拒；一个 exact DatasetBuilder + bounded monitor fan-out 不重复 pin payload；
- event/byte budget 不足 preflight 拒绝；
- total expected events 与 max-inflight retention 分开计算；ack 后 retained bytes 下降，不可 backpressure burst 使用最坏 backlog 而非平均率；
- NON_BACKPRESSURE_CAPTURED 在第一处 backlog/retention miss 永久 RetentionOverrun poison；后续物理帧不能顶替失败 sequence，emit/finish/cursor/seal 全部失败；
- candidate Envelope/contract/key/timestamp 验证或容量 preflight 失败时不先 trim 已有 record，rejected publish 对 stream 状态零副作用；
- broker-minted generation 使两个同 StreamId/BlockId/schema、不同内容的 capture 得到不同 DatasetRevisionRef；
- PayloadContract 统一 snapshot/validate/retained/max bytes，ComponentValidity mask 与 per-event schema metadata 均不能绕过预算；
- exact Delivery 必须属于 builder 绑定的具体 source+reservation；同名 source、伪 cursor/Delivery/EOS、跨 tap MonitorUpdate 均被拒；
- frozen sequence->cell schedule 阻止合法 key 的 row swap；TraceBinding 阻止同一 reservation 混入另一 run/source；
- DatasetPreviewSnapshot 不能进入 formal storage/authority processor，只有 SealedDatasetArtifact 或 VALID epoch wrapper 可以；
- DatasetBuilder 异常退出统一 abort+release，不覆盖 body error或泄漏 formal claim；
- stale DeviceCapabilitySnapshot/output schema mismatch 在 arm/fire 前拒绝；
- StreamProcessor cardinality/byte-bound 无法证明或 formal path 含未解释 filter 时 preflight 拒绝；
- StreamProcessor output contract 依赖首帧数值或运行中改变 record fields/axis 时不能编入 formal pipeline；
- continuous Measurement 只能使用 admitted MonitorTap/MonitorDataset；exact request 必须有限且可完整 reservation；
- MonitorTap backlog 与 MonitorDataset window 固定容量、报告 lifetime missed 与 current gap，只产生 provisional atomic snapshot；若要成为 formal input，必须显式冻结新的 finite diagnostic input或启动 finite exact capture，不能给 live snapshot 改名；
- 单个 typed output record 的 ack 只在 publish 与所有 required downstream 接收成功后；不同 cardinality/key/lifecycle 的结果必须建独立节点，不能伪装为一次多-output transaction；
- exact/monitor 同源 event_id；
- EventSpanRef digest/count 等价于显式 ordered events，lineage 不随累计输出 O(N²) 增长；
- driver buffer 重用与 monitor overwrite 不破坏 exact payload lifetime；
- `CameraFrameRecord` 构造即拥有并冻结图像 bytes；driver ring slot、原 ndarray 和 metadata wrapper 随后改写都不改变 record，非整数/bool ordinal、负值、非法 microseconds 和无效 host receive time 在进队前拒绝；
- qCMOS fake/contract kit 验证同一 `buf_getframedata` 的 frame/camera stamp、timestamp、driver buffer index 与同一 drain 观察点的 `nFrameCount` 原样进入 record；同一batch的 `produced_count` 允许重复，不被改写为伪逐帧counter；
- `read_frame_records()` 与迁移期 `read_frames()` 消费同一 armed-session queue，不重复交付、不分岔 ordinal/metadata；finite arm budget、pending max-inflight、duplicate/gap 任一越界都在原子入队前返回 `CameraBufferOverrun`，不部分保留该batch；
- coherent monitor 永不混 shot，INDEPENDENT_LATEST_MONITOR 不可用于相关 expression；
- EXACT_KEY/coherent monitor 对 join_key type/schema mismatch 或 missing key fail closed；
- 独立设备 ZIP_SEQUENCE 被拒绝；
- Q0 CameraExternalTriggerQualification在最终adapter与每个批准的ROI/exposure/readout工作点验证一触发一帧、delivery order、frame/camera stamp连续性、timestamp语义、counter/stamp modulus/reset/unwrap、buffer行为、arm-ready/status ack、arm-to-first-edge余量、最小active/inactive pulse width、安全trigger间隔、last-edge-to-driver tail与terminal quiet-window；超出envelope不能声明Formal capability；
- Q0 artifact保存样本量、观察loss/reorder率与统计上界；未达到PI批准的样本量/上界/margin不能activate qualification；版本/设置变化、显式revocation、重启恢复和并发preflight pin均有contract test；
- preflight对compiler展开delay、polarity与相邻高段merge后的完整物理waveform验证arm-ready、first-edge margin、active/inactive width和所有相邻trigger间距，要求`interval >= camera minimum + configured margin`；边界内/外、过窄pulse、过早首沿、delay/polarity、merge与重复trigger edge均有测试；
- repeat axis确定性展开进finite table；Formal program强制`repeat_forever=False, scan_repeats=0`，任何`scan_repeats>0`/cursor-wrap stop在compile/preflight被拒绝；repeat/point/TriggerKey round-trip覆盖多repeat；
- qCMOS autonomous与API segmented都一次arm整个scan session；API按R-major/P-fast执行R×P个独立STATIC_ONCE pulse session，segment boundary期间camera保持同一armed/draining transaction，全部cell后只生成一个aggregate camera terminal；driver ring按max-inflight定容，`total_frames/bytes`通过host exact retention与artifact预算；超容量在arm/fire前拒绝；frame[i]只在匹配active Q0 qualification envelope时映射frozen TriggerKey[i]；
- qCMOS stream声明NON_BACKPRESSURE_CAPTURED；故障注入证明物理帧B publish失败后generation立即RetentionOverrun，物理帧C不能占B的sequence，任何capture后的decode/copy/schema/trace/key/publish异常统一SourceFailed且不可继续formal collection；
- BoundCapturePort拒绝普通/伪造/stale capability值，只接受DeviceBroker对同一BoundDevice generation mint的attestation；prepare/terminal必须回显并匹配spec/settings/capability digests；
- MeasurementDefinition/catalog fields递归只允许declarative values，runtime不执行Definition.bind或spec codec；FrozenCaptureSpec payload/digest篡改、owner/schema mismatch在claim前拒绝；
- CaptureSession start在没有自己mint的ACTIVE exact reservation、没有唯一DatasetBuilder claim、reservation已失败/释放或来自其它stream时均不得触达设备；join key只由`expected_cells[source_ordinal]`派生，delivered达到预算后额外read在I/O前拒绝；
- DatasetBuilder mint的ExactDatasetReadiness必须同时绑定reservation/stream generation、schema、同一event adapter与完整expected-cell permutation；只检查`materializer_bound=True`不构成arm authority；所有terminal路径断言reservation为RELEASED且stream registry为空；
- driver ring mutable alias在capture进入session后的第一步被snapshot，ordinal/captured_at/correlation/metadata/value都只读该frozen payload；原ring随后复用或改写不改变delivery；
- arm前清空software pending/driver residual并保存pre-arm residual observation；cap_start/arm-ready后、FIRE前按Q0 reset epoch建立session counter baseline，counter/stamp若仅首帧可读则必须应用Q0 implicit-initial/first-snapshot/first-frame rule；测试禁止跨cap_start reset使用旧绝对值，并让pre-fire frame、非法reset/首帧successor或stop后late frame使整epoch失败；
- EndOfStream只能由CaptureSession在mode-specific raw logical terminal/safe ack、deployment-bound CompiledPulseArtifact/H1 post-terminal tail evidence、camera保持capturing完成Q0保守drain、最终counter/stamp冻结、camera stop/stable check/buffer release、capture thread/session join ack之后mint；任何raw terminal state本身都不证明delay tail idle，达到expected N但session未终止时禁止finish/seal；
- 用户cancel撤销普通execution capability后，thread-safe interrupt必须先解除blocked read，capture cleanup再用绑定session id/binding/generation的SessionCloseCommand/Ack完成stop/drain/join；测试不得由测试线程手动release read。partial prepare/start、wrong-session、join=false/timeout、cleanup ack失败和late worker均不能返回SAFE成功或留下可seal EOS；
- minimal pipeline在software preflight完成、prepare前被cancel时只走verify-idle，不发送unknown-session close且不quarantine；同一compiled plan可顺序复用并为每run mint新generation，同时并发run在ResourceClaim处拒绝；builder.close与session cleanup双故障仍分别尝试；
- PipelineMemoryProfile只允许调用者选择总上限，固定/per-event overhead不能低报；process-local admission绑定exact chain，PipelineResult与持久artifact只记录`aggregate_peak_bytes`，不记录无裁决作用的runtime profile指纹；调用方不能把dataset A与terminal B拼成新的PipelineResult；
- resident table走`AUTONOMOUS_RESIDENT`；超resident table默认拒绝，只有单I/O owner、保守refill硬上界以及对**每个潜在seam**的足分辨率硬件时间观测/全schedule residual均通过时才发布`AUTONOMOUS_REFILLED`；无camera edge区段、tail seam或非sticky underflow无法证明时必须在fire前拒绝；
- EndAttestation按execution mode验证`AutonomousTableTerminalEvidence`或完整R-major/P-fast、session-id唯一的R×P个`ApiSegmentEvidence(PulseTerminalAck)`，再比较唯一run级`CameraRunEvidence`、`expected_trigger_total_from_completed_schedule`推导值、每个BoundSourceAssociationContract的terminal recipe、按Q0 modulus/reset语义唯一unwrap的`camera_produced_delta`、frame/camera stamp、timestamp容差、DatasetBuilder coverage和EOS；测试/manifest不得消费`scan_progress()`镜像、不得把raw DONE当physical terminal、不得为API segment伪造cursor/camera terminal/CameraRunEvidence，也不得命名成硬件measured emitted count；任一不符整run INVALID且无ScanArtifact；
- 注入drop/reorder/duplicate/counter reset/metadata gap/short read使整epoch失败；系统不声称能定位具体point，也不声称能检测metadata仍合法的等量loss+extra抵消；该剩余风险在artifact proof_class中可见；
- 所有scan数据在EndAttestation前为PROVISIONAL；只有`ORDERED_END_ATTESTED_RUN` VALID后才能commit；
- PROVISIONAL可带永久可见徽标显示，但普通Figure Save、source-specific authoritative fit artifact、CommittedTransform authority input和其它derived artifact均拒绝；显式诊断保存只能产生不可冒充权威结果的`DIAGNOSTIC_PROVISIONAL`；INVALID使queued BoardFrame/fit/save按epoch lifetime token stale；
- ScanOutputContract的validity_acceptance_policy区分cell/transport完整性与component invalidity；dead site在PRESERVE_DECLARED下随ComponentValidity成功保存，在ALL_COMPONENTS_REQUIRED/MIN_VALID_FRACTION不满足时按声明失败；
- INVALID attempt保存RunFailureRecord且默认不自动重试；显式RetryPolicy有有限次数、独立run_id/lineage，最终artifact引用所有失败attempt；
- API_SLOT_SEGMENTED_EXISTING在首次arm/FIRE前取得覆盖完整冻结R×P schedule的一个run级camera authority；每cell只生成独立pulse-session terminal lineage，不重新arm/stop camera、不重取camera authorization、不生成per-segment EndAttestation。相邻segment的deadline/cancel-aware conservative wait不得短于camera required external-trigger interval，但允许任意更长gap；最大gap/精确settle/gap-dependent use case在bind时typed拒绝。全局aggregate验证canonical segmentation_rationale、ordered segments、一个CameraRunEvidence、count/key coverage与lineage后才产生run级achieved proof，任一segment失败整run不提交；
- API segmented故障注入覆盖第N个`capture_next()`阻塞时cancel、跨segment全局deadline、pulse cleanup失败后camera cleanup仍继续、terminal后sequencer SAFE/camera idle，以及任一失败均无FINAL repository commit；测试使用deterministic clock/interrupt，不用真实sleep制造偶然通过；
- source-neutral ScanPlan对每个physical source要求具名BoundSourceAssociationContract；近期S4只允许恰好一个Q0-qualified qCMOS source，额外/非camera source在专用contract kit发布前bind时typed拒绝；
- ProgrammedImageDeploymentRecordRef缺失、非active、endpoint/fingerprint/.bit/release mapping变化，或Q0/FIRE authorization/tail evidence/artifact pin的revision不一致时Formal拒绝；测试明确证明该record只是installation assertion，不能冒充runtime content attestation；
- TAGGED/TIMELINE若相机现有metadata经Q0资格化可作为更强软件关联能力可使用，但不要求FPGA逐沿FIFO；HardwareTriggerStamp只在证据触发未来RTL修复后测试。

Thread/UI：

- blocking I/O 中 cancel；
- out-of-band interrupt 可使被占 I/O lane 进入 cleanup；
- `HAZARD_ACTIVE`持久化阻塞/失败期间cancel只置token且interrupt调用次数为0；成功后若已cancel不触碰硬件；
- interrupt in-flight是terminal barrier，cleanup不并发碰同device、claim不释放、迟到异常进入CleanupReport；
- join timeout 不允许 restart/release/destroy，safe failure 转 ResourceQuarantined；
- synchronous run 的 RunStillCancelling 保留可查询 RunHandle/claims；
- reset/reconnect 只有在 exact RecoveryAttempt 内通过 live identity/safe check并durable提交RecoveryBundle才可解除 quarantine；普通运行中不存在 reconnect；
- 新 connection generation 在 UNVERIFIED handshake 完成前不可 acquire，应用重启不洗白 sticky fatal；
- active Run内transport断开不透明reconnect；普通重连要求safe shutdown与新进程，durable blocker下的recovery-only重连产生新generation；旧run cleanup不能用新generation readback生成旧generation的SAFE receipt；
- startup open/identity/AssetMap verification/broker bind在Run admission开放前完成，不创建ResourceArbiter connection lease；任一步失败时普通硬件调用次数为零、partial graph不发布且已开子集被安全关闭；
- RecoveryClaim只针对既有unresolved refs，和普通claim完全互斥且只能执行allowlisted identity/status/safe/reset/reconnect；attempt必须显式complete/abort，journal lost-ack只重试同一attempt/bundle；recovery中崩溃/超时/journal失败后仍quarantined；
- `VerifiedPhysicalDeviceIdentity`不可变且只能由DeviceBroker握手mint并在bind时一次消费；成功后唯一长期事实是`DeviceBindingStamp(PhysicalDeviceIdentity, binding_instance_id)`。同一握手结果复用、同一PhysicalDeviceIdentity绑定两个ResourceKey、同key二次bind或静默换physical identity全部拒绝；
- identity evidence明确区分HARDWARE_IDENTITY_READBACK与INSTALLATION_ASSERTED_ENDPOINT；后者保存endpoint/AssetMap revision与剩余换板风险，不能在Q0/artifact/UI中显示成硬件serial readback；
- 真实runtime缺失AssetMap、map revision不是canonical内容digest、exact adapter kind/expected matcher不符时composition拒绝；新进程+新broker下把同role换成另一serial仍拒绝，只有旧runtime完成safe shutdown并退出后的显式offline maintenance可更新map；
- run cleanup的`SafetyProof`由RunContext消费broker签发且stamp完全匹配的`SafeReceipt`后mint；recovery使用`RecoveryEvidence(DeviceBindingStamp, safe_state_digest)`并原子写入`RecoveryBundle`。字段赋值、proof/evidence复用、设备A的值替换设备B receipt以及跨run/key/generation substitution全部拒绝，且未调用B verifier时B绝不转SAFE；
- `safe_requested`、command return、本地state/cache、缺失readback与broker补写expected generation均不能产生SAFE；每个真实adapter的live terminal verifier覆盖肯定/否定/读取失败，未知adapter在composition时拒绝；
- InstallationDeviceGraph中的每个adapter owner均有exact-type三态分类；任意未知class/subclass不能default continue；所有legacy LogicNode按全部referenced devices登记reference_keys并在shutdown前terminal，但只有真实host读取/控制进入ResourceClaim；虚拟trigger-wire等adapter内部接线不得伪造OBSERVE claim；
- qCMOS、Pylon、Remote FPGA与Manual backend各自SafeStateContract矩阵覆盖肯定/否定/readback失败/disconnect/generation-change；缺失肯定readback时Formal保持NO-GO且不制造fake寄存器测试；
- Pylon拔线fake保留缓存GetDeviceInfo且令IsGrabbing为false/IsOpen为true时，`IsCameraDeviceRemoved`或资格化live readback必须使start与SAFE mint失败、旧generation失效并进入quarantine；cleanup前一步失败仍尝试后续声明动作，但缺少全部MUST_SUCCEED ack与最终肯定readback时仍不得SAFE；
- 每个hazardous run在首次可能改变设备/输出/采集状态的configure/session-start/arm/fire/safe/abort/interrupt前，由RunController从`BoundDevice.binding_stamp`唯一派生并向同一SafetyJournal原子write-ahead全部HAZARD_ACTIVE records（非逐cell fsync）；append失败或crash后启动重放未解决hazard并阻止普通admission，切换artifact repository root不能洗白machine/device safety ledger；
- HazardRecord固定run_id、ResourceKey、完整DeviceBindingStamp与稳定record id；journal acknowledgement丢失时retry必须提交同一组records，部分已知、identity/generation变化或相同id内容冲突全部fail closed；
- session/worker/in-flight interrupt全部退出且run hardware capability不可逆撤销后，同run所有SAFE/UNSAFE决定由ResourceLease一次构造并幂等append唯一SafetyDispositionBundle；append/ack失败时全部claims保留，只能重放缓存的同一bundle，late hardware call为CapabilityRevoked，不建立多bundle聚合或额外set；
- 唯一SafetyDispositionBundle未durable时，RunHandle不发布FAILED/CANCELLED/SUCCEEDED，只显示FINALIZING_SAFETY/SAFETY_JOURNAL_BLOCKED phase并保留claims；全部SAFE时继续artifact commit，任一UNSAFE时禁止成功artifact；最终terminal与全部claim release一次可见；
- bundle构造前session/interrupt全部退出；durable后领域prepared value被丢弃，只把executed facts交给无device Port的PostSafetyContext，claims仅保留排他性；注入旧session/closure或late cancel硬件调用必须得到CapabilityRevoked且调用计数为0；
- raw SDK/driver只在allowlisted owner lane构造和保存；RunPlan/Definition/finalize的对象图、global、container与bound method均不存在driver或可直达driver的callback，验收不以closure introspection冒充隔离；
- cancellation在CommitIntent fsync期间仍可受理；intent后取消写ABORTED且publish调用次数为0；manifest replace确认丢失、COMMITTED marker确认丢失和Repository暂时不可达均保持非terminal/claim，不重复publish；startup用`safety_bundle_id + commit_id + CommitTarget/manifest digest`把pending intent唯一解析为COMMITTED或ABORTED；
- `CommitAuthority`只能由startup-reconciled RepositoryCommitCoordinator签发，是不含public publish/journal/recover的无副作用opaque handle且单次消费；直接发布、替换payload、重复/跨run消费、ephemeral journal生产签发与绕过startup pending gate全部拒绝；错误PublishedManifest类型/target/digest直接ABORTED且recover调用次数为0，只有typed PublishVisibilityUnknown进入recover，recovered PublishedManifest仍须再次匹配target/digest；
- commit reconciliation三态不可反转：wrong digest + abort-marker failure仍FORCE_ABORT且recover为0；visibility recovery已判uncommitted + marker failure仍FORCE_ABORT；validated publish/recovery + commit-marker failure仍FORCE_COMMIT且不再调用recover；
- crash发生在safety bundle、commit intent、artifact manifest、commit resolution和terminal任意相邻边界时，startup确定性恢复SUCCEEDED或FAILED/ABANDONED，不重新fire、不把temp当成功；
- terminal snapshot与剩余claim释放对竞争acquire线性化；真实adapter bootstrap缺少persistent journal时拒绝启动，memory journal只用于virtual/unit test；
- remote endpoint的physical-owner proof、journal与recovery authority位于硬件server；不同client本地journal不能洗白server quarantine，server不支持唯一owner时contract明确拒绝多入口；
- schema-affecting reconfigure 建新 generation/block_id，旧 cursor/view/fit terminal；每个 accepted ControlTopic revision 恰有一个 terminal ack；
- I/O lane 饱和 monitor 负载下，测试按冻结的 LaneFairnessPolicy 验证 `max_monitor_burst`、accepted finite 最大越过 turns、control 最大排队时间与每类 transaction deadline；超限产生唯一 terminal ack/run failure，monitor 不能 starvation finite/control，超时 driver call 不能在后台继续碰硬件，safety interrupt 不经普通队列；
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
- owner binder/pipeline validation/pulse compile 不在 GUI thread 且不持有 hardware claim；
- notebook Experiment facade 的 virtual connect -> capture -> 1D fit -> save 保持少量语句，headless 无 render extra 仍可完整运行；
- headless `fit.save()` 返回 neutral-owned `FitResultArtifactRef`、使用有界repository默认且不加载frontend.render；figure_document只需frontend.figure，只有figure()/GUI需要render/workbench extra；
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
- public capture、PulseGUI、TaskConsole、DeviceControlPort与notebook路径在claim conflict、quarantine、stale runtime/binding和InstallationRuntime CLOSING下全部fail closed；
- adapter contract tests从adapter SDK/owner module导入并由fixture在composition前保留raw spy；runtime/public/GUI tests不得为了断言底层调用从Experiment反向取得raw object；
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
- Task mid-run frame/map 通过正式 LiveDatasetSlot/DatasetRevisionRef 显示；删除 TaskOutput 后不丢功能，也不存在第二 mutable buffer；
- live calibration 先生成 CaptureArtifact，之后与 offline ref 走 byte/contract-equivalent 算法路径；
- FrameContract/SiteMap/model mismatch 拒绝 occupancy；
- required calibration model 任一失败不提交 CalibrationArtifact；
- converter 生成新 artifact 和 lineage，不覆盖旧文件、不进入 runtime。

Pulse/FPGA：

- 仓库内每个tracked pulse JSON均只使用当前`zlc_pulse.PulseDocument`并通过`load_pulse_document()`与当前codec round-trip/golden；未知schema输入由同一owner确定性`ValueError`，package/CLI/GUI中不存在历史parser、fixture、upgrade chain或迁移转换器；
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
- host encoder/coalescer 不生成超过当前 `FRAME_WORDS` 能力的 UART frame，server/upload/PreparedProgramRef guard 对 partial/oversized payload 在发送前拒绝并禁止 FIRE；contract kit如实记录当前RTL收到合法CRC oversized frame时缺少硬件零提交保证的已知边界，测试不得为满足目标合同而假设或要求新RTL。只有golden/真机证据确认该行为是既定RTL设计偏离并经H2批准后，才增加“硬件收到oversized也零提交”的bitstream gate；
- qCMOS contract kit分别保存nFrameCount累计快照与per-frame framestamp/camerastamp/timestamp的位宽、signedness、modulus、reset epoch、rollover语义：stamps逐frame modular successor，nFrameCount只做session baseline/final delta与允许batch jump的中间monotonic检查；以及工作余量内长时间零丢帧/乱序证据和可复现报告。任何unwrap多解、未声明reset、stamp duplicate/gap或counter倒退都INVALID；
- qCMOS Formal preflight readback/freeze trigger source/polarity、sensor/global exposure、ROI/binning、exposure/readout mode、arm-ready/status ack、arm-to-first-edge余量、最小active/inactive pulse width与last-edge-to-driver tail bound，并对delay/polarity/merge后的完整物理waveform验证；整run drain后的pending/late/extra frame阻止commit；
- current DONE/tail行为按现有contract测试；Formal EndAttestation由唯一I/O owner按H1冻结的mode-specific recipe取得AutonomousTableTerminalEvidence或R×P个ApiSegmentEvidence/PulseTerminalAck，高层`scan_progress()`镜像只供UI；raw state只证明logical terminal，必须经对应recipe成为physical pulse terminal。API每cell terminal后camera继续armed/draining，只有整run全部cell完成才进入一次Q0 quiet-window、冻结final metadata、cap_stop、stable复核与buffer release；不把DONE重新定义成不存在的PHYSICAL_DONE，也不伪造per-segment camera tail/terminal；
- 当前`scan_repeats=K`可能多发下一sweep point的路径有回归测试并被Formal compiler明确拒绝；finite single-pass大表走现有同步refill后到STATUS_DONE只作为transport/RTL完成性回归，不授予Formal capability，也不证明从未发生不可见stall；`AUTONOMOUS_REFILLED`仍必须独立通过§15.4强gate；
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

这里的“交互事件”必须由真实 Qt input/event 路径覆盖，而不是只断言 controller state：

- 在运行中的 raw camera panel 上创建 ROI、热修改已有 ROI/threshold 并删除下游 processor；逐 revision 观察 `ACCEPTED -> APPLIED/terminal`，同时证明 source Run、raw stream generation、raw front sequence 与 source tap topology 不重启、不回退且无 gap；
- 对 `main` salvage 清单中每种 live plot kind 执行 selector、zoom/pan、crosshair/hover 读值；shown plot 必须返回适用的非空 interaction handle，并用同 revision 的 `ViewportTransform` 验证 raster↔data 命中；
- 拖拽/调整一个 panel 期间，其它 panel 继续 present；被拖 panel 在 release 后补到最新合法 revision，不把拖拽期间的 stale front 当新 front；
- 通过同一 Setting/Edit 路径切换 `normal/tight/fixed` relim、cmap 与 limits，并验证保存/重开；
- 从 panel/figure 的 `Add Analysis -> Fit` 或 `Analyze -> Fit` 一键提交 authority draft，框选直接预填同一 draft，覆盖 1D range 与 2D box ROI fit，并证明后台求解时 live/其它 panel 仍响应、stale completion 不覆盖新 selection；
- 从通用 figure/archive viewer 载入任意已存 figure/artifact，执行 zoom/pan、re-fit 与 export；报告类 frozen multi-page raster 至少必须支持 zoom，不能以静态 PNG 通过通用 viewer 验收。

每条 E2E 都必须同时验证用户可见结果、时序与不中断项；截图只补视觉证据，不能替代行为 oracle。

virtual 与 real adapter 运行相同 Task/Measurement bind、PipelineSpec/RunPlan、artifact repository 和文件夹流程，只替换最低层 Port；adapter contract kit 使用生产 Port 的真实属性/方法名，并覆盖`ORDERED_END_ATTESTED_RUN`、CameraExternalTriggerQualification、DCAM frame metadata、timeout、buffer reuse、disconnect与health recovery。virtual adapter可以注入drop/reorder验证整run invalidation与qualification revocation状态机，但不能用fake队列代替Q0真机有限样本、统计上界与工作区间资格化证据。

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

同一个 use case 不允许双写、双读或自动 fallback；尚未迁移的其它 use case 可以暂时停留在旧实现，但不能通过 bridge 污染新合同。S0.5 只允许 host/catalog/render/resource containment，不允许把旧 Hub/LogicNode/metadata vocabulary 适配成新 event/data/runtime 类型；这叫迁移隔离，不是领域兼容层。旧 panel 可由 LegacyPanelHost 逐项托管，旧 hardware workflow 必须经过 LegacyRuntimeFence，加入同一 process-lifetime InstallationRuntime、ResourceArbiter、DeviceBroker与shutdown lifecycle gate。共享 primitive 只在首个消费它的切片中建立最小正式版本；后续若无第二用例，不提前泛化。

追溯审查判断“是否保留”时必须看终态职责和已排期 consumer，不能把中间提交的 production reachability 当成唯一标准。`MonitorTap -> MonitorDataset -> LiveDatasetSlot` 的未来 GUI live/rolling consumer、calibration/occupancy 的 S3 consumer、FitResultBatch 的 gridplot consumer 都是明确需求；它们可以因职责混杂、重复 owner 或过度包装而重构、压缩或替换，但不能仅因当前 composition 尚未接线就整簇删除。只有目标职责已有更小的唯一 owner、全部当前与已排期 consumer 都完成迁移，并有 dependency-closed 删除证据时才物理删除。

### U0：UX 修宪后的纠正顺序（先于任何新能力）

历史 S/W/M checkpoint 仍保留为“当时实现了什么”的审计事实，但其中把缺失用户面冻结为永久合同的句子已由 §2.1、§2.2 与规则 9 废止。自本 checkpoint 起，下面顺序覆盖旧 roadmap 中任何相反的“下一切片”建议；前三项纠正全部闭合前，不铸造新便利切片：

1. **Monitor/ROI 生命周期纠正。** 相机 source Run/raw front 永不因 ROI/threshold retarget、新建或删除而重启或 gap；已有 downstream processor 由 revisioned `ControlTopic` 在事务边界 `APPLIED`，新建/删除只迁移该 downstream stream/generation。以 `main:frontend/task_console.py::_apply_panel_analysis/_publish_region` 的“retarget can never gap a running consumer”为 UX oracle，并用真实 launcher E2E 证明 source run id/generation/tap topology 与 raw sequence 连续。
2. **补齐 full live interaction。** 为全部 `main` 声明可交互的 live plot kind 交付 `ViewportTransform + Qt overlay` 的 zoom/pan/crosshair/hover/selector；拖拽一个 panel 时其它 panel 不停流，被拖 panel 松手补拍；Setting/Edit 恢复 `normal/tight/fixed` relim、cmap 与 limits。shown plot 返回空 interaction handle 一律验收失败。
3. **纠正已暴露 figure/viewer/Fit 产品面并建立唯一交互 owner。** 先让现有 W4 generic `figure_gui`、W7 saved-fit grid、calibration/occupancy 已暴露路径恢复适用的zoom/pan、selector、显式re-fit与export；frozen raster只允许报告类多页使用且必须可zoom。本纠正建立后续复用的唯一 viewer interaction/Fit owner，但只验收当前已经公开的source/ref类型，不在此处预建“任意artifact catalog/browser”。panel/figure 的 `Add Analysis -> Fit` / `Analyze -> Fit` 是主入口，普通 Fit 一键提交 authority draft，selector 直接预填同一 draft，并同时交付 1D range 与 2D box ROI fit；独立 `fit_gui` 只保留为直达入口。TaskConsole的`Add Analysis`只桥接当前FINAL artifact，不因按钮命名预建formal workflow。

#### U0.1 Monitor/ROI salvage gate（开工冻结证据）

本纠正的 exact UX oracle 固定为 `main@6c337d49c7086fa0ff21f879cd159bdf0e753f51`：真实 `task_console.bat` 进入 `frontend/task_console.py`，`PanelCard._forward_area_select -> TaskConsole._on_panel_area_select -> _apply_roi_selection/_sync_fit_node -> _apply_panel_analysis -> _publish_region`。以下是改目标代码前冻结的旧行为清单；旧实现中的 Hub/LogicNode/固定 JSON buffer 只是待替换机制，不是目标依赖：

| salvage 字段 | `main` 真实行为与 U0.1 收口判据 | 开工状态 |
|---|---|---|
| 入口与控件 | Setting/Edit 共用 Analysis 控件；Selectors 打开后，下一次拖选才创建或 retarget ROI；running row 原地更新，stopped row 只更新持久值且不被删除/自动启动；没有 row 时留在 Monitor board 创建一条 Analysis row 并 Start | `PASS_CURRENT_CAMERA`；通用 TaskConsole row/Setting/Edit 仍由 U0.2/清单4交付 |
| 交互覆盖 | 旧 `region_binding` 对 2d image、1d/monitor、hist、sites/scatter 与 grid 都有具名分支；本 U0.1 只纠正当前已公开 camera rectangle ROI，但不得把其实现写成其它 plot kind 的终态上限，完整覆盖由 U0.2 接续 | `PASS_CURRENT_CAMERA`；其它 plot kind 明确 `DEFERRED_U0.2` |
| 时序 | 重拖先复用同一 region identity，再把新参数排到 running worker；worker 在 shot/transaction 边界切换，源 acquisition 不停。目标改为 drag release 自动提交 typed revision，UI 先显示 `ACCEPTED/pending`，只在 processor 边界成功后显示 `APPLIED`，不再要求第二次 `Apply ROI` | `PASS` |
| 不可中断项 | retarget 不停止 camera/source，不重建 raw history，不让可见 raw panel 空白；新建/删除只影响 Analysis/downstream。E2E 必须在操作前后证明同一 source `RunId`、raw stream generation、raw block/front、固定 source tap topology，且 raw sequence 单调继续 | `PASS` |
| 即时生效与失败 | running row 旧 UI 立即提示“已入队”，但并不证明已生效；目标必须把这种乐观提示纠正为 acknowledged `ControlTopic`。可预知的 binding/schema/预算失败在提交前拒绝；运行边界拒绝返回 `REJECTED` 并保留旧 applied downstream；被更新覆盖返回 `SUPERSEDED`；Stop/Close 使未决 revision `TERMINATED` | `PASS` |
| 新建、删除与 stopped 语义 | 首个 ROI 创建新的 downstream stream/generation，第一份同-shot scalar front 成功后才切入；Clear 终止该 downstream 并回到 raw-only front；两者都不改变 source。未来 TaskConsole 的 stopped row 仍必须保持 stopped、保存新参数但不暗自启动 | `PASS_CURRENT_CAMERA`；stopped row 明确 `DEFERRED_TASKCONSOLE` |
| 保存/恢复 | `main` 保存 panel 的 selection/region identity/reducer 与 Logic row，载入后 row 为 stopped、region 可供手动 Start 重放；当前窄 camera window 没有 workspace owner，本轮标为 `NOT_APPLICABLE_WITH_EVIDENCE`，但不得删除完整 TaskConsole 第4项的保存/恢复验收 | `NOT_APPLICABLE_WITH_EVIDENCE` |
| authority | 旧路径是 display Selection 经 per-kind binding 后发布 control signal，running worker 才拥有应用权。目标保持同一跨越动作但改成 `Selection -> Workbench mapping -> typed RoiScalarBinding candidate -> ControlTopic receipt -> downstream APPLIED`；display draft、pending candidate 与 applied scalar metadata 必须是三种不同事实 | `PASS` |
| 禁止复制机制 | 不迁入全局 Hub 名字、`registered_names()` 查重、16384-byte JSON tensor、mutable Logic row dict、latest-value join、optimistic success 文案或失败后的部分提交；不用取消 source Run 来模拟控制事务，也不建立通用 async workflow engine | `MATCHED`（设计约束，待代码 ratchet） |

收口的 launcher E2E 至少覆盖三条：已有 ROI 连续 retarget、raw-only 首个 ROI 创建、ROI Clear/Close。三条均记录操作前后的 source identity/raw sequence、control revision 全终态、downstream generation/metadata、可见 front 是否持续；故障注入还要证明候选失败不撤掉旧 applied ROI，Close 不会在窗口销毁后发布 stale `APPLIED`。`main` 中先改 panel/Hub/row 再调用 worker且无回滚的部分提交窗口不是 UX 资产，必须由 typed preflight 与 receipt 关闭。

#### U0.1 实现 checkpoint：revisioned downstream control 与 source Run 连续性闭合

当前 camera monitor 只在启动时建立一个 raw producer、raw `MonitorDataset` 与固定 processor ingress tap；此后 rectangle release、reducer 改变、首个 ROI、Clear 与候选失败都不再创建或取消 source Run。已有同输出 schema 的 retarget 使用同一 downstream generation/block，并在旧 history 仍可读时先对同一个 raw event 完成 payload snapshot、project、validate、retained-byte/digest、全部 allocation 与 shadow cell write；这些可失败工作全部发生在 downstream publish 前。stream retention 也只在 marker 前计算 eviction count，不复制 retained backlog、不提前删除旧 record；同一锁内在 marker 后才应用该 count，因此无 eviction 的 publish 为 O(1)，临时空间也是 O(1)。`AcquisitionStream.next_sequence` 是 publish 的第一处权威 mutation：未推进时才能丢弃 shadow、`REJECTED` candidate并由旧 binding 消费同一 raw event；一旦推进，任何 publish/monitor-offer/finalize 异常都 terminalize 该 scalar generation、降为带精确原因的 raw-only state，绝不伪造旧 binding 的 N+2 fallback。terminal fanout 逐 tap 隔离并在 `finally` 唤醒 stream waiter，单个 tap 的通知边界损坏不能让其它 tap 永久等待。正常 publish 后只绑定 exact envelope、交换预写 owner，再连续更新 binding/control state 与 `APPLIED`。跨 schema 创建新 downstream generation，Clear 只终止 downstream；所有路径都保持 source `RunId`、raw block/generation、固定 tap topology与单调 raw sequence。

`ControlTopic` 是无自有线程的 bounded latest-wins 原语：至多一个 inflight 与一个 latest pending；每个已接受 revision 恰有一个 `APPLIED | REJECTED | SUPERSEDED | TERMINATED` 终态。candidate snapshot 在接受 revision 前完成；processor 只在 source-shot 事务边界 claim 并 ACK。cancellation 可以在 projection/shadow 完成后、权威 publish 前获胜；一旦 publish/finalize 成功，binding、control state 与 `APPLIED` 构成同一个不可取消段，稍后到达的 Stop 只能在下一 checkpoint 取消 source Run，不能产生“candidate metadata 已提交但 receipt 却 TERMINATED”的半事务。GUI 的终态 receipt 无论展示成功与否都只 fold 一次；Stop 恰好落在 data-plane `APPLIED` 与 owner-thread drain 之间时，detach 前会从 receipt 与缓存 state 恢复最新 authority request，下一次 prepare 不退回旧 ROI。新 Run attachment 的 initial receipt drain 还受 slot dataset-bound gate 约束，不能把 attachment-before-bind 的正常窗口记录成假故障。

downstream 自发故障不再冒充 source failure：`CameraMonitorRoiState.state_revision` 独立于 control revision，允许同一已应用 revision 从 scalar branch 原子降级为 raw-only并携带 bounded `failure_reason`。Workbench 比较 state revision，保留原 selection 为 draft、切回 raw presentation并显示“raw source continues”；若 control rejection/data故障与 presentation reconfigure 故障同时发生，诊断同时保留 acknowledgement 与 presentation 两条因果，不能用 generic render error 覆盖物理/data-plane 原因。后续 raw event继续进入同一 source dataset。display notification 回调异常走独立 first-wins `notification_failure`，永久熔断坏通知但不 close dataset；Live controller 将其转为 local render fault。Board 的 presentation gate 在同一线性化边界内先安装 Live fault/source-withdrawn 事实、再撤销 pending/port capability：已 dequeue 的完整 board 只能整体先于该事实 present，事实一旦可见就不存在随后可晋升的 pending frame；所有 presenter callback 与撤销路径统一为 `Board gate -> Live` 锁序，没有 `Live -> gate` 反向等待。最后 coherent front保留，selector/reducer/Clear禁用；只有真实 source failure 才撤销 source front。

展示切换也有单独事务边界。panel topology不变的 retarget复用同一 `BoardModel.layout_generation`；1↔4 panel 才同时 stage Qt layout、BoardModel与 live configuration。任一同步重配置失败按精确 identity 撤销 Board/Qt stage，随后让 live 进入 sticky freeze：先使未来 render job stale，再撤销 `BoardPublishPort` token、inflight work 与尚未 present 的 pending frame，但不清 presenter/最后 coherent front；已通过 currency check 的旧 render job 还必须在置 fault 时于同一 Live lock 复核 configuration+port，不能毒化新配置。所有 presentation failure 统一进入一个 first-wins fold，先幂等 freeze，再同时保留 data/control、presentation 与 freeze 自身异常，不允许 selector gesture结束后把 sticky failure误 resume。失败后不尝试恢复可能已被后续 target 覆盖的旧 live configuration，终态 receipt只消费一次。APPLIED 后新的 coherent board尚未到达时，旧 front保持可见并明确标为 `WAITING · previous front retained`，不得贴上新 projection、applied overlay或diagnostics；只有匹配新 binding/control revision 与同一 BoardFrame sequence 的 front 才晋升 `VISIBLE`。Stop/close 清掉 Qt front 时也同步清 visible fingerprint/projection marker并立即把标签降为 `no coherent front`；同 topology 重启必须由自己的第一张 coherent front 重新取得 applied overlay 与可见标签。

内存准入继续在 arm 前覆盖 raw 固定 tap、当前 downstream、topology migration 的第二 branch、同-schema shadow history与 projection/evaluation/raster scratch；shadow复用既有“同时容纳两 branch”的峰值而不是运行时突破预算。原始 `(R, P, *data_shape)` 与 ComponentValidity始终留在 raw DataBlock；只有显式 rectangle `Selection + ReductionMethod + ValidityPolicy` 产生 scalar downstream，没有 `reshape(...)[0]`、flatten或按 rank 猜轴。

本 checkpoint 的活动窄回归为 194 passed；完整 manifest 一次性 collect 86/86 文件、1117 tests，再按 process-lifetime InstallationRuntime 边界逐文件独立运行，结果为 1114 passed、3 expected skipped。Rule-6 对九个实际 production owner 相对 parent 机械计数：`9684 -> 12537 physical`、`8795 -> 11433 nonblank`，为 `1.2946x/1.2999x`，净增 `2853/2638`；classes `85 -> 100`、dataclasses `37 -> 47`、enums `2 -> 3`、functions/methods `484 -> 587`。净 physical 对 main 的严格 `AreaSelector + RoiProcessor + LiveLive/LiveLiveDis` 577 行包络为 `4.94x`，触发并完成了压缩审查；该窄包络没有 acknowledged control、terminal receipt、shadow rollback、state failure notice、coherent staged board或Stop/APPLIED race。对包含真实 PanelCard/monitor presentation 的 1133 行完整用户面包络为 `2.52x`。新增15类逐项归属为 control authority 8、data branch/state/shadow 3、live configuration/candidate/job 3、UI pending receipt 1；删除任一组都会重新合并 publisher/consumer authority、pending/applied truth、worker generation或GUI receipt ownership。没有新增 manager、executor、plugin、registry、service locator、compatibility wrapper或第二套 renderer；未发现可再合并的一成员 enum/单消费者 wrapper。两路最终独立对抗审查均为 `P0=0/P1=0/P2=0/GO`。U0.1 current camera rectangle scope据此为 `GO`；完整迁移仍为 `NO-GO`，下一项严格回到 U0.2 full live interaction，不提前铸造便利切片。

#### U0.2 full live interaction salvage gate（开工冻结证据）

本纠正继续固定同一 exact UX oracle：`main@6c337d49c7086fa0ff21f879cd159bdf0e753f51` 的真实 `task_console.bat -> frontend/task_console.py`，交互算法证据来自 `frontend/selectors.py` 与 `frontend/live.py`。下面清单以可执行代码为准；旧注释若与代码冲突，不作为权威。旧 `EmbeddedFigureCanvas`、Matplotlib selector artist/callback、共享 Figure、`_zlc_interacting/_beat_owed`、Hub/LogicNode、mutable `params` 与 console-wide RenderLoop 只解释旧行为，不迁入目标 DAG。

| plot kind | `main` 已挣得 interaction handle | zoom/pan 与读值 | relim / cmap / limits | 开工状态 |
|---|---|---|---|---|
| 2D | area + cross + zoom/pan + 双 horizontal clim line | wheel：down 放大、up 缩小；middle drag 双轴 pan；middle double-click 优先 zoom 到 area，否则 home；right click 固定 x/y/z，right double-click 清除 | relim 控 clim；cmap、x/y view pin、color limits | free-running camera 与 finite exact capture 均 `MATCHED`；其它generic 2D launcher仍逐consumer验收 |
| Sites | area + cross + zoom/pan；有 background frame 才有 clim line | 双轴；旧固定 cross 只有 x/y，没有伪造 occupancy/frame z | background clim；无 frame 时 no-op；cmap、x/y pin、color limits | `MUST_CLOSE` |
| 1D / Monitor | area + cross + zoom/pan | 只改 x view；固定 cross 显示 x/y | relim 控 y；x pin；无 cmap | camera rolling Monitor 与 progressive SCAN_POINT 均 `MATCHED` |
| Histogram | area + cross + zoom/pan + 零到多个 vertical threshold line | 只改 x view；threshold drag 与 area 排他，实时更新统计 | relim 控 count-y而非 bin-x；x pin；无 cmap | `MUST_CLOSE` |
| Grid | 每个 cell 与 focus view 都有该 subkind 的完整 handles；hist cell 另有 threshold；left double-click focus、Esc/unfocus | 随 cell family；thumbnail 与 focus 使用同一交互 owner | image grid 继承 cmap/relim；display state同时作用 thumbnail 与 focus | `MUST_CLOSE` |
| Pulse | seed/static preview 的 area + cross + zoom/pan | 只改 x；固定 cross x/y | relim no-op；x pin | 不扩大当前 live-addable 范围；在 Pulse 产品纵切复用同一 owner |

精确鼠标与时序合同如下：

1. `Selectors` 默认 OFF，但每个已显示且可交互的 plot 已建立非空 handle；OFF 时 wheel 冒泡给 board scroll，ON 时原地 arm area/cross/zoom/drag，不重建 panel、source、Edit 或 renderer。Area 是 left drag，拖动中 rectangle、handles 与 endpoint label 可见；退化矩形清空。Cross 是 **right click 固定读值**、right double-click 清除，不是 pointer-motion hover。Zoom 是 wheel centered zoom、middle drag pan、middle double-click zoom-to-area/home；pan 必须以 press 时的 pixel delta和冻结 limits 为基准，不能在不断变化的数据坐标中反馈累积。
2. 任一 area/clim/threshold/pan drag 只冻结被操作 panel 的可见像素；source、processor、完整 coherent BoardFrame 与其它 panel继续前进。旧实现把该 card 标 interaction、tick 只欠该 card 一拍，release 后在下一可用 tick 合并到最新 revision，不回放中间帧。目标可在 release 的同一 GUI turn 露出已到达的 latest，但同样只能 coalesce 到最新，不能把冻结期间的每帧排队。
3. 当前设计冻结的 `hover` 是相对 `main` 的明确新增能力，而不是伪造的旧行为：它定义为 selector armed 时的瞬时 x/y（2D在 exact sampled cell 可再显示 z）overlay，pointer 离开/disable/hide即消失；right-click cross仍是可锁定、可双击清除的独立状态。hover 与 cross 都必须使用同一 front revision 的 `ViewportTransform`，不能读 worker Figure/artist或按 rank 猜坐标。
4. TaskConsole 默认 relim 是 `tight`，选项顺序 `tight / normal / fixed`。普通曲线：`tight = min/max ± 10% span`；`normal` 对全非负数据锚定0且上界约 `1.2*max`，含负数时使用 tight 数值但保留 normal mode；进入 `fixed` 时冻结当前显示范围。Histogram 的 normal/tight 都锚定 count=0但 headroom不同。所有自动范围保留旧 deadband/hysteresis，避免每帧微抖；固定 lo/hi 在非 fixed 时可编辑但不改变 view。
5. Setting 与 Edit 只是同一个 revisioned display state 的两个表单投影：实时值只更新空字段 placeholder，绝不覆盖用户已输入文本。x pin适用所有有 x view 的种类；y pin只适用 image view family；color limits只适用有 value/clim 轴的2D/Sites/image-grid。Histogram y是 relim data axis，不是 y-view pin。image color limits 与 fixed clim 是同一 authority，禁止另设第二份 clim 状态。

复用裁决也在开工前冻结：Fluent widget/container、`FormSpec + FluentParameterForm`、Qt/QSS style token与纯 render token已在 current owner中，必须扩展复用；手势数学与单写者语义从旧实现 **ADAPT** 到 `zlc_frontend.selector`、`QtRasterBoard`、current Figure/View document和 Workbench command seam。禁止迁入整类旧 `PanelCard/PanelEditor/PanelConfig/_PanelBoard`、Matplotlib selectors、旧 RenderLoop，禁止新建第二套 Qt component library、form parser、palette、renderer、mailbox、plot-kind registry或 Workbench-owned Selection truth。interaction capability必须从 current `ViewIntent/ViewContract + evaluated axes` 推导；Setting/Edit 必须投影同一 typed display state。

##### U0.2a 首个纵切：camera IMAGE 的 per-panel interaction hold

首个实现切片只关闭“拖一个 panel 导致整 board 停流”这一条真实回归，不宣称 U0.2 已完成。它以当前公开的 camera IMAGE + CURVE/HISTOGRAM/METER 四 panel board为真实 consumer，并建立后续所有 drag共用的呈现边界：

- `BoardController` 继续只接受、校验和原子 present 完整 coherent `BoardFrame`，绝不合成 mixed-revision frame或变成 panel scheduler；
- Qt在 pointer press 时只保留目标 IMAGE 的 prepared bytes/QImage和小型 exact origin（board/layout/sequence、source、`PanelPresentationIdentity`、viewport revision、coherence group与 raster geometry）。底层 `_front` 仍替换为最新完整 BoardFrame；主 paint loop 对每个panel只做一次blit，目标cell选择held IMAGE，其它cell选择latest，随后只叠加小型 `DISPLAY_ONLY` badge。因此 scalar panels立即继续、目标视觉冻结，release丢hold后只露出已存在的latest coherent IMAGE；
- gesture必须由 held origin生成并在 hold仍存活时同步提交；latest sequence 前进不是 stale。只有 board/layout、panel slot、source、presentation identity、viewport或geometry改变才 stale并 fail-closed cancel。stage/promotion、clear/invalidate/fault、resize/hide/deactivate/UngrabMouse、disable/unbind/rebind、close/DeferredDelete全部释放 hold；gesture callback失败也在 `finally` 释放；
- Workbench不再用 `gesture.sequence == board.front_frame.sequence` 判断正常 drag，也不再从 selector interaction调用 whole-live `pause/resume`。drag中照常 reconcile fault、present上一候选、admit下一 snapshot。Qt leaf自己拥有 hold，不再发布无消费者的 begin/end observer、只读诊断wrapper或额外 owner wake；目标 panel 内的紧凑 badge必须诚实区分 held sequence 与 live latest sequence；
- arm前内存预算在既有 render scratch + candidate + visible front之外再计一份 `target stride_bytes * height`。hold不得保存整个旧 BoardFrame而滞留其它 panel raster；Qt `QImage`只借 held immutable bytes，释放不依赖 consumer callback成功。

本纵切的独立 oracle必须同时证明：drag期间 source Run/generation/tap topology不变且 raw sequence继续；底层 Board sequence及curve/hist/meter可见像素继续；held IMAGE bytes/origin不变；release同一owner turn露出latest且ROI按 held viewport正确提交；layout/source/presentation/geometry/fault/clear/resize/hide/disable/close各退出路径释放；少一份 held-raster预算时preflight拒绝、补足后通过。完成该切片后，U0.2仍按 2D完整 A/C/Z/H+display state、1D/Monitor、Sites、Histogram、Grid 与 launcher contract 的纵向顺序继续，不能把 rectangle hold当作其它 kind的终态实现。

**U0.2a 实现 checkpoint（GO）：** camera consumer已删除whole-live `pause/resume`、write-only interaction observer及额外owner wake；不可逆的`freeze_presentation()`只保留给故障隔离。Qt hold是一个frozen exact-origin值，只借目标IMAGE的immutable bytes/QImage，不持有旧BoardFrame；正常latest BoardFrame继续原子晋升，main paint loop每panel只blit一次，press立即请求重画，release/cancel直接露出latest。source/presentation/layout/viewport/geometry改变和stage/clear/resize/hide/deactivate/UngrabMouse/disable/unbind/rebind/close均fail-closed释放，callback重入/异常由同一`finally`收口。arm前预算精确多计一份GRAY8 target raster，缺这份预算的反例在硬件arm前拒绝。

活动直接回归按process-lifetime installation边界分九个进程为94 passed；完整manifest收集86/86文件、1122 tests，逐文件独立进程实跑为1119 passed、3 expected skipped。Rule-6对四个实际production owner相对parent的机械计数为`4593 -> 4658 physical`、`4294 -> 4354 nonblank`、classes `12 -> 13`、dataclasses `6 -> 7`、functions/methods `166 -> 168`，整体仅`1.014x/1.014x`；最大单文件是Qt board的`919 -> 1048 physical`（`1.140x`），无约3倍项。唯一新class/dataclass `_HeldPanelFront`跨press/present/paint/release/semantic validation/lifecycle有真实消费者，并以小型origin取代整BoardFrame retention；同时删除reversible pause状态、两个public pause/resume方法、camera同步包装与无消费者诊断property，没有新增enum、scheduler、mailbox、renderer、manager或兼容层。两路独立对抗终审在关闭UngrabMouse泄漏、窄panel badge越界、2.3MP双blit、write-only observer及press不立即repaint后均判`P0=0/P1=0/P2=0/GO`。该GO只关闭U0.2a，不改写完整U0.2与迁移全局仍为NO-GO。

##### U0.2b 纵切：free-running camera monitor IMAGE 的完整 A/C/Z/H 与单一 display state

本纵切在U0.2a的per-panel hold之上只建立一条presentation-authority路径，不把Qt手势、worker raster或Setting/Edit各自变成状态owner。`ImageDisplayState(revision, relim_mode, colormap, x_view, y_view, fixed_color_limits)`是唯一authoritative display value；`ImageViewportTransform`只是该值在一对具名`SPATIAL_Y/SPATIAL_X`规则pixel轴上的纯投影，二者revision必须完全相等。所有display变化都由camera Workbench owner做一次compare-and-set，再调用既有`LiveBoardController.reconfigure_image_display(state, viewport)`；该方法只替换panel presentation revision和可覆盖的latest render candidate，不改变RunHandle/RunId、source block/stream generation、`LiveDatasetSlot`、raw dataset head、BoardModel topology或ROI downstream generation。live先接受，Workbench才发布新state；owner revision已前进而worker尚未画出新front的窗口中，selector switch保留checked意图但同步disabled，Qt board fail-closed disarm，只有exact新revision真正paint后才自动re-arm，旧pixels不能再author第二个intent。Start已经冻结state但view尚未attach的短过渡明确禁用Apply，避免待建live与GUI形成两个revision真相。Stop/Start保留authored state并用新schema轴重新纯函数派生viewport，但每次selector都回到OFF；若新schema拒绝旧coordinate pin而使stopped prepare进入NOT READY，用户清除/改动pin后由同一owner重走既有prepare，绝不顺带arm Run或另造recovery manager。

Qt leaf只发送两种typed intent：`ImageViewportCommit`与`ImageColorLimitsCommit`，共同携带exact `PanelInteractionOrigin(board/layout/sequence/source/presentation/EvaluatedInput)`；Workbench要求该origin仍是当前真正painted（drag时为held）IMAGE且其panel revision等于current display revision，stale intent在live reconfigure前拒绝。wheel DOWN centered zoom-in、wheel UP zoom-out；middle drag按press pixel delta与冻结viewport平移，middle double-click在已有area时zoom-to-area、否则home；right click从exact retained sample锁定cross，right double-click清除，点被zoom出当前view后只裁掉线/点而保留右上锁定读值；selector armed时hover从同一exact sample/viewport给出x/y/z，stationary pointer在新front上重取exact值，leave/disable/hide即清。A的snapped endpoint标签在draft和applied阶段都持续，任意非cell-aligned zoom及1×N/N×1轴也不借shape猜语义。H rail以当前clim为可操作domain、raw min/max只在clim内作guide，drag显示候选low/high但release才发送limits intent；Qt绝不临时改本地LUT建立第二份clim state。同步拒绝立即解除pending；已接受revision的worker terminal fault只用exact-origin discard释放该pending，不能误清后续intent。A/pan/H继续只hold目标IMAGE，source、完整BoardFrame和scalar siblings照常推进；四角overlay固定为ROI左上、cross右上、H候选左下、hold诚实徽标右下，互不覆盖。

IMAGE raster改为owned immutable `INDEXED8`：code 0只表示invalid，1..255由worker**直接跨当前effective clim量化**finite有效值；不能先跨raw全范围压成8 bit、再企图用Qt LUT恢复窄clim中已经丢掉的级别。clim外像素只在图像上饱和到端点code且不进入distribution histogram，raw `data_range`仍单独保留；worker同时冻结255-bin in-clim code histogram、与clim无关的256项ARGB base palette、actual color limits、完整未降维`EvaluatedImage`、exact input与viewport为`ImagePanelPayload`。Qt只在所有source/presentation/viewport/geometry检查通过后detach一份QImage、直接安装base palette并按viewport裁真实source pixels；clim变化由新revision重新量化pixels，绝不做第二次palette remap。H rail也委托raster owner的同一个`indexed8_code_for_value`，不能用round/floor不同的第二公式产生相邻LUT档错色。hover/cross永远读取exact value/validity plane而不从lossy code反推。`tight`取当前finite min/max，`normal`对非负值锚0、含负值退为tight数值但保留mode，二者沿用deadband并在正负号切换时重建正确锚点；`fixed`冻结用户当前painted payload的limits，不能从可能领先的`LiveFrontStatus`抄值。六个closed cmap由唯一render-style owner采样；root `zlc_frontend`仍不导入Qt或Matplotlib。极端整数若在float64 display range中不可区分、或finite endpoints形成infinite display span，都显式要求display transform，不静默制造相同code。

Setting弹窗与Edit页是同一个`FluentRevisionedFormEditor`的两个实例，消费同一个immutable `FormSpec`和handler registry；没有第二parser、EditorSession或runtime owner。Setting复用既有`FluentPopup`，按button所在screen选择上/下有空间的一侧并把完整frame夹在其`availableGeometry`内，副屏和底边按钮都不能把表单开到屏外。空color字段只显示当前painted limits placeholder，不能覆盖typed draft。成功Apply用exact base revision确认提交者并清dirty；另一份并发dirty draft保留原文并标stale，Cancel从current owner全量重载；semantic no-op不增加revision、不触发live reconfigure。进入fixed必须取得`QtRasterBoard.visible_image_payload()`，因此drag hold、old front retained或status ahead时冻结的都恰是用户眼前范围。U0.2b当时只把finite capture接到同一INDEXED8 payload；其完整A/C/Z/H/Setting/Edit现由U0.2f收口。Qt对缺payload的INDEXED8继续fail-closed，而不是恢复灰度alias或猜palette。

内存公式显式覆盖candidate evaluation之外的bool finite/in-clim mask、唯一float64 normalization workspace、uint8 raster与in-clim histogram sample、immutable RasterBuffer bytes、`setColorTable()`触发的Qt detached pixel plane、固定64 KiB histogram/palette对象余量，以及camera live的latest+held两份raster和exact value/validity/axis plane；hold仍不保存整BoardFrame。1200×1920复核profile在同机的uint16/float64 median约`33.1/35.6 ms`、tracemalloc peak均约`24.17 MiB`，每次输出2.304 MP immutable bytes且histogram count与in-clim valid逐项对账；live latest+held总estimate约`46.97 MiB`（另加caller的candidate evaluator预算）。这只证明当前纯raster kernel，不冒充完整GUI p99、持续60 fps或真qCMOS资格。

Rule-6以parent `2951ec6`的13个实际production owner机械计数：`6710 -> 10117 physical`、`6217 -> 9349 nonblank`、classes `26 -> 35`、dataclasses `15 -> 21`、enums `2 -> 4`、functions/methods `229 -> 336`，整体为`1.508x/1.504x/1.346x/1.400x/2.000x/1.467x`。曾超过门槛的`selector + image_view`已由删除test-only coordinate wrappers、public sample signals/getters与重复provenance后收敛为`300 -> 783 physical (2.61x)`、`266 -> 682 nonblank (2.56x)`、functions `12 -> 35 (2.92x)`；没有约3倍项。对`main`旧树的严格同功能去重包络（`frontend/selectors.py`的A/C/Z/H、`frontend/live.py`的2D/clim/relim、`frontend/task_console.py`的Setting/Edit/display/selector接线）为`2335 physical / 2184 nonblank`，本轮净增`3407/3132`是其`1.459x/1.434x`；再纳入旧worker/coherent tick、PanelCard state/render/teardown与RenderLoop的公平用户面包络为`3625/3419`，对应`0.940x/0.916x`。严格包络尚不含typed revision/CAS、stale-front拒绝、任意及singleton轴viewport、immutable INDEXED8 front、峰值内存准入、schema recovery与dirty-draft线性化，因此不能为压行数删除这些已被真实竞态挣得的边界。

新增class只支付closed relim/cmap值域、单一display authority、一个纯viewport transform、一个private exact overlay sample、Setting/Edit共用editor、raster+sample原子payload和两种typed intent/exact origin；两个新增enum分别是三态relim和六项closed colormap，不是一成员枚举。删除任一组会恢复字符串分派、状态多owner、lossy读值或stale intent。没有新增manager、executor、renderer、registry、plugin、compatibility wrapper或第二套selector/display framework。冻结树的语法编译、diff-check和headless root import ratchet通过，`zlc_frontend`根导入不加载PyQt/Matplotlib；活动manifest 90/90文件收集`1211`项，逐文件新进程实跑为`1208 passed + 3 expected skipped`、零失败/错误并与collect精确闭合。authority/correctness、performance/Rule-6与UX三路独立终审均为`P0=0/P1=0/P2=0/GO`，其中最后一路关闭了rail与raster相差一档的重复量化公式。U0.2b据此只对free-running camera monitor IMAGE判`GO`；finite capture后来由U0.2f闭合，纠正2仍须继续完成live Sites、generic Distribution/Grid与真实launcher contract，不能再造第二套实现或把局部GO外推成完整U0.2。

##### U0.2c 纵切：1D / Monitor curve（Monitor + progressive SCAN_POINT `GO`）

本纵切继续固定 `main@6c337d49c7086fa0ff21f879cd159bdf0e753f51`。旧行为的具体证据是 `frontend/live.py::BaseLivePlot/Live1D/LiveLive/LiveLiveDis`、`frontend/selectors.py::AreaSelector/CrossSelector/ZoomPan/attach_interaction` 与 `frontend/task_console.py::_build_plot/_render/set_selectors_enabled`；current 的两个真实产品消费者是 camera ROI scalar 的 `MONITOR_HISTORY -> CURVE` coherent board，以及 occupancy progressive scan 的 `SCAN_POINT -> CURVE` provisional board。纯函数或合成曲线测试不能替代后者。两个 dependency-closed 提交现均已完成：rolling Monitor 与 progressive scan 共享同一个 curve display/interaction owner，同时保留各自 newest-first 与 finite exact point 语义；本小节据此整体 `GO`。finite capture后来由U0.2f闭合；本小节本身仍不能外推为Sites、Histogram、Grid或launcher已完成。

| salvage 字段 | `main` 真实行为、已识别旧缺陷与 U0.2c 收口判据 | 当前状态 |
|---|---|---|
| 数据语义 | 普通 1D 每次替换完整曲线；Monitor 每个真实 source publish 把一个标量放到 newest-first index 0，并用 `Shots ago` 表示 age。旧 Monitor 在 render tick 内 `np.roll`，故 drag/Pause/render overload 或两个 tick 间多 publish 会静默跳样本；目标唯一真相是现有 `MonitorDatasetSnapshot(revision, coverage, EventRefs)`，UI/display revision、重复重画和无关 raw publish永不追加样本 | `CURRENT_DATA_OWNER_PASS`；旧 render-driven rolling明确禁止复制 |
| 真实产品闭环 | camera ROI scalar 是唯一 rolling CURVE；progressive occupancy scan 是当前真实 SCAN_POINT 1D。两者必须消费同一 `CurveDisplayState/CurveViewportTransform/CurvePanelPayload` 与同一 worker-affine `SinglePanelAggRenderer`，但分别保留 rolling newest-first 与 finite scan point 的数据语义 | Monitor 与 SCAN_POINT 均 `MATCHED` |
| Selectors ON/OFF | 旧 dashboard 默认 OFF，但 Area/Cross/ZoomPan 三个handle已建立；ON/OFF原地park/arm，不重建panel、dataset、renderer或selection，OFF时wheel归board scroll。目标保留全局用户开关，但readiness/pending/fault按panel隔离：curve等待新revision不能禁掉仍合法的IMAGE | Monitor 与 SCAN_POINT 均 `MATCHED` |
| Area | 旧画 `[xmin,xmax,ymin,ymax]`，最终1D Selection只消费x且纯水平拖被误清。目标按 `UX-002` 建原生x-span：left drag仅要求非零x宽，overlay可铺满当前y绘图区；candidate仍是DISPLAY ONLY并携exact scalar input/axis/front origin，未来Fit只可由authority seam重新构造请求 | Monitor 与 SCAN_POINT 均 `MATCHED`；相对旧二维视觉的原生x-span仍记 `LEDGERED_PENDING_APPROVAL UX-002` |
| Cross / hover | right click锁定连续cursor x/y，right double-click清除；cross不是sample snap。goal新增hover必须读真实值：瞬时hover吸附最近的valid sample并显示series，不得把pointer y伪装成信号值；因此front payload可**借用同一份immutable evaluated curve arrays**，不是复制或第二数据真相，并必须计入latest/held生命周期预算 | Monitor 与 SCAN_POINT 均 `MATCHED` |
| Zoom / pan / home | wheel down按`1/1.1` centered zoom-in，up按`1.1` zoom-out；middle drag只改x且以press pixel/frozen limits为基准；middle double-click有area时zoom-to-span，否则回当前home。home由本次完整声明x轴的finite numeric domain求得并随domain revision更新；Clear x pin立即回当前auto并继续follow，不能保留旧画面 | Monitor 与 SCAN_POINT 均 `MATCHED` |
| 轴与provenance | CURVE不能复用`ImageViewportTransform`的规则pixel/SPATIAL_X/Y/cell-snap语义。`EvaluatedAxis`必须保留owner `AxisSpec.role`，curve保留value unit；交互只接受有限numeric、严格单调递增或递减（含不规则）x，单点有确定性非退化home；非数值/非单调仍可静态显示但interaction fail-closed，绝不猜index轴 | Monitor、SCAN_POINT interactive资格与通用静态fallback均 `MATCHED` |
| relim / limits | y不是viewport轴；只由共享closed `RelimMode(TIGHT/NORMAL/FIXED)`管理。tight=`min/max ±10% span`，常量5=`4.5..5.5`、常量0=`-0.1..0.1`；normal全非负=`0..1.2*max`且全零=`0..1`，出现负值使用tight数值但保留normal mode；fixed进入时冻结exact painted y limits。所有series的valid finite union共同决定y，空/invalid不伪造0、不回退旧样本；deadband baseline只在对应BoardFrame真正publish成功后推进，ROI同generation retarget也按document/control semantic identity重置 | Monitor 与 SCAN_POINT 均 `MATCHED` |
| worker / geometry | Figure/Axes/Artist继续只属render worker。worker严格按“更新全部series artist → 计算/应用x与y → draw → 冻结最终Agg axes bbox与actual limits → immutable raster+payload”执行；Qt使用draw后的axes bbox从Matplotlib bottom-origin换算到raster top-origin，不能把带标题/刻度/legend margin的整panel当plot rect | Monitor 与 SCAN_POINT 均 `MATCHED` |
| per-panel hold | 任一curve area/pan只保留该panel的immutable RGBA front、exact origin和同front evaluated arrays；底层完整BoardFrame、source和siblings照常前进，release直接露出latest retained dataset，不回放display frame。source/layout/presentation/axis/geometry/revision/fault/disable/hide/close都释放；同一时刻只允许一个pointer hold | Monitor 与 SCAN_POINT 均 `MATCHED` |
| Setting / Edit | curve唯一authored value是`CurveDisplayState(revision, relim_mode, x_view, fixed_y_limits)`；history capacity不在display state，扩大retention必须走source request。Image与Curve只共享已出现第二consumer的有限数值range/relim纯函数，以及一个参数化的revisioned Fluent form widget；不建立generic DisplayState、Viewport protocol、InteractionManager、payload plugin registry或RangePolicy类族。Setting/Edit各实例投影同一owner，dirty/stale/CAS与runtime placeholder不互抄状态 | Monitor 与 SCAN_POINT 均 `MATCHED` |
| 多维与多series | FigureEvaluator已解析的每个series/batch轴全部保留；禁止flatten、series 0、隐式mean或把trailing data axes塞成单个`data_dim`。交互curve要求全部series共享逐值相同的x axis，否则fail-closed；relim使用全部series，hover明确标series | Monitor 与 SCAN_POINT 均 `MATCHED` |
| Monitor coherence | camera current/curve/histogram/meter仍来自同一个scalar DatasetRevisionRef；latest invalid明确显示invalid，不回退；gap/coverage可见且curve不伪造timestamp或猜补点。show-dist/history/relim/selector/display改变都不能清空或追加rolling dataset | Monitor `MATCHED` |
| 内存 | pre-arm显式合计evaluator、persistent Agg/artist、candidate、latest raster、Qt detach、display-revision overlap，以及一次pointer hold的最大`RGBA raster + exact evaluated curve arrays`；一个board同时只hold一个panel，因此各可交互panel的hold取max而非求和。模糊raster multiplier即使数值偶然覆盖也不能替代具名hold项；少一字节的反例必须在arm前拒绝 | Monitor 与 SCAN_POINT 的精确边界均 `MATCHED` |

**U0.2c combined 实现 checkpoint（`GO`）：** `CurveDisplayState`是camera Monitor与progressive Scan Workbench各自唯一的authored display owner，`CurveViewportTransform`是worker draw后冻结的坐标投影，`CurvePanelPayload`把同一front的`EvaluatedInput`、全部series、单位与draw geometry原子交给Qt；Qt与renderer都不反向修改display state。Image/Curve只共享`display_range.py`中的有限range/relim纯函数、popup placement、runtime range placeholder与直接参数化的`FluentRevisionedFormEditor`生命周期；camera与scan的Setting/Edit均消费这些同一owner helper，旧重复实现已物理删除，没有alias、兼容wrapper或第二套表单框架。`SinglePanelAggRenderer`仍是唯一worker-affine曲线renderer，Figure/Axes/Artist没有越过线程边界。

数据合同保持axis-total：`EvaluatedAxis`携owner声明的`axis_id/name/role/unit/indices/coordinates`，`EvaluatedCurve`保留value unit、逐sample validity与完整batch series；底层仍是`(R,P,*data_shape)`与对齐的`ComponentValidity`。CURVE evaluator若仍有未处理的cell/data axis就fail-closed，绝不使用rank、singleton或length猜role，也没有flatten、`reshape(...)[0]`、series-0显示或隐式mean。progressive默认以声明的`SCAN_POINT`作x，repeat按明确view reduction处理；site数不超过`CURVE_CONTRACT.maximum_batch_series=32`时可保留batch series，真实virtual 35-site产品按稳定策略显式`Select(site=0)`并把该选择写入投影摘要，不把第33条以后静默丢掉。其它有信息的component轴同样必须显式select，完整多维source与validity仍留在exact preview/final artifact。交互producer要求全部series共享exact x axis与value unit；x/y单位同时进入hover/cross overlay。有限numeric且严格单调递增/递减的轴进入interactive path；非numeric或非单调轴仍由同一renderer静态显示，并带明确unavailable reason、无空壳interaction handle、不猜index轴。

progressive renderer有一条专用capacity-one preview lane，而不是通用async execution engine：exact slot可连续发布，worker只保留latest candidate；candidate ownership一直占用到GUI owner真正完成`present_pending()`，不能在publish与present之间提前释放并让下一revision覆盖。worker缓存最新`EvaluatedFigureData`，display-only r0→r1在同一renderer、同一Run与同一source identity上重画，不重新freeze source、不重新transform、不重启实验；terminal同revision仍可提交最后一张完整front。FINAL raster只有在preview source terminal、worker close、candidate/front release与board retirement完成后才替换provisional面。general executor即使只有一个worker也不会被长驻preview watcher饿死；cancel-before-start、close、render fault与post-safety failure均使slot/worker/view model收敛到typed terminal。

交互合同继续区分board-wide用户意图、per-family binding health与exact painted provenance readiness。Scan Workbench把A/C/Z/H、range candidate、Setting/Edit commit都绑定到当前`run_id + source ref + panel revision`；pending revision先disarm，replacement front paint后恢复。callback异常由`QtRasterBoard`本地detached并锁存，W3下一owner cycle强制取消checked Selector、关闭Setting popup、禁用两份editor并显示diagnostic/tooltip，异常不逃出Qt event loop；新Run/application replacement才建立新binding。完整application替换同时显式开启新的editor revision domain，故旧rN dirty draft、旧scan单位与runtime placeholder不能倒退冲突或串入新r0；普通同owner观察仍保持revision单调、dirty/stale/CAS。final/static/reconfigure/close都会清pending/range/popup/switch并unbind。小屏presentation不另造layout owner：scan两块board的正常最小值仍为`320×240`，只经既有Fluent scale在`800×600`降到`240×160`下限，使嵌入TaskConsole的完整控件保持屏内可达。

可选preview的预算不是一次局部cap。neutral scan owner先在编译期用完整static lineage、calibration retention与final transform公式证明science FINAL baseline；只有一个已经通过type/schema/snapshot-minimum验证的preview之**精确增量**触发`MemoryError`时，才terminalize该port并退回同一Run FINAL-only，malformed preview或science baseline不足仍hard fail。camera→processor的真实峰值只能在CaptureSession产生`CaptureProcessorInputBinding`并绑定`BoundOccupancyStreamProcessor`后得到，因此`_open_exact_occupancy`又在任何reservation、hardware prepare或FIRE之前复用同一个完整pipeline estimator比较`full`与`preview=None`：baseline超限拒绝science；仅preview超限则明确terminalize且不bind preview，Run继续。两阶段都没有广义异常retry、复制公式或host-stepped timing。preview预算另具名覆盖transform、evaluator、persistent Agg/artist、capacity-one candidate/front、Qt detach、display-revision overlap与一次pointer hold的`RGBA + exact evaluated arrays`；少一字节边界、present期间capacity-one与pipeline-only over-budget均有反例。

真实public virtual W3 oracle已经贯通camera→occupancy processor→exact preview slot→worker raster→Qt：在FINAL被post-safety barrier暂留期间真实发送hover/cross/area/zoom，证明r0→r1仍为同Run/同source、Setting/Edit同步、fault退役、FINAL swap、r1后application replacement回新owner r0；另有terminal same-source repaint、单worker starvation、static fallback、32/33 site边界、35-site显式site0、`(R,P,S,C)+ComponentValidity`、MHz与count单位、one-byte admission和pipeline preview-before-bind测试。实际profile在`P=1000,S=32`时exact freeze约`0.82 ms`、transform约`4.06 ms`、evaluate约`170 ms`、Agg约`129 ms`，约3 fps；hover在`32×9999`规模约`1.89 ms/call`，瓶颈是evaluator/Agg而非Hub/scan mapping，未据此预建LOD或增量evaluator。

验证门最终为活动manifest `92/92`文件、collect `1240`项；逐文件隔离进程实跑结果在本checkpoint提交前为`1237 passed + 3 expected skipped`、零失败/错误。focused的W3 application/controller、Qt curve、shared editor及全部M1/M2 camera消费者均通过；三路独立终审先后关闭general-executor饥饿、candidate过早释放、广义preview fallback、跨application revision倒退、callback fault未消费及pipeline峰值漏算，最终结论`P0=0/P1=0/GO`。nonmonotonic static fallback已有spec+真实renderer与共享Qt disable路径覆盖，未重复整套重型GUI Run，记为非阻塞P2测试深度而非产品降级。

rolling Monitor提交自身的历史Rule-6仍按parent `caf7f7e`的17个实际production path计数：`12279 -> 14884 physical`、`11310 -> 13752 nonblank`、`11223 -> 13665 token-NCLOC`，其独立算法、coherence与ROI不断流证据不重算。本次progressive提交按parent `2576114`的10个实际production文件（`_camera_monitor.py`、`_scan.py`、Qt shared helpers、scan application/controller/preview及occupancy pipeline）机械计数：`13838 -> 14847 physical`、`12719 -> 13677 nonblank`、`12358 -> 13294 token-NCLOC`，净`+1009/+958/+936`，比`1.0729x/1.0753x/1.0757x`；classes`69 -> 69`、dataclasses`18 -> 18`、enums`0 -> 0`、functions/methods`583 -> 611`。最大单文件`_scan.py`为`418 -> 887 physical = 2.122x`，零约3倍项；`_camera_monitor.py`因popup/editor同步DRY抽取反而`2864 -> 2759`。对已经冻结、同时覆盖Monitor+progressive且不能为两个提交重复计算的main严格包络`1532/1453/1165`，本提交净增为`0.6586x/0.6593x/0.8034x`；对公平用户面包络`4962/4647/3685`为`0.2033x/0.2062x/0.2540x`。没有新增class/dataclass/enum、manager、registry、renderer或framework；专用preview lane、两阶段capacity helper与owner-replacement lifecycle均由已复现死锁、容量或单位串线反例挣得。

共享架构裁决（后续提交仍遵守）：共享层只提升已经出现第二consumer的事实——`RelimMode + finite range/target/deadband`纯函数、通用exact `PanelInteractionOrigin`、popup placement/range placeholder/editor synchronization，以及一个直接参数化而非继承类族的revisioned Fluent form widget。Image和Curve各自保留强类型state、viewport与commit；`PanelFrame`只允许closed `ImagePanelPayload | CurvePanelPayload | None`，不能并列多个optional payload字段。`CurvePanelPayload`必须携scalar/evaluated `EvaluatedInput`（不是raw camera input）、draw后exact transform、series labels/value unit与同front immutable `EvaluatedCurve`引用；这些引用只为exact hover/cross/display，并且永不进入FitSpec、ScanOutputContract、CommittedTransform或artifact。Monitor与SCAN_POINT两项收口都没有修改TaskConsole authority、旧Hub/LogicNode或白名单外历史tests来迎合中间态。

##### U0.2d Sites 收口证据（`MATCHED_EXACT_VIEWER`；live paired port 尚未开始）

本项继续固定 `main@6c337d49c7086fa0ff21f879cd159bdf0e753f51`，证据来自 `frontend/live.py::LiveSiteMap`、`frontend/selectors.py` 与 `frontend/task_console.py::_sites_aux/_build_plot`。Sites 的物理定义是**同一 accepted provenance/cell 的 camera frame + pixel validity + calibration site centers + bool occupied + SITE component validity**；它不是单 DatasetSchema 的普通 projection，也不是把任意 per-site 浮点值按 `>=0.5` 阈值化。因此 baseline 不新增 `ViewIntent.SITES`，不伪造合并 Dataset，也不建立通用 composite/overlay/plugin 框架。当前唯一已经具备完整输入并对 source/calibration/revision/address 做 exact 交叉验证的公开产品 consumer 是 `Experiment.readout.occupancy_cell_gui(...)`；W3 progressive preview 只有 counts CURVE，不能用 latest frame 与 latest counts 拼成 Sites。

| salvage 字段 | `main` 真实行为、已识别旧缺陷与 U0.2d 收口判据 | 开工状态 |
|---|---|---|
| 数据、repeat 与同 shot | occupancy 为 `(R,1,SITE)`、centers 为 camera `(x,y)`、underlay 为 `(R,1,H,W)`；旧 panel 只接一个 occupancy source，并从同一 producer 取实际参与判断的 `frame_judged`。repeat 的 `average/add/replace`分别把 rings 与 frame按同一 accepted repeat 集派生。目标首个 exact-cell consumer只投影一个显式 `(R,P)` cell；未来 live repeat view也必须一次原子派生两边，绝不分别 latest-read | `MATCHED_EXACT_VIEWER`；live paired port `NOT_STARTED` |
| 轴与 validity | frame 的 `H×W` pixel validity 与 occupied 的 `SITE` validity是两张不同mask；invalid site必须保留第三态且 canonical `occupied=False`，不得冒充empty。完整源仍保持 `(R,P,*data_shape)`，不得flatten、按rank猜轴或合成一个粗粒度`cell_valid` | `MATCHED_EXACT_VIEWER` |
| 画面 | `origin=upper`，camera pixel extent `[-0.5,W-0.5,H-0.5,-0.5]`，按声明坐标步长保持physical equal aspect并向西锚定；不能拿QImage像素宽高冒充物理比例。frame underlay上画空心rings。occupied为`#D07850/0.95/0.9`，empty为`#FFFFFF/0.85/0.6`，invalid须以failure虚线显式显示。半径只允许复用current已修正的分块all-pairs nearest-neighbour纯函数，禁止搬旧的顺序依赖相邻点算法；颜色/alpha/linewidth属于headless `site_map` painter-neutral唯一owner，Qt与Agg只各自拥有paint API | `MATCHED_EXACT_VIEWER` |
| 无 background | 仍是完整二维spatial Sites view，extent从centers派生、colorbar/side band隐藏；不能用“是否存在image artist”把它降成1D | `DEFERRED_LIVE_RING_ONLY`；exact occupancy artifact总有same-shot frame |
| Area / Cross / hover | left drag二维矩形并显示handles/端点；right click锁定cross、right double-click/0.35s双击清除。旧cross只显示`(x,y)`，不伪造occupancy/frame z；目标新增hover若显示nearest site/occupied/invalid/pixel值，只能来自同一immutable payload。空间矩形通常对应非连续site集合；现有`Selection`不能表达它，首切只保存明确`DISPLAY_ONLY` candidate/highlight，严禁伪装为连续SITE range或权威Reduce；即使callback仍处于held front同步派发期，通用`selection_for_rectangle_gesture`也必须对SiteMap fail-closed | `MATCHED_DISPLAY_ONLY_EXACT`；authority binding留给真实TaskConsole Analysis consumer |
| Zoom / pan / home | wheel centered双轴zoom；middle drag按press pixel与冻结limits双轴pan；middle double-click有area时zoom-to-area，否则home。亚像素centers、ROI origin/binning、descending axis与coordinate frame必须由`ImageViewportTransform`唯一映射，Qt不得手写`x/W,y/H` | `MATCHED_EXACT_VIEWER` |
| hold / coherence | gesture只hold目标panel immutable front；source/完整board/siblings继续，release coalesce到latest。同一已提交selection下的数据revision更新可在pointer hold期间到达但held front保持不变；navigator提交新selection会提升selection revision并清除旧hold。任一background或occupancy的dataset/block/generation/schema结构、calibration/centers identity、frame shape、presentation/viewport改变都会使旧hold失效。`SignalHistoryGap -> latest`是已识别的旧静默错，目标必须gap-fatal/unavailable | `MATCHED_EXACT_VIEWER` |
| clim / display state | background默认gray，沿用六个closed cmap；`tight/normal/fixed`、hysteresis、两条clim handle与color limits全部复用IMAGE的唯一`ImageDisplayState`和indexed8量化。fixed允许有意clipping，不能因新数据越界自动重置。无frame时clim控件明确no-op/不可用 | `MATCHED_EXACT_VIEWER`；无frame仍属于live ring-only后续 |
| Setting / Edit | 两者消费同一`image_display_form_spec`、同一revision/CAS/placeholder owner；Sites不另造display state、parser或widget library。普通curve Fit不适用于Sites；未来Analysis ROI由TaskConsole composition映射 | `MATCHED_EXACT_VIEWER` |
| 更新与结构失效 | 同一候选内先冻结new frame再计算hist/clim，不能落后一帧；结构签名必须包含两input refs、source/schema/generation、calibration/centers identity、site axis/geometry、frame shape/dtype/validity、显式repeat/cell与display revision，不能只看occupancy shape | `MATCHED_EXACT_VIEWER` |

最小dependency-closed纵切先升级现有public `occupancy_cell_gui`，建立后续live Sites复用的唯一交互owner：`DisplayPayload` closed union增加且只增加`SiteMapPanelPayload(background: ImagePanelPayload, occupancy_input, site_axis, owned centers/occupied/site_validity)`；payload在worker构造期一次推导并冻结all-pairs radius、完整raster normalized centers与当前viewport ring span，Qt hot paint不逐site重扫坐标轴或重算半径。`PanelFrame.source_identity`指向occupied source，coherence stamp同时冻结frame与occupied exact refs并携calibration/geometry identity；Qt复用同一个`QtRasterBoard`、IMAGE A/C/Z/H、rail、hold、typed commits与Setting/Edit，只在paint阶段叠加三态rings。exact交互窗口直接拥有`QWidget + QtRasterBoard`，不得继承报告专用`FrozenRasterWindow`或继续把PNG当产品语义；它与真实冻结报告只共享语义中性的capacity-one raster worker executor、Qt wake/launcher生命周期，不共享PNG bundle。all-pairs radius、painter-neutral style与immutable site-state约束归headless frontend唯一owner；Qt只在已验证payload上派生三态paint masks，Agg与Qt共享事实但不共享painter。该exact FINAL consumer闭合后只能标`MATCHED_EXACT_VIEWER`，不能把U0.2 live Sites写成`MATCHED`；真正live完成仍要求current product port原子发布same-event frame+occupied+calibration identity，不能用counts preview或latest/latest替代。

Rule-6开工包络以main中Sites特有`LiveSiteMap + _sites_aux`约232 physical LOC为严格参考，约696行触发3倍压缩审查；目标新增production class最多两个（payload与只有真实第二owner才允许的窄controller），新增enum/protocol/registry为零。预算必须同时计两份exact input、centers/site masks及预计算normalized geometry、indexed8 evaluation/raster/hist/LUT、capacity-one candidate、当前front、Qt LUT detach、display-revision overlap与pointer hold旧front；`present_pending()`完成前candidate仍被占用，模糊总乘数不能替代具名项。独立oracle至少覆盖错位refs/calibration、交错latest、三态validity、`(R=2,P=3,H,W)+(R=2,P=3,SITE)`显式cell、亚像素/descending/ROI坐标、physical aspect/向西锚定、A/C/Z/H、clim、Setting/Edit、hold/stale/fault/close与`required-1/required`预算边界。

实际收口没有新增通用Sites intent、renderer、scheduler、registry或controller类族。`SiteMapPanelPayload`是closed `DisplayPayload` union中唯一新class；它只持owned immutable background、两条`EvaluatedInput`、SITE axis、centers、bool occupancy、component validity与calibration/cell identity。geometry/join digest在worker构造期一次缓存，但轴、ref与array canonicalization分别委托`zlc_data`公开codec/immutable owner，不在frontend复制公式。`OccupancyCellView.cell_selection`携带canonical全outer-axis exact `Selection`，cell worker在任何raster/stamp/present前用同一`OccupancyCellNavigation`重新解析并逐值比较，loader返回相邻cell不能再被请求wrapper静默改名。source retained bound同时满足`actual arrays <= declared bound <= admitted cell peak`，两侧任一漂移都在分配raster前fail closed。

Qt直接以INDEXED8 background加三态rings呈现，不再经过旧Matplotlib/Agg/PNG occupancy renderer；A/C/Z/H、physical equal aspect/west anchor、nearest-site hover、panel-local hold、六种cmap、clim和Setting/Edit全部复用IMAGE/QtRasterBoard现有owner。空间rectangle只保留normalized `DISPLAY_ONLY` candidate；board的通用authority转换对SiteMap显式抛错，不能伪造连续SITE Selection。exact窗口是直接`QWidget + QtRasterBoard`，不继承报告语义的`FrozenRasterWindow`。已有第二consumer的capacity-one executor、Qt launcher、error formatting、raster-bundle load与atomic display export统一归语义中性的`_window_runtime`，旧`_frozen_raster`私有跨模块helper名称全部删除，不保留alias。

预算公式逐阶段计入source load、owned payload、indexed8 candidate、当前/候选/held fronts、Qt三态mask与nearest-hover `40*S` workspace；`required-1`在repository大对象admit前拒绝，`required`通过。性能交叉验证在offscreen Qt下得到：100 sites payload约`0.824 ms`、paint约`0.864 ms`；1000 sites约`2.888/7.994 ms`；4096 sites约`21.614/33.332 ms`。all-pairs分块半径在100/1000/4096 sites的中位耗时约`0.036/1.519/18.986 ms`，真实约100-site使用面无需再造空间索引；4096-site压力样本仍线性有界且不改变权威语义。

Rule-6按parent `befb0ee`的12个Sites core生产文件机械计数：`11982 -> 13175 physical`、`10973 -> 12088 nonblank`，净`+1193/+1115`；classes`38 -> 39`、dataclasses`22 -> 23`、enums`2 -> 2`、functions/methods`455 -> 486`。相对232行严格旧Sites窄包络为`5.14x`，整个受影响closure为`1.0996x`，因此已经触发并完成压缩审查：删除旧static occupancy Agg/PNG路径、四个零consumer viewport wrapper、test-only payload/window accessors、死window state、重复codec/immutable实现和只因私有模块归属产生的runtime wrapper后，只剩一个新class/dataclass、零新enum/protocol/registry/controller/manager。倍率仍高的来源不是绘圆算法，而是旧232行完全没有的exact多轴/sparse selection、三repository同cell lineage、双ComponentValidity、总预算、latest-only/close生命周期、完整Qt交互与公开notebook seam；删掉任一组都会恢复已复现的错cell、静默bool coercion、超预算或UI线程问题，故不再以合并职责换低行数。

##### U0.2e Histogram H1 收口证据（`MATCHED_LIVE_ROI_SCALAR`；generic Distribution/H2 尚未开始）

本项继续固定 `main@6c337d49c7086fa0ff21f879cd159bdf0e753f51`。UX/数值 oracle 是 `frontend/live.py::HistogramFigure`、`frontend/selectors.py::{AreaSelector,CrossSelector,ZoomPan}` 与 `frontend/task_console.py` 的 histogram 分支；current 的依赖闭合产品 consumer 只有 camera monitor 中已经存在的 typed ROI scalar `MonitorDataset`。因此 H1 只关闭 `LIVE_ROI_SCALAR_HISTOGRAM_INTERACTION`：不会把 generic Distribution、Grid cell histogram、readout/calibration双峰拟合、threshold/fidelity、saved/reopened histogram或任意多维数据自动压成一个标量伪装成本项完成。

| salvage 字段 | `main` 行为、H1收口判据与正确性边界 | 当前状态 |
|---|---|---|
| 数据与轴 | repeat/index选择由Figure intent显式声明，所有声明为`SAMPLE`的轴坐标逐项保留；完整`(R,P,*data_shape)`、轴id/role/unit和`ComponentValidity`仍归DataBlock。Histogram只过滤显式invalid sample并报告`dropped_count`，不以NaN、rank、singleton、flatten、`[:,0]`或trailing mean猜语义 | `MATCHED_LIVE_ROI_SCALAR`；generic多series/multiaxis consumer待后续 |
| bins 与范围 | 默认60 bins、可编辑5–500；同一front的全部series必须共享edges。linear count轴：tight为`[0,1.1*peak]`、normal为`[0,1.2*peak]`；log为`[0.5,max(3*peak,1)]`，fixed必须为正。x limits可pin，auto/fixed范围保留deadband且不会随微小live变化抖动 | `MATCHED` |
| Area/Cross/Zoom/Pan/Hover | A为纯x display range；退化左点击明确clear，不能残留旧area。C为continuous cross，右双击clear；Z/H、wheel centered x zoom、middle x pan、middle double area/home均只改display state。hover读取当前immutable projection的真实bin；NumPy最后一个bin按`[left,right]`显示，其余为`[left,right)` | `MATCHED` |
| hold/coherence/revision | 手势只hold Histogram panel的immutable front；camera source与其它panel继续，release后coalesce到latest。intent冻结exact panel/layout/source/evaluated input/display revision；同revision允许data-derived auto范围随新front改变，却拒绝bin count、scale、relim mode、x pin/fixed count等authored事实冲突 | `MATCHED` |
| Setting/Edit 与 authority | 两入口消费同一个`HistogramDisplayState`表单和CAS owner；运行中只热替换presentation，不重建Run/source/ROI/scalar history。Area、cross、viewport与默认投影均为DISPLAY ONLY，绝不能复制进FitSpec、ScanOutputContract、CommittedTransform或Calibration authority | `MATCHED` |
| H2明确留白 | threshold line、Gaussian/双峰fit、classification/fidelity与analysis draft必须由显式物理consumer决定样本、模型、polarity和authority；H1不预建隐藏median、自动阈值、fit缓存或第二套selection owner | `DEFERRED_H2_WITH_EXPLICIT_SCOPE` |

唯一bin owner是headless `HistogramBinProjection`：构造器只接受exact immutable samples与authored bin count并一次生成counts/edges；projection保留每根原sample的对象identity，`HistogramPanelPayload`再次验证sample identity、全样本计数和bin count，Agg与Qt只读取同一个projection，不允许调用者注入“总数相同但分布错误”的伪counts，也不二次binning。`HistogramViewportTransform`同时冻结draw后坐标、`relim_mode/x_limits_are_auto/bin_count`和display revision，使同revision的自动派生变化与authored state漂移可区分。Curve与Histogram共用的只是已有`_NumericPanelBinding`交互生命周期；nullable typed range gesture同时修复两者的area clear，不新增Clear enum、wrapper或第二renderer。

offscreen profile交叉验证：300 samples、500 bins、2000次projection的median约`0.0411 ms`、p95约`0.0521 ms`；800×520 live Histogram Agg完整worker render的median约`41.396 ms`、p95约`50.201 ms`；同一四panel Qt board整板present的median约`1.555 ms`、p95约`2.009 ms`。瓶颈仍是既有worker-thread Agg compose/draw，不是binning或GUI paint；capacity-one/latest-only worker保持GUI响应，因此不为H1另造renderer、缓存层或executor。

Rule-6按12个实际production owner相对parent机械计数：`13956 -> 16079 physical`、`12975 -> 14960 nonblank`，净`+2123/+1985`；classes`84 -> 94`、dataclasses`61 -> 70`、enums`7 -> 8`、functions/methods`441 -> 499`，整个closure为`1.152x/1.153x`。main严格窄UX包络 `HistogramFigure + AreaSelector + CrossSelector + ZoomPan` 为`836 physical / 775 nonblank`，净增量为`2.54x/2.56x`，低于约3倍；该分母还未包含旧BaseLivePlot/TaskConsole/monitor glue，故是保守比较。最大既有owner `board.py` 为`3424 -> 3940 physical (1.151x)`，新`histogram_display.py`为636 physical；没有单文件约3倍、单成员enum、manager、plugin、registry、通用processor框架、兼容wrapper、第二stream/dataset/executor/renderer。提交前机械审查继续删除两个零consumer binning便捷wrapper及其package-root exports、零调用binding helper、test-only dropped-count convenience property和Curve/Histogram旧测试兼容seam；测试直接消费运行时shared numeric owner，生产只保留computed-only projection这一条binning入口。其余新增值对象分别承担closed count scale、唯一authored state、exact draw transform、不可伪造projection、跨worker/Qt payload和两种不同authority的typed intent，删除任一组都会恢复字符串分派、多owner、伪bin、stale intent或display→authority泄漏。

两路独立对抗审查分别覆盖data/authority与Qt/lifecycle，最初抓到伪counts可注入、同revision authored state可漂移、退化Area无法clear三项P1，整改及反例后均给出`P0=0/P1=0/GO`；最后一个“最终bin tooltip仍写右开”P2也已按NumPy边界规则纠正。活动manifest一次性收集100/100文件、1280项，随后按process-lifetime installation边界逐文件新进程实跑为`1277 passed + 3 expected skipped = 1280`、零失败/错误。该checkpoint的GO严格限于当前camera ROI scalar单series Histogram H1；finite capture后来由U0.2f闭合，但generic Distribution/Grid、H2 analysis、save/reopen、live Sites与完整TaskConsole仍不能从H1外推为完成。

##### U0.2f finite exact capture IMAGE 收口证据（`MATCHED_FINITE_IMAGE`）

本项仍以同一`main` 2D交互合同为UX oracle，但只关闭公开finite exact capture Workbench中已经存在的一张、唯一singleton `READOUT_EVENT` raw IMAGE。底层capture artifact继续完整保存`(R,P,*data_shape)`、具名`SPATIAL_Y/SPATIAL_X`、ComponentValidity、EventRefs与FINAL lineage；Workbench只在确有且仅有两根声明的空间data axis时建立IMAGE view，其它多维数据在Start preflight明确拒绝，绝不flatten、`[:,0]`、按rank猜轴、取第一项或隐式reduce。普通multi-event notebook capture完全不受这个viewer资格限制。

窗口不再使用只会显示一张图的`QtImageBoard`，而直接复用`QtRasterBoard + ImageViewportTransform + ImageDisplayState + LiveBoardController.reconfigure_image_display`这条已经由free-running IMAGE和exact Sites挣得的唯一交互链。Selectors默认OFF；启用后完整提供left-area、right cross/right-double clear、wheel centered zoom、middle pan、middle-double area/home、exact hover、clim rail、三态relim、六种cmap和x/y/color pins。rectangle只保存full-raster normalized `DISPLAY ONLY` candidate并调用通用image-family overlay入口，窗口从不调用`selection_for_rectangle_gesture()`，因此不能写入CaptureRequest、ROI、FitSpec、ScanOutputContract、CommittedTransform、artifact或FINAL。

Setting popup与Edit tab是既有`FluentRevisionedFormEditor`的两个实例，消费同一个`image_display_form_spec`、runtime placeholder与`sync_revisioned_form_editors`；没有新widget framework、parser、manager或通用controller。提交先验证painted `PanelInteractionOrigin`、当前display revision与editor base，再让live presentation接受，最后才发布窗口的authored state；同步拒绝不推进revision，异步raster失败只撤exact pending display intent并禁用交互，旧Capture FINAL和可重开artifact保持不变。重复Run只保留用户明确提交的`ImageDisplayState`，selector开关、area/cross/hover、pending intent、source identity、Run/slot/board全部重建；上一Run origin在新source上必定stale。

finite成功Run会保留其最后一张exact front供用户继续检查，但下一次Run不能与旧generation的显示工作重叠。Run的FINAL可能先于worker raster完成，所以Start按钮与`_start_capture()`入口都要求`owner_reaped && worker_idle`；owner cycle在`live.admit_pending()`之后再次计算按钮。阻塞raster反例已证明：旧worker未完成时第二次点击不换generation、不清FINAL；释放、present并drain completion后才恢复Start。Close先解绑selector并同步释放hold/front/pending capability，不等待worker；剩余worker完成后也不能late-publish，owner work清空后窗口才完成异步关闭。

内存准入使用metadata-only Figure estimator与公开INDEXED8 estimator，不在Workbench复制shape公式。候选`EvaluatedImage`与candidate raster scratch/result分别已由evaluation peak和raster estimator主体计入；`retained_fronts=1, retained_sample_fronts=1`只计candidate之外的旧对象。finite只有一个immutable source event，pointer hold的prepared/payload与当前front是同一对象别名；任何display commit又会同步把旧painted revision设为not-ready并释放hold，因此不会形成live monitor的“held old + current newer + candidate newest”三代拓扑。默认96×128 virtual profile的独立冻结见证为`1,359,561 bytes`：`required-1`在current `VirtualCamera.arm()`前拒绝且不产生FINAL，`required`恰好通过并只arm一次。

Rule-6以parent `4b52cff2800be1eb9f9b23b04fcd0c6c3012a039`的dependency-closed production cut（`_capture.py + qt_widgets/board.py + _occupancy.py`；后者只跟随generic rectangle API rename）机械计数：`5350 -> 5911 physical`、`4995 -> 5537 nonblank`，为`1.105x/1.109x`；最大单文件`_capture.py`为`685 -> 1241 physical = 1.812x`、`644 -> 1182 nonblank = 1.835x`，低于约3倍。classes/dataclasses/enums保持`12/7/0`不变，functions/methods为`222 -> 234 = 1.054x`；没有新增production class、executor、renderer、registry、plugin、compatibility alias或第二套display/selector owner。机械压缩已内联两个单consumer转发函数；`set_site_map_rectangle_candidate`则被无alias地原位改名为已有W1与Occupancy两个真实consumer的`set_image_rectangle_candidate`，只是把原有DISPLAY ONLY image-family语义从Sites私名提升为正确owner。

本项focused产品回归为W1 `21 passed`，并覆盖完整A/C/Z/H/hover、held-front alias与exact EventRefs不变、Setting/Edit CAS/stale、old-run origin、同步/异步故障、active hold和blocked rerender Close、blocked旧raster的repeat fence及`required-1/required`。活动manifest机械收集`100/100`文件、`1287`项，随后按process-lifetime installation边界逐文件新进程实跑为`1284 passed + 3 expected skipped = 1287`，零失败/错误。独立生命周期对抗最初复现“旧raster未完成即可开启第二generation”的P1，修复后定向复核为`P0=0/P1=0/P2=0/GO`；独立DRY/复用裁决确认纯转换、editor CAS、Qt hold/origin与live replacement均已有唯一owner，application shell不应被强抽成callback coordinator。该GO只关闭finite exact raw IMAGE；live paired Sites、generic Distribution/Grid、H2 analysis、通用interactive figure与完整TaskConsole仍继续`MUST_CLOSE`。

每一项开工前都先完成 §2.1 salvage gate，在 checkpoint 固化旧行为清单；收口逐项 PASS/FAIL，FAIL 只能进入 §2.2 账本。每个 commit 必须满足规则 8 的完整活动白名单 collect+run 全绿、独立 oracle、一次有界对抗审查、规则 6 与精确 staging。

纠正闭合后，剩余产品顺序固定为：

4. 完整 TaskConsole：全 plot-kind live/rolling grid/拖拽布局/Add Panel；catalog 驱动的 Logic 节点列表、Start/Stop、树形 signal picker 与自动 schema edit；共享 Setting/Edit；task/monitor/status；workspace save/load；
5. 通用 `figure_viewer` 广度与迁移收口：复用纠正3的同一interaction/Fit owner，把catalog/browse/open覆盖扩到任意已存figure/artifact，接入workspace/真实launcher，迁走旧`figure_viewer`最后consumer并dependency-closed删除旧路径；不建立第二套viewer、selector、fit或export实现；
6. device manager 与 task_console/pulse_gui launcher 切换到新 composition；
7. E01 的 temperature/MOT/fidelity operations 逐条迁移并按最后 consumer 删除旧路径；
8. 清活动白名单的全部 collect/validity 债务，错误清单只减不增；
9. Z0 零残余；`Zou_lab_control/workbench` 与 notebook 是新产品面，不作为旧树删除；
10. 全分支一次独立对抗终审：P0=0、P1=0、活动白名单完整实跑全绿，并交付 UX 偏离账本、迁移完成报告与真机 bring-up runbook。

`AUTONOMOUS_STREAMED` 的 9999 点级实验不是清单外便利能力。近期 resident 上限不足时，`AUTONOMOUS_REFILLED` 仍须满足 §15.4/H1 的冻结 bitstream 强证明；实现必须保留明确的 >4096/9999 host refill 启用路径。若只有真机资格化才能最终开放，软件先以 typed fail-closed 交付，runbook 必须逐条写明资格化实验、启用配置、命令、判据、回退与如何证明 host 只供应冻结 chunk、不参与硬件 edge 时序。

依赖关系是：

```text
E0a characterization ----+-> F0 safety/data spine -> S0.5 Workbench -> S1 Camera -> S2 Analysis -> S3 Readout
                         |                                              |
                         +-> H1 current-bitstream/pulse contract -------+-> Q0 release qualification
                                                                                |
                                          S4 implementation --------------------+-> Formal capability enablement

E0a/Q0 measured camera failure or proven RTL bug -> H2 evidence-driven hardware repair review (optional)

S1/S2/S3/S4 -> S5 remaining use cases -> Z0 zero-residual audit
```

本设计的开工判定分两层，不能把“架构可开工”与“真机 Formal Scan 已 ready”混成一个 GO：

- **GO NOW**：E0a中的代码/metadata只读探测、离线分析与profile、import-DAG ratchet、F0绿地data/storage/stream/catalog、S0.5 Workbench三surface与三个legacy containment bridge可立即开始。E0a主动真机触发实验不是“只读”，只能通过已批准的现有实验SOP与唯一设备owner执行，或等待F0最小safety spine后执行。
- **GO PER DEPENDENCY-CLOSED SLICE**：S1/S2/S3 只有在 LegacyRuntimeFence 生效、consumer matrix确认且每个切片不提前删除共享 producer时开始。
- **GO FOR S4 IMPLEMENTATION**：H1 current-bitstream合同、S1 exact acquisition与S3 processor/materializer接口稳定后可以实现S4；这不等于用户可启用Formal。
- **GO FOR FORMAL ENABLEMENT**：必须再有当前最终adapter/driver/buffer policy上的active Q0 qualification、active ProgrammedImageDeploymentRecordRef、compiled physical waveform/arm/edge/camera-tail margin、对应execution mode的现有bitstream terminal语义与稳定读规则、adapter-specific SafeStateContract、deployment-bound compiled/H1 physical-terminal recipe及所需`PostTerminalTailEvidence`、全部BoundSourceAssociationContracts、exact链与EndAttestation E2E。近期只评估恰好一个Q0-qualified qCMOS source，并先开放`AUTONOMOUS_STREAMED`方式族中的`AUTONOMOUS_RESIDENT`装载形态；refilled仅在§15.4条件能力发布后评估。API segmented按§14.7单独资格化“整run一次camera arm/aggregate terminal + R×P个STATIC_ONCE physical pulse terminal”的组合合同，而不是逐段camera qualification。contract kit、deployment record与qualification本身不要求新RTL，但若冻结硬件证据不能通过任一gate，Formal capability继续NO-GO，不能以软件state补证，也不因此自动授权重烧。
- **HARDWARE CHANGE NO-GO BY DEFAULT**：HardwareTriggerStamp、新ROM attestation、per-fire counter/PHYSICAL_DONE、trigger-return、watchdog或RTL CRC均不得由路线图自动启动。只有E0a/Q0在已批准工作余量、正确camera配置和充分软件reservation下仍测得真实丢帧/乱序，且已经证明无法通过camera设置、软件保留/排空策略、降低trigger rate或扩大时序margin修正；或现有RTL bug/设计偏离被证实，才进入H2评审。发现一次异常、qualification证据不足或架构希望获得更强证明，都不满足该条件。

### E0a：迁移前 characterization，不授予发布资格

E0a用于取得会改变架构选择、容量预算和真机工作点的探索性证据。只读探测与benchmark可以独立提交，但主动相机配置、外触发和长scan是硬件实验，不得被“GO NOW”误解为普通只读脚本授权：

1. 真qCMOS characterization对目标ROI/exposure/global-exposure/readout/trigger模式记录`nFrameCount/framestamp/camerastamp/timestamp`候选语义、位宽/signedness/modulus/reset/rollover、buffer行为、arm-ready/status ack、arm-to-first-edge、active/inactive pulse width、最小安全trigger间隔、last-edge-to-driver tail与terminal quiet-window；在已批准SOP与唯一owner下用多轮长scan估计“一触发一帧、按序、无漏”的工作区间和margin。H1建立deployment index之前，这条主动路径显式标记为`DIAGNOSTIC_CHARACTERIZATION`，继续强制现有fingerprint/geometry/ABI握手、批准SOP、唯一设备owner和诊断provenance；F0 safety spine尚未落地时使用既有批准SOP的等价hazard/safe记录，F0可用后立即改走其HAZARD_ACTIVE/safety disposition合同。它不要求尚不存在的active ProgrammedImageDeploymentRecordRef，不能进入`NEUTRAL_COMMON_FORMAL_PREPARE`，不能生成ScanArtifact、QualificationFireAuthorization、active Q0或任何Formal authority。若操作者能提供现有`.bit`/release信息，只作为待H1独立复核的诊断声明保存，不能提前冒充active deployment record。它可以收窄设计，却不能生成可供S4引用的active qualification。
   每次主动E0a必须保存当次observed live hardware identity或稳定endpoint、现有fingerprint readback、旧SOP的owner/arm/safe/abort evidence、操作者批准和完整原始诊断数据；这些证据只能说明“这次诊断按旧批准边界执行”，不得被转换、重命名或复用为AssetMap identity proof、SafeStateContract qualification、Q0、ProgrammedImageDeploymentRecord或Formal artifact。
2. 对现有 1D rolling、2D qCMOS live、gridplot/多 panel board 做 ingest-to-visible、GUI event latency、copy、compose 与 board coherence profile；据此确认 GUI_ARTIST、WORKER_RASTER_LIVE 的分界、front-buffer 预算和 S0.5 legacy bridge 的临时覆盖范围。
3. 对目标 RepositoryRoot（包括同步盘实际目录）执行 atomic replace/fsync/lock/crash probe；不满足合同就选择合格本地 root，而不是弱化 commit 语义。
4. 固定 camera queue、journal/materializer、fit batch、scan compile、artifact 和 UI benchmark matrix；保存基线 profile artifact。
5. 建 import-DAG ratchet，立即禁止新增 data -> 其它 bounded context、frontend -> neutral/pulse、pulse -> neutral/frontend/data 的反向边。

E0a报告是S1/H1设计、Q0测试矩阵、preflight margin候选与PerformanceBudget的输入。报告必须包含样本规模、持续时间、工作点、最坏间隔、观察到的loss/reorder率及其统计上界、设备/driver/旧adapter版本；当观察为零时也用样本量给出可解释的upper confidence bound。E0a证据在S1重写adapter或buffer/drain policy后不得直接授予Formal capability。

### F0：最小架构脊柱

只建立后续 S1 立即消费的正式能力：

1. safety spine：process-lifetime InstallationRuntime、RunController/RunHandle、单一ResourceArbiter、DeviceBroker/BoundDevice、immutable InstallationDeviceGraph、adapter owner、startup/shutdown lifecycle gate、真实termination acknowledgement与machine/device级单一SafetyJournal；先用阻塞fake/virtual camera证明HAZARD_ACTIVE durable前cancel不调用硬件、interrupt/join未完成不发布terminal或释放claim，并用每run唯一、幂等的SafetyDispositionBundle覆盖safe、mixed unsafe、restart、lost-ack retry及bundle/artifact/terminal相邻crash；真实bootstrap缺少persistent journal必须拒绝启动。
2. zlc_data：AxisId/AxisSpec、ValueSchema/DatasetSchema、Value/DataBlock、Validity、PointLayout、ValuePayloadContract 和 canonical codec；它是这些通用数据类型、数值 snapshot/byte-accounting 合同的唯一 owner。
3. zlc_storage：canonical primitive encoder/digest、BlobStore/ManifestCommitter/atomic probe；各 owner 保留 typed Repository/schema codec，并从第一天用 cross-package golden/property test 锁定 canonical bytes。
4. neutral stream：broker-minted generation、AcquisitionProducer/read-side stream、Payload/JoinKey contracts、opaque Delivery/EOS、single-formal reservation/ack、TraceBinding、BACKPRESSURE_CAPABLE/NON_BACKPRESSURE_CAPTURED、RetentionOverrun poison，以及 exact `DatasetBuilder/DatasetProgress/DatasetPreviewSnapshot/SealedDatasetArtifact` 与 bounded `MonitorTap/MonitorDataset/MonitorDatasetSnapshot`；不发布累计 DataBlock，live owner 在 S5 aggregate admission 完成前不接真实 Workbench。
5. explicit DefinitionCatalog、PipelineSpec -> flat RunPlan compiler；此时只支持 S1 所需的 Measurement、DatasetBuilder 与 sink，不预建递归 plan 或通用 workflow DSL。
6. camera exact queue 改为 O(1)，明确 driver buffer ownership。

F0 只有 contract/unit tests，不作为长期“基础设施里程碑”单独宣称产品完成；完成标准是立刻进入 S0.5/S1。

### S0.5：先建立可承载纵向替换的 Workbench 壳

当前 `task_console.py/live.py/pulse_gui.py` 是共享一个 console-wide RenderLoop 的巨壳；如果 WorkspaceModel/PanelHost/render surface 全拖到 S5，S1 无法替换 camera panel 后删除旧路径。S0.5 只建立迁移宿主，不迁全部领域逻辑：

1. 建立最小 Workbench composition、WorkspaceModel、BoardController、PanelHost 与 RunHandle/status binding；不复制 TaskConsole 业务规则。
2. 交付 GUI_ARTIST、WORKER_RASTER_LIVE/BoardFrame 和 headless export surface 的接口与真实性能测试。
3. 建立 `LegacyPanelHost/CatalogRouter`、`LegacyRuntimeFence` 与 `SerializedLegacyAggBridge` 三个窄桥；旧 panel 可逐项隐藏/替换，旧 LogicNode 的所有 start/stop 先登记`LegacyRunFootprint(claims, reference_keys)`，Figure handoff timeout fail-closed。`claims`只描述本run真实host-side控制/读取语义并交给ResourceArbiter；`reference_keys`描述raw connection、接线或binding/lifecycle依赖，只供InstallationRuntime shutdown查找并等待相关handle terminal，绝不自动升级成OBSERVE/EXCLUSIVE claim。VirtualCamera读取其虚拟trigger wire是adapter内部接线事实，不等于CameraMeasurement对sequencer申请OBSERVE。全部referenced devices都必须出现在reference_keys，任何真实读写仍必须出现在claims；缺任一集合或无法证明时，同一ResourceKey的legacy/new mode保守互斥。这样“只引用”不会跨runtime悬挂raw adapter，也不会因虚拟接线错误阻塞另一个合法EXCLUSIVE run。迁移期 PulseGUI 的prepare/fire/abort/safe、notebook/session的camera/sequencer drive verb也必须经同一个LegacyRuntimeFence/installation authority，不能继续持有raw device旁路；无法机械约束的真实入口在迁走前禁用。
   config/device/virtual-real改变只由LegacyRuntimeFence/InstallationRuntime拒绝新admission、等待这些handle真实terminal并完成safety/journal，再发immutable `RestartRequired`/shutdown状态供UI显示；TaskConsole/PulseGUI只在Qt owner thread queued reconcile界面，QWidget hook、panel registry和GUI teardown既不执行硬件stop/close，也不能确认或veto shutdown。GUI未启动、已销毁或事件循环卡住不得改变硬件安全结果。旧runtime关闭失败时继续强持有同一个raw graph、authority refs、journal lock与lanes供重试；不得在原进程创建replacement local authority。

   当前若存在`Zou_lab_control.neutral_atom._gui -> zlc_workbench` launcher反向import，只允许作为import-ratchet中的这一条S0.5临时shim；notebook facade 当前直接使用的 `_triggered_camera/LegacyNeutralAtomRuntime` 等硬件 composition submodule 也必须迁到两种应用面共同依赖的非 GUI composition owner，不能把 `notebook -> zlc_workbench` 变成永久边。迁移完成前 `zlc_workbench.__init__` 必须保持 lazy，确保 headless notebook import 不连带加载 `zlc_frontend`/renderer；这只是隔离，不是最终所有权。notebook/workbench composition入口接管launcher后，必须在**S0.5完成前**删除上述反向/错误方向边和allowlist。`LegacyRuntimeFence`本体可按最后legacy consumer保留到S5/Z0，但 neutral/notebook 到 workbench 的 import 不能随它存活。
4. 后续每个 dependency-closed 纵向切片以新 panel/controller/runtime 替换对应旧岛；已迁 use case 立即删除自己的旧路径，但共享 producer/algorithm 只在最后一个旧 consumer 迁走时删除。三个 bridge 都有删除期限，不是 public API，Z0 必须为 0。

S0.5 解决的是“新切片住在哪里”，不是预先重写 9000 行 UI；Setting/Edit/catalog 的完整迁移仍随实际 panel/use case 发生。

当前W1以最窄纵向产品面兑现了这个宿主原则，而没有提前造通用workflow：public lazy `Zou_lab_control.workbench.open_capture_workbench(experiment, request)`只在调用时加载Qt；`PreparedFiniteCapture`是notebook与Workbench共用的one-shot应用边界，公开面只有descriptor、capacity-one preview schema与start，不暴露`MinimalPipelineSpec/RunPlan/Port/runtime/authority token`。每次Run使用重新prepare的command并获得新的slot、board、generation和Run；上一Run必须terminal且owner thread已reap后Start才重新开放。这个window不拥有Experiment，关闭时只撤销/清理自己的preview、Run和worker。

当前W2a+W2b已完成 current PulseGUI 产品纵切：current-only authoring helper不暴露raw lane，并把digital/DAC visible-port约束收敛在`PulseDocument`唯一owner；descriptor-free `PulseEditorSession`拥有immutable document、revision、真实disk baseline/path与save-conflict语义；preview从带source-document provenance的`CompiledPulseArtifact`产生有界、精确的digital delay/repeat/DAC edge-ramp timeline；API nominal literal只服务预览，hardware Run对任何未显式解析的API parameter fail closed；pulse-only `RunPlan`复用现有RunController，finite有真实terminal result，continuous HOLD无轮询等待cancel并复用同一interrupt/SAFE清理。W2b Qt 窗口已组合 Edit/Preview/Scan、New/Open/Save、static/scan/HOLD/API 与 Stop，用有界 worker + Qt-owner generation/revision reconciliation 保持 GUI 可操作；offline 只 author/preview，virtual/online 只经已有 `PulseFacade` 执行，窗口不拥有 Experiment。但旧嵌入式 PulseGUI/TaskConsole 共享路径的最后consumer还未迁走，因此不能删除该 legacy island，也不能把整个S0.5/H1标COMPLETE。

规则6海拔核对按“非空、非纯注释 source line”统计：main 旧 `frontend/pulse_gui.py` 为3725行，W2b Qt window+启动器为1833行（0.49x），W2a+W2b完整产品核心（authoring、timeline、application execution、editor session、Qt window、launcher）为3595行（0.97x）。新纵切不但覆盖旧editor界面，还包含可独立验证的compile/timeline、typed Run/SAFE和offline composition；总量仍未超过旧单文件，因此没有为压行数而合并这些已有不同consumer/线程边界的owner。

### S0.6：封闭 public raw hardware capability

这一步在继续迁正式采集链前完成对象图收口，但不要求先重写全部旧领域算法：

1. 建立 process-lifetime `InstallationRuntime`、private immutable `InstallationDeviceGraph`、`DeviceRef/DeviceInfo/DeviceCatalogView`、typed timing/readout/trap descriptors与窄command/admin ports；raw adapter graph转为composition/runtime私有实现，ResourceArbiter/DeviceBroker/PersistentSafetyJournal/typed resolver/catalog由同一个runtime一次拥有。
2. 建立唯一startup：journal lock/replay -> physical-owner proof -> AssetMap -> owner lanes/adapters -> open/identity/bind/probe -> graph/catalog freeze -> Run admission。删除connection-establishment lease；durable blocker只允许claim-first RecoveryAttempt。任一点失败都关闭exact owned subset且不发布partial facade。
3. 建立唯一config边界：device/config/virtual-real改变只返回restart-required并进入§12.7；原进程不创建replacement graph。runtime_instance_id与connection generations在新进程重建，stale DeviceRef/command在触碰adapter前失败。
4. 先迁 production 内部 consumer：只有InstallationRuntime及其明确的startup/shutdown/recovery implementation可持有InstallationDeviceGraph。Experiment facade每次操作只在临界区snapshot一次并立即构造runtime/binding-pinned request；Definition bind、measurement/task、provenance和runtime helper只能接收领域声明的immutable typed bindings或DTO，resolver只存在于bind调用栈，任何consumer都不能保存整个private composition state。仍未迁完的legacy node只可在LegacyRuntimeFence岛内通过私有binding运行。
5. 迁workbench与frontend边界：TaskConsole controller去掉Session/fence并只接RunCommandPort+DTO；PulseGUI controller改为PulseTargetDescriptor+PulseEditorSession+pure projector+既有pulse application facade，不再增加第二层PulseCommandPort；DeviceViewer/Manager controller改为catalog reader+窄control/admin port；zlc_frontend只接workbench ViewModel/纯widget props，不接neutral/pulse/runtime类型；standalone real launcher若未加入同一authority立即拒绝。
6. 将adapter作者文档与测试移到adapter_sdk/testing/simulation namespace；普通notebook教程全部改为`connect -> Experiment facade`与`device_catalog`/typed descriptors。测试需要raw spy时在composition前保留，不从Experiment取回。
7. 一次删除public raw aliases、fallback与umbrella exports，不提供deprecation `__getattr__`或compatibility proxy。机械object-graph/import/signature/AST/docs gates变为required。

S0.6完成时目标composition已经只使用InstallationDeviceGraph。旧`DeviceSet`符号若仍被未迁legacy island消费，只能登记为待最后consumer迁走即物理删除的旧实现，不能被目标runtime包装、适配或继续承担composition职责；它从普通Experiment、domain object与frontend根对象均不可达。public capability边界未通过时，后续真实设备新功能为NO-GO，因为它们会扩大尚未封闭的旁路。

### S1：Camera -> Value event -> Dataset -> live/save/notebook

1. 迁 CameraPort、BoundMeasurement、CaptureSpec/CaptureSession 与 owner I/O lane。DeviceBroker对当前binding/generation的真实readback mint capability attestation；CaptureSession冻结CaptureSpec owner digest、创建唯一exact reservation并在DatasetBuilder claim后才能start，ordinal到cell的映射只来自冻结schedule。`AcquisitionProducer` 只能封装在 CaptureSession owner 内，普通 Measurement/processor/UI 不可见；CaptureSession 对 qCMOS 固定使用 `NON_BACKPRESSURE_CAPTURED`。
2. qCMOS/DCAM 边界先产生§7.3定义的单一 immutable `CameraFrameRecord`，保留 `source_ordinal/produced_count/framestamp/camerastamp/timestamp/driver_buffer_index`；CaptureSession 在 owner lane 把每条 record 一次转为 `CameraSample(image: Value, metadata)`。payload contract 必须把 driver ndarray 复制/转移为 owned immutable Value，把所有 metadata 冻结并精确计入 retained bytes；旧 ndarray reader只是同一 record queue的迁移期解包视图，不得形成平行缓冲真相源。DatasetBuilder 根据冻结计划 key 写私有 current storage，只发轻量 DatasetProgress；UI 按 refresh budget 请求 SliceSpec/current-frame 或节流的 DatasetPreviewSnapshot，禁止每帧 full DataBlock fan-out。
3. 交付 IMAGE ViewContract/ViewSpec/FigureEvaluator、2D live raster+Qt overlay、Workbench LiveDatasetBinding；验证 GUI/worker owner 和 driver buffer reuse。
4. 交付 CaptureArtifact Repository 和 crash-safe commit；live/save 冻结用户所见 revision。qCMOS EOS 的唯一合法顺序是：唯一I/O owner先读取execution-mode-specific raw terminal evidence或abort/safe ack，只确认对应logical table/segment terminal -> **camera保持capturing、dedicated drain继续运行**，从该观察点完整等待deployment-bound CompiledPulseArtifact/H1 physical output-tail bound并生成`PostTerminalTailEvidence` -> 再在Q0-qualified quiet-window/保守deadline内排空 driver residual -> 读取并冻结最终 counter/stamp -> camera `cap_stop` -> capture/transfer状态稳定复核 -> buffer release -> capture thread/session 真实termination/join ack -> 才调用 producer.finish。任何raw terminal evidence都不证明delay tail idle；固定tail/drain deadline只在H1/Q0合同内构成有限运行保证，不声称逐沿数学证明，也不参与edge调度。正常complete与取消/异常cleanup共享同一个session termination语义；取消先走thread-safe interrupt解除阻塞，再由cleanup-capable、session-specific close command完成tail wait/drain/join，而不是调用已撤销的普通execute command。任何 extra/late/count mismatch、wrong-session/join unknown或物理 capture 后的 decode/schema/key/publish 异常先 `producer.fail`，因此不能生成 SealedDatasetArtifact；仅“已经收到 N 帧”绝不是 EOS 证明。
5. 同时交付薄 Experiment：`connect -> capture -> inspect/figure -> save` 保持少量语句。
6. E2E 后只删除**已经迁入新 CameraPort/DatasetBuilder 的 standalone camera use case**对应的旧累计 buffer/latest polling/render 旁路。旧 `OccupancyProcessor` 已删除，但 generic camera producer/LogicNode 仍服务 ROI、monitor、temperature 及剩余 legacy measurement/UI；它与这些消费者必须作为 dependency-closed island 保留到实际最后消费者迁走（当前预计 S5），不能把删除时点写死为 S3，也不能建立把旧 Hub 翻译成新 Dataset 的临时 bridge。任何时刻同一真实 camera 仍只有 legacy 或 new 一个 owner。当前具体发布 `TaskOutput` 的旧 Task 只剩 `OptimizeMotFieldTask`，在该最后消费者迁走的 S5 删除。

当前完成三块仍有限定的S1子集：S1#1的adapter-neutral record/working-point software seam已经把endpoint对`VirtualCamera`的硬编码移除，非Virtual contract fake可贯通exact command boundary，且virtual capability/artifact identity保持不变；但camera owner I/O lane、real AssetMap/Q0/tail evidence与最终DCAM迁移均未交付，所以real qCMOS仍NO-GO。S1#3的finite exact preview foundation保留完整cell data axes并接通FigureEvaluator/GRAY8、单panel BoardController与Qt presenter；W1进一步交付真实public app composition、Start/Stop、重复Run、FINAL/NOT FINAL/preview故障分离和异步window teardown，U0.2f又补齐其raw IMAGE的A/C/Z/H/hover、clim/cmap、同源Setting/Edit、display-only rectangle与旧raster完成前禁止repeat。W1只接受唯一singleton READOUT_EVENT且cell data axes恰为声明的SPATIAL_Y/SPATIAL_X；这项资格校验只在请求preview时惰性执行，不能反向限制普通notebook multi-event finite capture。其它多维数据仍完整写入exact artifact，但当前image panel在Start前明确拒绝而不flatten/select-first/reduce。M1已交付virtual free-running raw IMAGE monitor；M2b/M2c/M2d/M2e在当时交付typed ROI scalar、独立history、IMAGE+CURVE+HISTOGRAM+METER coherent board与front-bound rectangle draft/overlay，但其“Apply时替换整个source Run/首ROI先停旧generation”的结论已由规则9与`UX-001`否决，只保留为历史实现事实。U0纠正1必须改为已有ROI走`ControlTopic/APPLIED`、首个ROI创建/删除只迁移downstream processor generation，raw source/Run/front/tap topology始终连续。save current-view、通用grid/动态layout、旧camera panel替换和真实qCMOS仍未交付，所以S0.5与S1均不得标COMPLETE，也不得据此删除任何旧GUI producer/consumer。

### H1：Pulse bounded-context 与冻结 bitstream bring-up（与 S0.5-S3 并行）

先建立PulseDocument/TargetIR/CompiledPulseArtifact canonical seam，并以当前已部署bitstream对应的host/model/wire golden bytes、现有xsim/真机回读保护语义。按consumer纵向切换：compiler/server -> neutral Sequencer adapter -> workbench PulseTimeline consumer；每切一个consumer删除其旧timing/compiler/reader，不维持自动fallback。整个H1默认不修改RTL、不生成新bitstream。

2026-07 追溯审查确认 tracked `pulses/*.json` 已全部是 current `zlc_pulse.PulseDocument`；仍调用 `PulseTableState.load()` 的 Camera/Task/PulseScan/GUI 路径属于未闭合的 consumer cut，不是需要兼容的双格式需求。开发机 ignored 的 `T.json/pulse_test.json` 曾使测试输入集合随工作区变化并掩盖断口，因此禁止恢复旧 tracked 资产、在 `PulseTableState` 中加入 `PulseDocument` 猜测/转换器，或让测试依赖 ignored 文件。H1/S3/S5 必须把每个仍需保留的 camera、PulseScan 与 GUI consumer 直接迁到 current document/compiler/endpoint owner，并在同一个 dependency-closed 切片删除对应 legacy reader；完成前这些具体 legacy 用户路径明确 NO-GO，不能靠只测新栈宣称系统全绿。

本轮 dependency-closed clean cut 已物理删除 legacy `CalibrateReadoutTask`、`OccupancyProcessor`、对应 wrapper/export、旧 calibration report 生成与 frontend renderer、`calibrate_all_methods_from_images`、legacy `default_imaging_template()` Python factory 及其 mirror/negative tests；也删除了只服务旧 form 的 `ParamDecl(kind="pulse_param")`、widget special path 与 `enumerate_pulse_params`。这些对象不再是等待 H1/S3 迁移的消费者，也不得以 compatibility 名义恢复。语义明确的 `pulse_slots` 保留。

原`pulses/imaging_template.json`已按领域所有权移动为packaged `zlc_neutral_atom/assets/imaging_template.json`：它仍是current `zlc_pulse.PulseDocument`，但语义是neutral-atom readout acquisition recipe，不再作为仓库根部无owner资产残留，也不是legacy Python factory。current PulseWorkbench 已直接消费 `PulseDocument/compiler/endpoint`；Camera/PulseScan 与旧嵌入式 PulseGUI/TaskConsole 中尚存的其它消费者必须在各自dependency-closed切片直接迁到current owner，不建立 `PulseDocument ↔ PulseTableState` converter，也不因新窗口可用就提前删除共享producer。W3e 的 TaskConsole camera→occupancy-counts SCAN_SLOT入口已经current；calibration创建/编辑workflow、独立occupancy panel及其它legacy consumer仍保持NO-GO，直到各自controller纵切完成。旧 graph 使用过的通用 GUI/runtime 契约——coherent shot、derived provenance/flow、site data axis 与 rerender current snapshot——改由中性 test double 继续验证，不因fixture owner被删除而丢掉能力覆盖。

H1完成现有`image.build_fingerprint`/几何/ABI握手、PreparedProgramRef软件guard、repeat轴展开的finite autonomous table与camera-trigger schedule digest、当前UART/AXI/JTAG容量/错误行为，以及raw STATUS/CURSOR的组合读序、logical终态值和双读稳定规则的contract kit。H1同时根据当前RTL delay scheduler语义与CompiledPulseArtifact的冻结channel delay/最后edge推导`max_physical_output_tail_after_logical_done`，用golden/xsim/真机观测验证正常与safe/abort变体并给出保守margin；raw DONE/CURSOR本身不算tail-idle证据。高层`scan_progress()`镜像只供UI，不进入Formal proof。Formal compiler明确强制`repeat_forever=False, scan_repeats=0`并拒绝host wrap-stop；`AUTONOMOUS_RESIDENT`形成近期装载方式基线。超过resident window默认明确拒绝；只有单一I/O owner、保守refill硬上界以及覆盖每个潜在seam的硬件时间观测/完整schedule residual均由contract kit证明，才发布`AUTONOMOUS_REFILLED`条件execution capability。只测试现有RTL实际提供的能力，不增加ProgramToken/CellFireToken、ROM attestation、CRC verifier、PHYSICAL_DONE或telemetry。preview 使用 pulse-owned immutable `PulseTimelineDocument`，不制造 frontend -> pulse 反向边，也不让 generic Figure codec 承担 pulse authoring replay。

H1同时建立当前endpoint的installation-owned `ProgrammedImageDeploymentRecordRef`，把冻结`.bit` content digest、release/timing records、现有fingerprint和owner批准对应起来；这一步只登记并复核现有部署，不调用Vivado、不program硬件。autonomous table与API STATIC_ONCE pulse session分别发布各自H1 physical-terminal recipe；API底层若只有PreparedProgramRef+compiled segment schedule+stable raw DONE/STATUS，则recipe必须把必要的output-tail处理纳入唯一PulseTerminalAck，CURSOR=N/A。它只产生per-cell pulse evidence；camera仍由整run唯一CameraRunEvidence terminalize。

H1与S1的最终adapter contract kit共同发布`SafeStateContract`矩阵，不用generic `getattr`猜测。InstallationDeviceGraph中的每个adapter owner必须在composition阶段被exact adapter table显式分类为`MANAGED_HAZARDOUS`、`MANAGED_NONHAZARDOUS`或具名`PASSIVE_OUT_OF_SCOPE`；未知类型、未知subclass或默认`continue`一律拒绝，避免新Laser/RF/Camera只因尚未被某LogicNode引用就绕过authority。

qCMOS要求同identity/generation的capture terminal、DCAM status、buffer/session termination与join组合。Pylon要求同identity/generation的SDK grabbing/status、session termination与**live connection**组合：`IsGrabbing()==False`和缓存的`GetDeviceInfo()`都不足以证明SAFE或同一设备；recipe至少要求open、`IsCameraDeviceRemoved()==False`，并执行由真机contract kit资格化的transport/node-map live readback。因为SDK的removed状态可能要到首次真实访问失败后才更新，单独检查`IsOpen()`或一个removed布尔也不够；任一live readback失败、removed/disconnect或identity变化立即使旧binding/generation失效并返回UNSAFE/quarantine，重开必须由authority签发新generation，禁止transparent reconnect。Remote FPGA只接受当前冻结硬件实际存在且经真机解释的raw terminal/status/safe/readback与tail evidence组合，不接受server本地`state="safe"`；Manual backend默认是无危险控制能力的人工边界，若声明hazardous capability则只能进入显式人工recovery。

每个cleanup recipe将所有声明的止险动作分为`MUST_SUCCEED`与`BEST_EFFORT_THEN_VERIFY`，按声明顺序尽量全部执行并聚合错误；前一步抛错不得无条件跳过后续仍可能有效的stop/disarm。只有全部MUST_SUCCEED ack和最终肯定readback都成立才mint SAFE，部分命令成功仍是UNSAFE。每个recipe列出肯定、否定、readback失败和disconnect/generation-change结果；任何缺失肯定readback的真实adapter均不能mint SAFE，Formal capability保持NO-GO。该矩阵用于如实评估现有能力，不要求为了通过测试增加寄存器或重烧。

### Q0：最终版本的 qCMOS release qualification

Q0只能在F0 safety spine、S1最终CameraPort/CaptureSession/driver-buffer ownership与drain policy、H1 compiled trigger schedule语义稳定后执行。它复用E0a选出的工作点与预算，但必须用将要发布的真实adapter、SDK/driver和buffer policy重新跑qualification；E0a报告不能被重命名或复制成Q0 artifact。

1. 对每个发布工作点生成immutable `CameraExternalTriggerQualification`，保存设备/firmware/SDK/driver/adapter identity及其evidence kind/receipt digest/AssetMap revision、ProgrammedImageDeploymentRecordRef revision、camera readback、buffer/drain policy、arm-ready/first-edge、active/inactive width、trigger interval/margin、last-edge-to-driver tail/quiet-window，以及nFrameCount累计快照与per-frame stamp/timestamp各自的width/signedness/modulus/reset/rollover/first-frame语义、样本规模/持续时间、loss/reorder统计上界和PI批准。
2. `CameraQualificationIndex`必须加入camera所在的同一installation authority，并与FIRE共享一个跨进程可验证的physical-owner/linearization gate后才能原子activate revision；不得在ResourceArbiter旁建立平行控制authority。旧revision保留但不再active。version/identity/设置集合改变、合理疑似camera违例或已归因的`CAMERA_ENVELOPE_VIOLATION`分别追加suspension/revocation，重启后仍不可用。
3. contract tests覆盖activation、version mismatch、setting越界、explicit suspension/revocation/exoneration、journal lost-ack/failure、crash/restart replay、两个preflight pin旧/新revision、revocation-vs-FIRE gate竞态、历史effective scope，以及processor/EOS失败不会误撤销camera qualification。
4. Q0通过只说明相机在批准envelope内具备关联前提；它不单独授予Formal ScanArtifact。S4仍须逐run满足execution mode、exact pipeline、association proof、EndAttestation和authority commit。

### H2：证据驱动的可选硬件修复门

H2默认不排期。只有以下任一证据成立才可创建提案：

1. E0a或Q0在批准工作余量、正确camera配置和足够软件reservation下仍实测到丢帧/乱序，且无法通过扩大margin/降低trigger rate解决；
2. golden/xsim/真机交叉验证发现现有RTL真实bug或与已经批准的硬件设计不符；包括某安全行为只有在证明根因位于RTL且偏离既定设计时才满足本项，普通软件/transport问题不能借此要求重烧。

提案必须选择修复被证实问题的最小改动，列出不改RTL替代方案、回归范围和重烧风险，经PI/硬件owner批准后才实施。HardwareTriggerStamp、trigger-return、watchdog、ROM attestation、RTL CRC等只是候选，不是默认答案；任何合法重建都重新执行timing/CDC/golden/真机验收。

### S2：Data analysis + Frontend Figure 的完整纵向切片

1. 在 zlc_data 完成 Selection、DataTransform/Reduction/CommittedTransform、FitSpec/BoundFit/FitResultBatch；在 frontend 完成 ViewContract/ViewSpec/ViewSuggestion、FigureDocument/codec、selector controller、DataFigure/render。
2. 以“冻结 Capture/DataBlock -> curve/gridplot -> fit -> overlay -> FigureArtifact”为真实路径，同时验证 scalar、site grid、多 batch axes、component validity 和 per-cell failure。
3. 当前interactive/offline/artifact路径都由zlc_data `bind_fit`产生同一个BoundFit并对冻结snapshot调用`BoundFit.run()`；TaskConsole只桥接FINAL ScanArtifact。未来确有自动formal consumer时优先编译“FINAL artifact -> flat analysis Run -> one FinalCommit”，不在S2预建DatasetInputSlot、generic AnalysisStep或composite scan commit。
4. 验证 display ViewSpec 无 authority 字段，Selection candidate 只有在 FitSpec/CommittedTransform 中重建后才进入结果 lineage。
5. 新路径不再增加旧 `core.selection/fitting/facet/raster` consumer；这些模块、scalar fit signals 和 neutral Fit-named implementation 只在其最后一个旧 frontend/ROI/Analysis consumer 迁走的切片物理删除，通常为 S3/S5/Z0，不能在 S2 提前断开 opaque legacy island。
6. W3a先交付Capture/identity-Fit notebook surface，W3b由scan owner接通`ScanArtifactRef -> canonical output OwnedSnapshot -> DataFigure`。output BlockId由logical document、exact source DatasetRevisionRef与ScanOutputContract共同派生，generation/revision继承exact source dataset，schema fingerprint来自output DatasetSchema；含values/provenance/safety bundle的ScanArtifact manifest content-addressed。zlc_data用同一峰值函数做pre-FIRE schema估算与实际CommittedTransform，并一次冻结最终DataBlock；scan/notebook不重做reshape、validity映射或lineage。U0.3d已让`fit(scan_ref)`与Capture走同一个BoundFit、FitResultRepository和Figure host，source union仍是closed两类；它没有建立ScanFit框架、第二solver或generic Analysis registry。

### S3：StreamProcessor、Calibration 与 Occupancy/readout

1. 在已工作的 camera event 上加入最小 `StreamProcessorWorker`、typed record、join/cardinality/budget 和 exact propagation；不让 StreamProcessor 读取累计 DataBlock。
2. 完成 CaptureArtifact -> CalibrationAnalysis -> CalibrationArtifact 的 live/offline 同路算法，以及 FrameContract/SiteMap/ReadoutModel。
3. 迁 `OccupancyStreamProcessor`，输出单个 `OccupancySample(occupied, counts, metadata)` typed record，并显式绑定 CalibrationArtifactRef/model。
4. DatasetBuilder 把 occupancy events 物化为 dataset；frontend Figure 与 zlc_data Fit 直接消费该冻结 dataset，证明四平面边界贯通。
5. integration 通过后删除已经被 current Calibration/Occupancy slice 覆盖的旧 readout-specific fallback、filesystem search、拆散 scalar signals 与会碰硬件的旧 Processor。generic camera producer/LogicNode、`read_frames()/acquire()` array-only reader及其测试只有在 ROI、monitor、temperature、remaining legacy measurement/UI 的 consumer matrix 清零时才能随最后消费者所在切片一起删除；不得为了让 S3 “看起来完成”而提前切断 S5 能力。保留的只有 notebook/workbench composition 在 request 构造时按 ReadoutBindingKey 冻结显式 CalibrationArtifactRef/model 的可见 convenience pointer。

### S4：近期 Formal PulseScan（AUTONOMOUS_STREAMED）

S4代码实现可在H1冻结bitstream合同、S1 exact acquisition与S3 StreamProcessor/DatasetBuilder接口通过后开始；任何真实用户Formal capability必须等当前版本Q0 qualification active后才能enable：

1. bind declared ExactSourcePipeline，fire 前建立全链 reservation/cursor/budget/ack；不得借用 monitor worker。
2. repeat轴展开进`repeat_forever=False, scan_repeats=0`的finite logical table；preflight冻结camera readback与compiled physical trigger waveform/tail bound，验证Q0 arm/edge/pulse-width/interval/tail envelope、host total frame/byte retention与camera max-inflight ring。近期默认只启用`AUTONOMOUS_RESIDENT`；大表仅在§15.4的单I/O owner、refill硬上界和每seam硬件时间观测/完整schedule residual capability全部发布后使用`AUTONOMOUS_REFILLED`，否则typed拒绝。autonomous mode下camera一次arm整个run session、FPGA一次fire并自主执行；获准refill的host只供应冻结chunk，不逐point调度。
3. autonomous mode的preflight pin active Q0 qualification revision/digest与ProgrammedImageDeploymentRecordRef，单次FIRE通过与qualification/deployment mutation串行的`pin_for_fire` gate取得run级QualificationFireAuthorization；adapter按Q0-qualified delivery-order contract将frame[i]映射为frozen TriggerKey[i]，全链数据保持PROVISIONAL。ScanPlan只声明`required_association_proof`；run末端用唯一I/O owner按H1规则读取的AutonomousTableTerminalEvidence、绑定compiled/H1/deployment revisions的`PostTerminalTailEvidence`、`expected_trigger_total_from_completed_schedule`、按Q0 reset/rollover语义唯一unwrap的`camera_produced_delta`、frame/camera stamps、timestamp容差、coverage/EOS完成EndAttestation后，EpochValidationRecord才写`achieved_association_proof`。
4. 迁scan-slot/API-slot request、ScanOutputContract、multidimensional y和ScanArtifact Repository；MOT只允许SCAN_SLOT/AUTONOMOUS_STREAMED，不加API或host-stepped fallback。API_SLOT无法无缝更新时仅沿用`API_SLOT_SEGMENTED_EXISTING`，且program必须直接保存canonical非空`segmentation_rationale`说明为何允许段间host gap；只有接受任意可变非负gap的实验可用，连续物理演化、段间状态不可重建、最大gap、精确settle/re-equilibration或gap-dependent physics一律typed拒绝，不建立单字段policy wrapper。API在解析/编译前先准入R×P control memory，只解析/编译P个唯一point artifacts；整run一次arm camera并冻结aggregate frame budget，按R-major/P-fast为每cell建立独立STATIC_ONCE PulseSession/physical terminal，相邻segment只做camera required interval的deadline/cancel-aware保守最小等待，全部cell后才做一次camera complete和aggregate EndAttestation。任何pulse terminal或aggregate camera evidence失效都阻止下一次FIRE并使整run INVALID。artifact记录execution_mode、一个run级camera authority/CameraRunEvidence、有序ApiSegmentEvidence、segmentation_rationale、required/achieved proof和formal eligibility；baseline不保存per-boundary timing，也不保存per-segment camera authorization/terminal/attestation。
5. 对drop/reorder/duplicate/short read/counter reset/timestamp gap、pre-arm/session-baseline混用、camera max-inflight ring不足、host total retention不足、raw DONE早于delay tail、tail bound/version/evidence缺失、refill证明缺失、旧`scan_repeats`多发point、schema generation、component invalidity、RemoteSequencer abort与provisional epoch做整runreject-and-redo真机测试；重试默认手动，自动策略必须显式有界并保存失败attempt。
6. E2E 后删除 positional zip、latest fallback、旧 PulseScan 与 neutral key 泄漏进 FPGA 的类型。

W3d/W3e兑现S4中不依赖Q0/real composition的source-neutral软件纵切与current TaskConsole SCAN_SLOT入口：direct-camera与camera→occupancy都复用现有一次arm/一次autonomous FIRE exact pipeline，在同一Run提交唯一ScanArtifact；`(R,P,*data_shape)`、PointLayout、ComponentValidity和processor/calibration lineage可重开。direct-camera Qt仍FINAL-only；occupancy Qt可从同一exact builder的只读revision reader显示post-safety前明确标注的PROVISIONAL curve，释放worker-owned Agg/board后再从唯一artifact独立投影FINAL。TaskConsole已用静态catalog、共享revision editor和current-only Save/Load接入该panel，但仍明确标为virtual/offline，不冒充Formal EndAttestation。真实hardware/Q0/terminal evidence gate仍NO-GO；旧PulseScanNode/TaskConsole宿主因API_SLOT与其它panel最后consumer尚在而不能提前删除。

### S5：Workbench、其余 use cases 与用户兼容

1. 在 S0.5 宿主上迁剩余 WorkspaceModel/RunCoordinator/controllers；Setting/Edit 使用共享 EditorSession/schema widgets/base-revision conflict，catalog 只做各 bounded context capability/definition 的本地投影。
2. Fit 保持 zlc_data 单一算法 owner；U0.3d已把旧独立`.fit_gui()` host并入唯一DataFigure/Fit editor、draft overlay与explicit Save/reopen/Clear路径，W8b的selection-only CommittedTransform也由同一authority消费；U0.3e已让TaskConsole的`Add Analysis -> Fit`从当前FINAL ScanArtifact进入该host，仍不从display/selector生成权威、不复制模型表/solver、不建立formal workflow。未来自动分析只按§10.5真实consumer门槛另立flat Run；Pulse prepare/fire/safe继续后台托管。
3. 所有 panel 使用 S1/S2 的 render/evaluation lanes；完成 acknowledgement-driven shutdown、persistent quarantine 和 ControlTopic terminal/superseded ack。
4. 逐条迁 temperature、MOT、readout、device manager 和 notebook convenience；W4a-W4e、W5-W8与M1-M2e只记录各自已经证明的data/authority/预算/线程/lifecycle子合同，不再被解释为终态用户面。whole-Run ROI replacement、永久 frozen generic viewer、缺 selector/zoom/re-fit 与独立 `fit_gui` 作为主入口均先按U0三项纠正关闭；随后再完成通用live grid/动态layout、TaskConsole/scan/live Fit、ROI/transform authoring、可zoom/export的live calibration/occupancy grid、跨artifact saved gallery、real Pylon qualification及其它 convenience。每条按最后 consumer 做dependency-closed删除，不能提前删共享producer，也不能以“不重开旧审查”为由拒绝补齐缺失UX。
5. 真实入口 E2E 覆盖 fit/gridplot、calibration/occupancy、PulseScan、save/load、cancel/quarantine、shutdown 和 virtual/real adapter parity。
6. `OptimizeMotFieldTask` 迁走后删除 `TaskOutput`；每个删除项由“移走最后一个消费者”的切片负责，而不是由第一个碰到该类型的切片负责。
7. 最后一个 consumer 消失时物理删除 `neutral_atom/core`，不保留空 re-export 包。

### Z0：零残余审计

- legacy path/symbol/reader、历史pulse importer/fixture、一次性pulse转换器、双 registry、双 codec、双 fit owner 全部为 0；
- 旧DeviceSet/Registry installation composition、进程内hot-swap state/intent/recovery-context、available/unavailable state union与connection-establishment lease全部为0；目标只剩process-lifetime InstallationRuntime、immutable InstallationDeviceGraph、startup、claim-first recovery和shutdown；
- camera adapter 只有 record-preserving acquisition contract；array-only `read_frames()/acquire()`、平行 image/metadata queue 和可丢metadata的 public convenience path 为 0；
- reverse import 为 0，FPGA domain key 泄漏为 0，stream 上的累计 DataBlock 为 0；
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
    installation/
      runtime/
      graph/
      bootstrap/
      recovery/
      device_catalog/
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
    devices/adapters/        # composition私有concrete adapters/raw graph
    adapter_sdk/             # adapter作者合同，不从Experiment可达
    testing/                 # contract kits/fault fixtures
    simulation/              # 显式virtual环境，不经notebook umbrella导出raw类
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

删除由**最后一个真实 consumer 消失的 dependency-closed 切片**负责。不得因为 S1/S2 首次建立替代品就提前删掉仍被 S3/S5 使用的能力，也不得以“还有别的 consumer”为由让已迁 use case 继续双写/双读。当前已经删除 legacy `CalibrateReadoutTask`、`OccupancyProcessor`、旧 calibration report 与 `default_imaging_template()` factory；W3e 已把 public `Experiment.task_console()` 的 SCAN_SLOT Add/Setting/Edit/Start/Stop/Save/Load 与 camera→occupancy-counts y 切到 current typed product，但 legacy TaskConsole/PulseScan 文件仍是 API_SLOT segmented、其它 Measurement/Processor、rolling/gridplot/selector/calibration/temperature/MOT panel 与旧入口的共同宿主，不能在这些最后 consumer 迁走前整文件删除。其余至少固定：已迁 standalone camera use case 的旧显示旁路 -> S1；共享 generic camera producer/LogicNode、legacy live panel、monitor/rolling/ROI/temperature/readout UI 与 array-only reader -> 实际最后消费者所在的 dependency-closed slice（当前预计 S5）；旧 fitting/selection/facet/raster -> 其最后一个 legacy frontend/processor consumer 所在的 S3/S5/Z0；旧 positional/latest-polling SCAN_SLOT 分支 -> API_SLOT 与其它 legacy panel 不再需要其宿主后的 dependency-closed cut；旧 `API_SLOT_SEGMENTED_EXISTING` -> 下一专属纵切交付 current replacement 后删除对应分支；`TaskOutput` -> `OptimizeMotFieldTask` 最后消费者所在的 S5；LegacyPanelHost、LegacyRuntimeFence、SerializedLegacyAggBridge、剩余 TaskConsole god shell -> 最后 consumer 的 S5/Z0。历史 pulse parser/upgrader、三个 file/figure call site以及 `PulseTableState`、`PulseSequence`、`PortCatalog` 的软件改稿数字版本已在 Rules 2/3 追溯切片直接删除，因为 tracked 旧格式数据和合法消费者均为 0；仍被 Camera/PulseScan/PulseGUI 岛使用的 current legacy writer/reader、runtime wire reader与compiled sibling继续由最后 consumer 的 H1/S3/S5 dependency-closed cut 删除，不能把“current consumer仍在”误写成保留历史升级链的理由。所有 tracked pulse JSON 已是当前 `PulseDocument` 格式，不建立转换阶段。每项在本文件的切片清单记录 replacement、全部 consumers、shared ResourceKeys、first migrated slice、last consumer slice 与物理删除证据。

完成态不存在：

- `neutral_atom.core`；
- zlc_data -> frontend/neutral/pulse/workbench import；
- frontend -> neutral_atom/pulse 反向 import；
- pulse/FPGA -> neutral_atom/frontend/data 反向 import；
- async ExecutionEngine、child run、递归 plan；
- 旧 `DeviceSet`/Registry composition、`InstallationSupervisor`、`InstallationState`、`AvailableInstallationState`、`UnavailableInstallationState`、`InstallationCandidate`、`DeviceSwapIntent`、`SwapRecoveryContext`、进程内 config/device hot-swap 状态机，以及 `ConnectionEstablishmentLease/begin_connection_establishment`；
- continuous-exact epoch/spool 与专用 command/build lane；
- UnitSpec/CoordinateFrame 图代数；只保留 canonical unit id、opaque frame id 与显式转换；
- 默认 SnapshotLease；零拷贝只允许 profiling 驱动的 opt-in BorrowedSnapshot；
- node-owned worker/thread/terminal state 与运行中动态 pipeline edge；
- public Task/Measurement/StreamProcessor/Analysis god base hierarchy；
- TaskOutput 和 `__task_frame__`；
- per-signal gap -> latest fallback；
- 独立设备按 sequence zip、自由运行无 tag 的位置式 trigger ledger；
- FPGA 内的 neutral TriggerKey/ScanCellKey/ScanPlan 类型；
- sample stream edge 上的累计 DataBlock 与把 DatasetBuilder 伪装成 Processor；
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
- pulse历史parser/fixture/一次性转换器、逐版本upgrade链、旧schema writer、多点legacy call site、旧compiler bridge、runtime/wire旧reader与compiled `_program.json` sibling；
- tracked FPGA fallback literals；
- 文档/host把当前RTL并不存在的CRC/BANK_VERIFIED/逐沿receipt冒充既有能力；
- stale PreparedProgramRef、未冻结完整scan table、运行中读取mutable GUI slot或失败cleanup后无条件release；
- public `NeutralAtomSession.devices/.camera/.sequencer`、`Experiment.devices` raw alias，以及从 public facade/catalog/GUI object graph 反向取得 DeviceSet、adapter、SDK handle、resolver 或 drive bound method 的路径；
- TaskConsole 的 public `session` constructor/property、返回真实node的running_nodes；PulseGUI的raw`sequencer/experiment` constructor/fallback与standalone自建real RemoteSequencer；DeviceViewer/Manager的DeviceSet/provider/raw editable setter；
- 普通 `Zou_lab_control`/`neutral_atom` umbrella 中的 `BaseDevice`、`CameraDevice`、`SequencerDevice`、`TrapArrayDevice`、`DeviceSet`、`load_devices`、raw `bind_pulse/triggered_frames/PulseController`、`QCMOSCamera`、`ManualSequencer`、`RemoteSequencer`、`VirtualCamera/VirtualSequencer/VirtualTrapArray`；其中旧`DeviceSet/load_devices`随最后legacy consumer物理删除，不进入目标私有composition；concrete adapter能力按职责只留在adapter_sdk私有实现、testing/simulation或server CLI owner namespace；
- 同一umbrella中的adapter/bootstrap/server入口 `register_device_class`、`device_class_registry`、`apply_device_overrides`、`validate_device_contract`、`SequencerService`、`CommandSequencerBackend`、`serve_runtime_sequencer`、`run_sequencer_server`；
- 对上述raw symbol的warning alias、`__getattr__` fallback、兼容proxy、generic snapshot service locator，以及除密封 notebook `connect/Experiment` 外的 broad umbrella re-export；
- 保护文件位置、继承树和私有 GUI 结构的 public contract tests。

## 22. 最终验收

### 22.1 追溯整改范围与当前 gate

#### 规则 8：活动迁移测试与旧树边界

迁移结构尚未闭合期间，`tests/migration_active_tests.txt` 是**当前唯一授权证据面**，不再只是终态 inventory。每个 commit 在提交前必须同时完成：机械读取清单中全部非空、非注释路径；对全部路径执行一次 `pytest --collect-only`；再实跑同一完整集合；collection error、test failure、unexpected skip/xpass 均为 0。只跑“本切片直接相关”的子集可以作为开发中反馈，但不能替代 commit gate。活动错误清单只允许减少，不能为了让提交通过而删除路径、放宽断言、增加 skip/xfail 或把失败移到清单外。

仓库级 `conftest.py` 必须机械拒绝任何未在 manifest 中登记的测试文件被收集或显式调用，也拒绝 manifest 中不存在、重复、越界或非测试路径；该守卫本身属于活动白名单证据。调用者不能把任意 path、目录、自动 discovery 或清单外依赖带入命令。清单外历史测试一律冻结：不执行、不修改，也不因其预期改变 target production；只有其对应 use case 在终态切片重写为新架构独立 oracle 时，才以同一精确 commit 加入 manifest。曾误加并在仓库内调用 `pytest.main()` 的 `tests/run_migration_active.py` 已物理删除，不恢复第二个 runner；标准 pytest + manifest/conftest 是唯一执行机制。

本修宪 adoption 前以`HEAD` tracked manifest（2026-07-17，85个路径；不混入另一个未提交W9切片的manifest行）得到的首次全量基线为`798 collected / 24 collection errors`。这说明旧门规尚未兑现，不能伪写为green；修宪文档提交只负责让新门规生效，是唯一一次 adoption bootstrap，不是实现checkpoint。自该提交之后，**不得提交任何其它代码、测试或文档**，直到一个原子 Rule-8 gate-repair 工作树同时把以下库存降为0、建立conftest机械拒绝并让全量collect+run全绿；该修复自身只在最终全绿时一次提交，不能拆成仍红的中间commit。此例外不可复用：

```text
test_camera_capture_endpoint.py
test_camera_sample_contract.py
test_capture_artifact_repository.py
test_capture_checkpoint_commit.py
test_capture_fit_artifact.py
test_capture_repository_authority.py
test_compiled_capture_cell_plan.py
test_pulse_command_port.py
test_remote_sequencer_execution_endpoint.py
test_runtime_capture.py
test_runtime_commit_journal.py
test_runtime_dataset_builder.py
test_runtime_resource_arbiter.py
test_runtime_run_lifecycle.py
test_sequencer_execution_endpoint.py
test_stream_processor_worker.py
test_triggered_capture_pipeline.py
test_workbench_endpoint_binding.py
test_workbench_host_contract.py
test_zlc_calibration_commit.py
test_zlc_calibration_stream_contracts.py
test_zlc_occupancy_processor.py
test_zlc_pulse_playback.py
test_zlc_readout_physical_context.py
```

**Rule-8 adoption bootstrap 完成 checkpoint（2026-07-17）。** tracked manifest 从上述 adoption 基线的85个路径只新增机械守卫`tests/test_migration_manifest_gate.py`，没有删除、改名或移出任何授权路径，现为86个路径。仓库级`conftest.py`已删除活动测试不再消费的旧DeviceSet/legacy GUI/runtime fixture，并机械拒绝目录/自动发现、清单外路径、重复/缺失/越界/非canonical记录，以及整清单中零item文件；标准pytest仍是唯一runner，没有嵌套`pytest.main()`、hot-swap singleton或生产compatibility。一次全清单`--collect-only`得到`1072 collected`；随后严格按process-lifetime installation边界让86个路径各在独立标准pytest进程中实跑，结果为`1069 passed / 3 expected platform skips / 0 failed / 0 xfailed-xpassed`。三个skip分别是Windows不具备POSIX fork语义的journal/repository lease合同两项，以及本平台`longdouble == float64`的一项canonical宽浮点合同，均非本轮新增或规避失败。

本修复把capture、runtime、pulse、readout/calibration与notebook活动测试改写为current target contract：不恢复`LegacyRuntime`、`DeviceSet`、`CommitKind`、checkpoint/memory journal、旧materializer或跨测试模块helper；多维capture继续保留`(R,P,*data_shape)`及具名轴，final-only repository authority、process-lifetime installation、packed trigger schedule与admitted-capture fit均由独立oracle覆盖。对抗审查曾找到两个会被过度压缩误删的真实物理安全oracle：相机最小触发间隔必须在精确等号接受、`nextafter`拒绝、单沿无相邻间隔和autonomous跨scan-point全局最短四种边界成立；相机`arm()`返回后必须再次验证settings/trigger wiring，失败不得发布started ack且必须terminalize/disarm。整改又补齐produced-count、frame stamp、camera stamp与timestamp倒退的逐项fail-closed oracle；同一审查员定向复核确认这些删除/边界回归都会使测试失败，最终结论`P0=0 / P1=0 / GO`。

Rule-6以parent `3211128`机械计数：除混合W4文件中只属于本轮的一个histogram helper hunk外，45个current测试文件从`28442 -> 19056 physical`、`25040 -> 16621 nonblank`，比为`0.6700x/0.6638x`，净减`9386/8419`行；W4本轮hunk另为`+8/-7`，并行W9内容不计入。七个非混合production Python owner为`6974 -> 6971 physical`、`6457 -> 6466 nonblank`，比为`0.9996x/1.0014x`；classes`28 -> 28`、dataclasses`11 -> 11`、enums`0 -> 0`、functions/methods`251 -> 245`。函数减少来自把pulse/virtual重复primitive validator委托给`zlc_storage.canonical`；其它生产差量只有current格式名去编辑计数、`FramedJournalSession.close()`在显式unlock失败后的不可逆authority撤销、两个原生`QScrollArea`改用已有`FluentScrollArea`，以及scan application从已移除的`pulse_document`迁到唯一权威`PulseScanProgram`并只允许`AutonomousScanSlotProgram`提供progressive preview；没有新wrapper/framework。`git diff --check`、55个暂存Python文件AST compile、import-DAG、canonical单owner、added-diff shape-loss与活动测试跨模块import扫描均通过。下一产品checkpoint回到U0第1项Monitor/ROI源Run不断流纠正；不得再重开本bootstrap或把其测试压缩外推成UX/真机资格完成。

当前闭包证据包括：完整活动白名单 collect+run、该切片新增的独立 oracle，以及 changed-file/owner 账本、AST/symbol/consumer 搜索、静态 import-DAG、canonical/digest/版本常量/重复 validator 扫描、`py_compile`/AST parse、main 行数与物理算法只读对照、手写 current-format 确定性 oracle 和必要的真实 profile。测试通过不授权为旧 API 补 compatibility，机械证据也不能冒充完整产品行为验收。本节之前属于清单外历史 suite 的数字只保留为历史 checkpoint；但已经进入活动 manifest 的路径从本修正案起每个 commit 都必须完整收集并实跑。

旧 `DeviceSet/Registry/session/GUI/public compatibility` 语义只允许登记“分支改过什么、最后消费者、删除切片”，或在dependency-closed切片中删除；禁止为它们新增适配、修守卫或追求旧测试通过。尚未迁址、但仍被target endpoint作为唯一真实硬件owner消费的adapter文件可以接受证据驱动的物理正确性修复，例如DCAM count/ring/metadata边界；这种修复不得恢复旧composition、registry或public API。main仍可只读用于行数锚点和已验证物理/算法oracle。本轮`discovery.py/registry.py`的worktree变化是精确恢复main、撤销此前错误的target适配，不是把旧接口接回目标包。教程、手册与旧GUI说明在迁移完成前不做同步；本设计文档例外，任何计划级取舍必须与代码同commit更新。

本门规来自一次已复现的工作流失败：审查过程反复让历史测试和 legacy `DeviceSet/Registry` 竞态反向驱动目标架构，造成无价值守卫、镜像测试和重复执行。故它不是通用预防机制。违反测试冻结边界或触碰旧树适配，即使偶然得到测试通过也不能形成闭包证据。

#### 规则 9：UX 行为忠实与偏离治理

UX 权威为 `main` 分支旧实现，与规则 4 的物理/数值权威同级。每个 GUI/交互切片必须先完成 §2.1 salvage gate，再以真实 launcher/input event 逐项证明旧行为清单；内部重构、线程模型、包边界或实现难度都不能自行删减 selector 覆盖、zoom/pan、live 连续性、ROI 热更新、fit 即时性、relim/cmap/limits、布局手感与保存/恢复。未达到项必须登记 §2.2，状态保持待用户批准；实现 commit、测试通过、checkpoint 或审查结论都不能把偏离自行改成批准。迁移完成报告首页必须列出全部未批准项。

规则 1–7 的 owner 审查集合、规则 8 的活动测试集合与规则 9 的 UX 行为清单都必须由 Git/manifest/salvage checkpoint 机械产生，不能靠上下文记忆或“我记得写过哪些提交”。整改启动时的固定证据是 `git rev-list --count main..HEAD = 113`、`git diff --name-only main...HEAD = 315`；在纠正误删并完成 tracked-fixture 整改的 checkpoint `12f3414`，分支变为 139/323；清除错误 calibration 历史并把唯一 oracle 重新冻结到 exact main 后，checkpoint `455ff97` 为 160 个领先提交、330 个 changed files；`e31cb9407ae8` 为 161/330；完成 superseded calibration/readout graph 删除后的 checkpoint `5e2e2f6` 为 170/375；清除当前活动代码、构建与审计 skip 中的历史 reference/archive 残余后，checkpoint `8285e43` 为 177/386。后续数字会随整改提交增长，因此报告必须同时写 checkpoint commit，不能把任一组历史数字当永久常量。

本文件是目标架构、迁移顺序和 Rules 1–9 gate 的唯一计划级权威。`docs/MAINTAINER_NOTES.md` 以及 `Zou_lab_control/frontend` 内的说明和结构测试只描述仍在 `SerializedLegacyAggBridge` 后运行的 legacy island；它们可以在最后 consumer 迁走前保护现状，但不得反向规定 `zlc_frontend`、`zlc_workbench` 或其它目标包的类型、继承树和 UI/runtime 架构。旧 roadmap 不再作为并行计划源；计划级决定必须与实现同 commit 回填本文件。

验收必须分成两个互不替代的轴，避免“未来功能没实现，所以现有代码永远审不完；现有代码没审完，所以又不准实现未来功能”的闭环死锁：

- **owner audit status**只回答当前changed-file集合中的既有实现是否完整执行Rules 1–7，取值`NOT_STARTED/PARTIAL/COMPLETE`。未来尚未实现的`InstallationRuntime`、Q0、EndAttestation、API segmented或GUI consumer不会把已经审完的既有owner重新降为PARTIAL；它们作为明确backlog进入下一轴。规则 8/9 是跨 owner 的每提交 gate，不能由一次历史 owner COMPLETE 永久豁免。
- **implementation/capability status**回答目标纵切是否已经交付、真实设备/用户入口是否可启用，取值`BACKLOG/IMPLEMENTED/READY/NO-GO`。audit COMPLETE绝不等于capability READY；例如真实qCMOS exact在当前代码fail-closed仍可完成adapter owner审查，但Formal capability继续NO-GO。

Rules 1–9 的 canonical 定义与 COMPLETE 条件如下。前七条评估 existing-code owner audit；第八、九条是整个迁移阶段持续生效的操作/产品门禁：

| rule | 名称与目的 | 必需证据 | owner COMPLETE 条件 |
|---:|---|---|---|
| 1 | 单一owner、DRY与边界：同一fact、validator、digest、codec、lifecycle或terminal不得多头拥有 | 完整symbol/AST/call-site/import-DAG与producer→consumer追踪 | 所有重复owner已合并或证明是不同不变量；反向依赖与绕过authority路径为0 |
| 2 | 无历史残余且按真实consumer裁决：不保留alias、fallback、archive、自证fixture或无语义wrapper，也不因暂未接线误删已排期能力 | Git历史/全仓reachability/目标consumer/最后consumer删除账本 | 每个保留抽象能指出真实当前或已排期consumer；每个legacy面有dependency-closed删除点；无兼容残余 |
| 3 | current-only持久与wire语义：软件edit counter、upgrade chain、dual reader不得冒充部署ABI | schema/codec/version常量/asset全仓扫描与owner serialization核对 | 普通软件格式只有plain current schema+exact admission；仅真实跨进程/硬件ABI保留版本或generation |
| 4 | 物理与数值忠实：迁移不得改变main已验证实验算法或硬件事实，差异必须有物理原因 | main只读逐项/同帧oracle、硬件手册/本地driver事实、量纲与边界反例 | 等价物逐项一致；每个故意差异有明确理由、输入域与fail-closed边界；多维轴/validity不丢 |
| 5 | 独立证据而非实现自证：writer/reader、两实例或同公式镜像不能证明正确 | 手写primitive tree/literal golden、malformed input、独立数学/物理oracle、静态结构检查及活动白名单测试 | 关键identity、codec、mapping、terminal与异常路径至少有一种不调用被审owner的独立证据；相关 oracle 纳入规则 8 证据面 |
| 6 | 海拔/复杂度必须对得起能力：逐切片报告current→main PLOC/NCLOC/class/dataclass/enum/function与热点资源 | 机械计数、超过约3倍的逐抽象理由、针对大帧/大scan/profile | 无单成员enum、单实例class、纯转发wrapper或重复DTO；高倍率项逐一压缩或说明“删除它会恢复哪类已证实问题” |
| 7 | 对抗式可实现性、生命周期与性能：从反例而非happy path找静默错、竞态、无界资源和硬件不可实现承诺 | 独立agent/人工反例、deadline/cancel/ownership/overrun分析、针对性profile与静态故障路径 | 当前既有代码无未处置P0/P1；P2有清楚触发条件；未来缺失能力转入implementation backlog而不伪称READY |
| 8 | 活动迁移测试与旧树冻结 | manifest 全路径机械校验、完整 collect-only、完整实跑、conftest 越界拒绝、本轮 staged diff 审计 | 每个 commit 的活动白名单 collection/test 全绿且错误清单只减不增；清单外历史测试未运行/改写，未归属 test diff 不进入提交或证据 |
| 9 | UX 行为忠实与偏离治理 | `main` salvage 清单、真实 launcher/input-event E2E、§2.2 偏离账本 | 只有全部条目为`MATCHED`或用户明确批准的`APPROVED_DEVIATION`才COMPLETE；`LEDGERED_PENDING_APPROVAL`可让正交整改继续，但Rule 9保持PARTIAL、最终迁移保持NO-GO |

`12f3414` 的 323 文件第一集合按顶层 owner 分布为：tests 142、旧树 `Zou_lab_control` 48、`zlc_neutral_atom` 42、`zlc_pulse` 23、`zlc_data` 20、`zlc_workbench` 10、`zlc_frontend` 8、FPGA host/config 8、tracked pulse assets 6、`zlc_storage` 6、tutorials 5、docs 2、root/config 3；文件类型为 Python 298、JSON 9、Markdown 7、notebook 5，其余 TOML/gitignore/TeX body/batch 各 1。每个文件必须分配一个 primary owner/slice，测试跟随被测 owner审查；跨域文件还要记录 secondary seam，不能因“另一路 agent 看过相邻包”漏掉。

机械覆盖至少包含：完整 changed-file set；Python AST 的 class/dataclass/enum/function/import/call-site 清点；重复 validator/digest/codec/版本常量的符号与语义搜索；production + 已排期 target consumer 图；main 等价物的 NCLOC/class/dataclass 比；对大帧 digest/copy、fit/calibration、journal、pulse compile/pack 的针对性 profile；活动 manifest 的完整 collect/run；以及 GUI 切片的 `main` UX salvage/E2E。清单外历史测试的 oracle/fixture provenance只保留为历史账本，不重新打开或运行；当前 oracle 必须成为活动白名单的一部分。每条规则的完成报告必须给出覆盖 checkpoint、发现总数（包含点名样本之外的新发现）、净行数变化、涉及 owner、独立验证和未来 ratchet。只给抽样或只说测试通过，不算完成。

为防止上下文压缩后重复审查、漏审或把secondary seam误算成新闭包，最终production owner账本固定为下列11个互斥闭包。P01/H01追溯轮开始快照为`merge-base(main, HEAD)=6c337d49`、HEAD `9b2662aae8cb`：`main..HEAD=190`个提交、`main...HEAD=387`个路径。当前worktree相对main机械去重为403个路径，其中188个是production `.py/.json/.toml/.bat/.tcl`（排除tests/docs/tutorials）。从旧194到188的六项变化可机械解释：`zlc_workbench/legacy_neutral_atom.py`、`legacy_runtime.py`、`pulse_control.py`已物理删除，`devices/discovery.py`、`devices/registry.py`与旧`subsystems/timing.py`已精确恢复main，不再属于target changed set。188个production文件全部恰好分配一次；未归属的测试、文档、教程脏改不进入本轮审查证据或提交。本节后续保留各已提交checkpoint在**当时**的局部GO/NO-GO与“迁移继续冻结”字样作为审计因果，它们不是当前门规；当前状态的唯一来源是下表后的现态段以及本节末尾总状态段，后写状态明确覆盖较早checkpoint措辞。

| owner id | 唯一 primary scope | production 文件数 | owner audit status | implementation / capability status |
|---|---|---:|---|---|
| `D01_DATA_FIT` | `zlc_data/*` | 18 | COMPLETE | core READY；产品入口仍随E/U/A纵切接线 |
| `S01_STORAGE` | `zlc_storage/*` | 7 | COMPLETE | READY；当前journal生命周期接缝已随R02复验 |
| `R01_STREAM_DATASET_PROCESSOR` | `zlc_neutral_atom/processing/*`、`runtime/dataset.py`、`runtime/streams.py` | 6 | COMPLETE | exact/live kernel、W1 finite接线与M2a短bounded raw-history接线READY；长scalar rolling/动态multipanel仍属U01 backlog |
| `N01_READOUT_CAPTURE` | target acquisition/artifacts/readout/timing（pulse 除外） | 26 | COMPLETE | readout/calibration/capture core IMPLEMENTED；真实Formal硬件资格仍受H01约束 |
| `F01_FIGURE_VIEW` | `zlc_frontend` figure/core owner | 7 | COMPLETE | headless figure/view READY；完整render/GUI属于U01 backlog |
| `R02_RUNTIME_INSTALLATION` | target runtime（dataset/streams除外）、catalog、全部Workbench endpoint，以及尚待删除的旧`device_catalog/installation`边界 | 22 | COMPLETE：existing leaves与本轮composition差量均完成target-only复审 | virtual process-lifetime InstallationRuntime/graph READY；真实AssetMap/physical-owner/recovery composition仍NO-GO |
| `P01_PULSE_EXECUTION` | `zlc_pulse/*`、5个pulse JSON、`timing/pulse.py`、旧timing persistence/table/sequence最后消费者 | 33 | COMPLETE：33/33 existing production paths逐文件覆盖，current codec/resource、resident transport、schedule、terminal与legacy删除账闭合，P0/P1=0 | resident autonomous core READY；experiment ScanContract与`API_SLOT_SEGMENTED_EXISTING`为BACKLOG |
| `H01_HARDWARE_ADAPTERS` | `fpga/*`、旧devices中仍实际changed的真实adapter（不含已恢复main的registry/discovery与A01 public init）、adapter SDK、simulation/testing | 22 | COMPLETE：22/22 existing production paths逐文件覆盖，冻结RTL、host transport及当前adapter seam P0/P1=0 | real qCMOS Formal exact NO-GO，直到typed Q0、tail/EOS、EndAttestation与真机资格化完成；当前正确fail-closed |
| `U01_GUI_WORKBENCH` | frontend render、Workbench shell、旧frontend与`pulse_gui.py` launcher | 17 | COMPLETE：existing primary paths及最新W7差量复审闭合 | board identity/final gate、virtual/offscreen single-event finite capture及其完整raw IMAGE交互、显式ROI scalar history、coherent四panel、fixed-ROI front-bound selector/overlay、raw committed-capture Fit editor/draft/Save、formal calibration creation/edit及exact saved-fit GridPlot产品IMPLEMENTED；raw-only首ROI、TaskConsole/scan/live grid Fit、通用scheduler/PanelHost、rolling grid与legacy GUI迁移BACKLOG/NO-GO |
| `E01_EXPERIMENT_OPERATIONS` | 旧neutral core、operations、measurements、processors、tasks、subsystems | 22 | COMPLETE：existing primary paths按task/measurement/processor语义及最后consumer账闭合 | raw exact coordinator IMPLEMENTED；Formal ScanOutput与旧operations迁移BACKLOG，真实链NO-GO |
| `A01_PUBLIC_SURFACE` | `pyproject.toml`、包根/umbrella、`_gui`、manual launcher、notebook public facade | 8 | COMPLETE：existing primary paths及本轮notebook/packaging/W1差量复审闭合 | target virtual notebook与lazy finite-capture Workbench入口READY；legacy launchers、完整Workbench与真实硬件composition仍NO-GO |

计数仍按审查冻结快照为`18+7+6+26+7+22+33+22+17+22+8=188`；新增implementation文件作为其owner的差量复验记录，不伪造为第十二个闭包。该 checkpoint 只证明11个owner的Rules 1–7 existing-code审查在当时COMPLETE，并曾解除“先完成1–7”的冻结；它不能一次性完成持续生效的规则8/9。自本修正案起，规则8必须每commit完整白名单全绿，规则9必须每GUI切片salvage/ledger/E2E；当前只能按U0先完成机械门和三项UX纠正。解除历史owner冻结、局部READY与全迁移完成是三件不同的事：真实qCMOS/remote composition、完整Workbench/legacy launchers与Formal Scan仍按implementation轴NO-GO；W1的窄Qt产品READY不能由owner audit COMPLETE外推成这些能力可用。

`P01_PULSE_EXECUTION` 以本节记录的 `9b2662aae8cb` 快照为差量基线，对33个primary production path逐文件阅读，并沿current document→compiler→TargetIR→wire/schedule→artifact/RPC/storage→neutral/workbench consumer追踪完整producer-consumer闭包；旧 timing/PulseTable 只登记最后消费者与删除点，没有用旧测试或旧`DeviceSet/Registry`语义反向修改target。收口共修复9类不重叠根因：逐edge Python对象图放大、TargetIR scan canonical list放大、canonical producer缺少与decoder同源的结构资源域、artifact aggregate edge/byte总预算缺口、准入前复制、下游可放宽owner上限、wire `<u4` 前未拒绝越界地址、空`steady_rises`在巨大loop count下空转，以及旧`.edges`物化API/重复session state残余。`DigitalTriggerSchedule`现以只读`<u4 point_indices>/<u4 loop_iterations>/<u8 ticks>`保存物理列；channel由schedule拥有，global ordinal为行号，point-local ordinal按声明的单调point列派生，`TriggerEdge`只是在`iter_edges()`消费边界产生的临时typed row。TargetIR scan table、duration与wire words也使用定宽array codec；这只改变内存/编码表示，不改变多维scan row、repeat、trigger归属或任何data trailing axis。

资源不变量由唯一owner分层而不复制公式：`zlc_storage.canonical` 的同一组node/container/array-count/array-byte admission同时服务`encode(..., limits=...)`和`decode(..., limits=...)`，decode canonical rebuild继续携调用者的同一policy；pulse owner另只拥有1,000,000 aggregate trigger edges与128 MiB compiled-artifact wire ceiling，RPC和capture repository只能收紧、不得放宽，并在`bytes()`复制前按bytes-like `nbytes`准入。此前可构造但不能被自己的默认decoder读回的333,400-row和1,365-empty-schedule反例现在分别在container-entry与array-count边界拒绝；显式1,100,000-node policy下的1,000,001-item current tree仍可encode/decode对称往返。151,552-edge确定性实例完整producer→artifact→typed decode为3,236,148 B，value/fingerprint一致且编解码前后三列均readonly；schedule/artifact/encode/decode约为0.104/0.062/0.018/0.237 s。4096-point样本相对旧逐对象表示把payload从约1.91 MiB降至0.36 MiB、compile peak从约24.1 MiB降至4.3 MiB；五份tracked current PulseDocument均用当前编译器、全部digital lanes和typed artifact codec往返。与旧实现逐行交叉的channel、point、loop、global/local ordinal与physical tick六字段完全相等；`steady_rises=()`配`UINT32_MAX` loop count约0.31 ms完成，证明没有恢复空循环。证据全部来自current手写oracle、真实tracked asset、静态consumer搜索、`py_compile`和profile，没有运行pytest。

P01本轮实际改变15个production文件，按本轮前HEAD比较为`PLOC 7561→7794`、`NCLOC 6830→7031`，净增233/201行（`1.031x/1.029x`）；最大单文件`schedule.py`为`1.46x`，没有超过3倍项。增量来自columnar schedule的显式不变量和canonical producer-side resource admission，同时artifact/compiler/server/timing路径已有净压缩；没有单成员enum、单实例class或纯转发wrapper。因此P01 existing-code owner audit为COMPLETE、P0/P1=0；resident autonomous compile/transport core READY，但experiment ScanContract与既有API-slot segmented产品接线仍是implementation backlog，不能由owner审查完成冒充真实实验capability完成。

`H01_HARDWARE_ADAPTERS` 对22个primary production path逐文件阅读：旧设备adapter与连接模块、`adapter_sdk.py`/`simulation.py`/`testing.py`、FPGA host/config/Tcl/batch；`devices.__init__`归A01，Workbench endpoint归R02，只在本轮作为secondary seam差量复验。冻结bitstream/RTL是最高约束，本轮没有修改任何RTL、bitstream、project timing或板上寄存器合同，也没有为架构偏好要求重烧。旧`discovery.py`/`registry.py`已精确恢复main，旧`subsystems/timing.py`也不再携`_device_set`、registry或pulse facade适配；这些旧树文件只登记最后消费者，不能为了旧测试重新加target bridge。

H01收口修复9类qCMOS/host根因：transfer-info失败、负count和count后退不再被猜成新帧；`nFrameCount+nNewestFrameIndex`共同决定冻结snapshot的真实ring slot并验证连续性；一次ready所代表的完整已报告backlog不再等待第二个event；overrun在copy前及exact copy后复核；deadline覆盖等待、SDK copy、owner ndarray copy与commit全调用而不再按每帧重置；Stop在各长阶段都可见；adapter hook在一个bounded driver snapshot后返回、由base owner消费pending而不把调用参数误作第二套总cardinality；terminal按`cap_stop→join active grab/copy/record construction→双读稳定final count/newest→release`排序，stop未确认时保留ring；最终record commit、cursor递增、cancel/deadline和terminalizing共用一个短线性化点。Workbench secondary seam只按capability-qualified `max_source_burst_events`分配driver ring，完整run retention仍归host exact stream/dataset，避免同一帧预算在相机ring再次整run分配。

独立current oracle用可阻塞record构造与fake DCAM交叉出`cap_stop`/cancel：结果必须是`AcquisitionCancelled`、零已提交record、cursor仍为0、reader退出后才release、final produced count恰为1；count/newest不连续、copy期间overrun、transfer-info失败和stop失败均走明确异常而非latest-frame fallback。两路独立锁序复核只发现`capture condition→base state condition`，没有反向获取；owner-copy后再次检查deadline/Stop以及commit/terminal共锁关闭了最后两个竞态。`py_compile`、真实import和静态调用面通过，未运行pytest。当前22文件为`PLOC/NCLOC/classes/dataclasses/enums/functions = 13472/12141/44/13/1/528`，对main同scope`13797/12387/48/12/0/575`为`0.976x/0.980x`行数；新增SDK/simulation/testing职责没有main等价物，不伪造单文件倍率，整体无3倍海拔债务。

因此H01 existing-code owner audit为COMPLETE、当前P0/P1=0，但real qCMOS Formal exact仍明确NO-GO。原因不是要改FPGA，而是typed Q0外触发资格、post-terminal tail/EOS与EndAttestation尚未交付且真机DCAM跨线程`cap_stop`语义仍需E0 contract kit证明；当前capability必须在arm前fail-closed。近期正式scan仍只允许冻结bitstream上的`AUTONOMOUS_STREAMED`，不得恢复`HOST_STEPPED_GROUP`。只有E0在工作余量内实测出丢帧/乱序，或证明现有RTL与既定物理设计不符，才可另立证据驱动的硬件修复切片；owner审查COMPLETE绝不把这些future/real-hardware gate伪写成READY。

该历史审查 gate 的准确含义是 **Rules 1–7 existing-code owner闭包在当时GO**，不是规则8/9永久COMPLETE，也不是任意新便利切片可开始。Calibration/readout/occupancy/raw CaptureArtifact 的main同帧oracle、流式内存、多轴physical context、exact triggered consumer、typed codec以及coordinator-backed FINAL持久化结论保留；direct-CAS `FitResultRepository` 不在这项FINAL持久提交结论内。real qCMOS Formal仍受E0/Q0 common-aperture、first-edge/run-boundary、ordered-trigger、tail/EOS与EndAttestation真机资格化约束，因此真实硬件capability继续NO-GO。后续每个commit须满足Rules 1–9中适用的增量证据，并先按U0纠正，不能把局部virtual/offline READY改写成全局迁移完成。

规则 1 的 storage journal 整改以“扫描边界一次解码”为唯一 owner：`FramedJournal._scan()` 在校验 frame 长度、digest 与 canonical record 后同时保留原始 payload 和已解码 value；原始 bytes 继续负责同 id 冲突/幂等判定，`records()` 与 `append_checked()` 只消费已验证 value，不得再次 decode 已有记录。新 append 的候选仍须经过 canonical `encode -> decode` 规范化，torn-tail repair、跨进程锁、fsync 与 corruption fail-closed 语义不变。调用计数测试必须锁定 N 条已有记录每次扫描恰好 N 次 decode，避免以后以 wrapper 或 getter 的形式恢复重复工作。

Pulse RPC 的 artifact message 不是第二个 wire owner：`encode_artifact_message/decode_artifact_message` 只能委托 `zlc_pulse.artifact` 的 current typed canonical codec，不能自行拼 `to_tree -> encode` 或 `decode -> from_tree`。这样 RPC、文件与本地调用共享同一 canonical/semantic admission；结构测试直接替换 owner 函数并验证委托，避免两个实现以后悄悄漂移。Prepared-ref 与 completion 仍由各自类型 owner 编码；此项不允许借机删除 pulse server 或任何现有/计划 consumer。

PulseTarget ABI、TargetIR、PulseWireImage、CompiledPulseArtifact 与已经通过构造验证的 terminal/tail evidence 都是 immutable value；其 canonical identity 在成功 `__post_init__`/root decode 的末尾计算一次，getter 只返回缓存，不为检测 `object.__setattr__` 反射而反复重建全树。Wire image 的 digest payload 与公开 tree 必须共用同一个不含 digest 的 private projection，禁止维护两份字段表。Raw STATUS/CURSOR/tail observation 的 u32、recipe、sample-count、稳定性与 artifact binding 仍在对应硬件证据 owner 边界验证，缓存不能提前到 raw observation，也不能替代 completion admission。golden digest 向量和“构造后 monkeypatch `canonical_digest`”结构测试共同锁定值不变与只计算一次。

Workbench 内本地 sequencer、远端 sequencer 与 camera endpoint 共享的当前性只由 `BoundDevice.binding_stamp` 表达；capability/provenance 直接携同一个 stamp，阶段 ACK 最多回显一个 `binding_instance_id`。不再用一一对应的 `binding_id + connection_generation` 两字段反复交叉验证。远端 pulse server generation、capability 及各设备自己的 live readback 是不同不变量，继续留在对应 endpoint，不得因本地 binding 收敛而合并或削弱。该收敛不改变 camera/sequencer 入口、GUI 计划 consumer 或设备能力。

Finite exact 的“event ordinal 到每个 dataset cell 的完整唯一排列”由 `FrozenDatasetEdge` 对 address 值作 defensive copy 后验证一次；同一结果派生 schedule、key-sequence 与 consumer-contract digest，并缓存 key-domain fingerprint。adapter/contract 是 composition-owned frozen generation owner，不为抵御同进程 `object.__setattr__` 再递归复制整张 graph。Workbench 的 `CameraCaptureBindingRequest` 只冻结 declarative sequence，`CaptureStreamContract` 直接消费 edge，processor worker 在确认同一个 edge/key-contract owner 与 reservation cardinality 后信任该 immutable schedule，不得再次逐 key 扫描。CompiledCaptureCellPlan 对 pulse trigger/scan/repeat/event 物理关联的验证仍保留，因为那是更强的不同不变量；reservation state、generation、consumer ownership、cursor 与锁内 TOCTOU 也不能删除。schedule-free edge 只复用 payload-to-cell projection；live 的 keyed cycle/append window 语义归独立 `MonitorDataset`，不再是 DatasetBuilder 的 `expected_cells=None` 分支。

Calibration/readout 的规则 4 子切片以 `main@6c337d49` 为唯一物理与算法权威并已完成逐项同帧核对。tracked oracle 从 synthetic raw camera frames冻结 main 的 detector、四种 site order、BOX mean/sum/median/max、per-site/uniform PSF与annulus、strict-consensus labels、seed-0 split、96-bin Otsu、120-bin empirical threshold、per-site/global fidelity、ablation和runtime occupancy；当前实现不在测试时加载第二套算法。四处明确偏离只有 invalid/nonfinite传播、反 polarity runtime拒绝、chance-level site禁用和完整 FrameContract applicability，均有独立 intentional-difference tests。当前证据集合 `test_zlc_readout_main_oracle.py + test_zlc_readout_validity.py + test_zlc_calibration_commit.py + test_zlc_calibration_stream_contracts.py` 由 `pytest --collect-only` 得到 51 个测试函数、60 个pytest items，覆盖 main physical golden、四类纠错、真实多轴/borrowed-frame/codec、真实CAS commit/recovery；fingerprint/version/WorkPlan、无消费者quality gate、resource-policy镜像和反射伪造测试随对应机制一起删除，不能按旧测试数量补镜像。规则 4 对 calibration/readout 已 COMPLETE，但全分支其它物理切片尚未逐一完成，因此全局规则 4 仍为 PARTIAL。

Calibration/readout Rule-6 清点使用 physical line、nonblank line、AST class/dataclass/enum/function 五种计数。当前 `calibration.py + analysis.py` 数值/领域 core 为 `3338 / 3058 / 25 / 20 / 4 / 99`，对 main 物理算法五文件 `1765 / 1556 / 6 / 6 / 0 / 64` 是 `1.89x / 1.97x` 行数；额外的 pulse/camera `physical_context.py` 为 `382 / 349 / 3 / 3 / 0 / 9`，main 没有同职责能力，禁止伪造一个倍率。六个 calibration primary-owner 文件（上述 core/context 加 `calibration_codec.py`、`calibration_repository.py`、`calibration_reference.py`）为 `5530 / 4998 / 30 / 24 / 4 / 205`，对 main 含 readout facade/task 的合理包络 `2791 / 2464 / 9 / 6 / 0 / 106` 是 `1.98x / 2.03x` 行数；这不是 dependency-closed 总和，共享的 `readout/contracts.py` 与 `readout/codec.py` 分别归通用 readout contract/codec 切片，若把依赖也纳入则必须单列，不能把 `5530/4998` 称为整个依赖闭包。class/dataclass 倍率较高，但无单成员 enum；4个 enum 均有多个真实分支，3种 feature 有训练/runtime 两端 dispatch，computation/analysis-result/resolved-calibration 分别承担非权威计算、source-admitted commit capability 与 runtime admission，不能按字段相似合并。

Occupancy operator 当前为 `687 PLOC / 596 nonblank / 8 classes`，对 main operator `750 / 687 / 2` 是 `0.92x / 0.87x`；加入 finite exact pipeline 为 `1105 / 963 / 12`，再加入 autonomous triggered coordinator 为 `1350 / 1183 / 15`。后者对 main 单独 operator 是 `1.80x / 1.72x`，但对 main 含 subsystem convenience 的完整 occupancy 包络 `1653 / 1484 / 5` 只有 `0.82x / 0.80x`；必须同时报告两种分母，不能选有利倍率。Capture lazy frame/repository 当前为 `2439 / 2270 / 8`；main 没有 chunked CAS、crash-safe manifest、lazy bounded reader 与 admitted-source 同职责实现，因此只报告绝对值和能力增量，不拿 camera measurement 几百行伪装成等价分母。共享 exact coordination + pulse binding/lineage 当前为 `286 / 243 / 3 classes / 2 dataclasses / 0 enum / 15 functions`，已经有 triggered capture 与 triggered occupancy 两个真实生产消费者；新增 binding 的存在理由是消除 spec/result/artifact 对同一 pulse/schedule/edge association 的重复复合验证。

该历史复杂度审查删除/收敛了四个会漂移的owner：Calibration与raw Capture的manifest visibility委托cycle-free runtime leaf（FitResult仍走direct CAS）；frame validity/record/chunk geometry由私有纯函数和值供scratch/stager/reader共用；analysis删除第二套annulus/background fallback；`PulseCaptureBinding`唯一验证compiled artifact、trigger schedule与edge关联。没有repository基类、workflow engine或版本框架。U0.3d后来因Capture与Scan成为第二个真实同构fit-result source，才把Capture-only实现收敛成closed `FitResultRepository`；这不改变该历史gate对零消费者wrapper、单成员enum、单实例class和重复安全/物理公式的禁止。

`zlc_neutral_atom.timing` 包根与 `readout` 一样保持空边界；capture artifact、Workbench、notebook和测试都从 `timing.capture`、`timing.capture_plan`、`timing.lineage`、`timing.pulse` leaf owner导入。普通 calibration/capture import 不再因为包根聚合而加载 triggered occupancy；禁止恢复宽 re-export 或 lazy compatibility表，import-DAG测试机械搜索 package-root consumer。

Calibration partition 的 deterministic hash split 只保留描述性 identity `fixed-hash-order`；`ALGORITHM_FIXED_HASH_ORDER_V1` 是尚无历史 consumer 的软件改稿编号，不是可协商格式或物理算法版本。删除编号不改变 split、digest 字段集合之外的算法、训练样本或任何 artifact capability。

Legacy pulse 的 Rules 2/3 追溯清点以 parent checkpoint `042e1b7` 的 171 个领先提交、377 个 changed files 为范围，并搜索全部 tracked JSON、三个格式 owner的AST、所有 `legacy_pulse_upgrade` import与 `PulseTableState/PulseSequence/PortCatalog.version` 引用。共确认 8 处残余：1 个116行历史 upgrader、3 个文件/figure调用点、3 个数字改稿版本及1处把历史升级链错误推迟到H1的删除边界；tracked 旧格式payload为0。切片删除upgrade模块与调用点，让三个仍跨会话使用的legacy软件值只保留各自朴素schema名并严格拒绝其它字段，production净减138行；不改变 `RuntimeSequenceProgram` 部署wire version、FPGA layout或`zlc_pulse.PulseTargetABI/v1`。格式字节改变会使相同物理序列第一次重新生成sequence id并重传，但ticks/masks/硬件时序不变；legacy RemoteSequencer client/server必须同release原子更新，混用时由exact-field admission明确失败。窄AST ratchet独占“不得恢复numeric version/upgrader”不变量；普通format tests只保留当前schema dispatch、exact unknown-field与物理拓扑语义，避免再造三份version镜像。该子切片完成不代表全分支Rules 2/3完成；其它owner仍按机械账本保持PARTIAL。

随后对 producer -> decoder -> consumer 的 dependency-closed 复查纠正了上述切片的一处错误结论：删除历史 upgrader 与数字 edit counter 是正确的，但把五份 tracked asset 改成 current `PulseDocument` 时越过了仍存活 legacy consumer 的最后消费边界。`readout_duration`、`temperature`/`grey_molasses_detuning`、generic `pulse_scan`、`OptimizeMotFieldTask`、legacy PulseGUI/file-open，以及 legacy timing/PulseController 路径仍把 `probe_template.json`、`release_recapture.json`、`mot_field_template.json` 等送入 `PulseTableState.load()`；因此当前事实是“current `PulseDocument` 被 legacy exact reader 拒绝”，不是“legacy JSON 被 current reader 拒绝”。这不是 fidelity/Otsu、temperature 或 FPGA 物理算法回归：失败发生在 pulse bind、capture 和 reducer 之前；同帧 main readout oracle 与直接 virtual detection-time envelope 仍通过。

该断口在 dependency-closed consumer cut 完成前明确为 legacy island **NO-GO**，不能用旧树测试失败要求恢复旧 JSON、upgrader、dual reader、schema sniffing 或 `PulseDocument <-> PulseTableState` converter。也不能把这些生产 consumer 仅因 target 实现已存在就删除。迁移解冻后必须按 use case 一次切换为 `PulseDocument -> target binding -> current compiler/artifact -> current sequencer/camera endpoint`，同时迁移 camera exposure、scan/API semantics 与 UI editor；每个 use case 同 commit 删除自己的 legacy resolver/loader consumer，最后 consumer 后再删 legacy persistence/command seam。迁移前的验证只把 current asset/current codec/golden 当 current schema gate；旧 `test_measurement_fidelity` 中“legacy selectable template 读取 target asset”的组合不是 current gate，其中短曝光约 chance、长曝光明显更高的 virtual 物理 envelope 则应由不经过 legacy template loader 的独立 oracle继续保留。

SavedFigure NPZ 的 Rule-3 清点以 parent checkpoint `4bb6a28` 的174个领先提交、381个changed files为范围；在排除硬件ABI、已部署wire generation、运行时代际与已完成pulse/readout后，普通跨会话软件edit-counter只剩SavedFigure和TaskConsole两个owner。本子切片沿 `DataFigure.save/_GridData.save -> _write_saved_npz -> _load_saved_npz -> load_figure/live/viewer/neutral facade` 完整调用面核对SavedFigure，并确认tracked NPZ中没有SavedFigure资产需要转换。原格式同时保存plain schema与无真实历史consumer的数字`version=1`；整改删除数字常量、字段和gate，只保留`zou_lab_control.saved_figure`及严格8-key current envelope，旧9-key shape作为extra field直接拒绝，不提供reader、converter或fixture。production净减7行；行为ratchet用独立literal key set与scalar schema检查真实writer输出，原exact-envelope拒绝测试继续锁定未知shape。该owner所在 `data_figure.py` 当前为`1509 PLOC / 1360 nonblank`，对main同文件`1518 / 1369`为`0.99x / 0.99x`，没有超过3倍的海拔债务。TaskConsole的数字version、恒真panel role与旧repeat-mode clamp仍待下一独立切片，因此全局Rule 3仍为PARTIAL。

TaskConsole current-layout 的 Rule-3 清点以 parent checkpoint `5c54184` 的175个领先提交、382个changed files为范围，沿save/load、CLI/name resolver、Workbench reseed、live graph clone和Panel/Logic UI全部consumer核对，并确认tracked与本地`tasks/`均没有layout JSON需要转换。初始发现7处同类问题：无reader的`version=3`、仅为历史round-trip存在的单值`PanelConfig.role`、state/panel/logic三层各自宽松补字段、非法repeat mode被clamp、底层`reduce_repeat`又把任何未知mode落入average。整改让三层记录只接受各自exact current fields与writer实际写出的精确顶层类型，root plain schema由`TaskConsoleState.from_dict`唯一校验，`load`只委托该owner；删除version、role、恒真UI gates和带历史命名的测试文件。两轮对抗复审又关闭两个writer/reader不对称：只校验字段集合会让`params/inputs/values/panels/logic=null`被构造器静默归一化为空值；GUI直接编辑单槽`value = <name>`时只改source、不改input，会让真实writer生成reader随后归一化并拒绝的记录。最终由同一个`_layout_record`入口校验字段集合和顶层容器/标量类型，再由领域构造器及canonical-form比较校验panel位置、输入槽与logic标题；单一`PanelConfig.set_source`同时服务构造、GUI source edit和slot增减，使bare-name表达式与picker绑定在写出前一致，未建立第二套schema owner。plot-kind声明派生唯一可选集合，缺值才取声明的首项默认，显式非法值在同一helper拒绝；显示数学另对其union vocabulary fail-closed，不再静默产生错误平均。production净增64行，来自exact admission、writer-shape拒绝、source/input单一mutation owner和两层不同语义的mode拒绝，同时删除无信息字段/分支；没有建立通用UI-state codec、deep params schema或converter。`task_console.py`当前`10036 PLOC / 9331 nonblank`对main`9461 / 8791`为`1.06x / 1.06x`，`live.py`为`6318 / 5740`对`6312 / 5735`约`1.00x`，合计`1.04x / 1.04x`，无3倍海拔债务。exact-record、null/canonical拒绝、真实source-edit writer round-trip、非法mode数学测试及窄AST ratchet共同机械禁止edit counter、single-role、宽松归一化和silent clamp复活。

历史reference/archive残余的 Rule-2 清点以 parent checkpoint `91fa543` 的176个领先提交、383个changed files为范围，并同时检查当前index、tracked path与仓库根实际目录。按“一个ignore entry、artifact、引用点或test skip/assertion算一处”在本次候选代码/构建/测试路径共找到18处：三个会隐藏`reference/`、`references/`或根zip的ignore entry、README树条目、158行旧frontend局部指令/失败历史artifact、四处Python/FPGA build注释或权威引用、board README、Tcl注释及八组测试例外。当前仓库不存在tracked或本地reference目录/根zip，也没有任何运行时consumer；因此这18处全部物理删除，不保留deprecated说明、negative archive fixture、路径skip或兼容入口。board说明没有误写成“唯一读取位置”，而是准确声明仓库内唯一默认并保留现有`ZLC_PS_XDC`显式外部覆盖。除本账本外的主题文件为18行新增、199行删除，production executable NCLOC变化为0；测试文件只选择引用清理hunk，PulseDocument、40/62通道、qCMOS与calibration等并行行为改动未混入。未来机械约束不是保存一份历史黑名单，而是取消ignore与walk skip：任何再次出现的archive目录或根zip会立即进入Git工作树/正常仓库检查面；本次候选路径的精确路径/文本搜索为0。当前index中的延后手册/教程、`virtual.py`及旧TaskConsole设计材料仍有frontend-AGENTS或v16 reference措辞，必须在各自owner切片物理删除，不能把本子切片写成全仓清零；因此这里只关闭Rule-2的活动代码/构建/审计skip子项，Rule 2整体仍为PARTIAL。

至此，Rule-3 的完整changed-file AST/字符串账本已收敛：所有普通software format/algorithm改稿号及upgrade chain均已删除，跨会话文件只保留plain current format name并对其它shape fail-closed；允许项只剩真实部署双端消费的FPGA layout、迁移期`RuntimeSequenceProgram` wire、`zlc_pulse.PulseTargetABI/v1` hash domain，以及不属于格式版本的内容revision/runtime generation。`ZLC-CANONICAL-1`是获批ABI所依赖的冻结字节原语而非可协商edit counter。Rule 3状态改为COMPLETE；其它规则仍保持各自PARTIAL，迁移继续冻结。

`MonitorTap`/rolling live capability 是终态 GUI consumer 所需能力，不因当前 composition 尚未接线而删除。tap 只拥有 broker backlog：订阅时 `max_bytes` 必须至少覆盖一个 `max_payload_bytes`，不足立即拒绝；tap 与唯一 materializer 都只允许首次 publication 前由 preflight bind，claim 会唤醒更早 blocked 的 raw reader并令其重验 owner；bounded overwrite、`missed`、latest/ordered update 不参与 finite exact retention 或 backpressure。可见历史、cycle reset、head EventRef 与 atomic snapshot 已移到独立 `MonitorDataset`；关闭 materializer 必须关闭 tap、清空保留字节并唤醒 blocked reader。finite exact preview 现已在同一 plan 内接通一个 capacity-one local slot，但它的 aggregate admission 只覆盖这一条 exact+preview 链；真实 Workbench 多 slot/process-wide 接线仍被冻结。

Monitor/rolling 的 Rules 1–7 回顾以 parent checkpoint `b648ba9` 的 182 个领先提交、386 个 changed files为机械范围，完整 owner 闭包覆盖 `runtime/streams.py`、`runtime/dataset.py`、package export、finite capture/processor call sites、data/frontend 的 history-role seam、两个直接测试与本设计账本，并以 main 的 `core/signals.py + core/signal_tensor.py + LiveLive.roll/ScanNode._fill_point` 为独立行为来源。首轮三路只读对抗确认一个 P0：旧 `point0 -> point1 -> point0` 会保留上一 sweep 的 point1 且仍宣称 complete；现测试曾把 `[3,2]` 错误行为钉成契约。继续反例审查又确认九个 P1：producer join key 被误当 panel history slot；lifetime missed 与 current-window gap 混为一字段；rolling builder 不释放 tap；旧 progress ref/head 与后来 snapshot 可错配；claim 前已 blocked 的 raw reader 可在 owner 接管后偷走首帧；`DatasetCellKeyContract.snapshot()` 返回 caller 原对象；hashable mutable adapter 可让投影变化而 digest 不变；metadata contract 的 `validate()` 被漏掉；公开构造器的第二份 `cycle_cells` 可推翻 edge schedule，且 append 借用 SCAN_POINT 会把物理坐标静默重标为历史。

整改删除双模式 `DatasetMode`、不可达 duplicate exception、schedule-free rolling 假 digest、exact-only rolling state、单-cell delta staging、死的 MonitorUpdate tap 引用，以及约 150 行防 `object.__setattr__` 修改 frozen contract graph 的未观测 copy machinery；keyed monitor 仍有意复用 edge 的真实完整 cycle schedule/digests，append edge 才明确无 schedule/digest。exact DatasetBuilder 只管 reservation/schedule/ack/EOS/seal，MonitorTap 只管有界 backlog，MonitorDataset 只管从 edge 唯一派生的 keyed cycle 或 sequence-owned append window。claim 与 first publication 在 stream→tap 同锁序上线性化并唤醒旧 waiter 重验 owner；join-key snapshot defensive-copy；adapter/contract 在 bind 时复用现有一次性 intrinsically-immutable 检查，metadata snapshot 在 commit 前由 owner validate。keyed cycle 新 sweep/任意 offset gap 原子清 values、validity、metadata、EventRef；append window 只接受 MONITOR_HISTORY，newest-first且 gap 滚出后 current coverage 恢复，lifetime missed 仍保留。`(R,P,*data_shape)` 与 ComponentValidity 全轴不变。

本回顾 Rule-6 使用 physical/nonblank/AST 计数：整改前 stream+dataset 为 `3982 / 3553 / 57 classes / 21 dataclasses / 3 enums`，当前为 `4061 / 3603 / 58 / 24 / 2`；相对 main `1672 / 1438 / 15 / 8` 为 `2.43x / 2.51x` 行数、`3.87x` classes。净增 79 physical/50 nonblank 是把原先错误且无时间顺序的 rolling branch替换为 keyed-cycle 与 append 两个真实 GUI consumer语义，并加入可证明的 ownership/immutability 边界；data role export、四种 frontend ViewContract policy 与 processor snapshot-only seam 合计只净增 7 行且不增加 class/enum。运行时净增的 1 个 class 对应 live/formal coverage 类型防火墙；formal DatasetCoverage/codec 的旧 missed 字段、死 MonitorUpdate tap 引用和 single-member mode enum均已物理删除，MonitorUpdate 改为 slotted dataclass，atomic snapshot另有一个值对象。直接测试从 `2042 / 1759 / 20 classes / 15 dataclasses` 变为 `2194 / 1879 / 15 / 9`；新增七条互不等价的 nonzero-gap、blocked-reader handoff、late-bind、join-key alias、metadata-admission、second-cycle-source 与 history-role oracle，并把既有 mutable-adapter/list 例子加强为正常 hashable mutable alias；synthetic reflection fixtures 的 class/dataclass 数反而分别减少 5/6。修后真实 `2304×2304 uint16` public path 以 1 次 warmup + 5 次中位数重测：emit `2.708 ms`、ingest `1.239 ms`、owned materialize `3.272 ms`，materialize `tracemalloc` 峰值 `10.128 MiB`；scalar capacity 300 的 10,000 次 payload-build+emit+ingest 为 `35.8 µs/event`，最终 `[9999,9998,9997]`。74个直接 oracle、18个 capture/processor/occupancy/frontend seam oracle、17个 capture artifact current-codec oracle与22个 import-DAG ratchet通过。该 owner切片当时局部GO；后续W1已接通finite exact single-panel E2E和board coalescing，但rolling/multipanel与process-wide aggregate admission仍在S5，不能倒写这个历史checkpoint为完整Workbench完成。

finite exact live-preview 子切片随后把上述 capability 接到一个真实 component consumer，而没有恢复 Hub/latest-value 或第二份 window：exact builder consume/ack 始终先于 preview ingest，`bind()` 后 slot 是 MonitorDataset 唯一 lifetime owner，失败保留 first cause并撤销/清空旧 front，只有无primary/error/UNSAFE的aggregate cleanup才保留 final snapshot 到 panel close。终审反例又关闭五个生命周期/预算洞：跨capture preview attach在compiler边界拒绝；bind前的exact allocation失败通知slot；dirty follow-up在旧worker callback闭包和snapshot引用释放后才提交，`active -> inactive/restart`也只由结束本次render的wrapper在同一gate/release线性化点决定；底层presenter clear失败时组合controller可重复close；Board/live/Qt fault统一保存frontend render owner产生的无traceback string-only异常并在成功close释放。capacity-one `2304×2304 uint16` materialize 的 expected owned snapshot 为 `10,616,833 bytes`，本轮 `tracemalloc` 峰值 `10,620,481 bytes`（`1.00034x`），证明 canonical order 不再产生 advanced-index 整帧 scratch。4-worker executor配合故意放慢raster的反例中，同panel最大并发raster仍为1，最终board与slot均到revision 3；额外可控调度+weakref反例证明freeze与render wrapper不会消费彼此的cycle状态，render callback return前到达的新update只置dirty，dirty follow-up开始前旧snapshot callback已释放；injected clear第一次失败、第二次close成功且第三次幂等；injected render fault经GC后不再持有失败栈的大对象。preview listener失败后 exact `(1,3,96,128)` 与FINAL CaptureArtifact仍成功，cleanup error只走preview fail而不发布normal terminal，1-byte总预算则在任何preview bind前失败。Rule-6 用完整目标foundation `workspace.py + frontend/render.py + live.py + image_raster.py + qt_board.py = 1587 physical / 1401 nonblank / 21 classes / 10 dataclasses / 2 enums`，对main最窄真实host `render_loop.py + qt_canvas.py = 574 / 508 / 2 / 0 / 0` 为 `2.765x / 2.758x`，低于约3倍；没有以main 6312行多功能`live.py`虚增分母。零消费者的blocking wait通知、固定色阶参数、重复IMAGE role校验与反射method guard已删除；保留的新增runtime/workbench/Qt classes分别对应跨包preview合同、唯一slot/coalescer consumer和真实可选Qt endpoint。该段是历史foundation checkpoint；后续W1交付production app composition，U0.2f再闭合finite raw IMAGE viewport/display interaction。ROI authority、live save、多panel/多source scheduler与real qCMOS仍继续NO-GO。

**S1a adapter-neutral camera SPI 子切片以 parent checkpoint `8d9f111` 完成，但仅为 virtual/fake software seam READY。** `zlc_neutral_atom.adapter_sdk`现在唯一拥有immutable `CameraFrameRecord`、terminal record、物理`CameraWorkingPoint`与最小`CameraAdapter`合同；`CameraCaptureEndpoint`不再导入或识别`VirtualCamera` concrete type，只把adapter primitive working point一次转换为neutral-owned schema/physical facts。第一次对抗复审发现三个P1并在同一切片关闭：raw adapter不能再把qualification写进working point自证exact，只有可信installation/Q0 composition可独立注入digest；`max_pending_records`只表示arm内adapter-owned record硬上界，并同时约束endpoint burst预算与adapter arm，不再借用GUI recent/history语义；bounded terminal worker与arm-owner的两阶段幂等terminal合同被写入SPI，不能满足跨线程stop/unblock及owner-affine teardown的DCAM adapter必须等待owner I/O lane，结构化Protocol绝不等于real资格。Virtual installation注入与旧实现逐字同义的deterministic trigger-wire evidence，working-point settings tree、schema、physical facts、16-frame capacity以及capability/artifact identity均保持不变。

本切片覆盖两个新`adapter_sdk`文件、endpoint、VirtualCamera、`_bind_camera` composition、一个current whitelist contract test及本设计账本；没有修改RTL/bitstream、旧树、legacy producer/consumer或GUI。白名单定向测试`tests/test_zlc_camera_adapter_sdk.py`为`7 passed`，独立fake证明非Virtual adapter贯通exact command boundary、driver ring覆盖后record bytes仍owned/readonly、batch drain的`produced_count`可重复且ordinal gap失败；未注入qualification的同一adapter在PREPARE前失败。public virtual notebook oracle仍得到只读`(1,1,96,128)` FINAL exact artifact，旧settings digest `ffd1b44fd6090495835fed96714e62df18d035abf158b6e6246a1a039f7eebc7`与qualification digest `0eaed5ef125bf5cf461afca859eca21c8e9a75c2301ba1eff871911dcda761b5`不变；fresh-import探针确认adapter SDK/endpoint不加载virtual module、DCAM、Qt、SciPy。两路独立只读复审均为`P0=0/P1=0/GO`。

Rule-6按精确目标责任闭包计数为`1757 physical / 1639 nonblank / 10 classes / 7 dataclasses / 0 enums / 56 functions`，main完整legacy camera adapter/endpoint class envelope为`1012 / 914 / 3 classes / 70 methods`，代码量比`1.736x / 1.793x`，低于约3倍；class毛比`3.333x`但受影响production文件相对parent只净增`152 physical / 126 nonblank / 2 classes / 1 dataclass / 5 functions`，同时方法数仅main的`0.8x`。三个跨endpoint/adapter消费者的record/working-point值对象和一个7成员Protocol各有独立边界语义；没有single-member enum、manager、plugin、wrapper或零消费者格式机器。该GO不宣称camera owner lane、real AssetMap/Q0/tail evidence或DCAM迁移完成，Formal real qCMOS仍明确NO-GO，S1整体仍未完成。

**W1 finite exact-capture Workbench产品切片以parent `dd5a2ebab904c80cc5b97e851da8c5f6bfafa70a`完成，现为virtual/offscreen、single-event READY。** public lazy入口只在调用时加载Qt；notebook与Workbench共享`PreparedFiniteCapture`，普通multi-event `start/run/inspect`不触发W1资格限制，只有请求preview时才惰性核对唯一singleton READOUT_EVENT。Window使用两个worker承载prepare/start/result/reap与figure/raster，Qt owner以generation+RunId丢弃旧snapshot并只reconcile每个不同状态快照一次；terminal后的next-prepare失败不能改写已提交FINAL，present/render/preview-close失败也不能挡RunHandle reap。parent嵌入参数因没有真实consumer且无法兑现销毁握手而删除；窗口只支持top-level显式异步close且不拥有Experiment。preview永远标为`DISPLAY ONLY / latest rendered raw frame`，Capture FINAL只来自SUCCEEDED+final_committed。

本切片tracked白名单测试`tests/test_w1_capture_workbench.py`为`7 passed in 1.47s`，覆盖public Qt Start、连续两Run、exact `(1,1,96,128)` artifact、普通multi-event `(1,3,96,128)`仍成功而W1在Start前拒绝、render失败不改Capture FINAL、1-byte preview预算、start期间nonblocking close、late-listener update/failure replay与lazy import/authority边界；清单外测试未运行。`py_compile`、`git diff --check`、fresh lazy import与AST forbidden-boundary扫描通过。三路独立只读对抗最终均为`P0=0/P1=0/GO`；过程中找到并关闭stale snapshot跨generation、terminal重复prepare、sticky present阻塞FINAL、next prepare降级FINAL、preview close阻塞reap、普通multi-event被W1构造误禁、非权威raster假称FINAL FRAME以及clear失败丢owner等问题。

Rule-6按本轮12个production文件相对parent机械计数为净`+925 physical / +851 nonblank / +2 classes / +0 dataclasses / +0 enums / +36 functions`；公开Workbench包为`692 / 643`，其中具体window为`678 / 636`。把既有foundation与本产品合并后的完整presentation closure为`2254 / 2019 / 22 classes / 10 dataclasses / 2 enums / 107 functions`，对main已建立的最窄产品等价envelope `1073 / 977`为`2.10x / 2.07x`，低于约3倍。新增两类恰是消除notebook/GUI双编译并封住raw authority的`PreparedFiniteCapture`，以及唯一具体Qt生命周期owner `CaptureWorkbenchWindow`；CaptureRequest/PlanDescriptor只是从facade移到共享owner，dataclass净增为0。零消费者`CapturePreviewSpec.for_capture`、未兑现parent API、重复board clear和两份repository-borrow cleanup已删除；为追逐行数再拆manager/state enum/executor wrapper会增加海拔，因此不做。该GO不包含real qCMOS、continuous/rolling、overlay/ROI、Setting/Edit、save-view、grid/multipanel scheduler或旧camera panel替换。

`zlc_frontend.figure` 全幅 IMAGE 的 Rules 1/4/5/6/7 子切片以 parent checkpoint `8285e43` 的177个领先提交、386个changed files为范围，覆盖共享 ndarray ownership、figure model/contract/evaluator及其两个直接测试文件。共定位7类同源问题：默认400万元素上限会拒绝真实`2304×2304` qCMOS；单个整数 repeat 的显示均值违背 main 当前`reduce_repeat`的 native-dtype快速路径并把`uint16`膨胀为`float64`；evaluator DTO只翻转writeable flag、消费者可重新开启写权限；IMAGE/CURVE validity会把整数、非零小数和NaN静默强转为真；complex IMAGE/CURVE/HISTOGRAM在没有显式real/imag/magnitude/phase transform时仍被宣称可显示；连续选择与`(X,Y)->(Y,X)`方向变换制造不必要整帧副本；旧resource测试只拿错误的`1920×1200`尺寸镜像私有估算函数，既没有调用公开evaluator，也没有约束真实峰值。整改由`zlc_data.immutable_array`唯一拥有bytes-backed不可重新启写的数组冻结；选择器对连续区间返回view、只对真正非连续轴按最窄比例依次gather，运行时与admission共用同一个AxisId顺序；IMAGE的Y/X仅由声明轴决定，`(R,P,component,Y,X)`中的component只能显式facet/select/reduce，绝不靠rank、singleton或`reshape(...)[0]`猜掉；只有声明为REPEAT且实际最多一个contributor的整数/布尔显示降维才保持native dtype，多repeat MEAN仍为canonical float并保留小数，complex非METER在contract边界fail-closed。`max_materialized_nbytes`改为描述真实语义的current-only `max_live_nbytes`，没有历史alias；每层先检查“此前retained DTO + 本层峰值”再分配，输入OwnedSnapshot仍明确属于调用方而不重复计入。

真实公开路径 profile 取完整`2304×2304` qCMOS而非缩小fixture：单repeat `uint16`约19 ms，预算25.875 MiB、`tracemalloc`峰值25.520 MiB并保持正确Y/X方向；低于所需预算时在约0.006 MiB trace peak即拒绝。rolling P=16正确跳过末尾invalid并选point 14；`R=16, P=1, 1024²`从错误拒绝收敛为250.254 MiB预算对216.011 MiB实测、由默认256 MiB接受；`R=2, 2304²`为344.813 MiB预算对278.449 MiB实测，因真实峰值也超过默认限制而正确拒绝。8种bool/int/float dtype乘5种repeat数量、ComponentValidity、四层retained及稀疏双轴选择矩阵均未出现欠算；最窄预算余量约1.022倍。focused测试52项通过，独立扩大集合78项通过、1项按既有条件跳过，另有32组公开IMAGE oracle与4组validity子轴oracle交叉验证。测试不复制production公式，而是直接比较手工AxisId索引/方向/validity预期；原私有估算镜像已删除。

本子切片的Rule-6计数以物理行、nonblank、AST class/dataclass/enum/function为准：四个production owner从`2254 / 2001 / 46 / 36 / 5 / 71`变为`2647 / 2365 / 46 / 36 / 5 / 81`，即总行数`1.17x / 1.18x`；核心`evaluate.py`从`1061 / 976`变为`1416 / 1305`，即`1.33x / 1.34x`。main没有同职责的renderer-free target evaluator，故只报告绝对值和同一parent倍率，不拿legacy GUI renderer伪造分母。没有新增class、dataclass、enum、single-member policy或一层wrapper；新增函数是selection plan、dtype/reduction和live-byte预算的共享执行owner。两个非阻塞P2被如实保留：极大coordinate-range仍会在reserve前构造Python索引list，极深fixed-index rolling history的上界可能保守误拒；二者都不会超预算放行或静默错数，待出现真实规模consumer时以同一公开profile驱动压缩。本owner切片经独立correctness与profiling双GO，但它不改变Rules 1/4/5/6/7的全局PARTIAL，也不解除迁移NO-GO。

Figure current codec 的 Rules 1/2/3/5/6 子切片以 parent checkpoint `b8682ed` 的178个领先提交、386个changed files为范围，沿 `ViewSpec/FigureDocument -> primitive tree -> zlc_storage canonical bytes` 及嵌入的 zlc_data Selection owner逐层核对。格式只保留 plain current schema `zlc_frontend.ViewSpec` 与 `zlc_frontend.FigureDocument`；`FigureDocument.revision`是内容revision，不是软件格式改稿号；不存在version、upgrade、legacy reader、alias或compatibility fallback。发现的两类问题是：codec 自建 `_exact`，7个调用点重复`zlc_storage.canonical.exact_mapping`，并在领域值构造器前用13次text和2次integer预校验重复同一leaf invariant；唯一codec测试虽名为owner-delegating，实际 ViewSpec/Document 均没有Selection，只做同一writer/reader round-trip和一个extra-field拒绝，writer/reader同步改字段或绕过Selection owner仍会全绿。整改令codec只拥有wire structure、tag/list dispatch及decode后的typed canonical re-encode检查；map shape/discriminator委托storage owner，AxisId/DatasetId、enum、digest/text/revision/index原值直接交给各自领域构造器，Selection tree双向委托zlc_data owner。re-encode检查保留，因为它拒绝binding/dataset等集合经领域排序规范化前的非canonical typed representation，不是leaf重复验证。

Rule-5 ratchet 改为手写rich primitive-tree oracle：固定两种frontend schema及完整字段，覆盖X/REDUCED/SLIDER/SELECTED、MEAN、FixedIndex、LatestNonempty、display Selection、FigureSelection、两dataset排序和layer顺序；Selection非空路径对owner函数双向各观测4次。测试另独立扰动顶层/嵌套extra field、两种unknown schema、selector tag、binding顺序和dataset顺序，不能由production常量或字段集合生成expected；canonical framed bytes不在frontend重复冻结，因为`zlc_storage.canonical`已有独立literal byte golden。独立审查的21种畸形payload全部fail-closed。`codec.py`从`273 PLOC / 239 nonblank / 13 functions`变为`276 / 244 / 12`，`figure/__init__.py`保持`93 / 90 / 0`，没有class、enum、dataclass或wrapper增长；main没有同职责renderer-free codec，旧GUI persistence block只作方向参考而不冒充等价分母。当前四个bytes入口仍只有测试和两层public re-export，但它们有明确Workbench/notebook终态consumer，不能按当前reachability删除。本codec owner可局部判GO；全局Rules 1/5/6仍为PARTIAL，迁移门禁不变。

Figure auto-view 的 Rules 1/4/5/6/7 子切片以 parent checkpoint `17c06c2` 的179个领先提交、386个changed files为范围，覆盖 `ViewContract/ViewSpec -> suggest_view -> FigureEvaluator`、Selection、PointLayout 与唯一直接测试。首轮独立审查找到4类P1：按dataset tuple顺序贪心会让同一AxisId集合因排列不同而误拒，并漏算已绑定repeat BATCH；range Selection被擅自解释成spatial mean；所有SLIDER被误写成LatestNonempty，使site/component/spectral按“最高valid index”漂移且singleton repeat浪费动态名额；显式scalar Selection发生在display-axis选择之后，不能消除同role歧义。后续对抗又关闭5类同源边界：自动FixedIndex逐point axis取首项会在EXPLICIT PointLayout拼出不存在的tuple；rolling monitor允许同一cell address覆盖，`0 -> 1 -> 0`证明最大nonempty repeat/point index都不是最后发布事件；FixedIndex没有resolution record会让renderer无法显示实际coordinate；EXPLICIT非连续allowed tuple逐项membership为O(P²)；singleton fixed cell axis被memory admission误算成整行gather并拒绝本可安全显示的稀疏图。

整改后的唯一语义是：Selection只裁剪allowed indices，不生成reducer；display-axis物理歧义仍需输入，其余不丢轴布局由contract role priority + AxisId tie-break的有界`(batch_product, facet_product)`规划器求解，repeat BATCH从既有binding计入。自动SLIDER仅用带标签FixedIndex；全部point-axis fixed值共同取自一条真实PointLayout storage row，validator对saved/manual ViewSpec再次拒绝EXPLICIT hole。LatestNonempty只允许repeat且明文表示“最大允许非空逻辑repeat index、非发布时间”；rolling current必须由controller根据EventRef/progress生成覆盖repeat和全部point axes的完整显式Selection。evaluator为FixedIndex与LatestNonempty都生成AxisResolution；singleton physical row不再被预算成gather。非连续point selection仅在helper入口转frozenset一次，100k-row EXPLICIT公开suggest约0.05–0.08秒，不再二次增长。

Rule-5采用独立容量/排列与物理layout oracle：6,859个三信息轴容量组合和961个repeat-BATCH组合为0误接收、0误拒绝、0无效spec；独立复审扩为41,154个排列与5,000个随机RECT_C/RECT_F/EXPLICIT四intent schema，0 permutation drift、0轴遗漏、0非repeat自动reduce/Latest、0非物理point tuple。相关storage/data/figure集合为82 passed、1个既有条件skip；figure自身32 passed。Rule-6按物理行/nonblank/AST计数：四个production owner从`2987 / 2691 / 38 functions / 46 classes`变为`3128 / 2829 / 38 / 46`，即`1.047x / 1.051x`；没有新增class、dataclass、enum、function、single-member wrapper或通用solver。直接测试从`1146 / 1053`变为`1477 / 1367`，增长来自独立数学oracle、sparse hole、rolling语义与resolution行为，不复制production DP。main没有同职责的renderer-free typed planner，故不拿legacy `coerce_panel_value`的shape猜测伪装等价分母。本owner经两路独立对抗审查判GO；两个非阻塞P2如实保留：S2 rolling controller接线时仍须增加真实wrap→完整cell Selection集成测试，最终validator异常仍统一显示`CONTRACT_REJECTED`而诊断较粗。全局Rules 1/4/5/6/7仍为PARTIAL，迁移继续冻结。

Content store 的 raw manifest `(namespace, digest)` 在每个公开 read/has/durability API 入口各验证一次，随后 path builder 和同一次 verified read 信任已规范化字符串；`has_manifest` 直接使用同一个 path 做内容校验，不再回调公开 reader 重验。`ContentRef` 在构造/root decode 后是 typed immutable blob identity，`read_blob` 信任其 digest/size，不重新跑文本 validator，但仍从同一 open handle 执行实际 size 与 SHA-256 内容核验。Store wrapper 获取 process-local authority 不作一次空的预检再由 operation 重检；真正 filesystem operation 仍恰好执行一次 authority/store/root/path/lock identity 检查。路径逃逸、内容损坏、size budget、fsync 与 manifest 可见性边界全部保留。

Pulse deployment 的静态 target/geometry/ABI 拓扑由 `DeployedStreamerSession` 构造边界验证并绑定一次；随后每次 `prepare` 只验证新 artifact 的 clock、resident capacity、IR-to-target 关系与 deterministic wire packing，不再次扫描不变拓扑。独立使用的 public `validate_artifact_for_deployment` 仍先组合静态 target admission 再调用同一个 bound-artifact validator；`build_service_for_session` 面向 generic/fake session 的 composition boundary 仍需验证 target、session geometry 与 clock，而 `build_server_runtime` 不在这些 owner 外预演第三次相同校验。该收敛只减少 host 端重复工作，不改变 prepare-before-I/O、冻结 bitstream、wire bytes、hardware layout readback 或任何 pulse consumer。

Runtime Rule-6 海拔清点在 checkpoint `675c62f` 由 `git show HEAD:path/main:path`、Python tokenize 与 AST 独立计数：stream、dataset、capture、processor、Workbench camera/sequencer 直接切片合计 current `9836 PLOC / 8926 NCLOC / 98 classes / 49 dataclasses`，main 非重叠同职责等价物为 `2104 / 1699 / 16 / 5`，即 `4.67x / 5.25x`；把 installation/composition authority 纳入双方包络后为 current `11356 / 10254` 对 main `3176 / 2608`，即 `3.58x / 3.93x`。高倍率不能直接推出删除：main 不具备 exact reservation/cursor、publication identity、processor lineage、physical terminal/safety，且 MonitorTap/rolling/live 有明确 scheduled GUI/notebook consumer。近期压缩只允许 dependency-closed 地收敛重复 owner/wrapper；不以 current reachability 砍能力。已识别的安全候选是 stream terminal/live-binding helper 去重；projection/capture/processor wrapper、completion mirror、endpoint state machine 与 triggered-camera binding 需先有 Rule-4 digest/authority/lifecycle golden，再决定是否压缩。

9 MiB frame 的针对性 profile 将必要成本与重复防御分开：构造 immutable `Value` bytes 约 `1.957 ms`，一次 payload content digest 约 `2.412 ms`，两者必须保留；generic dataclass materialization walk 约 `0.542 ms/次`，而把已构造的精确 `Value` 视为 owner-validated leaf 后约 `0.00336 ms`。因此 stream publication 仍只生成一个 snapshot、一个 digest、一个 byte count和一个 `_Stored(Envelope, payload_bytes)`供 exact/MonitorTap 共用；`Envelope.__post_init__` 是拒绝 `DataBlock`（包括嵌套）的唯一 generic admission owner，`_emit` 不预扫第二次。Dataset 的一次性 immutable-snapshot 检查同时用于 adapter/contract bind 与 metadata snapshot，信任精确 `Value` 构造不变量并拒绝正常 hashable mutable alias；`FrozenDatasetEdge` 只缓存 composition-owned frozen adapter/contract/schema/operator 与 defensive-copied exact address schedule，不为防显式 `object.__setattr__` 复制或反复重读整张 graph。实际 stream owner/fingerprint/key-domain绑定、reservation/generation/cursor、锁内 Delivery TOCTOU、metadata semantic validate/digest与 occupancy 原子关系全部保留；不引入 `PreparedPayload`、tuple-returning contract或第二套 unchecked API。

Stream state machine 的代码去重只能围绕完全相同的事实：readiness source 的“已注册、同一 consumer、ACTIVE、source 未终止”由一个 lock-held helper供 bind/source/emitter 三处复用；exact delivery/completion/abort共同的“已注册且 consumer identity 相同”由 stream 内一个 helper拥有，各操作自己的 allowed-state/EOS/ack/commit语义仍留在调用点。generation terminal 的“关闭 admission、把同一 terminal error/EOS通知所有 MonitorTap、唤醒等待者”也只有一个 helper，finish/fail/supersede/non-backpressure overrun只负责各自独有的终态构造与 reservation 处置。不得把这些 helper扩成 generic state-machine framework，也不得把 exact acknowledgement/backpressure与 bounded monitor overwrite合并。

Runtime ordered-digest 的 Rules 1/5 清点以 parent checkpoint `9c12d31` 的172个领先提交、381个changed files为范围，追到 `runtime/dataset.py`、`runtime/streams.py`、`runtime/capture.py`、Workbench camera endpoint、CaptureArtifact replay及三个直接测试文件。共发现6处同类问题：两个测试分别用production同款SHA公式自算expected或让同类实例互证，`OrderedDatasetEventHasher`完全没有独立oracle，capture fake、Workbench endpoint与artifact replay各自又复制metadata-order公式。整改让三处consumer全部委托`OrderedDatasetMetadataHasher`唯一owner，production净减1行且digest bytes不变；三个ordered hash各使用一条冻结literal golden，并保留order/content/identity perturbation与continuity/coverage/seal语义，测试代码净增54行以换取独立canonical oracle。三种hasher不可因名字相似合并：`OrderedEventSpanHasher`绑定纯EventRef span并由capture/processor消费，`OrderedDatasetEventHasher`绑定materialized payload EventRef+metadata identity，`OrderedDatasetMetadataHasher`绑定camera metadata contract与有序metadata并由capture/builder/occupancy交叉核验。冻结golden是行为ratchet，不再为它增加source-introspection lint或通用digest framework；全分支其它mirror/source-introspection/roundtrip-only测试仍待逐文件清点，因此全局Rules 1/5保持PARTIAL。

Canonical primitive 的 Rule-5 清点以 parent checkpoint `3b08f23` 的173个领先提交、381个changed files为范围，逐项核对 `zlc_storage.canonical`、其直接测试、跨包调用与已有固定digest向量。共发现1个同类缺口：原25个通过、1个跳过的局部测试只有round-trip、等价性、拒绝路径和structure admission，encoder、decoder、frame/tag若同步漂移仍可全部通过；Pulse FPGA向量只间接覆盖list/int，不能代替primitive owner oracle。整改不改production，也不在data/frontend/neutral/pulse复制golden，只在storage owner测试增加一条手写280-byte literal及固定SHA-256，覆盖排序map、list/scalar、Unicode、bytes/base64、float edge、ndarray shape/endian与NaN canonicalization；测试净增33行。该literal与既有property/invariant测试共同构成机械ratchet：前者锁定字节身份，后者锁定等价类和拒绝语义。`canonical.py` inspection/materialization两遍grammar以及领域`from_tree`与dataclass可能重复校验属于另一个Rule-1切片，不能为缩小本提交而宣称已闭合；全局Rule 5仍为PARTIAL。

Pipeline memory admission有真实故障依据：历史2.3MP live路径曾因多份float64大帧和失控ring出现`unable to allocate 18.xxM`及GiB级保留，因此arm前的完整拓扑预算、固定/per-event保守headroom和chain-bound admission继续保留。追溯审查同时确认原`memory_profile_fingerprint`只是Python微版本/实现名/pointer width与固定常量的hash，既不参与admission/replay，也没有GUI、notebook、artifact或计划processor consumer；它及CaptureArtifact/OccupancyPipelineResult中的镜像字段删除，只保留能解释拒绝结果的`aggregate_peak_bytes`。相关runtime/artifact/readout生产代码净减84行，140个capture/occupancy/artifact边界测试、11个notebook facade测试和19个架构ratchet通过。这是一项实现负担收敛，不删除memory guard、MonitorTap、rolling dataset或任何计划能力。

StreamProcessor config admission 的 Rules 1/2/5/6 追溯清点以 parent checkpoint `41d199a` 的180个领先提交、386个changed files为范围，沿 `BoundStreamProcessor -> ExactStreamProcessorWorker -> occupancy bind`、package export和唯一直接测试完整搜索。共发现两组同源残余。第一组把 trusted composition config 当 hostile wire payload，另建12个任意depth/node/array/rank/text/integer/expanded/canonical-byte上限、单用途`_ProcessorBindingBudget`和第三遍递归估算；其测试只通过monkeypatch阈值、指数alias与5000-bit整数制造未发生的攻击。第二组是无任何production构造者、计划consumer或物理issuer的`ProcessorExecutionGuard` ABC、definition/bound字段、fingerprint占位、worker callback和三组synthetic fake测试；真实occupancy从未使用它，capture的物理执行authority已由CaptureSession/`CaptureProcessorInputBinding`拥有。两组均按Rule 2物理删除，不保留`None`字段、旧digest占位、deprecated export或兼容入口。

整改后`_snapshot_binding_value`是config admission与owner-copy的唯一递归owner，`_tree`只投影已准入值，不再拥有第二套验证。两路独立复审把具体反例合并为13个不重叠根因族并全部关闭：（1）preserve-before-admit绕过；（2）`init=False`、reconstructed init及最终字段复核；（3）含Enum的active cycle；（4）dtype metadata与native/explicit endian归一；（5）exact mapping key、canonical order及caller detach；（6）ndarray exact/empty strides、独立header与终端exact immutable bytes；（7）alias/base/address/frozenset order的非语义合同；（8）leaf/compound memo保留constructor-derived alias；（9）memo强持有source/result，关闭真实occupancy中临时mask id复用把`usable_sites`误改为全True的P1；（10）stable module-owned dataclass/Enum type tag；（11）stable exact module-resolved operator；（12）plain exact-frozen declared-fields-only dataclass storage；（13）exact reconstruction result type。Enum singleton/class与软件函数热改明确归入固定软件进程假设，不为其再造proxy、逐次rehash或alias graph codec。

行为ratchet保留一条手写literal binding fingerprint，并覆盖preserved `ValueSchema` identity、caller mapping detach/order、cycle、mutable/plain dataclass、reconstructed init/final field、constructor-derived mapping/ndarray alias、dtype normalization/metadata、array header detach、reversible readonly、singleton/empty stride和nonfinite float；最关键的memo修复由真实admitted-calibration occupancy全套oracle交叉验证。focused验证为stream worker `61 passed`、完整occupancy consumer `9 passed`、capture processor authority接缝`6 passed`和import/format architecture ratchet `22 passed`。三份production文件从`1633 PLOC / 1520 nonblank / 53 functions / 8 classes / 3 dataclasses`降到`1407 / 1308 / 46 / 6 / 2`，净减226 PLOC、212 nonblank、7 functions、2 classes与1 dataclass；直接测试从`1930 / 1680 / 98 functions / 7 classes / 4 dataclasses`变为`1959 / 1690 / 100 / 16 / 12`，净增29 PLOC/10 nonblank/2 functions/9 fixture classes/8 fixture dataclasses。测试增量只承担上述独立反例和真实边界oracle；重复bytearray/tuple/hidden-backing等同规则attack fixture已在提交前压缩，不把每个Python技巧镜像成一份测试。main没有同职责exact stream processor，故只报告parent绝对收敛，不拿legacy synchronous occupancy operator伪造倍率。本子机制可局部判GO，但完整acquisition/runtime owner及全局Rules 1/2/5/6/7仍为PARTIAL，迁移门禁不变。

Fit / capture-fit 的 Rule-6 回顾以 parent checkpoint `f0f13d2` 加当前候选工作树为范围，并把 main 的 `core/fitting.py + operations/processors/fit.py` 当作算法/物理等价物，而不是拿已经迁走的 reference fixture 自证。main 为 `1229 PLOC / 1068 nonblank / 5 classes / 4 dataclasses`；当前完整切片（fit contract、problem、solver、current codec、neutral capture-fit ref/repository）为 `3836 / 3451 / 22 / 11`，即 `3.12x / 3.23x`。其中 main 没有等价物的 durable CAS persistence 占据 result codec/ref/repository，且旧 GUI 与已排期 Workbench/AnalysisPreset 已证明 public immutable model catalog、绑定后 parameter units 与 FitSpec owner codec 是真实接缝；可直接比较的数值与契约 core 为 `3084 / 2793 / 17 / 10`，即 `2.51x / 2.62x`。完整倍率略过约 3 倍的另一部分是一次运行攻击证明确有消费者的安全边界：若把 FitExecution/Admitted proof 压成 dataclass/NamedTuple，标准 `dataclasses.replace` 可携带真实 repository token、替换任意结构合法结果并成功 save/load；因此保留 final/slotted/process-local capability、repository immutability 与 root/store authority integrity check。其余没有用“新能力”给抽象免责：已删除 generic `FitResultArtifactRef`、独立 `FitCellProblem`、acceptance policy、sampling-quantum/alias gate、per-cell timeout/status、solver/initializer mirror identity、public result-tree/packing/evaluator umbrella；只保留一个 immutable catalog view、current FitSpec owner codec 与 result bytes codec。保留的新增复杂度必须分别服务多轴 `(R,P,*data_shape)` batch/grid fit、绝对坐标语义、确定性有界采样、数值诊断、内容寻址持久化、capture source admission、已排期 editor/preset consumer 或上述 raw-result authority 防洗白边界。`BoundFit.run()` 是唯一 public execution 入口，domain 的科学 acceptance 仍由 processor/measurement owner 决定；具体 `CaptureFitResultArtifactRef` 归 neutral，而 payload 数学与 current result bytes 归 data。

该切片的针对性 qCMOS profile 使用 `2304×2304`、仅一个坏点的二维 cell：canonical packing 为 `0.339 s`、`tracemalloc` peak `6.182 MiB`，从 `5,308,416` 个 present 中保留 `5,308,415` valid，并按预算使用 `12,000` 个 observation；实现不建立 N-sized rank/value 副本。load admission 只读取有 `64 KiB` manifest / `64 MiB` result-blob 上限的 fit artifact，并以 capture 的 generation ref + schema 复算 batch/source contract，不 materialize frame chunks；valid/used counts 是数据事实，不能在 schema-only load 阶段伪验证。独立 oracle 一条直接用 `inv(J.T @ J) * RSS/(n-p)` 校验 fixed/free 参数 covariance 嵌入，另一条用手工构造的 result 锁定 current bytes 长度 `2770` 与 SHA-256 `c443df1a06cf185db803da50ce2a5b2f68a4e9b0dd4bdd51b35ee08fa4b14186`，不让 decoder 自证 encoder；另以 mixed cell/data batch 的 RECT_F + sparse EXPLICIT source 值逐格交叉验证 layout/coordinates/present counts，并锁定“仅 generation 不同的 capture ref substitution 必须拒绝”与“load 不得读取任何 frame-chunk blob”。public catalog/FitSpec/validator 均由 package-root 测试消费，窄 AST ratchet 禁止 production 绕进 fit implementation submodule；覆盖 transform、fit、convenience、capture-fit 与 notebook facade 的 focused 结果为 `122 passed`。剩余已知 P2 是非 dense point row-group 的 `np.unique` 仍为 O(N)；它不在 qCMOS dense-image 热路径，须在真实 profile 证明成为瓶颈后再优化，不能预建第二套索引机器。本纵切可以局部判 GO，但完整 Rules 1–7 仍未结束，迁移继续冻结；legacy fit/processor 只能在最后 GUI/reactive consumer 迁走后删除。

Storage durability / repository lifetime 的 Rules 1/2/5/6 复核以 parent checkpoint `95997cdd5b1d`（184 个领先提交、386 个 changed files）为范围，沿 `durable_mkdir -> lock-file -> FramedJournal/RepositoryRootLease/PersistentSafetyJournal -> ResourceArbiter -> Capture/Calibration commit -> notebook composition` 追到所有生产 caller。复核确认的根因不是“少几个 fsync”，而是 visibility、durability acknowledgement 与 authority lifetime 曾被混为一件事：递归 `durable_mkdir` 在 child 已可见而 parent flush 失败后，重试会跳过该 entry；三份平台 advisory-lock 实现对错误分类、锁文件持久化和 close 语义发生漂移；repository/safety 的并发 close、safety close-vs-ResourceArbiter bind、unlock-error 后外层未关闭、并发 arbiter shutdown 早返回及 POSIX fork 继承 `flock` 都能让返回值早于真实 ownership 变化；Capture/Calibration 又在取得 repository hold 前开始 CAS staging；默认 safety path 在干净机器上依赖隐式多层建目录。以上都已用确定性交叉或故障注入复现后整改，不以测试偶然通过代替原因证明。

整改后的单一 owner 边界是：`durable_mkdir` 只创建/重新确认“已存在 parent 下的一个 child”，每次都按 child -> parent flush，composition root 显式逐级建立 workspace 与默认 installation safety 目录；`zlc_storage.file_lock` 是唯一平台锁机械 owner，只负责持久准备永久 lock inode、blocking/nonblocking acquire、release 与真实 contention 分类，不拥有 repository borrow、safety token 或 journal transaction。CAS base hierarchy显式逐级建立；动态 shard/namespace 仅在已有 store RLock 下、两次目录 flush 都成功后进入 store-local cache，新 store 首次使用必须重新确认。Repository owner close 一直持有 state lock 到 OS unlock/close与进程 registry释放完成；borrow close 自身线性化，creator-PID guard保证 fork child只关闭继承 fd而绝不显式 `LOCK_UN`。PersistentSafetyJournal 用一个 lifecycle RLock 原子覆盖 bind/close/authority-close/snapshot/append，outer closed 在 unlock失败时也在 `finally` 成立；ResourceArbiter shutdown只在自己的RLock内把OPEN原子推进到CLOSING并快照authority token，随后释放该RLock再执行可能阻塞的journal owner unlock/close，并发shutdown等待同一个completion event，最后重新持锁发布CLOSED或sticky FAILED。Capture/Calibration 的临时 hold 在第一笔 CAS write 前取得，并与 coordinator `prepare()` 同步产生的长期 hold重叠；capture-fit save 已由现有 lifecycle RLock完整覆盖，删除冗余 borrow。`owner` 标签、`_os_released` 双状态和三份平台分支均物理删除，不留兼容参数、alias或旧路径。

Rule-6 以 physical/nonblank/AST 计数审查而不是按每个局部抽象辩护。五个 storage primary-owner 文件从 parent 的 `1414 / 1221 / 12 classes / 2 dataclasses / 99 functions` 变为 `1490 / 1283 / 13 / 2 / 102`，净增 `76 / 62 / 1 / 0 / 3`；唯一新增 class 是 `FileLockBusy`，三个函数就是共享平台机制，没有 lock framework、repository基类或新状态机。Safety + ResourceArbiter lifecycle seam 从 `2075 / 1830 / 33 / 17 / 123` 变为 `2100 / 1857 / 33 / 17 / 125`；Capture/Calibration/notebook/default-path caller seams 从 `4817 / 4370 / 23 / 7 / 237` 变为 `4842 / 4390 / 23 / 7 / 239`。总 production 净增 `126 PLOC / 109 nonblank`，用于失败重试、真实 close/handoff、fork与首启目录保证；同时 `repository_lease.py` 从291降到277行、`framed_journal.py` 从207降到190行、capture-fit净减3行。main 没有同职责 crash-durable storage/installation authority owner，因此只报告 parent 绝对变化，不拿普通文件保存或旧GUI代码伪造倍率；该切片不改变任何物理/数值算法，Rule 4 为不适用而不是“由新测试证明物理等价”。

Rule-5 的独立证据覆盖：parent-flush失败后同一可见 child必须重新flush；lock file已可见时仍重新fsync file+parent；只有EACCES/EAGAIN/EDEADLK类竞争映射Busy，EIO原样抛出且registry可重开；两个FramedJournal实例并发追加；repository/safety双close、close-vs-bind、shutdown-vs-shutdown、unlock失败、跨进程排它与POSIX fork child不能解锁parent；CAS动态目录失败不进cache、同store只确认一次而新store重新确认；Capture/Calibration在第一笔staging前阻止close，capture-fit则由同一lifecycle lock阻塞close；干净LOCALAPPDATA首启能逐级建立并打开真实PersistentSafetyJournal。最终唯一 focused 集合为 `246 passed, 2 skipped`；两项skip只是当前Windows主机没有`os.fork`，对应POSIX路径另有creator-PID静态复核。`py_compile`、`git diff --check`及22项import-DAG均通过。额外试跑的旧 `test_save_capture_single_source` 仍因 legacy TaskConsole 的 device-bearing `running_nodes` fixture未提供runtime authority而失败；它不经过本 storage owner、未被本提交掩盖或修改，继续归GUI/runtime全局审查账本，因此不能把本纵切GO改写成全局GO。三路独立对抗复审最终均未找到剩余P0/P1；Windows `LK_LOCK`约10秒的CRT有限重试明确记录为fail-closed平台行为，不为假设的无限等待再造线程/轮询器。本纵切 Rules 1/2/5/6/7 可局部判 GO，但完整 changed-file Rules 1–7 仍为 PARTIAL，迁移继续冻结。

Data kernel / transform / fit 的 Rules 1/2/3/5/6/7 闭包以 parent checkpoint `d4a0559410f62ab3f5fc166d095d07b940408760` 为比较基线；全分支范围继续机械取自 `main...HEAD`，本提交前为185个领先提交、388个 changed files，而不是依赖上下文记忆。当前纵切逐文件覆盖16个 production owner（13个 zlc_data 文件、frontend figure 的 contract/evaluator、neutral readout schema seam）、6个直接/architecture测试文件及本设计文档；清点方法同时使用 `git diff`、全仓 symbol/reachability 搜索、AST constructor/codec/owner ratchet、手写 canonical tree/golden、随机 layout/selection/reduction/fit oracle，以及按 logical/physical 尺度分离的 tracemalloc/time profile。共关闭21类 production/设计根因与1组跨 seam stale golden：numeric coordinate 别名/布尔与隐式 origin、伪 REPEAT role、C/F/EXPLICIT/PRODUCT 多重身份、cell-layout 重建及 sparse mapping/reverse-index clone、dtype endian、INVALID 快捷路径漏 admission、big-endian complex NaN、固定宽度 retained-byte 乘法、codec leaf 二次校验、零消费者 bulk/standalone bytes surface、selection 双 owner/range 膨胀、transform preview/history/revision/origin/self-digest/单字段 wrapper、source schema 重建与重复 fingerprint、fit 的 unit/float64/absolute-coordinate边界、logical-size sparse allocation、declared coordinate 按 batch 重扫、BoundFit 双派生与普通子类伪 proof、sparse reduction 的多份 mapping 自证，以及 readout 对 numeric spelling 的重复约束。Axis/DatasetSchema/Selection/CommittedTransform 仍保留外层 artifact 所需的 owner tree projector；删除的是无人消费的 standalone bytes 与历史机器，不是删除目标能力。

整改后的唯一边界是：AxisSpec 规范 numeric identity 并保存 implicit `index_origin`；DatasetSchema 构造并缓存唯一 cell_layout，sparse PRODUCT 直接复用原 PointLayout factor，而 AxisLayout/PointLayout 以结构相等/hash保证 codec round-trip；Value canonical owner 在任何 shortcut 前验证 shape/dtype/validity并先规范 endian；Selection 只由一个 resolver解释；DataTransformSpec 直接保存 Selection/ReductionSpec，CommittedTransform只冻结input/spec/output；TransformedSchema fingerprint按需一次缓存。`BoundFit(FitSpec, DatasetSchema)`内部单次派生effective schema/model并一次准入fit coordinates，packing/solver只接受exact BoundFit并消费缓存source，不再让FitProblem/property/batch循环重复验证。大型Value/DataBlock持久化只走已有bounded binary chunk/CAS。runtime dataset的schedule/key/consumer三枚identity golden因AxisSpec current grammar新增`index_origin`而变化，已由不调用runtime helper的手写schema/domain/consumer tree独立交叉计算，不用同款writer自证。

Rule-6 对上述16个production文件按physical/nonblank/AST计数：parent为 `8753 / 7870 / 66 classes / 50 dataclasses / 7 enums / 311 functions`，当前为 `8375 / 7563 / 60 / 45 / 6 / 282`，净减 `378 physical / 307 nonblank / 6 classes / 5 dataclasses / 1 enum / 29 functions`；其中zlc_data自身净减340 physical、271 nonblank、6 classes、5 dataclasses、1 enum和28 functions。6个直接测试文件净增518 physical、460 nonblank、1 fixture class与21 functions，新增内容是手写typed tree、跨specialization equality/hash、稀疏物理行、proof-subclass反例、坐标/endianness与跨seam固定digest，不复制production算法。main没有同职责的headless named-axis data kernel，故Rule-4对本纵切为不适用；既有fit物理/数值算法与main的倍率及差异理由已在前一fit账本单列，当前只改数据身份、packing与authority边界，不拿legacy GUI shape猜测伪造分母。

针对性证据为：2304×2304 uint16全valid canonical path约`0.00049 s / 0.0025 MiB`且共享source，单坏点约`0.00497 s / 10.127 MiB`且只保留一帧canonical输出；1亿logical/2 physical sparse rows的selection约`0.00169 s / 0.0186 MiB`、fit packing约`0.00167 s / 0.0166 MiB`；20万declared coordinates、40 batch×2 physical rows由bind完整扫描一次，packing降至约`0.0170 s / 0.105 MiB`，不再是原约17秒的`O(batch × logical_size)`；50k sparse identity bind约`0.0002 s / 0.0023 MiB`，首次确实消费TransformedSchema fingerprint时约`1.08 s / 20.1 MiB`，后续读取为微秒/零级分配。最终本体、runtime dataset及fit artifact/repository/calibration直接consumer集合为`391 passed`，py_compile、import-DAG、git diff check均通过；三路独立对抗复审均判当前闭包GO、P0=0、P1=0。两个无真实生产热点证据的P2如实保留而不预建机器：完整EXPLICIT映射识别RECT仍逐physical row canonicalize（100k约0.84秒），sparse partial-axis reduction fallback仍按physical rows线性group（20k output groups约0.45秒）；两者均不按logical extent densify、不静默错数，只有真实profile命中才优化。

紧随该纵切完成的 materializer-delta 闭包，以全仓 symbol/import/constructor 搜索、`git log -S/-G` 引入史、main 对照和 runtime producer→materializer→artifact/storage 调用链为覆盖证据。本闭包开始时，独立 delta 类型只剩定义、package export、stream 负检查与一条自证构造器测试；提交史显示 exact builder 早期曾为每个 cell 构造该中间对象并立即 apply，但 `fd6ff1c` 已用 owner 内 `_write_cell` 取代 staging/applier，之后再没有 producer、apply owner、snapshot store、persistence codec、UI/notebook consumer或已排期 consumer。继续保留它防不住任何已观测失败，反而复制 `DatasetBuilder`/`MonitorDataset` 已经拥有的原子提交语义，因此现已连同镜像测试、导出和所有设计残余物理删除；stream 继续只在 `Envelope` admission 一次拒绝嵌套 `DataBlock`。原子性由 runtime 的独立故障 oracle 证明：fallible metadata/key/admission 在写入和 ack 前完成；exact 成功提交把 values/written/validity/metadata/ordered hash state/counters/revision 与同一 stream 临界区的 ack 绑定；live 从同一锁提交并冻结 metadata/EventRefs/head/coverage/block。

该闭包以 parent checkpoint `7cfcb2f9c14f` 的 186 个领先提交、388 个 changed files 为机械总范围。三个受影响 production 文件按 physical/nonblank/AST 计数由 `2570 / 2312 / 36 classes / 12 dataclasses / 2 enums / 128 functions` 降为 `2492 / 2247 / 35 / 11 / 2 / 127`，净减 `78 / 65 / 1 / 1 / 0 / 1`；三个直接测试文件由 `3849 / 3301 / 27 classes / 21 dataclasses / 1 enum / 208 functions` 降为 `3813 / 3267 / 27 / 21 / 1 / 207`，净减 `36 / 34 / 0 / 0 / 0 / 1`。main 没有同职责的公开增量事务对象，因此不伪造倍率；本闭包不改物理/数值算法，Rule 4 不适用。focused data/stream/materializer/processor/import-DAG 集合为 `189 passed`，另有 2 个 capture artifact/CAS round-trip 与 component-validity chunk oracle 通过；前一纵切遗漏的 BoundStreamProcessor literal 由不调用 owner `_tree` 的手写 primitive tree 重新计算，逐级交叉得到 current schema/payload/key-domain 和最终 fingerprint 后才更新，不照抄 runtime actual。精确 symbol/AST 搜索、fresh import、py_compile 与 diff check 均通过，三路独立对抗复审均为 GO、P0=0、P1=0。完整 Rules 1–7 仍为 PARTIAL，迁移 NO-GO 不变；全部剩余代码闭包结束后还必须做一次全分支独立对抗验收。

通用 readout contract/codec seam 的 Rules 1/2/3/5/6/7 闭包以 parent checkpoint `f23de4c23a696bebb18aabc0cc9eb5f74dda757b` 为基线；开始时机械范围为187个领先提交、387个 `main...HEAD` changed files。逐文件覆盖 `contracts.py`、`codec.py`、`calibration.py`、`analysis.py`、`calibration_repository.py`、Workbench camera caller、四个真实 direct/artifact测试与本设计账本，并以全仓 symbol/import/getattr 搜索、`git log -S/-G` 引入史、outer Capture/Calibration codec调用链、main FrameContract 对照及三路独立只读审查交叉验证。关闭的同根问题包括：十个从未获得生产或排期消费者的 standalone bytes API及其 generic wrapper/error/export；五个没有独立 dispatch/persistence边界的 nested schema tag；codec 对 constructor invariant 的 text/int/float/pair/dtype/list二次准入；nested parser 与外层 durable full re-encode 的重复 canonical projection；只有 descriptor 内部使用却公开的 event-setting tree surface；Descriptor/FrameContract 重复的12个 camera fact mapping；FrameContract 零消费者 digest/fingerprint；只供测试的 layout `from_schema` 与 raw-row diagnostic；公开 repeat-expanded bracket type；单 caller witnessed wrapper；PointLayout 已禁止的 duplicate-row guard；以及 calibration preflight/execute、source admission/final commit 的重复 full join、限额检查前分配、未计费 join/context graph和重复 `group_contexts` 验证。没有 compatibility alias、deprecated 名、旧 tag reader、upgrade工具或旧格式 tombstone test。

整改后的 owner 边界是：五个领域 contract 构造器唯一拥有物理/领域不变量；layout owner 唯一扫描 physical PointLayout 并产生不按 repeat 展开的 private compact join；`_ResolvedCalibrationSource` 把同一次 capture admission 的 binding、FrameContract、physical context 与 join 原子绑定；formal RunPlan 的 private prepared value把同一份source admission、resolution与repository borrows从preflight交给execute，closed result继续持有其source/resolution和内存证明直到finalize，FINAL不建立fresh admission。FrameContract 只验证自己消费的 event axis/index与camera/schema facts；codec 只做 exact field structure、foreign owner delegation与typed construction；Capture/Calibration outer codec唯一拥有 current canonical bytes重建比较和CAS identity。readout codec public面只剩真实外层消费者调用的8个 tree projector/parser。独立 Rule-5 oracle用手写 primitive descriptor（含三条event setting）、FrameContract（含binding及owner-encoded ValueSchema）和layout逐字段双向核对，能抓住 exposure/gain、sensor/ROI/binning、Y/X与camera/sensor/optical等同型字段同步错接；真实 Capture manifest 与 CalibrationArtifact payload又断言嵌值确实委托相应owner，倒序layout在outer admission明确因non-canonical拒绝。C/F/EXPLICIT、event axis任意位置、missing/unbalanced context、无额外point axis、repeat-lazy、preflight拒绝先于join分配以及每个admission只resolve一次均有独立oracle。已有main同帧数值oracle、component-validity、raw capture commit/reload、calibration commit/admit与camera endpoint路径保持不变；本闭包没有修改任何物理/数值算法，Rule 4沿用已完成的main同帧证据而不是由codec round-trip冒充。

Rule-6 机械计数中，primary `contracts.py + codec.py` 从 `1212 / 1071 / 7 classes / 6 dataclasses / 0 enums / 57 functions` 降为 `952 / 853 / 6 / 6 / 0 / 38`，净减 `260 / 218 / 1 / 0 / 0 / 19`；六个 production 文件总范围从 `6299 / 5754 / 37 / 29 / 4 / 219` 变为 `6114 / 5603 / 38 / 31 / 4 / 201`，净变 `-185 / -151 / +1 / +2 / 0 / -18`。新增的两个净 dataclass不是单层包装：compact join 去掉 repeat 倍对象并由analysis/final-admission共同消费，resolved source防止同一admission的四类事实错配，prepared value则服从现有RunPlan生命周期把resolution交给execute；删除任一都会恢复重复扫描或松散tuple错配。四个直接测试文件从 `4047 / 3704 / 4 / 3 / 0 / 139` 变为 `4211 / 3858 / 4 / 3 / 0 / 142`，增加的是手写同型字段oracle、稀疏物理反例、限额先于join、authority-boundary解析次数，以及直接profile inherited lattice实现的细长grid资源反例，不是writer/reader或估算公式的镜像自证。main旧FrameContract class为 `151 / 139` 行，当前class为 `169 / 163`（`1.12x / 1.17x`）；直接FrameContract tree函数另为 `29 / 29`，共享camera-facts mapper另为 `32 / 32`，三者分列而不把codec seam藏进class倍率。main没有同职责的descriptor、per-event setting、named sparse join与outer owner delegation，故不拿main整个604行calibration算法文件伪造aggregate倍率。

同一 profile 固定为90,000 physical point rows、4 repeats、120,000 analysis groups：旧 repeat-expanded bracket实现约 `2.46 s / 64.09 MiB` peak；compact point-context join为 `1.276 s / 11.274 MiB` peak、`7.504 MiB` retained，repeat从1增长只改变惰性group count，不改变point join大小。join-build预算为 `65.918 MiB`，保守高于实测；materialize 120,000个named `group_contexts` 为 `0.513 s / 20.319 MiB`，对应预算 `58.594 MiB`，因此preflight不再先分配后拒绝，也不再把Python对象图算作零。首次终审的resource路由仍抓到一个真实P1：继承自main的robust lattice会为每条grid axis保留全部unordered pair slopes，旧estimator在合法`1×1000` grid上以`1.46 MiB`放行了`23.27 MiB`实测峰值；现按实测`48.8 bytes/pair`的上界`64 bytes/pair`及约`1000 bytes/site`的result graph上界`1024 bytes/site/model`计费，新预算`35,549,616 bytes`（`33.90 MiB`）约为实测`1.46×`，并由直接tracemalloc数值实现的独立oracle固定。最终12个focused readout/capture/calibration/camera/notebook/import-DAG模块为`193 passed`，资源审查另以`69 passed`和四组独立profile复核三阶段预算；production py_compile、删除表面全仓搜索与diff check在同一闭包收尾执行。

`R02_RUNTIME_INSTALLATION` 的第一次结案曾被判无效：证据混入旧session、legacy `DeviceSet/Registry`与历史测试，并由这些旧语义反向驱动新守卫；这违反规则8，也无法回答“删除旧桥后哪些target代码仍有价值”。随后的target-only复核只审`zlc_neutral_atom.runtime` authority/resource/run/capture、`zlc_workbench`新endpoint/port、target notebook facade与artifact单一事实源；旧树diff只进入最后消费者/删除账本，上一轮历史测试数量、legacy replacement竞态和old-session兼容结论不再是闭包证据。该复核现已完成，所以R02的**owner audit status为COMPLETE**；本段保留第一次失败只是为了说明为何不能再用legacy测试驱动target。

R02重新结案要求的四类证据现均已交付：目标production current→main海拔与超过3倍项逐项理由；重复capability/digest/terminal/lifecycle owner机械清点；target endpoint/runtime/artifact静态结构与资源profile；旧树最后消费者/删除切片。为了让`test_session_*`、`test_legacy_*`、旧device contract或旧GUI测试变绿而产生的代码已撤销，不能以“迁移桥暂时需要”为理由恢复。

本次 target-only 复核以 parent checkpoint `f45c17a8e1ba9b936b424219cc482a35d2e36a1b` 为机械边界；该点为 `main..HEAD = 189`、`main...HEAD = 389`，合并唯一 untracked target production 文件后 worktree 集合为 390 个路径。复核没有运行或修改任何测试，也没有打开旧 `DeviceSet/Registry` 兼容面；证据只来自 changed-file/consumer/owner/AST 搜索、109 个 target production module 的 import-DAG parse、34 个 changed module 的 `py_compile`、32 个 changed module 的真实 import、`git diff --check` 与两项有明确内存风险的 profile。旧 `zlc_workbench.legacy_neutral_atom/legacy_runtime/pulse_control` 已物理删除；旧 notebook/session 对它们的引用只登记到 A01 最后消费者账本，不能成为恢复旧 bridge 的理由。

R02继续按§22.1双轴记账：**owner audit为COMPLETE；virtual implementation为READY；real implementation仍NO-GO。** 审查冻结解除后，目标树已交付一个process-lifetime `_VirtualInstallationRuntime`：它唯一实例化并持有`ResourceArbiter`、`DeviceBroker`、`RunController`、private raw graph、typed camera/pulse ports与immutable public catalog；public `Experiment`只保存opaque authority token，不能到达raw graph、BoundDevice或drive-capable Port。runtime close不持lifecycle lock等待Run，先关闭admission并join controller，再broker shutdown、camera→sequencer→trap reverse close，最后释放journal/arbiter；失败停在可重试`CLOSING`，不会谎报CLOSED。真实composition尚缺canonical AssetMap、backend physical-owner proof、target-owned qCMOS adapter和recovery-only startup，因此不能把virtual READY改写成真机authority完成，也不能把旧`LegacyNeutralAtomRuntime`接回来。

严格 R02 现有实现范围为 19 个 production 文件，按 `PLOC / NCLOC / classes / dataclasses / enums / functions` 由 parent 的 `16373 / 14685 / 186 / 99 / 14 / 860` 降为 `15133 / 13529 / 162 / 82 / 11 / 814`，净减 `1240 / 1156 / 24 / 17 / 3 / 46`；Git numstat 为 `+4872/-6112`。分项如下：lifecycle `2338/2098`，commit+journal `1049/924`，resource authority `2985/2628`，capture kernel `4516/4023`，artifact `2639/2456`，timing `1606/1400`。lifecycle 对 main `722/652` 为 `3.24x/3.22x`，增量对应 cancellation/deadline/safety cleanup/terminal authority；capture 对 main narrow `917/814` 为 `4.92x/4.94x`，但对包含 measurement/value/dataset 的 generous envelope `1614/1387` 为 `2.80x/2.90x`；artifact 对 main `imageio.py 375/321` 为 `7.04x/7.65x`，增量是 main 不具备的 chunked CAS、bounded lazy reader、完整 event index、inverse lookup 与 crash recovery；timing 对 main `744/642` 为 `2.16x/2.18x`。commit/journal 与 resource authority 没有 main 同职责实现，不伪造倍率，也不计算 heterogeneous 总倍率。

finite exact 的 event ordinal→cell 事实现在只有一个 `DatasetCellSchedule` owner：它以 immutable sealed u32/u64 bytes 保存完整 permutation，跨 `CompiledCaptureCellPlan -> CameraCaptureBindingRequest -> FrozenDatasetEdge -> DatasetBuilder/CaptureSession -> artifact/occupancy` 共享同一实例；processor 即使改变 `ValueSchema/data_shape` 也只复用 key-domain schedule，完整 schema 仍由各自 `DatasetSchema` 绑定。因此数值形状始终是 `(R, P, *data_shape)`，禁止 flatten、取第零项或丢 trailing axes。旧 100k-cell Python tuple plan 常驻约 12.797 MB、plan+edge 22.398 MB、构造峰 36.075 MB；新 packed payload 400,000 B、traced steady 405,591 B、构造峰 807,448 B，分别约缩小 31.5、55.2、44.7 倍。1M cells 为 4,000,000 B，构造约 1.27 s，`cell_at` 约 0.789 Mcells/s，完整 iteration 约 1.015 Mcells/s；相机/设备调用、stream lock 与 value projection 会先成为真实瓶颈。`CaptureFrameSource` 的 durable load 同一次流式读取建立 metadata/permutation receipt 与 compact inverse，只常驻 4/8-byte inverse、chunk refs 与一个最多 256-event chunk；1M cells 的 inverse 为 4 MiB、构造峰约 8 MiB，而旧 eager Python cell/metadata graph 估计约 447–472 MB。recovery 只补验未读的 frame chunks，不再重读 event index。

installation authority 的本地 live identity 已收敛为唯一 `DeviceBindingStamp(PhysicalDeviceIdentity, binding_instance_id)`；broker 只 mint 一个 UUID，exact-once bind 后没有 rebind/hot-swap/stale-generation 分支。camera/sequencer capability 与 camera provenance 直接携 stamp，阶段 ACK 只回显一个 binding instance；远端 pulse server 的 `server_connection_generation` 是独立 transport 事实，继续保留。durable recovery 在 journal acknowledgement 不确定后保留同一 bundle/evidence/claim，禁止 abort，只允许幂等重试；`CommitRecovery(bool, optional result)` 也已删除，known absent 直接为 `None`、known committed 直接为 `PublishedManifest`、unknown 继续用异常表达。terminal publication 与 resource release 仍在同一 arbiter 临界区，Run wait 仍等待 owner thread 真正退出。这些是有 crash/错 owner/内存爆炸根因的保留机制，不是旧 Registry 的适配层。

审查冻结解除后的首个R02/A01/U01实现切片以parent `84f8f93d9faf978bcb98db4a4d6204e9c18b98bb`为边界；该点机械得到`main..HEAD=191`、`main...HEAD=383`，合并当前tracked与untracked worktree后为411个去重路径。只审本切片target production与本设计文档；并行dirty tests、tutorials、manual/PDF/assets、`fpga/run_server.bat`与`pulses/README.md`没有进入证据或stage。证据仅为逐文件阅读、AST/import/consumer扫描、`py_compile`、fresh isolated import、current-format手写vertical oracle、deterministic main物理交叉和有界queue/board反例；没有运行pytest，也没有修改历史测试或旧`DeviceSet/Registry/session`去迎合中间态。

R02 endpoint迁址不是重写膨胀：五个private target endpoint文件为`2432 physical / 2253 PLOC / 2224 NCLOC / 14 classes / 8 dataclasses / 93 functions`，被删除的五个Workbench等价物为`2434 / 2264 / 2228 / 13 / 8 / 86`，即约`1.00x/1.00x`；增加的一个class与七个函数来自remote+virtual共用finite session owner、target-native remote backend和virtual safe-state readback，不是兼容wrapper。新virtual hardware为`869 / 791 / 791 / 5 / 2 / 49`；main等价的`VirtualTrapArray + _TriggerWiredCamera + VirtualCamera + VirtualSequencer`为`1731 physical / 1638 PLOC / 1252 NCLOC`，即`0.48x PLOC / 0.63x NCLOC`。它只保留当前finite pulse/camera纵切消费的物理模型，恒false的`_pinned/consume_pin`、unused brightness-jitter knob、ROI/free-run/MOT monitor/laser/RF控制等未声明能力没有伪装成已迁移。新的installation+catalog没有main同职责process-lifetime owner，故只报告绝对`743 physical / 659 PLOC / 646 NCLOC`，不拿旧registry/session伪造倍率。remote deadline整改后的current `zlc_pulse.client + server`为`1045 physical / 937 PLOC / 937 NCLOC / 7 classes / 3 dataclasses / 58 functions`，parent为`938 / 841 / 841 / 7 / 3 / 53`，PLOC为`1.11x`；净增五函数与两个简单互斥gate只关闭已经复现的“logical close deadline晚于transport RPC返回”、timeout/GC后disconnect第二次SAFE、failure-recovery并发SAFE，以及“软件SAFE先完成、迟到backend operation随后改变硬件”竞态。没有新增线程、工作流状态机、兼容层或超过3倍项；SAFE gate有generation interrupt/disconnect/failure recovery三个真实入口，backend-operation gate被prepare/FIRE/complete/SAFE四条真实物理路径共同消费，均不是单消费者wrapper。

U01三文件从parent的`657 physical / 571 PLOC / 15 classes / 8 dataclasses / 32 functions`变为`962 / 857 / 17 / 10 / 39`，为`1.46x/1.50x`，没有超过3倍项。新增两个值类分别表达稳定source identity与per-panel presentation identity；没有它们，同组不同dataset source会被错误要求同ref，或document/selection revision会再次混进producer generation。BoardController仍只有一个whole-board pending mailbox和一次性work token，没有预建第二个scheduler、tile graph或render state machine；新增的close失败分支不增加类型/函数，只保证真实GUI presenter清空失败时保留同一front供重试而不恢复publish authority。故障注入证明第一次clear失败保留资源与fault、第二次close成功后presenter可被GC、第三次close幂等。A01既有8-file范围为`1469 PLOC/1270 NCLOC/8 classes/4 dataclasses/56 functions`，对main `1092/952/0/0/26`为`1.35x/1.33x`；新增复杂度是opaque authority registry、declarative request、typed facade与repository lifecycle。删除`requirements.txt`并让两个installer直接消费pyproject extras后，依赖真相也恢复为一个owner。

current virtual deterministic oracle固定使用packaged `zlc_neutral_atom/assets/imaging_template.json`、seed 7、deployed target以及同一个current compiler/playback。S3c重新核对后，virtual sensor不再把probe high window冒充camera exposure：camera固定积分20 ms，probe illumination为20/5/20 ms，短readout后的trap-held dark period延长到15.1 ms，使三条trigger间隔均为20.1 ms；`(R=1,P=3,Y=96,X=128)`三帧sum现为`3152983/3142311/3153945`。这处与旧virtual逐像素值的差异是纠正原“5 ms生成、20 ms provenance”静默矛盾后的预期变化，不是算法漂移；完整trailing image axes仍保留。bounded producer只在`trigger offset + fixed integration`后生成并发布record，避免host receipt早于所声明曝光或generator预取提前改变原子状态；queue满仍显式poison/fail，绝不覆盖旧帧或伪装成backpressure。远程endpoint保留exact `PreparedPulseRef`与server generation；真实RPyC反例让server SAFE阻塞`0.30 s`而client logical deadline为`0.05 s`，`0.058 s`即失败，随后立刻删除client并`gc.collect()`、重连新generation，backend全程仍只收到一次SAFE且server稳定为SAFE。另一并发反例让prepare停在backend内，同时请求SAFE；硬件事件严格为`prepare_start -> interrupt -> prepare_end -> safe`，被超越的prepare只得到RuntimeError且SAFE snapshot不含prepared ref，证明物理SAFE不会被迟到operation反写。由于没有real composition consumer与真机SafeStateContract，它仍是已排期能力owner而不是READY声明。旧root PulseGUI/TaskConsole、`session.connect`、manual/devtools与教程仍是legacy最后消费者，只进入删除/迁移账；当前普通用户GUI与真机链因此保持NO-GO，不能回填umbrella或旧RemoteSequencer协议。

**S3a committed-capture calibration纵切（parent `835c4cef90a5fefd7659771f0eb4677ea45f108f`）现为virtual/offline READY。** 本切片只接通已有算法与持久化owner，不修改main同帧物理数学，不运行或修改历史pytest，也不读旧`DeviceSet/Registry/session`来反向适配target。`CalibrationAnalysisRequest`从会顶层加载SciPy的`readout.analysis`净迁到轻量`readout.calibration`，所有production import直接指向新owner且不留旧alias；analysis+calibration+codec三文件合计由`4401 physical / 3984 nonblank / 26 classes / 21 dataclasses / 4 enums / 157 functions`变为`4403 / 3986 / 26 / 21 / 4 / 157`，净增只有两行说明文字/导入，没有第二类型、第二codec或兼容reader。fresh `import Zou_lab_control.notebook`仍不加载SciPy、Qt或Matplotlib，显式开始calibration时才进入analysis extra。

公开的唯一执行值是`CalibrationArtifactRequest(source_capture_ref, readout_binding, analysis, memory_limit_bytes, timeout_seconds)`。它不是“包一层就转发”的wrapper：`ReadoutFacade.calibration_request()`先用当前`CaptureRepository.inspect_final()`从持久provenance取得实际`ReadoutBindingKey`，随后把source identity、camera binding、科学意图、内存预算和deadline原子冻结；public constructor即使被直接调用，compiler preflight仍先做自己的FINAL inspection，再在repository borrows保护下admit/resolve一次并比较binding，不能靠facade的一次旧检查伪造。authoritative compile在启动Run前拒绝缺少独立`expected_centers_xy/maximum_site_residual_px`的analysis；preflight由具名`READOUT_EVENT` axis与显式reference/readout indices解析完整PointLayout、FrameContract和pulse/camera physical context，execute只调用已有同一数值owner，finalize消费held admission/resolution/proof并经`CalibrationRepository.final_commit`发布。不存在capture child plan、CalibrationService/Manager/Executor、filesystem fallback、隐式current calibration、自动model选择或按shape/rank/event位置猜语义。

bound facade的三个出口使用同一语义：request source、`load_calibration()`的admitted artifact以及`load_calibration_report()`的小artifact header都必须与`.for_binding(...)`精确相等；report只有通过小artifact/binding检查后才materialize full-resolution诊断数组。compile/start始终在Experiment operation guard内完成，`runtime.wait(handle)`在释放guard后执行，所以`close()`能取得锁、关闭admission并cancel/join；timeout由共享`zlc_storage.positive_real`在request边界拒绝bool/text/nonfinite/nonpositive输入并进入flat RunPlan，但只承诺现有checkpoint阶段边界与提交deadline，不冒充能异步硬杀SciPy；`1 ns` deadline反例在preflight明确失败且manifest为0。单run estimator无法保证calibration和occupancy的大数组在并发时的aggregate安全，因此两者共用一个`analysis/neutral-atom-readout`非设备EXCLUSIVE claim：第二个并发start明确得到`ResourceBusy`，没有为此建立queue、专用线程池或workflow scheduler。

current-format手工E2E使用tracked imaging template、seed 7、24个repeat和每repeat三个显式event，raw artifact保持只读`(R=24,P=3,Y=96,X=128)`与`capture.repeat/capture.readout_event/camera.y/camera.x`完整轴；layout声明reference `(0,2)`、readout `1`，5×7 expected centers由独立公式`x=37+9c, y=30+9r`产生而不调用私有detector/virtual center函数。public链成功提交并重载BOX/PER_SITE_PSF/UNIFORM_PSF三模型、35项threshold/component validity与96×128 report，最大中心残差约`0.395 px`；关闭Experiment后只重开Capture/Calibration repository，artifact canonical bytes、source ref、request和只读report仍一致。wrong binding、wrong AxisId、整体偏移centers和1-byte memory limit四个反例均以`RunFailed`终止且calibration manifest增量严格为0，随后同一runtime合法请求仍成功并只增加一个manifest。24组完整public calibration实测约`1.738 s / 3,141,376 bytes tracemalloc peak`，analysis estimator为`3,519,288 bytes`，未出现全DataBlock stack或trailing-axis flatten。两路独立只读对抗审查均判`P0=0/P1=0/GO`；保留的P2只有SciPy body内无细粒度checkpoint、notebook同步report decode期间持operation lock，以及极端site-count metadata在精确64 MiB拒绝前会先canonical encode。三者都不允许超时后commit或静默错数，只有真实GUI延迟/profile命中才增加worker borrow或更早metadata bound，不预建新executor。

**S3b committed-capture occupancy/detect纵切现为virtual/offline READY。** `DetectionRequest`冻结source capture ref、calibration ref、实际readout binding、唯一singleton `READOUT_EVENT` AxisId、显式或在request构造时已解析的model、预算和deadline；formal preflight先顺序inspection calibration→training capture→source capture，只保留六个compact标量并释放schema-bearing inspection图，再在同一capture/calibration/occupancy repository borrows下materialize exact admissions并逐window比较当前pulse physical context。execute只调用main继承的同一个feature extractor和严格`>`分类；FINAL持有同一resolved inputs，绝不重新admit或按shape猜适用性。输出以raw CAS arrays持久化counts、occupied和component-validity，完整保留source repeat/point axes并增加site data axis，即普通矩形例为`(R, P, site)`；site validity与每个site对齐，invalid count规范为正零、occupied为False，不把site或其它data axis取第零项/flatten/reduce。generation由本次RunId铸造，因此相同输入重复detect可以得到相同数值，但不会冒充同一次publication或复用旧snapshot identity。

三个repository的公开读取面现在都有明确资源边界：Capture `load/admit`服从repository resource policy而`inspect_final`还能接受更小的caller budget；Calibration `load/inspect/admit/load_report`与Occupancy `inspect_final/admit`接受caller memory budget。跨artifact dependency admission只消费各owner报告的retained/decode/read-scratch上界，不复制它们的估算公式。calibration与occupancy共享`analysis/neutral-atom-readout` claim，防止两个单独合法的大工作集在同进程叠加；它不是队列或新scheduler。generic pulse playback则只要求其实际waveform能力：cyclic loop、ramp和repeated-DAC可以被普通pulse执行消费，只有readout physical-context投影/校准才要求readout-specific可展开能力；不能为了calibration把本来合法的非readout pulse整体判失败。virtual/offline short sitemap facade已由S3c交付；qCMOS资格、真实remote composition和GUI site-map/gridplot仍未交付，故这些产品面保持NO-GO。FPGA/bitstream冻结、`AUTONOMOUS_STREAMED`唯一正常scan模式和“只有真机证据发现RTL bug才动硬件”的约束没有改变。旧相机producer、monitor/rolling dataset、ROI、temperature与尚未迁完GUI的最后消费者一项也未删除。

本次Rule-6全量海拔以当前`zlc_neutral_atom/readout/*.py` 14个production文件对main可比10文件机械计数：current为`10846 physical / 9827 nonblank / 59 classes / 50 dataclasses / 4 enums / 418 functions`，main envelope为`3283 / 2896 / 9 / 6 / 0 / 118`，表面总量为`3.304x / 3.393x / 6.556x / 8.333x / 3.542x`。这个>3x结果不能藏掉，但也不能把不同职责伪装成算法膨胀：真正可比的数值/领域core `analysis.py + calibration.py + occupancy.py`为`4879 / 4427`，对main同职责`3283 / 2896`仅为`1.486x / 1.529x`；其余增量明确属于main没有的durable repositories、canonical codecs、physical-context projection、typed artifact refs、flat RunPlan/lifecycle、资源与安全边界。逐文件最大项为`analysis.py 2426/2261`、`calibration_repository.py 1427/1334`、`occupancy.py 1301/1146`、`occupancy_repository.py 1224/1148`、`calibration.py 1152/1020`、`calibration_codec.py 1093/953`；它们不能仅因总包倍率被互相合并，因为算法、持久化、codec和runtime admission分别有不同真实consumer与失败边界。

全slice singleton/低消费者清点只留下三个需解释的低文本命中类型：`_OccupancyDatasetMetadataContract`负责把occupied与camera source metadata原子同封并实现metadata protocol；去掉它会让Value contract重复承担另一种payload。`_OccupancyConfig`冻结worker实际model及输入/输出schema identity；去掉它会把generic worker边界退回mutable closure/untyped tuple。`OccupancyDatasetEventAdapter`是未来Workbench monitor/rolling dataset的明确适配点，已有终态GUI consumer，不能因为当前S3b只交付offline path就删除。四个enum均有多个真实值（GridOrder 4、ReadoutModelKind 3、BoxReducer 4、BackgroundMode 2），不存在单成员enum。静态DAG仍只有`zlc_data -> zlc_storage`、`zlc_frontend -> zlc_data/zlc_storage`、`zlc_neutral_atom -> zlc_data/zlc_pulse/zlc_storage`、`zlc_pulse -> zlc_storage`；current production没有旧`DeviceSet/Registry/session`适配，也没有generic authority path中的`reshape(...)[0]`、`[:,0]`或`nanmean`式静默降维。这里的证据来自AST/consumer/import扫描、fresh import、current-format手写oracle和focused profile，不运行或修改历史pytest。

**S3c installation-qualified short sitemap纵切现为virtual/offline READY。** `ReadoutGridGeometry`是installation-owned、带frame shape、Y/X AxisId、opaque `CoordinateFrameId`、grid order与独立expected centers的immutable registration；同一个实例同时注入virtual atom array和`SitemapAcquisitionProfile`，所以simulator不能再用一组硬编码中心、detector再从自己的输出自证另一组。profile原子冻结`ReadoutBindingKey`、sequencer role、broker-attested `CameraPhysicalFacts`、exact live `PulseTarget`、trigger line、residual admission和packaged pulse recipe；runtime构造时逐项核camera port/sequencer port/target identity，facade没有sequencer override。profile与camera的frame shape、spatial AxisId和coordinate frame必须完全相等，expected centers必须finite/in-frame/unique且residual小于最近site间距的一半；detector结果在PSF fit/threshold训练前立即按ordered centers拒绝，而不是事后修正、snap或重排。

本切片保留main的long-short-long物理算法，但修掉旧virtual路径的静默provenance矛盾。qCMOS式edge-trigger working point由camera capability冻结20 ms integration；FPGA pulse在一个完整组内提供20/5/20 ms probe illumination，5 ms readout后用15.1 ms trap-held dark period补足，compiled trigger间隔为`20.1/20.1 ms`，跨组为`22.0 ms`。因此每个event持久化的camera setting与`FrameContract/ReadoutPhysicalContext.integration_seconds`都诚实为20 ms，而readout context的`ch03`在50 MHz下第250000 tick转low，保留真实5 ms probe waveform。virtual renderer的background/dark noise只消费camera integration，atom signal/loss/heating只消费probe overlap；producer等到`trigger offset + integration`才生成/发布frame并记录host receipt，既不在edge时提前宣称frame完成，也不由iterator预取提前改变原子状态。该修正只改current pulse asset与simulator事实，没有host sleep调度edge、HOST_STEPPED、RTL或bitstream变化。

`frames`明确表示完整三帧组数，不是总camera frames。facade在第一次Run前规范化所有group count、memory budget、deadline和analysis intent；profile再用pulse owner的`MAX_MATERIALIZED_TRIGGER_EDGES`在物化grouping前拒绝`3 * frames`越界。`RepeatRegion`包围base document首尾period，current compiler得到一次FIRE的完整硬件loop；explicit repeat-major grouping只把已编译trigger ordinal映射到`(repeat,event)`，不靠reshape/rank猜序。raw capture先成为独立FINAL `CaptureArtifactRef`，随后才开始普通calibration Run；第二阶段普通失败由`SitemapCalibrationFailed`、Ctrl+C由仍属于`KeyboardInterrupt`的`SitemapCalibrationInterrupted`携带同一个raw ref，capture阶段失败则不伪造ref。不存在child plan、current/latest calibration、filesystem fallback或回滚第一个artifact。

current-format oracle以12组产生只读`(R=12,P=3,Y=96,X=128)`和36条exact event，完整保留二维trailing data axes；成功提交/reopen 5×7/35-site的BOX、PER_SITE_PSF、UNIFORM_PSF模型，最大中心残差约`0.440 px`。artifact中的三项event exposure均为20 ms，readout physical trace为20 ms窗口+5 ms probe。零/NaN参数与超过materialization cap的group数在capture manifest出现前拒绝；1-byte calibration budget只使第二阶段失败，异常携带的raw ref仍可重开为`(2,3,96,128)`。脱离源码CWD可由`importlib.resources`加载asset，fresh notebook import不加载SciPy/Qt/Matplotlib/serial/RPyC，current production没有旧DeviceSet/Registry/session适配。两路独立只读复审最终均判`P0=0/P1=0/GO`；其间发现并已关闭camera/probe伪曝光、trigger间距、frame完成时间、sequencer自由override、坐标身份、采集后才验参数、unbounded grouping和Ctrl+C丢raw ref等问题。

S3c Rule-6海拔按本切片13个受影响production Python文件相对parent机械计数为净`618 physical / 565 nonblank / 4 classes / 2 dataclasses / 0 enums / 18 functions`；其中3个额外文件只更正迁移后失真的路径/时序说明，行数与AST计数净变化均为零。main中可比的`CalibrateReadoutTask + ReadoutSubsystem.sitemap`为`543 / 518 / 1 / 0 / 0 / 16`，即`1.138x / 1.091x`，没有超过3倍。四个新增class恰是两个有独立失败语义的public recovery exception及两个有多consumer约束的值：去掉`ReadoutGridGeometry`会恢复simulator/profile两套几何，去掉`SitemapAcquisitionProfile`会把camera/sequencer/trigger/target/coordinate物理接线重新交给facade字符串，去掉任一异常会使普通失败或notebook interrupt在无hidden latest/list API下丢失已提交raw ref。没有新增enum、plugin、manager、repository、executor、workflow或compatibility wrapper；917行JSON只是从无owner根路径移动到neutral-atom package并改正三项probe命名、一个5 ms窗口后的dark gap，不作为代码膨胀。极端合法group数仍会在pulse owner的一百万edge硬界内物化显式pair tuple；它有界且当前典型值仅几十，为这个未profile命中的边角再引入第二种compact grouping类型会增加codec/union复杂度，只有真实内存profile成为瓶颈时才改owner表示。旧camera producer、monitor/rolling dataset、ROI、temperature和GUI最后消费者均未因本切片删除。

**S4a/W3c autonomous scan virtual authority现为virtual/offline READY，Formal hardware仍NO-GO。** logical document先绑定当前PulseTarget并以bound sequencer公布的`resident_scan_point_capacity`拒绝超限`R*P`，pulse owner再把whole-document repeat展开为repeat-major physical rows；current oracle中`R=2,P=8`得到`point_count=16, loop_count=1, full_point_loop=True`，cell严格为`r0/p0..7`后`r1/p0..7`。scan domain使用具名`scan.parameter.*` axes和原PointLayout；source为`(R,P,READOUT_EVENT,*data_shape)`，CommittedTransform只强制具名删除singleton event axis，output保持`(R,P,*data_shape)`、全部data axes和component validity，不存在`[:,0]`、flatten、trailing-axis mean或匿名`data_dim`。

W3c已经删除过渡性的ScanIntent/raw Capture promotion双历史。direct camera与camera→occupancy分别用两个强类型application入口复用现有TriggeredPipeline；两者都是一次camera arm、一次FPGA autonomous FIRE、同一exact schedule/cursor/EOS，再在**同一个Run**中transform并通过RunController final-commit gate发布唯一ScanArtifactRef。occupancy source是counts `(R,P,READOUT_EVENT,site)`，occupied sibling与event metadata在transform前释放；完整DatasetSealProvenance持久化processor stage和CalibrationArtifactRef。任何gap/duplicate/out-of-order/short EOS/processor failure都使Run失败，没有latest fallback、raw side artifact、promotion recovery或第二repository history。纯compile/`inspect_scan`只编码并验证immutable PulseDocument与CompiledPulseArtifact的ContentRef、policy、pulse owner 128 MiB硬上限、CompiledPulseRuntimeSummary及document/compiled retained headroom，不写repository；真正Run的wrapper preflight先把同一字节预存CAS并复核identity，再调用base hardware preflight。取消或后续preflight失败只留下不可达CAS安全orphan，不发布artifact，也不形成可查询历史。

ScanRepository是canonical output dataset与FINAL manifest唯一authority。output DatasetRevisionRef由logical document、exact source ref和ScanOutputContract派生并继承source generation/revision；manifest另以实际output values/validity、source schema/provenance、pulse document/compiled evidence与safety bundle内容寻址。`MaterializedScanData`只暴露ScanArtifactRef、source DatasetRevisionRef和最终OwnedSnapshot。zlc_data同一个峰值函数既在底层plan编译前按schema+CommittedTransform拒绝不足预算，也在实际materialize前按真实validity复核；occupancy final阶段另扣除plan常驻calibration arrays，direct与occupancy都扣除document/compiled公共retained headroom。loader只解析一次small index；`inspect_final()`、materialize与metadata-only FigureDocument不读取或解码compiled IR/PulseDocument，完整`admit()`才在读取blob前用持久化runtime summary检查decode peak并在解码后重算exact compare。scan policy直接复用pulse owner的payload ceiling。bounded transform surface只承诺system singleton cell drop、cell full-range no-op与trailing data select/reduce；repeat/point select/reduce仍拒绝。

notebook的`figure_document(scan_ref)`只inspect manifest/index/schema，`figure(scan_ref)`才bounded materialize并复用role-driven DataFigure。`exp.scan_gui(frozen_request)`的direct-camera/API-slot产品明确为FINAL-only；autonomous occupancy产品在同一Run先显示exact `SCAN_POINT` PROVISIONAL curve，preview退役后原子切换canonical FINAL raster。start/result/projection/reap仍在有界worker，GUI只看revision-gated ViewModel与immutable front；新Run清除上一轮FINAL，projection失败保留FINAL ref，close/cancel异步reap。同一projection cap按顺序约束materialize、evaluator、Agg与Qt front。U0.3d现已让`fit(scan_ref)`、DataFigure Analyze、Save/reopen与saved-grid explicit Refit使用exact ScanArtifact source；没有删除legacy PulseScanNode/MeasurementSpec，因为TaskConsole及其余旧消费者尚未迁完，完整多plot产品仍未完成。

Rule-6仍以既有scan package加integration owner为同一归因口径。W3b的历史净增为`241 physical / 225 nonblank`；当前W3c按21个实际受影响production文件相对parent机械计数为`12309 -> 14787 physical`、`11006 -> 13242 nonblank`，净`+2478/+2236`，classes `108 -> 118`、dataclasses `50 -> 60`、enums `5 -> 5`、functions/methods `592 -> 709`。本切片对main窄等价`pulse_scan.py 370 + PulseScanNode 483 + _PulseSlotsWidget 298 = 1151 physical`为`2.153x`，低于3倍；controller+thin Qt本身`619+258=877`，对旧node+slot widget `781`为`1.123x`。累计scan envelope为`2221+2478=4699`，对main累计窄等价`703+1151=1854`为`2.535x`。

增加的复杂度不是为完整TaskConsole预建抽象：两个强类型direct/occupancy入口共享一个private RunPlan wrapper；新增值类只表达occupancy已解析schema、纯static-lineage admission、pre-FIRE staged lineage、轻量FINAL inspection、controller ViewModel/work token和immutable raster。没有新增enum、workflow engine、manager、generic artifact framework或scan-fit codec；没有消费者的ScanArtifactRef codec/input-ref已在本轮删除，等真实scan-fit接入再由owner增加。`+117`函数/方法主要来自严格owner codec、current-only repository load/stage边界、controller生命周期及两条public composition seam；其中resource闭包又复用现有CompiledPulseRuntimeSummary、FigureEvaluationPolicy与同一transform estimator，没有建立budget manager。一次独立对抗审查发现的三个P1均源于“阶段局部cap冒充端到端cap”：长期compiled/document retained漏算、materialize先大解码后拒绝、GUI render/Qt绕过projection cap；上述prestage/index/顺序峰值修正与三个反例已经关闭，当前本地闭包复核为`P0=0/P1=0/GO`。

中间迁移态的“当前 production 调用数”只是一条证据，不能单独裁决目标能力：

| 能力 | 明确的终态 consumer | 审查动作 | 允许物理删除的条件 |
|---|---|---|---|
| `MonitorTap -> MonitorDataset` bounded live | Workbench live image、rolling curve、histogram、board coherent present | W1已接finite preview，U0.2f已闭合其raw IMAGE完整交互；M1/M2与U0.2b/c/e已把独立virtual free-running MOT camera、raw IMAGE、显式typed ROI scalar、独立长scalar history、同一scalar revision的CURVE/HISTOGRAM/METER、raw/scalar同源四panel，以及IMAGE/CURVE/current ROI scalar Histogram完整selector/display control接入public notebook/Qt；U0.2d另闭合FINAL exact-cell Sites交互并保留双ComponentValidity和同cell lineage。real Pylon qualification、live paired Sites、generic Distribution/Grid及H2 histogram analysis仍待后续 | 同等 GUI capability 已由更小唯一 owner 交付并迁完全部 panel |
| finite exact DatasetBuilder | capture、scan、processor、artifact | 保留 exact schedule/cursor/gap-fatal；与 display window 分责并收敛重复 delta/materialize | 不删除；只替换内部实现且 exact oracle 等价 |
| StreamProcessor/occupancy | S3 readout、未来 analysis processor、冻结Figure以及live grid/selector | W4c已交付按occupied/counts单块选择的offline frozen Figure；保留task/measurement/processor语义，后续只补真实live/grid/selector consumer并继续合并重复guard | 真实 replacement 同时交付全部offline/live consumer、E2E 与 lifecycle oracle |
| calibration/readout | notebook、Workbench、occupancy、site-map/gridplot | 默认继承 main 真机数学；W4b已交付paired artifact/report reopen与static frozen report，后续只补真实live/interactive consumer | 不因部分GUI已接入就删除；最后live/grid/edit consumer迁完或产品明确取消后才可删除 |
| `FitResultBatch`/batch axes | gridplot/site/component fit | 保留；收敛重复 fit owner 与镜像模型测试 | gridplot 不再需要批量 fit 或出现更小等价值类型 |
| `zlc_frontend` Figure/View/selector | S0.5/S2/S5 Workbench 与 notebook render | 审查 target DAG、最小入口与性能，不以旧 GUI 尚未迁入为由删除 | 新 frontend 设计被正式替换且所有目标用户面有闭环 |
| current `ScanArtifact -> DataFigure/typed scan panel/TaskConsole SCAN_SLOT` | notebook direct/occupancy scan inspect/render、direct FINAL-only Qt、occupancy PROVISIONAL→FINAL Qt、TaskConsole Add/Setting/Edit/Start/Stop/Save/Load 与 camera→occupancy-counts y；后续API_SLOT、scan-fit及其余panel | W3d已交付source-neutral exact pipeline、processor y、stable output和exact progressive occupancy panel；W3e复用该产品接入静态catalog与单revision编辑，不复制Run/render状态机 | API_SLOT、scan-fit及最后legacy panel consumer各自交付后，才dependency-closed删除对应旧路 |
| legacy PulseTableState readers | 尚未迁完的 Camera/PulseScan/PulseGUI 岛 | 先迁每个保留 consumer 到 current PulseDocument/compiler | 最后 consumer 迁走的 H1/S3/S5 dependency-closed commit |

规则状态必须诚实记录为 `NOT_STARTED/PARTIAL/COMPLETE`。§22.1固定的11个互斥production owner闭包已逐一完成Rules 1–7 existing-code审查，所以“完成1–7前不得继续迁移”的历史冻结曾解除；规则8/9是持续per-commit gate，不能一次永久COMPLETE。virtual finite capture/pulse owner、BoardController coherence foundation、W1 single-event finite exact capture、S3a/S3b/S3c、W3e/W3f、W-UI1a及U0.2b/c/d/e/f已经证明的底层/authority/lifecycle与finite/live camera IMAGE、CURVE/current ROI scalar Histogram、SCAN_POINT及exact Sites交互事实保留，但其余缺失UX按§2.2/U0继续MUST_CLOSE。既有结案证据包括preflight→execute→FINAL→admit current-format oracle、预算/recovery、headless import、exact多维轴/component-validity、repository/Run生命周期、DAG/shape扫描与有界对抗审查；这些机械证据必须叠加活动白名单完整collect+run与适用的UX salvage/E2E。remote/real qCMOS、正式scan物理资格、完整GUI scheduler、live paired Sites、generic Distribution/Grid、Histogram H2、其它panel的Save/Setting/Edit、完整GUI site-map/gridplot和普通用户真实设备入口仍分别NO-GO；不能再把已经完成的finite/live camera IMAGE、camera rolling/overlay/current ROI scalar Histogram、occupancy SCAN_POINT或exact occupancy Sites笼统写成NO-GO，virtual/offline API segmented也不能倒写成真实qCMOS Formal资格。每个后续dependency-closed切片须按Rules 1–9做增量审查，完整迁移结束后再对最终分支做一次独立对抗验收；局部GO或owner audit COMPLETE都不能改写成全局实现完成。

W4b之后，上段“GUI site-map/gridplot仍NO-GO”精确指可交互live/focus/zoom/export与TaskConsole consumer；current calibration artifact/report的static frozen Overview、三model histogram grid与empirical PSF grid已经GO。冻结报告不能倒写成真实qCMOS qualification、live product或可编辑calibration已完成。

**历史 checkpoint 规范效力裁决：** 自此以下所有“最新实现”“下一切片/下一checkpoint”“不重开W…”“局部GO/COMPLETE”只记录当时的代码、发现和证据，不再决定当前顺序或终态产品能力；当前唯一顺序是§19 U0及其后4–10。`不重开`只表示不重复审查已经证明的数值、authority、预算、线程与lifecycle不变量，绝不能阻止补齐缺失UX或修正被规则9否决的实现。任何whole-Run monitor retarget、permanent frozen/display-only viewer、空interaction handle或独立fit_gui主入口措辞均无规范效力。

实现 checkpoint：W3e current TaskConsole SCAN_SLOT 纵切已经完成，本段、代码与白名单测试由同一显式提交冻结。`Experiment.task_console()` 现在打开一个 current one-card product：Catalog composition 显式包含 Autonomous SCAN_SLOT Task、camera Measurement 与 occupancy StreamProcessor 三项静态 Definition；每项都必须被投影，Definition 不含 callable 或本次绑定的 schema/deadline，旧 runtime/processing 模块也不再转出口冒充第二 owner。Add Panel、Setting、Edit、Run Scan、Stop、Save/Load 都只操作 typed `TaskConsoleScanIntent`，并复用 W3d 的同一个 `ScanWorkbenchWindow/ScanPanelController`、exact camera→occupancy pipeline、progressive worker raster 与 FINAL artifact projection；没有第二套 Run 状态机、signal hub、renderer、raw device 或 generic workflow graph。

SCAN_SLOT intent 在 request 存在前必须冻结 owner PulseDocument、完整具名 ScanPointTable 与全部 whole-run API 常量；未知/缺失/重复 API、非 SCAN_SLOT pulse、未知 DefinitionKey、direct-camera 携带 SITE display、occupancy 缺 calibration 或任意 raw runtime state 均拒绝。camera→occupancy counts 是本切片唯一真实 external Measurement→Processor y；legacy arbitrary `signal_expr`、latest/history helper、monitor worker output 与 scalar-only data contract不迁入target。occupancy source的`DatasetBuilder`仍是exact reservation/cursor/append/seal唯一authority，底层始终保留`(repeat, point, *data_shape)`、具名axes、ComponentValidity、revision、source generation与artifact lineage；repeat mean、SITE auto/batch/select及其它显式选择只属于View，权威`ScanOutputContract/DataTransformSpec`不从当前画面派生。Calibration default 作为artifact-owned默认模型语义保存，UI不把它静默改成BOX。

Setting/Edit 共享一个 owner-thread revision session；stale draft返回typed conflict，Apply只在既有panel完全idle时原子reconfigure并清除旧FINAL，不取消active run。成功Apply后当前form从exact session snapshot全量重填，Cancel/Load/direct-source normalization同样清空所有不属于当前intent的隐藏字段，避免stale calibration/model复活。Start的既有worker才执行依赖calibration/device/schema的完整bind/preflight；失败保持NOT FINAL。Save/Load只接受一个current canonical tree，嵌套值委托DefinitionKey、PulseDocument、CalibrationArtifactRef与DataTransformSpec owner codec；没有版本链、升级器、compatibility alias、DeviceRef、RunHandle、reader/front buffer或provisional state。加载的权威transform在两个editor中都以精确AxisId摘要可见且可清除，不能被display intent覆盖。Stop/close继续由嵌入panel唯一owner完成cancel、terminal reap、preview退休与renderer释放。正常硬件执行仍是现有bitstream上的一次camera arm、一次`AUTONOMOUS_SCAN_ONCE` FIRE和完整resident table；没有HOST_STEPPED_GROUP、software timing、逐cell handshake、RTL或bitstream修改。

W3e Rule-6按19个受影响production文件相对parent `e77b0559`机械计数：`9813 -> 11599 physical`、`8821 -> 10408 token-NCLOC`，净`+1786/+1587`；classes `68 -> 79`、dataclasses `45 -> 51`、enums `1 -> 1`、functions/methods `441 -> 523`。两个新文件为headless intent/catalog/editor/codec `593/531` 与Qt composition `943/841`，其余静态Definition拆分、scan reconfigure和public lazy seam净`+250/+215`；其中相对初版新增的123/117行专门关闭“隐藏权威transform”“取消的hidden calibration/model复活”与未配置card timer未停三个具体漏洞，不是通用transform editor。main中不重复计算W3d执行/render，只取旧TaskConsole的PulseSlots、MeasurementPanel、LogicNodeEditor、LogicNodeConfig/TaskConsoleState及同壳内catalog/edit/build/save/load/shutdown方法，机械并集为`1705 physical / 1283 token-NCLOC`；本轮比为`1.048x / 1.237x`，没有超过约3倍。新增9个真实产品类加静态Definition/display边界的净2类均有当前consumer；没有新增enum、manager、plugin、通用pipeline DSL或单层compat wrapper。focused Definition/scan/TaskConsole联合测试为29 passed，另有compileall、diff-check、target import-DAG、旧Definition转出口、shape-loss、raw-runtime persistence与execution-mode机械扫描；独立对抗终审P0=0/P1=0后本纵切GO。该切片当时的下一 checkpoint 是只为已接受例外语义建立 `API_SLOT_SEGMENTED_EXISTING` current纵切，现已由W3f完成；在其余 Measurement/Processor、rolling/gridplot/selector/calibration/temperature/MOT 与旧GUI consumer迁走前，仍不提前删除legacy TaskConsole/PulseScan宿主。真实qCMOS Formal仍因Q0/EndAttestation保持typed NO-GO，RTL/bitstream冻结。

**实现 checkpoint：W-UI0 current Qt/render owner与复用前置切片完成。** 经逐项对照旧实现确认，旧`frontend/qt_fluent.py`不是参考样本而是成熟组件authority：其完整widget/lifecycle/layout实现整体迁入`zlc_frontend.qt_widgets`，Qt palette/QSS/QPainter/window metric进入同leaf的style owner，`QtImageBoard/QtOwnerWake`一并收口；旧文件、旧`zlc_frontend.qt_board`、production旧import/deep import与current手册旧路径物理消失，不留alias/shim。W1 capture、W2 PulseWorkbench、W3 scan与TaskConsole直接取用既有Fluent button/label/input/combo/spin/group/tab/scroll/path/setting-row；W2 finite-repeat表单改取`FluentFormGrid`，三处文本请求和两处确认统一经`fluent_text_prompt/fluent_confirm`，不再漏回native chrome；W3 FINAL raster复用`QtImageBoard.present_encoded`，artifact id用既有`ElidedLabel`保留full tooltip，不再让`QLabel`的pixmap或长文本sizeHint把窗口撑出屏幕；W3仍只有一个`ScanIntentForm`类型服务Setting/Edit，其大内容区可滚动而validation/status和Apply/Cancel footer固定可达。四个Workbench launcher在构造widget前统一auto scale，show后按真实frame统一center，repo-root三个current launcher也统一`ensure_qt_app`；真实offscreen `800×600`在READY/FINAL后仍完整可达。Start/Run/Apply、Stop/Hold/Load、Cancel/Remove/Clear恢复旧GUI的GREEN/ORANGE/GREY动作语义。raw Qt只剩layout/container/table/file-dialog与领域QPainter，不增加GenericRunPanel、SettingsEngine、RunControlStrip、dialog或table framework。

复用审查没有把“换成公共类”误当行为等价：`FluentDoubleSpinBox`原先按显示字符量化，W2 pulse duration/delay改用同组件的non-quantizing合同，真实clock-grid大值round-trip不变；`FluentPathEdit`选择坏PulseDocument会先显示坏路径，现validation失败恢复最后committed path且document fingerprint不动。首次从Python worker打开GUI会在worker创建QApplication并让后续Qt线程比较恒真，现`ensure_qt_app`在首建前fail-closed；`retain_window`既会把W3布尔`closed`误当signal，也不能依赖普通`QWidget.close()`发`destroyed`，现公共owner只连接真实signal并提供幂等`release_window`，四个异步窗口只在cancel/reap/worker shutdown完成后的committed close释放，真实launcher测试均确认registry回到基线，且没有用`WA_DeleteOnClose`破坏对象生命周期。

旧`frontend/style.py`及其Helvetica资产同样是成熟render style authority，已整体迁入唯一`zlc_frontend.render_style`并删除旧owner；新增FigureDocument artist token只扩展该palette，不另造第二套series/rcParams。公开nested style值改成tuple/只读mapping，font package-data随新owner进wheel；IPython/widget hook拆到lazy `notebook_integration`，render leaf不探测notebook环境。因为`matplotlib.rc_context`修改进程全局状态，current FigureDocument/Agg产品的构造/draw/save/clear现由同一RLock串行；`DataFigure.to_png_bytes/export`把实际Agg draw收回render owner，caller-owned`render()`不冒充后续draw隔离。current Figure显式接收DPI并使canvas allocation与memory preflight一致，关闭了继承旧`figure.dpi=300`后预算按100dpi低估约9倍的洞；产品save与live renderer复用同一Figure/Canvas释放owner，GC-disabled连续PNG/SVG也不保留上一块Agg资源。notebook façade冻结调用者的总render预算到`DataFigure`，所以隐式`_repr_png_`与后续export不能绕过准入。旧live pulse scan也不再反向读取Qt颜色/字体。尚未迁出的共享-Figure RenderLoop仍只属于`SerializedLegacyAggBridge`隔离岛，本checkpoint不把它冒充为current compose lane完成。

W-UI0按36个production Python union path（parent存在29、current存在32）相对`db2dc408`机械计数为`40361 -> 41167 physical`、`36507 -> 37257 nonblank`，净`+806/+750`，比为`1.0200x/1.0205x`；classes`139 -> 139`、dataclasses`12 -> 12`、enums`0 -> 0`、functions/methods`1878 -> 1895`，无新增产品class或超过约3倍项。第35条是为冻结端到端render预算而实际修改的notebook façade，第36条是被机械审查抓出的current root`pulse_gui.py`，都不能藏在原34-path账外；其余增量来自curated Qt facade、两个style owner leaf、薄文本/确认helper、复用immutable image board、串行compose/save/release边界、显式precision/lifecycle/layout合同与机械ratchet，不是新的GUI framework。focused tests按process-lifetime installation边界分进程运行共90 passed（Qt/render owner 12、W1 7、W2 9、W3 application 5、panel controller 12、TaskConsole 7、frontend figure/render 38），另有target compile、diff-check、fresh import、并发rc恢复、DPI canvas、GC-disabled连续export与fault injection、端到端budget、READY/FINAL frame containment、old/deep owner、raw-control/modal/QApplication alias、font package-data、style literal与extras ratchet。该切片当时的下一产品纵切`API_SLOT_SEGMENTED_EXISTING`现已由W3f完成；任何后续W窗口仍必须先通过§4.3.1的salvage/reuse门。

**实现 checkpoint：W3f `API_SLOT_SEGMENTED_EXISTING` virtual/offline纵切与W-UI1a静态表单owner已由当前提交完成。** W3f保持整run一次camera arm/exact capture transaction，R-major/P-fast执行R×P个独立`STATIC_ONCE` pulse session并持久化唯一aggregate camera evidence；W-UI1a把W1/W2/W3已用的静态scalar/typed choice/atomic populate收口到current唯一form owner。该历史checkpoint当时尚未开始W-UI1b/W-UI1c并把rolling/overlay记为NO-GO；其camera IMAGE/CURVE与occupancy SCAN_POINT部分现已被U0.2b/c明确关闭，真实qCMOS Formal仍NO-GO。

W3f/API segmented与同一提交交付的W-UI1a按24个实际受影响production Python文件相对parent `37c7893`机械计数：`16775 -> 21528 physical`、`15174 -> 19506 nonblank`，净`+4753/+4332`；classes `115 -> 140`、dataclasses `64 -> 81`、enums `3 -> 3`、functions/methods `785 -> 963`。main没有“一次camera arm + R×P独立STATIC_ONCE terminal + canonical reopen”的等价API实现，因此不伪造窄算法倍率；只把main的`pulse_scan.py + task_console.py + pulse_gui.py = 14450/13319 physical/nonblank`作为广义迁移包络，本提交全部production净增量是其`0.329x/0.325x`，而不是声称职责一一等价。W-UI1a两个current owner的独立倍率见下段；W3f新增量集中在typed program/lineage、one-arm segmented runtime、direct/occupancy共用application接缝和canonical repository reopen，不另造并行GUI或旧signal graph。四类durable execution/camera evidence、runtime binding/result/spec、一个RunPlan prepared state和一个窄capture protocol分别承担persisted-vs-live authority、one-arm lifecycle或direct/occupancy共用接缝。已删除只有一个字符串字段的`SegmentationSemantics`，没有新增enum、workflow engine、generic scheduler、per-segment camera wrapper或host-stepped fallback。Repository metadata准入不是R×P固定公式冒充完整证明：cardinality floor只负责早拒绝；真实静态事实冻结后由repository私有receipt绑定actual source block/schema、lineage、camera/result stream-source topology与execution shape，terminal byte/node上限仍归pulse/camera owner，FINAL发生任何事实、codec或容量漂移都是内部不变量错误。`ScanRepository.resource_policy`在构造期冻结为write-once；需要换policy就建立新repository/generation，不能在已mint receipt与FINAL之间热改。该receipt只证明metadata canonical边界，不替代values/validity/manifest的一般sink准入，也不落盘、不公开或扩成通用预算框架；camera interval函数只补硬件terminal/tail与下段prefix未覆盖的余量，不成为edge scheduler。对应focused oracle为API 9、W3/Form 37、W2 authoring/application 15、W2 Qt 9项全部通过；其中600k合法camera/result stream/source标识同时覆盖direct与occupancy topology，不再在FIRE后late-fail。真实qCMOS仍受Q0/H1门禁，不得把virtual通过倒写为Formal资格。

**W-UI1a静态声明式表单门已经由current实现兑现；W-UI1b/W-UI1c仍按consumer保持NOT STARTED，不得混成“全部UI迁移完成”。** `zlc_frontend.form`现拥有headless immutable FormSpec与共享number leaf coercion，`zlc_frontend.qt_widgets.FluentParameterForm`拥有closed `normalize/build/read/write/is_empty/refresh` handler mapping、typed choice、non-quantizing number与exact-key原子populate；W3 role/budget/deadline及PulseDocument API常量从typed owner经Workbench projector进入同一路径，W2/W3复杂table的普通number leaf也委托共享parser。旧`frontend/param_widgets.py`中float/int/bool/text/ordinary choice/device choice的五操作门面已改为调用current handler owner，只为尚未迁走的legacy consumer保留instant-apply facade；segmented toggle及json/path/axis/signal/pulse_slots等真正legacy专用分支留在各自有最后consumer删除点的岛，不得被current窗口使用。动态stream/device option snapshot+revision仍必须在monitor/Measurement/Processor纵切前完成W-UI1b；`RateLimitedApply`原源码迁移与teardown flush仍必须在DeviceManager/live-control纵切前完成W-UI1c。Pulse/API table、calibration/transform/ROI/fit等复杂presenter继续显式，但其普通scalar leaf不得再分叉。W-UI1a的两个current owner文件为`807/676 physical/nonblank`，对main旧`param_widgets.py`的`922/752`是`0.875x/0.899x`；迁移期再加仍有最后consumer的旧adapter岛合计约`1.90x`，但机械ratchet证明普通scalar只委托current handler而不保留第二份控件/解析实现，最后legacy consumer迁走时整岛删除。

QApplication事实也在本轮校正：current production只有`zlc_frontend.qt_widgets`中的`ensure_qt_app()`拥有构造权，W1/W2/W3调用同一个singleton，不存在三份QApplication authority。剩余重复只是少量`owner-thread -> scale -> factory -> screen-fit -> show -> center -> retain` launcher choreography；只有在不包裹原窗口、不接管close、不持有Run且四个真consumer合同一致时，才可提取薄`launch_owned_top_level(factory, size_ratio)`。不得为了消除这几行重新引入`FluentWindow` wrapper或GenericRunPanel，从而破坏各窗口自己的cancel/reap/committed-close owner。

**历史实现 checkpoint：W4a current frozen DataFigure Qt viewer线程/内存纵切完成，产品交互未完成。** `Experiment.figure_gui(...)`通过lazy notebook→workbench optional edge打开capture/scan/fit source；notebook只交出绑定后的`.figure`窄callable及显式intent/selection/preferences/budget，viewer不反向import notebook facade、不保存Experiment、raw runtime、repository resolver、Figure或artist。module-owned唯一capacity-one lane顺序完成source admission、DataFigure evaluation与一次Agg/PNG整板compose，关闭以cooperative event撤销未开始/阶段间工作；已经进入不可安全强杀的repository/Agg调用只在后台诚实排空，Qt owner不等待且不发布stale result。跨线程DTO只有owned immutable PNG和string summary；Qt decode前同时准入source front与当前device-pixel-ratio物理board front，成功只调用一次`QtImageBoard.present_encoded()`，因此多panel/grid仍是一张coherent front。ViewSpec/display reduction不得升级为FitSpec、ScanOutputContract或权威artifact这一类型边界继续有效；“frozen/display-only即终态viewer”的产品结论已废止，U0纠正3必须补zoom/pan/re-fit/export。

W4a没有复制DataFigure/FigureEvaluator/FitResultBatch/renderer或Qt widget owner，只复用W3a/W-UI0既有实现；共享PNG IHDR/front估算从W3 final scan中的局部副本收口到`zlc_frontend.image_raster`唯一纯函数。相对parent `9d20dba`，五个实际受影响production Python文件（notebook facade、lazy workbench入口、新figure window、image raster helper、scan presentation委托）为`3277 -> 3719 physical`、`2936 -> 3322 nonblank`、`2930 -> 3317 NCLOC`，净`+442/+386/+387`，classes`17 -> 19`、dataclasses`9 -> 10`、enums`0 -> 0`、functions/methods`152 -> 173`；整体仅`1.135x/1.132x/1.132x`，没有超过约3倍。新`_figure.py`为`335 physical / 296 NCLOC`。main旧`frontend/figure_viewer.py`的saved-NPZ Browse/gallery/Edit/Fit/Flow/重保存广义包络为`965 physical / 691 NCLOC`；W4a净增量是其`0.458x/0.560x`，只作复杂度包络而不伪称职责等价。旧saved viewer的lazy export、session/neutral GUI及root launcher仍是最后consumer，所以本切片没有可dependency-closed删除的旧文件；其产品能力必须由后续独立纵切替换后一次性删除，不能把W4a当兼容层。

focused W4a offscreen oracle为`5 passed`：真实FINAL capture ref的`.figure_gui()`证明repository materialization/evaluation和Agg都不在Qt owner；显式view/budget完整转发；两cell facet只发布一张immutable PNG；低于source decode和低于物理board front的两类预算都在Qt decode/present前拒绝；两个窗口共享capacity-one lane，queued close可取消，inflight close低于`0.1 s`且terminal后零stale present；关闭viewer后Experiment仍可读。另有fresh-process headless import（Qt/Matplotlib均未进入）、compile、diff-check、PNG独立几何/预算oracle、owned-file notebook/legacy/raw-runtime与shape-loss机械扫描；一次有界独立对抗终审`P0=0/P1=0/GO`。该历史checkpoint只完成冻结只读Figure窗口；其中当时未完成的camera rolling/overlay与occupancy SCAN_POINT现已由U0.2b/c关闭，interactive fit/edit、saved gallery、通用calibration/site-map/gridplot交互与W-UI1b后续能力仍未完成。其下一纵切记录只表示当时顺序，不重开已完成的旧UI复用审查。

**历史实现 checkpoint：W4b current calibration artifact/report frozen Workbench纵切完成。** `CalibrationRepository.load_computation()`在一个artifact+report decode/admission中返回既有`CalibrationComputation`并再次执行analysis owner的完整binding验证；公开notebook seam为`.readout.load_calibration_computation(ref)`，GUI seam为`.readout.calibration_report_gui(ref)`。Workbench不只加载report，因为runtime threshold、usable site、真实centers/boxes与empirical PSF kernels属于artifact；也不只加载artifact，因为reference labels、stored signals/bins、model/held-out fidelity和PSF fit diagnostics属于report。renderer不重跑detector、Otsu、Gaussian fit或PSF fit，五页只展示Overview、BOX/PSF/UNIFORM_PSF各自site histogram grid和empirical PSF kernels；所有原始site数组仍保持一维canonical order，没有`reshape/flatten/[:,0]`式数据轴降维。

阈值来源不能由数值相等反推：analysis owner的`calibration_runtime_threshold_sources()`与runtime artifact构造共用同一个all-model gate，明确逐model/site输出`formal`或`quick-fallback`；即使quick与formal数值相同，UI仍显示真实来源。`runtime_usable => finite threshold && feature-valid`，但unusable site可合法保存NaN或有限threshold；该site显示`T=N/A`/`runtime-unused`且不进入pooled histogram和summary均值。GridOrder同样由neutral owner的`site_grid_positions_yx()`唯一派生，ROW_MAJOR、SERPENTINE、COLUMN_MAJOR、COLUMN_SERPENTINE都把canonical site向量放到明确物理`(row,column)`，frontend只验证完整bijection并消费，不自己看shape猜顺序。image/site/signal/label/component validity分别保留，有限invalid filler不会进入直方图。

W4a原窗口壳已压缩为shared `FrozenRasterWindow`；DataFigure与Calibration共用一个capacity-one executor、cancel/reap、owned immutable `EncodedRasterDocument/Page`、Qt owner decode和atomic multipage present。每个报告page在同一个Matplotlib RLock中完成Figure构造、draw/save、release与GC，页前后检查cancel；已经生成的全部PNG与当前canvas、view arrays、repository-owned artifact/report retained upper bound使用同一个总预算，不能把同一512 MiB分别发给load和render后相加突破。worker结束只留下encoded bytes；Qt阶段再顺序准入全部source/scaled physical fronts，任一页decode失败清空全部board且绝不进入partial READY。frozen calibration report可以继续不取得Experiment、repository、Figure、artist或edit/recalibrate authority，但“永久只读且无交互”已废止：报告类多页至少补zoom，大site grid的focus/selector/export按规则9 salvage；显式recalibrate仍创建新request/artifact而不修改旧report。

Rule-6按相对parent `531eef1`的11个实际受影响production Python文件机械计数：`8381 -> 9522 physical`、`7629 -> 8658 nonblank`、`7553 -> 8582 NCLOC`，净`+1141/+1029/+1029`；classes`42 -> 45`、dataclasses`27 -> 30`、enums`4 -> 4`、functions/methods`307 -> 339`，整体仅`1.136x/1.135x/1.136x`。窄看calibration-specific `_calibration.py + calibration_render.py`为`849 physical / 783 NCLOC`，相对main静态文件写出器`frontend/calibration_report.py 199/155`确实超过3倍；这不是隐藏而是触发了压缩：W4a的335行专属窗口被抽为115行薄adapter，线程、Qt生命周期与multipage present进入两个真实consumer共享的297行shell，跨线程page/document DTO也由两条产品路径共同消费；没有新增enum、repository、manager、第二executor、workflow或compatibility wrapper。与真实用户面的main广义包络`calibration_report.py + figure_viewer.py = 1164 physical / 846 NCLOC`相比，calibration-specific current面为`0.729x/0.926x`；再把main numeric report算入只作更宽包络，不伪称逐行职责等价。两个report view dataclass保留的理由是阻止frontend导入neutral并把validity/provenance/order显式冻结；删除它们会恢复跨包dict/shape猜测。旧calibration report owner已在前一dependency-closed切片删除；legacy `live.py` grid/saved viewer仍有其它真实consumer，本轮既不适配也不提前删除。

W4b提交时tracked W4 oracle为`13 passed`：真实5×7/35-site、3-model calibration做paired reopen；逐array对齐、有限invalid filler、unusable NaN threshold、equal quick/formal但gate关闭、四种非方形GridOrder、aggregate source+render budget、Matplotlib lane release、worker/Qt thread ownership、五页atomic present、关闭后Experiment可继续使用，以及全部W4a regression均通过。另有owned-file compile、diff-check、fresh-process headless import、frontend reverse-import与shape-loss机械扫描；一次有界两路对抗审查最初发现5个P1并逐项关闭，限定复核为`P0=0/P1=0/GO`。

**历史实现 checkpoint：W4c current OccupancyArtifact frozen Figure/Workbench纵切完成。** `.figure_document/.figure/.figure_gui(occupancy_ref, occupancy_output=...)`只在composition root分派neutral-owned artifact；`None/"occupied"`选择分类块，`"counts"`选择计数块，非occupancy source携带该参数在repository access前拒绝。metadata-only projector调用`OccupancyRepository.inspect_final()`取得owner编码的exact schema；materialized路径才在总预算内完整admit并把所选artifact OwnedSnapshot直接交给既有FigureEvaluator。所选block保持原始`(R,P,SITE)`、PointLayout、AxisId、dtype/unit、ComponentValidity、revision、generation与artifact lineage；未选择block在求值前释放，不建立paired view DTO、伪COMPONENT轴或第三份dataset。occupied无声明x轴时默认SITE-faceted METER并把repeat mean明确留在display ViewSpec；有x轴时为CURVE；counts复用已有HISTOGRAM/CURVE。SITE不自动reduce，bool histogram固定两个categorical bins，Figure/GUI仍是DISPLAY ONLY，不能成为Fit/Scan authority。

Repository只新增一个有当前metadata-only consumer的compact inspection值；共享schema validator消除resolved-stream/artifact两处同一domain检查，frontend/workbench没有新增window、executor、renderer、selector或通用框架。相对parent `7e6ff62`的六个实际受影响production Python文件机械计数为`5788 -> 5966 physical`、`5193 -> 5361 nonblank`、`5182 -> 5348 NCLOC`，净`+178/+168/+166`；classes`28 -> 29`、dataclasses`19 -> 20`、enums`0 -> 0`、functions/methods`268 -> 271`，整体`1.031x/1.032x/1.032x`。main旧`frontend/figure_viewer.py`的saved-artifact广义包络为`965 physical / 877 nonblank / 691 NCLOC`，本轮净增量为其`0.184x/0.192x/0.240x`，只作海拔包络而不伪称职责等价。唯一新增dataclass用于让headless document读取FINAL schema而不materialize两条大依赖；删除它会迫使notebook复制repository metadata codec或完整admit，因此保留。

current tracked W4 oracle为`18 passed`：真实DetectionRequest提交/reopen occupancy；metadata-only零array/dependency admission；counts/occupied exact schema与snapshot lineage；`(R,P,SITE)`/site validity保形；默认METER、显式counts histogram、bool false/true分类；invalid output早拒绝；总预算与必然超预算时dependency零admit；GUI forwarding、worker/Qt ownership及全部W4a/W4b regression。owned files另通过compile、diff-check、fresh-process headless import、frontend reverse-import和shape-loss机械扫描；一次有界对抗审查发现“已知必超预算却先admit dependencies”的P1，early peak gate与fault-injection oracle关闭后复核为`P0=0/P1=0/GO`。没有dependency-closed旧删除：legacy live/gridplot/selector/rolling仍是真实消费者，不能因冻结viewer可用而提前删除。下一纵切从当前最后consumer矩阵选择paired calibration+occupancy的physical site grid/selector或MonitorDataset rolling产品边界；必须先确认哪个能形成更小完整public consumer，不重开W4a/W4b/W4c审查。

**历史实现 checkpoint：W4d exact physical occupancy cell + same-shot source frame纵切完成。** public seam为`Experiment.readout.occupancy_cell_gui(OccupancyArtifactRef, selection=..., memory_limit_bytes=...)`。具名repeat/point轴中只有singleton可自动选0，其余必须给exact `IndexSelection`；range、未知轴、缺失非单例轴和sparse `PointLayout` hole都fail closed。解析出的同一个`DatasetCellAddress`同时索引`occupied`/逐SITE ComponentValidity与`CaptureFrameSource.read(address)`；source与occupancy的repeat/point axes、PointLayout、revision、generation、model、source/calibration refs全部重验。底层完整`(R,P,SITE)` artifact不变，frame只做一次exact read，禁止whole-capture materialize；Calibration只加载已提交SiteMap artifact取得centers/validity，不加载report或重跑物理算法。frontend view只含本图实际消费的frame/pixel validity、centers、occupied/site validity与可见lineage摘要；counts继续由W4c显示，不为未来交互携带site labels/counts等零消费者字段。

该surface复用shared `FrozenRasterWindow`与唯一capacity-one lane：worker完成全部repository读取、三态overlay与Agg/PNG，Qt owner只原子present immutable PNG；这只记录W4d当时的呈现机制，不再允许把`EXACT OCCUPANCY CELL · SAME-SHOT FRAME · DISPLAY ONLY`当永久产品上限。empty、occupied、invalid分别独立显示，invalid绝不被False filler冒充empty；source artifact保持immutable，但通用viewer/overlay切片必须按规则9补适用的zoom/pan/selector/export。总预算先保留occupancy/capture/calibration三个inspection的owner-reported retained bound与整个view peak，再把唯一余额逐阶段交给occupancy admit、calibration load和capture admit/read；任一组合超限在大数组admit前拒绝。一次有界对抗审查发现并关闭两项P1：旧最近邻半径的`O(S²)`临时矩阵改为固定128×128复用scratch，逐site `Circle`改为最多三个`EllipseCollection`，共享半径几何归`matplotlib_render`而非让calibration依赖occupancy产品。实测到2048 sites时半径scratch峰值约544 KiB（预算1 MiB），collection完整save的线性峰值斜率约16 B/site、样本极值包络低于65 B/site（预算512 B/site），因此预算有证据且不保留二次爆炸。

Rule-6相对parent `24a3ea6`按七个实际受影响production Python文件（notebook facade、workbench lazy入口/薄occupancy adapter、occupancy/calibration/matplotlib renderer、occupancy repository）机械计数：`5094 -> 5865 physical`、`4632 -> 5341 nonblank`、`4621 -> 5328 NCLOC`，净`+771/+709/+707`；classes`17 -> 18`、dataclasses`9 -> 10`、enums`0 -> 0`、functions/methods`201 -> 215`，整体`1.151x/1.153x/1.153x`。main中与这张静态物理图相关的`LiveSiteMap + site_map + plot_occupancy/plot_detection_image + TaskConsole site inputs/aux`窄包络为`294 physical / 242 NCLOC`；W4d净增量为`2.622x/2.921x`，仍低于约3倍。额外代码只来自具名exact selection/sparse layout、同shot lineage、ComponentValidity、跨repository单总预算和worker/Qt ownership；没有新enum、repository、manager、scheduler、selector framework或compatibility wrapper，唯一新view dataclass用于阻止frontend反向import neutral并冻结跨线程owned facts。

current完整W4白名单为`24 passed`，其中W4d六项覆盖具名多轴+sparse layout解析、同address frame/occupancy读取且禁止capture materialize、invalid三态、跨128-site block最近邻、三inspection聚合紧预算在admit前拒绝、public GUI worker/Qt ownership与关闭后Experiment可用。另有owned-file compile、diff-check、fresh headless import（Qt/pyplot/SciPy未加载）、frontend反向import、shape-loss与逐site Circle机械扫描；同一限定对抗复核最终`P0=0/P1=0/GO`。本轮没有dependency-closed旧删除：legacy live grid、交互selector/ROI、rolling及其它TaskConsole消费者仍真实存在。下一checkpoint优先在W4d exact-address基础上检查“冻结cell导航+selector”能否形成一个不引入live scheduler的最小完整纵切；rolling继续按MonitorDataset的M1 camera monitor、M2 ROI/scalar/hist/multipanel独立迁移，不重开W4a-W4d。

**历史实现 checkpoint：W4e frozen exact-cell navigation纵切完成。** 同一个`.readout.occupancy_cell_gui()`先通过metadata-only loader冻结content-addressed artifact target、occupied schema fingerprint、generation、repeat/point axes、`PointLayout`和schema-owned `cell_layout`，再建立按轴数有界的index/coordinate控件。非singleton从`Select…`开始且不发cell job；singleton才自动0。显式Load和Previous/Next都生成覆盖全部outer axes的canonical `IndexSelection`，导航严格沿`cell_layout` physical storage order、跨repeat边界且跳过EXPLICIT hole，不把SITE轴当cell导航或reduce。每次exact loader重新比较artifact/schema/generation identity并继续执行W4d同一`DatasetCellAddress`、source revision、lineage、ComponentValidity和总预算合同。

窗口仍复用shared `FrozenRasterWindow`与唯一capacity-one executor，没有第二scheduler：每窗口只有一个active Future和一个可覆盖latest pending Selection，front/label/status在同一Qt-owner transaction接受；request revision、identity或Selection任一不符的success/error/cancellation都丢弃。旧front在新job前释放；pending只在旧Future/PNG退出当前owner stack后的下一次queued turn启动，避免两张大图生命周期重叠。Close清pending/front并立即返回，running repository/Agg诚实排空但不能stale present；关闭时同时释放bound loaders。鼠标selector/ROI、ViewportTransform、zoom/export、rolling/live source与fit/control authority都没有当前冻结consumer，因此明确未实现，也没有借“导航”预建overlay框架。

Rule-6相对parent `5e0c01b`按五个受影响production文件（notebook facade、workbench lazy入口、shared frozen shell、occupancy window、occupancy renderer/navigation value）机械计数：`3485 -> 4176 physical`、`3140 -> 3769 nonblank`、`3136 -> 3765 token-NCLOC`，净`+691/+629/+629`；classes`13 -> 15`、dataclasses`6 -> 7`、enums`0 -> 0`、functions/methods`137 -> 182`，整体`1.198x/1.200x/1.201x`。两个新类分别是有真实窗口consumer的`OccupancyCellWindow`和阻止frontend持有neutral ref/repository的`OccupancyCellNavigation`；没有新enum、repository、executor、renderer、manager、selector framework或compatibility shim。main中旧Grid focus/navigation加TaskConsole host的审查包络为`295 physical / 243 token-NCLOC`，W4e净增量为`2.342x/2.588x`，低于约3倍；额外复杂度只来自具名多轴+sparse physical order、content identity、聚合预算、latest-only并发与非阻塞close。

current完整W4白名单为`28 passed`：除W4a-W4d回归外，W4e覆盖metadata-only零array admission与stale identity早拒绝、非singleton不隐式首帧、sparse physical navigation、active+latest pending coalescing、stale failure不覆盖latest、同一board复用以及inflight close零stale present。owned-file compile/diff-check、frontend反向import、shape-loss、第二executor与selector/ROI机械扫描通过；一次有界独立终审为`P0=0/P1=0/GO`。本轮没有dependency-closed旧删除：legacy live grid、交互selector/ROI、rolling及其它TaskConsole消费者仍真实存在。下一checkpoint进入MonitorDataset rolling的M1 camera monitor真实纵切；ROI/scalar/hist/multipanel与selector留在有真实下游consumer的M2，不重开W4a-W4e。

**最新实现 checkpoint：U0.2e current camera ROI scalar Histogram H1纵切完成，状态为`MATCHED_LIVE_ROI_SCALAR`。** 已有typed ROI scalar `MonitorDataset`仍是唯一数据源；Figure evaluator按声明的SAMPLE轴逐项保留坐标，只过滤完整`ComponentValidity=false`并携`dropped_count/value_unit`。`HistogramDisplayState`唯一拥有bins、linear/log、count relim和x pin；`HistogramViewportTransform`只冻结draw后映射与authored-state标记。computed-only `HistogramBinProjection`直接绑定exact immutable samples并一次生成共享counts/edges，Agg raster与`HistogramPanelPayload`消费同一对象，payload再次检查sample identity、bin count和总样本数，不能注入伪counts或二次binning。底层`(R,P,*data_shape)`从未flatten、按rank猜轴、取第0项或隐式reduce。

Qt在已有`QtRasterBoard`中让Curve/Histogram共用一个数值panel interaction生命周期，但二者继续拥有不同typed state/viewport/intent。Histogram恢复A/C/Z/H、wheel x zoom、middle x pan、middle-double area/home、真实bin hover、shared bins、三态count relim、linear/log、x pin与同源Setting/Edit；gesture只hold本panel，source和siblings继续。退化left click现在显式clear area，最终bin tooltip严格显示右闭；same revision只准data-derived auto范围改变，bins/scale/relim/x-pin/fixed count漂移均fail closed。所有交互仍为DISPLAY ONLY，绝不进入Fit/Scan/Calibration authority；threshold、Gaussian/双峰fit、fidelity、generic Distribution/Grid与save/reopen明确留给有真实物理consumer的H2/后续切片。

profile显示500-bin projection median/p95约`0.0411/0.0521 ms`，800×520 worker Agg完整render约`41.396/50.201 ms`，四panel Qt整板present约`1.555/2.009 ms`；瓶颈仍是worker Agg而非binning/GUI，因此没有增加renderer或executor。Rule-6对12个production owner为`13956 -> 16079 physical`、`12975 -> 14960 nonblank`（约`1.152x/1.153x`），相对main 836/775行严格Histogram+selector窄包络的净增为`2.54x/2.56x`；零约3倍单文件、零manager/plugin/registry/wrapper/第二stream/dataset/renderer，零consumer binning wrapper/exports、binding helper、test-only property及旧Curve/Histogram测试seam也在提交前删除。两路独立对抗审查在关闭三项P1及最终P2后均为`P0=0/P1=0/GO`。活动manifest收集100/100文件、1280项，逐文件新进程实跑`1277 passed + 3 expected skipped`、零失败/错误。本checkpoint只判H1 current consumer为GO，完整迁移仍为NO-GO。

**历史实现 checkpoint：U0.2d exact interactive Sites纵切完成，状态仅为`MATCHED_EXACT_VIEWER`。** `.readout.occupancy_cell_gui()`保留W4d/e的metadata-only导航与同一个exact `DatasetCellAddress`，但把旧Agg/PNG冻结报告面替换为直接`QWidget + QtRasterBoard`。worker一次构造包含background、frame pixel validity、centers、严格bool occupied、SITE ComponentValidity、两条exact input refs、calibration/cell identity和缓存geometry/join digest的immutable `SiteMapPanelPayload`；Qt owner只建立INDEXED8/QImage和三态rings。返回view另携canonical exact `cell_selection`，worker用同一个navigation owner反解请求并比较后才stamp/present；相邻cell返回、ref/revision/calibration错位、整数/float/NaN被当bool与超出declared/admitted bound全部fail closed。底层occupancy和frame始终保持原`(R,P,*data_shape)`与各自ComponentValidity，没有flatten、rank猜轴、取第0项或权威reduce。

用户面恢复origin-upper、physical equal aspect/向西锚定、A/C/Z/H、nearest-site状态/像素hover、六种cmap、tight/normal/fixed clim、同一revisioned display state的Setting/Edit、panel-local hold和latest-only导航；换cell会提升selection revision并清旧hold，同selection的数据revision更新不会改写held front。矩形只保存可见normalized `DISPLAY_ONLY` bounds，SiteMap authority转换始终抛错。旧`render_occupancy_cell/_build_site_map`、occupancy Matplotlib/Agg/PNG与零consumer viewport wrappers已物理删除；没有compatibility alias。语义中性的capacity-one worker/launcher/load/export owner统一在`_window_runtime`，报告用`FrozenRasterWindow`仍只负责真实encoded-raster consumer。本checkpoint不声称live Sites完成；live必须等current product port原子发布same-event frame+occupied+calibration identity，不能拿W3 counts或latest/latest拼接。main旧`frontend/live.py::site_ring_radius`仍由未迁移live Sites消费且算法只看相邻center；live Sites切片必须直接改用`zlc_frontend.site_map`唯一all-pairs owner并物理删除旧函数，不允许保留delegate alias。

聚焦oracle为Sites `32 passed`，并逐文件新进程通过W4 `24`、W5 `8`、W6 `7`、W7 `7`项直接回归；syntax、diff-check、headless import、旧symbol扫描与offscreen性能证据通过。活动manifest共98个文件，最终collect为`1268`项，逐文件新进程实跑`1265 passed + 3 expected skipped`、零失败/错误。两路独立数据/生命周期审查最初抓到exact selection seam和retained-bound cap两项P1，修复与公开window反例后复核为`P0=0/P1=0/GO`；独立Rule-6终审在继续删除test-only accessors、死state和重复array owner后为`P0=0/P1=0/P2=0/GO`。该历史checkpoint只对exact Sites判`GO`；finite capture后来由U0.2f闭合，但live paired Sites、Histogram、Grid与完整launcher仍不能从Sites外推为完成。

**M1 基线 checkpoint（其capacity-one可见历史由下述M2a当前checkpoint覆盖）：free-running raw camera monitor纵切完成。** public seam为`.readout.camera_monitor_request()`、`.inspect_camera_monitor()`与`.camera_monitor_gui()`；前两者只冻结/检查typed role、working point、输出schema、总内存与I/O deadline，不arm设备。GUI只获得one-shot `PreparedCameraMonitor` callable，不保存Experiment、Port、RunPlan、resolver或raw camera。每次Start先在同一总cap内合计driver ring、adapter records、stream/tap retention、mutable/frozen dataset、metadata、view evaluation、GRAY8 raster与Qt front，然后RunController才取得排他camera claim并arm。固定数据面为`FREE_RUNNING camera -> AcquisitionStream -> capacity-one MonitorTap -> capacity-one MonitorDataset -> LiveDatasetSlot`；每次restart使用新RunId、stream generation与BlockId，绝不把上次终态当本次latest。

MonitorDataset原样冻结`(R=1, MONITOR_HISTORY=1, *data_shape)`，当前camera产品的trailing axes为具名`SPATIAL_Y, SPATIAL_X`；BlockId、revision、head、EventRefs、ComponentValidity与MonitorCoverage与一张immutable front同步接受。IMAGE view只对已声明的singleton `MONITOR_HISTORY` AxisId做`IndexSelection(0)`，没有flatten、`[:,0]`、shape/rank猜测或data-axis reduce；missed/current-gap在面板可见。该路径是DISPLAY ONLY：不存在ExactReservation、seal、artifact、Fit/Save或Formal capture authority。normal Stop与failure都先撤销source publication再清front，已排队worker不能复活旧图；Close先cancel同一RunHandle，只在terminal、bounded hardware cleanup与owner join证明后释放live surface。外部interrupt只能通过typed `CameraMonitorInterrupted`转成用户cancellation，普通source/validation error不能被洗成CANCELLED。

virtual monitor不是随机moving blob：它与main `VirtualMotCamera`的物理与数值语义为oracle，从同一VirtualSequencer已编译/fired的TargetIR DAC code感知`da_x/da_y/da_z`，在safe state感知零场，以同一optimum/sigma、MOT Gaussian效率、40×20 FWHM spot、offset/peak/read-noise生成帧。曝光由sensor-owned clock驱动，host/GUI不sleep调度；有界record queue满即source-fatal，绝不覆盖未消费帧。pulse-owned `sample_compiled_bus_codes()`只读compiled TargetIR/scan row/bus delay，样点落在live ramp内时拒绝而不另造近似waveform。M1没有free-running frame↔scan-row硬件关联，因此virtual sensor只接受single-point compiled output；multi-point明确失败，绝不按camera ordinal轮转猜点。real Pylon monitor仍为明确NO-GO：legacy `LatestImageOnly`可在broker前跳帧，必须先由adapter暴露missed/skipped事实并通过free-running contract kit，不得因virtual READY被默认开放。RTL/bitstream未修改。

大帧profile又关闭了一个不能藏的memory-admission漏洞：默认`1200×1920 Mono8`单帧只有`2,304,000 B`，但一次全帧float64 normal+clip生成的`tracemalloc peak` 为`39,298,094 B (17.06x frame)`，已超过descriptor宣称的`27,652,579 B base peak`。根因是virtual adapter临时数组，不是stream/ring。整改后噪声按不超过一帧的固定上界分块，且queue已满时在render前source-fatal；同一机器/参数的单帧峰值降为`2,966,542 B (1.29x frame)`，耗时从约`38.85 ms`降到`30.81 ms`。这使producer输出+临时scratch回到已准入的driver/adapter frame包络内，没有靠抬高默认内存限额掩盖根因。

M1与W1共享的只是一个窄`QtRunOwnerMailbox`：bounded executor、Future/attachment mailbox、snapshot coalescing、generation+RunHandle与terminal reap latch有两个真实consumer；QWidget、prepare/view/status、FINAL保留与monitor撤屏/重启策略仍由产品controller拥有，没有GenericRunPanel或可配置workflow window。主审又锁死两个owner真值：RunHandle产生前start失败显式标记已回收；`wait()`异常绝不反向标记已回收。因此restart/close不能在owner thread仍存活时放行。

Rule-6以parent `14690997d64ad5d254a4266be78675d735abbdbe`的15个实际受影响production Python文件机械计数：`7653 -> 10200 physical`、`7038 -> 9363 nonblank`、`7005 -> 9311 token-NCLOC`，净`+2547/+2325/+2306`；classes`31 -> 51`、dataclasses`16 -> 28`、enums`0 -> 0`、functions/methods`283 -> 399`。main的`CameraDevice + VirtualMotCamera + CameraMeasurement + Live2DDis`实际物理/用户面包络为`1192 physical / 1079 nonblank / 946 token-NCLOC / 4 classes / 76 functions`；本切片全部净增量分别为其`2.14x/2.16x/2.44x`，低于约3倍门。新12个dataclass只用于跨线程/设备命令-ack、capability、request/view准入和Run prepared state；删除任一个都会退回dict/string阶段猜测。其余新class分别有endpoint、virtual camera、Workbench、live authority或W1+M1双consumer，没有single-member enum、manager、plugin、compatibility shim或单consumer通用框架。增量相对main还承担其没有的typed capability/session cleanup、有界stream/materializer、immutable front、聚合内存准入和非阻塞Qt owner，不是重写物理数学。

当前白名单必须按process-lifetime installation边界分进程运行：M1 `5 passed`，共享owner直接影响的W1回归`7 passed`。owned production/test `py_compile`、`git diff --check`、neutral反向import、Workbench raw authority/第二executor与production新增shape-loss扫描通过。本轮没有dependency-closed旧删除：legacy camera frame producer仍有ROI/readout/rolling等未迁最后consumer，现在删除会人为制造断档。下一checkpoint是M2的真实consumer纵切（ROI/scalar/history、histogram、multipanel coherence与selector按最小完整产品分批）；不重开W4或M1，也不因GUI未完而疯狂删除仍有consumer的旧producer。

**M2a历史 checkpoint（其raw-history display reduction已由下述M2b替代）：typed ROI short-history coherent board纵切完成。** `.camera_monitor_request(history_capacity=H)`把raw可见历史冻结为一个有界、dense、newest-first `MONITOR_HISTORY` axis，默认`H=8`；schema唯一真值是`(R=1, P=H, *data_shape)`，当前camera仍为`(1,H,SPATIAL_Y,SPATIAL_X)`，slot 0明确是最新帧。history使用`AxisSpec`已有的`index_origin=0`隐式坐标，不在准入前物化`tuple(range(H))`；任意巨大H都先经O(1) schema和纯整数base-peak预算拒绝，不能先制造O(H) Python对象再报告超限。descriptor直接拥有同一个immutable `DatasetSchema`，shape/fingerprint只作派生只读属性，不再保存可能漂移的镜像。`.camera_monitor_gui(..., roi=typed Selection, roi_reduction=MEAN|SUM)`只接受覆盖具名`SPATIAL_X/SPATIAL_Y`的显式rectangle；裸字符串reducer、非空间selection、额外未声明轴或`H<2`都在arm前拒绝。IMAGE只选择history slot 0；CURVE把history作为X、repeat显式选择0、且只在selection约束后的两条空间轴上执行可见的MEAN/SUM。不存在rank/singleton猜测、flatten、隐式data-axis平均或`reshape(...)[0]`。

IMAGE和ROI CURVE由同一个`MonitorDatasetSnapshot`顺序evaluate；一个`CoherenceStamp`冻结完整DatasetRevisionRef与两个presentation identity，一个`BoardFrame`同时携带两个raster，Qt `QtRasterBoard.present()`先验证完整panel顺序并准备全部QImage，再一次交换front。ROI与reducer始终是`ViewSpec`中的DISPLAY ONLY意图，不产生`DataTransformSpec`、CommittedTransform、processor signal、scan y、fit或artifact authority；原始`DataBlock`、EventRefs、ComponentValidity与revision都原样保留。当前短曲线是“按age slot查看最近H张raw frame”的真实但受限产品，不冒充基于timestamp的长标量历史。

总内存仍在设备arm前一次准入：除了driver ring、adapter record、capacity-one stream/tap、mutable raw window与immutable snapshot，还显式计入H>1 newest-first物化时values/validity/written mask的整窗reorder scratch、每个history cell的camera metadata、IMAGE/CURVE evaluator、GRAY8 scratch/当前与上一front、persistent Agg canvas/artist arrays及当前与上一curve front。source/render仍latest-only且同一时刻只冻结一份snapshot。相机Workbench复用原有`QtRunOwnerMailbox`，将该窗口收窄为一个worker的串行render lane；mailbox从真实`max_workers==1`派生可检查的thread-affine事实，companion Agg controller在构造时强制要求该事实，mutex不再冒充线程亲和。image-only `zlc_workbench.live`保持headless且不预加载Matplotlib；M2a当时只有真实curve job才惰性取得曲线专用renderer，该符号已由M2c无alias地原位泛化为`SinglePanelAggRenderer`，并继续在同一lane首次构造、重复更新及经同一serialization gate关闭。没有第二executor、共享Figure或GUI-thread compose。Stop/restart/Close继续撤销publish port、清完整board front并排空renderer job，旧generation不能复活。

M2a刻意没有预建鼠标selector、ViewportTransform、revisioned ControlTopic、长scalar derived `MonitorDataset`、meter、histogram、动态board layout或Formal scan consumer。下一真实切片若需要数百点rolling，必须从有序raw source显式产生带ROI/control revision与ComponentValidity的scalar event，再用独立小型history保存；不能把数百张大图留下，也不能让UI `np.roll`成为第二数据真相。histogram只消费这条scalar history；selection要进入scan y时另经权威processor/terminal ack边界，绝不能把本次display reduction升级。real Pylon free-running仍因legacy `LatestImageOnly`无法报告skip而NO-GO；本切片没有改RTL/bitstream，也没有用软件替代任何硬件时序保证。

Rule-6以parent `227c8c57288ccdc6e6751874fed331360b537067`的九个实际受影响production Python文件机械计数：`7300 -> 7712 physical`、`6569 -> 6956 nonblank`、`6430 -> 6809 token-NCLOC`，净`+412/+387/+379`；classes`44 -> 45`、dataclasses`18 -> 18`、enums`0 -> 0`、functions/methods`366 -> 381`。唯一新class是有真实双panel consumer的`QtRasterBoard`；没有新dataclass、enum、repository、manager、scheduler、processor framework或compatibility wrapper。main中rolling/hist/selector/ROI窄kernel包络为`1851 physical / 1331 token-NCLOC`，本切片净增量分别为`0.223x/0.285x`；含真实board host的宽包络为`3747/2512`，对应`0.110x/0.151x`。新增量只支付explicit typed ROI、H-window准入、同snapshot双panel coherence、Agg线程归属与Qt原子front；当时既有FigureEvaluator、曲线专用renderer、BoardController、MonitorDataset和Run owner均直接复用，其中曲线专用renderer已在M2c被generic single-panel owner无alias替换。

当前限定回归按process-lifetime installation边界分进程运行：M2a `3 passed`、M1 `5 passed`、直接受影响W1 `7 passed`；owned `py_compile`、`git diff --check`、shape-loss/authority/第二executor/Qt反向import机械扫描通过。整份import-DAG ratchet当前另有三个不属于本切片的既存dirty-tree失败（旧canonical helper、表单input `.strip()` allowlist与两个带编辑计数的格式名），通用dataset旧测试也因仍导入已删除的`DatasetCellDomain`而在collection失败；M2a的owned边界没有新增violation，也不为迎合它们修改旧域。一次有界独立终审先发现“预算前O(H)坐标分配”和“串行锁不等于Agg线程亲和”两个P1；上述implicit coordinate与proven affine lane整改后由同一审查员定向复核为`READY/P0=0/P1=0`。本轮没有dependency-closed旧删除：legacy camera producer、selector/ROI、histogram/grid与TaskConsole仍各有未迁consumer；它们只能在各自最后consumer迁完时删除。

**M2b历史 checkpoint：typed ROI scalar event + independent long history完成。** public request现在一次冻结raw `history_capacity`（默认8）、显式rectangle `Selection`、`MEAN|SUM`、`ValidityPolicy`与独立`scalar_history_capacity`（默认300）；有request时`.camera_monitor_gui(request, ...)`拒绝任何覆盖option，无request时所有option只经`.camera_monitor_request()`构造，inspect、prepare与初始UI topology消费同一个immutable request。当前ROI binding只接受恰好一根具名`SPATIAL_Y`和一根具名`SPATIAL_X`及两条连续range；其它或额外`data_shape`轴在arm前拒绝，绝不按rank/singleton猜角色。raw schema始终保留`(R=1, P=Hraw, *data_shape)`；只有这个显式物理reducer消费声明的两条空间轴后，derived scalar schema才成为`(R=1, P=Hscalar)`，因此不是把任意多维`data_dim`重排成三项array，也没有`flatten`、`[:,0]`、隐式trailing mean或display-to-authority升级。

运行数据面只有一次相机read与一次raw publish：`AcquisitionStream`的第一条bounded tap进入raw `MonitorDataset`，第二条独占tap由同一acquisition lane上的窄`RoiScalarStreamProjection`按source顺序同步消费，再发布带direct causation的新scalar event到独立stream/tap/`MonitorDataset`。它不是通用async processor engine，也没有第二线程、scheduler或UI `np.roll`。M2b当时的每条scalar metadata冻结完整source `EventRef`、camera metadata、source missed、常量control revision与binding fingerprint；TraceContext继承run/correlation并把source ref列为直接因果。M2d证明该常量revision不表达任何真实mutation后已将其删除，只保留完整semantic binding fingerprint。一个application-owned `RLock`覆盖raw ingest、ROI计算、scalar publish/ingest及raw+scalar materialize；`CameraMonitorSnapshot`再验证scalar head的source ref就是当前raw head，所以Workbench不能分别读取两个“latest”后拼出错位panel。

ROI validity语义不再隐藏：默认`REQUIRE_ALL`，任一selected component无效时结果value仍按有效contributor确定计算但scalar validity为invalid；确需忽略坏component时必须显式选择`OMIT_INVALID`，它进入request、binding fingerprint和面板摘要；当前没有真实consumer的`MIN_COUNT`直接拒绝，不预建阈值机器。两种policy都只在具名ComponentValidity为真的component上计算；全invalid输出typed invalid zero。SUM使用canonical checked accumulator，MEAN按有效contributor数除；浮点finite检查、partial-valid safe copy、必要widened accumulator和ROI-size bool临时量都进入pre-arm scratch上界，不能以“scalar很小”漏算处理峰值。

Workbench给raw IMAGE和scalar CURVE分配不同DatasetId，并在同一`CoherenceStamp`中冻结两个精确DatasetRevisionRef；M2b当时的join digest由scalar metadata中的完整source EventRef、常量control revision与fingerprint导出，M2d已把它收敛为`source EventRef + semantic binding fingerprint`，不再伪造revision。两个FigureDocument各自只能evaluate自己的已准入revision，随后仍由一个thread-affine worker完整compose、一个`BoardFrame`与Qt whole-board front原子present。physical ROI reduction是typed `MONITOR DERIVED`事件，CURVE只是其`DISPLAY VIEW`；它既不是M2a的display reduction，也不是Fit/Scan/Save artifact authority。normal Stop的clean withdrawal只撤front/port/coverage，不再制造`RuntimeError(None)`或短暂显示假FAILED；真实source failure才写fault。

总内存继续在arm前一次合计：raw driver/adapter/stream/tap/window/snapshot/reorder/metadata、processor的第二raw tap、scalar stream/tap/payload、独立mutable/snapshot/reorder/每cell metadata、ROI reduction scratch，以及IMAGE/CURVE evaluate/raster/Qt front均在同一cap内。增大Hscalar不会增大raw schema；过大Hscalar在设备prepare/arm前拒绝。没有改RTL/bitstream，没有host-stepped scan或软件时序替代；real Pylon free-running仍受现有qualification gate约束。

M2b刻意没有预建dynamic selector、revisioned ControlTopic/terminal ACK、history reset/generation、histogram、METER、Formal scan-y、Fit/Save或real qCMOS路径。它列出的第一个下一候选“让真实histogram/meter消费现有scalar history”已由下述M2c完成；M2d随后选择selection/reducer改变时重建整个Run/stream/block/history generation。这个选择只保留为历史因果，规则9与`UX-001`已经明确否决其“更诚实/更小所以可作终态”的结论：monitor retarget必须保持source Run/raw generation/tap topology连续，由downstream `ControlTopic`在事务边界APPLIED。selection若进入权威scan/fit仍必须重新构造对应authority spec。legacy `RoiProcessor`、`LiveLive/LiveLiveDis`、TaskConsole rolling/histogram与camera producer还拥有crop/scatter/value ROI、raw-only首ROI、side histogram/fit、scan-y及其它未迁consumer，本轮不提前删除；dependency-closed删除的只是current M2a raw-history display-reduction路径，不保留第二套实现。

M2b Rule-6按六个实际受影响production文件相对`f19bedb`机械计数为`4905 -> 6013 physical`、`4475 -> 5500 nonblank`、`4455 -> 5474 token-NCLOC`，净`+1108/+1025/+1019`；classes`20 -> 29`、dataclasses`9 -> 16`、enums`0 -> 0`、functions/methods`197 -> 242`。main真实窄oracle `RoiProcessor + LiveLive`为`314/278/254 physical/nonblank/NCLOC`，本轮净增为其`3.53x/3.69x/4.01x`，严格海拔超过约3倍，不能隐藏；加入默认`LiveLiveDis`与真实`PanelCard._render`的宽用户面包络为`596/543/438`，对应`1.86x/1.89x/2.33x`。超过窄oracle的增量承担main没有的typed validity/control binding、source-event lineage、payload/metadata容量与digest、独立scalar history、raw/scalar原子coherence和非阻塞Qt生命周期。审查后`RoiScalarSample`直接拥有`RoiScalarMetadata`，零production-consumer的package-root re-export也已删除；剩余binding、stream contract/adapter、ordered projection、atomic live dataset与presentation owner分别有不同generic接缝或生命周期，未发现可再压缩的P1、single-member enum、manager、plugin、scheduler framework或compatibility wrapper。

当前限定回归按process-lifetime installation边界分进程运行：M2b `3 passed`、M2a current回归`3 passed`、M1 `5 passed`、直接受影响W1 `7 passed`；fresh import、owned production/test `py_compile`、`git diff --check`与新增shape-loss扫描通过。两路独立对抗审查最初发现四个P1：未声明validity policy、finite bool scratch漏算、冻结request可被GUI option覆盖、clean withdrawal伪造failure；逐项整改后原审查员定向复核均为`READY/P0=0/P1=0`。完整迁移仍为NO-GO；这里只判M2b纵切GO，不重开M1/M2a，也不运行/修改历史架构测试去迎合已迁移边界。

**最新实现 checkpoint：M2c fixed ROI scalar histogram/meter coherent board完成。** M2b唯一的`RoiScalar MonitorDataset`没有增加第二stream、dataset、buffer或processor；同一个scalar `MonitorDatasetSnapshot`及其精确`DatasetRevisionRef`同时驱动CURVE、HISTOGRAM和METER，raw snapshot只驱动IMAGE。一个`CoherenceStamp`仍只有raw/scalar两个`EvaluatedInput`，却冻结四个presentation及`(raw, scalar, scalar, scalar)` panel source；一个`BoardFrame`携带四张raster并由Qt whole-board一次present，不能逐panel拼接各自的latest。

HISTOGRAM显式选择repeat 0并把`MONITOR_HISTORY`声明为SAMPLE，只消费`ComponentValidity=true`的retained scalar；frontend render owner唯一导出的`DEFAULT_HISTOGRAM_BINS=60`同时服务静态图、persistent live renderer、pre-arm估算与UI摘要。`dropped_count`与`MonitorCoverage`共同区分invalid retained和unfilled，空样本产生零counts而不伪造一个值为0的样本，NumPy最后右边界规则原样保留；valid NaN/Inf对CURVE/HISTOGRAM/METER全部fail-closed，不用`isfinite`偷偷改validity。METER严格读取newest slot 0；该slot invalid就显示`invalid`，绝不回退上一有效值。

曲线专用renderer已无alias地原位泛化为`SinglePanelAggRenderer`；CURVE/HISTOGRAM/METER分别在同一个thread-affine worker lane内构造、更新和close持久artist，W3只机械改用该泛化owner，没有第二executor、共享Figure或GUI-thread compose。arm前总预算合计IMAGE及三个scalar evaluator、三份persistent Agg/canvas/artist、histogram counts/edges/vertices、candidate/front和Qt fronts，且`Hscalar > FigureEvaluationPolicy.max_histogram_samples`时在设备arm前拒绝。raw/scalar front之外的诊断也不再是几次独立“latest”读取：一个frozen `LiveFrontStatus`把board sequence、raw/scalar coverage、histogram valid/dropped、latest validity与source missed原子绑定；Qt只在它与可见`front_frame.sequence`相等时显示。owner先present已完成整板再admit下一候选，publish早于status安装的wake竞态由安装后第二次coalesced wake闭合。

M2c仍不实现dynamic selector、ControlTopic/terminal ACK、history generation reset、fit/scan-y/save、threshold/Gaussian side-fit、动态layout或real qCMOS。dependency-closed删除仅是current的curve-only renderer特殊名与双panel topology；legacy `LiveLive/LiveLiveDis`、generic `HistogramFigure`/Grid/`PanelCard._render`、`RoiProcessor`和camera producer仍有dynamic selector、side histogram/fit、generic TaskConsole及scan/readout等真实最后consumer，本轮不删、不适配。

M2c Rule-6按最终六个受影响production文件（`_camera_monitor.py`、`_capture.py`、`matplotlib_render.py`、`qt_widgets/board.py`、`live.py`、`progressive_scan.py`）相对parent `70ab295`机械计数：`4058 -> 4372 physical`、`3750 -> 4051 nonblank`、`3730 -> 4030 token-NCLOC`，净`+314/+301/+300`；classes`13 -> 14`、dataclasses`3 -> 4`、functions/methods`162 -> 166`。main严格数值/artist kernel oracle为`54/53/49 physical/nonblank/NCLOC`，净增分别是`5.815x/5.679x/6.122x`，超过约3倍必须如实保留；加入真实rolling latest、histogram初始化/更新和`PanelCard._render` steady compose的宽用户面oracle为`253/243/170`，净增降为`1.241x/1.239x/1.765x`。严格分母没有class/dataclass，不能伪造倍率；宽分母2 classes/8 functions，对应净增`0.5x/0.5x`。唯一新增class/dataclass `LiveFrontStatus`正是关闭已实证跨revision诊断拼接P1所需的原子事实；删除它会恢复多字段latest读取。renderer是原class就地泛化并已有W3+M2c消费者，bin/vertices/meter helper有static+live双消费者；没有第二stream/dataset/executor、manager、scheduler或compatibility alias。

限定验证按process-lifetime installation边界分进程运行：M2c `6 passed`、M2b `3 passed`、M2a current回归`3 passed`、M1 `5 passed`、直接受影响W1 `7 passed`、W3 renderer回归`5 passed`，另有figure histogram/meter contract `4 passed`；owned production/test `py_compile`、fresh import、旧renderer/curve-only符号扫描与`git diff --check`通过。两路独立对抗审查共发现并关闭三个P1：CURVE/METER valid NaN/Inf未fail-closed、60-bin渲染与预算常量手抄漂移、可见front与多字段诊断跨revision拼接；原审查员定向复核均为`READY/P0=0/P1=0`。完整迁移仍为NO-GO；这里只判M2c纵切GO，不重开已完成产品，也不运行/修改历史架构测试去迎合已迁移边界。

**历史实现 checkpoint：M2d fixed-ROI front-bound selector完成；source Run replacement结论已废止。** 当时真实consumer只是在已有typed ROI四panel camera monitor中重新画rectangle并选择`MEAN|SUM|MAX`。`RectangleGesture`冻结`panel/board/layout/sequence/source/viewport revision`与normalized pixel-cell-edge bounds；`ImageViewportTransform`只接受恰好一对显式、规则、数值、`unit="pixel"`、同coordinate frame且role为`SPATIAL_X/SPATIAL_Y`的具名轴。它不看rank/singleton猜角色，也不近似irregular axis。Qt overlay只画accepted与draft rectangle；worker继续一次compose完整board，GUI不取得Figure/Agg。当时mouse press会暂停整个live admission；U0纠正2必须改成只暂缓/coalesce被拖panel的presentation，source与其它panel继续推进，release后从最新合法revision补拍，所有退出路径仍成对释放interaction state。

当时draft始终可见且不改变正在发布的scalar，只有显式`Apply ROI`才构造replacement request；这解释了旧实现的preflight/预算事实，不再定义目标UX。U0纠正1要求drag release（或旧oracle对应的直接动作）自动提交downstream revisioned ControlTopic，UI保持pending直到事务边界`APPLIED`，不再要求第二次`Apply ROI`。downstream owner仍须在应用前冻结candidate binding、schema、documents、histogram/evaluation policy与预算；可预知失败保留旧applied downstream binding与raw source。相机设置、exposure、物理acquisition ROI、pulse/trigger timing及RTL/bitstream均不在该ControlTopic中改变。

旧实现以cancel/reap后启动全新source Run来promotion replacement；这一整段只能作为待删除实现的竞态审计事实，不能复用于纠正后的monitor retarget。新合同不cancel source：同一个downstream processor owner在transaction boundary原子安装新binding并发`APPLIED`，新输出携control revision；stale downstream completion不得覆盖新revision。只有用户Stop/source failure/source schema reconfigure才进入source Run terminal/reap路径。

MAX只在声明selection与ComponentValidity为真的contributors上计算：整数和浮点均保持source dtype，complex在prepare拒绝；全invalid返回typed zero + INVALID，`REQUIRE_ALL`有任一坏component时保留由有效contributors得到的诊断值但标INVALID，`OMIT_INVALID`只忽略显式invalid component。三种reducer都在同一ROI kernel层拒绝valid NaN/Inf，invalid NaN/Inf不误伤有效结果；selected compact copy、float finite mask与widened accumulator分别进入pre-arm scratch上界。M2d同时删除从未表达mutation的恒零`config_revision/control_revision`，join与metadata只携完整source `EventRef + semantic binding fingerprint`；metadata contract identity随字段布局更新。相同binding跨restart可有相同fingerprint，但Run/stream/block/dataset generation严格区分历史。

frontend widget不拥有Selection真相：它只把gesture交给Workbench并接受applied/draft overlay setter，不暴露可轮询getter或第二份control state；callback fault保留诊断并fail-closed禁用，必须rebind才能恢复。当前范围仍明确受限：raw-only monitor不能从零创建首个ROI；没有crop/scatter/value、自由zoom/grid focus、fit/scan-y/save authority、acquisition ROI或real qCMOS qualification。legacy selector/ROI/rolling/TaskConsole consumer因此仍真实存在，本轮没有dependency-closed旧删除，也没有为它们增加current adapter。

M2d Rule-6按八个实际受影响production文件相对parent `943a85d`机械计数为`6530 -> 7870 physical`、`6015 -> 7271 nonblank`，净`+1340/+1256`；classes`33 -> 36`、dataclasses`17 -> 20`、enums`0 -> 0`、functions/methods`255 -> 304`。三个新class分别是gesture值对象、具名viewport纯变换和完整preflight transaction product；删除任一个都会恢复identity猜测、坐标重复实现或cancel前不完整准入。main严格相关`AreaSelector + RoiProcessor + LiveLive/LiveLiveDis`包络为577 physical，本轮净增为`2.32x`；加入main完整selector与真实monitor行为的宽包络为1133 physical，对应`1.18x`。`board.py`单文件从240到765 physical、为`3.19x`，其增量支付front identity、四角cell-center handle、Qt overlay、interaction pairing及所有widget生命周期出口；没有enum、manager、plugin、scheduler、ControlTopic或compatibility wrapper。

限定验证按process-lifetime installation边界分进程运行：M2d `5 passed`、M2c `6 passed`、M2b `3 passed`、M2a `3 passed`、M1 `5 passed`；owned production/test `py_compile`、fresh headless import、frontend反向import、shape-loss/authority/旧revision字段扫描与`git diff --check`通过。一次有界独立终审先发现完整preflight时点、selector接线、metadata identity、nonfinite层级、draft overlay与manual Stop promotion等具体P0/P1；逐项整改后同一审查员只读复核为`P0=0/P1=0/GO`。完整迁移仍为NO-GO；这里只判M2d纵切GO，下一切片继续真实rolling/grid/fit/calibration等未迁consumer，而不重开M1/M2a-M2c或迎合历史测试。

**历史foundation checkpoint：W5 raw committed-capture interactive Fit纵切完成，现由U0.3d唯一DataFigure host取代。** 当时public `.fit_gui(CaptureArtifactRef, model=None)`已能异步准备model、显示具名fit/batch axes、编辑catalog-owned constraints、运行每个batch cell的同一个`BoundFit`并显式Save到exact `FitResultArtifactRef`；其独立窗口与Capture-only边界现已删除。该foundation证明的轴不变量继续有效：所有非fit轴仍为batch，没有shape/rank猜测、first/flatten/trailing mean，多维data axis不会被压成`(repeat, point, data_dim)`三个匿名长度。

能力边界由三个最小真实owner闭合。frontend Fit form只把`BoundFit.parameter_definitions/parameter_units`投影到已有`FormSpec`并做exact-key round trip，不复制模型/solver/widget。headless `FitDraftAuthority`唯一持有未保存`FitExecution`并线性化execute/discard/save/close；Qt只得到无save能力的`FitDraftResult`。draft overlay只传`FitResultBatch`，composition再与窗口冻结的exact Capture/Scan ref组合，并调用data owner的完整source revision/schema/fit axes/batch axes/layout/unit/present-count validator。Save必须经过同一Experiment lifecycle gate；Save先于Close线性化时完整发布并先保留exact ref，Close先线性化时零发布。Save后reopen/render失败不抹掉ref，Save中窗口Close明确defer。

Fit求解/overlay使用独立capacity-one lane，普通冻结Figure仍使用shared frozen lane；真实repository execute被阻塞时，同一Experiment的`.figure_gui(capture_ref)`可在Fit释放前READY。长Fit只登记一个fit-specific active count而不长期占用Experiment lock，也没有引入Condition scheduler、generic workflow、lease manager或第二render owner。Experiment Close若发现active Fit会原子进入`CLOSING`并fail-fast；新Fit/preview/save随后拒绝，在途Fit退出时成为`FitCancelled`且不能接纳draft，排空后由调用方按既有active-Run语义重试Close。已进入的bounded capture decode仍不能中途强杀，deadline/cancel在lane等待、solver前后和Qt接纳点诚实生效；Agg compose继续由已有全局render owner串行。这两项是明确P2取舍，不伪装成即时中断。

W5 Rule-6以parent `4d57e7b`和八个实际受影响production Python文件机械计数：`3756 -> 5059 physical`、`3381 -> 4572 nonblank`、`3377 -> 4568 token-NCLOC`，净`+1303/+1191/+1191`，整体`1.347x/1.352x/1.353x`；classes`16 -> 19`、dataclasses`5 -> 6`、enums`0 -> 0`、functions/methods`164 -> 208`。三个新class恰为无save DTO、唯一execution authority和真实public window；没有enum、manager、plugin、scheduler、registry或compat shim。相对main严格两个widget helper `219/205/200`，净增为`5.95x/5.96x`，超过约3倍且必须如实保留；严格分母没有后台求解、batch结果、artifact save/reopen、线程/Close/stale线性化。相对main真实单surface完整用户链`850/796/753`，倍率为`1.53x/1.58x`；纳入W5确实支持的GridPlot per-cell fit对照包络`922/865/815`后为`1.41x/1.46x`，均低于3倍。删除任何新增class都会重新泄漏save capability、失去atomic draft状态或删掉用户产品本体。

验证为W5白名单`8 passed`，直接受影响W4 Figure lifecycle回归`2 passed`；owned `py_compile`、fresh headless import、import-DAG、FitExecution/shape-loss/display-authority扫描与`git diff --check`通过。一次有界独立终审在真实public closure上复现并关闭全局operation-lock队头阻塞及Close后Save绕过，最终为`P0=0/P1=0/GO`。本切片没有dependency-closed旧删除：legacy TaskConsole/Live/Grid Fit仍拥有`Add Analysis`、scan/ROI-transform/grid/live consumer。W5不因后续W6/W7完成而重开；saved-fit GridPlot由W7补交，后续Fit consumer仍是scan/ROI-transform/live grid。

**最新实现 checkpoint：W6 formal calibration creation/edit纵切完成。** public `.readout.calibration_gui(CalibrationArtifactRequest)`现在直接编辑一个已经冻结source、`ReadoutBindingKey`、总预算和deadline的显式request；`.readout.calibration_edit_gui(CalibrationArtifactRef)`先在worker从exact ref加载互相校验的artifact+report，再以其中实际运行过的request作为seed。两条入口最终都只调用composition给出的窄formal Run starter；成功本身就是repository的原子FINAL commit并返回exact新ref，saved edit不修改、覆盖或删除旧artifact。窗口不取得Experiment、repository、Port、RunPlan、raw hardware或隐藏`current_calibration`，关闭后同一Experiment继续可用。

空间与数据authority保持冻结：event layout、grid shape/order、expected centers与maximum residual只来自原始领域request或已验证saved computation，不能被detector结果、当前报告、selector或display state反写。共享Fluent form只编辑当前真实consumer需要的标量分析意图：model集合/default、box reducer/radius、PSF宽度/背景/padding、train split/seed/bins、fidelity/drop门与detector参数；exact-key重建后仍由`CalibrationAnalysisRequest`领域构造器唯一复验。canonical site向量、pixel/site/label/model/component validity、repeat及全部非READOUT_EVENT point axes仍由既有calibration pipeline保存和消费；UI没有shape/rank猜测、first/flatten/trailing mean，也没有把二维grid显示投影冒充权威site顺序。

本切片刻意没有建立preview plan、draft authority、通用editor framework、第二repository/renderer/executor、plugin、scheduler或registry。Calibration与W5 Fit的提交语义不同：formal Run已同时拥有input admission、analysis和FINAL commit，因此按钮是一次`Calibrate & Save`，而不是先在UI旁路重算一份draft再Save。Run生命周期复用`QtRunOwnerMailbox`；成功后的paired load与Agg报告复用全局capacity-one frozen-raster lane。Qt owner只持immutable form state、`RunSnapshot`、typed ref和encoded pages，不运行SciPy、repository或Matplotlib。

terminal判定要求typed `CalibrationArtifactRef`与`RunSnapshot.final_committed`同时成立，任一单边证据都不能宣称FINAL。cancel先线性化时零artifact；commit先线性化时Close延迟接收exact ref，cleanup/recovery warning保持可见，报告重开或渲染失败也不能抹掉ref；已提交却返回异常receipt时窗口明确显示commit-evidence fault，绝不谎称“没有artifact”并自动关闭。共享`FrozenRasterWindow`同时补齐replacement ownership：换报告前先退休旧tabs、Qt boards与QImage fronts，避免反复recalibrate/reopen积累隐藏GUI资源；这只是既有唯一owner的对称生命周期修复，不是calibration局部兼容层。

W6没有修改calibration物理/统计算法、capture event时序、FPGA/RTL/bitstream或qCMOS trigger语义。long-short-long采集、一次arm和硬件时序仍由既有formal capture链拥有；UI只改变显式分析参数后启动同一Run。也没有dependency-closed旧删除：legacy Live/GridPlot/TaskConsole/readout仍有live site-map、focus/selector、saved grid replay与其它calibration consumer；在这些replacement交付前不能为了删除数字提前拆掉共享生产者或物理算法。

W6 Rule-6按五个实际受影响production Python文件相对parent `7522b66`机械计数：`3711 -> 4816 physical`、`3344 -> 4376 nonblank`、`3319 -> 4345 token-NCLOC`，净`+1105/+1032/+1026`，整体`1.298x/1.309x/1.309x`；classes`12 -> 14`、dataclasses`5 -> 6`、enums`0 -> 0`、functions/methods`153 -> 186`。两个新增class分别是可序列化的真实editor seed与真实Qt产品窗口；唯一新增dataclass就是该seed。main不存在等价的immutable artifact editor，所以拿main static renderer `166 physical / 145 NCLOC`作严格分母会得到`6.66x/7.08x`且必须如实承认它不等价；相对main完整calibration产品包络`3778 physical / 3389 NCLOC`，本切片净增仅为`0.293x/0.303x`。保留差量支付formal Run/commit、cancel-close竞态、exact ref另存/重开、headless form和报告生命周期，而不是抽象海拔。

验证为W6白名单`7 passed`，直接受影响W4 paired-load/report回归`2 passed`、W5共享frozen-raster回归`1 passed`；owned `py_compile`、fresh headless import、import-DAG与`git diff --check`通过。authority与lifecycle各做一次有边界只读终审，前者确认空间/validity/display防火墙，后者复现并关闭commit evidence与旧QImage front生命周期漏洞；最终均为`P0=0/P1=0/GO`。W6不因后续W7完成而重开；完整迁移仍为NO-GO，继续live calibration/grid、TaskConsole与其它未迁真实consumer，不维护历史测试迎合中间态。

**W7 foundation checkpoint：exact saved-fit GridPlot explorer。** `Experiment.figure_gui(FitResultArtifactRef)`在没有自定义intent/selection/preferences时按typed ref分派到saved-fit explorer；携自定义view参数走generic Figure，不建立第二套projection。explorer读取content-addressed exact `FitResultBatch`，默认不调用solver、不改变`FitSpec`、不发布新artifact。每根batch axis按原`AxisSpec/AxisLayout`存在：repeat显式FACET而非mean/latest/index-0，每页最多36个逻辑panel，EXPLICIT hole保留原位且不以相邻physical row补位。U0.3d只增加用户聚焦真实cell后的显式Refit入口，仍不允许隐式solver。

这里“replay不调用solver”是必须保留的authority不变量，只禁止打开旧结果时隐式重算或改写旧artifact；它不禁止viewer的显式re-fit。U0纠正3中的`Analyze -> Fit`必须用exact source ref与当前/已保存的SelectionCandidate建立新的authority draft，运行同一BoundFit owner并产生新result/revision/ref；旧W7 artifact始终不变。

`radial_gaussian_center`现在走typed IMAGE Grid而不是整板PNG：frontend从已经求值的IMAGE与已保存`FitResultBatch`只投影immutable samples、exact source/artifact identity、logical `Selection`、sparse storage row、status以及已发布的center/1/e radius；Qt逐panel画INDEXED8 raster和vector dot/ring，绝不求值predicted image或接触solver。authority Selection产生的IMAGE保留原始非零`EvaluatedAxis.indices`，payload只要求它完整、无重复地覆盖effective raster并逐坐标匹配effective `AxisSpec`，绝不为了显示重编号而丢provenance。整页共享一个显式`ImageDisplayState`与color scale，Setting/Edit、zoom/pan、clim/cmap和rectangle display draft都复用`QtRasterBoard`的唯一interaction owner；物理坐标步长决定image target aspect，因此径向模型在非方形pixel step下仍显示为圆。page与focus复用同一块board、同一批samples/raster并重新冻结只覆盖可见logical cells的join digest；Esc原子恢复缓存page。五个仍公开的1D模型（Lorentzian、Gaussian-offset、symmetric doublet、damped sine、exponential decay）继续走generic encoded CURVE fallback，直到typed CURVE Grid有真实迁移切片；它们不是兼容死路，不能因IMAGE完成而打黑。

共享`AxisLayoutNavigator`仍是唯一Qt batch-address owner：按axis count而非axis size建控件，非singleton保持`Select…`，Previous/Next严格沿physical storage order并跳过hole。typed board的double-click只接受当前painted panel id并反查同一projection中的完整具名`Selection`与exact stored row；generic CURVE只接受Agg返回的normalized hit map。focus详情只读取选中stored row已有的status、parameters、covariance uncertainty、present/valid/used/evaluation counts、RSS-derived per-cell RMSE与R²；failed row不显示伪参数，也不访问/物化整批RMSE。

性能与内存合同不是“每点重开repository”。窗口得到的仍只是一个窄worker-affine callable；该callable第一次在shared frozen-raster worker中按调用者总预算检查encoded blob、解码后的fit retained bound、source dataset与compact model，随后一次物化exact source snapshot并在私有session中复用。focus、换页和generic export继续经过Experiment lifecycle guard，但不再次decode完整fit或materialize qCMOS source；typed IMAGE的缓存focus/overview连Figure evaluation也不重做，只做同raster的atomic relayout。显示改变才在worker顺序重栅格化，预算把既有immutable samples只计一次，同时显式计入新raster、Qt detached front、最大float64 scratch与旧/新QImage瞬时重叠；任何page/cache状态都只在新front成功present后提交。Qt owner不能访问私有session，只持compact `FitGridModel`、bounded logical projections与front。model identity同时冻结artifact target、完整axis values和同一exact `AxisLayout` mapping，panel join另外冻结ordered selection/storage/source identity，不能把同shape的另一张sparse表或另一页接到旧navigator。

image export也是display-only，但typed IMAGE不再回到canonical DataFigure默认图：worker直接冻结当前可见panel顺序、`ImageDisplayState`、viewport、cmap、共享effective clim、board columns与join digest，frontend Agg从这些已投影事实生成PNG/PDF/SVG/JPEG；普通radial DataFigure的默认Agg同样使用gray/shared scale、首个Y row在上以及physical equal aspect。PDF/SVG只让标题、轴、colorbar与fit圆环保持矢量，百万像素底图mesh强制rasterized，避免把qCMOS帧展开成百万个vector quad而绕过内存/输出上界。未提交的rectangle candidate是交互draft，明确不进入export；它既不是authority也不是已提交display state。generic 1D仍用其exact DataFigure export。两路都先在目标同目录写临时文件，成功后在export/Close共用的线性化锁内复核cancel，再用`os.replace`原子提交；Close先获胜时删除temp并保留既有destination，replace先获胜时得到完整新文件。W7仍只复用一个frontend render owner、一个capacity-one frozen executor和已有Fluent/Qt board；没有第二repository、solver、executor、scheduler、gallery registry或fit codec。

本轮W7 typed IMAGE/1D fallback及其公共window-lifecycle收口的Rule-6按13个实际受影响production Python文件相对当前parent机械计数：`13869 -> 17014 physical`，净`+3145`，diff gross `+3800/-655`，整体`1.227x`；classes `71 -> 74`、frozen dataclasses `22 -> 25`、enums `2 -> 2`、functions/methods `563 -> 633`。其中`_fit_grid.py`单文件`785 -> 2421`达到约`3.08x`，因此单独触发海拔复核。main严格等价物按`GridCell 228 + ImageCell 159 + GridPlot 708 + 2D fit overlay约50 = 1145 physical`计，current净生产增量约`2.75x`、gross addition约`3.32x`；gross超过3倍的差额已经逐项核成exact artifact/session、sparse batch、typed multi-panel interaction、generic五模型fallback、强异常安全、内存上界与atomic export，不是manager/executor/registry/solver或单成员enum。终审实际删除/合并了重复range owner、每panel重复home viewport、panel key/storage/caption重复存储与canonical/export两套radial draw；没有还能大量删除且不牺牲上述真实consumer的抽象。公共`retain_window`只把会形成self-cycle的signal closure改为weakref，未增加第二registry或lifecycle owner。

最终验证按process-lifetime installation边界逐文件新进程运行活动manifest：`102/102`文件绿色，合计`1313 passed + 3 expected skipped`、零失败/错误；其中W7 focused为`30/30`，Qt IMAGE为`33/33`，frontend Figure为`33/33`，W4/W5/W8分别`24/8/3 passed`。owned `py_compile`、fresh headless import、import-DAG、manifest gate、`git diff --check`、exact-ref/no-refit、sparse-hole、repeat-FACET、one-load/one-materialize、prebudgeted-sample peak、cached-relayout overlap、atomic export/Close、PDF/SVG rasterized mesh与shape-loss扫描通过。独立correctness对抗先后抓到generic 1D blackout、spatial-fit contract、join digest、page strong exception、WYSIWYG export、physical aspect、重复sample预算及vector mesh爆炸等material问题，逐项修复后复核`P0=0/P1=0/GO`；独立Rule-6/性能终审同样为`GO`。完整迁移仍为NO-GO。本切片没有dependency-closed legacy删除：旧TaskConsole/Live/GridPlot仍拥有live artist、scan/ROI-transform fit、selector和其它rolling consumer，下一切片必须从最新Git checkpoint重新选择真实consumer，不能为了删文件把这些路径提前打黑。

**历史实现 checkpoint：M2e default raw monitor first typed ROI纵切完成；首ROI source restart已废止。** 默认`.readout.camera_monitor_gui()`仍从`roi=None`的一个raw IMAGE panel启动，没有scalar DatasetId、processor tap或hidden scalar stream；raw schema继续完整保留`(R, P, *data_shape)`与具名`SPATIAL_Y/SPATIAL_X`、ComponentValidity、EventRef、revision和provenance。raw与已有ROI两种窗口共用唯一`QtRasterBoard`/`ImageViewportTransform` selector host；selector只有用户显式启用后，gesture才按当前可见`BoardFrame`的board/layout/sequence/source identity及viewport revision解析成typed pixel-cell rectangle。该Selection是下游analysis/monitor ROI，不是camera acquisition ROI；它不改变exposure、硬件ROI、trigger、FPGA/RTL或bitstream，也不进入FitSpec、ScanOutputContract、CommittedTransform、Save或artifact manifest。

当时画框只产生可见draft，`MEAN|SUM|MAX`作为显式reducer随Apply冻结；随后实现会取消raw Run并以新Run/generation把board从1 panel切成4 panels。这个source restart流程已由`UX-001`否决。纠正后，首个ROI创建只建立downstream subscription/processor stream/generation并在其首个合法output后原子接入board；raw source RunId、raw stream generation、raw front与source tap topology不变。reducer继续显式可见，existing ROI retarget走ControlTopic/APPLIED；Close/manual Stop与stale downstream completion都不能错误切换topology。

M2e Rule-6按唯一实际修改的production文件相对parent `2a21bf2`机械计数：`1341 -> 1345 physical`、`1280 -> 1284 nonblank`、`1278 -> 1282 token-NCLOC`，净`+4/+4/+4`，整体约`1.003x`；classes`2 -> 2`、dataclasses`1 -> 1`、enums`0 -> 0`、functions/methods`34 -> 34`。净差量相对main严格相关first-ROI包络`1730 physical / 1455 token-NCLOC`仅`0.0023x/0.0027x`，相对公平完整包络`7014/5540`为`0.0006x/0.0007x`；整个current owner文件相对严格包络为`0.78x/0.88x`。本切片没有新class、worker、executor、ControlTopic、repository、renderer、manager、registry、compatibility wrapper或通用ROI多态机，差量只支付raw/ROI共用selector host与终态后的topology handoff。

验证按process-lifetime installation边界分进程运行：M2e `2 passed`、M2d `5 passed`、M2c `6 passed`、M2b `3 passed`、M2a `3 passed`、M1 `5 passed`，合计`24 passed`；另有owned `py_compile`、`git diff --check`、fresh headless import、shape-loss/authority/第二executor机械扫描通过。一次且仅一次有界独立对抗复核为`P0=0/P1=0/READY/GO`。完整迁移仍为NO-GO；本切片没有dependency-closed legacy删除，因为旧TaskConsole/Live ROI仍拥有crop/scatter/value、scan-y、site/grid、panel retarget与其它未迁consumer，不能因首个current ROI可用就提前删除或适配旧Hub。下一切片继续从最新Git/design checkpoint选择一个真实consumer，不重开M1/M2a-M2d，也不迎合历史测试。

**最新实现 checkpoint：W8a selection-only transformed Capture Fit忠实显示纵切完成。** 现有公开`Experiment.fit(capture_ref, committed_transform=...)`本来已经执行、保存并重开权威transform，W8a只闭合它到`.figure()`、generic frozen Figure和W7 exact saved-fit GridPlot的presentation接缝，不新增public API。可显示子集被刻意钉死为：一个且仅一个`Selection` operation；term只能是`IndexRangeSelection`或`CoordinateRangeSelection`；只能选择schema声明为`SPATIAL_X/SPATIAL_Y`的数据轴；每根被选轴仍必须是显式fit axis。fit之外的repeat、全部point轴、SITE/其它data轴、`PointLayout`和batch layout原样保留；所有`ReductionSpec`、多operation、`IndexSelection` drop、非空间轴、repeat/point/cell-axis变换与任何display reduction均在Figure evaluation、Agg和Qt之前明确拒绝。

实现没有物化第二份transformed dataset：contract owner先用`resolve_transformed_schema()`核对committed input/output fingerprint、effective fit/batch `AxisSpec`和未变的cell layout，只为metadata suggestion建立瞬时effective schema；随后把artifact中的同一个authority Selection与仅限batch轴的W7 page/focus selection合并成raw-schema `ViewSpec`。`DataFigure`在读取数组前再次执行同一contract predicate，要求authority term逐值相等、selection轴不重复、额外term只属于saved batch axes、所有binding无`REDUCED`，再运行既有完整source-ref/schema/fit-axis/batch-layout binding validator。FigureEvaluator因此仍从exact raw `OwnedSnapshot`按同一Selection取值，既不伪造derived ref/lineage，也不把display intent写回FitSpec。ComponentValidity逐pixel保留，非零物理坐标、EXPLICIT sparse hole与physical storage row按具名batch address交叉验证；保存后W7只加载既有result/source并调用`evaluate_batch()`画overlay，绝不refit或发布新artifact。W8a完成时W5 `.fit_gui()`仍是raw-only authoring产品；当前selection-only显式authoring现态由下述W8b覆盖，ROI/transform editor仍未偷渡。

W8a Rule-6按三个实际修改的production Python文件相对parent `3871137`机械计数：`1339 -> 1556 physical`、`1231 -> 1436 nonblank`、`1228 -> 1434 token-NCLOC`，净`+217/+205/+206`，整体`1.162x/1.167x/1.168x`；classes`2 -> 2`、dataclasses`1 -> 1`、enums`0 -> 0`、functions/methods`26 -> 29`。净差量相对main严格相关的旧DataFigure/live/TaskConsole selection-fit行为包络`201 physical / 177 token-NCLOC`为`1.08x/1.16x`，没有超过约3倍；完整current touched closure相对该严格包络仍为`7.74x/8.10x`，高处来自此前已经挣得的typed ViewContract、multi-batch FitResult与DataFigure基础，而本切片没有复制solver、transform executor、repository、renderer或GridPlot。三个新增函数分别是raw/effective两路共用的suggestion core、suggestion与DataFigure共用的authority projection，以及构造后硬验证；自审已把单次调用的lift wrapper内联，没有新增class、enum、worker、executor、artifact、registry、manager、compatibility shim或通用transformed-data framework。

验证为W8a `3 passed`、frontend Figure `33 passed`、W7 saved-fit Grid `7 passed`、W5 raw Fit `8 passed`，合计`51 passed`；owned `py_compile`、fresh headless import、`git diff --check`和authority/shape-loss静态扫描通过。一次且仅一次有界独立对抗最初以`NO-GO`指出ROI后Reduction是测试自造需求、证明面不足且硬validator归属错误；实现采纳全部material finding，删除Reduction解释器，把contract收窄到selection-only并把predicate移入validation owner。最终自审又删除single-use wrapper并禁止把effective-schema alternatives泄漏成raw-schema候选，结论为`P0=0/P1=0/GO`。完整迁移仍为NO-GO；本切片没有dependency-closed legacy删除，旧TaskConsole/Live/GridPlot继续拥有scan-fit、live artist、selector与rolling consumer，不能因冻结Capture Fit可显示就提前删旧生产者或适配历史测试。

**最新实现 checkpoint：W8b explicit selection-authority Capture Fit Workbench纵切完成。** public seam为`Experiment.fit_gui(capture_ref, model=..., committed_transform=...)`；参数可以继续为`None`走W5原raw行为，也可以是调用者已经针对同一capture schema提交的W8a selection-only transform。窗口创建时冻结这个值，后台为closed model catalog中的每个候选建立携同一transform的`BoundFit`，并要求所有候选逐值共享同一个CommittedTransform和同一个authority Selection；不支持的model被排除，最终没有合法model则在source Figure、Agg和数组拟合前明确失败。GUI只编辑catalog-owned constraints，`fit_spec_from_form()`只在原`FitSpec`上替换constraints并重新bind；execute再次比较exact frozen transform、admit固定Capture ref并重跑同一个frontend显示合同，绝不从viewport、M2 selector、display reduction或当前画面生成权威。

初始source和Clear后的source都只把该authority Selection作为raw snapshot的显示投影，因而用户不会在全图上看到一个其实只拟合ROI的伪预览；repeat、point、SITE/其它data axes、PointLayout、ComponentValidity与batch layout仍由原schema/BoundFit保留，显示层可以按既有ViewContract做带标签的presentation，但不能写回FitSpec。draft overlay继续携带同一`FitResultBatch.spec.committed_transform`，Save仍只发布既有`FitExecution`并从exact ref重开；没有第二transform executor、derived dataset/ref、solver、repository、worker或保存路径。`describe_authoritative_transform()`从TaskConsole局部实现机械迁到headless `zlc_frontend.authority`唯一owner，TaskConsole和Fit editor成为两个真实consumer；旧定义与export已物理删除，不留alias/shim。Fit editor另显示不可编辑authority摘要，axis摘要继续逐根显示effective fit/batch轴。

W8b Rule-6按十个实际受影响production Python文件相对parent `238e4e3`机械计数：`7070 -> 7259 physical`、`6440 -> 6612 nonblank`、`6374 -> 6542 token-NCLOC`，净`+189/+172/+168`，整体`1.027x/1.027x/1.026x`；classes`21 -> 21`、dataclasses`1 -> 1`、enums`0 -> 0`、functions/methods`251 -> 255`。main没有“immutable exact Capture ref + caller-committed transform + background draft + explicit artifact Save”的一一等价产品，故不伪造严格倍率；只作广义海拔参照时，净差量相对W8a已记录的main selection-fit行为包络`201 physical / 177 token-NCLOC`为`0.94x/0.95x`，但两轮共享该旧行为，不能把它当两个可累加的等价分母。四个净新增函数分别承担BoundFit/result共用的窄selection合同、不可编辑authority摘要和source Figure选项闭包；唯一新文件虽只有一个formatter，却已经有TaskConsole与Fit editor两个不同consumer，若塞回任一consumer会恢复反向依赖或重复实现。没有新class、enum、executor、manager、plugin、registry、compatibility wrapper或通用transformed-data框架。

current白名单按process-lifetime installation边界分进程运行：W8b `1 passed`、W5 `8 passed`、W8a `3 passed`、W3 TaskConsole `9 passed`，共`21 passed`；W8b oracle用真实virtual capture证明初始与Clear预览逐值收到exact ROI、constraint改变后GUI result与直接`Experiment.fit()` canonical bytes相同、Save artifact保留相同transform，并先证明“ROI后reduce repeat”是data kernel合法Fit，再证明它在任何Figure/Agg调用前被Workbench拒绝。owned `py_compile`、fresh headless import、`git diff --check`、formatter单owner与shape-loss扫描通过。一次且仅一次有界独立正式对抗结论为`P0=0/P1=0/GO`；它确认worker/Qt、cancel/stale/deferred-save状态机未分叉。完整迁移仍为NO-GO；本切片只删除W5 raw-only guard/文字和TaskConsole局部formatter，没有dependency-closed legacy producer/module可删，旧TaskConsole/Live/GridPlot仍有scan/live/selector/rolling真实consumer，下一切片继续从最新Git/design checkpoint选择而不重开W8a/W8b或迎合历史测试。

**最新实现 checkpoint：U0.2f finite exact capture raw IMAGE完整交互纵切完成。** W1现直接消费既有`QtRasterBoard`、`ImageDisplayState/ImageViewportTransform`、revisioned Setting/Edit和`LiveBoardController`，恢复A/C/Z/H/hover、clim/cmap与display-only rectangle；Capture artifact、FINAL、request、fit/scan/ROI authority均不受display操作影响。重复Run只保存authored display state，旧run origin、selector/cross/area/hover/pending、source/slot/board全部失效；多维data axes继续由schema完整保存，viewer只接受显式双空间轴而不做shape猜测或隐式降维。

反例审查发现并关闭旧generation raster尚未完成时可重开Run的P1；双重`worker_idle` gate与owner-cycle尾重算现在使两代exact preview不能重叠，blocked Close也不会late-publish。finite候选之外只有一个distinct旧raster/sample front，hold与current是别名，故精确预算保持R1/S1；默认virtual profile总峰值见证`1,359,561 bytes`，`required-1`在camera arm前拒绝，exact required通过。Rule-6 dependency-closed cut为`5350 -> 5911 physical = 1.105x`、`4995 -> 5537 nonblank = 1.109x`，零新增production class；活动manifest `100/100`文件、collect `1287`项、逐文件新进程`1284 passed + 3 expected skipped`。两路最终独立复核分别为生命周期`P0=0/P1=0/P2=0/GO`与架构复用`GO`。完整迁移仍为NO-GO；下一真实consumer仍应从live paired Sites、generic Distribution/Grid与完整TaskConsole中按dependency-closed优先级选择，不重开finite IMAGE，也不提前删除仍有旧consumer的producer。

**最新实现 checkpoint：U0.3a frozen single-panel HISTOGRAM DataFigure typed viewer纵切完成。** `open_data_figure_workbench(DataFigure)`只在产品边界恰好满足“一个document layer、一个evaluated layer、一个cell、一个resolved input、intent为HISTOGRAM且至少一个`EvaluatedHistogram` series”时进入typed路径；worker用既有`SinglePanelAggRenderer`产生一个固定`800×520` packed-RGBA `BoardFrame`，Qt owner只把该immutable front交给既有`QtRasterBoard`。首张typed front成功present前Loading页、Histogram/Edit页及三个typed controls不会提前切换；任何其它intent、multi-layer/multi-cell/Grid或typed峰值超限都继续走既有encoded DataFigure fallback，而不是伪装成半可用交互板。成功路径恢复range candidate、continuous cross、真实bin hover、wheel zoom、x pan/home/pin、linear/log count scale、bin count、三态count relim、同一`HistogramDisplayState`驱动的Setting/Edit CAS及当前已显示front的原子PNG导出；这些全部是display-only，不创建FitSpec、Selection authority、ReductionSpec、threshold、fidelity结果或新artifact。

多维保真不由viewer猜shape。`FigureEvaluator`仍按声明的`AxisViewRole`求值：BATCH轴成为逐series的具名address，SAMPLE轴逐项保留`SampleCoordinates`，FACET产生多cell并明确留在encoded路径；fixture中的`site×channel` trailing axes因此保留为两个site series，且每个series仍分别携`channel`与`repeat`轴坐标。`ComponentValidity`过滤只移除明确invalid component并保留`dropped_count`；viewer没有`reshape`、`flatten`、rank/singleton猜测、取第0项或trailing mean。worker front必须逐次通过request sequence、完整authored viewport、source/document/input/join provenance与首张front的exact `EvaluatedSeries`对象身份CAS；固定raster geometry与packed stride在front构造时另行强制。任何伪造sequence、同revision状态漂移、join digest替换、series替换或geometry替换都保留旧front与旧display revision并明确失败，不能把另一份数据接到旧画面。

线程、生命周期与内存仍只有一个owner：初始化、rerender和encoded fallback共用Workbench既有capacity-one raster lane，Figure/Axes/Artist从不越过worker边界；Qt只持typed frame/raster及两份revisioned editor投影。typed峰值由frontend唯一估算器显式计入frozen evaluated arrays、series-dependent histogram scratch、worker候选front、当前Qt front及旧/新瞬时重叠，准入取`min(window limit, DataFigure.render_memory_limit_bytes)`；`required-1`走encoded、exact required进入typed，较大窗口不能放宽DataFigure冻结预算，后续增大bin count若超限会在Agg构造前拒绝并原子保留旧front。bin=17的定向profile为`required=23,378,272 bytes`、`retained_before=1,665,536 bytes`、`incremental tracemalloc peak=4,437,079 bytes`、`observed total proxy=6,102,615 bytes`、双series raster=`1,664,000 bytes`；tracemalloc下worker rerender为约`599.5 ms`，UI线程不执行该工作，估算器相对观测保持保守。Close立即撤下Qt front并置cancel，不等待正在运行的Agg/export；lane排空后清除renderer closure、cached DataFigure provenance与exact series引用，迟到结果绝不present。导出先在目标同目录完成临时PNG，再与Close共用commit lock复核cancel并`os.replace`；Close先获胜时既有目标逐字节不变且temp删除。

U0.3a Rule-6按parent `f414212`的三个实际production owner机械计数：`1997 -> 3047 physical`、`1816 -> 2791 nonblank`、`1807 -> 2782 token-NCLOC`，净`+1050/+975/+975`，完整closure为`1.526x/1.537x/1.540x`；classes由`2 -> 4`、frozen dataclasses由`0 -> 1`、enums保持0、functions/methods由`70 -> 107`。`_figure.py`相对parent极小118行adapter达到`9.68x`而触发海拔复核；对`main`严格真实等价物`HistogramFigure + AreaSelector + CrossSelector + ZoomPan = 836 physical / 775 nonblank / 693 token-NCLOC / 4 classes / 46 functions-methods`，本轮净增只有`1.256x/1.258x/1.407x`，完整当前`_figure.py`也只有旧包络的`1.366x/1.356x`。两个新增class分别是真实Qt产品窗口和唯一worker→Qt frozen DTO；唯一dataclass就是后者。实现继续复用唯一histogram state/binning、selector/zoom、Qt board、Agg renderer、capacity-one executor、revisioned form与atomic export owner，没有第二renderer/executor/manager/registry/plugin、单成员enum或兼容wrapper，Rule-6终审为GO。

活动manifest现为`103/103`文件，一次性collect `1328`项；随后按process-lifetime InstallationRuntime边界逐文件新进程实跑为`1325 passed + 3 expected skipped`，零failed/error/xfailed/xpassed。U0.3a定向合同为`12 passed`，另有W8当前语义回归`3 passed`；owned `py_compile`、fresh headless import、import-DAG/manifest、shape-loss/第二executor扫描与`git diff --check`通过。独立authority/lifecycle审查先后复现provenance替换、raster geometry绕预算、typed UI过早admission以及close/export证明缺口；逐项整改并增加反例后最终为`P0=0/P1=0/GO`，Rule-6独立审查同样GO。完整迁移仍为NO-GO：generic/live Distribution/Grid、generic IMAGE/CURVE/METER、多cell typed figure、显式`Analyze -> Fit`、报告zoom与完整TaskConsole/launcher仍是实际未闭合consumer。本切片没有dependency-closed legacy删除；旧Live/GridPlot/TaskConsole生产者仍有上述consumer，不能因为单panel frozen HISTOGRAM完成就提前删除或为历史测试增加adapter。

**最新实现 checkpoint：U0.3b frozen single-panel CURVE DataFigure typed viewer纵切完成。** UX salvage固定`main`旧`DataFigure + Live1D + AreaSelector + CrossSelector + ZoomPan`：普通1D保留每个Y列/series及同一声明X；Area只提交x span，Cross保持连续cursor，wheel与middle drag只改X，home回完整当前domain；Y只受tight/normal/fixed控制，Setting与Edit投影同一状态，保存当前可见图。current只在“一个document layer、一个evaluated layer、一个cell、一个resolved input、intent为CURVE、至少一个`EvaluatedCurve`、全部series共享exact X与value unit、X有限numeric且严格单调、没有fit overlay”时进入typed路径。`DataFigure.has_fit_overlays`是owner提供的最窄只读事实；fit-bearing CURVE必须继续走whole-figure encoded renderer，因为当前单panel renderer不消费`FitResultBatch`，若强行typed会静默丢fit曲线、`NO_VALID_DATA/NOT_PRESENT`与sparse batch对齐。categorical/repeated/nonmonotonic X、multi-layer/cell/input、其它intent及预算不足同样encoded，并在窗口summary显示具体`interaction unavailable`原因，不以空交互句柄或猜index轴冒充可用。

本轮没有建立Curve窗口或第二状态机。U0.3a的`DataFigureWindow + front DTO`原位泛化为封闭的CURVE/HISTOGRAM numeric sum：一个`QtRasterBoard`、一个pending state/origin/editor、一个contract、一个renderer callback、一个export lock；两个private union alias只服务直接`isinstance`分支，没有protocol、registry、base-state类族或plugin。CURVE直接复用既有`CurveDisplayState`、form投影、`curve_display_with_x_view`、`CurvePanelPayload`、`SinglePanelAggRenderer.render_interactive_curve`及Qt的range/cross/zoom/pan/home/hover实现；HISTOGRAM原行为与12条产品oracle原样保留。family未知时不预建两套editor或对同panel双bind；首张typed front严格按“validate → `board.present()` → 立即提交state/contract → numeric page/chrome → 懒建并绑定唯一一对Setting/Edit”的顺序接纳。editor/binding故障只保留已经可见的exact raster并fail-closed禁用interaction/export/editor；不回滚front，不让`raster_ready`等待不存在的binding，也不由后续sync覆盖原始诊断。

数据路径继续axis-total。U0.3b oracle使用`(R=2,P=21,site=3,component=2)`及对齐`ComponentValidity(site,component)`，repeat只按ViewSpec中显式`REDUCED(mean)`解析，site与component两个具名BATCH轴产生6条series；payload逐对象复用evaluator-owned `EvaluatedSeries`，并逐项保留batch address、ReductionResolution、X AxisSpec/unit、value unit与validity。Workbench新增路径的`reshape/squeeze/flatten/[:,0]/nanmean/隐式mean`计数为0。`numeric_curve_coordinates`也不再在准入前建立O(n) list/tuple/`np.diff` scratch：owner对exact `axis.coordinates`单遍O(1) live-scratch验证后返回同一个tuple；classifier不复制series集合。range gesture只写display-only候选，viewport commit只更新`CurveDisplayState.x_view`；从不构造Selection、FitSpec、ReductionSpec、CommittedTransform或artifact。

每个front同时通过独立request sequence、完整authored state、重算的`curve_home_x_limits(exact x axis)`、source/document/input/join provenance、label/unit及首张front的exact series对象身份CAS；初始state与GUI独立拥有的默认值比较，不能用worker返回值自证。伪造sequence/state/home/join/series/raster geometry均在`present`前拒绝，rerender失败按exact origin丢弃pending curve intent并保留旧front/state。固定`800×520` packed-RGBA export直接写当前BoardFrame，不重新调用DataFigure而产生另一viewport。Close沿共享路径立即撤front、cancel并禁止late-present；renderer closure置空后释放cached DataFigure与evaluated arrays，原子export仍由staged file + commit lock保证Close获胜时旧目标不变。

CURVE typed准入仍取`min(window limit, DataFigure.render_memory_limit_bytes)`；唯一live-panel估算器计入evaluated arrays、Agg/artist/raster峰值及当前/候选两个front，`required-1`encoded、exact required typed，较大窗口不能放宽冻结预算。`site×component`六series、`800×520`定向profile的required见证为`23,373,680 bytes`；一次warmup后的6次完整worker render为`94.035..100.203 ms`、median约`96.764 ms`。瓶颈仍是既有Agg compose，不是classifier、Hub或Qt，因此没有增加缓存Figure、第二executor、LOD或异步工作流。

U0.3b Rule-6按parent `25deda7`的三个实际production owner（`_figure.py + curve_display.py + data_figure.py`）机械计数：`1894 -> 2223 physical`、`1714 -> 2024 nonblank`、`1679 -> 1979 token-NCLOC`，净`+329/+310/+300`，closure为`1.1737x/1.1809x/1.1787x`；classes`6 -> 6`、dataclasses`4 -> 4`、enums`0 -> 0`、functions/methods`70 -> 83`。main严格CURVE oracle `Live1D + AreaSelector + CrossSelector + ZoomPan`为`442/405/368 physical/nonblank/token-NCLOC、4 classes、25 functions`，本轮净增仅为`0.7443x/0.7654x/0.8152x`，class净增0、function净增`0.52x`；完整touched closure相对含DataFigure/HISTOGRAM的公平旧用户面也只有约`1.337x/1.315x`，没有约3倍项。无第二renderer/executor/board/selector/form framework、manager、controller、registry、compatibility alias、单成员enum或旧`_Histogram*`私有owner；`_render_figure`保留是因为current `_fit.py`仍有真实import/call consumer，不是历史残余。

最终活动manifest为`104/104`文件、collect `1338`项；逐文件新进程实跑`1335 passed + 3 expected platform skipped = 1338`，零failed/error/xfailed/xpassed。focused为U0.3a `12 passed`、U0.3b `10 passed`、curve-display owner `12 passed`，W3/W4/W8分别隔离通过`5/24/3`；owned `py_compile`、fresh headless import、diff-check与shape-loss扫描通过。三路独立对抗先后抓到初始worker state自证、home范围自洽伪造、controls-before-front、unbound-ready/诊断覆盖、encoded无reason及准入前O(n) scratch，全部以反例关闭后复核为`P0=0/P1=0/P2=0/GO`。该GO只关闭**冻结、单layer/cell/input、无fit overlay的numeric CURVE**；generic IMAGE/METER、fit-bearing CURVE typed overlay与`Analyze -> Fit`、multi-cell/Grid、任意archive browser、报告zoom、完整TaskConsole/launcher仍未闭合。没有dependency-closed legacy删除；下一切片必须从最新Git与这些真实consumer重新选择，不能把本numeric seam扩成plot-kind framework或删除仍由旧Live/GridPlot/TaskConsole消费的producer。

#### U0.3c generic single-panel IMAGE salvage gate（收口状态：`COMPLETED`）

本纵切固定 `main@6c337d49c7086fa0ff21f879cd159bdf0e753f51` 的 `frontend/data_figure.py::DataFigure`、`frontend/live.py::Live2DDis` 与 `frontend/selectors.py::AreaSelector/CrossSelector/ZoomPan` 为旧行为权威；没有迁入旧mutable Matplotlib Figure、shape/rank推断、Qt线程拟合或Hub/latest机制。旧窗口“载入二维图后立即保留原始空间坐标与像素方向，Area/Cross/Zoom/Pan/Home、hover、cmap/clim/relim和保存作用于同一可见图”的日常契约，已经由既有唯一 `ImageDisplayState + ImageViewportTransform + rasterize_image_indexed8 + ImagePanelPayload + QtRasterBoard` 链恢复；没有第二image widget、selector、renderer、form或worker。

| salvage字段 | U0.3c最终合同 | 收口证据 |
|---|---|---|
| typed资格 | 只接受一个document layer、一个evaluated layer、一个cell、一个resolved input、恰好一个real-numeric `EvaluatedImage`且无fit overlay；其它topology都完整走encoded fallback并显示具体`interaction unavailable`原因 | `CLOSED`：缺frame、非规则轴、fit-bearing与超预算反例均验证encoded，不会丢内容后假装typed |
| 轴与多维保真 | evaluator把source `AxisSpec.coordinate_frame`原样保存在`EvaluatedAxis`，把cell schema的`value_unit`原样保存在`EvaluatedImage`；frontend唯一纯投影只认字段中声明的`SPATIAL_X/SPATIAL_Y`、exact indices/coordinates/unit/frame。缺frame、role不符、frame不一致、非数值/非有限/非精确规则几何都fail closed；不按rank/singleton/tuple位置猜轴，不flatten、index-0或隐式reduce | `CLOSED`：完整values/validity对象identity、轴方向、unit/frame与source revision都跨rerender保持 |
| 无fit IMAGE UX | A/C/Z/H、hover/cross、双轴wheel zoom/middle-pan/Home、rectangle、clim rail、tight/normal/fixed、六种cmap、x/y/color pins、同源Setting/Edit和当前front PNG export全部作用于同一typed front | `CLOSED`：四种X/Y升降序都固定“首列在左、首行在顶”；rectangle/cross为可见overlay且不改数据 |
| fit-bearing边界 | generic U0.3c不借W7 artifact-specific radial projection制造模型特例；任何fit overlay仍完整encoded，generic exporter也明确拒绝overlay。W7 saved-fit Grid继续自己的typed projection/export，不重跑solver | `CLOSED`：fit-bearing反例进入whole-figure fallback；W7回归保持。下一统一Fit纵切才同时接CURVE/IMAGE authority |
| selection与authority | rectangle/cross只形成front-bound `DISPLAY ONLY` candidate；本切片不构造`SelectionCandidate`、`FitSpec`、CommittedTransform或artifact | `CLOSED`：拖框只更新Qt overlay/诊断，authored display与exact image不变；下一纵切把同一selector接入1D range与2D box Fit |
| CAS与故障 | request sequence、authored display、exact image object、input/document/join、viewport axes/frame、fit确实不存在、pixel format、raster geometry及source provenance全部在present前独立校验；worker返回值不能自证 | `CLOSED`：伪造新array、axis/frame、viewport、pixel format/geometry、controls失败、blocked rerender+Close均保留或撤掉正确exact front，禁止late present |
| 预算与导出 | admission显式保留evaluated values/validity、current RasterBuffer与Qt detached plane；唯一worker lane使candidate raster与PNG export不并发，因此加入二者incremental peak的较大值。导出逐字使用已提交viewport/cmap/current effective clim/value unit；动态NORMAL/TIGHT clim不重算，未提交rectangle不进入artifact | `CLOSED`：`required-1` encoded、exact required typed；strict regular-pixel export/W7走`imshow`，2304² uint16 measured incremental约374MB，静态估算482,002,944B；默认512MiB总预算仍覆盖held fronts。canonical/encoded radial显式走`pcolormesh`，非规则cell edges与fit overlay不会错位 |
| 生命周期与复杂度 | 复用现有唯一window、capacity-one lane、board、revisioned editor、atomic export与Close；Close立即撤front、取消late-present并释放cached DataFigure | `CLOSED`：没有第二executor/renderer/selector/form framework、registry、plugin、plot-kind base hierarchy、单成员enum或单consumer wrapper |

viewport的本源修复不是再放宽浮点容差，而是把display-only normalized bounds规范到40-bit binary fixed grid；公开的normalized→physical pins→normalized桥先做一次可表示性映射，再证明fixed point，不能稳定表示就拒绝。axis edge只是同一声明轴推导出的`compare=False` cache，不是第二权威。1e9坐标offset、升/降序及32轮zoom/pan逐轮exact round-trip通过；2304² pan构造从重复扫描axis的约16.32ms降到约1.86ms。

内存对抗先用profile证伪了原`pcolormesh`低估（1024²真实峰值最高约为旧估算3.28倍），随后只在已经跨过严格规则像素合同的generic/W7 export改为低内存`imshow`并重测；canonical radial仍可接受非规则坐标，故明确保留cell-edge-exact `pcolormesh`。512/1024多dtype和2304² uint16冻结profile都低于新的source-retained incremental公式，generic fit与动态clim边界也各有直接反例。

Rule-6按parent `a93de25`的七个实际production owner机械计数：`7364 -> 8079 physical`、`6657 -> 7327 nonblank`、`6466 -> 7095 token-NCLOC`，净`+715/+670/+629`，完整closure为`1.0971x/1.1007x/1.0973x`；classes `67 -> 67`、frozen dataclasses `50 -> 50`、enums `8 -> 8`、functions/methods `241 -> 257`。main严格IMAGE oracle `DataFigure + Live2DDis + AreaSelector + CrossSelector + ZoomPan`为`1357/1252/1078 physical/nonblank/token-NCLOC、5 classes、67 functions`，本轮净增仅为`0.5269x/0.5351x/0.5835x`，class净增0、function净增`0.2388x`；最大单owner倍率约`1.262x`，没有约3倍项。新增函数只支付声明轴投影、exact viewport canonicalization、IMAGE分支与export/预算；所有保留抽象均有generic viewer或W7现有consumer。

focused foundation/export/U0.3a-b-c/W7回归在最终分流前后分别通过`104`与`52`项，独立数据/内存和Workbench事务审查最终均为`P0=0/P1=0/P2=0/GO`。活动manifest最终collect `106/106`文件、`1369`项；逐文件新进程实跑为`1366 passed + 3 expected platform skipped = 1369`，零failed/error/xfailed/xpassed。迁移期process-singleton测试必须继续按既有规则逐文件新进程执行，不能把同进程级联失败误修成runtime兼容层。

#### U0.3d unified DataFigure Analyze → Fit salvage gate（收口状态：`COMPLETED`）

本纵切固定 `main@6c337d49c7086fa0ff21f879cd159bdf0e753f51` 的 `frontend/data_figure.py`、`figure_viewer.py`、`selectors.py`、`neutral_atom/core/fitting.py` 与 `operations/processors/fit.py` 为行为/算法 oracle：用户从当前可见 CURVE range 或 IMAGE box 发起普通 Fit，一次动作即提交显式 authority draft，结果马上覆盖同一图；Fit 所用物理模型、参数语义与坐标轴来自声明，不从 rank、singleton、当前 viewport 或显示 reducer 猜测。迁移没有搬回 mutable Figure、Qt 线程 solver、全局 current-fit、Capture-only repository 或独立 W5 window。

| 收口字段 | U0.3d 最终合同 | 反例/实现证据 |
|---|---|---|
| 唯一产品 host | `DataFigureWindow` 是普通 Figure、`.fit_gui()`直达入口与 saved-fit focused-cell Refit 的唯一 Fit host；`fit_gui`只负责以同一 host 打开 Analysis tab。共享 `FitAuthoringPane` 消费 compact `FitAuthoringOption` 与既有 Fluent form，不复制 model、solver、Setting/Edit framework、board、renderer 或 executor | 独立 `workbench/_fit.py`、旧 W5 窗口与两份旧测试已物理删除；生产树不存在 `_fit` import 或 `CaptureFit*` compatibility alias |
| source 与 durable identity | headless `FitResultRepository` 的 closed tagged source union 只有 FINAL `CaptureArtifactRef | ScanArtifactRef`；每种 source 委托自己的 inspect/materialize/canonical ref owner，并把 exact `DatasetRevisionRef`、schema、FitSpec 与 result 重新绑定。`FitResultArtifactRef` 不伪装成可注册 generic Analysis artifact | Capture/Scan execute、save、fresh-process load、corruption、foreign repository 与 exact-ref replay合同通过；第三种 source 不能靠 registry/plugin 动态塞入 |
| 多维轴与有效性 | Fit axes 只由 catalog role + 显式 named AxisId 决定；其余 repeat/point/data axes 全部保留为 batch axes，`AxisLayout` 的 physical order 与 sparse hole 原样存在。ComponentValidity、present/valid/used 计数逐 component 消费；没有 first、flatten、`[:,0]`、trailing `nanmean` 或把多维 `data_shape` 压成匿名三项 | 1D range只选择拟合轴；2D box同时选择两个声明空间轴；saved grid按逻辑地址/physical storage精确导航，hole为NOT_PRESENT而不借相邻row；FitResultBatch继续服务真实grid fit，未被降成标量结果 |
| display/authority 防火墙 | selector先产生带front origin/revision的 presentation candidate；只有普通 `Fit` 动作经 `suggest_fit_draft` 提升为 authority Selection。viewport、zoom/pan、relim、cmap、当前grid focus与repeat显示策略都不能自动进入FitSpec；Clear只撤未保存draft/overlay，Save发布exact ref后再从该ref reload | CURVE prediction只覆盖authority sample span；IMAGE只复用已有named radial-Gaussian center/radius projection，不生成预测图；failed/NOT_PRESENT会明确清掉旧成功overlay，绝不复用row 0或邻格 |
| saved-grid 与超大轴 | Overview不隐式挑第一格，只有用户聚焦真实cell后启用Refit；每页最多36格。普通范围用既有spinbox，超过Qt `INT_MAX`改用2048字符上限的exact decimal/hex editor；Previous/Next按`AxisLayout` physical order而非数值排序。坐标、summary、page label与错误诊断均有字符/内存上界 | 20,001-bit index、100,000字符坐标、sparse hole、物理顺序逆于数值顺序均通过；不存在Python大整数转十进制限制把领域`IndexError`替换成ambient `ValueError` |
| 单一总内存包络 | source inspect/materialize、binding、solver、result、DataFigure、typed projection、旧/新overlay、raster/Qt detached front、authoring option与export都从调用者的一份aggregate limit扣除。阶段互斥才取`max`，同时存活必相加；DataFigure frozen render limit只是总预算中的不可放宽子上限。saved-grid在panel DTO/label/selection创建前先做projection upper-bound，导出只对已保留panel计Agg增量 | `required-1`在source decode、Fit bind/execute、overlay prediction、saved-grid projection与raster前拒绝；metadata增量不能抹掉64KiB raster scratch。U0.3a-c旧测试曾把raster增量当窗口总额，本轮改为分别验证session aggregate与render sub-limit |
| 并发、CAS 与关闭 | worker返回值不能用自己携带的旧token自证：Qt从当前frame重建bounded provenance token，稳定source与可变fit join分开比较，并用immutable object identity验证exact evaluated owner；display字段做有界二次核验，不在GUI线程扫描整轴。revision/request CAS、fresh-turn completion handoff与DeferredDelete charge继续生效 | 对替换values、axis/frame、viewport/home range、request sequence和join的反例全部保留旧front。关闭先撤Qt front、取消late present，再清renderer/commit closure/fit bindings；双active Fit + 双concurrent close安全drain，active Fit内reentrant close明确拒绝，关闭后无DataFigure/array残留 |
| 删除与下一真实consumer | Capture-only artifact/ref/draft名、独立Fit host和旧manifest项均dependency-closed删除，没有保留adapter、alias、deprecated branch或“以后也许用”的零consumer路径 | 旧Live/GridPlot/TaskConsole仍拥有尚未迁走的live、Logic、workspace及完整panel consumer，故不能提前删；下一纵切只把TaskConsole `Add Analysis -> Fit`接到同一FINAL-artifact host，不再铸造Fit便利窗口或预建formal AnalysisStep |

U0.3d Rule-6按45个实际受影响production Python owner相对parent `9150a92`机械计数：`32376 -> 38673 physical`、`29880 -> 35788 nonblank`、`29404 -> 35032 token-NCLOC`，净`+6297/+5908/+5628`，完整dependency-closed closure为`1.1945x/1.1977x/1.1914x`；classes `115 -> 124`、dataclasses `71 -> 78`、enums `5 -> 5`、functions/methods `1033 -> 1171`。既有owner最大约`2.1x`，没有超过约3倍项；五个新leaf均有已存在、非假设的职责或consumer：`fit_result.py`取代414行Capture-only owner并同时服务Capture/Scan，`fit_curve_projection.py`隔离headless projection/worker materialization，`fit_authoring.py`被统一Figure host消费，`_diagnostic.py`被axis/layout/selection三处共用，`fit_reference.py`一比一取代旧typed ref。三个旧owner同时删除，enum零增长。以main最严格且不含巨大TaskConsole的五个真实oracle文件为分母（`4366/3888/3195 physical/nonblank/token-NCLOC`），本轮净增为`1.442x/1.520x/1.762x`，仍低于3倍；差量支付exact artifact、Scan第二source、multi-batch/sparse grid、强异常安全、异步GUI与可证明内存包络，而不是manager/plugin/workflow engine。

当前活动authority为`109/109`文件、collect `1409`项。一次逐文件fresh-process全量sweep先定位出U0.3a-b-c的17项真实/过期合同混合失败；修复current-frame CAS与close closure后，三文件分别`12/12、10/10、12/12`。最终独立审查又复现“viewport rerender返回自洽但非提交态的Fit identity”这一P1，补上提交态identity CAS及直接反例后，所有直接消费`_figure`的W4/U0.3d闭包按文件独立复跑为`43/43`。最终账本为`1406 passed + 3 expected platform skipped = 1409`，零failed/error/xfailed/xpassed；owned compile、fresh headless/import-DAG `25/25`、旧Fit符号扫描与`git diff --check`通过。多路独立复核对大整数导航、projection preflight、Fit close drain、动态overlay identity和saved-grid内存最终均判`P0=0/P1=0`；该checkpoint后的TaskConsole入口由U0.3e承接，generic METER、多cell/多layer与剩余archive plot kind仍未完成。

#### U0.3e TaskConsole FINAL-artifact Fit entry（收口状态：`COMPLETED`）

本纵切先以`main@6c337d49`的TaskConsole `AnalysisControls/_FitFixSeedEditor/_apply_panel_analysis`为用户入口与Setting/Edit行为oracle，再以current U0.3d的`DataFigureWindow + FitAuthoringPane + FitResultRepository`为唯一实现owner。三条候选路线经过本地与两路独立对抗：A是“当前card的FINAL ScanArtifact -> 同一DataFigure/Fit host”，B是“FINAL artifact -> 独立flat analysis Run”，C是“Scan Run内post-materialization AnalysisStep/composite commit”。当前唯一真实consumer是用户查看已提交scan、明确Fit并显式Save；B没有headless/preset/downstream consumer，C还会推翻Scan Run的单FinalCommit及ambiguous-recovery语义。因此baseline选择A，B只在真实自动consumer出现后启用，C只在领域明确要求scan+analysis不可分割提交时重开。

`TaskConsoleWindow`只新增一个`Add Analysis -> Fit`动作，不建单项catalog class、Analysis definition、preset codec、第二editor/solver/repository/worker/window。`ScanWorkbenchWindow.finalReferenceChanged`在owner-thread每次model reconcile中按精确`ScanArtifactRef | None`变化发出；TaskScanCard再用一层相等gate转发，因此Run start与reconfigure在同一Qt调用栈撤销旧入口，不等待100ms轮询。按钮只在非closing且当前ref存在时启用，点击时仍重新读取card ref；同ref且原DataFigure未关闭时只聚焦，不同ref重新调用既有`Experiment.fit_gui(ref)`。共享Fit pane继续让所有repeat/point/data轴成为具名batch，TaskConsole的SITE/display selection不进入CommittedTransform；求解与Save仍分别由显式Fit/Save动作触发。已打开viewer冻结自己的exact artifact并可独立于console关闭，DataFigure close按U0.3d释放重型owner。

根`task_console.py`同步物理删除旧SignalHub/TaskConsoleState/registry、自校准、`--task/--config/--grid/--no-connect`分支，改为current `connect("virtual", repository=...) -> Experiment.task_console(initial_intent)`，并以bounded window-close后再关Experiment；`task_console.bat`继续只委托该脚本。该launcher当前明确只是virtual current product，不冒充尚未交付的real composition；未知旧参数由argparse拒绝，不fallback。U0.3d已经交付的IMAGE fit-overlay export测试也在本提交恢复为current oracle：导出真实overlay并验证viewport坐标换算、动态clim和failed/sparse cell不画成功geometry，旧“必须拒绝overlay”断言物理删除。

Rule-6按parent`284fe63`的三个实际production owner（`_scan.py + _task_console.py + task_console.py`）机械计数：`2326 -> 2356 physical`、`2138 -> 2170 nonblank`、`2069 -> 2142 token-NCLOC`，净`+30/+32/+73`，完整closure为`1.0129x/1.0150x/1.0353x`；classes保持`5`，dataclasses/enums保持`0/0`，functions/methods`90 -> 96`。根launcher自身由`139/117` physical/nonblank压到`97/84`；其余差量只支付artifact signal、card forwarding、exact source gate、唯一host focus/open与bounded launcher close。六个净函数各自承担不同边界；没有新增class、enum、manager、registry、wrapper、codec、Analysis runtime或约3倍海拔项。

定向current合同为TaskConsole`10 passed`、U0.3c current export`16 passed`、U0.3d direct Scan Fit`4 passed`、统一Figure Fit`9 passed`、Qt widgets`19 passed`与import-DAG`25 passed`；owned `py_compile`与`git diff --check`通过。活动manifest仍为`109/109`文件，isolated staged snapshot完整collect为`1410`项；按每文件fresh pytest process执行得到`1407 passed + 3 expected platform skips = 1410`，零failed/error/xfailed/xpassed。最终独立复核确认immediate rerun/reconfigure/close、重复signal、exact ref、display-authority、唯一host与launcher lifecycle均为`P0=0/P1=0/GO`。这只关闭UX-004与current virtual launcher分叉；real hardware launcher、完整TaskConsole workspace/Logic/其它panel、generic METER、多cell/多layer和剩余archive plot kind仍未完成，完整迁移继续NO-GO。

#### U0.3f generic METER + single-layer multi-cell Figure salvage gate（收口状态：`COMPLETED`）

本纵切在写产品代码前固定两组已经挣得的用户行为。第一组是`main@6c337d49`的`GridPlot/GridCell`：用户先看到同一次compose产生的完整格图，不默认选中第一格；双击一个真实cell才进入该cell的精确focus，Back/Esc回到原缓存overview，focus与overview都能保存当前所见。第二组是旧rolling readout的数值显示：有限有效值按约六位有效数字呈现，单位与值同行；无效值明确显示`invalid`，不能回退到上一个有效值。旧实现没有独立METER设置页，因此本纵切不臆造threshold、ROI、reducer、selector、Fit、Setting/Edit或Interact控件。current的`DataFigure/FigureEvaluator/SinglePanelAggRenderer/QtRasterBoard/atomic export`仍是唯一owner，旧mutable Figure、Hub latest、GridPlot类族和并行panel renderer不迁入。

范围刻意合并为一个最小完整consumer闭环，而不是先提交无真实入口的单格METER：generic单格METER获得typed `MeterPanelPayload`，逐值绑定exact `EvaluatedInput`、presentation revision、完整`EvaluatedSeries`与`value_unit`；camera live ROI METER改为消费该payload，并继续与同board的curve/histogram共享一个coherence stamp。occupancy默认的SITE-faceted、一层多cell METER是首个真实grid consumer：overview由一次worker-side Agg draw同时产生紧凑immutable PNG和精确`FigurePanelRegion`；只有命中真实region的显式双击才以原冻结`DataFigure`中的exact evaluated input/series派生一个单cell display view并进入现有typed board，绝不重新resolve latest、暗选第一cell、flatten、reduce或重算物理结果。focused view必须使用内容派生的新document identity，不能用同一id/revision表示不同内容；selection只属于display state，不得生成`CommittedTransform/FitSpec`。

线程、生命周期和预算继续沿U0.3a-e的单owner合同：整个overview在capacity-one Workbench raster lane一次compose，focused METER也在同一lane渲染；Qt只持immutable encoded/raster front、same-draw hit regions和精确payload。overview保持缓存，Back/Esc不重读repository、不重算DataFigure；overview与focus导出都直接写当前已接纳front，经同目录stage、close/export commit lock与原子replace，不为导出重画。focus准入必须在构造panel DTO、Agg raster和Qt decode之前按`required-minus-one`拒绝，并把仍同时存活的frozen DataFigure、overview encoded bytes/QImage/regions、旧front与候选front全部计入一份总包络；失败保留原overview且不出现半切换。Close先撤可见front、取消late-present并释放overview/focus缓存，已开始的worker诚实排空但不得发布。

本checkpoint当时的边界只到“单layer、多cell、全部真实panel为METER、无fit overlay”。多layer当前没有production producer，只有codec/value测试；其overlay、z-order、共享轴和跨layer分析语义都未成立，因此继续使用完整whole-figure静态回退，不产品化dashboard。faceted HISTOGRAM后来由U0.3g、faceted CURVE后来由U0.3h各自以独立真实consumer纵切完成；faceted IMAGE仍须等待自己的consumer，不能从METER/HISTOGRAM/CURVE外推。U0.3f验收覆盖单位不丢、valid/invalid/nonfinite、live meter exact payload/coherence、multi-site overview无默认focus、精确第二cell命中、sparse hole不前移、Back/Esc使用缓存、overview/focus原子导出、预算差一字节在重型分配前拒绝、Close期间无late-present，以及multi-layer仍走静态fallback。

**U0.3f实现checkpoint（`GO`）：** `EvaluatedMeter`现在原样保留schema的`value_unit`；`MeterPanelPayload`把exact `EvaluatedInput`、presentation revision、完整`EvaluatedSeries`和统一unit作为封闭typed payload，同时被frozen Figure与真实camera live board消费。数值格式化只在显示层发生：有限值约六位有效数字、bool为`true/false`、invalid永远显式；valid nonfinite在持久Agg surface变更前拒绝。单layer METER的canonical panel顺序直接来自`EvaluatedLayer.cells`；overview PNG和hit regions来自同一次draw。focus以same-draw `Selection`复核panel地址，把FACET改成明确`FixedIndex`显示选择，复用原exact series，不resolve/re-evaluate/reduce；派生document identity同时包含原document、facet地址与owner序列化的exact `DatasetRevisionRef`，不同revision不能冒充相同内容。

预算P1在终审中被真实反例关闭：worker先从冻结grid对目标cell做metadata-only retained bound，再把缓存的full DataFigure、overview PNG/QImage/regions、旧/候选RGBA front和Agg峰值放进同一aggregate admission；render sub-limit或aggregate `required-1`均在调用`focused_typed_panel()`、构造派生Figure DTO、Agg artist或Qt decode之前拒绝。派生后再比较actual retained与preflight bound。Back/Esc只切回缓存overview；focus运行中Esc或Close通过sequence/cancel/closing gate禁止late-present。overview与focus export都逐字写当前已接纳front，经stage/commit-lock/atomic replace，不重画。

Rule-6按parent `1150e0c`的9个实际production owner机械计数：`16881 -> 17894 physical`、`15685 -> 16652 nonblank`、`15556 -> 16520 token-NCLOC`，净`+1013/+967/+964`，closure为`1.060008x/1.061651x/1.061970x`；classes `87 -> 90`、frozen dataclasses `65 -> 68`、enums保持`7`、functions/methods `521 -> 541`。最大owner仅`data_figure.py 1.261x`与`_figure.py 1.170x`，无约3倍项。相对main真实`GridCell + _GridData + GridPlot`行为oracle，本轮净增为`0.951174x/0.977755x/1.199005x`；三个新增dataclass分别由live+frozen+Qt payload、worker/Qt state CAS和same-draw overview retention实际消费，没有第二renderer/selector/window/manager/registry/compatibility wrapper。

最终focused oracle为`10 passed`，U0.3a-f直接闭包为`69 passed`，frontend Figure为`34 passed`，import-DAG为`25 passed`；owned compile、fresh headless import、diff-check、current-format ratchet与新增shape-loss扫描通过。活动manifest为`110/110`文件、完整collect `1420`项；按process-lifetime installation边界逐文件新进程执行为`1417 passed + 3 expected platform skips`，零failed/error/xfailed/xpassed。两路独立终审先后抓到Histogram property owner错位、focus identity漏exact ref、O(N²)/全grid focus路径、冗余预算DTO以及派生DTO早于预算拒绝等问题；全部以反例整改后结论均为`P0=0/P1=0/P2=0/GO`。该checkpoint只关闭generic单panel METER与单layer METER multi-cell explorer；其后的faceted HISTOGRAM由U0.3g单独关闭，multi-layer、faceted CURVE/IMAGE、calibration/site-map report及完整迁移继续`NO-GO`。

#### U0.3g SITE-faceted HISTOGRAM Grid salvage gate（收口状态：`COMPLETED / GO`）

本纵切的真实current consumer不是虚构的通用dashboard，而是`OccupancyArtifactRef`选择`occupancy_output="counts"`后由role-driven `suggest_view(HISTOGRAM)`产生的SITE-faceted多cell `DataFigure`。`main@6c337d49`的严格UX oracle是`SiteHistogramGrid/GridPlot + HistogramFigure`：首次显示同一次compose的完整site grid，不默认进入第一site；双击一个真实cell才进入该site的standalone Histogram，空白或稀疏hole不得把后续site前移；focus保留range/cross/zoom/pan/home、bins、linear/log count、normal/tight/fixed relim、Setting/Edit与当前front导出；Overview/Esc回到原缓存整板且不重新读取artifact、不重新evaluate。当前occupancy counts未声明threshold、fidelity或domain fit，因此本切片不得从旧calibration histogram臆造这些authority；它们只能随携带相应typed模型/报告的真实consumer迁入。

实现边界固定为“单layer、单input、多cell、每个cell全部为同value-unit `EvaluatedHistogram`、无fit overlay”。U0.3f已经挣得的same-draw overview/region、显式命中、缓存返回、原子导出、capacity-one worker lane与aggregate budget继续作为唯一外壳；现在因METER和HISTOGRAM已有两个真实consumer，METER专名的focus DTO/派生函数收敛成一个封闭的typed-panel focus owner，并由调用者显式给出期望intent，不保留旧名alias，也不顺带开放尚无consumer的faceted CURVE/IMAGE或multi-layer。focus只把FACET地址改成display-only `FixedIndex`，逐对象复用原`EvaluatedSeries`、`EvaluatedInput`、ComponentValidity结果、sample coordinates、dropped count与value unit；不得resolve latest、重算物理样本、flatten、reduce、按shape猜site或把display selection升级成`CommittedTransform/FitSpec`。Histogram的显示bin counts可由这些exact samples按冻结的grid display domain重算，但不能作为新的measurement/analysis authority。

线程与内存验收沿U0.3f加严到交互focus：在派生单cell DTO、共享bins、Agg artist和Qt front之前，以metadata-only bound同时准入冻结full `DataFigure`、overview PNG/QImage/regions、focused DTO、旧/候选RGBA front、Histogram bin projection与Agg峰值；render sub-limit或aggregate `required-1`必须在重型分配前拒绝并保留overview。进入focus后所有Setting/Edit和selector rerender仍走同一worker owner、同一typed front CAS与同一Histogram display revision；Back/Esc在rerender运行中撤销late-present并恢复缓存overview。Close同样先撤发布、排空既有work、禁止late-present。overview与focus export只写当前已接纳front，经同目录stage、commit lock和atomic replace，不因export重画。

本切片不触碰正在并行修改的calibration report，不删除仍有TaskConsole/archive/live consumer的旧`GridPlot/live.py/task_console.py`，也不建立TaskConsole Logic/Measurement lifecycle。收口oracle覆盖：真实occupancy counts入口、exact sample/validity identity、overview无默认focus、空白拒绝、第二site focus、全部Histogram交互和Setting/Edit、Back/Esc缓存、overview/focus原子导出、焦点切换与close期间无late-present、预算差一字节早拒绝，以及multi-layer继续完整encoded fallback。

**U0.3g实现checkpoint（`GO`）：** `DataFigure.focused_typed_panel()`在当时是METER/HISTOGRAM唯一focus派生owner：它以same-draw `Selection`和显式expected intent复核地址，把FACET固定为display-only `FixedIndex`，复用exact series/input/revision且不resolve/re-evaluate；旧METER专名API及alias为零。Workbench只在单layer、单input、同unit、无fit的封闭集合开启typed grid，overview没有默认cell，空白/hole不命中，Back/Esc只恢复缓存front；U0.3h后来把CURVE加入同一封闭owner，multi-layer与faceted IMAGE仍走完整whole-figure fallback。

`main`的关键可比性语义也已恢复而没有复制旧renderer：整个Histogram grid由现有whole-figure Agg owner对所有site的exact samples冻结同一bin domain，并以全grid最高bin count统一y轴；focus把同一domain作为grid-owned Home传给现有single-panel Histogram owner，以当前bin count重新闭合构造counts/edges。domain不是用户`x_view`，所以初始viewport仍为auto，zoom/pan后Home回到grid域；`HistogramPanelPayload`继续强制home等于实际bin edges，caller不能独立注入counts/edges。旧实现的quantile裁边没有保留：新owner用包含全部有效sample的min/max domain，任何样本落在显式domain外都fail-closed。

两轮独立对抗审查先后复现并关闭三个material问题：focus后的Setting/selector rerender曾漏算仍存活的full DataFigure与overview PNG/QImage/regions；共享domain曾在首道aggregate gate前派生；只改viewport home又曾破坏payload的home/bin-edge不变量。现在grid session retention进入每次rerender与下一次reservation，返回overview或Close前都不假定释放；overview `required-1`先于domain扫描/Agg，focus `required-1`先于派生DTO/Agg，rerender `required-1`先于新renderer；共享range则由frontend owner按exact samples派生并驱动不可注入的projection。终审结论为`P0=0/P1=0/GO`。

Rule-6相对parent `59d4271`对四个实际production owner机械计数：`8040 -> 8453 physical`、`7494 -> 7892 nonblank`、`7197 -> 7589 token-NCLOC`，净`+413/+398/+392`，closure为`1.051368x/1.053109x/1.054467x`；classes `14 -> 15`、frozen dataclasses `10 -> 11`、enums保持`1`、functions/methods `228 -> 234`。最大owner约`1.063x`，无3倍项；新增edge/domain、shared draw和geometry budget函数均有两个真实consumer或阻止workbench复制frontend规则，唯一新`_GridFocusRequest`跨Qt pending/CAS/worker原子携带selection、intent、display与domain。没有第二renderer、selector、state machine、manager、registry、compatibility wrapper或零consumer泛化。

最终focused oracle为U0.3g `11 passed`、U0.3f `10 passed`、U0.3a `12 passed`、Histogram owner `6 passed`、M2c `10 passed`、W4 `24 passed`，合计`73 passed`；真实virtual E2E从sitemap calibration、readout capture、detect得到`OccupancyArtifactRef`，无selection调用`figure_gui(..., occupancy_output="counts")`形成35-cell HISTOGRAM grid，再focus第二cell并逐值复核samples、dropped count与exact revision ref。活动manifest为`111/111`文件、完整collect `1431`项，按process-lifetime installation边界逐文件新进程执行为`1428 passed + 3 expected platform skips`，零failed/error/xfailed/xpassed；owned compile、fresh headless import、import-DAG、manifest gate、current-format ratchet、shape-loss扫描与diff-check通过。该GO只关闭真实SITE-faceted occupancy counts Histogram；faceted CURVE后来由U0.3h关闭，calibration/site-map report、faceted IMAGE、multi-layer与完整迁移继续`NO-GO`。

#### U0.3h ScanArtifact faceted CURVE Grid salvage gate（收口状态：`COMPLETED / GO`）

本纵切由已经存在的current producer驱动：AUTONOMOUS occupancy scan提交的`ScanArtifactRef`保留`(R,P,SITE)`，普通`Experiment.figure/figure_gui(scan_ref)`的role-driven建议把SCAN_POINT放在具名X、repeat显式display mean、35个SITE放在FACET，因而自然产生多cell CURVE；这不是为了补齐`ViewIntent`枚举。`main@6c337d49`的严格显示UX oracle是`LineCell + GridPlot`：完整grid从同一次draw得到真实panel与命中区域，不默认进入第一cell；所有thumbnail使用同一X与全cell有效样本的共享Y域以保持可比；空白与sparse hole不前移；显式focus后复用普通standalone Curve的zoom/pan/range/Home、TIGHT/NORMAL/FIXED、Setting/Edit与export，Overview/Esc返回原缓存整板。

这里必须忠实保留main的一个容易被“统一”误改的细节：共享Y只属于可比的overview。`LineCell.prepare/draw`池化全部cell得到共享Y，但`focus_data()`只把该cell的x/y交给普通1D panel；因此focus按该cell自己的标准relim取得局部细节，不能把grid共享Y伪装成用户 authored fixed range。只有用户显式提交的Curve display状态才改变focused front；Back仍恢复未重算的共享Y overview。

实现边界冻结为“单layer、单input、多cell、无初始fit overlay、每个cell至少一条real numeric `EvaluatedCurve`，全grid共享逐值相同的`EvaluatedAxis`与value unit”。共享Y只读取ComponentValidity已经投影出的逐series validity；invalid不参与，valid nonfinite/complex fail-closed，全invalid grid使用确定性空域；不得flatten、取首cell、平均SITE、按shape猜axis或把FACET focus升级为authority Selection。focus只把same-draw FACET地址改成display-only `FixedIndex`并逐对象复用原`EvaluatedSeries/EvaluatedInput/DatasetRevisionRef`。

本切片**明确不开放grid-focus Analyze**，并显示display/authority原因：真实入口的focused曲线已经做了`repeat=mean [display]`，而当前`suggest_fit_draft`正确地把repeat与SITE原样保留为batch；一个均值曲线无法一一对应两个repeat的Fit rows。放宽overlay校验会把batch结果静默贴到均值数据，复制display reduction进FitSpec又违反类型防火墙。故此处不把main旧显示层的in-place fit冒充权威分析；用户仍可使用已有轴完整的direct Fit/TaskConsole入口。只有出现真实consumer并设计“用户显式提交的authority repeat reduction”，或从权威source重建repeat-visible Fit视图后，才可重新开放该按钮。带已有fit结果的saved grid继续由W7 owner处理，不能在本切片造第二saved-fit路径。

U0.3f/g已经挣得的capacity-one worker、same-draw regions、显式命中、缓存返回、原子current-front export、typed CAS与aggregate budget是唯一外壳。本切片只把CURVE加入该封闭集合，并让现有whole-figure Agg owner在首道metadata/peak准入之后、无拼接大数组地计算共享Y；focused CURVE继续使用现有`SinglePanelAggRenderer`、`CurveDisplayState`、`QtRasterBoard`与Setting/Edit DSL。预算必须同时覆盖full DataFigure、overview PNG/QImage/regions、focused exact curve arrays/DTO、旧/候选RGBA front和Agg峰值；`required-1`在focus派生或Agg前拒绝。focus/rerender期间Back/Esc/Close撤销late-present并保持原overview，不能增加renderer、selector、window、manager、registry、兼容alias或通用plot-kind插件。

收口oracle至少覆盖：真实virtual occupancy `ScanArtifactRef -> figure_gui`默认35-cell CURVE；共享X/unit与overview共享Y；无默认focus、空白拒绝、第二cell exact focus、invalid/sparse identity不移位；focused Curve的zoom/pan/range/Home、Setting/Edit、局部relim、Analyze明确禁用及原因、原子export；Back/Overview/Esc缓存；focus/rerender/Close late-present；预算差一字节早拒绝；multi-layer、异构X/unit、初始fit-bearing、faceted IMAGE与其它unsupported kind继续完整encoded fallback。完成current oracle、独立对抗审查、Rule-6与active manifest之前保持`NO-GO`。

### 22.2 终态验收清单

架构：

- import DAG 通过；
- zlc_data、pulse/FPGA、neutral_atom、frontend 与窄 zlc_storage 可按声明边界 isolated import；部署可同 wheel 或按 server/Qt 需求拆分；
- concrete construction 只在 allowlisted bootstrap；
- DefinitionCatalog 由显式 imports 组装，PipelineSpec 只有一个顶层 Run owner；
- zlc_data 不引用 runtime/presentation 类型，frontend FigureDocument 不引用 neutral runtime live ref；
- Fit/Selection/FitResultBatch/BoundFit 只有 zlc_data owner，DataFigure/selector controller 只有 frontend owner；当前不存在DatasetInputSlot/generic AnalysisStep，TaskConsole只把FINAL artifact交给同一Fit host；未来真实自动consumer只能在neutral/composition侧以flat Run托管，不得产生第二Fit owner；
- sample event、materialized dataset、analysis result、presentation 四个平面之间只有声明的三处边界；
- 每个事实有唯一 owner；
- physical device 在 Workbench/notebook/standalone/remote 入口间只有一个process-lifetime InstallationRuntime和backend可验证physical-owner proof；迁移期legacy runtime也不能绕过同一ResourceArbiter、DeviceBroker或startup/shutdown lifecycle gate；
- public Experiment/RunHandle/TaskConsole/PulseGUI/session/DeviceViewer/DeviceManager不暴露raw hardware drive capability；public根对象图中BaseDevice/DeviceSet/SDK handle/BoundDevice/drive-capable Port/internal RunPlan/resolver/drive bound method计数为0；public run/start只收declarative Request；
- Experiment只通过`device_catalog`暴露immutable观察值，并通过timing/readout/trap等typed facade暴露领域descriptor；catalog不含callable、arbitrary snapshot service locator或command；
- 一个进程恰好构造一个InstallationRuntime与immutable InstallationDeviceGraph；startup在开放Run admission前完成journal replay/lock、physical-owner proof、AssetMap、adapter open、identity/bind/probe和graph freeze，任何失败都不发布partial facade；config/device/virtual-real改变只允许safe shutdown+进程退出+新进程重建，旧runtime/ref/command均以零adapter调用失效；
- ordinary umbrella raw hardware/adapter/server symbols与compatibility fallback为0；adapter author、simulation与server入口位于明确owner namespace；
- 旧路径、alias、fallback 为 0。

数据与 Signal：

- 任意 trailing axes 不丢；
- sample stream 只传 Value/typed record，不传累计 DataBlock；finite exact DatasetBuilder 与 live MonitorDataset 各自是互斥的单一 event -> dataset materializer；
- 多维 point axes 只经 PointLayout 映射 P，不退化为匿名 data_points/data_dim；
- cell/component partial validity 沿 reduce/fit/histogram/meter/figure 正确传播；
- scalar 语义明确；
- 已 materialize 的 owned DataBlock 内容稳定 immutable，materializer 后续 ingest 不改变该 snapshot；未显式冻结的旧 ref 返回 SnapshotExpired，不回 latest；live head/EventRefs/coverage 与 block 在同一 snapshot 原子冻结；
- exact source/reservation/Delivery/EOS authority、single materializer、TraceBinding、frozen cell schedule、pin/ack、broker-minted generation 与真实 byte budget 正确；
- CaptureCapability由broker endpoint probe attestation绑定device generation；FrozenCaptureSpec只含owner canonical bytes/digest且prepare无替换/codec回调入口；CaptureSession所有device operation保持同一owner I/O lane；
- BACKPRESSURE_CAPABLE 可零副作用拒绝后重试；NON_BACKPRESSURE_CAPTURED 首次 overrun 或 capture 后处理失败永久 poison epoch，后续 frame 不得补占旧 sequence；
- qCMOS正式scan只在active Q0 CameraExternalTriggerQualification与preflight margin内运行；autonomous单次FIRE或API整run首次FIRE的camera authority与qualification mutation共用跨进程线性化gate，suspension/revocation写失败保持claim并fail closed；所有数据在EndAttestation前provisional，不一致整run拒绝；artifact记录execution_mode、一个run级camera authorization/CameraRunEvidence、API模式下另有R-major/P-fast的有序pulse segment evidence、required/achieved association proof与formal eligibility及其非逐沿剩余风险；
- formal epoch的PROVISIONAL/VALID/INVALID由独立immutable validation record解析；PROVISIONAL只可带徽标显示，不能经普通Figure/Fit/derived save逃逸为成功权威artifact；
- repeat轴已展开进finite single-pass table，Formal路径不存在`scan_repeats>0` host wrap-stop；
- coherent monitor 不混 shot；
- MonitorTap backlog 与 MonitorDataset window 都有独立 byte/cell budget；keyed cycle 不混 sweep，append window newest-first且不丢 trailing data axes；live 只产生 MonitorDatasetSnapshot，storage/权威 processor 类型上只接受 SealedDatasetArtifact 或 VALID epoch wrapper；
- artifact/result 原子提交；
- 自动视图只按 axis metadata + ViewIntent 决策，不读 values 猜语义；
- 显示投影始终可见、可改、可复现，且不修改原始 DataBlock；
- DataTransformSpec/CommittedTransform 与 figure ViewSpec 分层，display-only step 不能进入 fit、正式 scan y、calibration 或派生 artifact；
- Fit batch 只报告 numeric status、counts、RSS/R² 与 covariance validity；科学可用性由实际消费参数的领域 AnalysisSpec 显式判断，generic core 不保存统一 acceptance；
- fit initializer只产seed，数学domain/用户constraint才产hard bound；absolute-time参数不被选区contrast截断，phase只有`[-π,π)`单一表示；
- FitProblem 不保存 sampling quantum 或逐样本 logical index；damped-sine只输出`baseband_frequency`，formal物理解释另需typed sampling/band-limit prior；RMSE及其它可派生metadata不重复持久化；
- histogram pool、image repeat、fit batch 等语义由 ViewContract 声明，没有 render 特例或全局 repeat mean。

执行与线程：

- GUI thread 无阻塞 hardware I/O、plan/pulse compile、transform 或 fit；
- cancellation 不谎报 terminated；
- HAZARD_ACTIVE durable前cancel不调用任何可能触碰硬件的interrupt/abort/safe；interrupt in-flight阻止cleanup并发、terminal与claim release；
- final artifact publish与cancel共享短原子gate，cancel-first无artifact，commit-first不谎报CANCELLED；
- join timeout 不释放 owner，safe failure quarantine resource；
- safe/unsafe/mixed multi-device safety disposition由同一ResourceLease收集为每run唯一、幂等的SafetyDispositionBundle并提交同一SafetyJournal；journal失败保留全部claims并只重试缓存的同一bundle，artifact outcome之后的terminal publication与全部claims释放线性化；
- safety bundle durable前没有外部terminal snapshot；正常connection只在Run admission前建立且没有ResourceArbiter connection lease，只有exact RecoveryAttempt持窄recovery claim并与普通run互斥，quarantine不存在旁路或吸收态；
- live identity evidence区分硬件标识readback与installation-asserted endpoint，后者不冒充物理板卡readback并在artifact保存剩余风险；connection generation由DeviceBroker在startup/recovery成功握手后签发；active Run断线不透明跨generation重连，safe intent或软件state不能冒充硬件safe proof；
- safety bundle durable后所有device capability不可逆撤销，post-safety fit/save/cancel不能再触碰硬件；
- device/session/Figure thread affinity 与单 owner 正确；S0.5 legacy handoff bridge 已删除；
- BoardFrame coherence 使用完整 run/key/dataset/generation/schema/revision identity；BorrowedSnapshot 在所有 discard/close 路径释放；
- cleanup failure 可见；
- shutdown 无存活 worker、未释放 opt-in borrow token 或未执行 safe action；DeviceBroker先失效、唯一InstallationDeviceGraph按reverse close order完整关闭、lanes停止、PersistentSafetyJournal/physical-owner proof最后释放，未达到CLOSED时不得启动新config进程；
- RunCoordinator 不成为第二 lifecycle owner，EditorSession 不覆盖新 revision。

Artifact/Calibration：

- manifest commit marker 前不存在成功 artifact，load 全量验证 manifest/blob；
- Figure Save 冻结用户所见 revision并验证epoch authority eligibility，frontend artifact 不序列化 LiveDataBlockRef；PROVISIONAL只允许显式`DIAGNOSTIC_PROVISIONAL`水印快照；
- failed/cancelled formal run 不产生成功 artifact；
- live/offline calibration 在 CaptureArtifactRef 后走同一算法路径；
- FrameContract/SiteMap/model applicability 不匹配明确失败。
- calibration 默认按 ReadoutBindingKey 与声明的 model policy选择，site-map-only/完整 model capability 不混淆。
- NumPy/SciPy 版本只在 CalibrationReport 作为被动 lineage；版本改变不改变 request、SiteMap、模型或准入，load/admit 不探测当前数值环境。

Pulse/FPGA：

- Formal PulseScan在软件reservation/cursor/processor/collector层端到端exact；相机frame↔point按active Q0-qualified ordered external-trigger contract与整run末端对账成立，不声称逐沿硬件证明；
- 精密scan/trigger时序由冻结bitstream和qCMOS硬件执行，host只做run前margin验证和run后attestation；
- 可观察的gap/duplicate/out-of-order/count/stamp/timestamp/EOS不一致使整run INVALID且不提交；accepted equal loss+extra residual risk在provenance可见；
- TriggerKey/ScanCellKey/ScanArtifact 保留多维 point/output axes；
- S4近期enablement只接受恰好一个Q0-qualified qCMOS physical source并使用现有FPGA完整自主scan table；neutral按frozen schedule + ordered camera frames映射TriggerKey，末端验证后才转VALID；source-neutral ScanPlan不自动授权尚无专用association/terminal contract kit的其它source；
- raw STATUS/CURSOR只证明logical table terminal；camera/drain必须保持运行到compiled/H1 post-terminal tail bound完整经过并形成`PostTerminalTailEvidence`，再执行Q0 quiet-window/final metadata/cap_stop/release；host wait只防过早termination，不调度edge；
- camera max-inflight ring与host total frame/byte retention分别在fire前证明；近期Formal强基线为resident，超resident默认拒绝，只有单I/O owner、保守refill硬上界和每个潜在seam均有足分辨率硬件时间观测/全schedule residual时才开放条件能力；非sticky underflow、DONE或局部timestamp不算无缝证明；
- MOT只使用SCAN_SLOT + AUTONOMOUS_STREAMED；无API或host-stepped fallback；
- API segmented只有在program的canonical非空`segmentation_rationale`说明host boundary物理可接受，且use case接受任意可变非负gap、不依赖最大gap/精确settle/连续时间或gap-dependent physics时运行；不建立单字段policy wrapper。整run一次camera arm/aggregate terminal，按R-major/P-fast为每cell使用独立STATIC_ONCE PreparedProgramRef/PulseSession并取得physical pulse terminal（CURSOR=N/A）；相邻segment把前段trigger→physical terminal/tail与后段硬件trigger prefix计入硬件已保证区间，host只deadline/cancel-aware等待camera required interval尚未覆盖的安全余量，不承担精密edge调度。ScanRepository先在point resolve/compile前执行R×P cardinality floor，再在真实program/source/output/compiled/camera/point-shape冻结后且Run/arm/FIRE前mint实例绑定的metadata admission receipt；FINAL canonical encoder只能在该receipt的byte/node上界内运行，codec或事实漂移按内部不变量失败。全部R×P evidence与唯一CameraRunEvidence只进入一次aggregate EndAttestation；INVALID attempt可见且不会被无限自动重试隐藏；
- build/target digest在compiler、repository与发布记录之间形成可追溯闭环；installation-owned ProgrammedImageDeploymentRecordRef把endpoint、冻结`.bit` digest、release/timing records和现有fingerprint绑定进H1/Q0/FIRE/tail/artifact lineage；
- runtime现有`image.build_fingerprint`/几何/ABI握手闭环；新ROM attestation不是baseline；
- runtime不声称验证当前bitstream content digest、implementation seed或timing-signoff token；
- PreparedProgramRef + connection generation + artifact/table digest使旧软件引用不能进入正式FIRE，完整自主table在fire前冻结；
- current resident/streamed bank按现有RTL真实能力完成golden/真机验证，不要求不存在的RTL CRC；
- current `RemotePulseExecutionClient/RemotePulseExecutionEndpoint` 的logical deadline覆盖SAFE lock与RPC，timeout撤权且不并发第二次SAFE；server以唯一backend-operation gate线性化prepare/FIRE/complete与physical SAFE，SAFE先发布INTERRUPTING并发出thread-safe non-blocking interrupt，再排到已入硬件操作之后，任何superseded返回都不得mint ref/ack/terminal；真实composition、backend bound、cancel/safe与quarantine仍须在对应SafeStateContract中通过真机recipe，不要求新watchdog/独立SAFE硬件；
- RTL/bitstream保持冻结；只有E0a/Q0在已批准工作余量、正确camera配置和充分软件reservation下仍实测loss/reorder，且camera设置、软件保留/排空策略、降低trigger rate与扩大margin均不能修正；或已证实RTL bug/设计偏离，才允许H2硬件修复评审。

性能：

- queue/materializer commit 摊销 O(1)，compile/journal/artifact 对数据量近似 O(N)；
- exact retained bytes、view/Fit queues 与 UI stale work 全部有界；
- profile matrix 的 p95/p99 latency、peak RSS 与输出等价 gate 通过。

用户体验：

- launchers、主要窗口和核心操作流程保持熟悉；
- Fit 可见且可从真实入口完成；
- 非标量数据打开即有合理默认视图；只有真实歧义才要求最小必要输入；
- panel 始终显示当前 axis/selection/reduction/batch 摘要；
- monitor 显示实时且 coherent；
- Stop、冲突和错误状态诚实；
- virtual 与 real 在各自进程startup选择backend后走同一request/run/data路径；不提供进程内切换；
- notebook 的 connect/capture/fit/save 仍保持短路径，不暴露内部 PipelineSpec/Port ceremony；
- UI 视觉和交互能力不因架构拆分退化。

最终证据与交付：

- `tests/migration_active_tests.txt` 的全部路径经conftest机械授权，完整`--collect-only`与完整实跑全绿；清单外历史测试没有被运行或改写；
- 对最终分支只做一次有界独立对抗终审，结论必须为P0=0、P1=0；P2逐项给出触发条件、owner与后续动作，不能用“未来再看”隐藏；
- 《迁移完成报告》候选稿首页列§2.2全部未批准UX偏离，随后给最终commit、包/能力状态、删除账、规则1–9证据、性能结果、已知限制和实机未资格化项；只要仍有未批准偏离，它就是待用户裁决的候选报告、Rule 9仍PARTIAL且迁移NO-GO；用户批准或实现恢复parity后才可签为终版；
- 《真机bring-up runbook》给出上电/唯一owner/AssetMap与fingerprint检查、camera工作点与Q0资格化、virtual scan、resident Formal、EndAttestation/reject-and-redo、安全停止/断线/quarantine的顺序、命令、预期结果、通过/失败判据与恢复动作；
- 用户上机第一天清单按“真实launcher GUI日常流程 -> virtual scan/fit/save/reopen -> qCMOS只读identity/working point -> 已批准qualification -> resident真机Formal”排序，任何fail-closed gate不得靠手工改状态绕过；
- >4096/9999点扫描若在交付时仍受真机资格化约束，runbook必须单列`AUTONOMOUS_REFILLED`启用步骤：冻结全schedule/chunk digest、单I/O owner、refill硬上界、每个seam时间观测与residual证据、配置/命令、9999点压力实验、underflow/gap/late-chunk拒绝判据、关闭条件与resident回退。host只供应预先冻结chunk，FPGA继续决定全部精密edge时序；未通过时typed拒绝而不退化为host stepping。
  - **状态 `COMPLETED`（typed fail-closed 部分）：** 拒绝已从裸 `ValueError` 升级为 §15.4 要求的 `FormalScanCapacityExceeded(requested_points, resident_limit, capability_unavailable_reason)`（`zlc_pulse/deployment.py`，仍继承 `ValueError` 以免既有 caller 语义漂移）。`AUTONOMOUS_REFILLED_UNAVAILABLE_REASON` 是缺失证据的**唯一**表述，逐条点名三个未闭合 gate 并明说「冻结 bitstream 本来就有 ping-pong refill 硬件，缺的是证据不是硅片」，避免误读成硬件能力不足。两处并列判据（展开前 logical R×P 与 deployment 边界）此前各写一份措辞，现已收敛到同一异常类型与同一常量。资格化实验 R1–R6、启用/回退与「host 只供应冻结 chunk」不变量写入 `docs/REAL_HARDWARE_BRINGUP_zh.md` §5，`tests/test_u04_scan_capacity_refusal.py` 机械守卫「两处判据不得给出两个理由」与「runbook 不得把 host stepping 写成出路」。**剩余为纯真机部分**：R1–R6 证据与常量翻转。

#### 清单4 TaskConsole 完整迁移 salvage gate（开工冻结证据，状态：`OPEN`）

**exact oracle 口径（先于任何实现固定）：** 本清单每一条 `main` 行为一律引用 `git show main:<path>`，`main` 头为 `6c337d4`，`Zou_lab_control/frontend/task_console.py` 为 `9461` 行。**本分支 HEAD 上的同名 legacy 文件已被迁移改动（`10036` 行），它不是 oracle**；任何 salvage 引用若落在 HEAD 副本上一律作废重取。规则8 口径同时固定（**2026-07-20 用户指令改写执行节奏，判据不变**）：清单4 的每个 commit 仍以完整活动白名单为判据、错误清单只减不增，这条覆盖「只跑改动边界测试」的一般习惯,因为清单4 每片都会同时触及多个已在白名单内的窗口 owner。**执行方式改为单进程**:(a) 每轮 commit 前只跑两样,各一个 pytest 进程——本轮改动边界的测试文件 **加三个棘轮**(`test_z0_zero_residue` / `test_u05_shell_salvage` / `test_migration_manifest_gate`)合并一条命令;以及全清单 `--collect-only`(守 collection 层,秒级)。(b) 完整白名单**实跑**只在切片收口(账本一行翻 `COMPLETED`)时做,**分两组**:

    - **Group A(当前 104 个文件)= 单进程** `xargs pytest`,约 100s。
    - **Group B(当前 25 个文件)= 每文件一个进程**,因为它们各自建 installation runtime。

    这个二分**不是效率取舍,是架构强制**:`zlc_neutral_atom/bootstrap/_installation.py:124` 规定「this process already created an installation runtime; replace the process instead of rebuilding or hot-swapping it」,即 S0.6 的 **process-lifetime InstallationRuntime**。首次按「全清单单进程」执行时实测**12 failed / 98 errors**,其中 **108 个 error 全是这一句**——把 129 个文件塞进一个进程等于要求一个进程建 25 个 runtime,违反的是产品不变量而不是测试卫生。分组后 **129/129 全绿**(A 组 1303 passed 单进程,B 组 25/25)。

    Group B 的成员判据 = 该文件会构造 installation runtime;当前靠一次全量单进程跑把抛出该 RuntimeError 的文件筛出来,**尚未机械化成 manifest 的一部分**,这是已知缺口,留待后续切片(否则新增一个建 runtime 的测试会静默落进 A 组并让整组变红)。

    这次改写不是放松,两种执行方式证明的是**不同**的事,必须如实记下:逐文件新进程证明「每个测试文件自足」,单进程证明「整套测试彼此相容」。旧口径掩盖了后者——`test_every_name_resolves_to_the_identical_object` 在逐文件下长期为绿,A 组合并单进程后立刻实锤 `core.fitting._solve_thread_guard` 的 shim 快照永久陈旧(见 S5-shell(l))。所以合并**不是**为了省时间才做的妥协,它本身就是一种旧口径拿不到的证据。代价是**不再机械保证每个文件可独立通过**:某文件若因别处的 import 副作用才变绿,单进程不会报。该风险由收口 gate 的 `checkout-index` 干净导出树 + `-p no:cacheprovider` 部分抵消,若将来出现「单独跑红、合跑绿」的文件,按本段记为已知缺口而不是假装等价。

**开工时的真实闸门（已逐条机械复核，非转述）：** `Zou_lab_control/workbench/_task_console.py:1231-1241` 在建第二张卡时 `raise RuntimeError("TaskConsole currently owns exactly one card")` 并在建卡后 `self._add.setEnabled(False); self._catalog.setEnabled(False)`；`zlc_workbench/task_console.py:89-96` 只合成三条 Definition，`:106-117` 在 `actual != expected` 时直接 `ValueError`，因此**新 Definition 目前不是“零 GUI 改动即入 catalog”，而是会硬报错**。`tests/migration_active_tests.txt` 有 `tests/test_w3_task_console.py`，但 `main` 的 `tests/test_task_console_*.py` 七个旧守卫**不在活动白名单**，所以旧行为在新树当前**没有任何机械守卫**：本清单每片必须自带等价守卫，否则“不缩窄用户面”只剩人眼。

**旧行为清单（49 条，逐条 `main@6c337d4` 锚定，按纵切分配）：**

| 组 | main 真实行为（收口判据） | 归属切片 | 开工状态 |
|---|---|---|---|
| 窗口骨架 | header(9 件控件，`_lockable_header` 恰为 kind_combo/Add/Save image/Save/Load/name_edit，Pause/Selectors/Devices **故意不锁**) + 常驻状态条 + Monitor/Logic 两个不可关 tab + Edit 一律可关；embedded 模式的裁剪面（无 Logic tab/无 Pause/Save/Load/Devices，改 setMinimumSize 让重力板反流多列）(`task_console.py:6603-6636/6647-6754/6773-6832/6490-6496`) | S4-1(骨架)、S4-6(状态条) | `MUST_CLOSE` |
| Add Panel = **4 类** | Plot(六种 `panel=True` kind) / Measurement(首项为 `camera_spec().name`) / Processor / Task；plot 默认 title=indexed_unique_name、size=1x2、**source 空白**，grid 默认 facet=repeat；加 logic 行后立即 focus 打开其 Edit (`:6668-6691/7811-7845/182-191`) | S4-1(Plot)、S4-2(其余三类) | `MUST_CLOSE` |
| VIEW/LOGIC 解耦 | plot 面板永不 build/own/start 任何节点（`PanelEditor.meas_panel` 恒 None）；`logic_nodes`(行) 与 `running_nodes`(在跑节点) 是两个必须分开的集合 (`:1754-1760`) | S4-1 立、S4-2 保 | `MUST_CLOSE` |
| 面板卡片 | FluentGroupBox，标题条左=kind+来源图例、右=Setting 按钮（resize 重定位）；卡边框即拖拽把手(OpenHandCursor)；构造末尾 `waiting for data…` (`:1937-1958/3127-3129`) | S4-1、S4-5 | `MUST_CLOSE` |
| 拖拽布局 | 拖动=改序；松手先 `drop_index`（落在已有卡槽上=插到它前面把它挤下去，落在末尾之后=追加）再 `pack()` 严格 NW 重力压实；**卡序列是布局唯一真相源**，pack 宽度=滚动视口实时宽度；视口宽变自动重排 + 首次 show 后一次性重排 (`:3097-3125/7767-7808/1397-1449/7753-7765`)。注意 `docs/task_console_design` 仍写 `_compact/_pos_to_slot`，**该处文档陈旧，以代码为准** | S4-5 | `MUST_CLOSE` |
| Logic 行 | 状态点(stopped 灰/running 绿/error 红)+名称+kind+状态文本+Start/Stop/Edit/Remove；`set_state` 变更门控；publishes 图例逐信号一行“名+形状”，形状由真实值抽取、含义进 tooltip、文本真变才 setText (`:6265-6372/7344-7358`) | S4-2 | `MUST_CLOSE` |
| 未启动即可绑定 | `_declared_signal_keys` 与运行后的 `published_signals()` **逐字相同**，因此“先接线后 Start”与存档载入的绑定都能在生产者一发布时自动接上；prefix 一经 Start 即锁死为实例身份；task 为 off-hub，显示 `<mid_run_key> (mid-run)` (`:7379-7392/7360-7377/162-164`) | S4-2 | `MUST_CLOSE` |
| catalog 自动发现 | console 只吃 measurements/processors/tasks 三个 spec 序列，下拉项与 Edit 表单全部由 spec 派生，新 `@measurement/@processor/@task` 零 GUI 改动即出现 (`:6684-6691/7866-7881`, `operations/_spec.py:68-160`) | S4-2 | `MUST_CLOSE`（当前新树硬 `ValueError`） |
| Start 顺序刚性 | BUILD(表单值优先合并，保住表单看不见的键如 ROI selection) → 反应环校验 → 按 `occupied_devices()` 交集停冲突节点(不相交硬件共存) → schema 所有权交接 → `start()` → **成功才 COMMIT**；build 失败绝不动正在跑的 (`:8329-8474`) | S4-2 | `MUST_CLOSE` |
| Stop vs Remove | Stop 只停节点、**信号留在 hub**（跑完的扫描仍可画、面板可预先接线）；Remove 才 purge 该节点发过且无其它在跑节点拥有的信号 (`:8688-8766`) | S4-2 | `MUST_CLOSE` |
| 节点轮询终点单源 | 每 base tick 轮询 error/finished/进度；自行结束或出错一律走 `_stop_logic_node(_silent=True)` 这唯一终点，否则僵尸节点冻住全板 shot clock (`:8768-8820`) | S4-2、S4-6 | `MUST_CLOSE` |
| Setting 弹窗 | 五段竖排 Source/Display/[Analysis]/Panel，**除 Source 的表达式 Apply 外没有全局 Apply，一切即时生效**；label 列宽按本弹窗最长标签测量；真 toggle 带 0.25s 去抖 (`:1961-2271`) | S4-1(Source/Display/Panel)、S4-3(全 kind 参数) | `MUST_CLOSE` |
| 树形 signal picker | 每 slot 一个按产出节点分组的可折叠 `FluentTreeComboBox`（含 `(none)`、含已声明未运行输出）；多 slot kind 有 `+/- signal`，单 slot kind 由 `PANEL_SINGLE_SLOT_KINDS` 数据驱动地没有；面板顶部显示 `accepts <input_format>` (`:2055-2126/221-260/2531-2542`) | S4-1 | `MUST_CLOSE` |
| Display 控件 | size 预设、该 kind `display=True` 的声明式参数(cmap/length/show_dist/bins/fit 三态/ylog…)、relim(tight/normal/fixed 单源 `_RELIM_PARAM`)、repeat_mode、fixed lo/hi(仅 fixed 时显示)、grid 的 facet + sub plot 两个下拉 (`:2128-2225/815-872/1458-1483`) | S4-1(通用)、S4-4(grid) | `MUST_CLOSE` |
| Analysis 段 | `AnalysisControls` 是 Setting 与 Edit 共用的**唯一**构件(surface 参数区分)，action 下拉统管 none/curve fit/ROI 并按面板 kind 能力过滤；所有控件每次 derive 都从 `params['fit_request']/['selection_*']` 重新播种 (`:1109-1197/2227-2235/5240-5258`) | S4-3(与 U0.3d Fit owner 对接) | `MUST_CLOSE` |
| 参数即时生效三路径 | `_set_param` 单一写入口：值没变直接 return；先取 render barrier；relim 走活体路径不重建（进 fixed 时用 `current_lims()` 冻结当前视图）；plotter 能就地吃的不重建；其余结构类旋钮打 `_force_rebuild` 并用 **90ms 单发定时器合并成一次 build-then-swap**，失败保留旧图、卡片永不空白 (`:3345-3426/1492-1497/1874-1889`) | S4-1 | `MUST_CLOSE` |
| PanelEditor 段落 | Panel→Acquisition(产出节点自报参数，预填当前值 + `now: X` + Apply)→Source→Parameters→Display→Processing(冻结快照+Refresh)→Analysis→Limits→Save；Edit 快照照搬卡片 size、build-then-swap、保持放大 cell (`:4988-5546`) | S4-1(骨架)、S4-2(Acquisition/Source)、S4-3(Limits) | `MUST_CLOSE` |
| 框选写回采集参数 | 2d 面板 area 与 zoom 都经 `region_to_acquisition_parameters` 把选区填进 Acquisition 输入框并提示 `Apply to use it`（**只填表不改采集**）；1d 仅 area 框选写回 spec 声明的 `axis_range`(必要时自动打开该节点的 Logic Edit) (`:5557-5574/5710-5770/6999-7030`) | S4-2 | `MUST_CLOSE` |
| `now:` 运行值 + Apply 不重启 | 每拍只刷新当前可见 PanelEditor 的 `now:` 标签与灰色 placeholder（**永不覆盖用户输入**）；Apply 走 `apply_acquisition_parameters`，运行中由采集环在两帧之间自己应用，**绝不从 GUI 线程停/启采集线程** (`:5090-5100/5772-5784/7636-7656`) | S4-2 | `MUST_CLOSE` |
| 自动表单 | `ParamDecl → PARAM_WIDGETS` 单循环(13 种 kind)，`display=False` 为不进表单的黑名单但保存值原样保留；标签列宽统一；required 未填则禁用 Start (`:4712-4791/4915-4941`, `param_widgets.py:894-913`) | S4-2 | `MUST_CLOSE` |
| repeat 自动注入 | `repeat`(0=∞) 是唯一自动注入的采集旋钮，仅 measurement/camera 注入(camera 默认 0、扫描默认 1)，processor/task 不注入；plot 侧只有 repeat_mode，**永不能反过来命令 measurement 跑几次** (`:4536-4553/6411-6419/8593-8604`) | S4-2 | `MUST_CLOSE` |
| tab 切换统一刷新 | `currentChanged` → 当前 widget 若实现 `refresh_on_show()` 就调一次，无类型特判；新编辑器实现该方法即自动参与 (`:7732-7745`) | S4-1 | `MUST_CLOSE` |
| task 接管 | Start 一个 task 即造一张**普通** PanelConfig 的 mid-run 面板（扫描类→grid+facet 到最后扫描轴，0/1 维退化成 1d），走与手加面板完全相同的建卡路径，source 固定 `value = __task_frame__`；`_apply_task_lock` 禁用六件 header 并在状态条给出 Stop task；Pause/Selectors/Devices 不锁；所有变更入口逐个 short-circuit (`:8471-8548`) | S4-6 | `MUST_CLOSE` |
| task 数据不经 hub | `__task_frame__` 每拍由 console 从 `node.output.latest_tensor(mid_run_key)` 注入表达式命名空间，不 bump hub version，因此在 `_tick` 的 version 门之前无条件跑；渲染忙则跳过下一拍补 (`:149-160/8569-8591/9066-9068`) | S4-6 | `MUST_CLOSE` |
| 错误三层 + 状态条优先级 | 节点行红点红字 → 该节点 Edit 的长文 → 常驻状态条按 **error > task > warning(display-behind) > idle** (`:8354-8440/8785-8789/9185-9234`) | S4-6 | `MUST_CLOSE` |
| workspace 存什么 | `{schema, version:3, name, interval_ms, panels[], logic[]}`；panel 存 kind/title/row/col/size/source/params/inputs/role，logic 存 kind/name/title/values；**row/col 只是重力压实的种子**，真正 round-trip 的是阅读顺序；schema 不符直接报错 (`:1650-1707/1585-1613`) | S4-7 | `MUST_CLOSE` |
| 存前 flush、载入后一律 STOPPED | `read_state` 先把打开的 Logic Edit 表单回灌 `row.node.values`（从未 Start 过的节点参数也能存）；`load_state` 建行后**一律 STOPPED**，Start 永远手动，并**重放每个面板持久化的 region 控制信号**，否则 Analysis 派生关联与“载入后一 Start 就用当年选区”会静默丢失 (`:6835-6892`) | S4-7 | `MUST_CLOSE` |
| 面板↔Analysis 关联是持久化派生 | `params['region_signal']`(跨库唯一名) + `params['region']` 是单源，关联由名字而非运行期 id 成立，故存/载自动重连、两个面板不可能串线；清除统一走 `_remove_panel_analysis` (`:6531-6537/8020-8092/8284-8290`) | S4-7 | `MUST_CLOSE` |
| 采集永不被显示限速 | `UPDATE_INTERVALS=(100,200,400,800)` 是谐波集合，定时器 base=所有面板 update_ms 的最小值，每面板每 `update_ms//base` 拍重画；**acquisition 端从不设上限**，落后只在状态条给 amber 提醒 (`:1475-1483/8963-8972`) | S4-1、S4-6 | `MUST_CLOSE` |
| 全板一个 shot clock | `_display_shot` = 所有“被面板绑定且生产者仍在跑”的信号最新 provenance 的 **min**（排除已停节点与故障节点，自由标量不约束）；`_panel_frame_key` 一变全板一起脏，所以一发脉冲的多帧不可能停在不同 shot (`:8830-8871/9040-9058`) | S4-1 | `MUST_CLOSE` |
| 渲染在 worker + barrier + 欠拍公平 | `_tick` 只收集并提交，worker 建共享 namespace 并 compose，整批一起 present；**所有 GUI 线程触图路径必须先取 render barrier**；渲染忙或该面板正在交互则本拍不提交但置 `_beat_owed`，下一个空闲拍无视模数优先服务（取代已删的 rotor） (`:9060-9157/3768-3774/9076-9106`) | S4-1、S4-4 | `MUST_CLOSE`（新架构等价物=BoardFrame + CoherenceStamp + layout_generation + 单调 sequence） |
| Pause / Selectors 语义 | Pause 只冻显示（`_tick` 只做轮询与状态条就 return），因为上一拍是 compose-all→present-all，冻住的板是一个一致 shot；Selectors 开关逐卡就地武装/停用选择器层，**零重建、不碰采集**，新加/载入的卡自动继承开关态 (`:8974-8993/6695-6707/6933-6937`) | S4-1、S4-3 | `MUST_CLOSE` |
| 什么必须即时生效 | source Apply、每个声明式参数、size、title、unit、relim/fixed lo-hi、facet/sub plot、Setting 里的一切编辑都即时作用到活面板（生产者已停也用 `_last_namespace` 重画）；size 改动是 `setUpdatesEnabled(False)` 内的一次原子事务，避免出现一帧空白大卡 (`:1890-1896/3151-3178/3463-3489`) | S4-1、S4-5 | `MUST_CLOSE` |
| 关窗语义 | `hide_on_close=True`(notebook) 把 X 接到 `stop_all_nodes`：逐行走 `_stop_logic_node` 停干净（**绝不用裸 `node.stop()`**，否则僵尸节点冻住 shot clock 且行仍显示 running），但保留面板与编辑器；独立窗 closed→shutdown 完全拆除 (`:9277-9301/9333-9372/9419-9451`) | S4-1、S4-2 | `MUST_CLOSE` |
| 存取文件 | Save/Load 默认目录 = 上次地址或 `ZLC_TASK_DIR`/`<repo>/tasks`(自动建)；任何配置改动把 Save 点亮 YELLOW；`show_task_console(task=)` 支持 `.json` 路径或 `tasks/<name>.json`；默认开局是**空板** (`:9237-9268/1710-1729`) | S4-7 | `MUST_CLOSE` |

**纵切序列（依赖有序，每片自身行为闭合、独立验收、独立 commit；片内可分阶段 commit，但每个 commit 都过完整白名单）：**

1. **S4-0 口径冻结**（本节，无用户面变更）。
2. **S4-1 通用 N 面板 live board**——破 one-card。必须最先：one-card 是硬 `raise` 而非缺功能，拖拽布局、rolling grid、workspace 几何与 §18.4「拖拽一个 panel 期间其它 panel 继续 present」全部建立在 ≥2 面板之上；同时先用可运行代码钉死 VIEW/LOGIC 解耦，S4-2 才有正确落点。本片只把 `LiveBoardController` 的固定拓扑参数化，**保留** spec 类型断言，不碰 catalog 与 neutral 数据面。
3. **S4-2 catalog 驱动的 Logic 行**——破 one-Task（当前 `task_console_catalog_items` 对非预期 key 直接 `ValueError`）。必须同时接通三种真实 Definition 的表单，**不得**先建通用 schema→Form 引擎、`BoundOperation` 协议、Analysis registry 或 `DatasetInputSlot/AnalysisStep`（后两者在当前生产树零命中，本清单不得引入）。
4. **S4-3 全 plot-kind live 覆盖与非空交互句柄**：每个已显示 panel 必须给出适用的 zoom/pan/crosshair/hover/selector；不适用者只能以 `NOT_APPLICABLE_WITH_EVIDENCE` + main 行号立案，**禁止返回空句柄冒充完成**。
5. **S4-4 live rolling grid**：facet 与 sub plot 两个下拉即时生效；overview/exact-cell focus/返回复用 U0.3f-h 已交付的同一层（不暗选第一格、稀疏 hole 原位）；per-tick 渲染预算与公平跳帧取代旧 rotor/`_beat_owed`。
6. **S4-5 拖拽布局与几何**：drop_index + 严格 NW 压实 + 视口宽变重排 + size 原子事务。
7. **S4-6 task / monitor / status**：状态条四级优先级、task 锁与 mid-run 面板、display-behind amber 计量。
8. **S4-7 workspace save/load 与清单4 收口**：codec round-trip、region 重放、收口对照表逐条 PASS/FAIL，并交付真实 launcher 的 tutorial E2E（UX-005 的关闭证据）。

**禁止复制机制（本清单全程）：** 不得迁入旧 `SignalHub` 的全局名空间/latest-value join/gap→latest fallback；不得把 node-owned worker/thread 或运行期动态 pipeline edge 带进新架构；不得复制 console-wide `RenderLoop`/`_zlc_interacting`/`_beat_owed`/`_display_shot` 手写相干快门（等价物是 BoardFrame + CoherenceStamp + layout_generation + 单调 sequence）；不得按 rank/singleton/index-0/global-nanmean/anonymous-flatten 猜轴；不得让 GUI 与 worker 无确认共享同一 Figure/artist。

**S4 进度账本（每片收口时追加一行，不写 changelog，只写状态与证据）：**

| 片 | 状态 | 已闭合的冻结行为 | 证据 |
| --- | --- | --- | --- |
| S4-0 口径冻结 | `COMPLETED` | oracle 口径、49 条行为表、纵切序列、禁止机制、删除边界 | 本节 |
| S4-1a 破 one-card | `COMPLETED` | Add Panel 每次真的再加一块板（旧实现第二次 Add 直接 `RuntimeError("TaskConsole currently owns exactly one card")` 且 Add/catalog 自禁用）；每块板自带 Remove，**Remove 拒绝未 idle 的板而不是杀掉它**（沿用 `load_intent` 既有的「must be stopped and idle」判据）；板名 `Pulse scan #N` 对齐 main 的 indexed_unique_name；`scan_card` 收敛为「最新一块板」，Analysis 目标收敛为「最新一块持有 FINAL artifact 的板」；关窗等待**每一块**板的嵌入面板排空 | `tests/test_u04_task_console_multi_card.py`（7 项 oracle，含拒绝语义与关窗等待） |
| S4-1b 常驻状态条 | `COMPLETED` | 「窗口骨架」行的常驻状态条:一条永远挂载、固定高度、按 error>task>warning>notice 排序的 `FluentStatusStrip`(旧实现是每 tick 一条 ladder;迁移后的 console 退化成了会随文本长高的裸 `FluentLabel`)。ladder 落成 `zlc_frontend.qt_widgets.arbitrate_status_line` 单源,任何窗口共享同一排序;卡片 `stateChanged` 变更门控驱动 task 级 | `tests/test_u04_status_strip_priority.py`(6 项);偏离登记 `UX-013` |
| S4-1c(a) METER display owner | `COMPLETED` | 异构面板的前置依赖:`ViewIntent` 四成员中 METER 的 display state 原先私有在 `workbench/_figure.py`,导致 `LiveBoardController` 的 panel→revision 映射结构上没有 METER 槽位。已提升为 `zlc_frontend/meter_display.py::MeterDisplayState`(**不**造空 form spec,理由写在模块里) | `tests/test_u04_meter_display_owner.py`(6 项,含「每个 ViewIntent 都有 owner」机械守卫) |
| S4-1c(b) 拓扑参数化 | `SUPERSEDED`(路线由清单4 改道作废,恢复方式改为 shell 接线) | `LiveBoardController` 目前只能表达两种拓扑(1×IMAGE,或 IMAGE+CURVE+HISTOGRAM+METER 定序四联),`live.py` 有四处硬断言(scalars 全有全无、panel 0 必须 IMAGE、恰好三个 scalar、必须按 CURVE/HISTOGRAM/METER 排序)与一处按 index 写死的 display-revision 阶梯。需换成有序 `PanelSpec` 元组 + intent→renderer 表 | — |
| S4-1c(c) 面板卡片与 Setting | `SUPERSEDED`(路线由清单4 改道作废,恢复方式改为 shell 接线) | 面板卡片、树形 signal picker(`FluentTreeComboBox` 已迁但其数据 helper 仍在旧树 `frontend/param_widgets.py`)、Display 控件、参数即时生效三路径、shot clock、Pause/Selectors、tab 刷新 | — |
| **清单4 改道(用户指令 2026-07-20,最高优先级)** | `ACTIVE` | 放弃「从零重建新 console + 对账清单逼近旧行为」路线,改为 **shell salvage**:旧 GUI 文件本体即骨架——展示件原样搬进目标包、旧路径留转发壳(新旧共跑**同一份代码**,保真度由构造保证),两大壳(`frontend/task_console.py` 10036 行 / `frontend/live.py` 6321 行)函数体内的 61 处领域懒 import 固化为**缝台账**(=接新数据面的端口规格,双向棘轮)。S4-1c(b2)/(c) 的从零参数化路线**作废**;已落的 S4-1a/1b/1c(a)/(b1) 作为 workbench 组件保留,最终由 shell 接管入口;`UX-014` 的差距恢复方式改为「shell 接线」而非逐控件重造 | `tests/test_u05_shell_salvage.py` |
| S5-shell(a) 展示层搬家 | `COMPLETED` | `canvas/selectors/ticks/_validate` → `zlc_frontend/live_plot/`;`param_widgets` → `zlc_frontend/qt_widgets/`(包 `__init__` 末尾收编,fluent/board 同款属性可达、不进 `__all__`);`_paths` → `zlc_storage/paths.py`(PROJECT_ROOT 两级上溯不变)。缝:`display_path` 改经 `zlc_storage.paths`;`_positive_float/_int` 因 canonical 名字守卫改名(壳内保旧名恒等别名);`.strip()` 六处按 named-adapter 先例登记。**放置公理(守卫机械强制)**:zlc_frontend 内只有 qt_widgets 可 import PyQt5,而 qt_widgets 不可 import matplotlib——因此旧渲染模型的 Qt+mpl 联姻件 `render_loop/qt_canvas` **留在旧树**,随壳最终转 worker-raster(S12.5)。旧 console 实测在搬家后代码上原样构建(9 按钮/双 tab 不变) | `tests/test_u05_shell_salvage.py`(壳纯度/全名恒等/task_console 38+live 23 条缝台账双向棘轮) |
| S5-shell(b) 并行计划源清除 | `COMPLETED` | 违反 L3855「唯一计划级权威……旧 roadmap 不再作为并行计划源」的三份文件已删:`docs/MIGRATION_GOAL_zh.md`(326 行)、`docs/HANDOFF_GOAL_zh.md`(133 行),以及把 Z 预算镜像绑进前者的 `test_every_budget_is_mirrored_in_the_goal_document`(棘轮本身保留)。同时删除 L4648 已判「作废」路线的遗留验收件:`test_u04_console_ui_parity.py`(从零重建+对账清单逼近旧行为)、`test_w1_console_ux_oracle.py`(直接构造 `TaskConsole` 类断言,而入口从不返回该类=代理指标)。白名单 128→126。**记录两次实证违规**:(1) 拟建 `CameraMonitorLiveDataset -> SignalHub` adapter 让旧 console 渲染,撞 L4636「不得迁入旧 `SignalHub` 的全局名空间」与 L3261「不允许把旧 Hub/LogicNode/metadata vocabulary 适配成新 event/data/runtime 类型」,且 camera monitor 已由 M1 迁完,属回滚;(2) 曾写入「入口先切、缝后清」规则并 commit,撞 L596「尚未迁出的旧共享-Figure RenderLoop 只允许留在 `SerializedLegacyAggBridge` 隔离岛」。两者均未进入生产代码,规则随文件删除一并撤销 | 本行 + 白名单 diff |
| S5-shell(c) 词汇下沉 | `COMPLETED` | `task_console.py` 缝台账 **11 → 5**。六条纯字面量词汇(`NO_LINEAGE=-1`、`SWEEP_SCAN_SLOT/API_SLOT`、`ANALYSIS_SPEC_NAME`、`ANALYSIS_ACTIONS`、`DEFAULT_MID_RUN_KEY`)按 sink/port 判据(纯常量/描述沉 `zlc_data`,活对象才建端口)迁入新模块 `zlc_data/vocabulary.py`;四个旧域模块反向 import 它,单一定义不变(实测 `ANALYSIS_ACTIONS` 为同一对象,非副本)。其中 `DEFAULT_MID_RUN_KEY` 原是 `task_console.py:149` 的**模块级** import,该行独自把 `operations.task` 及其约 45 个 legacy 模块拖进 console 的导入图。旧 console 实测仍原样构建:49 widget / `Monitor`+`Logic` 双 tab / 7 FluentButton,零 UX 漂移。顺带把 3 个测试文件里 6 处指向已删 `MIGRATION_GOAL_zh.md` 的失效文案改指本文件 | 全白名单 index 导出门 126/126(一文件一进程) |
| S5-shell(d) 扫描模板下沉 | `COMPLETED` | `task_console.py` 缝台账 **5 → 4**。`ScanColumnSpec`(5 个标量的 frozen dataclass)与 `scan_table_template(kind, columns)->str`(纯文本生成)按 AST 行段**字节等价**搬入 `zlc_data/scan_template.py`,`pulse_table.py` 反向 import;两条路径取到**同一对象**(`timing.scan_table_template is zlc_data.scan_template.scan_table_template`)。同刀受益的是**两个壳**:`task_console` 两处懒 import 与 `pulse_gui` 顶部批量 import 均改指 `zlc_data`。**`scan_column_spec` 刻意未搬**——它由 `bus_signed_range`/`UNITS_TO_NS`/`positive_time_step_ns`(后者走 `eval_time_expr`)派生默认扫描区间,属 pulse 几何,按 live.py 已立先例「纯的搬走,不纯的倒置」留在域内,后续由端口交付成品 columns。两壳实测仍原样构建(console 49 widget/双 tab;pulse_gui import 正常) | 白名单门 125/126;唯一红项 `test_m2e_camera_monitor_first_roi.py::test_close_terminalizes_a_pending_roi_revision_without_stale_layout_promotion` 经实测为 **flaky**(导出树 2/5 失败、真实树 0/3、上一刀同文件同内容为绿),失败点 `assert len(receipts) == 2` 紧随 `_submit_roi_control` 且未泵事件循环;与本刀 5 个改动文件无因果,按精确 staging 不在本 commit 顺手改控制面时序语义,单独立项根因 |
| S5-shell(e) 扫描列端口倒置 | `COMPLETED` | `task_console.py` 缝台账 **4 → 3**,最后一条非活对象缝闭合。`scan_column_spec` **不搬**(依赖 bus signed range/单位表/时钟 tick,属 pulse 几何):改由 `PulseTemplateRows` 增 `api_columns`/`scan_columns` 两字段,域侧 reader(`frontend/__init__.py` 组合根)算好成品 `ScanColumnSpec` 交给表单;`_PulseSlotsWidget.rebuild` 收下即用,不再自算。至此 console 拿到的全是**描述**,渲染层不碰 pulse 编译器。新增契约测试 `test_the_template_port_hands_over_columns_the_gui_could_not_have_built` 钉住必须保住的行为:DAC 列 = signed code ±512 且 `is_dac`,duration 列 = 时间区间,两者区间不得相同(操作员踩过的真实 bug)。该守卫**经变异测试证明非真空**(把域侧 kind 判别改坏即红,还原即绿),且 fixture 刻意绑定两种 slot——旧有 row 断言跑在无 slot 模板上,空元组会让任何 columns 断言天然为真 | 全白名单 index 导出门 126/126;console 仍 49 widget/双 tab |
| S5-shell(f) 环守卫改按声明识别 | `COMPLETED` | `task_console.py` 缝台账 **3 → 2**。`_reactive_ring` 原以 `isinstance(start_node, Processor)` 识别反应式节点(渲染层里的域 import),改为问节点自己声明的 `consumes`——这正是域侧基类对同一属性已有的做法(`LogicNode._collect_provenance` L780:「probed by attribute, not by type」)。等价性的承重事实是「`consumes` 只由 `Processor.__init__` 赋值、无任何类级默认」,已由新 oracle 用 AST 机械钉死,而非停留在口头。新增独立 oracle `tests/test_u05_reactive_ring_oracle.py`(白名单 126→127):环闭合被拒 / 纯链放行 / 无 `consumes` 节点不被走;行为用例刻意用**真 `Processor` 子类**而非鸭子替身——用为取悦守卫而造的假对象去测一个按声明识别的守卫等于自证。两条变异均被抓(环检测恒返回 None → 红;给 `Measurement` 加类级 `consumes` → 红)。**唯一覆盖 `_reactive_ring` 的旧测试 `test_signal_cycle_guard.py` 在白名单外**,按 L3806 冻结不执行不修改,故本切片必须自带 oracle | 全白名单门 127/127;Z2b 具名台账 5→6(新 oracle 需构造旧 console,按守卫要求具名登记而非计数) |
| S5-shell(g) fit 叠加改按声明识别 | `COMPLETED` | `task_console.py` 缝台账 **2 → 1**,壳内只剩 `ProcessorRun` 一条(真构造器,待工厂端口)。`_push_fit_overlays` 原用 `isinstance(node, FitProcessor) and node.action == "fit"` 找发布 fit 参数的节点,改问节点是否声明 `<prefix>fit_valid`——`AnalysisProcessor.provides` 本就按 live action 分派且域侧保持「declared == published, per action」,故声明集精确回答该问题;顺带消掉与下一行版本门重复拼的信号名,判据与门再不能各自漂移。等价性由新 oracle 在三种真实节点形态上**同时求值新旧两个谓词并要求一致**钉死(FitProcessor / Analysis(fit) / Analysis(roi)),并要求结果跨形态有真有假,使恒真谓词无法蒙混;变异测试(令 `provides` 不按 action 分派)即红。**`_push_fit_overlays` 全仓零覆盖**(白名单内外皆无),故该等价 oracle 是唯一屏障。两条「按声明非按类」事实合并进 `tests/test_u05_declaration_not_class.py`(由 `test_u05_reactive_ring_oracle.py` 改名而来),避免为塞入行为测试而让纯结构文件 `test_u05_shell_salvage.py` 新增旧树依赖 | 全白名单门 127/127(一文件一进程)。**同进程共跑时 `test_every_name_resolves_to_the_identical_object` 会红**:MOVES 台账含运行时安装的可变全局 `_solve_thread_guard`,其值随 import 顺序在 `None`/函数间变动;经 stash 复算确认该敏感性由上一刀 S5-shell(f) 的 oracle 引入而非本刀,且门的一文件一进程协议不触发,按精确 staging 单独立项 |
| S5-shell(h) 一次性 run 由 spec 自造,**两壳缝台账归零** | `COMPLETED` | `task_console.py` 缝台账 **1 → 0**;`live.py` 早已为空,故**两大壳的领域缝全部闭合**。最后一条是真构造器:reactive 分支本就由 spec 自造(`spec.make_node`),one-shot 分支却要壳去够 `ProcessorRun` 类——不对称即病灶。`ProcessorSpec` 增 `make_run(hub, *, readout, camera, sequencer, params, prefix)` 与 `make_node` 对称(域内 `processor.py` 局部 import `logic.py`,实测无环),壳改为 `spec.make_run(...)`:壳仍决定**借哪台设备**(只有它知道哪个运行中节点独占持有),但不再知道该构造哪个类。reactive spec 调 `make_run` 显式报错(实证)。顺带修一个真陷阱:台账清空后写成 `{}` 会**从 set 静默变成 dict**(AST 实测 `type=dict`),改为显式 `set()` 并把五刀的处置方式记进注释——sink / 端口交付成品 / 两处改按声明 / 构造器归还 spec,四种答案各有其因 | 全白名单门 127/127(一文件一进程);console 仍 49 widget / 双 tab / 7 FluentButton |
| S5-shell(i) 两壳互不依赖 | `COMPLETED` | 缝归零后复查发现的**壳↔壳耦合**:`task_console` 向 `pulse_gui` 借 `slot_label`(它从 pulse_gui 拿的唯一符号),等于把 6321 行的脉冲编辑器变成开 console 的前置,并把两壳的搬迁次序绑死。`slot_label` 是 state-free 的纯文本(由 `(kind,target)` 直接产标签,零依赖),按 AST 行段字节等价沉入 `zlc_data/shape_text.py`,`pulse_gui` 反向 import;三条路径实测取到**同一对象**。该模块的准入表述**如实拓宽**而非偷塞:原写"domain 发布 / GUI 复现",现覆盖"多于一个表面必须拼法一致",并说明本例是两个**壳**要一致(其状态相关的互补件 `timing.scan_target_label` 仍在域内)。新增 AST 守卫 `test_neither_shell_imports_the_other`(用解析而非 import,保持该文件零旧树依赖),变异回插跨壳 import 即红。**下一阶段被放置公理挡住**:L4649 判定 `render_loop`/`qt_canvas` 这对 Qt+mpl 联姻件留在旧树、"随壳最终转 worker-raster(S12.5)",而 `task_console.py` 仍 import 两者(L119/L6637),故壳整体搬入 `zlc_frontend` 必须等渲染改造完成 | 全白名单门 127/127;console 49 widget/双 tab,pulse_gui 正常 |
| S5-shell(j) console 接入渲染隔离岛 | `COMPLETED` | 补上 L596 一直未被满足的强制:未迁出的共享-Figure RenderLoop **只允许**活在 `SerializedLegacyAggBridge` 内,但该岛此前**只有测试在用**,console 一直直接 new `RenderLoop` 并裸调 9 处。现全部经岛,`task_console.py` 中裸 `_render_loop` 归零。接岛过程暴露并修掉岛自身两处缺陷:(a) 缺 `busy`(纯查询,shell 定跳拍要用,否则被迫绕过岛);(b) `settle/close` 用 `positive_real` **拒绝 `timeout=0`**,而"预算耗尽、不再等待"正是面板拆除路径的合法语义——若不修,该路径只能保留裸 loop 引用,反而击穿隔离。新增 `_render_barrier()` 单一转换点:岛抛异常,面板拿到的仍是原来的 bool 契约,九个调用点不各自 try。**三处原先忽略 barrier 返回值的 GUI 路径改为 fail-closed**(登记 §2.2 `UX-015`)。两个覆盖该协议的旧测试 `test_render_loop_contract.py`/`test_figure_viewer.py` 均在白名单外冻结、按 L3806 不执行不修改,故本刀自带 oracle `tests/test_u05_render_island.py`(白名单 127→128,Z2b 具名 +1) | 全白名单门 128/128(一文件一进程,含新 oracle)。console 仍 49 widget/双 tab |
| S5-shell(k) 旧 front 改用文档化的边界值 | `COMPLETED` | §12.5 `WORKER_RASTER_LIVE` 要求 GUI 只接 immutable front,而 `zlc_frontend/render.py` 开篇把反例点名写死:「No live Figure, Artist, mutable or aliased ndarray view, **or QImage storage** crosses this module's boundary」。旧 canvas 的 render-thread→GUI 交接恰恰是**存着一个 QImage**(`_zlc_front`)。改存 `RasterBuffer`(owned immutable bytes),`paintEvent` 只在一次 blit 期间临时成像。这不是措辞之争:QImage 是**由建它的线程拥有的 Qt 对象**,一旦渲染者不再是这个 widget 本身,它就当不成交接值;owned bytes 可以——所以这是把 presenter 从 `FigureCanvas` 解绑(进而让壳能搬进 `zlc_frontend`)的必要第一步,放置公理仍由 L4650 挡住整体搬家。同刀立**单源** `RasterBuffer.from_agg_rgba(w,h,buffer)`:目标侧 `SinglePanelAggRenderer._draw_raster` 与旧侧 canvas 本要各自拼同一个三元事实(tight stride / RGBA8888 / **必须拷贝**),现由一处说;第三项才是会**默默损坏像素**而非报错的那个(`memoryview`/`bytearray` 都能过长度校验却仍别名 worker 缓冲)。旧 canvas 的 fallback blit 也并入同一构造。新 oracle `tests/test_u05_raster_front.py`(白名单 128→129,Z2b 具名 +1)钉五件事:交接类型、像素与 Agg 逐 bit 相同、drop-front 回落、拷贝语义在源被改写后仍成立、两个 Agg 表面都经单源构造(AST 结构断言,非取值比较)。两条变异均被抓(回插 QImage 存储→3 红;像素格式换 ARGB32→1 红)。生产路径实证:`BaseLivePlot.draw()` → `canvas._zlc_present()` 后 front 为 `RasterBuffer 480x357 RGBA8888` 且可成像;console 仍 49 widget/双 tab | 新节奏:边界+三棘轮单进程、全清单 collect-only 1551 项/6.8s;收口全白名单单进程实跑(见 Rule 8 改写) |
| S5-shell(l) 转发壳由**快照**改为**实时代理** | `COMPLETED` | 18 个 `MOVED to` 壳都用 `globals().update(vars(_moved))` 转发,而每个壳的 docstring 都写着「Every name … resolves to the SAME objects」。**这句话是假的**:`globals().update` 复制的是**绑定**,不是模块;被移出模块之后**重新绑定**的任何全局,壳里的副本永远停在导入那一刻。这不是理论——`zlc_data/curve_fitting._solve_thread_guard` 初值 `None`,由 `register_solve_thread_guard` 装入真正的 Qt 线程守卫,而旧路径 `core.fitting._solve_thread_guard` **永远答 `None`**。改为 PEP 562 `__getattr__` 逐次转发(`from <shim> import name` 也走这条路),承诺由**构造**保证而非碰巧成立;补 `__dir__` 以免 `dir()`/补全变瞎。壳内显式赋值(如 `_validate` 的 `_positive_float` 别名)仍正常遮蔽 hook,实测为真。**这一刀由新测试节奏逼出来**:逐文件跑长期全绿,A 组合并单进程后立刻实锤——这是旧口径拿不到的证据。消费者安全性自查四条全空:全仓无任何壳的 `import *`、无指向壳路径字符串的 `monkeypatch/mock.patch`、无对壳取 `vars()/dir()/__all__`、无对壳属性赋值。棘轮只遍历导入时存在的名字,抓不到重绑,故新增 oracle `test_a_shim_follows_a_name_the_moved_module_REBINDS_later`(在真实 pair 上重绑并还原);变异回插快照即红。`__annotations__` 加入 `_MODULE_INFRA`——它与已排除的 `__doc__/__spec__` 同类,壳自己没有注解,旧快照匹配只是因为 `vars(_moved)` 恰好带着它。**过程事故须记录**:本轮曾派 18 个只读审计 agent,其中若干**擅自改了三个生产文件**(`_validate.py`/`selectors.py`/`raster.py`),且两个 agent 给出的版本已经互不一致(一个带 `__dir__` 一个不带);已存证到 scratchpad 并全部回退,最终 18 个由一次脚本统一改写 | 全清单 129/129(A 组 104 文件单进程 1303 passed;B 组 25 文件各自进程 25/25) |
| S5-shell(m) 面板尺寸词汇下沉,解开板面几何的搬家闸 | `COMPLETED` | 壳内**1840 行 / 44 项完全不碰 render**(AST 词边界实测;先前用子串 `fig` 扫会把 `config` 误判,该结论已作废重算),按放置公理今天就能搬——但最连贯的一组「板面几何」(shelf packer `pack` / 落位 `drop_index` / `_first_free_slot` / `_card_size`,`drop_index` docstring 自称 "Pure geometry (no Qt)")**两个包都进不去**:它要 `CARD_PAD` 等 Qt chrome token(在 `qt_widgets`,带 PyQt5),又要 `panel_size_cells`(在 matplotlib 模块 `live_plot/live.py`),而 L4650 规定 qt_widgets **不可** import matplotlib。真闸是**面板尺寸词汇被关在渲染模块里**。本刀只搬这个闸:`PANEL_SIZES` + `panel_size_cells`(纯字符串解析,零依赖)按 AST 行段字节等价沉入 `zlc_data/panel_size.py`,`live.py` 反向 import,两壳改为直接从新家读——想**知道**有哪些尺寸不再需要拖进整个渲染栈。**`panel_display_size` 刻意不搬**:它经 `panel_plot_spec` 依赖 `FigureSpec`/`PANEL_UNIT_PX`/margins,是真的 figure 布局,留在渲染层是对的;因此板面几何**本刀仍未搬成**,如实记为下一刀。DAG 守卫 `test_text_normalization_is_confined_to_named_input_adapters` 当场抓到 `.strip()` 换了家而具名登记没跟——代码字节未变,登记随之迁移(这正是该守卫的用途)。覆盖该词汇的两个测试 `test_size_driven_geometry.py`/`test_panel_hug_layout.py` 均在白名单外冻结(L3806 不执行不修改),故自带 oracle `tests/test_u05_panel_size_vocabulary.py`(白名单 129→130):同一对象、每个预设都解析且大小写/空白容忍、未知拒绝、**该模块零 import**(它能下沉的全部理由,断言而非假设)、两壳从新家读(结构断言,故本文件自身零旧树依赖)。两条变异被抓(给词汇模块加一个 import → 红;壳改回从渲染层读 → 红;后者今天仍能跑通,正因为再导出是同一对象,所以只有 import 图能抓) | 全清单 130/130(A 组 105 文件单进程 1307 passed;B 组 25 文件各自进程 25/25);collect-only 1556 项/6.5s |
| S5-shell(n) 板面几何搬进目标包 | `COMPLETED` | 上一刀解闸,本刀搬本体:shelf packer(`pack`)、落位规则(`drop_index`)、`first_free_slot`/`min_board_width`/`board_width`/`GeomProxy` → 新模块 `zlc_frontend/board_layout.py`,壳内只剩薄转发。**剩余依赖的处置判定**:`_card_size` 依赖 `scaled_px`(运行时 Qt 缩放可变)与 `panel_display_size`(figure 布局),两者都不该进这个模块;若按 mapping **快照**交进去,就会复刻我刚在 S5-shell(l) 修掉的那个缺陷——用户把窗口挪到另一块屏,卡片像素变了而快照不变。故按 sink/port 判据走**端口倒置**:`BoardMetrics(gap, card_size=callable)`,并把「必须可调用」做成构造期硬校验(传 Mapping 直接 TypeError),让这个理由机械化而不是留在注释里。`_cell_size`/`_card_size` 明确留在壳内,它们是 figure 尺寸与 Qt chrome 的**桥**,两侧都不许被搬走的模块 import。搬迁**逐像素等价**由改动前抓的 golden 证明:5 组卡片 × 3 种板宽 = 15 个布局,加 12 个落位判定,新旧输出完全一致。覆盖该代码的 `tests/test_board_gravity.py` 在白名单外冻结(L3806),故自带 oracle `tests/test_u05_board_layout.py`(白名单 130→131):golden 逐像素、settled board 再 pack 不动(幂等)、任意两卡不近于 gap、落位四例、`min/board_width` 随最宽卡缩放、**该模块 import 集 ⊆ {`__future__`,`dataclasses`,`typing`}**、以及壳确实转发(AST,故本文件零旧树依赖——一个偷偷留副本的壳会通过上面每一条,只有调用图看得见)。真实卡片尺寸表烘成字面量,所以 oracle 不需要 Qt 也能复现 GUI 的真实像素。两条变异被抓(重力扫描顺序 y,x→x,y → 6 红;放开 callable 校验 → 1 红) | 全清单 131/131(A 组 106 文件单进程 1319 passed;B 组 25/25);collect-only 1568 项/6.5s;真窗口 console 仍 49 widget/双 tab |
| S5-shell(o) signal-expr 复合件搬家,**注入式工厂端口 2→1** | `COMPLETED` | 先按「搬前查同类件」筛掉一个陷阱:`_FitFixSeedEditor`(92 行 fix/seed 编辑器)**不能搬**——目标包已有同一关切的 `zlc_frontend/fit_editor.py`(`fit_constraint_form`/`fit_spec_from_form`)+ `qt_widgets/fit_authoring.py`(`FitAuthoringPane`,"One shared model/constraint editor"),照搬就是造第二个源。那是**收敛**问题不是搬家问题,如实记为待办而非本刀顺手做掉。改搬 `_SignalExprWidget`(192 行):它用的每个原语(grouped picker / `read_editable_combo` / fluent 叶子 / `seed_source_for_slots` / `SIGNAL_EXPR_HELP`)**都已在目标包**,它只是消费者,不是重复件。搬后的真正收获是**消除一次倒置**:`ParamWidgetContext.signal_expr_factory` 的存在理由由该文件 docstring 亲口写明——「the two COMPOSITE widgets that live in `task_console` … injected so this module needn't import them」;widget 一到位,handler 直接构造,该字段**删除**(不是留着不用),console 少传一个回调。`labels_provider` 原本只经工厂闭包传入,现改由 ctx 携带,不丢短名映射。壳内保留 `_SignalExprWidget = SignalExprWidget` 纯别名。等价性由改动前抓的 golden 证明:7 例 `set_value → values_dict` 往返(含 `{}`/`None`/空 inputs 三种回落到单槽 `frame_0` 默认)全等价。三个守卫当场发挥作用并全部按其本意处理:(a) `.strip()` 具名上下文随代码迁到新家;(b) Z2b 具名登记(本 oracle 要断言壳别名 IS 同一个类);(c) **Z3 抓到我自己的措辞污染**——`"MOVED to"` 是纯转发壳的约定标记,我在非 shim 的 console 里写了它,守卫把 console 计成 shim,改词即消。新 oracle `tests/test_u05_signal_expr_widget.py`(白名单 131→132):golden 七例、`signal_expr_factory` 字段确已消失**而 `pulse_slots_factory` 仍在**(诚实标注本刀只搬了一个复合件)、handler 结构上直接 new(AST)、新模块不 import matplotlib、壳别名 is 同一类。变异回插工厂字段即红 | 全清单 132/132(A 组 107 文件单进程 1330 passed;B 组 25/25);collect-only 1579 项/6.6s;真窗口 console 仍 49 widget/双 tab |
| S5-shell(p) 最后一个复合件搬家,**注入式工厂端口 1→0** | `COMPLETED` | `_PulseSlotsWidget`(278 行)→ `zlc_frontend/qt_widgets/pulse_slots_widget.py`,`ParamWidgetContext` 从此**不含任何 `*_factory` 字段**——它不再有任何回头够旧壳的回调。本刀之所以现在能做,是因为它的依赖恰好是前几刀逐个沉下去的:`SWEEP_*` 来自 (c)、`scan_table_template` 来自 (d)、`slot_label` 来自 (i)、`api_columns/scan_columns` 由 (e) 的 `PulseTemplateRows` 端口交付成品;**唯一的壳内依赖只剩 7 行的纯谓词 `_is_number`**,随类一起搬。等价性由改动前抓的 golden 证明,而这组 golden 是**重抓过一次的**:第一版对 seed 完全不敏感(六个种子输出全同),因为 `seed_value` 是**待应用**语义——按 `program_id` 存,只有下一次 id 匹配的 `rebuild` 才恢复(存档在模板行已知之前就到达,急着应用会写进还不存在的字段)。重抓后的六例真正覆盖该契约,含一个易错点:**非数值 seed 把该键整个丢掉**而不是把文本写进数值字段。守卫两响:`.strip()` 三处具名上下文随代码迁到新家;**上一刀 S5-shell(o) 的双向棘轮按设计翻红**——它当时断言 `pulse_slots_factory` **仍在**以诚实标注"只搬了一个",现在两个都搬完,那句话成了旧闻,连同理由一起退役(这正是那条双向断言存在的意义)。新 oracle `tests/test_u05_pulse_slots_widget.py`(白名单 132→133):六例 golden、`ParamWidgetContext` 中 `*_factory` 字段计数为零、**两个 handler 都结构上自建**(参数化 AST,一个偷偷改回工厂的 handler 会通过所有行为断言)、新模块不 import matplotlib、壳别名 is 同一类。变异回插 `pulse_slots_factory` 即红 | 全清单 133/133(A 组 108 文件单进程 1341 passed;B 组 25/25);collect-only 1590 项/6.6s;真窗口 console 仍 49 widget/双 tab |
| S5-shell(q) 逻辑节点记录下沉 + 行卡搬家 | `COMPLETED` | 按铁律1 先筛,**两个候选被判不可搬**并如实记录而非顺手做:(a) `AnalysisControls` 依赖 `_FitFixSeedEditor`,后者已确认与目标包 `fit_editor.py`+`fit_authoring.py` 同关切,是收敛问题;(b) `PanelConfig` 的 `PANEL_KINDS` 派生自渲染层 plot-kind 注册表——**与 S5-shell(m) 同形**,闸在「词汇被关在渲染层」,是另一条多米诺,本刀不硬开(顺带确认它与 `zlc_workbench.workspace.PanelSlot` **不是**同一关切:后者是新板的 panel 身份+相干组,不含 kind/params/像素)。实搬 `LogicNodeConfig`(36)+`LogicNodeRow`(108),目标包无同类件(只有它用的 `FluentStatusDot` 叶子,早已迁)。**`layout_record` 一并沉下去**是本刀的判断点:它是 panel / logic-node / console-state **三个**记录共用的校验器,只搬 logic-node 那一半会把它与另外两个调用者劈开,故随第一个搬走的记录下沉,壳反向 import。**DAG 守卫当场纠正了我的放置**:`zlc_data` 不得 import `zlc_storage` 包根,只允许 `zlc_storage.canonical` 这一个纯校验模块——而 `exact_mapping` 恰在其中,守卫指的位置分毫不差,改窄 import 即通(理由已写进模块 docstring)。golden 两组:记录往返(缺 title 回落为 name),行卡**通用观测**七个快照(所有 QLabel 文本 + 所有按钮可用性,不猜属性名),含 publishes 空态文案与短名拆分。新 oracle `tests/test_u05_logic_node.py`(白名单 133→134)另钉一条易漏语义:**未知状态原样显示、不静默映射成 stopped**。两条变异被抓(import 放宽回包根 → DAG 红;未知状态静默映射 → oracle 红) | 全清单 134/134(A 组 109 文件单进程 1353 passed;B 组 25/25);collect-only 1602 项/6.8s;真窗口 console 仍 49 widget/双 tab |
| S5-shell(r) plot-kind 词汇多米诺第一层:repeat-mode 下沉 | `COMPLETED` | 按上一刀定的路线开这条闸,但**开工侦察改变了拆分设计,如实记录而非硬推**:`PlotKind` 的 9 个字段里只有 `cls` 是渲染器绑定,其余 8 个是词汇——本以为一刀可拆,实测发现表是**两步装配**的:字面量 6 个 kind,`grid` 由 L5779 `PLOT_KINDS = PLOT_KINDS + (...)` 稍后追加(我的剥离正则漏掉的那个 `cls=` 其实在注释里,而那条注释正好写着 "APPENDED to this table",是它暴露了这个结构)。既然它是渲染层**最承重**的注册表,不在侦察当轮半吊子拆;`PlotKindSpec` 拆分连同两步装配的处理留作下一刀,证据口径已写进本刀 oracle 的 docstring。本刀落该多米诺**第一层且自洽**的一段:五个 repeat-display 词汇表(`BASE`/`TRACE`/`ROLLING`/`IMAGE`/`HIST`)按行段搬入 `zlc_data/repeat_modes.py`,`live.py` 反向 import,task_console 改从新家读。它们是纯字符串元组、零 import,却被关在 matplotlib 模块里——而**三个表面必须拼法一致**(console 建 repeat 下拉、measurement 校验、figure 层应用)。覆盖它的三个测试 `test_plot_kind_table` / `test_frontend_plot_contract` / `test_scan_repeat` 均在白名单外冻结(L3806),故自带 oracle `tests/test_u05_repeat_modes.py`(白名单 134→135):五表逐值(**顺序即默认项**)、渲染层读回**同一对象**、**特化必须由 base 扩展而非各自重述**(否则就能各自漂移)、`roll` 只在 rolling trace 出现、该模块零 import、壳从新家读(结构断言)。两条变异被抓(HIST 把 `pool` 从首位挪走 → 红;IMAGE 混进 `roll` → 红) | 全清单 135/135(A 组 110 文件单进程 1358 passed;B 组 25/25);collect-only 1607 项/6.8s;真窗口 console 仍 49 widget/双 tab |

**删除边界：** 清单4 迁完之前**禁止**整文件删除 legacy `Zou_lab_control/frontend/task_console.py`（它仍是 API_SLOT segmented、其它 Measurement/Processor、rolling/gridplot/selector/calibration/temperature/MOT panel 与旧入口的共同宿主）；U0.1 表的「保存/恢复」验收项不得删除，必须由 S4-7 正面交付。可删项按最后 consumer 逐项记账，留待清单9 Z0。

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
+ process-lifetime InstallationRuntime + immutable hardware graph
+ capability-evidence gates（近期可做什么与终态想做什么分开）
```

扩展性来自稳定数据与能力边界、显式组合和机械 contract tests，而不是更多继承层、Protocol、Service 或动态注册。

近期最重要的落地顺序不是先拆完五个namespace，而是先让lifecycle/resource状态诚实、建立Workbench/render宿主、跑通Camera event -> DatasetBuilder -> live/save，并用E0a真qCMOS characterization确定工作点；S1最终adapter与H1 schedule语义稳定后重新执行Q0 release qualification。随后在冻结bitstream上以`AUTONOMOUS_RESIDENT`运行近期无缝装载方式基线，用active qualification + preflight margin + ordered metadata + per-run EndAttestation共同授予Formal eligibility；refilled仍默认拒绝。逐沿stamp、新ROM、trigger-return等只有§1.1/H2的证据条件、根因因果关系和独立批准全部满足后才评估。

终版GO/NO-GO裁决分开写：顶层架构与E0a只读/离线部分、F0、S0.5 **GO**；S0.6的process-lifetime InstallationRuntime、immutable InstallationDeviceGraph、startup/claim-first recovery/shutdown、内部consumer与frontend窄port迁移 **GO**，但新增任何普通用户真实设备入口在public object-graph/umbrella/docs gates清零前 **NO-GO**；S1-S3按dependency-closed cut **GO**；S4代码实现于H1/S1/S3接口稳定后 **GO**。除§19明文限定的H1前`DIAGNOSTIC_CHARACTERIZATION`迁移例外（批准legacy SOP、唯一owner、现有fingerprint/ABI、observed live identity/endpoint与旧SOP safety evidence齐全，且绝不产生target runtime/AssetMap/Q0/Formal authority）外，任何target runtime或普通实验真实设备drive capability在installation AssetMap（canonical内容digest、exact adapter kind、expected live matcher）生效、该adapter的identity/disconnect/SafeStateContract真机recipe通过前均为 **NO-GO**；Pylon尤其必须通过removed+live readback拔线测试，不能以`IsOpen/IsGrabbing/GetDeviceInfo`缓存组合放行。用户可用Formal PulseScan capability仍为 **NO-GO**，直到current deployment有active ProgrammedImageDeploymentRecordRef、当前最终adapter的Q0 qualification active、完整physical waveform/arm/edge/camera-tail margin、mode-specific raw terminal稳定读语义、adapter-specific SafeStateContract、deployment-bound compiled/H1 post-terminal output-tail bound与`PostTerminalTailEvidence`、近期单qCMOS BoundSourceAssociationContract、软件exact链和EndAttestation E2E全部通过。deployment record/H1/Q0/contract-kit评估本身不要求重烧，也不冒充runtime content attestation；冻结硬件最终能否通过这些gate由真机证据决定，不能预先承诺。硬件改变默认 **NO-GO**；唯一解锁条件是E0a/Q0在已批准余量、正确camera配置和充分软件reservation下仍实测loss/reorder，并且camera设置、软件保留/排空、trigger rate与margin调整均无法修正；或现有RTL bug/既定设计偏离被证实；之后仍须PI/硬件owner单独批准。

最终实现应让用户继续使用熟悉的TaskConsole、PulseGUI和notebook工作流；重型board保持下线程raster性能，notebook保持短路径，MOT保持SCAN_SLOT自主扫描。resident/refilled只标记同一`AUTONOMOUS_STREAMED`方式族下的装载方式，不能单独授予Formal资格；只有active Q0 qualification、exact链、`ORDERED_END_ATTESTED_RUN`和本run EndAttestation共同通过时才标记Formal eligible。内部消除软件缓冲跳帧、线程竞态、隐式降维和重复算法；相机↔point的物理保证明确采用整run末端证明而非逐沿硬件tag，任何不一致整run拒绝，所有迁移bridge在Z0物理删除。
