# 设计与迁移执行宪法

本文件只保存每次实现都必须遵守的稳定规则。领域细节以
`SYSTEM_ARCHITECTURE_DESIGN_zh.md` 的相关章节为准；`MIGRATION_LEDGER_zh.md` 只记录历史
checkpoint，不能反向要求重做已经闭合的工作。

## A. 权威与上下文连续性

1. 权威顺序是：用户最新要求 → 当前完整 `/goal` → 物理/算法事实与 `main` 正式用户行为 → 本宪法与相关设计章节 → 当前代码 → 测试。
2. 上下文压缩后必须完整读取当前 `/goal`，从 Git、plan 和最近 checkpoint 恢复，只读当前子系统相关设计。不得重复回答旧问题、重做已闭合审查或从旧台账重新启动迁移。
3. 设计不是代码的辩护词。若真实产品流、profiling 或硬件事实证伪设计，先重新推导 owner 与最小机制，再同时修设计和实现。
4. 测试只约束仍然有效的物理不变量、产品行为和包边界；历史实现形状、私有方法、调用次数和旧兼容面没有权威。

## B. 包、owner 与复杂度

5. 依赖方向固定为 `storage/data/pulse -> neutral_atom -> frontend -> workbench -> Zou_lab_control`。`Zou_lab_control` 只做 public notebook/composition facade，不保存旧 GUI/runtime、raw device graph 或第二套领域实现。
6. 每条不变量只有一个 owner。外层消费已经验证的 immutable typed value，不重复字段表、shape 推断、codec、digest、算法、生命周期或安全状态。
7. GUI 只拥有用户意图、展示状态和 composition；Measurement/Task/Processor application 拥有自己的 request、物理输出 vocabulary、schema/materialization 与 artifact lineage；`zlc_data` 拥有通用 `(R,P,*data_shape)`、axis/layout/validity/selection/transform/Fit 机械规则。
8. 每个抽象必须有当前 consumer。只有一个成员的 enum、只实例化一次且不隔离真实边界的 class、一层 forwarding wrapper、无生产消费者的 public surface 和未来机器都删除。复杂度必须与观测到的问题相称。
9. 不保留兼容层、双格式、旧 reader、版本迁移体系、历史 archive、迁移 adapter 或改名残余。最后 consumer 迁走时，同一 dependency-closed cut 删除完整闭包。

## C. 数据、Figure 与 GUI

10. Dataset 永久是 `(R,P,*data_shape)`；R 与 P 是物理存储维度，point axes/PointLayout 单独描述 P 的逻辑多维结构。标量固定为 `(R,P,1)` 和 canonical scalar axis。禁止 first、flatten、按 singleton/rank 猜语义或隐式平均信息轴。
11. ComponentValidity 必须覆盖其声明的 data axes；reduce/Fit/histogram/meter/派生 signal 全部消费同一 validity。显示 projection 不能静默升级成 fit/scan/artifact 的权威 transform。
12. Area、锁定 Cross 和 Fit 属于 Figure。它们只在明确手势或提交后发布派生 signal，不重配 Measurement、不建立 ROI processor、不弹第二个 DataFigure。普通 pointer motion 不发布 hover 数据。
13. 所有 plot kind 共用 FigureSpec/Divider、renderer、selector 几何、panel size 与 Setting/Edit owner。Qt 只接收 immutable raster/front 和 typed geometry；matplotlib 只在 worker-owned headless render 路径。
14. 普通编辑事件只原位更新稳定 widget。Add/Remove/Reorder 只增删移动对应控件；unit/name/value/delay/binding/visibility 不得重建整树。全量 reconcile 只允许 document generation 或 target topology 真正替换。
15. TaskConsole、PulseGUI、DeviceManager、FigureViewer 的正式外观和手感默认与 `ZLC_main` 一致；后端架构可以改变。偏离必须有明确产品或正确性收益，不能由实现方便产生。

## D. 线程、运行与硬件

16. GUI thread 不执行 blocking I/O、重 fit/calibration 或大图 compose。先 profiling 定位，再修 owner/copy/algorithm 根因；不以假 zoom、旧 raster stretch、任意防抖或新状态机掩盖问题。
17. Immutable snapshot 只出现在真实 ownership/consistency boundary：数据 revision、worker completion、连接/Run/cancel/close、capability 或 coherent board front。普通 Qt draft、定时器 tick 和局部命令不能冒充 snapshot 边界。
18. ResourceArbiter 只做当前进程内 exact owner 互斥。TaskConsole 遇到 typed `ResourceBusy(conflicting_run)` 时只停止本窗口的 exact conflicting row，等待真实 terminal 后重试同一冻结 request；外部 owner 仍拒绝。禁止持久 safety journal、历史 quarantine 或解析错误文本。
19. 每个领域 session 的 `close_session` 是正常 cleanup 唯一 SAFE/stop owner。cleanup 失败使本次 run/session 失败，但不制造跨连接永久门禁；新连接只由实时 identity、当前 SAFE 初始化和 capability 建立 authority。
20. bitstream/RTL 冻结。硬件能保证的 trigger/pulse 时序必须由 FPGA/qCMOS 等硬件执行，软件不 sleep 调度边沿。只有 E0 或代码证据证明现有 RTL 真 bug 时才单独评估硬件变更。
21. SCAN_SLOT/MOT 正常扫描只用现有 bitstream 的自主流式路径；API-slot 无法无缝更新时才使用已经存在且明确标记的 segmented `STATIC_ONCE` 例外。不存在通用 HOST_STEPPED baseline。
22. Formal qCMOS 结果依赖 qualified working point、preflight timing margin、完整 host drain 与整 run EndAttestation；任一计数/stamp/timestamp/coverage 不符使整 run INVALID，不提交拼接或错位结果。

## E. 资源、验证与交付

23. virtual 与 real 走同一 application/Run/data/Figure 路径，只替换最底层设备 Port；分析层不得读仿真真值。能力缺失用 typed unavailable/rejection 明示，不隐藏产品或伪造设备。
24. calibration/readout 的物理算法以 `main@6c337d49c7086fa0ff21f879cd159bdf0e753f51` 为基线。偏离必须指出 main 的错误并用同一原始输入的独立 oracle 证明；不得用实现同款公式自证。
25. 工作中只跑能证明当前边界的最窄验证。迁移期间不修改架构迎合历史测试；全部产品纵切与逐文件清理完成后再做合并/全量验证。
26. GUI 快轨固定为 `QT_QPA_PLATFORM=offscreen -> ensure_qt_app() -> 正式 composition root -> 真实 Qt input -> outer grab`；慢轨从正式 launcher 按人类流程运行。手工造 QWidget、直接调 handler/controller 或无文字的假截图不能作为验收。
27. subagent 用于并行完成互不重叠的真实工作，可在明确文件范围内修改；不得反复对同一小改动做多轮对抗审查。共享树中发现重叠先避让或协调。
28. “逐文件读过”不等于架构审查完成；每个文件还必须进入跨文件 change-impact 清单，检查同一语义是否在别处重述，以及新增一种能力需要改动哪些 owner。对 Task/Measurement/Processor 的硬判据是：普通字段、Dataset/Artifact 输入与 output contract 全由领域 request/config owner 声明；除 composition root 显式登记 builder/prepare command 外，新增 Definition 不得要求修改 TaskConsole 的具名 form/editor、DefinitionKey/field-key picker、fallback 或 binding intent。把错误文件改名、转发或搬目录不算修复。
29. 每个主题完成后只逐文件 stage 该主题，禁止 `git add -A`、push、破坏性回退或覆盖用户改动。修改用 `apply_patch`，删除后做死符号/历史残余搜索和 `git diff --check`。
30. 最终完成必须同时满足：正式用户产品流可复现；设计与实现一致；全部当前文件进入计数清单并逐文件审查；无零消费者机制、错层 owner、历史残余或已知 P0/P1。测试通过本身不能替代这些条件。
31. `zlc_neutral_atom` 内部固定区分骨架与附件：catalog/authoring/input-output contract、installation 与 generic runtime 是 framework 骨架；具体设备连接、adapter、SDK glue 和 virtual physical implementation 只能在 `devices/`；具体 Task/Measurement/Processor 的 Definition、Request/Config、输入输出契约、prepare/evaluate/materialize 与专用 artifact 只能在 `logic_nodes/`。目录移动必须同时修正依赖方向，改名或转发不能掩盖错层 owner。
32. `Processor` 是唯一产品与领域概念；`latest-only`、`exact`、`finite` 是 host/delivery/execution policy，不得派生第二套公开 `MonitorProcessor` 类型、catalog、form、binding 或 lifecycle。当前 live Processor 可以由 framework 的一条内部 latest-only lane 执行，但领域 Definition、request、output 和算法仍全部属于原 Processor attachment。
33. concrete attachment 只通过普通 import 与 composition root 中一条显式、不可变的 `Definition -> builder/preparer/presenter` 接线进入产品。禁止包扫描、动态 registry、service locator、FQCN 构造业务能力或把 callable 塞进持久 Definition；同时 composition root 不得重述某个节点需要哪些字段、输入 signal、artifact 或 output，这些事实只能由该 logic-node owner 声明。
34. `zlc_workbench/task_console` 是通用 host，不是 logic-node 实现目录。其核心只能机械投影 owner schema、解析 owner-declared Dataset/Artifact 输入、按 Task/Measurement/Processor host policy 托管 lifecycle 并路由 typed outputs；具名 DefinitionKey 分支、Camera/Occupancy/Calibration/MOT/PulseScan 字段组装、专用 binding/node/form 和物理 schema/materializer 全部禁止。确有产品专属显示适配时放在独立 composition/presentation attachment，并由显式 composition 接线，不得反向污染通用 host。
35. 设备与 logic-node 物理搬迁按 dependency-closed group 一次完成：更新全部 imports、canonical type/FQCN、composition 与测试后删除旧路径；不留 re-export、alias、兼容 shim 或“迁移期”双 owner。`zlc_pulse` 的 FPGA target/compiler/transport/server/RTL 是独立 bounded context，不因目录整齐搬进 neutral devices；neutral sequencer attachment 只消费其 public API。
36. 软件不得设置或推导任何内存预算、pending上限、retention quota、max-inflight/backlog或因预测内存而拒绝合法采集。有限采集按完整`expected_frames`分配相机物理buffer；continuous monitor按声明的`history_cycles * frames_per_cycle`分配，rolling window是Dataset cardinality而非预算；未ack exact事件自然保留，内存不足由SDK/Python分配自然失败。DCAM真实ring overwrite、FPGA/RF/transport物理几何等硬件容量仍必须验证，但不得复制成第二层软件policy。
37. 已经进入正式 Qt object tree 的 QWidget 不得通过 `setParent(None)` 或 parentless candidate 暂时脱离 owner：Qt 会把它升级成瞬时 top-level，造成闪窗、隐藏窗口与泄漏。终态删除统一为 `hide() + deleteLater()` 并保持原 QObject owner；复用/排序必须直接移入稳定 container；异步 admission 的 candidate 从构造开始就归最终 pane/host 所有。Monitor 可按内容横向滚动，但 Logic/Setting/Edit 是 viewport 的宽度 consumer，长路径、错误、signal/schema 必须 elide/wrap，不能用 sizeHint 反向扩大产品窗口。
38. Setting/Edit 复用的是同一个 typed declaration、state owner 与提交 handler，不是搬运同一个 QWidget。任何会在挂载前调用 `show/setVisible` 的 row/widget factory 必须在构造时取得最终 page parent；完整 subtree 构造后再一次挂入可见 container。新的 Task/Measurement/Processor row 同样必须先用其 `ConsoleNodeSpec.editor_default_values()` 形成完整有效 draft，再进入 signal topology；空 GUI placeholder 不能被当成领域 request 或动态 output 的输入。
39. `PulsePortSpec.label` 只供操作者显示和编辑，任何 Measurement/Task/Processor 都不得用 label、字符串 key 或 lane 序号猜“probe/trap/trigger”等物理角色。角色必须来自冻结 request 的显式 binding，或从已冻结 pulse waveform 与设备/Calibration 事实唯一推导并交叉验证；执行阶段只使用稳定 port key/endpoint identity。
