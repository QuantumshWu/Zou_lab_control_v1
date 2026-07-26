# AGENTS.md — 当前仓库执行契约

## 恢复与权威

- 权威顺序固定为：用户最新明确要求 → 当前完整 `/goal` → 物理/算法事实与 `main` 正式用户行为 → `docs/DESIGN_CHARTER_zh.md` 与相关设计章节 → 当前实现 → 测试。
- 每次上下文压缩后先完整读取当前 `/goal`，再从 Git 状态、当前 plan 与最近实现 checkpoint 恢复；只读取本任务相关的设计章节。不得重新回答用户已经得到答案的问题，不得重做已闭合审查，也不得把历史台账状态冒充当前任务。
- `docs/MIGRATION_LEDGER_zh.md` 只保存历史因果和已完成 checkpoint，不是新的架构权威；除非正在核对某个明确 checkpoint，不全文读取。
- 发现设计与事实冲突时，先由代码、真实产品流和物理约束重新推导，再同步修正设计；不能为了维护旧文档或旧测试而保留错误实现。

## 当前包与产品边界

目标依赖方向按路径而不是只按顶层包名判定；下列箭头表示“左侧可以导入右侧”：

```text
zlc_data -> zlc_storage.canonical
zlc_neutral_atom(headless framework/capability core) -> zlc_data + zlc_pulse + zlc_storage
zlc_frontend(presentation-only; no neutral dependency) -> zlc_data + zlc_storage
zlc_workbench(domain-neutral hosts) -> zlc_frontend + zlc_neutral_atom(headless API) + data/pulse
zlc_neutral_atom/logic_nodes/<capability>/workbench_adapter.py -> own core（仅真实启动差异，可选）
zlc_neutral_atom/logic_nodes/<capability>/ui/<leaf> -> own capability core + generic frontend/workbench
Zou_lab_control -> installation/runtime/repository + declarations/preparers/loaders + optional frontend/workbench
```

- `zlc_neutral_atom/logic_nodes/` 只拥有实验领域能力，不等于“每个直接子目录都是一个节点”。独立 capability 可以直接闭合；同一物理族的多个 capability 必须放在一个具名 family 下，由 family 根拥有真正共享的领域机制、每个可枚举叶节点各自导出一份 headless `LogicNodeDeclaration`。例如 Readout family 拥有 Calibration/Occupancy/Fidelity 共享的读出合同，Release-recapture family 拥有两帧同 loading→survival 的机制，Temperature 与 Grey-molasses detuning 仍是设备 claim/扫描轴/输出语义不同的两个叶节点。不得把 family 共享机制伪装成第二个节点，也不得为了消除 sibling import 把它提升进 framework/runtime。
- Definition、request/config、算法、输出/schema/materializer 与 artifact 必须在自己的 capability leaf 或其唯一 family owner 内闭合。`LogicNodeDeclaration` 一次包含字段、path hints、dynamic-choice resolver、typed inputs/outputs、default views 及 request build/bind；普通节点由通用 TaskConsole projector 自动生成 form、Setting/Edit、signal 与默认 panel，不得另建 per-node UI/attachment。
- 只有通用 host 无法表达的真实启动调用差异才允许根下可选、headless `workbench_adapter.py`；它只适配 prepared command 的启动形状，不拥有字段、presenter 或 lifecycle。只有 declaration + generic Figure 无法表达的特殊产品交互才允许 inert `ui/**`；当前有证据的例外仅为 PulseScan scan-table/slot、Calibration 多页报告/创建面和 Occupancy exact-cell 导航面。它们的普通字段仍走 declaration projector，SiteMap/Figure 交互仍委托 frontend 唯一 owner。capability 根不得 eager import adapter/UI，`ui/__init__.py` 必须 inert。
- `zlc_neutral_atom` 的 Calibration/Occupancy capability 独占 SiteMap 领域事实（site axis/centers/validity/coordinate frame 与 calibration/source identity），并在 publication 前验证 source revision/event、same-shot inputs、join digest 与 sibling outputs 的原子 causal closure。`zlc_frontend/` 独占由这些已闭合事实建立的 SiteMap view、Area materialization、render 与 selector；不得替领域拼 latest/latest。`zlc_workbench/` 只保存领域中立 host/project/lifecycle；linked-front 只按已接纳 identity/revision 阻止 GUI 显示 N/N-1 混合，不证明物理 same-shot，也不对独立 producer 作 board-wide coherence 声明。
- `Zou_lab_control/` 必须保留为唯一稳定 public notebook API、installation/repository/runtime binding 与 desktop composition adapter；并入 neutral 会让领域层拥有应用层 facade，并入 workbench 会让 headless API 依赖 Qt，因此两者都禁止。它必须保持薄，只显式接线和委托，不能拥有 capability schema、算法、materializer、presentation、旧 runtime/registry/DeviceSet 或第二套窗口实现。
- 领域类型、canonical codec、digest、数据 shape/validity、生命周期、硬件 I/O、Figure/selector/Fit 和 Qt composition 各有且只有一个 owner。跨包嵌值对象时调用 owner 的公开 API，不复制字段表、validator、shape 规则或算法。
- 数据内核永久保留 `(R, P, *data_shape)`；标量物理表示固定为 `(R,P,1)`。禁止按 rank/singleton 猜语义，禁止隐式 first/flatten/trailing mean，禁止把多维 `data_shape` 压成一个 item。

## 实现方法

- 先问产生现象的机制是否应该存在，再修代码。每个非 `main` 新机制必须能点名真实需求、唯一 owner、现有 consumer 和相对直接方案的必要收益；答不出就删除完整依赖闭包。
- “两个消费者”绝不是进入骨架的充分条件。一个共享 owner 只有同时满足三项才可离开 `logic_nodes`：消费者跨越彼此独立的领域族；公开 vocabulary 不含某一实验的物理语义；删除任一具体 capability 后它仍有完整独立职责。任一项失败就属于 domain family。移动前必须记录语义、消费者集合、删除测试与依赖方向；不得用目录对称、避免 sibling import 或代码行数作为升层理由。
- 不打局部补丁，不留 alias、wrapper、兼容 reader、迁移态、历史 archive、零消费者抽象、第二套实现或改名残余。删除必须覆盖生产者、消费者、导出、文档与已经失去意义的测试。
- bitstream/RTL 冻结。只有证据证明现有 RTL 有真实 bug 或违背既定设计时才单独评估修改；不能为了架构偏好要求重烧。精密 pulse/trigger 时序由现有 FPGA、qCMOS 等硬件执行，host 只冻结计划、验证 envelope、排空数据和做末端对账。
- calibration/readout 的物理与算法以 `main@6c337d49c7086fa0ff21f879cd159bdf0e753f51` 为基线；偏离必须指出 main 的具体错误并用同一原始输入的独立 oracle 证明。
- Calibration Task 的 operator folder 不是 artifact repository：相对路径锚定项目根，默认是 `_output/calibrations`，显式绝对路径按用户选择使用；机器消费者只能解析最后替换的 `calibration_ref.json`，再向canonical repositories admit其中的typed refs，`report/frames`不是authority。当前 writer 只提供同进程异常回滚和 pointer-last 可见顺序，没有目录 fsync/recovery journal，禁止把 `report/frames/pointer` 整体描述成 crash-atomic 或 durable transaction。
- 普通 Qt 编辑只修改稳定 widget/editor draft；不得构造全应用 snapshot、周期轮询或重建控件树。只有真实 worker、连接、Run、数据 revision 或 presentation-coherent board front 边界可发布 immutable snapshot；最后一种只冻结GUI展示revision，不证明物理same-shot。
- UI 的正式界面、操作与美术默认逐项继承 `ZLC_main`；新后端可以不同，但没有明确更优理由时像素和手感必须一致。通用 widget、Figure/Divider、selector、Setting/Edit 和 renderer 必须复用唯一实现。

## 工作与验证纪律

- 可以并鼓励使用 subagent 并行完成真实、互不重叠的文件切片或产品流；agent 可在明确文件范围内修改。禁止用 agent 反复复审同一个小改动、重复已闭合工作或制造审查仪式。
- 修改文件使用 `apply_patch`；保留用户及其它 agent 的无关改动。只在主题闭合后逐文件 stage；禁 `git add -A`，禁 push，禁破坏性回退。
- 迁移期间不为历史测试适配架构。先阅读测试所表达的物理/产品原理，只有它仍是 current contract 才更新或运行；普通改动不跑宽测试。关键边界用最窄的 `py_compile`、静态残余搜索、`git diff --check` 和一个真实产品流证明，全部迁移完成后再做合并验证。
- GUI 快轨：先设 `QT_QPA_PLATFORM=offscreen`，再由唯一 `ensure_qt_app()` 经正式 composition root 打开窗口，使用真实 Qt input 和 outer-window `grab()`；不得另造窗口、DPI、尺寸或样式。慢轨从正式 `.py/.bat` launcher 按人类流程运行，只用于最终或争议复核。
- 每个最终审查切片必须建立实际文件清单并逐文件读完，报告文件数、真实问题、保留抽象的 consumer 和删除内容。上下文压缩后从该 checkpoint 继续，不能把“文件不在”或某个测试通过当作审查完成。
