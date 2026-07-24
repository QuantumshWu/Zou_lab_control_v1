# AGENTS.md — 当前仓库执行契约

## 恢复与权威

- 权威顺序固定为：用户最新明确要求 → 当前完整 `/goal` → 物理/算法事实与 `main` 正式用户行为 → `docs/DESIGN_CHARTER_zh.md` 与相关设计章节 → 当前实现 → 测试。
- 每次上下文压缩后先完整读取当前 `/goal`，再从 Git 状态、当前 plan 与最近实现 checkpoint 恢复；只读取本任务相关的设计章节。不得重新回答用户已经得到答案的问题，不得重做已闭合审查，也不得把历史台账状态冒充当前任务。
- `docs/MIGRATION_LEDGER_zh.md` 只保存历史因果和已完成 checkpoint，不是新的架构权威；除非正在核对某个明确 checkpoint，不全文读取。
- 发现设计与事实冲突时，先由代码、真实产品流和物理约束重新推导，再同步修正设计；不能为了维护旧文档或旧测试而保留错误实现。

## 当前包与产品边界

目标依赖方向为：

```text
zlc_storage   zlc_data   zlc_pulse
      \          |          /
             zlc_neutral_atom
                     |
               zlc_frontend
                     |
               zlc_workbench
                     |
             Zou_lab_control
```

- `Zou_lab_control/` 现在只允许存在 public notebook/composition facade 和 launcher glue；它不是 legacy GUI/runtime 岛。旧 runtime、registry、DeviceSet、第二套算法或窗口实现不得留在该包。
- 领域类型、canonical codec、digest、数据 shape/validity、生命周期、硬件 I/O、Figure/selector/Fit 和 Qt composition 各有且只有一个 owner。跨包嵌值对象时调用 owner 的公开 API，不复制字段表、validator、shape 规则或算法。
- 数据内核永久保留 `(R, P, *data_shape)`；标量物理表示固定为 `(R,P,1)`。禁止按 rank/singleton 猜语义，禁止隐式 first/flatten/trailing mean，禁止把多维 `data_shape` 压成一个 item。

## 实现方法

- 先问产生现象的机制是否应该存在，再修代码。每个非 `main` 新机制必须能点名真实需求、唯一 owner、现有 consumer 和相对直接方案的必要收益；答不出就删除完整依赖闭包。
- 不打局部补丁，不留 alias、wrapper、兼容 reader、迁移态、历史 archive、零消费者抽象、第二套实现或改名残余。删除必须覆盖生产者、消费者、导出、文档与已经失去意义的测试。
- bitstream/RTL 冻结。只有证据证明现有 RTL 有真实 bug 或违背既定设计时才单独评估修改；不能为了架构偏好要求重烧。精密 pulse/trigger 时序由现有 FPGA、qCMOS 等硬件执行，host 只冻结计划、验证 envelope、排空数据和做末端对账。
- calibration/readout 的物理与算法以 `main@6c337d49c7086fa0ff21f879cd159bdf0e753f51` 为基线；偏离必须指出 main 的具体错误并用同一原始输入的独立 oracle 证明。
- 普通 Qt 编辑只修改稳定 widget/editor draft；不得构造全应用 snapshot、周期轮询或重建控件树。只有真实 worker、连接、Run、数据 revision 或 coherent board front 边界可发布 immutable snapshot。
- UI 的正式界面、操作与美术默认逐项继承 `ZLC_main`；新后端可以不同，但没有明确更优理由时像素和手感必须一致。通用 widget、Figure/Divider、selector、Setting/Edit 和 renderer 必须复用唯一实现。

## 工作与验证纪律

- 可以并鼓励使用 subagent 并行完成真实、互不重叠的文件切片或产品流；agent 可在明确文件范围内修改。禁止用 agent 反复复审同一个小改动、重复已闭合工作或制造审查仪式。
- 修改文件使用 `apply_patch`；保留用户及其它 agent 的无关改动。只在主题闭合后逐文件 stage；禁 `git add -A`，禁 push，禁破坏性回退。
- 迁移期间不为历史测试适配架构。先阅读测试所表达的物理/产品原理，只有它仍是 current contract 才更新或运行；普通改动不跑宽测试。关键边界用最窄的 `py_compile`、静态残余搜索、`git diff --check` 和一个真实产品流证明，全部迁移完成后再做合并验证。
- GUI 快轨：先设 `QT_QPA_PLATFORM=offscreen`，再由唯一 `ensure_qt_app()` 经正式 composition root 打开窗口，使用真实 Qt input 和 outer-window `grab()`；不得另造窗口、DPI、尺寸或样式。慢轨从正式 `.py/.bat` launcher 按人类流程运行，只用于最终或争议复核。
- 每个最终审查切片必须建立实际文件清单并逐文件读完，报告文件数、真实问题、保留抽象的 consumer 和删除内容。上下文压缩后从该 checkpoint 继续，不能把“文件不在”或某个测试通过当作审查完成。
