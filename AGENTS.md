# AGENTS.md — 当前仓库工作契约

## 权威边界

- 当前目标架构与迁移 gate 的唯一权威是 `docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md`。
- `tests/README.md` 只说明测试入口；其它旧 roadmap、TaskConsole v2 设计及历史记录都不能反向定义目标架构。
- 用户已经授权仓库内正常实现与验证；不要为普通命令重复索要权限或额外的继续确认。
- `Zou_lab_control/` 中尚未迁走的 GUI/runtime 是 `SerializedLegacyAggBridge` legacy island。它的测试只约束该岛在 dependency-closed cut 前保持既有行为，不能约束 `zlc_frontend`、`zlc_workbench` 或其它目标包的架构。

## 硬约束

- 中文沟通；代码标识符与 commit message 可用英文。
- 所有工作只在 `codex/system-architecture-migration`；分主题详细 commit，`Co-authored-by: Codex <codex@openai.com>`；不 push。
- 不保留兼容层、双格式、历史 archive、转换器或无消费者的防御机器。旧机制在最后消费者迁走的同一 dependency-closed cut 删除。
- bitstream/RTL 冻结。只有证据证明现有 RTL 有真实 bug 或违背既定设计时，才单独评估硬件修改；不得为了架构偏好要求重烧。
- 精密 pulse/trigger 时序由现有 FPGA、qCMOS 等硬件执行；host 只冻结计划、验证工作 envelope、排空数据与做末端对账，不用 sleep 调度硬件边沿。
- calibration/readout 的物理与算法唯一权威是 `main@6c337d49c7086fa0ff21f879cd159bdf0e753f51`。偏离必须指出 main 的具体问题，并用同一原始帧的独立 oracle 证明差异。
- 数据内核永久保留 `(R, P, *data_shape)`；不得把多维 `data_shape` 折成单一 `data_dim` item，也不得用隐式 `reshape(...)[0]`、flatten 或 trailing-axis 平均制造权威标量。

## 当前包边界

目标 DAG 为：

```text
zlc_storage   zlc_data   zlc_pulse
      \          |          /
             zlc_neutral_atom
                     |
               zlc_frontend
                     |
               zlc_workbench
```

领域类型、canonical codec、digest、生命周期与硬件 I/O 各有且只有一个 owner；跨包嵌入值对象时调用 owner 的公开序列化器，不复制字段表或 validator。

## Rules 1–7 追溯整改 gate

在设计文档记录的 Rules 1–7 全局状态全部为 `COMPLETE` 且独立对抗审查无 P0/P1 前，不开始新的迁移切片。整改范围每次由 `git rev-list main..HEAD` 与 `git diff --name-only main...HEAD` 机械产生，不能凭会话记忆抽样。

1. 每条不变量一个 owner；其它边界信任已验证的不可变类型。
2. 机制必须对应已经观测的失败；没有失败依据的守门、防伪、兼容和未来机器删除。
3. 只给真实跨会话持久格式保留朴素格式名；不建版本迁移体系。
4. 已验证物理逻辑选择性继承 main，并使用同帧独立 oracle。
5. 测试必须有独立 oracle；不以实现同款公式镜像自证。
6. 每个切片报告与 main 等价物的 PLOC/NCLOC/class 比；超过约 3 倍必须解释或压缩。
7. 计划级决定与代码同 commit 写回活设计文档；教程/手册可等完整迁移后统一更新。

## 验证与文件纪律

- 先跑能证明改动边界的测试；跨子系统收敛或交付前再跑合并/全量验证。
- GUI 视觉验收必须走用户真实入口和真实交互；headless 单元测试不能冒充最终视觉验收。
- 修改文件使用 `apply_patch`；保留工作树中不属于当前主题的改动，提交时只 stage 选定文件/块。
- 删除或重构后执行死符号/历史残余搜索、`git diff --check` 和相关测试；不得留下 TODO/FIXME 代替本目标内的实现。
