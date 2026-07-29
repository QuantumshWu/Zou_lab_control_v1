# Zou_lab_control 实现执行协议

本文件只规定如何恢复、实现、审查、验证和提交。规范架构与产品合同只见 `docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md`；不得在这里复制第二份 owner、数据模型或 UI 设计。

## 1. 权威与恢复

1. 权威顺序：用户最新明确要求 → 当前完整 `/goal` → 物理/算法事实 → System Architecture 相关章节 → 当前实现 → 仍有效的公开合同测试。
2. 每次上下文压缩、暂停恢复或新进程开始时：
   - 完整读取当前 `/goal`；
   - 检查当前分支、HEAD/tree、`git status --short`、最近提交和 `docs/MAINTAINER_NOTES.md`；
   - 只读取当前 dependency cut 相关的 System Architecture 章节；
   - 从 Git 和 checkpoint 推导“已完成/正在做/下一步”，不得凭记忆或固定切片编号恢复。
3. 不重复回答、重审、重做或重新解释已经闭合的事项。新证据与已闭合结论冲突时，明确指出证据和受影响 owner，再更新当前方案。
4. `pulses/scan_test.json` 是用户文件：不得读取、修改、删除、移动、stage 或提交。
5. `main` 只在调查一个明确旧行为或独立科学算法时定点查阅；不默认扫描旧树，不以旧测试、旧包结构或旧 UI 实现反向约束当前架构。

## 2. 实现方法

6. 先定位物理语义、唯一 owner、producer/consumer 和生命周期，再修改。不得以 fallback、额外状态、特殊分支、防抖、重试或 test-only guard 掩盖根因。
7. 每个工作单元必须是 dependency-closed cut：
   - 定义或修正一个公开合同；
   - 迁移全部生产者、消费者、codec、artifact、UI/API 接缝；
   - 删除被替代实现、reader、alias、wrapper、fixture、测试和文档；
   - 搜索死符号与反向依赖；
   - 用最窄真实产品流证明。
8. 若真实设备、profiling 或代码依赖证伪设计，先重新推导最小机制，再在同一 cut 更新 System Architecture、实现和当前测试。不得为保护已写代码而辩护旧设计。
9. 新 abstraction 必须指出当前 consumer、被消除的重复/风险、生命周期 owner 和 contract test。单成员 enum、单实例 forwarding wrapper、重复 DTO、未来扩展点和无消费者 public surface 默认删除。
10. 不保留兼容层、双格式、旧 reader、migration adapter、改名 re-export、历史 archive 或“暂时残余”。最后一个 consumer 迁走时同切片删除完整闭包。
11. 未进入“已由证据证明现有 RTL/bitstream 有 bug 或偏离既定设计”的独立任务，不得修改 RTL、Tcl、XDC 或 bitstream。普通软件实现不能借机设计硬件新能力。
12. 不把用户报告逐条变成特判。先横扫同一 owner 的所有入口和同根机制；真正只属于唯一 owner 的局部不变量可以局部修复，不为它创建全局框架。

## 3. 协作、文件与 Git

13. subagent 只用于范围互不重叠、可独立交付并能实质提高速度或质量的工作。不要让多名 agent 反复审查同一小改动；共享树内先声明文件范围并避免覆盖。
14. 文件修改统一使用 `apply_patch`；格式化或机械生成可使用正式工具。不得用临时脚本、shell 重定向或 Python 偷写源码。
15. 保留用户及其他 agent 的不相关改动。发现重叠时先检查 diff，能绕开就绕开；不能安全绕开才报告阻塞。
16. 搜索优先 `rg` / `rg --files`。不要运行破坏性 Git 命令，不要 `git add -A`，不要未经用户要求 push。
17. 每个主题只逐文件 stage 自己的闭包，先运行 `git diff --check`、死符号/反向边扫描和最窄证据，再 commit。commit message 描述被闭合的系统边界，不描述临时症状。
18. commit 后更新 Maintainer checkpoint；不要把活动审查台账、GUI evidence、截图、cache、临时 output 或本机路径加入 Git。
19. 普通只读、构建和测试命令直接运行，不向用户请求手动批准。若环境本身拒绝命令，先核实当前权限/命令用法，不重复弹出相同请求。

## 4. 验证与性能

20. 迁移期间只运行能证明当前边界的最窄 current test、public API 流或真实 product flow。历史测试失败先判断其物理/public contract 是否仍有效；不得改实现去迎合已删除架构。
21. **测试源码与 owner 同 cut 收口**：生产 owner、public contract 或 codec 被替换时，所有直接相关测试必须在该 cut 改写为当前物理/public contract，或与被删行为一起删除；禁止把仍导入旧类型、私有 lane、旧字符串 vocabulary 或兼容入口的测试留给 M7。留到 M7 的只有 broad-suite 执行、跨 cut E2E 补强和最终总清点。反复运行同一历史测试、为了绿灯恢复私有方法/调用次数或在每个微改后跑全仓，均不构成验证。
22. 性能问题先阅读调用链并 profiling，交叉验证 copy、algorithm、compose、lock、I/O 与线程边界。修正复杂度和 owner 后再选择 executor/process；不得用假缩放、旧 raster stretch、任意 debounce 或预算拒绝掩盖。
23. 每个 dependency cut 对照旧树同等物理/产品能力报告生产行数、frozen dataclass/class/enum 数量和主要复杂度来源。超过约三倍必须压缩或逐项说明不可替代的 consumer/invariant；行数不是单独删除依据。
24. 每个文件审查都必须进入 change-impact 清单：它拥有何种事实、由谁调用、又调用谁、是否重述别处合同、增加同类能力要修改哪些文件、能否删除。仅“逐文件读过”不算架构审查完成。

## 5. GUI 证据

25. 快轨：`QT_QPA_PLATFORM=offscreen -> ensure_qt_app() -> 正式 composition root/launcher entry -> 真实 Qt input -> outer grab`。不得手工造另一套 QWidget、尺寸、DPI、style 或直接调用 controller 代替用户交互。
26. 慢轨：从正式 `.bat/.py` launcher 按人类流程运行，用真实桌面鼠标、输入与截图做最终或争议复核。用户正在操作电脑时不要占用桌面；优先快轨。
27. 快轨和慢轨必须共享同一个 QApplication owner、composition、window sizing/style、数据路径和交互步骤；offscreen 行为测试若没有正式 composition 或文字/DPR 不可信，不能作为视觉验收。
28. GUI 验收同时检查用户可见结果、时序、不中断项、selector/Fit 发布、resize/DPR、close/cancel 和真实 signal flow。静态截图不能替代行为证据。

## 6. 完成条件

29. 当前 cut 完成前不得跳到下一 cut。完成意味着公开合同、全部生产/消费路径、删除闭包、最窄 product evidence、复杂度说明、diff 和 commit 同时闭合。
30. 最终完成必须逐项通过 System Architecture 的验收门，并证明所有 tracked 文件均已计数和审查、无错层 owner、平行真相源、历史残余、零消费者机制或已知 P0/P1。
