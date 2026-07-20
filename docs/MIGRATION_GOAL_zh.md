# 迁移 Goal(可循环执行版 · v2)

> v1 已作废。v1 里的四个 "DECISION" 是我**没读设计文档**时发明的,全部撤销。
> 根因记录见 §九,规则已改。

## 〇、权威分工(先记住这三行,越权即错)

| 谁 | 管什么 | 不管什么 |
|---|---|---|
| `docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md` | **架构唯一权威**:§2.1 冻结条款 · §12.5 Render ownership · §2.2 UX 偏离账本 · §22.1 规则 1–9 | 进度数字(缝数/状态)一律过期 |
| `tests/**` | **进度唯一权威**:`test_u05_shell_salvage.py::MOVES/TENDRILS`、`migration_active_tests.txt` | 架构 |
| 本文 | **执行顺序 + 机械验收 + 已发现的坑** | **不产生任何架构裁定**。架构问题一律回去读设计文档 |

**铁律:任何架构判断,先亲自打开设计文档对应小节读原文。禁止依据 subagent 的
"某某不存在"这类负向结论做决定**(v1 就是这么废掉的)。

---

## 一、终局与现状(一手核实,2026-07-20)

目标:旧单体 `Zou_lab_control` 迁到六包,**终态零历史残余**,交付实机测试。

**§12.5 的渲染栈已经建成并在用**,不是待办:

```
zlc_frontend/render.py            1219  BoardFrame:1100      worker→presenter hand-off
zlc_frontend/image_view.py         970                       headless 坐标变换/命中
zlc_frontend/matplotlib_render.py 2214                       worker 侧 Agg 渲染
zlc_frontend/encoded_raster.py      51                       不可变 encoded raster 文档
zlc_frontend/image_raster.py       328                       高速 live image→indexed raster
zlc_frontend/selector.py           229                       headless, front-bound 手势
zlc_frontend/qt_widgets/board.py         QtImageBoard:552 / present_encoded:595
```

**两棵 GUI 树各对一半**:

| | UX(布局/控件/tab/工作流/观感) | 渲染与交互架构 |
|---|---|---|
| `Zou_lab_control/frontend/**` 20109 行 | **权威**(§22.1 规则 9,UX 权威 = `main`) | **禁止**:worker 与 Qt canvas/selector 共享 mpl Figure(设计文档 :260 明写不能成为终态) |
| `Zou_lab_control/workbench/**` 18100 行 | **不合格**(实测 20 控件 vs 旧 49、无 Monitor/Logic tab、Add Panel 一个 plot kind 都没有) | **正确**:§12.5 栈的唯一消费者 |

→ **两边都不能整树删,也都不能整树留。**

其它一手数字(权威见 §〇):`Zou_lab_control/` 127 个 .py = 18 转发壳 + 14 新产品面 + 95 legacy;
`MOVES` 18 项;`TENDRILS[task_console]` 18 条;测试 301 个 / 白名单 125 个。

---

## 二、方法(唯一,不再摇摆)

**逐个窗口,把 `main` 的 UX 恢复到已建成的 §12.5 board 上。**
这就是设计文档 UX-003 说的"按真实 consumer 逐个恢复 typed 交互面",分支上
U0.3a-h / W7 已在做,只是没做完——**没做完的部分正是用户打开窗口看到的缺失**。

**画布为界,上下两套规矩:**

- **画布以上(布局/控件/tab/PanelCard/Add Panel 目录/Setting/Edit 表单/工作流/文案/观感)**
  → **照搬旧代码**。这是最快满足"UX 权威 = main"的办法,也是用户明令的做法。
  能 `git mv` 的整块搬,不能整块搬的逐控件抄,**禁止凭印象重画**。
- **画布及其交互层(Figure 所有权 / 渲染线程 / selector / zoom-pan / 命中)**
  → **必须走 §12.5**,绑已建成的栈。**这一层禁止照搬旧代码**——旧的共享-Figure 模型
  是文档点名要替换的东西。

**不丢功能的保证**不来自"跑同一份代码",来自两条硬条款:
§2.1 的 **behaviour salvage gate**(每个 GUI 切片开工前,只读 `main` 旧实现并在
checkpoint 首段冻结行为清单,**未做完不得改目标代码**)+ §12.5:2074
(**每个 `WORKER_RASTER_LIVE` panel 必须按 `main` salvage 清单提供适用的
zoom/pan/crosshair/hover/selector;不适用的 plot kind 必须有旧行为证据,
不能返回空交互句柄冒充完成**)。

**工作单元 = 一个窗口**。每窗一个 commit:恢复 UX → 入口切过去 → 删掉被取代的重画壳
及其专属测试。**禁止只搬基础设施不切入口**(2026-07-20 教训:5 个 commit 全在搬底层,
用户打开的窗口一个没变)。

---

## 三、立即修复(优先于任何窗口)

**F1 守卫已静默失效**:`tests/test_architecture_import_dag.py:910` 仍扫
`frontend/data_figure.py` 找 `_SAVED_FIGURE_VERSION`,该文件已是 17 行壳,真内容在
`zlc_frontend/live_plot/plot_figure.py`。→ repoint,**并把所有"按路径钉死"的守卫改成
按属性扫描**;同类嫌疑 `dag:632`(对不存在目录 `rglob` 返回 `[]` 不报错)。

**F2 pulse-replay 端口只由 legacy 注册**:唯一注册点
`Zou_lab_control/frontend/__init__.py:275`,消费者却是新包 `live_plot/plot_figure.py`;
端口严格单源(第二个不同工厂直接 raise)→ **只能搬不能加**。搬到 `zlc_workbench` 组合根,
并在 `test_u06` 加"产品入口进程里 `pulse_state_factory_is_registered()` 必须为 True"。
否则删旧树后存档 pulse 图静默打不开,而唯一还绿的断言(`test_u06:175`)断言的恰恰是
"回放已坏"。

**F3 两条已断的死引用**:`neutral_atom/session.py:596` → `zlc_workbench.legacy_neutral_atom`、
`_gui.py:113` → `zlc_workbench.pulse_control`,**两模块都不存在**,lazy import 调用才炸,
后者在 pulse GUI 路径上。

**F4 设计文档的幽灵与过期**:`LegacyPanelHost` 在 Z0 删除清单里但全仓 0 命中;§2.2 与
§22 的缝数(38/23/26)全部过期。→ 修正为"数字以测试为准",删幽灵条目。
**不要动 §2.1/§12.5 的条款本身。**

---

## 四、零残余的机械定义(`tests/test_z0_zero_residue.py`)

**每条断言都是对 `git ls-files` 的属性计算,绝不是硬编码名字清单**;失败必须打印
`sorted(offenders)` + 修复命令。本文件进白名单。

| id | 断言 | 今天 |
|---|---|---|
| Z1 | `Zou_lab_control/{frontend,neutral_atom}/` 目录与 tracked 前缀均不存在 | FAIL(预算制) |
| Z2 | 无文件 import `Zou_lab_control.frontend` / `.neutral_atom`(allowlist 只减不增) | **FAIL**:`_gui.py`、`content/manuals.py`、`notes.py`、`docs/task_console_design/build.py:31,33`、`fpga/pulse_streamer/sim/_gen_replay_t.py:9-10` |
| Z3 | 无转发壳(兜底子句:≤40 行且只有 import + `globals().update`) | FAIL(18) |
| Z4 | 六包 AST 无 `Zou_lab_control` 根 | **PASS → 今天就落成棘轮**,并给 `FORBIDDEN` 补 `zlc_workbench` 与 `fpga` 两个缺失 key |
| Z5 | `test_u05_shell_salvage.py` / `test_u04_console_ui_parity.py` **被删除**(非清空) | 末期 |
| Z6 | `LegacyRuntimeFence`/`SerializedLegacyAggBridge` 零定义零引用 | FAIL(`zlc_workbench/legacy.py:87`) |
| Z7 | `Zou_lab_control/workbench/**` 只剩裁定保留件;每个窗口类全仓仅 1 处定义 | FAIL(预算制) |
| Z8 | **白名单 == 全部 `tests/test_*.py`** | **FAIL(125/301)**;反作弊:让"把红测试移出白名单"不可能 |
| Z9 | `test_migration_manifest_gate.py` 删除、`conftest.py` 无 collection 过滤 | 末期,**必须在 Z8 之后** |
| Z10 | §2.2 每行状态 = CLOSED 或 APPROVED+日期 | FAIL(14 行中 13 待批) |
| Z12 | 每个入口 `inspect.getsourcefile` 落在 `zlc_frontend/` 或 `zlc_workbench/` | FAIL(14 入口) |

**预算棘轮**:今天就落 Z1/Z2/Z3/Z4/Z6/Z7/Z8/Z10/Z12,写成 `assert len(offenders) <= BUDGET`,
`BUDGET` **只减不增**,并有一条测试断言文档记录值 == 代码值。**残余从此是一个每 commit
只降不升的数字。** Z5/Z9 末期落。

**同时补**:`test_migration_manifest_gate.py` 增一条——每个**未列**的 `tests/test_*.py`
必须能 import(`py_compile`+`importlib` 冒烟,不跑用例)。今天就会红
(`test_legacy_runtime_fence.py:38` 早已 dead-on-import),红得有价值。

---

## 五、每窗完成定义(D1–D9,全机械,同一 commit 内全绿)

```
D1 行为 gate     开工前只读 main 对应旧实现,在 commit message 首段冻结行为清单
                 (§2.1 硬性前置:未做完不得改目标代码)
D2 UX 等价       把 `git show main:<旧窗口>` 在同进程构造,与新窗口逐项 ==:
                 按钮文本集 / tab 标题 / combo 条目集(含 Add Panel 全部 plot kind)。
                 **是等号,不是冻结缺口集**
D3 交互不空转    §12.5:2074 —— 本窗每个 WORKER_RASTER_LIVE panel 提供适用的
                 zoom/pan/crosshair/hover/selector;不适用的 plot kind 附旧行为证据。
                 **空交互句柄 = 未完成**
D4 入口已绑      inspect.getsourcefile(入口 callable) 在新实现下,不在 Zou_lab_control/workbench/
D5 重画件已删    被取代的 workbench/*.py 不在 git ls-files;其 Qt-free 前段符号
                 仍可 import 且是**同一对象**(identity 断言)
D6 缝已清零      TENDRILS[旧路径] == set();新窗口 AST 任意缩进层级无 Zou_lab_control 根
D7 端口已接      每条缝在 zlc_frontend/domain_ports.py 有五件套,test_u06 钉住,
                 且**未注册时在干净子进程实测拒绝**
D8 真入口 E2E    §2.1 条款 4 / §18.4:真实 launcher headless exit 0,并回放真实输入事件——
                 拖放提交 ROI 且不重启源 Run;缩放/悬停返回数值;Fit 一步完成。
                 **只验控制器状态或只截 PNG 明令不算**
D9 账本与测试    (a) 本窗 §2.2 行已删除或获批;(b) import 被删重画件的白名单测试同
                 commit 删除并移出白名单;(c) 全白名单 collect + 实跑全绿
```

---

## 六、窗口顺序

先做 **F1–F4 + §四 的 Z 守卫**,再逐窗。顺序按"依赖先行"与"用户最先看到"排:

| W | 窗口 | 入口 | UX 源(`main:`) | 被取代的重画壳 | §2.2 |
|---|---|---|---|---|---|
| W1 | Task console | `facade.py:2482`;根 `task_console.py:61` | `frontend/task_console.py` | `_task_console.py`+`_scan.py` | UX-005/013/014 |
| W2 | Pulse 编辑器 | `facade.py:2460`;根 `pulse_gui.py:56` | `frontend/pulse_gui.py` | `_pulse.py` | — |
| W3 | Figure/存档查看 | 根 `figure_viewer.py:37`(仍直连 legacy) | `frontend/figure_viewer.py` | `_figure.py`+`_fit_grid.py` | UX-003/004 |
| W4 | 相机 monitor | `facade.py:805` | task_console 的 PanelCard + 2d/hist/monitor | `_camera_monitor.py` | UX-001/002/012 |
| W5 | 有限 capture | `workbench/__init__.py`(**无 facade 调用者,先补入口**) | task_console 的 MeasurementPanel | `_capture.py` | UX-002 |
| W6 | 校准 | `facade.py:1282,1310,1396` | ⚠ `git show main:` 取回 `operations/tasks/calibrate.py`(**本分支已删**)+ task_console 校准卡片 | `_calibration.py` | UX-006 |
| W7 | 占据 cell | `facade.py:1948` | sites plot kind + PanelCard | `_occupancy.py` | UX-006 |
| W8 | 设备管理器/查看器 | `_gui.py:184,212,248` | `frontend/device_manager.py` | 无重画件 | — |

**注**:`_frozen_raster.py` 按设计文档 :2042 处理——它是 **W4a 历史 checkpoint,不是终态合同**;
**报告类多页保留 frozen raster 并补 zoom(已闭合),其余必须补 source/overlay/
ViewportTransform/交互 selector**。不是删,也不是原样留。
`_window_runtime.py`(129 行,QtCore-only)保留复用。

---

## 七、没有任何账的东西(动 W1 前必须建账)

1. **5 份 PDF 手册的全部源与生成器**(`frontend/notes.py`、`*/content/manual_templates/*.texbody`、
   `templates/*.sty`、现场调 legacy API 产图的 `content/manuals.py`)——六包内零对应物。
2. **6 个 tutorial notebook 100% legacy import**,生成器与 5 个 `.cells.md` 全在旧树;
   同步守卫不在白名单;且 `neutral_atom/__init__.py` 已是空 legacy island(`__all__=[]`)
   而 tutorial 还在调 `na.connect(...)` → **tutorial 现在就是坏的**。
3. **`fpga/` 双向耦合且不受任何 import 规则管**:`sim/_gen_replay_t.py:9-10` import 旧树,
   而 `zlc_pulse`/`zlc_neutral_atom` 有 11 处 import `fpga`。
4. **3 个设备 config 模板** `neutral_atom/configs/*.json`(`devices/registry.py:328` 按名装载),
   新树无对应机制;`pyproject.toml` 仍声明旧路径为包数据。
5. **三个顶层 seam** `_clock.py`/`_streamer_geometry.py`/`_viewer_registry.py`:无账、10 处活跃 consumer。
6. **根启动器 `figure_viewer.py` + `.bat`**:发布的双击入口,直连 legacy,不在任何账上。
7. **`README.md:89-90`** 仍教用户跑 legacy 命令。

---

## 八、纪律

- 每切片 = 行为 gate + 实现 + 独立 oracle 测试 + 一次对抗审查 + 精确 commit;
  详细英文 commit message,末尾 `Co-Authored-By: <agent 名> <noreply@anthropic.com>`;
  **绝不 push**;全中文回复(代码/commit/token 除外)。
- **每个 commit 前跑 index 导出全白名单**(一文件一进程,`InstallationRuntime` 进程级单例):
  ```bash
  S=<scratch>; rm -rf $S/tree; mkdir -p $S/tree
  git checkout-index -a -f --prefix=$S/tree/
  cd $S/tree && git init -q . && git add -A && git commit -qm export
  while read f; do f=$(echo "$f" | tr -d '\r'); [ -z "$f" ] && continue
    python -m pytest "$f" -q -p no:cacheprovider || echo "FAIL $f"
  done < tests/migration_active_tests.txt
  ```
- RTL/bitstream 冻结(不跑 Vivado build/program;xsim 可);物理算法只从 `main`;
  依赖闭合删除、无后向兼容;做不动 → 拆更小切片,**绝不绕过守卫、绝不把红测试移出白名单**。
- **绝不用 `git checkout <path>` 清理试探改动**(会连未提交真改动一起回滚,已犯过)。
- 中途不等确认,自己判断最优决定。

---

## 九、v1 为什么废掉(规则来源,别再犯)

v1 的四个 DECISION 全错,根因是**用 subagent 的结论替代了读原文**:agent 搜字面量
`S12.5`、文档写 `§12.5`,它报"全仓无定义",我未核实即采信,并在其上叠了两个架构裁定:
- 假事实:"S12.5 无定义" → 真相:`§12.5 Render ownership` 在第 2034 行,
  `WORKER_RASTER_LIVE` 是 §2.1 五条**冻结条款**之一(:135)。v1 还提议删掉这个"悬空引用"。
- 伪二选一:"活 mpl 画布 vs 冻结 PNG" → 真相是三种 render surface
  (`GUI_ARTIST` / `WORKER_RASTER_LIVE` / `WORKER_HEADLESS_EXPORT`),
  且 §12.5 栈**早已建成**(约 5200 行),裁什么都是多余。
- 由此误判 `Zou_lab_control/workbench/**` 该整树删 → 真相:它是 §12.5 栈的唯一消费者,
  错的是它的 **UX**,不是它的架构。

**改成的规则**:① 架构问题一律先亲自读设计文档对应小节;② subagent 的负向存在性结论
(某某"不存在")一律不作为决策依据,必须一手复核;③ 本文不产生架构裁定。

## 十、终态

Z1–Z12 全绿 + 全白名单实跑全绿 + §2.2 账本清零/获批 → 输出《迁移完成报告》(首页列未获批
偏离、上机第一天测试清单、已知限制、runbook 位置),然后停止。
