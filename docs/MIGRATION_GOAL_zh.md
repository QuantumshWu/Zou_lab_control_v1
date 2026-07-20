# 迁移 Goal(可循环执行版)

> 本文即 `/goal` 正文。写作对象是接手的 agent(含未来的我)。
> **权威顺序:本文 > `docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md` > 其它文档。**
> 凡数字与账本,**以测试文件为准,文档里的数字一律视为过期**(已实测:设计文档说
> task_console 38 条缝、交接文档说 26,`tests/test_u05_shell_salvage.py::TENDRILS` 说 18)。

---

## 一、使命

分支 `codex/system-architecture-migration`。把旧单体 `Zou_lab_control` 迁到六包
(`zlc_data`/`zlc_storage`/`zlc_pulse`/`zlc_neutral_atom`/`zlc_frontend`/`zlc_workbench`),
**终态零历史残余**,交付用户实机测试。

**唯一方法:GUI 一律 shell salvage,禁止从零重画。** 旧 GUI 文件本体就是骨架:
`git mv` 进目标包 → 数据逻辑换成端口注入 → 旧路径留转发壳。保真度由"跑同一份代码"
构造保证,**不由测试对账保证**。此前从零重建的产物(`Zou_lab_control/workbench/**`)
已被用户否决,按依赖闭合删除。

**工作单元 = 一个窗口,不是一个模块。** 每个窗口一个 commit,同时做完:
① 旧 GUI 文件搬进新包 → ② 入口切到它 → ③ 删掉对应的重画件 + 其专属测试。
**禁止只搬基础设施而不切入口**——那会让用户看不到任何变化(2026-07-20 的教训:
H1a–H2a 搬了 5 个 commit,用户打开的窗口一个都没变)。

---

## 二、现状(机械核过,2026-07-20,HEAD `f03b97a`)

| 事实 | 数值 | 权威 |
|---|---|---|
| `Zou_lab_control/` .py 总数 | 127 = 18 SHIM + 14 新产品面 + 95 LEGACY | AST 扫描 |
| `frontend/` | 23 .py / 20109 行 = 8 SHIM + 15 LEGACY | 同上 |
| `Zou_lab_control/workbench/`(重画件) | 12 .py / 18100 行 | 同上 |
| 已搬模块 | **18** | `tests/test_u05_shell_salvage.py::MOVES` |
| task_console 剩余缝 | **18** | 同文件 `TENDRILS` |
| 测试文件 / 白名单 | **301 / 125** → 176 个不可 collect | `tests/migration_active_tests.txt` |

**关键结构事实(纠正此前误解):**
- `Zou_lab_control/workbench/**` **没有一个纯数据适配文件**。5 个纯 GUI 壳、5 个混合
  (Qt-free 投影前段 + Qt 壳)、2 个基础设施(`__init__.py` 懒入口、`_window_runtime.py`)。
- **真正的"新后端"在兄弟包 `zlc_workbench/`(6605 行)** —— salvage 壳要绑的是它。
  `Zou_lab_control/workbench/` 只是壳,该死;`zlc_workbench/` 是数据面,该活。
- 混合件里的 Qt-free 前段(总计约 3235 行)**必须抢救**,不能随壳一起删。逐个行号见 §六。

---

## 三、四个阻塞裁决(**动手前必须解决,否则 salvage 走不通**)

> 用户已授权自主判断。下列是我的裁定与理由;用户可随时否决,否决即改。
> 每条都在 `tests/test_z0_window_done.py` 里落成 `pytest.fail("DECISION-n unresolved")`,
> 未裁决的窗口**不可能报绿**。

### DECISION-1:Qt+matplotlib 窗口的合法新家 → 新建 `zlc_frontend/windows/`

**问题**:`task_console.py`(10036)、`pulse_gui.py`(4503)、`figure_viewer.py`(1033)、
`device_manager.py`(1234)、`render_loop.py`(205)、`qt_canvas.py`(374) 同时依赖 PyQt5 与
matplotlib。现有公理:`qt_widgets/` 是唯一 Qt 拥有者**且禁 matplotlib**;`live_plot/` 是
matplotlib **且 Qt-free**。→ **桌面窗口无处可放**。挂着的解禁条款 `S12.5` 在全仓**无任何定义**。

**裁定**:新建 `zlc_frontend/windows/`,**允许同时 import PyQt5 与 matplotlib**,
并加单向守卫:`qt_widgets/` 与 `live_plot/` **不得 import `windows/`**(依赖单向,窗口是叶子)。
**理由**:桌面窗口本来就是两者的联姻点,把这个事实写成一个具名包比让六个文件永远无家可归诚实。
删除 `S12.5` 这个悬空引用及其三处前向提及。

### DECISION-2:呈现模型 → 活 matplotlib 画布(旧架构模型)胜出

**问题**:`Zou_lab_control/workbench/_frozen_raster.py`(304 行)是"冻结 PNG 分页呈现"模型,
**旧树无任何对应物**(旧架构是活画布 `qt_canvas.py:358 panel_canvas`),却是 figure/
fit-grid/calibration 三窗口的基类。

**裁定**:按 §22.1 规则 9(UX 权威 = `main`),salvage 壳用**活画布**。`_frozen_raster.py`
及其三个派生窗口随各自窗口删除。**理由**:用户验收的是旧 UI 的操作手感;冻结 PNG 无法交付
"拖选即出 ROI""缩放/悬停有数值"这些 §2.1 冻结条款要求的交互。
**若用户要保留冻结 PNG(如为远程/低带宽),此裁定作废,W4/W5/W6 改为双模式。**

### DECISION-3:W6 校准窗口的 salvage 源已在本分支被删

**事实**:`neutral_atom/operations/tasks/calibrate.py` **在 `main` 上存在,本分支已删**。
旧校准 GUI 是 `frontend/task_console.py` 里的 Task 卡片。

**裁定**:W6 的 oracle = `git show main:` 的两个文件(task 文件 + task_console 校准分支)。
先只读取回,再 salvage。**在取回并逐行比对前,W6 不得开工。**

### DECISION-4:W4/W5 争夺同一个 salvage 源

`frontend/figure_viewer.py`(1033 行)同时是 DataFigure 窗口与 saved-fit-grid 窗口的源。
**裁定**:先做 W4(DataFigure),W5 作为 W4 的一个 tab/模式内联进去,不拆两个窗口——
因为旧架构里它们本来就是同一个窗口的两条路径。

---

## 四、立即修复(我在 H1d/H1e 造成的两处,优先于任何新窗口)

**F1 — 守卫已静默失效。** `tests/test_architecture_import_dag.py:910` 仍扫
`Zou_lab_control/frontend/data_figure.py` 找 `_SAVED_FIGURE_VERSION`,而该文件在 H1e 后
已是 17 行转发壳,真内容在 `zlc_frontend/live_plot/plot_figure.py`。→ repoint 到新路径,
**并把这类"按路径钉死"的守卫全部改成按属性扫描**(同一文件 `:561-567` 我写的注释正好警告
过这个失效模式,却没做到)。同类嫌疑:`dag:632` 对不存在目录 `rglob` 返回 `[]` 不报错。

**F2 — pulse-replay 端口只由 legacy 注册。** 全仓唯一注册点
`Zou_lab_control/frontend/__init__.py:275`,消费者却是新包
`zlc_frontend/live_plot/plot_figure.py`。端口是严格单源(第二个不同工厂直接 raise),
**只能搬不能加**。→ 注册点搬到 `zlc_workbench` 组合根,并在
`tests/test_u06_shell_domain_ports.py` 加一条:**产品入口进程里 `pulse_state_factory_is_registered()`
必须为 True**。否则删旧树后存档 pulse 图静默打不开,而唯一还绿的断言(`:175`)断言的
恰恰是"回放已坏"。

**F3 — 两条已断的死引用。** `neutral_atom/session.py:596` → `zlc_workbench.legacy_neutral_atom`、
`_gui.py:113` → `zlc_workbench.pulse_control`,**两个模块都不存在**,函数内 lazy import,
调用才炸,`_gui.py:113` 在 pulse GUI 路径上。→ 删或改指向真实模块。

---

## 五、零残余的机械定义:`tests/test_z0_zero_residue.py`

**设计铁律:每条断言都是对 `git ls-files` 的属性计算,绝不是硬编码名字清单**
(名字清单抓不到没人记得的文件)。每条失败必须打印 `sorted(offenders)` 与修复命令。
本文件进白名单。

| id | 断言 | 今天状态 |
|---|---|---|
| Z1 | `Zou_lab_control/{frontend,neutral_atom}/` 目录不存在 **且** `git ls-files` 无该前缀 | FAIL(预算制) |
| Z2 | 无任何文件 import `Zou_lab_control.frontend` / `.neutral_atom`(允许清单只减不增) | **FAIL**:`_gui.py`、`content/manuals.py`、`notes.py`、`docs/task_console_design/build.py:31,33`、`fpga/pulse_streamer/sim/_gen_replay_t.py:9-10` |
| Z3 | 无转发壳(含"≤40 行且只有 import + `globals().update`"的兜底子句) | FAIL(今 18 个) |
| Z4 | 六包 AST 无 `Zou_lab_control` 根 | **PASS,今天就落成棘轮**,并给 `FORBIDDEN` 补上缺失的 `zlc_workbench` key |
| Z5 | `test_u05_shell_salvage.py` / `test_u04_console_ui_parity.py` **被删除**(不是清空) | 末期 |
| Z6 | `LegacyPanelHost`/`LegacyRuntimeFence`/`SerializedLegacyAggBridge` 零定义零引用 | FAIL(`zlc_workbench/legacy.py:87`);`LegacyPanelHost` 是幽灵条目,全仓 0 命中→从设计文档删掉 |
| Z7 | `Zou_lab_control/workbench/**` 为空(或仅剩裁定保留的 `_window_runtime.py`);每个窗口类全仓只有 1 个定义 | FAIL(预算制) |
| Z8 | **白名单 == 全部 `tests/test_*.py`** | **FAIL(125/301)**;这是反作弊条款:让"把红测试移出白名单"不再可能 |
| Z9 | `test_migration_manifest_gate.py` 删除、`conftest.py` 无 collection 过滤 | 末期,**必须在 Z8 之后** |
| Z10 | 设计文档 §2.2 每行状态为 CLOSED 或 APPROVED+日期 | FAIL(14 行中 13 行待批) |
| Z11 | 每个 salvage 件 `git log --follow` 可达 `main@6c337d49`(声明的 oracle 基线) | 首个窗口落地后可写 |
| Z12 | 每个入口 `inspect.getsourcefile` 落在 `zlc_frontend/` 或 `zlc_workbench/`,**不在** `Zou_lab_control/workbench/` | FAIL(14 个入口) |

**预算棘轮**:今天就落 Z1/Z2/Z3/Z4/Z6/Z7/Z8/Z10/Z12,写成
`assert len(offenders) <= BUDGET`,`BUDGET` 是**只减不增**的字面量,并有一条测试断言
文档记录的数字与代码一致。**残余从此是一个每 commit 只降不升的数字**,不是一段散文。
Z5/Z9/Z11 末期落。

**同时补 G2**:`test_migration_manifest_gate.py` 增一条——每个**未列**的
`tests/test_*.py` 必须能 import(纯 `py_compile`+`importlib` 冒烟,不跑用例)。
这条今天就会红(`test_legacy_runtime_fence.py:38` 早已 dead-on-import),红得有价值。

---

## 六、每个窗口的完成定义(D1–D10,全机械)

一个窗口 DONE 当且仅当十条在**同一个 commit** 内全绿:
`pytest tests/test_z0_window_done.py -k W<n>`。

```
D1  入口已绑     inspect.getsourcefile(入口 callable) 在 salvage 路径下
D2  重画件已删   对应 Zou_lab_control/workbench/*.py 不在 git ls-files
D3  血统可溯     git log --follow 可达 main@6c337d49;且 salvage diff 只触及
                 SALVAGE_EDITS[W<n>] 枚举的 hunk(import 重指 + 领域调用点换端口)。
                 出现其它改动 = 重写,"构造性保真"作废
D4  旧路径留壳   ≤40 行 + "MOVED to",计入 Z3 预算,Z0 删
D5  缝已清零     TENDRILS[旧路径] == set() 且 salvage 件 AST 任意缩进层级
                 都不出现 Zou_lab_control 根
D6  端口已接     每条缝在 zlc_frontend/domain_ports.py 有五件套(register_* /
                 *_is_registered / 指名修复方法的 typed 拒绝 / __all__ / 组合根调用),
                 在 test_u06 钉住,且**未注册时在干净子进程里实测拒绝**
D7  适配已抢救   被删重画件里的 Qt-free 前段符号仍可 import,且是**同一对象**
                 (identity 断言,仿 test_u05:86)
D8  控件等价     把 `git show main:<旧路径>` 在同进程构造,与 salvage 窗口逐项 ==:
                 按钮文本集 / tab 标题 / combo 条目集 / findChildren 计数。
                 **是等号,不是冻结缺口集**。salvage 下这近乎恒等式——这正是重点:
                 零成本,且能抓住搬砸的 git mv
D9  真入口 E2E   真实入口 headless 跑通 exit 0,并按 §2.1 冻结条款回放真实输入事件:
                 拖放提交 ROI 且不重启源 Run;缩放/悬停返回数值;Fit 一步完成。
                 **只验控制器状态或只截 PNG 不算**
D10 账本与测试   (a) 本窗口拥有的 §2.2 行已删除或已批准;(b) 每个 import 被删重画件的
                 白名单测试**同 commit 删除并移出白名单**;(c) 全白名单 collect+跑全绿
```

---

## 七、窗口清单与顺序

> 顺序铁律:**一个文件的所有前端依赖必须先于它搬**(`zlc_frontend` 禁 import
> `Zou_lab_control`)。交接文档原来写反了,已纠正。

| W | 窗口 | 入口 | 删除的重画件 | salvage 源(oracle=`main:`) | 抢救行段 | 阻塞 |
|---|---|---|---|---|---|---|
| W0 | — | — | — | **F1/F2/F3 + Z 守卫落地** | — | 无,**先做** |
| W1 | Task console | `facade.py:2482`;根 `task_console.py:61` | `_task_console.py` 1495 + `_scan.py` | `frontend/task_console.py` | `_scan.py:73-181` | D-1 |
| W2 | Pulse 编辑器 | `facade.py:2460`;根 `pulse_gui.py:56` | `_pulse.py` 1963 | `frontend/pulse_gui.py` 4503 | — | D-1 |
| W3 | Figure viewer | 根 `figure_viewer.py:37`(**已是 legacy**) | — | 自身 | — | D-1 |
| W4 | DataFigure(含 W5 fit-grid) | `facade.py:3536,3479` | `_figure.py` 4608 + `_fit_grid.py` 2473 + `_frozen_raster.py` | `frontend/figure_viewer.py` + plot_figure | `_figure.py:156-1884`、`_fit_grid.py:103-823` | D-1,D-2,D-4 |
| W6 | 校准 | `facade.py:1282,1310,1396` | `_calibration.py` 890 | ⚠ `main:` 取回 | `:44-284` | D-3 |
| W7 | 相机 monitor | `facade.py:805` | `_camera_monitor.py` 3171 | task_console 的 PanelCard+2d/hist/monitor | `:131-461` | D-1 |
| W8 | 有限 capture | `workbench/__init__.py`(**无 facade 调用者**) | `_capture.py` 1233 | task_console 的 MeasurementPanel | — | 先裁"给不给 facade 入口" |
| W9 | 占据 cell | `facade.py:1948` | `_occupancy.py` 724 | sites plot kind + PanelCard | `:50-153` | D-1 |
| W11 | 设备管理器 | `_gui.py:184,248` | 无重画件 | 自身 1234 | — | D-1 |
| W12 | 设备查看器 | `_gui.py:212` | 无重画件 | 自身 | — | D-1 |

**保留不删**:`Zou_lab_control/workbench/_window_runtime.py`(129 行,QtCore-only 的进程级
窗口管理,任何 salvage 壳都能用);`Zou_lab_control/{notebook,workbench}/__init__` 作为入口面
(`__init__.py` 是**重写入口指向**,不是删除)。

---

## 八、没有任何账的东西(**必须先建账,否则 Z0 一删就没了**)

按严重度,每条必须在动 W1 之前拿到归宿或 owner:

1. **5 份 PDF 手册的全部源与生成器**:`frontend/notes.py`、`neutral_atom/notes.py`、
   `*/content/manual_templates/*.texbody`、`frontend/templates/*.sty`、
   现场调 legacy API 产图的 `content/manuals.py`。六包内**零**对应物。
2. **6 个 tutorial notebook 100% legacy import**,生成器 `frontend/jupyter.py` +
   `content/tutorials.py` + 5 个 `.cells.md` 全在旧树;同步守卫
   `test_tutorial_notebooks_in_sync.py` **不在白名单** → 已不可运行。
   且 `neutral_atom/__init__.py` 已被声明为空 legacy island(`__all__ = []`),
   而所有 tutorial 都在调 `na.connect(...)` → **tutorial 现在就是坏的**。
3. **`fpga/` 双向耦合且完全不受 import 规则管**:`fpga/.../sim/_gen_replay_t.py:9-10`
   import 旧树,而 `zlc_pulse`/`zlc_neutral_atom` 有 11 处 import `fpga`。
   → 给 `FORBIDDEN` 补 `fpga` 与 `zlc_workbench` 两个 key。
4. **3 个设备 config 模板** `neutral_atom/configs/*.json`(由 `devices/registry.py:328`
   按名装载),`zlc_neutral_atom` 无对应机制;`pyproject.toml` 仍把旧路径声明为包数据。
5. **三个顶层 seam** `_clock.py`/`_streamer_geometry.py`/`_viewer_registry.py`:
   无 MOVES 行、无文档提及,却有 10 处活跃 consumer,且自身 import `fpga`。
6. **根启动器 `figure_viewer.py` + `figure_viewer.bat`**:发布的双击入口,直连 legacy,
   不在任何账上。
7. **`README.md:89-90`** 仍教用户跑 legacy 命令。

---

## 九、纪律

- 每切片 = 实现 + 独立 oracle 测试 + 一次对抗审查 + 精确 commit;
  commit message 详细英文,末尾 `Co-Authored-By: <agent 名> <noreply@anthropic.com>`;
  **绝不 push**;全中文回复(代码/commit/token 除外)。
- **每个 commit 前必跑 index 导出全白名单**(一文件一进程,`InstallationRuntime` 是进程级单例):
  ```bash
  S=<scratch>; rm -rf $S/tree; mkdir -p $S/tree
  git checkout-index -a -f --prefix=$S/tree/
  cd $S/tree && git init -q . && git add -A && git commit -qm export
  while read f; do f=$(echo "$f" | tr -d '\r'); [ -z "$f" ] && continue
    python -m pytest "$f" -q -p no:cacheprovider || echo "FAIL $f"
  done < tests/migration_active_tests.txt
  ```
- RTL/bitstream 冻结(不跑 Vivado build/program;xsim 可);物理算法只从 `main` 取;
  依赖闭合删除、无后向兼容;做不动 → 拆更小切片,**绝不绕过守卫、绝不把红测试移出白名单**。
- **绝不用 `git checkout <path>` 清理试探性改动**——它会连未提交的真改动一起回滚(已犯过一次)。
  用精确逆向 patch。
- 中途不等用户确认,自己判断最优决定;但 §三 的四个裁决若被用户否决,立即改。

## 十、终态

Z1–Z12 全绿 + 全白名单实跑全绿 + §2.2 账本清零/获批 → 输出《迁移完成报告》:
首页列未获批偏离、用户上机第一天测试清单(GUI→virtual scan→真机资格化)、已知限制、
runbook 位置(`docs/REAL_HARDWARE_BRINGUP_zh.md`),然后停止。
