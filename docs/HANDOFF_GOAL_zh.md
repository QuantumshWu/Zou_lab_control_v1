# 移交 Goal:GUI shell salvage 完成迁移(给下一个主 agent)

> 本文即 `/goal` 正文。写作对象是接手的 Opus 4.8 agent:所有步骤已拆到机械可执行,
> 不需要重新做架构判断;需要判断的地方都已给出判据。

---

## 【使命与权威】

你在分支 `codex/system-architecture-migration` 上,继续把旧单体 `Zou_lab_control`
迁移到六包架构(zlc_data/zlc_storage/zlc_pulse/zlc_neutral_atom/zlc_frontend/zlc_workbench)。
唯一计划权威 = `docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md`(§22.1 规则 1-9 + S4/S5-shell 进度账本)。
先读该文档与本文件恢复上下文,从最新 git log 继续,**不重做已闭合切片**。
终点 = 完成本 goal 全部清单后输出《迁移完成报告》并停止,交付用户实机测试。

commit 纪律:分阶段精确提交、详细英文 message、末尾
`Co-Authored-By: <你的 agent 名> <noreply@anthropic.com>`、**绝不 push**;
全中文回复(代码/commit 除外);自主推进不问用户;上下文压缩后凭本文 + 设计文档 + git log 自行恢复。

## 【已定案的方法——不允许偏离】

**用户明令(2026-07-20,最高优先级):GUI 一律 shell salvage,禁止从零重画。**
旧 GUI 文件本体就是骨架:展示代码原样搬进目标包,旧路径留转发壳,新旧跑同一份代码,
外观/交互保真度由构造保证。此前"从零重建 + 对账清单"路线已作废(账本有记录)。

**已验证的搬家配方(照抄,勿发明新方法):**
1. `git mv 旧文件 新家`;
2. 旧路径写转发壳(样板见 `Zou_lab_control/frontend/canvas.py`:
   `import 新模块 as _moved` + `globals().update(...)`,连下划线名和 `__all__` 一起转发);
3. 被 DAG 守卫拦下的符号:验证器重名 → 移动件内改名 + 壳内恒等别名;
   `.strip()` → 按 named-adapter 先例登记进 `tests/test_architecture_import_dag.py` 的
   `ALLOWED_STRIP_CONTEXTS`;领域注册动作(如 fit-guard)→ 留在壳侧执行;
4. `tests/test_u05_shell_salvage.py` 的 `MOVES` 表加行(恒等 + 壳纯度自动覆盖);
5. index 导出跑全白名单(命令模板见下),绿了才 commit。

**验证命令模板(每个 commit 前必跑):**
```bash
S=<scratch>; rm -rf $S/tree; mkdir -p $S/tree
git checkout-index -a -f --prefix=$S/tree/
cd $S/tree && git init -q . && git add -A && git commit -qm export
while read f; do f=$(echo "$f" | tr -d '\r'); [ -z "$f" ] && continue
  python -m pytest "$f" -q -p no:cacheprovider || echo "FAIL $f"
done < tests/migration_active_tests.txt
```
一文件一进程是硬约束(InstallationRuntime 进程级单例)。

## 【H1 已闭合(2026-07-20,commit 0ac4b79→626bf40)】

live.py **已搬完**,缝清零。五个 commit 的可复用结论:

- **两种切缝法,判据是"能不能搬"而非"想不想搬"**。纯函数/描述/声明 → 整模块下沉 `zlc_data`
  (H1a/H1b/H1c 共下沉 10 个模块:readout_math/facet/signal_tensor/param_decl/raster/
  plot_region/curve_fitting/figure_capture…)。搬不动的两类必须**反转依赖**:
  ① 需要"活对象回答自己"→ 让领域对象长出方法(`PulseTableState.analog_bus_samples`,
  渲染层问它已经拿在手里的 state,零 import);
  ② 需要"构造一个活对象"→ 构造期注入(`zlc_frontend/domain_ports.py`,组合根注册工厂,
  未注册给 typed 拒绝)。**H2 的端口就加在 `domain_ports.py`,别新开文件**;该文件顶部
  写死了准入规则(只收"渲染层需要但造不出的活对象")。
- **交接文档原来写错一条**:「`.data_figure`/`._watcher` 随壳留旧侧,live 搬走后经转发壳回引」
  是**违反 DAG 的**——`zlc_frontend` 禁 import `Zou_lab_control`,搬过去的件绝不能回引旧树。
  正确顺序:一个文件的**所有前端依赖必须先于它搬**。H2 同理:task_console 依赖的
  `pulse_gui`/`figure_viewer`/`qt_fluent`/`render_loop` 等要先排依赖序。
- **同名不同概念已出现两次**(`Selection`→`plot_region`、`data_figure`→`plot_figure`)。
  撞名一律**按概念给模块改名**,不改类名,并在移动件开头写 NAME WARNING;别让一个名字担两个意思。
- **搬家会暴露旧树里靠 import 顺序遮住的真实循环**。H1e 就撞上 `core` ↔ `timing`
  (results.py 顶层 import PulseSequence,而 timing.sequence import core.analysis;
  旧 data_figure 顶层 import `core.*` 恰好让 core 先初始化)。根因修=**纯类型依赖降为
  `TYPE_CHECKING`**,不是调 import 顺序。同类症状再现按此处理。
- 新守卫两条:`zlc_data/zlc_storage/zlc_pulse/zlc_neutral_atom/zlc_workbench` **一律不得
  import matplotlib/PyQt5/PySide/tkinter**(五包今天全清白,是棘轮);
  `tests/test_u06_shell_domain_ports.py` = 对象端口契约的家,H2 的端口测试写这里。
- **教训(我犯的)**:`git checkout <path>` 会连未提交改动一起回滚。清理试探性改动只用
  精确逆向 patch,绝不在复合命令里放 `git checkout`。

## 【当前状态(2026-07-20)】

- **已搬家 17 个模块**(权威=`tests/test_u05_shell_salvage.py::MOVES`,别照抄本文列表):
  `zlc_frontend/live_plot/{canvas,selectors,ticks,_validate,_watcher,plot_figure,live}`、
  `zlc_frontend/qt_widgets/param_widgets.py`(包属性可达,不进 `__all__`)、`zlc_storage/paths.py`、
  `zlc_data/{readout_math,facet,signal_tensor,param_decl,raster,plot_region,curve_fitting,figure_capture}`。
  旧 console 已实测跑在这份代码上(9 按钮/Monitor+Logic 双 tab 原样)。
- **放置公理(守卫机械强制,搬任何 GUI 文件前先对号)**:
  ① `zlc_frontend` 内**只有** `qt_widgets/` 可 import PyQt5(`test_zlc_frontend_qt_widgets` qt_leaks 条款);
  ② `qt_widgets/` **不可** import matplotlib(reverse-dependency 条款);
  ③ 包外不得 deep-import `zlc_frontend.qt_widgets.<submodule>`,也不得 from-import 非 `__all__` 名——
     子模块收编走「`__init__` 末尾 `from . import X`+属性访问」先例(param_widgets 即样板);
  ④ 纯 mpl 件 → `live_plot/`;纯 Qt 件 → `qt_widgets/`;Qt+mpl 联姻件(`render_loop/qt_canvas`)
     **留旧树**,直到壳改用 worker-raster(S12.5)才有合法新家。
- **只剩一个壳未搬**:`Zou_lab_control/frontend/task_console.py`(10036 行,**26 条**领域缝)。
  缝台账在 `test_u05_shell_salvage.py::TENDRILS`,**双向棘轮**:长新缝立即红;
  接掉一条必须同 commit 删行。live.py 已闭合并搬走。
- **workbench 新组件保留**(`Zou_lab_control/workbench/_task_console.py` 等):
  N 面板/状态条/METER owner/display-revision-by-intent 已落,最终由 shell 接管入口后按依赖闭合处置。
- `tests/test_u04_console_ui_parity.py`:新 workbench console vs 旧 console 的控件对账棘轮,
  shell 接管入口后应改为「入口窗口 == shell 窗口」的恒等断言。
- **真机边界已完成件**:>4096 点 typed 拒绝 `FormalScanCapacityExceeded` + runbook §5
  (R1–R6 资格化实验,9999 点路径);报告页 zoom;`docs/REAL_HARDWARE_BRINGUP_zh.md` 为 runbook 底稿。

## 【封闭清单(按序做完即终点)】

**H1. live.py 搬家——已闭合**(见上「H1 已闭合」节;`test_u05_shell_salvage.MOVES` 17 项为准)。
**H2. task_console.py 搬家**——26 条缝分两类:
   a. 与 H1 同源的数学/声明缝(fitting/selection/raster/signal_tensor/params/facet)**已随 H1 消失**;
   b. 真领域缝(signal_expr 8 条、logic 5 条、measurement(s) 6 条、processors 3 条、timing 2 条、
      signals.NO_LINEAGE、task.DEFAULT_MID_RUN_KEY):在**已存在的** `zlc_frontend/domain_ports.py` 加
      Protocol/常量端口(H1d 已立此文件与准入规则,勿另开 console_ports.py),壳改为构造期注入;旧入口 `show_task_console` 在旧侧组装旧实现作默认注入
      (行为零变),新入口由 `zlc_workbench` 适配层用新数据面(DefinitionCatalog/LiveDatasetSlot 等)实现同一端口。
      **判据:凡是"读文本/常量/纯函数"直接下沉;凡是"活对象(hub/node/spec)"走端口注入。**
   c. 搬家后 `Zou_lab_control/workbench` 的入口(`exp.task_console()`/launcher `task_console.py`)
      切到 shell + 新适配层;`test_u04_console_ui_parity.py` 改为恒等断言。
**H3. 其余 GUI 同配方**:`pulse_gui.py`(4503 行)、`figure_viewer.py`、
   `device_manager.py`、`session.py`/`jupyter.py` 按依赖顺序搬;每个先跑缝扫描
   (`test_u05` 里的 `_scan_tendrils` 即工具),小缝直接接,活对象走端口。
**H4. 原 goal 清单 5–7**:figure_viewer 通用存档浏览器、device manager + launcher 切换、
   E01 temperature/MOT/fidelity operations 迁移(物理算法只从 main 取)。
**H5. 清单 8**:白名单 collect 错误清零(错误清单只减不增)。
**H6. 清单 9(Z0)**:删全部转发壳与旧树 legacy(frontend/neutral_atom 旧部分;
   `Zou_lab_control/workbench` 与 `notebook` 是新产品面,不删),零残余验证。
**H7. 清单 10**:全分支对抗终审 P0=0/P1=0 + 白名单全绿实跑 + UX 偏离账本(UX-001…UX-014)随
   《迁移完成报告》呈交,未批准项列首页;报告含用户上机第一天测试清单
   (GUI→virtual scan→真机资格化顺序)、已知限制、runbook 位置(`docs/REAL_HARDWARE_BRINGUP_zh.md`)。

## 【纪律(不加戏)】

每切片 = 实现 + 独立 oracle 测试 + 一次对抗审查 + 精确 commit;RTL/bitstream 冻结
(不跑 Vivado build/program;xsim 可);物理算法只从 main;依赖闭合删除、无后向兼容;
不碰白名单外历史测试(它们冻结,Z0 一并删);GUI 视觉验收 = 真窗口三档 DPR 截图,
但 shell salvage 下首选证据是"同一份代码"的构造性保真 + `test_u05` 恒等;
中途遇到任何问题不要等用户确认,自己判断最优决定;做不动 → 拆更小切片,绝不绕过守卫。

## 【终态交付与停止】

全部完成后输出《迁移完成报告》(内容见 H7),然后停止。
