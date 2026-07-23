# 设计宪法(法条权威,2026-07-20 立)

> **每轮开工必须全文读完本文件**——它被刻意压在 300 行以内,读不完就是它写坏了。
> 权威层级:**宪法(本文,法)> 台账(`MIGRATION_LEDGER_zh.md`,进度)> 设计文档(`SYSTEM_ARCHITECTURE_DESIGN_zh.md`,叙事与细节)**。
> 冲突以宪法为准;修宪必须经用户批准。法条来源标注为文档行号(L)或台账条目;
> 从台账行**提升**进宪法的法条,以宪法文本为最终措辞。
> 机械守卫:`tests/test_design_charter.py`。

## A. 权威与流程(治"不读文档/自我发明"的病根)

- **C1** 计划级权威只有一条链:宪法→台账→设计文档。**绝不新建任何计划/goal/roadmap 文档**(血训:L3855 明写唯一权威时另立 goal 文档,整日返工)。
- **C2** 台账新行(2026-07-20 起的"新台账"节)**≤5 行文本、≤700 字符**,必须引用 ≥1 个宪法条号 + 当轮新测量;**禁止引用其他台账行作为架构依据**——台账记录事实,不产生法(血训:「壳必须搬进 zlc_frontend」是台账行自我引用 25 轮的漂移,设计文档从未这么要求)。
- **C3** 每轮开工仪式四步,顺序不可换:(1) 读宪法全文;(2) 只读台账 OPEN 项;(3) **当轮重新推导**本刀依赖的代码事实(AST/grep,不引记忆不引台账);(4) 动手前写下切口 + 机械验收判据。
- **C4** 被反复依赖的架构宣称必须冻成**再推导测试**(每次运行从源码重算,如 `test_the_render_worker_is_not_married_to_matplotlib`)。对两个对象的合并判断不是对其中任何一个的证据。
- **C5** 守卫/测量翻红或给出惊人结论时,**先怀疑判据本身**。判据锚在唯一特征上:AST 不用子串(`config` 含 `fig`;`from .live import` 的点在 `node.level`;`def` 不产生 `ast.Name`,死码阈值是 `==0`;全仓按名计数分不清同名重复;先插注释再计数会数到自己)。
- **C7 领域法索引**:宪法只收每轮必用的过程/结构法。领域法留在设计文档各节,**切入该子系统前必须先读对应节**(C3 第 3 步的一部分):formal gate/INVALID 不可修补(§1.1 区,L36/L97)、bitstream 冻结(L30)、自主执行唯一性(L31)、硬件拥有精密时序(L33)、CommittedTransform(L114)、显式 validity(L228)、单 InstallationRuntime/重启换配置(L364/L466)、fit 接受权/交互 fit 升格/PROVISIONAL 门(L1902-1968)、Command/ViewModel 与 Qt 线程规则(L1989-2024)、原子 Apply(L2014)。
- **C6** 只察觉增长的棘轮(`<=`)在忘记压低的那一刻就不再是棘轮——**计数棘轮一律用等值**,变化必须显式改常量。

## B. 包结构与依赖

- **C10** 六包职责(文档 L299-357):`zlc_data`=领域中立、无头、可序列化数据语义 + 值上纯算法(L303,只准 import `zlc_storage.canonical`);`zlc_storage`=canonical 原语 + bytes/blob/manifest,**不定义** ArtifactRef/领域 schema/artifact kind(L315);`zlc_pulse`=脉冲域;`zlc_neutral_atom`=中性原子域;`zlc_frontend`=渲染与展示(→storage 合法,L327);`zlc_workbench`=**桌面应用与唯一 Qt composition root**(L299),DAG 有意不设 import 约束。
- **C11** 依赖方向由 `tests/test_architecture_import_dag.py` 的 `FORBIDDEN` 表机械强制;改表=修宪。
- **C12** 放置公理:`zlc_frontend` 内**只有 `qt_widgets` 可 import PyQt5**,且 `qt_widgets` **不可 import matplotlib**(源:台账 S5-shell(a),守卫:DAG 测试 + `test_zlc_frontend_qt_widgets`)。外部不得深 import `zlc_frontend.qt_widgets.<子模块>`,一律走包门面。
- **C13** 顶层 import 纯净(L506):`zlc_frontend`、`zlc_frontend.figure`、`zlc_workbench` 及各应用包根的顶层 import 不得加载 matplotlib backend、PyQt/qframelesswindow、repository backend 或真实硬件 adapter;调用者显式进入 `zlc_frontend.matplotlib_render` / `qt_widgets` / notebook leaf。
- **C14 渲染终态**:Qt 侧只见像素(`RasterBuffer`/QImage)+ `ViewportTransform` 做命中换算;matplotlib 只活在无头渲染叶(§12.5,L2035-2080:GUI 不读 worker 的 Figure/artist,静态 axes/colorbar 由 worker raster 缓存,动态 overlay 由 Qt 画,export 从 document 重画)。**任何文件不得同时 import PyQt5 与 matplotlib**;唯一过渡豁免 = `zlc_workbench/*/plot_bridge*` 与旧树,§12.5 完成后清空。

## C. GUI 结构(2026-07-20 定案,经用户批准)

- **C20 巨石死刑**:GUI 按三层落位——纯 Qt 控件一件一档进 `zlc_frontend/qt_widgets`;应用接线进 `zlc_workbench/<app>/app.py`(目标 <1k 行);持 mpl 对象的控件暂进 `zlc_workbench/<app>/plot_bridge`。`Zou_lab_control/frontend/task_console.py` 与 `pulse_gui.py` 的终态是**删除**,不是搬家。
- **C21 文件行数棘轮**(等值,守卫机械强制):qt_widgets 新文件 ≤600 行;存量超限件(board.py/fluent.py/param_widgets.py)记录现值**只准降**。放置事实不只看 import——**持有**活 Qt/mpl 对象的记录,其家在对象所在层(血训:`_GridFocus`/`_StopAttempt` 按 import 判是 render-free,实持 canvas/线程)。
- **C22 行为权威 = main**:UX 逐项继承 `ZLC_main`(独立 clone)的真实窗口行为,验收 = **两棵活树 A/B 真窗口对比**并记录 main 当日 HEAD;偏离只能进 UX 偏离台账待用户批准(L129/§2.2),默认动作是恢复 main 行为。**取到 A 树的方式(C44 修宪后)= 从 `ZLC_main` 目录起进程**(cwd 优先),不是 import——editable 现在指本仓库,任何目录 import 拿到的都是 B 树。
- **C23** §12.5 worker-raster 是**迁移完成后的独立质量项**,不是迁移前置;其前置 = 先建交互验收矩阵(逐 plot kind × 逐手势的行为表,因交互栈曾零活动覆盖);完成后 plot_bridge 控件提纯毕业进 qt_widgets。
- **C24 后端清剿指标**:迁移进度 = 删除台账上"不能删的旧树文件数"单调降,每个不能删的文件点名最后 consumer 与目标包;不用"壳掉了多少行"作指标。
- **C25 删除边界**:两壳在解体完成前不得整文件删除(仍是行为骨架宿主);旧树后端文件在其最后 consumer 断开的同一批次删除;转发壳计数(Z3)只在搬迁产生新壳时上升,Z0 归零。

## D. 数据与领域

- **C30** 虚拟与真机走**同一代码路径**,只 fake 最底层(相机帧/硬件会话);分析层绝不 import 后端或读仿真真值(守卫:`test_virtual_equals_real_contract`)。
- **C31** 持久化判别串/schema 字面量**绝不随模块路径改**——已写进用户存档,改了是静默数据损坏。
- **C32** 读出数学只在单源(`test_readout_math_single_source`);magic number 单源,测试派生不重打字面量。
- **C33** 不变量守卫在基类单源;生命周期哨兵绝不 `or` 兜底;表单显示值必须等于持久化值。

## E. 过程纪律

- **C40** **绝不 push**;commit 是常态义务(每主题完成即分阶段 commit,不等批准);**逐文件 git add,禁 `git add -A`**;subagent 只用只读工具;不跑 Vivado build/program(xsim/xelatex 可)。
- **C41 测试体系**:套件 == 清单(`migration_active_tests.txt`,Z8=0 等值守卫)——不存在"冻结测试",不要的测试**删除**(git 留底)。脚手架测试必须声明 `DIES_WITH = "<守护的旧物路径>"`,其守护物删除的同一 commit 删它(守卫扫描:守护物已消失而测试还在 = 红)。新增测试只有两类:宪法强制 / 新结构行为契约。GUI smoke flaky——非 GUI 改动不跑。commit 前只跑改动边界 + 棘轮;收口跑全套(A 组单进程,B 组=import `Zou_lab_control.notebook` 的文件各自进程,进程级不变量所迫)。
- **C42 操作安全**:回退变异用事先 `cp` 的副本,**绝不 `git checkout -- <file>`**;大段删改用内容锚点 + `assert start < end`,绝不靠行号,动手前全量校验锚点;golden 抓完必须检查**有没有鉴别力**,脚本内随机量(tmpdir 名)要归一;纯搬迁靠 A/B 窗口等同验收,只有行为件才抓 golden。
- **C43** 可机械强制的准则**必须写成测试**,散文与 regex hook 只是提醒不是强制。
- **C44 环境单源**(2026-07-21 修宪,用户批准):pip editable 指向**本仓库**——迁移树就是这个包的家,从任何目录 import 拿到的是迁移架构(含 `zlc_workbench`/`zlc_frontend`/`zlc_data`/`zlc_neutral_atom`/`zlc_storage` 五个兄弟包;ZLC_main 的 pyproject 只声明 `Zou_lab_control` 一个,装它拿不到这些)。**ZLC_main 仍是活的行为权威树**,但拿到它的方式是**从 ZLC_main 目录起进程**(cwd 优先于 editable finder),不再靠 import 解析。**修宪起因**:editable 指 main 时,notebook 的 `exp.task_console()` 开的是迁移前的旧窗口(main@6c337d4),而 `task_console.bat`(先 `cd` 到本仓库根)开的是新窗口——同一句 API 在两个入口给出两个窗口,操作者没有任何提示能分辨。改 editable 指向 = 修宪。
- **C45** 全中文回复(代码/commit/token 除外);"完成"的唯一定义是用户真实验收路径可复现,测试绿只是代理指标;猜错 ≥2 次先确认再动手。
- **C46 GUI证据双轨**(2026-07-22 用户批准并校正):所有GUI的验收与debug共享同一QApplication owner、正式composition root、窗口size/style与真实input序列。快轨先设`QT_QPA_PLATFORM=offscreen`，再由唯一`ensure_qt_app()`建立应用，随后走正式open/launch→真实Qt input→outer-window `grab()`；不得另建`QApplication`、强设DPI/尺寸/样式或弹出桌面窗口。Windows offscreen font database若不含产品声明字体，只能由该application owner注册同一系统字体文件，且快轨必须以真实glyph像素证明文字已栅格化，空字图不能作为视觉证据。慢轨从正式`.py/.bat` launcher真正打开桌面GUI，再用桌面鼠标/键盘和屏幕截图完整复现人类流程，用于最终或争议复核。TaskConsole、PulseGUI、DeviceManager、FigureViewer及未来GUI无例外。
- **C47 UI变化驱动、局部提交**(2026-07-22 用户纠正):完整不可变application snapshot只用于首次composition，以及worker完成、连接切换、Run/cancel/close等真实ownership或线程边界上已经产生的**新结果 / 新硬件状态**；禁止固定周期无条件构造它，禁止把`pump()`、`snapshot()`或全窗口投影当事件循环，禁止在Qt线程同步轮询remote/device。保留的snapshot必须冻结真实ownership/consistency boundary（immutable dataset revision、Run/ack/capability跨线程观察或同一次board coherent front）；普通Qt编辑事件、周期性全应用投影不是snapshot边界。文本、code、spin/combo临时值和selector drag由稳定widget/editor session持有本地draft；普通输入不得触发全局snapshot、preview/render或重建。`editingFinished`、Apply等语义提交只返回当前editor revision/document引用的typed local delta，UI只投影实际改变的字段及其明确derived dependents；Scan tab component front也只在scan schema/candidate/source事实变化时更新，不得再拼成伪全局snapshot。Add只建新增控件，Remove只销毁对应控件，Reorder只移动现有控件；scalar/unit/name/delay/visibility变化全部原位更新。Open/Load、Target topology或document generation替换可以做一次完整editor reconcile，但也不得销毁未受影响的稳定控件。必须以机械ratchet禁止`QTimer -> controller.pump/snapshot -> whole-window apply`和`textChanged/textEdited/valueChanged -> owner wake -> whole-window apply`回流，并以真实Qt输入证明局部控件identity与交互延迟；C47整改账本中未关闭的实现不得被复用、复制或推广到其它GUI。
- **C48 当前硬件事实优先**(2026-07-22 用户纠正):软件历史不能证明当前硬件状态。`ResourceArbiter`只做当前进程内互斥；每个领域session的`close_session`是正常cleanup唯一SAFE/stop owner，每次关闭只执行一次并返回该领域的当前终态证据。cancel可走out-of-band interrupt，框架异常可走emergency interrupt；二者都不是正常close的第二套SAFE。cleanup失败只令本次run/session失败并保留诊断，不写持久设备隔离、不制造进程永久拒绝或“必须重启”状态。新连接必须以实时握手和当前硬件SAFE初始化建立authority；当下验证失败就拒绝该次连接。数据`CommitJournal`/CAS的crash consistency保留，与设备安全解耦。
