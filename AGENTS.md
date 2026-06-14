# AGENTS.md — 在本仓库怎么干活(给 agent 和维护者)

> **第一次来**:先读仓库根 `README.md`(环境安装 `install_requirements.bat`、启动器、目录树、`references/` 说明、四本手册),再读本页。
> 本仓库**没有 `CLAUDE.md`**;工作守则以本 `AGENTS.md` 为唯一权威。
> 这是**顶层工作守则**,任何改动前先读这页。
> 子系统细则:`Zou_lab_control/frontend/AGENTS.md`(前端密封 API 契约)、`docs/MAINTAINER_NOTES.md`(FPGA/host §编号深档)。
> 当前焦点与待决方向:`docs/ROADMAP.md`。测试怎么跑:`tests/README.md`。文档/记忆各管什么:见本页最后"文档地图"。

## 0. 项目一句话
中性原子实验控制系统:**notebook-first** + PyQt/Fluent GUI + 远程 FPGA 脉冲流送器。
分层:`neutral_atom/`(session/devices/timing/core/operations/subsystems 实验框架)+ `frontend/`(绘图/GUI/PDF)+ `fpga/`(RTL/Vivado/host)。当前阶段焦点见 `docs/ROADMAP.md`。

---

## 1. 交互与流程规则

- **全中文回复**(代码、commit message、标识符 token 除外)。用户看不懂英文。
- **本地 commit,绝不 push**(用户没明确要求就不 push)。commit message 用**英文**、祈使句、首行简洁概括 + 正文讲清"为什么";只在用户要求时才 commit/或按节点自觉提交。
- **绝不跑 Vivado build / program / 综合 / 上板**。`xsim` / `xvlog` / `xelatex` 可以跑。需要真机动作时,产出脚本/命令让用户在 FPGA 机上跑。
- **无后向兼容、无历史残留**:做最干净的设计,旧机制**彻底删除**而不是留双轨;旧结论在 memory/文档里标 `superseded`;删完 `git grep` 死标识符必须 = 0。
- **真机问题的根因在我的代码里**:模型/编译/RTL 全绿但真机错时,先查**pack/上传的寄存器残留、跨连续程序的状态、新固件配旧主机**这类,别甩锅 build;让用户在**真实文件**上跑诊断,别猜着改正确的代码。
- 不确定就别装懂:UI/视觉效果不确定时**必须截图看**(见 §3 可视化验收),别瞎改瞎确认。
- **能并行就并行,优先省用户的时间**:凡是能并行加速的工作——多文件搜索/审查、彼此独立的子任务、对抗式验证、多方案对比、跨子系统盘点——就**开多 agent / workflow 并行跑**,不要串行慢慢做。**不必顾虑 token 消耗**;首要目标是缩短用户的等待(wall-clock)。默认就用这种"铺开并行"的心态做事;只有真正琐碎或纯对话的一两步才单线程。
  - **本项目用户已把"能并行就并行"定为核心铁律 = 永久 opt-in。** 所以遇到 substantive / major 任务(理解一个子系统、跨多文件审查/盘点、设计+实现一个大改、对抗式验证),**默认动作就是 author 一个 `Workflow` 多 agent 跑**(理解阶段并行 readers → 实现阶段 pipeline → 审查阶段对抗 verify),而不是自己串行手做。ultracode 开着时尤其如此。
  - **写进文档却继续串行 = 没做到**(2026-06 反复栽:把这条记进 .md 却一直单线程,用户追问"为什么完全没看到你开 workflow")。**记 ≠ 做**;该开 workflow 就开,别只记不用。
- **对照用户的验收标准验证,不是对照自己的代理指标**(2026-06 反复栽在这):测试绿 / 像素一致 / 对象已挂上 ≠ 完成。每件事先问"**用户真正会怎么用它,我验证的是不是那个**"。具体落地:
  - **"虚拟==实机"要把真机的逐步操作流程写出来**——包括**数据存在哪个文件夹、用户怎么指过去、结果写到哪**——让虚拟跑**同一串**、只换最底层数据源(假帧写进同一个文件夹)。**绝不用内存便利(`characterize(groups=150,...)` 这种)替掉真机的文件/文件夹流程**;真机怎么读盘,虚拟就怎么读盘。
  - **交互/视觉类**:必须**模拟事件证明真的有反应**(scroll→xlim 变了、drag→线移动了),不是只确认"selector 对象挂上了"。
  - **用户指了参考实现(如 `references/rb87_readout_v16`)就通读当权威 spec**,按它的流程形态实现;**不许自作主张只挑算法移植、跳过文件 IO/编排**(正是这个自作主张害我返工)。
  - **已经猜错 ≥2 次、或要做"和 X 一模一样"的大改**:动手前先用 AskUserQuestion 确认验收场景(目标流程形态 / 文件格式 / "selector"具体指什么),**别再盲猜第三次**——确认一次比错三次省用户时间。
  - 收尾**不说"做完了"**;说"对照标准 Y 验证了 X;Z 还没对你的真实用法验过"。详见 [[verify-against-user-acceptance]]。

---

## 2. 设计原则

- **解耦:子模块只通过接口互联**(最重要的总纲,关系到扩展性/可维护性)。
  - 上层依赖**抽象契约**而非具体类:`devices/base.py` 的 `CameraDevice`/`SequencerDevice`/`TrapArrayDevice` 让 readout/timing 不绑死具体硬件(虚拟/真机/远程三后端共享同一 session)。
  - 跨模块调用走**文档化的接口**,不伸手进别的模块内部:Task 控制台三层只经 `SignalHub` 耦合(采集/feed/GUI 互不直接引用);frontend 对外只暴露密封接口(见下条)。
  - 新代码沿用这条:加功能先想"它通过哪个接口接入",而不是直接 import 别人的内部实现。
- **复用优先:同一件事只实现一次,到处复用**(解耦的另一面,直接关系到正确性——重复实现必然漂移出 bug)。
  - 加任何功能,先找"已有哪个原语/层能复用",别重造:frontend 可复用层(`BaseLivePlot`/selectors/`data_figure`/`style.py` 设计 token——见 `frontend/AGENTS.md` 规则7)、`core/` 纯算法 + `operations/` 标准array函数(box/otsu/psf/bimodal/fidelity 各只一份)、`SignalHub`、单一配置源。
  - 同一算法/常量/样式出现第二处 = 提成共享函数/令牌,不复制粘贴。提取后用等价证明(像素 diff / 数值一致)确认无行为漂移。
  - 共享件要可复用就得**解耦**:它只依赖抽象契约、不反向依赖调用方;这样"加新 plot / 新设备 / 新面板"都是复用既有层,而不是另起炉灶。
- **虚拟(假数据)测试必须走与实机完全相同的代码路径——只有最底层数据源是假的。** 这是"为真机做准备"的全部意义:换真机时只改 `na.connect("virtual", ...)` → `na.connect("qcmos", ...)`,其余每一格 notebook、每一行分析代码不动就能跑对。
  - 分析/编排层(`core/`、`operations/`、`subsystems/`)**只依赖设备抽象契约**(`devices/base.py`),**绝不 import 具体后端**(`devices/virtual.py`、`qcmos.py`),**绝不读仿真真值**(`occupancy`、已知站点中心 `_site_centers`、`render_image` 等):站点中心从图像**检测**(`find_site_centers`),PSF 从数据**拟合**(`fit_site_psfs`),阈值从数据分布**学习**(bimodal/otsu),保真度从 reference 帧**严格共识**推——全是真机要跑的同一函数。
  - 虚拟后端只 fake **数据源**(相机帧),并实现与真机**相同的 `CameraDevice` 契约**,是 drop-in 替换;假只能假在"帧怎么来",不能假在"帧怎么被分析"。
  - tutorial 必须把这条流程**显式**走出来(让读者看到"从图像提取站点 / 从数据拟合 PSF / 从数据分布定阈值 / 从 reference 推保真度"),而不是黑盒一行带过;并写明"唯一虚拟之处是相机数据,换真机只改 connect"。
  - 由 `tests/test_virtual_equals_real_contract.py` 强制(分析层一旦 import 具体后端或读仿真真值即挂测)。配套总纲见下条「能机械强制的准则必须写成测试」。
- **单一真相源**:同一事实只有一个权威定义处。例:板级/容量配置只在 `fpga/board_config/streamer_config.json`;前端排版只有一套 300dpi 体系;memory 对原则只放指针,权威定义在仓库 AGENTS。
- **能机械强制的设计准则,必须写成测试,不能只留在文档里**(架构契约测试 / fitness function)。只写在 `.md` 里的准则会被(包括我自己,尤其长会话里局部模式匹配时)悄悄违背且无人报错——这正是整理这些 `.md` 却仍被违背的根因。所以:每立一条"所有 X 都必须 Y"的结构性准则(例:**所有 plot 都必须继承 `BaseLivePlot`** 才能复用 selectors/data_figure),就同时写一个会在违背时**失败**的测试(例:`tests/test_frontend_plot_contract.py`),并在准则旁注明强制它的测试。文档讲"为什么",测试保证"不退化"。
- **借鉴参考实现:取原则,不照搬具体设计**。`references/` 里的实现(confocal GUI、rb87 readout 等)是灵感来源,要遵守的是上面这些**设计原则**,不是它们的具体形态/代码结构;有更干净的思路就用自己的,别为"照着参考写"而牺牲解耦或引入残留。
- **前端密封 API**:几何/dpi/字号/配色/阴影/缩放**由 frontend 拥有**,外部只传数据。完整六规则见 `frontend/AGENTS.md`——加任何前端公共参数前先读它,并把参数分类为 DATA(允许)还是 ART/几何(禁止)。
- **改寄存器映射必带版本握手**:改 host↔RTL 的寄存器布局必须 bump 两边的 LAYOUT_ID 并在 prepare 时校验,不匹配明确报"重建+重启"(否则新主机配旧 bitstream 会踩进死字)。

---

## 3. 测试与验证原则

> 这里只写**原则**;具体命令、targeted matrix、截图函数签名都在 `tests/README.md`(单一真相源,别在这里复制命令)。

- **只跑能证明"改动边界"的测试,别为"求安心"跑 full pytest**(铁律,反复栽过)。改了哪层就跑那层对应的几个测试文件:改 `neutral_atom`/分析/某个 plot → 跑 `test_neutral_atom_lightweight` + 相关契约(`test_virtual_equals_real_contract` / `test_readout_math_single_source` / `test_frontend_plot_contract`)+ 该 plot 的 smoke;**不碰 pulse GUI / task_console 的测试**。`full pytest -q` 只在**用户说大改 / 跨多子系统 / 交付扫尾**时跑(大不大由用户定,不要自作主张全量)。
  - **`test_frontend_smoke` 里的 GUI 测试(`demo_editor`/`demo_console`)跑的是 demo fixture(`demo_state()`、且不套真窗口 `FluentWindow`),不是用户真正打开的 `show_pulse_gui`/`show_task_console`;满载下还偶发 flaky**。所以:**非 GUI 改动绝不跑它们**——否则它们的 flaky 会反复制造与你改动无关的 "X failed",反复浪费双方注意力(这正是 neutral_atom 改动里反复栽的坑)。
- **性能优化必须 logic/appearance-neutral**:只能让同样的输出更快(如解析 Jacobian、skip-if-unchanged 守卫、缓存不变量),**不能改刷新节奏/外观**(如降低拟合频率就是改外观,不做)。改完要能证明等价(如 popt 数值一致)。
- **Python 侧契约测试**:仓库**没有 iverilog/cocotb**。RTL 行为用 Python 忠实镜像 + `xsim`(真 IP 网表,最强证据)验证;verilog 端口宽度由 Python 契约测试核对。
- **所有可视化改动都要"看到用户所见"再算通过**:三档 QT_SCALE_FACTOR(1.0/1.25/1.5)整窗截图 + 1:1 像素裁剪。**DPR=1 离屏通过不算通过**,否则就是"瞎改"浪费时间。**截图/验收必须对着用户真正打开的入口**(`show_pulse_gui`/`show_task_console` 的真窗口),**不是测试用的 demo fixture**——测到的必须是你真正看到的那个东西。静态 notebook 图(matplotlib)则**渲染实际输出并肉眼检查**。
- **布局美术铁律:不重叠、不裁切、对齐**(复合/多面板图同样适用,注记不许压在数据上)。多面板用 `frontend.canvas.create_axes_grid`(固定格 + 显式间距 + 图尺寸自适应)使三者天然成立,通用 N 不写死站点数;细则见 `frontend/AGENTS.md`「Layout」节。
- **删除/重构后**:`git grep` 死标识符 = 0;`python -m compileall` 干净;无遗留 TODO/FIXME。

---

## 4. 文档与教程原则

- **文档只在大版本/里程碑更新**,别每个小修改都更新文档——非常耗时。攒到一个有意义的版本节点再统一更新对应文档/手册。
- **文档要达到"陌生人看完能上手"的程度**;改或加文档时,跑一个**假装完全不懂这个项目的 agent 做对抗式独立审查**——它能否仅凭文档理解设计/原理/用法?不能就补。这种审查很有用,需要时多用,**不必担心 token**。
- **教程(notebook)承接文档**:教基本用法,内容与文档一致(文档讲清楚,教程带着走一遍)。改了接口要同步教程。
- 手册用**中立教学语气**;agent/审查/维护注记进 AGENTS 或 MAINTAINER_NOTES,**不进用户手册**。
- PDF 一律走 `frontend.render_tex_pdf` / `render_notes_pdf`:临时目录编译、**只产 .pdf**(失败留 `.build.log`);`docs/**/*.tex`、`*.sty` 是中间产物,已 gitignore,不入库。

---

## 5. 常犯错误目录(踩过的坑,改前对照)

> 这些是反复踩到的自造 bug。**命中规则见本节末"如何持续记录"——发现新坑立刻追加。**

1. **无作用域 `QFrame { border }` 级联**:Qt 里 QLabel/QComboBox/QSpinBox 内部都是 QFrame,容器 styleSheet 里写裸 `border` 会级联到**所有子控件**,边缘冒杂线。→ 容器描边用 `#objectName` 作用域或 `paintEvent`,绝不拼无作用域 border。(踩过两次:period 卡 / Setting 弹窗)
2. **缩放分叉**:`set_fluent_scale(None)` 曾静默回退 1.0,而别的 GUI 按屏幕 fit → 两窗控件大小不一致。→ 唯一规则 `resolve_fluent_auto_scale`,`show_*` 默认 `scale=None`。
3. **高 DPI figure 变形**:matplotlib stock backend 会从控件尺寸**反推 figure inches**,毁掉 spec 拥有的 fixed-inch 几何。→ `EmbeddedFigureCanvas` 三不变量,resizeEvent 不碰 figure。
4. **demo helper 钉 scale=1 = 截图盲区**:`demo_editor`/`demo_console` 写死 scale=1.0,绕过真实入口,掩盖了缩放分叉。→ 用 `capture_user_view` 的真实路径 / `parity` 目标验。
5. **PyQt slot 里 NameError → 进程无声 abort**(假"挂死",无报错)。→ slot 体里小心 sed 重命名留下的悬空变量;必要时 try/except 汇到状态行。
6. **texbody 经 Python 写盘的转义陷阱**:`\f`/`\t` 会变控制符(FF 还会把行裂开)→ 用 raw 字符串或 `chr(92)` 构造反斜杠;**章节标题里裸 `_ & #` 会让 TOC moving-arg 报 Missing $** → 转义或避免。
7. **agent 生成 LaTeX 的泄漏**:会混入"I'll write…"闲聊 / markdown `---` / 裸 Windows 路径(路径里 `\U \Z` 当未定义控制序列报错)→ 写盘前 declutter;codeblock 内反斜杠安全,普通正文不安全。
8. **真机错而模型全绿**:别改正确的代码,查 pack/上传残留、跨程序寄存器状态、固件/主机版本错配;在真实文件上诊断。(见 §1)
9. **编辑器/联动改 config 别走会 teardown 的路径**:如 task console 编辑器改 relim 不能调 `card._set_param`(它 `_reset_plot` 把 plotter 置 None,随后读到 None 崩)→ 直接写 `config.params` + 重建本地视图。
10. **markdown 里别用 LaTeX 排版宏**:`.md`(AGENTS/MEMORY/ROADMAP/README)用 markdown 语法(`**粗体**`、`` `代码` ``);`\tfocus{}`/`\pyapi{}`/`\filepath{}` 只属于 `.texbody`。写 .md 误用 `\tfocus{}` 会原样显示。(整理本文件时刚犯,30 处;用 chr(92) 脚本批量改回——同坑 #6。)
11. **新增 plot 写成裸 matplotlib 图,绕过 `BaseLivePlot`**:`site_histogram_grid` 曾写成 `class SiteHistogramGrid:`(裸图),于是 selectors(缩放/拖阈值线)和 `data_figure`(拟合栈)**全部用不上**——而复用这层正是 frontend 存在的意义。根因不只是写错,而是该准则只在文档里、没有测试守。→ 新 plot **必须**继承 `BaseLivePlot` 走其 `show()` 生命周期(多轴图 override `_create_axes` + 每格挂工具,`DataFigure(ax=...)` 绑定单格);现已由 `tests/test_frontend_plot_contract.py` 强制(裸 plot 类直接挂测)。配套总纲见 §2「能机械强制的准则必须写成测试」。
12. **改 scoped 的东西却跑 full pytest "求安心"**:改 neutral_atom/分析/单个 plot,却跑了整套(含 ~98 个 `demo_editor`/`demo_console` 的 GUI 测试),把满载下偶发 flaky 的 pulse GUI/task_console 测试反复拖进来,造出一堆与改动无关的 "15 failed",反复浪费双方注意力去排查。**根因**:把"想要信心"等同于"跑全部",违背 §3「只跑改动边界」。→ 只跑改动边界那几个文件;full 只在用户定的大改/交付时跑;**GUI smoke 是 demo fixture 不是真窗口,非 GUI 改动绝不跑**。见 §3。

### 如何持续记录(防上下文压缩遗忘)
- 每当我造成一个**自造的或重复的 bug**,立刻在上面加一条(一句话:现象 → 根因 → 规则)。这是标准收尾动作,和"跑测试/截图验收"同级。
- 本目录是**版本化文件**,且 `MEMORY.md` 顶部有指针指向这里——所以会话被压缩后,我重新读到 MEMORY 时仍会被指回本目录,不会忘。
- (可选,用户决定)可加一个 Stop hook,在每次收尾时提醒"是否有新坑要追加"——需要改 `settings.json`,默认不加。

---

## 6. 文档地图(每个文件管什么)

| 文件 | 受众 | 管什么 |
|---|---|---|
| `README.md` / 各子目录 `README.md` | 所有人 | 仓库布局、入口、怎么启动 |
| `docs/{main,frontend,fpga,device}_manual_zh.pdf` | 用户 | **教学**:怎么用(notebook + GUI + FPGA 流程),教材语气 |
| frontend 手册"对外接口完整参考"章 | 用户/调用方 | 公共 API **参考**:签名/参数/数组契约/示例/不可配置边界 |
| `docs/task_console_design/*.pdf` | 你/维护者 | Task 控制台**设计审查**专档(单一主题深档) |
| `docs/MAINTAINER_NOTES.md` | 维护者 | FPGA/host **子系统深档**(§编号:为什么这么写、握手、容量、性能…) |
| `AGENTS.md`(本页) | agent/维护者 | **工作守则**:流程/设计/测试/文档原则 + 常犯错误目录 + 文档地图 |
| `frontend/AGENTS.md` | agent/维护者 | 前端**密封 API 契约**六规则 + 事故史 |
| `tests/README.md` | agent/维护者 | **测试怎么跑**(targeted matrix / Vivado / 截图) |
| `docs/ROADMAP.md` | 你/维护者 | **当前焦点、需求、待决方向、暂缓项** |
| `memory/`(在 `.claude/`,不入库) | 我(跨会话) | 一行索引 + 根因记录;对原则只放指针指向上面这些权威源 |

**原则:能进仓库版本化、全队/未来都要遵守的(规则/原则/坑/接口契约)→ 进上面这些 .md;我的跨会话工作记忆(做了什么、为什么、当前状态、去哪看)→ memory。同一条不两处各存一份。**
