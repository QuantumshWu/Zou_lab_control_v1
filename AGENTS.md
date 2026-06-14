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

---

## 2. 设计原则

- **解耦:子模块只通过接口互联**(最重要的总纲,关系到扩展性/可维护性)。
  - 上层依赖**抽象契约**而非具体类:`devices/base.py` 的 `CameraDevice`/`SequencerDevice`/`TrapArrayDevice` 让 readout/timing 不绑死具体硬件(虚拟/真机/远程三后端共享同一 session)。
  - 跨模块调用走**文档化的接口**,不伸手进别的模块内部:Task 控制台三层只经 `SignalHub` 耦合(采集/feed/GUI 互不直接引用);frontend 对外只暴露密封接口(见下条)。
  - 新代码沿用这条:加功能先想"它通过哪个接口接入",而不是直接 import 别人的内部实现。
- **单一真相源**:同一事实只有一个权威定义处。例:板级/容量配置只在 `fpga/board_config/streamer_config.json`;前端排版只有一套 300dpi 体系;memory 对原则只放指针,权威定义在仓库 AGENTS。
- **前端密封 API**:几何/dpi/字号/配色/阴影/缩放**由 frontend 拥有**,外部只传数据。完整六规则见 `frontend/AGENTS.md`——加任何前端公共参数前先读它,并把参数分类为 DATA(允许)还是 ART/几何(禁止)。
- **改寄存器映射必带版本握手**:改 host↔RTL 的寄存器布局必须 bump 两边的 LAYOUT_ID 并在 prepare 时校验,不匹配明确报"重建+重启"(否则新主机配旧 bitstream 会踩进死字)。

---

## 3. 测试与验证原则

> 这里只写**原则**;具体命令、targeted matrix、截图函数签名都在 `tests/README.md`(单一真相源,别在这里复制命令)。

- **跑能证明"改动边界"的最小检查**。**小且你很确定的改动可以不跑 full pytest**,省时间;full `pytest -q` 留给大改、跨多子系统、或交付前的扫尾。
- **性能优化必须 logic/appearance-neutral**:只能让同样的输出更快(如解析 Jacobian、skip-if-unchanged 守卫、缓存不变量),**不能改刷新节奏/外观**(如降低拟合频率就是改外观,不做)。改完要能证明等价(如 popt 数值一致)。
- **Python 侧契约测试**:仓库**没有 iverilog/cocotb**。RTL 行为用 Python 忠实镜像 + `xsim`(真 IP 网表,最强证据)验证;verilog 端口宽度由 Python 契约测试核对。
- **所有可视化改动都要"看到用户所见"再算通过**:三档 QT_SCALE_FACTOR(1.0/1.25/1.5)整窗截图 + 1:1 像素裁剪。**DPR=1 离屏通过不算通过**,否则就是"瞎改"浪费时间。
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
