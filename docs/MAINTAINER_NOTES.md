# Maintainer checkpoint

本文件只记录最新交接点。规范架构见`docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md`，执行方法见`AGENTS.md`，活动问题见仓库外`../ARCHITECTURE_AUDIT_CURRENT.md`。

## 当前状态

- Branch：`codex/system-architecture-migration`。
- 重开基线：`476d125304fc90b6ef4f5009184dd2119206fcc2`；C0规范提交：`ed6dfe21c0d99999444c099d8c644a20b44e561d`。
- 当前checkpoint：§8 C4 Task takeover、Edit 与输出已 dependency-closed；下一唯一工作是 C5 PulseGUI narrow hold/step。不得返回旧 M1–M7 叙事，也不得重做已经闭合的 C1/C2/C3/C4。
- 预期worktree例外：未跟踪用户文件`pulses/scan_test.json`。永远不得读取、修改、移动、删除、stage或commit。
- RTL、Tcl、XDC、bitstream和wire protocol仍冻结；C1/C2没有硬件改动。

## C1 已闭合的 owner

- 本仓`zlc_plot`现在是PlotSpec、投影、六种plot、selector、Fit/overlay、Matplotlib artists/blit、raster worker、Qt front、Divider/固定data box、size/DPR、style和export的唯一owner。外部`zlc_data`、`qt_controls.py`、notebook/PNG/build/cache均未纳入；唯一Helvetica asset归`zlc_plot/assets`。
- `zlc_data`只保留权威`(R,P,*data_shape)`、PointTable/GridTopology、dtype、validity与通用authority selection；旧Fit/layout/display闭包全部删除。Camera monitor每次publication为最新`(1,1,*frame)`，不再用`MONITOR_HISTORY`、`history_cycles`或P轴保存显示历史。
- TaskConsole、Calibration、Occupancy、DataFigure、FigureViewer、Edit和Pulse preview全部消费同一PlotSession/Qt5PlotWidget surface。旧frontend plot/projection/selector/Fit/render/raster/style/layout、neutral Fit artifact、Workbench/leaf专用composer与旧测试均在同cut删除；没有compatibility re-export或presentation sidecar。
- SignalPlane的selector/Fit派生值携ordered exact parents。Processor route退役与execution退役已分开：in-flight prepare/evaluate完成后由lane唯一回执terminal，不会因提前detach卡住TaskConsole/Experiment关闭。
- Camera物理binding明确区分Camera event stream与另行materialize的Dataset generation；Occupancy不再错误比较两种generation。Calibration推荐视图只使用当前`PlotKind.IMAGE`词汇，未保留`2d/sites`旧kind。
- `RasterFront.source_revisions`是实际绘入front的事实；Workbench publication关联严格有界于pending、latest-worker和当前presented/rolling window，并在失败、supersede、过期成功、替换和retire释放。Task preview从leaf declaration到PanelConfig全程保持`PlotKind | PlotSpec`，只在layout文件I/O编码。
- Calibration不再有report Projection/ResultBundle/workbench jobs或通用Logic-node report owner。领域层只物化七个普通FINAL Dataset outputs；一个lazy leaf UI薄adapter把同一outputs映射到三个共享`zlc_plot`页面，Qt和headless export共用该映射。

## C1 证据与规模

- 正式Qt快轨通过：Camera→live Image→Area→第二Image→两处live Fit及参数signal；Occupancy启动不auto-open、手动绑定并连续更新；Calibration产出；FigureViewer；Pulse preview真实输入/resize/selector/export。
- Distribution双高斯/threshold、FacetGrid all-facets Fit、MOT 7×7×7共343 cell的live→FINAL以及headless public/import边界均通过。
- C1 current-contract集合得到206个批量PASS；当次仅有的两项旧测试合同已改写并分别PASS。另有显式Stop和上游Camera退役两条受控in-flight Processor测试，Occupancy完整正式流程再次PASS。
- 最终三项残余收口集合83项全部PASS；正式TaskConsole Calibration流程再次PASS，并证明artifact FINAL后七个typed outputs完成、`report/{site_map,fidelity,distribution}.png`均非空、无post-FINAL warning、默认不保存raw frames且无旧manifest。便捷`sitemap()`不是Task，不被错误赋予Task的`save_frames/post-FINAL warning`生命周期。
- 2304×2304 uint16规则图Fit packing实测约0.58 s、约4.18 MiB诊断峰值；保持原dtype readonly view且不展开H×W coordinate mesh。
- 固定口径生产包：331 modules / 139,278 physical LOC / 933 classes / 558 dataclasses / 38 enums。相对C0为-50 modules / -22,288 LOC；其中本仓`zlc_plot`为33 modules / 27,983 LOC，扣除它后旧包净删83 modules / 50,271 LOC。tests为112 modules / 36,173 LOC，相对C0净删19 modules / 13,649 LOC。
- `git diff --check`无内容错误；生产残余扫描没有发现旧plot owner/import、history-in-P、ordinary presentation sidecar、Calibration第二report owner或包外Matplotlib实现。headless API只加载`zlc_plot`纯值模块，不加载session/render/Matplotlib/Qt。source-empty旧目录在Git tree中不存在，本机ignored cache/output不属于提交。

## C2 已闭合的 owner

- `InstallationConfigDocument`现在只保存ordered heterogeneous `DeviceInstanceConfig(instance_id, role, type_id, parameters)`；role只作人类投影，依赖只保存stable instance id。普通JSON没有digest/CAS/version counter；保存复用`zlc_storage.atomic_write_text`。
- fixed-namespace discovery得到7个leaf device type和3个graph template。descriptor唯一声明schema/defaults/factory/capabilities/stable-id requirements/optional discovery；纯preflight在任何设备副作用前完成schema、missing/wrong-capability、duplicate与cycle检查。runtime只保留`require_capability(DeviceRef, token)`和反向cleanup，不再按具体Port类型分栏。
- DeviceManager按domain显示稳定per-instance cards；Add/Remove/Retype只改目标结构，普通字段原位更新，New/Load按stable id reconcile。requirement显示为当前graph内兼容instance的typed下拉；discovered hardware与loaded session分开。Save/Load是普通JSON，Apply是application topology replacement。
- active Run期间Apply不触碰旧runtime；idle成功或失败恢复均保持同一public `Experiment`身份。候选连接失败恢复previous graph；恢复也失败进入明确no-active-session。除DeviceManager外的旧runtime Workbench在replacement前退役，manager继续绑定同一public对象。
- Camera/Pulse/RF消费者都经generic capability解析。MOT camera资格由Pylon/virtual-MOT leaf的`camera.mot_field_capture`能力表达，不再靠`mot_camera` role充当身份；virtual camera association由观察sequencer FIRE和frame ordinal的`VirtualCamera`拥有。
- 已删除installation package/plan/dispatch/assets、hardware/sequencer/simulation backend-wide config/package/installation、asset-map revision、installation digest/CAS与storage file lock；旧tests同cut改写或删除，没有compatibility入口。

## C2 证据与规模

- DeviceManager正式offscreen流程通过Add/Remove/Retype/New/Load/Save/Apply、typed requirement、discovered/loaded分离、同一Experiment Apply与owner close；standalone TaskConsole从DeviceManager Apply后打开TaskConsole和PulseGUI。
- active continuous Run拒绝Apply且runtime generation不变；候选连接失败恢复previous graph；双连接失败明确no-active-session。真实qCMOS/Pylon fake-E0五项保留active qualification、stamp gap、association/ordinal/undrained与Mono8 dtype证据。
- current modified合同集合195项通过；architecture/import集合41项通过；Camera/Occupancy/MOT正式纵切3项通过。全仓collect得到885项且无旧模块import collection错误。
- 固定口径生产包：324 modules / 138,722 physical LOC / 929 classes / 555 dataclasses / 38 enums；相对C1净删7 modules / 556 LOC / 4 classes / 3 dataclasses。tests为111 modules / 36,405 LOC。C2没有硬件文件diff。

## C3 已闭合的 owner

- 所有当前 Logic Node leaf 通过 `zlc_neutral_atom.logic_node.LogicNodeDescriptor` 发现；通用 `LogicNodeHost`/`LogicNodeExecutionContext` 统一 request freeze、device capability binding、Run 生命周期、Processor cursor 与最终输出发布。删除旧的 per-leaf API/Package/Prepared/Bound/HostedProcessor/LiveDatasetHost/ArtifactDispatch/NodeInput 闭包；没有兼容 re-export。
- `Experiment.nodes.<api>` 对所有 leaf 只投影同一个轻量 `NodeApi`（`build/start/run/stop/open_ui` 与 descriptor operations）；TaskConsole 只消费 descriptor、通用 host factory 和 SignalPlane，不再列举具体 calibration/occupancy/camera/pulse 分支。
- Calibration 的嵌套 capture/analysis 继续由同一个 host 生命周期承载；Camera/Occupancy/MOT/PulseScan/Duration/Release-recapture 均保留各自领域算法和必要 device Port seam，不复制 host。Occupancy 是纯 Processor，Calibration 的 raw-frame export 与 report 只写项目 `runs/` 记录。
- 终端 live Dataset 的 FINAL front 由 `SignalDataPlane.detach_live()` 保留，失败/取消仍 retire；Calibration source facts 通过 capture artifact 的 typed adapter properties 读取，不再引入 per-event binding digest。
- zlc_plot pointer/front 修正把 data/selector revision 与 pointer surface validity 分离，持续拖动不再因新帧取消已按下的 pointer；Fluent signal-tree reconcile 从当前 Qt model 恢复 expansion，删除 expanded producer 不再解引用已移除 `QStandardItem`。

## C3 证据与规模

- 定向产品流：TaskConsole Camera live→Area→第二 Image→Fit 与 Occupancy `2 passed`；MOT-field final card/shape/output `2 passed`；Signal picker topology delta `9 passed`；current Logic host/descriptor/calibration/capture/runtime contracts `97 passed`；另一 device/workbench group `73 passed`。
- 过期合同已按当前 API 改写或删除：duration 测试改为 `exp.nodes.readout_duration_fidelity.build(...)`，pulse failure test 使用通用 `require_capability("pulse.execute")`，旧 tutorial 完整重写为 current `Experiment.nodes` 并由 nbclient 执行。`tests/test_pulse_scan_signal_consumer.py` 的旧 API 闭包已删除，不恢复兼容。
- 全套回归：`823 passed, 1 skipped`（含 1 个 zmq Proactor 环境 warning），总耗时约 144 秒；没有生产失败或旧 API 兼容失败。
- C3 checkpoint 已由 `7cf8b64` 提交；保护性用户文件 `pulses/scan_test.json` 未读取、未修改、未 stage、未 commit。

## C4 已闭合的 owner

- TaskConsole 现在只有一个 active Task takeover 投影：固定 status/header surface 显示 task/stage，唯一 Stop task action 走原有 `_stop_logic_node` 生命周期；所有 graph、node、layout、Edit/producer、panel-setting mutation 在 takeover 期间由同一 gate 置灰/拒绝，selector/zoom 等 view-only 操作仍可用。terminal、failure、Stop 都通过同一 host cleanup 出口清除 takeover。
- Logic Edit 与 Plot Edit 使用同一 `LogicNodeConfig.authored/inputs` draft；WeakSet 只登记 Qt projections，不存第二份参数 truth。Panel Editor 的 snapshot/config controls 在 takeover 期间也由同一 gate 禁用。
- selector 的 1-D interval 可预填 leaf 声明的 `axis_range`；Camera leaf 额外声明通用 `SelectionParameterPatch`，Area 只生成 stable camera instance + `roi_*` 草案。composition root 将它转交已打开的 Device Manager draft；Device Manager 原位校验/重绘卡片，只有用户点击 Apply 才改变 active installation，TaskConsole 不含 Camera/ROI 特判。
- TaskConsole/DeviceManager 输出路径保持 project-root `tasks/`、`runs/`、`figures/`；Calibration 的 `save_frames` 与 concise reloadable report 已有正式纵切。没有加入 SHA、bytes、size manifest、软件预算或第二 presentation owner。

## C4 证据与规模

- 定向产品流：Camera live→Area→第二 Image→两处 Fit、Occupancy 手动绑定、MOT live→FINAL、DeviceManager graph flow、TaskConsole start gate 全部通过；新增 Camera Area patch contract 与 headless discovery 也通过。
- 当前全套回归：`824 passed, 1 skipped`，总耗时约 146 秒；headless Logic discovery 未加载 Matplotlib/Qt。`git diff --check`通过，RTL/Tcl/XDC/bitstream/wire protocol无diff。
- C4 已由 `2e83365` 提交，包含 16 个生产/测试/设计文件（806 insertions, 28 deletions）；受保护用户文件 `pulses/scan_test.json` 仍未读取、未修改、未 stage、未 commit。

## C5 已闭合的 owner

- Pulse hold/step 已改成当前 Run 内的 typed replacement：`PulseApplicationOwner.replace_active()` 返回 receipt future；Run owner lane 复用同一 compile/upload/prepare/FIRE seam，SAFE readback 后可重用同一 sequencer session/lease，不走 cancel→terminal→reap→new Run。replacement 编译结果按 document fingerprint/API 值在当前 plan 内缓存。
- PulseGUI 只在 receipt 到达后原子更新 held point/applied front；首尾 clamp、不 wrap、失败不清空旧文本。离线仍只可编辑/预览；普通 On-Pulse 文档替换不被这条 narrow seam 改写。
- Virtual/Remote/Qt 流已覆盖 hold、step、Stop；产品测试额外断言 hold/step 前后 `RunId` 不变。endpoint interrupt 只有在 SAFE readback 成功后才把旧段标为可再次 prepare；RTL/Tcl/XDC/bitstream/wire protocol 未改。

## C5 证据与规模

- C5 已提交；以当前 `git log -1` checkpoint 为准。
- 定向 PulseGUI/应用集合：`32 passed`，16.20 秒；Remote server 流 5.44 秒，Virtual hold/step 流 1.77 秒，编译与 transport I/O 均不在 Qt 线程。
- 全套回归：`824 passed, 1 skipped`，总耗时 150.31 秒；仅有既存 zmq Proactor selector-thread warning。`git diff --check`通过，C5 未改 RTL/Tcl/XDC/bitstream/wire protocol，受保护用户文件仍未读取、未修改、未 stage、未 commit。

## 当前下一步：C6 领域E2E、性能与全仓清理

只按 `docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md` §8 C6、§9 推进：领域 E2E、性能证据和全仓删除扫描。不得重做已经闭合的 C1–C5；不得为旧测试/API留兼容。若发现设计与实现矛盾，先停止该 cut，更新设计与删除清单后再实现。

恢复时严格依次完整读取当前 Goal、`AGENTS.md`、本文件、Git branch/HEAD/status/recent log，再只读台账当前未闭合项和设计对应章节。不得重答、重审或重做 C1/C2/C3/C4/C5；若发现矛盾，先停止该 cut，更新设计与删除清单后再实现。
