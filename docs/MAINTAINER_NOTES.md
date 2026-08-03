# Maintainer checkpoint

本文件只记录最新交接点。规范架构见`docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md`，执行方法见`AGENTS.md`，活动问题见仓库外`../ARCHITECTURE_AUDIT_CURRENT.md`。

## 当前状态

- Branch：`codex/system-architecture-migration`。
- 重开基线：`476d125304fc90b6ef4f5009184dd2119206fcc2`；C0规范提交：`ed6dfe21c0d99999444c099d8c644a20b44e561d`。
- 当前checkpoint：§8 C1 `zlc_plot`单一绘图全栈替换已经dependency-closed；下一唯一工作是C2 `DeviceInstance` graph。不得返回旧M1–M7叙事，也不得重做已经闭合的C1。
- 预期worktree例外：未跟踪用户文件`pulses/scan_test.json`。永远不得读取、修改、移动、删除、stage或commit。
- RTL、Tcl、XDC、bitstream和wire protocol仍冻结；C1没有硬件改动。

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

## 当前下一步：C2 DeviceInstance graph

只按`docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md` §2、§6.6、§8 C2、§9推进：原位改写installation document为ordered heterogeneous instances；把schema/defaults/factory/capability移到device leaf；迁移DeviceManager、composition和全部消费者；同cut删除flat backend config/package/plan/dispatch、digest/CAS链、中央concrete imports和旧tests。开始修改前先从当前tree列出C2精确owner/consumer/deletion manifest，不复核C1绘图算法。

恢复时严格依次完整读取当前Goal、`AGENTS.md`、本文件、Git branch/HEAD/status/recent log，再只读台账当前未闭合项和设计§2、§6.6、§8 C2、§9。不得重答、重审或重做C1。
