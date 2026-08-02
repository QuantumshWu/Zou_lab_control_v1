# Maintainer checkpoint

本文件只记录当前交接点。规范架构见`docs/SYSTEM_ARCHITECTURE_DESIGN_zh.md`，执行方法见`AGENTS.md`，活动问题清单见仓库外`../ARCHITECTURE_AUDIT_CURRENT.md`。

## 当前状态

- Branch：`codex/system-architecture-migration`。
- 重开基线HEAD：`476d125304fc90b6ef4f5009184dd2119206fcc2`。
- 当前checkpoint：C0规范与清单冻结；旧“M7软件已闭合/只剩真机E0”结论已被用户现场证据推翻，不得继续引用。
- 当前实现仍为NOT DONE；下一唯一工作是§8 C1的`zlc_plot`单一绘图全栈替换。
- 预期worktree例外：未跟踪用户文件`pulses/scan_test.json`。永远不得读取、修改、移动、删除、stage或commit。
- RTL、Tcl、XDC、bitstream和wire protocol冻结；C0无硬件改动。

## C0 已冻结的事实

- 生产包固定基线：381 Python modules / 161,566 physical LOC / 918 top-level classes / 552 dataclasses / 32 enums。tests为131 modules / 49,822 LOC。
- 外部接纳源：`zlc_plot main@4fca73fcafc5b0a65a994399cf4641ed3b52bc8a`。只纳入tracked核心、`py.typed`和唯一Helvetica asset；排除外部`zlc_data`、`qt_controls.py`、notebook/PNG/build/egg-info/cache。
- 拟纳入zlc_plot基线：30 modules / 25,000 LOC / 178 classes / 131 dataclasses / 17 enums。
- 当前frontend旧绘图删除闭包：69 modules / 36,910 LOC；当前data Fit删除闭包：7 modules / 4,931 LOC；DataFigure第二Fit/worker闭包另有3 modules。C1必须明显净删，不能再包wrapper。
- 数据权威仍是当前`zlc_data`的`OwnedSnapshot/DataSchema/PointTable/GridTopology/dtype/validity`；zlc_plot只通过一个私有readonly、尽量零拷贝adapter消费，不引入第二public Dataset。
- 接纳源的未引用轴聚合/全数组Histogram flatten/R-as-Rolling-history已被证伪。必要public修订固定为`HistogramPlot.samples`、移除`RollingPlot.x`并使用session-private revision history、`PlotSession.replace_spec()`、现有`SelectionData/FitSelection.source_revisions`、`fit_all_facets: bool`和`FacetFitBatchResult`；不增加public axis-selection/history Dataset/DTO。`FitScope`继续表示selector/viewport/all样本范围。
- Device终态是ordered heterogeneous DeviceInstance graph + leaf type descriptor；Logic终态是一个generic host + 极小leaf；Task takeover、Plot/Logic Edit同draft、visible project task/run outputs与Pulse narrow hold/step均已写入唯一架构文档。

## C1 完成条件

1. 先建立接纳源characterization：六plot kinds、selector、Fit、DPR/resize、live/raster/style/export/PulseTimeline、N-D/validity/non-grid/grid/dtype/zero-copy，以及物理维覆盖、Histogram显式samples、无topology双point-axis拒绝、Rolling私有history/monitor-latest/exact displayed revisions/spec-change reset、Camera monitor publication=`(1,1,*frame)`。
2. 在同一未提交dependency-closed cut中纳入zlc_plot、完成唯一data adapter并迁移TaskConsole、Calibration、Occupancy、DataFigure、FigureViewer、Edit、Pulse preview与public API。
3. 同cut删除全部旧plot/projection/selector/Fit/render/raster/style/layout owners、`MONITOR_HISTORY` Dataset role、Camera `history_cycles`与point-history/capture-preview路径、`MonitorDataset.append_window`环形分支、旧tests、font副本和死imports；`MonitorDataset.latest_cell`必须保留原stream validation/revision/event-ref/gap/atomic snapshot职责。不得出现兼容re-export、history-in-P或双runtime checkpoint。
4. 正式Qt快轨至少证明Camera→live Image→Area→第二Image→Fit、Distribution、Calibration report、FacetGrid all-facets、FigureViewer与Pulse preview。
5. 记录固定口径LOC/class/dataclass/enum、删除owner、profiling和GUI证据后才能commit并进入C2。

## 恢复协议

每次恢复严格依次完整读取当前Goal、`AGENTS.md`、本文件，再检查branch/HEAD/status/recent log；只读临时台账§0与当前C1相关的系统设计§2–§4、§6.4–§6.5、§8 C1、§9。不得重答或重做C0，不得从旧M1–M7叙事恢复。
