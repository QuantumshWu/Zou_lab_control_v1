# ROADMAP / 当前焦点 / 待决方向

> 这里记**现在在做什么、需求/约束、暂缓的事、要拍板的决定**,以及一些**下一阶段值得采纳的设计想法**。
> 工作守则见 `AGENTS.md`;子系统深档见 `docs/MAINTAINER_NOTES.md`;Task 控制台设计见 `docs/task_console_design/`。

## 当前焦点(2026-06)

notebook 调用侧的解耦 + Rb87 读出 + **task_console 大改(Monitor/Control + 通用 Measurement 框架 + 一键温度)已落地**(见下"已完成");下一步是真机 qCMOS 接线与性能激进档拍板。

**已完成(2026-06)**
1. **解耦**:`neutral_atom` 不再 import `frontend`(IoC viewer 注册表 `_viewer_registry`,双向 import 期互不拉对方;实验层可 headless 导入)。
2. **子系统拥有逻辑**:读出编排(sitemap/thresholds/detect/detection-time)从 `session` 上帝对象搬进 `ReadoutSubsystem`,session 退成门面;签名明确不再 `**kwargs` 转发。
3. **Rb87 读出接入**:PSF 匹配滤波提取(`core/psf.py`)+ 双高斯定阈/保真度(`core/bimodal.py`),`TrapCalibration` 加 `method='box'|'psf'` 经 `signals()`/`detect()` 单点分派(box 仍默认);`readout.sitemap(method="psf")` / `thresholds(method="bimodal")`;虚拟后端端到端可测。**只移植算法,不带 rb_qcmos 的文件IO/缓存/批处理脚手架**。
4. **虚拟==实机机械强制**:核心准则"虚拟测试走实机同一代码路径,只 fake 数据源"写进 AGENTS §2 并由 `tests/test_virtual_equals_real_contract.py` 强制(分析层 import 具体后端/读仿真真值即挂测,含 `session.py`)。配套:`session.connect()` 去掉对 `VirtualTrapArray` 的 import + 字段内省(虚拟配置搬进后端,经 registry `resolve_connect_config` 分派);task-console 的 loading 读出从绑死虚拟 trap 的单体节点重构成走 `CameraDevice` 契约的 backend-agnostic 逻辑节点组合(`CameraMeasurement` 产 frame + `OccupancyProcessor` 逐帧真 detect + `CalibrateReadoutTask` 标定,真机只换相机)。读出数学(高斯/正态CDF/双高斯/保真度)提到 dependency-free `_readout_math.py`,core 与 frontend 共用一份(`tests/test_readout_math_single_source.py` 守);所有 plot 必须继承 `BaseLivePlot`(`tests/test_frontend_plot_contract.py` 守)。
5. **task_console 大改(commit 4525c1e,对抗式验收 PASS)**:① Tab 改 **Monitor**(实时拖拽 live 网格)+ **Control**(Processing 处理/保存 + Measurement);② 每图 **Setting 瘦身**=只放基本(信号源/size/colormap/colorbar 显隐/unit 切换/lim auto-manual),Fit + 自适应 range 迁 Control,cbar/unit/lim 持久化进 `PanelConfig.params` 重建时重应用;③ **通用 Measurement 框架**(`operations/measurement.py`):`MeasurementSpec`/`ParamDecl` 声明式单一真相源 → `ScanAxis`/`ShotPlan`/`PointReducer` 三角色 → `ScannedMeasurement` 引擎 → `ScannedMeasurementNode`(`operations/logic.py`)推扫描点到 hub(自停);单一真相源建器 `build_*_scan`,GUI 目录 `exp.readout.measurement_specs()`;④ **一键温度**(`operations/temperature.py`):release-recapture 弹道重捕(6 周期双触发、t_off=duration scan slot、`SurvivalReducer`、`fit_temperature` 纯后处理、capture_radius 简并),虚拟 trap-off 损失模型只在数据源侧;⑤ **空白收紧** `PANEL_MARGINS_PX` 现为 `(110,86,80,70)`(右/下/顶相对 confocal 收紧到数据占比~50%,左保 confocal `110`=`STOCK_MARGINS_PX[0]` 不裁 4-5 位 y 刻度+y 标题) + footer 紧贴 canvas(残留右 gutter/tiling slack 是固定拖拽网格的结构性权衡);⑥ **性能结论(诚实)**:瓶颈=密封 300dpi 文字栅格化(intrinsic ~72ms/5面板),blit 与位置冻结实测否决,激进提速需用户拍板。文档:`docs/task_console_design/`(Monitor/Control + Measurement + 一键温度 + 性能) + frontend/main 手册 + MAINTAINER_NOTES §19 + tutorial 一键测温小节。

6. **MOT 磁场寻优全链(commit de46622,2026-07)**:第二只相机 `monitor_camera` 设备对(`devices/pylon.py::PylonCamera` 真机骨架 / `VirtualMotCamera` 只感知 fired sequence:`decode_analog_bus` 逆变换 + 3D 高斯 MOT 模型);`mot_roi_intensity` 单源(live `MOT intensity` 处理器与 task 共用);Pulse scan 加 `camera` 选择 + api 软件扫补 `scan_shape` 声明(与硬件表同构,3 维线圈扫可 facet);一键 `Optimize MOT field` task(argmax+3³ 质心细化,虚拟 e2e 收敛 b0±半步);顺手修 `_set_api_field(dac)` 不 apply states 的既有 bug。守卫 `tests/test_mot_field.py`;device manual 新增"第二只相机"教学节。

**下一步(待做)**
- 真机 qCMOS 相机后端(`devices/qcmos.py`)接 PSF/bimodal 读出,在真实数据上验证保真度;4-shot group / 参考帧定 ground-truth 标签作为可选标定流程(算法已具备,缺采集编排)。换真机时:`na.connect("qcmos", ...)` + loading 节点组合(`CameraMeasurement`+`OccupancyProcessor`+`CalibrateReadoutTask`),分析/逻辑节点代码不动;**温度/读出测量(`exp.readout.temperature`/`readout_duration_fidelity` 与 GUI 一键 Start)走同一路径**,只换 connect。
- 真机 MOT 监视相机接线:装 `pypylon`,配置里 `monitor_camera` 换 `PylonCamera`(serial/trigger_source/exposure),脉冲模板三条线圈总线对准真实 DAC 通道——分析/task 代码不动(见 device manual"第二只相机"节)。
- **性能激进档需用户拍板**:迟滞 autoscale / 错峰重绘 / 嵌入面板低 dpi 性能模式(均改可见行为或视觉,见性能结论)。
- 其余 GUI 待决方向(见下"暂缓:GUI 相关")。
- **✅ Pulse-scan 解耦重设计(已完成,commit b318deb+cc79700)**:pulse-scan 现是**新的 device-driving `PulseScanNode`**(`operations/logic.py`),不再带内置 reducer。① api slot 固定、scan slot 每点 `with_slots_resolved` 解析成 x(1D=一 slot 值,2D 两参 lockstep);② y **解耦**=订阅其它跑着节点的 hub 信号(如 occupancy `rate`)+`value=...` 表达式;③ 设备锁定只一 driver→节点自己 fire+采帧+publish 裸 `frame`,消费者(自己线程)刷新后**等其 y 信号 per-signal 版本号 bump**(只读 hub 无竞态;headless 传 inline settle)再读 y;④ **唯一可复用求值器 `operations/signal_expr.py::SignalExpr`**(多槽 signal+`value=`契约+co_names+resolve+`hub_namespace`),PanelCard 委托它,occupancy 等所有 `kind="signal"` source 升级新 ParamDecl kind `signal_expr`(GUI `_SignalExprWidget`)。三档 DPR 几何验收过;守卫 `tests/test_signal_expr.py`+`test_pulse_param_scan.py`。

> `references/` 是历史源码归档(`rb87_readout_v16`、confocal GUI 等),**git ignore、只在本地存在**,是借鉴/移植的来源,不是被本仓库 import 的依赖(见 README 目录树)。
> **最小跑通入口**:`task_console.bat`(虚拟仪表盘)/ `tutorials/` 里的 notebook / `na.connect("virtual")` 起一个全虚拟 session——先在虚拟后端把调用链串通,再换真机后端。

## 长期约束(贯穿,见 AGENTS.md)
notebook-first;子模块只经接口互联(解耦);无后向兼容;前端密封;所有可视化三档 DPR 截图验收;全中文;真机正确性根因在自己代码。

---

## 下一阶段值得采纳的设计想法(灵感来自 confocal GUI)

> **这些只是想法库,不是必须照搬的规范。** 要遵守的是**设计原则**(解耦 / 只经接口互联 / 单一真相源 / 虚拟可替换 / 显式优于隐式),**不是 confocal 的具体实现**。哪条有更好的思路就用更好的——confocal 只是一个被认可的参考实现,不是模板。
>
> (来源 `references/.../Confocal_GUIv2_refactored_v6`,git-ignore 历史归档。)

1. **声明式实验/测量元数据**:一个测量类用装饰器(confocal 的 `@measurement_gui_meta`)或 `caller()` 签名**自带它的参数与分类**(context / 要保存的 config / 设备槽 device-overrides / 额外参数),单位和默认值就在签名里。借鉴:让每个"实验任务"自带参数接口,notebook/GUI 从签名/schema 生成交互面板,不手写。(我们的 frontend `ParamDecl` 声明 + `ParamWidgetHandler` 注入注册表[#H3r-F5,`frontend/param_widgets.py`]是这个思想在面板层的局部实现:每 kind 一个 handler,measurement 表单/Setting/Edit 共用,可向实验层推广。)
2. **设备契约 + 注册表**:设备通过抽象基类定义契约(`BaseCounter`/`BaseLaser`…),上层不绑死硬件;按名字注册/取设备(device manager)。我们已有 `devices/base.py` 三契约 + registry,继续沿用并补齐。
3. **虚拟设备信号注册表 + 物理 dataclass**:`@VirtualCounter.register_signal('ple')` 按测量名查表产期望计数;物理常数集中在 dataclass(峰位/宽/衰减/漂移)。借鉴:为 Rb87 建 `@dataclass VirtualAtom`(能级/亮暗率/loss/drift),按实验名注册信号模型——**加新实验不用改相机/虚拟设备代码**。我们的 loading 读出节点组合(`CameraMeasurement`+`OccupancyProcessor`,虚拟相机驱动)是雏形,可泛化成注册表。
4. **measurement 生命周期 + 线程分工**:worker daemon 线程跑 `_loop()` 只产数据,前端定时器/notebook 轮询只读;matplotlib artist 由一处统一管;update 策略用**显式 dispatch 字典**(add/replace/create/roll)而非动态 getattr。我们的 LogicNode/SignalHub 已是这套(生产者线程 + hub 拷贝 + GUI 读副本),继续保持。
5. **协作式取消(cancellation token)**:`cancel()`/`check_cancel()`/`reset_cancel()` + 阻塞操作后放检查点,优于 `is_cancel` 布尔属性。借鉴:长采集/扫描的可中断性标准化成一个小模块。
6. **分层 cleanup 钩子**:base 类在 `__init_subclass__` 里 wrap `close()`,分层清理(monitor → remote → user → 缓存)+ atexit。借鉴:Task/Device/Session 关闭时自动清 worker 线程、临时文件、远程连接。

> 接 notebook+Rb87 时**按需取用**:能让架构更解耦/更可测/更易扩展就用,否则不用。核心是原则,不是具体形态。

---

## 暂缓:GUI 相关(等 notebook 侧满意后再回来)

来自 `docs/task_console_design/` 的"待决方向"(GUI 部分先放一放):
- editor 是否要直接编辑 live 图(目前是快照,简单安全);2D clim 数值框 + cmap 下拉;拟合模型扩展。
- 时序/漂移面板、逐站保真度条、ablation 曲线等新面板类型。
- 控制台 ↔ 脉冲 GUI 联动(同进程触发 on_pulse / 改 detection time)。
- 多 group / 4-shot 结构在仪表盘层的抽象(多数情况"循环里算完再 publish"已足够)。

## 真机逻辑节点与测量(已落地)
- ~~入库一个从 `NeutralAtomSession` 采集循环 publish 的 loading 节点~~ → 已落地为 backend-agnostic 逻辑节点组合(`CameraMeasurement`+`OccupancyProcessor`+`CalibrateReadoutTask`,走 `CameraDevice` 契约)。
- ~~readout 提供"逐 shot 派生量"helper(保真度/温度)~~ → 已落地为通用 Measurement 框架 + 一键温度(`exp.readout.temperature`/`readout_duration_fidelity`/`measurement_specs`);GUI Control 段一键 Start。`atom_temp_monitor` 类任务现可直接由 Measurement 段驱动。

---

## 要你拍板的决定(开放)
- **task_console 性能激进档**:瓶颈=密封 300dpi 文字栅格化(intrinsic),合规提速(空白收紧)已做;再快需迟滞 autoscale / 错峰重绘 / 低 dpi 性能模式——均会改可见行为或视觉,**要不要做、接受哪档由你定**(blit 与位置冻结已实测否决)。
- root `AGENTS.md` 之外是否还要单独 `CLAUDE.md`?(目前合并在 AGENTS。)
- 常犯错误目录的"自动记录"是否要加 Stop hook 提醒?(目前靠版本化目录 + MEMORY 指针 + 收尾自觉追加。)
- notebook 实验任务是否采用 confocal 式声明式装饰器(想法 1),还是更轻的约定?
