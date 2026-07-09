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

7. **设备自动发现 + Basler 监视相机接入(2026-07)**:`na.discover_devices()`(仿 confocal:枚举 Basler/pypylon + VISA `*IDN?`,缺库/空总线=提示行绝不 raise;相机行自带 ready config 片段直接喂 `load_devices`);camera 选择单源 `DeviceSet.camera_names()`(Camera 测量 + Pulse scan 两下拉,契约钉相等);`camera_spec/camera_measurement` 加 `camera` 参数(console Add Panel 下拉切 `monitor_camera` 即看图);PylonCamera 补 `pixel_format`,Software 触发自由跑**免脉冲免 sequencer 出帧**;`configs/basler_monitor.json`=虚拟读出链+真 Basler 渐进接线;**exposure 改为逐相机状态**(表单空白=保持选中相机现值——对抗审查揪出的主相机默认冻结误写 bug);tutorial 两本更新;守卫 `tests/test_device_discovery.py`。本机已装 pypylon。

8. **设备层去硬编码 + 五项体验修(2026-07,Y 轮)**:`discover()` 成为设备类的自描述协议(discovery 只汇集,import 失败=提示行);session **缺角色容忍**(按类型 bind 全部相机、`DeviceSet.default_camera_name()` 单源、camera_names 空即空、开序按类型)——config 只声明真实硬件(`basler_monitor` 仅一台 Basler);grid 单元字号恰两档(<2x2 小一档)+ 行缝=恰一行 title、列缝纯分隔(渲染 bbox 契约钉死);pulse_gui Scan 加 `◀ step`/`step ▶` 逐点步进调试(hold 单源);PylonCamera 分模式抓流(Software 常驻 LatestImageOnly 修 live 卡顿+旧帧显示;外触发每次重启保证一触发一帧不错位)+ ROI 停流/Offset 归零/硬件 Inc 对齐;console header **Selectors** 开关(就地武装/停用全部面板选择器)。守卫 test_grid_font_gap / test_console_selector_toggle / test_pulse_gui_scan_step + test_device_discovery 扩充。
- 待议(confocal 对照中未采纳,将来可讨论):DeviceManager 式 reload(改 config 就地重建变更设备)、unique_id 单例。

9. **设备选择中心化注入 + 设备管理 GUI(2026-07,AA 轮)**:把"按测量选设备"从逐 spec 手写收成**装饰器声明式**——spec 只写 `@measurement(devices=["camera"])` / `@task(devices=[("camera",{"default":"monitor_camera"})])`,唯一漏斗 `_open_registry.discovered_specs`(`readout.session.devices` 只此在作用域)调 `CatalogSpec.with_devices_bound` 一处①追加 `choice` 下拉(choices=`device_names(base_type)`,**camera 域列出 `camera`+`monitor_camera` 两台物理设备**——同类型、非两种角色)②`_bind_device_args` 把选中的**名字解析成设备实例**注入 build。**注意 `camera`/`monitor_camera` 不是两种角色类型**,是同一 `CameraDevice` 域里两台物理相机(读出 qCMOS 在 emCCD 线 / MOT 监视相机在 mot_trigger 线);"主相机惯例命名 `camera`"仅用于默认解析(`default_device_name` 无命中回退首台,不崩)。这**取代**了 ROADMAP §7 里"camera 选择单源 `DeviceSet.camera_names()`"的旧表述(单源现在是 `device_param`/`with_devices_bound`,`camera_names()` 只是 `device_names(CameraDevice)` 的薄封装)。四 spec(pulse_scan/camera_spec/mot_field/calibrate)变声明;温度/保真**故意锁读出科学相机**不声明。GUI:`exp.device_manager()`(按域列设备+Scan hardware)+ task console 顶栏 "Devices" 按钮 + tutorial/手册补教学。真 bug 修:`exp.capture(camera="monitor_camera")` 曾 `IndexError`(读出成像序列只脉冲 emCCD 线,监视相机在 mot_trigger 线→0 帧→空索引)→改清晰可操作 RuntimeError。守卫 `tests/test_device_role_injection_contract.py`(Pin A/B/C)。e2e 真人流程回放全合理(MOT 寻优 monitor 相机误差<1格)。**待办:N6 choice 下拉热更新、qCMOS config 扁平化(需真机)、S3/S4 已进本文件权威源。**

**下一步(待做)**
- 真机 qCMOS 相机后端(`devices/qcmos.py`)接 PSF/bimodal 读出,在真实数据上验证保真度;4-shot group / 参考帧定 ground-truth 标签作为可选标定流程(算法已具备,缺采集编排)。换真机时:`na.connect("qcmos", ...)` + loading 节点组合(`CameraMeasurement`+`OccupancyProcessor`+`CalibrateReadoutTask`),分析/逻辑节点代码不动;**温度/读出测量(`exp.readout.temperature`/`readout_duration_fidelity` 与 GUI 一键 Start)走同一路径**,只换 connect。
- 真机 MOT 监视相机验收(代码侧已备好,等相机插上):`na.discover_devices()` 看到 acA1920-155um → `na.connect("basler_monitor")` notebook capture 出图 → console Camera 下拉 `monitor_camera` 出图;接 FPGA 触发线后 `trigger_source` 改 `"Line1"`,脉冲模板三条线圈总线对准真实 DAC 通道(见 device manual"第二只相机"节)。
- 表单跨字段联动(如 camera 下拉切换时刷新 exposure 显示为该相机现值)——需要 ParamDecl `depends_on` 推广到非 pulse_param 字段,frontend 通用机制,待拍板。
- **性能激进档需用户拍板**:迟滞 autoscale / 错峰重绘 / 嵌入面板低 dpi 性能模式(均改可见行为或视觉,见性能结论)。
- 其余 GUI 待决方向(见下"暂缓:GUI 相关")。
- **✅ Pulse-scan 解耦重设计(已完成,commit b318deb+cc79700)**:pulse-scan 现是**新的 device-driving `PulseScanNode`**(`operations/logic.py`),不再带内置 reducer。① api slot 固定、scan slot 每点 `with_slots_resolved` 解析成 x(1D=一 slot 值,2D 两参 lockstep);② y **解耦**=订阅其它跑着节点的 hub 信号(如 occupancy `rate`)+`value=...` 表达式;③ 设备锁定只一 driver→节点自己 fire+采帧+publish 裸 `frame`,消费者(自己线程)刷新后**等其 y 信号 per-signal 版本号 bump**(只读 hub 无竞态;headless 传 inline settle)再读 y;④ **唯一可复用求值器 `operations/signal_expr.py::SignalExpr`**(多槽 signal+`value=`契约+co_names+resolve+`hub_namespace`),PanelCard 委托它,occupancy 等所有 `kind="signal"` source 升级新 ParamDecl kind `signal_expr`(GUI `_SignalExprWidget`)。三档 DPR 几何验收过;守卫 `tests/test_signal_expr.py`+`test_pulse_param_scan.py`。

- **🔬 段描述符 DAC 延迟重设计(RTL/host/测试/文档已落地,待上机 — 用户 /goal 要求)**:根因=旧 `g_busdly` 逐 bit 事件调度器逐 tick 采样 post-Bresenham 输出、每 bit 变化推一个事件,所以一条密集 ramp 被延时后在途 = 每帧值变化数 × ⌈d/帧⌉(pulse_test +1s ≈ 25050 ≫ `BUS_EVT_DEPTH`=64,被拒),而等价负延时经全局平移 G 折成净 0 直通(见 FPGA 手册第六章新增"正/负不对称"段)。**定案架构**=逐**总线**"段描述符"延迟替换逐 bit 值-变化 FIFO:主段播放器每应用一个段时捕获**已解析**的段 `{emit=apply_tick+d, vstart, target, span, step/rem, mode}` 推浅 per-bus FIFO,延迟重播器到点重跑 Bresenham,首个前 SAFE(首帧正确)、done 后跑 d(done-tail)。深度=在途**段数**(pulse_test +1s=**100** vs 25050),密度无关,逐总线(4)非逐 bit(40)→省 LUT。**否决** skip 计数器(loop_end 带 scan 系数逐点变、无单一周期,`d mod 周期` 会 first-frame 错=回归 e3fb639)与"重跑生成器 at t−d"(#ramp-carry + scan slot 历史缓冲太险)。TTL 保留事件 FIFO。**P1 完成**:`fpga/pulse_streamer/host/engine_model.py` 加 `bus_play(apply_log/carry_out)`+`bus_undelayed_and_log`+`rtl_bus_segment_delay_mirror`+`_segment_replay_step`,`tests/test_bus_segment_delay_equivalence.py` **60 例字节精确**对拍不变 reference `bus_delay_line_reference`(含 dense/scan/repeat/finite/done-tail/帧边界)。**已落地(待上机)**:D3 RTL(`zlc_edge_streamer.v` `g_busseg` 逐总线段延迟 + 延迟重播器 + done-tail SAFE-hold)、D4 host 段计数容量界 + 打包 + 清 value-change 残余、D5 测试/文档/记忆均已 commit(见 [[dac-segment-delay-rtl-landed-2026-07-08]])。真机报"加 delay 后 DAC 变常数"=两个 RTL-only NBA 陈旧 bug(fend freeze 塌 + tick0 陈旧 `del_bus_ticks`),已修(51cf378)。**LUT 装不下 → A+B(不是 A+C)**:含 g_busseg 的 build 在 `evt_fifo_depth`=128 / `bus_seg_addr_width`=6 下布局失败(21958/20800 Slice LUT)。**先试 A+C 撞大坑**:`bus_seg_addr_width`(C)不是省 LUT 旋钮而是**寄存器映射/ABI 参数**——改它把 R_DELAY 区基址前移 896 words,host 按新映射打包但真机跑旧 bitstream→**连不加 delay 的 scan 都错**(用户报"乱修")。**发布配置=A+B**(均寄存器映射无关的内部 FIFO 深度):`evt_fifo_depth`=64(TTL 事件 LUTRAM 减半 ~-1.2k)+ `bus_evt_fifo_depth`=32(逐总线段 FIFO 减半 ~-0.6k),`bus_seg_addr_width` **保持 6** → 估 ~20.2k(余 ~0.6k),待用户 rebuild + 上机验证。铁律:**ABI/寄存器映射参数绝不为省资源去动;LUT 只从寄存器映射无关的内部深度省**。

- **🔒 几何指纹握手 = host↔bitstream 兼容根修(已落地,待上机 — 用户 /goal「找本源、高层设计、系统性修同类」)**:本源=旧 `REGISTER_LAYOUT_ID`(静态 0x5A4C4C02,CTRL word 63)只版本化寄存器**结构**、不覆盖决定实际地址的**几何**,所以 config↔bitstream 任一几何参数漂移都**静默损坏**(上面 bus_seg 那次正踩中)。修=`image.build_fingerprint(params)`=hash(`LAYOUT_STRUCT_VERSION` + 所有 bitstream-affecting StreamerParams 字段;高字节 0x5A 永不 0、自识别),build 作 `LAYOUT_FINGERPRINT` generic 驱到 word 63,`axi_session.check_register_layout` connect 时比对 `build_fingerprint(self.params)`,不符**硬报错"rebuild"**(旧 bitstream 返回旧值/0→被抓)。DRY:哈希只在 image.py 一处,RTL 只携带 build 期算好的值。契约测试:`build_fingerprint_covers_geometry`(逐字段 bump 断言几何字段必变/host-only 不变)、`all_geometry_params_config_matches_rtl_defaults`(**全**几何 config==top.v==engine.v .v 默认 + NUM_DELAY_CH,补指纹"能撒谎"盲区:非-generic 参数的 .v 默认可与 config 漂移而指纹是从 config 算的看不见)、`test_final_top_regions_match_image` 扩成钉**全 21** CtrlWords 偏移(原只 6,CLK_ENABLE-46→20 garbled-strobe 类)。**geometry-drift-audit workflow(5 agent)**证实并已修上述 + fallback 字面量 128/64→64/32。**待办同类**(latent/别子系统,详见 [[geometry-fingerprint-handshake-2026-07-09]]):coeff_frac_bits 编译器硬编码 8(config 惰性=假单源,`affine_coeffs`/`RuntimeSequenceProgram` 应读 config)、SCAN_ADDR_WIDTH/BUS_INDEX_WIDTH 理想应 RTL `$clog2` 派生(现由契约测试守漂移)、bus_evt 5-way host 常量测试、busimg BRAM Write_Depth_A 硬 2048 未由 bus_rows 派生、clock_hz build-time 守卫(哈希观测不到真实晶振)、`PulseTableState.from_dict` 不校验 version、UART FRAME_WORDS wrap(xsim-gated,上机前修)、saved-figure .npz 无 schema/version、IMAGE_MAGIC/STATUS_ERROR 死握手。本机无 xsim,RTL 由 Python 镜像 + 逐寄存器自审 + 对抗审查锁定。

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
