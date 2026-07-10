# ROADMAP / 当前焦点 / 待决方向

> 这里记**现在在做什么、需求/约束、暂缓的事、要拍板的决定**,以及一些**下一阶段值得采纳的设计想法**。
> 工作守则见 `AGENTS.md`;子系统深档见 `docs/MAINTAINER_NOTES.md`;Task 控制台设计见 `docs/task_console_design/`。

## 当前焦点(2026-06)

notebook 调用侧的解耦 + Rb87 读出 + **task_console 大改(Monitor/Control + 通用 Measurement 框架 + 一键温度)已落地**(见下"已完成");下一步是真机 qCMOS 接线与性能激进档拍板。

**已完成(2026-06)**
1. **解耦**:`neutral_atom` 不再 import `frontend`(IoC viewer 注册表 `_viewer_registry`,双向 import 期互不拉对方;实验层可 headless 导入)。
2. **子系统拥有逻辑**:读出编排(sitemap/thresholds/detect/detection-time)从 `session` 上帝对象搬进 `ReadoutSubsystem`,session 退成门面;签名明确不再 `**kwargs` 转发。
3. **Rb87 读出接入**:PSF 匹配滤波提取(`core/psf.py`)+ 双高斯定阈/保真度(`core/bimodal.py`),`TrapCalibration` 加 `method='box'|'psf'` 经 `signals()`/`detect()` 单点分派(box 仍默认);`readout.sitemap(method="psf")` / `thresholds(method="bimodal")`;虚拟后端端到端可测。**只移植算法,不带 rb_qcmos 的文件IO/缓存/批处理脚手架**。
4. **虚拟==实机机械强制**:核心准则"虚拟测试走实机同一代码路径,只 fake 数据源"写进 AGENTS §2 并由 `tests/test_virtual_equals_real_contract.py` 强制(分析层 import 具体后端/读仿真真值即挂测,含 `session.py`)。配套:`session.connect()` 去掉对 `VirtualTrapArray` 的 import + 字段内省(虚拟配置搬进后端,经 registry `resolve_connect_config` 分派);task-console 的 loading 读出重构成 `CameraMeasurement(frame_0) → OccupancyProcessor(typed detect)`，标定由独立 `CalibrateReadoutTask` 产文件，真机只换相机。读出数学提到 dependency-free `_readout_math.py`,core 与 frontend 共用一份;所有 plot 必须继承 `BaseLivePlot`。
5. **task_console 大改(commit 4525c1e,对抗式验收 PASS)**:① Tab 改 **Monitor**(实时拖拽 live 网格)+ **Control**(Processing 处理/保存 + Measurement);② 每图 **Setting 瘦身**=只放基本(信号源/size/colormap/colorbar 显隐/unit 切换/lim auto-manual),Fit + 自适应 range 迁 Control,cbar/unit/lim 持久化进 `PanelConfig.params` 重建时重应用;③ **通用 Measurement 框架**(`operations/measurement.py`):`MeasurementSpec`/`ParamDecl` 声明式单一真相源 → `ScanAxis`/`ShotPlan`/`PointReducer` 三角色 → `ScannedMeasurement` 引擎 → `ScannedMeasurementNode`(`operations/logic.py`)推扫描点到 hub(自停);单一真相源建器 `build_*_scan`,GUI 目录 `exp.readout.measurement_specs()`;④ **一键温度**(`operations/temperature.py`):release-recapture 弹道重捕(6 周期双触发、t_off=duration scan slot、`SurvivalReducer`、`fit_temperature` 纯后处理、capture_radius 简并),虚拟 trap-off 损失模型只在数据源侧;⑤ **空白收紧** `PANEL_MARGINS_PX` 现为 `(110,86,80,70)`(右/下/顶相对 confocal 收紧到数据占比~50%,左保 confocal `110`=`STOCK_MARGINS_PX[0]` 不裁 4-5 位 y 刻度+y 标题) + footer 紧贴 canvas(残留右 gutter/tiling slack 是固定拖拽网格的结构性权衡);⑥ **性能结论(诚实)**:瓶颈=密封 300dpi 文字栅格化(intrinsic ~72ms/5面板),blit 与位置冻结实测否决,激进提速需用户拍板。文档:`docs/task_console_design/`(Monitor/Control + Measurement + 一键温度 + 性能) + frontend/main 手册 + MAINTAINER_NOTES §19 + tutorial 一键测温小节。

6. **MOT 磁场寻优全链(commit de46622,2026-07)**:第二只相机 `monitor_camera` 设备对(`PylonCamera` / `VirtualMotCamera`);`mot_roi_intensity` 为 live processor 与 task 共用数学单源。一键 `Optimize MOT field` 要求模板三根 DAC 都绑定 hardware scan slots，整张三维表一次上传，由 task 读取相机与强度并求最优点。通用 PulseScan 不拥有相机，但同时支持 `scan_slot` 整表执行与 `api_slot` 逐点有限 pulse；MOT-field 模板只允许前者。

7. **设备自动发现 + Basler 监视相机接入(2026-07)**:`na.discover_devices()`(仿 confocal:枚举 Basler/pypylon + VISA `*IDN?`,缺库/空总线=提示行绝不 raise;相机行自带 ready config 片段直接喂 `load_devices`);相机选择由设备域声明式注入到真正拥有相机的 Camera measurement / task；Pulse scan 没有相机参数。`camera_spec/camera_measurement` 的下拉可切 `monitor_camera`;PylonCamera 补 `pixel_format`,Software 触发自由跑**免脉冲免 sequencer 出帧**;`configs/basler_monitor.json`=虚拟读出链+真 Basler 渐进接线;**exposure 改为逐相机状态**。守卫 `tests/test_device_discovery.py`。

8. **设备层去硬编码 + 五项体验修(2026-07,Y 轮)**:`discover()` 成为设备类的自描述协议(discovery 只汇集,import 失败=提示行);session **缺角色容忍**(按类型 bind 全部相机、`DeviceSet.default_camera_name()` 单源、camera_names 空即空、开序按类型)——config 只声明真实硬件(`basler_monitor` 仅一台 Basler);grid 单元字号恰两档(<2x2 小一档)+ 行缝=恰一行 title、列缝纯分隔(渲染 bbox 契约钉死);pulse_gui Scan 加 `◀ step`/`step ▶` 逐点步进调试(hold 单源);PylonCamera 分模式抓流(Software 常驻 LatestImageOnly 修 live 卡顿+旧帧显示;外触发每次重启保证一触发一帧不错位)+ ROI 停流/Offset 归零/硬件 Inc 对齐;console header **Selectors** 开关(就地武装/停用全部面板选择器)。守卫 test_grid_font_gap / test_console_selector_toggle / test_pulse_gui_scan_step + test_device_discovery 扩充。
- 待议(confocal 对照中未采纳,将来可讨论):DeviceManager 式 reload(改 config 就地重建变更设备)、unique_id 单例。

9. **设备选择中心化注入 + 设备管理 GUI(2026-07,AA 轮)**:真正拥有相机的 measurement/task 通过装饰器声明 `devices=["camera"]`，注册表唯一漏斗把设备名下拉解析为 `CameraDevice` 实例。`camera` 与 `monitor_camera` 是同一设备域里的两台物理相机，不是两种角色类型。Camera measurement、MOT-field 与 calibrate 声明该域；Pulse scan 不声明相机，温度/保真则故意锁定读出科学相机。GUI 入口是 `exp.device_manager()` 与 task console 的 Devices 按钮；守卫为 `tests/test_device_role_injection_contract.py`。

10. **统一数据/拓扑/交互契约(2026-07)**:`SignalTensor` 固定物理 `(R,P,*data_shape)`，只把逻辑 `point_shape` 展成 P，任意维 `data_shape` 原样保留并带 `(R,P)` validity；`PortCatalog` 是 sequencer 唯一不可变拓扑，`PulseTableState` 不再携带平行 channel/bus/clock 结构；fit 由 `FitRequest`/`FitResult` + `core.fitting` 单源，selector 只产生 typed `Selection`，显式 action 决定 fit/ROI。对应守卫集中在 signal transport/scan tensor/port catalog/shared fit 测试。

**下一步(待做)**
- 真机 qCMOS 相机后端(`devices/qcmos.py`)接 PSF/bimodal 读出,在真实数据上验证保真度;4-shot group / 参考帧定 ground-truth 标签作为可选标定流程(算法已具备,缺采集编排)。换真机时:`na.connect("qcmos", ...)` + loading 节点组合(`CameraMeasurement`+`OccupancyProcessor`+`CalibrateReadoutTask`),分析/逻辑节点代码不动;**温度/读出测量(`exp.readout.temperature`/`readout_duration_fidelity` 与 GUI 一键 Start)走同一路径**,只换 connect。
- 真机 MOT 监视相机验收(代码侧已备好,等相机插上):`na.discover_devices()` 看到 acA1920-155um → `na.connect("basler_monitor")` notebook capture 出图 → console Camera 下拉 `monitor_camera` 出图;接 FPGA 触发线后 `trigger_source` 改 `"Line1"`,脉冲模板三条线圈总线对准真实 DAC 通道(见 device manual"第二只相机"节)。
- 表单跨字段联动(如 camera 下拉切换时刷新 exposure 显示为该相机现值)——需要 ParamDecl `depends_on` 推广到非 pulse_param 字段,frontend 通用机制,待拍板。
- **性能激进档需用户拍板**:迟滞 autoscale / 错峰重绘 / 嵌入面板低 dpi 性能模式(均改可见行为或视觉,见性能结论)。
- 其余 GUI 待决方向(见下"暂缓:GUI 相关")。
- **✅ Pulse-scan 解耦重设计(已完成)**:`PulseScanNode` 只拥有 sequencer 与外部 y cursor。`sweep_kind=scan_slot` 时完整硬件表只 prepare/fire 一次；`sweep_kind=api_slot` 时 program 每行解析 API handles 并运行一个有限 pulse。表单唯一值为 `{program_id, api, sweep_kind, program}`；scan 坐标用 `ScanSlot.name`，API 坐标用 semantic target，`sN/aN` 都不泄漏。两条路径都从另一个 producer 消费下一条 lineage-coherent typed 更新，迟到/缺失即安全中止；节点不拥有相机、不 relay frame。`SignalExpr` 是 y 表达式单源。

- **🔬 段描述符 DAC 延迟重设计(RTL/host/测试/文档已落地,待上机 — 用户 /goal 要求)**:根因=旧 `g_busdly` 逐 bit 事件调度器逐 tick 采样 post-Bresenham 输出、每 bit 变化推一个事件,所以一条密集 ramp 被延时后在途 = 每帧值变化数 × ⌈d/帧⌉(pulse_test +1s ≈ 25050 ≫ `BUS_EVT_DEPTH`=64,被拒),而等价负延时经全局平移 G 折成净 0 直通(见 FPGA 手册第六章新增"正/负不对称"段)。**定案架构**=逐**总线**"段描述符"延迟替换逐 bit 值-变化 FIFO:主段播放器每应用一个段时捕获**已解析**的段 `{emit=apply_tick+d, vstart, target, span, step/rem, mode}` 推浅 per-bus FIFO,延迟重播器到点重跑 Bresenham,首个前 SAFE(首帧正确)、done 后跑 d(done-tail)。深度=在途**段数**(pulse_test +1s=**100** vs 25050),密度无关,逐总线(4)非逐 bit(40)→省 LUT。**否决** skip 计数器(loop_end 带 scan 系数逐点变、无单一周期,`d mod 周期` 会 first-frame 错=回归 e3fb639)与"重跑生成器 at t−d"(#ramp-carry + scan slot 历史缓冲太险)。TTL 保留事件 FIFO。**P1 完成**:`fpga/pulse_streamer/host/engine_model.py` 加 `bus_play(apply_log/carry_out)`+`bus_undelayed_and_log`+`rtl_bus_segment_delay_mirror`+`_segment_replay_step`,`tests/test_bus_segment_delay_equivalence.py` **60 例字节精确**对拍不变 reference `bus_delay_line_reference`(含 dense/scan/repeat/finite/done-tail/帧边界)。**已落地(待上机)**:D3 RTL(`zlc_edge_streamer.v` `g_busseg` 逐总线段延迟 + 延迟重播器 + done-tail SAFE-hold)、D4 host 段计数容量界 + 打包 + 清 value-change 残余、D5 测试/文档/记忆均已 commit(见 [[dac-segment-delay-rtl-landed-2026-07-08]])。真机报"加 delay 后 DAC 变常数"=两个 RTL-only NBA 陈旧 bug(fend freeze 塌 + tick0 陈旧 `del_bus_ticks`),已修(51cf378)。**LUT 装不下 → A+B(不是 A+C)**:含 g_busseg 的 build 在 `evt_fifo_depth`=128 / `bus_seg_addr_width`=6 下布局失败(21958/20800 Slice LUT)。**先试 A+C 撞大坑**:`bus_seg_addr_width`(C)不是省 LUT 旋钮而是**寄存器映射/ABI 参数**——改它把 R_DELAY 区基址前移 896 words,host 按新映射打包但真机跑旧 bitstream→**连不加 delay 的 scan 都错**(用户报"乱修")。**发布配置=A+B**(均寄存器映射无关的内部 FIFO 深度):`evt_fifo_depth`=64(TTL 事件 LUTRAM 减半 ~-1.2k)+ `bus_evt_fifo_depth`=32(逐总线段 FIFO 减半 ~-0.6k),`bus_seg_addr_width` **保持 6** → 估 ~20.2k(余 ~0.6k),待用户 rebuild + 上机验证。铁律:**ABI/寄存器映射参数绝不为省资源去动;LUT 只从寄存器映射无关的内部深度省**。

- **🔒 几何指纹握手 = host↔bitstream 兼容根修(已落地,待上机 — 用户 /goal「找本源、高层设计、系统性修同类」)**:本源=旧 `REGISTER_LAYOUT_ID`(静态 0x5A4C4C02,CTRL word 63)只版本化寄存器**结构**、不覆盖决定实际地址的**几何**,所以 config↔bitstream 任一几何参数漂移都**静默损坏**(上面 bus_seg 那次正踩中)。修=`image.build_fingerprint(params)`=hash(`LAYOUT_STRUCT_VERSION` + 所有 bitstream-affecting StreamerParams 字段;高字节 0x5A 永不 0、自识别),build 作 `LAYOUT_FINGERPRINT` generic 驱到 word 63,`axi_session.check_register_layout` connect 时比对 `build_fingerprint(self.params)`,不符**硬报错"rebuild"**(旧 bitstream 返回旧值/0→被抓)。DRY:哈希只在 image.py 一处,RTL 只携带 build 期算好的值。契约测试:`build_fingerprint_covers_geometry`(逐字段 bump 断言几何字段必变/host-only 不变)、`all_geometry_params_config_matches_rtl_defaults`(**全**几何 config==top.v==engine.v .v 默认 + NUM_DELAY_CH,补指纹"能撒谎"盲区:非-generic 参数的 .v 默认可与 config 漂移而指纹是从 config 算的看不见)、`test_final_top_regions_match_image` 扩成钉**全 21** CtrlWords 偏移(原只 6,CLK_ENABLE-46→20 garbled-strobe 类)。**geometry-drift-audit workflow(5 agent)**证实并已修上述 + fallback 字面量 128/64→64/32。**待办同类**(latent/别子系统,详见 [[geometry-fingerprint-handshake-2026-07-09]]):coeff_frac_bits 编译器硬编码 8(config 惰性=假单源,`affine_coeffs`/`RuntimeSequenceProgram` 应读 config)、SCAN_ADDR_WIDTH/BUS_INDEX_WIDTH 理想应 RTL `$clog2` 派生(现由契约测试守漂移)、bus_evt 5-way host 常量测试、busimg BRAM Write_Depth_A 硬 2048 未由 bus_rows 派生、clock_hz build-time 守卫(哈希观测不到真实晶振)、`PulseTableState.from_dict` 不校验 version、UART FRAME_WORDS wrap(xsim-gated,上机前修)、saved-figure .npz 无 schema/version、IMAGE_MAGIC/STATUS_ERROR 死握手。本机无 xsim,RTL 由 Python 镜像 + 逐寄存器自审 + 对抗审查锁定。

- **🧩 几何唯一真相源 = 一算两投影(已落地 f9d39ab+7b396c7,待上机 — 用户 /goal「改 config 后 verilog/python 全自动带入、别散落;系统 overlap/越界/overflow」)**:本源=config 名义单源但值散 6+ 份手抄 + 派生量硬写字面量,config 一大静默溢出。根修=`image.py` 唯一计算源投影成 **`zlc_geometry.vh`**(`emit_geometry_vh`:RTL 全参数默认 + `LAYOUT_FINGERPRINT` 做 `` `define `` 宏,top/engine/tb `` `include ``,**删整套 `-generic`**)+ **`geom.tcl`**(`emit_geom_tcl` 经 `build_ip_sizes`:busimg/axi_bram/portb 尺寸**派生** pow2,灭 2048/65536 字面量溢出);`check_rtl_assumptions` = overflow 单门,扩 **BUS_COUNTS `bus_count*(bus_seg_addr_width+1)<=32`**(指纹看不见的溢出)+ **pow2 evt/bus_evt** + 32b delay-cap;`StreamerParams` 默认 import 读 config(`_geom`);`coeff_frac_bits`+`slot_mul_width` 走无依赖 seam `Zou_lab_control/_streamer_geometry.py`(仿 `_clock`)灌进 timing/sequencer 编译器 + engine_model;`delay_region_words`/`ttl_delay_max_ticks` 补进 JSON。**对默认 config 字节等价**(指纹仍 `0x5A87FD36`,在跑 bitstream 照连、**免重建**),仅未来改 config 生效。侦察 + 对抗验证双 workflow;契约测试重写 + 新增(325 绿)。详见 [[geometry-single-source-header-2026-07-09]]。**这解决了 §36 待办里的**:coeff_frac_bits 假单源 ✅、SCAN_ADDR_WIDTH/BUS_INDEX_WIDTH 现走 header 宏(不再靠测试守漂移)✅、busimg Write_Depth 现派生 ✅、bus_evt 单源(header + 契约测试)✅、run_server clock/channel fallback 镜像已删除（只读 XDC + streamer config）✅。**对抗验证 workflow(3 agent,无 xsim)**证默认 config 字节等价、抓到并已修 1 个 MEDIUM——`NUM_DELAY_CH`/`DELAY_CH_MAP` 原是手写字面量+发了 `ZLC_NUM_DELAY_CH` 宏却没用=同 BUS_COUNTS 类盲区(改 channel_count 指纹变但 map 停 18-slot、孤立新通道且握手照绿),**已由构造闭合**:top 从 header 派生 count/idx_w + `zlc_delay_identity_map` 常量函数,与旧 `{17..0}` 逐 bit 等价(测试钉);另 emit_geom_tcl 也调 check_rtl_assumptions、REGISTER_LAYOUT_ID 改规范路径、create_project 加 include_dirs、.gitattributes 钉 .vh eol。**仍剩(小项/低优先)**:pulse_table.`DELAY_MAX_TICKS`/engine_model 延时帽仍各拼 `(1<<31)-1`(第 3 份,可再单源化)、`bus_delay_line_reference` 的 `safe_value=512`/`bus_width=10` 函数默认(调用方都传真值)、create_project.tcl 头注释硬写派生数(纯注释会随 config 陈旧);另 clock_hz build-time 守卫 / UART FRAME_WORDS wrap / IMAGE_MAGIC 死握手仍属别子系统。

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

---

## 交接重构(3f2e049,~17k 行)残余工作 — 精确根因 + 可执行计划

> 上下文:另一 agent 的 typed signal-tensor / port-catalog / fitting 重构整体 `sound-with-cleanups`(7 子系统 + core 评审;高层设计合理不该重写)。已修并提交:#1 1D external-datum 重载根因(b978467)· 过时 info 信封夹具迁移(9ab2e49)· core 死代码 snapshot_at_version/point_data(9c87d6b)· #6 ROI 门单源化(6618caf)。#2/#5/#7-crash/#8/#9 早于本轮已在 3f2e049/741e2c5 落地。以下是**尚未落地**的,按"能否 headless 验证"排序。

### A. 需真 GUI 逐帧验证(headless 已排除多路,不盲改)
- **#4 cbar 颜色柱某些情况没填满框**:headless 四条路径(静态 / clim 拖拽 / cmap 切换 / live update)**全部 100% 填满**——`image.set_clim` 会同步 cax ylim + solids QuadMesh 几何;`DragHLine.on_motion`(selectors.py:536)走 `draw_idle()` 全画会重建 colorbar。故只可能出现在**嵌入式 Qt canvas 的 offscreen-Agg / FRONT-QImage stretch-blit 前缓冲路径**(`qt_canvas` 的 `_zlc_embedded` present/paint),需真 console 拖 clim / 切 cmap 逐帧截图定位是否 FRONT 快照漏刷 colorbar 区域。
- **#6 余项 hist 无框选 fit/crop**:fit 对 1d/2d/sites/monitor/grid 已在;`_kind_offers_general_fit('hist')=False` 是**有意**(hist 的 fit=自带 bimodal 旋钮 none/single/double,不叠通用峰模型)。真缺口=`PanelCard._build_settings` 的 selection-action 组合框整段在 `if models:` 内(task_console.py:2049),hist(models=[])拿不到"框选→fit/crop"。修法:把组合框抬出 `if models:`;action=="fit" 时对 hist 路由到 bimodal 而非通用模型(`_apply_fit_selection` @7667 需按 kind 分派)。需真 hist 面板验证框选→bimodal 联动。

### B. 可 headless / 契约测试验证(下一轮硬骨头)
- **#7-secondary scan_table 载入覆盖损坏**:pulse_gui 生成缓存空时 load 覆盖损坏;复现存档链(`_current_scan_table`/snap)后根修 + 往返契约测试。
- **shape 词汇双源(最大 DRY 收益)**:transport 用 `SignalSchema.point_shape`(单,signal_tensor.py:110),node/GUI 用 `SignalSpec.points_shape`(复,logic.py:113)+ stringly `structure` dict 横跨 ~199 处/27 文件;shape 派生数学在 `describe_shape`(logic.py)/`coerce_panel_value`(live.py:2311)/`SignalSchema` 各重打一遍。收敛成一套词汇 + 一个 `physical_shape` 校验,全量契约测试守。**大范围多文件,独立一轮。**
- **canonical shape 校验四处重打**:task_console._validate_canonical_block(3378)/data_figure._validate_signals(179)/figure_viewer(258)/live.coerce grid(2322)→ 复用 core `signal_tensor.physical_shape`。
- **schema-lifecycle 三处判定收敛**:register_signal(signals.py:267)/publish install loop(463)/logic._register_output_schemas(363),提 hub-internal `_install_schema_locked` + 把"only installer may replace"策略移进 hub(可让 node 丢掉 `_inherit_output_schema_ownership`)。
- **pulse_table.from_dict 手搓反序列化**(pulse_table.py:2016-2038)→ 复用 timing/sequence.py 已用的 `require_*` helper 集。
- **DAC 有符号范围三源**:PortSpec.signed_range(ports.py:97)/bus_signed_range(pulse_table.py:57)/bus_signed_bounds(live.py:2509)→ 收敛到 topology 对象那一个。

### C. 收尾
- 上述落地后一次性更新 PDF 手册(frontend/fpga/main) + tutorial,清历史残余,补关键概念示意图。
