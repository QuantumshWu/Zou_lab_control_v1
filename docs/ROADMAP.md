# ROADMAP / 当前焦点 / 待决方向

> 这里记**现在在做什么、需求/约束、暂缓的事、要拍板的决定**,以及一些**下一阶段值得采纳的设计想法**。
> 工作守则见 `AGENTS.md`;子系统深档见 `docs/MAINTAINER_NOTES.md`;Task 控制台设计见 `docs/task_console_design/`。

## 当前焦点(2026-06)

notebook 调用侧的解耦 + Rb87 读出**已落地**(见下"已完成");下一步是真机 qCMOS 接线与 GUI 回归。

**已完成(2026-06)**
1. **解耦**:`neutral_atom` 不再 import `frontend`(IoC viewer 注册表 `_viewer_registry`,双向 import 期互不拉对方;实验层可 headless 导入)。
2. **子系统拥有逻辑**:读出编排(sitemap/thresholds/detect/detection-time)从 `session` 上帝对象搬进 `ReadoutSubsystem`,session 退成门面;签名明确不再 `**kwargs` 转发。
3. **Rb87 读出接入**:PSF 匹配滤波提取(`core/psf.py`)+ 双高斯定阈/保真度(`core/bimodal.py`),`TrapCalibration` 加 `method='box'|'psf'` 经 `signals()`/`detect()` 单点分派(box 仍默认);`readout.sitemap(method="psf")` / `thresholds(method="bimodal")`;虚拟后端端到端可测。**只移植算法,不带 rb_qcmos 的文件IO/缓存/批处理脚手架**。

**下一步(待做)**
- 真机 qCMOS 相机后端(`devices/qcmos.py`)接 PSF/bimodal 读出,在真实数据上验证保真度;4-shot group / 参考帧定 ground-truth 标签作为可选标定流程(算法已具备,缺采集编排)。
- 真机 feed 类(`AtomLoadingFeed`)从 `NeutralAtomSession` 采集循环 publish(骨架见设计文档)。
- 回头调 GUI(见下"暂缓:GUI 相关")。

> `references/` 是历史源码归档(`rb87_readout_v16`、confocal GUI 等),**git ignore、只在本地存在**,是借鉴/移植的来源,不是被本仓库 import 的依赖(见 README 目录树)。
> **最小跑通入口**:`task_console.bat`(虚拟 feed 仪表盘)/ `tutorials/` 里的 notebook / `na.connect("virtual")` 起一个全虚拟 session——先在虚拟后端把调用链串通,再换真机后端。

## 长期约束(贯穿,见 AGENTS.md)
notebook-first;子模块只经接口互联(解耦);无后向兼容;前端密封;所有可视化三档 DPR 截图验收;全中文;真机正确性根因在自己代码。

---

## 下一阶段值得采纳的设计想法(灵感来自 confocal GUI)

> **这些只是想法库,不是必须照搬的规范。** 要遵守的是**设计原则**(解耦 / 只经接口互联 / 单一真相源 / 虚拟可替换 / 显式优于隐式),**不是 confocal 的具体实现**。哪条有更好的思路就用更好的——confocal 只是一个被认可的参考实现,不是模板。
>
> (来源 `references/.../Confocal_GUIv2_refactored_v6`,git-ignore 历史归档。)

1. **声明式实验/测量元数据**:一个测量类用装饰器(confocal 的 `@measurement_gui_meta`)或 `caller()` 签名**自带它的参数与分类**(context / 要保存的 config / 设备槽 device-overrides / 额外参数),单位和默认值就在签名里。借鉴:让每个"实验任务"自带参数接口,notebook/GUI 从签名/schema 生成交互面板,不手写。(我们的 frontend `ParamSpec` 是这个思想在面板层的局部实现,可向实验层推广。)
2. **设备契约 + 注册表**:设备通过抽象基类定义契约(`BaseCounter`/`BaseLaser`…),上层不绑死硬件;按名字注册/取设备(device manager)。我们已有 `devices/base.py` 三契约 + registry,继续沿用并补齐。
3. **虚拟设备信号注册表 + 物理 dataclass**:`@VirtualCounter.register_signal('ple')` 按测量名查表产期望计数;物理常数集中在 dataclass(峰位/宽/衰减/漂移)。借鉴:为 Rb87 建 `@dataclass VirtualAtom`(能级/亮暗率/loss/drift),按实验名注册信号模型——**加新实验不用改相机/虚拟设备代码**。我们的 `VirtualLoadingFeed` 是雏形,可泛化成注册表。
4. **measurement 生命周期 + 线程分工**:worker daemon 线程跑 `_loop()` 只产数据,前端定时器/notebook 轮询只读;matplotlib artist 由一处统一管;update 策略用**显式 dispatch 字典**(add/replace/create/roll)而非动态 getattr。我们的 Feed/SignalHub 已是这套(生产者线程 + hub 拷贝 + GUI 读副本),继续保持。
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

## 暂缓:真机 feed 类与派生量(notebook 侧定稿后做)
- 入库一个从 `NeutralAtomSession` 采集循环 publish 的 `AtomLoadingFeed`(骨架见设计文档),让真机监控开箱即用。
- readout 提供"逐 shot 派生量"helper(保真度/温度),让 `atom_temp_monitor`/`fidelity_monitor` 类任务开箱即用。

---

## 要你拍板的决定(开放)
- root `AGENTS.md` 之外是否还要单独 `CLAUDE.md`?(目前合并在 AGENTS。)
- 常犯错误目录的"自动记录"是否要加 Stop hook 提醒?(目前靠版本化目录 + MEMORY 指针 + 收尾自觉追加。)
- notebook 实验任务是否采用 confocal 式声明式装饰器(想法 1),还是更轻的约定?
