# 实机上线 checklist（remote FPGA + qCMOS DCAM + Pylon MOT camera）

> 核心原则：virtual 与 real 共用 declarative Request、typed Port、RunController 和 artifact
> 语义；GUI 不接 raw device。当前同时提供 sequencer-only `remote_pulse` 与完整 `hardware`
> graph template。每个 device leaf 独占自己的 schema、factory 与 capability；完整 graph 已闭合
> remote FPGA、qCMOS DCAM 和 Pylon MOT camera 的软件 composition，可以进入真实设备
> E0/bring-up；“软件入口存在”不等于“这台装置已经合格”。
> 每次完整 installation 初始化都必须在当前设备上通过主动 E0，失败时不发布 runtime。

> ⚠️ 运行前确认 import 的是这份代码(`python -c "import Zou_lab_control, sys; print(Zou_lab_control.__file__)"`),
> 别误跑到机器上另一份旧 checkout。

---

## 0. 前置环境(到机器前先备齐)

### FPGA 端(运行 sequencer server 的那台)
- [ ] Python 使用安装器记录的 `.zlc_python_path`，或显式设 `ZLC_FPGA_PYTHON`；Vivado 在 PATH
      （或设 `ZLC_PS_VIVADO_BIN`）；`hw_server` 能起。
- [ ] JTAG 线连好、板子上电;Vivado 硬件管理器能单独看到目标。
- [ ] **bitstream 已 program**,且其 `ZLC_LAYOUT_ID` 与主机 `image.REGISTER_LAYOUT_ID` 一致。
      不一致时第一次 `prepare()` 会**在写任何配置寄存器前**明确报 `geometry/layout mismatch`(这是设计的保护,
      不是 bug；current owner 在 `zlc_pulse/transport/session.py` 通过
      `image.build_fingerprint`/geometry handshake 校验)。冻结 RTL/bitstream 不因软件架构偏好重烧；
      只有证实现有 RTL bug 或偏离既定设计才进入独立硬件变更流程。
- [ ] 启动 `fpga\run_server.bat`(`jtag-axi` 后端);确认启动摘要同时列出当前
      `ZLC_PS_TARGET`与**server-side** `ZLC_PS_XDC`，并通过target/XDC逐lane校验后监听端口(默认18861)。

### 主机端(跑 notebook / GUI 的那台)
- [ ] 安装 hardware/workbench extra（包含 current pulse RPC 的 `rpyc` 与 `pypylon`；DCAM SDK
      仍由相机厂商安装）。
- [ ] Hamamatsu DCAM SDK 装好,`dcamapi.dll` 在 PATH;qCMOS 物理连接、`device_index` 对。
- [ ] Basler pylon runtime 与 `pypylon` 装好；相机 serial、trigger source 和物理连接正确。
- [ ] 网络能 ping 通 FPGA 端 IP;防火墙放行 server 端口(18861)。

### 当前可上线边界
- [ ] `remote_pulse` real composition 不读取旧 `remote_template.json`或客户端XDC，显式使用server的`host:port`；
      target manifest（含package-pin endpoints）、clock、geometry与connection generation全从current server snapshot取得并在每次Run重验。
- [ ] 完整 `hardware` composition 由 DeviceManager 或
      `installation_template("hardware", ...)` 创建 ordered `DeviceInstance` graph；它会在发布 Experiment 前
      主动验证两个相机的工作点、Target endpoint、帧顺序/计数与 FPGA terminal evidence。
- [ ] Pulse server 连通只证明 sequencer transport。只有本次完整 initialization 的主动 E0 成功，
      才能说当前 connection generation 上的相机 trigger path 已取得运行期 qualification；不得恢复
      raw `RemoteSequencer/QCMOSCamera` session 或旧 config 绕过这一边界。

---

## 1. pulse GUI(脉冲编辑 + 触发硬件)

打开后可直接在窗口里选连接：

```bash
python pulse_gui.py            # 默认 Offline，可编辑/Preview但执行按钮禁用
```

- 顶部 **Connection**:下拉选 `Remote server` → 填 `host:port`(默认 `127.0.0.1:18861`)→ 点 **Connect**。
  standalone PulseGUI 只通过这条显式连接命令替换自己拥有的 connection；绑定完整 Experiment 时，
  topology 只能由 DeviceManager 的原子 **Apply** 替换，不能在旧 runtime 内偷换 Port。
- 也可启动即连(脚本/无人值守):`python pulse_gui.py --remote-host <FPGA_IP>`(显式 host 视为必须连,
  连接失败会在同一窗口明确显示，修正地址后可重试；尚未取得installation authority前绝不调用硬件)。
- 连上后：**Run Once** 编译整段 `PulseDocument`、上传并执行一次；**On Pulse (HOLD)** 在FPGA侧持续；
  **Run Scan** 执行冻结的无缝自主scan table；**Stop** 经RunHandle cancel、远端interrupt SAFE与安全验证收尾。
- 脚本/notebook 使用同一个 public API：

  ```python
  from pathlib import Path
  from Zou_lab_control.api import WorkspacePaths, connect, installation_template

  installation = installation_template(
      "remote_pulse",
      host="<FPGA_IP>",
      port=18861,
  )
  workspace = WorkspacePaths.for_workspace(Path.cwd())
  exp = connect(installation, workspace=workspace)
  exp.pulse_gui()
  ```

  该 Experiment 只有 sequencer 能力，没有 camera/readout 能力。窗口复用该 Experiment，
  关闭窗口不关闭调用者持有的 Experiment；standalone 窗口则拥有并在关闭时安全关闭自己的
  Experiment。`remote_pulse` 只是创建 ordered graph 的模板名，不是 runtime backend dispatch；
  连接时仍由 graph 中 `sequencer` stable instance 的 leaf factory 发布 `pulse.execute` capability。

---

## 2. 完整 hardware installation（首选 DeviceManager）

先运行 `device_manager.bat`，在 **New** 选择 `hardware`，逐张 device card 填写 pulse server、
qCMOS、Pylon 的真实硬件工作点，保存 config 后点 **Apply**。Device Manager 只保存设备身份和
硬件工作点；FPGA trigger endpoint 由 pulse server 发布的 Target manifest 按语义 label 解析，
不在 camera card 中复制 raw lane。lattice 的 grid rows/columns 属于 Calibration Task，site centers
是校准输出，绝不写入 camera device 配置。Apply 是唯一真实
bring-up 边界：它先连接 remote FPGA，再建立两个相机 adapter，读取并冻结 working point，随后
分别运行一段只切换目标 Target endpoint、其余数字/DAC 保持 SAFE 的四触发 E0 program。只有相机帧
ordinal、hardware stamp、produced count、terminal drain 与 FPGA completed schedule 全部一致，才
发布可供 TaskConsole/PulseGUI 共用的同一个 Experiment。任一步失败都会清理已打开设备，不发布
部分 runtime。

也可以显式构造同一个 ordered graph。模板只负责当前默认实例与不歧义的便捷覆盖；同名但属于
不同设备的字段（例如两个 camera 的 `exposure_seconds`/ROI）必须按 stable instance id 分别修改，
不能重新合并成 backend-wide 参数袋：

```python
from dataclasses import replace
from pathlib import Path
from Zou_lab_control.api import (
    InstallationConfigDocument,
    WorkspacePaths,
    connect,
    installation_template,
)

template = installation_template(
    "hardware",
    host="<FPGA_IP>",
    serial="<BASLER_SERIAL>",
)
per_instance = {
    "camera": {
        # 按真实 qCMOS 工作点设置 exposure/ROI；trigger endpoint 来自 Target manifest。
    },
    "mot-camera": {
        # 按真实 Pylon 工作点设置 exposure/ROI/trigger_source；trigger endpoint 来自 Target manifest。
    },
}
installation = InstallationConfigDocument(tuple(
    replace(
        device,
        parameters={
            **device.parameters,
            **per_instance.get(device.instance_id, {}),
        },
    )
    for device in template.devices
))
exp = connect(
    installation,
    workspace=WorkspacePaths.for_workspace(Path.cwd()),
)
exp.task_console()
```

Calibration Task 再单独填写 `grid_rows` 与 `grid_columns`。它从相机实际输出帧中发现并保存
site centers；如果已有独立的中心先验，可作为 Calibration 的显式 admission 输入，不能伪装成
相机硬件参数。

这里不再在 hardware installation 中填写 site centers；真机只需填写实际 camera 工作点，
Calibration Task 负责发现并保存中心。`camera`/`mot-camera` 的 `sequencer_ref="sequencer"` 引用的是 stable instance id；role 可以改名，
requirement 不随 role 漂移。若修改 instance id，必须同时更新所有引用该 id 的 leaf 参数，graph
preflight 会在连接任何设备前拒绝 missing/wrong-capability/cycle。旧 `remote_template.json`、
`open_devices=True`、raw SDK/session、backend-wide config 与已删除 constructor 都不是 current 入口。

---

## 3. 首次上电逐步验证

1. FPGA 端起 `run_server.bat`；先用 Pulse GUI 的 Remote 模式连接。状态必须显示 READY，
   Target tab 必须只读，Edit 左列必须显示 server XDC 发布的 package pin（如 `F15`）而非
   `ch00`；否则不运行。
2. 用 **Run Once** 依次跑全 SAFE 短 pulse 与一个单通道短 pulse；示波器确认实际波形、lane 与
   编译 Preview 一致。再用 **On Pulse (HOLD)** / **Stop** 验证 terminal SAFE。
3. 在 DeviceManager 载入完整 `hardware` graph 并点 **Apply**。此动作会在真实输出上主动
   运行两个四触发 E0；先确保 Target manifest 的 camera endpoint 已接好、其它输出的 SAFE 值正确。只有两个
   qualification 都成功且 DeviceManager 发布 active Experiment 才继续。
4. 在 TaskConsole 分别运行 qCMOS 与 MOT camera 的 monitor/finite measurement，确认 shape、dtype、
   frame ordinal、working point 与实际设备一致；这一步不得用 GUI 是否有图替代 terminal evidence。
5. 运行一个最小 Calibration → Occupancy 链，核对 site centers、validity、artifact identity 与原始
   qCMOS frame；再运行 MOT-field 路径确认 Pylon 数据来自同一 installation。算法结果必须由真实
   输入交叉验证，不能把 E0 成功外推成物理标定成功。
6. 冻结一张小 scan table，用 qCMOS signal 运行 Formal PulseScan。FPGA 必须一次 FIRE、自主无缝
   执行；FIRE 前 association boundary、FIRE 后 camera produced count/stamp、pulse terminal trigger
   count 与 collector coverage 必须完全对账。任何 gap/错序/多帧/少帧都使整 run INVALID，不提交
   ScanArtifact，也不得改成 host 逐点 fire-and-wait。
7. 通过小表后再按 §5 的单 bank、跨 bank、长表、cyclic/cancel 顺序扩大；每次保存 server
   snapshot、Run diagnostics、camera terminal 与示波器证据。只有这些真机证据通过，才把该装置
   标记为 qualified；软件包存在或 virtual 测试通过都不能代替这一步。

---

## 4. 最易当场报错的点(对照表)

| 现象 | 根因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: ...dcam` / `failed to open qCMOS` | DCAM SDK / DLL 缺失或相机没连 | 装 SDK、确认 `dcamapi.dll` 在 PATH、`device_index` 对 |
| `pypylon` 缺失 / Basler serial 找不到 | pylon runtime、Python binding、serial 或相机连接错误 | 安装匹配版本的 pylon/pypylon，使用设备工具核对 serial 与 trigger source |
| `ConnectionRefused` / `socket.timeout` | server 没起 / IP 端口错 / 防火墙 | 先起 `run_server.bat`;核对Pulse GUI的host:port;放行端口 |
| 首次 `prepare()` 报 `geometry/layout mismatch` | 运行image的几何指纹与current host `build_fingerprint(params)`不一致 | 停止运行并核对已批准的软件/bitstream资产；不得为迁就架构自动重烧，只有证实现有RTL bug或偏离既定设计才启动bitstream变更流程 |
| server 起不来 / JTAG 报错 | hw_server 没起 / JTAG 接触 / 板掉电 | 查电源、JTAG 线;Vivado 硬件管理器单独验证 |
| `qCMOS timed out` 等不到帧 | 相机收不到触发(Target endpoint/触发名不匹配) | 核对 server Target manifest 的 endpoint label 与 XDC/示波器实际接线 |
| Apply 在 E0 拒绝 stamp/count/terminal | 工作点不满足 deterministic trigger contract、发生漏帧/乱序，或 Target endpoint 接线错误 | 保留本次 pulse terminal、camera records 与示波器证据；先修实际布线/工作点/adapter，不绕过 qualification、不伪造 digest |

> 真机出问题记录current server snapshot、Run diagnostics、示波器/相机证据，并按
> `docs/MAINTAINER_NOTES.md` 的现行排查边界定位；不要依赖仓外memory key或旧session路径。

---

## 5. 当前 frozen RTL ping-pong streamed baseline

当前已批准 bitstream 本身就实现双 bank ping-pong scan；跨越两个 bank 的表与较短表走同一
`AUTONOMOUS_STREAMED` 路径，不需要重烧 FPGA 或另一项 capability。`prepare()` 在任何 I/O 前
验证 clock、target、geometry/ABI 与 deterministic wire packing；这些事实成立后，完整物理
scan table 会在 `FIRE` 前冻结，长度不改变 streamed 执行路径。

### 5.1 唯一 observer owner

`DeployedStreamerSession` 在 `FIRE` 后只允许一个 observer 拥有该 transport 的运行期 I/O。
它同时读取 STATUS/CURSOR、选择并补充下一个 bank、发布 progress、判定 terminal、处理 cancel，
以及在失败时进入 SAFE；`await_completion()` 只等待该 owner 的结果，不再创建第二个读写方。
禁止 public API client、GUI、server handler 或另一线程并行轮询状态或写 bank。

每个 refill 事务遵守同一个硬件握手：先将对应 `BANK_READY` 清零，再写该 bank 的全部 wire
words 与 `BANKx_CHUNK`，最后重新置位 `BANK_READY`。finite scan 按冻结 chunk 的单调序号装入；
cyclic scan 仍由同一序号选择 bank，并按冻结 chunk 数回绕数据。host 只供应预先冻结的 chunk，
不选择下一个 point、不逐点 fire、不调度任何精密 edge；point/edge 时序始终由 FPGA 决定。
每次重新置位后，同一个 observer 立即读取 `STATUS/CURSOR`；若已出现 error/underflow，或 cursor
已跨出 refill 开始时所在的 chunk，则不能证明该 bank 在边界前完成，整 run 必须失败。这个运行时
边界证明也覆盖“非 sticky underflow 在重新置位后迅速清零”的窗口。

### 5.2 整 run 的 fail-closed 判据

observer 只要在任一采样中看见 `STATUS_UNDERFLOW`，即使该位随后清零或最终又出现 `DONE`，
本次 run 也必须整体失败并进入 SAFE；不得把已取得的局部 frame、最终 cursor 或 DONE 改写成
成功证据。错误、cursor 越界、chunk/handshake 失败、超时、取消和断连同样由这个 owner 收口。
没有 host-stepped fallback，也不得通过拆成逐点 host 命令来掩盖 underflow。

### 5.3 真机 bring-up 检查

这些检查验证当前 installation 与单次 run：

| # | 检查 | 通过判据 |
|---|---|---|
| S1 | 分别运行只占一个 bank、恰跨 bank、超过两个 bank 的 finite 表 | 每次只有一个 FIRE；point/camera 总数、ordered schedule、single-use association finish、ordered EventRef/direct parents 与 collector coverage 完全一致 |
| S2 | 运行奇数与偶数 chunk 数，并以 9999 点表做 streamed 压力检查 | bank/chunk 顺序与冻结 table digest 一致；无 observed UNDERFLOW、ERROR、错序、重复或遗漏 |
| S3 | 运行 cyclic 表并在多个 wrap 后取消 | wrap 顺序保持冻结映射；cancel 由同一 observer 收口并进入 SAFE，不遗留第二个 I/O worker |
| S4 | 在测试环境故意把一次 refill 延迟到 bank 边界附近 | observer 必须由 UNDERFLOW 或 refill 后的 cursor 跨 chunk 证明拒绝该 run，随后 SAFE 成功；不得以稍后的 RUNNING/DONE 判成功 |
| S5 | 正常完成后核对 server snapshot、Run diagnostics 与 producer 证据 | terminal、完整 schedule、producer association、collector coverage 和 camera drain/tail（若适用）互相一致 |

任一检查失败时保留 snapshot、diagnostics、table/chunk digest 和仪器证据，停止该 run 并排查
transport/installation；不要改写状态、不要切到 host stepping，也不要把重烧 bitstream 当作常规
恢复步骤。只有证实现有 RTL 偏离已批准设计时，才另行启动 bitstream 变更流程。
