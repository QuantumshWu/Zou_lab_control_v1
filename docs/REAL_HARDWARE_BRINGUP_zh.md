# 实机上线 checklist(真 FPGA + 真 qCMOS)

> 核心原则：virtual 与 real 共用 declarative Request、typed Port、RunController 和 artifact
> 语义；GUI 不接 raw device。当前已发布的真实入口仅是 **pulse-only remote installation**，
> 完整 qCMOS + sequencer installation 仍为 NO-GO，不能用旧 config/session 绕过这一边界。
> 本清单只把已经交付的能力写成可执行步骤。

> ⚠️ 运行前确认 import 的是这份代码(`python -c "import Zou_lab_control, sys; print(Zou_lab_control.__file__)"`),
> 别误跑到机器上另一份旧 checkout。

---

## 0. 前置环境(到机器前先备齐)

### FPGA 端(运行 sequencer server 的那台)
- [ ] Vivado 已装,`vivado` 在 PATH(或设 `ZLC_PS_VIVADO_BIN`);`hw_server` 能起。
- [ ] JTAG 线连好、板子上电;Vivado 硬件管理器能单独看到目标。
- [ ] **bitstream 已 program**,且其 `ZLC_LAYOUT_ID` 与主机 `image.REGISTER_LAYOUT_ID` 一致。
      不一致时第一次 `prepare()` 会**在写任何配置寄存器前**明确报 `geometry/layout mismatch`(这是设计的保护,
      不是 bug；current owner 在 `zlc_pulse/transport/session.py` 通过
      `image.build_fingerprint`/geometry handshake 校验)。冻结 RTL/bitstream 不因软件架构偏好重烧；
      只有证实现有 RTL bug 或偏离既定设计才进入独立硬件变更流程。
- [ ] 启动 `fpga\run_server.bat`(`jtag-axi` 后端);确认启动摘要同时列出当前
      `ZLC_PS_TARGET`与**server-side** `ZLC_PS_XDC`，并通过target/XDC逐lane校验后监听端口(默认18861)。

### 主机端(跑 notebook / GUI 的那台)
- [ ] 安装 hardware/workbench extra（其中包含 current pulse RPC 所需的 `rpyc`）。
- [ ] Hamamatsu DCAM SDK 装好,`dcamapi.dll` 在 PATH;qCMOS 物理连接、`device_index` 对。
- [ ] 网络能 ping 通 FPGA 端 IP;防火墙放行 server 端口(18861)。

### 当前可上线边界
- [ ] Pulse-only real composition 不读取旧 `remote_template.json`或客户端XDC，显式使用server的`host:port`；
      target manifest（含package-pin endpoints）、clock、geometry与connection generation全从current server snapshot取得并在每次Run重验。
- [ ] 完整 qCMOS + sequencer real installation 尚未闭合，不能把 pulse server 连通冒充相机也已可用。
      相机 bring-up 继续先做独立contract qualification；不得恢复旧 `RemoteSequencer/QCMOSCamera` raw session 绕过runtime。

---

## 1. pulse GUI(脉冲编辑 + 触发硬件)

打开后可直接在窗口里选连接：

```bash
python pulse_gui.py            # 默认 Offline，可编辑/Preview但执行按钮禁用
```

- 顶部 **Connection**:下拉选 `Remote server` → 填 `host:port`(默认 `127.0.0.1:18861`)→ 点 **Connect**。
  成功后该process-lifetime installation不可热换；要换server必须安全关闭窗口后启动新进程。
- 也可启动即连(脚本/无人值守):`python pulse_gui.py --remote-host <FPGA_IP>`(显式 host 视为必须连,
  连接失败会在同一窗口明确显示，修正地址后可重试；尚未取得installation authority前绝不调用硬件)。
- 连上后：**Run Once** 编译整段 `PulseDocument`、上传并执行一次；**On Pulse (HOLD)** 在FPGA侧持续；
  **Run Scan** 执行冻结的无缝自主scan table；**Stop** 经RunHandle cancel、远端interrupt SAFE与安全验证收尾。
- notebook使用同一入口：

  ```python
  from pathlib import Path
  import Zou_lab_control.notebook as zlc

  installation = zlc.InstallationConfigDocument.remote_pulse(
      host="<FPGA_IP>",
      port=18861,
  )
  repository = Path("results") / "pulse-only"
  repository.parent.mkdir(exist_ok=True)
  exp = zlc.connect(installation, repository=repository)
  exp.pulse_gui()
  ```

  该Experiment只有sequencer能力，没有camera/readout能力。窗口复用该Experiment，关闭窗口不关闭notebook中的Experiment；standalone窗口则拥有并在关闭时安全关闭自己的Experiment。

---

## 2. task console(实时看板)

当前没有可发布的完整 qCMOS real installation，因此这里**没有**合法的真机 TaskConsole、
calibration 或 occupancy 启动命令。旧 `remote_template.json`、`open_devices=True`、
`task_console.py --config/--grid` 与 raw `RemoteSequencer/QCMOSCamera` 路径均不是 current 产品入口，
不得用于首光。相机 AssetMap、adapter qualification、SAFE/terminal evidence 与 pulse/camera
同一 installation composition 闭合后，本节才增加真实的人类操作步骤；在此之前只验收第 1 节的
pulse-only 路径。

---

## 3. 首次上电逐步验证

1. FPGA端起`run_server.bat`；主机打开Pulse GUI，选Remote并连接。状态必须显示READY，
   Target tab必须只读，Edit左列必须显示server XDC发布的package pin（如`F15`）而非`ch00`；否则不运行。
2. 先用 **Run Once** 跑一个全safe短pulse，再跑一个单通道短pulse；示波器确认波形与编译Preview一致。
3. 用 **On Pulse (HOLD)** 验证持续输出，再点 **Stop**；只有窗口显示STOPPED/SAFE且server snapshot为SAFE才继续。
4. 冻结一张小scan table，用 **Run Scan** 验证整表由FPGA自主无缝运行；不得改成host逐点fire-and-wait。
5. qCMOS、calibration、occupancy必须等完整real installation与相机qualification闭合后再按独立清单验收；
   当前pulse-only成功不等于这些流程已READY。

---

## 4. 最易当场报错的点(对照表)

| 现象 | 根因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: ...dcam` / `failed to open qCMOS` | DCAM SDK / DLL 缺失或相机没连 | 装 SDK、确认 `dcamapi.dll` 在 PATH、`device_index` 对 |
| `ConnectionRefused` / `socket.timeout` | server 没起 / IP 端口错 / 防火墙 | 先起 `run_server.bat`;核对Pulse GUI的host:port;放行端口 |
| 首次 `prepare()` 报 `geometry/layout mismatch` | 运行image的几何指纹与current host `build_fingerprint(params)`不一致 | 停止运行并核对已批准的软件/bitstream资产；不得为迁就架构自动重烧，只有证实现有RTL bug或偏离既定设计才启动bitstream变更流程 |
| server 起不来 / JTAG 报错 | hw_server 没起 / JTAG 接触 / 板掉电 | 查电源、JTAG 线;Vivado 硬件管理器单独验证 |
| `qCMOS timed out` 等不到帧 | 相机收不到触发(通道/触发名不匹配) | 核对 XDC 的 `channels` 与相机 config 的 `capture_trigger_channels`;示波器看触发线 |

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
禁止 notebook、GUI、server handler 或另一线程并行轮询状态或写 bank。

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
| S1 | 分别运行只占一个 bank、恰跨 bank、超过两个 bank 的 finite 表 | 每次只有一个 FIRE；point/camera 总数、ordered schedule、cursor terminal、producer-owned SignalAssociationEvidence 与 collector coverage 完全一致 |
| S2 | 运行奇数与偶数 chunk 数，并以 9999 点表做 streamed 压力检查 | bank/chunk 顺序与冻结 table digest 一致；无 observed UNDERFLOW、ERROR、错序、重复或遗漏 |
| S3 | 运行 cyclic 表并在多个 wrap 后取消 | wrap 顺序保持冻结映射；cancel 由同一 observer 收口并进入 SAFE，不遗留第二个 I/O worker |
| S4 | 在测试环境故意把一次 refill 延迟到 bank 边界附近 | observer 必须由 UNDERFLOW 或 refill 后的 cursor 跨 chunk 证明拒绝该 run，随后 SAFE 成功；不得以稍后的 RUNNING/DONE 判成功 |
| S5 | 正常完成后核对 server snapshot、Run diagnostics 与 producer 证据 | terminal、完整 schedule、producer association、collector coverage 和 camera drain/tail（若适用）互相一致 |

任一检查失败时保留 snapshot、diagnostics、table/chunk digest 和仪器证据，停止该 run 并排查
transport/installation；不要改写状态、不要切到 host stepping，也不要把重烧 bitstream 当作常规
恢复步骤。只有证实现有 RTL 偏离已批准设计时，才另行启动 bitstream 变更流程。
