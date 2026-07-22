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
  exp = connect("remote", repository=repo,
                sequencer_host="<FPGA_IP>", sequencer_port=18861)
  exp.pulse_gui()
  ```

  这里窗口复用该Experiment，关闭窗口不关闭notebook中的Experiment；standalone窗口则拥有并在关闭时安全关闭自己的Experiment。

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

## 5. >4096 点扫描(9999 点级):`AUTONOMOUS_REFILLED` 资格化

本节是完整 qCMOS real installation、Q0 与 EndAttestation 闭合后的**未来资格化**，不是当前
pulse-only 首测步骤，也不改变≤4096点 `AUTONOMOUS_RESIDENT` 的当前可用路径。

### 5.1 现在会看到什么

超过常驻窗口的扫描在 **fire 之前**被 typed 拒绝,不会退化成 host 逐点驱动:

```
zlc_pulse.FormalScanCapacityExceeded:
  formal autonomous scan exceeds the frozen bitstream's fully resident
  capacity: 9999 points > 4096. AUTONOMOUS_REFILLED is not published ...
```

异常对象带三个字段,GUI/notebook 直接读,不用抠字符串:

| 字段 | 含义 |
|---|---|
| `requested_points` | 本次请求的物理 scan 行数(已含 repeat 展开) |
| `resident_limit` | `2 * bank_size`,当前 `fpga/board_config/streamer_config.json` 里 `bank_size=2048` → **4096** |
| `capability_unavailable_reason` | 缺哪几项证据(单源常量 `AUTONOMOUS_REFILLED_UNAVAILABLE_REASON`) |

**这不是"硬件做不到"。** 冻结 bitstream 里 ping-pong bank refill 硬件本来就在,
`zlc_edge_streamer.v` 的流式 scan 与无缝 wrap 都已验证过。缺的是**证据**,
不是硅片,所以本节是一份资格化实验清单,不是硬件改动申请。

### 5.2 必须先成立的三件事(§15.4 强 gate)

1. **单一 I/O owner。** 一个 `FiniteScanStreamer` 同时负责 status、cursor、bank refill、
   progress、cancel、completion。今天 `zlc_pulse/transport/session.py` 的 worker 只读
   STATUS/CURSOR,没有 refill 写方;绝不允许再开第二个线程去写同一 transport。
2. **refill 事务的保守硬上界。** 要的是上界,不是 measured worst / p99。
   "measured worst refill + Windows/Python 调度余量"**不是**确定性上界,不可用来发布能力。
3. **每个 seam 的硬件时间观测 + 全 schedule residual。**
   当前 RTL 的 `STATUS_UNDERFLOW` 在 bank 恢复后会**自己清零**(非 sticky),
   所以最终 `STATUS_DONE`、局部 camera timestamp、`scan_progress()` 镜像
   **都不能**证明"从未 stall"。没有 camera edge 的区段、最后一个 trigger 之后的 seam、
   任何不可观测的 stall,只要有一个,能力就不可发布。

### 5.3 资格化实验(按顺序;每步不过就停在这一步)

| # | 实验 | 命令/配置 | 判据(全部满足才算过) |
|---|---|---|---|
| R1 | 常驻基线复核 | 4096 点 SCAN_SLOT,走现有 `AUTONOMOUS_RESIDENT` | 一次 fire 跑完;terminal evidence 合法;camera 帧数 == `expected_trigger_total_from_completed_schedule` |
| R2 | refill 事务上界测量 | 单 I/O owner 下,反复写满一个 bank,记录每次事务耗时分布与**理论上界推导** | 有书面上界(transport 字节数 × 最坏 per-word 时间 + 协议开销),且实测最大值 < 上界;只有分布没有推导 = 不过 |
| R3 | seam 时间观测分辨率 | 在每个 bank 边界安排camera trigger；使用经Q0资格化的qCMOS metadata、现有回读或外部仪器取得时间证据 | 每个潜在 seam 都落在一个可观测区间内；来源、分辨率和误差必须入证据；不得假定当前FPGA已有逐沿timestamp；**存在无edge的seam即判不过** |
| R4 | 全 schedule residual | R3 的逐 seam 观测与 compiled schedule 做残差比对 | 残差 ≤ 由 R2 上界推出的允许值;无未解释异常点 |
| R5 | 9999 点压力实验 | 9999 点 SCAN_SLOT,连续 ≥20 次 | 每次:无 underflow、无 late chunk、terminal 合法、帧数与 schedule 完全一致;**一次不过即整轮不过** |
| R6 | 拒绝路径反证 | 人为把 refill 延迟到超上界 | 必须 fail closed(拒绝/INVALID),不得"看起来跑完了" |

### 5.4 启用与回退

- **启用**:R1–R6 全过后，新增由deployment证据签发的typed
  `AUTONOMOUS_REFILLED` capability，并让 `validate_resident_scan_capacity` 与
  `require_autonomous_scan_resident_capacity` 共同消费这一唯一判据；同时保存R2–R4证据digest。
  `AUTONOMOUS_REFILLED_UNAVAILABLE_REASON` 只是拒绝诊断文本，修改它本身绝不能发布能力。
- **回退**:任何一次实验不过、或换 transport/换机器/改 `bank_size`,立即把常量改回未发布,
  实验退回 `AUTONOMOUS_RESIDENT`(≤4096 点)。分批扫描是合法的临时办法;
  **host 逐点驱动不是**。
- **不变量**:即使能力发布,host 也只**供应预先冻结的 chunk**,
  全部精密 edge 时序仍由 FPGA 自主决定;host 不选下一个 point、不调度 edge。
  单次 run 的 Formal 资格另由 Q0 qualification、association proof、exact 链与
  EndAttestation 独立决定,与装载方式名称无关。
