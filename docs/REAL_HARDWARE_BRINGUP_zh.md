# 实机上线 checklist(真 FPGA + 真 qCMOS)

> 核心原则:**虚拟 == 实机**。分析层(`core`/`operations`/`subsystems`/`session.py`)只碰
> 设备契约(`camera.acquire(...)` / `sequencer.prepare/fire/...`),从不 import 具体后端、不读仿真真值
> (由 `tests/test_virtual_equals_real_contract.py` 机械守卫)。所以**换真机只改 `na.connect()` 的 config**,
> GUI / 逻辑节点 / 读出 / 测量代码一字不改。本清单按"先确认、再上线"的顺序,尽量不在机器前踩坑。

> ⚠️ 运行前确认 import 的是这份代码(`python -c "import Zou_lab_control, sys; print(Zou_lab_control.__file__)"`),
> 别误跑到机器上另一份旧 checkout。

---

## 0. 前置环境(到机器前先备齐)

### FPGA 端(运行 sequencer server 的那台)
- [ ] Vivado 已装,`vivado` 在 PATH(或设 `ZLC_PS_VIVADO_BIN`);`hw_server` 能起。
- [ ] JTAG 线连好、板子上电;Vivado 硬件管理器能单独看到目标。
- [ ] **bitstream 已 program**,且其 `ZLC_LAYOUT_ID` 与主机 `image.REGISTER_LAYOUT_ID` 一致。
      不一致时第一次 `prepare()` 会**在写任何寄存器前**明确报 `register-layout mismatch`(这是设计的保护,
      不是 bug;见 `devices/axi_session.py` 的 `_check_layout`)。要改寄存器映射务必重 build + 重启 server。
- [ ] 启动 `fpga\run_server.bat`(`jtag-axi` 后端);确认监听端口(默认 18861)。

### 主机端(跑 notebook / GUI 的那台)
- [ ] `pip install rpyc`(RemoteSequencer 需要)。
- [ ] Hamamatsu DCAM SDK 装好,`dcamapi.dll` 在 PATH;qCMOS 物理连接、`device_index` 对。
- [ ] 网络能 ping 通 FPGA 端 IP;防火墙放行 server 端口(18861)。

### 配置文件
- [ ] `Zou_lab_control/neutral_atom/configs/remote_template.json` 按实际改:
      `camera`(QCMOSCamera:exposure / roi / device_index / timeout_ms、`capture_trigger_channels`)、
      `sequencer`(RemoteSequencer:**host / port** = FPGA 端 IP:端口)。端口拓扑只从 server 的
      `PortCatalog` 与 `clock_hz` 读取，control computer 不再维护第二份拓扑或时钟。
      序列器是纯脉冲流送器,不再有 `trigger_channels`:相机被哪条线触发是**相机**的属性
      (`camera.config.capture_trigger_channels`),由相机持有、向上暴露,序列器不感知。
- [ ] server 发布的 `PortCatalog` 与板子 XDC 一致;`capture_trigger_channels` 是相机外部触发接的那条线(模板里是 `ch11`)。
      成像默认用 `ch09=trap / ch00=cooling / ch03=probe` + 相机的 `capture_trigger_channels[0]`。
      这套映射在 `session.py:_imaging_channel_kwargs` 里(它从 `camera.capture_trigger_channels[0]`
      取触发通道喂给 `imaging_channel_kwargs`),若通道名变了要同步。

---

## 1. pulse GUI(脉冲编辑 + 触发硬件)

打开后**在窗口里选连接**(不必启动时指定):

```bash
python pulse_gui.py            # 默认 Virtual(仿真,可离线编辑/模拟 On Pulse)
```

- 底部 **Connection** 卡:下拉选 `Remote server` → 填 `host:port`(默认 `127.0.0.1:18861`)→ 点 **Connect**。
  状态行显示当前连接;选 `Offline (edit only)` 可断开只编辑。
- 也可启动即连(脚本/无人值守):`python pulse_gui.py --remote-host <FPGA_IP>`(显式 host 视为必须连,
  连不上会直接报错而非回退)。
- 连上后:**On Pulse** 编译 + 上传 + 运行;**Stop Pulse** 安全态;**Sync** 把设备上实际生效的脉冲拉回编辑器
  (notebook/裸 API 改过设备后用)。

---

## 2. task console(实时看板)

**首光必须先在 notebook 标定**(要肉眼看 loading 图 + 计数直方图,确认站点检测和阈值对):

```python
import Zou_lab_control.neutral_atom as na
exp = na.connect("remote_template.json", open_devices=True)   # 开 qCMOS + RemoteSequencer
exp.readout.sitemap(method="box", frames=20, display=True)     # 看站点中心检测
exp.readout.thresholds(frames=100, display=True)               # 看每站点阈值直方图
exp.readout.detect(display=True)                               # 单帧检查阈值极性与占据结果
```

当前 task console 可用于原始相机与通用 plot 的真机检查:

```bash
python task_console.py --config remote_template.json --grid 5x7
```

- 控制台开局**空、全停**。从 Add Panel 添加 camera Measurement，再把 2D/hist/monitor/1D
  plot 绑定到 `frame_0` 等原始信号即可检查实时采集；布局可 Save/Load。
- 正式 readout calibration/occupancy GUI 尚在迁移，Add Panel 不提供替代 detector。不要用
  demo、私有类或旧 PulseTableState workflow 绕过这一 NO-GO，也不要建立格式转换器。
- **无内置预设**——布局都是你自己拼好再 Save 出来的(`--task <你存的名字>` 载回)。
- 每个 logic 节点的参数表单在它**自己**的 Edit 标签(由 ParamDecl 自动生成);plot 面板的 Edit 也给产它信号
  的那个 measurement/processor 的参数表单。task 运行时占一张固定面板看中途过程、并锁定其他操作只留 Stop。

---

## 3. 首次上电逐步验证

1. FPGA 端起 `run_server.bat`;主机 `exp = na.connect("remote_template.json", open_devices=True)`
   应不报错;`exp.devices.snapshot()` 看相机 + sequencer 都连上。
2. pulse GUI 连 Remote,跑一个简单脉冲 **On Pulse**;示波器确认通道波形 + emCCD 触发时序对。
3. notebook 跑 `exp.readout.sitemap(display=True)` / `thresholds(display=True)`,肉眼确认。
4. `exp.readout.detect(frames=1)` 应收到一帧并给出占据;再开 task console 看实时。

---

## 4. 最易当场报错的点(对照表)

| 现象 | 根因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: ...dcam` / `failed to open qCMOS` | DCAM SDK / DLL 缺失或相机没连 | 装 SDK、确认 `dcamapi.dll` 在 PATH、`device_index` 对 |
| `ConnectionRefused` / `socket.timeout` | server 没起 / IP 端口错 / 防火墙 | 先起 `run_server.bat`;核对 `remote_template.json` 的 host:port;放行端口 |
| 首次 `prepare()` 报 `register-layout mismatch` | bitstream 旧,LAYOUT_ID 与 host 不符 | 重 build + 重烧 bitstream + 重启 server(这是保护,别绕过) |
| server 起不来 / JTAG 报错 | hw_server 没起 / JTAG 接触 / 板掉电 | 查电源、JTAG 线;Vivado 硬件管理器单独验证 |
| `qCMOS timed out` 等不到帧 | 相机收不到触发(通道/触发名不匹配) | 核对 XDC 的 `channels` 与相机 config 的 `capture_trigger_channels`;示波器看触发线 |

> 真机出问题先翻 memory 根因记录(`register-layout-handshake` / `stale-bus-delay` / `prefetch-pipeline-depth`
> / `fire-seed-stale-count` 等)与 `docs/MAINTAINER_NOTES.md`,多数历史坑已在那里定位过。
