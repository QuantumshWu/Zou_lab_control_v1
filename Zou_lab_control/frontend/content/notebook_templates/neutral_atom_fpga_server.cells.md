<!-- cell:markdown -->
# Neutral atom FPGA pulse-streamer server

这个 notebook 在 Verilog/FPGA 电脑上运行。它启动 `Zou_lab_control.neutral_atom` sequencer server，等待控制电脑上的 `RemoteSequencer` 连接。

推荐架构是固定烧一个 `zlc_pulse_streamer` bitstream。control 电脑每次 acquisition 只发送 `PulseSequence`；FPGA 电脑把它编译成 `ticks/masks` edge table，上传到 FPGA 的 runtime RAM，再给一个 start edge。网络只负责上传表和发 start，不参与微秒级 timing。

```text
control computer RemoteSequencer
  -> RPyC
  -> SequencerService on FPGA/Vivado computer
  -> fpga_pulse_streamer backend
  -> fixed zlc_pulse_streamer bitstream edge-table RAM
```

旧 `address_switch` backend 只适合应急 first-light，不再作为默认路径。

<!-- cell:code -->
{{BOOTSTRAP_CELL}}

<!-- cell:code -->
import os
from pathlib import Path
import sys

import Zou_lab_control.frontend as zf
import Zou_lab_control.neutral_atom as na

zf.enable_long_output()

<!-- cell:markdown -->
## 1. Build and program the fixed FPGA pulse-streamer bitstream

仓库里已经有 first-light Vivado 工程入口：

```text
fpga/pulse_streamer/
  zlc_pulse_streamer.v
  zlc_pulse_streamer_top_4ch.v
  zlc_pulse_streamer_top_40ch.v
  zlc_pulse_streamer_4ch.xdc
  zlc_pulse_streamer_40ch.xdc.template
  create_project_4ch.tcl
  create_project_40ch.tcl
  program_fpga_4ch.tcl
  program_fpga_40ch.tcl
  build_4ch_bitstream.bat
  program_4ch_fpga.bat
```

在 Verilog/FPGA 电脑 PowerShell 运行：

```powershell
cd D:\GitHub\Zou_lab_control_v1
.\fpga\pulse_streamer\build_4ch_bitstream.bat
.\fpga\pulse_streamer\program_4ch_fpga.bat
.\fpga\pulse_streamer\smoke_test_4ch_upload.bat
```

这些 bat 会把 Vivado/Python 的 errorlevel 传回 PowerShell；如果 synth/impl 没有 Complete，或者 `.bit/.ltx` 缺失，会提前失败，不要继续启动 server。

这个 bitstream 把 FPGA 变成 runtime pulse-streamer。后续 notebook/server 只通过 VIO 写 `ticks/masks` edge table，不需要每次重新综合。

`smoke_test_4ch_upload.bat` 不需要 qCMOS。先用示波器确认 `trap/cooling/probe/qcm_trigger` 输出符合：

```text
trap         high 0-10 us
cooling      high 0-3 us
probe        high 2-6 us
qcm_trigger  high 2-3 us
```

这个 smoke-test bat 默认使用上一节 build 目录里的 `.xpr/.bit/.ltx`，因此新的 Vivado batch 进程也会加载 VIO probes。

工程里需要一个 VIO IP，probe 约定如下：

```text
probe_out0 zlc_reset      width 1
probe_out1 zlc_start      width 1
probe_out2 zlc_prog_we    width 1
probe_out3 zlc_prog_addr  width 10   # MAX_EDGES=1024
probe_out4 zlc_prog_tick  width 32
probe_out5 zlc_prog_mask  width 4    # CHANNELS=trap/cooling/probe/qcm_trigger
probe_out6 zlc_prog_count width 11
probe_in0  zlc_running    width 1
probe_in1  zlc_done       width 1
```

server 查 probe 时会先找 `zlc_reset/zlc_start/...` 这些语义名；如果 Vivado `.ltx` 里只有 `probe_out0/probe_out1/...`，也会自动 fallback 到对应端口名。

`out[0:3]` 分别接到 `trap/cooling/probe/qcm_trigger` 的真实 FPGA 输出 pin。qCMOS 必须接 `qcm_trigger` 这个输出。

<!-- cell:code -->
PROJECT_ROOT = Path("..").resolve()

CHANNELS = ["trap", "cooling", "probe", "qcm_trigger"]
TRIGGER_CHANNELS = ["qcm_trigger"]
CLOCK_HZ = 100_000_000.0
MAX_EDGES = 1024
TICK_WIDTH = 32

# 40-channel example after building/programming the 40ch top and filling the XDC:
# CHANNELS = ["trap", "cooling", "probe", "qcm_trigger"] + [f"ch{i:02d}" for i in range(4, 40)]
# TRIGGER_CHANNELS = ["qcm_trigger"]

FPGA_DIR = PROJECT_ROOT / "fpga" / "pulse_streamer"
for filename in (
    "zlc_pulse_streamer.v",
    "zlc_pulse_streamer_top_4ch.v",
    "zlc_pulse_streamer_top_40ch.v",
    "zlc_pulse_streamer_4ch.xdc",
    "zlc_pulse_streamer_40ch.xdc.template",
    "create_project_4ch.tcl",
    "create_project_40ch.tcl",
    "program_fpga_4ch.tcl",
    "program_fpga_40ch.tcl",
    "build_4ch_bitstream.bat",
    "program_4ch_fpga.bat",
    "build_40ch_bitstream.bat",
    "program_40ch_fpga.bat",
    "smoke_test_4ch.py",
    "smoke_test_4ch_upload.bat",
):
    path = FPGA_DIR / filename
    print(path.exists(), path)

<!-- cell:markdown -->
## 2. Configure Vivado paths and backend commands

如果用仓库里的 4ch first-light 入口，上一节的 `build_4ch_bitstream.bat` 已经完成建工程、创建 VIO、综合、实现和生成 bitstream。下面三个路径应指向这个脚本生成的 `.xpr/.bit/.ltx`。

第一次烧板子时设 `ZLC_PS_VIVADO_PROGRAM_ON_RUN="1"`；烧成功后可以改回 `"0"`，后续只加载 probes 和写 runtime table。

<!-- cell:code -->
HOST = "0.0.0.0"
PORT = 18861
STATE_DIR = Path(r"D:\zlc_sequencer_state")

os.environ["ZLC_PS_VIVADO_BIN"] = r"C:\Xilinx\Vivado\2019.2\bin\vivado.bat"
os.environ["ZLC_PS_VIVADO_PROJECT"] = str(PROJECT_ROOT / "fpga" / "pulse_streamer" / "build" / "zlc_pulse_streamer_4ch" / "zlc_pulse_streamer_4ch.xpr")
os.environ["ZLC_PS_VIVADO_BIT"] = str(PROJECT_ROOT / "fpga" / "pulse_streamer" / "build" / "zlc_pulse_streamer_4ch" / "zlc_pulse_streamer_4ch.runs" / "impl_1" / "zlc_pulse_streamer_top_4ch.bit")
os.environ["ZLC_PS_VIVADO_LTX"] = str(PROJECT_ROOT / "fpga" / "pulse_streamer" / "build" / "zlc_pulse_streamer_4ch" / "zlc_pulse_streamer_4ch.runs" / "impl_1" / "zlc_pulse_streamer_top_4ch.ltx")
os.environ["ZLC_PS_VIVADO_PROGRAM_ON_RUN"] = "0"
os.environ["ZLC_PS_VIO_FILTER"] = 'CELL_NAME=~"*vio*"'
os.environ["ZLC_PS_MAX_EDGES"] = str(MAX_EDGES)
os.environ["ZLC_PS_TICK_WIDTH"] = str(TICK_WIDTH)
os.environ["ZLC_PS_CHANNEL_COUNT"] = str(len(CHANNELS))

PREPARE_COMMAND = f'"{sys.executable}" -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer prepare'
FIRE_COMMAND = f'"{sys.executable}" -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer fire'
WAIT_DONE_COMMAND = f'"{sys.executable}" -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer wait_done'
SAFE_STATE_COMMAND = f'"{sys.executable}" -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer safe_state'

for key in ("ZLC_PS_VIVADO_BIN", "ZLC_PS_VIVADO_PROJECT", "ZLC_PS_VIVADO_BIT", "ZLC_PS_VIVADO_LTX"):
    print(key, Path(os.environ[key]).exists(), os.environ[key])

PREPARE_COMMAND, FIRE_COMMAND, WAIT_DONE_COMMAND, SAFE_STATE_COMMAND

<!-- cell:markdown -->
## 3. PowerShell command equivalent

不用 Jupyter 时，在 Verilog 电脑 PowerShell 运行这段。server 启动后 terminal 会一直阻塞，这是正常的。

<!-- cell:code -->
print(fr"""
cd "{PROJECT_ROOT}"
$env:PYTHONPATH = (Get-Location).Path

$env:ZLC_PS_VIVADO_BIN = "{os.environ["ZLC_PS_VIVADO_BIN"]}"
$env:ZLC_PS_VIVADO_PROJECT = "{os.environ["ZLC_PS_VIVADO_PROJECT"]}"
$env:ZLC_PS_VIVADO_BIT = "{os.environ["ZLC_PS_VIVADO_BIT"]}"
$env:ZLC_PS_VIVADO_LTX = "{os.environ["ZLC_PS_VIVADO_LTX"]}"
$env:ZLC_PS_VIVADO_PROGRAM_ON_RUN = "{os.environ["ZLC_PS_VIVADO_PROGRAM_ON_RUN"]}"
$env:ZLC_PS_VIO_FILTER = '{os.environ["ZLC_PS_VIO_FILTER"]}'
$env:ZLC_PS_MAX_EDGES = "{os.environ["ZLC_PS_MAX_EDGES"]}"
$env:ZLC_PS_TICK_WIDTH = "{os.environ["ZLC_PS_TICK_WIDTH"]}"
$env:ZLC_PS_CHANNEL_COUNT = "{os.environ["ZLC_PS_CHANNEL_COUNT"]}"

Test-Path $env:ZLC_PS_VIVADO_BIN
Test-Path $env:ZLC_PS_VIVADO_PROJECT
Test-Path $env:ZLC_PS_VIVADO_BIT
Test-Path $env:ZLC_PS_VIVADO_LTX

python -m Zou_lab_control.neutral_atom.devices.sequencer_server `
  --host {HOST} `
  --port {PORT} `
  --channels {" ".join(CHANNELS)} `
  --trigger-channels {" ".join(TRIGGER_CHANNELS)} `
  --clock-hz {CLOCK_HZ:g} `
  --state-dir "{STATE_DIR}" `
  --prepare-command "{PREPARE_COMMAND}" `
  --fire-command "{FIRE_COMMAND}" `
  --wait-done-command "{WAIT_DONE_COMMAND}" `
  --safe-state-command "{SAFE_STATE_COMMAND}"
""")

<!-- cell:markdown -->
## 4. Start the server

运行下面这个 cell 后它会一直阻塞，保持这个 notebook/kernel 不要关，然后去控制电脑运行 `neutral_atom_hardware_quickstart.ipynb`。

如果只想生成 HDL 或检查路径，不要运行这个 cell。

<!-- cell:code -->
na.run_sequencer_server(
    channels=CHANNELS,
    trigger_channels=TRIGGER_CHANNELS,
    host=HOST,
    port=PORT,
    clock_hz=CLOCK_HZ,
    state_dir=STATE_DIR,
    prepare_command=PREPARE_COMMAND,
    fire_command=FIRE_COMMAND,
    wait_done_command=WAIT_DONE_COMMAND,
    safe_state_command=SAFE_STATE_COMMAND,
)

<!-- cell:markdown -->
## 5. Optional: run the pulse GUI on this FPGA computer

`run_sequencer_server(...)` 会阻塞当前 kernel。要在 FPGA/Vivado 电脑本机打开 pulse GUI，请在另一个 Python 进程或另一个 notebook kernel 里运行下面的代码，并连接 `127.0.0.1:{PORT}`。GUI 仍然只是前端；实际 prepare/fire/wait 通过正在运行的 server 执行。新建 pulse 默认名是 `pulse_YYYYMMDD_HHMMSS`。`channels` 是硬件 channel 名和 FPGA bit order，例如 `ch00/ch01/...`；Name 面板左侧固定显示硬件 channel，右侧是可选 display label，GUI 不会自动猜物理含义。右侧 name 改动后，Delay 行、period checkbox 和 Preview y 轴会显示这个 label。传入 `clock_hz=CLOCK_HZ` 后，GUI 左侧 `step (ns)` 会默认等于 FPGA tick，例如 100 MHz 时是 10 ns。40-channel 展开时，channel name、delay 和 period checkbox 共用整体纵向滚动，方便检查每一路是否对齐。`X` 会把该 channel 的所有 period 设为 off，但不自动隐藏；`Hide Off` 只看 period 是否为 on，delay/name 会保留，重新 Add Channel 会按硬件顺序回原位。Preview 页会画未展开 period table 的 pulse 图，默认隐藏 always-off channel；如果 channel 有 display label，Preview y 轴显示 label。没有 bracket 时是 `repeat ∞`；bracket 覆盖所有 period 时是有限外层 repeat；bracket 在内部时整体仍然是 `repeat ∞`，Preview 会画整段 `∞` 和内部 `xN` 两套不同颜色的 bracket，状态栏显示 `repeat ∞ + Pm-Pn xN`。bracket 画在真实 start/stop 时间节点上，xlim 只负责留显示空间，负时间 tick label 会被隐藏。`Save Pulse` 默认保存到仓库 `pulses/` 目录；`Save Figure` 是 Preview 顶栏最右侧的一行按钮，单独保存 preview PNG。若窗口在当前显示器上偏大，可以传 `scale=0.82, window_ratio=0.90`。

<!-- cell:code -->
# import Zou_lab_control.frontend as zf
# import Zou_lab_control.neutral_atom as na
#
# local_sequencer = na.RemoteSequencer(
#     host="127.0.0.1",
#     port=PORT,
#     channels=CHANNELS,
#     clock_hz=CLOCK_HZ,
#     trigger_channels=TRIGGER_CHANNELS,
# )
# pulse_gui = zf.show_pulse_gui(channels=CHANNELS, sequencer=local_sequencer, scale=0.82, window_ratio=0.90)
# pulse_gui
