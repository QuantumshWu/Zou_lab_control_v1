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
## 1. Generate the fixed FPGA pulse-streamer HDL

第一次部署时先生成 HDL。把 `zlc_pulse_streamer.v` 加到 Vivado 工程，把 `zlc_pulse_streamer_top_example.v` 当作 wiring reference：它说明 VIO probe 的名字、宽度和方向。

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

`out[0:3]` 分别接到 `trap/cooling/probe/qcm_trigger` 的真实 FPGA 输出 pin。qCMOS 必须接 `qcm_trigger` 这个输出。

<!-- cell:code -->
PROJECT_ROOT = Path("..").resolve()

CHANNELS = ["trap", "cooling", "probe", "qcm_trigger"]
TRIGGER_CHANNELS = ["qcm_trigger"]
CLOCK_HZ = 100_000_000.0
MAX_EDGES = 1024
TICK_WIDTH = 32

HDL_DIR = Path(r"D:\zlc_pulse_streamer_hdl")
hdl_files = na.write_pulse_streamer_hdl_bundle(
    HDL_DIR,
    channels=CHANNELS,
    max_edges=MAX_EDGES,
    tick_width=TICK_WIDTH,
)
hdl_files

<!-- cell:markdown -->
## 2. Configure Vivado paths and backend commands

在 Vivado 里用上面生成的 HDL 建工程、连接 VIO 和 output pins、generate bitstream。然后把下面三个路径改成真实 `.xpr/.bit/.ltx`。

第一次烧板子时设 `ZLC_PS_VIVADO_PROGRAM_ON_RUN="1"`；烧成功后可以改回 `"0"`，后续只加载 probes 和写 runtime table。

<!-- cell:code -->
HOST = "0.0.0.0"
PORT = 18861
STATE_DIR = Path(r"D:\zlc_sequencer_state")

os.environ["ZLC_PS_VIVADO_BIN"] = r"C:\Xilinx\Vivado\2019.2\bin\vivado.bat"
os.environ["ZLC_PS_VIVADO_PROJECT"] = r"D:\time_sequence\zlc_pulse_streamer\zlc_pulse_streamer.xpr"
os.environ["ZLC_PS_VIVADO_BIT"] = r"D:\time_sequence\zlc_pulse_streamer\zlc_pulse_streamer.runs\impl_1\main.bit"
os.environ["ZLC_PS_VIVADO_LTX"] = r"D:\time_sequence\zlc_pulse_streamer\zlc_pulse_streamer.runs\impl_1\main.ltx"
os.environ["ZLC_PS_VIVADO_PROGRAM_ON_RUN"] = "1"
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
