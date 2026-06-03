<!-- cell:markdown -->
# Neutral atom FPGA pulse-streamer server

这个 notebook 在 Verilog/FPGA 电脑上运行。它启动 40ch FPGA pulse-streamer server，等待控制电脑上的 `RemoteSequencer` 连接。

推荐架构是固定烧一个 `zlc_pulse_streamer_top_40ch` bitstream。control 电脑每次 acquisition 只发送 `PulseSequence` 或 GUI 的 `PulseTableState`；FPGA 电脑把它编译成未展开的 40-bit `ticks/masks` edge table 和 repeat metadata，上传到 FPGA runtime RAM，再给一个 start toggle。网络和 Vivado 只负责上传表和发 start，不参与微秒级 timing。

```text
control computer RemoteSequencer
  -> RPyC
  -> SequencerService on FPGA/Vivado computer
  -> fpga_pulse_streamer backend
  -> fixed 40ch zlc_pulse_streamer bitstream
```

GUI 只决定前端显示/编辑哪些 channel。无论 GUI 显示 4 路还是 40 路，server 上传到 FPGA 的 program 都按 `ch00...ch39` 补全；未配置 channel 的 mask bit 是 0。

<!-- cell:code -->
{{BOOTSTRAP_CELL}}

<!-- cell:code -->
import os
from pathlib import Path

import Zou_lab_control.frontend as zf
import Zou_lab_control.neutral_atom as na

zf.enable_long_output()

<!-- cell:markdown -->
## 1. Build and program the 40ch bitstream

PowerShell 推荐入口：

```powershell
cd D:\ZLC
.\fpga\build_and_program.bat --help
.\fpga\build_and_program.bat --check
.\fpga\build_and_program.bat
```

`--check` 不需要真实 pin XDC，只做 40ch HDL/VIO 宽度自查。真实 build/program 需要先把 `fpga\pulse_streamer\zlc_pulse_streamer_40ch.xdc.template` 复制成 `zlc_pulse_streamer_40ch.xdc` 并填完 `ch00...ch39` 的真实 package pin。也可以设置：

```powershell
$env:ZLC_PS_40CH_XDC = "D:\fpga_pin_maps\zlc_pulse_streamer_40ch_my_board.xdc"
```

如果 Vivado 不在默认路径：

```powershell
$env:ZLC_PS_VIVADO_BIN = "C:\Xilinx\Vivado\2019.2\bin\vivado.bat"
```

`fpga\build_and_program.bat --diagnose` 可以列出 Vivado hardware target 和 FPGA device，不会 program 或 fire pulse。若 Vivado GUI 能看到 Digilent target 但 `Number of devices: 0`，先检查板卡供电、JTAG/mode jumper、线缆、power-source jumper，再重新 Auto Connect。

生成物通常在：

```text
fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.xpr
fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.runs\impl_1\zlc_pulse_streamer_top_40ch.bit
fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.runs\impl_1\zlc_pulse_streamer_top_40ch.ltx
```

VIO probe 约定：

```text
probe_out0  zlc_reset           width 1
probe_out1  zlc_start           width 1
probe_out2  zlc_prog_we         width 1
probe_out3  zlc_prog_addr       width 7
probe_out4  zlc_prog_tick       width 32
probe_out5  zlc_prog_mask       width 40
probe_out6  zlc_prog_count      width 8
probe_out7  zlc_repeat_forever  width 1
probe_out8  zlc_loop_start_addr width 7
probe_out9  zlc_loop_end_tick   width 32
probe_out10 zlc_loop_count      width 32
probe_in0   zlc_running         width 1
probe_in1   zlc_done            width 1
```

Verilog 原理：Python 上传的是 edge table，每一行是绝对 FPGA tick 和这一刻之后的 40-bit output mask。`Fire` 只把 `zlc_start` 翻转一次，Verilog 把任意方向的翻转都当成 start event；之后 FPGA 自己用 `time_count` 按 clock 数 tick，在 `time_count == tick_mem[edge_index]` 时更新 `state_mask`。所以微秒级 pulse timing 不依赖 Python、RPyC、Vivado 或 Windows 调度。`repeat_forever` 和 repeat bracket 也由 Verilog metadata 执行，不会展开成超长表。

<!-- cell:code -->
PROJECT_ROOT = Path("..").resolve()
FPGA_DIR = PROJECT_ROOT / "fpga" / "pulse_streamer"

CHANNELS = [f"ch{i:02d}" for i in range(40)]
TRIGGER_CHANNELS = ["ch03"]
CLOCK_HZ = 100_000_000.0
MAX_EDGES = 128
TICK_WIDTH = 32

for filename in (
    "zlc_pulse_streamer.v",
    "zlc_pulse_streamer_top_40ch.v",
    "zlc_pulse_streamer_40ch.xdc.template",
    "create_project_40ch.tcl",
    "program_fpga_40ch.tcl",
    "check_40ch_synth.tcl",
    "diagnose_hw_target.tcl",
):
    path = FPGA_DIR / filename
    print(path.exists(), path)

for filename in ("build_and_program.bat", "run_server.bat"):
    path = PROJECT_ROOT / "fpga" / filename
    print(path.exists(), path)

print((PROJECT_ROOT / "pulse_gui.bat").exists(), PROJECT_ROOT / "pulse_gui.bat")

<!-- cell:markdown -->
## 2. Configure Vivado paths and backend

如果已经运行 `.\fpga\build_and_program.bat`，下面三个 Vivado 路径应指向 40ch build 目录。第一次烧板时用 bat program；server 默认不再重复 program。

<!-- cell:code -->
HOST = "0.0.0.0"
PORT = 18861
STATE_DIR = PROJECT_ROOT / "fpga" / "pulse_streamer" / "build" / "zlc_sequencer_state_40ch"

def find_vivado_bin():
    if os.environ.get("ZLC_PS_VIVADO_BIN"):
        return os.environ["ZLC_PS_VIVADO_BIN"]
    for root in (Path(r"C:\Xilinx\Vivado"), Path(r"D:\Xilinx\Vivado")):
        if root.exists():
            candidates = sorted(root.glob(r"*\bin\vivado.bat"))
            if candidates:
                return str(candidates[-1])
    return "vivado"

os.environ["ZLC_PS_VIVADO_BIN"] = find_vivado_bin()
os.environ["ZLC_PS_VIVADO_PROJECT"] = str(FPGA_DIR / "build" / "zlc_pulse_streamer_40ch" / "zlc_pulse_streamer_40ch.xpr")
os.environ["ZLC_PS_VIVADO_BIT"] = str(FPGA_DIR / "build" / "zlc_pulse_streamer_40ch" / "zlc_pulse_streamer_40ch.runs" / "impl_1" / "zlc_pulse_streamer_top_40ch.bit")
os.environ["ZLC_PS_VIVADO_LTX"] = str(FPGA_DIR / "build" / "zlc_pulse_streamer_40ch" / "zlc_pulse_streamer_40ch.runs" / "impl_1" / "zlc_pulse_streamer_top_40ch.ltx")
os.environ["ZLC_PS_VIVADO_PROGRAM_ON_RUN"] = "0"
os.environ["ZLC_PS_SERVER_BACKEND"] = "vivado-session"
os.environ["ZLC_PS_VIO_FILTER"] = 'CELL_NAME=~"*vio*"'
os.environ["ZLC_PS_MAX_EDGES"] = str(MAX_EDGES)
os.environ["ZLC_PS_TICK_WIDTH"] = str(TICK_WIDTH)
os.environ["ZLC_PS_CHANNEL_COUNT"] = "40"

for key in ("ZLC_PS_VIVADO_BIN", "ZLC_PS_VIVADO_PROJECT", "ZLC_PS_VIVADO_BIT", "ZLC_PS_VIVADO_LTX"):
    print(key, Path(os.environ[key]).exists(), os.environ[key])
print("channels", CHANNELS)
print("trigger", TRIGGER_CHANNELS)

<!-- cell:markdown -->
## 3. PowerShell command equivalent

不用 Jupyter 时，推荐直接运行 bat。server 启动后 terminal 会一直阻塞，这是正常的。

```powershell
cd D:\ZLC
.\fpga\run_server.bat
```

下面是等价的展开版命令，便于检查环境变量：

<!-- cell:code -->
print(fr"""
cd "{PROJECT_ROOT}"
$env:PYTHONPATH = (Get-Location).Path

$env:ZLC_PS_VIVADO_BIN = "{os.environ["ZLC_PS_VIVADO_BIN"]}"
$env:ZLC_PS_VIVADO_PROJECT = "{os.environ["ZLC_PS_VIVADO_PROJECT"]}"
$env:ZLC_PS_VIVADO_BIT = "{os.environ["ZLC_PS_VIVADO_BIT"]}"
$env:ZLC_PS_VIVADO_LTX = "{os.environ["ZLC_PS_VIVADO_LTX"]}"
$env:ZLC_PS_VIVADO_PROGRAM_ON_RUN = "{os.environ["ZLC_PS_VIVADO_PROGRAM_ON_RUN"]}"
$env:ZLC_PS_SERVER_BACKEND = "{os.environ["ZLC_PS_SERVER_BACKEND"]}"
$env:ZLC_PS_VIO_FILTER = '{os.environ["ZLC_PS_VIO_FILTER"]}'
$env:ZLC_PS_MAX_EDGES = "{os.environ["ZLC_PS_MAX_EDGES"]}"
$env:ZLC_PS_TICK_WIDTH = "{os.environ["ZLC_PS_TICK_WIDTH"]}"
$env:ZLC_PS_CHANNEL_COUNT = "40"

python -m Zou_lab_control.neutral_atom.devices.sequencer_server `
  --backend vivado-session `
  --host {HOST} `
  --port {PORT} `
  --channels {" ".join(CHANNELS)} `
  --trigger-channels {" ".join(TRIGGER_CHANNELS)} `
  --clock-hz {CLOCK_HZ:g} `
  --state-dir "{STATE_DIR}"
""")

<!-- cell:markdown -->
## 4. Start the server

运行下面这个 cell 后它会一直阻塞，保持这个 notebook/kernel 不要关，然后去控制电脑运行 `neutral_atom_hardware_quickstart.ipynb`。

<!-- cell:code -->
na.run_sequencer_server(
    channels=CHANNELS,
    trigger_channels=TRIGGER_CHANNELS,
    host=HOST,
    port=PORT,
    clock_hz=CLOCK_HZ,
    state_dir=STATE_DIR,
    backend="vivado-session",
)

<!-- cell:markdown -->
## 5. Optional: run the pulse GUI on this FPGA computer

`run_sequencer_server(...)` 会阻塞当前 kernel。要在 FPGA/Vivado 电脑本机打开 pulse GUI，请在另一个 PowerShell 或另一个 notebook kernel 里运行：

```powershell
.\pulse_gui.bat --remote-host 127.0.0.1 --remote-port 18861 --state .\pulses\camera_imaging_40ch.json
```

GUI 仍然只是前端；实际 prepare/fire/wait 通过正在运行的 40ch server 执行。默认 preset 只显示前 4 路，但上传时仍是 40ch full-width program，`ch04..ch39` 全部是 off。若窗口在当前显示器上偏大，可以传 `scale=0.82, window_ratio=0.90`。

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
# pulse_gui = zf.show_pulse_gui(
#     state=na.PulseTableState.load(PROJECT_ROOT / "pulses" / "camera_imaging_40ch.json"),
#     channels=CHANNELS,
#     sequencer=local_sequencer,
#     scale=0.82,
#     window_ratio=0.90,
# )
# pulse_gui
