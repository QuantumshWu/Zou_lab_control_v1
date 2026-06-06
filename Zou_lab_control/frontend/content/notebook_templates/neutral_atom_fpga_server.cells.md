<!-- cell:markdown -->
# Neutral atom FPGA pulse-streamer server

这个 notebook 在 Verilog/FPGA 电脑上运行。它对应 `fpga\build_and_program.bat`
和 `fpga\run_server.bat` 的同一条 address-switch 路线。

推荐工作流：

```text
control computer RemoteSequencer / Pulse GUI
  -> RPyC
  -> SequencerService on FPGA/Vivado computer
  -> fpga_pulse_streamer Vivado/VIO runtime backend
  -> fixed address-switch pulse-streamer bitstream
```

GUI 只决定前端显示/编辑哪些 channel。无论 GUI 显示几路，server 上传到
FPGA 的 program 都按 XDC 推断的完整 channel order 补全；未配置 channel 的
mask bit 是 0。

<!-- cell:code -->
{{BOOTSTRAP_CELL}}

<!-- cell:code -->
import os
from pathlib import Path

import Zou_lab_control.frontend as zf
import Zou_lab_control.neutral_atom as na

zf.enable_long_output()

<!-- cell:markdown -->
## 1. Build and program the address-switch bitstream

PowerShell 推荐入口：

```powershell
cd D:\ZLC
.\fpga\build_and_program.bat --help
.\fpga\build_and_program.bat --check
.\fpga\build_and_program.bat
```

默认 XDC 是历史 address-switch pin map：

```text
references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc
```

如果 Vivado 不在默认路径：

```powershell
$env:ZLC_PS_VIVADO_BIN = "C:\Xilinx\Vivado\2019.1\bin\vivado.bat"
```

如果要使用另一份已确认的板级 XDC：

```powershell
$env:ZLC_PS_XDC = "D:\fpga_pin_maps\my_address_switch_board.xdc"
```

生成物默认在：

```text
fpga\build\address_switch\address_switch.xpr
fpga\build\address_switch\address_switch.runs\impl_1\zlc_pulse_streamer_top_address_switch.bit
fpga\build\address_switch\address_switch.runs\impl_1\zlc_pulse_streamer_top_address_switch.ltx
```

`--diagnose` 可以列出 Vivado hardware target 和 FPGA device，不会 program 或
fire pulse。若 Vivado GUI 能看到 Digilent target 但 `Number of devices: 0`，
先检查板卡供电、JTAG/mode jumper、线缆、power-source jumper，再重新
Auto Connect。

<!-- cell:code -->
PROJECT_ROOT = Path("..").resolve()
FPGA_DIR = PROJECT_ROOT / "fpga" / "pulse_streamer"
XDC = PROJECT_ROOT / "references" / "source_archives" / "address_switch" / "address_switch.srcs" / "constrs_1" / "new" / "addre.xdc"

CHANNELS = na.infer_xdc_channels(XDC)
CHANNEL_LABELS = na.infer_xdc_channel_labels(XDC)
CHANNEL_PINS = na.infer_xdc_channel_pins(XDC)
TRIGGER_CHANNELS = na.infer_xdc_trigger_channels(XDC)
if not TRIGGER_CHANNELS:
    raise RuntimeError("The selected XDC must label the camera trigger output as emCCD.")
CLOCK_HZ = 50_000_000.0
MAX_EDGES = int(os.environ.get("ZLC_PS_MAX_EDGES", "1024"))
MAX_SCAN_POINTS = int(os.environ.get("ZLC_PS_MAX_SCAN_POINTS", "1024"))
TICK_WIDTH = int(os.environ.get("ZLC_PS_TICK_WIDTH", "32"))
RESOURCE_TARGET_PCT = int(os.environ.get("ZLC_PS_RESOURCE_TARGET_PCT", "70"))

print("channel count:", len(CHANNELS))
print("trigger:", TRIGGER_CHANNELS, {ch: CHANNEL_LABELS.get(ch) for ch in TRIGGER_CHANNELS})
print("camera subset:", {ch: (CHANNEL_LABELS.get(ch), CHANNEL_PINS.get(ch)) for ch in ("ch09", "ch00", "ch03", "ch11")})
print("trig output still exists:", "ch06", CHANNEL_LABELS.get("ch06"), CHANNEL_PINS.get("ch06"))
print("clock:", CLOCK_HZ, "Hz; step:", 1e9 / CLOCK_HZ, "ns")

for filename in (
    "zlc_pulse_streamer.v",
    "zlc_pulse_streamer_top_address_switch.v",
    "create_project_address_switch.tcl",
    "program_fpga_address_switch.tcl",
    "check_address_switch_synth.tcl",
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

如果已经运行 `.\fpga\build_and_program.bat`，下面三个 Vivado 路径应指向
`address_switch` build 目录。第一次烧板时用 bat program；server 默认不再
重复 program。

<!-- cell:code -->
HOST = "0.0.0.0"
PORT = 18861

BUILD_ROOT = PROJECT_ROOT / "fpga" / "build"
STATE_DIR = Path(os.environ.get("ZLC_PS_STATE_DIR", BUILD_ROOT / "state_address_switch"))
PROJECT_DIR = Path(os.environ.get("ZLC_PS_PROJECT_DIR", BUILD_ROOT / "address_switch"))

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
os.environ["ZLC_PS_PROJECT_DIR"] = str(PROJECT_DIR)
os.environ["ZLC_PS_VIVADO_PROJECT"] = str(PROJECT_DIR / "address_switch.xpr")
os.environ["ZLC_PS_VIVADO_BIT"] = str(PROJECT_DIR / "address_switch.runs" / "impl_1" / "zlc_pulse_streamer_top_address_switch.bit")
os.environ["ZLC_PS_VIVADO_LTX"] = str(PROJECT_DIR / "address_switch.runs" / "impl_1" / "zlc_pulse_streamer_top_address_switch.ltx")
os.environ["ZLC_PS_VIVADO_PROGRAM_ON_RUN"] = "0"
os.environ["ZLC_PS_SERVER_BACKEND"] = "vivado-session"
os.environ["ZLC_PS_VIO_FILTER"] = 'CELL_NAME=~"*vio*"'
os.environ["ZLC_PS_MAX_EDGES"] = str(MAX_EDGES)
os.environ["ZLC_PS_MAX_SCAN_POINTS"] = str(MAX_SCAN_POINTS)
os.environ["ZLC_PS_TICK_WIDTH"] = str(TICK_WIDTH)
os.environ["ZLC_PS_RESOURCE_TARGET_PCT"] = str(RESOURCE_TARGET_PCT)
os.environ["ZLC_PS_CLOCK_HZ"] = str(int(CLOCK_HZ))
os.environ["ZLC_PS_CHANNEL_COUNT"] = str(len(CHANNELS))
os.environ["ZLC_PS_XDC"] = str(XDC)

for key in ("ZLC_PS_VIVADO_BIN", "ZLC_PS_VIVADO_PROJECT", "ZLC_PS_VIVADO_BIT", "ZLC_PS_VIVADO_LTX", "ZLC_PS_XDC"):
    print(key, Path(os.environ[key]).exists(), os.environ[key])
print("server config", {"channels": len(CHANNELS), "trigger_channels": TRIGGER_CHANNELS, "clock_hz": CLOCK_HZ, "max_scan_points": MAX_SCAN_POINTS})

<!-- cell:markdown -->
## 3. PowerShell command equivalent

不用 Jupyter 时，推荐直接运行 bat。server 启动后 terminal 会一直阻塞，这是正常的。

```powershell
cd D:\ZLC
.\fpga\run_server.bat --check-config
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
$env:ZLC_PS_XDC = "{os.environ["ZLC_PS_XDC"]}"
$env:ZLC_PS_VIVADO_PROGRAM_ON_RUN = "0"
$env:ZLC_PS_SERVER_BACKEND = "vivado-session"
$env:ZLC_PS_VIO_FILTER = 'CELL_NAME=~"*vio*"'
$env:ZLC_PS_MAX_EDGES = "{MAX_EDGES}"
$env:ZLC_PS_MAX_SCAN_POINTS = "{MAX_SCAN_POINTS}"
$env:ZLC_PS_TICK_WIDTH = "{TICK_WIDTH}"
$env:ZLC_PS_RESOURCE_TARGET_PCT = "{RESOURCE_TARGET_PCT}"
$env:ZLC_PS_CLOCK_HZ = "{int(CLOCK_HZ)}"
$env:ZLC_PS_CHANNEL_COUNT = "{len(CHANNELS)}"

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

运行下面这个 cell 后它会一直阻塞。保持这个 notebook/kernel 不要关，然后去
控制电脑运行 `neutral_atom_hardware_quickstart.ipynb` 或打开 pulse GUI。

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

`run_sequencer_server(...)` 会阻塞当前 kernel。要在 FPGA/Vivado 电脑本机打开
pulse GUI，请在另一个 PowerShell 或另一个 notebook kernel 里运行：

```powershell
.\pulse_gui.bat --remote-host 127.0.0.1 --remote-port 18861 --state .\pulses\camera_imaging_address_switch.json
```

GUI 仍然只是前端；实际 prepare/fire/wait 通过正在运行的 sequencer server
执行。默认 preset 只显示相机成像子集，但上传时仍是 full-width program。

First-light checklist on the FPGA computer:

```text
1. build_and_program.bat completed and programmed the address_switch bitstream.
2. run_server.bat --check-config prints the same bit/LTX/XDC paths.
3. run_server.bat is still open and blocking.
4. pulse_gui.bat connects to 127.0.0.1:18861.
5. Load pulses/camera_imaging_address_switch.json.
6. Edit tab raw column shows package pins: M17, F15, N15, M13.
7. Press On Pulse for a repeat-forever scope test, or use finite API for one shot.
```

Expected camera-imaging physical outputs:

```text
ch09 trap    M17
ch00 cooling F15
ch03 probe   N15
ch11 emCCD   M13   qCMOS/emCCD trigger for this preset
ch06 trig    R17   still available, but not the preset camera trigger
```

If `On Pulse` reports no error but the scope is flat, check these before editing
Python code: the probe is on the correct address-switch pin, the programmed
bit/LTX belongs to the current `fpga/build/address_switch` directory, Vivado
Hardware Manager sees a device rather than only a Digilent target, and the GUI
is connected to the running server rather than offline mode.

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
#     state=na.PulseTableState.load(PROJECT_ROOT / "pulses" / "camera_imaging_address_switch.json"),
#     channels=CHANNELS,
#     sequencer=local_sequencer,
#     scale=0.82,
#     window_ratio=0.90,
# )
# pulse_gui

<!-- cell:markdown -->
## 6. Runtime table principle

The bitstream is fixed after programming. It does not contain a hard-coded
experiment pulse. Each GUI/API `prepare` uploads runtime metadata:

```text
edge template:
  tick[i]          absolute FPGA clock tick
  mask[i]          full-width output state
  x_coeff[i]       optional symbolic x contribution
  y_coeff[i]       optional symbolic y contribution

scan RAM:
  scan_x[j], scan_y[j]

repeat metadata:
  repeat_forever, loop_start_addr, loop_end_tick, loop_count
```

One edge row is a complete state mask at a time point. If trap and emCCD change
at the same tick, that is still one row, not two. GUI hidden channels do not
change row width; they simply have zero bits in the uploaded masks.

For `Scan X`, the GUI/API sends an ordered list like `[x0, x1, ...]`, internally
normalized to `(x, 0)` points. For `Scan XY`, it sends `[(x0, y0), ...]`.
The compiler accepts a scan template only when edge ordering stays the same for
every scan point. If two symbolic edges would swap order, split the scan or
prepare one point at a time.

Analog buses are compiled through a separate bus segment memory with
`bus_id/start_tick/stop_tick/start_value/stop_value/mode`. The FPGA generates
10-bit stair-step ramps locally, so dense bus ramps do not consume ordinary
digital edge rows.
