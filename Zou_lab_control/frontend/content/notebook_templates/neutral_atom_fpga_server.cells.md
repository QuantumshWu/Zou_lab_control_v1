<!-- cell:markdown -->
# Neutral atom FPGA pulse-streamer server

这个 notebook 在 Verilog/FPGA 电脑上运行。它对应 `fpga\build_and_program.bat`
和 `fpga\run_server.bat` 的同一条 **最终单一设计** 路线(1-tick 无缝预取 +
无限流式 scan,JTAG-to-AXI 控制,无 VIO/loader/变体)。

推荐工作流：

```text
control computer RemoteSequencer / Pulse GUI
  -> RPyC
  -> SequencerService on FPGA/Vivado computer
  -> axi_session.VivadoAxiStreamerSession (JTAG-to-AXI / hw_axi)
  -> zlc_pulse_streamer_top bitstream (edge + scan BRAM, 1-tick FIFO prefetch)
```

GUI 只决定前端显示/编辑哪些 channel。无论 GUI 显示几路,server 上传到
FPGA 的 program 都按 XDC 推断的完整 channel order 补全;未配置 channel 的
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
## 1. Build and program the bitstream

PowerShell 推荐入口：

```powershell
cd D:\ZLC
.\fpga\build_and_program.bat --help
.\fpga\build_and_program.bat --check
.\fpga\build_and_program.bat
```

默认 XDC 是板级 pin map（见 fpga\board_config\README.md）：

```text
fpga\board_config\board.xdc
```

如果 Vivado 不在默认路径：

```powershell
$env:ZLC_PS_VIVADO_BIN = "C:\Xilinx\Vivado\2019.1\bin\vivado.bat"
```

如果要使用另一份已确认的板级 XDC：

```powershell
$env:ZLC_PS_XDC = "D:\fpga_pin_maps\my_board.xdc"
```

生成物默认在：

```text
fpga\build\ps\ps.runs\impl_1\zlc_pulse_streamer_top.bit
fpga\build\ps\ps.runs\impl_1\zlc_pulse_streamer_top.ltx
```

`--diagnose` 可以列出 Vivado hardware target 和 FPGA device,不会 program 或
fire pulse。若 Vivado GUI 能看到 Digilent target 但 `Number of devices: 0`,
先检查板卡供电、JTAG/mode jumper、线缆、power-source jumper,再重新
Auto Connect。

<!-- cell:code -->
PROJECT_ROOT = Path("..").resolve()
FPGA_DIR = PROJECT_ROOT / "fpga" / "pulse_streamer"
XDC = PROJECT_ROOT / "fpga" / "board_config" / "board.xdc"

CHANNELS = na.infer_xdc_channels(XDC)
CHANNEL_LABELS = na.infer_xdc_channel_labels(XDC)
CHANNEL_PINS = na.infer_xdc_channel_pins(XDC)
TRIGGER_CHANNELS = na.infer_xdc_trigger_channels(XDC)
if not TRIGGER_CHANNELS:
    raise RuntimeError("The selected XDC must label the camera trigger output as emCCD.")
CLOCK_HZ = 50_000_000.0

print("channel count:", len(CHANNELS))
print("trigger:", TRIGGER_CHANNELS, {ch: CHANNEL_LABELS.get(ch) for ch in TRIGGER_CHANNELS})
print("camera subset:", {ch: (CHANNEL_LABELS.get(ch), CHANNEL_PINS.get(ch)) for ch in ("ch09", "ch00", "ch03", "ch11")})
print("trig output still exists:", "ch06", CHANNEL_LABELS.get("ch06"), CHANNEL_PINS.get("ch06"))
print("clock:", CLOCK_HZ, "Hz; step:", 1e9 / CLOCK_HZ, "ns")

# The final design is ONE clean build target -- these are the only RTL/tcl files.
for filename in (
    "zlc_edge_streamer.v",
    "zlc_pulse_streamer_top.v",
    "create_project.tcl",
    "program_fpga.tcl",
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

如果已经运行 `.\fpga\build_and_program.bat`,下面两个 Vivado 路径应指向
`fpga\build\ps` build 目录。第一次烧板时用 bat program;server
默认不再重复 program。后端固定是 `jtag-axi`(持久 Vivado hw_axi 会话)。

<!-- cell:code -->
HOST = "0.0.0.0"
PORT = 18861

BUILD_ROOT = PROJECT_ROOT / "fpga" / "build"
STATE_DIR = Path(os.environ.get("ZLC_PS_STATE_DIR", BUILD_ROOT / "state"))
PROJECT_DIR = Path(os.environ.get("ZLC_PS_PROJECT_DIR", BUILD_ROOT / "ps"))
RUNS = PROJECT_DIR / "ps.runs" / "impl_1"

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
os.environ["ZLC_PS_VIVADO_BIT"] = str(RUNS / "zlc_pulse_streamer_top.bit")
os.environ["ZLC_PS_VIVADO_LTX"] = str(RUNS / "zlc_pulse_streamer_top.ltx")
os.environ["ZLC_PS_VIVADO_PROGRAM_ON_RUN"] = "0"
os.environ["ZLC_PS_SERVER_BACKEND"] = "jtag-axi"
os.environ["ZLC_PS_CLOCK_HZ"] = str(int(CLOCK_HZ))
os.environ["ZLC_PS_CHANNEL_COUNT"] = str(len(CHANNELS))
os.environ["ZLC_PS_XDC"] = str(XDC)

for key in ("ZLC_PS_VIVADO_BIN", "ZLC_PS_VIVADO_BIT", "ZLC_PS_VIVADO_LTX", "ZLC_PS_XDC"):
    print(key, Path(os.environ[key]).exists(), os.environ[key])
print("server config", {"channels": len(CHANNELS), "trigger_channels": TRIGGER_CHANNELS, "clock_hz": CLOCK_HZ})

<!-- cell:markdown -->
## 3. PowerShell command equivalent

不用 Jupyter 时,推荐直接运行 bat。server 启动后 terminal 会一直阻塞,这是正常的。

```powershell
cd D:\ZLC
.\fpga\run_server.bat --check-config
.\fpga\run_server.bat
```

下面是等价的展开版命令,便于检查环境变量：

<!-- cell:code -->
print(fr"""
cd "{PROJECT_ROOT}"
$env:PYTHONPATH = (Get-Location).Path

$env:ZLC_PS_VIVADO_BIN = "{os.environ["ZLC_PS_VIVADO_BIN"]}"
$env:ZLC_PS_VIVADO_BIT = "{os.environ["ZLC_PS_VIVADO_BIT"]}"
$env:ZLC_PS_VIVADO_LTX = "{os.environ["ZLC_PS_VIVADO_LTX"]}"
$env:ZLC_PS_XDC = "{os.environ["ZLC_PS_XDC"]}"
$env:ZLC_PS_VIVADO_PROGRAM_ON_RUN = "0"
$env:ZLC_PS_SERVER_BACKEND = "jtag-axi"
$env:ZLC_PS_CLOCK_HZ = "{int(CLOCK_HZ)}"
$env:ZLC_PS_CHANNEL_COUNT = "{len(CHANNELS)}"

python -m Zou_lab_control.neutral_atom.devices.sequencer_server `
  --backend jtag-axi `
  --host {HOST} `
  --port {PORT} `
  --channels {" ".join(CHANNELS)} `
  --trigger-channels {" ".join(TRIGGER_CHANNELS)} `
  --clock-hz {CLOCK_HZ:g} `
  --state-dir "{STATE_DIR}"
""")

<!-- cell:markdown -->
## 4. Start the server

运行下面这个 cell 后它会一直阻塞。保持这个 notebook/kernel 不要关,然后去
控制电脑运行 `neutral_atom_hardware_quickstart.ipynb` 或打开 pulse GUI。

<!-- cell:code -->
na.run_sequencer_server(
    channels=CHANNELS,
    trigger_channels=TRIGGER_CHANNELS,
    host=HOST,
    port=PORT,
    clock_hz=CLOCK_HZ,
    state_dir=STATE_DIR,
    backend="jtag-axi",
)

<!-- cell:markdown -->
## 5. Optional: run the pulse GUI on this FPGA computer

`run_sequencer_server(...)` 会阻塞当前 kernel。要在 FPGA/Vivado 电脑本机打开
pulse GUI,请在另一个 PowerShell 或另一个 notebook kernel 里运行：

```powershell
.\pulse_gui.bat --remote-host 127.0.0.1 --remote-port 18861 --state .\pulses\camera_imaging_address_switch.json
```

GUI 仍然只是前端;实际 prepare/fire/wait 通过正在运行的 sequencer server
执行。默认 preset 只显示相机成像子集,但上传时仍是 full-width program。

First-light checklist on the FPGA computer:

```text
1. build_and_program.bat completed and programmed the bitstream.
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
Python code: the probe is on the correct board pin, the programmed bit/LTX
belongs to the current `fpga/build/ps` directory, Vivado Hardware
Manager sees a device rather than only a Digilent target, and the GUI is
connected to the running server rather than offline mode.

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
experiment pulse. Each GUI/API `prepare` uploads runtime metadata over JTAG-to-AXI
into block RAM (the host packs it with `fpga.pulse_streamer.host.image`):

```text
edge tables (3 parallel BRAMs, read in lockstep, 1-tick FIFO prefetch):
  tick[i]            absolute FPGA clock tick (base)
  coeffs[i][j]       per-slot affine coefficients (s0..sN)  -> effective_tick =
                     tick + (sum coeff_j * slot_j) >> frac
  mask[i]            full-width output state

scan window (2-bank ping-pong; total scan points are UNBOUNDED):
  bank 0 / bank 1    bank_size points each; the host streams later chunks behind
                     the engine cursor (bank_chunk handshake), and re-sweeps for
                     repeat_forever.

repeat metadata:
  repeat_forever, loop_start_addr, loop_end_tick, loop_count
```

One edge row is a complete state mask at a time point. If trap and emCCD change
at the same tick, that is still one row, not two. GUI hidden channels do not
change row width; they simply have zero bits in the uploaded masks.

A hardware scan binds duration / DAC-value fields to slots `s0..sN`; the
scan table is one row per scan point. Because the edge ticks are affine in the
slots, a scanned duration moves the edges (and any analog ramp) in lockstep.
(A channel delay is a fixed per-channel value and is not a scan slot.)
The compiler/host reject a scan only when it would make the merged edge order
non-monotonic at some scan point (split the scan or simplify timing).

Analog buses are compiled into a separate bus-segment memory with
`bus_id / start_tick / stop_tick / start_value / stop_value / mode / value_select /
stop_value_select` and affine start/stop tick coefficients. The FPGA generates
10-bit stair-step ramps locally, so dense bus ramps do not consume digital edge
rows; the dual value_select lets a ramp scan both its endpoints.
