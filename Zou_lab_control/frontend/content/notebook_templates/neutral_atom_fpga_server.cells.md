<!-- cell:markdown -->
# Neutral atom FPGA/Vivado sequencer server

这个 notebook 在 Verilog/FPGA 电脑上运行。它启动一个新的 `Zou_lab_control.neutral_atom` sequencer server，等待控制电脑上的 `RemoteSequencer` 连接。

它不调用 `PythonCamDemo` 或旧控制代码接口。新的边界是：

```text
control computer RemoteSequencer
  -> RPyC
  -> SequencerService on FPGA/Vivado computer
  -> device-layer command backend
  -> legacy_address_switch Vivado/VIO adapter or a future pulse-table FPGA backend
```

如果你不想用 Jupyter，也可以在 Verilog 电脑的 PowerShell 中运行本 notebook 第 3 节给出的命令行。

<!-- cell:code -->
{{BOOTSTRAP_CELL}}

<!-- cell:code -->
import os
from pathlib import Path
import json
import sys

import Zou_lab_control.frontend as zf
import Zou_lab_control.neutral_atom as na

zf.enable_long_output()

<!-- cell:markdown -->
## 1. Configure this FPGA/Vivado computer

把下面路径改成 Verilog 电脑上的真实路径。这里默认接的是当前 `address_switch` 那种固定状态机 bitstream：`prepare` 根据 control 电脑传来的 `RuntimeSequenceProgram` 计算 `pulse_lasting` 和 `cycle_counts`，通过 Vivado/VIO 写进 FPGA；`fire` 再把 `config_ready` 拉高启动序列。也就是说 scan readout time/fidelity 时，每个 detection time 都会重新写 probe pulse width。

如果你的 bitstream 已经换成真正的 pulse-table runtime，后面只需要换 `PREPARE_COMMAND/FIRE_COMMAND` 后面的 backend；control 电脑的 notebook 不应该变。

<!-- cell:code -->
PROJECT_ROOT = Path("..").resolve()

HOST = "0.0.0.0"
PORT = 18861
CHANNELS = ["trap", "cooling", "probe", "qcm_trigger"]
TRIGGER_CHANNELS = ["qcm_trigger"]
CLOCK_HZ = 100_000_000.0  # address_switch tick clock; change if your FPGA state machine uses another clk

STATE_DIR = Path(r"D:\zlc_sequencer_state")

os.environ["ZLC_VIVADO_PROJECT"] = r"D:\zlc_fpga\neutral_atom_sequence\neutral_atom_sequence.xpr"
os.environ["ZLC_VIVADO_BIT"] = r"D:\zlc_fpga\neutral_atom_sequence\neutral_atom_sequence.runs\impl_1\main.bit"
os.environ["ZLC_VIVADO_LTX"] = r"D:\zlc_fpga\neutral_atom_sequence\neutral_atom_sequence.runs\impl_1\main.ltx"
os.environ["ZLC_VIVADO_PROGRAM_ON_RUN"] = "0"  # set to "1" only when you intentionally want to reprogram the bitstream
os.environ["ZLC_VIO_FILTER"] = 'CELL_NAME=~"vio"'

# Probe names in address_switch/address_switch.srcs/sources_1/new/main.v.
os.environ["ZLC_LEGACY_START_PARAM"] = "config_ready"
os.environ["ZLC_LEGACY_DEBUG_PARAM"] = "debug"
os.environ["ZLC_LEGACY_PULSE_PARAM"] = "pulse_lasting"
os.environ["ZLC_LEGACY_CYCLE_PARAM"] = "cycle_counts"
os.environ["ZLC_LEGACY_PROBE_CHANNEL"] = "probe"
os.environ["ZLC_LEGACY_WAIT_FOR_DURATION"] = "1"

# Keep this at "0" until an oscilloscope confirms that the qCMOS trigger input
# sees exactly one positive edge per address-switch cycle in run mode.  The
# original address_switch Verilog can produce two emCCD/readout pulses per cycle
# and does not explicitly drive trig in run mode.
os.environ["ZLC_LEGACY_SINGLE_CAMERA_TRIGGER_CONFIRMED"] = "0"

# Optional defaults for other legacy VIO probes. Fill these with known-good
# values for your current experiment; they are written during prepare before
# pulse_lasting/cycle_counts are written.
LEGACY_VIO_DEFAULTS = {
    # "PGC_lasting": 0,
    # "PGC_waiting": 0,
    # "probe_waiting": 0,
    # "CCD_waiting": 0,
}
os.environ["ZLC_LEGACY_VIO_DEFAULTS"] = json.dumps(LEGACY_VIO_DEFAULTS)

PREPARE_COMMAND = f'"{sys.executable}" -m Zou_lab_control.neutral_atom.devices.legacy_address_switch prepare'
FIRE_COMMAND = f'"{sys.executable}" -m Zou_lab_control.neutral_atom.devices.legacy_address_switch fire'
WAIT_DONE_COMMAND = f'"{sys.executable}" -m Zou_lab_control.neutral_atom.devices.legacy_address_switch wait_done'
SAFE_STATE_COMMAND = f'"{sys.executable}" -m Zou_lab_control.neutral_atom.devices.legacy_address_switch safe_state'

PREPARE_COMMAND, FIRE_COMMAND, WAIT_DONE_COMMAND, SAFE_STATE_COMMAND

<!-- cell:markdown -->
## 2. PowerShell command equivalent

在 Verilog 电脑上也可以不用 Jupyter，直接在 PowerShell 运行：

<!-- cell:code -->
print(fr"""
cd "{PROJECT_ROOT}"
$env:PYTHONPATH = (Get-Location).Path
$env:ZLC_VIVADO_PROJECT = "{os.environ["ZLC_VIVADO_PROJECT"]}"
$env:ZLC_VIVADO_BIT = "{os.environ["ZLC_VIVADO_BIT"]}"
$env:ZLC_VIVADO_LTX = "{os.environ["ZLC_VIVADO_LTX"]}"
$env:ZLC_VIVADO_PROGRAM_ON_RUN = "{os.environ["ZLC_VIVADO_PROGRAM_ON_RUN"]}"

$env:ZLC_VIO_FILTER = '{os.environ["ZLC_VIO_FILTER"]}'
$env:ZLC_LEGACY_START_PARAM = "{os.environ["ZLC_LEGACY_START_PARAM"]}"
$env:ZLC_LEGACY_DEBUG_PARAM = "{os.environ["ZLC_LEGACY_DEBUG_PARAM"]}"
$env:ZLC_LEGACY_PULSE_PARAM = "{os.environ["ZLC_LEGACY_PULSE_PARAM"]}"
$env:ZLC_LEGACY_CYCLE_PARAM = "{os.environ["ZLC_LEGACY_CYCLE_PARAM"]}"
$env:ZLC_LEGACY_PROBE_CHANNEL = "{os.environ["ZLC_LEGACY_PROBE_CHANNEL"]}"
$env:ZLC_LEGACY_WAIT_FOR_DURATION = "{os.environ["ZLC_LEGACY_WAIT_FOR_DURATION"]}"
$env:ZLC_LEGACY_SINGLE_CAMERA_TRIGGER_CONFIRMED = "{os.environ["ZLC_LEGACY_SINGLE_CAMERA_TRIGGER_CONFIRMED"]}"
$env:ZLC_LEGACY_VIO_DEFAULTS = '{os.environ["ZLC_LEGACY_VIO_DEFAULTS"]}'

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
## 3. Start the server

运行下面这个 cell 后它会一直阻塞，这是正常的：它在等待控制电脑连接。保持这个 notebook/kernel 不要关，然后去控制电脑运行 `neutral_atom_hardware_quickstart.ipynb`。

如果你只是想检查参数，不要运行这个 cell。

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
