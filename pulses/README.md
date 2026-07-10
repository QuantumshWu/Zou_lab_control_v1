# Pulse Presets

Checked-in `PulseTableState` JSON presets. These are frontend/API data, not
hardware projects. For the editing workflow see the **frontend manual**
(`docs/frontend_manual/`); for how a preset compiles and uploads see the **FPGA
manual** (`docs/fpga_manual/`).

## `camera_imaging_address_switch.json`

Default qCMOS camera-imaging preset for the address-switch FPGA pulse-streamer.

- The server derives one immutable `PortCatalog` from the address-switch XDC.
  The preset persists that catalog: raw lanes remain `ch00..ch61` in FPGA bit
  order, while the GUI/API expose logical digital, DAC, and clock ports. The GUI
  shows four programmable ports by default:
  - `ch09 trap (M17)`, `ch00 cooling (F15)`, `ch03 probe (N15)`, `ch11 emCCD (M13)`.
- The qCMOS/external camera trigger is `emCCD` (`ch11/M13`). `ch06/trig/R17`
  exists in the XDC but is not the checked-in camera trigger.
- Hidden or unconfigured channels stay off in the full-width upload.
- DAC ports `da_dipole`, `da_bias_x`, `da_bias_y`, `da_bias_z` own their member
  lanes, runtime bus index, width, encoding, safe code, and latch-clock topology
  in that single catalog. A pulse document cannot relabel lanes or toggle clocks.

Typical control-computer workflow:

```python
import Zou_lab_control.neutral_atom as na

exp = na.connect("remote_template", sequencer={"host": "192.168.0.20"})
pulse = exp.timing.bind_pulse("pulses/camera_imaging_address_switch.json")
pulse.on_pulse(wait=True, repeat_forever=False)
```

For a steady oscilloscope train use `pulse.on_pulse(wait=False,
repeat_forever=True)`; the whole table then repeats. For camera acquisition or
finite debugging keep `repeat_forever=False` so the sequencer can report done.

## Scanning (named slots)

Every scan slot has a semantic public name such as `exposure`, `da_x`, or
`t_off`. Internal tokens `s0`, `s1`, ... appear only inside the compiled pulse
expressions. Bind any period duration or DAC-port value to a slot, then provide an
`N_points x N_slots` scan table. A channel delay is a fixed per-channel output
delay set through the API; it is never scanned:

```python
state = na.PulseTableState.load("pulses/camera_imaging_address_switch.json")
slot = state.bind_field("duration", "1", name="exposure")
state.set_scan_table([[1_000], [2_000], [4_000]])  # 3 points, 1 slot, ns
program = state.compile_scan(clock_hz=50_000_000)
```

The host compiles to one affine edge template plus a streamed scan-point table;
the FPGA iterates the points seamlessly. Scan tables can also be loaded from
`.npy/.csv/.txt` (`state.load_scan_table(path)`), or built in the GUI Scan tab.

## Analog bus rows

The sequencer's `PortCatalog` exposes `da_dipole` as one logical GUI/API row;
its raw member lanes can never reappear as independent TTL controls. Each period
uses `Edge` (jump to the code at the period start),
`Ramp` (staircase from the previous anchor over the period), or `Hold`. A
10-bit offset-binary DAC is edited in signed LSB (`-512..511`, with `0` meaning
physical zero); the compiler alone converts it to the wire code. Buses
upload through a separate segment table, so a ramp costs one segment instead of
many TTL edge rows. The preview draws bus rows as hollow stair-step traces.
