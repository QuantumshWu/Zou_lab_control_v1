# Pulse Presets

This directory stores checked-in `PulseTableState` JSON presets.  They are
frontend/API data, not hardware projects.

## `camera_imaging_address_switch.json`

Default qCMOS camera-imaging preset for the address-switch FPGA
pulse-streamer.

- Hardware channels are inferred from the original address-switch XDC and saved
  as `ch00..ch61` in FPGA bit order.
- The GUI shows only four rows by default: `trap`, `cooling`, `probe`, and
  `emCCD`.
- In the default XDC, those rows are:
  - `ch09`: `trap`, package pin `M17`
  - `ch00`: `cooling`, package pin `F15`
  - `ch03`: `probe`, package pin `N15`
  - `ch11`: `emCCD`, package pin `M13`
- The qCMOS/external camera trigger for this preset is `emCCD` (`ch11/M13`).
  The XDC still contains `ch06/trig/R17`, but that is not the checked-in camera
  trigger pulse.
- Hidden or unconfigured channels stay off when uploaded to the full-width FPGA
  mask.
- `camera_exposure` uses `duration="camera_exposure_ns"` and
  `unit="str (ns)"`.
- The default `camera_exposure_ns` value is `19980000`, so the preset starts as
  the original 19.98 ms exposure segment. Changing
  `pulse.set_variable("camera_exposure_ns", value_ns)` in a notebook changes
  this readout/probe exposure.
- The preset has no inner finite repeat bracket.  Its table head keeps `trig`
  low, so it does not create a slow table-boundary camera-trigger spike.
- The saved JSON includes logical analog buses inferred from the XDC:
  `da_dipole`, `da_bias_x`, `da_bias_y`, and `da_bias_z`.

Typical control-computer workflow:

```python
import Zou_lab_control.neutral_atom as na

exp = na.connect("remote_template", sequencer={"host": "192.168.0.20"})
pulse = exp.timing.bind_pulse("pulses/camera_imaging_address_switch.json")

pulse.set_variable("camera_exposure_ns", 2_000_000)  # ns
pulse.on_pulse(wait=True, repeat_forever=False)

scan = exp.readout.detection_time(
    [2e-3, 4e-3, 8e-3],
    shots=30,
    pulse=pulse,
    live=False,
    display=True,
)
```

For repeated oscilloscope output from the API use:

```python
pulse.on_pulse(wait=False, repeat_forever=True)
```

In that mode the whole table repeats, so the intended `ch11/emCCD` pulse appears
once per table period.  For camera acquisition or finite debugging, keep
`repeat_forever=False` so the sequencer can report done.

## Named Scan Tables

For scan-style pulses, keep period duration, per-channel delay, or analog-bus
value expressions symbolic with named parameters. Link one ordered scan table
file whose columns match those names:

```text
# vars: camera_exposure_ns(ns), trig_delay(ns)
1000 0
2000 20
4000 40
```

```python
pulse.set_scan_table_path("scan_points.txt")
pulse.on_pulse(wait=False)
```

The checked-in `camera_exposure_scan_example.txt` is a runnable camera-exposure
scan file for `camera_imaging_address_switch.json`; link it in the GUI `File`
field or call `pulse.set_scan_table_path("pulses/camera_exposure_scan_example.txt")`.
It includes both `camera_exposure_ns` and `p4_duration` with identical values
so it works with the preset binding and with a manual fourth-period duration
dot binding.

The GUI preview keeps symbolic segments such as `camera_exposure_ns` and
`100000-camera_exposure_ns` visible instead of expanding every scan point into
separate columns. The compact Artix-7 35T profile accepts at most five active
scan parameters per prepared program, with no more than two timing parameters
per FPGA chunk. Static analog/DAC edge values can scan through the packed
per-row bus-value RAM. If a linked scan file has more rows than the bitstream's
single-program scan RAM, `pulse.on_pulse(wait=True, repeat_forever=False)` can
run it as consecutive chunks. Ramp-mode DA scans still require preparing one
ramp pulse per scan row.

## Analog Bus Rows

XDC labels such as `da_dipole[0] ... da_dipole[9]` are folded into one logical
GUI/API row named `da_dipole`.  Each period can use:

- `Edge`: jump to the edited integer code at the period start.
- `Ramp`: linearly staircase from the previous numeric anchor to the edited
  code over the period.
- `Hold`: keep the previous value, or keep following the active ramp.

The code range is limited by the bus width; a 10-bit bus accepts `0..1023`.
The edit field is a line edit, not a spinbox, and the GUI clamps typed values to
the allowed range.  Digital bit channels are still uploaded as TTL mask bits,
but the GUI preview draws buses as one hollow/stair-step analog trace instead
of filled digital blocks or extra range labels.

The current runtime core still realizes a ramp by changing the underlying TTL
bus value over time, so a dense ramp can consume many edge rows.  The intended
scalable upgrade is a separate analog-bus segment table:

```text
bus_id, start_tick, stop_tick, start_value, stop_value, mode
```

The FPGA would then generate bus stair steps locally and keep the digital edge
table for laser/shutter/camera transitions.
