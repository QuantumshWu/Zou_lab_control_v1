# Device configs

Each file here is a named device config `na.connect("<name>")` / `na.load_devices("<name>")` loads
(bare name resolves from this folder, with or without `.json`). One JSON object, one entry per device
**role**; each entry is `{"type": "<DeviceClass>", "params": {…}}`. The role name (`camera`,
`sequencer`, `trap_array`, `monitor_camera`, a future `rf`, …) is what measurements/tasks select from
in their **Camera dropdown** and what `exp.device_manager()` groups by. `"virtual"` is a built-in
string shortcut built **programmatically** from the device-class defaults (there is no `virtual.json`
on disk) and needs no hardware — `load_devices("virtual")` always takes that in-code path.

Point `na.discover_devices()` at your hardware to get a ready `{role: {type, params}}` fragment to
paste here, and open `exp.device_manager()` (or `na.<session>.device_manager()`) to see what a config
loaded. A device is only constructed at load; it does not touch hardware until first use, so a bad
type/param fails **loudly** at `load_devices`, and a wrong host/serial fails at first `open()`.

## Per-device params (units + gotchas)

**`QCMOSCamera`** — params nest under a single `config` object (unlike the flat sequencer params),
holding the `QCMOSConfig` fields:
- `exposure` — seconds.
- `roi` — `[x, width, y, height]` in sensor pixels (NOT x0,x1,y0,y1). Omit / `null` = full sensor.
- `readout_speed` — DCAM enum int (`1` = the standard mode). `sensor_mode` / `trigger_global_exposure`
  are raw DCAMPROP enum ints; leave unset to keep the camera's power-on default (area rolling shutter).
- `device_index` — which attached qCMOS (0 = first) when several are on the DCAM bus.
- `capture_trigger_channels` — the sequencer line the camera's external-trigger input is physically
  wired to (e.g. `["ch11"]`). **Must match the real TTL cable**; the imaging pulse is built to pulse
  THIS line. Metadata only — the camera never drives the sequencer.

**`PylonCamera`** (`basler_monitor.json`) — `serial: ""` = first Basler found (ambiguous with two on
the bench; pin the serial from `na.discover_devices()`). `trigger_source: "Software"` = free-running
grab (no pulse needed), the natural mode for a MOT monitor viewer; any other value is the Basler
hardware trigger line name (e.g. `"Line1"`). `capture_trigger_channels` — as for the qCMOS above:
the sequencer line the camera's trigger input is physically wired to. Inert in `"Software"` mode
(nothing counts edges for a free-running grab); **required to match the real cable in a
hardware-trigger config**, or multi-frame acquisitions count edges on the wrong line.

**`RemoteSequencer`** (`remote_template.json`) — set `host` to your FPGA/Vivado server's address
(the placeholder `"FPGA_SERVER_IP"` fails clearly until replaced); `port` must match `run_server`.
`ManualSequencer` (`manual_template.json`) prints its `message` and waits for you to arm the FPGA
by hand — no server needed.

## Notes

- **`manual_template` / `remote_template` omit `trap_array` by design** — tweezer site centres come
  from running sitemap **calibration** (`exp.readout.sitemap()` / the Calibrate-readout task), not
  from a device. Load one, then calibrate before an occupancy readout, or you get a "no sitemap" error.
- Unknown **param** keys are rejected (a typo raises `TypeError` at load); unknown top-level device
  roles are tolerated. Keep comments here in this README, not as `_comment` keys in the JSON.
- A camera that images single atoms in tweezers (readout / survival / fidelity) must be the science
  camera; a MOT `monitor_camera` cannot. The temperature/fidelity measurements are pinned to the
  readout camera on purpose (no dropdown); `pulse_scan` / camera-live / MOT-field expose the choice.
- **`camera` vs `monitor_camera` are NOT two device types — both are `CameraDevice`, one domain.**
  The names are just config keys for two physical sensors (a readout qCMOS wired to the emCCD line;
  a MOT viewer wired to `mot_trigger`). Every camera-using form lists BOTH in its Camera dropdown.
  Convention: name your primary/readout camera `camera` — `default_camera_name()` / `exp.camera`
  resolve to that name when present, else to the first camera by name (no crash, but the default may
  then not be the one you meant, so with two unconventionally-named cameras, name one `camera` or pick
  it in the dropdown). A single-camera lab just has `camera` and never sees the distinction.
