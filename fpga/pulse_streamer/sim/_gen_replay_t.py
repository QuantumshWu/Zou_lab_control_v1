"""Generate replay_t.vh: the EXACT image words (real pack_program) for a T.json-STRUCTURED
program with scaled durations, so tb_t_ff.v can replay SAFE->upload->LOAD->FIRE through the
REAL top + loader and compare da_bias_y frame 0 vs frames 1+."""
import json, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.devices.sequencer import compile_runtime_program_for_payload
from fpga.pulse_streamer.host import image as img

payload = json.load(open(ROOT / "pulses" / "T.json", encoding="utf-8"))
# scale each period to a small tick count, structure preserved (9 periods)
ticks = [80, 15, 4, 2, 1, 5, 4, 3, 2]   # 116 ticks/frame
assert len(payload["periods"]) == len(ticks), f'{len(payload["periods"])} periods'
for p, t in zip(payload["periods"], ticks):
    p["duration"] = str(t * 20)
    p["unit"] = "ns"

state = na.PulseTableState.from_dict(payload)
prog = compile_runtime_program_for_payload(state, channels=list(state.channels), clock_hz=50e6)
print("repeat_forever=", prog.repeat_forever, " loop_end_tick=", prog.loop_end_tick)
print("edge ticks:", list(prog.ticks))
print("masks:", [hex(m) for m in prog.masks])
bn = list(prog.bus_names or [])
print("bus_names:", bn)
for s in (prog.bus_segments or []):
    print(f"  bus{s.bus_index}({bn[s.bus_index]}) start={s.start_tick} stop={s.stop_tick} "
          f"v={s.start_value}->{s.stop_value} mode={s.mode}")
print("clk_enable=", hex(prog.clk_enable or 0))

p = img.StreamerParams()
words = img.pack_program(prog, p)
print("words:", len(words), " frame_ticks:", prog.loop_end_tick)

out = pathlib.Path(__file__).with_name("replay_t.vh")
with open(out, "w") as f:
    for off in sorted(words):
        f.write(f"wr(30'd{off}, 32'h{words[off] & 0xFFFFFFFF:08x});\n")
print("wrote", out)

# also emit frame length for the TB
with open(pathlib.Path(__file__).with_name("replay_t_frame.vh"), "w") as f:
    f.write(f"localparam integer T_FRAME = {int(prog.loop_end_tick)};\n")
    f.write(f"localparam integer T_PROGCOUNT = {len(list(prog.ticks))};\n")
