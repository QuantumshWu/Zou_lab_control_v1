# Build the FINAL affine edge-table pulse streamer (zlc_pulse_streamer_top +
# zlc_edge_streamer): BRAM edge/scan tables + 1-tick FIFO prefetch + 2-bank
# streaming scan, JTAG-to-AXI control.  ONE clean build (no variants).
#
# 35T solved geometry (host.image.solve_capacity, <=90% target): 4096 edges +
# bank_size 2048 (4096 resident scan points) + UNBOUNDED streaming @ 78% RAMB36.
#
# *** The engine + control FSM are PROVEN by cycle-accurate Python models
# (engine_model.rtl_mirror_play == reference at read latency 1/2/3 incl. 1-tick;
# streaming_scan_play gapless/stall) and contract tests, but the multi-BRAM AXI
# integration is BLIND (no Verilog sim in repo) and needs on-board bring-up.  IP
# property names are version-specific: each is set defensively (zlc_try warns,
# does not abort); the real CONFIG.* are dumped -- grep "ZLC IPDUMP"/"ZLC
# TRY-FAIL".  CRITICAL: the 3 edge BRAMs are forced to READ_LATENCY_B = 2 (both
# port-B output registers) so the engine's RD_LAT=2 prefetch is deterministic;
# the dump MUST show latency 2 or 1-tick playback will be off-by-cycles. ***
#
# Geometry MUST match zlc_pulse_streamer_top.v localparams AND host.image:
#   EDGE_ADDR_WIDTH=12 (4096 edges):
#     tick  BRAM 32b/32b  depth 4096
#     coeff BRAM 32b(A)/64b(B)  port-A depth 8192, port-B depth 4096
#     mask  BRAM 32b(A)/64b(B)  port-A depth 8192, port-B depth 4096
#   BANK_SIZE=2048 -> scan depth 2*2048=4096:
#     scan  BRAM 32b(A)/128b(B) port-A depth 16384, port-B depth 4096
#   bus image 256*7=1792 words; CTRL 64; total 38720 -> axi_bram depth 65536.
#   There is NO lane/delay image BRAM: the per-channel OUTPUT delay is held CTRL scalars
#   (delay_count/bits/off/skip) and an engine-internal distributed-RAM/SRL ring (+0 RAMB36).

set script_dir [file normalize [file dirname [info script]]]
proc env_or {name default} {
    if {[info exists ::env($name)] && $::env($name) ne ""} { return [file normalize $::env($name)] }
    return $default
}
proc zlc_default_project_root {script_dir} {
    if {[info exists ::env(ZLC_PS_PROJECT_ROOT)] && $::env(ZLC_PS_PROJECT_ROOT) ne ""} {
        return [file normalize $::env(ZLC_PS_PROJECT_ROOT)]
    }
    return [file normalize [file join $script_dir .. build]]
}
# Vivado writes a debug-core temp dir (<project>/<name>.runs/impl_1/.Xil/
# Vivado-PID-HOST) during implementation; files created under it must stay below
# the Windows MAX_PATH (260).  There is NO Vivado knob to relocate just that temp
# dir, so the only fix is a SHORT project path.  The build deliberately stays in
# fpga/build (in-repo, tracked, no extra drive/junction, cross-platform clean) and
# uses the SHORT project name "ps" (-> ps.runs) to keep the run path well under the
# limit even on a deep checkout (e.g. .../Zou_lab_control_v1/fpga/build/ps -> ~141
# chars here vs ~165 with the old long name).  If even that is too long, the repo
# itself is checked out too deep -- shorten the CHECKOUT location (keep the build
# in fpga/), do not split the build out.
proc zlc_debug_tmp_path {dir project_name} {
    return [file normalize [file join $dir ${project_name}.runs impl_1 .Xil Vivado-00000-QuantumPad]]
}
proc zlc_safe_project_dir {project_dir script_dir project_name} {
    set out [file normalize $project_dir]
    set debug_tmp [zlc_debug_tmp_path $out $project_name]
    if {[string length $debug_tmp] > 146} {
        error "Vivado debug-core temp path too long ($debug_tmp).\n  The build must stay under fpga/build, so the REPO is checked out too deep.\n  Check out the repo at a shorter path (e.g. C:/src/zlc) and rebuild, or set ZLC_PS_PROJECT_DIR to a shorter in-repo dir."
    }
    return $out
}

set xdc_path [env_or ZLC_PS_XDC [file join $script_dir .. .. references source_archives address_switch address_switch.srcs constrs_1 new addre.xdc]]
if {![file exists $xdc_path]} { error "board XDC not found: $xdc_path. Set ZLC_PS_XDC." }
set xdc_file [open $xdc_path r]; set xdc_text [read $xdc_file]; close $xdc_file
if {[string match "*<PIN_CH*" $xdc_text]} { error "$xdc_path still has <PIN_CHxx> placeholders." }

# SHORT project name/subdir (ps) keeps Vivado's deep run/.Xil temp path under the
# Windows MAX_PATH limit while staying in fpga/build (see zlc_safe_project_dir).
set project_root [zlc_default_project_root $script_dir]
set project_name ps
set project_dir [zlc_safe_project_dir [env_or ZLC_PS_PROJECT_DIR [file join $project_root ps]] $script_dir $project_name]
set top zlc_pulse_streamer_top
set part xc7a35tfgg484-2

# Geometry (single source: keep in sync with the top + host.image.solve_capacity).
set zlc_edge_addr_width 12
set zlc_bank_size 2048
set zlc_scan_depth [expr {2 * $zlc_bank_size}]
set zlc_max_edges [expr {1 << $zlc_edge_addr_width}]
set zlc_coeff_portb_bits 64
set zlc_mask_portb_bits 64
set zlc_scan_portb_bits 128
set zlc_coeff_porta_depth [expr {$zlc_max_edges * ($zlc_coeff_portb_bits / 32)}]
set zlc_mask_porta_depth  [expr {$zlc_max_edges * ($zlc_mask_portb_bits / 32)}]
set zlc_scan_porta_depth  [expr {$zlc_scan_depth * ($zlc_scan_portb_bits / 32)}]
set zlc_axi_bram_depth 65536

puts "ZLC create_project: FINAL engine (1-tick FIFO prefetch + 2-bank streaming), 4096 edges + bank 2048"
puts "ZLC create_project project_dir: $project_dir"

proc zlc_require_run_complete {run_name expected_status} {
    set status [get_property STATUS [get_runs $run_name]]
    if {[string first $expected_status $status] < 0} {
        error "Vivado run $run_name did not complete. STATUS='$status', expected '$expected_status'."
    }
}
proc zlc_try {label body} {
    if {[catch {uplevel 1 $body} zlc_e]} { puts "ZLC TRY-FAIL ($label): $zlc_e"; return 0 }
    return 1
}
proc zlc_dump_ip {ip} {
    puts "ZLC IPDUMP ===== $ip ====="
    foreach p [lsort [list_property [get_ips $ip]]] {
        if {[string match CONFIG.* $p]} { puts "ZLC IPDUMP   $p = [get_property $p [get_ips $ip]]" }
    }
    puts "ZLC IPDUMP ===== end $ip ====="
}
# force port-B read latency 2 (deterministic) on an edge BRAM; verify the dump.
proc zlc_force_latency2 {ip} {
    zlc_try "$ip regPrimB" {set_property CONFIG.Register_PortB_Output_of_Memory_Primitives {true} [get_ips $ip]}
    zlc_try "$ip regCoreB" {set_property CONFIG.Register_PortB_Output_of_Memory_Core {true} [get_ips $ip]}
}

if {[file exists $project_dir]} { file delete -force $project_dir }
file mkdir [file dirname $project_dir]
create_project $project_name $project_dir -part $part -force
set_property target_language Verilog [current_project]

read_verilog [file join $script_dir zlc_edge_streamer.v]
read_verilog [file join $script_dir zlc_pulse_streamer_top.v]
read_xdc $xdc_path
set_property top $top [current_fileset]

# --- jtag_axi master ------------------------------------------------------
# FULL AXI4 (not AXI4-Lite) so the host can issue INCR burst writes -- one
# run_hw_axi moves up to 256 words instead of one, turning a multi-second BRAM
# upload into a ~100 ms one.  PROTOCOL is a string property; Vivado recomputes
# the burst/lock/cache/qos interface flags from it.  ID width 1 (1-bit awid..rid).
create_ip -name jtag_axi -vendor xilinx.com -library ip -module_name jtag_axi_0
zlc_try "jtag PROTOCOL=AXI4"  {set_property CONFIG.PROTOCOL {AXI4} [get_ips jtag_axi_0]}
zlc_try "jtag ID=1" {set_property CONFIG.M_AXI_ID_WIDTH {1} [get_ips jtag_axi_0]}
zlc_try "jtag DATA=32" {set_property CONFIG.M_AXI_DATA_WIDTH {32} [get_ips jtag_axi_0]}
zlc_try "jtag ADDR=32" {set_property CONFIG.M_AXI_ADDR_WIDTH {32} [get_ips jtag_axi_0]}
zlc_dump_ip jtag_axi_0
generate_target all [get_ips jtag_axi_0]

# --- AXI BRAM controller (single; the top decodes its BRAM port to 5 BRAMs) ---
create_ip -name axi_bram_ctrl -vendor xilinx.com -library ip -module_name axi_bram_ctrl_0
zlc_try "bramc DATA=32"      {set_property CONFIG.DATA_WIDTH {32} [get_ips axi_bram_ctrl_0]}
zlc_try "bramc SINGLE_PORT"  {set_property CONFIG.SINGLE_PORT_BRAM {1} [get_ips axi_bram_ctrl_0]}
# FULL AXI4 with ID width matching the master (1) so it accepts INCR burst writes.
zlc_try "bramc PROTOCOL=AXI4" {set_property CONFIG.PROTOCOL {AXI4} [get_ips axi_bram_ctrl_0]}
zlc_try "bramc ID=1"        {set_property CONFIG.ID_WIDTH {1} [get_ips axi_bram_ctrl_0]}
zlc_try "bramc narrow=0"    {set_property CONFIG.SUPPORTS_NARROW_BURST {0} [get_ips axi_bram_ctrl_0]}
zlc_try "bramc readlat=1"   {set_property CONFIG.READ_LATENCY {1} [get_ips axi_bram_ctrl_0]}
zlc_try "bramc BMG=EXTERNAL" {set_property CONFIG.BMG_INSTANCE {EXTERNAL} [get_ips axi_bram_ctrl_0]}
set_property CONFIG.MEM_DEPTH $zlc_axi_bram_depth [get_ips axi_bram_ctrl_0]
if {[get_property CONFIG.MEM_DEPTH [get_ips axi_bram_ctrl_0]] != $zlc_axi_bram_depth} {
    error "axi_bram_ctrl MEM_DEPTH did not take (want $zlc_axi_bram_depth)."
}
zlc_dump_ip axi_bram_ctrl_0
generate_target all [get_ips axi_bram_ctrl_0]

# --- EDGE TICK BRAM: symmetric 32b TDP, depth 4096, forced port-B latency 2 ----
create_ip -name blk_mem_gen -vendor xilinx.com -library ip -module_name blk_mem_gen_edge_tick
zlc_try "tick TDP"      {set_property CONFIG.Memory_Type {True_Dual_Port_RAM} [get_ips blk_mem_gen_edge_tick]}
zlc_try "tick ByteWE"   {set_property CONFIG.Use_Byte_Write_Enable {true} [get_ips blk_mem_gen_edge_tick]}
zlc_try "tick ByteSize8" {set_property CONFIG.Byte_Size {8} [get_ips blk_mem_gen_edge_tick]}
zlc_try "tick WWA=32"   {set_property CONFIG.Write_Width_A {32} [get_ips blk_mem_gen_edge_tick]}
zlc_try "tick RWA=32"   {set_property CONFIG.Read_Width_A {32} [get_ips blk_mem_gen_edge_tick]}
zlc_try "tick WDA"      {set_property CONFIG.Write_Depth_A $zlc_max_edges [get_ips blk_mem_gen_edge_tick]}
zlc_try "tick WWB=32"   {set_property CONFIG.Write_Width_B {32} [get_ips blk_mem_gen_edge_tick]}
zlc_try "tick RWB=32"   {set_property CONFIG.Read_Width_B {32} [get_ips blk_mem_gen_edge_tick]}
zlc_try "tick ENA"      {set_property CONFIG.Enable_A {Use_ENA_Pin} [get_ips blk_mem_gen_edge_tick]}
zlc_try "tick ENB"      {set_property CONFIG.Enable_B {Use_ENB_Pin} [get_ips blk_mem_gen_edge_tick]}
zlc_try "tick noRSTA"   {set_property CONFIG.Use_RSTA_Pin {false} [get_ips blk_mem_gen_edge_tick]}
zlc_try "tick noRSTB"   {set_property CONFIG.Use_RSTB_Pin {false} [get_ips blk_mem_gen_edge_tick]}
zlc_force_latency2 blk_mem_gen_edge_tick
zlc_dump_ip blk_mem_gen_edge_tick
generate_target all [get_ips blk_mem_gen_edge_tick]

# --- EDGE COEFF BRAM: asymmetric 32b(A)/64b(B), forced port-B latency 2 --------
create_ip -name blk_mem_gen -vendor xilinx.com -library ip -module_name blk_mem_gen_edge_coeff
zlc_try "coeff TDP"     {set_property CONFIG.Memory_Type {True_Dual_Port_RAM} [get_ips blk_mem_gen_edge_coeff]}
zlc_try "coeff ByteWE"  {set_property CONFIG.Use_Byte_Write_Enable {true} [get_ips blk_mem_gen_edge_coeff]}
zlc_try "coeff ByteSize8" {set_property CONFIG.Byte_Size {8} [get_ips blk_mem_gen_edge_coeff]}
zlc_try "coeff WWA=32"  {set_property CONFIG.Write_Width_A {32} [get_ips blk_mem_gen_edge_coeff]}
zlc_try "coeff RWA=32"  {set_property CONFIG.Read_Width_A {32} [get_ips blk_mem_gen_edge_coeff]}
zlc_try "coeff WDA"     {set_property CONFIG.Write_Depth_A $zlc_coeff_porta_depth [get_ips blk_mem_gen_edge_coeff]}
zlc_try "coeff WWB=64"  {set_property CONFIG.Write_Width_B $zlc_coeff_portb_bits [get_ips blk_mem_gen_edge_coeff]}
zlc_try "coeff RWB=64"  {set_property CONFIG.Read_Width_B $zlc_coeff_portb_bits [get_ips blk_mem_gen_edge_coeff]}
zlc_try "coeff ENA"     {set_property CONFIG.Enable_A {Use_ENA_Pin} [get_ips blk_mem_gen_edge_coeff]}
zlc_try "coeff ENB"     {set_property CONFIG.Enable_B {Use_ENB_Pin} [get_ips blk_mem_gen_edge_coeff]}
zlc_try "coeff noRSTA"  {set_property CONFIG.Use_RSTA_Pin {false} [get_ips blk_mem_gen_edge_coeff]}
zlc_try "coeff noRSTB"  {set_property CONFIG.Use_RSTB_Pin {false} [get_ips blk_mem_gen_edge_coeff]}
zlc_force_latency2 blk_mem_gen_edge_coeff
zlc_dump_ip blk_mem_gen_edge_coeff
generate_target all [get_ips blk_mem_gen_edge_coeff]

# --- EDGE MASK BRAM: asymmetric 32b(A)/64b(B), forced port-B latency 2 ---------
create_ip -name blk_mem_gen -vendor xilinx.com -library ip -module_name blk_mem_gen_edge_mask
zlc_try "mask TDP"      {set_property CONFIG.Memory_Type {True_Dual_Port_RAM} [get_ips blk_mem_gen_edge_mask]}
zlc_try "mask ByteWE"   {set_property CONFIG.Use_Byte_Write_Enable {true} [get_ips blk_mem_gen_edge_mask]}
zlc_try "mask ByteSize8" {set_property CONFIG.Byte_Size {8} [get_ips blk_mem_gen_edge_mask]}
zlc_try "mask WWA=32"   {set_property CONFIG.Write_Width_A {32} [get_ips blk_mem_gen_edge_mask]}
zlc_try "mask RWA=32"   {set_property CONFIG.Read_Width_A {32} [get_ips blk_mem_gen_edge_mask]}
zlc_try "mask WDA"      {set_property CONFIG.Write_Depth_A $zlc_mask_porta_depth [get_ips blk_mem_gen_edge_mask]}
zlc_try "mask WWB=64"   {set_property CONFIG.Write_Width_B $zlc_mask_portb_bits [get_ips blk_mem_gen_edge_mask]}
zlc_try "mask RWB=64"   {set_property CONFIG.Read_Width_B $zlc_mask_portb_bits [get_ips blk_mem_gen_edge_mask]}
zlc_try "mask ENA"      {set_property CONFIG.Enable_A {Use_ENA_Pin} [get_ips blk_mem_gen_edge_mask]}
zlc_try "mask ENB"      {set_property CONFIG.Enable_B {Use_ENB_Pin} [get_ips blk_mem_gen_edge_mask]}
zlc_try "mask noRSTA"   {set_property CONFIG.Use_RSTA_Pin {false} [get_ips blk_mem_gen_edge_mask]}
zlc_try "mask noRSTB"   {set_property CONFIG.Use_RSTB_Pin {false} [get_ips blk_mem_gen_edge_mask]}
zlc_force_latency2 blk_mem_gen_edge_mask
zlc_dump_ip blk_mem_gen_edge_mask
generate_target all [get_ips blk_mem_gen_edge_mask]

# --- SCAN BRAM: asymmetric 32b(A)/128b(B), 2*BANK_SIZE deep -------------------
create_ip -name blk_mem_gen -vendor xilinx.com -library ip -module_name blk_mem_gen_scan
zlc_try "scan TDP"      {set_property CONFIG.Memory_Type {True_Dual_Port_RAM} [get_ips blk_mem_gen_scan]}
zlc_try "scan ByteWE"   {set_property CONFIG.Use_Byte_Write_Enable {true} [get_ips blk_mem_gen_scan]}
zlc_try "scan ByteSize8" {set_property CONFIG.Byte_Size {8} [get_ips blk_mem_gen_scan]}
zlc_try "scan WWA=32"   {set_property CONFIG.Write_Width_A {32} [get_ips blk_mem_gen_scan]}
zlc_try "scan RWA=32"   {set_property CONFIG.Read_Width_A {32} [get_ips blk_mem_gen_scan]}
zlc_try "scan WDA"      {set_property CONFIG.Write_Depth_A $zlc_scan_porta_depth [get_ips blk_mem_gen_scan]}
zlc_try "scan WWB=128"  {set_property CONFIG.Write_Width_B $zlc_scan_portb_bits [get_ips blk_mem_gen_scan]}
zlc_try "scan RWB=128"  {set_property CONFIG.Read_Width_B $zlc_scan_portb_bits [get_ips blk_mem_gen_scan]}
zlc_try "scan ENA"      {set_property CONFIG.Enable_A {Use_ENA_Pin} [get_ips blk_mem_gen_scan]}
zlc_try "scan ENB"      {set_property CONFIG.Enable_B {Use_ENB_Pin} [get_ips blk_mem_gen_scan]}
zlc_try "scan noRSTA"   {set_property CONFIG.Use_RSTA_Pin {false} [get_ips blk_mem_gen_scan]}
zlc_try "scan noRSTB"   {set_property CONFIG.Use_RSTB_Pin {false} [get_ips blk_mem_gen_scan]}
zlc_force_latency2 blk_mem_gen_scan
zlc_dump_ip blk_mem_gen_scan
generate_target all [get_ips blk_mem_gen_scan]

# --- BUS image BRAM: symmetric 32b TDP (A=AXI write, B=mini-loader read) ------
create_ip -name blk_mem_gen -vendor xilinx.com -library ip -module_name blk_mem_gen_busimg
zlc_try "busimg TDP"    {set_property CONFIG.Memory_Type {True_Dual_Port_RAM} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg ByteWE" {set_property CONFIG.Use_Byte_Write_Enable {true} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg ByteSize8" {set_property CONFIG.Byte_Size {8} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg WWA=32" {set_property CONFIG.Write_Width_A {32} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg WDA=2048" {set_property CONFIG.Write_Depth_A {2048} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg ENA"    {set_property CONFIG.Enable_A {Use_ENA_Pin} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg ENB"    {set_property CONFIG.Enable_B {Use_ENB_Pin} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg noRSTA" {set_property CONFIG.Use_RSTA_Pin {false} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg noRSTB" {set_property CONFIG.Use_RSTB_Pin {false} [get_ips blk_mem_gen_busimg]}
zlc_dump_ip blk_mem_gen_busimg
generate_target all [get_ips blk_mem_gen_busimg]

# NOTE: there is NO lane / delay image BRAM.  The per-channel OUTPUT delay is a set of
# held CTRL scalars (delay_count/delay_bits/delay_off/delay_skip in the CTRL regfile),
# wired straight to the engine; its per-channel history ring is an inferred distributed-RAM
# / SRL inside zlc_edge_streamer (ram_style="distributed", +0 RAMB36, +0 DSP) -- no IP.

update_compile_order -fileset sources_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
zlc_require_run_complete synth_1 "synth_design Complete!"

# Implementation runs IN-PROCESS (open the finished synth run, then opt/place/route/
# bitstream in this same Vivado session) instead of as a separate project "impl_1"
# run.  The project impl run is launched with `launch_runs impl_1 -jobs N`, which
# re-checks every out-of-context IP run for currency and can relaunch one in
# parallel -- leaving impl_1 stuck at STATUS "Scripts Generated" and producing no
# bitstream.  The in-process flow has a single, deterministic dependency (the
# completed synth_1) and writes the bit/ltx to the same impl_1 path the server +
# program_fpga.tcl expect.
set impl_dir [file join $project_dir ${project_name}.runs impl_1]
file mkdir $impl_dir
open_run synth_1 -name impl_1
opt_design
place_design
phys_opt_design
route_design
set bit_path [file join $impl_dir ${top}.bit]
set ltx_path [file join $impl_dir ${top}.ltx]
write_bitstream -force $bit_path
catch {write_debug_probes -force $ltx_path}
report_utilization -file [file join $impl_dir ${top}_utilization_routed.rpt]
catch {report_timing_summary -file [file join $impl_dir ${top}_timing_summary_routed.rpt]}
if {![file exists $bit_path]} { error "Bitstream was not generated: $bit_path" }
puts "ZLC bitstream: $bit_path"
