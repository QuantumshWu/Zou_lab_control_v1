# Build the Architecture-D edge-table pulse streamer (BRAM edge/scan tables +
# depth-1 prefetch engine, JTAG-to-AXI control).  Reaches 2048 edges + 4096 scan
# points at <=75% of the 35T.
#
# *** First integration build: the engine + control FSM are proven by Python
# models (tests/test_neutral_atom_lightweight.py: test_edgetable_*), but the
# multi-BRAM AXI integration is BLIND (no Verilog sim in repo) and needs on-board
# bring-up.  IP property names are version-specific: each is set defensively
# (zlc_try warns, does not abort) and the real CONFIG.*/ports are dumped -- grep
# the log for "ZLC IPDUMP" / "ZLC TRY-FAIL" to converge.  Run synth first. ***
#
# Geometry must match zlc_pulse_streamer_d_top.v localparams AND
# edgetable_image.solve_capacity("xc7a35t"):
#   EDGE_ADDR_WIDTH=11 (2048 edges), edge port-B 256b -> port-A depth 2048*8=16384
#   SCAN_ADDR_WIDTH=12 (4096 points), scan port-B 128b -> port-A depth 4096*4=16384
#   bus image 256*7=1792 words; CTRL 64 words; total ~34624 -> axi_bram depth 65536

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
proc zlc_safe_project_dir {project_dir script_dir project_name} {
    set out [file normalize $project_dir]
    set debug_tmp [file normalize [file join $out ${project_name}.runs impl_1 .Xil Vivado-00000-QuantumPad]]
    if {[string length $debug_tmp] > 146} {
        error "Vivado debug-core temp path too long ($debug_tmp). Check out at a shorter path or set ZLC_PS_PROJECT_DIR."
    }
    return $out
}

set xdc_path [env_or ZLC_PS_XDC [file join $script_dir .. .. references source_archives address_switch address_switch.srcs constrs_1 new addre.xdc]]
if {![file exists $xdc_path]} { error "board XDC not found: $xdc_path. Set ZLC_PS_XDC." }
set xdc_file [open $xdc_path r]; set xdc_text [read $xdc_file]; close $xdc_file
if {[string match "*<PIN_CH*" $xdc_text]} { error "$xdc_path still has <PIN_CHxx> placeholders." }

set project_root [zlc_default_project_root $script_dir]
set project_name d
set project_dir [zlc_safe_project_dir [env_or ZLC_PS_PROJECT_DIR [file join $project_root d]] $script_dir $project_name]
set top zlc_pulse_streamer_d_top
set part xc7a35tfgg484-2

# Geometry (single source: keep in sync with the top + solve_capacity).
set zlc_edge_addr_width 11
set zlc_scan_addr_width 12
set zlc_edge_portb_bits 256
set zlc_scan_portb_bits 128
set zlc_edge_porta_depth [expr {(1 << $zlc_edge_addr_width) * ($zlc_edge_portb_bits / 32)}]
set zlc_scan_porta_depth [expr {(1 << $zlc_scan_addr_width) * ($zlc_scan_portb_bits / 32)}]
set zlc_axi_bram_depth 65536

puts "ZLC create_project_d: D engine (BRAM tables + depth-1 prefetch), 2048 edges + 4096 points"
puts "ZLC create_project_d project_dir: $project_dir"

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

if {[file exists $project_dir]} { file delete -force $project_dir }
file mkdir [file dirname $project_dir]
create_project $project_name $project_dir -part $part -force
set_property target_language Verilog [current_project]

read_verilog [file join $script_dir zlc_pulse_streamer_d.v]
read_verilog [file join $script_dir zlc_pulse_streamer_d_top.v]
read_xdc $xdc_path
set_property top $top [current_fileset]

# --- jtag_axi master ------------------------------------------------------
create_ip -name jtag_axi -vendor xilinx.com -library ip -module_name jtag_axi_0
zlc_try "jtag PROTOCOL=AXI4LITE"  {set_property CONFIG.PROTOCOL {AXI4LITE} [get_ips jtag_axi_0]}
zlc_try "jtag DATA=32" {set_property CONFIG.M_AXI_DATA_WIDTH {32} [get_ips jtag_axi_0]}
zlc_try "jtag ADDR=32" {set_property CONFIG.M_AXI_ADDR_WIDTH {32} [get_ips jtag_axi_0]}
zlc_dump_ip jtag_axi_0
generate_target all [get_ips jtag_axi_0]

# --- AXI BRAM controller (single; the top decodes its BRAM port to 3 BRAMs) ---
create_ip -name axi_bram_ctrl -vendor xilinx.com -library ip -module_name axi_bram_ctrl_0
zlc_try "bramc DATA=32"      {set_property CONFIG.DATA_WIDTH {32} [get_ips axi_bram_ctrl_0]}
zlc_try "bramc SINGLE_PORT"  {set_property CONFIG.SINGLE_PORT_BRAM {1} [get_ips axi_bram_ctrl_0]}
zlc_try "bramc PROTOCOL=AXI4LITE" {set_property CONFIG.PROTOCOL {AXI4LITE} [get_ips axi_bram_ctrl_0]}
zlc_try "bramc BMG=EXTERNAL" {set_property CONFIG.BMG_INSTANCE {EXTERNAL} [get_ips axi_bram_ctrl_0]}
set_property CONFIG.MEM_DEPTH $zlc_axi_bram_depth [get_ips axi_bram_ctrl_0]
if {[get_property CONFIG.MEM_DEPTH [get_ips axi_bram_ctrl_0]] != $zlc_axi_bram_depth} {
    error "axi_bram_ctrl MEM_DEPTH did not take (want $zlc_axi_bram_depth)."
}
zlc_dump_ip axi_bram_ctrl_0
generate_target all [get_ips axi_bram_ctrl_0]

# --- EDGE BRAM: asymmetric TDP, port A 32b write / port B 256b read -----------
create_ip -name blk_mem_gen -vendor xilinx.com -library ip -module_name blk_mem_gen_edge
zlc_try "edge TDP"        {set_property CONFIG.Memory_Type {True_Dual_Port_RAM} [get_ips blk_mem_gen_edge]}
zlc_try "edge ByteWE"     {set_property CONFIG.Use_Byte_Write_Enable {true} [get_ips blk_mem_gen_edge]}
zlc_try "edge ByteSize8"  {set_property CONFIG.Byte_Size {8} [get_ips blk_mem_gen_edge]}
zlc_try "edge WWA=32"     {set_property CONFIG.Write_Width_A {32} [get_ips blk_mem_gen_edge]}
zlc_try "edge RWA=32"     {set_property CONFIG.Read_Width_A {32} [get_ips blk_mem_gen_edge]}
zlc_try "edge WDA"        {set_property CONFIG.Write_Depth_A $zlc_edge_porta_depth [get_ips blk_mem_gen_edge]}
zlc_try "edge WWB=256"    {set_property CONFIG.Write_Width_B $zlc_edge_portb_bits [get_ips blk_mem_gen_edge]}
zlc_try "edge RWB=256"    {set_property CONFIG.Read_Width_B $zlc_edge_portb_bits [get_ips blk_mem_gen_edge]}
zlc_try "edge ENA"        {set_property CONFIG.Enable_A {Use_ENA_Pin} [get_ips blk_mem_gen_edge]}
zlc_try "edge ENB"        {set_property CONFIG.Enable_B {Always_Enabled} [get_ips blk_mem_gen_edge]}
zlc_try "edge noRSTA"     {set_property CONFIG.Use_RSTA_Pin {false} [get_ips blk_mem_gen_edge]}
zlc_try "edge noRSTB"     {set_property CONFIG.Use_RSTB_Pin {false} [get_ips blk_mem_gen_edge]}
zlc_dump_ip blk_mem_gen_edge
generate_target all [get_ips blk_mem_gen_edge]

# --- SCAN BRAM: asymmetric TDP, port A 32b write / port B 128b read -----------
create_ip -name blk_mem_gen -vendor xilinx.com -library ip -module_name blk_mem_gen_scan
zlc_try "scan TDP"        {set_property CONFIG.Memory_Type {True_Dual_Port_RAM} [get_ips blk_mem_gen_scan]}
zlc_try "scan ByteWE"     {set_property CONFIG.Use_Byte_Write_Enable {true} [get_ips blk_mem_gen_scan]}
zlc_try "scan ByteSize8"  {set_property CONFIG.Byte_Size {8} [get_ips blk_mem_gen_scan]}
zlc_try "scan WWA=32"     {set_property CONFIG.Write_Width_A {32} [get_ips blk_mem_gen_scan]}
zlc_try "scan RWA=32"     {set_property CONFIG.Read_Width_A {32} [get_ips blk_mem_gen_scan]}
zlc_try "scan WDA"        {set_property CONFIG.Write_Depth_A $zlc_scan_porta_depth [get_ips blk_mem_gen_scan]}
zlc_try "scan WWB=128"    {set_property CONFIG.Write_Width_B $zlc_scan_portb_bits [get_ips blk_mem_gen_scan]}
zlc_try "scan RWB=128"    {set_property CONFIG.Read_Width_B $zlc_scan_portb_bits [get_ips blk_mem_gen_scan]}
zlc_try "scan ENA"        {set_property CONFIG.Enable_A {Use_ENA_Pin} [get_ips blk_mem_gen_scan]}
zlc_try "scan ENB"        {set_property CONFIG.Enable_B {Always_Enabled} [get_ips blk_mem_gen_scan]}
zlc_try "scan noRSTA"     {set_property CONFIG.Use_RSTA_Pin {false} [get_ips blk_mem_gen_scan]}
zlc_try "scan noRSTB"     {set_property CONFIG.Use_RSTB_Pin {false} [get_ips blk_mem_gen_scan]}
zlc_dump_ip blk_mem_gen_scan
generate_target all [get_ips blk_mem_gen_scan]

# --- BUS image BRAM: symmetric 32b TDP (A=AXI write, B=mini-loader read) ------
create_ip -name blk_mem_gen -vendor xilinx.com -library ip -module_name blk_mem_gen_busimg
zlc_try "busimg TDP"      {set_property CONFIG.Memory_Type {True_Dual_Port_RAM} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg ByteWE"   {set_property CONFIG.Use_Byte_Write_Enable {true} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg ByteSize8" {set_property CONFIG.Byte_Size {8} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg WWA=32"   {set_property CONFIG.Write_Width_A {32} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg WDA=2048" {set_property CONFIG.Write_Depth_A {2048} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg ENA"      {set_property CONFIG.Enable_A {Use_ENA_Pin} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg ENB"      {set_property CONFIG.Enable_B {Always_Enabled} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg noRSTA"   {set_property CONFIG.Use_RSTA_Pin {false} [get_ips blk_mem_gen_busimg]}
zlc_try "busimg noRSTB"   {set_property CONFIG.Use_RSTB_Pin {false} [get_ips blk_mem_gen_busimg]}
zlc_dump_ip blk_mem_gen_busimg
generate_target all [get_ips blk_mem_gen_busimg]

update_compile_order -fileset sources_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
zlc_require_run_complete synth_1 "synth_design Complete!"
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
zlc_require_run_complete impl_1 "write_bitstream Complete!"

set bit_path [file join $project_dir ${project_name}.runs impl_1 ${top}.bit]
if {![file exists $bit_path]} { error "Bitstream was not generated: $bit_path" }
puts "ZLC D bitstream: $bit_path"
