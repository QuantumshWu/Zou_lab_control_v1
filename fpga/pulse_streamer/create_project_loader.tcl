# Build the affine edge-table pulse streamer with a JTAG-to-AXI loader path.
#
# Restores the VALIDATED seamless engine (zlc_pulse_streamer): single global edge
# pointer, single-cycle hardware loop (repeat NOT unrolled), single-cycle
# scan-point advance, N-slot affine scan of delay/duration, and DAC-value scan via
# bus value_select.  An on-chip loader (zlc_axi_program_loader) copies the program
# image out of the AXI BRAM into the engine's prog_* ports, then pulses start, so
# the engine plays from its (validated, gapless) LUTRAM tables -- the control path
# is JTAG-to-AXI, the engine is unchanged.
#
# There is no Verilog simulator here; the loader FSM + image packer are verified by
# the Python contract/co-sim tests (tests/test_neutral_atom_lightweight.py:
# test_edgetable_*).  Run this, paste any Vivado errors back, and converge.
#
# Path fix (in-repo, portable): a SHORT project name "l" keeps the Vivado 2019
# debug-core temp path under ~146 chars even from a deep checkout.

set script_dir [file normalize [file dirname [info script]]]
proc env_or {name default} {
    if {[info exists ::env($name)] && $::env($name) ne ""} { return [file normalize $::env($name)] }
    return $default
}
proc env_value {name default} {
    if {[info exists ::env($name)] && $::env($name) ne ""} { return $::env($name) }
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
        error "Vivado debug-core temp path would be too long ($debug_tmp, [string length $debug_tmp] chars > 146). Check out the repo at a shorter path, or set ZLC_PS_PROJECT_DIR to a short in-repo path."
    }
    return $out
}

set xdc_path [env_or ZLC_PS_XDC [file join $script_dir .. .. references source_archives address_switch address_switch.srcs constrs_1 new addre.xdc]]
if {![file exists $xdc_path]} {
    error "board XDC not found: $xdc_path. Restore the board XDC or set ZLC_PS_XDC."
}
set xdc_file [open $xdc_path r]
set xdc_text [read $xdc_file]
close $xdc_file
if {[string match "*<PIN_CH*" $xdc_text]} {
    error "$xdc_path still contains <PIN_CHxx> placeholders. Fill all real package pins before building."
}

# SHORT project name "l" is the path-length fix; keep it short on purpose.
set project_root [zlc_default_project_root $script_dir]
set project_name l
set project_dir [zlc_safe_project_dir [env_or ZLC_PS_PROJECT_DIR [file join $project_root l]] $script_dir $project_name]
set top zlc_pulse_streamer_loader_top
set part xc7a35tfgg484-2

puts "ZLC create_project_loader: CHANNEL_COUNT=62 NUM_SLOTS=4 control=JTAG-to-AXI (edge-table loader)"
puts "ZLC create_project_loader XDC: $xdc_path"
puts "ZLC create_project_loader project_dir: $project_dir"

proc zlc_require_run_complete {run_name expected_status} {
    set status [get_property STATUS [get_runs $run_name]]
    if {[string first $expected_status $status] < 0} {
        error "Vivado run $run_name did not complete. STATUS='$status', expected '$expected_status'."
    }
}

if {[file exists $project_dir]} {
    puts "Removing previous loader project: $project_dir"
    file delete -force $project_dir
}
file mkdir [file dirname $project_dir]
create_project $project_name $project_dir -part $part -force
set_property target_language Verilog [current_project]

read_verilog [file join $script_dir zlc_pulse_streamer.v]
read_verilog [file join $script_dir zlc_axi_program_loader.v]
read_verilog [file join $script_dir zlc_pulse_streamer_loader_top.v]
read_xdc $xdc_path
set_property top $top [current_fileset]

# --- Control path: JTAG-to-AXI master -> AXI BRAM controller -> dual-port BRAM.
# Host writes the program image (edgetable_image.py) over JTAG via hw_axi txns; the
# loader reads it from BRAM port B into the engine's prog_* ports.  Both AXI ends are
# AXI4-Lite (single-beat word txns).  IP property names are Vivado-version specific;
# the helpers set each defensively and DUMP the real CONFIG.*/ports (grep "ZLC IPDUMP").

proc zlc_try {label body} {
    if {[catch {uplevel 1 $body} zlc_e]} {
        puts "ZLC TRY-FAIL ($label): $zlc_e"
        return 0
    }
    return 1
}
proc zlc_dump_ip {ip} {
    puts "ZLC IPDUMP ===== $ip ====="
    foreach p [lsort [list_property [get_ips $ip]]] {
        if {[string match CONFIG.* $p]} {
            puts "ZLC IPDUMP   $p = [get_property $p [get_ips $ip]]"
        }
    }
    foreach pn {CONFIG.PROTOCOL CONFIG.Memory_Type} {
        if {![catch {set vv [list_property_value $pn [get_ips $ip]]}]} {
            puts "ZLC IPDUMP   $pn valid = $vv"
        }
    }
    if {![catch {generate_target {instantiation_template} [get_ips $ip]}]} {
        set veo [get_files -quiet -all -of_objects [get_ips $ip] *.veo]
        if {$veo ne ""} {
            puts "ZLC IPDUMP   ports template: [lindex $veo 0]"
            if {![catch {set fh [open [lindex $veo 0] r]}]} {
                puts [read $fh]
                close $fh
            }
        }
    }
    puts "ZLC IPDUMP ===== end $ip ====="
}

create_ip -name jtag_axi -vendor xilinx.com -library ip -module_name jtag_axi_0
zlc_try "jtag PROTOCOL=AXI4LITE"  {set_property CONFIG.PROTOCOL {AXI4LITE} [get_ips jtag_axi_0]}
zlc_try "jtag M_AXI_DATA_WIDTH=32" {set_property CONFIG.M_AXI_DATA_WIDTH {32} [get_ips jtag_axi_0]}
zlc_try "jtag M_AXI_ADDR_WIDTH=32" {set_property CONFIG.M_AXI_ADDR_WIDTH {32} [get_ips jtag_axi_0]}
zlc_dump_ip jtag_axi_0
generate_target all [get_ips jtag_axi_0]

create_ip -name axi_bram_ctrl -vendor xilinx.com -library ip -module_name axi_bram_ctrl_0
zlc_try "bramc DATA_WIDTH=32"      {set_property CONFIG.DATA_WIDTH {32} [get_ips axi_bram_ctrl_0]}
zlc_try "bramc SINGLE_PORT_BRAM=1" {set_property CONFIG.SINGLE_PORT_BRAM {1} [get_ips axi_bram_ctrl_0]}
zlc_try "bramc PROTOCOL=AXI4LITE"  {set_property CONFIG.PROTOCOL {AXI4LITE} [get_ips axi_bram_ctrl_0]}
zlc_try "bramc BMG_INSTANCE=EXTERNAL" {set_property CONFIG.BMG_INSTANCE {EXTERNAL} [get_ips axi_bram_ctrl_0]}
# 32768 words = power-of-two depth that holds the edge-table image (<=11040 words)
# and fits the 35T (32/50 RAMB36).  bram_addr_a is then 17 bits; word address = [16:2].
# STRICT (NOT zlc_try): a silently-reverted depth produced a 16-word BRAM once; abort
# loudly if Vivado rejects it instead of shipping an aliasing BRAM.
set zlc_bram_depth 32768
set_property CONFIG.MEM_DEPTH $zlc_bram_depth [get_ips axi_bram_ctrl_0]
if {[get_property CONFIG.MEM_DEPTH [get_ips axi_bram_ctrl_0]] != $zlc_bram_depth} {
    error "axi_bram_ctrl MEM_DEPTH did not take (got [get_property CONFIG.MEM_DEPTH [get_ips axi_bram_ctrl_0]], want $zlc_bram_depth). Aborting before a garbage BRAM is synthesized."
}
zlc_dump_ip axi_bram_ctrl_0
generate_target all [get_ips axi_bram_ctrl_0]

create_ip -name blk_mem_gen -vendor xilinx.com -library ip -module_name blk_mem_gen_0
zlc_try "bram Memory_Type=TDP"     {set_property CONFIG.Memory_Type {True_Dual_Port_RAM} [get_ips blk_mem_gen_0]}
zlc_try "bram Use_Byte_Write_Enable" {set_property CONFIG.Use_Byte_Write_Enable {true} [get_ips blk_mem_gen_0]}
zlc_try "bram Byte_Size=8"          {set_property CONFIG.Byte_Size {8} [get_ips blk_mem_gen_0]}
zlc_try "bram Write_Width_A=32"     {set_property CONFIG.Write_Width_A {32} [get_ips blk_mem_gen_0]}
# STRICT depth (must match axi_bram_ctrl MEM_DEPTH); abort if Vivado reverts it.
set_property CONFIG.Write_Depth_A $zlc_bram_depth [get_ips blk_mem_gen_0]
if {[get_property CONFIG.Write_Depth_A [get_ips blk_mem_gen_0]] != $zlc_bram_depth} {
    error "blk_mem_gen Write_Depth_A reverted to [get_property CONFIG.Write_Depth_A [get_ips blk_mem_gen_0]] (want $zlc_bram_depth). Vivado rejected the depth (exceeds device BRAM?)."
}
zlc_try "bram Read_Width_A=32"      {set_property CONFIG.Read_Width_A {32} [get_ips blk_mem_gen_0]}
zlc_try "bram Write_Width_B=32"     {set_property CONFIG.Write_Width_B {32} [get_ips blk_mem_gen_0]}
zlc_try "bram Read_Width_B=32"      {set_property CONFIG.Read_Width_B {32} [get_ips blk_mem_gen_0]}
zlc_try "bram Enable_A=Use_ENA_Pin" {set_property CONFIG.Enable_A {Use_ENA_Pin} [get_ips blk_mem_gen_0]}
zlc_try "bram Enable_B=Use_ENB_Pin" {set_property CONFIG.Enable_B {Use_ENB_Pin} [get_ips blk_mem_gen_0]}
zlc_try "bram Use_RSTA_Pin=false"   {set_property CONFIG.Use_RSTA_Pin {false} [get_ips blk_mem_gen_0]}
zlc_try "bram Use_RSTB_Pin=false"   {set_property CONFIG.Use_RSTB_Pin {false} [get_ips blk_mem_gen_0]}
# The loader holds each row's address/data stable for several cycles around the
# prog_we toggle and settles every read, so it tolerates port-B read latency 1 OR 2;
# do NOT force the IP latency (forcing it off can be rejected and abort the build).
catch {puts "ZLC blk_mem_gen READ_LATENCY_B = [get_property CONFIG.READ_LATENCY_B [get_ips blk_mem_gen_0]]"}
zlc_dump_ip blk_mem_gen_0
generate_target all [get_ips blk_mem_gen_0]

update_compile_order -fileset sources_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
zlc_require_run_complete synth_1 "synth_design Complete!"
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
zlc_require_run_complete impl_1 "write_bitstream Complete!"

set bit_path [file join $project_dir ${project_name}.runs impl_1 ${top}.bit]
set ltx_path [file join $project_dir ${project_name}.runs impl_1 ${top}.ltx]
if {![file exists $bit_path]} { error "Bitstream was not generated: $bit_path" }
puts "ZLC loader bitstream: $bit_path"
if {[file exists $ltx_path]} { puts "ZLC loader probes: $ltx_path" }
