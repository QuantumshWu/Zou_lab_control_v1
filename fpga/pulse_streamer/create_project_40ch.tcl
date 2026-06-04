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
proc zlc_path_under {child parent} {
    set child_norm [string tolower [file normalize $child]]
    set parent_norm [string tolower [file normalize $parent]]
    set parent_len [string length $parent_norm]
    if {[string range $child_norm 0 [expr {$parent_len - 1}]] ne $parent_norm} {
        return 0
    }
    set next_char [string range $child_norm $parent_len $parent_len]
    return [expr {$next_char eq "" || $next_char eq "/" || $next_char eq "\\"}]
}
proc zlc_safe_project_dir {project_dir fallback_dir script_dir project_name} {
    set out [file normalize $project_dir]
    if {[zlc_path_under $out [file join $script_dir build]]} {
        puts "Ignoring old fpga/pulse_streamer/build ZLC_PS_PROJECT_DIR: $out"
        set out [file normalize $fallback_dir]
    }
    set debug_tmp [file normalize [file join $out ${project_name}.runs impl_1 .Xil Vivado-00000-QuantumPad]]
    if {[string length $debug_tmp] > 146} {
        error "Vivado debug-core temporary path would be too long ($debug_tmp). Move the repo to a shorter project folder such as D:/ZLC, or set ZLC_PS_PROJECT_DIR to a shorter project-local build path."
    }
    return $out
}
set xdc_path [env_or ZLC_PS_40CH_XDC [env_or ZLC_PS_XDC [file join $script_dir zlc_pulse_streamer_40ch.xdc]]]
if {![file exists $xdc_path]} {
    error "40ch XDC not found: $xdc_path. The repo normally includes zlc_pulse_streamer_40ch.xdc derived from address_switch; restore it or set ZLC_PS_40CH_XDC/ZLC_PS_XDC to a board-specific 40-output XDC."
}
set xdc_file [open $xdc_path r]
set xdc_text [read $xdc_file]
close $xdc_file
if {[string match "*<PIN_CH*" $xdc_text]} {
    error "$xdc_path still contains <PIN_CHxx> placeholders. Fill all 40 real package pins before building."
}

set zlc_default_project_root [zlc_default_project_root $script_dir]
set project_name p40
set project_dir [zlc_safe_project_dir [env_or ZLC_PS_PROJECT_DIR [file join $zlc_default_project_root p40]] [file join $zlc_default_project_root p40] $script_dir $project_name]
set top zlc_pulse_streamer_top_40ch
set part xc7a35tfgg484-2

puts "ZLC create_project_40ch contract: CHANNEL_COUNT=40 MAX_EDGES=1024 EDGE_ADDR_WIDTH=10"
puts "ZLC create_project_40ch XDC: $xdc_path"
puts "ZLC create_project_40ch project_dir: $project_dir"

proc zlc_require_run_complete {run_name expected_status} {
    set status [get_property STATUS [get_runs $run_name]]
    if {[string first $expected_status $status] < 0} {
        error "Vivado run $run_name did not complete successfully. STATUS='$status', expected '$expected_status'."
    }
}

if {[file exists $project_dir]} {
    puts "Removing previous ZLC pulse-streamer project: $project_dir"
    file delete -force $project_dir
}
file mkdir [file dirname $project_dir]
create_project $project_name $project_dir -part $part -force
set_property target_language Verilog [current_project]

read_verilog [file join $script_dir zlc_pulse_streamer.v]
read_verilog [file join $script_dir zlc_pulse_streamer_top_40ch.v]
read_xdc $xdc_path
set_property top $top [current_fileset]

create_ip -name vio -vendor xilinx.com -library ip -version 3.0 -module_name vio_0
set_property -dict [list \
    CONFIG.C_NUM_PROBE_IN {2} \
    CONFIG.C_NUM_PROBE_OUT {11} \
    CONFIG.C_PROBE_IN0_WIDTH {1} \
    CONFIG.C_PROBE_IN1_WIDTH {1} \
    CONFIG.C_PROBE_OUT0_WIDTH {1} \
    CONFIG.C_PROBE_OUT1_WIDTH {1} \
    CONFIG.C_PROBE_OUT2_WIDTH {1} \
    CONFIG.C_PROBE_OUT3_WIDTH {10} \
    CONFIG.C_PROBE_OUT4_WIDTH {32} \
    CONFIG.C_PROBE_OUT5_WIDTH {40} \
    CONFIG.C_PROBE_OUT6_WIDTH {11} \
    CONFIG.C_PROBE_OUT7_WIDTH {1} \
    CONFIG.C_PROBE_OUT8_WIDTH {10} \
    CONFIG.C_PROBE_OUT9_WIDTH {32} \
    CONFIG.C_PROBE_OUT10_WIDTH {32} \
] [get_ips vio_0]
generate_target all [get_ips vio_0]

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
if {![file exists $ltx_path]} { error "VIO probe file was not generated: $ltx_path" }
puts "ZLC pulse-streamer bitstream:"
puts "  $bit_path"
puts "ZLC pulse-streamer probes:"
puts "  $ltx_path"
