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
proc zlc_safe_check_project_dir {project_dir fallback_dir script_dir project_name} {
    set out [file normalize $project_dir]
    if {[zlc_path_under $out [file join $script_dir build]]} {
        puts "Ignoring old fpga/pulse_streamer/build ZLC_PS_CHECK_PROJECT_DIR: $out"
        set out [file normalize $fallback_dir]
    }
    set debug_tmp [file normalize [file join $out ${project_name}.runs synth_1 .Xil Vivado-00000-QuantumPad]]
    if {[string length $debug_tmp] > 146} {
        error "Vivado debug-core temporary path would be too long ($debug_tmp). Move the repo to a shorter project folder such as D:/ZLC, or set ZLC_PS_CHECK_PROJECT_DIR to a shorter project-local build path."
    }
    return $out
}
set zlc_default_project_root [zlc_default_project_root $script_dir]
set project_name c40
set project_dir [zlc_safe_check_project_dir [env_or ZLC_PS_CHECK_PROJECT_DIR [file join $zlc_default_project_root c40]] [file join $zlc_default_project_root c40] $script_dir $project_name]
set top zlc_pulse_streamer_top_40ch
set part xc7a35tfgg484-2

puts "ZLC check_40ch_synth contract: CHANNEL_COUNT=40 MAX_EDGES=1024 EDGE_ADDR_WIDTH=10"
puts "ZLC check_40ch_synth project_dir: $project_dir"

proc zlc_require_run_complete {run_name expected_status} {
    set status [get_property STATUS [get_runs $run_name]]
    if {[string first $expected_status $status] < 0} {
        error "Vivado run $run_name did not complete successfully. STATUS='$status', expected '$expected_status'."
    }
}

proc zlc_check_utilization {report_path} {
    set fh [open $report_path r]
    set text [read $fh]
    close $fh
    if {![regexp {\|[[:space:]]*Slice LUTs\*[[:space:]]*\|[[:space:]]*([0-9]+)[[:space:]]*\|[[:space:]]*[0-9]+[[:space:]]*\|[[:space:]]*([0-9]+)[[:space:]]*\|[[:space:]]*([0-9.]+)[[:space:]]*\|} $text _ lut_used lut_avail lut_pct]} {
        error "Could not parse Slice LUTs from utilization report: $report_path"
    }
    if {![regexp {\|[[:space:]]*Slice Registers[[:space:]]*\|[[:space:]]*([0-9]+)[[:space:]]*\|[[:space:]]*[0-9]+[[:space:]]*\|[[:space:]]*([0-9]+)[[:space:]]*\|[[:space:]]*([0-9.]+)[[:space:]]*\|} $text _ reg_used reg_avail reg_pct]} {
        error "Could not parse Slice Registers from utilization report: $report_path"
    }
    puts "ZLC 40ch utilization: LUT ${lut_used}/${lut_avail} (${lut_pct}%), FF ${reg_used}/${reg_avail} (${reg_pct}%)"
    if {[expr {double($lut_pct)}] > 90.0} {
        error "40ch synth LUT utilization is too high for this first-light profile: ${lut_pct}%"
    }
    if {[expr {double($reg_pct)}] > 90.0} {
        error "40ch synth register utilization is too high for this first-light profile: ${reg_pct}%"
    }
}

if {[file exists $project_dir]} {
    puts "Removing previous ZLC 40ch synth-check project: $project_dir"
    file delete -force $project_dir
}
file mkdir [file dirname $project_dir]
create_project $project_name $project_dir -part $part -force
set_property target_language Verilog [current_project]

read_verilog [file join $script_dir zlc_pulse_streamer.v]
read_verilog [file join $script_dir zlc_pulse_streamer_top_40ch.v]
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

open_run synth_1
set utilization_report [file join $project_dir zlc_40ch_synth_utilization.rpt]
report_utilization -file $utilization_report
zlc_check_utilization $utilization_report
report_timing_summary -file [file join $project_dir zlc_40ch_synth_timing_summary.rpt]

puts "ZLC 40ch synth check complete:"
puts "  project: $project_dir"
puts "  top: $top"
puts "  VIO prog_mask width: 40"
