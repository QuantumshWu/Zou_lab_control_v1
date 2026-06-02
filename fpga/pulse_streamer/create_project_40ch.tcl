set script_dir [file normalize [file dirname [info script]]]
set xdc_path [file join $script_dir zlc_pulse_streamer_40ch.xdc]
if {![file exists $xdc_path]} {
    error "Create zlc_pulse_streamer_40ch.xdc from zlc_pulse_streamer_40ch.xdc.template and fill the real 40 output pins before building."
}
set xdc_file [open $xdc_path r]
set xdc_text [read $xdc_file]
close $xdc_file
if {[string match "*<PIN_CH*" $xdc_text]} {
    error "zlc_pulse_streamer_40ch.xdc still contains <PIN_CHxx> placeholders. Fill all 40 real package pins before building."
}

set project_dir [file join $script_dir build zlc_pulse_streamer_40ch]
set project_name zlc_pulse_streamer_40ch
set top zlc_pulse_streamer_top_40ch
set part xc7a35tfgg484-2

proc zlc_require_run_complete {run_name expected_status} {
    set status [get_property STATUS [get_runs $run_name]]
    if {[string first $expected_status $status] < 0} {
        error "Vivado run $run_name did not complete successfully. STATUS='$status', expected '$expected_status'."
    }
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
    CONFIG.C_NUM_PROBE_OUT {7} \
    CONFIG.C_PROBE_IN0_WIDTH {1} \
    CONFIG.C_PROBE_IN1_WIDTH {1} \
    CONFIG.C_PROBE_OUT0_WIDTH {1} \
    CONFIG.C_PROBE_OUT1_WIDTH {1} \
    CONFIG.C_PROBE_OUT2_WIDTH {1} \
    CONFIG.C_PROBE_OUT3_WIDTH {10} \
    CONFIG.C_PROBE_OUT4_WIDTH {32} \
    CONFIG.C_PROBE_OUT5_WIDTH {40} \
    CONFIG.C_PROBE_OUT6_WIDTH {11} \
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
