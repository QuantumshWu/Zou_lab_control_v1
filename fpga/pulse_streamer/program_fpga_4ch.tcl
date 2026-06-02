proc env_or {name default} {
    if {[info exists ::env($name)]} { return $::env($name) }
    return $default
}

set script_dir [file normalize [file dirname [info script]]]
set project_dir [env_or ZLC_PS_PROJECT_DIR [file join $script_dir build zlc_pulse_streamer_4ch]]
set bit_path [env_or ZLC_PS_VIVADO_BIT [env_or ZLC_PS_BIT [file join $project_dir zlc_pulse_streamer_4ch.runs impl_1 zlc_pulse_streamer_top_4ch.bit]]]
set ltx_path [env_or ZLC_PS_VIVADO_LTX [env_or ZLC_PS_LTX [file join $project_dir zlc_pulse_streamer_4ch.runs impl_1 zlc_pulse_streamer_top_4ch.ltx]]]

if {![file exists $bit_path]} { error "Bitstream not found: $bit_path" }
if {![file exists $ltx_path]} { error "VIO probe file not found: $ltx_path" }

open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target
set device [lindex [get_hw_devices] 0]
if {$device eq ""} { error "No Vivado hardware device found." }

set_property PROGRAM.FILE $bit_path $device
set_property PROBES.FILE $ltx_path $device
set_property FULL_PROBES.FILE $ltx_path $device
program_hw_devices $device
refresh_hw_device $device

puts "Programmed $device"
puts "Bitstream: $bit_path"
puts "Probes: $ltx_path"
