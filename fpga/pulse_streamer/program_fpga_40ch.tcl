proc env_or {name default} {
    if {[info exists ::env($name)]} { return $::env($name) }
    return $default
}

set script_dir [file normalize [file dirname [info script]]]
set project_dir [env_or ZLC_PS_PROJECT_DIR [file join $script_dir build zlc_pulse_streamer_40ch]]
set bit_path [env_or ZLC_PS_VIVADO_BIT [env_or ZLC_PS_BIT [file join $project_dir zlc_pulse_streamer_40ch.runs impl_1 zlc_pulse_streamer_top_40ch.bit]]]
set ltx_path [env_or ZLC_PS_VIVADO_LTX [env_or ZLC_PS_LTX [file join $project_dir zlc_pulse_streamer_40ch.runs impl_1 zlc_pulse_streamer_top_40ch.ltx]]]
set hw_server_url [env_or ZLC_PS_HW_SERVER_URL [env_or ZLC_HW_SERVER_URL ""]]

if {![file exists $bit_path]} { error "Bitstream not found: $bit_path" }
if {![file exists $ltx_path]} { error "VIO probe file not found: $ltx_path" }

if {[llength [info commands load_features]]} { catch {load_features labtools} }
if {[llength [info commands open_hw_manager]]} {
    open_hw_manager
} elseif {[llength [info commands open_hw]]} {
    open_hw
}
if {![llength [info commands connect_hw_server]]} {
    error "Vivado hardware Tcl commands are unavailable. Install/enable Vivado LabTools or set ZLC_PS_VIVADO_BIN to a Vivado with Hardware Manager support."
}
if {$hw_server_url ne ""} {
    connect_hw_server -url $hw_server_url
} elseif {[catch {connect_hw_server} zlc_connect_error]} {
    puts "connect_hw_server failed: $zlc_connect_error"
    connect_hw_server
}
catch {refresh_hw_server}
set zlc_targets {}
if {[catch {set zlc_targets [get_hw_targets]} zlc_target_error]} {
    puts "get_hw_targets failed after refresh: $zlc_target_error"
    set zlc_targets {}
}
puts "Available hardware targets: $zlc_targets"
set zlc_target [lindex $zlc_targets 0]
if {$zlc_target eq ""} { error "No Vivado hardware target found. Check the USB/JTAG cable, board power, and hw_server connection." }
current_hw_target $zlc_target
if {[catch {open_hw_target $zlc_target} zlc_open_target_error]} {
    puts "open_hw_target failed: $zlc_open_target_error"
    catch {close_hw_target}
    puts "Retrying open_hw_target with -jtag_mode on."
    if {[catch {open_hw_target -jtag_mode on $zlc_target} zlc_open_target_jtag_error]} {
        error "Vivado sees hardware target '$zlc_target' but no FPGA device could be opened. Check board power, JTAG chain/mode jumpers, power-source jumper, cable seating, then disconnect/reconnect hw_server. Last error: $zlc_open_target_jtag_error"
    }
}
set device [lindex [get_hw_devices] 0]
if {$device eq ""} { error "Vivado opened the hardware target but found no FPGA device. Check board power, JTAG chain/mode jumpers, power-source jumper, and Hardware Manager Auto Connect." }

set_property PROGRAM.FILE $bit_path $device
set_property PROBES.FILE $ltx_path $device
set_property FULL_PROBES.FILE $ltx_path $device
program_hw_devices $device
refresh_hw_device $device

puts "Programmed $device"
puts "Bitstream: $bit_path"
puts "Probes: $ltx_path"
