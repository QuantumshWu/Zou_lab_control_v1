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
proc zlc_safe_project_dir {project_dir fallback_dir script_dir} {
    set out [file normalize $project_dir]
    if {[zlc_path_under $out [file join $script_dir build]]} {
        puts "Ignoring old fpga/pulse_streamer/build ZLC_PS_PROJECT_DIR: $out"
        set out [file normalize $fallback_dir]
    }
    return $out
}
proc zlc_safe_artifact_path {artifact_path fallback_path script_dir label} {
    set out [file normalize $artifact_path]
    if {[zlc_path_under $out [file join $script_dir build]]} {
        puts "Ignoring old fpga/pulse_streamer/build $label: $out"
        return [file normalize $fallback_path]
    }
    return $out
}

set script_dir [file normalize [file dirname [info script]]]
set zlc_default_project_root [zlc_default_project_root $script_dir]
set project_dir [zlc_safe_project_dir [env_or ZLC_PS_PROJECT_DIR [file join $zlc_default_project_root address_switch]] [file join $zlc_default_project_root address_switch] $script_dir]
set default_bit_path [file join $project_dir address_switch.runs impl_1 zlc_pulse_streamer_top_address_switch.bit]
set default_ltx_path [file join $project_dir address_switch.runs impl_1 zlc_pulse_streamer_top_address_switch.ltx]
set bit_path [zlc_safe_artifact_path [env_or ZLC_PS_VIVADO_BIT [env_or ZLC_PS_BIT $default_bit_path]] $default_bit_path $script_dir "bitstream path"]
set ltx_path [zlc_safe_artifact_path [env_or ZLC_PS_VIVADO_LTX [env_or ZLC_PS_LTX $default_ltx_path]] $default_ltx_path $script_dir "probe path"]
set hw_server_url [env_or ZLC_PS_HW_SERVER_URL [env_or ZLC_HW_SERVER_URL ""]]

puts "ZLC program_fpga_address_switch contract: CHANNEL_COUNT=62 MAX_EDGES=512 MAX_SCAN_POINTS=256 EDGE_ADDR_WIDTH=9 SCAN_ADDR_WIDTH=8"
puts "ZLC program_fpga_address_switch project_dir: $project_dir"
puts "ZLC program_fpga_address_switch bitstream: $bit_path"
puts "ZLC program_fpga_address_switch probes: $ltx_path"

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
