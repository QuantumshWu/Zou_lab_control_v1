proc env_or {name default} {
    if {[info exists ::env($name)]} { return $::env($name) }
    return $default
}

set hw_server_url [env_or ZLC_PS_HW_SERVER_URL [env_or ZLC_HW_SERVER_URL ""]]

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
set targets {}
if {[catch {set targets [get_hw_targets]} err]} {
    puts "get_hw_targets failed after refresh: $err"
    set targets {}
}
puts "ZLC hardware targets: $targets"
if {[llength $targets] == 0} {
    error "No hardware targets found. Check USB/JTAG cable, board power, cable drivers, and hw_server."
}

foreach target $targets {
    puts "ZLC opening target: $target"
    current_hw_target $target
    set opened 0
    if {[catch {open_hw_target $target} err]} {
        puts "open_hw_target failed: $err"
        catch {close_hw_target}
        puts "Retrying open_hw_target with -jtag_mode on."
        if {[catch {open_hw_target -jtag_mode on $target} jtag_err]} {
            puts "open_hw_target -jtag_mode on failed: $jtag_err"
        } else {
            set opened 1
        }
    } else {
        set opened 1
    }
    if {$opened} {
        set devices [get_hw_devices]
        puts "ZLC devices on $target: $devices"
        foreach device $devices {
            set line "  NAME=[get_property NAME $device]"
            foreach prop {PART IDCODE PROGRAM.FILE PROBES.FILE} {
                if {![catch {get_property $prop $device} value]} { append line " $prop=$value" }
            }
            puts $line
        }
        if {[llength $devices] == 0} {
            error "Target opened but no FPGA devices were detected. Check board power, JTAG chain, mode jumpers, and power-source jumper."
        }
    } else {
        error "Vivado sees hardware target '$target' but no FPGA device could be opened. Check board power, JTAG chain/mode jumpers, power-source jumper, cable seating, then disconnect/reconnect hw_server."
    }
}
