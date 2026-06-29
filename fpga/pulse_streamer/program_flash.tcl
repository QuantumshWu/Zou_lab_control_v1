# Program the FINAL pulse-streamer bitstream into the board's SPI CONFIGURATION FLASH so the
# FPGA AUTO-BOOTS from flash on power-up -- the program then SURVIVES a power cycle / reboot
# with NO re-run of build_and_program.  (program_fpga.tcl only loads the VOLATILE FPGA config
# SRAM via program_hw_devices, which is lost the moment the board loses power -- this script is
# the persistent counterpart.)  The board.xdc already sets the QSPI boot bitstream properties
# (CONFIG_VOLTAGE 3.3 / CONFIGRATE 50 / SPI_BUSWIDTH 4 / SPI_FALL_EDGE Yes), so the built .bit is
# flash-boot ready; this script only generates the .mcs and writes it to flash.
#
# REQUIREMENTS on the bench (one-time):
#   * the board's MODE jumper/strap must select SPI/QSPI boot (not JTAG) so it loads from flash;
#   * set ZLC_PS_CFGMEM_PART to YOUR board's SPI flash if it is not the default below -- list the
#     valid names in Vivado with:  get_cfgmem_parts
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

set script_dir [file normalize [file dirname [info script]]]
set project_root [zlc_default_project_root $script_dir]
set project_dir [env_or ZLC_PS_PROJECT_DIR [file join $project_root ps]]
set top zlc_pulse_streamer_top
set default_bit_path [file join $project_dir ps.runs impl_1 ${top}.bit]
set bit_path [env_or ZLC_PS_VIVADO_BIT [env_or ZLC_PS_BIT $default_bit_path]]
set hw_server_url [env_or ZLC_PS_HW_SERVER_URL [env_or ZLC_HW_SERVER_URL ""]]
# The board's SPI configuration flash.  Default targets the common xc7a35tfgg484 boards
# (Arty-A7-class, 128 Mb QSPI); OVERRIDE with ZLC_PS_CFGMEM_PART for a different flash.
set cfgmem_part_name [env_or ZLC_PS_CFGMEM_PART "s25fl128sxxxxxx0-spi-x1_x2_x4"]
set flash_size_mb [env_or ZLC_PS_CFGMEM_SIZE_MB 16]

puts "ZLC program_flash: persisting bitstream to SPI flash (survives power cycle)"
puts "ZLC program_flash bitstream: $bit_path"
puts "ZLC program_flash cfgmem part: $cfgmem_part_name (override with ZLC_PS_CFGMEM_PART)"
if {![file exists $bit_path]} { error "Bitstream not found: $bit_path. Build it first (build_and_program.bat --build-only)." }

# --- generate the .mcs configuration-memory image from the .bit (QSPI x4, matching board.xdc) ---
set mcs_path [file rootname $bit_path].mcs
write_cfgmem -force -format mcs -interface SPIx4 -size $flash_size_mb \
    -loadbit [list up 0x0 $bit_path] $mcs_path
puts "ZLC program_flash generated: $mcs_path"

# --- connect to the board (same proven path as program_fpga.tcl) ---
if {[llength [info commands load_features]]} { catch {load_features labtools} }
if {[llength [info commands open_hw_manager]]} {
    open_hw_manager
} elseif {[llength [info commands open_hw]]} {
    open_hw
}
if {![llength [info commands connect_hw_server]]} {
    error "Vivado hardware Tcl commands are unavailable. Install/enable Vivado LabTools."
}
if {$hw_server_url ne ""} {
    connect_hw_server -url $hw_server_url
} elseif {[catch {connect_hw_server} zlc_connect_error]} {
    puts "connect_hw_server failed: $zlc_connect_error"
    connect_hw_server
}
catch {refresh_hw_server}
set zlc_targets {}
catch {set zlc_targets [get_hw_targets]}
puts "Available hardware targets: $zlc_targets"
set zlc_target [lindex $zlc_targets 0]
if {$zlc_target eq ""} { error "No Vivado hardware target found. Check the USB/JTAG cable, board power, and hw_server." }
current_hw_target $zlc_target
if {[catch {open_hw_target $zlc_target}]} { catch {close_hw_target}; after 2000; open_hw_target $zlc_target }
set device [lindex [get_hw_devices] 0]
if {$device eq ""} { error "Vivado opened the target but found no FPGA device. Check power, JTAG chain/mode jumpers." }
current_hw_device $device
refresh_hw_device -update_hw_probes false $device

# --- attach the configuration flash and program the .mcs into it ---
if {[catch {create_hw_cfgmem -hw_device $device [lindex [get_cfgmem_parts $cfgmem_part_name] 0]} zlc_cfgmem_err]} {
    error "Could not attach cfgmem part '$cfgmem_part_name': $zlc_cfgmem_err.\n\
           Run 'get_cfgmem_parts' in Vivado and set ZLC_PS_CFGMEM_PART to your board's SPI flash."
}
set cfgmem [get_property PROGRAM.HW_CFGMEM $device]
set_property PROGRAM.BLANK_CHECK  0 $cfgmem
set_property PROGRAM.ERASE        1 $cfgmem
set_property PROGRAM.CFG_PROGRAM  1 $cfgmem
set_property PROGRAM.VERIFY       1 $cfgmem
set_property PROGRAM.CHECKSUM     0 $cfgmem
set_property PROGRAM.ADDRESS_RANGE {use_file} $cfgmem
set_property PROGRAM.FILES [list $mcs_path] $cfgmem
set_property PROGRAM.UNUSED_PIN_TERMINATION {pull-none} $cfgmem

# load the cfgmem-helper bitstream into the FPGA, then write + verify the flash
create_hw_bitstream -hw_device $device [get_property PROGRAM.HW_CFGMEM_BITFILE $device]
program_hw_devices $device
program_hw_cfgmem -hw_cfgmem $cfgmem
puts "ZLC program_flash: flash written + verified."

# reboot the FPGA so it loads the freshly-written flash NOW (and on every future power-up)
boot_hw_device $device
puts "ZLC program_flash: rebooted from flash. The program now survives power cycles."
