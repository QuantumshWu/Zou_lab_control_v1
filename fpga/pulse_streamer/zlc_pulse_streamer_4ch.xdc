## First-light constraints copied from the historical address_switch project.
## Verify connector wiring before connecting lab hardware.

set_property -dict {PACKAGE_PIN R4 IOSTANDARD LVCMOS33} [get_ports clk]
create_clock -period 10.000 -name clk_100mhz [get_ports clk]

set_property -dict {PACKAGE_PIN M17 IOSTANDARD LVCMOS33} [get_ports trap]
set_property -dict {PACKAGE_PIN F15 IOSTANDARD LVCMOS33} [get_ports cooling]
set_property -dict {PACKAGE_PIN N15 IOSTANDARD LVCMOS33} [get_ports probe]
set_property -dict {PACKAGE_PIN R17 IOSTANDARD LVCMOS33} [get_ports qcm_trigger]

set_property -dict {PACKAGE_PIN R2 IOSTANDARD LVCMOS33} [get_ports zlc_running_led]
set_property -dict {PACKAGE_PIN R3 IOSTANDARD LVCMOS33} [get_ports zlc_done_led]

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property BITSTREAM.GENERAL.COMPRESS true [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 50 [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.CONFIG.SPI_FALL_EDGE Yes [current_design]
