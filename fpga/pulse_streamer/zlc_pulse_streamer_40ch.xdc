## 40-channel ZLC pulse-streamer constraints.
##
## Pin map derived from:
##   references/source_archives/address_switch/address_switch.srcs/constrs_1/new/addre.xdc
##
## FPGA bit order is the server/GUI hardware order ch00..ch39.  GUI display
## labels may call the first four channels trap/cooling/probe/qcm_trigger, but
## the uploaded mask is still ch[0]..ch[39].
##
## First four channels intentionally preserve the camera-imaging convention:
##   ch00 = trap
##   ch01 = cooling
##   ch02 = probe
##   ch03 = qcm_trigger, physically the old address_switch "trig" output

set_property -dict {PACKAGE_PIN R4 IOSTANDARD LVCMOS33} [get_ports clk]
create_clock -period 10.000 -name clk_100mhz [get_ports clk]

set_property -dict {PACKAGE_PIN R2 IOSTANDARD LVCMOS33} [get_ports zlc_running_led]
set_property -dict {PACKAGE_PIN R3 IOSTANDARD LVCMOS33} [get_ports zlc_done_led]

## ch00..ch17: former address_switch named TTL/control outputs.
set_property -dict {PACKAGE_PIN M17 IOSTANDARD LVCMOS33} [get_ports {ch[0]}]  ;# ch00 <- trap
set_property -dict {PACKAGE_PIN F15 IOSTANDARD LVCMOS33} [get_ports {ch[1]}]  ;# ch01 <- cooling
set_property -dict {PACKAGE_PIN N15 IOSTANDARD LVCMOS33} [get_ports {ch[2]}]  ;# ch02 <- probe
set_property -dict {PACKAGE_PIN R17 IOSTANDARD LVCMOS33} [get_ports {ch[3]}]  ;# ch03 <- trig / qcm_trigger
set_property -dict {PACKAGE_PIN F13 IOSTANDARD LVCMOS33} [get_ports {ch[4]}]  ;# ch04 <- cooling_pgc
set_property -dict {PACKAGE_PIN E14 IOSTANDARD LVCMOS33} [get_ports {ch[5]}]  ;# ch05 <- repump
set_property -dict {PACKAGE_PIN M13 IOSTANDARD LVCMOS33} [get_ports {ch[6]}]  ;# ch06 <- emCCD
set_property -dict {PACKAGE_PIN H15 IOSTANDARD LVCMOS33} [get_ports {ch[7]}]  ;# ch07 <- UV
set_property -dict {PACKAGE_PIN F14 IOSTANDARD LVCMOS33} [get_ports {ch[8]}]  ;# ch08 <- address
set_property -dict {PACKAGE_PIN H14 IOSTANDARD LVCMOS33} [get_ports {ch[9]}]  ;# ch09 <- microwave
set_property -dict {PACKAGE_PIN G13 IOSTANDARD LVCMOS33} [get_ports {ch[10]}] ;# ch10 <- state_pre
set_property -dict {PACKAGE_PIN G18 IOSTANDARD LVCMOS33} [get_ports {ch[11]}] ;# ch11 <- coil
set_property -dict {PACKAGE_PIN E13 IOSTANDARD LVCMOS33} [get_ports {ch[12]}] ;# ch12 <- grey_cooling
set_property -dict {PACKAGE_PIN G17 IOSTANDARD LVCMOS33} [get_ports {ch[13]}] ;# ch13 <- pushout
set_property -dict {PACKAGE_PIN M16 IOSTANDARD LVCMOS33} [get_ports {ch[14]}] ;# ch14 <- cooling_shutter
set_property -dict {PACKAGE_PIN L15 IOSTANDARD LVCMOS33} [get_ports {ch[15]}] ;# ch15 <- repump_shutter
set_property -dict {PACKAGE_PIN J17 IOSTANDARD LVCMOS33} [get_ports {ch[16]}] ;# ch16 <- probe_shutter
set_property -dict {PACKAGE_PIN K18 IOSTANDARD LVCMOS33} [get_ports {ch[17]}] ;# ch17 <- bias

## ch18..ch21: former DAC clock pins, now available as digital pulse outputs.
set_property -dict {PACKAGE_PIN Y8 IOSTANDARD LVCMOS33} [get_ports {ch[18]}]  ;# ch18 <- da_clk0
set_property -dict {PACKAGE_PIN R14 IOSTANDARD LVCMOS33} [get_ports {ch[19]}] ;# ch19 <- da_clk1
set_property -dict {PACKAGE_PIN E19 IOSTANDARD LVCMOS33} [get_ports {ch[20]}] ;# ch20 <- da_clk2
set_property -dict {PACKAGE_PIN F21 IOSTANDARD LVCMOS33} [get_ports {ch[21]}] ;# ch21 <- da_clk3

## ch22..ch39: former DAC data pins used as additional digital pulse outputs.
set_property -dict {PACKAGE_PIN V9 IOSTANDARD LVCMOS33} [get_ports {ch[22]}]  ;# ch22 <- da_dipole[0]
set_property -dict {PACKAGE_PIN W9 IOSTANDARD LVCMOS33} [get_ports {ch[23]}]  ;# ch23 <- da_dipole[1]
set_property -dict {PACKAGE_PIN Y9 IOSTANDARD LVCMOS33} [get_ports {ch[24]}]  ;# ch24 <- da_dipole[2]
set_property -dict {PACKAGE_PIN V8 IOSTANDARD LVCMOS33} [get_ports {ch[25]}]  ;# ch25 <- da_dipole[3]
set_property -dict {PACKAGE_PIN U7 IOSTANDARD LVCMOS33} [get_ports {ch[26]}]  ;# ch26 <- da_dipole[4]
set_property -dict {PACKAGE_PIN AB7 IOSTANDARD LVCMOS33} [get_ports {ch[27]}] ;# ch27 <- da_dipole[5]
set_property -dict {PACKAGE_PIN V7 IOSTANDARD LVCMOS33} [get_ports {ch[28]}]  ;# ch28 <- da_dipole[6]
set_property -dict {PACKAGE_PIN AB6 IOSTANDARD LVCMOS33} [get_ports {ch[29]}] ;# ch29 <- da_dipole[7]
set_property -dict {PACKAGE_PIN AB8 IOSTANDARD LVCMOS33} [get_ports {ch[30]}] ;# ch30 <- da_dipole[8]
set_property -dict {PACKAGE_PIN AA8 IOSTANDARD LVCMOS33} [get_ports {ch[31]}] ;# ch31 <- da_dipole[9]
set_property -dict {PACKAGE_PIN N22 IOSTANDARD LVCMOS33} [get_ports {ch[32]}] ;# ch32 <- da_bias_x[0]
set_property -dict {PACKAGE_PIN M22 IOSTANDARD LVCMOS33} [get_ports {ch[33]}] ;# ch33 <- da_bias_x[1]
set_property -dict {PACKAGE_PIN M21 IOSTANDARD LVCMOS33} [get_ports {ch[34]}] ;# ch34 <- da_bias_x[2]
set_property -dict {PACKAGE_PIN L21 IOSTANDARD LVCMOS33} [get_ports {ch[35]}] ;# ch35 <- da_bias_x[3]
set_property -dict {PACKAGE_PIN K22 IOSTANDARD LVCMOS33} [get_ports {ch[36]}] ;# ch36 <- da_bias_x[4]
set_property -dict {PACKAGE_PIN K21 IOSTANDARD LVCMOS33} [get_ports {ch[37]}] ;# ch37 <- da_bias_x[5]
set_property -dict {PACKAGE_PIN J22 IOSTANDARD LVCMOS33} [get_ports {ch[38]}] ;# ch38 <- da_bias_x[6]
set_property -dict {PACKAGE_PIN H22 IOSTANDARD LVCMOS33} [get_ports {ch[39]}] ;# ch39 <- da_bias_x[7]

set_property CFGBVS VCCO [current_design]
set_property CONFIG_VOLTAGE 3.3 [current_design]
set_property BITSTREAM.GENERAL.COMPRESS true [current_design]
set_property BITSTREAM.CONFIG.CONFIGRATE 50 [current_design]
set_property BITSTREAM.CONFIG.SPI_BUSWIDTH 4 [current_design]
set_property BITSTREAM.CONFIG.SPI_FALL_EDGE Yes [current_design]
