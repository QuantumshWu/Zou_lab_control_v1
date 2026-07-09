// ==========================================================================
// zlc_geometry.vh -- AUTO-GENERATED from fpga/board_config/streamer_config.json by
//   python -m fpga.pulse_streamer.host.image --emit-geometry-vh <path>
// DO NOT EDIT.  Every RTL geometry parameter (+ the LAYOUT_FINGERPRINT the host connect-
// check verifies) defaults to a macro here, so editing the config + rebuilding propagates
// to the bitstream and testbenches with no hand-carried literal.  Regenerated from the
// active config by build_and_program.bat; the committed copy is pinned to the shipped
// config by test_geometry_vh_matches_config.
// ==========================================================================
`ifndef ZLC_GEOMETRY_VH
`define ZLC_GEOMETRY_VH
`define ZLC_CHANNEL_COUNT       62
`define ZLC_NUM_SLOTS           4
`define ZLC_COEFF_WIDTH         16
`define ZLC_TICK_WIDTH          32
`define ZLC_COEFF_FRAC_BITS     8
`define ZLC_EDGE_ADDR_WIDTH     12
`define ZLC_BANK_SIZE           2048
`define ZLC_SCAN_ADDR_WIDTH     12
`define ZLC_BUS_COUNT           4
`define ZLC_BUS_INDEX_WIDTH     2
`define ZLC_BUS_WIDTH           10
`define ZLC_BUS_SEG_ADDR_WIDTH  6
`define ZLC_BUS_SEL_WIDTH       3
`define ZLC_EVT_FIFO_DEPTH      64
`define ZLC_BUS_EVT_FIFO_DEPTH  64
`define ZLC_NUM_DELAY_CH        18
`define ZLC_DELAY_CH_IDX_W      6
`define ZLC_DELAY_REG_WORDS     128
`define ZLC_LAYOUT_FINGERPRINT  32'h5AFC7CFB
`endif // ZLC_GEOMETRY_VH
