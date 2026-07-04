"""FPGA design tree for Zou_lab_control (RTL + build tcl + host-side Python).

Kept OUT of the Python device-driver package on purpose: everything FPGA-specific
(the BRAM image layout, the cycle-accurate RTL behavioural models, the capacity
solver) lives under ``fpga/`` next to the Verilog it describes.  The host device
driver (``...devices.axi_session``) imports from here.
"""
