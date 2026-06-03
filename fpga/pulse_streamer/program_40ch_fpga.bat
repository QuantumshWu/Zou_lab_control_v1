@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
if "%~1"=="--help" (
  echo Program the FPGA with the generated 40-channel ZLC pulse-streamer bitstream and LTX probes.
  exit /b 0
)
if "%~1"=="/?" (
  echo Program the FPGA with the generated 40-channel ZLC pulse-streamer bitstream and LTX probes.
  exit /b 0
)
call "%SCRIPT_DIR%vivado_run_tcl.bat" program_fpga_40ch.tcl
set "ZLC_STATUS=%ERRORLEVEL%"
endlocal & exit /b %ZLC_STATUS%
