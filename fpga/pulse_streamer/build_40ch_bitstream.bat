@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
if "%~1"=="--help" (
  echo Build the 40-channel ZLC pulse-streamer bitstream after zlc_pulse_streamer_40ch.xdc has real pins.
  echo Use check_40ch_synth.bat for a no-XDC synthesis self-check.
  exit /b 0
)
if "%~1"=="/?" (
  echo Build the 40-channel ZLC pulse-streamer bitstream after zlc_pulse_streamer_40ch.xdc has real pins.
  echo Use check_40ch_synth.bat for a no-XDC synthesis self-check.
  exit /b 0
)
call "%SCRIPT_DIR%vivado_run_tcl.bat" create_project_40ch.tcl
set "ZLC_STATUS=%ERRORLEVEL%"
endlocal & exit /b %ZLC_STATUS%
