@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
if "%~1"=="--help" (
  echo Synthesize the 40-channel top and VIO contract without requiring real output-pin XDC.
  echo This is a logic/API self-check, not a board-ready bitstream build.
  exit /b 0
)
if "%~1"=="/?" (
  echo Synthesize the 40-channel top and VIO contract without requiring real output-pin XDC.
  echo This is a logic/API self-check, not a board-ready bitstream build.
  exit /b 0
)
call "%SCRIPT_DIR%vivado_run_tcl.bat" check_40ch_synth.tcl
set "ZLC_STATUS=%ERRORLEVEL%"
endlocal & exit /b %ZLC_STATUS%
