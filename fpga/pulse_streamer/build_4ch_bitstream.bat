@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
if "%~1"=="--help" (
  echo Build the 4-channel ZLC pulse-streamer bitstream.
  echo Override Vivado with ZLC_PS_VIVADO_BIN if auto-detection is not correct.
  exit /b 0
)
if "%~1"=="/?" (
  echo Build the 4-channel ZLC pulse-streamer bitstream.
  echo Override Vivado with ZLC_PS_VIVADO_BIN if auto-detection is not correct.
  exit /b 0
)
call "%SCRIPT_DIR%vivado_run_tcl.bat" create_project_4ch.tcl
set "ZLC_STATUS=%ERRORLEVEL%"
endlocal & exit /b %ZLC_STATUS%
