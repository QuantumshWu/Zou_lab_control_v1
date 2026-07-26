@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ============================================================================
rem  Double-click this file to check whether the FPGA part configured in
rem  fpga\board_config\streamer_config.json has enough resources (LUT / FF / DSP
rem  / BRAM) for the configured pulse-streamer geometry.  Edit that JSON to change
rem  the part, edge count, delay depth, etc., then re-run this.  The window stays
rem  open with the report.
rem ============================================================================

if /I not "%~1"=="--inner" (
  call "%~f0" --inner %*
  set "ZLC_STATUS=!ERRORLEVEL!"
  echo.
  if "!ZLC_STATUS!"=="0" (
    echo ZLC resource estimate: the configured part HAS enough resources.
  ) else if "!ZLC_STATUS!"=="1" (
    echo ZLC resource estimate: INSUFFICIENT -- see the OVER BUDGET lines above.
  ) else (
    echo ZLC resource estimate failed with code !ZLC_STATUS! -- read the messages above.
  )
  echo You can close this window, or press any key to exit.
  if "%ZLC_NO_PAUSE%"=="" pause
  exit /b !ZLC_STATUS!
)
shift /1

set "REPO_ROOT=%~dp0"
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

call "%REPO_ROOT%\fpga\_resolve_tools.bat" python "%REPO_ROOT%"
if errorlevel 1 exit /b 2

pushd "%REPO_ROOT%"
set "PYTHONPATH=%CD%;%PYTHONPATH%"
if "%ZLC_PS_CONFIG%"=="" if exist "%CD%\fpga\board_config\streamer_config.json" set "ZLC_PS_CONFIG=%CD%\fpga\board_config\streamer_config.json"

echo Reading config: %ZLC_PS_CONFIG%
echo.
%ZLC_PY_CMD% -m fpga.pulse_streamer.host.image --config "%ZLC_PS_CONFIG%"
set "ZLC_RC=%ERRORLEVEL%"
popd
exit /b %ZLC_RC%
