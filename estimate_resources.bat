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

call :zlc_find_python
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

:zlc_find_python
if defined ZLC_PY_CMD goto zlc_python_found
if defined ZLC_FPGA_SERVER_PYTHON (
  if exist "%ZLC_FPGA_SERVER_PYTHON%" (
    set "ZLC_PY_CMD=call "%ZLC_FPGA_SERVER_PYTHON%""
  ) else (
    set "ZLC_PY_CMD=%ZLC_FPGA_SERVER_PYTHON%"
  )
  goto zlc_python_found
)
if exist "%REPO_ROOT%\.zlc_python_path" (
  set /p "ZLC_STORED_PY="<"%REPO_ROOT%\.zlc_python_path"
  if exist "!ZLC_STORED_PY!" (
    set "ZLC_PY_CMD=call "!ZLC_STORED_PY!""
    goto zlc_python_found
  )
  echo Ignoring stale .zlc_python_path: !ZLC_STORED_PY!
)
where python >nul 2>nul
if not errorlevel 1 set "ZLC_PY_CMD=python"
if defined ZLC_PY_CMD goto zlc_python_found
where py >nul 2>nul
if not errorlevel 1 set "ZLC_PY_CMD=py -3"
if defined ZLC_PY_CMD goto zlc_python_found
echo Could not find python or py. Run fpga\install_requirements.bat first.
exit /b 1
:zlc_python_found
echo ZLC Python: %ZLC_PY_CMD%
exit /b 0
