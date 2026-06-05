@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I not "%~1"=="--inner" (
  set "ZLC_ACTION=server"
  if /I "%~1"=="--check-config" set "ZLC_ACTION=server config check"
  call "%~f0" --inner %*
  set "ZLC_STATUS=!ERRORLEVEL!"
  if "!ZLC_STATUS!"=="0" (
    if "%~1"=="--help" exit /b 0
    if "%~1"=="/?" exit /b 0
    echo.
    echo ZLC !ZLC_ACTION! completed successfully.
    if /I "%~1"=="--check-config" (
      echo You can close this window, or press any key to exit.
    ) else (
      echo Server stopped normally. You can close this window, or press any key to exit.
    )
    if "%ZLC_NO_PAUSE%"=="" pause
  ) else (
    echo.
    echo ZLC !ZLC_ACTION! failed with code !ZLC_STATUS!.
    echo Keep this window open and read the messages above.
    if "%ZLC_NO_PAUSE%"=="" pause
  )
  exit /b !ZLC_STATUS!
)
shift /1

set "FPGA_DIR=%~dp0"
for %%I in ("%FPGA_DIR%..") do set "REPO_ROOT=%%~fI"
set "STREAMER_DIR=%FPGA_DIR%pulse_streamer"
set "ZLC_REPO_ROOT=%REPO_ROOT%"

set "ZLC_RUN_SERVER_CHECK=0"
if "%~1"=="--help" goto zlc_help
if "%~1"=="/?" goto zlc_help
if /I "%~1"=="--check-config" set "ZLC_RUN_SERVER_CHECK=1"
if not "%~1"=="" if not "%ZLC_RUN_SERVER_CHECK%"=="1" (
  echo Unknown option: %~1
  echo.
  goto zlc_help
)

call :zlc_find_python
if errorlevel 1 exit /b 1
call :zlc_find_vivado
if errorlevel 1 exit /b 1
call :zlc_default_paths
call :zlc_verify_address_switch_sources
if errorlevel 1 exit /b 1

pushd "%REPO_ROOT%"
set "PYTHONPATH=%CD%;%PYTHONPATH%"

if "%ZLC_PS_HOST%"=="" set "ZLC_PS_HOST=0.0.0.0"
if "%ZLC_PS_PORT%"=="" set "ZLC_PS_PORT=18861"
if "%ZLC_PS_SERVER_BACKEND%"=="" set "ZLC_PS_SERVER_BACKEND=vivado-session"
if "%ZLC_PS_VIVADO_PROGRAM_ON_RUN%"=="" set "ZLC_PS_VIVADO_PROGRAM_ON_RUN=0"
if "%ZLC_PS_MAX_EDGES%"=="" set "ZLC_PS_MAX_EDGES=512"
if "%ZLC_PS_MAX_SCAN_POINTS%"=="" set "ZLC_PS_MAX_SCAN_POINTS=256"
if "%ZLC_PS_TICK_WIDTH%"=="" set "ZLC_PS_TICK_WIDTH=32"
if "%ZLC_PS_CLOCK_HZ%"=="" set "ZLC_PS_CLOCK_HZ=50000000"
if "%ZLC_PS_MAX_CHANNEL_COUNT%"=="" (
  set "ZLC_PS_MAX_CHANNEL_COUNT_ARG="
) else (
  set "ZLC_PS_MAX_CHANNEL_COUNT_ARG=--max-channel-count %ZLC_PS_MAX_CHANNEL_COUNT%"
)
if "%ZLC_PS_XDC%"=="" if exist "%CD%\references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc" set "ZLC_PS_XDC=%CD%\references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc"
if "%ZLC_PS_CHANNEL_COUNT%"=="" (
  for /f "delims=" %%I in ('%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer infer_channel_count --xdc "%ZLC_PS_XDC%" --default-count 62 %ZLC_PS_MAX_CHANNEL_COUNT_ARG% 2^>nul') do if "!ZLC_PS_CHANNEL_COUNT!"=="" set "ZLC_PS_CHANNEL_COUNT=%%I"
)
if "%ZLC_PS_CHANNEL_COUNT%"=="" set "ZLC_PS_CHANNEL_COUNT=62"
if "%ZLC_PS_CHANNELS%"=="" (
  for /f "delims=" %%I in ('%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer infer_channels --xdc "%ZLC_PS_XDC%" --default-count %ZLC_PS_CHANNEL_COUNT% %ZLC_PS_MAX_CHANNEL_COUNT_ARG% 2^>nul') do if "!ZLC_PS_CHANNELS!"=="" set "ZLC_PS_CHANNELS=%%I"
)
if "%ZLC_PS_CHANNELS%"=="" (
  for /f "delims=" %%I in ('%ZLC_PY_CMD% -c "print(' '.join(f'ch{i:02d}' for i in range(62)))" 2^>nul') do if "!ZLC_PS_CHANNELS!"=="" set "ZLC_PS_CHANNELS=%%I"
)
if "%ZLC_PS_TRIGGER_CHANNELS%"=="" (
  for /f "delims=" %%I in ('%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer infer_trigger_channels --xdc "%ZLC_PS_XDC%" --default-count %ZLC_PS_CHANNEL_COUNT% %ZLC_PS_MAX_CHANNEL_COUNT_ARG% 2^>nul') do if "!ZLC_PS_TRIGGER_CHANNELS!"=="" set "ZLC_PS_TRIGGER_CHANNELS=%%I"
)
if "%ZLC_PS_TRIGGER_CHANNELS%"=="" (
  echo ERROR: could not infer the emCCD camera trigger channel from the selected XDC.
  echo.
  echo The default camera trigger must be the XDC output labelled emCCD. Check ZLC_PS_XDC,
  echo or set ZLC_PS_TRIGGER_CHANNELS explicitly only after confirming the camera trigger line.
  exit /b 1
)

if not "%ZLC_PS_PROJECT_DIR%"=="" (
  if "%ZLC_PS_VIVADO_PROJECT%"=="" set "ZLC_PS_VIVADO_PROJECT=%ZLC_PS_PROJECT_DIR%\address_switch.xpr"
  if "%ZLC_PS_VIVADO_BIT%"=="" set "ZLC_PS_VIVADO_BIT=%ZLC_PS_PROJECT_DIR%\address_switch.runs\impl_1\zlc_pulse_streamer_top_address_switch.bit"
  if "%ZLC_PS_VIVADO_LTX%"=="" set "ZLC_PS_VIVADO_LTX=%ZLC_PS_PROJECT_DIR%\address_switch.runs\impl_1\zlc_pulse_streamer_top_address_switch.ltx"
)
if "%ZLC_PS_VIVADO_LTX%"=="" (
  echo ERROR: no Vivado .ltx probe file was found.
  echo.
  echo The address-switch server controls the FPGA through Vivado VIO, so it must load
  echo the same .ltx Probes file used when the FPGA was programmed.
  echo.
  echo Fix one of these:
  echo   1. Check the address-switch XDC pin map, then run fpga\build_and_program.bat.
  echo   2. Or set ZLC_PS_VIVADO_LTX to the .ltx from Vivado Program Device.
  echo.
  echo Example:
  echo   set ZLC_PS_VIVADO_LTX=D:\path\zlc_pulse_streamer_top_address_switch.ltx
  echo   fpga\run_server.bat
  echo.
  popd
  exit /b 2
)
if not exist "%ZLC_PS_VIVADO_LTX%" (
  echo ERROR: Vivado .ltx probe file does not exist:
  echo   %ZLC_PS_VIVADO_LTX%
  echo.
  echo Build/program the address-switch bitstream first:
  echo   fpga\build_and_program.bat
  echo.
  echo Or set ZLC_PS_VIVADO_LTX to the exact Probes file used in Vivado Program Device.
  popd
  exit /b 2
)

echo ZLC FPGA pulse-streamer server: %ZLC_PS_CHANNEL_COUNT%ch
echo Host:    %ZLC_PS_HOST%:%ZLC_PS_PORT%
echo Backend: %ZLC_PS_SERVER_BACKEND%
echo Project: %ZLC_PS_VIVADO_PROJECT%
echo Bit:     %ZLC_PS_VIVADO_BIT%
echo LTX:     %ZLC_PS_VIVADO_LTX%
echo Channels: %ZLC_PS_CHANNELS%
echo Clock:   %ZLC_PS_CLOCK_HZ% Hz
echo Capacity: max_edges=%ZLC_PS_MAX_EDGES% scan_rows=%ZLC_PS_MAX_SCAN_POINTS% scan_bus_values=4x10 bus_segments=4x64

if "%ZLC_RUN_SERVER_CHECK%"=="1" (
  echo ZLC server config check complete.
  popd
  endlocal & exit /b 0
)

if /I "%ZLC_PS_SERVER_BACKEND%"=="command" goto zlc_run_command_backend

%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.sequencer_server ^
  --backend %ZLC_PS_SERVER_BACKEND% ^
  --host %ZLC_PS_HOST% ^
  --port %ZLC_PS_PORT% ^
  --channels %ZLC_PS_CHANNELS% ^
  --trigger-channels %ZLC_PS_TRIGGER_CHANNELS% ^
  --clock-hz %ZLC_PS_CLOCK_HZ% ^
  --state-dir "%ZLC_PS_STATE_DIR%"
set "ZLC_STATUS=%ERRORLEVEL%"
popd
endlocal & exit /b %ZLC_STATUS%

:zlc_run_command_backend
%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.sequencer_server ^
  --backend command ^
  --host %ZLC_PS_HOST% ^
  --port %ZLC_PS_PORT% ^
  --channels %ZLC_PS_CHANNELS% ^
  --trigger-channels %ZLC_PS_TRIGGER_CHANNELS% ^
  --clock-hz %ZLC_PS_CLOCK_HZ% ^
  --state-dir "%ZLC_PS_STATE_DIR%" ^
  --prepare-command "%ZLC_PY_ARG% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer prepare" ^
  --fire-command "%ZLC_PY_ARG% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer fire" ^
  --wait-done-command "%ZLC_PY_ARG% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer wait_done" ^
  --safe-state-command "%ZLC_PY_ARG% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer safe_state"
set "ZLC_STATUS=%ERRORLEVEL%"
popd
endlocal & exit /b %ZLC_STATUS%

:zlc_help
echo Start the XDC-inferred ZLC FPGA pulse-streamer server.
echo.
echo Usage:
echo   fpga\run_server.bat
echo   fpga\run_server.bat --check-config
echo.
echo Defaults:
echo   host/port: 0.0.0.0:18861
echo   backend:   vivado-session
echo   channels:  inferred from ZLC_PS_XDC, fallback ch00 ... ch61
echo   clock:     50000000 Hz ^(override with ZLC_PS_CLOCK_HZ^)
echo.
echo Optional:
echo   set ZLC_FPGA_SERVER_PYTHON=C:\path\to\python.exe
echo   set ZLC_PS_HOST=0.0.0.0
echo   set ZLC_PS_PORT=18861
echo   set ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.2\bin\vivado.bat
echo   set ZLC_PS_SERVER_BACKEND=vivado-session
  echo   set ZLC_PS_PROJECT_DIR=%%CD%%\fpga\build\address_switch
exit /b 0

:zlc_verify_address_switch_sources
set "ZLC_DEFAULT_XDC=%REPO_ROOT%\references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc"
if not defined ZLC_PS_XDC set "ZLC_PS_XDC=%ZLC_DEFAULT_XDC%"
set "ZLC_SELECTED_XDC=%ZLC_PS_XDC%"
if not defined ZLC_SELECTED_XDC set "ZLC_SELECTED_XDC=%ZLC_PS_XDC%"
if not defined ZLC_PS_XDC set "ZLC_PS_XDC=%ZLC_SELECTED_XDC%"
if not exist "%STREAMER_DIR%\zlc_pulse_streamer_top_address_switch.v" (
  echo ERROR: missing address-switch top HDL: %STREAMER_DIR%\zlc_pulse_streamer_top_address_switch.v
  exit /b 2
)
if not exist "%STREAMER_DIR%\create_project_address_switch.tcl" (
  echo ERROR: missing address-switch build Tcl: %STREAMER_DIR%\create_project_address_switch.tcl
  exit /b 2
)
findstr /C:".EDGE_ADDR_WIDTH(9)" "%STREAMER_DIR%\zlc_pulse_streamer_top_address_switch.v" >nul || (
  echo ERROR: address-switch top is not the 512-edge build. Expected .EDGE_ADDR_WIDTH^(9^).
  exit /b 2
)
findstr /C:".SCAN_ADDR_WIDTH(8)" "%STREAMER_DIR%\zlc_pulse_streamer_top_address_switch.v" >nul || (
  echo ERROR: address-switch top is not the 256-scan-row build. Expected .SCAN_ADDR_WIDTH^(8^).
  exit /b 2
)
findstr /C:"localparam integer CHANNEL_COUNT = 62" "%STREAMER_DIR%\zlc_pulse_streamer_top_address_switch.v" >nul || (
  echo ERROR: pulse-streamer top is not the 62-output address-switch wrapper.
  exit /b 2
)
findstr /C:"CONFIG.C_PROBE_OUT3_WIDTH {9}" "%STREAMER_DIR%\create_project_address_switch.tcl" >nul || (
  echo ERROR: create_project_address_switch.tcl has stale prog_addr width. Expected VIO probe_out3 width 9.
  exit /b 2
)
findstr /C:"CONFIG.C_PROBE_OUT5_WIDTH {62}" "%STREAMER_DIR%\create_project_address_switch.tcl" >nul || (
  echo ERROR: create_project_address_switch.tcl has stale mask width. Expected VIO probe_out5 width 62.
  exit /b 2
)
findstr /C:"CONFIG.C_PROBE_OUT6_WIDTH {10}" "%STREAMER_DIR%\create_project_address_switch.tcl" >nul || (
  echo ERROR: create_project_address_switch.tcl has stale prog_count width. Expected VIO probe_out6 width 10.
  exit /b 2
)
findstr /C:"CONFIG.C_PROBE_OUT8_WIDTH {9}" "%STREAMER_DIR%\create_project_address_switch.tcl" >nul || (
  echo ERROR: create_project_address_switch.tcl has stale loop_start_addr width. Expected VIO probe_out8 width 9.
  exit /b 2
)
findstr /C:"CONFIG.C_NUM_PROBE_OUT {31}" "%STREAMER_DIR%\create_project_address_switch.tcl" >nul || (
  echo ERROR: create_project_address_switch.tcl has stale VIO probe count. Expected 31 output probes for scan and analog bus support.
  exit /b 2
)
findstr /C:"CONFIG.C_PROBE_OUT15_WIDTH {8}" "%STREAMER_DIR%\create_project_address_switch.tcl" >nul || (
  echo ERROR: create_project_address_switch.tcl has stale scan_prog_addr width. Expected VIO probe_out15 width 8.
  exit /b 2
)
findstr /C:"CONFIG.C_PROBE_OUT18_WIDTH {9}" "%STREAMER_DIR%\create_project_address_switch.tcl" >nul || (
  echo ERROR: create_project_address_switch.tcl has stale scan_count width. Expected VIO probe_out18 width 9.
  exit /b 2
)
findstr /C:"CONFIG.C_PROBE_OUT29_WIDTH {28}" "%STREAMER_DIR%\create_project_address_switch.tcl" >nul || (
  echo ERROR: create_project_address_switch.tcl has stale bus_counts width. Expected VIO probe_out29 width 28.
  exit /b 2
)
findstr /C:"CONFIG.C_PROBE_OUT30_WIDTH {40}" "%STREAMER_DIR%\create_project_address_switch.tcl" >nul || (
  echo ERROR: create_project_address_switch.tcl has stale scan bus value width. Expected VIO probe_out30 width 40.
  exit /b 2
)
if not exist "!ZLC_SELECTED_XDC!" (
  echo ERROR: missing address-switch XDC: !ZLC_SELECTED_XDC!
  echo Restore references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc or set ZLC_PS_XDC.
  exit /b 2
)
findstr /C:"[get_ports trig]" "!ZLC_SELECTED_XDC!" >nul || (
  echo ERROR: selected XDC does not define the address_switch trig output.
  exit /b 2
)
findstr /C:"<PIN_CH" "!ZLC_SELECTED_XDC!" >nul && (
  echo ERROR: selected XDC still contains PIN_CH placeholders: !ZLC_SELECTED_XDC!
  exit /b 2
)
echo ZLC address-switch source contract: channels=62 max_edges=512 scan_rows=256 scan_bus_values=4x10 bus_segments=4x64 vio_outputs=31 edge_addr_width=9 scan_addr_width=8 bus_seg_addr_width=6 prog_count_width=10
echo ZLC address-switch XDC: !ZLC_SELECTED_XDC!
exit /b 0

:zlc_default_paths
if defined ZLC_PS_BUILD_ROOT if "!ZLC_PS_BUILD_ROOT: =!"=="" set "ZLC_PS_BUILD_ROOT="
if defined ZLC_PS_PROJECT_DIR if "!ZLC_PS_PROJECT_DIR: =!"=="" set "ZLC_PS_PROJECT_DIR="
if defined ZLC_PS_STATE_DIR if "!ZLC_PS_STATE_DIR: =!"=="" set "ZLC_PS_STATE_DIR="
if not defined ZLC_PS_BUILD_ROOT set "ZLC_PS_BUILD_ROOT=%FPGA_DIR%build"
if not exist "!ZLC_PS_BUILD_ROOT!\" mkdir "!ZLC_PS_BUILD_ROOT!" >nul 2>nul

:zlc_have_build_root
if not defined ZLC_PS_PROJECT_DIR set "ZLC_PS_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\address_switch"
if not defined ZLC_PS_STATE_DIR set "ZLC_PS_STATE_DIR=%ZLC_PS_BUILD_ROOT%\state_address_switch"
echo ZLC build root: %ZLC_PS_BUILD_ROOT%
if defined ZLC_PS_PROJECT_DIR if /I not "!ZLC_PS_PROJECT_DIR:pulse_streamer\build=!"=="!ZLC_PS_PROJECT_DIR!" (
  echo Ignoring old pulse_streamer build-local ZLC_PS_PROJECT_DIR: !ZLC_PS_PROJECT_DIR!
  set "ZLC_PS_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\address_switch"
)
call :zlc_clear_unsafe_artifact ZLC_PS_VIVADO_PROJECT
call :zlc_clear_unsafe_artifact ZLC_PS_VIVADO_BIT
call :zlc_clear_unsafe_artifact ZLC_PS_VIVADO_LTX
call :zlc_clear_unsafe_artifact ZLC_VIVADO_PROJECT
call :zlc_clear_unsafe_artifact ZLC_VIVADO_BIT
call :zlc_clear_unsafe_artifact ZLC_VIVADO_LTX
exit /b 0

:zlc_clear_unsafe_artifact
set "ZLC_ARTIFACT_VAR=%~1"
set "ZLC_ARTIFACT_VALUE=!%~1!"
if not defined ZLC_ARTIFACT_VALUE exit /b 0
if /I not "!ZLC_ARTIFACT_VALUE:pulse_streamer\build=!"=="!ZLC_ARTIFACT_VALUE!" (
  echo Ignoring old pulse_streamer build-local %~1: !ZLC_ARTIFACT_VALUE!
  set "%~1="
)
exit /b 0

:zlc_find_python
if defined ZLC_PY_CMD (
  call :zlc_normalize_python_cmd
  goto zlc_python_found
)
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
echo Could not find python or py. Run install_requirements.bat first.
exit /b 1
:zlc_python_found
set "ZLC_PY_ARG=%ZLC_PY_CMD:"=""%"
echo ZLC Python: %ZLC_PY_CMD%
exit /b 0

:zlc_normalize_python_cmd
set "ZLC_PY_RAW=%ZLC_PY_CMD:"=%"
if exist "%ZLC_PY_RAW%" set "ZLC_PY_CMD=call "%ZLC_PY_RAW%""
exit /b 0

:zlc_find_vivado
if not "%ZLC_PS_VIVADO_BIN%"=="" goto zlc_vivado_found
if not "%ZLC_VIVADO_BIN%"=="" set "ZLC_PS_VIVADO_BIN=%ZLC_VIVADO_BIN%"
if not "%ZLC_PS_VIVADO_BIN%"=="" goto zlc_vivado_found
for %%V in (2019.1 2019.2 2020.1 2020.2 2021.1 2021.2 2022.1 2022.2 2023.1 2023.2 2024.1 2024.2 2025.1 2025.2) do (
  if exist "C:\Xilinx\Vivado\%%V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\%%V\bin\vivado.bat"
  if exist "D:\Xilinx\Vivado\%%V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=D:\Xilinx\Vivado\%%V\bin\vivado.bat"
)
if not "%ZLC_PS_VIVADO_BIN%"=="" goto zlc_vivado_found
for /f "delims=" %%I in ('where vivado.bat 2^>nul') do if "%ZLC_PS_VIVADO_BIN%"=="" set "ZLC_PS_VIVADO_BIN=%%I"
if not "%ZLC_PS_VIVADO_BIN%"=="" goto zlc_vivado_found
where vivado >nul 2>nul
if not errorlevel 1 set "ZLC_PS_VIVADO_BIN=vivado"
if not "%ZLC_PS_VIVADO_BIN%"=="" goto zlc_vivado_found
echo Could not find Vivado. Set ZLC_PS_VIVADO_BIN to vivado.bat.
exit /b 1
:zlc_vivado_found
echo ZLC Vivado: %ZLC_PS_VIVADO_BIN%
exit /b 0
