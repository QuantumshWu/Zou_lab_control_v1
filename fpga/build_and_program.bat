@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I not "%~1"=="--inner" (
  set "ZLC_ACTION=build/program"
  if /I "%~1"=="--check" set "ZLC_ACTION=62ch synth check"
  if /I "%~1"=="--diagnose" set "ZLC_ACTION=hardware diagnose"
  if /I "%~1"=="--build-only" set "ZLC_ACTION=62ch build"
  if /I "%~1"=="--program-only" set "ZLC_ACTION=62ch program"
  call "%~f0" --inner %*
  set "ZLC_STATUS=!ERRORLEVEL!"
  if "!ZLC_STATUS!"=="0" (
    if "%~1"=="--help" exit /b 0
    if "%~1"=="/?" exit /b 0
    echo.
    echo ZLC !ZLC_ACTION! completed successfully.
    echo You can close this window, or press any key to exit.
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

set "MODE=all"
if "%~1"=="--help" goto zlc_help
if "%~1"=="/?" goto zlc_help
if /I "%~1"=="--check" set "MODE=check"
if /I "%~1"=="--diagnose" set "MODE=diagnose"
if /I "%~1"=="--build-only" set "MODE=build"
if /I "%~1"=="--program-only" set "MODE=program"
if not "%~1"=="" if "%MODE%"=="all" (
  echo Unknown option: %~1
  echo.
  goto zlc_help
)

call :zlc_find_vivado
if errorlevel 1 exit /b 1
call :zlc_default_paths
call :zlc_verify_address_switch_sources
if errorlevel 1 exit /b 1
call :zlc_print_capacity_estimate

if /I "%MODE%"=="check" (
  call :zlc_run_tcl "check_address_switch_synth.tcl"
  exit /b !ERRORLEVEL!
)

if /I "%MODE%"=="diagnose" (
  call :zlc_run_tcl "diagnose_hw_target.tcl"
  exit /b !ERRORLEVEL!
)

if /I "%MODE%"=="program" goto zlc_program

echo ZLC FPGA pulse streamer: build address-switch bitstream
call :zlc_run_tcl "create_project_address_switch.tcl"
if errorlevel 1 exit /b 1

if /I "%MODE%"=="build" exit /b 0

:zlc_program
echo ZLC FPGA pulse streamer: program address-switch bitstream
call :zlc_run_tcl "program_fpga_address_switch.tcl"
exit /b %ERRORLEVEL%

:zlc_help
echo Build/program the address-switch ZLC FPGA pulse-streamer.
echo.
echo Usage:
echo   fpga\build_and_program.bat              Build and program address-switch outputs
echo   fpga\build_and_program.bat --build-only Build only
echo   fpga\build_and_program.bat --program-only Program existing bit/LTX
echo   fpga\build_and_program.bat --check      No-XDC synthesis self-check
echo   fpga\build_and_program.bat --diagnose   List Vivado hw targets/devices
echo.
echo Real build XDC:
echo   references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc
echo   This original XDC is the default address_switch pin map.
echo   For a different board/cable map, set:
echo   set ZLC_PS_XDC=C:\path\to\board_address_switch.xdc
echo.
echo Optional:
echo   set ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.2\bin\vivado.bat
echo   set ZLC_PS_PROJECT_DIR=%%CD%%\fpga\build\address_switch
echo   set ZLC_PS_RESOURCE_TARGET_PCT=70
echo   set ZLC_PS_MAX_SCAN_POINTS=256
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
findstr /C:"localparam integer CHANNEL_COUNT = 62" "%STREAMER_DIR%\zlc_pulse_streamer_top_address_switch.v" >nul || (
  echo ERROR: pulse-streamer top is not the 62-output address-switch wrapper.
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
findstr /C:"zlc_safe_project_dir" "%STREAMER_DIR%\create_project_address_switch.tcl" >nul || (
  echo ERROR: create_project_address_switch.tcl is missing the Vivado path-length guard.
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

:zlc_print_capacity_estimate
if "%ZLC_PS_RESOURCE_TARGET_PCT%"=="" set "ZLC_PS_RESOURCE_TARGET_PCT=70"
if "%ZLC_PS_MAX_SCAN_POINTS%"=="" set "ZLC_PS_MAX_SCAN_POINTS=256"
where python >nul 2>nul
if errorlevel 1 (
  echo ZLC capacity estimate skipped: python was not found on PATH.
  echo ZLC resource target: %ZLC_PS_RESOURCE_TARGET_PCT%%% LUT, max_edges=512, scan_rows=%ZLC_PS_MAX_SCAN_POINTS%, scan_bus_values=4x10, bus_segments=4x64
  exit /b 0
)
pushd "%REPO_ROOT%"
set "PYTHONPATH=%CD%;%PYTHONPATH%"
python -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer capacity_estimate --channel-count 62 --max-edges 512 --max-scan-points %ZLC_PS_MAX_SCAN_POINTS% --tick-width 32 --resource-target-pct %ZLC_PS_RESOURCE_TARGET_PCT%
popd
exit /b 0

:zlc_default_paths
if defined ZLC_PS_BUILD_ROOT if "!ZLC_PS_BUILD_ROOT: =!"=="" set "ZLC_PS_BUILD_ROOT="
if defined ZLC_PS_PROJECT_DIR if "!ZLC_PS_PROJECT_DIR: =!"=="" set "ZLC_PS_PROJECT_DIR="
if defined ZLC_PS_CHECK_PROJECT_DIR if "!ZLC_PS_CHECK_PROJECT_DIR: =!"=="" set "ZLC_PS_CHECK_PROJECT_DIR="
if defined ZLC_PS_LOG_DIR if "!ZLC_PS_LOG_DIR: =!"=="" set "ZLC_PS_LOG_DIR="
if not defined ZLC_PS_BUILD_ROOT set "ZLC_PS_BUILD_ROOT=%FPGA_DIR%build"
if not exist "!ZLC_PS_BUILD_ROOT!\" mkdir "!ZLC_PS_BUILD_ROOT!" >nul 2>nul

:zlc_have_build_root
if not defined ZLC_PS_PROJECT_DIR set "ZLC_PS_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\address_switch"
if not defined ZLC_PS_CHECK_PROJECT_DIR set "ZLC_PS_CHECK_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\check_address_switch"
if not defined ZLC_PS_LOG_DIR set "ZLC_PS_LOG_DIR=%ZLC_PS_BUILD_ROOT%\logs"
echo ZLC build root: %ZLC_PS_BUILD_ROOT%
if defined ZLC_PS_PROJECT_DIR if /I not "!ZLC_PS_PROJECT_DIR:pulse_streamer\build=!"=="!ZLC_PS_PROJECT_DIR!" (
  echo Ignoring old pulse_streamer build-local ZLC_PS_PROJECT_DIR: !ZLC_PS_PROJECT_DIR!
  set "ZLC_PS_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\address_switch"
)
if defined ZLC_PS_CHECK_PROJECT_DIR if /I not "!ZLC_PS_CHECK_PROJECT_DIR:pulse_streamer\build=!"=="!ZLC_PS_CHECK_PROJECT_DIR!" (
  echo Ignoring old pulse_streamer build-local ZLC_PS_CHECK_PROJECT_DIR: !ZLC_PS_CHECK_PROJECT_DIR!
  set "ZLC_PS_CHECK_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\check_address_switch"
)
call :zlc_clear_unsafe_artifact ZLC_PS_VIVADO_PROJECT
call :zlc_clear_unsafe_artifact ZLC_PS_VIVADO_BIT
call :zlc_clear_unsafe_artifact ZLC_PS_VIVADO_LTX
call :zlc_clear_unsafe_artifact ZLC_PS_BIT
call :zlc_clear_unsafe_artifact ZLC_PS_LTX
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

echo Could not find Vivado.
echo Set ZLC_PS_VIVADO_BIN to the full path of vivado.bat.
exit /b 1

:zlc_vivado_found
echo ZLC Vivado: %ZLC_PS_VIVADO_BIN%
exit /b 0

:zlc_run_tcl
set "TCL_NAME=%~1"
set "TCL_STEM=%~n1"
set "DIRECT_TCL=%STREAMER_DIR%\%TCL_NAME%"
if not exist "%DIRECT_TCL%" (
  echo Missing Tcl script: %DIRECT_TCL%
  exit /b 2
)
if not exist "%ZLC_PS_LOG_DIR%" mkdir "%ZLC_PS_LOG_DIR%" >nul 2>nul

echo ZLC direct Vivado path: %DIRECT_TCL%
if /I "%TCL_NAME%"=="check_address_switch_synth.tcl" (
  echo ZLC check project dir: !ZLC_PS_CHECK_PROJECT_DIR!
) else (
  echo ZLC project dir: !ZLC_PS_PROJECT_DIR!
)
call "%ZLC_PS_VIVADO_BIN%" -mode batch -journal "!ZLC_PS_LOG_DIR!\!TCL_STEM!.jou" -log "!ZLC_PS_LOG_DIR!\!TCL_STEM!.log" -source "%DIRECT_TCL%"
exit /b %ERRORLEVEL%
