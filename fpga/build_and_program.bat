@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I not "%~1"=="--inner" (
  set "ZLC_ACTION=build/program"
  if /I "%~1"=="--check" set "ZLC_ACTION=40ch synth check"
  if /I "%~1"=="--diagnose" set "ZLC_ACTION=hardware diagnose"
  if /I "%~1"=="--build-only" set "ZLC_ACTION=40ch build"
  if /I "%~1"=="--program-only" set "ZLC_ACTION=40ch program"
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
call :zlc_verify_40ch_sources
if errorlevel 1 exit /b 1

if /I "%MODE%"=="check" (
  call :zlc_run_tcl "check_40ch_synth.tcl"
  exit /b !ERRORLEVEL!
)

if /I "%MODE%"=="diagnose" (
  call :zlc_run_tcl "diagnose_hw_target.tcl"
  exit /b !ERRORLEVEL!
)

if /I "%MODE%"=="program" goto zlc_program

echo ZLC FPGA pulse streamer: build 40ch bitstream
call :zlc_run_tcl "create_project_40ch.tcl"
if errorlevel 1 exit /b 1

if /I "%MODE%"=="build" exit /b 0

:zlc_program
echo ZLC FPGA pulse streamer: program 40ch bitstream
call :zlc_run_tcl "program_fpga_40ch.tcl"
exit /b %ERRORLEVEL%

:zlc_help
echo Build/program the 40-channel ZLC FPGA pulse-streamer.
echo.
echo Usage:
echo   fpga\build_and_program.bat              Build and program 40ch
echo   fpga\build_and_program.bat --build-only Build 40ch only
echo   fpga\build_and_program.bat --program-only Program existing 40ch bit/LTX
echo   fpga\build_and_program.bat --check      No-XDC 40ch synthesis self-check
echo   fpga\build_and_program.bat --diagnose   List Vivado hw targets/devices
echo.
echo Real build XDC:
echo   fpga\pulse_streamer\zlc_pulse_streamer_40ch.xdc
echo   This checked-in XDC is derived from the old address_switch pin map.
echo   For a different board/cable map, set:
echo   set ZLC_PS_40CH_XDC=C:\path\to\board_40ch.xdc
echo.
echo Optional:
echo   set ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.2\bin\vivado.bat
echo   set ZLC_PS_PROJECT_DIR=%%CD%%\fpga\build\p40
exit /b 0

:zlc_verify_40ch_sources
set "ZLC_DEFAULT_XDC=%STREAMER_DIR%\zlc_pulse_streamer_40ch.xdc"
if not defined ZLC_PS_40CH_XDC if not defined ZLC_PS_XDC set "ZLC_PS_40CH_XDC=%ZLC_DEFAULT_XDC%"
set "ZLC_SELECTED_XDC=%ZLC_PS_40CH_XDC%"
if not defined ZLC_SELECTED_XDC set "ZLC_SELECTED_XDC=%ZLC_PS_XDC%"
if not defined ZLC_PS_40CH_XDC set "ZLC_PS_40CH_XDC=%ZLC_SELECTED_XDC%"
if not defined ZLC_PS_XDC set "ZLC_PS_XDC=%ZLC_SELECTED_XDC%"
if not exist "%STREAMER_DIR%\zlc_pulse_streamer_top_40ch.v" (
  echo ERROR: missing 40ch top HDL: %STREAMER_DIR%\zlc_pulse_streamer_top_40ch.v
  exit /b 2
)
if not exist "%STREAMER_DIR%\create_project_40ch.tcl" (
  echo ERROR: missing 40ch build Tcl: %STREAMER_DIR%\create_project_40ch.tcl
  exit /b 2
)
findstr /C:".EDGE_ADDR_WIDTH(10)" "%STREAMER_DIR%\zlc_pulse_streamer_top_40ch.v" >nul || (
  echo ERROR: 40ch top is not the 1024-edge build. Expected .EDGE_ADDR_WIDTH^(10^).
  exit /b 2
)
findstr /C:"CONFIG.C_PROBE_OUT3_WIDTH {10}" "%STREAMER_DIR%\create_project_40ch.tcl" >nul || (
  echo ERROR: create_project_40ch.tcl has stale prog_addr width. Expected VIO probe_out3 width 10.
  exit /b 2
)
findstr /C:"CONFIG.C_PROBE_OUT6_WIDTH {11}" "%STREAMER_DIR%\create_project_40ch.tcl" >nul || (
  echo ERROR: create_project_40ch.tcl has stale prog_count width. Expected VIO probe_out6 width 11.
  exit /b 2
)
findstr /C:"zlc_safe_project_dir" "%STREAMER_DIR%\create_project_40ch.tcl" >nul || (
  echo ERROR: create_project_40ch.tcl is missing the Vivado path-length guard.
  exit /b 2
)
if not exist "!ZLC_SELECTED_XDC!" (
  echo ERROR: missing 40ch XDC: !ZLC_SELECTED_XDC!
  echo Restore fpga\pulse_streamer\zlc_pulse_streamer_40ch.xdc. It is derived from references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc.
  exit /b 2
)
findstr /C:"[get_ports {ch[39]}]" "!ZLC_SELECTED_XDC!" >nul || (
  echo ERROR: selected XDC does not define ch[39]; this is not a full 40ch pulse-streamer XDC.
  exit /b 2
)
findstr /C:"<PIN_CH" "!ZLC_SELECTED_XDC!" >nul && (
  echo ERROR: selected XDC still contains PIN_CH placeholders: !ZLC_SELECTED_XDC!
  exit /b 2
)
echo ZLC 40ch source contract: channels=40 max_edges=1024 edge_addr_width=10 prog_count_width=11
echo ZLC 40ch XDC: !ZLC_SELECTED_XDC!
exit /b 0

:zlc_default_paths
if defined ZLC_PS_BUILD_ROOT if "!ZLC_PS_BUILD_ROOT: =!"=="" set "ZLC_PS_BUILD_ROOT="
if defined ZLC_PS_PROJECT_DIR if "!ZLC_PS_PROJECT_DIR: =!"=="" set "ZLC_PS_PROJECT_DIR="
if defined ZLC_PS_CHECK_PROJECT_DIR if "!ZLC_PS_CHECK_PROJECT_DIR: =!"=="" set "ZLC_PS_CHECK_PROJECT_DIR="
if defined ZLC_PS_LOG_DIR if "!ZLC_PS_LOG_DIR: =!"=="" set "ZLC_PS_LOG_DIR="
if not defined ZLC_PS_BUILD_ROOT set "ZLC_PS_BUILD_ROOT=%FPGA_DIR%build"
if not exist "!ZLC_PS_BUILD_ROOT!\" mkdir "!ZLC_PS_BUILD_ROOT!" >nul 2>nul

:zlc_have_build_root
if not defined ZLC_PS_PROJECT_DIR set "ZLC_PS_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\p40"
if not defined ZLC_PS_CHECK_PROJECT_DIR set "ZLC_PS_CHECK_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\c40"
if not defined ZLC_PS_LOG_DIR set "ZLC_PS_LOG_DIR=%ZLC_PS_BUILD_ROOT%\logs"
echo ZLC build root: %ZLC_PS_BUILD_ROOT%
if defined ZLC_PS_PROJECT_DIR if /I not "!ZLC_PS_PROJECT_DIR:pulse_streamer\build=!"=="!ZLC_PS_PROJECT_DIR!" (
  echo Ignoring old pulse_streamer build-local ZLC_PS_PROJECT_DIR: !ZLC_PS_PROJECT_DIR!
  set "ZLC_PS_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\p40"
)
if defined ZLC_PS_CHECK_PROJECT_DIR if /I not "!ZLC_PS_CHECK_PROJECT_DIR:pulse_streamer\build=!"=="!ZLC_PS_CHECK_PROJECT_DIR!" (
  echo Ignoring old pulse_streamer build-local ZLC_PS_CHECK_PROJECT_DIR: !ZLC_PS_CHECK_PROJECT_DIR!
  set "ZLC_PS_CHECK_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\c40"
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
if /I "%TCL_NAME%"=="check_40ch_synth.tcl" (
  echo ZLC check project dir: !ZLC_PS_CHECK_PROJECT_DIR!
) else (
  echo ZLC project dir: !ZLC_PS_PROJECT_DIR!
)
call "%ZLC_PS_VIVADO_BIN%" -mode batch -journal "!ZLC_PS_LOG_DIR!\!TCL_STEM!.jou" -log "!ZLC_PS_LOG_DIR!\!TCL_STEM!.log" -source "%DIRECT_TCL%"
exit /b %ERRORLEVEL%
