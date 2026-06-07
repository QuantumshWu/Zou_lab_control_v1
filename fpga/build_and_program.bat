@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I not "%~1"=="--inner" (
  set "ZLC_ACTION=build/program"
  if /I "%~1"=="--check" set "ZLC_ACTION=edge-table loader synth check"
  if /I "%~1"=="--diagnose" set "ZLC_ACTION=hardware diagnose"
  if /I "%~1"=="--build-only" set "ZLC_ACTION=edge-table loader build"
  if /I "%~1"=="--program-only" set "ZLC_ACTION=edge-table loader program"
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

if /I "%ZLC_PS_VARIANT%"=="d" (
  set "ZLC_CREATE_TCL=create_project_d.tcl"
  set "ZLC_PROGRAM_TCL=program_fpga_d.tcl"
  set "ZLC_PROJ_SUB=d"
) else (
  set "ZLC_CREATE_TCL=create_project_loader.tcl"
  set "ZLC_PROGRAM_TCL=program_fpga_loader.tcl"
  set "ZLC_PROJ_SUB=l"
)

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
if /I not "%ZLC_PS_VARIANT%"=="d" call :zlc_verify_loader_sources
if errorlevel 1 exit /b 1
call :zlc_print_capacity_estimate

if /I "%MODE%"=="diagnose" (
  call :zlc_run_tcl "diagnose_hw_target.tcl"
  exit /b !ERRORLEVEL!
)

if /I "%MODE%"=="program" goto zlc_program

echo ZLC FPGA pulse streamer: build edge-table loader bitstream (JTAG-to-AXI)
call :zlc_run_tcl "!ZLC_CREATE_TCL!"
if errorlevel 1 exit /b 1

if /I "%MODE%"=="build" exit /b 0
if /I "%MODE%"=="check" exit /b 0

:zlc_program
echo ZLC FPGA pulse streamer: program edge-table loader bitstream
call :zlc_run_tcl "!ZLC_PROGRAM_TCL!"
exit /b %ERRORLEVEL%

:zlc_help
echo Build/program the per-channel edge-table loader ZLC FPGA pulse-streamer.
echo Control path: JTAG-to-AXI master -^> AXI BRAM controller -^> dual-port BRAM.
echo.
echo Usage:
echo   fpga\build_and_program.bat              Build and program edge-table loader outputs
echo   fpga\build_and_program.bat --build-only Build only
echo   fpga\build_and_program.bat --program-only Program existing bitstream
echo   fpga\build_and_program.bat --check      Build only (alias of --build-only)
echo   fpga\build_and_program.bat --diagnose   List Vivado hw targets/devices
echo.
echo Real build XDC:
echo   references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc
echo   This original XDC is the default 62-output pin map (edge-table loader top reuses it).
echo   For a different board/cable map, set:
echo   set ZLC_PS_XDC=C:\path\to\board.xdc
echo.
echo Optional:
echo   set ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.2\bin\vivado.bat
echo   set ZLC_PS_PROJECT_DIR=%%CD%%\fpga\build\l
echo   set ZLC_PS_RESOURCE_TARGET_PCT=70
exit /b 0

:zlc_verify_loader_sources
set "ZLC_DEFAULT_XDC=%REPO_ROOT%\references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc"
if not defined ZLC_PS_XDC set "ZLC_PS_XDC=%ZLC_DEFAULT_XDC%"
set "ZLC_SELECTED_XDC=%ZLC_PS_XDC%"
if not exist "%STREAMER_DIR%\zlc_axi_program_loader.v" (
  echo ERROR: missing edge-table loader playback engine HDL: %STREAMER_DIR%\zlc_axi_program_loader.v
  exit /b 2
)
if not exist "%STREAMER_DIR%\zlc_pulse_streamer_loader_top.v" (
  echo ERROR: missing edge-table loader top HDL: %STREAMER_DIR%\zlc_pulse_streamer_loader_top.v
  exit /b 2
)
findstr /C:"zlc_axi_program_loader.v" "%STREAMER_DIR%\create_project_loader.tcl" >nul || (
  echo ERROR: create_project_loader.tcl does not read the edge-table loader engine HDL.
  exit /b 2
)
if not exist "%STREAMER_DIR%\create_project_loader.tcl" (
  echo ERROR: missing edge-table loader build Tcl: %STREAMER_DIR%\create_project_loader.tcl
  exit /b 2
)
findstr /C:"localparam integer CHANNEL_COUNT = 62" "%STREAMER_DIR%\zlc_pulse_streamer_loader_top.v" >nul || (
  echo ERROR: edge-table loader top is not the 62-output wrapper.
  exit /b 2
)
findstr /C:"localparam integer NUM_SLOTS = 4" "%STREAMER_DIR%\zlc_pulse_streamer_loader_top.v" >nul || (
  echo ERROR: edge-table loader top is not the 5-slot build. Expected NUM_SLOTS = 4.
  exit /b 2
)
findstr /C:"module zlc_pulse_streamer_loader_top" "%STREAMER_DIR%\zlc_pulse_streamer_loader_top.v" >nul || (
  echo ERROR: edge-table loader top module name is wrong.
  exit /b 2
)
findstr /C:"create_ip -name jtag_axi" "%STREAMER_DIR%\create_project_loader.tcl" >nul || (
  echo ERROR: create_project_loader.tcl does not create the JTAG-to-AXI master IP.
  exit /b 2
)
findstr /C:"create_ip -name axi_bram_ctrl" "%STREAMER_DIR%\create_project_loader.tcl" >nul || (
  echo ERROR: create_project_loader.tcl does not create the AXI BRAM controller IP.
  exit /b 2
)
findstr /C:"create_ip -name blk_mem_gen" "%STREAMER_DIR%\create_project_loader.tcl" >nul || (
  echo ERROR: create_project_loader.tcl does not create the dual-port BRAM IP.
  exit /b 2
)
findstr /C:"zlc_safe_project_dir" "%STREAMER_DIR%\create_project_loader.tcl" >nul || (
  echo ERROR: create_project_loader.tcl is missing the Vivado path-length guard.
  exit /b 2
)
if not exist "!ZLC_SELECTED_XDC!" (
  echo ERROR: missing edge-table loader XDC: !ZLC_SELECTED_XDC!
  echo Restore references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc or set ZLC_PS_XDC.
  exit /b 2
)
findstr /C:"[get_ports trig]" "!ZLC_SELECTED_XDC!" >nul || (
  echo ERROR: selected XDC does not define the trig output.
  exit /b 2
)
findstr /C:"<PIN_CH" "!ZLC_SELECTED_XDC!" >nul && (
  echo ERROR: selected XDC still contains PIN_CH placeholders: !ZLC_SELECTED_XDC!
  exit /b 2
)
echo ZLC edge-table loader source contract: channels=62 num_slots=4 control=JTAG-to-AXI (jtag_axi+axi_bram_ctrl+blk_mem_gen)
echo ZLC edge-table loader XDC: !ZLC_SELECTED_XDC!
exit /b 0

:zlc_print_capacity_estimate
where python >nul 2>nul
if errorlevel 1 (
  echo ZLC BRAM estimate skipped: python was not found on PATH.
  exit /b 0
)
pushd "%REPO_ROOT%"
set "PYTHONPATH=%CD%;%PYTHONPATH%"
python -c "from Zou_lab_control.neutral_atom.devices.edgetable_image import EdgeTableImageParams as P; import math; p=P(); print('ZLC edge-table image span: {} words; program BRAM 32768 words = {}/50 RAMB36'.format(p.total_words, math.ceil(32768/1024)))"
popd
exit /b 0

:zlc_default_paths
if defined ZLC_PS_BUILD_ROOT if "!ZLC_PS_BUILD_ROOT: =!"=="" set "ZLC_PS_BUILD_ROOT="
if defined ZLC_PS_PROJECT_DIR if "!ZLC_PS_PROJECT_DIR: =!"=="" set "ZLC_PS_PROJECT_DIR="
if defined ZLC_PS_LOG_DIR if "!ZLC_PS_LOG_DIR: =!"=="" set "ZLC_PS_LOG_DIR="
if not defined ZLC_PS_BUILD_ROOT set "ZLC_PS_BUILD_ROOT=%FPGA_DIR%build"
if not exist "!ZLC_PS_BUILD_ROOT!\" mkdir "!ZLC_PS_BUILD_ROOT!" >nul 2>nul
rem Short project dir name "r" is the Vivado debug-core path-length fix.
if not defined ZLC_PS_PROJECT_DIR set "ZLC_PS_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\!ZLC_PROJ_SUB!"
if not defined ZLC_PS_LOG_DIR set "ZLC_PS_LOG_DIR=%ZLC_PS_BUILD_ROOT%\logs"
echo ZLC build root: %ZLC_PS_BUILD_ROOT%
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
echo ZLC project dir: !ZLC_PS_PROJECT_DIR!
call "%ZLC_PS_VIVADO_BIN%" -mode batch -journal "!ZLC_PS_LOG_DIR!\!TCL_STEM!.jou" -log "!ZLC_PS_LOG_DIR!\!TCL_STEM!.log" -source "%DIRECT_TCL%"
exit /b %ERRORLEVEL%
