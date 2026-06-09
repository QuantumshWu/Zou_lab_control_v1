@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I not "%~1"=="--inner" (
  set "ZLC_ACTION=build/program"
  if /I "%~1"=="--check" set "ZLC_ACTION=synth check"
  if /I "%~1"=="--diagnose" set "ZLC_ACTION=hardware diagnose"
  if /I "%~1"=="--build-only" set "ZLC_ACTION=build"
  if /I "%~1"=="--program-only" set "ZLC_ACTION=program"
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
set "ZLC_CREATE_TCL=create_project.tcl"
set "ZLC_PROGRAM_TCL=program_fpga.tcl"
set "ZLC_PROJ_SUB=ps"

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
call :zlc_verify_sources
if errorlevel 1 exit /b 1
call :zlc_print_capacity_estimate

if /I "%MODE%"=="diagnose" (
  call :zlc_run_tcl "diagnose_hw_target.tcl"
  exit /b !ERRORLEVEL!
)

if /I "%MODE%"=="program" goto zlc_program

echo ZLC FPGA pulse streamer: build FINAL bitstream (1-tick FIFO prefetch + streaming, JTAG-to-AXI)
call :zlc_run_tcl "!ZLC_CREATE_TCL!"
if errorlevel 1 exit /b 1

if /I "%MODE%"=="build" exit /b 0
if /I "%MODE%"=="check" exit /b 0

:zlc_program
echo ZLC FPGA pulse streamer: program FINAL bitstream
call :zlc_run_tcl "!ZLC_PROGRAM_TCL!"
exit /b %ERRORLEVEL%

:zlc_help
echo Build/program the FINAL ZLC FPGA pulse streamer (one clean design, no variants).
echo Control path: JTAG-to-AXI master -^> AXI BRAM controller -^> edge/scan BRAMs + bus loader.
echo Engine: 1-tick (20 ns) FIFO prefetch + unbounded 2-bank streaming scan.
echo.
echo Usage:
echo   fpga\build_and_program.bat              Build and program
echo   fpga\build_and_program.bat --build-only Build only
echo   fpga\build_and_program.bat --program-only Program existing bitstream
echo   fpga\build_and_program.bat --check      Build only (alias of --build-only)
echo   fpga\build_and_program.bat --diagnose   List Vivado hw targets/devices
echo.
echo Real build XDC:
echo   fpga\board_config\board.xdc
echo   This is the default 62-output board pin map (see fpga\board_config\README.md).
echo   For a different board, replace board.xdc or set:
echo   set ZLC_PS_XDC=C:\path\to\board.xdc
echo.
echo Optional:
echo   set ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.2\bin\vivado.bat
echo   set ZLC_PS_PROJECT_DIR=%%CD%%\fpga\build\ps
exit /b 0

:zlc_verify_sources
set "ZLC_DEFAULT_XDC=%REPO_ROOT%\fpga\board_config\board.xdc"
if not defined ZLC_PS_XDC set "ZLC_PS_XDC=%ZLC_DEFAULT_XDC%"
set "ZLC_SELECTED_XDC=%ZLC_PS_XDC%"
if not exist "%STREAMER_DIR%\zlc_edge_streamer.v" (
  echo ERROR: missing FINAL engine HDL: %STREAMER_DIR%\zlc_edge_streamer.v
  exit /b 2
)
if not exist "%STREAMER_DIR%\zlc_pulse_streamer_top.v" (
  echo ERROR: missing FINAL top HDL: %STREAMER_DIR%\zlc_pulse_streamer_top.v
  exit /b 2
)
if not exist "%STREAMER_DIR%\create_project.tcl" (
  echo ERROR: missing FINAL build Tcl: %STREAMER_DIR%\create_project.tcl
  exit /b 2
)
findstr /C:"zlc_edge_streamer.v" "%STREAMER_DIR%\create_project.tcl" >nul || (
  echo ERROR: create_project.tcl does not read the FINAL engine HDL.
  exit /b 2
)
findstr /C:"module zlc_pulse_streamer_top" "%STREAMER_DIR%\zlc_pulse_streamer_top.v" >nul || (
  echo ERROR: FINAL top module name is wrong.
  exit /b 2
)
findstr /C:"module zlc_edge_streamer" "%STREAMER_DIR%\zlc_edge_streamer.v" >nul || (
  echo ERROR: FINAL engine module name is wrong.
  exit /b 2
)
findstr /C:"create_ip -name jtag_axi" "%STREAMER_DIR%\create_project.tcl" >nul || (
  echo ERROR: create_project.tcl does not create the JTAG-to-AXI master IP.
  exit /b 2
)
findstr /C:"create_ip -name axi_bram_ctrl" "%STREAMER_DIR%\create_project.tcl" >nul || (
  echo ERROR: create_project.tcl does not create the AXI BRAM controller IP.
  exit /b 2
)
findstr /C:"blk_mem_gen_edge_tick" "%STREAMER_DIR%\create_project.tcl" >nul || (
  echo ERROR: create_project.tcl does not create the 3 parallel edge BRAMs.
  exit /b 2
)
findstr /C:"zlc_force_latency2" "%STREAMER_DIR%\create_project.tcl" >nul || (
  echo ERROR: create_project.tcl does not force the edge-BRAM read latency to 2.
  exit /b 2
)
findstr /C:"zlc_safe_project_dir" "%STREAMER_DIR%\create_project.tcl" >nul || (
  echo ERROR: create_project.tcl is missing the Vivado path-length guard.
  exit /b 2
)
if not exist "!ZLC_SELECTED_XDC!" (
  echo ERROR: missing board XDC: !ZLC_SELECTED_XDC!
  echo Put your board pin map at fpga\board_config\board.xdc (see its README) or set ZLC_PS_XDC.
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
echo ZLC FINAL source contract: channels=62 num_slots=4 control=JTAG-to-AXI (jtag_axi+axi_bram_ctrl+5 BRAMs, forced edge latency 2)
echo ZLC FINAL XDC: !ZLC_SELECTED_XDC!
exit /b 0

:zlc_print_capacity_estimate
where python >nul 2>nul
if errorlevel 1 (
  echo ZLC capacity estimate skipped: python was not found on PATH.
  exit /b 0
)
pushd "%REPO_ROOT%"
set "PYTHONPATH=%CD%;%PYTHONPATH%"
python -c "from fpga.pulse_streamer.host.image import solve_capacity as s; r=s('xc7a35tfgg484-2', channel_count=62, target_pct=90); rr=r.resource_report['ramb36']; print('ZLC capacity: {} edges + bank_size {} (2x resident) + UNBOUNDED streaming; RAMB36 {}/{} = {}pct'.format(r.params.max_edges, r.params.bank_size, rr['used'], rr['total'], rr['pct']))"
popd
exit /b 0

:zlc_default_paths
if defined ZLC_PS_BUILD_ROOT if "!ZLC_PS_BUILD_ROOT: =!"=="" set "ZLC_PS_BUILD_ROOT="
if defined ZLC_PS_PROJECT_DIR if "!ZLC_PS_PROJECT_DIR: =!"=="" set "ZLC_PS_PROJECT_DIR="
if defined ZLC_PS_LOG_DIR if "!ZLC_PS_LOG_DIR: =!"=="" set "ZLC_PS_LOG_DIR="
if not defined ZLC_PS_BUILD_ROOT set "ZLC_PS_BUILD_ROOT=%FPGA_DIR%build"
if not exist "!ZLC_PS_BUILD_ROOT!\" mkdir "!ZLC_PS_BUILD_ROOT!" >nul 2>nul
rem In-repo build (fpga\build\ps).  The SHORT subdir "ps" keeps Vivado's deep
rem run/.Xil temp path under the Windows MAX_PATH limit without leaving fpga/.
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
