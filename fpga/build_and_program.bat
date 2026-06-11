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
rem --force-build / --rebuild: rebuild even if the sources are unchanged (default mode otherwise
rem skips the build and programs the existing bitstream when nothing changed).
if /I "%~1"=="--force-build" set "ZLC_FORCE_BUILD=1"
if /I "%~1"=="--rebuild" set "ZLC_FORCE_BUILD=1"
if not "%~1"=="" if "%MODE%"=="all" if not defined ZLC_FORCE_BUILD (
  echo Unknown option: %~1
  echo.
  goto zlc_help
)

call :zlc_find_vivado
if errorlevel 1 exit /b 1
call :zlc_default_paths
call :zlc_verify_sources
if errorlevel 1 exit /b 1
call :zlc_resolve_part
call :zlc_emit_geom
call :zlc_print_capacity_estimate

if /I "%MODE%"=="diagnose" (
  call :zlc_run_tcl "diagnose_hw_target.tcl"
  exit /b !ERRORLEVEL!
)

if /I "%MODE%"=="program" goto zlc_program

rem Skip the (slow) synth+impl when a bitstream already exists and NONE of the sources that go
rem into it changed since it was built -- just program the existing .bit.  Only in the default
rem (build+program) mode, and only when --force-build / --rebuild was NOT given.
call :zlc_check_prebuilt
if /I "%MODE%"=="all" if not defined ZLC_FORCE_BUILD if defined ZLC_PREBUILT (
  echo ZLC bitstream is up to date ^(sources unchanged since last build^) -- skipping build.
  echo ZLC   bit: !ZLC_BIT!
  echo ZLC   ^(force a rebuild with: build_and_program.bat --force-build^)
  goto zlc_program
)

echo ZLC FPGA pulse streamer: build FINAL bitstream (1-tick FIFO prefetch + streaming, JTAG-to-AXI)
call :zlc_run_tcl "!ZLC_CREATE_TCL!"
if errorlevel 1 exit /b 1
call :zlc_save_src_hash

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
echo   fpga\build_and_program.bat              Build (only if sources changed) and program
echo   fpga\build_and_program.bat --force-build Rebuild even if the sources are unchanged
echo   fpga\build_and_program.bat --build-only Build only
echo   fpga\build_and_program.bat --program-only Program existing bitstream
echo   fpga\build_and_program.bat --check      Build only (alias of --build-only)
echo   fpga\build_and_program.bat --diagnose   List Vivado hw targets/devices
echo.
echo The default mode SKIPS the (slow) synth+impl when a bitstream already exists and none of
echo the sources that go into it (engine/top HDL, create/program tcl, board XDC, streamer_config,
echo geom tcl) changed since it was built -- it just programs the existing .bit.  --force-build
echo (or --rebuild) forces a rebuild.  The build-cache key is fpga\build\ps\.zlc_src_hash.
echo.
echo Real build XDC:
echo   fpga\board_config\board.xdc
echo   This is the default 62-output board pin map ^(see fpga\board_config\README.md^).
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
  echo Put your board pin map at fpga\board_config\board.xdc -- see its README -- or set ZLC_PS_XDC.
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

:zlc_resolve_part
rem Single source: take the synthesis part from fpga\board_config\streamer_config.json
rem (unless ZLC_PS_FPGA_PART is already set) and export it so create_project.tcl targets
rem the configured board.  Pure read; never fails the build if python/config is missing.
if not "%ZLC_PS_FPGA_PART%"=="" goto :zlc_resolve_part_done
where python >nul 2>nul
if errorlevel 1 goto :zlc_resolve_part_done
set "ZLC_CFG_JSON=%REPO_ROOT%\fpga\board_config\streamer_config.json"
if not exist "%ZLC_CFG_JSON%" goto :zlc_resolve_part_done
for /f "delims=" %%I in ('python -c "import json;print(json.load(open(r'%ZLC_CFG_JSON%'))['fpga_part'])" 2^>nul') do set "ZLC_PS_FPGA_PART=%%I"
:zlc_resolve_part_done
if not "%ZLC_PS_FPGA_PART%"=="" echo ZLC synthesis part: %ZLC_PS_FPGA_PART% (from streamer_config.json / env)
exit /b 0

:zlc_emit_geom
rem Single source: generate the Vivado geometry tcl (BRAM sizes + top -generic overrides) from
rem streamer_config.json so editing the config changes the SYNTHESIZED bitstream (e.g.
rem EVT_FIFO_DEPTH 256->128).  create_project.tcl sources it via ZLC_PS_GEOM_TCL; when this step
rem is skipped the tcl uses its literal defaults.  Pure read; never fails the build.
if not "%ZLC_PS_GEOM_TCL%"=="" goto :zlc_emit_geom_done
where python >nul 2>nul
if errorlevel 1 goto :zlc_emit_geom_done
set "ZLC_GEOM_OUT=%ZLC_PS_BUILD_ROOT%\geom.tcl"
pushd "%REPO_ROOT%"
set "PYTHONPATH=%CD%;%PYTHONPATH%"
python -m fpga.pulse_streamer.host.image --emit-geom-tcl "%ZLC_GEOM_OUT%" >nul 2>nul && set "ZLC_PS_GEOM_TCL=%ZLC_GEOM_OUT%"
popd
:zlc_emit_geom_done
if not "%ZLC_PS_GEOM_TCL%"=="" echo ZLC geometry tcl: %ZLC_PS_GEOM_TCL% (from streamer_config.json)
exit /b 0

:zlc_compute_src_hash
rem Hash the files that go into the bitstream (engine + top + build tcl + program tcl + XDC +
rem board config + the generated geom tcl).  ZLC_SRC_HASH stays empty if python is missing ->
rem then the skip is simply disabled (always rebuild), never a wrong skip.
set "ZLC_SRC_HASH="
where python >nul 2>nul
if errorlevel 1 exit /b 0
set "ZLC_HASH_GEOM="
if defined ZLC_PS_GEOM_TCL if exist "%ZLC_PS_GEOM_TCL%" set "ZLC_HASH_GEOM=%ZLC_PS_GEOM_TCL%"
for /f "delims=" %%H in ('python "%STREAMER_DIR%\host\src_hash.py" "%STREAMER_DIR%\zlc_edge_streamer.v" "%STREAMER_DIR%\zlc_pulse_streamer_top.v" "%STREAMER_DIR%\!ZLC_CREATE_TCL!" "%STREAMER_DIR%\!ZLC_PROGRAM_TCL!" "!ZLC_SELECTED_XDC!" "%REPO_ROOT%\fpga\board_config\streamer_config.json" "!ZLC_HASH_GEOM!" 2^>nul') do set "ZLC_SRC_HASH=%%H"
exit /b 0

:zlc_check_prebuilt
rem Set ZLC_PREBUILT=1 iff the bitstream exists AND the stored source hash matches the current
rem sources (i.e. nothing that affects the .bit changed since it was built).
set "ZLC_PREBUILT="
set "ZLC_BIT=%ZLC_PS_PROJECT_DIR%\%ZLC_PROJ_SUB%.runs\impl_1\zlc_pulse_streamer_top.bit"
set "ZLC_HASHFILE=%ZLC_PS_PROJECT_DIR%\.zlc_src_hash"
if not exist "%ZLC_BIT%" exit /b 0
if not exist "%ZLC_HASHFILE%" exit /b 0
call :zlc_compute_src_hash
if not defined ZLC_SRC_HASH exit /b 0
set "ZLC_STORED_HASH="
set /p ZLC_STORED_HASH=<"%ZLC_HASHFILE%"
if "%ZLC_STORED_HASH%"=="%ZLC_SRC_HASH%" set "ZLC_PREBUILT=1"
exit /b 0

:zlc_save_src_hash
rem Record the current source hash next to the freshly built bitstream so the next default-mode
rem run can skip the build when nothing changed.
call :zlc_compute_src_hash
if not defined ZLC_SRC_HASH exit /b 0
if not exist "%ZLC_PS_PROJECT_DIR%\" exit /b 0
> "%ZLC_PS_PROJECT_DIR%\.zlc_src_hash" echo %ZLC_SRC_HASH%
exit /b 0

:zlc_print_capacity_estimate
where python >nul 2>nul
if errorlevel 1 (
  echo ZLC capacity estimate skipped: python was not found on PATH.
  exit /b 0
)
set "ZLC_EST_PART=%ZLC_PS_FPGA_PART%"
if "%ZLC_EST_PART%"=="" set "ZLC_EST_PART=xc7a35tfgg484-2"
pushd "%REPO_ROOT%"
set "PYTHONPATH=%CD%;%PYTHONPATH%"
python -m fpga.pulse_streamer.host.image --part "%ZLC_EST_PART%"
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
rem Future-proof: also glob any Vivado version directory in the default install roots
rem (so a newer release than the list above is still auto-found); last match wins (newest).
for /d %%V in ("C:\Xilinx\Vivado\*" "D:\Xilinx\Vivado\*") do (
  if exist "%%~V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=%%~V\bin\vivado.bat"
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
