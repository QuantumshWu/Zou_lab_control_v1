@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I not "%~1"=="--inner" (
  call "%~f0" --inner %*
  set "ZLC_STATUS=!ERRORLEVEL!"
  if not "!ZLC_STATUS!"=="0" (
    echo.
    echo ZLC build/program failed with code !ZLC_STATUS!.
    echo Keep this window open and read the messages above.
    pause
  )
  exit /b !ZLC_STATUS!
)
shift /1

set "FPGA_DIR=%~dp0"
for %%I in ("%FPGA_DIR%..") do set "REPO_ROOT=%%~fI"
set "STREAMER_DIR=%FPGA_DIR%pulse_streamer"

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
echo Required for a real build:
echo   fpga\pulse_streamer\zlc_pulse_streamer_40ch.xdc
echo or:
echo   set ZLC_PS_40CH_XDC=C:\path\to\completed_40ch_board.xdc
echo.
echo Optional:
echo   set ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.2\bin\vivado.bat
echo   set ZLC_PS_DISABLE_SUBST=1
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
set "DIRECT_TCL=%STREAMER_DIR%\%TCL_NAME%"
if not exist "%DIRECT_TCL%" (
  echo Missing Tcl script: %DIRECT_TCL%
  exit /b 2
)

if "%ZLC_PS_DISABLE_SUBST%"=="1" goto zlc_direct_tcl

set "ZLC_SHORT_DRIVE="
set "ZLC_CREATED_SUBST=0"
for /f "tokens=1,2,*" %%A in ('subst 2^>nul') do (
  if /I "%%C"=="%REPO_ROOT%" (
    set "ZLC_SHORT_DRIVE=%%A"
    set "ZLC_SHORT_DRIVE=!ZLC_SHORT_DRIVE:~0,2!"
  )
)
if not "%ZLC_PS_SHORT_DRIVE%"=="" (
  set "ZLC_SHORT_DRIVE=%ZLC_PS_SHORT_DRIVE%"
)
if "!ZLC_SHORT_DRIVE!"=="" (
  for %%D in (Z: Y: X: W: V: U: T: S: R: Q:) do (
    if "!ZLC_SHORT_DRIVE!"=="" if not exist "%%D\NUL" set "ZLC_SHORT_DRIVE=%%D"
  )
)
if "!ZLC_SHORT_DRIVE!"=="" goto zlc_direct_tcl

if not exist "!ZLC_SHORT_DRIVE!\NUL" (
  subst !ZLC_SHORT_DRIVE! "%REPO_ROOT%" >nul
  if errorlevel 1 goto zlc_direct_tcl
  set "ZLC_CREATED_SUBST=1"
)

set "SHORT_TCL=!ZLC_SHORT_DRIVE!\fpga\pulse_streamer\%TCL_NAME%"
if "%ZLC_PS_PROJECT_DIR%"=="" set "ZLC_PS_PROJECT_DIR=!ZLC_SHORT_DRIVE!\fpga\pulse_streamer\build\zlc_pulse_streamer_40ch"
echo ZLC short Vivado path: !SHORT_TCL!
echo ZLC short project dir: %ZLC_PS_PROJECT_DIR%
pushd "!ZLC_SHORT_DRIVE!\"
call "%ZLC_PS_VIVADO_BIN%" -mode batch -source "!SHORT_TCL!"
set "ZLC_STATUS=!ERRORLEVEL!"
popd
if "!ZLC_CREATED_SUBST!"=="1" subst !ZLC_SHORT_DRIVE! /D >nul 2>nul
exit /b !ZLC_STATUS!

:zlc_direct_tcl
if "%ZLC_PS_PROJECT_DIR%"=="" set "ZLC_PS_PROJECT_DIR=%TEMP%\zlc_ps_40ch"
echo ZLC direct Vivado path: %DIRECT_TCL%
echo ZLC short project dir: %ZLC_PS_PROJECT_DIR%
call "%ZLC_PS_VIVADO_BIN%" -mode batch -source "%DIRECT_TCL%"
exit /b %ERRORLEVEL%
