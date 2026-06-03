@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "TCL_NAME=%~1"
if "%TCL_NAME%"=="" (
  echo Usage: vivado_run_tcl.bat create_project_4ch.tcl
  exit /b 2
)

call "%SCRIPT_DIR%vivado_env.bat"
if errorlevel 1 exit /b 1

set "REPO_ROOT=%SCRIPT_DIR%..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"
set "DIRECT_TCL=%SCRIPT_DIR%%TCL_NAME%"

if "%ZLC_PS_DISABLE_SUBST%"=="1" goto zlc_direct

set "ZLC_SHORT_DRIVE="
if not "%ZLC_PS_SHORT_DRIVE%"=="" (
  if not exist "%ZLC_PS_SHORT_DRIVE%\NUL" set "ZLC_SHORT_DRIVE=%ZLC_PS_SHORT_DRIVE%"
)

if "!ZLC_SHORT_DRIVE!"=="" (
  for %%D in (Z: Y: X: W: V: U: T: S: R: Q:) do (
    if "!ZLC_SHORT_DRIVE!"=="" if not exist "%%D\NUL" set "ZLC_SHORT_DRIVE=%%D"
  )
)

if "!ZLC_SHORT_DRIVE!"=="" goto zlc_direct

subst !ZLC_SHORT_DRIVE! "%REPO_ROOT%" >nul
if errorlevel 1 goto zlc_direct

set "ZLC_TCL=!ZLC_SHORT_DRIVE!\fpga\pulse_streamer\%TCL_NAME%"
echo ZLC short Vivado path: !ZLC_TCL!
pushd "!ZLC_SHORT_DRIVE!\"
call "%ZLC_PS_VIVADO_BIN%" -mode batch -source "!ZLC_TCL!"
set "ZLC_STATUS=!ERRORLEVEL!"
popd
subst !ZLC_SHORT_DRIVE! /D >nul 2>nul
endlocal & exit /b %ZLC_STATUS%

:zlc_direct
echo ZLC direct Vivado path: %DIRECT_TCL%
call "%ZLC_PS_VIVADO_BIN%" -mode batch -source "%DIRECT_TCL%"
set "ZLC_STATUS=%ERRORLEVEL%"
endlocal & exit /b %ZLC_STATUS%
