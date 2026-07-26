@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I not "%~1"=="--inner" (
  call "%~f0" --inner %*
  set "ZLC_STATUS=!ERRORLEVEL!"
  echo.
  if "!ZLC_STATUS!"=="0" (
    echo ZLC pulse server command completed successfully.
  ) else (
    echo ZLC pulse server command failed with code !ZLC_STATUS!.
  )
  if "%ZLC_NO_PAUSE%"=="" pause
  exit /b !ZLC_STATUS!
)
shift /1

set "FPGA_DIR=%~dp0"
for %%I in ("%FPGA_DIR%..") do set "REPO_ROOT=%%~fI"
set "ZLC_REPO_ROOT=%REPO_ROOT%"

if "%~1"=="--help" goto zlc_help
if "%~1"=="/?" goto zlc_help
set "ZLC_CHECK_ONLY=0"
if /I "%~1"=="--check-config" (
  set "ZLC_CHECK_ONLY=1"
) else if not "%~1"=="" (
  echo Unknown option: %~1
  goto zlc_help_error
)

call "%FPGA_DIR%_resolve_tools.bat" python "%REPO_ROOT%"
if errorlevel 1 exit /b 1

if not defined ZLC_PS_HOST set "ZLC_PS_HOST=0.0.0.0"
if not defined ZLC_PS_PORT set "ZLC_PS_PORT=18861"
if not defined ZLC_PS_SERVER_BACKEND set "ZLC_PS_SERVER_BACKEND=jtag-axi"
if not defined ZLC_PS_TARGET set "ZLC_PS_TARGET=%REPO_ROOT%\zlc_pulse\assets\deployed_target.json"
if not defined ZLC_PS_XDC set "ZLC_PS_XDC=%FPGA_DIR%board_config\board.xdc"
if not defined ZLC_PS_STATE_DIR set "ZLC_PS_STATE_DIR=%FPGA_DIR%build\state"
if not defined ZLC_PS_PROJECT_DIR set "ZLC_PS_PROJECT_DIR=%FPGA_DIR%build\ps"

if /I not "%ZLC_PS_SERVER_BACKEND%"=="jtag-axi" if /I not "%ZLC_PS_SERVER_BACKEND%"=="uart" (
  echo ERROR: ZLC_PS_SERVER_BACKEND must be jtag-axi or uart.
  exit /b 2
)
if not exist "%ZLC_PS_TARGET%" (
  echo ERROR: canonical PulseTarget file does not exist:
  echo   %ZLC_PS_TARGET%
  exit /b 2
)
if not exist "%ZLC_PS_XDC%" (
  echo ERROR: server-side pulse constraints file does not exist:
  echo   %ZLC_PS_XDC%
  exit /b 2
)
if /I "%ZLC_PS_SERVER_BACKEND%"=="uart" if not defined ZLC_PS_UART_PORT (
  echo ERROR: ZLC_PS_UART_PORT is required for the uart backend.
  exit /b 2
)
if /I "%ZLC_PS_SERVER_BACKEND%"=="jtag-axi" if "%ZLC_CHECK_ONLY%"=="0" (
  call "%FPGA_DIR%_resolve_tools.bat" vivado
  if errorlevel 1 exit /b 1
)

if not exist "%ZLC_PS_STATE_DIR%\" mkdir "%ZLC_PS_STATE_DIR%" >nul 2>nul
pushd "%REPO_ROOT%"
set "PYTHONPATH=%CD%;%PYTHONPATH%"

echo ZLC current-only pulse execution server
echo Host:    %ZLC_PS_HOST%:%ZLC_PS_PORT%
echo Backend: %ZLC_PS_SERVER_BACKEND%
echo Target:  %ZLC_PS_TARGET%
echo XDC:     %ZLC_PS_XDC%
echo Geometry/clock: fpga\board_config\streamer_config.json
echo Bitstream policy: frozen; this launcher never synthesizes or programs hardware

if "%ZLC_CHECK_ONLY%"=="1" (
  %ZLC_PY_CMD% -c "from fpga.pulse_streamer.host.image import build_fingerprint; from zlc_pulse import load_pulse_target,pulse_target_manifest_from_xdc; from zlc_pulse.server_app import load_deployed_streamer_config,validate_deployed_target; t=load_pulse_target(r'%ZLC_PS_TARGET%'); m=pulse_target_manifest_from_xdc(t,r'%ZLC_PS_XDC%'); p,c,s=load_deployed_streamer_config(); validate_deployed_target(t,p); print('Target ABI:',t.abi_fingerprint); print('Published ports:',len(m.ports)); print('Raw lanes:',len(t.raw_lanes)); print('Clock Hz:',c); print('Geometry:',s); print('Geometry fingerprint:',hex(build_fingerprint(p)))"
  set "ZLC_STATUS=!ERRORLEVEL!"
  popd
  endlocal & exit /b !ZLC_STATUS!
)

set "ZLC_UART_ARGS="
if /I "%ZLC_PS_SERVER_BACKEND%"=="uart" (
  set "ZLC_UART_ARGS=--uart-port %ZLC_PS_UART_PORT%"
  if defined ZLC_PS_UART_BAUD set "ZLC_UART_ARGS=!ZLC_UART_ARGS! --uart-baud %ZLC_PS_UART_BAUD%"
)

%ZLC_PY_CMD% -m zlc_pulse.server_app ^
  --target "%ZLC_PS_TARGET%" ^
  --xdc "%ZLC_PS_XDC%" ^
  --backend %ZLC_PS_SERVER_BACKEND% ^
  --state-dir "%ZLC_PS_STATE_DIR%" ^
  --host %ZLC_PS_HOST% ^
  --port %ZLC_PS_PORT% ^
  %ZLC_UART_ARGS%
set "ZLC_STATUS=%ERRORLEVEL%"
popd
endlocal & exit /b %ZLC_STATUS%

:zlc_help
echo Start the current-only ZLC pulse execution server against the approved frozen bitstream.
echo.
echo Usage:
echo   fpga\run_server.bat
echo   fpga\run_server.bat --check-config
echo.
echo Deployment environment:
echo   ZLC_PS_SERVER_BACKEND=jtag-axi ^| uart   ^(default jtag-axi; never auto-guessed^)
echo   ZLC_PS_TARGET=path\to\pulse_target.json
echo   ZLC_PS_XDC=path\to\board.xdc            ^(server-side authority^)
echo   ZLC_PS_UART_PORT=COM3                    ^(required for uart^)
echo   ZLC_PS_HOST=0.0.0.0
echo   ZLC_PS_PORT=18861
echo   ZLC_PS_STATE_DIR=path\to\state
echo.
echo This launcher never builds or programs a bitstream. Hardware changes use the separately
echo approved evidence-driven hardware-owner workflow.
exit /b 0

:zlc_help_error
call :zlc_help
exit /b 2
