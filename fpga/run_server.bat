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
if not defined ZLC_PS_SERVER_BACKEND set "ZLC_PS_SERVER_BACKEND=auto"
if not defined ZLC_PS_TARGET set "ZLC_PS_TARGET=%REPO_ROOT%\zlc_pulse\assets\deployed_target.json"
if not defined ZLC_PS_XDC set "ZLC_PS_XDC=%FPGA_DIR%board_config\board.xdc"
if not defined ZLC_PS_STATE_DIR set "ZLC_PS_STATE_DIR=%FPGA_DIR%build\state"
if not defined ZLC_PS_PROJECT_DIR set "ZLC_PS_PROJECT_DIR=%FPGA_DIR%build\ps"

if /I not "%ZLC_PS_SERVER_BACKEND%"=="auto" if /I not "%ZLC_PS_SERVER_BACKEND%"=="jtag-axi" if /I not "%ZLC_PS_SERVER_BACKEND%"=="uart" (
  echo ERROR: ZLC_PS_SERVER_BACKEND must be auto, uart or jtag-axi.
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
rem The UART bridge is the ordinary control path, so pyserial must exist for the
rem interpreter this launcher actually resolved -- which is not always the one
rem install_requirements.bat used.  Without it every port is silently
rem disqualified and the run looks like "the cable does nothing".
if /I not "%ZLC_PS_SERVER_BACKEND%"=="jtag-axi" if "%ZLC_CHECK_ONLY%"=="0" (
  %ZLC_PY_CMD% -c "import serial" >nul 2>nul
  if errorlevel 1 (
    echo pyserial is missing for this interpreter; installing it now...
    %ZLC_PY_CMD% -m pip install pyserial
    if errorlevel 1 (
      echo.
      echo WARNING: pyserial install failed. The UART probe cannot run, so this
      echo start can only reach the board over JTAG. Install it manually with:
      echo   install_requirements.bat
      echo.
    )
  )
)
rem jtag-axi demands Vivado up front. auto only needs it if the UART probe
rem finds nothing, so a missing Vivado is a note there, not a startup error.
if /I "%ZLC_PS_SERVER_BACKEND%"=="jtag-axi" if "%ZLC_CHECK_ONLY%"=="0" (
  call "%FPGA_DIR%_resolve_tools.bat" vivado
  if errorlevel 1 exit /b 1
)
if /I "%ZLC_PS_SERVER_BACKEND%"=="auto" if "%ZLC_CHECK_ONLY%"=="0" (
  call "%FPGA_DIR%_resolve_tools.bat" vivado
  if errorlevel 1 echo Continuing without Vivado: this start must resolve to the UART bridge.
)

if not exist "%ZLC_PS_STATE_DIR%\" mkdir "%ZLC_PS_STATE_DIR%" >nul 2>nul
pushd "%REPO_ROOT%"
set "PYTHONPATH=%CD%;%PYTHONPATH%"

echo ZLC current-only pulse execution server
echo Host:    %ZLC_PS_HOST%:%ZLC_PS_PORT%
echo Backend: %ZLC_PS_SERVER_BACKEND%
if /I "%ZLC_PS_SERVER_BACKEND%"=="auto" echo          ^(probes the UART bridge first; JTAG-to-AXI only if no port answers^)
echo Target:  %ZLC_PS_TARGET%
echo XDC:     %ZLC_PS_XDC%
echo Geometry/clock: fpga\board_config\streamer_config.json
echo Bitstream policy: frozen; this launcher never synthesizes or programs hardware

set "ZLC_UART_ARGS="
if defined ZLC_PS_UART_PORT set "ZLC_UART_ARGS=--uart-port %ZLC_PS_UART_PORT%"
if defined ZLC_PS_UART_BAUD set "ZLC_UART_ARGS=!ZLC_UART_ARGS! --uart-baud %ZLC_PS_UART_BAUD%"

set "ZLC_CHECK_ARGS="
if "%ZLC_CHECK_ONLY%"=="1" set "ZLC_CHECK_ARGS=--check-config"

%ZLC_PY_CMD% -m zlc_pulse.server_app ^
  --target "%ZLC_PS_TARGET%" ^
  --xdc "%ZLC_PS_XDC%" ^
  --backend %ZLC_PS_SERVER_BACKEND% ^
  --state-dir "%ZLC_PS_STATE_DIR%" ^
  --host %ZLC_PS_HOST% ^
  --port %ZLC_PS_PORT% ^
  !ZLC_UART_ARGS! !ZLC_CHECK_ARGS!
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
echo Transport policy:
echo   auto     probe every UART port for the deployed geometry, then fall back to JTAG-to-AXI
echo   uart     demand the UART bridge; fail loudly instead of falling back
echo   jtag-axi demand JTAG-to-AXI; never open a serial port
echo.
echo Deployment environment:
echo   ZLC_PS_SERVER_BACKEND=auto ^| uart ^| jtag-axi   ^(default auto^)
echo   ZLC_PS_TARGET=path\to\pulse_target.json
echo   ZLC_PS_XDC=path\to\board.xdc            ^(server-side authority^)
echo   ZLC_PS_UART_PORT=COM3                    ^(optional: probe only this port^)
echo   ZLC_PS_UART_BAUD=3000000
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
