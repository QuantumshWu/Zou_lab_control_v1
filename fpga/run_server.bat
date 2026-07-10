@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I not "%~1"=="--inner" (
  set "ZLC_ACTION=pulse streamer server"
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
call :zlc_verify_sources
if errorlevel 1 exit /b 1

pushd "%REPO_ROOT%"
set "PYTHONPATH=%CD%;%PYTHONPATH%"

if "%ZLC_PS_HOST%"=="" set "ZLC_PS_HOST=0.0.0.0"
if "%ZLC_PS_PORT%"=="" set "ZLC_PS_PORT=18861"
if "%ZLC_PS_SERVER_BACKEND%"=="" set "ZLC_PS_SERVER_BACKEND=auto"
if "%ZLC_PS_VIVADO_PROGRAM_ON_RUN%"=="" set "ZLC_PS_VIVADO_PROGRAM_ON_RUN=0"
set "ZLC_PS_CLOCK_HZ_ARG="
if not "%ZLC_PS_CLOCK_HZ%"=="" set "ZLC_PS_CLOCK_HZ_ARG=--clock-hz %ZLC_PS_CLOCK_HZ%"
if "%ZLC_PS_MAX_CHANNEL_COUNT%"=="" (
  set "ZLC_PS_MAX_CHANNEL_COUNT_ARG="
) else (
  set "ZLC_PS_MAX_CHANNEL_COUNT_ARG=--max-channel-count %ZLC_PS_MAX_CHANNEL_COUNT%"
)
if "%ZLC_PS_XDC%"=="" if exist "%CD%\fpga\board_config\board.xdc" set "ZLC_PS_XDC=%CD%\fpga\board_config\board.xdc"
if "%ZLC_PS_CHANNEL_COUNT%"=="" (
  for /f "delims=" %%I in ('%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer infer_channel_count --xdc "%ZLC_PS_XDC%" %ZLC_PS_MAX_CHANNEL_COUNT_ARG% 2^>nul') do if "!ZLC_PS_CHANNEL_COUNT!"=="" set "ZLC_PS_CHANNEL_COUNT=%%I"
)
if "%ZLC_PS_CHANNEL_COUNT%"=="" (
  echo ERROR: channel count could not be derived from board XDC/config.
  popd
  exit /b 2
)
if "%ZLC_PS_CHANNELS%"=="" (
  for /f "delims=" %%I in ('%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer infer_channels --xdc "%ZLC_PS_XDC%" --default-count %ZLC_PS_CHANNEL_COUNT% %ZLC_PS_MAX_CHANNEL_COUNT_ARG% 2^>nul') do if "!ZLC_PS_CHANNELS!"=="" set "ZLC_PS_CHANNELS=%%I"
)
if "%ZLC_PS_CHANNELS%"=="" (
  echo ERROR: channel names could not be derived from board XDC/config.
  popd
  exit /b 2
)
rem The sequencer is a PURE streamer: it streams pulses on its named channels and NEVER
rem needs to know which line gates a camera.  Which channel is a camera capture trigger is
rem the CAMERA device's property (its config's capture_trigger_channels), read by the
rem control-computer measurement layer -- not the server.  So the server infers no camera
rem trigger here and its launch line below passes only channels, never a trigger flag.

rem Default the bitstream + JTAG-to-AXI probes from the in-repo build (build\ps).
if not "%ZLC_PS_PROJECT_DIR%"=="" (
  if "%ZLC_PS_VIVADO_BIT%"=="" set "ZLC_PS_VIVADO_BIT=%ZLC_PS_PROJECT_DIR%\ps.runs\impl_1\zlc_pulse_streamer_top.bit"
  if "%ZLC_PS_VIVADO_LTX%"=="" set "ZLC_PS_VIVADO_LTX=%ZLC_PS_PROJECT_DIR%\ps.runs\impl_1\zlc_pulse_streamer_top.ltx"
)
rem The .ltx probes file is only needed for the Vivado JTAG-to-AXI path.  auto/uart don't require it
rem (auto only brings Vivado up if NO UART link answers, and prints its own error in that case).
if /I not "%ZLC_PS_SERVER_BACKEND%"=="jtag-axi" goto zlc_ltx_ok
if "%ZLC_PS_VIVADO_LTX%"=="" (
  echo ERROR: no Vivado .ltx probes file was found.
  echo.
  echo The server drives the FPGA over JTAG-to-AXI ^(hw_axi^); it loads the
  echo .ltx so Vivado can find the jtag_axi core in the programmed bitstream.
  echo.
  echo Fix: build + program the bitstream first:
  echo   fpga\build_and_program.bat
  echo Or set ZLC_PS_VIVADO_LTX to the .ltx from the build.
  popd
  exit /b 2
)
if not exist "%ZLC_PS_VIVADO_LTX%" (
  echo ERROR: Vivado .ltx probes file does not exist:
  echo   %ZLC_PS_VIVADO_LTX%
  echo.
  echo Build + program the bitstream first:
  echo   fpga\build_and_program.bat
  popd
  exit /b 2
)

:zlc_ltx_ok
echo ZLC FPGA pulse streamer server: %ZLC_PS_CHANNEL_COUNT%ch ^(control link: %ZLC_PS_SERVER_BACKEND%^)
echo Host:    %ZLC_PS_HOST%:%ZLC_PS_PORT%
echo Backend: %ZLC_PS_SERVER_BACKEND%
echo Bit:     %ZLC_PS_VIVADO_BIT%
echo LTX:     %ZLC_PS_VIVADO_LTX%
echo Channels: %ZLC_PS_CHANNELS%
if "%ZLC_PS_CLOCK_HZ%"=="" (
  echo Clock:   fpga\board_config\streamer_config.json
) else (
  echo Clock:   %ZLC_PS_CLOCK_HZ% Hz ^(explicit override^)
)
echo Program-on-start: %ZLC_PS_VIVADO_PROGRAM_ON_RUN% ^(0 = assume build_and_program already loaded the FPGA^)

if "%ZLC_RUN_SERVER_CHECK%"=="1" (
  echo ZLC server config check complete.
  popd
  endlocal & exit /b 0
)

set "ZLC_PS_UART_ARGS="
if not "%ZLC_PS_UART_PORT%"=="" set "ZLC_PS_UART_ARGS=--uart-port %ZLC_PS_UART_PORT%"
if not "%ZLC_PS_UART_BAUD%"=="" set "ZLC_PS_UART_ARGS=%ZLC_PS_UART_ARGS% --uart-baud %ZLC_PS_UART_BAUD%"

%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.sequencer_server ^
  --backend %ZLC_PS_SERVER_BACKEND% ^
  %ZLC_PS_UART_ARGS% ^
  --host %ZLC_PS_HOST% ^
  --port %ZLC_PS_PORT% ^
  --channels %ZLC_PS_CHANNELS% ^
  --xdc "%ZLC_PS_XDC%" ^
  %ZLC_PS_CLOCK_HZ_ARG% ^
  --state-dir "%ZLC_PS_STATE_DIR%"
set "ZLC_STATUS=%ERRORLEVEL%"
popd
endlocal & exit /b %ZLC_STATUS%

:zlc_help
echo Start the FINAL ZLC FPGA pulse-streamer server ^(auto: UART fast-control side-channel, else JTAG-to-AXI / hw_axi^).
echo Engine: 1-tick FIFO prefetch + unbounded 2-bank streaming scan.
echo.
echo Usage:
echo   fpga\run_server.bat
echo   fpga\run_server.bat --check-config
echo.
echo Defaults:
echo   host/port: 0.0.0.0:18861
echo   backend:   auto  ^(probe UART fast-control side-channel first, else JTAG-to-AXI / hw_axi; ZLC_PS_SERVER_BACKEND to force^)
echo   channels:  inferred from ZLC_PS_XDC + board config ^(no fabricated fallback^)
echo   clock:     fpga\board_config\streamer_config.json ^(override with ZLC_PS_CLOCK_HZ^)
echo   bit/ltx:   fpga\build\ps\ps.runs\impl_1\zlc_pulse_streamer_top.{bit,ltx}
echo.
echo Run fpga\build_and_program.bat first ^(it builds AND programs the FPGA^).
echo.
echo Optional:
echo   set ZLC_FPGA_SERVER_PYTHON=C:\path\to\python.exe
echo   set ZLC_PS_HOST=0.0.0.0
echo   set ZLC_PS_PORT=18861
echo   set ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.1\bin\vivado.bat
echo   set ZLC_PS_VIVADO_PROGRAM_ON_RUN=1   ^(re-program the FPGA when the server starts^)
echo   set ZLC_PS_HW_SERVER_URL=localhost:3121
echo   set ZLC_PS_PROJECT_DIR=%%CD%%\fpga\build\ps
exit /b 0

:zlc_verify_sources
set "ZLC_DEFAULT_XDC=%REPO_ROOT%\fpga\board_config\board.xdc"
set "ZLC_STREAMER_CONFIG=%REPO_ROOT%\fpga\board_config\streamer_config.json"
if not defined ZLC_PS_XDC set "ZLC_PS_XDC=%ZLC_DEFAULT_XDC%"
if not exist "%ZLC_PS_XDC%" (
  echo ERROR: missing board XDC: %ZLC_PS_XDC%
  exit /b 2
)
if not exist "%ZLC_STREAMER_CONFIG%" (
  echo ERROR: missing streamer geometry config: %ZLC_STREAMER_CONFIG%
  exit /b 2
)
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
findstr /C:"module zlc_pulse_streamer_top" "%STREAMER_DIR%\zlc_pulse_streamer_top.v" >nul || (
  echo ERROR: FINAL top module name is wrong.
  exit /b 2
)
findstr /C:"module zlc_edge_streamer" "%STREAMER_DIR%\zlc_edge_streamer.v" >nul || (
  echo ERROR: FINAL engine module name is wrong.
  exit /b 2
)
echo ZLC FINAL source contract: topology=XDC geometry=streamer_config.json control=JTAG-to-AXI
exit /b 0

:zlc_default_paths
if defined ZLC_PS_BUILD_ROOT if "!ZLC_PS_BUILD_ROOT: =!"=="" set "ZLC_PS_BUILD_ROOT="
if defined ZLC_PS_PROJECT_DIR if "!ZLC_PS_PROJECT_DIR: =!"=="" set "ZLC_PS_PROJECT_DIR="
if defined ZLC_PS_STATE_DIR if "!ZLC_PS_STATE_DIR: =!"=="" set "ZLC_PS_STATE_DIR="
if not defined ZLC_PS_BUILD_ROOT set "ZLC_PS_BUILD_ROOT=%FPGA_DIR%build"
if not exist "!ZLC_PS_BUILD_ROOT!\" mkdir "!ZLC_PS_BUILD_ROOT!" >nul 2>nul
rem In-repo build (fpga\build\ps); the SHORT "ps" subdir matches build_and_program
rem so the server finds the bitstream under ps.runs\impl_1.
if not defined ZLC_PS_PROJECT_DIR set "ZLC_PS_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\ps"
if not defined ZLC_PS_STATE_DIR set "ZLC_PS_STATE_DIR=%ZLC_PS_BUILD_ROOT%\state"
echo ZLC build root: %ZLC_PS_BUILD_ROOT%
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
echo Could not find Vivado. Set ZLC_PS_VIVADO_BIN to vivado.bat.
exit /b 1
:zlc_vivado_found
echo ZLC Vivado: %ZLC_PS_VIVADO_BIN%
exit /b 0
