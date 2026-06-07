@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I not "%~1"=="--inner" (
  set "ZLC_ACTION=edge-table loader server"
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
call :zlc_verify_loader_sources
if errorlevel 1 exit /b 1

pushd "%REPO_ROOT%"
set "PYTHONPATH=%CD%;%PYTHONPATH%"

if "%ZLC_PS_HOST%"=="" set "ZLC_PS_HOST=0.0.0.0"
if "%ZLC_PS_PORT%"=="" set "ZLC_PS_PORT=18861"
if "%ZLC_PS_SERVER_BACKEND%"=="" set "ZLC_PS_SERVER_BACKEND=jtag-axi"
if "%ZLC_PS_VIVADO_PROGRAM_ON_RUN%"=="" set "ZLC_PS_VIVADO_PROGRAM_ON_RUN=0"
if "%ZLC_PS_CLOCK_HZ%"=="" set "ZLC_PS_CLOCK_HZ=50000000"
if "%ZLC_PS_MAX_CHANNEL_COUNT%"=="" (
  set "ZLC_PS_MAX_CHANNEL_COUNT_ARG="
) else (
  set "ZLC_PS_MAX_CHANNEL_COUNT_ARG=--max-channel-count %ZLC_PS_MAX_CHANNEL_COUNT%"
)
if "%ZLC_PS_XDC%"=="" if exist "%CD%\references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc" set "ZLC_PS_XDC=%CD%\references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc"
if "%ZLC_PS_CHANNEL_COUNT%"=="" (
  for /f "delims=" %%I in ('%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer infer_channel_count --xdc "%ZLC_PS_XDC%" --default-count 62 %ZLC_PS_MAX_CHANNEL_COUNT_ARG% 2^>nul') do if "!ZLC_PS_CHANNEL_COUNT!"=="" set "ZLC_PS_CHANNEL_COUNT=%%I"
)
if "%ZLC_PS_CHANNEL_COUNT%"=="" set "ZLC_PS_CHANNEL_COUNT=62"
if "%ZLC_PS_CHANNELS%"=="" (
  for /f "delims=" %%I in ('%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer infer_channels --xdc "%ZLC_PS_XDC%" --default-count %ZLC_PS_CHANNEL_COUNT% %ZLC_PS_MAX_CHANNEL_COUNT_ARG% 2^>nul') do if "!ZLC_PS_CHANNELS!"=="" set "ZLC_PS_CHANNELS=%%I"
)
if "%ZLC_PS_CHANNELS%"=="" (
  for /f "delims=" %%I in ('%ZLC_PY_CMD% -c "print(' '.join(f'ch{i:02d}' for i in range(62)))" 2^>nul') do if "!ZLC_PS_CHANNELS!"=="" set "ZLC_PS_CHANNELS=%%I"
)
if "%ZLC_PS_TRIGGER_CHANNELS%"=="" (
  for /f "delims=" %%I in ('%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer infer_trigger_channels --xdc "%ZLC_PS_XDC%" --default-count %ZLC_PS_CHANNEL_COUNT% %ZLC_PS_MAX_CHANNEL_COUNT_ARG% 2^>nul') do if "!ZLC_PS_TRIGGER_CHANNELS!"=="" set "ZLC_PS_TRIGGER_CHANNELS=%%I"
)
if "%ZLC_PS_TRIGGER_CHANNELS%"=="" (
  echo ERROR: could not infer the emCCD camera trigger channel from the selected XDC.
  echo Check ZLC_PS_XDC, or set ZLC_PS_TRIGGER_CHANNELS explicitly.
  popd
  exit /b 1
)

rem Default the edge-table loader bitstream + JTAG-to-AXI probes from the in-repo build (build\r).
if /I "%ZLC_PS_VARIANT%"=="d" (
  if "%ZLC_PS_VIVADO_BIT%"=="" set "ZLC_PS_VIVADO_BIT=%ZLC_PS_PROJECT_DIR%\d.runs\impl_1\zlc_pulse_streamer_d_top.bit"
  if "%ZLC_PS_VIVADO_LTX%"=="" set "ZLC_PS_VIVADO_LTX=%ZLC_PS_PROJECT_DIR%\d.runs\impl_1\zlc_pulse_streamer_d_top.ltx"
) else (
if not "%ZLC_PS_PROJECT_DIR%"=="" (
  if "%ZLC_PS_VIVADO_BIT%"=="" set "ZLC_PS_VIVADO_BIT=%ZLC_PS_PROJECT_DIR%\l.runs\impl_1\zlc_pulse_streamer_loader_top.bit"
  if "%ZLC_PS_VIVADO_LTX%"=="" set "ZLC_PS_VIVADO_LTX=%ZLC_PS_PROJECT_DIR%\l.runs\impl_1\zlc_pulse_streamer_loader_top.ltx"
)
)
if "%ZLC_PS_VIVADO_LTX%"=="" (
  echo ERROR: no Vivado .ltx probes file was found.
  echo.
  echo The edge-table loader server drives the FPGA over JTAG-to-AXI ^(hw_axi^); it loads the
  echo .ltx so Vivado can find the jtag_axi core in the programmed bitstream.
  echo.
  echo Fix: build + program the edge-table loader bitstream first:
  echo   fpga\build_and_program.bat
  echo Or set ZLC_PS_VIVADO_LTX to the .ltx from the edge-table loader build.
  popd
  exit /b 2
)
if not exist "%ZLC_PS_VIVADO_LTX%" (
  echo ERROR: Vivado .ltx probes file does not exist:
  echo   %ZLC_PS_VIVADO_LTX%
  echo.
  echo Build + program the edge-table loader bitstream first:
  echo   fpga\build_and_program.bat
  popd
  exit /b 2
)

echo ZLC FPGA edge-table loader server: %ZLC_PS_CHANNEL_COUNT%ch ^(JTAG-to-AXI^)
echo Host:    %ZLC_PS_HOST%:%ZLC_PS_PORT%
echo Backend: %ZLC_PS_SERVER_BACKEND%
echo Bit:     %ZLC_PS_VIVADO_BIT%
echo LTX:     %ZLC_PS_VIVADO_LTX%
echo Channels: %ZLC_PS_CHANNELS%
echo Clock:   %ZLC_PS_CLOCK_HZ% Hz
echo Program-on-start: %ZLC_PS_VIVADO_PROGRAM_ON_RUN% ^(0 = assume build_and_program already loaded the FPGA^)

if "%ZLC_RUN_SERVER_CHECK%"=="1" (
  echo ZLC server config check complete.
  popd
  endlocal & exit /b 0
)

%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.sequencer_server ^
  --backend %ZLC_PS_SERVER_BACKEND% ^
  --host %ZLC_PS_HOST% ^
  --port %ZLC_PS_PORT% ^
  --channels %ZLC_PS_CHANNELS% ^
  --trigger-channels %ZLC_PS_TRIGGER_CHANNELS% ^
  --clock-hz %ZLC_PS_CLOCK_HZ% ^
  --state-dir "%ZLC_PS_STATE_DIR%"
set "ZLC_STATUS=%ERRORLEVEL%"
popd
endlocal & exit /b %ZLC_STATUS%

:zlc_help
echo Start the edge-table loader ZLC FPGA pulse-streamer server ^(JTAG-to-AXI / hw_axi^).
echo.
echo Usage:
echo   fpga\run_server.bat
echo   fpga\run_server.bat --check-config
echo.
echo Defaults:
echo   host/port: 0.0.0.0:18861
echo   backend:   jtag-axi  ^(persistent Vivado hw_axi session driving the edge-table loader engine^)
echo   channels:  inferred from ZLC_PS_XDC, fallback ch00 ... ch61
echo   clock:     50000000 Hz ^(override with ZLC_PS_CLOCK_HZ^)
echo   bit/ltx:   fpga\build\l\l.runs\impl_1\zlc_pulse_streamer_loader_top.{bit,ltx}
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
echo   set ZLC_PS_PROJECT_DIR=%%CD%%\fpga\build\l
exit /b 0

:zlc_verify_loader_sources
set "ZLC_DEFAULT_XDC=%REPO_ROOT%\references\source_archives\address_switch\address_switch.srcs\constrs_1\new\addre.xdc"
if not defined ZLC_PS_XDC set "ZLC_PS_XDC=%ZLC_DEFAULT_XDC%"
if not exist "%STREAMER_DIR%\zlc_axi_program_loader.v" (
  echo ERROR: missing edge-table loader playback engine HDL: %STREAMER_DIR%\zlc_axi_program_loader.v
  exit /b 2
)
if not exist "%STREAMER_DIR%\zlc_pulse_streamer_loader_top.v" (
  echo ERROR: missing edge-table loader top HDL: %STREAMER_DIR%\zlc_pulse_streamer_loader_top.v
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
echo ZLC edge-table loader source contract: channels=62 num_slots=4 control=JTAG-to-AXI
exit /b 0

:zlc_default_paths
if defined ZLC_PS_BUILD_ROOT if "!ZLC_PS_BUILD_ROOT: =!"=="" set "ZLC_PS_BUILD_ROOT="
if defined ZLC_PS_PROJECT_DIR if "!ZLC_PS_PROJECT_DIR: =!"=="" set "ZLC_PS_PROJECT_DIR="
if defined ZLC_PS_STATE_DIR if "!ZLC_PS_STATE_DIR: =!"=="" set "ZLC_PS_STATE_DIR="
if not defined ZLC_PS_BUILD_ROOT set "ZLC_PS_BUILD_ROOT=%FPGA_DIR%build"
if not exist "!ZLC_PS_BUILD_ROOT!\" mkdir "!ZLC_PS_BUILD_ROOT!" >nul 2>nul
rem Short project dir name "r" matches create_project_loader.tcl (path-length fix).
if /I "%ZLC_PS_VARIANT%"=="d" (if not defined ZLC_PS_PROJECT_DIR set "ZLC_PS_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\d") else (if not defined ZLC_PS_PROJECT_DIR set "ZLC_PS_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\l")
if not defined ZLC_PS_STATE_DIR set "ZLC_PS_STATE_DIR=%ZLC_PS_BUILD_ROOT%\state_loader"
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
