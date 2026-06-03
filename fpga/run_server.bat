@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I not "%~1"=="--inner" (
  call "%~f0" --inner %*
  set "ZLC_STATUS=!ERRORLEVEL!"
  if not "!ZLC_STATUS!"=="0" (
    echo.
    echo ZLC server failed with code !ZLC_STATUS!.
    echo Keep this window open and read the messages above.
    pause
  )
  exit /b !ZLC_STATUS!
)
shift /1

set "FPGA_DIR=%~dp0"
for %%I in ("%FPGA_DIR%..") do set "REPO_ROOT=%%~fI"
set "STREAMER_DIR=%FPGA_DIR%pulse_streamer"

if "%~1"=="--help" goto zlc_help
if "%~1"=="/?" goto zlc_help

call :zlc_find_python
if errorlevel 1 exit /b 1
call :zlc_find_vivado
if errorlevel 1 exit /b 1

pushd "%REPO_ROOT%"
set "PYTHONPATH=%CD%;%PYTHONPATH%"

if "%ZLC_PS_HOST%"=="" set "ZLC_PS_HOST=0.0.0.0"
if "%ZLC_PS_PORT%"=="" set "ZLC_PS_PORT=18861"
if "%ZLC_PS_SERVER_BACKEND%"=="" set "ZLC_PS_SERVER_BACKEND=vivado-session"
if "%ZLC_PS_VIVADO_PROGRAM_ON_RUN%"=="" set "ZLC_PS_VIVADO_PROGRAM_ON_RUN=0"
if "%ZLC_PS_MAX_EDGES%"=="" set "ZLC_PS_MAX_EDGES=128"
if "%ZLC_PS_TICK_WIDTH%"=="" set "ZLC_PS_TICK_WIDTH=32"
set "ZLC_PS_CHANNEL_COUNT=40"

if "%ZLC_PS_STATE_DIR%"=="" set "ZLC_PS_STATE_DIR=%CD%\fpga\pulse_streamer\build\zlc_sequencer_state_40ch"
if "%ZLC_PS_PROJECT_DIR%"=="" if exist "%TEMP%\zlc_ps_40ch\zlc_pulse_streamer_40ch.xpr" set "ZLC_PS_PROJECT_DIR=%TEMP%\zlc_ps_40ch"
if not "%ZLC_PS_PROJECT_DIR%"=="" (
  if "%ZLC_PS_VIVADO_PROJECT%"=="" if exist "%ZLC_PS_PROJECT_DIR%\zlc_pulse_streamer_40ch.xpr" set "ZLC_PS_VIVADO_PROJECT=%ZLC_PS_PROJECT_DIR%\zlc_pulse_streamer_40ch.xpr"
  if "%ZLC_PS_VIVADO_BIT%"=="" if exist "%ZLC_PS_PROJECT_DIR%\zlc_pulse_streamer_40ch.runs\impl_1\zlc_pulse_streamer_top_40ch.bit" set "ZLC_PS_VIVADO_BIT=%ZLC_PS_PROJECT_DIR%\zlc_pulse_streamer_40ch.runs\impl_1\zlc_pulse_streamer_top_40ch.bit"
  if "%ZLC_PS_VIVADO_LTX%"=="" if exist "%ZLC_PS_PROJECT_DIR%\zlc_pulse_streamer_40ch.runs\impl_1\zlc_pulse_streamer_top_40ch.ltx" set "ZLC_PS_VIVADO_LTX=%ZLC_PS_PROJECT_DIR%\zlc_pulse_streamer_40ch.runs\impl_1\zlc_pulse_streamer_top_40ch.ltx"
)
if "%ZLC_PS_VIVADO_PROJECT%"=="" if exist "%CD%\fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.xpr" set "ZLC_PS_VIVADO_PROJECT=%CD%\fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.xpr"
if "%ZLC_PS_VIVADO_BIT%"=="" if exist "%CD%\fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.runs\impl_1\zlc_pulse_streamer_top_40ch.bit" set "ZLC_PS_VIVADO_BIT=%CD%\fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.runs\impl_1\zlc_pulse_streamer_top_40ch.bit"
if "%ZLC_PS_VIVADO_LTX%"=="" if exist "%CD%\fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.runs\impl_1\zlc_pulse_streamer_top_40ch.ltx" set "ZLC_PS_VIVADO_LTX=%CD%\fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.runs\impl_1\zlc_pulse_streamer_top_40ch.ltx"
if "%ZLC_PS_VIVADO_LTX%"=="" (
  if exist "%CD%\fpga\pulse_streamer\build" for /r "%CD%\fpga\pulse_streamer\build" %%F in (*.ltx) do if "!ZLC_PS_VIVADO_LTX!"=="" set "ZLC_PS_VIVADO_LTX=%%~fF"
)
if "%ZLC_PS_VIVADO_LTX%"=="" (
  echo ERROR: no Vivado .ltx probe file was found.
  echo.
  echo The 40ch server controls the FPGA through Vivado VIO, so it must load
  echo the same .ltx Probes file used when the FPGA was programmed.
  echo.
  echo Fix one of these:
  echo   1. Run fpga\build_and_program.bat after completing the 40ch XDC.
  echo   2. Or set ZLC_PS_VIVADO_LTX to the .ltx from Vivado Program Device.
  echo.
  echo Example:
  echo   set ZLC_PS_VIVADO_LTX=D:\path\zlc_pulse_streamer_top_40ch.ltx
  echo   fpga\run_server.bat
  echo.
  popd
  exit /b 2
)

set "ZLC_40CH_CHANNELS=ch00 ch01 ch02 ch03 ch04 ch05 ch06 ch07 ch08 ch09 ch10 ch11 ch12 ch13 ch14 ch15 ch16 ch17 ch18 ch19 ch20 ch21 ch22 ch23 ch24 ch25 ch26 ch27 ch28 ch29 ch30 ch31 ch32 ch33 ch34 ch35 ch36 ch37 ch38 ch39"

echo ZLC FPGA pulse-streamer server: 40ch
echo Host:    %ZLC_PS_HOST%:%ZLC_PS_PORT%
echo Backend: %ZLC_PS_SERVER_BACKEND%
echo Project: %ZLC_PS_VIVADO_PROJECT%
echo Bit:     %ZLC_PS_VIVADO_BIT%
echo LTX:     %ZLC_PS_VIVADO_LTX%
echo Channels: %ZLC_40CH_CHANNELS%
echo Trigger:  ch03

%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.sequencer_server ^
  --backend %ZLC_PS_SERVER_BACKEND% ^
  --host %ZLC_PS_HOST% ^
  --port %ZLC_PS_PORT% ^
  --channels %ZLC_40CH_CHANNELS% ^
  --trigger-channels ch03 ^
  --clock-hz 100000000 ^
  --state-dir "%ZLC_PS_STATE_DIR%" ^
  --prepare-command "%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer prepare" ^
  --fire-command "%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer fire" ^
  --wait-done-command "%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer wait_done" ^
  --safe-state-command "%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.fpga_pulse_streamer safe_state"
set "ZLC_STATUS=%ERRORLEVEL%"
popd
endlocal & exit /b %ZLC_STATUS%

:zlc_help
echo Start the 40-channel ZLC FPGA pulse-streamer server.
echo.
echo Usage:
echo   fpga\run_server.bat
echo.
echo Defaults:
echo   host/port: 0.0.0.0:18861
echo   backend:   vivado-session
echo   channels:  ch00 ... ch39
echo   trigger:   ch03
echo.
echo Optional:
echo   set ZLC_PS_HOST=0.0.0.0
echo   set ZLC_PS_PORT=18861
echo   set ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.2\bin\vivado.bat
echo   set ZLC_PS_SERVER_BACKEND=vivado-session
exit /b 0

:zlc_find_python
if not "%ZLC_PY_CMD%"=="" goto zlc_python_found
where python >nul 2>nul
if not errorlevel 1 set "ZLC_PY_CMD=python"
if not "%ZLC_PY_CMD%"=="" goto zlc_python_found
where py >nul 2>nul
if not errorlevel 1 set "ZLC_PY_CMD=py -3"
if not "%ZLC_PY_CMD%"=="" goto zlc_python_found
echo Could not find python or py. Run install_requirements.bat first.
exit /b 1
:zlc_python_found
echo ZLC Python: %ZLC_PY_CMD%
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
