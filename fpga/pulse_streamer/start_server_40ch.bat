@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
if "%~1"=="--help" (
  echo Start the 40-channel ZLC sequencer server after the 40-channel bitstream is programmed.
  echo Override host/port/state with ZLC_PS_HOST, ZLC_PS_PORT, and ZLC_PS_STATE_DIR.
  exit /b 0
)
if "%~1"=="/?" (
  echo Start the 40-channel ZLC sequencer server after the 40-channel bitstream is programmed.
  echo Override host/port/state with ZLC_PS_HOST, ZLC_PS_PORT, and ZLC_PS_STATE_DIR.
  exit /b 0
)
set "REPO_ROOT=%SCRIPT_DIR%..\.."
for %%I in ("%REPO_ROOT%") do set "REPO_ROOT=%%~fI"

call "%SCRIPT_DIR%vivado_env.bat"
if errorlevel 1 exit /b 1

pushd "%REPO_ROOT%"
set "PYTHONPATH=%CD%;%PYTHONPATH%"

if "%ZLC_PS_VIVADO_PROJECT%"=="" set "ZLC_PS_VIVADO_PROJECT=%CD%\fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.xpr"
if "%ZLC_PS_VIVADO_BIT%"=="" set "ZLC_PS_VIVADO_BIT=%CD%\fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.runs\impl_1\zlc_pulse_streamer_top_40ch.bit"
if "%ZLC_PS_VIVADO_LTX%"=="" set "ZLC_PS_VIVADO_LTX=%CD%\fpga\pulse_streamer\build\zlc_pulse_streamer_40ch\zlc_pulse_streamer_40ch.runs\impl_1\zlc_pulse_streamer_top_40ch.ltx"
if "%ZLC_PS_VIVADO_PROGRAM_ON_RUN%"=="" set "ZLC_PS_VIVADO_PROGRAM_ON_RUN=0"
if "%ZLC_PS_VIO_FILTER%"=="" set "ZLC_PS_VIO_FILTER=CELL_NAME=~""*vio*"""
if "%ZLC_PS_MAX_EDGES%"=="" set "ZLC_PS_MAX_EDGES=1024"
if "%ZLC_PS_TICK_WIDTH%"=="" set "ZLC_PS_TICK_WIDTH=32"
if "%ZLC_PS_CHANNEL_COUNT%"=="" set "ZLC_PS_CHANNEL_COUNT=40"
if "%ZLC_PS_STATE_DIR%"=="" set "ZLC_PS_STATE_DIR=%CD%\fpga\pulse_streamer\build\zlc_sequencer_state_40ch"
if "%ZLC_PS_HOST%"=="" set "ZLC_PS_HOST=0.0.0.0"
if "%ZLC_PS_PORT%"=="" set "ZLC_PS_PORT=18861"

set "ZLC_40CH_CHANNELS=ch00 ch01 ch02 ch03 ch04 ch05 ch06 ch07 ch08 ch09 ch10 ch11 ch12 ch13 ch14 ch15 ch16 ch17 ch18 ch19 ch20 ch21 ch22 ch23 ch24 ch25 ch26 ch27 ch28 ch29 ch30 ch31 ch32 ch33 ch34 ch35 ch36 ch37 ch38 ch39"

echo Starting 40ch ZLC sequencer server on %ZLC_PS_HOST%:%ZLC_PS_PORT%
echo Project: %ZLC_PS_VIVADO_PROJECT%
echo Bit:     %ZLC_PS_VIVADO_BIT%
echo LTX:     %ZLC_PS_VIVADO_LTX%

%ZLC_PY_CMD% -m Zou_lab_control.neutral_atom.devices.sequencer_server ^
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
