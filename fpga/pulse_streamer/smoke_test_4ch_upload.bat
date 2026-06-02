@echo off
setlocal
set "REPO_ROOT=%~dp0..\.."
pushd "%REPO_ROOT%"
set "PYTHONPATH=%CD%"
if "%ZLC_PS_VIVADO_BIN%"=="" (
  set "ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.2\bin\vivado.bat"
)
if "%ZLC_PS_VIVADO_PROJECT%"=="" (
  set "ZLC_PS_VIVADO_PROJECT=%CD%\fpga\pulse_streamer\build\zlc_pulse_streamer_4ch\zlc_pulse_streamer_4ch.xpr"
)
if "%ZLC_PS_VIVADO_BIT%"=="" (
  set "ZLC_PS_VIVADO_BIT=%CD%\fpga\pulse_streamer\build\zlc_pulse_streamer_4ch\zlc_pulse_streamer_4ch.runs\impl_1\zlc_pulse_streamer_top_4ch.bit"
)
if "%ZLC_PS_VIVADO_LTX%"=="" (
  set "ZLC_PS_VIVADO_LTX=%CD%\fpga\pulse_streamer\build\zlc_pulse_streamer_4ch\zlc_pulse_streamer_4ch.runs\impl_1\zlc_pulse_streamer_top_4ch.ltx"
)
if "%ZLC_PS_VIVADO_PROGRAM_ON_RUN%"=="" (
  set "ZLC_PS_VIVADO_PROGRAM_ON_RUN=0"
)
if "%ZLC_PS_CHANNEL_COUNT%"=="" (
  set "ZLC_PS_CHANNEL_COUNT=4"
)
python "%CD%\fpga\pulse_streamer\smoke_test_4ch.py" --vivado "%ZLC_PS_VIVADO_BIN%"
set "ZLC_STATUS=%ERRORLEVEL%"
popd
endlocal & exit /b %ZLC_STATUS%
