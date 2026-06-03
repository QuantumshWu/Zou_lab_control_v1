@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
if "%~1"=="--help" (
  echo Run the 4-channel zlc_pulse_streamer xsim core simulation.
  echo This does not require an FPGA board.
  exit /b 0
)
if "%~1"=="/?" (
  echo Run the 4-channel zlc_pulse_streamer xsim core simulation.
  echo This does not require an FPGA board.
  exit /b 0
)

call "%SCRIPT_DIR%vivado_env.bat"
if errorlevel 1 exit /b 1

for %%I in ("%ZLC_PS_VIVADO_BIN%") do set "ZLC_VIVADO_BIN_DIR=%%~dpI"
if exist "%ZLC_VIVADO_BIN_DIR%xvlog.bat" (
  set "ZLC_XVLOG=%ZLC_VIVADO_BIN_DIR%xvlog.bat"
  set "ZLC_XELAB=%ZLC_VIVADO_BIN_DIR%xelab.bat"
  set "ZLC_XSIM=%ZLC_VIVADO_BIN_DIR%xsim.bat"
) else (
  set "ZLC_XVLOG=xvlog"
  set "ZLC_XELAB=xelab"
  set "ZLC_XSIM=xsim"
)

set "SIM_DIR=%SCRIPT_DIR%build\sim_4ch"
if not exist "%SIM_DIR%" mkdir "%SIM_DIR%"
pushd "%SIM_DIR%"

call "%ZLC_XVLOG%" "..\..\zlc_pulse_streamer.v" "..\..\tb_zlc_pulse_streamer_4ch.v"
if errorlevel 1 goto zlc_fail
call "%ZLC_XELAB%" tb_zlc_pulse_streamer_4ch -s tb_zlc_pulse_streamer_4ch_sim
if errorlevel 1 goto zlc_fail
call "%ZLC_XSIM%" tb_zlc_pulse_streamer_4ch_sim -runall
if errorlevel 1 goto zlc_fail

popd
endlocal & exit /b 0

:zlc_fail
set "ZLC_STATUS=%ERRORLEVEL%"
popd
endlocal & exit /b %ZLC_STATUS%
