@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
if "%ZLC_PS_VIVADO_BIN%"=="" (
  set "ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.2\bin\vivado.bat"
)
if not exist "%ZLC_PS_VIVADO_BIN%" (
  set "ZLC_PS_VIVADO_BIN=vivado"
)
"%ZLC_PS_VIVADO_BIN%" -mode batch -source "%SCRIPT_DIR%create_project_40ch.tcl"
set "ZLC_STATUS=%ERRORLEVEL%"
endlocal & exit /b %ZLC_STATUS%
