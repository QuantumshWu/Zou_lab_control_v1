@echo off
rem Common environment discovery for ZLC pulse-streamer batch files.
rem User overrides always win:
rem   set ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.2\bin\vivado.bat
rem   set ZLC_PY_CMD=py -3

if "%ZLC_PS_VIVADO_BIN%"=="" (
  for %%V in (2019.1 2019.2 2020.1 2020.2 2021.1 2021.2 2022.1 2022.2 2023.1 2023.2 2024.1 2024.2 2025.1 2025.2) do (
    if exist "C:\Xilinx\Vivado\%%V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\%%V\bin\vivado.bat"
    if exist "D:\Xilinx\Vivado\%%V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=D:\Xilinx\Vivado\%%V\bin\vivado.bat"
  )
)

if "%ZLC_PS_VIVADO_BIN%"=="" (
  for /f "delims=" %%I in ('where vivado.bat 2^>nul') do set "ZLC_PS_VIVADO_BIN=%%I"
)

if "%ZLC_PS_VIVADO_BIN%"=="" (
  where vivado >nul 2>nul
  if not errorlevel 1 set "ZLC_PS_VIVADO_BIN=vivado"
)

if "%ZLC_PS_VIVADO_BIN%"=="" (
  echo Could not find Vivado.
  echo Set ZLC_PS_VIVADO_BIN to the full path of vivado.bat, for example:
  echo   set ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.2\bin\vivado.bat
  exit /b 1
)

if "%ZLC_PY_CMD%"=="" (
  where python >nul 2>nul
  if not errorlevel 1 set "ZLC_PY_CMD=python"
)

if "%ZLC_PY_CMD%"=="" (
  where py >nul 2>nul
  if not errorlevel 1 set "ZLC_PY_CMD=py -3"
)

if "%ZLC_PY_CMD%"=="" (
  echo Could not find python or py.
  echo Run install_requirements.bat or set ZLC_PY_CMD to the Python command for this environment.
  exit /b 1
)

echo ZLC Vivado: %ZLC_PS_VIVADO_BIN%
echo ZLC Python: %ZLC_PY_CMD%
exit /b 0
