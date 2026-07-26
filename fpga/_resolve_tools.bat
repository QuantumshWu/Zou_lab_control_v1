@echo off

rem Single owner for the Python and Vivado executables used by FPGA launchers.
rem This file intentionally has no SETLOCAL: resolved variables must return to
rem the caller, whose launcher already owns the enclosing local environment.

if /I "%~1"=="python" goto zlc_find_python
if /I "%~1"=="vivado" goto zlc_find_vivado
echo ERROR: _resolve_tools.bat expects python or vivado.
exit /b 2

:zlc_find_python
if "%~2"=="" (
  echo ERROR: Python resolution requires the repository root.
  exit /b 2
)
for %%I in ("%~2") do set "ZLC_TOOL_REPO_ROOT=%%~fI"
if defined ZLC_PY_CMD goto zlc_python_found
if defined ZLC_FPGA_PYTHON (
  if exist "%ZLC_FPGA_PYTHON%" (
    set "ZLC_PY_CMD=call "%ZLC_FPGA_PYTHON%""
  ) else (
    set "ZLC_PY_CMD=%ZLC_FPGA_PYTHON%"
  )
  goto zlc_python_found
)
if exist "%ZLC_TOOL_REPO_ROOT%\.zlc_python_path" (
  set /p "ZLC_STORED_PY="<"%ZLC_TOOL_REPO_ROOT%\.zlc_python_path"
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
echo ERROR: Python was not found. Run install_requirements.bat or set ZLC_FPGA_PYTHON.
set "ZLC_TOOL_REPO_ROOT="
exit /b 1

:zlc_python_found
echo ZLC Python: %ZLC_PY_CMD%
set "ZLC_TOOL_REPO_ROOT="
exit /b 0

:zlc_find_vivado
if defined ZLC_PS_VIVADO_BIN goto zlc_vivado_found
for %%V in (2019.1 2019.2 2020.1 2020.2 2021.1 2021.2 2022.1 2022.2 2023.1 2023.2 2024.1 2024.2 2025.1 2025.2 2026.1 2026.2) do (
  if exist "C:\Xilinx\Vivado\%%V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\%%V\bin\vivado.bat"
  if exist "D:\Xilinx\Vivado\%%V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=D:\Xilinx\Vivado\%%V\bin\vivado.bat"
  if exist "C:\AMD\Vivado\%%V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=C:\AMD\Vivado\%%V\bin\vivado.bat"
  if exist "D:\AMD\Vivado\%%V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=D:\AMD\Vivado\%%V\bin\vivado.bat"
)
rem Preserve default-root discovery for releases newer than the explicit list.
for /d %%V in ("C:\Xilinx\Vivado\*" "D:\Xilinx\Vivado\*" "C:\AMD\Vivado\*" "D:\AMD\Vivado\*") do (
  if exist "%%~V\bin\vivado.bat" set "ZLC_PS_VIVADO_BIN=%%~V\bin\vivado.bat"
)
if defined ZLC_PS_VIVADO_BIN goto zlc_vivado_found
for /f "delims=" %%I in ('where vivado.bat 2^>nul') do if not defined ZLC_PS_VIVADO_BIN set "ZLC_PS_VIVADO_BIN=%%I"
if defined ZLC_PS_VIVADO_BIN goto zlc_vivado_found
where vivado >nul 2>nul
if not errorlevel 1 set "ZLC_PS_VIVADO_BIN=vivado"
if defined ZLC_PS_VIVADO_BIN goto zlc_vivado_found
echo ERROR: Vivado was not found. Set ZLC_PS_VIVADO_BIN.
exit /b 1

:zlc_vivado_found
echo ZLC Vivado: %ZLC_PS_VIVADO_BIN%
exit /b 0
