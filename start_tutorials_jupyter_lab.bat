@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "ZLC_TUTORIALS_MODE=run"
if "%~1"=="--help" goto zlc_help
if "%~1"=="/?" goto zlc_help
if /I "%~1"=="--check" set "ZLC_TUTORIALS_MODE=check"
if not "%~1"=="" if not "%ZLC_TUTORIALS_MODE%"=="check" (
    echo Unknown option: %~1
    echo.
    goto zlc_help
)

if not exist "tutorials\" (
    echo Cannot find the tutorials folder next to this .bat file.
    if "%ZLC_NO_PAUSE%"=="" pause
    exit /b 1
)

set "ZLC_TUTORIALS_DIR=%~dp0tutorials"

if defined ZLC_TUTORIALS_PYTHON (
    if exist "%ZLC_TUTORIALS_PYTHON%" (
        call :run_python_path "%ZLC_TUTORIALS_PYTHON%"
    ) else (
        call :run_python_cmd "%ZLC_TUTORIALS_PYTHON%"
    )
    exit /b !ERRORLEVEL!
)

if exist "%~dp0.zlc_python_path" (
    set /p "ZLC_STORED_PY="<"%~dp0.zlc_python_path"
    if exist "!ZLC_STORED_PY!" (
        call :run_python_path "!ZLC_STORED_PY!"
        exit /b !ERRORLEVEL!
    )
    echo Ignoring stale .zlc_python_path: !ZLC_STORED_PY!
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
    call :run_python_cmd "python"
    exit /b !ERRORLEVEL!
)

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    call :run_python_cmd "py -3"
    exit /b !ERRORLEVEL!
)

where jupyter >nul 2>nul
if %ERRORLEVEL%==0 (
    if "%ZLC_TUTORIALS_MODE%"=="check" (
        echo ZLC tutorials Jupyter command: jupyter
        jupyter --version
        exit /b !ERRORLEVEL!
    )
    cd /d "%ZLC_TUTORIALS_DIR%"
    jupyter lab .
    set "EXIT_CODE=!ERRORLEVEL!"
    if not "!EXIT_CODE!"=="0" if "%ZLC_NO_PAUSE%"=="" pause
    exit /b !EXIT_CODE!
)

echo Could not find python, py, or jupyter on PATH.
echo Run install_requirements.bat, set ZLC_TUTORIALS_PYTHON, or start Jupyter Lab from VS Code's selected Python environment.
if "%ZLC_NO_PAUSE%"=="" pause
exit /b 1

:zlc_help
echo Open the Zou_lab_control tutorial notebooks in Jupyter Lab.
echo.
echo Usage:
echo   start_tutorials_jupyter_lab.bat
echo   start_tutorials_jupyter_lab.bat --check
echo.
echo Python selection order:
echo   ZLC_TUTORIALS_PYTHON, .zlc_python_path, python, py -3, jupyter
echo.
exit /b 0

:run_python_path
set "ZLC_PYTHON_EXE=%~1"
if "%ZLC_TUTORIALS_MODE%"=="check" (
    echo ZLC tutorials Python: %ZLC_PYTHON_EXE%
    "%ZLC_PYTHON_EXE%" -c "import sys; print(sys.executable)"
    if errorlevel 1 exit /b !ERRORLEVEL!
    "%ZLC_PYTHON_EXE%" -m jupyter --version
    exit /b !ERRORLEVEL!
)
cd /d "%ZLC_TUTORIALS_DIR%"
"%ZLC_PYTHON_EXE%" -m jupyter lab .
set "EXIT_CODE=!ERRORLEVEL!"
if not "!EXIT_CODE!"=="0" if "%ZLC_NO_PAUSE%"=="" pause
exit /b !EXIT_CODE!

:run_python_cmd
set "ZLC_PYTHON_CMD=%~1"
if "%ZLC_TUTORIALS_MODE%"=="check" (
    echo ZLC tutorials Python command: %ZLC_PYTHON_CMD%
    %ZLC_PYTHON_CMD% -c "import sys; print(sys.executable)"
    if errorlevel 1 exit /b !ERRORLEVEL!
    %ZLC_PYTHON_CMD% -m jupyter --version
    exit /b !ERRORLEVEL!
)
cd /d "%ZLC_TUTORIALS_DIR%"
%ZLC_PYTHON_CMD% -m jupyter lab .
set "EXIT_CODE=!ERRORLEVEL!"
if not "!EXIT_CODE!"=="0" if "%ZLC_NO_PAUSE%"=="" pause
exit /b !EXIT_CODE!
