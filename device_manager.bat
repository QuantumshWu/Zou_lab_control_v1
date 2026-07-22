@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

if defined ZLC_DEVICE_MANAGER_PYTHON (
    "%ZLC_DEVICE_MANAGER_PYTHON%" "%~dp0device_manager.py" %*
    exit /b !ERRORLEVEL!
)

if exist "%~dp0.zlc_python_path" (
    set /p "ZLC_STORED_PY="<"%~dp0.zlc_python_path"
    if exist "!ZLC_STORED_PY!" (
        "!ZLC_STORED_PY!" "%~dp0device_manager.py" %*
        exit /b !ERRORLEVEL!
    )
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python "%~dp0device_manager.py" %*
    exit /b !ERRORLEVEL!
)

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3 "%~dp0device_manager.py" %*
    exit /b !ERRORLEVEL!
)

echo Could not find Python. Run install_requirements.bat or set ZLC_DEVICE_MANAGER_PYTHON.
exit /b 1
