@echo off
setlocal EnableExtensions EnableDelayedExpansion

if not defined ZLC_PULSE_GUI_INNER (
    set "ZLC_PULSE_GUI_INNER=1"
    call "%~f0" %*
    set "ZLC_STATUS=!ERRORLEVEL!"
    if "!ZLC_STATUS!"=="0" (
        if "%~1"=="--help" exit /b 0
        if "%~1"=="/?" exit /b 0
        echo.
        echo ZLC pulse GUI closed normally.
        echo You can close this window, or press any key to exit.
        if "%ZLC_NO_PAUSE%"=="" pause
    ) else (
        echo.
        echo ZLC pulse GUI failed with code !ZLC_STATUS!.
        echo Keep this window open and read the messages above.
        if "%ZLC_NO_PAUSE%"=="" pause
    )
    exit /b !ZLC_STATUS!
)

cd /d "%~dp0"

if defined ZLC_PULSE_GUI_PYTHON (
    if exist "%ZLC_PULSE_GUI_PYTHON%" (
        "%ZLC_PULSE_GUI_PYTHON%" "%~dp0pulse_gui.py" %*
    ) else (
        %ZLC_PULSE_GUI_PYTHON% "%~dp0pulse_gui.py" %*
    )
    exit /b !ERRORLEVEL!
)

if exist "%~dp0.zlc_python_path" (
    set /p "ZLC_STORED_PY="<"%~dp0.zlc_python_path"
    if exist "!ZLC_STORED_PY!" (
        "!ZLC_STORED_PY!" "%~dp0pulse_gui.py" %*
        exit /b !ERRORLEVEL!
    )
    echo Ignoring stale .zlc_python_path: !ZLC_STORED_PY!
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python "%~dp0pulse_gui.py" %*
    exit /b !ERRORLEVEL!
)

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3 "%~dp0pulse_gui.py" %*
    exit /b !ERRORLEVEL!
)

echo Could not find python or py on PATH.
echo Run install_requirements.bat, set ZLC_PULSE_GUI_PYTHON, or start this from the same terminal/kernel environment that has Zou_lab_control installed.
exit /b 1
