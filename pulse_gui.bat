@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python "%~dp0pulse_gui.py" %*
    set "EXIT_CODE=%ERRORLEVEL%"
    if not "!EXIT_CODE!"=="0" pause
    exit /b !EXIT_CODE!
)

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -3 "%~dp0pulse_gui.py" %*
    set "EXIT_CODE=%ERRORLEVEL%"
    if not "!EXIT_CODE!"=="0" pause
    exit /b !EXIT_CODE!
)

echo Could not find python or py on PATH.
echo Run install_requirements.bat or start this from the same terminal/kernel environment that has Zou_lab_control installed.
pause
exit /b 1
