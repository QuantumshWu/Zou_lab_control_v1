@echo off
setlocal

cd /d "%~dp0"

if not exist "tutorials\" (
    echo Cannot find the tutorials folder next to this .bat file.
    pause
    exit /b 1
)

cd /d "%~dp0tutorials"

where python >nul 2>nul
if %ERRORLEVEL%==0 (
    python -m jupyter lab .
    exit /b %ERRORLEVEL%
)

where py >nul 2>nul
if %ERRORLEVEL%==0 (
    py -m jupyter lab .
    exit /b %ERRORLEVEL%
)

where jupyter >nul 2>nul
if %ERRORLEVEL%==0 (
    jupyter lab .
    exit /b %ERRORLEVEL%
)

echo Could not find python, py, or jupyter on PATH.
echo If VS Code has a working kernel but PowerShell/cmd does not, run install_requirements.bat from that kernel first,
echo or start Jupyter Lab from VS Code's selected Python environment.
pause
exit /b 1
