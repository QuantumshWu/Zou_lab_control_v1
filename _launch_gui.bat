@echo off
setlocal EnableExtensions EnableDelayedExpansion

if not defined ZLC_GUI_ENTRY (
    echo ZLC GUI launcher requires ZLC_GUI_ENTRY.
    exit /b 2
)
if not defined ZLC_GUI_LABEL set "ZLC_GUI_LABEL=GUI"

cd /d "%~dp0"
set "ZLC_STATUS=1"

if defined ZLC_GUI_PYTHON (
    if exist "%ZLC_GUI_PYTHON%" (
        "%ZLC_GUI_PYTHON%" "%~dp0%ZLC_GUI_ENTRY%" %*
    ) else (
        %ZLC_GUI_PYTHON% "%~dp0%ZLC_GUI_ENTRY%" %*
    )
    set "ZLC_STATUS=!ERRORLEVEL!"
    goto finish
)

if exist "%~dp0.zlc_python_path" (
    set /p "ZLC_STORED_PY="<"%~dp0.zlc_python_path"
    if exist "!ZLC_STORED_PY!" (
        "!ZLC_STORED_PY!" "%~dp0%ZLC_GUI_ENTRY%" %*
        set "ZLC_STATUS=!ERRORLEVEL!"
        goto finish
    )
    echo Ignoring stale .zlc_python_path: !ZLC_STORED_PY!
)

where python >nul 2>nul
if !ERRORLEVEL!==0 (
    python "%~dp0%ZLC_GUI_ENTRY%" %*
    set "ZLC_STATUS=!ERRORLEVEL!"
    goto finish
)

where py >nul 2>nul
if !ERRORLEVEL!==0 (
    py -3 "%~dp0%ZLC_GUI_ENTRY%" %*
    set "ZLC_STATUS=!ERRORLEVEL!"
    goto finish
)

echo Could not find python or py on PATH.
echo Run install_requirements.bat, set the launcher's Python override, or use the environment that has Zou_lab_control installed.
set "ZLC_STATUS=1"

:finish
if "!ZLC_STATUS!"=="0" exit /b 0
echo.
echo ZLC %ZLC_GUI_LABEL% failed with code !ZLC_STATUS!.
echo Keep this window open and read the messages above.
if "%ZLC_NO_PAUSE%"=="" pause
exit /b !ZLC_STATUS!
