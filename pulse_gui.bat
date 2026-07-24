@echo off
setlocal EnableExtensions
set "ZLC_GUI_ENTRY=pulse_gui.py"
set "ZLC_GUI_LABEL=pulse GUI"
set "ZLC_GUI_PYTHON=%ZLC_PULSE_GUI_PYTHON%"
call "%~dp0_launch_gui.bat" %*
exit /b %ERRORLEVEL%
