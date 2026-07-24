@echo off
setlocal EnableExtensions
set "ZLC_GUI_ENTRY=task_console.py"
set "ZLC_GUI_LABEL=task console"
set "ZLC_GUI_PYTHON=%ZLC_TASK_CONSOLE_PYTHON%"
call "%~dp0_launch_gui.bat" %*
exit /b %ERRORLEVEL%
