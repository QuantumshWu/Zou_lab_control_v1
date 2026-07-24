@echo off
setlocal EnableExtensions
set "ZLC_GUI_ENTRY=device_manager.py"
set "ZLC_GUI_LABEL=device manager"
set "ZLC_GUI_PYTHON=%ZLC_DEVICE_MANAGER_PYTHON%"
call "%~dp0_launch_gui.bat" %*
exit /b %ERRORLEVEL%
