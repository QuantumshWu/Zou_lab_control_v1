@echo off
setlocal EnableExtensions
set "ZLC_GUI_ENTRY=figure_viewer.py"
set "ZLC_GUI_LABEL=figure viewer"
set "ZLC_GUI_PYTHON=%ZLC_FIGURE_VIEWER_PYTHON%"
call "%~dp0_launch_gui.bat" %*
exit /b %ERRORLEVEL%
