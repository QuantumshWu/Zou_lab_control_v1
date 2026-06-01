@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not "%~1"=="" (
    set "PYTHON_CMD="%~1""
    goto verify_python
)

call :from_vscode_settings
if defined PYTHON_CMD goto verify_python

call :from_jupyter_kernels
if defined PYTHON_CMD goto verify_python

call :from_path
if defined PYTHON_CMD goto verify_python

echo Could not find the Python interpreter used by VSCode.
echo.
echo Easiest option: in the VSCode notebook, run:
echo.
echo   %%run ../install_current_kernel.py
echo.
echo That installs into the currently selected notebook kernel directly.
echo.
echo In the VSCode notebook, run this cell:
echo.
echo   import sys
echo   print(sys.executable)
echo.
echo Then paste that full python.exe path here, or press Enter to skip.
set /p "PYTHON_EXE=python.exe path: "
if not "%PYTHON_EXE%"=="" (
    set "PYTHON_CMD="%PYTHON_EXE%""
    goto verify_python
)

echo.
echo No kernel path was provided. I will not install a separate Python automatically,
echo because VSCode might keep using the old kernel. Install Python manually only if
echo you actually need a new interpreter:
echo   https://www.python.org/downloads/windows/
pause
exit /b 1

:verify_python
echo Using Python:
%PYTHON_CMD% -c "import sys; print(sys.executable)"
if errorlevel 1 (
    echo.
    echo The selected Python command failed:
    echo   %PYTHON_CMD%
    pause
    exit /b 1
)

%PYTHON_CMD% -m pip --version >nul 2>nul
if errorlevel 1 (
    echo pip was not found in this kernel; trying ensurepip...
    %PYTHON_CMD% -m ensurepip --upgrade
    if errorlevel 1 (
        echo.
        echo Could not enable pip for this Python kernel.
        echo Try running this inside the VSCode notebook instead:
        echo   import sys, subprocess
        echo   subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        pause
        exit /b 1
    )
)

echo.
echo Installing Python packages from requirements.txt into this kernel...
%PYTHON_CMD% -m pip install --upgrade pip
if errorlevel 1 exit /b %errorlevel%

%PYTHON_CMD% -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 exit /b %errorlevel%

echo.
echo Registering this interpreter as a Jupyter kernel for VSCode...
%PYTHON_CMD% -m ipykernel install --user --name zou_lab_control --display-name "Python (Zou lab control)"
if errorlevel 1 exit /b %errorlevel%

echo.
echo Done. In VSCode, choose kernel: Python (Zou lab control).
echo Restart the Jupyter kernel after installing.
pause
exit /b 0

:from_vscode_settings
for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=(Get-Location).Path; $p=Join-Path $root '.vscode/settings.json'; if(Test-Path $p){ $j=Get-Content $p -Raw | ConvertFrom-Json; $v=$j.'python.defaultInterpreterPath'; if(-not $v){$v=$j.'python.pythonPath'}; if($v){ $v=$v.Replace('${workspaceFolder}', $root); $v=[Environment]::ExpandEnvironmentVariables($v); if((Test-Path $v -PathType Container)){ foreach($n in @('python.exe','Scripts/python.exe')){ $c=Join-Path $v $n; if(Test-Path $c){ $c; break } } } elseif(Test-Path $v){ $v } } }"`) do (
    set "PYTHON_CMD="%%I""
    exit /b 0
)
exit /b 0

:from_jupyter_kernels
for /f "usebackq delims=" %%I in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$dirs=@(); if($env:APPDATA){$dirs+=Join-Path $env:APPDATA 'jupyter/kernels'}; if($env:PROGRAMDATA){$dirs+=Join-Path $env:PROGRAMDATA 'jupyter/kernels'}; if($env:LOCALAPPDATA){$dirs+=Join-Path $env:LOCALAPPDATA 'jupyter/kernels'}; $items=@(); foreach($d in $dirs){ if(Test-Path $d){ Get-ChildItem $d -Filter kernel.json -Recurse -ErrorAction SilentlyContinue | ForEach-Object { try { $j=Get-Content $_.FullName -Raw | ConvertFrom-Json; $exe=$j.argv[0]; if($exe -and (Test-Path $exe)){ $items += [pscustomobject]@{Name=$j.display_name; Path=$exe} } } catch {} } } }; $items=@($items | Sort-Object Path -Unique); if($items.Count -eq 1){ $items[0].Path } elseif($items.Count -gt 1){ Write-Host 'Found Jupyter kernels:'; for($i=0;$i -lt $items.Count;$i++){ Write-Host ('[{0}] {1}  {2}' -f ($i+1),$items[$i].Name,$items[$i].Path) }; $n=Read-Host 'Choose kernel number, or press Enter to skip'; if($n -match '^[0-9]+$' -and [int]$n -ge 1 -and [int]$n -le $items.Count){ $items[[int]$n-1].Path } }"`) do (
    set "PYTHON_CMD="%%I""
    exit /b 0
)
exit /b 0

:from_path
py -3 -c "import sys" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=py -3"
    exit /b 0
)

python -c "import sys" >nul 2>nul
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    exit /b 0
)

exit /b 0
