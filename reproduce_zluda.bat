@echo off
setlocal

:: Check if ZLUDA is installed
if not exist "ZLUDA\zluda_with.exe" (
    echo ❌ ZLUDA not found in .\ZLUDA
    echo Please run setup.bat to install it automatically.
    pause
    exit /b 1
)

:: Add ZLUDA to PATH for this session
set "PATH=%~dp0ZLUDA;%PATH%"

echo ==================================================
echo == Running ZLUDA Reproduction Script ==
echo ==================================================
echo.

:: Get the absolute path to the Python 3.10 executable
set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"

if not exist "%PYTHON_EXE%" (
    echo ❌ Python 3.10 not found at: "%PYTHON_EXE%"
    echo    Please run setup.bat to install it.
    pause
    exit /b 1
)

echo INFO: Using Python at: "%PYTHON_EXE%"
echo INFO: Launching ZLUDA...

:: Run the command
ZLUDA\zluda_with.exe -- "%PYTHON_EXE%" reproduce_zluda_error.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ Command failed with error code %errorlevel%
    exit /b %errorlevel%
)

endlocal
pause
