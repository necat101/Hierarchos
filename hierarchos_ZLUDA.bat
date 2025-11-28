@echo off
setlocal

:: Check if ZLUDA is installed
if not exist "ZLUDA\zluda_with.exe" (
    echo ❌ ZLUDA not found in .\ZLUDA
    echo Please run setup.bat to install it automatically.
    pause
    exit /b 1
)

:: --- NEW: Pre-load MSVC Environment for torch.compile ---
:: Try to find vcvars64.bat in standard locations
set "VCVARS_PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VCVARS_PATH%" (
    echo INFO: Loading MSVC environment from "%VCVARS_PATH%"...
    call "%VCVARS_PATH%" >nul 2>&1
    if errorlevel 1 (
        echo WARNING: Failed to load vcvars64.bat. torch.compile might fail.
    ) else (
        echo INFO: MSVC environment loaded successfully.
    )
) else (
    echo WARNING: vcvars64.bat not found at default location. torch.compile might fail.
)

:: Add ZLUDA to PATH for this session
set "PATH=%~dp0ZLUDA;%PATH%"

echo ==================================================
echo == Running Hierarchos with ZLUDA (AMD Support) ==
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
:: We explicitly pass --device cuda because ZLUDA mocks CUDA
ZLUDA\zluda_with.exe -- "%PYTHON_EXE%" hierarchos.py %* --device cuda

if %errorlevel% neq 0 (
    echo.
    echo ❌ Command failed with error code %errorlevel%
    exit /b %errorlevel%
)

endlocal
