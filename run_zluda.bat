@echo off
setlocal

:: Find Python 3.10
set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"

if not exist "%PYTHON_EXE%" (
    echo ❌ Python 3.10 not found.
    exit /b 1
)

:: Add ZLUDA to PATH
set "PATH=%~dp0ZLUDA;%PATH%"

echo INFO: Launching ZLUDA with command: "%PYTHON_EXE%" %*
ZLUDA\zluda_with.exe -- "%PYTHON_EXE%" %*
