@echo off
echo ========================================
echo Hierarchos DirectML Setup
echo ========================================
echo.
echo This script will install DirectML support for AMD GPU acceleration
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.12 from python.org
    pause
    exit /b 1
)

echo Checking Python version...
echo.

REM Get Python version and check compatibility using Python itself
for /f "delims=" %%i in ('python -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}.{v.micro}')"') do set PYTHON_VERSION=%%i
for /f "delims=" %%i in ('python -c "import sys; v=sys.version_info; print(v.major)"') do set PYTHON_MAJOR=%%i
for /f "delims=" %%i in ('python -c "import sys; v=sys.version_info; print(v.minor)"') do set PYTHON_MINOR=%%i

echo Python detected: %PYTHON_VERSION%
echo.

REM Use Python to check if version is in valid range (3.8 to 3.12)
python -c "import sys; v=sys.version_info; exit(0 if (v.major == 3 and 8 <= v.minor <= 12) else 1)" >nul 2>&1

if errorlevel 1 (
    echo ================================================
    echo ERROR: Python %PYTHON_VERSION% is NOT compatible!
    echo ================================================
    echo.
    echo torch-directml requires Python 3.8 through 3.12
    echo You are using Python %PYTHON_VERSION%
    echo.
    echo SOLUTIONS:
    echo.
    echo Option 1: Install Python 3.12 ^(Recommended^)
    echo   1. Download from: https://www.python.org/downloads/release/python-3127/
    echo   2. Install it ^(keep "Add to PATH" checked^)
    echo   3. Rerun this setup script
    echo.
    echo Option 2: Use a Virtual Environment
    echo   1. Install Python 3.12 separately
    echo   2. Create venv: py -3.12 -m venv venv_dml
    echo   3. Activate: venv_dml\Scripts\activate
    echo   4. Rerun this setup script
    echo.
    echo ================================================
    pause
    exit /b 1
)

echo [OK] Python %PYTHON_VERSION% is compatible!
echo.

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed
    pause
    exit /b 1
)

echo Installing DirectML and dependencies...
echo.
echo WARNING: This will install torch-directml which includes PyTorch 2.4.1.
echo If you have a different PyTorch installation, it may be replaced.
echo.
pause

pip install -r requirements_dml.txt

if errorlevel 1 (
    echo.
    echo ================================================
    echo ERROR: Installation failed
    echo ================================================
    echo Please check the error messages above
    echo.
    echo Common issues:
    echo   - Python version incompatibility
    echo   - Network connection problems
    echo   - Conflicting packages
    echo.
    echo Try manual installation:
    echo   pip install torch-directml
    echo   pip install transformers datasets numpy tqdm safetensors
    echo ================================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Running DirectML verification tests...
echo.

python test_directml.py

if errorlevel 1 (
    echo.
    echo ================================================
    echo WARNING: Some DirectML tests failed
    echo ================================================
    echo Please review the errors above
    echo.
    echo If AMP tests failed, you can still train with --no-amp flag
    echo ================================================
) else (
    echo.
    echo ================================================
    echo SUCCESS: All DirectML tests passed!
    echo ================================================
)

echo.
echo ========================================
echo Setup Complete - You're Ready to Train!
echo ========================================
echo DirectML has been installed successfully
echo You can now train Hierarchos with AMD GPU acceleration
echo.
echo Quick Start:
echo   python hierarchos.py --mode train --train your_data.jsonl --epochs 3 --batch-size 4
echo.
echo DirectML will auto-detect, or use --device dml to force it
echo ========================================
echo.
pause
