# Quick Fix for Python 3.13 Compatibility Issue

## The Problem
You're running **Python 3.13**, but `torch-directml` only supports **Python 3.8 through 3.12**.

## Solution 1: Install Python 3.12 (Recommended)

### Step 1: Download Python 3.12
Visit: https://www.python.org/downloads/release/python-3127/

Download: **Windows installer (64-bit)**

### Step 2: Install Python 3.12
1. Run the installer
2. **IMPORTANT**: Check "Add python.exe to PATH"
3. Click "Install Now"

### Step 3: Verify Installation
Open a **NEW** PowerShell window:
```powershell
python --version
# Should show: Python 3.12.7
```

### Step 4: Install DirectML
```powershell
cd C:\Users\User\Downloads\Hierarchos-main\Hierarchos-main
.\setup_dml.bat
```

---

## Solution 2: Use Virtual Environment (Keep Python 3.13)

If you want to keep Python 3.13 for other projects:

### Step 1: Install Python 3.12 Alongside
- Install Python 3.12 as above
- During installation, you can choose a custom install location
- Don't check "Add to PATH" if you want Python 3.13 to remain default

### Step 2: Create Virtual Environment
```powershell
cd C:\Users\User\Downloads\Hierarchos-main\Hierarchos-main

# Create venv with Python 3.12
py -3.12 -m venv venv_dml

# If that doesn't work, try:
# C:\Python312\python.exe -m venv venv_dml
```

### Step 3: Activate Virtual Environment
```powershell
.\venv_dml\Scripts\Activate.ps1

# If you get execution policy error, run this first:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

You should see `(venv_dml)` in your prompt.

### Step 4: Verify Python Version in venv
```powershell
python --version
# Should show: Python 3.12.x
```

### Step 5: Install DirectML in venv
```powershell
pip install -r requirements_dml.txt
python test_directml.py
```

### Step 6: Always Activate venv Before Training
```powershell
# Every time you want to use DirectML:
cd C:\Users\User\Downloads\Hierarchos-main\Hierarchos-main
.\venv_dml\Scripts\Activate.ps1

# Then train:
python hierarchos.py --mode train --train data.jsonl ...
```

### Step 7: Deactivate venv When Done
```powershell
deactivate
```

---

## Solution 3: Direct pip Install (If setup.bat fails)

If you have Python 3.12 but the setup script errors:

```powershell
# Install directly
pip install torch-directml

# Install other dependencies
pip install transformers datasets numpy tqdm safetensors

# Test
python test_directml.py
```

---

## Verify Which Python is Being Used

```powershell
# Check Python version
python --version

# Check where Python is installed
where python

# List all Python versions (if installed via py launcher)
py --list
```

---

## Quick Decision Guide

**I'm okay reinstalling Python** → **Solution 1** (Simplest)

**I need Python 3.13 for other projects** → **Solution 2** (Virtual Environment)

**I just want to try one command** → **Solution 3** (Manual pip)

---

## After Installing

Run the verification script:
```powershell
python test_directml.py
```

You should see:
```
✓ torch-directml is installed
✓ DirectML device created
✓ Basic tensor operations successful
...
✓ All DirectML tests passed!
```

Then you're ready to train!

---

## Still Having Issues?

1. Make sure you're in a **NEW** PowerShell window after installing Python 3.12
2. Restart your computer (refreshes PATH)
3. Try the virtual environment approach instead
4. Check if you have conflicting PyTorch installations: `pip list | findstr torch`
