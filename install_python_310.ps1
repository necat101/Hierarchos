$ErrorActionPreference = "Stop"

# Python 3.10.11 (Stable, widely supported)
$pythonUrl = "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe"
$installerPath = "$env:TEMP\python-3.10.11-amd64.exe"
$installDir = "$env:LOCALAPPDATA\Programs\Python\Python310"

# Check if Python 3.10 is already installed
if (Test-Path "$installDir\python.exe") {
    Write-Host "✅ Python 3.10 is already installed at: $installDir"
    exit 0
}

Write-Host "Downloading Python 3.10.11..."
Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath

Write-Host "Installing Python 3.10.11 (this may take a minute)..."
# /quiet = silent install
# InstallAllUsers=0 = install for current user
# PrependPath=1 = add to PATH
# Include_test=0 = skip test suite
$process = Start-Process -FilePath $installerPath -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0 TargetDir=`"$installDir`"" -Wait -PassThru

if ($process.ExitCode -ne 0) {
    Write-Host "❌ Python installation failed with exit code $($process.ExitCode)"
    exit 1
}

Write-Host "✅ Python 3.10 installed successfully."

# --- Refresh PATH for the current session ---
$scriptsDir = "$installDir\Scripts"
$env:Path = "$installDir;$scriptsDir;$env:Path"

Write-Host "Environment updated. Python location: $installDir\python.exe"

# Clean up
Remove-Item -Path $installerPath -Force
