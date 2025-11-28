$ErrorActionPreference = "Stop"

# URL provided by user
$pythonUrl = "https://www.python.org/ftp/python/3.13.9/python-3.13.9-amd64.exe"
$installerPath = "$env:TEMP\python-3.13.9-amd64.exe"

# Check if standard Python is already installed (exclude WindowsApps)
$pythonPath = Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if ($pythonPath -and $pythonPath -notmatch "WindowsApps") {
    Write-Host "✅ Standard Python is already installed at: $pythonPath"
    exit 0
}

Write-Host "Standard Python not found (or Windows Store version detected)."
Write-Host "Downloading Python 3.13.9..."
Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath

Write-Host "Installing Python 3.13.9 (this may take a minute)..."
# /quiet = silent install
# InstallAllUsers=0 = install for current user (no admin needed usually, matches user request)
# PrependPath=1 = add to PATH
# Include_test=0 = skip test suite to save space/time
$process = Start-Process -FilePath $installerPath -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1 Include_test=0" -Wait -PassThru

if ($process.ExitCode -ne 0) {
    Write-Host "❌ Python installation failed with exit code $($process.ExitCode)"
    exit 1
}

Write-Host "✅ Python installed successfully."

# --- Disable App Execution Aliases (Windows Store Python) ---
Write-Host "Disabling Windows Store Python aliases..."
$aliasPath = "$env:LOCALAPPDATA\Microsoft\WindowsApps"
$aliases = @("python.exe", "python3.exe", "python3.13.exe")

foreach ($alias in $aliases) {
    $fullPath = Join-Path $aliasPath $alias
    if (Test-Path $fullPath) {
        try {
            # We can't easily "uncheck" the box via script, but we can rename the alias file 
            # to prevent it from intercepting the command.
            # Or better, ensure our new Python is EARLIER in the PATH.
            # But renaming is a safer bet to kill the store version interference.
            Rename-Item -Path $fullPath -NewName "$alias.bak" -ErrorAction SilentlyContinue
            Write-Host "Disabled alias: $alias"
        } catch {
            Write-Host "Warning: Could not disable alias $alias (might be in use or permission denied)."
        }
    }
}

# --- Refresh PATH for the current session ---
# The installer adds to the User registry PATH, but we need it NOW.
# We'll construct the expected path manually to be sure.
$installDir = "$env:LOCALAPPDATA\Programs\Python\Python313"
$scriptsDir = "$installDir\Scripts"

$env:Path = "$installDir;$scriptsDir;$env:Path"

Write-Host "Environment updated. Python location: $(Get-Command python.exe | Select-Object -ExpandProperty Source)"

# Clean up
Remove-Item -Path $installerPath -Force
