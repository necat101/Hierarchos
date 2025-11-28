Write-Host "===========================================" -ForegroundColor Green
Write-Host "== Setting up Hierarchos Environment...  ==" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green
Write-Host ""

# --- Change to script's directory ---
Push-Location (Split-Path $MyInvocation.MyCommand.Path)

# --- STEP 1: Check Python ---
Write-Host "[1/5] Checking for Python..."
$python_exists = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $python_exists) {
    Write-Host "❌ Python not found. Please install Python 3.10+ (3.13 supported) and add it to your PATH." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    Pop-Location
    exit 1
}

# --- STEP 2: Check/Install Build Tools ---
Write-Host "`n[2/5] Checking for Microsoft C++ Build Tools..."
$vswhere_path = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vcvars_path = ""

if (Test-Path $vswhere_path) {
    Write-Host "INFO: Checking for existing C++ tools via vswhere..."
    $vcvars_path = & $vswhere_path -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvarsall.bat" 2>$null
}

if ($vcvars_path -and (Test-Path $vcvars_path)) {
    Write-Host "✅ Microsoft C++ Build Tools (x64) already installed."
} else {
    Write-Host "INFO: C++ Build Tools (x64) not found. Proceeding with installation..."
    $installer_url = "https://aka.ms/vs/17/release/vs_buildtools.exe"
    $installer_out = Join-Path $env:TEMP "vs_buildtools.exe"
    
    Write-Host "Downloading Visual Studio Build Tools installer to $installer_out..."
    try {
        Invoke-WebRequest -Uri $installer_url -OutFile $installer_out -UseBasicParsing
        Write-Host "✅ Download complete."
    } catch {
        Write-Host "❌ Download failed. Please download manually from:" -ForegroundColor Red
        Write-Host "https://visualstudio.microsoft.com/visual-cpp-build-tools/"
        Read-Host "Press Enter to exit"
        Pop-Location
        exit 1
    }

    Write-Host "`nInstalling required components... (this can take several minutes)"
    Start-Process -FilePath $installer_out -ArgumentList "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --includeRecommended" -Wait
    
    Write-Host "✅ Build Tools installation completed."
    
    # Re-run vswhere
    $vcvars_path = & $vswhere_path -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvarsall.bat" 2>$null
}

# --- STEP 3: Initialize Compiler Environment ---
Write-Host "`n[3/5] Initializing compiler environment..."
if (-not ($vcvars_path -and (Test-Path $vcvars_path))) {
    Write-Host "❌ Could not find vcvarsall.bat automatically even after install." -ForegroundColor Red
    Write-Host "Please close this terminal, open an 'x64 Native Tools Command Prompt', and re-run this script."
    Read-Host "Press Enter to exit"
    Pop-Location
    exit 1
}

Write-Host "✅ Found vcvarsall.bat at: $vcvars_path"
Write-Host "Importing 64-bit (x64) compiler environment..."
# This is the PowerShell way to run a batch file and import its environment variables
# It runs vcvarsall.bat, then exports the environment to a temp file, then imports it.
$tempEnvFile = [System.IO.Path]::GetTempFileName()
cmd.exe /c "call `"$vcvars_path`" x64 >nul 2>&1 && set > `"$tempEnvFile`""
Get-Content $tempEnvFile | Foreach-Object {
  if ($_ -match "^(.*?)=(.*)$") {
    Set-Item -Path "Env:\$($Matches[1])" -Value $Matches[2]
  }
}
Remove-Item $tempEnvFile
Write-Host "✅ 64-bit (x64) compiler environment initialized."

# --- STEP 4: Install Python Dependencies ---
Write-Host "`n[4/5] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements_kernel.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install Python dependencies." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    Pop-Location
    exit 1
}

# --- STEP 5: Build Hierarchos Kernel ---
Write-Host "`n[5/5] Building Hierarchos C++ kernel..."
# The environment is already set, so pip will find the 64-bit compiler
pip install .
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed. Check the error log above." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    Pop-Location
    exit 1
}

Write-Host ""
Write-Host "==============================================================" -ForegroundColor Green
Write-Host "== ✅ Setup Complete!                                      ==" -ForegroundColor Green
Write-Host "== The Hierarchos kernel is built and ready to run.        ==" -ForegroundColor Green
Write-Host "==============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run the model. The 'torch.compile' environment"
Write-Host "is now handled *automatically* inside the Python script."
Write-Host ""
Read-Host "Press Enter to close"
Pop-Location
