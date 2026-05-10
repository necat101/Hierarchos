param(
    [string]$Python = "py",
    [string[]]$PythonArgs = @(),
    [switch]$InstallDeps,
    [switch]$SkipBackend,
    [switch]$SkipRust
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$GuiDir = Join-Path $Root "hierarchos-gui"
$ReleaseDir = Join-Path $Root "dist\Hierarchos-Windows"
$BackendDist = Join-Path $Root "dist\hierarchos-backend"
$BackendTarget = Join-Path $ReleaseDir "backend"

function Invoke-HierarchosPython {
    param([Parameter(Position = 0)][string[]]$Arguments)
    & $Python @PythonArgs @Arguments
    $script:LastPythonExitCode = $LASTEXITCODE
}

if ($InstallDeps) {
    Invoke-HierarchosPython @("-m", "pip", "install", "-r", (Join-Path $Root "tools\windows_backend_requirements.txt"))
    if ($script:LastPythonExitCode -ne 0) { throw "Python dependency installation failed." }

    $verifyTorch = "import torch; print('PyTorch', torch.__version__, 'CUDA build', torch.version.cuda); raise SystemExit(0 if torch.version.cuda else 2)"
    Invoke-HierarchosPython @("-c", $verifyTorch)
    if ($script:LastPythonExitCode -ne 0) {
        throw "Installed PyTorch is CPU-only. The Windows release must use the CUDA wheel so the same package supports CUDA and CPU fallback."
    }
}

if (-not $SkipBackend) {
    $BackendBuild = Join-Path $Root "build\hierarchos_backend"
    if (Test-Path $BackendDist) { Remove-Item -LiteralPath $BackendDist -Recurse -Force }
    if (Test-Path $BackendBuild) { Remove-Item -LiteralPath $BackendBuild -Recurse -Force }
    Invoke-HierarchosPython @("-m", "PyInstaller", "--clean", "--noconfirm", (Join-Path $Root "tools\hierarchos_backend.spec"))
    if ($script:LastPythonExitCode -ne 0) { throw "PyInstaller backend build failed." }
}

if (-not $SkipRust) {
    Push-Location $GuiDir
    try {
        cargo build --release
        if ($LASTEXITCODE -ne 0) { throw "Rust GUI build failed." }
    }
    finally {
        Pop-Location
    }
}

if (Test-Path $ReleaseDir) {
    Remove-Item -LiteralPath $ReleaseDir -Recurse -Force
}
New-Item -ItemType Directory -Path $ReleaseDir | Out-Null

$GuiExe = Join-Path $GuiDir "target\release\hierarchos-gui.exe"
if (-not (Test-Path $GuiExe)) {
    throw "GUI executable not found: $GuiExe"
}
Copy-Item -LiteralPath $GuiExe -Destination (Join-Path $ReleaseDir "Hierarchos.exe")

if (Test-Path $BackendDist) {
    New-Item -ItemType Directory -Path $BackendTarget | Out-Null
    Get-ChildItem -LiteralPath $BackendDist | Copy-Item -Destination $BackendTarget -Recurse -Force
}
elseif (-not $SkipBackend) {
    throw "Backend dist not found: $BackendDist"
}

foreach ($name in @("LICENSE.md", "README.md")) {
    $path = Join-Path $Root $name
    if (Test-Path $path) {
        Copy-Item -LiteralPath $path -Destination $ReleaseDir
    }
}

$readme = @"
Hierarchos Windows Release

Run Hierarchos.exe.

The GUI first looks for backend\hierarchos-backend.exe. That backend bundles
the Hierarchos Python package plus the PyTorch/Transformers runtime, so users
do not need to clone this repository or install Python for normal inference.

The backend is built with the CUDA-enabled PyTorch wheel. That single runtime
also includes CPU execution: Auto uses NVIDIA CUDA when PyTorch can see a CUDA
GPU, and otherwise falls back to CPU for non-NVIDIA systems and handheld PCs.
Selecting CUDA explicitly will report a clear error if the NVIDIA driver/GPU is
not available instead of silently pretending to run on GPU.

Model sources accepted by the GUI:
- A Hugging Face repo id, for example author/model-name
- A local model directory containing hierarchos.pt or model.pt plus tokenizer files
- A direct .pt inference checkpoint with config embedded or a neighboring
  hierarchos_config.json

Tokenizer path selection is not required. Hierarchos loads the tokenizer from
the model directory automatically.

When closing with a model loaded, the app asks whether to save runtime LTM
updates. Saving writes hierarchos_ltm_updates.pt next to the loaded model and
reloads that sidecar automatically on future loads. Discard closes without
writing new LTM updates.

Downloaded Hugging Face models are cached under the user's local Hierarchos
app data directory. If the bundled backend is missing, Settings can fall back
to a system Python by changing the backend field from bundled to python.
"@
Set-Content -LiteralPath (Join-Path $ReleaseDir "README_RELEASE.txt") -Value $readme -Encoding UTF8

Write-Host "Release bundle created:"
Write-Host "  $ReleaseDir"
