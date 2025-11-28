$ErrorActionPreference = "Stop"

$zludaPath = "ZLUDA\zluda_with.exe"
if (Test-Path $zludaPath) {
    Write-Host "✅ ZLUDA already installed in .\ZLUDA"
    exit 0
}

Write-Host "ZLUDA not found. Downloading release v5..."

$downloadUrl = "https://github.com/vosen/ZLUDA/releases/download/v5/zluda-windows-1c0c421.zip"
$zipFile = "zluda.zip"
$tempDir = "ZLUDA_TEMP"

try {
    # Clean up previous attempts
    if (Test-Path $tempDir) { Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue }
    if (Test-Path $zipFile) { Remove-Item -Path $zipFile -Force -ErrorAction SilentlyContinue }
    
    Write-Host "Downloading ZLUDA from: $downloadUrl"
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipFile
    
    # Wait for file handle release
    Start-Sleep -Seconds 2

    Write-Host "Extracting ZLUDA..."
    Expand-Archive -Path $zipFile -DestinationPath $tempDir -Force
    
    Start-Sleep -Seconds 2

    # Find the extracted folder
    $extractedDir = Get-ChildItem -Path $tempDir -Directory | Select-Object -First 1
    
    if ($extractedDir) {
        Write-Host "Moving extracted files to ZLUDA..."
        
        # If ZLUDA directory exists but check failed, remove it to ensure clean install
        if (Test-Path "ZLUDA") {
            Write-Host "Removing incomplete ZLUDA directory..."
            Remove-Item -Path "ZLUDA" -Recurse -Force
            Start-Sleep -Seconds 1
        }
        
        # Move the extracted directory to be ZLUDA
        Move-Item -Path $extractedDir.FullName -Destination "ZLUDA" -Force
    } else {
        # Files directly in tempDir?
        if (Test-Path "ZLUDA") { Remove-Item -Path "ZLUDA" -Recurse -Force }
        New-Item -ItemType Directory -Path "ZLUDA" | Out-Null
        Get-ChildItem -Path $tempDir | Move-Item -Destination "ZLUDA" -Force
    }

    Write-Host "Cleaning up..."
    Start-Sleep -Seconds 1
    if (Test-Path $tempDir) { Remove-Item -Path $tempDir -Recurse -Force }
    if (Test-Path $zipFile) { Remove-Item -Path $zipFile -Force }

    Write-Host "✅ ZLUDA installed successfully to .\ZLUDA"

} catch {
    Write-Host "❌ Failed to install ZLUDA: $($_.Exception.Message)"
    Write-Host "Debug: Failed at line $($_.InvocationInfo.ScriptLineNumber)"
    exit 1
}
