# PowerShell build script for Windows x64 Portable ZIP & Inno Setup EXE
$ErrorActionPreference = "Continue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$OutDir = Join-Path $RepoRoot "release_artifacts"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "=== [1/5] Building standalone Python backend via PyInstaller ==="
python "$RepoRoot\scripts\build_backend_binary.py"
if ($LASTEXITCODE -ne 0) {
  Write-Host "::error::build_backend_binary.py failed with exit code $LASTEXITCODE"
  exit $LASTEXITCODE
}

Write-Host "=== [2/5] Preparing Flutter Windows runner ==="
python "$RepoRoot\scripts\prepare_flutter_platforms.py"
if ($LASTEXITCODE -ne 0) {
  Write-Host "::error::prepare_flutter_platforms.py failed with exit code $LASTEXITCODE"
  exit $LASTEXITCODE
}

Write-Host "=== [3/5] Building Flutter Windows x64 Release ==="
Push-Location "$RepoRoot\apps\flutter_app"
flutter config --enable-windows-desktop
$BuildLog = Join-Path $OutDir "flutter_windows_build.log"
flutter build windows --release *> $BuildLog
$FlutterExit = $LASTEXITCODE
Get-Content $BuildLog -Tail 40
Pop-Location
if ($FlutterExit -ne 0) {
  $TailErr = (Get-Content $BuildLog -Tail 12) -join " | "
  Write-Host "::error::flutter build windows failed ($FlutterExit): $TailErr"
  exit $FlutterExit
}

Write-Host "=== [4/5] Staging Release bundle & creating Portable ZIP ==="
$CandidateA = Join-Path $RepoRoot "apps\flutter_app\build\windows\x64\runner\Release"
$CandidateB = Join-Path $RepoRoot "apps\flutter_app\build\windows\runner\Release"
if (Test-Path $CandidateA) {
  $ReleaseBundle = $CandidateA
} elseif (Test-Path $CandidateB) {
  $ReleaseBundle = $CandidateB
} else {
  Write-Host "::error::Flutter Windows Release folder not found under build\windows"
  exit 1
}

$WinBundle = Join-Path $OutDir "win_bundle"
if (Test-Path $WinBundle) { Remove-Item -Recurse -Force $WinBundle }
Copy-Item -Recurse -Force $ReleaseBundle $WinBundle
Copy-Item -Recurse -Force (Join-Path $OutDir "backend_bundle\modelforge-backend") (Join-Path $WinBundle "backend")

$ZipPath = Join-Path $OutDir "ModelForge-Windows-x64-Portable.zip"
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
Compress-Archive -Path "$WinBundle\*" -DestinationPath $ZipPath -Force
if (-not (Test-Path $ZipPath)) {
  Write-Host "::error::Failed to create ModelForge-Windows-x64-Portable.zip"
  exit 1
}

Write-Host "=== [5/5] Building Inno Setup Installer (ModelForge-Windows-x64-Setup.exe) ==="
$IsccCandidates = @(
  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
  "C:\Program Files\Inno Setup 6\ISCC.exe"
)
$Iscc = $IsccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Iscc) {
  choco install innosetup -y --no-progress
  $Iscc = $IsccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $Iscc) {
  Write-Host "::error::Inno Setup ISCC.exe not found"
  exit 1
}
& $Iscc "$RepoRoot\platform\windows\modelforge.iss"
if ($LASTEXITCODE -ne 0) {
  Write-Host "::error::ISCC.exe failed with exit code $LASTEXITCODE"
  exit $LASTEXITCODE
}

Get-ChildItem $OutDir -Filter "ModelForge-Windows-x64*" | Select-Object Name, Length
