# PowerShell build script for Windows x64 Portable ZIP & Inno Setup EXE
$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$OutDir = Join-Path $RepoRoot "release_artifacts"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

python "$RepoRoot\scripts\build_backend_binary.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python "$RepoRoot\scripts\prepare_flutter_platforms.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Push-Location "$RepoRoot\apps\flutter_app"
flutter build windows --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Pop-Location

$ReleaseBundle = Join-Path $RepoRoot "apps\flutter_app\build\windows\x64\runner\Release"
$BackendDest = Join-Path $ReleaseBundle "backend"
if (Test-Path $BackendDest) { Remove-Item -Recurse -Force $BackendDest }
Copy-Item -Recurse -Force (Join-Path $OutDir "backend_bundle\modelforge-backend") $BackendDest

# Create Portable ZIP
$ZipPath = Join-Path $OutDir "ModelForge-Windows-x64-Portable.zip"
if (Test-Path $ZipPath) { Remove-Item -Force $ZipPath }
Compress-Archive -Path "$ReleaseBundle\*" -DestinationPath $ZipPath

# Build Setup EXE via Inno Setup (ISCC.exe)
$IsccCandidates = @(
  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
  "C:\Program Files\Inno Setup 6\ISCC.exe"
)
$Iscc = $IsccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Iscc) {
  choco install innosetup -y --no-progress
  $Iscc = $IsccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
& $Iscc "$RepoRoot\platform\windows\modelforge.iss"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Get-ChildItem $OutDir -Filter "ModelForge-Windows-x64*" | Select-Object Name, Length
