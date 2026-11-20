# Non-interactive provisioning for Windows (winget). Run: powershell -ExecutionPolicy Bypass -File setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$wg = @("--silent", "--accept-package-agreements", "--accept-source-agreements", "-e")

if (-not (Get-Command py -ErrorAction SilentlyContinue) -or -not (py -3.12 --version 2>$null)) {
    winget install --id Python.Python.3.12 @wg
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { winget install --id Gyan.FFmpeg @wg }  # full build: libx264 + libass
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { winget install --id OpenJS.NodeJS.LTS @wg }
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")

if (-not (Test-Path .venv)) { py -3.12 -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -q --upgrade pip
& $py -m pip install -q -r requirements.txt
& $py -m pip uninstall -y -q opencv-contrib-python opencv-python 2>$null
& $py -m pip install -q --force-reinstall --no-deps opencv-python-headless==4.11.0.86
New-Item -ItemType Directory -Force temp_work, final_clips | Out-Null
& $py scripts\verify_env.py
