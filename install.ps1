# HK Universal AI Framework - Web Installer for Windows (PowerShell)
# Usage: irm https://raw.githubusercontent.com/harshitkhandelwal208/hk/main/install.ps1 | iex
$ErrorActionPreference = "Stop"

$Repo = "harshitkhandelwal208/hk"
$Version = "v1.0.0"
$BinaryName = "hk-windows-x86_64.exe"
$DownloadUrl = "https://github.com/$Repo/releases/download/$Version/$BinaryName"

$InstallDir = Join-Path $env:LOCALAPPDATA "hk\bin"
if (!(Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
}

$TargetPath = Join-Path $InstallDir "hk.exe"

Write-Host "==> Downloading HK CLI ($Version) for Windows x86_64..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $DownloadUrl -OutFile $TargetPath

# Add to user PATH if not present
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
    $env:Path = "$env:Path;$InstallDir"
    Write-Host "==> Added $InstallDir to user PATH." -ForegroundColor Green
}

Write-Host "==> HK CLI successfully installed to: $TargetPath" -ForegroundColor Green
Write-Host "==> Open a new terminal and run 'hk --help' to get started." -ForegroundColor Yellow
