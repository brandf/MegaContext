# Build script for Quartz static site generation
$ErrorActionPreference = "Stop"

$scriptDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptDir
$obsidianPath = Join-Path $repoRoot "obsidian"
$quartzPath = Join-Path $repoRoot "node_modules\@jackyzha0\quartz"
$quartzContent = Join-Path $quartzPath "content"

Write-Host "Setting up Quartz build environment..." -ForegroundColor Cyan

# Check if dependencies are installed
if (-not (Test-Path (Join-Path $repoRoot "node_modules"))) {
    Write-Host "Installing npm dependencies..." -ForegroundColor Yellow
    Set-Location $repoRoot
    npm install
}

# Check if Quartz dependencies are installed
if (-not (Test-Path (Join-Path $quartzPath "node_modules"))) {
    Write-Host "Installing Quartz dependencies..." -ForegroundColor Yellow
    Set-Location $quartzPath
    npm install
    Set-Location $repoRoot
}

# Remove existing content symlink if it exists
if (Test-Path $quartzContent) {
    Write-Host "Removing existing content directory..." -ForegroundColor Yellow
    Remove-Item $quartzContent -Recurse -Force
}

# Create symbolic link from quartz/content to obsidian folder
Write-Host "Creating symbolic link to vault content..." -ForegroundColor Yellow
New-Item -ItemType SymbolicLink -Path $quartzContent -Target $obsidianPath | Out-Null

# Copy config files to quartz directory
Write-Host "Copying configuration files..." -ForegroundColor Yellow
Copy-Item (Join-Path $repoRoot "quartz.config.ts") (Join-Path $quartzPath "quartz.config.ts") -Force

# Build the site
Write-Host "Building static site..." -ForegroundColor Cyan
Set-Location $quartzPath
npx quartz build

# Copy output back to repo root
Write-Host "Copying build output..." -ForegroundColor Yellow
$publicPath = Join-Path $quartzPath "public"
$repoPublic = Join-Path $repoRoot "public"

if (Test-Path $repoPublic) {
    Remove-Item $repoPublic -Recurse -Force
}

Copy-Item $publicPath $repoPublic -Recurse

Set-Location $repoRoot
Write-Host "Build complete! Output is in ./public/" -ForegroundColor Green
