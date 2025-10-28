param(
    [string]$SiteTitle,
    [string]$GitHubUser,
    [string]$RepoName,
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"
$scriptDir = $PSScriptRoot
$repoRoot = Split-Path -Parent $scriptDir

Write-Host "`n=== Obsidian + Quartz Digital Garden Setup ===`n" -ForegroundColor Cyan

# Check if already configured
$quartzConfig = Join-Path $repoRoot "quartz.config.ts"
$configContent = Get-Content $quartzConfig -Raw
if ($configContent -notmatch "yourusername.github.io") {
    Write-Host "Site appears to be already configured." -ForegroundColor Yellow
    $reconfigure = Read-Host "Reconfigure? (y/N)"
    if ($reconfigure -ne 'y') { exit 0 }
}

# Collect configuration
if (-not $SiteTitle) {
    $SiteTitle = Read-Host "Site title [My Digital Garden]"
    if ([string]::IsNullOrWhiteSpace($SiteTitle)) { $SiteTitle = "My Digital Garden" }
}

if (-not $GitHubUser) {
    $GitHubUser = Read-Host "GitHub username"
    while ([string]::IsNullOrWhiteSpace($GitHubUser)) {
        Write-Host "GitHub username is required" -ForegroundColor Yellow
        $GitHubUser = Read-Host "GitHub username"
    }
}

if (-not $RepoName) {
    $gitRemote = git remote get-url origin 2>$null
    $detectedRepo = ""
    if ($gitRemote -match "github\.com[:/]([^/]+)/([^/\.]+)") {
        $detectedRepo = $Matches[2]
    }
    if ($detectedRepo) {
        $RepoName = Read-Host "Repository name [$detectedRepo]"
        if ([string]::IsNullOrWhiteSpace($RepoName)) { $RepoName = $detectedRepo }
    } else {
        $RepoName = Read-Host "Repository name"
        while ([string]::IsNullOrWhiteSpace($RepoName)) {
            Write-Host "Repository name is required" -ForegroundColor Yellow
            $RepoName = Read-Host "Repository name"
        }
    }
}

$baseUrl = "$GitHubUser.github.io/$RepoName"

Write-Host "`nConfiguration Summary:" -ForegroundColor Green
Write-Host "  Site Title: $SiteTitle"
Write-Host "  Base URL:   https://$baseUrl/"
Write-Host ""

$confirm = Read-Host "Proceed? (Y/n)"
if ($confirm -eq 'n') { exit 0 }

Write-Host "`nUpdating configuration files..." -ForegroundColor Cyan

# Update quartz.config.ts
$configContent = $configContent -replace 'pageTitle: ".*?"', "pageTitle: `"$SiteTitle`""
$configContent = $configContent -replace 'baseUrl: ".*?"', "baseUrl: `"$baseUrl`""
Set-Content -Path $quartzConfig -Value $configContent -NoNewline

# Update index.md
$indexPath = Join-Path $repoRoot "obsidian\index.md"
$indexContent = @"
---
title: Welcome to $SiteTitle
---

# Welcome! 🌱

This is my personal digital garden built with Obsidian and [Quartz](https://quartz.jzhao.xyz/).

## Getting Started

Add your Obsidian notes to the ``obsidian/`` folder and they will automatically be published.

Happy gardening! ✨
"@
Set-Content -Path $indexPath -Value $indexContent

# Update README
$readmePath = Join-Path $repoRoot "README.md"
$readmeContent = Get-Content $readmePath -Raw
$readmeContent = $readmeContent -replace 'yourusername', $GitHubUser
$readmeContent = $readmeContent -replace 'your-repo-name', $RepoName
Set-Content -Path $readmePath -Value $readmeContent -NoNewline

Write-Host "`n✨ Setup Complete!" -ForegroundColor Green
Write-Host "`nNext steps:"
Write-Host "  1. Add notes to obsidian/"
Write-Host "  2. Test: npm run serve"
Write-Host "  3. Push: git add . && git commit && git push"
Write-Host "  4. Enable GitHub Pages at Settings -> Pages -> Source: GitHub Actions"
Write-Host "  5. Visit: https://$baseUrl/`n"
