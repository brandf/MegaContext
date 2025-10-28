<#
master-vault.ps1 (Windows-friendly newlines + array-safe appends)
Creates/updates symlinks in Projects folder pointing to ..\<Project>\obsidian
Keeps list in ./master-vault.txt (one name per line, '#' comments ok)

Usage:
  .\master-vault.ps1 [-Sync] [-NoPrune] [-DryRun]
  .\master-vault.ps1 -Add <ProjectName>
  .\master-vault.ps1 -Remove <ProjectName>
  .\master-vault.ps1 -List
#>

param(
  [string] $Add,
  [string] $Remove,
  [switch] $List,
  [switch] $Sync,
  [switch] $NoPrune,
  [switch] $DryRun,
  [string] $ProjectsDir  # Optional: override default Projects directory
)

# ---------- Paths (all relative to this script) ----------
$ScriptDir        = $PSScriptRoot
$RepoRoot         = Split-Path -Parent $ScriptDir
$GitHubRoot       = Split-Path -Parent $RepoRoot

# Allow override of ProjectsDir, default to ..\obsidian\🚧 Projects
if (-not $ProjectsDir) {
  $ProjectsDir = Join-Path $RepoRoot "obsidian\🚧 Projects"
}

$ListPath         = Join-Path $ScriptDir 'master-vault.txt'
$obsidianDir      = 'obsidian'   # subfolder of each project

# ---------- Helper: read lines from projects.txt ----------
function Get-ProjectList {
  if (Test-Path $ListPath) {
    (Get-Content $ListPath) |
      Where-Object {
        $line = $_.Trim()
        ($line -ne '') -and ($line -notmatch '^#')
      } |
      ForEach-Object { $_.Trim() }
  } else {
    @()
  }
}

# ---------- Helper: write lines back to projects.txt ----------
function Write-ProjectList {
  param([string[]]$Lines)
  # Force Windows newlines by converting to single string & Out-File with ASCII
  $text = ($Lines -join "`r`n")
  $text | Out-File -FilePath $ListPath -Encoding ASCII -NoNewline
  # Add final newline
  "`r`n" | Out-File -FilePath $ListPath -Encoding ASCII -Append -NoNewline
}

# ---------- Command: Add ----------
if ($Add) {
  $projectPath = Join-Path $GitHubRoot $Add
  if (-not (Test-Path $projectPath)) {
    Write-Error "Project folder does not exist: $projectPath"
    exit 1
  }
  $obsPath = Join-Path $projectPath $obsidianDir
  if (-not (Test-Path $obsPath)) {
    Write-Error "No '$obsidianDir' subfolder found: $obsPath"
    exit 1
  }

  $existing = Get-ProjectList
  if ($existing -notcontains $Add) {
    $newList = @($existing) + @($Add)
    Write-ProjectList $newList
    Write-Host "Added '$Add' to projects list." -ForegroundColor Green
  } else {
    Write-Host "'$Add' is already in the projects list." -ForegroundColor Yellow
  }
  exit
}

# ---------- Command: Remove ----------
if ($Remove) {
  $existing = Get-ProjectList
  if ($existing -contains $Remove) {
    $newList = $existing | Where-Object { $_ -ne $Remove }
    Write-ProjectList $newList

    $linkPath = Join-Path $ProjectsDir $Remove
    if (Test-Path $linkPath) {
      if ($DryRun) {
        Write-Host "[DryRun] Would remove symlink: $linkPath" -ForegroundColor Cyan
      } else {
        Remove-Item -Path $linkPath -Force -Recurse -ErrorAction Stop
        Write-Host "Removed symlink: $linkPath" -ForegroundColor Magenta
      }
    }

    Write-Host "Removed '$Remove' from projects list." -ForegroundColor Green
  } else {
    Write-Host "'$Remove' was not in the projects list." -ForegroundColor Yellow
  }
  exit
}

# ---------- Command: List ----------
if ($List) {
  $projects = Get-ProjectList
  if ($projects.Count -eq 0) {
    Write-Host "No projects in list (file: $ListPath)" -ForegroundColor Gray
  } else {
    Write-Host "Projects in list ($($projects.Count)):" -ForegroundColor Cyan
    $projects | ForEach-Object { Write-Host "  $_" -ForegroundColor White }
  }
  exit
}

# ---------- Internal helper: ensure a symlink exists & is correct ----------
function Sync-OneSymlink {
  param(
    [string]$LinkName,
    [string]$TargetPath
  )

  $lnk = Join-Path $ProjectsDir $LinkName

  # If target doesn't exist, skip
  if (-not (Test-Path $TargetPath)) {
    if ($DryRun) {
      Write-Host "[DryRun] Target missing for '$LinkName', would skip." -ForegroundColor DarkGray
    } else {
      Write-Warning "Target missing for '$LinkName': $TargetPath"
    }
    return $false
  }

  # Check if link is already correct
  if (Test-Path $lnk) {
    $item = Get-Item -Path $lnk -Force
    if ($item.LinkType -eq 'SymbolicLink') {
      $existing = (Get-Item -Path $lnk -ErrorAction SilentlyContinue)
      $existingTarget = if ($existing) {
        $targetItem = Get-Item -LiteralPath $existing.Target -ErrorAction SilentlyContinue
        if ($targetItem) { $targetItem.FullName }
      }
      $resolvedItem = Resolve-Path -LiteralPath $TargetPath -ErrorAction SilentlyContinue
      $resolved = if ($resolvedItem) { $resolvedItem.Path }

      if ($existingTarget -eq $resolved) {
        if ($DryRun) {
          Write-Host "[DryRun] Symlink OK: $LinkName" -ForegroundColor Green
        } else {
          Write-Host "Symlink OK: $LinkName" -ForegroundColor DarkGreen
        }
        return $true
      } else {
        if ($DryRun) {
          Write-Host "[DryRun] Would update symlink: $LinkName" -ForegroundColor Cyan
        } else {
          Remove-Item -Path $lnk -Force -Recurse -ErrorAction Stop
          Write-Host "Removed old symlink: $LinkName" -ForegroundColor Yellow
        }
      }
    } else {
      if ($DryRun) {
        Write-Host "[DryRun] Would remove non-symlink item: $LinkName" -ForegroundColor Cyan
      } else {
        Remove-Item -Path $lnk -Force -Recurse -ErrorAction Stop
        Write-Host "Removed non-symlink item: $LinkName" -ForegroundColor Yellow
      }
    }
  }

  # Create new symlink
  if ($DryRun) {
    Write-Host "[DryRun] Would create symlink: $LinkName -> $TargetPath" -ForegroundColor Cyan
  } else {
    New-Item -ItemType SymbolicLink -Path $lnk -Value $TargetPath -Force | Out-Null
    Write-Host "Created symlink: $LinkName" -ForegroundColor Green
  }
  return $true
}

# ---------- Command: Sync (default if no other command given) ----------
# Default to sync if no flags other than -List
if (-not $PSBoundParameters.ContainsKey('Sync') -and -not $Add -and -not $Remove -and -not $List) { $Sync = $true }

if ($Sync) {
  if ($DryRun) {
    Write-Host "=== Dry Run Mode ===" -ForegroundColor Cyan
  }

  # Ensure the Projects directory exists
  if (-not (Test-Path $ProjectsDir)) {
    if ($DryRun) {
      Write-Host "[DryRun] Would create directory: $ProjectsDir" -ForegroundColor Cyan
    } else {
      New-Item -ItemType Directory -Path $ProjectsDir -Force | Out-Null
      Write-Host "Created directory: $ProjectsDir" -ForegroundColor Green
    }
  }

  $projectNames = Get-ProjectList
  if ($projectNames.Count -eq 0) {
    Write-Host "No projects listed in $ListPath" -ForegroundColor Gray
    exit
  }

  $synced = @()
  foreach ($pName in $projectNames) {
    $target = Join-Path (Join-Path $GitHubRoot $pName) $obsidianDir
    $ok = Sync-OneSymlink -LinkName $pName -TargetPath $target
    if ($ok) {
      $synced += $pName
    }
  }

  # Prune orphaned symlinks (unless -NoPrune)
  if (-not $NoPrune) {
    if (Test-Path $ProjectsDir) {
      Get-ChildItem -Path $ProjectsDir -Force | ForEach-Object {
        if ($_.LinkType -eq 'SymbolicLink') {
          if ($synced -notcontains $_.Name) {
            if ($DryRun) {
              Write-Host "[DryRun] Would prune orphaned symlink: $($_.Name)" -ForegroundColor Magenta
            } else {
              Remove-Item -Path $_.FullName -Force -Recurse -ErrorAction Stop
              Write-Host "Pruned orphaned symlink: $($_.Name)" -ForegroundColor Magenta
            }
          }
        }
      }
    }
  }

  if ($DryRun) {
    Write-Host "=== Dry Run Complete ===" -ForegroundColor Cyan
  } else {
    Write-Host "Sync complete." -ForegroundColor Green
  }
}
