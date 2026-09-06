<#
.SYNOPSIS
    Deploy the Gradio application to Hugging Face Spaces from Windows (PowerShell).

.DESCRIPTION
    This script automates deploying the Stereo Disparity and Depth Estimation
    Gradio application to a Hugging Face Space using Git.
    It checks prerequisites, prompts for Space details or uses existing remotes,
    optionally commits pending changes, and pushes to Hugging Face.

.PARAMETER SpaceTarget
    Optional. The Space URL (e.g. https://huggingface.co/spaces/username/space-name)
    or target in 'username/space-name' format.

.PARAMETER Force
    Optional switch to force push if remote history differs.

.EXAMPLE
    .\deploy_to_hf.ps1
    .\deploy_to_hf.ps1 -SpaceTarget "username/space-name"
    .\deploy_to_hf.ps1 -SpaceTarget "https://huggingface.co/spaces/username/my-stereo-app" -Force
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$SpaceTarget,

    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Write-Step ([string]$msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Cyan
}

function Write-Success ([string]$msg) {
    Write-Host "[SUCCESS] $msg" -ForegroundColor Green
}

function Write-Warn ([string]$msg) {
    Write-Host "[WARNING] $msg" -ForegroundColor Yellow
}

function Write-Err ([string]$msg) {
    Write-Host "[ERROR] $msg" -ForegroundColor Red
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   Hugging Face Spaces Deployment Tool (Windows PowerShell) " -ForegroundColor Cyan
Write-Host "   Repository: e23-co5430-Stereo-disparity_depth-estimation " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Verify git is installed
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Err "Git is not installed or not in PATH."
    Write-Host "Please install Git for Windows: https://git-scm.com/"
    exit 1
}

# 2. Check repository files
if (-not (Test-Path "app.py")) {
    Write-Err "app.py was not found in the current directory ($pwd)."
    Write-Host "Please run this script from the root directory of the repository."
    exit 1
}

if (-not (Test-Path "README.md")) {
    Write-Err "README.md was not found."
    Write-Host "Hugging Face Spaces requires a README.md with YAML metadata frontmatter."
    exit 1
}

# 3. Detect current branch
$currentBranch = (git rev-parse --abbrev-ref HEAD 2>$null)
if (-not $currentBranch) { $currentBranch = "main" } else { $currentBranch = $currentBranch.Trim() }
Write-Step "Current Git branch: $currentBranch"

# 4. Check for uncommitted changes
$statusOutput = (git status --porcelain 2>$null)
if ($statusOutput) {
    Write-Warn "You have uncommitted changes or untracked files:"
    git status -s
    Write-Host ""
    $commitAnswer = Read-Host "Do you want to stage and commit these changes before deploying? [Y/n]"
    if ([string]::IsNullOrWhiteSpace($commitAnswer) -or $commitAnswer -match '^[Yy]') {
        $msg = Read-Host "Enter commit message [Deploy to Hugging Face Spaces]"
        if ([string]::IsNullOrWhiteSpace($msg)) {
            $msg = "Deploy to Hugging Face Spaces"
        }
        git add .
        git commit -m "$msg"
        Write-Success "Changes staged and committed."
    } else {
        Write-Step "Continuing without committing. Only committed changes will be deployed."
    }
}

# 5. Remote configuration
$spaceUrl = ""
$existingRemote = (git remote get-url space 2>$null)
if ($existingRemote) {
    $existingRemote = $existingRemote.Trim()
    if (-not $SpaceTarget) {
        Write-Step "Existing 'space' remote detected: $existingRemote"
        $useExisting = Read-Host "Use this remote? [Y/n]"
        if ([string]::IsNullOrWhiteSpace($useExisting) -or $useExisting -match '^[Yy]') {
            $spaceUrl = $existingRemote
        }
    }
}

if (-not $spaceUrl) {
    if (-not $SpaceTarget) {
        Write-Host ""
        Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
        Write-Host "Please enter your Hugging Face Space details."
        Write-Host "Example: https://huggingface.co/spaces/USERNAME/SPACE-NAME"
        Write-Host "         or simply: USERNAME/SPACE-NAME"
        Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
        $SpaceTarget = Read-Host "Enter Space URL or username/space-name"
    }

    if ([string]::IsNullOrWhiteSpace($SpaceTarget)) {
        Write-Err "Space target cannot be empty."
        exit 1
    }

    $SpaceTarget = $SpaceTarget.Trim()
    if ($SpaceTarget -match "^https://huggingface\.co/spaces/") {
        $spaceUrl = $SpaceTarget
    } else {
        $spaceUrl = "https://huggingface.co/spaces/$SpaceTarget"
    }

    # Add or update git remote 'space'
    $remotes = git remote
    if ($remotes -contains "space") {
        git remote set-url space $spaceUrl
    } else {
        git remote add space $spaceUrl
    }
    Write-Success "Configured remote 'space' -> $spaceUrl"
}

# 6. Authentication guidance
Write-Host ""
Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
Write-Host "AUTHENTICATION TIP:" -ForegroundColor Yellow
Write-Host "When Git prompts for credentials:"
Write-Host "  - Username: Your Hugging Face username"
Write-Host "  - Password: A User Access Token with WRITE permissions"
Write-Host "              (Generate at https://huggingface.co/settings/tokens)"
Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
Write-Host ""

# 7. Push to Hugging Face
Write-Step "Pushing branch '$currentBranch' to Hugging Face Space ($spaceUrl)..."

$pushArgs = @("push", "space", "$($currentBranch):main")
if ($Force) {
    $pushArgs += "--force"
}

$pushSuccess = $false
try {
    & git @pushArgs
    if ($LASTEXITCODE -eq 0) {
        $pushSuccess = $true
    }
} catch {
    $pushSuccess = $false
}

if (-not $pushSuccess) {
    Write-Warn "Standard push was rejected."
    Write-Host "This usually happens when a new Hugging Face Space is initialized with a default file (e.g. README.md)."
    $forceAnswer = Read-Host "Do you want to force-push to overwrite the Space with your local repository? [y/N]"
    if ($forceAnswer -match '^[Yy]') {
        Write-Step "Force-pushing to space..."
        & git push space "$($currentBranch):main" --force
        if ($LASTEXITCODE -eq 0) {
            $pushSuccess = $true
        } else {
            Write-Err "Force push failed. Please verify your write token permissions and connection."
            exit 1
        }
    } else {
        Write-Step "Deployment aborted."
        exit 1
    }
}

if ($pushSuccess) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Success "Deployment completed successfully!"
    Write-Host "Your Space is now building at:" -ForegroundColor Cyan
    Write-Host "  $spaceUrl" -ForegroundColor White -BackgroundColor DarkBlue
    Write-Host "============================================================" -ForegroundColor Green
}
