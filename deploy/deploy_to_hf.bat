@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Hugging Face Spaces Deployment Tool (Windows)
echo   Repository: e23-co5430-Stereo-disparity_depth-estimation
echo ============================================================
echo.

:: 1. Navigate to repository root (parent directory of this script)
cd /d "%~dp0.."

:: 2. Verify Git installation
where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git was not found in your PATH.
    echo Please install Git for Windows from https://git-scm.com/ and try again.
    pause
    exit /b 1
)

:: 3. Verify repository root files
if not exist "app.py" (
    echo [ERROR] app.py not found in repository root.
    echo Please ensure this script is in the deploy\ directory of the repository:
    echo   deploy\deploy_to_hf.bat
    pause
    exit /b 1
)

if not exist "README.md" (
    echo [ERROR] README.md not found.
    echo Hugging Face Spaces requires a README.md with YAML metadata frontmatter.
    pause
    exit /b 1
)

:: 3. Detect current branch
set "CURRENT_BRANCH="
for /f "tokens=*" %%i in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CURRENT_BRANCH=%%i"
if "%CURRENT_BRANCH%"=="" set "CURRENT_BRANCH=main"
echo [INFO] Current Git branch: %CURRENT_BRANCH%
echo.

:: 4. Check for uncommitted changes
set "HAS_CHANGES="
for /f "tokens=*" %%i in ('git status --porcelain 2^>nul') do set "HAS_CHANGES=1"

if defined HAS_CHANGES (
    echo [WARNING] You have uncommitted changes or untracked files:
    echo.
    git status -s
    echo.
    set "COMMIT_CHOICE="
    set /p "COMMIT_CHOICE=Do you want to stage and commit these changes before deploying? [Y/n]: "
    if "!COMMIT_CHOICE!"=="" set "COMMIT_CHOICE=y"
    if /i "!COMMIT_CHOICE!"=="y" (
        set "COMMIT_MSG="
        set /p "COMMIT_MSG=Enter commit message [Deploy to Hugging Face Spaces]: "
        if "!COMMIT_MSG!"=="" set "COMMIT_MSG=Deploy to Hugging Face Spaces"
        git add .
        git commit -m "!COMMIT_MSG!"
        echo [OK] Changes committed.
        echo.
    ) else (
        echo [INFO] Proceeding without committing. Note that uncommitted changes will NOT be deployed.
        echo.
    )
)

:: 5. Hugging Face Space Remote Setup
set "SPACE_URL="
for /f "tokens=*" %%i in ('git remote get-url space 2^>nul') do set "SPACE_URL=%%i"

if defined SPACE_URL (
    echo [INFO] Found existing 'space' remote: !SPACE_URL!
    set "USE_EXISTING="
    set /p "USE_EXISTING=Use this remote? [Y/n]: "
    if "!USE_EXISTING!"=="" set "USE_EXISTING=y"
    if /i "!USE_EXISTING!"=="n" (
        set "SPACE_URL="
    )
)

if not defined SPACE_URL (
    echo.
    echo ------------------------------------------------------------
    echo Setting up Hugging Face Space Remote:
    echo 1. Create a Space at: https://huggingface.co/new-space
    echo    - Select 'Gradio' as the Space SDK
    echo 2. Enter your details below:
    echo ------------------------------------------------------------
    echo.
    set "HF_TARGET="
    set /p "HF_TARGET=Enter Space URL (e.g. https://huggingface.co/spaces/username/space-name) or 'username/space-name': "
    if "!HF_TARGET!"=="" (
        echo [ERROR] Space target cannot be empty.
        pause
        exit /b 1
    )

    :: Check if target starts with https://
    echo !HF_TARGET! | findstr /i "^https://huggingface.co/spaces/" >nul
    if !errorlevel! equ 0 (
        set "SPACE_URL=!HF_TARGET!"
    ) else (
        set "SPACE_URL=https://huggingface.co/spaces/!HF_TARGET!"
    )

    :: Check if 'space' remote exists
    git remote | findstr /x "space" >nul
    if !errorlevel! equ 0 (
        git remote set-url space !SPACE_URL!
    ) else (
        git remote add space !SPACE_URL!
    )
    echo [OK] Configured remote 'space' -^> !SPACE_URL!
)

echo.
echo ------------------------------------------------------------
echo AUTHENTICATION TIP:
echo When prompted for Git credentials:
echo   Username: Your Hugging Face username
echo   Password: A User Access Token with WRITE permissions
echo             (Generate at https://huggingface.co/settings/tokens)
echo ------------------------------------------------------------
echo.

:: 6. Push to Hugging Face Space
echo [INFO] Pushing '%CURRENT_BRANCH%' to Hugging Face Space 'main'...
echo.
git push space %CURRENT_BRANCH%:main

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Standard push was rejected.
    echo This is normal if the Space on Hugging Face was created with default files
    echo (such as a template README.md) that do not match your local history.
    echo.
    set "FORCE_CHOICE="
    set /p "FORCE_CHOICE=Do you want to force-push to overwrite the Space with this repository? [y/N]: "
    if /i "!FORCE_CHOICE!"=="y" (
        echo [INFO] Force-pushing to Space...
        git push space %CURRENT_BRANCH%:main --force
        if !errorlevel! neq 0 (
            echo.
            echo [ERROR] Force-push failed. Please verify your token and network connection.
            pause
            exit /b 1
        )
    ) else (
        echo [INFO] Deployment aborted.
        pause
        exit /b 1
    )
)

echo.
echo ============================================================
echo [SUCCESS] Deployment completed!
echo Your Hugging Face Space is building at:
echo !SPACE_URL!
echo ============================================================
echo.
pause
exit /b 0
