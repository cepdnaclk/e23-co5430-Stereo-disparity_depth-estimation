#!/usr/bin/env bash
# Deploy to Hugging Face Spaces (Linux / macOS / WSL / Git Bash)
set -e

echo "============================================================"
echo "  Hugging Face Spaces Deployment Tool"
echo "  Repository: e23-co5430-Stereo-disparity_depth-estimation"
echo "============================================================"
echo ""

# 1. Check git
if ! command -v git &> /dev/null; then
    echo "[ERROR] git is not installed or not in PATH."
    exit 1
fi

# 2. Check repository files
if [ ! -f "app.py" ]; then
    echo "[ERROR] app.py not found. Please run this script from the repository root."
    exit 1
fi

if [ ! -f "README.md" ]; then
    echo "[ERROR] README.md not found. Hugging Face Spaces requires README.md with YAML metadata."
    exit 1
fi

# 3. Detect branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
echo "[INFO] Current Git branch: $CURRENT_BRANCH"

# 4. Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo ""
    echo "[WARNING] You have uncommitted changes or untracked files:"
    git status -s
    echo ""
    read -r -p "Do you want to stage and commit these changes before deploying? [Y/n]: " COMMIT_CHOICE
    COMMIT_CHOICE=${COMMIT_CHOICE:-y}
    if [[ "$COMMIT_CHOICE" =~ ^[Yy]$ ]]; then
        read -r -p "Enter commit message [Deploy to Hugging Face Spaces]: " COMMIT_MSG
        COMMIT_MSG=${COMMIT_MSG:-Deploy to Hugging Face Spaces}
        git add .
        git commit -m "$COMMIT_MSG"
    else
        echo "[INFO] Proceeding without committing changes. Only committed files will be deployed."
    fi
fi

# 5. Remote configuration
SPACE_URL=""
if git remote get-url space &> /dev/null; then
    SPACE_URL=$(git remote get-url space)
    echo "[INFO] Existing 'space' remote found: $SPACE_URL"
    read -r -p "Use this remote? [Y/n]: " USE_EXISTING
    USE_EXISTING=${USE_EXISTING:-y}
    if [[ ! "$USE_EXISTING" =~ ^[Yy]$ ]]; then
        SPACE_URL=""
    fi
fi

if [ -z "$SPACE_URL" ]; then
    echo ""
    echo "Please enter your Hugging Face Space details."
    echo "Example: https://huggingface.co/spaces/USERNAME/SPACE-NAME or USERNAME/SPACE-NAME"
    read -r -p "Enter Space URL or username/space-name: " INPUT_TARGET
    if [ -z "$INPUT_TARGET" ]; then
        echo "[ERROR] Space target cannot be empty."
        exit 1
    fi

    if [[ "$INPUT_TARGET" =~ ^https://huggingface.co/spaces/ ]]; then
        SPACE_URL="$INPUT_TARGET"
    else
        SPACE_URL="https://huggingface.co/spaces/$INPUT_TARGET"
    fi

    if git remote | grep -q "^space$"; then
        git remote set-url space "$SPACE_URL"
    else
        git remote add space "$SPACE_URL"
    fi
    echo "[OK] Configured remote 'space' -> $SPACE_URL"
fi

echo ""
echo "------------------------------------------------------------"
echo "AUTHENTICATION NOTICE:"
echo "When Git prompts for credentials:"
echo "  - Username: Your Hugging Face username"
echo "  - Password: A User Access Token (Write role)"
echo "  Generate one at: https://huggingface.co/settings/tokens"
echo "------------------------------------------------------------"
echo ""

# 6. Push to Hugging Face Space
echo "[INFO] Pushing $CURRENT_BRANCH to Hugging Face Space (main)..."
if ! git push space "$CURRENT_BRANCH:main"; then
    echo ""
    echo "[WARNING] Normal push failed."
    echo "This often happens if the Space was created with default files on Hugging Face."
    read -r -p "Would you like to force-push to overwrite the Space? [y/N]: " FORCE_PUSH
    FORCE_PUSH=${FORCE_PUSH:-n}
    if [[ "$FORCE_PUSH" =~ ^[Yy]$ ]]; then
        echo "[INFO] Force-pushing to space..."
        git push space "$CURRENT_BRANCH:main" --force
    else
        echo "[INFO] Deployment aborted."
        exit 1
    fi
fi

echo ""
echo "============================================================"
echo "[SUCCESS] Deployment completed successfully!"
echo "Your Space will be built and hosted at:"
echo "$SPACE_URL"
echo "============================================================"
