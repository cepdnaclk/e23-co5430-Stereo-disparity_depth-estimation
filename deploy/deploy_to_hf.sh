#!/usr/bin/env bash
# ==============================================================================
# Deploy to Hugging Face Spaces (Linux / macOS / WSL / Git Bash)
#
# CO543 / CO5430 Computer Vision Project
# Group G03 | Project ID: P12 | University of Peradeniya
# ==============================================================================
set -eo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${CYAN}${BOLD}============================================================${NC}"
echo -e "${CYAN}${BOLD}  Hugging Face Spaces Deployment Tool${NC}"
echo -e "${CYAN}  Repository: e23-co5430-Stereo-disparity_depth-estimation${NC}"
echo -e "${CYAN}${BOLD}============================================================${NC}"
echo ""

# 0. Parse arguments
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n|--check)
            DRY_RUN=1
            ;;
        --help|-h)
            echo "Usage: ./deploy/deploy_to_hf.sh [--dry-run]"
            echo ""
            echo "Options:"
            echo "  --dry-run, -n, --check   Verify packaging and tree creation without pushing"
            echo "  --help, -h               Show this help message"
            exit 0
            ;;
    esac
done

# 1. Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# 2. Check git
if ! command -v git &> /dev/null; then
    echo -e "${RED}[ERROR] git is not installed or not found in PATH.${NC}"
    exit 1
fi

# 3. Check repository root files
if [ ! -f "app.py" ] || [ ! -f "README.md" ]; then
    echo -e "${RED}[ERROR] app.py or README.md not found in repository root ($REPO_ROOT).${NC}"
    echo -e "${YELLOW}Please run this script from the repository root or deploy directory:${NC}"
    echo "  ./deploy/deploy_to_hf.sh"
    exit 1
fi

# 4. Build clean isolated deployment commit (NO binary files, NO working-tree changes)
echo -e "${CYAN}[INFO] Assembling clean code package for Hugging Face Spaces...${NC}"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/hf_deploy_XXXXXX")
TMP_INDEX="$TMP_DIR/index"
cleanup() {
    rm -rf "$TMP_DIR" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Fallback object directory if .git/objects is read-only
GIT_EXTRA_ENV=()
if [ ! -w ".git/objects" ]; then
    mkdir -p "$TMP_DIR/objects"
    GIT_EXTRA_ENV=(
        "GIT_OBJECT_DIRECTORY=$TMP_DIR/objects"
        "GIT_ALTERNATE_OBJECT_DIRECTORIES=$(pwd)/.git/objects"
    )
fi

run_git() {
    env GIT_INDEX_FILE="$TMP_INDEX" "${GIT_EXTRA_ENV[@]}" git "$@"
}

# Stage only source code and metadata files into temporary index
run_git add -f app.py requirements.txt README.md
if [ -f "packages.txt" ]; then
    run_git add -f packages.txt
fi

# Stage python source files from src/ (ignoring any binary images in subfolders)
for pyfile in src/*.py; do
    if [ -f "$pyfile" ]; then
        run_git add -f "$pyfile"
    fi
done

# Stage bundled RAFT-Stereo core python files
if [ -d "core" ]; then
    run_git add -f core/*.py core/utils/*.py
fi

# Create tree object from isolated index
TREE_HASH=$(run_git write-tree)

CURRENT_REV=$(git rev-parse --short HEAD 2>/dev/null || echo "local")
COMMIT_MSG="Deploy to Hugging Face Spaces ($(date -u +'%Y-%m-%d %H:%M:%S UTC') from commit $CURRENT_REV)"

# Create a clean root commit (no parent commits, no binary history)
DEPLOY_COMMIT=$(run_git commit-tree "$TREE_HASH" -m "$COMMIT_MSG")

echo -e "${GREEN}[OK] Created clean deployment commit:${NC} $DEPLOY_COMMIT"
echo -e "${CYAN}[INFO] Packaged files:${NC}"
run_git ls-tree --name-only -r "$DEPLOY_COMMIT" | sed 's/^/  - /'
echo ""

if [ "$DRY_RUN" -eq 1 ]; then
    echo -e "${GREEN}${BOLD}============================================================${NC}"
    echo -e "${GREEN}${BOLD}[SUCCESS] Dry run passed! Clean package verified successfully.${NC}"
    echo -e "${GREEN}${BOLD}============================================================${NC}"
    exit 0
fi

# 5. Remote configuration
SPACE_URL=""
if git remote get-url space &> /dev/null; then
    SPACE_URL=$(git remote get-url space)
    # Mask any token for display
    MASKED_URL=$(echo "$SPACE_URL" | sed -E 's/:[^@]+@/:***@/')
    echo -e "${GREEN}[INFO] Existing 'space' remote detected:${NC} $MASKED_URL"
else
    echo -e "${YELLOW}[INFO] 'space' remote not configured.${NC}"
    echo ""
    echo "Please enter your Hugging Face Space details."
    echo "Example: https://huggingface.co/spaces/USERNAME/SPACE-NAME or USERNAME/SPACE-NAME"
    read -r -p "Enter Space URL or username/space-name: " INPUT_TARGET
    if [ -z "$INPUT_TARGET" ]; then
        echo -e "${RED}[ERROR] Space target cannot be empty.${NC}"
        exit 1
    fi

    if [[ "$INPUT_TARGET" =~ ^https?:// ]]; then
        SPACE_URL="$INPUT_TARGET"
    else
        SPACE_URL="https://huggingface.co/spaces/$INPUT_TARGET"
    fi

    git remote add space "$SPACE_URL"
    echo -e "${GREEN}[OK] Added remote 'space' -> $SPACE_URL${NC}"
fi

# 6. Check for write token in remote URL
if [[ ! "$SPACE_URL" =~ :hf_ && ! "$SPACE_URL" =~ :api_ && ! "$SPACE_URL" =~ git@ ]]; then
    echo ""
    echo -e "${YELLOW}------------------------------------------------------------${NC}"
    echo -e "${BOLD}AUTHENTICATION NOTICE:${NC}"
    echo "If git prompts for credentials:"
    echo "  - Username: Your Hugging Face username"
    echo "  - Password: A User Access Token with 'Write' permission"
    echo "  (Generate at: https://huggingface.co/settings/tokens)"
    echo -e "${YELLOW}------------------------------------------------------------${NC}"
    echo ""
fi

# 7. Push directly to Hugging Face Space
echo -e "${CYAN}[INFO] Pushing to Hugging Face Space (main branch)...${NC}"
if git push space "$DEPLOY_COMMIT:refs/heads/main" --force; then
    # Extract clean display URL without token
    CLEAN_URL=$(echo "$SPACE_URL" | sed -E 's/https?:\/\/[^@]+@/https:\/\//')
    echo ""
    echo -e "${GREEN}${BOLD}============================================================${NC}"
    echo -e "${GREEN}${BOLD}[SUCCESS] Deployment completed successfully!${NC}"
    echo -e "${CYAN}Your Space is building and will be live at:${NC}"
    echo -e "  ${BOLD}$CLEAN_URL${NC}"
    echo -e "${GREEN}${BOLD}============================================================${NC}"
else
    echo ""
    echo -e "${RED}[ERROR] Push to Hugging Face failed.${NC}"
    echo "Please check your authentication token and network connection."
    exit 1
fi
