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

# 1. Check git
if ! command -v git &> /dev/null; then
    echo -e "${RED}[ERROR] git is not installed or not found in PATH.${NC}"
    exit 1
fi

# 2. Check repository root
if [ ! -f "app.py" ] || [ ! -f "README.md" ]; then
    echo -e "${RED}[ERROR] app.py or README.md not found.${NC}"
    echo -e "${YELLOW}Please run this script from the repository root directory:${NC}"
    echo "  cd ~/projects/e23-co5430-Stereo-disparity_depth-estimation"
    echo "  ./deploy_to_hf.sh"
    exit 1
fi

# 3. Remote configuration
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

# 4. Check for write token in remote URL
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

# 5. Build clean isolated deployment commit (NO binary files, NO working-tree changes)
echo -e "${CYAN}[INFO] Assembling clean code package for Hugging Face Spaces...${NC}"

TMP_INDEX=$(mktemp -u "${TMPDIR:-/tmp}/hf_index_XXXXXX")
cleanup() {
    rm -f "$TMP_INDEX" "$TMP_INDEX.lock" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Stage only source code and metadata files into temporary index
GIT_INDEX_FILE="$TMP_INDEX" git add -f app.py requirements.txt README.md
if [ -f "packages.txt" ]; then
    GIT_INDEX_FILE="$TMP_INDEX" git add -f packages.txt
fi

# Stage python source files from src/ (ignoring any binary images in subfolders)
for pyfile in src/*.py; do
    if [ -f "$pyfile" ]; then
        GIT_INDEX_FILE="$TMP_INDEX" git add -f "$pyfile"
    fi
done

# Stage bundled RAFT-Stereo core python files
if [ -d "core" ]; then
    GIT_INDEX_FILE="$TMP_INDEX" git add -f core/*.py core/utils/*.py
fi

# Create tree object from isolated index
TREE_HASH=$(GIT_INDEX_FILE="$TMP_INDEX" git write-tree)

CURRENT_REV=$(git rev-parse --short HEAD 2>/dev/null || echo "local")
COMMIT_MSG="Deploy to Hugging Face Spaces ($(date -u +'%Y-%m-%d %H:%M:%S UTC') from commit $CURRENT_REV)"

# Create a clean root commit (no parent commits, no binary history)
DEPLOY_COMMIT=$(git commit-tree "$TREE_HASH" -m "$COMMIT_MSG")

echo -e "${GREEN}[OK] Created clean deployment commit:${NC} $DEPLOY_COMMIT"
echo -e "${CYAN}[INFO] Packaged files:${NC}"
git ls-tree --name-only -r "$DEPLOY_COMMIT" | sed 's/^/  - /'
echo ""

# 6. Push directly to Hugging Face Space
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
