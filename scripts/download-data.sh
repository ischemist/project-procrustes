#!/opt/homebrew/bin/bash
set -e

BASE_URL="https://files.ischemist.com/retrocast/data"
DATA_DIR="data"

# colors
R='\033[0;31m'
G='\033[0;32m'
Y='\033[1;33m'
B='\033[0;34m'
NC='\033[0m' # No Color

usage() {
    echo -e "${B}RetroCast Data Downloader${NC}"
    echo "usage: $0 [target]"
    echo "targets:"
    echo "  all          - everything"
    echo "  benchmarks   - complete benchmarks (definitions + stocks)"
    echo "  definitions  - just the benchmark definitions (json)"
    echo "  stocks       - just the stock files"
    echo "  raw          - raw model outputs"
    echo "  processed    - standard format routes"
    echo "  scored       - scored routes"
    echo "  results      - summary json files"
    exit 1
}

if [ -z "$1" ]; then
    usage
fi

TARGET="$1"
PATTERN=""

case $TARGET in
    all)         PATTERN="" ;;
    benchmarks)  PATTERN="^1-benchmarks" ;;
    definitions) PATTERN="^1-benchmarks/definitions" ;;
    stocks)      PATTERN="^1-benchmarks/stocks" ;;
    raw)         PATTERN="^2-raw" ;;
    processed)   PATTERN="^3-processed" ;;
    scored)      PATTERN="^4-scored" ;;
    results)     PATTERN="^5-results" ;;
    *)           echo -e "${R}error: unknown target '$TARGET'${NC}"; usage ;;
esac

echo -e "${B}:: initializing retrocast sync ::${NC}"
mkdir -p $DATA_DIR
cd $DATA_DIR

# 1. fetch manifest
echo -n "fetching manifest... "
curl -s -O "$BASE_URL/SHA256SUMS"
echo -e "${G}done${NC}"

# 2. filter files using AWK
if [ -z "$PATTERN" ]; then
    mapfile -t FILES < SHA256SUMS
else
    mapfile -t FILES < <(awk -v pat="$PATTERN" '$2 ~ pat' SHA256SUMS)
fi

TOTAL=${#FILES[@]}

if [ $TOTAL -eq 0 ]; then
    echo -e "${Y}no files found for target: $TARGET${NC}"
    exit 0
fi

echo -e "found ${B}$TOTAL${NC} files for target: ${Y}$TARGET${NC}"
echo "---------------------------------------------------"

CURRENT=0

# 3. iterate
for line in "${FILES[@]}"; do
    ((CURRENT++)) || true
    read -r EXPECTED_HASH FILEPATH <<< "$line" || true
    PFX="[${CURRENT}/${TOTAL}]"

    mkdir -p "$(dirname "$FILEPATH")" || true

    NEEDS_DOWNLOAD=1

    # check local file
    if [ -f "$FILEPATH" ]; then
        printf "${B}%s${NC} checking %-40s" "$PFX" "$(basename "$FILEPATH")"

        CALCULATED_HASH=$(sha256sum "$FILEPATH" | awk '{print $1}')

        if [ "$CALCULATED_HASH" == "$EXPECTED_HASH" ]; then
            NEEDS_DOWNLOAD=0
            # File exists and matches - SKIP DOWNLOAD
            printf "\r${B}%s${NC} %-50s ${G}[OK - SKIP]${NC}   \n" "$PFX" "$FILEPATH"
        else
            # File exists but hash is wrong - RE-DOWNLOAD
            printf "\r${B}%s${NC} %-50s ${Y}[HASH MISMATCH]${NC}\n" "$PFX" "$FILEPATH"
        fi
    fi

    if [ $NEEDS_DOWNLOAD -eq 1 ]; then
        printf "${B}%s${NC} downloading %-40s" "$PFX" "$FILEPATH"

        if curl -f -s -o "$FILEPATH" "$BASE_URL/$FILEPATH"; then
        CALCULATED_HASH=$(sha256sum "$FILEPATH" | awk '{print $1}') || true
             if [ "$CALCULATED_HASH" == "$EXPECTED_HASH" ]; then
                printf "\r${B}%s${NC} %-50s ${G}[DOWNLOADED]${NC} \n" "$PFX" "$FILEPATH"
             else
                printf "\r${B}%s${NC} %-50s ${R}[CORRUPT]${NC}    \n" "$PFX" "$FILEPATH"
                echo "expected: $EXPECTED_HASH"
                echo "got:      $CALCULATED_HASH"
                exit 1
             fi
        else
            printf "\r${B}%s${NC} %-50s ${R}[HTTP ERROR]${NC}\n" "$PFX" "$FILEPATH"
            exit 1
        fi
    fi
done

echo "---------------------------------------------------"
echo -e "${G}sync complete.${NC}"
