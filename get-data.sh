#!/bin/bash
set -eo pipefail

VERSION="2026.1.1"
LAST_UPDATED="Jan 2026"

main() {
    BASE_URL="https://files.ischemist.com/retrocast/data"
    DATA_DIR="${RETROCAST_DATA_DIR:-data/retrocast}"

    # --- colors ---
    R='\033[0;31m'
    G='\033[0;32m'
    Y='\033[1;33m'
    B='\033[0;34m'
    NC='\033[0m'

    usage() {
        echo ""
        echo -e "${B}usage:${NC}"
        if [ -t 0 ]; then
            echo "  $0 [target] [flags]"
            echo "  example: $0 mkt-lin-500 --dry-run"
        else
            echo "  ... | bash -s -- [target] [flags]"
            echo "  example: curl ... | bash -s -- mkt-lin-500"
        fi
        echo ""
        echo -e "${B}flags:${NC}"
        echo "  --dir=PATH  override download directory (default: data/retrocast)"
        echo "  --dry-run   show what would be downloaded without doing it"
        echo "  -V          show version"
        echo "  -h          help"
        exit 1
    }

    show_version() {
        echo "retrocast data sync v${VERSION} (${LAST_UPDATED})"
        exit 0
    }

    show_sizes_and_usage() {
        echo -e "${B}RetroCast Data Downloader${NC}"
        echo "Available data targets and estimated sizes:"
        printf "  ${Y}%-15s %10s${NC}\n" "Target" "Size"
        printf "  ${Y}%-15s %10s${NC}\n" "----------------" "----------"
        printf "  %-17s %10s\n" "all" "~554M"
        printf "  %-17s %10s\n" "benchmarks" "79M"
        printf "    %-15s %10s\n" "definitions" "26M"
        printf "    %-15s %10s\n" "stocks" "53M"
        printf "  %-17s %10s\n" "raw" "352M"
        printf "  %-17s %10s\n" "processed" "118M"
        printf "  %-17s %10s\n" "scored" "3.0M"
        printf "  %-17s %10s\n" "results" "1.6M"
        usage
    }


    # --- arg parsing ---
    TARGET=""
    DRY_RUN=0

    while [ "$#" -gt 0 ]; do
        case "$1" in
            -h|--help) usage ;;
            -V|--version) show_version ;;
            --dry-run) DRY_RUN=1; shift ;;
            --dir=*) DATA_DIR="${1#*=}"; shift ;;
            -*)
                echo -e "${R}error: unknown option: $1${NC}" >&2
                usage
                ;;
            *)
                if [ -z "$TARGET" ]; then
                    TARGET="$1"
                else
                    echo -e "${R}error: multiple targets specified ('$TARGET' and '$1')${NC}" >&2
                    usage
                fi
                shift
                ;;
        esac
    done

    if [ -z "$TARGET" ]; then
        show_sizes_and_usage
    fi

    # --- dependency checks ---
    for cmd in curl awk; do
        if ! command -v $cmd &> /dev/null; then
            echo -e "${R}error: '$cmd' is not installed. please install it to continue.${NC}" >&2
            exit 1
        fi
    done

    # --- portable sha256 command ---
    if command -v sha256sum &> /dev/null; then
        SHACMD="sha256sum"
    elif command -v shasum &> /dev/null; then
        SHACMD="shasum -a 256"
    else
        echo -e "${R}error: could not find a sha256 command (sha256sum or shasum).${NC}" >&2
        exit 1
    fi

    PATTERN=""

    case $TARGET in
        help|-h|--help) show_sizes_and_usage ;;

        # --- groups ---
        all)         PATTERN="." ;; # Match everything
        benchmarks)  PATTERN="^1-benchmarks" ;;
        definitions) PATTERN="^1-benchmarks/definitions" ;;
        stocks)      PATTERN="^1-benchmarks/stocks" ;;

        # --- specific benchmark bundles (hardcoded dependencies) ---
        # anchored regex: matches lines starting with 1-benchmarks
        # AND containing either the benchmark name OR the required stock
        mkt-lin-500) PATTERN="^1-benchmarks/.*(mkt-lin-500|buyables-stock)" ;;
        mkt-cnv-160) PATTERN="^1-benchmarks/.*(mkt-cnv-160|buyables-stock)" ;;

        ref-lin-600) PATTERN="^1-benchmarks/.*(ref-lin-600|n5-stock)" ;;
        ref-cnv-400) PATTERN="^1-benchmarks/.*(ref-cnv-400|n5-stock)" ;;
        ref-lng-84)  PATTERN="^1-benchmarks/.*(ref-lng-84|n1-n5-stock)" ;;

        # --- bulk data ---
        raw)         PATTERN="^2-raw" ;;
        processed)   PATTERN="^3-processed" ;;
        scored)      PATTERN="^4-scored" ;;
        results)     PATTERN="^5-results" ;;
        *)           echo -e "${R}error: unknown target '$TARGET'${NC}"; usage ;;
    esac

    echo -e "${B}:: initializing retrocast sync ::${NC}"
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"

    # always fetch manifest first (it's small)
    echo -n "fetching manifest... "
    if ! curl -sfSL -o SHA256SUMS "$BASE_URL/SHA256SUMS"; then
        echo -e "${R}failed to download manifest. check url or connection.${NC}" >&2
        exit 1
    fi
    echo -e "${G}done${NC}"

    MANIFEST_FILTERED=$(mktemp)
    awk -v pat="$PATTERN" '$2 ~ pat' SHA256SUMS > "$MANIFEST_FILTERED"
    trap 'rm -f "$MANIFEST_FILTERED"' EXIT

    TOTAL=$(wc -l < "$MANIFEST_FILTERED" | awk '{print $1}')

    if [ "$TOTAL" -eq 0 ]; then
        echo -e "${Y}no files found for target: $TARGET${NC}"
        exit 0
    fi

    echo -e "found ${B}$TOTAL${NC} files for target: ${Y}$TARGET${NC}"
    echo "---------------------------------------------------"

    CURRENT=0
    while read -r EXPECTED_HASH FILEPATH; do
        CURRENT=$((CURRENT + 1))
        PFX="[${CURRENT}/${TOTAL}]"

        if [ "$DRY_RUN" -eq 1 ]; then
            printf "${B}%s${NC} [DRY-RUN] would download %-40s\n" "$PFX" "$FILEPATH"
            continue
        fi

        mkdir -p "$(dirname "$FILEPATH")"

        if [ -f "$FILEPATH" ]; then
            printf "${B}%s${NC} checking %-40s" "$PFX" "$(basename "$FILEPATH")"
            CALCULATED_HASH=$($SHACMD "$FILEPATH" | awk '{print $1}')

            if [ "$CALCULATED_HASH" = "$EXPECTED_HASH" ]; then
                printf "\r${B}%s${NC} %-50s ${G}[VERIFIED - LOCAL]${NC}\n" "$PFX" "$FILEPATH"
                continue
            else
                printf "\r${B}%s${NC} %-50s ${Y}[HASH MISMATCH]${NC}   \n" "$PFX" "$FILEPATH"
            fi
        fi

        printf "${B}%s${NC} downloading %-40s" "$PFX" "$FILEPATH"
        if curl -f# -sL -o "$FILEPATH" "$BASE_URL/$FILEPATH"; then
            CALCULATED_HASH=$($SHACMD "$FILEPATH" | awk '{print $1}')
            if [ "$CALCULATED_HASH" = "$EXPECTED_HASH" ]; then
                printf "\r${B}%s${NC} %-50s ${G}[VERIFIED]${NC}          \n" "$PFX" "$FILEPATH"
            else
                printf "\r${B}%s${NC} %-50s ${R}[CORRUPT]${NC}           \n" "$PFX" "$FILEPATH"
                echo "expected: $EXPECTED_HASH"
                echo "got:      $CALCULATED_HASH"
                exit 1
            fi
        else
            printf "\r${B}%s${NC} %-50s ${R}[HTTP ERROR]${NC}        \n" "$PFX" "$FILEPATH"
            exit 1
        fi
    done < "$MANIFEST_FILTERED"

    echo "---------------------------------------------------"
    echo -e "${G}sync complete.${NC}"
    echo -e "files located in: ${B}$(pwd)${NC}"
}

main "$@"
