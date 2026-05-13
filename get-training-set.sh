#!/bin/bash
set -eo pipefail

VERSION="2026.05.12"

main() {
    BASE_URL="${RETROCAST_TRAINING_SET_BASE_URL:-https://files.ischemist.com/retrocast/training-sets}"
    CACHE_DIR="${RETROCAST_TRAINING_SET_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/retrocast/training-sets}"
    OUTPUT_DIR=""
    DATASET="paroutes"
    ARTIFACT=""
    SPLIT="training"
    FORMAT="jsonl"
    RELEASE="latest"
    DRY_RUN=0

    R='\033[0;31m'
    B='\033[0;34m'
    NC='\033[0m'

    usage() {
        echo ""
        echo -e "${B}usage:${NC}"
        if [ -t 0 ]; then
            echo "  $0 <artifact> [flags]"
        else
            echo "  ... | bash -s -- <artifact> [flags]"
        fi
        echo ""
        echo -e "${B}artifacts:${NC}"
        echo "  route-holdout-n1-n5"
        echo "  reaction-holdout-n1-n5"
        echo "  single-step-reaction-holdout-n1-n5"
        echo ""
        echo -e "${B}flags:${NC}"
        echo "  --split=all|training|validation"
        echo "  --format=jsonl|rsmi"
        echo "  --release=latest|vYYYY-MM-DD"
        echo "  --dataset=paroutes"
        echo "  --dir=PATH  materialize into an explicit project-owned directory"
        echo "  --dry-run"
        echo "  -V, --version"
        echo "  -h, --help"
        exit 1
    }

    show_version() {
        echo "retrocast training-set downloader v${VERSION}"
        exit 0
    }

    while [ "$#" -gt 0 ]; do
        case "$1" in
            -h|--help) usage ;;
            -V|--version) show_version ;;
            --split=*) SPLIT="${1#*=}"; shift ;;
            --format=*) FORMAT="${1#*=}"; shift ;;
            --release=*) RELEASE="${1#*=}"; shift ;;
            --dataset=*) DATASET="${1#*=}"; shift ;;
            --dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
            --dry-run) DRY_RUN=1; shift ;;
            -*)
                echo -e "${R}error: unknown option: $1${NC}" >&2
                usage
                ;;
            *)
                if [ -z "$ARTIFACT" ]; then
                    ARTIFACT="$1"
                else
                    echo -e "${R}error: multiple artifacts specified${NC}" >&2
                    usage
                fi
                shift
                ;;
        esac
    done

    [ -n "$ARTIFACT" ] || usage

    for cmd in curl awk; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            echo -e "${R}error: '$cmd' is not installed.${NC}" >&2
            exit 1
        fi
    done

    if command -v sha256sum >/dev/null 2>&1; then
        SHACMD="sha256sum"
    elif command -v shasum >/dev/null 2>&1; then
        SHACMD="shasum -a 256"
    else
        echo -e "${R}error: sha256sum or shasum is required.${NC}" >&2
        exit 1
    fi

    case "$DATASET" in
        paroutes) ;;
        *)
            echo -e "${R}error: unsupported dataset '$DATASET'${NC}" >&2
            exit 1
            ;;
    esac

    case "$ARTIFACT" in
        route-holdout-n1-n5|reaction-holdout-n1-n5|single-step-reaction-holdout-n1-n5) ;;
        *)
            echo -e "${R}error: unsupported artifact '$ARTIFACT'${NC}" >&2
            exit 1
            ;;
    esac

    case "$SPLIT" in
        all|training|validation) ;;
        *)
            echo -e "${R}error: unsupported split '$SPLIT'${NC}" >&2
            exit 1
            ;;
    esac

    FILENAME="$(resolve_filename "$ARTIFACT" "$SPLIT" "$FORMAT")"
    RESOLVED_RELEASE="$RELEASE"
    if [ "$RELEASE" = "latest" ]; then
        RESOLVED_RELEASE="$(resolve_latest_release "$BASE_URL" "$DATASET")"
    fi

    ROOT_DIR="$CACHE_DIR"
    if [ -n "$OUTPUT_DIR" ]; then
        ROOT_DIR="$OUTPUT_DIR"
    fi

    if [ -n "$OUTPUT_DIR" ]; then
        RELEASE_DIR="$ROOT_DIR/$RESOLVED_RELEASE"
    else
        RELEASE_DIR="$ROOT_DIR/$DATASET/$RESOLVED_RELEASE"
    fi
    LOCAL_DIR="$RELEASE_DIR/$ARTIFACT"
    LOCAL_PATH="$LOCAL_DIR/$FILENAME"
    MANIFEST_PATH="$LOCAL_DIR/manifest.json"
    CHECKSUMS_PATH="$RELEASE_DIR/SHA256SUMS"
    FILE_URL="$BASE_URL/$DATASET/$RESOLVED_RELEASE/$ARTIFACT/$FILENAME"
    MANIFEST_URL="$BASE_URL/$DATASET/$RESOLVED_RELEASE/$ARTIFACT/manifest.json"
    CHECKSUMS_URL="$BASE_URL/$DATASET/$RESOLVED_RELEASE/SHA256SUMS"
    CHECKSUMS_KEY="$ARTIFACT/$FILENAME"
    MANIFEST_CHECKSUM_KEY="$ARTIFACT/manifest.json"

    mkdir -p "$LOCAL_DIR"
    curl -fsSL "$CHECKSUMS_URL" -o "$CHECKSUMS_PATH"
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "$LOCAL_PATH"
        exit 0
    fi

    download_and_verify "$FILE_URL" "$LOCAL_PATH" "$CHECKSUMS_KEY"
    download_and_verify "$MANIFEST_URL" "$MANIFEST_PATH" "$MANIFEST_CHECKSUM_KEY"
    echo "$LOCAL_PATH"
}

download_and_verify() {
    local url="$1"
    local local_path="$2"
    local checksum_key="$3"
    local tmp_path actual_hash expected_hash

    expected_hash="$(awk -v file="$checksum_key" '$2 == file { print $1 }' "$CHECKSUMS_PATH")"
    if [ -z "$expected_hash" ]; then
        echo -e "${R}error: could not resolve hash for '$checksum_key'${NC}" >&2
        exit 1
    fi

    if [ -f "$local_path" ]; then
        actual_hash=$($SHACMD "$local_path" | awk '{print $1}')
        if [ "$actual_hash" = "$expected_hash" ]; then
            return 0
        fi
    fi

    tmp_path="${local_path}.tmp"
    curl -fsSL "$url" -o "$tmp_path"
    actual_hash=$($SHACMD "$tmp_path" | awk '{print $1}')
    if [ "$actual_hash" != "$expected_hash" ]; then
        rm -f "$tmp_path"
        echo -e "${R}error: hash mismatch for '$checksum_key'${NC}" >&2
        exit 1
    fi

    mv "$tmp_path" "$local_path"
}

resolve_latest_release() {
    local base_url="$1"
    local dataset="$2"
    local latest_json
    local latest_release

    latest_json="$(curl -fsSL "$base_url/$dataset/latest.json")"
    latest_release="$(echo "$latest_json" | sed -n 's/.*"latest_release"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
    if [ -z "$latest_release" ]; then
        echo "failed to resolve latest release for dataset '$dataset'" >&2
        exit 1
    fi
    echo "$latest_release"
}

resolve_filename() {
    local artifact="$1"
    local split="$2"
    local format="$3"

    case "$artifact" in
        route-holdout-n1-n5|reaction-holdout-n1-n5)
            case "$format" in
                jsonl) echo "${split}.jsonl.gz" ;;
                *)
                    echo "unsupported format '$format' for artifact '$artifact'" >&2
                    exit 1
                    ;;
            esac
            ;;
        single-step-reaction-holdout-n1-n5)
            case "$format" in
                jsonl) echo "${split}.jsonl.gz" ;;
                rsmi) echo "${split}.rsmi.txt.gz" ;;
                *)
                    echo "unsupported format '$format' for artifact '$artifact'" >&2
                    exit 1
                    ;;
            esac
            ;;
    esac
}

main "$@"
