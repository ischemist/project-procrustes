#!/bin/bash
set -eo pipefail

VERSION="2026.05.12"

main() {
    BASE_URL="${RETROCAST_TRAINING_SET_BASE_URL:-https://files.ischemist.com/retrocast/training-sets}"
    CACHE_ROOT="${RETROCAST_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/retrocast}"
    CACHE_DIR="${RETROCAST_TRAINING_SET_CACHE_DIR:-$CACHE_ROOT/training-sets}"
    OUTPUT_DIR=""
    DATASET="paroutes"
    ARTIFACT=""
    SPLIT=""
    FORMAT=""
    RELEASE="latest"
    RELEASE_EXPLICIT=0
    OMIT=""
    DRY_RUN=0

    R='\033[0;31m'
    B='\033[0;34m'
    NC='\033[0m'

    usage() {
        echo ""
        echo -e "${B}usage:${NC}"
        if [ -t 0 ]; then
            echo "  $0 [artifact|release] [flags]"
        else
            echo "  ... | bash -s -- [artifact|release] [flags]"
        fi
        echo ""
        echo -e "${B}artifacts:${NC}"
        echo "  n1-routes"
        echo "  n5-routes"
        echo "  route-holdout-n1-n5"
        echo "  reaction-holdout-n1-n5"
        echo "  n1-single-step-reactions"
        echo "  n5-single-step-reactions"
        echo "  single-step-route-holdout-n1-n5"
        echo "  single-step-reaction-holdout-n1-n5"
        echo ""
        echo -e "${B}flags:${NC}"
        echo "  --split=all|training|validation  only download one split"
        echo "  --format=jsonl|rsmi              only download one format"
        echo "  --omit=PART[,PART...]            omit splits or formats"
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
            --omit=*) OMIT="${1#*=}"; shift ;;
            --release=*) RELEASE="${1#*=}"; RELEASE_EXPLICIT=1; shift ;;
            --dataset=*) DATASET="${1#*=}"; shift ;;
            --dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
            --dry-run) DRY_RUN=1; shift ;;
            -*)
                echo -e "${R}error: unknown option: $1${NC}" >&2
                usage
                ;;
            *)
                if [ -z "$ARTIFACT" ]; then
                    case "$1" in
                        v[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9])
                            RELEASE="$1"
                            RELEASE_EXPLICIT=1
                            ;;
                        *)
                            ARTIFACT="$1"
                            ;;
                    esac
                else
                    echo -e "${R}error: multiple positional arguments specified${NC}" >&2
                    usage
                fi
                shift
                ;;
        esac
    done

    if [ -z "$ARTIFACT" ] && [ "$RELEASE_EXPLICIT" -eq 0 ]; then
        usage
    fi

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

    if [ -n "$ARTIFACT" ]; then
        case "$ARTIFACT" in
            n1-routes|n5-routes|route-holdout-n1-n5|reaction-holdout-n1-n5|n1-single-step-reactions|n5-single-step-reactions|single-step-route-holdout-n1-n5|single-step-reaction-holdout-n1-n5) ;;
            *)
                echo -e "${R}error: unsupported artifact '$ARTIFACT'${NC}" >&2
                exit 1
                ;;
        esac
    fi

    case "$SPLIT" in
        ""|all|training|validation) ;;
        *)
            echo -e "${R}error: unsupported split '$SPLIT'${NC}" >&2
            exit 1
            ;;
    esac

    case "$FORMAT" in
        ""|jsonl|rsmi) ;;
        *)
            echo -e "${R}error: unsupported format '$FORMAT'${NC}" >&2
            exit 1
            ;;
    esac

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
    CHECKSUMS_PATH="$RELEASE_DIR/SHA256SUMS"
    CHECKSUMS_URL="$BASE_URL/$DATASET/$RESOLVED_RELEASE/SHA256SUMS"

    mkdir -p "$RELEASE_DIR"
    curl -fsSL "$CHECKSUMS_URL" -o "$CHECKSUMS_PATH"

    mapfile -t DOWNLOAD_KEYS < <(resolve_download_keys "$ARTIFACT" "$SPLIT" "$FORMAT" "$OMIT")
    if [ "${#DOWNLOAD_KEYS[@]}" -eq 0 ]; then
        echo -e "${R}error: no published files match request${NC}" >&2
        exit 1
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        for key in "${DOWNLOAD_KEYS[@]}"; do
            echo "$RELEASE_DIR/$key"
        done
        exit 0
    fi

    for key in "${DOWNLOAD_KEYS[@]}"; do
        local_path="$RELEASE_DIR/$key"
        mkdir -p "$(dirname "$local_path")"
        download_and_verify "$BASE_URL/$DATASET/$RESOLVED_RELEASE/$key" "$local_path" "$key"
        echo "$local_path"
    done
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

resolve_download_keys() {
    local artifact="$1"
    local split="$2"
    local format="$3"
    local omit="$4"

    awk -v artifact="$artifact" -v split_filter="$split" -v format_filter="$format" -v omit="$omit" '
        function has_omitted_part(file, parts, count, i) {
            count = split_csv(omit, parts)
            for (i = 1; i <= count; i++) {
                if (parts[i] != "" && file_part_matches(file, parts[i])) {
                    return 1
                }
            }
            return 0
        }
        function split_csv(value, parts, n, raw, i) {
            n = split(value, raw, ",")
            for (i = 1; i <= n; i++) {
                gsub(/^[[:space:]]+|[[:space:]]+$/, "", raw[i])
                parts[i] = raw[i]
            }
            return n
        }
        function file_part_matches(file, part) {
            return file == part ".jsonl.gz" || file == part ".rsmi.txt.gz" || file ~ ("\\." part "\\.")
        }
        $2 != "" {
            key = $2
            if (artifact != "" && key !~ ("^" artifact "/")) {
                next
            }
            file = key
            sub(/^.*\//, "", file)
            if (file == "manifest.json") {
                if (split_filter == "" && format_filter == "") {
                    print key
                }
                next
            }
            if (split_filter != "" && !file_part_matches(file, split_filter)) {
                next
            }
            if (format_filter == "jsonl" && file !~ /\.jsonl\.gz$/) {
                next
            }
            if (format_filter == "rsmi" && file !~ /\.rsmi\.txt\.gz$/) {
                next
            }
            if (has_omitted_part(file)) {
                next
            }
            print key
        }
    ' "$CHECKSUMS_PATH"
}

main "$@"
