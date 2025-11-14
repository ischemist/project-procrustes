#!/bin/bash

# create-tarball.sh - Create tarballs of processed data with flexible options
#
# Usage: ./create-tarball.sh [OPTIONS] <dataset_name>
#
# Options:
#   --dry-run       Show what would be included without creating tarball
#   --help          Show this help message
#   --verbose       Show verbose output
#   --output FILE   Specify output filename (default: processed_<dataset>_data.tar.gz)
#   --exclude-manifest  Exclude manifest.json files (default: true)
#   --include-pattern   File pattern to include (default: *.json.gz)
#   --recent-days N     Only include subfolders created within N days (default: all)
#
# Examples:
#   ./create-tarball.sh rs-first-25
#   ./create-tarball.sh --dry-run uspto-190
#   ./create-tarball.sh --output my_data.tar.gz --verbose uspto-190
#   ./create-tarball.sh --recent-days 2 rs-first-25

set -euo pipefail

# Default values
DRY_RUN=false
VERBOSE=false
EXCLUDE_MANIFEST=true
INCLUDE_PATTERN="*.json.gz"
OUTPUT_FILE=""
DATASET_NAME=""
RECENT_DAYS=""
MODEL_HASH=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_verbose() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[VERBOSE]${NC} $1"
    fi
}

# Help function
show_help() {
    cat << EOF
create-tarball.sh - Create tarballs of processed data with flexible options

USAGE:
    ./create-tarball.sh [OPTIONS] <dataset_name>

ARGUMENTS:
    <dataset_name>      Name of the dataset directory in data/processed/
                       Available: uspto-190

OPTIONS:
    --dry-run          Show what would be included without creating tarball
    --help             Show this help message
    --verbose          Show verbose output during processing
    --output FILE      Specify output filename (default: data/transfers/processed_<dataset>_data.tar.gz)
    --exclude-manifest Exclude manifest.json files (default: true)
    --include-pattern  File pattern to include (default: *.json.gz)
    --recent-days N    Only include subfolders created within N days (default: all)
    --model-hash HASH  Only include subfolders containing the specified hash (e.g., df005226)

EXAMPLES:
    ./create-tarball.sh rs-first-25
    ./create-tarball.sh --dry-run uspto-190
    ./create-tarball.sh --output data/transfers/my_data.tar.gz --verbose uspto-190
    ./create-tarball.sh --include-pattern "*.json*" rs-first-25
    ./create-tarball.sh --recent-days 2 rs-first-25
    ./create-tarball.sh --model-hash df005226 uspto-190

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --output|-o)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --exclude-manifest)
            EXCLUDE_MANIFEST=true
            shift
            ;;
        --include-manifest)
            EXCLUDE_MANIFEST=false
            shift
            ;;
        --include-pattern)
            INCLUDE_PATTERN="$2"
            shift 2
            ;;
         --recent-days)
             RECENT_DAYS="$2"
             if ! [[ "$RECENT_DAYS" =~ ^[0-9]+$ ]]; then
                 print_error "Invalid value for --recent-days: '$RECENT_DAYS'. Must be a positive integer."
                 exit 1
             fi
             shift 2
             ;;
         --model-hash)
             MODEL_HASH="$2"
             shift 2
             ;;
         --*)
             print_error "Unknown option: $1"
             echo "Use --help for usage information."
             exit 1
             ;;
        *)
            if [ -z "$DATASET_NAME" ]; then
                DATASET_NAME="$1"
            else
                print_error "Multiple dataset names provided: '$DATASET_NAME' and '$1'"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATASET_NAME" ]; then
    print_error "Dataset name is required"
    echo "Use --help for usage information."
    exit 1
fi

# Create transfers directory if it doesn't exist
TRANSFERS_DIR="data/transfers"
if [ ! -d "$TRANSFERS_DIR" ]; then
    print_verbose "Creating transfers directory: $TRANSFERS_DIR"
    mkdir -p "$TRANSFERS_DIR"
fi

# Set default output filename if not provided
if [ -z "$OUTPUT_FILE" ]; then
    OUTPUT_FILE="$TRANSFERS_DIR/processed_${DATASET_NAME}_data.tar.gz"
fi

# Check if we're in the right directory
if [ ! -d "data/processed" ]; then
    print_error "data/processed directory not found. Please run from project root."
    exit 1
fi

# Check if dataset directory exists
DATASET_PATH="data/processed/$DATASET_NAME"
if [ ! -d "$DATASET_PATH" ]; then
    print_error "Dataset directory '$DATASET_PATH' does not exist"
    echo "Available datasets:"
    find data/processed -maxdepth 1 -type d -not -path "data/processed" | sed 's|data/processed/||' | sort
    exit 1
fi

print_info "Processing dataset: $DATASET_NAME"
print_verbose "Dataset path: $DATASET_PATH"
print_verbose "Include pattern: $INCLUDE_PATTERN"
print_verbose "Exclude manifest: $EXCLUDE_MANIFEST"
if [ -n "$RECENT_DAYS" ]; then
    print_verbose "Recent days filter: $RECENT_DAYS days"
fi
if [ -n "$MODEL_HASH" ]; then
    print_verbose "Model hash filter: $MODEL_HASH"
fi

# Create temporary file for file list
TEMP_FILE=$(mktemp)
trap "rm -f $TEMP_FILE" EXIT

# Find files matching the pattern
print_verbose "Searching for files matching pattern: $INCLUDE_PATTERN"

if [ -n "$MODEL_HASH" ]; then
    print_verbose "Filtering subfolders containing hash: $MODEL_HASH"
    # Create temporary file for matching subfolders
    TEMP_SUBFOLDERS=$(mktemp)
    trap "rm -f $TEMP_FILE $TEMP_SUBFOLDERS" EXIT

    # Find subfolders whose names contain the specified hash
    find "$DATASET_PATH" -maxdepth 1 -type d -not -path "$DATASET_PATH" -name "*$MODEL_HASH*" > "$TEMP_SUBFOLDERS"

    # Check if any subfolders matched
    SUBFOLDER_COUNT=$(wc -l < "$TEMP_SUBFOLDERS")
    if [ "$SUBFOLDER_COUNT" -eq 0 ]; then
        print_warning "No subfolders found containing hash: $MODEL_HASH"
        exit 0
    fi

    # Initialize empty file list
    > "$TEMP_FILE"

    # For each matching subfolder, find matching files
    while IFS= read -r subfolder; do
        if [ -d "$subfolder" ]; then
            print_verbose "Including files from subfolder: $(basename "$subfolder")"
            find "$subfolder" -name "$INCLUDE_PATTERN" -type f >> "$TEMP_FILE"
        fi
    done < "$TEMP_SUBFOLDERS"
elif [ -n "$RECENT_DAYS" ]; then
    print_verbose "Filtering subfolders created within last $RECENT_DAYS days"
    # Create temporary file for recent subfolders
    TEMP_SUBFOLDERS=$(mktemp)
    trap "rm -f $TEMP_FILE $TEMP_SUBFOLDERS" EXIT

    # Find subfolders created within the specified number of days
    find "$DATASET_PATH" -maxdepth 1 -type d -not -path "$DATASET_PATH" -ctime -"$RECENT_DAYS" > "$TEMP_SUBFOLDERS"

    # Initialize empty file list
    > "$TEMP_FILE"

    # For each recent subfolder, find matching files
    while IFS= read -r subfolder; do
        if [ -d "$subfolder" ]; then
            print_verbose "Including files from recent subfolder: $(basename "$subfolder")"
            find "$subfolder" -name "$INCLUDE_PATTERN" -type f >> "$TEMP_FILE"
        fi
    done < "$TEMP_SUBFOLDERS"
else
    find "$DATASET_PATH" -name "$INCLUDE_PATTERN" -type f > "$TEMP_FILE"
fi

# Exclude manifest files if requested
if [ "$EXCLUDE_MANIFEST" = true ]; then
    print_verbose "Excluding manifest.json files"
    grep -v "manifest\.json$" "$TEMP_FILE" > "${TEMP_FILE}.filtered" || true
    mv "${TEMP_FILE}.filtered" "$TEMP_FILE"
fi

# Check if any files were found
FILE_COUNT=$(wc -l < "$TEMP_FILE")
if [ "$FILE_COUNT" -eq 0 ]; then
    print_warning "No files found matching criteria"
    exit 0
fi

print_info "Found $FILE_COUNT files that match the criteria:"
echo

# Print files that will be included
while IFS= read -r file; do
    if [ -f "$file" ]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "  ✓ $file ($SIZE)"
    fi
done < "$TEMP_FILE"

echo

# Calculate total size
TOTAL_SIZE=$(du -ch $(cat "$TEMP_FILE") 2>/dev/null | tail -1 | cut -f1)
print_info "Total size: $TOTAL_SIZE"

# If dry run, exit here
if [ "$DRY_RUN" = true ]; then
    print_info "Dry run complete - no tarball created"
    print_info "Would create: $OUTPUT_FILE"
    exit 0
fi

# Create the tarball
print_info "Creating tarball: $OUTPUT_FILE"

if [ "$VERBOSE" = true ]; then
    tar -czvf "$OUTPUT_FILE" -T "$TEMP_FILE"
else
    tar -czf "$OUTPUT_FILE" -T "$TEMP_FILE"
fi

# Verify the tarball was created
if [ -f "$OUTPUT_FILE" ]; then
    TARBALL_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    print_success "Tarball created successfully: $OUTPUT_FILE ($TARBALL_SIZE)"

    if [ "$VERBOSE" = true ]; then
        print_info "Tarball contents:"
        tar -tzf "$OUTPUT_FILE" | head -10
        if [ "$(tar -tzf "$OUTPUT_FILE" | wc -l)" -gt 10 ]; then
            echo "  ... and $(($(tar -tzf "$OUTPUT_FILE" | wc -l) - 10)) more files"
        fi
    fi
else
    print_error "Failed to create tarball"
    exit 1
fi
