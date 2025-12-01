#!/bin/bash

# Batch run script for retrocast workflows
# Part 1: Run DMS workflow for dms-explorer-xl model across multiple benchmarks
# Part 2: Run standard workflow for multiple models across the same benchmarks

set -e  # Exit on error

# Define variables
DMS_MODEL="dms-explorer-xl"
# BENCHMARKS=("mkt-cnv-160" "mkt-lin-500" "ref-lin-600" "ref-cnv-400" "ref-lng-84")
BENCHMARKS=("uspto-190")
STANDARD_MODELS=("retro-star" "retro-star-high" "askcos" "syntheseus-retro0-local-retro" "aizynthfinder-mcts" "aizynthfinder-retro-star")

# Progress bar function with ETA and elapsed time
show_progress() {
    local current=$1
    local total=$2
    local description=$3
    local width=40
    local percentage=$((current * 100 / total))
    local completed=$((width * current / total))
    local remaining=$((width - completed))

    # Calculate elapsed time
    local elapsed=$(($(date +%s) - START_TIME))
    local elapsed_hours=$((elapsed / 3600))
    local elapsed_minutes=$(((elapsed % 3600) / 60))
    local elapsed_secs=$((elapsed % 60))

    local elapsed_msg=""
    if [ "$elapsed_hours" -gt 0 ]; then
        elapsed_msg=$(printf "%dh %dm %ds" "$elapsed_hours" "$elapsed_minutes" "$elapsed_secs")
    elif [ "$elapsed_minutes" -gt 0 ]; then
        elapsed_msg=$(printf "%dm %ds" "$elapsed_minutes" "$elapsed_secs")
    else
        elapsed_msg=$(printf "%ds" "$elapsed_secs")
    fi

    # Calculate ETA
    local eta_msg=""
    if [ "$current" -gt 0 ]; then
        local avg_time_per_step=$((elapsed / current))
        local remaining_steps=$((total - current))
        local eta_seconds=$((avg_time_per_step * remaining_steps))

        local eta_hours=$((eta_seconds / 3600))
        local eta_minutes=$(((eta_seconds % 3600) / 60))
        local eta_secs=$((eta_seconds % 60))

        if [ "$eta_hours" -gt 0 ]; then
            eta_msg=$(printf "%dh %dm %ds" "$eta_hours" "$eta_minutes" "$eta_secs")
        elif [ "$eta_minutes" -gt 0 ]; then
            eta_msg=$(printf "%dm %ds" "$eta_minutes" "$eta_secs")
        else
            eta_msg=$(printf "%ds" "$eta_secs")
        fi
    else
        eta_msg="calculating..."
    fi

    printf "\r["
    printf "%${completed}s" | tr ' ' '='
    printf "%${remaining}s" | tr ' ' '-'
    printf "] %3d%% (%d/%d) Elapsed: %s | ETA: %s | %s" "$percentage" "$current" "$total" "$elapsed_msg" "$eta_msg" "$description"
}

# Calculate total steps
TOTAL_DMS_STEPS=$((${#BENCHMARKS[@]} * 3))  # 3 steps per benchmark
TOTAL_STANDARD_STEPS=$((${#STANDARD_MODELS[@]} * ${#BENCHMARKS[@]} * 3))  # 3 steps per model-benchmark combo
TOTAL_STEPS=$((TOTAL_DMS_STEPS + TOTAL_STANDARD_STEPS))
CURRENT_STEP=0

# Track start time for ETA calculation
START_TIME=$(date +%s)

echo "=========================================="
echo "Part 1: DMS Explorer XL Workflow"
echo "=========================================="
echo ""

for benchmark in "${BENCHMARKS[@]}"; do
    # Ingest DMS legacy data
    ((CURRENT_STEP++))
    show_progress "$CURRENT_STEP" "$TOTAL_STEPS" "${DMS_MODEL}/${benchmark}: Ingesting..."
    uv run scripts/directmultistep/ingest-dms-legacy.py --model "${DMS_MODEL}" --benchmark "${benchmark}" > /dev/null 2>&1

    # Score
    ((CURRENT_STEP++))
    show_progress "$CURRENT_STEP" "$TOTAL_STEPS" "${DMS_MODEL}/${benchmark}: Scoring...  "
    retrocast score --model "${DMS_MODEL}" --dataset "${benchmark}" > /dev/null 2>&1

    # Analyze
    ((CURRENT_STEP++))
    show_progress "$CURRENT_STEP" "$TOTAL_STEPS" "${DMS_MODEL}/${benchmark}: Analyzing..."
    retrocast analyze --model "${DMS_MODEL}" --dataset "${benchmark}" > /dev/null 2>&1
done

echo ""
echo ""

echo "=========================================="
echo "Part 2: Standard Models Workflow"
echo "=========================================="
echo ""

for model in "${STANDARD_MODELS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        # Ingest
        ((CURRENT_STEP++))
        show_progress "$CURRENT_STEP" "$TOTAL_STEPS" "${model}/${benchmark}: Ingesting..."
        retrocast ingest --model "${model}" --dataset "${benchmark}" > /dev/null 2>&1

        # Score
        ((CURRENT_STEP++))
        show_progress "$CURRENT_STEP" "$TOTAL_STEPS" "${model}/${benchmark}: Scoring...  "
        retrocast score --model "${model}" --dataset "${benchmark}" > /dev/null 2>&1

        # Analyze
        ((CURRENT_STEP++))
        show_progress "$CURRENT_STEP" "$TOTAL_STEPS" "${model}/${benchmark}: Analyzing..."
        retrocast analyze --model "${model}" --dataset "${benchmark}" > /dev/null 2>&1
    done
done

echo ""
echo ""

echo ""
echo "=========================================="
echo "All workflows completed successfully!"
echo "=========================================="
