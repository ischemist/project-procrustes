#!/bin/bash

MODELS=(
    "dms-flash"
    "dms-flash-20M"
    "dms-flex-duo"/
    "dms-wide"
    "dms-deep"
    "dms-explorer-xl"
)

BENCHMARKS=(
    "stratified-convergent-250"
    "stratified-linear-600"
    "random-n5-100"
    "random-n5-250"
    "random-n5-500"
    "random-n5-1000"
)

for model in "${MODELS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        echo "Running ingestion for model: $model, benchmark: $benchmark"
        uv run scripts/00-score.py --model "$model" --benchmark "$benchmark"
        uv run scripts/01-analyze.py --stock n5-stock --model "$model" --benchmark "$benchmark"
    done
done
