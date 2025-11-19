#!/bin/bash

MODELS=(
    "dms-flash"
    "dms-flash-20M"
    "dms-flex-duo"
    "dms-wide"
    "dms-deep"
    "dms-explorer-xl"
)

BENCHMARKS=(
    "stratified-convergent-450"
    "stratified-linear-600"
    # "random-n5-100"
    # "random-n5-250"
    # "random-n5-500"
    # "random-n5-1000"
)

seeds=(
    # 299792458
    # 19910806
    # 20260317
    # 17760704
    # 17890304
    # 42
    # 20251030
    # 662607015
    # 20180329
    20170612
    20180818
    20151225
    19690721
    20160310
    19450716
)

for model in "${MODELS[@]}"; do
    for benchmark in "${BENCHMARKS[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Running ingestion for model: $model, benchmark: $benchmark"
            uv run scripts/directmultistep/ingest-dms-legacy.py --model "$model" --benchmark "$benchmark-seed=$seed"
        done
    done
done
