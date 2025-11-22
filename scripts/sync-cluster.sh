#!/usr/bin/env bash

set -e

SERVERS=(
    # "ubuntu@129.213.94.122"
    # "ubuntu@193.122.153.180"
    # "ubuntu@129.213.82.162"
    # "aws-spot-1"
    # "aws-spot-2"
    "aws-spot-3"
)

FILES=(
    "uspto-190.json.gz"
)
STOCK_FILES=(
    # "buyables-stock.hdf5"
    "buyables-stock.txt"
    # "n5-stock.hdf5"
    # "n1-n5-stock.hdf5"
    "n1-n5-stock.txt"
)

for server in "${SERVERS[@]}"; do
    echo "Syncing to ${server}..."
    for stock_file in "${STOCK_FILES[@]}"; do
        rsync -avz data/1-benchmarks/stocks/${stock_file} "${server}:~/project-procrustes/data/1-benchmarks/stocks/"
    done
    # for file in "${FILES[@]}"; do
    #     rsync -avz data/1-benchmarks/definitions/${file} "${server}:~/project-procrustes/data/1-benchmarks/definitions/"
    # done
done

echo "Done!"
