#!/usr/bin/env bash

set -e

SERVERS=(
    "ubuntu@157.151.200.193"
    # "ubuntu@193.122.153.180"
    # "ubuntu@129.213.82.162"
    # "aws-spot-1"
    # "aws-spot-2"
    # "aws-spot-3"
    # "aws-spot-4"
)

FILES=(
    # "uspto-190.json.gz"
    # "mkt-cnv-160.json.gz"
    # "mkt-lin-500.json.gz"
    # "ref-cnv-400.json.gz"
    # "ref-lin-600.json.gz"
    # "ref-lng-84.json.gz"
    "random-n5-50.json.gz"
)
STOCK_FILES=(
    "buyables-stock.hdf5"
    "n5-stock.hdf5"
    "n1-n5-stock.hdf5"
    # "buyables-stock-canon.txt"
    # "buyables-stock.txt"
    # "n1-n5-stock.txt"
    # "n5-stock.txt"
)

for server in "${SERVERS[@]}"; do
    echo "Syncing to ${server}..."
    rsync -avz data/1-benchmarks/stocks "${server}:~/project-procrustes/data/1-benchmarks/"
    for file in "${FILES[@]}"; do
        rsync -avz data/1-benchmarks/definitions/${file} "${server}:~/project-procrustes/data/1-benchmarks/definitions/"
    done
done

echo "Done!"
