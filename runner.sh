#!/bin/bash

echo "Starting full testing process..."

DATASETS=(
    "english5"
    "english6"
    "english7"
    "english8"
    "random5"
    "random6"
    "random7"
    "random8"
)

for dataset in "${DATASETS[@]}"; do
    echo "Running benchmark on datasets/${dataset}.json -> ./outputs/output_${dataset}.txt"
    python -u game.py "datasets/${dataset}.json" | tee "./outputs/output_${dataset}.txt"
done

echo "All tests completed successfully!"