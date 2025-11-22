#!/bin/bash

mkdir -p data/0-assets/model-configs/dms/checkpoints

# Define URLs
CKPT_URL="https://files.batistalab.com/DirectMultiStep/ckpts"
DATASET_URL="https://files.batistalab.com/DirectMultiStep/datasets"

# Model checkpoint configurations
model_names=(
    "Flash"
    "Flex"
    "Wide"
    "Explorer-XL"
)
model_info=(
    "flash.ckpt|38"
    "flex.ckpt|74"
    "wide.ckpt|147"
    "explorer_xl.ckpt|192"
)

# Download model checkpoints
read -p "Do you want to download all model checkpoints? [y/N]: " all_choice
case "$all_choice" in
    y|Y )
        for i in "${!model_names[@]}"; do
            model="${model_names[$i]}"
            info="${model_info[$i]}"
            IFS="|" read -r filename size <<< "$info"
            echo "Downloading ${model} model ckpt (${size} MB)..."
            curl -o "data/0-assets/model-configs/dms/checkpoints/${filename}" "${CKPT_URL}/${filename}"
        done
        ;;
    * )
        for i in "${!model_names[@]}"; do
            model="${model_names[$i]}"
            info="${model_info[$i]}"
            IFS="|" read -r filename size <<< "$info"
            read -p "Do you want to download ${model} model ckpt? (${size} MB) [y/N]: " choice
            case "$choice" in
                y|Y )
                    curl -o "data/0-assets/model-configs/dms/checkpoints/${filename}" "${CKPT_URL}/${filename}"
                    ;;
                * )
                    echo "Skipping ${model} ckpt."
                    ;;
            esac
        done
        ;;
esac
