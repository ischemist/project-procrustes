#!/bin/bash

set -e

BASE_URL="https://files.ischemist.com/retro-star"
TARGET_DIR="data/0-assets/model-configs/retro-star"

mkdir -p "${TARGET_DIR}/one_step_model"
mkdir -p "${TARGET_DIR}/saved_models"

echo "Downloading RetroStar assets..."

echo "Downloading one_step_model files..."
curl -L "${BASE_URL}/one_step_model/saved_rollout_state_1_2048.ckpt" -o "${TARGET_DIR}/one_step_model/saved_rollout_state_1_2048.ckpt"
curl -L "${BASE_URL}/one_step_model/template_rules_1.dat" -o "${TARGET_DIR}/one_step_model/template_rules_1.dat"

echo "Downloading saved_models files..."
curl -L "${BASE_URL}/saved_models/best_epoch_final_4.pt" -o "${TARGET_DIR}/saved_models/best_epoch_final_4.pt"

echo "Download complete! Assets saved to ${TARGET_DIR}"
