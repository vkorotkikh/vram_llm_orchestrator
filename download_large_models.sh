#!/bin/bash
# save as: download_large_models.sh

MODELS_DIR="$HOME/models"
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

echo "Downloading large GGUF models for 60GB VRAM setup..."

# Llama 3.1 70B Q4_K_M - Best all-rounder
huggingface-cli download bartowski/Meta-Llama-3.1-70B-Instruct-GGUF \
    Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf \
    --local-dir "$MODELS_DIR"

# Mixtral 8x7B - Fast MoE
huggingface-cli download TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF \
    mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf \
    --local-dir "$MODELS_DIR"

echo "Done! Models saved to $MODELS_DIR"
ls -lh "$MODELS_DIR"/*.gguf