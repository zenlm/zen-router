#!/usr/bin/env bash
# Quantize trained zen-router to GGUF (Q4_K_M) and MLX.
set -euo pipefail
MODEL_DIR="${1:-out/zen-router}"
python ../llama.cpp/convert_hf_to_gguf.py "$MODEL_DIR" --outfile "$MODEL_DIR/zen-router-f16.gguf"
../llama.cpp/build/bin/llama-quantize "$MODEL_DIR/zen-router-f16.gguf" "$MODEL_DIR/zen-router-Q4_K_M.gguf" Q4_K_M
python -m mlx_lm convert --hf-path "$MODEL_DIR" --mlx-path "$MODEL_DIR/mlx" -q
