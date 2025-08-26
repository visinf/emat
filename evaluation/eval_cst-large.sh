#!/bin/bash
cd ..

if [[ $# -lt 1 ]]; then
  echo "Error: GPU ID not specified" 
  echo "Usage: $0 <GPU ID>"            
  echo "Example: $0 2"            
  exit 1
fi

echo "Running all evaluations of CST-Large..."
for FOLD in 0 1 2 3; do
    echo "▶️ Fold=$FOLD | All Dataset"
    python main.py \
        --experiment_path "./experiments/cst-large-pascal" \
        --results_path "./results" \
        --fold "$FOLD" \
        --way 2 \
        --shot 1 \
        --setting "original" \
        --eval \
        --gpus "$1"
    echo
    echo "▶️ Fold=$FOLD | Small Objects"
    python main.py \
        --experiment_path "./experiments/cst-large-pascal" \
        --results_path "./results" \
        --fold "$FOLD" \
        --way 1 \
        --shot 1 \
        --object_size_split "0-15" \
        --eval \
        --no_empty_masks \
        --gpus "$1"
    echo
done

echo "✅ All experiments finished."
