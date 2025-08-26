#!/bin/bash
cd ..

if [[ $# -lt 1 ]]; then
  echo "Error: GPU ID not specified" 
  echo "Usage: $0 <GPU ID>"            
  echo "Example: $0 2"            
  exit 1
fi

echo "Running all evaluations of EMATSeg..."
for DATASET in pascal coco; do
    for FOLD in 0 1 2 3; do
        for SHOT in 1 5; do
            echo "▶️ Benchmark=$DATASET | Fold=$FOLD | Shot=$SHOT"
            python main.py \
                --experiment_path "./experiments/ematseg-$DATASET" \
                --results_path "./results" \
                --fold "$FOLD" \
                --way 1 \
                --shot "$SHOT" \
                --eval \
                --only_seg \
                --no_empty_masks \
                --gpus "$1"
            echo
        done
    done
done

echo "✅ All experiments finished."
