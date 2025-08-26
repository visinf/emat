#!/bin/bash
cd ..

if [[ $# -lt 1 ]]; then
  echo "Error: GPU ID not specified" 
  echo "Usage: $0 <GPU ID>"            
  echo "Example: $0 2"            
  exit 1
fi

echo "Running all evaluations of CST..."
for DATASET in pascal coco; do
    for FOLD in 0 1 2 3; do
        for SETTING in original partially-augmented fully-augmented; do
            echo "▶️ Benchmark=$DATASET | Fold=$FOLD | Setting=$SETTING"
            python main.py \
                --experiment_path "./experiments/cst-$DATASET" \
                --results_path "./results" \
                --fold "$FOLD" \
                --way 2 \
                --shot 1 \
                --setting "$SETTING" \
                --eval \
                --gpus "$1"
            echo
        done
        for INTERVAL in 0-5 5-10 10-15; do
            echo "▶️ Benchmark=$DATASET | Fold=$FOLD | Interval=$INTERVAL"
            python main.py \
                --experiment_path "./experiments/cst-$DATASET" \
                --results_path "./results" \
                --fold "$FOLD" \
                --way 1 \
                --shot 1 \
                --object_size_split "$INTERVAL" \
                --eval \
                --no_empty_masks \
                --gpus "$1"
            echo
        done
        if [ "$DATASET" = "pascal" ]; then
            echo "▶️ Benchmark=$DATASET | Fold=$FOLD | Interval=0-15"
            python main.py \
                --experiment_path "./experiments/cst-$DATASET" \
                --results_path "./results" \
                --fold "$FOLD" \
                --way 1 \
                --shot 1 \
                --object_size_split "0-15" \
                --eval \
                --no_empty_masks \
                --gpus "$1"
            echo
        fi
    done
done

echo "✅ All experiments finished."
