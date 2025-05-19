#!/bin/bash
# Get the number of GPUs
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
SWEEP_ID="your_sweep_id_here"

for i in $(seq 0 $((GPU_COUNT-1)))
do
  echo "Launching agent on GPU $i"
  CUDA_VISIBLE_DEVICES=$i wandb agent $SWEEP_ID &
  sleep 2
done

wait
