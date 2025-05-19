#!/bin/bash
SWEEP_ID=$(wandb sweep --project RL-HW3-SAC-HalfCheetah sweep.yaml | grep 'wandb agent' | awk '{print $4}')
echo "Sweep ID: $SWEEP_ID"

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

for i in $(seq 0 $((GPU_COUNT-1)))
do
  echo "Launching agent on GPU $i"
  CUDA_VISIBLE_DEVICES=$i wandb agent $SWEEP_ID &
  sleep 2
done

wait
