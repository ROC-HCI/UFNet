#!/bin/bash

for i in {1..10}; do
    nohup wandb agent mislam6/park_rebuttal_experiments/9l21mj51 > label_smoothing_platt_scaling_$i.txt &
    echo "Started run $i, logging to label_smoothing_platt_scaling_$i.txt"
    sleep 1  # Optional: Small delay to ensure proper initialization
done
