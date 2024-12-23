#!/bin/bash

for i in {1..5}; do
    nohup wandb agent mislam6/park_rebuttal_experiments/egc1f010 > platt_scaling_$i.txt &
    echo "Started run $i, logging to platt_scaling_$i.txt"
    sleep 1  # Optional: Small delay to ensure proper initialization
done
