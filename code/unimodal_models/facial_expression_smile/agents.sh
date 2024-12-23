#!/bin/bash

for i in {1..35}; do
    nohup wandb agent mislam6/park_rebuttal_experiments/ppr1y6xj > yt_$i.txt &
    echo "Started run $i, logging to yt_$i.txt"
    sleep 1  # Optional: Small delay to ensure proper initialization
done
