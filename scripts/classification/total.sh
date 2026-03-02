#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

seed=(2018)

datasets=("FaceDetection")

time_block_num=(8)
for seed in "${seed[@]}"; do
    for datasets in "${datasets[@]}"; do
        for time_block_num in "${time_block_num[@]}"; do
            echo "seed: $seed, datasets: $datasets, time_block_num: $time_block_num"
            ./scripts/classification/Sptt/SpttLSTM.sh "$datasets" "$seed" "$time_block_num"


            # ./scripts/classification/Bptt/BpttLSTM.sh "$datasets" "$seed" "$time_block_num"


            # ./scripts/classification/Sptt/SpttGRU.sh "$datasets" "$seed" "$time_block_num"


            # ./scripts/classification/Bptt/BpttGRU.sh "$datasets" "$seed" "$time_block_num"

        done
        sleep 60
    done
    sleep 60
done




