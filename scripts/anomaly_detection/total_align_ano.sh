#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0
seeds=(2025 2024 2023 2022 2021 2020)
data_names=("MSL" "PSM" "SMAP" "SMD" "SWAT")
time_block_nums=(4 8)
kranks=(10)
learning_rates=0.0001
inverse_model=align

for data_name in "${data_names[@]}"; do 
    for seed in "${seeds[@]}"; do
        for time_block_num in "${time_block_nums[@]}"; do
            for krank in "${kranks[@]}"; do
                for learning_rate in "${learning_rates[@]}"; do
                    echo "Running $data_name, seed=$seed, time_block_num=$time_block_num, krank=$krank, learning_rate=$learning_rate"

                    echo "seed: $seed ---- $data_name "
                    ./scripts/anomaly_detection/Bptt/BpttLSTM.sh "$data_name" "$seed" "$time_block_num" "$krank" "$learning_rate" "$inverse_model"
                    ./scripts/anomaly_detection/Sptt/SpttLSTM.sh "$data_name" "$seed" "$time_block_num" "$krank" "$learning_rate" "$inverse_model"

                    ./scripts/anomaly_detection/Sptt/SpttGRU.sh "$data_name" "$seed" "$time_block_num" "$krank" "$learning_rate" "$inverse_model"
                    ./scripts/anomaly_detection/Bptt/BpttGRU.sh "$data_name" "$seed" "$time_block_num" "$krank" "$learning_rate" "$inverse_model"
                    sleep 30
                    echo "Finish $data_name, seed=$seed time_block_num=$time_block_num" 
                done
            done
        done
        echo "Finish $data_name, seed=$seed time_block_num=$time_block_num" 
        sleep 30
    done
done