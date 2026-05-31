#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
seeds=(2025 2024 2023 2022 2021 2020)
# data_names=("SelfRegulationSCP1" "SelfRegulationSCP2" "FaceDetection" "Handwriting" "Heartbeat" "EthanolConcentration")
# data_names=("JapaneseVowels" "PEMS-SF" "SpokenArabicDigits")
data_names=("Handwriting" "EthanolConcentration")
time_block_nums=(4 8)
kranks=3
learning_rates=0.001
inverse_model=align

for data_name in "${data_names[@]}"; do 
    for seed in "${seeds[@]}"; do
        for time_block_num in "${time_block_nums[@]}"; do
            for krank in "${kranks[@]}"; do
                for learning_rate in "${learning_rates[@]}"; do
                    echo "开始运行 $data_name, seed=$seed, time_block_num=$time_block_num, krank=$krank, learning_rate=$learning_rate"
                    if [[ "$inverse_model" = "align" ]]; then
                        echo "seed: $seed 可以执行 $data_name 数据集"
                        # ./scripts/classification/Bptt/BpttLSTM.sh "$data_name" "$seed" "$time_block_num" "$learning_rate" "$inverse_model"
                        ./scripts/classification/Sptt/SpttLSTM.sh "$data_name" "$seed" "$time_block_num" "$krank" "$learning_rate" "$inverse_model"

                        ./scripts/classification/Sptt/SpttGRU.sh "$data_name" "$seed" "$time_block_num" "$krank" "$learning_rate" "$inverse_model"
                        # ./scripts/classification/Bptt/BpttGRU.sh "$data_name" "$seed" "$time_block_num" "$learning_rate" "$inverse_model"
                    else
                        echo "执行模式: $inverse_model"
                        ./scripts/classification/Sptt_no_align/SpttLSTM.sh "$data_name" "$seed" "$time_block_num" "$krank" "$learning_rate" "$inverse_model"

                        ./scripts/classification/Sptt_no_align/SpttGRU.sh "$data_name" "$seed" "$time_block_num" "$krank" "$learning_rate" "$inverse_model"
                    fi
                    sleep 30
                    echo "结束运行 $data_name, seed=$seed" 
                done
            done
        done
        echo "运行结束 $data_name, seed=$seed time_block_num=$time_block_num" 
        sleep 30
    done
done