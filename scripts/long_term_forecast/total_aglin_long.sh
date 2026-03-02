#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0
seeds=(2025 2024 2023 2022 2021 2020)
data_names=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "traffic" "electricity" "exchange_rate" "weather")
# data_names=("illness")
time_block_nums=(4 8)
kranks=7
# learning_rates=0.0001 # 只针对illness数据集
learning_rates=0.001
inverse_model=align


for data_name in "${data_names[@]}"; do
    for seed in "${seeds[@]}"; do
        for time_block_num in "${time_block_nums[@]}"; do
            for krank in "${kranks[@]}"; do
                for learning_rate in "${learning_rates[@]}"; do
                    echo "开始运行 $data_name, seed=$seed, time_block_num=$time_block_num, krank=$krank, learning_rate=$learning_rate"
                    if [[ "$inverse_model" = "align" ]]; then
                        echo "执行模式: $inverse_model"
                        ./scripts/long_term_forecast/Bptt/BpttLSTM.sh "$data_name" "$seed" "$time_block_num" "$learning_rate" "$inverse_model"
                        ./scripts/long_term_forecast/Sptt/SpttLSTM.sh "$data_name" "$seed" "$time_block_num" "$krank" "$learning_rate" "$inverse_model"

                        ./scripts/long_term_forecast/Sptt/SpttGRU.sh "$data_name" "$seed" "$time_block_num" "$krank" "$learning_rate" "$inverse_model"
                        ./scripts/long_term_forecast/Bptt/BpttGRU.sh "$data_name" "$seed" "$time_block_num" "$learning_rate" "$inverse_model"
                    else
                        echo "执行模式: $inverse_model"
                        ./scripts/long_term_forecast/Sptt_no_align/SpttLSTM.sh "$data_name" "$seed" "$time_block_num" "$krank" "$learning_rate" "$inverse_model"

                        ./scripts/long_term_forecast/Sptt_no_align/SpttGRU.sh "$data_name" "$seed" "$time_block_num" "$krank" "$learning_rate" "$inverse_model"
                    fi
                done
            done
        done
        sleep 30
        echo "运行结束 $data_name, seed=$seed" 
    done
    sleep 30
done
echo "全部任务运行结束"