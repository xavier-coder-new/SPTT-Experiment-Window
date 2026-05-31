#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0
seeds=(2023 2024)
# data_names=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "traffic" "electricity" "exchange_rate" "weather")
data_names=("ETTh1")
# data_names=("illness")
time_block_nums=(4)
kranks=(7)
# learning_rates=0.0001 # 只针对illness数据集
learning_rates=0.001
inverse_model=align

for data_name in "${data_names[@]}"; do
    for seed in "${seeds[@]}"; do
        for time_block_num in "${time_block_nums[@]}"; do
            for krank in "${kranks[@]}"; do
                for learning_rate in "${learning_rates[@]}"; do
                    job_name="long_term_forecast-Dataset--${data_name}--Seed--${seed}--Slide_Num--${time_block_num}--Krank--${krank}--LR--${learning_rate}--Mode--${inverse_model}"

                    echo "--------------------------------------------------------------"
                    echo "Running job: ${job_name}"
                    echo "Learning rate: ${learning_rate}"
                    echo "Slide window nums: ${time_block_num}"
                    echo "Krank: ${krank}"
                    echo "Seed: ${seed}"
                    echo "--------------------------------------------------------------"
                    
                    args=(
                        "$data_name"
                        "$seed"
                        "$time_block_num"
                        "$krank"
                        "$learning_rate"
                        "$inverse_model"
                    )
                    
                    if [[ "$inverse_model" = "align" ]]; then
                        echo "执行模式: $inverse_model"
                        # ./scripts/long_term_forecast/Sptt/SpttGRU.sh "${args[@]}"
                        # ./scripts/long_term_forecast/Bptt/BpttGRU.sh "${args[@]}"
                        # ./scripts/long_term_forecast/Sptt/SpttGRU_T.sh "${args[@]}"

                        ./scripts/long_term_forecast/Sptt/SpttLSTM.sh "${args[@]}"
                        ./scripts/long_term_forecast/Bptt/BpttLSTM.sh "${args[@]}"
                        # ./scripts/long_term_forecast/Sptt/SpttLSTM_T.sh "${args[@]}"
                        
                        # ./scripts/long_term_forecast/Bptt/BpttSRU.sh "${args[@]}"
                        # ./scripts/long_term_forecast/Sptt/SpttSRU.sh "${args[@]}"
                        # ./scripts/long_term_forecast/Sptt/SpttSRU_T.sh "${args[@]}"

                        # ./scripts/long_term_forecast/Bptt/BpttMinLSTM.sh "${args[@]}"
                        # ./scripts/long_term_forecast/Sptt/SpttMinLSTM.sh "${args[@]}"
                        # ./scripts/long_term_forecast/Sptt/SpttMinLSTM_T.sh "${args[@]}"

                        # ./scripts/long_term_forecast/Bptt/BpttMinGRU.sh "${args[@]}"
                        # ./scripts/long_term_forecast/Sptt/SpttMinGRU.sh "${args[@]}"
                        # ./scripts/long_term_forecast/Sptt/SpttMinGRU_T.sh "${args[@]}"
                    else
                        echo "执行模式: $inverse_model"
                        ./scripts/long_term_forecast/Sptt_no_align/SpttLSTM.sh "${args[@]}"
                        ./scripts/long_term_forecast/Sptt_no_align/SpttLSTM_T.sh "${args[@]}"

                        ./scripts/long_term_forecast/Sptt_no_align/SpttGRU.sh "${args[@]}"
                        ./scripts/long_term_forecast/Sptt_no_align/SpttGRU_T.sh "${args[@]}"
                    fi
                done
            done
        done
        echo "--------------------------------------------------------------"
        echo "Finished job: ${job_name}"
        echo "--------------------------------------------------------------"

        sleep 30
    done
    sleep 30
done
echo "全部任务运行结束"