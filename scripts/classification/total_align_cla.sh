#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0
seeds=(2021 2022)
data_names=("FaceDetection")
time_block_nums=(4)
kranks=(6)
learning_rates=0.001
inverse_model=align

for seed in "${seeds[@]}"; do
    for data_name in "${data_names[@]}"; do
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
                    ./scripts/classification/Bptt/BpttLSTM.sh "${args[@]}"
                    ./scripts/classification/Sptt/SpttLSTM.sh "${args[@]}"
                    # ./scripts/classification/Sptt/SpttLSTM_T.sh "${args[@]}"

                    # ./scripts/classification/Bptt/BpttSRU.sh "${args[@]}"
                    # ./scripts/classification/Sptt/SpttSRU.sh "${args[@]}"
                    # ./scripts/classification/Sptt/SpttSRU_T.sh "${args[@]}"

                    # ./scripts/classification/Bptt/BpttMinGRU.sh "${args[@]}"
                    # ./scripts/classification/Sptt/SpttMinGRU.sh "${args[@]}"
                    # ./scripts/classification/Sptt/SpttMinGRU_T.sh "${args[@]}"

                    # ./scripts/classification/Sptt/SpttGRU.sh "${args[@]}"
                    # ./scripts/classification/Bptt/BpttGRU.sh "${args[@]}"
                    # ./scripts/classification/Sptt/SpttGRU_T.sh "${args[@]}"
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
echo "All jobs finished"