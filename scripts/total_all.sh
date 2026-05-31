#!/usr/bin/bash
export CUDA_VISIBLE_DEVICES=0
seeds=(2020 2021 2022)
# task_name=("long_term_forecast" "classification" "anomaly_detection" "imputation")
# task_name=("long_term_forecast")
task_name=("imputation" "anomaly_detection")
# data_names=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "traffic" "electricity" "exchange_rate" "weather")
# data_names=("illness")
time_block_nums=(8)
# learning_rate=0.0001 # 只针对illness数据集
# learning_rate=0.001
inverse_model=align
for task in "${task_name[@]}"; do
    if [[ "$task" = "long_term_forecast" ]]; then
        echo "当前任务: $task"

        data_names=("ETTm2" "traffic")
        kranks=(7)
        learning_rate=0.001

        for data in "${data_names[@]}"; do
            for seed in "${seeds[@]}"; do
                for time_block_num in "${time_block_nums[@]}"; do
                    for krank in "${kranks[@]}"; do
                        job_name="long_term_forecast-Dataset--${data}--Seed--${seed}--Slide_Num--${time_block_num}--Krank--${krank}--LR--${learning_rate}--Mode--${inverse_model}"

                        echo "--------------------------------------------------------------"
                        echo "Running job: ${job_name}"
                        echo "Learning rate: ${learning_rate}"
                        echo "Slide window nums: ${time_block_num}"
                        echo "Krank: ${krank}"
                        echo "Seed: ${seed}"
                        echo "--------------------------------------------------------------"
                        
                        args=(
                            "$data"
                            "$seed"
                            "$time_block_num"
                            "$krank"
                            "$learning_rate"
                            "$inverse_model"
                        )
                        
                        echo "执行模式: $inverse_model"
                        # ./scripts/long_term_forecast/Sptt/SpttGRU.sh "${args[@]}"
                        # ./scripts/long_term_forecast/Bptt/BpttGRU.sh "${args[@]}"

                        # ./scripts/long_term_forecast/Sptt/SpttLSTM.sh "${args[@]}"
                        # ./scripts/long_term_forecast/Bptt/BpttLSTM.sh "${args[@]}"
                        
                        ./scripts/long_term_forecast/Bptt/BpttSRU.sh "${args[@]}"
                        ./scripts/long_term_forecast/Sptt/SpttSRU.sh "${args[@]}"

                        ./scripts/long_term_forecast/Bptt/BpttMinLSTM.sh "${args[@]}"
                        ./scripts/long_term_forecast/Sptt/SpttMinLSTM.sh "${args[@]}"

                        ./scripts/long_term_forecast/Bptt/BpttMinGRU.sh "${args[@]}"
                        ./scripts/long_term_forecast/Sptt/SpttMinGRU.sh "${args[@]}"
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

    elif [[ "$task" = "classification" ]]; then
        echo "当前任务: $task"
        data_names=("SelfRegulationSCP1" "SelfRegulationSCP2" "FaceDetection" "Handwriting" "Heartbeat")
        kranks=(6)
        learning_rate=0.001

        for seed in "${seeds[@]}"; do
            for data in "${data_names[@]}"; do
                for time_block_num in "${time_block_nums[@]}"; do
                    for krank in "${kranks[@]}"; do
                        job_name="classification-Dataset--${data}--Seed--${seed}--Slide_Num--${time_block_num}--Krank--${krank}--LR--${learning_rate}--Mode--${inverse_model}"

                        echo "--------------------------------------------------------------"
                        echo "Running job: ${job_name}"
                        echo "Learning rate: ${learning_rate}"
                        echo "Slide window nums: ${time_block_num}"
                        echo "Krank: ${krank}"
                        echo "Seed: ${seed}"
                        echo "--------------------------------------------------------------"

                        args=(
                            "$data"
                            "$seed"
                            "$time_block_num"
                            "$krank"
                            "$learning_rate"
                            "$inverse_model"
                        )
                    
                        echo "执行模式: $inverse_model"
                        # ./scripts/classification/Bptt/BpttLSTM.sh "${args[@]}"
                        # ./scripts/classification/Sptt/SpttLSTM.sh "${args[@]}"

                        ./scripts/classification/Bptt/BpttSRU.sh "${args[@]}"
                        ./scripts/classification/Sptt/SpttSRU.sh "${args[@]}"

                        ./scripts/classification/Bptt/BpttMinGRU.sh "${args[@]}"
                        ./scripts/classification/Sptt/SpttMinGRU.sh "${args[@]}"

                        ./scripts/classification/Bptt/BpttMinLSTM.sh "${args[@]}"
                        ./scripts/classification/Sptt/SpttMinLSTM.sh "${args[@]}"

                        # ./scripts/classification/Sptt/SpttGRU.sh "${args[@]}"
                        # ./scripts/classification/Bptt/BpttGRU.sh "${args[@]}"
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
    elif [[ "$task" = "imputation" ]]; then
        echo "当前任务: $task"
        data_names=("ETTh2" "weather")
        kranks=(7)
        learning_rate=0.001

        for seed in "${seeds[@]}"; do
            for data in "${data_names[@]}"; do
                for time_block_num in "${time_block_nums[@]}"; do
                    for krank in "${kranks[@]}"; do
                        job_name="imputation-Dataset--${data}--Seed--${seed}--Slide_Num--${time_block_num}--Krank--${krank}--LR--${learning_rate}--Mode--${inverse_model}"

                        echo "--------------------------------------------------------------"
                        echo "Running job: ${job_name}"
                        echo "Learning rate: ${learning_rate}"
                        echo "Slide window nums: ${time_block_num}"
                        echo "Krank: ${krank}"
                        echo "Seed: ${seed}"
                        echo "--------------------------------------------------------------"

                        args=(
                            "$data"
                            "$seed"
                            "$time_block_num"
                            "$krank"
                            "$learning_rate"
                            "$inverse_model"
                        )
                    
                        echo "执行模式: $inverse_model"
                        # ./scripts/classification/Bptt/BpttLSTM.sh "${args[@]}"
                        # ./scripts/classification/Sptt/SpttLSTM.sh "${args[@]}"

                        ./scripts/imputation/Bptt/BpttSRU.sh "${args[@]}"
                        ./scripts/imputation/Sptt/SpttSRU.sh "${args[@]}"

                        ./scripts/imputation/Bptt/BpttMinGRU.sh "${args[@]}"
                        ./scripts/imputation/Sptt/SpttMinGRU.sh "${args[@]}"

                        ./scripts/imputation/Bptt/BpttMinLSTM.sh "${args[@]}"
                        ./scripts/imputation/Sptt/SpttMinLSTM.sh "${args[@]}"

                        # ./scripts/imputation/Sptt/SpttGRU.sh "${args[@]}"
                        # ./scripts/imputation/Bptt/BpttGRU.sh "${args[@]}"
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

    elif [[ "$task" = "anomaly_detection" ]]; then
        echo "当前任务: $task"
        data_names=("PSM" "SMAP")
        kranks=(10)
        learning_rate=0.0001

        for seed in "${seeds[@]}"; do
            for data in "${data_names[@]}"; do
                for time_block_num in "${time_block_nums[@]}"; do
                    for krank in "${kranks[@]}"; do
                        job_name="anomaly_detection-Dataset--${data}--Seed--${seed}--Slide_Num--${time_block_num}--Krank--${krank}--LR--${learning_rate}--Mode--${inverse_model}"

                        echo "--------------------------------------------------------------"
                        echo "Running job: ${job_name}"
                        echo "Learning rate: ${learning_rate}"
                        echo "Slide window nums: ${time_block_num}"
                        echo "Krank: ${krank}"
                        echo "Seed: ${seed}"
                        echo "--------------------------------------------------------------"

                        args=(
                            "$data"
                            "$seed"
                            "$time_block_num"
                            "$krank"
                            "$learning_rate"
                            "$inverse_model"
                        )
                    
                        echo "执行模式: $inverse_model"
                        # ./scripts/anomaly_detection/Bptt/BpttLSTM.sh "${args[@]}"
                        # ./scripts/anomaly_detection/Sptt/SpttLSTM.sh "${args[@]}"

                        ./scripts/anomaly_detection/Bptt/BpttSRU.sh "${args[@]}"
                        ./scripts/anomaly_detection/Sptt/SpttSRU.sh "${args[@]}"

                        ./scripts/anomaly_detection/Bptt/BpttMinGRU.sh "${args[@]}"
                        ./scripts/anomaly_detection/Sptt/SpttMinGRU.sh "${args[@]}"

                        ./scripts/anomaly_detection/Bptt/BpttMinLSTM.sh "${args[@]}"
                        ./scripts/anomaly_detection/Sptt/SpttMinLSTM.sh "${args[@]}"

                        # ./scripts/anomaly_detection/Sptt/SpttGRU.sh "${args[@]}"
                        # ./scripts/anomaly_detection/Bptt/BpttGRU.sh "${args[@]}"
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

    else
        echo "未知任务: $task"
    fi
done